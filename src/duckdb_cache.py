import functools
import inspect
from collections.abc import Callable
from typing import Literal

import pandas as pd
import pandera.pandas as pa
from br_py_log_n_profile import log_d, log_exception

from config import app_config
from helper.date_utils import (
    find_gaped_ranges,
    get_floor,
    normalize_timeframes,
    time_range,
    timeframe_to_period,
)
from infrastructure.datastore_engine.duckdb_cache_helpers import (
    _TIMEFRAME_COLUMN,
    _TIMESTAMP_COLUMN,
    Generator,
    PostFetch,
    dispatch_duckdb_integrity_check,
    schema_model_from_generator,
    to_storage_frame,
)
from infrastructure.datastore_engine.duckdb_cache_registry import DatastoreRegistry
from infrastructure.datastore_engine.iceberg_base import _write_gap, iceberg_fetch_from_datastore

_DropNotCacheable = Callable[[pd.DataFrame], pd.DataFrame]


def _drop_not_finished_candles(df: pd.DataFrame) -> pd.DataFrame:
    dt = df.index.get_level_values("date")
    tf: pd.Timedelta | pd.Series[pd.Timedelta] | pd.TimedeltaIndex
    if "timeframe" in df.index.names:
        tf = pd.to_timedelta(df.index.get_level_values("timeframe"))
    else:
        tf = pd.to_timedelta(app_config.timeframes[0])

    drop_mask = dt + tf >= pd.Timestamp.now(tz="UTC")
    return df.loc[~drop_mask]


def _assemble_final_result(
    cached_rows: pd.DataFrame,
    generated_frames: list[pd.DataFrame],
    overall_start: object,
    overall_end: object,
    post_fetch: PostFetch | None,
) -> pd.DataFrame:
    parts = [frame for frame in (cached_rows, *generated_frames) if not frame.empty]
    assembled = pd.concat(parts, ignore_index=True) if parts else cached_rows
    assembled[_TIMESTAMP_COLUMN] = pd.to_datetime(assembled[_TIMESTAMP_COLUMN], utc=True).astype("datetime64[ns, UTC]")
    # Trailing-edge widening can make two adjacent gap windows both regenerate the same coarse
    # candle; _write_gap upserts idempotently on disk, but the in-RAM concat would keep both.
    # generated_frames come after cached_rows, so keep="last" prefers the freshly generated row.
    assembled = assembled.drop_duplicates(subset=[_TIMEFRAME_COLUMN, _TIMESTAMP_COLUMN], keep="last")
    assembled = assembled.sort_values([_TIMEFRAME_COLUMN, _TIMESTAMP_COLUMN]).reset_index(drop=True)
    mask = (assembled[_TIMESTAMP_COLUMN] >= overall_start) & (assembled[_TIMESTAMP_COLUMN] <= overall_end)
    final_result = assembled[mask].reset_index(drop=True)

    final_result = final_result.set_index([_TIMEFRAME_COLUMN, _TIMESTAMP_COLUMN]).rename_axis(["timeframe", "date"])
    if post_fetch is not None:
        final_result = post_fetch(final_result)

    return final_result


def duckdb_cache(
    datastore_registry: DatastoreRegistry,
    boundary_arg: str = "time_range_str",
    freqs: tuple[str, ...] | None = None,
    post_fetch: PostFetch | None = None,
    nan_means: Literal["not-cached", "not-available"] = "not-cached",
    drop_not_cacheable: _DropNotCacheable = _drop_not_finished_candles,
    # allow_one_second_freq: bool = False,
) -> Callable[[Generator], Generator]:
    # Every @duckdb_cache dataset carries a 'timeframe' index level -- a nominally single-timeframe
    # artifact passes its native cadence (e.g. freqs=("1min",)). There is no no-timeframe path.
    # (app_config.default_cache_window_freq / cache_window_freq_overrides are now read only by the
    # legacy disk_cache path, not here -- gap detection is precise, not calendar-windowed.)

    decorated_freqs = normalize_timeframes(freqs, all_if_empty=True)  # , allow_one_second_freq=allow_one_second_freq)
    if not decorated_freqs:
        raise ValueError("@duckdb_cache: freqs must be non-empty (pass the dataset's native cadence)")

    def decorator(generator: Generator) -> Generator:
        signature = inspect.signature(generator)
        if boundary_arg not in signature.parameters:
            raise TypeError(f"@duckdb_cache: {generator.__name__!r} has no {boundary_arg!r} parameter")
        if freqs is not None and "freqs" not in signature.parameters:
            raise TypeError(
                f"@duckdb_cache: {generator.__name__!r} has no 'freqs' parameter (required when freqs= is passed)"
            )
        schema_model = schema_model_from_generator(generator)
        # A cached generator is itself a Pandera validation boundary. This validates
        # DataFrame arguments and every freshly generated frame; the explicit schema
        # validation below also covers frames restored entirely from the cache. Do not
        # use ``pandera_validate`` here: it intentionally bypasses checks in production.
        validated_generator = pa.check_types(lazy=True)(generator)

        @functools.wraps(generator)
        def _wrapper(*args: object, **kwargs: object) -> pd.DataFrame:
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            requested_boundary = bound.arguments[boundary_arg]
            if not isinstance(requested_boundary, str):
                raise TypeError(
                    f"@duckdb_cache: {boundary_arg!r} must be a time_range_str, got {type(requested_boundary)!r}"
                )
            requested_runtime_freqs = bound.arguments["freqs"]
            runtime_freqs = set(requested_runtime_freqs) if requested_runtime_freqs is not None else set()

            unexpected_runtime_freqs = set(runtime_freqs) - set(decorated_freqs)
            if unexpected_runtime_freqs:
                log_exception(f"Runtime-freqs{runtime_freqs} are not in decorated-freqs:{decorated_freqs}", ValueError)

            effective_freqs = normalize_timeframes(
                runtime_freqs if runtime_freqs else decorated_freqs,
                allowed_timeframes=decorated_freqs,  # allow_one_second_freq=allow_one_second_freq
            )

            cached_rows = iceberg_fetch_from_datastore(
                datastore_registry,
                requested_boundary,
                effective_freqs,
                schema_model,
            )
            gaps = find_gaped_ranges(cached_rows, requested_boundary, effective_freqs, nan_means)

            call_info = f"{boundary_arg}={requested_boundary!r}" if boundary_arg in kwargs else repr(requested_boundary)
            prefix = f"{generator.__name__}({call_info})"
            if not gaps:
                log_d(f"{prefix} already up to date over {requested_boundary}")
            else:
                log_d(f"{prefix} gapped at {gaps}")

            generated_frames: list[pd.DataFrame] = []
            for gap in gaps:
                log_d(f"{prefix} gap fetching at {gap}...")
                expanded_gap = expand_range_to_cover_end_of_included_candles(gap, effective_freqs)

                frame = _fetch_one_window(
                    validated_generator,
                    signature,
                    args,
                    kwargs,
                    boundary_arg,
                    expanded_gap,
                    # schema_model,
                    effective_freqs,
                    datastore_registry,
                    drop_not_cacheable=drop_not_cacheable,
                    # allow_one_second_freq=allow_one_second_freq,
                )
                log_d(f"{prefix} gap fetched at {gap}")
                generated_frames.append(frame)

            overall_start, overall_end = time_range(requested_boundary)
            final_result = _assemble_final_result(
                to_storage_frame(cached_rows, effective_freqs),
                generated_frames,
                overall_start,
                overall_end,
                post_fetch,
            )

            if app_config.environment == "development":
                dispatch_duckdb_integrity_check(
                    datastore_registry,
                    requested_boundary,
                    final_result.copy(),
                    schema_model,
                    effective_freqs,
                )

            return schema_model.validate(final_result, lazy=True)

        return _wrapper

    return decorator


def expand_range_to_cover_end_of_included_candles(time_range_st: str, effective_freqs: tuple[str, ...]) -> str:
    """Widen a precise gap window so the generator fetches enough raw data to compute every
    candle *whose label falls inside the window* in full.

    Every candle is labelled by its open time (exchange convention), so a `tf` candle labelled
    `L` owns base data over `[L, L + timeframe_to_period(tf))`. Walking the freqs coarsest-first,
    each step takes the last `tf` candle whose open-label lies inside `[start, widened_end]` and
    pushes `widened_end` out to that candle's close: a coarser tf whose open-label lies before
    `start` (its candle is already cached, or it has none in range at all) must not drag the end
    forward, while a coarser widening can pull a finer tf's last in-window candle further out.
    """
    start_dt, end_dt = time_range(time_range_st)

    widened_end = end_dt
    for tf in sorted(effective_freqs, key=timeframe_to_period, reverse=True):
        floored_end = get_floor(widened_end, tf)
        if floored_end >= start_dt:
            candle_close = floored_end + timeframe_to_period(tf) - pd.Timedelta(nanoseconds=1)
            widened_end = max(widened_end, candle_close)

    return f"{start_dt.strftime('%y-%m-%d.%H-%M')}T{widened_end.strftime('%y-%m-%d.%H-%M')}"


def _fetch_one_window(
    generator: Generator,
    signature: inspect.Signature,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    boundary_arg: str,
    window: str,
    # schema_model: type[pa.DataFrameModel],
    effective_freqs: tuple[str, ...],
    datastore_registry: DatastoreRegistry,
    drop_not_cacheable: _DropNotCacheable,
    # allow_one_second_freq: bool,
) -> pd.DataFrame:
    # Bind the original call so a boundary_arg/timeframe passed positionally is overridden
    # in place rather than colliding as a duplicate keyword ("got multiple values for ...").
    effective_freqs = normalize_timeframes(
        effective_freqs,
        # allow_one_second_freq=allow_one_second_freq
    )
    bound = signature.bind_partial(*args, **kwargs)
    bound.arguments[boundary_arg] = window
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if "timeframe" in signature.parameters or accepts_var_kw:
        bound.arguments["timeframe"] = None
    result = generator(*bound.args, **bound.kwargs)
    cacheable_result = drop_not_cacheable(result)
    storage_frame = to_storage_frame(cacheable_result, effective_freqs)
    _write_gap(datastore_registry, storage_frame)
    return storage_frame
