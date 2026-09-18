import os
import threading
from collections.abc import Callable
from typing import get_args, get_type_hints

import pandas as pd
import pandera.pandas as pa
from br_py_log_n_profile import log_d, log_e, log_i
from iceberg_base import iceberg_fetch_from_datastore

from duckdb_cache_registry import DatastoreRegistry
from helper.pandera import pandera_validate

BASE_TIMEFRAME = "1m"
Generator = Callable[..., pd.DataFrame]  # type: ignore[explicit-any]

PostFetch = Callable[[pd.DataFrame], pd.DataFrame]

_TIMESTAMP_COLUMN = "date"
_TIMEFRAME_COLUMN = "timeframe"

_abort: Callable[[int], None] = os._exit


@pandera_validate(allow_pandas_dataframe=True)
def to_storage_frame(df: pd.DataFrame, freqs: tuple[str, ...]) -> pd.DataFrame:
    assert freqs, "@duckdb_cache: freqs must be non-empty"
    flat = df.reset_index()
    if _TIMESTAMP_COLUMN not in flat.columns:
        raise TypeError("@duckdb_cache: generator's frame has no 'date' index or column")
    if _TIMEFRAME_COLUMN not in flat.columns:
        raise TypeError("@duckdb_cache: generator's frame has no 'timeframe' column/index level")
    return flat


def schema_model_from_generator(generator: Generator) -> type[pa.DataFrameModel]:
    hints = get_type_hints(generator)
    return_hint = hints.get("return")
    args = get_args(return_hint) if return_hint is not None else ()
    if not args or not (isinstance(args[0], type) and issubclass(args[0], pa.DataFrameModel)):
        raise TypeError(
            f"@duckdb_cache requires {generator.__name__!r} to declare a pt.DataFrame[SchemaModel] return "
            f"type annotation (design doc § Schema contract) — none found."
        )
    return args[0]


@pandera_validate(allow_pandas_dataframe=True)
def dispatch_duckdb_integrity_check(
    datastore_registry: DatastoreRegistry,
    requested: str,
    returned: pd.DataFrame,
    schema_model: type[pa.DataFrameModel],
    effective_freqs: tuple[str, ...],
) -> None:
    threading.Thread(
        target=_run_integrity_check,
        args=(datastore_registry, requested, returned, schema_model, effective_freqs),
        daemon=True,
    ).start()


def _run_integrity_check(
    datastore_registry: DatastoreRegistry,
    requested: str,
    returned: pd.DataFrame,
    schema_model: type[pa.DataFrameModel],
    effective_freqs: tuple[str, ...] = (),
) -> None:
    log_i(f"duckdb_cache: integrity check starting for {datastore_registry.get_auto_table_name()!r} {requested!r}")
    prefetched = iceberg_fetch_from_datastore(
        datastore_registry=datastore_registry,
        time_range_str=requested,
        timeframes=effective_freqs if effective_freqs is not None else (BASE_TIMEFRAME,),
        schema_model=schema_model,
    )
    if prefetched.empty:
        if not returned.empty:
            log_e(
                f"duckdb_cache: integrity check re-fetch failed for {datastore_registry.get_auto_table_name()!r} "
                f"{requested!r}: empty result"
            )
        return

    prefetched = prefetched.sort_index()
    comparable_returned = returned.sort_index()
    comparable_returned = comparable_returned.loc[comparable_returned.index.isin(prefetched.index)]
    try:
        pd.testing.assert_frame_equal(prefetched, comparable_returned[prefetched.columns], check_like=True)
    except AssertionError as err:
        log_e(
            f"duckdb_cache: INTEGRITY CHECK FAILED for "
            f"{datastore_registry.get_auto_table_name()!r} {requested!r} — in-memory result "
            f"disagrees with what's on disk: {err}"
        )
        _abort(1)
        return
    log_d(f"duckdb_cache: integrity check passed for {datastore_registry.get_auto_table_name()!r} {requested!r}")
