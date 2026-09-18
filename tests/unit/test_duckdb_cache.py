from typing import Annotated

import pandas as pd
import pandera.pandas as pa
import pandera.typing as pt
import pytest

import duckdb_cache
from duckdb_cache_registry import DatastoreRegistry


class CacheSchema(pa.DataFrameModel):
    class Config:
        coerce = True

    timeframe: pt.Index[str]
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    value: pt.Series[float]


def _indexed_frame(values: list[float], timeframe: str = "1min") -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            (timeframe, pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(offset, unit="min"))
            for offset in range(len(values))
        ],
        names=["timeframe", "date"],
    )
    return pd.DataFrame({"value": values}, index=index)


def _empty_cached_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [pd.Series(dtype="object"), pd.Series(dtype="datetime64[ns, UTC]")],
        names=["timeframe", "date"],
    )
    return pd.DataFrame({"value": pd.Series(dtype=float)}, index=index)


@pytest.mark.unit
def test_assemble_final_result_deduplicates_filters_and_applies_post_fetch():
    cached = pd.DataFrame(
        {
            "timeframe": ["1min", "1min"],
            "date": pd.to_datetime(["2024-01-01 00:00:00+00:00", "2024-01-01 00:02:00+00:00"], utc=True),
            "value": [1.0, 3.0],
        }
    )
    generated = pd.DataFrame(
        {
            "timeframe": ["1min", "1min"],
            "date": pd.to_datetime(["2024-01-01 00:01:00+00:00", "2024-01-01 00:02:00+00:00"], utc=True),
            "value": [2.0, 20.0],
        }
    )

    result = duckdb_cache._assemble_final_result(
        cached,
        [generated],
        pd.Timestamp("2024-01-01 00:00:00+00:00"),
        pd.Timestamp("2024-01-01 00:02:00+00:00"),
        lambda frame: frame.assign(value=frame["value"] + 10),
    )

    assert result.index.names == ["timeframe", "date"]
    assert result["value"].tolist() == [11.0, 12.0, 30.0]


@pytest.mark.unit
def test_expand_range_covers_the_end_of_the_coarsest_candle():
    expanded = duckdb_cache.expand_range_to_cover_end_of_included_candles(
        "24-01-01.00-00T24-01-01.00-01",
        ("1min", "5min"),
    )

    assert expanded == "24-01-01.00-00T24-01-01.00-04"


@pytest.mark.unit
def test_duckdb_cache_writes_and_returns_a_cold_cache_frame(monkeypatch):
    calls: list[str] = []
    written: list[pd.DataFrame] = []

    def fetch(*args: object, **kwargs: object) -> pd.DataFrame:
        return _empty_cached_frame()

    def gaps(*args: object, **kwargs: object) -> list[str]:
        return ["24-01-01.00-00T24-01-01.00-01"]

    def write(registry: object, frame: pd.DataFrame) -> None:
        written.append(frame)

    monkeypatch.setattr(duckdb_cache, "iceberg_fetch_from_datastore", fetch)
    monkeypatch.setattr(duckdb_cache, "find_gaped_ranges", gaps)
    monkeypatch.setattr(duckdb_cache, "_write_gap", write)
    monkeypatch.setattr(duckdb_cache, "dispatch_duckdb_integrity_check", lambda *args: None)
    monkeypatch.setattr(duckdb_cache.app_config, "environment", "production")

    @duckdb_cache.duckdb_cache(
        DatastoreRegistry.UnifiedNoNAN,
        freqs=("5min",),
        post_fetch=lambda frame: frame.assign(value=frame["value"] + 10),
    )
    def generate(*, time_range_str: str, freqs: tuple[str, ...]) -> pt.DataFrame[CacheSchema]:
        calls.append(time_range_str)
        return _indexed_frame([1.0, 2.0], "5min")

    result = generate(time_range_str="24-01-01.00-00T24-01-01.00-01", freqs=("5min",))

    assert calls == ["24-01-01.00-00T24-01-01.00-04"]
    assert len(written) == 1
    assert written[0]["timeframe"].tolist() == ["5min", "5min"]
    assert result.index.names == ["timeframe", "date"]
    assert result["value"].tolist() == [11.0, 12.0]


@pytest.mark.unit
def test_duckdb_cache_returns_a_warm_cache_without_calling_generator(monkeypatch):
    cached = _indexed_frame([1.0, 2.0])
    calls: list[str] = []

    def fetch(*args: object, **kwargs: object) -> pd.DataFrame:
        return cached

    def gaps(*args: object, **kwargs: object) -> list[str]:
        return []

    def generate(*, time_range_str: str, freqs: tuple[str, ...]) -> pt.DataFrame[CacheSchema]:
        calls.append(time_range_str)
        return cached

    monkeypatch.setattr(duckdb_cache, "iceberg_fetch_from_datastore", fetch)
    monkeypatch.setattr(duckdb_cache, "find_gaped_ranges", gaps)
    monkeypatch.setattr(duckdb_cache, "dispatch_duckdb_integrity_check", lambda *args: None)
    monkeypatch.setattr(duckdb_cache.app_config, "environment", "production")

    cached_generator = duckdb_cache.duckdb_cache(DatastoreRegistry.UnifiedNoNAN, freqs=("1min",))(generate)

    result = cached_generator(time_range_str="24-01-01.00-00T24-01-01.00-01", freqs=("1min",))

    assert calls == []
    assert result.index.names == ["timeframe", "date"]
    assert result["value"].tolist() == [1.0, 2.0]
