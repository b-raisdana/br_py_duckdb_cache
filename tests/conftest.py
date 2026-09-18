from __future__ import annotations

import re
import sys
from datetime import UTC, datetime
from enum import Enum
from types import ModuleType

import pandas as pd


def _ensure_module(name: str) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


enum_utils = _ensure_module("helper.enum_utils")


class AutoSnakeEnum(Enum):
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


enum_utils.AutoSnakeEnum = AutoSnakeEnum

date_utils = _ensure_module("helper.date_utils")


def timeframe_to_period(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(days=7) if timeframe == "1W" else pd.Timedelta(timeframe)


def time_range(value: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_text, end_text = value.split("T")
    start = datetime.strptime(start_text, "%y-%m-%d.%H-%M").replace(tzinfo=UTC)
    end = datetime.strptime(end_text, "%y-%m-%d.%H-%M").replace(tzinfo=UTC)
    return pd.Timestamp(start), pd.Timestamp(end)


def get_floor(value: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    if timeframe == "1W":
        return value.normalize() - pd.Timedelta(days=value.weekday())
    return value.floor(timeframe)


def normalize_timeframes(timeframes: tuple[str, ...] | list[str] | set[str] | None, **_: object) -> tuple[str, ...]:
    values = tuple(timeframes or ())
    return tuple(sorted(values, key=timeframe_to_period, reverse=True))


def find_gaped_ranges(*_: object, **__: object) -> list[str]:
    return []


date_utils.find_gaped_ranges = find_gaped_ranges
date_utils.get_floor = get_floor
date_utils.normalize_timeframes = normalize_timeframes
date_utils.time_range = time_range
date_utils.timeframe_to_period = timeframe_to_period

iceberg_base = _ensure_module("iceberg_base")
iceberg_base.write_calls = []
iceberg_base.fetch_result = pd.DataFrame()


def _write_gap(datastore_registry: object, frame: pd.DataFrame) -> None:
    iceberg_base.write_calls.append((datastore_registry, frame.copy()))


def iceberg_fetch_from_datastore(*args: object, **kwargs: object) -> pd.DataFrame:
    return iceberg_base.fetch_result


iceberg_base._write_gap = _write_gap
iceberg_base.iceberg_fetch_from_datastore = iceberg_fetch_from_datastore

infrastructure = _ensure_module("infrastructure")
datastore_engine = _ensure_module("infrastructure.datastore_engine")

import duckdb_cache_helpers as local_cache_helpers  # noqa: E402
import duckdb_cache_registry as local_cache_registry  # noqa: E402

datastore_engine.duckdb_cache_helpers = local_cache_helpers
datastore_engine.duckdb_cache_registry = local_cache_registry
sys.modules["infrastructure.datastore_engine.duckdb_cache_helpers"] = local_cache_helpers
sys.modules["infrastructure.datastore_engine.duckdb_cache_registry"] = local_cache_registry
sys.modules["infrastructure.datastore_engine.iceberg_base"] = iceberg_base

dataframe_indexing = _ensure_module("infrastructure.datastore_engine.dataframe_indexing")


def index_by_date(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True)
    return result.set_index("date")


def add_timeframe_index(frame: pd.DataFrame, data_frame_type: str) -> pd.DataFrame:
    result = frame.copy()
    if "multi_timeframe" in data_frame_type:
        result = result.set_index("timeframe", append=True)
        return result.swaplevel()
    return result


dataframe_indexing.add_timeframe_index = add_timeframe_index
dataframe_indexing.index_by_date = index_by_date
