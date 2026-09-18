from datetime import UTC, datetime

import pandas as pd
import pytest

from duckdb_reader import read_duckdb


@pytest.mark.unit
def test_read_duckdb_filters_and_indexes_parquet_files(tmp_path):
    first = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01 00:10:00+00:00", "2024-01-01 00:30:00+00:00"]),
            "timeframe": ["1min", "1min"],
            "value": [1.0, 2.0],
        }
    )
    second = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01 01:00:00+00:00"]),
            "timeframe": ["5min"],
            "value": [3.0],
        }
    )
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    first.to_parquet(first_path, index=False)
    second.to_parquet(second_path, index=False)

    result = read_duckdb(
        [first_path, second_path],
        "multi_timeframe_market",
        datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
        datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
    )

    assert result.index.names == ["timeframe", "date"]
    assert result.index.get_level_values("timeframe").tolist() == ["1min", "5min"]
    assert result.index.get_level_values("date").tolist() == [
        pd.Timestamp("2024-01-01 00:30:00+00:00"),
        pd.Timestamp("2024-01-01 01:00:00+00:00"),
    ]
    assert result["value"].tolist() == [2.0, 3.0]
