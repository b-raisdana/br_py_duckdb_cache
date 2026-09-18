import pandas as pd
import pytest

from duckdb_retention import remove_overlapping_ranges


@pytest.mark.unit
def test_remove_overlapping_ranges_keeps_coarse_and_uncovered_fine_rows():
    indexes = pd.MultiIndex.from_tuples(
        [
            ("1h", pd.Timestamp("2024-01-01 00:00:00+00:00")),
            ("1min", pd.Timestamp("2024-01-01 00:00:00+00:00")),
            ("1min", pd.Timestamp("2024-01-01 00:30:00+00:00")),
            ("1min", pd.Timestamp("2024-01-01 01:00:00+00:00")),
        ],
        names=["timeframe", "date"],
    )

    result = remove_overlapping_ranges(indexes)

    assert result.tolist() == [
        ("1h", pd.Timestamp("2024-01-01 00:00:00+00:00")),
        ("1min", pd.Timestamp("2024-01-01 01:00:00+00:00")),
    ]
