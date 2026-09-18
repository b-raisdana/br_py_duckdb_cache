import pandas as pd
import pytest
from domain.datastore_engine.parquet_normalization import flatten_index_to_columns, has_non_default_index


@pytest.mark.unit
def test_has_non_default_index_false_for_flat_frame():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, tz="UTC"), "value": [1, 2, 3]})

    assert has_non_default_index(df) is False


@pytest.mark.unit
def test_has_non_default_index_true_for_date_indexed_frame():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, tz="UTC"), "value": [1, 2, 3]}).set_index("date")

    assert has_non_default_index(df) is True


@pytest.mark.unit
def test_flatten_index_to_columns_is_noop_for_already_flat_frame():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, tz="UTC"), "value": [1, 2, 3]})

    flattened = flatten_index_to_columns(df)

    pd.testing.assert_frame_equal(flattened, df)


@pytest.mark.unit
def test_flatten_index_to_columns_restores_date_index_as_column():
    indexed = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, tz="UTC"), "value": [1, 2, 3]}).set_index(
        "date"
    )

    flattened = flatten_index_to_columns(indexed)

    assert has_non_default_index(flattened) is False
    assert list(flattened.columns) == ["date", "value"]
    assert flattened["date"].tolist() == list(pd.date_range("2024-01-01", periods=3, tz="UTC"))


@pytest.mark.unit
def test_flatten_index_to_columns_restores_multi_timeframe_index_as_columns():
    indexed = pd.DataFrame(
        {
            "timeframe": ["1D", "1D"],
            "date": pd.date_range("2024-01-01", periods=2, tz="UTC"),
            "value": [1, 2],
        }
    ).set_index(["timeframe", "date"])

    flattened = flatten_index_to_columns(indexed)

    assert has_non_default_index(flattened) is False
    assert set(flattened.columns) == {"timeframe", "date", "value"}
