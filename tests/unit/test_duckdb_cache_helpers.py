from typing import Annotated

import pandas as pd
import pandera.pandas as pa
import pandera.typing as pt
import pytest

from duckdb_cache_helpers import schema_model_from_generator, to_storage_frame


class CacheSchema(pa.DataFrameModel):
    class Config:
        coerce = True

    timeframe: pt.Index[str]
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    value: pt.Series[float]


@pytest.mark.unit
def test_to_storage_frame_flattens_indexed_frame():
    indexed = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [("1min", pd.Timestamp("2024-01-01", tz="UTC")), ("1min", pd.Timestamp("2024-01-01 00:01", tz="UTC"))],
            names=["timeframe", "date"],
        ),
    )

    stored = to_storage_frame(indexed, ("1min",))

    assert list(stored.columns) == ["timeframe", "date", "value"]
    assert stored["value"].tolist() == [1.0, 2.0]


@pytest.mark.unit
def test_schema_model_from_generator_extracts_dataframe_model():
    def generate(*, time_range_str: str, freqs: tuple[str, ...]) -> pt.DataFrame[CacheSchema]:
        raise NotImplementedError

    assert schema_model_from_generator(generate) is CacheSchema
