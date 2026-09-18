import pandas as pd

from helper.pandera import pandera_validate

"""
Pure DataFrame-shape check/fix behind infrastructure.datastore_engine.parquet_housekeeping's Parquet
conversion and repair: the on-disk convention every dataset_db cache file is written to
(disk_cache.write_data_file()'s own `df.reset_index().to_parquet(...)`) is a flat frame — `date` (and,
for multi_timeframe_* types, `timeframe`) as plain columns, default RangeIndex — never a DataFrame with
those set as the pandas index. No I/O here; the file read/write/verify lives in the infrastructure layer.
"""


@pandera_validate(allow_pandas_dataframe=True)  # type: ignore[untyped-decorator]
def has_non_default_index(df: pd.DataFrame) -> bool:
    """True when df carries a real (named/Datetime/Multi) index instead of the default RangeIndex —
    the shape that, once written straight to Parquet, resurfaces on read as a missing `date` column
    (pandas restores the persisted index from the file's own metadata) and crashes
    disk_cache_layout.index_by_date()/read_by_date() with KeyError: 'date'."""
    return not isinstance(df.index, pd.RangeIndex)


@pandera_validate(allow_pandas_dataframe=True)  # type: ignore[untyped-decorator]
def flatten_index_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Restore df to the flat, default-RangeIndex shape write_data_file() always writes, moving any
    real index level(s) (e.g. `date`, or `timeframe`+`date`) back to plain columns. A no-op (returns df
    unchanged) when the index is already the default RangeIndex."""
    return df.reset_index() if has_non_default_index(df) else df
