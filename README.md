# br-py-duckdb-cache

DuckDB-backed windowed cache with Parquet/ZSTD storage, gap-aware regeneration, and Iceberg datastore integration.

## Installation

### From PyPI

```bash
pip install br-py-duckdb-cache
```

### From source

```bash
git clone https://github.com/b-raisdana/br_py_duckdb_cache.git
cd br_py_duckdb_cache
pip install .
```

### Development install

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode along with all development dependencies (pytest, ruff, mypy, pre-commit, etc.).

## Requirements

- Python >= 3.12
- See `requirements.txt` for runtime dependencies

## Usage

### Core API

#### `duckdb_cache` — caching decorator

The main entry point. Decorate a data generator function to add automatic DuckDB-backed caching with gap detection and on-demand regeneration.

```python
from duckdb_cache import duckdb_cache
from duckdb_cache_registry import DatastoreRegistry

registry = DatastoreRegistry()

@duckdb_cache(
    datastore_registry=registry,
    boundary_arg="time_range_str",
    freqs=("1min", "5min"),
)
def my_generator(*, time_range_str: str, freqs: tuple[str, ...]) -> pd.DataFrame:
    """Your data generation logic here."""
    ...

result = my_generator(time_range_str="20-01-01.00-00T20-01-01.01-00")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `datastore_registry` | `DatastoreRegistry` | required | Registry for Iceberg/Parquet storage backends |
| `boundary_arg` | `str` | `"time_range_str"` | Name of the time boundary argument |
| `freqs` | `tuple[str, ...]` | `None` | Timeframe frequencies for this dataset |
| `post_fetch` | `Callable` | `None` | Post-processing hook applied to the final result |
| `nan_means` | `Literal` | `"not-cached"` | How to treat NaN-mean gaps |
| `drop_not_cacheable` | `Callable` | `_drop_not_finished_candles` | Filter to drop non-cacheable rows |

#### `read_duckdb` — batched Parquet reads

Read and concatenate multiple Parquet/ZSTD cache files in a single DuckDB query with date-column filtering.

```python
from duckdb_reader import read_duckdb
from pathlib import Path
from datetime import datetime

paths = [Path("data/file1.parquet"), Path("data/file2.parquet")]
df = read_duckdb(
    paths=paths,
    data_frame_type="market_data",
    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
    end=datetime(2024, 1, 2, tzinfo=timezone.utc),
)
```

#### Parquet normalization

Utilities to ensure Parquet files conform to the flat-frame on-disk convention (no `date`/`timeframe` index levels).

```python
from parquet_normalization import flatten_index_to_columns, has_non_default_index

if has_non_default_index(df):
    df = flatten_index_to_columns(df)
```

### Registry

`DatastoreRegistry` defines named storage namespaces derived from configuration (market, symbol, exchange):

```python
from duckdb_cache_registry import DatastoreRegistry

registry = DatastoreRegistry.UnifiedNoNAN  # or DatastoreRegistry.Tick
table_name = registry.get_auto_table_name()
```

### Configuration

Runtime settings are managed via `app_config` (Pydantic BaseSettings). There are three ways to override defaults:

#### 1. Programmatic override (must be called before first `app_config` access)

```python
from config import configure, app_config

configure(
    timeframes=("1min", "5min", "1h"),
    under_process_market="crypto",
    under_process_symbol="BTC",
    under_process_exchange="binance",
)

print(app_config.timeframes)              # ('1min', '5min', '1h')
print(app_config.under_process_market)    # 'crypto'
```

#### 2. Environment variables (`DLF_` prefix)

```bash
export DLF_TIMEFRAMES="1min,5min,1h"
export DLF_UNDER_PROCESS_MARKET="crypto"
export DLF_UNDER_PROCESS_SYMBOL="BTC"
export DLF_UNDER_PROCESS_EXCHANGE="binance"
```

```python
from config import app_config

print(app_config.timeframes)              # ('1min', '5min', '1h')
```

#### 3. `.env` file

Create a `.env` file in your project root:

```env
DLF_TIMEFRAMES=1min,5min,1h
DLF_UNDER_PROCESS_MARKET=crypto
DLF_UNDER_PROCESS_SYMBOL=BTC
DLF_UNDER_PROCESS_EXCHANGE=binance
```

**Available settings:**

| Field | Type | Default | Env var | Description |
|-------|------|---------|---------|-------------|
| `timeframes` | `tuple[str, ...]` | `("1min", "5min", "15min", "1h", "4h", "1D", "1W")` | `DLF_TIMEFRAMES` | Comma-separated timeframe strings |
| `under_process_market` | `str` | `""` | `DLF_UNDER_PROCESS_MARKET` | Market identifier (e.g., "crypto", "forex") |
| `under_process_symbol` | `str` | `""` | `DLF_UNDER_PROCESS_SYMBOL` | Symbol/ticker (e.g., "BTC", "EURUSD") |
| `under_process_exchange` | `str` | `""` | `DLF_UNDER_PROCESS_EXCHANGE` | Exchange identifier (e.g., "binance", "oanda") |

**Note:** The `configure()` function must be called **before** the first access to `app_config`. Once `app_config` is accessed, the configuration is frozen and further calls to `configure()` will raise a `RuntimeError`.

## Project Structure

```
src/
├── config/               # Runtime configuration (Pydantic)
│   └── Config.py
├── helper/               # Shared helpers
│   └── pandera.py        # Pandera validation decorator
├── duckdb_cache.py       # Main caching decorator
├── duckdb_cache_helpers.py  # Schema/model helper utilities
├── duckdb_cache_registry.py # Datastore namespace registry
├── duckdb_reader.py      # Batched DuckDB Parquet reads
├── duckdb_retention.py   # Overlapping range removal
└── parquet_normalization.py # DataFrame index flattening
```

## Development

### Running tests

```bash
pytest -m unit
```

### Linting & type checking

```bash
ruff check src/ tests/
mypy src/
```

### Pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

## Publishing

See [br_pypi_publisher](br_pypi_publisher/README.md) for the release workflow.

## License

MIT
