import pytest

from config import app_config
from duckdb_cache_registry import DatastoreRegistry


@pytest.mark.unit
def test_registry_exposes_snake_case_values_and_configured_namespace(monkeypatch):
    monkeypatch.setattr(app_config, "under_process_market", "crypto")
    monkeypatch.setattr(app_config, "under_process_symbol", "BTC")
    monkeypatch.setattr(app_config, "under_process_exchange", "binance")

    assert DatastoreRegistry.UnifiedNoNAN.value == "unified_no_n_a_n"
    assert DatastoreRegistry.Tick.value == "tick"
    assert DatastoreRegistry.UnifiedNoNAN.get_auto_table_name() == "crypto.BTC.binance"
    assert DatastoreRegistry.UnifiedNoNAN._table_identifier("ohlcv") == "crypto.BTC.binance.ohlcv"
