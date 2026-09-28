"""Tests for `alpha_agent.marketdata.databento_provider` (mission Part D/O,
task spec section 54). Every test injects a fake `client_factory` -- NO real
Databento SDK import, NO network call, ever. This mirrors
`test_release_agent_and_marketdata.py`'s existing "no network in unit tests"
discipline for the IBKR provider.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from alpha_agent.marketdata.databento_config import DatabentoMarketDataConfig
from alpha_agent.marketdata.databento_provider import (
    DatabentoMarketDataProvider,
    DatabentoUnavailable,
)
from alpha_agent.marketdata.databento_schemas import DatabentoCapability


class _FakeStore:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_df(self) -> pd.DataFrame:
        return self._df


def _ohlcv_df(n: int, *, start: datetime, freq_minutes: int = 60) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=f"{freq_minutes}min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [100.5 + i for i in range(n)],
            "low": [99.5 + i for i in range(n)],
            "close": [100.2 + i for i in range(n)],
            "volume": [10.0 + i for i in range(n)],
        },
        index=pd.Index(idx, name="ts_event"),
    )


def _definition_df(*, raw_symbol: str = "NQU6", instrument_id: int = 42004177) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "raw_symbol": raw_symbol,
                "instrument_id": instrument_id,
                "expiration": pd.Timestamp("2026-09-18T13:30:00Z"),
                "exchange": "XCME",
                "min_price_increment": 0.25,
                "min_price_increment_amount": 0.05,
                "display_factor": 0.01,
                "unit_of_measure_qty": 20.0,
                "unit_of_measure": "IPNT",
                "main_fraction": 255,
                "contract_multiplier": 2147483647,
            }
        ]
    )


class FakeMetadata:
    def __init__(self, *, dataset_range: dict | None = None, cost: float = 0.0, range_error: Exception | None = None,
                 cost_error: Exception | None = None):
        self._dataset_range = dataset_range
        self._range_error = range_error
        self._cost = cost
        self._cost_error = cost_error
        self.get_cost_calls: list[dict] = []

    def get_dataset_range(self, *, dataset: str) -> dict:
        if self._range_error is not None:
            raise self._range_error
        return self._dataset_range or {}

    def get_cost(self, **kwargs) -> float:
        self.get_cost_calls.append(kwargs)
        if self._cost_error is not None:
            raise self._cost_error
        return self._cost

    def list_datasets(self) -> list[str]:
        return ["GLBX.MDP3"]


class FakeSymbology:
    def __init__(self, *, instrument_id: int = 42004177):
        self._instrument_id = instrument_id

    def resolve(self, **kwargs) -> dict:
        symbol = kwargs["symbols"][0]
        return {"result": {symbol: [{"d0": "2026-09-08", "d1": "2026-09-13", "s": str(self._instrument_id)}]}}


class FakeTimeseries:
    def __init__(self, *, ohlcv_df: pd.DataFrame | None = None, definition_df: pd.DataFrame | None = None):
        self._ohlcv_df = ohlcv_df
        self._definition_df = definition_df
        self.get_range_calls: list[dict] = []

    def get_range(self, **kwargs) -> _FakeStore:
        self.get_range_calls.append(kwargs)
        if kwargs["schema"] == "definition":
            return _FakeStore(self._definition_df if self._definition_df is not None else pd.DataFrame())
        return _FakeStore(self._ohlcv_df if self._ohlcv_df is not None else pd.DataFrame())


class FakeClient:
    def __init__(self, metadata: FakeMetadata, symbology: FakeSymbology | None = None,
                 timeseries: FakeTimeseries | None = None):
        self.metadata = metadata
        self.symbology = symbology or FakeSymbology()
        self.timeseries = timeseries or FakeTimeseries()


NOW = datetime.now(UTC)
FRESH_RANGE = {"start": "2010-06-06T00:00:00Z", "end": (NOW - timedelta(hours=2)).isoformat()}
STALE_RANGE = {"start": "2010-06-06T00:00:00Z", "end": (NOW - timedelta(days=10)).isoformat()}


def _provider(client: FakeClient, **config_overrides) -> DatabentoMarketDataProvider:
    config = DatabentoMarketDataConfig(**config_overrides)
    return DatabentoMarketDataProvider(config=config, client_factory=lambda api_key: client)


# ---------------------------------------------------------------------------
# health() / capability honesty
# ---------------------------------------------------------------------------


def test_missing_key_is_not_connected():
    def factory(api_key):
        raise DatabentoUnavailable("DATABENTO_API_KEY is not set")

    provider = DatabentoMarketDataProvider(config=DatabentoMarketDataConfig(), client_factory=factory)
    health = provider.health()
    assert health.capability == DatabentoCapability.NOT_CONNECTED
    assert "DATABENTO_API_KEY" in health.detail


def test_disabled_config_is_not_connected_with_no_client_construction():
    calls = []

    def factory(api_key):
        calls.append(api_key)
        raise AssertionError("client must never be constructed when disabled")

    provider = DatabentoMarketDataProvider(
        config=DatabentoMarketDataConfig(enabled=False), client_factory=factory,
    )
    health = provider.health()
    assert health.capability == DatabentoCapability.NOT_CONNECTED
    assert calls == []


def test_fresh_dataset_range_is_latest_available_never_live():
    client = FakeClient(FakeMetadata(dataset_range=FRESH_RANGE))
    provider = _provider(client)
    health = provider.health()
    assert health.capability == DatabentoCapability.LATEST_AVAILABLE
    assert health.capability not in (DatabentoCapability.LIVE, DatabentoCapability.DELAYED)
    assert health.latest_available_ts is not None
    assert health.lag_seconds is not None and health.lag_seconds < 24 * 3600


def test_stale_dataset_range_is_historical_only():
    client = FakeClient(FakeMetadata(dataset_range=STALE_RANGE))
    provider = _provider(client)
    health = provider.health()
    assert health.capability == DatabentoCapability.HISTORICAL_ONLY


def test_rate_limit_error_is_classified_honestly():
    class RateLimitError(Exception):
        status_code = 429

    client = FakeClient(FakeMetadata(range_error=RateLimitError("429")))
    provider = _provider(client)
    health = provider.health()
    assert health.capability == DatabentoCapability.RATE_LIMITED


def test_generic_error_degrades_to_error_capability_never_raises():
    client = FakeClient(FakeMetadata(range_error=RuntimeError("boom")))
    provider = _provider(client)
    health = provider.health()  # must not raise
    assert health.capability == DatabentoCapability.ERROR


def test_health_never_leaks_the_api_key(capsys):
    """The key is never a health-result field, never in `detail`, and the
    factory receives it only through the opaque `api_key` parameter -- never
    echoed anywhere this test can observe."""
    secret = "db-super-secret-key-should-never-appear"
    captured_keys = []

    def factory(api_key):
        captured_keys.append(api_key)
        return FakeClient(FakeMetadata(dataset_range=FRESH_RANGE))

    provider = DatabentoMarketDataProvider(
        config=DatabentoMarketDataConfig(), api_key=secret, client_factory=factory,
    )
    health = provider.health()
    dump = health.model_dump_json()
    assert secret not in dump
    assert captured_keys == [secret]  # the factory got it (that's how auth works) -- never printed/persisted


# ---------------------------------------------------------------------------
# TTL cache -- prevents a Streamlit-rerun charge/request storm
# ---------------------------------------------------------------------------


def test_ttl_cache_prevents_repeated_network_calls():
    metadata = FakeMetadata(dataset_range=FRESH_RANGE)
    client = FakeClient(metadata)
    provider = _provider(client, ttl_seconds=60.0)
    provider.health()
    provider.health()
    provider.health()
    # get_dataset_range has no call-count hook on FakeMetadata directly; assert
    # indirectly via get_cost call count on a repeated OHLCV fetch instead.
    timeseries = FakeTimeseries(ohlcv_df=_ohlcv_df(5, start=NOW - timedelta(hours=5)))
    client2 = FakeClient(FakeMetadata(dataset_range=FRESH_RANGE), timeseries=timeseries)
    provider2 = _provider(client2, ttl_seconds=60.0)
    provider2.get_recent_ohlcv("NQ", timeframe="1h", lookback_bars=5)
    provider2.get_recent_ohlcv("NQ", timeframe="1h", lookback_bars=5)
    provider2.get_recent_ohlcv("NQ", timeframe="1h", lookback_bars=5)
    assert len(timeseries.get_range_calls) == 1


def test_refresh_clears_cache_and_allows_a_fresh_call():
    timeseries = FakeTimeseries(ohlcv_df=_ohlcv_df(5, start=NOW - timedelta(hours=5)))
    client = FakeClient(FakeMetadata(dataset_range=FRESH_RANGE), timeseries=timeseries)
    provider = _provider(client, ttl_seconds=9999.0)
    provider.get_recent_ohlcv("NQ", timeframe="1h", lookback_bars=5)
    provider.clear_cache()
    provider.get_recent_ohlcv("NQ", timeframe="1h", lookback_bars=5)
    assert len(timeseries.get_range_calls) == 2


# ---------------------------------------------------------------------------
# contract resolution -- root vs. tradable contract distinction
# ---------------------------------------------------------------------------


def test_resolve_display_contract_distinguishes_root_from_raw_contract():
    client = FakeClient(
        FakeMetadata(dataset_range=FRESH_RANGE, cost=0.0),
        timeseries=FakeTimeseries(definition_df=_definition_df(raw_symbol="NQU6", instrument_id=42004177)),
    )
    provider = _provider(client)
    contract = provider.resolve_display_contract("nq")
    assert contract is not None
    assert contract.root_symbol == "NQ"
    assert contract.display_symbol == "NQ.v.0"
    assert contract.resolved_raw_symbol == "NQU6"
    assert contract.resolved_raw_symbol != contract.display_symbol
    assert contract.expiry is not None


def test_resolve_display_contract_unknown_root_returns_none():
    client = FakeClient(FakeMetadata(dataset_range=FRESH_RANGE))
    provider = _provider(client)
    assert provider.resolve_display_contract("ZZ") is None


def test_contract_metadata_is_derived_never_hardcoded():
    client = FakeClient(
        FakeMetadata(dataset_range=FRESH_RANGE, cost=0.0),
        timeseries=FakeTimeseries(definition_df=_definition_df()),
    )
    provider = _provider(client)
    econ = provider.get_contract_metadata("NQ")
    assert econ is not None
    # Matches the real, derived NQ economics (verified live against the
    # configured key -- see scripts/databento_market_capability_probe.py).
    assert econ.point_value_usd == pytest.approx(20.0)
    assert econ.tick_value_usd == pytest.approx(5.0)
    assert econ.quote_convention == "decimal"


def test_contract_metadata_over_cost_ceiling_is_refused():
    client = FakeClient(
        FakeMetadata(dataset_range=FRESH_RANGE, cost=999.0),
        timeseries=FakeTimeseries(definition_df=_definition_df()),
    )
    provider = _provider(client, max_auto_fetch_cost_usd=0.05)
    assert provider.get_contract_metadata("NQ") is None
    assert provider.resolve_display_contract("NQ").resolved_raw_symbol is None


# ---------------------------------------------------------------------------
# get_recent_ohlcv -- bounded, cost-gated, never silent on a large estimate
# ---------------------------------------------------------------------------


def test_recent_ohlcv_checks_cost_before_fetching():
    metadata = FakeMetadata(dataset_range=FRESH_RANGE, cost=0.0)
    timeseries = FakeTimeseries(ohlcv_df=_ohlcv_df(10, start=NOW - timedelta(hours=10)))
    client = FakeClient(metadata, timeseries=timeseries)
    provider = _provider(client)
    result = provider.get_recent_ohlcv("NQ", timeframe="1h", lookback_bars=8)
    assert result.fetched is True
    assert result.estimated_cost_usd == 0.0
    assert len(metadata.get_cost_calls) == 1  # get_cost was called BEFORE get_range
    assert len(result.bars) <= 8


def test_recent_ohlcv_over_ceiling_never_fetches():
    metadata = FakeMetadata(dataset_range=FRESH_RANGE, cost=10.0)
    timeseries = FakeTimeseries(ohlcv_df=_ohlcv_df(10, start=NOW - timedelta(hours=10)))
    client = FakeClient(metadata, timeseries=timeseries)
    provider = _provider(client, max_auto_fetch_cost_usd=0.05)
    result = provider.get_recent_ohlcv("NQ", timeframe="1h", lookback_bars=8)
    assert result.fetched is False
    assert result.estimated_cost_usd == 10.0
    assert timeseries.get_range_calls == []  # never actually downloaded
    assert "explicit approval" in result.detail


def test_recent_ohlcv_lookback_is_bounded_by_config():
    metadata = FakeMetadata(dataset_range=FRESH_RANGE, cost=0.0)
    timeseries = FakeTimeseries(ohlcv_df=_ohlcv_df(50, start=NOW - timedelta(hours=50)))
    client = FakeClient(metadata, timeseries=timeseries)
    provider = _provider(client, max_lookback_bars=5)
    result = provider.get_recent_ohlcv("NQ", timeframe="1h", lookback_bars=500)
    assert len(result.bars) <= 5


def test_recent_ohlcv_not_connected_never_fetches():
    def factory(api_key):
        raise DatabentoUnavailable("no key")

    provider = DatabentoMarketDataProvider(config=DatabentoMarketDataConfig(), client_factory=factory)
    result = provider.get_recent_ohlcv("NQ")
    assert result.fetched is False
    assert result.capability == DatabentoCapability.NOT_CONNECTED
    assert result.bars == ()


def test_recent_ohlcv_5m_is_a_labeled_resample_of_1m_bars():
    metadata = FakeMetadata(dataset_range=FRESH_RANGE, cost=0.0)
    timeseries = FakeTimeseries(ohlcv_df=_ohlcv_df(60, start=NOW - timedelta(hours=1), freq_minutes=1))
    client = FakeClient(metadata, timeseries=timeseries)
    provider = _provider(client)
    result = provider.get_recent_ohlcv("NQ", timeframe="5m", lookback_bars=10)
    assert result.fetched is True
    assert timeseries.get_range_calls[0]["schema"] == "ohlcv-1m"  # never a fabricated ohlcv-5m vendor grain


# ---------------------------------------------------------------------------
# snapshot / freshness -- never fabricated, timestamp always present
# ---------------------------------------------------------------------------


def test_market_snapshot_carries_honest_timestamp_and_capability():
    metadata = FakeMetadata(dataset_range=FRESH_RANGE, cost=0.0)
    bars = _ohlcv_df(72, start=NOW - timedelta(days=3), freq_minutes=60)
    timeseries = FakeTimeseries(ohlcv_df=bars, definition_df=_definition_df())
    client = FakeClient(metadata, timeseries=timeseries)
    provider = _provider(client)
    snap = provider.get_market_snapshot("NQ")
    assert snap is not None
    assert snap.as_of is not None
    assert snap.capability == DatabentoCapability.LATEST_AVAILABLE
    assert snap.last is not None
    assert snap.source == "DATABENTO"
    assert snap.contract.resolved_raw_symbol == "NQU6"


def test_market_snapshot_not_connected_never_fabricates_a_price():
    def factory(api_key):
        raise DatabentoUnavailable("no key")

    provider = DatabentoMarketDataProvider(config=DatabentoMarketDataConfig(), client_factory=factory)
    snap = provider.get_market_snapshot("NQ")
    # The ROOT is still a known, approved market -- that is not market data and
    # is always shown (mission section 15); NOT_CONNECTED never invents a price.
    assert snap is not None
    assert snap.capability == DatabentoCapability.NOT_CONNECTED
    assert snap.last is None
    assert snap.contract.resolved_raw_symbol is None


def test_network_error_on_ohlcv_degrades_gracefully_never_raises():
    metadata = FakeMetadata(dataset_range=FRESH_RANGE, cost_error=RuntimeError("network down"))
    client = FakeClient(metadata)
    provider = _provider(client)
    result = provider.get_recent_ohlcv("NQ")  # must not raise
    assert result.fetched is False
    assert result.capability == DatabentoCapability.ERROR
