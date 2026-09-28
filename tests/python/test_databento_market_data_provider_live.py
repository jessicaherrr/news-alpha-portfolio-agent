"""REAL Databento capability smoke (mission Part O, section 59/60) --
skipped outright when ``DATABENTO_API_KEY`` is not configured, exactly like
`test_alpha_discovery_live_c5_github_connector.py`'s LIVE tests. Every call
here is a free metadata endpoint or a bounded, cost-checked OHLCV/definition
fetch (verified against the real configured key to price at effectively $0 --
see ``scripts/databento_market_capability_probe.py``); this file NEVER prints
the key and NEVER performs an unbounded historical download.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from alpha_agent.knowledge.connector_support import load_dotenv_if_available
from alpha_agent.marketdata.databento_provider import DatabentoMarketDataProvider
from alpha_agent.marketdata.databento_schemas import DatabentoCapability

# A local `.env` (never committed, never read directly by this module) may
# carry DATABENTO_API_KEY -- load it once so the skipif below sees the same
# environment the provider itself would see.
load_dotenv_if_available()

_HAS_KEY = bool(os.environ.get("DATABENTO_API_KEY"))
_SKIP_REASON = "DATABENTO_API_KEY not configured in this environment"


@pytest.mark.skipif(not _HAS_KEY, reason=_SKIP_REASON)
def test_live_health_proves_a_real_non_fabricated_capability():
    provider = DatabentoMarketDataProvider()
    health = provider.health()
    assert health.capability not in (DatabentoCapability.LIVE, DatabentoCapability.DELAYED), (
        "this release never proves a live/delayed streaming entitlement -- "
        "only the historical metadata surface was probed"
    )
    assert health.capability in (
        DatabentoCapability.LATEST_AVAILABLE, DatabentoCapability.HISTORICAL_ONLY,
    ), f"unexpected real capability: {health.capability}"
    assert health.dataset == "GLBX.MDP3"
    assert health.checked_at.tzinfo is not None


@pytest.mark.skipif(not _HAS_KEY, reason=_SKIP_REASON)
def test_live_nq_snapshot_never_fabricates_and_carries_real_timestamp():
    provider = DatabentoMarketDataProvider()
    snap = provider.get_market_snapshot("NQ")
    assert snap is not None
    assert snap.contract.root_symbol == "NQ"
    assert snap.contract.display_symbol == "NQ.v.0"
    if snap.capability not in (DatabentoCapability.NOT_CONNECTED, DatabentoCapability.ERROR,
                                DatabentoCapability.RATE_LIMITED):
        # A real entitlement resolved -- the display proxy must differ from
        # the actual resolved contract (mission section 15).
        assert snap.contract.resolved_raw_symbol is not None
        assert snap.contract.resolved_raw_symbol != snap.contract.display_symbol
        assert snap.last is not None
        assert snap.as_of <= datetime.now(UTC)


@pytest.mark.skipif(not _HAS_KEY, reason=_SKIP_REASON)
def test_live_contract_metadata_matches_published_nq_economics():
    provider = DatabentoMarketDataProvider()
    econ = provider.get_contract_metadata("NQ")
    if econ is None:
        pytest.skip("contract metadata not resolvable in this environment/session")
    # Published CME NQ spec: $20/point, $5/tick (0.25 tick). Derived, not
    # hardcoded -- see alpha_agent.data.contract_economics.
    assert econ.point_value_usd == pytest.approx(20.0)
    assert econ.tick_value_usd == pytest.approx(5.0)
