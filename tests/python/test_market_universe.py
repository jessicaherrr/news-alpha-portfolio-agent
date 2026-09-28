"""Market Intelligence + Futures Universe campaign, Checkpoint A / mission
Part 40 -- the THREE UNIVERSES stay real, distinct concepts
(`alpha_agent.ui.market_universe`), and typed `FuturesProductCapability`
derivation is deterministic and never infers trading eligibility from market
data availability alone (mission Part 2: "Do not infer: market data
available => trading enabled").
"""
from __future__ import annotations

from datetime import UTC, datetime

from alpha_agent.marketdata.capability import CapabilityState
from alpha_agent.marketdata.databento_schemas import (
    ContractResolution,
    DatabentoCapability,
    DatabentoHealth,
    MarketSnapshot,
)
from alpha_agent.ui import market_universe


def _health(capability=DatabentoCapability.LATEST_AVAILABLE) -> DatabentoHealth:
    return DatabentoHealth(capability=capability, dataset="GLBX.MDP3", checked_at=datetime.now(UTC))


def _snapshot(root: str, *, raw_symbol: str | None = "NQU6", last: float | None = 100.0) -> MarketSnapshot:
    return MarketSnapshot(
        root_symbol=root,
        contract=ContractResolution(
            root_symbol=root, display_symbol=f"{root}.v.0", resolved_raw_symbol=raw_symbol,
            resolved_at=datetime.now(UTC),
        ),
        last=last, as_of=datetime.now(UTC), capability=DatabentoCapability.LATEST_AVAILABLE,
    )


# ---------------------------------------------------------------------------
# Market != Research != Trading
# ---------------------------------------------------------------------------


def test_market_universe_is_the_full_catalog():
    roots = {e.root_symbol for e in market_universe.market_universe()}
    assert "NG" in roots  # catalogued energy product, not a research root
    assert "6E" in roots  # catalogued FX product
    assert len(roots) > len(market_universe.research_universe())


def test_research_universe_is_the_frozen_five():
    assert market_universe.research_universe() == ("ES", "NQ", "CL", "GC", "ZN")


def test_trading_universe_is_never_inferred_from_market_data_alone(monkeypatch):
    """Even a root with real live market data (NG is catalogued and could in
    principle be wired) must not become trading-eligible just because it is
    observable -- trading eligibility comes only from a real paper-eligible
    registry experiment."""
    from alpha_agent.ui import services

    monkeypatch.setattr(services, "paper_eligible_experiments", list)
    assert market_universe.trading_universe() == ()


def test_trading_universe_reflects_real_paper_eligible_roots(monkeypatch):
    from alpha_agent.ui import services

    monkeypatch.setattr(
        services, "paper_eligible_experiments",
        lambda: [{"root_symbol": "NQ"}, {"root_symbol": "ES"}, {"root_symbol": "NQ"}],
    )
    assert market_universe.trading_universe() == ("ES", "NQ")


# ---------------------------------------------------------------------------
# capability_for -- deterministic, explicit, never a bare boolean where
# UNKNOWN matters
# ---------------------------------------------------------------------------


def test_capability_for_unknown_root_raises_rather_than_fabricating():
    import pytest

    with pytest.raises(ValueError):
        market_universe.capability_for("NOT_A_ROOT")


def test_capability_for_research_root_with_connected_databento():
    cap = market_universe.capability_for(
        "NQ", health=_health(), wired_roots=frozenset({"ES", "NQ", "CL", "GC", "ZN"}),
        research_roots=frozenset({"ES", "NQ", "CL", "GC", "ZN"}), trading_roots=frozenset(),
        historical_roots=frozenset({"NQ"}), attempted=True, snapshot=_snapshot("NQ"),
    )
    assert cap.root_symbol == "NQ"
    assert cap.market_data_available is CapabilityState.AVAILABLE
    assert cap.historical_data_available is CapabilityState.AVAILABLE
    assert cap.research_enabled is True
    assert cap.paper_trading_enabled is False
    assert cap.challenge_eligible is False
    assert cap.contract_economics_verified is CapabilityState.AVAILABLE
    assert cap.roll_logic_verified is CapabilityState.AVAILABLE
    assert cap.cpp_execution_verified is CapabilityState.AVAILABLE
    assert cap.verified_at is not None
    assert cap.capability_reason  # never blank


def test_capability_for_wired_root_not_yet_fetched_is_not_loaded_not_unavailable():
    """Section 1: a root whose provider support is verified (wired) but has
    had no fetch attempt yet this session/cache window must read as
    NOT_LOADED -- never a bare blank, and never confused with a real failed
    attempt (UNAVAILABLE)."""
    cap = market_universe.capability_for(
        "NQ", health=_health(), wired_roots=frozenset({"NQ"}), research_roots=frozenset({"NQ"}),
        trading_roots=frozenset(), historical_roots=frozenset({"NQ"}), attempted=False,
    )
    assert cap.market_data_available is CapabilityState.NOT_LOADED
    assert cap.verified_at is None


def test_capability_for_attempted_fetch_with_no_usable_data_is_unavailable():
    cap = market_universe.capability_for(
        "NQ", health=_health(), wired_roots=frozenset({"NQ"}), research_roots=frozenset({"NQ"}),
        trading_roots=frozenset(), historical_roots=frozenset({"NQ"}), attempted=True, snapshot=None,
    )
    assert cap.market_data_available is CapabilityState.UNAVAILABLE


def test_capability_for_price_without_resolved_raw_contract_is_degraded_not_unavailable():
    """Section 1's worked SI/HG example: a real price + economics with an
    unresolved raw display contract is a PARTIAL observation, not a failure
    -- the price must still be displayed, never discarded."""
    cap = market_universe.capability_for(
        "SI", health=_health(), wired_roots=frozenset({"SI"}), research_roots=frozenset(),
        trading_roots=frozenset(), historical_roots=frozenset(),
        attempted=True, snapshot=_snapshot("SI", raw_symbol=None, last=32.5),
    )
    assert cap.market_data_available is CapabilityState.DEGRADED


def test_capability_for_catalogued_only_root_is_not_tested_not_available():
    """NG is catalogued but not in the research universe and not wired for
    live observation this checkpoint -- CATALOGUED must never collapse into
    DATA VERIFIED (mission Part 4)."""
    cap = market_universe.capability_for(
        "NG", health=_health(), wired_roots=frozenset({"ES", "NQ", "CL", "GC", "ZN"}),
        research_roots=frozenset({"ES", "NQ", "CL", "GC", "ZN"}), trading_roots=frozenset(),
        historical_roots=frozenset(),
    )
    assert cap.market_data_available is CapabilityState.NOT_TESTED
    assert cap.historical_data_available is CapabilityState.NOT_TESTED
    assert cap.research_enabled is False
    assert cap.paper_trading_enabled is False
    assert cap.contract_economics_verified is CapabilityState.NOT_TESTED
    assert cap.verified_at is None  # never fabricate a "just checked" timestamp


def test_capability_for_wired_root_when_databento_disconnected_is_not_connected():
    """A provider-wide outage reads as NOT_CONNECTED (Section 1) -- distinct
    from a per-product UNAVAILABLE, which means the provider IS up but this
    one root's attempt returned nothing usable."""
    cap = market_universe.capability_for(
        "NQ", health=_health(DatabentoCapability.NOT_CONNECTED),
        wired_roots=frozenset({"NQ"}), research_roots=frozenset({"NQ"}),
        trading_roots=frozenset(), historical_roots=frozenset({"NQ"}), attempted=True,
    )
    assert cap.market_data_available is CapabilityState.NOT_CONNECTED


def test_capability_for_paper_trading_enabled_only_when_root_is_in_trading_universe():
    cap = market_universe.capability_for(
        "NQ", health=_health(), wired_roots=frozenset({"NQ"}), research_roots=frozenset({"NQ"}),
        trading_roots=frozenset({"NQ"}), historical_roots=frozenset({"NQ"}),
    )
    assert cap.paper_trading_enabled is True

    cap2 = market_universe.capability_for(
        "NQ", health=_health(), wired_roots=frozenset({"NQ"}), research_roots=frozenset({"NQ"}),
        trading_roots=frozenset(), historical_roots=frozenset({"NQ"}),
    )
    assert cap2.paper_trading_enabled is False


def test_capability_for_news_and_event_default_to_disabled_when_not_supplied():
    """Section 33: a caller that has not computed real market_intel state
    yet must still get a well-formed, honest DISABLED default -- never a
    crash from a missing keyword."""
    cap = market_universe.capability_for(
        "NQ", health=_health(), wired_roots=frozenset({"NQ"}), research_roots=frozenset({"NQ"}),
        trading_roots=frozenset(), historical_roots=frozenset({"NQ"}),
    )
    assert cap.news_available is CapabilityState.DISABLED
    assert cap.event_calendar_available is CapabilityState.DISABLED


def test_capability_for_wires_real_news_and_event_state_when_supplied():
    cap = market_universe.capability_for(
        "NQ", health=_health(), wired_roots=frozenset({"NQ"}), research_roots=frozenset({"NQ"}),
        trading_roots=frozenset(), historical_roots=frozenset({"NQ"}),
        news_state=CapabilityState.AVAILABLE, event_calendar_state=CapabilityState.NOT_CONNECTED,
    )
    assert cap.news_available is CapabilityState.AVAILABLE
    assert cap.event_calendar_available is CapabilityState.NOT_CONNECTED


def test_capability_derivation_is_deterministic_same_input_same_output():
    kwargs = {
        "health": _health(), "wired_roots": frozenset({"NQ"}), "research_roots": frozenset({"NQ"}),
        "trading_roots": frozenset(), "historical_roots": frozenset({"NQ"}),
    }
    a = market_universe.capability_for("NQ", **kwargs)
    b = market_universe.capability_for("NQ", **kwargs)
    assert a == b
