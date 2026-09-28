"""Market Intelligence + Futures Universe campaign, Checkpoint B -- Product
Detail tabbed sub-navigation (mission Part 7-8/30) and its hand-off into
Discover/Research (mission Part 29-30). Databento is mocked CONNECTED with a
real-shaped snapshot (mirrors `test_market_ui.py`'s own fixture) so the
Overview tab's content -- gated behind a health check exactly like the rest
of this platform -- is actually reachable; the honest-degradation path when
disconnected is already covered by `test_market_ui.py`.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from alpha_agent.marketdata.databento_schemas import (
    ContractEconomicsView,
    ContractLadderResult,
    ContractResolution,
    CurveShape,
    DatabentoCapability,
    DatabentoHealth,
    FuturesContract,
    MarketSnapshot,
    OhlcvBar,
    RecentOhlcvResult,
    SessionSummary,
    TermStructurePoint,
    TermStructureResult,
)
from alpha_agent.ui import databento_context


def _fake_bars(n: int = 24) -> tuple[OhlcvBar, ...]:
    return tuple(
        OhlcvBar(ts_event=datetime(2026, 9, 11, 13 + i % 8, tzinfo=UTC), open=100.0 + i,
                  high=100.5 + i, low=99.5 + i, close=100.2 + i, volume=100.0)
        for i in range(n)
    )


def _fake_snapshot(root: str) -> MarketSnapshot:
    return MarketSnapshot(
        root_symbol=root,
        contract=ContractResolution(root_symbol=root, display_symbol=f"{root}.v.0", resolved_raw_symbol=f"{root}U6",
                                     resolved_at=datetime.now(UTC)),
        last=100.0, change=0.5, change_pct=0.5,
        session=SessionSummary(root_symbol=root, session_date="2026-09-11", session_high=101.0, session_low=99.0, volume=1000.0),
        as_of=datetime.now(UTC), capability=DatabentoCapability.LATEST_AVAILABLE, freshness_seconds=1800.0,
    )


@pytest.fixture(autouse=True)
def _patch_databento(monkeypatch):
    monkeypatch.setattr(databento_context, "health", lambda: DatabentoHealth(
        capability=DatabentoCapability.LATEST_AVAILABLE, dataset="GLBX.MDP3", checked_at=datetime.now(UTC),
        latest_available_ts=datetime(2026, 9, 11, 20, tzinfo=UTC), lag_seconds=8 * 3600.0,
    ))
    monkeypatch.setattr(databento_context, "market_snapshot", lambda root: _fake_snapshot(root))
    monkeypatch.setattr(databento_context, "resolve_display_contract", lambda root: _fake_snapshot(root).contract)
    monkeypatch.setattr(databento_context, "contract_metadata", lambda root: ContractEconomicsView(
        quote_tick_size=0.25, contract_size=20.0, unit_of_measure="IPNT", point_value_usd=20.0,
        tick_value_usd=5.0, quote_convention="decimal", source="derived",
    ))
    monkeypatch.setattr(databento_context, "recent_ohlcv", lambda root, **kw: RecentOhlcvResult(
        root_symbol=root, timeframe=kw.get("timeframe", "1h"), bars=_fake_bars(),
        as_of=datetime.now(UTC), capability=DatabentoCapability.LATEST_AVAILABLE,
        estimated_cost_usd=0.0, fetched=True, detail="24 real bar(s)",
    ))
    monkeypatch.setattr(databento_context, "clear_cache", lambda: None)

    def _fake_ladder(root, **kw):
        contracts = (
            FuturesContract(
                root_symbol=root, raw_symbol=f"{root}U6", instrument_id=1,
                expiration=datetime(2026, 9, 18, tzinfo=UTC), is_front_month=True, is_display_contract=True,
                last=100.0, volume=1000.0, open_interest=5000.0, days_to_expiry=4, venue="XCME",
            ),
            FuturesContract(
                root_symbol=root, raw_symbol=f"{root}Z6", instrument_id=2,
                expiration=datetime(2026, 12, 18, tzinfo=UTC), last=101.5, volume=200.0, days_to_expiry=95,
                venue="XCME",
            ),
        )
        return ContractLadderResult(
            root_symbol=root, contracts=contracts, as_of=datetime.now(UTC),
            capability=DatabentoCapability.LATEST_AVAILABLE, fetched=True, detail="2 real outright contract(s).",
        )

    def _fake_term_structure(root, **kw):
        points = (
            TermStructurePoint(raw_symbol=f"{root}U6", expiration=datetime(2026, 9, 18, tzinfo=UTC), price=100.0,
                                price_source="last"),
            TermStructurePoint(raw_symbol=f"{root}Z6", expiration=datetime(2026, 12, 18, tzinfo=UTC), price=101.5,
                                price_source="last"),
        )
        return TermStructureResult(
            root_symbol=root, points=points, front_second_spread=1.5, curve_slope=1.5,
            curve_shape=CurveShape.CONTANGO, tolerance_pct=0.05, as_of=datetime.now(UTC),
            capability=DatabentoCapability.LATEST_AVAILABLE, detail="2 of 2 contract(s) priced.",
        )

    monkeypatch.setattr(databento_context, "contract_ladder", _fake_ladder)
    monkeypatch.setattr(databento_context, "term_structure", _fake_term_structure)
    yield


def _multipage_script() -> str:
    """A small, self-contained multipage registration covering exactly the
    pages under test -- `streamlit.testing.v1.AppTest.switch_page()` only
    matches FILE-based `st.Page(...)`, and this platform's pages are all
    callable-sourced (see `alpha_agent.ui.app.main`), so there is no way to
    land the test harness on a non-default page via that method. Registering
    Market as this script's OWN default page instead exercises the exact
    same `st.switch_page(st.Page(target.render, url_path=...))` mechanism
    the real app uses (Streamlit matches by `url_path`, not by object
    identity -- see `alpha_agent.ui.views.agent._view_research_details_
    button`'s docstring), just with a smaller registry."""
    return (
        "import streamlit as st\n"
        "from alpha_agent.ui.views import market, research, discover\n"
        "st.set_page_config(layout='wide')\n"
        "pages = [\n"
        "    st.Page(market.render, title='Market', url_path='market', default=True),\n"
        "    st.Page(research.render, title='Research', url_path='research'),\n"
        "    st.Page(discover.render, title='Discover', url_path='discover'),\n"
        "]\n"
        "st.navigation(pages, position='sidebar').run()\n"
    )


def _market_app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


# ---------------------------------------------------------------------------
# tab structure (mission Part 7)
# ---------------------------------------------------------------------------


def test_product_detail_has_all_five_tabs():
    at = _market_app()
    tab_labels = {t.label for t in at.tabs}
    assert tab_labels == {"Overview", "Contracts", "Relative", "News & Events", "Research"}


def test_every_tab_now_has_real_content_no_placeholder_left():
    """Checkpoints C/D/E/F closed the last "not yet built" gaps -- all five
    Product Detail tabs render real content now."""
    at = _market_app()
    full_text = " ".join(m.value for m in at.markdown)
    assert "Not yet built" not in full_text


# ---------------------------------------------------------------------------
# News (Checkpoint E)
# ---------------------------------------------------------------------------


def test_news_tab_honestly_shows_not_connected_when_no_connector_is_available():
    """The repo-wide test default mocks every `market_intel_context` read to
    an empty/disconnected state -- this proves the page renders that
    honestly rather than crashing or claiming zero real news."""
    at = _market_app()
    full_text = " ".join(m.value for m in at.markdown)
    assert "LATEST NEWS" in full_text
    assert "No connector is currently AVAILABLE" in full_text


def test_overview_tab_shows_chart_marker_toggles():
    at = _market_app()
    full_text = " ".join(m.value for m in at.markdown)
    assert "NEWS / EVENT MARKERS" in full_text
    checkbox_keys = {c.key for c in at.checkbox}
    assert any(k and "show-news" in k for k in checkbox_keys)
    assert any(k and "show-events" in k for k in checkbox_keys)


def test_overview_tab_observed_reaction_honestly_empty_by_default():
    """The repo-wide test default mocks `recent_news` to `()` -- proves the
    section renders that honestly rather than crashing."""
    at = _market_app()
    full_text = " ".join(m.value for m in at.markdown)
    assert "OBSERVED MARKET REACTION" in full_text
    assert "No real news item" in full_text


def test_overview_tab_observed_reaction_computes_real_metrics(monkeypatch):
    from datetime import UTC, datetime, timedelta

    from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
    from alpha_agent.ui import market_intel_context

    now = datetime.now(UTC)
    item = MarketNewsItem(
        news_id="fed:test", headline="FOMC holds rates steady", source_name="Federal Reserve",
        source_type=NewsSourceType.OFFICIAL, source_url="https://www.federalreserve.gov/test.htm",
        published_at=now - timedelta(hours=2), retrieved_at=now, category=NewsCategory.FOMC_POLICY,
        mapping_reason="test",
    )
    monkeypatch.setattr(market_intel_context, "recent_news", lambda **kw: (item,))

    at = _market_app()
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown) + " ".join(str(c.value) for c in at.caption)
    assert "OBSERVED MARKET REACTION" in full_text
    assert "FOMC holds rates steady" in full_text
    assert "never a causal claim" in full_text.lower()


def test_events_section_honestly_shows_no_events_when_disconnected():
    at = _market_app()
    full_text = " ".join(m.value for m in at.markdown)
    assert "UPCOMING EVENTS" in full_text
    assert "no real upcoming events" in full_text.lower()


def test_events_section_shows_a_real_shaped_upcoming_event(monkeypatch):
    from datetime import UTC, datetime, timedelta

    from alpha_agent.market_intel.event_schemas import EventImportance, ScheduledMarketEvent
    from alpha_agent.marketdata.capability import CapabilityState
    from alpha_agent.ui import market_intel_context

    now = datetime.now(UTC)
    event = ScheduledMarketEvent(
        event_id="fomc:test", name="FOMC Policy Statement", source_name="Federal Reserve",
        source_url="https://www.federalreserve.gov/test.htm", scheduled_at=now + timedelta(days=2),
        timezone="America/New_York", category="FOMC_POLICY", importance=EventImportance.HIGH,
        importance_rule="fomc-policy-decision-always-high/1", affected_products=("ES", "NQ"),
        mapping_reason="test", retrieved_at=now,
    )
    monkeypatch.setattr(market_intel_context, "event_connector_health", lambda: {"Federal Reserve": CapabilityState.AVAILABLE})
    monkeypatch.setattr(market_intel_context, "upcoming_events", lambda **kw: (event,))
    monkeypatch.setattr(market_intel_context, "next_event_for_product", lambda root, **kw: event)
    monkeypatch.setattr(market_intel_context, "next_high_impact_event_for_product", lambda root, **kw: event)

    at = _market_app()
    full_text = " ".join(m.value for m in at.markdown)
    assert "FOMC Policy Statement" in full_text


def test_news_tab_shows_real_shaped_items_when_connected(monkeypatch):
    from datetime import UTC, datetime, timedelta

    from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
    from alpha_agent.marketdata.capability import CapabilityState
    from alpha_agent.ui import market_intel_context

    now = datetime.now(UTC)
    item = MarketNewsItem(
        news_id="fed:test", headline="FOMC holds rates steady", source_name="Federal Reserve",
        source_type=NewsSourceType.OFFICIAL, source_url="https://www.federalreserve.gov/test.htm",
        published_at=now - timedelta(hours=2), retrieved_at=now,
        related_products=("ES", "NQ"), related_asset_classes=("EQUITY_INDEX",),
        category=NewsCategory.FOMC_POLICY, mapping_reason="test",
    )
    monkeypatch.setattr(market_intel_context, "connector_health", lambda: {"Federal Reserve": CapabilityState.AVAILABLE})
    monkeypatch.setattr(market_intel_context, "recent_news", lambda **kw: (item,))

    at = _market_app()
    full_text = " ".join(m.value for m in at.markdown)
    assert "LATEST NEWS" in full_text
    assert "sentiment" not in full_text.lower()
    assert "bullish" not in full_text.lower() and "bearish" not in full_text.lower()


# ---------------------------------------------------------------------------
# Relative tab (Checkpoint D)
# ---------------------------------------------------------------------------


def test_relative_tab_shows_related_markets_and_a_comparison_table():
    at = _market_app()
    full_text = " ".join(m.value for m in at.markdown)
    root = at.session_state["market_selected_root"]
    assert "RELATIVE MARKETS" in full_text
    assert root in full_text


# ---------------------------------------------------------------------------
# Contracts tab (Checkpoint C)
# ---------------------------------------------------------------------------


def test_contracts_tab_shows_a_real_ladder_and_term_structure():
    at = _market_app()
    full_text = " ".join(m.value for m in at.markdown)
    root = at.session_state["market_selected_root"]
    assert f"{root} CONTRACTS" in full_text
    assert "TERM STRUCTURE" in full_text
    assert "FRONT" in full_text and "DISPLAY" in full_text
    assert "Contango".upper() in full_text.upper()


def test_contracts_tab_honestly_degrades_when_databento_disconnected(monkeypatch):
    from alpha_agent.marketdata.databento_schemas import ContractLadderResult

    monkeypatch.setattr(databento_context, "health", lambda: DatabentoHealth(
        capability=DatabentoCapability.NOT_CONNECTED, dataset="GLBX.MDP3", checked_at=datetime.now(UTC),
    ))
    monkeypatch.setattr(databento_context, "contract_ladder", lambda root, **kw: ContractLadderResult(
        root_symbol=root, as_of=datetime.now(UTC), capability=DatabentoCapability.NOT_CONNECTED, fetched=False,
        detail="Databento is not currently connected.",
    ))
    at = _market_app()
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    assert "not available right now" in full_text.lower()


def test_overview_tab_still_has_the_deep_observation_content():
    """Checkpoint A/B carried the hero/chart/metrics/context panels forward
    into the Overview tab -- unchanged in substance, just relocated."""
    at = _market_app()
    full_text = " ".join(m.value for m in at.markdown)
    assert "COMPARE MARKETS" in full_text


# ---------------------------------------------------------------------------
# Research tab (mission Part 30)
# ---------------------------------------------------------------------------


def test_research_tab_honestly_refuses_a_non_research_root():
    at = _market_app()
    at.session_state["market_selected_root"] = "NG"  # catalogued Energy, not certified
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    assert "Not a Certified Research Root" in full_text
    assert "NG" in full_text


def test_research_tab_shows_real_registry_evidence_for_a_research_root():
    at = _market_app()
    at.session_state["market_selected_root"] = "NQ"
    at.run(timeout=90)
    assert not list(at.exception)
    view_buttons = [b for b in at.button if b.key == "market-research-tab-view"]
    discover_buttons = [b for b in at.button if b.key == "market-research-tab-discover"]
    assert view_buttons, "expected a View Research button on a certified research root"
    assert discover_buttons, "expected a Discover Strategies button on a certified research root"


def test_research_tab_shows_the_full_section_30_field_set_when_a_candidate_exists(monkeypatch):
    from alpha_agent.recommendation.fit import PersonalizationState
    from alpha_agent.recommendation.promise import ResearchPromiseBreakdown
    from alpha_agent.ui import services

    breakdown = ResearchPromiseBreakdown(
        performance=0.5, statistical_evidence=0.5, stability=0.5, robustness=0.5, evidence_quality=0.5,
        total=0.5, label="Promising",
    )
    candidate = services.ResearchCandidateSummary(
        experiment_id="exp1", experiment_identity="id1", root_symbol="NQ", strategy_family="tsmom",
        friendly_strategy_name="Time-Series Momentum", scientific_verdict="PASS",
        research_promise_score=0.8, research_promise_label="Promising", research_promise_breakdown=breakdown,
        user_fit_score=70.0, user_fit_label="Good fit", user_fit_evidence_coverage={},
        user_fit_personalization_state=PersonalizationState.MEASURED, user_fit_measurable_dimension_count=3,
        annualized_sharpe=1.25, net_pnl_usd=10000.0, trade_count=42, fold_consistency=0.9,
        why_promising="test", validation_blockers=(), next_research_direction="test",
    )
    monkeypatch.setattr(services, "research_candidate_summaries", lambda *a, **kw: [candidate])

    at = _market_app()
    at.session_state["market_selected_root"] = "NQ"
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    assert "1.25" in full_text  # Best Historical Sharpe
    assert "42" in full_text  # Trade Count
    assert "Number of Canonical Experiments" in full_text


# ---------------------------------------------------------------------------
# hand-off into Discover / Research (mission Part 29-30)
# ---------------------------------------------------------------------------


def test_discover_strategies_button_opens_discover_with_the_same_market_selected():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_multipage_script())
    at.run(timeout=90)
    at.session_state["market_selected_root"] = "NQ"
    at.run(timeout=90)
    assert not list(at.exception)

    btn = next(b for b in at.button if b.key == "market-research-tab-discover")
    btn.click().run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    assert "Discover Strategies" in full_text
    assert at.session_state["market_selected_root"] == "NQ"  # never a second, competing selection


def test_view_research_button_hands_off_the_selected_root_and_candidate():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_multipage_script())
    at.run(timeout=90)
    at.session_state["market_selected_root"] = "NQ"
    at.run(timeout=90)
    assert not list(at.exception)

    btn = next(b for b in at.button if b.key == "market-research-tab-view")
    btn.click().run(timeout=90)
    assert not list(at.exception)

    target = at.session_state["research_details_target"]
    assert target["source"] == "market"
    assert target["root"] == "NQ"
