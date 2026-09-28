"""Tests for `alpha_agent.ui.market_home` -- the shared MARKET REALITY
rendering (ticker strip, Selected Market panel, price/volume chart, Market
Context) used by both the Agent/Home page and the Market page. Pure
descriptive-statistic helpers are tested directly (no Streamlit needed);
the UI-session cache-first layer is tested through a real `AppTest` run of
the Agent page (mirrors `test_market_ui.py`'s pattern).
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from alpha_agent.marketdata.databento_config import DatabentoMarketDataConfig
from alpha_agent.marketdata.databento_schemas import (
    ContractResolution,
    DatabentoCapability,
    MarketSnapshot,
    OhlcvBar,
    SessionSummary,
)
from alpha_agent.ui import market_home


def _fake_bars():
    return tuple(
        OhlcvBar(ts_event=datetime(2026, 9, 11, 13 + i % 8, tzinfo=UTC), open=100.0 + i,
                  high=100.5 + i, low=99.5 + i, close=100.2 + i, volume=100.0)
        for i in range(24)
    )


def _fake_snapshot(root: str) -> MarketSnapshot:
    return MarketSnapshot(
        root_symbol=root,
        contract=ContractResolution(root_symbol=root, display_symbol=f"{root}.v.0",
                                     resolved_raw_symbol=f"{root}U6", resolved_at=datetime.now(UTC)),
        last=100.0, change_pct=0.5,
        session=SessionSummary(root_symbol=root, session_date="2026-09-11", session_high=101.0,
                                session_low=99.0, volume=1000.0),
        as_of=datetime.now(UTC), capability=DatabentoCapability.LATEST_AVAILABLE, freshness_seconds=1800.0,
    )


# ---------------------------------------------------------------------------
# pure descriptive statistics -- never a trading signal
# ---------------------------------------------------------------------------


def test_trend_state_labels_never_call_themselves_a_signal():
    up = [{"close": 100.0}, {"close": 105.0}]
    down = [{"close": 100.0}, {"close": 90.0}]
    flat = [{"close": 100.0}, {"close": 100.2}]
    assert market_home.trend_state(up) == "Up"
    assert market_home.trend_state(down) == "Down"
    assert market_home.trend_state(flat) == "Sideways"
    assert market_home.trend_state([{"close": 100.0}]) == "Insufficient data"
    for label in ("BUY", "SELL", "SIGNAL"):
        assert label not in (market_home.trend_state(up), market_home.trend_state(down))


def test_volatility_state_is_descriptive_only():
    calm = [{"close": 100.0}, {"close": 100.05}, {"close": 100.02}]
    wild = [{"close": 100.0}, {"close": 110.0}, {"close": 95.0}]
    assert market_home.volatility_state(calm) == "Low"
    assert market_home.volatility_state(wild) == "High"


def test_realized_volatility_pct_is_none_for_insufficient_bars():
    assert market_home.realized_volatility_pct([]) is None
    assert market_home.realized_volatility_pct([{"close": 100.0}]) is None


def test_realized_volatility_pct_is_a_positive_annualized_percent():
    bars = [{"close": 100.0 + (1 if i % 2 == 0 else -1)} for i in range(30)]
    rv = market_home.realized_volatility_pct(bars)
    assert rv is not None
    assert rv > 0.0


def test_realized_volatility_pct_zero_for_flat_series():
    bars = [{"close": 100.0} for _ in range(10)]
    assert market_home.realized_volatility_pct(bars) == 0.0


def test_volume_context_state_never_zero_fills_missing_volume():
    with_volume = [{"volume": 100.0}] * 5 + [{"volume": 500.0}]
    no_volume = [{"volume": None}] * 5
    missing_key = [{}] * 5
    assert market_home.volume_context_state(with_volume) == "Elevated"
    assert market_home.volume_context_state(no_volume) == "Unavailable"
    assert market_home.volume_context_state(missing_key) == "Unavailable"
    assert market_home.volume_context_state([{"volume": 100.0}]) == "Unavailable"


def test_volume_context_state_quiet_and_normal():
    quiet = [{"volume": 100.0}] * 5 + [{"volume": 10.0}]
    normal = [{"volume": 100.0}] * 5 + [{"volume": 100.0}]
    assert market_home.volume_context_state(quiet) == "Quiet"
    assert market_home.volume_context_state(normal) == "Normal"


def test_session_position_state_five_way():
    def bars_at(pct: float) -> list[dict]:
        last = 95.0 + pct * 10.0
        base = [{"high": 105.0, "low": 95.0, "close": 100.0} for _ in range(2)]
        return [*base, {"high": 105.0, "low": 95.0, "close": last}]

    assert market_home.session_position_state(bars_at(0.95)) == "Near High"
    assert market_home.session_position_state(bars_at(0.7)) == "Upper Half"
    assert market_home.session_position_state(bars_at(0.5)) == "Middle"
    assert market_home.session_position_state(bars_at(0.3)) == "Lower Half"
    assert market_home.session_position_state(bars_at(0.05)) == "Near Low"
    assert market_home.session_position_state([]) == "Insufficient data"


def test_session_range_and_distance_pct():
    assert market_home.session_range_pct(105.0, 100.0) == pytest.approx(5.0)
    assert market_home.session_range_pct(None, 100.0) is None
    assert market_home.session_range_pct(105.0, 0.0) is None

    assert market_home.distance_from_high_pct(100.0, 105.0) == pytest.approx(-4.7619, rel=1e-3)
    assert market_home.distance_from_high_pct(None, 105.0) is None

    assert market_home.distance_from_low_pct(102.0, 100.0) == pytest.approx(2.0)
    assert market_home.distance_from_low_pct(100.0, None) is None


def test_age_label_distinguishes_minutes_hours_days():
    assert market_home.age_label(None) == "N/A"
    assert "min" in market_home.age_label(90.0)
    assert "hours" in market_home.age_label(3 * 3600.0)
    assert "days" in market_home.age_label(60 * 3600.0)


# ---------------------------------------------------------------------------
# config -- the UI-session cache window is bounded and configurable
# ---------------------------------------------------------------------------


def test_ui_refresh_interval_has_a_bounded_default():
    cfg = DatabentoMarketDataConfig()
    assert cfg.ui_refresh_interval_seconds > 0
    # Never faster than the provider's own request-level TTL -- refreshing
    # the UI more often than the provider itself would re-query buys nothing.
    assert cfg.ui_refresh_interval_seconds >= cfg.ttl_seconds


# ---------------------------------------------------------------------------
# cache-first plumbing, exercised through a real page (Agent) -- mirrors
# test_market_ui.py's equivalent Market-page test so both pages are held to
# the same "no rerun request storm" standard.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_databento_for_agent(monkeypatch):
    from alpha_agent.marketdata.databento_schemas import DatabentoHealth, RecentOhlcvResult
    from alpha_agent.ui import databento_context

    monkeypatch.setattr(databento_context, "health", lambda: DatabentoHealth(
        capability=DatabentoCapability.LATEST_AVAILABLE, dataset="GLBX.MDP3", checked_at=datetime.now(UTC),
        latest_available_ts=datetime(2026, 9, 11, 20, tzinfo=UTC), lag_seconds=8 * 3600.0,
    ))
    monkeypatch.setattr(databento_context, "market_snapshot", lambda root: _fake_snapshot(root))
    monkeypatch.setattr(databento_context, "resolve_display_contract", lambda root: _fake_snapshot(root).contract)
    monkeypatch.setattr(databento_context, "contract_metadata", lambda root: None)
    monkeypatch.setattr(databento_context, "recent_ohlcv", lambda root, **kw: RecentOhlcvResult(
        root_symbol=root, timeframe=kw.get("timeframe", "1h"), bars=_fake_bars(),
        as_of=datetime.now(UTC), capability=DatabentoCapability.LATEST_AVAILABLE,
        estimated_cost_usd=0.0, fetched=True, detail="24 real bar(s)",
    ))
    monkeypatch.setattr(databento_context, "clear_cache", lambda: None)
    yield


def test_get_snapshots_for_universe_fetches_roots_concurrently(monkeypatch):
    """A cold ticker strip's real bottleneck was 5 roots' independent
    provider round trips paid SERIALLY (measured against the real
    Databento entitlement: ~11s/root, ~55s total). Each root is fetched
    concurrently instead -- assert wall time is well under what 5 serial
    calls would take, and that every root still gets its OWN correct
    snapshot (never mixed up across threads)."""
    pytest.importorskip("streamlit")
    import time as time_module

    from alpha_agent.ui import databento_context

    def slow_snapshot(root):
        time_module.sleep(0.3)
        return _fake_snapshot(root)

    monkeypatch.setattr(databento_context, "market_snapshot", slow_snapshot)

    from streamlit.testing.v1 import AppTest

    script = (
        "import time\n"
        "import streamlit as st\n"
        "from alpha_agent.ui import market_home\n"
        "market_home.get_health_cached()\n"
        "t0 = time.monotonic()\n"
        "result = market_home.get_snapshots_for_universe(('ES', 'NQ', 'CL', 'GC', 'ZN'))\n"
        "st.session_state['elapsed'] = time.monotonic() - t0\n"
        "st.session_state['roots'] = sorted(result.keys())\n"
        "st.session_state['mismatched'] = [r for r, snap in result.items() if snap.root_symbol != r]\n"
    )
    at = AppTest.from_string(script)
    at.run(timeout=90)
    assert not list(at.exception)
    assert at.session_state["roots"] == ["CL", "ES", "GC", "NQ", "ZN"]
    assert at.session_state["mismatched"] == []
    # 5 roots x 0.3s serially would take >=1.5s; concurrent must be well under that.
    assert at.session_state["elapsed"] < 1.0


def test_get_snapshots_for_universe_reuses_cache_without_refetching(monkeypatch):
    pytest.importorskip("streamlit")
    from alpha_agent.ui import databento_context

    calls = {"n": 0}
    original = databento_context.market_snapshot

    def counting(root):
        calls["n"] += 1
        return original(root)

    monkeypatch.setattr(databento_context, "market_snapshot", counting)

    from streamlit.testing.v1 import AppTest

    script = (
        "from alpha_agent.ui import market_home\n"
        "market_home.get_health_cached()\n"
        "first = market_home.get_snapshots_for_universe(('ES', 'NQ'))\n"
        "second = market_home.get_snapshots_for_universe(('ES', 'NQ'))\n"
        "import streamlit as st\n"
        "st.session_state['ok'] = first == second\n"
    )
    at = AppTest.from_string(script)
    at.run(timeout=90)
    assert not list(at.exception)
    assert calls["n"] == 2, "second batch call must reuse the session cache, not re-fetch"


def test_get_ohlcv_for_universe_fetches_roots_concurrently(monkeypatch):
    """Product UI Polish pass, section 7: the Market Scanner's own OHLCV
    batch (`market_scanner._fetch_wired_market_state`) used to loop over
    roots SERIALLY -- exactly the class of bug `test_get_snapshots_for_
    universe_fetches_roots_concurrently` already documents for the ticker
    strip. Same proof, for the OHLCV batch: concurrent, never mixed up
    across threads."""
    pytest.importorskip("streamlit")
    import time as time_module

    from alpha_agent.ui import databento_context

    def slow_ohlcv(root, **kw):
        time_module.sleep(0.3)
        from alpha_agent.marketdata.databento_schemas import RecentOhlcvResult

        return RecentOhlcvResult(
            root_symbol=root, timeframe=kw.get("timeframe", "1h"), bars=_fake_bars(),
            as_of=datetime.now(UTC), capability=DatabentoCapability.LATEST_AVAILABLE,
            estimated_cost_usd=0.0, fetched=True, detail="24 real bar(s)",
        )

    monkeypatch.setattr(databento_context, "recent_ohlcv", slow_ohlcv)

    from streamlit.testing.v1 import AppTest

    script = (
        "import time\n"
        "import streamlit as st\n"
        "from alpha_agent.ui import market_home\n"
        "market_home.get_health_cached()\n"
        "t0 = time.monotonic()\n"
        "result = market_home.get_ohlcv_for_universe(('ES', 'NQ', 'CL', 'GC', 'ZN'), timeframe='1h', lookback_bars=72)\n"
        "st.session_state['elapsed'] = time.monotonic() - t0\n"
        "st.session_state['roots'] = sorted(result.keys())\n"
        "st.session_state['mismatched'] = [r for r, res in result.items() if res.root_symbol != r]\n"
    )
    at = AppTest.from_string(script)
    at.run(timeout=90)
    assert not list(at.exception)
    assert at.session_state["roots"] == ["CL", "ES", "GC", "NQ", "ZN"]
    assert at.session_state["mismatched"] == []
    # 5 roots x 0.3s serially would take >=1.5s; concurrent must be well under that.
    assert at.session_state["elapsed"] < 1.0


def test_get_ohlcv_for_universe_reuses_cache_without_refetching(monkeypatch):
    pytest.importorskip("streamlit")
    from alpha_agent.ui import databento_context

    calls = {"n": 0}
    original = databento_context.recent_ohlcv

    def counting(root, **kw):
        calls["n"] += 1
        return original(root, **kw)

    monkeypatch.setattr(databento_context, "recent_ohlcv", counting)

    from streamlit.testing.v1 import AppTest

    script = (
        "from alpha_agent.ui import market_home\n"
        "market_home.get_health_cached()\n"
        "first = market_home.get_ohlcv_for_universe(('ES', 'NQ'), timeframe='1h', lookback_bars=72)\n"
        "second = market_home.get_ohlcv_for_universe(('ES', 'NQ'), timeframe='1h', lookback_bars=72)\n"
        "import streamlit as st\n"
        "st.session_state['ok'] = first == second\n"
    )
    at = AppTest.from_string(script)
    at.run(timeout=90)
    assert not list(at.exception)
    assert calls["n"] == 2, "second batch call must reuse the session cache, not re-fetch"


def test_agent_home_page_shows_attention_hero_before_the_composer():
    """Agent Experience Consolidation campaign (section 14): Agent is a
    SIMPLE landing page -- "What deserves attention?" -> Top Opportunities ->
    Conversation Timeline -> "Ask Agentic Alpha" composer, in that order (the
    composer sits LAST in source order so it reads as following the
    conversation, and is pinned to the bottom of the viewport via sticky
    CSS) -- and no longer embeds the full market terminal (ticker strip /
    chart / Core Metrics / Market Context) that the Market page already
    owns."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    assert "DISPLAY CONTRACT" not in full_text
    assert "CORE METRICS" not in full_text
    assert "MARKET CONTEXT" not in full_text

    hero_idx = full_text.find("What deserves attention?")
    opp_idx = full_text.find("Top Opportunities")
    ask_idx = full_text.find("Ask Agentic Alpha")
    assert -1 not in (hero_idx, ask_idx, opp_idx)
    assert hero_idx < opp_idx < ask_idx, "hero -> Top Opportunities -> Ask Agentic Alpha, in that order"


def test_agent_home_page_works_without_anthropic_key(monkeypatch):
    """Mission section 16: substantial value with no ANTHROPIC_API_KEY."""
    pytest.importorskip("streamlit")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    assert "What deserves attention?" in full_text
    assert "Ask Agentic Alpha" in full_text
    assert "Top Opportunities" in full_text


def test_agent_home_page_top_opportunities_renders_real_cards_when_connected():
    """This file's own autouse fixture fakes Databento CONNECTED with real-
    shaped bars for every root -- Top Opportunities should render real,
    evidence-backed cards (never the NOT_AVAILABLE empty state) after an
    explicit Refresh (Product Acceptance Fix Pass: first paint never
    auto-fetches -- see `test_agent_opportunity_cards.py`), each with both
    hand-off buttons. That file also covers the disconnected/empty-state,
    first-paint-zero-fetches, and hand-off-preserves-root tests."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    refresh = next(b for b in at.button if b.key == "agent-opp-refresh")
    refresh.click().run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    assert "Top Opportunities" in full_text
    assert "WHY NOW" in full_text
    assert "RESEARCH" in full_text
    research_this_buttons = [b for b in at.button if b.key and b.key.endswith("-research-this")]
    open_market_buttons = [b for b in at.button if b.key and b.key.endswith("-open-market")]
    assert research_this_buttons and open_market_buttons


def test_agent_home_page_developer_status_moved_out_of_the_header_entirely():
    """Sidebar IA pass (task spec sections 10/12): the global header is now
    brand + a page-specific subtitle only -- no backend-engineering pills
    (C++ Core / pybind11 / Data / Registry / Holdout), no commit hash, and no
    "System status" expander competing with normal product pages. That
    detail is not deleted -- it moved to Settings -> Runtime / About (see
    `test_settings_runtime_and_about_carry_the_relocated_detail` in
    `test_sidebar_nav.py`)."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    assert "System status" not in [e.label for e in at.expander]
    assert "pybind11" not in full_text
    assert "C++ Core" not in full_text
    assert "commit " not in full_text


def test_topbar_html_never_has_a_blank_line():
    """Regression (caught in real live-browser verification, not AppTest --
    see `layout._topbar_html`'s own docstring for the full mechanism): a
    blank line inside a raw-HTML `st.markdown(..., unsafe_allow_html=True)`
    block made Streamlit's frontend markdown parser escape the closing
    `</div>` tags AFTER it as literal text instead of rendering them as
    HTML. `AppTest` cannot catch this class of bug itself (it inspects the
    raw source string Streamlit was given, not how the browser's markdown
    parser interprets it) -- this tests the actual root cause directly:
    the builder must never produce a whitespace-only line, empty
    `right_side` included."""
    from alpha_agent.ui import layout

    for subtitle in ("", "A page-specific subtitle"):
        for right_side in ("", '<span class="aa-badge">Attention needed</span>'):
            html = layout._topbar_html("<svg></svg>", subtitle, right_side)
            lines = html.split("\n")
            assert all(line.strip() for line in lines), f"blank line in topbar html: {lines!r}"
            assert html.count("<div") == html.count("</div>")


def test_market_page_no_repeated_provider_calls_across_a_rerun(monkeypatch):
    """The market_home session-cache layer this test actually exercises --
    retargeted to the Market page (Product Consolidation campaign,
    Checkpoint C moved the full market terminal off Agent; Market is now
    the one page that unconditionally calls `market_snapshot` on a bare
    render)."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from alpha_agent.ui import databento_context
    from streamlit.testing.v1 import AppTest

    calls = {"snapshot": 0}
    original = databento_context.market_snapshot

    def counting_snapshot(root):
        calls["snapshot"] += 1
        return original(root)

    monkeypatch.setattr(databento_context, "market_snapshot", counting_snapshot)

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    first = calls["snapshot"]
    assert first > 0
    at.run(timeout=90)
    assert calls["snapshot"] == first, "a bare rerun must not re-query the provider"
