"""Tests for the Market page (HOME TERMINAL + RUNTIME CONNECTIVITY pass).
Streamlit rendering is exercised via a real `AppTest` run (mirrors
`test_phase_21_streamlit_ui.py`'s pattern); `alpha_agent.ui.databento_context`
is monkeypatched so the test suite makes no real network call (CLAUDE.md:
"no billable call in unit tests"). Pure descriptive-statistic helpers live in
`alpha_agent.ui.market_home` (shared with the Agent/Home page) -- see
`test_market_home.py` for those.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from alpha_agent.marketdata.databento_schemas import (
    ContractEconomicsView,
    ContractResolution,
    DatabentoCapability,
    DatabentoHealth,
    MarketSnapshot,
    OhlcvBar,
    RecentOhlcvResult,
    SessionSummary,
)
from alpha_agent.ui import databento_context


def _fake_bars(n: int = 24, *, start_price: float = 100.0) -> tuple[OhlcvBar, ...]:
    return tuple(
        OhlcvBar(ts_event=datetime(2026, 9, 11, 13 + i % 8, tzinfo=UTC), open=start_price + i,
                  high=start_price + i + 0.5, low=start_price + i - 0.5, close=start_price + i + 0.2, volume=100.0)
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
    yield


def test_market_page_renders_without_exception():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)


def test_market_page_shows_real_snapshot_values_never_fabricated():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown) + " ".join(str(c.value) for c in at.caption)
    assert "LATEST_AVAILABLE" in full_text
    assert "DISPLAY CONTRACT" in full_text.upper()
    assert "CORE METRICS" in full_text
    # The fixture's real fabricated-for-test snapshot value (100.0 -> "100.00")
    # must appear; the mission's own illustrative example values must not.
    assert "100.00" in full_text
    assert "6,892.25" not in full_text and "6892.25" not in full_text
    assert "29,068.25" not in full_text


def test_market_page_never_says_live_from_databento():
    """LATEST_AVAILABLE is not LIVE -- the page must never claim it is."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    full_text = (" ".join(m.value for m in at.markdown) + " ".join(str(c.value) for c in at.caption)).lower()
    assert "live from databento" not in full_text
    assert "market data from databento" in full_text


def test_market_page_freshness_truth_shows_distinct_concepts():
    """Last refreshed / Latest observation / Data age are three different
    concepts and must never collapse into one -- a short, human-readable
    phrase carries all three on the primary surface ("observed ... old ...
    refreshed ..."), and the exact field-labeled values are one click away
    in Data Details. LATEST_AVAILABLE must never be described as LIVE or
    15-minute delayed anywhere."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    caption_text = " ".join(str(c.value) for c in at.caption)
    assert "observed" in caption_text.lower()
    assert "refreshed" in caption_text.lower()
    assert "old)" in caption_text  # the hero's "(<age> old)" phrasing

    # Markdown elements include the page's own injected <style> block, whose
    # CSS prose can contain incidental substrings (e.g. "sliver" contains
    # "live") -- the LIVE/DELAYED wording check stays scoped to real prose
    # (captions), never the stylesheet.
    detail_text = " ".join(m.value for m in at.markdown)
    full_text = caption_text + detail_text
    assert "Latest observation" in full_text
    assert "Observation age" in full_text
    assert "Last refreshed" in full_text
    assert "LIVE" not in caption_text.upper().replace("LATEST_AVAILABLE", "")
    assert "15-MIN" not in caption_text.upper() and "15 MIN" not in caption_text.upper()


def test_market_page_root_vs_contract_distinction():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    full_text = " ".join(m.value for m in at.markdown)
    # Root and resolved raw contract must both be visible and DIFFERENT --
    # the fixture resolves every root's display contract to "<ROOT>U6".
    root = at.session_state["market_selected_root"]
    assert root in full_text
    assert f"{root}U6" in full_text


def test_market_page_timeframe_resample_labeling(monkeypatch):
    """5m/15m are a real resample of 1m bars, never a separate vendor grain
    -- the page must say so honestly when that timeframe is selected."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    radios = [r for r in at.radio if set(r.options) == {"1m", "5m", "15m", "1h", "1d"}]
    assert radios, "expected a timeframe radio control"
    radios[0].set_value("5m").run(timeout=90)
    full_text = " ".join(str(c.value) for c in at.caption)
    assert "resample" in full_text.lower()


def test_market_page_scanner_syncs_with_the_local_selected_root():
    """The Market Scanner's row-click handler writes the SAME
    `market_selected_root` Market-local state every other Market selector
    drives -- never a second, competing selection key -- and the scanner table itself is configured for
    single-row click-to-select. The actual index -> root translation is a
    pure function (`market_scanner.root_for_selection`), unit-tested
    directly in `test_market_scanner.py`; `streamlit.testing.v1.AppTest` has
    no way to simulate a real frontend dataframe row-click, so this test
    only verifies the rendered page exposes exactly one such selectable
    table wired to the shared `market_selected_root` key."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    assert "market_selected_root" in at.session_state

    scanner_tables = [d for d in at.dataframe if d.key == "market-scanner-table"]
    assert len(scanner_tables) == 1, "expected exactly one Market Scanner selection table, no duplication"


def test_market_page_disconnected_state_never_fabricates_a_price(monkeypatch):
    """Mission Part 4 (Checkpoint A): the Market Scanner is a CATALOG browser
    -- it legitimately still renders its table when Databento is
    disconnected, honestly marking every row UNAVAILABLE/NOT_TESTED (never
    hidden, per mission Part 37: "If data is unavailable: explain exactly
    why"). What must still collapse to ONE concise card, never a grid of
    fabricated N/A metrics, is the deeper per-market observation panel below
    the scanner (mission section 13's original invariant)."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(databento_context, "health", lambda: DatabentoHealth(
        capability=DatabentoCapability.NOT_CONNECTED, dataset="GLBX.MDP3", checked_at=datetime.now(UTC),
        detail="DATABENTO_API_KEY is not set",
    ))
    monkeypatch.setattr(databento_context, "market_snapshot", lambda root: None)
    monkeypatch.setattr(databento_context, "resolve_display_contract", lambda root: None)
    monkeypatch.setattr(databento_context, "contract_metadata", lambda root: None)
    monkeypatch.setattr(databento_context, "recent_ohlcv", lambda root, **kw: None)

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    assert "MARKET DATA UNAVAILABLE" in full_text  # the deeper hero panel's honest collapse
    assert "NOT_CONNECTED" in full_text
    # The scanner table itself still renders (catalog browsing, mission Part
    # 4) -- but every cell for a disconnected root is an honest placeholder,
    # never a fabricated price; verified directly on the pure row builder in
    # test_market_scanner.py, not re-parsed out of the rendered grid here.
    assert at.dataframe, "the scanner table is a legitimate catalog view even when disconnected"
    retry_buttons = [b for b in at.button if b.key and "retry" in b.key]
    assert retry_buttons, "expected a Retry action"


def test_market_page_no_repeated_provider_calls_across_a_rerun(monkeypatch):
    """A widget interaction elsewhere on the page (any Streamlit rerun) must
    not multiply real provider calls -- the UI-session cache-first layer
    must serve the second render from cache."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    calls = {"snapshot": 0, "health": 0}

    def counting_health():
        calls["health"] += 1
        return DatabentoHealth(
            capability=DatabentoCapability.LATEST_AVAILABLE, dataset="GLBX.MDP3", checked_at=datetime.now(UTC),
            latest_available_ts=datetime(2026, 9, 11, 20, tzinfo=UTC), lag_seconds=8 * 3600.0,
        )

    def counting_snapshot(root):
        calls["snapshot"] += 1
        return _fake_snapshot(root)

    monkeypatch.setattr(databento_context, "health", counting_health)
    monkeypatch.setattr(databento_context, "market_snapshot", counting_snapshot)

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    first_snapshot_calls = calls["snapshot"]
    assert first_snapshot_calls > 0

    # Re-run the SAME AppTest instance (session_state persists, mirroring a
    # real Streamlit rerun from an unrelated widget interaction) without
    # clicking Refresh.
    at.run(timeout=90)
    assert calls["snapshot"] == first_snapshot_calls, "a bare rerun must not re-query the provider"
    assert calls["health"] == 1, "a bare rerun must not re-query health either"


def test_market_page_compare_markets_present():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    full_text = " ".join(m.value for m in at.markdown)
    assert "COMPARE MARKETS" in full_text


def test_market_page_selecting_a_category_lazily_loads_it_without_a_full_catalog_fetch(monkeypatch):
    """Section 3: selecting a category (e.g. Energy) must trigger bounded
    fetches for that category's own roots -- not the whole 36-product
    catalog -- and the scanner shell/table must render on the very same
    interaction (via the existing one-shot rerun idiom), not hang."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    fetched: set[str] = set()

    def counting_snapshot(root):
        fetched.add(root)
        return _fake_snapshot(root)

    monkeypatch.setattr(databento_context, "market_snapshot", counting_snapshot)

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    baseline = set(fetched)

    energy_radio = next(r for r in at.radio if r.key == "market-scanner-category")
    energy_radio.set_value("Energy").run(timeout=90)
    assert not list(at.exception)

    energy_roots = {"CL", "MCL", "NG", "RB", "HO"}
    assert (fetched - baseline) <= energy_roots
    # Every other catalogued category must NOT have been eagerly fetched as a
    # side effect of selecting Energy -- ZC/6E etc. may already be part of the
    # DEFAULT landing set (Section 6: one featured representative per
    # uncovered asset class), but selecting Energy must never be what ADDS
    # them.
    assert not ({"ZC", "ZS", "ZW", "ZM"} & (fetched - baseline))  # Agriculture, untouched by selecting Energy
    assert not ({"6J", "6B", "6A"} & (fetched - baseline))  # FX, untouched by selecting Energy


def test_market_page_new_category_root_starts_not_loaded_before_selection():
    """Section 3: a catalogued root outside the default eager set AND outside
    every asset class's FEATURED_COVERAGE basket (e.g. MYM, a Micro E-mini
    Dow variant no featured basket names) must not be silently pre-loaded --
    `streamlit.testing.v1.AppTest` cannot simulate a real dataframe row-click
    (see `test_market_page_selection_syncs_with_sidebar_root`'s own
    docstring), so the actual on-demand-fetch-on-click behavior is covered
    structurally in `render_scanner`'s own code path and via the pure
    `root_for_selection` / lazy-loading-key tests in
    `test_market_scanner.py`; this test only proves the PRECONDITION -- an
    unselected, non-default, non-featured root really does start NOT_LOADED,
    matching `market_universe.capability_for`'s own tested NOT_LOADED
    semantics."""
    pytest.importorskip("streamlit")
    from alpha_agent.ui.market_scanner import _loaded_roots_key
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    assert "MYM" not in at.session_state[_loaded_roots_key("market")]


def test_market_scanner_reflects_real_news_state_when_connected(monkeypatch):
    """Section 32: once a real NewsStore is connected, the scanner's News
    column shows a real count for a mapped root -- never the pre-Checkpoint-E
    NOT_CONNECTED stub."""
    from alpha_agent.marketdata.capability import CapabilityState
    from alpha_agent.ui import market_intel_context

    monkeypatch.setattr(market_intel_context, "connector_health", lambda: {"Federal Reserve": CapabilityState.AVAILABLE})
    monkeypatch.setattr(market_intel_context, "news_count_for_product", lambda root, **kw: 3 if root == "NQ" else 0)

    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)

    scanner_tables = [d for d in at.dataframe if d.key == "market-scanner-table"]
    assert scanner_tables
    df = scanner_tables[0].value  # a pandas DataFrame
    nq_rows = df[df["Symbol"] == "NQ"]
    assert len(nq_rows) == 1
    assert nq_rows.iloc[0]["News 24h"] == "3"


# ---------------------------------------------------------------------------
# Product UI Polish pass -- primary/secondary split + capability summary
# (task spec sections 3/4/5)
# ---------------------------------------------------------------------------


def test_market_page_primary_table_excludes_unavailable_rows_by_default():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)

    primary = next(d for d in at.dataframe if d.key == "market-scanner-table")
    symbols = set(primary.value["Symbol"])
    assert "GC" in symbols  # research-universe root, always eagerly fetched
    assert "ZL" not in symbols  # Soybean Oil -- not default-wired, not featured, never fetched
    assert "Research" not in primary.value.columns
    assert "Trading" not in primary.value.columns


def test_market_page_capability_summary_is_truthful_and_shown_once():
    """Section 4: the Research/Trading capability explanation renders ONCE,
    never repeated per row."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)

    caption_text = " ".join(str(c.value) for c in at.caption)
    assert caption_text.count("Research Universe:") == 1
    assert caption_text.count("Scientific Trading Eligibility:") == 1
    assert "ES" in caption_text and "NQ" in caption_text and "CL" in caption_text  # real approved roots
    assert "36 products catalogued" in caption_text


def test_market_page_unavailable_products_stay_discoverable_in_data_details():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)

    # A plain (non-`on_select`) `st.dataframe` carries no AppTest-visible
    # widget key, so the secondary table is identified by its own distinct,
    # compact column set instead (Section 5: Symbol/Product/Asset Class/
    # Reason only -- never the primary table's richer columns).
    secondary = next(d for d in at.dataframe if list(d.value.columns) == ["Symbol", "Product", "Asset Class", "Reason"])
    assert "ZL" in set(secondary.value["Symbol"])
    reasons = set(secondary.value["Reason"])
    assert reasons <= {"Not currently observed", "Not loaded", "Provider unavailable", "No usable price data"}


def test_market_page_deeper_than_home_page():
    """Mission section 17: Market is deeper exploration -- it carries
    Compare Markets, which Home does not."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at_market = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at_market.run(timeout=90)
    market_text = " ".join(m.value for m in at_market.markdown)
    assert "COMPARE MARKETS" in market_text

    at_home = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at_home.run(timeout=90)
    home_text = " ".join(m.value for m in at_home.markdown)
    assert "COMPARE MARKETS" not in home_text
