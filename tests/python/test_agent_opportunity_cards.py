"""Product Consolidation + Opportunity V1 campaign, Checkpoint C (Agent's
"Top Opportunities" surface and its two hand-off buttons), hardened by the
subsequent Product Acceptance Fix Pass: a real-browser check found first
paint blocking on a live multi-root Databento fetch for 80-200+ seconds, and
found the same expensive fetch re-running on every ordinary navigation/
rerun, not just an explicit refresh. Runs against this suite's GLOBAL
default (Databento disconnected, see `conftest.py`'s autouse isolation
fixtures) unless a test explicitly connects it -- `test_market_home.py`
covers the connected/real-cards path with its own module-level fixture.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

_SNAPSHOT_KEY = "agent_opportunities_snapshot"


def _fresh_app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


def _connect_fake_databento(monkeypatch, *, count_key: str = "recent_ohlcv"):
    """Mirrors `test_market_home.py::_patch_databento_for_agent` -- identical
    real-shaped fake bars for every root, so every certified research root
    honestly produces an "Up" trend and Top Opportunities has real cards to
    hand off from. Returns a `{"n": int}` counter tracking real
    `recent_ohlcv` calls -- the expensive multi-root fetch this whole file
    is about not triggering on an ordinary render/rerun/navigation."""
    from alpha_agent.marketdata.databento_schemas import (
        ContractResolution,
        DatabentoCapability,
        DatabentoHealth,
        MarketSnapshot,
        OhlcvBar,
        RecentOhlcvResult,
        SessionSummary,
    )
    from alpha_agent.ui import databento_context

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

    calls = {"n": 0}

    def _counting_recent_ohlcv(root, **kw):
        calls["n"] += 1
        return RecentOhlcvResult(
            root_symbol=root, timeframe=kw.get("timeframe", "1h"), bars=_fake_bars(),
            as_of=datetime.now(UTC), capability=DatabentoCapability.LATEST_AVAILABLE,
            estimated_cost_usd=0.0, fetched=True, detail="24 real bar(s)",
        )

    monkeypatch.setattr(databento_context, "health", lambda: DatabentoHealth(
        capability=DatabentoCapability.LATEST_AVAILABLE, dataset="GLBX.MDP3", checked_at=datetime.now(UTC),
        latest_available_ts=datetime(2026, 9, 11, 20, tzinfo=UTC), lag_seconds=8 * 3600.0,
    ))
    monkeypatch.setattr(databento_context, "market_snapshot", lambda root: _fake_snapshot(root))
    monkeypatch.setattr(databento_context, "resolve_display_contract", lambda root: _fake_snapshot(root).contract)
    monkeypatch.setattr(databento_context, "contract_metadata", lambda root: None)
    monkeypatch.setattr(databento_context, "recent_ohlcv", _counting_recent_ohlcv)
    monkeypatch.setattr(databento_context, "clear_cache", lambda: None)
    return calls


def _fresh_full_app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    # `app.py`'s page registry, landing on Ask (the Agent page): the real entrypoint
    # lands on News since the Research Thread workspace.
    from test_agent_to_research_navigation import _ASK_FIRST_APP

    at = AppTest.from_string(_ASK_FIRST_APP)
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


def _click_refresh(at):
    refresh = next(b for b in at.button if b.key == "agent-opp-refresh")
    refresh.click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


# ---------------------------------------------------------------------------
# FIRST PAINT (Product Acceptance Fix Pass, item 1) -- Agent must never
# block first paint on a live multi-root Databento fetch.
# ---------------------------------------------------------------------------


def test_agent_first_paint_makes_zero_databento_calls(monkeypatch):
    """The core acceptance requirement: opening Agent must not, by itself,
    perform the expensive multi-root fetch -- only the explicit Refresh
    Opportunities button may."""
    calls = _connect_fake_databento(monkeypatch)
    _fresh_app()
    assert calls["n"] == 0


def test_top_opportunities_shows_not_loaded_yet_before_any_refresh():
    """Honest empty state on first paint -- never a fabricated card, and
    never a blank silent gap either."""
    at = _fresh_app()
    full_text = " ".join(m.value for m in at.markdown)
    assert "Top Opportunities" in full_text
    assert "Opportunity data not loaded yet" in full_text
    assert "WHY NOW" not in full_text
    assert "Refresh Opportunities" in {b.label for b in at.button}


def test_refresh_opportunities_shows_honest_empty_state_when_disconnected():
    """Explicitly clicking Refresh while disconnected still never fabricates
    a card -- it shows the (different, now-refreshed) honest empty state."""
    at = _fresh_app()
    _click_refresh(at)
    full_text = " ".join(m.value for m in at.markdown)
    assert "No opportunity evidence was available at the last refresh" in full_text
    assert "WHY NOW" not in full_text


def test_refresh_opportunities_populates_real_cards_and_freshness(monkeypatch):
    calls = _connect_fake_databento(monkeypatch)
    at = _fresh_app()
    _click_refresh(at)
    assert calls["n"] > 0
    full_text = " ".join(m.value for m in at.markdown)
    assert "WHY NOW" in full_text
    caption_text = " ".join(c.value for c in at.caption)
    assert "Observed" in caption_text  # freshness caption
    snapshot = at.session_state[_SNAPSHOT_KEY]
    assert snapshot["opportunities"]
    assert snapshot["observed_at"] is not None


def test_top_opportunities_writes_nothing_to_the_registry(monkeypatch):
    calls = _connect_fake_databento(monkeypatch)
    before = services.registry_summary()["content_digest"]
    at = _fresh_app()
    _click_refresh(at)
    assert calls["n"] > 0
    after = services.registry_summary()["content_digest"]
    assert before == after


def test_no_opportunity_card_button_shares_a_key_with_the_hypothesis_composer(monkeypatch):
    """A regression this key-naming discipline exists to prevent: two
    Streamlit elements with the same key raise `StreamlitDuplicateElementId`
    -- asserting the button keys are all unique, both before and after a
    refresh populates the Opportunity Cards, proves Opportunity Cards and
    the (also-present, collapsed) hypothesis composer never collide."""
    _connect_fake_databento(monkeypatch)
    at = _fresh_app()
    keys = [b.key for b in at.button if b.key]
    assert len(keys) == len(set(keys))

    _click_refresh(at)
    keys_after = [b.key for b in at.button if b.key]
    assert len(keys_after) == len(set(keys_after))


# ---------------------------------------------------------------------------
# NAVIGATION MUST NEVER TRIGGER A REFRESH (Product Acceptance Fix Pass,
# item 1B). `st.session_state` is the smallest reliable mechanism that
# survives both an ordinary rerun and real page navigation (one Python
# session backs every page `st.navigation` renders -- this is the exact
# same property `market_home.py`'s own cross-page `_CACHE_KEY` already
# relies on). Pre-seeding a fresh AppTest run with an already-computed
# snapshot is functionally identical to "returning to Agent after
# navigating away": the script executes fresh either way, and the only
# thing that persists across that is `st.session_state`.
# `streamlit.testing.v1.AppTest.switch_page()` cannot drive this directly --
# it only matches FILE-based `st.Page(...)`, and every page in this app is
# callable-sourced (see `test_market_product_detail.py::_multipage_script`'s
# own docstring for the established precedent this codebase already
# documents that limitation with).
# ---------------------------------------------------------------------------


def test_returning_to_agent_with_an_existing_snapshot_makes_zero_new_fetches(monkeypatch):
    calls = _connect_fake_databento(monkeypatch)
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = _fresh_app()
    assert calls["n"] == 0  # first paint: no fetch
    _click_refresh(at)
    first_call_count = calls["n"]
    assert first_call_count > 0
    snapshot = at.session_state[_SNAPSHOT_KEY]

    # A brand-new script execution (== what "returning to a page" is under
    # the hood), pre-seeded with the SAME session_state a real navigate-
    # away-and-back would have preserved.
    at2 = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at2.session_state[_SNAPSHOT_KEY] = snapshot
    at2.run(timeout=90)
    assert not list(at2.exception), list(at2.exception)

    assert calls["n"] == first_call_count, "revisiting Agent with an existing snapshot must fetch nothing new"
    full_text = " ".join(m.value for m in at2.markdown)
    assert "WHY NOW" in full_text
    caption_text = " ".join(c.value for c in at2.caption)
    assert snapshot["observed_at"].strftime("%Y-%m-%d") in caption_text


def test_a_bare_rerun_of_agent_refetches_nothing(monkeypatch):
    """The simplest case of the same invariant: interacting with anything
    else on the page (a plain rerun, no Refresh click) must never re-fetch."""
    calls = _connect_fake_databento(monkeypatch)
    at = _fresh_app()
    _click_refresh(at)
    first_call_count = calls["n"]
    assert first_call_count > 0

    at.run(timeout=90)  # a bare rerun, e.g. from an unrelated widget interaction
    assert not list(at.exception), list(at.exception)
    assert calls["n"] == first_call_count


# ---------------------------------------------------------------------------
# HANDOFF -- Opportunity -> Open Market (explicit LOCAL Market selection) /
# Research This (Agent Experience Consolidation campaign, section 25: stays
# on Agent, seeds the SAME existing advanced research composer inline --
# never navigates away, never sets the sidebar's Workspace Market, never
# touches the ExperimentRegistry, a scientific verdict, Research Promise, or
# 2025). Cards only exist after an explicit Refresh now (item 1), so every
# handoff test refreshes first.
# ---------------------------------------------------------------------------


def test_research_this_stays_on_agent_and_seeds_the_inline_research_composer(monkeypatch):
    _connect_fake_databento(monkeypatch)
    at = _fresh_full_app()
    _click_refresh(at)
    buttons = [b for b in at.button if b.key and b.key.endswith("-research-this")]
    assert buttons, "expected at least one Opportunity Card with a Research This button"
    product = buttons[0].key[len("agent-opp-"): -len("-research-this")]

    buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    # Never navigates away, and never touches the sidebar's Workspace Market
    # -- "Research This" is a LOCAL, Agent-only research-composer hand-off.
    markdown_text = " ".join(md.value for md in at.markdown)
    assert "What deserves attention?" in markdown_text  # still on Agent
    assert "Discover Strategies" not in markdown_text

    # The SAME existing advanced composer was seeded and forced open, in
    # place -- never a second research engine, never automatic execution.
    assert at.session_state["agent-root"] == product
    assert product in at.session_state["agent-objective"]
    assert at.session_state["agent-research-expander"] is True
    assert "lab_compiled" not in at.session_state or not at.session_state["lab_compiled"]
    assert "agent_run_outcome" not in at.session_state or at.session_state["agent_run_outcome"] is None


def test_open_market_sets_markets_own_local_selection_and_reaches_market(monkeypatch):
    """Sidebar IA pass (task spec section 1E): "Open Market" is an explicit
    Agent -> Market hand-off that sets Market's OWN local selected-product
    state (`market_selected_root`) -- never a hidden global selector, and
    never Agent's own evidence scope."""
    _connect_fake_databento(monkeypatch)
    at = _fresh_full_app()
    _click_refresh(at)
    buttons = [b for b in at.button if b.key and b.key.endswith("-open-market")]
    assert buttons, "expected at least one Opportunity Card with an Open Market button"
    product = buttons[0].key[len("agent-opp-"): -len("-open-market")]

    buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    assert at.session_state["market_selected_root"] == product
    markdown_text = " ".join(md.value for md in at.markdown)
    assert "ALL FUTURES" in markdown_text  # actually landed on Market, not still on Agent
    assert f"{product} ·" in markdown_text  # Product Detail header shows the selected product


def test_handoff_never_touches_the_registry_or_2025(monkeypatch):
    _connect_fake_databento(monkeypatch)
    before = services.registry_summary()["content_digest"]
    at = _fresh_full_app()
    _click_refresh(at)
    buttons = [b for b in at.button if b.key and b.key.endswith("-research-this")]
    assert buttons
    buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    after = services.registry_summary()["content_digest"]
    assert before == after


# ---------------------------------------------------------------------------
# OPPORTUNITY PROVENANCE (Agent Evidence + Research Provenance acceptance
# pass, task spec sections 1/1A/1B/13): a card must make clear it was
# generated by the deterministic Opportunity Engine, never Claude, and its
# displayed "evidence used" must correspond to the snapshot's own real
# `evidence_refs` -- never an invented input.
# ---------------------------------------------------------------------------


def test_opportunity_card_provenance_names_the_opportunity_engine_not_claude(monkeypatch):
    _connect_fake_databento(monkeypatch)
    at = _fresh_app()
    _click_refresh(at)
    full_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Evidence & provenance" in full_text
    assert "Generated by" in full_text
    assert "Opportunity Engine" in full_text
    assert "Claude is never the source" in full_text
    # scientific status is explicitly sourced from a separate system
    assert "Experiment Registry" in full_text
    assert "not the Opportunity Engine" in full_text


def test_opportunity_card_provenance_lists_only_evidence_the_engine_actually_used(monkeypatch):
    _connect_fake_databento(monkeypatch)
    at = _fresh_app()
    _click_refresh(at)
    snapshot = at.session_state[_SNAPSHOT_KEY]
    snap = snapshot["opportunities"][0]
    caption_text = " ".join(c.value for c in at.caption)
    # every real evidence_refs entry (other than the research_verdict tag,
    # which is shown separately as "Scientific status") is represented
    for ref in snap.evidence_refs:
        key, _, value = ref.partition("=")
        if key == "research_verdict":
            continue
        assert value in caption_text, f"evidence value {value!r} from {ref!r} missing from the provenance panel"


def test_opportunity_card_explains_it_is_not_a_trade_signal(monkeypatch):
    _connect_fake_databento(monkeypatch)
    at = _fresh_app()
    _click_refresh(at)
    full_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "current research attention" in full_text
    assert "not a trade signal" in full_text


# ---------------------------------------------------------------------------
# POST-HOLDOUT RESEARCH SEED (task spec sections 2/2A/2B/2D/14): "Research
# This" on a live Opportunity (whose `observed_at` is always "now", which in
# this environment is 2026) must never trip a raw `HoldoutAccessError` in the
# normal Generate Hypothesis flow, and the origin observation's own vintage
# must be preserved and honestly marked ineligible for the 2025 holdout.
# ---------------------------------------------------------------------------


def test_research_this_then_generate_hypothesis_raises_no_raw_holdout_error(monkeypatch):
    _connect_fake_databento(monkeypatch)
    at = _fresh_app()
    _click_refresh(at)
    research_buttons = [b for b in at.button if b.key and b.key.endswith("-research-this")]
    assert research_buttons
    research_buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    # the objective actually sent to the composer must not carry the
    # contemporary observation's own ISO timestamp
    objective = at.session_state["agent-objective"]
    today_iso_date = at.session_state["agent_research_origin"]["origin_vintage"]
    assert today_iso_date not in objective

    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    full_text = " ".join(m.value for m in at.markdown) + " ".join(e.value for e in at.error)
    assert "HoldoutAccessError" not in full_text
    # the pipeline actually succeeded end to end (scripted mode default)
    assert "ra_proposal" in at.session_state and at.session_state["ra_proposal"] is not None
    assert at.session_state["ra_proposal"]["accepted"] is True


def test_research_this_marks_origin_as_not_eligible_for_the_2025_holdout(monkeypatch):
    _connect_fake_databento(monkeypatch)
    at = _fresh_app()
    _click_refresh(at)
    research_buttons = [b for b in at.button if b.key and b.key.endswith("-research-this")]
    assert research_buttons
    research_buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    origin = at.session_state["agent_research_origin"]
    # the observed_at PROVENANCE itself is still preserved verbatim...
    assert origin["source"] == "opportunity"
    assert origin.get("observed_at")
    # ...but is honestly marked as NOT eligible for the sealed 2025 holdout,
    # since a live observation's own wall clock is always "now" (>= 2025).
    assert origin["holdout_eligible"] is False
    assert "origin_vintage" in origin

    full_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "NOT ELIGIBLE FOR 2025 HOLDOUT" in full_text
    assert "Historical research remains available" in full_text
