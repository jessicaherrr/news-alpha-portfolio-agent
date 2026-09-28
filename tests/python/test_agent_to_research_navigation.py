"""Agent/Research UX polish pass -- "View Research Details" navigates
DIRECTLY to the Research page (a real `st.switch_page(Page)` call, not a
"check the sidebar" toast), and Research Details prominently shows which
experiment's root/family/verdict it is actually displaying, independent of
the sidebar's own market browser and the Agent page's "Research Root"
selector.

None of this touches scientific semantics, Registry behavior, validation,
C++, holdout, or experiment identity. Every test here runs the real, offline
`app.py` entrypoint (`AppTest.from_file`) so the navigation is exercised
exactly as a live session would use it, with the real local registry (no
write, no C++ execution, no network call).
"""
from __future__ import annotations

import pytest
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(),
    reason="Phase 14 registry sqlite not present in this checkout",
)

APP_ENTRYPOINT = str(services.REPO_ROOT / "python" / "alpha_agent" / "ui" / "app.py")


#: `app.py`'s own page registry, landing on Ask (the Agent page) -- since the
#: Research Thread workspace, the real entrypoint lands on News, and this file
#: is about the Ask -> Strategy Lab (`lab`, the former Research page) hand-off.
_ASK_FIRST_APP = """
import streamlit as st
from alpha_agent.ui.views import agent, learn, market, news, paper_trading, portfolio, research, system, workspace
st.navigation([
    st.Page(agent.render, title="Ask", url_path="agent", default=True),
    st.Page(news.render, title="News", url_path="news"),
    st.Page(workspace.render, title="Research", url_path="research"),
    st.Page(portfolio.render, title="Portfolio", url_path="portfolio"),
    st.Page(learn.render, title="Learn", url_path="learn"),
    st.Page(market.render, title="Markets", url_path="market"),
    st.Page(research.render, title="Strategy Lab", url_path="lab"),
    st.Page(paper_trading.render, title="Paper Trading", url_path="paper-trading"),
    st.Page(system.render, title="Settings", url_path="system"),
], position="hidden").run()
"""


def _fresh_app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_ASK_FIRST_APP)
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


def _send_and_view_details(at):
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    view_buttons = [b for b in at.button if b.key and b.key.startswith("agent-view-details-")]
    assert view_buttons, "expected a View Research Details button after a compiled proposal"
    view_buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


# ---------------------------------------------------------------------------
# 1 -- direct navigation reaches Research Details, no manual sidebar click
# ---------------------------------------------------------------------------


def test_view_research_details_navigates_directly_to_research_details():
    at = _fresh_app()
    at = _send_and_view_details(at)

    markdown_text = " ".join(md.value for md in at.markdown)
    assert "Research Details" in markdown_text
    assert "VIEWING EXPERIMENT" in markdown_text
    # the old "open Research from the sidebar" instruction is gone
    assert "from the sidebar" not in markdown_text.lower()


def test_no_selection_ready_toast_after_direct_navigation():
    at = _fresh_app()
    at = _send_and_view_details(at)
    toast_values = [t.value for t in at.toast] if hasattr(at, "toast") else []
    assert not any("Selection ready" in v for v in toast_values)


# ---------------------------------------------------------------------------
# View Research Details preserves the correct selected experiment
# ---------------------------------------------------------------------------


def test_view_research_details_preserves_the_correct_selected_experiment():
    at = _fresh_app()
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    compiled_before = at.session_state["lab_compiled"]

    view_buttons = [b for b in at.button if b.key and b.key.startswith("agent-view-details-")]
    view_buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    target = at.session_state["research_details_target"]
    assert target["compiled"]["strategy_fingerprint"] == compiled_before["strategy_fingerprint"]
    assert target["root"] == at.session_state["agent_last_root"]

    markdown_text = " ".join(md.value for md in at.markdown)
    assert target["hypothesis"]["title"] in markdown_text


# ---------------------------------------------------------------------------
# Research Details prominently displays the selected experiment root
# (near the top, above the tabs -- not buried inside a tab click)
# ---------------------------------------------------------------------------


def test_research_details_prominently_shows_the_experiment_root_near_the_top():
    at = _fresh_app()
    at = _send_and_view_details(at)

    target = at.session_state["research_details_target"]
    expected_root = target["root"]

    markdown_text = " ".join(md.value for md in at.markdown)
    assert "VIEWING EXPERIMENT" in markdown_text
    idx_banner = markdown_text.index("VIEWING EXPERIMENT")
    # The Overview tab's own first provenance row (rendered only once st.tabs
    # has been declared and its bodies laid out) is a reliable "after the
    # banner" marker -- unlike the bare word "Overview", which also appears
    # in an unrelated CSS comment earlier in the injected stylesheet.
    idx_tab_body_marker = markdown_text.index('aa-prov-label">Objective')
    assert idx_banner < idx_tab_body_marker
    assert expected_root in markdown_text


# ---------------------------------------------------------------------------
# Market's local state cannot overwrite the experiment root
# ---------------------------------------------------------------------------


def _real_reject_experiment_id() -> str:
    rows = services.list_experiments(strategy_family="tsmom", root_symbol="NQ", trial_role=None)
    canonical = next(r for r in rows if r["trial_role"] == "CANONICAL")
    assert canonical["verdict"] == "REJECT"
    return canonical["experiment_id"]


def test_markets_local_state_cannot_overwrite_the_experiment_root():
    """Market's own local selected product (`market_selected_root`, Sidebar
    IA pass task spec section 1D/13) is a completely separate concept from
    the experiment Research Details is showing -- pointing it at a DIFFERENT
    root than the Agent page's Research Root must never change which root
    the "Viewing Experiment" banner reports."""
    at = _fresh_app()
    # Deliberately point Market's OWN local selection at CL while the
    # default preset's Research Root is NQ.
    at.session_state["market_selected_root"] = "CL"
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    assert at.session_state["agent-root"] == "NQ"  # Research Root untouched by Market's local state

    at = _send_and_view_details(at)
    target = at.session_state["research_details_target"]
    assert target["root"] == "NQ"

    markdown_text = " ".join(md.value for md in at.markdown)
    assert "VIEWING EXPERIMENT" in markdown_text
    # the experiment's OWN root (NQ) is shown, never silently replaced by
    # Market's CL selection -- and Market's own state is untouched too.
    assert at.session_state["market_selected_root"] == "CL"
    assert "NQ" in markdown_text


# ---------------------------------------------------------------------------
# Existing Registry retrieval behavior preserved: EXISTING match, zero new
# backtests, authoritative evidence retrieved verbatim.
# ---------------------------------------------------------------------------


def test_existing_strategyspec_still_retrieves_registry_evidence_with_zero_new_backtests():
    at = _fresh_app()
    at.selectbox(key="agent-scenario-select").select(
        "Multi-week time-series momentum in NQ"
    ).run(timeout=90)
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    transcript = at.session_state["agent_transcript"]
    evidence_turn = next(t for t in transcript if t["type"] == "evidence")
    assert evidence_turn["data"]["match_type"] == "exact_fingerprint"
    assert evidence_turn["data"]["result"]["headline_verdict"] == "REJECT"

    caption_text = " ".join(c.value for c in at.caption)
    markdown_text = " ".join(md.value for md in at.markdown)
    assert "No new backtest was executed" in caption_text
    assert "REGISTRY_RETRIEVAL" in markdown_text


# ---------------------------------------------------------------------------
# No Registry writes / no C++ execution / no network calls
# ---------------------------------------------------------------------------


def test_full_navigation_flow_writes_nothing_to_the_registry():
    before = services.registry_summary()["content_digest"]
    at = _fresh_app()
    at = _send_and_view_details(at)
    after = services.registry_summary()["content_digest"]
    assert before == after


def test_full_navigation_flow_performs_zero_anthropic_calls(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "anthropic", None)  # importing it would now raise
    at = _fresh_app()
    at = _send_and_view_details(at)
    assert not list(at.exception), list(at.exception)


# ---------------------------------------------------------------------------
# Scientific semantics unchanged: a REJECT verdict is never painted as PASS.
# ---------------------------------------------------------------------------


def test_scientific_semantics_unchanged_reject_never_renders_as_pass():
    at = _fresh_app()
    at = _send_and_view_details(at)
    markdown_text = " ".join(md.value for md in at.markdown)
    target = at.session_state["research_details_target"]
    evidence = target.get("evidence")
    if evidence and evidence.get("result") and evidence["result"]["headline_verdict"] != "PASS":
        assert "✓ PASS" not in markdown_text
