"""Phase 7 -- Research Map UI (prompt section 7/10).

Real `streamlit.testing.v1.AppTest` renders, mirroring
`test_alpha_library_ui.py`'s own pattern. Proves the Research Map renders
inside Research (never a new top-level page), a Mechanism's detail view
opens and can navigate into the existing Factor Library detail view, and
browsing never runs a backtest, calls Claude, calls a market-data API, or
writes the registry.
"""
from __future__ import annotations

import re

import pytest
from alpha_agent.ui import services
from alpha_agent.ui.views import alpha_graph

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)


def _fresh_app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.alpha_graph import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


# ---------------------------------------------------------------------------
# Lives inside Research, not a new top-level page (prompt section 7/9)
# ---------------------------------------------------------------------------


def test_research_map_is_a_tab_on_learn_not_a_new_top_level_page():
    """Research Map moved from the old Research landing to Learn (the Research
    Thread workspace): "what has been researched before" belongs with the
    research memory, apart from a thread's economic Mechanism Graph."""
    from alpha_agent.ui.views import learn

    assert learn._TABS["alpha_graph"] == "Research Map"


def test_research_map_is_not_registered_as_its_own_app_page():
    with open(str(services.REPO_ROOT / "python" / "alpha_agent" / "ui" / "app.py"), encoding="utf-8") as fh:
        app_src = fh.read()
    assert "alpha_graph.render" not in app_src


def test_research_map_is_reachable_from_the_shared_sidebar_subnav():
    from alpha_agent.ui import layout

    assert ("alpha_graph", "Research Map") in layout._LEARN_SUBVIEWS


# ---------------------------------------------------------------------------
# Renders without running research / calling Claude / calling market data /
# writing the registry, and triggers no network call (prompt section 10)
# ---------------------------------------------------------------------------


def test_landing_renders_without_running_research_calling_claude_or_market_data(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("Research Map must never call a live LLM client just by rendering")

    from alpha_agent.ui import llm_demo

    monkeypatch.setattr(llm_demo, "build_llm_client", _boom, raising=False)
    at = _fresh_app()
    body = " ".join(m.value for m in at.markdown)
    assert "Research Map" in body


def test_landing_shows_real_mechanism_cards_and_a_research_gaps_table():
    at = _fresh_app()
    body = " ".join(m.value for m in at.markdown)
    assert "TREND" in body
    open_buttons = [b for b in at.button if b.key and b.key.startswith("alpha-graph-open-")]
    assert open_buttons
    assert len(at.dataframe) >= 1  # the research-gaps table


def test_opening_a_mechanism_never_triggers_execution(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("opening a Mechanism must never trigger execution")

    from alpha_agent.ui import deep_research

    monkeypatch.setattr(deep_research, "run_deep_research", _boom, raising=False)
    at = _fresh_app()
    open_buttons = [b for b in at.button if b.key and b.key.startswith("alpha-graph-open-")]
    assert open_buttons
    open_buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)


def test_module_never_calls_a_registry_write_method_static_guard():
    forbidden = [
        r"\.insert_experiment\(", r"\.record_failure\(", r"\.record_lineage\(",
        r"\.apply_bundle\(", r"\.record_attempt", r"INSERT OR REPLACE",
    ]
    with open(alpha_graph.__file__, encoding="utf-8") as fh:
        text = fh.read()
    for pattern in forbidden:
        assert not re.search(pattern, text), f"alpha_graph.py must never call {pattern!r}"


def test_module_never_imports_network_or_llm_clients():
    with open(alpha_graph.__file__, encoding="utf-8") as fh:
        text = fh.read()
    for token in ("AnthropicClient", "databento", "requests."):
        assert token not in text


# ---------------------------------------------------------------------------
# Mechanism detail -- Event/Factor/Instrument evidence, Cross-Asset Synthesis
# ---------------------------------------------------------------------------


def _open_trend(at):
    trend_button = next(
        b for b in at.button
        if b.key and b.key.startswith("alpha-graph-open-") and b.key.endswith("TREND")
    )
    trend_button.click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


def test_mechanism_detail_shows_both_asset_domains_for_trend():
    at = _fresh_app()
    at = _open_trend(at)
    assert at.session_state["alpha_graph_selected_mechanism"] == "TREND"
    body = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "CL" in body and "XLE" in body
    assert "FUTURES" in body and "ETF" in body
    for heading in (
        "Where does this come from?", "Which Factors represent this Mechanism?",
        "Where has it been tested?", "Cross-Asset Synthesis",
    ):
        assert heading in body


def test_cross_asset_synthesis_never_transfers_a_verdict_in_rendered_text():
    at = _fresh_app()
    at = _open_trend(at)
    body = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    banned = re.compile(r"\bbuy\b|\bsell\b|\bshould trade\b|\bgo long\b|\bgo short\b", re.IGNORECASE)
    assert not banned.search(body)
    assert "never merged or transferred" in body


def test_detail_view_can_navigate_into_the_existing_factor_library_detail():
    """Clicking "Open evidence" on a researched instrument reuses the
    EXISTING Factor Library detail view (prompt section 7: click Mechanism/
    Factor/Instrument -> navigate to existing evidence detail) -- never a
    duplicated rendering."""
    at = _fresh_app()
    at = _open_trend(at)
    goto_buttons = [b for b in at.button if b.key and b.key.startswith("alpha-graph-goto-")]
    assert goto_buttons
    goto_buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    assert at.session_state["learn_landing_focus"] == "alpha_library"
    assert at.session_state["alpha_library_selected_id"]
    assert "alpha_graph_selected_mechanism" not in at.session_state


def test_back_button_returns_to_the_landing():
    at = _fresh_app()
    at = _open_trend(at)
    at.button(key="alpha-graph-back").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    assert "alpha_graph_selected_mechanism" not in at.session_state


def test_rendered_page_never_shows_an_alpha_score_or_confidence_percentage():
    at = _fresh_app()
    at = _open_trend(at)
    body = (
        " ".join(m.value for m in at.markdown)
        + " ".join(c.value for c in at.caption)
        + " ".join(str(m.value) for m in at.metric)
    )
    assert not re.search(r"alpha\s*score", body, re.IGNORECASE)
    assert not re.search(r"confidence\s*:\s*\d", body, re.IGNORECASE)
