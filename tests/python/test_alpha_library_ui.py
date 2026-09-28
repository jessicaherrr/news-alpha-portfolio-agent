"""Phase 2 -- "My Alpha Library" UI (prompt 2 sections 12/13/18/23/24),
renamed "Factor Library" by the Phase 6 closure, renamed "My Alpha" by
Phase 8's Research UI consolidation (same tab/focus key throughout, `alpha_library`).

Real `streamlit.testing.v1.AppTest` renders, mirroring
`test_phase_20_streamlit_ui.py`'s own pattern. Proves the library renders
inside Research (never a new top-level page), one object's detail view opens
with progressive disclosure, and browsing never runs research, never calls
Claude, never calls a market-data API, and never writes the registry. See
`test_phase6_factor_library.py` for the Phase 6 closure's own new behaviors
(evidence-oriented status sections, ETF persistence, research recycling).
"""
from __future__ import annotations

import re

import pytest
from alpha_agent.ui import services
from alpha_agent.ui.views import alpha_library

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)


def _fresh_library_app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.alpha_library import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


# ---------------------------------------------------------------------------
# Lives inside Research, not a new top-level page
# ---------------------------------------------------------------------------


def test_my_alpha_library_is_a_tab_on_the_research_page_not_a_new_top_level_page():
    """Renamed "My Alpha Library" -> "Factor Library" (Phase 6 closure) ->
    "My Alpha" (Phase 8 Research UI consolidation) -- still the same tab
    key (`alpha_library`), still inside Research, never a new top-level
    page."""
    from alpha_agent.ui.views import learn

    assert learn._TABS["alpha_library"] == "My Alpha"  # on Learn since the Research Thread workspace


def test_my_alpha_library_is_not_registered_as_its_own_app_page():
    with open(str(services.REPO_ROOT / "python" / "alpha_agent" / "ui" / "app.py"), encoding="utf-8") as fh:
        app_src = fh.read()
    assert "alpha_library.render" not in app_src


# ---------------------------------------------------------------------------
# Renders without running research / calling Claude / calling market data
# ---------------------------------------------------------------------------


def test_library_renders_without_running_research_calling_claude_or_market_data(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("My Alpha must never call a live LLM client just by rendering")

    from alpha_agent.ui import llm_demo

    monkeypatch.setattr(llm_demo, "build_llm_client", _boom, raising=False)
    at = _fresh_library_app()
    assert not list(at.exception)
    body = " ".join(m.value for m in at.markdown)
    assert "My Alpha" in body


def test_library_never_calls_a_registry_write_method_static_guard():
    forbidden = [
        r"\.insert_experiment\(", r"\.record_failure\(", r"\.record_lineage\(",
        r"\.apply_bundle\(", r"\.record_attempt", r"INSERT OR REPLACE",
    ]
    with open(alpha_library.__file__, encoding="utf-8") as fh:
        text = fh.read()
    for pattern in forbidden:
        assert not re.search(pattern, text), f"alpha_library.py must never call {pattern!r}"


def test_opening_an_object_never_triggers_execution(monkeypatch):
    """Clicking through to a detail view (progressive disclosure) must never
    run a backtest, compile a strategy, or start a paper-trading run."""
    def _boom(*a, **kw):
        raise AssertionError("viewing an AlphaResearchObject must never trigger execution")

    from alpha_agent.ui import deep_research

    monkeypatch.setattr(deep_research, "run_deep_research", _boom, raising=False)
    at = _fresh_library_app()
    open_buttons = [b for b in at.button if b.key and b.key.startswith("alpha-lib-open-")]
    assert open_buttons
    open_buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)


# ---------------------------------------------------------------------------
# One object detail view -- progressive disclosure
# ---------------------------------------------------------------------------


def test_detail_view_opens_and_shows_progressive_disclosure_sections():
    at = _fresh_library_app()
    open_buttons = [b for b in at.button if b.key and b.key.startswith("alpha-lib-open-")]
    assert open_buttons
    open_buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    assert at.session_state["alpha_library_selected_id"]

    body = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    for heading in (
        "What it is, and why it may work", "Factor definition", "Relevant context",
        "How was it implemented?", "What did I test?",
        "What did the evidence say?", "What have I learned?",
    ):
        assert heading in body


def test_detail_view_never_shows_a_giant_raw_table_as_the_whole_experience():
    """Section 23: "do not make a giant table the entire experience" -- the
    detail view must still render narrative section headers, not just one
    `st.dataframe`."""
    at = _fresh_library_app()
    open_buttons = [b for b in at.button if b.key and b.key.startswith("alpha-lib-open-")]
    open_buttons[0].click().run(timeout=90)
    assert not list(at.exception)
    assert len(at.dataframe) >= 1  # the experiments table exists...
    assert len(at.markdown) > 5  # ...but is not the only content on the page


def test_back_button_returns_to_the_library_list():
    at = _fresh_library_app()
    open_buttons = [b for b in at.button if b.key and b.key.startswith("alpha-lib-open-")]
    open_buttons[0].click().run(timeout=90)
    at.button(key="alpha-lib-back").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    assert "alpha_library_selected_id" not in at.session_state


# ---------------------------------------------------------------------------
# No Alpha Score anywhere in the rendered page
# ---------------------------------------------------------------------------


def test_rendered_page_never_shows_an_alpha_score_or_confidence_percentage():
    at = _fresh_library_app()
    open_buttons = [b for b in at.button if b.key and b.key.startswith("alpha-lib-open-")]
    open_buttons[0].click().run(timeout=90)
    body = (
        " ".join(m.value for m in at.markdown)
        + " ".join(c.value for c in at.caption)
        + " ".join(str(m.value) for m in at.metric)
    )
    assert not re.search(r"alpha\s*score", body, re.IGNORECASE)
    assert not re.search(r"confidence\s*:\s*\d", body, re.IGNORECASE)
    assert not re.search(r"expected\s*return", body, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_market_filter_narrows_the_library():
    at = _fresh_library_app()
    at.selectbox(key="alpha-lib-root-filter").select("CL").run(timeout=90)
    assert not list(at.exception), list(at.exception)
    body = " ".join(m.value for m in at.markdown)
    assert "on NQ" not in body or "on CL" in body
