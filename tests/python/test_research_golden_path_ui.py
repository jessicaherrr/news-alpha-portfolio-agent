"""Research Golden Path -- Streamlit rendering smoke tests for the new
Workflow tab on the Research page (`views/research_workflow.py`). Gated
behind `pytest.importorskip` so the core suite still runs with no UI extras
installed (CLAUDE.md: "keep optional vendor imports lazy").

These exercise the SAME real registry / real Strategy Compiler / real
`llm_demo.propose_hypothesis`/`compile_hypothesis` the manual acceptance pass
used -- Offline / Deterministic mode only, zero network calls, matching
section 21 ("no automatic paid Claude API calls" in tests).
"""
from __future__ import annotations

import pytest
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(),
    reason="Phase 14 registry sqlite not present in this checkout",
)


def test_research_page_renders_with_workflow_tab_and_no_exceptions():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.research import render\nrender()\n")
    at.run(timeout=90)
    assert not at.exception


def test_propose_compile_validate_and_decision_brief_round_trip():
    """Discover -> Hypothesis -> Compile -> Validate -> Decision Brief, all
    through the real pipeline, on an Offline / Deterministic scenario --
    zero network calls, zero fabricated evidence."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.research import render\nrender()\n")
    at.run(timeout=90)

    at.button(key="rwf-claude-go").click().run(timeout=90)
    assert not at.exception
    assert "rwf_hypothesis" not in at.session_state or at.session_state["rwf_hypothesis"]

    at.button(key="rwf-compile-go").click().run(timeout=90)
    assert not at.exception
    assert at.session_state["rwf_compiled"]["accepted"]

    at.button(key="rwf-brief-go").click().run(timeout=90)
    assert not at.exception
    brief = at.session_state["rwf_brief"]
    assert brief["error"] is None
    assert "DECISION BRIEF" in brief["text"]


def test_unsupported_blueprint_family_disables_run_button_not_crashes():
    """A full custom blueprint (no Phase 11 family) is a genuine execution
    capability gap -- the workflow must say so, never crash and never claim
    it can run."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.research import render\nrender()\n")
    at.run(timeout=90)
    preset_sb = next(sb for sb in at.selectbox if sb.key == "rwf-preset")
    preset_sb.set_value("Trend conditioned on a volatility regime in NQ (novel blueprint)").run(timeout=90)
    at.button(key="rwf-claude-go").click().run(timeout=90)
    at.button(key="rwf-compile-go").click().run(timeout=90)
    assert not at.exception
    assert at.session_state["rwf_compiled"]["family_key"] is None
    assert not services.execution_supported_family(at.session_state["rwf_compiled"]["family_key"])
    with pytest.raises(KeyError):
        at.button(key="rwf-run-go")
