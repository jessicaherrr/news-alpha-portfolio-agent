"""Product-surface refactor -- Agent as the primary research workspace,
Research as a details-only viewer, System owning runtime status.

Every test here is either offline/deterministic (scripted LLM, real local
registry reads, zero network) or -- for the one test that actually executes
"Run This Hypothesis" (`test_G_*`) -- runs against an ISOLATED temporary
registry (`monkeypatch.setattr(services, "REGISTRY_PATH", ...)`), following
the exact same established pattern as
`test_deep_research_ui_action.py::test_run_deep_research_executes_for_real_against_an_isolated_registry`.
The PRODUCTION registry (`data/registry/experiments.sqlite`) is never written
to by anything in this file.
"""
from __future__ import annotations

import pytest
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(),
    reason="Phase 14 registry sqlite not present in this checkout",
)


def _fresh_agent():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


# ---------------------------------------------------------------------------
# B -- Agent shows the primary-workspace controls
# ---------------------------------------------------------------------------


def test_B_agent_shows_engine_objective_root_context_and_primary_action():
    at = _fresh_agent()
    labels = {r.label for r in at.radio}
    assert "Research Engine" in labels
    assert any(ta.label == "Objective" for ta in at.text_area)
    assert any(sb.label == "Research Root" for sb in at.selectbox)
    markdown_text = " ".join(md.value for md in at.markdown)
    assert "Research Context" in markdown_text
    assert at.button(key="agent-send").label == "Generate Hypothesis"


# ---------------------------------------------------------------------------
# C -- Generate Hypothesis is proposal-only: no C++ execution, no registry write
# ---------------------------------------------------------------------------


def test_C_generate_hypothesis_never_writes_the_registry():
    before = services.registry_summary()["content_digest"]
    at = _fresh_agent()
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    after = services.registry_summary()["content_digest"]
    assert before == after, "Generate Hypothesis (proposal-only) must never write the registry"
    # and no execution outcome exists yet -- only an explicit Confirm Run can create one
    assert "agent_run_outcome" not in at.session_state or at.session_state["agent_run_outcome"] is None


# ---------------------------------------------------------------------------
# D -- generated proposal displays hypothesis, compilation, Run This Hypothesis
# ---------------------------------------------------------------------------


def test_D_generated_proposal_shows_hypothesis_compilation_and_run_action():
    at = _fresh_agent()
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    markdown_text = " ".join(md.value for md in at.markdown)
    assert "HYPOTHESIS" in markdown_text
    assert "Reasoning summary" in markdown_text
    assert "COMPILATION" in markdown_text
    assert "Research Actions" in markdown_text
    assert at.button(key="agent-run-hypothesis") is not None


# ---------------------------------------------------------------------------
# E -- execution root/family are derived from the compiled proposal; no
# independent normal-flow editable root/family re-entry exists
# ---------------------------------------------------------------------------


def test_E_no_independent_execution_root_or_family_controls_exist():
    at = _fresh_agent()
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    # The old Research-page pattern ("Root for this Deep Research run" +
    # free-text "Family label") must not exist anywhere on Agent.
    select_labels = {sb.label for sb in at.selectbox}
    text_input_labels = {ti.label for ti in at.text_input}
    assert "Root for this Deep Research run" not in select_labels
    assert "Family label" not in text_input_labels
    assert select_labels == {"Research Root", "Preset research objective"}

    compiled = at.session_state["lab_compiled"]
    markdown_text = " ".join(md.value for md in at.markdown)
    # The Execution Target metric row shows the SAME root/family the
    # compiled StrategySpec carries -- never a separately editable value.
    assert compiled["root_symbol"] in markdown_text or any(
        m.value == compiled["root_symbol"] for m in at.metric
    )
    assert any(m.value == compiled["root_symbol"] for m in at.metric)
    from alpha_agent.ui import services

    strategy_label = services.strategy_name(compiled["family_key"]) if compiled["family_key"] else "blueprint"
    assert any(m.value == strategy_label + " / " + compiled["build_mode"] for m in at.metric)


# ---------------------------------------------------------------------------
# F -- Run This Hypothesis requires a second explicit confirmation
# ---------------------------------------------------------------------------


def test_F_run_this_hypothesis_requires_a_second_confirmation():
    at = _fresh_agent()
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    assert at.session_state["agent-run-confirm-pending"] is False

    at.button(key="agent-run-hypothesis").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    assert at.session_state["agent-run-confirm-pending"] is True
    assert at.button(key="agent-run-cancel") is not None
    assert at.button(key="agent-run-confirm-button") is not None
    # clicking "Run This Hypothesis" alone must NEVER have executed anything
    assert "agent_run_outcome" not in at.session_state or at.session_state["agent_run_outcome"] is None

    # Cancel resets the pending flag and still executes nothing.
    at.button(key="agent-run-cancel").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    assert at.session_state["agent-run-confirm-pending"] is False
    assert "agent_run_outcome" not in at.session_state or at.session_state["agent_run_outcome"] is None


# ---------------------------------------------------------------------------
# G -- the Agent result summary never equates VALID execution with PASS
# (the ONE test in this file that actually executes -- against an ISOLATED
# temporary registry only, exactly like
# test_deep_research_ui_action.py::test_run_deep_research_executes_for_real_against_an_isolated_registry)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_G_confirm_run_executes_against_an_isolated_registry_and_never_shows_green_for_reject(
    tmp_path, monkeypatch
):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    isolated_path = tmp_path / "isolated_agent_product_refactor.sqlite"
    monkeypatch.setattr(services, "REGISTRY_PATH", isolated_path)

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    at.button(key="agent-run-hypothesis").click().run(timeout=90)
    at.button(key="agent-run-confirm-button").click().run(timeout=300)
    assert not list(at.exception), list(at.exception)

    outcome = at.session_state["agent_run_outcome"]
    assert outcome.accepted, outcome.error
    assert outcome.report is not None
    member = outcome.report.member_results[0]

    markdown_text = " ".join(md.value for md in at.markdown)
    assert "Research Result" in markdown_text
    # Whatever the real committed verdict turns out to be for this isolated,
    # freshly-executed NQ TSMOM run, "Execution: VALID" must never be
    # rendered together with a bare, unqualified claim of scientific success:
    # a REJECT/INCONCLUSIVE verdict must never show as a green PASS badge.
    if member.final_verdict is not None and member.final_verdict.value != "PASS":
        assert "✓ PASS" not in markdown_text

    # and the isolated registry -- never production -- now has the row.
    with services.open_registry() as reg:
        view = reg.get(member.experiment_identity)
    assert view.result is not None


# ---------------------------------------------------------------------------
# H -- View Research Details resolves to the correct experiment/result
# ---------------------------------------------------------------------------


def test_H_view_research_details_hands_off_the_correct_selection():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    ag = _fresh_agent()
    ag.button(key="agent-send").click().run(timeout=90)
    assert not list(ag.exception), list(ag.exception)

    view_buttons = [b for b in ag.button if b.key and b.key.startswith("agent-view-details-")]
    assert view_buttons, "expected at least one 'View Research Details' button after a compiled proposal"
    view_buttons[0].click().run(timeout=90)
    assert not list(ag.exception), list(ag.exception)

    target = ag.session_state["research_details_target"]
    assert target["compiled"]["strategy_fingerprint"] == ag.session_state["lab_compiled"]["strategy_fingerprint"]
    assert target["root"] == ag.session_state["agent_last_root"]

    rd = AppTest.from_string("from alpha_agent.ui.views.research import render\nrender()\n")
    rd.session_state["research_details_target"] = target
    rd.run(timeout=90)
    assert not list(rd.exception), list(rd.exception)
    markdown_text = " ".join(md.value for md in rd.markdown)
    assert "Research Details" in markdown_text
    assert target["hypothesis"]["title"] in markdown_text


# ---------------------------------------------------------------------------
# I -- Research still never duplicates Agent's own primary-workspace controls
# ---------------------------------------------------------------------------


def test_I_research_page_execution_is_confined_to_the_golden_path_workflow_tab():
    """Research Golden Path campaign (`views/research.py`'s own module
    docstring): Research is no longer execution-free -- its new Workflow tab
    is a SECOND, explicit, user-triggered execution surface that reuses the
    SAME `deep_research.run_deep_research` boundary Agent's "Run This
    Hypothesis" uses, never a duplicated orchestration path. This test's
    original intent survives the campaign: Research still never presents
    Agent's own PRIMARY-workspace controls verbatim (its literal "Generate
    Hypothesis" / "Run Deep Research" button labels), and gains no engine
    selector beyond the two the Workflow tab itself introduces -- "Research
    Engine" (Discover's Claude-proposed mechanism) and "Brief Engine" (the
    Decision Brief) -- so a stray third control (e.g. a leftover "LLM mode"
    radio) would still be caught."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.research import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)

    caption_text = " ".join(str(c.value) for c in at.caption)
    assert "Test one strategy hypothesis end to end" in caption_text  # Strategy Lab's own subtitle, actually rendered
    button_labels = {b.label for b in at.button}
    assert "Generate Hypothesis" not in button_labels
    assert "Run Deep Research" not in button_labels
    radio_labels = {r.label for r in at.radio}
    assert radio_labels == {"Research Engine", "Brief Engine"}, radio_labels


# ---------------------------------------------------------------------------
# J -- Research tabs render for REJECT / INCONCLUSIVE / missing evidence
# ---------------------------------------------------------------------------


def _real_reject_experiment_id() -> str:
    rows = services.list_experiments(strategy_family="tsmom", root_symbol="NQ", trial_role=None)
    canonical = next(r for r in rows if r["trial_role"] == "CANONICAL")
    assert canonical["verdict"] == "REJECT"
    return canonical["experiment_id"]


def test_J_research_details_renders_a_real_committed_reject_experiment():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    exp_id = _real_reject_experiment_id()
    at = AppTest.from_string("from alpha_agent.ui.views.research import render\nrender()\n")
    at.session_state["research_details_target"] = {
        "source": "registry_lookup", "objective": None, "root": None, "hypothesis": None,
        "compiled": None, "evidence": None, "run_outcome": None, "experiment_id": exp_id,
    }
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    joined = "\n".join(m.value for m in at.markdown)
    assert "✓ PASS" not in joined


def test_J_research_details_handles_a_selection_with_no_committed_evidence_gracefully():
    """A proposal-only selection (no compiled/evidence yet) must render every
    tab without exception, never fabricating backtest/validation content."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.research import render\nrender()\n")
    at.session_state["research_details_target"] = {
        "source": "agent", "objective": "A brand-new untested idea.", "root": "NQ",
        "hypothesis": None, "compiled": None, "evidence": None, "run_outcome": None,
        "experiment_id": None,
    }
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    joined = "\n".join(m.value for m in at.markdown)
    assert "✓ PASS" not in joined


# ---------------------------------------------------------------------------
# L -- Agent Status / Recent Activity live under System, not Research
# ---------------------------------------------------------------------------


def test_L_agent_runtime_status_and_recent_activity_moved_to_system():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    sysapp = AppTest.from_string("from alpha_agent.ui.views.system import render\nrender()\n")
    sysapp.run(timeout=90)
    assert not list(sysapp.exception), list(sysapp.exception)
    sys_markdown = " ".join(m.value for m in sysapp.markdown)
    assert "Agent Runtime" in sys_markdown
    assert "Recent Activity" in sys_markdown

    research_app = AppTest.from_string("from alpha_agent.ui.views.research import render\nrender()\n")
    research_app.run(timeout=90)
    assert not list(research_app.exception), list(research_app.exception)
    research_markdown = " ".join(m.value for m in research_app.markdown)
    assert "Agent Runtime" not in research_markdown
    assert "Recent Activity" not in research_markdown
