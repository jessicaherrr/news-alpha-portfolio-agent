"""Agent/Research UX correctness pass -- focused regression tests for three
fixes, none of which touch scientific semantics, validation, Registry schema,
C++ execution, multiple testing, holdout rules, or paper eligibility:

1. ROOT CONSISTENCY -- a scripted/offline proposal always agrees with the
   selected Root, never the curated preset's own predeclared root
   (``llm_demo.rooted_hypothesis`` / ``rooted_plan``).
2. FAILURE MEMORY UX -- the Agent transcript shows a compact "PRIOR RESEARCH
   MEMORY" digest by default; the full evidence stays reachable in an
   expander, and the actual pre-proposal ``ResearchContext.failure_memory``
   is unchanged (presentation-only).
3. RESEARCH DETAILS DEFAULT SELECTION -- selection priority is explicit Agent
   hand-off > latest Agent/session result > most recent authoritative
   committed Registry experiment > a subtle plain-text empty state (never a
   large colored alert).

Every test here is offline/deterministic (scripted LLM or real local registry
reads), writes nothing to the registry, executes no C++, and makes no network
call.
"""
from __future__ import annotations

import json

import pytest
from alpha_agent.agents.llm import ScriptedLLMClient
from alpha_agent.agents.research_agent import ResearchAgent
from alpha_agent.ui import llm_demo, services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(),
    reason="Phase 14 registry sqlite not present in this checkout",
)

_ROOTS = ("CL", "ES", "GC", "NQ", "ZN")


# ---------------------------------------------------------------------------
# A / B -- Root=<root> + Offline mode produces a <root> hypothesis, never the
# preset's own predeclared root, for every supported root and every curated
# preset (including a Root/preset mismatch, e.g. the tsmom_nq preset with
# Root=CL from the bug report).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("root", _ROOTS)
def test_offline_proposal_and_compilation_always_match_the_selected_root(root):
    for sc in llm_demo.SCENARIOS:
        proposal, error, _fm = llm_demo.propose_hypothesis(
            mode=llm_demo.SCRIPTED_MODE, scenario_key=sc.key, universe=(root,),
        )
        assert error is None, f"{sc.key} -> Root={root}: {error}"
        assert proposal is not None and proposal.accepted, (
            f"{sc.key} -> Root={root}: "
            f"{proposal.rejection_code if proposal else 'no proposal'} "
            f"{proposal.rejection_detail if proposal else ''}"
        )
        assert proposal.hypothesis.universe == [root]

        compiled, cerror = llm_demo.compile_hypothesis(
            mode=llm_demo.SCRIPTED_MODE, scenario_key=sc.key,
            hypothesis=proposal.hypothesis, universe=services.approved_universe(),
        )
        assert cerror is None, f"{sc.key} -> Root={root}: {cerror}"
        assert compiled.accepted, compiled.rejection_detail
        assert compiled.root_symbol == root


def test_reported_bug_tsmom_nq_preset_with_root_cl_no_longer_fails():
    """The exact reported failure: Agent Root=CL, the (NQ-authored) offline
    tsmom preset selected -- must propose CL, not MARKET_OUTSIDE_UNIVERSE:
    ['NQ']."""
    proposal, error, _fm = llm_demo.propose_hypothesis(
        mode=llm_demo.SCRIPTED_MODE, scenario_key="tsmom_nq", universe=("CL",),
    )
    assert error is None
    assert proposal.accepted, proposal.rejection_detail if not proposal.accepted else None
    assert proposal.hypothesis.universe == ["CL"]
    assert "NQ" not in proposal.hypothesis.universe


def test_a_preset_mentioning_one_root_does_not_silently_force_it_when_root_differs():
    """Audit preset behaviour: the mean_reversion_cl preset's own hypothesis
    text/id/universe must not leak CL when Root=ES is selected."""
    proposal, error, _fm = llm_demo.propose_hypothesis(
        mode=llm_demo.SCRIPTED_MODE, scenario_key="mean_reversion_cl", universe=("ES",),
    )
    assert error is None
    assert proposal.accepted
    h = proposal.hypothesis
    assert h.universe == ["ES"]
    assert "CL" not in h.universe
    assert "CL" not in h.hypothesis_id.split("-")  # no stray CL token in the id


def test_rooted_hypothesis_is_a_no_op_when_already_on_the_target_root():
    sc = llm_demo.scenario("tsmom_nq")
    out = llm_demo.rooted_hypothesis(sc.hypothesis, "NQ")
    assert out == sc.hypothesis


def test_rooted_plan_rewrites_template_and_blueprint_root_symbol():
    template_plan = llm_demo.scenario("tsmom_nq").plan
    rewritten = llm_demo.rooted_plan(template_plan, "GC")
    assert rewritten["template"]["root_symbol"] == "GC"

    blueprint_plan = llm_demo.scenario("trend_vol_regime_nq_novel").plan
    rewritten_bp = llm_demo.rooted_plan(blueprint_plan, "ZN")
    assert rewritten_bp["blueprint"]["root_symbol"] == "ZN"


# ---------------------------------------------------------------------------
# C -- the MARKET_OUTSIDE_UNIVERSE guard remains intact: a hypothesis that
# genuinely declares a market outside the given universe is still rejected,
# never silently weakened.
# ---------------------------------------------------------------------------


def test_market_outside_universe_guard_still_rejects_a_real_mismatch():
    hypothesis = dict(llm_demo.scenario("tsmom_nq").hypothesis)
    hypothesis["universe"] = ["NQ"]  # deliberately left unrooted
    client = ScriptedLLMClient([json.dumps(hypothesis)])
    agent = ResearchAgent(client)
    context, _fm = llm_demo.build_context_for_objective(objective="x", universe=("CL",))
    proposal = agent.propose(context)
    assert not proposal.accepted
    assert proposal.rejection_code == "MARKET_OUTSIDE_UNIVERSE"
    assert "NQ" in proposal.rejection_detail


# ---------------------------------------------------------------------------
# D -- Failure Memory is still present in ResearchContext before proposal
# (the actual scientific input is unchanged -- only its UI presentation is).
# ---------------------------------------------------------------------------


def test_failure_memory_still_reaches_research_context_before_proposal():
    context, fm = llm_demo.build_context_for_objective(objective="anything", universe=("NQ",))
    assert fm  # the real registry has a known NQ tsmom history
    assert context.failure_memory
    assert any(
        d.strategy_family == "tsmom" and d.root_symbol == "NQ" for d in context.failure_memory
    )


# ---------------------------------------------------------------------------
# E / F -- Agent default view is compact; full memory stays reachable.
# ---------------------------------------------------------------------------


def _fresh_agent():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


def test_agent_default_view_shows_the_compact_digest_not_the_full_dump():
    at = _fresh_agent()
    at.selectbox(key="agent-scenario-select").select(
        "Multi-week time-series momentum in NQ"
    ).run(timeout=90)
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    markdown_text = " ".join(md.value for md in at.markdown)
    caption_text = " ".join(c.value for c in at.caption)
    assert "PRIOR RESEARCH MEMORY" in markdown_text
    assert "related research neighbourhood(s)" in caption_text
    assert "prior canonical failure(s)" in caption_text
    # the old always-open full-dump heading is gone
    assert "FAILURE MEMORY" not in markdown_text


def test_full_failure_memory_remains_reachable_in_an_expander():
    at = _fresh_agent()
    at.selectbox(key="agent-scenario-select").select(
        "Multi-week time-series momentum in NQ"
    ).run(timeout=90)
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    expander_labels = {e.label for e in at.expander}
    assert "View full Failure Memory" in expander_labels
    # the full per-(family,root) evidence table is still rendered somewhere
    # in the tree -- nothing was deleted from the pre-proposal evidence.
    assert at.dataframe  # at least one prior-experiments table present


# ---------------------------------------------------------------------------
# G / H -- Research Details default selection priority.
# ---------------------------------------------------------------------------


def test_latest_authoritative_experiment_returns_real_committed_evidence():
    latest = services.latest_authoritative_experiment()
    assert latest is not None
    detail = services.get_experiment(latest["experiment_id"])
    assert detail["result"] is not None


def test_research_opens_on_the_strategies_landing_by_default():
    """Product Consolidation campaign (Checkpoint D): with no explicit
    selection, Research now opens on the Strategies-first LANDING (task
    spec section 17: "Strategies answers: What have we studied?"), never an
    auto-picked Detail view for the latest registry experiment -- that
    auto-fallback is retired; only an explicit hand-off opens Detail."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.research import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)

    markdown_text = " ".join(md.value for md in at.markdown)
    assert "## Strategy Lab" in markdown_text  # the former Research page (url `lab`)
    tab_labels = {t.proto.label for t in at.tabs}
    # Phase 8 Research UI consolidation: Strategies/Experiments/Validation
    # are no longer top-level tabs -- they remain reachable via the
    # landing's own "More detail views" drill-down expander.
    # My Alpha / Research Map / Community moved to Learn (Research Thread workspace).
    assert {"Workflow", "Provenance"} <= tab_labels
    latest = services.latest_authoritative_experiment()
    # the latest experiment's raw id must not be auto-surfaced anywhere --
    # opening its Detail view is now an explicit action, not a default.
    assert latest["experiment_id"] not in markdown_text


def test_explicit_agent_handoff_overrides_the_registry_fallback():
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
    markdown_text = " ".join(md.value for md in at.markdown)
    assert "A brand-new untested idea." in markdown_text
    latest = services.latest_authoritative_experiment()
    # the explicit hand-off's (missing) experiment id must not be silently
    # replaced by the registry fallback's id.
    assert latest["experiment_id"] not in markdown_text


# ---------------------------------------------------------------------------
# I -- no large blue "Nothing to inspect yet" st.info banner remains: the
# Strategies-first landing itself IS the honest "nothing selected yet" state
# now, even against a completely empty registry.
# ---------------------------------------------------------------------------


def test_research_landing_renders_gracefully_against_an_empty_registry(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(services, "REGISTRY_PATH", tmp_path / "empty_research_details.sqlite")

    at = AppTest.from_string("from alpha_agent.ui.views.research import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    # no st.info/st.warning/st.error banner was used for the empty state --
    # the landing's own Research Map gaps table (still rendered by default
    # after the Phase 8 Research UI consolidation moved Strategies' own
    # "All Tested"/"Untested" table behind the "More detail views"
    # drill-down) honestly shows NO_EVIDENCE/UNDEREXPLORED coverage instead.
    assert not at.info
    tab_labels = {t.proto.label for t in at.tabs}
    # My Alpha / Research Map / Community moved to Learn (Research Thread workspace).
    assert {"Workflow", "Provenance"} <= tab_labels
    # The Research Map (now on Learn) honestly shows NO_EVIDENCE/UNDEREXPLORED
    # coverage against the same empty registry -- never a banner.
    learn_at = AppTest.from_string("from alpha_agent.ui.views.learn import render\nrender()\n")
    learn_at.session_state["learn_landing_focus"] = "alpha_graph"
    learn_at.run(timeout=90)
    assert not list(learn_at.exception), list(learn_at.exception)
    assert not learn_at.info
    dataframe_text = " ".join(
        df.value.to_string() if hasattr(df.value, "to_string") else str(df.value) for df in learn_at.dataframe
    )
    assert "NO_EVIDENCE" in dataframe_text or "UNDEREXPLORED" in dataframe_text


# ---------------------------------------------------------------------------
# J / K / L -- no Registry writes, no C++ execution, no network calls.
# ---------------------------------------------------------------------------


def test_root_and_research_default_flows_never_write_the_registry():
    before = services.registry_summary()["content_digest"]
    for root in _ROOTS:
        llm_demo.propose_hypothesis(
            mode=llm_demo.SCRIPTED_MODE, scenario_key="tsmom_nq", universe=(root,),
        )
    services.latest_authoritative_experiment()
    after = services.registry_summary()["content_digest"]
    assert before == after


def test_offline_root_consistency_fix_performs_zero_anthropic_calls(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "anthropic", None)  # importing it would now raise
    for root in _ROOTS:
        proposal, error, _fm = llm_demo.propose_hypothesis(
            mode=llm_demo.SCRIPTED_MODE, scenario_key="tsmom_nq", universe=(root,),
        )
        assert error is None
        assert proposal.accepted
