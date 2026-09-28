"""True Inline Research Workspace (Release UX product-consolidation
acceptance pass, task spec section 3/24): the Research Workspace renders as
ONE `type="research_workspace"` conversation-timeline artifact, appearing
immediately once a research thread starts (even before any proposal
exists), updated in place as the workflow progresses -- never a separate
panel pinned below the conversation, and never duplicated across multiple
research-pipeline stages.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)


def _fresh_app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


def _seed_opportunity(at) -> str:
    """Seeds one real-shaped Opportunity snapshot directly into session
    state -- mirrors `test_agent_opportunity_cards.py`'s own approach, but
    without a live/mocked Databento round trip, since this file only cares
    about the Research Workspace hand-off, not the opportunity data itself.

    `observed_at` is deliberately a 2024 date, not "today" -- this
    snapshot's timestamp is embedded verbatim into the seeded Objective text
    (`_start_inline_research_from_opportunity`), which a test that goes on
    to actually call Generate Hypothesis sends through the real
    `ResearchAgent` pipeline; that pipeline's holdout guard fails loudly on
    ANY date token >= 2025-01-01 appearing in research-context input text
    (CLAUDE.md's 2025 holdout rule), regardless of why the date is there."""
    from alpha_agent.opportunity.schemas import OpportunitySnapshot, OpportunityState

    snap = OpportunitySnapshot(
        product="NQ", display_name="Nasdaq 100", setup="test setup",
        why_now=("reason one",), risks=("risk one",), catalyst="test catalyst",
        strengthen_if=("x",), invalidate_if=("y",),
        opportunity_state=OpportunityState.INVESTIGATE, next_action="do X",
        observed_at=datetime(2024, 9, 20, 12, 0, 0, tzinfo=UTC),
        evidence_refs=(),
    )
    at.session_state["agent_opportunities_snapshot"] = {
        "opportunities": [snap], "observed_at": datetime(2024, 9, 20, 12, 0, 0, tzinfo=UTC),
    }
    at.run(timeout=90)
    return "NQ"


def test_research_this_creates_the_workspace_turn_immediately():
    at = _fresh_app()
    product = _seed_opportunity(at)
    research_this = next(b for b in at.button if b.key and b.key.endswith("-research-this"))
    research_this.click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    transcript = at.session_state["agent_transcript"]
    workspace_turns = [t for t in transcript if t["type"] == "research_workspace"]
    assert len(workspace_turns) == 1, "expected exactly one research_workspace turn"

    # nothing was auto-executed -- proposal-only, explicit-action-only
    assert "ra_proposal" not in at.session_state or not at.session_state["ra_proposal"]
    assert "lab_compiled" not in at.session_state or not at.session_state["lab_compiled"]
    assert "agent_run_outcome" not in at.session_state or at.session_state["agent_run_outcome"] is None

    origin = at.session_state["agent_research_origin"]
    assert origin["source"] == "opportunity"
    assert origin["product"] == product

    expanders = [e for e in at.expander if e.label == "Research Workspace"]
    assert expanders, "expected the inline Research Workspace expander, not the advanced fallback"
    assert not [e for e in at.expander if e.label == "Propose a hypothesis directly (advanced)"]


def test_generate_hypothesis_updates_the_same_workspace_turn_in_place():
    at = _fresh_app()
    _seed_opportunity(at)
    research_this = next(b for b in at.button if b.key and b.key.endswith("-research-this"))
    research_this.click().run(timeout=90)

    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    transcript = at.session_state["agent_transcript"]
    workspace_turns = [t for t in transcript if t["type"] == "research_workspace"]
    assert len(workspace_turns) == 1, "the SAME workflow must update ONE artifact, never create a duplicate"

    types = [t["type"] for t in transcript]
    assert "hypothesis" in types
    # the workspace marker stays BEFORE the pipeline-stage cards it produced
    assert types.index("research_workspace") < types.index("hypothesis")

    assert at.session_state["ra_proposal"]
    assert at.session_state["lab_compiled"]
    # Run This Hypothesis was never auto-clicked -- still an explicit action
    assert "agent_run_outcome" not in at.session_state or at.session_state["agent_run_outcome"] is None


def test_workspace_never_auto_executes_run_this_hypothesis():
    at = _fresh_app()
    _seed_opportunity(at)
    at.button(key=next(b.key for b in at.button if b.key and b.key.endswith("-research-this"))).click().run(timeout=90)
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception)
    # A confirmation gate must exist and be UNCONFIRMED -- Run This
    # Hypothesis is a two-step explicit action, never triggered by proposing.
    # `AppTest`'s session_state proxy does not support the plain dict `.get`
    # method (it treats any missing attribute as a state-key lookup), hence
    # the explicit `in` check rather than `.get(..., False)`.
    assert "agent-run-confirm-pending" not in at.session_state or at.session_state["agent-run-confirm-pending"] is False
    assert "agent_run_outcome" not in at.session_state or at.session_state["agent_run_outcome"] is None
