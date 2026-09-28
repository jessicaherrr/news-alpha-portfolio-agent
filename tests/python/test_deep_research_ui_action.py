"""``alpha_agent.ui.deep_research`` -- the one UI boundary allowed to trigger a
real, registry-writing `ResearchOrchestrator` run (Agent runtime-integration
release, sections 9/10).

Fast tests cover the honest-failure paths (no network, no registry touched).
The one slow test proves the actual UI-facing wrapper function -- not just
`orchestrator_factory` directly -- runs the real end-to-end path, using an
ISOLATED registry (monkeypatched `services.REGISTRY_PATH`) so the PRODUCTION
registry is never touched by this test.
"""
from __future__ import annotations

import pytest
from alpha_agent.registry.enums import AttemptStatus
from alpha_agent.ui import deep_research, llm_demo, services


def test_scripted_mode_without_scenario_is_an_honest_error_not_a_run():
    outcome = deep_research.run_deep_research(
        mode=llm_demo.SCRIPTED_MODE, objective="anything", root="NQ", family_stem="test",
    )
    assert not outcome.accepted
    assert "scenario" in outcome.error.lower()
    assert outcome.manifest is None and outcome.report is None


def test_live_mode_missing_key_is_an_honest_error_never_a_silent_fallback(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    outcome = deep_research.run_deep_research(
        mode=llm_demo.LIVE_MODE, objective="anything", root="NQ", family_stem="test",
    )
    assert not outcome.accepted
    assert "ANTHROPIC_API_KEY" not in (outcome.error or "")  # never echoes the variable's value
    assert outcome.report is None


@pytest.mark.slow
def test_run_deep_research_executes_for_real_against_an_isolated_registry(tmp_path, monkeypatch):
    """The actual UI-facing wrapper, exercised end to end: real planning,
    real C++ backtest execution via `ProductionExecutionValidationService`,
    real frozen validation, and a real write -- but to an ISOLATED registry
    path only, never `data/registry/experiments.sqlite`."""
    isolated_path = tmp_path / "isolated_ui_test_registry.sqlite"
    monkeypatch.setattr(services, "REGISTRY_PATH", isolated_path)

    outcome = deep_research.run_deep_research(
        mode=llm_demo.SCRIPTED_MODE, objective="Test the Deep Research UI action end to end.",
        root="NQ", family_stem="deep-research-ui-test", scenario_key="tsmom_nq",
    )

    assert outcome.accepted, outcome.error
    assert outcome.report is not None
    assert outcome.report.predeclared_family_size == 1
    member_result = outcome.report.member_results[0]
    assert member_result.attempt_status is AttemptStatus.VALID
    assert member_result.final_verdict is not None  # a real, adjudicated verdict was written

    # the isolated registry -- and ONLY the isolated one -- now has this row.
    with services.open_registry() as reg:
        view = reg.get(member_result.experiment_identity)
    assert view.result is not None
