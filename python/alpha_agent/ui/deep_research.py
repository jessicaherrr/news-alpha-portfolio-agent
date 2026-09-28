"""The ONE UI boundary that may trigger a REAL, registry-writing
``ResearchOrchestrator`` run (Agent runtime-integration release, sections
9/10).

Every other UI module (``services.py``, ``llm_demo.py``, ``market_context.py``)
is read-only or LLM-only -- ``services.py`` is statically grep-tested to never
call a registry write method. This module is different ON PURPOSE:
"Run Deep Research" is an explicit, user-triggered action that may execute a
genuinely new scientific experiment through the C++ Quant Core and the frozen
validation pipeline and write it to the registry -- but the write itself
still only ever happens inside ``ResearchOrchestrator.finalize_family``
(``ExperimentRegistry.apply_bundle``), never from code in this module. This
module only opens a registry handle and drives the orchestrator's own public
plan/execute/finalize sequence -- the same sequence
``tests/python/test_agent_feedback_loop_acceptance.py`` exercises against an
isolated registry.

This is PROPOSE/INSPECT's opposite number (see ``llm_demo.py`` /
``views/research.py``): that path never executes anything or writes to the
registry; this path always does, deliberately, once per explicit click.

Two callers use this SAME function today (Research Golden Path campaign):
``views/agent.py``'s "Run This Hypothesis" and ``views/research_workflow.py``'s
"Run Historical Test" (the Research page's own Workflow tab). Both are thin
UI glue around this one orchestration boundary -- neither duplicates it.
"""
from __future__ import annotations

from dataclasses import dataclass

from alpha_agent.agents.llm import LLMClientUnavailable
from alpha_agent.agents.orchestrator import FamilyManifest, FamilyReport
from alpha_agent.agents.orchestrator_factory import build_deep_research_orchestrator
from alpha_agent.ui import llm_demo, services

__all__ = ["DeepResearchOutcome", "run_deep_research"]


@dataclass(frozen=True)
class DeepResearchOutcome:
    """Typed outcome of one "Run Deep Research" click. Never a bare exception
    or a raw stack trace to the UI -- and never silently equivalent to a
    normal chat message (CLAUDE.md section 9: "Never make a normal chat
    message unexpectedly start a long scientific run")."""

    accepted: bool
    error: str | None = None
    manifest: FamilyManifest | None = None
    report: FamilyReport | None = None


def run_deep_research(
    *,
    mode: str,
    objective: str,
    root: str,
    family_stem: str,
    scenario_key: str | None = None,
    target_family_size: int = 1,
) -> DeepResearchOutcome:
    """Runs ONE real predeclared family end to end, against the PRODUCTION
    registry: PLAN (ResearchAgent -> StrategyCompilerAgent -> full scientific
    ``experiment_identity`` -> dedup/novelty check) -> EXECUTE (the real
    ``ProductionExecutionValidationService`` -- the real C++ backtest and the
    real frozen ``ReliabilityPolicy``) -> FINALIZE (family-wide BH/FDR,
    written to the registry).

    ``mode`` is the SAME Offline / Deterministic vs. Claude Research engine
    choice the rest of the app uses (``llm_demo.build_llm_client``) -- no
    second LLM integration. In Offline / Deterministic mode, ``scenario_key``
    selects which curated, schema-valid canned proposal is REALLY compiled
    and REALLY executed (the offline default always means "replay a canned
    response", whether for inspection or for a real run); it is required in
    that mode and ignored in Claude Research mode, where the model proposes
    freely from ``objective``.

    Bounded by ``alpha_agent.agents.orchestrator_factory.DEFAULT_DEEP_RESEARCH_BUDGET``
    (section 12: a professional autonomous researcher has bounded authority) --
    never overridden from the UI in this release.
    """
    if mode == llm_demo.SCRIPTED_MODE and not scenario_key:
        return DeepResearchOutcome(
            accepted=False,
            error="Offline / Deterministic Deep Research needs a selected scenario to replay.",
        )
    sc = llm_demo.scenario(scenario_key) if scenario_key else None

    # `root` here is always explicit (the family runs against exactly one
    # market -- see `build_deep_research_orchestrator`'s `market_universe=
    # (root,)`), so a scripted scenario's own predeclared hypothesis/plan is
    # ALWAYS re-rooted to it before replay -- Root is the authoritative
    # research target, never the curated preset's own predeclared root (see
    # `llm_demo.rooted_hypothesis`/`rooted_plan`). This is the same fix
    # `llm_demo.propose_hypothesis`/`compile_hypothesis` apply for the
    # proposal-only path; "Run This Hypothesis" replays the identical
    # scenario_key and must agree with it, not re-diverge to the preset's
    # own original root.
    try:
        research_client = llm_demo.build_llm_client(
            mode, hypothesis_json=llm_demo.rooted_hypothesis(sc.hypothesis, root) if sc else None
        )
        compiler_client = llm_demo.build_llm_client(
            mode, plan_json=llm_demo.rooted_plan(sc.plan, root) if sc else None
        )
    except LLMClientUnavailable as exc:
        return DeepResearchOutcome(accepted=False, error=f"Live LLM unavailable: {exc}")

    try:
        with services.open_registry() as reg:
            orch = build_deep_research_orchestrator(
                registry=reg,
                objective=objective,
                root=root,
                family_stem=family_stem,
                research_client=research_client,
                compiler_client=compiler_client,
                target_family_size=target_family_size,
            )
            manifest = orch.plan_family()
            if not manifest.planning_complete:
                dispositions = [o.disposition.value for o in manifest.planning_log]
                return DeepResearchOutcome(
                    accepted=False,
                    manifest=manifest,
                    error=(
                        "Planning could not fill the predeclared family: every candidate "
                        "was blocked (duplicate, near-duplicate with no stated novelty, "
                        "or a guardrail rejection). Dispositions: " + ", ".join(dispositions)
                    ),
                )
            evidence = orch.execute_family(manifest)
            report = orch.finalize_family(manifest, evidence)
            return DeepResearchOutcome(accepted=True, manifest=manifest, report=report)
    except Exception as exc:  # noqa: BLE001 -- an honest UI error, never a raw stack trace
        return DeepResearchOutcome(accepted=False, error=f"{type(exc).__name__}: {exc}")
