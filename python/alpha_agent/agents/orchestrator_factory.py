"""Wiring for a real, product ``ResearchOrchestrator`` (Agent runtime-
integration release, sections 6/7/9).

This is the ONE place that assembles every real component
(``ExperimentRegistry`` handle, ``ResearchAgent``, ``StrategyCompilerAgent``,
``ProductionExecutionValidationService``, ``ReliabilityPolicy``,
``IdentityPlanes``, ``OrchestratorConfig``) into a runnable
``ResearchOrchestrator``, so a caller (the "Run Deep Research" UI action, or a
test) only supplies what actually varies: the objective, the target root, the
LLM transport(s), and how big a family to plan. The UI layer never builds a
``ResearchOrchestrator`` by hand -- this keeps there being exactly one place
the pieces are wired together.
"""
from __future__ import annotations

from pathlib import Path

from alpha_agent.agents.compiler_agent import StrategyCompilerAgent
from alpha_agent.agents.execution_service import ProductionExecutionValidationService
from alpha_agent.agents.llm import LLMClient
from alpha_agent.agents.orchestrator import (
    ExecutionValidationService,
    FamilyPlanSpec,
    OrchestratorBudget,
    OrchestratorConfig,
    ResearchOrchestrator,
)
from alpha_agent.agents.research_agent import ResearchAgent
from alpha_agent.registry.models import MarketWindow
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.validation.policy import ReliabilityPolicy

__all__ = ["build_deep_research_orchestrator"]

#: A conservative default for one "Run Deep Research" click (Agent runtime-
#: integration release, section 12: a professional autonomous researcher must
#: have bounded authority). `target_family_size=1` bounds the number of
#: genuinely NEW scientific experiment identities a single click can create;
#: `max_execution_attempts_per_identity=2` allows exactly one engineering
#: re-run after an INVALID_EXECUTION, never more.
DEFAULT_DEEP_RESEARCH_BUDGET = OrchestratorBudget(
    max_planning_attempts_per_family=4,
    max_execution_attempts_per_identity=2,
    agent_schema_retries=2,
    plan_successor_family_after_finalization=False,
)


def build_deep_research_orchestrator(
    *,
    registry: ExperimentRegistry,
    objective: str,
    root: str,
    family_stem: str,
    research_client: LLMClient,
    compiler_client: LLMClient,
    target_family_size: int = 1,
    budget: OrchestratorBudget | None = None,
    reliability_policy: ReliabilityPolicy | None = None,
    execution_service: ExecutionValidationService | None = None,
    repo_root: str | Path | None = None,
    knowledge_base: tuple[str, ...] = (),
    phase: str = "agent-runtime",
    code_commit: str = "",
) -> ResearchOrchestrator:
    """Assemble a real `ResearchOrchestrator` for ONE root.

    One root per run: `DatasetIdentity`/`ValidationSpec` in this codebase are
    inherently per-root, and Phase 18.2's "one evaluation plane, one
    outcome-adaptive campaign" already assumes one frozen dataset/split/
    validation/policy plane per orchestrator config (see
    `alpha_agent.agents.execution_service`'s module docstring).

    `research_client` / `compiler_client` are separate on purpose (mirroring
    `ResearchAgent(client)` / `StrategyCompilerAgent(client)`'s own
    separation): a `ScriptedLLMClient` carries one fixed response list, and
    the research/compile calls happen in a fixed relative order across
    multiple planning slots, so sharing one scripted instance between the two
    roles would require the caller to interleave two different kinds of
    canned response by hand. Passing the SAME live `AnthropicClient()`
    instance (or two separate ones) for both is fine -- it is a stateless
    transport either way.

    `execution_service` defaults to a real `ProductionExecutionValidationService`
    bound to `reliability_policy` (default: the frozen `ReliabilityPolicy()`).
    The returned orchestrator's `IdentityPlanes` come from
    `execution_service.identity_planes_for_root(root)` -- the SAME derivation
    the execution service asserts against at run time, so planning and
    execution can never disagree about the declared protocol.
    """
    policy = reliability_policy or ReliabilityPolicy()
    exec_svc = execution_service or ProductionExecutionValidationService(
        reliability_policy=policy, repo_root=repo_root,
    )
    if not hasattr(exec_svc, "identity_planes_for_root"):
        raise TypeError(
            "execution_service must expose identity_planes_for_root(root) -- pass a real "
            "ProductionExecutionValidationService, or build OrchestratorConfig.planes "
            "yourself when injecting a test double"
        )
    planes = exec_svc.identity_planes_for_root(root)

    config = OrchestratorConfig(
        objective=objective,
        market_universe=(root,),
        market_window=MarketWindow(
            label="VALIDATION_2023_2024", start_date="2023-01-01", end_date="2024-12-31"
        ),
        planes=planes,
        family_plan=FamilyPlanSpec(family_stem=family_stem, target_family_size=target_family_size),
        budget=budget or DEFAULT_DEEP_RESEARCH_BUDGET,
        knowledge_base=knowledge_base,
        phase=phase,
        code_commit=code_commit,
    )
    research_agent = ResearchAgent(research_client)
    compiler_agent = StrategyCompilerAgent(compiler_client)
    return ResearchOrchestrator(
        registry=registry,
        research_agent=research_agent,
        compiler_agent=compiler_agent,
        execution_service=exec_svc,
        config=config,
        reliability_policy=policy,
    )
