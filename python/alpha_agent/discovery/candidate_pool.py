"""Alpha Discovery campaign, Part D -- diverse candidate-pool generation
(task spec sections 26-29), built entirely on top of the FROZEN Phase 18
`ResearchOrchestrator.plan_family` -- this module never modifies, subclasses,
or monkeypatches that orchestrator. It only calls `plan_family()` several
times, once per prioritized economic mechanism, each with a different
(objective, knowledge_base) pair, and merges the resulting `FamilyMember`s
into one pool.

Why this is genuine mechanism diversity, not repeated parameter variants
(task spec section 26): each `plan_family()` call gets an objective that
names ONE specific economic mechanism and explicitly asks the agent not to
repeat a mechanism already covered earlier in the same pool, backed by real
`StrategyKnowledgeBase` snippets for that mechanism. FailureMemory is already
wired into every `plan_family()` call by the frozen orchestrator itself
(`ResearchOrchestrator._planning_context`) -- nothing to duplicate here
(task spec section 29).

Every produced `FamilyMember` is a fully compiled, identity-computed,
duplicate-checked candidate exactly as Phase 18 already guarantees. This
module adds NO new execution authority and computes no PnL, Sharpe, or
verdict -- it only decides what gets PROPOSED and in what order.
"""
from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from alpha_agent.agents import (
    FamilyMember,
    FamilyPlanSpec,
    OrchestratorBudget,
    OrchestratorConfig,
    PlanningSlotOutcome,
    ResearchAgent,
    ResearchOrchestrator,
    StrategyCompilerAgent,
)
from alpha_agent.agents.llm import LLMClient
from alpha_agent.agents.orchestrator import ExecutionValidationService, IdentityPlanes
from alpha_agent.discovery.mechanisms import DEFAULT_MECHANISM_ORDER, prioritize_mechanisms
from alpha_agent.knowledge import EconomicMechanism, StrategyKnowledgeBase
from alpha_agent.recommendation.profile import InvestorProfile
from alpha_agent.registry.models import MarketWindow
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.validation.policy import ReliabilityPolicy

#: Bounded defaults (task spec section 40). Never an uncontrolled loop.
DEFAULT_MAX_MECHANISMS = 8
DEFAULT_PER_MECHANISM_FAMILY_SIZE = 2
DEFAULT_MAX_PLANNING_ATTEMPTS_PER_MECHANISM = 4


class MechanismOutcome(BaseModel):
    """What one mechanism nudge produced. `ideas_considered` counts every
    planning slot attempted for this mechanism (accepted or not); `members`
    is what actually survived into the pool."""

    model_config = {"frozen": True, "extra": "forbid"}

    mechanism: EconomicMechanism
    objective: str
    ideas_considered: int
    members: tuple[FamilyMember, ...]
    planning_log: tuple[PlanningSlotOutcome, ...]


class CandidatePoolResult(BaseModel):
    """The merged, re-ordinaled candidate pool -- Part F's (Fast Screen)
    input. Never itself a `FamilyManifest`: nothing here is frozen yet (task
    spec section 42 -- freezing happens only after fast-screen ranking, in
    `alpha_agent.screening.freeze`)."""

    model_config = {"frozen": True, "extra": "forbid"}

    market: str
    ideas_considered: int
    mechanisms_attempted: int
    mechanisms_supported: int
    members: tuple[FamilyMember, ...]
    per_mechanism: tuple[MechanismOutcome, ...]

    @property
    def strategy_families(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(m.strategy_family for m in self.members))


def _mechanism_objective(
    *, base_objective: str, market: str, mechanism: EconomicMechanism, already_covered: tuple[str, ...]
) -> str:
    lines = [
        base_objective.rstrip("."),
        f"for {market}.",
        f"Focus this proposal on the following economic mechanism category: {mechanism.value}.",
    ]
    if already_covered:
        lines.append(
            "Do NOT repeat a mechanism already covered in this research pass: "
            + ", ".join(already_covered) + "."
        )
    return " ".join(lines)


def generate_candidate_pool(
    *,
    market: str,
    objective: str,
    registry: ExperimentRegistry,
    research_llm: LLMClient,
    compiler_llm: LLMClient,
    execution_service: ExecutionValidationService,
    reliability_policy: ReliabilityPolicy,
    planes: IdentityPlanes,
    market_window: MarketWindow,
    fdr_q_threshold: float = 0.10,
    knowledge_base: StrategyKnowledgeBase | None = None,
    profile: InvestorProfile | None = None,
    max_mechanisms: int = DEFAULT_MAX_MECHANISMS,
    per_mechanism_family_size: int = DEFAULT_PER_MECHANISM_FAMILY_SIZE,
    max_planning_attempts_per_mechanism: int = DEFAULT_MAX_PLANNING_ATTEMPTS_PER_MECHANISM,
    family_stem: str = "discovery",
    orchestrator_factory: Callable[..., ResearchOrchestrator] = ResearchOrchestrator,
) -> CandidatePoolResult:
    """Run one bounded, mechanism-diverse discovery pass for `market`.

    `research_llm` / `compiler_llm` are shared `LLMClient`s (in production, one
    real Anthropic client each; in tests, a `ScriptedLLMClient` whose response
    queue is consumed IN ORDER across every mechanism nudge below) -- a fresh
    `ResearchAgent`/`StrategyCompilerAgent` wrapper is built per mechanism so
    each can carry its own mechanism-specific objective/knowledge base, while
    every wrapper still shares the SAME underlying client and the SAME
    `registry` (so cross-mechanism duplicate detection still works).
    """
    if max_mechanisms < 1:
        raise ValueError("max_mechanisms must be >= 1")
    if per_mechanism_family_size < 1:
        raise ValueError("per_mechanism_family_size must be >= 1")

    ordered_mechanisms = prioritize_mechanisms(profile, order=DEFAULT_MECHANISM_ORDER)[:max_mechanisms]

    all_members: list[FamilyMember] = []
    per_mechanism: list[MechanismOutcome] = []
    covered_labels: list[str] = []
    ideas_considered_total = 0

    for mechanism in ordered_mechanisms:
        snippets: tuple[str, ...] = ()
        if knowledge_base is not None:
            items = knowledge_base.query(market=market, mechanism=mechanism)
            snippets = knowledge_base.render_for_research_context(items)

        mech_objective = _mechanism_objective(
            base_objective=objective, market=market, mechanism=mechanism,
            already_covered=tuple(covered_labels),
        )
        cfg = OrchestratorConfig(
            objective=mech_objective,
            market_universe=(market,),
            market_window=market_window,
            planes=planes,
            family_plan=FamilyPlanSpec(
                family_stem=f"{family_stem}_{mechanism.value.lower()}",
                target_family_size=per_mechanism_family_size,
            ),
            budget=OrchestratorBudget(
                max_planning_attempts_per_family=max_planning_attempts_per_mechanism,
            ),
            knowledge_base=snippets,
        )
        orch = orchestrator_factory(
            registry=registry,
            research_agent=ResearchAgent(research_llm),
            compiler_agent=StrategyCompilerAgent(compiler_llm),
            execution_service=execution_service,
            reliability_policy=reliability_policy,
            config=cfg,
        )
        manifest = orch.plan_family()
        outcome = MechanismOutcome(
            mechanism=mechanism, objective=mech_objective,
            ideas_considered=len(manifest.planning_log),
            members=manifest.members, planning_log=manifest.planning_log,
        )
        per_mechanism.append(outcome)
        ideas_considered_total += outcome.ideas_considered
        if manifest.members:
            covered_labels.append(mechanism.value)
        all_members.extend(manifest.members)

    reordinaled = tuple(m.model_copy(update={"ordinal": i}) for i, m in enumerate(all_members))
    return CandidatePoolResult(
        market=market,
        ideas_considered=ideas_considered_total,
        mechanisms_attempted=len(ordered_mechanisms),
        mechanisms_supported=sum(1 for o in per_mechanism if o.members),
        members=reordinaled,
        per_mechanism=tuple(per_mechanism),
    )


__all__ = [
    "DEFAULT_MAX_MECHANISMS",
    "DEFAULT_MAX_PLANNING_ATTEMPTS_PER_MECHANISM",
    "DEFAULT_PER_MECHANISM_FAMILY_SIZE",
    "CandidatePoolResult",
    "MechanismOutcome",
    "generate_candidate_pool",
]
