"""Deterministic prior-research lookup for a translated observation (prompt 1
sections 10/11).

Every count here comes from ``alpha_agent.registry.failure_memory.FailureMemory``
-- the same deterministic, offline scientific memory the runtime Research
Agent already queries before proposing (``alpha_agent.agents.context``). This
module never optimizes a parameter, never shrinks a multiple-testing family,
and never treats an exploratory translation as a new scientific trial: it only
READS what has already been tried and returns a typed digest.
"""
from __future__ import annotations

from alpha_agent.agents.context import FailureMemoryDigest, build_failure_memory_digest
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.enums import AssetDomain
from alpha_agent.registry.failure_memory import FailureMemory
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.translation.schemas import ResearchMemoryNote

__all__ = ["MECHANISM_TO_KNOWN_FAMILIES", "mechanism_research_memory"]

#: A conservative, documented bridge from an ``EconomicMechanism`` to the
#: frozen Phase 11 candidate family key(s)
#: (``alpha_agent.strategy.baselines.families``) that structurally implement
#: it TODAY. Deliberately NOT exhaustive: a mechanism absent from this table
#: has no representation in the current candidate manifest yet -- that is
#: itself a real, honest research-memory fact, never something this table
#: papers over by widening a mapping just to force a match. Never root-
#: specific (CLAUDE.md: "never special-case a root").
MECHANISM_TO_KNOWN_FAMILIES: dict[EconomicMechanism, tuple[str, ...]] = {
    EconomicMechanism.TREND: ("tsmom", "ma_trend"),
    EconomicMechanism.MOMENTUM: ("tsmom",),
    EconomicMechanism.MEAN_REVERSION: ("mean_reversion",),
    EconomicMechanism.BREAKOUT: ("breakout",),
    EconomicMechanism.FAILED_BREAKOUT: ("breakout",),
    EconomicMechanism.VOLATILITY_BREAKOUT: ("breakout",),
}


def mechanism_research_memory(
    *,
    mechanisms: tuple[EconomicMechanism, ...],
    root_symbol: str,
    registry: ExperimentRegistry,
) -> ResearchMemoryNote:
    """Prior evidence for every strategy family known to implement one of
    ``mechanisms`` on ``root_symbol``, plus root-scoped engineering lessons
    that constrain any new work on this market regardless of family."""
    families: list[str] = []
    for m in mechanisms:
        for f in MECHANISM_TO_KNOWN_FAMILIES.get(m, ()):
            if f not in families:
                families.append(f)

    fm = FailureMemory(registry)
    # this translation layer has no ETF wiring yet -- a stated fact, not an
    # inference (Phase 6 ETF Research Pilot's translation integration is a
    # separate, later phase).
    digests: tuple[FailureMemoryDigest, ...] = tuple(
        build_failure_memory_digest(
            fm.lookup(strategy_family=f, root_symbol=root_symbol, asset_domain=AssetDomain.FUTURES)
        )
        for f in families
    )
    engineering = fm.related_engineering_failures(root_symbol=root_symbol)
    lessons = tuple(f"{f.failure_code}: {f.summary}" for f in engineering)

    # A family being MAPPED (has a candidate-family representation at all)
    # is not the same claim as that family having actually been TESTED (a
    # real execution attempt exists for it on this root) -- see
    # `ResearchMemoryNote`'s own docstring. `len(families)` alone previously
    # overstated tested_family_count for a mapped-but-never-attempted family.
    mapped_family_count = len(families)
    tested_family_count = sum(1 for d in digests if (d.valid_execution_attempts + d.invalid_execution_attempts) > 0)
    underexplored = tested_family_count == 0

    if not families:
        mech_names = ", ".join(m.value for m in mechanisms) or "this mechanism"
        summary = (
            f"No strategy family in the current candidate manifest directly implements {mech_names} yet -- "
            "this is a real research-representation gap, not evidence the mechanism was tried and failed."
        )
    elif underexplored:
        # NEVER "genuinely new" / "novel" / "never tested before" -- this
        # lookup only establishes that no attempt was found in the currently
        # MAPPED family/root scope, not genuine novelty across the whole
        # registry (Phase 1 acceptance patch, section 3: that claim would
        # need deterministic similarity evidence this lookup does not have).
        summary = (
            f"No prior registry attempts were found in the currently mapped family/root scope "
            f"({', '.join(families)} on {root_symbol}). This appears underexplored in the current memory."
        )
    else:
        parts = [
            f"{d.strategy_family}: {d.valid_execution_attempts} valid attempt(s), "
            f"{d.scientific_verdict_counts or 'no adjudicated verdict yet'}"
            for d in digests
        ]
        summary = "; ".join(parts)

    return ResearchMemoryNote(
        root_symbol=root_symbol,
        related_strategy_families=tuple(families),
        digests=digests,
        engineering_lessons=lessons,
        mapped_family_count=mapped_family_count,
        tested_family_count=tested_family_count,
        underexplored=underexplored,
        summary=summary,
    )
