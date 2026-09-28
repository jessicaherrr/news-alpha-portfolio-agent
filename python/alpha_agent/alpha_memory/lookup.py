"""Phase 1 -> Phase 2 hand-off: "have I researched this mechanism before?"
(prompt 2 section 17; hardened by the identity-hardening patch, section 5,
then by the final Phase 2 semantic fix, section 2).

A Phase 1 candidate (``MechanismCandidate`` / ``FactorCandidate``) only ever
knows its ``EconomicMechanism`` at the point this lookup runs today -- Phase
1 stops before compiling a concrete ``StrategySpec``, so no
``strategy_family`` exists yet to compute an exact candidate
``FactorIdentity`` from. Mechanism alone can NEVER establish an exact Factor
match, however few historical Factor objects exist under that mechanism:
cardinality is not proof. "If exact Factor match cannot be established, say
RELATED MECHANISM MEMORY rather than MATCHING FACTOR MEMORY" -- so a
mechanism-only lookup (``strategy_family=None``) returns at most
RELATED_MECHANISM, never MATCHING_FACTOR. It is acceptable for today's real
Phase 1 hand-off to return only RELATED_MECHANISM or NONE; MATCHING_FACTOR
is reachable only once a caller supplies an explicit candidate
``strategy_family`` whose own deterministic Factor identity equals a stored
object's exactly.
"""
from __future__ import annotations

from alpha_agent.alpha_memory.builder import build_alpha_research_objects_for_mechanism
from alpha_agent.alpha_memory.factor_identity import compute_factor_identity
from alpha_agent.alpha_memory.schemas import MechanismMemoryLookup, MechanismMemoryMatchKind
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.sqlite_registry import ExperimentRegistry

__all__ = ["mechanism_memory_lookup"]


def mechanism_memory_lookup(
    registry: ExperimentRegistry,
    *,
    mechanism: EconomicMechanism,
    root_symbol: str,
    strategy_family: str | None = None,
) -> MechanismMemoryLookup:
    """Real, registry-grounded AlphaResearchObject(s) for one (root,
    mechanism) pair, with an honest match-confidence label:

    * no real evidence at all -> :attr:`MechanismMemoryMatchKind.NONE`
      (``objects`` empty).
    * ``strategy_family`` supplied AND its own deterministic
      ``FactorIdentity`` equals a stored object's exactly ->
      :attr:`MechanismMemoryMatchKind.MATCHING_FACTOR` (``objects`` holds
      just the exact match(es)).
    * otherwise (no ``strategy_family`` given, or it matches no stored
      Factor exactly) but real evidence exists under this mechanism ->
      :attr:`MechanismMemoryMatchKind.RELATED_MECHANISM` (``objects`` lists
      every real Factor under the mechanism, for the caller/user to judge --
      never a guess, and never inferred from there being only one).
    """
    objects = build_alpha_research_objects_for_mechanism(registry, root_symbol=root_symbol, mechanism=mechanism)
    if not objects:
        return MechanismMemoryLookup(
            mechanism=mechanism, root_symbol=root_symbol, match_kind=MechanismMemoryMatchKind.NONE, objects=(),
        )

    if strategy_family is not None:
        candidate_factor = compute_factor_identity(mechanism, (strategy_family,))
        matches = tuple(o for o in objects if o.factor.factor_identity == candidate_factor.factor_identity)
        if matches:
            return MechanismMemoryLookup(
                mechanism=mechanism, root_symbol=root_symbol,
                match_kind=MechanismMemoryMatchKind.MATCHING_FACTOR, objects=matches,
            )

    return MechanismMemoryLookup(
        mechanism=mechanism, root_symbol=root_symbol,
        match_kind=MechanismMemoryMatchKind.RELATED_MECHANISM, objects=objects,
    )
