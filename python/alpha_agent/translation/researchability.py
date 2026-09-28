"""Deterministic researchability classification (prompt 1 section 7).

The ONLY authority for whether a proposed factor concept is
``AVAILABLE``/``PARTIALLY_AVAILABLE``/``DATA_MISSING``/``NOT_EXECUTABLE`` is
live repository state -- ``alpha_agent.features.REGISTRY.kinds()`` -- never a
proposer's (Claude's or the deterministic library's) own claim. Missing data
is never converted to zero, and a weaker proxy is never silently substituted
for the same factor concept and called available.
"""
from __future__ import annotations

from alpha_agent.features.registry import REGISTRY, FeatureRegistry
from alpha_agent.translation.schemas import ResearchabilityStatus

__all__ = ["classify_factor"]


def classify_factor(
    *,
    proposed_feature_kinds: tuple[str, ...],
    required_external_data: tuple[str, ...],
    structurally_expressible: bool = True,
    registry: FeatureRegistry | None = None,
) -> tuple[ResearchabilityStatus, str, tuple[str, ...], tuple[str, ...]]:
    """Returns ``(status, reason, available_feature_kinds, missing_requirements)``.

    ``structurally_expressible=False`` is reserved for a concept the
    deterministic mechanism library already knows today's StrategySpec /
    FeatureRegistry model cannot express AT ALL regardless of data (e.g. a
    cross-instrument/multi-contract join -- no registered feature kind ever
    operates over more than one instrument's own series). This is never set
    from a proposer's own claim; it is a documented, hand-reviewed property of
    the concept itself (see ``alpha_agent.translation.mechanism_library``).
    """
    reg = registry or REGISTRY
    known = set(reg.kinds())
    available = tuple(k for k in proposed_feature_kinds if k in known)
    missing_kinds = tuple(k for k in proposed_feature_kinds if k not in known)
    missing_requirements = missing_kinds + tuple(required_external_data)

    if not structurally_expressible:
        reason = (
            "This concept is not expressible in today's StrategySpec / FeatureRegistry model "
            "(e.g. it needs a cross-instrument or multi-contract join, and every registered "
            "feature kind computes over exactly one instrument's own series)."
        )
        return (ResearchabilityStatus.NOT_EXECUTABLE, reason, available, missing_requirements)

    if proposed_feature_kinds and not missing_kinds and not required_external_data:
        return (
            ResearchabilityStatus.AVAILABLE,
            "Every required input is already a registered FeatureRegistry kind.",
            available,
            (),
        )

    if available and (missing_kinds or required_external_data):
        reason = (
            f"{', '.join(available)} already registered, but this concept also needs: "
            f"{', '.join(missing_requirements)}."
        )
        return (ResearchabilityStatus.PARTIALLY_AVAILABLE, reason, available, missing_requirements)

    if missing_requirements:
        reason = (
            "No registered feature covers this concept yet; it needs: "
            f"{', '.join(missing_requirements)}, which is not currently ingested with point-in-time provenance."
        )
        return (ResearchabilityStatus.DATA_MISSING, reason, available, missing_requirements)

    reason = (
        "No registered feature kind or external data requirement was named for this concept -- "
        "it has no measurable representation yet."
    )
    return (ResearchabilityStatus.NOT_EXECUTABLE, reason, available, missing_requirements)
