"""Alpha Discovery campaign, Part C -- mechanism deduplication (task spec
section 24).

The knowledge base must deduplicate exact source duplicates and collapse
"many references, one mechanism" (five GitHub repos implementing the same
20/50 MA crossover is one economic hypothesis, not five). Both operations are
purely structural -- keyed on `economic_mechanism` (a closed enum, task spec
section 24's own resolution: "preserve source lineage while normalizing
mechanism identity") plus `knowledge_id` for exact duplicates. Nothing here
inspects prose similarity or LLM judgment; the boundary is deliberately
mechanical and auditable.
"""
from __future__ import annotations

from collections.abc import Iterable

from alpha_agent.knowledge.models import EconomicMechanism, MechanismCluster, StrategyKnowledgeItem


def deduplicate_exact(items: Iterable[StrategyKnowledgeItem]) -> tuple[StrategyKnowledgeItem, ...]:
    """Drop an exact repeat `knowledge_id` (same item ingested twice, e.g. by
    two adapters). First occurrence wins; order otherwise preserved."""
    seen: set[str] = set()
    out: list[StrategyKnowledgeItem] = []
    for it in items:
        if it.knowledge_id in seen:
            continue
        seen.add(it.knowledge_id)
        out.append(it)
    return tuple(out)


def cluster_by_mechanism(items: Iterable[StrategyKnowledgeItem]) -> tuple[MechanismCluster, ...]:
    """Group items by `economic_mechanism`. Deterministic ordering: clusters
    sorted by mechanism name; items within a cluster keep their input order."""
    buckets: dict[EconomicMechanism, list[StrategyKnowledgeItem]] = {}
    for it in items:
        buckets.setdefault(it.economic_mechanism, []).append(it)
    return tuple(
        MechanismCluster(economic_mechanism=mech, items=tuple(buckets[mech]))
        for mech in sorted(buckets, key=lambda m: m.value)
    )


def mechanism_reference_count(clusters: Iterable[MechanismCluster]) -> dict[str, int]:
    """How many source references back each mechanism -- the "5 GitHub repos ->
    1 mechanism" collapse made visible, e.g. for a UI/report."""
    return {c.economic_mechanism.value: len(c.items) for c in clusters}
