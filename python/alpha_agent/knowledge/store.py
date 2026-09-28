"""Alpha Discovery campaign, Part C -- the `StrategyKnowledgeBase` aggregate.

Composes the classic library (always present), the internal registry source
(present when the caller has registry rows), and the external adapters
(GITHUB/ACADEMIC/COMMUNITY/PRACTITIONER -- `NOT_CONNECTED` by default, task
spec section 23), deduplicates, and exposes a small deterministic query
surface. This is IDEA MEMORY (module docstring of `.models`): nothing here
computes PnL, assigns a verdict, or is capable of reaching execution on its
own -- see `alpha_agent.knowledge.external_sources`'s safety boundary.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from alpha_agent.knowledge.classic_library import build_classic_library
from alpha_agent.knowledge.dedup import cluster_by_mechanism, deduplicate_exact
from alpha_agent.knowledge.external_sources import (
    ExternalSourceAdapter,
    default_adapters,
    ingest_all,
)
from alpha_agent.knowledge.models import (
    EconomicMechanism,
    IngestionResult,
    MechanismCluster,
    SourceType,
    StrategyKnowledgeItem,
)
from alpha_agent.knowledge.registry_source import internal_items_from_registry_rows


class StrategyKnowledgeBase:
    """Built once per query context (a market + objective), never mutated in
    place -- `refresh()` returns a NEW instance rather than mutating `self`,
    matching the platform's frozen-model convention elsewhere."""

    def __init__(
        self,
        *,
        registry_rows: Sequence[dict[str, Any]] = (),
        adapters: dict[SourceType, ExternalSourceAdapter] | None = None,
        query: str = "",
        markets: tuple[str, ...] = (),
    ):
        self._adapters = adapters if adapters is not None else default_adapters()
        classic = build_classic_library()
        internal = internal_items_from_registry_rows(list(registry_rows))
        self._external_results: tuple[IngestionResult, ...] = ingest_all(
            self._adapters, query=query, markets=markets
        )
        external_items = tuple(it for r in self._external_results for it in r.items)
        self._items: tuple[StrategyKnowledgeItem, ...] = deduplicate_exact(
            classic + internal + external_items
        )

    @property
    def items(self) -> tuple[StrategyKnowledgeItem, ...]:
        return self._items

    @property
    def external_ingestion_results(self) -> tuple[IngestionResult, ...]:
        """Per-source honest status (task spec section 23) -- e.g. for a UI
        "Connected Research" panel showing GITHUB/ACADEMIC/COMMUNITY/
        PRACTITIONER as NOT_CONNECTED today."""
        return self._external_results

    def query(
        self,
        *,
        market: str | None = None,
        mechanism: EconomicMechanism | None = None,
        source_types: tuple[SourceType, ...] | None = None,
    ) -> tuple[StrategyKnowledgeItem, ...]:
        out = self._items
        if market is not None:
            out = tuple(i for i in out if not i.markets or market in i.markets)
        if mechanism is not None:
            out = tuple(i for i in out if i.economic_mechanism == mechanism)
        if source_types is not None:
            out = tuple(i for i in out if i.source_type in source_types)
        return out

    def mechanism_clusters(
        self, *, market: str | None = None
    ) -> tuple[MechanismCluster, ...]:
        return cluster_by_mechanism(self.query(market=market))

    def available_mechanisms(self, *, market: str | None = None) -> tuple[EconomicMechanism, ...]:
        return tuple(c.economic_mechanism for c in self.mechanism_clusters(market=market))

    @staticmethod
    def render_for_research_context(
        items: Sequence[StrategyKnowledgeItem], *, limit: int = 12
    ) -> tuple[str, ...]:
        """Deterministic short text snippets safe for
        `alpha_agent.agents.context.ResearchContext.knowledge_base` /
        `alpha_agent.agents.orchestrator.OrchestratorConfig.knowledge_base`
        (both already holdout-guarded downstream, and both already treat this
        exact shape -- ``tuple[str, ...]`` -- as an observational-context path
        that never itself carries a market-data value)."""
        return tuple(i.render_snippet() for i in items[:limit])
