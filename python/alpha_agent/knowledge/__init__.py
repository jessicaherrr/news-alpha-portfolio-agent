"""Alpha Discovery campaign, Part C -- the Strategy Knowledge Base.

Idea memory, never execution authority. See `alpha_agent.knowledge.models`
for the full boundary docstring and `alpha_agent.knowledge.store.
StrategyKnowledgeBase` for the aggregate query surface.
"""
from __future__ import annotations

from alpha_agent.knowledge.classic_library import build_classic_library
from alpha_agent.knowledge.dedup import cluster_by_mechanism, deduplicate_exact
from alpha_agent.knowledge.external_sources import (
    FORBIDDEN_OPERATIONS,
    DisabledAdapter,
    ExternalSourceAdapter,
    NotConnectedAdapter,
    default_adapters,
    disabled_adapters,
    ingest_all,
    live_adapters,
)
from alpha_agent.knowledge.models import (
    TIER_ORDER,
    EconomicMechanism,
    IngestionResult,
    IngestionStatus,
    MechanismCluster,
    SourceQualityTier,
    SourceType,
    StrategyKnowledgeItem,
)
from alpha_agent.knowledge.registry_source import internal_items_from_registry_rows
from alpha_agent.knowledge.store import StrategyKnowledgeBase

__all__ = [
    "FORBIDDEN_OPERATIONS",
    "TIER_ORDER",
    "DisabledAdapter",
    "EconomicMechanism",
    "ExternalSourceAdapter",
    "IngestionResult",
    "IngestionStatus",
    "MechanismCluster",
    "NotConnectedAdapter",
    "SourceQualityTier",
    "SourceType",
    "StrategyKnowledgeBase",
    "StrategyKnowledgeItem",
    "build_classic_library",
    "cluster_by_mechanism",
    "deduplicate_exact",
    "default_adapters",
    "disabled_adapters",
    "ingest_all",
    "internal_items_from_registry_rows",
    "live_adapters",
]
