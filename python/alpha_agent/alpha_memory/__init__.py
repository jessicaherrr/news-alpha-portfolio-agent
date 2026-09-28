"""Phase 2 -- Personal Alpha Memory (prompt 2), hardened by the identity-
hardening patch (Mechanism != Factor).

Turns isolated registry experiments into reusable personal research memory:
a typed, read-only aggregation layer over the existing
:mod:`alpha_agent.registry` (scientific truth) and
:mod:`alpha_agent.translation` (Phase 1's mechanism vocabulary). This package
never writes to the registry, never calls Claude, never calls a market-data
API, and never runs a backtest -- see ``builder.py`` and ``comparison.py``
module docstrings.
"""
from __future__ import annotations

from alpha_agent.alpha_memory.builder import (
    alpha_research_object_id,
    build_alpha_research_objects_for_mechanism,
    get_alpha_research_object,
    list_alpha_research_objects,
)
from alpha_agent.alpha_memory.comparison import what_changed
from alpha_agent.alpha_memory.factor_identity import (
    compute_factor_identity,
    family_structural_signature,
    group_families_for_factor_identity,
)
from alpha_agent.alpha_memory.lookup import mechanism_memory_lookup
from alpha_agent.alpha_memory.schemas import (
    ALPHA_OBJECT_SCHEMA,
    FACTOR_IDENTITY_SCHEMA,
    AlphaResearchObject,
    EvidenceProfile,
    ExperimentEvidenceRef,
    FactorIdentity,
    MechanismMemoryLookup,
    MechanismMemoryMatchKind,
    ResearchMaturity,
    StrategyVariantEvidence,
    WhatChanged,
)

__all__ = [
    "ALPHA_OBJECT_SCHEMA",
    "FACTOR_IDENTITY_SCHEMA",
    "AlphaResearchObject",
    "EvidenceProfile",
    "ExperimentEvidenceRef",
    "FactorIdentity",
    "MechanismMemoryLookup",
    "MechanismMemoryMatchKind",
    "ResearchMaturity",
    "StrategyVariantEvidence",
    "WhatChanged",
    "alpha_research_object_id",
    "build_alpha_research_objects_for_mechanism",
    "compute_factor_identity",
    "family_structural_signature",
    "get_alpha_research_object",
    "group_families_for_factor_identity",
    "list_alpha_research_objects",
    "mechanism_memory_lookup",
    "what_changed",
]
