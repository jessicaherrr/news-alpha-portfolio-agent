"""Alpha Discovery campaign, Part D -- mechanism-diverse candidate generation,
built on top of the frozen Phase 18 `ResearchOrchestrator`. See
`alpha_agent.discovery.candidate_pool` / `.mechanisms` for the module-level
boundary docstrings.
"""
from __future__ import annotations

from alpha_agent.discovery.candidate_pool import (
    DEFAULT_MAX_MECHANISMS,
    DEFAULT_MAX_PLANNING_ATTEMPTS_PER_MECHANISM,
    DEFAULT_PER_MECHANISM_FAMILY_SIZE,
    CandidatePoolResult,
    MechanismOutcome,
    generate_candidate_pool,
)
from alpha_agent.discovery.mechanisms import DEFAULT_MECHANISM_ORDER, prioritize_mechanisms
from alpha_agent.discovery.replication import (
    RELATED_MARKET_GROUPS,
    ReplicationCandidate,
    build_replication_suggestions,
    propose_replication_targets,
)

__all__ = [
    "DEFAULT_MAX_MECHANISMS",
    "DEFAULT_MAX_PLANNING_ATTEMPTS_PER_MECHANISM",
    "DEFAULT_MECHANISM_ORDER",
    "DEFAULT_PER_MECHANISM_FAMILY_SIZE",
    "RELATED_MARKET_GROUPS",
    "CandidatePoolResult",
    "MechanismOutcome",
    "ReplicationCandidate",
    "build_replication_suggestions",
    "generate_candidate_pool",
    "prioritize_mechanisms",
    "propose_replication_targets",
]
