"""Phase 8 -- the Community Alpha Network.

Reproducible research and independent replication, not a strategy
marketplace: no copy trading, no raw-return leaderboard, no "Like" (see this
package's module docstrings for the full boundary). A typed READ/WRITE model
over its OWN separate store (:mod:`alpha_agent.community.store`,
``data/community/community.sqlite``) -- never a second scientific registry;
every :class:`~alpha_agent.community.schemas.Contribution` /
:class:`~alpha_agent.community.schemas.ReplicationRecord` anchors to real,
already-committed evidence in the Phase 14 experiment registry
(:mod:`alpha_agent.registry`), verified at creation time and never re-derived.
"""
from __future__ import annotations

from alpha_agent.community.contributions import (
    CherryPickedTrialError,
    CommunityHoldoutError,
    build_evidence_reference,
    contributor_ref,
    create_contribution,
    get_contribution,
    list_contributions,
)
from alpha_agent.community.memory import community_memory_lookup
from alpha_agent.community.replication import (
    NotIndependentReplicationError,
    compare_replication,
    create_replication,
    evidence_maturity,
    list_replications,
)
from alpha_agent.community.reputation import reputation_profile
from alpha_agent.community.schemas import (
    COMMUNITY_SCHEMA,
    CommunityAggregateRow,
    CommunityMemoryLookup,
    Contribution,
    ContributionKind,
    ContributorRef,
    EvidenceMaturity,
    EvidenceReference,
    ReplicationComparison,
    ReplicationOutcome,
    ReplicationRecord,
    ReputationProfile,
    VisibilityLevel,
)
from alpha_agent.community.store import (
    DEFAULT_COMMUNITY_STORE_PATH,
    CommunityStore,
    UnknownContribution,
)
from alpha_agent.community.visibility import (
    aggregate_for_mechanism_root,
    community_feed,
    my_contributions,
)

__all__ = [
    "COMMUNITY_SCHEMA",
    "DEFAULT_COMMUNITY_STORE_PATH",
    "CherryPickedTrialError",
    "CommunityAggregateRow",
    "CommunityHoldoutError",
    "CommunityMemoryLookup",
    "CommunityStore",
    "Contribution",
    "ContributionKind",
    "ContributorRef",
    "EvidenceMaturity",
    "EvidenceReference",
    "NotIndependentReplicationError",
    "ReplicationComparison",
    "ReplicationOutcome",
    "ReplicationRecord",
    "ReputationProfile",
    "UnknownContribution",
    "VisibilityLevel",
    "aggregate_for_mechanism_root",
    "build_evidence_reference",
    "community_feed",
    "community_memory_lookup",
    "compare_replication",
    "contributor_ref",
    "create_contribution",
    "create_replication",
    "evidence_maturity",
    "get_contribution",
    "list_contributions",
    "list_replications",
    "my_contributions",
    "reputation_profile",
]
