"""Phase 8 -- reputation (prompt section 6).

"Reputation should emphasize reproducibility, research quality, provenance,
independent replication, novelty. Rejected ideas can still be high-quality
contributions." :func:`reputation_profile` never computes a single blended
"reputation score" -- see :class:`~alpha_agent.community.schemas.ReputationProfile`,
deliberately multi-dimensional (matching this platform's standing
no-combined-score convention in :mod:`alpha_agent.alpha_memory`), and never
weighted by view count, "likes" (no such interaction exists anywhere in this
package), or raw PnL/return (prompt section 7: "popularity bias").

``n_contributions`` counts every contribution a contributor has made
REGARDLESS of its own evidence's headline verdict -- a REJECT contribution
counts exactly the same as a PASS one. This is not an oversight: it is the
one deliberate design decision "rejected ideas can still be high-quality
contributions" requires.
"""
from __future__ import annotations

from alpha_agent.community.contributions import list_contributions
from alpha_agent.community.replication import list_replications
from alpha_agent.community.schemas import (
    ContributorRef,
    ReplicationOutcome,
    ReputationProfile,
    VisibilityLevel,
)
from alpha_agent.community.store import CommunityStore

__all__ = ["reputation_profile"]


def reputation_profile(store: CommunityStore, contributor_id: str) -> ReputationProfile:
    contributions = list_contributions(store)
    mine = [c for c in contributions if c.contributor.contributor_id == contributor_id]
    display_name = mine[0].contributor.display_name if mine else contributor_id

    all_replications = list_replications(store)
    performed = [r for r in all_replications if r.replicator.contributor_id == contributor_id]

    my_ids = {c.contribution_id for c in mine}
    received = [r for r in all_replications if r.contribution_id in my_ids]
    confirms = [r for r in received if r.comparison.outcome == ReplicationOutcome.CONFIRMS]
    conflicts = [r for r in received if r.comparison.outcome == ReplicationOutcome.CONFLICTS]

    if received:
        note = (
            f"{len(confirms)} of {len(received)} independent replication(s) of this contributor's shared "
            "work confirmed the original result; a REJECT or INCONCLUSIVE contribution counts toward "
            "n_contributions exactly the same as a PASS one."
        )
    else:
        note = "No independent replication of this contributor's shared work exists yet."

    return ReputationProfile(
        contributor=ContributorRef(contributor_id=contributor_id, display_name=display_name),
        n_contributions=len(mine),
        n_public_contributions=sum(1 for c in mine if c.visibility == VisibilityLevel.PUBLIC),
        n_shared_contributions=sum(1 for c in mine if c.visibility == VisibilityLevel.SHARED),
        n_replications_performed=len(performed),
        n_times_own_work_replicated=len(received),
        n_confirming_replications_received=len(confirms),
        n_conflicting_replications_received=len(conflicts),
        reproducibility_note=note,
    )
