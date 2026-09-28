"""Phase 8 -- privacy boundaries and aggregation (prompt sections 2/5).

"Users must not be forced to reveal proprietary strategy logic to benefit
from aggregates." This module is the ONE place that decision is enforced:

* :func:`community_feed` never returns a ``PRIVATE`` contribution, for
  anyone, under any viewer identity -- full privacy, not merely redaction.
  Its own contributor still sees it, but only through :func:`my_contributions`,
  never through the shared feed.
* :func:`redact_for_viewer` strips ``evidence.strategy_params`` from a
  ``SHARED`` contribution whenever the viewer is not its own contributor --
  the ONE field visibility ever redacts (see
  :class:`alpha_agent.community.schemas.EvidenceReference`'s own docstring).
  A ``PUBLIC`` contribution is never redacted.
* :func:`aggregate_for_mechanism_root` -- "aggregate without collapsing"
  (prompt section 5) -- is computed ONLY over ``SHARED``/``PUBLIC``
  contributions. A ``PRIVATE`` contribution contributes NOTHING to any
  aggregate, not even an anonymized count: a count is still a signal that
  someone is privately researching something, and full privacy means zero
  signal, not a smaller one. Every count on
  :class:`~alpha_agent.community.schemas.CommunityAggregateRow` stays a
  separate, inspectable dimension -- evidence maturity distribution and
  verdict distribution are never merged into one blended figure.
"""
from __future__ import annotations

from alpha_agent.community.contributions import list_contributions
from alpha_agent.community.replication import evidence_maturity, list_replications
from alpha_agent.community.schemas import CommunityAggregateRow, Contribution, VisibilityLevel
from alpha_agent.community.store import CommunityStore
from alpha_agent.knowledge.models import EconomicMechanism

__all__ = ["aggregate_for_mechanism_root", "community_feed", "my_contributions", "redact_for_viewer"]


def redact_for_viewer(contribution: Contribution, viewer_contributor_id: str | None) -> Contribution:
    """Strip ``strategy_params`` from a ``SHARED`` contribution unless the
    viewer IS its own contributor. ``PRIVATE``/``PUBLIC`` are unaffected here
    -- a caller must exclude ``PRIVATE`` before this function ever sees it
    (see :func:`community_feed`)."""
    is_owner = viewer_contributor_id is not None and contribution.contributor.contributor_id == viewer_contributor_id
    if contribution.visibility == VisibilityLevel.SHARED and not is_owner and contribution.evidence.strategy_params is not None:
        return contribution.model_copy(
            update={"evidence": contribution.evidence.model_copy(update={"strategy_params": None})}
        )
    return contribution


def community_feed(store: CommunityStore, *, viewer_contributor_id: str | None = None) -> list[Contribution]:
    """``PUBLIC`` (full detail) + ``SHARED`` (redacted unless the viewer is
    its own contributor), newest first. ``PRIVATE`` never appears here for
    anyone, including its own contributor -- see :func:`my_contributions`."""
    visible = [c for c in list_contributions(store) if c.visibility != VisibilityLevel.PRIVATE]
    return [redact_for_viewer(c, viewer_contributor_id) for c in reversed(visible)]


def my_contributions(store: CommunityStore, contributor_id: str) -> list[Contribution]:
    """Every contribution by ``contributor_id``, at full (unredacted) detail,
    any visibility level -- the owner's own personal view, newest first."""
    mine = [c for c in list_contributions(store) if c.contributor.contributor_id == contributor_id]
    return list(reversed(mine))


def aggregate_for_mechanism_root(
    store: CommunityStore, *, mechanism: EconomicMechanism, root_symbol: str
) -> CommunityAggregateRow | None:
    """The privacy-safe community aggregate for one (mechanism, root) pair,
    or ``None`` when no ``SHARED``/``PUBLIC`` evidence exists yet -- absence
    is never rendered as a negative signal, matching
    :mod:`alpha_agent.alpha_graph`'s own ``EvidenceCoverage`` convention."""
    visible = [
        c
        for c in list_contributions(store)
        if c.visibility != VisibilityLevel.PRIVATE and c.mechanism == mechanism and c.root_symbol == root_symbol
    ]
    if not visible:
        return None

    n_public = sum(1 for c in visible if c.visibility == VisibilityLevel.PUBLIC)
    n_shared = sum(1 for c in visible if c.visibility == VisibilityLevel.SHARED)
    contributors = {c.contributor.contributor_id for c in visible}

    maturity_distribution: dict[str, int] = {}
    verdict_distribution: dict[str, int] = {}
    n_replications = 0
    for c in visible:
        reps = list_replications(store, c.contribution_id)
        n_replications += len(reps)
        maturity, _ = evidence_maturity(c, reps)
        maturity_distribution[maturity.value] = maturity_distribution.get(maturity.value, 0) + 1
        verdict_label = c.evidence.headline_verdict.value if c.evidence.headline_verdict else "NOT_ADJUDICATED"
        verdict_distribution[verdict_label] = verdict_distribution.get(verdict_label, 0) + 1

    return CommunityAggregateRow(
        mechanism=mechanism,
        root_symbol=root_symbol,
        n_public_contributions=n_public,
        n_shared_contributions=n_shared,
        n_contributors=len(contributors),
        n_replications=n_replications,
        evidence_maturity_distribution=maturity_distribution,
        verdict_distribution=verdict_distribution,
    )
