"""Phase 8 -- the typed Community Alpha Network object model (prompt 8).

NOT a strategy marketplace, no copy trading, no raw-return leaderboard (prompt
section headline). This module defines the closed vocabulary every other
module in this package reasons over: contribution kinds, visibility levels,
evidence maturity, and the typed evidence a contribution/replication is
allowed to carry.

Structural invariants, held throughout the package (never just in prose):

* A contribution's :class:`EvidenceReference` is built ONLY by looking up a
  real registry ``experiment_id`` (:mod:`alpha_agent.community.contributions`)
  -- every field on it is copied verbatim from the registry, never accepted
  as free text. This is the package's primary defense against fake
  provenance (prompt section 7).
* :attr:`EvidenceReference.strategy_params` is the one field visibility
  redaction ever removes (:mod:`alpha_agent.community.visibility`) -- a
  ``SHARED`` contribution benefits from aggregates and is independently
  replicable without ever forcing its owner to reveal exact parameters
  (prompt section 2).
* ``EvidenceMaturity`` is a closed, conservative ladder
  (:mod:`alpha_agent.community.replication`) -- PROPOSED / REPLICATING /
  EMERGING_EVIDENCE never outrank what the underlying registry verdict and
  replication history actually support (prompt section 3: "use stronger
  labels only if scientific policy supports them").
* Nothing here computes a single blended "reputation score", "community
  score", or "alpha score" -- see :class:`ReputationProfile` and
  :class:`CommunityAggregateRow`, both deliberately multi-dimensional.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.enums import AssetDomain, RegistryVerdict, TrialRole

__all__ = [
    "COMMUNITY_SCHEMA",
    "CommunityAggregateRow",
    "CommunityMemoryLookup",
    "Contribution",
    "ContributionKind",
    "ContributorRef",
    "EvidenceMaturity",
    "EvidenceReference",
    "ReplicationComparison",
    "ReplicationOutcome",
    "ReplicationRecord",
    "ReputationProfile",
    "VisibilityLevel",
]

#: v1 -- first cut of the Phase 8 Community Alpha Network object model.
COMMUNITY_SCHEMA = "community/1"


class ContributionKind(str, Enum):
    """What kind of research object is being shared (prompt section 1) --
    deliberately NOT an unsupported BUY/SELL call, which has no
    representation anywhere in this enum."""

    HYPOTHESIS = "HYPOTHESIS"
    MECHANISM = "MECHANISM"
    FACTOR_DEFINITION = "FACTOR_DEFINITION"
    STRATEGY_SPEC = "STRATEGY_SPEC"
    EVIDENCE_BUNDLE = "EVIDENCE_BUNDLE"
    RESEARCH_NOTES = "RESEARCH_NOTES"


class VisibilityLevel(str, Enum):
    """Prompt section 2. ``PRIVATE`` is full privacy -- excluded from every
    community feed, aggregate, and replication surface; only its own
    contributor ever sees it (indistinguishable, from the outside, from a
    contribution that was never made at all). ``SHARED`` benefits from
    community aggregates and is independently replicable while its exact
    ``EvidenceReference.strategy_params`` stay redacted
    (:mod:`alpha_agent.community.visibility`) -- "users must not be forced to
    reveal proprietary strategy logic to benefit from aggregates". ``PUBLIC``
    is fully visible, parameters included."""

    PRIVATE = "PRIVATE"
    SHARED = "SHARED"
    PUBLIC = "PUBLIC"


class EvidenceMaturity(str, Enum):
    """Prompt section 3: "use stronger labels only if scientific policy
    supports them". A conservative, documented ladder --
    :func:`alpha_agent.community.replication.evidence_maturity` is the ONE
    function that ever computes this; it is never asserted by a contributor.
    ``REJECTED`` / ``INCONCLUSIVE`` mirror the registry's own
    ``RegistryVerdict`` semantics exactly rather than inventing a parallel
    vocabulary."""

    PROPOSED = "PROPOSED"
    REPLICATING = "REPLICATING"
    EMERGING_EVIDENCE = "EMERGING_EVIDENCE"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class ReplicationOutcome(str, Enum):
    """The factual relationship between an original contribution's evidence
    and one independent replication's evidence -- never a judgement call,
    always derived from the two sides' own committed ``headline_verdict``s."""

    CONFIRMS = "CONFIRMS"
    CONFLICTS = "CONFLICTS"
    PARTIAL = "PARTIAL"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class ContributorRef(BaseModel):
    """A community identity. There is no authentication system anywhere in
    this local research tool (CLAUDE.md's MVP is historical research + paper
    trading only) -- ``identity_verified`` is always ``False`` today, and is
    never presented as a verified real-world identity anywhere in the UI.
    ``contributor_id`` is a stable, normalized slug of ``display_name`` --
    two contributions with the same typed name resolve to the SAME
    contributor for reputation/aggregation purposes."""

    model_config = {"frozen": True, "extra": "forbid"}

    contributor_id: str
    display_name: str
    identity_verified: bool = False


class EvidenceReference(BaseModel):
    """Real registry evidence, and ONLY real registry evidence -- every field
    is read verbatim from :class:`~alpha_agent.registry.sqlite_registry.ExperimentView`
    at contribution/replication-create time
    (:mod:`alpha_agent.community.contributions`), never accepted as free text
    from a contributor. This is the structural defense against fake
    provenance (prompt section 7): nothing on this model is user-suppliable
    except the ``experiment_id`` used to look it up.

    ``strategy_params`` is the one field :mod:`alpha_agent.community.visibility`
    ever redacts for a ``SHARED`` contribution viewed by someone other than
    its own contributor.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    experiment_id: str
    experiment_identity: str
    root_symbol: str
    asset_domain: AssetDomain
    strategy_family: str
    trial_role: TrialRole
    headline_verdict: RegistryVerdict | None = None
    reason_codes: tuple[str, ...] = ()
    holdout_eligible: bool = False
    market_window_label: str = ""
    market_window_start: str = ""
    market_window_end: str = ""
    reliability_policy_fingerprint: str = ""
    dataset_fingerprint: str = ""
    code_commit: str = ""
    #: redacted (set to ``None``) for a SHARED contribution viewed by anyone
    #: other than its own contributor -- see class docstring.
    strategy_params: dict | None = None


class Contribution(BaseModel):
    """One shared research unit (prompt section 1): a hypothesis, mechanism,
    factor definition, reproducible ``StrategySpec``, evidence bundle, or
    research notes -- always anchored to real, verified registry evidence,
    never an unsupported trade call.

    ``recycling_decision`` / ``recycling_explanation`` are computed ONCE, at
    creation time, by :func:`alpha_agent.community.contributions._detect_duplicate_contribution`
    and preserved verbatim -- the package's duplicate-contribution defense
    (prompt section 7): a contribution repeating evidence the community
    already has is never hidden or silently blocked, only transparently
    flagged.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = COMMUNITY_SCHEMA
    contribution_id: str
    kind: ContributionKind
    visibility: VisibilityLevel
    contributor: ContributorRef
    mechanism: EconomicMechanism | None = None
    root_symbol: str
    title: str
    notes: str = ""
    novelty_notes: str = ""
    evidence: EvidenceReference
    created_at: str
    recycling_decision: str
    recycling_explanation: str


class ReplicationComparison(BaseModel):
    """Prompt section 4/5: replication preserves contributor, experiment
    identity, dataset/universe, context, factor/strategy identity, result,
    validation, and provenance -- as SEPARATE, individually inspectable
    facts, never collapsed into one score or one "reproduced: yes/no"
    boolean. ``same_factor_identity`` is ``None`` when the original
    contribution declared no ``mechanism`` (not computable, never guessed)."""

    model_config = {"frozen": True, "extra": "forbid"}

    same_root_symbol: bool
    same_asset_domain: bool
    same_strategy_family: bool
    same_factor_identity: bool | None = None
    same_market_window: bool
    same_reliability_policy: bool
    same_dataset_fingerprint: bool
    original_verdict: RegistryVerdict | None = None
    replication_verdict: RegistryVerdict | None = None
    outcome: ReplicationOutcome
    notes: str = ""


class ReplicationRecord(BaseModel):
    """One independent replication attempt against one
    :class:`Contribution`. ``replicator`` and ``evidence`` are always the
    replicator's OWN identity and OWN independently-produced registry
    evidence -- :mod:`alpha_agent.community.replication` refuses a
    replication citing the exact same ``experiment_identity`` as the
    original, AND refuses a replication from the same (self-attested)
    ``contributor_id`` as the original contribution's own contributor
    (prompt section 7: "threat-model ... copied strategies")."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = COMMUNITY_SCHEMA
    replication_id: str
    contribution_id: str
    replicator: ContributorRef
    evidence: EvidenceReference
    comparison: ReplicationComparison
    notes: str = ""
    created_at: str


class ReputationProfile(BaseModel):
    """Prompt section 6: reputation emphasizes reproducibility, research
    quality, provenance, and independent replication -- NEVER raw return, and
    NEVER a single blended "reputation score" (this platform's standing
    no-combined-score convention, held everywhere else in
    :mod:`alpha_agent.alpha_memory`/:mod:`alpha_agent.alpha_graph`). A
    contributor whose work was REJECTed under the frozen validation policy is
    not penalized here: ``n_contributions`` counts every contribution
    regardless of verdict ("rejected ideas can still be high-quality
    contributions", prompt section 6)."""

    model_config = {"frozen": True, "extra": "forbid"}

    contributor: ContributorRef
    n_contributions: int = 0
    n_public_contributions: int = 0
    n_shared_contributions: int = 0
    n_replications_performed: int = 0
    n_times_own_work_replicated: int = 0
    n_confirming_replications_received: int = 0
    n_conflicting_replications_received: int = 0
    reproducibility_note: str = ""


class CommunityAggregateRow(BaseModel):
    """A privacy-safe aggregate for one (mechanism, root) pair -- computed
    ONLY over ``SHARED``/``PUBLIC`` contributions
    (:mod:`alpha_agent.community.visibility`). A ``PRIVATE`` contribution
    never influences this row in any way, not even as an anonymized count --
    full privacy, not merely redaction (prompt section 2). ``None`` from
    :func:`alpha_agent.community.visibility.aggregate_for_mechanism_root`
    means no SHARED/PUBLIC evidence exists yet, never zero-as-a-negative-signal."""

    model_config = {"frozen": True, "extra": "forbid"}

    mechanism: EconomicMechanism
    root_symbol: str
    n_public_contributions: int = 0
    n_shared_contributions: int = 0
    n_contributors: int = 0
    n_replications: int = 0
    evidence_maturity_distribution: dict[str, int] = Field(default_factory=dict)
    #: verdict label -> count, across every distinct contribution's own
    #: registry evidence -- "MIXED" is never collapsed away; see
    #: `alpha_agent.community.visibility` for how this is built.
    verdict_distribution: dict[str, int] = Field(default_factory=dict)


class CommunityMemoryLookup(BaseModel):
    """Prompt section 8: Event/Context -> Personal + Community Memory ->
    factors/strategies/replications -> research prioritization. ``personal``
    reuses Phase 1/2's own
    :class:`~alpha_agent.alpha_memory.schemas.MechanismMemoryLookup` verbatim
    (never a second personal-memory computation); ``community`` is
    :class:`CommunityAggregateRow` or ``None`` (no shared/public evidence
    yet). ``research_prioritization_note`` is deterministic, free of any
    trade-instruction vocabulary (swept by a dedicated test mirroring Phase
    7's own), and is explicitly NEVER a guaranteed trade recommendation
    (prompt section 8)."""

    model_config = {"frozen": True, "extra": "forbid"}

    mechanism: EconomicMechanism
    root_symbol: str
    personal: dict
    community: CommunityAggregateRow | None = None
    research_prioritization_note: str = ""
