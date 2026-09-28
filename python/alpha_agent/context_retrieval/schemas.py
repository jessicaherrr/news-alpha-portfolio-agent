"""Phase 3 typed schemas.

``MarketContextFingerprint`` is built from AVAILABLE OBSERVATIONS ONLY
(prompt section 1) -- every field is a direct read of an already-established
deterministic accessor (``alpha_agent.ui.market_home``/``market_relative``
for trend/volatility/curve/related-market confirmation,
``alpha_agent.market_intel.importance`` for CATEGORY IMPORTANCE), never
free-text headline similarity (prompt section 2). ``fingerprint_hash`` is
computed over DISCRETE, reproducible fields only -- see
:mod:`alpha_agent.context_retrieval.fingerprint`.

EVENT IMPORTANCE != OBJECTIVE EVENT MAGNITUDE (Phase 3 semantic hardening
patch, section 3): ``event_importance`` is
``alpha_agent.market_intel.event_schemas.EventImportance`` -- a CATEGORY-
level classification (every PETROLEUM release is HIGH, regardless of how big
the actual inventory draw was), never itself a measured surprise/magnitude.
``objective_event_magnitude`` is a SEPARATE field that stays honestly
``"UNKNOWN"`` unless a real, typed, structured magnitude value (e.g. an
inventory-change surprise, a CPI surprise, a payroll surprise) is present on
the observation -- no such source is ingested anywhere in this repository
today (see ``alpha_agent.translation.mechanism_library``'s own
``_RELEASE_SURPRISE_VAR``/``_CONSENSUS_SURPRISE_VAR``, both
``point_in_time_available=False``), so this field is always ``"UNKNOWN"`` in
V1. Never fabricated by parsing free-text headlines.

Six dimensions stay architecturally separate everywhere in this package
(prompt section 4: "Keep separate: Scientific Reliability; Context Match;
Execution Robustness; Independent Replication; User Fit; Research
Coverage") -- :class:`RankDimensionScores` never collapses them into one
blended number, mirroring
``alpha_agent.alpha_memory.schemas.EvidenceProfile``'s own "never combined
into one number" discipline. ``RETRIEVAL_RANK_POLICY`` names the one place
an ORDER is allowed to emerge (prompt section 5): a documented, versioned,
deterministic sort over these six dimensions for RESEARCH PRIORITIZATION
ONLY -- never expected return, probability of success, or trade confidence.
There is deliberately no ``confidence``/``probability``/``expected_return``/
``score`` field anywhere on this schema (mirrors
``alpha_agent.opportunity.schemas.FORBIDDEN_FIELD_NAMES``).

INDEPENDENT REPLICATION != RELATED EXPERIMENT COUNT (semantic hardening
patch, section 2): Phase 2's own ``alpha_memory.EvidenceProfile`` docstring
already established that ``related_experiment_count`` (canonical + neighbour
+ ablation + re-execution rows) is coverage of the SAME hypothesis, never
independent reproduction of a result -- "independent_replication" is
reserved terminology for a future phase with a real evidence source for it.
:class:`IndependentReplicationLevel` is therefore always ``NOT_AVAILABLE``
in V1; the real per-family related-experiment counts stay visible, but only
under Research Coverage (``RankDimensionScores.related_experiment_counts``),
never relabeled as replication.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.translation.schemas import FactorCandidate, MechanismCandidate

__all__ = [
    "CONTEXT_FINGERPRINT_SCHEMA",
    "RETRIEVAL_RANK_POLICY",
    "RETRIEVAL_RESULT_SCHEMA",
    "UNKNOWN_OBJECTIVE_MAGNITUDE",
    "ContextMatchLevel",
    "ContextRetrievalResult",
    "FreshnessBucket",
    "IndependentReplicationLevel",
    "MarketContextFingerprint",
    "NextExperimentSuggestion",
    "RankDimensionScores",
    "RankedResearchCandidate",
    "RelatedPriorEventOccurrence",
    "ResearchCoverageLevel",
    "ScientificReliabilityLevel",
]

CONTEXT_FINGERPRINT_SCHEMA = "market-context-fingerprint/2"
RETRIEVAL_RESULT_SCHEMA = "context-aware-retrieval/2"

#: The honest default for `MarketContextFingerprint.objective_event_magnitude`
#: -- see the module docstring's "EVENT IMPORTANCE != OBJECTIVE EVENT
#: MAGNITUDE" section. Never silently replaced with a category-importance
#: value or a value parsed from free text.
UNKNOWN_OBJECTIVE_MAGNITUDE = "UNKNOWN"

#: The declared, versioned priority order candidates are sorted in (see
#: ``ranking.rank_key``): Context Match, then Scientific Reliability, then
#: Research Coverage, then Execution Robustness, then Independent
#: Replication, then User Fit, then mechanism name as a final deterministic
#: tiebreak. Independent Replication is always ``NOT_AVAILABLE`` in V1 (see
#: module docstring), so it never actually discriminates between candidates
#: today -- it stays in the declared order for forward-compatibility with a
#: future phase that supplies real independent-replication evidence. This is
#: a RESEARCH-PRIORITIZATION ordering only -- changing it is a visible,
#: reviewable change (mirrors ``alpha_agent.registry.similarity.WEIGHTS``'s
#: own "never a hidden tuning knob" discipline), never a claim about which
#: candidate will make money.
RETRIEVAL_RANK_POLICY = "context-match-first/1"


class FreshnessBucket(str, Enum):
    """How long ago the observation was made, bucketed so the fingerprint
    hash stays stable across nearby evaluation times (never raw continuous
    seconds -- see ``fingerprint.build_market_context_fingerprint``)."""

    FRESH = "FRESH"      # <= 6 hours
    RECENT = "RECENT"    # <= 3 days
    STALE = "STALE"      # <= 30 days
    OLD = "OLD"          # > 30 days


class ContextMatchLevel(str, Enum):
    """How much of a mechanism's own declared context signals
    (``mechanism_signals.MECHANISM_CONTEXT_SIGNALS``) are actually
    informative in the CURRENT fingerprint. Never a probability, never a
    percentage presented as confidence."""

    NONE = "NONE"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class ScientificReliabilityLevel(str, Enum):
    """A CONSERVATIVE, meaning-preserving reduction of the real per-family
    verdicts already committed to the registry (``RegistryVerdict``, via
    ``alpha_memory.EvidenceProfile.scientific_evidence``) -- never a new
    scientific judgment, and never a collapse that changes what a committed
    verdict meant (semantic hardening patch, section 1).

    ``INCONCLUSIVE`` is its OWN level -- it must never be folded into
    ``REJECTED`` (an inconclusive result is not a rejection). ``MIXED``
    means MORE THAN ONE distinct real verdict (``PASS``/``REJECT``/
    ``INCONCLUSIVE``) was found across this mechanism's mapped families, OR
    one family's own canonical rows already disagree
    (``scientific_evidence``'s own ``"MIXED(...)"`` string) -- the reduction
    NEVER cherry-picks the most favorable verdict when families disagree
    (the old ``PASS + REJECT -> PASSED`` bug this patch fixes).
    ``PASSED``/``REJECTED`` here mean "every real verdict found among this
    mechanism's mapped families was PASS" / "...was REJECT" -- never a
    statement about this specific candidate's own untested factors, and
    never chosen over a genuine disagreement."""

    NO_EVIDENCE = "NO_EVIDENCE"
    NOT_ADJUDICATED = "NOT_ADJUDICATED"
    INCONCLUSIVE = "INCONCLUSIVE"
    REJECTED = "REJECTED"
    MIXED = "MIXED"
    PASSED = "PASSED"


class IndependentReplicationLevel(str, Enum):
    """Always ``NOT_AVAILABLE`` in V1 -- see module docstring's "INDEPENDENT
    REPLICATION != RELATED EXPERIMENT COUNT" section. No real independent-
    replication evidence source (a different researcher/dataset vintage
    reproducing the same result) is ingested anywhere in this repository
    today; a neighbour/ablation/re-execution of the SAME hypothesis is
    Research Coverage, never this."""

    NOT_AVAILABLE = "NOT_AVAILABLE"


class ResearchCoverageLevel(str, Enum):
    """Phase 1's own ``mapped_family_count``/``tested_family_count``
    distinction, bucketed (``ResearchMemoryNote``) -- how much of the
    candidate-family territory for this mechanism has been mapped, and how
    much of that has actually been tested at least once."""

    NOT_MAPPED = "NOT_MAPPED"
    MAPPED_UNTESTED = "MAPPED_UNTESTED"
    PARTIALLY_TESTED = "PARTIALLY_TESTED"
    FULLY_TESTED = "FULLY_TESTED"


class MarketContextFingerprint(BaseModel):
    """A transparent, deterministic, versioned snapshot of the current
    observation plane -- see module docstring. Every field beyond
    ``fingerprint_hash``/``component_detail`` is independently inspectable;
    the hash is provided only as a stable retrieval/dedup key, never as the
    thing a reader is asked to trust blindly."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = CONTEXT_FINGERPRINT_SCHEMA
    root_symbol: str
    event_type: str
    event_category: str
    affected_products: tuple[str, ...] = ()

    #: ``alpha_agent.market_intel.event_schemas.EventImportance`` value, and
    #: the deterministic rule id that produced it
    #: (``alpha_agent.market_intel.importance.importance_for_category``) --
    #: a CATEGORY-level classification (objective and never an LLM
    #: judgment), but NOT itself a measured event magnitude/surprise -- see
    #: ``objective_event_magnitude`` and the module docstring.
    event_importance: str
    event_importance_rule: str

    #: A real, typed, structured magnitude (e.g. an inventory-change
    #: surprise, a CPI surprise, a payroll surprise) when the observation
    #: actually carries one -- honestly ``UNKNOWN_OBJECTIVE_MAGNITUDE``
    #: otherwise (always, in V1: no such source is ingested anywhere in this
    #: repository yet). NEVER derived from ``event_importance`` and NEVER
    #: parsed from a free-text headline.
    objective_event_magnitude: str = UNKNOWN_OBJECTIVE_MAGNITUDE
    objective_event_magnitude_detail: str = ""

    #: ``alpha_agent.ui.market_home.trend_state``/``volatility_state`` on
    #: real, already-fetched recent bars for ``root_symbol`` -- descriptive
    #: only, never a trading signal.
    trend: str
    volatility: str
    #: ``alpha_agent.marketdata.databento_schemas.CurveShape`` label, or
    #: `None` when a term-structure read was not available.
    curve_state: str | None

    #: How many of ``root_symbol``'s declared related-market peers moved the
    #: same direction over the same window (``alpha_agent.ui.market_relative``)
    #: -- `(0, 0)` when the root has no clear trend or no declared peer group.
    related_market_confirming: int
    related_market_total: int

    freshness_seconds: float
    freshness_bucket: FreshnessBucket

    observed_at: datetime
    evaluated_at: datetime

    #: SHA-256 fingerprint over the DISCRETE fields above (never raw
    #: ``freshness_seconds``, which would make the hash different on every
    #: call) -- see ``fingerprint.build_market_context_fingerprint``.
    fingerprint_hash: str
    #: The exact `key=value` pairs that entered the hash, in hash order --
    #: full transparency, never a black box.
    component_detail: tuple[str, ...] = ()


class RankDimensionScores(BaseModel):
    """The six dimensions kept separate throughout this package (module
    docstring). Every field is a small, closed, explainable label or a
    transparent per-family detail dict -- never a blended number."""

    model_config = {"frozen": True, "extra": "forbid"}

    context_match: ContextMatchLevel
    context_match_reasons: tuple[str, ...] = ()

    scientific_reliability: ScientificReliabilityLevel
    #: strategy_family -> verdict label (verbatim from
    #: ``alpha_memory.EvidenceProfile.scientific_evidence``), merged across
    #: every family this mechanism maps to. This is the ONLY source
    #: ``scientific_reliability`` is reduced from -- see
    #: ``ScientificReliabilityLevel`` for the (meaning-preserving) reduction
    #: rule.
    scientific_reliability_detail: dict[str, str] = Field(default_factory=dict)

    #: strategy_family -> "NOT_EVALUATED" | "EVALUATED" | "FAILED_STRESS"
    #: (verbatim from ``alpha_memory.EvidenceProfile.cost_robustness``).
    execution_robustness_detail: dict[str, str] = Field(default_factory=dict)

    #: Always ``NOT_AVAILABLE`` in V1 -- see ``IndependentReplicationLevel``.
    independent_replication: IndependentReplicationLevel

    #: strategy_family -> related_experiment_count (verbatim from
    #: ``alpha_memory.EvidenceProfile.related_experiment_count``). This is
    #: RESEARCH COVERAGE evidence (how many related registry rows exist for
    #: this family), NOT independent replication -- see
    #: ``IndependentReplicationLevel`` and the module docstring.
    related_experiment_counts: dict[str, int] = Field(default_factory=dict)

    #: strategy_family -> "{personalization_state}:{label}", from
    #: ``alpha_agent.recommendation.fit.score_user_fit`` against that
    #: family's own canonical trial and the caller's ``InvestorProfile`` --
    #: reused verbatim, never re-scored. Empty when no family has a
    #: canonical trial to score against.
    user_fit_detail: dict[str, str] = Field(default_factory=dict)

    research_coverage: ResearchCoverageLevel
    research_coverage_detail: str = ""


class RankedResearchCandidate(BaseModel):
    """One (mechanism, real evidence) candidate this context retrieved --
    the "Related Mechanisms" / "Relevant Factors" / "Strategy Evidence" /
    "Repeated Failures" / "Research Gaps" bundle for one mechanism, plus its
    six-dimension score and its position in the deterministic rank."""

    model_config = {"frozen": True, "extra": "forbid"}

    mechanism: EconomicMechanism
    mechanism_candidate: MechanismCandidate
    factors: tuple[FactorCandidate, ...] = ()
    strategy_families: tuple[str, ...] = ()
    #: ``alpha_memory.AlphaResearchObject.alpha_id`` values -- the UI's link
    #: out to the "Factor Library" for the full evidence detail.
    alpha_ids: tuple[str, ...] = ()
    strategy_evidence_summary: tuple[str, ...] = ()
    repeated_failure_reason_codes: dict[str, int] = Field(default_factory=dict)
    repeated_failure_classes: dict[str, int] = Field(default_factory=dict)
    research_gap_notes: tuple[str, ...] = ()

    scores: RankDimensionScores
    #: The readable (positive, "higher is better") ordinal reduction that
    #: actually decided this candidate's sort position -- see
    #: ``ranking.rank_key``. Exposed so "#1 vs #2" is independently
    #: checkable, never a claim the reader must trust blindly.
    rank_key_components: tuple[int, ...] = ()
    #: 1-based position after sorting by ``RETRIEVAL_RANK_POLICY`` --
    #: research-prioritization ORDER only, never a score, confidence, or
    #: predicted return (prompt section 5).
    retrieval_rank: int


class RelatedPriorEventOccurrence(BaseModel):
    """One real, past event of the SAME (root, event category) -- e.g. a
    prior week's EIA petroleum release -- proving that this KIND of event
    genuinely recurs, demonstrated with real timestamps, never a fabricated
    recurrence.

    NOT a claim of "same historical context" or "matched historical
    context" (semantic hardening patch, section 5): this proves only
    root + event-category recurrence. It does NOT reconstruct the
    historical trend/volatility/curve/related-market fingerprint at that
    past timestamp -- no historical Databento context reconstruction exists
    in this phase. Render this concept as "RELATED PRIOR EVENT OCCURRENCE",
    never as a historical market-context match."""

    model_config = {"frozen": True, "extra": "forbid"}

    source: str
    reference_id: str
    headline: str
    observed_at: datetime
    match_reason: str


class NextExperimentSuggestion(BaseModel):
    """"Next Experiment" means highest INFORMATION/RESEARCH VALUE, never
    predicted profit (prompt section 6) -- deliberately never built from
    ``scientific_reliability`` (re-confirming an already-adjudicated result
    has near-zero new information value); see
    ``ranking.suggest_next_experiment``."""

    model_config = {"frozen": True, "extra": "forbid"}

    mechanism: EconomicMechanism
    concept: str | None = None
    rationale: str
    information_value_reason: str


class ContextRetrievalResult(BaseModel):
    """The full Agent output bundle (prompt section 6): Current Context;
    Related Mechanisms; Relevant Factors; Strategy Evidence; Repeated
    Failures; Research Gaps; Suggested Next Experiment."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = RETRIEVAL_RESULT_SCHEMA
    rank_policy: str = RETRIEVAL_RANK_POLICY
    context: MarketContextFingerprint
    #: See ``RelatedPriorEventOccurrence`` -- root+category recurrence only,
    #: never a claim of historical market-context reconstruction.
    related_prior_event_occurrences: tuple[RelatedPriorEventOccurrence, ...] = ()
    ranked_candidates: tuple[RankedResearchCandidate, ...] = ()
    #: Explains #1 vs #2 (prompt section 7's acceptance requirement) --
    #: empty only when fewer than two candidates were retrieved.
    rank_explanation: str = ""
    evidence_gaps: tuple[str, ...] = ()
    suggested_next_experiment: NextExperimentSuggestion | None = None
    generated_by: str = "DETERMINISTIC_LIBRARY"
