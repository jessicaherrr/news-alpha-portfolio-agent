"""Phase 1 -- the smallest typed translation model for
NEWS/MARKET OBSERVATION -> MECHANISM -> MEASURABLE VARIABLE -> FACTOR ->
RESEARCHABILITY -> FALSIFIABLE HYPOTHESIS (prompt 1).

This module is the OBSERVATION -> HYPOTHESIS translation layer's own domain
schema. It deliberately reuses existing typed concepts rather than inventing
parallel ones (prompt 1 section 2/4):

* ``EconomicMechanism`` is the SAME closed enum
  ``alpha_agent.knowledge.models.EconomicMechanism`` mechanism-diverse
  generation already uses -- never a second, independently-invented
  mechanism vocabulary.
* The Falsifiable Hypothesis this pipeline produces IS an ordinary
  ``alpha_agent.schemas.hypothesis.HypothesisSpec`` -- the same typed object
  ``ResearchAgent``/the Agent composer already understand. Phase 1 stops
  there: nothing here compiles, executes, or writes the registry.

OBSERVATION PLANE BOUNDARY (prompt 1 section 3): ``Observation`` carries a
current ``MarketNewsItem`` / ``ScheduledMarketEvent`` (or a user-described
current market state) purely as inspiration/context. It is never scientific
evidence, is never itself entered into the experiment registry or
``FailureMemory``, and its own ``observed_at`` is display-only provenance --
never scanned into a scientific payload (see ``origin_vintage_fields``,
mirroring ``alpha_agent.ui.views.agent``'s existing Opportunity/Claude-
hypothesis hand-off pattern).
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from alpha_agent.agents.context import FailureMemoryDigest
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.holdout_guard import HOLDOUT_START
from alpha_agent.schemas.hypothesis import HypothesisSpec

__all__ = [
    "CERTIFIED_ROOTS",
    "SCHEMA_VERSION",
    "FactorCandidate",
    "FactorProvenance",
    "MeasurableVariable",
    "MechanismCandidate",
    "MechanismProvenance",
    "Observation",
    "ObservationTranslation",
    "ResearchMemoryNote",
    "ResearchabilityStatus",
    "origin_vintage_fields",
]

#: Prompt 1 section 1: "Use only the current mature Futures product... Do not
#: broaden the research universe merely for this phase." Order is meaningful
#: (see ``alpha_agent.translation.pipeline._pick_certified_root``) but is not
#: itself a claim about which market matters most.
CERTIFIED_ROOTS: tuple[str, ...] = ("ES", "NQ", "CL", "GC", "ZN")

SCHEMA_VERSION = "observation-translation/1"

_HOLDOUT_START_DT = datetime.fromisoformat(HOLDOUT_START).replace(tzinfo=UTC)


def origin_vintage_fields(observed_at: datetime) -> dict:
    """Display-only post-holdout-origin marker -- identical formula to
    ``alpha_agent.ui.views.agent._origin_vintage_fields``, reproduced here
    (rather than imported from a Streamlit-rendering module) so this typed
    layer stays free of any UI dependency. ``holdout_eligible`` is honestly
    False for any observation dated on/after the locked 2025 boundary; it
    never changes what historical (2018-2024) research is available, and it
    is never used to decide whether the observation itself may be read --
    only whether a hypothesis seeded from it may honestly claim eligibility
    for the untouched 2025 holdout."""
    return {
        "origin_vintage": observed_at.date().isoformat(),
        "holdout_eligible": observed_at < _HOLDOUT_START_DT,
    }


class ResearchabilityStatus(str, Enum):
    """Prompt 1 section 7's deterministic researchability vocabulary,
    verbatim. Never inferred from prose -- always computed by
    ``alpha_agent.translation.researchability.classify_factor`` against live
    repository state (``alpha_agent.features.REGISTRY``), regardless of
    whether the factor idea itself came from the deterministic mechanism
    library or from Claude."""

    #: Every required input is already a registered FeatureRegistry kind, and
    #: nothing beyond that is needed -- this factor can be represented and
    #: executed by the existing research stack today.
    AVAILABLE = "AVAILABLE"
    #: Some required components exist (at least one registered feature kind),
    #: but the full proposed factor cannot currently be built faithfully.
    PARTIALLY_AVAILABLE = "PARTIALLY_AVAILABLE"
    #: The economic idea is measurable in principle, but the required
    #: historical point-in-time data is not currently available/ingested.
    DATA_MISSING = "DATA_MISSING"
    #: The concept (or the math) may exist, but today's StrategySpec /
    #: FeatureRegistry / execution stack cannot express it correctly (e.g. it
    #: needs a cross-instrument join no registered feature kind provides).
    NOT_EXECUTABLE = "NOT_EXECUTABLE"


class FactorProvenance(str, Enum):
    """Prompt 1 section 9's provenance vocabulary, verbatim. Distinguishes
    what the deterministic system already has from what a proposer (Claude or
    the deterministic library) merely thinks would be interesting -- never
    silently mixed."""

    REGISTERED_FEATURE = "REGISTERED_FEATURE"
    DERIVED_FROM_REGISTERED_FEATURES = "DERIVED_FROM_REGISTERED_FEATURES"
    CLAUDE_PROPOSED_PROXY = "CLAUDE_PROPOSED_PROXY"
    RESEARCH_MEMORY = "RESEARCH_MEMORY"
    UNSUPPORTED_IDEA = "UNSUPPORTED_IDEA"


class MechanismProvenance(str, Enum):
    """Where a mechanism candidate's narrative reasoning came from -- a small,
    closed vocabulary mirroring ``alpha_agent.ui.candidate_mechanisms``'s own
    ``PROVENANCE_*`` split (never merged into one undifferentiated list)."""

    DETERMINISTIC_LIBRARY = "DETERMINISTIC_LIBRARY"
    CLAUDE_PROPOSED = "CLAUDE_PROPOSED"


class Observation(BaseModel):
    """A real, current market observation or news/event item, narrowed to
    exactly what the translation pipeline needs. Prompt 1 section 3: this is
    Observation Plane data -- inspiration/context for a hypothesis, never
    itself scientific evidence."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "research-observation/1"
    event_type: str
    root_symbol: str
    affected_products: tuple[str, ...] = ()
    observed_at: datetime
    source: str
    summary: str
    structured_attributes: dict[str, str] = Field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    #: Display-only provenance (see ``origin_vintage_fields``) -- never fed
    #: into a scientific payload or the holdout guard.
    origin_vintage: str
    holdout_eligible: bool

    @model_validator(mode="after")
    def _checks(self) -> Observation:
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be UTC-aware -- never a naive datetime")
        if self.root_symbol not in CERTIFIED_ROOTS:
            raise ValueError(
                f"root_symbol {self.root_symbol!r} is outside the certified Phase 1 futures "
                f"universe {CERTIFIED_ROOTS} -- this phase does not broaden the research universe"
            )
        return self


class MechanismCandidate(BaseModel):
    """WHY could the observation matter economically? One candidate
    explanation, never presented as confirmed fact (prompt 1 section 5)."""

    model_config = {"frozen": True, "extra": "forbid"}

    mechanism: EconomicMechanism
    explanation: str
    causal_chain: tuple[str, ...] = ()
    evidence_basis: str
    provenance: MechanismProvenance


class MeasurableVariable(BaseModel):
    """WHAT observable variable would represent this mechanism
    quantitatively? ``point_in_time_available`` is always a deterministic
    read of repository state (never a proposer's own claim -- prompt 1
    section 6): ``True``/``False`` only when a typed repository capability
    (e.g. an exact registered feature/data-source identity) actually
    confirms it, ``None`` (UNKNOWN) otherwise. A Claude-proposed variable's
    own free-text ``required_source`` is NEVER scanned for availability
    markers -- prose can propose, it can never establish capability
    (Phase 1 acceptance patch, section 1)."""

    model_config = {"frozen": True, "extra": "forbid"}

    name: str
    economic_meaning: str
    required_source: str
    point_in_time_available: bool | None
    point_in_time_note: str


class FactorCandidate(BaseModel):
    """WHICH of those variables can become a testable factor? Every field
    beyond the narrative ones (``concept``/``transform_or_proxy``) is
    deterministically computed against live repository state -- see
    ``alpha_agent.translation.researchability.classify_factor``."""

    model_config = {"frozen": True, "extra": "forbid"}

    concept: str
    mechanism: EconomicMechanism
    transform_or_proxy: str
    proposed_feature_kinds: tuple[str, ...] = ()
    #: Deterministic subset of ``proposed_feature_kinds`` actually present in
    #: ``alpha_agent.features.REGISTRY`` -- ground truth, never a proposer's
    #: own claim of availability.
    available_feature_kinds: tuple[str, ...] = ()
    required_external_data: tuple[str, ...] = ()
    researchability: ResearchabilityStatus
    researchability_reason: str
    missing_requirements: tuple[str, ...] = ()
    provenance: FactorProvenance


class ResearchMemoryNote(BaseModel):
    """Prompt 1 section 10/11: what does the existing scientific memory
    already say about mechanisms related to this observation? Every count
    here is deterministic ``FailureMemory``/``ExperimentRegistry`` evidence
    (``alpha_agent.agents.context.FailureMemoryDigest``, reused verbatim --
    never a second, independent judgment).

    ``mapped_family_count`` and ``tested_family_count`` are deliberately
    distinct (Phase 1 acceptance patch, section 3): a family being MAPPED
    (this mechanism has a candidate-family representation at all) is not the
    same claim as that family having actually been TESTED (at least one real
    execution attempt exists for it on this root). Conflating the two would
    let an untested-but-mapped family understate how little evidence
    actually exists."""

    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str
    related_strategy_families: tuple[str, ...] = ()
    digests: tuple[FailureMemoryDigest, ...] = ()
    engineering_lessons: tuple[str, ...] = ()
    #: Count of mechanism-mapped candidate families (`len(related_strategy_families)`).
    mapped_family_count: int = 0
    #: Count of those mapped families with at least one real (valid or
    #: invalid) execution attempt on this root -- never just `len(mapped)`.
    tested_family_count: int = 0
    underexplored: bool
    summary: str


class ObservationTranslation(BaseModel):
    """The full, typed translation trace for one observation -- everything
    prompt 1's Agent card (section 12/18) renders, and everything a
    "Research This" hand-off preserves (section 13)."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = SCHEMA_VERSION
    observation: Observation
    mechanism_candidates: tuple[MechanismCandidate, ...]
    measurable_variables: tuple[MeasurableVariable, ...]
    factor_candidates: tuple[FactorCandidate, ...]
    research_memory: ResearchMemoryNote
    #: `None` when no factor candidate is researchable enough to support a
    #: falsifiable hypothesis yet -- an honest RESEARCH GAP (section 8), never
    #: a hallucinated testable factor.
    hypothesis: HypothesisSpec | None = None
    research_gap_note: str | None = None
    generated_by: str
    reasoning_provenance_note: str
