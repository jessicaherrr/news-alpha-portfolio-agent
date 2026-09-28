"""Phase 7 -- the typed Cross-Asset Alpha Graph read model (prompt 7).

A RESEARCH RELATIONSHIP / NAVIGATION layer over existing sources of truth
(:mod:`alpha_agent.registry`, :mod:`alpha_agent.alpha_memory`,
:mod:`alpha_agent.etf.alpha_memory_bridge`,
:mod:`alpha_agent.translation.mechanism_library`) -- never a second
scientific database, never a graph database, never an Alpha Score.

Node types (prompt section 2), deliberately small: Event, Mechanism, Factor,
Instrument, Strategy, Experiment. Evidence stays attached to
:class:`InstrumentEvidence` / the existing
:class:`~alpha_agent.alpha_memory.schemas.AlphaResearchObject` rather than
becoming a decorative standalone node -- Strategy/Experiment detail is never
duplicated here; :attr:`InstrumentEvidence.alpha_ids` points back at the real
:class:`~alpha_agent.alpha_memory.schemas.AlphaResearchObject` (Factor Library
detail view) for that.

SCIENTIFIC BOUNDARY (prompt section 4): cross-asset evidence is RELATED
evidence, never transferred proof. Every :class:`InstrumentEvidence` keeps its
OWN ``scientific_evidence``/``research_maturity`` -- this module never
computes a combined/blended verdict across instruments or asset domains, and
never invents an Alpha Score or graph score (section 6/9). Absence of evidence
is surfaced as :attr:`EvidenceCoverage.UNDEREXPLORED` /
:attr:`EvidenceCoverage.NO_EVIDENCE`, never as a failure.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from alpha_agent.alpha_memory.schemas import FactorIdentity
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.market_intel.news_schemas import NewsCategory
from alpha_agent.registry.enums import AssetDomain

__all__ = [
    "ALPHA_GRAPH_SCHEMA",
    "CrossAssetSynthesis",
    "EventCategoryRef",
    "EvidenceCoverage",
    "FactorUnderMechanism",
    "GraphNodeType",
    "InstrumentEvidence",
    "InstrumentRef",
    "MechanismGraphSummary",
    "MechanismGraphView",
    "ResearchGapRow",
]

#: v1 -- first cut of the Phase 7 Cross-Asset Alpha Graph read model. Nothing
#: here is persisted outside one call, so no migration path is needed; a
#: schema bump only matters for a caller pinning today's field shape.
ALPHA_GRAPH_SCHEMA = "alpha-graph/1"


class GraphNodeType(str, Enum):
    """The minimal node vocabulary (prompt section 2). Not every node type
    is materialized as its own standalone schema below -- Strategy and
    Experiment detail already live on
    :class:`~alpha_agent.alpha_memory.schemas.AlphaResearchObject`
    (``strategy_variants`` / ``experiments``), reachable via
    :attr:`InstrumentEvidence.alpha_ids`; duplicating them here would be a
    second scientific database (prompt section 1)."""

    EVENT = "EVENT"
    MECHANISM = "MECHANISM"
    FACTOR = "FACTOR"
    INSTRUMENT = "INSTRUMENT"
    STRATEGY = "STRATEGY"
    EXPERIMENT = "EXPERIMENT"


class EvidenceCoverage(str, Enum):
    """How much REAL registry evidence exists for one (Mechanism, Instrument)
    pair today -- never how promising or profitable it is (prompt section 6:
    "do not treat absence of evidence as negative evidence"). Computed purely
    from already-existing facts:

    * whether the mechanism maps to a real candidate strategy family for that
      instrument's asset domain at all (
      :mod:`alpha_agent.translation.research_memory`'s frozen Futures bridge,
      or :mod:`alpha_agent.etf.alpha_memory_bridge`'s ETF bridge), and
    * whether any real :class:`~alpha_agent.alpha_memory.schemas.AlphaResearchObject`
      materializes for it (:mod:`alpha_agent.alpha_memory.builder`'s own
      ``research_maturity`` ladder).
    """

    #: >=1 real AlphaResearchObject with research_maturity REPLICATED or
    #: ADJUDICATED (multiple real experiments and/or a canonical headline
    #: verdict has been reached) -- whatever that verdict says.
    RESEARCHED = "RESEARCHED"
    #: >=1 real AlphaResearchObject exists, but the deepest evidence is a
    #: single real execution attempt (research_maturity TESTED) -- early,
    #: not yet adjudicated.
    WEAKLY_RESEARCHED = "WEAKLY_RESEARCHED"
    #: the mechanism IS mapped to a real candidate strategy family for this
    #: instrument's asset domain (structurally researchable today), but zero
    #: real registry experiments exist for it on this instrument yet.
    UNDEREXPLORED = "UNDEREXPLORED"
    #: the mechanism has no mapped candidate strategy family at all for this
    #: instrument's asset domain -- not structurally represented today,
    #: genuinely nothing to show (not merely "not yet tried").
    NO_EVIDENCE = "NO_EVIDENCE"


class InstrumentRef(BaseModel):
    """One Instrument node: a root/ticker plus which research domain it
    belongs to. Futures and ETF symbols never collide (disjoint namespaces),
    but the domain is carried explicitly rather than inferred from spelling."""

    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str
    asset_domain: AssetDomain


class EventCategoryRef(BaseModel):
    """One Event node: a real, frozen ``NewsCategory`` -> Mechanism edge from
    Phase 1's already-committed, deterministic, no-network mechanism library
    (:mod:`alpha_agent.translation.mechanism_library`). Never a live news
    fetch -- browsing the graph must never make a network call (prompt
    section 10)."""

    model_config = {"frozen": True, "extra": "forbid"}

    category: NewsCategory
    evidence_basis: str


class FactorUnderMechanism(BaseModel):
    """One Factor node under a Mechanism: a real, deterministic
    :class:`~alpha_agent.alpha_memory.schemas.FactorIdentity`, computed the
    same way :mod:`alpha_agent.alpha_memory.factor_identity` always does --
    this is a pure function of (mechanism, strategy family), so it exists
    (as a structural possibility) even before any experiment has run.
    ``instruments_with_evidence`` is the real, current set of instruments
    carrying >=1 registry experiment for this exact Factor."""

    model_config = {"frozen": True, "extra": "forbid"}

    factor: FactorIdentity
    asset_domain: AssetDomain
    instruments_with_evidence: tuple[InstrumentRef, ...] = ()


class InstrumentEvidence(BaseModel):
    """One Mechanism -> Instrument edge's evidence, kept entirely
    experiment-specific (prompt section 4: "cross-asset evidence is related
    evidence, never transferred proof"). ``scientific_evidence`` and
    ``repeated_failure_*`` are copied verbatim from this instrument's OWN
    real :class:`~alpha_agent.alpha_memory.schemas.AlphaResearchObject`
    entries -- never merged with another instrument's, never averaged into
    one score."""

    model_config = {"frozen": True, "extra": "forbid"}

    instrument: InstrumentRef
    coverage: EvidenceCoverage
    #: real AlphaResearchObject ids materialized for this (mechanism,
    #: instrument) -- the click-through target into the existing Factor
    #: Library detail view; never re-rendered here.
    alpha_ids: tuple[str, ...] = ()
    research_maturity: str | None = None
    #: strategy_family -> canonical-trial verdict label, unioned verbatim
    #: from this instrument's own AlphaResearchObject(s); absent when
    #: coverage is UNDEREXPLORED/NO_EVIDENCE.
    scientific_evidence: dict[str, str] = Field(default_factory=dict)
    repeated_failure_reason_codes: dict[str, int] = Field(default_factory=dict)
    repeated_failure_classes: dict[str, int] = Field(default_factory=dict)


class MechanismGraphView(BaseModel):
    """The full navigable view for one Mechanism node: Event provenance,
    the Factor(s) that represent it, every considered Instrument's evidence
    (across BOTH asset domains), and the aggregate research-gap / repeated-
    failure facts a reader needs -- all read straight off existing evidence,
    never a new score (prompt sections 2/3/6)."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = ALPHA_GRAPH_SCHEMA
    mechanism: EconomicMechanism
    event_categories: tuple[EventCategoryRef, ...] = ()
    factors: tuple[FactorUnderMechanism, ...] = ()
    instrument_evidence: tuple[InstrumentEvidence, ...] = ()
    #: asset domains with >=1 instrument at RESEARCHED or WEAKLY_RESEARCHED
    #: coverage -- real, current cross-asset confirmation breadth, never a
    #: score.
    domains_with_evidence: tuple[AssetDomain, ...] = ()
    #: reason code -> total count, summed ACROSS every instrument's own
    #: AlphaResearchObject.repeated_failure_reason_codes -- an honest
    #: cross-asset aggregate, always traceable back to
    #: `instrument_evidence` (never presented without it).
    repeated_failure_reason_codes: dict[str, int] = Field(default_factory=dict)
    repeated_failure_classes: dict[str, int] = Field(default_factory=dict)


class MechanismGraphSummary(BaseModel):
    """One compact landing-table row per Mechanism -- counts only, no new
    score. Derived purely from `MechanismGraphView.instrument_evidence`."""

    model_config = {"frozen": True, "extra": "forbid"}

    mechanism: EconomicMechanism
    n_researched: int = 0
    n_weakly_researched: int = 0
    n_underexplored: int = 0
    n_no_evidence: int = 0
    domains_with_evidence: tuple[AssetDomain, ...] = ()
    is_cross_asset: bool = False


class ResearchGapRow(BaseModel):
    """One flattened (Mechanism, Instrument) research gap -- UNDEREXPLORED
    or NO_EVIDENCE coverage only. First-class, never hidden or treated as a
    failure (prompt section 6)."""

    model_config = {"frozen": True, "extra": "forbid"}

    mechanism: EconomicMechanism
    instrument: InstrumentRef
    coverage: EvidenceCoverage


class CrossAssetSynthesis(BaseModel):
    """The deterministic, Agent-facing synthesis for one Mechanism (prompt
    section 5/8): "what is common, what differs, repeated failures, what
    remains underexplored" -- built entirely from already-computed
    `MechanismGraphView` fields, never a live LLM call, never a merged
    verdict, and never phrased as a trade instruction (see this package's
    ``retrieval.py`` module docstring and its dedicated regression tests)."""

    model_config = {"frozen": True, "extra": "forbid"}

    mechanism: EconomicMechanism
    common: tuple[str, ...] = ()
    differs: tuple[str, ...] = ()
    repeated_failures: tuple[str, ...] = ()
    underexplored: tuple[str, ...] = ()
    note: str = (
        "Related evidence across instruments is context for research prioritization only -- a "
        "verdict on one instrument is never transferred to another (CLAUDE.md / prompt 7 section 4)."
    )
