"""News Alpha Portfolio Agent -- the canonical research pipeline's front end.

Canonical pipeline, and where each stage lives:

    User Research Mandate        -> news_alpha.mandate        (Phase A, NEW; reuses InvestorProfile)
    Allowed Asset Universe       -> news_alpha.universe       (Phase A, NEW scope over etf/equities/marketdata universes)
    News / Events                -> alpha_agent.market_intel  (EXISTING, observation plane; news_alpha.events is a view)
    Initial Impact Scan          -> news_alpha.impact_scan    (Phase A, NEW; first pass of a two-pass design)
    -> translation boundary      -> news_alpha.handoff -> alpha_agent.translation.pipeline (EXISTING Phase 1)
    Economic Mechanism Graph     -> news_alpha.mechanism_graph (Phase B, NEW; typed graph in
                                    news_alpha.transmission, reviewed seeds in news_alpha.transmission_library)
    Signal Path Discovery        -> news_alpha.signal_paths   (Phase C, NEW; DIRECT / SUPPLY_CHAIN /
                                    CROSS_SECTOR paths ending at an economic consequence; reviewed
                                    sectors + target concepts in news_alpha.signal_path_library)
    Mechanism-adjusted impact    -> news_alpha.impact_adjustment (Phase C, NEW; second triage pass,
                                    kept beside the untouched initial scan)
    Asset Expression             -> news_alpha.asset_expression (Phase D, NEW; where each consequence
                                    could be expressed, and whether the mandate lets it continue;
                                    reviewed rules + measurement templates in news_alpha.expression_library)
    Measurement / PIT resolution -> news_alpha.measurement   (Phase D, NEW; MeasurementSpec over the
                                    Phase 1 MeasurableVariable + DataFieldResolution read from repo state)
    Candidate Signal Spec        -> news_alpha.candidate_signals (Phase E, NEW; typed SignalSpec + lineage,
                                    only through the Phase D gate; the formula is a
                                    features.expression.FactorExpression over REGISTERED feature kinds)
    Factor series + diagnostics  -> alpha_agent.screening.candidate_signal_screen / .factor_diagnostics
                                    (Phase E, NEW; the empirical SCREENING plane, outside this package)
    Multi-asset signal ranking   -> alpha_agent.recommendation.signal_ranking (Phase F, NEW; candidate
                                    + diagnostics ranked for portfolio CONSIDERATION under the mandate:
                                    research merit vs user fit, exposure groups, NOT_RANKABLE tier;
                                    outside this package -- it reads the screening plane)
    Portfolio construction ... Alpha Memory                   (later phases; not started here)

"Mechanism" in the Mechanism Graph means ECONOMIC TRANSMISSION (AI capex ->
compute demand -> HBM demand), deliberately distinct from the QUANT
mechanism vocabulary (`knowledge.models.EconomicMechanism`: TREND, CARRY,
...) that `translation` and the evidence-plane `alpha_graph` use -- see
`news_alpha.transmission`'s module docstring.

Boundaries (all test-enforced in `tests/python/test_news_alpha_phase_a.py`):

* Research triage only. No expected return, probability, BUY/SELL, weight,
  score, or verdict field exists anywhere in this package.
* The mandate scopes investigation; it never changes scientific truth.
  Risk preferences never alter an impact level.
* No registry write, no `experiment_identity` input, no network, no LLM.
  News is observation, never scientific evidence; no second news store.
* The mechanism graph is the HYPOTHESIS plane: a relationship's support
  status is computed from verified provenance, never asserted -- a model
  proposal stays PROPOSED until a source is verified.
* A signal path ends at an ECONOMIC CONSEQUENCE and a target concept, never
  an instrument or a trade; a model may only propose walks over existing
  graph links, validated or REJECTED with typed reasons.
* A candidate signal is a testable factor hypothesis built only from a real,
  point-in-time safe, executable measurement -- never a validated factor, and
  never read off news text, an LLM, or `execution_capable` alone.
* An asset expression names where a consequence COULD be expressed; economic
  relevance is mandate-independent, admission is the mandate's. A measurement
  is usable on history only when real, point-in-time safe and executable --
  synthetic data and unsupported domains never are.
"""
from __future__ import annotations

from alpha_agent.news_alpha.asset_expression import (
    AssetExpression,
    AssetExpressionPlan,
    CandidateSignalPrerequisiteError,
    DomainExpressionSummary,
    ExpressionPressure,
    ExpressionStatus,
    MeasurementBlocker,
    build_asset_expressions,
)
from alpha_agent.news_alpha.candidate_signals import (
    CandidateNote,
    CandidateOrigin,
    CandidateRefusal,
    CandidateSignal,
    CandidateSignalSet,
    EventConditioning,
    ExpectedRelationship,
    RefusalReason,
    SignalHorizon,
    SignalSpec,
    build_candidate_signals,
)
from alpha_agent.news_alpha.channels import (
    CHANNELS,
    EconomicChannel,
    ExposureStrength,
    ImpactHorizon,
    PressureSign,
)
from alpha_agent.news_alpha.events import (
    ImpactEvent,
    ImpactEventKind,
    UserDescribedEvent,
    impact_event_from,
)
from alpha_agent.news_alpha.expression_library import (
    DEFAULT_EXPRESSION_LIBRARY,
    DataRequirement,
    ExpressionFidelity,
    ExpressionForm,
    ExpressionLibrary,
    ExpressionRule,
    MeasurementRole,
    MeasurementTemplate,
)
from alpha_agent.news_alpha.handoff import (
    CategoryBasis,
    TranslationGap,
    TranslationHandoff,
    TranslationHandoffPlan,
    build_translation_handoffs,
)
from alpha_agent.news_alpha.impact_adjustment import (
    ExposureRevision,
    LevelChange,
    MechanismAdjustedImpact,
    MechanismAdjustedImpactAssessment,
    MechanismSupport,
    adjust_impact,
)
from alpha_agent.news_alpha.impact_scan import scan_initial_impact, scan_many
from alpha_agent.news_alpha.mandate import (
    DEFAULT_MANDATE,
    DOMAIN_LABELS,
    LiquidityRequirement,
    MandateDomain,
    MandateStore,
    ResearchMandate,
    mandate_domain_for,
    registry_domain_for,
)
from alpha_agent.news_alpha.measurement import (
    DataFieldResolution,
    DataFrequency,
    MeasurementSpec,
    PublicationLag,
    ResolutionGap,
    ResolutionIssue,
    ResolutionStatus,
    resolve_field,
)
from alpha_agent.news_alpha.mechanism_graph import (
    EconomicMechanismGraph,
    GraphAnchor,
    ImpliedMovement,
    StateImplication,
    TransmissionPath,
    UnseededChannel,
    build_economic_mechanism_graph,
)
from alpha_agent.news_alpha.schemas import (
    AvailabilityStatus,
    CandidateMarket,
    ChannelEvidence,
    DetectionStrength,
    DirectionalPressure,
    ExposureContribution,
    HorizonAlignment,
    ImpactLevel,
    InitialImpactAssessment,
    InitialImpactScan,
    TriageConfidence,
)
from alpha_agent.news_alpha.signal_path_library import (
    DEFAULT_PATH_LIBRARY,
    EconomicSector,
    SignalPathLibrary,
)
from alpha_agent.news_alpha.signal_paths import (
    PathConflict,
    PathStatus,
    PathType,
    RejectedPathProposal,
    SignalPath,
    SignalPathDiscovery,
    SignalPathProposal,
    StatusReason,
    discover_signal_paths,
)
from alpha_agent.news_alpha.transmission import (
    EconomicState,
    EdgeOrigin,
    GraphIssue,
    GraphIssueKind,
    LinkConfidence,
    Movement,
    Polarity,
    ProvenanceSource,
    SupportStatus,
    TransmissionChannel,
    TransmissionClaim,
    TransmissionEdge,
    TransmissionGraph,
    TransmissionLag,
    TransmissionProposal,
    assemble_graph,
    claims_from_proposal,
)
from alpha_agent.news_alpha.transmission_library import DEFAULT_LIBRARY, TransmissionLibrary
from alpha_agent.news_alpha.universe import (
    AllowedAssetUniverse,
    DomainCapabilities,
    DomainScope,
    DomainSupport,
    FuturesBarCoverage,
    UniverseInstrument,
    probe_domain_capabilities,
    resolve_allowed_universe,
)

__all__ = [
    "CHANNELS",
    "DEFAULT_EXPRESSION_LIBRARY",
    "DEFAULT_LIBRARY",
    "DEFAULT_MANDATE",
    "DEFAULT_PATH_LIBRARY",
    "DOMAIN_LABELS",
    "AllowedAssetUniverse",
    "AssetExpression",
    "AssetExpressionPlan",
    "AvailabilityStatus",
    "CandidateMarket",
    "CandidateNote",
    "CandidateOrigin",
    "CandidateRefusal",
    "CandidateSignal",
    "CandidateSignalPrerequisiteError",
    "CandidateSignalSet",
    "CategoryBasis",
    "ChannelEvidence",
    "DataFieldResolution",
    "DataFrequency",
    "DataRequirement",
    "DetectionStrength",
    "DirectionalPressure",
    "DomainCapabilities",
    "DomainExpressionSummary",
    "DomainScope",
    "DomainSupport",
    "EconomicChannel",
    "EconomicMechanismGraph",
    "EconomicSector",
    "EconomicState",
    "EdgeOrigin",
    "EventConditioning",
    "ExpectedRelationship",
    "ExposureContribution",
    "ExposureRevision",
    "ExposureStrength",
    "ExpressionFidelity",
    "ExpressionForm",
    "ExpressionLibrary",
    "ExpressionPressure",
    "ExpressionRule",
    "ExpressionStatus",
    "FuturesBarCoverage",
    "GraphAnchor",
    "GraphIssue",
    "GraphIssueKind",
    "HorizonAlignment",
    "ImpactEvent",
    "ImpactEventKind",
    "ImpactHorizon",
    "ImpactLevel",
    "ImpliedMovement",
    "InitialImpactAssessment",
    "InitialImpactScan",
    "LevelChange",
    "LinkConfidence",
    "LiquidityRequirement",
    "MandateDomain",
    "MandateStore",
    "MeasurementBlocker",
    "MeasurementRole",
    "MeasurementSpec",
    "MeasurementTemplate",
    "MechanismAdjustedImpact",
    "MechanismAdjustedImpactAssessment",
    "MechanismSupport",
    "Movement",
    "PathConflict",
    "PathStatus",
    "PathType",
    "Polarity",
    "PressureSign",
    "ProvenanceSource",
    "PublicationLag",
    "RefusalReason",
    "RejectedPathProposal",
    "ResearchMandate",
    "ResolutionGap",
    "ResolutionIssue",
    "ResolutionStatus",
    "SignalHorizon",
    "SignalPath",
    "SignalPathDiscovery",
    "SignalPathLibrary",
    "SignalPathProposal",
    "SignalSpec",
    "StateImplication",
    "StatusReason",
    "SupportStatus",
    "TranslationGap",
    "TranslationHandoff",
    "TranslationHandoffPlan",
    "TransmissionChannel",
    "TransmissionClaim",
    "TransmissionEdge",
    "TransmissionGraph",
    "TransmissionLag",
    "TransmissionLibrary",
    "TransmissionPath",
    "TransmissionProposal",
    "TriageConfidence",
    "UniverseInstrument",
    "UnseededChannel",
    "UserDescribedEvent",
    "adjust_impact",
    "assemble_graph",
    "build_asset_expressions",
    "build_candidate_signals",
    "build_economic_mechanism_graph",
    "build_translation_handoffs",
    "claims_from_proposal",
    "discover_signal_paths",
    "impact_event_from",
    "mandate_domain_for",
    "probe_domain_capabilities",
    "registry_domain_for",
    "resolve_allowed_universe",
    "resolve_field",
    "scan_initial_impact",
    "scan_many",
]
