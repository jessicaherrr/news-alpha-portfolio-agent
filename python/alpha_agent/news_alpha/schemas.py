"""News Alpha Phase A -- the typed INITIAL IMPACT ASSESSMENT.

RESEARCH TRIAGE ONLY. An `InitialImpactAssessment` answers one question:
"given this user's mandate, does this news/event deserve deeper research in
this asset domain, and where should that research start?" It is NOT an
expected return, a probability of profit, a BUY/SELL instruction, a portfolio
weight, or a scientific verdict -- none of those fields exist on any schema
here (test-enforced), and there is deliberately no numeric score: every
level/confidence is a closed categorical value with a stated rule.

TWO-PASS DESIGN. This is the FIRST pass (``assessment_pass="INITIAL"``),
computed from the event text/category and a static channel table before any
mechanism reasoning exists. The `MechanismAdjustedImpactAssessment`
(`news_alpha.impact_adjustment`, Phase C), built after the Mechanism Graph
and its signal paths, supersedes it as a SEPARATE object -- the initial scan
is never modified and never claims to be final (see
`InitialImpactScan.not_final_note`).
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel

from alpha_agent.news_alpha.channels import (
    EconomicChannel,
    ExposureStrength,
    ImpactHorizon,
    PressureSign,
)
from alpha_agent.news_alpha.events import ImpactEvent
from alpha_agent.news_alpha.mandate import MandateDomain

__all__ = [
    "SCAN_SCHEMA_VERSION",
    "AvailabilityStatus",
    "CandidateMarket",
    "ChannelEvidence",
    "DetectionBasis",
    "DetectionStrength",
    "DirectionalPressure",
    "ExposureContribution",
    "HorizonAlignment",
    "ImpactLevel",
    "InitialImpactAssessment",
    "InitialImpactScan",
    "TriageConfidence",
]

SCAN_SCHEMA_VERSION = "initial-impact-scan/1"
GENERATED_BY = "DETERMINISTIC_CHANNEL_TABLE"


class ImpactLevel(str, Enum):
    """Research-triage relevance, ordered. Never a return or probability."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

    @property
    def rank(self) -> int:
        return list(ImpactLevel).index(self)


class AvailabilityStatus(str, Enum):
    """Whether research can actually proceed on this domain's candidates."""

    RESEARCH_READY = "RESEARCH_READY"
    OBSERVATION_ONLY = "OBSERVATION_ONLY"
    DATA_NOT_ACQUIRED = "DATA_NOT_ACQUIRED"
    FOUNDATION_ONLY = "FOUNDATION_ONLY"
    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"
    NO_CANDIDATE_MARKETS = "NO_CANDIDATE_MARKETS"
    EXCLUDED_BY_MANDATE = "EXCLUDED_BY_MANDATE"
    DOMAIN_NOT_SUPPORTED = "DOMAIN_NOT_SUPPORTED"


class DetectionBasis(str, Enum):
    #: The item's own market_intel category maps to this channel.
    SOURCE_CATEGORY = "SOURCE_CATEGORY"
    #: Channel lexicon terms matched in the headline/summary.
    LEXICON = "LEXICON"
    #: The issuing institution is itself a channel actor (e.g. the Fed).
    ISSUER = "ISSUER"


class DetectionStrength(str, Enum):
    #: Authoritative source category.
    STRONG = "STRONG"
    #: Two or more independent textual cues.
    MODERATE = "MODERATE"
    #: A single cue -- topic plausible, not established.
    WEAK = "WEAK"


class TriageConfidence(str, Enum):
    """Confidence that the TRIAGE classification is right -- never
    confidence in any price outcome."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class HorizonAlignment(str, Enum):
    ALIGNED = "ALIGNED"
    MISALIGNED = "MISALIGNED"
    #: The mandate states a flexible horizon.
    UNCONSTRAINED = "UNCONSTRAINED"


class ChannelEvidence(BaseModel):
    """Why the scan believes the event runs through one economic channel."""

    model_config = {"frozen": True, "extra": "forbid"}

    channel: EconomicChannel
    channel_label: str
    rule_id: str
    bases: tuple[DetectionBasis, ...]
    strength: DetectionStrength
    matched_terms: tuple[str, ...] = ()
    #: The direction shift established by a direction cue, if exactly one
    #: shift's cues matched.
    shift: str | None = None
    #: True when cues for more than one opposing shift matched -- direction
    #: is then deliberately left undetermined.
    conflicting_shift_cues: bool = False
    salience_terms: tuple[str, ...] = ()


class CandidateMarket(BaseModel):
    """Where deeper research could START in one domain -- a candidate, never a
    recommendation."""

    model_config = {"frozen": True, "extra": "forbid"}

    symbol: str
    display_name: str
    group_label: str
    research_ready: bool
    channel: EconomicChannel
    exposure_label: str
    exposure_strength: ExposureStrength
    rationale: str


class DirectionalPressure(BaseModel):
    """A possible first-order direction, present ONLY when a direction cue
    established the shift and the channel table records a textbook sign for
    this market group. A hypothesis to test, never a forecast."""

    model_config = {"frozen": True, "extra": "forbid"}

    channel: EconomicChannel
    shift: str
    exposure_label: str
    symbols: tuple[str, ...]
    pressure: PressureSign
    rationale: str


class ExposureContribution(BaseModel):
    """One (channel, exposure) pair that contributed to a domain's level,
    exactly as ``impact-level/1`` scored it. Kept on the assessment so the
    mechanism-adjusted pass can revise the triage exposure by exposure
    instead of re-deriving it from the flattened candidate list."""

    model_config = {"frozen": True, "extra": "forbid"}

    channel: EconomicChannel
    exposure_label: str
    exposure_strength: ExposureStrength
    detection_strength: DetectionStrength
    salient: bool
    impact_level: ImpactLevel
    symbols: tuple[str, ...] = ()


class InitialImpactAssessment(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    assessment_pass: Literal["INITIAL"] = "INITIAL"
    event_id: str
    asset_domain: MandateDomain
    #: ``None`` = NOT ASSESSED (the domain is outside the mandate, or the
    #: platform has no universe for it) -- never conflated with NONE.
    impact_level: ImpactLevel | None
    availability: AvailabilityStatus
    availability_note: str
    #: Every exposure that contributed, strongest first; the domain level is
    #: the highest of their levels.
    contributions: tuple[ExposureContribution, ...] = ()
    candidate_channels: tuple[EconomicChannel, ...] = ()
    candidate_groups: tuple[str, ...] = ()
    candidate_markets: tuple[CandidateMarket, ...] = ()
    possible_direction: tuple[DirectionalPressure, ...] = ()
    expected_horizon: ImpactHorizon | None = None
    horizon_rationale: str | None = None
    horizon_alignment: HorizonAlignment | None = None
    confidence: TriageConfidence | None = None
    uncertainty: tuple[str, ...] = ()
    #: How the user's constraints bear on expressing this domain's research
    #: (shorting, horizon, instrument filters). Never changes impact_level.
    mandate_notes: tuple[str, ...] = ()
    reasoning_provenance: tuple[str, ...] = ()

    @property
    def assessed(self) -> bool:
        return self.impact_level is not None

    @property
    def deserves_research(self) -> bool:
        return self.impact_level is not None and self.impact_level.rank >= ImpactLevel.MEDIUM.rank


class InitialImpactScan(BaseModel):
    """One event x one mandate -> one assessment per `MandateDomain`, in
    canonical order (excluded domains included, marked not-assessed)."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = SCAN_SCHEMA_VERSION
    assessment_pass: Literal["INITIAL"] = "INITIAL"
    event: ImpactEvent
    mandate_fingerprint: str
    mandate_summary: str
    channels: tuple[ChannelEvidence, ...]
    assessments: tuple[InitialImpactAssessment, ...]
    generated_by: str = GENERATED_BY
    not_final_note: str = (
        "Initial impact scan: research triage from the event's category/keywords and a static economic-channel "
        "table, before any mechanism reasoning. A mechanism-adjusted assessment will supersede it. Not an "
        "expected return, probability, trade instruction, portfolio weight, or scientific verdict."
    )

    def assessment(self, domain: MandateDomain) -> InitialImpactAssessment:
        return next(a for a in self.assessments if a.asset_domain is domain)

    @property
    def top_level(self) -> ImpactLevel | None:
        levels = [a.impact_level for a in self.assessments if a.impact_level is not None]
        return max(levels, key=lambda lvl: lvl.rank) if levels else None
