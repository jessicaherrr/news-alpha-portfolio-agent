"""News Alpha Phase H -- PATH-LEVEL STUDY DESIGNS: the questions the platform
must answer empirically, never assume, and whether it can answer them yet.

Questions
    DIRECT_PRICES_FASTER     does information on DIRECT paths price faster than on
                             SUPPLY_CHAIN / CROSS_SECTOR paths?
    INDIRECT_DECAY           do indirect effects decay differently, by transmission
                             depth, across horizons?
    MECHANISM_VS_SENTIMENT   does mechanism-grounded measurement outperform direct
                             news sentiment?

Arms, compared under ONE set of scientific rules (the same validation
protocol, windows, costs, and one multiple-testing family declared up front
over every arm x stratum x horizon cell):
    A  RAW_NEWS_SENTIMENT            conventional sentiment of the news text
    B  DIRECT_LLM_DIRECTION          an LLM's bullish / bearish call
    C  EVENT_EXPOSURE                event x exposure
    D  EVENT_EXPOSURE_CONFIRMATION   C + a confirmation measurement
    E  FULL_TRANSMISSION             D + transmission-aware weighting

Every question needs an EVENT-CONDITIONED sample: real, dated, point-in-time
safe historical events inside the 2018-2024 research span, each with the
outcomes measured from its own publication timestamp. The Phase E factor
diagnostics are UNCONDITIONAL daily factors -- they are refused here as
evidence (`EventConditionedEvidenceRef`), whatever they show.

`assess_study` reports TESTABLE only when every requirement of the question's
arms is really available on the platform; otherwise NOT_YET_TESTABLE with
every typed blocker. Requirements are probed from real stores and the
pipeline's own declared conditioning -- never assumed present. This module
reads; it never writes a store, calls a network or runs a statistic.
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from alpha_agent.news_alpha.candidate_signals import ConditioningStatus, EventConditioning
from alpha_agent.news_alpha.signal_paths import PathType
from alpha_agent.registry.holdout_guard import HOLDOUT_START

__all__ = [
    "STUDY_DESIGN_RULE",
    "ComparisonDesign",
    "EventConditionedEvidenceRef",
    "StudyArm",
    "StudyBlocker",
    "StudyCapabilities",
    "StudyQuestion",
    "StudyReadiness",
    "StudyRequirement",
    "StudyTestability",
    "assess_study",
    "default_designs",
    "probe_study_capabilities",
]

STUDY_DESIGN_RULE = "path-study-design/1"
RESEARCH_SPAN = (date(2018, 1, 1), date.fromisoformat(HOLDOUT_START))


class StudyQuestion(str, Enum):
    DIRECT_PRICES_FASTER = "DIRECT_PRICES_FASTER"
    INDIRECT_DECAY = "INDIRECT_DECAY"
    MECHANISM_VS_SENTIMENT = "MECHANISM_VS_SENTIMENT"


class StudyArm(str, Enum):
    A_RAW_NEWS_SENTIMENT = "A_RAW_NEWS_SENTIMENT"
    B_DIRECT_LLM_DIRECTION = "B_DIRECT_LLM_DIRECTION"
    C_EVENT_EXPOSURE = "C_EVENT_EXPOSURE"
    D_EVENT_EXPOSURE_CONFIRMATION = "D_EVENT_EXPOSURE_CONFIRMATION"
    E_FULL_TRANSMISSION = "E_FULL_TRANSMISSION"


class StudyRequirement(str, Enum):
    #: Dated, sourced events inside the research span, known at their timestamps.
    HISTORICAL_EVENT_SAMPLE = "HISTORICAL_EVENT_SAMPLE"
    #: Each event's transmission paths (type, depth) built from what was known then.
    PATH_TYPES_PER_EVENT = "PATH_TYPES_PER_EVENT"
    #: The news text as published, timestamped.
    HISTORICAL_NEWS_TEXT = "HISTORICAL_NEWS_TEXT"
    #: A sentiment model fixed before the sample it scores.
    PIT_SENTIMENT_MODEL = "PIT_SENTIMENT_MODEL"
    #: An LLM whose knowledge ends before the sample's outcomes -- otherwise its
    #: "direction" can encode what happened next.
    LLM_WITHOUT_HINDSIGHT = "LLM_WITHOUT_HINDSIGHT"
    EXPOSURE_MAGNITUDE = "EXPOSURE_MAGNITUDE"
    CONFIRMATION_MEASUREMENT = "CONFIRMATION_MEASUREMENT"
    TRANSMISSION_WEIGHT = "TRANSMISSION_WEIGHT"
    #: Forward outcomes measured from each event's own publication timestamp.
    EVENT_CONDITIONED_OUTCOMES = "EVENT_CONDITIONED_OUTCOMES"


_SAMPLE = (StudyRequirement.HISTORICAL_EVENT_SAMPLE, StudyRequirement.EVENT_CONDITIONED_OUTCOMES)
ARM_REQUIREMENTS: dict[StudyArm, tuple[StudyRequirement, ...]] = {
    StudyArm.A_RAW_NEWS_SENTIMENT: (*_SAMPLE, StudyRequirement.HISTORICAL_NEWS_TEXT,
                                    StudyRequirement.PIT_SENTIMENT_MODEL),
    StudyArm.B_DIRECT_LLM_DIRECTION: (*_SAMPLE, StudyRequirement.HISTORICAL_NEWS_TEXT,
                                      StudyRequirement.LLM_WITHOUT_HINDSIGHT),
    StudyArm.C_EVENT_EXPOSURE: (*_SAMPLE, StudyRequirement.EXPOSURE_MAGNITUDE),
    StudyArm.D_EVENT_EXPOSURE_CONFIRMATION: (*_SAMPLE, StudyRequirement.EXPOSURE_MAGNITUDE,
                                             StudyRequirement.CONFIRMATION_MEASUREMENT),
    StudyArm.E_FULL_TRANSMISSION: (*_SAMPLE, StudyRequirement.EXPOSURE_MAGNITUDE,
                                   StudyRequirement.CONFIRMATION_MEASUREMENT, StudyRequirement.TRANSMISSION_WEIGHT,
                                   StudyRequirement.PATH_TYPES_PER_EVENT),
}


class ComparisonDesign(BaseModel):
    """One predeclared comparison. Its family is every (arm x stratum x
    horizon) cell -- declared before any data is read, never grown after."""

    model_config = {"frozen": True, "extra": "forbid"}

    rule: Literal["path-study-design/1"] = STUDY_DESIGN_RULE
    question: StudyQuestion
    arms: tuple[StudyArm, ...]
    #: Path-type strata (empty = not stratified by path type).
    path_types: tuple[PathType, ...] = ()
    transmission_depths: tuple[int, ...] = ()
    horizons_trading_days: tuple[int, ...] = (1, 5, 20, 60)
    scientific_rules: str = (
        "Same validation protocol (TRAIN 2018-2022 / VALIDATION 2023-2024 / sealed 2025 holdout), same costs, one "
        "BH-FDR family over every cell below, event-conditioned outcomes only."
    )

    @property
    def declared_family_size(self) -> int:
        strata = max(1, len(self.path_types)) * max(1, len(self.transmission_depths))
        return len(self.arms) * strata * len(self.horizons_trading_days)

    def requirements(self) -> tuple[StudyRequirement, ...]:
        needed = {req for arm in self.arms for req in ARM_REQUIREMENTS[arm]}
        if self.path_types or self.transmission_depths:
            needed.add(StudyRequirement.PATH_TYPES_PER_EVENT)
        return tuple(r for r in StudyRequirement if r in needed)


def default_designs() -> tuple[ComparisonDesign, ...]:
    direct_vs_indirect = (PathType.DIRECT, PathType.SUPPLY_CHAIN, PathType.CROSS_SECTOR)
    return (
        ComparisonDesign(question=StudyQuestion.DIRECT_PRICES_FASTER, arms=(StudyArm.E_FULL_TRANSMISSION,),
                         path_types=direct_vs_indirect),
        ComparisonDesign(question=StudyQuestion.INDIRECT_DECAY, arms=(StudyArm.E_FULL_TRANSMISSION,),
                         path_types=direct_vs_indirect, transmission_depths=(1, 2, 3)),
        ComparisonDesign(question=StudyQuestion.MECHANISM_VS_SENTIMENT, arms=tuple(StudyArm)),
    )


class StudyCapabilities(BaseModel):
    """What the platform can actually supply today, each with the fact it rests on."""

    model_config = {"frozen": True, "extra": "forbid"}

    available: dict[StudyRequirement, bool]
    basis: dict[StudyRequirement, str]
    probed_on: date


def _dated_in_span(values: list[datetime]) -> int:
    lo = datetime.combine(RESEARCH_SPAN[0], datetime.min.time(), tzinfo=UTC)
    hi = datetime.combine(RESEARCH_SPAN[1], datetime.min.time(), tzinfo=UTC)
    return sum(1 for v in values if lo <= (v if v.tzinfo else v.replace(tzinfo=UTC)) < hi)


def probe_study_capabilities(*, news_store=None, event_store=None, today: date | None = None) -> StudyCapabilities:
    """Reads the platform's own stores and declared conditioning. Counts
    dated items inside the research span only -- nothing later is kept."""
    from alpha_agent.market_intel.store import EventStore, NewsStore

    news_store = news_store or NewsStore()
    event_store = event_store or EventStore()
    start = datetime.combine(RESEARCH_SPAN[0], datetime.min.time(), tzinfo=UTC)
    try:
        news_in_span = _dated_in_span([i.published_at for i in news_store.list_recent(since=start, limit=1_000_000)])
    except Exception:  # noqa: BLE001 -- an unreadable store supplies nothing
        news_in_span = 0
    horizon = (RESEARCH_SPAN[1] - RESEARCH_SPAN[0]).days
    try:
        events_in_span = _dated_in_span([e.scheduled_at for e in event_store.list_upcoming(now=start,
                                                                                           horizon_days=horizon)])
    except Exception:  # noqa: BLE001
        events_in_span = 0
    conditioning = EventConditioning()
    derived = conditioning.transmission_weight is not ConditioningStatus.NOT_DERIVED
    magnitude = conditioning.exposure is not ConditioningStatus.UNKNOWN_MAGNITUDE
    event_conditioned = conditioning.variant != "UNCONDITIONAL_MEASUREMENT"
    sample = events_in_span + news_in_span > 0
    available = {
        StudyRequirement.HISTORICAL_EVENT_SAMPLE: sample,
        StudyRequirement.PATH_TYPES_PER_EVENT: sample,
        StudyRequirement.HISTORICAL_NEWS_TEXT: news_in_span > 0,
        StudyRequirement.PIT_SENTIMENT_MODEL: False,
        StudyRequirement.LLM_WITHOUT_HINDSIGHT: False,
        StudyRequirement.EXPOSURE_MAGNITUDE: magnitude,
        StudyRequirement.CONFIRMATION_MEASUREMENT: event_conditioned,
        StudyRequirement.TRANSMISSION_WEIGHT: derived,
        StudyRequirement.EVENT_CONDITIONED_OUTCOMES: sample,
    }
    basis = {
        StudyRequirement.HISTORICAL_EVENT_SAMPLE: (
            f"{events_in_span} scheduled event(s) and {news_in_span} news item(s) dated inside 2018-2024 in the "
            "market-intelligence stores (their contents are recent, not historical)"),
        StudyRequirement.PATH_TYPES_PER_EVENT: "path types exist only for events the pipeline has seen",
        StudyRequirement.HISTORICAL_NEWS_TEXT: f"{news_in_span} news item(s) dated inside 2018-2024",
        StudyRequirement.PIT_SENTIMENT_MODEL: "no sentiment model is part of the platform",
        StudyRequirement.LLM_WITHOUT_HINDSIGHT: (
            "the platform's LLM was trained after 2024 -- its call on 2018-2024 news can encode the outcome; arm B is "
            "only testable prospectively, on events recorded after the model's knowledge ends"),
        StudyRequirement.EXPOSURE_MAGNITUDE: f"candidate conditioning: exposure {conditioning.exposure.value}",
        StudyRequirement.CONFIRMATION_MEASUREMENT: (
            f"candidate conditioning: {conditioning.variant} -- confirmation {conditioning.confirmation.value} is the "
            "unconditional price response itself, not an event-window measurement"),
        StudyRequirement.TRANSMISSION_WEIGHT: f"candidate conditioning: transmission weight "
                                              f"{conditioning.transmission_weight.value}",
        StudyRequirement.EVENT_CONDITIONED_OUTCOMES: "needs the event sample above",
    }
    return StudyCapabilities(available=available, basis=basis, probed_on=today or datetime.now(UTC).date())


class StudyReadiness(str, Enum):
    TESTABLE = "TESTABLE"
    NOT_YET_TESTABLE = "NOT_YET_TESTABLE"


class StudyBlocker(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    requirement: StudyRequirement
    arms: tuple[StudyArm, ...]
    basis: str


class StudyTestability(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    design: ComparisonDesign
    status: StudyReadiness
    blockers: tuple[StudyBlocker, ...]
    note: str = "Not assumed either way: the answer is empirical, and this study has not been run."


def assess_study(design: ComparisonDesign, capabilities: StudyCapabilities) -> StudyTestability:
    blockers = []
    for req in design.requirements():
        if capabilities.available.get(req, False):
            continue
        arms = tuple(a for a in design.arms if req in ARM_REQUIREMENTS[a]) or design.arms
        blockers.append(StudyBlocker(requirement=req, arms=arms, basis=capabilities.basis.get(req, "")))
    return StudyTestability(design=design, status=StudyReadiness.NOT_YET_TESTABLE if blockers else StudyReadiness.TESTABLE,
                            blockers=tuple(blockers))


class EventConditionedEvidenceRef(BaseModel):
    """The only evidence shape a path-level study accepts. An unconditional
    daily factor diagnostic is refused, whatever it shows."""

    model_config = {"frozen": True, "extra": "forbid"}

    scope: Literal["EVENT_CONDITIONED", "UNCONDITIONAL_FACTOR"]
    source_id: str
    event_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _event_conditioned_only(self) -> EventConditionedEvidenceRef:
        if self.scope != "EVENT_CONDITIONED":
            raise ValueError("unconditional daily factor diagnostics are not event-conditioned evidence")
        return self
