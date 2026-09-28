"""News Alpha Phase G -- SIGNAL SELECTION: which ranked signals may enter a
portfolio, and which way each points today. Never how much.

Stage 1 (`screen_eligibility`) reads the `RankedSignalSet` only:

* the tier -- only RANKED signals; excluded and not-rankable ones keep their
  Phase F reason;
* the eligibility policy -- QUALIFIED needs screen support; EXPLORATORY
  admits evidence that at least leans the declared way (never a
  contradicted signal);
* the domain -- only domains the platform can size and execute;
* the risk taxonomy -- an unclassified instrument cannot be constrained.

Stage 2 (`resolve_directions`) reads each signal's OWN factor value at the
as-of date: direction = sign(factor) x declared sign. That is all a signal
contributes -- screening statistics are evidence about the hypothesis, never
an expected return, and never a weight. An undefined factor value is
NO_CURRENT_SIGNAL (never filled); a short under a no-shorting mandate is
SHORT_NOT_PERMITTED (its long/flat form is flat today).

Stage 3 (`apply_count_limits`) keeps the best-ranked exposures and, inside
each, the best-ranked members: RANK DECIDES PRIORITY, NEVER SIZE. An exposure
cluster is Phase F's exposure group -- its alternates share its budget.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from enum import Enum
from typing import Protocol

from pydantic import BaseModel

from alpha_agent.news_alpha.mandate import DOMAIN_LABELS, MandateDomain, ResearchMandate
from alpha_agent.portfolio.classification import classify_instrument
from alpha_agent.portfolio.policy import (
    EXECUTABLE_DOMAINS,
    EligibilityMode,
    PortfolioConstructionPolicy,
)
from alpha_agent.portfolio.risk_model import instrument_key
from alpha_agent.recommendation.signal_ranking import (
    EvidenceDirection,
    RankedSignal,
    RankedSignalSet,
    RankingTier,
    SignalRole,
)
from alpha_agent.screening.factor_diagnostics import ScreenStatus

__all__ = [
    "REJECTION_TEXT",
    "FactorSource",
    "FactorValues",
    "RejectedSignal",
    "RejectionReason",
    "RejectionStage",
    "SelectedSignal",
    "apply_count_limits",
    "resolve_directions",
    "screen_eligibility",
]


class FactorValues(Protocol):
    """One factor value per trading day (``None`` = undefined, never filled)."""

    @property
    def days(self) -> tuple[str, ...]: ...

    @property
    def values(self) -> tuple[float | None, ...]: ...


class FactorSource(Protocol):
    """Where a signal's direction is read from: a Phase E `CandidateScreen`
    (the discovery series -- construction) or, in Phase H, the same factor
    evaluated causally through the validation window."""

    @property
    def series(self) -> FactorValues: ...


class RejectionStage(str, Enum):
    SELECTION = "SELECTION"
    ALLOCATION = "ALLOCATION"


class RejectionReason(str, Enum):
    # stage 1 -- from the ranked set
    EXCLUDED_BY_MANDATE = "EXCLUDED_BY_MANDATE"
    NOT_RANKABLE = "NOT_RANKABLE"
    NO_SCREEN_SUPPORT = "NO_SCREEN_SUPPORT"
    CONTRADICTS_DECLARED_SIGN = "CONTRADICTS_DECLARED_SIGN"
    EVIDENCE_AGAINST_DECLARED_SIGN = "EVIDENCE_AGAINST_DECLARED_SIGN"
    ALTERNATES_EXCLUDED = "ALTERNATES_EXCLUDED"
    DOMAIN_NOT_EXECUTABLE = "DOMAIN_NOT_EXECUTABLE"
    NO_RISK_CLASSIFICATION = "NO_RISK_CLASSIFICATION"
    # stage 2 -- today's direction
    NO_MARKET_DATA = "NO_MARKET_DATA"
    NO_SCREEN = "NO_SCREEN"
    NO_CURRENT_SIGNAL = "NO_CURRENT_SIGNAL"
    SIGNAL_FLAT = "SIGNAL_FLAT"
    SHORT_NOT_PERMITTED = "SHORT_NOT_PERMITTED"
    # stage 3 -- priority by rank
    EXPOSURE_LIMIT = "EXPOSURE_LIMIT"
    EXPOSURE_MEMBER_LIMIT = "EXPOSURE_MEMBER_LIMIT"
    # allocation (C++)
    INSTRUMENT_NOT_ADMITTED = "INSTRUMENT_NOT_ADMITTED"
    CLUSTER_NETS_TO_ZERO = "CLUSTER_NETS_TO_ZERO"
    DEGENERATE_RISK_MODEL = "DEGENERATE_RISK_MODEL"
    RISK_INPUTS_UNAVAILABLE = "RISK_INPUTS_UNAVAILABLE"


REJECTION_TEXT: dict[RejectionReason, str] = {
    RejectionReason.EXCLUDED_BY_MANDATE: "Excluded by your mandate (Phase F).",
    RejectionReason.NOT_RANKABLE: "Not rankable yet -- no comparable screening evidence.",
    RejectionReason.NO_SCREEN_SUPPORT: "No screening support -- only signals whose screen says continue are qualified.",
    RejectionReason.CONTRADICTS_DECLARED_SIGN: "Its screen contradicts its declared sign.",
    RejectionReason.EVIDENCE_AGAINST_DECLARED_SIGN: "Its evidence leans against its declared sign.",
    RejectionReason.ALTERNATES_EXCLUDED: "Alternates are excluded by this policy (leads only).",
    RejectionReason.DOMAIN_NOT_EXECUTABLE: "Its domain cannot be sized and executed on this platform yet.",
    RejectionReason.NO_RISK_CLASSIFICATION: "No risk classification for the instrument -- limits cannot apply.",
    RejectionReason.NO_MARKET_DATA: "No usable real market data to size it.",
    RejectionReason.NO_SCREEN: "No cached factor series -- its current direction is unknown.",
    RejectionReason.NO_CURRENT_SIGNAL: "Its factor is undefined on the as-of date (never filled).",
    RejectionReason.SIGNAL_FLAT: "Its factor is exactly zero on the as-of date -- no direction.",
    RejectionReason.SHORT_NOT_PERMITTED: "It points short today, and your mandate allows no shorting -- flat.",
    RejectionReason.EXPOSURE_LIMIT: "Lower-ranked than the exposures that filled the exposure limit.",
    RejectionReason.EXPOSURE_MEMBER_LIMIT: "Lower-ranked than the members that filled its exposure's member limit.",
    RejectionReason.INSTRUMENT_NOT_ADMITTED: "The allocator did not admit its instrument.",
    RejectionReason.CLUSTER_NETS_TO_ZERO: "Its exposure's signals cancel on the same instrument -- no net risk.",
    RejectionReason.DEGENERATE_RISK_MODEL: (
        "Its exposure hedges another exactly -- equal risk contributions do not exist, so nothing is sized."),
    RejectionReason.RISK_INPUTS_UNAVAILABLE: "Risk statistics could not be estimated for it.",
}


class SelectedSignal(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    candidate_signal_id: str
    name: str
    domain: MandateDomain
    instrument: str
    instrument_key: str
    expression: str
    prediction_horizon_days: int
    rank: int
    role: SignalRole
    #: The Phase F exposure group -- the risk-budget cluster.
    exposure_group_id: str
    exposure_label: str
    screen_status: ScreenStatus
    aligned_t: float
    expected_sign: int
    #: ``None`` until stage 2.
    factor_value: float | None = None
    direction: int | None = None
    event_ids: tuple[str, ...] = ()


class RejectedSignal(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    candidate_signal_id: str
    name: str
    domain: MandateDomain
    instrument: str
    rank: int | None
    stage: RejectionStage
    reason: RejectionReason
    detail: str


def _rejected(s: RankedSignal, reason: RejectionReason, detail: str | None = None,
              stage: RejectionStage = RejectionStage.SELECTION) -> RejectedSignal:
    return RejectedSignal(candidate_signal_id=s.candidate_signal_id, name=s.name, domain=s.domain,
                          instrument=s.instrument, rank=s.rank, stage=stage, reason=reason,
                          detail=detail or REJECTION_TEXT[reason])


def screen_eligibility(
    ranked: RankedSignalSet, policy: PortfolioConstructionPolicy,
) -> tuple[list[SelectedSignal], list[RejectedSignal]]:
    eligible: list[SelectedSignal] = []
    rejected: list[RejectedSignal] = []
    for s in ranked.signals:
        if s.tier is RankingTier.EXCLUDED_BY_MANDATE:
            rejected.append(_rejected(s, RejectionReason.EXCLUDED_BY_MANDATE, s.summaries.user_fit))
            continue
        if s.tier is RankingTier.NOT_RANKABLE:
            rejected.append(_rejected(s, RejectionReason.NOT_RANKABLE, s.summaries.scientific_quality))
            continue
        assert s.merit is not None and s.rank is not None and s.role is not None and s.exposure_group_id
        status, evidence = s.merit.screen_status, s.merit.evidence
        if status is ScreenStatus.CONTRADICTS_EXPECTED_SIGN or evidence.direction is EvidenceDirection.CONTRADICTS:
            rejected.append(_rejected(s, RejectionReason.CONTRADICTS_DECLARED_SIGN))
            continue
        if policy.eligibility is EligibilityMode.QUALIFIED and status is not ScreenStatus.SCREEN_CONTINUE:
            rejected.append(_rejected(s, RejectionReason.NO_SCREEN_SUPPORT))
            continue
        if policy.eligibility is EligibilityMode.EXPLORATORY and evidence.aligned_t <= 0:
            rejected.append(_rejected(s, RejectionReason.EVIDENCE_AGAINST_DECLARED_SIGN,
                                      f"Its evidence leans against its declared sign (t {evidence.aligned_t:+.2f})."))
            continue
        if s.role is SignalRole.ALTERNATE and not policy.include_alternates:
            rejected.append(_rejected(s, RejectionReason.ALTERNATES_EXCLUDED))
            continue
        if s.domain not in EXECUTABLE_DOMAINS:
            rejected.append(_rejected(s, RejectionReason.DOMAIN_NOT_EXECUTABLE,
                                      f"{DOMAIN_LABELS[s.domain]} cannot be sized and executed on this platform yet."))
            continue
        if classify_instrument(s.domain, s.instrument) is None:
            rejected.append(_rejected(s, RejectionReason.NO_RISK_CLASSIFICATION))
            continue
        eligible.append(SelectedSignal(
            candidate_signal_id=s.candidate_signal_id, name=s.name, domain=s.domain, instrument=s.instrument,
            instrument_key=instrument_key(s.domain, s.instrument), expression=s.expression,
            prediction_horizon_days=s.prediction_horizon_days, rank=s.rank, role=s.role,
            exposure_group_id=s.exposure_group_id, exposure_label=ranked.group(s.exposure_group_id).label,
            screen_status=status, aligned_t=evidence.aligned_t, expected_sign=evidence.metrics.expected_sign,
            event_ids=s.mechanism.event_ids,
        ))
    return eligible, rejected


def resolve_directions(
    eligible: Sequence[SelectedSignal],
    screens: Mapping[str, FactorSource],
    *,
    as_of: date,
    mandate: ResearchMandate,
    ranked: RankedSignalSet,
) -> tuple[list[SelectedSignal], list[RejectedSignal]]:
    kept: list[SelectedSignal] = []
    rejected: list[RejectedSignal] = []
    day = as_of.isoformat()
    for s in eligible:
        source = ranked.signal(s.candidate_signal_id)
        screen = screens.get(s.candidate_signal_id)
        if screen is None:
            rejected.append(_rejected(source, RejectionReason.NO_SCREEN))
            continue
        series = dict(zip(screen.series.days, screen.series.values, strict=True))
        value = series.get(day)
        if value is None:
            rejected.append(_rejected(source, RejectionReason.NO_CURRENT_SIGNAL,
                                      f"Its factor is undefined on {day} (never filled)."))
            continue
        if value == 0:
            rejected.append(_rejected(source, RejectionReason.SIGNAL_FLAT))
            continue
        direction = (1 if value > 0 else -1) * s.expected_sign
        if direction < 0 and not mandate.shorting_allowed:
            rejected.append(_rejected(source, RejectionReason.SHORT_NOT_PERMITTED,
                                      f"On {day} its factor ({value:+.4g}) points short, and your mandate allows no "
                                      "shorting -- its long/flat form is flat today."))
            continue
        kept.append(s.model_copy(update={"factor_value": value, "direction": direction}))
    return kept, rejected


def apply_count_limits(
    selected: Sequence[SelectedSignal], policy: PortfolioConstructionPolicy, ranked: RankedSignalSet,
) -> tuple[list[SelectedSignal], list[RejectedSignal]]:
    by_rank = sorted(selected, key=lambda s: s.rank)
    exposures: list[str] = []
    members: dict[str, int] = {}
    kept: list[SelectedSignal] = []
    rejected: list[RejectedSignal] = []
    for s in by_rank:
        group = s.exposure_group_id
        if group not in exposures:
            if len(exposures) >= policy.max_exposures:
                rejected.append(_rejected(ranked.signal(s.candidate_signal_id), RejectionReason.EXPOSURE_LIMIT))
                continue
            exposures.append(group)
        if members.get(group, 0) >= policy.max_signals_per_exposure:
            rejected.append(_rejected(ranked.signal(s.candidate_signal_id), RejectionReason.EXPOSURE_MEMBER_LIMIT))
            continue
        members[group] = members.get(group, 0) + 1
        kept.append(s)
    return kept, rejected
