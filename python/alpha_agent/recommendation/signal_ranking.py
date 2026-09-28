"""News Alpha Phase F -- MULTI-ASSET SIGNAL RANKING: which candidate signals
deserve portfolio consideration first, under one research mandate.

    ... -> Candidate Signal Spec -> Factor Expression            (Phase E, news_alpha)
        -> factor series -> factor diagnostics                   (Phase E, screening)
        -> MULTI-ASSET SIGNAL RANKING (this module)              (Phase F)
        -> portfolio construction                                (later -- not here)

ONE OWNER. The recommendation layer is the platform's research-
prioritization owner; this module extends it to a new ranked object rather
than starting a second engine. It reuses the layer's own split between a
profile-free merit and a profile-dependent fit (`recommendation.fit`'s
`PersonalizationState` and trading-frequency bands). It does NOT reuse
`score_research_promise`: that ranks registry results (Sharpe, DSR, BH q)
a screened candidate does not have, and it turns missing evidence into a
low score -- here missing evidence is never low quality. Every other
ranker ranks a different object: `context_retrieval.ranking` ranks quant
mechanisms for a market context, `opportunity` ranks current market
conditions, `screening.fast_screen` ranks strategy backtest screens.

WHAT IS RANKED. A candidate signal TOGETHER WITH its screening diagnostics
-- never a naked candidate. Every signal lands in one tier:

* RANKED -- real, comparable, sufficient screening evidence, admitted by
  the mandate;
* EXCLUDED_BY_MANDATE -- the mandate's access constraints exclude it; its
  research merit is kept, unchanged, beside the exclusion;
* NOT_RANKABLE -- not screened, too little data, no diagnostic method for
  its domain, non-comparable or non-real evidence. Missing evidence is
  NOT low quality: these carry no rank, sit in their own list, and say
  what would make them rankable.

RESEARCH MERIT vs USER FIT. `ResearchMerit` is computed from the screen
alone and never reads the mandate; `UserFit` is computed from the mandate
and never changes merit. The mandate can exclude a signal (domain or
instrument access); its soft preferences (holding period, trading
frequency, shorting) are reported beside merit and break exact merit ties
only. Mechanism confidence and expression fidelity travel as provenance
(`MechanismContext`) and never order anything.

NO UNIVERSAL SCORE. A futures Sharpe, an equity IC and a crypto funding
signal are not interchangeable. Evidence enters through a normalizer keyed
by its diagnostic METHOD (the peer group): the normalizer keeps every raw
metric and maps them onto a small set of typed grades using that method's
OWN predeclared thresholds. Today there is one method -- Phase E's
unconditional single-instrument time-series factor screen, identical for
futures and ETFs -- so a futures and an ETF candidate are comparable
because they were measured the same way, not because their numbers were
forced onto one scale. Evidence from any other method is
NON_COMPARABLE_EVIDENCE until a normalizer for it exists. Grades are
compared lexicographically in a documented order; nothing is summed.

REDUNDANCY. Structural duplicates are merged first (one structural
candidate reached by several events is one signal). Two ranked signals are
then linked when EITHER:

* they are the SAME STRUCTURAL SIGNAL FAMILY on the same instrument --
  same instrument AND same `signal_rule_id` (the same generation rule,
  e.g. both `price-momentum-continuation`; a different lookback or horizon
  is still the same family); or
* their expected-sign-aligned factor series RANK-correlate at |rho| >= the
  policy threshold, REGARDLESS of instrument or family.

SAME INSTRUMENT IS NOT SAME EXPOSURE. A different factor family on the same
instrument (e.g. a momentum candidate and a mean-reversion candidate on NQ,
once more than one signal rule exists) is never merged by co-location
alone -- it links only if it is actually correlated. The structural/family
identity that decides this is `SignalSpec.signal_rule_id`, never guessed
from the instrument.

THE REDUNDANCY THRESHOLD IS A VERSIONED POLICY, NOT A SCIENTIFIC FINDING.
`SignalRankingPolicy.redundancy_correlation` (default 0.8, a conventional
factor-redundancy cut-off) and `.correlation_method` are fixed, documented,
FINGERPRINTED choices (`policy_fingerprint`) -- changing either is a new
policy whose result carries a different fingerprint, never a claim that the
underlying signals became more or less related.

Linked signals form one EXPOSURE GROUP. Each group has one LEAD (its
best-merit member). Leads -- the independent exposures -- are ranked first;
alternates follow, each naming its lead. Signals in different groups reached
by the same event share that EVENT THESIS: they are distinct exposures to
one news hypothesis, and are marked so. An exposure reached by several
events belongs to each of their theses (never chained into one).

RANKED IS NOT PORTFOLIO-ELIGIBLE. This module answers "what deserves
research attention first", never "what may enter a portfolio". A future
portfolio-construction phase needs its OWN separate allocation/eligibility
gate (capital limits, margin, position-count caps, risk budget) before any
ranked signal here is actually held -- nothing here computes or implies
that gate, and `RankedSignalSet.rank`/`.tier` must never be read as one.

NOT HERE: portfolio weights, sizes or allocations, BUY/SELL, expected
return, probability, validation, registry writes or verdicts. Pure and
deterministic: no IO, no network, no LLM, no Streamlit.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Literal

from pydantic import BaseModel, model_validator

from alpha_agent.crypto.provenance import DataProvenanceRole
from alpha_agent.news_alpha.candidate_signals import (
    CandidateNote,
    CandidateSignal,
    CandidateSignalSet,
)
from alpha_agent.news_alpha.mandate import (
    DOMAIN_LABELS,
    MandateDomain,
    ResearchMandate,
    registry_domain_for,
)
from alpha_agent.news_alpha.measurement import ResolutionStatus
from alpha_agent.news_alpha.signal_paths import PathStatus, PathType
from alpha_agent.recommendation.fit import PersonalizationState, frequency_band
from alpha_agent.recommendation.profile import HoldingPeriod, TradingFrequency
from alpha_agent.screening.candidate_signal_screen import SCREENABLE_DOMAINS, CandidateScreen
from alpha_agent.screening.factor_diagnostics import (
    FACTOR_DIAGNOSTICS_SCHEMA,
    FACTOR_SCREEN_RULE,
    DiagnosticScope,
    FactorScreenPolicy,
    ICKind,
    ScreenStatus,
    signal_correlations,
)

__all__ = [
    "FACTOR_SCREEN_NORMALIZATION",
    "HOLDING_PERIOD_TRADING_DAYS",
    "SIGNAL_RANKING_POLICY",
    "SIGNAL_RANKING_SCHEMA",
    "CostHeadroom",
    "DataQualityGrade",
    "EventThesis",
    "EvidenceDirection",
    "ExclusionReason",
    "ExposureGroup",
    "FitCheck",
    "FitDimension",
    "FitStatus",
    "MechanismContext",
    "NormalizedEvidence",
    "NotRankableReason",
    "PairCorrelation",
    "PeerGroup",
    "RankedSignal",
    "RankedSignalSet",
    "RankingTier",
    "RedundancyRelation",
    "ResearchMerit",
    "ScreenMetrics",
    "SignalRankingPolicy",
    "SignalRole",
    "SignalSummaries",
    "StabilityGrade",
    "Uncertainty",
    "UncertaintyCode",
    "UserFit",
    "assess_user_fit",
    "normalize_screen_evidence",
    "rank_candidate_signals",
]

SIGNAL_RANKING_SCHEMA = "ranked-signal-set/1"
SIGNAL_RANKING_POLICY = "signal-ranking/1"
FACTOR_SCREEN_NORMALIZATION = "factor-screen-grades/1"
_TRADING_DAYS_PER_YEAR = 252

#: The holding periods an `InvestorProfile` states, as inclusive windows of
#: trading days a prediction horizon must fall in to fit. ``None`` = no
#: preference. INTRADAY fits no daily-bar signal (every one holds >= 1 day).
HOLDING_PERIOD_TRADING_DAYS: dict[HoldingPeriod, tuple[int, int] | None] = {
    HoldingPeriod.INTRADAY: (0, 0),
    HoldingPeriod.ONE_TO_THREE_DAYS: (1, 3),
    HoldingPeriod.SEVERAL_DAYS: (2, 10),
    HoldingPeriod.WEEKS: (5, 30),
    HoldingPeriod.FLEXIBLE: None,
}


class SignalRankingPolicy(BaseModel):
    """``signal-ranking/1`` -- fixed before any signal is ranked, never
    fitted to the signals it ranks. Every field, including
    `redundancy_correlation`, is a versioned POLICY CHOICE, not a scientific
    finding: changing it is a new, explicitly fingerprinted policy
    (`fingerprint()` / `policy_fingerprint` on the result), never a silent
    reinterpretation of the same evidence."""

    model_config = {"frozen": True, "extra": "forbid"}

    #: |expected-sign-aligned factor correlation| at or above which two
    #: signals are one exposure (a conventional factor-redundancy cut-off).
    redundancy_correlation: float = 0.8
    #: Rank correlation: a +/-1 position follows the factor's sign and order,
    #: not its outliers (found on real CL bars -- see `signal_correlations`).
    correlation_method: Literal["spearman"] = "spearman"
    #: Common trading days a correlation needs before it can link two signals.
    min_common_days: int = 250

    def fingerprint(self) -> str:
        payload = {
            "rule": SIGNAL_RANKING_POLICY, "normalization": FACTOR_SCREEN_NORMALIZATION, **self.model_dump(),
            "holding_periods": {k.value: v for k, v in HOLDING_PERIOD_TRADING_DAYS.items()},
        }
        return "rankpolicy1:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------


class RankingTier(str, Enum):
    RANKED = "RANKED"
    EXCLUDED_BY_MANDATE = "EXCLUDED_BY_MANDATE"
    NOT_RANKABLE = "NOT_RANKABLE"


class NotRankableReason(str, Enum):
    """Why a signal carries no rank. None of these is a quality judgement."""

    NOT_SCREENED = "NOT_SCREENED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NO_DIAGNOSTIC_METHOD = "NO_DIAGNOSTIC_METHOD"
    NON_COMPARABLE_EVIDENCE = "NON_COMPARABLE_EVIDENCE"
    NOT_REAL_EVIDENCE = "NOT_REAL_EVIDENCE"


NOT_RANKABLE_TEXT: dict[NotRankableReason, str] = {
    NotRankableReason.NOT_SCREENED: "Not screened yet -- no diagnostics to rank on.",
    NotRankableReason.INSUFFICIENT_EVIDENCE: "Screened, but too little data to evaluate the declared horizon.",
    NotRankableReason.NO_DIAGNOSTIC_METHOD: "No diagnostic method for this domain yet -- nothing can screen it.",
    NotRankableReason.NON_COMPARABLE_EVIDENCE: (
        "Its evidence comes from another method, scope or screening policy, so it cannot be compared with the "
        "ranked signals."
    ),
    NotRankableReason.NOT_REAL_EVIDENCE: "Its screen was computed from non-real data, which is never evidence.",
}
NEXT_STEP_TEXT: dict[NotRankableReason, str] = {
    NotRankableReason.NOT_SCREENED: "Run the factor diagnostics (2018-2022 discovery data).",
    NotRankableReason.INSUFFICIENT_EVIDENCE: "Needs a longer history for this instrument.",
    NotRankableReason.NO_DIAGNOSTIC_METHOD: "Needs point-in-time daily bars and a loader for this domain.",
    NotRankableReason.NON_COMPARABLE_EVIDENCE: "Re-screen it under the current method and policy.",
    NotRankableReason.NOT_REAL_EVIDENCE: "Re-screen it on real, already-acquired data.",
}


class ExclusionReason(str, Enum):
    DOMAIN_NOT_ALLOWED = "DOMAIN_NOT_ALLOWED"
    INSTRUMENT_NOT_IN_ALLOWLIST = "INSTRUMENT_NOT_IN_ALLOWLIST"
    INSTRUMENT_DENYLISTED = "INSTRUMENT_DENYLISTED"


class EvidenceDirection(str, Enum):
    """The effective-sample t, in the DECLARED direction, against the screen
    policy's own threshold (never a new one)."""

    SUPPORTS = "SUPPORTS"  # >= threshold
    LEANS_EXPECTED = "LEANS_EXPECTED"  # between 0 and threshold
    LEANS_OPPOSITE = "LEANS_OPPOSITE"  # between -threshold and 0
    CONTRADICTS = "CONTRADICTS"  # <= -threshold


class StabilityGrade(str, Enum):
    """Share of yearly subperiods with the declared sign, against the screen
    policy's own agreement share."""

    CONSISTENT = "CONSISTENT"
    MIXED = "MIXED"
    OPPOSED = "OPPOSED"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class CostHeadroom(str, Enum):
    #: Positive drift-free timing edge: some trading cost can be afforded.
    TIMING_EDGE = "TIMING_EDGE"
    NO_TIMING_EDGE = "NO_TIMING_EDGE"
    NOT_ASSESSED = "NOT_ASSESSED"


class DataQualityGrade(str, Enum):
    CLEAN = "CLEAN"
    #: A proxy measurement or partial history.
    QUALIFIED = "QUALIFIED"
    #: Feature QA issues, or undefined values after warm-up (never filled).
    FLAGGED = "FLAGGED"


class SignalRole(str, Enum):
    #: The best-merit member of its exposure group: an independent exposure.
    LEAD = "LEAD"
    #: Another expression of a lead's exposure -- not an independent one.
    ALTERNATE = "ALTERNATE"


class RedundancyRelation(str, Enum):
    """Why two ranked signals were linked into one exposure group -- never
    by instrument alone (see the module docstring's REDUNDANCY section)."""

    #: Same instrument AND same structural signal family (`signal_rule_id`).
    #: A different family on the same instrument is NOT this relation.
    SAME_STRUCTURAL_FAMILY = "SAME_STRUCTURAL_FAMILY"
    #: Empirically correlated aligned factor series, at or above the
    #: policy's threshold -- regardless of instrument or family.
    CORRELATED_FACTOR = "CORRELATED_FACTOR"


class FitDimension(str, Enum):
    DOMAIN_ACCESS = "DOMAIN_ACCESS"
    INSTRUMENT_ACCESS = "INSTRUMENT_ACCESS"
    SHORTING = "SHORTING"
    HOLDING_PERIOD = "HOLDING_PERIOD"
    TRADING_FREQUENCY = "TRADING_FREQUENCY"
    LIQUIDITY = "LIQUIDITY"
    DRAWDOWN = "DRAWDOWN"


class FitStatus(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    #: Usable only in a restricted form the diagnostics did not measure.
    CONSTRAINED = "CONSTRAINED"
    MISMATCH = "MISMATCH"
    #: The mandate excludes it outright.
    EXCLUDED = "EXCLUDED"
    NO_PREFERENCE = "NO_PREFERENCE"
    #: A preference exists but no evidence measures it yet.
    NOT_MEASURED = "NOT_MEASURED"


class UncertaintyCode(str, Enum):
    SCREENING_ONLY = "SCREENING_ONLY"
    UNCONDITIONAL_EVIDENCE = "UNCONDITIONAL_EVIDENCE"
    EFFECTIVE_SAMPLE = "EFFECTIVE_SAMPLE"
    UNSTABLE_ACROSS_YEARS = "UNSTABLE_ACROSS_YEARS"
    DATA_QUALIFIED = "DATA_QUALIFIED"
    MECHANISM_UNRESOLVED = "MECHANISM_UNRESOLVED"
    REDUNDANT_EXPOSURE = "REDUNDANT_EXPOSURE"
    SHARED_THESIS = "SHARED_THESIS"
    MULTIPLE_TESTING = "MULTIPLE_TESTING"
    NOT_EVALUATED = "NOT_EVALUATED"


# ---------------------------------------------------------------------------
# evidence normalization (peer group = diagnostic method)
# ---------------------------------------------------------------------------


class PeerGroup(BaseModel):
    """The diagnostic METHOD a signal's evidence came from. Raw metrics are
    only ever compared inside one peer group."""

    model_config = {"frozen": True, "extra": "forbid"}

    evidence_method: str
    scope: DiagnosticScope
    ic_kind: ICKind
    structure: Literal["SINGLE_INSTRUMENT_TIME_SERIES"] = "SINGLE_INSTRUMENT_TIME_SERIES"
    frequency: Literal["DAILY"] = "DAILY"
    policy_fingerprint: str
    forward_return_convention: str
    label: str = (
        "Unconditional single-instrument time-series factor screen on daily bars -- the same engine, forward-return "
        "convention and effective-sample t for every domain it covers"
    )

    @property
    def key(self) -> str:
        return "peer1:" + hashlib.sha256(self.model_dump_json(exclude={"label"}).encode()).hexdigest()[:24]


class ScreenMetrics(BaseModel):
    """The raw screening metrics at the declared horizon -- preserved as
    measured, never rescaled."""

    model_config = {"frozen": True, "extra": "forbid"}

    horizon_days: int
    expected_sign: Literal[1, -1]
    n_pairs: int
    n_independent: float
    ts_spearman_ic: float | None
    ts_spearman_t: float | None
    ts_pearson_ic: float | None
    years_with_expected_sign: int
    evaluable_years: int
    subperiod_sign_share: float | None
    ts_spearman_ir_yearly: float | None
    coverage: float
    n_days: int
    undefined_after_warmup: int
    sign_flips_per_year: float | None
    gross_edge_bps: float | None
    drift_bps: float | None
    timing_edge_bps: float | None
    break_even_cost_bps: float | None
    position_change_per_rebalance: float | None


class NormalizedEvidence(BaseModel):
    """Raw metrics + the method that graded them + the peer group they are
    comparable within."""

    model_config = {"frozen": True, "extra": "forbid"}

    peer_group: PeerGroup
    normalization_method: str = FACTOR_SCREEN_NORMALIZATION
    normalization_note: str = (
        "Grades use the screen policy's own thresholds: the effective-sample t (n/h non-overlapping windows) in the "
        "declared direction against the screen's t threshold and 0; the share of years with the declared sign against "
        "the screen's agreement share; the sign of the drift-free timing edge. The t is dimensionless and "
        "sample-size adjusted, so it compares across instruments, domains and horizons measured by this method; "
        "raw ICs and bps are shown, never rescaled."
    )
    metrics: ScreenMetrics
    #: ``ts_spearman_t x expected sign`` -- positive supports the hypothesis.
    aligned_t: float
    t_threshold: float
    direction: EvidenceDirection
    stability: StabilityGrade
    cost_headroom: CostHeadroom


def current_peer_group() -> PeerGroup:
    """The peer group of every screen the current Phase E engine produces."""
    from alpha_agent.screening.factor_diagnostics import FORWARD_RETURN_CONVENTION

    return PeerGroup(
        evidence_method=f"{FACTOR_DIAGNOSTICS_SCHEMA} · {FACTOR_SCREEN_RULE}",
        scope=DiagnosticScope.UNCONDITIONAL_FACTOR, ic_kind=ICKind.TIME_SERIES,
        policy_fingerprint=FactorScreenPolicy().fingerprint(), forward_return_convention=FORWARD_RETURN_CONVENTION,
    )


def _direction(aligned_t: float, threshold: float) -> EvidenceDirection:
    if aligned_t >= threshold:
        return EvidenceDirection.SUPPORTS
    if aligned_t > 0:
        return EvidenceDirection.LEANS_EXPECTED
    if aligned_t > -threshold:
        return EvidenceDirection.LEANS_OPPOSITE
    return EvidenceDirection.CONTRADICTS


def normalize_screen_evidence(screen: CandidateScreen) -> NormalizedEvidence | None:
    """``factor-screen-grades/1``: grade one screen within its peer group, or
    ``None`` when the screen does not belong to the current peer group (other
    schema, rule, scope, IC kind or screening policy) or its statistic is
    undefined."""
    d = screen.diagnostics
    group = current_peer_group()
    own = PeerGroup(
        evidence_method=f"{d.schema_version} · {d.rule}", scope=d.scope, ic_kind=d.ic_kind,
        policy_fingerprint=d.policy_fingerprint, forward_return_convention=d.forward_return_convention,
    )
    if own.key != group.key:
        return None
    dec = d.declared
    if dec.ts_spearman_t is None:
        return None
    evaluable = [p for p in d.subperiods if p.evaluable]
    agree = sum(p.ts_spearman_ic is not None and (p.ts_spearman_ic > 0) == (d.expected_sign > 0) for p in evaluable)
    share, need = d.subperiod_sign_share, d.policy.min_subperiod_sign_share
    if share is None:
        stability = StabilityGrade.NOT_EVALUABLE
    elif share >= need:
        stability = StabilityGrade.CONSISTENT
    elif share <= 1 - need:
        stability = StabilityGrade.OPPOSED
    else:
        stability = StabilityGrade.MIXED
    timing = d.cost.timing_edge_bps
    headroom = (CostHeadroom.NOT_ASSESSED if timing is None
                else CostHeadroom.TIMING_EDGE if timing > 0 else CostHeadroom.NO_TIMING_EDGE)
    aligned = dec.ts_spearman_t * d.expected_sign
    return NormalizedEvidence(
        peer_group=group,
        metrics=ScreenMetrics(
            horizon_days=dec.horizon_days, expected_sign=d.expected_sign, n_pairs=dec.n_pairs,
            n_independent=dec.n_independent, ts_spearman_ic=dec.ts_spearman_ic, ts_spearman_t=dec.ts_spearman_t,
            ts_pearson_ic=dec.ts_pearson_ic, years_with_expected_sign=agree, evaluable_years=len(evaluable),
            subperiod_sign_share=share, ts_spearman_ir_yearly=d.ts_spearman_ir_yearly, coverage=d.coverage.coverage,
            n_days=d.coverage.n_days, undefined_after_warmup=d.coverage.undefined_after_warmup,
            sign_flips_per_year=d.turnover.sign_flips_per_year, gross_edge_bps=d.cost.gross_edge_bps,
            drift_bps=d.cost.drift_bps, timing_edge_bps=timing, break_even_cost_bps=d.cost.break_even_cost_bps,
            position_change_per_rebalance=d.cost.position_change_per_rebalance,
        ),
        aligned_t=round(aligned, 6), t_threshold=d.policy.t_threshold,
        direction=_direction(aligned, d.policy.t_threshold), stability=stability, cost_headroom=headroom,
    )


# ---------------------------------------------------------------------------
# research merit (mandate-independent)
# ---------------------------------------------------------------------------

_STATUS_ORDER = {ScreenStatus.SCREEN_CONTINUE: 2, ScreenStatus.NO_SCREEN_SUPPORT: 1,
                 ScreenStatus.CONTRADICTS_EXPECTED_SIGN: 0}
_DIRECTION_ORDER = {EvidenceDirection.SUPPORTS: 3, EvidenceDirection.LEANS_EXPECTED: 2,
                    EvidenceDirection.LEANS_OPPOSITE: 1, EvidenceDirection.CONTRADICTS: 0}
#: Unknown is neutral -- never worse than a measured failure.
_STABILITY_ORDER = {StabilityGrade.CONSISTENT: 2, StabilityGrade.MIXED: 1, StabilityGrade.NOT_EVALUABLE: 1,
                    StabilityGrade.OPPOSED: 0}
_HEADROOM_ORDER = {CostHeadroom.TIMING_EDGE: 2, CostHeadroom.NOT_ASSESSED: 1, CostHeadroom.NO_TIMING_EDGE: 0}
_QUALITY_ORDER = {DataQualityGrade.CLEAN: 2, DataQualityGrade.QUALIFIED: 1, DataQualityGrade.FLAGGED: 0}
#: ``signal-ranking/1``'s merit dimensions, in the order they are compared.
MERIT_DIMENSIONS: tuple[str, ...] = (
    "screen outcome", "direction of evidence", "subperiod stability", "cost headroom", "data quality",
    "effective-sample t",
)

_STATUS_TEXT = {
    ScreenStatus.SCREEN_CONTINUE: "continue to validation",
    ScreenStatus.NO_SCREEN_SUPPORT: "no screen support",
    ScreenStatus.CONTRADICTS_EXPECTED_SIGN: "contradicts its declared sign",
    ScreenStatus.INSUFFICIENT_DATA: "insufficient data",
}
_DIRECTION_TEXT = {
    EvidenceDirection.SUPPORTS: "supports the declared sign",
    EvidenceDirection.LEANS_EXPECTED: "leans the declared way",
    EvidenceDirection.LEANS_OPPOSITE: "leans the opposite way",
    EvidenceDirection.CONTRADICTS: "contradicts the declared sign",
}


class ResearchMerit(BaseModel):
    """The scientific / diagnostic quality of a screened signal. Computed
    from the screen and the candidate's own data binding only -- never from
    the mandate, the mechanism's confidence or the expression's fidelity."""

    model_config = {"frozen": True, "extra": "forbid"}

    screen_status: ScreenStatus
    screen_note: str
    screen_fingerprint: str
    evidence: NormalizedEvidence
    data_quality: DataQualityGrade
    data_quality_flags: tuple[str, ...] = ()

    def key(self) -> tuple[float, ...]:
        """Higher is better on every component, compared in
        `MERIT_DIMENSIONS` order."""
        e = self.evidence
        return (
            _STATUS_ORDER[self.screen_status], _DIRECTION_ORDER[e.direction], _STABILITY_ORDER[e.stability],
            _HEADROOM_ORDER[e.cost_headroom], _QUALITY_ORDER[self.data_quality], e.aligned_t,
        )

    def component_text(self) -> tuple[str, ...]:
        e, m = self.evidence, self.evidence.metrics
        cost = (f"timing edge {m.timing_edge_bps:+.1f} bp per {m.horizon_days}D hold"
                if m.timing_edge_bps is not None else "cost not assessed")
        return (
            _STATUS_TEXT[self.screen_status], f"{_DIRECTION_TEXT[e.direction]} (t {e.aligned_t:+.2f})",
            f"{m.years_with_expected_sign} of {m.evaluable_years} years with the declared sign", cost,
            self.data_quality.value.lower(), f"t {e.aligned_t:+.2f}",
        )


def _data_quality(candidate: CandidateSignal, screen: CandidateScreen) -> tuple[DataQualityGrade, tuple[str, ...]]:
    flags = [f"feature QA: {q}" for q in screen.series.qa_issues]
    undefined = screen.diagnostics.coverage.undefined_after_warmup
    if undefined:
        flags.append(f"{undefined} undefined factor value(s) after warm-up (never filled)")
    if flags:
        return DataQualityGrade.FLAGGED, tuple(flags)
    status = candidate.spec.data.resolution_status
    if status is ResolutionStatus.AVAILABLE_WITH_PROXY:
        return DataQualityGrade.QUALIFIED, (f"proxy measurement: {candidate.spec.data.proxy_note or 'proxy'}",)
    if status is ResolutionStatus.PARTIAL:
        return DataQualityGrade.QUALIFIED, ("partial history over the research window",)
    return DataQualityGrade.CLEAN, ()


# ---------------------------------------------------------------------------
# user fit (mandate-dependent; never changes merit)
# ---------------------------------------------------------------------------


class FitCheck(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    dimension: FitDimension
    status: FitStatus
    preference: str
    evidence: str
    #: One readable sentence (built from the fields above).
    note: str


class UserFit(BaseModel):
    """How a signal fits the mandate. `admitted=False` removes it from
    portfolio consideration; nothing here touches research merit."""

    model_config = {"frozen": True, "extra": "forbid"}

    mandate_fingerprint: str
    admitted: bool
    exclusion_reasons: tuple[ExclusionReason, ...]
    checks: tuple[FitCheck, ...]
    #: Over the stated soft preferences (holding period, trading frequency),
    #: exactly as `recommendation.fit` uses it.
    personalization_state: PersonalizationState

    @model_validator(mode="after")
    def _admission_consistent(self) -> UserFit:
        if self.admitted == bool(self.exclusion_reasons):
            raise ValueError("a signal is excluded exactly when an exclusion reason exists")
        return self

    def key(self) -> tuple[int, int]:
        """Fewer mismatches, then fewer constraints -- ascending is better."""
        return (sum(c.status is FitStatus.MISMATCH for c in self.checks),
                sum(c.status is FitStatus.CONSTRAINED for c in self.checks))

    def check(self, dimension: FitDimension) -> FitCheck:
        return next(c for c in self.checks if c.dimension is dimension)


def _short_share(screen: CandidateScreen | None) -> float | None:
    """Share of defined days on which the +/-1 expected-sign position is short."""
    if screen is None:
        return None
    sign = screen.diagnostics.expected_sign
    values = [v for v in screen.series.values if v is not None]
    if not values:
        return None
    return sum(sign * v < 0 for v in values) / len(values)


def _flips_per_year(merit: ResearchMerit | None) -> float | None:
    """Position flips per year under the diagnostics' own rebalance-every-h
    model (the model the cost headroom was measured under)."""
    if merit is None or merit.evidence.metrics.position_change_per_rebalance is None:
        return None
    m = merit.evidence.metrics
    return m.position_change_per_rebalance / 2 * _TRADING_DAYS_PER_YEAR / m.horizon_days


def assess_user_fit(
    candidate: CandidateSignal,
    mandate: ResearchMandate,
    *,
    screen: CandidateScreen | None = None,
    merit: ResearchMerit | None = None,
) -> UserFit:
    spec = candidate.spec
    domain, symbol = spec.domain, spec.instrument
    reasons: list[ExclusionReason] = []
    checks: list[FitCheck] = []

    allowed = mandate.allows(domain)
    if not allowed:
        reasons.append(ExclusionReason.DOMAIN_NOT_ALLOWED)
    domains = " + ".join(DOMAIN_LABELS[d] for d in mandate.allowed_domains)
    checks.append(FitCheck(
        dimension=FitDimension.DOMAIN_ACCESS, status=FitStatus.COMPATIBLE if allowed else FitStatus.EXCLUDED,
        preference=domains, evidence=DOMAIN_LABELS[domain],
        note=(f"{DOMAIN_LABELS[domain]} is an allowed domain" if allowed
              else f"{DOMAIN_LABELS[domain]} is not an allowed domain ({domains})"),
    ))
    registry_domain = registry_domain_for(domain)
    allowlist = {i.symbol for i in mandate.instrument_allowlist if i.asset_domain is registry_domain}
    denied = any(i.asset_domain is registry_domain and i.symbol == symbol for i in mandate.instrument_denylist)
    if allowlist and symbol not in allowlist:
        reasons.append(ExclusionReason.INSTRUMENT_NOT_IN_ALLOWLIST)
    if denied:
        reasons.append(ExclusionReason.INSTRUMENT_DENYLISTED)
    outside = bool(allowlist) and symbol not in allowlist
    checks.append(FitCheck(
        dimension=FitDimension.INSTRUMENT_ACCESS,
        status=FitStatus.EXCLUDED if denied or outside else FitStatus.COMPATIBLE,
        preference=(f"limited to {', '.join(sorted(allowlist))}" if allowlist else "whole universe")
        + (" (denylisted)" if denied else ""),
        evidence=symbol,
        note=(f"{symbol} is on your denylist" if denied
              else f"{symbol} is outside your instrument list ({', '.join(sorted(allowlist))})" if outside
              else f"{symbol} is within your instrument scope"),
    ))

    short = _short_share(screen)
    short_text = f"short on {short:.0%} of screened days" if short is not None else "short-side share not measured"
    if mandate.shorting_allowed:
        shorting = FitCheck(dimension=FitDimension.SHORTING, status=FitStatus.COMPATIBLE,
                            preference="shorting allowed", evidence=short_text,
                            note=f"your mandate allows shorting ({short_text})")
    elif short is None:
        shorting = FitCheck(dimension=FitDimension.SHORTING, status=FitStatus.NOT_MEASURED,
                            preference="no shorting", evidence=short_text,
                            note="your mandate allows no shorting; how often this signal is short is not measured yet")
    elif short > 0:
        shorting = FitCheck(
            dimension=FitDimension.SHORTING, status=FitStatus.CONSTRAINED, preference="no shorting",
            evidence=short_text,
            note=(f"needs the short side on {short:.0%} of days, but your mandate allows no shorting -- only a "
                  "long/flat form would be admissible, and that form was not measured"),
        )
    else:
        shorting = FitCheck(dimension=FitDimension.SHORTING, status=FitStatus.COMPATIBLE, preference="no shorting",
                            evidence=short_text, note="never short over the screened window")
    checks.append(shorting)

    profile = mandate.risk_profile
    window = HOLDING_PERIOD_TRADING_DAYS[profile.holding_period]
    h = spec.prediction_horizon.days
    stated = measured = 0
    if window is None:
        checks.append(FitCheck(dimension=FitDimension.HOLDING_PERIOD, status=FitStatus.NO_PREFERENCE,
                               preference=profile.holding_period.value, evidence=f"{h}-day prediction horizon",
                               note="no holding-period preference stated"))
    else:
        stated += 1
        measured += 1
        fits = window[0] <= h <= window[1]
        wanted = f"{profile.holding_period.value} ({window[0]}-{window[1]} trading days)"
        checks.append(FitCheck(
            dimension=FitDimension.HOLDING_PERIOD, status=FitStatus.COMPATIBLE if fits else FitStatus.MISMATCH,
            preference=wanted, evidence=f"{h}-day prediction horizon",
            note=(f"{h}D horizon fits your {wanted} holding period" if fits
                  else f"{h}D horizon vs your {wanted} holding period"),
        ))

    flips = _flips_per_year(merit)
    freq = profile.trading_frequency.value
    if profile.trading_frequency is TradingFrequency.NO_PREFERENCE:
        checks.append(FitCheck(
            dimension=FitDimension.TRADING_FREQUENCY, status=FitStatus.NO_PREFERENCE, preference=freq,
            evidence=f"~{flips:.1f} position flips per year" if flips is not None else "not measured",
            note="no trading-frequency preference stated",
        ))
    else:
        stated += 1
        if flips is None:
            checks.append(FitCheck(dimension=FitDimension.TRADING_FREQUENCY, status=FitStatus.NOT_MEASURED,
                                   preference=freq, evidence="not screened",
                                   note=f"your {freq} trading-frequency preference cannot be checked until screened"))
        else:
            measured += 1
            band = frequency_band(flips)
            evidence = f"~{flips:.1f} position flips per year ({band}) when rebalanced every {h} days"
            checks.append(FitCheck(
                dimension=FitDimension.TRADING_FREQUENCY,
                status=FitStatus.COMPATIBLE if band == freq else FitStatus.MISMATCH, preference=freq,
                evidence=evidence,
                note=(f"~{flips:.1f} position flips/yr fits your {freq} trading frequency" if band == freq
                      else f"~{flips:.1f} position flips/yr ({band}) vs your {freq} trading frequency"),
            ))

    requirement = mandate.liquidity_requirement.value
    checks.append(FitCheck(
        dimension=FitDimension.LIQUIDITY,
        status=FitStatus.NO_PREFERENCE if requirement == "ANY" else FitStatus.NOT_MEASURED,
        preference=requirement.title(),
        evidence="not measured -- screens record prices, not traded notional; the instrument comes from a curated "
                 "liquid universe",
        note=f"your {requirement.title()} liquidity requirement is not applied yet -- no measured liquidity evidence",
    ))
    checks.append(FitCheck(
        dimension=FitDimension.DRAWDOWN, status=FitStatus.NOT_MEASURED,
        preference=f"{profile.max_drawdown.value} max drawdown ({profile.risk_style.value})",
        evidence="not measurable before sizing and a backtest -- a screen has no drawdown",
        note="drawdown cannot be judged before sizing and a backtest",
    ))

    if stated == 0:
        state = PersonalizationState.NOT_PERSONALIZED
    elif measured == stated:
        state = PersonalizationState.MEASURED
    else:
        state = PersonalizationState.PARTIAL
    return UserFit(
        mandate_fingerprint=mandate.fingerprint(), admitted=not reasons, exclusion_reasons=tuple(reasons),
        checks=tuple(checks), personalization_state=state,
    )


# ---------------------------------------------------------------------------
# mechanism provenance (never ranks)
# ---------------------------------------------------------------------------


class MechanismContext(BaseModel):
    """Where the signal came from, across every event that reached it.
    Provenance only: nothing here orders, filters or scores a signal."""

    model_config = {"frozen": True, "extra": "forbid"}

    event_ids: tuple[str, ...]
    event_headlines: tuple[str, ...]
    consequence_states: tuple[str, ...]
    consequence_labels: tuple[str, ...]
    n_paths: int
    path_types: tuple[str, ...]
    researchable_paths: int
    unresolved_or_proposed_paths: int
    opposing_pressures: bool
    fidelities: tuple[str, ...]
    note: str = "Provenance only -- mechanism confidence and expression fidelity never order signals."


def _mechanism(occurrences: Sequence[CandidateSignal]) -> MechanismContext:
    origins = [o for c in occurrences for o in c.origins]
    paths = {o.path_id: o for o in origins}
    type_order = {t: i for i, t in enumerate(PathType)}
    return MechanismContext(
        event_ids=tuple(dict.fromkeys(o.event_id for o in origins)),
        event_headlines=tuple(dict.fromkeys(o.event_headline for o in origins)),
        consequence_states=tuple(dict.fromkeys(o.consequence_state for o in origins)),
        consequence_labels=tuple(dict.fromkeys(o.consequence_label for o in origins)),
        n_paths=len(paths),
        path_types=tuple(t.value if t else "UNCLASSIFIED" for t in sorted(
            {o.path_type for o in origins}, key=lambda t: type_order.get(t, 99))),
        researchable_paths=sum(o.path_status is PathStatus.RESEARCHABLE for o in paths.values()),
        unresolved_or_proposed_paths=sum(o.path_status is not PathStatus.RESEARCHABLE for o in paths.values()),
        opposing_pressures=any(CandidateNote.OPPOSING_PRESSURES in c.notes for c in occurrences),
        fidelities=tuple(dict.fromkeys(o.expression_fidelity.value for o in origins)),
    )


# ---------------------------------------------------------------------------
# the ranked set
# ---------------------------------------------------------------------------


class PairCorrelation(BaseModel):
    """Expected-sign-aligned correlation of two ranked signals' factor series
    (positive: their positions move together)."""

    model_config = {"frozen": True, "extra": "forbid"}

    a: str
    b: str
    n_common: int
    correlation: float | None
    links: bool


class ExposureGroup(BaseModel):
    """Ranked signals that are one exposure: the same instrument, or
    correlated factors -- counted as ONE independent opportunity."""

    model_config = {"frozen": True, "extra": "forbid"}

    group_id: str
    label: str
    lead_id: str
    member_ids: tuple[str, ...]
    instruments: tuple[str, ...]
    domains: tuple[MandateDomain, ...]
    relations: tuple[RedundancyRelation, ...]
    max_abs_correlation: float | None
    #: The events whose theses reach this exposure.
    event_ids: tuple[str, ...] = ()


class EventThesis(BaseModel):
    """One originating event and every ranked exposure it reaches: distinct
    exposures to ONE news hypothesis -- if the event's reading is wrong, all
    of them are. Diversifying across exposures is not diversifying across
    theses. An exposure several events reach appears in each of their
    theses; theses are never chained into one through shared consequences."""

    model_config = {"frozen": True, "extra": "forbid"}

    event_id: str
    event_headline: str
    exposure_group_ids: tuple[str, ...]
    member_ids: tuple[str, ...]
    #: The consequences this event reaches through these signals, most widely shared first.
    consequence_labels: tuple[str, ...]


class Uncertainty(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    code: UncertaintyCode
    text: str


class SignalSummaries(BaseModel):
    """One readable line per aspect -- built from the typed fields beside
    them, never stored as evidence."""

    model_config = {"frozen": True, "extra": "forbid"}

    scientific_quality: str
    user_fit: str
    data_quality: str
    liquidity_cost: str
    redundancy: str
    reason_for_rank: str


class RankedSignal(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    candidate_signal_id: str
    factor_identity: str
    name: str
    instrument: str
    domain: MandateDomain
    expression: str
    prediction_horizon_days: int
    formation_lookback: int
    expected_relationship: str
    tier: RankingTier
    #: 1..n over RANKED signals only; ``None`` in every other tier.
    rank: int | None = None
    role: SignalRole | None = None
    exposure_group_id: str | None = None
    lead_id: str | None = None
    merit: ResearchMerit | None = None
    not_rankable_reason: NotRankableReason | None = None
    user_fit: UserFit
    mechanism: MechanismContext
    shares_thesis_with: tuple[str, ...] = ()
    summaries: SignalSummaries
    uncertainty: tuple[Uncertainty, ...]

    @model_validator(mode="after")
    def _tier_consistent(self) -> RankedSignal:
        ranked = self.tier is RankingTier.RANKED
        if ranked != (self.rank is not None) or ranked != (self.role is not None):
            raise ValueError("only a RANKED signal has a rank and a role")
        if ranked and (self.merit is None or self.not_rankable_reason is not None or not self.user_fit.admitted):
            raise ValueError("a RANKED signal has merit, is admitted, and has no not-rankable reason")
        if self.tier is RankingTier.NOT_RANKABLE and (self.not_rankable_reason is None or self.merit is not None):
            raise ValueError("a NOT_RANKABLE signal has a reason and no merit")
        if self.tier is RankingTier.EXCLUDED_BY_MANDATE and self.user_fit.admitted:
            raise ValueError("an EXCLUDED signal is not admitted")
        if self.role is SignalRole.ALTERNATE and self.lead_id in (None, self.candidate_signal_id):
            raise ValueError("an alternate names another signal as its lead")
        return self


class RankedSignalSet(BaseModel):
    """Every candidate signal of one or more events, ordered for portfolio
    CONSIDERATION under one mandate. Research prioritization only:
    `rank`/`tier` say what deserves attention first, never what may be
    HELD. Portfolio eligibility (capital, margin, risk-budget, position-
    count gates) is a separate, NOT-YET-BUILT allocation gate a later
    phase must add -- this schema deliberately has no field for it."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = SIGNAL_RANKING_SCHEMA
    plane: Literal["RESEARCH_PRIORITIZATION"] = "RESEARCH_PRIORITIZATION"
    policy: SignalRankingPolicy
    policy_fingerprint: str
    mandate_fingerprint: str
    candidate_set_fingerprints: tuple[str, ...]
    event_ids: tuple[str, ...]
    peer_groups: tuple[PeerGroup, ...]
    #: RANKED by rank, then EXCLUDED_BY_MANDATE, then NOT_RANKABLE.
    signals: tuple[RankedSignal, ...]
    exposure_groups: tuple[ExposureGroup, ...]
    event_theses: tuple[EventThesis, ...]
    #: Every ranked pair -- linked pairs first, then by |rho| (display order owned here, not by a UI).
    correlations: tuple[PairCorrelation, ...]
    #: Every distinct candidate considered -- the multiple-testing family.
    family_size: int
    merit_dimensions: tuple[str, ...] = MERIT_DIMENSIONS
    not_portfolio_note: str = (
        "Research prioritization for portfolio CONSIDERATION -- not portfolio weights, position sizes, allocations, "
        "trade instructions, expected returns or verdicts. RANKED is not PORTFOLIO-ELIGIBLE: a later portfolio-"
        "construction phase must apply its own allocation/eligibility gate (capital, margin, risk budget) before any "
        "ranked signal is actually held. Rankings rest on 2018-2022 screening diagnostics; nothing here is validated."
    )

    @model_validator(mode="after")
    def _ordered(self) -> RankedSignalSet:
        ids = [s.candidate_signal_id for s in self.signals]
        if len(ids) != len(set(ids)):
            raise ValueError("one entry per structural candidate")
        ranks = [s.rank for s in self.signals if s.rank is not None]
        if ranks != list(range(1, len(ranks) + 1)) or any(
            s.tier is RankingTier.RANKED for s in self.signals[len(ranks):]
        ):
            raise ValueError("ranked signals come first, numbered 1..n")
        return self

    @property
    def ranked(self) -> tuple[RankedSignal, ...]:
        return tuple(s for s in self.signals if s.tier is RankingTier.RANKED)

    @property
    def excluded(self) -> tuple[RankedSignal, ...]:
        return tuple(s for s in self.signals if s.tier is RankingTier.EXCLUDED_BY_MANDATE)

    @property
    def not_rankable(self) -> tuple[RankedSignal, ...]:
        return tuple(s for s in self.signals if s.tier is RankingTier.NOT_RANKABLE)

    @property
    def leads(self) -> tuple[RankedSignal, ...]:
        return tuple(s for s in self.signals if s.role is SignalRole.LEAD)

    @property
    def independent_exposures(self) -> int:
        return len(self.exposure_groups)

    def headline(self) -> str:
        """One honest sentence about the whole set -- above all, whether
        anything ranked has screening support at all."""
        if not self.signals:
            return "No candidate signals to rank."
        tail = []
        if self.not_rankable:
            tail.append(f"{len(self.not_rankable)} not rankable yet")
        if self.excluded:
            tail.append(f"{len(self.excluded)} excluded by your mandate")
        rest = f" ({', '.join(tail)})" if tail else ""
        if not self.ranked:
            return (f"Nothing ranked yet -- none of the {len(self.signals)} candidate signal(s) has comparable "
                    f"screening evidence{rest}. Missing evidence is not low quality.")
        shared = sum(len(g.event_ids) > 1 for g in self.exposure_groups)
        head = (f"{len(self.ranked)} ranked signal(s) = {self.independent_exposures} independent exposure(s) "
                f"from {len(self.event_theses)} event {'thesis' if len(self.event_theses) == 1 else 'theses'}"
                + (f", {shared} of them reached by more than one event" if shared else "") + f"{rest}.")
        support = sum(s.merit is not None and s.merit.screen_status is ScreenStatus.SCREEN_CONTINUE
                      for s in self.ranked)
        if support:
            return f"{head} {support} with screening support (not validated)."
        return (f"{head} None has screening support -- the order shows which hypotheses are least contradicted "
                "by 2018-2022 data, not which are promising.")

    def signal(self, candidate_signal_id: str) -> RankedSignal:
        return next(s for s in self.signals if s.candidate_signal_id == candidate_signal_id)

    def group(self, group_id: str) -> ExposureGroup:
        return next(g for g in self.exposure_groups if g.group_id == group_id)

    def fingerprint(self) -> str:
        return "ranked1:" + hashlib.sha256(self.model_dump_json().encode()).hexdigest()


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------


class _Entry:
    """Working state for one structural candidate while the set is built."""

    def __init__(self, occurrences: list[CandidateSignal]) -> None:
        self.occurrences = occurrences
        self.candidate = occurrences[0]
        self.screen: CandidateScreen | None = None
        self.merit: ResearchMerit | None = None
        self.reason: NotRankableReason | None = None
        self.fit: UserFit | None = None
        self.tier = RankingTier.NOT_RANKABLE

    @property
    def cid(self) -> str:
        return self.candidate.candidate_signal_id

    def order_key(self) -> tuple:
        assert self.merit is not None and self.fit is not None
        return (*(-x for x in self.merit.key()), *self.fit.key(), self.cid)


def _merge(candidate_sets: Sequence[CandidateSignalSet]) -> dict[str, _Entry]:
    occurrences: dict[str, list[CandidateSignal]] = {}
    for cset in candidate_sets:
        for c in cset.candidates:
            occurrences.setdefault(c.candidate_signal_id, []).append(c)
    return {cid: _Entry(occ) for cid, occ in occurrences.items()}


def _evaluate(entry: _Entry, screen: CandidateScreen | None) -> None:
    c = entry.candidate
    if screen is not None and (screen.candidate_signal_id, screen.expression) != (c.candidate_signal_id, c.expression):
        raise ValueError(f"the screen given for {c.candidate_signal_id} belongs to another candidate")
    entry.screen = screen
    if screen is None:
        entry.reason = (NotRankableReason.NOT_SCREENED if c.spec.domain in SCREENABLE_DOMAINS
                        else NotRankableReason.NO_DIAGNOSTIC_METHOD)
        return
    if screen.series.data_role is not DataProvenanceRole.REAL:
        entry.reason = NotRankableReason.NOT_REAL_EVIDENCE
        return
    if screen.diagnostics.status is ScreenStatus.INSUFFICIENT_DATA:
        entry.reason = NotRankableReason.INSUFFICIENT_EVIDENCE
        return
    evidence = normalize_screen_evidence(screen)
    if evidence is None:
        entry.reason = NotRankableReason.NON_COMPARABLE_EVIDENCE
        return
    quality, flags = _data_quality(c, screen)
    entry.merit = ResearchMerit(
        screen_status=screen.diagnostics.status, screen_note=screen.diagnostics.status_note,
        screen_fingerprint=screen.fingerprint(), evidence=evidence, data_quality=quality, data_quality_flags=flags,
    )


def _aligned_series(entry: _Entry):
    assert entry.screen is not None
    return entry.screen.series.as_series() * entry.screen.diagnostics.expected_sign


def _redundancy(
    ranked: list[_Entry], policy: SignalRankingPolicy,
) -> tuple[tuple[PairCorrelation, ...], dict[str, set[str]], dict[frozenset, RedundancyRelation]]:
    """Pairwise aligned correlations, the linked neighbours of each signal,
    and the strongest relation of every linked pair."""
    by_id = {e.cid: e for e in ranked}
    pairs = signal_correlations({cid: _aligned_series(e) for cid, e in by_id.items()},
                                method=policy.correlation_method)
    neighbours: dict[str, set[str]] = {cid: set() for cid in by_id}
    relation: dict[frozenset, RedundancyRelation] = {}
    out = []
    for p in pairs:
        a, b = by_id[p.a].candidate.spec, by_id[p.b].candidate.spec
        # Same instrument alone is NEVER enough -- the structural signal family (the
        # generation rule) must match too, or two unrelated strategies on one
        # instrument would be forced into one exposure by co-location alone.
        same_family = (a.domain, a.instrument) == (b.domain, b.instrument) and a.signal_rule_id == b.signal_rule_id
        correlated = (p.correlation is not None and p.n_common >= policy.min_common_days
                      and abs(p.correlation) >= policy.redundancy_correlation)
        if same_family or correlated:
            neighbours[p.a].add(p.b)
            neighbours[p.b].add(p.a)
            relation[frozenset((p.a, p.b))] = (RedundancyRelation.SAME_STRUCTURAL_FAMILY if same_family
                                               else RedundancyRelation.CORRELATED_FACTOR)
        out.append(PairCorrelation(a=p.a, b=p.b, n_common=p.n_common,
                                   correlation=round(p.correlation, 6) if p.correlation is not None else None,
                                   links=same_family or correlated))
    # Display order owned here: linked pairs first, then the strongest |rho|.
    out.sort(key=lambda x: (not x.links, -abs(x.correlation) if x.correlation is not None else 0.0, x.a, x.b))
    return tuple(out), neighbours, relation


def _components(neighbours: dict[str, set[str]], order: list[str]) -> list[list[str]]:
    seen: set[str] = set()
    groups = []
    for start in order:
        if start in seen:
            continue
        stack, members = [start], []
        seen.add(start)
        while stack:
            x = stack.pop()
            members.append(x)
            for y in sorted(neighbours[x]):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        groups.append(sorted(members, key=order.index))
    return groups


def _first_difference(a: _Entry, b: _Entry) -> str:
    """The first ranking dimension on which ``a`` is ahead of ``b``."""
    assert a.merit is not None and b.merit is not None and a.fit is not None and b.fit is not None
    ka, kb = a.merit.key(), b.merit.key()
    ta, tb = a.merit.component_text(), b.merit.component_text()
    for label, x, y, sx, sy in zip(MERIT_DIMENSIONS, ka, kb, ta, tb, strict=True):
        if x != y:
            return f"{label}: {sx} vs {sy}"
    fa, fb = a.fit.key(), b.fit.key()
    if fa != fb:
        return (f"fit to your mandate (a merit tie): {fa[0]} mismatch(es), {fa[1]} constraint(s) vs {fb[0]}, "
                f"{fb[1]}")
    return "tied on every ranked dimension -- ordered by identity only, for determinism"


def _pct(x: float) -> str:
    return f"{x:.1%}" if x < 0.9995 else "100%"


def _scientific_summary(entry: _Entry) -> str:
    if entry.merit is None:
        text = NOT_RANKABLE_TEXT[entry.reason] if entry.reason else "Not ranked."
        if entry.reason is NotRankableReason.INSUFFICIENT_EVIDENCE and entry.screen is not None:
            text += f" {entry.screen.diagnostics.status_note}"
        return text
    e, m = entry.merit.evidence, entry.merit.evidence.metrics
    ic = f"{m.ts_spearman_ic:+.3f}" if m.ts_spearman_ic is not None else "n/a"
    return (
        f"{_STATUS_TEXT[entry.merit.screen_status].capitalize()} · TS Spearman IC {ic}, t {m.ts_spearman_t:+.2f} on "
        f"~{m.n_independent:.0f} non-overlapping {m.horizon_days}D windows · {m.years_with_expected_sign} of "
        f"{m.evaluable_years} years with the declared sign · evidence {_DIRECTION_TEXT[e.direction]}"
    )


def _data_summary(entry: _Entry) -> str:
    data = entry.candidate.spec.data
    head = f"{data.dataset} · point-in-time safe · real data"
    if entry.merit is None or entry.screen is None:
        return f"{head} · coverage {data.coverage_start} → {data.coverage_end_exclusive} (not read yet)"
    m = entry.merit.evidence.metrics
    body = (f"{head} · {_pct(m.coverage)} of {m.n_days} screened days defined, "
            f"{m.undefined_after_warmup} undefined after warm-up (never filled)")
    grade = entry.merit.data_quality
    if grade is DataQualityGrade.CLEAN:
        return f"Clean · {body}"
    return f"{grade.value.title()} ({'; '.join(entry.merit.data_quality_flags)}) · {body}"


_LIQUIDITY_NOTE = "liquidity not measured (screens record prices, not traded notional; curated liquid universe)"


def _cost_summary(entry: _Entry) -> str:
    if entry.merit is None:
        return f"Cost sensitivity not measured (not screened) · {_LIQUIDITY_NOTE}"
    m = entry.merit.evidence.metrics
    if m.timing_edge_bps is None:
        return f"Cost sensitivity not assessable · {_LIQUIDITY_NOTE}"
    flips = _flips_per_year(entry.merit)
    turnover = f" · ~{flips:.1f} position flips/yr" if flips is not None else ""
    if m.timing_edge_bps <= 0:
        return (f"No timing edge ({m.timing_edge_bps:+.1f} bp per {m.horizon_days}D hold after "
                f"{m.drift_bps:+.1f} bp drift) -- no trading cost can be afforded{turnover} · {_LIQUIDITY_NOTE}")
    be = (f"break-even one-way cost {m.break_even_cost_bps:.1f} bp" if m.break_even_cost_bps is not None
          else "the position never changes, so turnover cost is not the constraint")
    return (f"Timing edge {m.timing_edge_bps:+.1f} bp per {m.horizon_days}D hold (after {m.drift_bps:+.1f} bp drift); "
            f"{be}{turnover} · {_LIQUIDITY_NOTE}")


_EXCLUSION_TEXT = {
    ExclusionReason.DOMAIN_NOT_ALLOWED: "its domain is not allowed by your mandate",
    ExclusionReason.INSTRUMENT_NOT_IN_ALLOWLIST: "the instrument is outside your mandate's instrument list",
    ExclusionReason.INSTRUMENT_DENYLISTED: "the instrument is on your mandate's denylist",
}


def _fit_summary(fit: UserFit) -> str:
    if not fit.admitted:
        return "Excluded -- " + "; ".join(_EXCLUSION_TEXT[r] for r in fit.exclusion_reasons) + \
            ". Its research merit is unchanged."
    parts = [f"{c.status.value.title()}: {c.note}" for c in fit.checks
             if c.status in (FitStatus.MISMATCH, FitStatus.CONSTRAINED)]
    parts += [c.note for c in fit.checks if c.status is FitStatus.COMPATIBLE
              and c.dimension in (FitDimension.HOLDING_PERIOD, FitDimension.TRADING_FREQUENCY)]
    if fit.personalization_state is PersonalizationState.NOT_PERSONALIZED:
        parts.append("no holding-period or trading-frequency preference stated")
    return "Admitted" + (" · " + " · ".join(parts) if parts else "")


def _uncertainty(entry: _Entry, family_size: int) -> tuple[Uncertainty, ...]:
    out = []
    if entry.merit is None:
        out.append(Uncertainty(code=UncertaintyCode.NOT_EVALUATED, text=(
            "No comparable screening evidence -- its standing is unknown, not low. "
            + (NEXT_STEP_TEXT[entry.reason] if entry.reason else ""))))
    else:
        m = entry.merit.evidence.metrics
        out.append(Uncertainty(code=UncertaintyCode.SCREENING_ONLY, text=(
            "Screening on 2018-2022 discovery data only -- not validated; 2023-2024 and the 2025 holdout are "
            "unread.")))
        out.append(Uncertainty(code=UncertaintyCode.UNCONDITIONAL_EVIDENCE, text=(
            "Unconditional factor evidence -- the factor on every day, not the reaction to this event.")))
        out.append(Uncertainty(code=UncertaintyCode.EFFECTIVE_SAMPLE, text=(
            f"Rests on ~{m.n_independent:.0f} non-overlapping {m.horizon_days}-day windows.")))
        if entry.merit.evidence.stability in (StabilityGrade.MIXED, StabilityGrade.OPPOSED):
            out.append(Uncertainty(code=UncertaintyCode.UNSTABLE_ACROSS_YEARS, text=(
                f"Only {m.years_with_expected_sign} of {m.evaluable_years} years show the declared sign.")))
        if entry.merit.data_quality is not DataQualityGrade.CLEAN:
            out.append(Uncertainty(code=UncertaintyCode.DATA_QUALIFIED, text="; ".join(entry.merit.data_quality_flags)))
    notes = {n for c in entry.occurrences for n in c.notes}
    mech = [t for n, t in (
        (CandidateNote.UNRESOLVED_PATH_ORIGIN, "an originating path is unresolved"),
        (CandidateNote.NO_RESEARCHABLE_PATH, "no originating path is researchable"),
        (CandidateNote.OPPOSING_PRESSURES, "origins imply opposite pressure on the instrument"),
    ) if n in notes]
    if mech:
        out.append(Uncertainty(code=UncertaintyCode.MECHANISM_UNRESOLVED, text=(
            "Mechanism: " + "; ".join(mech) + " (reported, never used to rank).")))
    out.append(Uncertainty(code=UncertaintyCode.MULTIPLE_TESTING, text=(
        f"One of {family_size} candidates considered -- screening applies no multiple-testing correction; "
        "validation must correct across all of them.")))
    return tuple(out)


def rank_candidate_signals(
    candidate_sets: CandidateSignalSet | Sequence[CandidateSignalSet],
    screens: Mapping[str, CandidateScreen],
    mandate: ResearchMandate,
    *,
    policy: SignalRankingPolicy | None = None,
) -> RankedSignalSet:
    """``signal-ranking/1``: rank the candidate signals of one or more events
    for portfolio consideration under ``mandate``. ``screens`` maps
    candidate ids to their screens (a missing id is NOT_SCREENED). Pure and
    deterministic: the same inputs give the same set, in any input order of
    candidates within a set."""
    policy = policy or SignalRankingPolicy()
    sets = (candidate_sets,) if isinstance(candidate_sets, CandidateSignalSet) else tuple(candidate_sets)
    # Canonical input order, so the result never depends on the order events were passed in.
    sets = tuple(sorted(sets, key=lambda x: (x.event_id, x.fingerprint())))
    entries = _merge(sets)
    family_size = len(entries)

    for entry in entries.values():
        _evaluate(entry, screens.get(entry.cid))
        entry.fit = assess_user_fit(entry.candidate, mandate, screen=entry.screen, merit=entry.merit)
        if not entry.fit.admitted:
            entry.tier = RankingTier.EXCLUDED_BY_MANDATE
        elif entry.merit is not None:
            entry.tier = RankingTier.RANKED

    ranked = sorted((e for e in entries.values() if e.tier is RankingTier.RANKED), key=_Entry.order_key)
    correlations, neighbours, relation = _redundancy(ranked, policy)
    by_id = {e.cid: e for e in ranked}
    merit_order = [e.cid for e in ranked]
    groups = _components(neighbours, merit_order)

    # Leads (the independent exposures) first, in merit order; then alternates.
    leads = [g[0] for g in groups]
    alternates = sorted((cid for g in groups for cid in g[1:]), key=merit_order.index)
    order = leads + alternates
    rank_of = {cid: i for i, cid in enumerate(order, start=1)}
    lead_of = {cid: g[0] for g in groups for cid in g}
    corr = {frozenset((p.a, p.b)): p for p in correlations}

    exposure_groups = []
    group_of: dict[str, str] = {}
    for g in groups:
        gid = "exposure1:" + hashlib.sha256("|".join(sorted(g)).encode()).hexdigest()[:16]
        members = sorted(g, key=rank_of.__getitem__)
        rels = {relation[frozenset((a, b))] for i, a in enumerate(members) for b in members[i + 1:]
                if frozenset((a, b)) in relation}
        linked = [abs(corr[frozenset((a, b))].correlation) for i, a in enumerate(members) for b in members[i + 1:]
                  if corr[frozenset((a, b))].correlation is not None]
        specs = [by_id[m].candidate.spec for m in members]
        instruments = tuple(dict.fromkeys(s.instrument for s in specs))
        exposure_groups.append(ExposureGroup(
            group_id=gid, label=" · ".join(instruments), lead_id=g[0], member_ids=tuple(members),
            instruments=instruments, domains=tuple(dict.fromkeys(s.domain for s in specs)),
            relations=tuple(r for r in RedundancyRelation if r in rels),
            max_abs_correlation=round(max(linked), 4) if len(members) > 1 and linked else None,
        ))
        for m in members:
            group_of[m] = gid
    exposure_groups.sort(key=lambda x: rank_of[x.lead_id])

    # Event theses: one per originating event, over the ranked signals it reaches (never chained).
    events_of = {e.cid: {o.event_id for c in e.occurrences for o in c.origins} for e in ranked}
    shares_thesis = {
        cid: tuple(sorted((o for o in by_id if group_of[o] != group_of[cid] and events_of[o] & events_of[cid]),
                          key=rank_of.__getitem__))
        for cid in by_id
    }
    event_theses = []
    for event_id in sorted({x for ev in events_of.values() for x in ev}):
        members = sorted((cid for cid in by_id if event_id in events_of[cid]), key=rank_of.__getitem__)
        origins = [o for cid in members for c in by_id[cid].occurrences for o in c.origins if o.event_id == event_id]
        labels = {o.consequence_state: o.consequence_label for o in origins}
        reach = {st: len({cid for cid in members for c in by_id[cid].occurrences for o in c.origins
                          if o.event_id == event_id and o.consequence_state == st}) for st in labels}
        order_seen = list(labels)
        event_theses.append(EventThesis(
            event_id=event_id, event_headline=origins[0].event_headline,
            exposure_group_ids=tuple(dict.fromkeys(group_of[m] for m in members)), member_ids=tuple(members),
            consequence_labels=tuple(labels[st] for st in sorted(order_seen, key=lambda x: (-reach[x],
                                                                                          order_seen.index(x)))),
        ))
    event_theses.sort(key=lambda t: rank_of[t.member_ids[0]])
    exposure_groups = [g.model_copy(update={"event_ids": tuple(sorted(
        {x for m in g.member_ids for x in events_of[m]}))}) for g in exposure_groups]

    def reason_for_rank(cid: str) -> str:
        e, r = by_id[cid], rank_of[cid]
        if cid in leads:
            i = leads.index(cid)
            text = (f"#{r} of {len(leads)} independent exposure(s)" if len(leads) > 1
                    else "The only independent exposure")
            if i + 1 < len(leads):
                nxt = by_id[leads[i + 1]]
                text += f" -- above #{rank_of[nxt.cid]} {nxt.candidate.name} on {_first_difference(e, nxt)}"
            elif len(leads) > 1:
                text += " -- the lowest-ranked independent exposure"
            group = next(g for g in groups if g[0] == cid)
            if len(group) > 1:
                text += f"; best-merit member of its {len(group)}-signal exposure group"
            return text + "."
        lead = by_id[lead_of[cid]]
        text = (f"Alternate expression of #{rank_of[lead.cid]} {lead.candidate.name}'s exposure -- listed after all "
                f"{len(leads)} independent exposure(s)")
        i = alternates.index(cid)
        if i + 1 < len(alternates):
            nxt = by_id[alternates[i + 1]]
            text += f"; above #{rank_of[nxt.cid]} on {_first_difference(e, nxt)}"
        return text + "."

    def redundancy_text(cid: str) -> str:
        e = by_id[cid]
        group = next(g for g in exposure_groups if g.group_id == group_of[cid])
        if cid == group.lead_id:
            if len(group.member_ids) == 1:
                text = (f"Independent -- no other ranked signal on {e.candidate.spec.instrument} or with "
                        f"|rho| >= {policy.redundancy_correlation:g}")
            else:
                rho = f", |rho| up to {group.max_abs_correlation:.2f}" if group.max_abs_correlation is not None else ""
                text = (f"Lead of exposure {group.label}: {len(group.member_ids) - 1} alternate(s) "
                        f"({', '.join(r.value.replace('_', ' ').lower() for r in group.relations)}{rho})")
        else:
            lead = by_id[group.lead_id]
            # The direct link that ties it in: to the lead if one exists, else its
            # strongest link to any other member (it may join the group only through
            # members ranked below it -- the group is connected, so one exists).
            direct = sorted(neighbours[cid])

            def strength(o: str) -> tuple:
                pair = corr[frozenset((cid, o))]
                same_family = relation[frozenset((cid, o))] is RedundancyRelation.SAME_STRUCTURAL_FAMILY
                return (o != lead.cid, not same_family, -abs(pair.correlation or 0.0), rank_of[o])

            via = min(direct, key=strength)
            pair, rel = corr[frozenset((cid, via))], relation[frozenset((cid, via))]
            if via == lead.cid:
                how = (f"same instrument, same factor family ({e.candidate.spec.instrument})"
                       if rel is RedundancyRelation.SAME_STRUCTURAL_FAMILY
                       else f"aligned factor correlation {pair.correlation:+.2f} on {pair.n_common} days")
            else:
                other = f"#{rank_of[via]} {by_id[via].candidate.name}, itself in this exposure"
                how = (f"same instrument, same factor family as {other}"
                       if rel is RedundancyRelation.SAME_STRUCTURAL_FAMILY
                       else f"aligned factor correlation {pair.correlation:+.2f} on {pair.n_common} days with {other}")
            text = f"Alternate of #{rank_of[lead.cid]} {lead.candidate.name} -- {how}"
        other_leads = [o for o in shares_thesis[cid] if o == lead_of[o]]
        if other_leads:
            text += " · same event thesis as " + ", ".join(f"#{rank_of[o]}" for o in other_leads) + \
                ", a different exposure"
        return text

    def unranked_redundancy(entry: _Entry) -> str:
        spec = entry.candidate.spec
        same_instrument = [o for o in ranked if (o.candidate.spec.domain, o.candidate.spec.instrument) ==
                           (spec.domain, spec.instrument)]
        if same_instrument:
            other = same_instrument[0]
            family_note = ("same factor family" if other.candidate.spec.signal_rule_id == spec.signal_rule_id
                          else "a DIFFERENT factor family -- not established as the same exposure")
            return (f"Not ranked -- same instrument as ranked #{rank_of[other.cid]} {other.candidate.name} "
                    f"({family_note}); factor correlation not measured")
        return "Not ranked -- correlation with ranked signals not measured"

    def build(entry: _Entry) -> RankedSignal:
        c, spec = entry.candidate, entry.candidate.spec
        ranked_here = entry.tier is RankingTier.RANKED
        cid = entry.cid
        if ranked_here:
            reason, redundancy = reason_for_rank(cid), redundancy_text(cid)
        elif entry.tier is RankingTier.EXCLUDED_BY_MANDATE:
            reason, redundancy = "Not ranked -- excluded by your mandate.", "Not assessed (excluded by your mandate)"
        else:
            reason = f"Not ranked -- {NOT_RANKABLE_TEXT[entry.reason]} {NEXT_STEP_TEXT[entry.reason]}"
            redundancy = unranked_redundancy(entry)
        uncertainty = list(_uncertainty(entry, family_size))
        if ranked_here and cid != lead_of[cid]:
            uncertainty.append(Uncertainty(code=UncertaintyCode.REDUNDANT_EXPOSURE, text=(
                f"Not an independent opportunity -- the same exposure as #{rank_of[lead_of[cid]]}.")))
        if ranked_here and shares_thesis[cid]:
            uncertainty.append(Uncertainty(code=UncertaintyCode.SHARED_THESIS, text=(
                "Depends on the same originating event as "
                + ", ".join(f"#{rank_of[o]}" for o in shares_thesis[cid]) + " -- one event thesis, several exposures.")))
        assert entry.fit is not None
        return RankedSignal(
            candidate_signal_id=cid, factor_identity=c.factor_identity, name=c.name, instrument=spec.instrument,
            domain=spec.domain, expression=c.expression, prediction_horizon_days=spec.prediction_horizon.days,
            formation_lookback=spec.formation_lookback, expected_relationship=spec.expected_relationship.value,
            tier=entry.tier, rank=rank_of.get(cid) if ranked_here else None,
            role=(SignalRole.LEAD if lead_of[cid] == cid else SignalRole.ALTERNATE) if ranked_here else None,
            exposure_group_id=group_of.get(cid), lead_id=lead_of.get(cid), merit=entry.merit,
            not_rankable_reason=entry.reason, user_fit=entry.fit, mechanism=_mechanism(entry.occurrences),
            shares_thesis_with=shares_thesis.get(cid, ()),
            summaries=SignalSummaries(
                scientific_quality=_scientific_summary(entry), user_fit=_fit_summary(entry.fit),
                data_quality=_data_summary(entry), liquidity_cost=_cost_summary(entry), redundancy=redundancy,
                reason_for_rank=reason,
            ),
            uncertainty=tuple(uncertainty),
        )

    domain_order = {d: i for i, d in enumerate(MandateDomain)}

    def unranked_key(e: _Entry) -> tuple:
        return (domain_order[e.candidate.spec.domain], e.candidate.spec.instrument,
                e.candidate.spec.prediction_horizon.days, e.candidate.spec.formation_lookback, e.cid)

    excluded = sorted((e for e in entries.values() if e.tier is RankingTier.EXCLUDED_BY_MANDATE), key=unranked_key)
    not_rankable = sorted((e for e in entries.values() if e.tier is RankingTier.NOT_RANKABLE), key=unranked_key)
    signals = tuple(build(by_id[cid]) for cid in order) + tuple(build(e) for e in (*excluded, *not_rankable))
    peer = tuple({s.merit.evidence.peer_group.key: s.merit.evidence.peer_group
                  for s in signals if s.merit is not None}.values())
    return RankedSignalSet(
        policy=policy, policy_fingerprint=policy.fingerprint(), mandate_fingerprint=mandate.fingerprint(),
        candidate_set_fingerprints=tuple(dict.fromkeys(s.fingerprint() for s in sets)),
        event_ids=tuple(dict.fromkeys(s.event_id for s in sets)), peer_groups=peer, signals=signals,
        exposure_groups=tuple(exposure_groups), event_theses=tuple(event_theses), correlations=correlations,
        family_size=family_size,
    )
