"""News Alpha Phase G -- the PORTFOLIO PLAN and the path that builds it.

    RankedSignalSet (Phase F) + cached screens + ResearchMandate
      -> selection.screen_eligibility          which signals may enter
      -> market snapshots (real, cached)       which day is "today" (as-of)
      -> selection.resolve_directions          which way each points today
      -> selection.apply_count_limits          priority by rank
      -> risk_model.estimate_risk_model        statistics
      -> allocator.run_allocator  (C++)        sizes, limits, units
      -> PortfolioPlan

A plan is CONSTRUCTION, not validation: it says how capital would be
allocated to these signals under this mandate on the as-of date. It carries
no verdict, no expected return and no claim that the portfolio works --
that is Phase H. Every size in it is the C++ allocator's, copied verbatim.

Every way a plan can come out empty is a typed status with its reason --
no eligible signal, no market data, too little history, the C++ allocator
not built -- never an exception at a user and never a silent fallback.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel

from alpha_agent.news_alpha.mandate import ResearchMandate
from alpha_agent.portfolio.allocator import (
    AllocationOutput,
    AllocatorError,
    AllocatorInputs,
    AllocatorUnavailable,
    build_allocator_inputs,
    run_allocator,
)
from alpha_agent.portfolio.policy import (
    ELIGIBILITY_TEXT,
    EligibilityMode,
    PortfolioConstraints,
    PortfolioConstructionPolicy,
    resolve_constraints,
)
from alpha_agent.portfolio.risk_model import (
    InstrumentMarketSnapshot,
    InsufficientRiskData,
    RiskModel,
    SnapshotProvider,
    common_as_of,
    estimate_risk_model,
    fingerprint_payload,
)
from alpha_agent.portfolio.selection import (
    REJECTION_TEXT,
    FactorSource,
    RejectedSignal,
    RejectionReason,
    RejectionStage,
    SelectedSignal,
    apply_count_limits,
    resolve_directions,
    screen_eligibility,
)
from alpha_agent.recommendation.signal_ranking import RankedSignalSet

__all__ = [
    "PORTFOLIO_PLAN_SCHEMA",
    "PlanStatus",
    "PortfolioPlan",
    "construct_portfolio_plan",
]

PORTFOLIO_PLAN_SCHEMA = "portfolio-plan/1"


class PlanStatus(str, Enum):
    CONSTRUCTED = "CONSTRUCTED"
    #: Allocated, but every target rounds to zero units (or was removed).
    NO_EXECUTABLE_POSITION = "NO_EXECUTABLE_POSITION"
    NO_ELIGIBLE_SIGNAL = "NO_ELIGIBLE_SIGNAL"
    NO_ALLOCATABLE_SIGNAL = "NO_ALLOCATABLE_SIGNAL"
    #: No equal-risk-contribution solution: two exposures hedge each other exactly.
    DEGENERATE_RISK_MODEL = "DEGENERATE_RISK_MODEL"
    RISK_INPUTS_UNAVAILABLE = "RISK_INPUTS_UNAVAILABLE"
    ALLOCATOR_UNAVAILABLE = "ALLOCATOR_UNAVAILABLE"


class PortfolioPlan(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = PORTFOLIO_PLAN_SCHEMA
    plane: Literal["PORTFOLIO_CONSTRUCTION"] = "PORTFOLIO_CONSTRUCTION"
    method: str
    status: PlanStatus
    status_detail: str
    eligibility: EligibilityMode
    as_of: date | None
    mandate_fingerprint: str
    ranked_set_fingerprint: str
    policy: PortfolioConstructionPolicy
    policy_fingerprint: str
    constraints: PortfolioConstraints
    risk_model: RiskModel | None
    #: Signals that reached the allocator, in rank order.
    selected: tuple[SelectedSignal, ...]
    #: Every other signal of the ranked set, with its typed reason.
    rejected: tuple[RejectedSignal, ...]
    allocator_inputs: AllocatorInputs | None
    #: The C++ allocator's output, verbatim.
    allocation: AllocationOutput | None
    validation_note: str = (
        "Portfolio construction, not validation: how capital would be allocated to these signals under your mandate "
        "on the as-of date. No expected return, no verdict -- whether the portfolio works is tested separately "
        "(Phase H), on data this plan never read."
    )

    def fingerprint(self) -> str:
        data = self.model_dump(mode="json")
        if self.risk_model is not None:
            # a DISCOVERY-window plan fingerprints exactly as before windows existed
            data["risk_model"] = fingerprint_payload(self.risk_model)
        payload = json.dumps(data, sort_keys=True)
        return "portplan1:" + hashlib.sha256(payload.encode()).hexdigest()

    # -- read-only views (formatting helpers for renderers; no sizing) -------

    @property
    def is_exploratory(self) -> bool:
        return self.eligibility is EligibilityMode.EXPLORATORY

    @property
    def positions(self):
        """Instruments with a non-zero executable position, largest risk first."""
        if self.allocation is None:
            return ()
        held = [i for i in self.allocation.instruments if i.units != 0]
        return tuple(sorted(held, key=lambda i: (-abs(i.risk_contribution), i.instrument_key)))

    def selected_signal(self, candidate_signal_id: str) -> SelectedSignal:
        return next(s for s in self.selected if s.candidate_signal_id == candidate_signal_id)

    def headline(self) -> str:
        mode = "Exploratory preview" if self.is_exploratory else "Qualified plan"
        if self.status is PlanStatus.NO_ELIGIBLE_SIGNAL:
            return f"{mode}: no signal can enter the portfolio -- {self.status_detail}"
        if self.status in (PlanStatus.RISK_INPUTS_UNAVAILABLE, PlanStatus.ALLOCATOR_UNAVAILABLE,
                           PlanStatus.NO_ALLOCATABLE_SIGNAL, PlanStatus.DEGENERATE_RISK_MODEL):
            return f"{mode}: not constructed -- {self.status_detail}"
        a = self.allocation
        assert a is not None
        exposures = sum(c.allocated for c in a.clusters)
        held = len(self.positions)
        vol = a.executable.annual_vol
        head = (f"{mode} as of {self.as_of}: {len(self.selected)} signal(s) in {exposures} independent exposure(s) "
                f"-> {held} position(s), ex-ante volatility {vol:.1%} vs a {a.target_annual_vol:.1%} target, "
                f"gross {a.executable.gross_exposure:.2f}x")
        if self.status is PlanStatus.NO_EXECUTABLE_POSITION:
            return head + " -- every target rounds to zero units."
        return head + "."


def _plan(**kw) -> PortfolioPlan:
    policy: PortfolioConstructionPolicy = kw["policy"]
    return PortfolioPlan(method=policy.method, eligibility=policy.eligibility, policy_fingerprint=policy.fingerprint(),
                         **kw)


def _no_eligible_detail(rejected: list[RejectedSignal], total: int, mode: EligibilityMode) -> str:
    if total == 0:
        return "the ranked set is empty."
    counts: dict[RejectionReason, int] = {}
    for r in rejected:
        counts[r.reason] = counts.get(r.reason, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].value))
    parts = ", ".join(f"{n} {reason.value.replace('_', ' ').lower()}" for reason, n in top[:4])
    text = f"0 of {total} signal(s) eligible ({parts})."
    if mode is EligibilityMode.QUALIFIED and counts.get(RejectionReason.NO_SCREEN_SUPPORT):
        text += " None has screening support yet."
    return text


def construct_portfolio_plan(
    ranked: RankedSignalSet,
    screens: Mapping[str, FactorSource],
    mandate: ResearchMandate,
    *,
    snapshots: SnapshotProvider,
    policy: PortfolioConstructionPolicy | None = None,
    as_of: date | None = None,
    allocator_cli=None,
) -> PortfolioPlan:
    """``portfolio-construction/1``: the plan for ``ranked`` under ``mandate``.
    ``snapshots`` supplies real market snapshots (a `MarketSnapshotStore`
    provider in production). ``screens`` supplies each signal's factor series:
    the Phase E screens (discovery) when constructing, or the same factors
    evaluated causally through the validation window when Phase H re-sizes a
    frozen portfolio at a later date. Eligibility always comes from ``ranked``
    alone. Deterministic for identical inputs."""
    policy = policy or PortfolioConstructionPolicy()
    if ranked.mandate_fingerprint != mandate.fingerprint():
        raise ValueError("the ranked set was ranked under a different mandate")
    constraints = resolve_constraints(mandate, policy)
    common = {"policy": policy, "mandate_fingerprint": mandate.fingerprint(),
              "ranked_set_fingerprint": ranked.fingerprint(), "constraints": constraints}

    eligible, rejected = screen_eligibility(ranked, policy)
    if not eligible:
        return _plan(status=PlanStatus.NO_ELIGIBLE_SIGNAL,
                     status_detail=_no_eligible_detail(rejected, len(ranked.signals), policy.eligibility),
                     as_of=None, risk_model=None, selected=(), rejected=tuple(rejected), allocator_inputs=None,
                     allocation=None, **common)

    # Market data for every eligible instrument; an instrument without it cannot be sized.
    loaded: dict[str, InstrumentMarketSnapshot] = {}
    missing: dict[str, str] = {}
    for s in eligible:
        if s.instrument_key in loaded or s.instrument_key in missing:
            continue
        try:
            loaded[s.instrument_key] = snapshots(s.domain, s.instrument)
        except InsufficientRiskData as exc:
            missing[s.instrument_key] = str(exc)
    for s in eligible:
        if s.instrument_key in missing:
            rejected.append(RejectedSignal(
                candidate_signal_id=s.candidate_signal_id, name=s.name, domain=s.domain, instrument=s.instrument,
                rank=s.rank, stage=RejectionStage.SELECTION, reason=RejectionReason.NO_MARKET_DATA,
                detail=f"{REJECTION_TEXT[RejectionReason.NO_MARKET_DATA]} {missing[s.instrument_key]}"))
    eligible = [s for s in eligible if s.instrument_key in loaded]
    if not eligible:
        return _plan(status=PlanStatus.NO_ELIGIBLE_SIGNAL,
                     status_detail=_no_eligible_detail(rejected, len(ranked.signals), policy.eligibility),
                     as_of=None, risk_model=None, selected=(), rejected=tuple(rejected), allocator_inputs=None,
                     allocation=None, **common)

    day = as_of or common_as_of(list(loaded.values()))
    directed, dropped = resolve_directions(eligible, screens, as_of=day, mandate=mandate, ranked=ranked)
    rejected += dropped
    selected, limited = apply_count_limits(directed, policy, ranked)
    rejected += limited
    if not selected:
        return _plan(status=PlanStatus.NO_ELIGIBLE_SIGNAL,
                     status_detail=_no_eligible_detail(rejected, len(ranked.signals), policy.eligibility),
                     as_of=day, risk_model=None, selected=(), rejected=tuple(rejected), allocator_inputs=None,
                     allocation=None, **common)

    keys = sorted({s.instrument_key for s in selected})
    try:
        risk = estimate_risk_model({k: loaded[k] for k in keys}, as_of=day, lookback_days=policy.risk_lookback_days,
                                   min_observations=policy.min_risk_observations,
                                   annualization_days=policy.annualization_days)
    except InsufficientRiskData as exc:
        return _plan(status=PlanStatus.RISK_INPUTS_UNAVAILABLE, status_detail=str(exc), as_of=day, risk_model=None,
                     selected=tuple(selected), rejected=tuple(rejected), allocator_inputs=None, allocation=None,
                     **common)

    inputs = build_allocator_inputs(selected, risk, constraints)
    try:
        allocation = run_allocator(inputs, cli=allocator_cli)
    except AllocatorUnavailable as exc:
        return _plan(status=PlanStatus.ALLOCATOR_UNAVAILABLE, status_detail=str(exc), as_of=day, risk_model=risk,
                     selected=tuple(selected), rejected=tuple(rejected), allocator_inputs=inputs, allocation=None,
                     **common)
    except AllocatorError as exc:
        # A malformed input is a bug in this pipeline, never a property of the portfolio.
        raise RuntimeError(f"the C++ allocator refused the plan's input: {exc}") from exc

    # The allocator's own refusals, per signal.
    for s in selected:
        out = allocation.signal(s.candidate_signal_id)
        if out.allocated:
            continue
        reason = RejectionReason(out.reason)
        detail = REJECTION_TEXT[reason]
        if reason is RejectionReason.INSTRUMENT_NOT_ADMITTED:
            why = allocation.instrument(s.instrument_key).reason
            detail = f"The allocator did not admit {s.instrument}: {why.replace('_', ' ').lower()}."
        rejected.append(RejectedSignal(
            candidate_signal_id=s.candidate_signal_id, name=s.name, domain=s.domain, instrument=s.instrument,
            rank=s.rank, stage=RejectionStage.ALLOCATION, reason=reason, detail=detail))

    status = PlanStatus(allocation.status)
    details = {
        PlanStatus.CONSTRUCTED: ELIGIBILITY_TEXT[policy.eligibility],
        PlanStatus.NO_EXECUTABLE_POSITION: "every target is smaller than one tradable unit at this capital",
        PlanStatus.NO_ALLOCATABLE_SIGNAL: "the allocator admitted none of the selected signals",
        PlanStatus.DEGENERATE_RISK_MODEL: ("two exposures hedge each other exactly, so equal risk contributions do "
                                           "not exist -- nothing is sized"),
    }
    return _plan(status=status, status_detail=details[status], as_of=day, risk_model=risk, selected=tuple(selected),
                 rejected=tuple(rejected), allocator_inputs=inputs, allocation=allocation, **common)
