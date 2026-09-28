"""News Alpha Phase H -- the TESTED PORTFOLIO: a frozen strategy and its causal
rebalance schedule.

A `PortfolioPlan` (Phase G) is one allocation on one day. Holding that one
allocation for two years would test a basket, not the signals that chose it,
and would produce a handful of fills. What Phase H tests is the portfolio the
plan DEFINES:

    frozen before any validation-window data is read (`PortfolioStrategySpec`)
      members      the ranked set's eligible signals under the plan's policy
                   (stage 1 of Phase G selection) -- never re-selected later
      policy       the plan's construction policy + constraints + mandate
      rebalance    every N common trading days (`RebalanceRule`, a convention)

    re-derived causally at each rebalance day t (`build_rebalance_schedule`)
      direction    each member's own factor on t (x its declared sign)
      sizing       Phase G `construct_portfolio_plan` as of t: the risk model
                   from the trailing window ending t, the C++ allocator
      targets      integer units per root at t's decision bar -> the C++ engine
                   fills at the next bar

Selection never sees validation data; direction and sizing only ever see data
up to the rebalance day (the estimator slices at its as-of date and the
factor is a trailing operator). A day on which nothing can be sized is a
typed decision to be flat -- never a carried-over or invented position.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from alpha_agent.news_alpha.mandate import MandateDomain, ResearchMandate
from alpha_agent.portfolio.allocator import AllocatorUnavailable
from alpha_agent.portfolio.plan import PlanStatus, PortfolioPlan, construct_portfolio_plan
from alpha_agent.portfolio.policy import (
    EligibilityMode,
    PortfolioConstraints,
    PortfolioConstructionPolicy,
)
from alpha_agent.portfolio.risk_model import (
    InstrumentMarketSnapshot,
    InsufficientRiskData,
    SnapshotProvider,
    SnapshotWindow,
)
from alpha_agent.portfolio.selection import FactorSource, screen_eligibility
from alpha_agent.recommendation.signal_ranking import RankedSignalSet, SignalRole

__all__ = [
    "PORTFOLIO_STRATEGY_SCHEMA",
    "REBALANCE_METHOD",
    "PortfolioStrategySpec",
    "RebalanceDecision",
    "RebalanceRule",
    "RebalanceSchedule",
    "StrategyMember",
    "build_rebalance_schedule",
    "execution_root",
    "freeze_portfolio_strategy",
]

PORTFOLIO_STRATEGY_SCHEMA = "portfolio-strategy/1"
REBALANCE_METHOD = "portfolio-rebalance/1"


def _sha(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def execution_root(domain: MandateDomain, symbol: str) -> str:
    """The root a target schedule names: the futures root, or the ETF's synthetic ``E<ticker>``."""
    if domain is MandateDomain.ETF:
        from alpha_agent.etf.universe import etf_dsl_root_symbol

        return etf_dsl_root_symbol(symbol)
    return symbol


class RebalanceRule(BaseModel):
    """When the frozen portfolio is re-sized. A convention fixed before any
    validation-window data is read -- not fitted, not evidence."""

    model_config = {"frozen": True, "extra": "forbid"}

    method: Literal["portfolio-rebalance/1"] = REBALANCE_METHOD
    #: Rebalance on every N-th trading day every instrument of the portfolio
    #: traded, starting with the first one of the span.
    every_trading_days: int = Field(default=20, ge=1)

    def identity(self) -> str:
        return "rebalance1:" + _sha(self.model_dump(mode="json"))


class StrategyMember(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    candidate_signal_id: str
    name: str
    domain: MandateDomain
    instrument: str
    instrument_key: str
    execution_root: str
    expression: str
    prediction_horizon_days: int
    expected_sign: int
    exposure_group_id: str
    rank: int
    role: SignalRole
    event_ids: tuple[str, ...]


class PortfolioStrategySpec(BaseModel):
    """The tested portfolio's pre-run definition. Its fingerprint is the
    strategy fingerprint of every Phase H backtest and registry experiment."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = PORTFOLIO_STRATEGY_SCHEMA
    plane: Literal["PORTFOLIO_STRATEGY"] = "PORTFOLIO_STRATEGY"
    source_plan_fingerprint: str
    source_plan_as_of: date | None
    ranked_set_fingerprint: str
    mandate_fingerprint: str
    eligibility: EligibilityMode
    policy: PortfolioConstructionPolicy
    constraints: PortfolioConstraints
    members: tuple[StrategyMember, ...]
    rebalance: RebalanceRule
    scope_note: str = (
        "A result for this strategy applies to this portfolio only -- its members, policy, constraints and "
        "rebalance rule together. It never validates a member signal, proves a signal path, or establishes an "
        "economic mechanism."
    )

    def fingerprint(self) -> str:
        return "portstrat1:" + _sha(self.model_dump(mode="json"))

    @property
    def instrument_keys(self) -> tuple[str, ...]:
        return tuple(sorted({m.instrument_key for m in self.members}))

    @property
    def domains(self) -> tuple[MandateDomain, ...]:
        return tuple(sorted({m.domain for m in self.members}, key=lambda d: d.value))

    @property
    def execution_roots(self) -> dict[str, str]:
        """instrument_key -> execution root."""
        return {m.instrument_key: m.execution_root for m in self.members}

    def instrument(self, key: str) -> StrategyMember:
        return next(m for m in self.members if m.instrument_key == key)

    def with_rebalance(self, rule: RebalanceRule) -> PortfolioStrategySpec:
        return self.model_copy(update={"rebalance": rule})


def freeze_portfolio_strategy(
    plan: PortfolioPlan, ranked: RankedSignalSet, *, rebalance: RebalanceRule | None = None,
) -> PortfolioStrategySpec:
    """The portfolio ``plan`` defines, frozen. Members are the ranked set's
    eligible signals under the plan's own policy -- the set construction
    chose from, not just the ones that happened to point the right way on the
    plan's as-of date."""
    if plan.ranked_set_fingerprint != ranked.fingerprint():
        raise ValueError("the plan was built from a different ranked set")
    eligible, _ = screen_eligibility(ranked, plan.policy)
    if not eligible:
        raise ValueError("the plan's policy admits no signal of the ranked set -- there is no portfolio to freeze")
    members = tuple(StrategyMember(
        candidate_signal_id=s.candidate_signal_id, name=s.name, domain=s.domain, instrument=s.instrument,
        instrument_key=s.instrument_key, execution_root=execution_root(s.domain, s.instrument),
        expression=s.expression, prediction_horizon_days=s.prediction_horizon_days, expected_sign=s.expected_sign,
        exposure_group_id=s.exposure_group_id, rank=s.rank, role=s.role, event_ids=s.event_ids,
    ) for s in sorted(eligible, key=lambda s: (s.rank, s.candidate_signal_id)))
    return PortfolioStrategySpec(
        source_plan_fingerprint=plan.fingerprint(), source_plan_as_of=plan.as_of,
        ranked_set_fingerprint=plan.ranked_set_fingerprint, mandate_fingerprint=plan.mandate_fingerprint,
        eligibility=plan.eligibility, policy=plan.policy, constraints=plan.constraints, members=members,
        rebalance=rebalance or RebalanceRule(),
    )


# ---------------------------------------------------------------------------
# the causal rebalance schedule
# ---------------------------------------------------------------------------


class RebalanceDecision(BaseModel):
    """One rebalance day: the Phase G plan as of that day, reduced to what
    execution and audit need (the plan itself is reproducible from its
    inputs and fingerprint)."""

    model_config = {"frozen": True, "extra": "forbid"}

    day: date
    plan_fingerprint: str
    status: PlanStatus
    status_detail: str
    #: instrument_key -> target units (every instrument of the strategy; 0 = flat).
    units: dict[str, int]
    #: instrument_key -> the decision bar's timestamp on ``day``.
    decision_ts_ns: dict[str, int]
    #: candidate_signal_id -> +1 / -1 for the signals that were sized.
    directions: dict[str, int]
    ex_ante_annual_vol: float | None
    gross_exposure: float | None


class ScheduleRow(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    ts_event_ns: int
    root_symbol: str
    target_units: int
    instrument_key: str


class RebalanceSchedule(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "rebalance-schedule/1"
    strategy_fingerprint: str
    span_start: date
    span_end_exclusive: date
    rebalance: RebalanceRule
    decisions: tuple[RebalanceDecision, ...]

    def rows(self, roots: Mapping[str, str]) -> tuple[ScheduleRow, ...]:
        out = [ScheduleRow(ts_event_ns=d.decision_ts_ns[key], root_symbol=roots[key], target_units=units,
                           instrument_key=key)
               for d in self.decisions for key, units in sorted(d.units.items())]
        return tuple(sorted(out, key=lambda r: (r.ts_event_ns, r.root_symbol)))

    def schedule_hash(self) -> str:
        payload = [[d.day.isoformat(), sorted(d.units.items()), sorted(d.decision_ts_ns.items())]
                   for d in self.decisions]
        return "portsched1:" + _sha({"strategy": self.strategy_fingerprint, "decisions": payload})

    def status_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.decisions:
            out[d.status.value] = out.get(d.status.value, 0) + 1
        return out


def _memoized(provider: SnapshotProvider) -> SnapshotProvider:
    cache: dict[tuple[MandateDomain, str], InstrumentMarketSnapshot] = {}

    def get(domain: MandateDomain, symbol: str) -> InstrumentMarketSnapshot:
        key = (domain, symbol)
        if key not in cache:
            cache[key] = provider(domain, symbol)
        return cache[key]
    return get


def rebalance_days(
    snapshots: Mapping[str, InstrumentMarketSnapshot], *, start: date, end_exclusive: date, every: int,
) -> tuple[date, ...]:
    """Every ``every``-th day in ``[start, end)`` on which every instrument traded."""
    common = set.intersection(*(set(s.days) for s in snapshots.values()))
    days = sorted(d for d in common if start <= date.fromisoformat(d) < end_exclusive)
    return tuple(date.fromisoformat(d) for d in days[::every])


def build_rebalance_schedule(
    strategy: PortfolioStrategySpec,
    ranked: RankedSignalSet,
    mandate: ResearchMandate,
    *,
    factor_sources: Mapping[str, FactorSource],
    snapshots: SnapshotProvider,
    start: date,
    end_exclusive: date,
    allocator_cli=None,
) -> RebalanceSchedule:
    """The strategy re-sized on every rebalance day of ``[start, end)``.
    ``snapshots`` must serve EVALUATION-window snapshots when the span
    reaches past the discovery window; ``factor_sources`` must cover every
    member (the evaluation series, see `portfolio.evaluation`)."""
    if ranked.fingerprint() != strategy.ranked_set_fingerprint:
        raise ValueError("the strategy was frozen from a different ranked set")
    if mandate.fingerprint() != strategy.mandate_fingerprint:
        raise ValueError("the strategy was frozen under a different mandate")
    missing = [m.candidate_signal_id for m in strategy.members if m.candidate_signal_id not in factor_sources]
    if missing:
        raise ValueError(f"no factor series for {len(missing)} member(s): {missing[:3]}")
    provider = _memoized(snapshots)
    snaps = {k: provider(strategy.instrument(k).domain, strategy.instrument(k).instrument)
             for k in strategy.instrument_keys}
    if any(s.window is SnapshotWindow.DISCOVERY and end_exclusive > s.window.end for s in snaps.values()):
        raise InsufficientRiskData("a span past the discovery window needs EVALUATION-window snapshots")
    days = rebalance_days(snaps, start=start, end_exclusive=end_exclusive,
                          every=strategy.rebalance.every_trading_days)
    ts_by_day = {k: dict(zip(s.days, s.ts_event_ns, strict=True)) for k, s in snaps.items()}

    decisions: list[RebalanceDecision] = []
    for day in days:
        plan = construct_portfolio_plan(ranked, factor_sources, mandate, snapshots=provider,
                                        policy=strategy.policy, as_of=day, allocator_cli=allocator_cli)
        if plan.status is PlanStatus.ALLOCATOR_UNAVAILABLE:
            raise AllocatorUnavailable(plan.status_detail)
        units = dict.fromkeys(strategy.instrument_keys, 0)
        directions: dict[str, int] = {}
        vol = gross = None
        if plan.allocation is not None and plan.status in (PlanStatus.CONSTRUCTED, PlanStatus.NO_EXECUTABLE_POSITION):
            for i in plan.allocation.instruments:
                units[i.instrument_key] = i.units
            directions = {s.candidate_signal_id: int(s.direction) for s in plan.selected
                          if s.direction is not None and plan.allocation.signal(s.candidate_signal_id).allocated}
            vol, gross = plan.allocation.executable.annual_vol, plan.allocation.executable.gross_exposure
        iso = day.isoformat()
        decisions.append(RebalanceDecision(
            day=day, plan_fingerprint=plan.fingerprint(), status=plan.status, status_detail=plan.status_detail,
            units=units, decision_ts_ns={k: int(ts_by_day[k][iso]) for k in strategy.instrument_keys},
            directions=directions, ex_ante_annual_vol=vol, gross_exposure=gross,
        ))
    return RebalanceSchedule(strategy_fingerprint=strategy.fingerprint(), span_start=start,
                             span_end_exclusive=end_exclusive, rebalance=strategy.rebalance,
                             decisions=tuple(decisions))
