"""News Alpha Phase H -- EXECUTION of a portfolio's rebalance schedule through
the unchanged C++ engine.

    RebalanceSchedule (integer units per root at each decision bar)
      + real engine inputs for every instrument of the strategy
          futures  native 1-minute RAW-contract bars (the ones fills resolve
                   to), the contract definitions, the roll-overlap close marks
          ETF      primary-listing daily RAW bars, the one-share synthetic
                   contract (etf_contract_spec), sourced splits
      -> quant_portfolio_plan_replay_csv (BacktestEngine + PortfolioRiskManager,
         end-of-test force-liquidation, trading-day equity trace)
      -> validation.runner.BacktestRun   (the SAME daily-return aggregation
                                          the Validation Engine uses)

Python writes files and reads the engine's JSON; every fill, cost, roll,
position, equity point and risk decision is the engine's. Back-adjusted
prices never reach this module -- fills resolve to the raw contract traded.

COSTS (Phase H acceptance patch). Every root of a portfolio carries its OWN
commission, charged by the C++ engine
(`ExecutionConfig::commission_per_contract_usd_by_root`, keyed by the
contract's root: every raw contract of a futures root, across rolls, shares
one entry; an ETF is its `E<ticker>` root and a unit is one share). The replay
passes a complete schedule and the CLI refuses an incomplete one, so no root
is ever charged a rate meant for another unit. Rates come from a per-domain
convention with its provenance (`CommissionConvention`):

    FUTURES  $2.00 per contract   the frozen Phase 13 reference assumption
    ETF      $0.005 per share     a declared Phase H research assumption

Both are DECLARED research assumptions, not sourced broker schedules, and are
stressed x1.5 and x2.0 by the frozen cost-stress scenarios; slippage and
spread stay at the frozen 0 ticks. A root with no convention refuses the
portfolio (`COST_MODEL_INCOMPLETE`) -- there is no fallback rate.

A book whose roots follow different trading-day conventions (a CME session
and a US-equity day) is refused (`EXECUTION_UNSUPPORTED`): one daily return
across two day definitions is a validation decision this module does not make.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from alpha_agent.adapters.cpp_cli import write_boundary_bundle
from alpha_agent.data.real_market_dataset import (
    HOLDOUT_START,
    HOLDOUT_START_NS,
    RESEARCH_WINDOW,
    _iso_ns,
    assert_no_holdout_ts,
)
from alpha_agent.news_alpha.mandate import MandateDomain
from alpha_agent.paper.risk_policy import PaperRiskPolicy
from alpha_agent.portfolio.handoff import ReplayResult, hard_gate_policy, run_replay_cli
from alpha_agent.portfolio.strategy import PortfolioStrategySpec, RebalanceSchedule, ScheduleRow
from alpha_agent.validation.cost_stress import (
    CommissionBasis,
    CostScenario,
    CostStressPlan,
    RootCommission,
)
from alpha_agent.validation.dataset import frame_content_hash
from alpha_agent.validation.runner import BacktestRun, backtest_run_from_cpp_json
from alpha_agent.validation.trading_day import (
    TradingDayConvention,
    ValidationDay,
    ValidationDayPlan,
    build_validation_day_plan,
)

__all__ = [
    "DEFAULT_COMMISSION_CONVENTIONS",
    "ETF_COMMISSION_PER_SHARE_USD",
    "CommissionConvention",
    "ExecutionInputs",
    "ExecutionRefusal",
    "ExecutionUnsupported",
    "PortfolioRun",
    "ScheduleBarMismatch",
    "commission_schedule_for",
    "cost_plan_for",
    "etf_engine_instrument_id",
    "execution_support",
    "load_execution_inputs",
    "merged_day_plan",
    "replay_risk_policy",
    "run_schedule",
]

#: A declared research assumption, not evidence: a per-share commission of
#: the order a retail/prop broker charges on US-listed ETFs. The Phase 6 ETF
#: pilot charged the futures $2 per unit on single-share targets; on a
#: portfolio sized in thousands of shares that figure is not a cost model.
ETF_COMMISSION_PER_SHARE_USD = 0.005
_SCENARIOS = (
    CostScenario(label="baseline_1_0x", multiplier=1.0),
    CostScenario(label="stress_1_5x", multiplier=1.5),
    CostScenario(label="stress_2_0x", multiplier=2.0),
)


class CommissionConvention(BaseModel):
    """How one domain's roots are charged, and where the rate comes from."""

    model_config = {"frozen": True, "extra": "forbid"}

    commission_per_unit_usd: float
    unit: Literal["contract", "share"]
    basis: CommissionBasis
    source: str


DEFAULT_COMMISSION_CONVENTIONS: dict[MandateDomain, CommissionConvention] = {
    MandateDomain.FUTURES: CommissionConvention(
        commission_per_unit_usd=2.0, unit="contract", basis=CommissionBasis.DECLARED_RESEARCH_ASSUMPTION,
        source="the frozen Phase 13 reference cost plan ($2.00 per contract) -- a research assumption, not a "
               "sourced fee schedule"),
    MandateDomain.ETF: CommissionConvention(
        commission_per_unit_usd=ETF_COMMISSION_PER_SHARE_USD, unit="share",
        basis=CommissionBasis.DECLARED_RESEARCH_ASSUMPTION,
        source="declared Phase H research assumption ($0.005 per share) -- not observed broker truth"),
}
EVALUATION_START_NS = _iso_ns(RESEARCH_WINDOW[0])
#: ETF engine instrument ids: one per ticker, deterministic, far above any
#: GLBX.MDP3 futures id (collisions are still checked, and fail loudly).
_ETF_ENGINE_ID_BASE = 4_000_000_000


def etf_engine_instrument_id(symbol: str) -> int:
    """One stable engine id per ETF ticker. The primary-listing vendor id is
    not an identity: XNAS.ITCH reassigns it (a locate code) from day to day,
    so a Nasdaq-listed ETF's bars carry hundreds of ids over 2018-2024."""
    from alpha_agent.etf.universe import PILOT_UNIVERSE

    if symbol not in PILOT_UNIVERSE:
        raise ExecutionUnsupported(f"{symbol} is not in the ETF pilot universe")
    return _ETF_ENGINE_ID_BASE + sorted(PILOT_UNIVERSE).index(symbol)


class ExecutionUnsupported(RuntimeError):
    """The engine cannot execute this portfolio faithfully (typed reason in the message)."""


class ScheduleBarMismatch(RuntimeError):
    """A target row names a timestamp at which its root has no engine bar -- a pipeline defect."""


class ExecutionRefusal(BaseModel):
    """Why the engine cannot execute and cost a portfolio faithfully."""

    model_config = {"frozen": True, "extra": "forbid"}

    kind: Literal["COST_MODEL_INCOMPLETE", "EXECUTION_UNSUPPORTED"]
    detail: str
    unpriced_roots: tuple[str, ...] = ()


_Conventions = dict[MandateDomain, CommissionConvention]


def _day_convention(domain: MandateDomain, symbol: str) -> str:
    if domain is MandateDomain.ETF:
        from alpha_agent.etf.calendar import etf_calendar

        return str(sorted(etf_calendar().trading_day_convention(symbol).items()))
    from alpha_agent.data.calendars import default_calendar

    return str(sorted(default_calendar().trading_day_convention(symbol).items()))


def execution_support(strategy: PortfolioStrategySpec,
                      conventions: _Conventions | None = None) -> ExecutionRefusal | None:
    """``None`` when the engine can execute AND cost every root of the
    strategy faithfully; otherwise the typed refusal. Reads no market data."""
    conventions = DEFAULT_COMMISSION_CONVENTIONS if conventions is None else conventions
    unpriced = tuple(sorted({m.execution_root for m in strategy.members if m.domain not in conventions}))
    if unpriced:
        return ExecutionRefusal(
            kind="COST_MODEL_INCOMPLETE", unpriced_roots=unpriced,
            detail=(f"no declared commission for {', '.join(unpriced)} -- every traded root needs an explicit rate "
                    "(a contract rate is never charged per share, nor a share rate per contract)"))
    unsupported = [d for d in strategy.domains if d not in (MandateDomain.FUTURES, MandateDomain.ETF)]
    if unsupported:
        return ExecutionRefusal(kind="EXECUTION_UNSUPPORTED",
                                detail=f"no engine inputs for {', '.join(d.value for d in unsupported)}")
    day_conventions = {_day_convention(m.domain, m.instrument) for m in strategy.members}
    if len(day_conventions) > 1:
        return ExecutionRefusal(
            kind="EXECUTION_UNSUPPORTED",
            detail=("its roots follow different trading-day conventions (a CME session and a US-equity day); one "
                    "daily validation return across two day definitions is a validation decision not made here"))
    return None


def commission_schedule_for(strategy: PortfolioStrategySpec,
                            conventions: _Conventions | None = None) -> tuple[RootCommission, ...]:
    """One explicit entry per execution root of the strategy."""
    conventions = DEFAULT_COMMISSION_CONVENTIONS if conventions is None else conventions
    roots = {m.execution_root: m.domain for m in strategy.members}
    missing = sorted(r for r, d in roots.items() if d not in conventions)
    if missing:
        raise ExecutionUnsupported(f"COST_MODEL_INCOMPLETE: no declared commission for {', '.join(missing)}")
    return tuple(RootCommission(root_symbol=root, commission_per_unit_usd=conventions[d].commission_per_unit_usd,
                                unit=conventions[d].unit, basis=conventions[d].basis, source=conventions[d].source)
                 for root, d in sorted(roots.items()))


def cost_plan_for(strategy: PortfolioStrategySpec, conventions: _Conventions | None = None) -> CostStressPlan:
    """The strategy's cost plan: its explicit per-root commission schedule,
    stressed x1.0 / x1.5 / x2.0. The scalar default is 0 and never applies --
    every traded root is in the schedule, and the replay refuses one that is not."""
    refusal = execution_support(strategy, conventions)
    if refusal is not None:
        raise ExecutionUnsupported(f"{refusal.kind}: {refusal.detail}")
    return CostStressPlan(base_commission_per_contract_usd=0.0, base_slippage_ticks=0.0, base_spread_ticks=0.0,
                          scenarios=_SCENARIOS, commission_schedule=commission_schedule_for(strategy, conventions))


# ---------------------------------------------------------------------------
# engine inputs
# ---------------------------------------------------------------------------

_BAR_COLUMNS = ["ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class ExecutionInputs:
    """Every engine input of one strategy over one window (frozen boundary
    schemas). ``calendars`` maps each execution root to the session calendar
    its trading days are defined by and the symbol that calendar knows it by
    (the futures root; the ETF ticker, not its synthetic ``E<ticker>`` root)."""

    bars: pd.DataFrame
    contracts: pd.DataFrame
    roll_close_marks: pd.DataFrame
    splits: pd.DataFrame
    #: execution root -> the instrument ids its bars carry
    instrument_ids: dict[str, tuple[int, ...]]
    calendars: dict[str, tuple[object, str]]
    provenance: tuple[str, ...]

    def window(self, start_ns: int, end_ns: int) -> ExecutionInputs:
        assert_no_holdout_ts([end_ns - 1])
        b = self.bars
        bars = b[(b["ts_event_ns"] >= start_ns) & (b["ts_event_ns"] < end_ns)].reset_index(drop=True)
        m = self.roll_close_marks
        marks = m[(m["ts_event_ns"] >= start_ns) & (m["ts_event_ns"] < end_ns)].reset_index(drop=True)
        s = self.splits
        splits = s[(s["effective_ts_ns"] >= start_ns) & (s["effective_ts_ns"] < end_ns)].reset_index(drop=True)
        return replace(self, bars=bars, roll_close_marks=marks, splits=splits)

    def fingerprint(self) -> str:
        parts = [frame_content_hash(f.reset_index(drop=True))
                 for f in (self.bars, self.contracts, self.roll_close_marks, self.splits)]
        return "execinputs1:" + hashlib.sha256("|".join(parts).encode()).hexdigest()


ExecutionLoader = Callable[[PortfolioStrategySpec], ExecutionInputs]


@lru_cache(maxsize=8)
def _reconstituted(root: str):
    from alpha_agent.data.real_market_dataset import reconstitute_root

    return reconstitute_root(root)


def _futures_inputs(root: str, lo: int, hi: int):
    from alpha_agent.data.calendars import default_calendar
    from alpha_agent.data.real_market_dataset import execution_bars, roll_close_marks

    recon = _reconstituted(root)
    bars = execution_bars(recon, lo, hi)
    ids = sorted(int(i) for i in bars["instrument_id"].unique())
    contracts = recon.contracts[recon.contracts["instrument_id"].astype("int64").isin(ids)]
    marks = roll_close_marks(recon, lo, hi)
    note = (f"{root}: GLBX.MDP3 ohlcv-1m raw-contract bars ({len(ids)} contracts) rebuilt offline from the raw store; "
            f"{len(marks)} roll close marks")
    return bars, contracts, marks, default_calendar(), note


def _etf_inputs(symbol: str, lo: int, hi: int):
    from alpha_agent.etf.calendar import etf_calendar
    from alpha_agent.etf.corporate_actions import known_splits
    from alpha_agent.etf.data_source import etf_contract_spec, load_primary_listing_bars

    frame, source = load_primary_listing_bars(symbol)
    bars = frame.rename(columns={"ts_event": "ts_event_ns"})
    bars = bars[(bars["ts_event_ns"] >= lo) & (bars["ts_event_ns"] < hi)].sort_values("ts_event_ns")
    if bars["ts_event_ns"].duplicated().any():
        raise ExecutionUnsupported(f"{symbol}: more than one primary-listing bar at one timestamp")
    vendor_ids = int(bars["instrument_id"].nunique())
    instrument_id = etf_engine_instrument_id(symbol)
    bars = bars.assign(instrument_id=instrument_id)[_BAR_COLUMNS].reset_index(drop=True)
    spec = etf_contract_spec(instrument_id, symbol, activation_ns=int(bars["ts_event_ns"].iloc[0]))
    contracts = pd.DataFrame([{
        "instrument_id": spec.instrument_id, "raw_symbol": spec.raw_symbol, "root_symbol": spec.root_symbol,
        "exchange": spec.exchange, "tick_size": spec.tick_size, "multiplier": spec.multiplier,
        "activation_ns": spec.activation_ns, "expiration_ns": spec.expiration_ns, "first_notice_ns": "",
        "last_trade_ns": "",
    }])
    days = pd.to_datetime(bars["ts_event_ns"], unit="ns", utc=True).dt.date
    splits = []
    for split in known_splits(symbol):
        on_or_after = bars[days >= split.effective_date]
        if len(on_or_after):
            splits.append({"instrument_id": instrument_id, "effective_ts_ns": int(on_or_after["ts_event_ns"].iloc[0]),
                           "ratio": float(split.ratio)})
    remap = (f"{vendor_ids} vendor instrument ids in the window -- {source.dataset} reassigns them, the ticker is "
             "the identity" if vendor_ids > 1 else "its one vendor instrument id")
    note = (f"{symbol}: {source.dataset} ohlcv-1d primary-listing RAW bars as engine instrument {instrument_id} "
            f"({remap}), one-share contract, {len(splits)} sourced split(s) applied by the engine; price return only "
            "(no distribution history)")
    return bars, contracts, pd.DataFrame(splits, columns=["instrument_id", "effective_ts_ns", "ratio"]), \
        etf_calendar(), note


def load_execution_inputs(strategy: PortfolioStrategySpec, *, end_ns: int = HOLDOUT_START_NS) -> ExecutionInputs:
    """Real, offline engine inputs for every instrument of ``strategy`` from
    2018-01-01 up to ``end_ns`` (default: the start of the 2025 holdout, never
    into it). A discovery-only caller passes the discovery end, so no
    validation-window bar is ever materialized."""
    if end_ns > HOLDOUT_START_NS:
        raise ValueError("engine inputs never reach the 2025 holdout")
    lo, hi = EVALUATION_START_NS, end_ns
    bars, contracts, marks, splits, ids, calendars, notes = [], [], [], [], {}, {}, []
    for key in strategy.instrument_keys:
        member = strategy.instrument(key)
        root = member.execution_root
        if member.domain is MandateDomain.FUTURES:
            b, c, m, cal, note = _futures_inputs(member.instrument, lo, hi)
            marks.append(m)
        elif member.domain is MandateDomain.ETF:
            b, c, s, cal, note = _etf_inputs(member.instrument, lo, hi)
            splits.append(s)
        else:
            raise ExecutionUnsupported(f"no engine inputs for {member.domain.value}")
        bars.append(b)
        contracts.append(c)
        ids[root] = tuple(sorted(int(i) for i in b["instrument_id"].unique()))
        calendars[root] = (cal, member.instrument)
        notes.append(note)
    all_ids = [i for group in ids.values() for i in group]
    if len(all_ids) != len(set(all_ids)):
        raise ExecutionUnsupported("two instruments share an instrument id across datasets -- refusing to merge them")
    frame = pd.concat(bars, ignore_index=True).sort_values(["ts_event_ns", "instrument_id"], kind="stable")
    assert_no_holdout_ts(frame["ts_event_ns"])
    return ExecutionInputs(
        bars=frame.reset_index(drop=True), contracts=pd.concat(contracts, ignore_index=True),
        roll_close_marks=(pd.concat(marks, ignore_index=True) if marks
                          else pd.DataFrame(columns=["instrument_id", "ts_event_ns", "close"])),
        splits=(pd.concat(splits, ignore_index=True) if splits
                else pd.DataFrame(columns=["instrument_id", "effective_ts_ns", "ratio"])),
        instrument_ids=ids, calendars=calendars, provenance=tuple(notes),
    )


def merged_day_plan(inputs: ExecutionInputs) -> ValidationDayPlan:
    """One canonical trading day per session day across the portfolio's
    roots: its boundary is the latest root's last bar that day. Every root
    must share one trading-day convention -- two conventions cannot define
    one daily return."""
    per_root = []
    conventions = set()
    for root, ids in sorted(inputs.instrument_ids.items()):
        b = inputs.bars[inputs.bars["instrument_id"].isin(ids)]
        calendar, calendar_symbol = inputs.calendars[root]
        plan = build_validation_day_plan(b, root_symbol=calendar_symbol, calendar=calendar)
        conventions.add(plan.convention.model_dump_json())
        per_root.append(plan)
    if len(conventions) != 1:
        raise ExecutionUnsupported("the portfolio's roots follow different trading-day conventions")
    days: dict[str, list[ValidationDay]] = {}
    for plan in per_root:
        for d in plan.days:
            days.setdefault(d.trading_day, []).append(d)
    merged = tuple(ValidationDay(trading_day=label, first_ts_ns=min(d.first_ts_ns for d in ds),
                                 boundary_ts_ns=max(d.boundary_ts_ns for d in ds), n_bars=sum(d.n_bars for d in ds))
                   for label, ds in sorted(days.items()))
    for a, b in pairwise(merged):
        if b.boundary_ts_ns <= a.boundary_ts_ns:
            raise ExecutionUnsupported("merged trading-day boundaries are not strictly ascending")
    if merged and merged[-1].trading_day >= HOLDOUT_START:
        raise ExecutionUnsupported(f"a trading day labelled {merged[-1].trading_day} reaches the locked holdout")
    convention = TradingDayConvention.model_validate_json(conventions.pop())
    return ValidationDayPlan(root_symbol="+".join(sorted(inputs.instrument_ids)), convention=convention, days=merged)


# ---------------------------------------------------------------------------
# one replay
# ---------------------------------------------------------------------------


def replay_risk_policy(strategy: PortfolioStrategySpec, rows: Sequence[ScheduleRow]) -> PaperRiskPolicy:
    """The engine's hard gate for a rebalance schedule -- the SAME construction
    as a single plan's handoff (`handoff.hard_gate_policy`)."""
    return hard_gate_policy(strategy.constraints, rows)


class PortfolioRun(BaseModel):
    """One C++ replay of one schedule over one span at one cost setting."""

    model_config = {"frozen": True, "extra": "forbid"}

    label: str
    span_start_ns: int
    span_end_ns: int
    schedule_hash: str
    #: root -> USD per unit, as the C++ replay echoed it back
    commission_schedule: dict[str, float]
    slippage_ticks: float
    spread_ticks: float
    run: BacktestRun
    risk_rejects: int
    risk_resizes: int
    rolls: int
    rolls_deferred: int
    target_rows: int
    target_rows_applied: int
    day_plan_identity: str
    inputs_fingerprint: str


def _schedule_rows(schedule: RebalanceSchedule, strategy: PortfolioStrategySpec, start_ns: int,
                   end_ns: int) -> list[ScheduleRow]:
    return [r for r in schedule.rows(strategy.execution_roots) if start_ns <= r.ts_event_ns < end_ns]


def _check_rows_on_bars(rows: Sequence[ScheduleRow], inputs: ExecutionInputs) -> None:
    b = inputs.bars
    for root, ids in inputs.instrument_ids.items():
        have = set(b.loc[b["instrument_id"].isin(ids), "ts_event_ns"].astype("int64"))
        for r in rows:
            if r.root_symbol == root and r.ts_event_ns not in have:
                raise ScheduleBarMismatch(f"{root}: no engine bar at the decision timestamp {r.ts_event_ns}")


def run_schedule(
    schedule: RebalanceSchedule,
    strategy: PortfolioStrategySpec,
    inputs: ExecutionInputs,
    *,
    start_ns: int,
    end_ns: int,
    commission_schedule: Mapping[str, float],
    workdir: Path,
    label: str,
    slippage_ticks: float = 0.0,
    spread_ticks: float = 0.0,
    cli: Path | None = None,
) -> PortfolioRun:
    """Replay the schedule's rows in ``[start_ns, end_ns)`` over the same
    window's bars; the engine force-liquidates at the end so the last equity
    point carries complete realized PnL. ``commission_schedule`` (root ->
    USD per unit) must price every traded root: the C++ replay charges it and
    refuses an incomplete one."""
    if schedule.strategy_fingerprint != strategy.fingerprint():
        raise ValueError("the schedule belongs to another strategy")
    window = inputs.window(start_ns, end_ns)
    rows = _schedule_rows(schedule, strategy, start_ns, end_ns)
    if not rows:
        raise ValueError(f"{label}: no rebalance falls inside the span")
    _check_rows_on_bars(rows, window)
    day_plan = merged_day_plan(window)
    fp = strategy.fingerprint()
    run_dir = Path(workdir) / label
    bars_csv, contracts_csv = write_boundary_bundle(window.bars, window.contracts, run_dir)
    targets_csv = run_dir / "plan_targets.csv"
    targets_csv.write_text("ts_event_ns,root_symbol,target_units,portfolio_plan_fingerprint\n" + "".join(
        f"{r.ts_event_ns},{r.root_symbol},{r.target_units},{fp}\n" for r in rows), encoding="utf-8")
    risk_csv = replay_risk_policy(strategy, rows).write_csv(run_dir / "risk_config.csv")
    days_csv = day_plan.write_csv(run_dir / "validation_days.csv")
    marks_csv = splits_csv = None
    if len(window.roll_close_marks):
        marks_csv = run_dir / "roll_close_marks.csv"
        window.roll_close_marks.to_csv(marks_csv, index=False)
    if len(window.splits):
        splits_csv = run_dir / "splits.csv"
        window.splits.to_csv(splits_csv, index=False)
    replay, raw = run_replay_cli(
        targets_csv, risk_csv, bars_csv, contracts_csv, cli=cli, commission_per_contract_usd=0.0,
        commission_schedule=commission_schedule, slippage_ticks=slippage_ticks, spread_ticks=spread_ticks, validation_days_csv=days_csv,
        roll_close_marks_csv=marks_csv, splits_csv=splits_csv, force_liquidate_at_end=True,
        timeout_s=900.0,
    )
    run = backtest_run_from_cpp_json(raw, capital_base_usd=strategy.constraints.capital_usd,
                                     validation_day_plan=day_plan)
    return _portfolio_run(label, start_ns, end_ns, schedule, slippage_ticks, spread_ticks, run, replay, day_plan,
                          window)


def _portfolio_run(label, start_ns, end_ns, schedule, slippage, spread, run: BacktestRun,
                   replay: ReplayResult, day_plan: ValidationDayPlan, window: ExecutionInputs) -> PortfolioRun:
    return PortfolioRun(
        label=label, span_start_ns=start_ns, span_end_ns=end_ns, schedule_hash=schedule.schedule_hash(),
        commission_schedule={c.root_symbol: c.commission_per_unit_usd for c in replay.commission_schedule},
        slippage_ticks=slippage, spread_ticks=spread, run=run,
        risk_rejects=replay.risk_rejects, risk_resizes=replay.risk_resizes, rolls=replay.rolls,
        rolls_deferred=replay.rolls_deferred, target_rows=replay.target_rows,
        target_rows_applied=replay.target_rows_applied, day_plan_identity=day_plan.identity(),
        inputs_fingerprint=window.fingerprint(),
    )
