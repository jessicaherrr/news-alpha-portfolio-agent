"""The single boundary between the Phase 15B orchestration and the economics engine.

Every official Phase 15 economic number is Fill-derived from the C++ Quant Core.
This module defines the typed result the orchestration consumes and the two
engine runners that can produce it:

* :class:`CppEngineRunner` -- the real path (``--run-real`` only): one
  ``quant_backtest_targets_csv`` subprocess per (schedule, cost scenario), with
  the additive, read-only ``--trades-out`` / ``--fills-out`` audit exports.
  Python never reprices; it parses what the engine already decided.
* :class:`SyntheticEngineRunner` -- the deterministic integration stand-in
  (``--synthetic-integration`` only). It fabricates a self-consistent
  trades / fills / daily-equity trace from a target schedule with a seeded RNG,
  so the whole pipeline can be exercised end to end without market data, a C++
  build or a real result. The frozen candidate manifest refuses it for research,
  and :func:`assert_synthetic_runner_forbidden_for_real` enforces that at the
  ``--run-real`` boundary.

Neither runner is the model. Neither invents a position size. The synthetic
runner's numbers are software-test fixtures, never a research result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from alpha_agent.adapters.targets_bridge import run_targets_backtest_cli
from alpha_agent.backtest.targets import TargetSchedule
from alpha_agent.ml.episodes import PrimaryEpisode, extract_primary_episodes
from alpha_agent.ml.errors import MLProtocolError
from alpha_agent.ml.trade_export import (
    ClosedTradeRecord,
    FillRecord,
    read_closed_trades_csv,
    read_fills_csv,
)
from alpha_agent.validation.cost_stress import (
    REFERENCE_COMMISSION_USD,
    REFERENCE_SLIPPAGE_TICKS,
    REFERENCE_SPREAD_TICKS,
)
from alpha_agent.validation.returns import DailyReturnSeries, daily_returns_from_cpp_trace

#: The canonical Phase 13.5C capital base, inherited unchanged.
CAPITAL_BASE_USD = 100_000.0

_NS_PER_DAY = 86_400_000_000_000

#: A synthetic runner is a software fixture. Any research/registry/report path
#: that carries this string in its provenance is refused at the ``--run-real``
#: boundary.
SYNTHETIC_ENGINE_TAG = "synthetic_deterministic_engine__software_fixture_only"
CPP_ENGINE_TAG = "cpp_quant_core__quant_backtest_targets_csv"


class EngineRunResult(BaseModel):
    """One economics-engine run over one schedule at one cost scenario.

    ``daily`` is the official daily return series. ``closed_trades`` / ``fills``
    are the additive audit trails; they are empty when the caller did not ask for
    them (an economics-only run does not need them).
    """

    model_config = {"frozen": True, "extra": "forbid"}

    engine: str
    schedule_hash: str
    cost_scenario_label: str
    cost_multiplier: float = Field(gt=0.0)

    daily: DailyReturnSeries
    n_signals: int = Field(ge=0)
    n_fills: int = Field(ge=0)
    n_trades: int = Field(ge=0)
    gross_pnl_usd: float
    costs_usd: float
    net_pnl_usd: float
    starting_capital_usd: float = Field(gt=0.0)
    commission_per_contract_usd: float = Field(ge=0.0)

    closed_trades: tuple[ClosedTradeRecord, ...] = ()
    fills: tuple[FillRecord, ...] = ()
    audit_trails_present: bool = False

    @property
    def is_synthetic(self) -> bool:
        return self.engine == SYNTHETIC_ENGINE_TAG


class MetaEngineRunner(Protocol):
    """The only economics interface the Phase 15B orchestration knows about."""

    engine_tag: str

    def run(
        self,
        schedule: TargetSchedule,
        *,
        cost_scenario_label: str,
        cost_multiplier: float,
        want_audit_trails: bool,
    ) -> EngineRunResult: ...


class CachingEngineRunner:
    """Memoises engine runs by ``(schedule_hash, cost_label, want_audit_trails)``.

    The PRIMARY schedule of a ``(family, root)`` is identical across every model
    family and regime, so without caching the 60 economic trials would each
    re-run their root's primary -- 180 primary C++ runs instead of the 60 the
    frozen compute estimate budgets. Meta-labeled schedules differ per trial and
    never collide, so caching changes no result, only the run count. It also
    records how many runs were served from cache, for the report.
    """

    def __init__(self, inner: MetaEngineRunner):
        self._inner = inner
        self.engine_tag = inner.engine_tag
        self._cache: dict[tuple[str, str, bool], EngineRunResult] = {}
        self.n_calls = 0
        self.n_engine_runs = 0
        self.n_cache_hits = 0

    def run(
        self,
        schedule: TargetSchedule,
        *,
        cost_scenario_label: str,
        cost_multiplier: float,
        want_audit_trails: bool,
    ) -> EngineRunResult:
        self.n_calls += 1
        key = (schedule.schedule_hash(), cost_scenario_label, want_audit_trails)
        hit = self._cache.get(key)
        if hit is not None:
            self.n_cache_hits += 1
            return hit
        result = self._inner.run(
            schedule,
            cost_scenario_label=cost_scenario_label,
            cost_multiplier=cost_multiplier,
            want_audit_trails=want_audit_trails,
        )
        self._cache[key] = result
        self.n_engine_runs += 1
        return result


def assert_synthetic_runner_forbidden_for_real(runner: MetaEngineRunner) -> None:
    """A ``--run-real`` invocation may never use the synthetic engine."""
    if getattr(runner, "engine_tag", "") == SYNTHETIC_ENGINE_TAG:
        raise MLProtocolError(
            "the synthetic deterministic engine is a software fixture and must never "
            "produce a Phase 15 research result; --run-real requires the C++ Quant Core"
        )


# --------------------------------------------------------------------------
# the real path
# --------------------------------------------------------------------------
class CppEngineRunner:
    """Real path: shells ``quant_backtest_targets_csv`` once per call.

    Reproduces the frozen Phase 13.5C execution wiring exactly:

    * a per-root ``ValidationDayPlan`` (canonical CME trading-day boundaries) is
      always emitted, so ``daily_equity`` is a ``trading_day`` series -- the same
      daily-return basis Phase 13.5C's gating null / DSR / Sharpe are computed on,
      never legacy UTC-day buckets;
    * the observed roll ``effective_ts_ns`` get a mechanical roll-continuation
      target row (the HELD target re-stamped) injected before the schedule
      reaches C++ -- an execution-only overlay, never a strategy decision and
      never seen by the null test / schedule hash / statistics;
    * the roll-close auxiliary same-timestamp marks are passed as positional
      arg 9 (which needs arg 8, the vdays plan, filled).

    Bars, contracts, the vdays plan and the roll timestamps are fixed per root
    and supplied at construction. Only constructed under ``--run-real``.
    """

    engine_tag = CPP_ENGINE_TAG

    def __init__(
        self,
        *,
        executable: str | Path,
        work_dir: str | Path,
        bars: pd.DataFrame,
        contracts: pd.DataFrame,
        root_symbol: str,
        validation_day_plan: object,
        roll_effective_ts_ns: tuple[int, ...] = (),
        roll_close_marks_path: str | Path | None = None,
        capital_base_usd: float = CAPITAL_BASE_USD,
        base_commission_per_contract_usd: float = REFERENCE_COMMISSION_USD,
    ):
        self.executable = Path(executable)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._bars = bars
        self._contracts = contracts
        self._root = root_symbol
        self._day_plan = validation_day_plan
        self._roll_ts = tuple(sorted(int(t) for t in roll_effective_ts_ns))
        self._roll_marks = (
            str(roll_close_marks_path) if roll_close_marks_path is not None else None
        )
        self._capital = float(capital_base_usd)
        self._base_commission = float(base_commission_per_contract_usd)
        self._counter = 0

    def run(
        self,
        schedule: TargetSchedule,
        *,
        cost_scenario_label: str,
        cost_multiplier: float,
        want_audit_trails: bool,
    ) -> EngineRunResult:
        from alpha_agent.validation.phase_13_5c_matrix import _inject_roll_continuation

        self._counter += 1
        wd = self.work_dir / f"run_{self._counter:04d}_{cost_scenario_label}"
        wd.mkdir(parents=True, exist_ok=True)
        trades_out = str(wd / "trades.csv") if want_audit_trails else None
        fills_out = str(wd / "fills.csv") if want_audit_trails else None
        commission = self._base_commission * float(cost_multiplier)

        vdays_path = self._day_plan.write_csv(wd / "validation_days.csv")
        exec_schedule = _inject_roll_continuation(schedule, self._roll_ts)

        cpp = run_targets_backtest_cli(
            self._bars,
            self._contracts,
            exec_schedule,
            executable=self.executable,
            work_dir=wd,
            commission_per_contract_usd=commission,
            slippage_ticks=REFERENCE_SLIPPAGE_TICKS,
            spread_ticks=REFERENCE_SPREAD_TICKS,
            validation_days_path=vdays_path,
            roll_close_marks_path=self._roll_marks,
            trades_out_path=trades_out,
            fills_out_path=fills_out,
        )
        daily = daily_returns_from_cpp_trace(
            cpp, capital_base_usd=self._capital, validation_day_plan=self._day_plan
        )
        trades = read_closed_trades_csv(trades_out) if trades_out else ()
        fills = read_fills_csv(fills_out) if fills_out else ()
        return EngineRunResult(
            engine=CPP_ENGINE_TAG,
            schedule_hash=schedule.schedule_hash(),
            cost_scenario_label=cost_scenario_label,
            cost_multiplier=float(cost_multiplier),
            daily=daily,
            n_signals=int(cpp.get("signals", 0)),
            n_fills=int(cpp.get("fills", 0)),
            n_trades=int(cpp.get("trades", 0)),
            gross_pnl_usd=float(cpp.get("gross_pnl_usd", 0.0)),
            costs_usd=float(cpp.get("costs_usd", 0.0)),
            net_pnl_usd=float(cpp.get("net_pnl_usd", 0.0)),
            starting_capital_usd=float(cpp.get("starting_capital_usd", self._capital)),
            commission_per_contract_usd=commission,
            closed_trades=tuple(trades),
            fills=tuple(fills),
            audit_trails_present=want_audit_trails,
        )


# --------------------------------------------------------------------------
# the synthetic integration path
# --------------------------------------------------------------------------
def _rng_for(schedule_hash: str, salt: str) -> np.random.Generator:
    seed = int.from_bytes(
        __import__("hashlib").sha256(f"{schedule_hash}|{salt}".encode()).digest()[:8],
        "big",
    )
    return np.random.default_rng(seed)


class SyntheticEngineRunner:
    """Deterministic trades / fills / daily-equity from a schedule. TESTS ONLY.

    For each primary episode it emits one opening fill, one closing fill and one
    :class:`ClosedTradeRecord` whose gross PnL is drawn from a seeded RNG keyed on
    the schedule hash. Commission is a flat per-contract charge scaled by the cost
    multiplier; the closing fill's commission alone lands on
    ``ClosedTrade.costs_usd`` exactly as the real ledger does, so the
    episode-attribution and reconciliation paths get the same shape of input they
    would from C++. An unterminated final episode emits an opening fill only.
    """

    engine_tag = SYNTHETIC_ENGINE_TAG

    def __init__(
        self,
        *,
        capital_base_usd: float = CAPITAL_BASE_USD,
        base_commission_per_contract_usd: float = REFERENCE_COMMISSION_USD,
        gross_pnl_scale_usd: float = 900.0,
        gross_pnl_bias_usd: float = 40.0,
        instrument_id: int = 1,
    ):
        self._capital = float(capital_base_usd)
        self._base_commission = float(base_commission_per_contract_usd)
        self._scale = float(gross_pnl_scale_usd)
        self._bias = float(gross_pnl_bias_usd)
        self._instrument_id = int(instrument_id)

    def run(
        self,
        schedule: TargetSchedule,
        *,
        cost_scenario_label: str,
        cost_multiplier: float,
        want_audit_trails: bool,
    ) -> EngineRunResult:
        episodes = extract_primary_episodes(schedule)
        sched_hash = schedule.schedule_hash()
        commission = self._base_commission * float(cost_multiplier)
        rows = list(schedule.rows)
        raw_symbol = f"{schedule.root_symbol}Z9"

        # deterministic per-episode gross PnL keyed on the PRIMARY schedule hash
        # is not available here (this may be a meta-labeled schedule), so key on
        # (root, episode entry ts) which is stable across TAKE/SKIP rewrites.
        trades: list[ClosedTradeRecord] = []
        fills: list[FillRecord] = []
        fill_ix = 0
        trade_ix = 0
        # daily pnl bucketed by the schedule row (one row per synthetic day)
        day_pnl = np.zeros(len(rows), dtype=float)
        ts_by_row = [int(r.ts_event_ns) for r in rows]
        row_of_ts = {t: i for i, t in enumerate(ts_by_row)}

        traded_episodes = [ep for ep in _active_episodes(schedule, episodes)]
        for ep in episodes:
            side_sign = ep.side
            open_side = "buy" if side_sign > 0 else "sell"
            close_side = "sell" if side_sign > 0 else "buy"
            fills.append(
                _fill(fill_ix, ep.entry_decision_ts_ns, open_side, 1, commission,
                      self._instrument_id, raw_symbol)
            )
            fill_ix += 1
            if ep not in traded_episodes or not ep.is_terminated:
                continue
            ep_rng = _rng_for(f"{schedule.root_symbol}|{ep.entry_decision_ts_ns}", "gross")
            gross = float(self._bias + self._scale * ep_rng.standard_normal())
            fills.append(
                _fill(fill_ix, ep.exit_decision_ts_ns, close_side, 1, commission,
                      self._instrument_id, raw_symbol)
            )
            fill_ix += 1
            trades.append(
                ClosedTradeRecord(
                    trade_index=trade_ix,
                    instrument_id=self._instrument_id,
                    raw_symbol=raw_symbol,
                    root_symbol=schedule.root_symbol,
                    ts_open_ns=ep.entry_decision_ts_ns,
                    ts_close_ns=ep.exit_decision_ts_ns,
                    quantity=1,
                    direction=1 if side_sign > 0 else -1,
                    entry_price=100.0,
                    exit_price=100.0 + gross / 50.0,
                    gross_pnl_usd=gross,
                    costs_usd=commission,          # closing fill commission only
                    net_pnl_usd=gross - commission,
                    close_reason="signal",
                )
            )
            trade_ix += 1
            close_row = row_of_ts.get(int(ep.exit_decision_ts_ns))
            if close_row is None:
                close_row = len(rows) - 1
            day_pnl[close_row] += gross - 2.0 * commission   # full round turn on the close day

        # gross/costs totals: engine headline
        total_gross = float(sum(t.gross_pnl_usd for t in trades))
        total_costs = float(commission * len(fills))
        total_net = total_gross - total_costs
        # push the entry commission of every fill into the close day too (the
        # headline already carries it); keep the daily trace reconciling to net.
        # day_pnl currently carries gross - round-turn; sum == total_net when
        # every episode is terminated. Any residual is the open episode's entry.
        daily = self._daily_series(ts_by_row, day_pnl)
        return EngineRunResult(
            engine=SYNTHETIC_ENGINE_TAG,
            schedule_hash=sched_hash,
            cost_scenario_label=cost_scenario_label,
            cost_multiplier=float(cost_multiplier),
            daily=daily,
            n_signals=len(rows),
            n_fills=len(fills),
            n_trades=len(trades),
            gross_pnl_usd=total_gross,
            costs_usd=total_costs,
            net_pnl_usd=total_net,
            starting_capital_usd=self._capital,
            commission_per_contract_usd=commission,
            closed_trades=tuple(trades) if want_audit_trails else (),
            fills=tuple(fills) if want_audit_trails else (),
            audit_trails_present=want_audit_trails,
        )

    def _daily_series(self, ts_by_row: list[int], day_pnl: np.ndarray) -> DailyReturnSeries:
        equity = self._capital + np.cumsum(day_pnl)
        trace = []
        for i, (t, eq) in enumerate(zip(ts_by_row, equity)):
            trace.append([i, int(t), float(eq), float(eq - self._capital), 0.0, 0.0, i + 1, i + 1])
        cpp_json = {
            "daily_equity": trace,
            "starting_capital_usd": self._capital,
            "daily_equity_basis": "utc_day",
        }
        return daily_returns_from_cpp_trace(cpp_json, capital_base_usd=self._capital)


def _fill(
    ix: int, ts: int, side: str, qty: int, commission_per_contract: float,
    instrument_id: int, raw_symbol: str,
) -> FillRecord:
    return FillRecord(
        fill_index=ix,
        fill_id=ix,
        order_id=ix,
        ts_fill_ns=ts,
        instrument_id=instrument_id,
        raw_symbol=raw_symbol,
        side=side,
        quantity=qty,
        fill_price=100.0,
        commission_usd=commission_per_contract * qty,
        slippage_ticks=0.0,
    )


def _active_episodes(
    schedule: TargetSchedule, episodes: tuple[PrimaryEpisode, ...]
) -> list[PrimaryEpisode]:
    """Episodes the schedule actually holds a non-zero target through.

    After a SKIP rewrite an episode's rows are all neutral, so it produces no
    trade -- which is exactly how a meta-labeled skip reduces trade count.
    """
    active: list[PrimaryEpisode] = []
    for ep in episodes:
        stop = ep.exit_row_index if ep.exit_row_index is not None else len(schedule.rows)
        if any(schedule.rows[i].target_units != 0 for i in range(ep.entry_row_index, stop)):
            active.append(ep)
    return active
