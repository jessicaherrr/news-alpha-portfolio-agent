"""Backtest-runner boundary for the validation framework.

Every economic number a validation run consumes comes from ONE C++ backtest per
(schedule, cost scenario). The framework never reconstructs fills, positions,
commissions, execution prices or equity -- it only aggregates the official
``daily_equity`` trace into a :class:`DailyReturnSeries` (see
:mod:`.returns`) and reads the verbatim count / PnL fields.

* :class:`CliBacktestRunner` -- the real path: shells the frozen
  ``quant_backtest_targets_csv`` CLI.
* :func:`backtest_run_from_cpp_json` -- wrap one CLI JSON result.
* :func:`synthetic_backtest_run` -- build a run straight from a synthetic daily
  equity trace, for the section-23 statistical fixtures ONLY (software tests, not
  market results).
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from alpha_agent.adapters.targets_bridge import run_targets_backtest_cli
from alpha_agent.backtest.targets import TargetSchedule
from alpha_agent.validation.cost_stress import (
    REFERENCE_COMMISSION_USD,
    REFERENCE_SLIPPAGE_TICKS,
    REFERENCE_SPREAD_TICKS,
)
from alpha_agent.validation.metrics import annualized_sharpe, daily_sharpe
from alpha_agent.validation.returns import DailyReturnSeries, daily_returns_from_cpp_trace
from alpha_agent.validation.trading_day import ValidationDayPlan


class BacktestRun(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    daily: DailyReturnSeries
    n_fills: int = Field(ge=0)
    n_trades: int = Field(ge=0)
    n_signals: int = Field(ge=0)
    gross_pnl_usd: float
    costs_usd: float
    net_pnl_usd: float
    commission_per_contract_usd: float
    slippage_ticks: float
    spread_ticks: float
    official_source: str = "cpp_fill_events"

    @property
    def daily_sharpe(self) -> float:
        return daily_sharpe(self.daily.returns_array())

    @property
    def annualized_sharpe(self) -> float:
        return annualized_sharpe(self.daily.returns_array())


def backtest_run_from_cpp_json(
    cpp_json: dict,
    *,
    capital_base_usd: float,
    validation_day_plan: ValidationDayPlan | None = None,
    initial_equity_usd: float | None = None,
) -> BacktestRun:
    daily = daily_returns_from_cpp_trace(
        cpp_json,
        capital_base_usd=capital_base_usd,
        validation_day_plan=validation_day_plan,
        initial_equity_usd=initial_equity_usd,
    )
    return BacktestRun(
        daily=daily,
        n_fills=int(cpp_json.get("fills", 0)),
        n_trades=int(cpp_json.get("trades", 0)),
        n_signals=int(cpp_json.get("signals", 0)),
        gross_pnl_usd=float(cpp_json.get("gross_pnl_usd", 0.0)),
        costs_usd=float(cpp_json.get("costs_usd", 0.0)),
        net_pnl_usd=float(cpp_json.get("net_pnl_usd", 0.0)),
        commission_per_contract_usd=float(
            cpp_json.get("commission_per_contract_usd", REFERENCE_COMMISSION_USD)
        ),
        slippage_ticks=float(cpp_json.get("slippage_ticks", REFERENCE_SLIPPAGE_TICKS)),
        spread_ticks=float(cpp_json.get("spread_ticks", REFERENCE_SPREAD_TICKS)),
    )


class BacktestRunner(Protocol):
    def run(
        self,
        *,
        schedule: TargetSchedule,
        bars: pd.DataFrame,
        contracts: pd.DataFrame,
        commission_per_contract_usd: float,
        slippage_ticks: float,
        spread_ticks: float,
        capital_base_usd: float,
        validation_day_plan: ValidationDayPlan | None,
        initial_equity_usd: float | None,
    ) -> BacktestRun: ...


class CliBacktestRunner:
    """Real path -- one ``quant_backtest_targets_csv`` subprocess per call."""

    def __init__(
        self,
        executable: str | Path,
        work_dir: str | Path,
        *,
        roll_close_marks_path: str | Path | None = None,
    ):
        self.executable = Path(executable)
        self.work_dir = Path(work_dir)
        # Phase 13.5C: auxiliary same-timestamp outgoing-contract closes for the
        # C++ roll close-leg (never a MarketEvent). Per-root constant, so it lives
        # on the runner rather than each run() call. None => frozen RejectDefer.
        self.roll_close_marks_path = (
            str(roll_close_marks_path) if roll_close_marks_path is not None else None
        )
        self._counter = 0

    def run(
        self,
        *,
        schedule: TargetSchedule,
        bars: pd.DataFrame,
        contracts: pd.DataFrame,
        commission_per_contract_usd: float = REFERENCE_COMMISSION_USD,
        slippage_ticks: float = REFERENCE_SLIPPAGE_TICKS,
        spread_ticks: float = REFERENCE_SPREAD_TICKS,
        capital_base_usd: float,
        validation_day_plan: ValidationDayPlan | None = None,
        initial_equity_usd: float | None = None,
        trades_out_path: str | Path | None = None,
        fills_out_path: str | Path | None = None,
    ) -> BacktestRun:
        """`trades_out_path` / `fills_out_path` (Alpha Discovery campaign
        Part I, reusing the Phase 15A additive CLI export
        `alpha_agent.adapters.targets_bridge.run_targets_backtest_cli` already
        supports): request a real, persisted closed-trade / fill audit export
        for THIS run. `None` (the default -- every existing caller) changes
        nothing: no flag reaches the CLI, no new file is written, and the
        returned `BacktestRun` is byte-identical to before this parameter
        existed."""
        self._counter += 1
        wd = self.work_dir / f"run_{self._counter:04d}"
        wd.mkdir(parents=True, exist_ok=True)
        vdays_path = None
        if validation_day_plan is not None and validation_day_plan.n_days:
            vdays_path = validation_day_plan.write_csv(wd / "validation_days.csv")
        marks_path = self.roll_close_marks_path
        if marks_path is not None and vdays_path is None:
            # the CLI needs arg 8 (validation_days) filled to disambiguate arg 9;
            # emit the plan even when the caller did not force the trading-day basis
            if validation_day_plan is not None:
                vdays_path = validation_day_plan.write_csv(wd / "validation_days.csv")
            else:
                marks_path = None  # cannot position arg 9 without arg 8
        cpp = run_targets_backtest_cli(
            bars,
            contracts,
            schedule,
            executable=self.executable,
            work_dir=wd,
            commission_per_contract_usd=commission_per_contract_usd,
            slippage_ticks=slippage_ticks,
            spread_ticks=spread_ticks,
            validation_days_path=vdays_path,
            roll_close_marks_path=marks_path,
            trades_out_path=trades_out_path,
            fills_out_path=fills_out_path,
        )
        return backtest_run_from_cpp_json(
            cpp,
            capital_base_usd=capital_base_usd,
            validation_day_plan=validation_day_plan if vdays_path is not None else None,
            initial_equity_usd=initial_equity_usd,
        )


def synthetic_daily_equity_trace(
    daily_pnl_usd: np.ndarray,
    *,
    starting_capital_usd: float = 100_000.0,
    fills_per_day: int = 1,
    first_day_index: int = 0,
    ns_per_day: int = 86_400_000_000_000,
) -> list[list]:
    """Build a C++-shaped ``daily_equity`` array from a synthetic per-day PnL
    vector. FOR SECTION-23 STATISTICAL FIXTURES ONLY -- these are software tests,
    not market results."""
    pnl = np.asarray(daily_pnl_usd, dtype=float)
    equity = starting_capital_usd + np.cumsum(pnl)
    rows: list[list] = []
    for i, (p, eq) in enumerate(zip(pnl, equity)):
        d = first_day_index + i
        rows.append(
            [d, int(d * ns_per_day + 1), float(eq), float(eq - starting_capital_usd),
             0.0, 0.0, int((i + 1) * fills_per_day), int(i + 1)]
        )
    return rows


def synthetic_backtest_run(
    daily_pnl_usd: np.ndarray,
    *,
    starting_capital_usd: float = 100_000.0,
    capital_base_usd: float | None = None,
    n_fills: int | None = None,
    n_trades: int | None = None,
    costs_usd: float = 0.0,
    commission_per_contract_usd: float = REFERENCE_COMMISSION_USD,
    slippage_ticks: float = REFERENCE_SLIPPAGE_TICKS,
    spread_ticks: float = REFERENCE_SPREAD_TICKS,
    first_day_index: int = 0,
) -> BacktestRun:
    """Construct a :class:`BacktestRun` from a synthetic daily PnL vector."""
    pnl = np.asarray(daily_pnl_usd, dtype=float)
    trace = synthetic_daily_equity_trace(
        pnl, starting_capital_usd=starting_capital_usd, first_day_index=first_day_index
    )
    cpp_json = {"daily_equity": trace, "starting_capital_usd": starting_capital_usd}
    daily = daily_returns_from_cpp_trace(
        cpp_json, capital_base_usd=capital_base_usd or starting_capital_usd
    )
    gross = float(pnl.sum()) + costs_usd
    return BacktestRun(
        daily=daily,
        n_fills=int(n_fills if n_fills is not None else max(1, pnl.size)),
        n_trades=int(n_trades if n_trades is not None else max(1, pnl.size // 2)),
        n_signals=int(pnl.size),
        gross_pnl_usd=gross,
        costs_usd=float(costs_usd),
        net_pnl_usd=float(pnl.sum()),
        commission_per_contract_usd=commission_per_contract_usd,
        slippage_ticks=slippage_ticks,
        spread_ticks=spread_ticks,
    )
