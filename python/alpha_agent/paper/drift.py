"""Phase 21 / 21.1 -- live/paper vs backtest drift diagnostics.

DIAGNOSTIC ONLY. This module never computes an official PnL, fill, Sharpe, or
pass/fail verdict of its own -- it reuses the SAME frozen helper the Phase 13
validation framework uses (:func:`alpha_agent.validation.runner.
backtest_run_from_cpp_json`) to turn one paper-trading step's official C++
``daily_equity`` trace into a :class:`~alpha_agent.validation.runner.
BacktestRun`, then compares its descriptive statistics to the experiment's own
committed :class:`~alpha_agent.registry.models.ResultRecord`. It defines no new
statistical test, threshold, or multiple-testing family -- CLAUDE.md's
BH-FDR / DSR / null methodology stays entirely inside the Phase 13 validation
engine. A :class:`DriftReport`'s ``flags`` are advisory labels for a human /
the UI, never an authority that halts trading (the deterministic C++ hard-risk
gate is the only thing that does that -- see
:mod:`alpha_agent.paper.alerts`).

Prompt 21 requires monitoring fill rate, slippage, trade frequency, and PnL
distribution, each compared to the committed backtest where a comparable
artifact actually exists (Phase 21.1):

* **realized slippage** (:class:`RealizedSlippageEvidence`) is computed from
  the official C++ Fill audit trail ONLY (``PaperFillRow.slippage_ticks`` per
  fill, from ``--fills-out=``) times the contract's own tick size / multiplier
  -- never the single scalar ``slippage_ticks`` cost ASSUMPTION a run was
  configured with, which is an input, not evidence of what happened.
* **trade frequency** (:class:`TradeFrequencyEvidence`) is the closed-trade
  count normalized by the run's own ``ValidationDayPlan.n_days`` (elapsed
  trading days in the replayed window) -- never a raw, unnormalized count.
* **PnL distribution** (:class:`PnlDistributionEvidence`) is descriptive
  statistics (mean/std/min/max/sign counts) over the official per-day PnL
  series the C++ ``daily_equity`` trace already produced -- no new series is
  computed.
* the committed :class:`~alpha_agent.registry.models.ResultRecord` carries a
  headline ``daily_sharpe`` / ``net_pnl_usd`` / ``n_trades`` for the SAME
  experiment, so those three comparisons are ``"AVAILABLE"``. It carries NO
  committed fill-rate, slippage-distribution, elapsed-day-normalized
  trade-frequency, or daily-PnL-distribution artifact for any experiment
  today, so those four comparisons are ALWAYS reported ``"NOT_AVAILABLE"`` --
  never reconstructed from an unrelated artifact (CLAUDE.md; the Phase 20.1
  trade-ledger-provenance lesson).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel

from alpha_agent.paper.ledger import PaperFillRow
from alpha_agent.validation.runner import BacktestRun, backtest_run_from_cpp_json
from alpha_agent.validation.trading_day import ValidationDayPlan

#: the two states every comparison-availability field may take. Never a
#: silent third option -- an unavailable comparison is always explicit.
NOT_AVAILABLE = "NOT_AVAILABLE"
AVAILABLE = "AVAILABLE"


class RealizedSlippageEvidence(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    status: str = NOT_AVAILABLE
    n_fills: int = 0
    mean_slippage_ticks: float | None = None
    mean_slippage_usd: float | None = None
    total_slippage_usd: float | None = None
    #: fills whose instrument_id had no matching contracts-frame row this
    #: step (should not happen in practice -- contracts is the same bundle
    #: sent to the C++ engine -- but never silently dropped without a count).
    n_fills_missing_contract_spec: int = 0
    backtest_comparison_status: str = NOT_AVAILABLE
    backtest_mean_slippage_ticks: float | None = None


class TradeFrequencyEvidence(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    status: str = NOT_AVAILABLE
    n_trades: int = 0
    elapsed_trading_days: int = 0
    trades_per_day: float | None = None
    backtest_comparison_status: str = NOT_AVAILABLE
    backtest_trades_per_day: float | None = None


class PnlDistributionEvidence(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    status: str = NOT_AVAILABLE
    n_days: int = 0
    mean_daily_pnl_usd: float | None = None
    std_daily_pnl_usd: float | None = None
    min_daily_pnl_usd: float | None = None
    max_daily_pnl_usd: float | None = None
    n_positive_days: int = 0
    n_negative_days: int = 0
    backtest_comparison_status: str = NOT_AVAILABLE


class DriftReport(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "paper-drift-report/2"
    run_id: str
    as_of_ts_ns: int

    backtest_daily_sharpe: float | None
    backtest_annualized_sharpe: float | None
    backtest_net_pnl_usd: float | None
    backtest_n_trades: int | None

    paper_daily_sharpe: float | None
    paper_annualized_sharpe: float | None
    paper_net_pnl_usd: float
    paper_n_trades: int
    paper_n_fills: int

    orders_generated: int
    fill_rate: float | None  # fills / orders, None if no orders yet

    configured_commission_per_contract_usd: float
    configured_slippage_ticks: float
    configured_spread_ticks: float

    realized_slippage: RealizedSlippageEvidence
    trade_frequency: TradeFrequencyEvidence
    pnl_distribution: PnlDistributionEvidence

    #: per-comparison-dimension AVAILABLE / NOT_AVAILABLE, computed ONLY from
    #: whether a real, experiment-bound comparable backtest artifact exists --
    #: never inferred, never approximated from an unrelated artifact.
    comparison_availability: dict[str, str]

    flags: tuple[str, ...] = ()


def paper_backtest_run(
    cpp_json: dict, *, capital_base_usd: float, validation_day_plan: ValidationDayPlan
) -> BacktestRun:
    """The paper run's own cumulative-window ``BacktestRun`` -- the exact same
    helper the research validation path calls, over the paper CLI's official
    ``daily_equity`` trace."""
    return backtest_run_from_cpp_json(
        cpp_json, capital_base_usd=capital_base_usd, validation_day_plan=validation_day_plan
    )


def _realized_slippage_evidence(fills: Sequence[PaperFillRow], contracts: pd.DataFrame) -> RealizedSlippageEvidence:
    """Descriptive $-and-tick slippage stats from the official per-fill C++
    audit trail (``PaperFillRow.slippage_ticks``), converted to USD via the
    SAME contract tick_size/multiplier the C++ engine itself used (the
    ``contracts`` bundle sent to it this step) -- never the run's configured
    scalar assumption."""
    if not fills:
        return RealizedSlippageEvidence(status=NOT_AVAILABLE)
    spec_by_instrument = {
        int(row.instrument_id): (float(row.tick_size), float(row.multiplier))
        for row in contracts.itertuples()
    }
    ticks: list[float] = []
    usd: list[float] = []
    missing = 0
    for f in fills:
        ticks.append(f.slippage_ticks)
        spec = spec_by_instrument.get(f.instrument_id)
        if spec is None:
            missing += 1
            continue
        tick_size, multiplier = spec
        usd.append(abs(f.slippage_ticks) * tick_size * multiplier * abs(f.quantity))
    return RealizedSlippageEvidence(
        status=AVAILABLE,
        n_fills=len(fills),
        mean_slippage_ticks=float(np.mean(ticks)),
        mean_slippage_usd=float(np.mean(usd)) if usd else None,
        total_slippage_usd=float(np.sum(usd)) if usd else None,
        n_fills_missing_contract_spec=missing,
    )


def _trade_frequency_evidence(*, n_trades: int, elapsed_trading_days: int) -> TradeFrequencyEvidence:
    if elapsed_trading_days <= 0:
        return TradeFrequencyEvidence(
            status=NOT_AVAILABLE, n_trades=n_trades, elapsed_trading_days=elapsed_trading_days
        )
    return TradeFrequencyEvidence(
        status=AVAILABLE,
        n_trades=n_trades,
        elapsed_trading_days=elapsed_trading_days,
        trades_per_day=n_trades / elapsed_trading_days,
    )


def _pnl_distribution_evidence(daily_pnl_usd: Sequence[float]) -> PnlDistributionEvidence:
    if not daily_pnl_usd:
        return PnlDistributionEvidence(status=NOT_AVAILABLE)
    arr = np.asarray(daily_pnl_usd, dtype=float)
    return PnlDistributionEvidence(
        status=AVAILABLE,
        n_days=int(arr.size),
        mean_daily_pnl_usd=float(arr.mean()),
        std_daily_pnl_usd=float(arr.std(ddof=0)),
        min_daily_pnl_usd=float(arr.min()),
        max_daily_pnl_usd=float(arr.max()),
        n_positive_days=int(np.count_nonzero(arr > 0)),
        n_negative_days=int(np.count_nonzero(arr < 0)),
    )


def build_drift_report(
    *,
    run_id: str,
    as_of_ts_ns: int,
    cpp_json: dict,
    paper_run: BacktestRun,
    fills: Sequence[PaperFillRow],
    contracts: pd.DataFrame,
    elapsed_trading_days: int,
    backtest_daily_sharpe: float | None,
    backtest_annualized_sharpe: float | None,
    backtest_net_pnl_usd: float | None,
    backtest_n_trades: int | None,
) -> DriftReport:
    orders = int(cpp_json.get("orders", 0))
    fills_count = int(cpp_json.get("fills", 0))
    fill_rate = (fills_count / orders) if orders > 0 else None

    realized_slippage = _realized_slippage_evidence(fills, contracts)
    trade_frequency = _trade_frequency_evidence(
        n_trades=paper_run.n_trades, elapsed_trading_days=elapsed_trading_days
    )
    pnl_distribution = _pnl_distribution_evidence(paper_run.daily.daily_pnl_usd)

    # No committed ResultRecord carries a fill-rate, a slippage distribution, an
    # elapsed-day-normalized trade-frequency, or a daily-PnL-distribution array
    # for ANY experiment today (alpha_agent.registry.models.ResultRecord has no
    # such field) -- these four stay NOT_AVAILABLE until a future phase commits
    # a real, experiment-bound artifact for them. Never approximated from an
    # unrelated artifact (the Phase 20.1 trade-ledger-provenance lesson).
    comparison_availability = {
        "daily_sharpe": AVAILABLE if backtest_daily_sharpe is not None else NOT_AVAILABLE,
        "net_pnl_usd": AVAILABLE if backtest_net_pnl_usd is not None else NOT_AVAILABLE,
        "n_trades": AVAILABLE if backtest_n_trades is not None else NOT_AVAILABLE,
        "fill_rate": NOT_AVAILABLE,
        "slippage": NOT_AVAILABLE,
        "trade_frequency": NOT_AVAILABLE,
        "pnl_distribution": NOT_AVAILABLE,
    }

    flags: list[str] = []
    if backtest_n_trades and backtest_n_trades > 0 and paper_run.n_trades == 0:
        flags.append("PAPER_HAS_NO_CLOSED_TRADES_YET")
    if (
        backtest_daily_sharpe is not None
        and paper_run.n_trades >= 5  # avoid a noisy verdict on a handful of days
    ):
        delta = paper_run.daily_sharpe - backtest_daily_sharpe
        # a materially worse realized Sharpe than the committed backtest,
        # scaled by the backtest's own magnitude so this is not a fixed
        # arbitrary constant.
        if delta < -(abs(backtest_daily_sharpe) * 0.5 + 0.1):
            flags.append("PAPER_SHARPE_MATERIALLY_BELOW_BACKTEST")
    if backtest_net_pnl_usd is not None and backtest_net_pnl_usd > 0 and paper_run.net_pnl_usd < 0:
        flags.append("PAPER_PNL_SIGN_DIVERGES_FROM_BACKTEST")

    return DriftReport(
        run_id=run_id,
        as_of_ts_ns=as_of_ts_ns,
        backtest_daily_sharpe=backtest_daily_sharpe,
        backtest_annualized_sharpe=backtest_annualized_sharpe,
        backtest_net_pnl_usd=backtest_net_pnl_usd,
        backtest_n_trades=backtest_n_trades,
        paper_daily_sharpe=paper_run.daily_sharpe if paper_run.daily.n_days > 0 else None,
        paper_annualized_sharpe=paper_run.annualized_sharpe if paper_run.daily.n_days > 0 else None,
        paper_net_pnl_usd=paper_run.net_pnl_usd,
        paper_n_trades=paper_run.n_trades,
        paper_n_fills=paper_run.n_fills,
        orders_generated=orders,
        fill_rate=fill_rate,
        configured_commission_per_contract_usd=paper_run.commission_per_contract_usd,
        configured_slippage_ticks=paper_run.slippage_ticks,
        configured_spread_ticks=paper_run.spread_ticks,
        realized_slippage=realized_slippage,
        trade_frequency=trade_frequency,
        pnl_distribution=pnl_distribution,
        comparison_availability=comparison_availability,
        flags=tuple(flags),
    )
