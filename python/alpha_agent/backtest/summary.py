"""Descriptive baseline run summary (Phase 11 sections 16 & 23).

Every number here is read **verbatim** from the C++ ``BacktestResult`` JSON. This
module performs no PnL arithmetic of its own -- official fills, realized PnL,
costs, positions, drawdown and portfolio output remain C++ Fill-derived. It
deliberately does **not** compute Sharpe / statistical significance (that would
duplicate Phase 13); only simple descriptive metrics.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

_DISCLAIMER = (
    "Phase 11 research baseline. NOT a claim of profitability, robustness, "
    "statistical significance or production readiness -- those belong to Phase 13 "
    "reliability validation. Official PnL is C++ Fill-derived."
)


class BaselineRunSummary(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    strategy_fingerprint: str
    root_symbol: str
    date_range_ns: tuple[int, int] | None

    # counts (from the C++ result)
    bars: int
    events: int
    signals: int
    orders: int
    fills: int
    closed_trades: int
    roll_count: int
    contracts_traded: list[str]
    n_contracts_traded: int

    # official realized PnL (C++ Fill-derived)
    gross_realized_pnl_usd: float
    costs_usd: float
    net_realized_pnl_usd: float
    max_drawdown_usd: float
    unrealized_pnl_usd_at_end: float

    # end state
    open_positions: int
    equity_usd: float
    starting_capital_usd: float

    # risk gate activity (reporting-only on the frozen reference path)
    risk_rejects: int
    risk_resizes: int

    official_pnl_source: str = Field(default="cpp_fill_events")
    disclaimer: str = Field(default=_DISCLAIMER)


def summarize_cpp_result(
    result: dict,
    *,
    strategy_fingerprint: str,
    root_symbol: str,
    date_range_ns: tuple[int, int] | None,
    contracts_traded: list[str] | None = None,
) -> BaselineRunSummary:
    """Wrap one line of ``quant_backtest_targets_csv`` JSON output in a typed,
    descriptive summary. ``contracts_traded`` (the sorted unique raw symbols) is
    not on the CLI JSON as a list -- pass it if known, else only the count is
    reported."""
    def _f(key: str, default: float = 0.0) -> float:
        return float(result.get(key, default))

    def _i(key: str, default: int = 0) -> int:
        return int(result.get(key, default))

    traded = list(contracts_traded or [])
    n_traded = len(traded) if traded else _i("unique_contracts")
    return BaselineRunSummary(
        strategy_fingerprint=strategy_fingerprint,
        root_symbol=root_symbol,
        date_range_ns=date_range_ns,
        bars=_i("bars"),
        events=_i("events"),
        signals=_i("signals"),
        orders=_i("orders"),
        fills=_i("fills"),
        closed_trades=_i("trades"),
        roll_count=_i("rolls"),
        contracts_traded=traded,
        n_contracts_traded=n_traded,
        gross_realized_pnl_usd=_f("gross_pnl_usd"),
        costs_usd=_f("costs_usd"),
        net_realized_pnl_usd=_f("net_pnl_usd"),
        max_drawdown_usd=_f("max_drawdown_usd"),
        unrealized_pnl_usd_at_end=_f("unrealized_pnl_usd"),
        open_positions=_i("open_positions"),
        equity_usd=_f("equity_usd"),
        starting_capital_usd=_f("starting_capital_usd"),
        risk_rejects=_i("risk_rejects"),
        risk_resizes=_i("risk_resizes"),
    )
