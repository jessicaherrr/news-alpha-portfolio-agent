"""The canonical DAILY validation return series (sections 2 & 3).

Official economics stay C++-sourced. This module only **aggregates** the
authoritative ``BacktestResult::daily_equity`` trace (one end-of-session-day
equity snapshot the C++ ``PortfolioAccountant`` already computed) into daily
PnL and daily returns. It does not reconstruct fills, positions, commissions,
execution prices or portfolio equity.

Convention (documented, fixed):

    daily_pnl[D]         = official_C++_EOD_equity[D] - official_C++_EOD_equity[D-1]
    (for the first observed day D0, "EOD_equity[D-1]" == starting_capital_usd)
    validation_return[D] = daily_pnl[D] / validation_capital_base

``validation_capital_base`` is explicit, finite, > 0 and fixed for the whole run
(``ValidationSpec.capital_base_usd``; normally the initial equity). A session day
with no engine event produces **no** C++ trace point and therefore no return -- a
missing day never silently becomes a zero. A day that *was* observed but had no
economic PnL is a real ``0.0`` return, kept as an observation.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, model_validator

from alpha_agent.validation.trading_day import ValidationDayPlan

OFFICIAL_SOURCE = "cpp_portfolio_accountant_daily_equity_trace"

# C++ CLI ``daily_equity`` entry layout (backtest_targets_csv.cpp):
#   [session_day_index, ts_ns, equity_usd, net_realized_pnl_usd,
#    unrealized_pnl_usd, costs_usd, fills_cumulative, bars_cumulative]
_DE_DAY, _DE_TS, _DE_EQUITY, _DE_NETREAL, _DE_UNREAL, _DE_COSTS, _DE_FILLS, _DE_BARS = range(8)


class DailyReturnSeries(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    session_day_index: tuple[int, ...]
    trading_day: tuple[str, ...]           # canonical futures trading_day label per obs
    ts_ns: tuple[int, ...]                 # EOD event ts of that trading_day
    equity_usd: tuple[float, ...]          # official C++ end-of-day equity
    daily_pnl_usd: tuple[float, ...]
    returns: tuple[float, ...]             # daily_pnl / capital_base
    bars_cumulative: tuple[int, ...]
    fills_cumulative: tuple[int, ...]

    capital_base_usd: float = Field(gt=0)
    starting_capital_usd: float = Field(gt=0)
    # "trading_day" (canonical, section 1) or "utc_day" (legacy diagnostics only).
    day_basis: str = "trading_day"
    day_plan_identity: str | None = None
    official_source: str = OFFICIAL_SOURCE

    @model_validator(mode="after")
    def _lengths(self) -> DailyReturnSeries:
        n = len(self.session_day_index)
        for name in ("trading_day", "ts_ns", "equity_usd", "daily_pnl_usd", "returns",
                     "bars_cumulative", "fills_cumulative"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"DailyReturnSeries.{name} length {len(getattr(self, name))} != {n}")
        if n >= 2 and any(
            self.session_day_index[i] <= self.session_day_index[i - 1] for i in range(1, n)
        ):
            raise ValueError("session_day_index must be strictly increasing (chronological)")
        if not all(np.isfinite(self.returns)):
            raise ValueError("non-finite daily return")
        return self

    # -- derived --------------------------------------------------------------
    @property
    def n_days(self) -> int:
        return len(self.returns)

    @property
    def n_nonzero_return_days(self) -> int:
        return int(np.count_nonzero(np.asarray(self.returns, dtype=float)))

    def returns_array(self) -> np.ndarray:
        return np.asarray(self.returns, dtype=float)

    def pnl_array(self) -> np.ndarray:
        return np.asarray(self.daily_pnl_usd, dtype=float)

    @property
    def gross_oos_net_pnl_usd(self) -> float:
        """Total PnL over the series == final EOD equity - starting equity of the
        first observed day (still C++-sourced, just summed)."""
        return float(np.sum(self.pnl_array()))


def daily_returns_from_cpp_trace(
    cpp_json: dict,
    *,
    capital_base_usd: float | None = None,
    validation_day_plan: ValidationDayPlan | None = None,
    initial_equity_usd: float | None = None,
) -> DailyReturnSeries:
    """Aggregate one C++ ``quant_backtest_targets_csv`` JSON result into the
    canonical daily validation return series.

    ``validation_day_plan`` -- the canonical futures ``trading_day`` boundaries
    the C++ run was given (Phase 13.1). When present, the C++ ``daily_equity``
    points align 1:1 with ``plan.days`` and each observation carries its
    ``trading_day`` label. When absent, the C++ trace is legacy UTC-day buckets
    (``day_basis = "utc_day"``, diagnostics only -- not canonical for CME).

    ``capital_base_usd`` defaults to ``starting_capital_usd``; explicit, finite,
    > 0. ``initial_equity_usd`` overrides the equity used for the first day's PnL
    (an independently flat-started fold seeds it explicitly, section 3).
    """
    trace = cpp_json.get("daily_equity")
    if trace is None:
        raise ValueError(
            "C++ result has no 'daily_equity' trace -- rebuild the C++ core "
            "(BacktestResult::daily_equity, Phase 13 additive export)"
        )
    starting = float(cpp_json.get("starting_capital_usd", 0.0))
    if not (starting > 0.0):
        raise ValueError(f"starting_capital_usd must be > 0, got {starting!r}")
    base = float(capital_base_usd) if capital_base_usd is not None else starting
    if not (np.isfinite(base) and base > 0.0):
        raise ValueError(f"validation capital base must be finite and > 0, got {base!r}")
    initial_equity = float(initial_equity_usd) if initial_equity_usd is not None else starting

    basis = str(cpp_json.get("daily_equity_basis", "utc_day"))
    if validation_day_plan is not None:
        if basis != "trading_day":
            raise ValueError(
                f"a ValidationDayPlan was supplied but the C++ trace basis is {basis!r}; "
                "pass the validation_days.csv boundaries to the CLI"
            )
        if len(trace) != validation_day_plan.n_days:
            raise ValueError(
                f"C++ trading-day trace has {len(trace)} points but the plan has "
                f"{validation_day_plan.n_days} trading days"
            )
        td_labels = validation_day_plan.trading_day_labels()
        day_plan_identity: str | None = validation_day_plan.identity()
    else:
        td_labels = None
        day_plan_identity = None

    days: list[int] = []
    tday: list[str] = []
    ts: list[int] = []
    equity: list[float] = []
    pnl: list[float] = []
    rets: list[float] = []
    bars: list[int] = []
    fills: list[int] = []

    prev_equity = initial_equity
    prev_day: int | None = None
    for i, row in enumerate(trace):
        d = int(row[_DE_DAY])
        if prev_day is not None and d <= prev_day:
            raise ValueError("C++ daily_equity trace is not strictly day-ordered")
        eq = float(row[_DE_EQUITY])
        day_pnl = eq - prev_equity
        days.append(d)
        tday.append(td_labels[i] if td_labels is not None
                    else _utc_day_label(int(row[_DE_TS])))
        ts.append(int(row[_DE_TS]))
        equity.append(eq)
        pnl.append(day_pnl)
        rets.append(day_pnl / base)
        bars.append(int(row[_DE_BARS]))
        fills.append(int(row[_DE_FILLS]))
        prev_equity = eq
        prev_day = d

    return DailyReturnSeries(
        session_day_index=tuple(days),
        trading_day=tuple(tday),
        ts_ns=tuple(ts),
        equity_usd=tuple(equity),
        daily_pnl_usd=tuple(pnl),
        returns=tuple(rets),
        bars_cumulative=tuple(bars),
        fills_cumulative=tuple(fills),
        capital_base_usd=base,
        starting_capital_usd=starting,
        day_basis="trading_day" if validation_day_plan is not None else "utc_day",
        day_plan_identity=day_plan_identity,
    )


def _utc_day_label(ts_ns: int) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(ts_ns / 1e9, tz=_dt.UTC).date().isoformat() + " (utc)"


def concat_oos_series(parts: list[DailyReturnSeries]) -> DailyReturnSeries:
    """Concatenate per-fold out-of-sample daily series into one OOS series
    (section 6: each fold is an independent flat-start backtest, so its daily
    PnL / returns are self-contained and simply appended). Session-day indices
    are re-based to stay strictly increasing across folds while preserving each
    fold's internal spacing."""
    if not parts:
        raise ValueError("concat_oos_series needs at least one fold series")
    base = parts[0].capital_base_usd
    starting = parts[0].starting_capital_usd
    for p in parts[1:]:
        if p.capital_base_usd != base:
            raise ValueError("all folds must share one validation capital base")

    basis = parts[0].day_basis
    days: list[int] = []
    tday: list[str] = []
    ts: list[int] = []
    equity: list[float] = []
    pnl: list[float] = []
    rets: list[float] = []
    bars: list[int] = []
    fills: list[int] = []
    day_cursor = 0
    bar_cursor = 0
    fill_cursor = 0
    for p in parts:
        first = p.session_day_index[0] if p.n_days else 0
        for i in range(p.n_days):
            days.append(day_cursor + (p.session_day_index[i] - first) + 1)
            tday.append(p.trading_day[i])
            ts.append(p.ts_ns[i])
            equity.append(p.equity_usd[i])
            pnl.append(p.daily_pnl_usd[i])
            rets.append(p.returns[i])
            bars.append(bar_cursor + p.bars_cumulative[i])
            fills.append(fill_cursor + p.fills_cumulative[i])
        if p.n_days:
            day_cursor = days[-1]
            bar_cursor = bars[-1]
            fill_cursor = fills[-1]

    return DailyReturnSeries(
        session_day_index=tuple(days),
        trading_day=tuple(tday),
        ts_ns=tuple(ts),
        equity_usd=tuple(equity),
        daily_pnl_usd=tuple(pnl),
        returns=tuple(rets),
        bars_cumulative=tuple(bars),
        fills_cumulative=tuple(fills),
        capital_base_usd=base,
        starting_capital_usd=starting,
        day_basis=basis,
    )
