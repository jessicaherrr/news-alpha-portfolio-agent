"""Descriptive statistics computed from the canonical DAILY validation series
(section 9). Every input series is already DAILY (one observation per observed
session day), so Sharpe annualises with ``sqrt(TRADING_DAYS_PER_YEAR)`` only.
There is no per-minute return series and no intraday-bar annualisation factor
anywhere in this framework -- the previous incorrect intraday convention is not
reused (section 9).
"""
from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, Field

# The series is DAILY. This is the ONLY annualisation constant in the framework.
TRADING_DAYS_PER_YEAR = 252.0


def mean_daily_return(returns: np.ndarray) -> float:
    x = np.asarray(returns, dtype=float)
    return float(x.mean()) if x.size else float("nan")


def daily_volatility(returns: np.ndarray) -> float:
    x = np.asarray(returns, dtype=float)
    return float(x.std(ddof=1)) if x.size >= 2 else float("nan")


def daily_sharpe(returns: np.ndarray) -> float:
    """Sharpe of a DAILY return series, NOT annualised (mean / sd)."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0.0:
        return float("nan")
    return float(x.mean() / sd)


def annualized_sharpe(returns: np.ndarray, days_per_year: float = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe from a DAILY series: daily_sharpe * sqrt(252)."""
    s = daily_sharpe(returns)
    return float(s * math.sqrt(days_per_year)) if np.isfinite(s) else float("nan")


def downside_deviation(returns: np.ndarray, mar: float = 0.0) -> float:
    x = np.asarray(returns, dtype=float)
    below = np.minimum(x - mar, 0.0)
    if x.size < 2:
        return float("nan")
    return float(math.sqrt(np.mean(below**2)))


def sortino_ratio(returns: np.ndarray, days_per_year: float = TRADING_DAYS_PER_YEAR) -> float:
    x = np.asarray(returns, dtype=float)
    dd = downside_deviation(x)
    if not np.isfinite(dd) or dd == 0.0:
        return float("nan")
    return float(x.mean() / dd * math.sqrt(days_per_year))


def equity_curve_from_returns(returns: np.ndarray) -> np.ndarray:
    """Cumulative (additive) return path, seeded at 0."""
    return np.cumsum(np.asarray(returns, dtype=float))


def max_drawdown_return(returns: np.ndarray) -> float:
    """Max drawdown of the additive cumulative-return path (<= 0)."""
    curve = equity_curve_from_returns(returns)
    if curve.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(np.concatenate(([0.0], curve)))[1:]
    return float(np.min(curve - peak))


def max_drawdown_usd(equity_usd: np.ndarray) -> float:
    x = np.asarray(equity_usd, dtype=float)
    if x.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(x)
    return float(np.min(x - peak))


def sample_skewness(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    n = a.size
    if n < 3:
        return float("nan")
    m = a.mean()
    s = a.std(ddof=0)
    if s == 0.0:
        return 0.0
    return float(np.mean(((a - m) / s) ** 3))


def sample_kurtosis(x: np.ndarray) -> float:
    """Non-excess (Pearson) kurtosis; a normal sample -> ~3.0."""
    a = np.asarray(x, dtype=float)
    n = a.size
    if n < 4:
        return float("nan")
    m = a.mean()
    s = a.std(ddof=0)
    if s == 0.0:
        return 0.0
    return float(np.mean(((a - m) / s) ** 4))


class OosMetrics(BaseModel):
    """Descriptive out-of-sample metrics, all from the official daily series
    (section 9). Contains no verdict and no pass/fail semantics."""

    model_config = {"frozen": True, "extra": "forbid"}

    n_trading_days: int = Field(ge=0)
    n_nonzero_return_days: int = Field(ge=0)
    n_fills: int = Field(ge=0)
    n_trades: int = Field(ge=0)

    mean_daily_return: float
    daily_volatility: float
    daily_sharpe: float
    annualized_sharpe: float
    sortino_ratio: float
    downside_deviation: float
    max_drawdown_return: float

    return_skewness: float
    return_kurtosis: float

    oos_gross_pnl_usd: float
    oos_costs_usd: float
    oos_net_pnl_usd: float
    capital_base_usd: float

    positive_fold_count: int = Field(ge=0)
    total_fold_count: int = Field(ge=0)


def describe_oos(
    returns: np.ndarray,
    *,
    n_fills: int,
    n_trades: int,
    oos_gross_pnl_usd: float,
    oos_costs_usd: float,
    oos_net_pnl_usd: float,
    capital_base_usd: float,
    positive_fold_count: int,
    total_fold_count: int,
) -> OosMetrics:
    x = np.asarray(returns, dtype=float)
    return OosMetrics(
        n_trading_days=int(x.size),
        n_nonzero_return_days=int(np.count_nonzero(x)),
        n_fills=int(n_fills),
        n_trades=int(n_trades),
        mean_daily_return=mean_daily_return(x),
        daily_volatility=daily_volatility(x),
        daily_sharpe=daily_sharpe(x),
        annualized_sharpe=annualized_sharpe(x),
        sortino_ratio=sortino_ratio(x),
        downside_deviation=downside_deviation(x),
        max_drawdown_return=max_drawdown_return(x),
        return_skewness=sample_skewness(x),
        return_kurtosis=sample_kurtosis(x),
        oos_gross_pnl_usd=float(oos_gross_pnl_usd),
        oos_costs_usd=float(oos_costs_usd),
        oos_net_pnl_usd=float(oos_net_pnl_usd),
        capital_base_usd=float(capital_base_usd),
        positive_fold_count=int(positive_fold_count),
        total_fold_count=int(total_fold_count),
    )
