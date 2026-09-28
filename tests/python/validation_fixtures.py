"""Deterministic, network-free fixtures for Phase 13 reliability validation.

Two kinds:

* **synthetic statistical fixtures** (section 23) -- pure daily-PnL vectors that
  exercise the STATISTICS framework independently of any market claim. A/B/C/D/
  E/F/G below. These are software tests, never presented as market results.
* **daily bar fixtures** -- one OHLC bar per UTC day so the real C++ path
  (features -> DSL -> targets -> engine -> daily_equity trace) can be driven
  end-to-end without minute-resolution data volume.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from alpha_agent.schemas.market_data import CONTRACT_COLUMNS

DAY_NS = 86_400_000_000_000
BASE_NS = pd.Timestamp("2026-01-05T00:00:00Z").value  # a Monday, UTC midnight


# ======================================================================
# daily bar fixtures (real C++ path)
# ======================================================================
def daily_bars(
    closes, *, instrument_id: int, start_ns: int = BASE_NS, spread: float = 1.0
) -> pd.DataFrame:
    close = np.asarray(closes, dtype="float64")
    openp = np.empty_like(close)
    openp[0] = close[0]
    openp[1:] = close[:-1]
    high = np.maximum(openp, close) + spread
    low = np.minimum(openp, close) - spread
    ts = start_ns + np.arange(len(close)) * DAY_NS
    return pd.DataFrame(
        {
            "ts_event_ns": ts,
            "instrument_id": instrument_id,
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000,
        }
    )


def one_contract(*, instrument_id: int, raw_symbol: str, root: str, expiration_ns: int):
    return pd.DataFrame(
        [
            {
                "instrument_id": instrument_id,
                "raw_symbol": raw_symbol,
                "root_symbol": root,
                "exchange": "XCME",
                "tick_size": 0.25,
                "multiplier": 20.0,
                "activation_ns": 1,
                "expiration_ns": expiration_ns,
                "first_notice_ns": "",
                "last_trade_ns": "",
            }
        ]
    )[list(CONTRACT_COLUMNS)]


# ---- intraday CME fixture: multiple bars per trading_day, each day spanning a
#      UTC midnight (17:00 America/Chicago boundary). Used to prove the canonical
#      validation series follows trading_day, not the UTC date.
_CME_EVENING_LOCAL = ("17:30", "20:00", "23:00")   # on calendar date D-1  -> trading_day D
_CME_MORNING_LOCAL = ("02:00", "08:00", "14:00")   # on calendar date D    -> trading_day D
CME_BARS_PER_DAY = len(_CME_EVENING_LOCAL) + len(_CME_MORNING_LOCAL)


def _weekday_trading_days(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:  # Mon..Fri
            out.append(d)
        d += timedelta(days=1)
    return out


def cme_intraday_bars(
    *,
    instrument_id: int,
    n_trading_days: int = 40,
    start: date = date(2026, 1, 6),
    tz: str = "America/Chicago",
    slope: float = 0.4,
    seed: int = 3,
    spread: float = 1.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (bars, trading_day_iso_labels). Each CME trading day gets
    ``CME_BARS_PER_DAY`` bars whose UTC timestamps straddle a UTC midnight."""
    rng = np.random.default_rng(seed)
    tdays = _weekday_trading_days(start, n_trading_days)
    ts_list: list[int] = []
    labels: list[str] = []
    for i, d in enumerate(tdays):
        prev = d - timedelta(days=1)
        for hhmm in _CME_EVENING_LOCAL:
            ts_list.append(pd.Timestamp(f"{prev} {hhmm}", tz=tz).tz_convert("UTC").value)
            labels.append(d.isoformat())
        for hhmm in _CME_MORNING_LOCAL:
            ts_list.append(pd.Timestamp(f"{d} {hhmm}", tz=tz).tz_convert("UTC").value)
            labels.append(d.isoformat())
    order = np.argsort(ts_list, kind="stable")
    ts = np.asarray(ts_list, dtype="int64")[order]
    labels = [labels[i] for i in order]
    n = len(ts)
    base = 100.0 + slope * np.arange(n) + np.cumsum(rng.normal(0, 0.3, size=n)) * 0.2
    openp = np.empty(n)
    openp[0] = base[0]
    openp[1:] = base[:-1]
    bars = pd.DataFrame(
        {
            "ts_event_ns": ts,
            "instrument_id": instrument_id,
            "open": openp,
            "high": np.maximum(openp, base) + spread,
            "low": np.minimum(openp, base) - spread,
            "close": base,
            "volume": 500,
        }
    )
    return bars, labels


def trend_closes(n: int = 300, *, slope: float = 0.25, noise: float = 0.8, seed: int = 7) -> list[float]:
    """A persistent up-trend with bounded noise -- a time-series momentum edge."""
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    path = 100.0 + slope * x + np.cumsum(rng.normal(0.0, noise, size=n)) * 0.3
    return list(path)


def choppy_closes(n: int = 300, *, amp: float = 4.0, seed: int = 11) -> list[float]:
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    return list(100.0 + amp * np.sin(x / 6.0) + rng.normal(0.0, 0.5, size=n))


# ======================================================================
# synthetic statistical fixtures (section 23) -- daily PnL (USD) vectors
# ======================================================================
def fixture_A_zero_alpha(n_days: int = 180, *, sigma: float = 250.0, seed: int = 0) -> np.ndarray:
    """A. zero-alpha / null strategy -- iid mean-zero daily PnL."""
    return np.random.default_rng(seed).normal(0.0, sigma, size=n_days)


def fixture_B_strong_signal(
    n_days: int = 180, *, mu: float = 120.0, sigma: float = 200.0, seed: int = 1
) -> np.ndarray:
    """B. clearly positive synthetic signal -- high, stable daily Sharpe."""
    return np.random.default_rng(seed).normal(mu, sigma, size=n_days)


def fixture_C_unstable_spike(seed: int = 2) -> dict[str, np.ndarray]:
    """C. unstable parameter spike -- the canonical point is strongly positive,
    every predeclared neighbour is ~zero-alpha."""
    rng = np.random.default_rng(seed)
    canonical = rng.normal(150.0, 200.0, size=180)
    neighbours = [rng.normal(0.0, 220.0, size=180) for _ in range(8)]
    return {"canonical": canonical, "neighbours": neighbours}


def fixture_D_broad_plateau(seed: int = 3) -> dict[str, np.ndarray]:
    """D. broad parameter plateau -- canonical and every neighbour show a similar
    positive edge."""
    rng = np.random.default_rng(seed)
    canonical = rng.normal(90.0, 190.0, size=180)
    neighbours = [rng.normal(85.0 + 6 * rng.normal(), 195.0, size=180) for _ in range(8)]
    return {"canonical": canonical, "neighbours": neighbours}


def fixture_E_cost_fragile(seed: int = 4) -> dict:
    """E. cost-fragile strategy -- positive gross edge, but the per-fill cost
    drag wipes it out at 2x. Returned as (gross daily PnL, fills/day, cost/fill)."""
    rng = np.random.default_rng(seed)
    gross = rng.normal(60.0, 150.0, size=180)
    return {"gross_daily_pnl": gross, "fills_per_day": 20, "cost_per_fill_1x": 3.0}


def fixture_F_too_small(seed: int = 5) -> np.ndarray:
    """F. too-small sample -- only a handful of observed days."""
    return np.random.default_rng(seed).normal(200.0, 150.0, size=12)


def fixture_G_regime_concentrated(seed: int = 6) -> dict:
    """G. regime-concentrated strategy -- essentially all PnL earned on
    'high_vol' days; flat elsewhere."""
    rng = np.random.default_rng(seed)
    n = 180
    labels = np.array(["low_vol", "mid_vol", "high_vol"])[rng.integers(0, 3, size=n)]
    pnl = np.where(labels == "high_vol", rng.normal(300.0, 120.0, size=n), rng.normal(0.0, 40.0, size=n))
    return {"daily_pnl": pnl, "labels": tuple(labels.tolist())}
