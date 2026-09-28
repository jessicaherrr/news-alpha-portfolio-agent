"""Deterministic, hand-computable fixtures for the Phase 09 feature engine.

No network. Every frame here is small enough to check feature values by hand.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from alpha_agent.features.source import SourceSeries
from alpha_agent.schemas.market_data import PriceDomain

MIN_NS = 60_000_000_000
BASE_NS = pd.Timestamp("2026-09-01T14:00:00Z").value


def _ohlc_from_close(close: np.ndarray, *, spread: float = 1.0) -> dict:
    close = np.asarray(close, dtype="float64")
    openp = np.empty_like(close)
    openp[0] = close[0]
    openp[1:] = close[:-1]
    high = np.maximum(openp, close) + spread
    low = np.minimum(openp, close) - spread
    return {"open": openp, "high": high, "low": low, "close": close}


def bars_from_close(
    close,
    *,
    start_ns: int = BASE_NS,
    interval_ns: int = MIN_NS,
    volume=None,
    ts: np.ndarray | None = None,
    spread: float = 1.0,
    trading_day: list | None = None,
) -> pd.DataFrame:
    close = np.asarray(close, dtype="float64")
    n = len(close)
    if ts is None:
        ts = start_ns + np.arange(n) * interval_ns
    ts = np.asarray(ts, dtype="int64")
    cols = {"ts_event_ns": ts, **_ohlc_from_close(close, spread=spread)}
    cols["volume"] = (np.full(n, 100, dtype="int64") if volume is None
                      else np.asarray(volume, dtype="int64"))
    if trading_day is not None:
        cols["trading_day"] = pd.to_datetime(pd.Series(trading_day)).dt.date.to_numpy()
    return pd.DataFrame(cols)


def _mode_for(domain: PriceDomain, adjustment_mode: str | None) -> str | None:
    if domain is PriceDomain.BACK_ADJUSTED and adjustment_mode is None:
        return "point_in_time"
    return adjustment_mode


def linear_source(n: int = 40, *, start: float = 100.0, step: float = 0.5,
                  domain: PriceDomain = PriceDomain.RAW_CONTINUOUS,
                  adjustment_mode: str | None = None) -> SourceSeries:
    close = start + step * np.arange(n)
    return SourceSeries(frame=bars_from_close(close), price_domain=domain,
                        adjustment_mode=_mode_for(domain, adjustment_mode),
                        identity={"continuous_symbol": "NQ.v.0"})


def source_from_close(
    close, *, domain: PriceDomain = PriceDomain.RAW_CONTINUOUS,
    adjustment_mode: str | None = None, **kw
) -> SourceSeries:
    return SourceSeries(frame=bars_from_close(close, **kw), price_domain=domain,
                        adjustment_mode=_mode_for(domain, adjustment_mode),
                        identity={"continuous_symbol": "NQ.v.0"})


# A CL-style path straight through zero into negative territory (Phase 08.2).
SIGNED_CLOSES = [5.0, 3.0, 1.0, 0.0, -5.0, -20.0, -10.0, -12.0, -8.0, -15.0]


def signed_source(domain: PriceDomain = PriceDomain.RAW_CONTINUOUS,
                  adjustment_mode: str | None = None) -> SourceSeries:
    return SourceSeries(
        frame=bars_from_close(SIGNED_CLOSES, spread=0.5),
        price_domain=domain,
        adjustment_mode=_mode_for(domain, adjustment_mode),
        identity={"continuous_symbol": "CL.v.0"},
    )


def gap_source(*, gap_after: int = 5, gap_intervals: int = 200) -> SourceSeries:
    """10 one-minute bars with a large hole after bar ``gap_after``."""
    n = 10
    ts = list(BASE_NS + np.arange(gap_after) * MIN_NS)
    nxt = ts[-1] + gap_intervals * MIN_NS
    ts += list(nxt + np.arange(n - gap_after) * MIN_NS)
    close = 100.0 + np.arange(n)
    return SourceSeries(
        frame=bars_from_close(close, ts=np.asarray(ts, dtype="int64")),
        price_domain=PriceDomain.RAW_CONTINUOUS,
        identity={"continuous_symbol": "NQ.v.0"},
    )


def session_source() -> SourceSeries:
    """12 bars spanning two trading days (6 + 6)."""
    close = 100.0 + np.arange(12) * 0.25
    td = [date(2026, 9, 1)] * 6 + [date(2026, 9, 2)] * 6
    return SourceSeries(
        frame=bars_from_close(close, trading_day=td),
        price_domain=PriceDomain.RAW_CONTINUOUS,
        identity={"continuous_symbol": "NQ.v.0"},
    )


def synthetic_roll_sources() -> dict:
    """Run the real Phase-04 pipeline on a hand-made single roll:

      old contract (NQU6) ~= 100 for 5 bars, new contract (NQZ6) ~= 110 for 5
      bars, roll basis (additive_gap) = +10.

    The unadjusted continuous series has an artificial +10 jump at the roll.
    Returns a dict of SourceSeries plus the key timestamps.
    """
    from alpha_agent.data.backadjust import AdjustmentMode, build_back_adjusted_series
    from alpha_agent.data.continuous import build_continuous_series
    from alpha_agent.data.rolls import build_roll_events
    from futures_fixtures import registry as p4_registry
    from futures_fixtures import two_contract_history

    reg = p4_registry()
    bars = two_contract_history(u6_close=100.0, z6_open=110.0)
    rolls, _ = build_roll_events(bars, continuous_symbol="NQ.v.0", registry=reg)
    cont, _ = build_continuous_series(bars, continuous_symbol="NQ.v.0", registry=reg, rolls=rolls)
    roll_ts = int(rolls[0].effective_ts_ns)
    pre_ts = int(cont.loc[cont["ts_event_ns"] < roll_ts, "ts_event_ns"].iloc[-1])

    retro, _ = build_back_adjusted_series(
        cont, rolls, continuous_symbol="NQ.v.0",
        mode=AdjustmentMode.RETROSPECTIVE_RESEARCH,
    )
    pit, _ = build_back_adjusted_series(
        cont, rolls, continuous_symbol="NQ.v.0",
        mode=AdjustmentMode.POINT_IN_TIME, as_of_ts_ns=pre_ts,
    )
    front_raw = bars[bars["raw_symbol"] == "NQU6"].reset_index(drop=True)

    return {
        "roll_ts": roll_ts,
        "pre_ts": pre_ts,
        "additive_gap": float(rolls[0].additive_gap),
        "raw_contract": SourceSeries(
            frame=front_raw, price_domain=PriceDomain.RAW_CONTRACT,
            identity={"raw_symbol": "NQU6", "root_symbol": "NQ"},
        ),
        "raw_continuous": SourceSeries(
            frame=cont, price_domain=PriceDomain.RAW_CONTINUOUS,
            identity={"continuous_symbol": "NQ.v.0"},
        ),
        "backadj_retro": SourceSeries(
            frame=retro, price_domain=PriceDomain.BACK_ADJUSTED,
            identity={"continuous_symbol": "NQ.v.0"},
        ),
        "backadj_pit": SourceSeries(
            frame=pit, price_domain=PriceDomain.BACK_ADJUSTED,
            identity={"continuous_symbol": "NQ.v.0"},
        ),
    }


def continuous_roll_source() -> SourceSeries:
    """A continuous feed: NQM6 (id 501) for 5 bars, then NQU6 (id 502) for 6 bars.
    One observed roll at index 5."""
    n1, n2 = 5, 6
    close = np.concatenate([100.0 + np.arange(n1), 200.0 + np.arange(n2)])
    ts = BASE_NS + np.arange(n1 + n2) * MIN_NS
    aid = [501] * n1 + [502] * n2
    araw = ["NQM6"] * n1 + ["NQU6"] * n2
    td = [date(2026, 9, 1)] * 3 + [date(2026, 9, 2)] * 4 + [date(2026, 9, 3)] * 4
    frame = bars_from_close(close, ts=ts)
    frame["active_instrument_id"] = aid
    frame["active_raw_symbol"] = araw
    frame["is_roll_boundary"] = [False] * n1 + [True] + [False] * (n2 - 1)
    frame["trading_day"] = pd.to_datetime(pd.Series(td)).dt.date.to_numpy()
    return SourceSeries(frame=frame, price_domain=PriceDomain.RAW_CONTINUOUS,
                        identity={"continuous_symbol": "NQ.v.0"})
