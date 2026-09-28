"""Deterministic, hand-constructed OHLC fixtures for the Phase 12 Silver Bullet
market-structure benchmark. No network. Every bar is specified explicitly so the
sweep / displacement / FVG / retracement geometry is checkable by hand."""
from __future__ import annotations

import numpy as np
import pandas as pd
from alpha_agent.features.source import SourceSeries
from alpha_agent.schemas.market_data import CONTRACT_COLUMNS, PriceDomain

MIN_NS = 60_000_000_000
# 2026-09-01 14:00:00Z == 09:00 America/Chicago (CDT) == NY-AM Silver Bullet.
BASE_NS = pd.Timestamp("2026-09-01T14:00:00Z").value


def sb_bars(
    rows: list[tuple[float, float, float, float]],
    *,
    start_ns: int = BASE_NS,
    interval_ns: int = MIN_NS,
    ts: np.ndarray | None = None,
    trading_day: list | None = None,
    is_roll_boundary: list[bool] | None = None,
    volume: int = 100,
) -> pd.DataFrame:
    arr = np.asarray(rows, dtype="float64")
    n = len(arr)
    if ts is None:
        ts = start_ns + np.arange(n) * interval_ns
    out = pd.DataFrame(
        {
            "ts_event_ns": np.asarray(ts, dtype="int64"),
            "open": arr[:, 0],
            "high": arr[:, 1],
            "low": arr[:, 2],
            "close": arr[:, 3],
            "volume": np.full(n, volume, dtype="int64"),
        }
    )
    if trading_day is not None:
        out["trading_day"] = pd.to_datetime(pd.Series(trading_day)).dt.date.to_numpy()
    if is_roll_boundary is not None:
        out["is_roll_boundary"] = np.asarray(is_roll_boundary, dtype=bool)
    return out


def sb_source(frame: pd.DataFrame, *, root: str = "NQ", raw: str = "NQU6",
              domain: PriceDomain = PriceDomain.RAW_CONTRACT,
              adjustment_mode: str | None = None) -> SourceSeries:
    ident = {"root_symbol": root, "raw_symbol": raw}
    if domain is PriceDomain.RAW_CONTINUOUS:
        ident = {"continuous_symbol": f"{root}.v.0", "root_symbol": root}
    return SourceSeries(frame=frame, price_domain=domain, identity=ident,
                        adjustment_mode=adjustment_mode)


_FLAT = (100.0, 100.5, 99.5, 100.0)


def _bullish_block() -> list[tuple[float, float, float, float]]:
    """sweep (Ti) -> displacement (Ti+1) -> FVG confirm (Ti+2) -> retracement (Ti+3).

    Prior flat range: high 100.5 / low 99.5.
    """
    return [
        (100.0, 101.5, 98.0, 101.0),   # sell-side sweep: low 98 < prior low 99.5; close 101 back above
        (101.0, 108.5, 100.5, 108.0),  # bullish displacement: body +7, close near the high
        (108.0, 110.0, 104.0, 109.0),  # FVG confirm: low 104 > sweep-bar high 101.5 -> gap [101.5, 104]
        (109.0, 109.0, 102.0, 103.0),  # retracement: low 102 <= midpoint 102.75
    ]


def _bearish_block() -> list[tuple[float, float, float, float]]:
    return [
        (100.0, 102.0, 98.5, 99.0),    # buy-side sweep: high 102 > prior high 100.5; close 99 back below
        (99.0, 99.5, 90.5, 91.0),      # bearish displacement: body -8, close near the low
        (91.0, 96.0, 90.0, 95.0),      # FVG confirm: high 96 < sweep-bar low 98.5 -> gap [96, 98.5]
        (95.0, 98.0, 95.0, 97.0),      # retracement: high 98 >= midpoint 97.25
    ]


def bullish_setup_source(*, preamble: int = 12, tail: int = 8, **kw) -> SourceSeries:
    rows = [_FLAT] * preamble + _bullish_block() + [
        (103.0 + i, 104.0 + i, 102.5 + i, 103.5 + i) for i in range(tail)
    ]
    return sb_source(sb_bars(rows), **kw)


def bearish_setup_source(*, preamble: int = 12, tail: int = 8, **kw) -> SourceSeries:
    rows = [_FLAT] * preamble + _bearish_block() + [
        (97.0 - i, 98.0 - i, 96.5 - i, 97.5 - i) for i in range(tail)
    ]
    return sb_source(sb_bars(rows), **kw)


def bullish_setup_rows(preamble: int = 12, tail: int = 8):
    return [_FLAT] * preamble + _bullish_block() + [
        (103.0 + i, 104.0 + i, 102.5 + i, 103.5 + i) for i in range(tail)
    ]


def prior_session_rows(*, day1: int = 10, day2_flat: int = 4):
    """Day 1 establishes a wide range (high 105 / low 95). Day 2: flat, then a
    sell-side sweep of DAY 1's low, then displacement / FVG / retracement."""
    d1 = [(100.0, 105.0, 95.0, 100.0)] * day1
    d2 = (
        [(100.0, 100.5, 99.5, 100.0)] * day2_flat
        + [
            (100.0, 100.5, 93.0, 100.0),   # sell-side sweep of day-1 low 95 (low 93, close back to 100)
            (100.0, 108.0, 99.5, 107.5),   # bullish displacement: body +7.5
            (107.5, 109.0, 103.0, 108.0),  # FVG: low 103 > sweep-bar high 100.5 -> gap [100.5, 103]
            (108.0, 108.0, 101.0, 102.0),  # retracement: low 101 <= midpoint 101.75
        ]
        + [(102.0, 103.0, 101.5, 102.5)] * 4
    )
    rows = d1 + d2
    td = ["2026-09-01"] * len(d1) + ["2026-09-02"] * len(d2)
    return rows, td


def contracts_one(*, instrument_id: int, raw: str = "NQU6", root: str = "NQ",
                  expiration_ns: int = BASE_NS + 10**15) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "instrument_id": instrument_id, "raw_symbol": raw, "root_symbol": root,
            "exchange": "XCME", "tick_size": 0.25, "multiplier": 20.0,
            "activation_ns": 1, "expiration_ns": expiration_ns,
            "first_notice_ns": "", "last_trade_ns": "",
        }]
    )[list(CONTRACT_COLUMNS)]


def boundary_bars_from_frame(frame: pd.DataFrame, *, instrument_id: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_event_ns": frame["ts_event_ns"].to_numpy("int64"),
            "instrument_id": instrument_id,
            "open": frame["open"], "high": frame["high"],
            "low": frame["low"], "close": frame["close"],
            "volume": frame["volume"],
        }
    )
