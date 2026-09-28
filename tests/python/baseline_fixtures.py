"""Deterministic, network-free fixtures for the Phase 11 baseline library."""
from __future__ import annotations

import numpy as np
import pandas as pd
from alpha_agent.features.source import SourceSeries
from alpha_agent.schemas.market_data import CONTRACT_COLUMNS, PriceDomain

MIN_NS = 60_000_000_000
BASE_NS = pd.Timestamp("2026-09-01T14:00:00Z").value


def _ohlc(close: np.ndarray, spread: float = 1.0) -> dict:
    close = np.asarray(close, dtype="float64")
    openp = np.empty_like(close)
    openp[0] = close[0]
    openp[1:] = close[:-1]
    high = np.maximum(openp, close) + spread
    low = np.minimum(openp, close) - spread
    return {"open": openp, "high": high, "low": low, "close": close}


def price_frame(close, *, start_ns: int = BASE_NS, spread: float = 1.0) -> pd.DataFrame:
    close = np.asarray(close, dtype="float64")
    ts = start_ns + np.arange(len(close)) * MIN_NS
    return pd.DataFrame({"ts_event_ns": ts, **_ohlc(close, spread), "volume": 100})


def price_source(
    close,
    *,
    root: str = "NQ",
    raw_symbol: str = "NQU6",
    domain: PriceDomain = PriceDomain.RAW_CONTRACT,
    spread: float = 1.0,
) -> SourceSeries:
    return SourceSeries(
        frame=price_frame(close, spread=spread),
        price_domain=domain,
        identity={"root_symbol": root, "raw_symbol": raw_symbol},
    )


def boundary_bars(close, *, instrument_id: int, start_ns: int = BASE_NS, spread: float = 1.0):
    f = price_frame(close, start_ns=start_ns, spread=spread)
    return pd.DataFrame(
        {
            "ts_event_ns": f["ts_event_ns"],
            "instrument_id": instrument_id,
            "open": f["open"],
            "high": f["high"],
            "low": f["low"],
            "close": f["close"],
            "volume": f["volume"],
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


def trending_closes(n: int = 80, *, slope: float = 0.4, wiggle: float = 0.6) -> list[float]:
    x = np.arange(n)
    return list(100.0 + slope * x + wiggle * np.sin(x / 3.0))


def mean_reverting_closes(n: int = 90, *, amp: float = 6.0) -> list[float]:
    x = np.arange(n)
    return list(100.0 + amp * np.sin(x / 5.0))


def breakout_closes() -> list[float]:
    # flat range, then a decisive upside break, then a downside break
    return (
        [100.0, 100.4, 99.6, 100.2, 99.8, 100.1, 99.9, 100.3] * 2
        + [104, 106, 108, 110, 112, 113, 114, 115]
        + [110, 105, 100, 96, 92, 90, 88, 86]
    )


# CL-style path straight through zero into negative territory (Phase 08.2 / section 20).
SIGNED_CLOSES = [
    5.0, 4.0, 3.0, 1.0, 0.0, -3.0, -8.0, -20.0, -15.0, -10.0,
    -12.0, -6.0, -2.0, -5.0, -9.0, -4.0, 1.0, 3.0, -1.0, -4.0,
]
