"""Deterministic vendor-like fixtures for the Phase 03 data pipeline.

No network, no databento SDK. These build the exact frames the pipeline consumes:
DBN-style OHLCV records (fixed-point int prices, raw ns timestamps, mapped
symbols) and DBN-style instrument-definition records.
"""
from __future__ import annotations

import io

import pandas as pd
from alpha_agent.data.raw_store import RawArtifact, store_raw
from alpha_agent.schemas.market_data import (
    PRICE_SCALE,
    VENDOR_DEFINITION_COLUMNS,
    VENDOR_OHLCV_COLUMNS,
)

MIN_NS = 60_000_000_000

# A normal NQ RTH morning (08:30-09:30 America/Chicago == 13:30-14:30 UTC on a
# non-DST-transition Monday).
SESSION_START_NS = pd.Timestamp("2025-03-17T13:30:00Z").value
NQ_INSTRUMENT_ID = 4021
NQ_RAW_SYMBOL = "NQM5"
# What a real continuous OHLCV response reports in its "symbol" column
# (stype_out=instrument_id + map_symbols): the requested smart symbol, not a
# tradable contract.
NQ_CONTINUOUS = "NQ.v.0"


def to_fixed(price: float) -> int:
    return round(price * PRICE_SCALE)


def vendor_definition_frame(rows: list[dict] | None = None) -> pd.DataFrame:
    """One NQ contract by default, using the REAL observed GLBX.MDP3 NQU6
    definition values. Each row may override any field."""
    default = {
        "instrument_id": NQ_INSTRUMENT_ID,
        "raw_symbol": NQ_RAW_SYMBOL,
        "asset": "NQ",
        "exchange": "XCME",
        "instrument_class": "F",
        "min_price_increment": 0.25,
        "min_price_increment_amount": 0.05,
        "display_factor": 0.01,
        "unit_of_measure": "IPNT",
        "unit_of_measure_qty": 20.0,
        "main_fraction": "",              # real value is the 255 sentinel -> unset
        "contract_multiplier": 2147483647,  # INT32_MAX sentinel -- must be ignored
        "activation": pd.Timestamp("2024-06-01T00:00:00Z").value,
        "expiration": pd.Timestamp("2025-06-20T13:30:00Z").value,
        "first_notice": "",
        "last_trade": "",
    }
    recs = [default] if rows is None else [{**default, **r} for r in rows]
    return pd.DataFrame(recs, columns=list(VENDOR_DEFINITION_COLUMNS))


def vendor_ohlcv_frame(
    *,
    n_bars: int = 60,
    instrument_id: int = NQ_INSTRUMENT_ID,
    symbol: str = NQ_CONTINUOUS,  # realistic default: the smart-symbol label
    start_ns: int = SESSION_START_NS,
    base_price: float = 20000.0,
    step: float = 0.25,
    rows_override: list[dict] | None = None,
) -> pd.DataFrame:
    """A clean run of ``n_bars`` one-minute bars, optionally extended/overridden."""
    recs: list[dict] = []
    for i in range(n_bars):
        o = base_price + i * step
        c = o + step
        recs.append(
            {
                "ts_event": start_ns + i * MIN_NS,
                "instrument_id": instrument_id,
                "open": to_fixed(o),
                "high": to_fixed(max(o, c) + step),
                "low": to_fixed(min(o, c) - step),
                "close": to_fixed(c),
                "volume": 100 + i,
                "symbol": symbol,
            }
        )
    if rows_override:
        recs.extend({**recs[0], **r} for r in rows_override)
    return pd.DataFrame(recs, columns=list(VENDOR_OHLCV_COLUMNS))


def _to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def store_raw_ohlcv(
    df: pd.DataFrame,
    root_dir,
    *,
    symbols: list[str] | None = None,
    dataset: str = "GLBX.MDP3",
    start: str = "2025-03-17",
    end: str = "2025-03-18",
) -> RawArtifact:
    return store_raw(
        _to_parquet_bytes(df),
        vendor="databento",
        dataset=dataset,
        schema="ohlcv-1m",
        stype_in="continuous",
        stype_out="instrument_id",
        symbols=symbols or ["NQ.v.0"],
        start=start,
        end=end,
        artifact_format="parquet",
        row_count=len(df),
        root_dir=root_dir,
    )


def store_raw_definitions(
    df: pd.DataFrame,
    root_dir,
    *,
    dataset: str = "GLBX.MDP3",
    start: str = "2025-03-17",
    end: str = "2025-03-18",
) -> RawArtifact:
    return store_raw(
        _to_parquet_bytes(df),
        vendor="databento",
        dataset=dataset,
        schema="definition",
        stype_in="parent",
        stype_out="instrument_id",
        symbols=["NQ.FUT"],
        start=start,
        end=end,
        artifact_format="parquet",
        row_count=len(df),
        root_dir=root_dir,
    )
