"""Decode immutable raw DBN artifacts into vendor-shaped staging frames.

The authoritative raw artifact is the original ``.dbn`` / ``.dbn.zst`` bytes.
Everything this module produces (DataFrames, staging Parquet) is *derived* and
regenerable -- never the source of truth.

This module only reshapes real Databento fields into the vendor-frame layout.
Contract economics (tick size, point value) are derived from those fields by
``alpha_agent.data.contract_economics`` / ``parse_definition_frame`` -- never
guessed from ``contract_multiplier`` (a sentinel on GLBX.MDP3).
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from alpha_agent.schemas.market_data import (
    VENDOR_DEFINITION_COLUMNS,
    VENDOR_OHLCV_COLUMNS,
)

# Real Databento definition columns we read (verified against a live GLBX.MDP3
# NQ.FUT response -- see docs/DATABENTO_REALITY.md). "asset" is the root symbol.
# Contract economics are derived downstream (alpha_agent.data.contract_economics),
# never taken from "contract_multiplier" (a sentinel on GLBX.MDP3).
_DEF_REQUIRED = (
    "instrument_id", "raw_symbol", "asset", "exchange", "instrument_class",
    "min_price_increment", "min_price_increment_amount", "display_factor",
    "unit_of_measure", "unit_of_measure_qty", "activation", "expiration",
)


class SchemaMismatch(RuntimeError):
    """A real Databento response does not carry a field the pipeline needs."""


def _opt_float(value) -> float | str:
    """Sentinel / missing numeric -> "" ; otherwise the float."""
    from alpha_agent.data.contract_economics import usable_number

    v = usable_number(value)
    return "" if v is None else v


def load_dbn(path: str | Path):
    """Lazy-import wrapper around ``databento.DBNStore.from_file``."""
    try:
        from databento import DBNStore
    except ImportError as exc:  # pragma: no cover - requires data extra
        raise RuntimeError("Install data extras: pip install -e '.[data]'") from exc
    return DBNStore.from_file(str(path))


# --- OHLCV ----------------------------------------------------------------

def ohlcv_df_to_vendor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map a decoded ``ohlcv-1m`` ``to_df(price_type='fixed', pretty_ts=False)``
    frame to ``VENDOR_OHLCV_COLUMNS``.

    Prices stay as DBN fixed-point int64; ``ts_event`` stays raw UTC ns. The
    ``symbol`` column is a *label* -- with a continuous request + ``map_symbols``
    it is the requested smart symbol (e.g. ``NQ.v.0``), not a tradable contract,
    and it is optional. The canonical ``raw_symbol`` is resolved downstream from
    ``instrument_id``.
    """
    df = df.reset_index() if df.index.name in ("ts_event", "ts_recv") else df.copy()
    need = {"ts_event", "instrument_id", "open", "high", "low", "close", "volume"}
    missing = need - set(df.columns)
    if missing:
        raise SchemaMismatch(f"ohlcv response is missing expected columns: {sorted(missing)}")
    symbol = (
        df["symbol"].astype("string").fillna("")
        if "symbol" in df.columns
        else pd.Series([""] * len(df), index=df.index, dtype="string")
    )
    out = pd.DataFrame(
        {
            "ts_event": df["ts_event"].astype("int64"),
            "instrument_id": df["instrument_id"].astype("int64"),
            "open": df["open"].astype("int64"),
            "high": df["high"].astype("int64"),
            "low": df["low"].astype("int64"),
            "close": df["close"].astype("int64"),
            "volume": df["volume"].astype("int64"),
            "symbol": symbol,
        },
        columns=list(VENDOR_OHLCV_COLUMNS),
    )
    return out


# --- definitions --------------------------------------------------------

def definition_df_to_vendor_frame(
    df: pd.DataFrame,
    *,
    keep_instrument_ids: set[int] | None = None,
    outrights_only: bool = True,
) -> pd.DataFrame:
    """Map a decoded ``definition`` ``to_df`` frame to ``VENDOR_DEFINITION_COLUMNS``.

    ``keep_instrument_ids`` restricts output to the contracts actually seen in
    the bars (join on ``instrument_id`` -- the authoritative key, since Databento
    does not support continuous -> raw_symbol). Raw fields are passed through
    unchanged; economics are derived by ``parse_definition_frame``. Raises
    ``SchemaMismatch`` rather than adapting silently.
    """
    df = df.reset_index() if df.index.name in ("ts_event", "ts_recv") else df.copy()
    missing = [c for c in _DEF_REQUIRED if c not in df.columns]
    if missing:
        raise SchemaMismatch(f"definition response is missing expected columns: {missing}")

    if outrights_only and "instrument_class" in df.columns:
        df = df[df["instrument_class"].astype(str).str.upper().isin({"F", "FUTURE"})]
    if keep_instrument_ids is not None:
        df = df[df["instrument_id"].astype("int64").isin(keep_instrument_ids)]
    # one row per contract (definitions can repeat intraday); keep the latest
    df = df.sort_values("ts_event").drop_duplicates("instrument_id", keep="last")

    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "instrument_id": int(r["instrument_id"]),
                "raw_symbol": str(r["raw_symbol"]).strip(),
                "asset": str(r["asset"]).strip(),
                "exchange": str(r["exchange"]).strip(),
                "instrument_class": str(r["instrument_class"]).strip(),
                "min_price_increment": float(r["min_price_increment"]),
                "min_price_increment_amount": float(r["min_price_increment_amount"]),
                "display_factor": float(r["display_factor"]),
                "unit_of_measure": str(r["unit_of_measure"]).strip(),
                "unit_of_measure_qty": float(r["unit_of_measure_qty"]),
                "main_fraction": _opt_float(r.get("main_fraction")),
                "contract_multiplier": r.get("contract_multiplier", ""),
                "activation": int(r["activation"]),
                "expiration": int(r["expiration"]),
                # Databento definitions carry no distinct first-notice / last-trade
                # field. expiration == last trade for cash-settled index futures.
                "first_notice": "",
                "last_trade": "",
            }
        )
    return pd.DataFrame(rows, columns=list(VENDOR_DEFINITION_COLUMNS))


# --- staging -----------------------------------------------------------

def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def stage_frame(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Write a decoded vendor-shaped frame to a staging location (regenerable)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out
