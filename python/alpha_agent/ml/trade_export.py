"""Typed reader for the ADDITIVE C++ closed-trade audit export (Phase 15A).

``cpp/include/quant_core/trade_export.hpp`` serialises the engine's already
recorded ``BacktestResult::trades`` -- the Fill-derived closed/reduced position
events -- to CSV. This module is the matching typed reader.

The point of the round trip is CLAUDE.md architecture boundary 3: the meta-label
target must be derived from **official C++ Fill-derived economics**, not from a
PnL number Python invented. Nothing here computes PnL; it parses what the engine
already decided.

Why the FILL export exists as well
----------------------------------
``ClosedTrade::costs_usd`` is, by the ledger's own definition
(``cpp/src/position_ledger.cpp``), the commission of the **closing fill only**.
The opening fill's commission is real, is counted in the engine's headline
``costs_usd`` / ``net_pnl_usd``, and is attributed to no ``ClosedTrade``. So
summing ``ClosedTrade.net_pnl_usd`` understates the cost of every episode by its
entry commission -- which would bias every meta-label optimistic, exactly in the
direction that makes a filter look useful.

Correcting the attribution inside the ledger would change the official C++
accounting path, so Phase 15 does not do that. It reads the fill audit trail
instead and attributes the FULL round-turn commission by summing what the engine
already charged. Python still prices nothing.
"""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, field_validator

#: Frozen column contract. Mirrors ``quant::kClosedTradeCsvHeader`` exactly.
CLOSED_TRADE_COLUMNS: tuple[str, ...] = (
    "trade_index",
    "instrument_id",
    "raw_symbol",
    "root_symbol",
    "ts_open_ns",
    "ts_close_ns",
    "quantity",
    "direction",
    "entry_price",
    "exit_price",
    "gross_pnl_usd",
    "costs_usd",
    "net_pnl_usd",
    "close_reason",
)

#: The engine's closed set of close reasons.
CLOSE_REASONS: frozenset[str] = frozenset({"signal", "roll", "eot"})

#: Frozen column contract. Mirrors ``quant::kFillCsvHeader`` exactly.
FILL_COLUMNS: tuple[str, ...] = (
    "fill_index",
    "fill_id",
    "order_id",
    "ts_fill_ns",
    "instrument_id",
    "raw_symbol",
    "side",
    "quantity",
    "fill_price",
    "commission_usd",
    "slippage_ticks",
)


class ClosedTradeRecord(BaseModel):
    """One Fill-derived closed/reduced position event, verbatim from C++."""

    model_config = {"frozen": True, "extra": "forbid"}

    trade_index: int = Field(ge=0)
    instrument_id: int = Field(ge=0)
    raw_symbol: str
    root_symbol: str
    ts_open_ns: int = Field(gt=0)
    ts_close_ns: int = Field(gt=0)
    quantity: int = Field(gt=0)
    direction: int
    entry_price: float
    exit_price: float
    gross_pnl_usd: float
    costs_usd: float
    net_pnl_usd: float
    close_reason: str

    @field_validator("direction")
    @classmethod
    def _dir(cls, v: int) -> int:
        if v not in (-1, 1):
            raise ValueError(f"direction must be +1 (closed a long) or -1 (closed a short), got {v}")
        return v

    @field_validator("close_reason")
    @classmethod
    def _reason(cls, v: str) -> str:
        if v not in CLOSE_REASONS:
            raise ValueError(f"close_reason {v!r} is not one of {sorted(CLOSE_REASONS)}")
        return v


def read_closed_trades_csv(path: str | Path) -> tuple[ClosedTradeRecord, ...]:
    """Parse the additive C++ export. Fails loudly on any column drift.

    ``ts_close_ns >= ts_open_ns`` is checked here rather than in the model so the
    error names the offending trade index.
    """
    df = pd.read_csv(path)
    cols = tuple(str(c) for c in df.columns)
    if cols != CLOSED_TRADE_COLUMNS:
        raise ValueError(
            f"closed-trade export header {cols} != frozen contract {CLOSED_TRADE_COLUMNS}; "
            "the C++ quant::kClosedTradeCsvHeader and this reader must not drift apart"
        )
    out: list[ClosedTradeRecord] = []
    for rec in df.itertuples(index=False):
        t = ClosedTradeRecord(
            trade_index=int(rec.trade_index),
            instrument_id=int(rec.instrument_id),
            raw_symbol=str(rec.raw_symbol),
            root_symbol=str(rec.root_symbol),
            ts_open_ns=int(rec.ts_open_ns),
            ts_close_ns=int(rec.ts_close_ns),
            quantity=int(rec.quantity),
            direction=int(rec.direction),
            entry_price=float(rec.entry_price),
            exit_price=float(rec.exit_price),
            gross_pnl_usd=float(rec.gross_pnl_usd),
            costs_usd=float(rec.costs_usd),
            net_pnl_usd=float(rec.net_pnl_usd),
            close_reason=str(rec.close_reason),
        )
        if t.ts_close_ns < t.ts_open_ns:
            raise ValueError(
                f"closed trade {t.trade_index} closes ({t.ts_close_ns}) before it opens "
                f"({t.ts_open_ns})"
            )
        out.append(t)
    if any(a.trade_index >= b.trade_index for a, b in pairwise(out)):
        raise ValueError("closed-trade export trade_index must be strictly increasing")
    return tuple(out)


class FillRecord(BaseModel):
    """One Fill, verbatim from C++. Read for its commission, never for its price.

    The fill price is present for audit only. Phase 15 never uses it to compute a
    PnL -- that is what ``ClosedTrade.gross_pnl_usd`` is for, and it was priced by
    the engine.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    fill_index: int = Field(ge=0)
    fill_id: int = Field(ge=0)
    order_id: int = Field(ge=0)
    ts_fill_ns: int = Field(gt=0)
    instrument_id: int = Field(ge=0)
    raw_symbol: str
    side: str
    quantity: int = Field(gt=0)
    fill_price: float
    commission_usd: float = Field(ge=0.0)
    slippage_ticks: float

    @field_validator("side")
    @classmethod
    def _side(cls, v: str) -> str:
        if v not in ("buy", "sell"):
            raise ValueError(f"side {v!r} must be 'buy' or 'sell'")
        return v


def read_fills_csv(path: str | Path) -> tuple[FillRecord, ...]:
    """Parse the additive C++ fill export. Fails loudly on any column drift."""
    df = pd.read_csv(path)
    cols = tuple(str(c) for c in df.columns)
    if cols != FILL_COLUMNS:
        raise ValueError(
            f"fill export header {cols} != frozen contract {FILL_COLUMNS}; the C++ "
            "quant::kFillCsvHeader and this reader must not drift apart"
        )
    out = tuple(
        FillRecord(
            fill_index=int(rec.fill_index),
            fill_id=int(rec.fill_id),
            order_id=int(rec.order_id),
            ts_fill_ns=int(rec.ts_fill_ns),
            instrument_id=int(rec.instrument_id),
            raw_symbol=str(rec.raw_symbol),
            side=str(rec.side),
            quantity=int(rec.quantity),
            fill_price=float(rec.fill_price),
            commission_usd=float(rec.commission_usd),
            slippage_ticks=float(rec.slippage_ticks),
        )
        for rec in df.itertuples(index=False)
    )
    if any(a.fill_index >= b.fill_index for a, b in pairwise(out)):
        raise ValueError("fill export fill_index must be strictly increasing")
    return out
