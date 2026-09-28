"""Deterministic multi-contract fixtures for Phase 04 (rolls / continuous /
back-adjustment). No network. Builds canonical-bar DataFrames + a registry."""
from __future__ import annotations

from datetime import date

import pandas as pd
from alpha_agent.data.definitions import DefinitionRegistry
from alpha_agent.schemas.market_data import ContractSpecModel

MIN_NS = 60_000_000_000
BASE_NS = pd.Timestamp("2026-09-01T14:00:00Z").value  # a plain weekday RTH slot

# Three NQ contracts for cumulative-roll tests.
NQU6 = ContractSpecModel(
    instrument_id=101, raw_symbol="NQU6", root_symbol="NQ", exchange="XCME",
    tick_size=0.25, multiplier=20.0,
    activation_ns=pd.Timestamp("2025-06-01T00:00:00Z").value,
    expiration_ns=pd.Timestamp("2026-09-18T13:30:00Z").value,
)
NQZ6 = ContractSpecModel(
    instrument_id=102, raw_symbol="NQZ6", root_symbol="NQ", exchange="XCME",
    tick_size=0.25, multiplier=20.0,
    activation_ns=pd.Timestamp("2025-09-01T00:00:00Z").value,
    expiration_ns=pd.Timestamp("2026-12-18T14:30:00Z").value,
)
NQH7 = ContractSpecModel(
    instrument_id=103, raw_symbol="NQH7", root_symbol="NQ", exchange="XCME",
    tick_size=0.25, multiplier=20.0,
    activation_ns=pd.Timestamp("2025-12-01T00:00:00Z").value,
    expiration_ns=pd.Timestamp("2027-03-19T13:30:00Z").value,
)
ESZ6 = ContractSpecModel(
    instrument_id=201, raw_symbol="ESZ6", root_symbol="ES", exchange="XCME",
    tick_size=0.25, multiplier=50.0,
    activation_ns=pd.Timestamp("2025-09-01T00:00:00Z").value,
    expiration_ns=pd.Timestamp("2026-12-18T14:30:00Z").value,
)


def registry(*specs: ContractSpecModel) -> DefinitionRegistry:
    return DefinitionRegistry(list(specs) or [NQU6, NQZ6, NQH7])


def _segment(spec: ContractSpecModel, *, start_ns: int, n: int, base_price: float, step: float):
    rows = []
    for i in range(n):
        o = base_price + i * step
        c = o + step
        rows.append({
            "ts_event_ns": start_ns + i * MIN_NS,
            "instrument_id": spec.instrument_id,
            "raw_symbol": spec.raw_symbol,
            "root_symbol": spec.root_symbol,
            "open": o, "high": max(o, c) + step, "low": min(o, c) - step, "close": c,
            "volume": 100 + i,
            "trading_day": date(2026, 9, 1),
            "session": "RTH",
        })
    return rows


def canonical_bars(segments: list[dict]) -> pd.DataFrame:
    """segments: [{spec, n, base_price, step}] laid end to end (one minute apart)."""
    rows: list[dict] = []
    ts = BASE_NS
    for seg in segments:
        s = _segment(seg["spec"], start_ns=ts, n=seg["n"],
                     base_price=seg["base_price"], step=seg.get("step", 0.25))
        rows.extend(s)
        ts = s[-1]["ts_event_ns"] + MIN_NS
    return pd.DataFrame(rows)


def two_contract_history(*, u6_close: float = 29000.0, z6_open: float = 29100.0):
    """NQU6 (5 bars, ending close ~= u6_close) then NQZ6 (5 bars, opening ~= z6_open).
    One roll; roll gap = z6_open - u6_close."""
    step = 0.0
    u6 = _segment(NQU6, start_ns=BASE_NS, n=5, base_price=u6_close, step=step)
    z6 = _segment(NQZ6, start_ns=u6[-1]["ts_event_ns"] + MIN_NS, n=5,
                  base_price=z6_open, step=step)
    return pd.DataFrame(u6 + z6)
