"""Layer 3 -- unadjusted continuous series.

A relabelling of the canonical raw-contract bars for one continuous symbol.
Prices are the **actual raw prices** of whichever contract was active; the roll
gap is preserved, never smoothed. ``price_domain = RAW_CONTINUOUS``.
"""
from __future__ import annotations

import pandas as pd

from alpha_agent.data.definitions import DefinitionRegistry
from alpha_agent.data.diagnostics import DiagnosticKind, DiagnosticsReport, PipelineError
from alpha_agent.schemas.market_data import CONTINUOUS_BAR_COLUMNS, RollEvent

_SRC = ("ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume")


def build_continuous_series(
    canonical_bars: pd.DataFrame,
    *,
    continuous_symbol: str,
    registry: DefinitionRegistry,
    rolls: list[RollEvent] | None = None,
) -> tuple[pd.DataFrame, DiagnosticsReport]:
    """Ordered canonical bars (one continuous symbol's history) -> ContinuousBar
    rows in ``CONTINUOUS_BAR_COLUMNS`` order. Prices unchanged."""
    missing = [c for c in _SRC if c not in canonical_bars.columns]
    if missing:
        raise ValueError(f"canonical bars missing columns: {missing}")

    report = DiagnosticsReport(n_input_rows=len(canonical_bars))
    b = canonical_bars.sort_values("ts_event_ns", kind="stable").reset_index(drop=True)
    b["ts_event_ns"] = b["ts_event_ns"].astype("int64")
    b["instrument_id"] = b["instrument_id"].astype("int64")

    # overlapping active contracts
    if b.duplicated("ts_event_ns", keep=False).any():
        conflict = b.groupby("ts_event_ns")["instrument_id"].nunique().gt(1)
        if conflict.any():
            report.add(DiagnosticKind.OVERLAPPING_ACTIVE_CONTRACTS,
                       f"{int(conflict.sum())} ts_event_ns have >1 active instrument_id")
            raise PipelineError("overlapping active contracts", report)

    specs = {s.instrument_id: s for s in registry.specs()}
    unknown = ~b["instrument_id"].isin(specs)
    if unknown.any():
        report.add(DiagnosticKind.CONTINUOUS_MISSING_ACTIVE_CONTRACT,
                   f"{int(unknown.sum())} bars reference an instrument_id not in the registry",
                   instrument_ids=sorted({int(i) for i in b.loc[unknown, "instrument_id"].unique()}))
        raise PipelineError("continuous series references an unknown active contract", report)

    roll_ts = {int(r.effective_ts_ns) for r in (rolls or [])}
    # a bar is a roll boundary if its instrument_id differs from the previous bar
    is_boundary = b["instrument_id"].ne(b["instrument_id"].shift()).fillna(False)
    is_boundary.iloc[0] = False
    if roll_ts:
        is_boundary = is_boundary | b["ts_event_ns"].isin(roll_ts)

    out = pd.DataFrame({
        "ts_event_ns": b["ts_event_ns"],
        "continuous_symbol": continuous_symbol,
        "active_instrument_id": b["instrument_id"],
        "active_raw_symbol": b["instrument_id"].map(lambda i: specs[i].raw_symbol),
        "open": b["open"].astype("float64"),
        "high": b["high"].astype("float64"),
        "low": b["low"].astype("float64"),
        "close": b["close"].astype("float64"),
        "volume": b["volume"].astype("int64"),
        "trading_day": b["trading_day"] if "trading_day" in b.columns else pd.NaT,
        "session": b["session"].astype(str) if "session" in b.columns else "",
        "is_roll_boundary": is_boundary,
    }, columns=list(CONTINUOUS_BAR_COLUMNS)).reset_index(drop=True)

    report.n_output_rows = len(out)
    return out, report
