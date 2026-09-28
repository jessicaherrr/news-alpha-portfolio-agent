"""Layer 6 -- roll map.

A roll is detected purely from the **``instrument_id`` transition** of the
ordered continuous-series bars. Contract-month strings are never used.

Roll reference-price policy (Phase 04.5 decision C):

  1. ``same_timestamp_close_close``  -- preferred: gap = new.close[t] - old.close[t]
     at a timestamp ``t`` where BOTH real raw contracts have a bar.
  2. ``same_timestamp_open_open``    -- same, using opens.
  3. ``prev_close_new_open``         -- explicitly-marked FALLBACK only:
     old-contract close then new-contract open at ADJACENT timestamps. This gap
     mixes contract basis with genuine market movement (see FUTURES_HISTORY.md
     section 7), so it is warned + flagged (``used_fallback=True``), never
     silently treated as an aligned basis.

The aligned policies need ``overlap_bars`` -- contemporaneous bars for both raw
contracts around the transition (Stage B of the roll validation).
"""
from __future__ import annotations

from enum import Enum

import pandas as pd

from alpha_agent.data.contract_lifecycle import tradable_until_ns
from alpha_agent.data.definitions import DefinitionRegistry
from alpha_agent.data.diagnostics import DiagnosticKind, DiagnosticsReport, PipelineError, Severity
from alpha_agent.data.price_domain import assert_same_price_domain
from alpha_agent.schemas.market_data import (
    ROLL_EVENT_COLUMNS,
    RollEvent,
    is_normalized_price,
)


class RollPricePolicy(str, Enum):
    SAME_TIMESTAMP_CLOSE_CLOSE = "same_timestamp_close_close"
    SAME_TIMESTAMP_OPEN_OPEN = "same_timestamp_open_open"
    PREV_CLOSE_NEW_OPEN = "prev_close_new_open"       # explicit fallback
    PREV_CLOSE_NEW_CLOSE = "prev_close_new_close"     # legacy, adjacent-timestamp


_ALIGNED = {RollPricePolicy.SAME_TIMESTAMP_CLOSE_CLOSE, RollPricePolicy.SAME_TIMESTAMP_OPEN_OPEN}


def _fail(report: DiagnosticsReport, msg: str) -> None:
    raise PipelineError(msg, report)


def _aligned_gap(
    overlap: pd.DataFrame, from_id: int, to_id: int, near_ts: int, field: str
) -> tuple[int, float, float] | None:
    """Return (t, from_price, to_price) at the aligned timestamp closest to
    ``near_ts`` (preferring t <= near_ts) where both contracts have a bar."""
    if overlap is None or overlap.empty:
        return None
    o = overlap[overlap["instrument_id"].isin((from_id, to_id))].copy()
    o["ts_event_ns"] = o["ts_event_ns"].astype("int64")
    piv = o.pivot_table(index="ts_event_ns", columns="instrument_id", values=field, aggfunc="last")
    if from_id not in piv.columns or to_id not in piv.columns:
        return None
    both = piv.dropna(subset=[from_id, to_id])
    if both.empty:
        return None
    at_or_before = both[both.index <= near_ts]
    row = at_or_before.iloc[-1] if not at_or_before.empty else both.iloc[
        (both.index - near_ts).abs().argmin()
    ]
    t = int(row.name)
    fp, tp = float(row[from_id]), float(row[to_id])
    # Roll-basis reference prices may be zero or negative (historical CL); reject
    # only non-finite / un-normalized values. The additive gap tp - fp is well
    # defined across a sign change.
    if not (is_normalized_price(fp) and is_normalized_price(tp)):
        return None
    return t, fp, tp


def build_roll_events(
    bars: pd.DataFrame,
    *,
    continuous_symbol: str,
    registry: DefinitionRegistry,
    rule: str = "instrument_id_transition",
    price_policy: RollPricePolicy = RollPricePolicy.SAME_TIMESTAMP_CLOSE_CLOSE,
    overlap_bars: pd.DataFrame | None = None,
) -> tuple[list[RollEvent], DiagnosticsReport]:
    """Build the roll map for one continuous symbol from its ordered canonical
    bars (``ts_event_ns, instrument_id, open, close``). ``overlap_bars`` (same
    columns) supplies contemporaneous raw-contract bars for the aligned policies.
    """
    report = DiagnosticsReport(n_input_rows=len(bars))
    need = {"ts_event_ns", "instrument_id", "open", "close"}
    if need - set(bars.columns):
        raise ValueError(f"bars missing columns: {sorted(need - set(bars.columns))}")

    b = bars.reset_index(drop=True).copy()
    b["ts_event_ns"] = b["ts_event_ns"].astype("int64")
    b["instrument_id"] = b["instrument_id"].astype("int64")

    if not b["ts_event_ns"].is_monotonic_increasing:
        report.add(DiagnosticKind.BACKWARD_ROLL_TIMESTAMP,
                   "continuous feed timestamps are not monotonic non-decreasing")
        _fail(report, "non-monotonic timestamps in the continuous feed")

    dup_ts = b[b.duplicated("ts_event_ns", keep=False)]
    if not dup_ts.empty and dup_ts.groupby("ts_event_ns")["instrument_id"].nunique().gt(1).any():
        report.add(DiagnosticKind.OVERLAPPING_ACTIVE_CONTRACTS,
                   "two instrument_ids share a ts_event_ns in the continuous feed",
                   count=len(dup_ts))
        _fail(report, "overlapping active contracts in the continuous feed")

    # overlap bars must be in the SAME normalized price domain as the feed --
    # raw DBN fixed-point (~1e13) vs normalized points (~1e4) would corrupt the gap.
    if overlap_bars is not None and not overlap_bars.empty:
        assert_same_price_domain(b, overlap_bars, label="continuous vs overlap_bars")

    change = b["instrument_id"].ne(b["instrument_id"].shift())
    boundaries = [i for i in b.index[change] if i > 0]

    rolls: list[RollEvent] = []
    prev_effective = -1
    for i in boundaries:
        from_id = int(b.at[i - 1, "instrument_id"])
        to_id = int(b.at[i, "instrument_id"])
        effective = int(b.at[i, "ts_event_ns"])

        if from_id == to_id:
            report.add(DiagnosticKind.ROLL_SAME_INSTRUMENT, f"roll {from_id}->{to_id}")
            _fail(report, "roll with same instrument")
        from_spec = registry.get(from_id)
        to_spec = registry.get(to_id)
        if from_spec is None or to_spec is None:
            report.add(DiagnosticKind.UNKNOWN_INSTRUMENT_DURING_ROLL,
                       f"roll references unknown instrument_id(s): "
                       f"{[x for x, s in ((from_id, from_spec), (to_id, to_spec)) if s is None]}")
            _fail(report, "unknown instrument during roll")
        if from_spec.root_symbol != to_spec.root_symbol:
            report.add(DiagnosticKind.ROLL_ROOT_MISMATCH,
                       f"roll {from_spec.raw_symbol}({from_spec.root_symbol}) -> "
                       f"{to_spec.raw_symbol}({to_spec.root_symbol})")
            _fail(report, "roll crosses roots")
        if effective <= prev_effective:
            report.add(DiagnosticKind.BACKWARD_ROLL_TIMESTAMP,
                       f"roll at {effective} <= previous roll at {prev_effective}")
            _fail(report, "backward roll timestamp")
        prev_effective = effective

        from_ts = to_ts = None
        from_price = to_price = gap = None
        used_fallback = False
        effective_policy = price_policy

        if price_policy in _ALIGNED:
            field = "close" if price_policy is RollPricePolicy.SAME_TIMESTAMP_CLOSE_CLOSE else "open"
            hit = _aligned_gap(overlap_bars, from_id, to_id, effective, field)
            if hit is not None:
                from_ts = to_ts = hit[0]
                from_price, to_price = hit[1], hit[2]
                gap = round(to_price - from_price, 9)
            else:
                used_fallback = True
                effective_policy = RollPricePolicy.PREV_CLOSE_NEW_OPEN
                report.add(
                    DiagnosticKind.ROLL_BASIS_FALLBACK_USED,
                    f"{from_spec.raw_symbol}->{to_spec.raw_symbol}: no aligned overlap bar for "
                    f"'{price_policy.value}'; fell back to prev_close_new_open. This gap mixes "
                    f"contract basis with any market move between the two bars.",
                    severity=Severity.WARNING, instrument_id=to_id, ts_event_ns=effective,
                )

        if from_price is None:  # explicit prev_close_new_open / prev_close_new_close, or fallback
            from_ts = int(b.at[i - 1, "ts_event_ns"])
            to_ts = effective
            fp = float(b.at[i - 1, "close"])
            tp = (float(b.at[i, "close"])
                  if effective_policy is RollPricePolicy.PREV_CLOSE_NEW_CLOSE
                  else float(b.at[i, "open"]))
            if pd.notna(fp) and pd.notna(tp) and is_normalized_price(fp) and is_normalized_price(tp):
                from_price, to_price, gap = fp, tp, round(tp - fp, 9)
            else:
                report.add(DiagnosticKind.MISSING_ALIGNED_ROLL_REFERENCE_PRICE,
                           f"roll {from_spec.raw_symbol}->{to_spec.raw_symbol} at {effective}",
                           severity=Severity.WARNING, instrument_id=to_id)

        for spec, ts, iid in ((from_spec, from_ts, from_id), (to_spec, to_ts, to_id)):
            if ts is not None and not (spec.activation_ns <= ts <= tradable_until_ns(spec)):
                report.add(DiagnosticKind.CONTRACT_ROLLED_PAST_TRADABLE_WINDOW,
                           f"{spec.raw_symbol} reference bar {ts} outside its tradable window",
                           severity=Severity.WARNING, instrument_id=iid)

        rolls.append(RollEvent(
            continuous_symbol=continuous_symbol, effective_ts_ns=effective,
            from_instrument_id=from_id, to_instrument_id=to_id,
            from_raw_symbol=from_spec.raw_symbol, to_raw_symbol=to_spec.raw_symbol,
            rule=rule, price_policy=effective_policy.value, used_fallback=used_fallback,
            from_ts_ns=from_ts, to_ts_ns=to_ts,
            from_price=from_price, to_price=to_price, additive_gap=gap,
        ))

    seen: set[tuple] = set()
    for r in rolls:
        key = (r.effective_ts_ns, r.from_instrument_id, r.to_instrument_id)
        if key in seen:
            report.add(DiagnosticKind.DUPLICATE_ROLL, f"{key}")
            _fail(report, "duplicate roll")
        seen.add(key)

    report.n_output_rows = len(rolls)
    return rolls, report


def rolls_frame(rolls: list[RollEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [{k: getattr(r, k) for k in ROLL_EVENT_COLUMNS} for r in rolls],
        columns=list(ROLL_EVENT_COLUMNS),
    )
