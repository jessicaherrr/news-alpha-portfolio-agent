"""Layer 4 -- research-only additive back-adjustment.

RESEARCH ONLY. The output carries **no** ``instrument_id`` / ``raw_symbol`` --
the adjusted price never traded and must never reach the execution path
(``price_domain = BACK_ADJUSTED``; the C++ ``make_fill`` rejects that domain).

Additive back-adjustment: at each roll ``gap = to_price - from_price``. A bar at
timestamp ``T`` is shifted by the sum of the gaps of every roll that takes effect
*after* ``T``; the newest contract is left unadjusted.

Modes:

* ``retrospective_research`` -- offline full-history study; uses every roll
  (adjustment at ``T`` = sum of gaps of rolls effective strictly AFTER ``T``).
* ``point_in_time`` -- walk-forward safe; given ``as_of_ts_ns`` only rolls with
  ``effective_ts_ns <= as_of_ts_ns`` are known, and the series stops at
  ``as_of_ts_ns``. A feature computed at T <= as_of can never see a later roll.
* ``forward_adjusted`` (Phase 13.5C) -- fully causal, no ``as_of`` truncation.
  The FIRST contract segment is left unadjusted; adjustment at ``T`` =
  ``-sum(gap for rolls with effective_ts_ns <= T)``. A bar at ``T`` therefore
  depends only on rolls that have already happened by ``T`` -- so a feature
  computed at ``T`` over the whole span never uses a future roll basis, and the
  series has no roll-jump artifact. Research signal only; never executable.
"""
from __future__ import annotations

from enum import Enum

import pandas as pd

from alpha_agent.data.diagnostics import DiagnosticKind, DiagnosticsReport, PipelineError
from alpha_agent.schemas.market_data import BACKADJUSTED_BAR_COLUMNS, RollEvent

_PRICE_COLS = ("open", "high", "low", "close")
_EXECUTABLE_IDENTITY = ("instrument_id", "raw_symbol", "active_instrument_id", "active_raw_symbol")


class AdjustmentMode(str, Enum):
    RETROSPECTIVE_RESEARCH = "retrospective_research"
    POINT_IN_TIME = "point_in_time"
    FORWARD_ADJUSTED = "forward_adjusted"


def build_back_adjusted_series(
    continuous_bars: pd.DataFrame,
    rolls: list[RollEvent],
    *,
    continuous_symbol: str,
    mode: AdjustmentMode = AdjustmentMode.RETROSPECTIVE_RESEARCH,
    as_of_ts_ns: int | None = None,
) -> tuple[pd.DataFrame, DiagnosticsReport]:
    """Continuous (layer 3) bars + roll map -> back-adjusted rows in
    ``BACKADJUSTED_BAR_COLUMNS`` order."""
    report = DiagnosticsReport(n_input_rows=len(continuous_bars))
    need = {"ts_event_ns", *_PRICE_COLS, "volume"}
    if need - set(continuous_bars.columns):
        raise ValueError(f"continuous bars missing columns: {sorted(need - set(continuous_bars.columns))}")

    b = continuous_bars.sort_values("ts_event_ns", kind="stable").reset_index(drop=True)
    b["ts_event_ns"] = b["ts_event_ns"].astype("int64")

    if mode is AdjustmentMode.POINT_IN_TIME:
        if as_of_ts_ns is None:
            raise ValueError("point_in_time mode requires as_of_ts_ns")
        as_of = int(as_of_ts_ns)
        b = b[b["ts_event_ns"] <= as_of].reset_index(drop=True)
        usable_rolls = [r for r in rolls if int(r.effective_ts_ns) <= as_of]
        adjusted_through = as_of
    else:
        as_of = None
        usable_rolls = list(rolls)
        adjusted_through = int(b["ts_event_ns"].max()) if len(b) else 0

    # rolls missing a reference gap cannot be applied -> loud
    bad = [r for r in usable_rolls if r.additive_gap is None]
    if bad:
        report.add(DiagnosticKind.MISSING_ALIGNED_ROLL_REFERENCE_PRICE,
                   f"{len(bad)} roll(s) have no additive_gap; back-adjustment cannot proceed",
                   count=len(bad))
        raise PipelineError("back-adjustment needs a gap for every roll", report)

    roll_ts = sorted((int(r.effective_ts_ns), float(r.additive_gap)) for r in usable_rolls)
    ts = b["ts_event_ns"].to_numpy()
    adj = pd.Series(0.0, index=b.index)
    if mode is AdjustmentMode.FORWARD_ADJUSTED:
        # CAUSAL: adjustment(T) = -sum(gap for rolls effective at or before T).
        # The first segment (before any roll) is unadjusted; every later bar is
        # shifted so its price is continuous with the first segment, using ONLY
        # rolls that have already occurred -- no future roll basis touches T.
        for rt, gap in roll_ts:
            adj = adj - (ts >= rt) * gap
    else:
        # adjustment(T) = sum of gaps of rolls effective strictly after T
        for rt, gap in roll_ts:
            adj = adj + (ts < rt) * gap

    out = pd.DataFrame({
        "ts_event_ns": b["ts_event_ns"],
        "continuous_symbol": continuous_symbol,
        "open": b["open"].astype("float64") + adj,
        "high": b["high"].astype("float64") + adj,
        "low": b["low"].astype("float64") + adj,
        "close": b["close"].astype("float64") + adj,
        "volume": b["volume"].astype("int64"),
        "adjustment_method": "additive",
        "adjustment_mode": mode.value,
        "cumulative_adjustment": adj.round(9),
        "adjusted_through_ts_ns": adjusted_through,
    }, columns=list(BACKADJUSTED_BAR_COLUMNS)).reset_index(drop=True)

    assert_no_executable_identity(out, report)
    report.n_output_rows = len(out)
    return out, report


def build_forward_adjusted_series(
    continuous_bars: pd.DataFrame,
    rolls: list[RollEvent],
    *,
    continuous_symbol: str,
) -> tuple[pd.DataFrame, DiagnosticsReport]:
    """Fully-causal forward-adjusted continuous series (Phase 13.5C signal path).

    Convenience wrapper over :func:`build_back_adjusted_series` with
    ``mode = FORWARD_ADJUSTED``: the first contract segment is unadjusted and a
    bar at ``T`` reflects only rolls with ``effective_ts_ns <= T``. Removes the
    roll-jump artifact of the raw continuous series without ever using a future
    roll basis. Research signal only -- carries no executable identity and is
    rejected by the execution guard.
    """
    return build_back_adjusted_series(
        continuous_bars, rolls, continuous_symbol=continuous_symbol,
        mode=AdjustmentMode.FORWARD_ADJUSTED,
    )


def assert_no_executable_identity(frame: pd.DataFrame, report: DiagnosticsReport) -> None:
    """A back-adjusted frame must never carry a fillable contract identity."""
    leaked = [c for c in _EXECUTABLE_IDENTITY if c in frame.columns]
    if leaked:
        report.add(DiagnosticKind.ADJUSTED_SERIES_HAS_EXECUTABLE_IDENTITY,
                   f"back-adjusted frame carries executable identity columns: {leaked}")
        raise PipelineError("back-adjusted series carries executable contract identity", report)


def reject_adjusted_execution(price_domain: str, report: DiagnosticsReport | None = None) -> None:
    """Guard for any code that would route a price into the execution path."""
    if str(price_domain) in ("back_adjusted", "PriceDomain.BACK_ADJUSTED",
                             "raw_continuous", "PriceDomain.RAW_CONTINUOUS"):
        if report is not None:
            report.add(DiagnosticKind.ADJUSTED_EXECUTION_ATTEMPT,
                       f"attempt to execute against price_domain={price_domain}")
        raise PipelineError(
            f"price_domain={price_domain} may not enter the execution path", report or DiagnosticsReport()
        )
