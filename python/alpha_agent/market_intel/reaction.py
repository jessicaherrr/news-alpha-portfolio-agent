"""Deterministic OBSERVED MARKET REACTION -- Checkpoint G, Section 29. Pure,
no network: operates on already-fetched real OHLCV bars. Uses ONLY bars
STRICTLY AFTER the reference (news/event) timestamp -- no lookahead. Labeled
"OBSERVED MARKET REACTION" everywhere it renders, never "EVENT CAUSED THIS
MOVE" -- this module computes a correlation-free, purely descriptive
before/after comparison, never a causal claim.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel


class ObservedReaction(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "observed-market-reaction/1"
    reference_at: datetime
    baseline_price: float | None = None
    baseline_at: datetime | None = None
    n_bars_after: int = 0
    return_5m_pct: float | None = None
    return_30m_pct: float | None = None
    return_1h_pct: float | None = None
    volume_change_pct: float | None = None
    realized_volatility_change_pct: float | None = None
    #: Always explains WHY a field is None (insufficient data / bar
    #: granularity too coarse) -- never a silent blank.
    detail: str = ""


def _return_at_horizon(after_bars: list[dict[str, Any]], reference_at: datetime, baseline: float, horizon: timedelta) -> float | None:
    """Requires a REAL bar AT OR AFTER the target horizon -- never
    substitutes an earlier bar (e.g. a 5-minute-after close standing in for
    a "1 hour" return would misrepresent how much time has actually
    elapsed)."""
    target_time = reference_at + horizon
    candidates = [b for b in after_bars if b["ts_event"] >= target_time]
    if not candidates or not baseline:
        return None
    closest = min(candidates, key=lambda b: b["ts_event"])
    return (closest["close"] / baseline - 1.0) * 100.0


def compute_observed_reaction(bars: list[dict[str, Any]], reference_at: datetime) -> ObservedReaction:
    """`bars` need not be pre-sorted. Returns an honest, mostly-`None`
    result when there are no real bars on one or both sides of
    `reference_at` -- never a fabricated reaction."""
    ordered = sorted(bars, key=lambda b: b["ts_event"])
    before = [b for b in ordered if b["ts_event"] <= reference_at]
    after = [b for b in ordered if b["ts_event"] > reference_at]

    if not before:
        return ObservedReaction(reference_at=reference_at, detail="No real bar exists at/before the reference time.")
    if not after:
        return ObservedReaction(
            reference_at=reference_at, baseline_price=before[-1]["close"], baseline_at=before[-1]["ts_event"],
            detail="No real bar exists strictly after the reference time yet.",
        )

    baseline = before[-1]["close"]
    baseline_at = before[-1]["ts_event"]

    return_5m = _return_at_horizon(after, reference_at, baseline, timedelta(minutes=5))
    return_30m = _return_at_horizon(after, reference_at, baseline, timedelta(minutes=30))
    return_1h = _return_at_horizon(after, reference_at, baseline, timedelta(hours=1))

    volume_change = None
    before_vols = [b.get("volume") for b in before[-5:] if b.get("volume") is not None]
    after_vols = [b.get("volume") for b in after[:5] if b.get("volume") is not None]
    if before_vols and after_vols:
        avg_before = sum(before_vols) / len(before_vols)
        avg_after = sum(after_vols) / len(after_vols)
        if avg_before > 0:
            volume_change = (avg_after / avg_before - 1.0) * 100.0

    vol_change = None
    if len(before) >= 6 and len(after) >= 6:
        before_rets = [
            before[i]["close"] / before[i - 1]["close"] - 1.0
            for i in range(len(before) - 5, len(before))
            if before[i - 1]["close"]
        ]
        after_rets = [
            after[i]["close"] / after[i - 1]["close"] - 1.0
            for i in range(1, min(6, len(after)))
            if after[i - 1]["close"]
        ]
        if len(before_rets) >= 2 and len(after_rets) >= 2:
            import statistics

            stdev_before = statistics.pstdev(before_rets)
            stdev_after = statistics.pstdev(after_rets)
            if stdev_before > 0:
                vol_change = (stdev_after / stdev_before - 1.0) * 100.0

    detail_parts = [f"{len(after)} real bar(s) after reference."]
    if return_5m is None and return_30m is None and return_1h is None:
        detail_parts.append("Bar granularity too coarse (or too little elapsed time) for any horizon yet.")

    return ObservedReaction(
        reference_at=reference_at, baseline_price=baseline, baseline_at=baseline_at, n_bars_after=len(after),
        return_5m_pct=return_5m, return_30m_pct=return_30m, return_1h_pct=return_1h,
        volume_change_pct=volume_change, realized_volatility_change_pct=vol_change, detail=" ".join(detail_parts),
    )


__all__ = ["ObservedReaction", "compute_observed_reaction"]
