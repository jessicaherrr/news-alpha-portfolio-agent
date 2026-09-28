"""Fractional (32nds) quote conversion for CBOT Treasury futures.

Treasuries are *quoted* in points and fractions of 1/32, but stored and computed
here as a **normalized decimal number of points** (e.g. 110.515625). This module
only converts between the two representations; the economics
(`contract_economics`) work on the normalized number regardless of convention.

Real ZN Databento field interpretation is verified in Phase 04.5 -- until then
these helpers are driven by the explicit `tick_fraction` a fixture supplies.
"""
from __future__ import annotations

from enum import Enum


class QuoteConvention(str, Enum):
    DECIMAL = "decimal"          # NQ, ES, CL, GC -- quoted directly in decimal
    FRACTIONAL_32 = "fractional_32"  # CBOT Treasuries -- points + 1/32 fractions


def display_to_decimal(handle: int, thirty_seconds: float, *, tick_fraction: int = 32) -> float:
    """``110`` + ``17`` (32nds) -> ``110.53125``. ``tick_fraction`` is the number
    of sub-units per point implied by the tick (32 for ZB, 64 for ZN half-ticks,
    128 for ZF/ZT quarter-ticks)."""
    if thirty_seconds < 0 or thirty_seconds >= tick_fraction:
        raise ValueError(f"fraction {thirty_seconds} out of range [0, {tick_fraction})")
    return handle + thirty_seconds / tick_fraction


def decimal_to_display(price: float, *, tick_fraction: int = 32) -> str:
    """``110.53125`` -> ``"110'170"`` (CME 32nds notation: handle, 32nds, then
    the fractional 32nd as a digit)."""
    handle = int(price)
    frac_pts = round((price - handle) * tick_fraction, 6)
    whole_32 = int(frac_pts)
    sub = frac_pts - whole_32
    # sub is in units of 1/tick_fraction of a point; express the remaining
    # fraction of a 32nd as an eighth-digit where it divides evenly.
    if tick_fraction == 32:
        return f"{handle}'{whole_32:02d}0"
    ratio = 32 / tick_fraction
    return f"{handle}'{int(whole_32 * ratio):02d}{round(sub * 8) if sub else 0}"


def normalized_tick_size(tick_fraction: int) -> float:
    """The tick in normalized points, e.g. 64 -> 0.015625 (half a 32nd)."""
    return 1.0 / tick_fraction
