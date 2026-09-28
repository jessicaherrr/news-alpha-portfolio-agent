"""Backward-looking, segment-aware window primitives.

Every function here is strictly point-in-time: the value at row ``i`` uses only
rows ``<= i`` and only rows in the same segment. No centered windows, no
``expanding`` over the full sample, no forward fill.

``seg`` is the per-row segment id from :meth:`SourceSeries.segment_ids`.
``min_obs`` maps to pandas ``min_periods`` -- rows before it is satisfied are
returned as ``NaN`` (explicit missing), never a partial value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def _regroup(result: pd.Series, like: pd.Series) -> pd.Series:
    """Drop the groupby level a ``groupby(...).rolling`` adds and realign."""
    if isinstance(result.index, pd.MultiIndex):
        result = result.reset_index(level=0, drop=True)
    return result.reindex(like.index).astype("float64")


def seg_shift(s: pd.Series, seg: pd.Series, n: int) -> pd.Series:
    return s.groupby(seg, sort=False).shift(n)


def seg_diff(s: pd.Series, seg: pd.Series, n: int = 1) -> pd.Series:
    """Arithmetic change over ``n`` bars within the segment. Signed-price safe
    (no division): valid for zero and negative prices."""
    return (s - seg_shift(s, seg, n)).astype("float64")


def seg_pct_change(s: pd.Series, seg: pd.Series, n: int = 1) -> tuple[pd.Series, pd.Series]:
    """Percentage change over ``n`` bars. Returns ``(value, base)``.

    The value is ``NaN`` where the base price is exactly ``0`` (undefined) --
    callers turn that into an explicit missing + a ZERO_DENOMINATOR issue.
    A negative base is left to the caller to flag (the arithmetic is defined
    but the sign convention is unusual).
    """
    base = seg_shift(s, seg, n)
    out = (s - base) / base
    out = out.where(base != 0.0)
    return out.astype("float64"), base


def seg_log_return(s: pd.Series, seg: pd.Series, n: int = 1) -> tuple[pd.Series, pd.Series]:
    """Log return over ``n`` bars. Returns ``(value, invalid_mask)`` where
    ``invalid_mask`` is True on rows where both endpoints were present but at
    least one was ``<= 0`` (so the log is undefined) -- the value is ``NaN``
    there."""
    base = seg_shift(s, seg, n)
    valid = (s > 0) & (base > 0)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[valid] = np.log(s[valid].to_numpy() / base[valid].to_numpy())
    invalid = base.notna() & s.notna() & ~valid
    return out, invalid


def seg_rolling(s: pd.Series, seg: pd.Series, window: int, min_obs: int, how: str) -> pd.Series:
    """Backward rolling reduction (``mean`` / ``std`` / ``sum`` / ``min`` / ``max``
    / ``var``) within the segment. ``std``/``var`` use ddof=1."""
    g = s.groupby(seg, sort=False).rolling(window, min_periods=min_obs)
    r = getattr(g, how)()
    return _regroup(r, s)


def seg_rolling_apply(s: pd.Series, seg: pd.Series, window: int, min_obs: int, fn) -> pd.Series:
    g = s.groupby(seg, sort=False).rolling(window, min_periods=min_obs)
    r = g.apply(fn, raw=True)
    return _regroup(r, s)


def seg_ewm_mean(s: pd.Series, seg: pd.Series, span: int, min_obs: int) -> pd.Series:
    parts = []
    for idx in s.groupby(seg, sort=False).groups.values():
        chunk = s.loc[idx]
        parts.append(chunk.ewm(span=span, adjust=False, min_periods=min_obs).mean())
    return pd.concat(parts).reindex(s.index).astype("float64")


def rolling_rank_pct(window_values: np.ndarray) -> float:
    """Point-in-time percentile rank of the *last* value within the trailing
    window (fraction of window observations <= the current value, current
    value included). Used for volatility percentile / rank features."""
    x = window_values[~np.isnan(window_values)]
    if x.size == 0:
        return np.nan
    cur = x[-1]
    return float(np.mean(x <= cur))


def safe_ratio(num: pd.Series, den: pd.Series, *, eps: float = _EPS) -> tuple[pd.Series, pd.Series]:
    """``num / den`` with ``|den| <= eps`` -> ``NaN``. Returns ``(value, zero_mask)``
    where ``zero_mask`` marks rows suppressed for a ~zero denominator (both
    endpoints were otherwise present)."""
    small = den.abs() <= eps
    out = (num / den).where(~small)
    zero_mask = small & num.notna() & den.notna()
    return out.astype("float64"), zero_mask


def true_range(high: pd.Series, low: pd.Series, close: pd.Series, seg: pd.Series) -> pd.Series:
    """max(high-low, |high-prev_close|, |low-prev_close|). On the first bar of a
    segment (no prev close) it degrades to ``high - low``. Signed-price safe."""
    prev_close = seg_shift(close, seg, 1)
    hl = (high - low)
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    tr = tr.where(prev_close.notna(), hl)
    return tr.astype("float64")
