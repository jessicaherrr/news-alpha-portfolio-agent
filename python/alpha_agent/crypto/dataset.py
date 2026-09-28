"""Causal, point-in-time dataset assembly for the Phase 22 crypto scaffold.

This module is the ONLY place a Phase 22 alternative-data row
(:mod:`alpha_agent.crypto.schemas`) turns into a numeric feature column, and it
is the single causal gate: a value with ``available_ts_ns > T`` is NEVER
visible to a bar timestamped ``T``, full stop, regardless of how informative
or how close its ``ts_event_ns`` is.

This is a point-in-time AS-OF join (`pandas.merge_asof`, ``direction="backward"``),
not a market-data forward-fill. The two are easy to conflate and mean opposite
things:

* CLAUDE.md's "no silent forward-filling across missing market data" forbids
  papering over a GAP in a market-data series (an OHLCV bar that should exist
  but doesn't) by inventing a value.
* An as-of join over a slowly-updating external series is not filling a gap --
  it is the correct point-in-time semantics for "what value was actually known
  at T": a daily on-chain metric genuinely does not change between
  publications, so its value "as of" any bar before the next publication IS
  its last published reading. Before the FIRST publication, this module
  returns an explicit ``NaN`` -- it never invents a value, imputes zero, or
  reaches backward past ``available_ts_ns``.

Every aggregate this module builds (liquidations, exchange flows) is
`available` only once its whole bucket period has elapsed PLUS a publication
delay -- never at the start of the period it summarizes (prompt 22: "a full-day
liquidation, flow, or on-chain value cannot be treated as available at the
beginning of that same day unless the source contract explicitly says so"; no
Phase 22 synthetic source contract says so).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_agent.crypto.schemas import ExchangeFlowBar, LiquidationEvent


@dataclass(frozen=True)
class DailySeries:
    """One alt-data series ready for :func:`causal_as_of_join`: parallel arrays
    of (event timestamp, availability timestamp, value). Not required to be
    one-per-day despite the name -- :func:`causal_as_of_join` only cares that
    each row is a (ts_event_ns, available_ts_ns, value) triple."""

    ts_event_ns: np.ndarray
    available_ts_ns: np.ndarray
    value: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.ts_event_ns)
        if len(self.available_ts_ns) != n or len(self.value) != n:
            raise ValueError("ts_event_ns / available_ts_ns / value must be the same length")
        if np.any(np.asarray(self.available_ts_ns) < np.asarray(self.ts_event_ns)):
            raise ValueError("a DailySeries row has available_ts_ns < ts_event_ns")


def causal_as_of_join(
    bar_ts_ns: Sequence[int] | np.ndarray,
    *,
    obs_ts_event_ns: Sequence[int] | np.ndarray,
    obs_available_ts_ns: Sequence[int] | np.ndarray,
    obs_value: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """For every timestamp in ``bar_ts_ns``, return the value of the
    observation with the largest ``available_ts_ns <= T`` (ties broken by the
    largest ``ts_event_ns``), or ``NaN`` if no observation is available yet as
    of ``T``.

    This is the ONLY causal gate: an observation is a candidate for bar ``T``
    if and only if ``available_ts_ns <= T``. Nothing here ever looks at
    ``ts_event_ns`` to decide visibility -- only to break a same-``available_ts_ns``
    tie -- so a delayed publication can never leak into an earlier bar no
    matter how far in the past its ``ts_event_ns`` is, and a future
    observation (whatever its ``ts_event_ns``) can never influence a bar
    before its own ``available_ts_ns``.
    """
    bar_arr = np.asarray(bar_ts_ns, dtype="int64")
    if len(obs_value) == 0:
        return np.full(bar_arr.shape, np.nan, dtype="float64")

    obs = pd.DataFrame(
        {
            "available_ts_ns": np.asarray(obs_available_ts_ns, dtype="int64"),
            "ts_event_ns": np.asarray(obs_ts_event_ns, dtype="int64"),
            "value": np.asarray(obs_value, dtype="float64"),
        }
    ).sort_values(["available_ts_ns", "ts_event_ns"], kind="stable")

    bars = pd.DataFrame({"bar_ts_ns": bar_arr}).sort_values("bar_ts_ns", kind="stable")
    joined = pd.merge_asof(
        bars, obs, left_on="bar_ts_ns", right_on="available_ts_ns",
        direction="backward", allow_exact_matches=True,
    )
    # merge_asof requires the left frame sorted; restore the caller's original
    # bar order (bar timestamps are unique in every Phase 22 caller, matching
    # the same no-duplicate-ts_event_ns invariant `SourceSeries` enforces).
    return joined.set_index("bar_ts_ns").reindex(bar_arr)["value"].to_numpy()


def series_from_alt_data_rows(rows: Sequence, *, value_fn) -> DailySeries:
    """Build a :class:`DailySeries` directly from a list of
    :class:`~alpha_agent.crypto.schemas._AltDataRecord` rows (funding / basis /
    on-chain), reading each row's own ``ts_event_ns`` / ``available_ts_ns``."""
    ts = np.array([r.ts_event_ns for r in rows], dtype="int64")
    avail = np.array([r.available_ts_ns for r in rows], dtype="int64")
    val = np.array([value_fn(r) for r in rows], dtype="float64")
    return DailySeries(ts, avail, val)


def aggregate_daily_liquidations(
    events: Sequence[LiquidationEvent], *, start_ns: int, day_ns: int, n_days: int,
    publication_delay_ns: int,
) -> dict[str, DailySeries]:
    """Bucket individual liquidation events into UTC calendar days (by
    ``ts_event_ns``) and sum notional by side. A day's aggregate is not
    available until the whole day has elapsed PLUS ``publication_delay_ns`` --
    never at that day's own start, and never before its last contributing
    event's own timestamp."""
    long_usd = np.zeros(n_days)
    short_usd = np.zeros(n_days)
    for e in events:
        day = int((e.ts_event_ns - start_ns) // day_ns)
        if not (0 <= day < n_days):
            continue
        if e.side_liquidated == "LONG":
            long_usd[day] += e.notional_usd
        else:
            short_usd[day] += e.notional_usd
    day_start = start_ns + np.arange(n_days, dtype="int64") * day_ns
    day_end = day_start + day_ns
    available = day_end + publication_delay_ns
    imbalance = long_usd - short_usd
    return {
        "long_liquidation_usd": DailySeries(day_start, available, long_usd),
        "short_liquidation_usd": DailySeries(day_start, available, short_usd),
        "liquidation_imbalance": DailySeries(day_start, available, imbalance),
    }


def daily_series_from_exchange_flow_bars(bars: Sequence[ExchangeFlowBar]) -> dict[str, DailySeries]:
    """Each :class:`~alpha_agent.crypto.schemas.ExchangeFlowBar` already
    represents one day; this just re-exposes inflow / outflow / netflow
    uniformly as :class:`DailySeries`, using each bar's own
    ``available_ts_ns`` (already set past its own day's close by the
    synthetic generator -- never recomputed here)."""
    ts = np.array([b.ts_event_ns for b in bars], dtype="int64")
    avail = np.array([b.available_ts_ns for b in bars], dtype="int64")
    inflow = np.array([b.inflow_usd for b in bars], dtype="float64")
    outflow = np.array([b.outflow_usd for b in bars], dtype="float64")
    netflow = inflow - outflow
    return {
        "exchange_inflow_usd": DailySeries(ts, avail, inflow),
        "exchange_outflow_usd": DailySeries(ts, avail, outflow),
        "exchange_netflow_usd": DailySeries(ts, avail, netflow),
    }


def build_causal_feature_columns(
    bar_ts_ns: Sequence[int] | np.ndarray, series_by_column: dict[str, DailySeries],
) -> pd.DataFrame:
    """Join every named :class:`DailySeries` onto ``bar_ts_ns`` independently
    (one :func:`causal_as_of_join` call per column -- never a single multi-
    column join, which would risk a tie in one series silently picking a row
    from a different series). Returns one DataFrame column per series name,
    aligned to ``bar_ts_ns``."""
    bar_arr = np.asarray(bar_ts_ns, dtype="int64")
    out: dict[str, np.ndarray] = {}
    for name, s in series_by_column.items():
        out[name] = causal_as_of_join(
            bar_arr, obs_ts_event_ns=s.ts_event_ns, obs_available_ts_ns=s.available_ts_ns,
            obs_value=s.value,
        )
    return pd.DataFrame(out, index=pd.RangeIndex(len(bar_arr)))
