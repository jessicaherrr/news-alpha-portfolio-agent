"""Compute point-in-time features on a one-row-per-trading-day SIGNAL series.

A daily ``trading_day`` bar series is **ordinal**: consecutive rows are
consecutive bars *by construction* (one row per exchange trading day, built by
:func:`alpha_agent.data.real_market_dataset.daily_signal_series`). Its
wall-clock ``ts_event_ns`` spacing, however, jumps every weekend and holiday --
a Friday->Monday step is ~3 days. Fed straight into the feature engine that
looks like a missing-bar hole: :meth:`SourceSeries.gap_breaks` fires and every
:class:`SessionPolicy.RESET_ON_GAP` feature (``cum_log_return`` / ``realized_vol``
/ ``vol_percentile`` / ...) is reset into ~one-trading-week segments, so any
window longer than ~5 rows never reaches ``min_obs`` and the column is entirely
missing.

The fix -- the frozen Phase 13.5C baseline-signal technique
(``validation.phase_13_5c_matrix``): compute the features on a SYNTHETIC,
gap-free ``1-per-day`` index and realign the result to the real trading-day
timestamps. Row order, one-row-per-trading-day cardinality and the strictly
backward-looking window semantics are all unchanged; only the spurious
weekend/holiday "gap" disappears. This does NOT reimplement any feature -- it is
the identical :func:`compute_features` engine on a corrected index.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_agent.features.compute import compute_features
from alpha_agent.features.frame import FeatureFrame
from alpha_agent.features.spec import FeatureSpec
from alpha_agent.schemas.market_data import PriceDomain

#: The synthetic bar spacing -- one calendar day, so ``interval_ns`` matches and
#: no ``gap_breaks`` ever fires on the contiguous index.
NS_PER_DAY = 86_400_000_000_000

_DEFAULT_OHLCV = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class ContiguousDailyFeatures:
    """The result of :func:`compute_contiguous_daily_features`.

    ``frame`` is a normal :class:`FeatureFrame` whose ``identifiers.ts_event_ns``
    and ``availability`` carry the REAL trading-day timestamps; the feature
    values are positionally aligned to ``real_ts_ns``.
    """

    real_ts_ns: np.ndarray          # int64, ascending, one per trading day
    synthetic_ts_ns: np.ndarray     # int64, (1..n) * NS_PER_DAY -- what features were computed on
    frame: FeatureFrame

    def synthetic_to_real(self) -> dict[int, int]:
        return {int(s): int(r) for s, r in zip(self.synthetic_ts_ns, self.real_ts_ns)}


def compute_contiguous_daily_features(
    daily: pd.DataFrame,
    specs: list[FeatureSpec],
    *,
    root_symbol: str,
    price_domain: PriceDomain,
    adjustment_mode: str | None = None,
    require_point_in_time: bool = True,
    as_of_ts_ns: int | None = None,
    ohlcv_columns: tuple[str, ...] = _DEFAULT_OHLCV,
) -> ContiguousDailyFeatures:
    """Compute ``specs`` on ``daily`` using a gap-free synthetic day index.

    ``daily`` must be sorted by ``ts_event_ns`` ascending with unique timestamps
    (the shape :func:`daily_signal_series` returns). Every row is treated as the
    next contiguous bar.
    """
    if "ts_event_ns" not in daily.columns:
        raise ValueError("daily frame needs a 'ts_event_ns' column")
    real_ts = pd.to_numeric(daily["ts_event_ns"], errors="raise").to_numpy("int64")
    if real_ts.size and (np.diff(real_ts) <= 0).any():
        raise ValueError("daily frame must be sorted ascending with unique ts_event_ns")
    missing = [c for c in ohlcv_columns if c not in daily.columns]
    if missing:
        raise ValueError(f"daily frame is missing OHLCV column(s) {missing}")

    n = len(daily)
    synth = (np.arange(n, dtype="int64") + 1) * NS_PER_DAY
    frame = daily.assign(ts_event_ns=synth)[["ts_event_ns", *ohlcv_columns]].reset_index(drop=True)

    # local import: the feature engine core stays independent of this helper's callers
    from alpha_agent.features.source import SourceSeries

    src = SourceSeries(
        frame=frame,
        price_domain=price_domain,
        adjustment_mode=adjustment_mode,
        identity={"root_symbol": root_symbol},
        interval_ns=NS_PER_DAY,
    )
    feats = compute_features(
        src, list(specs),
        require_point_in_time=require_point_in_time,
        as_of_ts_ns=as_of_ts_ns,
    )

    # -- realign identifiers + availability to the REAL trading-day timestamps --
    ident = feats.identifiers.copy()
    ident["ts_event_ns"] = real_ts
    for k, v in ({"root_symbol": root_symbol}).items():
        if k not in ident.columns:
            ident[k] = v
    feats.identifiers = ident

    synth_to_real = {int(s): int(r) for s, r in zip(synth, real_ts)}
    if "first_available_ts_ns" in feats.availability.columns:
        feats.availability = feats.availability.assign(
            first_available_ts_ns=feats.availability["first_available_ts_ns"].map(
                lambda x: synth_to_real.get(int(x)) if pd.notna(x) else x
            )
        )

    return ContiguousDailyFeatures(real_ts_ns=real_ts, synthetic_ts_ns=synth, frame=feats)
