"""The typed input to the feature engine and the point-in-time primitives.

A :class:`SourceSeries` wraps one instrument's (or one continuous symbol's)
ordered OHLCV bars together with the **price domain** the prices live in
(``raw_contract`` / ``raw_continuous`` / ``back_adjusted``) and -- for
``back_adjusted`` -- the Phase 04 adjustment **mode** (``point_in_time`` vs
``retrospective_research``). ``SourceSeries.safety()`` turns that into the three
explicit safety concepts (:mod:`alpha_agent.features.safety`).

All window logic is **backward-looking**. ``segment_ids`` implements the
session-aware / gap-aware reset behaviour; ``gap_breaks`` marks rows whose
preceding interval is larger than one bar so a "1-bar return" is never silently
computed across a multi-hour hole.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha_agent.features.enums import SessionPolicy
from alpha_agent.features.safety import SourceSafety
from alpha_agent.schemas.market_data import PriceDomain, is_normalized_price

# Phase 04 adjustment modes (mirrors alpha_agent.data.backadjust.AdjustmentMode --
# imported by value so the feature engine core stays independent of the data layer).
ADJUSTMENT_MODE_POINT_IN_TIME = "point_in_time"
ADJUSTMENT_MODE_RETROSPECTIVE = "retrospective_research"
# Phase 13.5C: forward-adjusted continuous -- adjustment at T uses ONLY rolls
# with effective_ts_ns <= T, so it is fully causal / signal-safe (like
# point_in_time, but with no as_of truncation). Research signal only; the frame
# still carries no executable identity and is never a Fill price.
ADJUSTMENT_MODE_FORWARD = "forward_adjusted"
_VALID_ADJUSTMENT_MODES = frozenset(
    {ADJUSTMENT_MODE_POINT_IN_TIME, ADJUSTMENT_MODE_RETROSPECTIVE, ADJUSTMENT_MODE_FORWARD}
)

OHLC = ("open", "high", "low", "close")
_ONE_MINUTE_NS = 60_000_000_000

# Identity columns that describe *which* series a row belongs to. They are kept
# separate from the numeric feature columns in a FeatureFrame and are never
# emitted as features.
IDENTITY_COLUMNS = (
    "instrument_id",
    "raw_symbol",
    "root_symbol",
    "continuous_symbol",
    "active_instrument_id",
    "active_raw_symbol",
    "trading_day",
    "session",
    "is_roll_boundary",
)

# Execution identity must not leak into research features: a feature computed on
# a continuous / back-adjusted series must never carry a fillable raw contract id
# as a *feature* value (identity columns above stay, as identifiers only).
EXECUTION_IDENTITY_COLUMNS = ("instrument_id", "raw_symbol")

# Only a real raw-contract (Futures) or raw share (ETF, Phase 6) price may ever
# be an execution reference / Fill price. SPLIT_ADJUSTED/TOTAL_RETURN/
# BACK_ADJUSTED are research-signal-only domains and must never appear here.
EXECUTION_PRICE_DOMAINS = frozenset({PriceDomain.RAW_CONTRACT, PriceDomain.RAW})


@dataclass(frozen=True)
class SourceSeries:
    frame: pd.DataFrame
    price_domain: PriceDomain
    identity: dict[str, str] = field(default_factory=dict)
    source_paths: tuple[str, ...] = ()
    interval_ns: int = _ONE_MINUTE_NS
    max_gap_intervals: float = 1.5
    # For price_domain == BACK_ADJUSTED: "point_in_time" | "retrospective_research".
    # None otherwise. Inferred from an 'adjustment_mode' frame column when absent.
    adjustment_mode: str | None = None
    # Set False when the lineage that produced this series is not causal
    # (e.g. a full-sample transform other than back-adjustment).
    causal: bool = True

    def __post_init__(self) -> None:
        f = self.frame
        if "ts_event_ns" not in f.columns:
            raise ValueError("SourceSeries.frame needs a 'ts_event_ns' column")
        ts = pd.to_numeric(f["ts_event_ns"], errors="raise").astype("int64")
        if not ts.is_monotonic_increasing:
            raise ValueError("SourceSeries.frame must be sorted by ts_event_ns ascending")
        if ts.duplicated().any():
            raise ValueError("SourceSeries.frame has duplicate ts_event_ns rows")
        if (ts <= 0).any():
            raise ValueError("ts_event_ns must be positive UTC-ns")
        for c in OHLC:
            if c in f.columns:
                v = pd.to_numeric(f[c], errors="coerce")
                bad = v.notna() & ~v.map(is_normalized_price)
                if bad.any():
                    raise ValueError(
                        f"column {c!r} has non-finite / un-normalized prices at "
                        f"{f.loc[bad, 'ts_event_ns'].tolist()[:5]}"
                    )
        # normalize a working copy: stable order, int64 ts, contiguous index
        object.__setattr__(self, "frame", f.reset_index(drop=True).assign(ts_event_ns=ts.values))

        # -- resolve the back-adjustment mode (loud, never assumed) --
        mode = self.adjustment_mode
        if self.price_domain is PriceDomain.BACK_ADJUSTED:
            if mode is None and "adjustment_mode" in f.columns:
                uniq = {str(x) for x in f["adjustment_mode"].dropna().unique()}
                if len(uniq) == 1:
                    mode = next(iter(uniq))
                elif len(uniq) > 1:
                    raise ValueError(
                        f"back-adjusted source mixes adjustment modes {sorted(uniq)}"
                    )
            if mode not in _VALID_ADJUSTMENT_MODES:
                raise ValueError(
                    "a back_adjusted SourceSeries must declare adjustment_mode "
                    f"({sorted(_VALID_ADJUSTMENT_MODES)}) explicitly or via an "
                    "'adjustment_mode' column; got "
                    f"{mode!r}"
                )
        elif mode is not None:
            raise ValueError(
                f"adjustment_mode is only meaningful for back_adjusted sources, "
                f"not {self.price_domain.value}"
            )
        object.__setattr__(self, "adjustment_mode", mode)

    # -- convenience -------------------------------------------------------
    @property
    def is_retrospective_back_adjusted(self) -> bool:
        return (
            self.price_domain is PriceDomain.BACK_ADJUSTED
            and self.adjustment_mode == ADJUSTMENT_MODE_RETROSPECTIVE
        )

    @property
    def is_point_in_time_back_adjusted(self) -> bool:
        return (
            self.price_domain is PriceDomain.BACK_ADJUSTED
            and self.adjustment_mode == ADJUSTMENT_MODE_POINT_IN_TIME
        )

    def safety(self) -> SourceSafety:
        """The three safety concepts this source confers (before any feature)."""
        reasons: dict[str, str] = {}
        if self.is_retrospective_back_adjusted:
            pit = signal = False
            reasons["source"] = (
                "retrospective_research back-adjustment applies every roll gap, "
                "including rolls that occur after T -> the value at T depends on "
                "the future and is not point-in-time / signal safe"
            )
        elif not self.causal:
            pit = signal = False
            reasons["source"] = "source lineage is marked non-causal"
        else:
            pit = signal = True

        execution_price_safe = self.causal and self.price_domain in EXECUTION_PRICE_DOMAINS
        if not execution_price_safe and "source" not in reasons:
            if self.price_domain is PriceDomain.RAW_CONTINUOUS:
                reasons["execution_price"] = (
                    "raw_continuous prices are discontinuous at rolls -- never a real "
                    "tradable price for a Fill"
                )
            elif self.price_domain is PriceDomain.BACK_ADJUSTED:
                reasons["execution_price"] = (
                    "back-adjusted prices never traded -- research / Signal only, never a "
                    "Fill price (BOUNDARY_CONTRACT E)"
                )
        return SourceSafety(
            point_in_time_safe=pit,
            signal_safe=signal,
            execution_price_safe=execution_price_safe,
            reasons=reasons,
        )

    @property
    def ts(self) -> pd.Series:
        return self.frame["ts_event_ns"]

    def price(self, field_name: str) -> pd.Series:
        if field_name not in self.frame.columns:
            raise ValueError(f"source frame has no price field {field_name!r}")
        return pd.to_numeric(self.frame[field_name], errors="coerce").astype("float64")

    def identifiers(self) -> pd.DataFrame:
        cols = ["ts_event_ns"] + [c for c in IDENTITY_COLUMNS if c in self.frame.columns]
        out = self.frame.loc[:, cols].copy()
        for k, v in self.identity.items():
            if k not in out.columns:
                out[k] = v
        return out

    def truncate_as_of(self, as_of_ts_ns: int | None) -> SourceSeries:
        """Point-in-time: keep only rows at or before ``as_of_ts_ns``."""
        if as_of_ts_ns is None:
            return self
        keep = self.frame["ts_event_ns"] <= int(as_of_ts_ns)
        return SourceSeries(
            frame=self.frame.loc[keep].reset_index(drop=True),
            price_domain=self.price_domain,
            identity=dict(self.identity),
            source_paths=self.source_paths,
            interval_ns=self.interval_ns,
            max_gap_intervals=self.max_gap_intervals,
            adjustment_mode=self.adjustment_mode,
            causal=self.causal,
        )

    def fingerprint(self) -> str:
        """Deterministic SHA-256 of the source content (order-independent of the
        original DataFrame index; sensitive to values, columns and dtypes)."""
        f = self.frame.sort_values("ts_event_ns", kind="stable")
        cols = sorted(f.columns)
        h = hashlib.sha256()
        h.update(",".join(cols).encode())
        h.update(str(self.price_domain.value).encode())
        h.update(f"|mode={self.adjustment_mode}|causal={self.causal}".encode())
        for c in cols:
            h.update(pd.util.hash_pandas_object(f[c], index=False).values.tobytes())
        return h.hexdigest()

    # -- point-in-time window primitives ---------------------------------
    def gap_breaks(self) -> pd.Series:
        """True on a row whose preceding bar interval exceeds
        ``max_gap_intervals`` * ``interval_ns`` (a missing-bar hole)."""
        d = self.ts.to_numpy("int64")
        out = np.zeros(len(d), dtype=bool)
        if len(d) > 1:
            out[1:] = np.diff(d) > (self.interval_ns * self.max_gap_intervals)
        return pd.Series(out, index=self.frame.index)

    def segment_ids(self, policy: SessionPolicy) -> pd.Series:
        """0-based, contiguous segment id per row. Rolling / return windows are
        computed strictly within a segment, so they never look across a session
        boundary or a large data gap. ``CONTINUOUS`` -> all zeros."""
        f = self.frame
        n = len(f)
        new = np.zeros(n, dtype=bool)
        if n:
            new[0] = True
        if policy in (SessionPolicy.RESET_ON_SESSION, SessionPolicy.RESET_ON_SESSION_AND_GAP):
            key_col = "trading_day" if "trading_day" in f.columns else (
                "session" if "session" in f.columns else None
            )
            if key_col is not None:
                key = f[key_col].astype("object").to_numpy()
                changed = np.zeros(n, dtype=bool)
                changed[1:] = key[1:] != key[:-1]
                new |= changed
        if policy in (SessionPolicy.RESET_ON_GAP, SessionPolicy.RESET_ON_SESSION_AND_GAP):
            new |= self.gap_breaks().to_numpy(dtype=bool)
        return pd.Series(np.cumsum(new.astype("int64")) - 1, index=f.index, dtype="int64")


def coerce_source(
    obj: SourceSeries | pd.DataFrame, *, price_domain: PriceDomain | str | None = None, **kwargs
) -> SourceSeries:
    if isinstance(obj, SourceSeries):
        return obj
    if price_domain is None:
        raise ValueError("price_domain is required when passing a bare DataFrame")
    return SourceSeries(frame=obj, price_domain=PriceDomain(price_domain), **kwargs)


def is_execution_price_safe_domain(domain: PriceDomain) -> bool:
    """Only a real raw-contract price may back a Fill (BOUNDARY_CONTRACT E)."""
    return domain in EXECUTION_PRICE_DOMAINS


def assert_no_execution_identity_leak(feature_columns: list[str]) -> None:
    leaked = [c for c in feature_columns if c in EXECUTION_IDENTITY_COLUMNS]
    if leaked:
        raise ValueError(f"feature columns leak execution identity: {leaked}")
