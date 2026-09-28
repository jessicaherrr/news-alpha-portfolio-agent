"""Empirical null / placebo tests (section 11 + Phase 13.1/13.2 sections 8-11).

* **centered block bootstrap** (``NullMethod.CENTERED_BLOCK_BOOTSTRAP``) -- the
  OFFICIAL GATING null. Block bootstrap of the mean-removed daily returns: keeps
  short-range dependence, removes the edge. Always ``n_null_samples`` replicates.

* **schedule time-shift** (``NullMethod.SCHEDULE_TIME_SHIFT``) -- a
  DIAGNOSTIC-ONLY causal null (Phase 13.2 section 4): the ReliabilityPolicy never
  gates PASS/REJECT on it. It uses a proper per-segment COMMON SUPPORT so its
  p-value is still statistically comparable:

  1. A causal segment is a maximal run of consecutive bars sharing one root, one
     contract (no roll), one ``trading_day`` and one ``session``, with no gap
     (``causal_segment_ids``).
  2. Fix a predeclared family max forward shift ``K = max_shift_bars`` and the
     execution latency ``L = execution_latency_bars`` (0 on the frozen reference
     CLI). The COMMON-SUPPORT CONTROL schedule
     (``common_support_control_schedule``) keeps a target row at segment position
     ``p`` only if ``p + K + 1 + L <= segment_end`` -- i.e. even shifted by the
     largest ``k <= K`` its next-bar+latency execution still lands inside the
     segment.
  3. Every replicate shifts the SAME control rows forward by ``k`` in
     ``[min_shift_bars, K]``. By construction none overflows, none is dropped,
     none wraps -- the control and every replicate share an identical opportunity
     set. No replicate can cherry-pick a favourable support.
  4. The observed statistic for the p-value is computed on the CONTROL run
     (``null_test_observed_statistic``), NOT the full-sample canonical statistic
     (``canonical_oos_statistic``) -- never full-support observed vs
     reduced-support null (section 3).

* **circular schedule permutation** (``NullMethod.CIRCULAR_SCHEDULE_PERMUTATION``,
  research only) -- a circular rotation that DOES wrap late rows to earlier
  timestamps; explicitly named so it is never conflated with the causal shift.

Every schedule null is rerun through the full official path (targets -> C++
ScheduledTargetStrategy -> Risk -> ExecutionSimulator -> Fill ->
PortfolioAccountant -> official validation equity). The one-sided empirical
p-value uses the finite-sample correction

    p = (1 + #{ null_stat >= observed_stat }) / (1 + n_null)     -- never 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from alpha_agent.backtest.targets import TargetSchedule, TargetScheduleRow
from alpha_agent.data.calendars import SessionCalendar, default_calendar
from alpha_agent.validation.bootstrap import resample_indices
from alpha_agent.validation.enums import BootstrapMethod, NullMethod, NullRole
from alpha_agent.validation.fingerprint import fingerprint

# The schedule time-shift null never gates the verdict (Phase 13.2 section 4).
SCHEDULE_SHIFT_ROLE = NullRole.DIAGNOSTIC
CENTERED_BLOCK_BOOTSTRAP_ROLE = NullRole.GATING


class NullTestConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    methods: tuple[NullMethod, ...] = (
        NullMethod.SCHEDULE_TIME_SHIFT,
        NullMethod.CENTERED_BLOCK_BOOTSTRAP,
    )
    n_null_samples: int = Field(default=199, ge=19, le=100_000)
    # Predeclared family shift range for the schedule time-shift null. K is FIXED
    # (not data-derived) so the common-support control is unambiguous.
    min_shift_bars: int = Field(default=2, ge=1, le=1_000_000)
    max_shift_bars: int = Field(default=10, ge=1, le=1_000_000)
    # Execution latency assumed by the common-support rule. MUST equal the
    # engine's EngineConfig::latency_bars (0 on the frozen reference CLI).
    execution_latency_bars: int = Field(default=0, ge=0, le=1_000)
    gap_multiple: float = Field(default=1.5, gt=1.0, le=100.0)
    block_length: int = Field(default=20, ge=1, le=10_000)
    seed: int = Field(default=0, ge=0)
    # the statistic a positive edge should inflate; one-sided upper tail
    statistic: str = "daily_sharpe"

    @model_validator(mode="after")
    def _check(self) -> NullTestConfig:
        if self.min_shift_bars > self.max_shift_bars:
            raise ValueError("min_shift_bars must be <= max_shift_bars")
        return self

    def identity(self) -> str:
        return fingerprint("valnull3", self.model_dump(mode="json"))


def causal_segment_ids(
    bars: pd.DataFrame,
    *,
    root_symbol: str,
    calendar: SessionCalendar | None = None,
    gap_multiple: float = 1.5,
) -> list[int]:
    """A segment id per bar (bars taken chronologically). The id increments at a
    root change, an ``instrument_id`` change (roll), a ``trading_day`` change, a
    ``session`` change, or a time gap > ``gap_multiple`` x the median bar
    interval. A shifted target row may only move within one segment."""
    df = bars.sort_values("ts_event_ns").reset_index(drop=True)
    ts = df["ts_event_ns"].to_numpy(dtype=np.int64)
    n = len(df)
    if n == 0:
        return []
    cal = calendar or default_calendar()
    tdays, sessions = cal.classify_series(df["ts_event_ns"], root_symbol)
    tdays = [str(v) for v in tdays]
    sessions = [str(v) for v in sessions]
    inst = (
        df["instrument_id"].astype("int64").to_list()
        if "instrument_id" in df.columns
        else [0] * n
    )
    diffs = np.diff(ts)
    median_iv = float(np.median(diffs)) if diffs.size else 0.0
    seg = [0] * n
    for i in range(1, n):
        broke = (
            inst[i] != inst[i - 1]
            or tdays[i] != tdays[i - 1]
            or sessions[i] != sessions[i - 1]
            or (median_iv > 0 and (ts[i] - ts[i - 1]) > gap_multiple * median_iv)
        )
        seg[i] = seg[i - 1] + (1 if broke else 0)
    return seg


def _shift_grid(config: NullTestConfig) -> list[int]:
    """Every integer forward shift in ``[min_shift_bars, max_shift_bars]`` (K is
    fixed, predeclared -- never data-derived)."""
    lo, hi = config.min_shift_bars, config.max_shift_bars
    if hi < lo:
        return []
    return list(range(lo, min(hi, lo + config.n_null_samples - 1) + 1))


def _rows_by_pos(schedule: TargetSchedule, ts_index: dict[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in schedule.rows:
        p = ts_index.get(int(r.ts_event_ns))
        if p is not None:
            out.append((p, int(r.target_units)))
    return out


def _segment_end_positions(segment_ids: list[int]) -> list[int]:
    n = len(segment_ids)
    seg_end = [0] * n
    end = n - 1
    for i in range(n - 1, -1, -1):
        if i < n - 1 and segment_ids[i + 1] != segment_ids[i]:
            end = i
        seg_end[i] = end
    return seg_end


def common_support_control_schedule(
    schedule: TargetSchedule,
    bar_ts_sorted: list[int],
    segment_ids: list[int],
    config: NullTestConfig,
) -> TargetSchedule:
    """The COMMON-SUPPORT control schedule for the schedule time-shift null: drop
    any target row within ``max_shift_bars + 1 + execution_latency_bars`` bars of
    its causal segment's end, so that EVERY shift ``k <= max_shift_bars`` keeps
    that row's next-bar(+latency) execution inside the segment. The control and
    every shifted replicate then share an identical opportunity set."""
    ts = sorted({int(t) for t in bar_ts_sorted})
    n = len(ts)
    if n == 0 or len(segment_ids) != n:
        return schedule.model_copy(update={"rows": ()})
    ts_index = {t: i for i, t in enumerate(ts)}
    seg_end = _segment_end_positions(segment_ids)
    margin = config.max_shift_bars + 1 + config.execution_latency_bars
    kept = tuple(
        r for r in schedule.rows
        if (p := ts_index.get(int(r.ts_event_ns))) is not None
        and p + margin <= seg_end[p]
    )
    return schedule.model_copy(update={"rows": kept})


def shifted_schedules(
    control_schedule: TargetSchedule,
    bar_ts_sorted: list[int],
    segment_ids: list[int],
    config: NullTestConfig,
) -> list[tuple[int, TargetSchedule]]:
    """``[(k, shifted_control), ...]`` -- the COMMON-SUPPORT control schedule
    (from :func:`common_support_control_schedule`) shifted FORWARD by each
    ``k`` in ``[min_shift_bars, max_shift_bars]``. Every row stays in its segment
    by construction: no drop, no wrap, identical source rows for all ``k``."""
    ts = sorted({int(t) for t in bar_ts_sorted})
    n = len(ts)
    if n == 0 or len(segment_ids) != n:
        return []
    ts_index = {t: i for i, t in enumerate(ts)}
    seg = list(segment_ids)
    rows = _rows_by_pos(control_schedule, ts_index)
    if not rows:
        return []
    out: list[tuple[int, TargetSchedule]] = []
    for k in _shift_grid(config):
        moved: list[tuple[int, int]] = []
        for p, units in rows:
            q = p + k
            # invariant guaranteed by common_support_control_schedule
            assert q < n and seg[q] == seg[p], "control schedule support is not common"
            moved.append((q, units))
        new_rows = tuple(
            TargetScheduleRow(
                ts_event_ns=ts[q], root_symbol=control_schedule.root_symbol,
                target_units=u, strategy_fingerprint=control_schedule.strategy_fingerprint,
            )
            for q, u in sorted(moved)
        )
        out.append((k, control_schedule.model_copy(update={"rows": new_rows})))
    return out


def circular_permuted_schedules(
    schedule: TargetSchedule, bar_ts_sorted: list[int], config: NullTestConfig
) -> list[tuple[int, TargetSchedule]]:
    """RESEARCH-ONLY permutation null (``CIRCULAR_SCHEDULE_PERMUTATION``). A
    circular rotation that WRAPS late rows to earlier timestamps -- not causal,
    never the default. Kept explicit so it is not conflated with the causal
    shift (section 9)."""
    ts = sorted({int(t) for t in bar_ts_sorted})
    n = len(ts)
    if n == 0:
        return []
    ts_index = {t: i for i, t in enumerate(ts)}
    rows = _rows_by_pos(schedule, ts_index)
    if not rows:
        return []
    out: list[tuple[int, TargetSchedule]] = []
    perm_shifts = list(range(config.min_shift_bars, min(n - 1, config.n_null_samples)))
    for s in perm_shifts:
        seen: dict[int, int] = {}
        for p, units in rows:
            seen.setdefault((p + s) % n, units)
        new_rows = tuple(
            TargetScheduleRow(ts_event_ns=ts[q], root_symbol=schedule.root_symbol,
                              target_units=u, strategy_fingerprint=schedule.strategy_fingerprint)
            for q, u in sorted(seen.items())
        )
        out.append((s, schedule.model_copy(update={"rows": new_rows})))
    return out


def centered_block_bootstrap_null_stats(
    returns: np.ndarray, config: NullTestConfig, statfn
) -> np.ndarray:
    """Distribution of ``statfn`` under the block bootstrap of the *centered*
    daily returns (mean removed): dependence kept, edge removed."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.empty(0, dtype=float)
    centered = x - x.mean()
    rng = np.random.default_rng(config.seed)
    stats = np.empty(config.n_null_samples, dtype=float)
    for i in range(config.n_null_samples):
        idx = resample_indices(rng, centered.size, BootstrapMethod.MOVING_BLOCK, config.block_length)
        stats[i] = statfn(centered[idx])
    return stats[np.isfinite(stats)]


def empirical_p_value(observed_stat: float, null_stats: np.ndarray) -> tuple[float, int]:
    """One-sided (upper-tail) empirical p-value with the finite-sample
    correction. Returns ``(p_value, n_null_used)``. Never 0."""
    ns = np.asarray(null_stats, dtype=float)
    ns = ns[np.isfinite(ns)]
    n = int(ns.size)
    if n == 0 or not np.isfinite(observed_stat):
        return (1.0, 0)
    ge = int(np.count_nonzero(ns >= observed_stat))
    return ((1 + ge) / (1 + n), n)


class NullTestResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    method: NullMethod
    role: NullRole                            # GATING or DIAGNOSTIC (section 4)
    statistic: str
    # observed statistic on THIS null's own support convention (for schedule
    # shift: the common-support control run, NOT the full-sample canonical stat).
    observed_stat: float
    # the full-sample canonical OOS statistic, recorded for contrast (section 3).
    canonical_oos_statistic: float
    n_null_samples: int = Field(ge=0)
    n_null_ge_observed: int = Field(ge=0)
    p_value: float = Field(gt=0.0, le=1.0)
    null_mean: float
    null_std: float
    null_q95: float


def summarize_null(
    method: NullMethod,
    role: NullRole,
    statistic: str,
    observed_stat: float,
    null_stats: np.ndarray,
    *,
    canonical_oos_statistic: float,
) -> NullTestResult:
    ns = np.asarray(null_stats, dtype=float)
    ns = ns[np.isfinite(ns)]
    p, n = empirical_p_value(observed_stat, ns)
    ge = int(np.count_nonzero(ns >= observed_stat)) if n else 0
    return NullTestResult(
        method=method,
        role=role,
        statistic=statistic,
        observed_stat=float(observed_stat) if np.isfinite(observed_stat) else 0.0,
        canonical_oos_statistic=(
            float(canonical_oos_statistic) if np.isfinite(canonical_oos_statistic) else 0.0
        ),
        n_null_samples=n,
        n_null_ge_observed=ge,
        p_value=p,
        null_mean=float(ns.mean()) if n else 0.0,
        null_std=float(ns.std(ddof=1)) if n >= 2 else 0.0,
        null_q95=float(np.quantile(ns, 0.95)) if n else 0.0,
    )
