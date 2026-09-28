"""Phase 13.5C -- the real-market research + validation MATRIX orchestrator.

Runs every frozen candidate trial (:mod:`alpha_agent.strategy.candidates_phase_13_5c`)
through the frozen Phase 13 / 13.1 / 13.2 reliability machinery on the real
2018-2024 CME research / validation data, then:

* builds the CORRECTED unique global multiple-testing family (Part 1.1 -- 107
  hypotheses: 21 canonical + 86 predeclared neighbours; cross-market roots and
  the data-quality sensitivity pass are NOT trials) and re-derives every
  canonical trial's headline verdict from that global BH/FDR + global DSR
  effective-trial count -- the frozen :class:`ReliabilityPolicy` applied to the
  corrected family (sections 11, 12, 17);
* summarises cross-market evidence per baseline family from the per-root
  canonical results (section 16 -- descriptive, never a new ``TrialRecord``);
* re-runs the identical frozen process on the degraded-vendor-days-excluded
  sample and reports the metric deltas + ``verdict_changed`` (section 14 --
  robustness of the same hypotheses, NOT added to the 107).

NOTHING here downloads, queries, cost-fetches, loads, features or reports the
2025 ``LOCKED_HOLDOUT``. Every economic number is Fill-derived from the frozen
C++ ``quant_backtest_targets_csv`` path (``PassThroughRiskManager`` -- Phase 08
hard risk limits are NOT enforced on this path, recorded truthfully).

Design notes (smallest deterministic architecture-consistent choices):

* SIGNAL path -- the 4 baselines take a DAILY causal ``trading_day`` aggregation
  of the 1-minute forward-adjusted continuous front; Silver Bullet takes the
  native 1-minute forward-adjusted continuous. Both are signal-safe / never a
  Fill price. EXECUTION is always the native 1-minute RAW_CONTRACT canonical
  bars via the C++ engine (next eligible 1m bar -> Fill).
* :class:`_Phase135cAdapter` overrides ``schedule_for`` so the daily baseline
  feature frame is computed on a SYNTHETIC contiguous 1-per-trading-day index
  (no spurious ``RESET_ON_GAP`` across weekends / holidays) and the emitted
  target rows are then remapped to the REAL last-eligible 1m ``ts_event_ns`` of
  each trading day. The C++ engine still executes at the next real 1m bar.
* Headline OOS = the VALIDATION window (2023-2024) via the additive
  ``ValidationEngine(oos_split_role=SplitRole.VALIDATION)``; walk-forward folds
  are built inside TRAIN (2018-2022) only.
* The per-trial :class:`ValidationReport` from the engine keeps its LOCAL
  per-family trial family (frozen Phase 13 behaviour). The HEADLINE verdict this
  module reports is re-derived by :func:`_global_verdict` -- the same frozen
  policy, evaluated against the global 107-hypothesis BH/FDR family and the
  global DSR effective-trial count.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_agent.data.calendars import SessionCalendar, default_calendar
from alpha_agent.data.real_market_dataset import (
    HOLDOUT_START_NS,
    RESEARCH_WINDOW,
    ROOTS,
    VALIDATION_WINDOW,
    ReconstitutedRoot,
    _iso_ns,
    daily_signal_series,
    degraded_trading_days,
    reconstitute_root,
    roll_close_marks,
    write_roll_close_marks_csv,
)
from alpha_agent.strategy import compile_strategy
from alpha_agent.strategy.baselines.factories import (
    make_breakout_spec,
    make_ma_trend_spec,
    make_mean_reversion_spec,
    make_tsmom_spec,
)
from alpha_agent.strategy.baselines.params import (
    BreakoutParams,
    MaTrendParams,
    MeanReversionParams,
    TsmomParams,
)
from alpha_agent.strategy.baselines.silver_bullet import SilverBulletParams, make_silver_bullet_spec
from alpha_agent.strategy.candidates_phase_13_5c import (
    MT_FAMILY_GLOBAL,
    CandidateManifest,
    CandidateTrial,
    build_candidate_manifest,
)
from alpha_agent.strategy.spec import StrategySpec
from alpha_agent.validation.bootstrap import BootstrapConfig
from alpha_agent.validation.cost_stress import CostScenario, CostStressPlan
from alpha_agent.validation.crossmarket import (
    CrossMarketEvidence,
    RootRunSummary,
    evaluate_cross_market,
)
from alpha_agent.validation.dataset import DatasetIdentity, frame_content_hash
from alpha_agent.validation.dsr import DeflatedSharpeResult, deflated_sharpe_ratio
from alpha_agent.validation.engine import ValidationEngine
from alpha_agent.validation.enums import EvidenceStatus, RegimeKind, SplitRole
from alpha_agent.validation.multiple_testing import MultipleTestingFamily, TrialRecord
from alpha_agent.validation.nulls import NullTestConfig
from alpha_agent.validation.policy import (
    MinimumSampleRequirements,
    ReliabilityPolicy,
    evaluate_policy,
)
from alpha_agent.validation.regime import (
    RegimeLabelling,
    RegimeStabilityResult,
    causal_volatility_regime_labels,
    evaluate_regime_stability,
)
from alpha_agent.validation.report import ValidationReport
from alpha_agent.validation.runner import CliBacktestRunner
from alpha_agent.validation.spec import ValidationSpec
from alpha_agent.validation.splits import SplitPlan, SplitWindow
from alpha_agent.validation.stability import ParameterNeighbourhood
from alpha_agent.validation.walkforward import WalkForwardConfig

# --------------------------------------------------------------------------
# frozen configuration -- Phase 13 defaults + Phase 13.5C split roles
# --------------------------------------------------------------------------
DAY_NS = 86_400_000_000_000
CAPITAL_BASE_USD = 100_000.0
# feature warm-up before the headline OOS / each fold, in 1-minute bars:
# ~500 calendar days (>= 250 trading-day slow windows). 2023-01-01 - 500d is
# still well inside TRAIN (2018-2022) -- never touches the holdout.
WARMUP_BARS_1M = 720_000
DEFAULT_CLI = "build/cpp/cpp/quant_backtest_targets_csv"

_BASELINE_FACTORY: dict[str, tuple[Callable, type]] = {
    "tsmom": (make_tsmom_spec, TsmomParams),
    "ma_trend": (make_ma_trend_spec, MaTrendParams),
    "breakout": (make_breakout_spec, BreakoutParams),
    "mean_reversion": (make_mean_reversion_spec, MeanReversionParams),
}


def frozen_policy() -> ReliabilityPolicy:
    """The frozen Phase 13 default :class:`ReliabilityPolicy` (configs/validation.yaml).
    Only the cosmetic ``policy_name`` is set -- it is not in the fingerprint."""
    return ReliabilityPolicy(
        policy_name="phase_13_5c",
        minimum_sample=MinimumSampleRequirements(),  # 60 / 20 / 10 / 3 / 20
    )


def frozen_walk_forward() -> WalkForwardConfig:
    return WalkForwardConfig(
        n_folds=4, scheme="expanding", min_train_days=365, test_days=180,
        embargo_days=5, warmup_bars=WARMUP_BARS_1M,
    )


def frozen_null_config() -> NullTestConfig:
    # defaults already mirror configs/validation.yaml:
    #   methods = (schedule_time_shift DIAGNOSTIC, centered_block_bootstrap GATING)
    #   n_null_samples = 199, min/max shift 2/10, gap_multiple 1.5, block 20, seed 0
    return NullTestConfig()


def frozen_bootstrap_config() -> BootstrapConfig:
    return BootstrapConfig()  # stationary, 2000 resamples, block 20, ci 0.95, seed 0


def frozen_cost_plan() -> CostStressPlan:
    return CostStressPlan(
        base_commission_per_contract_usd=2.0,
        base_slippage_ticks=0.0,
        base_spread_ticks=0.0,
        scenarios=(
            CostScenario(label="baseline_1_0x", multiplier=1.0),
            CostScenario(label="stress_1_5x", multiplier=1.5),
            CostScenario(label="stress_2_0x", multiplier=2.0),
        ),
    )


def build_split_plan() -> SplitPlan:
    """TRAIN 2018-2022 / VALIDATION 2023-2024 / LOCKED_HOLDOUT 2025 (metadata only).

    The holdout window starts a 6-day embargo after the last research bar and is
    NEVER selectable / loaded / queried. Its only purpose here is to make the
    chronological split explicit and to force any 2025 access to fail loudly.
    """
    train = SplitWindow(
        role=SplitRole.TRAIN,
        start_ts_ns=_iso_ns(RESEARCH_WINDOW[0]),
        end_ts_ns=_iso_ns(RESEARCH_WINDOW[1]),
    )
    validation = SplitWindow(
        role=SplitRole.VALIDATION,
        start_ts_ns=_iso_ns(VALIDATION_WINDOW[0]),
        end_ts_ns=_iso_ns(VALIDATION_WINDOW[1]),  # 2025-01-01 exclusive -> data <= 2024-12-31
    )
    holdout = SplitWindow(
        role=SplitRole.LOCKED_HOLDOUT,
        start_ts_ns=HOLDOUT_START_NS + 6 * DAY_NS,
        end_ts_ns=_iso_ns("2026-01-01"),
    )
    return SplitPlan(windows=(train, validation, holdout), embargo_days=5)


# --------------------------------------------------------------------------
# strategy-family adapter -- daily-signal baselines + native-1m Silver Bullet
# --------------------------------------------------------------------------
def _spec_factory(family_key: str, root: str) -> Callable[[dict], StrategySpec]:
    if family_key == "silver_bullet":
        return lambda p: make_silver_bullet_spec(SilverBulletParams(**p))
    factory, model = _BASELINE_FACTORY[family_key]
    return lambda p: factory(model(**p))


class _Phase135cAdapter:
    """A :class:`~alpha_agent.validation.engine.StrategyFamilyAdapter`.

    * baselines -- features on a SYNTHETIC contiguous 1-per-trading-day index
      derived from the 1m forward-adjusted continuous; target rows remapped to
      the real last-eligible 1m ``ts_event_ns`` of each trading day.
    * silver_bullet -- features on the native 1m forward-adjusted continuous
      (with real ``trading_day`` / ``session`` labels for the causal detector).
    """

    def __init__(
        self,
        family_key: str,
        root: str,
        canonical_params: dict,
        *,
        signal_1m: pd.DataFrame,       # 1m forward-adjusted continuous, full 2018-2024
        calendar: SessionCalendar,
    ):
        self.strategy_key = family_key
        self.canonical_params = dict(canonical_params)
        self._root = root
        self._is_sb = family_key == "silver_bullet"
        self._spec_of = _spec_factory(family_key, root)
        self._cal = calendar
        from alpha_agent.schemas.market_data import PriceDomain

        self._PriceDomain = PriceDomain
        if self._is_sb:
            tday, sess = calendar.classify_series(signal_1m["ts_event_ns"], root)
            self._sig = signal_1m.assign(
                trading_day=tday.astype(str), session=sess.astype(str)
            ).reset_index(drop=True)
        else:
            self._daily = daily_signal_series(signal_1m, root, calendar=calendar)

    def spec_for(self, params: dict) -> StrategySpec:
        return self._spec_of(params)

    # -- baseline: synthetic contiguous daily index -----------------------
    def _baseline_schedule(self, spec: StrategySpec, lo: int, hi: int, emit_from_ts_ns: int):
        return daily_baseline_schedule(
            spec, self._daily, lo, hi, emit_from_ts_ns,
            root=self._root, price_domain=self._PriceDomain,
        )

    # -- silver bullet: native 1m ---------------------------------------
    def _sb_schedule(self, spec: StrategySpec, lo: int, hi: int, emit_from_ts_ns: int):
        return native_1m_schedule(
            spec, self._sig, lo, hi, emit_from_ts_ns,
            root=self._root, price_domain=self._PriceDomain,
        )

    def schedule_for(self, spec: StrategySpec, bars: pd.DataFrame, *, emit_from_ts_ns: int):
        lo = int(bars["ts_event_ns"].iloc[0])
        hi = int(bars["ts_event_ns"].iloc[-1])
        if self._is_sb:
            return self._sb_schedule(spec, lo, hi, emit_from_ts_ns)
        return self._baseline_schedule(spec, lo, hi, emit_from_ts_ns)


# --------------------------------------------------------------------------
# public, strategy-agnostic schedule builders (extracted, byte-identical, from
# `_Phase135cAdapter._baseline_schedule` / `_sb_schedule` above -- Agent
# runtime-integration release, section 6/7). `_Phase135cAdapter` now calls
# these directly; nothing about its own behaviour changed (proven by the
# unmodified Phase 13.5C test suite). Reused as-is by
# `alpha_agent.agents.execution_service` for an arbitrary already-compiled
# `StrategySpec` -- no frozen semantics are duplicated, only this pure feature
# / schedule composition glue.
# --------------------------------------------------------------------------
def daily_baseline_schedule(
    spec: StrategySpec, daily: pd.DataFrame, lo: int, hi: int, emit_from_ts_ns: int,
    *, root: str, price_domain,
):
    """The 4 baseline families' cadence: a SYNTHETIC contiguous 1-per-trading-
    day index (no spurious RESET_ON_GAP across weekends/holidays), remapped
    back to each trading day's real last-eligible ``ts_event_ns`` after target
    construction. ``daily`` is the causal daily aggregation of the 1m
    forward-adjusted continuous (see `real_market_dataset.daily_signal_series`).
    """
    from alpha_agent.backtest.targets import build_target_schedule
    from alpha_agent.features import compute_features
    from alpha_agent.features.source import SourceSeries

    d = daily
    w = d[(d["ts_event_ns"] >= lo) & (d["ts_event_ns"] <= hi)].reset_index(drop=True)
    if len(w) < 3:
        return None
    real_ts = w["ts_event_ns"].to_numpy("int64")
    synth = (np.arange(len(w), dtype="int64") + 1) * DAY_NS
    frame = w.assign(ts_event_ns=synth)
    src = SourceSeries(
        frame=frame[["ts_event_ns", "open", "high", "low", "close", "volume"]],
        price_domain=price_domain.BACK_ADJUSTED,
        adjustment_mode="forward_adjusted",
        identity={"root_symbol": root},
        interval_ns=DAY_NS,
    )
    plan = compile_strategy(spec)
    feats = compute_features(src, [b.spec for b in plan.feature_bindings], require_point_in_time=False)
    sched = build_target_schedule(plan, feats)
    by_synth = {int(s): int(r) for s, r in zip(synth, real_ts)}
    rows = tuple(
        row.model_copy(update={"ts_event_ns": by_synth[int(row.ts_event_ns)]})
        for row in sched.rows
        if by_synth[int(row.ts_event_ns)] >= emit_from_ts_ns
    )
    return sched.model_copy(update={"rows": rows})


def native_1m_schedule(
    spec: StrategySpec, signal_1m_labelled: pd.DataFrame, lo: int, hi: int, emit_from_ts_ns: int,
    *, root: str, price_domain,
):
    """Silver Bullet's cadence: the native 1-minute forward-adjusted
    continuous, with real ``trading_day``/``session`` labels already attached
    (see `_Phase135cAdapter.__init__`'s ``calendar.classify_series`` call)."""
    from alpha_agent.backtest.targets import build_target_schedule
    from alpha_agent.features import compute_features
    from alpha_agent.features.source import SourceSeries

    s = signal_1m_labelled
    w = s[(s["ts_event_ns"] >= lo) & (s["ts_event_ns"] <= hi)].reset_index(drop=True)
    if len(w) < 3:
        return None
    src = SourceSeries(
        frame=w[["ts_event_ns", "open", "high", "low", "close", "volume",
                 "trading_day", "session"]],
        price_domain=price_domain.BACK_ADJUSTED,
        adjustment_mode="forward_adjusted",
        identity={"root_symbol": root},
        interval_ns=60_000_000_000,
    )
    plan = compile_strategy(spec)
    feats = compute_features(src, [b.spec for b in plan.feature_bindings], require_point_in_time=False)
    full = build_target_schedule(plan, feats)
    kept = tuple(r for r in full.rows if r.ts_event_ns >= emit_from_ts_ns)
    return full.model_copy(update={"rows": kept})


def dataset_identity_for_root(recon: ReconstitutedRoot, exec_bars: pd.DataFrame) -> DatasetIdentity:
    """Public alias of :func:`_dataset_identity`, for reuse outside this
    module (Agent runtime-integration release, section 6/7) without reaching
    into a leading-underscore name. Byte-identical -- not a new formula."""
    return _dataset_identity(recon, exec_bars)


# --------------------------------------------------------------------------
# roll-continuation overlay -- an EXECUTION concern, applied at the runner
# --------------------------------------------------------------------------
def _inject_roll_continuation(sched, roll_ts: tuple[int, ...]):
    """Stamp a mechanical roll-continuation target row at each observed roll
    ``effective_ts_ns`` that falls strictly inside the schedule's own time span,
    carrying the target held just before it (skip: no prior row, a row already at
    that ts, or a flat prior target). This makes the C++ engine execute the roll
    close-leg promptly (at the auxiliary same-timestamp mark) instead of
    deferring to the next daily decision -- which for a short-dated contract (CL)
    can fall past the outgoing contract's expiry. It re-stamps the HELD target
    only: never a new strategy decision, and applied to the schedule that goes to
    the C++ engine, NOT to the schedule the null test / schedule hash / stats see.
    """
    import bisect

    from alpha_agent.backtest.targets import TargetScheduleRow

    if not roll_ts or not sched.rows:
        return sched
    rows = list(sched.rows)
    ts_sorted = [r.ts_event_ns for r in rows]
    have = set(ts_sorted)
    lo, hi = ts_sorted[0], ts_sorted[-1]
    extra = []
    for rt in sorted(int(t) for t in roll_ts):
        if not (lo < rt < hi) or rt in have:
            continue
        i = bisect.bisect_left(ts_sorted, rt)
        if i == 0 or rows[i - 1].target_units == 0:
            continue
        extra.append(TargetScheduleRow(
            ts_event_ns=rt, root_symbol=sched.root_symbol,
            target_units=rows[i - 1].target_units,
            strategy_fingerprint=sched.strategy_fingerprint,
            matched_rule_id="roll_continuation",
        ))
    if not extra:
        return sched
    merged = tuple(sorted(rows + extra, key=lambda r: r.ts_event_ns))
    return sched.model_copy(update={"rows": merged})


class _RollAwareCliRunner:
    """A :class:`~alpha_agent.validation.runner.BacktestRunner` that overlays the
    roll-continuation rows onto every schedule just before it reaches the C++
    engine, then delegates to :class:`CliBacktestRunner`."""

    def __init__(self, inner: CliBacktestRunner, roll_ts: tuple[int, ...]):
        self._inner = inner
        self._roll_ts = tuple(sorted(int(t) for t in roll_ts))

    @property
    def _counter(self) -> int:
        return self._inner._counter

    def run(self, *, schedule, **kw):
        return self._inner.run(
            schedule=_inject_roll_continuation(schedule, self._roll_ts), **kw
        )


#: Public alias (Agent runtime-integration release, section 6/7) -- the exact
#: same roll-aware runner, reused as-is so a production
#: `ExecutionValidationService` gets the same roll handling as the committed
#: 107-hypothesis matrix instead of a second, undocumented roll behaviour.
RollAwareCliRunner = _RollAwareCliRunner


# --------------------------------------------------------------------------
# per-trial run
# --------------------------------------------------------------------------
@dataclass
class TrialRun:
    family_key: str
    root_symbol: str
    strategy_fingerprint: str
    strategy_id: str
    report: ValidationReport
    canonical_params: dict
    neighbour_records: list[TrialRecord]       # role="neighbour", schedule_hash=None
    canonical_record: TrialRecord              # role="canonical", schedule_hash=None
    root_summary: RootRunSummary
    regime: RegimeStabilityResult
    runtime_s: float
    n_cpp_runs: int
    roll_close_marks_sha256: str = ""
    n_roll_close_marks: int = 0
    rolls_priced_auxiliary_marks: int = 0


def _dataset_identity(recon: ReconstitutedRoot, exec_bars: pd.DataFrame) -> DatasetIdentity:
    cal = default_calendar()
    src_fp = hashlib.sha256(
        f"phase_13_5c|{recon.root}|{recon.calendar_version}|"
        f"{len(recon.canonical_bars)}|{int(recon.canonical_bars['ts_event_ns'].max())}".encode()
    ).hexdigest()
    return DatasetIdentity(
        root_symbol=recon.root,
        price_domain="raw_contract",
        adjustment_mode=None,
        source_fingerprint=src_fp,
        bars_content_hash=frame_content_hash(exec_bars),
        n_bars=len(exec_bars),
        first_ts_ns=int(exec_bars["ts_event_ns"].iloc[0]),
        last_ts_ns=int(exec_bars["ts_event_ns"].iloc[-1]),
        contracts_content_hash=frame_content_hash(recon.contracts),
        trading_day_convention=cal.trading_day_convention(recon.root),
        note="Phase 13.5C offline reconstitution; execution = native 1m RAW_CONTRACT",
    )


def _neighbourhood(trial: CandidateTrial) -> ParameterNeighbourhood:
    # keep root_symbol -- it is a scalar and the engine passes each params dict
    # straight to ``adapter.spec_for`` which builds a full typed params model.
    # Stripping it would raise a missing-field error and the neighbour
    # fingerprints would not match the frozen manifest.
    return ParameterNeighbourhood(
        strategy_key=trial.family_key,
        canonical_params=dict(trial.canonical_params),
        neighbour_params=tuple(dict(p) for p in trial.neighbour_params),
    )


def _spec(trial: CandidateTrial, recon: ReconstitutedRoot, exec_bars: pd.DataFrame,
          policy: ReliabilityPolicy) -> ValidationSpec:
    return ValidationSpec(
        label=f"{trial.root_symbol}__{trial.family_key.upper()}",
        strategy_fingerprint=trial.strategy_fingerprint,
        strategy_key=trial.family_key,
        dataset=_dataset_identity(recon, exec_bars),
        split_plan=build_split_plan(),
        walk_forward=frozen_walk_forward(),
        capital_base_usd=CAPITAL_BASE_USD,
        cost_stress=frozen_cost_plan(),
        null_test=frozen_null_config(),
        bootstrap=frozen_bootstrap_config(),
        trial_family_id=trial.multiple_testing_family_id,
        parameter_neighbourhood=_neighbourhood(trial),
        minimum_sample=policy.minimum_sample,
        reliability_policy_fingerprint=policy.identity(),
    )


def _canonical_and_neighbour_records(
    trial: CandidateTrial, report: ValidationReport
) -> tuple[TrialRecord, list[TrialRecord]]:
    """Reconstruct role-labelled :class:`TrialRecord`s (schedule_hash=None, to
    match the frozen pre-run identity enumeration) from the engine's report."""
    p_by_label = {d.label: d.p_value for d in report.fdr_result.decisions}
    sharpe_by_fp: dict[str, float] = {}
    if report.parameter_stability is not None:
        for nb in report.parameter_stability.neighbours:
            sharpe_by_fp[nb.strategy_fingerprint] = nb.oos_daily_sharpe
    canonical = TrialRecord(
        label=f"canonical:{trial.family_key}/{trial.root_symbol}",
        role="canonical",
        strategy_fingerprint=trial.strategy_fingerprint,
        schedule_hash=None,
        p_value=float(p_by_label.get("canonical", 1.0)),
        null_tested=True,
        observed_daily_sharpe=report.oos_metrics.daily_sharpe,
        observed_net_pnl_usd=report.oos_metrics.oos_net_pnl_usd,
    )
    neighbours: list[TrialRecord] = []
    for i, fp in enumerate(trial.neighbour_fingerprints):
        neighbours.append(
            TrialRecord(
                label=f"neighbour:{trial.family_key}/{trial.root_symbol}/{i}",
                role="neighbour",
                strategy_fingerprint=fp,
                schedule_hash=None,
                p_value=float(p_by_label.get(f"neighbour_{i}", 1.0)),
                null_tested=True,
                observed_daily_sharpe=sharpe_by_fp.get(fp),
            )
        )
    return canonical, neighbours


def _regime_evidence(report: ValidationReport, adapter: _Phase135cAdapter,
                     policy: ReliabilityPolicy) -> RegimeStabilityResult:
    """Post-hoc CAUSAL volatility-regime evidence: tertile cut points derived
    from TRAIN (2018-2022) close-to-close abs moves of the forward-adjusted
    continuous, applied to the OOS (2023-2024) trading days. Never gates (the
    frozen policy has ``require_regime_stability = False``); reported for
    section 15."""
    if adapter._is_sb:
        daily = adapter._sig.groupby("trading_day", sort=True)["close"].last()
        daily = daily.reset_index().rename(columns={"index": "trading_day"})
    else:
        daily = adapter._daily[["trading_day", "close"]].copy()
    daily = daily.sort_values("trading_day").reset_index(drop=True)
    abs_move = daily["close"].diff().abs().fillna(0.0).to_numpy(float)
    tdays = daily["trading_day"].astype(str).to_numpy()
    val_start = str(pd.Timestamp(VALIDATION_WINDOW[0]).date())
    n_train = int((tdays < val_start).sum())
    lab = causal_volatility_regime_labels(abs_move, train_n_days=n_train, lookback=20)
    label_by_day = dict(zip(tdays, lab.labels))
    oos_days = list(report.oos_daily.trading_day)
    oos_labels = tuple(label_by_day.get(str(d), "unknown") for d in oos_days)
    labelling = RegimeLabelling(
        kind=RegimeKind.VOLATILITY, labels=oos_labels, definition=lab.definition,
    )
    return evaluate_regime_stability(
        report.oos_daily.returns_array(),
        report.oos_daily.pnl_array(),
        labelling,
        concentration_threshold=policy.regime_max_pnl_share,
    )


def run_trial(
    trial: CandidateTrial,
    recon: ReconstitutedRoot,
    *,
    policy: ReliabilityPolicy,
    work_dir: Path,
    cli: str = DEFAULT_CLI,
    exclude_trading_days: set[str] | None = None,
    calendar: SessionCalendar | None = None,
) -> TrialRun:
    """One canonical trial through the frozen Phase 13 engine over the real
    reconstituted data. ``exclude_trading_days`` (data-quality sensitivity)
    drops those exchange trading days from BOTH the signal and execution bars."""
    calendar = calendar or default_calendar()
    t0 = time.monotonic()
    span_lo = _iso_ns(RESEARCH_WINDOW[0])
    span_hi = HOLDOUT_START_NS  # strictly < 2025-01-01

    exec_bars = recon.canonical_bars
    exec_bars = exec_bars[
        (exec_bars["ts_event_ns"] >= span_lo) & (exec_bars["ts_event_ns"] < span_hi)
    ][["ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume"]]
    exec_bars = exec_bars.sort_values("ts_event_ns").reset_index(drop=True)

    sig_1m = recon.forward_adjusted
    sig_1m = sig_1m[(sig_1m["ts_event_ns"] >= span_lo) & (sig_1m["ts_event_ns"] < span_hi)]
    sig_1m = sig_1m.sort_values("ts_event_ns").reset_index(drop=True)

    marks = roll_close_marks(recon, span_lo, span_hi)
    if exclude_trading_days:
        et, _ = calendar.classify_series(exec_bars["ts_event_ns"], recon.root)
        exec_bars = exec_bars[~et.astype(str).isin(exclude_trading_days).to_numpy()].reset_index(drop=True)
        st, _ = calendar.classify_series(sig_1m["ts_event_ns"], recon.root)
        sig_1m = sig_1m[~st.astype(str).isin(exclude_trading_days).to_numpy()].reset_index(drop=True)
        if not marks.empty:
            mt, _ = calendar.classify_series(marks["ts_event_ns"], recon.root)
            marks = marks[~mt.astype(str).isin(exclude_trading_days).to_numpy()].reset_index(drop=True)

    work_dir.mkdir(parents=True, exist_ok=True)
    marks_path, marks_sha = write_roll_close_marks_csv(marks, work_dir / "roll_close_marks.csv")

    exec_days = set()
    if exclude_trading_days:
        exec_days = set(exclude_trading_days)
    roll_ts = []
    for rl in recon.rolls:
        et = int(rl.effective_ts_ns)
        if span_lo <= et < span_hi:
            if exec_days:
                tday, _ = calendar.classify(et, recon.root)
                if tday.isoformat() in exec_days:
                    continue
            roll_ts.append(et)

    adapter = _Phase135cAdapter(
        trial.family_key, trial.root_symbol, trial.canonical_params,
        signal_1m=sig_1m, calendar=calendar,
    )
    spec = _spec(trial, recon, exec_bars, policy)
    runner = _RollAwareCliRunner(
        CliBacktestRunner(cli, work_dir, roll_close_marks_path=marks_path),
        tuple(roll_ts),
    )
    engine = ValidationEngine(
        spec, policy, runner, adapter, exec_bars, recon.contracts,
        calendar=calendar, oos_split_role=SplitRole.VALIDATION,
    )
    report = engine.run_research()

    canonical, neighbours = _canonical_and_neighbour_records(trial, report)
    root_summary = RootRunSummary(
        root_symbol=trial.root_symbol,
        strategy_fingerprint=trial.strategy_fingerprint,
        schedule_hash=report.target_schedule_hash,
        n_trading_days=report.oos_metrics.n_trading_days,
        n_trades=report.oos_metrics.n_trades,
        oos_net_pnl_usd=report.oos_metrics.oos_net_pnl_usd,
        oos_daily_sharpe=report.oos_metrics.daily_sharpe,
        oos_annualized_sharpe=report.oos_metrics.annualized_sharpe,
        is_positive=report.oos_metrics.oos_net_pnl_usd > 0.0,
    )
    regime = _regime_evidence(report, adapter, policy)
    n_cpp = getattr(runner, "_counter", 0)
    return TrialRun(
        family_key=trial.family_key,
        root_symbol=trial.root_symbol,
        strategy_fingerprint=trial.strategy_fingerprint,
        strategy_id=trial.strategy_id,
        report=report,
        canonical_params=trial.canonical_params,
        neighbour_records=neighbours,
        canonical_record=canonical,
        root_summary=root_summary,
        regime=regime,
        runtime_s=time.monotonic() - t0,
        n_cpp_runs=int(n_cpp),
        roll_close_marks_sha256=marks_sha,
        n_roll_close_marks=len(marks),
    )


# --------------------------------------------------------------------------
# global multiple-testing family + re-derived headline verdict
# --------------------------------------------------------------------------
@dataclass
class GlobalFamily:
    family: MultipleTestingFamily
    fdr_q_threshold: float
    canonical_index: dict[str, int]            # strategy_fingerprint -> index in family.trials

    def q_value(self, fp: str) -> float:
        fdr = self.family.benjamini_hochberg(self.fdr_q_threshold)
        return fdr.decisions[self.canonical_index[fp]].q_value

    def fdr(self):
        return self.family.benjamini_hochberg(self.fdr_q_threshold)


def build_global_family(runs: list[TrialRun], *, policy: ReliabilityPolicy) -> GlobalFamily:
    from alpha_agent.validation.phase_13_5c_trials import build_global_trial_family

    recs: list[TrialRecord] = []
    root_of: dict[str, str] = {}
    for r in runs:
        recs.append(r.canonical_record)
        root_of[r.strategy_fingerprint] = r.root_symbol
        for nb in r.neighbour_records:
            recs.append(nb)
            root_of.setdefault(nb.strategy_fingerprint, r.root_symbol)
    fam = build_global_trial_family(recs, family_id=MT_FAMILY_GLOBAL, root_of=root_of)
    canon_idx = {
        t.strategy_fingerprint: i
        for i, t in enumerate(fam.trials)
        if t.role == "canonical"
    }
    return GlobalFamily(fam, policy.fdr_q_threshold, canon_idx)


def _global_dsr(run: TrialRun, gf: GlobalFamily) -> DeflatedSharpeResult:
    return deflated_sharpe_ratio(
        run.report.oos_daily.returns_array(),
        n_trials=gf.family.n_trials(),
        trial_daily_sharpes=gf.family.observed_daily_sharpes(),
    )


def global_verdict(run: TrialRun, gf: GlobalFamily, policy: ReliabilityPolicy):
    """The frozen :class:`ReliabilityPolicy` re-applied with the CORRECTED global
    107-hypothesis BH/FDR family and the global DSR effective-trial count
    (sections 11, 12, 17). Every other gate input is the per-trial engine
    report."""
    rep = run.report
    fdr = gf.fdr()
    idx = gf.canonical_index[run.strategy_fingerprint]
    dsr = _global_dsr(run, gf)
    outcome = evaluate_policy(
        policy,
        oos_metrics=rep.oos_metrics,
        walk_forward=rep.walk_forward,
        null_results=list(rep.null_results),
        fdr_result=fdr,
        canonical_trial_index=idx,
        dsr_result=dsr,
        cost_report=rep.cost_stress,
        parameter_stability=rep.parameter_stability,
        ablation_report=rep.ablation_report,
        regime_result=run.regime,
        cross_market=rep.cross_market,
    )
    return outcome, fdr.decisions[idx].q_value, dsr


# --------------------------------------------------------------------------
# cross-market evidence (section 16) -- descriptive only, no new TrialRecords
# --------------------------------------------------------------------------
def cross_market_by_family(runs: list[TrialRun]) -> dict[str, CrossMarketEvidence]:
    out: dict[str, CrossMarketEvidence] = {}
    for fam in ("tsmom", "ma_trend", "breakout", "mean_reversion"):
        summaries = [r.root_summary for r in runs if r.family_key == fam]
        out[fam] = evaluate_cross_market(summaries, concentration_threshold=0.9)
    out["silver_bullet"] = CrossMarketEvidence(status=EvidenceStatus.NOT_EVALUATED)
    return out


# --------------------------------------------------------------------------
# top-level matrix run
# --------------------------------------------------------------------------
@dataclass
class MatrixResult:
    manifest: CandidateManifest
    policy: ReliabilityPolicy
    canonical_runs: list[TrialRun]
    global_family: GlobalFamily
    sensitivity_runs: list[TrialRun]
    sensitivity_global_family: GlobalFamily | None
    cross_market: dict[str, CrossMarketEvidence]
    started_at: str
    finished_at: str
    total_cpp_runs: int
    failures: list[dict] = field(default_factory=list)


def run_matrix(
    *,
    roots: tuple[str, ...] | None = None,
    work_root: Path,
    cli: str = DEFAULT_CLI,
    run_sensitivity: bool = True,
    manifest: CandidateManifest | None = None,
    progress: Callable[[str], None] | None = None,
) -> MatrixResult:
    """Run the full Phase 13.5C matrix: every canonical trial, the global
    corrected BH/FDR family + re-derived verdicts, cross-market evidence, and
    (optionally) the degraded-days-excluded sensitivity pass."""
    import shutil

    roots = tuple(roots) if roots else ROOTS
    say = progress or (lambda _m: None)
    manifest = manifest or build_candidate_manifest()

    def _sweep(wd: Path) -> None:
        # per-trial CLI bundles (bars.csv ~45 MB each x ~13 runs) are large and
        # fully reproducible -- drop them once the TrialRun has captured its
        # results. Failed dirs are kept for debugging.
        try:
            shutil.rmtree(wd, ignore_errors=True)
        except OSError:
            pass
    policy = frozen_policy()
    cal = default_calendar()
    started = datetime.now(UTC).isoformat()

    recon_cache: dict[str, ReconstitutedRoot] = {}

    def _recon(root: str) -> ReconstitutedRoot:
        if root not in recon_cache:
            say(f"reconstituting {root} (offline)...")
            recon_cache[root] = reconstitute_root(root, calendar=cal)
            mx = int(recon_cache[root].canonical_bars["ts_event_ns"].max())
            if mx >= HOLDOUT_START_NS:
                raise RuntimeError(f"{root}: reconstituted bars reach the 2025 holdout ({mx})")
        return recon_cache[root]

    trials = [t for t in manifest.trials if t.root_symbol in roots]

    canonical_runs: list[TrialRun] = []
    failures: list[dict] = []
    for t in trials:
        say(f"[canonical] {t.root_symbol} {t.family_key} ...")
        wd = work_root / "canonical" / f"{t.root_symbol}__{t.family_key}"
        try:
            run = run_trial(
                t, _recon(t.root_symbol), policy=policy,
                work_dir=wd, cli=cli, calendar=cal,
            )
            canonical_runs.append(run)
            _sweep(wd)
            say(f"    verdict(local)={run.report.verdict.value} "
                f"net=${run.report.oos_metrics.oos_net_pnl_usd:,.0f} "
                f"dSR={run.report.oos_metrics.daily_sharpe:.3f} "
                f"cpp_runs={run.n_cpp_runs} {run.runtime_s:.0f}s")
        except Exception as exc:  # noqa: BLE001 -- preserved as a typed failed result
            failures.append({
                "phase": "canonical", "family_key": t.family_key,
                "root_symbol": t.root_symbol, "error_type": type(exc).__name__,
                "error": str(exc),
            })
            say(f"    FAILED: {type(exc).__name__}: {exc}")

    gf = build_global_family(canonical_runs, policy=policy)

    sensitivity_runs: list[TrialRun] = []
    sens_gf: GlobalFamily | None = None
    if run_sensitivity and canonical_runs:
        for t in trials:
            excl = set(degraded_trading_days(t.root_symbol, calendar=cal))
            say(f"[sensitivity] {t.root_symbol} {t.family_key} (-{len(excl)} degraded days) ...")
            wd = work_root / "sensitivity" / f"{t.root_symbol}__{t.family_key}"
            try:
                run = run_trial(
                    t, _recon(t.root_symbol), policy=policy,
                    work_dir=wd, cli=cli, exclude_trading_days=excl, calendar=cal,
                )
                sensitivity_runs.append(run)
                _sweep(wd)
            except Exception as exc:  # noqa: BLE001
                failures.append({
                    "phase": "sensitivity", "family_key": t.family_key,
                    "root_symbol": t.root_symbol, "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                say(f"    FAILED: {type(exc).__name__}: {exc}")
        if sensitivity_runs:
            sens_gf = build_global_family(sensitivity_runs, policy=policy)

    total_cpp = sum(r.n_cpp_runs for r in canonical_runs) + sum(r.n_cpp_runs for r in sensitivity_runs)
    return MatrixResult(
        manifest=manifest,
        policy=policy,
        canonical_runs=canonical_runs,
        global_family=gf,
        sensitivity_runs=sensitivity_runs,
        sensitivity_global_family=sens_gf,
        cross_market=cross_market_by_family(canonical_runs),
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        total_cpp_runs=total_cpp,
        failures=failures,
    )
