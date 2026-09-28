"""The REAL Phase 15B data path -- constructed only under ``--run-real``.

Nothing in Phase 15B.1 imports this module during a synthetic run. It wires the
three real providers the orchestrator needs:

* :class:`RealPrimaryProvider` -- the frozen Phase 13.5C primary target schedules,
  compiled through the existing ``_Phase135cAdapter`` on the offline
  reconstituted 2018-2024 CME data;
* :class:`Phase135cFeatureProvider` -- the frozen ordered Phase 15 feature set,
  computed once per root through the Feature Engine on the daily signal series;
* :class:`MultiRootCppEngineRunner` -- one ``quant_backtest_targets_csv``
  subprocess per (schedule, cost scenario), dispatched by the schedule's root,
  with the additive ``--trades-out`` / ``--fills-out`` audit exports and the
  frozen roll-close auxiliary-mark path.

STRICTLY OFFLINE. The reconstitution decodes local DBN bytes; it never builds a
Databento client or calls a cost / range endpoint, and every window ends at the
exclusive ``2025-01-01`` boundary. Any 2025 access fails loudly upstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from alpha_agent.backtest.targets import TargetSchedule
from alpha_agent.ml.corpus import PrimaryScope
from alpha_agent.ml.engine_io import (
    CAPITAL_BASE_USD,
    CPP_ENGINE_TAG,
    CppEngineRunner,
    EngineRunResult,
)
from alpha_agent.ml.feature_matrix import Phase135cFeatureProvider
from alpha_agent.ml.guards import DEVELOPMENT_CORPUS_START_NS
from alpha_agent.ml.manifest import ML_FEATURE_SET, PRIMARY_FAMILIES, ROOTS
from alpha_agent.strategy.candidates_phase_13_5c import baseline_params, baseline_spec
from alpha_agent.strategy.fingerprint import strategy_fingerprint

_DEFAULT_CLI = "build/cpp/cpp/quant_backtest_targets_csv"


class RealPrimaryProvider:
    """Frozen Phase 13.5C primary schedules over the 2018-2024 corpus."""

    provider_tag = "phase_13_5c_real_primary_provider"

    def __init__(self, repo: Path, *, families: tuple[str, ...] = PRIMARY_FAMILIES,
                 roots: tuple[str, ...] = ROOTS):
        self._repo = repo
        self._families = families
        self._roots = roots
        self._adapters: dict[tuple[str, str], object] = {}
        self._exec_bars: dict[str, pd.DataFrame] = {}
        self._recon: dict[str, object] = {}
        self._scopes = tuple(
            PrimaryScope(
                primary_family=f,
                root_symbol=r,
                strategy_fingerprint=strategy_fingerprint(baseline_spec(f, r)),
                strategy_id=baseline_spec(f, r).strategy_id,
            )
            for f in families
            for r in roots
        )
        self._prepared = False

    def _prepare(self) -> None:
        if self._prepared:
            return
        from alpha_agent.data.calendars import default_calendar
        from alpha_agent.data.real_market_dataset import (
            RESEARCH_WINDOW,
            VALIDATION_WINDOW,
            _iso_ns,
            execution_bars,
            reconstitute_root,
        )
        from alpha_agent.validation.phase_13_5c_matrix import _Phase135cAdapter

        cal = default_calendar()
        lo = _iso_ns(RESEARCH_WINDOW[0])
        hi = _iso_ns(VALIDATION_WINDOW[1])          # 2025-01-01 exclusive
        for root in self._roots:
            recon = reconstitute_root(root, calendar=cal)
            self._recon[root] = recon
            bars = execution_bars(recon, lo, hi)
            self._exec_bars[root] = bars
            signal_1m = recon.forward_adjusted
            for family in self._families:
                self._adapters[(family, root)] = _Phase135cAdapter(
                    family, root, baseline_params(family, root),
                    signal_1m=signal_1m, calendar=cal,
                )
        self._prepared = True

    def scopes(self) -> tuple[PrimaryScope, ...]:
        return self._scopes

    def exec_bars(self, root: str) -> pd.DataFrame:
        self._prepare()
        return self._exec_bars[root]

    def reconstituted(self, root: str):
        self._prepare()
        return self._recon[root]

    def schedule_for(self, scope: PrimaryScope) -> TargetSchedule:
        self._prepare()
        adapter = self._adapters[(scope.primary_family, scope.root_symbol)]
        spec = baseline_spec(scope.primary_family, scope.root_symbol)
        bars = self._exec_bars[scope.root_symbol]
        sched = adapter.schedule_for(spec, bars, emit_from_ts_ns=DEVELOPMENT_CORPUS_START_NS)
        if sched is None:
            raise ValueError(
                f"no primary schedule for {scope.primary_family}/{scope.root_symbol}"
            )
        return sched

    def feature_availability(self, scope: PrimaryScope, episodes) -> dict[int, bool]:
        # the FEATURE provider decides availability at build time; a missing
        # feature there returns None and the pooled matrix records the typed
        # exclusion, so nothing extra is refused here.
        return {ep.episode_index: True for ep in episodes}


class MultiRootCppEngineRunner:
    """Dispatches ``engine.run(schedule, ...)`` to a per-root C++ runner."""

    engine_tag = CPP_ENGINE_TAG

    def __init__(self, per_root: dict[str, CppEngineRunner]):
        self._per_root = per_root

    def run(self, schedule: TargetSchedule, *, cost_scenario_label: str,
            cost_multiplier: float, want_audit_trails: bool) -> EngineRunResult:
        runner = self._per_root.get(schedule.root_symbol)
        if runner is None:
            raise KeyError(f"no C++ engine runner for root {schedule.root_symbol}")
        return runner.run(
            schedule,
            cost_scenario_label=cost_scenario_label,
            cost_multiplier=cost_multiplier,
            want_audit_trails=want_audit_trails,
        )


@dataclass
class RealPath:
    primary_provider: RealPrimaryProvider
    feature_provider: Phase135cFeatureProvider
    engine: MultiRootCppEngineRunner
    #: per-root full FeatureFrame (identifiers realigned to real trading-day ts).
    #: For the offline engineering smoke -- warm-up / finiteness / QA inspection.
    feature_detail: dict = field(default_factory=dict)


def build_real_path(
    repo: Path,
    *,
    cli: str | None = None,
    work_root: Path | None = None,
    roots: tuple[str, ...] = ROOTS,
    primary: RealPrimaryProvider | None = None,
) -> RealPath:
    """Construct the real providers. Reconstitutes the offline 2018-2024 data.

    ``roots`` restricts the reconstitution (the full production run uses all
    five); ``primary`` reuses an already-prepared provider to avoid a second
    reconstitution.
    """
    from alpha_agent.data.calendars import default_calendar
    from alpha_agent.data.real_market_dataset import (
        HOLDOUT_START_NS,
        daily_signal_series,
        roll_close_marks,
        write_roll_close_marks_csv,
    )
    from alpha_agent.features.daily import compute_contiguous_daily_features
    from alpha_agent.schemas.market_data import PriceDomain
    from alpha_agent.validation.trading_day import build_validation_day_plan

    executable = repo / (cli or _DEFAULT_CLI)
    work = work_root or (repo / "outputs" / "phase_15" / "_run_real_work")
    work.mkdir(parents=True, exist_ok=True)
    cal = default_calendar()

    if primary is None:
        primary = RealPrimaryProvider(repo, roots=roots)
    primary._prepare()

    feature_frames: dict[str, pd.DataFrame] = {}
    feature_detail: dict = {}
    per_root_engine: dict[str, CppEngineRunner] = {}
    aliases = list(ML_FEATURE_SET.ordered_aliases)
    feature_specs = [f.spec for f in ML_FEATURE_SET.features]
    # compute_features returns feature columns keyed by the ENGINE CANONICAL NAME
    # and ORDERED ALPHABETICALLY -- never in spec order -- so the alias matrix is
    # rebuilt by explicit name lookup, in alias order.
    canonical = list(ML_FEATURE_SET.canonical_feature_names())
    canonical_to_alias = dict(zip(canonical, aliases))

    for root in roots:
        recon = primary.reconstituted(root)
        bars = primary.exec_bars(root)
        daily = daily_signal_series(recon.forward_adjusted, root, calendar=cal)
        # The daily trading-day signal series is ORDINAL -- its wall-clock
        # ts_event_ns jumps every weekend/holiday, which would trip
        # SessionPolicy.RESET_ON_GAP and starve every window longer than a
        # trading week (cum_log_return / realized_vol / vol_percentile). Compute
        # on a gap-free synthetic day index and realign -- the frozen Phase
        # 13.5C baseline-signal technique.
        result = compute_contiguous_daily_features(
            daily,
            feature_specs,
            root_symbol=root,
            price_domain=PriceDomain.BACK_ADJUSTED,
            adjustment_mode="forward_adjusted",
            require_point_in_time=True,
        )
        feats = result.frame
        feature_detail[root] = feats
        feat_cols = feats.features
        missing = [c for c in canonical if c not in feat_cols.columns]
        if missing:
            raise ValueError(
                f"the real feature frame for {root} is missing canonical columns {missing}; "
                f"got {list(feat_cols.columns)} for the frozen 11-feature contract"
            )
        frame = feats.identifiers[["ts_event_ns"]].copy()   # realigned to real trading-day ts
        for c in canonical:                     # alias order, explicit lookup
            frame[canonical_to_alias[c]] = feat_cols[c].to_numpy()
        feature_frames[root] = frame.reset_index(drop=True)

        # canonical CME trading-day plan over the exec window (arg 8 for C++)
        day_plan = build_validation_day_plan(bars, root_symbol=root, calendar=cal)

        # roll close-leg auxiliary marks + the observed roll effective timestamps
        span_lo = int(DEVELOPMENT_CORPUS_START_NS)
        span_hi = int(HOLDOUT_START_NS)
        marks = roll_close_marks(recon, span_lo, span_hi)
        marks_path, _sha = write_roll_close_marks_csv(
            marks, work / root / "roll_close_marks.csv"
        )
        roll_ts = tuple(
            int(rl.effective_ts_ns)
            for rl in recon.rolls
            if span_lo <= int(rl.effective_ts_ns) < span_hi
        )
        per_root_engine[root] = CppEngineRunner(
            executable=executable,
            work_dir=work / root,
            bars=bars,
            contracts=recon.contracts,
            root_symbol=root,
            validation_day_plan=day_plan,
            roll_effective_ts_ns=roll_ts,
            roll_close_marks_path=marks_path,
            capital_base_usd=CAPITAL_BASE_USD,
        )

    return RealPath(
        primary_provider=primary,
        feature_provider=Phase135cFeatureProvider(feature_frames),
        engine=MultiRootCppEngineRunner(per_root_engine),
        feature_detail=feature_detail,
    )
