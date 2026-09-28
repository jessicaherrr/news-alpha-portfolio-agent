"""``ValidationEngine`` -- the real-path orchestrator (sections 5-18).

It drives the official C++ backtest path for:

* one causal out-of-sample backtest of the PREDECLARED strategy over the OOS span
  (headline daily series -> metrics / bootstrap / null / DSR);
* ``k`` causal walk-forward folds (fold consistency, positive-fold count);
* the predeclared cost-stress scenarios (each a C++ rerun with scaled costs);
* the predeclared parameter neighbourhood (each neighbour its own spec + trial);
* the predeclared ablations (each its own spec + trial);
* the schedule time-shift null (each shifted schedule a C++ rerun);
* optional cross-market per-root runs supplied by the caller.

It never computes PnL itself. Feature computation + DSL compilation + target-
schedule construction is delegated to a :class:`StrategyFamilyAdapter`; the
economic result of every run comes from :class:`~.runner.BacktestRunner`.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from alpha_agent.backtest.targets import TargetSchedule
from alpha_agent.data.calendars import SessionCalendar, default_calendar
from alpha_agent.strategy import compile_strategy, strategy_fingerprint
from alpha_agent.strategy.spec import StrategySpec
from alpha_agent.validation.ablation import AblationResult, AblationSpec
from alpha_agent.validation.assemble import assemble_report
from alpha_agent.validation.cost_stress import CostScenarioResult
from alpha_agent.validation.crossmarket import RootRunSummary
from alpha_agent.validation.dataset import bars_in_window, sorted_unique_bar_ts
from alpha_agent.validation.enums import NullMethod, SplitRole
from alpha_agent.validation.metrics import daily_sharpe
from alpha_agent.validation.multiple_testing import MultipleTestingFamily, TrialRecord
from alpha_agent.validation.nulls import (
    causal_segment_ids,
    circular_permuted_schedules,
    common_support_control_schedule,
    empirical_p_value,
    shifted_schedules,
)
from alpha_agent.validation.policy import ReliabilityPolicy
from alpha_agent.validation.regime import RegimeLabelling
from alpha_agent.validation.report import ValidationReport
from alpha_agent.validation.returns import DailyReturnSeries
from alpha_agent.validation.runner import BacktestRun, BacktestRunner
from alpha_agent.validation.spec import ValidationSpec
from alpha_agent.validation.stability import NeighbourResult
from alpha_agent.validation.trading_day import ValidationDayPlan, build_validation_day_plan
from alpha_agent.validation.walkforward import (
    WalkForwardFoldResult,
    build_folds,
    summarize_walk_forward,
)


class StrategyFamilyAdapter(Protocol):
    strategy_key: str
    canonical_params: dict

    def spec_for(self, params: dict) -> StrategySpec: ...

    def schedule_for(
        self, spec: StrategySpec, bars: pd.DataFrame, *, emit_from_ts_ns: int
    ) -> TargetSchedule: ...


class DslFamilyAdapter:
    """A :class:`StrategyFamilyAdapter` for any Phase 10/11/12 DSL strategy.

    ``spec_factory(params) -> StrategySpec`` is the family's typed factory
    (``make_tsmom_spec``, ``make_silver_bullet_spec``, ...). ``compute_frame``
    turns a bar window into the point-in-time ``FeatureFrame`` the compiled plan
    needs (usually ``lambda plan, bars: compute_features(source(bars), specs)``).
    """

    def __init__(
        self,
        strategy_key: str,
        canonical_params: dict,
        spec_factory: Callable[[dict], StrategySpec],
        compute_frame: Callable[[object, pd.DataFrame], object],
    ):
        self.strategy_key = strategy_key
        self.canonical_params = canonical_params
        self._spec_factory = spec_factory
        self._compute_frame = compute_frame

    def spec_for(self, params: dict) -> StrategySpec:
        return self._spec_factory(params)

    def schedule_for(
        self, spec: StrategySpec, bars: pd.DataFrame, *, emit_from_ts_ns: int
    ) -> TargetSchedule:
        from alpha_agent.backtest.targets import build_target_schedule

        plan = compile_strategy(spec)
        frame = self._compute_frame(plan, bars)
        full = build_target_schedule(plan, frame)
        kept = tuple(r for r in full.rows if r.ts_event_ns >= emit_from_ts_ns)
        return full.model_copy(update={"rows": kept})


def _median_bar_span_ns(bars: pd.DataFrame) -> int:
    ts = np.asarray(sorted_unique_bar_ts(bars), dtype=np.int64)
    if ts.size < 2:
        return 0
    return int(np.median(np.diff(ts)))


class ValidationEngine:
    def __init__(
        self,
        spec: ValidationSpec,
        policy: ReliabilityPolicy,
        runner: BacktestRunner,
        family: StrategyFamilyAdapter,
        bars: pd.DataFrame,
        contracts: pd.DataFrame,
        *,
        ablations: list[AblationSpec] | None = None,
        regime_labelling: RegimeLabelling | None = None,
        cross_market_runs: list[RootRunSummary] | None = None,
        calendar: SessionCalendar | None = None,
        oos_split_role: SplitRole | None = None,
        headline_trades_out_path: str | Path | None = None,
        headline_fills_out_path: str | Path | None = None,
    ):
        if policy.identity() != spec.reliability_policy_fingerprint:
            raise ValueError(
                "ValidationSpec.reliability_policy_fingerprint does not match the policy "
                "passed to the engine -- the policy must be fixed before the run"
            )
        self.spec = spec
        self.policy = policy
        self.runner = runner
        self.family = family
        self.bars = bars.sort_values("ts_event_ns").reset_index(drop=True)
        self.contracts = contracts
        self.ablations = list(ablations or [])
        self.regime_labelling = regime_labelling
        self.cross_market_runs = list(cross_market_runs or [])
        self._bar_span_ns = _median_bar_span_ns(self.bars)
        self._calendar = calendar or default_calendar()
        self._root = spec.dataset.root_symbol
        # ADDITIVE, opt-in (Phase 13.5C). When set, the headline OOS run + its
        # null / cost-stress / neighbourhood are evaluated over THIS split
        # window (typically VALIDATION 2023-2024) while the walk-forward folds
        # are built over the remaining research windows only (TRAIN 2018-2022).
        # None => the frozen Phase 13 behaviour (headline OOS == the fold span
        # over the whole research span). It never selects the holdout window.
        self._oos_split_role = oos_split_role
        # ADDITIVE, opt-in (Alpha Discovery live-research campaign, Checkpoint
        # 11). `None` (every existing caller) changes nothing at all -- these
        # paths are threaded ONLY to the single headline `_run_span` call
        # below, never to any fold / null / cost-stress / neighbour /
        # ablation run, so no internal-only artifact can ever be mistaken for
        # the headline strategy path.
        self._headline_trades_out_path = headline_trades_out_path
        self._headline_fills_out_path = headline_fills_out_path
        if oos_split_role is not None:
            if oos_split_role == SplitRole.LOCKED_HOLDOUT:
                raise ValueError("oos_split_role must not be the LOCKED_HOLDOUT window")
            if not spec.split_plan.has_role(oos_split_role):
                raise ValueError(f"split_plan has no {oos_split_role} window")

    def _day_plan(self, exec_bars: pd.DataFrame) -> ValidationDayPlan:
        return build_validation_day_plan(
            exec_bars, root_symbol=self._root, calendar=self._calendar
        )

    # -- internal helpers -------------------------------------------------
    def _run_span(
        self, params: dict, warm_start_ts: int, emit_from_ts: int, end_ts: int,
        *, commission: float, slippage: float, spread: float,
        trades_out_path: str | Path | None = None, fills_out_path: str | Path | None = None,
    ) -> tuple[BacktestRun, TargetSchedule]:
        """One causal backtest: features + schedule computed on
        ``[warm_start_ts, end_ts)`` (feature warm-up only), the strategy emits
        no decision before ``emit_from_ts``, and the C++ engine executes over
        ``[emit_from_ts, end_ts)`` so no warm-up bar carries an economic
        position or a return observation (sections 6, 7).

        ``trades_out_path`` / ``fills_out_path`` (Alpha Discovery live-
        research campaign, Checkpoint 11): ``None`` for every caller except
        the ONE headline invocation in ``_run`` -- see that call site."""
        feat_window = bars_in_window(self.bars, warm_start_ts, end_ts)
        exec_window = bars_in_window(self.bars, emit_from_ts, end_ts)
        spec_obj = self.family.spec_for(params)
        schedule = self.family.schedule_for(spec_obj, feat_window, emit_from_ts_ns=emit_from_ts)
        run = self.runner.run(
            schedule=schedule, bars=exec_window, contracts=self.contracts,
            commission_per_contract_usd=commission, slippage_ticks=slippage,
            spread_ticks=spread, capital_base_usd=self.spec.capital_base_usd,
            validation_day_plan=self._day_plan(exec_window),
            trades_out_path=trades_out_path, fills_out_path=fills_out_path,
        )
        return run, schedule

    def _oos_span(self, research_span: tuple[int, int]) -> tuple[int, int, int, tuple]:
        folds = build_folds(research_span, self.spec.walk_forward, bar_span_ns=self._bar_span_ns)
        if not folds:
            # degenerate: no room for a fold -> OOS = the validation-role window
            if self.spec.split_plan.has_role(SplitRole.VALIDATION):
                w = self.spec.split_plan.window(SplitRole.VALIDATION)
                return w.start_ts_ns, w.start_ts_ns, w.end_ts_ns, ()
            s, e = research_span
            mid = (s + e) // 2
            return mid, mid, e, ()
        warm = folds[0].warmup_start_ts_ns
        emit = folds[0].test_start_ts_ns
        end = folds[-1].test_end_ts_ns
        return warm, emit, end, folds

    # -- public: research validation ------------------------------------
    def run_research(self) -> ValidationReport:
        return self._run(holdout=False)

    # -- public: single holdout evaluation ----------------------------
    def run_holdout(self) -> ValidationReport:
        return self._run(holdout=True)

    def _run(self, *, holdout: bool) -> ValidationReport:
        ref_c, ref_s, ref_sp = 2.0, 0.0, 0.0
        canonical_params = self.family.canonical_params

        if holdout:
            w = self.spec.split_plan.holdout_window()
            warm = max(0, w.start_ts_ns - self._bar_span_ns * self.spec.walk_forward.warmup_bars)
            oos_warm, oos_emit, oos_end = warm, w.start_ts_ns, w.end_ts_ns
            folds: tuple = ()
        elif self._oos_split_role is not None:
            # Phase 13.5C: headline OOS = the named split window; folds over the
            # other (earlier) research windows only.
            w = self.spec.split_plan.window(self._oos_split_role)
            warm = max(
                0, w.start_ts_ns - self._bar_span_ns * self.spec.walk_forward.warmup_bars
            )
            oos_warm, oos_emit, oos_end = warm, w.start_ts_ns, w.end_ts_ns
            train_windows = [
                x for x in self.spec.split_plan.windows
                if x.role != self._oos_split_role and x.role != SplitRole.LOCKED_HOLDOUT
            ]
            if train_windows:
                train_span = (
                    min(x.start_ts_ns for x in train_windows),
                    max(x.end_ts_ns for x in train_windows),
                )
                folds = build_folds(
                    train_span, self.spec.walk_forward, bar_span_ns=self._bar_span_ns
                )
            else:
                folds = ()
        else:
            research_span = self.spec.split_plan.research_span_ns()
            oos_warm, oos_emit, oos_end, folds = self._oos_span(research_span)

        # -- headline OOS run (canonical, predeclared spec) --
        # The opt-in headline trades/fills export is REFUSED for a holdout
        # run even if a caller mistakenly supplied one -- `run_holdout` (2025)
        # must never write an artifact file, on top of `Production
        # ExecutionValidationService.run` already refusing `holdout=True`
        # outright before an engine is ever constructed.
        headline_run, headline_sched = self._run_span(
            canonical_params, oos_warm, oos_emit, oos_end,
            commission=ref_c, slippage=ref_s, spread=ref_sp,
            trades_out_path=None if holdout else self._headline_trades_out_path,
            fills_out_path=None if holdout else self._headline_fills_out_path,
        )
        combined_oos_daily: DailyReturnSeries = headline_run.daily
        statfn = daily_sharpe

        # -- walk-forward folds --
        fold_results: list[WalkForwardFoldResult] = []
        for f in folds:
            frun, _ = self._run_span(
                canonical_params, f.warmup_start_ts_ns, f.test_start_ts_ns, f.test_end_ts_ns,
                commission=ref_c, slippage=ref_s, spread=ref_sp,
            )
            fold_results.append(
                WalkForwardFoldResult(
                    fold_index=f.fold_index,
                    test_start_ts_ns=f.test_start_ts_ns,
                    test_end_ts_ns=f.test_end_ts_ns,
                    n_trading_days=frun.daily.n_days,
                    n_fills=frun.n_fills,
                    n_trades=frun.n_trades,
                    oos_net_pnl_usd=frun.net_pnl_usd,
                    daily_sharpe=frun.daily_sharpe,
                    is_positive=frun.net_pnl_usd > 0.0,
                    daily=frun.daily,
                )
            )
        wf = summarize_walk_forward(self.spec.walk_forward, len(folds), fold_results)

        # -- schedule time-shift null (DIAGNOSTIC-ONLY, Phase 13.2). Proper
        #    per-segment common support: a COMMON-SUPPORT control schedule (rows
        #    too near a segment end pre-dropped) is shifted by every k in
        #    [min_shift_bars, max_shift_bars]; the control and every replicate
        #    share one opportunity set. The observed statistic for the p-value is
        #    the CONTROL run's statistic -- never the full-sample canonical one.
        exec_window = bars_in_window(self.bars, oos_emit, oos_end)
        exec_day_plan = self._day_plan(exec_window)
        bar_ts = sorted_unique_bar_ts(exec_window)
        seg_ids = causal_segment_ids(
            exec_window, root_symbol=self._root, calendar=self._calendar,
            gap_multiple=self.spec.null_test.gap_multiple,
        )

        def _run_null_schedule(sched: TargetSchedule) -> float:
            nrun = self.runner.run(
                schedule=sched, bars=exec_window, contracts=self.contracts,
                commission_per_contract_usd=ref_c, slippage_ticks=ref_s, spread_ticks=ref_sp,
                capital_base_usd=self.spec.capital_base_usd,
                validation_day_plan=exec_day_plan,
            )
            return float(statfn(nrun.daily.returns_array()))

        control_sched = common_support_control_schedule(
            headline_sched, bar_ts, seg_ids, self.spec.null_test
        )
        shift_observed_stat: float | None = None
        null_stats: list[float] = []
        perm_stats: list[float] = []
        if control_sched.rows and NullMethod.SCHEDULE_TIME_SHIFT in self.spec.null_test.methods:
            control_run = self.runner.run(
                schedule=control_sched, bars=exec_window, contracts=self.contracts,
                commission_per_contract_usd=ref_c, slippage_ticks=ref_s, spread_ticks=ref_sp,
                capital_base_usd=self.spec.capital_base_usd,
                validation_day_plan=exec_day_plan,
            )
            shift_observed_stat = float(statfn(control_run.daily.returns_array()))
            for _k, sched in shifted_schedules(control_sched, bar_ts, seg_ids, self.spec.null_test):
                null_stats.append(_run_null_schedule(sched))
        if control_sched.rows and NullMethod.CIRCULAR_SCHEDULE_PERMUTATION in self.spec.null_test.methods:
            for _k, sched in circular_permuted_schedules(control_sched, bar_ts, self.spec.null_test):
                perm_stats.append(_run_null_schedule(sched))
        null_arr = np.asarray(null_stats, dtype=float)
        perm_arr = np.asarray(perm_stats, dtype=float)

        # -- cost stress (C++ reruns with scaled costs) --
        cost_results: list[CostScenarioResult] = []
        for sc in self.spec.cost_stress.scenarios:
            comm, slip, spr = sc.resolve(self.spec.cost_stress)
            crun, _ = self._run_span(
                canonical_params, oos_warm, oos_emit, oos_end,
                commission=comm, slippage=slip, spread=spr,
            )
            base_net = headline_run.net_pnl_usd
            ratio = (
                crun.net_pnl_usd / base_net if abs(base_net) > 1e-9 else float("nan")
            )
            cost_results.append(
                CostScenarioResult(
                    label=sc.label,
                    commission_per_contract_usd=comm,
                    slippage_ticks=slip,
                    spread_ticks=spr,
                    n_fills=crun.n_fills,
                    oos_gross_pnl_usd=crun.gross_pnl_usd,
                    oos_costs_usd=crun.costs_usd,
                    oos_net_pnl_usd=crun.net_pnl_usd,
                    daily_sharpe=crun.daily_sharpe,
                    annualized_sharpe=crun.annualized_sharpe,
                    net_pnl_ratio_vs_baseline=ratio,
                    sharpe_delta_vs_baseline=crun.daily_sharpe - headline_run.daily_sharpe,
                )
            )

        # -- parameter neighbourhood (each neighbour its own spec + trial) --
        neighbour_results: list[NeighbourResult] = []
        trial_records: list[TrialRecord] = []
        canonical_fp = strategy_fingerprint(self.family.spec_for(canonical_params))
        # the canonical trial's family p-value is the GATING (centered block
        # bootstrap) null on the full-sample OOS returns -- consistent with the
        # neighbour / ablation trial p-values and never the diagnostic
        # schedule-shift null (Phase 13.2).
        canonical_null_p = _neighbour_p(combined_oos_daily.returns_array(), self.spec, statfn)
        trial_records.append(
            TrialRecord(
                label="canonical",
                role="canonical",
                strategy_fingerprint=canonical_fp,
                schedule_hash=headline_sched.schedule_hash(),
                p_value=canonical_null_p,
                null_tested=True,
                observed_daily_sharpe=headline_run.daily_sharpe,
                observed_net_pnl_usd=headline_run.net_pnl_usd,
            )
        )

        nbhd = self.spec.parameter_neighbourhood
        if nbhd is not None:
            for i, params in enumerate(nbhd.neighbour_params):
                nrun, nsched = self._run_span(
                    params, oos_warm, oos_emit, oos_end,
                    commission=ref_c, slippage=ref_s, spread=ref_sp,
                )
                fp = strategy_fingerprint(self.family.spec_for(params))
                p_n = _neighbour_p(nrun.daily.returns_array(), self.spec, statfn)
                neighbour_results.append(
                    NeighbourResult(
                        params_label=f"neighbour_{i}",
                        strategy_fingerprint=fp,
                        is_canonical=False,
                        n_trades=nrun.n_trades,
                        oos_net_pnl_usd=nrun.net_pnl_usd,
                        oos_daily_sharpe=nrun.daily_sharpe,
                        oos_annualized_sharpe=nrun.annualized_sharpe,
                    )
                )
                trial_records.append(
                    TrialRecord(
                        label=f"neighbour_{i}", role="neighbour",
                        strategy_fingerprint=fp, schedule_hash=nsched.schedule_hash(),
                        p_value=p_n, null_tested=True,
                        observed_daily_sharpe=nrun.daily_sharpe,
                        observed_net_pnl_usd=nrun.net_pnl_usd,
                    )
                )
            # canonical also participates in the neighbourhood stability stats
            neighbour_results.insert(
                0,
                NeighbourResult(
                    params_label="canonical", strategy_fingerprint=canonical_fp,
                    is_canonical=True, n_trades=headline_run.n_trades,
                    oos_net_pnl_usd=headline_run.net_pnl_usd,
                    oos_daily_sharpe=headline_run.daily_sharpe,
                    oos_annualized_sharpe=headline_run.annualized_sharpe,
                ),
            )

        # -- ablations (each its own spec + trial) --
        ablation_results: list[AblationResult] = []
        for ab in self.ablations:
            params = {**canonical_params, **ab.param_overrides}
            arun, asched = self._run_span(
                params, oos_warm, oos_emit, oos_end,
                commission=ref_c, slippage=ref_s, spread=ref_sp,
            )
            fp = strategy_fingerprint(self.family.spec_for(params))
            ablation_results.append(
                AblationResult(
                    label=ab.label, kind=ab.kind, ablation_fingerprint=ab.identity(),
                    strategy_fingerprint=fp, n_trades=arun.n_trades,
                    oos_net_pnl_usd=arun.net_pnl_usd,
                    oos_daily_sharpe=arun.daily_sharpe,
                    oos_annualized_sharpe=arun.annualized_sharpe,
                    sharpe_delta_vs_canonical=headline_run.daily_sharpe - arun.daily_sharpe,
                )
            )
            trial_records.append(
                TrialRecord(
                    label=ab.label, role="ablation", strategy_fingerprint=fp,
                    schedule_hash=asched.schedule_hash(),
                    p_value=_neighbour_p(arun.daily.returns_array(), self.spec, statfn),
                    null_tested=True,
                    observed_daily_sharpe=arun.daily_sharpe,
                    observed_net_pnl_usd=arun.net_pnl_usd,
                )
            )

        for rm in self.cross_market_runs:
            trial_records.append(
                TrialRecord(
                    label=f"root_{rm.root_symbol}", role="cross_market",
                    strategy_fingerprint=rm.strategy_fingerprint,
                    schedule_hash=rm.schedule_hash,
                    p_value=1.0, null_tested=False,
                    observed_daily_sharpe=rm.oos_daily_sharpe,
                    observed_net_pnl_usd=rm.oos_net_pnl_usd,
                )
            )

        trial_family = MultipleTestingFamily(
            family_id=self.spec.trial_family_id, trials=tuple(trial_records)
        )

        return assemble_report(
            self.spec,
            self.policy,
            walk_forward=wf,
            combined_oos_daily=combined_oos_daily,
            oos_n_fills=headline_run.n_fills,
            oos_n_trades=headline_run.n_trades,
            oos_gross_pnl_usd=headline_run.gross_pnl_usd,
            oos_costs_usd=headline_run.costs_usd,
            oos_net_pnl_usd=headline_run.net_pnl_usd,
            trial_family=trial_family,
            schedule_shift_null_stats=null_arr,
            schedule_shift_observed_stat=shift_observed_stat,
            circular_permutation_null_stats=perm_arr if perm_arr.size else None,
            cost_scenario_results=cost_results,
            neighbour_results=neighbour_results or None,
            ablation_results=ablation_results or None,
            regime_labelling=self.regime_labelling,
            cross_market_summaries=self.cross_market_runs or None,
            target_schedule_hash=headline_sched.schedule_hash(),
            holdout_evaluated=holdout,
        )


def _neighbour_p(returns: np.ndarray, spec: ValidationSpec, statfn) -> float:
    from alpha_agent.validation.nulls import centered_block_bootstrap_null_stats

    r = np.asarray(returns, dtype=float)
    if r.size < 3:
        return 1.0
    cbb = centered_block_bootstrap_null_stats(r, spec.null_test, statfn)
    p, _ = empirical_p_value(float(statfn(r)), cbb)
    return p
