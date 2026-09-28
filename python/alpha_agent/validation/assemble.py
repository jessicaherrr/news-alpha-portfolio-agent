"""Assemble a :class:`ValidationReport` from the per-run economic evidence.

This is the single place the framework turns official C++ backtest results into
statistics and a code-based verdict. Both callers use it:

* :class:`~alpha_agent.validation.engine.ValidationEngine` -- the real path
  (feeds it fold / cost / neighbour / ablation / cross-market C++ reruns);
* the section-23 statistical fixtures -- feed it synthetic runs to prove the
  statistics framework independently of any market claim.

Everything computed here is pure and deterministic given its inputs.
"""
from __future__ import annotations

import numpy as np

from alpha_agent.validation.ablation import AblationReport, AblationResult, summarize_ablations
from alpha_agent.validation.bootstrap import bootstrap_daily_returns
from alpha_agent.validation.cost_stress import (
    CostScenarioResult,
    CostStressReport,
    summarize_cost_stress,
)
from alpha_agent.validation.crossmarket import (
    CrossMarketEvidence,
    RootRunSummary,
    evaluate_cross_market,
)
from alpha_agent.validation.dsr import deflated_sharpe_ratio
from alpha_agent.validation.enums import NullMethod, NullRole
from alpha_agent.validation.metrics import daily_sharpe, describe_oos, mean_daily_return
from alpha_agent.validation.multiple_testing import MultipleTestingFamily
from alpha_agent.validation.nulls import (
    NullTestResult,
    centered_block_bootstrap_null_stats,
    summarize_null,
)
from alpha_agent.validation.policy import ReliabilityPolicy, evaluate_policy
from alpha_agent.validation.regime import (
    RegimeLabelling,
    RegimeStabilityResult,
    evaluate_regime_stability,
)
from alpha_agent.validation.report import FoldSummary, SampleDiagnostics, ValidationReport
from alpha_agent.validation.returns import DailyReturnSeries
from alpha_agent.validation.spec import ValidationSpec
from alpha_agent.validation.stability import (
    NeighbourResult,
    ParameterStabilityResult,
    summarize_parameter_stability,
)
from alpha_agent.validation.walkforward import WalkForwardResult

_STAT_FNS = {"daily_sharpe": daily_sharpe, "mean_daily_return": mean_daily_return}


def _statfn(name: str):
    return _STAT_FNS.get(name, daily_sharpe)


def assemble_report(
    spec: ValidationSpec,
    policy: ReliabilityPolicy,
    *,
    walk_forward: WalkForwardResult,
    combined_oos_daily: DailyReturnSeries,
    oos_n_fills: int,
    oos_n_trades: int,
    oos_gross_pnl_usd: float,
    oos_costs_usd: float,
    oos_net_pnl_usd: float,
    trial_family: MultipleTestingFamily,
    schedule_shift_null_stats: np.ndarray | None = None,
    schedule_shift_observed_stat: float | None = None,
    circular_permutation_null_stats: np.ndarray | None = None,
    cost_scenario_results: list[CostScenarioResult] | None = None,
    neighbour_results: list[NeighbourResult] | None = None,
    ablation_results: list[AblationResult] | None = None,
    regime_labelling: RegimeLabelling | None = None,
    cross_market_summaries: list[RootRunSummary] | None = None,
    target_schedule_hash: str | None = None,
    holdout_evaluated: bool = False,
    explanation: str = "",
    parameter_stability_required: bool = False,
) -> ValidationReport:
    """``parameter_stability_required`` is a plain passthrough to
    :func:`evaluate_policy` (validation-safety fix) -- default False
    preserves every existing caller's behaviour byte-for-byte (the frozen
    Phase 13.5C matrix never has a missing neighbourhood in practice, and the
    Phase 15 ML adjudication path relies on the default by design). See
    `evaluate_policy`'s docstring."""
    returns = combined_oos_daily.returns_array()
    statname = spec.null_test.statistic
    statfn = _statfn(statname)
    observed_stat = float(statfn(returns)) if returns.size else float("nan")

    # -- OOS metrics --
    oos_metrics = describe_oos(
        returns,
        n_fills=oos_n_fills,
        n_trades=oos_n_trades,
        oos_gross_pnl_usd=oos_gross_pnl_usd,
        oos_costs_usd=oos_costs_usd,
        oos_net_pnl_usd=oos_net_pnl_usd,
        capital_base_usd=spec.capital_base_usd,
        positive_fold_count=walk_forward.positive_fold_count,
        total_fold_count=walk_forward.n_folds_evaluated,
    )

    # -- bootstrap CIs --
    if returns.size >= 3:
        bootstrap = bootstrap_daily_returns(returns, spec.bootstrap)
    else:
        # not enough data -- still produce a typed (degenerate) result
        from alpha_agent.validation.bootstrap import BootstrapCI, BootstrapResult

        deg = BootstrapCI(
            statistic="n/a", point_estimate=float("nan"), ci_low=float("nan"),
            ci_high=float("nan"), ci_level=spec.bootstrap.ci_level, std_error=float("nan"),
            n_resamples=0, method=spec.bootstrap.method.value,
            block_length=spec.bootstrap.mean_block_length, seed=spec.bootstrap.seed,
        )
        bootstrap = BootstrapResult(
            config_fingerprint=spec.bootstrap.identity(), n_days=int(returns.size),
            mean_daily_return=deg, annualized_sharpe=deg,
        )

    # -- null tests --
    #    CENTERED_BLOCK_BOOTSTRAP is the GATING null (observed = full-sample
    #    canonical statistic). SCHEDULE_TIME_SHIFT / CIRCULAR_SCHEDULE_PERMUTATION
    #    are DIAGNOSTIC-ONLY -- their observed statistic is the common-support
    #    CONTROL run's statistic, never the full-sample canonical one (section 3).
    null_results: list[NullTestResult] = []
    control_stat = (
        observed_stat if schedule_shift_observed_stat is None
        else schedule_shift_observed_stat
    )
    if (
        NullMethod.SCHEDULE_TIME_SHIFT in spec.null_test.methods
        and schedule_shift_null_stats is not None
        and np.asarray(schedule_shift_null_stats).size > 0
    ):
        null_results.append(
            summarize_null(
                NullMethod.SCHEDULE_TIME_SHIFT, NullRole.DIAGNOSTIC, statname, control_stat,
                np.asarray(schedule_shift_null_stats, dtype=float),
                canonical_oos_statistic=observed_stat,
            )
        )
    if (
        circular_permutation_null_stats is not None
        and np.asarray(circular_permutation_null_stats).size > 0
    ):
        null_results.append(
            summarize_null(
                NullMethod.CIRCULAR_SCHEDULE_PERMUTATION, NullRole.DIAGNOSTIC, statname,
                control_stat, np.asarray(circular_permutation_null_stats, dtype=float),
                canonical_oos_statistic=observed_stat,
            )
        )
    if NullMethod.CENTERED_BLOCK_BOOTSTRAP in spec.null_test.methods and returns.size >= 3:
        cbb = centered_block_bootstrap_null_stats(returns, spec.null_test, statfn)
        null_results.append(
            summarize_null(
                NullMethod.CENTERED_BLOCK_BOOTSTRAP, NullRole.GATING, statname, observed_stat,
                cbb, canonical_oos_statistic=observed_stat,
            )
        )

    # -- multiple testing (BH/FDR over the trial family) --
    fdr_result = trial_family.benjamini_hochberg(policy.fdr_q_threshold)
    canonical_idx = trial_family.canonical_trial_index()

    # -- deflated Sharpe --
    dsr_result = deflated_sharpe_ratio(
        returns,
        n_trials=trial_family.n_trials(),
        trial_daily_sharpes=trial_family.observed_daily_sharpes(),
    )

    # -- robustness --
    cost_stress: CostStressReport | None = None
    if cost_scenario_results:
        cost_stress = summarize_cost_stress(spec.cost_stress, cost_scenario_results)

    parameter_stability: ParameterStabilityResult | None = None
    if neighbour_results and spec.parameter_neighbourhood is not None:
        parameter_stability = summarize_parameter_stability(
            spec.parameter_neighbourhood, neighbour_results
        )

    ablation_report: AblationReport | None = None
    if ablation_results:
        ablation_report = summarize_ablations(
            oos_metrics.daily_sharpe, ablation_results
        )

    regime_result: RegimeStabilityResult = evaluate_regime_stability(
        returns,
        combined_oos_daily.pnl_array(),
        regime_labelling,
        concentration_threshold=policy.regime_max_pnl_share,
    )

    cross_market: CrossMarketEvidence = evaluate_cross_market(
        cross_market_summaries or [],
        concentration_threshold=policy.cross_market_max_root_share,
    )

    # -- verdict --
    outcome = evaluate_policy(
        policy,
        oos_metrics=oos_metrics,
        walk_forward=walk_forward,
        null_results=null_results,
        fdr_result=fdr_result,
        canonical_trial_index=canonical_idx,
        dsr_result=dsr_result,
        cost_report=cost_stress,
        parameter_stability=parameter_stability,
        ablation_report=ablation_report,
        regime_result=regime_result,
        cross_market=cross_market,
        parameter_stability_required=parameter_stability_required,
    )

    fold_summaries = tuple(
        FoldSummary(
            fold_index=f.fold_index,
            test_start_ts_ns=f.test_start_ts_ns,
            test_end_ts_ns=f.test_end_ts_ns,
            n_trading_days=f.n_trading_days,
            n_fills=f.n_fills,
            n_trades=f.n_trades,
            oos_net_pnl_usd=f.oos_net_pnl_usd,
            daily_sharpe=f.daily_sharpe,
            is_positive=f.is_positive,
        )
        for f in walk_forward.folds
    )

    sample_diag = SampleDiagnostics(
        n_oos_trading_days=oos_metrics.n_trading_days,
        n_nonzero_return_days=oos_metrics.n_nonzero_return_days,
        n_fills=oos_metrics.n_fills,
        n_trades=oos_metrics.n_trades,
        n_folds_planned=walk_forward.n_folds_planned,
        n_folds_evaluated=walk_forward.n_folds_evaluated,
        minimum_sample_fingerprint=spec.minimum_sample.identity(),
    )

    return ValidationReport(
        strategy_fingerprint=spec.strategy_fingerprint,
        validation_fingerprint=spec.validation_fingerprint(),
        dataset_fingerprint=spec.dataset.identity(),
        split_fingerprint=spec.split_plan.split_fingerprint(),
        reliability_policy_fingerprint=policy.identity(),
        trial_family_fingerprint=trial_family.family_fingerprint(),
        target_schedule_hash=target_schedule_hash,
        holdout_evaluated=holdout_evaluated,
        split_windows=tuple(
            {"role": w.role.value, "start_ts_ns": w.start_ts_ns, "end_ts_ns": w.end_ts_ns}
            for w in spec.split_plan.windows
        ),
        fold_summaries=fold_summaries,
        walk_forward=walk_forward,
        oos_daily=combined_oos_daily,
        oos_metrics=oos_metrics,
        bootstrap=bootstrap,
        null_results=tuple(null_results),
        fdr_result=fdr_result,
        canonical_trial_index=canonical_idx,
        dsr_result=dsr_result,
        cost_stress=cost_stress,
        parameter_stability=parameter_stability,
        ablation_report=ablation_report,
        regime_result=regime_result,
        cross_market=cross_market,
        sample_diagnostics=sample_diag,
        verdict=outcome.verdict,
        reason_codes=outcome.reason_codes,
        gate_results=outcome.gate_results,
        explanation=explanation,
    )
