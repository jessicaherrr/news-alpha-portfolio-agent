"""Phase 13 -- reliability-aware strategy validation framework.

It answers "does this strategy have evidence that survives realistic
validation?" -- not merely "did the historical backtest make money?". The
verdict is one of ``PASS`` / ``REJECT`` / ``INCONCLUSIVE``; ``REJECT`` is a
normal, successful system outcome.

Official economics stay C++-sourced: every economic number comes from ONE C++
backtest per (target schedule, cost scenario) via the same
``ScheduledTargetStrategy -> BacktestEngine -> Fill -> PortfolioAccountant``
path used by Phases 11 and 12. This framework only aggregates the additive
``BacktestResult::daily_equity`` trace into a canonical DAILY validation return
series and runs deterministic statistics on it. It never reconstructs fills,
positions, commissions, execution prices or portfolio equity.

See ``docs/RELIABILITY_VALIDATION.md``.
"""
from __future__ import annotations

from alpha_agent.validation.ablation import (
    AblationReport,
    AblationResult,
    AblationSpec,
    summarize_ablations,
)
from alpha_agent.validation.assemble import assemble_report
from alpha_agent.validation.bootstrap import (
    BootstrapConfig,
    BootstrapResult,
    bootstrap_daily_returns,
)
from alpha_agent.validation.cost_stress import (
    DEFAULT_COST_STRESS_PLAN,
    CostScenario,
    CostScenarioResult,
    CostStressPlan,
    CostStressReport,
    summarize_cost_stress,
)
from alpha_agent.validation.crossmarket import (
    CrossMarketEvidence,
    RootRunSummary,
    evaluate_cross_market,
)
from alpha_agent.validation.dataset import (
    DatasetIdentity,
    bars_in_window,
    dataset_identity_from_bars,
    frame_content_hash,
)
from alpha_agent.validation.dsr import (
    DeflatedSharpeResult,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from alpha_agent.validation.engine import (
    DslFamilyAdapter,
    StrategyFamilyAdapter,
    ValidationEngine,
)
from alpha_agent.validation.enums import (
    AblationKind,
    BootstrapMethod,
    CostScenarioKind,
    EvidenceStatus,
    NullMethod,
    NullRole,
    PortfolioStatePolicy,
    ReasonCode,
    RegimeKind,
    SplitRole,
    Verdict,
)
from alpha_agent.validation.fdr import (
    FdrResult,
    benjamini_hochberg,
    benjamini_hochberg_decisions,
    bh_qvalues,
)
from alpha_agent.validation.fingerprint import VALIDATION_FRAMEWORK_VERSION, fingerprint
from alpha_agent.validation.holdout import (
    FrozenResearchBundle,
    HoldoutDisciplineError,
    HoldoutEvaluation,
    HoldoutRelease,
    freeze_research,
)
from alpha_agent.validation.metrics import (
    OosMetrics,
    annualized_sharpe,
    daily_sharpe,
    describe_oos,
)
from alpha_agent.validation.multiple_testing import (
    MultipleTestingFamily,
    TrialRecord,
)
from alpha_agent.validation.nulls import (
    NullTestConfig,
    NullTestResult,
    causal_segment_ids,
    centered_block_bootstrap_null_stats,
    circular_permuted_schedules,
    common_support_control_schedule,
    empirical_p_value,
    shifted_schedules,
)
from alpha_agent.validation.policy import (
    TEST_FIXTURE_POLICY,
    MinimumSampleRequirements,
    PolicyOutcome,
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
from alpha_agent.validation.returns import (
    DailyReturnSeries,
    concat_oos_series,
    daily_returns_from_cpp_trace,
)
from alpha_agent.validation.runner import (
    BacktestRun,
    BacktestRunner,
    CliBacktestRunner,
    backtest_run_from_cpp_json,
    synthetic_backtest_run,
)
from alpha_agent.validation.spec import ValidationSpec
from alpha_agent.validation.splits import SplitPlan, SplitWindow
from alpha_agent.validation.stability import (
    NeighbourResult,
    ParameterNeighbourhood,
    ParameterStabilityResult,
    summarize_parameter_stability,
)
from alpha_agent.validation.trading_day import (
    TradingDayConvention,
    ValidationDay,
    ValidationDayPlan,
    build_validation_day_plan,
)
from alpha_agent.validation.walkforward import (
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardFoldResult,
    WalkForwardResult,
    build_folds,
    summarize_walk_forward,
)

__all__ = [
    "DEFAULT_COST_STRESS_PLAN",
    "TEST_FIXTURE_POLICY",
    "VALIDATION_FRAMEWORK_VERSION",
    "AblationKind",
    "AblationReport",
    "AblationResult",
    "AblationSpec",
    "BacktestRun",
    "BacktestRunner",
    "BootstrapConfig",
    "BootstrapMethod",
    "BootstrapResult",
    "CliBacktestRunner",
    "CostScenario",
    "CostScenarioKind",
    "CostScenarioResult",
    "CostStressPlan",
    "CostStressReport",
    "CrossMarketEvidence",
    "DailyReturnSeries",
    "DatasetIdentity",
    "DeflatedSharpeResult",
    "DslFamilyAdapter",
    "EvidenceStatus",
    "FdrResult",
    "FrozenResearchBundle",
    "HoldoutDisciplineError",
    "HoldoutEvaluation",
    "HoldoutRelease",
    "MinimumSampleRequirements",
    "MultipleTestingFamily",
    "NeighbourResult",
    "NullMethod",
    "NullRole",
    "NullTestConfig",
    "NullTestResult",
    "OosMetrics",
    "ParameterNeighbourhood",
    "ParameterStabilityResult",
    "PolicyOutcome",
    "PortfolioStatePolicy",
    "ReasonCode",
    "RegimeKind",
    "RegimeLabelling",
    "RegimeStabilityResult",
    "ReliabilityPolicy",
    "RootRunSummary",
    "SplitPlan",
    "SplitRole",
    "SplitWindow",
    "StrategyFamilyAdapter",
    "TradingDayConvention",
    "TrialRecord",
    "ValidationDay",
    "ValidationDayPlan",
    "ValidationEngine",
    "ValidationReport",
    "ValidationSpec",
    "Verdict",
    "WalkForwardConfig",
    "WalkForwardFold",
    "WalkForwardFoldResult",
    "WalkForwardResult",
    "annualized_sharpe",
    "assemble_report",
    "backtest_run_from_cpp_json",
    "bars_in_window",
    "benjamini_hochberg",
    "benjamini_hochberg_decisions",
    "bh_qvalues",
    "bootstrap_daily_returns",
    "build_folds",
    "build_validation_day_plan",
    "causal_segment_ids",
    "causal_volatility_regime_labels",
    "centered_block_bootstrap_null_stats",
    "circular_permuted_schedules",
    "common_support_control_schedule",
    "concat_oos_series",
    "daily_returns_from_cpp_trace",
    "daily_sharpe",
    "dataset_identity_from_bars",
    "deflated_sharpe_ratio",
    "describe_oos",
    "empirical_p_value",
    "evaluate_cross_market",
    "evaluate_policy",
    "evaluate_regime_stability",
    "fingerprint",
    "frame_content_hash",
    "freeze_research",
    "probabilistic_sharpe_ratio",
    "shifted_schedules",
    "summarize_ablations",
    "summarize_cost_stress",
    "summarize_parameter_stability",
    "summarize_walk_forward",
    "synthetic_backtest_run",
]
