"""Phase 11 -- deterministic research bridge from the Strategy DSL to the C++
backtest engine, plus baseline run lineage and descriptive summaries.

    CompiledStrategyPlan + point-in-time FeatureFrame
        -> ReferenceEvaluator (target-position intent, KEEP_PREVIOUS resolved)
        -> TargetSchedule (targets.csv: TARGET INTENT ONLY)
        -> C++ ScheduledTargetStrategy -> BacktestEngine -> Hard Risk
        -> ExecutionSimulator -> Fill -> Portfolio -> official BacktestResult

Official fills / PnL / costs / positions / drawdown stay **C++ Fill-derived**.
Python only writes the target-intent bundle and reads the C++ result back into a
typed descriptive summary and a run manifest. There is no alternate Python PnL
calculator.
"""
from __future__ import annotations

from alpha_agent.backtest.manifest import (
    BaselineRunManifest,
    ExecutionConfigIdentity,
    build_run_manifest,
)
from alpha_agent.backtest.summary import BaselineRunSummary, summarize_cpp_result
from alpha_agent.backtest.targets import (
    TARGET_SCHEDULE_COLUMNS,
    TargetSchedule,
    TargetScheduleRow,
    build_target_schedule,
)

__all__ = [
    "TARGET_SCHEDULE_COLUMNS",
    "BaselineRunManifest",
    "BaselineRunSummary",
    "ExecutionConfigIdentity",
    "TargetSchedule",
    "TargetScheduleRow",
    "build_run_manifest",
    "build_target_schedule",
    "summarize_cpp_result",
]
