"""The frozen typed :class:`ValidationReport` (section 22).

The verdict semantics are entirely typed (``verdict`` + ``reason_codes``). Any
free-text field is explicitly non-semantic and never drives a decision.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from alpha_agent.validation.ablation import AblationReport
from alpha_agent.validation.bootstrap import BootstrapResult
from alpha_agent.validation.cost_stress import CostStressReport
from alpha_agent.validation.crossmarket import CrossMarketEvidence
from alpha_agent.validation.dsr import DeflatedSharpeResult
from alpha_agent.validation.enums import ReasonCode, Verdict
from alpha_agent.validation.fdr import FdrResult
from alpha_agent.validation.fingerprint import fingerprint
from alpha_agent.validation.metrics import OosMetrics
from alpha_agent.validation.nulls import NullTestResult
from alpha_agent.validation.regime import RegimeStabilityResult
from alpha_agent.validation.returns import DailyReturnSeries
from alpha_agent.validation.stability import ParameterStabilityResult
from alpha_agent.validation.walkforward import WalkForwardResult


class SampleDiagnostics(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    n_oos_trading_days: int
    n_nonzero_return_days: int
    n_fills: int
    n_trades: int
    n_folds_planned: int
    n_folds_evaluated: int
    minimum_sample_fingerprint: str


class FoldSummary(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    fold_index: int
    test_start_ts_ns: int
    test_end_ts_ns: int
    n_trading_days: int
    n_fills: int
    n_trades: int
    oos_net_pnl_usd: float
    daily_sharpe: float
    is_positive: bool


class ValidationReport(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "validation-report/1"

    # -- identity (section 21) --
    strategy_fingerprint: str
    validation_fingerprint: str
    dataset_fingerprint: str
    split_fingerprint: str
    reliability_policy_fingerprint: str
    trial_family_fingerprint: str
    target_schedule_hash: str | None = None
    holdout_evaluated: bool = False

    # -- split / walk-forward --
    split_windows: tuple[dict, ...]
    fold_summaries: tuple[FoldSummary, ...]
    walk_forward: WalkForwardResult

    # -- OOS observations + metrics --
    oos_daily: DailyReturnSeries
    oos_metrics: OosMetrics

    # -- inferential evidence --
    bootstrap: BootstrapResult
    null_results: tuple[NullTestResult, ...]
    fdr_result: FdrResult
    canonical_trial_index: int
    dsr_result: DeflatedSharpeResult

    # -- robustness evidence --
    cost_stress: CostStressReport | None = None
    parameter_stability: ParameterStabilityResult | None = None
    ablation_report: AblationReport | None = None
    regime_result: RegimeStabilityResult
    cross_market: CrossMarketEvidence

    # -- sample-size diagnostics --
    sample_diagnostics: SampleDiagnostics

    # -- verdict (typed; no free text drives this) --
    verdict: Verdict
    reason_codes: tuple[ReasonCode, ...]
    gate_results: dict

    # -- optional non-semantic explanation --
    explanation: str = Field(default="", description="non-semantic; never affects the verdict")

    def report_fingerprint(self) -> str:
        return fingerprint(
            "validationreport1",
            {
                "schema_version": self.schema_version,
                "validation_fingerprint": self.validation_fingerprint,
                "strategy_fingerprint": self.strategy_fingerprint,
                "verdict": self.verdict.value,
                "reason_codes": sorted(c.value for c in self.reason_codes),
                "oos_metrics": self.oos_metrics.model_dump(mode="json"),
                "fdr": self.fdr_result.model_dump(mode="json"),
                "dsr": self.dsr_result.model_dump(mode="json"),
                "null": sorted(
                    json.dumps(r.model_dump(mode="json"), sort_keys=True) for r in self.null_results
                ),
                "walk_forward_consistency": self.walk_forward.fold_consistency,
                "cost_stress": (
                    self.cost_stress.model_dump(mode="json") if self.cost_stress else None
                ),
                "parameter_stability": (
                    self.parameter_stability.model_dump(mode="json")
                    if self.parameter_stability
                    else None
                ),
                "regime": self.regime_result.model_dump(mode="json"),
                "cross_market": self.cross_market.model_dump(mode="json"),
                "holdout_evaluated": self.holdout_evaluated,
            },
            allow_non_finite=True,  # an undefined statistic (e.g. Sharpe with zero variance) is legitimately NaN
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=indent)
