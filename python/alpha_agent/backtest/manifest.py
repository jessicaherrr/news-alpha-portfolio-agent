"""Typed run manifest for a baseline backtest (Phase 11 section 15).

Enough lineage to reproduce a baseline run: source-data lineage, the
``FeatureSpec``s + feature-engine version, the canonical ``StrategySpec`` + its
fingerprint, the target-schedule hash, and the execution / risk config identity.
This is **not** the Phase 14 experiment registry -- just a simple typed manifest.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
from pydantic import BaseModel, Field

from alpha_agent.backtest.targets import TargetSchedule
from alpha_agent.data.lineage import code_commit
from alpha_agent.features.frame import FeatureFrame
from alpha_agent.strategy.fingerprint import canonical_strategy_payload
from alpha_agent.strategy.plan import CompiledStrategyPlan
from alpha_agent.strategy.spec import StrategySpec


class ExecutionConfigIdentity(BaseModel):
    """The execution assumptions a run used -- documented and held constant
    across strategy families for comparison runs (section 17)."""

    model_config = {"frozen": True, "extra": "forbid"}

    commission_per_contract_usd: float = 2.0
    slippage_ticks: float = 0.0
    spread_ticks: float = 0.0
    latency_bars: int = 0
    end_of_test_policy: str = "ForceLiquidateFinalClose"
    eot_stale_policy: str = "LeaveOpen"
    roll_price_fallback: str = "RejectDefer"

    def identity(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))


# The frozen reference CLI (quant_backtest_targets_csv) runs these exact
# assumptions -- a flat commission, no slippage, next-bar execution,
# force-liquidate at the final event. RiskConfig / MarginModel are wired only at
# the pybind boundary (Phase 19), so the reference path runs PassThroughRiskManager.
REFERENCE_EXECUTION_CONFIG = ExecutionConfigIdentity()
REFERENCE_RISK_IDENTITY = "PassThroughRiskManager (frozen reference CLI; no hard limits enforced here)"


class BaselineRunManifest(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "baseline-run/1"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    code_commit: str | None = None

    # -- source-data lineage
    source_price_domain: str
    source_adjustment_mode: str | None
    source_fingerprint: str
    source_paths: list[str]
    n_source_rows: int
    as_of_ts_ns: int | None

    # -- feature engine
    feature_engine_version: str
    feature_names: list[str]
    feature_specs: list[dict]

    # -- strategy (DSL)
    strategy_id: str
    strategy_name: str
    strategy_root_symbol: str
    strategy_dsl_version: str
    strategy_fingerprint: str
    strategy_spec_canonical: dict
    warmup_bars: int

    # -- target schedule
    target_schedule_hash: str
    n_target_rows: int
    target_date_range_ns: tuple[int, int] | None

    # -- execution / risk identity
    execution_config: ExecutionConfigIdentity
    execution_config_identity: str
    risk_config_identity: str

    # -- boundary bundle
    bars_row_count: int
    contracts_row_count: int

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=indent)


def build_run_manifest(
    *,
    spec: StrategySpec,
    plan: CompiledStrategyPlan,
    frame: FeatureFrame,
    schedule: TargetSchedule,
    bars: pd.DataFrame,
    contracts: pd.DataFrame,
    execution_config: ExecutionConfigIdentity = REFERENCE_EXECUTION_CONFIG,
    risk_config_identity: str = REFERENCE_RISK_IDENTITY,
) -> BaselineRunManifest:
    lin = frame.lineage
    return BaselineRunManifest(
        code_commit=code_commit(),
        source_price_domain=lin.source_price_domain,
        source_adjustment_mode=lin.source_adjustment_mode,
        source_fingerprint=lin.source_fingerprint,
        source_paths=list(lin.source_paths),
        n_source_rows=lin.n_source_rows,
        as_of_ts_ns=lin.as_of_ts_ns,
        feature_engine_version=lin.engine_version,
        feature_names=list(plan.required_features),
        feature_specs=[json.loads(b.spec.canonical_json()) for b in plan.feature_bindings],
        strategy_id=spec.strategy_id,
        strategy_name=spec.strategy_name,
        strategy_root_symbol=spec.root_symbol,
        strategy_dsl_version=plan.dsl_version,
        strategy_fingerprint=plan.fingerprint,
        strategy_spec_canonical=canonical_strategy_payload(spec),
        warmup_bars=plan.warmup_bars,
        target_schedule_hash=schedule.schedule_hash(),
        n_target_rows=schedule.n_rows,
        target_date_range_ns=schedule.date_range_ns,
        execution_config=execution_config,
        execution_config_identity=execution_config.identity(),
        risk_config_identity=risk_config_identity,
        bars_row_count=len(bars),
        contracts_row_count=len(contracts),
    )
