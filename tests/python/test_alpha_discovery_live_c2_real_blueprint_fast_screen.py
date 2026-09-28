"""Alpha Discovery live-research campaign, Checkpoint 2 -- required REAL,
non-legacy blueprint proof through the real C++ Fast Screen (task spec Part 1
section 8, tests section 51).

The campaign is explicit: it is not complete until at least one genuinely
non-legacy blueprint successfully runs

    blueprint -> real cadence -> real target schedule -> real C++ backtest
    -> ResearchScreenResult

with NO Python PnL substitute. This module drives the exact example mechanism
task spec section 8 names -- "volatility-conditioned MA trend"
(`MA spread > 0 AND realized volatility percentile < threshold`) -- through:

    ResearchAgent (ScriptedLLMClient)
        -> StrategyCompilerAgent (ScriptedLLMClient, BLUEPRINT mode, the new
           typed `signal_cadence`)
        -> ResearchOrchestrator.plan_family() (real experiment_identity,
           `strategy_family` starting with "dsl:" -- never one of the five
           legacy templates)
        -> alpha_agent.screening.fast_screen.run_fast_screen (the REAL
           compiled `quant_backtest_targets_csv` CLI, over the real,
           already-acquired local NQ 2018-2022 research window)

and asserts a genuine `FastScreenStatus.SCREENED` outcome with a real
`ResearchScreenScore` -- never `UNSUPPORTED`, never a fabricated score.

Skipped (not failed) when the compiled C++ CLI is not present in this
checkout -- mirrors `test_alpha_discovery_part_f_fast_screen.py`'s own
real-data integration test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from alpha_agent.agents import (
    FamilyPlanSpec,
    IdentityPlanes,
    OrchestratorBudget,
    OrchestratorConfig,
    ResearchAgent,
    ResearchOrchestrator,
    ScriptedExecutionValidationService,
    ScriptedLLMClient,
    StrategyCompilerAgent,
)
from alpha_agent.registry import ExperimentRegistry
from alpha_agent.screening.fast_screen import FastScreenStatus, run_fast_screen
from alpha_agent.validation.policy import ReliabilityPolicy

UNIVERSE = ("ES", "NQ", "CL", "GC", "ZN")


def _real_cli_path() -> str:
    root = Path(__file__).resolve().parents[2]
    return str(root / "build" / "cpp" / "cpp" / "quant_backtest_targets_csv")


_CLI_MISSING = not Path(_real_cli_path()).exists()


def _planes() -> IdentityPlanes:
    policy = ReliabilityPolicy()
    return IdentityPlanes(
        dataset_fingerprint="valdataset2:live_c2",
        split_identity="split1:live_c2",
        validation_spec_fingerprint="validationprotocol1:live_c2",
        reliability_policy_fingerprint=policy.identity(),
        execution_config_identity="execconfig1:live_c2",
        cost_config_identity="costconfig1:live_c2",
        risk_identity="riskconfig1:live_c2",
    )


def _config() -> OrchestratorConfig:
    from alpha_agent.registry.models import MarketWindow

    return OrchestratorConfig(
        objective="volatility-conditioned MA trend on NQ",
        market_universe=UNIVERSE,
        market_window=MarketWindow(label="RESEARCH_2018_2022", start_date="2018-01-01", end_date="2022-12-31"),
        planes=_planes(),
        family_plan=FamilyPlanSpec(family_stem="live_c2_vol_ma_trend", target_family_size=1),
        budget=OrchestratorBudget(max_planning_attempts_per_family=3),
        phase="alpha-discovery-live-c2",
    )


def _hypothesis_json() -> str:
    return json.dumps(
        {
            "hypothesis_id": "H-VOL-MA-TREND",
            "title": "NQ trend premium conditioned on a contained volatility regime",
            "economic_mechanism": (
                "Directional continuation (trend) is a robust cross-asset premium, but "
                "trend signals are noisiest and most whipsaw-prone exactly when realized "
                "volatility is elevated; conditioning the trend signal on a LOW realized-"
                "volatility percentile regime should improve the trend signal's "
                "signal-to-noise ratio without adding a second independent bet."
            ),
            "universe": ["NQ"],
            "horizon": "trend continuation, filtered daily",
            "required_features": ["ma_spread", "vol_percentile"],
            "signal_description": (
                "Long when the fast/slow moving-average spread is positive AND realized "
                "volatility is in a low percentile regime; short on the mirror image; "
                "flat otherwise."
            ),
            "expected_regime": "low-volatility trending regime",
            "failure_regime": "high-volatility whipsaw regime",
            "falsification_test": "No positive OOS net PnL after costs, or no improvement over plain MA trend.",
        }
    )


def _blueprint_plan_json() -> str:
    return json.dumps(
        {
            "expressible": True,
            "blueprint": {
                "root_symbol": "NQ",
                "signal_cadence": "daily_trading_day",
                "features": [
                    {"alias": "msp", "kind": "ma_spread", "params": {"fast": 20, "slow": 100}},
                    {"alias": "volp", "kind": "vol_percentile", "params": {"window": 20, "lookback": 252}},
                ],
                "rules": [
                    {"rule_id": "long", "target_units": 1,
                     "when": {"type": "boolean", "op": "all", "nodes": [
                         {"type": "comparison", "op": "gt",
                          "left": {"type": "feature", "feature": "msp"},
                          "right": {"type": "const", "value": 0.0}},
                         {"type": "comparison", "op": "lt",
                          "left": {"type": "feature", "feature": "volp"},
                          "right": {"type": "const", "value": 0.5}}]}},
                    {"rule_id": "short", "target_units": -1,
                     "when": {"type": "boolean", "op": "all", "nodes": [
                         {"type": "comparison", "op": "lt",
                          "left": {"type": "feature", "feature": "msp"},
                          "right": {"type": "const", "value": 0.0}},
                         {"type": "comparison", "op": "lt",
                          "left": {"type": "feature", "feature": "volp"},
                          "right": {"type": "const", "value": 0.5}}]}},
                    {"rule_id": "flat_high_vol", "target_units": 0,
                     "when": {"type": "comparison", "op": "gte",
                              "left": {"type": "feature", "feature": "volp"},
                              "right": {"type": "const", "value": 0.5}}},
                ],
                "default_action": "flat",
            },
            "rationale": (
                "Volatility-conditioned MA trend: an economically distinct mechanism "
                "from plain unconditional MA trend, not one of the five frozen legacy "
                "templates."
            ),
        }
    )


@pytest.mark.skipif(_CLI_MISSING, reason="compiled CLI not present in this checkout")
def test_volatility_conditioned_ma_trend_blueprint_screens_through_real_cpp(tmp_path):
    reg = ExperimentRegistry(tmp_path / "registry.sqlite")
    orch = ResearchOrchestrator(
        registry=reg,
        research_agent=ResearchAgent(ScriptedLLMClient([_hypothesis_json()])),
        compiler_agent=StrategyCompilerAgent(ScriptedLLMClient([_blueprint_plan_json()])),
        execution_service=ScriptedExecutionValidationService([]),  # never called by plan_family()
        reliability_policy=ReliabilityPolicy(),
        config=_config(),
    )
    manifest = orch.plan_family()
    assert manifest.planning_complete, [o.disposition.value for o in manifest.planning_log]
    (member,) = manifest.members

    # -- proof: a genuinely NEW, non-legacy blueprint, not a disguised template --
    assert member.strategy_family.startswith("dsl:"), member.strategy_family
    assert member.strategy_family not in {"tsmom", "ma_trend", "breakout", "mean_reversion", "silver_bullet"}
    assert member.signal_cadence == "daily_trading_day"  # real, typed -- not the old placeholder
    assert {f.spec.kind for f in member.strategy_spec.features} == {"ma_spread", "vol_percentile"}

    # -- proof: real cadence -> real schedule -> real C++ backtest --
    trial = run_fast_screen(member=member, cli_executable=_real_cli_path(), work_dir=tmp_path)

    assert trial.status is FastScreenStatus.SCREENED, (trial.capability_reason, trial.detail)
    assert trial.score is not None
    assert 0.0 <= trial.score.total <= 100.0
    assert trial.metrics.get("n_days", 0) > 0
    assert trial.metrics.get("n_trades", 0) >= 0  # a real (possibly zero) trade count, never fabricated

    # provenance snapshot -- a durable, human-inspectable record that this real
    # non-legacy blueprint proof actually ran, for the campaign's final report.
    _write_provenance(member, trial)


def _write_provenance(member, trial) -> None:
    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "alpha_discovery_live"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "alpha-discovery-live-checkpoint-2/1",
        "checkpoint": "2 -- real non-legacy blueprint C++ Fast Screen proof",
        "hypothesis_id": member.hypothesis_id,
        "strategy_family": member.strategy_family,
        "root_symbol": member.root_symbol,
        "signal_cadence": member.signal_cadence,
        "execution_cadence": member.execution_cadence,
        "experiment_identity": member.experiment_identity,
        "strategy_fingerprint": member.strategy_fingerprint,
        "feature_kinds": sorted({f.spec.kind for f in member.strategy_spec.features}),
        "fast_screen_status": trial.status.value,
        "fast_screen_score_total": trial.score.total if trial.score else None,
        "fast_screen_metrics": trial.metrics,
    }
    (out_dir / "CHECKPOINT_2_REAL_BLUEPRINT_FAST_SCREEN.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
