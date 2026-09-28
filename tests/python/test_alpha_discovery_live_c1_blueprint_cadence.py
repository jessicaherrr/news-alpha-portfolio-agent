"""Alpha Discovery live-research campaign, Checkpoint 1 -- typed blueprint
cadence + executable closed DSL (task spec Part 1, tests section 51).

Before this fix, `StrategyCompilerAgent._execution_semantics` set a placeholder
string for `signal_cadence` on every BLUEPRINT build (`build_mode !=
"template"`) -- "set by the orchestrator backtest configuration (not a
StrategySpec field)". That string is never in
`alpha_agent.execution.capability.SUPPORTED_CADENCES`, so EVERY custom
blueprint died with `UNSUPPORTED_CADENCE` the moment it reached
`ProductionExecutionValidationService` / `run_fast_screen`, regardless of how
well-formed its DSL was (`docs/ALPHA_DISCOVERY_CAMPAIGN.md`'s Part E scope
limit).

`StrategyBlueprint.signal_cadence` is now a real, typed, closed field
(`daily_trading_day` | `native_1m` -- the exact two cadences this release's
execution path can schedule, `SUPPORTED_CADENCES`), and
`_execution_semantics` uses it directly instead of a placeholder. This module
proves:

* the placeholder string is gone -- a blueprint's `ExecutionSemantics.
  signal_cadence` is always a real, capability-supported value;
* both real cadences compile and pass the static capability gate;
* an out-of-vocabulary cadence is a retryable SCHEMA failure (closed Literal),
  never a silent placeholder;
* `BlueprintSignalCadence` names exactly `SUPPORTED_CADENCES` (no drift);
* legacy TEMPLATE families are completely untouched -- byte-identical
  fingerprint, cadence, and `strategy_family` label to before this change;
* two structurally-identical-except-cadence blueprints get DIFFERENT
  `experiment_identity` (via `strategy_family`) -- cadence changes realized
  trades/PnL, so it must not silently collide into one scientific identity.
"""
from __future__ import annotations

import json

import pytest
from alpha_agent.agents import ScriptedLLMClient, StrategyCompilerAgent
from alpha_agent.agents.compiler_agent import (
    BlueprintSignalCadence,
    assert_blueprint_cadences_match_supported,
)
from alpha_agent.agents.orchestrator import _strategy_family_label
from alpha_agent.execution.capability import SUPPORTED_CADENCES, assess_static_capability
from alpha_agent.schemas.hypothesis import HypothesisSpec
from alpha_agent.strategy.candidates_phase_13_5c import spec_for_params
from test_phase_17_strategy_compiler_agent import (
    _TEMPLATE_CANONICAL,
    _context,
    _hypothesis,
)

UNIVERSE = ("ES", "NQ", "CL", "GC", "ZN")


def _blueprint_plan(root="NQ", *, signal_cadence: str | None = "native_1m") -> str:
    payload = {
        "expressible": True,
        "blueprint": {
            "root_symbol": root,
            "features": [
                {"alias": "msp", "kind": "ma_spread", "params": {"fast": 20, "slow": 100}},
                {"alias": "volp", "kind": "vol_percentile", "params": {"window": 60, "lookback": 252}},
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
                {"rule_id": "flat", "target_units": 0,
                 "when": {"type": "comparison", "op": "lte",
                          "left": {"type": "feature", "feature": "msp"},
                          "right": {"type": "const", "value": 0.0}}},
            ],
            "default_action": "flat",
        },
    }
    if signal_cadence is not None:
        payload["blueprint"]["signal_cadence"] = signal_cadence
    return json.dumps(payload)


def _regime_hypothesis(**overrides) -> HypothesisSpec:
    base = {
        "hypothesis_id": "H-VOL-REGIME",
        "title": "trend premium conditioned on the volatility regime",
        "universe": ["NQ"],
        "required_features": ["ma_spread", "vol_percentile"],
        "signal_description": "Long trend only while realized volatility is in a low percentile regime.",
    }
    base.update(overrides)
    return _hypothesis(**base)


def test_blueprint_signal_cadences_match_the_real_supported_set():
    assert_blueprint_cadences_match_supported()  # no raise
    from typing import get_args
    assert frozenset(get_args(BlueprintSignalCadence)) == SUPPORTED_CADENCES == {
        "daily_trading_day", "native_1m",
    }


def test_blueprint_declares_native_1m_cadence_not_a_placeholder():
    result = StrategyCompilerAgent(
        ScriptedLLMClient([_blueprint_plan(signal_cadence="native_1m")])
    ).compile_hypothesis(_regime_hypothesis(), _context())
    assert result.accepted, result.rejection_code
    sem = result.execution_semantics
    assert sem is not None
    assert sem.signal_cadence == "native_1m"
    assert "orchestrator backtest configuration" not in sem.signal_cadence  # the old placeholder
    capability = assess_static_capability(
        signal_cadence=sem.signal_cadence,
        required_features=[b for b in result.blueprint_feature_kinds],
    )
    assert capability.supported, capability.reason


def test_blueprint_defaults_to_daily_trading_day_cadence_when_unspecified():
    result = StrategyCompilerAgent(
        ScriptedLLMClient([_blueprint_plan(signal_cadence=None)])
    ).compile_hypothesis(_regime_hypothesis(), _context())
    assert result.accepted, result.rejection_code
    assert result.execution_semantics.signal_cadence == "daily_trading_day"


def test_blueprint_declares_daily_trading_day_cadence_explicitly():
    result = StrategyCompilerAgent(
        ScriptedLLMClient([_blueprint_plan(signal_cadence="daily_trading_day")])
    ).compile_hypothesis(_regime_hypothesis(), _context())
    assert result.accepted, result.rejection_code
    assert result.execution_semantics.signal_cadence == "daily_trading_day"


def test_blueprint_with_an_out_of_vocabulary_cadence_is_a_schema_retry_not_a_silent_placeholder():
    from alpha_agent.agents.compiler_agent import CompilerSchemaRetryExhausted

    with pytest.raises(CompilerSchemaRetryExhausted):
        StrategyCompilerAgent(
            ScriptedLLMClient([_blueprint_plan(signal_cadence="weekly")]),
            max_schema_retries=0,
        ).compile_hypothesis(_regime_hypothesis(), _context())


# -- legacy replay guarantee (task spec section 9 / 51) --------------------


def test_legacy_template_cadence_and_family_label_are_completely_unchanged():
    from alpha_agent.strategy.candidates_phase_13_5c import SIGNAL_CADENCE

    for family, params in _TEMPLATE_CANONICAL.items():
        plan = json.dumps({
            "expressible": True,
            "template": {"family_key": family, "root_symbol": "ES", "params": params},
        })
        hyp = _hypothesis(
            universe=["ES"],
            required_features={
                "tsmom": ["diff"], "ma_trend": ["ma"],
                "breakout": ["breakout_up", "breakout_down"], "mean_reversion": ["zscore"],
            }[family],
        )
        result = StrategyCompilerAgent(ScriptedLLMClient([plan])).compile_hypothesis(hyp, _context())
        assert result.accepted, (family, result.rejection_code)
        assert result.execution_semantics.signal_cadence == SIGNAL_CADENCE[family]
        assert _strategy_family_label(result) == family  # untouched by the dsl:<cadence>: fix
        rebuilt = spec_for_params(family, {"root_symbol": "ES", **params})
        from alpha_agent.strategy import strategy_fingerprint
        assert result.strategy_fingerprint == strategy_fingerprint(rebuilt)


# -- cadence enters scientific identity where required (task spec section 3) -


def test_same_features_different_cadence_get_different_strategy_family_label():
    """Two blueprints with an IDENTICAL feature composition but a different
    declared `signal_cadence` decide at different granularities -- daily
    aggregation vs native 1-minute -- which can change realized trades and
    PnL. `_strategy_family_label` (which flows into `experiment_identity`)
    must therefore discriminate them; colliding would let one predeclared
    BH/FDR trial silently stand in for two materially different execution
    schedules."""
    daily = StrategyCompilerAgent(
        ScriptedLLMClient([_blueprint_plan(signal_cadence="daily_trading_day")])
    ).compile_hypothesis(_regime_hypothesis(), _context())
    minute = StrategyCompilerAgent(
        ScriptedLLMClient([_blueprint_plan(signal_cadence="native_1m")])
    ).compile_hypothesis(_regime_hypothesis(), _context())
    assert daily.accepted and minute.accepted
    # the underlying StrategySpec (features/rules) is identical -- only
    # cadence differs -- so the fingerprint alone does NOT discriminate them.
    assert daily.strategy_fingerprint == minute.strategy_fingerprint
    label_daily = _strategy_family_label(daily)
    label_minute = _strategy_family_label(minute)
    assert label_daily != label_minute
    assert label_daily == "dsl:daily_trading_day:ma_spread+vol_percentile"
    assert label_minute == "dsl:native_1m:ma_spread+vol_percentile"


def test_blueprint_capability_gate_supports_both_real_cadences_for_the_same_dsl():
    for cadence in sorted(SUPPORTED_CADENCES):
        result = StrategyCompilerAgent(
            ScriptedLLMClient([_blueprint_plan(signal_cadence=cadence)])
        ).compile_hypothesis(_regime_hypothesis(), _context())
        assert result.accepted, (cadence, result.rejection_code)
        capability = assess_static_capability(
            signal_cadence=result.execution_semantics.signal_cadence,
            required_features=list(result.blueprint_feature_kinds),
        )
        assert capability.supported, (cadence, capability.reason)
