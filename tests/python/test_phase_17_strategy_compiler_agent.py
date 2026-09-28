"""Phase 17 / 17.1 -- runtime Strategy Compiler Agent (prompt 17).

Every test drives the agent with :class:`ScriptedLLMClient`: no network, no real
model, fully deterministic. What is exercised:

* the LLM output is a closed typed ``StrategyPlanRequest`` -- a ``template``
  (Phase 11 family + typed-bounded params) OR a full ``blueprint`` of the frozen
  Phase 10 closed DSL, OR a typed non-expressible result;
* a novel multi-feature hypothesis compiles from closed DSL primitives only;
* arbitrary code / operators / feature kinds remain impossible;
* every ``hypothesis.required_features`` kind is actually *referenced by a rule*
  in the compiled spec (declared-but-unused is rejected);
* the produced spec compiles through the UNCHANGED Phase 10 ``StrategyCompiler``
  and fingerprints identically to a deterministic rebuild;
* duplicate semantics: a seen fingerprint is prior evidence, not a block;
  INVALID_EXECUTION-only history permits re-execution; a VALID authoritative
  result is surfaced for Phase 18;
* the four Phase 11 templates still fingerprint byte-identically to
  ``spec_for_params``;
* typed parameter bounds reproduce the frozen Phase 11 declared ranges;
* fail-loud locked-holdout guard on the input hypothesis and the produced spec;
* retry only on schema failure; turn / token budgets; prompt / version logging;
* no filesystem / registry / broker surface.
"""
from __future__ import annotations

import inspect
import json

import pytest
from alpha_agent.agents import (
    CompilerBudgetExceeded,
    CompilerContext,
    CompilerSchemaRetryExhausted,
    KnownStrategyRecord,
    ScriptedLLMClient,
    StrategyCompilerAgent,
    StrategyPlanRequest,
    assert_phase_11_bounds_match_declared_ranges,
    build_compiler_context,
    build_family_catalog,
    frozen_template_strategy_records,
)
from alpha_agent.agents import compiler_agent as compiler_agent_module
from alpha_agent.registry.holdout_guard import HoldoutAccessError
from alpha_agent.schemas.hypothesis import HypothesisSpec
from alpha_agent.strategy import StrategyCompiler, strategy_fingerprint
from alpha_agent.strategy.candidates_phase_13_5c import (
    ROOTS,
    baseline_params,
    spec_for_params,
)
from pydantic import ValidationError

UNIVERSE = ("ES", "NQ", "CL", "GC", "ZN")

_TEMPLATE_CANONICAL = {
    "tsmom": {"fast_horizon": 20, "slow_horizon": 120, "size": 1},
    "ma_trend": {"fast_window": 50, "slow_window": 200, "size": 1},
    "breakout": {"lookback": 55, "size": 1},
    "mean_reversion": {"zscore_window": 20, "entry_z": 2.0, "exit_z": 0.5, "size": 1},
}
# the feature KINDS each template's rules actually reference -- a hypothesis
# compiled to a template must approve exactly these.
_TEMPLATE_KIND = {
    "tsmom": ["diff"],
    "ma_trend": ["ma"],
    "breakout": ["breakout_up", "breakout_down"],
    "mean_reversion": ["zscore"],
}


def _hypothesis(**overrides) -> HypothesisSpec:
    base = {
        "hypothesis_id": "H-1",
        "title": "ES multi-week time-series momentum",
        "economic_mechanism": (
            "Gradual diffusion of macro information leaves ES index returns "
            "positively autocorrelated at multi-week horizons."
        ),
        "universe": ["ES"],
        "horizon": "20 trading days",
        "required_features": ["diff"],
        "signal_description": "Long when both a fast and a slow price change are positive.",
        "expected_regime": "trending macro regime",
        "failure_regime": "choppy range-bound regime",
        "falsification_test": "No positive OOS net PnL after costs across ES folds.",
        "novelty_notes": "",
    }
    base.update(overrides)
    return HypothesisSpec.model_validate(base)


def _context(**overrides) -> CompilerContext:
    kwargs = {"approved_universe": UNIVERSE}
    kwargs.update(overrides)
    return build_compiler_context(**kwargs)


def _template_plan(family="tsmom", root="ES", params=None, **overrides) -> str:
    base = {
        "expressible": True,
        "template": {
            "family_key": family,
            "root_symbol": root,
            "params": params if params is not None else dict(_TEMPLATE_CANONICAL[family]),
        },
        "rationale": "multi-horizon trend maps to the tsmom family",
    }
    base.update(overrides)
    return json.dumps(base)


# a novel trend + volatility-regime blueprint (mirrors examples.trend_example_spec)
def _blueprint_plan(root="NQ", *, drop_vol_rule=False, extra_unused_feature=False) -> str:
    features = [
        {"alias": "msp", "kind": "ma_spread", "params": {"fast": 20, "slow": 100}},
        {"alias": "vol", "kind": "volatility", "params": {"window": 60}},
    ]
    if extra_unused_feature:
        features.append({"alias": "rv", "kind": "realized_vol", "params": {"window": 30}})
    long_nodes = [
        {"type": "comparison", "op": "gt",
         "left": {"type": "feature", "feature": "msp"},
         "right": {"type": "const", "value": 0.0}},
    ]
    if not drop_vol_rule:
        long_nodes.append(
            {"type": "comparison", "op": "lt",
             "left": {"type": "feature", "feature": "vol"},
             "right": {"type": "const", "value": 0.02}}
        )
    rules = [
        {"rule_id": "trend_long", "target_units": 1,
         "when": {"type": "boolean", "op": "all", "nodes": long_nodes}},
        {"rule_id": "trend_short", "target_units": -1,
         "when": {"type": "comparison", "op": "lt",
                  "left": {"type": "feature", "feature": "msp"},
                  "right": {"type": "const", "value": 0.0}}},
    ]
    return json.dumps(
        {
            "expressible": True,
            "blueprint": {
                "root_symbol": root,
                "features": features,
                "rules": rules,
                "default_action": "flat",
            },
        }
    )


def _regime_hypothesis(**overrides) -> HypothesisSpec:
    base = {
        "hypothesis_id": "H-REGIME",
        "title": "trend premium conditioned on contained volatility",
        "universe": ["NQ"],
        "required_features": ["ma_spread", "volatility"],
        "signal_description": "Long trend only while realized volatility is contained.",
    }
    base.update(overrides)
    return _hypothesis(**base)


# -- blueprint: a novel multi-feature hypothesis compiles ----------------


def test_novel_multi_feature_blueprint_compiles_from_closed_primitives():
    result = StrategyCompilerAgent(ScriptedLLMClient([_blueprint_plan()])).compile_hypothesis(
        _regime_hypothesis(), _context()
    )
    assert result.accepted
    assert result.build_mode == "blueprint"
    assert result.family_key is None
    spec = result.strategy_spec
    assert spec is not None
    assert {b.spec.kind for b in StrategyCompiler().compile(spec).feature_bindings} == {
        "ma_spread",
        "volatility",
    }
    assert set(result.feature_coverage.represented) == {"ma_spread", "volatility"}
    assert result.feature_coverage.missing == ()


def test_blueprint_spec_preserves_frozen_phase_10_semantics():
    result = StrategyCompilerAgent(ScriptedLLMClient([_blueprint_plan()])).compile_hypothesis(
        _regime_hypothesis(), _context()
    )
    spec = result.strategy_spec
    plan = StrategyCompiler().compile(spec)  # unchanged Phase 10 compiler
    assert result.strategy_fingerprint == strategy_fingerprint(spec) == plan.fingerprint
    blob = json.dumps(spec.model_dump(mode="json")).lower()
    for banned in ("fill", "slippage", "commission", "latency", "margin", "pnl", "eval", "lambda"):
        assert banned not in blob


def test_blueprint_build_is_deterministic():
    a = StrategyCompilerAgent(ScriptedLLMClient([_blueprint_plan()])).compile_hypothesis(
        _regime_hypothesis(), _context()
    )
    b = StrategyCompilerAgent(ScriptedLLMClient([_blueprint_plan()])).compile_hypothesis(
        _regime_hypothesis(), _context()
    )
    assert a.strategy_fingerprint == b.strategy_fingerprint
    assert a.strategy_spec.strategy_id == b.strategy_spec.strategy_id


# -- arbitrary code / operators / features remain impossible ------------


def test_unknown_operator_is_a_schema_failure():
    bad = json.loads(_blueprint_plan())
    bad["blueprint"]["rules"][0]["when"] = {
        "type": "comparison", "op": "approx_eq",
        "left": {"type": "feature", "feature": "msp"},
        "right": {"type": "const", "value": 0.0},
    }
    client = ScriptedLLMClient([json.dumps(bad), json.dumps(bad), json.dumps(bad)])
    with pytest.raises(CompilerSchemaRetryExhausted):
        StrategyCompilerAgent(client, max_schema_retries=2).compile_hypothesis(
            _regime_hypothesis(), _context()
        )


def test_extra_field_in_blueprint_is_rejected_by_schema():
    bad = json.loads(_blueprint_plan())
    bad["blueprint"]["rules"][0]["python_expr"] = "os.system('rm -rf /')"
    client = ScriptedLLMClient([json.dumps(bad), json.dumps(bad), json.dumps(bad)])
    with pytest.raises(CompilerSchemaRetryExhausted):
        StrategyCompilerAgent(client, max_schema_retries=2).compile_hypothesis(
            _regime_hypothesis(), _context()
        )


def test_nested_or_callable_feature_param_is_a_schema_failure():
    bad = json.loads(_blueprint_plan())
    bad["blueprint"]["features"][0]["params"] = {"fast": {"$eval": "20*2"}}
    client = ScriptedLLMClient([json.dumps(bad), _blueprint_plan()])
    assert StrategyCompilerAgent(client).compile_hypothesis(
        _regime_hypothesis(), _context()
    ).accepted
    assert client.call_count == 2


def test_unknown_feature_kind_is_rejected_not_retried():
    bad = json.loads(_blueprint_plan())
    bad["blueprint"]["features"][0]["kind"] = "moon_phase_oscillator"
    result = StrategyCompilerAgent(
        ScriptedLLMClient([json.dumps(bad)]), max_schema_retries=3
    ).compile_hypothesis(_regime_hypothesis(), _context())
    assert not result.accepted
    assert result.rejection_code == "UNKNOWN_FEATURE_KIND"


def test_lag_operand_cannot_look_into_the_future():
    bad = json.loads(_blueprint_plan())
    bad["blueprint"]["rules"][0]["when"]["nodes"][0]["left"] = {
        "type": "lag", "feature": "msp", "periods": 0
    }
    client = ScriptedLLMClient([json.dumps(bad), json.dumps(bad), json.dumps(bad)])
    with pytest.raises(CompilerSchemaRetryExhausted):
        StrategyCompilerAgent(client, max_schema_retries=2).compile_hypothesis(
            _regime_hypothesis(), _context()
        )


def test_condition_referencing_undeclared_alias_is_rejected():
    bad = json.loads(_blueprint_plan())
    bad["blueprint"]["rules"][0]["when"]["nodes"][0]["left"] = {
        "type": "feature", "feature": "not_declared"
    }
    result = StrategyCompilerAgent(ScriptedLLMClient([json.dumps(bad)])).compile_hypothesis(
        _regime_hypothesis(), _context()
    )
    assert result.rejection_code == "INVALID_BLUEPRINT"


def test_unsafe_feature_is_rejected_by_the_phase_10_gate():
    # a roll-timing feature that is not signal-safe must be rejected by the
    # UNCHANGED Phase 10 compiler (or the catalog gate if it is not offered).
    bad = json.loads(_blueprint_plan())
    bad["blueprint"]["features"] = [
        {"alias": "roll", "kind": "bars_until_next_roll", "params": {}},
        {"alias": "msp", "kind": "ma_spread", "params": {"fast": 20, "slow": 100}},
    ]
    bad["blueprint"]["rules"] = [
        {"rule_id": "r", "target_units": 1,
         "when": {"type": "comparison", "op": "gt",
                  "left": {"type": "feature", "feature": "roll"},
                  "right": {"type": "const", "value": 3.0}}},
    ]
    result = StrategyCompilerAgent(ScriptedLLMClient([json.dumps(bad)])).compile_hypothesis(
        _hypothesis(universe=["NQ"], required_features=["bars_until_next_roll"]),
        _context(),
    )
    assert result.rejection_code in {
        "STRATEGY_COMPILE_REJECTED",
        "UNKNOWN_FEATURE_KIND",
        "HYPOTHESIS_FEATURE_UNAVAILABLE",
        "INVALID_BLUEPRINT",
    }
    assert result.strategy_spec is None


# -- required-feature coverage -----------------------------------------


def test_declared_but_unused_required_feature_is_not_sufficient():
    plan = _blueprint_plan(extra_unused_feature=True)
    result = StrategyCompilerAgent(ScriptedLLMClient([plan])).compile_hypothesis(
        _regime_hypothesis(required_features=["ma_spread", "volatility", "realized_vol"]),
        _context(),
    )
    assert not result.accepted
    assert result.rejection_code == "REQUIRED_FEATURE_NOT_REPRESENTED"
    assert "realized_vol" in result.rejection_detail


def test_template_missing_a_required_kind_is_rejected():
    result = StrategyCompilerAgent(ScriptedLLMClient([_template_plan()])).compile_hypothesis(
        _hypothesis(required_features=["diff", "realized_vol"]), _context()
    )
    assert result.rejection_code == "REQUIRED_FEATURE_NOT_REPRESENTED"


def test_blueprint_dropping_a_required_feature_rule_is_rejected():
    plan = _blueprint_plan(drop_vol_rule=True)
    result = StrategyCompilerAgent(ScriptedLLMClient([plan])).compile_hypothesis(
        _regime_hypothesis(), _context()
    )
    assert result.rejection_code == "REQUIRED_FEATURE_NOT_REPRESENTED"
    assert "volatility" in result.rejection_detail


def test_extra_rule_referenced_feature_not_in_hypothesis_is_rejected():
    # a valid, registered feature -- but the hypothesis only approved
    # ma_spread + volatility. Introducing realized_vol is a fidelity violation.
    plan = json.loads(_blueprint_plan())
    plan["blueprint"]["features"].append(
        {"alias": "rv", "kind": "realized_vol", "params": {"window": 30}}
    )
    plan["blueprint"]["rules"][0]["when"]["nodes"].append(
        {"type": "comparison", "op": "lt",
         "left": {"type": "feature", "feature": "rv"},
         "right": {"type": "const", "value": 0.5}}
    )
    result = StrategyCompilerAgent(ScriptedLLMClient([json.dumps(plan)])).compile_hypothesis(
        _regime_hypothesis(), _context()
    )
    assert not result.accepted
    assert result.rejection_code == "UNAPPROVED_FEATURE_ADDED"
    assert "realized_vol" in result.rejection_detail


def test_declared_but_unused_unapproved_feature_is_also_rejected():
    # even without referencing it, a declared feature enters the StrategySpec
    # fingerprint -- it is a research feature the hypothesis did not approve.
    plan = _blueprint_plan(extra_unused_feature=True)  # declares realized_vol, unused
    result = StrategyCompilerAgent(ScriptedLLMClient([plan])).compile_hypothesis(
        _regime_hypothesis(), _context()  # approves only ma_spread + volatility
    )
    assert result.rejection_code == "UNAPPROVED_FEATURE_ADDED"
    assert "realized_vol" in result.rejection_detail


def test_multiple_parameterizations_of_an_approved_kind_are_allowed():
    # two diff FeatureSpecs (n=10 and n=50) of the one approved kind 'diff',
    # combined with a boolean AND -- fine, it is still the 'diff' mechanism.
    plan = json.dumps(
        {
            "expressible": True,
            "blueprint": {
                "root_symbol": "NQ",
                "features": [
                    {"alias": "d_fast", "kind": "diff", "params": {"n": 10}},
                    {"alias": "d_slow", "kind": "diff", "params": {"n": 50}},
                ],
                "rules": [
                    {"rule_id": "long", "target_units": 1,
                     "when": {"type": "boolean", "op": "all", "nodes": [
                         {"type": "comparison", "op": "gt",
                          "left": {"type": "feature", "feature": "d_fast"},
                          "right": {"type": "const", "value": 0.0}},
                         {"type": "comparison", "op": "gt",
                          "left": {"type": "feature", "feature": "d_slow"},
                          "right": {"type": "const", "value": 0.0}}]}},
                    {"rule_id": "short", "target_units": -1,
                     "when": {"type": "comparison", "op": "lt",
                              "left": {"type": "lag", "feature": "d_fast", "periods": 1},
                              "right": {"type": "const", "value": 0.0}}},
                ],
                "default_action": "flat",
            },
        }
    )
    result = StrategyCompilerAgent(ScriptedLLMClient([plan])).compile_hypothesis(
        _hypothesis(universe=["NQ"], required_features=["diff"]), _context()
    )
    assert result.accepted, result.rejection_code
    assert result.feature_coverage.referenced_kinds == ("diff",)
    assert result.feature_coverage.declared_kinds == ("diff",)
    assert result.feature_coverage.missing == ()
    assert result.feature_coverage.unapproved == ()


# -- templates still fingerprint identically --------------------------


def test_four_baseline_templates_fingerprint_identically():
    ctx = _context()
    for family, params in _TEMPLATE_CANONICAL.items():
        for root in ROOTS:
            hyp = _hypothesis(universe=[root], required_features=_TEMPLATE_KIND[family])
            plan = _template_plan(family=family, root=root, params=params)
            result = StrategyCompilerAgent(ScriptedLLMClient([plan])).compile_hypothesis(hyp, ctx)
            assert result.accepted, (family, root, result.rejection_code)
            rebuilt = spec_for_params(family, {"root_symbol": root, **params})
            assert result.strategy_fingerprint == strategy_fingerprint(rebuilt)
            assert result.strategy_spec.strategy_id == rebuilt.strategy_id


def test_template_canonical_params_match_baseline_params_helper():
    for family, params in _TEMPLATE_CANONICAL.items():
        assert baseline_params(family, "ES") == {"root_symbol": "ES", **params}


# -- typed bounds reproduce the frozen Phase 11 declared ranges ---------


def test_typed_param_bounds_match_declared_prose_ranges():
    assert_phase_11_bounds_match_declared_ranges()


def test_template_out_of_declared_bounds_is_rejected():
    result = StrategyCompilerAgent(
        ScriptedLLMClient([_template_plan(params={"fast_horizon": 3, "slow_horizon": 120, "size": 1})])
    ).compile_hypothesis(_hypothesis(), _context())
    assert result.rejection_code == "PARAMETER_OUT_OF_BOUNDS"
    assert "fast_horizon" in result.rejection_detail


def test_template_non_integer_for_integer_parameter_is_rejected():
    result = StrategyCompilerAgent(
        ScriptedLLMClient([_template_plan(params={"fast_horizon": 20.5, "slow_horizon": 120, "size": 1})])
    ).compile_hypothesis(_hypothesis(), _context())
    assert result.rejection_code == "NON_INTEGER_PARAMETER"


def test_template_cross_parameter_constraint_violation_is_rejected():
    result = StrategyCompilerAgent(
        ScriptedLLMClient([_template_plan(params={"fast_horizon": 55, "slow_horizon": 40, "size": 1})])
    ).compile_hypothesis(_hypothesis(), _context())
    assert result.rejection_code == "INVALID_STRATEGY_PARAMETERS"


def test_unknown_template_family_is_rejected():
    result = StrategyCompilerAgent(
        ScriptedLLMClient([_template_plan(family="silver_bullet", params={"liquidity_lookback": 20})])
    ).compile_hypothesis(_hypothesis(), _context())
    assert result.rejection_code == "UNKNOWN_STRATEGY_FAMILY"


# -- duplicate semantics: strategy fingerprint is EVIDENCE, never a block ----


def _tsmom_nq_fingerprint() -> str:
    return strategy_fingerprint(
        spec_for_params(
            "tsmom", {"root_symbol": "NQ", "fast_horizon": 20, "slow_horizon": 120, "size": 1}
        )
    )


def test_seen_fingerprint_alone_does_not_block():
    hyp = _hypothesis(universe=["NQ"])
    plan = _template_plan(family="tsmom", root="NQ")
    result = StrategyCompilerAgent(ScriptedLLMClient([plan])).compile_hypothesis(hyp, _context())
    assert result.accepted
    de = result.duplicate_evidence
    assert de.strategy_fingerprint_seen is True  # in the frozen candidate manifest
    assert de.prior_valid_authoritative_results == 0
    assert de.prior_invalid_execution_attempts == 0


def test_invalid_execution_only_history_is_not_blocked():
    fp = _tsmom_nq_fingerprint()
    ctx = _context(
        known_strategies=[
            KnownStrategyRecord(
                strategy_fingerprint=fp,
                source="experiment_registry",
                experiment_id="EXP-INV",
                has_valid_authoritative_result=False,
                invalid_execution_attempts=3,
                valid_execution_attempts=0,
            )
        ]
    )
    result = StrategyCompilerAgent(
        ScriptedLLMClient([_template_plan(family="tsmom", root="NQ")])
    ).compile_hypothesis(_hypothesis(universe=["NQ"]), ctx)
    assert result.accepted
    assert result.rejection_code is None
    de = result.duplicate_evidence
    assert de.prior_invalid_execution_attempts == 3
    assert de.prior_valid_authoritative_results == 0


def test_valid_authoritative_result_is_surfaced_as_evidence_not_blocked():
    fp = _tsmom_nq_fingerprint()
    ctx = _context(
        known_strategies=[
            KnownStrategyRecord(
                strategy_fingerprint=fp,
                source="experiment_registry",
                experiment_id="EXP-AUTH",
                experiment_identity="experiment1:deadbeef",
                has_valid_authoritative_result=True,
                valid_execution_attempts=1,
            )
        ]
    )
    result = StrategyCompilerAgent(
        ScriptedLLMClient([_template_plan(family="tsmom", root="NQ")])
    ).compile_hypothesis(_hypothesis(universe=["NQ"]), ctx)
    # Phase 17 does NOT block on a strategy fingerprint, even an authoritative one
    assert result.accepted
    assert result.rejection_code is None
    assert result.strategy_spec is not None
    de = result.duplicate_evidence
    assert de.strategy_fingerprint_seen is True
    assert de.prior_valid_authoritative_results == 1
    assert de.prior_experiment_identities == ("experiment1:deadbeef",)
    assert "EXP-AUTH" in de.prior_experiment_ids
    assert "Phase 18" in de.advisory


def test_same_fingerprint_prior_valid_result_but_a_new_scientific_identity_still_compiles():
    # Phase 17 cannot construct the scientific experiment_identity, so even a
    # fingerprint carrying a prior VALID authoritative result compiles normally --
    # the DIFFERENT future identity is Phase 18's to establish and adjudicate.
    fp = _tsmom_nq_fingerprint()
    ctx = _context(
        known_strategies=[
            KnownStrategyRecord(
                strategy_fingerprint=fp,
                source="experiment_registry",
                experiment_id="EXP-OLD",
                experiment_identity="experiment1:OLD_VALIDATIONSPEC",
                has_valid_authoritative_result=True,
                valid_execution_attempts=1,
            )
        ]
    )
    result = StrategyCompilerAgent(
        ScriptedLLMClient([_template_plan(family="tsmom", root="NQ")])
    ).compile_hypothesis(_hypothesis(universe=["NQ"]), ctx)
    assert result.accepted
    assert result.rejection_code is None
    assert result.strategy_fingerprint == fp
    assert result.duplicate_evidence.prior_experiment_identities == (
        "experiment1:OLD_VALIDATIONSPEC",
    )


def test_phase_17_never_makes_the_final_scientific_duplicate_decision():
    from alpha_agent.agents import REJECTION_CODES, DuplicateEvidence

    # no fingerprint-only rejection code exists at all
    assert "DUPLICATE_AUTHORITATIVE_RESULT" not in REJECTION_CODES
    assert not any("DUPLICATE" in c for c in REJECTION_CODES)
    # DuplicateEvidence carries no decision field, only evidence + an advisory
    fields = set(DuplicateEvidence.model_fields)
    assert "reexecution_permitted" not in fields
    assert {"strategy_fingerprint_seen", "prior_valid_authoritative_results", "advisory"} <= fields


def test_frozen_template_records_carry_no_authority():
    for rec in frozen_template_strategy_records():
        assert rec.source == "frozen_candidate_manifest"
        assert rec.has_valid_authoritative_result is False


# -- non-expressible: reject, do not invent -----------------------------


def test_non_expressible_hypothesis_is_a_typed_rejection():
    plan = json.dumps(
        {
            "expressible": False,
            "not_expressible_reason": (
                "the hypothesis needs an options-implied-vol term-structure signal; "
                "no registered feature kind expresses it"
            ),
        }
    )
    result = StrategyCompilerAgent(ScriptedLLMClient([plan]), max_schema_retries=3).compile_hypothesis(
        _hypothesis(), _context()
    )
    assert not result.accepted
    assert result.rejection_code == "HYPOTHESIS_NOT_EXPRESSIBLE"
    assert result.strategy_spec is None
    assert result.attempts[-1].parsed_ok is True


def test_plan_with_both_template_and_blueprint_is_a_schema_retry():
    bad = json.loads(_template_plan())
    bad["blueprint"] = json.loads(_blueprint_plan())["blueprint"]
    client = ScriptedLLMClient([json.dumps(bad), _template_plan()])
    assert StrategyCompilerAgent(client).compile_hypothesis(_hypothesis(), _context()).accepted
    assert client.call_count == 2


def test_missing_required_feature_kind_is_rejected():
    result = StrategyCompilerAgent(ScriptedLLMClient([_template_plan()])).compile_hypothesis(
        _hypothesis(required_features=["moon_phase"]), _context()
    )
    assert result.rejection_code == "HYPOTHESIS_FEATURE_UNAVAILABLE"


# -- markets / universe --------------------------------------------------


def test_market_outside_universe_is_rejected():
    result = StrategyCompilerAgent(
        ScriptedLLMClient([_template_plan(root="SPY")])
    ).compile_hypothesis(_hypothesis(universe=["ES", "SPY"]), _context())
    assert result.rejection_code == "MARKET_OUTSIDE_UNIVERSE"


def test_root_outside_hypothesis_universe_is_rejected():
    result = StrategyCompilerAgent(
        ScriptedLLMClient([_template_plan(root="NQ")])
    ).compile_hypothesis(_hypothesis(universe=["ES"]), _context())
    assert result.rejection_code == "ROOT_OUTSIDE_HYPOTHESIS_UNIVERSE"


# -- execution semantics ----------------------------------------------


def test_execution_semantics_are_stated_and_fixed():
    result = StrategyCompilerAgent(ScriptedLLMClient([_blueprint_plan()])).compile_hypothesis(
        _regime_hypothesis(), _context()
    )
    es = result.execution_semantics
    assert es.same_bar_fills_prohibited is True
    assert es.execution_cadence == "native_1m_raw_contract"
    assert "next eligible bar" in es.entry_timing
    assert "stop / target / bracket" in es.exit_timing
    assert es.cost_assumptions["commission_per_contract_usd"] == 2.0
    assert es.cost_assumptions["latency_bars"] == 0.0


# -- retry only on schema failure ------------------------------------


def test_schema_failure_then_success_retries_once():
    client = ScriptedLLMClient(["not json at all", _template_plan()])
    result = StrategyCompilerAgent(client, max_schema_retries=2).compile_hypothesis(
        _hypothesis(), _context()
    )
    assert result.accepted
    assert client.call_count == 2
    assert any("did not validate" in m["content"] for m in client.calls[1]["messages"])


def test_markdown_fenced_and_prose_wrapped_json_is_extracted():
    fenced = "```json\n" + _template_plan() + "\n```"
    assert StrategyCompilerAgent(ScriptedLLMClient([fenced])).compile_hypothesis(
        _hypothesis(), _context()
    ).accepted
    wrapped = "Here it is: " + _template_plan() + " -- done."
    assert StrategyCompilerAgent(ScriptedLLMClient([wrapped])).compile_hypothesis(
        _hypothesis(), _context()
    ).accepted


# -- locked holdout -------------------------------------------------


def test_holdout_reference_in_hypothesis_fails_loud():
    poisoned = _hypothesis(falsification_test="Evaluate on the 2025-03-01 to 2025-06-01 window.")
    with pytest.raises(HoldoutAccessError):
        StrategyCompilerAgent(ScriptedLLMClient([_template_plan()])).compile_hypothesis(
            poisoned, _context()
        )


def test_context_with_holdout_value_is_refused():
    with pytest.raises(HoldoutAccessError):
        build_compiler_context(
            approved_universe=["ES"],
            known_strategies=[
                KnownStrategyRecord(
                    strategy_fingerprint="stratdsl1:x", source="run dated 2025-02-01"
                )
            ],
        )
    with pytest.raises(ValidationError):
        CompilerContext(
            approved_universe=("ES",),
            feature_catalog=build_compiler_context(approved_universe=["ES"]).feature_catalog,
            family_catalog=build_family_catalog(),
            known_strategies=(
                KnownStrategyRecord(strategy_fingerprint="fp", source="x 2025-09-09"),
            ),
        )


# -- budgets ------------------------------------------------------


def test_token_budget_stops_the_loop():
    from alpha_agent.agents.llm import LLMResponse

    big = LLMResponse(text="garbage", model="scripted-model", output_tokens=10_000)
    client = ScriptedLLMClient([big, big, big])
    agent = StrategyCompilerAgent(client, max_schema_retries=5, token_budget=15_000)
    with pytest.raises(CompilerBudgetExceeded):
        agent.compile_hypothesis(_hypothesis(), _context())
    assert client.call_count == 2


# -- prompt / version logging ------------------------------------


def test_prompt_log_is_populated_and_stable():
    seen: list = []
    agent = StrategyCompilerAgent(ScriptedLLMClient([_template_plan()]), log_sink=seen.append)
    hyp, ctx = _hypothesis(), _context()
    result = agent.compile_hypothesis(hyp, ctx)
    log = result.prompt_log
    assert log.prompt_version == agent.prompt_version
    assert len(log.prompt_version) == 16
    assert log.context_sha256 == ctx.sha256_with_hypothesis(hyp)
    assert log.model == "claude-sonnet-5"
    assert seen and seen[0] == log

    agent2 = StrategyCompilerAgent(ScriptedLLMClient([_template_plan()]))
    log2 = agent2.compile_hypothesis(hyp, ctx).prompt_log
    assert (log2.prompt_version, log2.system_prompt_sha256, log2.context_sha256) == (
        log.prompt_version,
        log.system_prompt_sha256,
        log.context_sha256,
    )


# -- no filesystem / registry / broker surface -----------------------


def test_agent_module_has_no_execution_or_io_surface():
    src = inspect.getsource(compiler_agent_module)
    for forbidden in (
        "subprocess",
        "databento",
        "sqlite3",
        "requests",
        "socket",
        "broker",
        "ExperimentRegistry",
    ):
        assert forbidden not in src, f"compiler agent must not reference {forbidden!r}"


def test_agent_does_not_accept_a_registry_or_path_handle():
    params = set(inspect.signature(StrategyCompilerAgent.__init__).parameters)
    assert params == {
        "self",
        "client",
        "prompt_path",
        "model",
        "max_schema_retries",
        "max_output_tokens",
        "temperature",
        "token_budget",
        "log_sink",
    }


def test_compile_rejects_non_typed_input():
    agent = StrategyCompilerAgent(ScriptedLLMClient([_template_plan(), _template_plan()]))
    with pytest.raises(TypeError):
        agent.compile_hypothesis({"not": "a hypothesis"}, _context())
    with pytest.raises(TypeError):
        agent.compile_hypothesis(_hypothesis(), {"not": "a context"})


# -- schema-level closedness -----------------------------------------


def test_plan_request_rejects_unknown_top_level_field():
    with pytest.raises(ValidationError):
        StrategyPlanRequest.model_validate(
            {"expressible": True, "template": None, "blueprint": None, "code": "x"}
        )


def test_blueprint_operand_boolean_value_is_rejected():
    bad = json.loads(_blueprint_plan())
    bad["blueprint"]["rules"][0]["when"]["nodes"][0]["right"]["value"] = True
    with pytest.raises(ValidationError):
        StrategyPlanRequest.model_validate(bad)
