"""Phase 10 -- Closed Typed Strategy DSL and Compiler.

Covers checklist A-V of prompts/10_STRATEGY_DSL_AND_COMPILER_BOUNDARY.md.
Deterministic, no network; FeatureFrames are built from the hand-computable
Phase 09 fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest
from alpha_agent.features import FeatureSpec, compute_features
from alpha_agent.features.safety import FeatureSafetyError
from alpha_agent.strategy import (
    ReferenceEvaluator,
    StrategyCompileError,
    StrategyEvaluationError,
    compile_strategy,
    strategy_from_json,
    strategy_from_yaml,
    strategy_to_json,
    strategy_to_yaml,
)
from alpha_agent.strategy import errors as E
from alpha_agent.strategy.compiler import CompileLimits
from alpha_agent.strategy.enums import BooleanOp, Comparator, DefaultAction, MissingRulePolicy
from alpha_agent.strategy.examples import trend_example_spec
from alpha_agent.strategy.spec import (
    BooleanNode,
    ComparisonNode,
    ConstOperand,
    FeatureDeclaration,
    FeatureOperand,
    LagOperand,
    NotNode,
    Rule,
    StrategySpec,
    TargetAction,
)
from features_fixtures import source_from_close, synthetic_roll_sources
from pydantic import ValidationError


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _feat(alias: str) -> FeatureOperand:
    return FeatureOperand(feature=alias)


def _c(v: float) -> ConstOperand:
    return ConstOperand(value=v)


def _cmp(alias, op, v) -> ComparisonNode:
    return ComparisonNode(op=op, left=_feat(alias), right=_c(v))


def _mom_spec(
    *,
    default: DefaultAction = DefaultAction.FLAT,
    on_missing: MissingRulePolicy = MissingRulePolicy.SKIP_RULE,
    strategy_id: str = "T-MOM-001",
    name: str = "momentum demo",
) -> StrategySpec:
    return StrategySpec(
        strategy_name=name,
        strategy_id=strategy_id,
        root_symbol="NQ",
        on_missing=on_missing,
        features=[
            FeatureDeclaration(alias="mom", spec=FeatureSpec(kind="diff", params={"n": 1}))
        ],
        rules=[
            Rule(rule_id="long", when=_cmp("mom", Comparator.GT, 0.0), action=TargetAction(target_units=1)),
            Rule(rule_id="short", when=_cmp("mom", Comparator.LT, 0.0), action=TargetAction(target_units=-1)),
        ],
        default_action=default,
    )


def _mom_frame(closes):
    return compute_features(source_from_close(closes), [FeatureSpec(kind="diff", params={"n": 1})])


UP_DOWN = list(100 + np.arange(6) * 1.0) + list(105 - np.arange(5) * 2.0)


# ==========================================================================
# A. valid simple StrategySpec compiles
# ==========================================================================
def test_A_valid_spec_compiles():
    plan = compile_strategy(trend_example_spec())
    assert plan.required_features == ("ma_spread_20_100", "volatility_60")
    assert plan.warmup_bars == 100
    assert plan.default_action is DefaultAction.FLAT
    assert plan.fingerprint.startswith("stratdsl1:")
    assert len(plan.rules) == 2


# ==========================================================================
# B. unknown feature rejected
# ==========================================================================
def test_B_unknown_feature_rejected():
    spec = _mom_spec()
    bad = spec.model_copy(
        update={
            "features": [
                FeatureDeclaration(alias="mom", spec=FeatureSpec(kind="not_a_feature", params={}))
            ]
        }
    )
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(bad)
    assert ei.value.has(E.UNKNOWN_FEATURE)


def test_B2_unknown_feature_param_rejected():
    bad = _mom_spec().model_copy(
        update={
            "features": [
                FeatureDeclaration(
                    alias="mom", spec=FeatureSpec(kind="diff", params={"n": 1, "bogus": 3})
                )
            ]
        }
    )
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(bad)
    assert ei.value.has(E.INVALID_FEATURE_PARAMS)


# ==========================================================================
# C. unsafe retrospective feature rejected
# ==========================================================================
def test_C_retrospective_roll_feature_rejected_at_compile():
    bad = _mom_spec().model_copy(
        update={
            "features": [
                FeatureDeclaration(
                    alias="mom", spec=FeatureSpec(kind="bars_until_next_roll", params={})
                )
            ]
        }
    )
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(bad)
    assert ei.value.has(E.UNSAFE_FEATURE)


def test_C2_retrospective_back_adjusted_source_rejected_at_evaluation():
    src = synthetic_roll_sources()["backadj_retro"]
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 3})], require_point_in_time=False)
    with pytest.raises(FeatureSafetyError):
        ff.assert_signal_safe()

    spec = _mom_spec().model_copy(
        update={
            "features": [
                FeatureDeclaration(alias="mom", spec=FeatureSpec(kind="ma", params={"window": 3}))
            ],
            "rules": [
                Rule(rule_id="long", when=_cmp("mom", Comparator.GT, 0.0), action=TargetAction(target_units=1))
            ],
        }
    )
    plan = compile_strategy(spec)
    with pytest.raises(StrategyEvaluationError):
        ReferenceEvaluator(plan).evaluate_frame(ff)


# ==========================================================================
# D. point-in-time BackAdjusted feature accepted for signals
# ==========================================================================
def test_D_point_in_time_back_adjusted_feature_accepted():
    src = synthetic_roll_sources()["backadj_pit"]
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 3})])
    ff.assert_signal_safe()  # no raise

    spec = _mom_spec().model_copy(
        update={
            "features": [
                FeatureDeclaration(alias="mom", spec=FeatureSpec(kind="ma", params={"window": 3}))
            ],
            "rules": [
                Rule(rule_id="long", when=_cmp("mom", Comparator.GT, 0.0), action=TargetAction(target_units=1))
            ],
        }
    )
    plan = compile_strategy(spec)
    decisions = ReferenceEvaluator(plan).evaluate_frame(ff)
    assert len(decisions) == len(ff.features)


# ==========================================================================
# E. FeatureRef cannot specify execution price
# ==========================================================================
def test_E_feature_ref_cannot_carry_execution_price():
    with pytest.raises(ValidationError):
        FeatureOperand.model_validate({"type": "feature", "feature": "mom", "fill_price": 30000.0})
    with pytest.raises(ValidationError):
        FeatureDeclaration.model_validate(
            {"alias": "mom", "spec": {"kind": "diff", "params": {"n": 1}}, "reference_price": 1.0}
        )


def test_E2_execution_fields_rejected_anywhere_in_payload():
    payload = _mom_spec().model_dump(mode="json")
    payload["rules"][0]["action"]["slippage_ticks"] = 2
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(payload)
    assert ei.value.has(E.ILLEGAL_DSL_FIELD)


# ==========================================================================
# F. unknown DSL field rejected
# ==========================================================================
def test_F_unknown_dsl_field_rejected():
    payload = _mom_spec().model_dump(mode="json")
    payload["surprise"] = True
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(payload)
    assert ei.value.has(E.ILLEGAL_DSL_FIELD)
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(payload)


# ==========================================================================
# G. unknown operator rejected
# ==========================================================================
def test_G_unknown_operator_rejected():
    payload = _mom_spec().model_dump(mode="json")
    payload["rules"][0]["when"]["op"] = "approximately_equals"
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(payload)
    assert ei.value.has(E.UNKNOWN_OPERATOR)


def test_G2_unknown_action_rejected():
    payload = _mom_spec().model_dump(mode="json")
    payload["default_action"] = "moon"
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(payload)
    assert ei.value.has(E.UNKNOWN_ACTION)


# ==========================================================================
# H. ALL / ANY / NOT semantics
# ==========================================================================
def test_H_all_any_not_three_valued_semantics():
    from alpha_agent.strategy.enums import NodeType, OperandType
    from alpha_agent.strategy.evaluator import _FALSE, _TRUE, _UNKNOWN, _eval_condition
    from alpha_agent.strategy.plan import CompiledCondition, CompiledOperand

    def leaf(name):
        return CompiledCondition(
            node=NodeType.COMPARISON,
            op=Comparator.GT,
            left=CompiledOperand(kind=OperandType.FEATURE, feature=name),
            right=CompiledOperand(kind=OperandType.CONST, value=0.0),
        )

    env = {"t": 1.0, "f": -1.0, "u": None}
    resolve = lambda op: env[op.feature] if op.kind is OperandType.FEATURE else op.value

    ALL = lambda *n: CompiledCondition(node=NodeType.BOOLEAN, bool_op=BooleanOp.ALL, children=tuple(n))
    ANY = lambda *n: CompiledCondition(node=NodeType.BOOLEAN, bool_op=BooleanOp.ANY, children=tuple(n))
    NOT = lambda n: CompiledCondition(node=NodeType.NOT, children=(n,))

    assert _eval_condition(ALL(leaf("t"), leaf("t")), resolve) is _TRUE
    assert _eval_condition(ALL(leaf("t"), leaf("f")), resolve) is _FALSE
    assert _eval_condition(ALL(leaf("t"), leaf("u")), resolve) is _UNKNOWN
    assert _eval_condition(ALL(leaf("f"), leaf("u")), resolve) is _FALSE   # F dominates U
    assert _eval_condition(ANY(leaf("f"), leaf("f")), resolve) is _FALSE
    assert _eval_condition(ANY(leaf("f"), leaf("t")), resolve) is _TRUE
    assert _eval_condition(ANY(leaf("f"), leaf("u")), resolve) is _UNKNOWN
    assert _eval_condition(ANY(leaf("t"), leaf("u")), resolve) is _TRUE    # T dominates U
    assert _eval_condition(NOT(leaf("t")), resolve) is _FALSE
    assert _eval_condition(NOT(leaf("f")), resolve) is _TRUE
    assert _eval_condition(NOT(leaf("u")), resolve) is _UNKNOWN


# ==========================================================================
# I. ordered first-match rule precedence
# ==========================================================================
def test_I_first_matching_rule_wins():
    spec = StrategySpec(
        strategy_name="precedence", strategy_id="T-PREC-001", root_symbol="NQ",
        features=[FeatureDeclaration(alias="mom", spec=FeatureSpec(kind="diff", params={"n": 1}))],
        rules=[
            Rule(rule_id="first", when=_cmp("mom", Comparator.GT, 0.0), action=TargetAction(target_units=2)),
            Rule(rule_id="second", when=_cmp("mom", Comparator.GT, 0.0), action=TargetAction(target_units=1)),
        ],
        default_action=DefaultAction.FLAT,
    )
    plan = compile_strategy(spec, limits=CompileLimits(max_abs_target_units=5))
    ds = ReferenceEvaluator(plan).evaluate_frame(_mom_frame(list(100 + np.arange(6) * 1.0)))
    matched = [d for d in ds if d.matched_rule_id]
    assert matched and all(d.matched_rule_id == "first" and d.target_units == 2 for d in matched)


# ==========================================================================
# J. missing feature behavior
# ==========================================================================
def test_J_missing_feature_makes_rule_not_evaluable():
    plan = compile_strategy(_mom_spec(default=DefaultAction.FLAT))
    ds = ReferenceEvaluator(plan).evaluate_frame(_mom_frame(UP_DOWN))
    first = ds[0]  # diff_1 has no history on the first bar
    assert first.matched_rule_id is None
    assert first.default_applied is True
    assert first.target_units == 0
    assert "diff_1" in first.missing_features
    assert set(first.not_evaluable_rule_ids) == {"long", "short"}
    # NaN did not become True or False
    assert first.referenced_feature_values["diff_1"] is None


def test_J2_hold_policy_stops_rule_scan():
    plan = compile_strategy(_mom_spec(on_missing=MissingRulePolicy.HOLD, default=DefaultAction.FLAT))
    ds = ReferenceEvaluator(plan).evaluate_frame(_mom_frame(UP_DOWN))
    assert ds[0].not_evaluable_rule_ids == ("long",)  # stopped after the first


# ==========================================================================
# K. target units integral validation
# ==========================================================================
def test_K_target_units_must_be_integral():
    for v in (1.5, 1.0, "1", True, False, None):
        with pytest.raises(ValidationError):
            TargetAction(target_units=v)
    for v in (1, 0, -2):
        assert TargetAction(target_units=v).target_units == v


def test_K2_target_units_bounds_enforced_by_compiler():
    spec = _mom_spec().model_copy()
    over = spec.model_copy(
        update={
            "rules": [
                Rule(rule_id="long", when=_cmp("mom", Comparator.GT, 0.0), action=TargetAction(target_units=50))
            ]
        }
    )
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(over, limits=CompileLimits(max_abs_target_units=3))
    assert ei.value.has(E.INVALID_TARGET_UNITS)


# ==========================================================================
# L. KEEP_PREVIOUS_TARGET deterministic state
# ==========================================================================
def test_L_keep_previous_target_state():
    # A rule that only ever fires long; nothing sets short; between fires the
    # default keeps the last emitted target.
    spec = StrategySpec(
        strategy_name="kp", strategy_id="T-KP-001", root_symbol="NQ",
        features=[FeatureDeclaration(alias="mom", spec=FeatureSpec(kind="diff", params={"n": 1}))],
        rules=[Rule(rule_id="long", when=_cmp("mom", Comparator.GT, 0.5), action=TargetAction(target_units=1))],
        default_action=DefaultAction.KEEP_PREVIOUS_TARGET,
    )
    plan = compile_strategy(spec)
    # up (diff=+1 -> long), then flat-ish (diff=0 -> keep), then up again
    closes = [100, 101, 101, 101, 102]
    ds = ReferenceEvaluator(plan).evaluate_frame(_mom_frame(closes))
    targets = [d.target_units for d in ds]
    assert targets == [0, 1, 1, 1, 1]
    assert ds[2].default_applied and ds[2].matched_rule_id is None


# ==========================================================================
# M. invalid NaN/inf constants rejected
# ==========================================================================
def test_M_nan_inf_constants_rejected():
    with pytest.raises(ValidationError):
        ConstOperand(value=float("nan"))
    with pytest.raises(ValidationError):
        ConstOperand(value=float("inf"))
    payload = _mom_spec().model_dump(mode="json")
    payload["rules"][0]["when"]["right"]["value"] = "NaN"
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(payload)
    assert ei.value.has(E.INVALID_CONSTANT)


# ==========================================================================
# N. canonical serialization round-trip
# ==========================================================================
def test_N_serialization_round_trip_json_and_yaml():
    spec = trend_example_spec()
    for loads, dumps in ((strategy_from_json, strategy_to_json), (strategy_from_yaml, strategy_to_yaml)):
        rt = loads(dumps(spec))
        assert rt.model_dump() == spec.model_dump()
        assert rt.fingerprint == spec.fingerprint
        assert dumps(rt) == dumps(spec)  # stable / deterministic text


# ==========================================================================
# O. stable fingerprint  +  cosmetic-only changes don't move it
# ==========================================================================
def test_O_fingerprint_stable_under_cosmetic_change():
    a = trend_example_spec()
    b = StrategySpec(
        strategy_name="totally different name",
        strategy_id="OTHER-ID-999",
        root_symbol="NQ",
        rationale="different rationale",
        metadata={"author": "someone"},
        features=[
            FeatureDeclaration(alias="x1", spec=FeatureSpec(kind="volatility", params={"window": 60})),
            FeatureDeclaration(alias="x2", spec=FeatureSpec(kind="ma_spread", params={"fast": 20, "slow": 100})),
        ],
        rules=[
            Rule(
                rule_id="renamed_long",
                rationale="reworded",
                when=BooleanNode(
                    op=BooleanOp.ALL,
                    nodes=[  # child order swapped
                        _cmp("x1", Comparator.LT, 0.02),
                        _cmp("x2", Comparator.GT, 0.0),
                    ],
                ),
                action=TargetAction(target_units=1),
            ),
            Rule(
                rule_id="renamed_short",
                when=_cmp("x2", Comparator.LT, 0.0),
                action=TargetAction(target_units=-1),
            ),
        ],
        default_action=DefaultAction.FLAT,
    )
    assert a.fingerprint == b.fingerprint
    assert compile_strategy(a).fingerprint == compile_strategy(b).fingerprint


# ==========================================================================
# P. free-text rationale does not change fingerprint
# ==========================================================================
def test_P_rationale_does_not_change_fingerprint():
    a = _mom_spec()
    b = a.model_copy(update={"rationale": "a completely different explanation"})
    c = a.model_copy(
        update={"rules": [r.model_copy(update={"rationale": "why " + r.rule_id}) for r in a.rules]}
    )
    assert a.fingerprint == b.fingerprint == c.fingerprint


# ==========================================================================
# Q. semantic rule change DOES change fingerprint
# ==========================================================================
def test_Q_semantic_change_moves_fingerprint():
    base = _mom_spec()
    fp = base.fingerprint

    comparator = base.model_copy(
        update={"rules": [
            base.rules[0].model_copy(update={"when": _cmp("mom", Comparator.GTE, 0.0)}),
            base.rules[1],
        ]}
    )
    constant = base.model_copy(
        update={"rules": [
            base.rules[0].model_copy(update={"when": _cmp("mom", Comparator.GT, 0.5)}),
            base.rules[1],
        ]}
    )
    target = base.model_copy(
        update={"rules": [
            base.rules[0].model_copy(update={"action": TargetAction(target_units=2)}),
            base.rules[1],
        ]}
    )
    reordered = base.model_copy(update={"rules": [base.rules[1], base.rules[0]]})
    default_changed = base.model_copy(update={"default_action": DefaultAction.KEEP_PREVIOUS_TARGET})
    param_changed = base.model_copy(
        update={"features": [
            FeatureDeclaration(alias="mom", spec=FeatureSpec(kind="diff", params={"n": 2}))
        ]}
    )

    for variant in (comparator, constant, target, reordered, default_changed, param_changed):
        assert variant.fingerprint != fp


# ==========================================================================
# R. prefix-invariant decision at T
# ==========================================================================
def test_R_prefix_invariant_decision_incl_lag_and_keep_previous():
    spec = StrategySpec(
        strategy_name="prefix", strategy_id="T-PFX-001", root_symbol="NQ",
        features=[FeatureDeclaration(alias="ma", spec=FeatureSpec(kind="ma", params={"window": 3}))],
        rules=[
            Rule(
                rule_id="rising",
                when=ComparisonNode(op=Comparator.GT, left=_feat("ma"), right=LagOperand(feature="ma", periods=2)),
                action=TargetAction(target_units=1),
            )
        ],
        default_action=DefaultAction.KEEP_PREVIOUS_TARGET,
    )
    plan = compile_strategy(spec)
    closes = list(100 + np.cumsum(np.array([0, 1, -1, 2, 1, -3, 4, 2, -1, 0], dtype=float)))
    full = ReferenceEvaluator(plan).evaluate_frame(
        compute_features(source_from_close(closes), [FeatureSpec(kind="ma", params={"window": 3})])
    )
    for t in range(4, len(closes)):
        short = ReferenceEvaluator(plan).evaluate_frame(
            compute_features(source_from_close(closes[: t + 1]), [FeatureSpec(kind="ma", params={"window": 3})])
        )
        assert short[t].model_dump() == full[t].model_dump()


# ==========================================================================
# S. no future temporal access
# ==========================================================================
def test_S_no_future_temporal_primitive():
    with pytest.raises(ValidationError):
        LagOperand(feature="mom", periods=0)
    with pytest.raises(ValidationError):
        LagOperand(feature="mom", periods=-1)
    # there is no lead / future operand type in the closed vocabulary
    from alpha_agent.strategy.enums import OperandType
    assert {o.value for o in OperandType} == {"feature", "lag", "const"}
    payload = _mom_spec().model_dump(mode="json")
    payload["rules"][0]["when"]["right"] = {"type": "lag", "feature": "mom", "periods": -3}
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(payload)
    assert ei.value.has(E.FUTURE_TEMPORAL_REFERENCE)


# ==========================================================================
# T. feature-value audit in StrategyDecision
# ==========================================================================
def test_T_decision_audits_referenced_feature_values():
    plan = compile_strategy(trend_example_spec())
    closes = list(100 + np.arange(120) * 0.5)  # steady uptrend
    ff = compute_features(
        source_from_close(closes),
        [FeatureSpec(kind="ma_spread", params={"fast": 20, "slow": 100}),
         FeatureSpec(kind="volatility", params={"window": 60})],
    )
    ds = ReferenceEvaluator(plan).evaluate_frame(ff)
    last = ds[-1]
    assert set(last.referenced_feature_values) == {"ma_spread_20_100", "volatility_60"}
    assert last.referenced_feature_values["ma_spread_20_100"] == pytest.approx(
        float(ff.features["ma_spread_20_100"].iloc[-1])
    )
    # no execution / fill fields exist on the record
    forbidden = {"fill_price", "execution_price", "reference_price", "slippage", "commission", "order_quantity"}
    assert forbidden.isdisjoint(set(last.model_dump()))


# ==========================================================================
# U. no Order / Fill / RiskDecision object can be created by the DSL evaluator
# ==========================================================================
def test_U_dsl_has_no_execution_imports():
    import ast
    import pathlib

    pkg = pathlib.Path(__import__("alpha_agent.strategy", fromlist=["__file__"]).__file__).parent
    forbidden_import_fragments = ("adapter", "cpp", "quant_core", "execution", "risk_manager",
                                  "portfolio", "backtest", "order", "fill")
    for path in pkg.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            for m in mods:
                low = m.lower()
                assert not any(f in low for f in forbidden_import_fragments), (
                    f"{path.name} imports execution-plane module {m!r}"
                )
            # no call to the C++ fill constructor or similar
            if isinstance(node, ast.Call):
                fn = node.func
                fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                assert fname not in {"make_fill", "make_order_from_signal", "execute"}, (
                    f"{path.name} calls execution-plane function {fname!r}"
                )


def test_U2_evaluator_output_is_only_target_intent():
    plan = compile_strategy(_mom_spec())
    ds = ReferenceEvaluator(plan).evaluate_frame(_mom_frame(UP_DOWN))
    for d in ds:
        dumped = d.model_dump()
        assert isinstance(dumped["target_units"], int)
        assert "fill_price" not in dumped and "order_quantity" not in dumped


# ==========================================================================
# V. deterministic replay
# ==========================================================================
def test_V_deterministic_replay():
    plan = compile_strategy(trend_example_spec())
    closes = list(100 + np.arange(140) * 0.3)
    specs = [FeatureSpec(kind="ma_spread", params={"fast": 20, "slow": 100}),
             FeatureSpec(kind="volatility", params={"window": 60})]
    r1 = [d.model_dump() for d in ReferenceEvaluator(plan).evaluate_frame(compute_features(source_from_close(closes), specs))]
    r2 = [d.model_dump() for d in ReferenceEvaluator(plan).evaluate_frame(compute_features(source_from_close(closes), specs))]
    assert r1 == r2
    # compiling twice yields an identical plan
    assert compile_strategy(trend_example_spec()).model_dump() == plan.model_dump()


# ==========================================================================
# extra: condition tree bounds, duplicate alias / rule id, ref resolution
# ==========================================================================
def test_extra_condition_depth_and_node_bounds():
    deep = _cmp("mom", Comparator.GT, 0.0)
    for _ in range(6):
        deep = NotNode(node=deep)
    spec = _mom_spec().model_copy(
        update={"rules": [Rule(rule_id="deep", when=deep, action=TargetAction(target_units=1))]}
    )
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(spec, limits=CompileLimits(max_condition_depth=4))
    assert ei.value.has(E.CONDITION_TOO_DEEP)


def test_extra_unknown_feature_ref_rejected():
    spec = _mom_spec().model_copy(
        update={"rules": [
            Rule(rule_id="long", when=_cmp("typo_alias", Comparator.GT, 0.0), action=TargetAction(target_units=1))
        ]}
    )
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(spec)
    assert ei.value.has(E.UNKNOWN_FEATURE_REF)


def test_extra_duplicate_rule_id_rejected():
    spec = _mom_spec().model_copy(
        update={"rules": [
            Rule(rule_id="dup", when=_cmp("mom", Comparator.GT, 0.0), action=TargetAction(target_units=1)),
            Rule(rule_id="dup", when=_cmp("mom", Comparator.LT, 0.0), action=TargetAction(target_units=-1)),
        ]}
    )
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(spec)
    assert ei.value.has(E.DUPLICATE_RULE_ID)


def test_extra_duplicate_alias_rejected():
    with pytest.raises(ValidationError):
        StrategySpec(
            strategy_name="d", strategy_id="D1", root_symbol="NQ",
            features=[
                FeatureDeclaration(alias="mom", spec=FeatureSpec(kind="diff", params={"n": 1})),
                FeatureDeclaration(alias="mom", spec=FeatureSpec(kind="diff", params={"n": 2})),
            ],
            rules=[Rule(rule_id="l", when=_cmp("mom", Comparator.GT, 0.0), action=TargetAction(target_units=1))],
            default_action=DefaultAction.FLAT,
        )


def test_extra_canonical_name_collision_rejected():
    spec = StrategySpec(
        strategy_name="c", strategy_id="C1", root_symbol="NQ",
        features=[
            FeatureDeclaration(alias="a", spec=FeatureSpec(kind="ma", params={"window": 10})),
            FeatureDeclaration(alias="b", spec=FeatureSpec(kind="ma", params={"window": 10}, min_observations=5)),
        ],
        rules=[Rule(rule_id="l", when=ComparisonNode(op=Comparator.GT, left=_feat("a"), right=_feat("b")), action=TargetAction(target_units=1))],
        default_action=DefaultAction.FLAT,
    )
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(spec)
    assert ei.value.has(E.CANONICAL_NAME_COLLISION)


def test_extra_comparison_needs_a_feature():
    with pytest.raises(ValidationError):
        ComparisonNode(op=Comparator.GT, left=_c(1.0), right=_c(0.0))


def test_extra_feature_vs_feature_comparison_supported():
    spec = StrategySpec(
        strategy_name="ff", strategy_id="FF1", root_symbol="NQ",
        features=[
            FeatureDeclaration(alias="fast", spec=FeatureSpec(kind="ma", params={"window": 3})),
            FeatureDeclaration(alias="slow", spec=FeatureSpec(kind="ma", params={"window": 8})),
        ],
        rules=[Rule(rule_id="x", when=ComparisonNode(op=Comparator.GT, left=_feat("fast"), right=_feat("slow")), action=TargetAction(target_units=1))],
        default_action=DefaultAction.FLAT,
    )
    plan = compile_strategy(spec)
    ds = ReferenceEvaluator(plan).evaluate_frame(
        compute_features(source_from_close(list(100 + np.arange(20) * 1.0)),
                         [FeatureSpec(kind="ma", params={"window": 3}), FeatureSpec(kind="ma", params={"window": 8})])
    )
    assert ds[-1].target_units == 1  # fast MA above slow MA in a steady uptrend


def test_extra_root_mismatch_rejected():
    plan = compile_strategy(_mom_spec())  # root NQ
    ff = _mom_frame(UP_DOWN)
    ff.identifiers["root_symbol"] = "ES"
    with pytest.raises(StrategyEvaluationError):
        ReferenceEvaluator(plan).evaluate_frame(ff)


def test_extra_missing_required_column_rejected():
    plan = compile_strategy(_mom_spec())
    ff = compute_features(source_from_close(UP_DOWN), [FeatureSpec(kind="ma", params={"window": 3})])
    with pytest.raises(StrategyEvaluationError):
        ReferenceEvaluator(plan).evaluate_frame(ff)
