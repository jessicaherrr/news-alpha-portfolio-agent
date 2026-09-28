"""Phase 10.1 -- schema authority + strict input typing.

1. exactly one authoritative StrategySpec (the closed DSL one)
2. the legacy free-text schema cannot enter any compiler / runtime path
3. strict scalar types: no silent Pydantic coercion of target_units, lag
   periods, or comparison constants
4. YAML round-trip is safe-load only (no arbitrary object construction)
"""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest
from alpha_agent.strategy import (
    StrategyCompileError,
    compile_strategy,
    strategy_from_json,
    strategy_from_yaml,
    strategy_to_json,
    strategy_to_yaml,
)
from alpha_agent.strategy import StrategySpec as AuthoritativeStrategySpec
from alpha_agent.strategy import errors as E
from alpha_agent.strategy.enums import Comparator
from alpha_agent.strategy.examples import trend_example_spec
from alpha_agent.strategy.spec import ConstOperand, LagOperand, TargetAction
from pydantic import ValidationError

# The legacy Phase-01 free-text StrategySpec, as it used to be constructed.
LEGACY_FREETEXT_PAYLOAD = {
    "strategy_id": "MOM-001",
    "family": "time_series_momentum",
    "symbols": ["NQ"],
    "parameters": {"lookback": 20, "threshold": 0.01},
    "entry_rule": "next bar open after signal",
    "exit_rule": "when target position changes",
    "cost_model": {"commission_per_side_usd": 2.5, "slippage_ticks": 1.0},
}


# ==========================================================================
# 1. ONE AUTHORITATIVE STRATEGYSPEC
# ==========================================================================
def test_canonical_import_path_is_the_dsl_spec():
    from alpha_agent.strategy import StrategySpec  # the documented Agent import
    from alpha_agent.strategy.spec import StrategySpec as SpecModule

    assert StrategySpec is SpecModule is AuthoritativeStrategySpec
    # it is the closed DSL schema, not a free-text one
    fields = set(StrategySpec.model_fields)
    assert {"schema_version", "features", "rules", "default_action"} <= fields
    assert not ({"entry_rule", "exit_rule", "cost_model"} & fields)


def test_legacy_schema_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("alpha_agent.schemas.strategy")
    schemas = importlib.import_module("alpha_agent.schemas")
    assert not hasattr(schemas, "StrategySpec")


def test_no_production_module_imports_a_legacy_strategy_schema():
    pkg = pathlib.Path(importlib.import_module("alpha_agent").__file__).parent
    offenders: list[str] = []
    for path in pkg.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            mod = ""
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            elif isinstance(node, ast.Import):
                mod = ",".join(a.name for a in node.names)
            if "schemas.strategy" in mod:
                offenders.append(f"{path.name}: {mod}")
    assert not offenders, offenders


# ==========================================================================
# 2. LEGACY FREE-TEXT SCHEMA CANNOT ENTER THE COMPILER PATH
# ==========================================================================
def test_legacy_freetext_payload_rejected_by_compiler():
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(LEGACY_FREETEXT_PAYLOAD)
    # rejected as illegal DSL fields (free-text rules + nested slippage field)
    assert ei.value.has(E.ILLEGAL_DSL_FIELD)


def test_legacy_freetext_payload_rejected_by_schema():
    with pytest.raises(ValidationError):
        AuthoritativeStrategySpec.model_validate(LEGACY_FREETEXT_PAYLOAD)


# ==========================================================================
# 3. STRICT SCALAR TYPES -- no silent coercion
# ==========================================================================
def test_target_units_strict_int():
    for good in (1, 0, -2):
        assert TargetAction(target_units=good).target_units == good
    for bad in (1.0, 1.5, "1", True, False, None, "abc"):
        with pytest.raises(ValidationError):
            TargetAction(target_units=bad)


def test_lag_periods_strict_positive_int():
    for good in (1, 5):
        assert LagOperand(feature="x", periods=good).periods == good
    for bad in (0, -1, 1.0, "5", True, False, 2.5, None):
        with pytest.raises(ValidationError):
            LagOperand(feature="x", periods=bad)


def test_bool_is_never_an_int():
    with pytest.raises(ValidationError):
        TargetAction(target_units=True)
    with pytest.raises(ValidationError):
        LagOperand(feature="x", periods=True)
    with pytest.raises(ValidationError):
        ConstOperand(value=True)


def test_comparison_constant_strict_numeric():
    for good in (0, 1, -3, 0.5, -2.25, 1e6):
        assert ConstOperand(value=good).value == float(good)
    for bad in ("0.5", "1", "nan", "inf", True, False, None, [1]):
        with pytest.raises(ValidationError):
            ConstOperand(value=bad)


def test_nan_inf_constants_rejected_all_forms():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            ConstOperand(value=bad)


def test_string_numeric_constant_rejected_through_compiler():
    payload = trend_example_spec().model_dump(mode="json")
    payload["rules"][1]["when"]["right"]["value"] = "0.5"
    with pytest.raises(StrategyCompileError) as ei:
        compile_strategy(payload)
    assert ei.value.has(E.INVALID_CONSTANT)


# ==========================================================================
# 4. SERIALIZATION SAFETY
# ==========================================================================
_YAML_LOAD_ERRORS = (__import__("yaml").YAMLError, ValidationError, ValueError)


def test_yaml_rejects_arbitrary_python_object_tag():
    hostile = (
        "schema_version: strategy-dsl/1\n"
        "danger: !!python/object/apply:os.system ['echo pwned']\n"
    )
    with pytest.raises(_YAML_LOAD_ERRORS):
        strategy_from_yaml(hostile)


def test_yaml_rejects_python_name_tag():
    with pytest.raises(_YAML_LOAD_ERRORS):
        strategy_from_yaml("value: !!python/name:os.system\n")


def test_serialization_roundtrip_preserves_fingerprint_and_semantics():
    spec = trend_example_spec()
    for loads, dumps in ((strategy_from_json, strategy_to_json), (strategy_from_yaml, strategy_to_yaml)):
        rt = loads(dumps(spec))
        assert rt.model_dump() == spec.model_dump()
        assert rt.fingerprint == spec.fingerprint
        assert compile_strategy(rt).fingerprint == compile_strategy(spec).fingerprint
        assert dumps(rt) == dumps(spec)  # byte-stable


def test_yaml_load_uses_safe_loader_only():
    import inspect

    from alpha_agent.strategy import spec as spec_mod

    src = inspect.getsource(spec_mod.strategy_from_yaml)
    assert "safe_load" in src and "yaml.load(" not in src
    assert "safe_dump" in inspect.getsource(spec_mod.strategy_to_yaml)


def test_comparator_set_unchanged():
    assert {c.value for c in Comparator} == {"gt", "gte", "lt", "lte"}
