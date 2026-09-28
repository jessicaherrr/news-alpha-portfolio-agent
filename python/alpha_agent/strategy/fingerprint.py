"""Canonicalization and the deterministic strategy fingerprint (section 15).

The same *semantic* StrategySpec always produces the same canonical payload and
the same fingerprint, regardless of:

* ``strategy_name`` / ``strategy_id``
* ``rule_id`` labels and per-rule / top-level ``rationale`` free text
* ``metadata`` labels
* feature *alias* spelling (features are keyed by their canonical
  :class:`FeatureSpec`, not the local alias)
* the order of children inside an ``ALL`` / ``ANY`` node (commutative)
* dict / mapping iteration order
* creation time, machine path, or any random id

A semantic change -- a different comparator, constant, feature parameter,
target-units value, ``NOT`` wrapper, rule order, default action or missing-rule
policy -- always changes the fingerprint (section 15, tests O/P/Q).
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from alpha_agent.strategy.enums import NodeType
from alpha_agent.strategy.spec import (
    BooleanNode,
    ComparisonNode,
    ConstOperand,
    FeatureOperand,
    LagOperand,
    NotNode,
)

if TYPE_CHECKING:
    from alpha_agent.strategy.spec import StrategySpec

STRATEGY_DSL_VERSION = "1.0.0"
_FINGERPRINT_PREFIX = "stratdsl1"


def _feature_key(spec) -> str:
    """A canonical, registry-independent identity for one feature."""
    return spec.canonical_json()


def _operand_payload(operand: Any, alias_to_key: dict[str, str]) -> dict:
    if isinstance(operand, FeatureOperand):
        return {"k": "feature", "f": alias_to_key[operand.feature]}
    if isinstance(operand, LagOperand):
        return {"k": "lag", "f": alias_to_key[operand.feature], "n": operand.periods}
    if isinstance(operand, ConstOperand):
        return {"k": "const", "v": operand.value}
    raise TypeError(f"unknown operand type {type(operand).__name__}")  # pragma: no cover


def _condition_payload(node: Any, alias_to_key: dict[str, str]) -> dict:
    if isinstance(node, ComparisonNode):
        return {
            "t": NodeType.COMPARISON.value,
            "op": node.op.value,
            "l": _operand_payload(node.left, alias_to_key),
            "r": _operand_payload(node.right, alias_to_key),
        }
    if isinstance(node, NotNode):
        return {"t": NodeType.NOT.value, "n": _condition_payload(node.node, alias_to_key)}
    if isinstance(node, BooleanNode):
        children = [_condition_payload(c, alias_to_key) for c in node.nodes]
        # ALL / ANY are commutative: sort children by their canonical text.
        children.sort(key=lambda d: json.dumps(d, sort_keys=True, separators=(",", ":")))
        return {"t": node.op.value, "ns": children}
    raise TypeError(f"unknown condition node {type(node).__name__}")  # pragma: no cover


def canonical_strategy_payload(spec: StrategySpec) -> dict:
    """The semantic-only dict that the fingerprint hashes."""
    alias_to_key = {decl.alias: _feature_key(decl.spec) for decl in spec.features}
    return {
        "schema_version": spec.schema_version,
        "root_symbol": spec.root_symbol,
        "features": sorted(set(alias_to_key.values())),
        "rules": [
            {
                "when": _condition_payload(rule.when, alias_to_key),
                "target_units": rule.action.target_units,
            }
            for rule in spec.rules
        ],
        "default_action": spec.default_action.value,
        "on_missing": spec.on_missing.value,
    }


def canonical_strategy_json(spec: StrategySpec) -> str:
    return json.dumps(
        canonical_strategy_payload(spec), sort_keys=True, separators=(",", ":")
    )


def strategy_fingerprint(spec: StrategySpec) -> str:
    digest = hashlib.sha256(canonical_strategy_json(spec).encode("utf-8")).hexdigest()
    return f"{_FINGERPRINT_PREFIX}:{digest}"
