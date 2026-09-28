"""Human-readable rendering of a compiled `StrategySpec` (Research Golden
Path, section 7): "Render a human-readable StrategySpec summary... Do not
expose noisy implementation JSON by default."

Pure string formatting over the exact typed DSL shape frozen in
`alpha_agent.strategy.spec` (`Rule` / `ConditionNode` / `Operand` /
`TargetAction`) -- this module never changes, re-derives, or validates a
StrategySpec; it only describes the one it is given, from its already-
compiled JSON (`CompiledStrategyProposal.strategy_spec`). The raw JSON stays
available verbatim under a "View typed spec" expander alongside this.
"""
from __future__ import annotations

from typing import Any

_COMPARATOR_TEXT = {"gt": "above", "gte": "at or above", "lt": "below", "lte": "at or below"}
_BOOLEAN_TEXT = {"all": "AND", "any": "OR"}
_DEFAULT_ACTION_TEXT = {
    "flat": "go flat (0 contracts)",
    "keep_previous_target": "keep the previous target position unchanged",
}
_MISSING_POLICY_TEXT = {
    "skip_rule": "skip this rule for the bar and evaluate the next rule",
    "hold": "stop rule evaluation for the bar and apply the default action",
}


def _describe_operand(op: dict[str, Any]) -> str:
    kind = op.get("type")
    if kind == "feature":
        return op["feature"]
    if kind == "lag":
        return f"{op['feature']} lagged {op['periods']} bar(s)"
    if kind == "const":
        return f"{op['value']:g}"
    return str(op)


def describe_condition(node: dict[str, Any]) -> str:
    """Recursively renders a `ConditionNode` (comparison / boolean / not) as
    one plain-English clause, e.g. "trend_strength above 0 AND
    realized_vol below 0.05"."""
    kind = node.get("type")
    if kind == "comparison":
        left = _describe_operand(node["left"])
        right = _describe_operand(node["right"])
        comp = _COMPARATOR_TEXT.get(node["op"], node["op"])
        return f"{left} {comp} {right}"
    if kind == "not":
        return f"NOT ({describe_condition(node['node'])})"
    if kind == "boolean":
        joiner = f" {_BOOLEAN_TEXT.get(node['op'], node['op'])} "
        clauses = [describe_condition(n) for n in node["nodes"]]
        return "(" + joiner.join(clauses) + ")"
    return str(node)


def describe_rule(rule: dict[str, Any]) -> str:
    condition = describe_condition(rule["when"])
    target = rule["action"]["target_units"]
    position = "flat" if target == 0 else f"{target:+d} contract(s)"
    return f"IF {condition} THEN target position = {position}"


def summarize_strategy_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """A flat, presentation-only dict: root/features/rules-in-plain-English/
    default behaviour/missing-data policy/rationale. Every value is read
    straight from the given compiled `StrategySpec` JSON -- nothing here
    infers behaviour beyond what the typed DSL fields already say."""
    features = [
        {"alias": f["alias"], "kind": f.get("spec", {}).get("kind", "?")}
        for f in spec.get("features", [])
    ]
    rules = [
        {"rule_id": r["rule_id"], "text": describe_rule(r), "rationale": r.get("rationale") or ""}
        for r in spec.get("rules", [])
    ]
    return {
        "strategy_name": spec.get("strategy_name"),
        "root_symbol": spec.get("root_symbol"),
        "features": features,
        "rules": rules,
        "default_action": _DEFAULT_ACTION_TEXT.get(spec.get("default_action"), spec.get("default_action")),
        "on_missing": _MISSING_POLICY_TEXT.get(spec.get("on_missing"), spec.get("on_missing")),
        "rationale": spec.get("rationale") or "",
    }
