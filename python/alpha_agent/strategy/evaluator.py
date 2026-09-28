"""Deterministic Python reference evaluator for a :class:`CompiledStrategyPlan`.

This exists to validate DSL semantics, not to run production trading. It
consumes only:

* the compiled plan,
* a point-in-time :class:`FeatureFrame` (timestamps, identifiers, feature
  values + presence masks),
* one scalar of strategy-local state: the last emitted target
  (for ``KEEP_PREVIOUS_TARGET``).

It never sees -- and has no code path to -- a ``Fill``, ``Order``,
``RiskDecision``, execution price, portfolio cash/equity, or a future row. Its
output is target-position intent (:class:`StrategyDecision`). Official execution
and PnL stay in the C++ Quant Core.

No look-ahead: value lookups only ever index the current row ``i`` or an earlier
row ``i - periods`` (``periods >= 1``). Combined with the Phase 09 prefix
invariance of every feature value, the decision at ``T`` is identical whether
the frame ends at ``T`` or continues past it (section 19).
"""
from __future__ import annotations

import math

from alpha_agent.features.frame import FeatureFrame
from alpha_agent.features.safety import FeatureSafetyError
from alpha_agent.strategy.compiler import compile_strategy
from alpha_agent.strategy.decision import StrategyDecision
from alpha_agent.strategy.enums import (
    BooleanOp,
    DefaultAction,
    MissingRulePolicy,
    NodeType,
    OperandType,
)
from alpha_agent.strategy.errors import StrategyEvaluationError
from alpha_agent.strategy.plan import CompiledCondition, CompiledStrategyPlan
from alpha_agent.strategy.spec import StrategySpec

# Three-valued logic tags.
_TRUE = "true"
_FALSE = "false"
_UNKNOWN = "unknown"


def _not3(v: str) -> str:
    if v is _UNKNOWN:
        return _UNKNOWN
    return _FALSE if v is _TRUE else _TRUE


class ReferenceEvaluator:
    def __init__(self, plan: CompiledStrategyPlan) -> None:
        self.plan = plan

    # -- public ---------------------------------------------------------
    def evaluate_frame(self, frame: FeatureFrame) -> list[StrategyDecision]:
        self._gate(frame)

        ts = frame.identifiers["ts_event_ns"].to_numpy()
        n = len(ts)
        feature_data = {
            name: (
                frame.features[name].to_numpy(dtype="float64"),
                frame.mask[name].to_numpy(dtype="bool"),
            )
            for name in self.plan.required_features
        }

        decisions: list[StrategyDecision] = []
        last_target = 0
        for i in range(n):
            decision, last_target = self._decide_row(
                int(ts[i]), i, feature_data, last_target
            )
            decisions.append(decision)
        return decisions

    __call__ = evaluate_frame

    # -- gate ----------------------------------------------------------------
    def _gate(self, frame: FeatureFrame) -> None:
        try:
            frame.assert_signal_safe()
        except FeatureSafetyError as exc:
            raise StrategyEvaluationError(
                f"FeatureFrame is not signal-safe, cannot drive strategy "
                f"{self.plan.strategy_id!r}: {exc}"
            ) from exc

        ids = frame.identifiers
        if "root_symbol" in ids.columns:
            roots = {str(x) for x in ids["root_symbol"].dropna().unique()}
            bad = roots - {self.plan.root_symbol}
            if bad:
                raise StrategyEvaluationError(
                    f"FeatureFrame root(s) {sorted(bad)} do not match strategy "
                    f"root_symbol {self.plan.root_symbol!r}"
                )

        missing = [f for f in self.plan.required_features if f not in frame.features.columns]
        if missing:
            raise StrategyEvaluationError(
                f"FeatureFrame is missing required feature column(s): {missing}"
            )

    # -- per-row -----------------------------------------------------------
    def _decide_row(self, ts_ns: int, i: int, feature_data: dict, last_target: int):
        referenced: dict[str, float | None] = {}
        missing: set[str] = set()

        def resolve(operand):
            if operand.kind is OperandType.CONST:
                return operand.value
            name = operand.feature
            vals, present = feature_data[name]
            if operand.kind is OperandType.LAG:
                j = i - operand.periods
                label = f"{name}@lag{operand.periods}"
                ok = j >= 0 and bool(present[j]) and not math.isnan(vals[j])
                v = float(vals[j]) if ok else None
            else:
                label = name
                ok = bool(present[i]) and not math.isnan(vals[i])
                v = float(vals[i]) if ok else None
            referenced[label] = v
            if v is None:
                missing.add(label)
            return v

        matched_rule_id: str | None = None
        rationale_label: str | None = None
        target = last_target
        default_applied = True
        not_evaluable: list[str] = []

        for rule in self.plan.rules:
            result = _eval_condition(rule.condition, resolve)
            if result is _UNKNOWN:
                not_evaluable.append(rule.rule_id)
                if self.plan.on_missing is MissingRulePolicy.HOLD:
                    break
                continue
            if result is _TRUE:
                matched_rule_id = rule.rule_id
                rationale_label = rule.rationale_label or None
                target = rule.target_units
                default_applied = False
                break

        if default_applied:
            if self.plan.default_action is DefaultAction.FLAT:
                target = 0
            else:  # KEEP_PREVIOUS_TARGET
                target = last_target

        decision = StrategyDecision(
            ts_event_ns=ts_ns,
            strategy_fingerprint=self.plan.fingerprint,
            root_symbol=self.plan.root_symbol,
            target_units=int(target),
            matched_rule_id=matched_rule_id,
            default_applied=default_applied,
            referenced_feature_values=dict(sorted(referenced.items())),
            missing_features=tuple(sorted(missing)),
            not_evaluable_rule_ids=tuple(not_evaluable),
            rationale_label=rationale_label,
        )
        return decision, int(target)


def _eval_condition(cond: CompiledCondition, resolve) -> str:
    if cond.node is NodeType.COMPARISON:
        lv = resolve(cond.left)
        rv = resolve(cond.right)
        if lv is None or rv is None:
            return _UNKNOWN
        return _TRUE if cond.op.apply(lv, rv) else _FALSE

    if cond.node is NodeType.NOT:
        return _not3(_eval_condition(cond.children[0], resolve))

    # BOOLEAN -- evaluate every child (no short-circuit) for a complete audit.
    results = [_eval_condition(c, resolve) for c in cond.children]
    if cond.bool_op is BooleanOp.ALL:
        if any(r is _FALSE for r in results):
            return _FALSE
        if any(r is _UNKNOWN for r in results):
            return _UNKNOWN
        return _TRUE
    # ANY
    if any(r is _TRUE for r in results):
        return _TRUE
    if any(r is _UNKNOWN for r in results):
        return _UNKNOWN
    return _FALSE


def evaluate_strategy(
    strategy: StrategySpec | CompiledStrategyPlan | dict | str,
    frame: FeatureFrame,
) -> list[StrategyDecision]:
    plan = strategy if isinstance(strategy, CompiledStrategyPlan) else compile_strategy(strategy)
    return ReferenceEvaluator(plan).evaluate_frame(frame)
