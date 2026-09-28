"""The closed typed IR produced by the compiler: :class:`CompiledStrategyPlan`.

No source code is generated. The plan holds fully-resolved, canonical feature
names, a resolved condition tree, ordered rules, warm-up and safety
requirements, and the deterministic strategy fingerprint. It is the only thing
the reference evaluator consumes besides a :class:`FeatureFrame`.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from alpha_agent.features.spec import FeatureMetadata, FeatureSpec
from alpha_agent.strategy.enums import (
    BooleanOp,
    Comparator,
    DefaultAction,
    MissingRulePolicy,
    NodeType,
    OperandType,
)


class _Frozen(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}


class CompiledOperand(_Frozen):
    kind: OperandType
    # FEATURE / LAG: canonical feature name (resolved from the alias).
    feature: str | None = None
    # LAG only, >= 1.
    periods: int | None = None
    # CONST only, finite.
    value: float | None = None


class CompiledCondition(_Frozen):
    node: NodeType
    # COMPARISON
    op: Comparator | None = None
    left: CompiledOperand | None = None
    right: CompiledOperand | None = None
    # BOOLEAN
    bool_op: BooleanOp | None = None
    # BOOLEAN children, or the single NOT child.
    children: tuple[CompiledCondition, ...] = ()


CompiledCondition.model_rebuild()


class CompiledRule(_Frozen):
    rule_id: str
    condition: CompiledCondition
    target_units: int
    rationale_label: str = ""


class FeatureBinding(_Frozen):
    alias: str
    canonical_name: str
    spec: FeatureSpec
    metadata: FeatureMetadata


class SafetyRequirements(_Frozen):
    """What every FeatureFrame fed to the evaluator must satisfy (section 5)."""

    require_signal_safe: bool = True
    require_point_in_time_safe: bool = True
    forbid_execution_price_use: bool = True
    note: str = (
        "features drive Signals only; allowed = causal RawContract-derived, causal "
        "RawContinuous-derived, or point-in-time BackAdjusted-derived; rejected = "
        "retrospective BackAdjusted / retrospective future-roll / any look-ahead-"
        "unsafe feature. A FeatureRef is never an execution price."
    )


class CompiledStrategyPlan(_Frozen):
    dsl_version: str
    schema_version: str
    strategy_name: str
    strategy_id: str
    root_symbol: str
    fingerprint: str

    feature_bindings: tuple[FeatureBinding, ...]
    declared_features: tuple[str, ...]
    required_features: tuple[str, ...]

    rules: tuple[CompiledRule, ...]
    default_action: DefaultAction
    on_missing: MissingRulePolicy

    warmup_bars: int = Field(ge=0)
    safety: SafetyRequirements = SafetyRequirements()
    canonical_payload: dict

    def binding(self, canonical_name: str) -> FeatureBinding:
        for b in self.feature_bindings:
            if b.canonical_name == canonical_name:
                return b
        raise KeyError(canonical_name)  # pragma: no cover
