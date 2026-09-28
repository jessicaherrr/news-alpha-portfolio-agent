"""The versioned, frozen, typed :class:`StrategySpec` and its condition tree.

Executable semantics live entirely in typed, structured fields. Free-text
(``rationale``) and non-semantic ``metadata`` never alter behaviour and never
enter the fingerprint. Every model forbids unknown fields, so an illegal DSL
field -- including any execution-plane field -- is rejected at parse time
(sections 2, 3, 6, 22).
"""
from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from alpha_agent.features.spec import FeatureSpec
from alpha_agent.strategy.enums import (
    BooleanOp,
    Comparator,
    DefaultAction,
    MissingRulePolicy,
)

# The execution plane is inaccessible from the DSL (section 6). These names may
# never appear anywhere in a StrategySpec payload; `extra="forbid"` already
# rejects them, this list makes the intent explicit and the error legible.
FORBIDDEN_EXECUTION_FIELDS = frozenset(
    {
        "fill_price",
        "execution_price",
        "reference_price",
        "raw_symbol",
        "instrument_id",
        "contract_month",
        "contract",
        "slippage",
        "slippage_ticks",
        "spread",
        "spread_ticks",
        "commission",
        "commission_usd",
        "latency",
        "latency_bars",
        "margin",
        "margin_override",
        "risk_override",
        "risk_limit",
        "risk_decision",
        "fill",
        "order",
        "order_quantity",
        "quantity",
        "cash",
        "equity",
        "drawdown",
        "pnl",
    }
)

# A hard cap in the schema itself; the compiler enforces a (configurable,
# usually tighter) limit on top of this.
_MAX_ABS_TARGET_UNITS_SCHEMA = 1000


def _strict_int(v: Any, *, field: str) -> int:
    """A *true* Python ``int`` and nothing else.

    Pydantic's lax mode would coerce ``1.0`` / ``"1"`` / ``True`` into an int.
    The DSL must not allow that: a target unit or a lag offset is an exact
    integer or it is an error (Phase 10.1 sections 3, 4).
    """
    if isinstance(v, bool):
        raise ValueError(f"{field} must be a plain integer, not a bool")  # noqa: TRY004
    if isinstance(v, int):
        return v
    raise ValueError(
        f"{field} must be a plain integer with no coercion (got {type(v).__name__} {v!r}); "
        "floats such as 1.0, strings such as '1', and booleans are rejected"
    )


def _strict_number(v: Any, *, field: str) -> float:
    """A finite ``int`` or ``float`` numeric literal.

    Rejects ``bool``, strings such as ``"0.5"`` (no silent string->float
    coercion), and non-finite ``NaN`` / ``+inf`` / ``-inf`` (Phase 10.1
    section 5).
    """
    if isinstance(v, bool):
        raise ValueError(f"{field} must be a numeric constant, not a bool")  # noqa: TRY004
    if not isinstance(v, (int, float)):
        raise ValueError(  # noqa: TRY004
            f"{field} must be a numeric int/float literal with no coercion "
            f"(got {type(v).__name__} {v!r}); strings such as '0.5' are rejected"
        )
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{field} must be a finite number (no NaN / inf)")
    return f


def _deep_reject_forbidden(obj: Any, path: str = "") -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(key, str) and key.lower() in FORBIDDEN_EXECUTION_FIELDS:
                where = f"{path}.{key}" if path else key
                raise ValueError(
                    f"execution-plane field {where!r} is not permitted in a StrategySpec: "
                    "the DSL expresses target-position intent only; execution prices, "
                    "fills, slippage, commission, latency, margin and risk decisions "
                    "belong to the deterministic C++ Quant Core"
                )
            _deep_reject_forbidden(val, f"{path}.{key}" if path else str(key))
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            _deep_reject_forbidden(val, f"{path}[{i}]")


# --------------------------------------------------------------------------
# Operands
# --------------------------------------------------------------------------
class _Strict(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}


class FeatureOperand(_Strict):
    """The current point-in-time value of a declared feature."""

    type: Literal["feature"] = "feature"
    feature: str  # a declared feature alias


class LagOperand(_Strict):
    """The value of a declared feature ``periods`` bars ago. Strictly causal:
    ``periods >= 1`` (``periods == 0`` is ambiguous and forbidden; a negative
    lag / ``lead`` / ``future`` primitive does not exist -- section 9)."""

    type: Literal["lag"] = "lag"
    feature: str
    periods: int = Field(ge=1, le=100_000)

    @field_validator("periods", mode="before")
    @classmethod
    def _strict_periods(cls, v: Any) -> int:
        return _strict_int(v, field="lag periods")


class ConstOperand(_Strict):
    """A finite numeric literal (``int`` / ``float``). ``bool``, strings, and
    ``NaN`` / ``+-inf`` are rejected with no coercion (sections 5, 14)."""

    type: Literal["const"] = "const"
    value: float

    @field_validator("value", mode="before")
    @classmethod
    def _strict_value(cls, v: Any) -> float:
        return _strict_number(v, field="comparison constant")


Operand = Annotated[
    FeatureOperand | LagOperand | ConstOperand,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------
# Condition tree (finite-depth, recursively typed -- section 8)
# --------------------------------------------------------------------------
class ComparisonNode(_Strict):
    type: Literal["comparison"] = "comparison"
    op: Comparator
    left: Operand
    right: Operand

    @model_validator(mode="after")
    def _references_a_feature(self) -> ComparisonNode:
        kinds = {self.left.type, self.right.type}
        if kinds == {"const"}:
            raise ValueError(
                "a comparison must reference at least one feature (constant vs "
                "constant is not a trading condition)"
            )
        return self


class NotNode(_Strict):
    type: Literal["not"] = "not"
    node: ConditionNode


class BooleanNode(_Strict):
    type: Literal["boolean"] = "boolean"
    op: BooleanOp
    nodes: list[ConditionNode] = Field(min_length=1)


ConditionNode = Annotated[
    ComparisonNode | BooleanNode | NotNode,
    Field(discriminator="type"),
]

NotNode.model_rebuild()
BooleanNode.model_rebuild()


# --------------------------------------------------------------------------
# Rules / actions
# --------------------------------------------------------------------------
class TargetAction(_Strict):
    """Specifies a TARGET POSITION only, never an order quantity (section 11).

    The C++ Quant Core remains authoritative for
    ``order_delta = target_position - current_position``.
    """

    target_units: int = Field(
        ge=-_MAX_ABS_TARGET_UNITS_SCHEMA, le=_MAX_ABS_TARGET_UNITS_SCHEMA
    )

    @field_validator("target_units", mode="before")
    @classmethod
    def _strict_target_units(cls, v: Any) -> int:
        # A true int only: 1.0 / "1" / True are rejected, not coerced.
        return _strict_int(v, field="target_units")


class Rule(_Strict):
    rule_id: str = Field(min_length=1, max_length=128)
    when: ConditionNode
    action: TargetAction
    # Free-text; never affects behaviour or the fingerprint (section 3, 15).
    rationale: str = ""

    @field_validator("rule_id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("rule_id must not be blank")
        return v


class FeatureDeclaration(_Strict):
    """Binds a local ``alias`` (what conditions reference) to a Phase 09
    :class:`FeatureSpec`. The alias namespace is closed to declared names -- a
    condition can never reach an arbitrary DataFrame column (section 4)."""

    alias: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=128)
    spec: FeatureSpec


# --------------------------------------------------------------------------
# StrategySpec
# --------------------------------------------------------------------------
class StrategySpec(BaseModel):
    """A closed, versioned, frozen strategy definition.

    Scoped to exactly one ``root_symbol`` (section 20). Cross-market features
    may be inputs, but one spec never emits intent for multiple roots.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: Literal["strategy-dsl/1"] = "strategy-dsl/1"
    strategy_name: str = Field(min_length=1, max_length=200)
    strategy_id: str = Field(min_length=1, max_length=200)
    root_symbol: str = Field(pattern=r"^[A-Z0-9]{1,12}$")
    features: list[FeatureDeclaration] = Field(min_length=1)
    rules: list[Rule] = Field(min_length=1)
    default_action: DefaultAction
    on_missing: MissingRulePolicy = MissingRulePolicy.SKIP_RULE
    # Free-text rationale / non-semantic labels: never affect behaviour or hash.
    rationale: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _no_execution_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _deep_reject_forbidden(data)
        return data

    @model_validator(mode="after")
    def _unique_aliases(self) -> StrategySpec:
        seen: set[str] = set()
        for decl in self.features:
            if decl.alias in seen:
                raise ValueError(f"duplicate feature alias {decl.alias!r}")
            seen.add(decl.alias)
        return self

    def canonical_json(self) -> str:
        """Deterministic semantic representation (see
        :mod:`alpha_agent.strategy.fingerprint`)."""
        from alpha_agent.strategy.fingerprint import canonical_strategy_payload

        return json.dumps(
            canonical_strategy_payload(self), sort_keys=True, separators=(",", ":")
        )

    @property
    def fingerprint(self) -> str:
        from alpha_agent.strategy.fingerprint import strategy_fingerprint

        return strategy_fingerprint(self)


# --------------------------------------------------------------------------
# Serialization (section 22): deterministic, round-trip stable, no code tags
# --------------------------------------------------------------------------
def strategy_to_json(spec: StrategySpec, *, indent: int | None = 2) -> str:
    return json.dumps(spec.model_dump(mode="json"), sort_keys=True, indent=indent)


def strategy_from_json(text: str | bytes) -> StrategySpec:
    return StrategySpec.model_validate(json.loads(text))


def strategy_to_yaml(spec: StrategySpec) -> str:
    return yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=True, default_flow_style=False)


def strategy_from_yaml(text: str) -> StrategySpec:
    return StrategySpec.model_validate(yaml.safe_load(text))
