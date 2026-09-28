"""The auditable typed decision record emitted per bar (section 18).

A :class:`StrategyDecision` carries target-position intent and the exact feature
values that produced it. It has **no** execution or fill price, no order
quantity, no risk field -- those are the C++ Quant Core's outputs, not the
strategy's.
"""
from __future__ import annotations

from pydantic import BaseModel


class StrategyDecision(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    ts_event_ns: int
    strategy_fingerprint: str
    root_symbol: str

    target_units: int
    matched_rule_id: str | None
    default_applied: bool

    # Every feature value actually read while evaluating this bar's rules.
    # Key = canonical feature name, or "<name>@lag<N>" for a lagged reference.
    # Value = the float, or None when it was missing / NaN.
    referenced_feature_values: dict[str, float | None]
    missing_features: tuple[str, ...]
    # Rules whose condition was NOT EVALUABLE for this bar (missing feature).
    not_evaluable_rule_ids: tuple[str, ...]

    # Free-text label copied from the matched rule; informational only.
    rationale_label: str | None = None
