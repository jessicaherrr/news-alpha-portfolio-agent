"""Typed FACTOR EXPRESSIONS over the feature registry -- one closed operator
vocabulary for turning a measurement into a factor.

    FactorExpression --render()-------> "ts_return(close, 20)"          what a user reads
                     --feature_spec()-> FeatureSpec(kind="return", n=20)  what executes
                     --evaluate()-----> factor values                     what diagnostics test

All three read the SAME typed object, so the formula on screen is, by
construction, the transformation that runs: there is no second, hidden
formula. `canonical_json()` is the identity-bearing form.

No code, no free text. An operator is a name in `OPERATORS`, bound to exactly
one REGISTERED `FeatureRegistry` kind per input field (checked against the
live registry at import), with typed integer parameters validated by that
kind's own `ParamRule`s. This is not a second feature system: every operator
IS a registered feature, computed by the unchanged `compute_features` engine
on the gap-free daily index (`features.daily`). A model that ever proposes an
expression must produce this object -- anything else fails validation.

Scope. Every operator is a TIME-SERIES operator on one instrument's series.
Cross-sectional normalization (`rank(...)` across instruments) is part of
the vocabulary so that its refusal is typed: one instrument has no
cross-section, so its "rank" would be a constant -- a formula that looks
like a factor and carries no information. `evaluate` refuses it with
`CrossSectionalOperatorError` rather than silently computing something else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel, StrictInt, model_validator

from alpha_agent.features.daily import compute_contiguous_daily_features
from alpha_agent.features.registry import FEATURE_ENGINE_VERSION, REGISTRY, FeatureRegistry
from alpha_agent.features.spec import FeatureMetadata, FeatureSpec
from alpha_agent.schemas.market_data import PriceDomain

__all__ = [
    "EXPRESSION_VOCABULARY_VERSION",
    "OPERATORS",
    "CrossSectionalOperatorError",
    "EvaluatedFactor",
    "FactorExpression",
    "FactorOperator",
    "Normalization",
    "evaluate_expression",
]

EXPRESSION_VOCABULARY_VERSION = "factor-expression/1"


class CrossSectionalOperatorError(ValueError):
    """A cross-sectional operator was applied where there is no cross-section."""


class Normalization(str, Enum):
    NONE = "NONE"
    #: Rank across instruments on each date. Needs a universe; refused for a
    #: single-instrument series.
    CROSS_SECTIONAL_RANK = "CROSS_SECTIONAL_RANK"


@dataclass(frozen=True)
class FactorOperator:
    name: str
    #: Input field -> the registered feature kind that computes this operator on it.
    kinds: tuple[tuple[str, str], ...]
    #: Expression argument names, in order, each equal to a registered param name.
    params: tuple[str, ...]
    description: str

    def kind_for(self, field: str) -> str:
        return dict(self.kinds)[field]

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(f for f, _ in self.kinds)


OPERATORS: dict[str, FactorOperator] = {
    op.name: op
    for op in (
        FactorOperator("ts_return", (("close", "return"),), ("n",),
                       "simple return over the last n bars"),
        FactorOperator("ts_log_return", (("close", "log_return"),), ("n",),
                       "log return over the last n bars"),
        FactorOperator("ts_delta", (("close", "diff"),), ("n",),
                       "arithmetic change over the last n bars"),
        FactorOperator("ts_zscore", (("close", "zscore"), ("volume", "volume_zscore")), ("window",),
                       "(x - trailing mean) / trailing std over the window"),
        FactorOperator("ts_trend_strength", (("close", "trend_strength"),), ("fast", "slow"),
                       "fast-vs-slow moving-average spread, normalized"),
    )
}


def _validate_vocabulary(registry: FeatureRegistry) -> None:
    """Every operator must BE a registered kind with exactly its params."""
    for op in OPERATORS.values():
        for _field, kind in op.kinds:
            d = registry.get(kind)
            if tuple(d.param_order) != op.params:
                raise ValueError(f"{op.name}: params {op.params} != registered {kind!r} {d.param_order}")
            if not d.point_in_time_safe:
                raise ValueError(f"{op.name}: registered {kind!r} is not point-in-time safe")


_validate_vocabulary(REGISTRY)


class FactorExpression(BaseModel):
    """One registered operator on one input field, plus a normalization."""

    model_config = {"frozen": True, "extra": "forbid"}

    operator: str
    field: str
    #: (name, value) in the operator's declared order -- strict integers (a
    #: bool is never silently read as 1).
    params: tuple[tuple[str, StrictInt], ...]
    normalization: Normalization = Normalization.NONE

    @model_validator(mode="after")
    def _closed_vocabulary(self) -> FactorExpression:
        op = OPERATORS.get(self.operator)
        if op is None:
            raise ValueError(f"unknown operator {self.operator!r}; vocabulary: {sorted(OPERATORS)}")
        if self.field not in op.fields:
            raise ValueError(f"{self.operator} takes {op.fields}, not {self.field!r}")
        if tuple(n for n, _ in self.params) != op.params:
            raise ValueError(f"{self.operator} takes params {op.params} in that order")
        # The registered kind's own rules (bounds, fast < slow) -- the same
        # check the feature engine applies.
        REGISTRY.get(self.kind).resolve_params(dict(self.params))
        return self

    @classmethod
    def of(cls, operator: str, field: str, *args: int, normalization: Normalization = Normalization.NONE,
           ) -> FactorExpression:
        return cls(operator=operator, field=field, params=tuple(zip(OPERATORS[operator].params, args)),
                   normalization=normalization)

    @property
    def kind(self) -> str:
        return OPERATORS[self.operator].kind_for(self.field)

    @property
    def cross_sectional(self) -> bool:
        return self.normalization is not Normalization.NONE

    def render(self) -> str:
        """The human-readable formula -- derived from the same fields that
        compile to the executed `FeatureSpec`."""
        inner = f"{self.operator}({', '.join([self.field, *(str(v) for _, v in self.params)])})"
        return f"rank({inner})" if self.normalization is Normalization.CROSS_SECTIONAL_RANK else inner

    def feature_spec(self) -> FeatureSpec:
        """The ONE registered feature this expression executes."""
        return FeatureSpec(
            kind=self.kind, params=dict(self.params),
            price_field=self.field if self.field != "volume" else "close",
        )

    def feature_metadata(self) -> FeatureMetadata:
        return REGISTRY.get(self.kind).metadata(self.feature_spec())

    @property
    def lookback(self) -> int:
        return self.feature_metadata().lookback

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "vocabulary": EXPRESSION_VOCABULARY_VERSION,
                "expression": self.render(),
                "feature_spec": json.loads(self.feature_spec().canonical_json()),
                "normalization": self.normalization.value,
                "feature_engine_version": FEATURE_ENGINE_VERSION,
            },
            sort_keys=True, separators=(",", ":"),
        )


@dataclass(frozen=True)
class EvaluatedFactor:
    """Factor values aligned row-for-row with the input daily frame (NaN
    where the window is not yet full or an input is invalid -- never
    filled)."""

    expression: str
    feature_name: str
    values: np.ndarray
    metadata: FeatureMetadata
    source_fingerprint: str
    qa_issues: tuple[str, ...]


def evaluate_expression(
    expression: FactorExpression,
    daily: pd.DataFrame,
    *,
    instrument: str,
    price_domain: PriceDomain,
    adjustment_mode: str | None = None,
) -> EvaluatedFactor:
    """Compute ``expression`` on one instrument's one-row-per-trading-day
    series through the registered feature engine (point-in-time only)."""
    if expression.cross_sectional:
        raise CrossSectionalOperatorError(
            f"{expression.render()}: rank() is cross-sectional -- one instrument has no cross-section, so its "
            "rank is a constant with no information. Declare a time-series expression instead."
        )
    result = compute_contiguous_daily_features(
        daily, [expression.feature_spec()], root_symbol=instrument, price_domain=price_domain,
        adjustment_mode=adjustment_mode, require_point_in_time=True,
    )
    frame = result.frame
    frame.assert_point_in_time_safe()
    (name,) = frame.feature_names
    values = frame.features[name].to_numpy(dtype="float64", na_value=np.nan)
    return EvaluatedFactor(
        expression=expression.render(), feature_name=name, values=values, metadata=frame.metadata[name],
        source_fingerprint=frame.lineage.source_fingerprint,
        qa_issues=tuple(f"{i.kind.value}: {i.message}" for i in frame.qa.issues),
    )
