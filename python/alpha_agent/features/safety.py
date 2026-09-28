"""Source-price safety vs derived-feature safety (Phase 09.1 / 09.2).

The single ``research_only`` flag conflated distinctions the research
architecture needs to keep apart. Phase 09.2 further separates **SOURCE** price
safety from **DERIVED FEATURE** value safety:

* :class:`SourceSafety` -- what a *price series* confers, before any feature.
  ``execution_price_safe`` is a genuine source-level property: a real
  ``RawContract`` price may be an execution reference; a continuous or
  back-adjusted price may not.

* :class:`FeatureSafety` -- what a *computed feature column* confers. A feature
  value (a moving average, a return, an ATR, a z-score, a carry number, ...) is
  a transform, not a price. **It is never an execution reference / Fill /
  slippage price**, regardless of how execution-price-safe its source was.
  Features exist to drive ``Signal``s, not ``Fill``s.

  ``FeatureSafety.execution_price_safe`` is therefore **always ``False``** for
  every column the :class:`FeatureRegistry` can produce. Execution prices are
  selected only by the deterministic C++ execution path from ``RawContract``
  market data (BOUNDARY_CONTRACT section E; ``make_fill`` is the final guard).

Source safety matrix:

+---------------------------------+---------------+-------------+----------------------+
| source                          | point_in_time | signal_safe | execution_price_safe |
+---------------------------------+---------------+-------------+----------------------+
| RawContract (causal)            | yes           | yes         | yes  (source only)   |
| RawContinuous (causal)          | yes           | yes         | no                   |
| BackAdjusted, point_in_time     | yes           | yes         | no                   |
| BackAdjusted, retrospective     | no            | no          | no                   |
+---------------------------------+---------------+-------------+----------------------+

Derived-feature safety = source (point_in_time / signal) AND the feature's own
point-in-time property; ``execution_price_safe`` is hard-wired ``False``.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# The message attached to every feature column's reasons["execution_price"].
FEATURE_NEVER_EXECUTION_PRICE = (
    "a computed feature value is a transform, never an execution reference / Fill / "
    "slippage / contract execution price; execution prices are selected only by the "
    "deterministic C++ execution path from RawContract market data "
    "(BOUNDARY_CONTRACT E; make_fill is the final guard)"
)


class SourceSafety(BaseModel):
    """Safety a source *price series* confers, before any feature is applied."""

    model_config = {"frozen": True}

    point_in_time_safe: bool
    signal_safe: bool
    execution_price_safe: bool          # SOURCE-level: True only for a RawContract feed
    reasons: dict[str, str] = Field(default_factory=dict)


class FeatureSafety(BaseModel):
    """Effective safety of one computed feature column.

    ``point_in_time_safe`` / ``signal_safe`` = source safety AND the feature's own
    point-in-time property. ``execution_price_safe`` is **always ``False``** -- a
    feature value is never an execution price.
    """

    model_config = {"frozen": True}

    feature: str
    point_in_time_safe: bool
    signal_safe: bool
    execution_price_safe: bool = False
    reasons: dict[str, str] = Field(default_factory=dict)


class FeatureSafetyError(ValueError):
    """A FeatureFrame was asserted safe for a use it is not safe for."""


def combine(
    source: SourceSafety,
    *,
    feature: str,
    feature_point_in_time_safe: bool,
    feature_reason: str | None = None,
) -> FeatureSafety:
    reasons = dict(source.reasons)
    if not feature_point_in_time_safe:
        reasons["feature"] = feature_reason or (
            "the feature itself looks forward across the sample (retrospective)"
        )
    # A derived feature value never inherits source execution-price safety.
    reasons["execution_price"] = FEATURE_NEVER_EXECUTION_PRICE
    return FeatureSafety(
        feature=feature,
        point_in_time_safe=source.point_in_time_safe and feature_point_in_time_safe,
        signal_safe=source.signal_safe and feature_point_in_time_safe,
        execution_price_safe=False,
        reasons=reasons,
    )
