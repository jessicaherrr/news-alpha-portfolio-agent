"""Carry / term-structure baseline -- status and deferral (Phase 11 section 21).

A single :class:`StrategySpec` is one-root scoped and its only inputs are
registered :class:`FeatureSpec` columns on a point-in-time
:class:`~alpha_agent.features.frame.FeatureFrame`. A production carry signal
needs a **front / next curve** -- two or more overlapping raw-contract quotes at
the same timestamp -- which the Phase 09 carry interface models with a separate
typed object graph (``ContractQuote`` / ``CurveObservation`` / ``CarrySnapshot``
/ ``carry_frame`` in :mod:`alpha_agent.features.carry`). Those are **not**
registered feature ``kind``s, so the closed Phase 10 DSL has no operand that can
reference a curve.

Decision for Phase 11: **defer the production carry baseline.** Wiring carry in
cleanly would require new DSL primitives (a curve/relative-contract operand), and
prompt 11 explicitly says not to redesign Phase 10 solely for carry. Real
front/next curve data (overlapping raw contracts) is also not yet ingested.

Any future synthetic carry spec built on ``alpha_agent.features.carry`` fixtures
must be labelled *synthetic / interface validation only* until real curve data
exists -- no fabricated production performance.
"""
from __future__ import annotations

from pydantic import BaseModel


class CarryBaselineStatus(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    available: bool
    reason: str
    blocking_requirements: list[str]
    dsl_primitives_needed: list[str]


CARRY_STATUS = CarryBaselineStatus(
    available=False,
    reason=(
        "The closed Phase 10 DSL can only reference registered single-series FeatureSpec "
        "columns; carry needs an overlapping front/next raw-contract curve, which is a "
        "separate typed interface (alpha_agent.features.carry) with no registered feature "
        "kind. Deferring rather than redesigning Phase 10 for carry (prompt 11 section 21)."
    ),
    blocking_requirements=[
        "ingested real front/next curve data (overlapping raw contracts at one timestamp)",
        "a registered carry FeatureSpec kind OR a DSL curve operand",
    ],
    dsl_primitives_needed=[
        "a relative-contract / curve-point operand (e.g. front vs next settlement)",
        "an annualized-carry feature kind on the FeatureRegistry",
    ],
)


def carry_baseline_status() -> CarryBaselineStatus:
    return CARRY_STATUS
