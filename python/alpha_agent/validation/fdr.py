"""Benjamini-Hochberg FDR control across an experiment family (section 12).

Every inspected strategy variant is a trial and stays in the denominator -- the
failed ones are never dropped. Sorting and tie handling are deterministic:
ties in ``p`` are ordered by original index (a stable sort), and the step-up
threshold is applied with the standard "largest passing rank wins" rule.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


def benjamini_hochberg(p_values: list[float], q: float = 0.10) -> list[bool]:
    """Back-compatible boolean reject vector (kept from the Phase-01 stub)."""
    return [d.rejected for d in benjamini_hochberg_decisions(p_values, q).decisions]


def bh_qvalues(p_values: list[float]) -> list[float]:
    """BH-adjusted q-values, one per input p-value, in input order.

    q_(i) = min_{k >= i} ( p_(k) * m / k ), clamped to <= 1, then de-sorted.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return []
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.lexsort((np.arange(m), p))          # deterministic: (p, index)
    ranked = p[order]
    scaled = ranked * m / np.arange(1, m + 1)
    # step-up monotone minimum from the largest rank down
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return [float(v) for v in q]


class TrialDecision(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    trial_index: int
    label: str
    p_value: float
    q_value: float
    rejected: bool


class FdrResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    q_threshold: float = Field(gt=0.0, lt=1.0)
    n_trials: int = Field(ge=0)
    n_rejected: int = Field(ge=0)
    decisions: tuple[TrialDecision, ...]


def benjamini_hochberg_decisions(
    p_values: list[float],
    q_threshold: float = 0.10,
    labels: list[str] | None = None,
) -> FdrResult:
    p = list(p_values)
    m = len(p)
    labs = list(labels) if labels is not None else [f"trial_{i}" for i in range(m)]
    if len(labs) != m:
        raise ValueError("labels length must match p_values length")
    qs = bh_qvalues(p)

    # standard BH: find the largest rank k with p_(k) <= q * k / m; reject all
    # trials with rank <= k. Equivalent to q_value <= q_threshold given the
    # monotone step-up q_values above.
    rejected = [qv <= q_threshold for qv in qs]
    decisions = tuple(
        TrialDecision(
            trial_index=i, label=labs[i], p_value=float(p[i]),
            q_value=qs[i], rejected=bool(rejected[i]),
        )
        for i in range(m)
    )
    return FdrResult(
        q_threshold=q_threshold,
        n_trials=m,
        n_rejected=int(sum(rejected)),
        decisions=decisions,
    )
