"""Deterministic structural similarity for near-duplicate retrieval (section 10).

No embeddings, no vector database, no LLM. The score is a fixed weighted sum of
transparent, individually inspectable components, and every returned match
carries the reasons that produced its score. This is *memory and warning*
infrastructure: it never changes a ``ReliabilityPolicy`` and never rejects a
hypothesis on its own.

Parameter distance is normalised by the **declared** parameter grid ranges from
:mod:`alpha_agent.strategy.baselines.families` -- the ranges frozen before any
result was inspected -- so "close" means close inside the predeclared
neighbourhood, not close in raw units.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

SIMILARITY_SCHEMA = "registry-similarity/1"

#: Component weights. They sum to 1.0; changing one is a visible, reviewable
#: change to retrieval behaviour, never a hidden tuning knob.
WEIGHTS: dict[str, float] = {
    "same_family": 0.35,
    "same_root": 0.20,
    "same_structure": 0.15,
    "same_feature_set": 0.10,
    "parameter_proximity": 0.20,
}

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_range(text: str) -> tuple[float, float] | None:
    """Parse a declared range string such as ``"5 .. 60 bars"`` or
    ``"0.0 (near edge) .. 1.0 (far edge)"``. Non-numeric declarations
    (``"one of [...]"``) return ``None`` and contribute no distance."""
    if ".." not in text:
        return None
    left, _, right = text.partition("..")
    lo = _NUM.search(left)
    hi = _NUM.search(right)
    if lo is None or hi is None:
        return None
    a, b = float(lo.group()), float(hi.group())
    if b <= a:
        return None
    return a, b


def declared_param_ranges(strategy_family: str) -> dict[str, tuple[float, float]]:
    """Numeric ``(lo, hi)`` per parameter, derived from the frozen
    ``param_grid_ranges`` declarations. Lazily imported so the registry stays
    usable without the strategy layer's optional dependencies."""
    from alpha_agent.strategy.baselines.families import BASELINE_FAMILIES
    from alpha_agent.strategy.baselines.silver_bullet import SILVER_BULLET_FAMILY

    docs = {d.key: d for d in (*BASELINE_FAMILIES, SILVER_BULLET_FAMILY)}
    doc = docs.get(strategy_family)
    if doc is None:
        return {}
    out: dict[str, tuple[float, float]] = {}
    for name, text in doc.param_grid_ranges.items():
        rng = _parse_range(text)
        if rng is not None:
            out[name] = rng
    return out


#: Administrative parameter keys that carry no structural information --
#: `root_symbol` is already compared separately via `same_root`, but two
#: different WRITE PATHS disagree about whether it lives inside the stored
#: `params` dict at all: the Phase 14 historical import's
#: `reconstruct_variant` dumps every typed-params-model field (root_symbol
#: included), while `StrategyCompilerAgent._build_from_template`'s
#: `resolved_params` is deliberately just the plan's declared template params
#: (root_symbol is tracked on the side as the template's own field, since it
#: is never itself a tunable parameter of a family). Left uncorrected, that
#: bookkeeping difference alone changes `parameter_names`' SET and zeroes
#: `same_structure` for two experiments that are otherwise structurally
#: identical -- an artifact of which vintage wrote the row, not a real
#: difference in what the strategy is.
_ADMINISTRATIVE_PARAM_KEYS = frozenset({"root_symbol"})


def structural_shape(*, strategy_family: str, params: dict,
                     signal_cadence: str = "", execution_cadence: str = "") -> dict:
    """The ``StrategySpec`` *shape*: what kind of strategy this is, independent
    of the concrete parameter values."""
    return {
        "strategy_family": strategy_family,
        "parameter_names": sorted(
            str(k) for k in params if str(k) not in _ADMINISTRATIVE_PARAM_KEYS
        ),
        "signal_cadence": signal_cadence,
        "execution_cadence": execution_cadence,
    }


def parameter_distance(
    strategy_family: str, a: dict, b: dict
) -> tuple[float | None, dict[str, float]]:
    """Mean normalised distance over the shared numeric parameters.

    Returns ``(distance, per_parameter)``. ``distance`` is ``None`` when the two
    parameter sets share no comparable numeric parameter (then the proximity
    component contributes nothing rather than a fabricated zero).
    """
    ranges = declared_param_ranges(strategy_family)
    per: dict[str, float] = {}
    for key in sorted(set(a) & set(b)):
        va, vb = a[key], b[key]
        if isinstance(va, bool) or isinstance(vb, bool):
            continue
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            continue
        rng = ranges.get(key)
        if rng is None:
            continue
        lo, hi = rng
        per[key] = min(1.0, abs(float(va) - float(vb)) / (hi - lo))
    if not per:
        return None, {}
    return sum(per.values()) / len(per), per


class SimilarityBreakdown(BaseModel):
    """A transparent, human-checkable explanation of one similarity score."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = SIMILARITY_SCHEMA
    score: float
    components: dict[str, float] = Field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    parameter_distance: float | None = None
    per_parameter_distance: dict[str, float] = Field(default_factory=dict)

    def explain(self) -> str:
        return "; ".join(self.reasons)


def similarity(
    *,
    query_family: str,
    query_root: str,
    query_params: dict,
    query_shape: dict,
    query_features: tuple[str, ...] = (),
    candidate_family: str,
    candidate_root: str,
    candidate_params: dict,
    candidate_shape: dict,
    candidate_features: tuple[str, ...] = (),
) -> SimilarityBreakdown:
    """Score one candidate experiment against a proposed hypothesis."""
    comp: dict[str, float] = {}
    reasons: list[str] = []

    same_family = query_family == candidate_family
    comp["same_family"] = WEIGHTS["same_family"] if same_family else 0.0
    if same_family:
        reasons.append(f"same_family={candidate_family}")

    same_root = query_root == candidate_root
    comp["same_root"] = WEIGHTS["same_root"] if same_root else 0.0
    if same_root:
        reasons.append(f"same_root={candidate_root}")

    same_structure = query_shape == candidate_shape
    comp["same_structure"] = WEIGHTS["same_structure"] if same_structure else 0.0
    if same_structure:
        reasons.append("same_structure")

    same_features = bool(query_features) and tuple(query_features) == tuple(candidate_features)
    comp["same_feature_set"] = WEIGHTS["same_feature_set"] if same_features else 0.0
    if same_features:
        reasons.append("same_feature_set")

    dist: float | None = None
    per: dict[str, float] = {}
    if same_family:
        dist, per = parameter_distance(candidate_family, query_params, candidate_params)
    if dist is None:
        comp["parameter_proximity"] = 0.0
        if same_family:
            reasons.append("parameter_distance=n/a (no comparable declared parameter)")
    else:
        comp["parameter_proximity"] = WEIGHTS["parameter_proximity"] * (1.0 - dist)
        detail = ", ".join(f"{k}={v:.4f}" for k, v in sorted(per.items()))
        reasons.append(f"parameter_distance={dist:.4f} ({detail})")

    return SimilarityBreakdown(
        score=round(sum(comp.values()), 12),
        components={k: round(v, 12) for k, v in comp.items()},
        reasons=tuple(reasons),
        parameter_distance=None if dist is None else round(dist, 12),
        per_parameter_distance={k: round(v, 12) for k, v in per.items()},
    )


def rank_key(item: tuple[SimilarityBreakdown, Any]) -> tuple:
    """Deterministic ordering: score desc, then parameter distance asc, then id."""
    breakdown, ident = item
    return (
        -breakdown.score,
        1.0 if breakdown.parameter_distance is None else breakdown.parameter_distance,
        str(ident),
    )
