"""Factor Library (Research Golden Path, section 3A/3B) -- a presentation-only
projection of the REAL registered feature catalog (`services.feature_catalog`,
backed by `alpha_agent.features.REGISTRY`, 40 kinds today) into mechanism
families a researcher can actually reason about, plus real research-coverage
counts from the registry.

No new feature/family is invented here. `FEATURE_FAMILY_TO_BASELINE_FAMILIES`
is a static, documented editorial mapping from a real `FeatureFamily` to the
Phase 11 baseline strategy families (`services.family_catalog`) that are
PRIMARILY built from it -- it decides which registry experiments count as
"research coverage" for that family, never which features exist. A
`FeatureFamily` with no registered kinds (`carry`, `macro`, `cross_market`) is
reported as zero, not fabricated with placeholder features; the task-spec
vocabulary families with no schema representation at all (event/catalyst,
seasonality) are surfaced as an explicit, separate "not yet represented" note,
never as a card with invented counts.
"""
from __future__ import annotations

from typing import Any

from alpha_agent.registry import TrialRole
from alpha_agent.ui import services

#: Display label + one-line description for every real `FeatureFamily` value
#: (`alpha_agent.features.enums.FeatureFamily`). Duplicated as plain strings
#: (not imported) to keep this module import-light, matching
#: `services.GATE_DEFINITIONS`'s own convention; every key here is checked
#: against the live registry by `test_research_factor_library.py`.
_FAMILY_LABELS: dict[str, tuple[str, str]] = {
    "trend": ("Trend / Momentum", "Directional and moving-average-based signals."),
    "return": ("Returns", "Base price-change features that feed trend and other mechanisms."),
    "reversion": ("Mean Reversion", "Distance-from-mean and z-score-based signals."),
    "volatility": ("Volatility", "Realized volatility, ATR, and volatility-regime features."),
    "volume": ("Volume / Participation", "Traded volume and participation-based features."),
    "roll": ("Roll / Session Mechanics", "Contract-roll timing utility features (not a trading mechanism)."),
    "market_structure": ("Market Structure", "ICT-style market-structure / liquidity-sweep features (Silver Bullet)."),
    "carry": ("Term Structure / Carry", "Declared in the feature-family enum; no feature kinds registered yet."),
    "macro": ("Macro / COT Alignment", "Declared in the feature-family enum; no feature kinds registered yet."),
    "cross_market": ("Cross-Market / Relative", "Declared in the feature-family enum; no feature kinds registered yet."),
}

#: Families with zero registered feature kinds today -- reported honestly as
#: a research gap, never hidden and never given a fabricated feature list.
UNSUPPORTED_FEATURE_FAMILIES = ("carry", "macro", "cross_market")

#: Mechanism vocabulary this task spec names that has NO representation at
#: all in the closed feature schema (not even a declared, empty
#: `FeatureFamily`) -- distinct from `UNSUPPORTED_FEATURE_FAMILIES` above,
#: which at least exist as a typed enum member.
NOT_REPRESENTED_MECHANISMS = ("Event / Catalyst", "Seasonality")

#: Editorial mapping ONLY (never changes what features exist): which Phase 11
#: baseline strategy families (`strategy/baselines/families.py`) are PRIMARILY
#: built from each real, populated `FeatureFamily`. `volatility`/`volume`/
#: `roll` have no baseline family that uses them as its primary signal today
#: -- their features exist and are registered, but no strategy has been
#: proposed against them yet, so they report zero research coverage
#: (a real RESEARCH GAP, not an error).
FEATURE_FAMILY_TO_BASELINE_FAMILIES: dict[str, tuple[str, ...]] = {
    "trend": ("tsmom", "ma_trend", "breakout"),
    "return": ("tsmom", "ma_trend"),
    "reversion": ("mean_reversion",),
    "market_structure": ("silver_bullet",),
    "volatility": (),
    "volume": (),
    "roll": (),
}

TESTED = "TESTED"
PARTIALLY_TESTED = "PARTIALLY_TESTED"
RESEARCH_GAP = "RESEARCH_GAP"
NOT_APPLICABLE = "NOT_APPLICABLE"


def _coverage_status(baseline_families: tuple[str, ...], universe: tuple[str, ...]) -> tuple[str, int]:
    """`(status, n_experiments)` for one feature family, counting only real
    CANONICAL (headline-adjudicated) registry trials across `universe` for
    the mapped baseline families -- the same trial set `strategies.py`'s own
    research-status table reads. A family with no mapped baseline strategy at
    all is `NOT_APPLICABLE` (never `RESEARCH_GAP`, which implies a strategy
    exists but has not been run)."""
    if not baseline_families:
        return NOT_APPLICABLE, 0
    tested_cells = 0
    total_cells = len(baseline_families) * len(universe)
    n_experiments = 0
    for family in baseline_families:
        rows = services.list_experiments(strategy_family=family, trial_role=TrialRole.CANONICAL)
        n_experiments += len(rows)
        tested_roots = {r["root_symbol"] for r in rows}
        tested_cells += len(tested_roots & set(universe))
    if tested_cells == 0:
        return RESEARCH_GAP, n_experiments
    if tested_cells < total_cells:
        return PARTIALLY_TESTED, n_experiments
    return TESTED, n_experiments


def mechanism_families(universe: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """One card per real, populated `FeatureFamily` -- kind count, member
    feature kinds, mapped baseline strategies, and real research-coverage
    status/count. `universe` defaults to `services.approved_universe()`.

    Every number here is derived from `services.feature_catalog()` (the
    closed 40-kind registry projection already used by the runtime Research
    Agent, Phase 16) and `services.list_experiments()` -- nothing is invented.
    """
    universe = universe or tuple(services.approved_universe())
    entries = services.feature_catalog()
    by_family: dict[str, list[dict]] = {}
    for e in entries:
        by_family.setdefault(e["family"], []).append(e)

    cards: list[dict[str, Any]] = []
    for family, members in sorted(by_family.items()):
        label, description = _FAMILY_LABELS.get(family, (family, ""))
        baseline_families = FEATURE_FAMILY_TO_BASELINE_FAMILIES.get(family, ())
        status, n_experiments = _coverage_status(baseline_families, universe)
        cards.append(
            {
                "family": family,
                "label": label,
                "description": description,
                "n_features": len(members),
                "feature_kinds": sorted(m["kind"] for m in members),
                "baseline_families": baseline_families,
                "coverage_status": status,
                "n_experiments": n_experiments,
            }
        )
    return cards


def unsupported_axes() -> list[dict[str, str]]:
    """The task-spec-named mechanism axes with zero real backing today --
    declared-but-empty `FeatureFamily` members plus vocabulary with no schema
    representation at all. Kept separate from `mechanism_families()` so the
    UI never renders them as if they had real feature/coverage counts."""
    out = [
        {"label": _FAMILY_LABELS[f][0], "reason": "0 feature kinds registered in this family yet."}
        for f in UNSUPPORTED_FEATURE_FAMILIES
    ]
    out += [{"label": m, "reason": "No schema representation in the current feature registry."} for m in NOT_REPRESENTED_MECHANISMS]
    return out
