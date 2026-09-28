"""Phase 4 -- Three-Lens strategy explainers: economic idea -> equations /
assumptions -> Python feature -> `StrategySpec` -> C++ execution, for each of
the four compiler-supported baseline families.

Intuition and Quant text are projected directly from the frozen Phase 11
family docs (`alpha_agent.agents.compiler_agent.build_family_catalog`) --
never re-authored here, so this can never silently drift from the real
`formula_and_timing` a strategy is actually compiled from. Implementation
pointers are fixed, real module/file references (checked in
`test_phase4_strategy_explainers.py`).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from alpha_agent.agents.compiler_agent import StrategyFamilyCard, build_family_catalog

__all__ = [
    "ImplementationStep",
    "StrategyExplainer",
    "build_strategy_explainer",
    "list_strategy_explainers",
]


class ImplementationStep(BaseModel):
    """One stage of the idea -> execution chain, with a real code pointer."""

    model_config = {"frozen": True, "extra": "forbid"}

    stage: str
    text: str
    pointers: tuple[str, ...] = Field(default_factory=tuple)


class StrategyExplainer(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    family_key: str
    name: str
    intuition: str
    quant: str
    chain: tuple[ImplementationStep, ...]


#: Fixed, real pointers shared by every baseline family -- the Python
#: feature/compile/execution path is the SAME machinery regardless of which
#: family is being explained (only the family-specific formula differs, and
#: that comes from the frozen `StrategyFamilyCard` itself).
_SHARED_CHAIN_POINTERS: dict[str, tuple[str, ...]] = {
    "feature": ("alpha_agent.features",),
    "strategy_spec": (
        "alpha_agent.strategy.candidates_phase_13_5c:spec_for_params",
        "alpha_agent.strategy.compiler",
    ),
    "cpp_execution": (
        "cpp/include/quant_core/engine.hpp",
        "cpp/include/quant_core/momentum_strategy.hpp",
        "cpp/include/quant_core/execution_simulator.hpp",
    ),
}


def _chain_for(card: StrategyFamilyCard) -> tuple[ImplementationStep, ...]:
    return (
        ImplementationStep(
            stage="Economic idea",
            text=card.economic_mechanism,
            pointers=(),
        ),
        ImplementationStep(
            stage="Equations & timing",
            text=card.formula_and_timing,
            pointers=(),
        ),
        ImplementationStep(
            stage="Python feature",
            text=(
                f"The {', '.join(card.parameters)} inputs are computed as typed, causal (trailing-only) "
                "features from raw OHLC bars -- never from a bar's own future value."
            ),
            pointers=_SHARED_CHAIN_POINTERS["feature"],
        ),
        ImplementationStep(
            stage="StrategySpec",
            text=(
                f"Parameters are bound into a typed, closed `StrategySpec` for family '{card.family_key}' "
                f"with default action '{card.default_action}' -- the LLM may propose one of these within "
                "typed bounds, but the deterministic compiler builds the executable object, never the LLM."
            ),
            pointers=_SHARED_CHAIN_POINTERS["strategy_spec"],
        ),
        ImplementationStep(
            stage="C++ execution",
            text=(
                "The compiled spec drives the frozen C++ reference engine: next-bar-or-later execution, "
                "real fills, real contract economics, real commissions -- the official PnL source."
            ),
            pointers=_SHARED_CHAIN_POINTERS["cpp_execution"],
        ),
    )


def build_strategy_explainer(family_key: str) -> StrategyExplainer:
    cards = {c.family_key: c for c in build_family_catalog()}
    if family_key not in cards:
        raise KeyError(f"no compiler-supported family {family_key!r}; known: {sorted(cards)}")
    card = cards[family_key]
    return StrategyExplainer(
        family_key=card.family_key,
        name=card.name,
        intuition=card.economic_mechanism,
        quant=card.formula_and_timing,
        chain=_chain_for(card),
    )


def list_strategy_explainers() -> tuple[StrategyExplainer, ...]:
    return tuple(build_strategy_explainer(c.family_key) for c in build_family_catalog())
