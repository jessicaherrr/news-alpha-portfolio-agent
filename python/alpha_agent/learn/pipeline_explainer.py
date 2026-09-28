"""Phase 4 -- "make quant rigor felt": the fixed Signal -> Next-bar execution
-> Fill -> Contract economics -> Trade PnL -> Equity -> Validation pipeline,
as one ordered, typed walkthrough.

This module never computes a number. Each stage is fixed explanatory text
plus real code pointers; `build_pipeline_walkthrough` optionally attaches ONE
real example value per stage when the caller supplies a verifiably-bound
trade ledger (`alpha_agent.ui.services.find_experiment_bound_trade_ledger`) --
never a fabricated illustrative number. With no bound ledger, every example
field stays `None` and the UI renders the generic explanation only, honestly.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["PipelineStage", "PipelineWalkthrough", "build_pipeline_walkthrough"]


class PipelineStage(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    stage_id: str
    title: str
    text: str
    concept_id: str | None = None
    pointers: tuple[str, ...] = Field(default_factory=tuple)
    example: str | None = None


_STAGE_DEFS: tuple[tuple[str, str, str, str | None, tuple[str, ...]], ...] = (
    (
        "signal",
        "1. Signal",
        (
            "A strategy's decision function reads every bar known up to and including bar T and produces a "
            "target position -- long, short, or flat -- never touching a future bar's value."
        ),
        None,
        ("cpp/include/quant_core/momentum_strategy.hpp",),
    ),
    (
        "next_bar_execution",
        "2. Next-Bar Execution",
        (
            "That decision can be filled no earlier than the next eligible bar, plus modeled latency -- "
            "structurally enforced by the engine, not by strategy-author discipline."
        ),
        "look_ahead_bias",
        ("cpp/include/quant_core/engine.hpp",),
    ),
    (
        "fill",
        "3. Fill",
        (
            "The order is matched against the actual raw contract traded that day (never a back-adjusted "
            "price) with modeled slippage and spread -- a real, auditable execution event."
        ),
        None,
        ("cpp/include/quant_core/fill.hpp", "cpp/include/quant_core/execution_simulator.hpp"),
    ),
    (
        "contract_economics",
        "4. Contract Economics",
        (
            "The fill's price delta is converted to real dollars using the contract's derived point value and "
            "tick value -- never the unreliable raw multiplier sentinel."
        ),
        "contract_economics",
        ("alpha_agent.data.contract_economics:derive_contract_economics",),
    ),
    (
        "trade_pnl",
        "5. Trade PnL",
        (
            "Gross PnL minus real commissions, slippage, and spread costs for that one round-trip -- computed "
            "by the C++ portfolio accountant, never recomputed or approximated in Python."
        ),
        "positive_oos_pnl",
        ("cpp/include/quant_core/portfolio.hpp",),
    ),
    (
        "equity",
        "6. Equity",
        (
            "Trade-level PnL accumulates into a daily equity trace -- the official series every downstream "
            "statistic (Sharpe, drawdown, bootstrap) is computed from."
        ),
        None,
        ("cpp/include/quant_core/position_ledger.hpp",),
    ),
    (
        "validation",
        "7. Validation",
        (
            "The daily equity trace is run through every frozen gate -- null test, BH-FDR, DSR, walk-forward, "
            "cost stress, parameter stability, regime and cross-market robustness -- before any verdict is "
            "assigned."
        ),
        None,
        ("alpha_agent.validation.policy",),
    ),
)


class PipelineWalkthrough(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    stages: tuple[PipelineStage, ...]
    has_real_example: bool = False
    example_source: str | None = None


def _example_for_stage(stage_id: str, *, first_trade: dict[str, Any] | None) -> str | None:
    if not first_trade:
        return None
    if stage_id == "fill":
        px = first_trade.get("fill_price") or first_trade.get("price")
        return f"real fill price: {px}" if px is not None else None
    if stage_id == "trade_pnl":
        pnl = first_trade.get("net_pnl_usd")
        return f"real trade net PnL: ${float(pnl):,.2f}" if pnl is not None else None
    return None


def build_pipeline_walkthrough(*, trade_ledger: dict[str, Any] | None = None) -> PipelineWalkthrough:
    """``trade_ledger`` is exactly the dict
    ``alpha_agent.ui.services.find_experiment_bound_trade_ledger`` returns (or
    ``None``) -- this module never fetches or verifies binding itself."""
    first_trade = None
    if trade_ledger and trade_ledger.get("trades"):
        rows = sorted(trade_ledger["trades"], key=lambda r: int(r.get("trade_index", 0)))
        first_trade = rows[0] if rows else None

    stages = tuple(
        PipelineStage(
            stage_id=sid,
            title=title,
            text=text,
            concept_id=concept_id,
            pointers=pointers,
            example=_example_for_stage(sid, first_trade=first_trade),
        )
        for sid, title, text, concept_id, pointers in _STAGE_DEFS
    )
    return PipelineWalkthrough(
        stages=stages,
        has_real_example=first_trade is not None,
        example_source=trade_ledger.get("source_artifact") if (trade_ledger and first_trade) else None,
    )
