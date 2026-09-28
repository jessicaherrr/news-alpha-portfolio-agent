"""Predeclared typed strategy ablations (section 16).

An ablation removes one claimed mechanism component (e.g. the Silver Bullet time
window, or the displacement requirement, or the slow momentum horizon) and is
represented cleanly as its own ``StrategySpec`` -- never a dynamic mutation of
the canonical strategy. Each ablation is its own fingerprint and its own trial.

Purpose: decide whether apparent performance depends on the claimed mechanism or
on a simpler component. The best ablation is **not** adopted as a replacement
strategy in Phase 13.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from alpha_agent.validation.enums import AblationKind
from alpha_agent.validation.fingerprint import fingerprint


class AblationSpec(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    label: str
    kind: AblationKind
    # scalar param overrides applied to the canonical parameter dict to produce
    # the ablated StrategySpec (the runner maps params -> spec per family).
    param_overrides: dict = Field(default_factory=dict)
    description: str = ""

    def identity(self) -> str:
        return fingerprint(
            "valablation1",
            {"kind": self.kind.value, "overrides": self.param_overrides},
        )


class AblationResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    label: str
    kind: AblationKind
    ablation_fingerprint: str
    strategy_fingerprint: str
    n_trades: int
    oos_net_pnl_usd: float
    oos_daily_sharpe: float
    oos_annualized_sharpe: float
    sharpe_delta_vs_canonical: float           # canonical - ablation (positive => mechanism helps)


class AblationReport(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    canonical_daily_sharpe: float
    n_ablations: int = Field(ge=0)
    results: tuple[AblationResult, ...]
    best_ablation_label: str | None
    canonical_minus_best_ablation_sharpe: float
    # True => the canonical Sharpe is not explained away by any single-component
    # ablation (canonical beats every ablation by a positive margin).
    mechanism_adds_value: bool


def summarize_ablations(
    canonical_daily_sharpe: float, results: list[AblationResult]
) -> AblationReport:
    if not results:
        return AblationReport(
            canonical_daily_sharpe=float(canonical_daily_sharpe),
            n_ablations=0,
            results=(),
            best_ablation_label=None,
            canonical_minus_best_ablation_sharpe=float("nan"),
            mechanism_adds_value=False,
        )
    best = max(results, key=lambda r: (r.oos_daily_sharpe if np.isfinite(r.oos_daily_sharpe) else -np.inf))
    margin = float(canonical_daily_sharpe - best.oos_daily_sharpe)
    return AblationReport(
        canonical_daily_sharpe=float(canonical_daily_sharpe),
        n_ablations=len(results),
        results=tuple(results),
        best_ablation_label=best.label,
        canonical_minus_best_ablation_sharpe=margin,
        mechanism_adds_value=bool(margin > 0.0),
    )
