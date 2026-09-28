"""Cross-market evidence (section 18).

The same strategy-family hypothesis is applied SEPARATELY to each root (ES, NQ,
CL, GC, ZN, ...). One ``StrategySpec`` instance stays one-root scoped; raw
contract prices are never pooled. Each root has its own ``ContractSpec``,
execution economics, backtest and daily trace. Cross-market evidence is a
summary over those INDEPENDENT runs.

Not every market must be profitable. What is reported is whether apparent
performance is entirely concentrated in a single root. There is no universal
"passes in N of M markets" rule unless the ReliabilityPolicy states one.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from alpha_agent.validation.enums import EvidenceStatus


class RootRunSummary(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str
    strategy_fingerprint: str
    schedule_hash: str | None = None
    n_trading_days: int
    n_trades: int
    oos_net_pnl_usd: float
    oos_daily_sharpe: float
    oos_annualized_sharpe: float
    is_positive: bool


class CrossMarketEvidence(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    status: EvidenceStatus
    roots: tuple[RootRunSummary, ...] = ()
    n_roots: int = Field(default=0, ge=0)
    n_positive_roots: int = Field(default=0, ge=0)
    # share of total positive PnL contributed by the single best root
    max_single_root_pnl_share: float = 0.0
    herfindahl_pnl: float = 0.0                # sum of squared positive-PnL shares
    is_concentrated_in_one_root: bool = False
    concentration_threshold: float = 0.9


def evaluate_cross_market(
    roots: list[RootRunSummary],
    *,
    concentration_threshold: float = 0.9,
) -> CrossMarketEvidence:
    if not roots:
        return CrossMarketEvidence(status=EvidenceStatus.NOT_EVALUATED)
    pos_pnl = np.asarray([max(0.0, r.oos_net_pnl_usd) for r in roots], dtype=float)
    total = float(pos_pnl.sum())
    shares = (pos_pnl / total) if total > 0 else np.zeros(len(roots))
    max_share = float(shares.max()) if shares.size else 0.0
    return CrossMarketEvidence(
        status=EvidenceStatus.EVALUATED,
        roots=tuple(roots),
        n_roots=len(roots),
        n_positive_roots=sum(1 for r in roots if r.is_positive),
        max_single_root_pnl_share=max_share,
        herfindahl_pnl=float(np.sum(shares**2)),
        is_concentrated_in_one_root=bool(max_share > concentration_threshold and len(roots) >= 2),
        concentration_threshold=concentration_threshold,
    )
