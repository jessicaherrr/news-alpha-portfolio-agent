"""Phase 11 -- interpretable baseline strategy library.

Every baseline is expressed **only** through the closed Phase 10 Strategy DSL:
a typed factory takes economically-meaningful parameters and returns a
:class:`~alpha_agent.strategy.spec.StrategySpec` that passes the standard
Phase 10 compiler unchanged. There is no hidden Python execution logic and no
per-strategy code path.

    baseline config  ->  make_*_spec(...)  ->  StrategySpec  ->  StrategyCompiler

These are **research baselines / benchmarks**, not claimed alpha. They make no
claim to profitability, robustness, statistical significance or production
readiness -- those questions belong to Phase 13 reliability validation.

Families
--------
* ``make_tsmom_spec``          -- multi-horizon time-series momentum / trend
* ``make_ma_trend_spec``       -- fast vs slow moving-average trend
* ``make_breakout_spec``       -- Donchian / channel breakout (causal)
* ``make_mean_reversion_spec`` -- z-score mean reversion with explicit
                                  entry / exit bands

Sizing is deliberately a small explicitly-configured integer target
(``-size .. +size``). No strategy computes leverage, margin, or risk-budget
sizing -- Phase 08 Hard Risk stays authoritative.
"""
from __future__ import annotations

from alpha_agent.strategy.baselines.carry import CARRY_STATUS, carry_baseline_status
from alpha_agent.strategy.baselines.factories import (
    make_breakout_spec,
    make_ma_trend_spec,
    make_mean_reversion_spec,
    make_tsmom_spec,
)
from alpha_agent.strategy.baselines.families import (
    BASELINE_FAMILIES,
    BaselineFamilyDoc,
    baseline_family_docs,
)
from alpha_agent.strategy.baselines.params import (
    BreakoutParams,
    MaTrendParams,
    MeanReversionParams,
    TsmomParams,
)
from alpha_agent.strategy.baselines.silver_bullet import (
    SILVER_BULLET_BENCHMARK,
    SILVER_BULLET_FAMILY,
    SILVER_BULLET_GRID_RANGES,
    SilverBulletParams,
    make_silver_bullet_spec,
    silver_bullet_family_doc,
)

__all__ = [
    "BASELINE_FAMILIES",
    "CARRY_STATUS",
    "SILVER_BULLET_BENCHMARK",
    "SILVER_BULLET_FAMILY",
    "SILVER_BULLET_GRID_RANGES",
    "BaselineFamilyDoc",
    "BreakoutParams",
    "MaTrendParams",
    "MeanReversionParams",
    "SilverBulletParams",
    "TsmomParams",
    "baseline_family_docs",
    "carry_baseline_status",
    "make_breakout_spec",
    "make_ma_trend_spec",
    "make_mean_reversion_spec",
    "make_silver_bullet_spec",
    "make_tsmom_spec",
    "silver_bullet_family_doc",
]
