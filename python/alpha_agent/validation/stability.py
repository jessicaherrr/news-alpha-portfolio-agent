"""Parameter plateau / stability analysis (section 14).

A strategy should not look reliable only at one isolated parameter spike. A
PREDECLARED neighbourhood of parameter sets (declared before any results are
seen -- never grown after) is evaluated out-of-sample. Every neighbour is its
own ``StrategySpec`` fingerprint and its own multiple-testing trial.

This module holds the typed neighbourhood + the stability evidence. Mapping a
parameter dict to a ``StrategySpec`` and running each OOS backtest is the
runner's job (it is strategy-family specific).
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, model_validator

from alpha_agent.validation.fingerprint import fingerprint


class ParameterNeighbourhood(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "param-neighbourhood/1"
    strategy_key: str                          # "silver_bullet" | "tsmom" | ...
    canonical_params: dict                     # scalar params only
    neighbour_params: tuple[dict, ...]         # PREDECLARED, not grown after results

    @model_validator(mode="after")
    def _scalars(self) -> ParameterNeighbourhood:
        for d in (self.canonical_params, *self.neighbour_params):
            for k, val in d.items():
                if not isinstance(k, str) or isinstance(val, (list, dict, tuple)):
                    raise TypeError("neighbourhood params must be flat {str: scalar}")
        return self

    def all_param_sets(self) -> list[tuple[bool, dict]]:
        """[(is_canonical, params), ...] -- canonical first."""
        return [(True, self.canonical_params), *((False, d) for d in self.neighbour_params)]

    def identity(self) -> str:
        return fingerprint(
            "valparamnbhd1",
            {
                "schema_version": self.schema_version,
                "strategy_key": self.strategy_key,
                "canonical": self.canonical_params,
                "neighbours": sorted(
                    fingerprint("p", d) for d in self.neighbour_params
                ),
            },
        )


class NeighbourResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    params_label: str
    strategy_fingerprint: str
    is_canonical: bool
    n_trades: int
    oos_net_pnl_usd: float
    oos_daily_sharpe: float
    oos_annualized_sharpe: float


class ParameterStabilityResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    neighbourhood_fingerprint: str
    n_evaluated: int = Field(ge=0)
    fraction_positive_net: float
    fraction_positive_sharpe: float
    sharpe_dispersion_std: float
    net_pnl_dispersion_std: float
    canonical_daily_sharpe: float
    median_neighbour_daily_sharpe: float
    canonical_sharpe_rank: int                 # 1 == best of the neighbourhood
    canonical_sharpe_percentile: float         # fraction of the neighbourhood canonical beats
    canonical_is_isolated_spike: bool
    neighbours: tuple[NeighbourResult, ...]


def summarize_parameter_stability(
    neighbourhood: ParameterNeighbourhood,
    results: list[NeighbourResult],
) -> ParameterStabilityResult:
    n = len(results)
    sharpes = np.asarray(
        [r.oos_daily_sharpe for r in results if np.isfinite(r.oos_daily_sharpe)], dtype=float
    )
    nets = np.asarray([r.oos_net_pnl_usd for r in results], dtype=float)
    canon = next((r for r in results if r.is_canonical), None)
    canon_sharpe = float(canon.oos_daily_sharpe) if canon and np.isfinite(canon.oos_daily_sharpe) else float("nan")

    others = np.asarray(
        [r.oos_daily_sharpe for r in results
         if not r.is_canonical and np.isfinite(r.oos_daily_sharpe)],
        dtype=float,
    )
    frac_pos_sharpe = float(np.mean(sharpes > 0)) if sharpes.size else 0.0
    frac_pos_net = float(np.mean(nets > 0)) if nets.size else 0.0
    median_other = float(np.median(others)) if others.size else float("nan")

    if sharpes.size and np.isfinite(canon_sharpe):
        rank = int(1 + np.count_nonzero(sharpes > canon_sharpe))
        pct = float(np.mean(sharpes <= canon_sharpe))
    else:
        rank = n
        pct = 0.0

    # Isolated spike (documented default heuristic; the PASS/REJECT threshold on
    # `fraction_positive_*` lives in the ReliabilityPolicy): the canonical point
    # is clearly positive, most neighbours are not, and the canonical Sharpe sits
    # far above the neighbour median relative to the neighbourhood dispersion.
    disp = float(sharpes.std(ddof=1)) if sharpes.size >= 2 else 0.0
    isolated = bool(
        np.isfinite(canon_sharpe)
        and canon_sharpe > 0.0
        and others.size >= 2
        and float(np.mean(others > 0)) < 0.5
        and disp > 0.0
        and (canon_sharpe - median_other) > 2.0 * disp
    )

    return ParameterStabilityResult(
        neighbourhood_fingerprint=neighbourhood.identity(),
        n_evaluated=n,
        fraction_positive_net=frac_pos_net,
        fraction_positive_sharpe=frac_pos_sharpe,
        sharpe_dispersion_std=disp,
        net_pnl_dispersion_std=float(nets.std(ddof=1)) if nets.size >= 2 else 0.0,
        canonical_daily_sharpe=canon_sharpe if np.isfinite(canon_sharpe) else 0.0,
        median_neighbour_daily_sharpe=median_other if np.isfinite(median_other) else 0.0,
        canonical_sharpe_rank=rank,
        canonical_sharpe_percentile=pct,
        canonical_is_isolated_spike=isolated,
        neighbours=tuple(results),
    )
