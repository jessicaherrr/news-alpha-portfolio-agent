"""Predeclared cost-stress scenarios (section 15).

IMPORTANT: cost stress is never a post-hoc subtraction in Python. Each scenario
reruns the **same** C++ execution / accounting path with scaled cost assumptions
(the ``quant_backtest_targets_csv`` CLI takes optional
``[commission] [slippage_ticks] [spread_ticks]`` arguments); every non-cost
strategy semantic is held identical. This module declares the scenarios and
summarises the reruns -- the reruns themselves are orchestrated by :mod:`.runner`.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from alpha_agent.validation.enums import CostScenarioKind
from alpha_agent.validation.fingerprint import fingerprint

# The frozen reference-path execution assumptions (manifest.REFERENCE_EXECUTION_CONFIG).
REFERENCE_COMMISSION_USD = 2.0
REFERENCE_SLIPPAGE_TICKS = 0.0
REFERENCE_SPREAD_TICKS = 0.0


class CommissionBasis(str, Enum):
    """Where a commission rate comes from -- never blurred."""

    #: A sourced historical schedule (broker / exchange fee metadata).
    SOURCED_EXECUTION_METADATA = "SOURCED_EXECUTION_METADATA"
    #: A declared research assumption -- NOT observed broker truth.
    DECLARED_RESEARCH_ASSUMPTION = "DECLARED_RESEARCH_ASSUMPTION"


class RootCommission(BaseModel):
    """One root's commission, USD per unit (a futures contract, or one share
    for an ETF root) -- the C++ ``ExecutionConfig::commission_per_contract_usd_by_root``
    entry, with the provenance the engine has no concept of."""

    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str = Field(min_length=1)
    commission_per_unit_usd: float = Field(ge=0.0)
    unit: Literal["contract", "share"]
    basis: CommissionBasis
    source: str


class CostScenario(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    label: str
    kind: CostScenarioKind = CostScenarioKind.MULTIPLIER
    multiplier: float = Field(default=1.0, ge=0.0, le=100.0)
    commission_per_contract_usd: float | None = Field(default=None, ge=0.0)
    slippage_ticks: float | None = Field(default=None, ge=0.0)
    spread_ticks: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> CostScenario:
        if self.kind == CostScenarioKind.ABSOLUTE and self.commission_per_contract_usd is None:
            raise ValueError("ABSOLUTE cost scenario needs commission_per_contract_usd")
        return self

    def resolve(self, base: CostStressPlan) -> tuple[float, float, float]:
        if self.kind == CostScenarioKind.MULTIPLIER:
            return (
                base.base_commission_per_contract_usd * self.multiplier,
                base.base_slippage_ticks * self.multiplier,
                base.base_spread_ticks * self.multiplier,
            )
        return (
            float(self.commission_per_contract_usd or 0.0),
            float(self.slippage_ticks or 0.0),
            float(self.spread_ticks or 0.0),
        )


class CostStressPlan(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "cost-stress/1"
    base_commission_per_contract_usd: float = Field(default=REFERENCE_COMMISSION_USD, ge=0.0)
    base_slippage_ticks: float = Field(default=REFERENCE_SLIPPAGE_TICKS, ge=0.0)
    base_spread_ticks: float = Field(default=REFERENCE_SPREAD_TICKS, ge=0.0)
    scenarios: tuple[CostScenario, ...]
    #: ADDITIVE (News Alpha Phase H acceptance patch). An explicit per-root
    #: commission schedule; when set it -- not ``base_commission_per_contract_usd``
    #: -- is the commission of every run, scaled by each MULTIPLIER scenario.
    #: ``None`` (every pre-existing plan) leaves the plan and its identity
    #: byte-identical to before.
    commission_schedule: tuple[RootCommission, ...] | None = None

    @model_validator(mode="after")
    def _has_baseline(self) -> CostStressPlan:
        labels = [s.label for s in self.scenarios]
        if len(labels) != len(set(labels)):
            raise ValueError("cost scenario labels must be unique")
        base_like = [
            s for s in self.scenarios
            if s.kind == CostScenarioKind.MULTIPLIER and s.multiplier == 1.0
        ]
        if not base_like:
            raise ValueError("a cost-stress plan must include a 1.0x baseline scenario")
        if self.commission_schedule is not None:
            roots = [c.root_symbol for c in self.commission_schedule]
            if not roots or roots != sorted(set(roots)):
                raise ValueError("a commission schedule names each root once, sorted")
            if any(s.kind != CostScenarioKind.MULTIPLIER for s in self.scenarios):
                raise ValueError("a commission schedule is stressed by MULTIPLIER scenarios only")
        return self

    def schedule_for(self, scenario: CostScenario) -> dict[str, float]:
        """The per-root commission of one scenario (the schedule x its multiplier)."""
        if self.commission_schedule is None:
            raise ValueError("this plan has no commission schedule")
        return {c.root_symbol: c.commission_per_unit_usd * scenario.multiplier for c in self.commission_schedule}

    def baseline_label(self) -> str:
        for s in self.scenarios:
            if s.kind == CostScenarioKind.MULTIPLIER and s.multiplier == 1.0:
                return s.label
        raise ValueError("no baseline scenario")  # pragma: no cover -- validator guards

    def identity(self) -> str:
        payload = self.model_dump(mode="json")
        if payload["commission_schedule"] is None:
            del payload["commission_schedule"]  # a scalar plan fingerprints exactly as before the schedule existed
        return fingerprint("valcoststress1", payload)


DEFAULT_COST_STRESS_PLAN = CostStressPlan(
    scenarios=(
        CostScenario(label="baseline_1_0x", multiplier=1.0),
        CostScenario(label="stress_1_5x", multiplier=1.5),
        CostScenario(label="stress_2_0x", multiplier=2.0),
    ),
)


class CostScenarioResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    label: str
    commission_per_contract_usd: float
    slippage_ticks: float
    spread_ticks: float
    n_fills: int
    oos_gross_pnl_usd: float
    oos_costs_usd: float
    oos_net_pnl_usd: float
    daily_sharpe: float
    annualized_sharpe: float
    net_pnl_ratio_vs_baseline: float          # scenario_net / baseline_net (nan if baseline ~0)
    sharpe_delta_vs_baseline: float


class CostStressReport(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    plan_fingerprint: str
    baseline_label: str
    scenarios: tuple[CostScenarioResult, ...]
    # worst fractional net-PnL loss vs baseline across stressed (>1x) scenarios;
    # 0.0 == no degradation, 1.0 == all edge gone, >1.0 == turned to a loss.
    max_net_pnl_degradation: float
    worst_scenario_label: str


def summarize_cost_stress(
    plan: CostStressPlan,
    per_scenario: list[CostScenarioResult],
) -> CostStressReport:
    base = next(r for r in per_scenario if r.label == plan.baseline_label())
    worst = 0.0
    worst_label = plan.baseline_label()
    for r in per_scenario:
        if r.label == base.label:
            continue
        if abs(base.oos_net_pnl_usd) < 1e-9:
            degr = 0.0 if r.oos_net_pnl_usd >= 0 else 1.0
        else:
            degr = 1.0 - (r.oos_net_pnl_usd / base.oos_net_pnl_usd)
        if degr > worst:
            worst = degr
            worst_label = r.label
    return CostStressReport(
        plan_fingerprint=plan.identity(),
        baseline_label=base.label,
        scenarios=tuple(per_scenario),
        max_net_pnl_degradation=float(worst),
        worst_scenario_label=worst_label,
    )
