"""Phase 21 -- the deterministic hard-risk configuration for a paper run.

``PaperRiskPolicy`` is the ONLY way paper trading configures the C++
``RiskConfig`` a run is gated by (``quant_paper_trading_targets_csv`` /
``quant::PortfolioRiskManager``, cpp/include/quant_core/paper_trading_run.hpp).
It is set once, by a human, when a paper run STARTS, and is immutable for the
life of that run (CLAUDE.md risk rule 1: hard risk limits are deterministic and
cannot be overridden by an LLM). Changing risk limits mid-run is not supported
-- stop the run and start a new one under a new policy, which is exactly the
"a scientific/operational correction is a NEW run, never an in-place edit"
discipline the rest of this codebase already uses for the experiment registry.

Defaults mirror ``quant::RiskConfig``'s own frozen Phase 08 defaults (position
caps, $1,500 daily-loss kill switch, 10% drawdown kill switch) UNCHANGED, with
two deliberate overrides:

* ``missing_margin_policy = "treat_as_zero"`` and ``max_margin_utilization_pct
  = 0.0`` (disabled). CLAUDE.md forbids guessing a margin figure from
  notional; this system has no committed per-root CME margin schedule yet
  (``quant::MarginModel`` / ``--margin-config=``). Rather than fabricate one,
  margin utilization is simply not enforced as a limit -- the resulting gap is
  surfaced verbatim as ``portfolio_at_end.margin_complete == false``, never
  silently treated as satisfied. A future phase that commits a real, dated,
  sourced margin schedule can set ``missing_margin_policy = "reject"`` and a
  real ``max_margin_utilization_pct`` without touching this module's shape.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from alpha_agent.validation.fingerprint import fingerprint

RISK_CONFIG_CSV_HEADER: tuple[str, ...] = (
    "max_contracts_per_symbol", "max_contracts_per_root", "max_gross_contracts",
    "max_order_contracts", "max_gross_exposure_usd", "max_gross_leverage",
    "max_net_leverage", "max_margin_utilization_pct", "missing_margin_policy",
    "max_daily_loss_usd", "max_drawdown_pct", "max_drawdown_usd", "stale_mark_policy",
    "starting_capital_usd", "mark_staleness_tolerance_ns", "day_boundary_offset_ns",
)


class PaperRiskPolicy(BaseModel):
    """Mirrors ``quant::RiskConfig`` field-for-field (risk_config.hpp) plus the
    starting-capital / staleness / day-boundary knobs of ``quant::PortfolioConfig``
    it embeds. See cpp/include/quant_core/risk_config.hpp for the semantics of
    every field -- this module adds no risk semantics of its own, only the
    typed, versioned, human-set configuration boundary and its CSV
    serialisation for the C++ CLI."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "paper-risk-policy/1"

    max_contracts_per_symbol: int = Field(default=5, gt=0)
    max_contracts_per_root: int = Field(default=10, gt=0)
    max_gross_contracts: int = Field(default=20, gt=0)
    max_order_contracts: int = Field(default=0, ge=0)  # 0 == disabled

    max_gross_exposure_usd: float = Field(default=0.0, ge=0.0)   # 0 == disabled
    max_gross_leverage: float = Field(default=0.0, ge=0.0)       # 0 == disabled
    max_net_leverage: float = Field(default=0.0, ge=0.0)         # 0 == disabled

    max_margin_utilization_pct: float = Field(default=0.0, ge=0.0)  # 0 == disabled (see module docstring)
    missing_margin_policy: Literal["reject", "treat_as_zero"] = "treat_as_zero"

    max_daily_loss_usd: float = Field(default=1500.0, ge=0.0)    # 0 == disabled
    max_drawdown_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    max_drawdown_usd: float = Field(default=0.0, ge=0.0)         # 0 == disabled

    stale_mark_policy: Literal["reject_risk_increasing", "ignore"] = "reject_risk_increasing"

    starting_capital_usd: float = Field(default=100_000.0, gt=0.0)
    mark_staleness_tolerance_ns: int = Field(default=0, ge=0)
    day_boundary_offset_ns: int = Field(default=0)

    @model_validator(mode="after")
    def _margin_disabled_when_unknown(self) -> PaperRiskPolicy:
        if self.missing_margin_policy == "treat_as_zero" and self.max_margin_utilization_pct > 0.0:
            raise ValueError(
                "max_margin_utilization_pct must stay 0 (disabled) while "
                "missing_margin_policy is 'treat_as_zero' -- enforcing a utilization "
                "limit against a margin figure this policy also declares unknown "
                "would be an internally inconsistent risk configuration"
            )
        return self

    def identity(self) -> str:
        return fingerprint("paperrisk1", self.model_dump(mode="json"))

    def csv_rows(self) -> tuple[str, str]:
        header = ",".join(RISK_CONFIG_CSV_HEADER)
        data = ",".join(
            [
                str(self.max_contracts_per_symbol),
                str(self.max_contracts_per_root),
                str(self.max_gross_contracts),
                str(self.max_order_contracts),
                repr(float(self.max_gross_exposure_usd)),
                repr(float(self.max_gross_leverage)),
                repr(float(self.max_net_leverage)),
                repr(float(self.max_margin_utilization_pct)),
                self.missing_margin_policy,
                repr(float(self.max_daily_loss_usd)),
                repr(float(self.max_drawdown_pct)),
                repr(float(self.max_drawdown_usd)),
                self.stale_mark_policy,
                repr(float(self.starting_capital_usd)),
                str(self.mark_staleness_tolerance_ns),
                str(self.day_boundary_offset_ns),
            ]
        )
        return header, data

    def write_csv(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        header, data = self.csv_rows()
        p.write_text(header + "\n" + data + "\n", encoding="utf-8")
        return p
