"""News Alpha Phase G -- the EXECUTION HANDOFF: a PortfolioPlan in the form the
existing C++ execution boundary consumes.

    PortfolioPlan
      -> PortfolioExecutionHandoff
           rows         integer target units per ROOT at its decision bar
                        (the ScheduledTarget intent the engine replays --
                        no price, no raw contract, no fill)
           risk_policy  a PaperRiskPolicy (the typed mirror of quant::RiskConfig)
                        built from the SAME constraints the plan was sized under
      -> quant_portfolio_plan_replay_csv
           BacktestEngine + PortfolioRiskManager, unchanged (Phase 21 wiring)

The engine's hard risk gate re-checks the plan independently: it resolves
the real contract, fills at the NEXT bar and can RESIZE or REJECT an order
the plan asked for. Positions and exposure after replay are the C++
PortfolioAccountant's, never recomputed here.

UNIT CAPS. `quant::RiskConfig`'s per-symbol / per-root / gross unit caps
count a futures contract and an ETF share identically -- a 5-unit cap is
five contracts or five shares. They are therefore set to the plan's own
largest position (non-binding by construction); the plan's real limits are
USD-based and carried as gross / net leverage, which mean the same thing
for every asset class.

WINDOWS. A plan is built as of the last discovery day (2022); its entry
executes on the next bar, inside the 2023-2024 validation window. Replaying
real plans is Phase H's job -- here the handoff is written, and the replay
is proven on synthetic bars.
"""
from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from alpha_agent.paper.risk_policy import PaperRiskPolicy
from alpha_agent.portfolio.allocator import find_allocator_cli
from alpha_agent.portfolio.plan import PlanStatus, PortfolioPlan
from alpha_agent.portfolio.policy import PortfolioConstraints

__all__ = [
    "PLAN_TARGET_COLUMNS",
    "PlanTargetRow",
    "PortfolioExecutionHandoff",
    "ReplayCommission",
    "ReplayDecision",
    "ReplayPosition",
    "ReplayResult",
    "build_handoff",
    "hard_gate_policy",
    "replay_handoff",
    "run_replay_cli",
]

PLAN_TARGET_COLUMNS = ("ts_event_ns", "root_symbol", "target_units", "portfolio_plan_fingerprint")
REPLAY_CLI_NAME = "quant_portfolio_plan_replay_csv"
REPLAY_CLI_ENV = "QUANT_PORTFOLIO_PLAN_REPLAY_CLI"
#: Multi-root books are marked at different instants (a CME minute vs an ETF
#: daily bar); a mark this old is still fresh for the stale-mark gate.
MARK_STALENESS_TOLERANCE_NS = 5 * 86_400_000_000_000


class PlanTargetRow(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    ts_event_ns: int
    root_symbol: str
    target_units: int
    instrument_key: str
    traded_symbol: str


class PortfolioExecutionHandoff(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "portfolio-handoff/1"
    plan_fingerprint: str
    rows: tuple[PlanTargetRow, ...]
    risk_policy: PaperRiskPolicy
    note: str = (
        "Target INTENT per root at its decision bar. The engine resolves the real contract, fills at the next bar "
        "and gates every order through the hard risk manager -- the plan's units are a request, not a fill."
    )

    def targets_csv(self) -> str:
        lines = [",".join(PLAN_TARGET_COLUMNS)]
        lines += [f"{r.ts_event_ns},{r.root_symbol},{r.target_units},{self.plan_fingerprint}" for r in self.rows]
        return "\n".join(lines) + "\n"

    def write(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        targets = directory / "plan_targets.csv"
        targets.write_text(self.targets_csv(), encoding="utf-8")
        return targets, self.risk_policy.write_csv(directory / "risk_config.csv")


def build_handoff(plan: PortfolioPlan) -> PortfolioExecutionHandoff:
    if plan.allocation is None or plan.status not in (PlanStatus.CONSTRUCTED, PlanStatus.NO_EXECUTABLE_POSITION):
        raise ValueError(f"a {plan.status.value} plan has no executable targets")
    risk = plan.risk_model
    assert risk is not None
    rows = tuple(
        PlanTargetRow(ts_event_ns=i.decision_ts_ns, root_symbol=i.root_symbol, target_units=i.units,
                      instrument_key=i.instrument_key, traded_symbol=risk.instrument(i.instrument_key).traded_symbol)
        for i in plan.allocation.instruments if i.admitted and (i.target_weight != 0.0 or i.previous_units != 0)
    )
    return PortfolioExecutionHandoff(plan_fingerprint=plan.fingerprint(), rows=rows,
                                     risk_policy=hard_gate_policy(plan.constraints, rows))


class _TargetRow(Protocol):
    @property
    def instrument_key(self) -> str: ...

    @property
    def target_units(self) -> int: ...


def hard_gate_policy(constraints: PortfolioConstraints, rows: Sequence[_TargetRow]) -> PaperRiskPolicy:
    """The engine's hard gate for target ``rows`` sized under ``constraints``
    (one plan's handoff, or a whole Phase H rebalance schedule). Unit caps
    count a contract and a share alike, so they are made non-binding by
    construction: per symbol the largest request, gross the sum of every
    instrument's largest request -- an upper bound on the book at any
    instant, including mid-rebalance. The real limits are the USD gross / net
    leverage and the drawdown kill switch."""
    largest_by_key: dict[str, int] = {}
    for r in rows:
        largest_by_key[r.instrument_key] = max(largest_by_key.get(r.instrument_key, 0), abs(r.target_units))
    largest = max(largest_by_key.values(), default=0)
    c = constraints
    return PaperRiskPolicy(
        max_contracts_per_symbol=max(1, largest), max_contracts_per_root=max(1, largest),
        max_gross_contracts=max(1, sum(largest_by_key.values())), max_order_contracts=0,
        max_gross_exposure_usd=0.0, max_gross_leverage=c.max_gross_leverage,
        max_net_leverage=c.max_net_exposure or 0.0, max_margin_utilization_pct=0.0,
        missing_margin_policy="treat_as_zero", max_daily_loss_usd=0.0, max_drawdown_pct=c.max_drawdown_pct,
        max_drawdown_usd=0.0, stale_mark_policy="reject_risk_increasing", starting_capital_usd=c.capital_usd,
        mark_staleness_tolerance_ns=MARK_STALENESS_TOLERANCE_NS, day_boundary_offset_ns=0,
    )


# ---------------------------------------------------------------------------
# replay through the unchanged engine
# ---------------------------------------------------------------------------


class ReplayDecision(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    ts_decision_ns: int
    verdict: str
    approved_quantity: int
    reason_code: str


class ReplayPosition(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    instrument_id: int
    raw_symbol: str
    root_symbol: str
    units: int
    avg_entry_price: float
    multiplier: float
    mark_price: float
    gross_notional_usd: float
    signed_notional_usd: float


class ReplayCommission(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str
    commission_per_unit_usd: float


class ReplayResult(BaseModel):
    """`quant_portfolio_plan_replay_csv`'s JSON, verbatim."""

    model_config = {"frozen": True, "extra": "forbid"}

    portfolio_plan_fingerprint: str
    target_rows: int
    target_rows_applied: int
    bars: int
    signals: int
    orders: int
    fills: int
    non_fills: int
    risk_rejects: int
    risk_resizes: int
    costs_usd: float
    starting_capital_usd: float
    equity_usd: float
    gross_exposure_usd: float
    net_exposure_usd: float
    gross_leverage: float
    net_leverage: float
    margin_complete: bool
    valuation_complete: bool
    risk_decisions: tuple[ReplayDecision, ...]
    positions: tuple[ReplayPosition, ...]
    #: Phase H acceptance patch: the per-root schedule the engine charged
    #: (empty = the scalar commission_per_contract_usd for every root).
    commission_schedule: tuple[ReplayCommission, ...] = ()
    # Phase H (additive): the reference CLI's PnL / roll / cost fields and its
    # daily_equity trace, from the same BacktestResult.
    end_of_test: str
    commission_per_contract_usd: float
    slippage_ticks: float
    spread_ticks: float
    trades: int
    gross_pnl_usd: float
    net_pnl_usd: float
    unrealized_pnl_usd: float
    net_equity_usd_at_end: float
    max_drawdown_usd: float
    rolls: int
    rolls_priced_contemporaneous: int
    rolls_priced_auxiliary_marks: int
    rolls_priced_stale: int
    rolls_deferred: int
    eot_liquidations: int
    daily_equity_basis: str
    daily_equity: tuple[tuple[int, int, float, float, float, float, int, int], ...]


def run_replay_cli(
    targets_csv: Path,
    risk_config_csv: Path,
    bars_csv: Path,
    contracts_csv: Path,
    *,
    cli: Path | None = None,
    commission_per_contract_usd: float = 0.0,
    slippage_ticks: float = 0.0,
    spread_ticks: float = 0.0,
    validation_days_csv: Path | None = None,
    roll_close_marks_csv: Path | None = None,
    splits_csv: Path | None = None,
    force_liquidate_at_end: bool = False,
    commission_schedule: Mapping[str, float] | None = None,
    timeout_s: float = 120.0,
) -> tuple[ReplayResult, dict]:
    """One ``quant_portfolio_plan_replay_csv`` run: the parsed result and the
    raw JSON (the validation plane reads its ``daily_equity`` trace).
    ``commission_schedule`` (root -> USD per unit) is written for the CLI's
    ``--commission-schedule``; the engine charges it, and refuses a schedule
    that misses a traded root. ``None`` = the scalar for every root."""
    import json

    binary = cli or find_allocator_cli(REPLAY_CLI_NAME, REPLAY_CLI_ENV)
    if binary is None or not Path(binary).exists():
        raise FileNotFoundError(f"{REPLAY_CLI_NAME} is not built")
    optional = [str(p) if p is not None else "" for p in (validation_days_csv, roll_close_marks_csv, splits_csv)]
    while optional and not optional[-1]:
        optional.pop()
    args = [str(binary), str(bars_csv), str(contracts_csv), str(targets_csv), str(risk_config_csv),
            repr(commission_per_contract_usd), repr(slippage_ticks), repr(spread_ticks), *optional]
    if force_liquidate_at_end:
        args.append("--end-of-test=force_liquidate")
    if commission_schedule is not None:
        schedule_csv = Path(targets_csv).parent / "commission_schedule.csv"
        schedule_csv.write_text("root_symbol,commission_per_unit_usd\n" + "".join(
            f"{root},{rate!r}\n" for root, rate in sorted(commission_schedule.items())), encoding="utf-8")
        args.append(f"--commission-schedule={schedule_csv}")
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"{REPLAY_CLI_NAME} exited {proc.returncode}")
    return ReplayResult.model_validate_json(proc.stdout), json.loads(proc.stdout)


def replay_handoff(
    handoff: PortfolioExecutionHandoff,
    bars_csv: Path,
    contracts_csv: Path,
    workdir: Path,
    *,
    cli: Path | None = None,
    commission_per_contract_usd: float = 0.0,
    slippage_ticks: float = 0.0,
    spread_ticks: float = 0.0,
    commission_schedule: Mapping[str, float] | None = None,
    timeout_s: float = 120.0,
) -> ReplayResult:
    targets, risk = handoff.write(workdir)
    result, _ = run_replay_cli(targets, risk, bars_csv, contracts_csv, cli=cli,
                               commission_per_contract_usd=commission_per_contract_usd,
                               slippage_ticks=slippage_ticks, spread_ticks=spread_ticks,
                               commission_schedule=commission_schedule, timeout_s=timeout_s)
    return result
