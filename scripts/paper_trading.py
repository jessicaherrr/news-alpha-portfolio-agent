"""Phase 21 -- paper trading / drift monitor CLI.

    python scripts/phase_21_paper_trading.py list-eligible
    python scripts/phase_21_paper_trading.py start --experiment <experiment_id_or_identity>
    python scripts/phase_21_paper_trading.py step --run <run_id> [--days N]
    python scripts/phase_21_paper_trading.py stop --run <run_id>
    python scripts/phase_21_paper_trading.py status --run <run_id>
    python scripts/phase_21_paper_trading.py list-runs

Every command is OFFLINE and STRICTLY historical replay: the market-data
window is ``[2018-01-01, 2025-01-01)`` (the already-acquired, already-paid-for
Phase 13.5C research + validation data -- ``alpha_agent.data.real_market_dataset
.RESEARCH_WINDOW`` / ``VALIDATION_WINDOW``), never the locked 2025 holdout, and
never a live broker or a new paid data request. ``start`` refuses any
experiment that has not passed frozen validation (CLAUDE.md) or whose
StrategySpec this MVP cannot deterministically rebuild and fingerprint-verify
(see ``alpha_agent.paper.eligibility``).

This is deterministic REPLAY-based paper trading, not a connection to a live
market feed: "paper trading" here means continuing to run the exact same
frozen research pipeline forward through already-owned historical data via
the hard-risk-gated C++ engine, producing a persistent ledger + drift
diagnostics + alerts -- the plumbing a future live-data phase would plug a
real feed into without changing anything downstream of
``MarketWindowProvider``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from alpha_agent.data.real_market_dataset import HOLDOUT_START_NS, RESEARCH_WINDOW
from alpha_agent.paper.eligibility import (
    PaperTradingEligibilityError,
    assert_paper_trading_eligible,
)
from alpha_agent.paper.engine import DEFAULT_PAPER_TRADING_CLI, PaperTradingEngine
from alpha_agent.paper.errors import PaperTradingError
from alpha_agent.paper.ledger import DEFAULT_LEDGER_PATH, PaperLedger
from alpha_agent.paper.report import build_paper_run_report
from alpha_agent.paper.risk_policy import PaperRiskPolicy
from alpha_agent.paper.schedule_window import RealMarketWindowProvider
from alpha_agent.registry.sqlite_registry import DEFAULT_REGISTRY_PATH, ExperimentRegistry

REPO = Path(__file__).resolve().parents[1]
WINDOW_START_NS = int(datetime.fromisoformat(RESEARCH_WINDOW[0]).replace(tzinfo=UTC).timestamp() * 1e9)
WINDOW_END_NS = HOLDOUT_START_NS  # 2025-01-01, exclusive -- never crossed


def cmd_list_eligible(args: argparse.Namespace) -> int:
    with ExperimentRegistry(REPO / args.registry) as reg:
        n_checked = 0
        eligible = []
        for view in reg.experiments(authoritative_only=True):
            n_checked += 1
            try:
                elig = assert_paper_trading_eligible(reg, view.experiment_identity)
            except PaperTradingEligibilityError:
                continue
            eligible.append(elig)
    print(f"checked {n_checked} authoritative experiment(s); {len(eligible)} eligible for paper trading\n")
    for e in eligible:
        print(f"  {e.experiment_id:45s} {e.root_symbol:4s} {e.strategy_family:16s} {e.params}")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    with ExperimentRegistry(REPO / args.registry) as reg, PaperLedger(REPO / args.ledger) as ledger:
        provider = RealMarketWindowProvider()
        engine = PaperTradingEngine(
            registry=reg, ledger=ledger, provider=provider,
            executable=REPO / args.cli, work_dir=(REPO / args.work_dir) if args.work_dir else None,
        )
        risk_policy = PaperRiskPolicy(
            max_drawdown_pct=args.max_drawdown_pct,
            max_daily_loss_usd=args.max_daily_loss_usd,
            starting_capital_usd=args.capital,
        )
        try:
            run_id = engine.start_run(
                experiment_key=args.experiment,
                window_start_ns=WINDOW_START_NS,
                window_end_ns=WINDOW_END_NS,
                risk_policy=risk_policy,
                commission_per_contract_usd=args.commission,
            )
        except PaperTradingError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 1
    print(f"started paper run {run_id}")
    return 0


def cmd_step(args: argparse.Namespace) -> int:
    with ExperimentRegistry(REPO / args.registry) as reg, PaperLedger(REPO / args.ledger) as ledger:
        provider = RealMarketWindowProvider()
        engine = PaperTradingEngine(
            registry=reg, ledger=ledger, provider=provider,
            executable=REPO / args.cli, work_dir=(REPO / args.work_dir) if args.work_dir else None,
        )
        try:
            snap = engine.step(args.run, n_trading_days=args.days)
        except PaperTradingError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 1
    print(json.dumps({
        "run_id": snap.run_id, "step_ordinal": snap.step_ordinal, "as_of_ts_ns": snap.as_of_ts_ns,
        "status": snap.status, "kill_switch_active": snap.kill_switch_active,
        "drift_flags": list(snap.drift.flags),
        "comparison_availability": snap.drift.comparison_availability,
        "n_alerts": len(snap.alerts),
        "provenance_identity": snap.provenance.identity(),
    }, indent=2))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    with ExperimentRegistry(REPO / args.registry) as reg, PaperLedger(REPO / args.ledger) as ledger:
        provider = RealMarketWindowProvider()
        engine = PaperTradingEngine(
            registry=reg, ledger=ledger, provider=provider,
            executable=REPO / args.cli, work_dir=(REPO / args.work_dir) if args.work_dir else None,
        )
        try:
            snap = engine.stop_run(args.run)
        except PaperTradingError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 1
    print(f"stopped paper run {snap.run_id} (status={snap.status})")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    with PaperLedger(REPO / args.ledger) as ledger:
        report = build_paper_run_report(ledger, args.run)
    if args.out:
        out_path = REPO / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"written -> {out_path}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    with PaperLedger(REPO / args.ledger) as ledger:
        for run in ledger.list_runs():
            print(f"{run.run_id:28s} {run.status:8s} {run.root_symbol:4s} "
                  f"{run.strategy_family:16s} {run.experiment_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    ap.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    ap.add_argument("--cli", default=str(DEFAULT_PAPER_TRADING_CLI),
                    help="path to the compiled quant_paper_trading_targets_csv")
    ap.add_argument("--work-dir", default=None, help="persist per-step CSV artifacts here (debug)")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list-eligible")

    p_start = sub.add_parser("start")
    p_start.add_argument("--experiment", required=True)
    p_start.add_argument("--capital", type=float, default=100_000.0)
    p_start.add_argument("--max-drawdown-pct", type=float, default=0.10)
    p_start.add_argument("--max-daily-loss-usd", type=float, default=1500.0)
    p_start.add_argument("--commission", type=float, default=2.0)

    p_step = sub.add_parser("step")
    p_step.add_argument("--run", required=True)
    p_step.add_argument("--days", type=int, default=1)

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("--run", required=True)

    p_status = sub.add_parser("status")
    p_status.add_argument("--run", required=True)
    p_status.add_argument("--out", default=None)

    sub.add_parser("list-runs")

    args = ap.parse_args(argv)
    return {
        "list-eligible": cmd_list_eligible,
        "start": cmd_start,
        "step": cmd_step,
        "stop": cmd_stop,
        "status": cmd_status,
        "list-runs": cmd_list_runs,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
