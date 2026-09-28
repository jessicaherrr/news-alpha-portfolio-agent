"""News Alpha Phase H -- the real news-to-portfolio research loop, closed end to end.

    ResearchMandate -> News/Event -> Impact Scan -> Mechanism Graph -> Signal Path
      -> Asset Expression -> Measurement -> Candidate Signal -> Factor Diagnostics
      -> Multi-asset Ranking -> Portfolio Plan                  (Phases A-G, reused)
      -> VALIDATION GATE -> C++ backtest -> validation plane    (Phase H)
      -> registry: experiments + signal-path evidence (schema v7) -> Alpha Memory

Reuses the Phase G slice exactly (three events, the same mandate, the cached
Phase E screens). The QUALIFIED plan is the one the loop acts on; the gate
decides whether it may spend the 2023-2024 validation window. The exploratory
preview is gated too, and never enters validation. Whatever the outcome --
validated, rejected, insufficient evidence or no eligible portfolio -- every
hypothesis of the run is recorded as typed evidence (``--no-record`` to skip).

``--engineering-replay`` (on by default) additionally replays an ETF-only
exploratory book through the real C++ engine over 2022 -- DISCOVERY data only,
the days its signals were screened on. It is an execution-integrity check on
real bytes (fills, costs, roll/split handling, the hard risk gate, the daily
equity trace), explicitly NOT evidence: no verdict, no registry write.

No download, no vendor call, no cost; the 2025 holdout is never loaded.

    PYTHONPATH=python python scripts/news_alpha_phase_h_validation.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from alpha_agent.alpha_memory.signal_path_evidence import EventResearch
from alpha_agent.alpha_memory.signal_path_memory import (
    GUIDANCE_TEXT,
    signal_path_memory,
)
from alpha_agent.data.real_market_dataset import _iso_ns
from alpha_agent.news_alpha import (
    MandateDomain,
    ResearchMandate,
    UserDescribedEvent,
    build_asset_expressions,
    build_candidate_signals,
    build_economic_mechanism_graph,
    discover_signal_paths,
    probe_domain_capabilities,
    resolve_allowed_universe,
    scan_initial_impact,
)
from alpha_agent.portfolio import (
    EligibilityMode,
    PortfolioConstructionPolicy,
    construct_portfolio_plan,
)
from alpha_agent.portfolio.allocator import find_allocator_cli
from alpha_agent.portfolio.execution import (
    cost_plan_for,
    execution_support,
    load_execution_inputs,
    run_schedule,
)
from alpha_agent.portfolio.research_loop import close_research_loop
from alpha_agent.portfolio.risk_model import MarketSnapshotStore
from alpha_agent.portfolio.strategy import (
    build_rebalance_schedule,
    freeze_portfolio_strategy,
)
from alpha_agent.portfolio.validation import assess_validation_eligibility
from alpha_agent.recommendation.signal_ranking import rank_candidate_signals
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.screening.candidate_signal_screen import FactorScreenStore
from news_alpha_phase_g_portfolio import (
    DESCRIBED_AT,
    EVENTS,
    MANDATE,
)

OUT_DIR = REPO / "outputs" / "news_alpha" / "phase_h"


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _events(mandate: ResearchMandate) -> list[EventResearch]:
    universe = resolve_allowed_universe(mandate, capabilities=probe_domain_capabilities())
    out = []
    for _name, text in EVENTS:
        scan = scan_initial_impact(UserDescribedEvent.create(text, described_at=DESCRIBED_AT), mandate,
                                   universe=universe)
        paths = discover_signal_paths(build_economic_mechanism_graph(scan))
        expressions = build_asset_expressions(paths, universe, mandate)
        out.append(EventResearch(discovery=paths, expressions=expressions,
                                 candidates=build_candidate_signals(expressions, paths)))
    return out


def _screens(events: list[EventResearch]) -> dict:
    screens: dict = {}
    for e in events:
        screens.update(FactorScreenStore().load_many(e.candidates))
    return screens


def _plan(events, screens, mandate, mode):
    ranked = rank_candidate_signals(tuple(e.candidates for e in events), screens, mandate)
    plan = construct_portfolio_plan(ranked, screens, mandate, snapshots=MarketSnapshotStore().provider(),
                                    policy=PortfolioConstructionPolicy(eligibility=mode))
    return ranked, plan


def _engineering_replay(workdir: Path) -> dict:
    """ETF-only exploratory book, re-sized every 20 days over 2022, replayed through the
    real engine -- discovery data only, never evidence."""
    mandate = ResearchMandate(allowed_domains=(MandateDomain.ETF,))
    events = _events(mandate)
    screens = _screens(events)
    ranked, plan = _plan(events, screens, mandate, EligibilityMode.EXPLORATORY)
    strategy = freeze_portfolio_strategy(plan, ranked)
    assert execution_support(strategy) is None
    schedule = build_rebalance_schedule(strategy, ranked, mandate, factor_sources=screens,
                                        snapshots=MarketSnapshotStore().provider(), start=date(2022, 1, 1),
                                        end_exclusive=date(2023, 1, 1))
    inputs = load_execution_inputs(strategy, end_ns=_iso_ns("2023-01-01"))
    costs = cost_plan_for(strategy)
    baseline = next(sc for sc in costs.scenarios if sc.label == costs.baseline_label())
    run = run_schedule(schedule, strategy, inputs, start_ns=_iso_ns("2022-01-01"), end_ns=_iso_ns("2023-01-01"),
                       commission_schedule=costs.schedule_for(baseline), workdir=workdir,
                       label="engineering_replay_2022")
    r = run.run
    return {
        "label": "ENGINEERING REPLAY -- discovery window (2022) only, the days these signals were screened on; an "
                 "execution-integrity check on real bytes, NOT evidence (no verdict, no registry write)",
        "strategy_fingerprint": strategy.fingerprint(), "eligibility": plan.eligibility.value,
        "instruments": list(strategy.instrument_keys), "rebalances": len(schedule.decisions),
        "rebalance_status_counts": schedule.status_counts(), "schedule_hash": schedule.schedule_hash(),
        "commission_schedule": [c.model_dump(mode="json") for c in costs.commission_schedule],
        "commission_charged": run.commission_schedule, "trading_days": r.daily.n_days,
        "fills": r.n_fills, "closed_trades": r.n_trades, "costs_usd": round(r.costs_usd, 2),
        "net_pnl_usd": round(r.net_pnl_usd, 2), "risk_rejects": run.risk_rejects, "risk_resizes": run.risk_resizes,
        "target_rows": run.target_rows, "target_rows_applied": run.target_rows_applied,
        "equity_identity_holds": abs(r.daily.equity_usd[-1] - (r.daily.starting_capital_usd + r.net_pnl_usd)) < 1e-6,
        "inputs_fingerprint": run.inputs_fingerprint, "provenance": list(inputs.provenance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--registry", type=Path, default=None, help="default: data/registry/experiments.sqlite")
    parser.add_argument("--no-record", action="store_true", help="compute everything, write nothing to the registry")
    parser.add_argument("--no-engineering-replay", action="store_true")
    args = parser.parse_args()
    if find_allocator_cli() is None:
        raise SystemExit("build the C++ core first: cmake -S . -B build/cpp && cmake --build build/cpp")

    started = datetime.now(UTC)
    events = _events(MANDATE)
    screens = _screens(events)
    print(f"{sum(len(e.candidates.candidates) for e in events)} candidates, {len(screens)} cached screens")

    ranked, qualified = _plan(events, screens, MANDATE, EligibilityMode.QUALIFIED)
    _, exploratory = _plan(events, screens, MANDATE, EligibilityMode.EXPLORATORY)
    print(f"\nQUALIFIED PLAN   {qualified.headline()}")
    print(f"EXPLORATORY PLAN {exploratory.headline()}")

    registry = ExperimentRegistry(args.registry) if args.registry else ExperimentRegistry()
    with tempfile.TemporaryDirectory() as tmp, registry:
        outcome, evidence = close_research_loop(
            events, ranked, screens, qualified, MANDATE, workdir=Path(tmp), registry=registry,
            record=not args.no_record, code_commit=_git("rev-parse", "HEAD") or "",
        )
        exploratory_gate = assess_validation_eligibility(exploratory, ranked, screens)
        in_registry = len(registry.signal_path_evidence(research_run_id=outcome.research_run_id))
        memory = [m for e in events for m in signal_path_memory(registry, event_id=e.candidates.event_id)]
        replay = None if args.no_engineering_replay else _engineering_replay(Path(tmp) / "replay")

    v = outcome.validation
    print(f"\nPHASE H (qualified): {v.status.value} -- {outcome.headline()}")
    print(f"PHASE H (exploratory preview): eligible={exploratory_gate.eligible} "
          f"{[f.reason.value for f in exploratory_gate.findings]}")
    print(f"holdout: {v.holdout.status} (lifecycle {v.holdout.lifecycle_state}, accessed={v.holdout.accessed})")
    print(f"evidence: {len(evidence)} records  {outcome.evidence_by_outcome}  scopes {outcome.evidence_by_scope}")
    print(f"          reasons {outcome.evidence_by_reason}")
    print(f"recorded: {outcome.recorded_evidence} new evidence row(s), {outcome.recorded_experiments} experiment(s); "
          f"{in_registry} row(s) of this run in the registry")
    for s in outcome.studies:
        print(f"study {s.design.question.value:<24} {s.status.value}  blockers: "
              + ", ".join(b.requirement.value for b in s.blockers))
    seen = {m.path_signature: m for m in memory}
    print(f"memory: {len(seen)} route(s) for these events")
    for m in list(seen.values())[:6]:
        print(f"  {m.headline()}  {m.reasons}  -> {[g.value for g in m.guidance]}")
    if replay:
        print(f"\nENGINEERING REPLAY (2022, not evidence): {replay['fills']} fills, {replay['trading_days']} days, "
              f"costs ${replay['costs_usd']:,}, net ${replay['net_pnl_usd']:,}, gate rejects/resizes "
              f"{replay['risk_rejects']}/{replay['risk_resizes']}, equity identity {replay['equity_identity_holds']}")

    wall = round((datetime.now(UTC) - started).total_seconds(), 1)
    artifact = {
        "artifact": "news-alpha-phase-h-validation-and-memory/1",
        "plane": "VALIDATION + MEMORY -- a portfolio result would apply to that portfolio only",
        "generated_at": datetime.now(UTC).isoformat(),
        "code_commit": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "events": [{"name": n, "text": t, "event_id": e.candidates.event_id} for (n, t), e in zip(EVENTS, events,
                                                                                                  strict=True)],
        "mandate": MANDATE.summary(),
        "data_cost_usd": 0.0,
        "windows": {"read_by_the_gated_run": list(v.data_read) or ["none -- the gate stopped before any data"],
                    "engineering_replay": "2022 discovery bars only" if replay else None,
                    "untouched": ["2025 locked holdout"]},
        "qualified_plan": {"fingerprint": qualified.fingerprint(), "status": qualified.status.value,
                           "headline": qualified.headline()},
        "exploratory_plan": {"fingerprint": exploratory.fingerprint(), "status": exploratory.status.value,
                             "headline": exploratory.headline(), "validation_gate": json.loads(
                                 exploratory_gate.model_dump_json())},
        "research_loop": json.loads(outcome.model_dump_json(exclude={"validation": {"report", "strategy",
                                                                                    "validation_spec"}})),
        "validation_report_fingerprint": v.report.report_fingerprint() if v.report else None,
        "registry": {"evidence_rows_of_this_run": in_registry, "research_run_id": outcome.research_run_id,
                     "table": "signal_path_evidence (schema v7)"},
        "memory": [json.loads(m.model_dump_json()) for m in seen.values()],
        "guidance_text": {g.value: t for g, t in GUIDANCE_TEXT.items()},
        "evidence_sample": [json.loads(r.model_dump_json()) for r in evidence[:5]],
        "engineering_replay": replay,
        "wall_seconds": wall,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "THREE_EVENTS__validation_and_memory.json"
    text = json.dumps(artifact, indent=1, ensure_ascii=False, default=str) + "\n"
    out.write_text(text, encoding="utf-8")
    shown = out.relative_to(REPO) if out.is_relative_to(REPO) else out
    print(f"\nWROTE {shown} ({len(text) // 1024} KB, sha256 "
          f"{hashlib.sha256(text.encode()).hexdigest()[:16]}…, {wall}s)")


if __name__ == "__main__":
    main()
