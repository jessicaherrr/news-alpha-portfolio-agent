"""News Alpha Phase G -- the real multi-asset portfolio construction, end to end.

    News (3 events) -> ... -> Candidate Signals -> Factor Diagnostics      (Phases A-E)
        -> Multi-Asset Signal Ranking across the three events             (Phase F)
        -> PORTFOLIO CONSTRUCTION -> PortfolioPlan                         (Phase G)
        -> execution handoff files for the existing C++ engine

Reuses the Phase F slice (same events, same mandate, the cached Phase E
screens). Market snapshots are rebuilt offline from already-acquired bytes
(futures: about a minute per root, cached in data/news_alpha/market_snapshots/)
over the 2018-2022 discovery window only. Sizing is the compiled C++ allocator
(build it first). No download, no vendor call, no cost, no 2023-2024 validation
data, no 2025 holdout, no registry write -- and no replay: executing a plan
starts on the next bar, inside the validation window (Phase H).

Writes a provenance artifact (readable name; hashes inside) and the handoff
files of the exploratory plan.

    PYTHONPATH=python python scripts/news_alpha_phase_g_portfolio.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from alpha_agent.news_alpha import (
    CandidateSignalSet,
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
    PlanStatus,
    PortfolioConstructionPolicy,
    PortfolioPlan,
    construct_portfolio_plan,
)
from alpha_agent.portfolio.allocator import find_allocator_cli
from alpha_agent.portfolio.handoff import build_handoff
from alpha_agent.portfolio.risk_model import MarketSnapshotStore
from alpha_agent.recommendation.signal_ranking import rank_candidate_signals
from alpha_agent.screening.candidate_signal_screen import FactorScreenStore

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "outputs" / "news_alpha" / "phase_g"
EVENTS = (
    ("AI_INFRASTRUCTURE",
     "Hyperscalers announce a major AI infrastructure spending increase, lifting data center capex guidance"),
    ("OPEC_CUT", "OPEC+ agrees a production cut of 1 million barrels per day"),
    ("FED_SURPRISE", "Rate-policy surprise: the Fed unexpectedly raised rates by 50 basis points"),
)
DESCRIBED_AT = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
#: The Phase F slice's mandate (default profile: no shorting, no leverage cap, no capital stated).
MANDATE = ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES, MandateDomain.ETF))


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _print(plan: PortfolioPlan) -> None:
    print(f"  {plan.headline()}")
    a, risk = plan.allocation, plan.risk_model
    if a is None or risk is None:
        return
    for i in plan.positions:
        r = risk.instrument(i.instrument_key)
        print(f"  {r.symbol:<4} {r.traded_symbol:<6} {i.units:>+7} {r.unit}(s) @ {i.price:>10.2f}  "
              f"notional {i.executable_notional_usd:>+12,.0f}  vol {i.annual_vol:5.1%}  "
              f"risk {i.risk_contribution / a.executable.annual_vol:4.0%}  {i.sector}")
    for i in a.instruments:
        if i.units == 0:
            why = (i.reason.lower() if i.reason else "below one unit" if i.below_one_unit
                   else "its signals net to zero" if i.target_weight == 0 else "removed by a limit")
            print(f"  {i.instrument_key:<12} no position ({why})")
    for c in a.constraints:
        if c.status in ("BINDING", "OVERRIDDEN"):
            print(f"  limit {c.name:<26} {c.scope:<32} {c.stage:<10} {c.status:<10} {c.before:.4f} -> {c.after:.4f} "
                  f"(limit {c.limit:.4f})")
    reasons: dict[str, int] = {}
    for rej in plan.rejected:
        reasons[rej.reason.value] = reasons.get(rej.reason.value, 0) + 1
    print(f"  not in the portfolio: {reasons}")


def _summary(plan: PortfolioPlan) -> dict:
    a = plan.allocation
    return {
        "fingerprint": plan.fingerprint(), "status": plan.status.value, "headline": plan.headline(),
        "eligibility": plan.eligibility.value, "as_of": str(plan.as_of) if plan.as_of else None,
        "selected": len(plan.selected), "rejected_by_reason": {
            r: sum(x.reason.value == r for x in plan.rejected) for r in sorted({x.reason.value for x in plan.rejected})},
        "executable": a.executable.model_dump() if a else None,
        "target": a.target.model_dump() if a else None,
        "binding_constraints": [c.model_dump() for c in a.constraints if c.status in ("BINDING", "OVERRIDDEN")]
        if a else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if find_allocator_cli() is None:
        raise SystemExit("build the C++ core first: cmake -S . -B build/cpp && cmake --build build/cpp")

    mandate = MANDATE
    universe = resolve_allowed_universe(mandate, capabilities=probe_domain_capabilities())
    sets: dict[str, CandidateSignalSet] = {}
    for name, text in EVENTS:
        scan = scan_initial_impact(UserDescribedEvent.create(text, described_at=DESCRIBED_AT), mandate,
                                   universe=universe)
        paths = discover_signal_paths(build_economic_mechanism_graph(scan))
        sets[name] = build_candidate_signals(build_asset_expressions(paths, universe, mandate), paths)
    screens: dict = {}
    for cset in sets.values():
        screens.update(FactorScreenStore().load_many(cset))
    print(f"{sum(len(s.candidates) for s in sets.values())} candidates, {len(screens)} cached screens")

    store = MarketSnapshotStore()
    provider = store.provider()
    started = datetime.now(UTC)
    plans: dict[str, PortfolioPlan] = {}
    shorting = mandate.model_copy(update={"shorting_allowed": True})
    for name, m, mode in (("qualified", mandate, EligibilityMode.QUALIFIED),
                          ("exploratory", mandate, EligibilityMode.EXPLORATORY),
                          ("exploratory_shorting_allowed", shorting, EligibilityMode.EXPLORATORY)):
        ranked = rank_candidate_signals(tuple(sets.values()), screens, m)
        plans[name] = construct_portfolio_plan(ranked, screens, m, snapshots=provider,
                                               policy=PortfolioConstructionPolicy(eligibility=mode))
        print(f"\n{name.upper()}  (ranked {ranked.fingerprint()[:18]}…, {ranked.headline()})")
        _print(plans[name])
    wall = round((datetime.now(UTC) - started).total_seconds(), 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    handoffs = {}
    for name, plan in plans.items():
        if plan.status is PlanStatus.CONSTRUCTED:
            handoff = build_handoff(plan)
            targets, risk = handoff.write(args.out_dir / "handoff" / name)
            handoffs[name] = {"plan_targets_csv": str(targets.relative_to(REPO)),
                              "risk_config_csv": str(risk.relative_to(REPO)),
                              "rows": [r.model_dump() for r in handoff.rows],
                              "note": "not replayed: the next bar lies in the 2023-2024 validation window (Phase H)"}
    artifact = {
        "artifact": "news-alpha-phase-g-portfolio-construction/1",
        "plane": "PORTFOLIO CONSTRUCTION -- not validation, not a verdict, no expected return",
        "generated_at": datetime.now(UTC).isoformat(),
        "code_commit": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "events": [{"name": n, "text": t, "event_id": sets[n].event_id} for n, t in EVENTS],
        "data_cost_usd": 0.0,
        "data_note": "already-acquired raw store only; no vendor call, no download",
        "windows": {"read": "2018-01-01 -> 2023-01-01 (exclusive)",
                    "untouched": ["2023-2024 validation", "2025 locked holdout"]},
        "construction_wall_seconds": wall,
        "summaries": {n: _summary(p) for n, p in plans.items()},
        "plans": {n: json.loads(p.model_dump_json()) for n, p in plans.items()},
        "handoffs": handoffs,
    }
    out = args.out_dir / "THREE_EVENTS__portfolio_plan.json"
    out.write_text(json.dumps(artifact, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWROTE {out.relative_to(REPO)}  ({wall}s)")


if __name__ == "__main__":
    main()
