"""News Alpha Phase E -- the real end-to-end vertical slice.

    News (AI infrastructure capex) -> Mechanism Graph -> Signal Paths -> Economic Consequence
        -> Asset Expression -> Measurement Spec -> PIT Data Field Resolution
        -> Candidate Signal Spec -> Factor Expression -> Factor Series -> Factor Diagnostics

Every candidate is screened on REAL, already-acquired bars (GLBX.MDP3 futures
rebuilt offline from the raw store -- about a minute per root -- and the ETF
primary-listing daily bars), over the 2018-2022 discovery window only. No
download, no vendor call, no cost, no 2023-2024 validation data, no 2025
holdout, no registry write. The result is SCREENING evidence -- never a
validation or verdict.

Writes a provenance artifact (readable name; hashes inside) and, unless
``--no-store``, the operational screen cache the Agent page reads.

    PYTHONPATH=python python scripts/news_alpha_phase_e_vertical_slice.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

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
from alpha_agent.screening.candidate_signal_screen import FactorScreenStore, screen_candidates

EVENT = "Hyperscalers announce a major AI infrastructure spending increase, lifting data center capex guidance"
DESCRIBED_AT = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
OUT = Path(__file__).resolve().parents[1] / "outputs" / "news_alpha" / "phase_e" / (
    "AI_INFRASTRUCTURE__candidate_signal_screens.json"
)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-store", action="store_true", help="do not write the Agent page's screen cache")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    mandate = ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES, MandateDomain.ETF))
    universe = resolve_allowed_universe(mandate, capabilities=probe_domain_capabilities())
    scan = scan_initial_impact(UserDescribedEvent.create(EVENT, described_at=DESCRIBED_AT), mandate, universe=universe)
    graph = build_economic_mechanism_graph(scan)
    paths = discover_signal_paths(graph)
    plan = build_asset_expressions(paths, universe, mandate)
    cset = build_candidate_signals(plan, paths)
    print(f"EVENT       {EVENT}")
    print(f"PIPELINE    graph {graph.fingerprint()[:22]}… · {len(paths.paths)} paths · {len(plan.expressions)} "
          f"expressions · {len(plan.usable_measurements())} usable measurements · {len(cset.candidates)} candidates")

    started = datetime.now(UTC)
    screens, run = screen_candidates(
        cset, store=None if args.no_store else FactorScreenStore(),
        progress=lambda c: print(f"  screening {c.spec.instrument:<4} {c.expression} → {c.spec.prediction_horizon.value}",
                                 flush=True),
    )
    by_id = {s.candidate_signal_id: s for s in screens}
    print()
    print(f"{'candidate':<34} {'TS-Spear':>8} {'t(n/h)':>7} {'TS-Pear':>7} {'yrs+':>5} {'cover':>6} {'flips/yr':>9} "
          f"{'timing bp':>10} {'b/e bp':>7}  screen")
    for c in cset.candidates:
        d = by_id[c.candidate_signal_id].diagnostics
        dec = d.declared
        agree = sum(p.evaluable and p.ts_spearman_ic is not None and (p.ts_spearman_ic > 0) == (d.expected_sign > 0)
                    for p in d.subperiods)
        be = f"{d.cost.break_even_cost_bps:.1f}" if d.cost.break_even_cost_bps is not None else "--"
        print(f"{c.spec.instrument + ' ' + c.expression + ' → ' + c.spec.prediction_horizon.value:<34} {dec.ts_spearman_ic:>+8.3f} "
              f"{dec.ts_spearman_t:>+7.2f} {dec.ts_pearson_ic:>+7.3f} {agree:>3}/{sum(p.evaluable for p in d.subperiods)} "
              f"{d.coverage.coverage:>6.1%} {d.turnover.sign_flips_per_year:>9.1f} {d.cost.timing_edge_bps:>+10.1f} "
              f"{be:>7}  {d.status.value}")
    for c in cset.candidates:
        d = by_id[c.candidate_signal_id].diagnostics
        decay = "  ".join(f"{h.horizon_days}D {h.ts_spearman_ic:+.3f} (t {h.ts_spearman_t:+.2f})" for h in d.horizons)
        print(f"  decay {c.spec.instrument:<4} {c.expression:<22} {decay}")
    top = sorted((p for p in run.correlations if p.correlation is not None), key=lambda p: -abs(p.correlation))[:5]
    for p in top:
        print(f"  corr  {p.a:<36} ~ {p.b:<36} {p.correlation:+.2f} on {p.n_common} days")

    artifact = {
        "artifact": "news-alpha-phase-e-vertical-slice/1",
        "plane": "SCREENING -- not validation, not a verdict",
        "diagnostic_scope": "UNCONDITIONAL_FACTOR -- not event-conditioned evidence",
        "ic_kind": "TIME_SERIES (one instrument vs its own forward return; not a cross-sectional Rank IC)",
        "generation_policy": "candidate-generation/1: (20 bars, 20D) and (60 bars, 60D) for every directional "
                             "measurement; no economic transmission lag is converted into a prediction horizon",
        "generated_at": datetime.now(UTC).isoformat(),
        "screening_wall_seconds": round((datetime.now(UTC) - started).total_seconds(), 1),
        "code_commit": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "event": {"text": EVENT, "event_id": cset.event_id, "described_at": DESCRIBED_AT.isoformat()},
        "mandate_fingerprint": mandate.fingerprint(),
        "fingerprints": {
            "mechanism_graph": graph.fingerprint(), "signal_paths": paths.fingerprint(),
            "asset_expressions": plan.fingerprint(), "candidate_signals": cset.fingerprint(),
        },
        "data_cost_usd": 0.0,
        "data_note": "already-acquired raw store only; no vendor call, no download",
        "windows": {"screened": "2018-01-01 -> 2023-01-01 (exclusive)", "untouched": ["2023-2024 validation",
                                                                                     "2025 locked holdout"]},
        "run": run.model_dump(mode="json"),
        "candidates": [
            {
                "candidate": c.model_dump(mode="json"),
                "screen": {
                    "fingerprint": by_id[c.candidate_signal_id].fingerprint(),
                    "series": by_id[c.candidate_signal_id].series.model_dump(mode="json", exclude={"days", "values"}),
                    "diagnostics": by_id[c.candidate_signal_id].diagnostics.model_dump(mode="json"),
                },
            }
            for c in cset.candidates
        ],
        "refusals": [r.model_dump(mode="json") for r in cset.refusals],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWROTE       {args.out}  ({'' if args.no_store else 'screens cached for the Agent page; '}"
          f"{artifact['screening_wall_seconds']}s)")


if __name__ == "__main__":
    main()
