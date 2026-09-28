"""News Alpha Phase F -- the real multi-asset signal ranking, end to end.

    News (3 events) -> Mechanism Graph -> Signal Paths -> Asset Expression -> Measurement
        -> Candidate Signals -> Factor Series -> Factor Diagnostics          (Phases A-E)
        -> MULTI-ASSET SIGNAL RANKING across the three events               (Phase F)

Every candidate is screened on REAL, already-acquired bars (GLBX.MDP3 futures
rebuilt offline from the raw store -- about a minute per root -- and the ETF
primary-listing daily bars) over the 2018-2022 discovery window, reusing the
Phase E screen cache. No download, no vendor call, no cost, no 2023-2024
validation data, no 2025 holdout, no registry write. The ranking is research
prioritization for portfolio CONSIDERATION: no weights, no verdicts.

Writes a provenance artifact (readable name; hashes inside).

    PYTHONPATH=python python scripts/news_alpha_phase_f_signal_ranking.py
    PYTHONPATH=python python scripts/news_alpha_phase_f_signal_ranking.py --cached-only   # rank what is cached
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from alpha_agent.news_alpha import (
    DOMAIN_LABELS,
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
from alpha_agent.recommendation.signal_ranking import RankedSignalSet, rank_candidate_signals
from alpha_agent.screening.candidate_signal_screen import FactorScreenStore, screen_candidates

EVENTS = (
    ("AI_INFRASTRUCTURE",
     "Hyperscalers announce a major AI infrastructure spending increase, lifting data center capex guidance"),
    ("OPEC_CUT", "OPEC+ agrees a production cut of 1 million barrels per day"),
    ("FED_SURPRISE", "Rate-policy surprise: the Fed unexpectedly raised rates by 50 basis points"),
)
DESCRIBED_AT = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
MANDATE = ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES, MandateDomain.ETF))
OUT = Path(__file__).resolve().parents[1] / "outputs" / "news_alpha" / "phase_f" / "THREE_EVENTS__signal_ranking.json"


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _print(ranked: RankedSignalSet) -> None:
    print(f"  {ranked.headline()}")
    for s in ranked.ranked:
        m = s.merit.evidence
        role = "LEAD" if s.role.value == "LEAD" else f"alt #{ranked.signal(s.lead_id).rank}"
        print(f"  #{s.rank:<3} {role:<8} {DOMAIN_LABELS[s.domain]:<8} {s.instrument:<4} {s.expression:<22} "
              f"{s.prediction_horizon_days:>2}D  {s.merit.screen_status.value:<26} t {m.aligned_t:+.2f}  "
              f"{m.metrics.years_with_expected_sign}/{m.metrics.evaluable_years}y  {m.cost_headroom.value:<14} "
              f"{s.merit.data_quality.value}")
    for g in ranked.exposure_groups:
        rho = f", |rho| <= {g.max_abs_correlation:.2f}" if g.max_abs_correlation is not None else ""
        print(f"  exposure {g.label:<52} {len(g.member_ids):>2} signal(s), "
              f"{' + '.join(DOMAIN_LABELS[d] for d in g.domains)}{rho}, {len(g.event_ids)} event(s)")
    for t in ranked.event_theses:
        print(f"  thesis   {t.event_headline[:60]:<60} -> {len(t.exposure_group_ids)} exposure(s) via "
              f"{', '.join(t.consequence_labels[:3])}")
    for s in ranked.not_rankable:
        print(f"  not ranked {s.instrument:<5} {s.expression:<22} {s.not_rankable_reason.value}")


def _signal(s) -> dict:
    """One signal's decisive fields -- the full object is reproducible with
    --cached-only from the screen fingerprints recorded beside it."""
    merit = None
    if s.merit is not None:
        e = s.merit.evidence
        merit = {
            "screen_status": s.merit.screen_status.value, "direction": e.direction.value,
            "stability": e.stability.value, "cost_headroom": e.cost_headroom.value,
            "data_quality": s.merit.data_quality.value, "aligned_t": e.aligned_t, "peer_group": e.peer_group.key,
            "metrics": e.metrics.model_dump(mode="json"), "screen_fingerprint": s.merit.screen_fingerprint,
        }
    return {
        "candidate_signal_id": s.candidate_signal_id, "name": s.name, "domain": s.domain.value,
        "expression": s.expression, "prediction_horizon_days": s.prediction_horizon_days, "tier": s.tier.value,
        "rank": s.rank, "role": s.role.value if s.role else None, "exposure_group_id": s.exposure_group_id,
        "lead_id": s.lead_id, "not_rankable_reason": s.not_rankable_reason.value if s.not_rankable_reason else None,
        "merit": merit,
        "user_fit": {"admitted": s.user_fit.admitted, "exclusion_reasons": [r.value for r in s.user_fit.exclusion_reasons],
                     "checks": {c.dimension.value: c.status.value for c in s.user_fit.checks}},
        "events": list(s.mechanism.event_ids), "summaries": s.summaries.model_dump(mode="json"),
        "uncertainty": [u.code.value for u in s.uncertainty],
    }


def _dump(ranked: RankedSignalSet) -> dict:
    return {
        "fingerprint": ranked.fingerprint(), "headline": ranked.headline(),
        "policy": ranked.policy.model_dump(mode="json"), "policy_fingerprint": ranked.policy_fingerprint,
        "peer_groups": [g.model_dump(mode="json") | {"key": g.key} for g in ranked.peer_groups],
        "independent_exposures": ranked.independent_exposures, "family_size": ranked.family_size,
        "signals": [_signal(s) for s in ranked.signals],
        "exposure_groups": [g.model_dump(mode="json") for g in ranked.exposure_groups],
        "event_theses": [t.model_dump(mode="json") for t in ranked.event_theses],
        "linking_correlations": [p.model_dump(mode="json") for p in ranked.correlations if p.links],
        "n_correlation_pairs": len(ranked.correlations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cached-only", action="store_true", help="rank with cached screens only (no screening)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    universe = resolve_allowed_universe(MANDATE, capabilities=probe_domain_capabilities())
    sets: dict[str, CandidateSignalSet] = {}
    for name, text in EVENTS:
        scan = scan_initial_impact(UserDescribedEvent.create(text, described_at=DESCRIBED_AT), MANDATE,
                                   universe=universe)
        paths = discover_signal_paths(build_economic_mechanism_graph(scan))
        sets[name] = build_candidate_signals(build_asset_expressions(paths, universe, MANDATE), paths)
        print(f"EVENT {name:<18} {len(sets[name].candidates)} candidates on {', '.join(sets[name].instruments())}")

    store = FactorScreenStore()
    started = datetime.now(UTC)
    screens: dict = {}
    for name, cset in sets.items():
        if args.cached_only:
            screens.update(store.load_many(cset))
            continue
        done, _ = screen_candidates(
            cset, store=store,
            progress=lambda c: print(f"  screening {c.spec.instrument:<4} {c.expression} → "
                                     f"{c.spec.prediction_horizon.value}", flush=True),
        )
        screens.update({s.candidate_signal_id: s for s in done})
    wall = round((datetime.now(UTC) - started).total_seconds(), 1)

    print("\nACROSS ALL THREE EVENTS")
    across = rank_candidate_signals(tuple(sets.values()), screens, MANDATE)
    _print(across)
    per_event = {name: rank_candidate_signals(cset, screens, MANDATE) for name, cset in sets.items()}
    for name, ranked in per_event.items():
        print(f"\n{name}")
        print(f"  {ranked.headline()}")

    futures_only = ResearchMandate(allowed_domains=(MandateDomain.FUTURES,))
    narrowed = rank_candidate_signals(tuple(sets.values()), screens, futures_only)
    same_merit = all(narrowed.signal(s.candidate_signal_id).merit == s.merit for s in across.signals)
    print(f"\nFUTURES-ONLY MANDATE  {narrowed.headline()}  (merit unchanged for every signal: {same_merit})")

    artifact = {
        "artifact": "news-alpha-phase-f-signal-ranking/1",
        "plane": "RESEARCH PRIORITIZATION -- not portfolio weights, not validation, not a verdict",
        "generated_at": datetime.now(UTC).isoformat(),
        "code_commit": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "events": [{"name": n, "text": t, "event_id": sets[n].event_id} for n, t in EVENTS],
        "mandate_fingerprint": MANDATE.fingerprint(),
        "candidate_set_fingerprints": {n: s.fingerprint() for n, s in sets.items()},
        "screens": {"n": len(screens), "cached_only": args.cached_only, "screening_wall_seconds": wall,
                    "fingerprints": {k: v.fingerprint() for k, v in sorted(screens.items())}},
        "data_cost_usd": 0.0,
        "data_note": "already-acquired raw store only; no vendor call, no download",
        "windows": {"screened": "2018-01-01 -> 2023-01-01 (exclusive)",
                    "untouched": ["2023-2024 validation", "2025 locked holdout"]},
        "across_events": _dump(across),
        "per_event": {n: {"fingerprint": r.fingerprint(), "headline": r.headline(),
                          "independent_exposures": r.independent_exposures} for n, r in per_event.items()},
        "mandate_variant_futures_only": {"headline": narrowed.headline(), "merit_unchanged": same_merit,
                                         "excluded": len(narrowed.excluded)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWROTE {args.out}  (screening {wall}s)")


if __name__ == "__main__":
    main()
