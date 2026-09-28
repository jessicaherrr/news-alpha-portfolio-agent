"""News Alpha -- runnable vertical slice of the canonical pipeline so far.

    ResearchMandate -> Allowed Asset Universe -> News/Event
        -> Initial Impact Scan (Phase A) -> existing translation boundary
        -> Economic Mechanism Graph (Phase B)
        -> Signal Path Discovery: direct / supply-chain / cross-sector
           -> Economic Consequence (Phase C)
        -> Mechanism check of the initial triage, beside it; levels kept (Phase C)
        -> Asset Expression -> Measurement Spec -> PIT Data Field Resolution (Phase D)
        -> Candidate Signal Spec -> Factor Expression (Phase E; factor series and
           diagnostics on real bars: scripts/news_alpha_phase_e_vertical_slice.py)
        -> Multi-asset signal ranking across the examples, on cached real screens
           (Phase F; screening + ranking: scripts/news_alpha_phase_f_signal_ranking.py)
        -> Portfolio construction on cached market snapshots, sized by the C++
           allocator (Phase G; loading + plans: scripts/news_alpha_phase_g_portfolio.py)
        -> Validation gate + path-study readiness (Phase H; the C++ validation run,
           registry evidence and memory: scripts/news_alpha_phase_h_validation.py)

Offline and deterministic: no network, no LLM, no registry write (the
translation step opens the Phase 14 registry read-only for research memory).
Runs the three prompt examples as user-described events, plus the most
relevant already-cached real market_intel items when a local news cache
exists. ``--scripted-proposal`` adds hard-coded, clearly-labelled FIXTURES
shaped like model output to the AI example (no LLM is called): a link
proposal (a proposed link stays PROPOSED, a contradicting one is UNRESOLVED)
and a path proposal (a duplicate is recorded, not added; a ticker, a trade
and a skipped link are REJECTED with typed reasons).

    PYTHONPATH=python python scripts/news_alpha_pipeline_demo.py
    PYTHONPATH=python python scripts/news_alpha_pipeline_demo.py --no-cached --scripted-proposal
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from alpha_agent.market_intel.store import DEFAULT_DB_PATH, NewsStore
from alpha_agent.news_alpha import (
    DOMAIN_LABELS,
    AllowedAssetUniverse,
    AssetExpressionPlan,
    CandidateSignalSet,
    EconomicMechanismGraph,
    ExpressionStatus,
    ImpliedMovement,
    InitialImpactScan,
    MandateDomain,
    MechanismAdjustedImpact,
    PathStatus,
    ResearchMandate,
    SignalPathDiscovery,
    SignalPathProposal,
    TransmissionClaim,
    TransmissionProposal,
    UserDescribedEvent,
    adjust_impact,
    build_asset_expressions,
    build_candidate_signals,
    build_economic_mechanism_graph,
    build_translation_handoffs,
    claims_from_proposal,
    discover_signal_paths,
    resolve_allowed_universe,
    scan_initial_impact,
    scan_many,
)
from alpha_agent.news_alpha.study_design import (
    assess_study,
    default_designs,
    probe_study_capabilities,
)
from alpha_agent.portfolio import (
    EligibilityMode,
    PortfolioConstructionPolicy,
    construct_portfolio_plan,
)
from alpha_agent.portfolio.risk_model import InsufficientRiskData, MarketSnapshotStore
from alpha_agent.portfolio.validation import gate_portfolio
from alpha_agent.recommendation.signal_ranking import rank_candidate_signals
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.screening.candidate_signal_screen import FactorScreenStore
from alpha_agent.translation.pipeline import build_observation_translation

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "data" / "registry" / "experiments.sqlite"
DESCRIBED_AT = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)

EXAMPLES = (
    ("A", "Hyperscalers announce a major AI infrastructure spending increase, lifting data center capex guidance"),
    ("B", "OPEC+ agrees a production cut of 1 million barrels per day"),
    ("C", "Rate-policy surprise: the Fed unexpectedly raised rates by 50 basis points"),
)


_GLYPH = {"UP": "↑", "DOWN": "↓", "MIXED": "⇅", "INDETERMINATE": "?"}
_RELATIVE = {"POSITIVE": "+", "NEGATIVE": "−", "AMBIGUOUS": "±", "UNKNOWN": "?"}

#: A FIXTURE shaped like a model's `TransmissionProposal` -- hard-coded, not
#: produced by any LLM call. One new link, and one that contradicts a seed.
SCRIPTED_PROPOSAL = TransmissionProposal(
    proposer="scripted-fixture", prompt_fingerprint="demo",
    links=(
        {"source": "HBM demand", "target": "advanced packaging equipment demand", "polarity": "POSITIVE",
         "channel": "INPUT_DEMAND", "lag": "QUARTERS", "confidence": "MEDIUM",
         "rationale": "HBM stacking needs through-silicon-via and bonding tools",
         "citations": [{"title": "(fixture citation)", "reference": "https://example.org/fixture"}]},
        {"source": "electricity demand", "target": "grid investment", "polarity": "NEGATIVE",
         "rationale": "fixture contradiction: utilities defer grid spending when load growth is uncertain"},
    ),
)


#: A FIXTURE shaped like a model's `SignalPathProposal` -- hard-coded, not
#: produced by any LLM call.
SCRIPTED_PATH_PROPOSAL = SignalPathProposal(
    proposer="scripted-fixture", prompt_fingerprint="demo",
    paths=(
        {"states": ["AI capex", "compute demand", "GPU demand", "HBM"],
         "rationale": "fixture duplicate: the memory leg of the accelerator supply chain"},
        {"states": ["AI infrastructure investment", "data center buildout", "NVDA"],
         "rationale": "fixture: a ticker is not an economic state"},
        {"states": ["AI infrastructure investment", "compute demand"], "rationale": "fixture: BUY the spenders"},
        {"states": ["AI infrastructure investment", "HBM demand"],
         "rationale": "fixture: skips the links in between"},
    ),
)
_TYPE = {"DIRECT": "direct", "SUPPLY_CHAIN": "supply", "CROSS_SECTOR": "cross"}


def _print_paths(d: SignalPathDiscovery) -> None:
    if d.is_empty:
        print("  paths      none -- no seeded mechanism graph for this event")
        return
    counts = d.count_by_type()
    status = d.count_by_status()
    print(f"  paths      {len(d.paths)} signal paths ({', '.join(f'{n} {t.value.lower()}' for t, n in counts.items())}) "
          f"· {status[PathStatus.RESEARCHABLE]} researchable · {d.fingerprint()[:24]}…")
    for p in d.paths:
        kind = _TYPE.get(p.path_type.value, "?") if p.path_type else "unclassified"
        direction = _GLYPH.get(p.expected_direction.value) or _RELATIVE[p.relative_sign.value]
        why = f"  ({', '.join(r.value.lower() for r in p.status_reasons)})" if p.status_reasons else ""
        print(f"   {kind:<7} d{p.transmission_depth} {direction} {p.consequence_label:<34} {p.status.value:<12} "
              f"{p.expected_horizon.value.lower():<9} -> {p.candidate_target_concept or '--'}{why}")
    for c in d.conflicts:
        print(f"  opposing   {c.note}")
    for f in d.findings:
        print(f"  finding    {f.kind.value}: {f.message}")
    for r in d.rejected:
        print(f"  rejected   {' → '.join(r.states)}: {', '.join(x.value for x in r.reasons)}")


def _print_adjusted(adj: MechanismAdjustedImpact) -> None:
    for a in adj.assessments:
        if a.initial_level is None:
            continue
        print(f"  second     {DOMAIN_LABELS[a.asset_domain]:<8} level {a.initial_level.value} -> "
              f"{a.adjusted_level.value} (magnitude not established) · mechanism corroborates "
              f"{a.corroborated()} of {len(a.revisions)} exposures")
        for r in a.revisions:
            confidence = r.mechanism_confidence.value.lower() if r.mechanism_confidence else "--"
            print(f"             {r.exposure_label}: {r.mechanism_support.value.lower()} (route confidence "
                  f"{confidence})")


_PRESSURE = {"UP": "↑", "DOWN": "↓"}


def _print_expressions(plan: AssetExpressionPlan) -> None:
    if plan.is_empty:
        print("  express    none -- no signal path reaches a consequence")
        return
    counts = plan.count_by_resolution()
    print(f"  express    {len(plan.expressions)} asset expressions, {len(plan.measurements)} measurements "
          f"({', '.join(f'{n} {s.value.lower()}' for s, n in counts.items() if n)}) · {plan.fingerprint()[:24]}…")
    for e in plan.expressions:
        measured = set(e.measurement_ids)
        usable = sum(plan.measurement(i).resolution.historically_usable for i in measured)
        pressure = " ".join(
            f"{_PRESSURE.get(p.pressure.value, '?') if p.pressure else '?'}{p.horizon.value.lower()}" for p in e.pressures
        )
        where = ", ".join(e.symbols) or e.status.value.lower()
        line = (f"   {DOMAIN_LABELS[e.domain]:<8} {e.consequence_label[:30]:<30} {e.concept[:34]:<34} "
                f"{e.fidelity.value.lower():<20} "
                f"{where[:22]:<22} {pressure:<18}")
        if e.status is ExpressionStatus.CONTINUES:
            line += f" measurable {usable}/{len(measured)}{' · can run' if e.execution_capable else ''}"
        print(line)
    for m in plan.usable_measurements():
        r = m.resolution
        print(f"  usable     {m.target:<5} {m.label:<24} {r.status.value:<21} {r.dataset} {r.field} "
              f"{r.coverage_note.split(' · ')[0]} pit={r.pit_safe}")
    for b in plan.blockers()[:6]:
        print(f"  gap        {b.gap} -- primary blocker for {b.primary_for} (only blocker for "
              f"{b.only_blocker_for}), also blocks {b.also_blocks} ({', '.join(b.targets[:6])})")
    for s in plan.domains:
        if s.in_mandate and not s.expressions or not s.in_mandate:
            print(f"  domain     {DOMAIN_LABELS[s.domain]}: {s.note}")


def _print_candidates(cset: CandidateSignalSet) -> None:
    if cset.is_empty:
        print("  candidate  none -- no real, point-in-time safe, executable measurement to build one from")
    for c in cset.candidates:
        types = ", ".join(t.value if t else "UNCLASSIFIED" for t in c.path_types)
        print(f"  candidate  {c.spec.instrument:<5} {c.expression:<22} → {c.spec.prediction_horizon.value:<4} "
              f"{c.spec.expected_relationship.value.lower():<8} {len(c.origins)} origin(s) ({types}; depth "
              f"{min(c.depths)}-{max(c.depths)}) · {c.candidate_signal_id[:18]}…")
    by_reason = cset.refusals_by_reason()
    if by_reason:
        print("  not cand.  " + ", ".join(f"{len(v)} {k.value.lower().replace('_', ' ')}" for k, v in by_reason.items()))


def _print_ranking(mandate: ResearchMandate, universe: AllowedAssetUniverse) -> None:
    """Phase F across examples A-C, on whatever REAL screens the local cache
    already holds (read-only; screening itself is
    scripts/news_alpha_phase_f_signal_ranking.py). An unscreened candidate is
    NOT RANKABLE -- never ranked low."""
    sets = []
    for _tag, text in EXAMPLES:
        e = UserDescribedEvent.create(text, described_at=DESCRIBED_AT)
        paths = discover_signal_paths(build_economic_mechanism_graph(scan_initial_impact(e, mandate, universe=universe)))
        sets.append(build_candidate_signals(build_asset_expressions(paths, universe, mandate), paths))
    store = FactorScreenStore()
    screens = {k: v for cset in sets for k, v in store.load_many(cset).items()}
    ranked = rank_candidate_signals(tuple(sets), screens, mandate)
    print(f"SIGNAL RANKING        examples A-C, {len(screens)} cached real screen(s): {ranked.headline()}")
    _print_portfolio(ranked, screens, mandate)
    for s in ranked.leads:
        print(f"  lead       #{s.rank} {s.instrument:<5} {s.expression:<22} → {s.prediction_horizon_days}D  "
              f"{s.summaries.redundancy}")
    reasons: dict[str, int] = {}
    for s in ranked.not_rankable:
        reasons[s.not_rankable_reason.value] = reasons.get(s.not_rankable_reason.value, 0) + 1
    if reasons:
        print("  not ranked " + ", ".join(f"{n} {r.lower().replace('_', ' ')}" for r, n in reasons.items()))


def _print_portfolio(ranked, screens, mandate: ResearchMandate) -> None:
    """Phase G on CACHED market snapshots only (a futures snapshot takes a
    minute to rebuild -- scripts/news_alpha_phase_g_portfolio.py does that).
    Sizing is the compiled C++ allocator; without it the plan says so."""
    store = MarketSnapshotStore()

    def cached(domain, symbol):
        snap = store.load(domain, symbol)
        if snap is None:
            raise InsufficientRiskData("market snapshot not cached")
        return snap

    for mode in EligibilityMode:
        plan = construct_portfolio_plan(ranked, screens, mandate, snapshots=cached,
                                        policy=PortfolioConstructionPolicy(eligibility=mode))
        print(f"PORTFOLIO             {plan.headline()}")
        for i in plan.positions:
            print(f"  position   {i.instrument_key:<12} {i.units:>+7} unit(s)  notional {i.executable_notional_usd:>+12,.0f}")
        # Phase H: the gate alone -- no market data, no validation run, no registry write.
        gate = gate_portfolio(plan, ranked, screens)
        print(f"VALIDATION GATE       {'eligible' if gate is None else gate.headline()}")
    for study in (assess_study(d, probe_study_capabilities()) for d in default_designs()):
        print(f"PATH STUDY            {study.design.question.value}: {study.status.value} "
              f"({len(study.blockers)} missing requirement(s))")


def _print_graph(mg: EconomicMechanismGraph) -> None:
    if mg.is_empty:
        for u in mg.unseeded_channels:
            print(f"  graph      {u.note}")
        return
    labels = {s.state_id: s.label for s in mg.graph.states}
    counts = mg.graph.support_counts()
    print(f"  graph      {len(mg.graph.states)} economic states, {len(mg.graph.edges)} links "
          f"({', '.join(f'{n} {s.value.lower()}' for s, n in counts.items())}) · {mg.fingerprint()[:24]}…")
    for imp in mg.implications:
        glyph = (_RELATIVE[imp.relative_sign.value] if imp.movement is ImpliedMovement.UNANCHORED
                 else _GLYPH.get(imp.movement.value, ""))
        if imp.is_anchor:
            print(f"  anchor     {imp.label} {'•' if imp.movement is ImpliedMovement.UNANCHORED else glyph}")
            continue
        chain = " → ".join(labels[s] for s in imp.paths[0].state_ids[1:])
        print(f"   {imp.hops} {glyph:<2} {imp.label:<38} via {chain} "
              f"[slowest {imp.paths[0].slowest_lag.value.lower()}, weakest {imp.paths[0].weakest_support.value.lower()}]")
        if imp.note:
            print(f"             {imp.note}")
    for issue in mg.issues:
        if issue.kind.value not in ("UNKNOWN_LAG", "UNKNOWN_CONFIDENCE"):
            print(f"  ! {issue.kind.value:<20} {issue.message}")
    for u in mg.unseeded_channels:
        print(f"  unseeded   {u.channel_label}")


def _print_scan(scan: InitialImpactScan, registry: ExperimentRegistry | None,
                extra_claims: tuple[TransmissionClaim, ...] = (),
                path_proposals: tuple[SignalPathProposal, ...] = (),
                mandate: ResearchMandate | None = None, universe: AllowedAssetUniverse | None = None) -> None:
    print(f"  event      {scan.event.kind.value} {scan.event.event_id}: {scan.event.headline}")
    for ch in scan.channels:
        shift = f", shift={ch.shift}" if ch.shift else ""
        print(f"  channel    {ch.channel_label} [{ch.strength.value}; {', '.join(b.value for b in ch.bases)}{shift}]")
    for a in scan.assessments:
        label = DOMAIN_LABELS[a.asset_domain]
        level = a.impact_level.value if a.impact_level else "not assessed"
        markets = ", ".join(m.symbol + ("*" if m.research_ready else "") for m in a.candidate_markets[:6])
        line = f"  {label:<8}   {level:<12} {a.availability.value:<22}"
        if markets:
            line += f" {markets}"
        print(line)
        for d in a.possible_direction:
            print(f"             direction: {d.exposure_label} {d.pressure.value} ({d.shift})")
        for note in a.mandate_notes:
            print(f"             mandate: {note}")

    plan = build_translation_handoffs(scan)
    for h in plan.handoffs[:2]:
        print(f"  handoff    {h.root_symbol} via {h.category_basis.value} -> category {h.translation_category}")
        if registry is not None:
            t = build_observation_translation(h.observation, registry=registry)
            outcome = t.hypothesis.title if t.hypothesis else (t.research_gap_note or "")[:90]
            print(f"             translation: {[m.mechanism.value for m in t.mechanism_candidates]} -> {outcome}")
    for g in plan.gaps:
        print(f"  gap        {DOMAIN_LABELS[g.asset_domain]} ({g.impact_level.value}): {g.reason}")
    graph = build_economic_mechanism_graph(scan, extra_claims=extra_claims)
    _print_graph(graph)
    paths = discover_signal_paths(graph, proposals=path_proposals)
    _print_paths(paths)
    if not graph.is_empty:
        before = scan.model_dump_json()
        _print_adjusted(adjust_impact(scan, paths))
        assert scan.model_dump_json() == before, "the initial scan must never be modified"
    if mandate is not None and universe is not None:
        expressions = build_asset_expressions(paths, universe, mandate)
        _print_expressions(expressions)
        _print_candidates(build_candidate_signals(expressions, paths))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-cached", action="store_true", help="skip already-cached real market_intel items")
    parser.add_argument("--scripted-proposal", action="store_true",
                        help="add the hard-coded model-proposal FIXTURE to example A (no LLM call)")
    args = parser.parse_args()

    mandate = ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES))
    universe = resolve_allowed_universe(mandate)
    print(f"MANDATE    {mandate.summary()}")
    print(f"           {mandate.fingerprint()}")
    for scope in universe.scopes:
        state = "allowed" if scope.in_mandate else "excluded"
        print(f"UNIVERSE   {DOMAIN_LABELS[scope.domain]:<8} {state:<9} {scope.support.value:<18} "
              f"{len(scope.instruments)} instruments")
    for note in universe.constraint_notes:
        print(f"           {note}")
    print()

    registry = ExperimentRegistry(REGISTRY_PATH) if REGISTRY_PATH.exists() else None
    try:
        for tag, text in EXAMPLES:
            print(f"EXAMPLE {tag}")
            scripted = args.scripted_proposal and tag == "A"
            extra = claims_from_proposal(SCRIPTED_PROPOSAL) if scripted else ()
            _print_scan(scan_initial_impact(UserDescribedEvent.create(text, described_at=DESCRIBED_AT), mandate,
                                            universe=universe), registry, extra,
                        (SCRIPTED_PATH_PROPOSAL,) if scripted else (), mandate, universe)
        if not args.no_cached and DEFAULT_DB_PATH.exists():
            items = NewsStore().list_recent(since=datetime.now(UTC) - timedelta(days=30), limit=50)
            for scan in scan_many(items, mandate, universe=universe)[:3]:
                print("CACHED REAL ITEM")
                _print_scan(scan, registry, mandate=mandate, universe=universe)
        equity_only = ResearchMandate(allowed_domains=(MandateDomain.EQUITY,))
        event = UserDescribedEvent.create(EXAMPLES[0][1], described_at=DESCRIBED_AT)
        same = {
            build_economic_mechanism_graph(scan_initial_impact(event, m)).fingerprint() for m in (mandate, equity_only)
        }
        print(f"MANDATE INDEPENDENCE  example A graph identical under an Equity-only mandate: {len(same) == 1}")
        same_paths = {
            discover_signal_paths(build_economic_mechanism_graph(scan_initial_impact(event, m))).fingerprint()
            for m in (mandate, equity_only)
        }
        print(f"MANDATE INDEPENDENCE  example A signal paths identical under an Equity-only mandate: "
              f"{len(same_paths) == 1}")
        plans = []
        for m in (mandate, equity_only):
            u = resolve_allowed_universe(m)
            paths = discover_signal_paths(build_economic_mechanism_graph(scan_initial_impact(event, m, universe=u)))
            plans.append(build_asset_expressions(paths, u, m))
        same_ids = [e.expression_id for e in plans[0].expressions] == [e.expression_id for e in plans[1].expressions]
        admitted = [len(p.with_status(ExpressionStatus.CONTINUES)) for p in plans]
        print(f"ECONOMIC RELEVANCE    example A expression set identical under an Equity-only mandate: {same_ids} "
              f"(admission differs: {admitted[0]} vs {admitted[1]} continue)")
        ids = []
        for _tag, text in EXAMPLES[:2]:
            e = UserDescribedEvent.create(text, described_at=DESCRIBED_AT)
            paths = discover_signal_paths(build_economic_mechanism_graph(scan_initial_impact(e, mandate,
                                                                                               universe=universe)))
            cset = build_candidate_signals(build_asset_expressions(paths, universe, mandate), paths)
            ids.append({c.candidate_signal_id: c for c in cset.candidates if c.spec.instrument == "NQ"})
        shared = set(ids[0]) & set(ids[1])
        print(f"STRUCTURAL IDENTITY   examples A and B share {len(shared)} NQ candidate(s) "
              f"({', '.join(ids[0][i].expression for i in shared)}) -- one factor, both events kept as origins")
        _print_ranking(mandate, universe)
    finally:
        if registry is not None:
            registry.close()


if __name__ == "__main__":
    main()
