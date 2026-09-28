"""News Alpha Phase H -- PortfolioPlan -> validation gate -> C++ backtest -> the
existing validation plane -> registry + signal-path memory.

Fixtures follow Phases F/G: candidates from the offline pipeline, screens from
the real Phase E engine over in-memory bars labelled REAL (so REAL-only rules
run), pinned where a test needs a screen outcome. Bars span 2018-2024 so the
validation window exists; the portfolio is ETF-only (the engine refuses a
mixed-unit book). Sizing and execution always go through the real compiled
C++ binaries -- these tests skip, never fall back, when they are not built.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from alpha_agent.alpha_graph.hypothesis_links import link_records
from alpha_agent.alpha_graph.schemas import InstrumentRef
from alpha_agent.alpha_memory.signal_path_evidence import EventResearch, build_signal_path_evidence
from alpha_agent.alpha_memory.signal_path_memory import (
    MemoryGuidance,
    lineage,
    signal_path_memory,
    summarize_path_memory,
)
from alpha_agent.crypto.provenance import DataProvenanceRole
from alpha_agent.etf.calendar import etf_calendar
from alpha_agent.news_alpha import (
    MandateDomain,
    ResearchMandate,
    UserDescribedEvent,
    build_asset_expressions,
    build_candidate_signals,
    build_economic_mechanism_graph,
    discover_signal_paths,
    resolve_allowed_universe,
    scan_initial_impact,
)
from alpha_agent.news_alpha.study_design import (
    ComparisonDesign,
    EventConditionedEvidenceRef,
    StudyArm,
    StudyCapabilities,
    StudyQuestion,
    StudyReadiness,
    StudyRequirement,
    assess_study,
    default_designs,
    probe_study_capabilities,
)
from alpha_agent.portfolio import (
    EligibilityMode,
    PlanStatus,
    PortfolioConstructionPolicy,
    construct_portfolio_plan,
)
from alpha_agent.portfolio.evaluation import LookAheadDefect, evaluate_factor_through_validation
from alpha_agent.portfolio.execution import (
    DEFAULT_COMMISSION_CONVENTIONS,
    ExecutionInputs,
    ExecutionUnsupported,
    ScheduleBarMismatch,
    commission_schedule_for,
    cost_plan_for,
    etf_engine_instrument_id,
    execution_support,
    merged_day_plan,
    replay_risk_policy,
    run_schedule,
)
from alpha_agent.portfolio.handoff import find_allocator_cli
from alpha_agent.portfolio.research_loop import close_research_loop
from alpha_agent.portfolio.risk_model import InstrumentMarketSnapshot, SnapshotWindow
from alpha_agent.portfolio.strategy import (
    RebalanceRule,
    build_rebalance_schedule,
    freeze_portfolio_strategy,
)
from alpha_agent.portfolio.validation import (
    IneligibilityReason,
    PortfolioValidationStatus,
    assess_validation_eligibility,
    validate_portfolio,
)
from alpha_agent.recommendation.signal_ranking import rank_candidate_signals
from alpha_agent.registry.enums import (
    EvidenceScope,
    HypothesisStage,
    PathEvidenceOutcome,
    PathEvidenceReason,
)
from alpha_agent.registry.holdout_guard import HoldoutAccessError
from alpha_agent.registry.models import SignalPathEvidenceRecord
from alpha_agent.registry.sqlite_registry import ExperimentRegistry, UnknownExperiment
from alpha_agent.screening.candidate_signal_screen import screen_candidate
from alpha_agent.screening.factor_diagnostics import ScreenStatus
from alpha_agent.validation.cost_stress import CommissionBasis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_news_alpha_phase_f import AI_CAPEX, AT, CAPS, _loader, _pinned

ALLOCATOR = find_allocator_cli()
REPLAY = find_allocator_cli("quant_portfolio_plan_replay_csv", "QUANT_PORTFOLIO_PLAN_REPLAY_CLI")
needs_cpp = pytest.mark.skipif(ALLOCATOR is None or REPLAY is None, reason="the C++ portfolio CLIs are not built")

ETF_ONLY = ResearchMandate(allowed_domains=(MandateDomain.ETF,), shorting_allowed=True)
ETF_NO_SHORT = ResearchMandate(allowed_domains=(MandateDomain.ETF,))
MIXED = ResearchMandate(allowed_domains=(MandateDomain.FUTURES, MandateDomain.ETF), shorting_allowed=True)
N_DAYS = 1900  # 2018-01-02 .. 2025-04 -- the loaders must cut at 2025-01-01
EVAL_END = date(2025, 1, 1)


def _frame(log_returns: np.ndarray, *, start: str = "2018-01-02") -> pd.DataFrame:
    days = pd.bdate_range(start, periods=len(log_returns)).as_unit("ns")  # engine timestamps are UTC ns
    close = np.round(100 * np.exp(np.cumsum(log_returns)), 2)  # on the ETF tick grid
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "trading_day": days.strftime("%Y-%m-%d"), "ts_event_ns": days.asi8,  # 00:00 UTC, like real ETF bars
        "open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close),
        "close": close, "volume": np.full(len(close), 1e6),
    })


def _worlds(seed: int = 5) -> dict[str, pd.DataFrame]:
    """Tech (QQQ, XLK, NQ) with a slow momentum; utilities (XLU) independent."""
    rng = np.random.default_rng(seed)
    tech = rng.normal(0.0004, 0.011, N_DAYS)
    frames = {s: _frame(tech + rng.normal(0, 0.002, N_DAYS)) for s in ("QQQ", "XLK", "NQ")}
    frames["XLU"] = _frame(rng.normal(0.0002, 0.009, N_DAYS))
    return frames


FRAMES = _worlds()


def _event(mandate: ResearchMandate, text: str = AI_CAPEX) -> EventResearch:
    universe = resolve_allowed_universe(mandate, capabilities=CAPS)
    scan = scan_initial_impact(UserDescribedEvent.create(text, described_at=AT), mandate, universe=universe)
    paths = discover_signal_paths(build_economic_mechanism_graph(scan))
    expressions = build_asset_expressions(paths, universe, mandate)
    return EventResearch(discovery=paths, expressions=expressions,
                         candidates=build_candidate_signals(expressions, paths))


def _screens(event: EventResearch, *, supported: bool) -> dict:
    loader = _loader(FRAMES)
    out = {c.candidate_signal_id: screen_candidate(c, loader=loader, now=AT) for c in event.candidates.candidates
           if c.spec.instrument in FRAMES}
    return {k: _pinned(v, status=ScreenStatus.SCREEN_CONTINUE, t=2.5) for k, v in out.items()} if supported else out


def _snapshot(domain: MandateDomain, symbol: str, window: SnapshotWindow = SnapshotWindow.EVALUATION,
              until: date | None = None) -> InstrumentMarketSnapshot:
    f = FRAMES[symbol]
    day = pd.to_datetime(f["trading_day"]).dt.date
    f = f[(day < window.end) & (day <= (until or window.end))]
    fut = domain is MandateDomain.FUTURES
    close = tuple(float(x) for x in f["close"])
    return InstrumentMarketSnapshot(
        domain=domain, symbol=symbol, execution_root=symbol if fut else f"E{symbol}", dataset="FIXTURE",
        unit="contract" if fut else "share", multiplier=20.0 if fut else 1.0, tick_size=0.25 if fut else 0.01,
        adjustment="ADDITIVE" if fut else "MULTIPLICATIVE",
        volume_scope="FRONT_CONTRACT" if fut else "PRIMARY_LISTING_ONLY", data_role=DataProvenanceRole.REAL,
        window=window, days=tuple(f["trading_day"]), ts_event_ns=tuple(int(x) for x in f["ts_event_ns"]),
        research_close=close, raw_close=close, volume=tuple(1e6 for _ in close),
        traded_symbol=tuple(symbol for _ in close), provenance=("test fixture",),
    )


def _provider(window=SnapshotWindow.EVALUATION, until=None):
    return lambda domain, symbol: _snapshot(domain, symbol, window, until)


def _plan(event, screens, mandate, mode=EligibilityMode.QUALIFIED):
    ranked = rank_candidate_signals((event.candidates,), screens, mandate)
    plan = construct_portfolio_plan(ranked, screens, mandate, snapshots=_provider(SnapshotWindow.DISCOVERY),
                                    policy=PortfolioConstructionPolicy(eligibility=mode))
    return ranked, plan


def _factor_loader(event, screens):
    candidates = {c.candidate_signal_id: c for c in event.candidates.candidates}

    def load(strategy):
        return {m.candidate_signal_id: evaluate_factor_through_validation(
            candidates[m.candidate_signal_id], screen=screens[m.candidate_signal_id], loader=_loader(FRAMES))
            for m in strategy.members}
    return load


def _inputs(strategy) -> ExecutionInputs:
    bars, contracts, ids, calendars = [], [], {}, {}
    for key in strategy.instrument_keys:
        m = strategy.instrument(key)
        iid = etf_engine_instrument_id(m.instrument)
        f = FRAMES[m.instrument]
        f = f[pd.to_datetime(f["trading_day"]).dt.date < EVAL_END]
        bars.append(pd.DataFrame({"ts_event_ns": f["ts_event_ns"].astype("int64"), "instrument_id": iid,
                                  "open": f["open"], "high": f["high"], "low": f["low"], "close": f["close"],
                                  "volume": f["volume"].astype("int64")}))
        contracts.append({"instrument_id": iid, "raw_symbol": m.instrument, "root_symbol": m.execution_root,
                          "exchange": "ARCX", "tick_size": 0.01, "multiplier": 1.0, "activation_ns": 1,
                          "expiration_ns": 4_102_444_800_000_000_000, "first_notice_ns": "", "last_trade_ns": ""})
        ids[m.execution_root] = (iid,)
        calendars[m.execution_root] = (etf_calendar(), m.instrument)
    frame = pd.concat(bars, ignore_index=True).sort_values(["ts_event_ns", "instrument_id"]).reset_index(drop=True)
    return ExecutionInputs(bars=frame, contracts=pd.DataFrame(contracts),
                           roll_close_marks=pd.DataFrame(columns=["instrument_id", "ts_event_ns", "close"]),
                           splits=pd.DataFrame(columns=["instrument_id", "effective_ts_ns", "ratio"]),
                           instrument_ids=ids, calendars=calendars, provenance=("test fixture",))


def _never(*_a, **_k):
    raise AssertionError("a gated-out portfolio must never load validation-window data")


def _validate(event, screens, mandate, plan, ranked, tmp_path, registry=None, **kw):
    return validate_portfolio(plan, ranked, screens, mandate, load_factor_sources=_factor_loader(event, screens),
                              snapshots=_provider(), load_inputs=_inputs, workdir=tmp_path / "runs",
                              registry=registry, allocator_cli=ALLOCATOR, replay_cli=REPLAY, **kw)


@pytest.fixture(scope="module")
def etf_event():
    return _event(ETF_ONLY)


@pytest.fixture(scope="module")
def qualified(etf_event):
    screens = _screens(etf_event, supported=True)
    ranked, plan = _plan(etf_event, screens, ETF_ONLY)
    return screens, ranked, plan


@pytest.fixture(scope="module")
def validated(etf_event, qualified, tmp_path_factory):
    if ALLOCATOR is None or REPLAY is None:
        pytest.skip("the C++ portfolio CLIs are not built")
    screens, ranked, plan = qualified
    return _validate(etf_event, screens, ETF_ONLY, plan, ranked, tmp_path_factory.mktemp("phase_h"))


# ---------------------------------------------------------------------------
# the input gate: a plan is not eligibility
# ---------------------------------------------------------------------------


@needs_cpp
def test_an_unsupported_plan_is_refused_before_any_validation_data_is_read(etf_event, tmp_path):
    screens = _screens(etf_event, supported=False)
    ranked, plan = _plan(etf_event, screens, ETF_ONLY)
    assert plan.status is PlanStatus.NO_ELIGIBLE_SIGNAL
    report = validate_portfolio(plan, ranked, screens, ETF_ONLY, load_factor_sources=_never, snapshots=_never,
                                load_inputs=_never, workdir=tmp_path)
    assert report.status is PortfolioValidationStatus.NO_ELIGIBLE_PORTFOLIO
    assert IneligibilityReason.PLAN_NOT_CONSTRUCTED in report.eligibility.reasons
    assert report.report is None and report.headline_run is None and report.data_read == ()
    assert report.holdout.status == "HOLDOUT_NOT_RELEASED" and not report.holdout.accessed


@needs_cpp
def test_an_exploratory_preview_is_never_eligible_even_when_it_is_constructed(etf_event, tmp_path):
    screens = _screens(etf_event, supported=False)
    lean = {k: _pinned(v, status=ScreenStatus.NO_SCREEN_SUPPORT, t=1.0) for k, v in screens.items()}
    ranked, plan = _plan(etf_event, lean, ETF_ONLY, EligibilityMode.EXPLORATORY)
    assert plan.status is PlanStatus.CONSTRUCTED and plan.is_exploratory
    report = validate_portfolio(plan, ranked, lean, ETF_ONLY, load_factor_sources=_never, snapshots=_never,
                                load_inputs=_never, workdir=tmp_path)
    assert report.status is PortfolioValidationStatus.NO_ELIGIBLE_PORTFOLIO
    assert IneligibilityReason.EXPLORATORY_PLAN in report.eligibility.reasons
    assert IneligibilityReason.MEMBER_WITHOUT_SCREEN_SUPPORT in report.eligibility.reasons


@needs_cpp
def test_the_gate_rechecks_every_member_against_its_actual_screen_not_the_plans_label(etf_event, qualified):
    screens, ranked, plan = qualified
    downgraded = dict(screens)
    victim = next(iter(downgraded))
    downgraded[victim] = _pinned(downgraded[victim], status=ScreenStatus.NO_SCREEN_SUPPORT)
    gate = assess_validation_eligibility(plan, ranked, downgraded)
    assert not gate.eligible
    (finding,) = [f for f in gate.findings if f.reason is IneligibilityReason.MEMBER_WITHOUT_SCREEN_SUPPORT]
    assert finding.candidate_signal_ids == (victim,)
    missing = {k: v for k, v in screens.items() if k != victim}
    assert IneligibilityReason.MEMBER_SCREEN_MISSING in assess_validation_eligibility(plan, ranked, missing).reasons
    assert assess_validation_eligibility(plan, ranked, screens).eligible


@needs_cpp
def test_a_mixed_book_is_priced_per_root_and_refused_only_for_its_mixed_trading_days(tmp_path):
    """With a complete commission schedule the cost model no longer blocks a
    futures + ETF book; what remains is a typed refusal for one daily return
    across a CME session and a US-equity day (a validation decision)."""
    event = _event(MIXED)
    screens = _screens(event, supported=True)
    ranked, plan = _plan(event, screens, MIXED)
    report = validate_portfolio(plan, ranked, screens, MIXED, load_factor_sources=_never, snapshots=_never,
                                load_inputs=_never, workdir=tmp_path)
    assert report.status is PortfolioValidationStatus.EXECUTION_UNSUPPORTED
    assert "trading-day conventions" in report.status_detail and "commission" not in report.status_detail
    schedule = {c.root_symbol: c for c in commission_schedule_for(report.strategy)}
    assert schedule["NQ"].commission_per_unit_usd == 2.0 and schedule["NQ"].unit == "contract"
    assert all(schedule[r].commission_per_unit_usd == 0.005 and schedule[r].unit == "share"
               for r in schedule if r.startswith("E"))


@needs_cpp
def test_an_unpriced_root_is_a_typed_cost_model_refusal_never_a_fallback(etf_event, qualified, tmp_path):
    screens, ranked, plan = qualified
    futures_only = {MandateDomain.FUTURES: DEFAULT_COMMISSION_CONVENTIONS[MandateDomain.FUTURES]}
    report = validate_portfolio(plan, ranked, screens, ETF_ONLY, load_factor_sources=_never, snapshots=_never,
                                load_inputs=_never, workdir=tmp_path, commission_conventions=futures_only)
    assert report.status is PortfolioValidationStatus.COST_MODEL_INCOMPLETE
    assert report.report is None and report.data_read == ()
    refusal = execution_support(report.strategy, futures_only)
    assert refusal.kind == "COST_MODEL_INCOMPLETE" and set(refusal.unpriced_roots) == set(
        report.strategy.execution_roots.values())
    with pytest.raises(ExecutionUnsupported, match="COST_MODEL_INCOMPLETE"):
        cost_plan_for(report.strategy, futures_only)
    # a mixed book missing its ETF entry: the cost refusal comes first
    event = _event(MIXED)
    mixed_screens = _screens(event, supported=True)
    mixed_ranked, mixed_plan = _plan(event, mixed_screens, MIXED)
    mixed = validate_portfolio(mixed_plan, mixed_ranked, mixed_screens, MIXED, load_factor_sources=_never,
                               snapshots=_never, load_inputs=_never, workdir=tmp_path,
                               commission_conventions=futures_only)
    assert mixed.status is PortfolioValidationStatus.COST_MODEL_INCOMPLETE
    assert "NQ" not in execution_support(mixed.strategy, futures_only).unpriced_roots


def test_the_research_rates_are_labelled_assumptions_not_broker_truth():
    for convention in DEFAULT_COMMISSION_CONVENTIONS.values():
        assert convention.basis is CommissionBasis.DECLARED_RESEARCH_ASSUMPTION
    etf = DEFAULT_COMMISSION_CONVENTIONS[MandateDomain.ETF]
    assert etf.commission_per_unit_usd == 0.005 and "not observed broker truth" in etf.source


def test_a_scalar_cost_plan_and_cost_identity_are_byte_identical_to_before_the_schedule():
    from alpha_agent.registry.identity import cost_config_identity
    from alpha_agent.validation.cost_stress import DEFAULT_COST_STRESS_PLAN
    from alpha_agent.validation.phase_13_5c_matrix import frozen_cost_plan

    # pinned from commit 2cd296b, before commission schedules existed
    assert DEFAULT_COST_STRESS_PLAN.identity() == frozen_cost_plan().identity() == (
        "valcoststress1:98934bb08ba25afb4c0b043bfc59b7ca14786fbb2a5cd3943ce91f639b0d714d")
    scenarios = [s.model_dump(mode="json") for s in frozen_cost_plan().scenarios]
    legacy = cost_config_identity(base_commission_per_contract_usd=2.0, base_slippage_ticks=0.0,
                                  base_spread_ticks=0.0, scenarios=scenarios)
    assert legacy == "costconfig1:a732b8bfcceccb80957324536fc50ac6bfaa57b53eeaa0e708da39d34a4d0aca"
    assert cost_config_identity(base_commission_per_contract_usd=2.0, base_slippage_ticks=0.0,
                                base_spread_ticks=0.0, scenarios=scenarios, commission_schedule=[]) != legacy


def test_no_portfolio_module_computes_a_commission():
    """Official commissions are the C++ engine's: no Phase H module multiplies
    anything by a commission (scaling a declared RATE per stress scenario lives
    in validation.cost_stress, and is configuration, not a charge)."""
    import ast

    pkg = Path(__file__).resolve().parents[2] / "python" / "alpha_agent" / "portfolio"
    offenders = [f"{f.name}: {ast.unparse(n)}" for f in pkg.glob("*.py") for n in ast.walk(ast.parse(f.read_text()))
                 if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) and "commission" in ast.unparse(n)]
    assert offenders == []


# ---------------------------------------------------------------------------
# the frozen strategy and its causal schedule
# ---------------------------------------------------------------------------


@needs_cpp
def test_the_frozen_strategy_is_the_eligible_set_not_the_as_of_directed_subset(etf_event, qualified):
    _, ranked, plan = qualified
    strategy = freeze_portfolio_strategy(plan, ranked)
    assert {m.candidate_signal_id for m in strategy.members} >= {s.candidate_signal_id for s in plan.selected}
    assert strategy.source_plan_fingerprint == plan.fingerprint()
    assert strategy.fingerprint().startswith("portstrat1:")
    assert strategy.with_rebalance(RebalanceRule(every_trading_days=10)).fingerprint() != strategy.fingerprint()
    assert "never validates a member signal" in strategy.scope_note


@needs_cpp
def test_each_rebalance_reads_nothing_after_its_own_day(etf_event, qualified):
    """Causality: truncating every input after day t leaves the decision on t unchanged."""
    screens, ranked, plan = qualified
    strategy = freeze_portfolio_strategy(plan, ranked)
    sources = _factor_loader(etf_event, screens)(strategy)
    full = build_rebalance_schedule(strategy, ranked, ETF_ONLY, factor_sources=sources, snapshots=_provider(),
                                    start=date(2023, 1, 1), end_exclusive=EVAL_END, allocator_cli=ALLOCATOR)
    probe = full.decisions[len(full.decisions) // 2]
    cut = {k: s.model_copy(update={"days": tuple(d for d in s.days if d <= probe.day.isoformat()),
                                   "values": tuple(v for d, v in zip(s.days, s.values, strict=True)
                                                   if d <= probe.day.isoformat())}) for k, s in sources.items()}
    truncated = build_rebalance_schedule(strategy, ranked, ETF_ONLY, factor_sources=cut,
                                         snapshots=_provider(until=probe.day), start=probe.day,
                                         end_exclusive=date.fromordinal(probe.day.toordinal() + 1),
                                         allocator_cli=ALLOCATOR)
    assert truncated.decisions[0].units == probe.units
    assert truncated.decisions[0].directions == probe.directions


@needs_cpp
def test_a_no_short_mandate_never_schedules_a_short_and_a_flat_day_is_typed(tmp_path):
    event = _event(ETF_NO_SHORT)
    screens = _screens(event, supported=True)
    ranked, plan = _plan(event, screens, ETF_NO_SHORT)
    strategy = freeze_portfolio_strategy(plan, ranked)
    sources = _factor_loader(event, screens)(strategy)
    sched = build_rebalance_schedule(strategy, ranked, ETF_NO_SHORT, factor_sources=sources, snapshots=_provider(),
                                     start=date(2019, 1, 1), end_exclusive=EVAL_END, allocator_cli=ALLOCATOR)
    assert all(u >= 0 for d in sched.decisions for u in d.units.values())
    flat = [d for d in sched.decisions if d.status is not PlanStatus.CONSTRUCTED]
    assert all(set(d.units.values()) == {0} for d in flat)
    rows = sched.rows(strategy.execution_roots)
    assert [r.ts_event_ns for r in rows] == sorted(r.ts_event_ns for r in rows)
    assert len(rows) == len(sched.decisions) * len(strategy.instrument_keys)


def test_the_evaluation_series_never_reaches_the_holdout_and_guards_its_discovery_prefix(etf_event):
    screens = _screens(etf_event, supported=False)
    cid, screen = next(iter(screens.items()))
    candidate = next(c for c in etf_event.candidates.candidates if c.candidate_signal_id == cid)
    series = evaluate_factor_through_validation(candidate, screen=screen, loader=_loader(FRAMES))
    assert series.days[-1] < "2025-01-01" and FRAMES[candidate.spec.instrument]["trading_day"].iloc[-1] > "2025-01-01"
    assert series.prefix_days_verified == len(screen.series.days) > 0
    tampered_values = list(screen.series.values)
    i = next(j for j, v in enumerate(tampered_values) if v is not None)
    tampered_values[i] = tampered_values[i] + 1.0
    tampered = screen.model_copy(update={"series": screen.series.model_copy(update={"values": tuple(tampered_values)})})
    with pytest.raises(LookAheadDefect):
        evaluate_factor_through_validation(candidate, screen=tampered, loader=_loader(FRAMES))
    # even a measurement whose coverage runs past 2024 is cut at the holdout, never loaded into it
    spec = candidate.spec
    late = candidate.model_copy(update={"spec": spec.model_copy(update={"data": spec.data.model_copy(
        update={"coverage_end_exclusive": date(2026, 1, 1)})})})
    capped = evaluate_factor_through_validation(late, loader=_loader(FRAMES))
    assert capped.days[-1] < "2025-01-01" and capped.window_end_exclusive == EVAL_END


# ---------------------------------------------------------------------------
# execution through the unchanged C++ engine
# ---------------------------------------------------------------------------


def test_etf_engine_ids_are_stable_per_ticker_and_refuse_unknown_tickers():
    from alpha_agent.etf.universe import PILOT_UNIVERSE

    ids = [etf_engine_instrument_id(s) for s in PILOT_UNIVERSE]
    assert len(set(ids)) == len(ids) and all(4_000_000_000 <= i < 2**32 for i in ids)
    assert etf_engine_instrument_id("QQQ") == etf_engine_instrument_id("QQQ")
    with pytest.raises(ExecutionUnsupported):
        etf_engine_instrument_id("ZZZZ")


@needs_cpp
def test_merged_day_plan_uses_the_latest_root_and_refuses_mixed_conventions(qualified):
    _, ranked, plan = qualified
    inputs = _inputs(freeze_portfolio_strategy(plan, ranked))
    day_plan = merged_day_plan(inputs.window(0, 1_600_000_000_000_000_000))
    assert day_plan.n_days > 100 and all(d.n_bars == len(inputs.instrument_ids) for d in day_plan.days)
    from alpha_agent.data.calendars import default_calendar

    # two roots closing at different instants: each day's boundary is the LATER one
    second = sorted(inputs.instrument_ids)[1]
    later = inputs.bars["instrument_id"].isin(inputs.instrument_ids[second])
    staggered = inputs.bars.assign(ts_event_ns=inputs.bars["ts_event_ns"].where(~later, inputs.bars["ts_event_ns"]
                                                                                + 3_600_000_000_000))
    both = ExecutionInputs(**{**inputs.__dict__, "bars": staggered.sort_values(["ts_event_ns", "instrument_id"])
                              .reset_index(drop=True)})
    by_day = merged_day_plan(both.window(0, 1_600_000_000_000_000_000)).days
    second_ts = set(staggered.loc[later, "ts_event_ns"])
    assert by_day and all(d.boundary_ts_ns in second_ts for d in by_day)
    root = next(iter(inputs.calendars))
    mixed = ExecutionInputs(**{**inputs.__dict__, "calendars": {**inputs.calendars,
                                                               root: (default_calendar(), "NQ")}})
    with pytest.raises(ExecutionUnsupported):
        merged_day_plan(mixed.window(0, 1_600_000_000_000_000_000))


@needs_cpp
def test_a_target_off_its_roots_bars_is_a_pipeline_defect_not_a_fill(etf_event, qualified, tmp_path):
    screens, ranked, plan = qualified
    strategy = freeze_portfolio_strategy(plan, ranked)
    sched = build_rebalance_schedule(strategy, ranked, ETF_ONLY, factor_sources=_factor_loader(etf_event, screens)(
        strategy), snapshots=_provider(), start=date(2023, 1, 1), end_exclusive=date(2023, 3, 1),
        allocator_cli=ALLOCATOR)
    d = sched.decisions[0]
    shifted = d.model_copy(update={"decision_ts_ns": {k: v + 1 for k, v in d.decision_ts_ns.items()}})
    bad = sched.model_copy(update={"decisions": (shifted, *sched.decisions[1:])})
    with pytest.raises(ScheduleBarMismatch):
        run_schedule(bad, strategy, _inputs(strategy), start_ns=1_672_531_200_000_000_000,
                     end_ns=1_677_628_800_000_000_000, commission_schedule={r: 0.005 for r in strategy.execution_roots.values()},
                     workdir=tmp_path, label="bad",
                     cli=REPLAY)


@needs_cpp
def test_the_replay_gate_unit_caps_are_non_binding_by_construction(etf_event, qualified):
    screens, ranked, plan = qualified
    strategy = freeze_portfolio_strategy(plan, ranked)
    sched = build_rebalance_schedule(strategy, ranked, ETF_ONLY, factor_sources=_factor_loader(etf_event, screens)(
        strategy), snapshots=_provider(), start=date(2023, 1, 1), end_exclusive=EVAL_END, allocator_cli=ALLOCATOR)
    rows = sched.rows(strategy.execution_roots)
    policy = replay_risk_policy(strategy, rows)
    per_key = {}
    for r in rows:
        per_key[r.instrument_key] = max(per_key.get(r.instrument_key, 0), abs(r.target_units))
    assert policy.max_gross_contracts == max(1, sum(per_key.values()))
    assert policy.max_contracts_per_symbol == max(1, max(per_key.values()))
    assert policy.max_gross_leverage == strategy.constraints.max_gross_leverage


# ---------------------------------------------------------------------------
# validation: the existing plane, the frozen policy
# ---------------------------------------------------------------------------


@needs_cpp
def test_an_eligible_portfolio_is_validated_through_the_existing_plane(validated):
    r = validated
    assert r.status in (PortfolioValidationStatus.VALIDATED, PortfolioValidationStatus.REJECTED,
                        PortfolioValidationStatus.INSUFFICIENT_EVIDENCE)
    report = r.report
    assert report.verdict.value in ("PASS", "REJECT", "INCONCLUSIVE")
    assert not report.holdout_evaluated and r.holdout.status == "HOLDOUT_NOT_RELEASED"
    days = report.oos_daily.trading_day
    assert days[0] >= "2023-01-01" and days[-1] < "2025-01-01"
    assert r.headline_run.n_fills > 0 and r.headline_run.rebalances > 20
    # every economic number is the engine's: daily pnl sums to the C++ net
    assert sum(report.oos_daily.daily_pnl_usd) == pytest.approx(r.headline_run.net_pnl_usd, abs=1e-6)
    base, _, double = r.cost_runs
    assert double.commission_schedule == {r: 2 * v for r, v in base.commission_schedule.items()}
    assert set(base.commission_schedule) == set(r.strategy.execution_roots.values())
    assert all(v == 0.005 for v in base.commission_schedule.values())  # every ETF root per share
    assert {c.root_symbol for c in r.commission_schedule} == set(base.commission_schedule)
    # the hard gate may resize a fill differently on a different equity path -- the cost only roughly doubles
    assert double.costs_usd == pytest.approx(2 * base.costs_usd, rel=0.01)
    assert len(r.neighbour_runs) == 2 and report.parameter_stability is not None
    assert report.fdr_result.n_trials == 3 and report.dsr_result is not None
    assert len(r.fold_runs) == report.walk_forward.n_folds_evaluated >= 3
    assert all(f.span_end_exclusive <= date(2023, 1, 1) for f in r.fold_runs)
    assert report.regime_result.status.value == "evaluated"  # regime labels from TRAIN-only cut points
    assert r.validation_spec.strategy_key == "news_alpha_portfolio"
    assert "does not validate any member signal" in r.scope_note


@needs_cpp
def test_validation_is_deterministic(etf_event, qualified, validated, tmp_path):
    screens, ranked, plan = qualified
    again = _validate(etf_event, screens, ETF_ONLY, plan, ranked, tmp_path)
    assert again.report.report_fingerprint() == validated.report.report_fingerprint()
    assert again.experiment_identity == validated.experiment_identity


# ---------------------------------------------------------------------------
# the registry: experiments + schema v7 signal-path evidence
# ---------------------------------------------------------------------------


@needs_cpp
def test_the_loop_records_the_experiment_and_the_evidence_once(etf_event, qualified, tmp_path):
    screens, ranked, plan = qualified
    kw = {"workdir": tmp_path / "runs", "load_factor_sources": _factor_loader(etf_event, screens),
          "snapshots": _provider(), "load_inputs": _inputs, "allocator_cli": ALLOCATOR, "replay_cli": REPLAY,
          "recorded_at": "2026-09-27T00:00:00+00:00"}
    with ExperimentRegistry(tmp_path / "reg.sqlite") as reg:
        outcome, evidence = close_research_loop([etf_event], ranked, screens, plan, ETF_ONLY, registry=reg,
                                                record=True, **kw)
        assert outcome.recorded_experiments == 3  # canonical + 2 predeclared neighbours
        assert outcome.recorded_evidence == len(evidence) > 0
        view = reg.get(outcome.experiment_identity)
        assert view.experiment.strategy_family == "news_alpha_portfolio"
        assert view.experiment.asset_domain.value == "ETF" and "+" in view.experiment.root_symbol
        members = [r for r in evidence if r.portfolio_strategy_fingerprint]
        assert members and all(r.evidence_scope is EvidenceScope.PORTFOLIO for r in members)
        assert all(r.experiment_identity == outcome.experiment_identity for r in members)
        assert reg.signal_path_evidence(experiment_identity=outcome.experiment_identity)
        # the effective commission schedule is part of the recorded cost identity
        from alpha_agent.registry.identity import cost_config_identity

        costs = outcome.validation.validation_spec.cost_stress
        scalar_only = cost_config_identity(base_commission_per_contract_usd=0.0, base_slippage_ticks=0.0,
                                           base_spread_ticks=0.0,
                                           scenarios=[s.model_dump(mode="json") for s in costs.scenarios])
        assert view.experiment.cost_config_identity != scalar_only
        # the same hypothesis again: the registry result is cited, nothing new is written
        again, _ = close_research_loop([etf_event], ranked, screens, plan, ETF_ONLY, registry=reg, record=True, **kw)
        assert again.validation.status is PortfolioValidationStatus.PRIOR_RESULT_CITED
        assert again.recorded_experiments == 0 and again.validation.report is None
        # 11B: another portfolio validated against the same registry keeps its OWN
        # family -- canonical + predeclared neighbours; recorded portfolios are
        # not pooled by sharing a validation configuration
        other_event = _event(ETF_NO_SHORT)
        other_screens = _screens(other_event, supported=True)
        other_ranked, other_plan = _plan(other_event, other_screens, ETF_NO_SHORT)
        other = _validate(other_event, other_screens, ETF_NO_SHORT, other_plan, other_ranked, tmp_path / "other",
                          registry=reg)
        assert other.report.fdr_result.n_trials == 3
        assert {d.label for d in other.report.fdr_result.decisions} == {"canonical", "neighbour_0", "neighbour_1"}


def _record(**kw) -> SignalPathEvidenceRecord:
    base = {"research_run_id": "run", "recorded_at": "2026-09-27T00:00:00+00:00", "event_id": "e1",
            "event_headline": "h", "path_signature": "sig", "stage_reached": HypothesisStage.SCREEN,
            "outcome": PathEvidenceOutcome.FAILURE, "evidence_scope": EvidenceScope.SCREENING,
            "reason_code": PathEvidenceReason.SCREEN_NO_SUPPORT, "detail": "d"}
    return SignalPathEvidenceRecord.create(**{**base, **kw})


def test_signal_path_evidence_is_append_only_idempotent_and_guarded(tmp_path):
    with ExperimentRegistry(tmp_path / "reg.sqlite") as reg:
        digest = reg.content_digest()
        r = _record()
        assert reg.record_signal_path_evidence([r]) == 1 and reg.record_signal_path_evidence([r]) == 0
        assert reg.signal_path_evidence(path_signature="sig") == (r,)
        assert reg.content_digest() != digest  # present rows enter the digest
        with pytest.raises(UnknownExperiment):
            reg.record_signal_path_evidence([_record(event_id="e2", experiment_identity="experiment1:nope")])
        with pytest.raises(HoldoutAccessError):
            reg.record_signal_path_evidence([_record(event_id="e3", evidence={"as_of": "2025-02-03"})])
    with pytest.raises(ValueError, match="only a validated portfolio"):
        _record(outcome=PathEvidenceOutcome.SUCCESS).assert_consistent()
    assert _record().recorded_at != _record(recorded_at="later").recorded_at
    assert _record().evidence_id == _record(recorded_at="later").evidence_id  # provenance never moves the id
    assert _record().evidence_id.count("spevidence1:") == 1


def test_schema_v7_is_additive_and_an_empty_evidence_table_keeps_the_v6_digest(tmp_path):
    import sqlite3

    from alpha_agent.registry.schema import SCHEMA_VERSION

    path = tmp_path / "reg.sqlite"
    with ExperimentRegistry(path) as reg:
        before = reg.content_digest()
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE signal_path_evidence")
    conn.execute("UPDATE registry_meta SET value='6' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    with ExperimentRegistry(path) as reg:
        assert reg.schema_version == SCHEMA_VERSION == 7
        assert reg.content_digest() == before
        assert reg.signal_path_evidence() == ()


# ---------------------------------------------------------------------------
# memory: typed, scoped, never a blacklist
# ---------------------------------------------------------------------------


@needs_cpp
def test_every_hypothesis_of_the_run_is_accounted_for(etf_event, qualified, validated):
    screens, ranked, plan = qualified
    records = build_signal_path_evidence([etf_event], ranked=ranked, screens=screens, plan=plan, validation=validated,
                                         recorded_at="2026-09-27T00:00:00+00:00")
    origins = {(c.candidate_signal_id, o.path_id) for c in etf_event.candidates.candidates for o in c.origins}
    assert origins <= {(r.candidate_signal_id, r.path_id) for r in records if r.candidate_signal_id}
    refused = {r.expression_id for r in etf_event.candidates.refusals}
    assert refused <= {r.expression_id for r in records if r.candidate_signal_id is None}
    assert all(r.outcome is not PathEvidenceOutcome.SUCCESS or r.evidence_scope is EvidenceScope.PORTFOLIO
               for r in records)
    unexpressed = {p.path_id for p in etf_event.discovery.paths} - {
        pid for e in etf_event.expressions.expressions for pid in e.path_ids}
    assert unexpressed <= {r.path_id for r in records}


@needs_cpp
def test_a_portfolio_result_never_marks_a_signal_path_or_mechanism_validated(etf_event, qualified, validated):
    screens, ranked, plan = qualified
    passed = validated.model_copy(update={"status": PortfolioValidationStatus.VALIDATED})
    records = build_signal_path_evidence([etf_event], ranked=ranked, screens=screens, plan=plan, validation=passed,
                                         recorded_at="t")
    success = [r for r in records if r.outcome is PathEvidenceOutcome.SUCCESS]
    assert success and all(r.evidence_scope is EvidenceScope.PORTFOLIO
                           and r.reason_code is PathEvidenceReason.PORTFOLIO_VALIDATED for r in success)
    for memory in summarize_path_memory(records):
        assert memory.blocked is False and "Nothing here proves" in memory.scope_note
        if MemoryGuidance.PORTFOLIO_EVIDENCE_FOR in memory.guidance:
            assert memory.scopes.get("PORTFOLIO", 0) > 0
    assert all("proven" not in r.detail.lower() or "never" in r.detail.lower() for r in records)


def test_one_failure_never_blacklists_another_route_of_the_same_event():
    failed = _record(path_signature="route-a", reason_code=PathEvidenceReason.SCREEN_CONTRADICTS_SIGN)
    other = _record(path_signature="route-b", outcome=PathEvidenceOutcome.UNRESOLVED,
                    evidence_scope=EvidenceScope.HYPOTHESIS, reason_code=PathEvidenceReason.MEASUREMENT_UNAVAILABLE,
                    stage_reached=HypothesisStage.MEASUREMENT)
    memories = {m.path_signature: m for m in summarize_path_memory([failed, other])}
    assert memories["route-b"].reasons == {"MEASUREMENT_UNAVAILABLE": 1}
    assert MemoryGuidance.SCREEN_EVIDENCE_AGAINST not in memories["route-b"].guidance
    assert memories["route-b"].guidance == (MemoryGuidance.RETRY_WHEN_DATA_EXISTS,)
    mixed = summarize_path_memory([failed, _record(path_signature="route-a", event_id="e2"),
                                   _record(path_signature="route-a", event_id="e4"),
                                   _record(path_signature="route-a", event_id="e3", stage_reached=HypothesisStage.MEASUREMENT,
                                           reason_code=PathEvidenceReason.MEASUREMENT_UNAVAILABLE,
                                           evidence_scope=EvidenceScope.HYPOTHESIS)])[0]
    assert mixed.primary_reason is PathEvidenceReason.SCREEN_NO_SUPPORT  # 2 of 4 records
    assert mixed.guidance[0] is MemoryGuidance.SCREEN_EVIDENCE_AGAINST  # the primary reason's guidance leads
    assert all(not m.blocked for m in memories.values())


def test_memory_accumulates_a_route_across_events_and_splits_by_expression_domain(tmp_path):
    a = _record(event_id="ai", research_run_id="r1", expression_domain="EQUITY", stage_reached=HypothesisStage.MEASUREMENT,
                reason_code=PathEvidenceReason.MEASUREMENT_UNAVAILABLE, evidence_scope=EvidenceScope.HYPOTHESIS)
    b = _record(event_id="opec", research_run_id="r2", expression_domain="FUTURES")
    with ExperimentRegistry(tmp_path / "reg.sqlite") as reg:
        reg.record_signal_path_evidence([a, b])
        (memory,) = signal_path_memory(reg, event_id="ai")  # the route, summarized over every event
    assert memory.event_ids == ("ai", "opec") and memory.research_runs == 2
    domains = {d.domain: d for d in memory.by_expression_domain}
    assert domains["EQUITY"].furthest_stage is HypothesisStage.MEASUREMENT
    assert domains["FUTURES"].furthest_stage is HypothesisStage.SCREEN
    assert memory.furthest_stage is HypothesisStage.SCREEN
    chain = lineage(a)
    assert chain[0].kind == "EVENT" and chain[-1].kind == "EVIDENCE"
    assert chain[6].label == "no candidate"


def test_an_evidence_plane_instrument_navigates_back_to_its_hypotheses(tmp_path):
    from alpha_agent.alpha_graph.hypothesis_links import hypotheses_for_instrument

    r = _record(candidate_signal_id="csig2:x", instrument="XLK", expression_domain="ETF")
    with ExperimentRegistry(tmp_path / "reg.sqlite") as reg:
        reg.record_signal_path_evidence([r, _record(event_id="e9", candidate_signal_id="csig2:z", instrument="GLD",
                                                    expression_domain="ETF")])
        (link,) = hypotheses_for_instrument(reg, InstrumentRef(root_symbol="XLK", asset_domain="ETF"))
        assert link.candidate_signal_id == "csig2:x" and link.event_ids == ("e1",)
        assert hypotheses_for_instrument(reg, InstrumentRef(root_symbol="NQ", asset_domain="FUTURES")) == ()


def test_hypothesis_links_carry_ids_only_between_planes():
    r = _record(candidate_signal_id="csig2:x", instrument="XLK", expression_domain="ETF")
    q = _record(candidate_signal_id="csig2:y", instrument="HBM_MAKERS", expression_domain="EQUITY")
    links = {link.candidate_signal_id: link for link in link_records([r, q])}
    assert links["csig2:x"].instrument == InstrumentRef(root_symbol="XLK", asset_domain="ETF")
    assert links["csig2:y"].instrument is None  # not an evidence-plane node
    fields = set(type(links["csig2:x"]).model_fields)
    assert not fields & {"verdict", "coverage", "maturity", "scientific_evidence", "score"}


# ---------------------------------------------------------------------------
# path-level study hooks: never assumed, testable only on a real event sample
# ---------------------------------------------------------------------------


def test_path_studies_are_not_yet_testable_on_this_platform():
    caps = probe_study_capabilities()
    for design in default_designs():
        result = assess_study(design, caps)
        assert result.status is StudyReadiness.NOT_YET_TESTABLE
        assert StudyRequirement.HISTORICAL_EVENT_SAMPLE in {b.requirement for b in result.blockers}


def test_a_dated_event_sample_alone_does_not_make_a_study_testable():
    class Store:
        def __init__(self, dated):
            self.dated = dated

        def list_recent(self, **_):
            return [type("Item", (), {"published_at": d}) for d in self.dated]

        def list_upcoming(self, **_):
            return [type("Event", (), {"scheduled_at": d}) for d in self.dated]

    from datetime import UTC, datetime

    dated = [datetime(2021, 3, 4, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)]
    caps = probe_study_capabilities(news_store=Store(dated), event_store=Store(dated))
    assert caps.available[StudyRequirement.HISTORICAL_EVENT_SAMPLE]
    assert "2 news item" not in caps.basis[StudyRequirement.HISTORICAL_NEWS_TEXT]  # only the in-span item counts
    direct = assess_study(default_designs()[0], caps)
    assert direct.status is StudyReadiness.NOT_YET_TESTABLE
    assert {b.requirement for b in direct.blockers} >= {StudyRequirement.EXPOSURE_MAGNITUDE,
                                                        StudyRequirement.TRANSMISSION_WEIGHT}
    everything = StudyCapabilities(available=dict.fromkeys(StudyRequirement, True), basis={}, probed_on=date(2026, 9, 27))
    assert assess_study(default_designs()[2], everything).status is StudyReadiness.TESTABLE


def test_arms_share_one_predeclared_family_and_llm_hindsight_blocks_arm_b():
    design = ComparisonDesign(question=StudyQuestion.MECHANISM_VS_SENTIMENT, arms=tuple(StudyArm))
    assert design.declared_family_size == 5 * 4
    blockers = {b.requirement: b for b in assess_study(design, probe_study_capabilities()).blockers}
    assert blockers[StudyRequirement.LLM_WITHOUT_HINDSIGHT].arms == (StudyArm.B_DIRECT_LLM_DIRECTION,)
    assert "prospectively" in blockers[StudyRequirement.LLM_WITHOUT_HINDSIGHT].basis


def test_unconditional_factor_diagnostics_are_refused_as_event_conditioned_evidence():
    with pytest.raises(ValueError, match="not event-conditioned evidence"):
        EventConditionedEvidenceRef(scope="UNCONDITIONAL_FACTOR", source_id="candscreen1:x", event_ids=("e",))
    assert EventConditionedEvidenceRef(scope="EVENT_CONDITIONED", source_id="s", event_ids=("e",))


# ---------------------------------------------------------------------------
# commission schedule through the real replay CLI (acceptance patch)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed_handoff():
    """A Phase G plan holding futures contracts AND ETF shares."""
    if ALLOCATOR is None or REPLAY is None:
        pytest.skip("the C++ portfolio CLIs are not built")
    from alpha_agent.portfolio.handoff import build_handoff
    from test_news_alpha_phase_f import _candidates
    from test_news_alpha_phase_f import _screens as f_screens
    from test_news_alpha_phase_g import FRAMES as G_FRAMES
    from test_news_alpha_phase_g import SHORTING as G_SHORTING
    from test_news_alpha_phase_g import _plan as g_plan
    from test_news_alpha_phase_g import _supported

    ai = _candidates()
    plan = g_plan(ai, _supported(f_screens(ai, G_FRAMES)), G_SHORTING)
    handoff = build_handoff(plan)
    roots = {r.root_symbol for r in handoff.rows if r.target_units}
    assert "NQ" in roots and any(r.startswith("E") for r in roots)  # a genuinely mixed book
    return plan, handoff


def _replay(plan, handoff, tmp_path, **kw):
    from alpha_agent.portfolio.handoff import replay_handoff
    from test_news_alpha_phase_g import _write_replay_fixture

    tmp_path.mkdir(parents=True, exist_ok=True)
    bars, contracts = _write_replay_fixture(plan, tmp_path)
    return replay_handoff(handoff, bars, contracts, tmp_path / "run", cli=REPLAY, **kw)


def test_a_mixed_book_is_charged_per_contract_and_per_share_by_the_engine(mixed_handoff, tmp_path):
    plan, handoff = mixed_handoff
    roots = {r.root_symbol for r in handoff.rows}
    schedule = {r: 2.0 if r == "NQ" else 0.005 for r in roots}
    result = _replay(plan, handoff, tmp_path, commission_per_contract_usd=0.0, commission_schedule=schedule)
    expected = sum(abs(r.target_units) * schedule[r.root_symbol] for r in handoff.rows)
    assert result.costs_usd == pytest.approx(expected, rel=1e-12)
    assert {c.root_symbol: c.commission_per_unit_usd for c in result.commission_schedule} == schedule
    legacy = _replay(plan, handoff, tmp_path / "legacy", commission_per_contract_usd=2.0)
    assert legacy.costs_usd == pytest.approx(2.0 * sum(abs(r.target_units) for r in handoff.rows), rel=1e-12)
    assert legacy.costs_usd > result.costs_usd  # the scalar charged every share a contract rate


def test_a_uniform_schedule_reproduces_the_legacy_scalar_exactly(mixed_handoff, tmp_path):
    plan, handoff = mixed_handoff
    legacy = _replay(plan, handoff, tmp_path / "a", commission_per_contract_usd=2.0)
    uniform = _replay(plan, handoff, tmp_path / "b", commission_per_contract_usd=2.0,
                      commission_schedule={r.root_symbol: 2.0 for r in handoff.rows})
    assert legacy.commission_schedule == ()
    assert uniform.model_dump(exclude={"commission_schedule"}) == legacy.model_dump(exclude={"commission_schedule"})


def test_the_engine_refuses_a_schedule_missing_a_traded_root(mixed_handoff, tmp_path):
    plan, handoff = mixed_handoff
    with pytest.raises(RuntimeError, match="commission schedule incomplete"):
        _replay(plan, handoff, tmp_path, commission_per_contract_usd=2.0, commission_schedule={"NQ": 2.0})


@needs_cpp
def test_a_qualified_plan_is_screen_eligibility_never_validation(etf_event, qualified):
    """11A: QUALIFIED means 'screen-supported, may enter validation' -- the gate
    passes it on, and until a validation actually runs every member stays
    UNRESOLVED screening evidence, never a success or a verdict."""
    screens, ranked, plan = qualified
    assert plan.eligibility is EligibilityMode.QUALIFIED and plan.status is PlanStatus.CONSTRUCTED
    from alpha_agent.portfolio.validation import gate_portfolio

    assert gate_portfolio(plan, ranked, screens) is None  # eligible -- not validated
    assert not {"verdict", "validated", "headline_verdict"} & set(type(plan).model_fields)
    records = build_signal_path_evidence([etf_event], ranked=ranked, screens=screens, plan=plan, validation=None,
                                         recorded_at="t")
    members = [r for r in records if r.candidate_signal_id in {s.candidate_signal_id for s in plan.selected}]
    assert members and all(r.outcome is PathEvidenceOutcome.UNRESOLVED and r.evidence_scope is EvidenceScope.SCREENING
                           for r in members)
