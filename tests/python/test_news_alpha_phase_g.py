"""News Alpha Phase G -- RankedSignalSet -> PORTFOLIO CONSTRUCTION -> PortfolioPlan
-> the existing C++ execution boundary (`alpha_agent.portfolio`).

Candidates come from the same offline pipeline as Phases E/F; screens are
computed by the real Phase E engine from in-memory bars labelled REAL (a
fixture, so REAL-only rules can be exercised) and pinned where a test needs
a particular screen outcome. Market snapshots are built from the same bars.
Sizing always runs through the real compiled C++ allocator -- these tests
skip, never fall back, when it is not built.
"""
from __future__ import annotations

import ast
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from alpha_agent.crypto.provenance import DataProvenanceRole
from alpha_agent.news_alpha import MandateDomain, ResearchMandate
from alpha_agent.portfolio import (
    EligibilityMode,
    PlanStatus,
    PortfolioConstructionPolicy,
    construct_portfolio_plan,
    resolve_constraints,
)
from alpha_agent.portfolio.allocator import AllocatorInputs, find_allocator_cli, run_allocator
from alpha_agent.portfolio.classification import RiskAssetClass, classify_instrument
from alpha_agent.portfolio.handoff import build_handoff, replay_handoff
from alpha_agent.portfolio.policy import ConstraintSource
from alpha_agent.portfolio.risk_model import (
    DISCOVERY_END,
    InstrumentMarketSnapshot,
    InsufficientRiskData,
    MarketSnapshotStore,
    _returns,
    estimate_risk_model,
)
from alpha_agent.portfolio.selection import (
    REJECTION_TEXT,
    RejectionReason,
    RejectionStage,
    resolve_directions,
    screen_eligibility,
)
from alpha_agent.recommendation.profile import InvestorProfile, RiskStyle
from alpha_agent.recommendation.signal_ranking import RankedSignalSet, rank_candidate_signals
from alpha_agent.screening.factor_diagnostics import ScreenStatus

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_news_alpha_phase_f import _candidates, _frame, _pinned, _screens

REPO_ROOT = Path(__file__).resolve().parents[2]
PORTFOLIO_PKG = REPO_ROOT / "python" / "alpha_agent" / "portfolio"
CLI = find_allocator_cli()
needs_cpp = pytest.mark.skipif(CLI is None, reason="quant_portfolio_construct_csv is not built")

MANDATE = ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES, MandateDomain.ETF))
SHORTING = MANDATE.model_copy(update={"shorting_allowed": True})
TICK = {MandateDomain.FUTURES: 0.25, MandateDomain.ETF: 0.01}


def _worlds(n: int = 1300, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Tech (NQ, QQQ, XLK) trending UP into the as-of date, utilities (XLU) DOWN."""
    rng = np.random.default_rng(seed)
    tech = rng.normal(0.0004, 0.012, n)
    tech[-90:] += 0.004
    util = rng.normal(0.0002, 0.009, n)
    util[-90:] -= 0.004
    frames = {sym: _frame(tech + rng.normal(0, 0.0015, n)) for sym in ("NQ", "QQQ", "XLK")}
    frames["XLU"] = _frame(util)
    return frames


FRAMES = _worlds()


def _snapshot(domain: MandateDomain, symbol: str, frames=None, *, volume: float = 1e6,
              role: DataProvenanceRole = DataProvenanceRole.REAL) -> InstrumentMarketSnapshot:
    f = (frames or FRAMES)[symbol]
    f = f[pd.to_datetime(f["trading_day"]).dt.date < DISCOVERY_END]
    fut = domain is MandateDomain.FUTURES
    close = [round(round(float(x) / TICK[domain]) * TICK[domain], 10) for x in f["close"]]  # on the tick grid
    return InstrumentMarketSnapshot(
        domain=domain, symbol=symbol, execution_root=symbol if fut else f"E{symbol}", dataset="FIXTURE",
        unit="contract" if fut else "share", multiplier=20.0 if fut else 1.0, tick_size=TICK[domain],
        adjustment="ADDITIVE" if fut else "MULTIPLICATIVE",
        volume_scope="FRONT_CONTRACT" if fut else "PRIMARY_LISTING_ONLY", data_role=role,
        days=tuple(f["trading_day"]), ts_event_ns=tuple(int(x) for x in f["ts_event_ns"]),
        research_close=tuple(close), raw_close=tuple(close), volume=tuple(volume for _ in close),
        traded_symbol=tuple((f"{symbol}H3" if fut else symbol) for _ in close), provenance=("test fixture",),
    )


def _provider(frames=None, **kw):
    return lambda domain, symbol: _snapshot(domain, symbol, frames, **kw)


@pytest.fixture(scope="module")
def ai():
    return _candidates()


@pytest.fixture(scope="module")
def screens(ai):
    return _screens(ai, FRAMES)


def _supported(screens, t: float = 2.5, only: set[str] | None = None):
    return {k: (_pinned(v, status=ScreenStatus.SCREEN_CONTINUE, t=t)
                if only is None or v.series.instrument in only else v) for k, v in screens.items()}


def _rank(ai, screens, mandate=MANDATE) -> RankedSignalSet:
    return rank_candidate_signals(ai, screens, mandate)


def _plan(ai, screens, mandate=MANDATE, **kw):
    return construct_portfolio_plan(_rank(ai, screens, mandate), screens, mandate, snapshots=_provider(), **kw)


# ---------------------------------------------------------------------------
# constraints: where every limit comes from
# ---------------------------------------------------------------------------


def test_unstated_leverage_and_capital_resolve_to_labelled_conservative_defaults():
    c = resolve_constraints(MANDATE, PortfolioConstructionPolicy())
    assert c.max_gross_leverage == 1.0 and c.capital_usd == 1_000_000
    by_name = {x.name: x for x in c.limits}
    assert by_name["Gross leverage"].source is ConstraintSource.DEFAULT_FOR_UNSTATED
    assert "unlevered" in by_name["Gross leverage"].note
    assert by_name["Capital"].source is ConstraintSource.DEFAULT_FOR_UNSTATED
    assert c.allowed_domains == (MandateDomain.FUTURES, MandateDomain.ETF)  # equities: no bars to size with

    stated = MANDATE.model_copy(update={"max_gross_leverage": 2.5, "risk_profile": InvestorProfile(
        risk_style=RiskStyle.CONSERVATIVE, approximate_capital_usd=250_000, max_contracts=2)})
    c = resolve_constraints(stated, PortfolioConstructionPolicy())
    assert (c.max_gross_leverage, c.capital_usd, c.target_annual_vol, c.max_futures_contracts) == (2.5, 250_000, 0.06, 2)
    assert {x.name: x.source for x in c.limits}["Gross leverage"] is ConstraintSource.MANDATE


def test_the_risk_taxonomy_never_guesses():
    assert classify_instrument(MandateDomain.FUTURES, "ZN").asset_class is RiskAssetClass.RATES
    assert classify_instrument(MandateDomain.ETF, "XLE").sector == "US_ENERGY_EQUITY"
    assert classify_instrument(MandateDomain.FUTURES, "CL").sector == "ENERGY"  # crude is not energy equities
    assert classify_instrument(MandateDomain.ETF, "NOPE") is None
    assert classify_instrument(MandateDomain.EQUITY, "AAPL") is None


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def test_without_screen_support_the_qualified_plan_is_empty_and_says_why(ai, screens):
    plan = _plan(ai, screens)
    assert plan.status is PlanStatus.NO_ELIGIBLE_SIGNAL and plan.allocation is None
    assert "None has screening support yet" in plan.status_detail
    assert {r.reason for r in plan.rejected} <= {RejectionReason.NO_SCREEN_SUPPORT,
                                                 RejectionReason.CONTRADICTS_DECLARED_SIGN,
                                                 RejectionReason.NOT_RANKABLE}
    assert len(plan.rejected) == len(_rank(ai, screens).signals)  # every signal accounted for


@needs_cpp
def test_exploratory_is_an_explicit_labelled_mode_that_still_refuses_contradicted_evidence(ai, screens):
    pinned = dict(screens)
    ids = sorted(pinned)
    pinned[ids[0]] = _pinned(pinned[ids[0]], status=ScreenStatus.CONTRADICTS_EXPECTED_SIGN, t=-2.5)
    pinned[ids[1]] = _pinned(pinned[ids[1]], status=ScreenStatus.NO_SCREEN_SUPPORT, t=-0.5)
    for k in ids[2:]:
        pinned[k] = _pinned(pinned[k], status=ScreenStatus.NO_SCREEN_SUPPORT, t=0.8)
    plan = _plan(ai, pinned, SHORTING, policy=PortfolioConstructionPolicy(eligibility=EligibilityMode.EXPLORATORY))
    reasons = {r.candidate_signal_id: r.reason for r in plan.rejected}
    assert reasons[ids[0]] is RejectionReason.CONTRADICTS_DECLARED_SIGN
    assert reasons[ids[1]] is RejectionReason.EVIDENCE_AGAINST_DECLARED_SIGN
    assert plan.is_exploratory and plan.headline().startswith("Exploratory preview")
    assert plan.status is PlanStatus.CONSTRUCTED


@needs_cpp
def test_direction_is_the_factor_sign_on_the_as_of_date_times_the_declared_sign(ai, screens):
    supported = _supported(screens)
    plan = _plan(ai, supported, SHORTING)
    assert plan.as_of == date(2022, 12, 26)  # the last common discovery day of the fixture
    for s in plan.selected:
        series = dict(zip(supported[s.candidate_signal_id].series.days,
                          supported[s.candidate_signal_id].series.values, strict=True))
        value = series[plan.as_of.isoformat()]
        assert s.factor_value == value and s.direction == (1 if value > 0 else -1) * s.expected_sign
    directions = {s.instrument: s.direction for s in plan.selected}
    assert directions["XLU"] == -1 and directions["QQQ"] == 1  # the fixture's down/up trends


@needs_cpp
def test_a_short_under_a_no_shorting_mandate_is_flat_never_sized(ai, screens):
    supported = _supported(screens)
    plan = _plan(ai, supported, MANDATE)
    xlu = [r for r in plan.rejected if r.instrument == "XLU"]
    assert xlu and all(r.reason is RejectionReason.SHORT_NOT_PERMITTED for r in xlu)
    assert "long/flat form is flat today" in xlu[0].detail
    assert all(i.units >= 0 for i in plan.allocation.instruments)
    shorting = _plan(ai, supported, SHORTING)
    assert shorting.allocation.instrument("ETF:XLU").units < 0


@needs_cpp
def test_rank_decides_priority_never_weight(ai, screens):
    supported = _supported(screens)
    one = _plan(ai, supported, SHORTING, policy=PortfolioConstructionPolicy(max_exposures=1))
    assert min(s.rank for s in one.selected) == 1 and len({s.exposure_group_id for s in one.selected}) == 1
    assert any(r.reason is RejectionReason.EXPOSURE_LIMIT for r in one.rejected)

    plan = _plan(ai, supported, SHORTING)
    # Inside one exposure every member leg carries the same standalone risk, whatever its rank.
    for group in {s.exposure_group_id for s in plan.selected}:
        members = [s for s in plan.selected if s.exposure_group_id == group]
        risk = [abs(plan.allocation.signal(s.candidate_signal_id).target_weight)
                * plan.risk_model.instrument(s.instrument_key).annual_vol for s in members]
        assert max(risk) == pytest.approx(min(risk), rel=1e-9)
    assert any(len([s for s in plan.selected if s.exposure_group_id == g]) > 1
               for g in {s.exposure_group_id for s in plan.selected})


def test_a_negative_declared_sign_flips_the_factor_direction(ai, screens):
    supported = _supported(screens)
    ranked = _rank(ai, supported, SHORTING)
    eligible, _ = screen_eligibility(ranked, PortfolioConstructionPolicy())
    s = eligible[0]
    value = dict(zip(supported[s.candidate_signal_id].series.days, supported[s.candidate_signal_id].series.values,
                     strict=True))["2022-12-26"]
    flipped = s.model_copy(update={"expected_sign": -s.expected_sign})
    (a,), _ = resolve_directions([s], supported, as_of=date(2022, 12, 26), mandate=SHORTING, ranked=ranked)
    (b,), _ = resolve_directions([flipped], supported, as_of=date(2022, 12, 26), mandate=SHORTING, ranked=ranked)
    assert a.direction == -b.direction == (1 if value > 0 else -1) * s.expected_sign


@needs_cpp
def test_alternates_share_their_exposures_budget_and_exposures_get_equal_risk(ai, screens):
    plan = _plan(ai, _supported(screens), SHORTING,
                 policy=PortfolioConstructionPolicy(max_exposure_risk_share=1.0, max_instrument_gross_share=1.0,
                                                    max_sector_gross_share=1.0, max_asset_class_gross_share=1.0))
    a = plan.allocation
    clusters = [c for c in a.clusters if c.allocated]
    assert len(clusters) == 2  # tech complex (NQ/QQQ/XLK and alternates) + utilities
    assert clusters[0].target_risk_contribution == pytest.approx(clusters[1].target_risk_contribution, rel=1e-9)
    tech = next(c for c in clusters if len(c.signal_ids) > 1)
    assert len(tech.signal_ids) <= plan.policy.max_signals_per_exposure


# ---------------------------------------------------------------------------
# sizing is the C++ allocator's, in each asset class's own unit
# ---------------------------------------------------------------------------


@needs_cpp
def test_futures_contracts_and_etf_shares_are_sized_in_their_own_units(ai, screens):
    plan = _plan(ai, _supported(screens), SHORTING)
    a = plan.allocation
    nq, qqq = a.instrument("FUTURES:NQ"), a.instrument("ETF:QQQ")
    assert nq.multiplier == 20.0 and qqq.multiplier == 1.0
    assert nq.unit_notional_usd == pytest.approx(nq.price * 20.0) and qqq.unit_notional_usd == qqq.price
    for i in a.instruments:
        assert isinstance(i.units, int)
        assert abs(i.executable_notional_usd) <= abs(i.target_notional_usd) + 1e-6  # truncation never exceeds
        assert i.executable_notional_usd == pytest.approx(i.units * i.unit_notional_usd)


@needs_cpp
def test_every_size_in_the_plan_is_the_cpp_output_verbatim(ai, screens):
    plan = _plan(ai, _supported(screens), SHORTING)
    again = run_allocator(plan.allocator_inputs)
    assert again == plan.allocation  # byte-identical re-run from the recorded inputs


def test_python_never_computes_a_notional_or_a_unit():
    """No module that builds or hands off a plan multiplies by a price or a
    multiplier -- sizing lives in C++ only (risk_model.py multiplies volume by
    price for liquidity, a statistic, and is the one exemption)."""
    offenders = []
    for path in PORTFOLIO_PKG.glob("*.py"):
        if path.name == "risk_model.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div, ast.FloorDiv)):
                text = ast.unparse(node)
                if any(w in text for w in ("price", "multiplier", "unit_notional", "capital")):
                    offenders.append(f"{path.name}: {text}")
    assert offenders == []


@needs_cpp
def test_the_mandates_liquidity_requirement_refuses_a_thin_instrument(ai, screens):
    thin = {"XLU"}

    def provider(domain, symbol):
        return _snapshot(domain, symbol, volume=10.0 if symbol in thin else 1e6)

    plan = construct_portfolio_plan(_rank(ai, _supported(screens), SHORTING), _supported(screens), SHORTING,
                                    snapshots=provider)
    xlu = [r for r in plan.rejected if r.instrument == "XLU"]
    assert xlu and all(r.stage is RejectionStage.ALLOCATION and r.reason is RejectionReason.INSTRUMENT_NOT_ADMITTED
                       for r in xlu)
    assert "below min liquidity" in xlu[0].detail
    assert plan.allocation.instrument("ETF:XLU").units == 0


@needs_cpp
def test_the_plan_is_deterministic(ai, screens):
    supported = _supported(screens)
    a = _plan(ai, supported, SHORTING)
    b = _plan(ai, supported, SHORTING)
    assert a.fingerprint() == b.fingerprint() and a.fingerprint().startswith("portplan1:")


def test_a_missing_allocator_is_a_typed_status_never_a_python_fallback(ai, screens):
    plan = _plan(ai, _supported(screens), SHORTING, allocator_cli=Path("/nonexistent/quant_portfolio_construct_csv"))
    assert plan.status is PlanStatus.ALLOCATOR_UNAVAILABLE and plan.allocation is None
    assert plan.risk_model is not None and plan.allocator_inputs is not None  # everything up to sizing is kept


def test_a_ranked_set_from_another_mandate_is_refused(ai, screens):
    with pytest.raises(ValueError, match="different mandate"):
        construct_portfolio_plan(_rank(ai, screens, MANDATE), screens, SHORTING, snapshots=_provider())


# ---------------------------------------------------------------------------
# risk model
# ---------------------------------------------------------------------------


def test_futures_returns_are_exact_across_a_roll():
    snap = _snapshot(MandateDomain.FUTURES, "NQ")
    raw, research = [100.0, 101.0, 111.0, 112.0], [100.0, 101.0, 101.0, 102.0]  # +10 roll gap on day 3
    frame = pd.DataFrame({"raw_close": raw, "research_close": research})
    r = _returns(snap, frame)
    assert r.iloc[1] == pytest.approx(0.01) and r.iloc[2] == 0.0 and r.iloc[3] == pytest.approx(1 / 111)


def test_risk_model_statistics_and_refusals():
    snaps = {s.key: s for s in (_snapshot(MandateDomain.FUTURES, "NQ"), _snapshot(MandateDomain.ETF, "QQQ"),
                                _snapshot(MandateDomain.ETF, "XLU"))}
    risk = estimate_risk_model(snaps, as_of=date(2022, 12, 26))
    assert risk.n_observations == 252 and risk.correlation("ETF:QQQ", "FUTURES:NQ") > 0.95
    assert abs(risk.correlation("ETF:QQQ", "ETF:XLU")) < 0.3
    nq = risk.instrument("FUTURES:NQ")
    assert 0.15 < nq.annual_vol < 0.30 and nq.adv_usd == pytest.approx(1e6 * np.median(
        [c for c in snaps["FUTURES:NQ"].raw_close[-252:]]) * 20.0, rel=1e-9)
    with pytest.raises(InsufficientRiskData):
        estimate_risk_model(snaps, as_of=date(2023, 1, 3))  # the validation window
    with pytest.raises(InsufficientRiskData):
        estimate_risk_model(snaps, as_of=date(2022, 12, 26), min_observations=10_000)
    synthetic = {"ETF:XLU": _snapshot(MandateDomain.ETF, "XLU", role=DataProvenanceRole.SYNTHETIC)}
    with pytest.raises(InsufficientRiskData, match="only real data"):
        estimate_risk_model(synthetic, as_of=date(2022, 12, 26))


def test_a_day_one_venue_missed_is_absorbed_never_filled():
    """The joint panel keeps only days every instrument traded: a return then
    spans the same interval for all -- identical to dropping the day
    everywhere, never a forward-filled price."""
    nq, xlu = _snapshot(MandateDomain.FUTURES, "NQ"), _snapshot(MandateDomain.ETF, "XLU")
    gap = xlu.days[-40]

    def without(snap: InstrumentMarketSnapshot, day: str) -> InstrumentMarketSnapshot:
        keep = [i for i, d in enumerate(snap.days) if d != day]
        cols = ("days", "ts_event_ns", "research_close", "raw_close", "volume", "traded_symbol")
        return snap.model_copy(update={c: tuple(getattr(snap, c)[i] for i in keep) for c in cols})

    as_of = date(2022, 12, 26)
    partial = estimate_risk_model({nq.key: nq, xlu.key: without(xlu, gap)}, as_of=as_of)
    dropped = estimate_risk_model({nq.key: without(nq, gap), xlu.key: without(xlu, gap)}, as_of=as_of)
    assert partial.instrument(nq.key).annual_vol == dropped.instrument(nq.key).annual_vol
    assert partial.correlation(nq.key, xlu.key) == dropped.correlation(nq.key, xlu.key)
    assert partial.n_observations == 252


def test_a_snapshot_cannot_hold_validation_or_holdout_days():
    f = _frame(np.zeros(5), start="2022-12-28")
    with pytest.raises(ValueError, match="discovery-window days only"):
        InstrumentMarketSnapshot(
            domain=MandateDomain.ETF, symbol="QQQ", execution_root="EQQQ", dataset="X", unit="share", multiplier=1.0,
            tick_size=0.01, adjustment="MULTIPLICATIVE", volume_scope="PRIMARY_LISTING_ONLY",
            data_role=DataProvenanceRole.REAL, days=tuple(f["trading_day"]),
            ts_event_ns=tuple(int(x) for x in f["ts_event_ns"]), research_close=tuple(f["close"]),
            raw_close=tuple(f["close"]), volume=tuple(f["volume"]), traded_symbol=("QQQ",) * 5, provenance=())


def test_the_snapshot_store_caches_real_data_only(tmp_path):
    store = MarketSnapshotStore(tmp_path)
    calls = []

    def loader(domain, symbol):
        calls.append(symbol)
        return _snapshot(domain, symbol)

    get = store.provider(loader)
    assert get(MandateDomain.ETF, "QQQ") == get(MandateDomain.ETF, "QQQ") and calls == ["QQQ"]
    with pytest.raises(ValueError):
        store.save(_snapshot(MandateDomain.ETF, "XLU", role=DataProvenanceRole.SYNTHETIC))


def test_too_little_history_is_a_typed_status(ai, screens):
    plan = _plan(ai, _supported(screens), SHORTING, policy=PortfolioConstructionPolicy(min_risk_observations=5000))
    assert plan.status is PlanStatus.RISK_INPUTS_UNAVAILABLE and "joint daily returns" in plan.status_detail


# ---------------------------------------------------------------------------
# the existing execution boundary
# ---------------------------------------------------------------------------


def _write_replay_fixture(plan, directory: Path) -> tuple[Path, Path]:
    """Contracts for every targeted root + the decision bar and ONE next bar per
    instrument, opening (and closing) exactly at the plan's reference price."""
    contracts = [("instrument_id,raw_symbol,root_symbol,exchange,tick_size,multiplier,activation_ns,"
                  "expiration_ns,first_notice_ns,last_trade_ns")]
    bars = []
    handoff = build_handoff(plan)
    for n, row in enumerate(handoff.rows, start=1):
        i = plan.allocation.instrument(row.instrument_key)
        tick = plan.risk_model.instrument(row.instrument_key).tick_size
        contracts.append(f"{n},{row.traded_symbol},{row.root_symbol},XTST,{tick},{i.multiplier},1,"
                         f"{4_102_444_800_000_000_000},,")
        for ts in (row.ts_event_ns, row.ts_event_ns + 86_400_000_000_000):
            bars.append((ts, n, i.price))
    bars.sort()
    (directory / "contracts.csv").write_text("\n".join(contracts) + "\n")
    lines = ["ts_event_ns,instrument_id,open,high,low,close,volume"] + [
        f"{ts},{n},{px!r},{px!r},{px!r},{px!r},1000000" for ts, n, px in bars]
    (directory / "bars.csv").write_text("\n".join(lines) + "\n")
    return directory / "bars.csv", directory / "contracts.csv"


@needs_cpp
def test_the_plan_replays_through_the_unchanged_engine_and_the_accountant_agrees(ai, screens, tmp_path):
    plan = _plan(ai, _supported(screens), SHORTING)
    assert plan.status is PlanStatus.CONSTRUCTED and len(plan.positions) >= 2
    handoff = build_handoff(plan)
    assert handoff.plan_fingerprint == plan.fingerprint()
    assert all("." not in r.root_symbol for r in handoff.rows)  # roots, never contracts
    bars, contracts = _write_replay_fixture(plan, tmp_path)
    result = replay_handoff(handoff, bars, contracts, tmp_path / "run")
    assert result.risk_rejects == 0 and result.risk_resizes == 0
    held = {p.root_symbol: p for p in result.positions}
    for i in plan.positions:
        assert held[i.root_symbol].units == i.units
        assert held[i.root_symbol].signed_notional_usd == pytest.approx(i.executable_notional_usd, rel=1e-12)
    assert result.gross_exposure_usd == pytest.approx(
        sum(abs(i.executable_notional_usd) for i in plan.positions), rel=1e-12)
    assert result.gross_leverage == pytest.approx(plan.allocation.executable.gross_exposure, rel=1e-12)


@needs_cpp
def test_the_engines_hard_gate_rechecks_the_plan_independently(ai, screens, tmp_path):
    plan = _plan(ai, _supported(screens), SHORTING)
    handoff = build_handoff(plan)
    tighter = handoff.model_copy(update={"risk_policy": handoff.risk_policy.model_copy(
        update={"max_gross_leverage": plan.allocation.executable.gross_exposure / 3})})
    bars, contracts = _write_replay_fixture(plan, tmp_path)
    result = replay_handoff(tighter, bars, contracts, tmp_path / "run")
    assert result.risk_resizes + result.risk_rejects > 0  # the engine, not the plan, has the last word
    assert result.gross_leverage <= plan.allocation.executable.gross_exposure / 3 + 1e-9


@needs_cpp
def test_exposures_that_hedge_exactly_are_a_typed_status_not_a_crash(ai, screens):
    """Two exposures that hedge each other exactly (e.g. two signal families on
    one instrument, which Phase F's family-aware grouping keeps apart) have no
    equal-risk-contribution solution: DEGENERATE_RISK_MODEL, never an error."""
    inputs = AllocatorInputs(
        instruments_csv=("instrument_key,domain,root_symbol,asset_class,sector,price,multiplier,annual_vol,adv_usd,"
                         "max_units,decision_ts_ns\nFUTURES:NQ,FUTURES,NQ,EQUITY,US_GROWTH_EQUITY,11000.0,20.0,0.25,"
                         "1e12,0,1\n"),
        signals_csv=("signal_id,instrument_key,cluster_id,direction,priority\n"
                     "mom,FUTURES:NQ,momentum,1,1\nrev,FUTURES:NQ,reversion,-1,2\n"),
        correlations_csv="instrument_a,instrument_b,correlation\n",
        limits_csv=("capital_usd,target_annual_vol,max_gross_leverage,max_net_exposure,max_instrument_gross_share,"
                    "max_sector_gross_share,max_asset_class_gross_share,max_cluster_risk_share,max_adv_participation,"
                    "min_adv_usd,max_turnover,shorting_allowed,allowed_domains\n"
                    "1000000.0,0.1,1.0,0.0,0.4,0.6,0.8,0.5,0.01,0.0,0.0,true,FUTURES|ETF\n"),
    )
    out = run_allocator(inputs)
    assert out.status == "DEGENERATE_RISK_MODEL"
    assert {s.reason for s in out.signals} == {"DEGENERATE_RISK_MODEL"}
    assert all(i.units == 0 for i in out.instruments)
    for status in ("CONSTRUCTED", "NO_EXECUTABLE_POSITION", "NO_ALLOCATABLE_SIGNAL", "DEGENERATE_RISK_MODEL"):
        assert PlanStatus(status)  # every allocator status is a typed plan status
    assert RejectionReason("DEGENERATE_RISK_MODEL") in REJECTION_TEXT
