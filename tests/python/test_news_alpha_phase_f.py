"""News Alpha Phase F -- candidate signals + factor diagnostics -> MULTI-ASSET
SIGNAL RANKING (`alpha_agent.recommendation.signal_ranking`).

Candidates come from the same offline pipeline as Phase E (a fixed
`DomainCapabilities` snapshot, fixed timestamps). Screens are computed by the
real Phase E engine from in-memory bars whose `data_role` is labelled REAL --
a test fixture, so the ranker's REAL-only rule can be exercised -- except one
real-bytes test on two already-acquired ETFs that skips when the raw store is
absent. Tests that pin a particular screen outcome copy a computed screen and
change only the fields the ranker reads.
"""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from alpha_agent.core.instrument import InstrumentIdentity
from alpha_agent.crypto.provenance import DataProvenanceRole
from alpha_agent.news_alpha import (
    CandidateSignal,
    CandidateSignalSet,
    DomainCapabilities,
    ExpressionFidelity,
    FuturesBarCoverage,
    MandateDomain,
    PathStatus,
    ResearchMandate,
    SignalHorizon,
    UserDescribedEvent,
    build_asset_expressions,
    build_candidate_signals,
    build_economic_mechanism_graph,
    discover_signal_paths,
    resolve_allowed_universe,
    scan_initial_impact,
)
from alpha_agent.news_alpha.candidate_signals import candidate_signal_id, factor_identity
from alpha_agent.recommendation.fit import PersonalizationState, frequency_band
from alpha_agent.recommendation.profile import HoldingPeriod, InvestorProfile, TradingFrequency
from alpha_agent.recommendation.signal_ranking import (
    CostHeadroom,
    DataQualityGrade,
    EvidenceDirection,
    ExclusionReason,
    FitDimension,
    FitStatus,
    NotRankableReason,
    RankedSignalSet,
    RankingTier,
    RedundancyRelation,
    SignalRankingPolicy,
    SignalRole,
    StabilityGrade,
    UncertaintyCode,
    normalize_screen_evidence,
    rank_candidate_signals,
)
from alpha_agent.registry.enums import AssetDomain
from alpha_agent.schemas.market_data import PriceDomain
from alpha_agent.screening.candidate_signal_screen import (
    SCREENABLE_DOMAINS,
    CandidateScreen,
    DailyBars,
    screen_candidate,
)
from alpha_agent.screening.factor_diagnostics import FactorScreenPolicy, ScreenStatus

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "python" / "alpha_agent" / "recommendation" / "signal_ranking.py"
REGISTRY_PATH = REPO_ROOT / "data" / "registry" / "experiments.sqlite"
_ETF_RAW = REPO_ROOT / "data" / "raw" / "databento" / "ARCX.PILLAR"

CAPS = DomainCapabilities(
    etf_market_data_acquired=True, equity_market_data_acquired=False,
    futures_bars=tuple(
        FuturesBarCoverage(root=r, dataset="GLBX.MDP3", data_schema="ohlcv-1m", start=date(2018, 1, 1),
                           end_exclusive=date(2025, 1, 1), continuous_segments=2, roll_overlap_artifacts=28)
        for r in ("CL", "ES", "GC", "NQ", "ZN")
    ),
)
AT = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
ALL_ASSESSABLE = ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES, MandateDomain.ETF))
AI_CAPEX = "Hyperscalers announce a major AI infrastructure spending increase, lifting data center capex guidance"
OPEC_CUT = "OPEC+ agrees a production cut of 1 million barrels per day"
D20, D60 = SignalHorizon.D20, SignalHorizon.D60


def _candidates(text: str = AI_CAPEX, mandate: ResearchMandate = ALL_ASSESSABLE) -> CandidateSignalSet:
    universe = resolve_allowed_universe(mandate, capabilities=CAPS)
    scan = scan_initial_impact(UserDescribedEvent.create(text, described_at=AT), mandate, universe=universe)
    paths = discover_signal_paths(build_economic_mechanism_graph(scan))
    return build_candidate_signals(build_asset_expressions(paths, universe, mandate), paths)


def _one(cset: CandidateSignalSet, instrument: str, horizon: SignalHorizon) -> CandidateSignal:
    (c,) = [c for c in cset.candidates if c.spec.instrument == instrument and c.spec.prediction_horizon is horizon]
    return c


# ---------------------------------------------------------------------------
# bars: one shared "tech" walk (NQ, QQQ, XLK), an independent one (XLU)
# ---------------------------------------------------------------------------


def _frame(log_returns: np.ndarray, *, start: str = "2018-01-02") -> pd.DataFrame:
    days = pd.bdate_range(start, periods=len(log_returns))
    close = 100 * np.exp(np.cumsum(log_returns))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "trading_day": days.strftime("%Y-%m-%d"), "ts_event_ns": days.asi8 + 21 * 3600 * 10**9,
        "open": open_, "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998,
        "close": close, "volume": np.full(len(close), 1e6),
    })


def _worlds(n: int = 1300, *, seed: int = 11) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    tech = rng.normal(0.0004, 0.012, n)
    frames = {sym: _frame(tech + rng.normal(0, 0.0015, n)) for sym in ("NQ", "QQQ", "XLK")}
    frames["XLU"] = _frame(rng.normal(0.0002, 0.009, n))
    return frames


def _loader(frames: dict[str, pd.DataFrame], role: DataProvenanceRole = DataProvenanceRole.REAL):
    def load(candidate: CandidateSignal) -> DailyBars:
        return DailyBars(frame=frames[candidate.spec.instrument].copy(), price_domain=PriceDomain.RAW,
                         adjustment_mode=None, data_role=role, provenance=("test fixture",))
    return load


def _screens(cset: CandidateSignalSet, frames: dict[str, pd.DataFrame] | None = None, *,
             only: set[str] | None = None, role: DataProvenanceRole = DataProvenanceRole.REAL,
             ) -> dict[str, CandidateScreen]:
    loader = _loader(frames or _worlds(), role)
    return {c.candidate_signal_id: screen_candidate(c, loader=loader, now=AT) for c in cset.candidates
            if only is None or c.spec.instrument in only}


def _pinned(screen: CandidateScreen, *, status: ScreenStatus | None = None, t: float | None = None,
            share: float | None = None, timing: float | None = None) -> CandidateScreen:
    """A copy of a computed screen with only the fields the ranker grades changed."""
    d, update = screen.diagnostics, {}
    if t is not None:
        update["horizons"] = tuple(h.model_copy(update={"ts_spearman_t": t}) if h.horizon_days == d.declared_horizon_days
                                   else h for h in d.horizons)
    if status is not None:
        update["status"] = status
    if share is not None:
        update["subperiod_sign_share"] = share
    if timing is not None:
        update["cost"] = d.cost.model_copy(update={"timing_edge_bps": timing})
    return screen.model_copy(update={"diagnostics": d.model_copy(update=update)})


@pytest.fixture(scope="module")
def ai() -> CandidateSignalSet:
    return _candidates()


@pytest.fixture(scope="module")
def ai_screens(ai) -> dict[str, CandidateScreen]:
    return _screens(ai)


def _ids(ranked: RankedSignalSet) -> list[tuple[str, int | None, SignalRole | None]]:
    return [(s.candidate_signal_id, s.rank, s.role) for s in ranked.signals]


# ---------------------------------------------------------------------------
# tiers: ranked vs not rankable vs excluded
# ---------------------------------------------------------------------------


def test_the_ai_slice_ranks_every_screened_candidate_once(ai, ai_screens):
    assert {c.spec.instrument for c in ai.candidates} == {"NQ", "QQQ", "XLK", "XLU"}
    ranked = rank_candidate_signals(ai, ai_screens, ALL_ASSESSABLE)
    assert len(ranked.ranked) == len(ai.candidates) == ranked.family_size
    assert [s.rank for s in ranked.ranked] == list(range(1, len(ai.candidates) + 1))
    assert all(s.merit is not None and s.not_rankable_reason is None for s in ranked.ranked)
    assert ranked.plane == "RESEARCH_PRIORITIZATION"
    assert ranked.mandate_fingerprint == ALL_ASSESSABLE.fingerprint()


def test_missing_evidence_is_not_low_quality(ai, ai_screens):
    nq60 = _one(ai, "NQ", D60).candidate_signal_id
    # The only screened signal CONTRADICTS its sign -- it still ranks; the unscreened are unranked, not below it.
    contradicted = _pinned(ai_screens[nq60], status=ScreenStatus.CONTRADICTS_EXPECTED_SIGN, t=-3.0)
    ranked = rank_candidate_signals(ai, {nq60: contradicted}, ALL_ASSESSABLE)
    assert [s.candidate_signal_id for s in ranked.ranked] == [nq60] and ranked.ranked[0].rank == 1
    for s in ranked.not_rankable:
        assert s.tier is RankingTier.NOT_RANKABLE and s.rank is None and s.merit is None
        assert s.not_rankable_reason is NotRankableReason.NOT_SCREENED
        assert "not low" in s.uncertainty[0].text and s.uncertainty[0].code is UncertaintyCode.NOT_EVALUATED
        assert "Run the factor diagnostics" in s.summaries.reason_for_rank
    assert ranked.signals[0].candidate_signal_id == nq60  # ranked first, the rest after -- never interleaved
    assert "not rankable" in ranked.headline()


def test_adding_unscreened_candidates_never_changes_the_ranks_of_screened_ones(ai, ai_screens):
    screened = {c.candidate_signal_id for c in ai.candidates if c.spec.instrument in {"NQ", "XLU"}}
    subset = ai.model_copy(update={"candidates": tuple(c for c in ai.candidates if c.candidate_signal_id in screened)})
    screens = {k: v for k, v in ai_screens.items() if k in screened}
    alone = rank_candidate_signals(subset, screens, ALL_ASSESSABLE)
    crowded = rank_candidate_signals((ai, _candidates(OPEC_CUT)), screens, ALL_ASSESSABLE)
    assert [(s.candidate_signal_id, s.rank, s.role) for s in alone.ranked] == [
        (s.candidate_signal_id, s.rank, s.role) for s in crowded.ranked]
    assert len(crowded.not_rankable) == crowded.family_size - len(screened)


def test_insufficient_data_and_non_real_evidence_are_not_rankable(ai, ai_screens):
    xlu20 = _one(ai, "XLU", D20)
    short = {"XLU": _worlds()["XLU"].iloc[:200].reset_index(drop=True)}
    thin = screen_candidate(xlu20, loader=_loader(short), now=AT)
    assert thin.diagnostics.status is ScreenStatus.INSUFFICIENT_DATA
    synthetic = _screens(ai, only={"NQ"}, role=DataProvenanceRole.SYNTHETIC)
    ranked = rank_candidate_signals(ai, {**synthetic, xlu20.candidate_signal_id: thin}, ALL_ASSESSABLE)
    assert not ranked.ranked
    reasons = {s.instrument + str(s.prediction_horizon_days): s.not_rankable_reason for s in ranked.signals}
    assert reasons["XLU20"] is NotRankableReason.INSUFFICIENT_EVIDENCE
    assert reasons["NQ20"] is reasons["NQ60"] is NotRankableReason.NOT_REAL_EVIDENCE
    assert "Nothing ranked yet" in ranked.headline() and "Missing evidence is not low quality" in ranked.headline()
    assert "factor/forward-return pairs" in ranked.signal(xlu20.candidate_signal_id).summaries.scientific_quality


def test_unsupported_domains_have_no_diagnostic_method_and_are_never_ranked(ai, ai_screens):
    base = _one(ai, "NQ", D20)

    def as_domain(domain: MandateDomain, symbol: str) -> CandidateSignal:
        spec = base.spec.model_copy(update={"domain": domain, "instrument": symbol})
        return CandidateSignal(candidate_signal_id=candidate_signal_id(spec), factor_identity=factor_identity(spec),
                               name=f"{symbol} fixture", spec=spec, origins=base.origins)

    extra = tuple(as_domain(d, s) for d, s in ((MandateDomain.EQUITY, "NVDA"), (MandateDomain.OPTIONS, "NVDA"),
                                               (MandateDomain.CRYPTO, "BTC")))
    assert not {c.spec.domain for c in extra} & SCREENABLE_DOMAINS
    wide = ai.model_copy(update={"candidates": ai.candidates + extra})
    every = ResearchMandate(allowed_domains=tuple(MandateDomain))
    ranked = rank_candidate_signals(wide, ai_screens, every)
    for c in extra:
        s = ranked.signal(c.candidate_signal_id)
        assert s.tier is RankingTier.NOT_RANKABLE and s.rank is None and s.merit is None
        assert s.not_rankable_reason is NotRankableReason.NO_DIAGNOSTIC_METHOD
        assert "No diagnostic method" in s.summaries.scientific_quality
    baseline = rank_candidate_signals(ai, ai_screens, every)
    assert [(s.candidate_signal_id, s.rank) for s in ranked.ranked] == [
        (s.candidate_signal_id, s.rank) for s in baseline.ranked]


def test_a_screen_filed_under_another_candidate_is_refused(ai, ai_screens):
    a, b = _one(ai, "NQ", D20).candidate_signal_id, _one(ai, "XLU", D60).candidate_signal_id
    with pytest.raises(ValueError, match="another candidate"):
        rank_candidate_signals(ai, {a: ai_screens[b]}, ALL_ASSESSABLE)


# ---------------------------------------------------------------------------
# cross-domain comparability: one peer group per diagnostic method
# ---------------------------------------------------------------------------


def test_futures_and_etf_evidence_share_one_peer_group_with_raw_metrics_preserved(ai, ai_screens):
    nq, xlk = ai_screens[_one(ai, "NQ", D20).candidate_signal_id], ai_screens[_one(ai, "XLK", D20).candidate_signal_id]
    en, ex = normalize_screen_evidence(nq), normalize_screen_evidence(xlk)
    assert en is not None and ex is not None
    assert en.peer_group == ex.peer_group and en.peer_group.key.startswith("peer1:")
    assert en.normalization_method == "factor-screen-grades/1"
    for screen, ev in ((nq, en), (xlk, ex)):
        d = screen.diagnostics
        assert ev.metrics.ts_spearman_ic == d.declared.ts_spearman_ic  # raw, never rescaled
        assert ev.metrics.ts_spearman_t == d.declared.ts_spearman_t
        assert ev.metrics.break_even_cost_bps == d.cost.break_even_cost_bps
        assert ev.aligned_t == pytest.approx(d.declared.ts_spearman_t * d.expected_sign)
        assert ev.t_threshold == d.policy.t_threshold
    ranked = rank_candidate_signals(ai, ai_screens, ALL_ASSESSABLE)
    assert len(ranked.peer_groups) == 1
    assert {s.domain for s in ranked.ranked} == {MandateDomain.FUTURES, MandateDomain.ETF}


@pytest.mark.parametrize(("t", "grade"), [
    (2.0, EvidenceDirection.SUPPORTS), (1.99, EvidenceDirection.LEANS_EXPECTED),
    (0.01, EvidenceDirection.LEANS_EXPECTED), (0.0, EvidenceDirection.LEANS_OPPOSITE),
    (-1.99, EvidenceDirection.LEANS_OPPOSITE), (-2.0, EvidenceDirection.CONTRADICTS),
])
def test_grades_use_the_screen_policys_own_threshold(ai, ai_screens, t, grade):
    screen = ai_screens[_one(ai, "NQ", D20).candidate_signal_id]
    assert screen.diagnostics.policy.t_threshold == 2.0
    ev = normalize_screen_evidence(_pinned(screen, t=t))
    assert ev is not None and ev.direction is grade


@pytest.mark.parametrize(("share", "grade"), [
    (0.6, StabilityGrade.CONSISTENT), (0.59, StabilityGrade.MIXED), (0.41, StabilityGrade.MIXED),
    (0.4, StabilityGrade.OPPOSED), (0.0, StabilityGrade.OPPOSED),
])
def test_stability_uses_the_screen_policys_own_agreement_share(ai, ai_screens, share, grade):
    ev = normalize_screen_evidence(_pinned(ai_screens[_one(ai, "NQ", D20).candidate_signal_id], share=share))
    assert ev is not None and ev.stability is grade


def test_evidence_from_another_method_or_policy_is_not_comparable(ai):
    c = _one(ai, "NQ", D20)
    other_policy = screen_candidate(c, loader=_loader(_worlds()), policy=FactorScreenPolicy(t_threshold=1.5), now=AT)
    assert normalize_screen_evidence(other_policy) is None
    screen = screen_candidate(c, loader=_loader(_worlds()), now=AT)
    other_schema = screen.model_copy(update={"diagnostics": screen.diagnostics.model_copy(
        update={"schema_version": "factor-diagnostics/1"})})
    assert normalize_screen_evidence(other_schema) is None
    for s in (other_policy, other_schema):
        ranked = rank_candidate_signals(ai, {c.candidate_signal_id: s}, ALL_ASSESSABLE)
        assert ranked.signal(c.candidate_signal_id).not_rankable_reason is NotRankableReason.NON_COMPARABLE_EVIDENCE


def test_etf_proxy_measurements_are_qualified_data_not_hidden(ai, ai_screens):
    ranked = rank_candidate_signals(ai, ai_screens, ALL_ASSESSABLE)
    for s in ranked.ranked:
        expected = DataQualityGrade.QUALIFIED if s.domain is MandateDomain.ETF else DataQualityGrade.CLEAN
        assert s.merit.data_quality is expected, s.name
        if expected is DataQualityGrade.QUALIFIED:
            assert "proxy" in s.summaries.data_quality


# ---------------------------------------------------------------------------
# research merit vs user fit
# ---------------------------------------------------------------------------

_MANDATES = (
    ALL_ASSESSABLE,
    ALL_ASSESSABLE.model_copy(update={"shorting_allowed": True}),
    ALL_ASSESSABLE.model_copy(update={"risk_profile": InvestorProfile(
        holding_period=HoldingPeriod.ONE_TO_THREE_DAYS, trading_frequency=TradingFrequency.HIGH)}),
    ResearchMandate(allowed_domains=(MandateDomain.FUTURES,)),
    ALL_ASSESSABLE.model_copy(update={"instrument_denylist": (
        InstrumentIdentity(asset_domain=AssetDomain.FUTURES, symbol="NQ"),)}),
    # Fit differs ACROSS signals here (20D fits Weeks, 60D does not) -- the case that would expose fit
    # being compared before merit.
    ALL_ASSESSABLE.model_copy(update={"risk_profile": InvestorProfile(holding_period=HoldingPeriod.WEEKS)}),
)


def test_research_merit_never_reads_the_mandate(ai, ai_screens):
    merits = [{s.candidate_signal_id: s.merit.model_dump_json() for s in rank_candidate_signals(
        ai, ai_screens, m).signals if s.merit is not None} for m in _MANDATES]
    assert all(m == merits[0] for m in merits) and len(merits[0]) == len(ai.candidates)


def test_soft_preferences_only_ever_break_exact_merit_ties(ai, ai_screens):
    orders = [rank_candidate_signals(ai, ai_screens, m) for m in (*_MANDATES[:3], _MANDATES[5])]
    weeks = orders[-1]
    assert {s.user_fit.check(FitDimension.HOLDING_PERIOD).status for s in weeks.ranked} == {
        FitStatus.COMPATIBLE, FitStatus.MISMATCH}
    keys = {s.candidate_signal_id: s.merit.key() for s in orders[0].ranked}
    for other in orders[1:]:
        a = [s.candidate_signal_id for s in orders[0].ranked]
        b = [s.candidate_signal_id for s in other.ranked]
        for i, x in enumerate(a):
            for y in a[i + 1:]:
                if b.index(x) > b.index(y):
                    assert keys[x] == keys[y]


def test_the_mandate_excludes_without_rewriting_merit(ai, ai_screens):
    futures_only = rank_candidate_signals(ai, ai_screens, ResearchMandate(allowed_domains=(MandateDomain.FUTURES,)))
    assert {s.instrument for s in futures_only.ranked} == {"NQ"}
    assert [s.rank for s in futures_only.ranked] == [1, 2]
    for s in futures_only.excluded:
        assert s.domain is MandateDomain.ETF and s.rank is None and s.merit is not None
        assert s.user_fit.exclusion_reasons == (ExclusionReason.DOMAIN_NOT_ALLOWED,)
        assert s.user_fit.check(FitDimension.DOMAIN_ACCESS).status is FitStatus.EXCLUDED
        assert "research merit is unchanged" in s.summaries.user_fit
    denied = rank_candidate_signals(ai, ai_screens, _MANDATES[4])
    assert {s.instrument for s in denied.excluded} == {"NQ"}
    assert all(s.user_fit.exclusion_reasons == (ExclusionReason.INSTRUMENT_DENYLISTED,) for s in denied.excluded)
    only_xlk = ALL_ASSESSABLE.model_copy(update={"instrument_allowlist": (
        InstrumentIdentity(asset_domain=AssetDomain.ETF, symbol="XLK"),)})
    listed = rank_candidate_signals(ai, ai_screens, only_xlk)
    assert {s.instrument for s in listed.ranked} == {"NQ", "XLK"}  # futures unrestricted; ETFs limited to XLK
    assert {r for s in listed.excluded for r in s.user_fit.exclusion_reasons} == {
        ExclusionReason.INSTRUMENT_NOT_IN_ALLOWLIST}


def test_user_fit_dimensions(ai, ai_screens):
    weeks_high = ALL_ASSESSABLE.model_copy(update={"risk_profile": InvestorProfile(
        holding_period=HoldingPeriod.WEEKS, trading_frequency=TradingFrequency.HIGH)})
    ranked = rank_candidate_signals(ai, ai_screens, weeks_high)
    for s in ranked.ranked:
        hold = s.user_fit.check(FitDimension.HOLDING_PERIOD)
        assert hold.status is (FitStatus.COMPATIBLE if s.prediction_horizon_days == 20 else FitStatus.MISMATCH)
        freq = s.user_fit.check(FitDimension.TRADING_FREQUENCY)
        flips = float(freq.evidence.split("~")[1].split(" ")[0])
        assert freq.status is (FitStatus.COMPATIBLE if frequency_band(flips) == "High" else FitStatus.MISMATCH)
        assert s.user_fit.personalization_state is PersonalizationState.MEASURED
        assert s.user_fit.check(FitDimension.LIQUIDITY).status is FitStatus.NOT_MEASURED
        assert s.user_fit.check(FitDimension.DRAWDOWN).status is FitStatus.NOT_MEASURED
    intraday = ALL_ASSESSABLE.model_copy(update={"risk_profile": InvestorProfile(holding_period=HoldingPeriod.INTRADAY)})
    assert all(s.user_fit.check(FitDimension.HOLDING_PERIOD).status is FitStatus.MISMATCH
               for s in rank_candidate_signals(ai, ai_screens, intraday).ranked)
    default = rank_candidate_signals(ai, ai_screens, ALL_ASSESSABLE)
    assert all(s.user_fit.personalization_state is PersonalizationState.NOT_PERSONALIZED for s in default.ranked)
    unscreened = rank_candidate_signals(ai, {}, weeks_high)
    assert all(s.user_fit.personalization_state is PersonalizationState.PARTIAL for s in unscreened.signals)
    assert all(s.user_fit.check(FitDimension.TRADING_FREQUENCY).status is FitStatus.NOT_MEASURED
               for s in unscreened.signals)


def test_shorting_is_a_constraint_measured_from_the_factor_not_an_exclusion(ai, ai_screens):
    no_short = rank_candidate_signals(ai, ai_screens, ALL_ASSESSABLE)
    assert not ALL_ASSESSABLE.shorting_allowed
    for s in no_short.ranked:
        screen = ai_screens[s.candidate_signal_id]
        values = [v for v in screen.series.values if v is not None]
        share = sum(v < 0 for v in values) / len(values)  # POSITIVE relationship: short when the factor is negative
        check = s.user_fit.check(FitDimension.SHORTING)
        assert check.status is (FitStatus.CONSTRAINED if share > 0 else FitStatus.COMPATIBLE)
        assert f"{share:.0%}" in check.evidence and s.user_fit.admitted
    allowed = rank_candidate_signals(ai, ai_screens, _MANDATES[1])
    assert all(s.user_fit.check(FitDimension.SHORTING).status is FitStatus.COMPATIBLE for s in allowed.ranked)
    assert _ids(allowed) == _ids(no_short)  # a constraint note, never a reorder (no merit ties here)


# ---------------------------------------------------------------------------
# redundancy
# ---------------------------------------------------------------------------


def test_correlated_expressions_across_domains_are_one_exposure_not_several_opportunities(ai, ai_screens):
    ranked = rank_candidate_signals(ai, ai_screens, ALL_ASSESSABLE)
    assert len(ranked.ranked) == 8 and ranked.independent_exposures == 2
    tech, utilities = ranked.exposure_groups
    assert set(tech.instruments) == {"NQ", "QQQ", "XLK"} and set(tech.domains) == {MandateDomain.FUTURES,
                                                                                    MandateDomain.ETF}
    assert tech.relations == (RedundancyRelation.SAME_STRUCTURAL_FAMILY, RedundancyRelation.CORRELATED_FACTOR)
    assert tech.max_abs_correlation is not None and tech.max_abs_correlation > 0.9
    assert utilities.instruments == ("XLU",) and utilities.relations == (RedundancyRelation.SAME_STRUCTURAL_FAMILY,)
    # Leads (independent exposures) first, then alternates; each alternate names its lead.
    assert [s.role for s in ranked.ranked] == [SignalRole.LEAD] * 2 + [SignalRole.ALTERNATE] * 6
    for s in ranked.ranked:
        group = ranked.group(s.exposure_group_id)
        assert s.lead_id == group.lead_id
        if s.role is SignalRole.ALTERNATE:
            lead = ranked.signal(s.lead_id)
            assert lead.rank < s.rank and f"#{lead.rank}" in s.summaries.redundancy
            assert s.merit.key() <= lead.merit.key()  # the lead is its group's best-merit member
            assert UncertaintyCode.REDUNDANT_EXPOSURE in {u.code for u in s.uncertainty}
    links = {frozenset((p.a, p.b)): p for p in ranked.correlations}
    nq20, qqq20 = _one(ai, "NQ", D20).candidate_signal_id, _one(ai, "QQQ", D20).candidate_signal_id
    xlu20 = _one(ai, "XLU", D20).candidate_signal_id
    assert links[frozenset((nq20, qqq20))].links and links[frozenset((nq20, qqq20))].correlation > 0.9
    assert not links[frozenset((nq20, xlu20))].links
    assert "2 independent exposure(s) from 1 event thesis" in ranked.headline()


def test_an_alternate_linked_only_through_lower_ranked_members_names_that_link(ai, ai_screens):
    """Found on real ETF bars: QQQ 20D joined its exposure only through lower-merit members."""
    ids = {(c.spec.instrument, c.spec.prediction_horizon): c.candidate_signal_id for c in ai.candidates}
    base = {k: _pinned(v, status=ScreenStatus.NO_SCREEN_SUPPORT, t=-1.0, share=0.4, timing=-5.0)
            for k, v in ai_screens.items()}
    base[ids["XLK", D60]] = _pinned(ai_screens[ids["XLK", D60]], status=ScreenStatus.SCREEN_CONTINUE, t=3.0,
                                    share=0.8, timing=5.0)
    base[ids["NQ", D20]] = _pinned(ai_screens[ids["NQ", D20]], status=ScreenStatus.NO_SCREEN_SUPPORT, t=1.9,
                                   share=0.8, timing=5.0)
    ranked = rank_candidate_signals(ai, base, ALL_ASSESSABLE)
    nq20, lead = ranked.signal(ids["NQ", D20]), ranked.signal(ids["XLK", D60])
    assert lead.role is SignalRole.LEAD and nq20.role is SignalRole.ALTERNATE and nq20.lead_id == lead.candidate_signal_id
    link = next(p for p in ranked.correlations if {p.a, p.b} == {nq20.candidate_signal_id, lead.candidate_signal_id})
    assert not link.links  # no direct link to its lead ...
    assert "itself in this exposure" in nq20.summaries.redundancy  # ... so the link it does have is named
    assert nq20.rank == min(s.rank for s in ranked.ranked if s.role is SignalRole.ALTERNATE)


def test_one_instruments_signals_share_an_exposure_even_when_their_factors_differ(ai, ai_screens):
    """Same instrument, SAME structural family (`price-momentum-continuation`), different
    lookback/horizon -- still one exposure, and via the family relation, not correlation
    (the two series barely correlate)."""
    ranked = rank_candidate_signals(ai, ai_screens, ALL_ASSESSABLE)
    xlu20, xlu60 = (ranked.signal(_one(ai, "XLU", h).candidate_signal_id) for h in (D20, D60))
    pair = next(p for p in ranked.correlations if {p.a, p.b} == {xlu20.candidate_signal_id, xlu60.candidate_signal_id})
    assert pair.links and abs(pair.correlation) < SignalRankingPolicy().redundancy_correlation
    assert xlu20.exposure_group_id == xlu60.exposure_group_id
    group = ranked.group(xlu20.exposure_group_id)
    assert RedundancyRelation.SAME_STRUCTURAL_FAMILY in group.relations


def test_same_instrument_different_factor_family_is_not_automatically_one_exposure(ai, ai_screens):
    """The real defect this guards: co-location on one instrument must never merge two
    DIFFERENT factor families (signal rules) into one exposure by itself -- only actual
    correlation may link them. Built with a synthetic second family on NQ (today's rule set
    has only one family) and deliberately uncorrelated data (NQ's price series vs an
    unrelated one) so no correlation link can form either."""
    base = _one(ai, "NQ", D20)
    other_spec = base.spec.model_copy(update={"signal_rule_id": "fixture-mean-reversion"})
    other = CandidateSignal(
        candidate_signal_id=candidate_signal_id(other_spec), factor_identity=factor_identity(other_spec),
        name="NQ fixture (different factor family)", spec=other_spec, origins=base.origins,
    )
    assert other.spec.instrument == base.spec.instrument == "NQ"
    assert other.spec.signal_rule_id != base.spec.signal_rule_id
    worlds = _worlds()

    def loader(candidate: CandidateSignal) -> DailyBars:
        frame = worlds["XLU"] if candidate.candidate_signal_id == other.candidate_signal_id else worlds["NQ"]
        return DailyBars(frame=frame.copy(), price_domain=PriceDomain.RAW, adjustment_mode=None,
                         data_role=DataProvenanceRole.REAL, provenance=("test fixture",))

    other_screen = screen_candidate(other, loader=loader, now=AT)
    wide = ai.model_copy(update={"candidates": ai.candidates + (other,)})
    screens = {**ai_screens, other.candidate_signal_id: other_screen}
    ranked = rank_candidate_signals(wide, screens, ALL_ASSESSABLE)
    nq20, other_ranked = ranked.signal(base.candidate_signal_id), ranked.signal(other.candidate_signal_id)
    assert nq20.exposure_group_id != other_ranked.exposure_group_id  # same instrument alone never merges them
    pair = next(p for p in ranked.correlations
               if {p.a, p.b} == {base.candidate_signal_id, other.candidate_signal_id})
    assert not pair.links
    assert "DIFFERENT factor family" not in other_ranked.summaries.redundancy  # it's independent, not just unranked
    assert other_ranked.role is SignalRole.LEAD


def test_offsetting_expressions_are_linked_too(ai):
    frames = _worlds()
    tech = np.diff(np.log(frames["NQ"]["close"].to_numpy()), prepend=np.log(100))
    frames["XLU"] = _frame(-tech)  # the mirror image: positions always opposed
    screens = _screens(ai, frames)
    ranked = rank_candidate_signals(ai, screens, ALL_ASSESSABLE)
    assert ranked.independent_exposures == 1
    nq20, xlu20 = _one(ai, "NQ", D20).candidate_signal_id, _one(ai, "XLU", D20).candidate_signal_id
    pair = next(p for p in ranked.correlations if {p.a, p.b} == {nq20, xlu20})
    assert pair.links and pair.correlation < -0.9


def test_a_correlation_on_too_few_common_days_never_links(ai, ai_screens):
    strict = SignalRankingPolicy(min_common_days=10_000)
    ranked = rank_candidate_signals(ai, ai_screens, ALL_ASSESSABLE, policy=strict)
    assert ranked.independent_exposures == 4  # NQ, QQQ, XLK, XLU -- same-instrument links only
    assert all(g.relations == (RedundancyRelation.SAME_STRUCTURAL_FAMILY,) for g in ranked.exposure_groups)
    assert ranked.policy_fingerprint != SignalRankingPolicy().fingerprint()


def test_one_structural_candidate_reached_by_two_events_is_ranked_once(ai, ai_screens):
    opec = _candidates(OPEC_CUT)
    shared = {c.candidate_signal_id for c in ai.candidates} & {c.candidate_signal_id for c in opec.candidates}
    assert shared  # e.g. NQ momentum reached by both the AI and the OPEC events
    ranked = rank_candidate_signals((ai, opec), ai_screens, ALL_ASSESSABLE)
    assert ranked.family_size == len({c.candidate_signal_id for c in (*ai.candidates, *opec.candidates)})
    for cid in shared:
        s = ranked.signal(cid)
        assert set(s.mechanism.event_ids) == {ai.event_id, opec.event_id}
    assert sorted(ranked.event_ids) == sorted([ai.event_id, opec.event_id])


def test_event_theses_say_when_exposures_share_one_news_hypothesis(ai, ai_screens):
    ranked = rank_candidate_signals(ai, ai_screens, ALL_ASSESSABLE)
    (thesis,) = ranked.event_theses  # one event: two exposures, one thesis
    assert thesis.event_id == ai.event_id and thesis.event_headline == AI_CAPEX
    assert set(thesis.exposure_group_ids) == {g.group_id for g in ranked.exposure_groups}
    assert thesis.consequence_labels
    leads = ranked.leads
    assert leads[0].shares_thesis_with and f"#{leads[1].rank}" in leads[0].summaries.redundancy
    assert "from 1 event thesis" in ranked.headline()


def test_event_theses_are_never_chained_through_shared_consequences(ai, ai_screens):
    opec = _candidates(OPEC_CUT)
    ranked = rank_candidate_signals((ai, opec), _screens(ai) | _screens(opec, _worlds() | {
        s: _worlds(seed=20 + i)["XLU"] for i, s in enumerate(sorted({c.spec.instrument for c in opec.candidates}
                                                                    - {"NQ", "QQQ", "XLK", "XLU"}))}), ALL_ASSESSABLE)
    assert {t.event_id for t in ranked.event_theses} == {ai.event_id, opec.event_id}
    nq = ranked.signal(_one(ai, "NQ", D20).candidate_signal_id)
    both = [g for g in ranked.exposure_groups if len(g.event_ids) == 2]
    assert nq.exposure_group_id in {g.group_id for g in both}  # NQ momentum: reached by both events
    for t in ranked.event_theses:
        assert all(t.event_id in {o.event_id for o in origins}
                   for origins in ([o for c in (ai, opec) for x in c.candidates if x.candidate_signal_id == m
                                    for o in x.origins] for m in t.member_ids))
    assert "reached by more than one event" in ranked.headline()


def test_redundancy_uses_rank_correlation_so_one_outlier_cannot_split_an_exposure(ai):
    frames = _worlds()
    shock = np.zeros(1300)
    shock[600], shock[640] = -2.5, 3.0  # a near-zero price and its rebound (CL, April 2020)
    tech = np.diff(np.log(frames["NQ"]["close"].to_numpy()), prepend=np.log(100))
    frames["QQQ"] = _frame(tech + shock)
    screens = _screens(ai, frames)
    ranked = rank_candidate_signals(ai, screens, ALL_ASSESSABLE)
    nq20, qqq20 = _one(ai, "NQ", D20).candidate_signal_id, _one(ai, "QQQ", D20).candidate_signal_id
    pair = next(p for p in ranked.correlations if {p.a, p.b} == {nq20, qqq20})
    pearson = screens[nq20].series.as_series().corr(screens[qqq20].series.as_series())
    assert pearson < 0.8 < pair.correlation and pair.links
    assert ranked.policy.correlation_method == "spearman"


# ---------------------------------------------------------------------------
# ordering, explanation, determinism
# ---------------------------------------------------------------------------


def test_merit_dimensions_are_compared_in_their_declared_order(ai, ai_screens):
    nq60, xlu60 = _one(ai, "NQ", D60).candidate_signal_id, _one(ai, "XLU", D60).candidate_signal_id
    pick = ai.model_copy(update={"candidates": tuple(c for c in ai.candidates if c.candidate_signal_id in {nq60, xlu60})})

    def order(nq: CandidateScreen, xlu: CandidateScreen) -> tuple[list[str], str]:
        r = rank_candidate_signals(pick, {nq60: nq, xlu60: xlu}, ALL_ASSESSABLE)
        return [s.instrument for s in r.ranked], r.ranked[0].summaries.reason_for_rank

    base_nq, base_xlu = ai_screens[nq60], ai_screens[xlu60]
    # 1. screen outcome beats a larger t
    names, why = order(_pinned(base_nq, status=ScreenStatus.SCREEN_CONTINUE, t=2.1, share=0.8),
                       _pinned(base_xlu, status=ScreenStatus.NO_SCREEN_SUPPORT, t=2.9, share=0.4))
    assert names == ["NQ", "XLU"] and "on screen outcome" in why
    # 2. same outcome: direction of evidence
    names, why = order(_pinned(base_nq, status=ScreenStatus.NO_SCREEN_SUPPORT, t=-0.5, share=0.8),
                       _pinned(base_xlu, status=ScreenStatus.NO_SCREEN_SUPPORT, t=0.5, share=0.2))
    assert names == ["XLU", "NQ"] and "on direction of evidence" in why
    # 3. same direction: stability
    names, why = order(_pinned(base_nq, status=ScreenStatus.NO_SCREEN_SUPPORT, t=1.0, share=0.2),
                       _pinned(base_xlu, status=ScreenStatus.NO_SCREEN_SUPPORT, t=1.5, share=0.8))
    assert names == ["XLU", "NQ"] and "on subperiod stability" in why
    # 4. same stability: cost headroom (a positive timing edge)
    names, why = order(_pinned(base_nq, status=ScreenStatus.NO_SCREEN_SUPPORT, t=1.0, share=0.8, timing=5.0),
                       _pinned(base_xlu, status=ScreenStatus.NO_SCREEN_SUPPORT, t=1.5, share=0.8, timing=-5.0))
    assert names == ["NQ", "XLU"] and "on cost headroom" in why
    # 5. equal grades: data quality (NQ direct futures bars vs XLU ETF proxy)
    names, why = order(_pinned(base_nq, status=ScreenStatus.NO_SCREEN_SUPPORT, t=1.0, share=0.8, timing=5.0),
                       _pinned(base_xlu, status=ScreenStatus.NO_SCREEN_SUPPORT, t=1.5, share=0.8, timing=5.0))
    assert names == ["NQ", "XLU"] and "on data quality" in why


def test_cost_headroom_is_graded_from_the_timing_edge_not_the_drift(ai, ai_screens):
    screen = ai_screens[_one(ai, "NQ", D20).candidate_signal_id]
    for timing, grade in ((3.0, CostHeadroom.TIMING_EDGE), (0.0, CostHeadroom.NO_TIMING_EDGE),
                          (-3.0, CostHeadroom.NO_TIMING_EDGE)):
        ev = normalize_screen_evidence(_pinned(screen, timing=timing))
        assert ev is not None and ev.cost_headroom is grade


def test_ranking_is_deterministic_and_independent_of_input_order(ai, ai_screens):
    opec = _candidates(OPEC_CUT)
    a = rank_candidate_signals((ai, opec), ai_screens, ALL_ASSESSABLE)
    b = rank_candidate_signals((opec, ai), dict(reversed(list(ai_screens.items()))), ALL_ASSESSABLE)
    shuffled = ai.model_copy(update={"candidates": tuple(reversed(ai.candidates))})
    c = rank_candidate_signals((opec, shuffled), ai_screens, ALL_ASSESSABLE)
    assert a.fingerprint() == b.fingerprint()
    assert _ids(a) == _ids(c)
    assert a.fingerprint() == rank_candidate_signals((ai, opec), ai_screens, ALL_ASSESSABLE).fingerprint()


def _relabelled(cset: CandidateSignalSet, favoured: str) -> CandidateSignalSet:
    """Every origin of ``favoured`` gets the strongest-sounding provenance, every other one the weakest."""
    def origin(c: CandidateSignal, o):
        best = c.spec.instrument == favoured
        return o.model_copy(update={
            "expression_fidelity": ExpressionFidelity.DIRECT_UNDERLYING if best else ExpressionFidelity.MACRO_PROXY,
            "path_status": PathStatus.RESEARCHABLE if best else PathStatus.PROPOSED,
        })
    return cset.model_copy(update={"candidates": tuple(
        c.model_copy(update={"origins": tuple(origin(c, o) for o in c.origins)}) for c in cset.candidates)})


def test_mechanism_confidence_and_fidelity_never_change_the_ranking(ai, ai_screens):
    original = rank_candidate_signals(ai, ai_screens, ALL_ASSESSABLE)
    # Opposite provenance in two variants: any key that read fidelity or path status would disagree with one of them.
    for favoured in ("XLU", "NQ", "QQQ"):
        relabelled = rank_candidate_signals(_relabelled(ai, favoured), ai_screens, ALL_ASSESSABLE)
        assert _ids(relabelled) == _ids(original), favoured
        assert [s.merit for s in relabelled.signals] == [s.merit for s in original.signals]
        assert [g.member_ids for g in relabelled.exposure_groups] == [g.member_ids for g in original.exposure_groups]
        seen = relabelled.signal(_one(ai, favoured, D20).candidate_signal_id).mechanism
        assert seen.fidelities == ("DIRECT_UNDERLYING",) and seen.unresolved_or_proposed_paths == 0  # it saw them


def test_the_headline_never_calls_unsupported_signals_promising(ai, ai_screens):
    none = rank_candidate_signals(ai, {k: _pinned(v, status=ScreenStatus.NO_SCREEN_SUPPORT)
                                       for k, v in ai_screens.items()}, ALL_ASSESSABLE)
    assert "None has screening support" in none.headline() and "not which are promising" in none.headline()
    nq60 = _one(ai, "NQ", D60).candidate_signal_id
    some = rank_candidate_signals(ai, {**ai_screens, nq60: _pinned(ai_screens[nq60], status=ScreenStatus.SCREEN_CONTINUE,
                                                                    t=2.5, share=0.8)}, ALL_ASSESSABLE)
    assert "1 with screening support (not validated)" in some.headline()
    assert some.ranked[0].candidate_signal_id == nq60
    assert rank_candidate_signals((), {}, ALL_ASSESSABLE).headline() == "No candidate signals to rank."


# ---------------------------------------------------------------------------
# boundaries: no sizing, no verdict, no registry
# ---------------------------------------------------------------------------

_FORBIDDEN_FIELDS = {
    "weight", "weights", "portfolio_weight", "allocation", "position_size", "size", "notional", "target_position",
    "expected_return", "return_forecast", "probability", "probability_of_profit", "score", "total_score",
    "verdict", "scientific_verdict", "registry_verdict", "buy", "sell", "action", "trade",
}


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def test_no_portfolio_sizing_score_or_verdict_exists_anywhere(ai, ai_screens):
    ranked = rank_candidate_signals(ai, ai_screens, ALL_ASSESSABLE)
    dumped = json.loads(ranked.model_dump_json())
    assert not _keys(dumped) & _FORBIDDEN_FIELDS
    text = json.dumps(dumped).upper()
    for word in ('"BUY"', '"SELL"', "STRONG BUY", "ALLOCATE"):
        assert word not in text
    assert "not portfolio weights" in ranked.not_portfolio_note


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_the_ranker_is_pure_and_never_reaches_the_scientific_write_plane():
    forbidden = ("alpha_agent.registry.sqlite_registry", "alpha_agent.validation", "alpha_agent.strategy",
                 "alpha_agent.discovery", "alpha_agent.holdout", "alpha_agent.agents", "streamlit", "requests")
    for imported in _imports(MODULE):
        assert not any(imported == f or imported.startswith(f + ".") for f in forbidden), imported
    source = MODULE.read_text(encoding="utf-8")
    for token in ("ExperimentRegistry(", "INSERT", "UPDATE ", "open(", "write_text", "read_text", "urllib"):
        assert token not in source, token


@pytest.mark.skipif(not REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout")
def test_ranking_never_changes_the_registry(ai, ai_screens):
    before = hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
    rank_candidate_signals((ai, _candidates(OPEC_CUT)), ai_screens, ALL_ASSESSABLE)
    assert hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest() == before


# ---------------------------------------------------------------------------
# real bytes
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ETF_RAW.exists(), reason="acquired ETF raw bars not present in this checkout")
def test_real_qqq_and_xlk_momentum_are_one_exposure_on_real_bars(ai):
    pick = ai.model_copy(update={"candidates": tuple(
        c for c in ai.candidates if c.spec.instrument in {"QQQ", "XLK", "XLU"} and c.spec.prediction_horizon is D20)})
    screens = {c.candidate_signal_id: screen_candidate(c) for c in pick.candidates}
    assert all(s.series.data_role is DataProvenanceRole.REAL for s in screens.values())
    ranked = rank_candidate_signals(pick, screens, ALL_ASSESSABLE)
    assert len(ranked.ranked) == 3
    groups = {tuple(sorted(g.instruments)) for g in ranked.exposure_groups}
    assert ("QQQ", "XLK") in groups and ("XLU",) in groups
    qqq_xlk = next(g for g in ranked.exposure_groups if set(g.instruments) == {"QQQ", "XLK"})
    assert qqq_xlk.relations == (RedundancyRelation.CORRELATED_FACTOR,) and qqq_xlk.max_abs_correlation > 0.9
    assert ranked.fingerprint() == rank_candidate_signals(pick, screens, ALL_ASSESSABLE).fingerprint()
