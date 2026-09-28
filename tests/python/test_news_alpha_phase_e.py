"""News Alpha Phase E -- Asset Expression -> CANDIDATE SIGNAL SPEC -> FACTOR
EXPRESSION -> factor series -> factor diagnostics (screening).

Candidate construction is pure and offline: a fixed `DomainCapabilities`
snapshot (mirroring the committed CME catalog), fixed timestamps, no
network, no LLM. The series/screen tests use an in-memory bar loader (a
test fixture -- its screens are marked SYNTHETIC and can never be stored),
plus one real-bytes test on an already-acquired ETF that skips when the raw
store is absent.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from alpha_agent.crypto.provenance import DataProvenanceRole
from alpha_agent.features.daily import compute_contiguous_daily_features
from alpha_agent.features.expression import (
    OPERATORS,
    CrossSectionalOperatorError,
    FactorExpression,
    Normalization,
    evaluate_expression,
)
from alpha_agent.features.registry import REGISTRY, FeatureRegistry
from alpha_agent.features.spec import FeatureSpec
from alpha_agent.news_alpha import (
    DEFAULT_EXPRESSION_LIBRARY,
    AssetExpressionPlan,
    CandidateNote,
    CandidateSignal,
    CandidateSignalSet,
    DomainCapabilities,
    EventConditioning,
    ExpectedRelationship,
    ExpressionFidelity,
    ExpressionForm,
    ExpressionLibrary,
    ExpressionRule,
    FuturesBarCoverage,
    MandateDomain,
    PathType,
    Polarity,
    RefusalReason,
    ResearchMandate,
    ResolutionStatus,
    SignalHorizon,
    SignalPathDiscovery,
    TransmissionLag,
    UserDescribedEvent,
    build_asset_expressions,
    build_candidate_signals,
    build_economic_mechanism_graph,
    discover_signal_paths,
    resolve_allowed_universe,
    scan_initial_impact,
)
from alpha_agent.news_alpha.candidate_signals import (
    DEFAULT_GENERATION_GRID,
    SIGNAL_RULES,
    ConditioningStatus,
    GenerationPoint,
    SignalDataBinding,
    candidate_signal_id,
    factor_identity,
)
from alpha_agent.news_alpha.expression_library import MEASUREMENT_TEMPLATES
from alpha_agent.schemas.market_data import PriceDomain
from alpha_agent.screening.candidate_signal_screen import (
    CandidateScreen,
    DailyBars,
    FactorScreenStore,
    discovery_window,
    screen_candidate,
    screen_candidates,
)
from alpha_agent.screening.factor_diagnostics import FactorScreenPolicy, ScreenStatus
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE_E_SOURCES = [
    REPO_ROOT / "python" / "alpha_agent" / "news_alpha" / "candidate_signals.py",
    REPO_ROOT / "python" / "alpha_agent" / "features" / "expression.py",
    REPO_ROOT / "python" / "alpha_agent" / "screening" / "factor_diagnostics.py",
    REPO_ROOT / "python" / "alpha_agent" / "screening" / "candidate_signal_screen.py",
    REPO_ROOT / "python" / "alpha_agent" / "ui" / "candidate_signal_view.py",
]
REGISTRY_PATH = REPO_ROOT / "data" / "registry" / "experiments.sqlite"

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
EVERY_DOMAIN = ResearchMandate(allowed_domains=tuple(MandateDomain))

AI_CAPEX = "Hyperscalers announce a major AI infrastructure spending increase, lifting data center capex guidance"
OPEC_CUT = "OPEC+ agrees a production cut of 1 million barrels per day"
RATE_SURPRISE = "Rate-policy surprise: the Fed unexpectedly raised rates by 50 basis points"
_O, _C, _F = MandateDomain.OPTIONS, MandateDomain.CRYPTO, MandateDomain.FUTURES

_OPTIONS_AND_CRYPTO = ExpressionLibrary(
    rules=DEFAULT_EXPRESSION_LIBRARY.rules + (
        ExpressionRule("accelerator_demand", _O, "Options on AI accelerator designers", ExpressionForm.SINGLE_NAME,
                       Polarity.POSITIVE, "fixture", symbols=("NVDA",), fidelity=ExpressionFidelity.DIRECT_COMPANY),
        ExpressionRule("electricity_demand", _C, "Bitcoin (miners compete for power)", ExpressionForm.UNDERLYING,
                       Polarity.NEGATIVE, "fixture", fidelity=ExpressionFidelity.ECOSYSTEM_PROXY),
        ExpressionRule("electricity_demand", _F, "CME bitcoin futures", ExpressionForm.UNDERLYING,
                       Polarity.NEGATIVE, "fixture", symbols=("BTC",), fidelity=ExpressionFidelity.ECOSYSTEM_PROXY),
    ),
    templates=MEASUREMENT_TEMPLATES,
)


def _pipeline(
    text: str = AI_CAPEX, mandate: ResearchMandate = ALL_ASSESSABLE, **kwargs,
) -> tuple[SignalPathDiscovery, AssetExpressionPlan, CandidateSignalSet]:
    universe = resolve_allowed_universe(mandate, capabilities=CAPS)
    scan = scan_initial_impact(UserDescribedEvent.create(text, described_at=AT), mandate, universe=universe)
    paths = discover_signal_paths(build_economic_mechanism_graph(scan))
    plan = build_asset_expressions(paths, universe, mandate, **kwargs)
    return paths, plan, build_candidate_signals(plan, paths)


def _one(cset: CandidateSignalSet, instrument: str, horizon: SignalHorizon) -> CandidateSignal:
    (c,) = [c for c in cset.candidates if c.spec.instrument == instrument and c.spec.prediction_horizon is horizon]
    return c


# ---------------------------------------------------------------------------
# the AI vertical slice and the Phase D gate
# ---------------------------------------------------------------------------


def test_ai_slice_yields_momentum_candidates_on_the_measurable_instruments():
    _, plan, cset = _pipeline()
    # The same predeclared generation grid for every instrument -- no lag decides a horizon.
    assert [(c.spec.instrument, c.expression, c.spec.prediction_horizon) for c in cset.candidates] == [
        (sym, f"ts_return(close, {n})", h)
        for sym in ("NQ", "QQQ", "XLK", "XLU") for n, h in ((20, SignalHorizon.D20), (60, SignalHorizon.D60))
    ]
    assert set(cset.instruments()) <= {m.target for m in plan.usable_measurements()}
    nq = _one(cset, "NQ", SignalHorizon.D20)
    assert nq.name == "NQ 20-day price momentum → 20D"
    assert nq.spec.expected_relationship is ExpectedRelationship.POSITIVE and nq.status == "CANDIDATE"
    assert nq.spec.data.dataset == "GLBX.MDP3" and nq.spec.data.resolution_status is ResolutionStatus.AVAILABLE
    # No equity candidate: equity bars are not acquired, so nothing is fabricated.
    assert not [c for c in cset.candidates if c.spec.domain is MandateDomain.EQUITY]


def test_every_candidate_comes_through_the_phase_d_gate():
    for text in (AI_CAPEX, OPEC_CUT, RATE_SURPRISE):
        _, plan, cset = _pipeline(text, EVERY_DOMAIN, library=_OPTIONS_AND_CRYPTO)
        for c in cset.candidates:
            for o in c.origins:
                expression = next(e for e in plan.expressions if e.expression_id == o.expression_id)
                assert o.measurement_spec_id in {m.spec_id for m in plan.candidate_signal_basis(expression)}


def test_no_candidate_from_missing_not_pit_safe_not_executable_unavailable_or_synthetic_measurements():
    blocked = {ResolutionStatus.MISSING, ResolutionStatus.NOT_PIT_SAFE, ResolutionStatus.NOT_EXECUTABLE,
               ResolutionStatus.DOMAIN_UNAVAILABLE}
    seen = set()
    for text in (AI_CAPEX, OPEC_CUT, RATE_SURPRISE):
        _, plan, cset = _pipeline(text, EVERY_DOMAIN, library=_OPTIONS_AND_CRYPTO)
        used = {c.spec.data.measurement_spec_id for c in cset.candidates}
        for m in plan.measurements:
            if m.resolution.status in blocked or m.resolution.data_role is DataProvenanceRole.SYNTHETIC:
                seen.add(m.resolution.status)
                assert m.spec_id not in used, (m.measurement_id, m.target, m.resolution.status)
        assert not [c for c in cset.candidates if c.spec.domain in (_O, _C)]
        assert all(c.spec.data.data_role is DataProvenanceRole.REAL and c.spec.data.pit_safe is True
                   for c in cset.candidates)
    assert seen == blocked  # the sweep actually met every blocked status


def test_the_candidate_data_binding_refuses_unusable_or_holdout_data():
    _, _, cset = _pipeline()
    good = _one(cset, "NQ", SignalHorizon.D20).spec.data.model_dump()
    for bad in ({"resolution_status": ResolutionStatus.MISSING}, {"resolution_status": ResolutionStatus.NOT_PIT_SAFE},
                {"data_role": DataProvenanceRole.SYNTHETIC}, {"pit_safe": False},
                {"coverage_end_exclusive": date(2025, 6, 1)}):
        with pytest.raises(ValidationError):
            SignalDataBinding(**{**good, **bad})


def test_execution_capable_alone_never_yields_a_formula():
    _, plan, cset = _pipeline(registry=FeatureRegistry())  # nothing registered -> nothing executable
    assert any(e.execution_capable for e in plan.expressions)
    assert cset.is_empty
    assert {r.reason for r in cset.refusals} == {RefusalReason.NOT_CANDIDATE_READY}


def test_activity_is_a_confirmation_input_never_a_standalone_candidate():
    _, _, cset = _pipeline()
    confirmation = [r for r in cset.refusals if r.reason is RefusalReason.CONFIRMATION_ONLY_MEASUREMENT]
    assert {r.measurement_id for r in confirmation} == {"futures.activity", "etf.activity"}
    assert not [c for c in cset.candidates if "activity" in c.spec.measurement_id]
    assert {r.instrument for r in confirmation} == {"NQ", "QQQ", "XLK", "XLU"}


def _with_every_lag(plan: AssetExpressionPlan, lag: TransmissionLag) -> AssetExpressionPlan:
    return plan.model_copy(update={"expressions": tuple(
        e.model_copy(update={"pressures": tuple(p.model_copy(update={"horizon": lag}) for p in e.pressures)})
        for e in plan.expressions
    )})


def test_the_transmission_lag_never_decides_the_prediction_horizon():
    """Transmission horizon != prediction horizon: rewriting every path's
    economic lag (immediate, years, unknown) changes no candidate -- only
    the lag recorded on each origin."""
    paths, plan, cset = _pipeline()
    for lag in (TransmissionLag.IMMEDIATE, TransmissionLag.YEARS, TransmissionLag.UNKNOWN):
        other = build_candidate_signals(_with_every_lag(plan, lag), paths)
        assert [c.candidate_signal_id for c in other.candidates] == [c.candidate_signal_id for c in cset.candidates]
        assert [c.spec for c in other.candidates] == [c.spec for c in cset.candidates]
        assert {o.transmission_lag for c in other.candidates for o in c.origins} == {lag}
    assert not {"HORIZON_UNKNOWN", "HORIZON_CAPPED"} & {x.value for x in (*RefusalReason, *CandidateNote)}


def test_formation_lookback_and_prediction_horizon_are_independent_fields():
    paths, plan, cset = _pipeline()
    assert all(p.formation_lookback == p.prediction_horizon.days for p in DEFAULT_GENERATION_GRID)  # policy only
    crossed = build_candidate_signals(plan, paths, grid=(GenerationPoint(60, SignalHorizon.D20),))
    nq = _one(crossed, "NQ", SignalHorizon.D20)
    assert (nq.expression, nq.spec.formation_lookback, nq.spec.prediction_horizon) == (
        "ts_return(close, 60)", 60, SignalHorizon.D20)
    matched = _one(cset, "NQ", SignalHorizon.D20)
    assert nq.candidate_signal_id != matched.candidate_signal_id  # its own hypothesis ...
    assert nq.factor_identity == _one(cset, "NQ", SignalHorizon.D60).factor_identity  # ... on the 60-bar factor
    spec = matched.spec
    with pytest.raises(ValidationError, match="formation lookback"):
        type(spec)(**{**spec.model_dump(), "formation_lookback": 60})  # must match the executed expression


def test_a_plan_from_other_paths_is_refused():
    paths, _, _ = _pipeline(AI_CAPEX)
    _, other_plan, _ = _pipeline(OPEC_CUT)
    with pytest.raises(ValueError, match="different signal paths"):
        build_candidate_signals(other_plan, paths)


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------


def test_every_candidate_keeps_its_full_lineage():
    paths, plan, cset = _pipeline()
    by_id = {p.path_id: p for p in paths.paths}
    for c in cset.candidates:
        assert c.origins
        for o in c.origins:
            path = by_id[o.path_id]
            e = next(x for x in plan.expressions if x.expression_id == o.expression_id)
            assert (o.event_id, o.event_headline) == (plan.event_id, plan.event_headline)
            assert o.mechanism_graph_id == paths.mechanism_graph_id == path.mechanism_graph_id
            assert (o.path_signature, o.path_type, o.transmission_depth, o.route) == (
                path.signature, path.path_type, path.transmission_depth, path.state_labels)
            assert (o.consequence_state, o.consequence_label) == (path.consequence_state, path.consequence_label)
            assert (o.expression_concept, o.expression_fidelity, o.expression_relation) == (
                e.concept, e.fidelity, e.relation)
            assert o.path_id in e.path_ids and o.measurement_spec_id in e.measurement_ids
            assert o.transmission_lag is next(p.horizon for p in e.pressures if o.path_id in p.path_ids)
        m = plan.measurement(c.spec.data.measurement_spec_id)
        assert (m.symbol, m.measurement_id) == (c.spec.instrument, c.spec.measurement_id)
        assert c.spec.data.pit_rule == m.resolution.pit_rule and c.spec.data.dataset == m.resolution.dataset
        assert c.spec.rationale.sign and c.spec.rationale.prediction_horizon and c.spec.rationale.formation_lookback


def test_path_type_and_transmission_depth_are_preserved():
    _, _, cset = _pipeline()
    nq = _one(cset, "NQ", SignalHorizon.D20)
    assert set(nq.path_types) == {PathType.DIRECT, PathType.SUPPLY_CHAIN}
    assert nq.depths == (1, 2, 3, 5)
    xlu = _one(cset, "XLU", SignalHorizon.D60)
    assert xlu.path_types == (PathType.CROSS_SECTOR,) and xlu.depths == (2, 3)
    assert _one(cset, "XLU", SignalHorizon.D20).origins == xlu.origins  # the grid, not the lag, sets horizons


def test_fidelity_is_provenance_only_and_never_changes_candidates():
    flattened = ExpressionLibrary(
        rules=tuple(dataclasses.replace(r, fidelity=ExpressionFidelity.MACRO_PROXY)
                    for r in DEFAULT_EXPRESSION_LIBRARY.rules),
        templates=MEASUREMENT_TEMPLATES,
    )
    _, _, a = _pipeline()
    _, _, b = _pipeline(library=flattened)
    assert [c.candidate_signal_id for c in a.candidates] == [c.candidate_signal_id for c in b.candidates]
    assert [c.spec for c in a.candidates] == [c.spec for c in b.candidates]
    assert [c.notes for c in a.candidates] == [c.notes for c in b.candidates]
    strip = lambda s: [[o.model_dump(exclude={"expression_fidelity"}) for o in c.origins] for c in s.candidates]
    assert strip(a) == strip(b)
    assert {f for c in b.candidates for f in c.fidelities} == {ExpressionFidelity.MACRO_PROXY}


def test_the_conceptual_event_model_never_invents_a_magnitude():
    cond = EventConditioning()
    assert (cond.event_strength, cond.exposure, cond.confirmation, cond.transmission_weight) == (
        ConditioningStatus.UNKNOWN_MAGNITUDE, ConditioningStatus.UNKNOWN_MAGNITUDE, ConditioningStatus.IN_FACTOR,
        ConditioningStatus.NOT_DERIVED)
    for bad in ({"exposure": ConditioningStatus.IN_FACTOR}, {"event_strength": ConditioningStatus.IN_FACTOR},
                {"transmission_weight": ConditioningStatus.UNKNOWN_MAGNITUDE}):
        with pytest.raises(ValidationError):
            EventConditioning(**bad)


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_same_inputs_give_the_same_candidates_and_identities():
    _, _, a = _pipeline()
    _, _, b = _pipeline()
    assert a == b and a.fingerprint() == b.fingerprint()
    for c in a.candidates:
        assert c.factor_identity == factor_identity(c.spec) and c.candidate_signal_id == candidate_signal_id(c.spec)
    assert CandidateSignalSet.model_validate_json(a.model_dump_json()) == a


def test_parameter_changes_change_identity_where_they_change_the_hypothesis():
    _, _, cset = _pipeline()
    nq20, nq60 = _one(cset, "NQ", SignalHorizon.D20), _one(cset, "NQ", SignalHorizon.D60)
    assert nq20.factor_identity != nq60.factor_identity  # a different lookback is a different factor
    spec = nq20.spec
    other_horizon = spec.model_copy(update={"prediction_horizon": SignalHorizon.D5})
    assert factor_identity(other_horizon) == nq20.factor_identity  # same factor ...
    assert candidate_signal_id(other_horizon) != nq20.candidate_signal_id  # ... asked a different question
    flipped = spec.model_copy(update={"expected_relationship": ExpectedRelationship.NEGATIVE})
    assert candidate_signal_id(flipped) != nq20.candidate_signal_id  # a re-signed factor is a new hypothesis
    xlk20 = _one(cset, "XLK", SignalHorizon.D20)
    assert xlk20.expression == nq20.expression and xlk20.factor_identity != nq20.factor_identity  # other data


def test_several_paths_and_events_share_one_structural_candidate_with_all_their_provenance():
    _, _, ai = _pipeline(AI_CAPEX)
    _, _, opec = _pipeline(OPEC_CUT)
    ai_nq, opec_nq = _one(ai, "NQ", SignalHorizon.D20), _one(opec, "NQ", SignalHorizon.D20)
    assert ai_nq.candidate_signal_id == opec_nq.candidate_signal_id
    assert ai_nq.factor_identity == opec_nq.factor_identity and ai_nq.spec == opec_nq.spec
    assert ai_nq.event_ids != opec_nq.event_ids  # provenance kept apart
    assert len({o.path_id for o in ai_nq.origins}) >= 3 and len({o.expression_id for o in ai_nq.origins}) >= 3
    ids = [c.candidate_signal_id for c in ai.candidates]
    assert len(ids) == len(set(ids))  # never a duplicate executable factor


# ---------------------------------------------------------------------------
# the factor expression
# ---------------------------------------------------------------------------


def test_the_operator_vocabulary_is_closed_and_bound_to_registered_features():
    for op in OPERATORS.values():
        for _, kind in op.kinds:
            assert kind in REGISTRY and tuple(REGISTRY.get(kind).param_order) == op.params
    for bad in (
        {"operator": "ts_magic", "field": "close", "params": (("n", 20),)},
        {"operator": "ts_return", "field": "volume", "params": (("n", 20),)},
        {"operator": "ts_return", "field": "close", "params": (("window", 20),)},
        {"operator": "ts_return", "field": "close", "params": (("n", True),)},
        {"operator": "ts_return", "field": "close", "params": (("n", 0),)},
        {"operator": "ts_trend_strength", "field": "close", "params": (("fast", 50), ("slow", 20))},
    ):
        with pytest.raises(ValidationError):
            FactorExpression(**bad)


def test_the_expression_compiles_deterministically_to_one_registered_feature():
    e = FactorExpression.of("ts_return", "close", 20)
    assert e.render() == "ts_return(close, 20)"
    assert e.feature_spec() == FeatureSpec(kind="return", params={"n": 20})
    assert e.canonical_json() == FactorExpression.of("ts_return", "close", 20).canonical_json()
    assert FactorExpression.model_validate_json(e.model_dump_json()) == e
    assert FactorExpression.of("ts_zscore", "volume", 60).feature_spec().kind == "volume_zscore"
    assert e.lookback == 20


def test_a_cross_sectional_rank_is_refused_on_a_single_instrument():
    ranked = FactorExpression.of("ts_return", "close", 20, normalization=Normalization.CROSS_SECTIONAL_RANK)
    assert ranked.render() == "rank(ts_return(close, 20))"
    with pytest.raises(CrossSectionalOperatorError, match="no cross-section"):
        evaluate_expression(ranked, _bars(80).frame, instrument="X", price_domain=PriceDomain.RAW)
    _, _, cset = _pipeline()
    spec = _one(cset, "NQ", SignalHorizon.D20).spec
    with pytest.raises(ValidationError):
        type(spec)(**{**spec.model_dump(), "expression": ranked.model_dump()})


# ---------------------------------------------------------------------------
# factor series + screen (in-memory bars; one real ETF)
# ---------------------------------------------------------------------------


def _bars(n: int = 1400, *, start: str = "2018-01-02", seed: int = 7, drift: float = 0.0004) -> DailyBars:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n)
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.01, n))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.001, n))
    frame = pd.DataFrame({
        "trading_day": days.strftime("%Y-%m-%d"), "ts_event_ns": days.asi8 + 21 * 3600 * 10**9,
        "open": open_, "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998,
        "close": close, "volume": rng.integers(1_000, 5_000, n).astype("float64"),
    })
    return DailyBars(frame=frame, price_domain=PriceDomain.RAW, adjustment_mode=None,
                     data_role=DataProvenanceRole.SYNTHETIC, provenance=("test fixture",))


def _loader(bars: DailyBars):
    return lambda candidate: bars


def test_ui_expression_equals_the_executed_and_diagnosed_expression_with_no_hidden_transform():
    _, _, cset = _pipeline()
    c = _one(cset, "NQ", SignalHorizon.D20)
    bars = _bars()
    screen = screen_candidate(c, loader=_loader(bars), now=AT)
    assert c.expression == screen.expression == screen.series.expression == screen.diagnostics.expression
    assert screen.series.feature_name == "return_20"
    start, end = discovery_window(c)
    window = bars.frame[(bars.frame["trading_day"] >= str(start)) & (bars.frame["trading_day"] < str(end))]
    window = window.reset_index(drop=True)
    independent = compute_contiguous_daily_features(
        window, [FeatureSpec(kind="return", params={"n": 20})], root_symbol="NQ", price_domain=PriceDomain.RAW,
    ).frame.features["return_20"].to_numpy()
    by_hand = window["close"].pct_change(20, fill_method=None).to_numpy()
    got = screen.series.as_series().to_numpy()
    np.testing.assert_allclose(got, independent, equal_nan=True)
    np.testing.assert_allclose(got, by_hand, equal_nan=True, rtol=1e-9)
    with pytest.raises(ValidationError, match="own expression"):
        CandidateScreen(**{**screen.model_dump(), "expression": "rank(ts_return(close, 20))"})


def test_same_bars_and_spec_give_the_same_values_and_screen_fingerprint():
    _, _, cset = _pipeline()
    c = _one(cset, "XLK", SignalHorizon.D60)
    a = screen_candidate(c, loader=_loader(_bars()), now=AT)
    b = screen_candidate(c, loader=_loader(_bars()), now=datetime(2026, 9, 26, tzinfo=UTC))
    assert a.series.values_fingerprint == b.series.values_fingerprint
    assert a.series.data_fingerprint == b.series.data_fingerprint and a.fingerprint() == b.fingerprint()
    other = screen_candidate(c, loader=_loader(_bars(seed=8)), now=AT)
    assert other.series.data_fingerprint != a.series.data_fingerprint


def test_screening_reads_the_discovery_window_only_and_never_the_holdout():
    _, _, cset = _pipeline()
    c = _one(cset, "NQ", SignalHorizon.D20)
    bars = _bars(n=1900)  # runs into 2025
    assert bars.frame["trading_day"].max() >= "2025-01-01"
    screen = screen_candidate(c, loader=_loader(bars), now=AT)
    assert screen.series.window_end_exclusive == date(2023, 1, 1) and screen.series.days[-1] < "2023-01-01"
    assert screen.diagnostics.coverage.window_end_exclusive == date(2023, 1, 1)
    # The forward returns near the boundary are dropped, not reached across it.
    assert screen.diagnostics.declared.n_pairs <= screen.diagnostics.coverage.n_factor_defined - 20


def test_the_screen_store_keeps_real_screens_only_and_reuses_them_across_events(tmp_path):
    _, _, ai = _pipeline(AI_CAPEX)
    store = FactorScreenStore(tmp_path)
    synthetic = screen_candidate(_one(ai, "NQ", SignalHorizon.D20), loader=_loader(_bars()), now=AT)
    with pytest.raises(ValueError, match="REAL"):
        store.save(synthetic)
    screens, run = screen_candidates(ai, loader=_loader(_bars()), store=store)
    assert not list(tmp_path.iterdir())  # synthetic screens are computed but never stored
    assert run.family_size == len(ai.candidates) == len(screens)
    assert len(run.correlations) == len(screens) * (len(screens) - 1) // 2
    real = CandidateScreen(**{**synthetic.model_dump(), "series": {
        **synthetic.series.model_dump(), "data_role": DataProvenanceRole.REAL}})
    store.save(real)
    _, _, opec = _pipeline(OPEC_CUT)
    assert store.load(_one(opec, "NQ", SignalHorizon.D20)) == real  # same structural candidate, other event
    assert store.load(_one(opec, "XLE", SignalHorizon.D20)) is None
    # A screen made under another screening policy is stale, never shown as current.
    stale = real.model_copy(update={"diagnostics": real.diagnostics.model_copy(update={
        "policy_fingerprint": FactorScreenPolicy(t_threshold=1.0).fingerprint()})})
    store.save(stale)
    assert store.load(_one(ai, "NQ", SignalHorizon.D20)) is None


_ETF_RAW = REPO_ROOT / "data" / "raw" / "databento" / "ARCX.PILLAR"


@pytest.mark.skipif(not _ETF_RAW.exists(), reason="acquired ETF raw bars not present in this checkout")
def test_a_real_etf_candidate_screens_on_real_point_in_time_bars():
    _, _, cset = _pipeline()
    c = _one(cset, "XLK", SignalHorizon.D20)
    screen = screen_candidate(c)
    d = screen.diagnostics
    assert screen.series.data_role is DataProvenanceRole.REAL and screen.series.price_domain is PriceDomain.RAW
    assert screen.series.window_start == date(2018, 5, 1) and screen.series.days[-1] < "2023-01-01"
    assert [h.horizon_days for h in d.horizons] == [1, 5, 20, 60] and d.declared.horizon_days == 20
    assert [s.label for s in d.subperiods] == ["2018", "2019", "2020", "2021", "2022"]
    assert d.coverage.undefined_after_warmup == 0 and d.coverage.coverage > 0.95
    assert d.status in set(ScreenStatus)
    assert screen.fingerprint() == screen_candidate(c).fingerprint()  # deterministic on real bytes


# ---------------------------------------------------------------------------
# boundaries: screening is not validation
# ---------------------------------------------------------------------------


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


@pytest.mark.parametrize("path", PHASE_E_SOURCES, ids=lambda p: p.name)
def test_phase_e_never_touches_the_registry_validation_or_alpha_memory(path):
    forbidden = ("alpha_agent.registry.sqlite_registry", "alpha_agent.validation", "alpha_agent.alpha_memory",
                 "alpha_agent.agents.llm", "alpha_agent.knowledge", "alpha_agent.learn")
    for imported in _imports(path):
        assert not any(imported == f or imported.startswith(f + ".") for f in forbidden), (path.name, imported)
    source = path.read_text(encoding="utf-8")
    for token in ("ExperimentRegistry(", "append_attempt", "INSERT", "UPDATE ", "requests.", "urllib", "httpx",
                  "Databento", "get_cost"):
        assert token not in source, (path.name, token)


def test_no_candidate_or_screen_carries_a_return_forecast_weight_or_verdict_field():
    from alpha_agent.news_alpha import candidate_signals as cs
    from alpha_agent.screening import factor_diagnostics as fd

    forbidden = {"expected_return", "probability", "probability_of_profit", "weight", "portfolio_weight",
                 "position_size", "verdict", "scientific_verdict", "validated", "p_value", "q_value", "dsr",
                 "sharpe", "alpha", "experiment_id", "experiment_identity"}
    for model in (cs.CandidateSignal, cs.SignalSpec, cs.CandidateOrigin, cs.CandidateSignalSet, cs.SignalDataBinding,
                  CandidateScreen, fd.FactorDiagnostics, fd.HorizonDiagnostics, fd.CostSensitivity):
        assert not (set(model.model_fields) & forbidden), model.__name__
    assert not ({s.value for s in ScreenStatus} & {"PASS", "VALIDATED", "VALID", "BUY", "SELL"})
    assert {r.rule_id for r in SIGNAL_RULES} == {"price-momentum-continuation", "activity-confirmation"}


@pytest.mark.skipif(not REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout")
def test_building_and_screening_candidates_never_writes_the_registry(tmp_path):
    before = hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
    _, _, cset = _pipeline()
    screen_candidates(cset, loader=_loader(_bars()), store=FactorScreenStore(tmp_path))
    assert hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest() == before
