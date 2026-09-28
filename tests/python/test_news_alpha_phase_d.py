"""News Alpha Phase D -- Signal Path -> ASSET EXPRESSION -> MEASUREMENT SPEC ->
point-in-time DATA FIELD RESOLUTION.

Pure, offline tests: a fixed `DomainCapabilities` snapshot (mirroring the
committed CME catalog), fixed timestamps, no network, no LLM. Covers mandate
filtering, multiple expressions per path, unsupported domains, PIT safety,
proxies, missing fields, provenance, no synthetic-to-real promotion, and no
Options/Crypto execution path -- plus the reviewed library's own checks.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from alpha_agent.crypto.provenance import DataProvenanceRole
from alpha_agent.equities import data_availability
from alpha_agent.features.registry import FeatureRegistry
from alpha_agent.marketdata.product_catalog import RESEARCH_UNIVERSE
from alpha_agent.news_alpha import (
    DEFAULT_EXPRESSION_LIBRARY,
    AssetExpression,
    AssetExpressionPlan,
    CandidateSignalPrerequisiteError,
    DataFieldResolution,
    DataRequirement,
    DomainCapabilities,
    DomainSupport,
    ExpressionFidelity,
    ExpressionForm,
    ExpressionLibrary,
    ExpressionRule,
    ExpressionStatus,
    FuturesBarCoverage,
    MandateDomain,
    MeasurementRole,
    MeasurementTemplate,
    Polarity,
    PublicationLag,
    ResearchMandate,
    ResolutionIssue,
    ResolutionStatus,
    UserDescribedEvent,
    build_asset_expressions,
    build_economic_mechanism_graph,
    discover_signal_paths,
    resolve_allowed_universe,
    resolve_field,
    scan_initial_impact,
)
from alpha_agent.news_alpha.channels import PressureSign
from alpha_agent.news_alpha.expression_library import MEASUREMENT_TEMPLATES, _validate
from alpha_agent.news_alpha.measurement import build_measurement_spec, status_for
from alpha_agent.news_alpha.universe import read_futures_bar_coverage
from alpha_agent.registry.holdout_guard import HoldoutAccessError
from alpha_agent.translation.schemas import MeasurableVariable
from alpha_agent.ui import asset_expression_view
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "python" / "alpha_agent" / "news_alpha"
PHASE_D_SOURCES = [
    PACKAGE_DIR / "asset_expression.py", PACKAGE_DIR / "expression_library.py", PACKAGE_DIR / "measurement.py",
    REPO_ROOT / "python" / "alpha_agent" / "ui" / "asset_expression_view.py",
]
REGISTRY_PATH = REPO_ROOT / "data" / "registry" / "experiments.sqlite"

#: Mirrors the committed catalog (5 certified roots, 2018-01-01 .. 2025-01-01 exclusive).
FUTURES_BARS = tuple(
    FuturesBarCoverage(root=r, dataset="GLBX.MDP3", data_schema="ohlcv-1m", start=date(2018, 1, 1),
                       end_exclusive=date(2025, 1, 1), continuous_segments=2, roll_overlap_artifacts=28)
    for r in ("CL", "ES", "GC", "NQ", "ZN")
)
CAPS = DomainCapabilities(etf_market_data_acquired=True, equity_market_data_acquired=False, futures_bars=FUTURES_BARS)
AT = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
ALL_ASSESSABLE = ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES, MandateDomain.ETF))
EVERY_DOMAIN = ResearchMandate(allowed_domains=tuple(MandateDomain))

AI_CAPEX = "Hyperscalers announce a major AI infrastructure spending increase, lifting data center capex guidance"
OPEC_CUT = "OPEC+ agrees a production cut of 1 million barrels per day"
RATE_SURPRISE = "Rate-policy surprise: the Fed unexpectedly raised rates by 50 basis points"
GEOPOLITICAL = "Missile strikes and a naval blockade raise geopolitical escalation fears"

_F, _E, _Q = MandateDomain.FUTURES, MandateDomain.ETF, MandateDomain.EQUITY
_O, _C = MandateDomain.OPTIONS, MandateDomain.CRYPTO
_S, _I = ResolutionStatus, ResolutionIssue


def _plan(
    text: str = AI_CAPEX, mandate: ResearchMandate = ALL_ASSESSABLE, *, caps: DomainCapabilities = CAPS, **kwargs,
) -> AssetExpressionPlan:
    universe = resolve_allowed_universe(mandate, capabilities=caps)
    scan = scan_initial_impact(UserDescribedEvent.create(text, described_at=AT), mandate, universe=universe)
    paths = discover_signal_paths(build_economic_mechanism_graph(scan))
    return build_asset_expressions(paths, universe, mandate, **kwargs)


def _expr(plan: AssetExpressionPlan, state_id: str, domain: MandateDomain, concept: str | None = None) -> AssetExpression:
    (e,) = [
        e for e in plan.expressions
        if e.consequence_state == state_id and e.domain is domain and (concept is None or e.concept == concept)
    ]
    return e


def _spec(plan: AssetExpressionPlan, measurement_id: str, target: str):
    (m,) = [m for m in plan.measurements if m.measurement_id == measurement_id and m.target == target]
    return m


def _resolve(measurement_id: str, symbol: str | None, support=DomainSupport.RESEARCH_READY, caps=CAPS, **kw):
    return resolve_field(
        DEFAULT_EXPRESSION_LIBRARY.template(measurement_id), symbol, support=support, capabilities=caps, **kw,
    )


# ---------------------------------------------------------------------------
# the vertical slice: AI infrastructure
# ---------------------------------------------------------------------------


def test_ai_slice_expresses_semiconductor_power_and_commodity_consequences():
    plan = _plan()
    assert len(plan.expressions) == 38 and plan.unmapped_consequences == ()
    # equity: semiconductor / equipment / power-related measurements
    assert _expr(plan, "accelerator_demand", _Q).symbols == ("NVDA",)
    assert _expr(plan, "semiconductor_equipment_demand", _Q).status is ExpressionStatus.NO_INSTRUMENT
    assert _expr(plan, "power_equipment_demand", _Q).symbols == ("CAT",)
    nvda = {m.measurement_id for m in plan.measurements if m.symbol == "NVDA"}
    assert {"equity.revenue_growth", "equity.gross_margin", "equity.inventory_growth", "equity.momentum",
            "equity.relative_strength", "equity.activity", "equity.estimate_revisions"} <= nvda
    # futures only where a connected, catalogued contract exists -- and only NQ can run today
    futures = plan.of_domain(_F)
    assert {s for e in futures for s in e.symbols} == {"NQ", "MNQ", "HG", "NG"}
    assert {e.consequence_state for e in futures if e.execution_capable} == {
        "accelerator_demand", "hbm_demand", "advanced_foundry_utilization", "semiconductor_equipment_demand",
        "ai_spender_free_cash_flow", "ai_services_revenue",
    }
    assert not _expr(plan, "copper_demand", _F).execution_capable  # HG: catalogued, not certified, no data
    # no Options/Crypto expression forced in for symmetry
    assert not plan.of_domain(_O) and not plan.of_domain(_C)


def test_ai_slice_resolves_against_the_real_acquired_data():
    plan = _plan()
    nq = _spec(plan, "futures.momentum", "NQ").resolution
    assert nq.status is _S.AVAILABLE and nq.executable and nq.pit_safe is True
    assert (nq.dataset, nq.data_schema, nq.field) == ("GLBX.MDP3", "ohlcv-1m", "close")
    assert (nq.coverage_start, nq.coverage_end_exclusive) == (date(2018, 1, 1), date(2025, 1, 1))
    xlk = _spec(plan, "etf.momentum", "XLK").resolution
    assert xlk.status is _S.AVAILABLE_WITH_PROXY and xlk.dataset == "ARCX.PILLAR"
    # the declared window end is the vendor request's exclusive bound (last bar 2024-12-30)
    assert xlk.coverage_end_exclusive == date(2024, 12, 31)
    assert {m.target for m in plan.usable_measurements()} == {"NQ", "XLK", "QQQ", "XLU"}
    assert len(plan.measurable_expressions()) == 20


def test_real_catalog_probe_matches_the_certified_roots_and_stops_before_the_holdout():
    coverage = read_futures_bar_coverage()
    if not coverage:
        pytest.skip("CME catalog not present")
    assert {c.root for c in coverage} == set(RESEARCH_UNIVERSE)
    assert all(c.end_exclusive <= date(2025, 1, 1) and c.roll_overlap_artifacts > 0 for c in coverage)


def test_catalog_row_inside_the_holdout_fails_loudly(tmp_path):
    rows = [{"root": "ES", "component": "CONTINUOUS_FRONT", "dataset": "GLBX.MDP3", "schema": "ohlcv-1m",
             "start": "2024-01-01", "end": "2025-03-01"}]
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(rows))
    with pytest.raises(HoldoutAccessError):
        read_futures_bar_coverage(path)
    assert read_futures_bar_coverage(tmp_path / "absent.json") == ()


# ---------------------------------------------------------------------------
# multiple expressions per path; economic relevance vs mandate
# ---------------------------------------------------------------------------


def test_one_path_yields_several_expressions_across_domains():
    plan = _plan()
    copper = [e for e in plan.expressions if e.consequence_state == "copper_demand"]
    assert {(e.domain, e.status) for e in copper} == {
        (_F, ExpressionStatus.CONTINUES), (_Q, ExpressionStatus.NO_INSTRUMENT), (_E, ExpressionStatus.NO_INSTRUMENT),
    }
    path_id = copper[0].path_ids[0]
    assert set(plan.for_path(path_id)) == set(copper)
    accel = [e for e in plan.expressions if e.consequence_state == "accelerator_demand"]
    assert {(e.domain, e.form) for e in accel} == {
        (_Q, ExpressionForm.SINGLE_NAME), (_E, ExpressionForm.SECTOR_BASKET), (_E, ExpressionForm.BROAD_INDEX),
        (_F, ExpressionForm.BROAD_INDEX),
    }


def test_a_consequence_without_a_market_expression_is_a_concept_only_row():
    plan = _plan()
    for state in ("hbm_demand", "data_center_construction", "foundry_capacity_expansion"):
        e = _expr(plan, state, _Q)
        assert e.status is ExpressionStatus.NO_INSTRUMENT and not e.instruments and not e.measurement_ids
        assert "declared Equity universe" in e.status_note


def test_economic_relevance_is_identical_under_every_mandate_only_admission_changes():
    wide, narrow = _plan(mandate=ALL_ASSESSABLE), _plan(mandate=ResearchMandate(allowed_domains=(_Q, _F)))
    assert [e.expression_id for e in wide.expressions] == [e.expression_id for e in narrow.expressions]
    assert [e.rule_id for e in wide.expressions] == [e.rule_id for e in narrow.expressions]
    assert wide.signal_paths_id == narrow.signal_paths_id
    for e in narrow.of_domain(_E):
        assert e.status is ExpressionStatus.EXCLUDED_BY_MANDATE
        assert not e.instruments and not e.measurement_ids and not e.execution_capable
    assert not [m for m in narrow.measurements if m.domain is _E]
    assert next(s for s in narrow.domains if s.domain is _E).note.startswith("Excluded by your mandate (15")


def test_instrument_constraints_remove_expressions_honestly():
    from alpha_agent.core.instrument import InstrumentIdentity
    from alpha_agent.registry.enums import AssetDomain

    only_es = ResearchMandate(
        allowed_domains=(_F, _E),
        instrument_allowlist=(InstrumentIdentity(asset_domain=AssetDomain.FUTURES, symbol="ES"),),
        instrument_denylist=(InstrumentIdentity(asset_domain=AssetDomain.ETF, symbol="XLK"),),
    )
    plan = _plan(mandate=only_es)
    nq = _expr(plan, "accelerator_demand", _F)
    assert nq.status is ExpressionStatus.REMOVED_BY_CONSTRAINTS and nq.removed_by_constraints == ("NQ", "MNQ")
    assert _expr(plan, "accelerator_demand", _E, "Technology-sector fund").status is (
        ExpressionStatus.REMOVED_BY_CONSTRAINTS
    )
    assert _expr(plan, "accelerator_demand", _E, "Nasdaq-100 fund").status is ExpressionStatus.CONTINUES


def test_a_universe_from_another_mandate_is_refused():
    universe = resolve_allowed_universe(ALL_ASSESSABLE, capabilities=CAPS)
    scan = scan_initial_impact(UserDescribedEvent.create(AI_CAPEX, described_at=AT), ALL_ASSESSABLE, universe=universe)
    paths = discover_signal_paths(build_economic_mechanism_graph(scan))
    with pytest.raises(ValueError, match="different mandate"):
        build_asset_expressions(paths, universe, EVERY_DOMAIN)


# ---------------------------------------------------------------------------
# pressure (a hypothesis) and the shorting constraint
# ---------------------------------------------------------------------------


def test_pressure_follows_direction_times_relation_per_horizon():
    plan = _plan()
    fcf = _expr(plan, "ai_spender_free_cash_flow", _Q)
    assert [(p.horizon.value, p.pressure) for p in fcf.pressures] == [
        ("MONTHS", PressureSign.DOWN), ("YEARS", PressureSign.UP),
    ]
    assert _expr(plan, "foundry_capacity_expansion", _Q).pressures[0].pressure is None  # AMBIGUOUS relation
    fed = _plan(RATE_SURPRISE)
    treasuries = _expr(fed, "treasury_yields", _F)
    assert treasuries.relation is Polarity.NEGATIVE
    assert {p.pressure for p in treasuries.pressures} == {PressureSign.DOWN}  # yields up -> prices down


def test_shorting_constraint_is_a_note_never_a_filter():
    long_only = _plan(RATE_SURPRISE)
    treasuries = _expr(long_only, "treasury_yields", _F)
    assert treasuries.status is ExpressionStatus.CONTINUES
    assert any("Shorting not allowed" in n for n in treasuries.mandate_notes)
    shorting = ResearchMandate(allowed_domains=ALL_ASSESSABLE.allowed_domains, shorting_allowed=True)
    assert _expr(_plan(RATE_SURPRISE, shorting), "treasury_yields", _F).mandate_notes == ()


# ---------------------------------------------------------------------------
# unsupported domains: Options / Crypto
# ---------------------------------------------------------------------------

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


def test_allowed_but_unsupported_domains_are_reported_and_measured_as_unavailable():
    plan = _plan(mandate=EVERY_DOMAIN, library=_OPTIONS_AND_CRYPTO)
    options = _expr(plan, "accelerator_demand", _O)
    crypto = _expr(plan, "electricity_demand", _C)
    for e in (options, crypto):
        assert e.status is ExpressionStatus.DOMAIN_UNAVAILABLE and not e.instruments and not e.execution_capable
        specs = plan.measurements_for(e)
        assert specs and all(m.resolution.status is _S.DOMAIN_UNAVAILABLE for m in specs)
    assert {m.measurement_id for m in plan.measurements_for(options)} == {
        "options.atm_iv", "options.skew", "options.vol_term_structure", "options.expected_move",
    }
    assert {m.target for m in plan.measurements_for(options)} == {"NVDA"}
    assert all(m.resolution.data_role is DataProvenanceRole.SYNTHETIC for m in plan.measurements_for(crypto))
    # the default library never forces an Options/Crypto expression
    default = _plan(mandate=EVERY_DOMAIN)
    assert not default.of_domain(_O) and not default.of_domain(_C)
    summaries = {s.domain: s for s in default.domains}
    assert summaries[_O].support is DomainSupport.NOT_SUPPORTED and summaries[_O].note.startswith("Allowed, but")
    assert summaries[_C].support is DomainSupport.SYNTHETIC_ONLY


def test_no_options_or_crypto_execution_path_under_any_mandate():
    plan = _plan(mandate=EVERY_DOMAIN, library=_OPTIONS_AND_CRYPTO)
    for e in plan.expressions:
        if e.domain in (_O, _C):
            assert not e.execution_capable
    assert not [m for m in plan.usable_measurements() if m.domain in (_O, _C)]
    btc = _expr(plan, "electricity_demand", _F)  # CME crypto futures: catalogued, never acquired
    assert btc.status is ExpressionStatus.CONTINUES and not btc.execution_capable
    assert all(m.resolution.status is _S.MISSING for m in plan.measurements_for(btc)
               if m.requirement is DataRequirement.MARKET_BARS)
    with pytest.raises(ValidationError, match="execution_capable"):
        options = _expr(plan, "accelerator_demand", _O)
        AssetExpression.model_validate({**options.model_dump(), "execution_capable": True})


# ---------------------------------------------------------------------------
# point-in-time safety
# ---------------------------------------------------------------------------


def test_bars_are_knowable_only_at_interval_end():
    r = _resolve("futures.momentum", "NQ")
    assert r.pit_safe is True and r.publication_lag is PublicationLag.AT_BAR_CLOSE
    assert "interval START" in r.pit_rule and "next permitted execution point" in r.pit_rule


def test_hindsight_constituents_are_not_pit_safe_even_before_any_data_exists():
    r = _resolve("etf.breadth", "XLK")
    assert r.status is _S.NOT_PIT_SAFE and r.pit_safe is False
    assert _I.HINDSIGHT_MEMBERSHIP in r.issues and "2026-09-25" in r.missing_reason
    assert not r.historically_usable


def test_a_retrospective_registered_feature_makes_a_measurement_not_pit_safe():
    template = MeasurementTemplate(
        "futures.roll_timing", _F, "Roll timing", MeasurementRole.PRICE_RESPONSE, DataRequirement.MARKET_BARS,
        "fixture", "fixture", field="close", feature_kinds=("bars_until_next_roll",),
    )
    r = resolve_field(template, "NQ", support=DomainSupport.RESEARCH_READY, capabilities=CAPS)
    assert r.status is _S.NOT_PIT_SAFE and r.pit_safe is False and _I.RETROSPECTIVE_FEATURE in r.issues
    assert r.gap == "Retrospective (look-ahead) feature"


def test_missing_data_leaves_point_in_time_unknown_never_guessed():
    for measurement_id, symbol in (("equity.momentum", "NVDA"), ("futures.cot", "NQ"), ("equity.revenue_growth", "NVDA")):
        r = _resolve(measurement_id, symbol, support=DomainSupport.DATA_NOT_ACQUIRED if symbol == "NVDA"
                     else DomainSupport.RESEARCH_READY)
        assert r.status is _S.MISSING and r.pit_safe is None
    assert _resolve("futures.cot", "NQ").publication_lag is PublicationLag.WEEKLY_RELEASE
    assert _resolve("equity.revenue_growth", "NVDA").publication_lag is PublicationLag.FILING_ACCEPTANCE


def test_available_on_answers_history_and_refuses_the_holdout():
    r = _resolve("futures.momentum", "NQ")
    assert r.available_on(date(2020, 3, 16)) and not r.available_on(date(2017, 12, 29))
    with pytest.raises(HoldoutAccessError):
        r.available_on(date(2025, 1, 2))
    assert not _resolve("futures.momentum", "HG").available_on(date(2020, 3, 16))


def test_coverage_can_never_reach_into_the_holdout():
    leaked = CAPS.model_copy(update={"futures_bars": (
        FuturesBarCoverage(root="NQ", dataset="GLBX.MDP3", data_schema="ohlcv-1m", start=date(2018, 1, 1),
                           end_exclusive=date(2025, 6, 1), continuous_segments=1, roll_overlap_artifacts=1),
    )})
    with pytest.raises(HoldoutAccessError):
        _resolve("futures.momentum", "NQ", caps=leaked)
    ok = _resolve("futures.momentum", "NQ")
    with pytest.raises(ValidationError, match="holdout"):
        DataFieldResolution.model_validate({**ok.model_dump(), "coverage_end_exclusive": date(2025, 2, 1)})


def test_a_root_acquired_over_a_shorter_span_is_partial_and_usable_only_there():
    short = CAPS.model_copy(update={"futures_bars": FUTURES_BARS + (
        FuturesBarCoverage(root="HG", dataset="GLBX.MDP3", data_schema="ohlcv-1m", start=date(2023, 1, 1),
                           end_exclusive=date(2025, 1, 1), continuous_segments=1, roll_overlap_artifacts=0),
    )})
    r = _resolve("futures.momentum", "HG", caps=short)
    assert r.status is _S.PARTIAL and r.historically_usable
    assert r.available_on(date(2024, 5, 1)) and not r.available_on(date(2020, 5, 1))
    curve = _resolve("futures.term_structure", "HG", caps=short)
    assert curve.status is _S.MISSING and curve.gap == "Simultaneous contract months not acquired"


# ---------------------------------------------------------------------------
# proxies, missing fields, executability
# ---------------------------------------------------------------------------


def test_proxies_say_what_they_stand_in_for():
    etf = _resolve("etf.momentum", "XLU")
    assert etf.status is _S.AVAILABLE_WITH_PROXY and "total return" in etf.proxy_note
    volume = _resolve("futures.activity", "NQ")
    assert volume.status is _S.AVAILABLE_WITH_PROXY and "all listed months" in volume.proxy_note
    # the proxy is recorded, but only counts as PROXY once data is present
    revenue = _resolve("equity.revenue_growth", "NVDA", support=DomainSupport.DATA_NOT_ACQUIRED)
    assert revenue.proxy_note and _I.PROXY not in revenue.issues
    with pytest.raises(ValidationError, match="proxy"):
        DataFieldResolution.model_validate({**etf.model_dump(), "proxy_note": None})


def test_missing_fields_name_the_gap_and_never_a_dataset():
    r = _resolve("equity.momentum", "NVDA", support=DomainSupport.DATA_NOT_ACQUIRED)
    assert r.status is _S.MISSING and _I.DATA_NOT_ACQUIRED in r.issues
    assert (r.dataset, r.field, r.coverage_start) == (None, None, None)
    assert "XNAS.ITCH" in r.missing_reason and r.gap == "Equity daily bars not acquired"
    oi = _resolve("futures.open_interest", "NQ")
    assert oi.status is _S.MISSING and "statistics" in oi.missing_reason
    assert _resolve("etf.flows", "XLK").issues[0] is _I.NOT_INVESTIGATED


def test_fundamentals_follow_the_recorded_data_availability_finding(monkeypatch):
    assert _I.PAID_SOURCE_ONLY in _resolve("equity.revenue_growth", "NVDA").issues
    free = data_availability.finding("point_in_time_fundamentals").model_copy(
        update={"availability": data_availability.DataSourceAvailability.AVAILABLE_FREE_SELF_SERVE},
    )
    monkeypatch.setattr(data_availability, "FINDINGS",
                        tuple(free if f.concept == free.concept else f for f in data_availability.FINDINGS))
    r = _resolve("equity.revenue_growth", "NVDA")
    assert _I.CONNECTOR_NOT_BUILT in r.issues and _I.PAID_SOURCE_ONLY not in r.issues


def test_executability_reads_the_live_feature_registry():
    assert _resolve("futures.momentum", "NQ").executable
    empty = _resolve("futures.momentum", "NQ", registry=FeatureRegistry())
    assert empty.status is _S.NOT_EXECUTABLE and _I.FEATURE_NOT_REGISTERED in empty.issues
    rs = _resolve("etf.relative_strength", "XLU")
    assert rs.status is _S.NOT_EXECUTABLE and rs.pit_safe is True and rs.dataset == "ARCX.PILLAR"
    assert rs.gap == "No cross-instrument features in the feature registry"
    curve = _resolve("futures.term_structure", "NQ")
    assert curve.status is _S.NOT_EXECUTABLE and _I.PARTIAL_HISTORY in curve.issues
    assert "28 observed rolls" in curve.coverage_note


def test_status_follows_issues_by_fixed_precedence():
    assert status_for(()) is _S.AVAILABLE
    assert status_for({_I.PROXY, _I.PARTIAL_HISTORY}) is _S.PARTIAL
    assert status_for({_I.PARTIAL_HISTORY, _I.CROSS_INSTRUMENT}) is _S.NOT_EXECUTABLE
    assert status_for({_I.CROSS_INSTRUMENT, _I.DATA_NOT_ACQUIRED}) is _S.MISSING
    assert status_for({_I.DATA_NOT_ACQUIRED, _I.HINDSIGHT_MEMBERSHIP}) is _S.NOT_PIT_SAFE
    assert status_for({_I.HINDSIGHT_MEMBERSHIP, _I.SYNTHETIC_ONLY}) is _S.DOMAIN_UNAVAILABLE
    ok = _resolve("futures.momentum", "NQ")
    with pytest.raises(ValidationError, match="does not follow"):
        DataFieldResolution.model_validate({**ok.model_dump(), "status": _S.PARTIAL})


# ---------------------------------------------------------------------------
# no synthetic-to-real promotion
# ---------------------------------------------------------------------------


def test_synthetic_data_can_never_be_promoted_to_real():
    ok = _resolve("futures.momentum", "NQ")
    with pytest.raises(ValidationError, match="never promoted"):
        DataFieldResolution.model_validate({**ok.model_dump(), "data_role": DataProvenanceRole.SYNTHETIC})
    with pytest.raises(ValidationError, match="REAL"):
        DataFieldResolution.model_validate({**ok.model_dump(), "data_role": None})
    crypto = _resolve("crypto.funding", None, support=DomainSupport.SYNTHETIC_ONLY)
    assert crypto.status is _S.DOMAIN_UNAVAILABLE and crypto.data_role is DataProvenanceRole.SYNTHETIC
    assert not crypto.historically_usable and not crypto.available_on(date(2020, 1, 2))
    with pytest.raises(ValidationError):
        DataFieldResolution.model_validate({**crypto.model_dump(), "issues": (), "status": _S.AVAILABLE})


def test_usable_resolutions_are_always_real_pit_safe_and_executable():
    for text in (AI_CAPEX, OPEC_CUT, RATE_SURPRISE):
        plan = _plan(text, EVERY_DOMAIN, library=_OPTIONS_AND_CRYPTO)
        for m in plan.measurements:
            r = m.resolution
            if r.historically_usable:
                assert r.pit_safe is True and r.executable and r.data_role is DataProvenanceRole.REAL
                assert r.gap is None and r.coverage_end_exclusive <= date(2025, 1, 1)
            else:
                assert r.missing_reason and r.gap


# ---------------------------------------------------------------------------
# provenance, reuse, determinism
# ---------------------------------------------------------------------------


def test_the_ideal_measurement_is_the_phase_1_measurable_variable():
    spec = build_measurement_spec(
        DEFAULT_EXPRESSION_LIBRARY.template("etf.momentum"), "XLK", support=DomainSupport.RESEARCH_READY,
        capabilities=CAPS,
    )
    assert type(spec.variable) is MeasurableVariable
    assert spec.variable.point_in_time_available is spec.resolution.pit_safe is True
    assert spec.variable.point_in_time_note == spec.resolution.pit_rule
    with pytest.raises(ValidationError, match="point-in-time claim"):
        spec.model_validate({**spec.model_dump(), "variable": {**spec.variable.model_dump(),
                                                              "point_in_time_available": None}})


def test_resolutions_cite_the_authorities_they_read():
    assert any("data/catalog/real_cme_dataset.json" in p for p in _resolve("futures.momentum", "NQ").provenance)
    assert any("etf.universe.DATASET_WINDOWS" in p for p in _resolve("etf.momentum", "XLK").provenance)
    assert any("data_availability:point_in_time_fundamentals" in p
               for p in _resolve("equity.revenue_growth", "NVDA").provenance)
    assert any("features.registry: return, trend_strength" in p for p in _resolve("futures.momentum", "NQ").provenance)


def test_plan_is_deterministic_and_carries_its_inputs():
    a, b = _plan(), _plan()
    assert a.fingerprint() == b.fingerprint() and a.fingerprint().startswith("assetexpr1:")
    assert a.plane == "HYPOTHESIS" and a.capabilities == CAPS
    assert a.mandate_fingerprint == ALL_ASSESSABLE.fingerprint()
    assert AssetExpressionPlan.model_validate_json(a.model_dump_json()) == a
    # measurements are shared, never duplicated per expression
    assert len({m.spec_id for m in a.measurements}) == len(a.measurements)
    nq = _spec(a, "futures.momentum", "NQ")
    assert len(nq.expression_ids) == 6


def test_blockers_count_primary_only_and_additional_blockers_honestly():
    plan = _plan()
    blockers = plan.blockers()
    bars = blockers[0]
    assert bars.gap == "Equity daily bars not acquired"
    # primary for 24, but only 16 become usable by acquiring bars: relative strength also needs a
    # cross-instrument feature, and ETF breadth waits on hindsight-free constituents first.
    assert (bars.primary_for, bars.only_blocker_for, bars.also_blocks) == (24, 16, 3)
    assert set(bars.requirements) == {DataRequirement.MARKET_BARS, DataRequirement.BENCHMARK_BARS,
                                      DataRequirement.CONSTITUENT_MEMBERSHIP}
    unusable = [m for m in plan.measurements if not m.resolution.historically_usable]
    assert sum(b.primary_for for b in blockers) == len(unusable)
    assert sum(b.blocks for b in blockers) == sum(len(m.resolution.gaps) for m in unusable)
    assert [b.primary_for for b in blockers] == sorted((b.primary_for for b in blockers), reverse=True)
    # "only blocker for" is exactly what closing that gap alone would make usable
    for b in blockers:
        assert b.only_blocker_for == sum(1 for m in unusable if [g.label for g in m.resolution.gaps] == [b.gap])
    registered = next(b for b in blockers if b.gap == "No registered feature computes it")
    assert registered.primary_for == 0 and registered.also_blocks > 0  # never primary, still a real blocker


def test_a_measurement_keeps_every_blocker_primary_first():
    plan = _plan()
    rs = _spec(plan, "equity.relative_strength", "NVDA").resolution
    assert [g.label for g in rs.gaps] == [
        "Equity daily bars not acquired", "No cross-instrument features in the feature registry",
    ]
    assert rs.gap == "Equity daily bars not acquired"
    assert rs.additional_gaps == ("No cross-instrument features in the feature registry",)
    assert "Also blocked by: No cross-instrument features" in rs.missing_reason
    breadth = _spec(plan, "etf.breadth", "XLK").resolution
    assert [g.issue for g in breadth.gaps] == [
        _I.HINDSIGHT_MEMBERSHIP, _I.DATA_NOT_ACQUIRED, _I.FEATURE_NOT_REGISTERED,
    ]
    assert _spec(plan, "futures.momentum", "NQ").resolution.gaps == ()
    # dropping or reordering a blocker is refused
    with pytest.raises(ValidationError, match="none is dropped"):
        DataFieldResolution.model_validate({**rs.model_dump(), "gaps": rs.model_dump()["gaps"][:1]})
    with pytest.raises(ValidationError, match="primary gap"):
        DataFieldResolution.model_validate({**rs.model_dump(), "gaps": rs.model_dump()["gaps"][::-1]})


# ---------------------------------------------------------------------------
# fidelity: how directly an expression carries its consequence
# ---------------------------------------------------------------------------


def test_every_rule_states_its_fidelity_and_it_fits_the_form():
    assert all(r.fidelity is not None for r in DEFAULT_EXPRESSION_LIBRARY.rules)
    rules = DEFAULT_EXPRESSION_LIBRARY.rules
    with pytest.raises(ValueError, match="fidelity"):
        _validate(ExpressionLibrary(rules=(dataclasses.replace(rules[0], fidelity=None),) + rules[1:],
                                    templates=MEASUREMENT_TEMPLATES))
    with pytest.raises(ValueError, match="does not fit form"):
        _validate(ExpressionLibrary(
            rules=(dataclasses.replace(rules[0], fidelity=ExpressionFidelity.CONSTITUENT_EXPOSURE),) + rules[1:],
            templates=MEASUREMENT_TEMPLATES,
        ))


def test_reviewed_ambiguous_rows_are_labelled_honestly():
    plan = _plan()
    rate = _plan(RATE_SURPRISE)
    assert _expr(plan, "power_equipment_demand", _Q).fidelity is ExpressionFidelity.SEGMENT_EXPOSURE  # CAT
    for domain, concept in ((_E, "Technology-sector fund"), (_E, "Nasdaq-100 fund"), (_F, "Nasdaq-100 index futures")):
        assert _expr(plan, "advanced_foundry_utilization", domain, concept).fidelity is (
            ExpressionFidelity.ECOSYSTEM_PROXY
        )
        assert _expr(plan, "semiconductor_equipment_demand", domain, concept).fidelity is (
            ExpressionFidelity.CONSTITUENT_EXPOSURE
        )
    long_duration = _expr(rate, "equity_valuation_multiples", _Q)
    assert long_duration.symbols == ("AAPL", "MSFT", "NVDA")
    assert long_duration.fidelity is ExpressionFidelity.MACRO_PROXY
    assert _expr(rate, "housing_activity", _Q, "Home-improvement retailer").fidelity is (
        ExpressionFidelity.ECOSYSTEM_PROXY
    )
    # a direct underlying and a direct company are never the same claim as a basket
    assert _expr(plan, "copper_demand", _F).fidelity is ExpressionFidelity.DIRECT_UNDERLYING
    assert _expr(plan, "accelerator_demand", _Q).fidelity is ExpressionFidelity.DIRECT_COMPANY
    assert _expr(plan, "compute_demand", _Q, "Cloud platforms").fidelity is ExpressionFidelity.SEGMENT_EXPOSURE


def test_fidelity_is_descriptive_and_never_changes_admission_order_or_measurement():
    plan = _plan()
    flattened = ExpressionLibrary(
        rules=tuple(dataclasses.replace(r, fidelity=ExpressionFidelity.MACRO_PROXY) for r in
                    DEFAULT_EXPRESSION_LIBRARY.rules),
        templates=MEASUREMENT_TEMPLATES,
    )
    other = _plan(library=flattened)
    strip = lambda p: [e.model_dump(exclude={"fidelity"}) for e in p.expressions]
    assert strip(plan) == strip(other)
    assert plan.measurements == other.measurements and plan.blockers() == other.blockers()
    assert all(isinstance(f.value, str) for f in ExpressionFidelity)  # a category, not a number


# ---------------------------------------------------------------------------
# Phase E guard: execution-capable is not candidate-ready
# ---------------------------------------------------------------------------


def test_candidate_signal_basis_requires_a_usable_measurement_on_a_research_ready_instrument():
    plan = _plan()
    nq = _expr(plan, "accelerator_demand", _F)
    basis = plan.candidate_signal_basis(nq)
    assert {(m.symbol, m.measurement_id) for m in basis} == {("NQ", "futures.momentum"), ("NQ", "futures.activity")}
    assert all(m.resolution.historically_usable and m.resolution.pit_safe is True and m.resolution.executable
               and m.resolution.data_role is DataProvenanceRole.REAL for m in basis)
    for e in (_expr(plan, "copper_demand", _F), _expr(plan, "hbm_demand", _Q), _expr(plan, "accelerator_demand", _Q)):
        with pytest.raises(CandidateSignalPrerequisiteError):
            plan.candidate_signal_basis(e)


def test_execution_capable_alone_is_not_candidate_ready():
    plan = _plan(registry=FeatureRegistry())  # nothing registered: no measurement is executable
    nq = _expr(plan, "accelerator_demand", _F)
    assert nq.execution_capable
    with pytest.raises(CandidateSignalPrerequisiteError, match="not candidate-ready"):
        plan.candidate_signal_basis(nq)
    options = _expr(_plan(mandate=EVERY_DOMAIN, library=_OPTIONS_AND_CRYPTO), "accelerator_demand", _O)
    with pytest.raises(CandidateSignalPrerequisiteError):
        _plan(mandate=EVERY_DOMAIN, library=_OPTIONS_AND_CRYPTO).candidate_signal_basis(options)


def test_an_unseeded_event_has_no_expressions():
    plan = _plan(GEOPOLITICAL)
    assert plan.is_empty and not plan.measurements
    assert asset_expression_view.summary_line(plan) is None and asset_expression_view.chat_text(plan) is None


def test_a_consequence_without_a_rule_is_a_reported_library_gap():
    sparse = ExpressionLibrary(
        rules=tuple(r for r in DEFAULT_EXPRESSION_LIBRARY.rules if r.state_id != "copper_demand"),
        templates=MEASUREMENT_TEMPLATES,
    )
    plan = _plan(library=sparse)
    assert plan.unmapped_consequences == ("copper_demand",)
    assert not [e for e in plan.expressions if e.consequence_state == "copper_demand"]


# ---------------------------------------------------------------------------
# the reviewed library
# ---------------------------------------------------------------------------


def test_library_validation_refuses_drift():
    rules = DEFAULT_EXPRESSION_LIBRARY.rules
    with pytest.raises(ValueError, match="outside the ETF universe"):
        _validate(ExpressionLibrary(rules=rules + (
            dataclasses.replace(rules[0], domain=_E, symbols=("SMH",), measures=()),), templates=MEASUREMENT_TEMPLATES))
    with pytest.raises(ValueError, match="no expression rule"):
        _validate(ExpressionLibrary(rules=tuple(r for r in rules if r.state_id != "copper_demand"),
                                    templates=MEASUREMENT_TEMPLATES))
    with pytest.raises(ValueError, match="concept-specific"):
        _validate(ExpressionLibrary(rules=(dataclasses.replace(rules[0], measures=("equity.momentum",)),) + rules[1:],
                                    templates=MEASUREMENT_TEMPLATES))
    # Phase C's exposure bridge and Phase D's rules must agree
    with pytest.raises(ValueError, match="Utilities sector ETF"):
        _validate(ExpressionLibrary(rules=tuple(r for r in rules if "XLU" not in r.symbols),
                                    templates=MEASUREMENT_TEMPLATES))


def test_every_seeded_consequence_has_a_rule_and_every_template_resolves():
    for text in (AI_CAPEX, OPEC_CUT, RATE_SURPRISE):
        assert _plan(text).unmapped_consequences == ()
    for t in MEASUREMENT_TEMPLATES:
        support = {_O: DomainSupport.NOT_SUPPORTED, _C: DomainSupport.SYNTHETIC_ONLY}.get(
            t.domain, DomainSupport.RESEARCH_READY,
        )
        symbol = {_F: "NQ", _E: "XLK", _Q: "NVDA"}.get(t.domain)
        assert resolve_field(t, symbol, support=support, capabilities=CAPS).status in _S


# ---------------------------------------------------------------------------
# boundaries
# ---------------------------------------------------------------------------


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


@pytest.mark.parametrize("path", PHASE_D_SOURCES, ids=lambda p: p.name)
def test_phase_d_never_imports_the_evidence_plane_an_llm_or_the_network(path):
    forbidden = (
        "alpha_agent.registry.sqlite_registry", "alpha_agent.validation", "alpha_agent.strategy",
        "alpha_agent.screening", "alpha_agent.discovery", "alpha_agent.holdout", "alpha_agent.knowledge",
        "alpha_agent.agents", "alpha_agent.marketdata.databento_provider", "alpha_agent.marketdata.ibkr_provider",
        "anthropic", "databento", "requests", "httpx", "urllib",
    )
    for imported in _imports(path):
        assert not any(imported == f or imported.startswith(f + ".") for f in forbidden), (path.name, imported)


def test_no_phase_d_schema_carries_a_trade_return_or_verdict_field():
    from alpha_agent.news_alpha import asset_expression, measurement

    forbidden = {
        "expected_return", "return", "probability", "p_value", "sharpe", "score", "weight", "portfolio_weight",
        "position_size", "buy", "sell", "action", "signal", "verdict", "experiment_id", "experiment_identity",
        "alpha", "quantity", "order", "target_weight",
    }
    for model in (asset_expression.AssetExpression, asset_expression.AssetExpressionPlan,
                  asset_expression.ExpressionPressure, asset_expression.MeasurementBlocker,
                  asset_expression.DomainExpressionSummary, measurement.MeasurementSpec,
                  measurement.DataFieldResolution):
        assert not (set(model.model_fields) & forbidden), model.__name__
    rendered = _plan().model_dump_json().upper()
    for word in ('"BUY"', '"SELL"', "STRONG BUY", "PRICE TARGET"):
        assert word not in rendered


def test_building_expressions_never_touches_the_registry():
    if not REGISTRY_PATH.exists():
        pytest.skip("registry not present")
    before = hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
    for text in (AI_CAPEX, OPEC_CUT, RATE_SURPRISE):
        _plan(text, EVERY_DOMAIN)
    assert hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest() == before


# ---------------------------------------------------------------------------
# view helpers
# ---------------------------------------------------------------------------


def test_view_helpers_summarize_without_verdict_language():
    plan = _plan()
    line = asset_expression_view.summary_line(plan)
    assert line.startswith("Asset expression: 38 ways to express these consequences (Futures 8 · ETF 15 · Equity 15)")
    assert "20 measurable on 2018-2024 history via NQ, XLK, QQQ, XLU" in line
    assert ("Largest gap: Equity daily bars not acquired (primary blocker for 24 measurements; the only blocker "
            "for 16)") in line
    assert asset_expression_view.expander_label(plan).endswith("38 expressions · 20 measurable today")
    rows = asset_expression_view.readiness_rows(plan, _F)
    nq = next(r for r in rows if r["Instrument"] == "NQ")
    assert nq["Price momentum"] == "Available" and nq["Term structure & carry"] == "Not executable"
    assert nq["Spot-futures basis"] == "--"  # applies to the underlying only
    field = next(r for r in asset_expression_view.field_rows(plan)
                 if r["Instrument"] == "XLK" and r["Measurement"] == "Breadth")
    assert field["Status"] == "Not PIT-safe" and field["Point-in-time"] == "No"
    gap = asset_expression_view.gap_rows(plan)[0]
    assert (gap["Gap"], gap["Primary blocker for"], gap["Only blocker for"], gap["Also blocks"]) == (
        "Equity daily bars not acquired", 24, 16, 3,
    )
    rs = next(r for r in asset_expression_view.field_rows(plan)
              if r["Instrument"] == "NVDA" and r["Measurement"] == "Relative strength")
    assert rs["Blocked by"] == "Equity daily bars not acquired + No cross-instrument features in the feature registry"
    assert next(r for r in asset_expression_view.expression_rows(plan)
                if r["Expression"] == "Power-generation equipment makers")["Fidelity"] == "One segment of the company"
    assert asset_expression_view.chat_text(plan).startswith("ASSET EXPRESSION: 38 expressions")
    view_source = (REPO_ROOT / "python" / "alpha_agent" / "ui" / "asset_expression_view.py").read_text()
    assert "unlocks every measurement" not in view_source
