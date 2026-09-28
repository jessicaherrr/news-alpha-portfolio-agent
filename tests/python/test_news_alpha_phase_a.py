"""News Alpha Phase A -- Research Mandate -> Allowed Asset Universe ->
News/Event -> Initial Impact Scan -> existing translation boundary.

Pure, offline tests: fixed `DomainCapabilities`, fixed timestamps, no
network, no LLM. The one registry-touching test (the end-to-end hand-off into
`build_observation_translation`) proves the registry file is byte-identical
afterwards.
"""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alpha_agent.core.instrument import InstrumentIdentity
from alpha_agent.market_intel import mapping
from alpha_agent.market_intel.event_schemas import EventImportance, ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.marketdata.product_catalog import RESEARCH_UNIVERSE
from alpha_agent.news_alpha import (
    CHANNELS,
    DEFAULT_MANDATE,
    AvailabilityStatus,
    CategoryBasis,
    DomainCapabilities,
    DomainSupport,
    HorizonAlignment,
    ImpactLevel,
    InitialImpactAssessment,
    InitialImpactScan,
    LiquidityRequirement,
    MandateDomain,
    MandateStore,
    PressureSign,
    ResearchMandate,
    UserDescribedEvent,
    build_translation_handoffs,
    mandate_domain_for,
    registry_domain_for,
    resolve_allowed_universe,
    scan_initial_impact,
    scan_many,
)
from alpha_agent.recommendation.profile import (
    HoldingPeriod,
    InvestorProfile,
    MaxDrawdown,
    RiskStyle,
)
from alpha_agent.registry.enums import AssetDomain
from alpha_agent.translation.mechanism_library import CATEGORY_MECHANISM_TEMPLATES
from alpha_agent.translation.pipeline import observation_from_news
from alpha_agent.translation.schemas import CERTIFIED_ROOTS

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "python" / "alpha_agent" / "news_alpha"
UI_CONTEXT = REPO_ROOT / "python" / "alpha_agent" / "ui" / "news_alpha_context.py"

CAPS = DomainCapabilities(etf_market_data_acquired=True, equity_market_data_acquired=False)
AT = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)

EQUITY_FUTURES = ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES))

# The prompt's vertical-slice events, phrased as a user would type them.
AI_CAPEX = "Hyperscalers announce a major AI infrastructure spending increase, lifting data center capex guidance"
OPEC_CUT = "OPEC+ agrees a production cut of 1 million barrels per day"
RATE_SURPRISE = "Rate-policy surprise: the Fed unexpectedly raised rates by 50 basis points"


def _user(text: str) -> UserDescribedEvent:
    return UserDescribedEvent.create(text, described_at=AT)


def _scan(event, mandate: ResearchMandate = EQUITY_FUTURES) -> InitialImpactScan:
    return scan_initial_impact(event, mandate, capabilities=CAPS)


def _news(category: NewsCategory, headline: str, *, news_id: str = "N-1") -> MarketNewsItem:
    return MarketNewsItem(
        news_id=news_id, headline=headline, source_name="EIA", source_type=NewsSourceType.OFFICIAL,
        source_url=f"https://eia.gov/{news_id}", published_at=AT - timedelta(hours=2), retrieved_at=AT,
        related_products=mapping.products_for_category(category),
        related_asset_classes=mapping.asset_classes_for_category(category), category=category,
        mapping_reason=mapping.mapping_reason_for_category(category),
    )


def _levels(scan: InitialImpactScan) -> dict[MandateDomain, ImpactLevel | None]:
    return {a.asset_domain: a.impact_level for a in scan.assessments}


# ---------------------------------------------------------------------------
# ResearchMandate
# ---------------------------------------------------------------------------


def test_mandate_serialization_round_trips_and_fingerprint_is_stable():
    mandate = ResearchMandate(
        allowed_domains=(MandateDomain.FUTURES, MandateDomain.EQUITY),
        instrument_allowlist=(InstrumentIdentity(asset_domain=AssetDomain.FUTURES, symbol="CL"),),
        shorting_allowed=True, max_gross_leverage=2.0, liquidity_requirement=LiquidityRequirement.HIGH,
        risk_profile=InvestorProfile(holding_period=HoldingPeriod.WEEKS, risk_style=RiskStyle.CONSERVATIVE),
    )
    restored = ResearchMandate.model_validate_json(mandate.model_dump_json())
    assert restored == mandate
    assert restored.fingerprint() == mandate.fingerprint()
    assert mandate.investment_horizon is HoldingPeriod.WEEKS
    assert mandate.risk_style is RiskStyle.CONSERVATIVE
    assert json.loads(mandate.model_dump_json())["schema_version"] == "research-mandate/1"


def test_mandate_is_canonicalized_so_equivalent_mandates_fingerprint_identically():
    a = ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES, MandateDomain.EQUITY))
    b = ResearchMandate(allowed_domains=(MandateDomain.FUTURES, MandateDomain.EQUITY))
    assert a.allowed_domains == (MandateDomain.FUTURES, MandateDomain.EQUITY)
    assert a.fingerprint() == b.fingerprint()


def test_mandate_rejects_empty_domains_and_contradictory_instrument_constraints():
    with pytest.raises(ValueError):
        ResearchMandate(allowed_domains=())
    cl = InstrumentIdentity(asset_domain=AssetDomain.FUTURES, symbol="CL")
    with pytest.raises(ValueError, match="not an allowed domain"):
        ResearchMandate(allowed_domains=(MandateDomain.EQUITY,), instrument_allowlist=(cl,))
    with pytest.raises(ValueError, match="both allowed and denied"):
        ResearchMandate(allowed_domains=(MandateDomain.FUTURES,), instrument_allowlist=(cl,), instrument_denylist=(cl,))
    with pytest.raises(ValueError):
        ResearchMandate(allowed_domains=(MandateDomain.FUTURES,), max_gross_leverage=0)


def test_mandate_domains_cover_every_registry_asset_domain_by_identical_value():
    for domain in AssetDomain:
        assert mandate_domain_for(domain).value == domain.value
        assert registry_domain_for(mandate_domain_for(domain)) is domain
    assert registry_domain_for(MandateDomain.OPTIONS) is None
    assert registry_domain_for(MandateDomain.CRYPTO) is None
    assert DEFAULT_MANDATE.allowed_domains == tuple(mandate_domain_for(d) for d in AssetDomain)


def test_mandate_store_persists_access_only_and_composes_the_current_profile(tmp_path):
    store = MandateStore(tmp_path / "m.json")
    profile = InvestorProfile(risk_style=RiskStyle.AGGRESSIVE)
    assert store.load(risk_profile=profile).allowed_domains == DEFAULT_MANDATE.allowed_domains

    saved = ResearchMandate(allowed_domains=(MandateDomain.ETF,), shorting_allowed=True, risk_profile=profile)
    store.save(saved)
    assert "risk_profile" not in json.loads(store.path.read_text())
    other = InvestorProfile(risk_style=RiskStyle.CONSERVATIVE)
    loaded = store.load(risk_profile=other)
    assert loaded.allowed_domains == (MandateDomain.ETF,) and loaded.shorting_allowed
    assert loaded.risk_profile == other  # the profile half is never a stale second copy

    store.path.write_text("{not json")
    assert store.load(risk_profile=other).allowed_domains == DEFAULT_MANDATE.allowed_domains


# ---------------------------------------------------------------------------
# Allowed Asset Universe
# ---------------------------------------------------------------------------


def test_universe_resolution_is_deterministic_and_composes_existing_domain_universes():
    u1 = resolve_allowed_universe(EQUITY_FUTURES, capabilities=CAPS)
    u2 = resolve_allowed_universe(EQUITY_FUTURES, capabilities=CAPS)
    assert u1 == u2
    assert [s.domain for s in u1.scopes] == list(MandateDomain)
    assert u1.allowed_domains == (MandateDomain.FUTURES, MandateDomain.EQUITY)

    futures = u1.scope(MandateDomain.FUTURES)
    certified = tuple(i.symbol for i in futures.instruments if i.research_ready)
    assert set(certified) == set(RESEARCH_UNIVERSE) == set(CERTIFIED_ROOTS)
    assert u1.scope(MandateDomain.EQUITY).support is DomainSupport.DATA_NOT_ACQUIRED
    assert not any(i.research_ready for i in u1.scope(MandateDomain.EQUITY).instruments)


def test_excluded_and_unsupported_domains_never_enumerate_instruments():
    universe = resolve_allowed_universe(EQUITY_FUTURES, capabilities=CAPS)
    for domain in (MandateDomain.ETF, MandateDomain.OPTIONS, MandateDomain.CRYPTO):
        scope = universe.scope(domain)
        assert not scope.in_mandate and scope.instruments == ()
    all_domains = resolve_allowed_universe(ResearchMandate(allowed_domains=tuple(MandateDomain)), capabilities=CAPS)
    assert all_domains.scope(MandateDomain.OPTIONS).support is DomainSupport.NOT_SUPPORTED
    assert all_domains.scope(MandateDomain.CRYPTO).support is DomainSupport.SYNTHETIC_ONLY


def test_instrument_allowlist_restricts_and_reports_unresolved_symbols():
    mandate = ResearchMandate(
        allowed_domains=(MandateDomain.FUTURES,),
        instrument_allowlist=(
            InstrumentIdentity(asset_domain=AssetDomain.FUTURES, symbol="ES"),
            InstrumentIdentity(asset_domain=AssetDomain.FUTURES, symbol="NOPE"),
        ),
    )
    scope = resolve_allowed_universe(mandate, capabilities=CAPS).scope(MandateDomain.FUTURES)
    assert [i.symbol for i in scope.instruments] == ["ES"]
    assert scope.unresolved_allowlist == ("NOPE",)
    assert "CL" in {i.symbol for i in scope.removed_by_constraints}


def test_channel_table_exposures_all_resolve_against_the_live_domain_universes():
    universe = resolve_allowed_universe(ResearchMandate(allowed_domains=tuple(MandateDomain)), capabilities=CAPS)
    for defn in CHANNELS:
        if defn.translation_category is not None:
            assert CATEGORY_MECHANISM_TEMPLATES[defn.translation_category], defn.rule_id
        for exposure in defn.exposures:
            scope = universe.scope(exposure.domain)
            symbols = {i.symbol for i in scope.instruments}
            groups = {i.group for i in scope.instruments}
            assert set(exposure.symbols) <= symbols, (defn.rule_id, exposure.label)
            assert set(exposure.groups) <= groups, (defn.rule_id, exposure.label)
            if scope.support is not DomainSupport.SYNTHETIC_ONLY:
                assert exposure.symbols or exposure.groups, (defn.rule_id, exposure.label)


# ---------------------------------------------------------------------------
# Initial Impact Scan -- the vertical slice
# ---------------------------------------------------------------------------


def test_example_a_ai_infrastructure_spending():
    scan = _scan(_user(AI_CAPEX))
    levels = _levels(scan)
    assert levels[MandateDomain.EQUITY].rank >= ImpactLevel.HIGH.rank  # equity likely relevant
    assert ImpactLevel.LOW.rank <= levels[MandateDomain.FUTURES].rank < levels[MandateDomain.EQUITY].rank  # possibly
    assert levels[MandateDomain.OPTIONS] is None and levels[MandateDomain.CRYPTO] is None
    for domain in (MandateDomain.OPTIONS, MandateDomain.CRYPTO, MandateDomain.ETF):
        assert scan.assessment(domain).availability is AvailabilityStatus.EXCLUDED_BY_MANDATE
    equity = scan.assessment(MandateDomain.EQUITY)
    assert "Technology" in {m.group_label for m in equity.candidate_markets}
    # Spender vs. supplier: no price sign is justified before the Mechanism Graph.
    assert equity.possible_direction == ()
    assert any("Mechanism Graph" in u for u in equity.uncertainty)


def test_example_b_opec_production_cut():
    scan = _scan(_user(OPEC_CUT))
    levels = _levels(scan)
    assert levels[MandateDomain.FUTURES].rank >= ImpactLevel.HIGH.rank
    assert levels[MandateDomain.EQUITY].rank >= ImpactLevel.MEDIUM.rank
    assert levels[MandateDomain.FUTURES].rank > levels[MandateDomain.EQUITY].rank
    futures = scan.assessment(MandateDomain.FUTURES)
    assert "CL" in {m.symbol for m in futures.candidate_markets}
    assert futures.availability is AvailabilityStatus.RESEARCH_READY
    equity = scan.assessment(MandateDomain.EQUITY)
    assert {"XOM", "CVX"} <= {m.symbol for m in equity.candidate_markets if m.group_label == "Energy"}
    assert [(d.exposure_label, d.pressure) for d in futures.possible_direction] == [
        ("Crude & refined-product futures", PressureSign.UP)
    ]


def test_a_signed_group_plus_an_unsigned_tertiary_group_is_not_reported_as_unsigned():
    """Regression (live screenshot pass): OPEC's Equity exposure has a signed
    Energy group AND an unsigned tertiary Industrials group -- the assessment
    must show the Energy direction without ALSO claiming no sign applies."""
    equity = _scan(_user(OPEC_CUT)).assessment(MandateDomain.EQUITY)
    assert [d.exposure_label for d in equity.possible_direction] == ["Energy sector"]
    assert not any("no textbook first-order sign" in u for u in equity.uncertainty)


def test_example_c_rate_policy_surprise_maps_to_multiple_allowed_domains():
    mandate = ResearchMandate(allowed_domains=(MandateDomain.FUTURES, MandateDomain.ETF, MandateDomain.EQUITY))
    scan = _scan(_user(RATE_SURPRISE), mandate)
    relevant = [a.asset_domain for a in scan.assessments if a.deserves_research]
    assert set(relevant) == {MandateDomain.FUTURES, MandateDomain.ETF, MandateDomain.EQUITY}
    futures = scan.assessment(MandateDomain.FUTURES)
    assert futures.impact_level is ImpactLevel.VERY_HIGH  # PRIMARY exposure + a stated surprise
    assert {"ZN", "ES"} <= {m.symbol for m in futures.candidate_markets}
    # Textbook sign only: a tightening surprise pushes Treasury futures prices down.
    assert [(d.exposure_label, d.pressure) for d in futures.possible_direction] == [("Treasury futures", PressureSign.DOWN)]


def test_examples_are_not_special_cased_paraphrases_triage_the_same_way():
    for text, domain in (
        ("Big cloud providers boost spending on AI chips and new datacenters", MandateDomain.EQUITY),
        ("Saudi-led OPEC+ to cut crude output by 2 million barrels a day", MandateDomain.FUTURES),
        ("Central bank delivers a surprise rate hike; FOMC turns hawkish", MandateDomain.FUTURES),
    ):
        scan = _scan(_user(text))
        assert scan.assessment(domain).deserves_research, text
    for path in PACKAGE_DIR.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for example in (AI_CAPEX, OPEC_CUT, RATE_SURPRISE):
            assert example not in source, f"{path.name} hard-codes an example headline"


def test_an_unrelated_event_is_none_everywhere_it_is_assessed():
    scan = _scan(_user("Local bakery wins a regional pastry award"))
    assert scan.channels == ()
    for a in scan.assessments:
        assert a.impact_level in (None, ImpactLevel.NONE)


def test_surprise_raises_one_level_over_a_routine_item_on_the_same_topic():
    routine = _scan(_news(NewsCategory.PETROLEUM, "What goes into diesel prices?"))
    surprise = _scan(_news(NewsCategory.PETROLEUM, "EIA reports a larger-than-expected crude draw"))
    r, s = routine.assessment(MandateDomain.FUTURES), surprise.assessment(MandateDomain.FUTURES)
    assert r.impact_level is ImpactLevel.HIGH and s.impact_level is ImpactLevel.VERY_HIGH
    assert r.confidence == s.confidence  # both topics certain (source category); only salience differs


def test_scheduled_high_importance_event_counts_as_salient():
    event = ScheduledMarketEvent(
        event_id="FOMC-1", name="FOMC Meeting", source_name="Federal Reserve", source_url="https://fed.gov",
        scheduled_at=AT + timedelta(days=3), timezone="America/New_York", category="FOMC_POLICY",
        importance=EventImportance.HIGH, importance_rule="fomc-policy-decision-always-high/1",
        affected_products=mapping.products_for_category(NewsCategory.FOMC_POLICY),
        mapping_reason="m", retrieved_at=AT,
    )
    assert _scan(event).assessment(MandateDomain.FUTURES).impact_level is ImpactLevel.VERY_HIGH


def test_single_weak_cue_lowers_level_and_confidence():
    scan = _scan(_user("Commentary mentions gasoline"))
    futures = scan.assessment(MandateDomain.FUTURES)
    assert futures.impact_level is ImpactLevel.MEDIUM  # PRIMARY exposure, single cue
    assert futures.confidence.value == "LOW"
    assert any("single cue" in u for u in futures.uncertainty)


def test_conflicting_direction_cues_leave_direction_undetermined():
    scan = _scan(_user("Fed officials split between a rate hike and a rate cut at the FOMC"))
    ev = next(c for c in scan.channels if c.channel.value == "MONETARY_POLICY")
    assert ev.conflicting_shift_cues and ev.shift is None
    assert scan.assessment(MandateDomain.FUTURES).possible_direction == ()


def test_crypto_event_reaches_cme_crypto_futures_even_when_crypto_domain_is_excluded():
    scan = _scan(_user("Spot bitcoin and ether rally as stablecoin rules clear"))
    futures = scan.assessment(MandateDomain.FUTURES)
    assert {m.symbol for m in futures.candidate_markets} == {"BTC", "MBT", "ETH"}
    assert futures.availability is AvailabilityStatus.OBSERVATION_ONLY  # catalogued, not certified
    assert scan.assessment(MandateDomain.CRYPTO).availability is AvailabilityStatus.EXCLUDED_BY_MANDATE

    all_domains = ResearchMandate(allowed_domains=tuple(MandateDomain))
    crypto = _scan(_user("Spot bitcoin and ether rally as stablecoin rules clear"), all_domains).assessment(
        MandateDomain.CRYPTO
    )
    assert crypto.availability is AvailabilityStatus.SYNTHETIC_ONLY and crypto.impact_level is not None


def test_unsupported_options_domain_is_reported_not_assessed():
    all_domains = ResearchMandate(allowed_domains=tuple(MandateDomain))
    options = _scan(_user(RATE_SURPRISE), all_domains).assessment(MandateDomain.OPTIONS)
    assert options.impact_level is None
    assert options.availability is AvailabilityStatus.DOMAIN_NOT_SUPPORTED


def test_instrument_constraints_that_remove_every_candidate_are_explained():
    es_only = ResearchMandate(
        allowed_domains=(MandateDomain.FUTURES,),
        instrument_allowlist=(InstrumentIdentity(asset_domain=AssetDomain.FUTURES, symbol="ES"),),
    )
    futures = _scan(_user(OPEC_CUT), es_only).assessment(MandateDomain.FUTURES)
    assert futures.impact_level is ImpactLevel.NONE
    assert futures.availability is AvailabilityStatus.NO_CANDIDATE_MARKETS
    assert "instrument constraints exclude every candidate" in futures.availability_note
    assert "CL" in futures.availability_note


# ---------------------------------------------------------------------------
# asset access vs. risk tolerance
# ---------------------------------------------------------------------------


def _triage_signature(a: InitialImpactAssessment) -> tuple:
    return (a.asset_domain, a.impact_level, a.availability, tuple(m.symbol for m in a.candidate_markets),
            a.possible_direction, a.confidence)


@pytest.mark.parametrize("text", [AI_CAPEX, OPEC_CUT, RATE_SURPRISE, "Bank run hits regional banks; contagion fears"])
def test_risk_constraints_never_change_impact_or_scope(text):
    cautious = ResearchMandate(
        allowed_domains=EQUITY_FUTURES.allowed_domains, shorting_allowed=False, max_gross_leverage=1.0,
        liquidity_requirement=LiquidityRequirement.HIGH,
        risk_profile=InvestorProfile(risk_style=RiskStyle.CONSERVATIVE, max_drawdown=MaxDrawdown.PCT_5,
                                     holding_period=HoldingPeriod.INTRADAY),
    )
    bold = ResearchMandate(
        allowed_domains=EQUITY_FUTURES.allowed_domains, shorting_allowed=True, max_gross_leverage=5.0,
        liquidity_requirement=LiquidityRequirement.ANY,
        risk_profile=InvestorProfile(risk_style=RiskStyle.AGGRESSIVE, max_drawdown=MaxDrawdown.PCT_20_PLUS,
                                     holding_period=HoldingPeriod.WEEKS),
    )
    a, b = _scan(_user(text), cautious), _scan(_user(text), bold)
    assert [_triage_signature(x) for x in a.assessments] == [_triage_signature(x) for x in b.assessments]
    ua = resolve_allowed_universe(cautious, capabilities=CAPS)
    ub = resolve_allowed_universe(bold, capabilities=CAPS)
    assert [s.instruments for s in ua.scopes] == [s.instruments for s in ub.scopes]


def test_same_risk_profile_on_different_domain_access_changes_only_scope():
    """The dual of the test above: no domain is ever treated as inherently
    riskier -- a Conservative profile may allow Futures and an Aggressive one
    may be Equity-only; scope follows access alone."""
    conservative = InvestorProfile(risk_style=RiskStyle.CONSERVATIVE)
    futures_only = ResearchMandate(allowed_domains=(MandateDomain.FUTURES,), risk_profile=conservative)
    equity_only = ResearchMandate(allowed_domains=(MandateDomain.EQUITY,),
                                  risk_profile=InvestorProfile(risk_style=RiskStyle.AGGRESSIVE))
    assert _scan(_user(OPEC_CUT), futures_only).assessment(MandateDomain.FUTURES).deserves_research
    assert _scan(_user(OPEC_CUT), equity_only).assessment(MandateDomain.EQUITY).deserves_research


def test_shorting_and_horizon_only_add_mandate_notes():
    no_short = _scan(_user(RATE_SURPRISE)).assessment(MandateDomain.FUTURES)
    assert any("net short" in n for n in no_short.mandate_notes)
    shortable = ResearchMandate(allowed_domains=EQUITY_FUTURES.allowed_domains, shorting_allowed=True)
    assert not any("net short" in n for n in _scan(_user(RATE_SURPRISE), shortable).assessment(
        MandateDomain.FUTURES).mandate_notes)

    intraday = ResearchMandate(allowed_domains=EQUITY_FUTURES.allowed_domains,
                               risk_profile=InvestorProfile(holding_period=HoldingPeriod.INTRADAY))
    equity = _scan(_user(AI_CAPEX), intraday).assessment(MandateDomain.EQUITY)
    assert equity.horizon_alignment is HorizonAlignment.MISALIGNED
    assert any("horizon" in n for n in equity.mandate_notes)
    assert equity.impact_level == _scan(_user(AI_CAPEX)).assessment(MandateDomain.EQUITY).impact_level


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def test_scan_preserves_event_mandate_and_rule_provenance():
    item = _news(NewsCategory.PETROLEUM, "EIA reports a larger-than-expected crude draw")
    scan = _scan(item)
    assert scan.event.source == item and scan.event.event_id == item.news_id
    assert scan.mandate_fingerprint == EQUITY_FUTURES.fingerprint()
    assert scan.assessment_pass == "INITIAL" and "supersede" in scan.not_final_note
    futures = scan.assessment(MandateDomain.FUTURES)
    assert futures.event_id == item.news_id
    assert {"channel/crude-oil-supply/1", "impact-level/1", "salience-lexicon/1"} <= set(futures.reasoning_provenance)
    assert "alpha_agent.marketdata.product_catalog.PRODUCT_CATALOG" in futures.reasoning_provenance
    ev = scan.channels[0]
    assert ev.bases[0].value == "SOURCE_CATEGORY" and ev.strength.value == "STRONG"


def test_scan_many_orders_by_relevance_then_recency():
    routine = _news(NewsCategory.PETROLEUM, "What goes into diesel prices?", news_id="A")
    surprise = _news(NewsCategory.PETROLEUM, "EIA reports a larger-than-expected crude draw", news_id="B")
    other = _news(NewsCategory.OTHER, "Uranium production tripled", news_id="C")
    universe = resolve_allowed_universe(EQUITY_FUTURES, capabilities=CAPS)
    order = [s.event.event_id for s in scan_many([other, routine, surprise], EQUITY_FUTURES, universe=universe)]
    assert order == ["B", "A", "C"]


def test_scanning_with_a_universe_from_another_mandate_is_refused():
    universe = resolve_allowed_universe(DEFAULT_MANDATE, capabilities=CAPS)
    with pytest.raises(ValueError, match="different mandate"):
        scan_initial_impact(_user(OPEC_CUT), EQUITY_FUTURES, universe=universe)


def test_user_described_event_id_is_a_deterministic_content_hash():
    a, b = _user("  OPEC+   cuts output "), _user("OPEC+ cuts output")
    assert a.event_id == b.event_id and a.text == "OPEC+ cuts output"
    with pytest.raises(ValueError):
        UserDescribedEvent(event_id="x", text="abc", described_at=datetime(2026, 1, 1))  # noqa: DTZ001 -- deliberately naive


# ---------------------------------------------------------------------------
# hand-off into the EXISTING translation boundary
# ---------------------------------------------------------------------------


def test_source_category_handoff_reuses_the_existing_observation_constructor():
    item = _news(NewsCategory.PETROLEUM, "EIA reports a larger-than-expected crude draw", news_id="UI-1")
    plan = build_translation_handoffs(_scan(item))
    assert [h.root_symbol for h in plan.handoffs] == ["CL"]
    handoff = plan.handoffs[0]
    assert handoff.category_basis is CategoryBasis.SOURCE_CATEGORY
    assert handoff.key == "news:UI-1:CL"  # == translation_context.CandidateObservation.key
    existing = observation_from_news(item, root_symbol="CL")
    assert handoff.observation.model_dump(exclude={"evidence_refs"}) == existing.model_dump(exclude={"evidence_refs"})
    assert handoff.observation.evidence_refs[: len(existing.evidence_refs)] == existing.evidence_refs
    assert f"mandate={EQUITY_FUTURES.fingerprint()}" in handoff.observation.evidence_refs
    equity_gap = next(g for g in plan.gaps if g.asset_domain is MandateDomain.EQUITY)
    assert "certified Futures roots only" in equity_gap.reason


def test_user_event_handoff_routes_by_channel_with_an_explicit_fidelity_note():
    plan = build_translation_handoffs(_scan(_user(OPEC_CUT)))
    handoff = next(h for h in plan.handoffs if h.root_symbol == "CL")
    assert handoff.category_basis is CategoryBasis.IMPACT_CHANNEL
    assert handoff.translation_category == NewsCategory.PETROLEUM.value
    attrs = handoff.observation.structured_attributes
    assert attrs["source_category"] == "NONE" and attrs["category_basis"] == "IMPACT_CHANNEL"
    assert handoff.observation.event_type == "USER_DESCRIBED_EVENT"
    assert "approximation" in handoff.fidelity_note


def test_channel_without_template_hands_off_honestly_as_no_template():
    plan = build_translation_handoffs(_scan(_user(AI_CAPEX)))
    nq = next(h for h in plan.handoffs if h.root_symbol == "NQ")
    assert nq.category_basis is CategoryBasis.NO_TEMPLATE
    assert nq.observation.structured_attributes["category"] == NewsCategory.OTHER.value
    assert {g.asset_domain for g in plan.gaps} == {MandateDomain.EQUITY}


def test_excluded_or_irrelevant_domains_produce_no_handoff():
    equity_only = ResearchMandate(allowed_domains=(MandateDomain.EQUITY,))
    plan = build_translation_handoffs(_scan(_user(OPEC_CUT), equity_only))
    assert plan.handoffs == ()
    assert build_translation_handoffs(_scan(_user("Local bakery wins a pastry award"))).handoffs == ()


REGISTRY_PATH = REPO_ROOT / "data" / "registry" / "experiments.sqlite"


@pytest.mark.skipif(not REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout")
def test_end_to_end_mandate_to_translation_never_writes_the_registry():
    from alpha_agent.registry.sqlite_registry import ExperimentRegistry
    from alpha_agent.translation.pipeline import build_observation_translation

    before = hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
    plan = build_translation_handoffs(_scan(_user(OPEC_CUT)))
    with ExperimentRegistry(REGISTRY_PATH) as reg:
        translation = build_observation_translation(plan.handoffs[0].observation, registry=reg)
    assert translation.observation.root_symbol == "CL"
    assert translation.mechanism_candidates  # the PETROLEUM templates the channel routed to
    assert translation.hypothesis is not None and translation.hypothesis.universe == ["CL"]
    assert hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest() == before


# ---------------------------------------------------------------------------
# boundaries: no registry writes, no verdict/alpha fields, market_intel isolation
# ---------------------------------------------------------------------------


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


_NEWS_ALPHA_SOURCES = sorted(PACKAGE_DIR.glob("*.py")) + [UI_CONTEXT]


@pytest.mark.parametrize("path", _NEWS_ALPHA_SOURCES, ids=lambda p: p.name)
def test_news_alpha_never_imports_the_scientific_write_plane_or_knowledge(path):
    forbidden = (
        "alpha_agent.registry.sqlite_registry", "alpha_agent.validation", "alpha_agent.strategy",
        "alpha_agent.screening", "alpha_agent.discovery", "alpha_agent.holdout", "alpha_agent.knowledge",
        "alpha_agent.agents.llm", "alpha_agent.agents.mechanism_agent",
    )
    for imported in _imports(path):
        assert not any(imported == f or imported.startswith(f + ".") for f in forbidden), (path.name, imported)


@pytest.mark.parametrize("path", _NEWS_ALPHA_SOURCES, ids=lambda p: p.name)
def test_news_alpha_never_writes_a_registry_or_news_store(path):
    source = path.read_text(encoding="utf-8")
    for token in ("ExperimentRegistry(", "insert_", "append_attempt", "INSERT", "UPDATE ", ".add_item(",
                  ".upsert_event(", "requests.", "urllib", "httpx"):
        assert token not in source, (path.name, token)


def test_market_intel_never_imports_news_alpha():
    for path in (REPO_ROOT / "python" / "alpha_agent" / "market_intel").rglob("*.py"):
        assert not any(i.startswith("alpha_agent.news_alpha") for i in _imports(path)), path.name


def test_no_schema_carries_a_return_probability_trade_weight_or_verdict_field():
    from alpha_agent.news_alpha import handoff, schemas, universe

    forbidden = {
        "expected_return", "return", "probability", "probability_of_profit", "p_value", "sharpe", "score",
        "weight", "portfolio_weight", "position_size", "buy", "sell", "action", "signal", "verdict",
        "scientific_verdict", "research_promise", "experiment_id", "experiment_identity", "alpha",
    }
    models = [
        schemas.InitialImpactAssessment, schemas.InitialImpactScan, schemas.CandidateMarket,
        schemas.ChannelEvidence, schemas.DirectionalPressure, handoff.TranslationHandoff, handoff.TranslationGap,
        universe.AllowedAssetUniverse, universe.DomainScope, universe.UniverseInstrument, ResearchMandate,
    ]
    for model in models:
        assert not (set(model.model_fields) & forbidden), model.__name__
    assert not ({lvl.value for lvl in ImpactLevel} & {"BUY", "SELL", "HOLD", "PASS", "FAIL"})


def test_scan_text_disclaims_being_alpha_or_a_trade_instruction():
    scan = _scan(_user(OPEC_CUT))
    assert "Not an expected return, probability, trade instruction, portfolio weight, or scientific verdict" in (
        scan.not_final_note
    )
    rendered = json.dumps(scan.model_dump(mode="json")).upper()
    for word in ('"BUY"', '"SELL"', "STRONG BUY", "GUARANTEED"):
        assert word not in rendered
