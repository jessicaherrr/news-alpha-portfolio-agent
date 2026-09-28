"""Phase 1 -- end-to-end deterministic translation pipeline tests (prompt 1
sections 4/5/6/7/8/9/10). Uses the REAL local registry and the REAL
`alpha_agent.features.REGISTRY` -- everything asserted here is genuine
repository state, never a fabricated researchability label.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import alpha_agent.features.compute  # noqa: F401 -- register feature defs
import pytest
from alpha_agent.agents.mechanism_agent import (
    FactorCandidateProposal,
    MeasurableVariableProposal,
    MechanismCandidateProposal,
    MechanismTranslationProposal,
)
from alpha_agent.market_intel import mapping
from alpha_agent.market_intel.event_schemas import EventImportance, ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.translation.pipeline import (
    build_observation_translation,
    observation_from_event,
    observation_from_news,
    render_summary_text,
)
from alpha_agent.translation.schemas import ResearchabilityStatus
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

_NOW = datetime.now(UTC)


def _news_item(category: NewsCategory, *, news_id: str = "N1") -> MarketNewsItem:
    return MarketNewsItem(
        news_id=news_id,
        headline="A real official release headline",
        source_name="Official Source",
        source_type=NewsSourceType.OFFICIAL,
        source_url="https://example.gov/release",
        published_at=_NOW - timedelta(hours=2),
        retrieved_at=_NOW,
        related_products=mapping.products_for_category(category),
        related_asset_classes=mapping.asset_classes_for_category(category),
        category=category,
        mapping_reason=mapping.mapping_reason_for_category(category),
    )


def _event(category: NewsCategory, *, event_id: str = "E1") -> ScheduledMarketEvent:
    return ScheduledMarketEvent(
        event_id=event_id,
        name="A real scheduled release",
        source_name="Official Source",
        source_url="https://example.gov/calendar",
        scheduled_at=_NOW + timedelta(days=1),
        timezone="America/New_York",
        category=category.value,
        importance=EventImportance.HIGH,
        importance_rule="test",
        affected_products=mapping.products_for_category(category),
        mapping_reason=mapping.mapping_reason_for_category(category),
        retrieved_at=_NOW,
    )


def _registry() -> ExperimentRegistry:
    return ExperimentRegistry(services.REGISTRY_PATH)


# ---------------------------------------------------------------------------
# Observation construction
# ---------------------------------------------------------------------------


def test_observation_from_news_requires_root_in_items_own_mapping():
    item = _news_item(NewsCategory.PETROLEUM)
    with pytest.raises(ValueError, match="not among"):
        observation_from_news(item, root_symbol="ES")  # PETROLEUM never maps to ES


def test_observation_from_news_rejects_uncertified_root():
    item = _news_item(NewsCategory.FOMC_POLICY)
    with pytest.raises(ValueError, match="certified"):
        observation_from_news(item, root_symbol="6E")  # FOMC maps to FX roots, none certified


def test_observation_from_news_preserves_real_published_at_as_observed_at():
    item = _news_item(NewsCategory.PETROLEUM)
    obs = observation_from_news(item, root_symbol="CL")
    assert obs.observed_at == item.published_at
    assert obs.evidence_refs == (f"news_id={item.news_id}", f"source_url={item.source_url}", "category=PETROLEUM")


def test_observation_from_event_requires_root_in_items_own_mapping():
    event = _event(NewsCategory.PETROLEUM)
    with pytest.raises(ValueError, match="not among"):
        observation_from_event(event, root_symbol="ES")


# ---------------------------------------------------------------------------
# translation -- worked examples (energy, macro/rates, equity-index macro)
# ---------------------------------------------------------------------------


def test_petroleum_translation_matches_prompt_1_worked_example():
    """Prompt 1 section 8's worked example, produced by the real pipeline:
    realized inventory-change research may be possible (AVAILABLE trend
    factor from real registered features); true consensus-surprise research
    is NOT currently supported (DATA_MISSING); curve/term-structure
    confirmation is NOT_EXECUTABLE (no cross-contract FeatureRegistry kind)."""
    obs = observation_from_news(_news_item(NewsCategory.PETROLEUM), root_symbol="CL")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg)

    statuses = {f.concept: f.researchability for f in result.factor_candidates}
    assert ResearchabilityStatus.AVAILABLE in statuses.values()
    assert ResearchabilityStatus.DATA_MISSING in statuses.values()
    assert ResearchabilityStatus.NOT_EXECUTABLE in statuses.values()

    assert result.hypothesis is not None
    best = next(f for f in result.factor_candidates if f.researchability is ResearchabilityStatus.AVAILABLE)
    assert set(result.hypothesis.required_features) <= set(best.available_feature_kinds)
    assert result.research_gap_note is None
    assert result.generated_by == "DETERMINISTIC_LIBRARY"


def test_fomc_translation_on_rates_root():
    obs = observation_from_news(_news_item(NewsCategory.FOMC_POLICY, news_id="N2"), root_symbol="ZN")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg)
    assert result.hypothesis is not None
    assert result.hypothesis.universe == ["ZN"]
    mechs = {m.mechanism.value for m in result.mechanism_candidates}
    assert "VOLATILITY_BREAKOUT" in mechs
    assert "CROSS_MARKET_LEAD_LAG" in mechs


def test_fomc_translation_on_equity_index_root_is_a_distinct_observation():
    obs = observation_from_news(_news_item(NewsCategory.FOMC_POLICY, news_id="N3"), root_symbol="NQ")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg)
    assert result.hypothesis is not None
    assert result.hypothesis.universe == ["NQ"]
    assert result.research_memory.root_symbol == "NQ"


def test_event_derived_observation_translates_too():
    obs = observation_from_event(_event(NewsCategory.PETROLEUM, event_id="E2"), root_symbol="CL")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg)
    assert obs.event_type == "SCHEDULED_EVENT"
    assert result.mechanism_candidates


# ---------------------------------------------------------------------------
# research gap -- an honest, no-hypothesis outcome (prompt 1 section 8)
# ---------------------------------------------------------------------------


def test_other_category_produces_an_honest_research_gap_not_a_hallucinated_factor():
    obs = observation_from_news(
        MarketNewsItem(
            news_id="N-OTHER", headline="An uncategorized release", source_name="Official Source",
            source_type=NewsSourceType.OFFICIAL, source_url="https://example.gov/x",
            published_at=_NOW - timedelta(hours=1), retrieved_at=_NOW,
            related_products=("CL",), related_asset_classes=(), category=NewsCategory.OTHER,
            mapping_reason="no rule matched",
        ),
        root_symbol="CL",
    )
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg)
    assert result.hypothesis is None
    assert result.factor_candidates == ()
    assert result.mechanism_candidates == ()
    assert result.research_gap_note is not None
    assert "RESEARCH GAP" in result.research_gap_note


# ---------------------------------------------------------------------------
# provenance never mixes "what the system already has" with "what a
# proposer merely thinks would be interesting" (prompt 1 section 9)
# ---------------------------------------------------------------------------


def test_deterministic_library_factors_never_carry_claude_provenance():
    obs = observation_from_news(_news_item(NewsCategory.PETROLEUM, news_id="N4"), root_symbol="CL")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg)
    provenances = {f.provenance.value for f in result.factor_candidates}
    assert "CLAUDE_PROPOSED_PROXY" not in provenances
    mech_provenances = {m.provenance.value for m in result.mechanism_candidates}
    assert mech_provenances == {"DETERMINISTIC_LIBRARY"}


def test_render_summary_text_is_short_and_includes_falsification():
    obs = observation_from_news(_news_item(NewsCategory.PETROLEUM, news_id="N5"), root_symbol="CL")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg)
    text = render_summary_text(result)
    assert "OBSERVATION:" in text
    assert "FALSIFICATION:" in text
    assert len(text.splitlines()) < 12


# ---------------------------------------------------------------------------
# Phase 1 acceptance patch, section 1: Claude free text must never establish
# data availability -- only a typed repository capability can.
# ---------------------------------------------------------------------------


def _single_mechanism_proposal(measurable_variables=()) -> list[MechanismCandidateProposal]:
    return [
        MechanismCandidateProposal(
            mechanism="TREND", explanation="a Claude-proposed mechanism", causal_chain=[], evidence_basis="y",
        ),
    ]


def test_claude_proposed_variable_availability_is_always_unknown_regardless_of_wording():
    """Even text containing the exact marker words the pre-patch code used
    to trust ("ohlcv", "price series", "already-acquired", "raw contract
    data", "FeatureRegistry") must never establish availability -- prose is
    never scanned for capability claims."""
    proposal = MechanismTranslationProposal(
        mechanism_candidates=_single_mechanism_proposal(),
        measurable_variables=[
            MeasurableVariableProposal(
                name="v1", economic_meaning="m",
                required_source="the root's own OHLCV price series (already-acquired, raw contract data, in the FeatureRegistry)",
            ),
            MeasurableVariableProposal(name="v2", economic_meaning="m", required_source="an external consensus survey"),
        ],
        factor_candidates=[
            FactorCandidateProposal(
                concept="c", mechanism="TREND", transform_or_proxy="t",
                proposed_feature_kinds=["trend_strength"], required_external_data=[],
            ),
        ],
    )
    obs = observation_from_news(_news_item(NewsCategory.PETROLEUM, news_id="N6"), root_symbol="CL")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg, mechanism_proposal=proposal)
    assert len(result.measurable_variables) == 2
    for v in result.measurable_variables:
        assert v.point_in_time_available is None
        assert "not independently verified" in v.point_in_time_note.lower()


def test_deterministic_library_variables_keep_their_explicit_availability_facts():
    """The fix targets Claude-sourced free text only -- the hand-authored
    deterministic-library variables keep their explicit, code-reviewed
    True/False facts unchanged."""
    obs = observation_from_news(_news_item(NewsCategory.PETROLEUM, news_id="N7"), root_symbol="CL")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg)
    values = {v.point_in_time_available for v in result.measurable_variables}
    assert values == {True, False}
    assert None not in values


# ---------------------------------------------------------------------------
# Phase 1 acceptance patch, section 2: ONLY AVAILABLE may auto-produce a
# hypothesis / enable "Research This" -- never PARTIALLY_AVAILABLE (never
# silently substitute a weaker proxy and call it the same factor).
# ---------------------------------------------------------------------------


def _proposal_with_single_factor(*, feature_kinds: list[str], external_data: list[str]) -> MechanismTranslationProposal:
    return MechanismTranslationProposal(
        mechanism_candidates=_single_mechanism_proposal(),
        measurable_variables=[],
        factor_candidates=[
            FactorCandidateProposal(
                concept="only-factor-in-this-proposal", mechanism="TREND", transform_or_proxy="t",
                proposed_feature_kinds=feature_kinds, required_external_data=external_data,
            ),
        ],
    )


def _translate_single_factor(*, feature_kinds: list[str], external_data: list[str], news_id: str):
    proposal = _proposal_with_single_factor(feature_kinds=feature_kinds, external_data=external_data)
    obs = observation_from_news(_news_item(NewsCategory.PETROLEUM, news_id=news_id), root_symbol="CL")
    with _registry() as reg:
        return build_observation_translation(obs, registry=reg, mechanism_proposal=proposal)


def test_available_factor_auto_enables_a_hypothesis():
    result = _translate_single_factor(feature_kinds=["trend_strength"], external_data=[], news_id="AV1")
    assert result.factor_candidates[0].researchability is ResearchabilityStatus.AVAILABLE
    assert result.hypothesis is not None
    assert result.research_gap_note is None


def test_partially_available_factor_never_auto_creates_a_hypothesis():
    result = _translate_single_factor(
        feature_kinds=["trend_strength"], external_data=["a consensus expectation series"], news_id="PA1",
    )
    assert result.factor_candidates[0].researchability is ResearchabilityStatus.PARTIALLY_AVAILABLE
    assert result.hypothesis is None
    assert result.research_gap_note is not None
    assert result.research_gap_note.startswith("PARTIAL SUPPORT")


def test_data_missing_factor_never_auto_creates_a_hypothesis():
    result = _translate_single_factor(feature_kinds=[], external_data=["a consensus expectation series"], news_id="DM1")
    assert result.factor_candidates[0].researchability is ResearchabilityStatus.DATA_MISSING
    assert result.hypothesis is None
    assert result.research_gap_note is not None
    assert result.research_gap_note.startswith("RESEARCH GAP")


def test_not_executable_factor_never_auto_creates_a_hypothesis():
    result = _translate_single_factor(feature_kinds=[], external_data=[], news_id="NE1")
    assert result.factor_candidates[0].researchability is ResearchabilityStatus.NOT_EXECUTABLE
    assert result.hypothesis is None
    assert result.research_gap_note is not None
    assert result.research_gap_note.startswith("RESEARCH GAP")


def test_a_mix_of_partially_available_and_data_missing_never_promotes_the_partial_one():
    """Even when PARTIALLY_AVAILABLE is the 'best' status present, it must
    never be silently promoted into a hypothesis just because nothing better
    exists in the same proposal."""
    proposal = MechanismTranslationProposal(
        mechanism_candidates=_single_mechanism_proposal(),
        measurable_variables=[],
        factor_candidates=[
            FactorCandidateProposal(
                concept="partial-one", mechanism="TREND", transform_or_proxy="t",
                proposed_feature_kinds=["trend_strength"], required_external_data=["some external series"],
            ),
            FactorCandidateProposal(
                concept="missing-one", mechanism="TREND", transform_or_proxy="t",
                proposed_feature_kinds=[], required_external_data=["another external series"],
            ),
        ],
    )
    obs = observation_from_news(_news_item(NewsCategory.PETROLEUM, news_id="MIX1"), root_symbol="CL")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg, mechanism_proposal=proposal)
    statuses = {f.researchability for f in result.factor_candidates}
    assert statuses == {ResearchabilityStatus.PARTIALLY_AVAILABLE, ResearchabilityStatus.DATA_MISSING}
    assert result.hypothesis is None
    assert result.research_gap_note.startswith("PARTIAL SUPPORT")


def test_three_acceptance_examples_still_produce_a_hypothesis_after_the_gating_fix():
    """Section 5's explicit requirement: the existing 3 acceptance examples
    (all deterministic-library-sourced, all resolve via an AVAILABLE factor)
    must behave identically after restricting auto-hypothesis to AVAILABLE
    only."""
    cl = observation_from_news(_news_item(NewsCategory.PETROLEUM, news_id="ACC1"), root_symbol="CL")
    zn = observation_from_news(_news_item(NewsCategory.FOMC_POLICY, news_id="ACC2"), root_symbol="ZN")
    nq = observation_from_news(_news_item(NewsCategory.FOMC_POLICY, news_id="ACC3"), root_symbol="NQ")
    with _registry() as reg:
        for obs in (cl, zn, nq):
            result = build_observation_translation(obs, registry=reg)
            assert result.hypothesis is not None
            best = next(f for f in result.factor_candidates if f.researchability is ResearchabilityStatus.AVAILABLE)
            assert result.hypothesis.title.startswith(best.concept)
