"""Phase 3 (Agentic Alpha Evolution) -- Context-Aware Alpha Retrieval +
Transparent Ranking, hardened by the Phase 3 semantic hardening patch:

1. Scientific Reliability must preserve verdict meaning (INCONCLUSIVE never
   becomes REJECTED; PASS + REJECT across families becomes MIXED, never the
   most favorable verdict).
2. `related_experiment_count` is never labeled "replication" --
   `IndependentReplicationLevel` is always `NOT_AVAILABLE` in V1.
3. Category IMPORTANCE != objective event MAGNITUDE.
4. A genuinely personalized LOW User Fit must remain LOW, never floored to
   MEDIUM.
5. Prior same-category/root events are "RELATED PRIOR EVENT OCCURRENCES",
   never a claim of matched historical market context.

Mirrors the existing Phase 1/2/B1 test style: pure-function unit tests need
no live data at all; the integration tests at the bottom use the REAL,
already-committed local registry (read-only, never mutated), same pattern as
`test_alpha_memory_lookup.py`.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from alpha_agent.context_retrieval import build_context_retrieval, build_market_context_fingerprint
from alpha_agent.context_retrieval.mechanism_signals import context_match_for
from alpha_agent.context_retrieval.ranking import (
    explain_rank,
    rank_candidates,
    suggest_next_experiment,
)
from alpha_agent.context_retrieval.retrieval import (
    _independent_replication,
    _scientific_reliability,
)
from alpha_agent.context_retrieval.schemas import (
    UNKNOWN_OBJECTIVE_MAGNITUDE,
    ContextMatchLevel,
    ContextRetrievalResult,
    FreshnessBucket,
    IndependentReplicationLevel,
    MarketContextFingerprint,
    RankDimensionScores,
    RankedResearchCandidate,
    RelatedPriorEventOccurrence,
    ResearchCoverageLevel,
    ScientificReliabilityLevel,
)
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.translation.schemas import MechanismCandidate, MechanismProvenance, Observation
from alpha_agent.ui import services

_OBSERVED_AT = datetime(2026, 9, 22, 5, 3, 53, tzinfo=UTC)


def _fingerprint(**overrides) -> MarketContextFingerprint:
    base = {
        "root_symbol": "CL",
        "event_type": "MARKET_NEWS",
        "event_category": "PETROLEUM",
        "affected_products": ("CL", "MCL", "RB", "HO"),
        "observed_at": _OBSERVED_AT,
        "evaluated_at": _OBSERVED_AT,
        "trend": "Up",
        "volatility": "Moderate",
        "curve_state": "BACKWARDATION",
        "related_market_confirming": 1,
        "related_market_total": 2,
    }
    base.update(overrides)
    return build_market_context_fingerprint(**base)


# ---------------------------------------------------------------------------
# MarketContextFingerprint -- deterministic, reproducible, honest validation
# ---------------------------------------------------------------------------


def test_fingerprint_is_deterministic_and_reproducible():
    a = _fingerprint()
    b = _fingerprint()
    assert a.fingerprint_hash == b.fingerprint_hash
    assert a.fingerprint_hash.startswith("marketcontextfingerprint2:")


def test_fingerprint_changes_when_a_hashed_component_changes():
    a = _fingerprint(trend="Up")
    b = _fingerprint(trend="Down")
    assert a.fingerprint_hash != b.fingerprint_hash


def test_fingerprint_never_hashes_raw_continuous_freshness_seconds():
    """Two evaluations a few minutes apart (still the same freshness bucket)
    must hash identically -- only the discrete bucket enters the hash, never
    raw seconds (module docstring: "the hash would be different on every
    call" otherwise)."""
    a = _fingerprint(evaluated_at=_OBSERVED_AT)
    b = _fingerprint(evaluated_at=datetime(2026, 9, 22, 5, 8, 0, tzinfo=UTC))
    assert a.freshness_seconds != b.freshness_seconds
    assert a.freshness_bucket == b.freshness_bucket == FreshnessBucket.FRESH
    assert a.fingerprint_hash == b.fingerprint_hash


@pytest.mark.parametrize(
    ("hours_later", "expected"),
    [(1, FreshnessBucket.FRESH), (24, FreshnessBucket.RECENT), (24 * 10, FreshnessBucket.STALE), (24 * 60, FreshnessBucket.OLD)],
)
def test_freshness_bucketing(hours_later, expected):
    from datetime import timedelta

    fp = _fingerprint(evaluated_at=_OBSERVED_AT + timedelta(hours=hours_later))
    assert fp.freshness_bucket == expected


def test_fingerprint_rejects_naive_datetime():
    with pytest.raises(ValueError, match="UTC-aware"):
        build_market_context_fingerprint(
            root_symbol="CL", event_type="MARKET_NEWS", event_category="PETROLEUM", affected_products=("CL",),
            observed_at=datetime(2026, 9, 22), evaluated_at=_OBSERVED_AT,  # noqa: DTZ001 -- deliberately naive, testing the guard
            trend="Up", volatility="Moderate", curve_state=None,
            related_market_confirming=0, related_market_total=0,
        )


def test_fingerprint_rejects_evaluated_before_observed():
    from datetime import timedelta

    with pytest.raises(ValueError, match="cannot precede"):
        build_market_context_fingerprint(
            root_symbol="CL", event_type="MARKET_NEWS", event_category="PETROLEUM", affected_products=("CL",),
            observed_at=_OBSERVED_AT, evaluated_at=_OBSERVED_AT - timedelta(hours=1),
            trend="Up", volatility="Moderate", curve_state=None,
            related_market_confirming=0, related_market_total=0,
        )


def test_fingerprint_component_detail_is_fully_transparent():
    fp = _fingerprint()
    detail = dict(kv.split("=", 1) for kv in fp.component_detail)
    assert detail["trend"] == "Up"
    assert detail["curve_state"] == "BACKWARDATION"
    assert detail["freshness_bucket"] == "FRESH"


# ---------------------------------------------------------------------------
# Semantic hardening patch, section 3: EVENT IMPORTANCE != OBJECTIVE EVENT
# MAGNITUDE.
# ---------------------------------------------------------------------------


def test_event_importance_and_objective_magnitude_are_separate_fields():
    fields = set(MarketContextFingerprint.model_fields)
    assert {"event_importance", "event_importance_rule", "objective_event_magnitude",
            "objective_event_magnitude_detail"} <= fields
    # The old, conflated field names must be gone entirely.
    assert "event_magnitude" not in fields
    assert "event_magnitude_rule" not in fields


def test_objective_event_magnitude_is_unknown_by_default():
    """PETROLEUM is a HIGH-importance CATEGORY, but that is not itself a
    measured magnitude -- objective_event_magnitude must stay UNKNOWN unless
    a real typed structured value is supplied."""
    fp = _fingerprint(event_category="PETROLEUM")
    assert fp.event_importance == "HIGH"
    assert fp.objective_event_magnitude == UNKNOWN_OBJECTIVE_MAGNITUDE
    assert fp.objective_event_magnitude != fp.event_importance
    assert "never derived from event_importance" in fp.objective_event_magnitude_detail


def test_objective_event_magnitude_uses_a_real_typed_structured_value_when_present():
    fp = _fingerprint(structured_attributes={"inventory_change_surprise": "-4.2M bbl vs -1.8M bbl consensus"})
    assert fp.objective_event_magnitude == "-4.2M bbl vs -1.8M bbl consensus"
    assert fp.objective_event_magnitude != UNKNOWN_OBJECTIVE_MAGNITUDE


def test_objective_event_magnitude_never_parses_a_free_text_headline():
    """A headline-shaped key that is NOT one of the closed, real, typed
    magnitude keys must never be treated as a magnitude."""
    fp = _fingerprint(structured_attributes={"headline": "EIA reports a larger-than-expected crude draw"})
    assert fp.objective_event_magnitude == UNKNOWN_OBJECTIVE_MAGNITUDE


# ---------------------------------------------------------------------------
# Context Match -- mechanism_signals.context_match_for
# ---------------------------------------------------------------------------


def test_context_match_strong_when_every_declared_signal_is_informative():
    fp = _fingerprint(trend="Up", volatility="Moderate")
    level, reasons = context_match_for(EconomicMechanism.TREND, fp)
    assert level == ContextMatchLevel.STRONG
    assert any("trend=Up" in r for r in reasons)


def test_context_match_none_when_no_declared_signal_is_informative():
    from datetime import timedelta

    fp = _fingerprint(trend="Sideways", evaluated_at=_OBSERVED_AT + timedelta(days=60))
    level, _ = context_match_for(EconomicMechanism.TREND, fp)
    assert level == ContextMatchLevel.NONE


def test_context_match_moderate_when_half_of_two_signals_informative():
    from datetime import timedelta

    fp = _fingerprint(trend="Up", evaluated_at=_OBSERVED_AT + timedelta(days=60))
    level, _ = context_match_for(EconomicMechanism.TREND, fp)
    assert level == ContextMatchLevel.MODERATE


def test_context_match_for_term_structure_depends_only_on_curve_state():
    fp_with_curve = _fingerprint(curve_state="CONTANGO", trend="Sideways")
    fp_without_curve = _fingerprint(curve_state=None, trend="Up")
    strong, _ = context_match_for(EconomicMechanism.TERM_STRUCTURE, fp_with_curve)
    none, _ = context_match_for(EconomicMechanism.TERM_STRUCTURE, fp_without_curve)
    assert strong == ContextMatchLevel.STRONG
    assert none == ContextMatchLevel.NONE


def test_context_match_is_none_and_honest_for_an_undeclared_mechanism():
    fp = _fingerprint()
    level, reasons = context_match_for(EconomicMechanism.CARRY, fp)
    assert level == ContextMatchLevel.NONE
    assert "No declared context signal" in reasons[0]


# ---------------------------------------------------------------------------
# Semantic hardening patch, section 1: Scientific Reliability must preserve
# committed verdict meaning.
# ---------------------------------------------------------------------------


def test_inconclusive_never_becomes_rejected():
    result = _scientific_reliability({"tsmom": "INCONCLUSIVE"})
    assert result == ScientificReliabilityLevel.INCONCLUSIVE
    assert result != ScientificReliabilityLevel.REJECTED


def test_pass_and_reject_across_families_becomes_mixed_not_passed():
    """The exact bug this patch fixes: the old reduction picked the most
    favorable verdict (PASS) when families disagreed. It must return MIXED
    instead, never silently choosing a side."""
    result = _scientific_reliability({"tsmom": "PASS", "ma_trend": "REJECT"})
    assert result == ScientificReliabilityLevel.MIXED
    assert result != ScientificReliabilityLevel.PASSED


def test_pass_and_inconclusive_across_families_becomes_mixed():
    result = _scientific_reliability({"tsmom": "PASS", "ma_trend": "INCONCLUSIVE"})
    assert result == ScientificReliabilityLevel.MIXED


def test_a_familys_own_internal_mixed_string_forces_overall_mixed():
    result = _scientific_reliability({"tsmom": "MIXED(PASS,REJECT)"})
    assert result == ScientificReliabilityLevel.MIXED


def test_agreeing_verdicts_across_families_are_not_forced_to_mixed():
    """Real agreement (both families REJECT) must stay REJECTED, never
    inflated to MIXED just because more than one family was mapped."""
    result = _scientific_reliability({"tsmom": "REJECT", "ma_trend": "REJECT"})
    assert result == ScientificReliabilityLevel.REJECTED


def test_no_real_verdict_present_is_not_adjudicated_not_rejected():
    assert _scientific_reliability({"tsmom": "NOT_ADJUDICATED"}) == ScientificReliabilityLevel.NOT_ADJUDICATED
    assert _scientific_reliability({"tsmom": "NO_CANONICAL_TRIAL"}) == ScientificReliabilityLevel.NOT_ADJUDICATED


def test_empty_evidence_is_no_evidence():
    assert _scientific_reliability({}) == ScientificReliabilityLevel.NO_EVIDENCE


def test_scientific_reliability_ranking_order_preserves_meaning():
    """INCONCLUDE/REJECTED/MIXED/PASSED must be strictly distinguishable in
    the ranking ordinal too -- never collapsed pairwise."""
    a = _candidate(EconomicMechanism.TREND, scientific_reliability=ScientificReliabilityLevel.INCONCLUSIVE)
    b = _candidate(EconomicMechanism.BREAKOUT, scientific_reliability=ScientificReliabilityLevel.REJECTED)
    ranked = rank_candidates((a, b))
    assert ranked[0].mechanism == EconomicMechanism.BREAKOUT
    assert ranked[0].rank_key_components[1] != ranked[1].rank_key_components[1]


# ---------------------------------------------------------------------------
# Semantic hardening patch, section 2: related_experiment_count is not
# "replication".
# ---------------------------------------------------------------------------


def test_independent_replication_is_always_not_available_in_v1():
    assert _independent_replication() == IndependentReplicationLevel.NOT_AVAILABLE


def test_rank_dimension_scores_has_no_replication_field_name():
    """The old, misleading field names must be gone entirely -- only the
    honestly-named `independent_replication`/`related_experiment_counts`
    remain."""
    fields = set(RankDimensionScores.model_fields)
    assert "replication" not in fields
    assert "replication_detail" not in fields
    assert {"independent_replication", "related_experiment_counts"} <= fields


def test_independent_replication_ordinal_never_discriminates_on_experiment_count():
    """A candidate with many related experiments must rank identically to
    one with zero on the Independent Replication dimension -- it is always
    NOT_AVAILABLE, never inferred from neighbours/ablations/re-executions."""
    many = _candidate(EconomicMechanism.TREND, related_experiment_counts={"tsmom": 6, "ma_trend": 5})
    none = _candidate(EconomicMechanism.BREAKOUT, related_experiment_counts={})
    ranked_many = rank_candidates((many,))
    ranked_none = rank_candidates((none,))
    # Index 4 is Independent Replication in the declared priority order.
    assert ranked_many[0].rank_key_components[4] == ranked_none[0].rank_key_components[4]


# ---------------------------------------------------------------------------
# ranking.py -- pure, synthetic candidates (no registry, no network)
# ---------------------------------------------------------------------------


def _candidate(mechanism: EconomicMechanism, **score_overrides) -> RankedResearchCandidate:
    scores_base = {
        "context_match": ContextMatchLevel.STRONG, "context_match_reasons": (),
        "scientific_reliability": ScientificReliabilityLevel.NO_EVIDENCE, "scientific_reliability_detail": {},
        "execution_robustness_detail": {},
        "independent_replication": IndependentReplicationLevel.NOT_AVAILABLE, "related_experiment_counts": {},
        "user_fit_detail": {}, "research_coverage": ResearchCoverageLevel.NOT_MAPPED, "research_coverage_detail": "",
    }
    scores_base.update(score_overrides)
    return RankedResearchCandidate(
        mechanism=mechanism,
        mechanism_candidate=MechanismCandidate(
            mechanism=mechanism, explanation="x", causal_chain=(), evidence_basis="x",
            provenance=MechanismProvenance.DETERMINISTIC_LIBRARY,
        ),
        scores=RankDimensionScores(**scores_base),
        rank_key_components=(), retrieval_rank=0,
    )


def test_rank_candidates_orders_by_context_match_first():
    weak = _candidate(EconomicMechanism.MEAN_REVERSION, context_match=ContextMatchLevel.WEAK)
    strong = _candidate(EconomicMechanism.TREND, context_match=ContextMatchLevel.STRONG)
    ranked = rank_candidates((weak, strong))
    assert [c.mechanism for c in ranked] == [EconomicMechanism.TREND, EconomicMechanism.MEAN_REVERSION]
    assert [c.retrieval_rank for c in ranked] == [1, 2]


def test_rank_candidates_falls_through_to_scientific_reliability_on_context_tie():
    weak_evidence = _candidate(
        EconomicMechanism.MEAN_REVERSION, context_match=ContextMatchLevel.STRONG,
        scientific_reliability=ScientificReliabilityLevel.NO_EVIDENCE,
    )
    passed = _candidate(
        EconomicMechanism.TREND, context_match=ContextMatchLevel.STRONG,
        scientific_reliability=ScientificReliabilityLevel.PASSED,
    )
    ranked = rank_candidates((weak_evidence, passed))
    assert ranked[0].mechanism == EconomicMechanism.TREND


def test_rank_candidates_is_deterministic_regardless_of_input_order():
    a = _candidate(EconomicMechanism.TREND, context_match=ContextMatchLevel.STRONG)
    b = _candidate(EconomicMechanism.BREAKOUT, context_match=ContextMatchLevel.WEAK)
    c = _candidate(EconomicMechanism.MEAN_REVERSION, context_match=ContextMatchLevel.MODERATE)
    order1 = [x.mechanism for x in rank_candidates((a, b, c))]
    order2 = [x.mechanism for x in rank_candidates((c, a, b))]
    assert order1 == order2


def test_rank_candidates_breaks_ties_by_mechanism_name():
    a = _candidate(EconomicMechanism.TREND)
    b = _candidate(EconomicMechanism.BREAKOUT)
    ranked = rank_candidates((a, b))
    assert [c.mechanism for c in ranked] == [EconomicMechanism.BREAKOUT, EconomicMechanism.TREND]


def test_user_fit_not_personalized_is_never_treated_as_poor_fit():
    """A NOT_PERSONALIZED user-fit entry must rank the same as no entry at
    all -- alpha_agent.recommendation.fit's own discipline: its label is
    "LOW" only because nothing was scored, never a real measurement of poor
    fit. Regression test for the exact bug that discipline was written to
    prevent (fit.py module docstring)."""
    no_entry = _candidate(EconomicMechanism.TREND, user_fit_detail={})
    not_personalized_low = _candidate(EconomicMechanism.BREAKOUT, user_fit_detail={"tsmom": "NOT_PERSONALIZED:LOW"})
    ranked_a = rank_candidates((no_entry,))
    ranked_b = rank_candidates((not_personalized_low,))
    assert ranked_a[0].rank_key_components[-1] == ranked_b[0].rank_key_components[-1]


def test_personalized_low_user_fit_remains_low_semantic_hardening_patch_section_4():
    """The bug this patch fixes: the old reduction started its running-best
    at MEDIUM, so a genuinely personalized LOW result could never surface as
    LOW. A real MEASURED:LOW entry must rank strictly below the neutral
    (no-evidence) baseline, and strictly below a real MEASURED:HIGH entry."""
    neutral = _candidate(EconomicMechanism.MEAN_REVERSION, user_fit_detail={})
    low = _candidate(EconomicMechanism.TREND, user_fit_detail={"tsmom": "MEASURED:LOW"})
    high = _candidate(EconomicMechanism.BREAKOUT, user_fit_detail={"tsmom": "MEASURED:HIGH"})

    ranked_neutral = rank_candidates((neutral,))
    ranked_low = rank_candidates((low,))
    ranked_high = rank_candidates((high,))

    low_ordinal = ranked_low[0].rank_key_components[-1]
    neutral_ordinal = ranked_neutral[0].rank_key_components[-1]
    high_ordinal = ranked_high[0].rank_key_components[-1]

    assert low_ordinal < neutral_ordinal, "a real MEASURED:LOW must rank below the neutral no-evidence baseline"
    assert low_ordinal < high_ordinal
    # And the full ranking, head to head, must actually put HIGH ahead of LOW.
    ranked = rank_candidates((low, high))
    assert ranked[0].mechanism == EconomicMechanism.BREAKOUT


def test_explain_rank_names_the_first_differing_dimension():
    a = _candidate(EconomicMechanism.TREND, context_match=ContextMatchLevel.STRONG)
    b = _candidate(EconomicMechanism.BREAKOUT, context_match=ContextMatchLevel.WEAK)
    ranked = rank_candidates((a, b))
    text = explain_rank(ranked)
    assert "TREND ranks #1 over BREAKOUT" in text
    assert "Context Match" in text


def test_explain_rank_handles_zero_and_one_candidate():
    assert "No relevant mechanism candidates" in explain_rank(())
    only = rank_candidates((_candidate(EconomicMechanism.TREND),))
    assert "Only one relevant mechanism candidate" in explain_rank(only)


def test_explain_rank_reports_a_tie_honestly():
    a = _candidate(EconomicMechanism.BREAKOUT)
    b = _candidate(EconomicMechanism.TREND)
    ranked = rank_candidates((a, b))
    text = explain_rank(ranked)
    assert "tied on every ranked dimension" in text


def test_suggest_next_experiment_prefers_underexplored_over_well_evidenced():
    """The main rank can legitimately put a PASSED/well-evidenced mechanism
    first, but "Suggested Next Experiment" must prefer the underexplored one
    -- highest information value, never predicted profit (prompt section 6)."""
    well_evidenced = _candidate(
        EconomicMechanism.TREND, context_match=ContextMatchLevel.STRONG,
        scientific_reliability=ScientificReliabilityLevel.PASSED, research_coverage=ResearchCoverageLevel.FULLY_TESTED,
    )
    underexplored = _candidate(
        EconomicMechanism.TERM_STRUCTURE, context_match=ContextMatchLevel.STRONG,
        scientific_reliability=ScientificReliabilityLevel.NO_EVIDENCE, research_coverage=ResearchCoverageLevel.NOT_MAPPED,
    )
    ranked = rank_candidates((well_evidenced, underexplored))
    assert ranked[0].mechanism == EconomicMechanism.TREND  # main rank: best evidence first
    suggestion = suggest_next_experiment(ranked)
    assert suggestion.mechanism == EconomicMechanism.TERM_STRUCTURE  # next experiment: highest info value
    assert "already-adjudicated" in suggestion.information_value_reason


def test_suggest_next_experiment_falls_back_honestly_when_nothing_is_underexplored():
    only = rank_candidates((
        _candidate(EconomicMechanism.TREND, research_coverage=ResearchCoverageLevel.FULLY_TESTED),
    ))
    suggestion = suggest_next_experiment(only)
    assert suggestion is not None
    assert "No clearly underexplored" in suggestion.information_value_reason


def test_suggest_next_experiment_none_for_empty_candidates():
    assert suggest_next_experiment(()) is None


# ---------------------------------------------------------------------------
# Semantic hardening patch, section 5: RELATED PRIOR EVENT OCCURRENCE, never
# a claim of matched historical context.
# ---------------------------------------------------------------------------


def test_related_prior_event_occurrence_schema_is_honestly_named():
    """The old, overclaiming class name must be gone; the new one's own
    docstring must explicitly DISCLAIM a historical-context match (never
    silently assert one)."""
    import alpha_agent.context_retrieval.schemas as schemas_module

    assert not hasattr(schemas_module, "HistoricalContextOccurrence")
    assert hasattr(schemas_module, "RelatedPriorEventOccurrence")
    doc = (RelatedPriorEventOccurrence.__doc__ or "").lower()
    assert 'not a claim of "same historical context"' in doc
    assert "does not reconstruct" in doc


def test_context_retrieval_result_field_is_related_prior_event_occurrences():
    fields = set(ContextRetrievalResult.model_fields)
    assert "historical_occurrences" not in fields
    assert "related_prior_event_occurrences" in fields


# ---------------------------------------------------------------------------
# No forbidden field anywhere -- never expected return / probability /
# confidence / trade score, mirroring alpha_agent.opportunity.schemas'
# FORBIDDEN_FIELD_NAMES regression guard (prompt section 5).
# ---------------------------------------------------------------------------

_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "confidence", "confidence_score", "probability_of_profit", "win_probability",
        "expected_return", "expected_pnl", "priority_score", "signal", "recommendation",
        "buy_sell", "action_signal",
    }
)


@pytest.mark.parametrize(
    "model", [MarketContextFingerprint, RankDimensionScores, RankedResearchCandidate, ContextRetrievalResult],
)
def test_schemas_never_carry_a_forbidden_field(model):
    assert not set(model.model_fields) & _FORBIDDEN_FIELD_NAMES


# ---------------------------------------------------------------------------
# Real-registry integration -- CL / PETROLEUM, a genuinely recurring real
# futures context (weekly EIA petroleum releases already cached in this
# sandbox's local news store).
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)


def _registry() -> ExperimentRegistry:
    return ExperimentRegistry(services.REGISTRY_PATH)


def _cl_petroleum_observation() -> Observation:
    return Observation(
        event_type="MARKET_NEWS", root_symbol="CL", affected_products=("CL", "MCL", "RB", "HO"),
        observed_at=_OBSERVED_AT, source="EIA", summary="EIA reports a larger-than-expected crude draw",
        structured_attributes={
            "news_id": "N1", "category": "PETROLEUM", "mapping_reason": "x",
            "headline": "EIA reports a larger-than-expected crude draw",
        },
        evidence_refs=(), origin_vintage="2026-09-22", holdout_eligible=False,
    )


def test_real_recurring_context_retrieves_both_mapped_mechanisms():
    """The acceptance example: CL / PETROLEUM (EIA weekly release) maps to
    TREND and TERM_STRUCTURE in the deterministic mechanism library."""
    obs = _cl_petroleum_observation()
    ctx = _fingerprint()
    with _registry() as reg:
        result = build_context_retrieval(obs, context=ctx, registry=reg)
    mechanisms = {c.mechanism for c in result.ranked_candidates}
    assert mechanisms == {EconomicMechanism.TREND, EconomicMechanism.TERM_STRUCTURE}


def test_real_retrieval_is_reproducible():
    obs = _cl_petroleum_observation()
    ctx = _fingerprint()
    with _registry() as reg:
        first = build_context_retrieval(obs, context=ctx, registry=reg)
    with _registry() as reg:
        second = build_context_retrieval(obs, context=ctx, registry=reg)
    assert [c.mechanism for c in first.ranked_candidates] == [c.mechanism for c in second.ranked_candidates]
    assert [c.retrieval_rank for c in first.ranked_candidates] == [c.retrieval_rank for c in second.ranked_candidates]


def test_real_retrieval_ranks_are_a_valid_permutation():
    obs = _cl_petroleum_observation()
    ctx = _fingerprint()
    with _registry() as reg:
        result = build_context_retrieval(obs, context=ctx, registry=reg)
    ranks = sorted(c.retrieval_rank for c in result.ranked_candidates)
    assert ranks == list(range(1, len(result.ranked_candidates) + 1))


def test_real_retrieval_never_fabricates_an_alpha_id():
    from alpha_agent.alpha_memory import get_alpha_research_object

    obs = _cl_petroleum_observation()
    ctx = _fingerprint()
    with _registry() as reg:
        result = build_context_retrieval(obs, context=ctx, registry=reg)
        for candidate in result.ranked_candidates:
            for alpha_id in candidate.alpha_ids:
                assert get_alpha_research_object(reg, alpha_id) is not None


def test_real_retrieval_scientific_reliability_only_uses_real_verdict_vocabulary():
    obs = _cl_petroleum_observation()
    ctx = _fingerprint()
    with _registry() as reg:
        result = build_context_retrieval(obs, context=ctx, registry=reg)
    allowed_prefixes = ("PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED", "NO_CANONICAL_TRIAL", "MIXED(")
    for candidate in result.ranked_candidates:
        for verdict in candidate.scores.scientific_reliability_detail.values():
            assert verdict.startswith(allowed_prefixes), verdict


def test_real_retrieval_independent_replication_always_not_available():
    obs = _cl_petroleum_observation()
    ctx = _fingerprint()
    with _registry() as reg:
        result = build_context_retrieval(obs, context=ctx, registry=reg)
    for candidate in result.ranked_candidates:
        assert candidate.scores.independent_replication == IndependentReplicationLevel.NOT_AVAILABLE


def test_real_retrieval_produces_an_explanation_and_a_next_experiment():
    obs = _cl_petroleum_observation()
    ctx = _fingerprint()
    with _registry() as reg:
        result = build_context_retrieval(obs, context=ctx, registry=reg)
    assert len(result.ranked_candidates) >= 2
    assert result.rank_explanation
    assert " vs " in result.rank_explanation or "tied" in result.rank_explanation
    assert result.suggested_next_experiment is not None


def test_real_related_prior_event_occurrences_are_real_and_match_category():
    """Uses the SAME cached-news read `context_retrieval_context` uses --
    proves at least the deterministic filtering logic behaves on real
    already-cached rows when the local news store has them (never asserts
    a live network fetch)."""
    from alpha_agent.ui import context_retrieval_context

    obs = _cl_petroleum_observation()
    occurrences = context_retrieval_context.related_prior_event_occurrences_for(obs)
    for occ in occurrences:
        assert occ.reference_id != "N1"
        assert "category=PETROLEUM" in occ.match_reason
