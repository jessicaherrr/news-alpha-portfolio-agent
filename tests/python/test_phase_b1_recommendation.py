"""Phase B1 -- user profile, Research Promise, User Fit, candidate ranking.

`alpha_agent.recommendation` is plain, dependency-light Python (no
`streamlit`, no registry import at module scope) -- every test below runs
unconditionally against synthetic rows shaped like
`alpha_agent.ui.services.list_experiments()` output. A small number of tests
also exercise the real committed registry (skipped if it is not present in
this checkout, matching the convention in `test_phase_20_streamlit_ui.py`).
"""
from __future__ import annotations

import pytest
from alpha_agent.recommendation import (
    DEFAULT_PROFILE,
    HoldingPeriod,
    InvestorProfile,
    MaxDrawdown,
    OvernightPreference,
    PersonalizationState,
    ProfileStore,
    RiskStyle,
    StrategyPreference,
    TradingFrequency,
    build_candidate_summaries,
    build_candidate_summary,
    promising_candidates,
    score_research_promise,
    score_user_fit,
    validated_strategies,
)
from alpha_agent.recommendation.candidates import ResearchCandidateSummary
from alpha_agent.recommendation.promise import (
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    _threshold_relative_score,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# fixtures: synthetic ResultRecord-shaped dicts (the exact keys
# `alpha_agent.ui.services.list_experiments` exposes)
# ---------------------------------------------------------------------------


def _row(**overrides) -> dict:
    base = {
        "experiment_id": "TEST__EXPERIMENT",
        "experiment_identity": "experiment1:deadbeef",
        "root_symbol": "ES",
        "strategy_family": "mean_reversion",
        "verdict": "REJECT",
        "reason_codes": ["null_hypothesis_not_rejected", "fdr_qvalue_above_threshold"],
        "net_pnl_usd": 50_000.0,
        "annualized_sharpe": 1.0,
        "daily_sharpe": 0.063,
        "bh_q": 0.5,
        "dsr_probability": 0.3,
        "gating_null_p": 0.1,
        "n_trades": 30,
        "n_oos_days": 500,
        "fold_consistency": 0.75,
        "parameter_stability": {"fraction_positive_sharpe": 0.8, "n_evaluated": 5},
        "cost_stress": {"max_net_pnl_degradation": 0.01},
        "regime_evidence": {"status": "evaluated", "max_regime_pnl_share": 0.2},
        "cross_market_reference": {"status": "not_evaluated"},
        "evidence_completeness": "full",
    }
    base.update(overrides)
    return base


_EXCELLENT_ROW = _row(
    verdict="REJECT",
    annualized_sharpe=1.9,
    dsr_probability=0.94,
    bh_q=0.02,
    gating_null_p=0.01,
    fold_consistency=1.0,
    parameter_stability={"fraction_positive_sharpe": 1.0, "n_evaluated": 7},
    cost_stress={"max_net_pnl_degradation": 0.0},
    regime_evidence={"status": "evaluated", "max_regime_pnl_share": 0.05},
    n_trades=80,
    evidence_completeness="full",
)


def _pass_row(**overrides) -> dict:
    return _row(verdict="PASS", reason_codes=[], **overrides)


def _not_adjudicated_refusal_row(**overrides) -> dict:
    row = {
        "experiment_id": "TEST__REFUSED",
        "experiment_identity": "experiment1:refused",
        "root_symbol": "NQ",
        "strategy_family": "ml_meta_label.tsmom",
        "verdict": "NOT_ADJUDICATED",
        "reason_codes": ["PHASE_15_TRIAL_REFUSED", "INSUFFICIENT_TRAIN_EVENTS"],
        "net_pnl_usd": None,
        "annualized_sharpe": None,
        "bh_q": 1.0,
        "dsr_probability": None,
        "gating_null_p": None,
        "n_trades": None,
        "n_oos_days": None,
        "fold_consistency": None,
        "parameter_stability": {},
        "cost_stress": {},
        "regime_evidence": {"status": "descriptive_only"},
        "cross_market_reference": {"status": "descriptive_only"},
        "evidence_completeness": "partial:typed_refusal:INSUFFICIENT_TRAIN_EVENTS",
    }
    row.update(overrides)
    return row


# ===========================================================================
# USER PROFILE
# ===========================================================================


def test_profile_validates_allowed_values_only():
    InvestorProfile(risk_style=RiskStyle.AGGRESSIVE)  # a valid enum value works
    with pytest.raises(ValidationError):
        InvestorProfile.model_validate({"risk_style": "Not A Real Risk Style"})
    with pytest.raises(ValidationError):
        InvestorProfile.model_validate({"unexpected_field": "x"})  # extra="forbid"


def test_profile_defaults_are_deterministic():
    a = InvestorProfile()
    b = InvestorProfile()
    assert a == b == DEFAULT_PROFILE
    assert a.risk_style == RiskStyle.BALANCED
    assert a.holding_period == HoldingPeriod.FLEXIBLE
    assert a.max_drawdown == MaxDrawdown.PCT_15
    assert a.trading_frequency == TradingFrequency.NO_PREFERENCE
    assert a.overnight == OvernightPreference.NO_PREFERENCE
    assert a.strategy_preference == StrategyPreference.MIXED


def test_profile_is_frozen_immutable():
    p = InvestorProfile()
    with pytest.raises(ValidationError):
        p.risk_style = RiskStyle.AGGRESSIVE  # type: ignore[misc]


def test_profile_store_round_trips_through_local_file(tmp_path):
    store = ProfileStore(path=tmp_path / "profile.json")
    assert store.load() == DEFAULT_PROFILE  # missing file -> default, never an error

    custom = InvestorProfile(
        risk_style=RiskStyle.AGGRESSIVE, holding_period=HoldingPeriod.INTRADAY,
        strategy_preference=StrategyPreference.BREAKOUT,
    )
    store.save(custom)
    assert store.load() == custom


def test_profile_store_falls_back_to_default_on_corrupt_file(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = ProfileStore(path=path)
    assert store.load() == DEFAULT_PROFILE


def test_profile_never_changes_scientific_verdict():
    row = _row(verdict="REJECT")
    a = build_candidate_summary(row, profile=InvestorProfile(strategy_preference=StrategyPreference.TREND))
    b = build_candidate_summary(row, profile=InvestorProfile(strategy_preference=StrategyPreference.BREAKOUT))
    assert a.scientific_verdict == b.scientific_verdict == "REJECT"


def test_profile_never_changes_research_promise_score():
    row = _row()
    a = build_candidate_summary(row, profile=InvestorProfile(trading_frequency=TradingFrequency.LOW))
    b = build_candidate_summary(row, profile=InvestorProfile(trading_frequency=TradingFrequency.HIGH))
    assert a.research_promise_score == b.research_promise_score
    assert a.research_promise_label == b.research_promise_label


def test_score_research_promise_signature_takes_no_profile_argument():
    """`alpha_agent.recommendation.promise` never reads a user preference or
    touches `ReliabilityPolicy`'s mutable state -- it only reads the frozen
    DEFAULT policy's numeric thresholds at import time."""
    import inspect

    params = inspect.signature(score_research_promise).parameters
    assert "profile" not in params
    assert list(params) == ["result"]


# ===========================================================================
# RESEARCH PROMISE
# ===========================================================================


def test_research_promise_is_deterministic():
    row = _row()
    a = score_research_promise(row)
    b = score_research_promise(row)
    assert a == b


def test_research_promise_bounded_0_100():
    for row in (_row(), _EXCELLENT_ROW, _not_adjudicated_refusal_row(), {}):
        score = score_research_promise(row or None)
        assert 0.0 <= score.total <= 100.0
        assert score.label in ("HIGH", "MEDIUM", "LOW")


def test_research_promise_none_result_scores_zero():
    score = score_research_promise(None)
    assert score.total == 0.0
    assert score.label == "LOW"


def test_missing_evidence_cannot_create_an_artificially_high_score():
    empty = score_research_promise({})
    assert empty.total == 0.0
    partial = score_research_promise(_row(
        dsr_probability=None, bh_q=None, gating_null_p=None, fold_consistency=None,
        parameter_stability={}, cost_stress={}, regime_evidence={}, evidence_completeness="partial:x",
    ))
    full = score_research_promise(_row())
    assert partial.total < full.total


def test_missing_parameter_stability_is_not_the_same_as_passing_it():
    with_stability = score_research_promise(_row(parameter_stability={"fraction_positive_sharpe": 1.0}))
    without_stability = score_research_promise(_row(parameter_stability={}))
    assert without_stability.stability < with_stability.stability
    assert "parameter_stability" in without_stability.missing_inputs


def test_higher_sharpe_alone_does_not_override_severe_evidence_weakness():
    """A very high Sharpe with everything else missing must not out-score a
    fully-evidenced, more moderate candidate."""
    lucky_but_empty = score_research_promise({"annualized_sharpe": 5.0})
    solid_full_evidence = score_research_promise(_row(annualized_sharpe=0.8))
    assert lucky_but_empty.total < solid_full_evidence.total


def test_sample_size_limitation_is_reflected():
    few_trades = score_research_promise(_row(n_trades=10))
    many_trades = score_research_promise(_row(n_trades=60))
    assert few_trades.evidence_quality < many_trades.evidence_quality
    assert few_trades.total < many_trades.total


def test_reject_can_reach_high_promise():
    candidate = build_candidate_summary(_EXCELLENT_ROW)
    assert candidate.scientific_verdict == "REJECT"
    assert candidate.research_promise_score >= HIGH_THRESHOLD
    assert candidate.research_promise_label == "HIGH"


def test_pass_verdict_is_unaffected_by_promise_score():
    weak_pass = build_candidate_summary(_pass_row(annualized_sharpe=-0.5, n_trades=5, dsr_probability=0.01))
    assert weak_pass.scientific_verdict == "PASS"  # never relabeled regardless of a low promise score
    assert weak_pass.research_promise_label in ("LOW", "MEDIUM", "HIGH")


def test_not_adjudicated_partial_evidence_handled_conservatively():
    refused = score_research_promise(_not_adjudicated_refusal_row())
    assert refused.total == 0.0
    assert refused.label == "LOW"
    full_evidence = score_research_promise(_row())
    assert refused.total < full_evidence.total


def test_component_weights_sum_to_100():
    from alpha_agent.recommendation import promise as promise_mod

    total = (
        promise_mod.WEIGHT_PERFORMANCE
        + promise_mod.WEIGHT_STATISTICAL_EVIDENCE
        + promise_mod.WEIGHT_STABILITY
        + promise_mod.WEIGHT_ROBUSTNESS
        + promise_mod.WEIGHT_EVIDENCE_QUALITY
    )
    assert total == 100.0


def test_bucket_thresholds_are_ordered():
    assert 0 < MEDIUM_THRESHOLD < HIGH_THRESHOLD <= 100


# ---------------------------------------------------------------------------
# threshold-relative p/q transform (section 10 hardening -- was `1 - p`/`1 - q`)
# ---------------------------------------------------------------------------


def test_threshold_relative_transform_full_credit_at_or_better_than_threshold():
    assert _threshold_relative_score(0.05, 0.05) == 1.0
    assert _threshold_relative_score(0.01, 0.05) == 1.0  # better than threshold, capped at 1.0
    assert _threshold_relative_score(0.0, 0.05) == 1.0  # exact zero never divides by zero


def test_threshold_relative_transform_gives_zero_at_the_worst_case_sentinel():
    """`q = 1.0` (the conservative BH sentinel a typed refusal is recorded
    with) must earn exactly zero credit, not a small positive residual."""
    assert _threshold_relative_score(1.0, 0.10) == 0.0
    assert _threshold_relative_score(1.0, 0.05) == 0.0


def test_threshold_relative_transform_decays_steeply_past_threshold():
    """The exact bug the task spec calls out: `p = 0.50` against a `p <= 0.05`
    bar must NOT receive ~half credit (the naive `1 - p` transform's answer)."""
    at_threshold = _threshold_relative_score(0.05, 0.05)
    mediocre = _threshold_relative_score(0.50, 0.05)
    assert mediocre < at_threshold
    assert mediocre == pytest.approx(0.10)  # 0.05 / 0.50, nowhere near the naive 1-p answer of 0.50


def test_gating_p_score_full_credit_at_threshold_and_low_credit_when_poor():
    at_threshold = score_research_promise(_row(gating_null_p=0.05))
    poor = score_research_promise(_row(gating_null_p=0.50))
    assert poor.statistical_evidence < at_threshold.statistical_evidence


def test_bh_q_score_full_credit_at_threshold_and_low_credit_when_poor():
    at_threshold = score_research_promise(_row(bh_q=0.10))
    poor = score_research_promise(_row(bh_q=0.50))
    assert poor.statistical_evidence < at_threshold.statistical_evidence


def test_dsr_transform_is_monotonic_in_dsr():
    low = score_research_promise(_row(dsr_probability=0.10))
    mid = score_research_promise(_row(dsr_probability=0.50))
    high = score_research_promise(_row(dsr_probability=0.95))
    assert low.statistical_evidence < mid.statistical_evidence < high.statistical_evidence


# ---------------------------------------------------------------------------
# regime-score sign safety (section 11 hardening)
# ---------------------------------------------------------------------------


def test_regime_score_cannot_reward_a_net_losing_candidate():
    losing_diversified = _row(
        net_pnl_usd=-10_000.0, regime_evidence={"status": "evaluated", "max_regime_pnl_share": 0.1}
    )
    winning_diversified = _row(
        net_pnl_usd=10_000.0, regime_evidence={"status": "evaluated", "max_regime_pnl_share": 0.1}
    )
    losing = score_research_promise(losing_diversified)
    winning = score_research_promise(winning_diversified)
    assert losing.robustness < winning.robustness
    assert "regime_evidence" in losing.missing_inputs


def test_regime_score_gives_no_credit_at_zero_or_negative_total_pnl():
    for pnl in (-1.0, 0.0):
        row = _row(net_pnl_usd=pnl, regime_evidence={"status": "evaluated", "max_regime_pnl_share": 0.05})
        score = score_research_promise(row)
        assert "regime_evidence" in score.missing_inputs


def test_regime_score_still_penalizes_concentration_in_a_net_winner():
    concentrated = _row(net_pnl_usd=10_000.0, regime_evidence={"status": "evaluated", "max_regime_pnl_share": 0.95})
    diversified = _row(net_pnl_usd=10_000.0, regime_evidence={"status": "evaluated", "max_regime_pnl_share": 0.1})
    assert score_research_promise(concentrated).robustness < score_research_promise(diversified).robustness


# ===========================================================================
# USER FIT
# ===========================================================================


def test_user_fit_is_deterministic():
    a = score_user_fit(strategy_family="tsmom", result=_row(strategy_family="tsmom"), profile=DEFAULT_PROFILE)
    b = score_user_fit(strategy_family="tsmom", result=_row(strategy_family="tsmom"), profile=DEFAULT_PROFILE)
    assert a == b


def test_user_fit_bounded_0_100():
    for family in ("tsmom", "ma_trend", "breakout", "mean_reversion", "silver_bullet", None):
        fit = score_user_fit(strategy_family=family, result=_row(), profile=DEFAULT_PROFILE)
        assert 0.0 <= fit.total <= 100.0
        assert fit.label in ("HIGH", "MEDIUM", "LOW")


def test_changing_profile_changes_user_fit():
    row = _row(strategy_family="tsmom", n_trades=60, n_oos_days=252)  # ~60 trades/year -> High band
    trend_pref = score_user_fit(
        strategy_family="tsmom", result=row,
        profile=InvestorProfile(strategy_preference=StrategyPreference.TREND, trading_frequency=TradingFrequency.HIGH),
    )
    mismatched = score_user_fit(
        strategy_family="tsmom", result=row,
        profile=InvestorProfile(
            strategy_preference=StrategyPreference.MEAN_REVERSION, trading_frequency=TradingFrequency.LOW,
        ),
    )
    assert trend_pref.total > mismatched.total


def test_default_no_preference_profile_is_not_personalized_never_100_percent():
    """Section 6 fix: the untouched `DEFAULT_PROFILE` ("no preference" on
    every scoreable dimension) must NOT score `100.0` / `HIGH` -- there is
    nothing measured yet. This is the exact bug the task spec calls out."""
    fit = score_user_fit(
        strategy_family="breakout", result=_row(n_trades=5, n_oos_days=252),
        profile=InvestorProfile(strategy_preference=StrategyPreference.MIXED, trading_frequency=TradingFrequency.NO_PREFERENCE),
    )
    assert fit.personalization_state == PersonalizationState.NOT_PERSONALIZED
    assert fit.total == 0.0
    assert fit.measurable_dimension_count == 0
    assert fit.mechanism is None
    assert fit.trading_frequency is None


def test_stated_preference_with_full_evidence_is_measured():
    fit = score_user_fit(
        strategy_family="breakout", result=_row(strategy_family="breakout", n_trades=60, n_oos_days=252),
        profile=InvestorProfile(strategy_preference=StrategyPreference.BREAKOUT, trading_frequency=TradingFrequency.HIGH),
    )
    assert fit.personalization_state == PersonalizationState.MEASURED
    assert fit.measurable_dimension_count == 2
    assert fit.total == 100.0


def test_stated_preference_without_evidence_is_partial_not_measured():
    """The user asked to be scored on trading frequency, but this candidate
    carries no committed trade-count evidence -- that is PARTIAL, not a
    silently-excluded "no preference"."""
    fit = score_user_fit(
        strategy_family="breakout", result=_row(strategy_family="breakout", n_trades=None, n_oos_days=None),
        profile=InvestorProfile(strategy_preference=StrategyPreference.BREAKOUT, trading_frequency=TradingFrequency.HIGH),
    )
    assert fit.personalization_state == PersonalizationState.PARTIAL
    assert fit.measurable_dimension_count == 1
    assert "saved preference, evidence not available" in fit.evidence_coverage["trading_frequency"]


def test_unmapped_family_scores_mechanism_neutrally_not_zero_or_full():
    fit_pref = score_user_fit(
        strategy_family="silver_bullet", result=_row(n_trades=None, n_oos_days=None),
        profile=InvestorProfile(strategy_preference=StrategyPreference.TREND),
    )
    assert fit_pref.mechanism == pytest.approx(30.0)  # 0.5 * WEIGHT_MECHANISM(60)
    assert "no documented mechanism-category mapping" in fit_pref.evidence_coverage["strategy_mechanism"]


def test_missing_trade_evidence_is_explicit_not_silently_neutral():
    fit = score_user_fit(
        strategy_family="tsmom", result=_row(n_trades=None, n_oos_days=None),
        profile=InvestorProfile(trading_frequency=TradingFrequency.HIGH),
    )
    assert fit.trading_frequency is None
    assert "evidence not available" in fit.evidence_coverage["trading_frequency"]
    assert fit.personalization_state == PersonalizationState.PARTIAL  # mechanism stayed unstated -> excluded
    assert "unavailable" in fit.evidence_coverage["drawdown"]


def test_default_profile_no_preference_dimensions_are_explicitly_not_stated():
    fit = score_user_fit(strategy_family="tsmom", result=_row(n_trades=None, n_oos_days=None), profile=DEFAULT_PROFILE)
    assert fit.evidence_coverage["trading_frequency"] == "no preference set -- excluded from Fit score"
    assert fit.evidence_coverage["strategy_mechanism"] == "no preference set -- excluded from Fit score"
    for dim in ("overnight_positions", "holding_period"):
        assert fit.evidence_coverage[dim] == "no preference set"
    assert "unavailable" in fit.evidence_coverage["drawdown"]


def test_stated_overnight_and_holding_preferences_report_saved_not_measured():
    fit = score_user_fit(
        strategy_family="tsmom", result=_row(),
        profile=InvestorProfile(overnight=OvernightPreference.AVOID, holding_period=HoldingPeriod.INTRADAY),
    )
    assert "saved preference, evidence not available" in fit.evidence_coverage["overnight_positions"]
    assert "saved preference, evidence not available" in fit.evidence_coverage["holding_period"]


def test_changing_profile_cannot_change_promise_or_verdict_end_to_end():
    row = _row()
    profiles = [
        DEFAULT_PROFILE,
        InvestorProfile(strategy_preference=StrategyPreference.TREND, trading_frequency=TradingFrequency.HIGH),
        InvestorProfile(strategy_preference=StrategyPreference.RELATIVE_VALUE, overnight=OvernightPreference.AVOID),
    ]
    candidates = [build_candidate_summary(row, profile=p) for p in profiles]
    promise_scores = {c.research_promise_score for c in candidates}
    verdicts = {c.scientific_verdict for c in candidates}
    fit_scores = {c.user_fit_score for c in candidates}
    assert len(promise_scores) == 1
    assert len(verdicts) == 1
    assert len(fit_scores) >= 1  # allowed (and, here, expected) to vary


# ===========================================================================
# CANDIDATE RANKING
# ===========================================================================


def test_validated_list_contains_pass_only():
    rows = [_row(experiment_id="A", verdict="PASS"), _row(experiment_id="B", verdict="REJECT"),
            _row(experiment_id="C", verdict="INCONCLUSIVE"), _not_adjudicated_refusal_row(experiment_id="D")]
    candidates = build_candidate_summaries(rows)
    validated = validated_strategies(candidates)
    assert {c.experiment_id for c in validated} == {"A"}
    assert all(c.scientific_verdict == "PASS" for c in validated)


def test_promising_list_may_contain_reject_and_inconclusive_never_pass():
    rows = [_row(experiment_id="A", verdict="PASS"), _row(experiment_id="B", verdict="REJECT"),
            _row(experiment_id="C", verdict="INCONCLUSIVE")]
    candidates = build_candidate_summaries(rows)
    promising = promising_candidates(candidates)
    ids = {c.experiment_id for c in promising}
    assert ids == {"B", "C"}
    assert "A" not in ids


def test_promising_list_excludes_zero_evidence_not_adjudicated_refusals():
    rows = [_row(experiment_id="B", verdict="REJECT"), _not_adjudicated_refusal_row(experiment_id="D")]
    candidates = build_candidate_summaries(rows)
    promising = promising_candidates(candidates)
    assert {c.experiment_id for c in promising} == {"B"}


def test_promising_ranking_deterministic_tie_break():
    rows = [
        _row(experiment_id="Z", verdict="REJECT"),
        _row(experiment_id="A", verdict="REJECT"),
        _row(experiment_id="M", verdict="REJECT"),
    ]  # identical evidence -> identical promise/fit scores -> tie-break on experiment_id
    forward = promising_candidates(build_candidate_summaries(rows))
    backward = promising_candidates(build_candidate_summaries(list(reversed(rows))))
    assert [c.experiment_id for c in forward] == [c.experiment_id for c in backward] == ["A", "M", "Z"]


def test_personalized_fit_sort_never_relabels_scientific_status():
    rows = [_row(experiment_id="B", verdict="REJECT", strategy_family="tsmom"),
            _row(experiment_id="C", verdict="INCONCLUSIVE", strategy_family="mean_reversion")]
    profile = InvestorProfile(strategy_preference=StrategyPreference.MEAN_REVERSION)
    candidates = build_candidate_summaries(rows, profile=profile)
    by_promise = {c.experiment_id: c.scientific_verdict for c in promising_candidates(candidates, sort_by="promise")}
    by_fit = {c.experiment_id: c.scientific_verdict for c in promising_candidates(candidates, sort_by="fit")}
    assert by_promise == by_fit == {"B": "REJECT", "C": "INCONCLUSIVE"}


def test_candidate_summary_never_stores_generated_prose_as_a_verdict_field():
    candidate = build_candidate_summary(_row())
    assert isinstance(candidate, ResearchCandidateSummary)
    assert candidate.scientific_verdict in ("PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED")
    assert candidate.why_promising  # non-empty deterministic prose, separate field
    assert isinstance(candidate.validation_blockers, tuple)


# ===========================================================================
# real registry (skipped if the checkout has no committed sqlite)
# ===========================================================================

from alpha_agent.ui import services

pytestmark_real = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)


@pytestmark_real
def test_real_registry_has_zero_validated_strategies_today():
    """Matches the documented current research state (phase-status notes: 0
    PASS across the registry) -- an honest, non-fabricated empty state."""
    assert services.validated_strategy_summaries() == []


@pytestmark_real
def test_real_registry_promising_candidates_never_include_pass():
    promising = services.promising_candidate_summaries()
    assert all(c.scientific_verdict != "PASS" for c in promising)


@pytestmark_real
def test_real_registry_candidate_summaries_cover_every_canonical_trial():
    from alpha_agent.registry import TrialRole

    canonical_rows = services.list_experiments(trial_role=TrialRole.CANONICAL)
    candidates = services.research_candidate_summaries()
    assert len(candidates) == len(canonical_rows)


@pytestmark_real
def test_real_registry_promise_score_never_changes_with_profile():
    from alpha_agent.recommendation import InvestorProfile, StrategyPreference

    default_scores = {c.experiment_id: c.research_promise_score for c in services.research_candidate_summaries()}
    alt_profile = InvestorProfile(strategy_preference=StrategyPreference.RELATIVE_VALUE)
    alt_scores = {c.experiment_id: c.research_promise_score for c in services.research_candidate_summaries(alt_profile)}
    assert default_scores == alt_scores
