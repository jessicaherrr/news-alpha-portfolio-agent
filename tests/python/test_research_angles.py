"""Tests for `alpha_agent.discovery.research_angles` (Release UX Part C,
task spec sections 8-11/53).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from alpha_agent.discovery.research_angles import (
    DEFAULT_ANGLE_ORDER,
    ResearchAngle,
    augment_objective_for_angle,
    build_research_angle_plan,
    mechanism_diversity_report,
)
from alpha_agent.recommendation.profile import (
    DEFAULT_PROFILE,
    InvestorProfile,
    OvernightPreference,
)


def test_plan_is_built_before_generation_and_always_leads_with_baseline():
    plan = build_research_angle_plan(market="NQ")
    assert plan.assignments[0].angle == ResearchAngle.BASELINE_REPLICATION
    assert len(plan.assignments) <= 6


def test_plan_includes_failure_guided_only_when_prior_failures_exist():
    with_failures = build_research_angle_plan(market="NQ", has_prior_failures=True, failure_lesson="weak in high vol")
    without = build_research_angle_plan(market="NQ", has_prior_failures=False)
    assert ResearchAngle.FAILURE_GUIDED in with_failures.angles
    assert ResearchAngle.FAILURE_GUIDED not in without.angles
    fg = next(a for a in with_failures.assignments if a.angle == ResearchAngle.FAILURE_GUIDED)
    assert fg.addresses_failure == "weak in high vol"
    assert fg.falsification_hint is not None


def test_plan_includes_source_inspired_only_when_source_material_exists():
    with_source = build_research_angle_plan(market="NQ", has_source_material=True)
    without = build_research_angle_plan(market="NQ", has_source_material=False)
    assert ResearchAngle.SOURCE_INSPIRED in with_source.angles
    assert ResearchAngle.SOURCE_INSPIRED not in without.angles


def test_default_profile_never_produces_user_constraint_driven():
    """The untouched DEFAULT_PROFILE states no actual preference -- it must
    never be treated as a real constraint (mirrors
    alpha_agent.recommendation.fit's NOT_PERSONALIZED discipline)."""
    plan = build_research_angle_plan(market="NQ", profile=DEFAULT_PROFILE)
    assert ResearchAngle.USER_CONSTRAINT_DRIVEN not in plan.angles


def test_a_real_stated_constraint_produces_user_constraint_driven():
    profile = InvestorProfile(overnight=OvernightPreference.AVOID)
    plan = build_research_angle_plan(market="NQ", profile=profile)
    assignment = next(a for a in plan.assignments if a.angle == ResearchAngle.USER_CONSTRAINT_DRIVEN)
    assert "overnight" in assignment.rationale.lower()


def test_max_angles_bounds_the_plan_never_uncontrolled():
    plan = build_research_angle_plan(market="NQ", has_prior_failures=True, has_source_material=True, max_angles=3)
    assert len(plan.assignments) == 3


def test_max_angles_below_one_is_rejected():
    with pytest.raises(ValueError):
        build_research_angle_plan(market="NQ", max_angles=0)


def test_plan_never_duplicates_an_angle():
    plan = build_research_angle_plan(
        market="NQ", has_prior_failures=True, has_source_material=True,
        profile=InvestorProfile(overnight=OvernightPreference.AVOID), max_angles=10,
    )
    assert len(plan.angles) == len(set(plan.angles))


def test_default_order_is_fixed_and_covers_every_angle_exactly_once():
    assert len(DEFAULT_ANGLE_ORDER) == len(ResearchAngle)
    assert set(DEFAULT_ANGLE_ORDER) == set(ResearchAngle)


def test_augment_objective_never_mutates_the_orchestrator_only_builds_text():
    plan = build_research_angle_plan(market="NQ", has_prior_failures=True, failure_lesson="poor in chop")
    fg = next(a for a in plan.assignments if a.angle == ResearchAngle.FAILURE_GUIDED)
    augmented = augment_objective_for_angle("Find robust alpha opportunities.", fg)
    assert isinstance(augmented, str)
    assert "FAILURE_GUIDED" in augmented
    assert "poor in chop" in augmented


# ---------------------------------------------------------------------------
# mechanism_diversity_report -- duck-typed fake CandidatePoolResult
# ---------------------------------------------------------------------------


def _fake_member(kinds: frozenset[str]):
    return SimpleNamespace(strategy_spec_json={"features": [{"spec": {"kind": k}} for k in kinds]})


def _fake_pool(*, mechanisms_attempted, per_mechanism_members, members):
    per_mechanism = [SimpleNamespace(mechanism=SimpleNamespace(value=name), members=m) for name, m in per_mechanism_members]
    return SimpleNamespace(
        mechanisms_attempted=mechanisms_attempted,
        mechanisms_supported=sum(1 for _, m in per_mechanism_members if m),
        per_mechanism=per_mechanism,
        members=members,
    )


def test_diversity_report_flags_trivial_variation_when_only_one_feature_shape():
    members = [_fake_member(frozenset({"sma"})), _fake_member(frozenset({"sma"})), _fake_member(frozenset({"sma"}))]
    pool = _fake_pool(mechanisms_attempted=3, per_mechanism_members=[("TREND", members)], members=members)
    report = mechanism_diversity_report(pool)
    assert report.trivial_parameter_variation_only is True


def test_diversity_report_does_not_flag_real_feature_diversity():
    members = [_fake_member(frozenset({"sma"})), _fake_member(frozenset({"atr", "realized_vol"}))]
    pool = _fake_pool(
        mechanisms_attempted=2,
        per_mechanism_members=[("TREND", members[:1]), ("VOLATILITY_BREAKOUT", members[1:])],
        members=members,
    )
    report = mechanism_diversity_report(pool)
    assert report.trivial_parameter_variation_only is False
    assert set(report.distinct_mechanisms_with_members) == {"TREND", "VOLATILITY_BREAKOUT"}


def test_diversity_report_mechanism_coverage_ratio():
    members = [_fake_member(frozenset({"sma"}))]
    pool = _fake_pool(
        mechanisms_attempted=4, per_mechanism_members=[("TREND", members), ("CARRY", [])], members=members,
    )
    report = mechanism_diversity_report(pool)
    assert report.mechanism_coverage_ratio == pytest.approx(1 / 4)
