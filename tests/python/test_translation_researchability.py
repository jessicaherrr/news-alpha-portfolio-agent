"""Phase 1 -- deterministic researchability classification tests (prompt 1
section 7). Uses a small, isolated `FeatureRegistry` (never the shared global
`alpha_agent.features.REGISTRY`) so these tests never depend on -- or could be
broken by -- future additions to the real feature catalog.
"""
from __future__ import annotations

from alpha_agent.features.enums import FeatureFamily
from alpha_agent.features.registry import FeatureRegistry, feature
from alpha_agent.translation.researchability import classify_factor
from alpha_agent.translation.schemas import ResearchabilityStatus


def _isolated_registry() -> FeatureRegistry:
    reg = FeatureRegistry()

    @feature("dummy_trend", family=FeatureFamily.TREND, registry=reg)
    def _dummy(ctx):  # pragma: no cover - never actually computed in these tests
        raise NotImplementedError

    return reg


def test_available_when_every_feature_kind_is_registered_and_nothing_else_needed():
    reg = _isolated_registry()
    status, reason, available, missing = classify_factor(
        proposed_feature_kinds=("dummy_trend",), required_external_data=(), registry=reg,
    )
    assert status is ResearchabilityStatus.AVAILABLE
    assert available == ("dummy_trend",)
    assert missing == ()
    assert "registered" in reason.lower()


def test_partially_available_when_some_registered_and_more_is_needed():
    reg = _isolated_registry()
    status, _reason, available, missing = classify_factor(
        proposed_feature_kinds=("dummy_trend", "unregistered_kind"), required_external_data=(), registry=reg,
    )
    assert status is ResearchabilityStatus.PARTIALLY_AVAILABLE
    assert available == ("dummy_trend",)
    assert "unregistered_kind" in missing


def test_partially_available_when_registered_plus_external_data_needed():
    reg = _isolated_registry()
    status, _, available, missing = classify_factor(
        proposed_feature_kinds=("dummy_trend",),
        required_external_data=("consensus expectation series",),
        registry=reg,
    )
    assert status is ResearchabilityStatus.PARTIALLY_AVAILABLE
    assert available == ("dummy_trend",)
    assert "consensus expectation series" in missing


def test_data_missing_when_only_external_data_is_needed():
    reg = _isolated_registry()
    status, reason, available, missing = classify_factor(
        proposed_feature_kinds=(), required_external_data=("consensus expectation series",), registry=reg,
    )
    assert status is ResearchabilityStatus.DATA_MISSING
    assert available == ()
    assert missing == ("consensus expectation series",)
    assert "not currently ingested" in reason


def test_data_missing_when_proposed_kinds_are_all_unregistered():
    reg = _isolated_registry()
    status, _, available, missing = classify_factor(
        proposed_feature_kinds=("nonexistent_kind",), required_external_data=(), registry=reg,
    )
    assert status is ResearchabilityStatus.DATA_MISSING
    assert available == ()
    assert missing == ("nonexistent_kind",)


def test_not_executable_when_structurally_inexpressible_even_with_data():
    reg = _isolated_registry()
    status, reason, _, _ = classify_factor(
        proposed_feature_kinds=(), required_external_data=(), structurally_expressible=False, registry=reg,
    )
    assert status is ResearchabilityStatus.NOT_EXECUTABLE
    assert "cross-instrument" in reason or "not expressible" in reason


def test_not_executable_when_nothing_measurable_was_named_at_all():
    reg = _isolated_registry()
    status, reason, available, missing = classify_factor(
        proposed_feature_kinds=(), required_external_data=(), registry=reg,
    )
    assert status is ResearchabilityStatus.NOT_EXECUTABLE
    assert available == ()
    assert missing == ()
    assert "no measurable representation" in reason


def test_never_fabricates_availability_for_an_unregistered_kind_even_alongside_a_registered_one():
    """Missing data is never converted to zero (prompt 1 section 7): a factor
    naming one real kind and one fake kind must never come back AVAILABLE."""
    reg = _isolated_registry()
    status, _, available, _ = classify_factor(
        proposed_feature_kinds=("dummy_trend", "totally_made_up"), required_external_data=(), registry=reg,
    )
    assert status is not ResearchabilityStatus.AVAILABLE
    assert "totally_made_up" not in available
