"""Phase 2 -- Phase 1 hand-off lookup (prompt 2 section 17), hardened by the
identity-hardening patch (section 5) and the final Phase 2 semantic fix
(section 2): mechanism alone can NEVER establish an exact Factor match.

Uses the REAL, already-committed local registry (read-only, never mutated),
same pattern as `test_alpha_memory_builder.py`.
"""
from __future__ import annotations

import pytest
from alpha_agent.alpha_memory import mechanism_memory_lookup
from alpha_agent.alpha_memory.schemas import MechanismMemoryMatchKind
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)


def _registry() -> ExperimentRegistry:
    return ExperimentRegistry(services.REGISTRY_PATH)


# ---------------------------------------------------------------------------
# Mechanism-only lookup never claims MATCHING_FACTOR
# ---------------------------------------------------------------------------


def test_mechanism_only_lookup_never_claims_matching_factor_with_one_real_object():
    """MOMENTUM maps to exactly one real Factor (tsmom) on NQ -- a single
    candidate is NOT proof of an exact match; mechanism-only lookup must
    still report RELATED_MECHANISM, never MATCHING_FACTOR, however few
    objects exist."""
    with _registry() as reg:
        result = mechanism_memory_lookup(reg, mechanism=EconomicMechanism.MOMENTUM, root_symbol="NQ")
    assert len(result.objects) == 1
    assert result.match_kind == MechanismMemoryMatchKind.RELATED_MECHANISM
    assert result.match_kind != MechanismMemoryMatchKind.MATCHING_FACTOR


def test_mechanism_only_lookup_never_claims_matching_factor_with_multiple_real_objects():
    """TREND maps to two real Factors (tsmom, ma_trend) on NQ -- also
    RELATED_MECHANISM, for the same reason (never inferred from
    cardinality)."""
    with _registry() as reg:
        result = mechanism_memory_lookup(reg, mechanism=EconomicMechanism.TREND, root_symbol="NQ")
    assert len(result.objects) == 2
    assert result.match_kind == MechanismMemoryMatchKind.RELATED_MECHANISM


def test_mechanism_only_lookup_is_none_with_zero_real_objects():
    with _registry() as reg:
        result = mechanism_memory_lookup(reg, mechanism=EconomicMechanism.CARRY, root_symbol="NQ")
    assert result.objects == ()
    assert result.match_kind == MechanismMemoryMatchKind.NONE


@pytest.mark.parametrize(
    "mechanism", [EconomicMechanism.MOMENTUM, EconomicMechanism.TREND, EconomicMechanism.MEAN_REVERSION, EconomicMechanism.BREAKOUT],
)
def test_mechanism_only_lookup_never_returns_matching_factor_for_any_real_mechanism(mechanism):
    """Sweep every real, evidence-grounded mechanism on NQ: a mechanism-only
    call (no strategy_family) must NEVER resolve to MATCHING_FACTOR."""
    with _registry() as reg:
        result = mechanism_memory_lookup(reg, mechanism=mechanism, root_symbol="NQ")
    assert result.match_kind != MechanismMemoryMatchKind.MATCHING_FACTOR


# ---------------------------------------------------------------------------
# Exact FactorIdentity equality can claim MATCHING_FACTOR when supplied
# ---------------------------------------------------------------------------


def test_exact_strategy_family_match_claims_matching_factor():
    """Supplying the real strategy_family lets the candidate's own
    deterministic FactorIdentity be computed and compared -- an exact match
    is a genuine MATCHING_FACTOR, holding just that one object."""
    with _registry() as reg:
        result = mechanism_memory_lookup(
            reg, mechanism=EconomicMechanism.MOMENTUM, root_symbol="NQ", strategy_family="tsmom",
        )
    assert result.match_kind == MechanismMemoryMatchKind.MATCHING_FACTOR
    assert len(result.objects) == 1
    assert result.objects[0].factor.related_strategy_families == ("tsmom",)


def test_exact_strategy_family_match_disambiguates_between_two_real_factors():
    """TREND has two real Factors (tsmom, ma_trend) -- supplying the exact
    strategy_family resolves the ambiguity mechanism-only lookup cannot."""
    with _registry() as reg:
        tsmom_result = mechanism_memory_lookup(
            reg, mechanism=EconomicMechanism.TREND, root_symbol="NQ", strategy_family="tsmom",
        )
        ma_trend_result = mechanism_memory_lookup(
            reg, mechanism=EconomicMechanism.TREND, root_symbol="NQ", strategy_family="ma_trend",
        )
    assert tsmom_result.match_kind == MechanismMemoryMatchKind.MATCHING_FACTOR
    assert tsmom_result.objects[0].factor.related_strategy_families == ("tsmom",)
    assert ma_trend_result.match_kind == MechanismMemoryMatchKind.MATCHING_FACTOR
    assert ma_trend_result.objects[0].factor.related_strategy_families == ("ma_trend",)
    assert tsmom_result.objects[0].alpha_id != ma_trend_result.objects[0].alpha_id


def test_strategy_family_with_no_real_match_falls_back_to_related_mechanism():
    """A candidate strategy_family that does not equal any stored Factor
    (breakout is real evidence, but never part of NQ's TREND Factors) must
    fall back to RELATED_MECHANISM, never a fabricated MATCHING_FACTOR and
    never NONE (real evidence still exists under the mechanism)."""
    with _registry() as reg:
        result = mechanism_memory_lookup(
            reg, mechanism=EconomicMechanism.TREND, root_symbol="NQ", strategy_family="breakout",
        )
    assert result.match_kind == MechanismMemoryMatchKind.RELATED_MECHANISM
    assert len(result.objects) == 2  # every real Factor under the mechanism, not a guess


def test_strategy_family_supplied_but_no_evidence_at_all_is_none():
    with _registry() as reg:
        result = mechanism_memory_lookup(
            reg, mechanism=EconomicMechanism.CARRY, root_symbol="NQ", strategy_family="tsmom",
        )
    assert result.match_kind == MechanismMemoryMatchKind.NONE
    assert result.objects == ()


# ---------------------------------------------------------------------------
# services.py wrapper parity
# ---------------------------------------------------------------------------


def test_services_wrapper_matches_the_package_function():
    mechanism_only = services.mechanism_memory_lookup(mechanism="MOMENTUM", root_symbol="NQ")
    assert mechanism_only["match_kind"] == "RELATED_MECHANISM"

    exact = services.mechanism_memory_lookup(mechanism="MOMENTUM", root_symbol="NQ", strategy_family="tsmom")
    assert exact["match_kind"] == "MATCHING_FACTOR"
    assert len(exact["objects"]) == 1
