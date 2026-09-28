"""Phase 2 -- "what changed this time?" (prompt 2 section 15), hardened by
the identity-hardening patch (section 4: factor sameness must never be
inferred from mechanism sameness) and the final Phase 2 semantic fix
(section 1: equal data requirements do not prove equal Factor
representation).

Real registry-backed: uses NQ's real TREND-tsmom AlphaResearchObject (real
canonical params) so the parameter-difference comparison is checked against
genuine committed data, not a synthetic fixture.
"""
from __future__ import annotations

import pytest
from alpha_agent.alpha_memory import build_alpha_research_objects_for_mechanism, what_changed
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)


def _registry() -> ExperimentRegistry:
    return ExperimentRegistry(services.REGISTRY_PATH)


@pytest.fixture(scope="module")
def nq_trend_tsmom():
    """NQ's TREND mechanism resolves to TWO real, separate Factors today
    (tsmom, ma_trend) -- this fixture is specifically the tsmom one."""
    with _registry() as reg:
        objs = build_alpha_research_objects_for_mechanism(reg, root_symbol="NQ", mechanism=EconomicMechanism.TREND)
    matches = [o for o in objs if o.factor.related_strategy_families == ("tsmom",)]
    assert len(matches) == 1
    return matches[0]


def test_identical_root_mechanism_family_are_all_same(nq_trend_tsmom):
    wc = what_changed(
        nq_trend_tsmom, mechanism=EconomicMechanism.TREND, root_symbol="NQ", strategy_family="tsmom",
    )
    assert "root" in wc.same
    assert "mechanism" in wc.same
    assert "factor" in wc.same
    assert "strategy_family" in wc.same
    assert "root" not in wc.changed
    assert "mechanism" not in wc.changed
    assert "factor" not in wc.changed


def test_different_root_is_flagged_changed(nq_trend_tsmom):
    wc = what_changed(nq_trend_tsmom, mechanism=EconomicMechanism.TREND, root_symbol="CL", strategy_family="tsmom")
    assert "root" in wc.changed
    assert "root" not in wc.same


def test_different_mechanism_without_a_strategy_family_omits_factor_entirely(nq_trend_tsmom):
    """Identity-hardening patch, section 4: with no `strategy_family` given,
    the candidate's own Factor cannot be computed at all -- "factor" must be
    absent from BOTH lists, never inferred as "changed" just because the
    mechanism differs."""
    wc = what_changed(nq_trend_tsmom, mechanism=EconomicMechanism.MEAN_REVERSION, root_symbol="NQ")
    assert "mechanism" in wc.changed
    assert "mechanism" not in wc.same
    assert "factor" not in wc.changed
    assert "factor" not in wc.same


def test_same_mechanism_does_not_automatically_imply_same_factor_different_family(nq_trend_tsmom):
    """Same mechanism, genuinely different data requirements: `breakout`
    shares `nq_trend_tsmom`'s exact mechanism (TREND) -- so "mechanism"
    reads SAME -- but its declared `required_data` (needs high/low) differs
    from tsmom's, so "factor" must independently read CHANGED."""
    wc = what_changed(nq_trend_tsmom, mechanism=EconomicMechanism.TREND, root_symbol="NQ", strategy_family="breakout")
    assert "mechanism" in wc.same  # candidate mechanism == nq_trend_tsmom.mechanism (both TREND)
    assert "mechanism" not in wc.changed
    assert "factor" in wc.changed  # breakout was never part of THIS Factor
    assert "factor" not in wc.same
    assert "strategy_family" in wc.changed
    assert "genuinely new strategy variant" in wc.notes


def test_same_mechanism_and_same_data_requirements_still_does_not_imply_same_factor(nq_trend_tsmom):
    """The final Phase 2 semantic fix's core requirement, proven through the
    public `what_changed` API: ma_trend shares BOTH `nq_trend_tsmom`'s exact
    mechanism (TREND) AND its exact declared `required_data` (both only
    need "one continuous or raw OHLC price series per root") -- yet "factor"
    must still read CHANGED. Equal input-data requirements prove only shared
    data requirements, never equal quantitative factor representation."""
    from alpha_agent.alpha_memory.factor_identity import family_structural_signature

    assert family_structural_signature("tsmom") == family_structural_signature("ma_trend")
    wc = what_changed(nq_trend_tsmom, mechanism=EconomicMechanism.TREND, root_symbol="NQ", strategy_family="ma_trend")
    assert "mechanism" in wc.same
    assert "factor" in wc.changed
    assert "factor" not in wc.same
    assert "strategy_family" in wc.changed


def test_identical_params_are_same_not_changed(nq_trend_tsmom):
    assert nq_trend_tsmom.strategy_variants[0].canonical_params
    wc = what_changed(
        nq_trend_tsmom, mechanism=EconomicMechanism.TREND, root_symbol="NQ",
        strategy_family="tsmom", params=dict(nq_trend_tsmom.strategy_variants[0].canonical_params),
    )
    assert "parameterization" in wc.same
    assert "parameterization" not in wc.changed
    assert wc.parameter_differences == {}


def test_different_numeric_params_are_flagged_with_concrete_differences(nq_trend_tsmom):
    canonical = dict(nq_trend_tsmom.strategy_variants[0].canonical_params)
    bumped = {
        k: (v + 5 if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
        for k, v in canonical.items()
    }
    wc = what_changed(
        nq_trend_tsmom, mechanism=EconomicMechanism.TREND, root_symbol="NQ", strategy_family="tsmom", params=bumped,
    )
    assert "parameterization" in wc.changed
    assert "parameterization" not in wc.same
    for key, (old, new) in wc.parameter_differences.items():
        assert old == canonical[key]
        assert new == bumped[key]
        assert old != new


def test_no_strategy_family_given_makes_no_family_or_factor_claim(nq_trend_tsmom):
    wc = what_changed(nq_trend_tsmom, mechanism=EconomicMechanism.TREND, root_symbol="NQ")
    assert "strategy_family" not in wc.same
    assert "strategy_family" not in wc.changed
    assert "factor" not in wc.same
    assert "factor" not in wc.changed


def test_what_changed_never_infers_novelty_from_free_text(nq_trend_tsmom):
    """Section 15: "do not claim novelty from free text" -- `what_changed`
    takes no free-text argument at all (its signature is closed to
    mechanism/root/family/params), so novelty can never be smuggled in via a
    hypothesis title or narrative."""
    import inspect

    sig = inspect.signature(what_changed)
    assert set(sig.parameters) == {"alpha", "mechanism", "root_symbol", "strategy_family", "params"}
