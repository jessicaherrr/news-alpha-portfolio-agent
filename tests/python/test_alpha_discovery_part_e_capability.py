"""Alpha Discovery campaign, Part E -- capability-based execution gating
tests (task spec section 73).

`tests/python/test_execution_service.py` already proves the two most
important integration-level properties end to end:

* `test_replay_matches_committed_nq_tsmom_canonical_result` -- a legacy
  baseline family replays byte-identically through the generalized
  capability gate (this campaign's "preserve exact replay" requirement).
* `test_unsupported_family_is_invalid_execution_not_a_scientific_rejection`
  -- an unsupported strategy is a typed `INVALID_EXECUTION`, never a
  scientific `REJECT`.

This file covers the new `alpha_agent.execution.capability` module in
isolation (unit-level, no C++, no real data) plus the cadence-resolution
fallback `alpha_agent.agents.execution_service._resolve_cadence` added
alongside it.
"""
from __future__ import annotations

import inspect

import pytest
from alpha_agent.execution.capability import (
    SUPPORTED_CADENCES,
    CapabilityReason,
    assess_cadence,
    assess_required_features,
    assess_static_capability,
    classify_schedule_exception,
)
from alpha_agent.features import FeatureNameCollision, FeaturePriceDomainError, LookaheadUnsafeError
from alpha_agent.features.safety import FeatureSafetyError


def test_supported_cadences_match_the_frozen_legacy_signal_cadence_values():
    from alpha_agent.strategy.candidates_phase_13_5c import SIGNAL_CADENCE

    assert SUPPORTED_CADENCES == frozenset(SIGNAL_CADENCE.values())
    assert SUPPORTED_CADENCES == {"daily_trading_day", "native_1m"}


def test_assess_cadence_supported_and_unsupported():
    assert assess_cadence("daily_trading_day") is None
    assert assess_cadence("native_1m") is None
    assert assess_cadence("") is CapabilityReason.UNSUPPORTED_CADENCE
    assert assess_cadence("weekly") is CapabilityReason.UNSUPPORTED_CADENCE
    assert (
        assess_cadence("set by the orchestrator backtest configuration (not a StrategySpec field)")
        is CapabilityReason.UNSUPPORTED_CADENCE
    )


def test_assess_required_features_registered_and_unregistered():
    assert assess_required_features(["diff"]) is None  # real registered kind
    assert assess_required_features([]) is None  # vacuously fine
    assert assess_required_features(["not_a_real_feature_kind"]) is CapabilityReason.UNSUPPORTED_FEATURE


def test_assess_static_capability_reports_cadence_before_features():
    result = assess_static_capability(signal_cadence="bogus", required_features=["also_bogus"])
    assert result.supported is False
    assert result.reason is CapabilityReason.UNSUPPORTED_CADENCE  # cadence checked first


def test_assess_static_capability_supported_for_a_real_template_shape():
    result = assess_static_capability(signal_cadence="daily_trading_day", required_features=["diff"])
    assert result.supported is True
    assert result.reason is None


@pytest.mark.parametrize(
    "exc,expected",
    [
        (LookaheadUnsafeError("x"), CapabilityReason.NON_CAUSAL_FEATURE),
        (FeatureSafetyError("x"), CapabilityReason.NON_CAUSAL_FEATURE),
        (FeaturePriceDomainError("x"), CapabilityReason.DATA_ALIGNMENT_UNSUPPORTED),
        (FeatureNameCollision("x"), CapabilityReason.DATA_ALIGNMENT_UNSUPPORTED),
        (KeyError("x"), CapabilityReason.MISSING_DATA_LAYER),
        (RuntimeError("x"), CapabilityReason.EXECUTION_CAPABILITY_MISSING),
    ],
)
def test_classify_schedule_exception(exc, expected):
    assert classify_schedule_exception(exc) is expected


def test_capability_module_never_calls_eval_exec_or_shells_out():
    import ast

    import alpha_agent.execution.capability as mod

    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("eval", "exec")


# ---------------------------------------------------------------------------
# execution_service cadence-resolution fallback (backward-compat shim)
# ---------------------------------------------------------------------------


def test_resolve_cadence_prefers_the_members_own_declared_cadence():
    from alpha_agent.agents.execution_service import _resolve_cadence
    from alpha_agent.agents.orchestrator import FamilyMember
    from alpha_agent.strategy.baselines.factories import make_tsmom_spec
    from alpha_agent.strategy.baselines.params import TsmomParams

    spec = make_tsmom_spec(TsmomParams(fast_horizon=20, slow_horizon=120, size=1, root_symbol="NQ"))
    member = FamilyMember(
        ordinal=0, experiment_identity="experiment1:x", strategy_fingerprint="stratfp1:x",
        strategy_id=spec.strategy_id, strategy_family="tsmom", root_symbol="NQ",
        params={}, parameter_variant_identity="paramvariant1:x", feature_spec_fingerprint="featset1:x",
        hypothesis_id="H", hypothesis_title="t", strategy_spec=spec,
        strategy_spec_json={"schema": "registry-strategy-spec/1"},
        signal_cadence="native_1m",  # deliberately NOT what SIGNAL_CADENCE["tsmom"] says
    )
    assert _resolve_cadence(member) == "native_1m"


def test_resolve_cadence_falls_back_to_the_frozen_family_map_when_unset():
    from alpha_agent.agents.execution_service import _resolve_cadence
    from alpha_agent.agents.orchestrator import FamilyMember
    from alpha_agent.strategy.baselines.factories import make_tsmom_spec
    from alpha_agent.strategy.baselines.params import TsmomParams

    spec = make_tsmom_spec(TsmomParams(fast_horizon=20, slow_horizon=120, size=1, root_symbol="NQ"))
    member = FamilyMember(
        ordinal=0, experiment_identity="experiment1:x", strategy_fingerprint="stratfp1:x",
        strategy_id=spec.strategy_id, strategy_family="tsmom", root_symbol="NQ",
        params={}, parameter_variant_identity="paramvariant1:x", feature_spec_fingerprint="featset1:x",
        hypothesis_id="H", hypothesis_title="t", strategy_spec=spec,
        strategy_spec_json={"schema": "registry-strategy-spec/1"},
        # signal_cadence left at its empty default -- a hand-built / legacy member
    )
    assert _resolve_cadence(member) == "daily_trading_day"


def test_resolve_cadence_unknown_family_and_empty_cadence_stays_unresolved():
    from alpha_agent.agents.execution_service import _resolve_cadence
    from alpha_agent.agents.orchestrator import FamilyMember
    from alpha_agent.strategy.baselines.factories import make_tsmom_spec
    from alpha_agent.strategy.baselines.params import TsmomParams

    spec = make_tsmom_spec(TsmomParams(fast_horizon=20, slow_horizon=120, size=1, root_symbol="NQ"))
    member = FamilyMember(
        ordinal=0, experiment_identity="experiment1:x", strategy_fingerprint="stratfp1:x",
        strategy_id=spec.strategy_id, strategy_family="dsl:custom_blueprint", root_symbol="NQ",
        params={}, parameter_variant_identity="paramvariant1:x", feature_spec_fingerprint="featset1:x",
        hypothesis_id="H", hypothesis_title="t", strategy_spec=spec,
        strategy_spec_json={"schema": "registry-strategy-spec/1"},
    )
    resolved = _resolve_cadence(member)
    assert resolved not in SUPPORTED_CADENCES
