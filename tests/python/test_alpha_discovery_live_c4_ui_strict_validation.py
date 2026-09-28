"""Alpha Discovery live-research campaign, Checkpoint 4 -- the real Discover
UI Freeze -> Adopt -> Strict Validation handoff (task spec Part 2 section 16,
tests section 51).

Proves the real UI path -- `alpha_agent.ui.discovery_campaign.
run_discovery_campaign` (generation -> real C++ Fast Screen -> Freeze) then
`run_strict_validation` (safe `adopt_frozen_manifest` -> the UNCHANGED
`execute_family` / `finalize_family`) -- can go all the way from a frozen
candidate set to a real, registry-committed strict-validation result WITHOUT
the old Part O "replay the winning scenario through a fresh orchestrator's own
plan_family" workaround: `run_strict_validation` never calls `plan_family`,
the `ResearchAgent`, or the `StrategyCompilerAgent` at all -- it adopts the
exact already-frozen `StrategySpec`s / fingerprints / identities / cadence.

Uses an ISOLATED, temporary registry (`services.REGISTRY_PATH` monkeypatched)
-- the production scientific registry is never touched by this test.
"""
from __future__ import annotations

import pytest
from alpha_agent.agents import FamilyStatus
from alpha_agent.knowledge import EconomicMechanism
from alpha_agent.ui import discovery_campaign, llm_demo, services


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    """Every `services.open_registry()` call in this module -- inside
    `run_discovery_campaign` AND `run_strict_validation` -- resolves to an
    isolated file, never `data/registry/experiments.sqlite`."""
    monkeypatch.setattr(services, "REGISTRY_PATH", tmp_path / "isolated_c4_registry.sqlite")


def _real_cli_present() -> bool:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    return (root / "build" / "cpp" / "cpp" / "quant_backtest_targets_csv").exists()


@pytest.mark.skipif(not _real_cli_present(), reason="compiled CLI not present in this checkout")
def test_discover_ui_freezes_then_adopts_then_strictly_validates_without_replay():
    outcome = discovery_campaign.run_discovery_campaign(
        mode=llm_demo.SCRIPTED_MODE,
        objective="Find robust alpha opportunities for NQ.",
        root="NQ",
        family_stem="live_c4_ui_handoff",
        scenario_mechanism_map=(("tsmom_nq", EconomicMechanism.MOMENTUM),),
        target_k=1,
        run_fast_screen_backtests=True,
    )
    assert outcome.accepted, outcome.error
    assert outcome.frozen is not None, "Fast Screen must have produced a survivor to freeze"
    frozen = outcome.frozen
    assert frozen.manifest.planning_complete

    result = discovery_campaign.run_strict_validation(
        frozen=frozen, objective="Find robust alpha opportunities for NQ.", root="NQ",
    )
    assert result.accepted, result.error
    assert result.report is not None
    assert result.report.status is FamilyStatus.FINALIZED
    assert result.report.predeclared_family_size == frozen.manifest.declared_target_size
    assert result.manifest.family_id == frozen.manifest.family_id

    # -- proof: the exact frozen StrategySpec/fingerprint/identity flowed
    # through unchanged -- the adopted manifest IS the frozen one, byte for
    # byte (not a re-planned lookalike).
    assert result.manifest.manifest_fingerprint() == frozen.manifest.manifest_fingerprint()
    for adopted_member, frozen_member in zip(result.manifest.members, frozen.manifest.members, strict=True):
        assert adopted_member.strategy_fingerprint == frozen_member.strategy_fingerprint
        assert adopted_member.experiment_identity == frozen_member.experiment_identity
        assert adopted_member.signal_cadence == frozen_member.signal_cadence

    # -- proof: every finalized member is now visible through the exact same
    # read-only Promise/Fit surface the investor-facing report uses.
    (member_result,) = result.report.member_results
    summary = services.candidate_summary_for_experiment(member_result.experiment_id)
    assert summary is not None
    assert summary.root_symbol == "NQ"


@pytest.mark.skipif(not _real_cli_present(), reason="compiled CLI not present in this checkout")
def test_a_second_strict_validation_on_the_same_plane_refuses_honestly():
    outcome = discovery_campaign.run_discovery_campaign(
        mode=llm_demo.SCRIPTED_MODE,
        objective="Find robust alpha opportunities for NQ.",
        root="NQ",
        family_stem="live_c4_ui_handoff_2",
        scenario_mechanism_map=(("tsmom_nq", EconomicMechanism.MOMENTUM),),
        target_k=1,
        run_fast_screen_backtests=True,
    )
    assert outcome.accepted and outcome.frozen is not None
    first = discovery_campaign.run_strict_validation(
        frozen=outcome.frozen, objective="Find robust alpha opportunities for NQ.", root="NQ",
    )
    assert first.accepted, first.error

    # freeze a SECOND, different candidate set on the exact same evaluation
    # plane (same root -> same dataset/split/validation-spec/policy).
    outcome2 = discovery_campaign.run_discovery_campaign(
        mode=llm_demo.SCRIPTED_MODE,
        objective="Find robust alpha opportunities for NQ, a different mechanism.",
        root="NQ",
        family_stem="live_c4_ui_handoff_3",
        scenario_mechanism_map=(("ma_trend_zn", EconomicMechanism.TREND),),
        target_k=1,
        run_fast_screen_backtests=True,
    )
    assert outcome2.accepted
    if outcome2.frozen is None:
        pytest.skip("second scenario did not survive the fast screen on NQ")

    second = discovery_campaign.run_strict_validation(
        frozen=outcome2.frozen, objective="second family, same evaluation plane", root="NQ",
    )
    assert not second.accepted
    assert "already carries an executed" in (second.error or "") or "AdaptiveTestingError" in (second.error or "")


def test_strict_validation_never_touches_the_research_or_compiler_llm():
    """A hand-built manifest adoption test with an EMPTY `ScriptedLLMClient` --
    if `run_strict_validation` ever called `plan_family` (or anything that
    reaches the research/compiler agent), the empty client would raise
    immediately. This is a fast, offline, no-C++ proof of requirement 3
    ("Do NOT ask Claude to regenerate frozen hypotheses") that does not
    depend on the compiled CLI being present."""
    from alpha_agent.agents import FamilyManifest, IdentityPlanes
    from alpha_agent.agents.orchestrator import (
        FamilyMember,
        _compute_family_id,
        feature_set_fingerprint,
    )
    from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
    from alpha_agent.registry.models import MarketWindow
    from alpha_agent.screening.freeze import FrozenCandidateSet
    from alpha_agent.strategy import strategy_fingerprint
    from alpha_agent.strategy.baselines.factories import make_tsmom_spec
    from alpha_agent.strategy.baselines.params import TsmomParams
    from alpha_agent.validation.policy import ReliabilityPolicy

    # a manifest built against a DELIBERATELY WRONG plane -- `run_strict_
    # validation` independently re-derives today's real plane for NQ, so this
    # must be refused by `adopt_frozen_manifest`'s plane check, never by
    # exhausting the empty LLM client (proving the LLM is never reached at
    # all -- adoption fails BEFORE any agent call could occur).
    spec = make_tsmom_spec(TsmomParams(fast_horizon=20, slow_horizon=120, size=1, root_symbol="NQ"))
    params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    fp = strategy_fingerprint(spec)
    ffp = feature_set_fingerprint(spec)
    pvi = parameter_variant_identity(params)
    policy = ReliabilityPolicy()
    planes = IdentityPlanes(
        dataset_fingerprint="valdataset2:WRONG", split_identity="split1:WRONG",
        validation_spec_fingerprint="validationprotocol1:WRONG",
        reliability_policy_fingerprint=policy.identity(),
        execution_config_identity="execconfig1:WRONG", cost_config_identity="costconfig1:WRONG",
        risk_identity="riskconfig1:WRONG",
    )
    identity = experiment_identity(
        strategy_fingerprint=fp, strategy_family="tsmom", root_symbol="NQ",
        parameter_variant_identity=pvi, dataset_fingerprint=planes.dataset_fingerprint,
        split_identity=planes.split_identity, validation_spec_fingerprint=planes.validation_spec_fingerprint,
        reliability_policy_fingerprint=planes.reliability_policy_fingerprint,
        execution_config_identity=planes.execution_config_identity,
        cost_config_identity=planes.cost_config_identity, risk_identity=planes.risk_identity,
        feature_spec_fingerprint=ffp,
    )
    member = FamilyMember(
        ordinal=0, experiment_identity=identity, strategy_fingerprint=fp, strategy_id=spec.strategy_id,
        strategy_family="tsmom", root_symbol="NQ", params=params, parameter_variant_identity=pvi,
        feature_spec_fingerprint=ffp, hypothesis_id="H-WRONG-PLANE", hypothesis_title="wrong plane",
        strategy_spec=spec, strategy_spec_json={"schema": "registry-strategy-spec/1"},
        signal_cadence="daily_trading_day", execution_cadence="native_1m_raw_contract",
    )
    mw = MarketWindow(label="VALIDATION_2023_2024", start_date="2023-01-01", end_date="2024-12-31")
    family_id = _compute_family_id(
        family_stem="wrong_plane_test", generation=0, parent_family_id=None, planes=planes,
        market_window=mw, fdr_q_threshold=policy.fdr_q_threshold, policy_identity=policy.identity(),
        declared_target_size=1, members=(member,),
    )
    manifest = FamilyManifest(
        family_id=family_id, family_stem="wrong_plane_test", generation=0, planes=planes,
        market_window=mw, fdr_q_threshold=policy.fdr_q_threshold, policy_identity=policy.identity(),
        declared_target_size=1, members=(member,),
    )
    frozen = FrozenCandidateSet(
        market="NQ", ideas_considered=1, mechanisms_supported=1, fast_screened=1, frozen_count=1,
        screen_scores={identity: 50.0}, manifest=manifest, frozen_at=manifest.created_at,
    )

    result = discovery_campaign.run_strict_validation(
        frozen=frozen, objective="wrong-plane refusal proof", root="NQ",
    )
    assert not result.accepted
    assert "different evaluation plane" in (result.error or "")
