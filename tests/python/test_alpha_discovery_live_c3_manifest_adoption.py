"""Alpha Discovery live-research campaign, Checkpoint 3 -- safe frozen-manifest
adoption (task spec Part 2, sections 10-16, tests section 52).

`ResearchOrchestrator.execute_family` has always required a `FamilyManifest` to
have been produced by THAT SAME orchestrator instance's own `plan_family()`
call (`_verify_manifest` checks `self._planned_snapshots`) -- a deliberate
Phase 18 anti-tampering invariant. A manifest built OUTSIDE any orchestrator
(e.g. `alpha_agent.screening.freeze.freeze_top_k`'s Top-K survivors) could
therefore never be executed by a fresh orchestrator (see
`docs/ALPHA_DISCOVERY_CAMPAIGN.md`'s "architectural finding").

`ResearchOrchestrator.adopt_frozen_manifest` closes that gap WITHOUT weakening
the invariant: it runs every check a locally-planned manifest gets "for free"
explicitly, then registers the manifest into `self._planned` /
`self._planned_snapshots` exactly as `plan_family` would, so `execute_family` /
`finalize_family` need no changes at all.

Every test here drives both LLM agents with `ScriptedLLMClient` and the
execution step with `ScriptedExecutionValidationService`: no network, no real
model, no real backtest, fully deterministic -- mirrors
`tests/python/test_phase_18_research_orchestrator.py`'s own fixtures.
"""
from __future__ import annotations

import json

import pytest
from alpha_agent.agents import (
    AdaptiveTestingError,
    FamilyManifest,
    FamilyManifestError,
    FamilyMember,
    FamilyPlanSpec,
    FamilyStatus,
    IdentityPlanes,
    NonFamilyVerdict,
    OrchestratorBudget,
    OrchestratorConfig,
    ResearchAgent,
    ResearchOrchestrator,
    ScriptedExecutionValidationService,
    ScriptedLLMClient,
    StrategyCompilerAgent,
    TrialEvidence,
)
from alpha_agent.agents import orchestrator as orch_mod
from alpha_agent.registry import ExperimentRegistry
from alpha_agent.registry.enums import AttemptStatus
from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
from alpha_agent.screening.freeze import adopt_frozen_candidate_set
from alpha_agent.strategy import strategy_fingerprint
from alpha_agent.strategy.baselines.factories import make_tsmom_spec
from alpha_agent.strategy.baselines.params import TsmomParams
from alpha_agent.validation.policy import ReliabilityPolicy

UNIVERSE = ("ES", "NQ", "CL", "GC", "ZN")
MW = orch_mod.MarketWindow(label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31")
OTHER_MW = orch_mod.MarketWindow(label="OTHER", start_date="2022-01-01", end_date="2022-12-31")
POLICY = ReliabilityPolicy()
OTHER_POLICY = ReliabilityPolicy(fdr_q_threshold=0.05)
VSPEC = "validationspec1:vs"


def _planes(**overrides) -> IdentityPlanes:
    base = {
        "dataset_fingerprint": "valdataset2:ds",
        "split_identity": "split1:sp",
        "validation_spec_fingerprint": VSPEC,
        "reliability_policy_fingerprint": POLICY.identity(),
        "execution_config_identity": "execconfig1:ex",
        "cost_config_identity": "costconfig1:co",
        "risk_identity": "riskconfig1:ri",
    }
    base.update(overrides)
    return IdentityPlanes(**base)


def _config(*, planes=None, target_size=1, market_window=MW, universe=UNIVERSE, **budget) -> OrchestratorConfig:
    b = {"max_planning_attempts_per_family": 10, "plan_successor_family_after_finalization": False}
    b.update(budget)
    return OrchestratorConfig(
        objective="adopt an externally-frozen family for strict validation",
        market_universe=universe,
        market_window=market_window,
        planes=planes or _planes(),
        family_plan=FamilyPlanSpec(family_stem="live_c3_adoption", target_family_size=target_size),
        budget=OrchestratorBudget(**b),
        phase="alpha-discovery-live-c3",
    )


def _hyp(hid: str, *, universe=("ES",)) -> str:
    return json.dumps(
        {
            "hypothesis_id": hid,
            "title": f"{universe[0]} multi-week time-series momentum",
            "economic_mechanism": (
                "Gradual diffusion of macro information leaves index returns "
                "positively autocorrelated at multi-week horizons."
            ),
            "universe": list(universe),
            "horizon": "20 trading days",
            "required_features": ["diff"],
            "signal_description": "Long when both a fast and a slow price change are positive.",
            "expected_regime": "trending macro regime",
            "failure_regime": "choppy range-bound regime",
            "falsification_test": "No positive OOS net PnL after costs across folds.",
        }
    )


def _tmpl(family: str = "tsmom", root: str = "ES") -> str:
    return json.dumps(
        {
            "expressible": True,
            "template": {
                "family_key": family, "root_symbol": root,
                "params": {"fast_horizon": 20, "slow_horizon": 120, "size": 1},
            },
            "rationale": "multi-horizon trend maps to the tsmom family",
        }
    )


def _valid(p: float = 0.2) -> TrialEvidence:
    return TrialEvidence(
        status=AttemptStatus.VALID,
        trial_p_value=p,
        non_family_verdict=NonFamilyVerdict.ELIGIBLE,
        metrics={"net_pnl_usd": 1000.0, "daily_sharpe": 0.6, "n_trades": 40},
        engine="synthetic-quant-core",
        report_fingerprint="valreport1:x",
        source_artifact="synthetic",
        source_artifact_sha256="a" * 64,
        produced_under_validation_spec_fingerprint=VSPEC,
        produced_under_reliability_policy_fingerprint=POLICY.identity(),
    )


def _registry(tmp_path, name="registry.sqlite") -> ExperimentRegistry:
    return ExperimentRegistry(tmp_path / name)


def _orch(reg, *, config, hyps=(), plans=(), evidence=(), policy=POLICY) -> ResearchOrchestrator:
    return ResearchOrchestrator(
        registry=reg,
        research_agent=ResearchAgent(ScriptedLLMClient(list(hyps))),
        compiler_agent=StrategyCompilerAgent(ScriptedLLMClient(list(plans))),
        execution_service=ScriptedExecutionValidationService(list(evidence)),
        reliability_policy=policy,
        config=config,
    )


def _produce_manifest(tmp_path):
    """Build a manifest exactly the way `freeze_top_k` would hand one to a
    fresh orchestrator: a real `plan_family()` call on a throwaway "producer"
    orchestrator, never touched again."""
    reg = _registry(tmp_path, "producer.sqlite")
    producer = _orch(
        reg, config=_config(target_size=1), hyps=[_hyp("H-1")], plans=[_tmpl()], evidence=[_valid()],
    )
    return producer.plan_family()


# ---------------------------------------------------------------------------
# positive: a legitimate frozen manifest can be adopted end to end
# ---------------------------------------------------------------------------
def test_a_legitimately_frozen_manifest_can_be_adopted_and_executed(tmp_path):
    manifest = _produce_manifest(tmp_path)

    reg = _registry(tmp_path, "consumer.sqlite")
    consumer = _orch(reg, config=_config(target_size=1), evidence=[_valid(0.001)])

    # never planned by `consumer` -- the pre-existing invariant still refuses it.
    with pytest.raises(FamilyManifestError, match="was not planned"):
        consumer.execute_family(manifest)

    adopted = consumer.adopt_frozen_manifest(manifest)
    assert adopted.family_id == manifest.family_id

    evidence = consumer.execute_family(manifest)
    report = consumer.finalize_family(manifest, evidence)
    assert report.status is FamilyStatus.FINALIZED
    assert report.predeclared_family_size == 1


def test_adopt_frozen_candidate_set_convenience_wrapper(tmp_path):
    """`alpha_agent.screening.freeze.adopt_frozen_candidate_set` is the thin
    wrapper a real Fast-Screen/Freeze caller uses; it must thread
    `screen_scores` through as `fast_screen_provenance`."""
    from alpha_agent.screening.freeze import FrozenCandidateSet

    manifest = _produce_manifest(tmp_path)
    frozen = FrozenCandidateSet(
        market="ES", ideas_considered=1, mechanisms_supported=1, fast_screened=1,
        frozen_count=1, screen_scores={manifest.members[0].experiment_identity: 0.75},
        manifest=manifest, frozen_at=manifest.created_at,
    )
    reg = _registry(tmp_path, "consumer2.sqlite")
    consumer = _orch(reg, config=_config(target_size=1), evidence=[_valid(0.001)])
    adopt_frozen_candidate_set(consumer, frozen)
    evidence = consumer.execute_family(manifest)
    report = consumer.finalize_family(manifest, evidence)
    assert report.status is FamilyStatus.FINALIZED


# ---------------------------------------------------------------------------
# negative: tampering / mismatch is refused
# ---------------------------------------------------------------------------
def test_manifest_with_wrong_schema_version_is_refused(tmp_path):
    manifest = _produce_manifest(tmp_path)
    tampered = manifest.model_copy(update={"schema_version": "family-manifest/1"})
    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(reg, config=_config(target_size=1))
    with pytest.raises(FamilyManifestError, match="schema_version"):
        consumer.adopt_frozen_manifest(tampered)


def test_manifest_with_altered_family_id_is_rejected(tmp_path):
    manifest = _produce_manifest(tmp_path)
    tampered = manifest.model_copy(update={"family_id": "familymanifest1:deadbeef"})
    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(reg, config=_config(target_size=1))
    with pytest.raises(FamilyManifestError):
        consumer.adopt_frozen_manifest(tampered)


def test_nested_param_mutation_is_detected_at_adoption(tmp_path):
    """`family_id` hashes each member's `identity_tuple()` only (ordinal,
    experiment_identity, strategy_fingerprint, parameter_variant_identity,
    trial_role, root_symbol) -- NOT `params` / `strategy_spec_json` /
    `strategy_spec`. A tamper confined to those fields passes
    `assert_consistent()` but must still be caught by adoption's
    `_manifest_snapshot` re-derivation."""
    manifest = _produce_manifest(tmp_path)
    manifest.members[0].params["fast_horizon"] = 999  # in-place: family_id unchanged
    manifest.assert_consistent()  # proves the tamper is invisible to the content hash alone

    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(reg, config=_config(target_size=1))
    with pytest.raises(FamilyManifestError):
        consumer.adopt_frozen_manifest(manifest)


def test_nested_strategy_spec_rule_mutation_is_detected_at_adoption(tmp_path):
    """A tamper to the SCIENTIFICALLY MEANINGFUL part of the spec (a rule's
    target) changes `strategy_fingerprint(spec)` while the member's stored
    `strategy_fingerprint` field stays the old value -- adoption's
    `_manifest_snapshot` re-derivation must catch the mismatch."""
    manifest = _produce_manifest(tmp_path)
    spec = manifest.members[0].strategy_spec
    old_rule = spec.rules[0]
    spec.rules[0] = old_rule.model_copy(
        update={"action": old_rule.action.model_copy(
            update={"target_units": old_rule.action.target_units + 1}
        )}
    )
    manifest.assert_consistent()  # family_id does not hash the full spec -- unchanged

    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(reg, config=_config(target_size=1))
    with pytest.raises(FamilyManifestError):
        consumer.adopt_frozen_manifest(manifest)


def test_manifest_from_a_different_dataset_plane_is_rejected(tmp_path):
    manifest = _produce_manifest(tmp_path)
    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(
        reg, config=_config(planes=_planes(dataset_fingerprint="valdataset2:OTHER"), target_size=1)
    )
    with pytest.raises(FamilyManifestError, match="different evaluation plane"):
        consumer.adopt_frozen_manifest(manifest)


def test_manifest_from_a_different_validation_spec_is_rejected(tmp_path):
    manifest = _produce_manifest(tmp_path)
    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(
        reg, config=_config(planes=_planes(validation_spec_fingerprint="validationspec1:OTHER"), target_size=1)
    )
    with pytest.raises(FamilyManifestError, match="different evaluation plane"):
        consumer.adopt_frozen_manifest(manifest)


def test_manifest_with_wrong_bh_q_threshold_is_rejected(tmp_path):
    """`fdr_q_threshold` is hashed directly into `family_id`
    (`_compute_family_id`), so this particular tamper is already caught by
    the manifest's own content-hash self-consistency check -- proving
    defense-in-depth: adoption's dedicated q-threshold check (against the
    orchestrator's OWN adjudicator) never even needs to run for this case."""
    manifest = _produce_manifest(tmp_path)
    tampered = manifest.model_copy(update={"fdr_q_threshold": 0.05})
    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(reg, config=_config(target_size=1))
    with pytest.raises(FamilyManifestError, match="does not match"):
        consumer.adopt_frozen_manifest(tampered)


def test_manifest_from_a_different_reliability_policy_is_rejected(tmp_path):
    manifest = _produce_manifest(tmp_path)
    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(
        reg,
        config=_config(
            planes=_planes(reliability_policy_fingerprint=OTHER_POLICY.identity()),
            target_size=1,
        ),
        policy=OTHER_POLICY,
    )
    with pytest.raises(FamilyManifestError, match="different evaluation plane"):
        consumer.adopt_frozen_manifest(manifest)


def test_manifest_with_a_root_outside_the_approved_universe_is_rejected(tmp_path):
    manifest = _produce_manifest(tmp_path)  # root ES
    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(reg, config=_config(target_size=1, universe=("NQ", "CL")))
    with pytest.raises(FamilyManifestError, match="approved market universe"):
        consumer.adopt_frozen_manifest(manifest)


def test_manifest_from_a_different_market_window_is_rejected(tmp_path):
    manifest = _produce_manifest(tmp_path)
    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(reg, config=_config(target_size=1, market_window=OTHER_MW))
    with pytest.raises(FamilyManifestError, match="market_window"):
        consumer.adopt_frozen_manifest(manifest)


def test_manifest_declaring_an_unsupported_cadence_is_rejected(tmp_path):
    """`resolve_cadence`'s legacy-family fallback silently self-heals a bad
    `signal_cadence` string for any of the 5 legacy family NAMES (by design --
    it exists for an older/hand-built caller that never populated the field).
    So a genuinely uncheckable cadence can only arise for a member whose
    `strategy_family` is not one of those names -- e.g. a buggy Fast-Screen /
    Freeze producer that emitted a non-legacy family label with a cadence this
    release cannot schedule. Hand-built (mirrors
    `test_execution_service.py::_member_for`) so the identity fields are
    self-consistent and only the cadence is genuinely bad."""
    spec = make_tsmom_spec(TsmomParams(fast_horizon=20, slow_horizon=120, size=1, root_symbol="ES"))
    params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    fp = strategy_fingerprint(spec)
    ffp = orch_mod.feature_set_fingerprint(spec)
    pvi = parameter_variant_identity(params)
    planes = _planes()
    identity = experiment_identity(
        strategy_fingerprint=fp, strategy_family="dsl:weekly:custom", root_symbol="ES",
        parameter_variant_identity=pvi, dataset_fingerprint=planes.dataset_fingerprint,
        split_identity=planes.split_identity,
        validation_spec_fingerprint=planes.validation_spec_fingerprint,
        reliability_policy_fingerprint=planes.reliability_policy_fingerprint,
        execution_config_identity=planes.execution_config_identity,
        cost_config_identity=planes.cost_config_identity, risk_identity=planes.risk_identity,
        feature_spec_fingerprint=ffp,
    )
    member = FamilyMember(
        ordinal=0, experiment_identity=identity, strategy_fingerprint=fp,
        strategy_id=spec.strategy_id, strategy_family="dsl:weekly:custom", root_symbol="ES",
        params=params, parameter_variant_identity=pvi, feature_spec_fingerprint=ffp,
        hypothesis_id="H-BAD-CADENCE", hypothesis_title="bad cadence",
        strategy_spec=spec, strategy_spec_json={"schema": "registry-strategy-spec/1"},
        signal_cadence="weekly", execution_cadence="native_1m_raw_contract",
    )
    family_id = orch_mod._compute_family_id(
        family_stem="live_c3_adoption", generation=0, parent_family_id=None, planes=planes,
        market_window=MW, fdr_q_threshold=POLICY.fdr_q_threshold, policy_identity=POLICY.identity(),
        declared_target_size=1, members=(member,),
    )
    manifest = FamilyManifest(
        family_id=family_id, family_stem="live_c3_adoption", generation=0, planes=planes,
        market_window=MW, fdr_q_threshold=POLICY.fdr_q_threshold, policy_identity=POLICY.identity(),
        declared_target_size=1, members=(member,),
    )

    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(reg, config=_config(target_size=1))
    with pytest.raises(FamilyManifestError, match="cannot schedule"):
        consumer.adopt_frozen_manifest(manifest)


def test_underfilled_manifest_is_rejected(tmp_path):
    reg = _registry(tmp_path, "producer.sqlite")
    producer = _orch(
        reg, config=_config(target_size=2, max_planning_attempts_per_family=1),
        hyps=[_hyp("H-1")], plans=[_tmpl()], evidence=[_valid()],
    )
    manifest = producer.plan_family()
    assert not manifest.planning_complete  # only 1 of 2 slots filled

    reg2 = _registry(tmp_path, "c.sqlite")
    consumer = _orch(reg2, config=_config(target_size=2))
    with pytest.raises(FamilyManifestError, match="not fully predeclared"):
        consumer.adopt_frozen_manifest(manifest)


def test_fast_screen_provenance_mismatch_is_rejected(tmp_path):
    manifest = _produce_manifest(tmp_path)
    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(reg, config=_config(target_size=1))
    with pytest.raises(FamilyManifestError, match="fast_screen_provenance"):
        consumer.adopt_frozen_manifest(
            manifest, fast_screen_provenance={"experiment1:not_a_real_member": 0.5}
        )


# ---------------------------------------------------------------------------
# no outcome-adaptive adoption / re-adoption
# ---------------------------------------------------------------------------
def test_the_same_family_id_cannot_be_adopted_twice(tmp_path):
    manifest = _produce_manifest(tmp_path)
    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(reg, config=_config(target_size=1), evidence=[_valid()])
    consumer.adopt_frozen_manifest(manifest)
    with pytest.raises(FamilyManifestError, match="already planned/adopted"):
        consumer.adopt_frozen_manifest(manifest)


def test_an_already_executed_plane_refuses_a_second_adopted_family(tmp_path):
    manifest = _produce_manifest(tmp_path)
    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(reg, config=_config(target_size=1), evidence=[_valid(0.001)])
    consumer.adopt_frozen_manifest(manifest)
    evidence = consumer.execute_family(manifest)
    consumer.finalize_family(manifest, evidence)

    # a second, DIFFERENT frozen family on the SAME evaluation plane.
    reg2 = _registry(tmp_path, "producer2.sqlite")
    producer2 = _orch(
        reg2, config=_config(target_size=1), hyps=[_hyp("H-2", universe=("NQ",))],
        plans=[_tmpl("tsmom", "NQ")], evidence=[_valid()],
    )
    second_manifest = producer2.plan_family()
    assert second_manifest.family_id != manifest.family_id

    with pytest.raises(AdaptiveTestingError):
        consumer.adopt_frozen_manifest(second_manifest)


def test_post_adoption_mutation_is_rejected_by_the_unchanged_execute_family_guard(tmp_path):
    manifest = _produce_manifest(tmp_path)
    reg = _registry(tmp_path, "c.sqlite")
    consumer = _orch(reg, config=_config(target_size=1), evidence=[_valid()])
    consumer.adopt_frozen_manifest(manifest)

    manifest.members[0].params["fast_horizon"] = 999  # mutate AFTER adoption
    with pytest.raises(FamilyManifestError):
        consumer.execute_family(manifest)


