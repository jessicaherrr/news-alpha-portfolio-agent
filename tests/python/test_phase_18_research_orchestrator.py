"""Phase 18 / 18.1 / 18.2 -- runtime Research Orchestrator (prompt 18).

Every test drives both LLM agents with :class:`ScriptedLLMClient` and the
"C++ Quant Core + frozen per-trial reliability gates" step with
:class:`ScriptedExecutionValidationService`: no network, no real model, no real
backtest, fully deterministic.

18.1 statistical-integrity proofs (the family is fully predeclared before any
member executes):

* every family member identity is frozen before the first execution call;
* no within-family reflection / outcome-adaptive membership is possible;
* changing or adding a member after execution starts fails loudly;
* BH q-values and final verdicts are produced only at family finalization;
* no partial family can produce a scientific PASS;
* an INVALID_EXECUTION attempt has no ResultRecord;
* invalid -> corrected VALID stays the same experiment identity;
* reflection after finalization creates a NEW family, never mutates the old one.

18.2 additional proofs:

* generation 2 cannot be scientifically executed on the same validation plane
  merely because generation 1 was finalized and reflected;
* reflection may create a planning-only next-family proposal without consuming a
  new statistical testing budget;
* changing the ReliabilityPolicy / FDR q breaks the bound policy identity;
* a mismatched adjudicator / policy fingerprint fails before execution;
* nested params / StrategySpec / strategy_spec_json mutation after planning is
  detected;
* target-family underfill cannot silently become a smaller tested BH family.
"""
from __future__ import annotations

import inspect
import json

import pytest
from alpha_agent.agents import (
    AdaptiveTestingError,
    EvidenceProvenanceError,
    FamilyManifestError,
    FamilyPlanSpec,
    FamilyStatus,
    IdentityPlanes,
    MemberDisposition,
    NonFamilyVerdict,
    OrchestratorBudget,
    OrchestratorConfig,
    PlanningDisposition,
    PolicyFamilyAdjudicator,
    ResearchAgent,
    ResearchContext,
    ResearchOrchestrator,
    ScriptedExecutionValidationService,
    ScriptedLLMClient,
    StrategyCompilerAgent,
    TrialEvidence,
)
from alpha_agent.agents import orchestrator as orch_mod
from alpha_agent.agents.orchestrator import HoldoutIsolationError, feature_set_fingerprint
from alpha_agent.core.instrument import AssetDomain
from alpha_agent.registry import ExperimentRegistry
from alpha_agent.registry.enums import AttemptStatus, InvalidationClass, RegistryVerdict, TrialRole
from alpha_agent.registry.holdout_guard import HoldoutAccessError
from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
from alpha_agent.strategy import strategy_fingerprint
from alpha_agent.strategy.candidates_phase_13_5c import spec_for_params
from alpha_agent.validation.policy import ReliabilityPolicy
from pydantic import ValidationError

UNIVERSE = ("ES", "NQ", "CL", "GC", "ZN")
MW = orch_mod.MarketWindow(label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31")
POLICY = ReliabilityPolicy()
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


def _config(*, planes=None, target_size=3, successor=False, **budget) -> OrchestratorConfig:
    b = {
        "max_planning_attempts_per_family": 10,
        "plan_successor_family_after_finalization": successor,
    }
    b.update(budget)
    return OrchestratorConfig(
        objective="propose robust trend hypotheses for index futures",
        market_universe=UNIVERSE,
        market_window=MW,
        planes=planes or _planes(),
        family_plan=FamilyPlanSpec(family_stem="phase18_tsmom", target_family_size=target_size),
        budget=OrchestratorBudget(**b),
        knowledge_base=("TSMOM earns a cross-asset premium (Moskowitz 2012).",),
        phase="18.2",
        code_commit="testcommit",
    )


def _hyp(hid: str, *, universe=("ES",), novelty: str = "") -> str:
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
            "novelty_notes": novelty,
        }
    )


def _tmpl(family: str = "tsmom", root: str = "ES", params: dict | None = None) -> str:
    canonical = {
        "tsmom": {"fast_horizon": 20, "slow_horizon": 120, "size": 1},
        "ma_trend": {"fast_window": 50, "slow_window": 200, "size": 1},
    }
    return json.dumps(
        {
            "expressible": True,
            "template": {"family_key": family, "root_symbol": root, "params": params or canonical[family]},
            "rationale": "multi-horizon trend maps to the tsmom family",
        }
    )


def _valid(
    p: float, nfv: NonFamilyVerdict = NonFamilyVerdict.ELIGIBLE, *, stamp=True, policy=POLICY, **metrics
) -> TrialEvidence:
    m = {"net_pnl_usd": 1000.0, "daily_sharpe": 0.6, "n_trades": 40}
    m.update(metrics)
    return TrialEvidence(
        status=AttemptStatus.VALID,
        trial_p_value=p,
        non_family_verdict=nfv,
        non_family_reason_codes=() if nfv is NonFamilyVerdict.ELIGIBLE else ("NON_FAMILY",),
        metrics=m,
        engine="synthetic-quant-core",
        report_fingerprint="valreport1:x",
        source_artifact="synthetic",
        source_artifact_sha256="a" * 64,
        produced_under_validation_spec_fingerprint=VSPEC if stamp else None,
        produced_under_reliability_policy_fingerprint=policy.identity() if stamp else None,
    )


def _invalid() -> TrialEvidence:
    return TrialEvidence(
        status=AttemptStatus.INVALID_EXECUTION,
        invalidation_class=InvalidationClass.FEATURE_PIPELINE_DEFECT,
        invalidation_detail="weekend gaps fragmented RESET_ON_GAP windows",
        engine="synthetic-quant-core",
    )


def _registry(tmp_path) -> ExperimentRegistry:
    return ExperimentRegistry(tmp_path / "registry.sqlite")


def _orch(reg, *, hyps, plans, evidence, config: OrchestratorConfig | None = None, policy=POLICY) -> ResearchOrchestrator:
    return ResearchOrchestrator(
        registry=reg,
        research_agent=ResearchAgent(ScriptedLLMClient(hyps)),
        compiler_agent=StrategyCompilerAgent(ScriptedLLMClient(plans)),
        execution_service=ScriptedExecutionValidationService(evidence),
        reliability_policy=policy,
        config=config or _config(),
    )


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------
def test_plan_execute_finalize_one_family(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(
        reg,
        hyps=[_hyp("H-1", universe=("ES",)), _hyp("H-2", universe=("NQ",)), _hyp("H-3", universe=("CL",))],
        plans=[_tmpl("tsmom", "ES"), _tmpl("tsmom", "NQ"), _tmpl("tsmom", "CL")],
        evidence=[_valid(0.001), _valid(0.30), _valid(0.90)],
        config=_config(target_size=3),
    )
    report = orch.run()

    (fam,) = report.generations
    assert fam.status is FamilyStatus.FINALIZED
    assert fam.predeclared_family_size == 3
    assert fam.n_bh_rejected == 1
    assert fam.verdict_counts == {"PASS": 1, "REJECT": 2}
    # the BH threshold and the policy identity came from the bound frozen policy
    assert fam.fdr_q_threshold == POLICY.fdr_q_threshold
    assert fam.policy_identity == POLICY.identity()
    qs = [m.bh_q_value for m in fam.member_results]
    assert qs[0] == pytest.approx(0.003)
    assert qs[1] == pytest.approx(0.45)
    assert report.passing_experiment_ids == (fam.member_results[0].experiment_id,)
    assert report.termination_reason == "family:finalized"

    summary = reg.summary()
    assert summary.authoritative_statistical_hypotheses == 3
    assert summary.canonical_verdict_counts == {"PASS": 1, "REJECT": 2, "INCONCLUSIVE": 0}


# --------------------------------------------------------------------------
# 18.1 proof: member identities frozen before any execution
# --------------------------------------------------------------------------
def test_all_member_identities_frozen_before_any_execution(tmp_path):
    reg = _registry(tmp_path)
    svc = ScriptedExecutionValidationService([_valid(0.2), _valid(0.2)])
    orch = ResearchOrchestrator(
        registry=reg,
        research_agent=ResearchAgent(ScriptedLLMClient([_hyp("H-1"), _hyp("H-2", universe=("NQ",))])),
        compiler_agent=StrategyCompilerAgent(ScriptedLLMClient([_tmpl("tsmom", "ES"), _tmpl("tsmom", "NQ")])),
        execution_service=svc,
        reliability_policy=POLICY,
        config=_config(target_size=2),
    )
    manifest = orch.plan_family()

    assert svc.call_count == 0
    assert len(manifest.members) == 2
    frozen_ids = manifest.member_identities()
    fp_before = manifest.manifest_fingerprint()

    orch.execute_family(manifest)
    assert {c["experiment_identity"] for c in svc.calls} == frozen_ids
    assert manifest.manifest_fingerprint() == fp_before


# --------------------------------------------------------------------------
# 18.1 proof: no within-family reflection / outcome-adaptive membership
# --------------------------------------------------------------------------
def test_planning_context_is_invariant_across_slots_and_carries_no_outcome(tmp_path):
    reg = _registry(tmp_path)
    seen: list[ResearchContext] = []
    real = ResearchAgent.propose

    def spy(self, ctx):
        seen.append(ctx)
        return real(self, ctx)

    orch = _orch(
        reg,
        hyps=[_hyp("H-1", universe=("ES",)), _hyp("H-2", universe=("NQ",)), _hyp("H-3", universe=("CL",))],
        plans=[_tmpl("tsmom", "ES"), _tmpl("tsmom", "NQ"), _tmpl("tsmom", "CL")],
        evidence=[_valid(0.001), _valid(0.5), _valid(0.5)],
        config=_config(target_size=3),
    )
    orch._research.propose = spy.__get__(orch._research, ResearchAgent)  # type: ignore[attr-defined]

    manifest = orch.plan_family()
    assert len(seen) == 3
    assert seen[0].sha256() == seen[1].sha256() == seen[2].sha256()
    blob = json.dumps(seen[0].model_dump(mode="json"))
    for leak in ("trial_p_value", "bh_q", "SCIENTIFIC_PASS", "0.001"):
        assert leak not in blob

    calls_after_plan = orch._research.propose.__self__._client.call_count  # type: ignore[attr-defined]
    evidence = orch.execute_family(manifest)
    orch.finalize_family(manifest, evidence)
    assert orch._research.propose.__self__._client.call_count == calls_after_plan  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# 18.1 proof: changing / adding a member after freeze fails loudly
# --------------------------------------------------------------------------
def test_adding_a_member_after_freeze_fails_loudly(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(
        reg,
        hyps=[_hyp("H-1"), _hyp("H-2", universe=("NQ",))],
        plans=[_tmpl("tsmom", "ES"), _tmpl("tsmom", "NQ")],
        evidence=[_valid(0.2), _valid(0.2)],
        config=_config(target_size=2),
    )
    manifest = orch.plan_family()
    evidence = orch.execute_family(manifest)

    extra = manifest.members[0].model_copy(update={"ordinal": 2, "experiment_identity": "experiment1:deadbeef"})
    tampered = manifest.model_copy(update={"members": manifest.members + (extra,)})
    with pytest.raises(FamilyManifestError):
        tampered.assert_consistent()
    with pytest.raises(FamilyManifestError):
        orch.finalize_family(tampered, evidence)


def test_a_manifest_this_orchestrator_did_not_plan_is_refused(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(
        reg,
        hyps=[_hyp("H-1")],
        plans=[_tmpl("tsmom", "ES")],
        evidence=[_valid(0.2)],
        config=_config(target_size=1),
    )
    manifest = orch.plan_family()
    other = ResearchOrchestrator(
        registry=_registry(tmp_path.joinpath("b")),
        research_agent=ResearchAgent(ScriptedLLMClient([])),
        compiler_agent=StrategyCompilerAgent(ScriptedLLMClient([])),
        execution_service=ScriptedExecutionValidationService([]),
        reliability_policy=POLICY,
        config=_config(target_size=1),
    )
    with pytest.raises(FamilyManifestError):
        other.execute_family(manifest)


# --------------------------------------------------------------------------
# 18.2 proof: nested-mutation of a frozen member is detected via re-derivation
# --------------------------------------------------------------------------
def test_nested_param_mutation_after_planning_is_detected(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(reg, hyps=[_hyp("H-1")], plans=[_tmpl("tsmom", "ES")], evidence=[_valid(0.2)], config=_config(target_size=1))
    manifest = orch.plan_family()
    manifest.members[0].params["fast_horizon"] = 999  # mutate the nested dict in place
    with pytest.raises(FamilyManifestError):
        orch.execute_family(manifest)


def test_nested_strategy_spec_json_mutation_after_planning_is_detected(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(reg, hyps=[_hyp("H-1")], plans=[_tmpl("tsmom", "ES")], evidence=[_valid(0.2)], config=_config(target_size=1))
    manifest = orch.plan_family()
    manifest.members[0].strategy_spec_json["params"]["injected"] = 7
    with pytest.raises(FamilyManifestError):
        orch.execute_family(manifest)


def test_nested_strategy_spec_metadata_mutation_after_planning_is_detected(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(reg, hyps=[_hyp("H-1")], plans=[_tmpl("tsmom", "ES")], evidence=[_valid(0.2)], config=_config(target_size=1))
    manifest = orch.plan_family()
    manifest.members[0].strategy_spec.metadata["tampered"] = "yes"
    with pytest.raises(FamilyManifestError):
        orch.execute_family(manifest)


# --------------------------------------------------------------------------
# 18.1 proof: BH + final verdicts only at finalization
# --------------------------------------------------------------------------
def test_no_verdict_or_bh_written_before_finalization(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(
        reg,
        hyps=[_hyp("H-1"), _hyp("H-2", universe=("NQ",))],
        plans=[_tmpl("tsmom", "ES"), _tmpl("tsmom", "NQ")],
        evidence=[_valid(0.001), _valid(0.001)],
        config=_config(target_size=2),
    )
    manifest = orch.plan_family()
    evidence = orch.execute_family(manifest)

    assert reg.summary().total_experiment_rows == 0
    assert reg.summary().canonical_verdict_counts == {"PASS": 0, "REJECT": 0, "INCONCLUSIVE": 0}

    report = orch.finalize_family(manifest, evidence)
    assert report.status is FamilyStatus.FINALIZED
    for m in manifest.members:
        result = reg.get(m.experiment_identity).result
        assert result is not None and result.bh_q is not None and result.bh_p_value is not None


# --------------------------------------------------------------------------
# 18.1 proof: no partial family can produce a scientific PASS
# --------------------------------------------------------------------------
def test_incomplete_family_is_not_adjudicated_and_yields_no_pass(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(
        reg,
        hyps=[_hyp("H-1"), _hyp("H-2", universe=("NQ",))],
        plans=[_tmpl("tsmom", "ES"), _tmpl("tsmom", "NQ")],
        evidence=[_valid(0.0001), _invalid(), _invalid()],
        config=_config(target_size=2, max_execution_attempts_per_identity=2),
    )
    manifest = orch.plan_family()
    evidence = orch.execute_family(manifest)
    report = orch.finalize_family(manifest, evidence)

    assert report.status is FamilyStatus.INCOMPLETE_NOT_ADJUDICATED
    assert report.passing_experiment_ids == ()
    assert report.n_bh_rejected == 0
    assert manifest.members[1].experiment_identity in report.unresolved_invalid_identities
    assert reg.summary().canonical_verdict_counts == {"PASS": 0, "REJECT": 0, "INCONCLUSIVE": 0}
    with pytest.raises(orch_mod.UnknownExperiment):
        reg.get(manifest.members[0].experiment_identity)


def test_run_stops_after_an_incomplete_family(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(
        reg,
        hyps=[_hyp("H-1"), _hyp("H-2", universe=("NQ",))],
        plans=[_tmpl("tsmom", "ES"), _tmpl("tsmom", "NQ")],
        evidence=[_valid(0.2), _invalid(), _invalid()],
        config=_config(target_size=2, successor=True, max_execution_attempts_per_identity=2),
    )
    report = orch.run()
    assert report.n_generations == 1
    assert report.termination_reason == "family:incomplete_not_adjudicated"


# --------------------------------------------------------------------------
# 18.1 proof: INVALID_EXECUTION has no ResultRecord
# --------------------------------------------------------------------------
def test_invalid_execution_attempt_has_no_result_record(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(
        reg,
        hyps=[_hyp("H-1")],
        plans=[_tmpl("tsmom", "ES")],
        evidence=[_invalid(), _invalid()],
        config=_config(target_size=1, max_execution_attempts_per_identity=2),
    )
    manifest = orch.plan_family()
    evidence = orch.execute_family(manifest)
    orch.finalize_family(manifest, evidence)

    view = reg.get(manifest.members[0].experiment_identity)
    assert [a.attempt_status for a in view.attempts] == [
        AttemptStatus.INVALID_EXECUTION,
        AttemptStatus.INVALID_EXECUTION,
    ]
    assert all(a.result is None for a in view.attempts)
    assert view.result is None
    codes = {f.failure_class.value for f in reg.failures(experiment_identity=manifest.members[0].experiment_identity)}
    assert codes == {"INVALID_EXECUTION_ATTEMPT"}


# --------------------------------------------------------------------------
# 18.1 proof: invalid -> corrected VALID stays the same identity
# --------------------------------------------------------------------------
def test_invalid_then_corrected_valid_is_the_same_identity(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(
        reg,
        hyps=[_hyp("H-1")],
        plans=[_tmpl("tsmom", "ES")],
        evidence=[_invalid(), _valid(0.001)],
        config=_config(target_size=1, max_execution_attempts_per_identity=2),
    )
    manifest = orch.plan_family()
    identity = manifest.members[0].experiment_identity
    evidence = orch.execute_family(manifest)
    report = orch.finalize_family(manifest, evidence)

    view = reg.get(identity)
    assert view.experiment_identity == identity
    assert [(a.attempt_ordinal, a.attempt_status) for a in view.attempts] == [
        (1, AttemptStatus.INVALID_EXECUTION),
        (2, AttemptStatus.VALID),
    ]
    assert view.result is not None and view.result.headline_verdict is RegistryVerdict.PASS
    assert report.member_results[0].disposition is MemberDisposition.INVALID_THEN_CORRECTED
    edge = reg._conn.execute(
        "SELECT relation_type FROM attempt_lineage WHERE target_attempt_id = ?",
        (view.attempts[0].attempt_id,),
    ).fetchone()
    assert edge and edge["relation_type"] == "CORRECTS_ATTEMPT"


# --------------------------------------------------------------------------
# 18.2 proof: one evaluation plane, one outcome-adaptive campaign
# --------------------------------------------------------------------------
def test_successor_family_is_planned_not_executed_on_the_same_plane(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(
        reg,
        hyps=[_hyp("H-1", universe=("ES",)), _hyp("H-2", universe=("NQ",))],
        plans=[_tmpl("tsmom", "ES"), _tmpl("tsmom", "NQ")],
        evidence=[_valid(0.3), _valid(0.3)],
        config=_config(target_size=1, successor=True),
    )
    report = orch.run()

    g0, g1 = report.generations
    assert g0.status is FamilyStatus.FINALIZED
    assert g1.status is FamilyStatus.PLANNED_NOT_EXECUTED
    assert g1.parent_family_id == g0.family_id
    assert g1.family_id != g0.family_id
    assert report.termination_reason == "reflection:successor_family_planned_not_executed"
    # execution happened ONLY for generation 0
    assert orch._exec.call_count == 1
    assert reg.summary().authoritative_statistical_hypotheses == 1

    # and it cannot be executed on the same plane even by hand
    successor = orch.planned_family(g1.family_id)
    with pytest.raises(AdaptiveTestingError):
        orch.execute_family(successor)


def test_a_second_run_on_a_plane_the_registry_already_used_is_refused(tmp_path):
    reg = _registry(tmp_path)
    cfg = _config(target_size=1)
    first = _orch(reg, hyps=[_hyp("H-1")], plans=[_tmpl("tsmom", "ES")], evidence=[_valid(0.001)], config=cfg)
    first.run()

    second = _orch(reg, hyps=[_hyp("H-2", universe=("NQ",))], plans=[_tmpl("tsmom", "NQ")], evidence=[_valid(0.001)], config=cfg)
    manifest = second.plan_family()
    with pytest.raises(AdaptiveTestingError):
        second.execute_family(manifest)


# --------------------------------------------------------------------------
# 18.2 proof: adjudication is bound to the frozen ReliabilityPolicy
# --------------------------------------------------------------------------
def test_mismatched_policy_identity_is_refused_at_construction(tmp_path):
    reg = _registry(tmp_path)
    tweaked = POLICY.model_copy(update={"fdr_q_threshold": 0.20})
    assert tweaked.identity() != POLICY.identity()
    with pytest.raises(orch_mod.OrchestrationError):
        _orch(
            reg,
            hyps=[_hyp("H-1")],
            plans=[_tmpl("tsmom", "ES")],
            evidence=[_valid(0.2)],
            config=_config(target_size=1),  # planes bind POLICY.identity()
            policy=tweaked,
        )


def test_changing_fdr_q_changes_the_bound_policy_identity(tmp_path):
    reg = _registry(tmp_path)
    tweaked = POLICY.model_copy(update={"fdr_q_threshold": 0.20})
    planes = _planes(reliability_policy_fingerprint=tweaked.identity())
    orch = _orch(
        reg,
        hyps=[_hyp("H-1")],
        plans=[_tmpl("tsmom", "ES")],
        evidence=[_valid(0.15, policy=tweaked)],  # significant at q=0.20, not at q=0.10
        config=_config(target_size=1, planes=planes),
        policy=tweaked,
    )
    report = orch.run()
    fam = report.generations[0]
    assert fam.fdr_q_threshold == 0.20
    assert fam.policy_identity == tweaked.identity()
    assert fam.member_results[0].final_verdict is RegistryVerdict.PASS


def test_default_adjudicator_uses_the_shared_bh_and_the_policy_threshold():
    adj = PolicyFamilyAdjudicator(POLICY)
    assert adj.policy_identity() == POLICY.identity()
    assert adj.fdr_q_threshold() == POLICY.fdr_q_threshold


def test_evidence_produced_under_the_wrong_validation_fingerprint_is_refused(tmp_path):
    reg = _registry(tmp_path)
    bad = TrialEvidence(
        status=AttemptStatus.VALID,
        trial_p_value=0.01,
        non_family_verdict=NonFamilyVerdict.ELIGIBLE,
        engine="synthetic",
        produced_under_validation_spec_fingerprint="validationspec1:WRONG",
        produced_under_reliability_policy_fingerprint=POLICY.identity(),
    )
    orch = _orch(reg, hyps=[_hyp("H-1")], plans=[_tmpl("tsmom", "ES")], evidence=[bad], config=_config(target_size=1))
    manifest = orch.plan_family()
    with pytest.raises(EvidenceProvenanceError):
        orch.execute_family(manifest)


# --------------------------------------------------------------------------
# 18.2 proof: underfill is a typed state, never a smaller tested BH family
# --------------------------------------------------------------------------
def test_target_family_underfill_is_planning_incomplete_not_a_smaller_family(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(
        reg,
        hyps=[_hyp("H-1"), _hyp("H-1")],
        plans=[_tmpl("tsmom", "ES"), _tmpl("tsmom", "ES")],
        evidence=[_valid(0.001)],
        config=_config(target_size=3, max_planning_attempts_per_family=1),
    )
    report = orch.run()
    fam = report.generations[0]
    assert fam.status is FamilyStatus.PLANNING_INCOMPLETE
    assert fam.declared_target_size == 3
    assert fam.predeclared_family_size == 1
    assert report.termination_reason == "planning:underfilled_family"
    assert orch._exec.call_count == 0

    manifest = orch.plan_family()
    assert not manifest.planning_complete
    with pytest.raises(FamilyManifestError):
        orch.execute_family(manifest)
    assert reg.summary().total_experiment_rows == 0


# --------------------------------------------------------------------------
# preserved: PRE-RUN identity + fingerprint-is-evidence
# --------------------------------------------------------------------------
def test_member_identity_is_reproducible_from_configuration(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(reg, hyps=[_hyp("H-1")], plans=[_tmpl("tsmom", "ES")], evidence=[_valid(0.2)], config=_config(target_size=1))
    member = orch.plan_family().members[0]

    spec = spec_for_params("tsmom", {"root_symbol": "ES", "fast_horizon": 20, "slow_horizon": 120, "size": 1})
    rebuilt = experiment_identity(
        strategy_fingerprint=strategy_fingerprint(spec),
        strategy_family="tsmom",
        root_symbol="ES",
        parameter_variant_identity=parameter_variant_identity({"fast_horizon": 20, "slow_horizon": 120, "size": 1}),
        dataset_fingerprint="valdataset2:ds",
        split_identity="split1:sp",
        validation_spec_fingerprint=VSPEC,
        reliability_policy_fingerprint=POLICY.identity(),
        execution_config_identity="execconfig1:ex",
        cost_config_identity="costconfig1:co",
        risk_identity="riskconfig1:ri",
        feature_spec_fingerprint=feature_set_fingerprint(spec),
    )
    assert member.experiment_identity == rebuilt


def test_strategy_fingerprint_is_evidence_only_not_the_family_key(tmp_path):
    reg = _registry(tmp_path)
    spec = spec_for_params("tsmom", {"root_symbol": "ES", "fast_horizon": 20, "slow_horizon": 120, "size": 1})
    fp = strategy_fingerprint(spec)
    other = experiment_identity(
        strategy_fingerprint=fp,
        strategy_family="tsmom",
        root_symbol="ES",
        parameter_variant_identity=parameter_variant_identity({"fast_horizon": 20, "slow_horizon": 120, "size": 1}),
        dataset_fingerprint="valdataset2:ds",
        split_identity="split1:sp",
        validation_spec_fingerprint=VSPEC,
        reliability_policy_fingerprint=POLICY.identity(),
        execution_config_identity="execconfig1:DIFFERENT",
        cost_config_identity="costconfig1:co",
        risk_identity="riskconfig1:ri",
        feature_spec_fingerprint=feature_set_fingerprint(spec),
    )
    reg.insert_experiment(
        orch_mod.ExperimentRecord(
            experiment_identity=other,
            experiment_id="OTHER_PLANE",
            display_name="other plane",
            created_at="2026-01-01T00:00:00+00:00",
            phase="test",
            status=orch_mod.ExperimentStatus.COMPLETED,
            root_symbol="ES",
            asset_domain=AssetDomain.FUTURES,
            strategy_family="tsmom",
            strategy_fingerprint=fp,
            strategy_id=spec.strategy_id,
            strategy_spec_json={"params": {"fast_horizon": 20, "slow_horizon": 120, "size": 1}},
            feature_spec_fingerprint=feature_set_fingerprint(spec),
            dataset_fingerprint="valdataset2:ds",
            split_identity="split1:sp",
            market_window=MW,
            validation_spec_fingerprint=VSPEC,
            reliability_policy_fingerprint=POLICY.identity(),
            execution_config_identity="execconfig1:DIFFERENT",
            cost_config_identity="costconfig1:co",
            risk_identity="riskconfig1:ri",
            trial_role=TrialRole.CANONICAL,
            parameter_variant_identity=parameter_variant_identity({"fast_horizon": 20, "slow_horizon": 120, "size": 1}),
            parameter_variant_label="canonical",
        ),
        orch_mod.ResultRecord(experiment_identity=other, headline_verdict=RegistryVerdict.PASS),
    )
    orch = _orch(reg, hyps=[_hyp("H-1")], plans=[_tmpl("tsmom", "ES")], evidence=[_valid(0.2)], config=_config(target_size=1))
    manifest = orch.plan_family()
    assert len(manifest.members) == 1
    assert manifest.members[0].strategy_fingerprint == fp
    assert manifest.members[0].experiment_identity != other


def test_exact_identity_with_valid_authoritative_result_is_not_admitted(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(
        reg,
        hyps=[_hyp("H-1", universe=("ES",)), _hyp("H-2", universe=("ES",))],
        plans=[_tmpl("tsmom", "ES"), _tmpl("tsmom", "ES")],
        evidence=[_valid(0.5), _valid(0.5)],
        config=_config(target_size=1, successor=True, max_planning_attempts_per_family=1),
    )
    report = orch.run()
    g1 = report.generations[1]
    assert g1.status is FamilyStatus.EMPTY
    assert g1.planning_dispositions.get(PlanningDisposition.DUPLICATE_AUTHORITATIVE_RESULT.value) == 1


# --------------------------------------------------------------------------
# no access to 2025
# --------------------------------------------------------------------------
def test_config_with_2025_window_is_refused():
    with pytest.raises((HoldoutAccessError, ValidationError)):
        OrchestratorConfig(
            objective="x",
            market_universe=UNIVERSE,
            market_window=orch_mod.MarketWindow(label="BAD", start_date="2024-01-01", end_date="2025-06-01"),
            planes=_planes(),
            family_plan=FamilyPlanSpec(family_stem="f", target_family_size=1),
        )


def test_trial_evidence_with_2025_value_is_refused():
    with pytest.raises((HoldoutAccessError, ValidationError)):
        TrialEvidence(
            status=AttemptStatus.VALID,
            trial_p_value=0.1,
            non_family_verdict=NonFamilyVerdict.ELIGIBLE,
            non_family_reason_codes=("evaluated on the 2025-03-01 window",),
        )


def test_execution_is_never_called_with_holdout_true(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(
        reg,
        hyps=[_hyp("H-1"), _hyp("H-2", universe=("NQ",))],
        plans=[_tmpl("tsmom", "ES"), _tmpl("tsmom", "NQ")],
        evidence=[_valid(0.2), _valid(0.2)],
        config=_config(target_size=2),
    )
    orch.run()
    assert all(c["holdout"] is False for c in orch._exec.calls)


# --------------------------------------------------------------------------
# holdout isolation
# --------------------------------------------------------------------------
class _FakeHoldout:
    def __init__(self, ev: TrialEvidence):
        self._ev = ev
        self.calls: list[str] = []

    def evaluate_holdout(self, *, strategy_spec, experiment_identity, root_symbol) -> TrialEvidence:
        self.calls.append(experiment_identity)
        return self._ev


def _finalized_pass(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(reg, hyps=[_hyp("H-1")], plans=[_tmpl("tsmom", "ES")], evidence=[_valid(0.001)], config=_config(target_size=1))
    report = orch.run()
    ident = report.generations[0].member_results[0].experiment_identity
    spec = spec_for_params("tsmom", {"root_symbol": "ES", "fast_horizon": 20, "slow_horizon": 120, "size": 1})
    return orch, ident, spec


def test_run_holdout_requires_explicit_opt_in(tmp_path):
    orch, ident, spec = _finalized_pass(tmp_path)
    with pytest.raises(HoldoutIsolationError):
        orch.run_holdout(ident, strategy_spec=spec, holdout_service=_FakeHoldout(_valid(0.01)), allow_holdout=False)


def test_run_holdout_happy_path_is_guarded(tmp_path):
    orch, ident, spec = _finalized_pass(tmp_path)
    fake = _FakeHoldout(_valid(0.01))
    out = orch.run_holdout(ident, strategy_spec=spec, holdout_service=fake, allow_holdout=True)
    assert out.status is AttemptStatus.VALID
    assert fake.calls == [ident]


def test_execution_service_that_is_also_a_holdout_service_is_refused(tmp_path):
    reg = _registry(tmp_path)

    class _Both:
        def run(self, **kw):
            return _valid(0.2)

        def evaluate_holdout(self, **kw):
            return _valid(0.2)

    with pytest.raises(TypeError):
        ResearchOrchestrator(
            registry=reg,
            research_agent=ResearchAgent(ScriptedLLMClient([])),
            compiler_agent=StrategyCompilerAgent(ScriptedLLMClient([])),
            execution_service=_Both(),
            reliability_policy=POLICY,
            config=_config(),
        )


# --------------------------------------------------------------------------
# LLM authority boundary + determinism
# --------------------------------------------------------------------------
def test_planning_context_carries_no_threshold_fields(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(reg, hyps=[_hyp("H-1")], plans=[_tmpl()], evidence=[_valid(0.2)], config=_config(target_size=1))
    ctx = orch._planning_context()
    blob = json.dumps(ctx.model_dump(mode="json")).lower()
    for forbidden in ("threshold", "dsr_min", "fdr_q", "null_p_value", "commission", "slippage", "bh_q"):
        assert forbidden not in blob


def test_orchestrator_module_has_no_arbitrary_code_or_network_surface():
    src = inspect.getsource(orch_mod)
    for forbidden in ("subprocess", "eval(", "exec(", "requests", "socket", "databento", "urllib"):
        assert forbidden not in src


def test_family_plan_cannot_change_mid_run(tmp_path):
    reg = _registry(tmp_path)
    orch = _orch(
        reg,
        hyps=[_hyp("H-1"), _hyp("H-2", universe=("NQ",))],
        plans=[_tmpl("tsmom", "ES"), _tmpl("tsmom", "NQ")],
        evidence=[_valid(0.2), _valid(0.2)],
        config=_config(target_size=2),
    )
    orch._plan_fp = "tampered"
    with pytest.raises(orch_mod.OrchestrationError):
        orch.run()


def test_run_is_deterministic(tmp_path):
    def build(path):
        return _orch(
            ExperimentRegistry(path),
            hyps=[_hyp("H-1", universe=("ES",)), _hyp("H-2", universe=("NQ",))],
            plans=[_tmpl("tsmom", "ES"), _tmpl("tsmom", "NQ")],
            evidence=[_valid(0.001), _valid(0.5)],
            config=_config(target_size=2),
        )

    r1 = build(tmp_path / "a.sqlite").run()
    r2 = build(tmp_path / "b.sqlite").run()
    assert [g.family_id for g in r1.generations] == [g.family_id for g in r2.generations]
    assert r1.generations[0].verdict_counts == r2.generations[0].verdict_counts
    assert r1.passing_experiment_ids == r2.passing_experiment_ids
