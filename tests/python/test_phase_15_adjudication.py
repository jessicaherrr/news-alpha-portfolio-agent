"""Phase 15A.3 -- the frozen Phase 15 validation / adjudication planes.

Offline, deterministic, no market data, no model fitted, no 2025, no network.
The claims under test are the ones the plane exists to make:

* every reliability gate names the protocol clause or frozen Phase 13 default it
  came from, and none was chosen from ML performance (there is none to choose
  from);
* the ``ReliabilityPolicy`` is the Phase 13.5C one, unchanged, by fingerprint;
* the ``ValidationSpec`` preserves every frozen research count;
* the six identity planes are real -- no placeholder reaches a recorded value;
* the 60 predeclared trials produce 60 complete, distinct pre-run identities,
  equal in number to the BH/FDR denominator;
* the duplicate query is answerable before any fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import alpha_agent.features.compute  # noqa: F401 -- registers the feature kinds
import pytest
from alpha_agent.ml.adjudication import (
    CAPITAL_BASE_USD,
    PHASE_13_5C_RELIABILITY_POLICY_FINGERPRINT,
    PHASE_15_REASON_META_NOT_BEAT_PRIMARY,
    PHASE_15_REASON_ML_VALUE_ADD_CONFIRMED,
    PHASE_15_REASON_PLACEBO_NOT_SIGNIFICANT,
    PHASE_15_REASON_TYPED_REFUSAL,
    RELIABILITY_GATE_SOURCES,
    MLBaselineComparisonPlan,
    MLMultipleTestingPlan,
    MLPlaceboPlan,
    MLValidationSpec,
    adjudicate_phase_15_trial,
    assert_null_plan_fits_frozen_compute_budget,
    assert_phase_15_adjudication_rules_frozen,
    assert_protocol_and_policy_agree,
    assert_reliability_policy_gates_are_predeclared,
    assert_reliability_policy_inherited_unchanged,
    build_phase_15_validation_spec,
    multiple_testing_plan_from_manifest,
    phase_15_bh_family_p_values,
    phase_15_cost_stress_plan,
    phase_15_null_test_config,
    phase_15_reliability_policy,
    placebo_empirical_p,
)
from alpha_agent.ml.guards import HOLDOUT_START_NS, HoldoutAccessError
from alpha_agent.ml.manifest import build_candidate_manifest
from alpha_agent.ml.models import FROZEN_TAKE_THRESHOLD
from alpha_agent.ml.planes import (
    assert_planes_are_real,
    assert_sixty_complete_identities,
    ml_dataset_plane_fingerprint,
    phase_13_5c_dataset_fingerprints,
    phase_15_identity_planes,
    phase_15_pre_run_identity_rows,
    phase_15_registry_preflight,
)
from alpha_agent.ml.trials import MLIdentityPlanes, experiment_spec_for_trial
from alpha_agent.registry.enums import TrialRole
from alpha_agent.registry.phase_13_5c_import import Phase135cSources
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.validation.enums import NullMethod
from alpha_agent.validation.policy import ReliabilityPolicy

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "outputs" / "phase_15"


@pytest.fixture(scope="module")
def manifest():
    return build_candidate_manifest()


@pytest.fixture(scope="module")
def spec(manifest):
    return build_phase_15_validation_spec(manifest)


@pytest.fixture(scope="module")
def planes(manifest):
    return phase_15_identity_planes(REPO, manifest=manifest)


@pytest.fixture(scope="module")
def rows(planes, manifest):
    return phase_15_pre_run_identity_rows(planes=planes, manifest=manifest)


# --------------------------------------------------------------------------
# the ReliabilityPolicy plane -- inherited, not invented
# --------------------------------------------------------------------------
def test_every_reliability_gate_names_its_source():
    policy = phase_15_reliability_policy()
    sources = assert_reliability_policy_gates_are_predeclared(policy)
    assert set(sources) == set(ReliabilityPolicy.model_fields)
    for field, why in sources.items():
        assert why.startswith(("PROTOCOL:", "INHERITED:", "COSMETIC:")), field


def test_a_new_unsourced_gate_fails_closed(monkeypatch):
    """The provenance map is an allowlist: an unsourced gate must not slip in."""
    monkeypatch.setitem(RELIABILITY_GATE_SOURCES, "invented_gate", "PROTOCOL: nowhere")
    with pytest.raises(ValueError, match="stale sources"):
        assert_reliability_policy_gates_are_predeclared(phase_15_reliability_policy())


def test_reliability_policy_is_the_phase_13_5c_one_unchanged():
    policy = phase_15_reliability_policy()
    assert (
        assert_reliability_policy_inherited_unchanged(policy)
        == PHASE_13_5C_RELIABILITY_POLICY_FINGERPRINT
    )


def test_the_committed_phase_13_5c_reports_carry_that_same_policy_fingerprint():
    """Not a constant copied into a test -- read back from the frozen artifacts."""
    seen = set()
    for path in sorted(OUT_DIR.parent.glob("phase_13_5c/*__validation_report.json")):
        seen.add(json.loads(path.read_text())["fingerprints"]["reliability_policy_fingerprint"])
    assert seen == {PHASE_13_5C_RELIABILITY_POLICY_FINGERPRINT}


def test_a_tuned_gate_changes_the_policy_fingerprint_and_is_refused():
    """The failure mode this guard exists for: a gate loosened after a result."""
    loosened = ReliabilityPolicy(policy_name="phase_15_ml", dsr_min=0.5)
    with pytest.raises(ValueError, match="replaces the ReliabilityPolicy with nothing"):
        assert_reliability_policy_inherited_unchanged(loosened)


def test_protocol_bh_q_and_inherited_gate_agree(manifest):
    policy = phase_15_reliability_policy()
    assert_protocol_and_policy_agree(policy, manifest)
    assert policy.fdr_q_threshold == manifest.multiple_testing_family["bh_q"] == 0.10


def test_a_protocol_policy_disagreement_raises_rather_than_picking_one(manifest):
    conflicting = manifest.model_copy(
        update={"multiple_testing_family": {**manifest.multiple_testing_family, "bh_q": 0.05}}
    )
    with pytest.raises(ValueError, match="resolved by a human"):
        assert_protocol_and_policy_agree(phase_15_reliability_policy(), conflicting)


# --------------------------------------------------------------------------
# the ValidationSpec plane
# --------------------------------------------------------------------------
def test_validation_spec_preserves_every_frozen_research_count(spec, manifest):
    mt = spec.multiple_testing
    est = manifest.compute_estimate
    assert (mt.n_trials, mt.n_headline_trials, mt.n_ablation_trials) == (60, 40, 20)
    assert mt.bh_q == 0.10
    assert mt.dsr_effective_trial_count == 380
    assert dict(mt.dsr_effective_trial_count_by_model_family) == {
        "LOGISTIC_L2": 3,
        "HIST_GRADIENT_BOOSTING": 8,
    }
    assert est.n_model_fits_total == 1200
    assert est.n_cpp_evaluations_total_excluding_placebo == 240
    assert spec.take_threshold == FROZEN_TAKE_THRESHOLD == 0.50
    assert (spec.nested_cv.min_train_events, spec.nested_cv.min_test_events) == (100, 20)
    assert spec.capital_base_usd == CAPITAL_BASE_USD


def test_phase_15_planes_are_inherited_from_phase_13_5c_by_identity():
    """Cost plane and policy are literally Phase 13.5C's objects, not lookalikes."""
    from alpha_agent.validation.phase_13_5c_matrix import (
        CAPITAL_BASE_USD as P135C_CAPITAL,
    )
    from alpha_agent.validation.phase_13_5c_matrix import (
        frozen_cost_plan,
        frozen_policy,
    )

    assert phase_15_cost_stress_plan().identity() == frozen_cost_plan().identity()
    assert phase_15_reliability_policy().identity() == frozen_policy().identity()
    assert CAPITAL_BASE_USD == P135C_CAPITAL


def test_null_plan_is_the_gating_bootstrap_only(spec, manifest):
    assert spec.null_test.methods == (NullMethod.CENTERED_BLOCK_BOOTSTRAP,)
    assert spec.null_test.identity() == phase_15_null_test_config().identity()
    assert_null_plan_fits_frozen_compute_budget(spec.null_test, manifest)


def test_multiple_testing_plan_is_derived_from_the_frozen_manifest(spec, manifest):
    """Derived, never hand-typed: the plan and the manifest cannot drift apart."""
    assert multiple_testing_plan_from_manifest(manifest).identity() == (
        spec.multiple_testing.identity()
    )


def test_a_cpp_rerunning_null_is_refused_against_the_frozen_budget(manifest):
    from alpha_agent.validation.nulls import NullTestConfig

    greedy = NullTestConfig(
        methods=(NullMethod.SCHEDULE_TIME_SHIFT, NullMethod.CENTERED_BLOCK_BOOTSTRAP)
    )
    with pytest.raises(ValueError, match="240 C\\+\\+ evaluations"):
        assert_null_plan_fits_frozen_compute_budget(greedy, manifest)


def test_dropping_the_gating_null_is_refused(manifest):
    from alpha_agent.validation.nulls import NullTestConfig

    with pytest.raises(ValueError, match="gating null"):
        assert_null_plan_fits_frozen_compute_budget(NullTestConfig(methods=()), manifest)


def test_spec_refuses_a_moved_take_threshold(spec):
    with pytest.raises(ValueError, match="FROZEN at 0.5"):
        MLValidationSpec.model_validate({**spec.model_dump(), "take_threshold": 0.55})


def test_spec_refuses_a_lowered_event_gate(spec):
    payload = spec.model_dump()
    payload["nested_cv"] = {**payload["nested_cv"], "min_train_events": 25}
    with pytest.raises(ValueError, match="FROZEN at min_train_events=100"):
        MLValidationSpec.model_validate(payload)


def test_spec_refuses_to_gate_on_ml_diagnostics(spec):
    with pytest.raises(ValueError, match="diagnostic only"):
        MLValidationSpec.model_validate({**spec.model_dump(), "ml_diagnostics_gate_verdict": True})


def test_spec_refuses_an_invented_plateau_gate(spec):
    with pytest.raises(ValueError, match="own predeclared threshold"):
        MLValidationSpec.model_validate(
            {**spec.model_dump(), "parameter_stability_gate_applies": True}
        )


def test_spec_refuses_a_gated_delta_vs_primary():
    with pytest.raises(ValueError, match="invented rule"):
        MLBaselineComparisonPlan(delta_vs_primary_is_gated=True)


def test_spec_refuses_a_holdout_touching_corpus(spec):
    payload = spec.model_dump()
    payload["development_corpus_end_ts_ns"] = HOLDOUT_START_NS + 1
    with pytest.raises((HoldoutAccessError, ValueError)):
        MLValidationSpec.model_validate(payload)
    payload["development_corpus_end_ts_ns"] = HOLDOUT_START_NS
    payload["locked_final_holdout_accessed"] = True
    with pytest.raises((HoldoutAccessError, ValueError)):
        MLValidationSpec.model_validate(payload)


def test_a_null_never_enters_the_bh_denominator():
    with pytest.raises(ValueError, match="never competes for significance"):
        MLPlaceboPlan(in_bh_denominator=True)
    with pytest.raises(ValueError, match="never enter this denominator"):
        MLMultipleTestingPlan(
            family_id="phase_15_ml.all",
            n_trials=60,
            n_headline_trials=40,
            n_ablation_trials=20,
            bh_q=0.10,
            dsr_effective_trial_count=380,
            dsr_effective_trial_count_by_model_family={"LOGISTIC_L2": 3},
            placebo_in_denominator=True,
        )


def test_phase_13_5c_trials_are_never_folded_in(spec):
    assert spec.multiple_testing.family_id == "phase_15_ml.all"
    assert spec.multiple_testing.phase_13_5c_trials_folded_in is False
    assert spec.multiple_testing.phase_13_5c_n_trials_unchanged == 107


def test_validation_fingerprint_is_deterministic_and_label_is_cosmetic(spec):
    assert spec.validation_fingerprint() == build_phase_15_validation_spec().validation_fingerprint()
    renamed = MLValidationSpec.model_validate({**spec.model_dump(), "label": "something else"})
    assert renamed.validation_fingerprint() == spec.validation_fingerprint()


@pytest.mark.parametrize(
    "field,value",
    [
        ("capital_base_usd", 250_000.0),
        ("fold_evidence_source", "phase_13_walk_forward"),
        ("hyperparameter_plateau_evidence", "economic_per_configuration"),
    ],
)
def test_a_changed_validation_semantic_changes_the_fingerprint(spec, field, value):
    moved = MLValidationSpec.model_validate({**spec.model_dump(), field: value})
    assert moved.validation_fingerprint() != spec.validation_fingerprint()


def test_a_changed_multiple_testing_plan_changes_the_validation_fingerprint(spec):
    loosened = spec.multiple_testing.model_copy(update={"bh_q": 0.20})
    moved = spec.model_copy(update={"multiple_testing": loosened})
    assert moved.validation_fingerprint() != spec.validation_fingerprint()


# --------------------------------------------------------------------------
# the six identity planes
# --------------------------------------------------------------------------
def test_planes_are_real_and_none_is_a_placeholder(planes):
    assert_planes_are_real(planes)
    assert planes.validation_spec_fingerprint.startswith("mlvalidationspec1:")
    assert planes.reliability_policy_fingerprint == PHASE_13_5C_RELIABILITY_POLICY_FINGERPRINT
    assert planes.dataset_fingerprint.startswith("mldatasetplane1:")
    assert planes.execution_config_identity.startswith("execconfig1:")
    assert planes.cost_config_identity.startswith("costconfig1:")
    assert planes.risk_identity.startswith("riskconfig1:")


def test_a_pending_plane_is_refused():
    pending = MLIdentityPlanes(
        dataset_fingerprint="PENDING_PHASE_15B",
        validation_spec_fingerprint="PENDING_PHASE_15B",
        reliability_policy_fingerprint="PENDING_PHASE_15B",
        execution_config_identity="PENDING_PHASE_15B",
        cost_config_identity="PENDING_PHASE_15B",
        risk_identity="PENDING_PHASE_15B",
    )
    with pytest.raises(ValueError, match="placeholder"):
        assert_planes_are_real(pending)


def test_dataset_plane_pools_the_committed_per_root_phase_13_5c_identities():
    sources = Phase135cSources(REPO)
    per_root = phase_13_5c_dataset_fingerprints(sources)
    assert sorted(per_root) == ["CL", "ES", "GC", "NQ", "ZN"]
    for root, fp in per_root.items():
        assert fp.startswith("valdataset2:"), root
    assert ml_dataset_plane_fingerprint(per_root) == ml_dataset_plane_fingerprint(per_root)
    moved = {**per_root, "ES": "valdataset2:" + "0" * 64}
    assert ml_dataset_plane_fingerprint(moved) != ml_dataset_plane_fingerprint(per_root)


def test_dataset_plane_needs_every_training_root():
    sources = Phase135cSources(REPO)
    per_root = phase_13_5c_dataset_fingerprints(sources)
    del per_root["ZN"]
    with pytest.raises(ValueError, match="training roots"):
        ml_dataset_plane_fingerprint(per_root)


# --------------------------------------------------------------------------
# 60 predeclared hypotheses -> 60 complete pre-run identities
# --------------------------------------------------------------------------
def test_sixty_trials_produce_sixty_complete_distinct_identities(rows, spec):
    assert_sixty_complete_identities(rows, expected=60)
    assert len(rows) == spec.multiple_testing.n_trials == 60
    assert len({r["experiment_identity"] for r in rows}) == 60


def test_post_run_provenance_is_null_before_the_run(rows):
    assert all(r["target_schedule_hash"] is None for r in rows)
    assert all(r["report_fingerprint"] is None for r in rows)


def test_identity_still_carries_the_search_not_the_selected_point(rows, manifest):
    """The Phase 15A.2 invariant, re-checked with the planes filled in."""
    grid_points = {
        json.dumps(h, sort_keys=True)
        for points in manifest.hyperparameter_grid.values()
        for h in points
    }
    blob = json.dumps(list(rows), sort_keys=True)
    assert grid_points, "the frozen grid must be non-empty for this test to mean anything"
    for point in grid_points:
        assert point not in blob
    for r in rows:
        assert "SEARCH" in r["parameter_variant_label"]


def test_identity_moves_when_a_plane_moves(planes, manifest):
    trial = manifest.trials[0]
    base = experiment_spec_for_trial(trial, planes=planes).experiment_identity()
    for field in (
        "dataset_fingerprint",
        "validation_spec_fingerprint",
        "reliability_policy_fingerprint",
        "execution_config_identity",
        "cost_config_identity",
        "risk_identity",
    ):
        moved = planes.model_copy(update={field: "moved1:" + "a" * 64})
        assert experiment_spec_for_trial(trial, planes=moved).experiment_identity() != base, field


def test_identity_is_computable_from_configuration_alone(planes, manifest):
    """No fit, no schedule, no market data -- the whole point of a pre-run identity."""
    trial = manifest.trials[0]
    spec = experiment_spec_for_trial(trial, planes=planes)
    assert spec.experiment_identity() == experiment_spec_for_trial(
        trial, planes=planes
    ).experiment_identity()
    assert not hasattr(spec, "selected_hyperparameters")


# --------------------------------------------------------------------------
# the mandatory pre-run registry query
# --------------------------------------------------------------------------
def test_registry_preflight_answers_from_the_proposal_alone(rows):
    """The preflight query is a pure read of the registry from the pre-run
    identities. Schema v5: the a31f571 run is an INVALID_EXECUTION attempt 1
    (feature-pipeline defect); the corrected cc206de/b92555d run is the VALID
    authoritative attempt 2 (all 60 typed REFUSED -- a legitimate
    insufficient-events outcome). A VALID authoritative result blocks a rerun."""
    registry = ExperimentRegistry(REPO / "data" / "registry" / "experiments.sqlite")
    result = phase_15_registry_preflight(registry, rows)
    assert result["n_trials_queried"] == 60
    assert result["n_exact_duplicates"] == 60          # all 60 identities are present
    assert result["n_blocking_duplicates"] == 60       # each has a VALID authoritative result
    assert result["n_re_executable_invalid_only"] == 0
    assert result["clear_to_run"] is False
    assert all(t["is_exact_duplicate"] is True for t in result["trials"])
    assert all(t["has_valid_authoritative_result"] is True for t in result["trials"])
    assert all(t["n_prior_attempts"] == 2 for t in result["trials"])
    assert all(t["n_invalid_prior_attempts"] == 1 for t in result["trials"])


def test_preflight_detects_a_duplicate(tmp_path, rows):
    """A registry that already holds one of these identities must refuse the rerun."""

    class _FakeDup:
        exists = True
        experiment_id = "PRIOR"

    class _FakeMiss:
        exists = False
        experiment_id = None

    target = rows[3]["experiment_identity"]

    class _FakeRegistry:
        def find_exact_duplicate(self, identity, **_kw):
            return _FakeDup() if identity == target else _FakeMiss()

        def find_related(self, **_kw):
            return ()

        def count_experiments(self):
            return 1

    result = phase_15_registry_preflight(_FakeRegistry(), rows)
    assert result["n_exact_duplicates"] == 1
    assert result["duplicate_trial_labels"] == [rows[3]["trial_label"]]
    assert result["clear_to_run"] is False


# --------------------------------------------------------------------------
# the committed artifacts
# --------------------------------------------------------------------------
CORRECTED_PLANE = (
    REPO / "data" / "manifests" / "phase_15" / "ml_validation_plane_adjudication_corrected.json"
)
#: The 15B.0 adjudication-corrected map is preserved as history; the AUTHORITATIVE
#: pre-run identity map is now the 15B.1b regime-corrected one.
ADJUDICATION_CORRECTED_IDENTITY_MAP = (
    OUT_DIR / "PRERUN_IDENTITY_MAP_COMPLETE_ADJUDICATION_CORRECTED.json"
)
CORRECTED_IDENTITY_MAP = OUT_DIR / "PRERUN_IDENTITY_MAP_COMPLETE_REGIME_CORRECTED.json"


def test_committed_validation_plane_matches_what_the_code_builds(spec, planes):
    """The AUTHORITATIVE plane is the Phase 15B adjudication-corrected one."""
    record = json.loads(CORRECTED_PLANE.read_text())
    assert record["validation_spec_fingerprint"] == spec.validation_fingerprint()
    assert record["reliability_policy_fingerprint"] == planes.reliability_policy_fingerprint
    assert record["identity_planes"]["planes"] == {
        "dataset_fingerprint": planes.dataset_fingerprint,
        "validation_spec_fingerprint": planes.validation_spec_fingerprint,
        "reliability_policy_fingerprint": planes.reliability_policy_fingerprint,
        "execution_config_identity": planes.execution_config_identity,
        "cost_config_identity": planes.cost_config_identity,
        "risk_identity": planes.risk_identity,
    }
    assert record["no_real_model_trained"] is True
    assert record["phase_15_adjudication_rules"]["core_phase_13_reliability_policy_modified"] is False


def test_phase_15a3_validation_plane_is_preserved_unchanged():
    """15B supersedes; it never edits the 15A.3 artifact in place."""
    prior = json.loads(
        (REPO / "data" / "manifests" / "phase_15" / "ml_validation_plane.json").read_text()
    )
    assert prior["phase"] == "15A.3"
    assert prior["validation_spec_fingerprint"].startswith("mlvalidationspec1:")
    assert prior["validation_spec_fingerprint"] != build_phase_15_validation_spec().validation_fingerprint()
    superseded = json.loads((OUT_DIR / "ADJUDICATION_PLANE_SUPERSESSION.json").read_text())
    assert superseded["superseded"]["validation_spec_fingerprint"] == (
        prior["validation_spec_fingerprint"]
    )
    assert superseded["identity_delta"]["n_identities_changed"] == 60
    assert superseded["identity_delta"]["n_labels_changed"] == 0


def test_committed_identity_map_holds_sixty_complete_identities(rows):
    record = json.loads(CORRECTED_IDENTITY_MAP.read_text())
    assert record["phase"] == "15B.1b"
    assert record["n_predeclared_hypotheses"] == 60
    assert record["n_complete_pre_run_identities"] == 60
    assert record["bh_fdr_denominator"] == 60
    committed = {t["trial_label"]: t["experiment_identity"] for t in record["trials"]}
    assert committed == {r["trial_label"]: r["experiment_identity"] for r in rows}


def test_the_adjudication_corrected_identity_map_is_preserved_as_history(rows):
    """15B.1b supersedes the 15B.0 map -- it does not edit it in place. Exactly
    the 40 regime-baseline identities move; the 20 NO_REGIME rows are unchanged."""
    from alpha_agent.ml.manifest import SUPERSEDED_15A2_MANIFEST_FINGERPRINT

    prior = json.loads(ADJUDICATION_CORRECTED_IDENTITY_MAP.read_text())
    assert prior["manifest_fingerprint"] == SUPERSEDED_15A2_MANIFEST_FINGERPRINT
    prior_ids = {t["trial_label"]: t["experiment_identity"] for t in prior["trials"]}
    now_ids = {r["trial_label"]: r["experiment_identity"] for r in rows}
    changed = {lbl for lbl in prior_ids if prior_ids[lbl] != now_ids[lbl]}
    unchanged = set(prior_ids) - changed
    assert len(changed) == 40
    assert len(unchanged) == 20
    regime = json.loads(CORRECTED_IDENTITY_MAP.read_text())
    assert set(regime["regime_baseline_trial_labels_regenerated"]) == changed
    assert set(regime["no_regime_ablation_trial_labels_unchanged"]) == unchanged


def test_committed_freeze_preserves_the_research_counts():
    record = json.loads((OUT_DIR / "PROTOCOL_FREEZE_ADJUDICATION_PLANE.json").read_text())
    assert record["frozen_counts"]["n_ml_hypotheses_bh_denominator"] == 60
    assert record["frozen_counts"]["n_inspected_configurations_dsr"] == 380
    assert record["frozen_counts"]["n_model_fits_total"] == 1200
    assert record["frozen_counts"]["n_cpp_evaluations_total_excluding_placebo"] == 240
    assert record["dsr_effective_trial_count_by_model_family"] == {
        "LOGISTIC_L2": 3,
        "HIST_GRADIENT_BOOSTING": 8,
    }
    assert record["take_threshold"] == 0.50
    assert record["event_gates"] == {"min_train_events": 100, "min_test_events": 20}
    assert record["adjudication_rules_chosen_from_ml_performance"] is False
    assert record["core_phase_13_reliability_policy_modified"] is False
    assert record["no_ml_performance_inspected"] is True


def test_committed_15a3_freeze_still_reads_15a3():
    """The Phase 15A.3 freeze artifact is untouched -- supersession is additive."""
    record = json.loads((OUT_DIR / "PROTOCOL_FREEZE_VALIDATION_PLANE.json").read_text())
    assert record["phase"] == "15A.3"


# --------------------------------------------------------------------------
# Phase 15B adjudication rule 1 -- the BH family is ALWAYS exactly 60
# --------------------------------------------------------------------------
def test_bh_family_size_is_fixed_at_sixty(spec):
    mt = spec.multiple_testing
    assert mt.family_size_is_fixed is True
    assert mt.n_trials == 60
    assert mt.refused_trial_bh_p_value == 1.0
    assert_phase_15_adjudication_rules_frozen(spec)


def test_a_refused_trial_stays_in_the_family_at_conservative_p(spec):
    labels = tuple(f"t{i}" for i in range(60))
    p_by = {label: 0.01 for label in labels[:58]}
    refused = {labels[58]: "INSUFFICIENT_TRAIN_EVENTS", labels[59]: "DATA_QUALITY_FAILURE"}
    rows = phase_15_bh_family_p_values(
        labels, p_value_by_trial=p_by, typed_refusal_by_trial=refused,
        plan=spec.multiple_testing,
    )
    assert len(rows) == 60
    assert rows[58] == ("t58", 1.0, "INSUFFICIENT_TRAIN_EVENTS")
    assert rows[59] == ("t59", 1.0, "DATA_QUALITY_FAILURE")
    assert all(r[2] is None for r in rows[:58])


def test_a_short_family_is_refused(spec):
    with pytest.raises(ValueError, match="fixed at 60"):
        phase_15_bh_family_p_values(
            tuple(f"t{i}" for i in range(59)),
            p_value_by_trial={}, typed_refusal_by_trial={f"t{i}": "R" for i in range(59)},
            plan=spec.multiple_testing,
        )


def test_a_trial_cannot_be_both_refused_and_significant(spec):
    labels = tuple(f"t{i}" for i in range(60))
    with pytest.raises(ValueError, match="also carries a statistical p-value"):
        phase_15_bh_family_p_values(
            labels,
            p_value_by_trial={"t0": 0.01},
            typed_refusal_by_trial={**{"t0": "R"}, **{labels[i]: "R" for i in range(1, 60)}},
            plan=spec.multiple_testing,
        )


def test_multiple_testing_plan_refuses_a_non_fixed_family():
    with pytest.raises(ValueError, match="FIXED"):
        MLMultipleTestingPlan(
            family_id="phase_15_ml.all", n_trials=60, n_headline_trials=40,
            n_ablation_trials=20, bh_q=0.10, dsr_effective_trial_count=380,
            dsr_effective_trial_count_by_model_family={"LOGISTIC_L2": 3},
            family_size_is_fixed=False,
        )


# --------------------------------------------------------------------------
# Phase 15B adjudication rule 2 -- the ML-value-add PASS gate
# --------------------------------------------------------------------------
def test_placebo_empirical_p_formula():
    assert placebo_empirical_p(2.0, tuple([0.0] * 20), draws=20) == pytest.approx(1 / 21)
    assert placebo_empirical_p(0.0, tuple([1.0] * 20), draws=20) == pytest.approx(21 / 21)
    assert placebo_empirical_p(0.5, tuple([1.0] * 3 + [0.0] * 17), draws=20) == pytest.approx(4 / 21)
    with pytest.raises(ValueError, match="20-draw"):
        placebo_empirical_p(0.0, tuple([0.0] * 19), draws=20)


def test_spec_refuses_to_disable_the_meta_beats_primary_gate(spec):
    with pytest.raises(ValueError, match="must demonstrate the ML"):
        MLValidationSpec.model_validate(
            {**spec.model_dump(), "require_meta_beats_primary_daily_sharpe": False}
        )


def test_spec_refuses_a_loosened_placebo_threshold(spec):
    with pytest.raises(ValueError, match="frozen at p <= 0.05"):
        MLValidationSpec.model_validate({**spec.model_dump(), "placebo_empirical_p_max": 0.10})
    with pytest.raises(ValueError, match="frozen at p <= 0.05"):
        MLPlaceboPlan(empirical_p_max=0.10)


def test_placebo_is_never_a_bh_hypothesis(spec):
    assert spec.placebo.in_bh_denominator is False
    assert spec.multiple_testing.placebo_in_denominator is False


def _pass_outcome():
    from alpha_agent.validation.enums import ReasonCode, Verdict
    from alpha_agent.validation.policy import PolicyOutcome

    return PolicyOutcome(
        verdict=Verdict.PASS,
        reason_codes=(ReasonCode.ALL_GATES_SATISFIED,),
        policy_fingerprint="x",
        gate_results={},
    )


def _reject_outcome():
    from alpha_agent.validation.enums import ReasonCode, Verdict
    from alpha_agent.validation.policy import PolicyOutcome

    return PolicyOutcome(
        verdict=Verdict.REJECT,
        reason_codes=(ReasonCode.DSR_BELOW_THRESHOLD,),
        policy_fingerprint="x",
        gate_results={},
    )


_TRIAL_KW = dict(
    trial_label="tsmom__NQ__LOGISTIC_L2",
    root_symbol="NQ",
    primary_family="tsmom",
    model_family="LOGISTIC_L2",
    regime_kind="DETERMINISTIC_CAUSAL_VOL_TREND",
    trial_role="HEADLINE",
)


def test_base_pass_kept_only_when_ml_adds_value(spec):
    a = adjudicate_phase_15_trial(
        spec, **_TRIAL_KW, bh_p_value=0.001, base_outcome=_pass_outcome(),
        meta_daily_sharpe=0.9, primary_daily_sharpe=0.5,
        placebo_observed_statistic=0.9, placebo_statistics=tuple([0.1] * 20),
    )
    assert a.verdict == "PASS"
    assert PHASE_15_REASON_ML_VALUE_ADD_CONFIRMED in a.reason_codes
    assert a.meta_beats_primary is True
    assert a.placebo_significant is True


def test_base_pass_rejected_when_meta_does_not_beat_primary(spec):
    a = adjudicate_phase_15_trial(
        spec, **_TRIAL_KW, bh_p_value=0.001, base_outcome=_pass_outcome(),
        meta_daily_sharpe=0.4, primary_daily_sharpe=0.5,
        placebo_observed_statistic=0.4, placebo_statistics=tuple([0.1] * 20),
    )
    assert a.verdict == "REJECT"
    assert PHASE_15_REASON_META_NOT_BEAT_PRIMARY in a.reason_codes


def test_base_pass_rejected_when_placebo_not_significant(spec):
    a = adjudicate_phase_15_trial(
        spec, **_TRIAL_KW, bh_p_value=0.001, base_outcome=_pass_outcome(),
        meta_daily_sharpe=0.9, primary_daily_sharpe=0.5,
        placebo_observed_statistic=0.2, placebo_statistics=tuple([0.9] * 20),
    )
    assert a.verdict == "REJECT"
    assert PHASE_15_REASON_PLACEBO_NOT_SIGNIFICANT in a.reason_codes


def test_base_reject_is_carried_through(spec):
    a = adjudicate_phase_15_trial(
        spec, **_TRIAL_KW, bh_p_value=0.4, base_outcome=_reject_outcome(),
    )
    assert a.verdict == "REJECT"
    assert a.base_reliability_verdict == "REJECT"


def test_a_typed_refusal_never_leaves_the_family(spec):
    a = adjudicate_phase_15_trial(
        spec, **_TRIAL_KW, bh_p_value=0.123, base_outcome=None,
        typed_refusal="INSUFFICIENT_TRAIN_EVENTS",
    )
    assert a.verdict == "REFUSED"
    assert a.bh_p_value == 1.0
    assert PHASE_15_REASON_TYPED_REFUSAL in a.reason_codes
    assert "INSUFFICIENT_TRAIN_EVENTS" in a.reason_codes


def test_config_mirror_declares_the_adjudication_rules():
    import yaml

    cfg = yaml.safe_load((REPO / "configs" / "phase_15_ml.yaml").read_text())
    rules = cfg["validation_plane"]["phase_15b_adjudication_rules"]
    assert rules["bh_family_size"] == 60
    assert rules["refused_trial_bh_p_value"] == 1.0
    assert rules["require_meta_beats_primary_daily_sharpe"] is True
    assert rules["placebo_empirical_p_max"] == 0.05
    assert rules["placebo_draws"] == 20


def test_earlier_phase_15_artifacts_are_untouched():
    """15A.3 adds artifacts; supersession is never an in-place edit."""
    prior = json.loads((OUT_DIR / "PRETRAIN_IDENTITY_MAP.json").read_text())
    assert prior["phase"] == "15A.2"
    assert prior["n_distinct_pre_run_identities"] == 60
    assert "planes_not_recorded_here" in prior


def test_phase_13_5c_registry_evidence_is_unchanged():
    """Scoped to the Phase 13.5C subset itself (`experiment.phase ==
    "13.5C"`), never to the registry's global total. The production registry
    is append-only and legitimately grows over time (schema v5's Phase 15
    import, later real orchestrator runs such as the Research Golden Path V1
    acceptance pass's own new tsmom/NQ (30,150) experiment) -- this test's
    actual claim is that schema v5 / an INVALID_EXECUTION attempt / any later
    unrelated append never rewrites a single Phase 13.5C row, which a
    global-total assertion cannot distinguish from "nothing else was ever
    added to the registry" and so breaks on every legitimate later append."""
    registry = ExperimentRegistry(REPO / "data" / "registry" / "experiments.sqlite")
    summary = registry.summary()
    assert summary.schema_version == 7
    auth = list(registry.experiments(authoritative_only=True))
    phase_13_5c = [v for v in auth if v.experiment.phase == "13.5C"]
    # 107 Phase 13.5C authoritative hypotheses -- unchanged by schema v5 and
    # by an INVALID_EXECUTION attempt; only a VALID attempt supplies a result.
    assert len(phase_13_5c) == 107
    assert summary.superseded_experiments == 21
    # the 13.5C adjudicated verdicts are intact and untouched by v5.
    canonical_13_5c = [v for v in phase_13_5c if v.experiment.trial_role is TrialRole.CANONICAL]
    verdict_counts: dict[str, int] = {}
    for v in canonical_13_5c:
        key = v.verdict.value if v.verdict is not None else "NOT_ADJUDICATED"
        verdict_counts[key] = verdict_counts.get(key, 0) + 1
    assert verdict_counts.get("PASS", 0) == 0
    assert verdict_counts.get("REJECT", 0) == 19
    assert verdict_counts.get("INCONCLUSIVE", 0) == 2


def test_config_mirror_matches_the_frozen_planes(planes):
    """`configs/phase_15_ml.yaml` is a mirror; a stale mirror is a lie, not a typo."""
    import yaml

    cfg = yaml.safe_load((REPO / "configs" / "phase_15_ml.yaml").read_text())
    declared = cfg["identity_planes"]
    for field in (
        "dataset_fingerprint",
        "validation_spec_fingerprint",
        "reliability_policy_fingerprint",
        "execution_config_identity",
        "cost_config_identity",
        "risk_identity",
    ):
        assert declared[field]["value"] == getattr(planes, field), field


def test_config_mirror_matches_the_inherited_reliability_gates():
    import yaml

    cfg = yaml.safe_load((REPO / "configs" / "phase_15_ml.yaml").read_text())
    declared = cfg["validation_plane"]["reliability_policy"]
    policy = phase_15_reliability_policy()
    for gate in (
        "fdr_q_threshold",
        "null_p_value_max",
        "dsr_min",
        "max_cost_degradation",
        "fold_consistency_min",
        "require_positive_oos_net_pnl",
        "require_regime_stability",
        "require_cross_market",
        "require_ablation_mechanism_value",
    ):
        assert declared[gate] == getattr(policy, gate), gate
    assert declared["minimum_sample"] == policy.minimum_sample.model_dump(mode="json")
