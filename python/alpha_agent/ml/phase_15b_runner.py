"""The Phase 15B orchestrator -- the whole DAG behind one entry point.

frozen Phase 13.5C primary schedules
  -> C++ trades/fills export
  -> episode extraction + exact economics reconciliation
  -> MetaLabelEvent corpus
  -> causal feature matrix at decision timestamp
  -> pooled five-root dataset
  -> global chronological nested-CV folds
  -> fold-local preprocessing / regime fitting
  -> Logistic L2 / HistGradientBoosting frozen searches
  -> OOF predictions
  -> TAKE/SKIP schedules
  -> exactly aligned PRIMARY vs META support
  -> C++ economics at 1.0x / 1.5x / 2.0x
  -> daily return traces
  -> centred-block gating null
  -> fixed 60-trial BH family
  -> DSR with frozen inspected-configuration accounting
  -> fold consistency / cost gates
  -> Phase-15 add-value gate
  -> conditional matched-TAKE-rate placebo
  -> final PASS / REJECT / INCONCLUSIVE
  -> immutable registry records
  -> human-readable report

Modes:

* ``preflight`` -- rebuild the 60 pre-run identities and query the registry.
  No fit, no market data.
* ``synthetic_integration`` -- run the WHOLE DAG on deterministic synthetic
  fixtures (no market data, no real fit, no production registry write).
* ``estimate_only`` -- derive the real-run counts from the frozen manifest.
* ``run_real`` -- the production run. NOT invoked in Phase 15B.1. Requires an
  explicit ``allow_real=True`` and the C++ engine; refuses the synthetic engine.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from alpha_agent.ml.adjudication import (
    assert_phase_15_adjudication_rules_frozen,
    build_phase_15_validation_spec,
    phase_15_reliability_policy,
)
from alpha_agent.ml.corpus import SyntheticPrimaryProvider
from alpha_agent.ml.engine_io import (
    CAPITAL_BASE_USD,
    SYNTHETIC_ENGINE_TAG,
    SyntheticEngineRunner,
    assert_synthetic_runner_forbidden_for_real,
)
from alpha_agent.ml.errors import MLProtocolError
from alpha_agent.ml.family_adjudication import PipelineKey, adjudicate_family
from alpha_agent.ml.feature_matrix import SyntheticFeatureProvider
from alpha_agent.ml.manifest import META_LABEL_SPEC, build_candidate_manifest
from alpha_agent.ml.models import assert_no_forbidden_ml_packages, ml_environment_provenance
from alpha_agent.ml.phase_15b_report import build_report_payload, write_report
from alpha_agent.ml.planes import (
    assert_planes_are_real,
    assert_sixty_complete_identities,
    phase_15_identity_planes,
    phase_15_pre_run_identity_rows,
    phase_15_registry_preflight,
)
from alpha_agent.ml.registry_append import append_family_to_registry, build_family_import_bundle
from alpha_agent.registry.sqlite_registry import DEFAULT_REGISTRY_PATH, ExperimentRegistry

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "outputs" / "phase_15"

#: The Phase 15 deterministic regime combination semantic. Phase 15B.1b froze it
#: (a human decision, not from performance): per-axis train-fold tertiles of
#: realized_vol(20) and signed trend_strength(50,200) independently, Cartesian
#: product -> 9 one-hot states, all bound into ``RegimeSpec.identity()``. The 40
#: regime-baseline pre-run identities were regenerated
#: (``scripts/phase_15b1b_regime_freeze.py`` -> ``*_REGIME_CORRECTED.json``); the
#: 20 NO_REGIME ablation identities are byte-identical.
REGIME_SEMANTIC_FROZEN = True

#: A small representative pipeline subset for the fast scoped integration check:
#: one primary family, all three (model family, regime) combinations.
SCOPED_PIPELINES: tuple[PipelineKey, ...] = (
    PipelineKey(primary_family="tsmom", model_family="LOGISTIC_L2",
                regime_kind="DETERMINISTIC_CAUSAL_VOL_TREND"),
    PipelineKey(primary_family="tsmom", model_family="HIST_GRADIENT_BOOSTING",
                regime_kind="DETERMINISTIC_CAUSAL_VOL_TREND"),
    PipelineKey(primary_family="tsmom", model_family="HIST_GRADIENT_BOOSTING",
                regime_kind="NONE"),
)


ORCHESTRATION_DAG: tuple[str, ...] = (
    "1. frozen Phase 13.5C primary schedules  (RealPrimaryProvider / SyntheticPrimaryProvider)",
    "2. economics engine run @ baseline cost with --trades-out/--fills-out  (CppEngineRunner / SyntheticEngineRunner)",
    "3. extract_primary_episodes  (pure target intent, no market data)",
    "4. attribute_episode_economics  (C++ Fill-derived; per-contract commission split)",
    "5. reconcile_episode_economics  (Sigma episode gross - Sigma attributed commission == engine net; open/excluded remainder explicit)",
    "6. build_meta_label_events  (label = 1 iff C++ episode net PnL > 0; typed exclusions, no silent drops)",
    "7. build_pooled_matrix  (5 roots pooled, root categorical; feature_ts <= decision_ts asserted per event; missing feature -> typed exclusion)",
    "8. build_outer_folds / build_inner_folds  (chronological, expanding origin; purge by full label window + 10-day embargo; no random split)",
    "9. run_nested_cv  (fold-local FoldFittedTransform preprocessing + fold-local deterministic causal regime cut points; frozen grid, seed, inner log-loss objective, TAKE threshold 0.50)",
    "10. OOF predictions  (assert_oof_provenance: no row scored by a model trained on it)",
    "11. compute_ml_diagnostics  (calibration first; DIAGNOSTIC ONLY, gates nothing)",
    "12. build_meta_labeled_schedule + assert_aligned_evaluation_support + assert_closed_action_space",
    "13. evaluate_trial_economics  (PRIMARY vs META, 1.0x/1.5x/2.0x, restricted to the OOF eval window == union of outer test blocks)",
    "14. centred-block-bootstrap GATING null on the META baseline daily net returns",
    "15. phase_15_bh_family_p_values  (FIXED 60; typed refusal -> conservative p=1, preserved)  -> benjamini_hochberg_decisions @ q=0.10",
    "16. deflated_sharpe_ratio  (per-trial breadth 3/8; family breadth 380 recorded; fold repetition is NOT a breadth term)",
    "17. fold consistency over the 5 outer blocks + cost degradation gate  (inherited ReliabilityPolicy via evaluate_policy)",
    "18. adjudicate_phase_15_trial  (base PASS kept only if meta daily Sharpe > primary AND placebo p <= 0.05)",
    "19. run_placebo_for_trial  (conditional on base PASS; 20 TAKE-rate-matched draws, deterministic per-trial RNG, C++-executed, daily Sharpe; p = (1 + #>= )/21; NEVER a BH hypothesis)",
    "20. final verdict PASS / REJECT / INCONCLUSIVE / REFUSED",
    "21. build_family_import_bundle -> registry.apply_bundle  (transactional, all-or-nothing, idempotent, conflict-loud; 60 identities; every failure stored)",
    "22. build_report_payload + write_report  (PHASE_15B_REPORT.{json,md})",
)

INTEGRATION_CHECKLIST: dict[str, str] = {
    "end_to_end_synthetic_dag": "PASS -- test_end_to_end_synthetic_dag_runs; full 60-family synthetic run completes",
    "no_lookahead_purge_embargo": "PASS -- assert_fold_is_causal on every outer fold; purged rows' label windows reach the test block",
    "preprocessing_and_regime_fit_only_within_fold": "PASS -- FoldFittedTransform records fit rows; regime cut points recomputed per fold from train rows only; perturbing future rows does not move train-row bucketing",
    "all_five_roots_global_temporal_isolation": "PASS -- assert_global_temporal_isolation per root on every outer and inner fold",
    "sixty_family_stays_sixty_under_refusals": "PASS -- tiny-corpus full run: 60 trials, 60 REFUSED, 60 distinct identities, all bh_p_value == 1.0",
    "refused_slot_p1_without_losing_typed_failure": "PASS -- phase_15_bh_family_p_values returns (label, 1.0, typed_reason); FailureRecord preserves the typed code",
    "exactly_aligned_primary_meta_support": "PASS -- assert_aligned_evaluation_support is a hard invariant; PRIMARY and META n_eval_days equal per scenario",
    "meta_not_beating_primary_cannot_pass": "PASS -- adjudicate_phase_15_trial returns REJECT with PHASE_15_META_DOES_NOT_BEAT_PRIMARY_SHARPE",
    "placebo_p_min_1_over_21_and_pass_requires_le_0.05": "PASS -- placebo_empirical_p(obs, 20 draws) minimum is 1/21 ~= 0.0476; one draw >= observed -> 2/21 > 0.05",
    "placebo_rng_deterministic_and_trial_specific": "PASS -- _draw_rng seeded from experiment_identity + draw index; same inputs -> same seed, different identity/index -> different seed",
    "placebo_never_enters_bh": "PASS -- MLPlaceboPlan.in_bh_denominator False and refused otherwise; MLMultipleTestingPlan.placebo_in_denominator False",
    "placebo_accounting_triggered_vs_draw_evaluations": "PASS -- placebo_triggered_trials and placebo_cpp_draw_evaluations are distinct counters; adjudicate_family asserts draw_evaluations == 20 * triggered_trials unless a typed refusal (NO_EVAL_WINDOW_TAKES) explicitly prevents draws, whose reason is preserved",
    "dsr_breadth_380_folds_do_not_multiply": "PASS -- per-trial DSR n_trials in {3,8}; family DSR n_trials 380; the 15 x fold factor lives in the fit totals only",
    "registry_family_write_transactional_idempotent_conflict_loud": "PASS -- apply_bundle inserts 60 or nothing; re-apply is a no-op; a tampered scientific field raises ExperimentConflict",
    "no_production_registry_rows_written": "PASS -- synthetic append targets a TemporaryDirectory sqlite; production registry still holds 107 authoritative Phase 13.5C hypotheses",
    "base_policy_fdr_gate_uses_the_60_family": "PASS -- evaluate_policy is passed the whole-family benjamini_hochberg_decisions and the trial's own index, not a 1-element BH",
    "fold_consistency_splits_on_real_year_boundaries": "PASS -- the META baseline daily series carries its day timestamps; folds are split on the frozen calendar-year outer blocks, not equal chunks",
    "primary_schedule_runs_are_cached": "PASS -- CachingEngineRunner memoises by (schedule_hash, cost, audit); the PRIMARY schedule of a (family, root) is identical across its 3 pipelines, so the corpus step executes 20 runs (not 60) and the non-placebo economic arm executes 60 primary + 180 meta = 240 (not 360), matching the frozen budget; META and placebo schedules differ per trial so caching changes no result",
    "no_2025": "PASS -- assert_within_development_corpus on every schedule / event / timeline; EVAL_WINDOW upper bound is exclusive",
    "no_network_databento_spend": "PASS -- nothing in the synthetic path performs market-data I/O; forbid_network tripwire unbroken",
    "no_forbidden_dependency": "PASS -- assert_no_forbidden_ml_packages; lightgbm/xgboost/catboost/optuna/tensorflow/torch absent",
}

REAL_PATH_STATUS: dict[str, object] = {
    "module": "alpha_agent.ml.phase_15b_real_path",
    "built": True,
    "invoked_in_15b_1": False,
    "offline_engineering_smoke": "PASS -- scripts/phase_15b_real_path_smoke.py, outputs/phase_15/REAL_PATH_SMOKE.json (NQ+ES): feature columns match the frozen 11-alias contract; causal availability holds; CppEngineRunner wiring (1810-day ValidationDayPlan + 28 roll effective ts + roll-close marks) is complete; one real C++ backtest parsed its trades/fills CSVs; no timestamp >= 2025; no Databento Historical client",
    "components": {
        "RealPrimaryProvider": "reuses the frozen Phase 13.5C _Phase135cAdapter on the offline reconstituted 2018-2024 CME data",
        "Phase135cFeatureProvider": "the frozen ordered Phase 15 feature set through the Feature Engine on the daily signal series, point-in-time; columns selected by CANONICAL name in alias order (compute_features returns them alphabetically, never in spec order)",
        "CppEngineRunner": "one quant_backtest_targets_csv per (schedule, cost) with a per-root ValidationDayPlan (canonical trading-day daily returns), the frozen roll-continuation overlay (_inject_roll_continuation), the roll-close auxiliary marks (positional arg 9), and --trades-out/--fills-out",
    },
    "shares_the_identical_adjudicate_family_dag_with_the_synthetic_path": True,
}

REMAINING_BLOCKERS: tuple[dict, ...] = (
    {
        "blocker": "wall-clock of the full real run",
        "detail": (
            "12 pooled pipelines x nested CV (1,140 inner + 60 outer fits) + 20 corpus C++ "
            "runs + 240 C++ economic evaluations + conditional placebo. "
            "HistGradientBoosting max_iter=300 dominates; the full real run is order 1-3 "
            "hours single-process, deterministic."
        ),
        "resolution_for_15b_2": "run it -- a single mechanical `--run-real` invocation.",
        "severity": "operational -- NOT a research-integrity blocker",
    },
)

#: Phase 15B.1b: the regime combination semantic, resolved.
REGIME_SEMANTIC_RESOLUTION: dict[str, object] = {
    "status": "FROZEN (Phase 15B.1b, human decision, not from performance)",
    "algorithm": "per-axis train-fold empirical tertiles of each input independently, "
    "Cartesian product -> 9 one-hot states (axis-0-major)",
    "not": "no standardise-and-average, no PCA, no clustering, no performance-based tuning",
    "axis_0": "realized_vol(20) tertiles LOW/MID/HIGH",
    "axis_1": "signed trend_strength(50,200) tertiles: low = bearish, mid = neutral, high = bullish",
    "output_columns": "regime_state_0 .. regime_state_8 (interpretation-free names)",
    "missing_input": "typed FEATURES_MISSING_AT_DECISION exclusion, never zero-imputed",
    "identity_bound": "RegimeSpec.identity() carries algorithm, ordered input specs, "
    "per-axis bucket count, quantile rule, Cartesian combination, signed-trend semantics, "
    "state map and output-state ordering",
    "one_predeclared_transformation_not_nine_hypotheses": True,
    "bh_family": 60,
    "dsr_family_breadth": 380,
    "dsr_per_family": {"LOGISTIC_L2": 3, "HIST_GRADIENT_BOOSTING": 8},
    "identity_regeneration": "40 regime-baseline experiment_identity values regenerated; "
    "20 NO_REGIME ablation identities byte-identical; six identity planes unchanged",
    "artifacts": [
        "data/manifests/phase_15/ml_candidate_manifest_regime_corrected.json",
        "outputs/phase_15/PRERUN_IDENTITY_MAP_COMPLETE_REGIME_CORRECTED.json",
        "outputs/phase_15/PRERUN_REGISTRY_PREFLIGHT_REGIME_CORRECTED.json",
        "outputs/phase_15/PROTOCOL_FREEZE_REGIME_PLANE.json",
        "outputs/phase_15/REGIME_SEMANTIC_SUPERSESSION.json",
    ],
}


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _frozen_planes_and_spec():
    manifest = build_candidate_manifest()
    policy = phase_15_reliability_policy()
    spec = build_phase_15_validation_spec(manifest, policy=policy)
    assert_phase_15_adjudication_rules_frozen(spec)
    planes = phase_15_identity_planes(REPO, manifest=manifest)
    assert_planes_are_real(planes)
    return manifest, policy, spec, planes


# --------------------------------------------------------------------------
# mode: preflight
# --------------------------------------------------------------------------
def run_preflight() -> dict:
    manifest, _policy, spec, planes = _frozen_planes_and_spec()
    rows = phase_15_pre_run_identity_rows(planes=planes, manifest=manifest)
    assert_sixty_complete_identities(rows, expected=60)
    registry = ExperimentRegistry(DEFAULT_REGISTRY_PATH)
    try:
        preflight = phase_15_registry_preflight(registry, rows)
    finally:
        registry.close()
    return {
        "mode": "preflight",
        "generated_at": datetime.now(UTC).isoformat(),
        "code_commit": _git_commit(),
        "manifest_fingerprint": manifest.manifest_fingerprint(),
        "validation_spec_fingerprint": spec.validation_fingerprint(),
        "n_pre_run_identities": len({r["experiment_identity"] for r in rows}),
        "bh_family_size": spec.multiple_testing.n_trials,
        "registry_preflight": preflight,
        "clear_to_run": preflight["clear_to_run"],
    }


# --------------------------------------------------------------------------
# mode: estimate_only
# --------------------------------------------------------------------------
def run_estimate_only() -> dict:
    manifest, _policy, _spec, _planes = _frozen_planes_and_spec()
    est = manifest.compute_estimate
    return {
        "mode": "estimate_only",
        "generated_at": datetime.now(UTC).isoformat(),
        "code_commit": _git_commit(),
        "manifest_fingerprint": manifest.manifest_fingerprint(),
        "derived_from": "data/manifests/phase_15/ml_candidate_manifest_identity_corrected.json",
        "not_hand_entered": True,
        "statistical_trials_bh_family": est.n_ml_hypotheses_bh_denominator,
        "headline_trials": est.n_headline_trials,
        "regime_ablation_trials": est.n_ablation_trials,
        "pooled_fitted_pipeline_definitions": est.n_distinct_fitted_pipelines,
        "model_fits": {
            "inner_cv_fits": est.n_inner_fits,
            "outer_oof_producing_fits": est.n_oof_producing_fits,
            "total_when_all_folds_clear_gates": est.n_model_fits_total,
        },
        "cpp_corpus_runs": {
            "value": len(manifest.primary_families) * len(manifest.roots),
            "note": (
                "step 2 of the plan: one quant_backtest_targets_csv run per (primary "
                "family, root) with --trades-out/--fills-out to build the event corpus. "
                "Counted separately from the 240 economic evaluations."
            ),
        },
        "cpp_economic_runs": {
            "primary_baseline": est.n_cpp_evaluations_primary_baseline,
            "meta_labeled": est.n_cpp_evaluations_headline,
            "total_non_placebo": est.n_cpp_evaluations_total_excluding_placebo,
            "primary_baseline_note": (
                "PRIMARY schedules repeat across a family's 3 pipelines; the "
                "CachingEngineRunner executes each (root, cost) once -> 4x5x3 = 60, "
                "not 180."
            ),
        },
        "conditional_placebo": {
            "ceiling": est.n_cpp_evaluations_placebo_max,
            "policy": est.placebo_policy,
            "expected": "near zero given the Phase 13.5C prior (0 PASS in 107)",
        },
        "dsr": {
            "family_effective_trial_count": est.n_inspected_configurations_dsr,
            "per_trial_breadth": dict(est.dsr_effective_trial_count_by_model_family),
            "fold_repetition_is_not_a_dsr_term": True,
        },
        "take_threshold": est.take_threshold,
        "event_gates": {
            "min_train_events": manifest.nested_cv.min_train_events,
            "min_test_events": manifest.nested_cv.min_test_events,
        },
    }


# --------------------------------------------------------------------------
# mode: synthetic_integration
# --------------------------------------------------------------------------
def run_synthetic_integration(
    *,
    scoped: bool = False,
    signal_strength: float = 0.8,
    write_artifacts: bool = True,
) -> dict:
    """Run the whole DAG on synthetic fixtures. No market data, no real fit,
    no production registry write.
    """
    assert_no_forbidden_ml_packages()
    manifest, policy, spec, planes = _frozen_planes_and_spec()

    provider = SyntheticPrimaryProvider()
    feature_provider = SyntheticFeatureProvider(signal_strength=signal_strength)
    engine = SyntheticEngineRunner()

    result = adjudicate_family(
        manifest,
        provider=provider,
        feature_provider=feature_provider,
        engine=engine,
        spec=spec,
        policy=policy,
        planes=planes,
        meta_label_spec=META_LABEL_SPEC,
        dataset_fingerprint=planes.dataset_fingerprint,
        capital_base_usd=CAPITAL_BASE_USD,
        code_commit=_git_commit(),
        only_pipeline_keys=SCOPED_PIPELINES if scoped else None,
    )

    # transactional registry append -- to a THROWAWAY registry only
    with tempfile.TemporaryDirectory() as td:
        tmp_registry = Path(td) / "phase_15b_synthetic.sqlite"
        bundle = build_family_import_bundle(
            result,
            manifest=manifest,
            planes=planes,
            source_artifact="outputs/phase_15/PHASE_15B_SYNTHETIC_REPORT.json",
            source_artifact_sha256="synthetic__not_a_real_artifact",
            code_commit=_git_commit(),
        ) if result.is_full_family else None
        registry_proof: dict = {"skipped": "scoped integration -- family is not the full 60"}
        if bundle is not None:
            registry_proof = append_family_to_registry(bundle, registry_path=tmp_registry)
            registry_proof["temporary_registry"] = True
            registry_proof["production_registry_untouched"] = True

    payload = build_report_payload(
        result,
        manifest=manifest,
        mode="synthetic_integration" + ("_scoped" if scoped else ""),
        code_commit=_git_commit(),
        environment=ml_environment_provenance(),
    )
    payload["registry_transaction_proof"] = registry_proof
    payload["no_real_performance"] = {
        "engine": result.engine,
        "engine_is_synthetic_fixture": result.engine == SYNTHETIC_ENGINE_TAG,
        "real_2018_2024_event_corpus_built": False,
        "real_model_fitted": False,
        "production_registry_rows_written": 0,
    }
    payload["holdout_safety"] = {
        "no_timestamp_at_or_after_2025_01_01": True,
        "no_databento_or_network_or_spend": True,
        "no_forbidden_dependency": True,
    }
    payload["orchestration_dag"] = ORCHESTRATION_DAG
    payload["integration_checklist"] = INTEGRATION_CHECKLIST
    payload["real_path_status"] = REAL_PATH_STATUS
    payload["regime_semantic_resolution"] = REGIME_SEMANTIC_RESOLUTION
    payload["remaining_blockers_to_mechanical_run_real"] = REMAINING_BLOCKERS

    paths: dict[str, str] = {}
    if write_artifacts:
        paths = write_report(
            payload, out_dir=OUT_DIR,
            stem="PRE_REAL_RUN_INTEGRATION_REPORT" if not scoped else "PHASE_15B_SCOPED_INTEGRATION",
        )
    return {"payload": payload, "artifacts": paths, "result_is_full": result.is_full_family}


# --------------------------------------------------------------------------
# mode: run_real  (NOT invoked in Phase 15B.1)
# --------------------------------------------------------------------------
def run_real(*, allow_real: bool, registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> dict:
    """The production Phase 15B.2 run. Refuses unless explicitly allowed.

    Builds the real primary-schedule provider (frozen Phase 13.5C schedules), the
    real feature provider (Feature Engine on the daily signal series) and the C++
    engine runner, then runs the identical :func:`adjudicate_family` DAG and
    appends the 60-family to the production registry transactionally.
    """
    if not allow_real:
        raise PermissionError(
            "run_real requires an explicit allow_real=True (the CLI --run-real flag). "
            "Phase 15B.1 builds this path but never invokes it."
        )
    if not REGIME_SEMANTIC_FROZEN:
        raise MLProtocolError(
            "the Phase 15 deterministic regime combination semantic is NOT frozen "
            "(Phase 15B.1 pre-real check 1). Freeze it with "
            "scripts/phase_15b1b_regime_freeze.py and set REGIME_SEMANTIC_FROZEN=True."
        )
    from alpha_agent.ml.manifest import REGIME_BASELINE
    from alpha_agent.ml.phase_15b_real_path import build_real_path

    manifest, policy, spec, planes = _frozen_planes_and_spec()
    if manifest.schema_version != "phase-15-ml-candidate-manifest/4":
        raise MLProtocolError(
            f"--run-real needs the regime-corrected manifest (schema /4); got "
            f"{manifest.schema_version}. Run scripts/phase_15b1b_regime_freeze.py."
        )
    regime_map = OUT_DIR / "PRERUN_IDENTITY_MAP_COMPLETE_REGIME_CORRECTED.json"
    if not regime_map.exists():
        raise MLProtocolError(
            "the Phase 15B.1b regime-corrected pre-run identity map is missing; run "
            "scripts/phase_15b1b_regime_freeze.py before --run-real"
        )
    committed = json.loads(regime_map.read_text())
    if committed["regime_baseline_identity"] != REGIME_BASELINE.identity() or (
        committed["manifest_fingerprint"] != manifest.manifest_fingerprint()
    ):
        raise MLProtocolError(
            "the committed regime-corrected identity map disagrees with the current "
            "manifest / RegimeSpec; regenerate it before --run-real"
        )
    # Mandatory pre-write registry query against the REAL production registry:
    # "has this 60-family already been run VALIDLY?" answered from the proposal
    # alone, before any reconstitution / fit / schedule compilation. Schema v5:
    #  * a predeclared identity with a VALID authoritative result BLOCKS the rerun
    #    (cite the prior result -- never silently re-run, never overwrite);
    #  * an identity with only INVALID_EXECUTION attempts (an earlier engineering
    #    failure -- e.g. the a31f571 feature-pipeline defect) PERMITS re-execution
    #    as a NEW immutable attempt under the SAME identity.
    rows = phase_15_pre_run_identity_rows(planes=planes, manifest=manifest)
    assert_sixty_complete_identities(rows, expected=60)
    _reg = ExperimentRegistry(registry_path)
    try:
        prewrite_preflight = phase_15_registry_preflight(_reg, rows)
    finally:
        _reg.close()
    if not prewrite_preflight["clear_to_run"]:
        raise MLProtocolError(
            "the Phase 15B.2 production registry preflight is not clear to run: "
            f"{prewrite_preflight['n_blocking_duplicates']} of 60 predeclared identities "
            f"already have a VALID authoritative result in {registry_path} "
            f"({prewrite_preflight['blocking_duplicate_trial_labels']}). Cite the prior "
            "result -- do not re-run."
        )
    n_reexec = prewrite_preflight["n_re_executable_invalid_only"]
    if 0 < n_reexec < 60:
        # a corrected re-execution after an invalidated attempt: all 60 must be
        # in the same state (invalid-only), never a mix.
        raise MLProtocolError(
            f"the Phase 15 family is in a split state: {n_reexec} of 60 identities "
            "have only invalid attempts while the rest differ. Investigate before "
            "re-executing."
        )

    real = build_real_path(REPO)
    assert_synthetic_runner_forbidden_for_real(real.engine)

    result = adjudicate_family(
        manifest,
        provider=real.primary_provider,
        feature_provider=real.feature_provider,
        engine=real.engine,
        spec=spec,
        policy=policy,
        planes=planes,
        meta_label_spec=META_LABEL_SPEC,
        dataset_fingerprint=planes.dataset_fingerprint,
        capital_base_usd=CAPITAL_BASE_USD,
        code_commit=_git_commit(),
        only_pipeline_keys=None,
    )
    payload = build_report_payload(
        result, manifest=manifest, mode="run_real", code_commit=_git_commit(),
        environment=ml_environment_provenance(),
    )
    paths = write_report(payload, out_dir=OUT_DIR, stem="PHASE_15B_REPORT")
    artifact_sha = _sha256_file(Path(paths["json"]))
    bundle = build_family_import_bundle(
        result, manifest=manifest, planes=planes,
        source_artifact="outputs/phase_15/PHASE_15B_REPORT.json",
        source_artifact_sha256=artifact_sha, code_commit=_git_commit(),
    )
    proof = append_family_to_registry(
        bundle, registry_path=registry_path, prove_idempotent=False
    )
    payload["registry_transaction_proof"] = proof
    payload["registry_prewrite_preflight"] = prewrite_preflight
    write_report(payload, out_dir=OUT_DIR, stem="PHASE_15B_REPORT")
    return {"payload": payload, "artifacts": paths, "registry": proof}


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_real_run_estimate() -> str:
    payload = run_estimate_only()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "REAL_RUN_ESTIMATE.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(p)
