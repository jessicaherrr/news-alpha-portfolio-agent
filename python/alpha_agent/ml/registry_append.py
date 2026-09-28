"""Transactional append of the frozen 60-trial Phase 15 family to the registry.

Phase 15 creates no ML registry: it uses ``data/registry/experiments.sqlite``
(schema v5, identity schema v2) exactly as Phase 14 defined it. The whole family
is applied as one :class:`~alpha_agent.registry.models.ImportBundle` through
:meth:`ExperimentRegistry.apply_bundle`, which validates every record before the
transaction opens and commits all-or-nothing -- there is never a
partially-written 60-family.

Schema v5: this family run is one immutable EXECUTION ATTEMPT of each of the 60
pre-run identities. A first run lands as ``attempt_ordinal = 1``; the corrected
run after the a31f571 ``INVALID_EXECUTION`` attempt (a feature-pipeline defect)
lands as ``attempt_ordinal = 2`` under the SAME 60 ``experiment_identity``
values -- no new hypothesis, BH family still 60. Execution commit / report /
schedule fingerprints are ATTEMPT provenance and never enter identity.

Rules honoured here (prompt 15B.1 s.16, CLAUDE.md registry section):

* exactly 60 pre-run ``experiment_identity`` values, one per predeclared trial;
* append, never replace -- a conflicting identity raises loudly; a re-execution
  after an INVALID_EXECUTION attempt is a NEW attempt, never a new identity;
* every failure stored, linked to the exact execution attempt (a refused trial
  is a ``FAILED`` experiment plus a typed ``FailureRecord``, never a scientific
  ``REJECT`` faked to get ``p = 1``);
* post-run provenance (selected hyperparameters, OOF hash, target-schedule hash,
  model-artifact fingerprint, report fingerprint) recorded as provenance only;
* ``target_schedule_hash`` / ``report_fingerprint`` are ``NULL`` for a trial
  whose committed artifact holds no such value (a refusal), with
  ``evidence_completeness`` stating why.

In Phase 15B.1 this runs ONLY against a temporary registry
(:func:`append_family_to_registry` with an explicit throwaway path); the
production write is Phase 15B.2.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from alpha_agent.ml.family_adjudication import Phase15FamilyResult, TrialResult
from alpha_agent.ml.manifest import ML_FEATURE_SET
from alpha_agent.ml.trials import MLIdentityPlanes, experiment_spec_for_trial
from alpha_agent.registry.enums import (
    AssetDomain,
    AttemptStatus,
    ExperimentStatus,
    FailureClass,
    FailureScope,
    RegistryVerdict,
    TrialRole,
)
from alpha_agent.registry.identity import IDENTITY_SCHEMA
from alpha_agent.registry.models import (
    ExecutionAttemptRecord,
    ExperimentRecord,
    FailureRecord,
    ImportBundle,
    MarketWindow,
    ResultRecord,
)
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.validation.fingerprint import fingerprint

PHASE_15_MARKET_WINDOW = MarketWindow(
    label="PHASE_15_DEVELOPMENT_CORPUS",
    start_date="2018-01-01",
    end_date="2024-12-31",           # inclusive; 2025 is the LOCKED_FINAL_HOLDOUT
)

_VERDICT_MAP = {
    "PASS": RegistryVerdict.PASS,
    "REJECT": RegistryVerdict.REJECT,
    "INCONCLUSIVE": RegistryVerdict.INCONCLUSIVE,
    "REFUSED": RegistryVerdict.NOT_ADJUDICATED,
}


def _trial_role(role: str) -> TrialRole:
    return TrialRole.ABLATION if role == "ABLATION" else TrialRole.CANONICAL


def _report_fingerprint(trial: TrialResult) -> str | None:
    if trial.refused or trial.economics is None:
        return None
    payload = {
        "experiment_identity": trial.experiment_identity,
        "adjudication": trial.adjudication.model_dump(mode="json"),
        "gating_null_p": trial.gating_null.p_value if trial.gating_null else None,
        "dsr": trial.dsr.deflated_sharpe_ratio if trial.dsr else None,
        "bh_p_value": trial.bh_p_value,
        "bh_q_value": trial.bh_q_value,
        "economics_delta": trial.economics.economic_delta_vs_primary_usd,
        "oof_prediction_hash": None,
    }
    return fingerprint("mltrialreport1", payload, allow_non_finite=True)


def build_family_import_bundle(
    result: Phase15FamilyResult,
    *,
    manifest,
    planes: MLIdentityPlanes,
    source_artifact: str,
    source_artifact_sha256: str,
    code_commit: str = "",
    phase: str = "15B",
    import_id: str | None = None,
) -> ImportBundle:
    """Turn the adjudicated family into one transactional import bundle."""
    trials_by_label = {t.trial_label: t for t in manifest.trials}
    created_at = datetime.now(UTC).isoformat()
    #: failures belong to THIS execution attempt -- a distinct id per run keeps
    #: the a31f571 INVALID attempt's failure evidence and the corrected attempt's
    #: side by side (schema v5), never colliding on failure_id.
    run_tag = (source_artifact_sha256 or "run")[:12]

    experiments: list[ExperimentRecord] = []
    attempts: list[ExecutionAttemptRecord] = []
    results: list[ResultRecord] = []
    failures: list[FailureRecord] = []

    for tr in result.trials:
        mtrial = trials_by_label[tr.trial_label]
        exp_spec = experiment_spec_for_trial(mtrial, planes=planes)
        if exp_spec.experiment_identity() != tr.experiment_identity:
            raise ValueError(
                f"{tr.trial_label}: rebuilt identity disagrees with the adjudicated one"
            )

        # per-fold selected hyperparameters (post-fit provenance)
        selected: list[dict] = []
        oof_hash: str | None = None
        report_fp: str | None = _report_fingerprint(tr)
        target_hash: str | None = None
        completeness = "full"
        if tr.refused:
            completeness = f"partial:typed_refusal:{tr.typed_refusal}"
        elif tr.economics is not None:
            base_meta = next(
                m for m in tr.economics.meta if m.cost_scenario_label == "baseline_1_0x"
            )
            target_hash = base_meta.schedule_hash
        # attach the training report's per-fold selections
        pipe = next(
            (
                p for p in result.pipelines
                if p.key.primary_family == tr.primary_family
                and p.key.model_family == tr.model_family
                and p.key.regime_kind == tr.regime_kind
            ),
            None,
        )
        if pipe is not None and pipe.training_report is not None:
            oof_hash = pipe.training_report.oof_prediction_hash
            selected = [
                {
                    "outer_fold": f.fold_index,
                    "selected_hyperparameters": f.selected_hyperparameters,
                    "model_artifact_fingerprint": f.model_artifact_fingerprint,
                    "inner_objective_value": f.inner_objective_value,
                }
                for f in pipe.training_report.folds
            ]

        strategy_spec_json = {
            "primary_family": tr.primary_family,
            "composite_strategy_fingerprint": exp_spec.composite_fingerprint(),
            "model_search_identity": exp_spec.model_search.identity(),
            "regime_kind": tr.regime_kind,
            "take_threshold": exp_spec.take_threshold,
            "training_roots": list(exp_spec.training_roots),
            "post_fit_provenance": {
                "selected_hyperparameters_per_outer_fold": selected,
                "oof_prediction_hash": oof_hash,
            },
        }

        experiments.append(
            ExperimentRecord(
                experiment_identity=tr.experiment_identity,
                identity_schema=IDENTITY_SCHEMA,
                experiment_id=tr.friendly_experiment_id,
                display_name=tr.trial_label,
                created_at=created_at,
                phase=phase,
                status=ExperimentStatus.FAILED if tr.refused else ExperimentStatus.COMPLETED,
                code_commit=code_commit,
                root_symbol=tr.root_symbol,
                # Phase 15 trains only on Futures roots -- a stated fact about
                # this family, not an inference (Phase 6 ETF Research Pilot).
                asset_domain=AssetDomain.FUTURES,
                strategy_family=exp_spec.strategy_family,
                strategy_fingerprint=exp_spec.composite_fingerprint(),
                strategy_id=exp_spec.label,
                strategy_spec_json=strategy_spec_json,
                feature_spec_fingerprint=ML_FEATURE_SET.identity(),
                target_schedule_hash=target_hash,
                candidate_manifest_fingerprint=manifest.manifest_fingerprint(),
                dataset_fingerprint=planes.dataset_fingerprint,
                split_identity=exp_spec.nested_cv.identity(),
                market_window=PHASE_15_MARKET_WINDOW,
                validation_spec_fingerprint=planes.validation_spec_fingerprint,
                reliability_policy_fingerprint=planes.reliability_policy_fingerprint,
                execution_config_identity=planes.execution_config_identity,
                cost_config_identity=planes.cost_config_identity,
                risk_identity=planes.risk_identity,
                trial_role=_trial_role(tr.trial_role),
                parameter_variant_identity=exp_spec.parameter_variant_identity(),
                parameter_variant_label=exp_spec.parameter_variant_label(),
                report_fingerprint=report_fp,
                notes=f"Phase 15 meta-labeling; verdict {tr.adjudication.verdict}",
            )
        )

        # schema v5: this family run is one immutable EXECUTION ATTEMPT of each
        # of the 60 pre-run identities. ``attempt_ordinal=None`` -> the registry
        # assigns the next ordinal (1 on a first run; 2 for the corrected run
        # after the a31f571 INVALID_EXECUTION attempt). Execution provenance
        # (commit / report / schedule) lives ONLY on the attempt, never identity.
        attempts.append(
            ExecutionAttemptRecord(
                experiment_identity=tr.experiment_identity,
                identity_schema=IDENTITY_SCHEMA,
                attempt_ordinal=None,
                attempt_status=AttemptStatus.VALID,
                code_commit=code_commit,
                engine=result.engine,
                target_schedule_hash=target_hash,
                report_fingerprint=report_fp,
                source_artifact=source_artifact,
                source_artifact_sha256=source_artifact_sha256,
                created_at=created_at,
                notes=(
                    f"Phase 15 meta-labeling family run; verdict "
                    f"{tr.adjudication.verdict}"
                ),
            )
        )

        econ = tr.economics
        base_meta = (
            next(m for m in econ.meta if m.cost_scenario_label == "baseline_1_0x")
            if econ else None
        )
        results.append(
            ResultRecord(
                experiment_identity=tr.experiment_identity,
                headline_verdict=_VERDICT_MAP[tr.adjudication.verdict],
                reason_codes=tuple(tr.adjudication.reason_codes),
                gross_pnl_usd=base_meta.gross_pnl_usd if base_meta else None,
                costs_usd=base_meta.costs_usd if base_meta else None,
                net_pnl_usd=base_meta.net_pnl_usd if base_meta else None,
                daily_sharpe=base_meta.daily_sharpe if base_meta else None,
                annualized_sharpe=base_meta.annualized_sharpe if base_meta else None,
                gating_null_p=tr.gating_null.p_value if tr.gating_null else None,
                bh_p_value=tr.bh_p_value,
                bh_q=tr.bh_q_value,
                bh_rejected_at_q=tr.bh_rejected_at_q,
                dsr_probability=tr.dsr.deflated_sharpe_ratio if tr.dsr else None,
                fold_consistency=tr.fold_consistency,
                n_trades=base_meta.n_trades if base_meta else None,
                n_fills=base_meta.n_fills if base_meta else None,
                n_oos_days=base_meta.n_eval_days if base_meta else None,
                cost_stress=(
                    {
                        "meta_max_cost_degradation": econ.meta_max_cost_degradation,
                        "economic_delta_vs_primary_usd": econ.economic_delta_vs_primary_usd,
                    }
                    if econ else {}
                ),
                regime_evidence={"kind": tr.regime_kind, "status": "descriptive_only"},
                cross_market_reference={"status": "descriptive_only"},
                bootstrap_evidence=(
                    {
                        "placebo_ran": tr.placebo.ran,
                        "placebo_empirical_p": tr.placebo.empirical_p_value,
                        "placebo_significant": tr.placebo.significant,
                        "placebo_draws": tr.placebo.n_draws,
                    }
                    if tr.placebo else {}
                ),
                holdout_eligible=tr.adjudication.verdict == "PASS",
                evidence_completeness=completeness,
                source_artifact=source_artifact,
                source_artifact_sha256=source_artifact_sha256,
            )
        )

        if tr.refused:
            failures.append(
                FailureRecord(
                    failure_id=f"phase_15_refusal__{tr.friendly_experiment_id}__{run_tag}",
                    scope=FailureScope.EXPERIMENT,
                    experiment_identity=tr.experiment_identity,
                    failure_class=FailureClass.STATISTICAL_INCONCLUSIVE,
                    failure_code=tr.typed_refusal or "TRIAL_REFUSED",
                    summary=(
                        f"{tr.trial_label}: refused with typed reason "
                        f"{tr.typed_refusal}; enters the BH-60 family at conservative p=1"
                    ),
                    mechanism=(
                        "a predeclared event-count gate (min_train_events=100 / "
                        "min_test_events=20) was not met; the scope is REFUSED, not trained "
                        "on a lowered bar, and is NOT converted to a scientific REJECT"
                    ),
                    evidence={"typed_refusal": tr.typed_refusal, "bh_p_value": 1.0},
                    action_taken="recorded as a FAILED experiment; slot preserved in the 60-family",
                    created_at=created_at,
                    root_symbol=tr.root_symbol,
                    strategy_family=f"ml_meta_label.{tr.primary_family}",
                )
            )
        elif tr.adjudication.verdict in ("REJECT", "INCONCLUSIVE"):
            failures.append(
                FailureRecord(
                    failure_id=f"phase_15_{tr.adjudication.verdict.lower()}__{tr.friendly_experiment_id}__{run_tag}",
                    scope=FailureScope.EXPERIMENT,
                    experiment_identity=tr.experiment_identity,
                    failure_class=(
                        FailureClass.SCIENTIFIC_REJECTION
                        if tr.adjudication.verdict == "REJECT"
                        else FailureClass.STATISTICAL_INCONCLUSIVE
                    ),
                    failure_code="|".join(tr.adjudication.reason_codes) or tr.adjudication.verdict,
                    summary=f"{tr.trial_label}: {tr.adjudication.verdict}",
                    mechanism="; ".join(tr.adjudication.reason_codes),
                    evidence={
                        "reason_codes": list(tr.adjudication.reason_codes),
                        "base_reliability_verdict": tr.adjudication.base_reliability_verdict,
                        "meta_beats_primary": tr.adjudication.meta_beats_primary,
                        "placebo_p": (
                            tr.adjudication.placebo_p_value
                            if tr.adjudication.placebo_p_value is not None
                            else None
                        ),
                    },
                    action_taken="recorded as first-class evidence; never deleted",
                    created_at=created_at,
                    root_symbol=tr.root_symbol,
                    strategy_family=f"ml_meta_label.{tr.primary_family}",
                )
            )

    if len({e.experiment_identity for e in experiments}) != 60:
        raise ValueError(
            f"the Phase 15 family must carry exactly 60 distinct experiment identities; "
            f"got {len({e.experiment_identity for e in experiments})}"
        )

    return ImportBundle(
        import_id=import_id or f"phase_15b__{result.validation_spec_fingerprint[:24]}",
        phase=phase,
        source_fingerprint=source_artifact_sha256,
        created_at=created_at,
        experiments=tuple(experiments),
        execution_attempts=tuple(attempts),
        results=tuple(results),
        failures=tuple(failures),
        lineage=(),
        metadata={
            "bh_family_size": 60,
            "engine": result.engine,
            "verdict_counts": result.verdict_counts,
            "placebo_triggered_trials": result.placebo_triggered_trials,
            "placebo_cpp_draw_evaluations": result.placebo_cpp_draw_evaluations,
            "execution_attempt_status": "VALID",
        },
    )


def append_family_to_registry(
    bundle: ImportBundle, *, registry_path: str | Path, prove_idempotent: bool = True
) -> dict[str, object]:
    """Apply the bundle to the registry at ``registry_path`` transactionally.

    ``apply_bundle`` validates every record before the transaction opens and
    commits all-or-nothing -- there is never a partially-written 60-family.
    Conflicting content under an existing identity raises loudly
    (``ExperimentConflict`` / ``ScheduleProvenanceConflict``).

    ``prove_idempotent`` re-applies the byte-equivalent bundle and confirms it
    is a no-op -- used for the Phase 15B.1 synthetic proof against a throwaway
    registry. The Phase 15B.2 production write against the real registry passes
    ``prove_idempotent=False`` (the append is done once).
    """
    reg = ExperimentRegistry(registry_path)
    try:
        before = reg.count_experiments()
        counts = reg.apply_bundle(bundle)
        idempotent_ok: bool | None = None
        if prove_idempotent:
            reg.apply_bundle(bundle)
            idempotent_ok = reg.count_experiments() == before + len(bundle.experiments)
        after = reg.count_experiments()
        return {
            "registry_path": str(registry_path),
            "experiments_before": before,
            "experiments_after": after,
            "inserted": after - before,
            "apply_counts": counts,
            "idempotent_reapply_ok": idempotent_ok,
            "schema_version": reg.summary().schema_version,
        }
    finally:
        reg.close()
