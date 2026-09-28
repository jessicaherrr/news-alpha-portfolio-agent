#!/usr/bin/env python
"""Phase 15B.1b -- freeze the deterministic regime transformation semantics.

OFFLINE, idempotent, no spend, no network, no market data, no 2025 and no model
fitted to anything. It runs BEFORE the real Phase 15B run.

REGIME_BASELINE declared two inputs -- realized_vol(20) and signed
trend_strength(50,200) -- on scales that differ by ~2000x, but never fixed how
they combine into 3 tertile buckets. The provisional raw-mean rule made the
regime ~1.000 correlated with trend_strength alone. Phase 15B.1b freezes the
transformation as decided by a human (NOT from performance):

* per-axis train-fold empirical tertiles of EACH input independently
  (no standardise-and-average, no PCA / clustering / performance tuning);
* Cartesian product of the two axes -> 3 x 3 = 9 states, axis-0-major;
* 9 one-hot columns ``regime_state_0 .. regime_state_8``;
* the whole transformation -- algorithm, ordered input specs, per-axis bucket
  count, quantile rule, Cartesian combination, signed-trend semantics and
  output-state ordering -- bound into ``RegimeSpec.identity()``.

ONE predeclared transformation, not 9 hypotheses: BH stays 60, DSR stays 3/8
and family breadth 380. Only the ``regime.identity()`` moves, which sits in
``ml_composite_fingerprint`` -> the 40 regime-baseline pre-run
``experiment_identity`` values regenerate; the 20 NO_REGIME ablation identities
are BYTE-IDENTICAL and this script proves it. The six identity planes
(dataset / ValidationSpec / ReliabilityPolicy / execution / cost / risk) are
unchanged -- regime is not a plane.

Writes (all NEW -- nothing is edited in place):

* ``data/manifests/phase_15/ml_candidate_manifest_regime_corrected.json``
* ``outputs/phase_15/PRERUN_IDENTITY_MAP_COMPLETE_REGIME_CORRECTED.json``
* ``outputs/phase_15/PRERUN_REGISTRY_PREFLIGHT_REGIME_CORRECTED.json``
* ``outputs/phase_15/PRETRAIN_TRIAL_ACCOUNTING_REGIME_CORRECTED.json``
* ``outputs/phase_15/PROTOCOL_FREEZE_REGIME_PLANE.json``
* ``outputs/phase_15/REGIME_SEMANTIC_SUPERSESSION.json``
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import alpha_agent.features.compute  # noqa: F401 -- registers feature kinds
from alpha_agent.ml.adjudication import (
    assert_phase_15_adjudication_rules_frozen,
    build_phase_15_validation_spec,
    phase_15_reliability_policy,
)
from alpha_agent.ml.manifest import (
    MANIFEST_SCHEMA_VERSION,
    REGIME_ABLATION,
    REGIME_BASELINE,
    SUPERSEDED_15A2_MANIFEST_COMMITS,
    SUPERSEDED_15A2_MANIFEST_FINGERPRINT,
    SUPERSEDED_15A2_MANIFEST_PATH,
    SUPERSESSION_15B1B_REASON,
    assert_no_performance_fields,
    build_candidate_manifest,
    freeze_candidate_manifest,
    trial_accounting_payload,
)
from alpha_agent.ml.models import assert_no_forbidden_ml_packages, ml_environment_provenance
from alpha_agent.ml.planes import (
    Phase135cSources,
    assert_planes_are_real,
    assert_sixty_complete_identities,
    phase_15_identity_planes,
    phase_15_pre_run_identity_rows,
    phase_15_registry_preflight,
    planes_payload,
)
from alpha_agent.ml.trials import assert_one_identity_per_trial
from alpha_agent.registry.sqlite_registry import DEFAULT_REGISTRY_PATH, ExperimentRegistry
from alpha_agent.validation.fingerprint import fingerprint

OUT_DIR = REPO / "outputs" / "phase_15"
MANIFEST_15A2 = REPO / SUPERSEDED_15A2_MANIFEST_PATH
MANIFEST_15B1B = REPO / "data" / "manifests" / "phase_15" / "ml_candidate_manifest_regime_corrected.json"
#: the Phase 15B.0 complete identity map -- the map this freeze diffs against.
MAP_15B0 = OUT_DIR / "PRERUN_IDENTITY_MAP_COMPLETE_ADJUDICATION_CORRECTED.json"

FROZEN_COUNTS = {
    "n_ml_hypotheses_bh_denominator": 60,
    "n_headline_trials": 40,
    "n_ablation_trials": 20,
    "n_distinct_fitted_pipelines": 12,
    "n_model_fits_total": 1200,
    "n_inspected_configurations_dsr": 380,
    "n_cpp_evaluations_total_excluding_placebo": 240,
}
FROZEN_DSR_BREADTH_BY_FAMILY = {"LOGISTIC_L2": 3, "HIST_GRADIENT_BOOSTING": 8}
#: RegimeSpec.identity() values BEFORE this freeze (Phase 15A/15B.0).
PRIOR_REGIME_BASELINE_IDENTITY = (
    "mlregimespec1:b6a173bf82a9e84dd8bef067ce1234adcd75b14ed41db4f37b9aee76bc49b9b1"
)
PRIOR_REGIME_ABLATION_IDENTITY = (
    "mlregimespec1:c879559bc3bc8647e590f9b77e6c3a1ca22be2f7d3fdc72c3a9aa09e8c7f4aa5"
)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    # -- the Phase 15A.2 manifest this builds on must not have drifted --------
    on_disk_15a2 = json.loads(MANIFEST_15A2.read_text(encoding="utf-8"))
    if on_disk_15a2.get("manifest_fingerprint") != SUPERSEDED_15A2_MANIFEST_FINGERPRINT:
        raise SystemExit(
            f"{SUPERSEDED_15A2_MANIFEST_PATH} no longer holds the Phase 15A.2 manifest "
            f"({on_disk_15a2.get('manifest_fingerprint')} != "
            f"{SUPERSEDED_15A2_MANIFEST_FINGERPRINT}); superseded history is preserved, "
            "never rewritten"
        )

    manifest = build_candidate_manifest()
    assert_no_performance_fields(manifest)
    assert_no_forbidden_ml_packages()

    if manifest.schema_version != "phase-15-ml-candidate-manifest/4":
        raise SystemExit(f"expected manifest schema /4, got {manifest.schema_version}")
    if manifest.manifest_fingerprint() == SUPERSEDED_15A2_MANIFEST_FINGERPRINT:
        raise SystemExit("the regime correction did not move the manifest fingerprint")

    # -- the regime transformation is frozen and identity-bound --------------
    if REGIME_BASELINE.identity() == PRIOR_REGIME_BASELINE_IDENTITY:
        raise SystemExit("REGIME_BASELINE.identity() did not change")
    if REGIME_ABLATION.identity() != PRIOR_REGIME_ABLATION_IDENTITY:
        raise SystemExit(
            f"REGIME_ABLATION.identity() moved to {REGIME_ABLATION.identity()}; the 20 "
            "NO_REGIME experiment identities are frozen and must not change"
        )
    if REGIME_BASELINE.n_states != 9 or REGIME_BASELINE.output_columns() != tuple(
        f"regime_state_{i}" for i in range(9)
    ):
        raise SystemExit("the deterministic regime must expose exactly 9 one-hot states")
    regime_id = json.loads(REGIME_BASELINE.model_dump_json())

    # -- the six identity planes are UNCHANGED (regime is not a plane) -------
    policy = phase_15_reliability_policy()
    spec = build_phase_15_validation_spec(manifest, policy=policy)
    assert_phase_15_adjudication_rules_frozen(spec)
    sources = Phase135cSources(REPO)
    planes = phase_15_identity_planes(REPO, manifest=manifest)
    assert_planes_are_real(planes)
    plane_15b0 = json.loads(
        (REPO / "data" / "manifests" / "phase_15" / "ml_validation_plane_adjudication_corrected.json")
        .read_text(encoding="utf-8")
    )
    if planes.validation_spec_fingerprint != plane_15b0["validation_spec_fingerprint"]:
        raise SystemExit("the validation-spec plane moved; the regime is not a plane")

    # -- the 60 pre-run identities: 40 regenerate, 20 are byte-identical -----
    rows = phase_15_pre_run_identity_rows(planes=planes, manifest=manifest)
    assert_sixty_complete_identities(rows, expected=60)
    assert_one_identity_per_trial(manifest)

    prior = {t["trial_label"]: t for t in json.loads(MAP_15B0.read_text())["trials"]}
    new = {r["trial_label"]: r for r in rows}
    if set(prior) != set(new):
        raise SystemExit("trial labels changed; only the 40 regime-baseline identities should move")

    baseline_labels = {
        t.trial_label for t in manifest.trials
        if t.regime_kind.value == "DETERMINISTIC_CAUSAL_VOL_TREND"
    }
    ablation_labels = {t.trial_label for t in manifest.trials if t.regime_kind.value == "NONE"}
    changed = {lb for lb in prior if prior[lb]["experiment_identity"] != new[lb]["experiment_identity"]}
    unchanged = {lb for lb in prior if lb not in changed}
    if changed != baseline_labels:
        raise SystemExit(
            f"expected exactly the 40 regime-baseline identities to move; changed={len(changed)} "
            f"baseline={len(baseline_labels)} (diff {changed ^ baseline_labels})"
        )
    if not ablation_labels <= unchanged:
        raise SystemExit("a NO_REGIME ablation identity changed; the 20 are frozen")
    # composite fingerprint / friendly id: baseline composite moved, ablation not;
    # friendly id and parameter_variant unchanged for BOTH (regime kind + variant
    # unchanged).
    for lb in ablation_labels:
        for k in ("composite_strategy_fingerprint", "parameter_variant_identity", "friendly_experiment_id"):
            if prior[lb][k] != new[lb][k]:
                raise SystemExit(f"ablation trial {lb}: {k} changed")
    for lb in baseline_labels:
        if prior[lb]["parameter_variant_identity"] != new[lb]["parameter_variant_identity"]:
            raise SystemExit(f"baseline trial {lb}: parameter_variant_identity changed (regime is NOT a variant)")
        if prior[lb]["composite_strategy_fingerprint"] == new[lb]["composite_strategy_fingerprint"]:
            raise SystemExit(f"baseline trial {lb}: composite fingerprint did not move")

    # -- counts unchanged --------------------------------------------------
    est = manifest.compute_estimate
    drift = {k: (v, getattr(est, k)) for k, v in FROZEN_COUNTS.items() if getattr(est, k) != v}
    if drift:
        raise SystemExit(f"a frozen research count moved: {drift}")
    mt = spec.multiple_testing
    if mt.n_trials != 60 or mt.dsr_effective_trial_count != 380 or (
        dict(mt.dsr_effective_trial_count_by_model_family) != FROZEN_DSR_BREADTH_BY_FAMILY
    ):
        raise SystemExit("BH 60 / DSR 380 / per-family 3-8 must not change")

    # -- the mandatory pre-run registry query for every trial --------------
    registry = ExperimentRegistry(DEFAULT_REGISTRY_PATH)
    preflight = phase_15_registry_preflight(registry, rows)
    if not preflight["clear_to_run"]:
        raise SystemExit(
            f"the registry already holds one of these experiments: {preflight['duplicate_trial_labels']}"
        )

    # -- write the corrected manifest + artifacts -------------------------
    freeze_candidate_manifest(str(MANIFEST_15B1B))
    commit, now = _git_commit(), datetime.now(UTC).isoformat()
    plane_payload = planes_payload(planes, sources)

    accounting = trial_accounting_payload(manifest)
    accounting["phase"] = "15B.1b"
    accounting["regime_transformation"] = regime_id
    _write(OUT_DIR / "PRETRAIN_TRIAL_ACCOUNTING_REGIME_CORRECTED.json", accounting)

    _write(
        OUT_DIR / "PRERUN_IDENTITY_MAP_COMPLETE_REGIME_CORRECTED.json",
        {
            "phase": "15B.1b",
            "generated_at": now,
            "code_commit": commit,
            "supersedes": (
                "outputs/phase_15/PRERUN_IDENTITY_MAP_COMPLETE_ADJUDICATION_CORRECTED.json "
                "(15B.0), preserved"
            ),
            "manifest_fingerprint": manifest.manifest_fingerprint(),
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "identity_schema": "experiment-identity/2",
            "identity_is_pre_run": True,
            "n_predeclared_hypotheses": len(rows),
            "n_complete_pre_run_identities": len({r["experiment_identity"] for r in rows}),
            "identities_per_hypothesis": 1,
            "bh_fdr_denominator": mt.n_trials,
            "regime_transformation": regime_id,
            "regime_baseline_identity": REGIME_BASELINE.identity(),
            "regime_ablation_identity": REGIME_ABLATION.identity(),
            "identity_planes": plane_payload["planes"],
            "what_changed_from_15b0": {
                "n_regime_baseline_identities_regenerated": len(changed),
                "n_no_regime_ablation_identities_byte_identical": len(ablation_labels),
                "validation_spec_fingerprint_unchanged": True,
                "bh_family": 60,
                "dsr_family_breadth": 380,
                "dsr_per_family": FROZEN_DSR_BREADTH_BY_FAMILY,
            },
            "regime_baseline_trial_labels_regenerated": sorted(changed),
            "no_regime_ablation_trial_labels_unchanged": sorted(ablation_labels),
            "trials": list(rows),
        },
    )

    _write(
        OUT_DIR / "PRERUN_REGISTRY_PREFLIGHT_REGIME_CORRECTED.json",
        {
            "phase": "15B.1b",
            "generated_at": now,
            "code_commit": commit,
            "supersedes": "outputs/phase_15/PRERUN_REGISTRY_PREFLIGHT_ADJUDICATION_CORRECTED.json (15B.0)",
            "registry_path": str(DEFAULT_REGISTRY_PATH),
            "registry_schema_version": registry.summary().schema_version,
            "identity_schema": "experiment-identity/2",
            "ran_before_any_fit": True,
            **preflight,
        },
    )

    protocol = {
        "phase": "15B.1b",
        "status": "REGIME_TRANSFORMATION_FROZEN",
        "generated_at": now,
        "code_commit": commit,
        "manifest_fingerprint": manifest.manifest_fingerprint(),
        "supersedes_manifest_fingerprint": SUPERSEDED_15A2_MANIFEST_FINGERPRINT,
        "regime_transformation": regime_id,
        "fingerprints": {
            "regime_baseline": REGIME_BASELINE.identity(),
            "regime_ablation": REGIME_ABLATION.identity(),
            "regime_baseline_prior_superseded": PRIOR_REGIME_BASELINE_IDENTITY,
            "validation_spec": spec.validation_fingerprint(),
            "reliability_policy": policy.identity(),
            **{f"plane__{k}": v for k, v in plane_payload["planes"].items()},
        },
        "frozen_counts": FROZEN_COUNTS,
        "dsr_effective_trial_count_by_model_family": FROZEN_DSR_BREADTH_BY_FAMILY,
        "identity_delta": {
            "n_regime_baseline_identities_regenerated": len(changed),
            "n_no_regime_ablation_identities_byte_identical": len(ablation_labels),
        },
        "registry_preflight_clear_to_run": preflight["clear_to_run"],
        "six_identity_planes_unchanged": True,
        "one_predeclared_transformation_not_nine_hypotheses": True,
        "decided_by_human_not_from_performance": True,
        "environment": ml_environment_provenance(),
        "holdout_declaration": (
            "the locked final holdout was never downloaded, queried, cost-fetched, loaded, "
            "featured, labelled, trained on, scored or inspected in any Phase 15 step"
        ),
        "no_real_model_trained": True,
        "no_market_data_network_access": True,
        "no_new_dependencies": True,
    }
    _write(OUT_DIR / "PROTOCOL_FREEZE_REGIME_PLANE.json", protocol)

    _write(
        OUT_DIR / "REGIME_SEMANTIC_SUPERSESSION.json",
        {
            "phase": "15B.1b",
            "generated_at": now,
            "code_commit": commit,
            "corrected_before_any_real_ml_training_or_performance": True,
            "decided_by_human": True,
            "superseded": {
                "artifact": SUPERSEDED_15A2_MANIFEST_PATH,
                "manifest_fingerprint": SUPERSEDED_15A2_MANIFEST_FINGERPRINT,
                "commits": list(SUPERSEDED_15A2_MANIFEST_COMMITS),
                "regime_baseline_identity": PRIOR_REGIME_BASELINE_IDENTITY,
                "status": "SUPERSEDED for regime-transformation-semantics purposes",
                "preserved": "unchanged on disk and in git history; never amended, never deleted",
            },
            "corrected": {
                "artifact": "data/manifests/phase_15/ml_candidate_manifest_regime_corrected.json",
                "manifest_fingerprint": manifest.manifest_fingerprint(),
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "regime_baseline_identity": REGIME_BASELINE.identity(),
            },
            "reason": SUPERSESSION_15B1B_REASON,
            "identity_delta": {
                "n_trials": len(rows),
                "n_regime_baseline_identities_changed": len(changed),
                "n_no_regime_ablation_identities_unchanged": len(unchanged),
                "n_labels_changed": len(set(prior) ^ set(new)),
                "bh_family": 60,
                "dsr_family_breadth": 380,
            },
            "regime_transformation_fingerprint": fingerprint(
                "regimefreeze1", {"regime": regime_id, "identity": REGIME_BASELINE.identity()}
            ),
        },
    )

    print(f"manifest schema      : {manifest.schema_version}")
    print(f"manifest fp          : {manifest.manifest_fingerprint()}")
    print(f"  (15A.2 superseded) : {SUPERSEDED_15A2_MANIFEST_FINGERPRINT}")
    print(f"regime baseline id   : {REGIME_BASELINE.identity()}")
    print(f"  (prior superseded) : {PRIOR_REGIME_BASELINE_IDENTITY}")
    print(f"regime ablation id   : {REGIME_ABLATION.identity()}  (UNCHANGED)")
    print(f"regime states        : {REGIME_BASELINE.n_states} -> {REGIME_BASELINE.output_columns()}")
    print(f"validation spec fp   : {spec.validation_fingerprint()}  (UNCHANGED)")
    print(f"identities changed   : {len(changed)} (all regime-baseline)")
    print(f"identities unchanged : {len(unchanged)} (all NO_REGIME ablation)")
    print(f"BH / DSR             : {mt.n_trials} / {mt.dsr_effective_trial_count} "
          f"({dict(mt.dsr_effective_trial_count_by_model_family)})")
    print(f"registry preflight   : {preflight['n_trials_queried']} queried, "
          f"{preflight['n_exact_duplicates']} duplicates, clear={preflight['clear_to_run']}")
    for name in (
        "PRETRAIN_TRIAL_ACCOUNTING_REGIME_CORRECTED.json",
        "PRERUN_IDENTITY_MAP_COMPLETE_REGIME_CORRECTED.json",
        "PRERUN_REGISTRY_PREFLIGHT_REGIME_CORRECTED.json",
        "PROTOCOL_FREEZE_REGIME_PLANE.json",
        "REGIME_SEMANTIC_SUPERSESSION.json",
    ):
        print(f"wrote {(OUT_DIR / name).relative_to(REPO)}")
    print(f"wrote {MANIFEST_15B1B.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
