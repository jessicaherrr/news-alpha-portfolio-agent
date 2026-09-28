"""The six Phase 15 identity planes, and the 60 complete PRE-RUN identities.

Phase 15A.2 froze 60 distinct trial *discriminators* and deliberately stopped
there: an ``experiment_identity`` also needs six plane fingerprints, two of which
(the Phase 15 ``ValidationSpec`` and ``ReliabilityPolicy``) did not exist, and a
fabricated plane value is worse than an absent one. Phase 15A.3 supplies them:

======================================  ==================================
plane                                   where this module gets it
======================================  ==================================
``dataset_fingerprint``                 the five roots' Phase 13.5C dataset
                                        identities, pooled -- read from the
                                        committed validation reports
``validation_spec_fingerprint``         :mod:`alpha_agent.ml.adjudication`
``reliability_policy_fingerprint``      :mod:`alpha_agent.ml.adjudication`
``execution_config_identity``           Phase 13.5C, unchanged, recomputed
                                        from the committed manifest + the
                                        contract-economics QA artifact
``cost_config_identity``                Phase 13.5C, unchanged
``risk_identity``                       Phase 13.5C, unchanged
======================================  ==================================

Everything here is artifact reconstruction: it reads committed JSON, computes
fingerprints, and touches no market data, no network, no 2025 and no model.

Why the dataset plane is POOLED
-------------------------------
A Phase 15 pipeline trains ONE model over ES/NQ/CL/GC/ZN with root as a
categorical feature, so the data a Phase 15 hypothesis is about is all five
roots' bars, not the one root whose economics are later adjudicated. The plane
is therefore a fingerprint over the five committed per-root Phase 13.5C dataset
identities. No new data and no new hashing of bars: Phase 13.5C's execution bars
span ``2018-01-01`` to ``2025-01-01`` exclusive, which is exactly the Phase 15
``PHASE_15_DEVELOPMENT_CORPUS``, so "the Phase 13.5C dataset identity, unchanged"
is literally true per root and this plane only records that all five are in
scope at once.

Because every trial shares this plane -- as it shares the other five -- the
Phase 15A.2 uniqueness proof carries over unchanged: two trials collide in
``experiment_identity`` if and only if they collide in ``trial_discriminator``.
"""
from __future__ import annotations

import json
from pathlib import Path

from alpha_agent.ml.adjudication import build_phase_15_validation_spec
from alpha_agent.ml.guards import (
    DEVELOPMENT_CORPUS_END_NS,
    DEVELOPMENT_CORPUS_START_NS,
    assert_exclusive_corpus_bound,
    assert_no_holdout_in_config_payload,
)
from alpha_agent.ml.manifest import ROOTS, MLCandidateManifest, build_candidate_manifest
from alpha_agent.ml.trials import (
    MLIdentityPlanes,
    experiment_spec_for_trial,
    trial_identity_rows,
)
from alpha_agent.registry.enums import AssetDomain
from alpha_agent.registry.phase_13_5c_import import (
    Phase135cSources,
    _config_identities,
)
from alpha_agent.validation.fingerprint import fingerprint

ML_DATASET_PLANE_SCHEMA = "ml-dataset-plane/1"

#: The Phase 13.5C families whose committed validation reports carry the
#: per-root dataset identity. Any one of them serves -- the dataset identity is a
#: property of the ROOT's bars, not of the strategy -- and reading them all is
#: what proves that, so a per-root disagreement fails loudly instead of being
#: resolved by whichever file happened to be read first.
_PHASE_13_5C_BASELINE_FAMILIES = ("tsmom", "ma_trend", "breakout", "mean_reversion")


def repo_root() -> Path:
    """The repository root, from this file's location."""
    return Path(__file__).resolve().parents[3]


def phase_13_5c_dataset_fingerprints(sources: Phase135cSources) -> dict[str, str]:
    """One committed Phase 13.5C dataset identity per Phase 15 training root.

    Reads every baseline family's report for the root and refuses a
    disagreement: the dataset identity is a property of the root's execution
    bars, so all four families must record the same value, and if they ever did
    not, choosing one would be choosing which data the experiment was about.
    """
    out: dict[str, str] = {}
    for root in ROOTS:
        seen: dict[str, str] = {}
        for family in _PHASE_13_5C_BASELINE_FAMILIES:
            fp = sources.validation_report(root, family)["fingerprints"]["dataset_fingerprint"]
            seen[family] = fp
        distinct = sorted(set(seen.values()))
        if len(distinct) != 1:
            raise ValueError(
                f"the committed Phase 13.5C validation reports disagree about {root}'s "
                f"dataset identity: {seen}. The dataset plane is a property of the root's "
                "execution bars; a disagreement is a data-provenance failure, not a choice."
            )
        out[root] = distinct[0]
    return out


def ml_dataset_plane_fingerprint(
    per_root_dataset_fingerprints: dict[str, str],
    *,
    corpus_start_ts_ns: int = DEVELOPMENT_CORPUS_START_NS,
    corpus_end_ts_ns: int = DEVELOPMENT_CORPUS_END_NS,
) -> str:
    """Identity of the POOLED Phase 15 dataset: all five roots, one corpus."""
    assert_exclusive_corpus_bound(corpus_end_ts_ns, what="ml_dataset_plane.corpus_end_ts_ns")
    if sorted(per_root_dataset_fingerprints) != sorted(ROOTS):
        raise ValueError(
            f"the pooled Phase 15 dataset plane needs exactly the training roots "
            f"{sorted(ROOTS)}; got {sorted(per_root_dataset_fingerprints)}"
        )
    return fingerprint(
        "mldatasetplane1",
        {
            "schema": ML_DATASET_PLANE_SCHEMA,
            "corpus": [corpus_start_ts_ns, corpus_end_ts_ns],
            "per_root_dataset_identity": {
                r: per_root_dataset_fingerprints[r] for r in sorted(per_root_dataset_fingerprints)
            },
            "pooling": "one model trained over all roots with root as a categorical feature",
        },
    )


def phase_15_identity_planes(
    repo: Path | None = None, *, manifest: MLCandidateManifest | None = None
) -> MLIdentityPlanes:
    """The six real Phase 15 identity planes. No placeholder survives here."""
    sources = Phase135cSources(repo if repo is not None else repo_root())
    ids = _config_identities(sources)
    spec = build_phase_15_validation_spec(manifest)
    return MLIdentityPlanes(
        dataset_fingerprint=ml_dataset_plane_fingerprint(
            phase_13_5c_dataset_fingerprints(sources)
        ),
        validation_spec_fingerprint=spec.validation_fingerprint(),
        reliability_policy_fingerprint=spec.reliability_policy_fingerprint,
        execution_config_identity=ids["execution_corrected"],
        cost_config_identity=ids["cost"],
        risk_identity=ids["risk"],
    )


def assert_planes_are_real(planes: MLIdentityPlanes) -> None:
    """No plane may carry a placeholder, a TODO or an empty string.

    The Phase 15A.2 pivot value ``PENDING_PHASE_15B`` exists only as a hashing
    pivot inside ``trial_discriminator``; if one ever reached a recorded
    identity it would be exactly the fabricated provenance CLAUDE.md forbids.
    """
    forbidden = ("PENDING", "TODO", "PLACEHOLDER", "UNKNOWN", "FIXME", "TBD")
    for field in (
        "dataset_fingerprint",
        "validation_spec_fingerprint",
        "reliability_policy_fingerprint",
        "execution_config_identity",
        "cost_config_identity",
        "risk_identity",
    ):
        value = str(getattr(planes, field))
        upper = value.upper()
        if any(token in upper for token in forbidden):
            raise ValueError(
                f"Phase 15 identity plane {field}={value!r} is a placeholder. A plane "
                "fingerprint is either known or the identity is not claimable; it is never "
                "invented."
            )
        if ":" not in value:
            raise ValueError(
                f"Phase 15 identity plane {field}={value!r} is not a "
                "'<prefix>:<sha256>' fingerprint"
            )


def phase_15_pre_run_identity_rows(
    *,
    planes: MLIdentityPlanes,
    manifest: MLCandidateManifest | None = None,
) -> tuple[dict, ...]:
    """One COMPLETE pre-run registry identity per predeclared Phase 15 trial.

    Extends the Phase 15A.2 discriminator rows with the full
    ``experiment_identity`` now that every plane is real. Still pre-run: no
    market data is read, no schedule is compiled and no model is fitted, which
    is precisely what makes the registry duplicate query worth doing first.
    """
    assert_planes_are_real(planes)
    manifest = manifest if manifest is not None else build_candidate_manifest()
    discriminator_rows = {r["trial_label"]: r for r in trial_identity_rows(manifest)}
    rows: list[dict] = []
    for trial in manifest.trials:
        spec = experiment_spec_for_trial(trial, planes=planes)
        row = dict(discriminator_rows[trial.trial_label])
        row.update(
            {
                "experiment_identity": spec.experiment_identity(),
                "friendly_experiment_id": spec.friendly_id(),
                "multiple_testing_family_id": spec.multiple_testing_family_id,
                "dataset_fingerprint": planes.dataset_fingerprint,
                "validation_spec_fingerprint": planes.validation_spec_fingerprint,
                "reliability_policy_fingerprint": planes.reliability_policy_fingerprint,
                "execution_config_identity": planes.execution_config_identity,
                "cost_config_identity": planes.cost_config_identity,
                "risk_identity": planes.risk_identity,
                "target_schedule_hash": None,      # POST-RUN provenance, never identity
                "report_fingerprint": None,        # POST-RUN provenance, never identity
            }
        )
        rows.append(row)
    return tuple(rows)


def assert_sixty_complete_identities(rows: tuple[dict, ...], *, expected: int = 60) -> None:
    """The invariant Phase 15A.3 exists to establish.

    ``expected`` predeclared hypotheses produce ``expected`` complete, distinct,
    truthful pre-run identities -- the same number as the BH/FDR denominator, so
    the registry and the multiple-testing family agree by construction rather
    than by reconciliation.
    """
    if len(rows) != expected:
        raise ValueError(f"expected {expected} predeclared Phase 15 trials, got {len(rows)}")
    identities = [r["experiment_identity"] for r in rows]
    if any(not i or not str(i).startswith("experiment1:") for i in identities):
        raise ValueError("every Phase 15 trial must carry a schema-v2 experiment identity")
    if len(set(identities)) != expected:
        seen: dict[str, str] = {}
        clashes = []
        for r in rows:
            i = r["experiment_identity"]
            if i in seen:
                clashes.append((seen[i], r["trial_label"]))
            seen[i] = r["trial_label"]
        raise ValueError(
            f"Phase 15 trials collide in complete pre-run identity: {clashes}. Two distinct "
            "predeclared economic hypotheses would share one registry row and one BH trial."
        )
    if len({r["trial_discriminator"] for r in rows}) != expected:
        raise ValueError(
            "discriminators and identities disagree in cardinality; the shared-plane premise "
            "of the Phase 15A.2 uniqueness proof no longer holds"
        )
    for r in rows:
        if r["target_schedule_hash"] is not None or r["report_fingerprint"] is not None:
            raise ValueError(
                f"{r['trial_label']}: target_schedule_hash / report_fingerprint are POST-RUN "
                "provenance and must be NULL before the run, never back-filled to look complete"
            )


def phase_15_registry_preflight(
    registry, rows: tuple[dict, ...], *, top_k_related: int = 5
) -> dict:
    """The mandatory pre-run registry query for every predeclared trial.

    Read-only. Answers "has this already been run?" from the proposal alone --
    no fit, no schedule, no market data -- which is the whole point of a pre-run
    identity.

    Schema v5 distinguishes:

    * identity present **with a VALID authoritative result** -> this is a genuine
      duplicate; cite the prior result, never silently re-run (``clear_to_run``
      is False, the trial is a ``blocking_duplicate``);
    * identity present **with only INVALID_EXECUTION attempts** (an earlier run
      failed for an engineering / data-pipeline / software / infrastructure
      reason) -> re-execution is permitted (the trial is ``re_executable``);
    * identity absent -> clear to run.
    """
    results: list[dict] = []
    blocking: list[str] = []
    re_executable: list[str] = []
    for row in rows:
        identity = row["experiment_identity"]
        # Phase 15's ML trials train only on Futures roots (ROOTS, above) -- a
        # stated fact, not an inference (Phase 6 ETF Research Pilot).
        dup = registry.find_exact_duplicate(identity, asset_domain=AssetDomain.FUTURES)
        related = registry.find_related(
            strategy_family=row["strategy_family"],
            root_symbol=row["root_symbol"],
            asset_domain=AssetDomain.FUTURES,
            params={
                "model_family": row["model_family"],
                "n_search_configurations": row["n_search_configurations"],
                "selection_objective": row["selection_objective"],
                "selection_tie_break": row["selection_tie_break"],
                "random_seed": row["model_search_random_seed"],
                "regime_kind": row["regime_kind"],
                "take_threshold": row["take_threshold"],
            },
            top_k=top_k_related,
        )
        blocks = bool(getattr(dup, "blocks_reexecution", dup.exists))
        invalid_only = bool(
            dup.exists and not getattr(dup, "has_valid_authoritative_result", True)
        )
        if blocks:
            blocking.append(row["trial_label"])
        elif invalid_only:
            re_executable.append(row["trial_label"])
        results.append(
            {
                "trial_label": row["trial_label"],
                "experiment_identity": identity,
                "is_exact_duplicate": bool(dup.exists),
                "blocks_reexecution": blocks,
                "has_valid_authoritative_result": bool(
                    getattr(dup, "has_valid_authoritative_result", dup.exists)
                ),
                "invalid_only_history": invalid_only,
                "n_prior_attempts": int(getattr(dup, "n_attempts", 1 if dup.exists else 0)),
                "n_invalid_prior_attempts": int(getattr(dup, "n_invalid_attempts", 0)),
                "duplicate_experiment_id": dup.experiment_id if dup.exists else None,
                "n_related": len(related),
                "top_related": [
                    {
                        "experiment_id": r.experiment_id,
                        "strategy_family": r.strategy_family,
                        "root_symbol": r.root_symbol,
                        "score": round(float(r.similarity.score), 6),
                        "headline_verdict": (
                            r.headline_verdict.value
                            if r.headline_verdict is not None
                            else None
                        ),
                        "reason_codes": list(r.reason_codes),
                    }
                    for r in related[:top_k_related]
                ],
            }
        )
    n_present = sum(1 for r in results if r["is_exact_duplicate"])
    return {
        "n_trials_queried": len(rows),
        "n_exact_duplicates": n_present,
        "n_blocking_duplicates": len(blocking),
        "n_re_executable_invalid_only": len(re_executable),
        "duplicate_trial_labels": blocking,          # only VALID-result duplicates block
        "blocking_duplicate_trial_labels": blocking,
        "re_executable_trial_labels": re_executable,
        "clear_to_run": not blocking,
        "registry_experiments_total": registry.count_experiments(),
        "trials": results,
    }


def planes_payload(planes: MLIdentityPlanes, sources: Phase135cSources) -> dict:
    """A committed, auditable record of where every plane value came from."""
    per_root = phase_13_5c_dataset_fingerprints(sources)
    payload = {
        "schema_version": planes.schema_version,
        "planes": {
            "dataset_fingerprint": planes.dataset_fingerprint,
            "validation_spec_fingerprint": planes.validation_spec_fingerprint,
            "reliability_policy_fingerprint": planes.reliability_policy_fingerprint,
            "execution_config_identity": planes.execution_config_identity,
            "cost_config_identity": planes.cost_config_identity,
            "risk_identity": planes.risk_identity,
        },
        "provenance": {
            "dataset_fingerprint": {
                "kind": "POOLED Phase 13.5C per-root dataset identities, unchanged",
                "per_root": per_root,
                "corpus_start_ts_ns": DEVELOPMENT_CORPUS_START_NS,
                "corpus_end_ts_ns": DEVELOPMENT_CORPUS_END_NS,   # EXCLUSIVE bound
                "source_artifacts": [
                    f"outputs/phase_13_5c/{root}__{family.upper()}__validation_report.json"
                    for root in sorted(per_root)
                    for family in _PHASE_13_5C_BASELINE_FAMILIES
                ],
                "note": (
                    "Phase 13.5C execution bars span the corpus recorded above -- from "
                    "2018-01-01 up to the exclusive locked-holdout boundary -- which is "
                    "exactly the PHASE_15_DEVELOPMENT_CORPUS, so each per-root identity is "
                    "the Phase 13.5C one unchanged. Pooling records that a Phase 15 model "
                    "is trained over all five roots at once."
                ),
            },
            "validation_spec_fingerprint": {
                "kind": "NEW in Phase 15A.3: the family-wide Phase 15 MLValidationSpec",
                "module": "alpha_agent.ml.adjudication.build_phase_15_validation_spec",
            },
            "reliability_policy_fingerprint": {
                "kind": "INHERITED UNCHANGED from Phase 13/13.5C",
                "module": "alpha_agent.ml.adjudication.phase_15_reliability_policy",
                "equals_phase_13_5c_committed_value": True,
            },
            "execution_config_identity": {
                "kind": "Phase 13.5C corrected execution plane, unchanged",
                "source_artifacts": [
                    "data/manifests/phase_13_5c/candidate_manifest.json",
                    "outputs/phase_13_5c/CONTRACT_ECONOMICS_QA.json",
                ],
            },
            "cost_config_identity": {
                "kind": "Phase 13.5C cost plane, unchanged",
                "source_artifacts": ["data/manifests/phase_13_5c/candidate_manifest.json"],
            },
            "risk_identity": {
                "kind": "Phase 13.5C risk plane, unchanged (PassThroughRiskManager)",
                "source_artifacts": ["data/manifests/phase_13_5c/candidate_manifest.json"],
            },
        },
    }
    assert_no_holdout_in_config_payload(
        json.loads(json.dumps(payload)),
        exclusive_bound_keys=frozenset({"corpus_end_ts_ns"}),
        half_open_window_keys=frozenset(),
        path="phase_15_identity_planes",
    )
    return payload
