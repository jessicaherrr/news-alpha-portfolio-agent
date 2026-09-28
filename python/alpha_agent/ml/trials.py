"""From a frozen Phase 15 manifest trial to its PRE-RUN registry identity.

This is the bridge the registry workflow needs and that Phase 15A/15A.1 never
built: given one predeclared :class:`~alpha_agent.ml.manifest.MLCandidateTrial`,
deterministically reconstruct the :class:`~alpha_agent.ml.identity.MLExperimentSpec`
whose ``experiment_identity()`` a Research Agent must look up **before** spending
any compute.

Everything here is configuration reconstruction: it reads no market data, fits no
model, compiles no schedule and inspects no performance. The primary strategies
are rebuilt with the frozen Phase 13.5C factories
(:func:`~alpha_agent.strategy.candidates_phase_13_5c.baseline_spec`), exactly as
the Phase 14.2 rule requires -- a variant's identity uses its OWN reconstructed
semantics, never an inherited copy of a neighbour's.

The two levels, and why both exist
----------------------------------
``experiment_identity`` needs six plane fingerprints that Phase 15 *inherits*
(dataset, execution, cost, risk) or that Phase 15B will *fix* (the Phase 15
``ValidationSpec`` and ``ReliabilityPolicy``). Those do not exist yet, and
inventing them would be fabricated provenance. So identity work splits in two:

* :func:`trial_discriminator` -- the part of the identity that VARIES across the
  60 trials. It is fully determined today, and because the six planes are shared
  by all 60 trials, two trials collide in ``experiment_identity`` **iff** they
  collide here. Uniqueness is therefore provable now, for every possible future
  plane assignment.
* :func:`experiment_spec_for_trial` -- the complete spec, once a caller supplies
  the planes as an explicit :class:`MLIdentityPlanes`. There is no default: a
  plane fingerprint is either known and passed in, or the caller has no business
  claiming an identity.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from alpha_agent.ml.identity import (
    MLExperimentSpec,
    ml_trial_discriminator,
)
from alpha_agent.ml.manifest import (
    META_LABEL_SPEC,
    ML_FEATURE_SET,
    MODEL_SEARCHES,
    NESTED_CV,
    PREPROCESSING_SPEC,
    REGIME_ABLATION,
    REGIME_BASELINE,
    MLCandidateManifest,
    MLCandidateTrial,
)
from alpha_agent.ml.regime import RegimeSpec
from alpha_agent.strategy.candidates_phase_13_5c import baseline_spec
from alpha_agent.strategy.fingerprint import strategy_fingerprint

#: Placeholder plane values are deliberately absent from this module. If a plane
#: fingerprint is unknown, the identity is unknown; see the module docstring.
IDENTITY_PLANES_SCHEMA = "ml-identity-planes/1"


class MLIdentityPlanes(BaseModel):
    """The six identity components a Phase 15 trial does not itself determine.

    All are required. Four are inherited unchanged from Phase 13.5C (dataset,
    execution, cost, risk) and two are fixed by Phase 15B (the Phase 15
    ``ValidationSpec`` and ``ReliabilityPolicy``). None is guessed here.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = IDENTITY_PLANES_SCHEMA
    dataset_fingerprint: str = Field(min_length=1)
    validation_spec_fingerprint: str = Field(min_length=1)
    reliability_policy_fingerprint: str = Field(min_length=1)
    execution_config_identity: str = Field(min_length=1)
    cost_config_identity: str = Field(min_length=1)
    risk_identity: str = Field(min_length=1)


def regime_for_trial(trial: MLCandidateTrial) -> RegimeSpec:
    """The frozen regime object a trial's declared ``regime_kind`` names."""
    for candidate in (REGIME_BASELINE, REGIME_ABLATION):
        if candidate.kind is trial.regime_kind:
            return candidate
    raise ValueError(
        f"trial {trial.trial_label!r} declares regime kind {trial.regime_kind.value}, which is "
        "not one of the two frozen Phase 15 regime designs"
    )


def primary_strategy_fingerprints(
    primary_family: str, training_roots: tuple[str, ...]
) -> tuple[str, ...]:
    """Rebuild the pooled scope's primary ``StrategySpec`` fingerprints.

    A pooled trial trains one model over every root's primary, so the whole set
    of primaries is part of what the composite strategy IS. Rebuilt from the
    frozen Phase 13.5C canonical parameters and factories -- configuration only.
    """
    return tuple(
        sorted(strategy_fingerprint(baseline_spec(primary_family, root)) for root in training_roots)
    )


def experiment_spec_for_trial(
    trial: MLCandidateTrial,
    *,
    planes: MLIdentityPlanes,
    take_threshold: float | None = None,
) -> MLExperimentSpec:
    """The full PRE-RUN :class:`MLExperimentSpec` for one predeclared trial.

    Constructible before any fit: every input is frozen configuration. In
    particular the model side is the whole predeclared search procedure, never a
    hyperparameter point -- nested CV has not selected one yet, and never will
    select just one.
    """
    search = MODEL_SEARCHES[trial.model_family]
    if trial.model_search_identity and trial.model_search_identity != search.identity():
        raise ValueError(
            f"trial {trial.trial_label!r} declares model search {trial.model_search_identity} "
            f"but the frozen search for {trial.model_family.value} is {search.identity()}; the "
            "declared search and the executed search must be the same procedure"
        )
    kwargs: dict[str, object] = {
        "label": trial.trial_label,
        "primary_strategy_family": trial.primary_family,
        "primary_strategy_fingerprints": primary_strategy_fingerprints(
            trial.primary_family, trial.training_roots
        ),
        "root_symbol": trial.root_symbol,
        "training_roots": trial.training_roots,
        "meta_label_spec": META_LABEL_SPEC,
        "feature_set": ML_FEATURE_SET,
        "regime": regime_for_trial(trial),
        "preprocessing": PREPROCESSING_SPEC,
        "model_search": search,
        "nested_cv": NESTED_CV,
        "trial_role": trial.trial_role,
        "multiple_testing_family_id": trial.multiple_testing_family_id,
        "dataset_fingerprint": planes.dataset_fingerprint,
        "validation_spec_fingerprint": planes.validation_spec_fingerprint,
        "reliability_policy_fingerprint": planes.reliability_policy_fingerprint,
        "execution_config_identity": planes.execution_config_identity,
        "cost_config_identity": planes.cost_config_identity,
        "risk_identity": planes.risk_identity,
    }
    if take_threshold is not None:
        kwargs["take_threshold"] = take_threshold
    else:
        kwargs["take_threshold"] = trial.take_threshold
    return MLExperimentSpec(**kwargs)


def trial_discriminator(trial: MLCandidateTrial) -> str:
    """The trial-varying part of the pre-run identity, computable today.

    Uses a single fixed placeholder plane assignment purely as a hashing pivot:
    the planes cancel out of the comparison because every trial uses the same
    ones, and none of these strings is recorded anywhere as provenance.
    """
    return ml_trial_discriminator(experiment_spec_for_trial(trial, planes=_PIVOT_PLANES))


#: A single shared, obviously non-real plane assignment used ONLY as the pivot in
#: :func:`trial_discriminator`. ``ml_trial_discriminator`` hashes no plane, so
#: these values never reach a fingerprint or an artifact; they exist because
#: :class:`MLIdentityPlanes` refuses to be empty, which is the property we want.
_PIVOT_PLANES = MLIdentityPlanes(
    dataset_fingerprint="PENDING_PHASE_15B",
    validation_spec_fingerprint="PENDING_PHASE_15B",
    reliability_policy_fingerprint="PENDING_PHASE_15B",
    execution_config_identity="PENDING_PHASE_15B",
    cost_config_identity="PENDING_PHASE_15B",
    risk_identity="PENDING_PHASE_15B",
)


def trial_identity_rows(manifest: MLCandidateManifest) -> tuple[dict, ...]:
    """One pre-run identity row per predeclared trial, in manifest order."""
    rows: list[dict] = []
    for trial in manifest.trials:
        spec = experiment_spec_for_trial(trial, planes=_PIVOT_PLANES)
        rows.append(
            {
                "trial_label": trial.trial_label,
                "primary_family": trial.primary_family,
                "root_symbol": trial.root_symbol,
                "training_roots": list(trial.training_roots),
                "model_family": trial.model_family.value,
                "regime_kind": trial.regime_kind.value,
                "trial_role": trial.trial_role.value,
                "strategy_family": spec.strategy_family,
                "primary_strategy_fingerprints": list(spec.primary_strategy_fingerprints),
                "composite_strategy_fingerprint": spec.composite_fingerprint(),
                "model_search_identity": spec.model_search.identity(),
                "n_search_configurations": spec.model_search.n_configurations,
                "selection_objective": spec.model_search.objective_value,
                "selection_tie_break": spec.model_search.tie_break_value,
                "model_search_random_seed": spec.model_search.random_seed,
                "take_threshold": spec.take_threshold,
                "parameter_variant_identity": spec.parameter_variant_identity(),
                "parameter_variant_label": spec.parameter_variant_label(),
                "split_identity": spec.nested_cv.identity(),
                "feature_spec_fingerprint": spec.feature_set.identity(),
                "trial_discriminator": ml_trial_discriminator(spec),
                "friendly_experiment_id": spec.friendly_id(),
            }
        )
    return tuple(rows)


def assert_one_identity_per_trial(manifest: MLCandidateManifest) -> tuple[dict, ...]:
    """Exactly one unique pre-run identity per predeclared trial, no collisions.

    The invariant Phase 15A.2 exists to restore: 60 scientific hypotheses map to
    60 registry-facing identities, not to 380 (one per hyperparameter point) and
    not to fewer than 60 (two hypotheses sharing a row).
    """
    rows = trial_identity_rows(manifest)
    discriminators = [r["trial_discriminator"] for r in rows]
    if len(set(discriminators)) != len(rows):
        seen: dict[str, str] = {}
        clashes = []
        for r in rows:
            d = r["trial_discriminator"]
            if d in seen:
                clashes.append((seen[d], r["trial_label"]))
            seen[d] = r["trial_label"]
        raise ValueError(
            f"Phase 15 trials collide in pre-run identity: {clashes}. Two distinct predeclared "
            "economic hypotheses would share one registry row and one BH trial."
        )
    if len(rows) != manifest.n_trials():
        raise ValueError("identity rows and manifest trials disagree in count")
    return rows
