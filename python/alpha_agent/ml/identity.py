"""Pre-run identity for a Phase 15 ML experiment, in the EXISTING registry.

Phase 15 creates no ML registry. It reuses ``data/registry/experiments.sqlite``
(schema v4, experiment identity schema v2) exactly as Phase 14 defined it, so a
Research Agent asks one question of one memory (CLAUDE.md, Phase 14 section).

Mapping onto the Phase 14 identity components
---------------------------------------------
``experiment_identity`` is a PRE-RUN quantity. Every component below is known
from the proposal and its configuration, before a single bar executes, so
``find_exact_duplicate`` answers "already run?" without fitting anything:

===============================  ==========================================
Phase 14 component               Phase 15 value
===============================  ==========================================
``strategy_fingerprint``         :func:`ml_composite_fingerprint` -- the
                                 PRIMARY spec plus the whole meta-label
                                 pipeline. A composite strategy has a
                                 composite identity.
``strategy_family``              ``ml_meta_label.<primary_family>``
``root_symbol``                  the root the ECONOMICS were measured on
``parameter_variant_identity``   the whole PREDECLARED model-search
                                 procedure (:class:`ModelSearchSpec`) + the
                                 FROZEN TAKE threshold (0.50, not a search
                                 dimension) -- never one selected point
``dataset_fingerprint``          the Phase 13.5C dataset identity, unchanged
``split_identity``               the :class:`NestedCVSpec` identity
``validation_spec_fingerprint``  the Phase 15 ValidationSpec
``reliability_policy_...``       the Phase 15 ReliabilityPolicy
``execution_config_identity``    unchanged Phase 13.5C execution plane
``cost_config_identity``         unchanged Phase 13.5C cost plane
``risk_identity``                unchanged Phase 13.5C risk plane
``feature_spec_fingerprint``     the ML feature set's OWN identity
===============================  ==========================================

The Phase 14.2 rule applies unchanged and is the reason
``feature_spec_fingerprint`` is the ML feature set's own: identity uses the
ACTUAL proposed feature semantics, never a neighbour's inherited copy.

``target_schedule_hash`` (the meta-labeled schedule) and ``report_fingerprint``
remain POST-RUN provenance and never enter identity -- otherwise the same ML
hypothesis, recompiled, would masquerade as a new experiment and slip past
duplicate detection.

The search PROCEDURE is identity; the selected point is provenance (15A.2)
-------------------------------------------------------------------------
Phase 15A and 15A.1 built the ML identity around one concrete
:class:`~alpha_agent.ml.models.MLModelSpec` -- a single hyperparameter point.
That is inconsistent with the frozen nested-CV protocol, in three ways that all
matter:

1. **It is not a pre-run quantity.** The hyperparameter is selected *inside each
   outer training fold*, so it does not exist until fitting has happened, and
   ``find_exact_duplicate`` could not be answered before spending the compute --
   which is the entire point of identity schema v2.
2. **There is no single point to name.** Five outer folds may legitimately
   select five different configurations. Naming one of them as "the" identity of
   the experiment states something the protocol never produced.
3. **It shatters the predeclared hypothesis count.** The frozen Phase 15 design
   declares 60 scientific economic hypotheses at the
   ``primary family x root x model family x regime`` level. Keying identity on a
   point would turn each one into ``H`` registry-facing identities -- 380 rows
   for 60 hypotheses -- silently disagreeing with the BH/FDR denominator.

So the pre-run identity carries the *whole frozen model-search procedure*: model
family, the full frozen grid in declaration order, the selection objective, the
tie-break, and the seed. The hyperparameters an outer fold actually chose are
POST-FIT provenance, recorded on every
:class:`~alpha_agent.ml.training.OuterFoldRecord` and in the training report, and
they change no identity. This is the same rule the registry already applies to
``target_schedule_hash``: a quantity that only exists after the run is evidence
about the run, never a component of which experiment it was.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from alpha_agent.ml.enums import MLTrialRole
from alpha_agent.ml.features import MLFeatureSetSpec
from alpha_agent.ml.labels import MetaLabelSpec
from alpha_agent.ml.models import (
    FROZEN_TAKE_THRESHOLD,
    ModelSearchSpec,
    assert_frozen_take_threshold,
)
from alpha_agent.ml.preprocessing import PreprocessingSpec
from alpha_agent.ml.regime import RegimeSpec
from alpha_agent.ml.splits import NestedCVSpec
from alpha_agent.registry.enums import AssetDomain
from alpha_agent.registry.identity import (
    experiment_identity as registry_experiment_identity,
)
from alpha_agent.registry.identity import (
    friendly_experiment_id,
    parameter_variant_identity,
)
from alpha_agent.validation.fingerprint import fingerprint

ML_COMPOSITE_SCHEMA = "ml-composite-strategy/2"
ML_EXPERIMENT_SCHEMA = "ml-experiment-spec/2"
ML_TRIAL_DISCRIMINATOR_SCHEMA = "ml-trial-discriminator/1"

#: Superseded by Phase 15A.2. ``/1`` hashed one selected-looking
#: :class:`~alpha_agent.ml.models.MLModelSpec` point where the search procedure
#: belongs; recorded so a stale artifact is recognisable, never reconstructed.
ML_COMPOSITE_SCHEMA_V1 = "ml-composite-strategy/1"
ML_EXPERIMENT_SCHEMA_V1 = "ml-experiment-spec/1"

#: The Phase 15 multiple-testing family. Deliberately SEPARATE from
#: ``phase_13_5c.all``: the 107 Phase 13 trials are historical and unchanged, and
#: Phase 15 trials are never folded into that denominator (prompt 15A s.13).
PHASE_15_MT_FAMILY = "phase_15_ml.all"


def ml_composite_fingerprint(
    *,
    primary_strategy_fingerprints: tuple[str, ...],
    meta_label_spec: MetaLabelSpec,
    feature_set: MLFeatureSetSpec,
    regime: RegimeSpec,
    preprocessing: PreprocessingSpec,
    model_search: ModelSearchSpec,
    nested_cv: NestedCVSpec,
    take_threshold: float,
) -> str:
    """Identity of the COMPOSITE strategy: primary + meta-label filter.

    ``primary_strategy_fingerprints`` is a tuple because a pooled scope trains
    one model over several roots' primaries; the set of primaries is part of what
    the composite strategy IS, so all of them enter the hash, sorted.

    ``model_search`` is the whole frozen search PROCEDURE, never one selected
    hyperparameter point: under nested CV the point does not exist until an outer
    fold has been fitted, and the composite strategy a Phase 15 trial proposes is
    "the primary, filtered by whatever this predeclared search selects".
    """
    return fingerprint(
        "mlcomposite1",
        {
            "schema": ML_COMPOSITE_SCHEMA,
            "primary_strategy_fingerprints": sorted(primary_strategy_fingerprints),
            "meta_label_spec": meta_label_spec.identity(),
            "feature_set": feature_set.identity(),
            "regime": regime.identity(),
            "preprocessing": preprocessing.identity(),
            "model_search": model_search.identity(),
            "nested_cv": nested_cv.identity(),
            "take_threshold": take_threshold,
        },
    )


def ml_parameter_variant_identity(
    model_search: ModelSearchSpec, *, take_threshold: float
) -> str:
    """The Phase 15 'parameter variant': the SEARCH PROCEDURE plus its threshold.

    Deliberately not a hyperparameter point. The registry's parameter variant
    answers "which configuration of this family was proposed?", and what a Phase
    15 hypothesis proposes is a predeclared search, not a point that only a fold
    can name.
    """
    return parameter_variant_identity(
        {
            "model_family": model_search.family.value,
            "model_search_identity": model_search.identity(),
            "search_grid": [
                {k: h[k] for k in sorted(h)} for h in model_search.search_grid
            ],
            "n_search_configurations": model_search.n_configurations,
            "selection_objective": model_search.objective_value,
            "selection_tie_break": model_search.tie_break_value,
            "random_seed": model_search.random_seed,
            "take_threshold": take_threshold,
        }
    )


class MLExperimentSpec(BaseModel):
    """Everything needed to compute a Phase 15 experiment's PRE-RUN identity.

    Nothing here is a result. The whole object is computable from the proposal,
    which is the point: it is what a Research Agent hashes to ask the registry
    "has this already been run?" before spending any compute.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = ML_EXPERIMENT_SCHEMA
    label: str = ""                                  # cosmetic
    primary_strategy_family: str
    primary_strategy_fingerprints: tuple[str, ...] = Field(min_length=1)
    #: the root whose ECONOMICS this experiment adjudicates
    root_symbol: str
    #: the roots the model was TRAINED over (a pooled scope has several)
    training_roots: tuple[str, ...] = Field(min_length=1)

    meta_label_spec: MetaLabelSpec
    feature_set: MLFeatureSetSpec
    regime: RegimeSpec
    preprocessing: PreprocessingSpec
    #: The frozen model-search PROCEDURE. NOT a selected hyperparameter point:
    #: nested CV selects a point inside each outer training fold, so no point
    #: exists pre-run and different folds may select different ones. What the
    #: hypothesis proposes -- and what identity therefore carries -- is the
    #: family, the full frozen grid, the objective, the tie-break and the seed.
    model_search: ModelSearchSpec
    nested_cv: NestedCVSpec
    #: FROZEN at 0.50. Kept as an explicit identity component -- a decision rule
    #: is part of what the composite strategy IS -- but refused if changed: a
    #: different threshold is a new predeclared decision-policy hypothesis, not a
    #: free parameter of this family.
    take_threshold: float = Field(default=FROZEN_TAKE_THRESHOLD, ge=0.0, le=1.0)
    trial_role: MLTrialRole = MLTrialRole.HEADLINE

    # -- inherited, unchanged, Phase 13.5C planes -----------------------------
    dataset_fingerprint: str
    validation_spec_fingerprint: str
    reliability_policy_fingerprint: str
    execution_config_identity: str
    cost_config_identity: str
    risk_identity: str

    multiple_testing_family_id: str = PHASE_15_MT_FAMILY

    @model_validator(mode="after")
    def _threshold_is_frozen(self) -> MLExperimentSpec:
        assert_frozen_take_threshold(
            self.take_threshold, what="MLExperimentSpec.take_threshold"
        )
        return self

    @property
    def strategy_family(self) -> str:
        return f"ml_meta_label.{self.primary_strategy_family}"

    def composite_fingerprint(self) -> str:
        return ml_composite_fingerprint(
            primary_strategy_fingerprints=self.primary_strategy_fingerprints,
            meta_label_spec=self.meta_label_spec,
            feature_set=self.feature_set,
            regime=self.regime,
            preprocessing=self.preprocessing,
            model_search=self.model_search,
            nested_cv=self.nested_cv,
            take_threshold=self.take_threshold,
        )

    def parameter_variant_identity(self) -> str:
        return ml_parameter_variant_identity(
            self.model_search, take_threshold=self.take_threshold
        )

    def parameter_variant_label(self) -> str:
        """The registry's HUMAN label -- names the search, not a chosen point.

        The regime tag is included because the regime ablation shares its model
        family and grid with its headline trial and differs only in the regime
        representation; a label that hid that would be misleading to a reader
        even though the identities already differ.
        """
        return (
            f"{self.model_search.family.value}"
            f"__SEARCH{self.model_search.n_configurations}"
            f"__{self.regime.kind.value}"
            f"__thr{self.take_threshold}"
        )

    def experiment_identity(self) -> str:
        """The PRE-RUN identity, in the existing Phase 14 identity schema v2."""
        return registry_experiment_identity(
            strategy_fingerprint=self.composite_fingerprint(),
            strategy_family=self.strategy_family,
            root_symbol=self.root_symbol,
            parameter_variant_identity=self.parameter_variant_identity(),
            dataset_fingerprint=self.dataset_fingerprint,
            split_identity=self.nested_cv.identity(),
            validation_spec_fingerprint=self.validation_spec_fingerprint,
            reliability_policy_fingerprint=self.reliability_policy_fingerprint,
            execution_config_identity=self.execution_config_identity,
            cost_config_identity=self.cost_config_identity,
            risk_identity=self.risk_identity,
            feature_spec_fingerprint=self.feature_set.identity(),
        )

    def friendly_id(self) -> str:
        return friendly_experiment_id(
            root_symbol=self.root_symbol,
            strategy_family=self.strategy_family,
            variant_label=self.parameter_variant_label(),
            split_label="PHASE_15_DEVELOPMENT_CORPUS",
            lineage_tag=self.trial_role.value,
        )


def ml_trial_discriminator(spec: MLExperimentSpec) -> str:
    """The part of a Phase 15 pre-run identity that VARIES between trials.

    All 60 frozen Phase 15 trials inherit the SAME six plane fingerprints
    (dataset, ValidationSpec, ReliabilityPolicy, execution, cost, risk), so
    ``experiment_identity`` is an injective function of this discriminator once
    those planes are fixed: two trials collide in identity **iff** they collide
    here, whatever values the shared planes eventually take.

    That is what lets the pre-run identity map be frozen honestly *now*, before
    the Phase 15 ValidationSpec and ReliabilityPolicy exist -- uniqueness is
    proven without inventing (that is, fabricating) plane fingerprints that Phase
    15B will supply.
    """
    return fingerprint(
        "mltrialdisc1",
        {
            "schema": ML_TRIAL_DISCRIMINATOR_SCHEMA,
            "strategy_fingerprint": spec.composite_fingerprint(),
            "strategy_family": spec.strategy_family,
            "root_symbol": spec.root_symbol,
            "parameter_variant_identity": spec.parameter_variant_identity(),
            "split_identity": spec.nested_cv.identity(),
            "feature_spec_fingerprint": spec.feature_set.identity(),
        },
    )


def query_registry_before_running(registry, spec: MLExperimentSpec) -> dict[str, object]:
    """The mandatory pre-run registry query (prompt 15A s.14, CLAUDE.md 1-3).

    Returns a typed answer; the caller REFUSES to run when
    ``is_exact_duplicate`` is true and cites the prior result instead. The LLM
    never overrides this -- the registry computes it.

    Nothing here needs a fitted model. Every input is the proposal's own frozen
    configuration, including the search procedure, so "already run?" is answered
    before the compute the question exists to avoid.
    """
    identity = spec.experiment_identity()
    # MLExperimentSpec trains only on Futures roots -- a stated fact, not an
    # inference (Phase 6 ETF Research Pilot).
    dup = registry.find_exact_duplicate(identity, asset_domain=AssetDomain.FUTURES)
    related = registry.find_related(
        strategy_family=spec.strategy_family,
        root_symbol=spec.root_symbol,
        asset_domain=AssetDomain.FUTURES,
        # The proposal's own PRE-RUN parameters. No selected hyperparameter
        # appears here: a duplicate lookup that needed one could only run AFTER
        # the compute it exists to avoid.
        params={
            "model_family": spec.model_search.family.value,
            "n_search_configurations": spec.model_search.n_configurations,
            "selection_objective": spec.model_search.objective_value,
            "selection_tie_break": spec.model_search.tie_break_value,
            "random_seed": spec.model_search.random_seed,
            "regime_kind": spec.regime.kind.value,
            "take_threshold": spec.take_threshold,
        },
        feature_fingerprints=spec.feature_set.feature_fingerprints(),
    )
    return {
        "experiment_identity": identity,
        "is_exact_duplicate": bool(dup.exists),
        "duplicate": dup,
        "n_related": len(related),
        "related": related,
    }
