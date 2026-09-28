"""Phase 15 FROZEN ML candidate manifest + the pre-run compute estimate.

Materialised and committed BEFORE any ML performance is observed, exactly as the
Phase 13.5C candidate manifest was. It records the whole predeclared search:
every primary scope, the meta-label definition, the ordered feature universe, the
regime design, the model families, the complete finite hyperparameter x threshold
grid, the nested-CV design, and the composition of the Phase 15 multiple-testing
family.

:func:`assert_no_performance_fields` refuses to freeze a manifest that mentions a
result, so the artifact cannot silently acquire one later.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from alpha_agent.features.spec import FeatureSpec
from alpha_agent.ml.enums import (
    MLModelFamily,
    MLTrialRole,
    ModelSelectionObjective,
    ModelSelectionTieBreak,
    PreprocessingKind,
    RegimeSpecKind,
)
from alpha_agent.ml.errors import ManifestFrozenError
from alpha_agent.ml.features import MLFeature, MLFeatureSetSpec
from alpha_agent.ml.identity import PHASE_15_MT_FAMILY
from alpha_agent.ml.labels import MetaLabelSpec
from alpha_agent.ml.models import (
    DEFAULT_ML_RANDOM_SEED,
    FROZEN_HYPERPARAMETER_GRID,
    FROZEN_TAKE_THRESHOLD,
    RESEARCH_MODEL_FAMILIES,
    ModelSearchSpec,
    frozen_model_search,
)
from alpha_agent.ml.preprocessing import PreprocessingSpec
from alpha_agent.ml.regime import RegimeSpec
from alpha_agent.ml.splits import NestedCVSpec
from alpha_agent.validation.fingerprint import fingerprint

PHASE = "15"
MANIFEST_SCHEMA_VERSION = "phase-15-ml-candidate-manifest/4"

#: The Phase 15A.2 manifest this one SUPERSEDES for regime-transformation
#: purposes. Preserved unchanged in git (commits 497a47b / 75af1f9) and on disk
#: at ``data/manifests/phase_15/ml_candidate_manifest_identity_corrected.json``.
SUPERSEDED_15A2_MANIFEST_FINGERPRINT = (
    "p15manifest1:7a4ef8a9633bd3f18b982421baf9903b60b857f05fc618de5cd130b9718227ec"
)
SUPERSEDED_15A2_MANIFEST_PATH = (
    "data/manifests/phase_15/ml_candidate_manifest_identity_corrected.json"
)
SUPERSEDED_15A2_MANIFEST_COMMITS = ("497a47b", "75af1f9")
SUPERSESSION_15B1B_REASON = (
    "REGIME TRANSFORMATION SEMANTICS. The frozen REGIME_BASELINE declared two inputs -- "
    "realized_vol(20) (~[0.02, 0.08], always positive) and trend_strength(50,200) "
    "(~[-40, +36], signed) -- but never fixed how they combine into 3 tertile buckets. "
    "The Phase 15B.1 implementation used a raw row-wise mean, which on real feature scales "
    "is ~1.000 correlated with trend_strength and ~-0.1 with realized_vol: the regime became "
    "a tertile bucketing of the trend input ALONE, so the 20 regime-ablation trials would "
    "have tested a different question than declared, and the combination rule was not bound "
    "to RegimeSpec.identity(). Phase 15B.1b freezes the transformation: per-axis train-fold "
    "tertiles of EACH input independently (no standardise-and-average, no PCA / clustering / "
    "performance tuning), Cartesian product -> 9 one-hot states, all of it bound into "
    "RegimeSpec.identity(). One predeclared transformation, not 9 hypotheses: BH stays 60, "
    "DSR stays 3/8 and family breadth 380. Only the 40 regime-baseline pre-run experiment "
    "identities regenerate; the 20 NO_REGIME ablation identities are byte-identical. "
    "Decided by a human, NOT from real performance, BEFORE any real ML fit."
)

#: The Phase 15A.1 manifest this one SUPERSEDES for identity purposes. Preserved
#: unchanged in git (commits 0bb6014 / c375399) and on disk at
#: ``data/manifests/phase_15/ml_candidate_manifest_corrected.json``.
SUPERSEDED_15A1_MANIFEST_FINGERPRINT = (
    "p15manifest1:82983cf4f04e101520bb882ddca9aed9dd779565a646e3f4a70c68b352897975"
)
SUPERSEDED_15A1_MANIFEST_PATH = "data/manifests/phase_15/ml_candidate_manifest_corrected.json"
SUPERSEDED_15A1_MANIFEST_COMMITS = ("0bb6014", "c375399")
SUPERSESSION_15A2_REASON = (
    "PRE-RUN EXPERIMENT-IDENTITY SEMANTICS. Phase 15A/15A.1 built the ML experiment identity "
    "around one concrete MLModelSpec -- a single hyperparameter point. Under the frozen nested-CV "
    "protocol a hyperparameter point is selected INSIDE each outer training fold, so it does not "
    "exist before the run and different outer folds may legitimately select different points. "
    "Keying identity on a point therefore (a) made identity un-computable pre-run, defeating "
    "duplicate detection, (b) named a configuration the protocol never uniquely produces, and "
    "(c) split each of the 60 predeclared economic hypotheses into H registry-facing identities "
    "(380 rows for 60 hypotheses), silently disagreeing with the BH/FDR denominator. Phase 15A.2 "
    "makes the whole predeclared model-search PROCEDURE -- family, full frozen grid in "
    "declaration order, selection objective, tie-break and seed -- the identity input, and "
    "demotes the selected hyperparameters to post-fit provenance. Corrected BEFORE any real ML "
    "performance was observed. No count changes: 60 trials, 380 DSR configurations, 1,200 model "
    "fits, 240 C++ evaluations, threshold 0.50, gates 100/20."
)

#: The Phase 15A manifest this one SUPERSEDES for training-protocol purposes.
#: Preserved unchanged in git (commits 72d714b / 7234b7a) and on disk at
#: ``data/manifests/phase_15/ml_candidate_manifest.json``. Superseding is never
#: an in-place edit: the old artifact stays exactly as it was frozen.
SUPERSEDED_MANIFEST_FINGERPRINT = (
    "p15manifest1:3c28e6431deb7c5b66848ec1c6f15ad791d56c153488cf3c57b50de5ac5d537d"
)
SUPERSEDED_MANIFEST_PATH = "data/manifests/phase_15/ml_candidate_manifest.json"
SUPERSEDED_MANIFEST_COMMITS = ("72d714b", "7234b7a")
SUPERSESSION_REASON = (
    "THRESHOLD-SELECTION SEMANTICS. The Phase 15A manifest declared a four-point TAKE-threshold "
    "grid {0.45, 0.50, 0.55, 0.60} selected inside the inner CV loop, while the predeclared "
    "inner objective was mean inner-fold log loss. Log loss is a function of the predicted "
    "PROBABILITY and is completely independent of the decision threshold, so all four "
    "thresholds scored identically on every fold and the 'selection' resolved through a "
    "deterministic tie-break rather than through cross-validation. The threshold was therefore "
    "not selected by CV at all, while still multiplying the DSR search-breadth term by four. "
    "Phase 15A.1 freezes TAKE_THRESHOLD = 0.50, removes the sweep from inner model selection, "
    "and rebuilds the pre-run search accounting from the corrected manifest. Corrected BEFORE "
    "any real ML performance was observed."
)

#: Phase 15 primary scopes. The four Phase 13.5C baseline families; the model is
#: TRAINED pooled across the five roots (root enters as a categorical feature)
#: and EVALUATED per root.
PRIMARY_FAMILIES: tuple[str, ...] = ("tsmom", "ma_trend", "breakout", "mean_reversion")
ROOTS: tuple[str, ...] = ("ES", "NQ", "CL", "GC", "ZN")

#: Silver Bullet is deliberately out of the Phase 15 MVP.
NOT_EVALUATED: tuple[dict, ...] = (
    {
        "primary_family": "silver_bullet",
        "roots": ["NQ"],
        "reason": "native_1m_cadence_needs_its_own_label_horizon",
        "note": (
            "SILVER_BULLET_BENCHMARK is a frozen native-1m NQ hypothesis. Its episodes are "
            "intraday, so the label horizon, the purge/embargo scale and the event density "
            "are all different from the daily-cadence baselines. Meta-labeling it is a "
            "separate predeclared hypothesis family for a later phase, not an MVP variant."
        ),
    },
)

# --------------------------------------------------------------------------
# The frozen feature universe (prompt 15A s.6)
# --------------------------------------------------------------------------
#: Ordered. The order is semantic: it is the model matrix column order.
ML_FEATURES: tuple[MLFeature, ...] = (
    MLFeature(alias="mom_20", spec=FeatureSpec(kind="cum_log_return", params={"window": 20}),
              rationale="short-horizon trend the primary is riding"),
    MLFeature(alias="mom_120", spec=FeatureSpec(kind="cum_log_return", params={"window": 120}),
              rationale="long-horizon trend context"),
    MLFeature(alias="rvol_20", spec=FeatureSpec(kind="realized_vol", params={"window": 20}),
              rationale="current realised volatility"),
    MLFeature(alias="rvol_60", spec=FeatureSpec(kind="realized_vol", params={"window": 60}),
              rationale="slower volatility level; the 20/60 pair expresses vol expansion"),
    MLFeature(alias="atr_14", spec=FeatureSpec(kind="atr", params={"window": 14}),
              rationale="range-based volatility, robust to gaps"),
    MLFeature(alias="zscore_20", spec=FeatureSpec(kind="zscore", params={"window": 20}),
              rationale="stretch from the local mean -- distance from a reversion threshold"),
    MLFeature(alias="dist_ma_50", spec=FeatureSpec(kind="dist_from_ma", params={"window": 50}),
              rationale="signal strength: how far price sits from the trend anchor"),
    MLFeature(alias="donchian_pos_55", spec=FeatureSpec(kind="donchian_pos", params={"window": 55}),
              rationale="position in the breakout channel -- distance from the breakout level"),
    MLFeature(alias="trend_strength_50_200",
              spec=FeatureSpec(kind="trend_strength", params={"fast": 50, "slow": 200}),
              rationale="fast/slow trend agreement"),
    MLFeature(alias="vol_percentile_20_252",
              spec=FeatureSpec(kind="vol_percentile", params={"window": 20, "lookback": 252}),
              rationale="causal volatility regime coordinate"),
    MLFeature(alias="volume_zscore_20",
              spec=FeatureSpec(kind="volume_zscore", params={"window": 20}),
              rationale="participation relative to its own recent norm"),
)

ML_FEATURE_SET = MLFeatureSetSpec(
    label="phase_15_mvp_causal_v1",
    features=ML_FEATURES,
    categoricals=("root_symbol", "primary_side"),
)

#: The deterministic causal regime baseline (prompt 15A s.7 concept A).
#: Phase 15B.1b froze its exact transformation: per-axis train-fold tertiles of
#: realized_vol(20) and signed trend_strength(50,200), Cartesian product -> 9
#: one-hot states. NEVER standardise-and-average, no PCA/clustering/perf tuning.
REGIME_BASELINE = RegimeSpec(
    kind=RegimeSpecKind.DETERMINISTIC_CAUSAL_VOL_TREND,
    inputs=(
        FeatureSpec(kind="realized_vol", params={"window": 20}),
        FeatureSpec(kind="trend_strength", params={"fast": 50, "slow": 200}),
    ),
    n_buckets=3,
    axis_bucket_count=3,
    axis_semantics=(
        (
            "axis 0 -- realized_vol(20) tertiles: lo = calmest third of the "
            "train fold, mid, hi = most volatile third"
        ),
        (
            "axis 1 -- signed trend_strength(50,200) tertiles: lo = most "
            "negative third (fast MA well below slow -- bearish), mid = "
            "neutral, hi = most positive third (bullish)"
        ),
    ),
)
#: The ablation: identical pipeline with the regime removed. A genuinely
#: different configuration, so it is a BH trial, not a free diagnostic.
REGIME_ABLATION = RegimeSpec(kind=RegimeSpecKind.NONE)

#: The FROZEN model-search procedure per research family. This -- not a single
#: hyperparameter point -- is the model-side unit of a Phase 15 hypothesis and
#: the model-side input to pre-run experiment identity (Phase 15A.2).
MODEL_SEARCHES: dict[MLModelFamily, ModelSearchSpec] = {
    f: frozen_model_search(f, random_seed=DEFAULT_ML_RANDOM_SEED)
    for f in RESEARCH_MODEL_FAMILIES
}

META_LABEL_SPEC = MetaLabelSpec()
PREPROCESSING_SPEC = PreprocessingSpec(kind=PreprocessingKind.MEDIAN_IMPUTE_STANDARDIZE,
                                       winsorize_sigma=5.0)

# --------------------------------------------------------------------------
# The frozen nested-CV design (prompt 15A s.10)
# --------------------------------------------------------------------------
#: Calendar-year outer test blocks. Chosen for auditability: a reader can check
#: the design by eye, and no boundary was tuned to a result.
_YEAR_NS = {
    2018: 1_514_764_800_000_000_000,
    2019: 1_546_300_800_000_000_000,
    2020: 1_577_836_800_000_000_000,
    2021: 1_609_459_200_000_000_000,
    2022: 1_640_995_200_000_000_000,
    2023: 1_672_531_200_000_000_000,
    2024: 1_704_067_200_000_000_000,
    2025: 1_735_689_600_000_000_000,      # EXCLUSIVE corpus end -- the locked holdout
}
OUTER_TEST_BLOCKS: tuple[tuple[int, int], ...] = (
    (_YEAR_NS[2020], _YEAR_NS[2021]),
    (_YEAR_NS[2021], _YEAR_NS[2022]),
    (_YEAR_NS[2022], _YEAR_NS[2023]),
    (_YEAR_NS[2023], _YEAR_NS[2024]),
    (_YEAR_NS[2024], _YEAR_NS[2025]),
)

NESTED_CV = NestedCVSpec(
    outer_test_blocks=OUTER_TEST_BLOCKS,
    n_inner_folds=3,
    embargo_days=10,
    min_train_days=365,
    min_train_events=100,
    min_test_events=20,
)

#: Cost scenarios: the frozen Phase 13.5C plan, unchanged, so the ML comparison
#: sits on the same cost surface as the primary it is judged against.
COST_SCENARIOS: tuple[dict, ...] = (
    {"label": "baseline_1_0x", "kind": "multiplier", "multiplier": 1.0},
    {"label": "stress_1_5x", "kind": "multiplier", "multiplier": 1.5},
    {"label": "stress_2_0x", "kind": "multiplier", "multiplier": 2.0},
)

#: Placebo control: a random meta-labeler matched on the realised TAKE rate.
#: NOT a BH trial -- it is a null, and a null never competes for significance.
PLACEBO_DRAWS = 20
PLACEBO_POLICY = "conditional_on_gating_null_rejection"


# --------------------------------------------------------------------------
# manifest models
# --------------------------------------------------------------------------
class MLCandidateTrial(BaseModel):
    """One predeclared Phase 15 BH/FDR trial."""

    model_config = {"frozen": True, "extra": "forbid"}

    trial_label: str
    primary_family: str
    root_symbol: str
    training_roots: tuple[str, ...]
    model_family: MLModelFamily
    regime_kind: RegimeSpecKind
    trial_role: MLTrialRole
    #: The predeclared model configurations this trial's inner loop inspects.
    #: This IS the trial-level DSR search breadth.
    n_hyperparameter_points: int
    #: The FROZEN search PROCEDURE, declared here because it is a pre-run
    #: identity input (Phase 15A.2). A trial predeclares a search, never a
    #: selected hyperparameter point: nested CV selects the point inside each
    #: outer training fold, so no point exists before the run and different folds
    #: may select different ones. Selected points are post-fit provenance.
    model_search_identity: str = ""
    selection_objective: ModelSelectionObjective = (
        ModelSelectionObjective.MEAN_INNER_FOLD_LOG_LOSS_MINIMISED
    )
    selection_tie_break: ModelSelectionTieBreak = (
        ModelSelectionTieBreak.FROZEN_GRID_DECLARATION_ORDER_FIRST
    )
    model_search_random_seed: int = DEFAULT_ML_RANDOM_SEED
    #: FROZEN, not searched. Recorded so the manifest states the decision rule.
    take_threshold: float = FROZEN_TAKE_THRESHOLD
    n_threshold_configurations: int = 1
    multiple_testing_family_id: str = PHASE_15_MT_FAMILY

    @property
    def n_inspected_configurations(self) -> int:
        """Distinct predeclared configurations inspected to produce this trial.

        Re-selecting among the SAME configurations in another outer fold is a
        training operation, not a new model hypothesis, so the outer-fold count
        does not appear here.
        """
        return self.n_hyperparameter_points * self.n_threshold_configurations


class ComputeEstimate(BaseModel):
    """The pre-run search accounting (prompt 15A.1 s.4).

    Every quantity below is a DIFFERENT thing, and conflating any two of them is
    how a search's real breadth gets understated. In particular a repeated fit of
    the SAME configuration on another CV fold is a TRAINING operation, not a new
    independent model hypothesis, so fold counts multiply the fit totals and
    never the DSR search-breadth term.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    # -- 1. BH/FDR: out-of-sample ECONOMIC hypotheses actually inspected -------
    n_ml_hypotheses_bh_denominator: int
    n_headline_trials: int
    n_ablation_trials: int

    # -- 2. Fitted pipelines (pooled over roots) ------------------------------
    n_distinct_fitted_pipelines: int

    # -- 3. Model hyperparameter configurations -------------------------------
    n_hyperparameter_points_by_model_family: dict[str, int]
    n_distinct_model_configurations_fitted: int
    n_threshold_configurations: int
    take_threshold: float

    # -- 4/5. Fits: inner-CV selection, and outer OOF-producing refits ---------
    n_inner_fits: int
    n_oof_producing_fits: int
    n_model_fits_total: int

    # -- 6. DSR search breadth ------------------------------------------------
    n_inspected_configurations_dsr: int
    dsr_effective_trial_count_by_model_family: dict[str, int]
    #: bookkeeping only: configurations x outer folds. NEVER a DSR term.
    n_configuration_fold_inspections: int

    # -- 7. C++ economic evaluations ------------------------------------------
    n_cpp_evaluations_headline: int
    n_cpp_evaluations_primary_baseline: int
    n_cpp_evaluations_total_excluding_placebo: int

    # -- 8. Conditional placebo runs ------------------------------------------
    n_cpp_evaluations_placebo_max: int
    n_cpp_evaluations_total_max: int
    placebo_policy: str

    n_diagnostic_reruns_planned: int
    superseded_n_inspected_configurations_dsr: int
    notes: tuple[str, ...] = ()


class MLCandidateManifest(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = MANIFEST_SCHEMA_VERSION
    phase: str = PHASE
    primary_families: tuple[str, ...] = PRIMARY_FAMILIES
    roots: tuple[str, ...] = ROOTS
    not_evaluated: tuple[dict, ...] = NOT_EVALUATED

    meta_label_spec: MetaLabelSpec = META_LABEL_SPEC
    feature_set: MLFeatureSetSpec = ML_FEATURE_SET
    regime_baseline: RegimeSpec = REGIME_BASELINE
    regime_ablation: RegimeSpec = REGIME_ABLATION
    preprocessing: PreprocessingSpec = PREPROCESSING_SPEC
    nested_cv: NestedCVSpec = NESTED_CV

    research_model_families: tuple[MLModelFamily, ...] = RESEARCH_MODEL_FAMILIES
    hyperparameter_grid: dict = Field(
        default_factory=lambda: {
            f.value: [dict(h) for h in FROZEN_HYPERPARAMETER_GRID[f]]
            for f in RESEARCH_MODEL_FAMILIES
        }
    )
    #: The frozen search PROCEDURE per family -- the model-side identity unit.
    model_searches: dict = Field(
        default_factory=lambda: {
            f.value: {
                "identity": MODEL_SEARCHES[f].identity(),
                "n_configurations": MODEL_SEARCHES[f].n_configurations,
                "selection_objective": MODEL_SEARCHES[f].objective_value,
                "selection_tie_break": MODEL_SEARCHES[f].tie_break_value,
                "random_seed": MODEL_SEARCHES[f].random_seed,
            }
            for f in RESEARCH_MODEL_FAMILIES
        }
    )
    #: What a PRE-RUN Phase 15 experiment identity is, and is not (Phase 15A.2).
    identity_semantics: dict = Field(
        default_factory=lambda: {
            "identity_schema": "experiment-identity/2",
            "identity_is_pre_run": True,
            "model_side_identity_unit": "predeclared_model_search_procedure",
            "pre_run_identity_includes": [
                "composite strategy fingerprint (primary specs + meta-label pipeline)",
                "model family",
                "the FULL frozen search grid, in declaration order",
                "selection objective",
                "selection tie-break",
                "model search random seed",
                "regime representation",
                "frozen TAKE threshold (0.50)",
                "nested-CV split identity",
                "the ML feature set's OWN fingerprint",
                "dataset / ValidationSpec / ReliabilityPolicy / execution / cost / risk planes",
            ],
            "post_fit_provenance_never_identity": [
                "the hyperparameters an outer fold selected",
                "the per-fold selected model identity and artifact fingerprint",
                "inner objective values",
                "target_schedule_hash",
                "report_fingerprint",
            ],
            "n_pre_run_experiment_identities": 60,
            "identities_per_predeclared_hypothesis": 1,
            "rationale": (
                "Under nested CV the hyperparameter point is selected inside each outer "
                "training fold, so no point exists before the run and different folds may "
                "select different ones. The scientific hypothesis a trial states is about the "
                "predeclared SEARCH, so the search is identity and the selection is evidence."
            ),
        }
    )
    #: FROZEN decision rule, NOT a search dimension: p >= 0.50 -> TAKE.
    take_threshold: float = FROZEN_TAKE_THRESHOLD
    threshold_is_searched: bool = False
    inner_objective: str = "mean_inner_fold_log_loss_minimised"
    inner_selection_dimensions: tuple[str, ...] = ("model_hyperparameter_configuration",)
    #: The IMMEDIATE predecessor. Supersession is a chain of preserved artifacts,
    #: never an in-place edit of any of them.
    supersedes: dict = Field(
        default_factory=lambda: {
            "manifest_fingerprint": SUPERSEDED_15A2_MANIFEST_FINGERPRINT,
            "path": SUPERSEDED_15A2_MANIFEST_PATH,
            "commits": list(SUPERSEDED_15A2_MANIFEST_COMMITS),
            "reason": SUPERSESSION_15B1B_REASON,
            "preserved": "unchanged in git history and on disk; never amended, never deleted",
        }
    )
    supersession_chain: tuple[dict, ...] = Field(
        default_factory=lambda: (
            {
                "phase": "15A",
                "manifest_fingerprint": SUPERSEDED_MANIFEST_FINGERPRINT,
                "path": SUPERSEDED_MANIFEST_PATH,
                "commits": list(SUPERSEDED_MANIFEST_COMMITS),
                "schema_version": "phase-15-ml-candidate-manifest/1",
                "superseded_by": "15A.1",
                "reason": SUPERSESSION_REASON,
            },
            {
                "phase": "15A.1",
                "manifest_fingerprint": SUPERSEDED_15A1_MANIFEST_FINGERPRINT,
                "path": SUPERSEDED_15A1_MANIFEST_PATH,
                "commits": list(SUPERSEDED_15A1_MANIFEST_COMMITS),
                "schema_version": "phase-15-ml-candidate-manifest/2",
                "superseded_by": "15A.2",
                "reason": SUPERSESSION_15A2_REASON,
            },
            {
                "phase": "15A.2",
                "manifest_fingerprint": SUPERSEDED_15A2_MANIFEST_FINGERPRINT,
                "path": SUPERSEDED_15A2_MANIFEST_PATH,
                "commits": list(SUPERSEDED_15A2_MANIFEST_COMMITS),
                "schema_version": "phase-15-ml-candidate-manifest/3",
                "superseded_by": "15B.1b",
                "reason": SUPERSESSION_15B1B_REASON,
            },
        )
    )
    cost_scenarios: tuple[dict, ...] = COST_SCENARIOS
    placebo_draws: int = PLACEBO_DRAWS
    placebo_policy: str = PLACEBO_POLICY

    trials: tuple[MLCandidateTrial, ...]
    multiple_testing_family: dict = Field(default_factory=dict)
    compute_estimate: ComputeEstimate

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def manifest_fingerprint(self) -> str:
        return fingerprint("p15manifest1", json.loads(self.canonical_json()))

    def n_trials(self) -> int:
        return len(self.trials)


def build_candidate_manifest() -> MLCandidateManifest:
    """Deterministically build the frozen Phase 15 manifest. Reads no data."""
    trials: list[MLCandidateTrial] = []

    # HEADLINE: every primary family x root x research model family.
    for family in PRIMARY_FAMILIES:
        for model_family in RESEARCH_MODEL_FAMILIES:
            for root in ROOTS:
                trials.append(
                    MLCandidateTrial(
                        trial_label=f"{root}__ML_{family.upper()}__{model_family.value}",
                        primary_family=family,
                        root_symbol=root,
                        training_roots=ROOTS,
                        model_family=model_family,
                        regime_kind=REGIME_BASELINE.kind,
                        trial_role=MLTrialRole.HEADLINE,
                        n_hyperparameter_points=len(FROZEN_HYPERPARAMETER_GRID[model_family]),
                        model_search_identity=MODEL_SEARCHES[model_family].identity(),
                    )
                )

    # ABLATION: the same pipeline with the regime removed, tree family only.
    ablation_family = MLModelFamily.HIST_GRADIENT_BOOSTING
    for family in PRIMARY_FAMILIES:
        for root in ROOTS:
            trials.append(
                MLCandidateTrial(
                    trial_label=f"{root}__ML_{family.upper()}__{ablation_family.value}__NO_REGIME",
                    primary_family=family,
                    root_symbol=root,
                    training_roots=ROOTS,
                    model_family=ablation_family,
                    regime_kind=REGIME_ABLATION.kind,
                    trial_role=MLTrialRole.ABLATION,
                    n_hyperparameter_points=len(FROZEN_HYPERPARAMETER_GRID[ablation_family]),
                    model_search_identity=MODEL_SEARCHES[ablation_family].identity(),
                )
            )

    estimate = compute_estimate(tuple(trials))
    mt = {
        "family_id": PHASE_15_MT_FAMILY,
        "n_trials": len(trials),
        "headline": sum(t.trial_role is MLTrialRole.HEADLINE for t in trials),
        "ablation": sum(t.trial_role is MLTrialRole.ABLATION for t in trials),
        "placebo_in_denominator": False,
        "diagnostic_reruns_in_denominator": False,
        "cross_root_summaries_in_denominator": False,
        "phase_13_5c_trials_folded_in": False,
        "phase_13_5c_family_id": "phase_13_5c.all",
        "phase_13_5c_n_trials_unchanged": 107,
        "bh_q": 0.10,
        "dsr_effective_trial_count": estimate.n_inspected_configurations_dsr,
        "dsr_effective_trial_count_by_model_family":
            estimate.dsr_effective_trial_count_by_model_family,
        "dsr_effective_trial_count_superseded":
            estimate.superseded_n_inspected_configurations_dsr,
        "take_threshold": FROZEN_TAKE_THRESHOLD,
        "threshold_is_searched": False,
        "note": (
            "Phase 15 is a SEPARATE multiple-testing family. The 107 Phase 13.5C trials are "
            "historical and unchanged and are never folded into this denominator. BH/FDR runs "
            "over the trials listed here; each is one out-of-sample economic result actually "
            "inspected. DSR uses the inspected-CONFIGURATION count, because DSR is the tool "
            "that deflates for how much SEARCH produced a result: each trial's inner loop "
            "chooses among a finite predeclared hyperparameter grid, and pretending that "
            "search did not happen would overstate every Sharpe. The count is 380, not the "
            "superseded 7,600: the TAKE threshold is frozen at 0.50 rather than swept (the "
            "inner objective is threshold-independent, so the sweep selected nothing), and "
            "refitting one configuration on another CV fold is a training operation rather "
            "than a new model hypothesis."
        ),
    }
    return MLCandidateManifest(
        trials=tuple(trials), multiple_testing_family=mt, compute_estimate=estimate
    )


def compute_estimate(trials: tuple[MLCandidateTrial, ...]) -> ComputeEstimate:
    """Derive the pre-run search accounting from the manifest itself, by hand nowhere.

    The derivation, with every quantity kept distinct (prompt 15A.1 s.4)::

        F  = outer folds                                          = 5
        I  = inner folds                                          = 3
        R  = roots (pooled training, per-root economics)          = 5
        P  = fitted pipelines = primary_family x (model_family, regime)
                                                                  = 4 x 3 = 12
        H(LOGISTIC_L2) = 3          H(HIST_GRADIENT_BOOSTING) = 8
        THR = threshold configurations = 1   (FROZEN at 0.50, not searched)

        (1) BH/FDR economic hypotheses  T = P x R = 60
                                          = 40 headline + 20 regime ablation
        (2) fitted pipelines            P = 12
        (3) distinct model configurations fitted
                                        C = sum_P H = 4x3 + 4x8 + 4x8 = 76
        (4) inner-CV fits               = sum_P (F x I x H) = 15 x 76 = 1140
        (5) outer OOF-producing fits    = P x F = 60
            total model fits            = 1140 + 60 = 1200
        (6) DSR inspected configurations
                                        N = sum_T H(trial) x THR
                                          = 20x3 + 20x8 + 20x8 = 380
        (7) C++ economic evaluations    = (T + P x R primary baselines) x 3 costs
                                          = 180 + 60 = 240
        (8) conditional placebo runs    = T x 20 draws = 1200 (a CEILING)

    Why (6) is 380 and not 7,600
    -----------------------------
    The superseded Phase 15A count was ``sum_T F x H x 4 thresholds`` =
    ``5 x 4 x 380`` = 7,600. Both multipliers were wrong:

    * ``x 4`` counted a threshold sweep that selected nothing -- the inner
      objective (log loss) is threshold-independent, so the four thresholds were
      never distinguishable by the selection rule;
    * ``x 5`` counted each outer fold's refit of the SAME configuration as a
      separate inspected configuration. Refitting one predeclared configuration
      on another fold is a training operation, not a new model hypothesis.

    380 is what the research procedure actually inspects: for each of the 60
    economic trials, the finite predeclared grid its inner loop chooses from.
    """
    n_outer = NESTED_CV.n_outer_folds
    n_inner = NESTED_CV.n_inner_folds
    n_costs = len(COST_SCENARIOS)

    headline = [t for t in trials if t.trial_role is MLTrialRole.HEADLINE]
    ablation = [t for t in trials if t.trial_role is MLTrialRole.ABLATION]

    # A fitted pipeline is per (primary_family, model_family, regime_kind): the
    # model is trained POOLED over roots, so the five per-root trials of one
    # combination share one fitted pipeline and are five distinct economic
    # questions asked of it.
    pipelines = {(t.primary_family, t.model_family, t.regime_kind) for t in trials}
    n_pipelines = len(pipelines)

    inner_fits = 0
    distinct_configs = 0
    for _family, model_family, _regime in sorted(
        pipelines, key=lambda p: (p[0], p[1].value, p[2].value)
    ):
        n_hyper = len(FROZEN_HYPERPARAMETER_GRID[model_family])
        inner_fits += n_outer * n_inner * n_hyper
        distinct_configs += n_hyper
    oof_fits = n_pipelines * n_outer

    # DSR search breadth: per trial, the predeclared configurations its inner
    # loop inspects. Fold repetition is training, so it is NOT a factor here.
    inspected_dsr = sum(t.n_inspected_configurations for t in trials)
    config_fold_inspections = sum(t.n_inspected_configurations * n_outer for t in trials)
    superseded_dsr = sum(
        t.n_hyperparameter_points * 4 * n_outer for t in trials      # the 15A formula
    )

    n_primary_cpp = len(PRIMARY_FAMILIES) * len(ROOTS) * n_costs
    n_headline_cpp = (len(headline) + len(ablation)) * n_costs
    n_placebo_cpp = (len(headline) + len(ablation)) * PLACEBO_DRAWS

    return ComputeEstimate(
        n_ml_hypotheses_bh_denominator=len(trials),
        n_headline_trials=len(headline),
        n_ablation_trials=len(ablation),
        n_distinct_fitted_pipelines=n_pipelines,
        n_hyperparameter_points_by_model_family={
            f.value: len(FROZEN_HYPERPARAMETER_GRID[f]) for f in RESEARCH_MODEL_FAMILIES
        },
        n_distinct_model_configurations_fitted=distinct_configs,
        n_threshold_configurations=1,
        take_threshold=FROZEN_TAKE_THRESHOLD,
        n_inner_fits=inner_fits,
        n_oof_producing_fits=oof_fits,
        n_model_fits_total=inner_fits + oof_fits,
        n_inspected_configurations_dsr=inspected_dsr,
        dsr_effective_trial_count_by_model_family={
            f.value: len(FROZEN_HYPERPARAMETER_GRID[f]) for f in RESEARCH_MODEL_FAMILIES
        },
        n_configuration_fold_inspections=config_fold_inspections,
        n_cpp_evaluations_headline=n_headline_cpp,
        n_cpp_evaluations_primary_baseline=n_primary_cpp,
        n_cpp_evaluations_total_excluding_placebo=n_primary_cpp + n_headline_cpp,
        n_cpp_evaluations_placebo_max=n_placebo_cpp,
        n_cpp_evaluations_total_max=n_primary_cpp + n_headline_cpp + n_placebo_cpp,
        placebo_policy=PLACEBO_POLICY,
        n_diagnostic_reruns_planned=0,
        superseded_n_inspected_configurations_dsr=superseded_dsr,
        notes=(
            (
                "TAKE_THRESHOLD is FROZEN at 0.50 and is not a search dimension. The Phase 15A "
                "inner loop swept {0.45, 0.50, 0.55, 0.60} while scoring every candidate with "
                "mean inner-fold log loss -- a probability-based objective that is independent "
                "of the decision threshold -- so the sweep selected nothing and only inflated "
                "the DSR term fourfold. Thresholds such as 0.55 / 0.60 may return only as an "
                "explicitly predeclared decision-policy family, never chosen on PnL."
            ),
            (
                "A repeated fit of the SAME configuration on another CV fold is a TRAINING "
                "operation, not a new independent model hypothesis. Fold counts therefore "
                "multiply the fit totals (1,140 inner + 60 outer) and never the DSR "
                "search-breadth term (380)."
            ),
            (
                "A fitted pipeline is per (primary_family, model_family, regime_kind): the "
                "model is trained POOLED across the five roots with root as a categorical "
                "feature, then evaluated per root. Five per-root trials therefore SHARE one "
                "fitted pipeline but are five separately inspected economic results, so all "
                "five stay in the BH denominator."
            ),
            (
                "dsr_effective_trial_count_by_model_family gives the TRIAL-level breadth used "
                "to deflate one trial's Sharpe (3 for LOGISTIC_L2, 8 for "
                "HIST_GRADIENT_BOOSTING). n_inspected_configurations_dsr = 380 is the "
                "FAMILY-level breadth, used when the best result of the whole Phase 15 search "
                "is what is being judged."
            ),
            (
                "n_oof_producing_fits counts only the outer refits that actually emit an OOF "
                "prediction. Every other fit is inner-loop selection and never scores a row it "
                "could have been trained on."
            ),
            (
                "Placebo runs are conditional on a trial rejecting its gating null. Given the "
                "Phase 13.5C prior (0 PASS in 107 trials) the expected placebo count is near "
                "zero; the number quoted is the ceiling, not a plan. Placebos are nulls and "
                "never enter the BH denominator."
            ),
            (
                "Diagnostic reruns are 0 by predeclaration. Any that later prove necessary are "
                "recorded as SensitivityEvidence, never as new trials -- unless they inspect a "
                "genuinely new model configuration, which makes them a new trial that must be "
                "predeclared first."
            ),
        ),
    )


def trial_accounting_payload(manifest: MLCandidateManifest) -> dict:
    """The PRETRAIN_TRIAL_ACCOUNTING artifact: the derivation, term by term.

    Every number is recomputed here from the manifest's own trials, and the
    identities that tie the terms together are asserted, not asserted-in-prose.
    """
    est = manifest.compute_estimate
    n_outer = manifest.nested_cv.n_outer_folds
    n_inner = manifest.nested_cv.n_inner_folds
    by_family: dict[str, int] = {}
    for t in manifest.trials:
        by_family[t.model_family.value] = by_family.get(t.model_family.value, 0) + 1

    dsr_terms = [
        {
            "trial_role": role.value,
            "model_family": family,
            "n_trials": n,
            "n_configurations_per_trial": len(FROZEN_HYPERPARAMETER_GRID[MLModelFamily(family)]),
            "contribution": n * len(FROZEN_HYPERPARAMETER_GRID[MLModelFamily(family)]),
        }
        for (role, family), n in sorted(
            _trial_census(manifest).items(), key=lambda kv: (kv[0][0].value, kv[0][1])
        )
    ]

    payload = {
        "phase": "15B.1b",
        "manifest_schema_version": manifest.schema_version,
        "manifest_fingerprint": manifest.manifest_fingerprint(),
        "supersedes": manifest.supersedes,
        "frozen_decision_rule": {
            "take_threshold": manifest.take_threshold,
            "rule": "probability_take >= 0.50 -> TAKE; probability_take < 0.50 -> SKIP",
            "threshold_is_searched": manifest.threshold_is_searched,
            "inner_objective": manifest.inner_objective,
            "inner_selection_dimensions": list(manifest.inner_selection_dimensions),
            "probability_never_becomes_position_size": True,
        },
        "design_constants": {
            "n_outer_folds": n_outer,
            "n_inner_folds": n_inner,
            "n_primary_families": len(manifest.primary_families),
            "n_roots": len(manifest.roots),
            "n_cost_scenarios": len(manifest.cost_scenarios),
            "placebo_draws": manifest.placebo_draws,
            "min_train_events": manifest.nested_cv.min_train_events,
            "min_test_events": manifest.nested_cv.min_test_events,
            "hyperparameter_points_by_model_family": est.n_hyperparameter_points_by_model_family,
        },
        "1_bh_fdr_economic_hypotheses": {
            "value": est.n_ml_hypotheses_bh_denominator,
            "derivation": "pipelines(12) x roots(5) = 60",
            "headline": est.n_headline_trials,
            "regime_ablation": est.n_ablation_trials,
            "by_model_family": by_family,
            "bh_q": manifest.multiple_testing_family.get("bh_q"),
            "family_id": manifest.multiple_testing_family.get("family_id"),
            "excluded_from_denominator": [
                "placebo/null runs", "cross-root summaries", "diagnostic reruns",
                "sensitivity reruns", "the 107 Phase 13.5C trials",
            ],
        },
        "2_fitted_pipelines": {
            "value": est.n_distinct_fitted_pipelines,
            "derivation": (
                "primary_families(4) x [(LOGISTIC_L2, regime baseline), "
                "(HIST_GRADIENT_BOOSTING, regime baseline), "
                "(HIST_GRADIENT_BOOSTING, no regime)] = 4 x 3 = 12"
            ),
            "note": (
                "Training is pooled across the five roots, so one fitted pipeline serves five "
                "per-root economic trials."
            ),
        },
        "3_model_hyperparameter_configurations": {
            "distinct_configurations_fitted": est.n_distinct_model_configurations_fitted,
            "derivation": "sum over pipelines of H: 4x3 + 4x8 + 4x8 = 12 + 32 + 32 = 76",
            "n_threshold_configurations": est.n_threshold_configurations,
            "threshold_note": (
                "1, not 4. The threshold is frozen at 0.50 and is not selected."
            ),
        },
        "4_inner_cv_fits": {
            "value": est.n_inner_fits,
            "derivation": (
                f"sum over pipelines of (outer {n_outer} x inner {n_inner} x H) = "
                f"{n_outer * n_inner} x 76 = {est.n_inner_fits}"
            ),
            "note": "Training operations. None of these is a BH trial or a DSR configuration.",
        },
        "5_outer_oof_producing_fits": {
            "value": est.n_oof_producing_fits,
            "derivation": f"pipelines(12) x outer folds({n_outer}) = {est.n_oof_producing_fits}",
            "note": "The only fits that emit an out-of-fold prediction.",
        },
        "total_model_fits": {
            "value": est.n_model_fits_total,
            "derivation": f"{est.n_inner_fits} inner + {est.n_oof_producing_fits} outer",
        },
        "6_dsr_inspected_configurations": {
            "value": est.n_inspected_configurations_dsr,
            "derivation": "sum over the 60 trials of H(trial) x 1 threshold = 20x3 + 20x8 + 20x8",
            "terms": dsr_terms,
            "per_trial_breadth": est.dsr_effective_trial_count_by_model_family,
            "family_level_breadth": est.n_inspected_configurations_dsr,
            "configuration_fold_inspections_not_a_dsr_term": (
                est.n_configuration_fold_inspections
            ),
            "superseded_value": est.superseded_n_inspected_configurations_dsr,
            "superseded_derivation": (
                "Phase 15A computed sum over trials of (outer folds x H x 4 thresholds) = "
                "5 x 4 x 380 = 7,600. The x4 counted a threshold sweep that the "
                "threshold-independent inner objective could never distinguish, and the x5 "
                "counted per-fold refits of the SAME configuration as separate inspected "
                "configurations. Both are removed."
            ),
            "correction_factor": "7600 / 380 = 20 = 4 (phantom thresholds) x 5 (fold refits)",
        },
        "7_cpp_economic_evaluations": {
            "primary_baseline": est.n_cpp_evaluations_primary_baseline,
            "meta_labeled": est.n_cpp_evaluations_headline,
            "total_excluding_placebo": est.n_cpp_evaluations_total_excluding_placebo,
            "derivation": (
                "primary: families(4) x roots(5) x cost scenarios(3) = 60; "
                "meta-labeled: trials(60) x cost scenarios(3) = 180"
            ),
        },
        "8_conditional_placebo_runs": {
            "ceiling": est.n_cpp_evaluations_placebo_max,
            "derivation": "trials(60) x placebo draws(20) = 1200",
            "policy": est.placebo_policy,
            "expected": (
                "near zero -- placebos run only for a trial that rejects its gating null, and "
                "the Phase 13.5C prior is 0 PASS in 107 trials"
            ),
            "in_bh_denominator": False,
        },
        "grand_total_cpp_evaluations_max": est.n_cpp_evaluations_total_max,
        "9_pre_run_experiment_identity": {
            "model_side_identity_unit": "predeclared_model_search_procedure",
            "identities_per_predeclared_hypothesis": 1,
            "n_pre_run_experiment_identities": len(manifest.trials),
            "model_searches": manifest.model_searches,
            "post_fit_provenance_never_identity": (
                manifest.identity_semantics["post_fit_provenance_never_identity"]
            ),
            "superseded_behaviour": (
                "Phase 15A/15A.1 keyed identity on ONE MLModelSpec hyperparameter point. Under "
                "nested CV the point is selected inside each outer training fold, so identity "
                "was not computable pre-run and each of the 60 predeclared hypotheses would "
                "have produced H registry-facing identities (20x3 + 20x8 + 20x8 = 380), "
                "disagreeing with its own BH/FDR denominator of 60."
            ),
            "superseded_n_identities_if_keyed_on_a_point": (
                est.n_inspected_configurations_dsr
            ),
        },
        "invariants_checked": [],
    }

    checks = {
        "total_fits == inner + outer":
            est.n_model_fits_total == est.n_inner_fits + est.n_oof_producing_fits,
        "bh_denominator == headline + ablation":
            est.n_ml_hypotheses_bh_denominator == est.n_headline_trials + est.n_ablation_trials,
        "bh_denominator == pipelines x roots":
            est.n_ml_hypotheses_bh_denominator
            == est.n_distinct_fitted_pipelines * len(manifest.roots),
        "dsr == sum of per-trial configuration counts":
            est.n_inspected_configurations_dsr == sum(t["contribution"] for t in dsr_terms),
        "dsr < superseded dsr":
            est.n_inspected_configurations_dsr < est.superseded_n_inspected_configurations_dsr,
        "superseded / corrected == 20":
            est.superseded_n_inspected_configurations_dsr
            == 20 * est.n_inspected_configurations_dsr,
        "inner fits == outer x inner x distinct configurations":
            est.n_inner_fits
            == n_outer * n_inner * est.n_distinct_model_configurations_fitted,
        "cpp total == primary + meta-labeled + placebo ceiling":
            est.n_cpp_evaluations_total_max
            == est.n_cpp_evaluations_primary_baseline
            + est.n_cpp_evaluations_headline
            + est.n_cpp_evaluations_placebo_max,
        "exactly one threshold configuration": est.n_threshold_configurations == 1,
        "take threshold is 0.50": est.take_threshold == FROZEN_TAKE_THRESHOLD,
        "event gates unchanged":
            manifest.nested_cv.min_train_events == 100
            and manifest.nested_cv.min_test_events == 20,
        "one pre-run identity per predeclared hypothesis":
            manifest.identity_semantics["n_pre_run_experiment_identities"]
            == est.n_ml_hypotheses_bh_denominator,
        "identity would have been 380 if keyed on a hyperparameter point":
            est.n_inspected_configurations_dsr
            == sum(t.n_hyperparameter_points for t in manifest.trials),
        "every trial declares the frozen search of its model family":
            all(
                t.model_search_identity == MODEL_SEARCHES[t.model_family].identity()
                for t in manifest.trials
            ),
        "search breadth per family is 3 logistic / 8 tree":
            manifest.model_searches[MLModelFamily.LOGISTIC_L2.value]["n_configurations"] == 3
            and manifest.model_searches[
                MLModelFamily.HIST_GRADIENT_BOOSTING.value
            ]["n_configurations"] == 8,
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        raise ManifestFrozenError(f"pre-run trial accounting is inconsistent: {failed}")
    payload["invariants_checked"] = sorted(checks)
    return payload


def _trial_census(manifest: MLCandidateManifest) -> dict[tuple[MLTrialRole, str], int]:
    census: dict[tuple[MLTrialRole, str], int] = {}
    for t in manifest.trials:
        key = (t.trial_role, t.model_family.value)
        census[key] = census.get(key, 0) + 1
    return census


_PERFORMANCE_KEYS = {
    "pnl", "net_pnl", "gross_pnl", "sharpe", "daily_sharpe", "annualized_sharpe",
    "p_value", "q_value", "dsr_probability", "verdict", "auc", "roc_auc", "pr_auc",
    "log_loss", "brier", "accuracy", "precision", "recall", "returns", "drawdown",
    "reason_codes", "take_rate", "oof_prediction_hash",
}


def assert_no_performance_fields(manifest: MLCandidateManifest) -> None:
    """The manifest is frozen BEFORE performance: no result field may appear."""
    blob = manifest.canonical_json().lower()
    hits = sorted(k for k in _PERFORMANCE_KEYS if f'"{k}"' in blob)
    if hits:
        raise ManifestFrozenError(f"Phase 15 candidate manifest leaks performance keys: {hits}")


def freeze_candidate_manifest(path: str | Path) -> MLCandidateManifest:
    """Write the deterministic frozen manifest + its fingerprint to ``path``."""
    m = build_candidate_manifest()
    assert_no_performance_fields(m)
    payload = {
        "manifest_fingerprint": m.manifest_fingerprint(),
        "n_trials": m.n_trials(),
        "manifest": json.loads(m.canonical_json()),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return m
