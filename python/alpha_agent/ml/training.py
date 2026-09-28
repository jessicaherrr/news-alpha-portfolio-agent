"""The nested chronological training loop and its reproducibility record.

The loop is deliberately boring and fully determined by the frozen specs:

1. For each OUTER fold: build the purged/embargoed training set.
2. Inside that training set only, run the INNER folds over the frozen
   hyperparameter grid and pick the best by the predeclared inner objective --
   mean inner-fold log loss, minimised. Nothing outside the outer training
   window is visible here. The TAKE threshold is NOT selected: it is the frozen
   constant 0.50 (prompt 15A.1 s.3).
3. Refit the selected configuration on the whole outer training set.
4. Score the outer TEST rows once. Those are the OOF predictions, turned into
   TAKE/SKIP by ``probability_take >= 0.50``.

Everything fitted -- preprocessing, regime cut points, the model -- is fitted in
step 2 or 3, never on step-4 rows. Every fitted object records the row set it was
fitted on, so :func:`~alpha_agent.ml.predictions.assert_oof_provenance` and the
fold-isolation checks can verify that rather than trust it.

Phase 15A note: this module is orchestration and is exercised by the synthetic
fixtures only. It fits no real market data, and the research model families
cannot even be constructed in this environment -- scikit-learn is absent and
installing it needs approval.
"""
from __future__ import annotations

import platform
import sys

import numpy as np
from pydantic import BaseModel, Field, model_validator

from alpha_agent.ml.allowlist import assert_model_inputs_allowed
from alpha_agent.ml.enums import MetaLabelAction, MLRefusalReason, RegimeSpecKind
from alpha_agent.ml.errors import InsufficientEventsError, LeakageError, MLProtocolError
from alpha_agent.ml.features import MLFeatureSetSpec
from alpha_agent.ml.labels import MetaLabelSpec
from alpha_agent.ml.models import (
    DEFAULT_ML_RANDOM_SEED,
    FROZEN_TAKE_THRESHOLD,
    MLModelSpec,
    ModelSearchSpec,
    assert_frozen_take_threshold,
    assert_selection_is_within_search,
    make_backend,
)
from alpha_agent.ml.predictions import MLPrediction, MLPredictionFrame
from alpha_agent.ml.preprocessing import FoldFittedTransform, PreprocessingSpec
from alpha_agent.ml.regime import RegimeSpec
from alpha_agent.ml.splits import (
    EventTimeline,
    NestedCVSpec,
    assert_fold_is_causal,
    assert_global_temporal_isolation,
    build_inner_folds,
    build_outer_folds,
)
from alpha_agent.validation.fingerprint import fingerprint

ML_TRAINING_SPEC_SCHEMA = "ml-training-spec/1"
ML_TRAINING_REPORT_SCHEMA = "ml-training-report/1"

#: The inner-loop selection objective. Predeclared: the inner loop never selects
#: on economics, only on a proper scoring rule, so the C++ economic evaluation
#: stays a genuinely out-of-sample question.
INNER_OBJECTIVE = "mean_inner_fold_log_loss_minimised"

#: The inner loop selects ONE thing: the model hyperparameter configuration.
#: Ties are broken by the frozen grid's declaration order (first wins), which is
#: deterministic and predeclared, not by any property of the result.
INNER_SELECTION_DIMENSIONS: tuple[str, ...] = ("model_hyperparameter_configuration",)


class MLTrainingSpec(BaseModel):
    """How training runs. Frozen; nothing here is chosen after a result."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = ML_TRAINING_SPEC_SCHEMA
    random_seed: int = Field(default=DEFAULT_ML_RANDOM_SEED, ge=0)
    nested_cv: NestedCVSpec
    preprocessing: PreprocessingSpec
    regime: RegimeSpec
    inner_objective: str = INNER_OBJECTIVE
    #: NOT a search dimension. Frozen at 0.50 and refused if changed: the inner
    #: objective is threshold-independent, so a threshold sweep selected nothing
    #: while inflating the DSR search-breadth term.
    take_threshold: float = FROZEN_TAKE_THRESHOLD
    inner_selection_dimensions: tuple[str, ...] = INNER_SELECTION_DIMENSIONS

    @model_validator(mode="after")
    def _threshold_is_frozen(self) -> MLTrainingSpec:
        assert_frozen_take_threshold(self.take_threshold, what="MLTrainingSpec.take_threshold")
        return self

    def identity(self) -> str:
        return fingerprint(
            "mltrainingspec1",
            {
                "schema_version": self.schema_version,
                "random_seed": self.random_seed,
                "nested_cv": self.nested_cv.identity(),
                "preprocessing": self.preprocessing.identity(),
                "regime": self.regime.identity(),
                "inner_objective": self.inner_objective,
                "take_threshold": self.take_threshold,
                "inner_selection_dimensions": list(self.inner_selection_dimensions),
            },
        )


class OuterFoldRecord(BaseModel):
    """What one outer fold selected, fitted and scored."""

    model_config = {"frozen": True, "extra": "forbid"}

    fold_index: int
    test_start_ts_ns: int
    test_end_ts_ns: int
    n_train_rows: int
    n_test_rows: int
    n_purged_rows: int
    n_embargoed_rows: int
    selected_model_identity: str
    selected_hyperparameters: dict
    selected_threshold: float
    inner_objective_value: float
    n_inner_fits: int
    preprocessing_fingerprint: str
    model_artifact_fingerprint: str
    train_positive_rate: float
    test_positive_rate: float


class MLTrainingReport(BaseModel):
    """The full reproducibility record of one ML experiment (prompt 15A s.15)."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = ML_TRAINING_REPORT_SCHEMA
    experiment_identity: str
    experiment_label: str = ""

    model_family: str
    #: The PREDECLARED search procedure this run executed. It is the experiment's
    #: identity input; every ``OuterFoldRecord.selected_hyperparameters`` below is
    #: post-fit provenance about it and changes no identity.
    model_search_identity: str = ""
    n_search_configurations: int = 0
    selection_objective: str = INNER_OBJECTIVE
    selection_tie_break: str = ""
    training_spec_fingerprint: str
    meta_label_spec_fingerprint: str
    feature_set_fingerprint: str
    feature_order: tuple[str, ...]
    regime_fingerprint: str
    preprocessing_fingerprint: str
    nested_cv_fingerprint: str
    dataset_fingerprint: str
    primary_strategy_fingerprints: tuple[str, ...]
    random_seed: int
    code_commit: str = ""

    training_window: str
    evaluation_window: str
    n_events: int
    n_oof_predictions: int
    n_model_fits: int
    n_inner_fits: int
    #: DISTINCT predeclared model configurations the inner loop inspected. This
    #: is the DSR search-breadth quantity: re-selecting among the SAME
    #: configurations in another outer fold is a training operation, not a new
    #: model hypothesis, so it does not enlarge this count.
    n_distinct_configurations_inspected: int
    #: Bookkeeping only: distinct configurations x outer folds. NEVER a DSR term.
    n_configuration_fold_inspections: int
    take_threshold: float = FROZEN_TAKE_THRESHOLD
    inner_objective: str = INNER_OBJECTIVE

    folds: tuple[OuterFoldRecord, ...]
    oof_prediction_hash: str
    take_rate: float
    environment: dict = Field(default_factory=dict)

    def report_fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("experiment_label", None)
        payload.pop("environment", None)     # interpreter build is not semantics
        return fingerprint("mltrainingreport1", payload, allow_non_finite=True)


def environment_record() -> dict:
    """Recorded alongside every artifact, never part of a fingerprint."""
    import importlib.util

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn_installed": importlib.util.find_spec("sklearn") is not None,
        "lightgbm_installed": importlib.util.find_spec("lightgbm") is not None,
    }


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _fit_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    *,
    model_spec: MLModelSpec,
    preprocessing: PreprocessingSpec,
    feature_order: tuple[str, ...],
) -> tuple[FoldFittedTransform, object]:
    """Fit preprocessing then the model, both on EXACTLY ``rows``."""
    pre = FoldFittedTransform(preprocessing, fit_row_indices=rows, feature_order=feature_order)
    pre.fit(X[rows], row_indices=rows)
    backend = make_backend(model_spec, feature_order=feature_order)
    backend.fit(pre.transform(X[rows]), y[rows])
    return pre, backend


def _regime_state_columns(
    X: np.ndarray,
    *,
    train_rows: np.ndarray,
    input_col_indices: tuple[int, ...],
    regime: RegimeSpec,
) -> np.ndarray:
    """Fold-local deterministic 2D vol x signed-trend regime (Phase 15B.1b).

    For EACH regime input axis independently: the ``regime.quantile_probs``
    empirical quantile cut points of that column over ``train_rows`` only
    (``regime.quantile_interpolation``), then ``np.digitize`` with
    ``regime.digitize_right`` into ``regime.axis_bucket_count`` ordered buckets.
    The regime state is the CARTESIAN PRODUCT of the per-axis buckets, indexed
    axis-0-major (``state = b0 * K + b1``), one-hot into ``regime.n_states``
    columns.

    The two inputs are used on their own scales -- NO standardisation, no
    averaging. A non-finite regime input on a row that reached here is an
    upstream bug (a missing regime feature must exclude the event via the typed
    ``FEATURES_MISSING_AT_DECISION`` reason, never be imputed): it raises.
    """
    n = X.shape[0]
    k = len(input_col_indices)
    bucket_count = regime.axis_bucket_count
    probs = list(regime.quantile_probs)
    per_axis = np.empty((n, k), dtype=np.int64)
    for ax, col in enumerate(input_col_indices):
        v = np.asarray(X[:, col], dtype=float)
        if not np.all(np.isfinite(v)):
            raise LeakageError(
                f"regime input axis {ax} (column {col}) has non-finite values on rows that "
                "reached the model matrix; a missing regime feature must exclude the event "
                "upstream with a typed reason, never be zero-imputed here"
            )
        train_v = v[train_rows]
        cuts = (
            np.quantile(train_v, probs, method=regime.quantile_interpolation)
            if train_v.size
            else np.zeros(len(probs))
        )
        b = np.digitize(v, cuts, right=regime.digitize_right)
        per_axis[:, ax] = np.clip(b, 0, bucket_count - 1)

    state = np.zeros(n, dtype=np.int64)
    for ax in range(k):
        state = state * bucket_count + per_axis[:, ax]     # axis-0-major
    onehot = np.zeros((n, bucket_count ** k), dtype=float)
    onehot[np.arange(n), state] = 1.0
    return onehot


def run_nested_cv(
    *,
    X: np.ndarray,
    y: np.ndarray,
    event_ids: tuple[str, ...],
    root_symbols: tuple[str, ...],
    timeline: EventTimeline,
    training_spec: MLTrainingSpec,
    model_search: ModelSearchSpec,
    feature_set: MLFeatureSetSpec,
    meta_label_spec: MetaLabelSpec,
    dataset_fingerprint: str,
    primary_strategy_fingerprints: tuple[str, ...],
    experiment_identity: str,
    experiment_label: str = "",
    code_commit: str = "",
    matrix_columns: tuple[str, ...] | None = None,
    feature_columns: tuple[str, ...] | None = None,
    regime_input_column_indices: tuple[int, ...] | None = None,
) -> tuple[MLPredictionFrame, MLTrainingReport]:
    """Run the frozen nested chronological protocol end to end.

    Returns the OOF prediction frame and the reproducibility report. It never
    computes an economic number: PnL comes from the C++ core, downstream.

    ``feature_columns`` overrides the fitted-matrix column identity (default: the
    feature set's ordered aliases). ``regime_input_column_indices`` -- when the
    training spec's regime is a deterministic causal one -- names the ``X``
    columns the fold-local regime bucketing reads; the one-hot bucket columns are
    appended for every fit in that fold and their cut points are TRAIN-fold only.
    """
    cv = training_spec.nested_cv
    n = timeline.n_events
    if X.shape[0] != n or y.shape[0] != n or len(event_ids) != n:
        raise ValueError("X / y / event_ids / timeline lengths disagree")
    regime = training_spec.regime
    regime_active = (
        regime is not None
        and regime.kind is not RegimeSpecKind.NONE
        and regime_input_column_indices is not None
    )

    # The model may consume ONLY decision-time inputs. Checked before a single
    # fit, so a label/audit column can never reach a matrix even transiently.
    assert_model_inputs_allowed(
        matrix_columns if matrix_columns is not None else feature_set.ordered_aliases,
        feature_set,
        what="run_nested_cv model matrix",
    )

    # The EXECUTED search must be the DECLARED search: the procedure that entered
    # this experiment's pre-run identity is the one that runs. A mismatch is an
    # integrity failure, not a second experiment.
    if model_search.objective_value != training_spec.inner_objective:
        raise MLProtocolError(
            f"declared selection objective {model_search.objective_value!r} does not "
            f"match the training spec's inner objective {training_spec.inner_objective!r}; the "
            "objective is part of pre-run identity and the loop may not silently use another"
        )
    if model_search.random_seed != training_spec.random_seed:
        raise MLProtocolError(
            f"declared search seed {model_search.random_seed} does not match the training "
            f"seed {training_spec.random_seed}; the seed is part of pre-run identity"
        )

    outer_folds = build_outer_folds(cv, timeline)
    grid = model_search.grid_specs()
    feature_order = feature_columns if feature_columns is not None else feature_set.ordered_aliases
    if X.shape[1] != len(feature_order):
        raise ValueError(
            f"X has {X.shape[1]} columns but feature_order declares {len(feature_order)}"
        )
    regime_cols = regime.output_columns() if regime_active else ()
    feature_order_effective = (*feature_order, *regime_cols)

    def _fold_matrix(train_rows: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
        if not regime_active:
            return X, feature_order
        reg = _regime_state_columns(
            X,
            train_rows=train_rows,
            input_col_indices=regime_input_column_indices,
            regime=regime,
        )
        return np.hstack([X, reg]), feature_order_effective

    predictions: list[MLPrediction] = []
    records: list[OuterFoldRecord] = []
    n_fits = 0
    n_inner_fits = 0
    n_inspected = 0          # configuration x outer-fold inspections (bookkeeping)

    for outer in outer_folds:
        assert_fold_is_causal(outer, timeline)
        # Training is POOLED across roots; chronological isolation is a GLOBAL
        # property, asserted per root rather than assumed from the construction.
        assert_global_temporal_isolation(outer, timeline, root_symbols)
        train_rows = np.asarray(outer.train_row_indices, dtype=int)
        test_rows = np.asarray(outer.test_row_indices, dtype=int)
        if train_rows.size < cv.min_train_events:
            raise InsufficientEventsError(
                f"outer fold {outer.fold_index} has {train_rows.size} training events, below the "
                f"predeclared minimum {cv.min_train_events}; this scope is REFUSED, not trained, "
                "and is recorded as an insufficient-evidence failure. The gate is frozen and is "
                "never lowered after real counts are seen.",
                reason=MLRefusalReason.INSUFFICIENT_TRAIN_EVENTS,
                detail={
                    "fold_kind": "OUTER",
                    "fold_index": outer.fold_index,
                    "n_events": int(train_rows.size),
                    "min_required": cv.min_train_events,
                },
            )
        if test_rows.size < cv.min_test_events:
            raise InsufficientEventsError(
                f"outer fold {outer.fold_index} has {test_rows.size} test events, below the "
                f"predeclared minimum {cv.min_test_events}; this scope is REFUSED, not scored",
                reason=MLRefusalReason.INSUFFICIENT_TEST_EVENTS,
                detail={
                    "fold_kind": "OUTER",
                    "fold_index": outer.fold_index,
                    "n_events": int(test_rows.size),
                    "min_required": cv.min_test_events,
                },
            )

        # -- INNER: selection, inside the outer training window only ----------
        inner_folds = build_inner_folds(cv, timeline, outer)
        for inner in inner_folds:
            assert_global_temporal_isolation(inner, timeline, root_symbols)
        inner_train_set = set(outer.train_row_indices)
        # (objective, grid position, spec). The ONLY selected dimension is the
        # model hyperparameter configuration; ties break on the frozen grid's
        # declaration order, which is predeclared and deterministic.
        best: tuple[float, int, MLModelSpec] | None = None
        for grid_position, model_spec in enumerate(grid):
            n_inspected += 1
            fold_losses: list[float] = []
            for inner in inner_folds:
                itr = np.asarray(
                    [i for i in inner.train_row_indices if i in inner_train_set], dtype=int
                )
                ite = np.asarray(
                    [i for i in inner.test_row_indices if i in inner_train_set], dtype=int
                )
                if itr.size == 0 or ite.size == 0:
                    continue
                Xi, feat_i = _fold_matrix(itr)
                pre, backend = _fit_pipeline(
                    Xi, y, itr,
                    model_spec=model_spec,
                    preprocessing=training_spec.preprocessing,
                    feature_order=feat_i,
                )
                n_inner_fits += 1
                n_fits += 1
                p = backend.predict_proba_positive(pre.transform(Xi[ite], row_indices=ite))
                fold_losses.append(_log_loss(y[ite], p))
            if not fold_losses:
                continue
            objective = float(np.mean(fold_losses))
            if best is None or (objective, grid_position) < (best[0], best[1]):
                best = (objective, grid_position, model_spec)

        if best is None:
            raise InsufficientEventsError(
                f"outer fold {outer.fold_index} produced no usable inner fold; the scope has too "
                "few events for nested selection",
                reason=MLRefusalReason.NO_USABLE_INNER_FOLD,
                detail={"fold_kind": "OUTER", "fold_index": outer.fold_index},
            )
        _, _grid_position, selected_spec = best
        # POST-FIT PROVENANCE, recorded and audited -- never identity. It must
        # still be a point the predeclared procedure could actually reach.
        assert_selection_is_within_search(
            model_search, selected_spec, what=f"outer fold {outer.fold_index} selection"
        )
        selected_thr = training_spec.take_threshold

        # -- OUTER: refit on the whole (purged, embargoed) training set --------
        Xo, feat_o = _fold_matrix(train_rows)
        pre, backend = _fit_pipeline(
            Xo, y, train_rows,
            model_spec=selected_spec,
            preprocessing=training_spec.preprocessing,
            feature_order=feat_o,
        )
        n_fits += 1
        proba = backend.predict_proba_positive(
            pre.transform(Xo[test_rows], row_indices=test_rows)
        )
        model_fp = fingerprint(
            "mlmodelartifact1",
            {
                "model_spec": selected_spec.identity(),
                "preprocessing": pre.fingerprint(),
                "train_rows": [int(i) for i in train_rows],
                "outer_fold": outer.fold_index,
            },
        )
        for row, p in zip(test_rows, proba):
            predictions.append(
                MLPrediction(
                    event_id=event_ids[int(row)],
                    row_index=int(row),
                    root_symbol=root_symbols[int(row)],
                    decision_timestamp=timeline.decision_ts_ns[int(row)],
                    probability_take=float(p),
                    threshold=selected_thr,
                    action=(
                        MetaLabelAction.TAKE if float(p) >= selected_thr
                        else MetaLabelAction.SKIP
                    ),
                    label=int(y[int(row)]),
                    outer_fold_index=outer.fold_index,
                    producing_model_fingerprint=model_fp,
                )
            )

        records.append(
            OuterFoldRecord(
                fold_index=outer.fold_index,
                test_start_ts_ns=outer.test_start_ts_ns,
                test_end_ts_ns=outer.test_end_ts_ns,
                n_train_rows=int(train_rows.size),
                n_test_rows=int(test_rows.size),
                n_purged_rows=len(outer.purged_row_indices),
                n_embargoed_rows=len(outer.embargoed_row_indices),
                selected_model_identity=selected_spec.identity(),
                selected_hyperparameters=dict(selected_spec.hyperparameters),
                selected_threshold=selected_thr,
                inner_objective_value=float(best[0]),
                n_inner_fits=len(inner_folds),
                preprocessing_fingerprint=pre.fingerprint(),
                model_artifact_fingerprint=model_fp,
                train_positive_rate=float(np.mean(y[train_rows])) if train_rows.size else 0.0,
                test_positive_rate=float(np.mean(y[test_rows])) if test_rows.size else 0.0,
            )
        )

    predictions.sort(key=lambda p: (p.decision_timestamp, p.event_id))
    frame = MLPredictionFrame(
        experiment_label=experiment_label, predictions=tuple(predictions)
    )
    report = MLTrainingReport(
        experiment_identity=experiment_identity,
        experiment_label=experiment_label,
        model_family=model_search.family.value,
        model_search_identity=model_search.identity(),
        n_search_configurations=model_search.n_configurations,
        selection_objective=model_search.objective_value,
        selection_tie_break=model_search.tie_break_value,
        training_spec_fingerprint=training_spec.identity(),
        meta_label_spec_fingerprint=meta_label_spec.identity(),
        feature_set_fingerprint=feature_set.identity(),
        feature_order=feature_order_effective,
        regime_fingerprint=training_spec.regime.identity(),
        preprocessing_fingerprint=training_spec.preprocessing.identity(),
        nested_cv_fingerprint=cv.identity(),
        dataset_fingerprint=dataset_fingerprint,
        primary_strategy_fingerprints=tuple(sorted(primary_strategy_fingerprints)),
        random_seed=training_spec.random_seed,
        code_commit=code_commit,
        training_window=f"{cv.corpus_start_ts_ns}..{cv.corpus_end_ts_ns} (exclusive)",
        evaluation_window=(
            f"{outer_folds[0].test_start_ts_ns}..{outer_folds[-1].test_end_ts_ns} (exclusive)"
            if outer_folds else ""
        ),
        n_events=n,
        n_oof_predictions=len(predictions),
        n_model_fits=n_fits,
        n_inner_fits=n_inner_fits,
        n_distinct_configurations_inspected=len(grid),
        n_configuration_fold_inspections=n_inspected,
        take_threshold=training_spec.take_threshold,
        inner_objective=training_spec.inner_objective,
        folds=tuple(records),
        oof_prediction_hash=frame.prediction_hash(),
        take_rate=frame.take_rate,
        environment=environment_record(),
    )
    return frame, report
