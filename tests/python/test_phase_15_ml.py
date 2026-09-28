"""Phase 15A -- the frozen ML meta-labeling protocol.

Every test here is offline, deterministic and synthetic. Nothing trains on real
market data, nothing inspects ML performance, nothing touches 2025 and nothing
touches the network -- which is itself part of what is being proven.
"""
from __future__ import annotations

import json
from pathlib import Path

import alpha_agent.features.compute  # noqa: F401 -- registers the feature kinds
import numpy as np
import pytest
from alpha_agent.features.spec import FeatureSpec
from alpha_agent.ml import (
    EventTimeline,
    FoldFittedTransform,
    FoldIsolationError,
    LeakageError,
    MetaLabelAction,
    MetaLabelSpec,
    MLFeature,
    MLFeatureSetSpec,
    MLModelFamily,
    MLModelSpec,
    MLTrainingSpec,
    NestedCVSpec,
    PreprocessingSpec,
    RandomSplitForbidden,
    RegimeSpec,
    assert_closed_action_space,
    assert_event_causality,
    assert_feature_set_signal_safe,
    assert_fold_is_causal,
    assert_oof_provenance,
    attribute_trades_to_episodes,
    build_meta_label_events,
    build_meta_labeled_schedule,
    build_outer_folds,
    extract_primary_episodes,
    ml_dependency_status,
    refuse_random_split,
    run_nested_cv,
)
from alpha_agent.ml.enums import EpisodeExclusionReason, PreprocessingKind, RegimeSpecKind
from alpha_agent.ml.errors import MLDependencyMissing
from alpha_agent.ml.execution import assert_aligned_evaluation_support
from alpha_agent.ml.guards import (
    HOLDOUT_START_NS,
    HoldoutAccessError,
    assert_exclusive_corpus_bound,
    assert_half_open_window,
    assert_no_holdout_in_config_payload,
    assert_no_holdout_timestamps,
    forbid_network,
)
from alpha_agent.ml.identity import MLExperimentSpec
from alpha_agent.ml.manifest import (
    ML_FEATURE_SET,
    NESTED_CV,
    assert_no_performance_fields,
    build_candidate_manifest,
)
from alpha_agent.ml.models import (
    DEFAULT_ML_RANDOM_SEED,
    ModelSearchSpec,
    assert_selection_is_within_search,
    frozen_model_search,
    make_backend,
)
from alpha_agent.ml.preprocessing import assert_no_whole_dataset_fit
from alpha_agent.ml.regime import assert_regime_label_is_not_economic
from alpha_agent.ml.splits import (
    assert_no_train_test_overlap_across_folds,
    build_inner_folds,
)
from alpha_agent.ml.trade_export import (
    CLOSED_TRADE_COLUMNS,
    FILL_COLUMNS,
    read_closed_trades_csv,
    read_fills_csv,
)
from alpha_agent.ml.trials import (
    MLIdentityPlanes,
    assert_one_identity_per_trial,
    experiment_spec_for_trial,
)
from ml_fixtures import (
    DAY_NS,
    make_economics,
    make_schedule,
    make_trade,
    synthetic_dataset,
    ts,
)

CLI = Path("build/cpp/cpp/quant_backtest_targets_csv")


# ==========================================================================
# 1. Episode + label semantics
# ==========================================================================
def test_episodes_are_pure_target_intent():
    # flat, long, long, flat, short, short, flat
    sched = make_schedule([0, 1, 1, 0, -1, -1, 0])
    eps = extract_primary_episodes(sched)
    assert [(e.side, e.entry_row_index, e.exit_row_index) for e in eps] == [(1, 1, 3), (-1, 4, 6)]
    assert all(e.is_terminated for e in eps)


def test_same_sign_size_change_does_not_open_a_new_episode():
    sched = make_schedule([0, 1, 2, 1, 0])
    eps = extract_primary_episodes(sched)
    assert len(eps) == 1
    assert eps[0].entry_row_index == 1 and eps[0].exit_row_index == 4


def test_reversal_closes_one_episode_and_opens_the_next():
    sched = make_schedule([0, 1, 1, -1, -1, 0])
    eps = extract_primary_episodes(sched)
    assert [(e.side, e.entry_row_index, e.exit_row_index) for e in eps] == [(1, 1, 3), (-1, 3, 5)]


def test_unterminated_episode_yields_no_label():
    sched = make_schedule([0, 1, 1, 1])
    eps = extract_primary_episodes(sched)
    assert len(eps) == 1 and not eps[0].is_terminated
    events = build_meta_label_events(
        episodes=eps,
        economics={0: make_economics(0, net=100.0, horizon_day=3)},
        spec=MetaLabelSpec(),
        strategy_family="tsmom",
        dataset_fingerprint="ds1",
        feature_set_fingerprint="fs1",
        target_schedule_hash="tsh1",
    )
    assert events.n_events == 0
    assert events.excluded[0].reason is EpisodeExclusionReason.UNTERMINATED_AT_CORPUS_END


def test_episode_with_no_fills_is_excluded_not_labelled_zero():
    sched = make_schedule([0, 1, 1, 0])
    eps = extract_primary_episodes(sched)
    econ, _ = attribute_trades_to_episodes(eps, ())
    events = build_meta_label_events(
        episodes=eps, economics=econ, spec=MetaLabelSpec(), strategy_family="tsmom",
        dataset_fingerprint="ds1", feature_set_fingerprint="fs1", target_schedule_hash="tsh1",
    )
    assert events.n_events == 0
    assert events.excluded[0].reason is EpisodeExclusionReason.NO_ATTRIBUTED_FILLS


def test_trade_attribution_includes_roll_legs_inside_the_episode():
    # long from row 1 to row 5; a roll closes the old contract mid-episode.
    sched = make_schedule([0, 1, 1, 1, 1, 0])
    eps = extract_primary_episodes(sched)
    trades = (
        make_trade(open_day=1, close_day=3, net=-40.0, reason="roll", index=0),
        make_trade(open_day=3, close_day=5, net=150.0, reason="signal", index=1),
    )
    econ, unattributed = attribute_trades_to_episodes(eps, trades)
    assert unattributed == ()
    assert econ[0].n_attributed_trades == 2
    assert econ[0].n_roll_closes == 1
    assert econ[0].net_pnl_usd == pytest.approx(110.0)
    # the information horizon is the LAST fill close, not the exit decision
    assert econ[0].information_horizon_ts_ns == ts(5)


def test_label_is_the_sign_of_cpp_fill_derived_net_pnl():
    sched = make_schedule([0, 1, 1, 0, -1, -1, 0])
    eps = extract_primary_episodes(sched)
    trades = (
        make_trade(open_day=1, close_day=3, net=120.0, index=0),
        make_trade(open_day=4, close_day=6, net=-55.0, direction=-1, index=1),
    )
    econ, _ = attribute_trades_to_episodes(eps, trades)
    events = build_meta_label_events(
        episodes=eps, economics=econ, spec=MetaLabelSpec(), strategy_family="tsmom",
        dataset_fingerprint="ds1", feature_set_fingerprint="fs1", target_schedule_hash="tsh1",
    )
    assert [e.label for e in events.events] == [1, 0]
    assert [e.primary_side for e in events.events] == [1, -1]


def test_zero_pnl_episode_labels_zero():
    sched = make_schedule([0, 1, 1, 0])
    eps = extract_primary_episodes(sched)
    econ, _ = attribute_trades_to_episodes(eps, (make_trade(open_day=1, close_day=3, net=0.0),))
    events = build_meta_label_events(
        episodes=eps, economics=econ, spec=MetaLabelSpec(), strategy_family="tsmom",
        dataset_fingerprint="ds1", feature_set_fingerprint="fs1", target_schedule_hash="tsh1",
    )
    assert events.events[0].label == 0


# ==========================================================================
# 2. Labels may look forward; features may not
# ==========================================================================
def test_label_looks_forward_but_features_do_not():
    sched = make_schedule([0, 1, 1, 0])
    eps = extract_primary_episodes(sched)
    econ, _ = attribute_trades_to_episodes(eps, (make_trade(open_day=1, close_day=3, net=10.0),))
    ev = build_meta_label_events(
        episodes=eps, economics=econ, spec=MetaLabelSpec(), strategy_family="tsmom",
        dataset_fingerprint="ds1", feature_set_fingerprint="fs1", target_schedule_hash="tsh1",
    ).events[0]
    assert ev.feature_timestamp <= ev.decision_timestamp
    assert ev.label_end_timestamp > ev.decision_timestamp     # the label DOES look forward
    assert_event_causality(ev)


def test_feature_after_the_decision_is_refused():
    sched = make_schedule([0, 1, 1, 0])
    eps = extract_primary_episodes(sched)
    econ, _ = attribute_trades_to_episodes(eps, (make_trade(open_day=1, close_day=3, net=10.0),))
    ev = build_meta_label_events(
        episodes=eps, economics=econ, spec=MetaLabelSpec(), strategy_family="tsmom",
        dataset_fingerprint="ds1", feature_set_fingerprint="fs1", target_schedule_hash="tsh1",
    ).events[0]
    with pytest.raises(LeakageError, match="AFTER"):
        ev.model_copy(update={"feature_timestamp": ev.decision_timestamp + DAY_NS}).model_validate(
            ev.model_copy(update={"feature_timestamp": ev.decision_timestamp + DAY_NS}).model_dump()
        )


def test_label_resolving_at_the_decision_is_refused():
    sched = make_schedule([0, 1, 1, 0])
    eps = extract_primary_episodes(sched)
    econ, _ = attribute_trades_to_episodes(eps, (make_trade(open_day=1, close_day=3, net=10.0),))
    ev = build_meta_label_events(
        episodes=eps, economics=econ, spec=MetaLabelSpec(), strategy_family="tsmom",
        dataset_fingerprint="ds1", feature_set_fingerprint="fs1", target_schedule_hash="tsh1",
    ).events[0]
    payload = ev.model_dump()
    payload["label_end_timestamp"] = ev.decision_timestamp
    with pytest.raises(LeakageError, match="does not extend past"):
        type(ev).model_validate(payload)


# ==========================================================================
# 3. Feature universe
# ==========================================================================
def test_frozen_feature_set_is_signal_safe_and_point_in_time():
    assert_feature_set_signal_safe(ML_FEATURE_SET)
    assert len(ML_FEATURE_SET.ordered_aliases) == len(set(ML_FEATURE_SET.ordered_aliases))


def test_feature_order_is_semantic():
    a = MLFeatureSetSpec(features=(
        MLFeature(alias="x", spec=FeatureSpec(kind="realized_vol", params={"window": 20})),
        MLFeature(alias="y", spec=FeatureSpec(kind="atr", params={"window": 14})),
    ))
    b = MLFeatureSetSpec(features=(a.features[1], a.features[0]))
    assert a.identity() != b.identity()


def test_ad_hoc_feature_kind_is_refused():
    with pytest.raises(ValueError, match="not registered"):
        MLFeature(alias="hack", spec=FeatureSpec(kind="my_pandas_column", params={}))


def test_a_look_ahead_unsafe_source_makes_the_feature_set_refused(monkeypatch):
    """A retrospectively back-adjusted source is not signal-safe, and the check
    that decides that is the Feature Engine's, not the ML layer's opinion."""
    from alpha_agent.features.safety import SourceSafety
    from alpha_agent.ml import features as mlf

    retrospective = SourceSafety(
        point_in_time_safe=False, signal_safe=False, execution_price_safe=False,
        reasons={"source": "retrospective back-adjustment applies future roll gaps"},
    )
    monkeypatch.setitem(
        mlf._CAUSAL_SOURCE_SAFETY, ML_FEATURE_SET.price_domain, retrospective
    )
    with pytest.raises(LeakageError, match="not signal-safe"):
        assert_feature_set_signal_safe(ML_FEATURE_SET)


# ==========================================================================
# 4. Preprocessing sees the training fold only
# ==========================================================================
def test_preprocessing_fit_outside_its_declared_fold_is_refused():
    X = np.arange(40, dtype=float).reshape(20, 2)
    train = np.arange(10)
    t = FoldFittedTransform(PreprocessingSpec(), fit_row_indices=train, feature_order=("a", "b"))
    with pytest.raises(FoldIsolationError, match="outside its declared training fold"):
        t.fit(X[np.arange(20)], row_indices=np.arange(20))


def test_transform_refuses_to_score_rows_it_was_fitted_on():
    X = np.arange(40, dtype=float).reshape(20, 2)
    train = np.arange(10)
    t = FoldFittedTransform(PreprocessingSpec(), fit_row_indices=train, feature_order=("a", "b"))
    t.fit(X[train], row_indices=train)
    t.transform(X[10:], row_indices=np.arange(10, 20))          # fine: genuinely OOS
    with pytest.raises(FoldIsolationError, match="not out-of-sample"):
        t.transform(X[:10], row_indices=train)


def test_whole_dataset_standardization_is_refused():
    X = np.arange(40, dtype=float).reshape(20, 2)
    everything = np.arange(20)
    t = FoldFittedTransform(PreprocessingSpec(), fit_row_indices=everything,
                            feature_order=("a", "b"))
    t.fit(X, row_indices=everything)
    with pytest.raises(LeakageError, match="whole dataset"):
        assert_no_whole_dataset_fit(t, all_row_indices=everything)


def test_preprocessing_statistics_come_only_from_training_rows():
    X = np.vstack([np.zeros((10, 1)), np.full((10, 1), 1000.0)])
    train = np.arange(10)
    t = FoldFittedTransform(
        PreprocessingSpec(kind=PreprocessingKind.MEDIAN_IMPUTE_STANDARDIZE),
        fit_row_indices=train, feature_order=("a",),
    )
    t.fit(X[train], row_indices=train)
    # The 1000.0 block is in the future; it must not have moved the centre.
    assert t.statistics()["center"] == [0.0]


# ==========================================================================
# 5. No random split; purge and embargo
# ==========================================================================
def test_random_split_is_refused():
    with pytest.raises(RandomSplitForbidden, match="chronological"):
        refuse_random_split(shuffle=True)


def test_outer_folds_are_chronological_and_causal():
    _X, _y, timeline, _ids, _roots = synthetic_dataset()
    folds = build_outer_folds(NESTED_CV, timeline)
    assert len(folds) == NESTED_CV.n_outer_folds
    for f in folds:
        assert_fold_is_causal(f, timeline)
    assert_no_train_test_overlap_across_folds(folds)


def test_purge_removes_training_rows_whose_label_overlaps_the_test_block():
    _X, _y, timeline, _ids, _roots = synthetic_dataset(label_horizon_days=20)
    folds = build_outer_folds(NESTED_CV, timeline)
    purged = sum(len(f.purged_row_indices) for f in folds)
    embargoed = sum(len(f.embargoed_row_indices) for f in folds)
    assert purged + embargoed > 0, "overlapping labels must actually be removed"
    for f in folds:
        for i in f.purged_row_indices:
            assert timeline.label_end_ts_ns[i] > f.test_start_ts_ns


def test_longer_label_horizon_purges_strictly_more():
    _X, _y, short, _ids, _roots = synthetic_dataset(label_horizon_days=5)
    _X2, _y2, long_, _ids2, _roots2 = synthetic_dataset(label_horizon_days=90)
    n_short = sum(len(f.train_row_indices) for f in build_outer_folds(NESTED_CV, short))
    n_long = sum(len(f.train_row_indices) for f in build_outer_folds(NESTED_CV, long_))
    assert n_long < n_short


def test_inner_folds_never_see_the_outer_test_block():
    _X, _y, timeline, _ids, _roots = synthetic_dataset()
    for outer in build_outer_folds(NESTED_CV, timeline):
        for inner in build_inner_folds(NESTED_CV, timeline, outer):
            assert inner.test_end_ts_ns <= outer.test_start_ts_ns
            for i in inner.train_row_indices + inner.test_row_indices:
                assert timeline.decision_ts_ns[i] < outer.test_start_ts_ns


def test_a_test_block_escaping_the_corpus_into_2025_is_refused():
    with pytest.raises(ValueError, match="escapes the development corpus"):
        NestedCVSpec(outer_test_blocks=((ts(30), HOLDOUT_START_NS + DAY_NS),))


# ==========================================================================
# 6. OOF provenance and determinism
# ==========================================================================
def _training_spec(seed: int = 20250101) -> MLTrainingSpec:
    return MLTrainingSpec(
        random_seed=seed,
        nested_cv=NESTED_CV,
        preprocessing=PreprocessingSpec(kind=PreprocessingKind.MEDIAN_IMPUTE_STANDARDIZE),
        regime=RegimeSpec(kind=RegimeSpecKind.NONE),
    )


#: The predeclared search procedure the synthetic fixture "searches": a
#: single-point grid. It is a PROCEDURE, not a selected point -- the loop still
#: runs the inner selection over it.
_SYNTHETIC_SEARCH = frozen_model_search(MLModelFamily.SYNTHETIC_DETERMINISTIC)


def _run(seed: int = 20250101):
    X, y, timeline, ids, roots = synthetic_dataset()
    fs = MLFeatureSetSpec(features=tuple(
        MLFeature(alias=f"f{i}", spec=FeatureSpec(kind="realized_vol", params={"window": 10 + i}))
        for i in range(X.shape[1])
    ))
    return run_nested_cv(
        X=X, y=y, event_ids=ids, root_symbols=roots, timeline=timeline,
        training_spec=_training_spec(seed),
        model_search=frozen_model_search(
            MLModelFamily.SYNTHETIC_DETERMINISTIC, random_seed=seed
        ),
        feature_set=fs, meta_label_spec=MetaLabelSpec(),
        dataset_fingerprint="ds-synthetic", primary_strategy_fingerprints=("stratdsl1:x",),
        experiment_identity="experiment1:synthetic",
    ), timeline


def _run_on(X, y, timeline, ids, roots, **kwargs):
    """Run the frozen protocol on an explicit synthetic dataset."""
    fs = MLFeatureSetSpec(features=tuple(
        MLFeature(alias=f"f{i}", spec=FeatureSpec(kind="realized_vol", params={"window": 10 + i}))
        for i in range(X.shape[1])
    ))
    return run_nested_cv(
        X=X, y=y, event_ids=ids, root_symbols=roots, timeline=timeline,
        training_spec=_training_spec(),
        model_search=_SYNTHETIC_SEARCH,
        feature_set=fs, meta_label_spec=MetaLabelSpec(),
        dataset_fingerprint="ds-synthetic", primary_strategy_fingerprints=("stratdsl1:x",),
        experiment_identity="experiment1:synthetic", **kwargs,
    )


def test_oof_prediction_never_comes_from_a_model_trained_on_that_row():
    (frame, _report), timeline = _run()
    folds = build_outer_folds(NESTED_CV, timeline)
    assert frame.n > 0
    assert_oof_provenance(frame, folds)


def test_every_row_is_scored_at_most_once():
    (frame, _report), _timeline = _run()
    ids = [p.event_id for p in frame.predictions]
    assert len(ids) == len(set(ids))


def test_same_seed_gives_identical_predictions():
    (frame_a, report_a), _ = _run(seed=11)
    (frame_b, report_b), _ = _run(seed=11)
    assert frame_a.prediction_hash() == frame_b.prediction_hash()
    assert report_a.report_fingerprint() == report_b.report_fingerprint()


def test_training_report_records_the_reproducibility_fields():
    (_frame, report), _ = _run()
    for field in (
        "training_spec_fingerprint", "meta_label_spec_fingerprint", "feature_set_fingerprint",
        "regime_fingerprint", "preprocessing_fingerprint", "nested_cv_fingerprint",
        "dataset_fingerprint", "oof_prediction_hash",
    ):
        assert getattr(report, field), f"{field} must be recorded"
    assert report.n_oof_producing_fits if hasattr(report, "n_oof_producing_fits") else True
    assert report.random_seed == 20250101
    assert report.n_model_fits > report.n_oof_predictions * 0  # sanity: fits were counted


def test_model_artifact_metadata_preserves_ordered_feature_identity():
    spec = MLModelSpec(family=MLModelFamily.SYNTHETIC_DETERMINISTIC, hyperparameters={"ridge": 1.0})
    order = ("mom_20", "rvol_20", "atr_14")
    backend = make_backend(spec, feature_order=order)
    backend.fit(np.arange(30, dtype=float).reshape(10, 3), np.array([0, 1] * 5))
    meta = backend.artifact_metadata()
    assert tuple(meta["feature_order"]) == order          # ORDER, not a set
    assert meta["model_spec"] == spec.identity()
    assert meta["random_seed"] == spec.random_seed


def test_prediction_action_cannot_disagree_with_its_probability():
    from alpha_agent.ml.predictions import MLPrediction

    with pytest.raises(ValueError, match="disagrees"):
        MLPrediction(
            event_id="E", row_index=0, root_symbol="NQ", decision_timestamp=ts(1),
            probability_take=0.9, threshold=0.5, action=MetaLabelAction.SKIP,
            outer_fold_index=0, producing_model_fingerprint="m",
        )


# ==========================================================================
# 7. Closed TAKE/SKIP execution semantics
# ==========================================================================
def test_skip_returns_the_episode_to_the_neutral_target():
    sched = make_schedule([0, 1, 1, 0, -1, -1, 0])
    eps = extract_primary_episodes(sched)
    meta, diff = build_meta_labeled_schedule(
        sched, eps, {0: MetaLabelAction.SKIP, 1: MetaLabelAction.TAKE}
    )
    assert [r.target_units for r in meta.rows] == [0, 0, 0, 0, -1, -1, 0]
    assert diff.n_skip == 1 and diff.n_take == 1 and diff.n_rows_changed == 2


def test_take_preserves_the_primary_exactly():
    sched = make_schedule([0, 1, 1, 0, -1, -1, 0])
    eps = extract_primary_episodes(sched)
    meta, diff = build_meta_labeled_schedule(sched, eps, {})     # default TAKE
    assert meta.schedule_hash() == sched.schedule_hash()
    assert diff.n_rows_changed == 0


def test_meta_labeled_schedule_keeps_aligned_evaluation_support():
    sched = make_schedule([0, 1, 1, 0, -1, -1, 0])
    eps = extract_primary_episodes(sched)
    meta, _ = build_meta_labeled_schedule(sched, eps, {0: MetaLabelAction.SKIP})
    assert_aligned_evaluation_support(sched, meta)
    assert_closed_action_space(meta, sched)
    # same fingerprint (the DSL boundary is frozen), different schedule hash
    assert meta.strategy_fingerprint == sched.strategy_fingerprint
    assert meta.schedule_hash() != sched.schedule_hash()


def test_model_cannot_invent_a_position_size():
    from alpha_agent.backtest.targets import TargetScheduleRow
    from alpha_agent.ml.errors import MLProtocolError

    sched = make_schedule([0, 1, 1, 0])
    rows = list(sched.rows)
    rows[1] = TargetScheduleRow(
        ts_event_ns=rows[1].ts_event_ns, root_symbol="NQ", target_units=3,
        strategy_fingerprint=rows[1].strategy_fingerprint,
    )
    forged = sched.model_copy(update={"rows": tuple(rows)})
    with pytest.raises(MLProtocolError, match="closed to TAKE/SKIP"):
        assert_closed_action_space(forged, sched)


# ==========================================================================
# 8. Registry identity
# ==========================================================================
def _experiment_spec(**overrides) -> MLExperimentSpec:
    base = {
        "primary_strategy_family": "tsmom",
        "primary_strategy_fingerprints": ("stratdsl1:aaa",),
        "root_symbol": "NQ",
        "training_roots": ("ES", "NQ", "CL", "GC", "ZN"),
        "meta_label_spec": MetaLabelSpec(),
        "feature_set": ML_FEATURE_SET,
        "regime": RegimeSpec(kind=RegimeSpecKind.NONE),
        "preprocessing": PreprocessingSpec(),
        "model_search": frozen_model_search(MLModelFamily.LOGISTIC_L2),
        "nested_cv": NESTED_CV,
        "take_threshold": 0.5,
        "dataset_fingerprint": "valdataset2:ds",
        "validation_spec_fingerprint": "validationspec1:vs",
        "reliability_policy_fingerprint": "valreliabilitypolicy1:rp",
        "execution_config_identity": "execconfig1:ex",
        "cost_config_identity": "costconfig1:co",
        "risk_identity": "riskconfig1:ri",
    }
    base.update(overrides)
    return MLExperimentSpec(**base)


def test_identity_is_computable_before_any_fit():
    spec = _experiment_spec()
    assert spec.experiment_identity().startswith("experiment1:")
    assert spec.strategy_family == "ml_meta_label.tsmom"


# --------------------------------------------------------------------------
# 8a. Phase 15A.2 -- the SEARCH PROCEDURE is identity, the SELECTION is not
# --------------------------------------------------------------------------
#: A single explicit plane assignment for the identity tests. Phase 15B fixes the
#: real ones; these exist so a full identity can be exercised, and they are
#: recorded nowhere as provenance.
_PLANES = MLIdentityPlanes(
    dataset_fingerprint="valdataset2:ds",
    validation_spec_fingerprint="validationspec1:vs",
    reliability_policy_fingerprint="valreliabilitypolicy1:rp",
    execution_config_identity="execconfig1:ex",
    cost_config_identity="costconfig1:co",
    risk_identity="riskconfig1:ri",
)
def test_identity_needs_no_fitted_model_and_names_no_hyperparameter_point():
    """(1) Identity is constructible strictly before fitting.

    ``MLExperimentSpec`` cannot even hold a selected hyperparameter: the model
    side is a search PROCEDURE. Nothing here touches data, a backend, or a fold.
    """
    spec = _experiment_spec()
    assert "model" not in type(spec).model_fields
    assert isinstance(spec.model_search, ModelSearchSpec)
    identity = spec.experiment_identity()
    assert identity.startswith("experiment1:")
    # deterministic and repeatable from configuration alone
    assert identity == _experiment_spec().experiment_identity()
    # the whole frozen grid is carried, not one point of it
    assert spec.model_search.n_configurations == 3
    assert [dict(h) for h in spec.model_search.search_grid] == [
        {"C": 0.1}, {"C": 1.0}, {"C": 10.0}
    ]


def _search_variants() -> list[tuple[str, ModelSearchSpec]]:
    """Perturbations of the frozen search PROCEDURE, one dimension at a time.

    ``model_copy`` is used deliberately: the constructor refuses a grid that is
    not the frozen one (that refusal is tested separately), and what must be
    proven here is that IF the frozen procedure changed, identity would change.
    """
    base = frozen_model_search(MLModelFamily.LOGISTIC_L2)
    return [
        ("model_family", frozen_model_search(MLModelFamily.HIST_GRADIENT_BOOSTING)),
        ("seed", frozen_model_search(MLModelFamily.LOGISTIC_L2, random_seed=7)),
        ("grid_extended", base.model_copy(
            update={"search_grid": (*base.search_grid, {"C": 100.0})})),
        ("grid_trimmed", base.model_copy(update={"search_grid": base.search_grid[:2]})),
        ("grid_reordered", base.model_copy(
            update={"search_grid": tuple(reversed(base.search_grid))})),
        ("selection_objective", base.model_copy(
            update={"selection_objective": "mean_inner_fold_brier_minimised"})),
        ("tie_break", base.model_copy(
            update={"selection_tie_break": "frozen_grid_declaration_order_last"})),
    ]


@pytest.mark.parametrize("what,variant", _search_variants(), ids=lambda v: v if isinstance(v, str) else "")
def test_changing_the_frozen_search_procedure_changes_identity(what, variant):
    """(2) Every dimension of the predeclared search is a pre-run identity input.

    ``model_copy`` bypasses the frozen-grid guard on purpose: the guard proves a
    changed procedure cannot be *slipped in*, and this proves that if it were
    predeclared differently it would be a different experiment.
    """
    base = _experiment_spec()
    changed = base.model_copy(update={"model_search": variant})
    assert base.experiment_identity() != changed.experiment_identity(), what
    assert base.parameter_variant_identity() != changed.parameter_variant_identity(), what
    assert base.composite_fingerprint() != changed.composite_fingerprint(), what


def test_an_experiment_spec_refuses_a_search_that_is_not_the_frozen_one():
    """The guard at the other end: a non-frozen procedure cannot be constructed."""
    base = frozen_model_search(MLModelFamily.LOGISTIC_L2)
    smuggled = base.model_copy(
        update={"search_grid": (*base.search_grid, {"C": 100.0})}
    )
    with pytest.raises(ValueError, match="not the FROZEN grid"):
        _experiment_spec(model_search=smuggled)


def test_a_post_fit_selected_hyperparameter_changes_no_identity():
    """(3) Which point a fold selected is provenance, and provenance is not identity.

    Two runs of ONE experiment whose outer folds select different points -- the
    normal nested-CV outcome -- must still be the same experiment.
    """
    spec = _experiment_spec()
    identity = spec.experiment_identity()
    for selected in spec.model_search.grid_specs():
        # a selected point is a real, auditable object ...
        assert assert_selection_is_within_search(spec.model_search, selected) is selected
        # ... and it appears nowhere in the identity inputs
        assert selected.identity() not in (
            identity, spec.composite_fingerprint(), spec.parameter_variant_identity(),
            spec.parameter_variant_label(),
        )
    # different folds selecting different points is exactly the nested-CV case
    assert len({m.identity() for m in spec.model_search.grid_specs()}) == 3
    assert spec.experiment_identity() == identity


def test_a_selection_outside_the_predeclared_search_is_refused():
    """Provenance is still audited: the executed search must be the declared one."""
    spec = _experiment_spec()
    outside = MLModelSpec(
        family=MLModelFamily.HIST_GRADIENT_BOOSTING,
        hyperparameters={"max_depth": 2, "learning_rate": 0.05, "max_iter": 100},
    )
    with pytest.raises(ValueError, match="not a point of the predeclared search"):
        assert_selection_is_within_search(spec.model_search, outside)


def test_an_expanded_or_trimmed_grid_cannot_be_constructed():
    base = frozen_model_search(MLModelFamily.LOGISTIC_L2)
    for grid in ((*base.search_grid, {"C": 100.0}), base.search_grid[:2],
                 tuple(reversed(base.search_grid))):
        with pytest.raises(ValueError, match="not the FROZEN grid"):
            ModelSearchSpec(family=MLModelFamily.LOGISTIC_L2, search_grid=grid)


def test_search_breadth_is_three_logistic_and_eight_tree():
    assert frozen_model_search(MLModelFamily.LOGISTIC_L2).n_configurations == 3
    assert frozen_model_search(
        MLModelFamily.HIST_GRADIENT_BOOSTING).n_configurations == 8
    assert frozen_model_search(MLModelFamily.LOGISTIC_L2).random_seed == DEFAULT_ML_RANDOM_SEED


def test_the_sixty_frozen_trials_map_one_to_one_onto_sixty_identities():
    """(4) 60 predeclared hypotheses -> exactly 60 registry-facing identities.

    Not 380 (one per hyperparameter point, the superseded behaviour) and not
    fewer than 60 (two hypotheses sharing a registry row and a BH trial).
    """
    manifest = build_candidate_manifest()
    assert manifest.n_trials() == 60

    identities = [
        experiment_spec_for_trial(t, planes=_PLANES).experiment_identity()
        for t in manifest.trials
    ]
    assert len(identities) == 60
    assert len(set(identities)) == 60

    # the same conclusion, proven without any plane assignment at all
    rows = assert_one_identity_per_trial(manifest)
    assert len({r["trial_discriminator"] for r in rows}) == 60
    assert len({r["friendly_experiment_id"] for r in rows}) == 60

    # and the count that keying on a point would have produced
    assert sum(t.n_hyperparameter_points for t in manifest.trials) == 380


def test_a_trial_identity_is_built_from_the_variants_own_reconstructed_primaries():
    """The Phase 14.2 rule: identity uses the proposal's OWN semantics."""
    from alpha_agent.strategy.candidates_phase_13_5c import baseline_spec
    from alpha_agent.strategy.fingerprint import strategy_fingerprint

    manifest = build_candidate_manifest()
    trial = next(t for t in manifest.trials if t.primary_family == "breakout")
    spec = experiment_spec_for_trial(trial, planes=_PLANES)
    expected = tuple(sorted(
        strategy_fingerprint(baseline_spec("breakout", r)) for r in trial.training_roots
    ))
    assert spec.primary_strategy_fingerprints == expected
    assert spec.feature_set.identity() == ML_FEATURE_SET.identity()


def test_duplicate_lookup_needs_no_selected_hyperparameters(tmp_path):
    """(5) "Already run?" is answerable from the proposal alone."""
    from alpha_agent.ml.identity import query_registry_before_running
    from alpha_agent.registry.sqlite_registry import ExperimentRegistry

    manifest = build_candidate_manifest()
    spec = experiment_spec_for_trial(manifest.trials[0], planes=_PLANES)
    with ExperimentRegistry(tmp_path / "reg.sqlite") as reg:
        answer = query_registry_before_running(reg, spec)
    assert answer["is_exact_duplicate"] is False
    assert answer["experiment_identity"] == spec.experiment_identity()
    # nothing in the query mentions a chosen point
    blob = json.dumps(
        {"identity": answer["experiment_identity"],
         "label": spec.parameter_variant_label()}
    )
    for point in spec.model_search.grid_specs():
        assert point.identity() not in blob


@pytest.mark.parametrize(
    "override",
    [
        {"model_search": frozen_model_search(MLModelFamily.HIST_GRADIENT_BOOSTING)},
        {"model_search": frozen_model_search(MLModelFamily.LOGISTIC_L2, random_seed=1)},
        {"nested_cv": NESTED_CV.model_copy(update={"embargo_days": 20})},
        {"regime": RegimeSpec(
            kind=RegimeSpecKind.DETERMINISTIC_CAUSAL_VOL_TREND,
            inputs=(FeatureSpec(kind="realized_vol", params={"window": 20}),),
        )},
        {"preprocessing": PreprocessingSpec(kind=PreprocessingKind.MEDIAN_IMPUTE)},
        {"meta_label_spec": MetaLabelSpec(positive_threshold_usd=25.0)},
        {"root_symbol": "ES"},
        {"primary_strategy_fingerprints": ("stratdsl1:bbb",)},
    ],
)
def test_different_semantics_give_a_different_experiment_identity(override):
    assert _experiment_spec().experiment_identity() != _experiment_spec(**override).experiment_identity()


def test_cosmetic_label_does_not_change_identity():
    assert (
        _experiment_spec(label="run A").experiment_identity()
        == _experiment_spec(label="run B").experiment_identity()
    )


def test_exact_duplicate_query_works_before_fitting(tmp_path):
    from alpha_agent.ml.identity import query_registry_before_running
    from alpha_agent.registry.sqlite_registry import ExperimentRegistry

    with ExperimentRegistry(tmp_path / "reg.sqlite") as reg:
        answer = query_registry_before_running(reg, _experiment_spec())
    assert answer["is_exact_duplicate"] is False
    assert answer["experiment_identity"].startswith("experiment1:")


def test_phase_13_registry_entries_are_untouched_by_phase_15():
    """Phase 15 adds no Phase-13 history rewrite; schema v5 leaves the identity
    formula and every Phase 13/14 authoritative result unchanged."""
    from alpha_agent.registry.identity import IDENTITY_SCHEMA
    from alpha_agent.registry.schema import SCHEMA_VERSION

    assert SCHEMA_VERSION == 7
    assert IDENTITY_SCHEMA == "experiment-identity/2"
    db = Path("data/registry/experiments.sqlite")
    if db.exists():
        from alpha_agent.registry.sqlite_registry import ExperimentRegistry

        with ExperimentRegistry(db) as reg:
            summary = reg.summary()
            auth = list(reg.experiments(authoritative_only=True))
        # Scoped to the Phase 13.5C subset itself (`experiment.phase ==
        # "13.5C"`), never to the registry's global total -- the production
        # registry is append-only and legitimately grows over time (later
        # real orchestrator runs, e.g. the Research Golden Path V1
        # acceptance pass's own new tsmom/NQ (30,150) experiment). This
        # test's actual claim is that Phase 15 never rewrites a Phase 13.5C
        # row, which a global-total assertion cannot distinguish from
        # "nothing else was ever added to the registry".
        phase_13_5c = [v for v in auth if v.experiment.phase == "13.5C"]
        assert len(phase_13_5c) == 107
        # the 60 Phase 15B.2 identities are likewise untouched by anything
        # appended after them.
        phase_15b = [v for v in auth if v.experiment.phase == "15B"]
        assert len(phase_15b) == 60
        assert summary.superseded_experiments == 21
        assert summary.identity_schema == "experiment-identity/2"


# ==========================================================================
# 9. The frozen manifest
# ==========================================================================
def test_manifest_is_deterministic_and_carries_no_performance():
    a = build_candidate_manifest()
    b = build_candidate_manifest()
    assert a.manifest_fingerprint() == b.manifest_fingerprint()
    assert_no_performance_fields(a)


def test_manifest_declares_a_separate_multiple_testing_family():
    m = build_candidate_manifest()
    mt = m.multiple_testing_family
    assert mt["family_id"] == "phase_15_ml.all"
    assert mt["phase_13_5c_trials_folded_in"] is False
    assert mt["phase_13_5c_n_trials_unchanged"] == 107
    assert mt["n_trials"] == m.n_trials()
    assert mt["placebo_in_denominator"] is False


def test_manifest_refuses_a_non_research_model_family():
    from alpha_agent.ml.models import assert_research_model_family

    m = build_candidate_manifest()
    for t in m.trials:
        assert_research_model_family(t.model_family)
    with pytest.raises(ValueError, match="not a research model family"):
        assert_research_model_family(MLModelFamily.SYNTHETIC_DETERMINISTIC)


def test_hyperparameters_outside_the_frozen_grid_are_refused():
    with pytest.raises(ValueError, match="FROZEN grid"):
        MLModelSpec(family=MLModelFamily.LOGISTIC_L2, hyperparameters={"C": 3.7})


def test_compute_estimate_separates_the_four_quantities():
    e = build_candidate_manifest().compute_estimate
    assert e.n_ml_hypotheses_bh_denominator == 60
    assert e.n_headline_trials == 40 and e.n_ablation_trials == 20
    assert e.n_model_fits_total == e.n_inner_fits + e.n_oof_producing_fits
    assert e.n_oof_producing_fits == 12 * NESTED_CV.n_outer_folds
    assert e.n_inspected_configurations_dsr > e.n_ml_hypotheses_bh_denominator
    assert e.n_cpp_evaluations_total_max == (
        e.n_cpp_evaluations_primary_baseline
        + e.n_cpp_evaluations_headline
        + e.n_cpp_evaluations_placebo_max
    )


# ==========================================================================
# 10. Regime discipline
# ==========================================================================
def test_learned_cluster_may_not_be_given_an_economic_name():
    with pytest.raises(LeakageError, match="economic interpretation"):
        assert_regime_label_is_not_economic("risk_on")
    assert_regime_label_is_not_economic("regime_cluster_2")


def test_learned_regime_is_declared_as_fitted():
    learned = RegimeSpec(
        kind=RegimeSpecKind.LEARNED_TRAIN_FOLD_KMEANS,
        inputs=(FeatureSpec(kind="realized_vol", params={"window": 20}),),
    )
    assert learned.is_fitted and learned.is_learned
    assert not RegimeSpec(kind=RegimeSpecKind.NONE).is_fitted


# ==========================================================================
# 11. The 2025 guard, and no network
# ==========================================================================
def test_holdout_guard_fires_on_a_2025_timestamp():
    with pytest.raises(HoldoutAccessError, match="LOCKED_FINAL_HOLDOUT"):
        assert_no_holdout_timestamps([ts(1), HOLDOUT_START_NS], what="test")


def test_holdout_guard_fires_inside_a_meta_label_event():
    sched = make_schedule([0, 1, 1, 0])
    eps = extract_primary_episodes(sched)
    econ = {0: make_economics(0, net=10.0, horizon_day=0)}
    bad = econ[0].model_copy(update={"information_horizon_ts_ns": HOLDOUT_START_NS})
    # The builder guards FIRST, so callers get the typed error, not a generic
    # validation complaint.
    with pytest.raises(HoldoutAccessError, match="LOCKED_FINAL_HOLDOUT"):
        build_meta_label_events(
            episodes=eps, economics={0: bad}, spec=MetaLabelSpec(), strategy_family="tsmom",
            dataset_fingerprint="ds1", feature_set_fingerprint="fs1", target_schedule_hash="t",
        )


def test_event_timeline_refuses_a_2025_label_horizon():
    """Defence in depth: the model itself refuses it too.

    Inside a pydantic validator the guard surfaces as a ValidationError (pydantic
    wraps AssertionError, and HoldoutAccessError deliberately is one so the Phase
    14 registry can keep catching it). The MESSAGE is still unambiguous, and the
    builder paths guard first so normal callers see the typed error.
    """
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="LOCKED_FINAL_HOLDOUT"):
        EventTimeline(
            decision_ts_ns=(ts(1),),
            label_start_ts_ns=(ts(1),),
            label_end_ts_ns=(HOLDOUT_START_NS + DAY_NS,),
        )


def test_event_timeline_holdout_guard_is_also_callable_directly():
    with pytest.raises(HoldoutAccessError, match="LOCKED_FINAL_HOLDOUT"):
        assert_no_holdout_timestamps(
            (ts(1), HOLDOUT_START_NS + DAY_NS), what="EventTimeline.label_end_ts_ns"
        )


def test_frozen_nested_cv_never_reaches_2025():
    assert NESTED_CV.corpus_end_ts_ns == HOLDOUT_START_NS
    assert max(b for _a, b in NESTED_CV.outer_test_blocks) == HOLDOUT_START_NS
    for a, b in NESTED_CV.outer_test_blocks:
        assert a < HOLDOUT_START_NS and b <= HOLDOUT_START_NS


def test_exclusive_boundary_is_allowed_only_as_an_exact_bound():
    """2025-01-01 as an END bound means "data through 2024-12-31", not access.

    CLAUDE.md's own naming rule says so; this makes the distinction enforceable
    instead of a convention, and refuses anything that merely approaches it.
    """
    assert_exclusive_corpus_bound(HOLDOUT_START_NS, what="corpus_end")
    with pytest.raises(HoldoutAccessError, match="not the exclusive"):
        assert_exclusive_corpus_bound(HOLDOUT_START_NS + 1, what="corpus_end")
    with pytest.raises(HoldoutAccessError, match="not the exclusive"):
        assert_exclusive_corpus_bound(ts(1), what="corpus_end")


def test_half_open_window_may_end_at_the_boundary_but_never_start_inside():
    assert_half_open_window(ts(1), HOLDOUT_START_NS, what="block")
    with pytest.raises(HoldoutAccessError, match="reaches past"):
        assert_half_open_window(ts(1), HOLDOUT_START_NS + DAY_NS, what="block")
    with pytest.raises(HoldoutAccessError, match="LOCKED_FINAL_HOLDOUT"):
        assert_half_open_window(HOLDOUT_START_NS, HOLDOUT_START_NS, what="block")


def test_the_boundary_allowance_is_granted_by_key_only():
    """A stray 2025 timestamp outside the declared bound keys still fails."""
    payload = {"corpus_end_ts_ns": HOLDOUT_START_NS, "some_other_ts": HOLDOUT_START_NS}
    with pytest.raises(HoldoutAccessError, match="some_other_ts"):
        assert_no_holdout_in_config_payload(
            payload, exclusive_bound_keys=frozenset({"corpus_end_ts_ns"})
        )


def test_frozen_manifest_passes_the_holdout_guard():
    m = build_candidate_manifest()
    assert_no_holdout_in_config_payload(
        json.loads(m.canonical_json()),
        exclusive_bound_keys=frozenset({"corpus_end_ts_ns"}),
        half_open_window_keys=frozenset({"outer_test_blocks"}),
    )


def test_network_access_is_refused():
    from alpha_agent.ml.errors import NetworkAccessForbidden

    with pytest.raises(NetworkAccessForbidden, match="no network access"):
        forbid_network("https://example.invalid")


def test_approved_dependencies_are_present_and_forbidden_ones_are_not():
    """scikit-learn + scipy were approved; nothing else was (prompt 15A.1 s.1-2)."""
    from alpha_agent.ml.models import (
        FORBIDDEN_ML_PACKAGES,
        assert_no_forbidden_ml_packages,
        ml_environment_provenance,
    )

    status = ml_dependency_status()
    assert status["sklearn"] is True and status["scipy"] is True
    assert status["forbidden_packages_present"] == []
    for pkg in FORBIDDEN_ML_PACKAGES:
        assert status[pkg] is False, f"{pkg} must not be installed"
    assert "lightgbm" in FORBIDDEN_ML_PACKAGES
    assert_no_forbidden_ml_packages()

    env = ml_environment_provenance()
    for pkg in ("python", "numpy", "pandas", "scipy", "sklearn"):
        assert env[pkg], f"{pkg} version must be recorded in Phase 15 provenance"


def test_the_research_model_family_stays_at_exactly_two():
    """Deliberate statistical control: one linear model, one controlled tree."""
    from alpha_agent.ml.models import RESEARCH_MODEL_FAMILIES

    assert RESEARCH_MODEL_FAMILIES == (
        MLModelFamily.LOGISTIC_L2,
        MLModelFamily.HIST_GRADIENT_BOOSTING,
    )
    assert {t.model_family for t in build_candidate_manifest().trials} == set(
        RESEARCH_MODEL_FAMILIES
    )


def test_a_missing_backend_package_still_fails_loudly():
    """The lazy-import refusal is intact for any family whose package is absent."""
    from alpha_agent.ml import models

    spec = MLModelSpec(family=MLModelFamily.LOGISTIC_L2, hyperparameters={"C": 1.0})
    backend = make_backend(spec, feature_order=("a",))
    with pytest.MonkeyPatch.context() as mp:
        def _boom(package: str, family) -> None:
            raise MLDependencyMissing(
                f"{family.value} needs {package!r} ... explicit human approval"
            )

        mp.setattr(models, "_require", _boom)
        with pytest.raises(MLDependencyMissing, match="explicit human approval"):
            backend.fit(np.zeros((4, 1)), np.array([0, 1, 0, 1]))


def test_sklearn_research_backends_run_on_SYNTHETIC_data_only():
    """A synthetic smoke test of both frozen families -- no market data anywhere.

    Phase 15A.1 may run synthetic ML tests. It may NOT fit the real 2018-2024
    event corpus or inspect any real-market ML metric; that is Phase 15B.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] - X[:, 1] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    order = ("a", "b", "c", "d")
    for family, params in (
        (MLModelFamily.LOGISTIC_L2, {"C": 1.0}),
        (MLModelFamily.HIST_GRADIENT_BOOSTING,
         {"max_depth": 2, "learning_rate": 0.05, "max_iter": 100}),
    ):
        backend = make_backend(
            MLModelSpec(family=family, hyperparameters=params), feature_order=order
        )
        backend.fit(X, y)
        p = backend.predict_proba_positive(X)
        assert p.shape == (200,)
        assert np.all((p >= 0.0) & (p <= 1.0))
        meta = backend.artifact_metadata()
        assert meta["backend"] == "sklearn" and tuple(meta["feature_order"]) == order
        assert meta["sklearn_version"]


# ==========================================================================
# 12. The additive C++ closed-trade export
# ==========================================================================
@pytest.mark.skipif(not CLI.exists(), reason="C++ CLI not built")
def test_trades_export_is_additive_and_changes_no_json(tmp_path):
    """The same run, with and without --trades-out, must emit the SAME JSON.

    This is the whole justification for the additive export: it may add an
    artifact, but it may not perturb one number of the official result.
    """
    import baseline_fixtures as bf
    from alpha_agent.adapters.targets_bridge import (
        run_targets_backtest_cli,
    )
    from alpha_agent.backtest.targets import (
        TargetSchedule,
        TargetScheduleRow,
    )

    closes = bf.trending_closes(40)
    bars = bf.boundary_bars(closes, instrument_id=915001)
    contracts = bf.one_contract(
        instrument_id=915001, raw_symbol="NQU6", root="NQ",
        expiration_ns=bf.BASE_NS + 10**15,
    )
    fp = "stratdsl1:" + "e" * 64
    tss = [int(t) for t in bars["ts_event_ns"]]
    sched = TargetSchedule(
        root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
        strategy_dsl_version="strategy-dsl/1", feature_engine_version="0.9.2", warmup_bars=0,
        rows=tuple(
            TargetScheduleRow(
                ts_event_ns=t, root_symbol="NQ", target_units=u, strategy_fingerprint=fp
            )
            for t, u in ((tss[5], 1), (tss[15], 0), (tss[20], -1), (tss[30], 0))
        ),
    )
    plain = run_targets_backtest_cli(
        bars, contracts, sched, executable=CLI, work_dir=tmp_path / "a"
    )
    out_csv = tmp_path / "trades.csv"
    fills_csv = tmp_path / "fills.csv"
    with_export = run_targets_backtest_cli(
        bars, contracts, sched, executable=CLI, work_dir=tmp_path / "b",
        trades_out_path=out_csv, fills_out_path=fills_csv,
    )
    assert plain == with_export, "the additive export must not perturb the JSON result"

    trades = read_closed_trades_csv(out_csv)
    fills = read_fills_csv(fills_csv)
    assert len(trades) == plain["trades"] > 0
    assert len(fills) == plain["fills"] > 0
    # The exports are the OFFICIAL audit trails and must reconcile with the
    # engine's own headline numbers, because Python only ever sums what C++ priced.
    assert sum(t.gross_pnl_usd for t in trades) == pytest.approx(
        plain["gross_pnl_usd"], abs=1e-6
    )
    assert sum(f.commission_usd for f in fills) == pytest.approx(plain["costs_usd"], abs=1e-6)
    assert (
        sum(t.gross_pnl_usd for t in trades) - sum(f.commission_usd for f in fills)
    ) == pytest.approx(plain["net_pnl_usd"], abs=1e-6)


@pytest.mark.skipif(not CLI.exists(), reason="C++ CLI not built")
def test_closed_trade_costs_alone_would_understate_the_episode_cost(tmp_path):
    """The reason the fill export exists, asserted rather than asserted-in-prose.

    ``ClosedTrade.costs_usd`` is the CLOSING fill's commission only, so summing it
    understates the true cost -- which would bias every meta-label optimistic.
    """
    import baseline_fixtures as bf
    from alpha_agent.adapters.targets_bridge import (
        run_targets_backtest_cli,
    )
    from alpha_agent.backtest.targets import (
        TargetSchedule,
        TargetScheduleRow,
    )

    bars = bf.boundary_bars(bf.trending_closes(40), instrument_id=915003)
    contracts = bf.one_contract(
        instrument_id=915003, raw_symbol="NQU6", root="NQ",
        expiration_ns=bf.BASE_NS + 10**15,
    )
    fp = "stratdsl1:" + "b" * 64
    tss = [int(t) for t in bars["ts_event_ns"]]
    sched = TargetSchedule(
        root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
        strategy_dsl_version="strategy-dsl/1", feature_engine_version="0.9.2", warmup_bars=0,
        rows=tuple(
            TargetScheduleRow(
                ts_event_ns=t, root_symbol="NQ", target_units=u, strategy_fingerprint=fp
            )
            for t, u in ((tss[5], 1), (tss[15], 0))
        ),
    )
    trades_csv, fills_csv = tmp_path / "t.csv", tmp_path / "f.csv"
    res = run_targets_backtest_cli(
        bars, contracts, sched, executable=CLI, work_dir=tmp_path,
        trades_out_path=trades_csv, fills_out_path=fills_csv,
    )
    trades = read_closed_trades_csv(trades_csv)
    fills = read_fills_csv(fills_csv)
    closing_only = sum(t.costs_usd for t in trades)
    full_round_turn = sum(f.commission_usd for f in fills)
    assert closing_only < full_round_turn == pytest.approx(res["costs_usd"], abs=1e-6)

    episodes = extract_primary_episodes(sched)
    econ, _ = attribute_trades_to_episodes(episodes, trades, fills)
    assert econ[0].costs_are_full_round_turn is True
    assert econ[0].costs_usd == pytest.approx(full_round_turn, abs=1e-6)
    assert econ[0].net_pnl_usd == pytest.approx(res["net_pnl_usd"], abs=1e-6)


@pytest.mark.skipif(not CLI.exists(), reason="C++ CLI not built")
def test_episode_label_pipeline_over_a_real_cpp_run(tmp_path):
    """End to end on synthetic bars: schedule -> episodes -> C++ trades -> labels."""
    import baseline_fixtures as bf
    from alpha_agent.adapters.targets_bridge import (
        run_targets_backtest_cli,
    )
    from alpha_agent.backtest.targets import (
        TargetSchedule,
        TargetScheduleRow,
    )

    closes = bf.trending_closes(40)
    bars = bf.boundary_bars(closes, instrument_id=915002)
    contracts = bf.one_contract(
        instrument_id=915002, raw_symbol="NQU6", root="NQ",
        expiration_ns=bf.BASE_NS + 10**15,
    )
    fp = "stratdsl1:" + "f" * 64
    tss = [int(t) for t in bars["ts_event_ns"]]
    sched = TargetSchedule(
        root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
        strategy_dsl_version="strategy-dsl/1", feature_engine_version="0.9.2", warmup_bars=0,
        rows=tuple(
            TargetScheduleRow(
                ts_event_ns=t, root_symbol="NQ", target_units=u, strategy_fingerprint=fp
            )
            for t, u in ((tss[5], 1), (tss[15], 0), (tss[20], -1), (tss[30], 0))
        ),
    )
    out_csv = tmp_path / "trades.csv"
    fills_csv = tmp_path / "fills.csv"
    run_targets_backtest_cli(
        bars, contracts, sched, executable=CLI, work_dir=tmp_path,
        trades_out_path=out_csv, fills_out_path=fills_csv,
    )
    episodes = extract_primary_episodes(sched)
    assert [e.side for e in episodes] == [1, -1]
    econ, unattributed = attribute_trades_to_episodes(
        episodes, read_closed_trades_csv(out_csv), read_fills_csv(fills_csv)
    )
    assert unattributed == (), "every C++ trade must belong to an episode"
    assert all(e.n_attributed_trades > 0 for e in econ.values())
    # Each episode's information horizon is after its own entry decision.
    for ep in episodes:
        assert econ[ep.episode_index].information_horizon_ts_ns > ep.entry_decision_ts_ns


def test_closed_trade_reader_rejects_column_drift(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("trade_index,instrument_id\n0,1\n")
    with pytest.raises(ValueError, match="frozen contract"):
        read_closed_trades_csv(p)


def test_export_columns_mirror_the_cpp_header():
    header = Path("cpp/include/quant_core/trade_export.hpp").read_text()
    for col in CLOSED_TRADE_COLUMNS + FILL_COLUMNS:
        assert col in header, f"{col} missing from the C++ header contract"


# ==========================================================================
# 13. Phase 15A.1 -- threshold-selection semantics
# ==========================================================================
def test_take_threshold_is_frozen_at_one_half():
    from alpha_agent.ml.models import FROZEN_TAKE_THRESHOLD

    assert FROZEN_TAKE_THRESHOLD == 0.50


def test_no_threshold_grid_survives_anywhere_in_research_model_selection():
    """The sweep is gone from the code, the spec and the frozen manifest.

    It was never a real search: the inner objective is mean inner-fold LOG LOSS,
    a function of the predicted probability alone, so every threshold scored
    identically and the "selection" fell through a deterministic tie-break --
    while still multiplying the DSR search-breadth term by four.
    """
    from alpha_agent.ml import models, training

    assert not hasattr(models, "FROZEN_THRESHOLD_GRID")
    assert not hasattr(MLTrainingSpec(
        nested_cv=NESTED_CV, preprocessing=PreprocessingSpec(),
        regime=RegimeSpec(kind=RegimeSpecKind.NONE),
    ), "threshold_grid")
    assert training.INNER_SELECTION_DIMENSIONS == ("model_hyperparameter_configuration",)
    assert training.INNER_OBJECTIVE == "mean_inner_fold_log_loss_minimised"

    m = build_candidate_manifest()
    assert m.take_threshold == 0.50
    assert m.threshold_is_searched is False
    assert m.inner_selection_dimensions == ("model_hyperparameter_configuration",)
    assert not hasattr(m, "threshold_grid")
    assert all(t.n_threshold_configurations == 1 for t in m.trials)


def test_a_non_frozen_threshold_is_refused_everywhere():
    from alpha_agent.ml.models import assert_frozen_take_threshold

    for bad in (0.45, 0.55, 0.60):
        with pytest.raises(ValueError, match="FROZEN Phase 15 TAKE threshold"):
            assert_frozen_take_threshold(bad)
        with pytest.raises(ValueError, match="FROZEN Phase 15 TAKE threshold"):
            MLTrainingSpec(
                nested_cv=NESTED_CV, preprocessing=PreprocessingSpec(),
                regime=RegimeSpec(kind=RegimeSpecKind.NONE), take_threshold=bad,
            )
        with pytest.raises(ValueError, match="FROZEN Phase 15 TAKE threshold"):
            _experiment_spec(take_threshold=bad)


def test_every_oof_decision_uses_the_frozen_threshold():
    (frame, report), _ = _run()
    assert report.take_threshold == 0.50
    assert {f.selected_threshold for f in report.folds} == {0.50}
    for p in frame.predictions:
        assert p.threshold == 0.50
        expected = MetaLabelAction.TAKE if p.probability_take >= 0.50 else MetaLabelAction.SKIP
        assert p.action is expected


def test_inner_loop_inspects_only_hyperparameter_configurations():
    """Distinct configurations, not configurations x folds x phantom thresholds."""
    from alpha_agent.ml.models import frozen_model_specs

    (_frame, report), _ = _run()
    n_grid = len(frozen_model_specs(MLModelFamily.SYNTHETIC_DETERMINISTIC))
    assert report.n_distinct_configurations_inspected == n_grid
    assert report.n_configuration_fold_inspections == n_grid * NESTED_CV.n_outer_folds


# ==========================================================================
# 14. Phase 15A.1 -- the rebuilt pre-run search accounting
# ==========================================================================
def _accounting() -> dict:
    from alpha_agent.ml.manifest import trial_accounting_payload

    return trial_accounting_payload(build_candidate_manifest())


def test_pretrain_accounting_keeps_every_quantity_distinct():
    e = build_candidate_manifest().compute_estimate
    # 1. BH/FDR economic hypotheses
    assert e.n_ml_hypotheses_bh_denominator == 60 == e.n_headline_trials + e.n_ablation_trials
    # 2. fitted pipelines
    assert e.n_distinct_fitted_pipelines == 12
    # 3. model hyperparameter configurations
    assert e.n_hyperparameter_points_by_model_family == {
        "LOGISTIC_L2": 3, "HIST_GRADIENT_BOOSTING": 8
    }
    assert e.n_distinct_model_configurations_fitted == 76
    assert e.n_threshold_configurations == 1 and e.take_threshold == 0.50
    # 4/5. fits
    assert e.n_inner_fits == 1140 and e.n_oof_producing_fits == 60
    assert e.n_model_fits_total == 1200 == e.n_inner_fits + e.n_oof_producing_fits
    # 6. DSR search breadth
    assert e.n_inspected_configurations_dsr == 380
    assert e.dsr_effective_trial_count_by_model_family == {
        "LOGISTIC_L2": 3, "HIST_GRADIENT_BOOSTING": 8
    }
    # 7. C++ economic evaluations
    assert e.n_cpp_evaluations_primary_baseline == 60
    assert e.n_cpp_evaluations_headline == 180
    assert e.n_cpp_evaluations_total_excluding_placebo == 240
    # 8. conditional placebo runs
    assert e.n_cpp_evaluations_placebo_max == 1200
    assert e.n_cpp_evaluations_total_max == 1440


def test_the_superseded_dsr_count_is_corrected_by_exactly_twenty():
    """7,600 -> 380: 4 phantom thresholds x 5 per-fold refits of the same config."""
    e = build_candidate_manifest().compute_estimate
    assert e.superseded_n_inspected_configurations_dsr == 7600
    assert e.superseded_n_inspected_configurations_dsr == 20 * e.n_inspected_configurations_dsr


def test_a_fold_refit_is_a_training_operation_not_a_new_hypothesis():
    e = build_candidate_manifest().compute_estimate
    n_outer = NESTED_CV.n_outer_folds
    assert e.n_configuration_fold_inspections == e.n_inspected_configurations_dsr * n_outer
    assert e.n_inspected_configurations_dsr < e.n_configuration_fold_inspections
    # ... and the fold factor lives in the FIT totals, where it belongs.
    assert e.n_inner_fits == n_outer * NESTED_CV.n_inner_folds * (
        e.n_distinct_model_configurations_fitted
    )


def test_dsr_terms_sum_to_the_declared_search_breadth():
    a = _accounting()
    terms = a["6_dsr_inspected_configurations"]["terms"]
    assert sum(t["contribution"] for t in terms) == 380
    assert {t["n_trials"] for t in terms} == {20}
    assert sorted(a["invariants_checked"])


def test_the_committed_accounting_artifact_matches_the_manifest():
    from alpha_agent.ml.manifest import SUPERSEDED_15A2_MANIFEST_FINGERPRINT

    # the 15A.2-era accounting artifact is preserved unchanged as superseded history
    prior = json.loads(
        Path("outputs/phase_15/PRETRAIN_TRIAL_ACCOUNTING_IDENTITY_CORRECTED.json").read_text()
    )
    assert prior["manifest_fingerprint"] == SUPERSEDED_15A2_MANIFEST_FINGERPRINT

    # the 15B.1b regime-corrected accounting artifact agrees with the current /4 manifest
    path = Path("outputs/phase_15/PRETRAIN_TRIAL_ACCOUNTING_REGIME_CORRECTED.json")
    assert path.exists(), "run scripts/phase_15b1b_regime_freeze.py"
    on_disk = json.loads(path.read_text())
    fresh = _accounting()
    for key in (k for k in fresh if k not in ("generated_at", "code_commit")):
        assert on_disk[key] == fresh[key], f"{key} drifted from the manifest"
    assert on_disk["manifest_fingerprint"] == build_candidate_manifest().manifest_fingerprint()
    assert on_disk["phase"] == "15B.1b"


def test_the_phase_15a1_accounting_artifact_is_preserved_unchanged():
    """Superseded evidence stays exactly as it was written."""
    prior = json.loads(Path("outputs/phase_15/PRETRAIN_TRIAL_ACCOUNTING.json").read_text())
    assert prior["phase"] == "15A.1"
    assert prior["manifest_fingerprint"] == (
        "p15manifest1:82983cf4f04e101520bb882ddca9aed9dd779565a646e3f4a70c68b352897975"
    )
    assert prior["6_dsr_inspected_configurations"]["value"] == 380


def test_the_committed_identity_map_proves_sixty_distinct_pre_run_identities():
    from alpha_agent.ml.manifest import SUPERSEDED_15A2_MANIFEST_FINGERPRINT

    # the 15A.2-era map is preserved unchanged as superseded history
    prior = json.loads(Path("outputs/phase_15/PRETRAIN_IDENTITY_MAP.json").read_text())
    assert prior["manifest_fingerprint"] == SUPERSEDED_15A2_MANIFEST_FINGERPRINT
    assert prior["n_distinct_pre_run_identities"] == 60

    # the 15B.1b regime-corrected complete map is authoritative and agrees with
    # the current /4 manifest
    path = Path("outputs/phase_15/PRERUN_IDENTITY_MAP_COMPLETE_REGIME_CORRECTED.json")
    assert path.exists(), "run scripts/phase_15b1b_regime_freeze.py"
    doc = json.loads(path.read_text())
    manifest = build_candidate_manifest()
    assert doc["manifest_fingerprint"] == manifest.manifest_fingerprint()
    assert doc["n_predeclared_hypotheses"] == 60
    assert doc["n_complete_pre_run_identities"] == 60
    assert doc["identities_per_hypothesis"] == 1
    assert doc["bh_fdr_denominator"] == 60

    rows = doc["trials"]
    assert len(rows) == 60
    assert len({r["experiment_identity"] for r in rows}) == 60
    assert len({r["trial_discriminator"] for r in rows}) == 60
    assert len({r["trial_label"] for r in rows}) == 60
    assert {r["take_threshold"] for r in rows} == {0.50}
    assert {r["n_search_configurations"] for r in rows} == {3, 8}

    # one predeclared regime transformation, not nine hypotheses
    assert len(doc["regime_baseline_trial_labels_regenerated"]) == 40
    assert len(doc["no_regime_ablation_trial_labels_unchanged"]) == 20

    # no selected hyperparameter appears anywhere in the frozen map
    blob = json.dumps(doc)
    for family in (MLModelFamily.LOGISTIC_L2, MLModelFamily.HIST_GRADIENT_BOOSTING):
        for point in frozen_model_search(family).grid_specs():
            assert point.identity() not in blob


def test_the_phase_15a_manifest_is_preserved_as_superseded():
    """Supersession is a NEW record plus an edge, never an in-place edit."""
    from alpha_agent.ml.manifest import (
        SUPERSEDED_15A1_MANIFEST_FINGERPRINT,
        SUPERSEDED_15A2_MANIFEST_FINGERPRINT,
        SUPERSEDED_MANIFEST_FINGERPRINT,
    )

    old = json.loads(Path("data/manifests/phase_15/ml_candidate_manifest.json").read_text())
    assert old["manifest_fingerprint"] == SUPERSEDED_MANIFEST_FINGERPRINT
    assert old["manifest"]["schema_version"] == "phase-15-ml-candidate-manifest/1"
    assert old["manifest"]["threshold_grid"] == [0.45, 0.5, 0.55, 0.6]

    corrected = json.loads(
        Path("data/manifests/phase_15/ml_candidate_manifest_corrected.json").read_text()
    )
    assert corrected["manifest_fingerprint"] == SUPERSEDED_15A1_MANIFEST_FINGERPRINT
    assert corrected["manifest"]["schema_version"] == "phase-15-ml-candidate-manifest/2"
    assert corrected["manifest"]["take_threshold"] == 0.50

    identity_corrected = json.loads(
        Path(
            "data/manifests/phase_15/ml_candidate_manifest_identity_corrected.json"
        ).read_text()
    )
    assert identity_corrected["manifest_fingerprint"] == SUPERSEDED_15A2_MANIFEST_FINGERPRINT
    assert identity_corrected["manifest"]["schema_version"] == (
        "phase-15-ml-candidate-manifest/3"
    )

    m = build_candidate_manifest()
    assert m.schema_version == "phase-15-ml-candidate-manifest/4"
    assert m.manifest_fingerprint() not in (
        SUPERSEDED_MANIFEST_FINGERPRINT,
        SUPERSEDED_15A1_MANIFEST_FINGERPRINT,
        SUPERSEDED_15A2_MANIFEST_FINGERPRINT,
    )
    # the immediate edge points at 15A.2; the reason is the regime transformation
    assert m.supersedes["manifest_fingerprint"] == SUPERSEDED_15A2_MANIFEST_FINGERPRINT
    assert "REGIME TRANSFORMATION SEMANTICS" in m.supersedes["reason"]
    # the chain is preserved end to end, oldest edge first
    chain = [c["manifest_fingerprint"] for c in m.supersession_chain]
    assert chain == [
        SUPERSEDED_MANIFEST_FINGERPRINT,
        SUPERSEDED_15A1_MANIFEST_FINGERPRINT,
        SUPERSEDED_15A2_MANIFEST_FINGERPRINT,
    ]
    assert "THRESHOLD-SELECTION SEMANTICS" in m.supersession_chain[0]["reason"]
    assert "PRE-RUN EXPERIMENT-IDENTITY SEMANTICS" in m.supersession_chain[1]["reason"]


def test_multiple_testing_family_reports_the_corrected_dsr_count():
    mt = build_candidate_manifest().multiple_testing_family
    assert mt["dsr_effective_trial_count"] == 380
    assert mt["dsr_effective_trial_count_superseded"] == 7600
    assert mt["take_threshold"] == 0.50 and mt["threshold_is_searched"] is False
    assert mt["n_trials"] == 60 and mt["phase_13_5c_n_trials_unchanged"] == 107


# ==========================================================================
# 15. Phase 15A.1 -- the event-count gates stay frozen
# ==========================================================================
def test_event_count_gates_are_unchanged():
    assert NESTED_CV.min_train_events == 100
    assert NESTED_CV.min_test_events == 20


def test_a_scope_below_the_train_gate_is_refused_with_a_typed_reason():
    from alpha_agent.ml.enums import MLRefusalReason
    from alpha_agent.ml.errors import InsufficientEventsError

    # 150 events spread thinly over the whole corpus: the 2020 outer fold has
    # enough test events but far fewer than 100 training events before it.
    X, y, timeline, ids, roots = synthetic_dataset(n_events=200, spacing_days=12)
    with pytest.raises(InsufficientEventsError) as exc:
        _run_on(X, y, timeline, ids, roots)
    assert exc.value.reason is MLRefusalReason.INSUFFICIENT_TRAIN_EVENTS
    assert exc.value.detail["min_required"] == 100
    assert exc.value.detail["n_events"] < 100
    assert "REFUSED" in str(exc.value)


def test_a_scope_below_the_test_gate_is_refused_with_a_typed_reason():
    from alpha_agent.ml.enums import MLRefusalReason
    from alpha_agent.ml.errors import InsufficientEventsError

    # A dense stream that stops before the first outer test block: plenty of
    # training events, no test events at all.
    X, y, timeline, ids, roots = synthetic_dataset(n_events=120)
    with pytest.raises(InsufficientEventsError) as exc:
        _run_on(X, y, timeline, ids, roots)
    assert exc.value.reason is MLRefusalReason.INSUFFICIENT_TEST_EVENTS
    assert exc.value.detail["min_required"] == 20


def test_the_gates_are_never_lowered_after_seeing_real_counts():
    """A frozen gate is a refusal, not a knob. Both are asserted from the manifest."""
    m = build_candidate_manifest()
    assert (m.nested_cv.min_train_events, m.nested_cv.min_test_events) == (100, 20)
    on_disk = json.loads(
        Path("outputs/phase_15/PROTOCOL_FREEZE_CORRECTED.json").read_text()
    )["event_count_gates"]
    assert on_disk["min_train_events"] == 100 and on_disk["min_test_events"] == 20
    assert on_disk["frozen"] is True
    assert "INSUFFICIENT_TRAIN_EVENTS" in on_disk["on_shortfall"]


# ==========================================================================
# 16. Phase 15A.1 -- episode economics attribution, proved case by case
# ==========================================================================
# Commission is a flat USD-per-contract charge, so a fill's commission splits by
# CONTRACT: the contracts that CLOSED exposure belong to the episode owning the
# resulting ClosedTrade, and every remaining contract OPENED exposure and belongs
# to the episode open at the fill instant. These six cases are the whole
# attribution contract (prompt 15A.1 s.6).
def _attribute(sched, trades, fills):
    from alpha_agent.ml.episodes import attribute_episode_economics

    episodes = extract_primary_episodes(sched)
    return episodes, attribute_episode_economics(episodes, trades, fills)


def test_case_A_enter_then_exit():
    from ml_fixtures import make_fill

    sched = make_schedule([0, 1, 1, 0])
    fills = (
        make_fill(day=1, side="buy", quantity=1, index=0),
        make_fill(day=3, side="sell", quantity=1, index=1),
    )
    trades = (make_trade(open_day=1, close_day=3, net=98.0, index=0, gross=100.0, costs=2.0),)
    episodes, att = _attribute(sched, trades, fills)

    assert len(episodes) == 1 and episodes[0].side == 1
    e = att.economics[0]
    assert e.gross_pnl_usd == pytest.approx(100.0)
    assert e.costs_usd == pytest.approx(4.0), "the FULL round turn, not the closing fill alone"
    assert e.net_pnl_usd == pytest.approx(96.0)
    assert att.unattributed_trades == () and att.unattributed_fill_costs_usd == 0.0


def test_case_B_same_sign_size_change_stays_one_episode():
    from ml_fixtures import make_fill

    sched = make_schedule([0, 1, 2, 0])
    fills = (
        make_fill(day=1, side="buy", quantity=1, index=0),
        make_fill(day=2, side="buy", quantity=1, index=1),
        make_fill(day=3, side="sell", quantity=2, index=2),
    )
    trades = (
        make_trade(open_day=1, close_day=3, net=0.0, index=0, quantity=2,
                   gross=200.0, costs=4.0),
    )
    episodes, att = _attribute(sched, trades, fills)

    assert len(episodes) == 1, "a same-sign size change never opens a new episode"
    e = att.economics[0]
    assert e.gross_pnl_usd == pytest.approx(200.0)
    assert e.costs_usd == pytest.approx(8.0)     # 2 + 2 + 4
    assert att.unattributed_fill_costs_usd == 0.0


def test_case_C_long_to_short_flip_splits_the_boundary_fill_by_contract():
    """The reversal fill closes one contract and opens another. So does its cost.

    Charging the whole two-contract commission to the outgoing episode would
    overstate its cost and understate the incoming episode's -- biasing two
    labels in opposite directions at every single reversal.
    """
    from ml_fixtures import make_fill

    sched = make_schedule([0, 1, -1, 0])
    fills = (
        make_fill(day=1, side="buy", quantity=1, index=0),
        make_fill(day=2, side="sell", quantity=2, index=1),      # flips through zero
        make_fill(day=3, side="buy", quantity=1, index=2),
    )
    trades = (
        make_trade(open_day=1, close_day=2, net=0.0, index=0, direction=1,
                   gross=50.0, costs=4.0),                        # whole fill commission
        make_trade(open_day=2, close_day=3, net=0.0, index=1, direction=-1,
                   gross=30.0, costs=2.0),
    )
    episodes, att = _attribute(sched, trades, fills)

    assert [e.side for e in episodes] == [1, -1]
    assert att.economics[0].costs_usd == pytest.approx(4.0)   # entry 2 + 1 closing contract
    assert att.economics[1].costs_usd == pytest.approx(4.0)   # 1 opening contract + exit 2
    assert att.costs_usd == pytest.approx(sum(f.commission_usd for f in fills)) == 8.0
    assert att.unattributed_fill_costs_usd == 0.0
    # The naive alternative loses the entry commission entirely.
    naive = sum(t.costs_usd for t in trades)
    assert naive == pytest.approx(6.0)
    assert naive < att.costs_usd


def test_case_D_latency_delayed_exit_still_belongs_to_its_episode():
    """The exit FILL lands after the exit DECISION -- and after the episode closed."""
    from ml_fixtures import make_fill

    sched = make_schedule([0, 1, 1, 0])
    fills = (
        make_fill(day=1, side="buy", quantity=1, index=0),
        make_fill(day=4, side="sell", quantity=1, index=1),      # decision day 3, fill day 4
    )
    trades = (make_trade(open_day=1, close_day=4, net=0.0, index=0, gross=70.0, costs=2.0),)
    episodes, att = _attribute(sched, trades, fills)

    assert episodes[0].exit_decision_ts_ns == ts(3)
    e = att.economics[0]
    assert e.costs_usd == pytest.approx(4.0)
    assert e.information_horizon_ts_ns == ts(4), (
        "the label horizon is the last FILL close, not the exit decision"
    )
    assert att.unattributed_fill_costs_usd == 0.0, "a delayed fill is never dropped"


def test_case_E_episode_spanning_a_roll_stays_one_episode():
    from ml_fixtures import make_fill

    sched = make_schedule([0, 1, 1, 1, 0])
    fills = (
        make_fill(day=1, side="buy", quantity=1, index=0, instrument_id=1, symbol="NQH4"),
        make_fill(day=2, side="sell", quantity=1, index=1, instrument_id=1, symbol="NQH4"),
        make_fill(day=2, side="buy", quantity=1, index=2, instrument_id=2, symbol="NQM4"),
        make_fill(day=4, side="sell", quantity=1, index=3, instrument_id=2, symbol="NQM4"),
    )
    trades = (
        make_trade(open_day=1, close_day=2, net=0.0, index=0, reason="roll",
                   instrument_id=1, symbol="NQH4", gross=40.0, costs=2.0),
        make_trade(open_day=2, close_day=4, net=0.0, index=1, reason="signal",
                   instrument_id=2, symbol="NQM4", gross=60.0, costs=2.0),
    )
    episodes, att = _attribute(sched, trades, fills)

    assert len(episodes) == 1, "a roll never creates a synthetic strategy episode"
    e = att.economics[0]
    assert e.n_attributed_trades == 2 and e.n_roll_closes == 1
    assert e.contracts_traded == ("NQH4", "NQM4")
    assert e.gross_pnl_usd == pytest.approx(100.0)
    assert e.costs_usd == pytest.approx(8.0), "all four roll-maintenance fills, one episode"
    assert att.unattributed_fill_costs_usd == 0.0


def test_case_F_roll_close_then_reopen_then_a_later_strategy_exit():
    """Two rolls inside one directional run, then the strategy's own exit."""
    from ml_fixtures import make_fill

    sched = make_schedule([0, 1, 1, 1, 1, 1, 0])
    fills = (
        make_fill(day=1, side="buy", quantity=1, index=0, instrument_id=1, symbol="NQH4"),
        make_fill(day=2, side="sell", quantity=1, index=1, instrument_id=1, symbol="NQH4"),
        make_fill(day=2, side="buy", quantity=1, index=2, instrument_id=2, symbol="NQM4"),
        make_fill(day=4, side="sell", quantity=1, index=3, instrument_id=2, symbol="NQM4"),
        make_fill(day=4, side="buy", quantity=1, index=4, instrument_id=3, symbol="NQU4"),
        make_fill(day=6, side="sell", quantity=1, index=5, instrument_id=3, symbol="NQU4"),
    )
    trades = (
        make_trade(open_day=1, close_day=2, net=0.0, index=0, reason="roll",
                   instrument_id=1, symbol="NQH4", gross=10.0, costs=2.0),
        make_trade(open_day=2, close_day=4, net=0.0, index=1, reason="roll",
                   instrument_id=2, symbol="NQM4", gross=20.0, costs=2.0),
        make_trade(open_day=4, close_day=6, net=0.0, index=2, reason="signal",
                   instrument_id=3, symbol="NQU4", gross=30.0, costs=2.0),
    )
    episodes, att = _attribute(sched, trades, fills)

    assert len(episodes) == 1, (
        "roll-maintenance fills stay in the SAME primary episode: the primary "
        "target direction never changed"
    )
    e = att.economics[0]
    assert e.n_attributed_trades == 3 and e.n_roll_closes == 2
    assert e.gross_pnl_usd == pytest.approx(60.0)
    assert e.costs_usd == pytest.approx(12.0)
    assert e.information_horizon_ts_ns == ts(6)
    assert att.unattributed_trades == () and att.unattributed_fill_costs_usd == 0.0


# ==========================================================================
# 17. Phase 15A.1 -- economic reconciliation against the engine's own totals
# ==========================================================================
def test_a_fully_closed_run_reconciles_to_the_engine_net_pnl():
    from alpha_agent.ml.episodes import reconcile_episode_economics
    from ml_fixtures import make_fill

    sched = make_schedule([0, 1, -1, 0])
    fills = (
        make_fill(day=1, side="buy", quantity=1, index=0),
        make_fill(day=2, side="sell", quantity=2, index=1),
        make_fill(day=3, side="buy", quantity=1, index=2),
    )
    trades = (
        make_trade(open_day=1, close_day=2, net=0.0, index=0, direction=1,
                   gross=50.0, costs=4.0),
        make_trade(open_day=2, close_day=3, net=0.0, index=1, direction=-1,
                   gross=30.0, costs=2.0),
    )
    _episodes, att = _attribute(sched, trades, fills)
    engine_gross = sum(t.gross_pnl_usd for t in trades)
    engine_costs = sum(f.commission_usd for f in fills)

    rec = reconcile_episode_economics(
        att,
        engine_gross_pnl_usd=engine_gross,
        engine_costs_usd=engine_costs,
        engine_net_pnl_usd=engine_gross - engine_costs,
    )
    assert rec.reconciles, rec
    assert rec.gross_residual_usd == pytest.approx(0.0, abs=1e-9)
    assert rec.costs_residual_usd == pytest.approx(0.0, abs=1e-9)
    assert rec.net_residual_usd == pytest.approx(0.0, abs=1e-9), "no unexplained residual"
    assert rec.labelled_net_pnl_usd == pytest.approx(engine_gross - engine_costs)


def test_an_open_final_episode_exactly_explains_the_difference():
    """Closed labelled, excluded/open, and engine economics -- reported separately."""
    from alpha_agent.ml.episodes import reconcile_episode_economics
    from ml_fixtures import make_fill

    # episode 0 closes; episode 1 opens and is still open at the schedule end.
    sched = make_schedule([0, 1, 0, -1, -1])
    fills = (
        make_fill(day=1, side="buy", quantity=1, index=0),
        make_fill(day=2, side="sell", quantity=1, index=1),
        make_fill(day=3, side="sell", quantity=1, index=2),
        make_fill(day=4, side="buy", quantity=1, index=3),      # engine's end-of-test close
    )
    trades = (
        make_trade(open_day=1, close_day=2, net=0.0, index=0, direction=1,
                   gross=100.0, costs=2.0),
        make_trade(open_day=3, close_day=4, net=0.0, index=1, direction=-1,
                   reason="eot", gross=-25.0, costs=2.0),
    )
    episodes, att = _attribute(sched, trades, fills)
    assert [e.is_terminated for e in episodes] == [True, False]

    engine_gross = sum(t.gross_pnl_usd for t in trades)
    engine_costs = sum(f.commission_usd for f in fills)
    rec = reconcile_episode_economics(
        att,
        engine_gross_pnl_usd=engine_gross,
        engine_costs_usd=engine_costs,
        engine_net_pnl_usd=engine_gross - engine_costs,
        labelled_episode_indices=[0],          # the open episode is EXCLUDED, never labelled 0
    )
    assert rec.reconciles, rec
    assert rec.n_labelled_episodes == 1 and rec.n_excluded_episodes == 1
    assert rec.labelled_net_pnl_usd == pytest.approx(96.0)      # 100 gross - 4 round turn
    assert rec.excluded_net_pnl_usd == pytest.approx(-29.0)     # -25 gross - 4 round turn
    # The difference is EXACTLY the open episode. Nothing is silently dropped.
    assert (
        rec.engine_net_pnl_usd - rec.labelled_net_pnl_usd
    ) == pytest.approx(rec.open_episode_net_pnl_usd)
    assert rec.unattributed_fill_costs_usd == 0.0


def test_reconciliation_surfaces_a_dropped_fill_instead_of_hiding_it():
    """If attribution ever lost a commission, the residual says so."""
    from alpha_agent.ml.episodes import reconcile_episode_economics
    from ml_fixtures import make_fill

    sched = make_schedule([0, 1, 1, 0])
    fills = (
        make_fill(day=1, side="buy", quantity=1, index=0),
        make_fill(day=3, side="sell", quantity=1, index=1),
    )
    trades = (make_trade(open_day=1, close_day=3, net=0.0, index=0, gross=100.0, costs=2.0),)
    _episodes, att = _attribute(sched, trades, fills)
    rec = reconcile_episode_economics(
        att,
        engine_gross_pnl_usd=100.0,
        engine_costs_usd=6.0,                  # the engine charged more than we attributed
        engine_net_pnl_usd=94.0,
    )
    assert not rec.reconciles
    assert rec.costs_residual_usd == pytest.approx(2.0)


@pytest.mark.skipif(not CLI.exists(), reason="C++ CLI not built")
def test_episode_economics_reconcile_with_a_real_cpp_run(tmp_path):
    """The same identity, against the engine's own headline numbers."""
    import baseline_fixtures as bf
    from alpha_agent.adapters.targets_bridge import run_targets_backtest_cli
    from alpha_agent.backtest.targets import TargetSchedule, TargetScheduleRow
    from alpha_agent.ml.episodes import (
        attribute_episode_economics,
        reconcile_episode_economics,
    )

    bars = bf.boundary_bars(bf.trending_closes(40), instrument_id=915004)
    contracts = bf.one_contract(
        instrument_id=915004, raw_symbol="NQU6", root="NQ",
        expiration_ns=bf.BASE_NS + 10**15,
    )
    fp = "stratdsl1:" + "c" * 64
    tss = [int(t) for t in bars["ts_event_ns"]]
    sched = TargetSchedule(
        root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
        strategy_dsl_version="strategy-dsl/1", feature_engine_version="0.9.2", warmup_bars=0,
        rows=tuple(
            TargetScheduleRow(
                ts_event_ns=t, root_symbol="NQ", target_units=u, strategy_fingerprint=fp
            )
            for t, u in ((tss[5], 1), (tss[15], -1), (tss[25], 0))
        ),
    )
    trades_csv, fills_csv = tmp_path / "t.csv", tmp_path / "f.csv"
    res = run_targets_backtest_cli(
        bars, contracts, sched, executable=CLI, work_dir=tmp_path,
        trades_out_path=trades_csv, fills_out_path=fills_csv,
    )
    episodes = extract_primary_episodes(sched)
    att = attribute_episode_economics(
        episodes, read_closed_trades_csv(trades_csv), read_fills_csv(fills_csv)
    )
    rec = reconcile_episode_economics(
        att,
        engine_gross_pnl_usd=res["gross_pnl_usd"],
        engine_costs_usd=res["costs_usd"],
        engine_net_pnl_usd=res["net_pnl_usd"],
    )
    assert rec.reconciles, rec
    assert att.unattributed_trades == ()
    assert att.unattributed_fill_costs_usd == pytest.approx(0.0)
    assert att.unmatched_trade_indices == ()


# ==========================================================================
# 18. Phase 15A.1 -- the model input allowlist
# ==========================================================================
def test_the_allowlist_is_exactly_the_frozen_feature_set():
    from alpha_agent.ml.allowlist import model_input_allowlist

    allowed = model_input_allowlist(ML_FEATURE_SET)
    assert allowed[: len(ML_FEATURE_SET.features)] == ML_FEATURE_SET.ordered_aliases
    assert allowed[len(ML_FEATURE_SET.features):] == ML_FEATURE_SET.categoricals
    assert set(allowed) == set(ML_FEATURE_SET.ordered_aliases) | {"root_symbol", "primary_side"}


@pytest.mark.parametrize(
    "column",
    [
        "label",
        "episode_net_pnl_usd",
        "episode_gross_pnl_usd",
        "episode_costs_usd",
        "episode_duration_ns",
        "exit_timestamp",
        "exit_decision_ts_ns",
        "label_end_timestamp",
        "information_horizon_ts_ns",
        "n_attributed_fills",
        "n_roll_closes",
        "mae_usd",
        "mfe_usd",
        "realised_regime",
        "close_reason",
    ],
)
def test_audit_and_label_fields_can_never_enter_the_model_matrix(column):
    from alpha_agent.ml.allowlist import assert_model_inputs_allowed
    from alpha_agent.ml.errors import ForbiddenModelInput

    with pytest.raises(ForbiddenModelInput, match="ALLOWLIST"):
        assert_model_inputs_allowed(
            (*ML_FEATURE_SET.ordered_aliases, column), ML_FEATURE_SET
        )


def test_the_allowlist_fails_closed_on_an_unforeseen_column():
    """An allowlist, not a blacklist: a column nobody anticipated is still refused."""
    from alpha_agent.ml.allowlist import assert_model_inputs_allowed
    from alpha_agent.ml.errors import ForbiddenModelInput

    with pytest.raises(ForbiddenModelInput, match="not one of the frozen feature set"):
        assert_model_inputs_allowed(("some_future_column_nobody_listed",), ML_FEATURE_SET)
    with pytest.raises(ForbiddenModelInput, match="more than once"):
        assert_model_inputs_allowed(("mom_20", "mom_20"), ML_FEATURE_SET)


def test_the_frozen_feature_set_declares_no_forward_looking_input():
    from alpha_agent.ml.allowlist import assert_no_audit_field_is_allowlisted

    assert_no_audit_field_is_allowlisted(ML_FEATURE_SET)


def test_training_refuses_a_forbidden_matrix_column_before_any_fit():
    from alpha_agent.ml.errors import ForbiddenModelInput

    X, y, timeline, ids, roots = synthetic_dataset()
    with pytest.raises(ForbiddenModelInput, match="episode_net_pnl_usd"):
        _run_on(
            X, y, timeline, ids, roots,
            matrix_columns=("f0", "f1", "f2", "episode_net_pnl_usd"),
        )


def test_the_label_may_look_forward_while_the_features_may_not():
    """The asymmetry, stated once, in both directions."""
    sched = make_schedule([0, 1, 1, 0])
    episodes = extract_primary_episodes(sched)
    econ = {0: make_economics(0, net=50.0, horizon_day=3)}
    events = build_meta_label_events(
        episodes=episodes, economics=econ, spec=MetaLabelSpec(),
        strategy_family="tsmom", dataset_fingerprint="ds",
        feature_set_fingerprint=ML_FEATURE_SET.identity(), target_schedule_hash="tsh",
    )
    ev = events.events[0]
    assert ev.label_end_timestamp > ev.decision_timestamp, "the label looks forward"
    assert ev.feature_timestamp <= ev.decision_timestamp, "the features do not"
    # ... and the forward-looking fields the event carries are audit-only.
    from alpha_agent.ml.allowlist import KNOWN_AUDIT_ONLY_FIELDS

    for field in ("episode_net_pnl_usd", "episode_costs_usd", "label", "label_end_timestamp"):
        assert field in KNOWN_AUDIT_ONLY_FIELDS
        assert field not in ML_FEATURE_SET.ordered_aliases


# ==========================================================================
# 19. Phase 15A.1 -- global temporal isolation across POOLED roots
# ==========================================================================
def test_every_outer_boundary_applies_to_every_root():
    from alpha_agent.ml.splits import assert_global_temporal_isolation

    X, y, timeline, ids, roots = synthetic_dataset(n_events=600)
    del X, y, ids
    folds = build_outer_folds(NESTED_CV, timeline)
    for fold in folds:
        census = assert_global_temporal_isolation(fold, timeline, roots)
        assert set(census) == {"ES", "NQ", "CL", "GC", "ZN"}, "all five roots are pooled"
        for root, counts in census.items():
            assert counts["n_train"] > 0 and counts["n_test"] > 0, root


def test_a_test_year_excludes_that_year_for_every_root_not_just_one():
    """If 2023 is the outer TEST block, 2023 ES/NQ/CL/GC/ZN are ALL out of training."""
    _X, _y, timeline, _ids, roots = synthetic_dataset(n_events=600)
    fold_2023 = next(
        f for f in build_outer_folds(NESTED_CV, timeline)
        if f.test_start_ts_ns == 1_672_531_200_000_000_000
    )
    for row in fold_2023.train_row_indices:
        assert timeline.decision_ts_ns[row] < fold_2023.test_start_ts_ns, (
            f"root {roots[row]} leaked a 2023-or-later event into training"
        )
    # and per root, explicitly
    by_root: dict[str, int] = {}
    for row in fold_2023.train_row_indices:
        if timeline.decision_ts_ns[row] >= fold_2023.test_start_ts_ns:
            by_root[roots[row]] = by_root.get(roots[row], 0) + 1
    assert by_root == {}


def test_pooling_never_weakens_isolation_and_the_breach_is_detected():
    from alpha_agent.ml.splits import Fold, assert_global_temporal_isolation

    _X, _y, timeline, _ids, roots = synthetic_dataset(n_events=600)
    fold = build_outer_folds(NESTED_CV, timeline)[2]
    leaked = fold.test_row_indices[0]
    doctored = Fold(
        kind=fold.kind,
        fold_index=fold.fold_index,
        test_start_ts_ns=fold.test_start_ts_ns,
        test_end_ts_ns=fold.test_end_ts_ns,
        # one root's in-test-block event smuggled into training
        train_row_indices=(*fold.train_row_indices, leaked),
        test_row_indices=fold.test_row_indices[1:],
    )
    with pytest.raises(FoldIsolationError, match="pooled temporal isolation broken"):
        assert_global_temporal_isolation(doctored, timeline, roots)


def test_root_is_a_feature_not_a_split_dimension():
    assert "root_symbol" in ML_FEATURE_SET.categoricals
    m = build_candidate_manifest()
    for t in m.trials:
        assert t.training_roots == m.roots, "every trial trains on the pooled universe"
        assert t.root_symbol in m.roots, "and is evaluated on one root's economics"


def test_inner_folds_are_globally_isolated_too():
    from alpha_agent.ml.splits import pooled_isolation_report

    _X, _y, timeline, _ids, roots = synthetic_dataset(n_events=600)
    outer = build_outer_folds(NESTED_CV, timeline)
    report = pooled_isolation_report(outer, timeline, roots)
    assert len(report) == NESTED_CV.n_outer_folds
    for o in outer:
        inner = build_inner_folds(NESTED_CV, timeline, o)
        for f in inner:
            assert f.test_end_ts_ns <= o.test_start_ts_ns
            per_root = pooled_isolation_report((f,), timeline, roots)[0]["per_root"]
            assert per_root, "an inner fold covers the pooled universe too"


# ==========================================================================
# 20. Phase 15A.1 -- no real-market model fitting or performance inspection
# ==========================================================================
#: Phase 15B synthetic-integration / run artifacts DO carry (synthetic or real)
#: results by design -- they are not pre-run freeze artifacts. Every other
#: committed Phase 15 JSON must stay pre-run.
_PHASE_15B_RESULT_ARTIFACT_PREFIXES = (
    "PHASE_15B_", "PRE_REAL_RUN_INTEGRATION_REPORT", "REAL_PATH_SMOKE",
)


def test_phase_15_artifacts_carry_no_ml_performance():
    """Every committed Phase 15 FREEZE artifact is pre-run. None may name a result."""
    forbidden = (
        "roc_auc", "pr_auc", "log_loss_value", "brier", "calibration",
        "sharpe", "net_pnl", "take_rate_observed", "filtered_pnl", "drawdown",
    )
    for path in sorted(Path("outputs/phase_15").glob("*.json")):
        if path.name.startswith(_PHASE_15B_RESULT_ARTIFACT_PREFIXES):
            continue
        blob = path.read_text().lower()
        for key in forbidden:
            assert f'"{key}"' not in blob, f"{path.name} leaks a performance field {key!r}"


def test_phase_15b_result_artifacts_declare_they_are_not_pre_run():
    """A Phase 15B synthetic report must be unmistakably a synthetic fixture, so a
    reader can never confuse it with a real Phase 15 result."""
    for path in sorted(Path("outputs/phase_15").glob("*.json")):
        if not path.name.startswith(_PHASE_15B_RESULT_ARTIFACT_PREFIXES):
            continue
        doc = json.loads(path.read_text())
        if path.name == "PHASE_15B_PREFLIGHT.json":
            continue  # preflight is genuinely pre-run: no fit, no economics
        if path.name == "REAL_PATH_SMOKE.json":
            assert doc["did_not_fit_model"] is True
            assert doc["did_not_inspect_meta_label_performance"] is True
            assert doc["did_not_write_production_registry"] is True
            continue
        assert doc.get("engine", "").startswith("synthetic") or doc.get("mode") == "run_real"
        if doc.get("engine", "").startswith("synthetic"):
            assert doc["no_real_performance"]["real_model_fitted"] is False
            assert doc["no_real_performance"]["production_registry_rows_written"] == 0


def test_no_phase_15_artifact_mentions_the_locked_holdout_as_data():
    from alpha_agent.ml.guards import assert_no_holdout_in_config_payload

    for name in ("PRETRAIN_TRIAL_ACCOUNTING.json", "MANIFEST_SUPERSESSION.json",
                 "ENVIRONMENT_PROVENANCE.json", "PROTOCOL_FREEZE_CORRECTED.json"):
        assert_no_holdout_in_config_payload(
            json.loads((Path("outputs/phase_15") / name).read_text()),
            exclusive_bound_keys=frozenset({"corpus_end_ts_ns"}),
            half_open_window_keys=frozenset({"outer_test_blocks"}),
            path=name,
        )


def test_the_corrected_manifest_still_never_reaches_2025():
    m = build_candidate_manifest()
    assert m.nested_cv.corpus_end_ts_ns == HOLDOUT_START_NS       # EXCLUSIVE bound
    for start, end in m.nested_cv.outer_test_blocks:
        assert start < HOLDOUT_START_NS and end <= HOLDOUT_START_NS
    assert_no_performance_fields(m)
