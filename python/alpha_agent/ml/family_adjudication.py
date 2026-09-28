"""Assemble and adjudicate the frozen 60-hypothesis Phase 15 family.

12 pooled pipelines ``(primary_family x {logistic+regime, tree+regime, tree})``
each train once (nested chronological CV, pooled over the five roots) and emit
OOF predictions. Each pipeline then yields five per-root economic trials, so the
family is exactly ``12 x 5 = 60`` -- 40 headline + 20 regime ablation.

Per trial:

* PRIMARY vs META economics on exactly aligned support, at 1.0x / 1.5x / 2.0x;
* the centred-block-bootstrap GATING null on the META baseline daily returns;
* the per-trial DSR (search breadth 3 for logistic, 8 for the tree);
* diagnostics (calibration first) -- reported, never gated;
* the inherited :func:`evaluate_policy` verdict;
* the two Phase-15-specific rules (:func:`adjudicate_phase_15_trial`);
* the conditional TAKE-rate-matched placebo, for a trial that rejected its
  gating null.

The predeclared BH family stays exactly 60 (rule 1): a trial that a typed
refusal keeps from producing a valid statistical result enters BH at the
conservative ``p = 1`` and its refusal is preserved, never dropped.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from alpha_agent.ml.adjudication import (
    MLValidationSpec,
    Phase15TrialAdjudication,
    adjudicate_phase_15_trial,
    phase_15_bh_family_p_values,
)
from alpha_agent.ml.corpus import (
    CorpusResult,
    PrimaryScheduleProvider,
    build_corpus_for_scope,
)
from alpha_agent.ml.diagnostics import MLDiagnostics, compute_ml_diagnostics
from alpha_agent.ml.economics import TrialEconomics, evaluate_trial_economics
from alpha_agent.ml.engine_io import CachingEngineRunner, MetaEngineRunner
from alpha_agent.ml.enums import MLModelFamily, MLRefusalReason, MLTrialRole, RegimeSpecKind
from alpha_agent.ml.errors import InsufficientEventsError, MLProtocolError
from alpha_agent.ml.feature_matrix import FeatureProvider, build_pooled_matrix
from alpha_agent.ml.identity import MLExperimentSpec
from alpha_agent.ml.labels import MetaLabelSpec
from alpha_agent.ml.manifest import (
    ML_FEATURE_SET,
    MODEL_SEARCHES,
    NESTED_CV,
    PLACEBO_DRAWS,
    PREPROCESSING_SPEC,
    REGIME_ABLATION,
    REGIME_BASELINE,
    ROOTS,
    MLCandidateManifest,
    MLCandidateTrial,
)
from alpha_agent.ml.placebo import PlaceboResult, run_placebo_for_trial
from alpha_agent.ml.predictions import MLPredictionFrame, assert_oof_provenance
from alpha_agent.ml.splits import build_outer_folds
from alpha_agent.ml.training import MLTrainingReport, MLTrainingSpec, run_nested_cv
from alpha_agent.ml.trials import MLIdentityPlanes, experiment_spec_for_trial
from alpha_agent.validation.cost_stress import (
    CostScenarioResult,
    CostStressPlan,
    summarize_cost_stress,
)
from alpha_agent.validation.crossmarket import CrossMarketEvidence
from alpha_agent.validation.dsr import DeflatedSharpeResult, deflated_sharpe_ratio
from alpha_agent.validation.enums import EvidenceStatus, NullMethod, NullRole, Verdict
from alpha_agent.validation.metrics import OosMetrics, daily_sharpe, describe_oos
from alpha_agent.validation.nulls import (
    NullTestConfig,
    NullTestResult,
    centered_block_bootstrap_null_stats,
    summarize_null,
)
from alpha_agent.validation.policy import PolicyOutcome, ReliabilityPolicy, evaluate_policy
from alpha_agent.validation.regime import RegimeStabilityResult
from alpha_agent.validation.walkforward import WalkForwardResult

_GATING_NULL = NullTestConfig(methods=(NullMethod.CENTERED_BLOCK_BOOTSTRAP,))


class PipelineKey(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    primary_family: str
    model_family: str
    regime_kind: str

    @property
    def label(self) -> str:
        return f"{self.primary_family}__{self.model_family}__{self.regime_kind}"


class PipelineOutput(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    key: PipelineKey
    refused: bool
    refusal_reason: str = ""
    refusal_detail: dict = Field(default_factory=dict)
    n_pooled_events: int = 0
    per_root_event_counts: dict[str, int] = Field(default_factory=dict)
    n_oof_predictions: int = 0
    training_report: MLTrainingReport | None = None
    corpus_reconciles: dict[str, bool] = Field(default_factory=dict)
    feature_missing_events: int = 0


class TrialResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    trial_label: str
    experiment_identity: str
    friendly_experiment_id: str
    root_symbol: str
    primary_family: str
    model_family: str
    regime_kind: str
    trial_role: str

    refused: bool
    typed_refusal: str | None = None

    economics: TrialEconomics | None = None
    diagnostics: MLDiagnostics | None = None
    gating_null: NullTestResult | None = None
    dsr: DeflatedSharpeResult | None = None
    dsr_family_380: DeflatedSharpeResult | None = None
    base_policy_outcome: dict | None = None
    adjudication: Phase15TrialAdjudication
    placebo: PlaceboResult | None = None
    fold_consistency: float | None = None
    meta_max_cost_degradation: float | None = None

    bh_p_value: float
    bh_q_value: float | None = None
    bh_rejected_at_q: bool | None = None


class Phase15FamilyResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    manifest_fingerprint: str
    validation_spec_fingerprint: str
    reliability_policy_fingerprint: str
    engine: str
    #: False when only a subset of pipelines was run (a scoped integration
    #: check); the production Phase 15B.2 run is always the full family.
    is_full_family: bool = True
    n_pipelines_run: int = 12
    bh_family_size: int
    bh_q: float
    n_headline: int
    n_ablation: int
    n_refused: int
    n_pass: int
    n_reject: int
    n_inconclusive: int
    verdict_counts: dict[str, int]
    dsr_family_effective_trial_count: int
    pipelines: tuple[PipelineOutput, ...]
    trials: tuple[TrialResult, ...]
    #: trials whose base ReliabilityPolicy PASS triggered the conditional placebo
    placebo_triggered_trials: int
    #: actual C++ meta-labeled runs = 20 frozen draws per triggered trial
    placebo_cpp_draw_evaluations: int
    #: {trial_label: reason} for any triggered trial where a typed refusal
    #: prevented one or more of its 20 draws (empty in the normal case)
    placebo_draws_prevented: dict[str, str] = Field(default_factory=dict)
    holdout_candidates: tuple[str, ...] = ()
    #: economics-engine calls served vs actually executed (PRIMARY schedules
    #: repeat across pipelines, so the executed count is the frozen 240 budget).
    n_engine_calls: int = 0
    n_engine_runs_executed: int = 0
    n_engine_cache_hits: int = 0


# --------------------------------------------------------------------------
# pipeline training
# --------------------------------------------------------------------------
def _regime_for(kind: RegimeSpecKind):
    return REGIME_ABLATION if kind is RegimeSpecKind.NONE else REGIME_BASELINE


def train_pipeline(
    key: PipelineKey,
    *,
    provider: PrimaryScheduleProvider,
    feature_provider: FeatureProvider,
    engine: MetaEngineRunner,
    meta_label_spec: MetaLabelSpec,
    dataset_fingerprint: str,
    planes: MLIdentityPlanes,
    code_commit: str = "",
) -> tuple[PipelineOutput, MLPredictionFrame | None, dict[str, object]]:
    """Train one pooled pipeline. Returns ``(output, oof_frame, context)``.

    ``context`` carries the per-root primary schedules and corpus results the
    downstream economics need, keyed by root symbol.
    """
    model_family = MLModelFamily(key.model_family)
    regime = _regime_for(RegimeSpecKind(key.regime_kind))
    scopes = [
        s for s in provider.scopes()
        if s.primary_family == key.primary_family and s.root_symbol in ROOTS
    ]
    corpora: dict[str, CorpusResult] = {}
    schedules: dict[str, object] = {}
    events_by_root: dict[str, tuple] = {}
    reconciles: dict[str, bool] = {}
    for scope in scopes:
        cr = build_corpus_for_scope(
            scope,
            provider=provider,
            engine=engine,
            meta_label_spec=meta_label_spec,
            dataset_fingerprint=dataset_fingerprint,
            feature_set_fingerprint=ML_FEATURE_SET.identity(),
        )
        corpora[scope.root_symbol] = cr
        schedules[scope.root_symbol] = provider.schedule_for(scope)
        events_by_root[scope.root_symbol] = cr.events.events
        reconciles[scope.root_symbol] = cr.reconciliation.reconciles

    dataset, missing = build_pooled_matrix(
        key.primary_family, events_by_root, provider=feature_provider
    )
    per_root = dict(dataset.meta.per_root_counts)

    training_spec = MLTrainingSpec(
        nested_cv=NESTED_CV, preprocessing=PREPROCESSING_SPEC, regime=regime
    )
    search = MODEL_SEARCHES[model_family]

    ctx: dict[str, object] = {
        "scopes": {s.root_symbol: s for s in scopes},
        "schedules": schedules,
        "corpora": corpora,
        "dataset": dataset,
    }

    try:
        frame, report = run_nested_cv(
            X=dataset.X,
            y=dataset.y,
            event_ids=dataset.meta.event_ids,
            root_symbols=dataset.meta.root_symbols,
            timeline=dataset.timeline,
            training_spec=training_spec,
            model_search=search,
            feature_set=ML_FEATURE_SET,
            meta_label_spec=meta_label_spec,
            dataset_fingerprint=dataset_fingerprint,
            primary_strategy_fingerprints=dataset.primary_strategy_fingerprints,
            experiment_identity=f"pipeline:{key.label}",
            experiment_label=key.label,
            code_commit=code_commit,
            matrix_columns=dataset.meta.allowlist_columns,
            feature_columns=dataset.meta.fitted_columns,
            regime_input_column_indices=dataset.meta.regime_input_column_indices,
        )
    except InsufficientEventsError as exc:
        return (
            PipelineOutput(
                key=key,
                refused=True,
                refusal_reason=exc.reason.value if exc.reason else MLRefusalReason.INSUFFICIENT_TRAIN_EVENTS.value,
                refusal_detail=dict(exc.detail or {}),
                n_pooled_events=dataset.meta.n_events,
                per_root_event_counts=per_root,
                corpus_reconciles=reconciles,
                feature_missing_events=len(missing),
            ),
            None,
            ctx,
        )

    folds = build_outer_folds(NESTED_CV, dataset.timeline)
    assert_oof_provenance(frame, folds)

    return (
        PipelineOutput(
            key=key,
            refused=False,
            n_pooled_events=dataset.meta.n_events,
            per_root_event_counts=per_root,
            n_oof_predictions=frame.n,
            training_report=report,
            corpus_reconciles=reconciles,
            feature_missing_events=len(missing),
        ),
        frame,
        ctx,
    )


# --------------------------------------------------------------------------
# per-trial adjudication
# --------------------------------------------------------------------------
def _oos_metrics(returns: np.ndarray, econ: TrialEconomics) -> OosMetrics:
    base_meta = next(m for m in econ.meta if m.cost_scenario_label == "baseline_1_0x")
    return describe_oos(
        returns,
        n_fills=base_meta.n_fills,
        n_trades=base_meta.n_trades,
        oos_gross_pnl_usd=base_meta.gross_pnl_usd,
        oos_costs_usd=base_meta.costs_usd,
        oos_net_pnl_usd=base_meta.net_pnl_usd,
        capital_base_usd=100_000.0,
        positive_fold_count=0,
        total_fold_count=0,
    )


def _fold_consistency(econ: TrialEconomics) -> WalkForwardResult:
    """Fold consistency over the five nested-CV outer test blocks (calendar years).

    Each outer block's META daily net return sign is the fold outcome, split on
    the REAL frozen calendar-year boundaries (not an equal-chunk proxy).
    """
    from alpha_agent.ml.manifest import OUTER_TEST_BLOCKS

    rets = np.asarray(econ.meta_baseline_daily_returns, dtype=float)
    ts = np.asarray(econ.meta_baseline_daily_ts_ns, dtype="int64")
    n_blocks = len(OUTER_TEST_BLOCKS)
    pos = 0
    evaluated = 0
    if ts.size == rets.size and ts.size:
        for lo, hi in OUTER_TEST_BLOCKS:
            chunk = rets[(ts >= lo) & (ts < hi)]
            if chunk.size == 0:
                continue
            evaluated += 1
            if float(chunk.sum()) > 0.0:
                pos += 1
    else:  # no timestamps (should not happen on the real path) -- equal chunks
        n = rets.size
        for b in range(n_blocks):
            chunk = rets[b * n // n_blocks : (b + 1) * n // n_blocks]
            if chunk.size == 0:
                continue
            evaluated += 1
            if float(chunk.sum()) > 0.0:
                pos += 1
    return WalkForwardResult(
        config_fingerprint="phase_15_nested_cv_outer_blocks",
        n_folds_planned=n_blocks,
        n_folds_evaluated=evaluated,
        positive_fold_count=pos,
        fold_consistency=(pos / evaluated) if evaluated else 0.0,
        combined_oos_net_pnl_usd=float(rets.sum()),
        combined_oos_daily_sharpe=daily_sharpe(rets) if rets.size else float("nan"),
        combined_oos_annualized_sharpe=float("nan"),
        folds=(),
    )


def _cost_report(econ: TrialEconomics):
    from alpha_agent.validation.cost_stress import CostScenario

    stress_plan = CostStressPlan(
        base_commission_per_contract_usd=2.0,
        base_slippage_ticks=0.0,
        base_spread_ticks=0.0,
        scenarios=tuple(
            CostScenario(label=m.cost_scenario_label, multiplier=m.cost_multiplier)
            for m in econ.meta
        ),
    )
    base = next(m for m in econ.meta if m.cost_scenario_label == "baseline_1_0x")

    def _sharpe(m) -> float:
        return float(m.daily_sharpe) if np.isfinite(m.daily_sharpe) else 0.0

    per = [
        CostScenarioResult(
            label=m.cost_scenario_label,
            commission_per_contract_usd=2.0 * m.cost_multiplier,
            slippage_ticks=0.0,
            spread_ticks=0.0,
            n_fills=m.n_fills,
            oos_gross_pnl_usd=m.gross_pnl_usd,
            oos_costs_usd=m.costs_usd,
            oos_net_pnl_usd=m.net_pnl_usd,
            daily_sharpe=_sharpe(m),
            annualized_sharpe=float(m.annualized_sharpe) if np.isfinite(m.annualized_sharpe) else 0.0,
            net_pnl_ratio_vs_baseline=(
                float(m.net_pnl_usd / base.net_pnl_usd)
                if abs(base.net_pnl_usd) > 1e-9
                else float("nan")
            ),
            sharpe_delta_vs_baseline=_sharpe(m) - _sharpe(base),
        )
        for m in econ.meta
    ]
    return summarize_cost_stress(stress_plan, per)


def _gating_null(econ: TrialEconomics) -> NullTestResult:
    rets = np.asarray(econ.meta_baseline_daily_returns, dtype=float)
    observed = daily_sharpe(rets) if rets.size >= 2 else 0.0
    null_stats = centered_block_bootstrap_null_stats(rets, _GATING_NULL, daily_sharpe)
    return summarize_null(
        NullMethod.CENTERED_BLOCK_BOOTSTRAP,
        NullRole.GATING,
        "daily_sharpe",
        observed,
        null_stats,
        canonical_oos_statistic=observed,
    )


def assemble_trial(
    trial: MLCandidateTrial,
    *,
    pipeline_out: PipelineOutput,
    oof_frame: MLPredictionFrame | None,
    ctx: dict[str, object],
    spec: MLValidationSpec,
    policy: ReliabilityPolicy,
    planes: MLIdentityPlanes,
    engine: MetaEngineRunner,
    family_meta_daily_sharpes: list[float],
    capital_base_usd: float,
    economics: TrialEconomics | None = None,
    gating_null: NullTestResult | None = None,
    family_fdr: object | None = None,
    fdr_index: int = 0,
) -> TrialResult:
    exp_spec: MLExperimentSpec = experiment_spec_for_trial(trial, planes=planes)
    identity = exp_spec.experiment_identity()
    root = trial.root_symbol
    common = {
        "trial_label": trial.trial_label,
        "experiment_identity": identity,
        "friendly_experiment_id": exp_spec.friendly_id(),
        "root_symbol": root,
        "primary_family": trial.primary_family,
        "model_family": trial.model_family.value,
        "regime_kind": trial.regime_kind.value,
        "trial_role": trial.trial_role.value,
    }
    adj_kw = {
        "trial_label": trial.trial_label,
        "root_symbol": root,
        "primary_family": trial.primary_family,
        "model_family": trial.model_family.value,
        "regime_kind": trial.regime_kind.value,
        "trial_role": trial.trial_role.value,
    }

    if pipeline_out.refused or oof_frame is None:
        refusal = pipeline_out.refusal_reason or MLRefusalReason.INSUFFICIENT_TRAIN_EVENTS.value
        adj = adjudicate_phase_15_trial(
            spec, **adj_kw, bh_p_value=1.0, base_outcome=None, typed_refusal=refusal,
        )
        return TrialResult(
            **common, refused=True, typed_refusal=refusal, adjudication=adj, bh_p_value=1.0,
        )

    schedules = ctx["schedules"]
    econ = economics if economics is not None else evaluate_trial_economics(
        trial_label=trial.trial_label,
        root_symbol=root,
        primary_family=trial.primary_family,
        primary_schedule=schedules[root],
        oof_frame=oof_frame,
        engine=engine,
        capital_base_usd=capital_base_usd,
    )
    # per-root OOF frame for diagnostics
    root_frame = MLPredictionFrame(
        predictions=tuple(p for p in oof_frame.predictions if p.root_symbol == root)
    )
    diagnostics = compute_ml_diagnostics(root_frame)

    meta_rets = np.asarray(econ.meta_baseline_daily_returns, dtype=float)
    gating = gating_null if gating_null is not None else _gating_null(econ)
    breadth = int(
        dict(spec.multiple_testing.dsr_effective_trial_count_by_model_family)[
            trial.model_family.value
        ]
    )
    dsr = deflated_sharpe_ratio(
        meta_rets, n_trials=breadth, trial_daily_sharpes=list(family_meta_daily_sharpes)
    )
    dsr_family = deflated_sharpe_ratio(
        meta_rets,
        n_trials=spec.multiple_testing.dsr_effective_trial_count,
        trial_daily_sharpes=list(family_meta_daily_sharpes),
    )

    oos = _oos_metrics(meta_rets, econ)
    wf = _fold_consistency(econ)
    cost_report = _cost_report(econ)

    fdr_result = (
        family_fdr
        if family_fdr is not None
        else _singleton_fdr(gating.p_value, policy.fdr_q_threshold)
    )
    fdr_idx = fdr_index if family_fdr is not None else 0
    base_outcome: PolicyOutcome = evaluate_policy(
        policy,
        oos_metrics=oos,
        walk_forward=wf,
        null_results=[gating],
        fdr_result=fdr_result,
        canonical_trial_index=fdr_idx,
        dsr_result=dsr,
        cost_report=cost_report,
        parameter_stability=None,
        ablation_report=None,
        regime_result=RegimeStabilityResult(status=EvidenceStatus.NOT_EVALUATED),
        cross_market=CrossMarketEvidence(status=EvidenceStatus.NOT_EVALUATED),
    )

    placebo_should_run = (
        base_outcome.verdict is Verdict.PASS
        and gating.p_value <= policy.null_p_value_max
    )
    placebo = run_placebo_for_trial(
        trial_label=trial.trial_label,
        experiment_identity=identity,
        root_symbol=root,
        primary_schedule=schedules[root],
        oof_frame=oof_frame,
        observed_meta_daily_sharpe=econ.baseline_meta_daily_sharpe,
        engine=engine,
        should_run=placebo_should_run,
        reason_not_run="" if placebo_should_run else "base ReliabilityPolicy verdict is not PASS",
    )

    adj = adjudicate_phase_15_trial(
        spec,
        **adj_kw,
        bh_p_value=float(gating.p_value),
        base_outcome=base_outcome,
        meta_daily_sharpe=econ.baseline_meta_daily_sharpe,
        primary_daily_sharpe=econ.baseline_primary_daily_sharpe,
        placebo_observed_statistic=econ.baseline_meta_daily_sharpe if placebo.ran else None,
        placebo_statistics=(
            tuple(d.daily_sharpe for d in placebo.draws) if placebo.ran else None
        ),
    )

    return TrialResult(
        **common,
        refused=False,
        economics=econ,
        diagnostics=diagnostics,
        gating_null=gating,
        dsr=dsr,
        dsr_family_380=dsr_family,
        base_policy_outcome={
            "verdict": base_outcome.verdict.value,
            "reason_codes": [c.value for c in base_outcome.reason_codes],
            "gate_results": base_outcome.gate_results,
        },
        adjudication=adj,
        placebo=placebo,
        fold_consistency=float(wf.fold_consistency),
        meta_max_cost_degradation=float(econ.meta_max_cost_degradation),
        bh_p_value=float(gating.p_value),
    )


def _singleton_fdr(p_value: float, q: float):
    from alpha_agent.validation.fdr import benjamini_hochberg_decisions

    return benjamini_hochberg_decisions([p_value], q, labels=["canonical"])


# --------------------------------------------------------------------------
# the family
# --------------------------------------------------------------------------
def adjudicate_family(
    manifest: MLCandidateManifest,
    *,
    provider: PrimaryScheduleProvider,
    feature_provider: FeatureProvider,
    engine: MetaEngineRunner,
    spec: MLValidationSpec,
    policy: ReliabilityPolicy,
    planes: MLIdentityPlanes,
    meta_label_spec: MetaLabelSpec,
    dataset_fingerprint: str,
    capital_base_usd: float,
    code_commit: str = "",
    only_pipeline_keys: tuple[PipelineKey, ...] | None = None,
) -> Phase15FamilyResult:
    """Run and adjudicate the predeclared Phase 15 hypotheses.

    ``only_pipeline_keys`` -- when given -- restricts the run to those pipelines
    (a SCOPED integration check); the result's ``is_full_family`` is then False
    and only the run trials appear. The production Phase 15B.2 run passes ``None``
    and adjudicates all 60.
    """
    engine = CachingEngineRunner(engine)     # PRIMARY schedules repeat across pipelines
    all_pipeline_keys = sorted(
        {
            PipelineKey(
                primary_family=t.primary_family,
                model_family=t.model_family.value,
                regime_kind=t.regime_kind.value,
            )
            for t in manifest.trials
        },
        key=lambda k: k.label,
    )
    if only_pipeline_keys is not None:
        wanted = {k.label for k in only_pipeline_keys}
        pipeline_keys = [k for k in all_pipeline_keys if k.label in wanted]
        run_trials = tuple(
            t for t in manifest.trials
            if PipelineKey(
                primary_family=t.primary_family,
                model_family=t.model_family.value,
                regime_kind=t.regime_kind.value,
            ).label in wanted
        )
    else:
        pipeline_keys = all_pipeline_keys
        run_trials = manifest.trials
    is_full = only_pipeline_keys is None

    outputs: dict[str, PipelineOutput] = {}
    frames: dict[str, MLPredictionFrame | None] = {}
    contexts: dict[str, dict[str, object]] = {}
    for key in pipeline_keys:
        out, frame, ctx = train_pipeline(
            key,
            provider=provider,
            feature_provider=feature_provider,
            engine=engine,
            meta_label_spec=meta_label_spec,
            dataset_fingerprint=dataset_fingerprint,
            planes=planes,
            code_commit=code_commit,
        )
        outputs[key.label] = out
        frames[key.label] = frame
        contexts[key.label] = ctx

    # first pass: economics for every non-refused trial, to build the family
    # Sharpe dispersion the DSR needs
    def _key_label(t: MLCandidateTrial) -> str:
        return PipelineKey(
            primary_family=t.primary_family,
            model_family=t.model_family.value,
            regime_kind=t.regime_kind.value,
        ).label

    def _refused(t: MLCandidateTrial) -> bool:
        kl = _key_label(t)
        return outputs[kl].refused or frames[kl] is None

    # -- PASS A: economics + the GATING null for every non-refused trial -------
    family_sharpes: list[float] = []
    prelim: dict[str, tuple[TrialEconomics, NullTestResult]] = {}
    for t in run_trials:
        if _refused(t):
            continue
        kl = _key_label(t)
        e = evaluate_trial_economics(
            trial_label=t.trial_label,
            root_symbol=t.root_symbol,
            primary_family=t.primary_family,
            primary_schedule=contexts[kl]["schedules"][t.root_symbol],
            oof_frame=frames[kl],
            engine=engine,
            capital_base_usd=capital_base_usd,
        )
        g = _gating_null(e)
        prelim[t.trial_label] = (e, g)
        if np.isfinite(e.baseline_meta_daily_sharpe):
            family_sharpes.append(e.baseline_meta_daily_sharpe)

    # -- the FIXED 60-family BH (rule 1) -- a refused trial enters at the
    # conservative p = 1 and is NEVER dropped. A scoped integration check runs
    # BH over just the run trials.
    from alpha_agent.validation.fdr import benjamini_hochberg_decisions

    labels = tuple(t.trial_label for t in run_trials)
    p_by = {lbl: prelim[lbl][1].p_value for lbl in labels if lbl in prelim}
    refusal_by = {
        t.trial_label: (outputs[_key_label(t)].refusal_reason or "TRIAL_REFUSED")
        for t in run_trials
        if _refused(t)
    }
    if is_full:
        bh_rows = phase_15_bh_family_p_values(
            labels, p_value_by_trial=p_by, typed_refusal_by_trial=refusal_by,
            plan=spec.multiple_testing,
        )
        bh_input = [p for _, p, _ in bh_rows]
    else:
        bh_input = [p_by.get(lbl, 1.0) for lbl in labels]
    family_fdr = benjamini_hochberg_decisions(
        bh_input, spec.multiple_testing.bh_q, labels=list(labels)
    )
    fdr_index = {lbl: i for i, lbl in enumerate(labels)}

    # -- PASS B: the inherited policy (with the REAL family BH) + the two
    # Phase-15 rules + the conditional placebo, per trial.
    trials: list[TrialResult] = []
    for t in run_trials:
        kl = _key_label(t)
        pe = prelim.get(t.trial_label)
        trials.append(
            assemble_trial(
                t,
                pipeline_out=outputs[kl],
                oof_frame=frames[kl],
                ctx=contexts[kl],
                spec=spec,
                policy=policy,
                planes=planes,
                engine=engine,
                family_meta_daily_sharpes=family_sharpes,
                capital_base_usd=capital_base_usd,
                economics=pe[0] if pe else None,
                gating_null=pe[1] if pe else None,
                family_fdr=family_fdr,
                fdr_index=fdr_index[t.trial_label],
            )
        )

    q_by = {d.label: d.q_value for d in family_fdr.decisions}
    rej_by = {d.label: d.rejected for d in family_fdr.decisions}
    final: list[TrialResult] = [
        t.model_copy(
            update={
                "bh_q_value": q_by[t.trial_label],
                "bh_rejected_at_q": rej_by[t.trial_label],
            }
        )
        for t in trials
    ]

    verdicts = [t.adjudication.verdict for t in final]
    vc = {v: verdicts.count(v) for v in sorted(set(verdicts))}
    headline = sum(1 for t in run_trials if t.trial_role is MLTrialRole.HEADLINE)
    ablation = sum(1 for t in run_trials if t.trial_role is MLTrialRole.ABLATION)

    # PLACEBO ACCOUNTING -- two distinct quantities.
    #   placebo_triggered_trials      = trials whose base ReliabilityPolicy PASS
    #                                   triggered the conditional placebo;
    #   placebo_cpp_draw_evaluations  = actual C++ meta-labeled runs = 20 per
    #                                   triggered trial, unless a typed refusal
    #                                   explicitly prevented a draw.
    triggered = [t for t in final if t.placebo is not None and t.placebo.ran]
    placebo_triggered_trials = len(triggered)
    placebo_cpp_draw_evaluations = sum(len(t.placebo.draws) for t in triggered)
    placebo_draws_prevented = {
        t.trial_label: t.placebo.reason_not_run
        for t in triggered
        if len(t.placebo.draws) != t.placebo.n_draws and t.placebo.reason_not_run
    }
    expected = PLACEBO_DRAWS * placebo_triggered_trials - sum(
        PLACEBO_DRAWS - len(t.placebo.draws)
        for t in triggered
        if t.trial_label in placebo_draws_prevented
    )
    if placebo_cpp_draw_evaluations != expected:
        raise MLProtocolError(
            f"placebo accounting is inconsistent: {placebo_cpp_draw_evaluations} C++ draw "
            f"evaluations for {placebo_triggered_trials} triggered trials at {PLACEBO_DRAWS} "
            f"frozen draws each (expected {expected}); "
            f"prevented draws: {placebo_draws_prevented or 'none'}"
        )

    holdout_candidates = tuple(
        t.trial_label for t in final if t.adjudication.verdict == "PASS"
    )

    return Phase15FamilyResult(
        manifest_fingerprint=manifest.manifest_fingerprint(),
        validation_spec_fingerprint=spec.validation_fingerprint(),
        reliability_policy_fingerprint=policy.identity(),
        engine=engine.engine_tag,
        is_full_family=is_full,
        n_pipelines_run=len(pipeline_keys),
        bh_family_size=spec.multiple_testing.n_trials if is_full else len(final),
        bh_q=spec.multiple_testing.bh_q,
        n_headline=headline,
        n_ablation=ablation,
        n_refused=sum(1 for t in final if t.refused),
        n_pass=verdicts.count("PASS"),
        n_reject=verdicts.count("REJECT"),
        n_inconclusive=verdicts.count("INCONCLUSIVE"),
        verdict_counts=vc,
        dsr_family_effective_trial_count=spec.multiple_testing.dsr_effective_trial_count,
        pipelines=tuple(outputs[k.label] for k in pipeline_keys),
        trials=tuple(final),
        placebo_triggered_trials=placebo_triggered_trials,
        placebo_cpp_draw_evaluations=placebo_cpp_draw_evaluations,
        placebo_draws_prevented=placebo_draws_prevented,
        holdout_candidates=holdout_candidates,
        n_engine_calls=engine.n_calls,
        n_engine_runs_executed=engine.n_engine_runs,
        n_engine_cache_hits=engine.n_cache_hits,
    )
