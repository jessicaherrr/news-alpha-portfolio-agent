"""News Alpha Phase H -- SCIENTIFIC VALIDATION of a portfolio (`portfolio-validation/1`).

    PortfolioPlan (Phase G)
      -> INPUT GATE      is this portfolio eligible for validation at all?
      -> EXECUTION GATE  can the engine execute and cost it faithfully?
      -> PortfolioStrategySpec (frozen: members, policy, rebalance rule)
      -> causal rebalance schedules      (portfolio.strategy)
      -> C++ replays                     (portfolio.execution, the unchanged engine)
      -> the EXISTING validation plane   (validation.assemble_report + the frozen
                                          Phase 13 ReliabilityPolicy, unchanged)
      -> PortfolioValidationReport

INPUT GATE. A plan existing is not eligibility. The eligibility authority is
the one the pipeline already has: a signal may continue into validation only
when its Phase E discovery screen says ``SCREEN_CONTINUE`` -- Phase G's
QUALIFIED policy. The gate re-checks every member against its actual screen
(never the plan's label alone). An EXPLORATORY plan, a plan that was not
constructed, or a member without screen support stops here with a typed
reason, before a single validation-window value is read.

PROTOCOL (the frozen Phase 13.5C one wherever it applies):

* split: TRAIN 2018-2022 / VALIDATION 2023-2024 / LOCKED_HOLDOUT 2025
  (`phase_13_5c_matrix.build_split_plan`); the holdout is never loaded;
* headline out-of-sample: the VALIDATION window;
* walk-forward: the frozen 4 expanding folds inside TRAIN (fold consistency);
* costs: the domain's cost plan x1.0 / x1.5 / x2.0 (`execution.cost_plan_for`);
* gating null: the centered block bootstrap on the headline daily returns
  (the diagnostic schedule-shift null is not run for portfolio schedules);
* parameter plateau: a PREDECLARED rebalance-interval neighbourhood
  (10 / 40 trading days around the canonical 20), each its own trial;
* multiple testing: BH over the report's own trial family -- the canonical
  trial and its predeclared rebalance neighbours -- exactly the Validation
  Engine's canonical per-report family; DSR over the same family. Pooling
  portfolios across research generations stays the registry / research-
  generation semantics' job, never a rule of this module;
* costs: every root's own commission (``execution.cost_plan_for``), charged by
  the C++ engine;
* regime: causal volatility tertiles of the portfolio universe (cut points
  from TRAIN days only) -- reported, not gating (as in 13.5C);
* verdict: `evaluate_policy` of the frozen policy, with parameter stability
  REQUIRED (a missing plateau is missing evidence, never a pass).

SCOPE. A result applies to the tested portfolio only. It never marks a member
signal validated, a signal path proven, or a mechanism established -- the
report says so, and Alpha Memory records it that way. A PASS here is a
validation-window result; the locked holdout stays unreleased
(`HOLDOUT_NOT_RELEASED`) until the existing holdout-release authority is used.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from alpha_agent.data.real_market_dataset import RESEARCH_WINDOW, VALIDATION_WINDOW, _iso_ns
from alpha_agent.news_alpha.mandate import ResearchMandate
from alpha_agent.portfolio.execution import (
    ExecutionInputs,
    PortfolioRun,
    cost_plan_for,
    execution_support,
    merged_day_plan,
    run_schedule,
)
from alpha_agent.portfolio.plan import PlanStatus, PortfolioPlan
from alpha_agent.portfolio.policy import EligibilityMode
from alpha_agent.portfolio.risk_model import SnapshotProvider, daily_returns
from alpha_agent.portfolio.selection import FactorSource, screen_eligibility
from alpha_agent.portfolio.strategy import (
    PortfolioStrategySpec,
    RebalanceRule,
    RebalanceSchedule,
    build_rebalance_schedule,
    freeze_portfolio_strategy,
)
from alpha_agent.recommendation.signal_ranking import RankedSignalSet
from alpha_agent.screening.candidate_signal_screen import CandidateScreen
from alpha_agent.screening.factor_diagnostics import ScreenStatus
from alpha_agent.validation.assemble import assemble_report
from alpha_agent.validation.cost_stress import CostScenarioResult, RootCommission
from alpha_agent.validation.dataset import DatasetIdentity, frame_content_hash
from alpha_agent.validation.enums import RegimeKind
from alpha_agent.validation.metrics import daily_sharpe
from alpha_agent.validation.multiple_testing import MultipleTestingFamily, TrialRecord
from alpha_agent.validation.nulls import centered_block_bootstrap_null_stats, empirical_p_value
from alpha_agent.validation.phase_13_5c_matrix import (
    build_split_plan,
    frozen_bootstrap_config,
    frozen_null_config,
    frozen_policy,
    frozen_walk_forward,
)
from alpha_agent.validation.policy import ReliabilityPolicy
from alpha_agent.validation.regime import RegimeLabelling, causal_volatility_regime_labels
from alpha_agent.validation.report import ValidationReport
from alpha_agent.validation.spec import ValidationSpec, validation_protocol_fingerprint
from alpha_agent.validation.stability import NeighbourResult, ParameterNeighbourhood
from alpha_agent.validation.walkforward import (
    WalkForwardFoldResult,
    build_folds,
    summarize_walk_forward,
)

__all__ = [
    "PORTFOLIO_STRATEGY_KEY",
    "PORTFOLIO_VALIDATION_METHOD",
    "HoldoutStage",
    "IneligibilityReason",
    "PortfolioEligibility",
    "PortfolioValidationReport",
    "PortfolioValidationStatus",
    "RunSummary",
    "assess_validation_eligibility",
    "gate_portfolio",
    "validate_portfolio",
]

PORTFOLIO_VALIDATION_METHOD = "portfolio-validation/1"
PORTFOLIO_STRATEGY_KEY = "news_alpha_portfolio"
#: Predeclared rebalance-interval neighbourhood around the canonical rule.
NEIGHBOUR_REBALANCE_DAYS: tuple[int, ...] = (10, 40)
_TRAIN = (_iso_ns(RESEARCH_WINDOW[0]), _iso_ns(RESEARCH_WINDOW[1]))
_VALIDATION = (_iso_ns(VALIDATION_WINDOW[0]), _iso_ns(VALIDATION_WINDOW[1]))


class PortfolioValidationStatus(str, Enum):
    #: The frozen policy's PASS on the 2023-2024 validation window -- for this
    #: portfolio only; the locked holdout is still unreleased.
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    #: The frozen policy's INCONCLUSIVE: too little evidence either way.
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NO_ELIGIBLE_PORTFOLIO = "NO_ELIGIBLE_PORTFOLIO"
    #: A traded root has no declared commission -- never charged another unit's rate.
    COST_MODEL_INCOMPLETE = "COST_MODEL_INCOMPLETE"
    EXECUTION_UNSUPPORTED = "EXECUTION_UNSUPPORTED"
    #: The same experiment already has a valid registry result -- cited, not re-run.
    PRIOR_RESULT_CITED = "PRIOR_RESULT_CITED"


class IneligibilityReason(str, Enum):
    PLAN_NOT_CONSTRUCTED = "PLAN_NOT_CONSTRUCTED"
    EXPLORATORY_PLAN = "EXPLORATORY_PLAN"
    MEMBER_WITHOUT_SCREEN_SUPPORT = "MEMBER_WITHOUT_SCREEN_SUPPORT"
    MEMBER_SCREEN_MISSING = "MEMBER_SCREEN_MISSING"
    NO_ELIGIBLE_MEMBER = "NO_ELIGIBLE_MEMBER"


INELIGIBILITY_TEXT: dict[IneligibilityReason, str] = {
    IneligibilityReason.PLAN_NOT_CONSTRUCTED: "The plan did not construct a portfolio.",
    IneligibilityReason.EXPLORATORY_PLAN: (
        "An exploratory preview is built from signals WITHOUT screening support -- it shows how the allocator "
        "treats them, and is never eligible to spend the validation window."),
    IneligibilityReason.MEMBER_WITHOUT_SCREEN_SUPPORT: "A member's discovery screen does not say continue.",
    IneligibilityReason.MEMBER_SCREEN_MISSING: "A member has no current discovery screen.",
    IneligibilityReason.NO_ELIGIBLE_MEMBER: "The plan's policy admits no signal of its ranked set.",
}


class IneligibilityFinding(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    reason: IneligibilityReason
    detail: str
    candidate_signal_ids: tuple[str, ...] = ()


class PortfolioEligibility(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    eligible: bool
    findings: tuple[IneligibilityFinding, ...]
    authority: str = (
        "Phase E discovery screen outcome SCREEN_CONTINUE, carried through the Phase F ranked set and the plan's "
        "QUALIFIED eligibility (Phase G); re-checked here against every member's actual screen."
    )

    @property
    def reasons(self) -> tuple[IneligibilityReason, ...]:
        return tuple(f.reason for f in self.findings)


def assess_validation_eligibility(
    plan: PortfolioPlan, ranked: RankedSignalSet, screens: Mapping[str, CandidateScreen],
) -> PortfolioEligibility:
    """Every reason the plan cannot enter validation (all of them, not the first)."""
    findings: list[IneligibilityFinding] = []
    if plan.status is not PlanStatus.CONSTRUCTED:
        findings.append(IneligibilityFinding(
            reason=IneligibilityReason.PLAN_NOT_CONSTRUCTED,
            detail=f"The plan was not constructed ({plan.status.value}): {plan.status_detail}"))
    if plan.eligibility is EligibilityMode.EXPLORATORY:
        findings.append(IneligibilityFinding(reason=IneligibilityReason.EXPLORATORY_PLAN,
                                             detail=INELIGIBILITY_TEXT[IneligibilityReason.EXPLORATORY_PLAN]))
    if plan.ranked_set_fingerprint != ranked.fingerprint():
        raise ValueError("the plan was built from a different ranked set")
    eligible, _ = screen_eligibility(ranked, plan.policy)
    if not eligible:
        findings.append(IneligibilityFinding(reason=IneligibilityReason.NO_ELIGIBLE_MEMBER,
                                             detail=INELIGIBILITY_TEXT[IneligibilityReason.NO_ELIGIBLE_MEMBER]))
    missing = tuple(s.candidate_signal_id for s in eligible if s.candidate_signal_id not in screens)
    unsupported = tuple(s.candidate_signal_id for s in eligible if s.candidate_signal_id in screens
                        and screens[s.candidate_signal_id].diagnostics.status is not ScreenStatus.SCREEN_CONTINUE)
    if missing:
        findings.append(IneligibilityFinding(
            reason=IneligibilityReason.MEMBER_SCREEN_MISSING, candidate_signal_ids=missing,
            detail=f"{len(missing)} member(s) have no current discovery screen."))
    if unsupported:
        findings.append(IneligibilityFinding(
            reason=IneligibilityReason.MEMBER_WITHOUT_SCREEN_SUPPORT, candidate_signal_ids=unsupported,
            detail=f"{len(unsupported)} of {len(eligible)} member(s) lack screening support (not SCREEN_CONTINUE)."))
    return PortfolioEligibility(eligible=not findings, findings=tuple(findings))


class HoldoutStage(BaseModel):
    """Phase H never opens the locked holdout; this records that it did not."""

    model_config = {"frozen": True, "extra": "forbid"}

    status: Literal["HOLDOUT_NOT_RELEASED"] = "HOLDOUT_NOT_RELEASED"
    lifecycle_state: str
    accessed: bool = False
    note: str = (
        "The 2025 locked holdout was not loaded, queried or evaluated. A final evaluation goes only through the "
        "existing holdout-release authority (a frozen research bundle + the holdout lifecycle's explicit "
        "authorization), never through Phase H."
    )


def _holdout_stage() -> HoldoutStage:
    from alpha_agent.holdout.lifecycle import HoldoutLifecycle

    record = HoldoutLifecycle().get("2025")
    return HoldoutStage(lifecycle_state=record.state.value, accessed=bool(record.accessed))


class RunSummary(BaseModel):
    """One C++ replay, reduced to what a reader audits (the full daily series
    lives in the ValidationReport for the headline run)."""

    model_config = {"frozen": True, "extra": "forbid"}

    label: str
    span_start: date
    span_end_exclusive: date
    rebalances: int
    rebalance_status_counts: dict[str, int]
    schedule_hash: str
    #: root -> USD per unit, as the engine charged it
    commission_schedule: dict[str, float]
    n_trading_days: int
    n_fills: int
    n_trades: int
    gross_pnl_usd: float
    costs_usd: float
    net_pnl_usd: float
    daily_sharpe: float
    risk_rejects: int
    risk_resizes: int
    rolls: int

    @classmethod
    def of(cls, run: PortfolioRun, schedule: RebalanceSchedule) -> RunSummary:
        r = run.run
        return cls(
            label=run.label, span_start=schedule.span_start, span_end_exclusive=schedule.span_end_exclusive,
            rebalances=len(schedule.decisions), rebalance_status_counts=schedule.status_counts(),
            schedule_hash=run.schedule_hash, commission_schedule=run.commission_schedule,
            n_trading_days=r.daily.n_days, n_fills=r.n_fills, n_trades=r.n_trades, gross_pnl_usd=r.gross_pnl_usd,
            costs_usd=r.costs_usd, net_pnl_usd=r.net_pnl_usd, daily_sharpe=r.daily_sharpe,
            risk_rejects=run.risk_rejects, risk_resizes=run.risk_resizes, rolls=run.rolls,
        )


class PortfolioValidationReport(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "portfolio-validation-report/1"
    method: Literal["portfolio-validation/1"] = PORTFOLIO_VALIDATION_METHOD
    plane: Literal["VALIDATION"] = "VALIDATION"
    status: PortfolioValidationStatus
    status_detail: str
    source_plan_fingerprint: str
    eligibility: PortfolioEligibility
    execution_support: str | None = None
    holdout: HoldoutStage
    strategy: PortfolioStrategySpec | None = None
    strategy_fingerprint: str | None = None
    validation_spec: ValidationSpec | None = None
    validation_spec_fingerprint: str | None = None
    validation_protocol_fingerprint: str | None = None
    experiment_identity: str | None = None
    report: ValidationReport | None = None
    headline_run: RunSummary | None = None
    fold_runs: tuple[RunSummary, ...] = ()
    cost_runs: tuple[RunSummary, ...] = ()
    neighbour_runs: tuple[RunSummary, ...] = ()
    #: the effective per-root commission, with each rate's provenance
    commission_schedule: tuple[RootCommission, ...] = ()
    cited_verdict: str | None = None
    data_read: tuple[str, ...] = ()
    data_provenance: tuple[str, ...] = ()
    scope_note: str = (
        "Applies to this portfolio only. It does not validate any member signal, prove any signal path, or "
        "establish any economic mechanism; the locked 2025 holdout was not used."
    )

    def headline(self) -> str:
        s = self.status
        if s is PortfolioValidationStatus.NO_ELIGIBLE_PORTFOLIO:
            return f"No eligible portfolio -- {self.status_detail}"
        if s is PortfolioValidationStatus.COST_MODEL_INCOMPLETE:
            return f"Cost model incomplete -- {self.status_detail}"
        if s is PortfolioValidationStatus.EXECUTION_UNSUPPORTED:
            return f"Not executed -- {self.status_detail}"
        if s is PortfolioValidationStatus.PRIOR_RESULT_CITED:
            return f"Already validated ({self.cited_verdict}) -- the registry result is cited, not re-run."
        assert self.report is not None and self.headline_run is not None
        h = self.headline_run
        return (f"{s.value.replace('_', ' ').title()} on the 2023-2024 validation window: net "
                f"${h.net_pnl_usd:,.0f} over {h.n_trading_days} days ({h.n_fills} fills, costs ${h.costs_usd:,.0f}); "
                f"policy verdict {self.report.verdict.value} ({', '.join(c.value for c in self.report.reason_codes)}).")


def _refused(plan: PortfolioPlan, status: PortfolioValidationStatus, detail: str, eligibility: PortfolioEligibility,
             **kw) -> PortfolioValidationReport:
    return PortfolioValidationReport(status=status, status_detail=detail, source_plan_fingerprint=plan.fingerprint(),
                                     eligibility=eligibility, holdout=_holdout_stage(), **kw)


def _ineligible_detail(e: PortfolioEligibility) -> str:
    """The primary finding (every finding stays in ``eligibility.findings``)."""
    first, rest = e.findings[0], len(e.findings) - 1
    more = f" (+{rest} further reason{'s' if rest > 1 else ''})" if rest else ""
    return first.detail + more


def _day(ts_ns: int) -> date:
    return pd.Timestamp(ts_ns, unit="ns", tz="UTC").date()


def _cbb_p(returns: np.ndarray, spec: ValidationSpec) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 3:
        return 1.0
    p, _ = empirical_p_value(float(daily_sharpe(r)), centered_block_bootstrap_null_stats(r, spec.null_test,
                                                                                          daily_sharpe))
    return p


def _dataset_identity(strategy: PortfolioStrategySpec, inputs: ExecutionInputs) -> DatasetIdentity:
    plan = merged_day_plan(inputs)
    return DatasetIdentity(
        root_symbol="+".join(sorted(m.instrument for m in {m.instrument_key: m for m in strategy.members}.values())),
        price_domain="raw", adjustment_mode=None, source_fingerprint=inputs.fingerprint(),
        bars_content_hash=frame_content_hash(inputs.bars), n_bars=len(inputs.bars),
        first_ts_ns=int(inputs.bars["ts_event_ns"].min()), last_ts_ns=int(inputs.bars["ts_event_ns"].max()),
        contracts_content_hash=frame_content_hash(inputs.contracts.astype(str)),
        trading_day_convention=plan.convention.model_dump(mode="json"),
        note="News Alpha Phase H portfolio engine inputs (raw execution bars of every member instrument)",
    )


def _regime_labelling(strategy: PortfolioStrategySpec, snapshots: SnapshotProvider,
                      oos_days: Sequence[str]) -> RegimeLabelling | None:
    """Causal volatility tertiles of the portfolio universe's mean absolute
    daily return; cut points from TRAIN days only."""
    series = []
    for key in strategy.instrument_keys:
        m = strategy.instrument(key)
        series.append(daily_returns(snapshots(m.domain, m.instrument)).abs().rename(key))
    frame = pd.concat(series, axis=1, join="inner").dropna().sort_index()
    if frame.empty:
        return None
    move = frame.mean(axis=1)
    days = move.index.astype(str).to_numpy()
    n_train = int((days < VALIDATION_WINDOW[0]).sum())
    lab = causal_volatility_regime_labels(move.to_numpy(float), train_n_days=n_train, lookback=20)
    by_day = dict(zip(days, lab.labels, strict=True))
    return RegimeLabelling(kind=RegimeKind.VOLATILITY, labels=tuple(by_day.get(d, "unknown") for d in oos_days),
                           definition={**lab.definition, "series": "mean |daily return| of the portfolio universe"})


def gate_portfolio(plan: PortfolioPlan, ranked: RankedSignalSet, screens: Mapping[str, CandidateScreen], *,
                   rebalance: RebalanceRule | None = None,
                   commission_conventions: dict | None = None) -> PortfolioValidationReport | None:
    """Both gates, and nothing else: the typed refusal of a portfolio that
    may not (or cannot) be validated, or ``None`` when it may. Reads no
    market data -- safe on every render."""
    eligibility = assess_validation_eligibility(plan, ranked, screens)
    if not eligibility.eligible:
        return _refused(plan, PortfolioValidationStatus.NO_ELIGIBLE_PORTFOLIO, _ineligible_detail(eligibility),
                        eligibility)
    strategy = freeze_portfolio_strategy(plan, ranked, rebalance=rebalance)
    refusal = execution_support(strategy, commission_conventions)
    if refusal is not None:
        return _refused(plan, PortfolioValidationStatus(refusal.kind), refusal.detail, eligibility,
                        execution_support=refusal.detail, strategy=strategy,
                        strategy_fingerprint=strategy.fingerprint())
    return None


def validate_portfolio(
    plan: PortfolioPlan,
    ranked: RankedSignalSet,
    screens: Mapping[str, CandidateScreen],
    mandate: ResearchMandate,
    *,
    load_factor_sources: Callable[[PortfolioStrategySpec], Mapping[str, FactorSource]],
    snapshots: SnapshotProvider,
    load_inputs: Callable[[PortfolioStrategySpec], ExecutionInputs],
    workdir: Path,
    registry=None,
    policy: ReliabilityPolicy | None = None,
    rebalance: RebalanceRule | None = None,
    commission_conventions: dict | None = None,
    allocator_cli=None,
    replay_cli: Path | None = None,
) -> PortfolioValidationReport:
    """``portfolio-validation/1`` for the portfolio ``plan`` defines. The
    loaders are called only after both gates pass: an ineligible or
    unexecutable portfolio never reads a validation-window value."""
    refused = gate_portfolio(plan, ranked, screens, rebalance=rebalance,
                             commission_conventions=commission_conventions)
    if refused is not None:
        return refused
    eligibility = assess_validation_eligibility(plan, ranked, screens)
    strategy = freeze_portfolio_strategy(plan, ranked, rebalance=rebalance)

    policy = policy or frozen_policy()
    costs = cost_plan_for(strategy, commission_conventions)
    inputs = load_inputs(strategy)
    dataset = _dataset_identity(strategy, inputs)
    split = build_split_plan()
    walk_forward = frozen_walk_forward()
    canonical_params = {"rebalance_every_trading_days": strategy.rebalance.every_trading_days}
    neighbourhood = ParameterNeighbourhood(
        strategy_key=PORTFOLIO_STRATEGY_KEY, canonical_params=canonical_params,
        neighbour_params=tuple({"rebalance_every_trading_days": n} for n in NEIGHBOUR_REBALANCE_DAYS),
    )
    protocol = validation_protocol_fingerprint(
        dataset=dataset, split_plan=split, walk_forward=walk_forward, cost_stress=costs,
        null_test=frozen_null_config(), bootstrap=frozen_bootstrap_config(),
        minimum_sample=policy.minimum_sample, reliability_policy_fingerprint=policy.identity(),
    )
    spec = ValidationSpec(
        label=f"news_alpha_phase_h::{strategy.fingerprint()[:24]}", strategy_fingerprint=strategy.fingerprint(),
        strategy_key=PORTFOLIO_STRATEGY_KEY, dataset=dataset, split_plan=split, walk_forward=walk_forward,
        capital_base_usd=strategy.constraints.capital_usd, cost_stress=costs, null_test=frozen_null_config(),
        bootstrap=frozen_bootstrap_config(), trial_family_id=f"{PORTFOLIO_STRATEGY_KEY}::{protocol}",
        parameter_neighbourhood=neighbourhood, minimum_sample=policy.minimum_sample,
        reliability_policy_fingerprint=policy.identity(),
    )

    from alpha_agent.portfolio.registry_record import portfolio_experiment_identity

    identity = portfolio_experiment_identity(strategy, spec, protocol, costs)
    if registry is not None:
        from alpha_agent.core.instrument import AssetDomain

        duplicate = registry.find_exact_duplicate(identity, asset_domain=AssetDomain(strategy.domains[0].value))
        if duplicate.blocks_reexecution:
            verdict = duplicate.headline_verdict.value if duplicate.headline_verdict else None
            return _refused(plan, PortfolioValidationStatus.PRIOR_RESULT_CITED,
                            f"experiment {duplicate.experiment_id} already holds a valid result ({verdict})",
                            eligibility, strategy=strategy, strategy_fingerprint=strategy.fingerprint(),
                            experiment_identity=identity, cited_verdict=verdict,
                            validation_protocol_fingerprint=protocol)

    sources = load_factor_sources(strategy)
    workdir = Path(workdir)
    baseline = next(sc for sc in costs.scenarios if sc.label == costs.baseline_label())

    def schedule_for(rule_strategy: PortfolioStrategySpec, lo_ns: int, hi_ns: int) -> RebalanceSchedule:
        return build_rebalance_schedule(rule_strategy, ranked, mandate, factor_sources=sources, snapshots=snapshots,
                                        start=_day(lo_ns), end_exclusive=_day(hi_ns), allocator_cli=allocator_cli)

    def run(rule_strategy, sched, lo_ns, hi_ns, label, scenario=baseline):
        _, slippage, spread = scenario.resolve(costs)
        return run_schedule(sched, rule_strategy, inputs, start_ns=lo_ns, end_ns=hi_ns,
                            commission_schedule=costs.schedule_for(scenario), slippage_ticks=slippage,
                            spread_ticks=spread, workdir=workdir, label=label, cli=replay_cli)

    # -- headline: the VALIDATION window -------------------------------------
    lo, hi = _VALIDATION
    headline_sched = schedule_for(strategy, lo, hi)
    headline = run(strategy, headline_sched, lo, hi, "headline_validation")

    # -- walk-forward folds inside TRAIN --------------------------------------
    folds = build_folds(_TRAIN, walk_forward)
    fold_results, fold_summaries = [], []
    for f in folds:
        sched = schedule_for(strategy, f.test_start_ts_ns, f.test_end_ts_ns)
        if not sched.decisions:
            continue
        frun = run(strategy, sched, f.test_start_ts_ns, f.test_end_ts_ns, f"fold_{f.fold_index}")
        r = frun.run
        fold_results.append(WalkForwardFoldResult(
            fold_index=f.fold_index, test_start_ts_ns=f.test_start_ts_ns, test_end_ts_ns=f.test_end_ts_ns,
            n_trading_days=r.daily.n_days, n_fills=r.n_fills, n_trades=r.n_trades, oos_net_pnl_usd=r.net_pnl_usd,
            daily_sharpe=r.daily_sharpe, is_positive=r.net_pnl_usd > 0.0, daily=r.daily))
        fold_summaries.append(RunSummary.of(frun, sched))
    wf = summarize_walk_forward(walk_forward, len(folds), fold_results)

    # -- cost stress (same schedule, scaled costs) ----------------------------
    cost_results, cost_summaries = [], []
    for sc in costs.scenarios:
        # the plan's scalar default (0) is unused -- the per-root schedule is the
        # commission; each run summary and the report carry its effective rates
        comm, slip, spr = sc.resolve(costs)
        crun = run(strategy, headline_sched, lo, hi, f"cost_{sc.label}", scenario=sc)
        base_net = headline.run.net_pnl_usd
        cost_results.append(CostScenarioResult(
            label=sc.label, commission_per_contract_usd=comm, slippage_ticks=slip, spread_ticks=spr,
            n_fills=crun.run.n_fills, oos_gross_pnl_usd=crun.run.gross_pnl_usd, oos_costs_usd=crun.run.costs_usd,
            oos_net_pnl_usd=crun.run.net_pnl_usd, daily_sharpe=crun.run.daily_sharpe,
            annualized_sharpe=crun.run.annualized_sharpe,
            net_pnl_ratio_vs_baseline=crun.run.net_pnl_usd / base_net if abs(base_net) > 1e-9 else float("nan"),
            sharpe_delta_vs_baseline=crun.run.daily_sharpe - headline.run.daily_sharpe))
        cost_summaries.append(RunSummary.of(crun, headline_sched))

    # -- predeclared rebalance neighbourhood + the trial family ---------------
    trials = [TrialRecord(label="canonical", role="canonical", strategy_fingerprint=strategy.fingerprint(),
                          schedule_hash=headline_sched.schedule_hash(),
                          p_value=_cbb_p(headline.run.daily.returns_array(), spec), null_tested=True, observed_daily_sharpe=headline.run.daily_sharpe,
                          observed_net_pnl_usd=headline.run.net_pnl_usd)]
    neighbour_results = [NeighbourResult(
        params_label="canonical", strategy_fingerprint=strategy.fingerprint(), is_canonical=True,
        n_trades=headline.run.n_trades, oos_net_pnl_usd=headline.run.net_pnl_usd,
        oos_daily_sharpe=headline.run.daily_sharpe, oos_annualized_sharpe=headline.run.annualized_sharpe)]
    neighbour_summaries = []
    for i, params in enumerate(neighbourhood.neighbour_params):
        variant = strategy.with_rebalance(RebalanceRule(every_trading_days=params["rebalance_every_trading_days"]))
        nsched = schedule_for(variant, lo, hi)
        nrun = run(variant, nsched, lo, hi, f"neighbour_{i}")
        neighbour_results.append(NeighbourResult(
            params_label=f"neighbour_{i}", strategy_fingerprint=variant.fingerprint(), is_canonical=False,
            n_trades=nrun.run.n_trades, oos_net_pnl_usd=nrun.run.net_pnl_usd, oos_daily_sharpe=nrun.run.daily_sharpe,
            oos_annualized_sharpe=nrun.run.annualized_sharpe))
        trials.append(TrialRecord(
            label=f"neighbour_{i}", role="neighbour", strategy_fingerprint=variant.fingerprint(),
            schedule_hash=nsched.schedule_hash(), p_value=_cbb_p(nrun.run.daily.returns_array(), spec),
            null_tested=True, observed_daily_sharpe=nrun.run.daily_sharpe, observed_net_pnl_usd=nrun.run.net_pnl_usd))
        neighbour_summaries.append(RunSummary.of(nrun, nsched))
    family = MultipleTestingFamily(family_id=spec.trial_family_id, trials=tuple(trials))

    oos_days = list(headline.run.daily.trading_day)
    report = assemble_report(
        spec, policy, walk_forward=wf, combined_oos_daily=headline.run.daily, oos_n_fills=headline.run.n_fills,
        oos_n_trades=headline.run.n_trades, oos_gross_pnl_usd=headline.run.gross_pnl_usd,
        oos_costs_usd=headline.run.costs_usd, oos_net_pnl_usd=headline.run.net_pnl_usd, trial_family=family,
        cost_scenario_results=cost_results, neighbour_results=neighbour_results,
        regime_labelling=_regime_labelling(strategy, snapshots, oos_days),
        target_schedule_hash=headline_sched.schedule_hash(), holdout_evaluated=False,
        explanation=("News Alpha Phase H portfolio validation: the frozen portfolio strategy re-sized causally at "
                     "each rebalance day; every economic number from the C++ engine."),
        parameter_stability_required=True,
    )
    status = {"PASS": PortfolioValidationStatus.VALIDATED, "REJECT": PortfolioValidationStatus.REJECTED,
              "INCONCLUSIVE": PortfolioValidationStatus.INSUFFICIENT_EVIDENCE}[report.verdict.value]
    return PortfolioValidationReport(
        status=status, status_detail=", ".join(c.value for c in report.reason_codes),
        source_plan_fingerprint=plan.fingerprint(), eligibility=eligibility, holdout=_holdout_stage(),
        strategy=strategy, strategy_fingerprint=strategy.fingerprint(), validation_spec=spec,
        validation_spec_fingerprint=spec.validation_fingerprint(), validation_protocol_fingerprint=protocol,
        experiment_identity=identity, report=report, headline_run=RunSummary.of(headline, headline_sched),
        fold_runs=tuple(fold_summaries), cost_runs=tuple(cost_summaries), neighbour_runs=tuple(neighbour_summaries),
        commission_schedule=costs.commission_schedule or (),
        data_read=("2018-2022 TRAIN (walk-forward folds)", "2023-2024 VALIDATION (headline, costs, neighbours)"),
        data_provenance=inputs.provenance,
    )

