"""News Alpha Phase H -- a validated portfolio as REGISTRY EXPERIMENTS.

The same authoritative construction every other result uses (the ETF pilot's
`etf.registry_append`, the orchestrator): one `ExperimentRecord` per trial, a
VALID `ExecutionAttemptRecord`, a `ResultRecord`, applied as one
`ImportBundle`. Nothing new about identity: `registry.identity.experiment_identity`
over the portfolio strategy fingerprint + family + a composite root label +
the rebalance variant + dataset + split + validation protocol + policy +
execution / cost / risk configuration + the member-expression set.

* ``strategy_family`` = ``news_alpha_portfolio``; ``root_symbol`` = the member
  instruments joined by ``+`` (a portfolio has no single root; the label is
  deterministic and human-readable, and a different book is a different root).
* ``asset_domain`` = the portfolio's one domain -- execution refuses a
  mixed-domain book before any run, so no cross-domain experiment can exist.
* the canonical rebalance rule is CANONICAL (headline adjudicated); each
  predeclared rebalance neighbour is a NEIGHBOUR trial (NOT_ADJUDICATED, its
  own p/q in the BH family).
* ``target_schedule_hash`` is the compiled rebalance schedule's hash (a
  committed value of this run), ``report_fingerprint`` the ValidationReport's.

Only a PASS / REJECT / INCONCLUSIVE report is recordable; a gate refusal is
not an experiment (it is recorded as signal-path evidence instead).
"""
from __future__ import annotations

from datetime import UTC, datetime

from alpha_agent.core.instrument import AssetDomain
from alpha_agent.portfolio.strategy import PortfolioStrategySpec
from alpha_agent.registry.enums import AttemptStatus, ExperimentStatus, RegistryVerdict, TrialRole
from alpha_agent.registry.identity import (
    IDENTITY_SCHEMA,
    cost_config_identity,
    execution_config_identity,
    experiment_identity,
    friendly_experiment_id,
    parameter_variant_identity,
    risk_config_identity,
)
from alpha_agent.registry.models import (
    ExecutionAttemptRecord,
    ExperimentRecord,
    ImportBundle,
    MarketWindow,
    ResultRecord,
)
from alpha_agent.validation.cost_stress import CostStressPlan
from alpha_agent.validation.fingerprint import fingerprint
from alpha_agent.validation.spec import ValidationSpec

__all__ = [
    "PHASE",
    "build_portfolio_import_bundle",
    "portfolio_experiment_identity",
    "portfolio_root_label",
]

PHASE = "news_alpha_h"
FAMILY = "news_alpha_portfolio"
ENGINE = "cpp_quant_core__quant_portfolio_plan_replay_csv"
_EXECUTION_ASSUMPTIONS = {
    "engine": "quant_portfolio_plan_replay_csv: BacktestEngine + PortfolioRiskManager (hard gate)",
    "bars": ("futures: native 1-minute GLBX.MDP3 raw-contract bars with roll close marks; ETF: primary-listing "
             "daily raw bars, one engine instrument per ticker, sourced splits applied by the engine"),
    "fill_timing": "decision at the rebalance day's decision bar -> fill at the next bar of that root",
    "latency_bars": 0,
    "slippage_ticks": 0.0,
    "spread_ticks": 0.0,
    "end_of_test": "force_liquidate",
    "daily_basis": "merged trading-day plan: the latest root's last bar of each session day",
    "official_pnl_source": "cpp_portfolio_accountant_daily_equity_trace",
}
_ECONOMICS_RULE = ("futures: point value derived from the stored definitions (docs/CONTRACT_ECONOMICS.md); "
                   "ETF: one share, point value 1.0, tick 0.01 (etf.data_source)")


def portfolio_root_label(strategy: PortfolioStrategySpec) -> str:
    return "+".join(sorted({m.instrument for m in strategy.members}))


def _feature_fingerprint(strategy: PortfolioStrategySpec) -> str:
    return fingerprint("portfeatures1", sorted([m.candidate_signal_id, m.expression] for m in strategy.members))


def _risk_assumptions(strategy: PortfolioStrategySpec) -> dict:
    c = strategy.constraints
    return {
        "risk_manager": "PortfolioRiskManager",
        "capital_usd": c.capital_usd,
        "max_gross_leverage": c.max_gross_leverage,
        "max_net_leverage": c.max_net_exposure,
        "max_drawdown_pct": c.max_drawdown_pct,
        "stale_mark_policy": "reject_risk_increasing",
        "missing_margin_policy": "treat_as_zero",
        "unit_caps": "non-binding by construction (per-instrument maxima of the schedule)",
        "constraints_fingerprint": c.fingerprint(),
    }


def _identities(strategy: PortfolioStrategySpec, spec: ValidationSpec, costs: CostStressPlan) -> dict:
    return {
        # every member contract's economics are in the dataset's contracts hash
        "execution": execution_config_identity(
            execution_assumptions=_EXECUTION_ASSUMPTIONS,
            contract_economics={"contracts_content_hash": spec.dataset.contracts_content_hash},
            economics_rule=_ECONOMICS_RULE),
        "cost": cost_config_identity(
            base_commission_per_contract_usd=costs.base_commission_per_contract_usd,
            base_slippage_ticks=costs.base_slippage_ticks, base_spread_ticks=costs.base_spread_ticks,
            scenarios=[s.model_dump(mode="json") for s in costs.scenarios],
            commission_schedule=([c.model_dump(mode="json") for c in costs.commission_schedule]
                                 if costs.commission_schedule is not None else None)),
        "risk": risk_config_identity(_risk_assumptions(strategy)),
    }


def portfolio_experiment_identity(strategy: PortfolioStrategySpec, spec: ValidationSpec, protocol: str,
                                  costs: CostStressPlan) -> str:
    """A trial's pre-run identity (no backtest needed): ``strategy`` carries
    the rebalance variant."""
    ids = _identities(strategy, spec, costs)
    return experiment_identity(
        strategy_fingerprint=strategy.fingerprint(), strategy_family=FAMILY, root_symbol=portfolio_root_label(strategy),
        parameter_variant_identity=parameter_variant_identity(
            {"rebalance_every_trading_days": strategy.rebalance.every_trading_days}),
        dataset_fingerprint=spec.dataset.identity(), split_identity=spec.split_plan.split_fingerprint(),
        validation_spec_fingerprint=protocol, reliability_policy_fingerprint=spec.reliability_policy_fingerprint,
        execution_config_identity=ids["execution"], cost_config_identity=ids["cost"], risk_identity=ids["risk"],
        feature_spec_fingerprint=_feature_fingerprint(strategy),
    )


def build_portfolio_import_bundle(report, *, code_commit: str = "", source_artifact: str | None = None,
                                  source_artifact_sha256: str | None = None) -> ImportBundle:
    """One bundle: the canonical trial + every predeclared neighbour, from a
    `PortfolioValidationReport`. Does not touch the registry -- apply it with
    `ExperimentRegistry.apply_bundle`."""
    from alpha_agent.portfolio.strategy import RebalanceRule
    from alpha_agent.portfolio.validation import PortfolioValidationStatus

    if report.status not in (PortfolioValidationStatus.VALIDATED, PortfolioValidationStatus.REJECTED,
                             PortfolioValidationStatus.INSUFFICIENT_EVIDENCE):
        raise ValueError(f"a {report.status.value} outcome is not an executed experiment")
    strategy: PortfolioStrategySpec = report.strategy
    spec: ValidationSpec = report.validation_spec
    costs: CostStressPlan = spec.cost_stress
    vr = report.report
    ids = _identities(strategy, spec, costs)
    now = datetime.now(UTC).isoformat()
    window = MarketWindow(label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31")
    domain = AssetDomain(strategy.domains[0].value)
    root = portfolio_root_label(strategy)
    canonical_id = portfolio_experiment_identity(strategy, spec, report.validation_protocol_fingerprint, costs)
    if report.experiment_identity not in (None, canonical_id):
        raise ValueError("the report's experiment identity does not match its strategy and spec")
    experiments, attempts, results = [], [], []
    runs = {"canonical": report.headline_run, **{f"neighbour_{i}": r for i, r in enumerate(report.neighbour_runs)}}
    for i, trial in enumerate(vr.fdr_result.decisions):
        label = trial.label
        if label not in runs:
            continue  # a prior portfolio trial: already in the registry
        run = runs[label]
        if label == "canonical":
            variant, role, verdict, identity = strategy, TrialRole.CANONICAL, RegistryVerdict(vr.verdict.value), \
                canonical_id
            reasons = tuple(c.value for c in vr.reason_codes)
        else:
            every = spec.parameter_neighbourhood.neighbour_params[int(label.split("_")[1])][
                "rebalance_every_trading_days"]
            variant = strategy.with_rebalance(RebalanceRule(every_trading_days=every))
            role, verdict, reasons = TrialRole.NEIGHBOUR, RegistryVerdict.NOT_ADJUDICATED, ()
            identity = portfolio_experiment_identity(variant, spec, report.validation_protocol_fingerprint, costs)
        params = {"rebalance_every_trading_days": variant.rebalance.every_trading_days}
        experiments.append(ExperimentRecord(
            experiment_identity=identity, identity_schema=IDENTITY_SCHEMA,
            experiment_id=friendly_experiment_id(root_symbol=root, strategy_family=FAMILY, variant_label=label,
                                                 split_label="VALIDATION", lineage_tag=identity.split(":")[-1][:10]),
            display_name=f"News-alpha portfolio {root} (rebalance {params['rebalance_every_trading_days']}d) -- {label}",
            created_at=now, phase=PHASE, status=ExperimentStatus.COMPLETED, code_commit=code_commit,
            root_symbol=root, asset_domain=domain, strategy_family=FAMILY, strategy_fingerprint=variant.fingerprint(),
            strategy_id=variant.fingerprint()[:40], strategy_spec_json={
                "params": params, "members": [m.candidate_signal_id for m in variant.members],
                "source_plan_fingerprint": variant.source_plan_fingerprint,
                "event_ids": sorted({e for m in variant.members for e in m.event_ids}),
                "scope": variant.scope_note},
            feature_spec_fingerprint=_feature_fingerprint(variant),
            target_schedule_hash=run.schedule_hash, dataset_fingerprint=spec.dataset.identity(),
            split_identity=spec.split_plan.split_fingerprint(), market_window=window,
            validation_spec_fingerprint=report.validation_protocol_fingerprint,
            reliability_policy_fingerprint=spec.reliability_policy_fingerprint,
            execution_config_identity=ids["execution"], cost_config_identity=ids["cost"], risk_identity=ids["risk"],
            trial_role=role, parameter_variant_identity=parameter_variant_identity(params),
            parameter_variant_label=label,
            parent_experiment_identity=None if role is TrialRole.CANONICAL else canonical_id,
            report_fingerprint=vr.report_fingerprint() if role is TrialRole.CANONICAL else None,
            notes="News Alpha Phase H: a frozen news-to-portfolio strategy validated through the C++ engine.",
        ))
        attempts.append(ExecutionAttemptRecord(
            experiment_identity=identity, attempt_status=AttemptStatus.VALID, code_commit=code_commit, engine=ENGINE,
            target_schedule_hash=run.schedule_hash,
            report_fingerprint=vr.report_fingerprint() if role is TrialRole.CANONICAL else None,
            source_artifact=source_artifact, source_artifact_sha256=source_artifact_sha256, created_at=now))
        om = vr.oos_metrics if role is TrialRole.CANONICAL else None
        results.append(ResultRecord(
            experiment_identity=identity, headline_verdict=verdict, reason_codes=reasons,
            gross_pnl_usd=run.gross_pnl_usd, costs_usd=run.costs_usd, net_pnl_usd=run.net_pnl_usd,
            daily_sharpe=run.daily_sharpe, annualized_sharpe=om.annualized_sharpe if om else None,
            gating_null_p=float(trial.p_value), bh_p_value=float(trial.p_value), bh_q=float(trial.q_value),
            bh_rejected_at_q=bool(trial.rejected),
            dsr_probability=(float(vr.dsr_result.deflated_sharpe_ratio)
                             if role is TrialRole.CANONICAL and vr.dsr_result.is_valid else None),
            fold_consistency=float(vr.walk_forward.fold_consistency) if role is TrialRole.CANONICAL else None,
            n_trades=run.n_trades, n_fills=run.n_fills, n_oos_days=run.n_trading_days,
            cost_stress=({r.label: r.net_pnl_usd for r in report.cost_runs} if role is TrialRole.CANONICAL else {}),
            holdout_eligible=verdict is RegistryVerdict.PASS,
            evidence_completeness="full" if role is TrialRole.CANONICAL else "partial:neighbour_trial",
            source_artifact=source_artifact, source_artifact_sha256=source_artifact_sha256,
        ))
    return ImportBundle(
        import_id=f"news_alpha_phase_h::{canonical_id}", phase=PHASE,
        source_fingerprint=fingerprint("portvalbundle1", {"report": vr.report_fingerprint(), "id": canonical_id}),
        created_at=now, experiments=tuple(experiments), execution_attempts=tuple(attempts), results=tuple(results),
        metadata={"bh_q_threshold": 0.10, "family": spec.trial_family_id,
                  "scope": "portfolio only -- member signals, paths and mechanisms are not validated by this"},
    )
