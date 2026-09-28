"""Phase 6 closure, instruction 6: persist a real completed ETF research
result through the SAME authoritative experiment / attempt / result
construction the runtime Research Orchestrator already uses for Futures
(``alpha_agent.agents.orchestrator``) -- no new identity path, no parallel
registry-write mechanism.

Mirrors the "standalone golden-path append" precedent already in the real
registry (``phase_research-golden-path-v1-acceptance_family``,
``phase_agent-runtime_family``): one real, already-executed hypothesis,
appended as its own single-member ``ImportBundle`` via
``ExperimentRegistry.apply_bundle`` -- the exact mechanism
``ResearchOrchestrator._experiment_record`` / ``_result_record`` /
``finalize_family`` already use, just invoked directly for one already-
computed :class:`~alpha_agent.etf.research.EtfResearchArtifact` instead of a
live multi-member family run.

``target_schedule_hash`` / ``report_fingerprint`` / ``candidate_manifest_fingerprint``
are honestly ``None`` -- no committed schedule/report/manifest artifact file
exists on disk with a hash for this specific run (CLAUDE.md: "NEVER FABRICATE
PROVENANCE"). ``asset_domain=AssetDomain.ETF`` throughout -- this experiment
can never be queried into the same family as any Futures "tsmom" row (the
strategy_family label ``etf_tsmom`` also differs, but the domain tag is the
structural guarantee, not the label).
"""
from __future__ import annotations

from datetime import UTC, datetime

from alpha_agent.agents.orchestrator import feature_set_fingerprint
from alpha_agent.core.instrument import AssetDomain
from alpha_agent.etf.research import EtfResearchArtifact
from alpha_agent.registry.enums import (
    AttemptStatus,
    ExperimentStatus,
    FailureClass,
    FailureScope,
    RegistryVerdict,
    TrialRole,
)
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
    FailureRecord,
    ImportBundle,
    MarketWindow,
    ResultRecord,
)
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.validation.spec import validation_protocol_fingerprint

PHASE = "6"
#: The frozen ReliabilityPolicy's real BH q-threshold (matches every other
#: real registry family's own recorded metadata, e.g. the Phase 13.5C /
#: agent-runtime / golden-path imports -- not re-derived here since this
#: function only has the policy's fingerprint, not the object itself).
_BH_Q_THRESHOLD = 0.10

#: This trial's own execution-plane facts, in the exact shape
#: identity.execution_config_identity expects -- NOT reused from Futures'
#: candidates_phase_13_5c.EXECUTION_ASSUMPTIONS, which describes a genuinely
#: different execution reality (native 1-minute RAW_CONTRACT bars, not daily
#: RAW ETF bars).
_EXECUTION_ASSUMPTIONS = {
    "bars": "daily RAW (unadjusted) primary-listing-venue OHLCV; C++ quant_backtest_targets_csv",
    "commission_per_contract_usd": 2.0,
    "slippage_ticks": 0.0,
    "spread_ticks": 0.0,
    "latency_bars": 0,
    "fill_timing": "decision at bar T -> fill at the next eligible daily bar (T+1)",
    "official_pnl_source": "cpp_portfolio_accountant_daily_equity_trace",
}
_ECONOMICS_RULE = (
    "ETF instrument economics: point_value_usd = 1.0 (1 share = $1 P&L per $1 price move, "
    "definitionally true for a single-share security, never derived from a Databento "
    "contract_multiplier field or guessed); tick_size = 0.01 (standard US equity minimum "
    "increment). See alpha_agent.etf.data_source.etf_contract_spec."
)
#: Same real risk manager / CLI as Futures (this session's ETF run used the
#: identical compiled CLI, PassThroughRiskManager), phrased for this domain
#: rather than reusing Futures' phase-specific wording verbatim.
_RISK_ASSUMPTIONS = {
    "risk_manager": "PassThroughRiskManager",
    "cpp_reference_cli": "quant_backtest_targets_csv",
    "hard_risk_limits_enforced": False,
    "limitation": (
        "Phase 08 MaxContractsRiskManager / hard risk limits are not wired into the "
        "reference CLI execution path for this ETF pilot, exactly as for Futures. Every "
        "target is passed through unchanged; the candidate size is a small integer "
        "target (1) by construction."
    ),
}


def _contract_economics(root_symbol: str) -> dict:
    return {root_symbol: {"point_value_usd": 1.0, "tick_size": 0.01, "tick_value_usd": 0.01}}


def _ts_to_date(ts_ns: int) -> str:
    import pandas as pd

    return pd.Timestamp(ts_ns, unit="ns", tz="UTC").date().isoformat()


def build_import_bundle(artifact: EtfResearchArtifact, *, code_commit: str = "") -> ImportBundle:
    """Deterministic -- calling this twice on the same artifact produces the
    same `experiment_identity` (proven by a dedicated test). Does not touch
    the registry; call `ExperimentRegistry.apply_bundle` with the result."""
    spec = artifact.validation_spec
    report = artifact.report
    root = artifact.root_symbol

    from alpha_agent.etf.strategy_family import etf_tsmom_spec
    from alpha_agent.strategy import strategy_fingerprint

    canonical_spec = etf_tsmom_spec(root)
    canon_fp = strategy_fingerprint(canonical_spec)
    if canon_fp != artifact.strategy_fingerprint:
        raise ValueError(
            "reconstructed strategy_fingerprint does not match the artifact's own -- "
            "refusing to persist a possibly-drifted identity"
        )
    feature_fp = feature_set_fingerprint(canonical_spec)

    # params EXCLUDING root_symbol -- root is already a separate identity
    # component (matches candidates_phase_13_5c's own convention).
    params = dict(artifact.params)
    params.pop("root_symbol", None)
    pvi = parameter_variant_identity(params)

    exec_cfg_id = execution_config_identity(
        execution_assumptions=_EXECUTION_ASSUMPTIONS,
        contract_economics=_contract_economics(root),
        economics_rule=_ECONOMICS_RULE,
    )
    cost_cfg_id = cost_config_identity(
        base_commission_per_contract_usd=2.0, base_slippage_ticks=0.0, base_spread_ticks=0.0,
        scenarios=[s.model_dump(mode="json") for s in spec.cost_stress.scenarios],
    )
    risk_id = risk_config_identity(_RISK_ASSUMPTIONS)
    protocol_fp = validation_protocol_fingerprint(
        dataset=spec.dataset, split_plan=spec.split_plan, walk_forward=spec.walk_forward,
        cost_stress=spec.cost_stress, null_test=spec.null_test, bootstrap=spec.bootstrap,
        minimum_sample=spec.minimum_sample, reliability_policy_fingerprint=spec.reliability_policy_fingerprint,
    )

    identity = experiment_identity(
        strategy_fingerprint=canon_fp, strategy_family=artifact.strategy_family, root_symbol=root,
        parameter_variant_identity=pvi, dataset_fingerprint=spec.dataset.identity(),
        split_identity=spec.split_plan.split_fingerprint(), validation_spec_fingerprint=protocol_fp,
        reliability_policy_fingerprint=spec.reliability_policy_fingerprint,
        execution_config_identity=exec_cfg_id, cost_config_identity=cost_cfg_id, risk_identity=risk_id,
        feature_spec_fingerprint=feature_fp,
    )

    from alpha_agent.validation.enums import SplitRole

    now = datetime.now(UTC).isoformat()
    validation_window = next((w for w in spec.split_plan.windows if w.role is SplitRole.VALIDATION), None)
    if validation_window is None:
        raise ValueError("validation_spec has no VALIDATION split window")
    market_window = MarketWindow(
        label="VALIDATION",
        start_date=_ts_to_date(validation_window.start_ts_ns),
        end_date=_ts_to_date(validation_window.end_ts_ns),
    )
    exp_id = friendly_experiment_id(
        root_symbol=root, strategy_family=artifact.strategy_family, variant_label="canonical",
        split_label=market_window.label, lineage_tag=identity.split(":")[-1][:10],
    )

    verdict = RegistryVerdict(report.verdict.value)
    idx = report.canonical_trial_index
    trial = report.fdr_result.decisions[idx]
    om = report.oos_metrics
    # ReasonCode is a str Enum -- .value gives the clean stored string
    # ("null_hypothesis_not_rejected"), never str(c) ("ReasonCode.NULL_NOT_REJECTED").
    reason_codes = tuple(c.value if hasattr(c, "value") else str(c) for c in report.reason_codes)

    experiment = ExperimentRecord(
        experiment_identity=identity,
        identity_schema=IDENTITY_SCHEMA,
        experiment_id=exp_id,
        display_name=f"XLE ETF tsmom(21,120) -- Phase 6 pilot [{root}]",
        created_at=now,
        phase=PHASE,
        status=ExperimentStatus.COMPLETED,
        code_commit=code_commit,
        root_symbol=root,
        asset_domain=AssetDomain.ETF,
        strategy_family=artifact.strategy_family,
        strategy_fingerprint=canon_fp,
        strategy_id=canonical_spec.strategy_id,
        strategy_spec_json={"params": params},
        feature_spec_fingerprint=feature_fp,
        target_schedule_hash=None,
        candidate_manifest_fingerprint=None,
        dataset_fingerprint=spec.dataset.identity(),
        split_identity=spec.split_plan.split_fingerprint(),
        market_window=market_window,
        validation_spec_fingerprint=protocol_fp,
        reliability_policy_fingerprint=spec.reliability_policy_fingerprint,
        execution_config_identity=exec_cfg_id,
        cost_config_identity=cost_cfg_id,
        risk_identity=risk_id,
        trial_role=TrialRole.CANONICAL,
        parameter_variant_identity=pvi,
        parameter_variant_label="canonical",
        parent_experiment_identity=None,
        report_fingerprint=None,
        notes=(
            "Phase 6 ETF Research Pilot: standalone real end-to-end tsmom hypothesis on "
            f"{root}, run through the unmodified ValidationEngine + the real compiled C++ "
            "CLI over real acquired primary-listing-venue data. No corporate-action "
            f"adjustment applied ({root} screened clean for splits; see "
            "alpha_agent.etf.corporate_actions.coverage())."
        ),
    )
    attempt = ExecutionAttemptRecord(
        experiment_identity=identity,
        identity_schema=IDENTITY_SCHEMA,
        attempt_ordinal=None,
        attempt_status=AttemptStatus.VALID,
        code_commit=code_commit,
        engine="cpp_quant_core__quant_backtest_targets_csv",
        target_schedule_hash=None,
        report_fingerprint=None,
        source_artifact=None,
        source_artifact_sha256=None,
        created_at=now,
        notes=f"Phase 6 ETF pilot standalone append; verdict {verdict.value}",
    )
    result = ResultRecord(
        experiment_identity=identity,
        headline_verdict=verdict,
        reason_codes=reason_codes,
        gross_pnl_usd=float(om.oos_gross_pnl_usd),
        costs_usd=float(om.oos_costs_usd),
        net_pnl_usd=float(om.oos_net_pnl_usd),
        daily_sharpe=float(om.daily_sharpe),
        annualized_sharpe=float(om.annualized_sharpe),
        gating_null_p=float(trial.p_value),
        bh_p_value=float(trial.p_value),
        bh_q=float(trial.q_value),
        bh_rejected_at_q=bool(trial.rejected),
        dsr_probability=(float(report.dsr_result.deflated_sharpe_ratio) if report.dsr_result.is_valid else None),
        fold_consistency=float(report.walk_forward.fold_consistency),
        n_trades=int(om.n_trades),
        n_fills=int(om.n_fills),
        n_oos_days=int(om.n_trading_days),
        holdout_eligible=(verdict is RegistryVerdict.PASS),
        evidence_completeness="full",
        source_artifact=None,
        source_artifact_sha256=None,
    )

    failures: tuple[FailureRecord, ...] = ()
    if verdict in (RegistryVerdict.REJECT, RegistryVerdict.INCONCLUSIVE):
        failure_class = (
            FailureClass.SCIENTIFIC_REJECTION if verdict is RegistryVerdict.REJECT
            else FailureClass.STATISTICAL_INCONCLUSIVE
        )
        failures = (
            FailureRecord(
                failure_id=f"phase_6_etf_{verdict.value.lower()}__{identity.split(':')[-1][:12]}",
                scope=FailureScope.EXPERIMENT,
                experiment_identity=identity,
                failure_class=failure_class,
                failure_code="|".join(reason_codes) or verdict.value,
                summary=f"Phase 6 ETF pilot, {root} etf_tsmom canonical: {verdict.value}",
                mechanism="; ".join(reason_codes) or "frozen per-trial gates",
                evidence={"reason_codes": list(reason_codes), "bh_q_value": float(trial.q_value)},
                action_taken="recorded as first-class evidence; never deleted",
                created_at=now,
                root_symbol=root,
                strategy_family=artifact.strategy_family,
            ),
        )

    return ImportBundle(
        import_id=f"phase_6_etf__{root.lower()}__{identity.split(':')[-1][:12]}",
        phase=PHASE,
        source_fingerprint=protocol_fp,
        created_at=now,
        experiments=(experiment,),
        execution_attempts=(attempt,),
        results=(result,),
        failures=failures,
        metadata={
            "family_id": f"phase6.etf.{root}.{artifact.strategy_family}",
            "bh_family_size": 1,
            "bh_q_threshold": _BH_Q_THRESHOLD,
            "n_bh_rejected": int(trial.rejected),
            "policy_identity": spec.reliability_policy_fingerprint,
            "asset_domain": AssetDomain.ETF.value,
        },
    )


def append_xle_result(artifact: EtfResearchArtifact, registry: ExperimentRegistry, *, code_commit: str = "") -> str:
    """Builds and applies the bundle; returns the real `experiment_identity`
    that was written (or already existed, idempotently)."""
    bundle = build_import_bundle(artifact, code_commit=code_commit)
    registry.apply_bundle(bundle)
    return bundle.experiments[0].experiment_identity
