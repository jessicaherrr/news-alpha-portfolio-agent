"""Production `ExecutionValidationService` (Agent runtime-integration release,
sections 6/7).

`test_replay_matches_committed_nq_tsmom_canonical_result` is the required
equivalence proof (section 6, test M): same `StrategySpec`, same real dataset
window, same frozen configuration -> the SAME deterministic economic outputs
(net PnL, Sharpe, trade/fill counts, and the same C++ target-schedule hash)
the already-committed NQ TSMOM canonical experiment carries in the real,
already-acquired local registry. It is a REPLAY/EQUIVALENCE check only -- it
does not write a duplicate scientific result anywhere.

The remaining tests are cheap (no real backtest) and exercise the wiring:
capability-gap handling, protocol-mismatch refusal, and config-identity
reuse.
"""
from __future__ import annotations

import pytest
from alpha_agent.agents.execution_service import (
    ProductionExecutionValidationService,
)
from alpha_agent.agents.orchestrator import (
    FamilyManifest,
    FamilyMember,
    IdentityPlanes,
    NonFamilyVerdict,
)
from alpha_agent.registry.enums import AttemptStatus, InvalidationClass, TrialRole
from alpha_agent.registry.identity import (
    experiment_identity,
    parameter_variant_identity,
)
from alpha_agent.registry.phase_13_5c_import import config_identities_for_orchestrator
from alpha_agent.strategy import strategy_fingerprint
from alpha_agent.strategy.baselines.factories import make_tsmom_spec
from alpha_agent.strategy.baselines.params import TsmomParams
from alpha_agent.ui import services
from alpha_agent.validation.phase_13_5c_matrix import (
    HOLDOUT_START_NS,
    RESEARCH_WINDOW,
    build_split_plan,
    dataset_identity_for_root,
    frozen_bootstrap_config,
    frozen_cost_plan,
    frozen_null_config,
    frozen_walk_forward,
)
from alpha_agent.validation.policy import ReliabilityPolicy
from alpha_agent.validation.spec import validation_protocol_fingerprint

_NQ_TSMOM_PARAMS = {"fast_horizon": 20, "slow_horizon": 120, "size": 1, "root_symbol": "NQ"}


def _member_for(spec, *, root: str, family: str, params: dict, feature_fp: str = "featset1:test") -> FamilyMember:
    fp = strategy_fingerprint(spec)
    pvi = parameter_variant_identity(params)
    identity = experiment_identity(
        strategy_fingerprint=fp, strategy_family=family, root_symbol=root,
        parameter_variant_identity=pvi, dataset_fingerprint="valdataset2:test",
        split_identity="splitplan1:test", validation_spec_fingerprint="validationprotocol1:test",
        reliability_policy_fingerprint=ReliabilityPolicy().identity(),
        execution_config_identity="execconfig1:test", cost_config_identity="costconfig1:test",
        risk_identity="riskconfig1:test", feature_spec_fingerprint=feature_fp,
    )
    return FamilyMember(
        ordinal=0, experiment_identity=identity, strategy_fingerprint=fp,
        strategy_id=spec.strategy_id, strategy_family=family, root_symbol=root,
        params=params, parameter_variant_identity=pvi, feature_spec_fingerprint=feature_fp,
        trial_role=TrialRole.CANONICAL, hypothesis_id="H-TEST", hypothesis_title="replay test",
        strategy_spec=spec, strategy_spec_json={"schema": "registry-strategy-spec/1", "params": params},
    )


def _manifest_for(member: FamilyMember, planes: IdentityPlanes) -> FamilyManifest:
    from alpha_agent.agents.orchestrator import _compute_family_id
    from alpha_agent.registry.models import MarketWindow

    window = MarketWindow(label="VALIDATION_2023_2024", start_date="2023-01-01", end_date="2024-12-31")
    policy_id = planes.reliability_policy_fingerprint
    fam_id = _compute_family_id(
        family_stem="agent-test", generation=0, parent_family_id=None, planes=planes,
        market_window=window, fdr_q_threshold=0.10, policy_identity=policy_id,
        declared_target_size=1, members=(member,),
    )
    return FamilyManifest(
        family_id=fam_id, family_stem="agent-test", generation=0, planes=planes,
        market_window=window, fdr_q_threshold=0.10, policy_identity=policy_id,
        declared_target_size=1, members=(member,),
    )


def test_unsupported_family_is_invalid_execution_not_a_scientific_rejection():
    spec = make_tsmom_spec(TsmomParams(**_NQ_TSMOM_PARAMS))
    member = _member_for(spec, root="NQ", family="dsl:custom_blueprint", params=_NQ_TSMOM_PARAMS)
    planes = IdentityPlanes(
        dataset_fingerprint="valdataset2:x", split_identity="splitplan1:x",
        validation_spec_fingerprint="validationprotocol1:x",
        reliability_policy_fingerprint=ReliabilityPolicy().identity(),
        execution_config_identity="execconfig1:x", cost_config_identity="costconfig1:x",
        risk_identity="riskconfig1:x",
    )
    manifest = _manifest_for(member, planes)
    svc = ProductionExecutionValidationService()
    ev = svc.run(member=member, manifest=manifest, attempt_ordinal=1, holdout=False)
    assert ev.status is AttemptStatus.INVALID_EXECUTION
    assert ev.invalidation_class is InvalidationClass.SOFTWARE_DEFECT
    assert "dsl:custom_blueprint" in ev.invalidation_detail
    assert ev.trial_p_value is None and ev.non_family_verdict is None  # no scientific content at all


def test_holdout_true_is_refused_outright():
    from alpha_agent.agents.orchestrator import HoldoutIsolationError

    spec = make_tsmom_spec(TsmomParams(**_NQ_TSMOM_PARAMS))
    member = _member_for(spec, root="NQ", family="tsmom", params=_NQ_TSMOM_PARAMS)
    planes = IdentityPlanes(
        dataset_fingerprint="valdataset2:x", split_identity="splitplan1:x",
        validation_spec_fingerprint="validationprotocol1:x",
        reliability_policy_fingerprint=ReliabilityPolicy().identity(),
        execution_config_identity="execconfig1:x", cost_config_identity="costconfig1:x",
        risk_identity="riskconfig1:x",
    )
    manifest = _manifest_for(member, planes)
    svc = ProductionExecutionValidationService()
    with pytest.raises(HoldoutIsolationError):
        svc.run(member=member, manifest=manifest, attempt_ordinal=1, holdout=True)


@pytest.mark.slow
def test_protocol_mismatch_refuses_rather_than_silently_executing(tmp_path):
    """A manifest declaring a validation-protocol fingerprint the service did
    not itself derive must be refused, not silently executed under a
    different protocol than the frozen family was planned against. Marked
    slow: computing the real dataset identity to compare against requires a
    real (offline, already-acquired) dataset reconstitution."""
    spec = make_tsmom_spec(TsmomParams(**_NQ_TSMOM_PARAMS))
    member = _member_for(spec, root="NQ", family="tsmom", params=_NQ_TSMOM_PARAMS)
    planes = IdentityPlanes(
        dataset_fingerprint="valdataset2:wrong", split_identity="splitplan1:wrong",
        validation_spec_fingerprint="validationprotocol1:definitely_not_it",
        reliability_policy_fingerprint=ReliabilityPolicy().identity(),
        execution_config_identity="execconfig1:x", cost_config_identity="costconfig1:x",
        risk_identity="riskconfig1:x",
    )
    manifest = _manifest_for(member, planes)
    svc = ProductionExecutionValidationService(work_dir=tmp_path)
    ev = svc.run(member=member, manifest=manifest, attempt_ordinal=1, holdout=False)
    assert ev.status is AttemptStatus.INVALID_EXECUTION
    assert ev.invalidation_class is InvalidationClass.SOFTWARE_DEFECT
    assert "protocol" in ev.invalidation_detail.lower()


def test_config_identities_reuse_the_authoritative_registry_values():
    """The SAME execution/cost/risk config identities already used by the
    real, authoritative registry rows -- not a second, parallel numbering."""
    ids = config_identities_for_orchestrator(services.REPO_ROOT)
    with services.open_registry() as reg:
        views = list(reg.experiments(strategy_family="tsmom", root_symbol="NQ", authoritative_only=True))
    assert views
    exp = views[0].experiment
    assert ids["execution_config_identity"] == exp.execution_config_identity
    assert ids["cost_config_identity"] == exp.cost_config_identity
    assert ids["risk_identity"] == exp.risk_identity


@pytest.mark.slow
def test_replay_matches_committed_nq_tsmom_canonical_result(tmp_path):
    """Section 6, test M -- the required replay/equivalence proof. Runs the
    REAL C++ backtest + frozen validation over the REAL already-acquired NQ
    2018-2024 data via `ProductionExecutionValidationService`, for the exact
    same canonical TSMOM(20, 120) StrategySpec the already-committed
    `NQ__TSMOM__CANONICAL__VALIDATION_2023_2024` experiment used, and asserts
    the deterministic economic outputs match. This does NOT write to the
    production registry -- it only calls the execution service directly."""
    with services.open_registry() as reg:
        views = list(reg.experiments(strategy_family="tsmom", root_symbol="NQ", authoritative_only=True))
    canonical = next(v for v in views if v.experiment.trial_role is TrialRole.CANONICAL)
    committed = canonical.result
    assert committed is not None

    spec = make_tsmom_spec(TsmomParams(**_NQ_TSMOM_PARAMS))
    member = _member_for(spec, root="NQ", family="tsmom", params=_NQ_TSMOM_PARAMS)

    svc = ProductionExecutionValidationService(work_dir=tmp_path)
    recon = svc._reconstitute("NQ")
    span_lo, span_hi = 0, HOLDOUT_START_NS
    from alpha_agent.data.real_market_dataset import _iso_ns

    span_lo = _iso_ns(RESEARCH_WINDOW[0])
    exec_bars = recon.canonical_bars
    exec_bars = exec_bars[(exec_bars["ts_event_ns"] >= span_lo) & (exec_bars["ts_event_ns"] < span_hi)][
        ["ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume"]
    ].sort_values("ts_event_ns").reset_index(drop=True)
    dataset = dataset_identity_for_root(recon, exec_bars)
    policy = ReliabilityPolicy()
    protocol_fp = validation_protocol_fingerprint(
        dataset=dataset, split_plan=build_split_plan(), walk_forward=frozen_walk_forward(),
        cost_stress=frozen_cost_plan(), null_test=frozen_null_config(), bootstrap=frozen_bootstrap_config(),
        minimum_sample=policy.minimum_sample, reliability_policy_fingerprint=policy.identity(),
    )
    planes = IdentityPlanes(
        dataset_fingerprint=dataset.identity(), split_identity=build_split_plan().split_fingerprint(),
        validation_spec_fingerprint=protocol_fp, reliability_policy_fingerprint=policy.identity(),
        execution_config_identity="execconfig1:x", cost_config_identity="costconfig1:x",
        risk_identity="riskconfig1:x",
    )
    manifest = _manifest_for(member, planes)

    ev = svc.run(member=member, manifest=manifest, attempt_ordinal=1, holdout=False)

    assert ev.status is AttemptStatus.VALID, ev.invalidation_detail if ev.status is AttemptStatus.INVALID_EXECUTION else None
    assert ev.non_family_verdict in (NonFamilyVerdict.ELIGIBLE, NonFamilyVerdict.REJECT, NonFamilyVerdict.INCONCLUSIVE)

    # -- the required equivalence: deterministic C++/validation outputs match --
    assert ev.metrics["net_pnl_usd"] == pytest.approx(committed.net_pnl_usd, abs=1.0)
    assert ev.metrics["gross_pnl_usd"] == pytest.approx(committed.gross_pnl_usd, abs=1.0)
    assert ev.metrics["costs_usd"] == pytest.approx(committed.costs_usd, abs=1.0)
    assert ev.metrics["daily_sharpe"] == pytest.approx(committed.daily_sharpe, abs=1e-4)
    assert ev.metrics["n_trades"] == committed.n_trades
    assert ev.metrics["n_fills"] == committed.n_fills
    assert ev.metrics["n_oos_days"] == committed.n_oos_days
    assert ev.target_schedule_hash == canonical.experiment.target_schedule_hash
    # C++ PnL/fills remain authoritative -- Python never substitutes its own
    # number for what the C++ engine actually produced (test N).
    assert ev.engine == "alpha_agent.agents.execution_service.ProductionExecutionValidationService/1"
