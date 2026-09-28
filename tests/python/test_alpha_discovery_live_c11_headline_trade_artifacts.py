"""Alpha Discovery live-research campaign, Checkpoint 11 -- real headline
fills/trades persistence (task spec Part 5, "STRICT-VALIDATION ARTIFACTS").

Closes Part I's documented scope limit ("Per-fill/per-trade CSV export is
NOT wired here: `ValidationEngine` calls the runner many times internally...
and wiring a trade export through every one of those calls would touch that
frozen engine's internals"). The fix is additive and narrow:
`ValidationEngine` now accepts optional `headline_trades_out_path` /
`headline_fills_out_path`, threaded to ONLY its single headline `_run_span`
call (never a fold / null / cost-stress / neighbour / ablation run) --
`ValidationEngine`'s scientific computation is completely unchanged, proven
by the untouched `test_reliability_validation.py` / `test_phase_13_5c.py`
suites staying green and by `test_replay_matches_committed_nq_tsmom_
canonical_result` (unmodified) still passing byte-for-byte.
"""
from __future__ import annotations

import pytest
from alpha_agent.agents.execution_service import ProductionExecutionValidationService
from alpha_agent.agents.orchestrator import IdentityPlanes
from alpha_agent.registry.enums import AttemptStatus, TrialRole
from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
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


def _member_for(spec, *, root, family, params, feature_fp="featset1:test"):
    from alpha_agent.agents.orchestrator import FamilyMember

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
        trial_role=TrialRole.CANONICAL, hypothesis_id="H-TEST", hypothesis_title="headline artifact test",
        strategy_spec=spec, strategy_spec_json={"schema": "registry-strategy-spec/1", "params": params},
    )


def _manifest_for(member, planes):
    from alpha_agent.agents.orchestrator import FamilyManifest, _compute_family_id
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


@pytest.mark.slow
def test_headline_trades_and_fills_are_persisted_alongside_daily_equity(tmp_path):
    with services.open_registry() as reg:
        views = list(reg.experiments(strategy_family="tsmom", root_symbol="NQ", authoritative_only=True))
    canonical = next(v for v in views if v.experiment.trial_role is TrialRole.CANONICAL)
    committed = canonical.result
    assert committed is not None

    spec = make_tsmom_spec(TsmomParams(**_NQ_TSMOM_PARAMS))
    member = _member_for(spec, root="NQ", family="tsmom", params=_NQ_TSMOM_PARAMS)

    artifact_dir = tmp_path / "artifacts"
    svc = ProductionExecutionValidationService(work_dir=tmp_path / "work", artifact_dir=artifact_dir)
    recon = svc._reconstitute("NQ")
    from alpha_agent.data.real_market_dataset import _iso_ns

    span_lo = _iso_ns(RESEARCH_WINDOW[0])
    span_hi = HOLDOUT_START_NS
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
    assert ev.status is AttemptStatus.VALID, getattr(ev, "invalidation_detail", None)

    # -- the pre-existing replay equivalence still holds with artifact_dir set --
    assert ev.metrics["n_trades"] == committed.n_trades
    assert ev.metrics["n_fills"] == committed.n_fills

    # -- the NEW proof: real headline trades/fills were actually written --
    assert ev.source_artifact is not None
    import json

    with open(ev.source_artifact, encoding="utf-8") as fh:
        manifest_json = json.load(fh)
    assert manifest_json["daily_equity"] is not None
    if committed.n_trades > 0:
        assert manifest_json["trades"] is not None
        assert manifest_json["trades"]["n_rows"] == committed.n_trades
    if committed.n_fills > 0:
        assert manifest_json["fills"] is not None
        assert manifest_json["fills"]["n_rows"] == committed.n_fills

    # -- the real executed price bars for the headline window, for a future
    # Price + Position chart aligned to real fill timestamps.
    assert manifest_json["price_bars"] is not None
    assert manifest_json["price_bars"]["n_rows"] > 0
    price_bars_path = manifest_json["price_bars"]["path"]
    with open(price_bars_path, encoding="utf-8") as fh:
        header = fh.readline().strip()
    assert header == "ts_event_ns,instrument_id,open,high,low,close,volume"


def test_run_holdout_never_writes_a_headline_artifact_even_if_paths_are_supplied(tmp_path):
    """A direct, cheap `ValidationEngine` unit test (small synthetic trend
    series, real CLI, seconds not minutes): `run_holdout()` must ignore
    `headline_trades_out_path` / `headline_fills_out_path` even if a caller
    mistakenly supplied them -- on top of `ProductionExecutionValidationService.
    run` already refusing `holdout=True` outright before an engine even
    exists."""
    import test_reliability_validation as trv
    from alpha_agent.validation.enums import SplitRole

    _spec, _pol, engine = trv._tsmom_spec_and_engine(tmp_path, trv.vf.trend_closes(400))
    trades_path = tmp_path / "should_never_exist_trades.csv"
    fills_path = tmp_path / "should_never_exist_fills.csv"
    engine._headline_trades_out_path = trades_path
    engine._headline_fills_out_path = fills_path
    if not engine.spec.split_plan.has_role(SplitRole.LOCKED_HOLDOUT):
        pytest.skip("fixture split plan carries no LOCKED_HOLDOUT window")

    engine.run_holdout()
    assert not trades_path.exists()
    assert not fills_path.exists()
