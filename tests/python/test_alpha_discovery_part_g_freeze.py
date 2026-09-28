"""Alpha Discovery campaign, Part G -- Freeze-before-strict-validation tests
(task spec section 75)."""
from __future__ import annotations

import ast
import inspect

import pytest
from alpha_agent.agents.orchestrator import FamilyMember, IdentityPlanes
from alpha_agent.discovery.candidate_pool import CandidatePoolResult, MechanismOutcome
from alpha_agent.knowledge import EconomicMechanism
from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
from alpha_agent.registry.models import MarketWindow
from alpha_agent.screening.fast_screen import FastScreenStatus, FastScreenTrial, ResearchScreenScore
from alpha_agent.screening.freeze import (
    FreezeError,
    freeze_top_k,
    read_frozen_manifest,
    write_frozen_manifest,
)
from alpha_agent.strategy import strategy_fingerprint
from alpha_agent.strategy.baselines.factories import make_ma_trend_spec, make_tsmom_spec
from alpha_agent.strategy.baselines.params import MaTrendParams, TsmomParams
from alpha_agent.validation.policy import ReliabilityPolicy

POLICY = ReliabilityPolicy()
PLANES = IdentityPlanes(
    dataset_fingerprint="valdataset2:ds", split_identity="split1:sp",
    validation_spec_fingerprint="validationspec1:vs",
    reliability_policy_fingerprint=POLICY.identity(),
    execution_config_identity="execconfig1:ex", cost_config_identity="costconfig1:co",
    risk_identity="riskconfig1:ri",
)
MW = MarketWindow(label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31")


def _member(idx: int, *, family: str, params: dict, spec) -> FamilyMember:
    fp = strategy_fingerprint(spec)
    pvi = parameter_variant_identity(params)
    identity = experiment_identity(
        strategy_fingerprint=fp, strategy_family=family, root_symbol="NQ",
        parameter_variant_identity=pvi, dataset_fingerprint=PLANES.dataset_fingerprint,
        split_identity=PLANES.split_identity, validation_spec_fingerprint=PLANES.validation_spec_fingerprint,
        reliability_policy_fingerprint=PLANES.reliability_policy_fingerprint,
        execution_config_identity=PLANES.execution_config_identity,
        cost_config_identity=PLANES.cost_config_identity, risk_identity=PLANES.risk_identity,
        feature_spec_fingerprint=f"featset1:test{idx}",
    )
    return FamilyMember(
        ordinal=idx, experiment_identity=identity, strategy_fingerprint=fp, strategy_id=spec.strategy_id,
        strategy_family=family, root_symbol="NQ", params=params, parameter_variant_identity=pvi,
        feature_spec_fingerprint=f"featset1:test{idx}", hypothesis_id=f"H-{idx}",
        hypothesis_title=f"test hypothesis {idx}", strategy_spec=spec,
        strategy_spec_json={"schema": "registry-strategy-spec/1"},
    )


def _pool_and_trials(n: int = 3) -> tuple[CandidatePoolResult, list[FastScreenTrial]]:
    members = []
    trials = []
    specs = [
        ("tsmom", {"fast_horizon": 20, "slow_horizon": 120, "size": 1, "root_symbol": "NQ"},
         make_tsmom_spec(TsmomParams(fast_horizon=20, slow_horizon=120, size=1, root_symbol="NQ"))),
        ("ma_trend", {"fast_window": 50, "slow_window": 200, "size": 1, "root_symbol": "NQ"},
         make_ma_trend_spec(MaTrendParams(fast_window=50, slow_window=200, size=1, root_symbol="NQ"))),
        ("tsmom", {"fast_horizon": 10, "slow_horizon": 60, "size": 1, "root_symbol": "NQ"},
         make_tsmom_spec(TsmomParams(fast_horizon=10, slow_horizon=60, size=1, root_symbol="NQ"))),
    ][:n]
    for i, (family, params, spec) in enumerate(specs):
        m = _member(i, family=family, params=params, spec=spec)
        members.append(m)
        score = ResearchScreenScore(
            sharpe_component=float(40 - i * 5), cost_component=10.0, drawdown_component=10.0,
            activity_component=10.0, total=float(70 - i * 10),
        )
        trials.append(FastScreenTrial(
            experiment_identity=m.experiment_identity, strategy_family=family, root_symbol="NQ",
            status=FastScreenStatus.SCREENED, score=score,
        ))
    pool = CandidatePoolResult(
        market="NQ", ideas_considered=n, mechanisms_attempted=n, mechanisms_supported=n,
        members=tuple(members),
        per_mechanism=(
            MechanismOutcome(
                mechanism=EconomicMechanism.TREND, objective="x", ideas_considered=1,
                members=(members[0],), planning_log=(),
            ),
        ),
    )
    return pool, trials


def test_freeze_selects_top_k_by_screen_score():
    pool, trials = _pool_and_trials(3)
    frozen = freeze_top_k(
        pool=pool, trials=trials, k=2, family_stem="test_freeze", planes=PLANES, market_window=MW,
    )
    assert frozen.frozen_count == 2
    assert frozen.manifest.declared_target_size == 2
    assert frozen.manifest.planning_complete
    # highest two scores (70, 60) -> members 0 and 1
    ids = {m.experiment_identity for m in frozen.manifest.members}
    assert ids == {trials[0].experiment_identity, trials[1].experiment_identity}


def test_freeze_is_deterministic():
    pool, trials = _pool_and_trials(3)
    a = freeze_top_k(pool=pool, trials=trials, k=2, family_stem="test_freeze", planes=PLANES, market_window=MW)
    b = freeze_top_k(pool=pool, trials=trials, k=2, family_stem="test_freeze", planes=PLANES, market_window=MW)
    assert a.manifest.family_id == b.manifest.family_id
    assert a.manifest.manifest_fingerprint() == b.manifest.manifest_fingerprint()


def test_freeze_is_a_real_family_not_a_family_of_one():
    pool, trials = _pool_and_trials(3)
    frozen = freeze_top_k(pool=pool, trials=trials, k=3, family_stem="test_freeze", planes=PLANES, market_window=MW)
    assert frozen.manifest.predeclared_family_size >= 2


def test_freeze_rejects_mismatched_trials():
    pool, trials = _pool_and_trials(3)
    with pytest.raises(FreezeError):
        freeze_top_k(pool=pool, trials=trials[:2], k=2, family_stem="x", planes=PLANES, market_window=MW)


def test_freeze_handles_fewer_screened_than_k():
    pool, trials = _pool_and_trials(2)
    frozen = freeze_top_k(pool=pool, trials=trials, k=10, family_stem="x", planes=PLANES, market_window=MW)
    assert frozen.frozen_count == 2  # never padded to a fake 10


def test_freeze_raises_when_nothing_survives():
    pool, trials = _pool_and_trials(2)
    unsupported = [
        FastScreenTrial(experiment_identity=t.experiment_identity, strategy_family=t.strategy_family,
                        root_symbol=t.root_symbol, status=FastScreenStatus.UNSUPPORTED)
        for t in trials
    ]
    with pytest.raises(FreezeError):
        freeze_top_k(pool=pool, trials=unsupported, k=2, family_stem="x", planes=PLANES, market_window=MW)


def test_write_and_read_frozen_manifest_round_trips(tmp_path):
    pool, trials = _pool_and_trials(2)
    frozen = freeze_top_k(pool=pool, trials=trials, k=2, family_stem="x", planes=PLANES, market_window=MW)
    path = write_frozen_manifest(frozen, out_dir=tmp_path)
    assert path.exists()
    reloaded = read_frozen_manifest(path)
    assert reloaded.manifest.family_id == frozen.manifest.family_id
    assert reloaded.screen_scores == frozen.screen_scores


def test_write_frozen_manifest_never_overwrites_the_same_family_id(tmp_path):
    pool, trials = _pool_and_trials(2)
    frozen = freeze_top_k(pool=pool, trials=trials, k=2, family_stem="x", planes=PLANES, market_window=MW)
    path1 = write_frozen_manifest(frozen, out_dir=tmp_path)
    mtime1 = path1.stat().st_mtime_ns
    path2 = write_frozen_manifest(frozen, out_dir=tmp_path)
    assert path1 == path2
    assert path1.stat().st_mtime_ns == mtime1  # not rewritten


def test_freeze_module_never_imports_execution_or_validation_engine():
    """Task spec section 42/75: freezing happens strictly BEFORE strict
    validation -- this module must not be able to execute anything itself."""
    import alpha_agent.screening.freeze as mod

    tree = ast.parse(inspect.getsource(mod))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    forbidden_prefixes = (
        "alpha_agent.agents.execution_service", "alpha_agent.validation.engine",
        "alpha_agent.validation.runner",
    )
    for m in imported:
        assert not m.startswith(forbidden_prefixes), f"freeze.py must not import {m}"
