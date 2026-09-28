"""Alpha Discovery campaign, Part I -- experiment-bound execution artifacts
tests (task spec section 76)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from alpha_agent.artifacts.store import (
    ArtifactBundle,
    manifest_path_for,
    persist_run_artifacts,
    read_artifact_bundle,
)
from alpha_agent.validation.runner import synthetic_backtest_run


def test_persist_run_artifacts_writes_a_real_daily_equity_csv(tmp_path):
    run = synthetic_backtest_run(np.array([100.0, -50.0, 200.0]), n_trades=3, n_fills=6)
    bundle = persist_run_artifacts(
        experiment_identity="experiment1:deadbeef", attempt_ordinal=1,
        strategy_fingerprint="stratfp1:x", daily=run.daily, out_dir=tmp_path,
    )
    assert bundle.daily_equity is not None
    assert Path(bundle.daily_equity.path).exists()
    assert bundle.daily_equity.n_rows == run.daily.n_days
    assert bundle.fills is None
    assert bundle.trades is None


def test_persist_run_artifacts_copies_real_fills_and_trades_csv(tmp_path):
    fills_src = tmp_path / "src_fills.csv"
    fills_src.write_text("fill_id,ts_ns,price\n1,100,50.0\n2,200,51.0\n", encoding="utf-8")
    trades_src = tmp_path / "src_trades.csv"
    trades_src.write_text("trade_id,net_pnl\n1,10.0\n", encoding="utf-8")

    run = synthetic_backtest_run(np.array([10.0]), n_trades=1, n_fills=2)
    bundle = persist_run_artifacts(
        experiment_identity="experiment1:cafef00d", attempt_ordinal=2,
        strategy_fingerprint="stratfp1:y", daily=run.daily,
        fills_csv_path=fills_src, trades_csv_path=trades_src, out_dir=tmp_path / "out",
    )
    assert bundle.fills is not None and bundle.fills.n_rows == 2
    assert bundle.trades is not None and bundle.trades.n_rows == 1
    assert Path(bundle.fills.path).read_text(encoding="utf-8") == fills_src.read_text(encoding="utf-8")


def test_persist_run_artifacts_never_fabricates_a_missing_export(tmp_path):
    run = synthetic_backtest_run(np.array([1.0]), n_trades=1, n_fills=1)
    bundle = persist_run_artifacts(
        experiment_identity="experiment1:x", attempt_ordinal=1, strategy_fingerprint="stratfp1:x",
        daily=run.daily, fills_csv_path=tmp_path / "does_not_exist.csv", out_dir=tmp_path / "out2",
    )
    assert bundle.fills is None  # nonexistent path never becomes a fake artifact


def test_manifest_and_bundle_round_trip(tmp_path):
    run = synthetic_backtest_run(np.array([5.0, 5.0]), n_trades=2, n_fills=2)
    bundle = persist_run_artifacts(
        experiment_identity="experiment1:roundtrip", attempt_ordinal=1,
        strategy_fingerprint="stratfp1:z", daily=run.daily, out_dir=tmp_path,
    )
    path = manifest_path_for(bundle, out_dir=tmp_path)
    assert path.exists()
    reloaded = read_artifact_bundle(path)
    assert reloaded == bundle


def _real_cli_path() -> str:
    root = Path(__file__).resolve().parents[2]
    return str(root / "build" / "cpp" / "cpp" / "quant_backtest_targets_csv")


def test_run_fast_screen_persists_real_artifacts_when_requested(tmp_path):
    import pytest

    if not Path(_real_cli_path()).exists():
        pytest.skip("compiled CLI not present in this checkout")

    from alpha_agent.agents.orchestrator import FamilyMember
    from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
    from alpha_agent.screening.fast_screen import FastScreenStatus, run_fast_screen
    from alpha_agent.strategy import strategy_fingerprint
    from alpha_agent.strategy.baselines.factories import make_tsmom_spec
    from alpha_agent.strategy.baselines.params import TsmomParams
    from alpha_agent.validation.policy import ReliabilityPolicy

    params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1, "root_symbol": "NQ"}
    spec = make_tsmom_spec(TsmomParams(**params))
    fp = strategy_fingerprint(spec)
    pvi = parameter_variant_identity(params)
    identity = experiment_identity(
        strategy_fingerprint=fp, strategy_family="tsmom", root_symbol="NQ",
        parameter_variant_identity=pvi, dataset_fingerprint="valdataset2:artifacts",
        split_identity="splitplan1:artifacts", validation_spec_fingerprint="validationprotocol1:artifacts",
        reliability_policy_fingerprint=ReliabilityPolicy().identity(),
        execution_config_identity="execconfig1:artifacts", cost_config_identity="costconfig1:artifacts",
        risk_identity="riskconfig1:artifacts", feature_spec_fingerprint="featset1:artifacts",
    )
    member = FamilyMember(
        ordinal=0, experiment_identity=identity, strategy_fingerprint=fp, strategy_id=spec.strategy_id,
        strategy_family="tsmom", root_symbol="NQ", params=params, parameter_variant_identity=pvi,
        feature_spec_fingerprint="featset1:artifacts", hypothesis_id="H-ARTIFACTS",
        hypothesis_title="artifact persistence replay", strategy_spec=spec,
        strategy_spec_json={"schema": "registry-strategy-spec/1"},
    )
    work_dir = tmp_path / "work"
    artifact_dir = tmp_path / "artifacts"
    trial = run_fast_screen(
        member=member, cli_executable=_real_cli_path(), work_dir=work_dir, artifact_dir=artifact_dir,
    )
    assert trial.status is FastScreenStatus.SCREENED
    assert "artifact_manifest" in trial.metrics
    manifest_path = Path(trial.metrics["artifact_manifest"])
    assert manifest_path.exists()
    bundle = read_artifact_bundle(manifest_path)
    assert bundle.daily_equity is not None
    assert bundle.experiment_identity == identity


def test_bundle_sha256_is_stable_for_identical_content(tmp_path):
    run = synthetic_backtest_run(np.array([1.0, 2.0]), n_trades=2, n_fills=2)
    b1 = persist_run_artifacts(
        experiment_identity="experiment1:a", attempt_ordinal=1, strategy_fingerprint="stratfp1:a",
        daily=run.daily, out_dir=tmp_path / "a",
    )
    b2 = persist_run_artifacts(
        experiment_identity="experiment1:b", attempt_ordinal=1, strategy_fingerprint="stratfp1:a",
        daily=run.daily, out_dir=tmp_path / "b",
    )
    # different experiment_identity -> different bundle content -> different hash
    assert b1.bundle_sha256 != b2.bundle_sha256
    assert isinstance(b1, ArtifactBundle)
