"""Alpha Discovery campaign, Part J -- investor visualization tests (task
spec section 76's chart-adjacent coverage: real artifacts only, no
fabricated equity curve for an unbound experiment)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from alpha_agent.artifacts.store import manifest_path_for, persist_run_artifacts
from alpha_agent.ui import services
from alpha_agent.ui.charts_discovery import (
    daily_drawdown_chart,
    daily_equity_curve_chart,
    max_drawdown_fraction,
)
from alpha_agent.validation.runner import synthetic_backtest_run


def _daily_rows():
    run = synthetic_backtest_run(np.array([100.0, -300.0, 50.0, 200.0]), n_trades=4, n_fills=8)
    return [
        {"trading_day": run.daily.trading_day[i], "equity_usd": run.daily.equity_usd[i],
         "daily_pnl_usd": run.daily.daily_pnl_usd[i]}
        for i in range(run.daily.n_days)
    ]


def test_daily_equity_curve_chart_returns_a_figure_with_real_points():
    rows = _daily_rows()
    fig = daily_equity_curve_chart(rows)
    assert isinstance(fig, go.Figure)
    assert list(fig.data[0].y) == [r["equity_usd"] for r in rows]


def test_daily_drawdown_chart_never_positive():
    rows = _daily_rows()
    fig = daily_drawdown_chart(rows)
    assert all(v <= 1e-9 for v in fig.data[0].y)


def test_max_drawdown_fraction_matches_hand_computation():
    rows = _daily_rows()
    equity = [r["equity_usd"] for r in rows]
    peak = float("-inf")
    worst = 0.0
    for e in equity:
        peak = max(peak, e)
        worst = min(worst, (e - peak) / peak) if peak > 0 else worst
    assert max_drawdown_fraction(rows) == abs(worst)


def test_max_drawdown_fraction_empty_is_none():
    assert max_drawdown_fraction([]) is None


# ---------------------------------------------------------------------------
# services.find_experiment_bound_artifact_bundle -- hash-verified binding
# ---------------------------------------------------------------------------


def test_find_bound_bundle_none_when_no_result():
    assert services.find_experiment_bound_artifact_bundle({}) is None


def test_find_bound_bundle_none_when_source_artifact_is_not_a_manifest():
    detail = {"result": {"source_artifact": "outputs/phase_13_5c/foo_validation_report.json"}}
    assert services.find_experiment_bound_artifact_bundle(detail) is None


def test_find_bound_bundle_none_when_source_artifact_missing():
    detail = {"result": {"source_artifact": None}}
    assert services.find_experiment_bound_artifact_bundle(detail) is None


def test_find_bound_bundle_real_round_trip(tmp_path):
    run = synthetic_backtest_run(np.array([10.0, 20.0]), n_trades=2, n_fills=2)
    bundle = persist_run_artifacts(
        experiment_identity="experiment1:visualtest", attempt_ordinal=1,
        strategy_fingerprint="stratfp1:v", daily=run.daily, out_dir=tmp_path,
    )
    manifest = manifest_path_for(bundle, out_dir=tmp_path)
    detail = {"result": {"source_artifact": str(manifest), "source_artifact_sha256": bundle.bundle_sha256}}
    loaded = services.find_experiment_bound_artifact_bundle(detail)
    assert loaded is not None
    assert loaded["experiment_identity"] == "experiment1:visualtest"

    rows = services.daily_equity_rows_from_bundle(loaded)
    assert rows is not None
    assert len(rows) == run.daily.n_days


def test_find_bound_bundle_refuses_on_hash_mismatch(tmp_path):
    run = synthetic_backtest_run(np.array([10.0]), n_trades=1, n_fills=1)
    bundle = persist_run_artifacts(
        experiment_identity="experiment1:tampertest", attempt_ordinal=1,
        strategy_fingerprint="stratfp1:t", daily=run.daily, out_dir=tmp_path,
    )
    manifest = manifest_path_for(bundle, out_dir=tmp_path)
    detail = {"result": {"source_artifact": str(manifest), "source_artifact_sha256": "0" * 64}}
    assert services.find_experiment_bound_artifact_bundle(detail) is None


def test_find_bound_bundle_refuses_when_referenced_file_tampered(tmp_path):
    run = synthetic_backtest_run(np.array([10.0]), n_trades=1, n_fills=1)
    bundle = persist_run_artifacts(
        experiment_identity="experiment1:filetamper", attempt_ordinal=1,
        strategy_fingerprint="stratfp1:f", daily=run.daily, out_dir=tmp_path,
    )
    manifest = manifest_path_for(bundle, out_dir=tmp_path)
    detail = {"result": {"source_artifact": str(manifest), "source_artifact_sha256": bundle.bundle_sha256}}
    # tamper with the daily-equity CSV after the manifest recorded its hash
    from pathlib import Path

    Path(bundle.daily_equity.path).write_text("tampered\n", encoding="utf-8")
    assert services.find_experiment_bound_artifact_bundle(detail) is None


def test_find_bound_bundle_refuses_when_price_bars_file_tampered(tmp_path):
    """Alpha Discovery live-research campaign, Checkpoint 11: `price_bars`
    (new, additive) must be hash-verified exactly like fills/trades/
    daily_equity -- a bundle whose recorded `price_bars` file was edited
    after the fact must be refused wholesale, not silently accepted with an
    unverified price-bar file."""
    import pandas as pd

    run = synthetic_backtest_run(np.array([10.0]), n_trades=1, n_fills=1)
    price_bars = pd.DataFrame({
        "ts_event_ns": [1, 2], "instrument_id": [1, 1], "open": [1.0, 1.0],
        "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1, 1],
    })
    bundle = persist_run_artifacts(
        experiment_identity="experiment1:pricebarstamper", attempt_ordinal=1,
        strategy_fingerprint="stratfp1:pb", daily=run.daily, price_bars=price_bars, out_dir=tmp_path,
    )
    assert bundle.price_bars is not None
    manifest = manifest_path_for(bundle, out_dir=tmp_path)
    detail = {"result": {"source_artifact": str(manifest), "source_artifact_sha256": bundle.bundle_sha256}}
    assert services.find_experiment_bound_artifact_bundle(detail) is not None  # untampered -- verifies fine

    from pathlib import Path

    Path(bundle.price_bars.path).write_text("tampered\n", encoding="utf-8")
    assert services.find_experiment_bound_artifact_bundle(detail) is None


def test_daily_equity_rows_from_bundle_none_without_daily_equity():
    assert services.daily_equity_rows_from_bundle({"daily_equity": None}) is None
    assert services.daily_equity_rows_from_bundle(None) is None


def test_real_registry_has_no_artifact_bundle_bound_experiments_today():
    """Matches the documented current state -- these artifacts only exist
    for a NEW discovery run executed with artifact_dir set; every existing
    committed experiment stays PATH DATA NOT AVAILABLE."""
    import pytest

    if not services.REGISTRY_PATH.exists():
        pytest.skip("Phase 14 registry sqlite not present in this checkout")
    for row in services.list_experiments():
        detail = {"result": row}
        assert services.find_experiment_bound_artifact_bundle(detail) is None
