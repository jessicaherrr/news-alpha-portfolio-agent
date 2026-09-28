"""Phase 19 -- pybind11 fast boundary.

The CLI (``targets_bridge.run_targets_backtest_cli`` /
``quant_backtest_targets_csv``) is the frozen reference implementation. The
pybind path (``pybind_bridge.run_targets_backtest_pybind``) uses the thin
``quant::run_targets_backtest`` helper, whose engine wiring mirrors the CLI's but
is wired independently -- it does NOT share an execution wrapper with the CLI.

These tests ARE the equivalence enforcement: they run both transports on the
same fixtures and assert every result field agrees (counts / strings exact;
floats within a tolerance that absorbs only the CLI's 6-significant-digit
``std::ostream`` serialisation vs the pybind path's full ``f64`` -- any residual
beyond that is a real disagreement and fails).

Deterministic, no network, no market-data spend. Skips cleanly when the optional
module is not built.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import validation_fixtures as vf
from alpha_agent.adapters.pybind_bridge import (
    BOUNDARY_TRANSPORT,
    assert_boundary_parity,
    is_pybind_available,
    pybind_boundary_provenance,
    run_targets_backtest_pybind,
)
from alpha_agent.adapters.targets_bridge import run_targets_backtest_cli
from alpha_agent.backtest.targets import TargetSchedule, TargetScheduleRow

CLI = Path("build/cpp/cpp/quant_backtest_targets_csv")
DAY = vf.DAY_NS
BIG_EXP = vf.BASE_NS + 10_000 * DAY

pytestmark = pytest.mark.skipif(not is_pybind_available(), reason="quant_core_py not built")


def _sched(rows: tuple[TargetScheduleRow, ...], fp: str) -> TargetSchedule:
    return TargetSchedule(
        root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
        strategy_dsl_version="1.0.0", feature_engine_version="0.9.2", warmup_bars=0,
        rows=rows,
    )


def _dense_case():
    closes = vf.trend_closes(40)
    bars = vf.daily_bars(closes, instrument_id=1, start_ns=vf.BASE_NS)
    contracts = vf.one_contract(instrument_id=1, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    fp = "stratdsl1:" + "a" * 64
    ts = [int(t) for t in bars["ts_event_ns"]]
    rows = tuple(
        TargetScheduleRow(ts_event_ns=t, root_symbol="NQ",
                          target_units=(1 if i % 3 else -1), strategy_fingerprint=fp)
        for i, t in enumerate(ts[:-1])
    )
    return bars, contracts, _sched(rows, fp), ts


# ==========================================================================
# A. plain run: CLI == pybind
# ==========================================================================
def test_A_plain_run_parity(tmp_path):
    bars, contracts, sched, _ = _dense_case()
    cli = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path / "cli")
    pb = run_targets_backtest_pybind(bars, contracts, sched, work_dir=tmp_path / "pb")
    assert_boundary_parity(cli, pb)
    # the pybind result is a literal drop-in: same key set
    assert set(cli) == set(pb)
    assert pb["fills"] > 0 and pb["daily_equity_basis"] == "utc_day"


# ==========================================================================
# B. cost overrides flow through identically
# ==========================================================================
@pytest.mark.parametrize("commission,slip,spread", [(2.0, 0.0, 0.0), (5.5, 1.0, 0.5), (0.0, 3.0, 0.0)])
def test_B_cost_override_parity(tmp_path, commission, slip, spread):
    bars, contracts, sched, _ = _dense_case()
    kw = {"commission_per_contract_usd": commission, "slippage_ticks": slip, "spread_ticks": spread}
    cli = run_targets_backtest_cli(bars, contracts, sched, executable=CLI,
                                   work_dir=tmp_path / "cli", **kw)
    pb = run_targets_backtest_pybind(bars, contracts, sched, work_dir=tmp_path / "pb", **kw)
    assert_boundary_parity(cli, pb)
    assert pb["commission_per_contract_usd"] == pytest.approx(commission)


# ==========================================================================
# C. validation-day plan -> trading_day daily-equity basis, parity on the trace
# ==========================================================================
def test_C_validation_days_parity(tmp_path):
    bars, contracts, sched, ts = _dense_case()
    vdays = tmp_path / "vd.csv"
    vdays.write_text("boundary_ts_ns\n" + "\n".join(str(t) for t in ts) + "\n")
    cli = run_targets_backtest_cli(bars, contracts, sched, executable=CLI,
                                   work_dir=tmp_path / "cli", validation_days_path=vdays)
    pb = run_targets_backtest_pybind(bars, contracts, sched, work_dir=tmp_path / "pb",
                                     validation_days_path=vdays)
    assert cli["daily_equity_basis"] == pb["daily_equity_basis"] == "trading_day"
    assert len(pb["daily_equity"]) == len(cli["daily_equity"]) == len(ts)
    assert_boundary_parity(cli, pb)


# ==========================================================================
# D. roll-close auxiliary marks: identical roll accounting
# ==========================================================================
def test_D_roll_close_marks_parity(tmp_path):
    m6, u6 = 10, 11
    exp6 = vf.BASE_NS + 100 * DAY
    b6 = vf.daily_bars([100.0, 101.0, 102.0], instrument_id=m6, start_ns=vf.BASE_NS)
    b7 = vf.daily_bars([202.0, 203.0, 204.0, 205.0], instrument_id=u6, start_ns=vf.BASE_NS + 3 * DAY)
    bars = pd.concat([b6, b7], ignore_index=True)
    contracts = pd.concat([
        vf.one_contract(instrument_id=m6, raw_symbol="NQM6", root="NQ", expiration_ns=exp6),
        vf.one_contract(instrument_id=u6, raw_symbol="NQU6", root="NQ", expiration_ns=exp6 + 90 * DAY),
    ], ignore_index=True)
    fp = "stratdsl1:" + "b" * 64
    ts = [int(t) for t in bars["ts_event_ns"]]
    sched = _sched(
        tuple(TargetScheduleRow(ts_event_ns=t, root_symbol="NQ", target_units=1,
                                strategy_fingerprint=fp) for t in ts[:-1]),
        fp,
    )
    vdays = tmp_path / "vd.csv"
    vdays.write_text("boundary_ts_ns\n" + "\n".join(str(t) for t in ts) + "\n")
    marks = tmp_path / "marks.csv"
    marks.write_text(f"instrument_id,ts_event_ns,close\n{m6},{ts[3]},102.5\n")

    kw = {"validation_days_path": vdays, "roll_close_marks_path": marks}
    cli = run_targets_backtest_cli(bars, contracts, sched, executable=CLI,
                                   work_dir=tmp_path / "cli", **kw)
    pb = run_targets_backtest_pybind(bars, contracts, sched, work_dir=tmp_path / "pb", **kw)
    assert pb["rolls"] == cli["rolls"] == 1
    assert pb["rolls_priced_contemporaneous"] == cli["rolls_priced_contemporaneous"] == 1
    assert pb["rolls_priced_auxiliary_marks"] == cli["rolls_priced_auxiliary_marks"] == 1
    assert_boundary_parity(cli, pb)


def test_D_roll_marks_without_validation_days_rejected(tmp_path):
    bars, contracts, sched, _ = _dense_case()
    with pytest.raises(ValueError):
        run_targets_backtest_pybind(bars, contracts, sched, work_dir=tmp_path,
                                    roll_close_marks_path=str(tmp_path / "m.csv"))


# ==========================================================================
# E. audit exports byte-identical to the CLI's --trades-out / --fills-out
# ==========================================================================
def test_E_audit_export_parity(tmp_path):
    bars, contracts, sched, _ = _dense_case()
    cli = run_targets_backtest_cli(
        bars, contracts, sched, executable=CLI, work_dir=tmp_path / "cli",
        trades_out_path=str(tmp_path / "cli_trades.csv"),
        fills_out_path=str(tmp_path / "cli_fills.csv"),
    )
    pb = run_targets_backtest_pybind(
        bars, contracts, sched, work_dir=tmp_path / "pb",
        trades_out_path=str(tmp_path / "pb_trades.csv"),
        fills_out_path=str(tmp_path / "pb_fills.csv"),
    )
    assert_boundary_parity(cli, pb)
    assert (tmp_path / "cli_trades.csv").read_text() == (tmp_path / "pb_trades.csv").read_text()
    assert (tmp_path / "cli_fills.csv").read_text() == (tmp_path / "pb_fills.csv").read_text()
    assert (tmp_path / "pb_trades.csv").read_text().splitlines()[0].startswith("trade_index,")


# ==========================================================================
# F. determinism -- same inputs, same result, twice
# ==========================================================================
def test_F_pybind_is_deterministic(tmp_path):
    bars, contracts, sched, _ = _dense_case()
    a = run_targets_backtest_pybind(bars, contracts, sched, work_dir=tmp_path / "a")
    b = run_targets_backtest_pybind(bars, contracts, sched, work_dir=tmp_path / "b")
    assert a == b


# ==========================================================================
# G. hard errors match the CLI (unresolved instrument_id)
# ==========================================================================
def test_G_unknown_instrument_id_raises(tmp_path):
    bars, contracts, sched, _ = _dense_case()
    bad = bars.copy()
    bad.loc[bad.index[-1], "instrument_id"] = 999_999
    # pybind maps the C++ std::runtime_error hard check to RuntimeError
    with pytest.raises(RuntimeError):
        run_targets_backtest_pybind(bad, contracts, sched, work_dir=tmp_path)


# ==========================================================================
# H. provenance surface
# ==========================================================================
def test_H_boundary_provenance():
    p = pybind_boundary_provenance()
    assert p["transport"] == BOUNDARY_TRANSPORT
    assert p["boundary_abi"].startswith("phase19-targets/")
    assert p["engine_path"] == "cpp_quant_core__run_targets_backtest"
    assert len(p["module_sha256"]) == 64
    assert Path(p["module_file"]).exists()
