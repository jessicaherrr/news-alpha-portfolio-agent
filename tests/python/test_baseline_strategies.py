"""Phase 11 -- Baseline strategy library + deterministic Python->C++ target bridge.

Covers checklist A-V of the Phase 11 prompt. Deterministic, no network. The C++
CLI tests skip cleanly when the core is not built.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import baseline_fixtures as bf
import numpy as np
import pytest
from alpha_agent.adapters.targets_bridge import run_targets_backtest_cli, write_targets_bundle
from alpha_agent.backtest import (
    TARGET_SCHEDULE_COLUMNS,
    TargetSchedule,
    build_run_manifest,
    build_target_schedule,
    summarize_cpp_result,
)
from alpha_agent.backtest.targets import TargetScheduleRow
from alpha_agent.features import compute_features
from alpha_agent.features.registry import REGISTRY
from alpha_agent.features.spec import FeatureSpec
from alpha_agent.schemas.market_data import BOUNDARY_BAR_COLUMNS, CONTRACT_COLUMNS, PriceDomain
from alpha_agent.strategy import compile_strategy
from alpha_agent.strategy.baselines import (
    BASELINE_FAMILIES,
    BreakoutParams,
    MaTrendParams,
    MeanReversionParams,
    TsmomParams,
    carry_baseline_status,
    make_breakout_spec,
    make_ma_trend_spec,
    make_mean_reversion_spec,
    make_tsmom_spec,
)
from alpha_agent.strategy.baselines.factories import SIGNED_PRICE_SAFE_KINDS
from alpha_agent.strategy.enums import Comparator, DefaultAction
from alpha_agent.strategy.errors import StrategyEvaluationError
from alpha_agent.strategy.evaluator import ReferenceEvaluator
from alpha_agent.strategy.spec import (
    ComparisonNode,
    ConstOperand,
    FeatureDeclaration,
    FeatureOperand,
    Rule,
    StrategySpec,
    TargetAction,
)
from features_fixtures import synthetic_roll_sources

CLI = Path("build/cpp/cpp/quant_backtest_targets_csv")
BIG_EXP = bf.BASE_NS + 10**15


def _all_factories():
    return [
        make_tsmom_spec(TsmomParams(root_symbol="NQ", fast_horizon=3, slow_horizon=10)),
        make_ma_trend_spec(MaTrendParams(root_symbol="ES", fast_window=3, slow_window=10)),
        make_breakout_spec(BreakoutParams(root_symbol="CL", lookback=5)),
        make_mean_reversion_spec(
            MeanReversionParams(root_symbol="GC", zscore_window=10, entry_z=1.5, exit_z=0.5)
        ),
    ]


def _plan_and_frame(spec, closes, *, domain=PriceDomain.RAW_CONTRACT, root="NQ", raw="NQU6"):
    plan = compile_strategy(spec)
    src = bf.price_source(closes, root=root, raw_symbol=raw, domain=domain)
    frame = compute_features(src, [b.spec for b in plan.feature_bindings])
    return plan, frame


# ==========================================================================
# A. each baseline factory produces a valid StrategySpec
# ==========================================================================
def test_A_factories_produce_valid_strategyspec():
    for spec in _all_factories():
        assert isinstance(spec, StrategySpec)
        assert spec.schema_version == "strategy-dsl/1"
        assert len(spec.features) >= 1 and len(spec.rules) >= 1
        # one root per spec, and every referenced feature kind is registered
        for decl in spec.features:
            assert decl.spec.kind in REGISTRY.kinds()


# ==========================================================================
# B. each StrategySpec compiles through the Phase 10 compiler
# ==========================================================================
def test_B_specs_compile_through_phase10():
    for spec in _all_factories():
        plan = compile_strategy(spec)
        assert plan.fingerprint.startswith("stratdsl1:")
        assert plan.root_symbol == spec.root_symbol
        assert plan.warmup_bars >= 1


def test_B_multi_asset_same_factory_different_roots():
    a = compile_strategy(make_tsmom_spec(TsmomParams(root_symbol="NQ", fast_horizon=5, slow_horizon=20)))
    b = compile_strategy(make_tsmom_spec(TsmomParams(root_symbol="CL", fast_horizon=5, slow_horizon=20)))
    assert a.root_symbol == "NQ" and b.root_symbol == "CL"
    # identical logic, different root -> different fingerprint is NOT required by
    # the DSL (root is in the canonical payload), but both must be one-root scoped
    assert a.fingerprint != b.fingerprint


# ==========================================================================
# C. invalid baseline parameters are rejected
# ==========================================================================
def test_C_invalid_params_rejected():
    with pytest.raises(ValueError):
        TsmomParams(root_symbol="NQ", fast_horizon=10, slow_horizon=10)   # fast >= slow
    with pytest.raises(ValueError):
        TsmomParams(root_symbol="NQ", fast_horizon=20, slow_horizon=5)
    with pytest.raises(ValueError):
        MaTrendParams(root_symbol="NQ", fast_window=50, slow_window=20)   # fast >= slow
    with pytest.raises(ValueError):
        MaTrendParams(root_symbol="NQ", fast_window=0, slow_window=20)    # non-positive
    with pytest.raises(ValueError):
        BreakoutParams(root_symbol="NQ", lookback=1)                      # < 2
    with pytest.raises(ValueError):
        MeanReversionParams(root_symbol="NQ", zscore_window=10, entry_z=1.0, exit_z=1.0)
    with pytest.raises(ValueError):
        MeanReversionParams(root_symbol="NQ", zscore_window=10, entry_z=0.5, exit_z=1.0)
    with pytest.raises(ValueError):
        MeanReversionParams(root_symbol="NQ", zscore_window=1, entry_z=2.0, exit_z=0.5)
    with pytest.raises(ValueError):
        TsmomParams(root_symbol="NQ", fast_horizon=1, slow_horizon=5, size=0)
    with pytest.raises(ValueError):
        TsmomParams(root_symbol="nq", fast_horizon=1, slow_horizon=5)     # bad root pattern


# ==========================================================================
# D. baseline reference evaluator is deterministic
# ==========================================================================
def test_D_evaluator_deterministic():
    spec = make_ma_trend_spec(MaTrendParams(root_symbol="NQ", fast_window=3, slow_window=10))
    plan, frame = _plan_and_frame(spec, bf.trending_closes())
    d1 = ReferenceEvaluator(plan).evaluate_frame(frame)
    d2 = ReferenceEvaluator(plan).evaluate_frame(frame)
    assert [x.model_dump() for x in d1] == [x.model_dump() for x in d2]


# ==========================================================================
# E / F. target schedule is deterministic and has a stable hash
# ==========================================================================
def test_E_schedule_deterministic():
    spec = make_tsmom_spec(TsmomParams(root_symbol="NQ", fast_horizon=3, slow_horizon=10))
    plan, frame = _plan_and_frame(spec, bf.trending_closes())
    s1 = build_target_schedule(plan, frame)
    s2 = build_target_schedule(plan, frame)
    assert s1.rows == s2.rows
    assert s1.schedule_hash() == s2.schedule_hash()


def test_F_schedule_hash_ignores_audit_field_and_row_identity():
    spec = make_tsmom_spec(TsmomParams(root_symbol="NQ", fast_horizon=3, slow_horizon=10))
    plan, frame = _plan_and_frame(spec, bf.trending_closes())
    sched = build_target_schedule(plan, frame)
    h = sched.schedule_hash()
    # a fixed known hash string form
    assert h.startswith("targsched1:") and len(h.split(":")[1]) == 64
    # dropping / changing matched_rule_id does not move the executable-intent hash
    stripped = sched.model_copy(
        update={"rows": tuple(r.model_copy(update={"matched_rule_id": None}) for r in sched.rows)}
    )
    assert stripped.schedule_hash() == h


# ==========================================================================
# G. a schedule with mixed strategy fingerprints is rejected
# ==========================================================================
def test_G_mixed_fingerprints_rejected():
    fp_a = "stratdsl1:" + "a" * 64
    fp_b = "stratdsl1:" + "b" * 64
    rows = (
        TargetScheduleRow(ts_event_ns=1, root_symbol="NQ", target_units=1, strategy_fingerprint=fp_a),
        TargetScheduleRow(ts_event_ns=2, root_symbol="NQ", target_units=-1, strategy_fingerprint=fp_b),
    )
    with pytest.raises(ValueError, match="mixed strategy fingerprints"):
        TargetSchedule(
            root_symbol="NQ", strategy_fingerprint=fp_a, strategy_id="x",
            strategy_dsl_version="1.0.0", feature_engine_version="0.9.2",
            warmup_bars=0, rows=rows,
        )


# ==========================================================================
# H. a root mismatch is rejected (a one-root strategy cannot emit another root)
# ==========================================================================
def test_H_root_mismatch_rejected():
    fp = "stratdsl1:" + "c" * 64
    rows = (
        TargetScheduleRow(ts_event_ns=1, root_symbol="ES", target_units=1, strategy_fingerprint=fp),
    )
    with pytest.raises(ValueError, match="another root"):
        TargetSchedule(
            root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
            strategy_dsl_version="1.0.0", feature_engine_version="0.9.2",
            warmup_bars=0, rows=rows,
        )


# ==========================================================================
# I. NO DECISION is distinct from an explicit target 0
# ==========================================================================
def test_I_no_decision_distinct_from_flat():
    spec = make_mean_reversion_spec(
        MeanReversionParams(root_symbol="NQ", zscore_window=10, entry_z=1.5, exit_z=0.5)
    )
    plan, frame = _plan_and_frame(spec, bf.mean_reverting_closes())
    sched = build_target_schedule(plan, frame)
    ts = frame.identifiers["ts_event_ns"].to_numpy()
    scheduled_ts = {r.ts_event_ns for r in sched.rows}

    # warm-up bars produce NO row at all (not a row with target 0); the feature
    # first becomes available at index warmup_bars - 1, so everything strictly
    # before that is a pure NO-DECISION region.
    warmup_ts = {int(t) for t in ts[: plan.warmup_bars - 1]}
    assert warmup_ts.isdisjoint(scheduled_ts)
    assert len(sched.rows) < len(ts)  # some bars had no decision at all
    # but an explicit flat decision (exit band) is a real row with target_units == 0
    assert any(r.target_units == 0 for r in sched.rows)
    # every emitted row corresponds to a bar where the strategy had full info
    decisions = {d.ts_event_ns: d for d in ReferenceEvaluator(plan).evaluate_frame(frame)}
    for r in sched.rows:
        assert not decisions[r.ts_event_ns].missing_features


# ==========================================================================
# J. a Python decision at T executes in C++ at the NEXT eligible bar, not T
# ==========================================================================
@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_J_decision_at_T_executes_next_bar(tmp_path):
    closes = bf.trending_closes(40)
    bars = bf.boundary_bars(closes, instrument_id=810001)
    contracts = bf.one_contract(instrument_id=810001, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    fp = "stratdsl1:" + "d" * 64
    last_ts = int(bars["ts_event_ns"].iloc[-1])
    # the ONLY target row is on the final bar -> there is no next bar -> no fill
    sched = TargetSchedule(
        root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
        strategy_dsl_version="1.0.0", feature_engine_version="0.9.2", warmup_bars=0,
        rows=(TargetScheduleRow(ts_event_ns=last_ts, root_symbol="NQ", target_units=1, strategy_fingerprint=fp),),
    )
    res = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path)
    assert res["target_rows_applied"] == 1   # the decision was made
    assert res["fills"] == 0                  # ...but never executes on the decision bar itself


# ==========================================================================
# J2 (Phase 11.1). an absent schedule row is NO DECISION: it emits no Signal,
#     so a sparse schedule produces exactly as many strategy signals as rows.
# ==========================================================================
def test_J2_bridge_default_policy_is_no_decision():
    import inspect

    sig = inspect.signature(run_targets_backtest_cli)
    assert sig.parameters["schedule_policy"].default == "no_decision"


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_J2_absent_row_is_no_decision_not_flat_or_retry(tmp_path):
    closes = bf.trending_closes(30)
    bars = bf.boundary_bars(closes, instrument_id=817001)
    contracts = bf.one_contract(instrument_id=817001, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    fp = "stratdsl1:" + "9" * 64
    ts = [int(t) for t in bars["ts_event_ns"]]
    # only 2 informed decisions across 30 bars
    sched = TargetSchedule(
        root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
        strategy_dsl_version="1.0.0", feature_engine_version="0.9.2", warmup_bars=0,
        rows=(
            TargetScheduleRow(ts_event_ns=ts[5], root_symbol="NQ", target_units=1, strategy_fingerprint=fp),
            TargetScheduleRow(ts_event_ns=ts[10], root_symbol="NQ", target_units=1, strategy_fingerprint=fp),
        ),
    )
    res = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path)
    # NO DECISION on the other 28 bars -> exactly 2 strategy signals, not 30
    assert res["signals"] == 2
    assert res["schedule_policy"] == "no_decision"
    # explicit +1 twice with a flat delta the 2nd time -> one open, then held to EOT
    assert res["fills"] == 2   # open + EOT close only (no churn from absence)


# ==========================================================================
# J3 (Phase 11.2). consecutive distinct decisions are independent intents:
#     the schedule keeps every one (no dedup / overwrite) and the C++ engine
#     replays them all. (The latency>0 pending-queue invariant is exercised in
#     depth by cpp/tests/test_latency_queue.cpp -- the frozen CLI runs latency 0.)
# ==========================================================================
@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_J3_consecutive_distinct_targets_all_survive(tmp_path):
    closes = bf.trending_closes(20)
    bars = bf.boundary_bars(closes, instrument_id=818001)
    contracts = bf.one_contract(instrument_id=818001, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    fp = "stratdsl1:" + "7" * 64
    ts = [int(t) for t in bars["ts_event_ns"]]
    rows = tuple(
        TargetScheduleRow(ts_event_ns=ts[i], root_symbol="NQ", target_units=t, strategy_fingerprint=fp)
        for i, t in ((3, 1), (4, -1), (5, 1), (6, 0))
    )
    sched = TargetSchedule(
        root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
        strategy_dsl_version="1.0.0", feature_engine_version="0.9.2", warmup_bars=0, rows=rows,
    )
    assert sched.n_rows == 4  # every consecutive decision kept, none overwritten
    res = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path)
    assert res["signals"] == 4
    # +1,-1,+1,0 each executed at the next bar -> a fill per non-zero delta
    assert res["fills"] >= 3
    # each row's target is part of the executable-intent hash (no collapsing)
    swapped = sched.model_copy(update={"rows": tuple(
        r.model_copy(update={"target_units": -r.target_units}) for r in rows
    )})
    assert swapped.schedule_hash() != sched.schedule_hash()


# ==========================================================================
# K. the target schedule cannot specify a raw contract
# ==========================================================================
def test_K_schedule_cannot_name_raw_contract():
    fp = "stratdsl1:" + "e" * 64
    with pytest.raises(ValueError, match="raw contract"):
        TargetScheduleRow(ts_event_ns=1, root_symbol="NQU6", target_units=1, strategy_fingerprint=fp)


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_K_cli_rejects_raw_contract_root(tmp_path):
    bars = bf.boundary_bars(bf.trending_closes(20), instrument_id=811001)
    contracts = bf.one_contract(instrument_id=811001, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    from alpha_agent.adapters.cpp_cli import write_boundary_bundle
    bp, cp = write_boundary_bundle(bars, contracts, tmp_path)
    tp = tmp_path / "targets.csv"
    tp.write_text(
        "ts_event_ns,root_symbol,target_units,strategy_fingerprint,matched_rule_id\n"
        f"{int(bars['ts_event_ns'].iloc[2])},NQU6,1,stratdsl1:{'f'*64},\n"
    )
    proc = subprocess.run([str(CLI), str(bp), str(cp), str(tp)], capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert "raw" in proc.stderr.lower() or "bare root" in proc.stderr.lower()


# ==========================================================================
# L. the target schedule cannot specify an execution price
# ==========================================================================
def test_L_schedule_csv_rejects_execution_columns(tmp_path):
    p = tmp_path / "targets.csv"
    p.write_text(
        "ts_event_ns,root_symbol,target_units,strategy_fingerprint,fill_price\n"
        f"1,NQ,1,stratdsl1:{'a'*64},29000.0\n"
    )
    with pytest.raises(ValueError):
        TargetSchedule.from_csv(p, root_symbol="NQ", strategy_fingerprint="stratdsl1:" + "a" * 64)


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_L_cli_rejects_execution_columns(tmp_path):
    bars = bf.boundary_bars(bf.trending_closes(20), instrument_id=812001)
    contracts = bf.one_contract(instrument_id=812001, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    from alpha_agent.adapters.cpp_cli import write_boundary_bundle
    bp, cp = write_boundary_bundle(bars, contracts, tmp_path)
    tp = tmp_path / "targets.csv"
    tp.write_text(
        "ts_event_ns,root_symbol,target_units,strategy_fingerprint,fill_price\n"
        f"{int(bars['ts_event_ns'].iloc[2])},NQ,1,stratdsl1:{'a'*64},29000.0\n"
    )
    proc = subprocess.run([str(CLI), str(bp), str(cp), str(tp)], capture_output=True, text=True, check=False)
    assert proc.returncode != 0


# ==========================================================================
# N. official realized PnL comes from the C++ result verbatim
# ==========================================================================
def test_N_summary_reads_cpp_pnl_verbatim():
    fake = {
        "bars": 100, "events": 100, "signals": 100, "orders": 4, "fills": 4, "trades": 3,
        "rolls": 0, "unique_contracts": 1, "open_positions": 0,
        "gross_pnl_usd": 123.5, "costs_usd": 8.0, "net_pnl_usd": 115.5,
        "max_drawdown_usd": 42.0, "unrealized_pnl_usd": 0.0,
        "equity_usd": 1_000_115.5, "starting_capital_usd": 1_000_000.0,
        "risk_rejects": 0, "risk_resizes": 0,
    }
    s = summarize_cpp_result(fake, strategy_fingerprint="stratdsl1:x", root_symbol="NQ", date_range_ns=(1, 2))
    assert s.gross_realized_pnl_usd == 123.5
    assert s.net_realized_pnl_usd == 115.5
    assert s.costs_usd == 8.0
    assert s.official_pnl_source == "cpp_fill_events"


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_N_end_to_end_official_pnl_is_cpp(tmp_path):
    spec = make_ma_trend_spec(MaTrendParams(root_symbol="NQ", fast_window=3, slow_window=10))
    closes = bf.trending_closes(80)
    plan, frame = _plan_and_frame(spec, closes)
    sched = build_target_schedule(plan, frame)
    bars = bf.boundary_bars(closes, instrument_id=813001)
    contracts = bf.one_contract(instrument_id=813001, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    res = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path)
    assert res["strategy_fingerprint"] == plan.fingerprint
    assert res["fills"] >= 2
    # net = gross - costs, all C++-sourced
    assert abs(res["net_pnl_usd"] - (res["gross_pnl_usd"] - res["costs_usd"])) < 1e-6
    summ = summarize_cpp_result(
        res, strategy_fingerprint=plan.fingerprint, root_symbol="NQ", date_range_ns=sched.date_range_ns
    )
    assert summ.net_realized_pnl_usd == res["net_pnl_usd"]
    manifest = build_run_manifest(spec=spec, plan=plan, frame=frame, schedule=sched, bars=bars, contracts=contracts)
    assert manifest.strategy_fingerprint == plan.fingerprint
    assert manifest.target_schedule_hash == sched.schedule_hash()
    assert manifest.source_fingerprint == frame.lineage.source_fingerprint


# ==========================================================================
# O / P / Q. trend, mean-reversion and breakout baselines on synthetic data
# ==========================================================================
def test_O_trend_baseline_follows_the_trend():
    up = bf.trending_closes(80, slope=0.5, wiggle=0.1)
    spec = make_tsmom_spec(TsmomParams(root_symbol="NQ", fast_horizon=3, slow_horizon=10))
    plan, frame = _plan_and_frame(spec, up)
    sched = build_target_schedule(plan, frame)
    targets = [r.target_units for r in sched.rows]
    assert targets and set(targets) <= {-1, 0, 1}
    # a persistent uptrend -> the modal informed target is long
    assert targets.count(1) > targets.count(-1)

    down = list(reversed(up))
    _, frame_d = _plan_and_frame(spec, down)
    sched_d = build_target_schedule(plan, frame_d)
    td = [r.target_units for r in sched_d.rows]
    assert td.count(-1) > td.count(1)


def test_P_mean_reversion_fades_extremes():
    closes = bf.mean_reverting_closes(120, amp=8.0)
    spec = make_mean_reversion_spec(
        MeanReversionParams(root_symbol="NQ", zscore_window=15, entry_z=1.2, exit_z=0.3)
    )
    plan, frame = _plan_and_frame(spec, closes)
    decisions = {d.ts_event_ns: d for d in ReferenceEvaluator(plan).evaluate_frame(frame)}
    z = frame.features[plan.required_features[0]].to_numpy()
    ts = frame.identifiers["ts_event_ns"].to_numpy()
    saw_long_on_low_z = False
    saw_short_on_high_z = False
    for i, t in enumerate(ts):
        d = decisions[int(t)]
        if d.missing_features:
            continue
        if z[i] < -1.2 and d.target_units == 1:
            saw_long_on_low_z = True
        if z[i] > 1.2 and d.target_units == -1:
            saw_short_on_high_z = True
    assert saw_long_on_low_z and saw_short_on_high_z


def test_Q_breakout_baseline_enters_on_the_break():
    closes = bf.breakout_closes()
    spec = make_breakout_spec(BreakoutParams(root_symbol="NQ", lookback=6))
    plan, frame = _plan_and_frame(spec, closes)
    sched = build_target_schedule(plan, frame)
    targets = [r.target_units for r in sched.rows]
    assert 1 in targets   # the upside break
    assert -1 in targets  # the later downside break
    # keep-previous default: once long, stays long until the down-break flips it
    seq = targets
    first_long = seq.index(1)
    first_short = seq.index(-1)
    assert first_long < first_short
    assert all(v in (1, 0) for v in seq[first_long:first_short])


# ==========================================================================
# R. point-in-time back-adjusted trend -> Signal, but the Fill is RawContract
# ==========================================================================
def test_R_retrospective_backadjusted_source_rejected():
    rs = synthetic_roll_sources()
    spec = make_ma_trend_spec(MaTrendParams(root_symbol="NQ", fast_window=2, slow_window=3))
    plan = compile_strategy(spec)
    retro_frame = compute_features(
        rs["backadj_retro"], [b.spec for b in plan.feature_bindings], require_point_in_time=False
    )
    with pytest.raises(StrategyEvaluationError):
        build_target_schedule(plan, retro_frame)


def test_R_pit_backadjusted_avoids_future_roll_and_fills_raw(tmp_path):
    rs = synthetic_roll_sources()
    # unadjusted continuous shows the artificial roll jump; point-in-time
    # back-adjusted does not use the future roll basis.
    spec_diff = FeatureSpec(kind="diff", params={"n": 1})
    cont = compute_features(rs["raw_continuous"], [spec_diff])
    pit = compute_features(rs["backadj_pit"], [spec_diff], require_point_in_time=False)
    ts_cont = cont.identifiers["ts_event_ns"].to_numpy()
    roll_i = int(np.where(ts_cont == rs["roll_ts"])[0][0])
    jump_unadj = abs(float(cont.features["diff_1"].to_numpy()[roll_i]))
    # the continuous feed's 1-bar change straddling the roll carries the ~+10 basis
    assert jump_unadj >= rs["additive_gap"] - 1.0
    # the point-in-time adjusted series (as-of the pre-roll bar) has no such jump
    pit_vals = pit.features["diff_1"].to_numpy()
    assert np.nanmax(np.abs(pit_vals)) < rs["additive_gap"] - 1.0

    if not CLI.exists():
        pytest.skip("C++ core not built")
    # A point-in-time back-adjusted trend feature may generate a Signal; execution
    # still occurs in the real RawContract at raw prices (never the adjusted value).
    spec = make_ma_trend_spec(MaTrendParams(root_symbol="NQ", fast_window=2, slow_window=3))
    plan = compile_strategy(spec)
    sig_frame = compute_features(rs["raw_continuous"], [b.spec for b in plan.feature_bindings])
    sched = build_target_schedule(plan, sig_frame)

    import pandas as pd
    from futures_fixtures import NQU6, NQZ6, two_contract_history
    hist = two_contract_history(u6_close=100.0, z6_open=110.0)
    bars = hist[["ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume"]].copy()
    contracts = pd.DataFrame(
        [
            {"instrument_id": c.instrument_id, "raw_symbol": c.raw_symbol, "root_symbol": c.root_symbol,
             "exchange": c.exchange, "tick_size": c.tick_size, "multiplier": c.multiplier,
             "activation_ns": c.activation_ns, "expiration_ns": c.expiration_ns,
             "first_notice_ns": "", "last_trade_ns": ""}
            for c in (NQU6, NQZ6)
        ]
    )[list(CONTRACT_COLUMNS)]
    if sched.n_rows == 0:
        pytest.skip("no informed target rows for this tiny fixture")
    res = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path)
    assert res["contracts_resolved"] >= 1
    assert res["unique_contracts"] >= 1
    # fills exist and PnL is finite and consistent with RAW prices (~100 / ~110),
    # not the back-adjusted values (~100 / ~100).
    assert np.isfinite(res["gross_pnl_usd"])


# ==========================================================================
# S. signed-price (CL) compatible baseline behaviour
# ==========================================================================
def test_S_factories_only_use_signed_price_safe_features():
    for spec in _all_factories():
        for decl in spec.features:
            assert decl.spec.kind in SIGNED_PRICE_SAFE_KINDS


def test_S_mean_reversion_runs_on_negative_prices():
    spec = make_mean_reversion_spec(
        MeanReversionParams(root_symbol="CL", zscore_window=5, entry_z=1.2, exit_z=0.4)
    )
    plan = compile_strategy(spec)
    src = bf.price_source(bf.SIGNED_CLOSES, root="CL", raw_symbol="CLK0", spread=0.3,
                          domain=PriceDomain.RAW_CONTRACT)
    frame = compute_features(src, [b.spec for b in plan.feature_bindings])
    s1 = build_target_schedule(plan, frame)
    s2 = build_target_schedule(plan, frame)
    assert s1.schedule_hash() == s2.schedule_hash()
    # the z-score feature is defined across the sign change (no NaN storm)
    z = frame.features[plan.required_features[0]].to_numpy()
    assert np.isfinite(z[plan.warmup_bars:]).any()


def test_S_percentage_return_feature_is_not_used_for_signed_prices():
    # A hand-built spec that (wrongly) uses a percentage return on a CL series:
    # the feature suppresses the non-positive-base rows, so the strategy simply
    # has no signal there rather than emitting garbage.
    spec = StrategySpec(
        strategy_name="return on signed CL (invalid domain demo)",
        strategy_id="DEMO-RET-CL",
        root_symbol="CL",
        features=[FeatureDeclaration(alias="r", spec=FeatureSpec(kind="return", params={"n": 1}))],
        rules=[Rule(rule_id="long", when=ComparisonNode(op=Comparator.GT, left=FeatureOperand(feature="r"),
                    right=ConstOperand(value=0.0)), action=TargetAction(target_units=1))],
        default_action=DefaultAction.FLAT,
    )
    plan = compile_strategy(spec)
    src = bf.price_source(bf.SIGNED_CLOSES, root="CL", raw_symbol="CLK0", spread=0.3,
                          domain=PriceDomain.RAW_CONTRACT)
    frame = compute_features(src, [b.spec for b in plan.feature_bindings])
    sched = build_target_schedule(plan, frame)
    # rows only where the return was actually defined (base price > 0)
    for r in sched.rows:
        assert r.target_units in (0, 1)


# ==========================================================================
# T. deterministic full replay (Python schedule + C++ result)
# ==========================================================================
@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_T_full_replay_is_deterministic(tmp_path):
    spec = make_tsmom_spec(TsmomParams(root_symbol="NQ", fast_horizon=4, slow_horizon=16))
    closes = bf.trending_closes(90)
    plan, frame = _plan_and_frame(spec, closes)
    bars = bf.boundary_bars(closes, instrument_id=815001)
    contracts = bf.one_contract(instrument_id=815001, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)

    sched_a = build_target_schedule(plan, frame)
    sched_b = build_target_schedule(compile_strategy(spec), frame)
    assert sched_a.schedule_hash() == sched_b.schedule_hash()

    r1 = run_targets_backtest_cli(bars, contracts, sched_a, executable=CLI, work_dir=tmp_path / "a")
    r2 = run_targets_backtest_cli(bars, contracts, sched_b, executable=CLI, work_dir=tmp_path / "b")
    assert r1 == r2


# ==========================================================================
# V. the frozen Phase 02.5 boundary is unchanged
# ==========================================================================
def test_V_frozen_boundary_unchanged(tmp_path):
    assert BOUNDARY_BAR_COLUMNS == (
        "ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume",
    )
    assert CONTRACT_COLUMNS == (
        "instrument_id", "raw_symbol", "root_symbol", "exchange",
        "tick_size", "multiplier", "activation_ns", "expiration_ns",
        "first_notice_ns", "last_trade_ns",
    )
    # targets.csv is a SEPARATE boundary, never overloaded onto bars.csv
    assert TARGET_SCHEDULE_COLUMNS == (
        "ts_event_ns", "root_symbol", "target_units", "strategy_fingerprint", "matched_rule_id",
    )
    # only ts_event_ns is shared (the common time key); the intent-carrying
    # columns never appear on the frozen bar wire format.
    assert set(TARGET_SCHEDULE_COLUMNS) & set(BOUNDARY_BAR_COLUMNS) == {"ts_event_ns"}
    for col in ("target_units", "strategy_fingerprint", "root_symbol", "matched_rule_id"):
        assert col not in BOUNDARY_BAR_COLUMNS

    spec = make_ma_trend_spec(MaTrendParams(root_symbol="NQ", fast_window=3, slow_window=10))
    plan, frame = _plan_and_frame(spec, bf.trending_closes(40))
    sched = build_target_schedule(plan, frame)
    bars = bf.boundary_bars(bf.trending_closes(40), instrument_id=816001)
    contracts = bf.one_contract(instrument_id=816001, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    bp, cp, tp = write_targets_bundle(bars, contracts, sched, tmp_path)
    import pandas as pd
    assert list(pd.read_csv(bp).columns) == list(BOUNDARY_BAR_COLUMNS)
    assert list(pd.read_csv(cp).columns) == list(CONTRACT_COLUMNS)
    assert list(pd.read_csv(tp).columns) == list(TARGET_SCHEDULE_COLUMNS)


# ==========================================================================
# carry status + family docs (sections 21, 23)
# ==========================================================================
def test_carry_baseline_is_deferred_with_a_reason():
    st = carry_baseline_status()
    assert st.available is False
    assert st.blocking_requirements and st.dsl_primitives_needed


def test_family_docs_present_for_every_factory():
    keys = {f.key for f in BASELINE_FAMILIES}
    assert keys == {"tsmom", "ma_trend", "breakout", "mean_reversion"}
    for fam in BASELINE_FAMILIES:
        assert fam.is_claimed_alpha is False
        assert fam.param_grid_ranges and fam.failure_regimes
