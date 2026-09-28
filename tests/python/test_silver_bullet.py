"""Phase 12 -- Quantified ICT Silver Bullet / market-structure benchmark.

Checklist A-Z of ``prompts/12``. Deterministic, no network. Phase 12 asserts
MECHANICS and FREQUENCY only -- never profitability, alpha, robustness or
statistical significance (those are Phase 13).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import silver_bullet_fixtures as sbf
from alpha_agent.adapters.targets_bridge import run_targets_backtest_cli
from alpha_agent.backtest import build_run_manifest, build_target_schedule, summarize_cpp_result
from alpha_agent.data.calendars import SessionCalendar
from alpha_agent.features import compute_features
from alpha_agent.features.market_structure import (
    DISTANCE_UNITS,
    LIQUIDITY_REFERENCES,
    SilverBulletDetectorParams,
    SilverBulletSessionLabelsMissing,
    SilverBulletState,
    silver_bullet_result,
)
from alpha_agent.features.registry import REGISTRY
from alpha_agent.features.spec import FeatureSpec
from alpha_agent.schemas.market_data import PriceDomain
from alpha_agent.strategy import compile_strategy
from alpha_agent.strategy.baselines import (
    SILVER_BULLET_BENCHMARK,
    SILVER_BULLET_FAMILY,
    SILVER_BULLET_GRID_RANGES,
    SilverBulletParams,
    make_silver_bullet_spec,
)
from alpha_agent.strategy.errors import StrategyEvaluationError
from features_fixtures import synthetic_roll_sources

CLI = Path("build/cpp/cpp/quant_backtest_targets_csv")

TEST_PARAMS = {
    "root_symbol": "NQ",
    "liquidity_lookback": 6,
    "displacement_atr_window": 3,
    "displacement_atr_multiple": 1.5,
    "displacement_close_loc": 0.6,
    "retracement_fraction": 0.5,
    "setup_expiry_bars": 8,
    "max_holding_bars": 5,
}


def _params(**over) -> SilverBulletParams:
    return SilverBulletParams(**{**TEST_PARAMS, **over})


def _result(src, params: SilverBulletParams):
    return silver_bullet_result(src, params.feature_params())


# ==========================================================================
# A / C. prior liquidity reference is strictly causal (no centered / future swing)
# ==========================================================================
def test_A_C_liquidity_reference_is_prior_bars_only():
    src = sbf.bullish_setup_source(preamble=12)
    p = _params()
    res = _result(src, p)
    audit = res.audits[0]
    # rolling N-bar low reference = min of the 6 bars BEFORE the sweep (all 99.5)
    assert audit.liquidity_reference_type == "rolling_nbar_extreme"
    assert audit.liquidity_level == pytest.approx(99.5)

    # centered / future knowledge test: a deep low placed AFTER the reference
    # window must not change the level a sweep at T uses.
    rows = sbf.bullish_setup_rows(preamble=12)
    rows[20] = (95.0, 96.0, 80.0, 95.0)  # a far-future spike low
    src2 = sbf.sb_source(sbf.sb_bars(rows))
    a2 = _result(src2, p).audits[0]
    assert a2.liquidity_level == pytest.approx(99.5)  # unchanged -> not centered
    assert a2.sweep_ts_ns == audit.sweep_ts_ns


# ==========================================================================
# B. bullish / bearish liquidity sweep symmetry
# ==========================================================================
def test_B_sweep_bullish_and_bearish():
    p = _params()
    bull = _result(sbf.bullish_setup_source(), p)
    bear = _result(sbf.bearish_setup_source(), p)
    assert bull.diagnostics["entries_long"] == 1 and "entries_short" not in bull.diagnostics
    assert bear.diagnostics["entries_short"] == 1 and "entries_long" not in bear.diagnostics
    assert bull.audits[0].setup_direction == "long"
    assert bear.audits[0].setup_direction == "short"
    # symmetric geometry
    assert bull.audits[0].fvg_upper > bull.audits[0].fvg_lower
    assert bear.audits[0].fvg_upper > bear.audits[0].fvg_lower


def test_B_sweep_needs_close_back_across_the_level():
    # low pierces the prior low but the bar CLOSES below it -> not a sweep
    rows = [sbf._FLAT] * 12 + [
        (100.0, 100.2, 98.0, 98.5),   # low pierces prior low 99.5 but close 98.5 stays below it
        (98.5, 98.7, 97.0, 97.5),     # (and the high never takes out the prior high) -> no sweep
    ] + [sbf._FLAT] * 6
    res = _result(sbf.sb_source(sbf.sb_bars(rows)), _params())
    assert res.diagnostics.get("setups_started", 0) == 0


# ==========================================================================
# D. displacement threshold (body / ATR and close location)
# ==========================================================================
def test_D_displacement_threshold_enforced():
    src = sbf.bullish_setup_source()
    # a very high ATR multiple -> the displacement bar no longer qualifies
    strict = _result(src, _params(displacement_atr_multiple=50.0))
    assert strict.diagnostics.get("reached_displacement", 0) == 0
    assert strict.diagnostics.get("entries", 0) == 0
    # a close-location threshold above 1.0 is impossible to meet
    loc = _result(src, _params(displacement_close_loc=1.0, displacement_atr_multiple=1.5))
    # close_loc of the displacement bar is 0.94 -> fails a 1.0 requirement
    assert loc.diagnostics.get("reached_displacement", 0) == 0


def test_D_displacement_must_follow_the_sweep_bar():
    # N. no intrabar path invention -- displacement is only accepted on a bar
    # STRICTLY after the sweep bar.
    res = _result(sbf.bullish_setup_source(), _params())
    a = res.audits[0]
    assert a.displacement_ts_ns > a.sweep_ts_ns


# ==========================================================================
# E / F. bullish & bearish FVG geometry
# ==========================================================================
def test_E_F_fvg_geometry():
    pb = _result(sbf.bullish_setup_source(), _params()).audits[0]
    # bullish FVG lower = high of the pre-displacement (sweep) bar = 101.5
    # upper = low of the post-displacement bar = 104.0
    assert (pb.fvg_lower, pb.fvg_upper) == pytest.approx((101.5, 104.0))

    ps = _result(sbf.bearish_setup_source(), _params()).audits[0]
    assert (ps.fvg_lower, ps.fvg_upper) == pytest.approx((96.0, 98.5))


def test_E_fvg_min_width_rejects_a_narrow_gap():
    res = _result(sbf.bullish_setup_source(), _params(fvg_min_width=5.0))  # gap is 2.5
    assert res.diagnostics.get("fvg_too_small", 0) >= 1
    assert res.diagnostics.get("entries", 0) == 0


# ==========================================================================
# G. the FVG timestamp is only stamped when it is fully observable (Td+1)
# ==========================================================================
def test_G_fvg_timestamp_is_the_third_bar():
    src = sbf.bullish_setup_source()
    res = _result(src, _params())
    ts = src.frame["ts_event_ns"].to_numpy("int64")
    a = res.audits[0]
    disp_i = int(np.where(ts == a.displacement_ts_ns)[0][0])
    fvg_i = int(np.where(ts == a.fvg_created_ts_ns)[0][0])
    assert fvg_i == disp_i + 1  # first bar the 3-bar gap is knowable, never Td


# ==========================================================================
# H / I. retracement occurs after the FVG exists, never before
# ==========================================================================
def test_H_I_retracement_after_fvg_only():
    res = _result(sbf.bullish_setup_source(), _params())
    a = res.audits[0]
    assert a.retracement_ts_ns is not None
    assert a.retracement_ts_ns > a.fvg_created_ts_ns
    assert a.decision_ts_ns == a.retracement_ts_ns


def test_I_no_retracement_without_a_confirmed_fvg():
    # displacement present but NO gap (post-displacement bar overlaps the
    # pre-displacement bar) -> no FVG -> no entry even though price later dips.
    rows = [sbf._FLAT] * 12 + [
        (100.0, 101.5, 98.0, 101.0),
        (101.0, 108.5, 100.5, 108.0),
        (108.0, 109.0, 101.0, 102.0),   # low 101.0 < sweep-bar high 101.5 -> NO gap
        (102.0, 102.0, 100.0, 100.5),
    ] + [sbf._FLAT] * 6
    res = _result(sbf.sb_source(sbf.sb_bars(rows)), _params())
    assert res.diagnostics.get("fvg_absent", 0) >= 1
    assert res.diagnostics.get("entries", 0) == 0


# ==========================================================================
# J. setup expiration
# ==========================================================================
def test_J_setup_expires_without_a_retracement():
    # after the FVG, price never comes back -> EXPIRED, no entry.
    rows = [sbf._FLAT] * 12 + sbf._bullish_block()[:3] + [
        (109.0 + i, 110.0 + i, 108.0 + i, 109.5 + i) for i in range(10)
    ]
    res = _result(sbf.sb_source(sbf.sb_bars(rows)), _params(setup_expiry_bars=3))
    assert res.diagnostics.get("expired_no_retracement", 0) >= 1
    assert res.diagnostics.get("entries", 0) == 0
    assert res.audits[0].expiry_reason == "retracement_not_reached"


# ==========================================================================
# K. session reset -- market-structure state does not leak across a session
# ==========================================================================
def test_K_session_reset_clears_state():
    # a sweep + displacement at the end of day 1, then a new trading day: the
    # partial setup is dropped, not carried into day 2.
    day1 = [sbf._FLAT] * 12 + sbf._bullish_block()[:2]
    day2 = [sbf._FLAT] * 10
    rows = day1 + day2
    td = ["2026-09-01"] * len(day1) + ["2026-09-02"] * len(day2)
    src = sbf.sb_source(sbf.sb_bars(rows, trading_day=td))
    res = _result(src, _params())
    assert res.diagnostics.get("session_resets", 0) >= 1
    # the setup that started on day 1 is closed out with a session_reset reason
    assert any(a.expiry_reason == "session_reset" for a in res.audits)
    # nothing entered on day 2 from day-1 structure
    day2_start = int(sbf.sb_bars(rows, trading_day=td)["ts_event_ns"].to_numpy()[len(day1)])
    for a in res.audits:
        if a.decision_ts_ns is not None:
            assert a.decision_ts_ns < day2_start


# ==========================================================================
# L / M. time-window enforcement + DST via the calendar abstraction
# ==========================================================================
def test_L_window_blocks_a_retracement_outside_the_window():
    # window 09:00-10:00 America/Chicago; place the whole fixture at 10:30 CT.
    late_start = sbf.BASE_NS + 90 * sbf.MIN_NS
    src_in = sbf.bullish_setup_source()
    src_out = sbf.sb_source(sbf.sb_bars(sbf.bullish_setup_rows(), start_ns=late_start))
    p = _params(window_start_local="09:00", window_end_local="10:00", window_calendar="NQ")
    assert _result(src_in, p).diagnostics.get("entries", 0) == 1
    out = _result(src_out, p)
    assert out.diagnostics.get("entries", 0) == 0
    assert out.diagnostics.get("retracement_blocked_by_window", 0) >= 1


def test_M_window_is_dst_safe_through_the_calendar():
    import pandas as pd

    cal = SessionCalendar.from_config()
    # 10:30 local Chicago: summer (CDT, UTC-5) -> 15:30Z ; winter (CST, UTC-6) -> 16:30Z
    s = pd.Timestamp("2026-07-15T15:30:00Z").value
    w = pd.Timestamp("2026-01-15T16:30:00Z").value
    assert cal.in_local_window(s, "NQ", "10:00", "11:00")
    assert cal.in_local_window(w, "NQ", "10:00", "11:00")
    # same wall-clock, different UTC hour -> proves it is not a fixed offset
    assert pd.Timestamp(s, unit="ns").hour != pd.Timestamp(w, unit="ns").hour
    with pytest.raises(ValueError):
        cal.in_local_window(s, "NQ", "11:00", "10:00")  # end <= start


# ==========================================================================
# N. no intrabar sequence invention -- one bar cannot hold the whole setup
# ==========================================================================
def test_N_single_bar_cannot_produce_a_setup():
    # one huge bar that sweeps the low, closes strongly up, and whose range also
    # spans a would-be FVG + retracement. Bar separation is required, so: no entry.
    rows = [sbf._FLAT] * 12 + [
        (100.0, 112.0, 90.0, 111.0),   # sweep + displacement geometry in ONE bar
        (111.0, 111.5, 103.0, 104.0),
        (104.0, 104.5, 103.5, 104.0),
    ] + [sbf._FLAT] * 6
    res = _result(sbf.sb_source(sbf.sb_bars(rows)), _params())
    # a sweep may be observed, but displacement on the SAME bar is never accepted
    assert res.diagnostics.get("entries", 0) == 0


# ==========================================================================
# O. signed-price compatibility (a CL path through zero)
# ==========================================================================
def test_O_signed_prices_do_not_break_the_detector():
    base = [(-2.0, -1.5, -2.5, -2.0)] * 12
    block = [
        (-2.0, -1.0, -4.0, -1.2),      # sell-side sweep below prior low -2.5, close back above
        (-1.2, 5.0, -1.5, 4.5),        # bullish displacement across zero
        (4.5, 6.0, 1.0, 5.0),          # FVG: low 1.0 > sweep-bar high -1.0
        (5.0, 5.0, 0.0, 0.5),          # retracement into the gap
    ]
    tail = [(0.5, 1.0, 0.0, 0.5)] * 6
    src = sbf.sb_source(sbf.sb_bars(base + block + tail), root="CL", raw="CLF7")
    res = _result(src, _params(displacement_atr_multiple=1.0, root_symbol="CL"))
    assert np.all(np.isfinite(res.intent[~np.isnan(res.intent)]))
    assert res.diagnostics.get("entries", 0) >= 1  # geometry works across zero


# ==========================================================================
# P. a large futures-roll basis does not create a false setup / execution price
# ==========================================================================
def test_P_large_roll_basis_does_not_manufacture_a_setup():
    n1, n2 = 14, 12
    rows = [sbf._FLAT] * n1 + [(150.0 + 0.1 * i, 150.6 + 0.1 * i, 149.4 + 0.1 * i, 150.0 + 0.1 * i)
                               for i in range(n2)]
    roll = [False] * n1 + [True] + [False] * (n2 - 1)  # +50 basis jump at the roll
    td = ["2026-09-01"] * (n1 + n2)
    frame = sbf.sb_bars(rows, trading_day=td, is_roll_boundary=roll)
    src = sbf.sb_source(frame, domain=PriceDomain.RAW_CONTINUOUS)
    res = _result(src, _params())
    roll_i = n1
    # no entry is generated on the roll bar or the bar right after it from the
    # artificial +50 gap; the detector resets its state at the boundary.
    assert res.intent[roll_i] == 0.0 or np.isnan(res.intent[roll_i])
    assert res.diagnostics.get("roll_resets", 0) >= 1
    assert res.diagnostics.get("entries", 0) == 0


# ==========================================================================
# Q / R. PIT back-adjusted may drive a Signal; retrospective is rejected
# ==========================================================================
def test_R_retrospective_backadjusted_source_is_rejected():
    rs = synthetic_roll_sources()
    spec = make_silver_bullet_spec(_params(liquidity_lookback=2, displacement_atr_window=2))
    plan = compile_strategy(spec)
    retro = compute_features(
        rs["backadj_retro"], [b.spec for b in plan.feature_bindings], require_point_in_time=False
    )
    with pytest.raises(StrategyEvaluationError):
        build_target_schedule(plan, retro)


def test_Q_pit_backadjusted_frame_can_drive_the_strategy():
    rs = synthetic_roll_sources()
    spec = make_silver_bullet_spec(_params(liquidity_lookback=2, displacement_atr_window=2))
    plan = compile_strategy(spec)
    pit = compute_features(
        rs["backadj_pit"], [b.spec for b in plan.feature_bindings], require_point_in_time=False
    )
    s1 = build_target_schedule(plan, pit)
    s2 = build_target_schedule(plan, pit)
    assert s1.schedule_hash() == s2.schedule_hash()  # signal-safe + deterministic


# ==========================================================================
# S / T / U. factory -> valid StrategySpec; compiler / evaluator / schedule
#            deterministic
# ==========================================================================
def test_S_factory_returns_a_valid_strategyspec():
    spec = make_silver_bullet_spec(SILVER_BULLET_BENCHMARK)
    assert spec.schema_version == "strategy-dsl/1"
    assert spec.root_symbol == "NQ"
    assert len(spec.features) == 1 and spec.features[0].spec.kind == "silver_bullet_intent"
    assert spec.default_action.value == "flat"
    for decl in spec.features:
        assert decl.spec.kind in REGISTRY.kinds()
    # the benchmark param set is one canonical set; grid ranges are pre-declared
    assert set(SILVER_BULLET_GRID_RANGES) and SILVER_BULLET_FAMILY.is_claimed_alpha is False


def test_S_invalid_params_rejected():
    with pytest.raises(ValueError):
        SilverBulletParams(root_symbol="nq")  # bad root pattern
    with pytest.raises(ValueError):
        _params(liquidity_reference="centered_swing")  # not an allowed reference
    with pytest.raises(ValueError):
        _params(displacement_atr_multiple=0.0)  # must be > 0
    with pytest.raises(ValueError):
        _params(window_start_local="10:00")  # start without end
    with pytest.raises(ValueError):
        _params(size=0)
    with pytest.raises(ValueError):
        _params(distance_unit="pips")  # not an allowed distance unit
    assert set(LIQUIDITY_REFERENCES) == {"rolling_nbar_extreme", "prior_session_extreme"}
    assert set(DISTANCE_UNITS) == {"price_units", "ticks"}


def test_T_compiler_and_evaluator_deterministic():
    spec = make_silver_bullet_spec(_params())
    p1, p2 = compile_strategy(spec), compile_strategy(spec)
    assert p1.fingerprint == p2.fingerprint
    assert p1.canonical_payload == p2.canonical_payload
    src = sbf.bullish_setup_source()
    frame = compute_features(src, [b.spec for b in p1.feature_bindings])
    from alpha_agent.strategy.evaluator import ReferenceEvaluator

    d1 = ReferenceEvaluator(p1).evaluate_frame(frame)
    d2 = ReferenceEvaluator(p1).evaluate_frame(frame)
    assert [x.model_dump() for x in d1] == [x.model_dump() for x in d2]


def test_U_target_schedule_deterministic_and_warmup_is_no_decision():
    spec = make_silver_bullet_spec(_params())
    plan = compile_strategy(spec)
    src = sbf.bullish_setup_source()
    frame = compute_features(src, [b.spec for b in plan.feature_bindings])
    s1 = build_target_schedule(plan, frame)
    s2 = build_target_schedule(plan, frame)
    assert s1.schedule_hash() == s2.schedule_hash()
    ts = frame.identifiers["ts_event_ns"].to_numpy()
    scheduled = {r.ts_event_ns for r in s1.rows}
    # the first bars (no liquidity reference yet) are NO DECISION -> no row
    assert int(ts[0]) not in scheduled
    assert len(s1.rows) < len(ts)
    # a real long entry row exists, and it is +size
    assert any(r.target_units == SILVER_BULLET_BENCHMARK.size for r in s1.rows)


# ==========================================================================
# V / W / X / Y / Z. end-to-end through the SAME Phase 11 C++ bridge
# ==========================================================================
@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_V_decision_at_T_executes_at_the_next_bar_not_T(tmp_path):
    from alpha_agent.backtest import TargetSchedule
    from alpha_agent.backtest.targets import TargetScheduleRow

    spec = make_silver_bullet_spec(_params())
    plan = compile_strategy(spec)
    src = sbf.bullish_setup_source()
    bars = sbf.boundary_bars_from_frame(src.frame, instrument_id=920001)
    contracts = sbf.contracts_one(instrument_id=920001)
    fp = plan.fingerprint
    last_ts = int(bars["ts_event_ns"].to_numpy()[-1])
    # the ONLY decision is on the final bar -> there is no next bar -> no fill,
    # proving the Python decision is not executed on its own bar.
    sched = TargetSchedule(
        root_symbol="NQ", strategy_fingerprint=fp, strategy_id=plan.strategy_id,
        strategy_dsl_version=plan.dsl_version, feature_engine_version="0.9.2", warmup_bars=0,
        rows=(TargetScheduleRow(ts_event_ns=last_ts, root_symbol="NQ", target_units=1,
                                strategy_fingerprint=fp),),
    )
    res = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path)
    assert res["target_rows_applied"] == 1
    assert res["fills"] == 0


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_WXYZ_official_pnl_is_cpp_fill_derived_and_replay_is_deterministic(tmp_path):
    spec = make_silver_bullet_spec(_params())
    plan = compile_strategy(spec)
    src = sbf.bullish_setup_source(tail=12)
    frame = compute_features(src, [b.spec for b in plan.feature_bindings])
    sched = build_target_schedule(plan, frame)
    bars = sbf.boundary_bars_from_frame(src.frame, instrument_id=921001)
    contracts = sbf.contracts_one(instrument_id=921001)

    r1 = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path / "a")
    r2 = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path / "b")
    assert r1 == r2  # Z. deterministic full replay
    assert r1["strategy_fingerprint"] == plan.fingerprint
    assert r1["fills"] >= 1  # W. RawContract fills only (bridge guarantee)
    # X. official PnL is Fill-derived in C++: net == gross - costs, verbatim
    assert abs(r1["net_pnl_usd"] - (r1["gross_pnl_usd"] - r1["costs_usd"])) < 1e-6
    # Y. the C++ risk gate is in the path (reference CLI runs PassThrough)
    assert "risk_rejects" in r1 and "risk_resizes" in r1

    summ = summarize_cpp_result(
        r1, strategy_fingerprint=plan.fingerprint, root_symbol="NQ",
        date_range_ns=sched.date_range_ns,
    )
    assert summ.official_pnl_source == "cpp_fill_events"
    assert summ.net_realized_pnl_usd == r1["net_pnl_usd"]

    manifest = build_run_manifest(
        spec=spec, plan=plan, frame=frame, schedule=sched, bars=bars, contracts=contracts
    )
    assert manifest.target_schedule_hash == sched.schedule_hash()
    assert manifest.strategy_fingerprint == plan.fingerprint


# ==========================================================================
# audit record (section 20) -- typed, deterministic, NO fill / risk fields
# ==========================================================================
def test_audit_record_has_no_execution_fields():
    res = _result(sbf.bullish_setup_source(), _params())
    a = res.audits[0]
    dumped = a.model_dump()
    forbidden = {"fill_price", "fill", "execution_price", "risk_decision", "pnl", "order", "quantity"}
    assert forbidden.isdisjoint(dumped)
    assert dumped["parameter_fingerprint"].startswith("sbparams1:")
    # same params -> same fingerprint (spec identity)
    b = _result(sbf.bullish_setup_source(), _params()).audits[0]
    assert a.parameter_fingerprint == b.parameter_fingerprint
    assert SilverBulletState.RETRACEMENT_TRIGGERED == 4


# ==========================================================================
# same pipeline as the Phase 11 baselines (section 21) -- no second backtester
# ==========================================================================
def test_uses_the_same_target_bridge_as_baselines():
    from alpha_agent.backtest.targets import TARGET_SCHEDULE_COLUMNS

    spec = make_silver_bullet_spec(_params())
    plan = compile_strategy(spec)
    frame = compute_features(
        sbf.bullish_setup_source(), [b.spec for b in plan.feature_bindings]
    )
    sched = build_target_schedule(plan, frame)
    assert list(sched.to_dataframe().columns) == list(TARGET_SCHEDULE_COLUMNS)
    for r in sched.rows:
        assert r.root_symbol == "NQ"  # bare root; C++ resolves the real contract
        assert r.strategy_fingerprint == plan.fingerprint


# ==========================================================================
# the registered state column is a real FeatureFrame output (section 9)
# ==========================================================================
def test_state_feature_column_is_registered_and_causal():
    p = _params()
    spec = FeatureSpec(kind="silver_bullet_state", params=p.feature_params())
    frame = compute_features(sbf.bullish_setup_source(), [spec])
    col = frame.features[frame.feature_names[0]].to_numpy()
    present = col[~np.isnan(col)]
    assert set(np.unique(present)).issubset({float(s) for s in SilverBulletState})
    # RETRACEMENT_TRIGGERED (in a position) appears while the setup is live
    assert float(SilverBulletState.RETRACEMENT_TRIGGERED) in set(present)
    assert frame.safety[frame.feature_names[0]].signal_safe  # causal / point-in-time


def test_benchmark_window_spec_runs_end_to_end():
    spec = make_silver_bullet_spec(SILVER_BULLET_BENCHMARK)
    plan = compile_strategy(spec)
    src = sbf.bullish_setup_source(preamble=22, tail=10)  # 09:00 CT start -> in window
    frame = compute_features(src, [b.spec for b in plan.feature_bindings])
    s1 = build_target_schedule(plan, frame)
    assert s1.schedule_hash() == build_target_schedule(plan, frame).schedule_hash()
    assert any(r.target_units == SILVER_BULLET_BENCHMARK.size for r in s1.rows)


# ==========================================================================
# Phase 12.1 -- explicit distance-unit semantics (no silent tick_size -> 1.0)
# ==========================================================================
def _detector_params(**over) -> SilverBulletDetectorParams:
    base = {
        "liquidity_lookback": 6, "liquidity_reference": "rolling_nbar_extreme",
        "sweep_penetration": 0.0, "distance_unit": "price_units", "tick_size": 0.0,
        "displacement_atr_window": 3, "displacement_atr_multiple": 1.5,
        "displacement_close_loc": 0.6, "fvg_min_width": 0.0, "retracement_fraction": 0.5,
        "setup_expiry_bars": 8, "max_holding_bars": 5,
        "window_start_local": "", "window_end_local": "", "window_calendar": "",
    }
    return SilverBulletDetectorParams(**{**base, **over})


def test_12_1_A_nq_one_tick_uses_supplied_tick_size_not_one():
    p = _detector_params(distance_unit="ticks", tick_size=0.25, sweep_penetration=1.0)
    assert p.unit_price == 0.25
    assert p.sweep_penetration_price == pytest.approx(0.25)   # 1 tick, NOT 1.0
    p2 = _detector_params(distance_unit="ticks", tick_size=0.25, sweep_penetration=4.0)
    assert p2.sweep_penetration_price == pytest.approx(1.0)


def test_12_1_B_cl_one_tick_uses_its_supplied_tick_size():
    p = _detector_params(distance_unit="ticks", tick_size=0.01, sweep_penetration=1.0)
    assert p.sweep_penetration_price == pytest.approx(0.01)
    assert p.fvg_min_width_price == 0.0  # fvg_min_width 0 -> 0 regardless of unit


def test_12_1_C_missing_tick_size_under_ticks_fails_loudly():
    with pytest.raises(ValueError, match="tick_size"):
        _detector_params(distance_unit="ticks", tick_size=0.0)
    with pytest.raises(ValueError, match="tick_size"):
        _detector_params(distance_unit="ticks", tick_size=float("nan"))
    with pytest.raises(ValueError):
        SilverBulletParams(root_symbol="NQ", distance_unit="ticks")  # no tick_size
    with pytest.raises(ValueError):
        make_silver_bullet_spec(
            SilverBulletParams(root_symbol="NQ", distance_unit="ticks", tick_size=-1.0)
        )


def test_12_1_D_price_units_needs_no_tick_metadata():
    p = _detector_params(distance_unit="price_units", sweep_penetration=0.5)
    assert p.unit_price == 1.0
    assert p.sweep_penetration_price == pytest.approx(0.5)
    # a non-zero tick_size under price_units is a loud error (no ambiguity)
    with pytest.raises(ValueError):
        _detector_params(distance_unit="price_units", tick_size=0.25)


def test_12_1_E_ticks_vs_price_units_have_different_fingerprints():
    ticks = _detector_params(distance_unit="ticks", tick_size=1.0, sweep_penetration=1.0)
    price = _detector_params(distance_unit="price_units", sweep_penetration=1.0)
    assert ticks.fingerprint() != price.fingerprint()  # 1 tick != 1.0 price unit
    # and it propagates to the StrategySpec fingerprint Phase 13 keys on
    s_ticks = make_silver_bullet_spec(
        SilverBulletParams(root_symbol="NQ", distance_unit="ticks", tick_size=1.0,
                           sweep_penetration=1.0)
    )
    s_price = make_silver_bullet_spec(
        SilverBulletParams(root_symbol="NQ", distance_unit="price_units",
                           sweep_penetration=1.0)
    )
    assert s_ticks.fingerprint != s_price.fingerprint


def test_12_1_F_fvg_min_width_obeys_the_same_unit_semantics():
    # gap in the bullish fixture is 2.5 price units. fvg_min_width = 4 ticks @ 0.25
    # tick -> 1.0 price -> passes; = 12 ticks -> 3.0 price -> fails.
    src = sbf.bullish_setup_source()

    def _run(fvg_min_width_ticks: float):
        p = SilverBulletParams(
            root_symbol="NQ", liquidity_lookback=6, displacement_atr_window=3,
            displacement_atr_multiple=1.5, displacement_close_loc=0.6,
            retracement_fraction=0.5, setup_expiry_bars=8, max_holding_bars=5,
            distance_unit="ticks", tick_size=0.25, fvg_min_width=fvg_min_width_ticks,
        )
        return silver_bullet_result(src, p.feature_params())

    assert _run(4.0).diagnostics.get("entries", 0) == 1           # 1.0 price -> ok
    too_wide = _run(12.0)                                         # 3.0 price -> fails
    assert too_wide.diagnostics.get("fvg_too_small", 0) >= 1
    assert too_wide.diagnostics.get("entries", 0) == 0


# ==========================================================================
# Phase 12.1 -- prior_session_extreme means PRIOR SESSION (no gap-segment alias)
# ==========================================================================
def test_12_1_G_prior_session_extreme_with_real_session_labels():
    rows, td = sbf.prior_session_rows()
    src = sbf.sb_source(sbf.sb_bars(rows, trading_day=td))
    res = silver_bullet_result(
        src,
        SilverBulletParams(root_symbol="NQ", liquidity_lookback=3,
            liquidity_reference="prior_session_extreme", displacement_atr_window=3,
            displacement_atr_multiple=1.0, displacement_close_loc=0.5,
            retracement_fraction=0.5, setup_expiry_bars=8, max_holding_bars=5).feature_params(),
    )
    assert res.diagnostics.get("setups_started", 0) >= 1
    a = res.audits[0]
    assert a.liquidity_reference_type == "prior_session_extreme"
    assert a.liquidity_level == pytest.approx(95.0)  # day-1 low, not any gap segment
    assert a.setup_direction == "long"


def test_12_1_H_prior_session_extreme_without_labels_fails_loudly():
    rows, _td = sbf.prior_session_rows()
    src = sbf.sb_source(sbf.sb_bars(rows))  # NO trading_day column
    spec = SilverBulletParams(
        root_symbol="NQ", liquidity_lookback=3,
        liquidity_reference="prior_session_extreme", displacement_atr_window=3,
        displacement_atr_multiple=1.0,
    ).feature_params()
    with pytest.raises(SilverBulletSessionLabelsMissing):
        silver_bullet_result(src, spec)
    # and through the normal feature-compute path
    plan = compile_strategy(make_silver_bullet_spec(SilverBulletParams(
        root_symbol="NQ", liquidity_lookback=3,
        liquidity_reference="prior_session_extreme", displacement_atr_window=3,
        displacement_atr_multiple=1.0,
    )))
    with pytest.raises(SilverBulletSessionLabelsMissing):
        compute_features(src, [b.spec for b in plan.feature_bindings])


def test_12_1_no_silent_gap_segment_substitution():
    # a large intrabar gap (would create a new gap-segment) is NOT a "prior
    # session" -- with labels present, only the trading_day boundary counts.
    rows, _td = sbf.prior_session_rows()
    # force all bars onto ONE trading day but with a big time gap mid-series
    ts = list(sbf.BASE_NS + np.arange(6) * sbf.MIN_NS)
    ts += list(ts[-1] + 500 * sbf.MIN_NS + np.arange(len(rows) - 6) * sbf.MIN_NS)
    one_day = ["2026-09-01"] * len(rows)
    src = sbf.sb_source(sbf.sb_bars(rows, ts=np.asarray(ts, dtype="int64"), trading_day=one_day))
    res = silver_bullet_result(
        src,
        SilverBulletParams(root_symbol="NQ", liquidity_lookback=3,
            liquidity_reference="prior_session_extreme", displacement_atr_window=3,
            displacement_atr_multiple=1.0).feature_params(),
    )
    # only one trading day -> there is NO prior session -> no reference -> no setup
    assert res.diagnostics.get("setups_started", 0) == 0
    assert res.diagnostics.get("bars_without_liquidity_reference", 0) >= 1
