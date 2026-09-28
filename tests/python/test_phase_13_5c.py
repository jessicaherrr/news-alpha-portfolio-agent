"""Phase 13.5C -- real-market research + validation: tooling tests.

No real market performance is inspected here; these prove the offline
reconstitution helpers, the frozen candidate manifest, the causal
forward-adjusted signal series, and the additive ``oos_split_role`` engine
parameter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import validation_fixtures as vf
from alpha_agent.data.acquisition import HoldoutViolation, continuous_request
from alpha_agent.data.backadjust import (
    build_forward_adjusted_series,
    reject_adjusted_execution,
)
from alpha_agent.data.real_market_dataset import (
    HOLDOUT_START_NS,
    MissingRawArtifact,
    assert_no_holdout_ts,
    assert_offline_window,
    daily_signal_series,
    degraded_trading_days,
    load_stored_ohlcv_or_fail,
)
from alpha_agent.schemas.market_data import RollEvent
from alpha_agent.strategy import strategy_fingerprint
from alpha_agent.strategy.baselines import TsmomParams, make_tsmom_spec
from alpha_agent.strategy.candidates_phase_13_5c import (
    RISK_ASSUMPTIONS,
    assert_no_performance_fields,
    build_candidate_manifest,
    freeze_candidate_manifest,
)
from alpha_agent.validation import (
    DatasetIdentity,
    DslFamilyAdapter,
    MinimumSampleRequirements,
    ReliabilityPolicy,
    SplitPlan,
    SplitRole,
    SplitWindow,
    ValidationEngine,
    ValidationSpec,
    Verdict,
    WalkForwardConfig,
)
from alpha_agent.validation.bootstrap import BootstrapConfig
from alpha_agent.validation.cost_stress import CostScenario, CostStressPlan
from alpha_agent.validation.multiple_testing import TrialRecord
from alpha_agent.validation.nulls import NullTestConfig
from alpha_agent.validation.phase_13_5c_trials import (
    CrossMarketIsNotATrial,
    DuplicateTrialIdentity,
    build_global_trial_family,
    trial_semantic_identity,
    unique_trial_identities,
)

CLI = Path("build/cpp/cpp/quant_backtest_targets_csv")
DAY = vf.DAY_NS
B = vf.BASE_NS
BIG_EXP = B + 4000 * DAY


# ======================================================================
# candidate manifest
# ======================================================================
def test_candidate_manifest_is_21_canonical_trials_and_deterministic():
    m1 = build_candidate_manifest()
    m2 = build_candidate_manifest()
    assert m1.n_canonical_trials() == 21
    assert m1.manifest_fingerprint() == m2.manifest_fingerprint()
    # 20 baseline (4 families x 5 roots) + 1 Silver Bullet (NQ)
    fam_root = sorted((t.family_key, t.root_symbol) for t in m1.trials)
    assert ("silver_bullet", "NQ") in fam_root
    assert sum(1 for f, _ in fam_root if f == "silver_bullet") == 1


def test_silver_bullet_non_nq_roots_are_not_evaluated_with_typed_reason():
    m = build_candidate_manifest()
    ne = {d["root_symbol"]: d["reason"] for d in m.not_evaluated}
    assert set(ne) == {"ES", "CL", "GC", "ZN"}
    assert set(ne.values()) == {"benchmark_not_predeclared_for_root"}
    assert all(t.root_symbol == "NQ" for t in m.trials if t.family_key == "silver_bullet")


def test_candidate_manifest_has_no_performance_fields():
    assert_no_performance_fields(build_candidate_manifest())


# ======================================================================
# Part 1.1 -- unique multiple-testing trial identity
# ======================================================================
def test_unique_bh_fdr_trial_family_is_107_canonical_plus_neighbours_only():
    m = build_candidate_manifest()
    b = unique_trial_identities(m)
    assert b.unique_trial_count == 107
    assert b.canonical == 21
    assert b.neighbour == 86
    assert b.ablation == 0
    # cross-market roots are the SAME per-root canonical results -> not trials
    assert b.cross_market_excluded == 20
    assert len(set(b.identities)) == len(b.identities) == 107
    # the manifest carries the same recomputed breakdown
    assert m.bh_fdr_trial_family["unique_trial_count"] == 107
    assert m.bh_fdr_trial_family["data_quality_sensitivity_in_denominator"] is False


def test_canonical_root_result_appears_exactly_once_in_the_global_family():
    m = build_candidate_manifest()
    root_of = {t.strategy_fingerprint: t.root_symbol for t in m.trials}
    for t in m.trials:
        for fp in t.neighbour_fingerprints:
            root_of.setdefault(fp, t.root_symbol)
    # canonical + neighbour records (as the run would emit them), plus a
    # cross-market record that re-references an existing canonical result
    recs = []
    for t in m.trials:
        recs.append(TrialRecord(label=f"canon:{t.family_key}/{t.root_symbol}", role="canonical",
                                strategy_fingerprint=t.strategy_fingerprint, p_value=0.5, null_tested=True))
        for i, fp in enumerate(t.neighbour_fingerprints):
            recs.append(TrialRecord(label=f"n{i}", role="neighbour", strategy_fingerprint=fp,
                                    p_value=1.0, null_tested=True))
    fam = build_global_trial_family(recs, root_of=root_of)
    assert fam.n_trials() == 107
    # every canonical fingerprint present exactly once
    from collections import Counter

    c = Counter(tr.strategy_fingerprint for tr in fam.trials)
    for t in m.trials:
        assert c[t.strategy_fingerprint] == 1


def test_cross_market_record_is_refused_by_the_global_family():
    m = build_candidate_manifest()
    t0 = m.trials[0]
    with pytest.raises(CrossMarketIsNotATrial):
        build_global_trial_family([
            TrialRecord(label="root_ES", role="cross_market",
                        strategy_fingerprint=t0.strategy_fingerprint, p_value=1.0),
        ])


def test_duplicate_semantic_identity_with_conflicting_p_raises():
    fp = "stratdsl1:" + "a" * 64
    with pytest.raises(DuplicateTrialIdentity):
        build_global_trial_family([
            TrialRecord(label="x", role="canonical", strategy_fingerprint=fp, p_value=0.01, null_tested=True),
            TrialRecord(label="y", role="neighbour", strategy_fingerprint=fp, p_value=0.90, null_tested=True),
        ], root_of={fp: "NQ"})


def test_data_quality_sensitivity_does_not_grow_the_family():
    m = build_candidate_manifest()
    root_of = {t.strategy_fingerprint: t.root_symbol for t in m.trials}
    recs = [TrialRecord(label=f"c{i}", role="canonical", strategy_fingerprint=t.strategy_fingerprint,
                        p_value=0.4, null_tested=True) for i, t in enumerate(m.trials)]
    base = build_global_trial_family(recs, root_of=root_of).n_trials()
    # the sensitivity rerun emits the SAME identities -> deduped away
    with_sens = build_global_trial_family(recs + recs, root_of=root_of).n_trials()
    assert with_sens == base == 21


def test_neighbour_variants_do_grow_the_family():
    m = build_candidate_manifest()
    t = m.trials[0]
    root_of = {t.strategy_fingerprint: t.root_symbol}
    for fp in t.neighbour_fingerprints:
        root_of[fp] = t.root_symbol
    only_canon = build_global_trial_family(
        [TrialRecord(label="c", role="canonical", strategy_fingerprint=t.strategy_fingerprint,
                     p_value=0.3, null_tested=True)], root_of=root_of).n_trials()
    with_neighbours = build_global_trial_family(
        [TrialRecord(label="c", role="canonical", strategy_fingerprint=t.strategy_fingerprint,
                     p_value=0.3, null_tested=True)]
        + [TrialRecord(label=f"n{i}", role="neighbour", strategy_fingerprint=fp, p_value=1.0, null_tested=True)
           for i, fp in enumerate(t.neighbour_fingerprints)],
        root_of=root_of).n_trials()
    assert with_neighbours == only_canon + len(t.neighbour_fingerprints)


def test_dsr_effective_trial_input_is_the_unique_family_size():
    m = build_candidate_manifest()
    root_of = {t.strategy_fingerprint: t.root_symbol for t in m.trials}
    for t in m.trials:
        for fp in t.neighbour_fingerprints:
            root_of.setdefault(fp, t.root_symbol)
    recs = []
    for t in m.trials:
        recs.append(TrialRecord(label="c", role="canonical", strategy_fingerprint=t.strategy_fingerprint,
                                p_value=0.5, null_tested=True, observed_daily_sharpe=0.05))
        for fp in t.neighbour_fingerprints:
            recs.append(TrialRecord(label="n", role="neighbour", strategy_fingerprint=fp,
                                    p_value=1.0, null_tested=True, observed_daily_sharpe=0.04))
    fam = build_global_trial_family(recs, root_of=root_of)
    # assemble_report feeds deflated_sharpe_ratio(n_trials=family.n_trials())
    assert fam.n_trials() == unique_trial_identities(m).unique_trial_count == 107


def test_trial_semantic_identity_is_role_free_and_deterministic():
    a = trial_semantic_identity(strategy_fingerprint="fp", root_symbol="NQ", schedule_hash="h")
    b = trial_semantic_identity(strategy_fingerprint="fp", root_symbol="NQ", schedule_hash="h")
    c = trial_semantic_identity(strategy_fingerprint="fp", root_symbol="ES", schedule_hash="h")
    assert a == b and a != c


def test_candidate_manifest_records_the_actual_cpp_risk_manager():
    # prompt correction 6: the frozen CLI uses PassThroughRiskManager.
    src = Path("cpp/apps/backtest_targets_csv.cpp").read_text()
    assert "PassThroughRiskManager" in src
    assert "MaxContractsRiskManager" not in src
    assert RISK_ASSUMPTIONS["risk_manager"] == "PassThroughRiskManager"
    assert RISK_ASSUMPTIONS["hard_risk_limits_enforced"] is False
    assert build_candidate_manifest().risk_assumptions["risk_manager"] == "PassThroughRiskManager"


def test_freeze_candidate_manifest_is_byte_identical(tmp_path):
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    freeze_candidate_manifest(str(p1))
    freeze_candidate_manifest(str(p2))
    assert p1.read_bytes() == p2.read_bytes()


def test_a_canonical_param_change_moves_the_manifest_fingerprint(monkeypatch):
    import alpha_agent.strategy.candidates_phase_13_5c as C

    base = build_candidate_manifest().manifest_fingerprint()
    monkeypatch.setitem(C.CANONICAL_PARAMS, "tsmom",
                        {"fast_horizon": 21, "slow_horizon": 120, "size": 1})
    assert build_candidate_manifest().manifest_fingerprint() != base


# ======================================================================
# causal forward-adjusted signal series
# ======================================================================
def _one_roll_continuous(gap: float = 10.0):
    # old contract closes 100,101,102 ; new (raw) contract closes 102+gap.. so
    # the de-based continuation is exactly 102,103,104 (forward-adjust -> flat step).
    close = [100, 101, 102, 102 + gap, 103 + gap, 104 + gap, 105 + gap]
    bars = pd.DataFrame({
        "ts_event_ns": [10, 20, 30, 40, 50, 60, 70],
        "open": close, "high": [c + 0.5 for c in close], "low": [c - 0.5 for c in close],
        "close": close, "volume": [5] * 7,
    })
    rolls = [RollEvent(
        continuous_symbol="NQ.v.0", effective_ts_ns=40,
        from_instrument_id=1, to_instrument_id=2,
        from_raw_symbol="NQH0", to_raw_symbol="NQM0", additive_gap=gap,
    )]
    return bars, rolls


def test_forward_adjust_removes_the_roll_jump_but_raw_continuous_has_it():
    bars, rolls = _one_roll_continuous(10.0)
    fwd, _ = build_forward_adjusted_series(bars, rolls, continuous_symbol="NQ.v.0")
    # raw continuous close jumps +10 across the roll (102 -> 112); forward-adjusted is flat
    assert bars["close"].diff().iloc[3] == pytest.approx(10.0)
    assert fwd["close"].diff().iloc[3] == pytest.approx(0.0)
    assert (fwd["adjustment_mode"] == "forward_adjusted").all()


def test_forward_adjust_is_prefix_invariant_only_uses_past_rolls():
    bars, rolls = _one_roll_continuous(10.0)
    full, _ = build_forward_adjusted_series(bars, rolls, continuous_symbol="NQ.v.0")
    for k in (2, 4, 6):
        pre, _ = build_forward_adjusted_series(bars.iloc[:k], rolls, continuous_symbol="NQ.v.0")
        assert np.allclose(pre["close"].to_numpy(), full["close"].to_numpy()[:k])


def test_forward_adjusted_frame_is_never_executable():
    bars, rolls = _one_roll_continuous()
    fwd, _ = build_forward_adjusted_series(bars, rolls, continuous_symbol="NQ.v.0")
    assert not ({"instrument_id", "raw_symbol", "active_instrument_id"} & set(fwd.columns))
    from alpha_agent.data.diagnostics import PipelineError

    with pytest.raises(PipelineError):
        reject_adjusted_execution("back_adjusted")


def test_forward_adjusted_mode_is_accepted_by_sourceseries():
    from alpha_agent.features.source import SourceSeries
    from alpha_agent.schemas.market_data import PriceDomain

    bars, rolls = _one_roll_continuous()
    fwd, _ = build_forward_adjusted_series(bars, rolls, continuous_symbol="NQ.v.0")
    src = SourceSeries(frame=fwd.rename(columns={}), price_domain=PriceDomain.BACK_ADJUSTED,
                       interval_ns=10)
    assert src.adjustment_mode == "forward_adjusted"
    assert src.safety().signal_safe and not src.safety().execution_price_safe


# ======================================================================
# daily signal series -- one row per (root, trading_day)
# ======================================================================
def test_daily_signal_series_one_row_per_trading_day_even_across_a_roll():
    from datetime import date

    bars, _labels = vf.cme_intraday_bars(instrument_id=42, n_trading_days=8, start=date(2021, 3, 1))
    ds = daily_signal_series(bars, "NQ")
    assert len(ds) == ds["trading_day"].nunique() == 8
    assert ds["ts_event_ns"].is_monotonic_increasing
    assert set(ds["ts_event_ns"]).issubset(set(bars["ts_event_ns"]))
    assert (ds["high"] >= ds[["open", "close"]].max(axis=1)).all()
    assert (ds["low"] <= ds[["open", "close"]].min(axis=1)).all()


def test_daily_signal_stamp_never_leaks_a_future_minute():
    from datetime import date

    from alpha_agent.data.calendars import default_calendar

    bars, _ = vf.cme_intraday_bars(instrument_id=7, n_trading_days=5, start=date(2021, 3, 1))
    ds = daily_signal_series(bars, "NQ")
    cal = default_calendar()
    tday, _s = cal.classify_series(bars["ts_event_ns"], "NQ")
    for _, row in ds.iterrows():
        day_ts = bars.loc[tday.astype(str).values == row["trading_day"], "ts_event_ns"]
        assert row["ts_event_ns"] == int(day_ts.max())


# ======================================================================
# degraded-day mapping + holdout guards
# ======================================================================
def test_degraded_trading_days_is_deterministic_and_per_root():
    a = degraded_trading_days("NQ")
    b = degraded_trading_days("NQ")
    assert a == b
    assert "2020-02-27" in a and "2024-09-18" in a
    assert all(d < "2025-01-01" for d in a)
    # a different-calendar root resolves independently (may differ at the edges)
    assert isinstance(degraded_trading_days("CL"), list)


def test_holdout_guards_refuse_2025():
    with pytest.raises(HoldoutViolation):
        assert_offline_window("2024-06-01", "2025-06-01")
    with pytest.raises(HoldoutViolation):
        assert_offline_window("2025-01-01", "2025-02-01")
    with pytest.raises(HoldoutViolation):
        assert_no_holdout_ts([HOLDOUT_START_NS])
    with pytest.raises(HoldoutViolation):
        assert_no_holdout_ts([HOLDOUT_START_NS + 60_000_000_000])
    assert_no_holdout_ts([HOLDOUT_START_NS - 1])  # 2024-12-31 is fine


def test_missing_raw_artifact_fails_loudly_and_offline(monkeypatch):
    # an unrequested window has no stored artifact -> loud, no network
    req = continuous_request("NQ", "2019-03-01", "2019-03-05")
    import alpha_agent.data.databento_source as dbs

    def _boom(*a, **k):  # any network call is a bug
        raise AssertionError("Phase 13.5C reconstitution must not contact Databento")

    monkeypatch.setattr(dbs, "estimate_cost_usd", _boom, raising=False)
    monkeypatch.setattr(dbs, "fetch_and_store_raw", _boom, raising=False)
    with pytest.raises(MissingRawArtifact):
        load_stored_ohlcv_or_fail(req)


# ======================================================================
# additive engine parameter: oos_split_role
# ======================================================================
def _tsmom_adapter(root="NQ"):
    from alpha_agent.features import compute_features
    from alpha_agent.features.source import SourceSeries
    from alpha_agent.schemas.market_data import PriceDomain

    def spec_factory(p: dict):
        return make_tsmom_spec(TsmomParams(**p))

    def compute_frame(plan, bars):
        df = bars[["ts_event_ns", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        src = SourceSeries(frame=df, price_domain=PriceDomain.RAW_CONTRACT,
                           identity={"root_symbol": root, "raw_symbol": "NQU6"},
                           interval_ns=DAY)
        return compute_features(src, [b.spec for b in plan.feature_bindings])

    return DslFamilyAdapter(
        "tsmom", {"root_symbol": root, "fast_horizon": 5, "slow_horizon": 20, "size": 1},
        spec_factory, compute_frame,
    )


def _split_plan_3():
    return SplitPlan(windows=(
        SplitWindow(role=SplitRole.TRAIN, start_ts_ns=B, end_ts_ns=B + 300 * DAY),
        SplitWindow(role=SplitRole.VALIDATION, start_ts_ns=B + 300 * DAY, end_ts_ns=B + 420 * DAY),
        SplitWindow(role=SplitRole.LOCKED_HOLDOUT, start_ts_ns=B + 430 * DAY, end_ts_ns=B + 480 * DAY),
    ), embargo_days=5)


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_oos_split_role_puts_headline_on_validation_and_folds_in_train(tmp_path):
    from alpha_agent.validation.runner import CliBacktestRunner

    closes = vf.trend_closes(470)
    bars = vf.daily_bars(closes, instrument_id=7001)
    contracts = vf.one_contract(instrument_id=7001, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    adapter = _tsmom_adapter()
    pol = ReliabilityPolicy(
        policy_name="p135c-test",
        minimum_sample=MinimumSampleRequirements(min_oos_trading_days=30, min_fills=0, min_trades=0,
                                                 min_walk_forward_folds=2, min_nonzero_return_days=5),
        null_p_value_max=0.2, dsr_min=0.5, max_cost_degradation=9.0,
        fold_consistency_min=0.0, param_stability_min_fraction_positive_sharpe=0.0,
    )
    from alpha_agent.data.calendars import default_calendar

    spec = ValidationSpec(
        strategy_fingerprint=strategy_fingerprint(make_tsmom_spec(TsmomParams(**adapter.canonical_params))),
        strategy_key="tsmom",
        dataset=DatasetIdentity(root_symbol="NQ", price_domain="raw_contract",
                                source_fingerprint="fixture:135c", bars_content_hash="f" * 64,
                                n_bars=len(closes),
                                trading_day_convention=default_calendar().trading_day_convention("NQ")),
        split_plan=_split_plan_3(),
        walk_forward=WalkForwardConfig(n_folds=3, min_train_days=60, test_days=45, embargo_days=5, warmup_bars=20),
        capital_base_usd=100_000.0,
        cost_stress=CostStressPlan(scenarios=(CostScenario(label="baseline_1_0x", multiplier=1.0),)),
        null_test=NullTestConfig(n_null_samples=25, seed=0),
        bootstrap=BootstrapConfig(n_resamples=200, seed=0),
        trial_family_id="phase_13_5c.tsmom",
        reliability_policy_fingerprint=pol.identity(),
    )
    runner = CliBacktestRunner(CLI, tmp_path)
    engine = ValidationEngine(spec, pol, runner, adapter, bars, contracts,
                              oos_split_role=SplitRole.VALIDATION)
    rep = engine.run_research()
    assert rep.verdict in (Verdict.PASS, Verdict.REJECT, Verdict.INCONCLUSIVE)
    # headline OOS spans exactly the VALIDATION window, never the holdout
    assert rep.oos_daily.ts_ns[0] >= B + 300 * DAY
    assert rep.oos_daily.ts_ns[-1] < B + 420 * DAY
    # folds were built and all lie inside TRAIN
    assert rep.walk_forward.n_folds_evaluated >= 2
    for f in rep.fold_summaries:
        assert f.test_end_ts_ns <= B + 300 * DAY
    assert not rep.holdout_evaluated


# ======================================================================
# Phase 13.5C -- auxiliary roll-close marks (data plumbing for the frozen
# RejectDefer roll close-leg; the C++ side is proven in quant_engine_tests P-T)
# ======================================================================
def _fake_recon_for_marks():
    from types import SimpleNamespace

    from alpha_agent.schemas.market_data import RollEvent

    # NQM6 (id 10, expires ts 3000) -> NQU6 (id 11) roll effective at ts 1500.
    overlap = pd.DataFrame({
        "instrument_id": [10, 10, 10, 10, 11, 11],
        "ts_event_ns":   [1400, 1500, 1600, 3100, 1500, 1600],   # 3100 is past NQM6 expiry
        "open":  [100.0] * 6, "high": [101.0] * 6, "low": [99.0] * 6,
        "close": [100.5, 100.6, 100.7, 100.9, 200.1, 200.2], "volume": [5] * 6,
    })
    contracts = pd.DataFrame({
        "instrument_id": [10, 11], "raw_symbol": ["NQM6", "NQU6"], "root_symbol": ["NQ", "NQ"],
        "exchange": ["XCME", "XCME"], "tick_size": [0.25, 0.25], "multiplier": [20.0, 20.0],
        "activation_ns": [1, 1], "expiration_ns": [3000, 9000],
        "first_notice_ns": ["", ""], "last_trade_ns": ["", ""],
    })
    rolls = [RollEvent(
        continuous_symbol="NQ.v.0", effective_ts_ns=1500,
        from_instrument_id=10, to_instrument_id=11,
        from_raw_symbol="NQM6", to_raw_symbol="NQU6", additive_gap=99.5,
    )]
    return SimpleNamespace(root="NQ", overlap_bars=overlap, rolls=rolls, contracts=contracts)


def test_roll_close_marks_builder_is_deterministic_causal_and_bounded():
    from alpha_agent.data.real_market_dataset import roll_close_marks

    recon = _fake_recon_for_marks()
    a = roll_close_marks(recon, 0, 10_000)
    b = roll_close_marks(recon, 0, 10_000)
    assert list(a.columns) == ["instrument_id", "ts_event_ns", "close"]
    pd.testing.assert_frame_equal(a, b)
    # only the OUTGOING contract (id 10), only ts >= effective_ts (1500),
    # only strictly before its own expiry (3000) -> ts 1500, 1600
    assert set(a["instrument_id"]) == {10}
    assert sorted(a["ts_event_ns"]) == [1500, 1600]
    assert (a["close"] > 0).all()


def test_roll_close_marks_builder_empty_when_no_overlap():
    from types import SimpleNamespace

    from alpha_agent.data.real_market_dataset import roll_close_marks

    empty = roll_close_marks(SimpleNamespace(root="NQ", overlap_bars=None, rolls=[], contracts=pd.DataFrame()), 0, 1)
    assert empty.empty and list(empty.columns) == ["instrument_id", "ts_event_ns", "close"]


def test_run_targets_cli_rejects_roll_marks_without_validation_days(tmp_path):
    from alpha_agent.adapters.targets_bridge import run_targets_backtest_cli
    from alpha_agent.backtest.targets import TargetSchedule

    bars = vf.daily_bars(vf.trend_closes(5), instrument_id=1)
    contracts = vf.one_contract(instrument_id=1, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    fp = "stratdsl1:" + "a" * 64
    sched = TargetSchedule(root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
                           strategy_dsl_version="1.0.0", feature_engine_version="0.9.2",
                           warmup_bars=0, rows=())
    with pytest.raises(ValueError):
        run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path,
                                 roll_close_marks_path=str(tmp_path / "m.csv"))


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_roll_close_marks_end_to_end_prices_a_deferred_roll(tmp_path):
    from alpha_agent.adapters.targets_bridge import run_targets_backtest_cli
    from alpha_agent.backtest.targets import TargetSchedule, TargetScheduleRow

    m6, u6 = 10, 11
    exp6 = vf.BASE_NS + 100 * DAY
    # clean continuous-front feed: NQM6 for 3 days, then the feed rolls to NQU6.
    b6 = vf.daily_bars([100.0, 101.0, 102.0], instrument_id=m6, start_ns=vf.BASE_NS)
    b7 = vf.daily_bars([202.0, 203.0, 204.0, 205.0], instrument_id=u6,
                       start_ns=vf.BASE_NS + 3 * DAY)
    bars = pd.concat([b6, b7], ignore_index=True)
    contracts = pd.concat([
        vf.one_contract(instrument_id=m6, raw_symbol="NQM6", root="NQ", expiration_ns=exp6),
        vf.one_contract(instrument_id=u6, raw_symbol="NQU6", root="NQ", expiration_ns=exp6 + 90 * DAY),
    ], ignore_index=True)
    fp = "stratdsl1:" + "b" * 64
    ts = [int(t) for t in bars["ts_event_ns"]]
    # a dense schedule (like the daily-cadence baselines): +1 every bar, so a
    # decision lands after the feed rolls and the held-position close-leg fires.
    rows = tuple(
        TargetScheduleRow(ts_event_ns=t, root_symbol="NQ", target_units=1, strategy_fingerprint=fp)
        for t in ts[:-1]
    )
    sched = TargetSchedule(root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
                           strategy_dsl_version="1.0.0", feature_engine_version="0.9.2",
                           warmup_bars=0, rows=rows)
    vdays = tmp_path / "vd.csv"
    vdays.write_text("boundary_ts_ns\n" + "\n".join(str(t) for t in ts) + "\n")

    # (a) no marks -> the held NQM6->NQU6 roll defers (no contemporaneous NQM6 bar)
    a = run_targets_backtest_cli(bars, contracts, sched, executable=CLI,
                                 work_dir=tmp_path / "a", validation_days_path=vdays)
    assert a["rolls_deferred"] >= 1
    assert a["rolls_priced_contemporaneous"] == 0
    assert a["rolls_priced_auxiliary_marks"] == 0

    # (b) an auxiliary NQM6 close at EXACTLY the roll execution instant (ts[3],
    # the first NQU6 bar) -> contemporaneous, from the auxiliary mark
    marks = tmp_path / "marks.csv"
    marks.write_text(f"instrument_id,ts_event_ns,close\n{m6},{ts[3]},102.5\n")
    b = run_targets_backtest_cli(bars, contracts, sched, executable=CLI,
                                 work_dir=tmp_path / "b", validation_days_path=vdays,
                                 roll_close_marks_path=marks)
    assert b["rolls"] == 1
    assert b["rolls_priced_contemporaneous"] == 1
    assert b["rolls_priced_auxiliary_marks"] == 1
    assert b["rolls_deferred"] == 0
    assert b["rolls_priced_stale"] == 0
    assert 0 <= b["rolls_priced_auxiliary_marks"] <= b["rolls_priced_contemporaneous"]
    # the primary event stream / strategy loop is untouched
    assert a["bars"] == b["bars"]
    assert a["signals"] == b["signals"]
    assert a["target_rows_applied"] == b["target_rows_applied"]
    assert len(a["daily_equity"]) == len(b["daily_equity"])


# ======================================================================
# Phase 13.5C matrix -- frozen configuration consistency (no C++)
# ======================================================================
def test_matrix_frozen_config_matches_the_frozen_policy_and_manifest():
    from alpha_agent.validation import ReliabilityPolicy
    from alpha_agent.validation.phase_13_5c_matrix import (
        build_split_plan,
        frozen_bootstrap_config,
        frozen_cost_plan,
        frozen_null_config,
        frozen_policy,
        frozen_walk_forward,
    )

    # the frozen policy is byte-identical to the Phase 13 default (cosmetic name aside)
    assert frozen_policy().identity() == ReliabilityPolicy().identity()
    # cost stress = exactly 1.0x / 1.5x / 2.0x
    assert [s.label for s in frozen_cost_plan().scenarios] == ["baseline_1_0x", "stress_1_5x", "stress_2_0x"]
    # frozen null config = 199 samples, centered-block gating + schedule-shift diagnostic
    nc = frozen_null_config()
    assert nc.n_null_samples == 199
    assert frozen_bootstrap_config().n_resamples == 2000
    assert frozen_walk_forward().n_folds == 4
    # split plan: TRAIN 2018-2022 / VALIDATION 2023-2024 / one LOCKED_HOLDOUT, last
    sp = build_split_plan()
    from alpha_agent.validation import SplitRole

    assert sp.has_role(SplitRole.TRAIN) and sp.has_role(SplitRole.VALIDATION)
    assert sp.windows[-1].role == SplitRole.LOCKED_HOLDOUT
    hv = sp.window(SplitRole.VALIDATION)
    assert hv.end_ts_ns == pd.Timestamp("2025-01-01T00:00:00Z").value  # exclusive boundary
    assert sp.holdout_window().start_ts_ns >= hv.end_ts_ns


def test_roll_continuation_overlay_restamps_the_held_target_only():
    from alpha_agent.backtest.targets import TargetSchedule, TargetScheduleRow
    from alpha_agent.validation.phase_13_5c_matrix import _inject_roll_continuation

    fp = "stratdsl1:" + "a" * 64
    base = TargetSchedule(
        root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
        strategy_dsl_version="1.0.0", feature_engine_version="0.9.2", warmup_bars=0,
        rows=(
            TargetScheduleRow(ts_event_ns=100, root_symbol="NQ", target_units=1, strategy_fingerprint=fp),
            TargetScheduleRow(ts_event_ns=500, root_symbol="NQ", target_units=-1, strategy_fingerprint=fp),
            TargetScheduleRow(ts_event_ns=900, root_symbol="NQ", target_units=0, strategy_fingerprint=fp),
        ),
    )
    # roll at 300 -> continuation +1 (held before 300); at 700 -> held is -1;
    # at 950 -> held is 0 -> skipped; at 50 (< first row) -> skipped; at 500 (== a
    # row) -> skipped; at 1000 (> last row) -> skipped.
    out = _inject_roll_continuation(base, (50, 300, 500, 700, 950, 1000))
    added = [(r.ts_event_ns, r.target_units, r.matched_rule_id) for r in out.rows
             if r.matched_rule_id == "roll_continuation"]
    assert added == [(300, 1, "roll_continuation"), (700, -1, "roll_continuation")]
    assert [r.ts_event_ns for r in out.rows] == [100, 300, 500, 700, 900]
    # an empty schedule is returned untouched (null-test shifted-empty case)
    empty = base.model_copy(update={"rows": ()})
    assert _inject_roll_continuation(empty, (300,)).rows == ()
    # no roll ts -> identical object
    assert _inject_roll_continuation(base, ()) is base


def test_contract_economics_reference_matches_published_cme_specs():
    """The audit reference is the PUBLISHED CME contract specification -- it must
    never be edited to match whatever the derivation happens to produce."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_econ_qa", Path("scripts/phase_13_5c_contract_economics_qa.py")
    )
    mod = importlib.util.module_from_spec(spec)
    import sys as _sys

    _sys.path.insert(0, "scripts")
    spec.loader.exec_module(mod)
    ref = mod.REFERENCE
    assert set(ref) == {"ES", "NQ", "CL", "GC", "ZN"}
    # point value x tick size == the published tick value
    expected_tick_value = {"ES": 12.50, "NQ": 5.00, "CL": 10.00, "GC": 10.00, "ZN": 15.625}
    for root, r in ref.items():
        assert r["point_value_usd"] * r["tick_size"] == pytest.approx(expected_tick_value[root])


def test_report_embeds_contract_economics_qa_and_blocks_holdout_eligibility(tmp_path):
    import json as _json

    from alpha_agent.validation.phase_13_5c_report import _contract_economics_qa

    assert _contract_economics_qa(tmp_path)["status"] == "NOT_RUN"
    (tmp_path / "CONTRACT_ECONOMICS_QA.json").write_text(_json.dumps({
        "pass": False,
        "roots": [{"root": "ZN", "status": "DATA_QUALITY_FAILURE",
                   "derived_point_value_usd": 0.064, "reference_point_value_usd": 1000.0,
                   "point_value_error_factor": 15625.0},
                  {"root": "NQ", "status": "PASS"}],
        "impact_note": "x",
    }))
    q = _contract_economics_qa(tmp_path)
    assert q["status"] == "DATA_QUALITY_FAILURE"
    assert q["failed_roots"] == ["ZN"]


def test_matrix_global_family_is_the_107_unique_hypotheses():
    from alpha_agent.validation.multiple_testing import TrialRecord
    from alpha_agent.validation.phase_13_5c_matrix import GlobalFamily, frozen_policy
    from alpha_agent.validation.phase_13_5c_trials import build_global_trial_family

    m = build_candidate_manifest()
    # a fake run per canonical trial + its neighbours (schedule_hash=None, matching
    # the frozen pre-run identity enumeration)
    root_of = {t.strategy_fingerprint: t.root_symbol for t in m.trials}
    recs = []
    for t in m.trials:
        recs.append(TrialRecord(label=f"c/{t.root_symbol}", role="canonical",
                                strategy_fingerprint=t.strategy_fingerprint, p_value=0.4,
                                null_tested=True, observed_daily_sharpe=0.03))
        for i, fp in enumerate(t.neighbour_fingerprints):
            root_of.setdefault(fp, t.root_symbol)
            recs.append(TrialRecord(label=f"n{i}/{t.root_symbol}", role="neighbour",
                                    strategy_fingerprint=fp, p_value=1.0, null_tested=True,
                                    observed_daily_sharpe=0.02))
    fam = build_global_trial_family(recs, root_of=root_of)
    assert fam.n_trials() == 107
    gf = GlobalFamily(fam, frozen_policy().fdr_q_threshold, {})
    fdr = gf.fdr()
    assert fdr.n_trials == 107


def test_oos_split_role_rejects_the_holdout_window():
    adapter = _tsmom_adapter()
    pol = ReliabilityPolicy(policy_name="x")
    from alpha_agent.data.calendars import default_calendar

    spec = ValidationSpec(
        strategy_fingerprint="stratdsl1:" + "0" * 64, strategy_key="tsmom",
        dataset=DatasetIdentity(root_symbol="NQ", price_domain="raw_contract",
                                source_fingerprint="x", bars_content_hash="0" * 64, n_bars=1,
                                trading_day_convention=default_calendar().trading_day_convention("NQ")),
        split_plan=_split_plan_3(),
        walk_forward=WalkForwardConfig(),
        capital_base_usd=100_000.0,
        cost_stress=CostStressPlan(scenarios=(CostScenario(label="baseline_1_0x", multiplier=1.0),)),
        null_test=NullTestConfig(), bootstrap=BootstrapConfig(),
        trial_family_id="t", reliability_policy_fingerprint=pol.identity(),
    )
    bars = vf.daily_bars(vf.trend_closes(10), instrument_id=1)
    contracts = vf.one_contract(instrument_id=1, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    with pytest.raises(ValueError):
        ValidationEngine(spec, pol, object(), adapter, bars, contracts,
                         oos_split_role=SplitRole.LOCKED_HOLDOUT)
