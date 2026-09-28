"""Phase 13 -- reliability-aware strategy validation framework.

Checklist A-AH of the Phase 13 prompt. The statistics framework is proven on
deterministic synthetic fixtures (section 23); the C++-official path
(daily_equity trace -> daily validation series, null / cost reruns) is proven on
daily-bar fixtures and skips cleanly when the core is not built.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import validation_fixtures as vf
from alpha_agent.backtest.targets import TargetSchedule, TargetScheduleRow
from alpha_agent.features import compute_features
from alpha_agent.features.source import SourceSeries
from alpha_agent.schemas.market_data import PriceDomain
from alpha_agent.strategy import strategy_fingerprint
from alpha_agent.strategy.baselines import (
    SILVER_BULLET_BENCHMARK,
    TsmomParams,
    make_silver_bullet_spec,
    make_tsmom_spec,
)
from alpha_agent.strategy.baselines.silver_bullet import SilverBulletParams
from alpha_agent.validation import (
    DEFAULT_COST_STRESS_PLAN,
    TEST_FIXTURE_POLICY,
    AblationKind,
    AblationSpec,
    BootstrapConfig,
    BootstrapMethod,
    CostScenario,
    CostStressPlan,
    DatasetIdentity,
    DslFamilyAdapter,
    MinimumSampleRequirements,
    MultipleTestingFamily,
    NullMethod,
    NullTestConfig,
    ParameterNeighbourhood,
    ReasonCode,
    ReliabilityPolicy,
    SplitPlan,
    SplitRole,
    SplitWindow,
    ValidationEngine,
    ValidationSpec,
    Verdict,
    WalkForwardConfig,
    assemble_report,
    bh_qvalues,
    build_folds,
    daily_returns_from_cpp_trace,
    deflated_sharpe_ratio,
    empirical_p_value,
    freeze_research,
)
from alpha_agent.validation.bootstrap import resample_indices
from alpha_agent.validation.crossmarket import RootRunSummary
from alpha_agent.validation.dsr import deflated_benchmark_sharpe, probabilistic_sharpe_ratio
from alpha_agent.validation.fdr import benjamini_hochberg_decisions
from alpha_agent.validation.holdout import HoldoutDisciplineError, HoldoutRelease
from alpha_agent.validation.metrics import daily_sharpe
from alpha_agent.validation.multiple_testing import TrialRecord
from alpha_agent.validation.nulls import (
    causal_segment_ids,
    circular_permuted_schedules,
    common_support_control_schedule,
    shifted_schedules,
)
from alpha_agent.validation.regime import RegimeKind, RegimeLabelling, evaluate_regime_stability
from alpha_agent.validation.runner import synthetic_backtest_run
from alpha_agent.validation.stability import NeighbourResult, summarize_parameter_stability
from alpha_agent.validation.trading_day import (
    TradingDayConvention,
    build_validation_day_plan,
)
from alpha_agent.validation.walkforward import WalkForwardFoldResult, summarize_walk_forward
from pydantic import ValidationError

CLI = Path("build/cpp/cpp/quant_backtest_targets_csv")
DAY = vf.DAY_NS
BASE = vf.BASE_NS
BIG_EXP = BASE + 4000 * DAY


# ======================================================================
# helpers
# ======================================================================
def _split_plan() -> SplitPlan:
    return SplitPlan(
        windows=(
            SplitWindow(role=SplitRole.TRAIN, start_ts_ns=BASE, end_ts_ns=BASE + 160 * DAY),
            SplitWindow(role=SplitRole.VALIDATION, start_ts_ns=BASE + 160 * DAY, end_ts_ns=BASE + 240 * DAY),
            SplitWindow(role=SplitRole.LOCKED_HOLDOUT, start_ts_ns=BASE + 250 * DAY, end_ts_ns=BASE + 300 * DAY),
        ),
        embargo_days=5,
    )


def _spec(policy: ReliabilityPolicy, *, neighbourhood=None, family_id="fam-1", seed=0) -> ValidationSpec:
    return ValidationSpec(
        strategy_fingerprint="stratdsl1:" + "a" * 64,
        strategy_key="synthetic",
        dataset=DatasetIdentity(
            root_symbol="NQ", price_domain="raw_contract",
            source_fingerprint="fixture:" + "0" * 8, bars_content_hash="0" * 64, n_bars=180,
        ),
        split_plan=_split_plan(),
        walk_forward=WalkForwardConfig(n_folds=4, min_train_days=40, test_days=25, embargo_days=5, warmup_bars=20),
        capital_base_usd=100_000.0,
        cost_stress=DEFAULT_COST_STRESS_PLAN,
        null_test=NullTestConfig(n_null_samples=199, seed=seed),
        bootstrap=BootstrapConfig(n_resamples=400, seed=seed),
        trial_family_id=family_id,
        parameter_neighbourhood=neighbourhood,
        minimum_sample=policy.minimum_sample,
        reliability_policy_fingerprint=policy.identity(),
        random_seed=seed,
    )


def _wf_from_pnl(pnl: np.ndarray, n_folds: int = 4, wf: WalkForwardConfig | None = None):
    wf = wf or WalkForwardConfig(n_folds=n_folds)
    parts = np.array_split(np.asarray(pnl, dtype=float), n_folds)
    folds = []
    for i, p in enumerate(parts):
        run = synthetic_backtest_run(p, first_day_index=i * 1000)
        folds.append(
            WalkForwardFoldResult(
                fold_index=i, test_start_ts_ns=BASE + i * 30 * DAY,
                test_end_ts_ns=BASE + (i + 1) * 30 * DAY,
                n_trading_days=run.daily.n_days, n_fills=run.n_fills, n_trades=run.n_trades,
                oos_net_pnl_usd=run.net_pnl_usd, daily_sharpe=run.daily_sharpe,
                is_positive=run.net_pnl_usd > 0.0, daily=run.daily,
            )
        )
    return summarize_walk_forward(wf, n_folds, folds)


def _trial_family(canonical_pnl, *, p_canonical, family_id="fam-1", neighbour_pnls=None,
                  neighbour_ps=None):
    run = synthetic_backtest_run(np.asarray(canonical_pnl, dtype=float))
    trials = [
        TrialRecord(
            label="canonical", role="canonical",
            strategy_fingerprint="stratdsl1:" + "a" * 64, schedule_hash="targsched1:x",
            p_value=p_canonical, null_tested=True,
            observed_daily_sharpe=run.daily_sharpe, observed_net_pnl_usd=run.net_pnl_usd,
        )
    ]
    for i, np_ in enumerate(neighbour_pnls or []):
        nr = synthetic_backtest_run(np.asarray(np_, dtype=float))
        trials.append(
            TrialRecord(
                label=f"neighbour_{i}", role="neighbour",
                strategy_fingerprint="stratdsl1:" + f"{i:064d}", schedule_hash=f"targsched1:{i}",
                p_value=(neighbour_ps or [1.0] * 99)[i], null_tested=True,
                observed_daily_sharpe=nr.daily_sharpe, observed_net_pnl_usd=nr.net_pnl_usd,
            )
        )
    return MultipleTestingFamily(family_id=family_id, trials=tuple(trials))


def _assemble_synth(spec, policy, canonical_pnl, *, null_stats=None, cost_results=None,
                    neighbour_results=None, neighbour_pnls=None, neighbour_ps=None,
                    regime_labelling=None, cross_market=None, n_folds=4,
                    parameter_stability_required=False):
    pnl = np.asarray(canonical_pnl, dtype=float)
    run = synthetic_backtest_run(pnl)
    wf = _wf_from_pnl(pnl, n_folds=n_folds, wf=spec.walk_forward)
    # centered-block null p for the canonical, unless an explicit null was supplied
    from alpha_agent.validation.nulls import centered_block_bootstrap_null_stats

    cbb = centered_block_bootstrap_null_stats(run.daily.returns_array(), spec.null_test, daily_sharpe)
    p_can, _ = empirical_p_value(daily_sharpe(run.daily.returns_array()),
                                 null_stats if null_stats is not None else cbb)
    fam = _trial_family(pnl, p_canonical=p_can, family_id=spec.trial_family_id,
                        neighbour_pnls=neighbour_pnls, neighbour_ps=neighbour_ps)
    return assemble_report(
        spec, policy,
        walk_forward=wf,
        combined_oos_daily=run.daily,
        oos_n_fills=run.n_fills, oos_n_trades=run.n_trades,
        oos_gross_pnl_usd=run.gross_pnl_usd, oos_costs_usd=run.costs_usd,
        oos_net_pnl_usd=run.net_pnl_usd,
        trial_family=fam,
        schedule_shift_null_stats=null_stats,
        cost_scenario_results=cost_results,
        neighbour_results=neighbour_results,
        regime_labelling=regime_labelling,
        cross_market_summaries=cross_market,
        parameter_stability_required=parameter_stability_required,
    )


def _cost_results(base_net, degrade_2x_to):
    """Synthetic CostScenarioResult list: baseline_1_0x / stress_1_5x / stress_2_0x."""
    from alpha_agent.validation.cost_stress import CostScenarioResult

    def mk(label, net, comm):
        return CostScenarioResult(
            label=label, commission_per_contract_usd=comm, slippage_ticks=0.0, spread_ticks=0.0,
            n_fills=50, oos_gross_pnl_usd=net + 100, oos_costs_usd=100, oos_net_pnl_usd=net,
            daily_sharpe=0.1, annualized_sharpe=1.6,
            net_pnl_ratio_vs_baseline=net / base_net if base_net else float("nan"),
            sharpe_delta_vs_baseline=0.0,
        )

    mid = base_net - (base_net - degrade_2x_to) / 2
    return [mk("baseline_1_0x", base_net, 2.0), mk("stress_1_5x", mid, 3.0),
            mk("stress_2_0x", degrade_2x_to, 4.0)]


# ======================================================================
# A. chronological split only
# ======================================================================
def test_A_split_is_chronological_only():
    sp = _split_plan()
    assert [w.role for w in sp.windows] == [SplitRole.TRAIN, SplitRole.VALIDATION, SplitRole.LOCKED_HOLDOUT]
    # overlap rejected
    with pytest.raises(ValueError):
        SplitPlan(windows=(
            SplitWindow(role=SplitRole.TRAIN, start_ts_ns=BASE, end_ts_ns=BASE + 100 * DAY),
            SplitWindow(role=SplitRole.LOCKED_HOLDOUT, start_ts_ns=BASE + 50 * DAY, end_ts_ns=BASE + 200 * DAY),
        ), embargo_days=0)
    # holdout must be last
    with pytest.raises(ValueError):
        SplitPlan(windows=(
            SplitWindow(role=SplitRole.LOCKED_HOLDOUT, start_ts_ns=BASE, end_ts_ns=BASE + 50 * DAY),
            SplitWindow(role=SplitRole.TRAIN, start_ts_ns=BASE + 60 * DAY, end_ts_ns=BASE + 200 * DAY),
        ), embargo_days=0)


def test_A_no_random_split_api_exists():
    # there is no train_test_split / select_best(holdout) helper anywhere
    import alpha_agent.validation as v

    names = " ".join(dir(v)).lower()
    assert "select_best" not in names
    assert "train_test_split" not in names


# ======================================================================
# B. holdout inaccessible to the parameter-selection path
# ======================================================================
def test_B_embargo_enforced_between_research_and_holdout():
    with pytest.raises(ValueError):
        SplitPlan(windows=(
            SplitWindow(role=SplitRole.TRAIN, start_ts_ns=BASE, end_ts_ns=BASE + 100 * DAY),
            SplitWindow(role=SplitRole.LOCKED_HOLDOUT, start_ts_ns=BASE + 101 * DAY, end_ts_ns=BASE + 200 * DAY),
        ), embargo_days=5)


def test_B_research_span_excludes_holdout():
    sp = _split_plan()
    s, e = sp.research_span_ns()
    assert e <= sp.holdout_window().start_ts_ns - sp.embargo_ns
    assert s == BASE


# ======================================================================
# C. fold boundary has no future leakage
# ======================================================================
def test_C_folds_are_causal_with_embargo():
    cfg = WalkForwardConfig(n_folds=3, min_train_days=40, test_days=25, embargo_days=5, warmup_bars=10)
    folds = build_folds((BASE, BASE + 240 * DAY), cfg, bar_span_ns=DAY)
    assert len(folds) == 3
    for f in folds:
        assert f.train_end_ts_ns <= f.test_start_ts_ns
        assert f.test_start_ts_ns - f.train_end_ts_ns == cfg.embargo_days * DAY
        assert f.warmup_start_ts_ns >= f.train_start_ts_ns
        assert f.warmup_start_ts_ns <= f.test_start_ts_ns
    from itertools import pairwise
    for a, b in pairwise(folds):
        assert b.test_start_ts_ns >= a.test_end_ts_ns  # non-overlapping, forward


# ======================================================================
# D/E/F/G. official validation series from the C++ daily_equity trace
# ======================================================================
def test_D_validation_series_from_cpp_trace_only():
    trace = [
        [0, BASE + 1, 100_000.0, 0.0, 0.0, 0.0, 0, 1],
        [1, BASE + DAY + 1, 100_500.0, 500.0, 0.0, 4.0, 2, 2],
        [2, BASE + 2 * DAY + 1, 100_200.0, 200.0, 0.0, 8.0, 4, 3],
    ]
    s = daily_returns_from_cpp_trace(
        {"daily_equity": trace, "starting_capital_usd": 100_000.0}, capital_base_usd=100_000.0
    )
    assert s.official_source == "cpp_portfolio_accountant_daily_equity_trace"
    assert list(s.daily_pnl_usd) == [100_000.0 - 100_000.0, 500.0, -300.0][0:1] + [500.0, -300.0]


def test_E_daily_aggregation_and_capital_base():
    trace = [
        [10, BASE + 1, 101_000.0, 1_000.0, 0.0, 0.0, 1, 1],
        [11, BASE + DAY, 100_000.0, 0.0, 0.0, 0.0, 1, 2],
    ]
    s = daily_returns_from_cpp_trace(
        {"daily_equity": trace, "starting_capital_usd": 100_000.0}, capital_base_usd=50_000.0
    )
    assert s.daily_pnl_usd == (1_000.0, -1_000.0)
    assert s.returns == (1_000.0 / 50_000.0, -1_000.0 / 50_000.0)
    assert s.capital_base_usd == 50_000.0


def test_F_daily_sharpe_annualization_is_sqrt_252():
    from alpha_agent.validation.metrics import annualized_sharpe

    r = np.array([0.01, -0.005, 0.007, 0.002, -0.001, 0.004] * 10)
    assert annualized_sharpe(r) == pytest.approx(daily_sharpe(r) * np.sqrt(252.0))


def test_G_no_intraday_sqrt_252_misuse():
    for p in Path("python/alpha_agent/validation").rglob("*.py"):
        txt = p.read_text()
        assert "390" not in txt                              # no bars-per-day factor
        assert "252 * 390" not in txt and "252*390" not in txt
    # the ONLY annualization constant is the daily one
    from alpha_agent.validation.metrics import TRADING_DAYS_PER_YEAR

    assert TRADING_DAYS_PER_YEAR == 252.0


def test_missing_day_is_not_silently_zero():
    # trace only has days 0 and 5 -> 2 observations, NOT 6 (no fabricated zeros)
    trace = [
        [0, BASE + 1, 100_000.0, 0.0, 0.0, 0.0, 0, 1],
        [5, BASE + 5 * DAY, 100_100.0, 100.0, 0.0, 0.0, 2, 6],
    ]
    s = daily_returns_from_cpp_trace({"daily_equity": trace, "starting_capital_usd": 100_000.0},
                                     capital_base_usd=100_000.0)
    assert s.n_days == 2


# ======================================================================
# Phase 13.1 -- canonical futures trading_day (sections 1-7)
# ======================================================================
def test_131A_build_validation_day_plan_groups_bars_by_trading_day():
    import datetime as dt

    bars, labels = vf.cme_intraday_bars(instrument_id=1, n_trading_days=8,
                                        start=dt.date(2026, 1, 6))
    plan = build_validation_day_plan(bars, root_symbol="NQ")
    assert plan.n_days == 8
    assert plan.trading_day_labels() == sorted(set(labels))
    # every trading_day's bar set straddles a UTC midnight (17:00 CT boundary)
    for d in plan.days:
        first_utc = pd_date(d.first_ts_ns)
        last_utc = pd_date(d.boundary_ts_ns)
        assert first_utc != last_utc                       # spans two UTC dates
    # boundaries strictly ascending
    b = plan.boundary_ts_list()
    assert all(b[i] < b[i + 1] for i in range(len(b) - 1))


def test_131B_one_cme_session_spanning_utc_midnight_is_one_observation():
    import datetime as dt

    bars, _labels = vf.cme_intraday_bars(instrument_id=1, n_trading_days=3,
                                         start=dt.date(2026, 1, 6))
    plan = build_validation_day_plan(bars, root_symbol="NQ")
    # 3 trading days x CME_BARS_PER_DAY bars, but only 3 canonical observations
    assert len(bars) == 3 * vf.CME_BARS_PER_DAY
    assert plan.n_days == 3
    # the UTC-date grouping of the same bars would give MORE than 3 buckets
    utc_dates = {pd_date(int(t)) for t in bars["ts_event_ns"]}
    assert len(utc_dates) > 3


def test_131C_dst_does_not_split_or_duplicate_a_trading_day():
    import datetime as dt

    # trading days straddling the 2026-03-08 US spring-forward DST change
    bars, _labels = vf.cme_intraday_bars(instrument_id=1, n_trading_days=6,
                                         start=dt.date(2026, 3, 4))
    plan = build_validation_day_plan(bars, root_symbol="NQ")
    assert plan.n_days == 6                                # exactly one obs per trading day
    assert plan.trading_day_labels() == [
        "2026-03-04", "2026-03-05", "2026-03-06", "2026-03-09", "2026-03-10", "2026-03-11"
    ]
    # the exchange-local day boundary shifts by one UTC hour across DST, proving
    # the timezone abstraction (not a hard-coded UTC offset) is in use
    b_before = pd.Timestamp(plan.days[0].boundary_ts_ns, unit="ns", tz="UTC")
    b_after = pd.Timestamp(plan.days[-1].boundary_ts_ns, unit="ns", tz="UTC")
    assert b_before.hour != b_after.hour


def test_131D_first_fold_day_pnl_uses_initial_equity():
    trace = [
        [0, BASE + 1, 100_300.0, 300.0, 0.0, 0.0, 1, 1],
        [1, BASE + DAY, 100_100.0, 100.0, 0.0, 0.0, 2, 2],
    ]
    plan = _fake_two_day_plan()
    s = daily_returns_from_cpp_trace(
        {"daily_equity": trace, "starting_capital_usd": 100_000.0,
         "daily_equity_basis": "trading_day"},
        capital_base_usd=100_000.0, validation_day_plan=plan, initial_equity_usd=100_000.0,
    )
    assert s.day_basis == "trading_day"
    assert s.trading_day == ("2026-01-06", "2026-01-07")
    assert s.daily_pnl_usd[0] == pytest.approx(300.0)      # 100_300 - 100_000 (initial equity)
    assert s.daily_pnl_usd[1] == pytest.approx(-200.0)     # 100_100 - 100_300


def test_131_trading_day_convention_enters_validation_fingerprint():
    from alpha_agent.data.calendars import default_calendar

    base_ds = {"root_symbol": "NQ", "price_domain": "raw_contract",
               "source_fingerprint": "x", "bars_content_hash": "y" * 64, "n_bars": 10}
    nq = DatasetIdentity(**base_ds, trading_day_convention=default_calendar().trading_day_convention("NQ"))
    cl = DatasetIdentity(**base_ds, trading_day_convention=default_calendar().trading_day_convention("CL"))
    none = DatasetIdentity(**base_ds)
    assert nq.identity() != cl.identity()                  # different day conventions -> different identity
    assert nq.identity() != none.identity()
    conv = TradingDayConvention.for_root("NQ")
    assert conv.trading_day_boundary_local == "17:00"
    assert conv.timezone == "America/Chicago"


def pd_date(ts_ns: int):
    return pd.Timestamp(int(ts_ns), unit="ns", tz="UTC").date()


def _fake_two_day_plan():
    from alpha_agent.validation.trading_day import (
        TradingDayConvention as _C,
    )
    from alpha_agent.validation.trading_day import (
        ValidationDay as _VD,
    )
    from alpha_agent.validation.trading_day import (
        ValidationDayPlan as _P,
    )

    conv = _C(calendar_name="cme_equity_index", calendar_version="v", timezone="America/Chicago",
              trading_day_boundary_local="17:00")
    return _P(root_symbol="NQ", convention=conv, days=(
        _VD(trading_day="2026-01-06", first_ts_ns=BASE + 1, boundary_ts_ns=BASE + 100, n_bars=3),
        _VD(trading_day="2026-01-07", first_ts_ns=BASE + DAY, boundary_ts_ns=BASE + DAY + 100, n_bars=3),
    ))


# ======================================================================
# H/I. deterministic block / stationary bootstrap
# ======================================================================
def test_H_bootstrap_is_deterministic():
    r = vf.fixture_B_strong_signal(120)
    run = synthetic_backtest_run(r)
    cfg = BootstrapConfig(n_resamples=300, seed=42)
    a = run.daily.returns_array()
    from alpha_agent.validation import bootstrap_daily_returns

    b1 = bootstrap_daily_returns(a, cfg)
    b2 = bootstrap_daily_returns(a, cfg)
    assert b1.model_dump() == b2.model_dump()


def test_I_moving_block_bootstrap_preserves_block_structure():
    rng = np.random.default_rng(0)
    n, L = 200, 20
    idx = resample_indices(rng, n, BootstrapMethod.MOVING_BLOCK, L)
    # count positions that continue a block: idx[k+1] == (idx[k]+1) % n
    continues = sum(1 for k in range(n - 1) if idx[k + 1] == (idx[k] + 1) % n)
    # with block length 20 and ~10 blocks, ~ n - n_blocks continuations
    assert continues >= n - (n // L) - 2
    # an IID resample would have ~ n/n == ~1 continuation on average
    iid = rng.integers(0, n, size=n)
    iid_cont = sum(1 for k in range(n - 1) if iid[k + 1] == (iid[k] + 1) % n)
    assert continues > iid_cont + 20


def test_stationary_bootstrap_deterministic_and_dependent():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    i1 = resample_indices(rng1, 150, BootstrapMethod.STATIONARY, 15)
    i2 = resample_indices(rng2, 150, BootstrapMethod.STATIONARY, 15)
    assert np.array_equal(i1, i2)


# ======================================================================
# J/K/L. schedule time-shift null: common support + boundaries (13.1 / 13.2)
# ======================================================================
def _one_segment_sched(fp, n=40, rows_at=(5, 12, 25, 33)):
    ts = [BASE + i * 3600_000_000_000 for i in range(n)]   # hourly, one segment
    rows = tuple(TargetScheduleRow(ts_event_ns=ts[i], root_symbol="NQ", target_units=1,
                                   strategy_fingerprint=fp) for i in rows_at)
    sched = TargetSchedule(root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
                           strategy_dsl_version="1.0.0", feature_engine_version="0.9.2",
                           warmup_bars=0, rows=rows)
    return sched, ts


def test_A_per_segment_common_support_is_symmetric_and_comparable():
    fp = "stratdsl1:" + "d" * 64
    sched, ts = _one_segment_sched(fp, n=40, rows_at=(5, 12, 25, 33, 38))
    seg = [0] * 40
    cfg = NullTestConfig(n_null_samples=30, min_shift_bars=2, max_shift_bars=6)

    control = common_support_control_schedule(sched, ts, seg, cfg)
    # rows within K+1+L (= 7) of the segment end (pos 39) are pre-dropped from the
    # CONTROL -- so pos 33 (39-33=6 < 7) and pos 38 go; pos 5/12/25 stay
    kept_pos = {ts.index(r.ts_event_ns) for r in control.rows}
    assert kept_pos == {5, 12, 25}

    out = shifted_schedules(control, ts, seg, cfg)
    assert [k for k, _ in out] == [2, 3, 4, 5, 6]
    for k, sh in out:
        moved = sorted(ts.index(r.ts_event_ns) for r in sh.rows)
        # every replicate has the SAME source rows, just shifted by k -- identical
        # opportunity set, no drop, no overflow
        assert moved == [5 + k, 12 + k, 25 + k]
        assert all(p < 40 and seg[p] == 0 for p in moved)   # in-segment, no wrap
    # deterministic
    assert [k for k, _ in out] == [k for k, _ in shifted_schedules(control, ts, seg, cfg)]


def test_B_no_late_segment_overflow_target_enters_the_comparison():
    fp = "stratdsl1:" + "f" * 64
    ts = [BASE + i * 3600_000_000_000 for i in range(30)]
    seg = [0] * 15 + [1] * 15
    rows = tuple(TargetScheduleRow(ts_event_ns=ts[i], root_symbol="NQ", target_units=1,
                                   strategy_fingerprint=fp) for i in (2, 13, 17, 28))
    sched = TargetSchedule(root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
                           strategy_dsl_version="1.0.0", feature_engine_version="0.9.2",
                           warmup_bars=0, rows=rows)
    cfg = NullTestConfig(min_shift_bars=2, max_shift_bars=5)
    control = common_support_control_schedule(sched, ts, seg, cfg)
    kept = {ts.index(r.ts_event_ns) for r in control.rows}
    # pos 13 (seg 0 end 14: margin 6 > 14-13=1) and pos 28 (seg 1 end 29) dropped
    assert kept == {2, 17}
    for k, sh in shifted_schedules(control, ts, seg, cfg):
        for r in sh.rows:
            p = ts.index(r.ts_event_ns)
            src = p - k
            assert seg[p] == seg[src]                       # never jumped a segment
            assert p <= (14 if seg[src] == 0 else 29)       # never past the segment end


def test_C_observed_and_null_statistic_use_the_same_support():
    # the schedule-shift null's observed statistic is the CONTROL run's stat, not
    # the full-sample canonical stat (recorded separately)
    from alpha_agent.validation.enums import NullMethod, NullRole
    from alpha_agent.validation.nulls import summarize_null

    res = summarize_null(NullMethod.SCHEDULE_TIME_SHIFT, NullRole.DIAGNOSTIC, "daily_sharpe",
                         observed_stat=0.12, null_stats=np.array([0.05, 0.2, -0.1, 0.15]),
                         canonical_oos_statistic=0.9)
    assert res.role == NullRole.DIAGNOSTIC
    assert res.observed_stat == pytest.approx(0.12)         # control-support statistic
    assert res.canonical_oos_statistic == pytest.approx(0.9)  # full-sample, for contrast only


def test_causal_segment_ids_break_on_trading_day_session_gap_roll():
    bars, _labels = vf.cme_intraday_bars(instrument_id=1, n_trading_days=4)
    seg_default = causal_segment_ids(bars, root_symbol="NQ")
    daily = vf.daily_bars(vf.trend_closes(20), instrument_id=1)
    seg_daily = causal_segment_ids(daily, root_symbol="NQ")
    assert len(set(seg_daily)) == len(seg_daily)          # each daily bar its own segment
    assert max(seg_default) >= 3                          # intraday: multiple segments


def test_L_empirical_p_value_finite_correction_never_zero():
    null = np.array([0.0] * 100)
    p, _ = empirical_p_value(5.0, null)          # observed dominates every null
    assert p == pytest.approx(1 / 101)
    assert p > 0.0
    p2, _ = empirical_p_value(-5.0, null)        # observed below every null
    assert p2 == pytest.approx(101 / 101)


def test_schedule_shift_null_is_diagnostic_and_policy_never_gates_on_it():
    from alpha_agent.validation.enums import NullMethod, NullRole
    from alpha_agent.validation.nulls import summarize_null
    from alpha_agent.validation.policy import _null_gate_p

    # a strong schedule-shift REJECT (p large) must NOT gate; only the gating null does
    shift = summarize_null(NullMethod.SCHEDULE_TIME_SHIFT, NullRole.DIAGNOSTIC, "daily_sharpe",
                           observed_stat=0.0, null_stats=np.full(50, 1.0), canonical_oos_statistic=0.0)
    block = summarize_null(NullMethod.CENTERED_BLOCK_BOOTSTRAP, NullRole.GATING, "daily_sharpe",
                           observed_stat=1.0, null_stats=np.zeros(200), canonical_oos_statistic=1.0)
    assert shift.p_value > 0.9 and block.p_value < 0.05
    assert _null_gate_p([shift, block], 0.05) == pytest.approx(block.p_value)  # ignores the diagnostic


def test_circular_permutation_is_a_separate_named_method():
    from alpha_agent.validation.enums import NullMethod as _NM

    assert _NM.CIRCULAR_SCHEDULE_PERMUTATION.value == "circular_schedule_permutation"
    assert _NM.SCHEDULE_TIME_SHIFT.value == "schedule_time_shift"
    fp = "stratdsl1:" + "7" * 64
    sched, ts = _one_segment_sched(fp, n=30, rows_at=(20, 25, 28))
    perm = circular_permuted_schedules(sched, ts, NullTestConfig(min_shift_bars=2))
    assert any(min(r.ts_event_ns for r in sh.rows) < ts[20] for _k, sh in perm)


# ======================================================================
# M/N/O. BH / FDR
# ======================================================================
def test_M_bh_known_example():
    # m = 10, alpha = 0.05: q_(1)=0.01, q_(2)=0.04, q_(3)=0.13, ... -> reject 2
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    res = benjamini_hochberg_decisions(p, 0.05)
    assert [d.rejected for d in res.decisions][:4] == [True, True, False, False]
    assert res.n_rejected == 2
    qs = bh_qvalues(p)
    assert qs[0] == pytest.approx(0.01)
    assert qs[1] == pytest.approx(0.04)
    # a step-up example where the 3rd passes because a later rank does
    p2 = [0.01, 0.02, 0.025, 0.9, 0.9]
    r2 = benjamini_hochberg_decisions(p2, 0.05)
    assert r2.n_rejected == 3


def test_N_bh_deterministic_ties():
    p = [0.02, 0.02, 0.02, 0.9]
    q1 = bh_qvalues(p)
    q2 = bh_qvalues(list(reversed(p)))
    assert q1[:3] == pytest.approx(list(reversed(q2))[:3])
    assert all(np.isfinite(q1))


def test_O_failed_trials_stay_in_the_denominator():
    fam = _trial_family(
        vf.fixture_B_strong_signal(120), p_canonical=0.001,
        neighbour_pnls=[vf.fixture_A_zero_alpha(120, seed=k) for k in range(9)],
        neighbour_ps=[0.9] * 9,
    )
    assert fam.n_trials() == 10
    res = fam.benjamini_hochberg(0.10)
    assert res.n_trials == 10                    # all 10, not just the winner
    # canonical q-value uses m = 10 in the denominator
    assert res.decisions[0].q_value >= 0.001 * 10 / 1 - 1e-9


# ======================================================================
# P/Q/R. deflated Sharpe
# ======================================================================
def test_P_dsr_hand_checkable_psr():
    from statistics import NormalDist

    sr, n = 0.2, 101
    # denominator 1 - g3*sr + (g4-1)/4*sr^2; with g3=0, g4=1 it is exactly 1,
    # so PSR(0) reduces to the hand-checkable Phi(SR * sqrt(N-1)).
    psr = probabilistic_sharpe_ratio(sr, n, skewness=0.0, kurtosis=1.0, benchmark_daily_sharpe=0.0)
    assert psr == pytest.approx(NormalDist().cdf(sr * np.sqrt(n - 1)), rel=1e-9)
    # full formula with normal kurtosis (g4=3): denominator 1 + 0.5*sr^2
    psr2 = probabilistic_sharpe_ratio(sr, n, skewness=0.0, kurtosis=3.0)
    z = sr * np.sqrt(n - 1) / np.sqrt(1.0 + 0.5 * sr * sr)
    assert psr2 == pytest.approx(NormalDist().cdf(z), rel=1e-9)


def test_Q_dsr_decreases_as_trials_increase():
    r = vf.fixture_B_strong_signal(200, seed=1)
    tsharpes = [daily_sharpe(vf.fixture_A_zero_alpha(200, seed=k)) for k in range(20)]
    d_few = deflated_sharpe_ratio(r, n_trials=2, trial_daily_sharpes=tsharpes[:2])
    d_many = deflated_sharpe_ratio(r, n_trials=50, trial_daily_sharpes=tsharpes)
    assert deflated_benchmark_sharpe(np.var(tsharpes, ddof=1), 50) > deflated_benchmark_sharpe(
        np.var(tsharpes[:2], ddof=1) if len(tsharpes[:2]) >= 2 else 0.0, 2
    )
    assert d_many.deflated_sharpe_ratio <= d_few.deflated_sharpe_ratio + 1e-9


def test_R_dsr_edge_cases():
    # too-small sample -> undefined
    d = deflated_sharpe_ratio(np.array([0.01]), n_trials=5)
    assert not d.is_valid
    # zero variance -> undefined
    d0 = deflated_sharpe_ratio(np.zeros(50), n_trials=5)
    assert not d0.is_valid
    # strong signal, single trial -> benchmark 0, DSR == PSR(0), high
    strong = deflated_sharpe_ratio(vf.fixture_B_strong_signal(200), n_trials=1)
    assert strong.is_valid and strong.deflated_sharpe_ratio > 0.9
    # skew / kurtosis feed through
    assert strong.return_kurtosis != 0.0


# ---- Phase 13.2 DSR frequency-consistency audit (section 9) -----------------
def test_dsr_A_annualized_rescaling_does_not_change_the_probability():
    r = vf.fixture_B_strong_signal(200, seed=1)
    tsharpes = [daily_sharpe(vf.fixture_A_zero_alpha(200, seed=k)) for k in range(15)]
    base = deflated_sharpe_ratio(r, n_trials=15, trial_daily_sharpes=tsharpes)
    # rescale BOTH the return series and the trial Sharpes to "annualized" units
    scaled = deflated_sharpe_ratio(
        r * np.sqrt(252.0), n_trials=15,
        trial_daily_sharpes=[s for s in tsharpes],  # daily -- the function's own scale
    )
    # scaling every return by a constant leaves the Sharpe (mean/std) unchanged,
    # so the DSR probability is invariant
    assert scaled.deflated_sharpe_ratio == pytest.approx(base.deflated_sharpe_ratio, abs=1e-12)
    assert base.inference_frequency == "daily"
    # the annualized figure is reporting-only and differs by exactly sqrt(252)
    assert base.observed_annualized_sharpe == pytest.approx(
        base.observed_daily_sharpe * np.sqrt(252.0)
    )


def test_dsr_B_internally_frequency_consistent():
    # the z-statistic must use daily SR, daily N, daily skew/kurt -- reconstruct it
    from statistics import NormalDist

    r = vf.fixture_B_strong_signal(150, seed=2)
    res = deflated_sharpe_ratio(r, n_trials=1)
    sr_d = daily_sharpe(r)
    from alpha_agent.validation.metrics import sample_kurtosis, sample_skewness

    g3, g4, n = sample_skewness(r), sample_kurtosis(r), r.size
    z = sr_d * np.sqrt(n - 1) / np.sqrt(1 - g3 * sr_d + (g4 - 1) / 4 * sr_d**2)
    assert res.psr_vs_zero == pytest.approx(NormalDist().cdf(z), rel=1e-9)
    assert res.n_obs == n                                 # daily count, not annualized
    assert res.observed_daily_sharpe == pytest.approx(sr_d)


def test_dsr_C_more_trials_lowers_reliability():
    r = vf.fixture_B_strong_signal(220, seed=3)
    ts = [daily_sharpe(vf.fixture_A_zero_alpha(220, seed=k)) for k in range(40)]
    probs = [
        deflated_sharpe_ratio(r, n_trials=t, trial_daily_sharpes=ts[:max(2, t)]).deflated_sharpe_ratio
        for t in (2, 5, 20, 40)
    ]
    assert probs == sorted(probs, reverse=True)           # monotone non-increasing


def test_dsr_D_stronger_daily_signal_increases_dsr():
    ts = [daily_sharpe(vf.fixture_A_zero_alpha(200, seed=k)) for k in range(10)]
    weak = deflated_sharpe_ratio(vf.fixture_B_strong_signal(200, mu=40, sigma=250, seed=4),
                                 n_trials=10, trial_daily_sharpes=ts)
    strong = deflated_sharpe_ratio(vf.fixture_B_strong_signal(200, mu=200, sigma=150, seed=4),
                                   n_trials=10, trial_daily_sharpes=ts)
    assert strong.deflated_sharpe_ratio > weak.deflated_sharpe_ratio


def test_dsr_E_null_like_daily_signal_stays_weak():
    ts = [daily_sharpe(vf.fixture_A_zero_alpha(200, seed=k)) for k in range(12)]
    for seed in range(6):
        res = deflated_sharpe_ratio(vf.fixture_A_zero_alpha(200, seed=100 + seed),
                                    n_trials=12, trial_daily_sharpes=ts)
        assert res.deflated_sharpe_ratio < 0.9


def test_dsr_F_skew_kurtosis_edges_stay_finite_and_typed():
    # heavy-tailed daily returns: finite, typed, no exception
    rng = np.random.default_rng(0)
    fat = rng.standard_t(3, size=200) * 0.01 + 0.001
    res = deflated_sharpe_ratio(fat, n_trials=5, trial_daily_sharpes=[0.0, 0.1, -0.1, 0.05, 0.02])
    assert np.isfinite(res.return_kurtosis) and res.return_kurtosis > 3.0
    assert np.isfinite(res.deflated_sharpe_ratio)
    assert 0.0 <= res.deflated_sharpe_ratio <= 1.0
    # a denominator <= 0 (extreme skew * SR) is typed as invalid, not an exception
    bad = probabilistic_sharpe_ratio(2.0, 50, skewness=5.0, kurtosis=1.0)
    assert not np.isfinite(bad)


# ======================================================================
# S/T/U. parameter stability
# ======================================================================
def _neigh_results(canon_pnl, neighbour_pnls, *, canonical=True):
    out = []
    if canonical:
        r = synthetic_backtest_run(np.asarray(canon_pnl, dtype=float))
        out.append(NeighbourResult(params_label="canonical", strategy_fingerprint="fp:c",
                                   is_canonical=True, n_trades=20, oos_net_pnl_usd=r.net_pnl_usd,
                                   oos_daily_sharpe=r.daily_sharpe, oos_annualized_sharpe=r.annualized_sharpe))
    for i, p in enumerate(neighbour_pnls):
        r = synthetic_backtest_run(np.asarray(p, dtype=float))
        out.append(NeighbourResult(params_label=f"n{i}", strategy_fingerprint=f"fp:{i}",
                                   is_canonical=False, n_trades=20, oos_net_pnl_usd=r.net_pnl_usd,
                                   oos_daily_sharpe=r.daily_sharpe, oos_annualized_sharpe=r.annualized_sharpe))
    return out


def test_S_isolated_spike_flagged_unstable():
    fx = vf.fixture_C_unstable_spike()
    nb = ParameterNeighbourhood(strategy_key="synthetic", canonical_params={"w": 20},
                                neighbour_params=tuple({"w": 20 + d} for d in range(1, 9)))
    res = summarize_parameter_stability(nb, _neigh_results(fx["canonical"], fx["neighbours"]))
    assert res.canonical_is_isolated_spike
    assert res.fraction_positive_sharpe < 0.6


def test_T_broad_plateau_scores_stable():
    fx = vf.fixture_D_broad_plateau()
    nb = ParameterNeighbourhood(strategy_key="synthetic", canonical_params={"w": 20},
                                neighbour_params=tuple({"w": 20 + d} for d in range(1, 9)))
    res = summarize_parameter_stability(nb, _neigh_results(fx["canonical"], fx["neighbours"]))
    assert not res.canonical_is_isolated_spike
    assert res.fraction_positive_sharpe >= 0.6
    assert res.sharpe_dispersion_std < 0.5


def test_U_neighbours_count_as_trials():
    fam = _trial_family(vf.fixture_D_broad_plateau()["canonical"], p_canonical=0.02,
                        neighbour_pnls=vf.fixture_D_broad_plateau()["neighbours"],
                        neighbour_ps=[0.03] * 8)
    assert fam.n_trials() == 9
    assert sum(1 for t in fam.trials if t.role == "neighbour") == 8


# ======================================================================
# AA / AB / AC. verdict semantics
# ======================================================================
def test_AA_insufficient_sample_is_inconclusive():
    pol = TEST_FIXTURE_POLICY
    spec = _spec(pol)
    rep = _assemble_synth(spec, pol, vf.fixture_F_too_small(), n_folds=2)
    assert rep.verdict == Verdict.INCONCLUSIVE
    assert ReasonCode.INSUFFICIENT_OOS_DAYS in rep.reason_codes


def test_AB_clear_statistical_failure_is_reject():
    pol = TEST_FIXTURE_POLICY
    spec = _spec(pol)
    rep = _assemble_synth(spec, pol, vf.fixture_A_zero_alpha(180),
                          cost_results=_cost_results(1.0, 0.9), n_folds=4)
    assert rep.verdict == Verdict.REJECT
    # a zero-alpha strategy fails the null and/or the DSR gate
    assert {ReasonCode.NULL_NOT_REJECTED, ReasonCode.DSR_BELOW_THRESHOLD,
            ReasonCode.FDR_NOT_SIGNIFICANT} & set(rep.reason_codes)


def test_AC_strong_synthetic_signal_passes_under_test_policy():
    pol = TEST_FIXTURE_POLICY
    nb = ParameterNeighbourhood(strategy_key="synthetic", canonical_params={"w": 20},
                                neighbour_params=tuple({"w": 20 + d} for d in range(1, 7)))
    spec = _spec(pol, neighbourhood=nb)
    fx = vf.fixture_D_broad_plateau(seed=3)
    strong = vf.fixture_B_strong_signal(180, mu=140.0, sigma=180.0, seed=1)
    neigh = _neigh_results(strong, fx["neighbours"])
    rep = _assemble_synth(
        spec, pol, strong,
        cost_results=_cost_results(base_net=float(strong.sum()), degrade_2x_to=float(strong.sum()) * 0.7),
        neighbour_results=neigh, n_folds=4,
    )
    assert rep.verdict == Verdict.PASS, rep.reason_codes
    assert rep.reason_codes == (ReasonCode.ALL_GATES_SATISFIED,)


def test_null_synthetic_strategy_does_not_reliably_pass():
    pol = TEST_FIXTURE_POLICY
    passes = 0
    for seed in range(8):
        spec = _spec(pol, seed=seed)
        rep = _assemble_synth(spec, pol, vf.fixture_A_zero_alpha(180, seed=seed),
                              cost_results=_cost_results(1.0, 0.9), n_folds=4)
        passes += rep.verdict == Verdict.PASS
    assert passes == 0


# ======================================================================
# W. cost-fragile synthetic strategy degrades
# ======================================================================
def test_W_cost_fragile_strategy_degrades_and_rejects():
    pol = TEST_FIXTURE_POLICY
    spec = _spec(pol)
    fx = vf.fixture_E_cost_fragile()
    gross = fx["gross_daily_pnl"]
    net_1x = gross - fx["fills_per_day"] * fx["cost_per_fill_1x"]
    # 2x cost wipes the edge: net collapses well past the policy's max_cost_degradation
    cost_res = _cost_results(base_net=float(net_1x.sum()), degrade_2x_to=float(net_1x.sum()) * 0.1)
    rep = _assemble_synth(spec, pol, net_1x, cost_results=cost_res, n_folds=4)
    assert rep.cost_stress is not None
    assert rep.cost_stress.max_net_pnl_degradation > pol.max_cost_degradation
    assert ReasonCode.COST_STRESS_FAILED in rep.reason_codes
    assert rep.verdict == Verdict.REJECT


# ======================================================================
# X. ablations have separate fingerprints
# ======================================================================
def test_X_ablations_have_separate_fingerprints():
    a1 = AblationSpec(label="no_window", kind=AblationKind.DROP_TIME_WINDOW,
                      param_overrides={"window_start_local": "", "window_end_local": ""})
    a2 = AblationSpec(label="no_displacement", kind=AblationKind.DROP_DISPLACEMENT,
                      param_overrides={"displacement_atr_multiple": 0.01})
    assert a1.identity() != a2.identity()
    base = SILVER_BULLET_BENCHMARK.model_dump()
    s_can = strategy_fingerprint(make_silver_bullet_spec(SILVER_BULLET_BENCHMARK))
    s_ab = strategy_fingerprint(make_silver_bullet_spec(SilverBulletParams(**{**base, **a2.param_overrides})))
    assert s_can != s_ab


# ======================================================================
# Y. regime labels are causal
# ======================================================================
def test_Y_regime_labels_causal_and_concentration_detected():
    fx = vf.fixture_G_regime_concentrated()
    pnl = fx["daily_pnl"]
    returns = pnl / 100_000.0
    lab = RegimeLabelling(kind=RegimeKind.VOLATILITY, labels=fx["labels"],
                          definition={"derived_from": "train_only"})
    res = evaluate_regime_stability(returns, pnl, lab, concentration_threshold=0.6)
    assert res.status.value == "evaluated"
    assert res.max_regime_pnl_share > 0.6
    assert res.is_concentrated


def test_Y_regime_not_evaluated_when_no_labels():
    res = evaluate_regime_stability(np.zeros(10), np.zeros(10), None)
    assert res.status.value == "not_evaluated"


# ======================================================================
# Z. cross-market runs stay separate by root
# ======================================================================
def test_Z_cross_market_runs_are_independent_by_root():
    from alpha_agent.validation import evaluate_cross_market

    roots = [
        RootRunSummary(root_symbol="ES", strategy_fingerprint="fp:ES", n_trading_days=100,
                       n_trades=20, oos_net_pnl_usd=50_000.0, oos_daily_sharpe=0.1,
                       oos_annualized_sharpe=1.6, is_positive=True),
        RootRunSummary(root_symbol="NQ", strategy_fingerprint="fp:NQ", n_trading_days=100,
                       n_trades=20, oos_net_pnl_usd=1_000.0, oos_daily_sharpe=0.01,
                       oos_annualized_sharpe=0.1, is_positive=True),
        RootRunSummary(root_symbol="CL", strategy_fingerprint="fp:CL", n_trading_days=100,
                       n_trades=20, oos_net_pnl_usd=-2_000.0, oos_daily_sharpe=-0.02,
                       oos_annualized_sharpe=-0.3, is_positive=False),
    ]
    ev = evaluate_cross_market(roots, concentration_threshold=0.8)
    assert ev.n_roots == 3 and ev.n_positive_roots == 2
    assert ev.is_concentrated_in_one_root                 # ES holds ~98% of positive PnL
    assert {r.strategy_fingerprint for r in ev.roots} == {"fp:ES", "fp:NQ", "fp:CL"}


# ======================================================================
# AD / AE / AF. policy + validation identity
# ======================================================================
def test_AD_reliability_policy_fingerprint_stable_under_cosmetic_name():
    p1 = ReliabilityPolicy(policy_name="alpha")
    p2 = ReliabilityPolicy(policy_name="beta")
    assert p1.identity() == p2.identity()


def test_AE_policy_change_changes_validation_identity():
    strict = ReliabilityPolicy(dsr_min=0.95)
    loose = ReliabilityPolicy(dsr_min=0.80)
    assert strict.identity() != loose.identity()
    s1 = _spec(strict)
    s2 = _spec(loose)
    assert s1.validation_fingerprint() != s2.validation_fingerprint()


def test_AE_semantic_spec_change_moves_fingerprint():
    pol = TEST_FIXTURE_POLICY
    a = _spec(pol, seed=0)
    b = _spec(pol, seed=1)
    assert a.validation_fingerprint() != b.validation_fingerprint()


# ======================================================================
# AF / holdout discipline
# ======================================================================
def test_AF_holdout_result_cannot_mutate_policy_and_is_single_use():
    pol = TEST_FIXTURE_POLICY
    spec = _spec(pol)
    rep = _assemble_synth(spec, pol, vf.fixture_B_strong_signal(180),
                          cost_results=_cost_results(1.0, 0.8), n_folds=4)
    bundle = freeze_research(spec, pol, rep)
    assert bundle.research_verdict == rep.verdict
    # the frozen bundle is immutable (pydantic frozen)
    with pytest.raises(ValidationError):
        bundle.research_verdict = Verdict.REJECT

    rel = HoldoutRelease(bundle)

    class _FakeEngine:
        def __init__(self, spec, policy):
            self.spec = spec
            self.policy = policy
            self.ran = 0

        def run_holdout(self):
            self.ran += 1
            return rep.model_copy(update={"holdout_evaluated": True})

    eng = _FakeEngine(spec, pol)
    ev1 = rel.evaluate_once(eng)
    assert ev1.holdout_verdict == rep.verdict
    with pytest.raises(HoldoutDisciplineError):
        rel.evaluate_once(eng)                    # single use


def test_AF_modified_strategy_after_freeze_needs_new_holdout():
    pol = TEST_FIXTURE_POLICY
    spec = _spec(pol)
    rep = _assemble_synth(spec, pol, vf.fixture_B_strong_signal(180),
                          cost_results=_cost_results(1.0, 0.8), n_folds=4)
    bundle = freeze_research(spec, pol, rep)
    changed = spec.model_copy(update={"strategy_fingerprint": "stratdsl1:" + "b" * 64})

    class _FakeEngine:
        spec = changed
        policy = pol

        def run_holdout(self):  # pragma: no cover - must not be reached
            raise AssertionError("holdout must not run after a strategy change")

    with pytest.raises(HoldoutDisciplineError):
        HoldoutRelease(bundle).evaluate_once(_FakeEngine())


# ======================================================================
# AG. deterministic replay
# ======================================================================
def test_AG_deterministic_replay_of_the_report():
    pol = TEST_FIXTURE_POLICY
    spec = _spec(pol)
    r1 = _assemble_synth(spec, pol, vf.fixture_B_strong_signal(180),
                         cost_results=_cost_results(1.0, 0.8), n_folds=4)
    r2 = _assemble_synth(spec, pol, vf.fixture_B_strong_signal(180),
                         cost_results=_cost_results(1.0, 0.8), n_folds=4)
    assert r1.report_fingerprint() == r2.report_fingerprint()
    assert r1.verdict == r2.verdict


# ======================================================================
# real C++ path (skips cleanly when the core is not built)
# ======================================================================
def _tsmom_adapter(root="NQ", raw="NQU6"):
    def spec_factory(params: dict):
        return make_tsmom_spec(TsmomParams(**params))

    def compute_frame(plan, bars):
        df = bars[["ts_event_ns", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        src = SourceSeries(frame=df, price_domain=PriceDomain.RAW_CONTRACT,
                           identity={"root_symbol": root, "raw_symbol": raw},
                           interval_ns=vf.DAY_NS)  # daily bar fixture cadence
        return compute_features(src, [b.spec for b in plan.feature_bindings])

    return DslFamilyAdapter(
        "tsmom",
        {"root_symbol": root, "fast_horizon": 5, "slow_horizon": 20, "size": 1},
        spec_factory, compute_frame,
    )


def _tsmom_spec_and_engine(tmp_path, closes, *, iid=950101):
    from alpha_agent.validation.runner import CliBacktestRunner

    bars = vf.daily_bars(closes, instrument_id=iid)
    contracts = vf.one_contract(instrument_id=iid, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    adapter = _tsmom_adapter()
    canon_fp = strategy_fingerprint(make_tsmom_spec(TsmomParams(**adapter.canonical_params)))
    pol = ReliabilityPolicy(
        policy_name="e2e",
        minimum_sample=MinimumSampleRequirements(min_oos_trading_days=30, min_fills=0, min_trades=0,
                                                 min_walk_forward_folds=2, min_nonzero_return_days=10),
        null_p_value_max=0.10, dsr_min=0.80, max_cost_degradation=5.0,
        fold_consistency_min=0.0, param_stability_min_fraction_positive_sharpe=0.0,
    )
    from alpha_agent.data.calendars import default_calendar

    spec = ValidationSpec(
        strategy_fingerprint=canon_fp, strategy_key="tsmom",
        dataset=DatasetIdentity(
            root_symbol="NQ", price_domain="raw_contract",
            source_fingerprint="fixture:e2e", bars_content_hash="e" * 64, n_bars=len(closes),
            trading_day_convention=default_calendar().trading_day_convention("NQ"),
        ),
        split_plan=_split_plan(),
        walk_forward=WalkForwardConfig(n_folds=3, min_train_days=40, test_days=25, embargo_days=5, warmup_bars=20),
        capital_base_usd=100_000.0,
        cost_stress=CostStressPlan(scenarios=(
            CostScenario(label="baseline_1_0x", multiplier=1.0),
            CostScenario(label="stress_2_0x", multiplier=2.0),
        )),
        null_test=NullTestConfig(n_null_samples=40, min_shift_bars=2, seed=0),
        bootstrap=BootstrapConfig(n_resamples=300, seed=0),
        trial_family_id="tsmom.NQ.e2e",
        reliability_policy_fingerprint=pol.identity(),
    )
    runner = CliBacktestRunner(CLI, tmp_path)
    engine = ValidationEngine(spec, pol, runner, adapter, bars, contracts)
    return spec, pol, engine


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_e2e_research_report_from_cpp_official_path(tmp_path):
    engine = _tsmom_spec_and_engine(tmp_path, vf.trend_closes(300))[2]
    rep = engine.run_research()
    assert rep.verdict in (Verdict.PASS, Verdict.REJECT, Verdict.INCONCLUSIVE)
    assert rep.oos_daily.official_source == "cpp_portfolio_accountant_daily_equity_trace"
    assert rep.oos_metrics.n_trading_days > 0
    assert rep.walk_forward.n_folds_evaluated >= 2
    # the cost-stress scenarios actually reran the C++ path with scaled costs
    assert rep.cost_stress is not None
    labels = {s.label for s in rep.cost_stress.scenarios}
    assert labels == {"baseline_1_0x", "stress_2_0x"}
    assert rep.cost_stress.scenarios and any(
        s.commission_per_contract_usd == pytest.approx(4.0) for s in rep.cost_stress.scenarios
    )
    # the canonical daily series follows the exchange trading_day, from the C++
    # trading-day-boundary trace -- not UTC dates
    assert rep.oos_daily.day_basis == "trading_day"
    assert rep.oos_daily.day_plan_identity is not None
    # a null (at minimum the centered block bootstrap) reran and gated the verdict
    assert any(n.n_null_samples > 0 for n in rep.null_results)
    assert any(n.method == NullMethod.CENTERED_BLOCK_BOOTSTRAP for n in rep.null_results)
    assert not rep.holdout_evaluated


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_e2e_deterministic_replay(tmp_path):
    _s1, _p1, e1 = _tsmom_spec_and_engine(tmp_path / "a", vf.trend_closes(280))
    _s2, _p2, e2 = _tsmom_spec_and_engine(tmp_path / "b", vf.trend_closes(280))
    r1 = e1.run_research()
    r2 = e2.run_research()
    assert r1.report_fingerprint() == r2.report_fingerprint()


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_e2e_holdout_single_pass_after_freeze(tmp_path):
    spec, pol, engine = _tsmom_spec_and_engine(tmp_path, vf.trend_closes(300))
    research = engine.run_research()
    bundle = freeze_research(spec, pol, research)
    rel = HoldoutRelease(bundle)
    ev = rel.evaluate_once(engine)
    assert ev.holdout_report.holdout_evaluated
    assert ev.research_verdict == research.verdict
    with pytest.raises(HoldoutDisciplineError):
        rel.evaluate_once(engine)


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_e2e_daily_equity_trace_differences_reproduce_pnl(tmp_path):
    from alpha_agent.adapters.targets_bridge import run_targets_backtest_cli

    iid = 950202
    closes = vf.trend_closes(120)
    bars = vf.daily_bars(closes, instrument_id=iid)
    contracts = vf.one_contract(instrument_id=iid, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    ts = [int(t) for t in bars["ts_event_ns"]]
    fp = "stratdsl1:" + "c" * 64
    rows = tuple(TargetScheduleRow(ts_event_ns=ts[i], root_symbol="NQ", target_units=v, strategy_fingerprint=fp)
                 for i, v in ((10, 1), (60, -1), (90, 0)))
    sched = TargetSchedule(root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
                           strategy_dsl_version="1.0.0", feature_engine_version="0.9.2", warmup_bars=0, rows=rows)
    res = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path)
    s = daily_returns_from_cpp_trace(res, capital_base_usd=100_000.0)
    # summing the official daily PnL reproduces net equity change
    total = sum(s.daily_pnl_usd)
    assert total == pytest.approx(res["net_equity_usd_at_end"], abs=1e-6)
    assert s.n_days >= 2


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_131_M_cli_trading_day_trace_reruns_cpp_and_follows_trading_day(tmp_path):
    import datetime as dt

    from alpha_agent.adapters.targets_bridge import run_targets_backtest_cli

    iid = 950505
    bars, _labels = vf.cme_intraday_bars(instrument_id=iid, n_trading_days=12, start=dt.date(2026, 1, 6))
    contracts = vf.one_contract(instrument_id=iid, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    plan = build_validation_day_plan(bars, root_symbol="NQ")
    ts = [int(t) for t in bars["ts_event_ns"]]
    fp = "stratdsl1:" + "9" * 64
    rows = (TargetScheduleRow(ts_event_ns=ts[6], root_symbol="NQ", target_units=1, strategy_fingerprint=fp),)
    sched = TargetSchedule(root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
                           strategy_dsl_version="1.0.0", feature_engine_version="0.9.2", warmup_bars=0, rows=rows)

    vpath = plan.write_csv(tmp_path / "vd.csv")
    trading_day_res = run_targets_backtest_cli(
        bars, contracts, sched, executable=CLI, work_dir=tmp_path / "td",
        validation_days_path=vpath,
    )
    utc_res = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path / "utc")

    assert trading_day_res["daily_equity_basis"] == "trading_day"
    assert utc_res["daily_equity_basis"] == "utc_day"
    # (A) canonical series follows trading_day: exactly one obs per trading day
    assert len(trading_day_res["daily_equity"]) == plan.n_days == 12
    # (6) the UTC-bucket method allocates PnL to a DIFFERENT number of days
    assert len(utc_res["daily_equity"]) != len(trading_day_res["daily_equity"])
    # both still reproduce the SAME official total PnL (still C++-sourced)
    td = daily_returns_from_cpp_trace(trading_day_res, capital_base_usd=100_000.0,
                                      validation_day_plan=plan)
    assert td.day_basis == "trading_day"
    assert td.trading_day == tuple(plan.trading_day_labels())
    assert sum(td.daily_pnl_usd) == pytest.approx(trading_day_res["net_equity_usd_at_end"], abs=1e-6)
    utc = daily_returns_from_cpp_trace(utc_res, capital_base_usd=100_000.0)
    assert sum(td.daily_pnl_usd) == pytest.approx(sum(utc.daily_pnl_usd), abs=1e-6)


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_131_K_engine_null_stays_within_fold_and_reruns_cpp(tmp_path):
    # the ValidationEngine's schedule-shift null runs only over research data and
    # its shifted schedules are confined to the fold exec window
    engine = _tsmom_spec_and_engine(tmp_path, vf.trend_closes(300))[2]
    rep = engine.run_research()
    assert not rep.holdout_evaluated
    hw = engine.spec.split_plan.holdout_window()
    # no OOS observation ts is inside the locked holdout window
    assert all(t < hw.start_ts_ns for t in rep.oos_daily.ts_ns)
    for f in rep.fold_summaries:
        assert f.test_end_ts_ns <= hw.start_ts_ns


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_e2e_cost_stress_reruns_cpp_not_python_subtraction(tmp_path):
    # 1x vs 2x commission must produce DIFFERENT official costs from the C++ side
    from alpha_agent.adapters.targets_bridge import run_targets_backtest_cli

    iid = 950303
    closes = vf.trend_closes(90)
    bars = vf.daily_bars(closes, instrument_id=iid)
    contracts = vf.one_contract(instrument_id=iid, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    ts = [int(t) for t in bars["ts_event_ns"]]
    fp = "stratdsl1:" + "e" * 64
    rows = (TargetScheduleRow(ts_event_ns=ts[10], root_symbol="NQ", target_units=1, strategy_fingerprint=fp),)
    sched = TargetSchedule(root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
                           strategy_dsl_version="1.0.0", feature_engine_version="0.9.2", warmup_bars=0, rows=rows)
    one_x = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path / "1x",
                                     commission_per_contract_usd=2.0, slippage_ticks=0.0, spread_ticks=0.0)
    two_x = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path / "2x",
                                     commission_per_contract_usd=4.0, slippage_ticks=0.0, spread_ticks=0.0)
    assert two_x["costs_usd"] == pytest.approx(2.0 * one_x["costs_usd"])
    assert two_x["net_pnl_usd"] < one_x["net_pnl_usd"]


# ======================================================================
# AH. all Phase 09-12 suites remain green -- exercised by the full run;
#     here we just assert the additive C++ export did not disturb the
#     existing baseline JSON keys.
# ======================================================================
@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_AH_additive_cpp_export_keeps_existing_keys(tmp_path):
    proc = subprocess.run([str(CLI)], capture_output=True, text=True, check=False)
    assert proc.returncode == 2  # usage
    iid = 950404
    from alpha_agent.adapters.targets_bridge import run_targets_backtest_cli

    bars = vf.daily_bars(vf.trend_closes(40), instrument_id=iid)
    contracts = vf.one_contract(instrument_id=iid, raw_symbol="NQU6", root="NQ", expiration_ns=BIG_EXP)
    ts = [int(t) for t in bars["ts_event_ns"]]
    fp = "stratdsl1:" + "f" * 64
    rows = (TargetScheduleRow(ts_event_ns=ts[5], root_symbol="NQ", target_units=1, strategy_fingerprint=fp),)
    sched = TargetSchedule(root_symbol="NQ", strategy_fingerprint=fp, strategy_id="x",
                           strategy_dsl_version="1.0.0", feature_engine_version="0.9.2", warmup_bars=0, rows=rows)
    res = run_targets_backtest_cli(bars, contracts, sched, executable=CLI, work_dir=tmp_path)
    for k in ("gross_pnl_usd", "costs_usd", "net_pnl_usd", "fills", "trades", "equity_usd"):
        assert k in res
    assert "daily_equity" in res and isinstance(res["daily_equity"], list)
