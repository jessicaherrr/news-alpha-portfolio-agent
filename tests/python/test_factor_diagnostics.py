"""Factor diagnostics (News Alpha Phase E screening plane): the math on
constructed series whose answer is known -- forward-return convention, IC /
the time-series Spearman/Pearson IC, the effective-sample t under overlap, decay, subperiods, coverage, turnover, the
drift-vs-timing cost split, the screen rule, correlations and the window /
holdout guards. Pure numpy/pandas, no data, no network."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from alpha_agent.registry.holdout_guard import HoldoutAccessError
from alpha_agent.screening.factor_diagnostics import (
    DiagnosticScope,
    FactorDiagnostics,
    FactorScreenPolicy,
    HorizonDiagnostics,
    ICKind,
    ScreenReason,
    ScreenStatus,
    diagnose_factor,
    forward_returns,
    signal_correlations,
)

START, END = date(2018, 1, 1), date(2023, 1, 1)


def _prices(n: int = 1250, *, seed: int = 0, drift: float = 0.0) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.01, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.Series(pd.bdate_range("2018-01-02", periods=n)), open_, close


def _diag(factor, *, days=None, open_=None, close=None, sign=1, h=20, **kw):
    if days is None:
        days, open_, close = _prices(len(factor))
    return diagnose_factor(days=days, factor=np.asarray(factor, float), open_=open_, close=close,
                           expression="test(x)", expected_sign=sign, declared_horizon_days=h,
                           window_start=START, window_end_exclusive=END, **kw)


def test_forward_returns_enter_at_the_next_open_and_never_leave_the_data():
    open_ = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    close = np.array([10.5, 11.5, 12.5, 13.5, 14.5])
    np.testing.assert_allclose(forward_returns(open_, close, 1), [11.5 / 11 - 1, 12.5 / 12 - 1, 13.5 / 13 - 1,
                                                                  14.5 / 14 - 1, np.nan])
    np.testing.assert_allclose(forward_returns(open_, close, 2), [12.5 / 11 - 1, 13.5 / 12 - 1, 14.5 / 13 - 1,
                                                                  np.nan, np.nan])
    assert np.isnan(forward_returns(np.array([10.0, 0.0, 12.0]), np.array([10.0, 11.0, 12.0]), 1)[0])


def test_a_factor_that_knows_the_forward_return_screens_and_its_negative_contradicts():
    days, open_, close = _prices()
    oracle = forward_returns(open_, close, 20)  # a look-ahead fixture: perfect by construction
    d = _diag(oracle, days=days, open_=open_, close=close)
    assert d.declared.ts_spearman_ic == pytest.approx(1.0) and d.declared.ts_pearson_ic == pytest.approx(1.0)
    assert d.status is ScreenStatus.SCREEN_CONTINUE and d.subperiod_sign_share == 1.0
    flipped = _diag(-oracle, days=days, open_=open_, close=close)
    assert flipped.status is ScreenStatus.CONTRADICTS_EXPECTED_SIGN  # never re-signed into a pass
    assert _diag(-oracle, days=days, open_=open_, close=close, sign=-1).status is ScreenStatus.SCREEN_CONTINUE


def test_noise_gets_no_screen_support_and_too_little_data_is_insufficient():
    days, open_, close = _prices()
    noise = np.random.default_rng(42).normal(size=len(days))
    d = _diag(noise, days=days, open_=open_, close=close)
    assert d.status is ScreenStatus.NO_SCREEN_SUPPORT and ScreenReason.BELOW_THRESHOLD in d.reasons
    short_days, so, sc = _prices(200)
    short = _diag(np.random.default_rng(1).normal(size=200), days=short_days, open_=so, close=sc)
    assert short.status is ScreenStatus.INSUFFICIENT_DATA
    assert {ScreenReason.TOO_FEW_PAIRS, ScreenReason.TOO_FEW_SUBPERIODS} <= set(short.reasons)


def test_the_t_rests_on_the_effective_sample_not_on_overlapping_daily_pairs():
    days, open_, close = _prices(seed=3)
    # A persistent factor (trailing 20-day return) against overlapping 20-day
    # forward returns: both autocorrelated, so a naive t overstates evidence.
    factor = pd.Series(close).pct_change(20, fill_method=None).to_numpy(copy=True)
    d = _diag(factor, days=days, open_=open_, close=close)
    dec = d.declared
    assert dec.n_independent == pytest.approx(dec.n_pairs / 20, abs=0.01)
    assert dec.ts_spearman_t == pytest.approx(
        dec.ts_spearman_ic * np.sqrt(dec.n_pairs / 20 - 2) / np.sqrt(1 - dec.ts_spearman_ic ** 2), rel=1e-9)
    naive_t = dec.ts_spearman_ic * np.sqrt(dec.n_pairs)
    assert abs(dec.ts_spearman_t) < abs(naive_t) / 4  # ~sqrt(20) smaller
    one_day = next(h for h in d.horizons if h.horizon_days == 1)
    assert one_day.n_independent == one_day.n_pairs  # no overlap at 1 day


def test_the_effective_sample_t_is_calibrated_under_no_relationship():
    """Across independent null worlds the |t| > 2 rate stays near or below
    the nominal ~5% -- the property Newey-West at lag h failed on real data."""
    exceed = 0
    for seed in range(60):
        days, open_, close = _prices(seed=100 + seed)
        factor = pd.Series(close).pct_change(60, fill_method=None).to_numpy(copy=True)  # persistent, no edge
        t = _diag(factor, days=days, open_=open_, close=close, h=60).declared.ts_spearman_t
        exceed += t is not None and abs(t) > 2
    assert exceed / 60 <= 0.10


def test_decay_covers_1_5_20_60_and_adds_a_declared_horizon_outside_them():
    days, open_, close = _prices()
    d = _diag(np.random.default_rng(0).normal(size=len(days)), days=days, open_=open_, close=close)
    assert [h.horizon_days for h in d.horizons] == [1, 5, 20, 60]
    assert [h.n_pairs for h in d.horizons] == [len(days) - h for h in (1, 5, 20, 60)]
    d10 = _diag(np.random.default_rng(0).normal(size=len(days)), days=days, open_=open_, close=close, h=10)
    assert [h.horizon_days for h in d10.horizons] == [1, 5, 10, 20, 60] and d10.declared.horizon_days == 10


def test_subperiods_are_calendar_years_with_an_icir_across_them():
    days, open_, close = _prices()
    d = _diag(forward_returns(open_, close, 20), days=days, open_=open_, close=close)
    assert [s.label for s in d.subperiods] == ["2018", "2019", "2020", "2021", "2022"]
    assert all(s.evaluable for s in d.subperiods) and sum(s.n_pairs for s in d.subperiods) == d.declared.n_pairs
    ics = [s.ts_spearman_ic for s in d.subperiods]
    assert d.ts_spearman_ir_yearly == pytest.approx(np.mean(ics) / np.std(ics, ddof=1), abs=1e-3)


def test_coverage_counts_warm_up_and_never_fills_a_gap():
    days, open_, close = _prices()
    factor = pd.Series(close).pct_change(20, fill_method=None).to_numpy(copy=True)
    factor[500:505] = np.nan  # an invalid input mid-sample
    d = _diag(factor, days=days, open_=open_, close=close)
    assert d.coverage.n_factor_defined == len(days) - 20 - 5
    assert d.coverage.undefined_after_warmup == 5
    assert d.coverage.first_defined_day == days.iloc[20].date()


def test_turnover_counts_sign_flips_and_autocorrelation():
    days, open_, close = _prices()
    alternating = np.where(np.arange(len(days)) % 2 == 0, 1.0, -1.0)
    d = _diag(alternating, days=days, open_=open_, close=close)
    assert d.turnover.sign_flips_per_year == pytest.approx(252, rel=0.01)
    assert d.turnover.mean_daily_position_change == pytest.approx(2.0)
    assert d.turnover.autocorrelation_1d == pytest.approx(-1.0)
    steady = _diag(np.linspace(1, 2, len(days)), days=days, open_=open_, close=close)
    assert steady.turnover.sign_flips_per_year == 0 and steady.cost.break_even_cost_bps is None


def test_cost_sensitivity_splits_drift_from_timing_before_any_break_even():
    days, open_, close = _prices(drift=0.002, seed=9)  # a strongly rising market
    always_long_noise = np.abs(np.random.default_rng(2).normal(size=len(days))) + 0.01
    d = _diag(always_long_noise, days=days, open_=open_, close=close)
    c = d.cost
    assert c.gross_edge_bps > 0 and c.drift_bps == pytest.approx(c.gross_edge_bps, abs=1e-6)
    assert abs(c.timing_edge_bps) < 1e-6 and c.break_even_cost_bps is None  # drift is not the factor's edge
    oracle = forward_returns(open_, close, 20) - np.nanmean(forward_returns(open_, close, 20))
    good = _diag(oracle, days=days, open_=open_, close=close).cost
    assert good.timing_edge_bps > 0 and good.break_even_cost_bps == pytest.approx(
        good.timing_edge_bps / good.position_change_per_rebalance, rel=1e-3)


def test_signal_correlations_find_duplicate_signals_on_common_days():
    idx = [f"2020-01-{d:02d}" for d in range(1, 29)]
    a = pd.Series(np.arange(28, dtype=float), index=idx)
    b = pd.Series(np.arange(28, dtype=float) * 3 + 1, index=idx)
    c = pd.Series(np.cos(np.arange(20)), index=idx[:20])
    pairs = {(p.a, p.b): p for p in signal_correlations({"a": a, "b": b, "c": c})}
    assert pairs[("a", "b")].correlation == pytest.approx(1.0) and pairs[("a", "b")].n_common == 28
    assert pairs[("a", "c")].n_common == 20


def test_days_outside_the_window_or_in_the_holdout_are_refused():
    days, open_, close = _prices(100)
    with pytest.raises(ValueError, match="outside the window"):
        diagnose_factor(days=days, factor=np.zeros(100), open_=open_, close=close, expression="x",
                        expected_sign=1, declared_horizon_days=5, window_start=START, window_end_exclusive=date(2018, 3, 1))
    late = pd.Series(pd.bdate_range("2024-12-01", periods=100))
    with pytest.raises(HoldoutAccessError):
        diagnose_factor(days=late, factor=np.zeros(100), open_=open_, close=close, expression="x",
                        expected_sign=1, declared_horizon_days=5, window_start=date(2024, 1, 1),
                        window_end_exclusive=date(2026, 1, 1))
    with pytest.raises(ValueError, match="ascending"):
        diagnose_factor(days=days[::-1].reset_index(drop=True), factor=np.zeros(100), open_=open_, close=close,
                        expression="x", expected_sign=1, declared_horizon_days=5, window_start=START,
                        window_end_exclusive=END)


def test_the_screen_policy_is_predeclared_and_fingerprinted():
    policy = FactorScreenPolicy()
    assert policy.fingerprint() == FactorScreenPolicy().fingerprint()
    assert policy.fingerprint() != FactorScreenPolicy(t_threshold=1.5).fingerprint()
    d = _diag(np.random.default_rng(3).normal(size=1250))
    assert d.policy_fingerprint == policy.fingerprint() and d.plane == "SCREENING"


def test_every_diagnostic_is_scoped_unconditional_and_event_conditioned_is_refused():
    d = _diag(np.random.default_rng(4).normal(size=1250))
    assert d.scope is DiagnosticScope.UNCONDITIONAL_FACTOR and "Not event-conditioned evidence" in d.scope_note
    days, open_, close = _prices()
    with pytest.raises(ValueError, match="comparable events"):
        diagnose_factor(days=days, factor=np.zeros(len(days)), open_=open_, close=close, expression="x",
                        expected_sign=1, declared_horizon_days=20, window_start=START, window_end_exclusive=END,
                        scope=DiagnosticScope.EVENT_CONDITIONED)
    with pytest.raises(ValueError):  # the model itself cannot claim event-conditioned evidence
        FactorDiagnostics(**{**d.model_dump(), "scope": DiagnosticScope.EVENT_CONDITIONED})


def test_ics_are_time_series_and_never_labelled_as_cross_sectional_rank_ic():
    d = _diag(np.random.default_rng(5).normal(size=1250))
    assert d.ic_kind is ICKind.TIME_SERIES
    with pytest.raises(ValueError):
        FactorDiagnostics(**{**d.model_dump(), "ic_kind": ICKind.CROSS_SECTIONAL})
    fields = set(HorizonDiagnostics.model_fields)
    assert {"ts_spearman_ic", "ts_pearson_ic", "ts_spearman_t"} <= fields
    assert not {"ic", "rank_ic", "rank_ic_t"} & fields
    assert "ts_spearman_ir_yearly" in FactorDiagnostics.model_fields and "icir" not in FactorDiagnostics.model_fields
