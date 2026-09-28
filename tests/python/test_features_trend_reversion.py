"""Phase 09 -- trend/momentum and mean-reversion features
(checklist H moving averages, I momentum, L z-score)."""
from __future__ import annotations

import math

import numpy as np
from alpha_agent.features import FeatureIssueKind, FeatureSpec, compute_features
from features_fixtures import signed_source, source_from_close


# ---- H. moving averages -------------------------------------------
def test_simple_moving_average_is_the_trailing_mean():
    close = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ff = compute_features(source_from_close(close), [FeatureSpec(kind="ma", params={"window": 3})])
    ma = ff.features["ma_3"].to_numpy()
    assert math.isnan(ma[0]) and math.isnan(ma[1])
    np.testing.assert_allclose(ma[2:], [2.0, 3.0, 4.0, 5.0])


def test_ma_spread_is_fast_minus_slow_and_signed_safe():
    ff = compute_features(signed_source(), [
        FeatureSpec(kind="ma_spread", params={"fast": 2, "slow": 4}),
    ])
    s = ff.features["ma_spread_2_4"].to_numpy()
    closes = np.asarray(signed_source().frame["close"])
    # fast(2) - slow(4) at the last bar, across negative prices
    exp_last = closes[-2:].mean() - closes[-4:].mean()
    np.testing.assert_allclose(s[-1], exp_last)
    assert not np.isinf(s).any()


def test_ema_matches_a_hand_rolled_recursion():
    close = [10.0, 11.0, 12.0, 13.0, 14.0]
    ff = compute_features(source_from_close(close), [FeatureSpec(kind="ema", params={"window": 3})])
    alpha = 2 / (3 + 1)
    e = close[0]
    seq = [e]
    for c in close[1:]:
        e = alpha * c + (1 - alpha) * e
        seq.append(e)
    # min_periods=3 -> first two are missing
    got = ff.features["ema_3"].to_numpy()
    np.testing.assert_allclose(got[2:], seq[2:])


# ---- I. momentum -------------------------------------------------
def test_multi_horizon_momentum_and_vol_scaled_trend():
    close = 100.0 + np.arange(30) * 2.0        # steady up-trend, +2 per bar
    ff = compute_features(source_from_close(close), [
        FeatureSpec(kind="diff", params={"n": 10}),
        FeatureSpec(kind="vol_scaled_trend", params={"n": 10, "vol_window": 5}),
    ])
    np.testing.assert_allclose(ff.features["diff_10"].to_numpy()[10:], 20.0)
    # 1-bar diffs are all +2 -> rolling std ~ 0 -> vol-scaled trend suppressed (not inf)
    vst = ff.features["vol_scaled_trend_10_5"].to_numpy()
    assert not np.isinf(vst).any()


def test_breakout_distance_uses_the_prior_window_only():
    close = [10, 10, 10, 10, 10, 15, 12, 12]
    ff = compute_features(source_from_close(close, spread=0.0),
                          [FeatureSpec(kind="breakout_up", params={"window": 3})])
    b = ff.features["breakout_up_3"].to_numpy()
    # at idx 5 the prior 3-bar high (idx 2..4) is 10 -> breakout = 15 - 10 = 5
    np.testing.assert_allclose(b[5], 5.0)
    # at idx 6 prior high (idx 3..5) is 15 -> 12 - 15 = -3 (no breakout)
    np.testing.assert_allclose(b[6], -3.0)


# ---- L. z-score -----------------------------------------------
def test_zscore_is_standardized_against_trailing_stats_only():
    close = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ff = compute_features(source_from_close(close),
                          [FeatureSpec(kind="zscore", params={"window": 4})])
    z = ff.features["zscore_4"].to_numpy()
    assert np.isnan(z[:3]).all()
    # window at idx 4 = [2,3,4,5], mean 3.5, sample std sqrt(5/3)
    import statistics
    exp = (5.0 - 3.5) / statistics.stdev([2, 3, 4, 5])
    np.testing.assert_allclose(z[4], exp)


def test_zscore_near_zero_std_is_missing_not_infinite():
    close = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    ff = compute_features(source_from_close(close),
                          [FeatureSpec(kind="zscore", params={"window": 3})])
    z = ff.features["zscore_3"].to_numpy()
    assert np.isnan(z[2:]).all()
    assert not np.isinf(z).any()
    assert ff.qa.has_kind(FeatureIssueKind.ZERO_DENOMINATOR, feature="zscore_3")


def test_reversal_is_signed_safe_negative_of_the_change():
    ff = compute_features(signed_source(), [FeatureSpec(kind="reversal", params={"n": 1})])
    closes = np.asarray(signed_source().frame["close"])
    np.testing.assert_allclose(ff.features["reversal_1"].to_numpy()[1:], -np.diff(closes))
