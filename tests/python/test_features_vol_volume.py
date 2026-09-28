"""Phase 09 -- volatility and volume/liquidity features
(checklist J volatility, K ATR/range, M volume)."""
from __future__ import annotations

import math

import numpy as np
from alpha_agent.features import FeatureIssueKind, FeatureSpec, compute_features
from alpha_agent.features.source import SourceSeries
from alpha_agent.schemas.market_data import PriceDomain
from features_fixtures import bars_from_close, source_from_close


# ---- J. volatility ----------------------------------------------
def test_return_volatility_is_the_trailing_std_of_returns():
    close = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0]
    ff = compute_features(source_from_close(close),
                          [FeatureSpec(kind="volatility", params={"window": 4})])
    v = ff.features["volatility_4"].to_numpy()
    assert np.isnan(v[:3]).all()
    r = np.diff(close) / np.asarray(close)[:-1]      # 1-bar simple returns
    exp = np.std(r[0:4], ddof=1)
    np.testing.assert_allclose(v[4], exp)


def test_vol_percentile_is_a_trailing_rank_between_zero_and_one():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, 40))
    spec = FeatureSpec(kind="vol_percentile", params={"window": 5, "lookback": 10})
    ff = compute_features(source_from_close(close), [spec])
    p = ff.features["vol_percentile_5_10"].to_numpy()
    p = p[~np.isnan(p)]
    assert p.size > 0
    assert ((p >= 0.0) & (p <= 1.0)).all()


# ---- K. ATR / range -------------------------------------------
def test_atr_is_the_mean_true_range_and_signed_safe():
    # explicit OHLC, including a bar straddling zero
    frame = bars_from_close([2.0, -1.0, -4.0, -2.0], spread=0.0)
    # override highs/lows to known values
    frame["high"] = [3.0, 1.0, -1.0, 0.0]
    frame["low"] = [1.0, -2.0, -5.0, -3.0]
    frame["close"] = [2.0, -1.0, -4.0, -2.0]
    frame["open"] = [1.5, 2.0, -1.0, -4.0]
    src = SourceSeries(frame=frame, price_domain=PriceDomain.RAW_CONTINUOUS,
                       identity={"continuous_symbol": "CL.v.0"})
    ff = compute_features(src, [FeatureSpec(kind="atr", params={"window": 2})])
    # TR: bar0 = high-low = 2 ; bar1 = max(3, |1-2|, |-2-2|) = 4 ; bar2 = max(4, |-1+1|, |-5+1|)=4
    # atr_2 at idx 2 = mean(TR1, TR2) = (4 + 4)/2 = 4
    a = ff.features["atr_2"].to_numpy()
    assert math.isnan(a[0])
    np.testing.assert_allclose(a[2], 4.0)
    assert not np.isinf(a).any()


def test_range_vol_is_mean_high_minus_low():
    frame = bars_from_close([10.0, 11.0, 12.0, 13.0], spread=0.0)
    frame["high"] = [11.0, 13.0, 15.0, 16.0]
    frame["low"] = [9.0, 10.0, 11.0, 12.0]
    src = SourceSeries(frame=frame, price_domain=PriceDomain.RAW_CONTINUOUS,
                       identity={"continuous_symbol": "NQ.v.0"})
    ff = compute_features(src, [FeatureSpec(kind="range_vol", params={"window": 2})])
    rv = ff.features["range_vol_2"].to_numpy()
    # ranges = [2, 3, 4, 4] -> mean over 2 at idx1 = 2.5, idx2 = 3.5, idx3 = 4.0
    np.testing.assert_allclose(rv[1:], [2.5, 3.5, 4.0])


# ---- M. volume features --------------------------------------
def test_volume_passthrough_and_rolling_average():
    vol = [100, 200, 300, 400, 500]
    src = source_from_close([1.0, 2.0, 3.0, 4.0, 5.0], volume=vol)
    ff = compute_features(src, [
        FeatureSpec(kind="volume", params={}),
        FeatureSpec(kind="avg_volume", params={"window": 2}),
    ])
    np.testing.assert_array_equal(ff.features["volume"].to_numpy(), vol)
    np.testing.assert_allclose(ff.features["avg_volume_2"].to_numpy()[1:], [150, 250, 350, 450])


def test_relative_volume_uses_the_prior_window_average():
    vol = [100, 100, 100, 400, 100]
    src = source_from_close([1.0] * 5, volume=vol)
    ff = compute_features(src, [FeatureSpec(kind="rel_volume", params={"window": 2})])
    rv = ff.features["rel_volume_2"].to_numpy()
    # at idx 3: prior 2-bar avg (idx 1,2) = 100 -> 400/100 = 4
    np.testing.assert_allclose(rv[3], 4.0)


def test_volume_zscore_missing_when_volume_is_constant():
    src = source_from_close([1.0] * 6, volume=[100] * 6)
    ff = compute_features(src, [FeatureSpec(kind="volume_zscore", params={"window": 3})])
    z = ff.features["volume_zscore_3"].to_numpy()
    assert np.isnan(z[2:]).all() and not np.isinf(z).any()
    assert ff.qa.has_kind(FeatureIssueKind.ZERO_DENOMINATOR, feature="volume_zscore_3")


def test_volume_feature_without_a_volume_column_raises():
    frame = bars_from_close([1.0, 2.0, 3.0]).drop(columns=["volume"])
    src = SourceSeries(frame=frame, price_domain=PriceDomain.RAW_CONTINUOUS,
                       identity={"continuous_symbol": "NQ.v.0"})
    try:
        compute_features(src, [FeatureSpec(kind="avg_volume", params={"window": 2})])
        assert False, "expected a ValueError"
    except ValueError:
        pass
