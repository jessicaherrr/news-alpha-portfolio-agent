"""Phase 09 -- explicit anti-look-ahead tests (checklist B, and O replay).

The engine's core guarantee: the value of a point-in-time feature at timestamp
``T`` is identical whether the source series ends at ``T`` or continues far
beyond it. Equivalently, no backward-looking window may include a future row.
"""
from __future__ import annotations

import numpy as np
import pytest
from alpha_agent.features import (
    FeatureSafetyError,
    FeatureSpec,
    LookaheadUnsafeError,
    compute_features,
)
from features_fixtures import continuous_roll_source, source_from_close

PIT_SPECS = [
    FeatureSpec(kind="return", params={"n": 1}),
    FeatureSpec(kind="return", params={"n": 5}),
    FeatureSpec(kind="log_return", params={"n": 3}),
    FeatureSpec(kind="diff", params={"n": 4}),
    FeatureSpec(kind="mean_return", params={"window": 6}),
    FeatureSpec(kind="ma", params={"window": 10}),
    FeatureSpec(kind="ema", params={"window": 8}),
    FeatureSpec(kind="ma_spread", params={"fast": 5, "slow": 15}),
    FeatureSpec(kind="trend_strength", params={"fast": 5, "slow": 15}),
    FeatureSpec(kind="breakout_up", params={"window": 10}),
    FeatureSpec(kind="donchian_pos", params={"window": 10}),
    FeatureSpec(kind="zscore", params={"window": 12}),
    FeatureSpec(kind="norm_dev", params={"window": 12}),
    FeatureSpec(kind="volatility", params={"window": 10}),
    FeatureSpec(kind="realized_vol", params={"window": 10}),
    FeatureSpec(kind="atr", params={"window": 9}),
    FeatureSpec(kind="range_vol", params={"window": 9}),
    FeatureSpec(kind="vol_change", params={"window": 5}),
    FeatureSpec(kind="vol_percentile", params={"window": 5, "lookback": 10}),
    FeatureSpec(kind="avg_volume", params={"window": 8}),
    FeatureSpec(kind="rel_volume", params={"window": 8}),
    FeatureSpec(kind="volume_zscore", params={"window": 8}),
]


def _spec_id(s: FeatureSpec) -> str:
    return "_".join([s.kind, *(str(v) for v in s.params.values())])


@pytest.mark.parametrize("spec", PIT_SPECS, ids=_spec_id)
def test_feature_value_is_unchanged_when_more_future_data_is_appended(spec):
    close = 100.0 + np.cumsum(np.sin(np.arange(60) / 3.0)) + np.arange(60) * 0.05
    volume = 100 + (np.arange(60) % 7) * 10
    full = source_from_close(close, volume=volume)

    ff_full = compute_features(full, [spec])
    fname = ff_full.feature_names[0]

    for cut in (25, 40, 55):
        as_of = int(full.frame["ts_event_ns"].iloc[cut])
        ff_cut = compute_features(full, [spec], as_of_ts_ns=as_of)
        assert len(ff_cut.features) == cut + 1
        a = ff_full.features[fname].to_numpy()[: cut + 1]
        b = ff_cut.features[fname].to_numpy()
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-12, atol=1e-12)


def test_no_centered_window_leaks_the_next_bar():
    # a single large spike: a look-ahead MA would react one bar early
    close = [100.0] * 10 + [130.0] + [100.0] * 10
    src = source_from_close(close)
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 3})])
    ma = ff.features["ma_3"].to_numpy()
    # ma at the bar BEFORE the spike (idx 9) must not yet see 130
    np.testing.assert_allclose(ma[9], 100.0)
    np.testing.assert_allclose(ma[10], (100 + 100 + 130) / 3)


def test_retrospective_roll_feature_is_blocked_under_the_point_in_time_contract():
    src = continuous_roll_source()
    with pytest.raises(LookaheadUnsafeError):
        compute_features(src, [FeatureSpec(kind="bars_until_next_roll")])
    # explicit opt-out for offline research
    ff = compute_features(src, [FeatureSpec(kind="bars_until_next_roll")],
                          require_point_in_time=False)
    assert "bars_until_next_roll" in ff.feature_names
    assert ff.safety["bars_until_next_roll"].point_in_time_safe is False
    with pytest.raises(FeatureSafetyError, match="signal-safe"):
        ff.assert_signal_safe()
