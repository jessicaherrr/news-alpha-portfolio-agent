"""The daily trading-day signal-series feature helper.

Regression cover for the Phase 15B.2 defect: on the real daily forward-adjusted
signal path, ``SessionPolicy.RESET_ON_GAP`` features (cum_log_return /
realized_vol / vol_percentile) were entirely non-finite because every weekend
looked like a missing-bar hole. ``compute_contiguous_daily_features`` computes
on a gap-free synthetic day index -- the frozen Phase 13.5C technique.
"""
from __future__ import annotations

import alpha_agent.features.compute  # noqa: F401
import numpy as np
import pandas as pd
import pytest
from alpha_agent.features import compute_features
from alpha_agent.features.daily import NS_PER_DAY, compute_contiguous_daily_features
from alpha_agent.features.source import SourceSeries
from alpha_agent.features.spec import FeatureSpec
from alpha_agent.schemas.market_data import PriceDomain

CAL_DAY = 86_400_000_000_000


def _weekend_gapped_daily(n: int, *, seed: int = 0) -> pd.DataFrame:
    """One row per trading day, Mon-Fri, with a real ~3-day step every weekend --
    the shape ``daily_signal_series`` returns."""
    rng = np.random.default_rng(seed)
    close = 4000.0 * np.exp(np.cumsum(0.0003 + 0.011 * rng.standard_normal(n)))
    openp = np.empty_like(close)
    openp[0] = close[0]
    openp[1:] = close[:-1]
    high = np.maximum(openp, close) * 1.004
    low = np.minimum(openp, close) * 0.996
    vol = rng.integers(400_000, 800_000, n)
    # walk a Monday-anchored calendar: +1 day Tue-Fri, +3 days over the weekend
    start = pd.Timestamp("2019-01-07T22:00:00Z").value      # a Monday
    ts = np.empty(n, dtype="int64")
    ts[0] = start
    dow = 0
    for i in range(1, n):
        step = 3 if dow == 4 else 1
        ts[i] = ts[i - 1] + step * CAL_DAY
        dow = (dow + step) % 7
    return pd.DataFrame(
        {"ts_event_ns": ts, "open": openp, "high": high, "low": low,
         "close": close, "volume": vol}
    )


FROZEN_11 = [
    ("cum_log_return", {"window": 20}, 20),
    ("cum_log_return", {"window": 120}, 120),
    ("realized_vol", {"window": 20}, 20),
    ("realized_vol", {"window": 60}, 60),
    ("atr", {"window": 14}, 13),
    ("zscore", {"window": 20}, 19),
    ("dist_from_ma", {"window": 50}, 49),
    ("donchian_pos", {"window": 55}, 54),
    ("trend_strength", {"fast": 50, "slow": 200}, 199),
    ("vol_percentile", {"window": 20, "lookback": 252}, 270),
    ("volume_zscore", {"window": 20}, 19),
]


def _specs():
    return [FeatureSpec(kind=k, params=p) for k, p, _ in FROZEN_11]


def test_reset_on_gap_features_are_starved_without_the_contiguous_index():
    """The bug: fed the real (weekend-gapped) timestamps straight in, every
    window longer than a trading week never fills."""
    daily = _weekend_gapped_daily(400)
    src = SourceSeries(
        frame=daily[["ts_event_ns", "open", "high", "low", "close", "volume"]],
        price_domain=PriceDomain.BACK_ADJUSTED, adjustment_mode="forward_adjusted",
        identity={"root_symbol": "NQ"}, interval_ns=CAL_DAY,
    )
    feats = compute_features(src, _specs(), require_point_in_time=True)
    finite = feats.features.notna().sum()
    assert finite["realized_vol_20"] == 0        # starved
    assert finite["cum_log_return_120"] == 0
    assert finite["vol_percentile_20_252"] == 0
    assert finite["atr_14"] > 300               # CONTINUOUS-policy feature is fine


def test_contiguous_helper_makes_every_frozen_feature_finite_after_warmup():
    daily = _weekend_gapped_daily(600)
    res = compute_contiguous_daily_features(
        daily, _specs(), root_symbol="NQ",
        price_domain=PriceDomain.BACK_ADJUSTED, adjustment_mode="forward_adjusted",
        require_point_in_time=True,
    )
    f = res.frame.features
    from alpha_agent.features.registry import REGISTRY
    for kind, params, expected_warmup in FROZEN_11:
        cname = REGISTRY.get(kind).canonical_name(FeatureSpec(kind=kind, params=params))
        v = f[cname].to_numpy(float)
        first = int(np.isfinite(v).argmax())
        assert np.isfinite(v).any(), f"{cname} never finite"
        assert first == expected_warmup, f"{cname} warm-up {first} != {expected_warmup}"
        assert np.all(np.isfinite(v[first:])), f"{cname} has interior NaN after warm-up"


def test_helper_realigns_identifiers_and_availability_to_real_timestamps():
    daily = _weekend_gapped_daily(300)
    res = compute_contiguous_daily_features(
        daily, _specs(), root_symbol="ES",
        price_domain=PriceDomain.BACK_ADJUSTED, adjustment_mode="forward_adjusted",
    )
    assert np.array_equal(
        res.frame.identifiers["ts_event_ns"].to_numpy("int64"),
        daily["ts_event_ns"].to_numpy("int64"),
    )
    assert np.array_equal(res.synthetic_ts_ns, (np.arange(300) + 1) * NS_PER_DAY)
    avail = res.frame.availability["first_available_ts_ns"].dropna().to_numpy("int64")
    assert set(avail.tolist()).issubset(set(daily["ts_event_ns"].tolist()))


def test_helper_equals_the_frozen_13_5c_synthetic_index_technique():
    """`phase_13_5c_matrix._baseline_schedule` predates this helper and is left
    byte-stable; prove the helper reproduces its inline technique exactly."""
    daily = _weekend_gapped_daily(500, seed=3)
    specs = _specs()

    # -- the inline technique, copied from phase_13_5c_matrix._baseline_schedule
    real_ts = daily["ts_event_ns"].to_numpy("int64")
    synth = (np.arange(len(daily), dtype="int64") + 1) * NS_PER_DAY
    frame = daily.assign(ts_event_ns=synth)
    src = SourceSeries(
        frame=frame[["ts_event_ns", "open", "high", "low", "close", "volume"]],
        price_domain=PriceDomain.BACK_ADJUSTED, adjustment_mode="forward_adjusted",
        identity={"root_symbol": "NQ"}, interval_ns=NS_PER_DAY,
    )
    inline = compute_features(src, specs, require_point_in_time=False)

    helper = compute_contiguous_daily_features(
        daily, specs, root_symbol="NQ", price_domain=PriceDomain.BACK_ADJUSTED,
        adjustment_mode="forward_adjusted", require_point_in_time=False,
    )
    pd.testing.assert_frame_equal(
        inline.features.reset_index(drop=True),
        helper.frame.features.reset_index(drop=True),
    )
    assert np.array_equal(helper.real_ts_ns, real_ts)


def test_return_and_realized_vol_match_an_independent_reference():
    daily = _weekend_gapped_daily(120, seed=7)
    res = compute_contiguous_daily_features(
        daily, [FeatureSpec(kind="cum_log_return", params={"window": 20}),
                FeatureSpec(kind="realized_vol", params={"window": 20})],
        root_symbol="NQ", price_domain=PriceDomain.BACK_ADJUSTED,
        adjustment_mode="forward_adjusted", require_point_in_time=True,
    )
    close = daily["close"].to_numpy(float)
    lr = np.diff(np.log(close))                          # 1-bar log returns, len n-1
    n = len(close)
    ref_cum = np.full(n, np.nan)
    ref_rv = np.full(n, np.nan)
    for i in range(n):
        # rows lr[i-20 .. i-1] are the 20 one-bar returns ending at row i
        if i >= 20:
            w = lr[i - 20:i]
            ref_cum[i] = w.sum()
            ref_rv[i] = np.sqrt((w * w).sum())
    got_cum = res.frame.features["cum_log_return_20"].to_numpy(float)
    got_rv = res.frame.features["realized_vol_20"].to_numpy(float)
    np.testing.assert_allclose(got_cum[20:], ref_cum[20:], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got_rv[20:], ref_rv[20:], rtol=1e-9, atol=1e-12)


def test_contiguous_helper_introduces_no_lookahead():
    """A value at row i is identical whether the series ends at i or later."""
    daily = _weekend_gapped_daily(400, seed=11)
    specs = _specs()
    full = compute_contiguous_daily_features(
        daily, specs, root_symbol="NQ", price_domain=PriceDomain.BACK_ADJUSTED,
        adjustment_mode="forward_adjusted",
    ).frame.features
    cut = 320
    trunc = compute_contiguous_daily_features(
        daily.iloc[:cut].reset_index(drop=True), specs, root_symbol="NQ",
        price_domain=PriceDomain.BACK_ADJUSTED, adjustment_mode="forward_adjusted",
    ).frame.features
    pd.testing.assert_frame_equal(
        full.iloc[:cut].reset_index(drop=True), trunc.reset_index(drop=True)
    )


def test_helper_rejects_unsorted_or_gappy_frames():
    daily = _weekend_gapped_daily(30)
    bad = daily.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        compute_contiguous_daily_features(
            bad, _specs(), root_symbol="NQ", price_domain=PriceDomain.BACK_ADJUSTED,
            adjustment_mode="forward_adjusted",
        )
    with pytest.raises(ValueError):
        compute_contiguous_daily_features(
            daily.drop(columns=["volume"]), _specs(), root_symbol="NQ",
            price_domain=PriceDomain.BACK_ADJUSTED, adjustment_mode="forward_adjusted",
        )
