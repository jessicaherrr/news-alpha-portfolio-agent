"""Phase 09 -- return features: 1-bar / multi-bar, signed prices, zero prices,
log-return validity (checklist A, F, G)."""
from __future__ import annotations

import math

import numpy as np
from alpha_agent.features import FeatureIssueKind, FeatureSpec, SessionPolicy, compute_features
from features_fixtures import SIGNED_CLOSES, signed_source, source_from_close


# ---- A. one-bar and multi-bar returns --------------------------------
def test_one_bar_and_multi_bar_returns_are_hand_computable():
    close = [100.0, 101.0, 103.0, 102.0, 108.0]
    src = source_from_close(close)
    ff = compute_features(src, [
        FeatureSpec(kind="return", params={"n": 1}),
        FeatureSpec(kind="return", params={"n": 2}),
        FeatureSpec(kind="diff", params={"n": 1}),
        FeatureSpec(kind="log_return", params={"n": 1}),
    ])
    r1 = ff.features["return_1"].to_numpy()
    assert math.isnan(r1[0])
    np.testing.assert_allclose(r1[1:], [0.01, 2 / 101, -1 / 103, 6 / 102])
    r2 = ff.features["return_2"].to_numpy()
    np.testing.assert_allclose(r2[2:], [3 / 100, 1 / 101, 5 / 103])
    np.testing.assert_allclose(ff.features["diff_1"].to_numpy()[1:], [1.0, 2.0, -1.0, 6.0])
    np.testing.assert_allclose(
        ff.features["log_return_1"].to_numpy()[1:],
        [math.log(101 / 100), math.log(103 / 101), math.log(102 / 103), math.log(108 / 102)],
    )


def test_return_uses_only_past_information():
    # return_1 at row i depends on close[i] and close[i-1] only
    src = source_from_close([10.0, 20.0, 40.0, 80.0])
    ff = compute_features(src, [FeatureSpec(kind="return", params={"n": 1})])
    np.testing.assert_allclose(ff.features["return_1"].to_numpy()[1:], [1.0, 1.0, 1.0])


# ---- F. signed-price behavior --------------------------------------
def test_diff_is_defined_across_zero_and_negative_prices():
    ff = compute_features(signed_source(), [FeatureSpec(kind="diff", params={"n": 1})])
    d = ff.features["diff_1"].to_numpy()
    expected = np.diff(np.asarray(SIGNED_CLOSES))
    np.testing.assert_allclose(d[1:], expected)
    assert not np.isinf(d).any()


def test_percentage_return_is_suppressed_where_the_base_is_negative_not_inf():
    ff = compute_features(signed_source(), [FeatureSpec(kind="return", params={"n": 1})])
    r = ff.features["return_1"].to_numpy()
    assert not np.isinf(r).any()
    # closes: 5, 3, 1, 0, -5, -20, ...
    np.testing.assert_allclose(r[3], -1.0)           # (0 - 1) / 1
    assert math.isnan(r[4])                          # base 0 -> undefined
    assert np.isnan(r[5:]).all()                     # negative base -> suppressed
    assert ff.qa.has_kind(FeatureIssueKind.NON_POSITIVE_RETURN_BASE, feature="return_1")
    assert ff.qa.has_kind(FeatureIssueKind.NEGATIVE_RETURN_BASE, feature="return_1")


# ---- G. zero-price behavior ---------------------------------------
def test_zero_price_makes_percentage_return_explicit_missing():
    src = source_from_close([10.0, 0.0, 5.0, 5.0])
    ff = compute_features(src, [
        FeatureSpec(kind="return", params={"n": 1}),
        FeatureSpec(kind="log_return", params={"n": 1}),
        FeatureSpec(kind="diff", params={"n": 1}),
    ])
    r = ff.features["return_1"].to_numpy()
    np.testing.assert_allclose(r[1], -1.0)           # (0-10)/10
    assert math.isnan(r[2])                          # (5-0)/0 -> undefined, not inf
    assert not np.isinf(r).any()
    lr = ff.features["log_return_1"].to_numpy()
    assert math.isnan(lr[1]) and math.isnan(lr[2])   # log of / by zero -> missing
    assert ff.qa.has_kind(FeatureIssueKind.NON_POSITIVE_LOG_INPUT, feature="log_return_1")
    np.testing.assert_allclose(ff.features["diff_1"].to_numpy()[1:], [-10.0, 5.0, 0.0])


def test_gap_aware_return_does_not_span_a_hole_when_policy_resets():
    # default session policy for returns is RESET_ON_GAP
    close = [100.0, 101.0, 102.0, 200.0, 201.0]
    ts = np.array([0, 1, 2, 500, 501], dtype="int64") * 60_000_000_000 + 1_000_000_000_000
    src = source_from_close(close, ts=ts)
    ff = compute_features(src, [FeatureSpec(kind="return", params={"n": 1})])
    r = ff.features["return_1"].to_numpy()
    assert math.isnan(r[3])                          # first bar after the hole: no in-segment base
    np.testing.assert_allclose(r[4], 1 / 200)
    # with CONTINUOUS the return spans the hole (still not silently -- caller opted in)
    ff2 = compute_features(src, [FeatureSpec(kind="return", params={"n": 1},
                                             session_policy=SessionPolicy.CONTINUOUS)])
    assert not math.isnan(ff2.features["return_1"].to_numpy()[3])
