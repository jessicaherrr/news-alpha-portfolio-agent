"""Phase 09 -- window behaviour: minimum observations, session-boundary reset,
missing-bar handling (checklist C, D, E)."""
from __future__ import annotations

import numpy as np
from alpha_agent.features import FeatureIssueKind, FeatureSpec, SessionPolicy, compute_features
from features_fixtures import (
    gap_source,
    linear_source,
    session_source,
)


# ---- C. rolling minimum observations --------------------------------
def test_rolling_feature_is_missing_until_min_observations_are_available():
    src = linear_source(20)
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 8})])
    m = ff.mask["ma_8"].to_numpy()
    assert m[:7].sum() == 0 and m[7:].all()
    first_ts = int(src.frame["ts_event_ns"].iloc[7])
    assert ff.availability.loc["ma_8", "first_available_ts_ns"] == first_ts


def test_explicit_min_observations_override_shrinks_the_warmup():
    src = linear_source(20)
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 8}, min_observations=3)])
    m = ff.mask["ma_8"].to_numpy()
    assert m[:2].sum() == 0 and m[2:].all()


def test_feature_that_never_reaches_min_obs_is_flagged():
    src = linear_source(4)
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 10})])
    assert not ff.mask["ma_10"].any()
    assert ff.qa.has_kind(FeatureIssueKind.INSUFFICIENT_LOOKBACK, feature="ma_10")


# ---- D. session boundary behaviour --------------------------------
def test_session_reset_restarts_the_window_at_a_new_trading_day():
    src = session_source()  # 6 bars on 2026-09-01, 6 on 2026-09-02
    spec = FeatureSpec(kind="ma", params={"window": 3},
                       session_policy=SessionPolicy.RESET_ON_SESSION)
    ff = compute_features(src, [spec])
    m = ff.mask["ma_3"].to_numpy()
    # day 1: first 2 missing, then present; day 2 (idx 6..): first 2 missing again
    assert list(m) == [False, False, True, True, True, True,
                       False, False, True, True, True, True]
    assert ff.qa.has_kind(FeatureIssueKind.SESSION_BOUNDARY_RESET, feature="ma_3")


def test_continuous_policy_does_not_reset_at_the_boundary():
    src = session_source()
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 3},
                                            session_policy=SessionPolicy.CONTINUOUS)])
    m = ff.mask["ma_3"].to_numpy()
    assert list(m) == [False, False] + [True] * 10


def test_session_reset_ma_value_uses_only_current_session_bars():
    src = session_source()
    close = src.frame["close"].to_numpy()
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 3},
                                            session_policy=SessionPolicy.RESET_ON_SESSION)])
    # first present value of day 2 is at idx 8 -> mean(close[6:9])
    np.testing.assert_allclose(ff.features["ma_3"].iloc[8], close[6:9].mean())


# ---- E. missing-bar behaviour ------------------------------------
def test_missing_bar_is_not_treated_as_a_zero_return():
    src = gap_source(gap_after=5, gap_intervals=200)
    ff = compute_features(src, [FeatureSpec(kind="diff", params={"n": 1})])  # RESET_ON_GAP default
    d = ff.features["diff_1"].to_numpy()
    # the bar right after the hole is explicit missing -- NOT a fabricated zero return
    assert np.isnan(d[5])
    np.testing.assert_allclose(d[6:], 1.0)
    assert ff.qa.has_kind(FeatureIssueKind.MISSING_BAR_GAP, feature="diff_1")


def test_rolling_window_does_not_span_a_large_gap_under_reset_on_gap():
    src = gap_source(gap_after=5, gap_intervals=200)
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 3},
                                            session_policy=SessionPolicy.RESET_ON_GAP)])
    m = ff.mask["ma_3"].to_numpy()
    # segment 1 = idx 0..4 (present from idx 2), segment 2 = idx 5..9 (present from idx 7)
    assert list(m) == [False, False, True, True, True, False, False, True, True, True]


def test_no_forward_fill_across_missing_data():
    src = gap_source()
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 2})])
    # rows are preserved 1:1 with the source; nothing is synthesised
    assert len(ff.features) == len(src.frame)
    assert ff.identifiers["ts_event_ns"].tolist() == src.frame["ts_event_ns"].tolist()
