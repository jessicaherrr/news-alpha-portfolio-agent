"""Phase 09 -- futures roll features and the retrospective vs point-in-time
distinction (checklist P)."""
from __future__ import annotations

import numpy as np
import pytest
from alpha_agent.features import (
    FeatureSafetyError,
    FeatureSpec,
    LookaheadUnsafeError,
    compute_features,
)
from features_fixtures import continuous_roll_source


def test_point_in_time_roll_features_use_only_observed_transitions():
    src = continuous_roll_source()  # id 501 x5, id 502 x6, observed roll at idx 5
    ff = compute_features(src, [
        FeatureSpec(kind="is_roll_day"),
        FeatureSpec(kind="bars_since_roll"),
        FeatureSpec(kind="days_since_roll"),
    ])
    np.testing.assert_array_equal(
        ff.features["is_roll_day"].to_numpy(),
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    )
    np.testing.assert_array_equal(
        ff.features["bars_since_roll"].to_numpy(),
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5],
    )
    np.testing.assert_array_equal(
        ff.features["days_since_roll"].to_numpy(),
        [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
    )
    ff.assert_signal_safe()  # all three are point-in-time / signal safe on a raw_continuous feed


def test_point_in_time_roll_feature_is_prefix_stable():
    src = continuous_roll_source()
    spec = FeatureSpec(kind="bars_since_roll")
    full = compute_features(src, [spec]).features["bars_since_roll"].to_numpy()
    for cut in (2, 3, 4, 7):
        as_of = int(src.frame["ts_event_ns"].iloc[cut])
        part = compute_features(src, [spec], as_of_ts_ns=as_of)
        np.testing.assert_array_equal(
            part.features["bars_since_roll"].to_numpy(), full[: cut + 1]
        )


def test_retrospective_roll_feature_would_leak_and_is_gated():
    src = continuous_roll_source()
    spec = FeatureSpec(kind="bars_until_next_roll")

    # blocked under the default point-in-time contract
    with pytest.raises(LookaheadUnsafeError):
        compute_features(src, [spec])

    # explicit offline research: it looks forward to the next observed roll
    full = compute_features(src, [spec], require_point_in_time=False)
    v = full.features["bars_until_next_roll"].to_numpy()
    np.testing.assert_array_equal(v[:6], [5, 4, 3, 2, 1, 0])
    assert np.isnan(v[6:]).all()                       # no "next" roll after the last one

    # and it is NOT prefix stable -- proving why it must never be point-in-time
    as_of = int(src.frame["ts_event_ns"].iloc[3])      # before the roll
    part = compute_features(src, [spec], require_point_in_time=False, as_of_ts_ns=as_of)
    assert np.isnan(part.features["bars_until_next_roll"].to_numpy()).all()
    assert not np.isnan(v[3])                          # the full series "knew" the future

    # retrospective roll features are neither point-in-time nor signal safe
    assert full.safety["bars_until_next_roll"].signal_safe is False
    with pytest.raises(FeatureSafetyError, match="signal-safe"):
        full.assert_signal_safe()


def test_contract_transition_indicator_is_retrospective():
    src = continuous_roll_source()
    ff = compute_features(src, [FeatureSpec(kind="contract_transition_indicator", params={"k": 1})],
                          require_point_in_time=False)
    np.testing.assert_array_equal(
        ff.features["contract_transition_indicator_1"].to_numpy(),
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    )
    assert ff.metadata["contract_transition_indicator_1"].point_in_time_safe is False
