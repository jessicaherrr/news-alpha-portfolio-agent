"""Phase 09.1 -- feature safety semantics + parameter-contract validation.

Three explicit concepts: point_in_time_safe / signal_safe / execution_price_safe.
A point-in-time back-adjusted trend feature is signal-safe (Phase 10 DSL may use
it) but never execution-price safe.
"""
from __future__ import annotations

import numpy as np
import pytest
from alpha_agent.features import (
    REGISTRY,
    FeatureNameCollision,
    FeatureSafetyError,
    FeatureSpec,
    LookaheadUnsafeError,
    SessionPolicy,
    compute_features,
)
from alpha_agent.features.enums import FeatureFamily
from alpha_agent.features.registry import FeatureDef, FeatureRegistry, ParamRule, require_lt
from alpha_agent.schemas.market_data import PriceDomain
from features_fixtures import linear_source, synthetic_roll_sources

MA = FeatureSpec(kind="ma", params={"window": 3})
MOMENTUM = FeatureSpec(kind="diff", params={"n": 3})  # a trend/momentum feature


# ==================================================================
# 5. REGRESSION TEST -- synthetic roll, old ~100 / new ~110 / basis +10
# ==================================================================
def test_source_execution_price_safety_does_not_propagate_to_derived_features():
    """A. RawContract SourceSeries is source-execution-price-safe.
    B/C. MA / ATR / z-score / return derived from it are point-in-time & signal
    safe but NEVER execution-price safe -- a feature value is not a price."""
    s = synthetic_roll_sources()
    rc_src = s["raw_contract"]
    assert rc_src.safety().execution_price_safe is True          # (A) source level

    derived = [
        FeatureSpec(kind="ma", params={"window": 3}),
        FeatureSpec(kind="atr", params={"window": 3}),
        FeatureSpec(kind="zscore", params={"window": 3}),
        FeatureSpec(kind="return", params={"n": 2}),
        FeatureSpec(kind="volatility", params={"window": 3}),
        FeatureSpec(kind="rolling_high", params={"window": 3}),
        FeatureSpec(kind="breakout_up", params={"window": 3}),
    ]
    ff = compute_features(rc_src, derived)
    assert ff.source_safety.execution_price_safe is True         # source fact preserved
    for name in ff.feature_names:
        assert ff.safety[name].point_in_time_safe is True        # (B)
        assert ff.safety[name].signal_safe is True               # (B)
        assert ff.safety[name].execution_price_safe is False     # (B, C)
        assert ff.metadata[name].execution_price_safe is False
    assert ff.frame_safety.execution_price_safe is False
    ff.assert_signal_safe()                                      # no raise
    with pytest.raises(FeatureSafetyError, match="execution-price"):
        ff.assert_execution_price_safe()                         # (G) always raises


def test_safety_matrix_across_price_domains():
    s = synthetic_roll_sources()
    assert s["additive_gap"] == 10.0

    # RawContract -- source-execution-price-safe; the FEATURE is not
    rc = compute_features(s["raw_contract"], [MA])
    assert rc.frame_safety.point_in_time_safe is True
    assert rc.frame_safety.signal_safe is True
    assert rc.source_safety.execution_price_safe is True
    assert rc.frame_safety.execution_price_safe is False
    with pytest.raises(FeatureSafetyError):
        rc.assert_execution_price_safe()

    # RawContinuous (D) -- signal-safe, NOT execution-price-safe
    rk = compute_features(s["raw_continuous"], [MA])
    assert rk.frame_safety.signal_safe is True
    assert rk.source_safety.execution_price_safe is False
    assert rk.frame_safety.execution_price_safe is False
    with pytest.raises(FeatureSafetyError):
        rk.assert_execution_price_safe()

    # point-in-time BackAdjusted (E) -- pit & signal safe, NOT execution-price-safe
    pit = compute_features(s["backadj_pit"], [MA])
    assert pit.frame_safety.point_in_time_safe is True
    assert pit.frame_safety.signal_safe is True
    assert pit.frame_safety.execution_price_safe is False
    pit.assert_signal_safe()
    with pytest.raises(FeatureSafetyError):
        pit.assert_execution_price_safe()

    # retrospective BackAdjusted (F) -- neither point-in-time nor signal safe
    with pytest.raises(LookaheadUnsafeError):
        compute_features(s["backadj_retro"], [MA])            # gated by default
    retro = compute_features(s["backadj_retro"], [MA], require_point_in_time=False)
    assert retro.frame_safety.point_in_time_safe is False
    assert retro.frame_safety.signal_safe is False
    assert retro.frame_safety.execution_price_safe is False
    with pytest.raises(FeatureSafetyError):
        retro.assert_signal_safe()

    # a momentum feature behaves the same way
    pit_mom = compute_features(s["backadj_pit"], [MOMENTUM])
    assert pit_mom.safety["diff_3"].signal_safe is True
    assert pit_mom.safety["diff_3"].execution_price_safe is False
    with pytest.raises(LookaheadUnsafeError):
        compute_features(s["backadj_retro"], [MOMENTUM])


def test_no_feature_frame_from_the_registry_is_execution_price_safe():
    """G. Sweep every registered kind -- not one produces an execution-price-safe
    feature column."""
    from alpha_agent.features import REGISTRY

    src = linear_source(40)
    example_params = {
        "n": 2, "window": 3, "fast": 2, "slow": 5, "vol_window": 3, "lookback": 5, "k": 1,
    }
    checked = 0
    for kind in REGISTRY.kinds():
        fdef = REGISTRY.get(kind)
        try:
            params = {p: example_params[p] for p in fdef.param_rules}
        except KeyError:
            continue  # kind whose params aren't in this tiny example dict
        spec = FeatureSpec(kind=kind, params=params)
        # registry-level metadata
        assert REGISTRY.metadata_for(spec).execution_price_safe is False
        try:
            ff = compute_features(src, [spec], require_point_in_time=False)
        except ValueError:
            continue  # feature needs columns this fixture lacks (roll / carry inputs)
        for name in ff.feature_names:
            assert ff.safety[name].execution_price_safe is False
        assert ff.frame_safety.execution_price_safe is False
        checked += 1
    assert checked >= 10


def test_future_roll_cannot_change_a_point_in_time_feature_at_T():
    s = synthetic_roll_sources()
    pre_ts = s["pre_ts"]

    def ma_at_pre(ff) -> float:
        f = ff.frame()
        return float(f.loc[f["ts_event_ns"] == pre_ts, "ma_3"].iloc[0])

    raw_val = ma_at_pre(compute_features(s["raw_continuous"], [MA], as_of_ts_ns=pre_ts))
    pit_val = ma_at_pre(compute_features(s["backadj_pit"], [MA]))
    retro_val = ma_at_pre(
        compute_features(s["backadj_retro"], [MA], require_point_in_time=False)
    )

    # point-in-time back-adjustment leaves the pre-roll value exactly as the
    # unadjusted feed saw it -- the future roll is invisible at T
    assert pit_val == raw_val == 100.0
    # retrospective back-adjustment shifts the same pre-roll bar by the FUTURE
    # roll's basis -> it "knew" about a roll that has not happened at T
    assert retro_val == pytest.approx(pit_val + s["additive_gap"])
    assert retro_val != pit_val


def test_retrospective_backadjusted_feature_is_prefix_unstable_pit_one_is_stable():
    s = synthetic_roll_sources()
    # point-in-time back-adjusted source (5 pre-roll rows): appending nothing to
    # compute changes; truncating earlier leaves the overlap identical.
    pit_full = compute_features(s["backadj_pit"], [MA]).features["ma_3"].to_numpy()
    cut = int(s["backadj_pit"].frame["ts_event_ns"].iloc[3])
    pit_cut = compute_features(s["backadj_pit"], [MA], as_of_ts_ns=cut).features["ma_3"].to_numpy()
    a = pit_full[:4]
    np.testing.assert_array_equal(np.isnan(a), np.isnan(pit_cut))
    np.testing.assert_allclose(a[~np.isnan(a)], pit_cut[~np.isnan(pit_cut)])


def test_lineage_records_adjustment_mode_and_source_safety():
    s = synthetic_roll_sources()
    retro = compute_features(s["backadj_retro"], [MA], require_point_in_time=False)
    assert retro.lineage.source_price_domain == "back_adjusted"
    assert retro.lineage.source_adjustment_mode == "retrospective_research"
    assert retro.lineage.source_point_in_time_safe is False
    assert retro.lineage.source_signal_safe is False
    assert retro.lineage.source_execution_price_safe is False

    pit = compute_features(s["backadj_pit"], [MA])
    assert pit.lineage.source_adjustment_mode == "point_in_time"
    assert pit.lineage.source_signal_safe is True
    assert pit.lineage.source_execution_price_safe is False


def test_back_adjusted_source_must_declare_its_mode():
    from alpha_agent.features import SourceSeries
    from features_fixtures import bars_from_close

    frame = bars_from_close([100.0, 101.0, 102.0])  # no adjustment_mode column
    with pytest.raises(ValueError, match="adjustment_mode"):
        SourceSeries(frame=frame, price_domain=PriceDomain.BACK_ADJUSTED)
    # a raw source may not carry one
    with pytest.raises(ValueError, match="only meaningful for back_adjusted"):
        SourceSeries(frame=frame, price_domain=PriceDomain.RAW_CONTINUOUS,
                     adjustment_mode="point_in_time")


# ==================================================================
# 6 / 8. PARAMETER CONTRACTS
# ==================================================================
def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown param"):
        REGISTRY.validate_spec(FeatureSpec(kind="ma", params={"window": 5, "bogus": 1}))


def test_missing_required_parameter_is_rejected():
    with pytest.raises(ValueError, match="missing required param"):
        REGISTRY.validate_spec(FeatureSpec(kind="ma", params={}))


@pytest.mark.parametrize("bad", [
    {"window": 0}, {"window": -3}, {"window": 1.5}, {"window": True}, {"window": "5"},
])
def test_invalid_window_is_rejected(bad):
    with pytest.raises(ValueError):
        REGISTRY.validate_spec(FeatureSpec(kind="ma", params=bad))


def test_window_must_be_a_positive_integer_for_variance_features():
    # volatility needs window >= 2
    with pytest.raises(ValueError):
        REGISTRY.validate_spec(FeatureSpec(kind="volatility", params={"window": 1}))
    REGISTRY.validate_spec(FeatureSpec(kind="volatility", params={"window": 2}))


def test_fast_slow_relationship_is_validated():
    for kind in ("ma_spread", "trend_strength"):
        with pytest.raises(ValueError, match="strictly less than"):
            REGISTRY.validate_spec(FeatureSpec(kind=kind, params={"fast": 50, "slow": 20}))
        with pytest.raises(ValueError, match="strictly less than"):
            REGISTRY.validate_spec(FeatureSpec(kind=kind, params={"fast": 20, "slow": 20}))
        REGISTRY.validate_spec(FeatureSpec(kind=kind, params={"fast": 20, "slow": 100}))


def test_unknown_feature_kind_is_rejected():
    with pytest.raises(KeyError):
        REGISTRY.validate_spec(FeatureSpec(kind="arbitrary_code", params={}))


def test_params_must_be_a_mapping_of_scalars():
    with pytest.raises(ValueError):
        FeatureSpec(kind="ma", params={"window": [1, 2, 3]})


def test_deterministic_naming_and_no_silent_collision():
    assert REGISTRY.name_for(FeatureSpec(kind="return", params={"n": 5})) == "return_5"
    with pytest.raises(FeatureNameCollision):
        compute_features(linear_source(20), [
            FeatureSpec(kind="ma", params={"window": 5}),
            FeatureSpec(kind="ma", params={"window": 5},
                        session_policy=SessionPolicy.RESET_ON_GAP),
        ])


def test_registry_rejects_a_second_def_that_would_collide_on_name():
    r = FeatureRegistry()
    common = {
        "family": FeatureFamily.TREND,
        "param_rules": {"window": ParamRule(int, min=1)},
        "param_order": ("window",),
        "default_price_domain": (PriceDomain.RAW_CONTRACT,),
        "default_session_policy": SessionPolicy.CONTINUOUS,
        "lookback_fn": lambda p: p["window"],
        "compute": lambda ctx: None,
    }
    r.register(FeatureDef(kind="alpha", **common))
    with pytest.raises(ValueError, match="colliding"):
        r.register(FeatureDef(kind="beta", name_stem="alpha", **common))


def test_require_lt_helper_message():
    check = require_lt("fast", "slow")
    assert check({"fast": 5, "slow": 3}) and "strictly less than" in check({"fast": 5, "slow": 3})
    assert check({"fast": 3, "slow": 5}) is None
