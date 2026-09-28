"""Phase 09 -- feature engine core: naming, metadata, registry safety,
price-domain enforcement, lineage, missing representation, deterministic replay."""
from __future__ import annotations

import numpy as np
import pytest
from alpha_agent.features import (
    FEATURE_ENGINE_VERSION,
    REGISTRY,
    FeatureFrame,
    FeatureNameCollision,
    FeaturePriceDomainError,
    FeatureSafetyError,
    FeatureSpec,
    SessionPolicy,
    SourceSeries,
    compute_features,
    feature_cache_key,
)
from alpha_agent.schemas.market_data import PriceDomain
from features_fixtures import linear_source, signed_source


# ---- N. deterministic feature naming -------------------------------------
def test_canonical_names_are_stable_and_parameterized():
    name = REGISTRY.name_for
    assert name(FeatureSpec(kind="return", params={"n": 20})) == "return_20"
    assert name(FeatureSpec(kind="volatility", params={"window": 60})) == "volatility_60"
    assert name(FeatureSpec(kind="zscore", params={"window": 30})) == "zscore_30"
    ms = FeatureSpec(kind="ma_spread", params={"fast": 20, "slow": 100})
    assert name(ms) == "ma_spread_20_100"
    # identical spec -> identical name, every time
    s = FeatureSpec(kind="atr", params={"window": 14})
    assert REGISTRY.name_for(s) == REGISTRY.name_for(s) == "atr_14"


def test_non_default_price_field_is_reflected_in_the_name():
    assert (
        REGISTRY.name_for(FeatureSpec(kind="ma", params={"window": 10}, price_field="open"))
        == "ma_10_open"
    )


def test_two_specs_with_the_same_name_but_different_meaning_collide():
    src = linear_source(30)
    a = FeatureSpec(kind="ma", params={"window": 10})
    b = FeatureSpec(kind="ma", params={"window": 10}, session_policy=SessionPolicy.RESET_ON_GAP)
    with pytest.raises(FeatureNameCollision):
        compute_features(src, [a, b])


# ---- 13. feature metadata ----------------------------------------------
def test_feature_metadata_is_deterministic_and_complete():
    spec = FeatureSpec(kind="zscore", params={"window": 30})
    md = REGISTRY.metadata_for(spec)
    assert md.feature_name == "zscore_30"
    assert md.version == FEATURE_ENGINE_VERSION
    assert md.family == "reversion"
    assert md.lookback == 30
    assert md.minimum_observations == 30
    assert md.session_policy == "continuous"
    assert md.point_in_time_safe is True
    assert md.signal_safe is True
    assert md.execution_price_safe is False  # a computed feature value is never an execution price
    assert md.price_field == "close"
    assert md.parameters == {"window": 30}
    assert md.required_price_domain  # non-empty


def test_min_observations_override_is_respected():
    spec = FeatureSpec(kind="volatility", params={"window": 20}, min_observations=5)
    assert REGISTRY.metadata_for(spec).minimum_observations == 5


# ---- 15. registry does not execute arbitrary code ---------------------
def test_registry_rejects_unknown_kinds_and_params():
    src = linear_source(20)
    with pytest.raises(KeyError):
        compute_features(src, [FeatureSpec(kind="__import__", params={})])
    with pytest.raises(ValueError, match="unknown param"):
        compute_features(src, [FeatureSpec(kind="ma", params={"window": 5, "evil": 1})])


def test_param_rules_enforce_type_and_bounds():
    src = linear_source(20)
    with pytest.raises(ValueError):
        compute_features(src, [FeatureSpec(kind="ma", params={"window": 0})])
    with pytest.raises(ValueError):
        compute_features(src, [FeatureSpec(kind="ma", params={"window": 1.5})])


# ---- 2. price-domain enforcement -------------------------------------
def test_price_domain_gate_blocks_a_disallowed_source():
    src = signed_source(domain=PriceDomain.BACK_ADJUSTED)
    spec = FeatureSpec(
        kind="ma", params={"window": 3},
        required_price_domain=(PriceDomain.RAW_CONTRACT,),
    )
    with pytest.raises(FeaturePriceDomainError):
        compute_features(src, [spec])


def test_point_in_time_back_adjusted_source_is_signal_safe_not_execution_price_safe():
    src = linear_source(20, domain=PriceDomain.BACK_ADJUSTED)  # mode defaults to point_in_time
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 5})])
    assert ff.frame_safety.point_in_time_safe is True
    assert ff.frame_safety.signal_safe is True
    ff.assert_signal_safe()  # no raise -- a PIT back-adjusted trend feature is a valid Signal input
    assert ff.frame_safety.execution_price_safe is False
    with pytest.raises(FeatureSafetyError, match="execution-price"):
        ff.assert_execution_price_safe()


def test_raw_continuous_source_is_signal_safe_not_execution_price_safe():
    src = linear_source(20)  # RAW_CONTINUOUS
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 5})])
    ff.assert_signal_safe()  # no raise
    assert ff.frame_safety.execution_price_safe is False
    with pytest.raises(FeatureSafetyError, match="execution-price"):
        ff.assert_execution_price_safe()


def test_raw_contract_source_is_source_execution_price_safe_but_the_feature_is_not():
    from features_fixtures import bars_from_close

    frame = bars_from_close([100.0, 101.0, 102.0, 103.0, 104.0], spread=0.0)
    frame["raw_symbol"] = "NQU6"
    frame["root_symbol"] = "NQ"
    src = SourceSeries(frame=frame, price_domain=PriceDomain.RAW_CONTRACT,
                       identity={"raw_symbol": "NQU6"})
    assert src.safety().execution_price_safe is True             # SOURCE level: a real contract feed
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 3})])
    assert ff.source_safety.execution_price_safe is True         # source fact preserved on the frame
    # ...but the DERIVED feature value is never an execution price
    assert ff.frame_safety.execution_price_safe is False
    assert ff.safety["ma_3"].execution_price_safe is False
    assert ff.metadata["ma_3"].execution_price_safe is False
    with pytest.raises(FeatureSafetyError, match="execution-price"):
        ff.assert_execution_price_safe()
    ff.assert_signal_safe()  # it IS signal-safe


# ---- 16. FeatureFrame shape & missing representation ------------------
def test_feature_frame_separates_identifiers_from_features_and_never_drops_rows():
    src = linear_source(30)
    ff = compute_features(src, [FeatureSpec(kind="ma", params={"window": 10})])
    assert isinstance(ff, FeatureFrame)
    assert len(ff.features) == len(src.frame) == len(ff.identifiers) == len(ff.mask)
    assert "continuous_symbol" in ff.identifiers.columns
    assert list(ff.features.columns) == ["ma_10"]
    # first 9 rows are explicit missing, not dropped
    assert ff.mask["ma_10"].tolist()[:9] == [False] * 9
    assert bool(ff.mask["ma_10"].iloc[9]) is True
    assert np.isnan(ff.features["ma_10"].iloc[:9]).all()
    rep = ff.missing_report().set_index("feature").loc["ma_10"]
    assert rep["n_missing"] == 9 and rep["n_present"] == 21


# ---- R. source lineage preservation ---------------------------------
def test_lineage_records_source_provenance():
    src = linear_source(25)
    ff = compute_features(src, [FeatureSpec(kind="return", params={"n": 1}),
                                FeatureSpec(kind="ma", params={"window": 5})])
    lin = ff.lineage
    assert lin.engine_version == FEATURE_ENGINE_VERSION
    assert lin.source_price_domain == "raw_continuous"
    assert lin.n_source_rows == 25
    assert lin.source_fingerprint == src.fingerprint()
    assert sorted(lin.feature_names) == ["ma_5", "return_1"]
    assert lin.identity == {"continuous_symbol": "NQ.v.0"}
    assert lin.source_adjustment_mode is None
    assert lin.source_signal_safe is True
    assert lin.source_execution_price_safe is False   # raw_continuous
    assert len(lin.feature_specs) == 2


def test_source_fingerprint_changes_with_content():
    a = linear_source(20).fingerprint()
    b = linear_source(20, step=0.6).fingerprint()
    assert a != b


# ---- O. deterministic replay + cache keys ---------------------------
def test_identical_inputs_reproduce_identical_values():
    specs = [
        FeatureSpec(kind="return", params={"n": 3}),
        FeatureSpec(kind="volatility", params={"window": 10}),
        FeatureSpec(kind="zscore", params={"window": 15}),
        FeatureSpec(kind="atr", params={"window": 10}),
    ]
    ff1 = compute_features(linear_source(60), specs)
    ff2 = compute_features(linear_source(60), specs)
    for name in ff1.feature_names:
        np.testing.assert_array_equal(
            ff1.features[name].to_numpy(), ff2.features[name].to_numpy()
        )
    assert ff1.cache_keys() == ff2.cache_keys()


def test_cache_key_depends_on_spec_source_and_engine_version():
    fp = linear_source(10).fingerprint()
    s1 = FeatureSpec(kind="ma", params={"window": 5})
    s2 = FeatureSpec(kind="ma", params={"window": 6})
    assert feature_cache_key(fp, s1) != feature_cache_key(fp, s2)
    assert feature_cache_key(fp, s1) != feature_cache_key(fp, s1, engine_version="9.9.9")
    assert feature_cache_key(fp, s1) == feature_cache_key(fp, s1)


def test_cache_is_used_and_returns_consistent_values():
    cache: dict = {}
    specs = [FeatureSpec(kind="ma", params={"window": 5})]
    ff1 = compute_features(linear_source(20), specs, cache=cache)
    assert len(cache) == 1
    ff2 = compute_features(linear_source(20), specs, cache=cache)
    np.testing.assert_array_equal(
        ff1.features["ma_5"].to_numpy(), ff2.features["ma_5"].to_numpy()
    )
