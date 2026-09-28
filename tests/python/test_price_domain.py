"""Phase 04.5 -- the single price-normalization primitive + domain guards."""
from __future__ import annotations

import pandas as pd
import pytest
from alpha_agent.data.price_domain import (
    PriceScaleError,
    assert_same_price_domain,
    descale_fixed_point,
)


def test_fixed_point_decodes_like_the_canonical_path():
    # the exact real Stage B values
    df = pd.DataFrame({
        "ts_event_ns": [1, 2],
        "instrument_id": [42004058, 42004177],
        "open": [30031_750_000_000, 30343_250_000_000],
        "high": [30032_000_000_000, 30344_000_000_000],
        "low": [30031_000_000_000, 30343_000_000_000],
        "close": [30031_750_000_000, 30343_250_000_000],
        "volume": [10, 12],
    })
    out = descale_fixed_point(df)
    assert out["close"].tolist() == [30031.75, 30343.25]
    assert out["open"].tolist() == [30031.75, 30343.25]
    assert out["close"].dtype == "float64"
    # unchanged columns pass through
    assert out["instrument_id"].tolist() == [42004058, 42004177]


def test_undef_price_sentinel_becomes_nan():
    df = pd.DataFrame({"open": [1_000_000_000], "high": [2**63 - 1],
                       "low": [1_000_000_000], "close": [1_000_000_000]})
    out = descale_fixed_point(df)
    assert out["open"].iloc[0] == 1.0
    assert pd.isna(out["high"].iloc[0])


def test_refuses_to_double_scale_a_normalized_frame():
    df = pd.DataFrame({"open": [30031.75], "high": [30032.0], "low": [30031.0], "close": [30031.75]})
    with pytest.raises(PriceScaleError, match="already normalized"):
        descale_fixed_point(df)


def test_assert_same_price_domain_catches_fixed_vs_normalized():
    canonical = pd.DataFrame({"close": [30012.75, 30013.0, 30011.5]})
    normalized_overlap = pd.DataFrame({"close": [30031.75, 30343.25]})
    raw_overlap = pd.DataFrame({"close": [30031_750_000_000, 30343_250_000_000]})

    assert_same_price_domain(canonical, normalized_overlap, label="ok")   # no raise
    with pytest.raises(PriceScaleError, match="price-domain mismatch"):
        assert_same_price_domain(canonical, raw_overlap, label="bad")
