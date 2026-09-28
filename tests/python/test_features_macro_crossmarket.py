"""Phase 09 -- COT / macro point-in-time alignment and cross-market backward
alignment. No data is acquired; these test the typed extension points."""
from __future__ import annotations

import math

import pandas as pd
import pytest
from alpha_agent.features.crossmarket import PairSpec, align_backward, build_cross_market_feature
from alpha_agent.features.macro import (
    COTReport,
    MacroRelease,
    PointInTimeDatum,
    PointInTimeSeries,
    align_as_of,
)

DAY = 86_400_000_000_000


def test_macro_release_published_after_T_never_appears_at_T():
    series = PointInTimeSeries(
        series_id="CPI",
        kind="macro_release",
        data=(
            PointInTimeDatum(reference_ts_ns=10 * DAY, published_ts_ns=15 * DAY, value=3.1),
            PointInTimeDatum(reference_ts_ns=40 * DAY, published_ts_ns=45 * DAY, value=3.4),
        ),
    )
    feature_ts = [12 * DAY, 15 * DAY, 20 * DAY, 44 * DAY, 45 * DAY, 60 * DAY]
    out = align_as_of(feature_ts, series)
    # before the first publication -> missing; then 3.1 until the second publication
    assert math.isnan(out.iloc[0])
    assert out.iloc[1] == 3.1 and out.iloc[2] == 3.1 and out.iloc[3] == 3.1
    assert out.iloc[4] == 3.4 and out.iloc[5] == 3.4


def test_as_of_clamp_hides_later_publications_for_walk_forward():
    series = PointInTimeSeries(
        series_id="NFP",
        data=(
            PointInTimeDatum(reference_ts_ns=1 * DAY, published_ts_ns=5 * DAY, value=100.0),
            PointInTimeDatum(reference_ts_ns=30 * DAY, published_ts_ns=35 * DAY, value=200.0),
        ),
    )
    out = align_as_of([40 * DAY], series, as_of_ts_ns=20 * DAY)
    assert out.iloc[0] == 100.0            # the 35-day publication is not yet known at as_of=20d


def test_cot_report_uses_the_release_date_not_the_report_date():
    rep = COTReport(report_date_ns=10 * DAY, release_ts_ns=13 * DAY, net_position=1234.0)
    d = rep.to_datum()
    assert d.reference_ts_ns == 10 * DAY and d.published_ts_ns == 13 * DAY
    series = PointInTimeSeries(series_id="COT_NQ", kind="cot", data=(d,))
    out = align_as_of([11 * DAY, 13 * DAY], series)
    assert math.isnan(out.iloc[0])        # report exists but is not released yet
    assert out.iloc[1] == 1234.0


def test_macro_release_surprise_requires_a_consensus():
    r = MacroRelease(reference_period_ns=DAY, release_ts_ns=2 * DAY, actual=3.5, consensus=3.2)
    assert r.surprise() == pytest.approx(0.3)
    assert r.to_datum(use_surprise=True).value == pytest.approx(0.3)
    r2 = MacroRelease(reference_period_ns=DAY, release_ts_ns=2 * DAY, actual=3.5)
    assert r2.surprise() is None
    with pytest.raises(ValueError):
        r2.to_datum(use_surprise=True)


def test_datum_rejects_publication_before_its_reference_period():
    with pytest.raises(ValueError):
        PointInTimeDatum(reference_ts_ns=10 * DAY, published_ts_ns=5 * DAY, value=1.0)


def test_max_staleness_drops_an_old_value():
    series = PointInTimeSeries(
        series_id="X",
        data=(PointInTimeDatum(reference_ts_ns=DAY, published_ts_ns=2 * DAY, value=9.0),),
    )
    fresh = align_as_of([3 * DAY], series, max_staleness_ns=5 * DAY)
    stale = align_as_of([100 * DAY], series, max_staleness_ns=5 * DAY)
    assert fresh.iloc[0] == 9.0 and math.isnan(stale.iloc[0])


# ---- cross-market -----------------------------------------------
def test_backward_alignment_never_uses_a_future_other_market_row():
    base = pd.DataFrame({"ts_event_ns": [10, 20, 30], "close": [1.0, 2.0, 3.0]})
    other = pd.DataFrame({"ts_event_ns": [12, 25], "close": [100.0, 200.0]})
    merged = align_backward(base, other, value_cols=("close",))
    # ts=10 -> no earlier other row -> NaN ; ts=20 -> the 12 row (100) ; ts=30 -> the 25 row (200)
    assert math.isnan(merged["close_other"].iloc[0])
    assert merged["close_other"].iloc[1] == 100.0
    assert merged["close_other"].iloc[2] == 200.0


def test_cross_market_spread_is_timestamp_aligned_backward():
    es = pd.DataFrame({"ts_event_ns": [0, 60, 120, 180], "close": [5000.0, 5010.0, 5020.0, 5030.0]})
    nq = pd.DataFrame({"ts_event_ns": [30, 90, 150], "close": [18000.0, 18010.0, 18020.0]})
    spec = PairSpec(base_root="ES", other_root="NQ", kind="spread")
    out = build_cross_market_feature({"ES": es, "NQ": nq}, spec)
    assert list(out.columns) == ["ts_event_ns", "xm_spread_ES_NQ"]
    assert math.isnan(out["xm_spread_ES_NQ"].iloc[0])           # nothing earlier in NQ
    assert out["xm_spread_ES_NQ"].iloc[1] == 5010.0 - 18000.0    # NQ@30 is the latest <= 60
    assert out["xm_spread_ES_NQ"].iloc[3] == 5030.0 - 18020.0
