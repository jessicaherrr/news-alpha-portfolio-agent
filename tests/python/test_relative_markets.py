"""Market Intelligence Data Completion Pass, Checkpoint D -- the related-
market registry + deterministic cross-market metrics. Pure, no network.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from alpha_agent.marketdata.relative_markets import (
    MIN_COMMON_OBSERVATIONS,
    RelationType,
    align_by_timestamp,
    build_relative_snapshot,
    group_for,
    log_returns,
    normalized_returns,
    pearson_correlation,
    related_markets_for,
)

# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_related_markets_never_include_the_root_itself():
    peers = related_markets_for("NQ")
    assert "NQ" not in {p.peer_root_symbol for p in peers}


def test_related_markets_for_nq_matches_the_equity_index_group():
    peers = {p.peer_root_symbol for p in related_markets_for("NQ")}
    assert peers == {"ES", "YM", "RTY"}


def test_related_markets_for_cl_matches_the_energy_group():
    peers = {p.peer_root_symbol for p in related_markets_for("CL")}
    assert peers == {"RB", "HO", "NG"}


def test_refined_product_relationship_is_explicit_not_generic_same_asset_class():
    peers = {p.peer_root_symbol: p for p in related_markets_for("CL")}
    assert peers["RB"].relation_type is RelationType.REFINED_PRODUCT_RELATIONSHIP
    assert peers["HO"].relation_type is RelationType.REFINED_PRODUCT_RELATIONSHIP
    # NG is the same broad energy asset class but NOT refined from crude --
    # must not be conflated with the crack-spread relationship.
    assert peers["NG"].relation_type is RelationType.SAME_ASSET_CLASS


def test_rates_group_is_curve_neighbor_not_same_asset_class():
    peers = related_markets_for("ZN")
    assert all(p.relation_type is RelationType.CURVE_NEIGHBOR for p in peers)


def test_mapping_reason_is_never_blank():
    for root in ("NQ", "CL", "ZN", "6E", "ZC"):
        for peer in related_markets_for(root):
            assert peer.mapping_reason.strip()


def test_uncatalogued_group_root_returns_no_peers_never_a_guess():
    assert related_markets_for("BTC") == ()  # crypto is not yet grouped


def test_group_for_is_case_insensitive_and_symmetric_with_related_markets_for():
    assert group_for("nq") == group_for("NQ")
    assert group_for("nq") is not None


# ---------------------------------------------------------------------------
# alignment (Section 12 -- intersection only, never forward-filled)
# ---------------------------------------------------------------------------


def _bar(ts: datetime, close: float, *, high=None, low=None) -> dict:
    return {"ts_event": ts, "open": close, "high": high or close, "low": low or close, "close": close, "volume": 100.0}


def test_align_by_timestamp_keeps_only_the_real_intersection():
    t0, t1, t2 = datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 2, tzinfo=UTC), datetime(2026, 9, 3, tzinfo=UTC)
    series_a = [_bar(t0, 100.0), _bar(t1, 101.0), _bar(t2, 102.0)]
    series_b = [_bar(t0, 50.0), _bar(t2, 52.0)]  # missing t1 -- a real closed-market gap
    aligned_a, aligned_b = align_by_timestamp(series_a, series_b)
    assert [b["ts_event"] for b in aligned_a] == [t0, t2]
    assert [b["close"] for b in aligned_b] == [50.0, 52.0]


def test_align_by_timestamp_never_forward_fills():
    t0, t1 = datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 2, tzinfo=UTC)
    series_a = [_bar(t0, 100.0), _bar(t1, 101.0)]
    series_b = [_bar(t0, 50.0)]  # no t1 bar at all
    aligned_a, aligned_b = align_by_timestamp(series_a, series_b)
    assert len(aligned_a) == 1 and len(aligned_b) == 1  # never padded to length 2


# ---------------------------------------------------------------------------
# normalized / log returns
# ---------------------------------------------------------------------------


def test_normalized_return_formula():
    bars = [_bar(datetime(2026, 9, 1, tzinfo=UTC), 100.0), _bar(datetime(2026, 9, 2, tzinfo=UTC), 110.0)]
    rets = normalized_returns(bars)
    assert rets[0] == 0.0
    assert rets[1] == pytest.approx(0.1)


def test_log_returns_never_divides_by_a_nonpositive_close():
    bars = [_bar(datetime(2026, 9, 1, tzinfo=UTC), 0.0), _bar(datetime(2026, 9, 2, tzinfo=UTC), 100.0)]
    assert log_returns(bars) == []  # first close is 0 -- no crash, no fabricated return


def test_pearson_correlation_perfect_positive():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    assert pearson_correlation(x, y) == 1.0


def test_pearson_correlation_none_when_degenerate():
    assert pearson_correlation([1.0], [2.0]) is None
    assert pearson_correlation([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) is None  # zero variance


# ---------------------------------------------------------------------------
# build_relative_snapshot -- insufficient data vs real metrics
# ---------------------------------------------------------------------------


def _series(n: int, *, start: float, drift: float = 0.0) -> list[dict]:
    return [
        _bar(datetime(2026, 9, 1, i, tzinfo=UTC), start + drift * i)
        for i in range(n)
    ]


def test_build_relative_snapshot_insufficient_data_below_minimum():
    small_a = _series(MIN_COMMON_OBSERVATIONS - 1, start=100.0)
    small_b = _series(MIN_COMMON_OBSERVATIONS - 1, start=50.0)
    snap = build_relative_snapshot(
        "NQ", "ES", selected_bars=small_a, peer_bars=small_b,
        relation_type=RelationType.EQUITY_INDEX_PEER, mapping_reason="test",
    )
    assert snap.insufficient_data is True
    assert snap.common_observations == MIN_COMMON_OBSERVATIONS - 1
    assert snap.correlation is None
    assert snap.window_return_pct is None


def test_build_relative_snapshot_real_metrics_when_enough_common_bars():
    n = MIN_COMMON_OBSERVATIONS + 5
    sel = _series(n, start=100.0, drift=1.0)
    peer = _series(n, start=50.0, drift=0.5)
    snap = build_relative_snapshot(
        "NQ", "ES", selected_bars=sel, peer_bars=peer,
        relation_type=RelationType.EQUITY_INDEX_PEER, mapping_reason="test",
    )
    assert snap.insufficient_data is False
    assert snap.common_observations == n
    assert snap.window_return_pct is not None
    assert snap.peer_window_return_pct is not None
    assert snap.correlation == 1.0  # both series move in lockstep (linear drift)
    assert snap.relative_performance_pct is not None
    assert snap.price_ratio is not None


def test_build_relative_snapshot_never_forward_fills_across_a_gap():
    t = [datetime(2026, 9, 1, i, tzinfo=UTC) for i in range(MIN_COMMON_OBSERVATIONS + 5)]
    sel = [_bar(ts, 100.0 + i) for i, ts in enumerate(t)]
    # Peer is missing every other bar -- a real, uneven closed-market gap.
    peer = [_bar(ts, 50.0 + i) for i, ts in enumerate(t) if i % 2 == 0]
    snap = build_relative_snapshot(
        "NQ", "ES", selected_bars=sel, peer_bars=peer,
        relation_type=RelationType.EQUITY_INDEX_PEER, mapping_reason="test",
    )
    assert snap.common_observations == len(peer)  # never padded up to len(sel)
