"""Checkpoint G, Section 29 -- deterministic observed-reaction computation.
Pure, no network.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from alpha_agent.market_intel.reaction import compute_observed_reaction


def _bar(ts, close, *, volume=100.0):
    return {"ts_event": ts, "open": close, "high": close, "low": close, "close": close, "volume": volume}


REF = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)


def test_no_bars_before_reference_is_honestly_insufficient():
    bars = [_bar(REF + timedelta(minutes=5), 100.0)]
    result = compute_observed_reaction(bars, REF)
    assert result.baseline_price is None
    assert result.return_5m_pct is None
    assert "before" in result.detail.lower()


def test_no_bars_after_reference_is_honestly_insufficient():
    bars = [_bar(REF - timedelta(minutes=5), 100.0)]
    result = compute_observed_reaction(bars, REF)
    assert result.baseline_price == 100.0
    assert result.return_5m_pct is None
    assert "after" in result.detail.lower()


def test_reaction_uses_only_bars_strictly_after_reference_no_lookahead():
    bars = [
        _bar(REF - timedelta(minutes=1), 100.0),
        _bar(REF, 999.0),  # AT the reference time -- must be treated as "before", never "after"
        _bar(REF + timedelta(minutes=1), 101.0),
    ]
    result = compute_observed_reaction(bars, REF)
    assert result.baseline_price == 999.0  # the bar AT the reference is baseline, not future-peeked
    assert result.n_bars_after == 1


def test_return_5m_computed_from_real_1m_bars():
    bars = [_bar(REF, 100.0)] + [_bar(REF + timedelta(minutes=i), 100.0 + i) for i in range(1, 10)]
    result = compute_observed_reaction(bars, REF)
    # 5 minutes after reference -> close 105.0 -> +5% from baseline 100.0
    assert result.return_5m_pct is not None
    assert abs(result.return_5m_pct - 5.0) < 0.01


def test_return_1h_none_when_bars_dont_reach_that_far():
    bars = [_bar(REF, 100.0), _bar(REF + timedelta(minutes=5), 101.0)]
    result = compute_observed_reaction(bars, REF)
    assert result.return_5m_pct is not None
    assert result.return_1h_pct is None


def test_volume_change_reflects_real_before_after_averages():
    before = [_bar(REF - timedelta(minutes=i), 100.0, volume=10.0) for i in range(5, 0, -1)]
    after = [_bar(REF + timedelta(minutes=i), 100.0, volume=50.0) for i in range(1, 6)]
    result = compute_observed_reaction(before + after, REF)
    assert result.volume_change_pct is not None
    assert result.volume_change_pct > 0  # volume rose after


def test_unsorted_input_bars_still_work():
    bars = [_bar(REF + timedelta(minutes=1), 101.0), _bar(REF - timedelta(minutes=1), 100.0)]
    result = compute_observed_reaction(bars, REF)
    assert result.baseline_price == 100.0


def test_never_a_causal_claim_in_the_schema():
    """Structural: the schema has no field even shaped like a causal claim."""
    from alpha_agent.market_intel.reaction import ObservedReaction

    forbidden = {"caused_by", "causal", "attribution"}
    assert not (set(ObservedReaction.model_fields) & forbidden)
