"""Checkpoint G, Section 28-29 -- pure marker-range filters. No Streamlit.
Streamlit-level integration (checkbox toggles, rendered observed-reaction
metrics) is covered in `test_market_product_detail.py`.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from alpha_agent.market_intel.event_schemas import EventImportance, ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.ui import market_reaction

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)


def _news(published_at) -> MarketNewsItem:
    return MarketNewsItem(
        news_id="x", headline="h", source_name="s", source_type=NewsSourceType.OFFICIAL, source_url="https://x.gov",
        published_at=published_at, retrieved_at=NOW, category=NewsCategory.OTHER, mapping_reason="r",
    )


def _event(scheduled_at, actual_release_at=None) -> ScheduledMarketEvent:
    return ScheduledMarketEvent(
        event_id="e", name="n", source_name="s", source_url="https://x.gov", scheduled_at=scheduled_at,
        timezone="America/New_York", category="OTHER", importance=EventImportance.LOW, importance_rule="r",
        mapping_reason="m", retrieved_at=NOW, actual_release_at=actual_release_at,
    )


def test_news_markers_only_include_items_within_the_range():
    inside = _news(NOW - timedelta(hours=1))
    outside = _news(NOW - timedelta(days=10))
    markers = market_reaction.news_markers_for_range((inside, outside), start=NOW - timedelta(hours=2), end=NOW)
    assert len(markers) == 1
    assert markers[0][0] == inside.published_at


def test_news_marker_hover_text_carries_source_and_headline():
    item = _news(NOW - timedelta(hours=1))
    markers = market_reaction.news_markers_for_range((item,), start=NOW - timedelta(hours=2), end=NOW)
    assert "s" in markers[0][1] and "h" in markers[0][1]


def test_event_markers_prefer_actual_release_at_when_known():
    scheduled = NOW - timedelta(hours=5)
    actual = NOW - timedelta(hours=1)
    event = _event(scheduled, actual_release_at=actual)
    markers = market_reaction.event_markers_for_range((event,), start=NOW - timedelta(hours=2), end=NOW)
    assert len(markers) == 1
    assert markers[0][0] == actual  # actual_release_at wins over scheduled_at


def test_event_markers_fall_back_to_scheduled_at_when_no_actual_release():
    event = _event(NOW - timedelta(hours=1))
    markers = market_reaction.event_markers_for_range((event,), start=NOW - timedelta(hours=2), end=NOW)
    assert markers[0][0] == event.scheduled_at


def test_event_markers_exclude_events_outside_the_range():
    event = _event(NOW - timedelta(days=30))
    markers = market_reaction.event_markers_for_range((event,), start=NOW - timedelta(hours=2), end=NOW)
    assert markers == []
