"""Pure filter/row tests for `alpha_agent.ui.market_news` (Checkpoint E) and
`alpha_agent.ui.market_events` (Checkpoint F) -- no Streamlit runtime.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from alpha_agent.market_intel.event_schemas import EventImportance, ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.ui import market_events, market_news

NOW = datetime.now(UTC)


def _news(*, category=NewsCategory.FOMC_POLICY, source_type=NewsSourceType.OFFICIAL, related_products=("ES",)) -> MarketNewsItem:
    return MarketNewsItem(
        news_id="x", headline="h", source_name="s", source_type=source_type, source_url="https://x.gov",
        published_at=NOW - timedelta(hours=1), retrieved_at=NOW, category=category, mapping_reason="r",
        related_products=related_products,
    )


def test_filter_all_returns_everything():
    items = (_news(), _news(category=NewsCategory.PETROLEUM))
    assert market_news.filter_news(items, filter_name="All") == items


def test_filter_macro_matches_fomc_and_cpi_only():
    macro = _news(category=NewsCategory.FOMC_POLICY)
    energy = _news(category=NewsCategory.PETROLEUM)
    result = market_news.filter_news((macro, energy), filter_name="Macro")
    assert result == (macro,)


def test_filter_product_specific_requires_a_root():
    item = _news(related_products=("CL",))
    assert market_news.filter_news((item,), filter_name="Product-specific", root=None) == ()
    assert market_news.filter_news((item,), filter_name="Product-specific", root="CL") == (item,)
    assert market_news.filter_news((item,), filter_name="Product-specific", root="NQ") == ()


def test_news_row_never_includes_a_sentiment_field():
    row = market_news.news_row(_news())
    assert "sentiment" not in {k.lower() for k in row}
    assert "bullish" not in str(row).lower() and "bearish" not in str(row).lower()


def test_news_row_shows_exact_published_timestamp_and_source():
    item = _news()
    row = market_news.news_row(item)
    assert row["Source"] == "s"
    assert item.published_at.strftime("%Y-%m-%d") in row["Published"]


def _event(*, importance=EventImportance.HIGH) -> ScheduledMarketEvent:
    return ScheduledMarketEvent(
        event_id="e", name="Test Event", source_name="s", source_url="https://x.gov",
        scheduled_at=NOW + timedelta(days=1), timezone="America/New_York", category="OTHER",
        importance=importance, importance_rule="r", mapping_reason="m", retrieved_at=NOW,
        affected_products=("CL",),
    )


def test_time_until_label_requires_timezone_aware_inputs():
    import pytest

    with pytest.raises(ValueError, match="aware"):
        market_events.time_until_label(datetime(2026, 9, 16))  # noqa: DTZ001 -- deliberately naive


def test_time_until_label_never_negative_for_the_past():
    past = NOW - timedelta(hours=1)
    assert market_events.time_until_label(past, now=NOW) == "past"


def test_event_row_shows_importance_and_affected_products():
    row = market_events.event_row(_event(), now=NOW)
    assert row["Importance"] == "HIGH"
    assert "CL" in row["Affected Products"]
