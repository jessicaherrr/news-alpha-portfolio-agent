"""Market Intelligence Data Completion Pass, Checkpoint G -- Section 28
(chart news/event markers) + Section 29 (observed market reaction). Pure
marker-building functions are separated from the Streamlit rendering entry
points for the same testability reason as every other `market_*` UI module
in this package.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem
from alpha_agent.market_intel.reaction import compute_observed_reaction
from alpha_agent.ui import charts_market, components, market_home, market_intel_context, palette

__all__ = [
    "event_markers_for_range",
    "news_markers_for_range",
    "render_chart_with_markers",
    "render_observed_reaction",
]


def news_markers_for_range(
    items: tuple[MarketNewsItem, ...], *, start: datetime, end: datetime,
) -> list[tuple[datetime, str]]:
    """Section 28: markers use REAL `published_at` only, filtered to the
    chart's own visible time range -- never placed at an inferred time."""
    return [
        (i.published_at, f"{i.published_at:%Y-%m-%d %H:%M} UTC · {i.source_name}<br>{i.headline}")
        for i in items if start <= i.published_at <= end
    ]


def event_markers_for_range(
    events: tuple[ScheduledMarketEvent, ...], *, start: datetime, end: datetime,
) -> list[tuple[datetime, str]]:
    """Section 28: uses `actual_release_at` when a real one is known, else
    `scheduled_at` -- never a fabricated time for either."""
    out = []
    for e in events:
        ts = e.actual_release_at or e.scheduled_at
        if start <= ts <= end:
            out.append((ts, f"{ts:%Y-%m-%d %H:%M} UTC · {e.source_name}<br>{e.name}"))
    return out


def render_chart_with_markers(root: str, *, key_prefix: str = "market-overlay") -> None:
    with components.card(f"{key_prefix}-card"):
        st.markdown('<div class="aa-gate-title">PRICE CHART -- NEWS / EVENT MARKERS</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        show_news = c1.checkbox("Show News", key=f"{key_prefix}-show-news")
        show_events = c2.checkbox("Show Scheduled Events", key=f"{key_prefix}-show-events")

        result, _ = market_home.get_ohlcv_cached(root, timeframe="1h", lookback_bars=96)
        if not result or not result.fetched or not result.bars:
            components.empty_state("Chart Markers", "No real bars available.", key=f"{key_prefix}-na")
            return

        bars: list[dict[str, Any]] = [b.model_dump(mode="python") for b in result.bars]
        fig = charts_market.market_price_chart(bars, height=320)
        start, end = bars[0]["ts_event"], bars[-1]["ts_event"]

        if show_news:
            items = market_intel_context.recent_news(related_product=root, window_hours=24 * 45, limit=300)
            markers = news_markers_for_range(items, start=start, end=end)
            charts_market.add_timestamp_markers(fig, bars, markers, color=palette.AMBER, symbol="diamond", name="News")
            if not markers:
                st.caption("No real news item falls within the shown chart window.")
        if show_events:
            events = market_intel_context.upcoming_events()
            markers = event_markers_for_range(events, start=start, end=end)
            charts_market.add_timestamp_markers(fig, bars, markers, color=palette.BLUE, symbol="star", name="Events")
            if not markers:
                st.caption(
                    "No real scheduled event falls within the shown chart window (the event calendar tracks "
                    "upcoming events, so this is most useful once the chart's own window reaches into the near future)."
                )

        components.plotly_chart(fig, key=f"{key_prefix}-fig")
        st.caption("Markers use real published/scheduled timestamps only -- never inferred or estimated.")


def render_observed_reaction(root: str, *, key_prefix: str = "market-reaction") -> None:
    with components.card(f"{key_prefix}-card"):
        st.markdown('<div class="aa-gate-title">OBSERVED MARKET REACTION</div>', unsafe_allow_html=True)
        items = market_intel_context.recent_news(related_product=root, window_hours=48, limit=10)
        if not items:
            components.empty_state(
                "Observed Market Reaction", f"No real news item for {root} in the last 48 hours.",
                key=f"{key_prefix}-na",
            )
            return

        most_recent = items[0]  # newest-first
        result, _ = market_home.get_ohlcv_cached(root, timeframe="5m", lookback_bars=96)
        if not result or not result.fetched or not result.bars:
            components.empty_state(
                "Observed Market Reaction", "No real fine-grained bars available to measure a reaction.",
                key=f"{key_prefix}-bars-na",
            )
            return

        bars = [b.model_dump(mode="python") for b in result.bars]
        reaction = compute_observed_reaction(bars, most_recent.published_at)
        st.caption(
            f"Most recent real news for {root}: \"{most_recent.headline}\" ({most_recent.source_name}, "
            f"{most_recent.published_at:%Y-%m-%d %H:%M} UTC)"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Return 5m", f"{reaction.return_5m_pct:+.2f}%" if reaction.return_5m_pct is not None else "N/A")
        c2.metric("Return 30m", f"{reaction.return_30m_pct:+.2f}%" if reaction.return_30m_pct is not None else "N/A")
        c3.metric("Return 1h", f"{reaction.return_1h_pct:+.2f}%" if reaction.return_1h_pct is not None else "N/A")
        if reaction.volume_change_pct is not None:
            st.caption(f"Volume change: {reaction.volume_change_pct:+.1f}%")
        st.caption(f"{reaction.detail} OBSERVED only -- never a causal claim.")
