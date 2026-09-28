"""Market Intelligence Data Completion Pass, Checkpoint E -- Market News UI.
Pure row/filter functions are separated from the Streamlit rendering entry
point (mirrors `alpha_agent.ui.market_scanner`'s split); everything renders
through `alpha_agent.ui.market_intel_context`, the sole UI boundary into
`alpha_agent.market_intel` -- never a connector or `NewsStore` directly
(Section 17).

Never renders an AI sentiment score or a bullish/bearish label (Section 22)
-- there is no such field anywhere in `MarketNewsItem`, so there is nothing
here that even COULD render one.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory
from alpha_agent.marketdata.capability import CapabilityState
from alpha_agent.ui import components, market_intel_context

__all__ = [
    "FILTERS",
    "filter_news",
    "news_row",
    "render_news_section",
]

#: Section 22's filter pills.
FILTERS: tuple[str, ...] = ("All", "Official", "Macro", "Energy", "Agriculture", "Product-specific")

_MACRO_CATEGORIES = frozenset({NewsCategory.FOMC_POLICY, NewsCategory.CPI_PPI_EMPLOYMENT})
_ENERGY_CATEGORIES = frozenset({NewsCategory.PETROLEUM, NewsCategory.NATURAL_GAS})
_AGRICULTURE_CATEGORIES = frozenset({NewsCategory.USDA_GRAIN_OILSEED})


def filter_news(items: tuple[MarketNewsItem, ...], *, filter_name: str, root: str | None = None) -> tuple[MarketNewsItem, ...]:
    if filter_name == "All":
        return items
    if filter_name == "Official":
        from alpha_agent.market_intel.news_schemas import NewsSourceType

        return tuple(i for i in items if i.source_type is NewsSourceType.OFFICIAL)
    if filter_name == "Macro":
        return tuple(i for i in items if i.category in _MACRO_CATEGORIES)
    if filter_name == "Energy":
        return tuple(i for i in items if i.category in _ENERGY_CATEGORIES)
    if filter_name == "Agriculture":
        return tuple(i for i in items if i.category in _AGRICULTURE_CATEGORIES)
    if filter_name == "Product-specific":
        if not root:
            return ()
        return tuple(i for i in items if root.upper() in i.related_products)
    return items


def news_row(item: MarketNewsItem) -> dict[str, Any]:
    return {
        "Published": item.published_at.strftime("%Y-%m-%d %H:%M UTC"),
        "Headline": item.headline,
        "Source": item.source_name,
        "Category": item.category.value.replace("_", " "),
        "Related Products": ", ".join(item.related_products) if item.related_products else "—",
        "Link": item.source_url,
    }


def _connector_health_caption(health: dict[str, CapabilityState]) -> str:
    parts = [f"{name}: {state.value}" for name, state in sorted(health.items())]
    return " · ".join(parts)


def render_news_section(root: str, *, key_prefix: str = "market-news") -> None:
    """Product Detail -> News & Events -> NEWS. Never shows a numeric
    activity count before a successful query (Section 2) -- callers reading
    `market_intel_context.recent_news` always get a real, already-refreshed
    (subject to `REFRESH_TTL_SECONDS`) list, so this component itself never
    needs a separate NOT_LOADED state."""
    with components.card(f"{key_prefix}-latest"):
        st.markdown('<div class="aa-gate-title">LATEST NEWS</div>', unsafe_allow_html=True)
        health = market_intel_context.connector_health()
        st.caption(f"Official sources · {_connector_health_caption(health)}")

        chosen = st.radio(
            "Filter", FILTERS, horizontal=True, key=f"{key_prefix}-filter", label_visibility="collapsed",
        )
        items = market_intel_context.recent_news(window_hours=24 * 7, limit=200)
        visible = filter_news(items, filter_name=chosen, root=root)

        if not visible:
            reason = (
                "No connector is currently AVAILABLE." if not any(s is CapabilityState.AVAILABLE for s in health.values())
                else "No real items matched this filter in the last 7 days."
            )
            components.empty_state("News", reason, key=f"{key_prefix}-empty")
            return

        st.dataframe(
            [news_row(i) for i in visible], width="stretch", hide_index=True,
            height=min(420, 44 + 35 * len(visible)),
            column_config={"Link": st.column_config.LinkColumn("Source Link", display_text="Open")},
        )
        st.caption(
            f"{len(visible)} item(s) shown (last 7 days) -- exact published timestamps and sources above; "
            "never an AI sentiment score, never bullish/bearish."
        )
