"""``CommercialMarketNewsConnector`` -- Section 19. Interface-compatible with
`alpha_agent.market_intel.connectors.base.MarketNewsConnector` so a future
licensed/supported broad breaking-news provider can be dropped in without a
NewsStore or UI change. NOT implemented, NOT scraped, and NEVER enabled by
this checkpoint -- no commercial or unsupported broad news source is bought,
hard-coded, or scraped here (Section 19: "Do not scrape Reuters, Bloomberg,
WSJ, CNBC, etc. without a proper licensed or supported connector.").
"""
from __future__ import annotations

from datetime import datetime

from alpha_agent.market_intel.news_schemas import MarketNewsItem
from alpha_agent.marketdata.capability import CapabilityState

SOURCE_NAME = "Commercial Market News (not configured)"


class CommercialMarketNewsConnector:
    """Always DISABLED/NOT_CONFIGURED this checkpoint -- `fetch_recent`
    always returns `()`, never a fabricated or scraped item."""

    source_name = SOURCE_NAME

    def health(self) -> CapabilityState:
        return CapabilityState.DISABLED

    def fetch_recent(self, *, since: datetime, limit: int = 50) -> tuple[MarketNewsItem, ...]:
        return ()


__all__ = ["CommercialMarketNewsConnector"]
