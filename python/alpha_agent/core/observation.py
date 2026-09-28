"""Phase 5 -- Event / Observation. ``MarketNewsItem`` and
``ScheduledMarketEvent`` (``alpha_agent.market_intel``) are already
asset-neutral in shape (``related_products``/``affected_products`` are plain
string tuples, not a Futures-typed field). ``Observation``
(``alpha_agent.translation.schemas``) additionally validates its
``root_symbol`` against the certified Futures universe -- that validator is
NOT loosened here (Phase 5 is architecture-only, never a research-universe
change); it is re-exported exactly as-is, Futures-scoped validation and all.
"""
from __future__ import annotations

from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem
from alpha_agent.translation.schemas import Observation

__all__ = ["MarketNewsItem", "Observation", "ScheduledMarketEvent"]
