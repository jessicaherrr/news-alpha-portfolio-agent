"""``MarketNewsConnector`` -- Section 17's typed connector interface, plus
the parallel ``EventCalendarConnector`` for Checkpoint F. Provider-specific
parsing lives in each connector module, never in a Streamlit page.
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem
from alpha_agent.marketdata.capability import CapabilityState


@runtime_checkable
class MarketNewsConnector(Protocol):
    source_name: str

    def health(self) -> CapabilityState:
        """Never assumed -- a real, cheap connectivity probe. AVAILABLE only
        when the source was actually reached this call/cache window."""
        ...

    def fetch_recent(self, *, since: datetime, limit: int = 50) -> tuple[MarketNewsItem, ...]:
        """Real items published at/after `since`, newest first, bounded to
        `limit`. Never raises for an ordinary connectivity failure -- returns
        `()` and lets `health()` explain why."""
        ...


@runtime_checkable
class EventCalendarConnector(Protocol):
    source_name: str

    def health(self) -> CapabilityState:
        ...

    def fetch_upcoming(self, *, now: datetime, horizon_days: int = 45) -> tuple[ScheduledMarketEvent, ...]:
        """Real scheduled events at/after `now`, within `horizon_days`."""
        ...


__all__ = ["EventCalendarConnector", "MarketNewsConnector"]
