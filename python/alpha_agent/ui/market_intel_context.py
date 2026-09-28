"""The UI's ONLY boundary into `alpha_agent.market_intel` (mirrors
`alpha_agent.ui.databento_context`'s role for the Databento observation
provider). Every page that wants real market news goes through this module,
never a connector or `NewsStore` directly -- one place to decide refresh
cadence/caching, and one place the never-fabricate-a-value invariant is
enforced for this plane.

OBSERVATION-plane boundary, exactly like `databento_context`: nothing here
writes to the `ExperimentRegistry`, computes a feature, or touches
`experiment_identity`. `MarketNewsItem` and `ResearchSource` remain
different concepts (Section 15/31) -- this module never imports
`alpha_agent.knowledge`.
"""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

from alpha_agent.market_intel.connectors.bls import BlsEventCalendarConnector, BlsNewsConnector
from alpha_agent.market_intel.connectors.commercial import CommercialMarketNewsConnector
from alpha_agent.market_intel.connectors.eia import EiaEventCalendarConnector, EiaNewsConnector
from alpha_agent.market_intel.connectors.fed import FedEventCalendarConnector, FedNewsConnector
from alpha_agent.market_intel.connectors.usda import (
    UsdaEventCalendarConnector,
    UsdaMyMarketNewsConnector,
    UsdaPublicationConnector,
)
from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem
from alpha_agent.market_intel.store import EventStore, NewsStore
from alpha_agent.marketdata.capability import CapabilityState

#: Section 34: News gets a "reasonable short TTL" -- bounds how often this
#: process re-hits the 4 real official sources, independent of how often a
#: Streamlit rerun asks for data (a further UI-session cache layer would be
#: redundant on top of this one, since a Streamlit rerun and a fresh Python
#: process both hit the SAME in-process cache here).
REFRESH_TTL_SECONDS = 900.0

#: Section 2/32: the fixed news window every "recent" read DEFAULTS to, so a
#: displayed count is always attributable to one documented cadence (the
#: Scanner's own "News 24h" column).
DEFAULT_WINDOW_HOURS = 24

#: How far back each connector is actually asked to fetch on every refresh --
#: deliberately WIDER than DEFAULT_WINDOW_HOURS. A source does not
#: necessarily publish something in every rolling 24h window (verified live:
#: several real quiet 24h stretches with zero Fed/EIA/BLS/USDA releases), so
#: an ingestion window tied to the 24h DISPLAY default would leave the store
#: sparse or empty for long stretches even though real older items exist and
#: are fetchable. This only widens what gets STORED; `recent_news()`'s own
#: default read window is unchanged (Section 2: "last 24h... do not count
#: lifetime news").
INGESTION_LOOKBACK_HOURS = 24 * 45

#: Section 26: how far ahead the event calendar looks.
DEFAULT_EVENT_HORIZON_DAYS = 45

__all__ = [
    "DEFAULT_EVENT_HORIZON_DAYS",
    "DEFAULT_WINDOW_HOURS",
    "INGESTION_LOOKBACK_HOURS",
    "REFRESH_TTL_SECONDS",
    "cached_recent_news",
    "cached_upcoming_events",
    "connector_health",
    "event_connector_health",
    "force_refresh",
    "last_refresh_at",
    "news_count_for_product",
    "next_event_for_product",
    "next_high_impact_event_for_product",
    "recent_news",
    "upcoming_events",
]


@lru_cache(maxsize=1)
def _store() -> NewsStore:
    return NewsStore()


@lru_cache(maxsize=1)
def _event_store() -> EventStore:
    return EventStore()


@lru_cache(maxsize=1)
def _connectors() -> dict[str, Any]:
    return {
        "Federal Reserve": FedNewsConnector(),
        "EIA": EiaNewsConnector(),
        "BLS": BlsNewsConnector(),
        "USDA": UsdaPublicationConnector(),
        "USDA MyMarketNews": UsdaMyMarketNewsConnector(),
        "Commercial": CommercialMarketNewsConnector(),
    }


@lru_cache(maxsize=1)
def _event_connectors() -> dict[str, Any]:
    return {
        "Federal Reserve": FedEventCalendarConnector(),
        "EIA": EiaEventCalendarConnector(),
        "BLS": BlsEventCalendarConnector(),
        "USDA": UsdaEventCalendarConnector(),
    }


_last_refresh_mono: float | None = None
_last_refresh_wall: datetime | None = None
_last_health: dict[str, CapabilityState] = {}
_last_event_health: dict[str, CapabilityState] = {}


def _refresh_if_stale() -> None:
    global _last_refresh_mono, _last_refresh_wall, _last_health, _last_event_health
    now_mono = time.monotonic()
    if _last_refresh_mono is not None and (now_mono - _last_refresh_mono) <= REFRESH_TTL_SECONDS:
        return

    since = datetime.now(UTC) - timedelta(hours=INGESTION_LOOKBACK_HOURS)
    health: dict[str, CapabilityState] = {}
    for name, connector in _connectors().items():
        try:
            state = connector.health()
        except Exception:  # noqa: BLE001 -- a connector's own bug must never break the page
            state = CapabilityState.NOT_CONNECTED
        health[name] = state
        if state is not CapabilityState.AVAILABLE:
            continue
        try:
            items: tuple[MarketNewsItem, ...] = connector.fetch_recent(since=since, limit=50)
        except Exception:  # noqa: BLE001, S112 -- one connector's failure must never block the others
            continue
        for item in items:
            _store().add_item(item)
    _last_health = health

    now = datetime.now(UTC)
    event_health: dict[str, CapabilityState] = {}
    for name, connector in _event_connectors().items():
        try:
            state = connector.health()
        except Exception:  # noqa: BLE001
            state = CapabilityState.NOT_CONNECTED
        event_health[name] = state
        if state is not CapabilityState.AVAILABLE:
            continue
        try:
            events: tuple[ScheduledMarketEvent, ...] = connector.fetch_upcoming(
                now=now, horizon_days=DEFAULT_EVENT_HORIZON_DAYS,
            )
        except Exception:  # noqa: BLE001, S112 -- one connector's failure must never block the others
            continue
        for event in events:
            _event_store().upsert_event(event)
    _last_event_health = event_health

    _last_refresh_mono = now_mono
    _last_refresh_wall = datetime.now(UTC)


def force_refresh() -> None:
    """The explicit "Refresh" UI action -- never called automatically on a
    bare page render/rerun."""
    global _last_refresh_mono
    _last_refresh_mono = None
    _refresh_if_stale()


def connector_health() -> dict[str, CapabilityState]:
    """Per-source health, from the last real refresh attempt (never
    re-probed on every call -- see `REFRESH_TTL_SECONDS`)."""
    _refresh_if_stale()
    return dict(_last_health)


def recent_news(
    *, related_product: str | None = None, window_hours: int = DEFAULT_WINDOW_HOURS, limit: int = 100,
) -> tuple[MarketNewsItem, ...]:
    _refresh_if_stale()
    since = datetime.now(UTC) - timedelta(hours=window_hours)
    return tuple(_store().list_recent(since=since, limit=limit, related_product=related_product))


def news_count_for_product(root: str, *, window_hours: int = DEFAULT_WINDOW_HOURS) -> int:
    _refresh_if_stale()
    since = datetime.now(UTC) - timedelta(hours=window_hours)
    return _store().count_recent(since=since, related_product=root)


def last_refresh_at() -> datetime | None:
    """Wall-clock time of the last successful connector refresh in this
    process -- `None` before the very first refresh ever runs here (Agent
    Evidence Pack acceptance pass, section 12: staleness must be shown
    honestly, never hidden or silently backfilled)."""
    return _last_refresh_wall


def cached_recent_news(
    *, related_product: str | None = None, window_hours: int = DEFAULT_WINDOW_HOURS, limit: int = 100,
) -> tuple[MarketNewsItem, ...]:
    """Same read as `recent_news`, but NEVER triggers `_refresh_if_stale` --
    a conversational question must not silently call a real external news
    API just because the in-process TTL happened to expire (task spec
    sections 12/16: "NO AUTO NETWORK FROM CHAT"). Reads exactly what this
    process has already fetched via an explicit Market/Opportunity refresh;
    empty before the first such refresh in this process."""
    since = datetime.now(UTC) - timedelta(hours=window_hours)
    return tuple(_store().list_recent(since=since, limit=limit, related_product=related_product))


def cached_upcoming_events(*, horizon_days: int = DEFAULT_EVENT_HORIZON_DAYS) -> tuple[ScheduledMarketEvent, ...]:
    """Read-only counterpart to `upcoming_events` -- never triggers a refresh
    (see `cached_recent_news`)."""
    return tuple(_event_store().list_upcoming(now=datetime.now(UTC), horizon_days=horizon_days))


def event_connector_health() -> dict[str, CapabilityState]:
    _refresh_if_stale()
    return dict(_last_event_health)


def upcoming_events(*, horizon_days: int = DEFAULT_EVENT_HORIZON_DAYS) -> tuple[ScheduledMarketEvent, ...]:
    _refresh_if_stale()
    return tuple(_event_store().list_upcoming(now=datetime.now(UTC), horizon_days=horizon_days))


def next_event_for_product(root: str, *, horizon_days: int = DEFAULT_EVENT_HORIZON_DAYS) -> ScheduledMarketEvent | None:
    """The single soonest real upcoming event affecting `root` -- `None`
    when no connector currently supplies one (never a fabricated
    placeholder event)."""
    for event in upcoming_events(horizon_days=horizon_days):
        if root.upper() in event.affected_products:
            return event
    return None


def next_high_impact_event_for_product(
    root: str, *, horizon_days: int = DEFAULT_EVENT_HORIZON_DAYS,
) -> ScheduledMarketEvent | None:
    from alpha_agent.market_intel.event_schemas import EventImportance

    for event in upcoming_events(horizon_days=horizon_days):
        if root.upper() in event.affected_products and event.importance is EventImportance.HIGH:
            return event
    return None
