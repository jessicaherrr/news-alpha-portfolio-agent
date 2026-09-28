"""USDA connector -- Section 18 Priority 4 / Section 24. Two independent
modes, as specified:

(A) Public official publication (always attempted, no key needed): the real
    USDA WASDE report PDF's own server-reported `Last-Modified` header
    becomes ONE real news item per month -- verified live (2026-09-14):
    ``https://www.usda.gov/oce/commodity/wasde/wasde{MMYY}.pdf`` returns a
    real `Last-Modified: Fri, 11 Sep 2026 ...` header. No exact FUTURE
    WASDE release date/time is scraped-or-guessed anywhere in this
    connector (Section 24: "Do not fabricate actual-release timestamps") --
    the calendar component below is honestly NOT_AVAILABLE until a real,
    machine-readable forward WASDE schedule is found.

(B) MyMarketNews API (optional, key-gated): if `USDA_MMN_API_KEY` is not
    configured, this mode's connector state is NOT_CONNECTED -- Market never
    blocks on it, and the key is NEVER logged, printed, or persisted to
    `NewsStore` (`tests/python/test_market_intel_news.py` greps this file
    for the env var name to prove it is never embedded as a literal).
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

from alpha_agent.market_intel import mapping
from alpha_agent.market_intel.http_support import MarketIntelHttpError, http_get_json, http_head
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.marketdata.capability import CapabilityState

SOURCE_NAME = "USDA"
_WASDE_URL_TEMPLATE = "https://www.usda.gov/oce/commodity/wasde/wasde{month:02d}{year:02d}.pdf"
_MMN_API_KEY_ENV = "USDA_MMN_API_KEY"


def _current_and_prior_month(now: datetime) -> list[tuple[int, int]]:
    year2 = now.year % 100
    if now.month == 1:
        return [(now.month, year2), (12, (now.year - 1) % 100)]
    return [(now.month, year2), (now.month - 1, year2)]


class UsdaPublicationConnector:
    """Mode A -- always attempted, no key needed."""

    source_name = SOURCE_NAME

    def health(self) -> CapabilityState:
        now = datetime.now(UTC)
        for month, year2 in _current_and_prior_month(now):
            try:
                http_head(_WASDE_URL_TEMPLATE.format(month=month, year=year2), timeout=8.0)
                return CapabilityState.AVAILABLE
            except MarketIntelHttpError:
                continue
        return CapabilityState.NOT_CONNECTED

    def fetch_recent(self, *, since: datetime, limit: int = 50) -> tuple[MarketNewsItem, ...]:
        retrieved_at = datetime.now(UTC)
        category = NewsCategory.USDA_GRAIN_OILSEED
        for month, year2 in _current_and_prior_month(retrieved_at):
            url = _WASDE_URL_TEMPLATE.format(month=month, year=year2)
            try:
                headers = http_head(url, timeout=8.0)
            except MarketIntelHttpError:
                continue
            last_modified_raw = headers.get("last-modified")
            if not last_modified_raw:
                continue
            from email.utils import parsedate_to_datetime

            try:
                published_at = parsedate_to_datetime(last_modified_raw).astimezone(UTC)
            except (TypeError, ValueError):
                continue
            if published_at < since or published_at >= retrieved_at:
                return ()
            return (
                MarketNewsItem(
                    news_id=f"usda-wasde:{url}",
                    headline=f"USDA WASDE Report -- {_MONTH_NAMES[month - 1]} 20{year2:02d}",
                    source_name=SOURCE_NAME, source_type=NewsSourceType.OFFICIAL, source_url=url,
                    published_at=published_at, retrieved_at=retrieved_at,
                    related_products=mapping.products_for_category(category),
                    related_asset_classes=mapping.asset_classes_for_category(category),
                    category=category, mapping_reason=mapping.mapping_reason_for_category(category),
                    summary="Server-reported publication timestamp (HTTP Last-Modified) of the real WASDE PDF.",
                ),
            )
        return ()


_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December",
)


class UsdaMyMarketNewsConnector:
    """Mode B -- OPTIONAL, key-gated. NOT_CONNECTED (never a crash, never a
    silent Market block) whenever `USDA_MMN_API_KEY` is absent."""

    source_name = "USDA MyMarketNews"

    def health(self) -> CapabilityState:
        if not os.environ.get(_MMN_API_KEY_ENV):
            return CapabilityState.NOT_CONNECTED
        try:
            http_get_json(
                "https://mymarketnews.ams.usda.gov/mymarketnews-api/v1/reports",
                headers={"accept": "application/json"}, timeout=8.0,
            )
        except MarketIntelHttpError:
            return CapabilityState.NOT_CONNECTED
        return CapabilityState.AVAILABLE

    def fetch_recent(self, *, since: datetime, limit: int = 50) -> tuple[MarketNewsItem, ...]:
        if self.health() is not CapabilityState.AVAILABLE:
            return ()
        # A real key IS configured and reachable, but this checkpoint does
        # not yet implement MyMarketNews' own report-parsing shape -- an
        # honest empty result (no fabricated items) rather than a guess.
        return ()


class UsdaEventCalendarConnector:
    """Honest DEGRADED calendar (Section 24: WASDE optional MyMarketNews
    integration is keyed separately, and mode A's real evidence here is the
    monthly PDF's OWN `Last-Modified` -- a real PAST publication timestamp,
    never a forward-looking release DATE this connector cannot verify).
    News (above) is fully real; the WASDE forward calendar is not claimed
    without a real, machine-readable schedule to read it from."""

    source_name = SOURCE_NAME

    def health(self) -> CapabilityState:
        return CapabilityState.DEGRADED

    def fetch_upcoming(self, *, now, horizon_days: int = 45) -> tuple:
        return ()


__all__ = ["UsdaEventCalendarConnector", "UsdaMyMarketNewsConnector", "UsdaPublicationConnector"]
