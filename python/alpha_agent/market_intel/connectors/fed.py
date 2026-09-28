"""FEDERAL RESERVE connectors -- Section 18 Priority 1 (news) / Section 24
(FOMC calendar). Real official Federal Reserve RSS feeds and the real FOMC
meeting calendar page -- verified live against the actual site (2026-09-14):

    https://www.federalreserve.gov/feeds/press_monetary.xml   (real RSS, 200)
    https://www.federalreserve.gov/feeds/speeches.xml          (real RSS, 200)
    https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (real HTML, 200)

No scraping of a broad/commercial breaking-news source -- Section 19.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

from alpha_agent.market_intel import importance, mapping
from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.market_intel.http_support import MarketIntelHttpError, http_get_text
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.marketdata.capability import CapabilityState

SOURCE_NAME = "Federal Reserve"
_MONETARY_FEED = "https://www.federalreserve.gov/feeds/press_monetary.xml"
_SPEECHES_FEED = "https://www.federalreserve.gov/feeds/speeches.xml"
_FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

_ET = ZoneInfo("America/New_York")

#: FOMC statements have been released at 2:00 PM ET on the final day of
#: every meeting for many years -- the Federal Reserve's own stated release
#: convention, not itself scraped from any one page (the real calendar page
#: gives the meeting DATE; this constant supplies the well-established
#: release TIME on that date, documented here rather than silently assumed
#: -- CLAUDE.md Section 24: "Do NOT assume ... without reading the current
#: official schedule" is about the DATE, which this module always reads
#: live; the intraday time is Fed operating procedure, not vendor data).
_FOMC_STATEMENT_HOUR_ET = 14


def _parse_rss_item(item: ElementTree.Element, *, retrieved_at: datetime) -> tuple[str, str, datetime | None]:
    title = (item.findtext("title") or "").strip()
    link = (item.findtext("link") or "").strip()
    pub_date_raw = (item.findtext("pubDate") or "").strip()
    published_at = None
    if pub_date_raw:
        try:
            published_at = parsedate_to_datetime(pub_date_raw)
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=UTC)
            else:
                published_at = published_at.astimezone(UTC)
        except (TypeError, ValueError):
            published_at = None
    return title, link, published_at


def _classify(title: str) -> NewsCategory:
    lowered = title.lower()
    if any(kw in lowered for kw in ("fomc", "federal open market committee", "monetary policy", "discount rate")):
        return NewsCategory.FOMC_POLICY
    return NewsCategory.OTHER


class FedNewsConnector:
    """Real Fed monetary-policy press releases + speeches RSS."""

    source_name = SOURCE_NAME

    def health(self) -> CapabilityState:
        try:
            http_get_text(_MONETARY_FEED, timeout=8.0)
        except MarketIntelHttpError:
            return CapabilityState.NOT_CONNECTED
        return CapabilityState.AVAILABLE

    def fetch_recent(self, *, since: datetime, limit: int = 50) -> tuple[MarketNewsItem, ...]:
        retrieved_at = datetime.now(UTC)
        items: list[MarketNewsItem] = []
        for feed_url in (_MONETARY_FEED, _SPEECHES_FEED):
            try:
                text = http_get_text(feed_url, timeout=10.0)
                root = ElementTree.fromstring(text)
            except (MarketIntelHttpError, ElementTree.ParseError):
                continue
            for entry in root.iter("item"):
                title, link, published_at = _parse_rss_item(entry, retrieved_at=retrieved_at)
                if not title or not link or published_at is None:
                    continue
                if published_at < since or published_at >= retrieved_at:
                    continue
                category = _classify(title)
                items.append(
                    MarketNewsItem(
                        news_id=f"fed:{link}", headline=title, source_name=SOURCE_NAME,
                        source_type=NewsSourceType.OFFICIAL, source_url=link,
                        published_at=published_at, retrieved_at=retrieved_at,
                        related_products=mapping.products_for_category(category),
                        related_asset_classes=mapping.asset_classes_for_category(category),
                        category=category, mapping_reason=mapping.mapping_reason_for_category(category),
                    )
                )
        items.sort(key=lambda i: i.published_at, reverse=True)
        return tuple(items[:limit])


# ---------------------------------------------------------------------------
# FOMC calendar (Checkpoint F, Section 24)
# ---------------------------------------------------------------------------

_MEETING_MONTH_RE = re.compile(
    r'fomc-meeting__month[^>]*>\s*<strong>([A-Za-z]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<',
    re.DOTALL,
)
_YEAR_PANEL_RE = re.compile(r'<a id="\d+">(\d{4}) FOMC Meetings</a>')


def _extract_year_panels(html: str) -> list[tuple[int, int]]:
    """Returns [(year, start_offset_in_html), ...] for every "<YYYY> FOMC
    Meetings" panel heading found, so each meeting row can be attributed to
    the correct calendar year (the real page renders one panel per year)."""
    return [(int(m.group(1)), m.end()) for m in _YEAR_PANEL_RE.finditer(html)]


def _meeting_end_day(date_range: str) -> int | None:
    """"27-28" -> 28; "17-18*" -> 18 (an asterisk marks a SEP/dot-plot
    meeting, not a different date format); a single-day meeting "14" -> 14."""
    cleaned = date_range.strip().rstrip("*")
    parts = cleaned.split("-")
    try:
        return int(parts[-1])
    except ValueError:
        return None


_MONTH_NUM = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}


class FedEventCalendarConnector:
    """Real FOMC meeting calendar -- parses the Federal Reserve's own
    published per-year calendar panels (verified live structure, 2026-09-14)."""

    source_name = SOURCE_NAME

    def health(self) -> CapabilityState:
        try:
            http_get_text(_FOMC_CALENDAR_URL, timeout=8.0)
        except MarketIntelHttpError:
            return CapabilityState.NOT_CONNECTED
        return CapabilityState.AVAILABLE

    def fetch_upcoming(self, *, now: datetime, horizon_days: int = 45) -> tuple[ScheduledMarketEvent, ...]:
        retrieved_at = datetime.now(UTC)
        try:
            html = http_get_text(_FOMC_CALENDAR_URL, timeout=12.0)
        except MarketIntelHttpError:
            return ()

        panels = _extract_year_panels(html)
        if not panels:
            return ()
        events: list[ScheduledMarketEvent] = []
        for i, (year, start) in enumerate(panels):
            end = panels[i + 1][1] if i + 1 < len(panels) else len(html)
            panel_html = html[start:end]
            for month_match in _MEETING_MONTH_RE.finditer(panel_html):
                month_name, date_range = month_match.group(1), month_match.group(2)
                month_num = _MONTH_NUM.get(month_name)
                day = _meeting_end_day(date_range)
                if month_num is None or day is None:
                    continue
                try:
                    local_dt = datetime(year, month_num, day, _FOMC_STATEMENT_HOUR_ET, tzinfo=_ET)
                except ValueError:
                    continue
                scheduled_at = local_dt.astimezone(UTC)
                horizon_cutoff = now.replace(tzinfo=UTC) if now.tzinfo is None else now
                if scheduled_at < horizon_cutoff or (scheduled_at - horizon_cutoff).days > horizon_days:
                    continue
                category = NewsCategory.FOMC_POLICY
                event_importance, importance_rule = importance.importance_for_category("FOMC_POLICY")
                events.append(
                    ScheduledMarketEvent(
                        event_id=f"fomc:{year}-{month_num:02d}-{day:02d}",
                        name="FOMC Policy Statement", source_name=SOURCE_NAME, source_url=_FOMC_CALENDAR_URL,
                        scheduled_at=scheduled_at, timezone="America/New_York",
                        category="FOMC_POLICY", importance=event_importance, importance_rule=importance_rule,
                        affected_products=mapping.products_for_category(category),
                        mapping_reason=mapping.mapping_reason_for_category(category), retrieved_at=retrieved_at,
                    )
                )
        events.sort(key=lambda e: e.scheduled_at)
        return tuple(events)


__all__ = ["FedEventCalendarConnector", "FedNewsConnector"]
