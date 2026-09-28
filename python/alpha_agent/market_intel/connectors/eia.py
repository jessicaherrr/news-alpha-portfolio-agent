"""EIA connectors -- Section 18 Priority 2 (news) / Section 24 (weekly
report calendars). Real official EIA feeds/pages, verified live
(2026-09-14):

    https://www.eia.gov/rss/todayinenergy.xml                          (RSS, 200)
    https://www.eia.gov/petroleum/supply/weekly/schedule.php             (HTML, 200)
    https://ir.eia.gov/ngs/schedule.html                                 (HTML, 200)
"""
from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

from alpha_agent.market_intel import importance, mapping
from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.market_intel.http_support import MarketIntelHttpError, http_get_text
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.marketdata.capability import CapabilityState

SOURCE_NAME = "U.S. Energy Information Administration"
_TODAY_IN_ENERGY_FEED = "https://www.eia.gov/rss/todayinenergy.xml"
_PETROLEUM_SCHEDULE_URL = "https://www.eia.gov/petroleum/supply/weekly/schedule.php"
_NATGAS_SCHEDULE_URL = "https://ir.eia.gov/ngs/schedule.html"

_ET = ZoneInfo("America/New_York")

#: Deterministic keyword classification (Section 21) -- petroleum vs natural
#: gas vs neither. Checked in this order (petroleum keywords first) since a
#: piece can legitimately mention both; the first real match wins.
_PETROLEUM_KEYWORDS = (
    "crude oil", "gasoline", "diesel", "petroleum", "refin", "wti", "brent", "distillate", "heating oil",
)
_NATGAS_KEYWORDS = ("natural gas", "henry hub", "lng", "gas storage")


def _classify(text: str) -> NewsCategory:
    lowered = text.lower()
    if any(kw in lowered for kw in _PETROLEUM_KEYWORDS):
        return NewsCategory.PETROLEUM
    if any(kw in lowered for kw in _NATGAS_KEYWORDS):
        return NewsCategory.NATURAL_GAS
    return NewsCategory.OTHER


class EiaNewsConnector:
    """Real EIA "Today in Energy" RSS -- no separate petroleum/natgas-only
    feed is published at a stable URL (verified live), so this connector
    classifies the single general feed's items deterministically instead."""

    source_name = SOURCE_NAME

    def health(self) -> CapabilityState:
        try:
            http_get_text(_TODAY_IN_ENERGY_FEED, timeout=8.0)
        except MarketIntelHttpError:
            return CapabilityState.NOT_CONNECTED
        return CapabilityState.AVAILABLE

    def fetch_recent(self, *, since: datetime, limit: int = 50) -> tuple[MarketNewsItem, ...]:
        retrieved_at = datetime.now(UTC)
        try:
            text = http_get_text(_TODAY_IN_ENERGY_FEED, timeout=10.0)
            root = ElementTree.fromstring(text)
        except (MarketIntelHttpError, ElementTree.ParseError):
            return ()

        items: list[MarketNewsItem] = []
        for entry in root.iter("item"):
            title = (entry.findtext("title") or "").strip()
            link = (entry.findtext("link") or "").strip()
            description = (entry.findtext("description") or "").strip()
            pub_raw = (entry.findtext("pubDate") or "").strip()
            if not title or not link or not pub_raw:
                continue
            try:
                published_at = parsedate_to_datetime(pub_raw)
            except (TypeError, ValueError):
                continue
            published_at = published_at.astimezone(UTC) if published_at.tzinfo else published_at.replace(tzinfo=UTC)
            if published_at < since or published_at >= retrieved_at:
                continue
            category = _classify(f"{title} {description}")
            items.append(
                MarketNewsItem(
                    news_id=f"eia:{link}", headline=title, source_name=SOURCE_NAME,
                    source_type=NewsSourceType.OFFICIAL, source_url=link,
                    published_at=published_at, retrieved_at=retrieved_at, summary=description or None,
                    related_products=mapping.products_for_category(category),
                    related_asset_classes=mapping.asset_classes_for_category(category),
                    category=category, mapping_reason=mapping.mapping_reason_for_category(category),
                )
            )
        items.sort(key=lambda i: i.published_at, reverse=True)
        return tuple(items[:limit])


# ---------------------------------------------------------------------------
# Weekly report calendars (Checkpoint F, Section 24) -- real holiday-shift
# exception tables parsed from EIA's own published release-schedule pages.
# ---------------------------------------------------------------------------

_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December",
)
_DATE_RE = re.compile(r"(" + "|".join(_MONTH_NAMES) + r")\s+(\d{1,2}),\s*(\d{4})")
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})\s*([ap])\.?m\.?", re.IGNORECASE)


def _strip_html(html: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", text)


def _parse_exceptions(clean_text: str) -> dict[tuple[int, int], tuple[object, int, int]]:
    """Real ISO (year, week) -> (alternate_date, hour_24, minute) overrides
    parsed from an EIA holiday-release-schedule table's own text. Both
    weekly report schedule pages list the ALTERNATE release date
    immediately followed by a release-time phrase -- this walks matched
    dates and grabs the nearest following time within a short window, never
    inventing one. Keyed by ISO WEEK (not the exact date) because the
    schedule only publishes the shifted date, not the ordinary date it
    replaces -- the whole calendar week the alternate date falls in is
    superseded, so the normal weekday inside that same week must never also
    be treated as a valid release."""
    exceptions: dict[tuple[int, int], tuple[object, int, int]] = {}
    for date_match in _DATE_RE.finditer(clean_text):
        month = _MONTH_NAMES.index(date_match.group(1)) + 1
        day, year = int(date_match.group(2)), int(date_match.group(3))
        window = clean_text[date_match.end():date_match.end() + 60]
        time_match = _TIME_RE.search(window)
        if not time_match:
            continue
        hour, minute, meridiem = int(time_match.group(1)), int(time_match.group(2)), time_match.group(3).lower()
        if meridiem == "p" and hour != 12:
            hour += 12
        alt_date = date(year, month, day)
        iso_year, iso_week, _ = alt_date.isocalendar()
        exceptions[(iso_year, iso_week)] = (alt_date, hour, minute)
    return exceptions


def _next_weekly_release(
    *, now: datetime, weekday: int, default_hour: int, default_minute: int,
    exceptions: dict[tuple[int, int], tuple[object, int, int]],
) -> datetime:
    """The next occurrence of `weekday` (Mon=0) at the default ET time,
    UNLESS that calendar WEEK carries a real, parsed holiday-shift exception
    -- in which case only its explicit alternate day/time counts for that
    week, never the ordinary weekday too (Section 24: "Handle holiday-
    shifted releases from the real schedule")."""
    now_utc = now.astimezone(UTC) if now.tzinfo else now.replace(tzinfo=UTC)
    cursor = now_utc.astimezone(_ET)
    for offset in range(14):
        candidate_date = (cursor + timedelta(days=offset)).date()
        iso_year, iso_week, _ = candidate_date.isocalendar()
        exception = exceptions.get((iso_year, iso_week))
        if exception is not None:
            alt_date, hour, minute = exception
            if candidate_date != alt_date:
                continue  # this week's ordinary weekday is superseded -- only alt_date counts
            candidate = datetime(alt_date.year, alt_date.month, alt_date.day, hour, minute, tzinfo=_ET)
        elif candidate_date.weekday() == weekday:
            candidate = datetime(
                candidate_date.year, candidate_date.month, candidate_date.day, default_hour, default_minute, tzinfo=_ET,
            )
        else:
            continue
        if candidate.astimezone(UTC) >= now_utc:
            return candidate.astimezone(UTC)
    # No exception and no plain weekday match found within 2 weeks -- should
    # not happen (every 14-day window contains the target weekday at least
    # once), but never fabricate a date if it somehow does.
    raise ValueError("could not determine the next weekly release date")


class EiaEventCalendarConnector:
    """Real EIA Weekly Petroleum Status Report (Wed 10:30 a.m. ET default)
    and Weekly Natural Gas Storage Report (Thu 10:30 a.m. ET default)
    calendars, both holiday-shift-adjusted from EIA's own published
    exception tables."""

    source_name = SOURCE_NAME

    def health(self) -> CapabilityState:
        try:
            http_get_text(_PETROLEUM_SCHEDULE_URL, timeout=8.0)
        except MarketIntelHttpError:
            return CapabilityState.NOT_CONNECTED
        return CapabilityState.AVAILABLE

    def fetch_upcoming(self, *, now: datetime, horizon_days: int = 45) -> tuple[ScheduledMarketEvent, ...]:
        retrieved_at = datetime.now(UTC)
        now_utc = now if now.tzinfo else now.replace(tzinfo=UTC)
        events: list[ScheduledMarketEvent] = []

        try:
            petroleum_html = _strip_html(http_get_text(_PETROLEUM_SCHEDULE_URL, timeout=12.0))
            petroleum_exceptions = _parse_exceptions(petroleum_html)
            petroleum_at = _next_weekly_release(
                now=now_utc, weekday=2, default_hour=10, default_minute=30, exceptions=petroleum_exceptions,
            )
            if (petroleum_at - now_utc).days <= horizon_days:
                event_importance, importance_rule = importance.importance_for_category("PETROLEUM")
                events.append(
                    ScheduledMarketEvent(
                        event_id=f"eia-wpsr:{petroleum_at.date().isoformat()}",
                        name="EIA Weekly Petroleum Status Report", source_name=SOURCE_NAME,
                        source_url=_PETROLEUM_SCHEDULE_URL, scheduled_at=petroleum_at, timezone="America/New_York",
                        category="PETROLEUM", importance=event_importance, importance_rule=importance_rule,
                        affected_products=mapping.products_for_category(NewsCategory.PETROLEUM),
                        mapping_reason=mapping.mapping_reason_for_category(NewsCategory.PETROLEUM),
                        retrieved_at=retrieved_at,
                    )
                )
        except (MarketIntelHttpError, ValueError):
            pass

        try:
            natgas_html = _strip_html(http_get_text(_NATGAS_SCHEDULE_URL, timeout=12.0))
            natgas_exceptions = _parse_exceptions(natgas_html)
            natgas_at = _next_weekly_release(
                now=now_utc, weekday=3, default_hour=10, default_minute=30, exceptions=natgas_exceptions,
            )
            if (natgas_at - now_utc).days <= horizon_days:
                event_importance, importance_rule = importance.importance_for_category("NATURAL_GAS")
                events.append(
                    ScheduledMarketEvent(
                        event_id=f"eia-ngs:{natgas_at.date().isoformat()}",
                        name="EIA Weekly Natural Gas Storage Report", source_name=SOURCE_NAME,
                        source_url=_NATGAS_SCHEDULE_URL, scheduled_at=natgas_at, timezone="America/New_York",
                        category="NATURAL_GAS", importance=event_importance, importance_rule=importance_rule,
                        affected_products=mapping.products_for_category(NewsCategory.NATURAL_GAS),
                        mapping_reason=mapping.mapping_reason_for_category(NewsCategory.NATURAL_GAS),
                        retrieved_at=retrieved_at,
                    )
                )
        except (MarketIntelHttpError, ValueError):
            pass

        events.sort(key=lambda e: e.scheduled_at)
        return tuple(events)


__all__ = ["EiaEventCalendarConnector", "EiaNewsConnector"]
