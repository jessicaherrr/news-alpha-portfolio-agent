"""BLS connector -- Section 18 Priority 3 (news) / Section 24 (release
calendar). Real official per-release BLS Atom feeds, verified live
(2026-09-14): https://www.bls.gov/feed/{cpi,ppi,empsit,eci,jolts}.rss all
return real Atom entries with real headline/link/published timestamps.
Note: BLS's WAF rejects a default HTTP client User-Agent as bot traffic
(verified live -- a plain `curl` request 403s) but accepts a real,
identifying one (`alpha_agent.market_intel.http_support`'s own UA) --
never spoofed as a browser.

The BLS release CALENDAR (Section 24) uses the SAME connector: each Atom
feed's `<published>` history already tells us this release's normal cadence
(monthly), and the item published MOST RECENTLY tells us this cycle already
ran -- so "next" is not independently observable from this feed alone
without guessing an exact future date. Rather than assume a fixed day-of-
month (explicitly forbidden -- Section 24), this connector's calendar
component stays a documented DEGRADED/NOT_AVAILABLE outcome unless BLS
publishes a real, machine-readable forward schedule this session can reach;
see the module docstring in `alpha_agent.market_intel.connectors.usda` for
the same honest-degradation posture applied to WASDE's own calendar.
"""
from __future__ import annotations

from datetime import UTC, datetime
from xml.etree import ElementTree

from alpha_agent.market_intel import mapping
from alpha_agent.market_intel.http_support import MarketIntelHttpError, http_get_text
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.marketdata.capability import CapabilityState

SOURCE_NAME = "U.S. Bureau of Labor Statistics"

#: Real BLS Atom feed slugs -> category (Section 18: CPI, PPI, Employment
#: Situation, JOLTS; ECI included as "productivity where appropriate").
_FEEDS: dict[str, NewsCategory] = {
    "cpi.rss": NewsCategory.CPI_PPI_EMPLOYMENT,
    "ppi.rss": NewsCategory.CPI_PPI_EMPLOYMENT,
    "empsit.rss": NewsCategory.CPI_PPI_EMPLOYMENT,
    "eci.rss": NewsCategory.CPI_PPI_EMPLOYMENT,
    "jolts.rss": NewsCategory.CPI_PPI_EMPLOYMENT,
}

#: BLS's Atom feed uses the default (unprefixed) Atom namespace.
_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


class BlsNewsConnector:
    source_name = SOURCE_NAME

    def health(self) -> CapabilityState:
        try:
            http_get_text("https://www.bls.gov/feed/cpi.rss", timeout=8.0)
        except MarketIntelHttpError:
            return CapabilityState.NOT_CONNECTED
        return CapabilityState.AVAILABLE

    def fetch_recent(self, *, since: datetime, limit: int = 50) -> tuple[MarketNewsItem, ...]:
        retrieved_at = datetime.now(UTC)
        items: list[MarketNewsItem] = []
        for slug, category in _FEEDS.items():
            try:
                text = http_get_text(f"https://www.bls.gov/feed/{slug}", timeout=10.0)
                root = ElementTree.fromstring(text)
            except (MarketIntelHttpError, ElementTree.ParseError):
                continue
            for entry in root.findall("a:entry", _ATOM_NS):
                title = (entry.findtext("a:title", namespaces=_ATOM_NS) or "").strip()
                link_el = entry.find("a:link", _ATOM_NS)
                link = link_el.get("href", "").strip() if link_el is not None else ""
                published_raw = (entry.findtext("a:published", namespaces=_ATOM_NS) or "").strip()
                content = (entry.findtext("a:content", namespaces=_ATOM_NS) or "").strip()
                if not title or not link or not published_raw:
                    continue
                try:
                    published_at = datetime.fromisoformat(published_raw).astimezone(UTC)
                except ValueError:
                    continue
                if published_at < since or published_at >= retrieved_at:
                    continue
                items.append(
                    MarketNewsItem(
                        news_id=f"bls:{link}", headline=title, source_name=SOURCE_NAME,
                        source_type=NewsSourceType.OFFICIAL, source_url=link,
                        published_at=published_at, retrieved_at=retrieved_at, summary=content or None,
                        related_products=mapping.products_for_category(category),
                        related_asset_classes=mapping.asset_classes_for_category(category),
                        category=category, mapping_reason=mapping.mapping_reason_for_category(category),
                    )
                )
        items.sort(key=lambda i: i.published_at, reverse=True)
        return tuple(items[:limit])


class BlsEventCalendarConnector:
    """Honest DEGRADED calendar -- see module docstring. News (above) is
    fully real and live; the forward release CALENDAR is not claimed
    without a real, machine-readable schedule to read it from."""

    source_name = SOURCE_NAME

    def health(self) -> CapabilityState:
        return CapabilityState.DEGRADED

    def fetch_upcoming(self, *, now, horizon_days: int = 45) -> tuple:
        return ()


__all__ = ["BlsEventCalendarConnector", "BlsNewsConnector"]
