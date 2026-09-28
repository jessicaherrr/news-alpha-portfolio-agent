"""Conservative MARKET NEWS deduplication -- Section 20. Two items are the
SAME story only when their canonical URL matches, or (as a fallback) their
normalized headline AND source both match AND their published times are
close together. Sharing a keyword is never enough -- Section 20: "Do not
merge materially different stories because titles share keywords."
"""
from __future__ import annotations

import re
from datetime import timedelta

from alpha_agent.market_intel.news_schemas import MarketNewsItem

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")

#: How close two published timestamps must be, once headline+source already
#: match, to be treated as the same real-world story rather than a
#: coincidentally-worded follow-up.
DEFAULT_PROXIMITY = timedelta(hours=2)


def normalize_headline(headline: str) -> str:
    return _NORMALIZE_RE.sub(" ", headline.lower()).strip()


def storage_key(item: MarketNewsItem) -> str:
    """The primary dedup key for `NewsStore` inserts -- canonical URL first
    (Section 20), since every connector in this package supplies a real
    per-article `source_url`."""
    return f"url:{item.source_url.strip().lower()}"


def is_probable_duplicate(a: MarketNewsItem, b: MarketNewsItem, *, proximity: timedelta = DEFAULT_PROXIMITY) -> bool:
    """Fallback check for two DIFFERENT URLs that are still, in practice,
    the same story (e.g. a syndicated re-publish) -- REQUIRES same source
    AND matching normalized headline AND published within `proximity`; a
    shared keyword alone never qualifies."""
    if a.source_url.strip().lower() == b.source_url.strip().lower():
        return True
    if a.source_name != b.source_name:
        return False
    if normalize_headline(a.headline) != normalize_headline(b.headline):
        return False
    return abs(a.published_at - b.published_at) <= proximity


__all__ = ["DEFAULT_PROXIMITY", "is_probable_duplicate", "normalize_headline", "storage_key"]
