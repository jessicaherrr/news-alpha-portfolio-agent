"""Alpha Discovery live-research campaign, Checkpoint 7 -- the REAL community
research connector.

Uses the Hacker News Algolia Search API (`https://hn.algolia.com`) -- a
public, keyless, documented, third-party-access-first API HN/Y Combinator
itself provides specifically for this kind of programmatic search. This is
NOT scraping: it is the same public API surface HN's own search page is
built on (task spec: "Use public APIs, RSS, documented public pages... Do
not depend on brittle unrestricted web scraping").

Community content gets a LOWER default epistemic weight than academic or
practitioner sources (`SourceQualityTier.TIER_C` -- "community posts, forum
research, individual blogs", the tier's own existing definition) -- this
affects research PRIORITY only, never a scientific threshold (task spec
section 27).

A second, OPTIONAL path -- `ingest_url` -- lets a caller (a future "paste a
research URL" UI action) submit ONE specific public page directly. It is a
plain bounded GET plus a trivial regex-based tag strip (never a headless
browser, never JS execution, never a general-purpose HTML parser dependency)
-- the page's prose is DATA to read, exactly like every other connector's
retrieved text.
"""
from __future__ import annotations

import re

from alpha_agent.knowledge.connector_support import (
    ConnectorBudget,
    ConnectorHttpError,
    http_get_json,
    http_get_text,
    now_iso,
    provenance_hash,
    truncate,
)
from alpha_agent.knowledge.models import (
    IngestionResult,
    IngestionStatus,
    SourceQualityTier,
    SourceType,
    StrategyKnowledgeItem,
)
from alpha_agent.knowledge.topic_relevance import classify_relevant_mechanism

__all__ = ["CommunityResearchConnector", "community_health", "generate_community_queries"]

HN_ALGOLIA_API = "https://hn.algolia.com/api/v1/search"

#: HN story titles rarely name a specific MARKET, but often name a specific
#: MECHANISM plainly (verified live: "mean reversion trading", "algorithmic
#: trading strategy" surface real, classifiable hits; a market-qualified
#: phrase like "Nasdaq futures strategy" almost never does) -- so, unlike
#: GitHub/academic, community queries favor mechanism vocabulary over market
#: vocabulary, keeping them aligned with `connector_support.classify_mechanism`'s
#: own keyword list.
_GENERIC_QUERY_TERMS: tuple[str, ...] = (
    "mean reversion trading strategy", "momentum trading strategy", "trend following strategy",
    "breakout trading strategy", "algorithmic trading strategy futures",
)

_TAG_RE = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


def community_health(*, timeout: float = 10.0) -> IngestionStatus:
    try:
        http_get_json(HN_ALGOLIA_API, params={"query": "futures", "tags": "story"}, timeout=timeout)
    except ConnectorHttpError:
        return IngestionStatus.ERROR
    return IngestionStatus.CONNECTED


def generate_community_queries(*, market: str, budget: ConnectorBudget | None = None) -> tuple[str, ...]:
    budget = budget or ConnectorBudget()
    market_name = {
        "NQ": "Nasdaq futures", "ES": "S&P 500 futures", "CL": "crude oil futures",
        "GC": "gold futures", "ZN": "Treasury futures",
    }.get(market, f"{market} futures")
    # mechanism-vocabulary terms FIRST (verified live to actually surface
    # classifiable HN story titles); a market-qualified phrase almost never
    # does, so it is a low-priority addition, not the primary query set.
    queries = list(dict.fromkeys((*_GENERIC_QUERY_TERMS, f"{market_name} strategy")))
    return tuple(queries[: budget.max_queries])


def _strip_html(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    return _WHITESPACE_RE.sub(" ", text).strip()


class CommunityResearchConnector:
    """Implements `ExternalSourceAdapter`. Real, keyless, budget-bounded."""

    def __init__(self, *, budget: ConnectorBudget | None = None, timeout: float | None = None):
        self._budget = budget or ConnectorBudget()
        self._timeout = timeout if timeout is not None else self._budget.timeout_seconds

    @property
    def source_type(self) -> SourceType:
        return SourceType.COMMUNITY

    def ingest(self, *, query: str, markets: tuple[str, ...] = ()) -> IngestionResult:
        market = markets[0] if markets else ""
        queries = generate_community_queries(market=market or "futures", budget=self._budget) if not query else (query,)

        items: list[StrategyKnowledgeItem] = []
        seen_ids: set[str] = set()
        degraded_notes: list[str] = []

        for q in queries:
            if len(items) >= self._budget.max_documents_total:
                break
            try:
                body, _h = http_get_json(
                    HN_ALGOLIA_API, timeout=self._timeout,
                    params={"query": q, "tags": "story", "hitsPerPage": self._budget.max_search_results},
                )
            except ConnectorHttpError as exc:
                degraded_notes.append(f"HN search {q!r} failed: {exc}")
                continue
            for hit in body.get("hits", []) or []:
                object_id = hit.get("objectID")
                if not object_id or object_id in seen_ids:
                    continue
                seen_ids.add(object_id)
                item = self._normalize_hit(hit)
                if item is not None:
                    items.append(item)
                if len(items) >= self._budget.max_documents_total:
                    break

        if not queries:
            return IngestionResult(source_type=SourceType.COMMUNITY, status=IngestionStatus.NOT_CONNECTED, items=(), detail="No query generated.")
        if degraded_notes and not items:
            return IngestionResult(source_type=SourceType.COMMUNITY, status=IngestionStatus.ERROR, items=(), detail="; ".join(degraded_notes[:3]))
        status = IngestionStatus.CONNECTED if not degraded_notes else IngestionStatus.DEGRADED
        detail = f"{len(items)} community post(s) normalized across {len(queries)} quer{'y' if len(queries)==1 else 'ies'} (Hacker News)."
        return IngestionResult(source_type=SourceType.COMMUNITY, status=status, items=tuple(items), detail=detail)

    def _normalize_hit(self, hit: dict) -> StrategyKnowledgeItem | None:
        title = hit.get("title") or hit.get("story_title") or ""
        if not title:
            return None
        mechanism = classify_relevant_mechanism(title)
        if mechanism is None:
            return None
        object_id = hit["objectID"]
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
        provenance = provenance_hash("community1", {"provider": "HN", "object_id": object_id, "title": title})
        return StrategyKnowledgeItem(
            knowledge_id=provenance,
            source_type=SourceType.COMMUNITY,
            source_quality=SourceQualityTier.TIER_C,
            ingestion_status=IngestionStatus.CONNECTED,
            title=title,
            authors=(hit.get("author"),) if hit.get("author") else (),
            source_url=url,
            accessed_at=now_iso(),
            publication_date=hit.get("created_at"),
            asset_classes=("futures",),
            economic_mechanism=mechanism,
            entry_logic_summary=truncate(title, 300),
            reported_results=None,
            limitations=(
                "Community discussion/forum post (Hacker News) -- lower default epistemic weight than "
                "academic or practitioner sources; used for research PRIORITY only, never scientific evidence."
            ),
            provenance_hash=provenance,
        )

    def ingest_url(self, url: str) -> StrategyKnowledgeItem | None:
        """Task spec "user-supplied public strategy URLs" -- a bounded,
        read-only fetch of ONE page, TEXT only (a trivial tag strip, never a
        browser, never JS). Returns `None` when the page's content does not
        classify into a supported mechanism (never a guess)."""
        try:
            html = http_get_text(url, timeout=self._timeout)
        except ConnectorHttpError:
            return None
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = _strip_html(title_match.group(1)) if title_match else url
        text = _strip_html(html)
        mechanism = classify_relevant_mechanism(f"{title}\n{text}")
        if mechanism is None:
            return None
        provenance = provenance_hash("community1", {"provider": "USER_URL", "url": url})
        return StrategyKnowledgeItem(
            knowledge_id=provenance,
            source_type=SourceType.COMMUNITY,
            source_quality=SourceQualityTier.TIER_C,
            ingestion_status=IngestionStatus.CONNECTED,
            title=truncate(title, 200),
            source_url=url,
            accessed_at=now_iso(),
            asset_classes=("futures",),
            economic_mechanism=mechanism,
            entry_logic_summary=truncate(text, 500),
            implementation_notes=truncate(text, 1500),
            reported_results=None,
            limitations="User-supplied public URL; content read as text only, never executed or imported.",
            provenance_hash=provenance,
        )
