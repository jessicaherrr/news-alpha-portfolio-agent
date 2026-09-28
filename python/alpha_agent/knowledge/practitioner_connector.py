"""Alpha Discovery live-research campaign, Checkpoint 7 -- the REAL
practitioner / asset-management research connector.

There is no single reliable, keyless, public JSON/RSS API dedicated to
"institutional futures research" -- CME/AQR/Man-Group-style feeds turned out
either unreachable or not real RSS in this environment (verified during
development; see the campaign's final readiness report). What IS real and
reliable: Crossref and OpenAlex -- the SAME public bibliographic APIs
`academic_connector.py` already uses -- index a great deal of genuine
institutional research (central-bank working-paper series, BIS, IMF, NBER,
and similar) with a real, queryable publisher/venue field.

This connector reuses that infrastructure but ADDS a mandatory institution
allowlist filter (task spec: "Do not hard-code one institution as
authoritative" -- so the list is broad and documented, never a single name)
on the publisher/venue string, so PRACTITIONER never just re-labels ordinary
academic output -- only a result whose publisher/venue plausibly IS a
reputable institution's own research series is ever included. Community
sources with a lower bar belong to `community_connector.py`, never here.

PREVIOUSLY-KNOWN LIMITATION, FIXED (Release UX Part K, task spec section 43):
keyword-vote mechanism classification alone over a broad institutional corpus
could produce a topical false positive -- e.g. a real-estate paper titled
"...Mean Reversion versus the Usual Suspects" matches the MEAN_REVERSION
trading keyword on title text alone, though its actual subject is rent
dynamics, not a trading strategy. A bare mechanism-name match cannot fix
this by itself (the false-positive title genuinely contains the mechanism
name); `alpha_agent.knowledge.topic_relevance.classify_relevant_mechanism`
now also requires co-occurrence with a trading-ACTIVITY word (a real-estate
paper's title/venue text names no such activity), so it replaces
`classify_mechanism` as this module's classification call. Still a fixed,
auditable keyword rule, never an LLM judgment call; a human reviewing
"Research Inspiration" still sees the full title, and a reported result is
never treated as internal evidence either way (`reported_results` stays
`None` regardless).
"""
from __future__ import annotations

from alpha_agent.knowledge.connector_support import (
    CROSSREF_API,
    OPENALEX_API,
    POLITE_CONTACT,
    ConnectorBudget,
    ConnectorHttpError,
    dedup_key,
    http_get_json,
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
from alpha_agent.knowledge.provider_registry import name_fragments_allowlist
from alpha_agent.knowledge.topic_relevance import classify_relevant_mechanism

__all__ = ["INSTITUTION_ALLOWLIST", "PractitionerResearchConnector", "generate_practitioner_queries"]

#: Task spec "PRACTITIONER / ASSET-MANAGEMENT SOURCES" -- a broad, reviewed,
#: NON-exhaustive allowlist of reputable institution/publisher name
#: fragments (case-insensitive substring match against a paper's
#: publisher/container-title/host-organization string). No single entry is
#: ever treated as uniquely authoritative; a result matches ANY of these.
#: DERIVED from the curated `alpha_agent.knowledge.provider_registry`
#: (Release UX Part K, task spec section 42) -- one source of truth for the
#: allowlist, kept as a module-level tuple here so this connector's own
#: matching logic (`_matches_institution`) and its existing tests are
#: unchanged.
INSTITUTION_ALLOWLIST: tuple[str, ...] = name_fragments_allowlist()

#: Verified live: pairing an institution NAME with a mechanism term in the
#: query text itself (rather than relying on the allowlist filter alone to
#: narrow a generic search) meaningfully raises the real hit rate -- a
#: query for e.g. "trend following" alone returns mostly non-institutional
#: academic finance papers that the allowlist then filters down to zero.
_GENERIC_QUERY_TERMS: tuple[str, ...] = (
    "Federal Reserve futures mean reversion", "Federal Reserve commodity futures term structure",
    "IMF working paper carry trade futures", "central bank managed futures trend following",
)


def generate_practitioner_queries(*, market: str, budget: ConnectorBudget | None = None) -> tuple[str, ...]:
    budget = budget or ConnectorBudget()
    market_name = {
        "NQ": "equity index futures", "ES": "equity index futures", "CL": "commodity futures",
        "GC": "commodity futures", "ZN": "interest rate futures",
    }.get(market, "futures")
    specific = (f"Federal Reserve {market_name} risk premia",)
    queries = list(dict.fromkeys((*_GENERIC_QUERY_TERMS, *specific)))
    return tuple(queries[: budget.max_queries])


def _matches_institution(text: str) -> bool:
    lowered = f" {text.lower()} "
    return any(frag in lowered for frag in INSTITUTION_ALLOWLIST)


class PractitionerResearchConnector:
    """Implements `ExternalSourceAdapter`. Real, keyless, budget-bounded --
    Crossref primary (more reliable in practice), OpenAlex merged in and
    deduplicated, BOTH filtered to the institution allowlist."""

    def __init__(self, *, budget: ConnectorBudget | None = None, timeout: float | None = None):
        self._budget = budget or ConnectorBudget()
        self._timeout = timeout if timeout is not None else self._budget.timeout_seconds

    @property
    def source_type(self) -> SourceType:
        return SourceType.PRACTITIONER

    def ingest(self, *, query: str, markets: tuple[str, ...] = ()) -> IngestionResult:
        market = markets[0] if markets else ""
        queries = generate_practitioner_queries(market=market or "futures", budget=self._budget) if not query else (query,)

        items: list[StrategyKnowledgeItem] = []
        seen: set[str] = set()
        degraded_notes: list[str] = []
        any_success = False

        for q in queries:
            if len(items) >= self._budget.max_documents_total:
                break
            try:
                body, _h = http_get_json(
                    CROSSREF_API, timeout=self._timeout,
                    params={"query": q, "rows": self._budget.max_search_results},
                )
                any_success = True
            except ConnectorHttpError as exc:
                degraded_notes.append(f"Crossref search {q!r} failed: {exc}")
                continue
            for work in (body.get("message") or {}).get("items", []) or []:
                item = self._normalize_crossref(work)
                if item is None:
                    continue
                key = dedup_key(doi=work.get("DOI"), title=item.title)
                if key in seen:
                    continue
                seen.add(key)
                items.append(item)
                if len(items) >= self._budget.max_documents_total:
                    break

        for q in queries[:2]:
            if len(items) >= self._budget.max_documents_total:
                break
            try:
                body, _h = http_get_json(
                    OPENALEX_API, timeout=self._timeout,
                    params={"search": q, "per_page": self._budget.max_search_results, "mailto": POLITE_CONTACT},
                )
                any_success = True
            except ConnectorHttpError as exc:
                degraded_notes.append(f"OpenAlex search {q!r} failed: {exc}")
                continue
            for work in body.get("results", []) or []:
                item = self._normalize_openalex(work)
                if item is None:
                    continue
                doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
                key = dedup_key(doi=doi, title=item.title)
                if key in seen:
                    continue
                seen.add(key)
                items.append(item)
                if len(items) >= self._budget.max_documents_total:
                    break

        if not any_success:
            return IngestionResult(
                source_type=SourceType.PRACTITIONER, status=IngestionStatus.ERROR, items=(),
                detail="No provider reachable: " + "; ".join(degraded_notes[:3]),
            )
        status = IngestionStatus.CONNECTED if not degraded_notes else IngestionStatus.DEGRADED
        detail = (
            f"{len(items)} institution-allowlisted research item(s) across {len(queries)} "
            f"quer{'y' if len(queries)==1 else 'ies'}."
        )
        if degraded_notes:
            detail += " Some providers failed: " + "; ".join(degraded_notes[:3])
        return IngestionResult(source_type=SourceType.PRACTITIONER, status=status, items=tuple(items), detail=detail)

    def _normalize_crossref(self, work: dict) -> StrategyKnowledgeItem | None:
        titles = work.get("title") or []
        title = titles[0] if titles else ""
        if not title:
            return None
        publisher = work.get("publisher") or ""
        container = " ".join(work.get("container-title") or [])
        if not (_matches_institution(publisher) or _matches_institution(container)):
            return None
        mechanism = classify_relevant_mechanism(f"{title} {container}")
        if mechanism is None:
            return None
        doi = work.get("DOI")
        authors = tuple(f"{a.get('given', '')} {a.get('family', '')}".strip() for a in (work.get("author") or []))
        date_parts = ((work.get("issued") or {}).get("date-parts") or [[None]])[0]
        year = date_parts[0] if date_parts else None
        provenance = provenance_hash("practitioner1", {"provider": "CROSSREF", "doi": doi, "title": title})
        return StrategyKnowledgeItem(
            knowledge_id=provenance,
            source_type=SourceType.PRACTITIONER,
            source_quality=SourceQualityTier.TIER_A,
            ingestion_status=IngestionStatus.CONNECTED,
            title=title,
            authors=tuple(a for a in authors if a.strip()),
            source_url=work.get("URL"),
            accessed_at=now_iso(),
            publication_date=str(year) if year else None,
            asset_classes=("futures",),
            economic_mechanism=mechanism,
            entry_logic_summary=truncate(f"{title} ({publisher or container})", 300),
            reported_results=None,
            limitations=(
                f"Institutional research ({publisher or container}); methodology and any performance claim "
                "read as external context only, never internal evidence. Possible institutional/promotional "
                "context not independently audited here."
            ),
            provenance_hash=provenance,
        )

    def _normalize_openalex(self, work: dict) -> StrategyKnowledgeItem | None:
        title = work.get("display_name") or work.get("title") or ""
        if not title:
            return None
        host = (work.get("primary_location") or {}).get("source") or {}
        venue = host.get("display_name") or ""
        host_org = host.get("host_organization_name") or ""
        if not (_matches_institution(venue) or _matches_institution(host_org)):
            return None
        mechanism = classify_relevant_mechanism(f"{title} {venue}")
        if mechanism is None:
            return None
        year = work.get("publication_year")
        authors = tuple(a.get("author", {}).get("display_name", "") for a in (work.get("authorships") or []) if a.get("author"))
        provenance = provenance_hash("practitioner1", {"provider": "OPENALEX", "id": work.get("id"), "title": title})
        return StrategyKnowledgeItem(
            knowledge_id=provenance,
            source_type=SourceType.PRACTITIONER,
            source_quality=SourceQualityTier.TIER_A,
            ingestion_status=IngestionStatus.CONNECTED,
            title=title,
            authors=tuple(a for a in authors if a),
            source_url=work.get("id"),
            accessed_at=now_iso(),
            publication_date=str(year) if year else None,
            asset_classes=("futures",),
            economic_mechanism=mechanism,
            entry_logic_summary=truncate(f"{title} ({venue or host_org})", 300),
            reported_results=None,
            limitations=(
                f"Institutional research ({venue or host_org}); methodology and any performance claim read "
                "as external context only, never internal evidence."
            ),
            provenance_hash=provenance,
        )
