"""Alpha Discovery live-research campaign, Checkpoint 6 -- the REAL academic
research connector.

Uses PUBLIC, free, keyless APIs only (task spec: "Use public/free APIs where
possible. Do not require the user to provide an academic API key unless
genuinely necessary."):

* OpenAlex (preferred -- rich metadata AND an abstract for most works, no
  key, generous rate limit) is the PRIMARY provider.
* Crossref (DOI-centric bibliographic metadata, no key) is a SECONDARY
  provider, merged in and deduplicated against OpenAlex by DOI/normalized
  title -- never counted as a second, independent research idea (task spec
  "ACADEMIC DEDUPLICATION": "the same paper found in OpenAlex + Crossref is
  ONE source document with multiple provider references").

Never bypasses a paywall: only the metadata + abstract this module's own
public API responses already contain is used. When a work carries no
abstract, the item is built from title/metadata alone and this is recorded
plainly (never fabricated ``abstract`` text).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from alpha_agent.knowledge.connector_support import (
    CROSSREF_API,
    OPENALEX_API,
    POLITE_CONTACT,
    ConnectorBudget,
    ConnectorHttpError,
    classify_mechanism,
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

__all__ = [
    "AcademicConnectorHealth",
    "AcademicProvider",
    "AcademicResearchConnector",
    "academic_health",
    "generate_academic_queries",
]

_GENERIC_QUERY_TERMS: tuple[str, ...] = (
    "time series momentum futures", "managed futures trend following",
    "futures term structure carry", "cross asset momentum futures",
)


class AcademicProvider(str, Enum):
    OPENALEX = "OPENALEX"
    CROSSREF = "CROSSREF"


class AcademicConnectorHealth(str, Enum):
    CONNECTED = "CONNECTED"
    RATE_LIMITED = "RATE_LIMITED"
    ERROR = "ERROR"


class AcademicHealthReport(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    health: AcademicConnectorHealth
    provider: AcademicProvider
    detail: str = ""


def academic_health(*, provider: AcademicProvider = AcademicProvider.OPENALEX, timeout: float = 10.0) -> AcademicHealthReport:
    """Real, read-only reachability probe. No credential is ever required."""
    url = OPENALEX_API if provider is AcademicProvider.OPENALEX else CROSSREF_API
    params = {"per_page": 1, "search": "futures"} if provider is AcademicProvider.OPENALEX else {"rows": 1, "query": "futures"}
    try:
        http_get_json(url, params=params, timeout=timeout)
    except ConnectorHttpError as exc:
        if exc.status_code == 429:
            return AcademicHealthReport(health=AcademicConnectorHealth.RATE_LIMITED, provider=provider, detail=str(exc))
        return AcademicHealthReport(health=AcademicConnectorHealth.ERROR, provider=provider, detail=str(exc))
    return AcademicHealthReport(health=AcademicConnectorHealth.CONNECTED, provider=provider, detail="Public API reachable.")


def generate_academic_queries(*, market: str, budget: ConnectorBudget | None = None) -> tuple[str, ...]:
    budget = budget or ConnectorBudget()
    market_name = {
        "NQ": "equity index futures", "ES": "equity index futures", "CL": "commodity futures",
        "GC": "commodity futures", "ZN": "Treasury futures",
    }.get(market, "futures")
    specific = (
        f"{market_name} momentum", f"{market_name} mean reversion", f"{market_name} opening range breakout",
        f"{market_name} overnight returns",
    )
    queries = list(dict.fromkeys((*specific, *_GENERIC_QUERY_TERMS)))
    return tuple(queries[: budget.max_queries])


def _reconstruct_openalex_abstract(inverted_index: dict | None) -> str:
    """OpenAlex returns the abstract as `{word: [positions...]}` (a licensing
    workaround, not a data-quality issue) -- reconstruct plain text
    deterministically."""
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def _normalize_openalex_work(work: dict) -> tuple[StrategyKnowledgeItem, str | None, str] | None:
    title = work.get("display_name") or work.get("title") or ""
    if not title:
        return None
    abstract = _reconstruct_openalex_abstract(work.get("abstract_inverted_index"))
    mechanism = classify_mechanism(f"{title}\n{abstract}")
    if mechanism is None:
        return None
    doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
    authors = tuple(
        a.get("author", {}).get("display_name", "") for a in (work.get("authorships") or []) if a.get("author")
    )
    year = work.get("publication_year")
    oa = work.get("open_access") or {}
    provenance = provenance_hash("academic1", {"provider": "OPENALEX", "doi": doi, "title": title})
    return StrategyKnowledgeItem(
        knowledge_id=provenance,
        source_type=SourceType.ACADEMIC,
        source_quality=SourceQualityTier.TIER_A,
        ingestion_status=IngestionStatus.CONNECTED,
        title=title,
        authors=tuple(a for a in authors if a),
        source_url=(oa.get("oa_url") if oa.get("is_oa") else work.get("id")),
        license="open-access" if oa.get("is_oa") else None,
        accessed_at=now_iso(),
        publication_date=str(year) if year else None,
        asset_classes=("futures",),
        economic_mechanism=mechanism,
        entry_logic_summary=truncate(abstract or "Abstract not available from provider.", 500),
        implementation_notes=truncate(abstract, 1500) if abstract else "",
        reported_results=None,  # a paper's own reported results NEVER become our evidence
        limitations=(
            "Published academic result; evidence coverage = "
            + ("abstract" if abstract else "metadata only, no public abstract")
            + ". Not automatically trusted for today's market -- reproduced/adapted independently."
        ),
        candidate_dsl_template=None,
        provenance_hash=provenance,
    ), doi, title


def _normalize_crossref_work(work: dict) -> tuple[StrategyKnowledgeItem, str | None, str] | None:
    titles = work.get("title") or []
    title = titles[0] if titles else ""
    if not title:
        return None
    mechanism = classify_mechanism(title)
    if mechanism is None:
        return None
    doi = work.get("DOI")
    authors = tuple(
        f"{a.get('given', '')} {a.get('family', '')}".strip() for a in (work.get("author") or [])
    )
    date_parts = ((work.get("issued") or {}).get("date-parts") or [[None]])[0]
    year = date_parts[0] if date_parts else None
    provenance = provenance_hash("academic1", {"provider": "CROSSREF", "doi": doi, "title": title})
    item = StrategyKnowledgeItem(
        knowledge_id=provenance,
        source_type=SourceType.ACADEMIC,
        source_quality=SourceQualityTier.TIER_A,
        ingestion_status=IngestionStatus.CONNECTED,
        title=title,
        authors=tuple(a for a in authors if a.strip()),
        source_url=work.get("URL"),
        accessed_at=now_iso(),
        publication_date=str(year) if year else None,
        asset_classes=("futures",),
        economic_mechanism=mechanism,
        entry_logic_summary=truncate(title, 300),
        reported_results=None,
        limitations="Crossref bibliographic metadata only (no abstract available from this provider).",
        provenance_hash=provenance,
    )
    return item, doi, title


class AcademicResearchConnector:
    """Implements `ExternalSourceAdapter`. Real, keyless, budget-bounded.
    OpenAlex primary; Crossref merged in and deduplicated by DOI/title."""

    def __init__(self, *, budget: ConnectorBudget | None = None, timeout: float | None = None):
        self._budget = budget or ConnectorBudget()
        self._timeout = timeout if timeout is not None else self._budget.timeout_seconds

    @property
    def source_type(self) -> SourceType:
        return SourceType.ACADEMIC

    def ingest(self, *, query: str, markets: tuple[str, ...] = ()) -> IngestionResult:
        market = markets[0] if markets else ""
        queries = generate_academic_queries(market=market or "futures", budget=self._budget) if not query else (query,)

        items: list[StrategyKnowledgeItem] = []
        seen: dict[str, int] = {}  # dedup key -> index into items (first-seen wins)
        degraded_notes: list[str] = []
        any_success = False

        for q in queries:
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
                body = {}
            for work in body.get("results", []) or []:
                normalized = _normalize_openalex_work(work)
                if normalized is None:
                    continue
                item, doi, title = normalized
                key = dedup_key(doi=doi, title=title)
                if key in seen:
                    continue
                seen[key] = len(items)
                items.append(item)
                if len(items) >= self._budget.max_documents_total:
                    break

        # Crossref pass -- merged in, deduplicated against what OpenAlex
        # already contributed (task spec "ACADEMIC DEDUPLICATION").
        for q in queries[:2]:  # a bounded secondary pass, not a second full sweep
            if len(items) >= self._budget.max_documents_total:
                break
            try:
                body, _h = http_get_json(
                    CROSSREF_API, timeout=self._timeout,
                    params={"query": q, "rows": min(5, self._budget.max_search_results)},
                )
                any_success = True
            except ConnectorHttpError as exc:
                degraded_notes.append(f"Crossref search {q!r} failed: {exc}")
                continue
            for work in (body.get("message") or {}).get("items", []) or []:
                normalized = _normalize_crossref_work(work)
                if normalized is None:
                    continue
                item, doi, title = normalized
                key = dedup_key(doi=doi, title=title)
                if key in seen:
                    continue  # already covered by OpenAlex -- one source document, not two
                seen[key] = len(items)
                items.append(item)
                if len(items) >= self._budget.max_documents_total:
                    break

        if not any_success:
            return IngestionResult(
                source_type=SourceType.ACADEMIC, status=IngestionStatus.ERROR, items=(),
                detail="No academic provider was reachable: " + "; ".join(degraded_notes[:3]),
            )
        status = IngestionStatus.CONNECTED if not degraded_notes else IngestionStatus.DEGRADED
        detail = f"{len(items)} paper(s) normalized (deduplicated) across {len(queries)} quer{'y' if len(queries)==1 else 'ies'}."
        if degraded_notes:
            detail += " Some providers failed: " + "; ".join(degraded_notes[:3])
        return IngestionResult(source_type=SourceType.ACADEMIC, status=status, items=tuple(items), detail=detail)
