"""Persistent, cross-run Strategy Research Source Cache (Release UX Part J,
task spec sections 37-40/57).

`alpha_agent.knowledge.dedup` already deduplicates WITHIN one
`StrategyKnowledgeBase` build (task spec section 24); nothing in this
codebase remembered a source across separate runs before this module. This
is a SEPARATE SQLite store (`data/knowledge_cache/source_cache.sqlite`,
gitignored -- operational cache, never registry truth), never mixed with
`alpha_agent.registry.sqlite_registry.ExperimentRegistry`: this cache never
computes a mechanism, a duplicate decision, or a verdict for the scientific
plane, and no registry table is ever read or written here.

VERSIONING (task spec section 39): a GitHub repository is identified by
`(provider, repository)`; a NEW `commit_sha` for the SAME repository creates
a NEW, preserved version row (`version_ordinal` increments, the prior row's
`is_current` flips to False -- it is never deleted or overwritten). An
academic/practitioner paper is identified by DOI when resolvable, else a
normalized title -- these dedupe to one row (papers do not get new
"commits"); a re-ingest only refreshes `retrieved_at`.

NEVER STORES A SECRET (task spec section 38): `GITHUB_TOKEN`,
`DATABENTO_API_KEY`, and `ANTHROPIC_API_KEY` have no column here and no
function in this module accepts a credential parameter --
`tests/python/test_source_cache.py` statically greps this file for those
names.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from alpha_agent.knowledge.models import StrategyKnowledgeItem

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DB_PATH = REPO_ROOT / "data" / "knowledge_cache" / "source_cache.sqlite"
SCHEMA_VERSION = 1

_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
_TITLE_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_title(title: str) -> str:
    return _TITLE_NORMALIZE_RE.sub(" ", title.lower()).strip()


def _extract_doi(url: str | None) -> str | None:
    if not url:
        return None
    m = _DOI_RE.search(url)
    return m.group(0).rstrip(".") if m else None


def _sha256_json(payload: dict) -> str:
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class CacheStatus(str, Enum):
    """What `ResearchSourceCache.record` actually did -- never silently
    conflated (task spec section 39: "do not silently overwrite")."""

    NEW = "NEW"
    REUSED = "REUSED"
    NEW_VERSION = "NEW_VERSION"


class SourceVersionRecord(BaseModel):
    """One immutable version of one source (task spec section 38's field
    list). `raw_content_hash` is the connector's own content-derived
    provenance hash (`StrategyKnowledgeItem.provenance_hash`) -- this cache
    sits above the connector, so it never sees the vendor's literal HTTP
    bytes; it is honestly the strongest content fingerprint actually
    available at this layer."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "source-version-record/1"
    provider: str
    source_id: str
    source_type: str
    url: str | None = None
    repository: str | None = None
    commit_sha: str | None = None
    doi: str | None = None
    publication_date: str | None = None
    retrieved_at: str
    raw_content_hash: str
    normalized_document_hash: str
    knowledge_item_hash: str
    mechanism_id: str
    source_quality: str
    version_ordinal: int
    is_current: bool
    refresh_policy: str = "MANUAL"


class CacheWriteOutcome(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    status: CacheStatus
    record: SourceVersionRecord


def _identity_and_version_key(item: StrategyKnowledgeItem) -> tuple[str, str | None]:
    """`(source_id, version_key)`. `version_key is None` means this source
    type has no versioning concept (dedupe-only, refresh in place);
    otherwise a change in `version_key` for the same `source_id` creates a
    new preserved version (task spec section 39)."""
    provider = item.source_type.value
    if item.repository:
        # A GitHub-shaped item -- versioned by commit SHA.
        return f"{provider.lower()}:{item.repository}", item.commit_sha
    doi = _extract_doi(item.source_url)
    if doi:
        return f"doi:{doi.lower()}", None
    if item.source_url:
        return f"{provider.lower()}:{item.source_url}", None
    return f"{provider.lower()}:title:{_normalize_title(item.title)}", None


def _normalized_document_hash(item: StrategyKnowledgeItem) -> str:
    return _sha256_json(
        {
            "title": _normalize_title(item.title),
            "economic_mechanism": item.economic_mechanism.value,
            "entry_logic_summary": item.entry_logic_summary,
        }
    )


class ResearchSourceCache:
    """SQLite-backed persistent source cache. Distinct from, and never
    imports, `alpha_agent.registry.sqlite_registry`."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else DEFAULT_CACHE_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS source_versions ("
                " row_id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " provider TEXT NOT NULL,"
                " source_id TEXT NOT NULL,"
                " source_type TEXT NOT NULL,"
                " url TEXT,"
                " repository TEXT,"
                " commit_sha TEXT,"
                " doi TEXT,"
                " publication_date TEXT,"
                " retrieved_at TEXT NOT NULL,"
                " raw_content_hash TEXT NOT NULL,"
                " normalized_document_hash TEXT NOT NULL,"
                " knowledge_item_hash TEXT NOT NULL,"
                " mechanism_id TEXT NOT NULL,"
                " source_quality TEXT NOT NULL,"
                " version_ordinal INTEGER NOT NULL,"
                " is_current INTEGER NOT NULL,"
                " refresh_policy TEXT NOT NULL DEFAULT 'MANUAL',"
                " UNIQUE(source_id, version_ordinal)"
                ")"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_source_versions_source_id ON source_versions(source_id)"
            )
            conn.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_record(row: tuple) -> SourceVersionRecord:
        (
            _row_id, provider, source_id, source_type, url, repository, commit_sha, doi,
            publication_date, retrieved_at, raw_content_hash, normalized_document_hash,
            knowledge_item_hash, mechanism_id, source_quality, version_ordinal, is_current,
            refresh_policy,
        ) = row
        return SourceVersionRecord(
            provider=provider, source_id=source_id, source_type=source_type, url=url,
            repository=repository, commit_sha=commit_sha, doi=doi, publication_date=publication_date,
            retrieved_at=retrieved_at, raw_content_hash=raw_content_hash,
            normalized_document_hash=normalized_document_hash, knowledge_item_hash=knowledge_item_hash,
            mechanism_id=mechanism_id, source_quality=source_quality, version_ordinal=version_ordinal,
            is_current=bool(is_current), refresh_policy=refresh_policy,
        )

    def _latest(self, conn: sqlite3.Connection, source_id: str) -> tuple | None:
        return conn.execute(
            "SELECT row_id, provider, source_id, source_type, url, repository, commit_sha, doi, "
            "publication_date, retrieved_at, raw_content_hash, normalized_document_hash, "
            "knowledge_item_hash, mechanism_id, source_quality, version_ordinal, is_current, refresh_policy "
            "FROM source_versions WHERE source_id = ? ORDER BY version_ordinal DESC LIMIT 1",
            (source_id,),
        ).fetchone()

    def record(self, item: StrategyKnowledgeItem) -> CacheWriteOutcome:
        source_id, version_key = _identity_and_version_key(item)
        provider = item.source_type.value
        knowledge_item_hash = _sha256_json(item.model_dump(mode="json"))
        normalized_hash = _normalized_document_hash(item)
        now = _now_iso()

        with self._connect() as conn:
            latest = self._latest(conn, source_id)

            if latest is None:
                record = SourceVersionRecord(
                    provider=provider, source_id=source_id, source_type=provider, url=item.source_url,
                    repository=item.repository, commit_sha=item.commit_sha,
                    doi=_extract_doi(item.source_url), publication_date=item.publication_date,
                    retrieved_at=now, raw_content_hash=item.provenance_hash,
                    normalized_document_hash=normalized_hash, knowledge_item_hash=knowledge_item_hash,
                    mechanism_id=item.economic_mechanism.value, source_quality=item.source_quality.value,
                    version_ordinal=1, is_current=True,
                )
                self._insert(conn, record)
                return CacheWriteOutcome(status=CacheStatus.NEW, record=record)

            latest_record = self._row_to_record(latest)

            is_new_version = version_key is not None and version_key != latest_record.commit_sha
            if not is_new_version:
                # Same version -- refresh retrieved_at in place. This is the
                # ONE field mutated on an existing row (a freshness
                # timestamp, never content); never `INSERT OR REPLACE`.
                conn.execute(
                    "UPDATE source_versions SET retrieved_at = ? WHERE row_id = ?",
                    (now, latest[0]),
                )
                refreshed = latest_record.model_copy(update={"retrieved_at": now})
                return CacheWriteOutcome(status=CacheStatus.REUSED, record=refreshed)

            # A genuinely new version (e.g. a new commit SHA) -- preserve the
            # old row untouched, insert a new one, flip currency.
            conn.execute("UPDATE source_versions SET is_current = 0 WHERE row_id = ?", (latest[0],))
            record = SourceVersionRecord(
                provider=provider, source_id=source_id, source_type=provider, url=item.source_url,
                repository=item.repository, commit_sha=item.commit_sha,
                doi=_extract_doi(item.source_url), publication_date=item.publication_date,
                retrieved_at=now, raw_content_hash=item.provenance_hash,
                normalized_document_hash=normalized_hash, knowledge_item_hash=knowledge_item_hash,
                mechanism_id=item.economic_mechanism.value, source_quality=item.source_quality.value,
                version_ordinal=latest_record.version_ordinal + 1, is_current=True,
            )
            self._insert(conn, record)
            return CacheWriteOutcome(status=CacheStatus.NEW_VERSION, record=record)

    def _insert(self, conn: sqlite3.Connection, record: SourceVersionRecord) -> None:
        conn.execute(
            "INSERT INTO source_versions (provider, source_id, source_type, url, repository, commit_sha, "
            "doi, publication_date, retrieved_at, raw_content_hash, normalized_document_hash, "
            "knowledge_item_hash, mechanism_id, source_quality, version_ordinal, is_current, refresh_policy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.provider, record.source_id, record.source_type, record.url, record.repository,
                record.commit_sha, record.doi, record.publication_date, record.retrieved_at,
                record.raw_content_hash, record.normalized_document_hash, record.knowledge_item_hash,
                record.mechanism_id, record.source_quality, record.version_ordinal,
                int(record.is_current), record.refresh_policy,
            ),
        )

    def history(self, source_id: str) -> list[SourceVersionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT row_id, provider, source_id, source_type, url, repository, commit_sha, doi, "
                "publication_date, retrieved_at, raw_content_hash, normalized_document_hash, "
                "knowledge_item_hash, mechanism_id, source_quality, version_ordinal, is_current, refresh_policy "
                "FROM source_versions WHERE source_id = ? ORDER BY version_ordinal ASC",
                (source_id,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def current(self, source_id: str) -> SourceVersionRecord | None:
        with self._connect() as conn:
            row = self._latest(conn, source_id)
        return self._row_to_record(row) if row else None

    def stats(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM source_versions").fetchone()[0]
            distinct_sources = conn.execute("SELECT COUNT(DISTINCT source_id) FROM source_versions").fetchone()[0]
            by_provider = dict(
                conn.execute(
                    "SELECT provider, COUNT(DISTINCT source_id) FROM source_versions GROUP BY provider"
                ).fetchall()
            )
        return {
            "total_version_rows": total, "distinct_sources": distinct_sources, "by_provider": by_provider,
        }


__all__ = [
    "DEFAULT_CACHE_DB_PATH",
    "CacheStatus",
    "CacheWriteOutcome",
    "ResearchSourceCache",
    "SourceVersionRecord",
]
