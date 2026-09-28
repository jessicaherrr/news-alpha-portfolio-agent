"""``NewsStore`` -- Section 16. A dedicated, OBSERVATIONAL SQLite cache,
explicitly OUTSIDE `alpha_agent.registry.sqlite_registry.ExperimentRegistry`
(mirrors `alpha_agent.knowledge.source_cache`'s own "separate, gitignored,
operational cache" pattern for the research-knowledge plane). Safe to delete
and rebuild at any time -- it is provenance/observation cache, never
scientific memory, and no registry table is ever read or written here.

NEVER STORES A SECRET: no column here holds `USDA_MMN_API_KEY` (or any other
credential), and no function in this module accepts one --
`tests/python/test_market_intel_news.py` statically greps this file for
that.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from alpha_agent.market_intel.dedup import is_probable_duplicate, storage_key
from alpha_agent.market_intel.event_schemas import EventImportance, ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "market_intelligence" / "news" / "market_news.sqlite"
DEFAULT_EVENTS_DB_PATH = REPO_ROOT / "data" / "market_intelligence" / "news" / "market_events.sqlite"
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_items (
    storage_key TEXT PRIMARY KEY,
    news_id TEXT NOT NULL,
    headline TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_url TEXT NOT NULL,
    published_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    related_products TEXT NOT NULL,
    related_asset_classes TEXT NOT NULL,
    category TEXT NOT NULL,
    mapping_reason TEXT NOT NULL,
    summary TEXT,
    event_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_published_at ON news_items(published_at);
"""


class NewsStore:
    """One SQLite file, WAL-free (this is a low-write observational cache,
    not a high-throughput ledger). ``db_path`` is injectable for tests --
    the real default lives under ``data/market_intelligence/news/``, which
    is gitignored exactly like `alpha_agent.knowledge.source_cache`'s own
    cache directory."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add_item(self, item: MarketNewsItem) -> bool:
        """Inserts `item` unless it is a real duplicate (Section 20) --
        returns whether a new row was actually written. Checks the fast
        canonical-URL primary key first, then scans same-source items
        already retrieved within the proximity window (bounded, not the
        whole table) for a probable near-duplicate."""
        key = storage_key(item)
        with self._connect() as conn:
            existing = conn.execute("SELECT 1 FROM news_items WHERE storage_key = ?", (key,)).fetchone()
            if existing is not None:
                return False
            for row in conn.execute(
                "SELECT * FROM news_items WHERE source_name = ? ORDER BY published_at DESC LIMIT 50",
                (item.source_name,),
            ):
                if is_probable_duplicate(item, _row_to_item(row)):
                    return False
            conn.execute(
                "INSERT INTO news_items (storage_key, news_id, headline, source_name, source_type, source_url, "
                "published_at, retrieved_at, related_products, related_asset_classes, category, mapping_reason, "
                "summary, event_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key, item.news_id, item.headline, item.source_name, item.source_type.value, item.source_url,
                    item.published_at.isoformat(), item.retrieved_at.isoformat(),
                    json.dumps(list(item.related_products)), json.dumps(list(item.related_asset_classes)),
                    item.category.value, item.mapping_reason, item.summary, item.event_type,
                ),
            )
        return True

    def list_recent(
        self, *, since: datetime, limit: int = 100, related_product: str | None = None,
    ) -> list[MarketNewsItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM news_items WHERE published_at >= ? ORDER BY published_at DESC LIMIT ?",
                (since.isoformat(), limit),
            ).fetchall()
        items = [_row_to_item(r) for r in rows]
        if related_product:
            items = [i for i in items if related_product.upper() in i.related_products]
        return items

    def count_recent(self, *, since: datetime, related_product: str | None = None) -> int:
        return len(self.list_recent(since=since, limit=100_000, related_product=related_product))


def _row_to_item(row: sqlite3.Row) -> MarketNewsItem:
    return MarketNewsItem(
        news_id=row["news_id"], headline=row["headline"], source_name=row["source_name"],
        source_type=NewsSourceType(row["source_type"]), source_url=row["source_url"],
        published_at=datetime.fromisoformat(row["published_at"]), retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
        related_products=tuple(json.loads(row["related_products"])),
        related_asset_classes=tuple(json.loads(row["related_asset_classes"])),
        category=NewsCategory(row["category"]), mapping_reason=row["mapping_reason"],
        summary=row["summary"], event_type=row["event_type"],
    )


def now_utc() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# EventStore -- Checkpoint F. Same observational-cache posture as NewsStore
# (Section 16): a separate SQLite file, gitignored, safe to delete/rebuild,
# outside ExperimentRegistry.
# ---------------------------------------------------------------------------

_EVENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS scheduled_events (
    event_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_url TEXT NOT NULL,
    scheduled_at TEXT NOT NULL,
    timezone TEXT NOT NULL,
    category TEXT NOT NULL,
    importance TEXT NOT NULL,
    importance_rule TEXT NOT NULL,
    affected_products TEXT NOT NULL,
    mapping_reason TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    actual_release_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_scheduled_at ON scheduled_events(scheduled_at);
"""


class EventStore:
    """Mirrors `NewsStore`'s structure exactly, for `ScheduledMarketEvent`.
    Upserts by `event_id` (each real connector derives a stable, content-
    addressed id -- e.g. ``fomc:2026-09-16`` -- so re-fetching the same
    real event just refreshes its row, never duplicates it)."""

    def __init__(self, db_path: str | Path = DEFAULT_EVENTS_DB_PATH):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_EVENT_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_event(self, event: ScheduledMarketEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO scheduled_events (event_id, name, source_name, source_url, scheduled_at, timezone, "
                "category, importance, importance_rule, affected_products, mapping_reason, retrieved_at, "
                "actual_release_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(event_id) DO UPDATE SET name=excluded.name, scheduled_at=excluded.scheduled_at, "
                "importance=excluded.importance, importance_rule=excluded.importance_rule, "
                "affected_products=excluded.affected_products, mapping_reason=excluded.mapping_reason, "
                "retrieved_at=excluded.retrieved_at, actual_release_at=excluded.actual_release_at",
                (
                    event.event_id, event.name, event.source_name, event.source_url,
                    event.scheduled_at.isoformat(), event.timezone, event.category, event.importance.value,
                    event.importance_rule, json.dumps(list(event.affected_products)), event.mapping_reason,
                    event.retrieved_at.isoformat(),
                    event.actual_release_at.isoformat() if event.actual_release_at else None,
                ),
            )

    def list_upcoming(self, *, now: datetime, horizon_days: int = 45) -> list[ScheduledMarketEvent]:
        cutoff = (now + timedelta(days=horizon_days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scheduled_events WHERE scheduled_at >= ? AND scheduled_at <= ? "
                "ORDER BY scheduled_at ASC",
                (now.isoformat(), cutoff),
            ).fetchall()
        return [_row_to_event(r) for r in rows]


def _row_to_event(row: sqlite3.Row) -> ScheduledMarketEvent:
    return ScheduledMarketEvent(
        event_id=row["event_id"], name=row["name"], source_name=row["source_name"], source_url=row["source_url"],
        scheduled_at=datetime.fromisoformat(row["scheduled_at"]), timezone=row["timezone"], category=row["category"],
        importance=EventImportance(row["importance"]), importance_rule=row["importance_rule"],
        affected_products=tuple(json.loads(row["affected_products"])), mapping_reason=row["mapping_reason"],
        retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
        actual_release_at=datetime.fromisoformat(row["actual_release_at"]) if row["actual_release_at"] else None,
    )


__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_EVENTS_DB_PATH",
    "SCHEMA_VERSION",
    "EventStore",
    "NewsStore",
    "now_utc",
]
