"""Phase 8 -- the persistent Community store.

A separate, append-only SQLite store (``data/community/community.sqlite`` by
default) -- deliberately NOT the Phase 14 experiment registry
(``data/registry/experiments.sqlite``), mirroring
:mod:`alpha_agent.paper.ledger`'s own "operational store, never confused with
the registry" pattern exactly. A :class:`Contribution` / :class:`ReplicationRecord`
is a piece of COMMUNITY metadata about real registry evidence -- it is never
itself scientific truth, and this module never writes to the scientific
registry (it only reads one, via :mod:`alpha_agent.community.contributions`,
to build the :class:`~alpha_agent.community.schemas.EvidenceReference` a row
carries).

``contributions`` / ``replications`` are append-only: nothing in this module
ever issues an ``UPDATE`` or ``DELETE`` against either table, matching
CLAUDE.md's "failed experiments are first-class data" ethos extended to
community evidence -- a REJECTed or superseded contribution is preserved
verbatim, never deleted or silently edited (prompt section 4/9's "conflicting/
rejected evidence handling").
"""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

DEFAULT_COMMUNITY_STORE_PATH = Path("data/community/community.sqlite")

CURRENT_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contributions (
    contribution_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    kind TEXT NOT NULL,
    visibility TEXT NOT NULL,
    contributor_id TEXT NOT NULL,
    contributor_display_name TEXT NOT NULL,
    mechanism TEXT,
    root_symbol TEXT NOT NULL,
    title TEXT NOT NULL,
    notes TEXT NOT NULL,
    novelty_notes TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    recycling_decision TEXT NOT NULL,
    recycling_explanation TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS replications (
    replication_id TEXT PRIMARY KEY,
    contribution_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    replicator_id TEXT NOT NULL,
    replicator_display_name TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    comparison_json TEXT NOT NULL,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (contribution_id) REFERENCES contributions(contribution_id)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class UnknownContribution(RuntimeError):
    pass


class CommunityStore:
    """Thin, typed wrapper over the append-only community SQLite store.

    Deliberately dict-in / dict-out at the storage boundary (mirroring
    ``alpha_agent.paper.ledger``'s row-model pattern) -- the caller
    (:mod:`alpha_agent.community.contributions` /
    :mod:`alpha_agent.community.replication`) owns constructing and
    validating the typed :class:`~alpha_agent.community.schemas.Contribution`
    / :class:`~alpha_agent.community.schemas.ReplicationRecord` objects this
    wraps.
    """

    def __init__(self, path: str | Path = DEFAULT_COMMUNITY_STORE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- contributions ------------------------------------------------------
    def insert_contribution(self, row: dict) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO contributions (
                    contribution_id, schema_version, kind, visibility,
                    contributor_id, contributor_display_name, mechanism,
                    root_symbol, title, notes, novelty_notes, evidence_json,
                    created_at, recycling_decision, recycling_explanation
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["contribution_id"], row["schema_version"], row["kind"], row["visibility"],
                    row["contributor_id"], row["contributor_display_name"], row["mechanism"],
                    row["root_symbol"], row["title"], row["notes"], row["novelty_notes"],
                    json.dumps(row["evidence"], sort_keys=True), row["created_at"],
                    row["recycling_decision"], row["recycling_explanation"],
                ),
            )

    def get_contribution(self, contribution_id: str) -> dict | None:
        r = self._conn.execute(
            "SELECT * FROM contributions WHERE contribution_id = ?", (contribution_id,)
        ).fetchone()
        return _row_to_contribution_dict(r) if r is not None else None

    def list_contributions(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM contributions ORDER BY created_at ASC"
        ).fetchall()
        return [_row_to_contribution_dict(r) for r in rows]

    # -- replications ---------------------------------------------------------
    def insert_replication(self, row: dict) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO replications (
                    replication_id, contribution_id, schema_version,
                    replicator_id, replicator_display_name, evidence_json,
                    comparison_json, notes, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["replication_id"], row["contribution_id"], row["schema_version"],
                    row["replicator_id"], row["replicator_display_name"],
                    json.dumps(row["evidence"], sort_keys=True),
                    json.dumps(row["comparison"], sort_keys=True),
                    row["notes"], row["created_at"],
                ),
            )

    def list_replications(self, contribution_id: str | None = None) -> list[dict]:
        if contribution_id is None:
            rows = self._conn.execute(
                "SELECT * FROM replications ORDER BY created_at ASC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM replications WHERE contribution_id = ? ORDER BY created_at ASC",
                (contribution_id,),
            ).fetchall()
        return [_row_to_replication_dict(r) for r in rows]


def _row_to_contribution_dict(r: sqlite3.Row) -> dict:
    return {
        "contribution_id": r["contribution_id"],
        "schema_version": r["schema_version"],
        "kind": r["kind"],
        "visibility": r["visibility"],
        "contributor_id": r["contributor_id"],
        "contributor_display_name": r["contributor_display_name"],
        "mechanism": r["mechanism"],
        "root_symbol": r["root_symbol"],
        "title": r["title"],
        "notes": r["notes"],
        "novelty_notes": r["novelty_notes"],
        "evidence": json.loads(r["evidence_json"]),
        "created_at": r["created_at"],
        "recycling_decision": r["recycling_decision"],
        "recycling_explanation": r["recycling_explanation"],
    }


def _row_to_replication_dict(r: sqlite3.Row) -> dict:
    return {
        "replication_id": r["replication_id"],
        "contribution_id": r["contribution_id"],
        "schema_version": r["schema_version"],
        "replicator_id": r["replicator_id"],
        "replicator_display_name": r["replicator_display_name"],
        "evidence": json.loads(r["evidence_json"]),
        "comparison": json.loads(r["comparison_json"]),
        "notes": r["notes"],
        "created_at": r["created_at"],
    }
