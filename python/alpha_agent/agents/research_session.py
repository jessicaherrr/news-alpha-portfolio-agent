"""Research Sessions (Release UX Part B, task spec sections 5-7).

A `ResearchSession` is the user's RESEARCH-WORKSPACE state -- what has been
asked, which mechanisms/sources have already been explored, which hypotheses
and candidates this conversation has touched, and (if Market-Aware Research
Mode is enabled, see `alpha_agent.agents.market_aware`) the provenance of
that choice. It is explicitly NOT the scientific `ExperimentRegistry`:

* the registry is append-only, deterministic, and never overwritten
  (CLAUDE.md: "INSERT OR REPLACE / UPDATE on any registry table is
  forbidden"); a `ResearchSession` is ordinary MUTABLE workspace state that a
  user's own conversation naturally grows and revises turn by turn, so
  `ResearchSessionStore.save` legitimately replaces a session's row in place.
  Unlike `InvestorProfile` (a frozen point-in-time preference snapshot) a
  session is a running log, so it is deliberately NOT a frozen pydantic
  model -- callers mutate a `ResearchSession` in place and then persist it.
* the registry decides duplicates/authority/verdicts; a session never does
  -- it only remembers what this conversation has already looked at
  (mechanisms explored, hypotheses proposed, candidates rejected) so
  `alpha_agent.ui.conversation_engine` and future discovery calls can avoid
  naive repetition (task spec section 6) without ever overriding registry
  truth (task spec section 7 -- identical inputs may legitimately produce
  the identical, real best-supported candidate).
* nothing here ever writes to `ExperimentRegistry` tables, and this module
  never imports `alpha_agent.registry.sqlite_registry`.

Persistence: a small, dedicated, gitignored SQLite file --
`data/sessions/research_sessions.sqlite` -- distinct from both the
scientific registry and the paper-trading ledger (mirrors `alpha_agent.paper.
ledger.PaperLedger`'s "own separate store" pattern).
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SESSION_DB_PATH = REPO_ROOT / "data" / "sessions" / "research_sessions.sqlite"

SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ResearchSessionStage(str, Enum):
    """Where this session's workspace currently stands. Purely descriptive
    UI/workflow state -- never a scientific gate, never read by the registry."""

    INITIATED = "INITIATED"
    SOURCES_QUERIED = "SOURCES_QUERIED"
    HYPOTHESES_PROPOSED = "HYPOTHESES_PROPOSED"
    FAST_SCREENED = "FAST_SCREENED"
    FROZEN = "FROZEN"
    VALIDATED = "VALIDATED"
    CLOSED = "CLOSED"


class MarketContextMode(str, Enum):
    """Task spec section 19. `HISTORICAL` (the default): current/delayed
    market data is observational only and never influences hypothesis
    generation. `MARKET_AWARE`: current market state MAY influence
    hypothesis-generation PRIORITY (never validity/verdict) -- and, once set,
    this session's provenance fields below become mandatory (task spec
    section 20's holdout consequence)."""

    HISTORICAL = "HISTORICAL"
    MARKET_AWARE = "MARKET_AWARE"


class MarketAwareProvenance(BaseModel):
    """Recorded ONLY when `market_context_mode == MARKET_AWARE` (task spec
    section 19/20). Makes explicit that any hypothesis frozen under this
    provenance was designed AFTER observing real market conditions at
    `market_context_as_of`, and therefore that a subsequent 2025 forward
    evaluation for THIS session's hypotheses is not chronologically pristine
    in the same sense as one designed before any 2026 observation -- prospective
    evaluation for such a hypothesis must begin strictly after
    `hypothesis_freeze_timestamp`, never treat 2025 as untouched-by-influence
    the way `alpha_agent.registry.holdout_guard`'s absolute 2025 lock already
    guarantees for ACCESS (this field records a separate, softer scientific
    caveat: influence, not access)."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "market-aware-provenance/1"
    market_context_as_of: str  # ISO timestamp of the observed snapshot
    market_snapshot_fingerprint: str
    data_source: str = "DATABENTO"
    research_session_id: str
    hypothesis_freeze_timestamp: str = Field(default_factory=_now_iso)


class ResearchSession(BaseModel):
    """Mutable research-workspace state for one user conversation. See
    module docstring for why this is NOT frozen and NOT the registry."""

    model_config = {"extra": "forbid"}

    schema_version: str = "research-session/1"
    research_session_id: str = Field(default_factory=lambda: f"session-{uuid.uuid4().hex[:24]}")
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)

    profile_snapshot: dict = Field(default_factory=dict)
    selected_market: str | None = None
    research_objective: str = ""
    research_mode: str = "conversational"
    market_context_mode: MarketContextMode = MarketContextMode.HISTORICAL
    market_aware_provenance: MarketAwareProvenance | None = None

    conversation_summary: str = ""
    research_questions: list[str] = Field(default_factory=list)
    source_queries: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    mechanism_ids: list[str] = Field(default_factory=list)
    hypothesis_ids: list[str] = Field(default_factory=list)
    strategy_spec_fingerprints: list[str] = Field(default_factory=list)

    fast_screen_manifest: str | None = None
    frozen_manifest_id: str | None = None
    validation_family_id: str | None = None
    current_stage: ResearchSessionStage = ResearchSessionStage.INITIATED

    def touch(self) -> None:
        self.updated_at = _now_iso()

    def record_question(self, question: str) -> None:
        self.research_questions.append(question)
        self.touch()

    def record_mechanism(self, mechanism_id: str) -> None:
        if mechanism_id not in self.mechanism_ids:
            self.mechanism_ids.append(mechanism_id)
        self.touch()

    def record_hypothesis(self, hypothesis_id: str) -> None:
        if hypothesis_id not in self.hypothesis_ids:
            self.hypothesis_ids.append(hypothesis_id)
        self.current_stage = ResearchSessionStage.HYPOTHESES_PROPOSED
        self.touch()

    def enable_market_aware(self, *, market_context_as_of: str, market_snapshot_fingerprint: str,
                             data_source: str = "DATABENTO") -> None:
        """Task spec section 19: switches this session into Market-Aware
        Research Mode and records the mandatory provenance in the same
        step -- there is no code path that sets `market_context_mode =
        MARKET_AWARE` without also recording `market_aware_provenance`."""
        self.market_context_mode = MarketContextMode.MARKET_AWARE
        self.market_aware_provenance = MarketAwareProvenance(
            market_context_as_of=market_context_as_of,
            market_snapshot_fingerprint=market_snapshot_fingerprint,
            data_source=data_source,
            research_session_id=self.research_session_id,
        )
        self.touch()

    @property
    def is_market_aware(self) -> bool:
        return self.market_context_mode == MarketContextMode.MARKET_AWARE

    def provenance_hash(self) -> str:
        import hashlib

        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResearchSessionStore:
    """SQLite-backed persistence for `ResearchSession` rows. One JSON column
    per row -- simplest safe shape for a moderately nested, evolving
    document; NOT a registry table, updates are ordinary replace-in-place."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else DEFAULT_SESSION_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS research_sessions ("
                " research_session_id TEXT PRIMARY KEY,"
                " created_at TEXT NOT NULL,"
                " updated_at TEXT NOT NULL,"
                " payload_json TEXT NOT NULL"
                ")"
            )
            # PRAGMA does not support parameter binding; SCHEMA_VERSION is a
            # fixed module constant, never user input.
            conn.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, session: ResearchSession) -> ResearchSession:
        session.touch()
        payload = session.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO research_sessions (research_session_id, created_at, updated_at, payload_json) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(research_session_id) DO UPDATE SET "
                " updated_at = excluded.updated_at, payload_json = excluded.payload_json",
                (session.research_session_id, session.created_at, session.updated_at, payload),
            )
        return session

    def get(self, research_session_id: str) -> ResearchSession | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM research_sessions WHERE research_session_id = ?",
                (research_session_id,),
            ).fetchone()
        if row is None:
            return None
        return ResearchSession.model_validate_json(row[0])

    def list_sessions(self) -> list[ResearchSession]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM research_sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [ResearchSession.model_validate_json(r[0]) for r in rows]

    def delete(self, research_session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM research_sessions WHERE research_session_id = ?", (research_session_id,)
            )


__all__ = [
    "DEFAULT_SESSION_DB_PATH",
    "MarketAwareProvenance",
    "MarketContextMode",
    "ResearchSession",
    "ResearchSessionStage",
    "ResearchSessionStore",
]
