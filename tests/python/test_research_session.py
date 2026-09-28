"""Tests for `alpha_agent.agents.research_session` (Release UX Part B, task
spec sections 5-7/52). Every test uses a temp-directory-backed
`ResearchSessionStore` -- never the real `data/sessions/` path.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from alpha_agent.agents.research_session import (
    MarketContextMode,
    ResearchSession,
    ResearchSessionStage,
    ResearchSessionStore,
)


@pytest.fixture
def store(tmp_path: Path) -> ResearchSessionStore:
    return ResearchSessionStore(path=tmp_path / "sessions.sqlite")


def test_new_session_has_a_stable_id_and_historical_default_mode():
    session = ResearchSession(selected_market="NQ", research_objective="Find robust alpha opportunities.")
    assert session.research_session_id.startswith("session-")
    assert session.market_context_mode == MarketContextMode.HISTORICAL
    assert session.market_aware_provenance is None
    assert session.current_stage == ResearchSessionStage.INITIATED


def test_save_and_get_round_trip(store):
    session = ResearchSession(selected_market="ES")
    store.save(session)
    fetched = store.get(session.research_session_id)
    assert fetched is not None
    assert fetched.selected_market == "ES"
    assert fetched.research_session_id == session.research_session_id


def test_get_missing_session_returns_none(store):
    assert store.get("session-does-not-exist") is None


def test_two_sessions_remain_isolated(store):
    a = ResearchSession(selected_market="NQ", research_objective="A")
    b = ResearchSession(selected_market="ES", research_objective="B")
    a.record_question("Find a trend strategy")
    b.record_question("Find a mean-reversion strategy")
    store.save(a)
    store.save(b)

    fetched_a = store.get(a.research_session_id)
    fetched_b = store.get(b.research_session_id)
    assert fetched_a.research_questions == ["Find a trend strategy"]
    assert fetched_b.research_questions == ["Find a mean-reversion strategy"]
    assert fetched_a.research_session_id != fetched_b.research_session_id


def test_save_updates_in_place_never_duplicates_a_row(store):
    session = ResearchSession(selected_market="NQ")
    store.save(session)
    session.record_question("first question")
    store.save(session)
    session.record_question("second question")
    store.save(session)

    all_sessions = store.list_sessions()
    matching = [s for s in all_sessions if s.research_session_id == session.research_session_id]
    assert len(matching) == 1
    assert matching[0].research_questions == ["first question", "second question"]


def test_record_hypothesis_advances_stage_and_dedupes():
    session = ResearchSession()
    session.record_hypothesis("hyp-1")
    session.record_hypothesis("hyp-1")  # idempotent
    session.record_hypothesis("hyp-2")
    assert session.hypothesis_ids == ["hyp-1", "hyp-2"]
    assert session.current_stage == ResearchSessionStage.HYPOTHESES_PROPOSED


def test_touch_advances_updated_at_but_not_created_at():
    session = ResearchSession()
    created = session.created_at
    original_updated = session.updated_at
    import time

    time.sleep(0.01)
    session.touch()
    assert session.created_at == created
    assert session.updated_at >= original_updated


def test_enable_market_aware_always_records_provenance_together():
    session = ResearchSession()
    session.enable_market_aware(
        market_context_as_of="2026-09-13T21:00:00+00:00", market_snapshot_fingerprint="sha256:abc123",
    )
    assert session.is_market_aware
    assert session.market_aware_provenance is not None
    assert session.market_aware_provenance.research_session_id == session.research_session_id
    assert session.market_aware_provenance.market_context_as_of == "2026-09-13T21:00:00+00:00"
    assert session.market_aware_provenance.hypothesis_freeze_timestamp is not None


def test_market_aware_provenance_is_never_set_without_enabling():
    session = ResearchSession()
    assert session.market_context_mode == MarketContextMode.HISTORICAL
    assert session.market_aware_provenance is None


def test_delete_removes_a_session(store):
    session = ResearchSession()
    store.save(session)
    assert store.get(session.research_session_id) is not None
    store.delete(session.research_session_id)
    assert store.get(session.research_session_id) is None


def test_provenance_hash_is_deterministic_for_identical_content():
    a = ResearchSession(research_session_id="fixed-id", created_at="t", updated_at="t", selected_market="NQ")
    b = ResearchSession(research_session_id="fixed-id", created_at="t", updated_at="t", selected_market="NQ")
    assert a.provenance_hash() == b.provenance_hash()


def test_store_never_imports_the_experiment_registry():
    import ast

    path = Path(__file__).resolve().parents[2] / "python" / "alpha_agent" / "agents" / "research_session.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            imported.update(names)
    assert not any("sqlite_registry" in n for n in imported)
