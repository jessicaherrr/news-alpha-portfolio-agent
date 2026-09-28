"""Checkpoint A regression coverage -- Agent transcript duplicate-element-key
fix (see `alpha_agent.ui.views.agent`).

OBSERVED REAL BUG: submitting the exact objective
"Propose a robust multi-week trend-following hypothesis for an index future."
(the default/first preset, scripted/offline mode -- fully deterministic)
twice in one session produced two transcript turns carrying the identical
`hypothesis_id` ("H-DEMO-TSMOM-NQ"), and the old hypothesis card was keyed
`agent-hyp-{hypothesis_id}` -- so the second occurrence reused
`card-agent-hyp-h-demo-tsmom-nq` and Streamlit raised
`StreamlitDuplicateElementKey`.

This is a UI OCCURRENCE-IDENTITY bug, not a scientific-identity bug: the same
hypothesis/strategy/experiment legitimately recurs across turns. The fix
(`_append` stamping a stable `event_id` at append time, every repeatable
element key built from `event_id`) is exercised here end-to-end through the
real Streamlit `AppTest` harness -- no scientific behavior is touched, and no
production registry write occurs (the Agent page's Send/PROPOSE path never
writes to the registry; see `alpha_agent.ui.views.agent`'s module docstring).
"""
from __future__ import annotations

import pytest
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(),
    reason="Phase 14 registry sqlite not present in this checkout",
)

_SCRIPT = "from alpha_agent.ui.views.agent import render\nrender()\n"


def _fresh_app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_SCRIPT)
    at.run(timeout=90)
    assert not list(at.exception)
    return at


def _send(at) -> None:
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)


def test_same_objective_sent_twice_does_not_crash_and_both_turns_render():
    """The exact observed failure scenario: submit the identical NQ TSMOM
    objective twice in one session. Must not raise
    `StreamlitDuplicateElementKey` (or anything else), and both conversation
    occurrences of the hypothesis must remain visible."""
    at = _fresh_app()

    _send(at)
    transcript_after_first = list(at.session_state["agent_transcript"])
    hyp_turns_1 = [t for t in transcript_after_first if t["type"] == "hypothesis"]
    assert len(hyp_turns_1) == 1
    assert hyp_turns_1[0]["data"]["hypothesis_id"] == "H-DEMO-TSMOM-NQ"

    _send(at)  # the exact same preset/objective/mode again
    transcript_after_second = list(at.session_state["agent_transcript"])
    hyp_turns_2 = [t for t in transcript_after_second if t["type"] == "hypothesis"]

    # Scientific identity: unchanged, both occurrences carry the same hypothesis_id.
    assert len(hyp_turns_2) == 2
    assert {t["data"]["hypothesis_id"] for t in hyp_turns_2} == {"H-DEMO-TSMOM-NQ"}

    # UI occurrence identity: the two turns are distinct entries with distinct event_ids.
    assert hyp_turns_2[0]["event_id"] != hyp_turns_2[1]["event_id"]

    # Rendering both occurrences did not crash -- a live proof the crashing
    # card (`agent-hyp-...`) rendered twice without a duplicate Streamlit key.
    # (The title text also appears in unrelated chrome such as the preset
    # selectbox option, so assert "at least twice", not an exact count.)
    markdown_text = " ".join(md.value for md in at.markdown)
    assert markdown_text.count("NQ multi-week time-series momentum") >= 2


def test_compiled_strategy_card_renders_twice_without_collision():
    """Same scripted objective twice -> identical `strategy_fingerprint`
    (research-object identity) in two "compiled" turns; the compiled-strategy
    card used to be keyed on the fingerprint alone."""
    at = _fresh_app()
    _send(at)
    _send(at)
    compiled_turns = [t for t in at.session_state["agent_transcript"] if t["type"] == "compiled"]
    assert len(compiled_turns) == 2
    assert compiled_turns[0]["data"]["strategy_fingerprint"] == compiled_turns[1]["data"]["strategy_fingerprint"]
    assert compiled_turns[0]["event_id"] != compiled_turns[1]["event_id"]


def test_evidence_and_failure_memory_cards_render_twice_without_collision():
    """Registry evidence and pre-proposal Failure Memory are re-fetched (not
    cached) on each Send, so the exact same evidence/FailureMemory content can
    legitimately appear in two turns -- must not collide either."""
    at = _fresh_app()
    _send(at)
    _send(at)
    transcript = at.session_state["agent_transcript"]
    evidence_turns = [t for t in transcript if t["type"] == "evidence"]
    assert len(evidence_turns) == 2
    assert evidence_turns[0]["event_id"] != evidence_turns[1]["event_id"]

    fm_turns = [t for t in transcript if t["type"] == "failure_memory"]
    if fm_turns:  # only present when failure memory exists for the objective's universe
        assert len({t["event_id"] for t in fm_turns}) == len(fm_turns)


def test_rerun_without_new_message_preserves_stable_element_ids():
    """A plain Streamlit rerun (no new Send) must not regenerate event_ids --
    they are computed once at append time, never during rendering."""
    at = _fresh_app()
    _send(at)
    before = [t["event_id"] for t in at.session_state["agent_transcript"]]

    at.run(timeout=90)  # rerun with no new interaction
    assert not list(at.exception)
    after = [t["event_id"] for t in at.session_state["agent_transcript"]]
    assert before == after


def test_clear_conversation_resets_state_and_first_new_message_works():
    at = _fresh_app()
    _send(at)
    assert at.session_state["agent_transcript"]

    at.button(key="agent-clear").click().run(timeout=90)
    assert not list(at.exception)
    assert at.session_state["agent_transcript"] == []

    _send(at)
    assert not list(at.exception)
    transcript = at.session_state["agent_transcript"]
    assert transcript
    # event_ids restart from a clean, deterministic sequence after Clear.
    assert transcript[0]["event_id"] == "agent-event-000000"


def test_new_transcript_entries_always_get_an_explicit_event_id():
    """Directly exercises `_append`/`_event_id`: never `uuid.uuid4()`, always
    a deterministic id derived from append-time transcript position."""
    from alpha_agent.ui.views import agent as agent_view

    class _FakeSessionState(dict):
        def setdefault(self, key, default):
            return dict.setdefault(self, key, default)

    fake_state = _FakeSessionState()
    import streamlit as st

    original_state = st.session_state
    st.session_state = fake_state  # type: ignore[assignment]
    try:
        agent_view._append("user", text="first")
        agent_view._append("user", text="second (same content)")
        turns = fake_state[agent_view._TRANSCRIPT_KEY]
        assert turns[0]["event_id"] == "agent-event-000000"
        assert turns[1]["event_id"] == "agent-event-000001"
        assert turns[0]["event_id"] != turns[1]["event_id"]
    finally:
        st.session_state = original_state


def test_legacy_transcript_entry_without_event_id_gets_a_deterministic_fallback():
    """Backward-compatibility path (A3): a transcript entry that predates this
    fix (no stored `event_id`) must still render with a stable, deterministic,
    non-random fallback id derived from its position."""
    from alpha_agent.ui.views import agent as agent_view

    legacy_turn = {"type": "user", "text": "pre-fix entry", "at": "2024-01-01 00:00:00 UTC"}
    assert agent_view._event_id(legacy_turn, 3) == "agent-legacy-000003"
    assert agent_view._event_id(legacy_turn, 3) == agent_view._event_id(legacy_turn, 3)  # stable, not random

    fresh_turn = {"type": "user", "event_id": "agent-event-000007"}
    assert agent_view._event_id(fresh_turn, 0) == "agent-event-000007"  # stored id always wins
