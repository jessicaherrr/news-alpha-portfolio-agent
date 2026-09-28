"""Tests for the "Ask the Agent" conversational surface on the Agent page
(Release UX Part A, task spec sections 1/52/58). Streamlit rendering via a
real `AppTest` run; `alpha_agent.ui.databento_context` is monkeypatched so no
real network call happens in this test.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from alpha_agent.marketdata.databento_schemas import (
    ContractResolution,
    DatabentoCapability,
    MarketSnapshot,
)
from alpha_agent.ui import databento_context


def _fake_snapshot(root: str) -> MarketSnapshot:
    return MarketSnapshot(
        root_symbol=root,
        contract=ContractResolution(root_symbol=root, display_symbol=f"{root}.v.0", resolved_raw_symbol=f"{root}U6",
                                     resolved_at=datetime.now(UTC)),
        last=100.0, change_pct=0.5, as_of=datetime.now(UTC), capability=DatabentoCapability.LATEST_AVAILABLE,
    )


@pytest.fixture(autouse=True)
def _patch_market(monkeypatch):
    monkeypatch.setattr(databento_context, "market_snapshot", lambda root: _fake_snapshot(root or "NQ"))
    yield


def test_ask_the_agent_answers_a_market_question_without_creating_an_experiment():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)

    at.text_input(key="agent-ask-input").set_value("What is happening in NQ today?")
    at.button(key="agent-ask-send").click().run(timeout=90)
    assert not list(at.exception)

    full_text = " ".join(m.value for m in at.markdown)
    caption_text = " ".join(c.value for c in at.caption)
    # The raw internal intent label is never shown prominently (item 18/37)
    # -- it only lives inside the per-turn "Details" expander's caption.
    assert "MARKET_QUESTION" not in full_text
    assert "MARKET_QUESTION" in caption_text
    # No compiled hypothesis / research action state was created by asking.
    assert "lab_compiled" not in at.session_state or not at.session_state["lab_compiled"]


def test_ask_the_agent_never_executes_a_fast_screen_from_chat():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)

    at.text_input(key="agent-ask-input").set_value("Run Fast Screen on these candidates")
    at.button(key="agent-ask-send").click().run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    caption_text = " ".join(c.value for c in at.caption)
    assert "RUN_FAST_SCREEN" not in full_text
    assert "RUN_FAST_SCREEN" in caption_text
    assert "Suggested action" in full_text or "click" in full_text.lower()


def test_ask_the_agent_empty_input_does_nothing():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    n_before = len(at.session_state["agent_transcript"])
    at.button(key="agent-ask-send").click().run(timeout=90)
    assert not list(at.exception)
    n_after = len(at.session_state["agent_transcript"])
    assert n_after == n_before


# ---------------------------------------------------------------------------
# EMPTY / LOW-QUALITY CLAUDE TURN HANDLING (Agent Evidence + Research
# Provenance acceptance pass, task spec section 6): an empty reply from
# Claude (e.g. cut off before any text content block) must never render as an
# ordinary, silently-blank Agentic Alpha conversation card.
# ---------------------------------------------------------------------------


class _EmptyReplyAnthropicClient:
    """A fake `AnthropicClient` that returns a response with no text -- the
    same shape a `max_tokens`-truncated or non-text-only reply would have."""

    def __init__(self, *args, **kwargs):
        pass

    def complete(self, **kwargs):
        from alpha_agent.agents.llm import LLMResponse

        return LLMResponse(text="", model="claude-sonnet-5-test", stop_reason="max_tokens",
                            input_tokens=10, output_tokens=0)


def test_empty_claude_response_renders_as_an_honest_error_not_a_blank_card(monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from alpha_agent.ui import claude_conversation
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(claude_conversation, "AnthropicClient", _EmptyReplyAnthropicClient)

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    at.text_input(key="agent-ask-input").set_value("Why CL instead of NQ?")
    at.button(key="agent-ask-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    # never a silently-blank "Agentic Alpha" conversation card
    transcript = at.session_state["agent_transcript"]
    conversation_turns = [t for t in transcript if t["type"] == "conversation"]
    assert conversation_turns == []

    error_turns = [t for t in transcript if t["type"] == "error"]
    assert error_turns, "an empty Claude reply must render an honest error turn instead"
    assert "empty response" in error_turns[-1]["text"].lower()
    error_text = " ".join(e.value for e in at.error)
    assert "empty response" in error_text.lower()
