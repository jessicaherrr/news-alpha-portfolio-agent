"""Typed `proposed_hypothesis` contract (Release UX product-consolidation
acceptance pass, task spec section 5/26). "Research This Hypothesis" must
appear ONLY when a response carries the typed `proposed_hypothesis` field --
never merely because its displayed text happens to contain the phrase
"NOT VALIDATED".
"""
from __future__ import annotations

import pytest
from alpha_agent.ui import claude_conversation

_SCRIPT = "from alpha_agent.ui.views.agent import render\nrender()\n"


# ---------------------------------------------------------------------------
# extraction -- structured, anchored, never a bare substring search
# ---------------------------------------------------------------------------


def test_extract_proposed_hypothesis_absent_by_default():
    text = "This candidate is NOT VALIDATED and never has been -- see Research Details for evidence."
    display, hypothesis = claude_conversation._extract_proposed_hypothesis(text)
    assert hypothesis is None
    assert display == text


def test_extract_proposed_hypothesis_parses_the_marker_line():
    text = (
        "Momentum in NQ has been persistent this week.\n"
        "PROPOSED_HYPOTHESIS: NQ exhibits short-horizon momentum continuation following a >1% overnight gap."
    )
    display, hypothesis = claude_conversation._extract_proposed_hypothesis(text)
    assert hypothesis == "NQ exhibits short-horizon momentum continuation following a >1% overnight gap."
    assert "PROPOSED_HYPOTHESIS" not in display
    assert "Momentum in NQ has been persistent this week." in display


def test_ask_claude_response_carries_the_typed_field(monkeypatch):
    class _Fake:
        def __init__(self, *a, **k):
            pass

        def complete(self, **kwargs):
            from alpha_agent.agents.llm import LLMResponse

            return LLMResponse(
                text="Here is an idea.\nPROPOSED_HYPOTHESIS: CL mean-reverts after a 2-sigma session move.",
                model="claude-sonnet-5-test", stop_reason="end_turn", input_tokens=1, output_tokens=1,
            )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(claude_conversation, "AnthropicClient", _Fake)
    from alpha_agent.ui.conversation_engine import ConversationBoundContext

    response, engine, error = claude_conversation.handle_turn(
        "What should we research next?",
        context=ConversationBoundContext(), has_history=False,
        transcript=[{"type": "user", "text": "What should we research next?"}],
        opportunity_snapshot=None,
    )
    assert error is None
    assert engine == claude_conversation.LIVE_MODE
    assert response.proposed_hypothesis == "CL mean-reverts after a 2-sigma session move."
    assert "PROPOSED_HYPOTHESIS" not in response.text


def test_ask_claude_response_field_is_none_without_the_marker(monkeypatch):
    class _Fake:
        def __init__(self, *a, **k):
            pass

        def complete(self, **kwargs):
            from alpha_agent.agents.llm import LLMResponse

            return LLMResponse(
                text="This candidate is NOT VALIDATED -- see Research Details.",
                model="claude-sonnet-5-test", stop_reason="end_turn", input_tokens=1, output_tokens=1,
            )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(claude_conversation, "AnthropicClient", _Fake)
    from alpha_agent.ui.conversation_engine import ConversationBoundContext

    response, _engine, error = claude_conversation.handle_turn(
        "Why did it fail?",
        context=ConversationBoundContext(), has_history=False,
        transcript=[{"type": "user", "text": "Why did it fail?"}],
        opportunity_snapshot=None,
    )
    assert error is None
    assert response.proposed_hypothesis is None


# ---------------------------------------------------------------------------
# UI -- the action only ever renders from the typed field, real AppTest
# ---------------------------------------------------------------------------


def _turn(*, text: str, proposed_hypothesis: str | None) -> dict:
    return {
        "type": "conversation", "event_id": "agent-event-000000", "at": "2026-09-21 00:00:00 UTC",
        "intent": "GENERAL_QUANT_QUESTION", "text": text, "engine": "live",
        "suggested_action": None, "citations": [], "proposed_hypothesis": proposed_hypothesis,
    }


def test_not_validated_text_alone_never_exposes_the_action():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_SCRIPT)
    at.run(timeout=90)
    at.session_state["agent_transcript"] = [
        _turn(text="This candidate is NOT VALIDATED and was never run.", proposed_hypothesis=None)
    ]
    at.run(timeout=90)
    assert not list(at.exception)
    assert not [b for b in at.button if "Research This Hypothesis" in (b.label or "")]


def test_typed_proposed_hypothesis_exposes_the_action_and_seeds_the_composer():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_SCRIPT)
    at.run(timeout=90)
    hypothesis_text = "CL mean-reverts after a 2-sigma session move."
    at.session_state["agent_transcript"] = [
        _turn(text="Here is an idea.", proposed_hypothesis=hypothesis_text)
    ]
    at.run(timeout=90)
    assert not list(at.exception)
    buttons = [b for b in at.button if "Research This Hypothesis" in (b.label or "")]
    assert len(buttons) == 1

    buttons[0].click().run(timeout=90)
    assert not list(at.exception)
    assert at.session_state["agent-objective"] == hypothesis_text
    assert at.session_state["agent_research_origin"]["source"] == "claude_hypothesis"
