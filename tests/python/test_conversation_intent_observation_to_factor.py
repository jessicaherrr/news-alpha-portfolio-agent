"""Phase 1 -- `ResearchConversationIntent.OBSERVATION_TO_FACTOR` routing
(prompt 1 section 12: "EIA inventories dropped sharply. What could I
research?"). A separate, additive test file rather than editing
`test_conversation_intent_router.py`'s own parametrize table.
"""
from __future__ import annotations

import pytest
from alpha_agent.agents.conversation import (
    ACTION_INTENTS,
    ConversationSignalContext,
    ResearchConversationIntent,
    classify_intent,
)

ROOTS = ("ES", "NQ", "CL", "GC", "ZN")
CTX = ConversationSignalContext(approved_roots=ROOTS)


@pytest.mark.parametrize(
    "text",
    [
        "EIA inventories dropped sharply. What could I research?",
        "What should I research given this FOMC decision?",
        "Turn this into a hypothesis.",
        "Translate this into a factor.",
        "What's the mechanism behind this CPI print?",
        "How does this become a hypothesis?",
    ],
)
def test_observation_to_factor_vocab_routes_correctly(text):
    result = classify_intent(text, context=CTX)
    assert result.intent == ResearchConversationIntent.OBSERVATION_TO_FACTOR


def test_observation_to_factor_is_never_an_action_intent():
    assert ResearchConversationIntent.OBSERVATION_TO_FACTOR not in ACTION_INTENTS


def test_observation_to_factor_does_not_shadow_existing_discovery_vocab():
    """"Find me strategies..." must keep routing to RESEARCH_DISCOVERY --
    adding OBSERVATION_TO_FACTOR must never regress an existing rule."""
    result = classify_intent("Find me strategies that do not depend on trend following.", context=CTX)
    assert result.intent == ResearchConversationIntent.RESEARCH_DISCOVERY


def test_observation_to_factor_does_not_shadow_causal_event_question():
    """"Why did NQ fall today?" must keep routing to the existing unsupported
    causal-question intent -- OBSERVATION_TO_FACTOR is a distinct question
    shape (asking what to RESEARCH, not asking WHY something moved)."""
    result = classify_intent("Why did NQ fall today?", context=CTX)
    assert result.intent == ResearchConversationIntent.UNSUPPORTED_CURRENT_EVENT_CAUSAL_QUESTION


def test_mentioned_roots_still_recognised_on_an_observation_to_factor_question():
    result = classify_intent("CL inventories dropped. What could I research?", context=CTX)
    assert result.intent == ResearchConversationIntent.OBSERVATION_TO_FACTOR
    assert "CL" in result.mentioned_roots
