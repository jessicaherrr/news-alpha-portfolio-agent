"""Tests for `alpha_agent.agents.conversation` -- the typed conversational
intent router (Release UX Part A, task spec sections 1-2/52). Covers the
mission's own example questions plus the structural guarantee that no
question, however phrased, can classify into anything but the closed intent
enum, and that ONLY the three named action intents are ever `is_action=True`.
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
CTX_BOUND = ConversationSignalContext(approved_roots=ROOTS, has_bound_strategy=True)
CTX_HISTORY = ConversationSignalContext(approved_roots=ROOTS, has_history=True)


@pytest.mark.parametrize(
    "text,context,expected",
    [
        ("What is happening in NQ today?", CTX, ResearchConversationIntent.MARKET_QUESTION),
        ("Compare ES and NQ volatility.", CTX, ResearchConversationIntent.MARKET_COMPARISON),
        ("Why did the top mean-reversion strategy fail validation?", CTX,
         ResearchConversationIntent.VALIDATION_EXPLANATION),
        ("Show me when this strategy lost money.", CTX_BOUND, ResearchConversationIntent.STRATEGY_EXPLANATION),
        ("Why is this candidate INCONCLUSIVE instead of PASS?", CTX_BOUND,
         ResearchConversationIntent.VALIDATION_EXPLANATION),
        ("What academic research inspired this?", CTX_BOUND, ResearchConversationIntent.SOURCE_QUESTION),
        ("Is this just another moving-average strategy?", CTX_BOUND,
         ResearchConversationIntent.STRATEGY_EXPLANATION),
        ("How different is this from our previous failed TSMOM?", CTX_BOUND,
         ResearchConversationIntent.CANDIDATE_COMPARISON),
        ("What would happen if I allowed overnight positions?", CTX, ResearchConversationIntent.WHAT_IF_PROFILE),
        ("Find me strategies that do not depend on trend following.", CTX,
         ResearchConversationIntent.RESEARCH_DISCOVERY),
        ("Look for session-based NQ ideas.", CTX, ResearchConversationIntent.RESEARCH_DISCOVERY),
        ("Research whether opening-range effects have academic support.", CTX,
         ResearchConversationIntent.RESEARCH_DISCOVERY),
        ("Compare this strategy with the second-ranked candidate.", CTX_BOUND,
         ResearchConversationIntent.CANDIDATE_COMPARISON),
        ("What should we test next?", CTX, ResearchConversationIntent.RESEARCH_DISCOVERY),
        ("Why did NQ fall today?", CTX, ResearchConversationIntent.UNSUPPORTED_CURRENT_EVENT_CAUSAL_QUESTION),
        ("Why was NQ Mean Reversion rejected?", CTX, ResearchConversationIntent.VALIDATION_EXPLANATION),
        ("Where did this come from?", CTX_BOUND, ResearchConversationIntent.SOURCE_QUESTION),
        ("Run Fast Screen on these candidates.", CTX, ResearchConversationIntent.RUN_FAST_SCREEN),
        ("Please freeze the top candidates.", CTX, ResearchConversationIntent.FREEZE_CANDIDATES),
        ("Run strict validation now.", CTX, ResearchConversationIntent.RUN_STRICT_VALIDATION),
        ("What is the Deflated Sharpe Ratio?", CTX, ResearchConversationIntent.GENERAL_QUANT_QUESTION),
        ("Explain multiple-testing correction.", CTX, ResearchConversationIntent.GENERAL_QUANT_QUESTION),
        ("What if I am willing to tolerate 15% drawdown?", CTX, ResearchConversationIntent.WHAT_IF_PROFILE),
    ],
)
def test_mission_example_questions_route_to_the_expected_intent(text, context, expected):
    result = classify_intent(text, context=context)
    assert result.intent == expected, f"{text!r} -> {result.intent} (rule={result.matched_rule}), expected {expected}"


def test_followup_uses_history_not_bare_discovery_vocab():
    text = "What about another one that avoids overnight risk?"
    with_history = classify_intent(text, context=CTX_HISTORY)
    without_history = classify_intent(text, context=CTX)
    assert with_history.intent == ResearchConversationIntent.RESEARCH_FOLLOWUP
    # Without history the SAME wording is still recognisably a fresh
    # discovery request -- never silently promoted to a follow-up it cannot
    # actually be (there is nothing to follow up on).
    assert without_history.intent != ResearchConversationIntent.RESEARCH_FOLLOWUP


def test_empty_and_whitespace_input_falls_back_safely():
    for text in ("", "   ", "\n\t"):
        result = classify_intent(text, context=CTX)
        assert result.intent == ResearchConversationIntent.GENERAL_QUANT_QUESTION
        assert not result.is_action


def test_only_the_three_named_action_intents_are_ever_actions():
    assert ACTION_INTENTS == {
        ResearchConversationIntent.RUN_FAST_SCREEN,
        ResearchConversationIntent.FREEZE_CANDIDATES,
        ResearchConversationIntent.RUN_STRICT_VALIDATION,
    }
    for intent in ResearchConversationIntent:
        classification = classify_intent(f"placeholder {intent.value}", context=CTX)
        if classification.is_action:
            assert classification.intent in ACTION_INTENTS


def test_router_never_returns_a_callable_or_executes_anything():
    """Structural guarantee (Part A section 4): the router's return value is a
    frozen pydantic model with only label/metadata fields -- there is no
    field, method, or side effect through which classifying a casual
    question could execute a scientific action."""
    result = classify_intent("Run Fast Screen on NQ candidates", context=CTX)
    assert result.is_action is True
    assert result.model_dump().keys() == {"intent", "mentioned_roots", "matched_rule", "is_action"}
    assert not hasattr(result, "execute")
    assert not hasattr(result, "run")


def test_market_question_extracts_mentioned_roots():
    result = classify_intent("How is NQ and ES doing right now?", context=CTX)
    assert set(result.mentioned_roots) >= {"NQ", "ES"}


def test_unknown_root_vocabulary_never_matches():
    result = classify_intent("What is happening in BTC today?", context=CTX)
    assert "BTC" not in result.mentioned_roots


def test_causal_current_event_question_is_distinct_from_plain_market_question():
    plain = classify_intent("What is the NQ price right now?", context=CTX)
    causal = classify_intent("Why did NQ crash today?", context=CTX)
    assert plain.intent == ResearchConversationIntent.MARKET_QUESTION
    assert causal.intent == ResearchConversationIntent.UNSUPPORTED_CURRENT_EVENT_CAUSAL_QUESTION


def test_validation_explanation_beats_strategy_explanation_when_both_vocab_present():
    result = classify_intent("Why did this strategy get REJECTED?", context=CTX_BOUND)
    assert result.intent == ResearchConversationIntent.VALIDATION_EXPLANATION
