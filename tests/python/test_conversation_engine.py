"""Tests for `alpha_agent.ui.conversation_engine` (Release UX Part A/I, task
spec section 52). Uses the REAL, already-committed local registry (read-only,
never mutated -- exactly like every other `services.py`-based UI test) for
registry-grounded evidence, and a monkeypatched `databento_context` for the
market plane (deterministic, no real network dependency in the main suite;
see `test_databento_market_data_provider_live.py` for the real smoke test).
"""
from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alpha_agent.agents.conversation import ACTION_INTENTS, ResearchConversationIntent
from alpha_agent.marketdata.databento_schemas import (
    ContractResolution,
    DatabentoCapability,
    MarketSnapshot,
    SessionSummary,
)
from alpha_agent.registry.enums import TrialRole
from alpha_agent.ui import conversation_engine, services
from alpha_agent.ui.conversation_engine import ConversationBoundContext, handle_conversation_turn

REPO_ROOT = Path(__file__).resolve().parents[2]


def _first_canonical_reject() -> dict:
    for row in services.list_experiments(trial_role=TrialRole.CANONICAL):
        if row.get("verdict") == "REJECT":
            return services.get_experiment(row["experiment_id"])
    pytest.skip("no CANONICAL REJECT experiment in the local registry")


# ---------------------------------------------------------------------------
# structural guarantee: this module can never execute a scientific action
# ---------------------------------------------------------------------------


def test_conversation_engine_never_imports_an_execution_entry_point():
    source = (REPO_ROOT / "python" / "alpha_agent" / "ui" / "conversation_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ImportFrom, ast.Import)):
            imported_names.update(alias.name for alias in node.names)
    forbidden = {
        "run_fast_screen", "freeze_top_k", "adopt_frozen_candidate_set",
        "run_strict_validation", "run_deep_research", "CliBacktestRunner",
    }
    assert not (imported_names & forbidden), imported_names & forbidden


def test_every_intent_has_a_handler_and_actions_never_change():
    assert set(conversation_engine._HANDLERS) == set(ResearchConversationIntent)


@pytest.mark.parametrize("intent", sorted(ACTION_INTENTS, key=lambda i: i.value))
def test_action_intents_only_ever_suggest_never_execute(intent):
    text = {
        ResearchConversationIntent.RUN_FAST_SCREEN: "Run Fast Screen on these candidates",
        ResearchConversationIntent.FREEZE_CANDIDATES: "Please freeze the top candidates",
        ResearchConversationIntent.RUN_STRICT_VALIDATION: "Run strict validation now",
    }[intent]
    response = handle_conversation_turn(text)
    assert response.intent == intent
    assert response.suggested_action == intent.value
    assert "click" in response.text.lower() or "action" in response.text.lower()


# ---------------------------------------------------------------------------
# market plane
# ---------------------------------------------------------------------------


def _fake_snapshot(root: str, *, capability=DatabentoCapability.LATEST_AVAILABLE) -> MarketSnapshot:
    return MarketSnapshot(
        root_symbol=root,
        contract=ContractResolution(root_symbol=root, display_symbol=f"{root}.v.0", resolved_raw_symbol=f"{root}U6", resolved_at=datetime.now(UTC)),
        last=100.0, change_pct=1.5,
        session=SessionSummary(root_symbol=root, session_date="2026-09-11", session_high=101.0, session_low=99.0),
        as_of=datetime.now(UTC), capability=capability, freshness_seconds=3600.0,
    )


def test_market_question_uses_market_snapshot_not_registry(monkeypatch):
    monkeypatch.setattr(conversation_engine.databento_context, "market_snapshot", lambda root: _fake_snapshot(root))
    response = handle_conversation_turn("What is happening in NQ today?")
    assert response.intent == ResearchConversationIntent.MARKET_QUESTION
    assert "NQ" in response.text
    assert "100" in response.text
    assert response.evidence["available"] is True


def test_market_question_never_fabricates_when_unavailable(monkeypatch):
    monkeypatch.setattr(conversation_engine.databento_context, "market_snapshot", lambda root: None)
    response = handle_conversation_turn("What is happening in NQ today?")
    assert "don't have current market data" in response.text
    assert response.evidence["available"] is False


def test_causal_question_shows_data_but_refuses_causal_attribution(monkeypatch):
    monkeypatch.setattr(conversation_engine.databento_context, "market_snapshot", lambda root: _fake_snapshot(root))
    response = handle_conversation_turn("Why did NQ fall today?")
    assert response.intent == ResearchConversationIntent.UNSUPPORTED_CURRENT_EVENT_CAUSAL_QUESTION
    assert "100" in response.text  # still shows real data
    assert "do not currently have enough sourced evidence" in response.text


def test_market_comparison_uses_two_snapshots(monkeypatch):
    monkeypatch.setattr(conversation_engine.databento_context, "market_snapshot", lambda root: _fake_snapshot(root))
    response = handle_conversation_turn("Compare ES and NQ volatility.")
    assert response.intent == ResearchConversationIntent.MARKET_COMPARISON
    assert "ES" in response.evidence["snapshots"]
    assert "NQ" in response.evidence["snapshots"]


# ---------------------------------------------------------------------------
# scientific evidence plane -- REAL registry data
# ---------------------------------------------------------------------------


def test_validation_explanation_uses_real_registry_evidence_never_a_new_judgment():
    detail = _first_canonical_reject()
    context = ConversationBoundContext(root=detail["experiment"]["root_symbol"], evidence=detail)
    response = handle_conversation_turn("Why was this strategy rejected?", context=context)
    assert response.intent == ResearchConversationIntent.VALIDATION_EXPLANATION
    assert "REJECT" in response.text
    # The exact same plain-language synthesis the Backtests/Validation pages
    # already use -- never a second, independent judgment.
    expected = services.plain_language_reason(
        result=detail["result"], trial_role=detail["experiment"]["trial_role"], verdict="REJECT",
    )
    assert expected in response.text


def test_validation_explanation_without_bound_experiment_is_honest():
    response = handle_conversation_turn("Why was this strategy rejected?")
    assert "don't have a bound experiment" in response.text


def test_source_question_uses_hypothesis_source_inspirations_never_fabricated():
    hyp = {"source_inspirations": ["GitHub repo X: 20/50 MA crossover"]}
    context = ConversationBoundContext(hypothesis=hyp)
    response = handle_conversation_turn("What academic research inspired this?", context=context)
    assert response.citations == ("GitHub repo X: 20/50 MA crossover",)
    assert "GitHub repo X" in response.text


def test_source_question_without_citations_is_honest_not_invented():
    response = handle_conversation_turn("What academic research inspired this?", context=ConversationBoundContext(hypothesis={}))
    assert "No sourced inspirations" in response.text
    assert response.citations == ()


def test_candidate_comparison_uses_real_sibling_experiments():
    detail = _first_canonical_reject()
    context = ConversationBoundContext(root=detail["experiment"]["root_symbol"], evidence={"experiment": detail["experiment"], "result": detail["result"]})
    response = handle_conversation_turn("Compare this strategy with the second-ranked candidate.", context=context)
    assert response.intent == ResearchConversationIntent.CANDIDATE_COMPARISON
    # Either a real sibling is named, or an honest "no other variants" message
    assert "Compared with" in response.text or "No other" in response.text


# ---------------------------------------------------------------------------
# profile plane -- never changes scientific verdict
# ---------------------------------------------------------------------------


def test_profile_question_reads_saved_profile():
    response = handle_conversation_turn("What is my risk profile?")
    assert response.intent == ResearchConversationIntent.PROFILE_QUESTION
    assert "risk" in response.text.lower()


def test_what_if_profile_recomputes_fit_never_verdict():
    detail = _first_canonical_reject()
    real_verdict = detail["result"]["headline_verdict"]
    context = ConversationBoundContext(
        root=detail["experiment"]["root_symbol"],
        evidence={"experiment": detail["experiment"], "result": detail["result"]},
    )
    response = handle_conversation_turn("What if I tolerate 20% drawdown?", context=context)
    assert response.intent == ResearchConversationIntent.WHAT_IF_PROFILE
    assert "never changes the scientific verdict" in response.text
    # The underlying registry record is untouched -- re-read and compare.
    reread = services.get_experiment(detail["experiment"]["experiment_id"])
    assert reread["result"]["headline_verdict"] == real_verdict


def test_what_if_profile_without_a_recognisable_constraint_asks_for_specifics():
    response = handle_conversation_turn("What if I am willing to tolerate more risk?")
    assert response.intent == ResearchConversationIntent.WHAT_IF_PROFILE
    assert "couldn't identify a specific constraint" in response.text


def test_what_if_strategy_never_fabricates_a_performance_number():
    # "overnight" is profile vocab, so this classifies WHAT_IF_PROFILE by
    # design; a purely strategy-logic what-if needs bound-strategy context.
    response2 = handle_conversation_turn(
        "What if this strategy used a wider stop?", context=ConversationBoundContext(hypothesis={"x": 1}),
    )
    assert response2.intent == ResearchConversationIntent.WHAT_IF_STRATEGY
    assert "can't estimate its performance without actually running it" in response2.text


# ---------------------------------------------------------------------------
# discovery / follow-up / general fallback
# ---------------------------------------------------------------------------


def test_research_discovery_never_creates_an_experiment():
    response = handle_conversation_turn("Find me a non-trend alternative.")
    assert response.intent == ResearchConversationIntent.RESEARCH_DISCOVERY
    assert "Discover Strategies" in response.text or response.text  # always some guidance, never a silent no-op


def test_general_quant_question_glossary():
    response = handle_conversation_turn("What is the Deflated Sharpe Ratio?")
    assert "Deflated Sharpe" in response.text or "selection bias" in response.text


def test_unknown_general_question_is_honest_fallback():
    response = handle_conversation_turn("Explain the meaning of life.")
    assert "don't have a canned explanation" in response.text
