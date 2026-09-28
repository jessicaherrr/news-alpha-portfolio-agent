"""Tests for `alpha_agent.ui.claude_conversation` (CLAUDE LIVE INTEGRATION V0).

Never makes a real network call: the `AnthropicClient` symbol this module
uses is monkeypatched to a small fake that records what it was asked and
returns a canned `LLMResponse` -- the same "swap the one call site"
technique the rest of this codebase already uses for `AnthropicClient`
(see `test_phase_16_research_agent.py`).
"""
from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alpha_agent.agents.conversation import ACTION_INTENTS
from alpha_agent.agents.llm import LLMResponse
from alpha_agent.opportunity.schemas import OpportunitySnapshot, OpportunityState
from alpha_agent.ui import claude_conversation
from alpha_agent.ui.conversation_engine import ConversationBoundContext

REPO_ROOT = Path(__file__).resolve().parents[2]


def _snap(product: str, *, verdict: str | None = None) -> OpportunitySnapshot:
    return OpportunitySnapshot(
        product=product, display_name=product, setup=f"{product} test setup",
        why_now=(f"{product} reason",), risks=(f"{product} risk",), catalyst=f"{product} catalyst",
        strengthen_if=("x",), invalidate_if=("y",),
        opportunity_state=OpportunityState.INVESTIGATE, next_action=f"do X for {product}",
        observed_at=datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC),
        evidence_refs=(f"research_verdict={verdict}",) if verdict else (),
    )


def _snapshot(*products: str) -> dict:
    return {
        "opportunities": [_snap(p) for p in products],
        "observed_at": datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC),
    }


class _FakeAnthropicClient:
    """Records the exact kwargs `_ask_claude` calls `.complete()` with."""

    last_kwargs: dict | None = None
    response_text: str = "Claude says hello."
    raise_exc: Exception | None = None

    def __init__(self, *args, **kwargs):
        pass

    def complete(self, **kwargs):
        _FakeAnthropicClient.last_kwargs = kwargs
        if _FakeAnthropicClient.raise_exc is not None:
            raise _FakeAnthropicClient.raise_exc
        return LLMResponse(
            text=_FakeAnthropicClient.response_text, model="claude-sonnet-5-test",
            stop_reason="end_turn", input_tokens=42, output_tokens=7,
        )


@pytest.fixture(autouse=True)
def _reset_fake():
    _FakeAnthropicClient.last_kwargs = None
    _FakeAnthropicClient.response_text = "Claude says hello."
    _FakeAnthropicClient.raise_exc = None
    yield


# ---------------------------------------------------------------------------
# structural guarantee: never a Databento refresh just because Claude was asked
# ---------------------------------------------------------------------------


def test_claude_conversation_never_imports_databento_context():
    source = (REPO_ROOT / "python" / "alpha_agent" / "ui" / "claude_conversation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any(name.startswith("alpha_agent.ui.databento_context") for name in imported)


# ---------------------------------------------------------------------------
# connectivity check
# ---------------------------------------------------------------------------


def test_is_connected_reflects_env_var_presence_only(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert claude_conversation.is_connected() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    assert claude_conversation.is_connected() is True


# ---------------------------------------------------------------------------
# dispatch: fallback / action-gating / live path / error path
# ---------------------------------------------------------------------------


def test_falls_back_to_deterministic_when_not_connected(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(claude_conversation, "AnthropicClient", _FakeAnthropicClient)
    response, engine, error = claude_conversation.handle_turn(
        "What is DSR?",
        context=ConversationBoundContext(),
        has_history=False,
        transcript=[{"type": "user", "text": "What is DSR?"}],
        opportunity_snapshot=None,
    )
    assert error is None
    assert engine == claude_conversation.SCRIPTED_MODE
    assert response is not None
    assert _FakeAnthropicClient.last_kwargs is None  # Claude never called


@pytest.mark.parametrize("intent_text", ["run fast screen", "freeze the candidates", "run strict validation"])
def test_action_intents_never_reach_claude_even_when_connected(monkeypatch, intent_text):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(claude_conversation, "AnthropicClient", _FakeAnthropicClient)
    response, engine, error = claude_conversation.handle_turn(
        intent_text,
        context=ConversationBoundContext(),
        has_history=False,
        transcript=[{"type": "user", "text": intent_text}],
        opportunity_snapshot=None,
    )
    assert error is None
    assert engine == claude_conversation.SCRIPTED_MODE
    assert response.intent in ACTION_INTENTS
    assert _FakeAnthropicClient.last_kwargs is None


def test_connected_general_question_calls_claude_with_grounded_context(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(claude_conversation, "AnthropicClient", _FakeAnthropicClient)
    _FakeAnthropicClient.response_text = "Here is a grounded answer."

    response, engine, error = claude_conversation.handle_turn(
        "What's interesting right now?",
        context=ConversationBoundContext(root="NQ"),
        has_history=False,
        transcript=[{"type": "user", "text": "What's interesting right now?"}],
        opportunity_snapshot=None,
    )
    assert error is None
    assert engine == claude_conversation.LIVE_MODE
    assert response.text == "Here is a grounded answer."
    assert response.evidence["engine"] == "claude"

    kwargs = _FakeAnthropicClient.last_kwargs
    assert kwargs is not None
    system = kwargs["system"]
    assert "CURRENT OBSERVATION" in system
    assert "RESEARCH EVIDENCE" in system
    assert "USER FIT" in system
    # no Opportunity snapshot was passed in -- must say so honestly, never guess
    assert "No current Opportunity evidence is loaded" in system
    assert kwargs["messages"][-1] == {"role": "user", "content": "What's interesting right now?"}
    assert kwargs["model"] == claude_conversation.DEFAULT_MODEL


def test_live_call_failure_is_an_honest_error_never_a_fabricated_answer(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(claude_conversation, "AnthropicClient", _FakeAnthropicClient)
    _FakeAnthropicClient.raise_exc = RuntimeError("simulated network failure")

    response, engine, error = claude_conversation.handle_turn(
        "Why is CL worth watching?",
        context=ConversationBoundContext(root="CL"),
        has_history=False,
        transcript=[{"type": "user", "text": "Why is CL worth watching?"}],
        opportunity_snapshot=None,
    )
    assert response is None
    assert engine == claude_conversation.LIVE_MODE
    assert error is not None
    assert "sk-test-not-a-real-key" not in error


# ---------------------------------------------------------------------------
# message history: compact, chat-only, excludes composer turns
# ---------------------------------------------------------------------------


def test_message_history_is_bounded_and_excludes_composer_turns():
    transcript = [
        {"type": "user", "text": "q1"},
        {"type": "conversation", "text": "a1"},
        {"type": "hypothesis", "data": {}},  # composer turn -- must be excluded
        {"type": "user", "text": "q2"},
        {"type": "conversation", "text": "a2"},
        {"type": "user", "text": "q3"},
    ]
    messages = claude_conversation._build_messages(transcript)
    assert [m["content"] for m in messages] == ["q1", "a1", "q2", "a2", "q3"]
    assert messages[-1] == {"role": "user", "content": "q3"}


# ---------------------------------------------------------------------------
# question-scoped context (Agent Experience Consolidation campaign, section
# 35): a page's LOCAL selection must never narrow/replace a question's own
# evidence scope -- `context` below never carries a `root` unless the test
# is explicitly checking the RESEARCH_OBJECT-scoped path.
# ---------------------------------------------------------------------------


def _ask(monkeypatch, text: str, *, opportunity_snapshot: dict | None, context: ConversationBoundContext | None = None):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(claude_conversation, "AnthropicClient", _FakeAnthropicClient)
    claude_conversation.handle_turn(
        text, context=context or ConversationBoundContext(), has_history=False,
        transcript=[{"type": "user", "text": text}], opportunity_snapshot=opportunity_snapshot,
    )
    return _FakeAnthropicClient.last_kwargs["system"]


def test_A_global_question_sees_every_cached_root(monkeypatch):
    system = _ask(monkeypatch, "What's interesting right now?", opportunity_snapshot=_snapshot("ES", "NQ", "CL"))
    assert "- ES (" in system
    assert "- NQ (" in system
    assert "- CL (" in system


def test_B_single_market_question_sees_only_that_root(monkeypatch):
    system = _ask(monkeypatch, "Why is CL interesting?", opportunity_snapshot=_snapshot("ES", "NQ", "CL"))
    assert "- CL (" in system
    assert "- NQ (" not in system
    assert "- ES (" not in system


def test_C_why_x_instead_of_y_sees_both_roots(monkeypatch):
    system = _ask(monkeypatch, "Why CL instead of NQ?", opportunity_snapshot=_snapshot("ES", "NQ", "CL"))
    assert "- CL (" in system
    assert "- NQ (" in system


def test_D_compare_two_roots_sees_both(monkeypatch):
    system = _ask(monkeypatch, "Compare ES and NQ.", opportunity_snapshot=_snapshot("ES", "NQ", "CL"))
    assert "- ES (" in system
    assert "- NQ (" in system


def test_E_global_question_ignores_a_market_pages_local_selection(monkeypatch):
    """`selected_root` (the sidebar's Workspace Market, a Market-page-local
    concept) is never even passed to `handle_turn` any more -- this test
    proves a global question is unaffected regardless of what any other page
    has locally selected, since `context` never carries that information at
    all."""
    system = _ask(
        monkeypatch, "What's interesting right now?",
        opportunity_snapshot=_snapshot("ES", "NQ", "CL"),
        context=ConversationBoundContext(),  # no bound root, mirrors a fresh Agent session
    )
    for root in ("ES", "NQ", "CL"):
        assert f"- {root} (" in system


def test_F_global_question_ignores_a_bound_research_root(monkeypatch):
    """Even when a research thread IS bound to one root (e.g. the user
    generated a hypothesis for NQ earlier), a fresh GLOBAL question must
    still see every cached root -- only a RESEARCH_OBJECT-shaped question
    (see the intent-scoped tests below) narrows to the bound root."""
    system = _ask(
        monkeypatch, "What's interesting right now?",
        opportunity_snapshot=_snapshot("ES", "NQ", "CL"),
        context=ConversationBoundContext(root="NQ"),
    )
    for root in ("ES", "NQ", "CL"):
        assert f"- {root} (" in system


def test_research_object_question_scopes_to_the_bound_root_only(monkeypatch):
    """A question about the bound research object itself ("why did it
    fail?") -- no market symbol named -- scopes CURRENT OBSERVATION to that
    object's own root, not a market-wide dump."""
    system = _ask(
        monkeypatch, "Why did it fail?",
        opportunity_snapshot=_snapshot("ES", "NQ", "CL"),
        context=ConversationBoundContext(root="NQ", compiled={"strategy_fingerprint": "x"}),
    )
    assert "- NQ (" in system
    assert "- ES (" not in system
    assert "- CL (" not in system


# ---------------------------------------------------------------------------
# no fabricated absence (section 36): a root genuinely missing from the
# cached snapshot must be reported as "not available in this snapshot",
# never as a claim that it failed some threshold.
# ---------------------------------------------------------------------------


def test_grounded_context_has_five_labelled_evidence_sections(monkeypatch):
    system = _ask(monkeypatch, "Why CL instead of NQ?", opportunity_snapshot=_snapshot("CL", "NQ"))
    assert "CURRENT OBSERVATION" in system
    assert "CATALYSTS / NEWS" in system
    assert "RESEARCH EVIDENCE" in system
    assert "USER FIT" in system
    assert "EVIDENCE GAPS" in system


def test_no_fabricated_absence_for_a_root_missing_from_the_snapshot(monkeypatch):
    system = _ask(monkeypatch, "Why CL instead of GC?", opportunity_snapshot=_snapshot("ES", "NQ", "CL"))
    assert "- CL (" in system
    # GC is genuinely absent from the cached snapshot -- reported as absence
    # from the scan, never as a claim it failed some threshold.
    assert "GC: current evidence is not available in this snapshot" in system
    assert "did not meet" not in system
    assert "GC failed" not in system
    assert "GC did not" not in system


# ---------------------------------------------------------------------------
# Agent Evidence Pack acceptance pass (task spec sections 4/11/12/15/16):
# CATALYSTS / NEWS is cache-only (never auto-fetched from a conversation),
# AVAILABLE TESTING is bounded and authoritative-only, and EVIDENCE GAPS
# names what is missing rather than letting Claude guess around it.
# ---------------------------------------------------------------------------


def test_catalysts_news_says_not_loaded_before_any_refresh_ever_ran(monkeypatch):
    """Default test isolation (`conftest.py`'s autouse market-intel fixture)
    never lets `market_intel_context._refresh_if_stale` run for real, so
    `last_refresh_at()` is honestly `None` here -- this also proves the
    grounded-context builder never itself calls a refreshing accessor."""
    system = _ask(monkeypatch, "Why CL instead of NQ?", opportunity_snapshot=_snapshot("CL", "NQ"))
    assert "Current news/event evidence is not loaded" in system


def test_catalysts_news_shows_cached_items_once_a_refresh_has_run(monkeypatch):
    from alpha_agent.market_intel.event_schemas import EventImportance, ScheduledMarketEvent
    from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
    from alpha_agent.ui import market_intel_context

    news_item = MarketNewsItem(
        news_id="n1", headline="EIA reports a large crude draw", source_name="EIA",
        source_type=NewsSourceType.OFFICIAL, source_url="https://example.gov/n1",
        published_at=datetime(2026, 9, 21, 10, 0, tzinfo=UTC),
        retrieved_at=datetime(2026, 9, 21, 10, 5, tzinfo=UTC),
        related_products=("CL",), category=NewsCategory.PETROLEUM, mapping_reason="mentions CL inventories",
    )
    event = ScheduledMarketEvent(
        event_id="e1", name="EIA Petroleum Status Report", source_name="EIA", source_url="https://example.gov/e1",
        scheduled_at=datetime(2026, 9, 24, 14, 30, tzinfo=UTC), timezone="UTC",
        category="PETROLEUM", importance=EventImportance.HIGH, importance_rule="weekly release",
        affected_products=("CL",), mapping_reason="weekly inventory report",
        retrieved_at=datetime(2026, 9, 21, 10, 5, tzinfo=UTC),
    )
    monkeypatch.setattr(market_intel_context, "last_refresh_at", lambda: datetime(2026, 9, 21, 10, 5, tzinfo=UTC))
    monkeypatch.setattr(
        market_intel_context, "cached_recent_news",
        lambda **kw: (news_item,) if kw.get("related_product") == "CL" else (),
    )
    monkeypatch.setattr(market_intel_context, "cached_upcoming_events", lambda **kw: (event,))

    system = _ask(monkeypatch, "Why CL instead of NQ?", opportunity_snapshot=_snapshot("CL", "NQ"))
    assert "EIA reports a large crude draw" in system
    assert "EIA Petroleum Status Report" in system


def test_catalysts_news_never_calls_a_refreshing_accessor(monkeypatch):
    """Structural guarantee (task spec section 12: "NO AUTO NETWORK FROM
    CHAT") -- even when a refresh HAS run, asking a question must never call
    `recent_news`/`upcoming_events` (the refreshing accessors), only the
    `cached_*` read-only ones."""
    from alpha_agent.ui import market_intel_context

    def _boom(**kw):
        raise AssertionError("a conversational question must never call a refreshing news/event accessor")

    monkeypatch.setattr(market_intel_context, "last_refresh_at", lambda: datetime(2026, 9, 21, 10, 5, tzinfo=UTC))
    monkeypatch.setattr(market_intel_context, "recent_news", _boom)
    monkeypatch.setattr(market_intel_context, "upcoming_events", _boom)
    monkeypatch.setattr(market_intel_context, "cached_recent_news", lambda **kw: ())
    monkeypatch.setattr(market_intel_context, "cached_upcoming_events", lambda **kw: ())

    _ask(monkeypatch, "Why CL instead of NQ?", opportunity_snapshot=_snapshot("CL", "NQ"))  # must not raise


def test_available_testing_reports_no_experiment_when_none_exists(monkeypatch):
    from alpha_agent.ui import services

    monkeypatch.setattr(services, "research_candidate_summaries", lambda *a, **kw: [])
    system = _ask(monkeypatch, "Why CL instead of NQ?", opportunity_snapshot=_snapshot("CL", "NQ"))
    assert "Available testing for CL: no authoritative registry experiment exists yet" in system
    assert "Available testing for NQ: no authoritative registry experiment exists yet" in system
    assert "$" not in system  # no fabricated PnL anywhere in the payload


def test_available_testing_reports_the_real_authoritative_result_only(monkeypatch):
    import types

    from alpha_agent.ui import services

    fake = types.SimpleNamespace(
        root_symbol="CL", scientific_verdict="PASS", research_promise_score=0.9,
        net_pnl_usd=12345.0, annualized_sharpe=1.23, trade_count=88,
        friendly_strategy_name="Time-Series Momentum", experiment_id="EXP-CL-1",
    )
    monkeypatch.setattr(services, "research_candidate_summaries", lambda *a, **kw: [fake])
    system = _ask(monkeypatch, "Why CL instead of NQ?", opportunity_snapshot=_snapshot("CL", "NQ"))
    assert "Time-Series Momentum" in system
    assert "EXP-CL-1" in system
    assert "verdict PASS" in system
    assert "$12,345" in system


def test_evidence_gaps_lists_missing_registry_and_news_and_the_holdout_statement(monkeypatch):
    from alpha_agent.ui import services

    monkeypatch.setattr(services, "research_candidate_summaries", lambda *a, **kw: [])
    system = _ask(monkeypatch, "Why CL instead of NQ?", opportunity_snapshot=_snapshot("CL", "NQ"))
    gaps = system.split("EVIDENCE GAPS")[-1]
    assert "No committed, authoritative registry experiment exists yet for CL." in gaps
    assert "No committed, authoritative registry experiment exists yet for NQ." in gaps
    assert "No news/event refresh has run this session." in gaps
    assert "2025 is a sealed research holdout" in gaps


# ---------------------------------------------------------------------------
# backtest authority (task spec section 16): only an authoritative,
# non-superseded record may ever be presented as clean registry evidence.
# `services.research_candidate_summaries` is already `list_experiments(
# include_superseded=False)`-backed, so an empty result here is the correct,
# honest representation of "nothing authoritative" -- never a fallback to a
# raw/superseded row.
# ---------------------------------------------------------------------------


def test_backtest_authority_never_presents_a_result_when_the_authoritative_source_is_empty(monkeypatch):
    """When the authoritative candidate source (`services.
    research_candidate_summaries`) has nothing for this root, the "Available
    testing" line must say so plainly -- it must never fabricate a verdict of
    its own. This is scoped to that ONE line, not the whole evidence payload:
    a SEPARATE, already-existing, genuinely authoritative source (the
    per-root failure-memory verdict-count sweep) may legitimately mention
    REJECT/INCONCLUSIVE elsewhere -- that is real evidence, not fabrication,
    and this test must not conflate the two sources."""
    from alpha_agent.ui import services

    monkeypatch.setattr(services, "research_candidate_summaries", lambda *a, **kw: [])
    system = _ask(monkeypatch, "Why is NQ interesting?", opportunity_snapshot=_snapshot("NQ"))
    grounded = system.split("PROJECT EVIDENCE FOR THIS TURN")[-1]
    available_testing_line = next(
        line for line in grounded.splitlines() if line.startswith("Available testing for NQ")
    )
    assert available_testing_line == (
        "Available testing for NQ: no authoritative registry experiment exists yet -- this would be a "
        "genuinely new test. Suggest Research This / Generate Hypothesis."
    )
