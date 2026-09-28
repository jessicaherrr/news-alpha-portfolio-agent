"""News Alpha Phase C -- signal paths and the mechanism-adjusted impact in a
Research Thread (News step: both impact passes side by side; Reasoning step:
paths and the mechanism check), and in the Agent chat's reply for a described
event (real `streamlit.testing.v1.AppTest` renders).
"""
from __future__ import annotations

from datetime import UTC, datetime

import alpha_agent.features.compute  # noqa: F401 -- register feature defs
import pytest
import streamlit as st
from alpha_agent.agents.conversation import ConversationSignalContext, classify_intent
from alpha_agent.market_intel.store import EventStore, NewsStore
from alpha_agent.news_alpha import MandateDomain, ResearchMandate
from alpha_agent.ui import conversation_engine, market_intel_context, news_alpha_context, services
from alpha_agent.ui.conversation_engine import ConversationBoundContext
from alpha_agent.ui.research_thread import ThreadStep
from news_alpha_thread_support import body, tables, thread_at

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

AI_CAPEX = "Hyperscalers announce a major AI infrastructure spending increase, lifting data center capex guidance"
GEOPOLITICAL = "Missile strikes and a naval blockade raise geopolitical escalation fears"


@pytest.fixture(autouse=True)
def _isolated_stores(monkeypatch, tmp_path):
    store = NewsStore(tmp_path / "news.sqlite")
    event_store = EventStore(tmp_path / "events.sqlite")
    monkeypatch.setattr(market_intel_context, "_store", lambda: store)
    monkeypatch.setattr(market_intel_context, "_event_store", lambda: event_store)
    monkeypatch.setattr(market_intel_context, "_last_refresh_wall", datetime.now(UTC), raising=False)
    for key in ("news_alpha_user_events", "news_alpha_mandate"):
        st.session_state.pop(key, None)
    yield store
    for key in ("news_alpha_user_events", "news_alpha_mandate"):
        st.session_state.pop(key, None)


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch):
    from alpha_agent.ui import translation_context

    def _boom(*a, **kw):
        raise AssertionError("signal paths must never construct a live LLM client")

    monkeypatch.setattr(translation_context, "AnthropicClient", _boom)


def _setup_three_domains():
    news_alpha_context.MANDATE_STORE.save(
        ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES, MandateDomain.ETF))
    )


def test_news_step_shows_both_impact_passes_side_by_side():
    _setup_three_domains()
    text = body(thread_at(AI_CAPEX, ThreadStep.NEWS))
    assert "Initial impact" in text and "After mechanism graph" in text
    assert "levels kept (economic magnitude not established)" in text and "▲ from" not in text
    assert "Futures 3 of 3 · ETF 2 of 2 · Equity 2 of 2" in text


def test_reasoning_step_renders_signal_paths_and_the_mechanism_check():
    _setup_three_domains()
    at = thread_at(AI_CAPEX, ThreadStep.REASONING)
    text = body(at)
    assert "MECHANISM CHECK (SECOND PASS)" in text
    assert "Signal paths: 6 direct · 7 supply-chain · 7 cross-sector -- 17 researchable." in text
    assert any(b.key and b.key.startswith("follow-sp-") for b in at.button)  # path cards can be followed
    values = tables(at)
    for value in ("HBM demand", "Researchable", "Copper (industrial-metal demand)", "Corroborated", "Copper demand ↑"):
        assert value in values, value
    assert "AI spenders' free cash flow: paths disagree" in text
    assert "Depth is recorded, never ranked" in text
    assert "not tradable assets" in text


def test_an_unseeded_event_shows_no_signal_paths_and_no_second_pass():
    assert "After mechanism graph" not in body(thread_at(GEOPOLITICAL, ThreadStep.NEWS))
    at = thread_at(GEOPOLITICAL, ThreadStep.REASONING)
    text = body(at)
    assert "no economic transmission is seeded yet for Geopolitical risk" in text
    assert "Signal paths:" not in text and "MECHANISM CHECK" not in text
    assert not any(b.key and b.key.startswith("follow-") for b in at.button)


def test_a_chat_described_event_carries_the_same_paths_and_second_pass():
    text = f"{AI_CAPEX}. What could I research?"
    classification = classify_intent(text, context=ConversationSignalContext(approved_roots=("ES", "NQ", "CL")))
    response = conversation_engine._handle_observation_to_factor(text, classification, ConversationBoundContext())
    assert "Signal paths: 6 direct · 7 supply-chain · 7 cross-sector" in response.text
    assert "MECHANISM-ADJUSTED IMPACT (second pass; the initial scan above is kept)" in response.text
    assert "economic magnitude not established" in response.text
    event_id = response.evidence["user_event_id"]
    assert response.evidence["signal_paths"]["event_id"] == event_id
    adjusted = response.evidence["mechanism_adjusted_impact"]
    assert adjusted["event_id"] == event_id and adjusted["assessment_pass"] == "MECHANISM_ADJUSTED"
    assert response.evidence["impact_scan"]["assessment_pass"] == "INITIAL"
