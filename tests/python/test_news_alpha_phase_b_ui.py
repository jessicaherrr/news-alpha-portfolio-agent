"""News Alpha Phase B -- the economic mechanism graph in a Research Thread's
Reasoning step (real `streamlit.testing.v1.AppTest` render; migrated from the
retired Agent-page triage card).
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from alpha_agent.market_intel.store import EventStore, NewsStore
from alpha_agent.news_alpha import MandateDomain, ResearchMandate
from alpha_agent.ui import market_intel_context, news_alpha_context, services
from alpha_agent.ui.research_thread import ThreadStep
from news_alpha_thread_support import body, tables, thread_at

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

AI_CAPEX = "Hyperscalers announce a major AI infrastructure spending increase, lifting data center capex guidance"


@pytest.fixture(autouse=True)
def _isolated_stores(monkeypatch, tmp_path):
    store = NewsStore(tmp_path / "news.sqlite")
    event_store = EventStore(tmp_path / "events.sqlite")
    monkeypatch.setattr(market_intel_context, "_store", lambda: store)
    monkeypatch.setattr(market_intel_context, "_event_store", lambda: event_store)
    monkeypatch.setattr(market_intel_context, "_last_refresh_wall", datetime.now(UTC), raising=False)
    yield store


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch):
    from alpha_agent.ui import translation_context

    def _boom(*a, **kw):
        raise AssertionError("the mechanism graph must never construct a live LLM client")

    monkeypatch.setattr(translation_context, "AnthropicClient", _boom)


def test_described_ai_event_renders_the_mechanism_graph_in_reasoning():
    at = thread_at(AI_CAPEX, ThreadStep.REASONING)
    text = body(at)
    assert "Mechanism graph: AI infrastructure investment ↑ → AI compute demand ↑" in text
    assert len(at.get("graphviz_chart")) == 1
    assert "Starts from" in text and "AI infrastructure investment (via Technology / AI investment cycle)" in text
    values = tables(at)
    for label in ("HBM demand", "Semiconductor equipment demand", "Power-equipment demand", "⇅ mixed"):
        assert label in values, label
    assert "Verified sources" in text and "International Energy Agency" in text
    assert "Balancing loop" in text
    assert "not evidence" in text and "not tradable assets" in text
    # the economic graph is labelled apart from the Research Map (historical evidence)
    assert "Learn → Research Map" in text


def test_an_equity_only_setup_still_shows_non_tradable_economic_states():
    news_alpha_context.MANDATE_STORE.save(ResearchMandate(allowed_domains=(MandateDomain.EQUITY,)))
    assert "Futures · not in your setup" in body(thread_at(AI_CAPEX, ThreadStep.NEWS))
    values = tables(thread_at(AI_CAPEX, ThreadStep.REASONING))
    for label in ("Electricity demand", "Copper demand", "Natural-gas demand for power"):
        assert label in values, label


def test_an_unseeded_channel_says_so_instead_of_drawing_an_empty_map():
    at = thread_at("Missile strikes and a naval blockade raise geopolitical escalation fears", ThreadStep.REASONING)
    assert "no economic transmission is seeded yet for Geopolitical risk" in body(at)
    assert not at.get("graphviz_chart")
