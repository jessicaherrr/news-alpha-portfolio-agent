"""News Alpha Phase D -- asset expression and measurement resolution in a
Research Thread's Market step (real `streamlit.testing.v1.AppTest` render;
migrated from the retired Agent-page triage card), and in the Agent chat's
reply for a described event.

The capability snapshot is pinned (a fresh checkout has no raw ETF bars), so
the counts below are the committed-catalog + acquired-ETF state.
"""
from __future__ import annotations

from datetime import UTC, date, datetime

import alpha_agent.features.compute  # noqa: F401 -- register feature defs
import pytest
import streamlit as st
from alpha_agent.agents.conversation import ConversationSignalContext, classify_intent
from alpha_agent.market_intel.store import EventStore, NewsStore
from alpha_agent.news_alpha import (
    DomainCapabilities,
    FuturesBarCoverage,
    MandateDomain,
    ResearchMandate,
)
from alpha_agent.ui import conversation_engine, market_intel_context, news_alpha_context, services
from alpha_agent.ui.conversation_engine import ConversationBoundContext
from alpha_agent.ui.research_thread import ThreadStep
from news_alpha_thread_support import body, tables, thread_at

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

AI_CAPEX = "Hyperscalers announce a major AI infrastructure spending increase, lifting data center capex guidance"
GEOPOLITICAL = "Missile strikes and a naval blockade raise geopolitical escalation fears"
CAPS = DomainCapabilities(
    etf_market_data_acquired=True, equity_market_data_acquired=False,
    futures_bars=tuple(
        FuturesBarCoverage(root=r, dataset="GLBX.MDP3", data_schema="ohlcv-1m", start=date(2018, 1, 1),
                           end_exclusive=date(2025, 1, 1), continuous_segments=2, roll_overlap_artifacts=28)
        for r in ("CL", "ES", "GC", "NQ", "ZN")
    ),
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    store = NewsStore(tmp_path / "news.sqlite")
    event_store = EventStore(tmp_path / "events.sqlite")
    monkeypatch.setattr(market_intel_context, "_store", lambda: store)
    monkeypatch.setattr(market_intel_context, "_event_store", lambda: event_store)
    monkeypatch.setattr(market_intel_context, "_last_refresh_wall", datetime.now(UTC), raising=False)
    monkeypatch.setattr(news_alpha_context, "_capabilities", lambda: CAPS)
    for key in ("news_alpha_user_events", "news_alpha_mandate"):
        st.session_state.pop(key, None)
    yield
    for key in ("news_alpha_user_events", "news_alpha_mandate"):
        st.session_state.pop(key, None)


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch):
    from alpha_agent.ui import translation_context

    def _boom(*a, **kw):
        raise AssertionError("asset expression must never construct a live LLM client")

    monkeypatch.setattr(translation_context, "AnthropicClient", _boom)


def test_market_step_renders_markets_measurements_and_gaps():
    news_alpha_context.MANDATE_STORE.save(
        ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES, MandateDomain.ETF))
    )
    at = thread_at(AI_CAPEX, ThreadStep.MARKET)
    text = body(at)
    assert "Asset expression: 38 ways to express these consequences (Futures 8 · ETF 15 · Equity 15)" in text
    assert "20 measurable on 2018-2024 history via NQ, XLK, QQQ, XLU" in text
    assert "<b style='color:var(--aa-text)'>4</b> can be measured" in text  # NQ, XLK, QQQ, XLU
    selector = at.button_group(key=f"mx-select-{at.thread_id}")
    assert len(selector.options) == 11  # one per affected market in the setup, measurable ones first
    tab_labels = [t.label for t in at.tabs]
    for label in ("Expressions (38)", "Measurement readiness", "Field resolution (90)"):
        assert label in tab_labels, tab_labels
    values = tables(at)
    for value in ("AI accelerator designers", "Concept only -- no instrument here", "NQ, MNQ", "Available",
                  "Proxy", "Not PIT-safe", "Not executable", "Equity daily bars not acquired", "GLBX.MDP3 · ohlcv-1m · close",
                  "Ecosystem proxy", "One segment of the company", "Holds exposed names", "Direct underlying"):
        assert value in values, value
    assert "Shorting not allowed" in text  # the spenders' cash-flow dip within months
    assert "not an expected return, probability, forecast, trade instruction" in text


def test_an_excluded_domain_stays_visible_but_is_never_enumerated():
    news_alpha_context.MANDATE_STORE.save(
        ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES))
    )
    at = thread_at(AI_CAPEX, ThreadStep.MARKET)
    values = tables(at)
    assert "Excluded by your mandate (15 economically relevant expressions not pursued)." in values
    assert "Outside your setup" in values  # listed under "Other possible exposures"
    assert "XLK" not in values and "XLK" not in body(at)  # never enumerated


def test_an_unseeded_event_shows_no_market():
    at = thread_at(GEOPOLITICAL, ThreadStep.MARKET)
    text = body(at)
    assert "No market expresses these consequences yet" in text
    assert "Asset expression:" not in text


def test_a_chat_described_event_carries_the_same_asset_expression():
    text = f"{AI_CAPEX}. What could I research?"
    classification = classify_intent(text, context=ConversationSignalContext(approved_roots=("ES", "NQ", "CL")))
    response = conversation_engine._handle_observation_to_factor(text, classification, ConversationBoundContext())
    assert "ASSET EXPRESSION: 38 expressions" in response.text
    assert "largest gap: Equity daily bars not acquired (primary blocker for 24 measurements; the only blocker for 16)" in response.text
    evidence = response.evidence["asset_expression"]
    assert evidence["event_id"] == response.evidence["user_event_id"]
    assert evidence["fingerprint"].startswith("assetexpr1:") and evidence["n_measurable_expressions"] == 20
