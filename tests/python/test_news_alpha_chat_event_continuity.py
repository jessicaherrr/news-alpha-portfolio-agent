"""News Alpha chat integration -- a chat message that DESCRIBES an event goes
through the Phase A `UserDescribedEvent -> scan_initial_impact` path (the same
one the "Describe a market event" box uses), and that SAME scan feeds the
Phase B mechanism graph and the existing translation. A message about cached
news keeps the Phase 1 cached-item path unchanged.

Regression target: before this fix, "OPEC+ cut output -- what could I
research?" was answered by translating whatever unrelated news item happened
to be cached first.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import alpha_agent.features.compute  # noqa: F401 -- register feature defs
import pytest
import streamlit as st
from alpha_agent.agents.conversation import (
    ConversationSignalContext,
    ResearchConversationIntent,
    classify_intent,
)
from alpha_agent.market_intel import mapping
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.market_intel.store import EventStore, NewsStore
from alpha_agent.news_alpha import build_economic_mechanism_graph, scan_initial_impact
from alpha_agent.ui import conversation_engine, market_intel_context, news_alpha_context, services
from alpha_agent.ui.conversation_engine import ConversationBoundContext

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

_NOW = datetime.now(UTC)
_CTX = ConversationBoundContext()
_SIG_CTX = ConversationSignalContext(approved_roots=("ES", "NQ", "CL", "GC", "ZN"))
OPEC_CHAT = "OPEC+ agreed a production cut of 1 million barrels per day. What could I research?"
OPEC_EVENT = "OPEC+ agreed a production cut of 1 million barrels per day."


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    store = NewsStore(tmp_path / "news.sqlite")
    monkeypatch.setattr(market_intel_context, "_store", lambda: store)
    monkeypatch.setattr(market_intel_context, "_event_store", lambda: EventStore(tmp_path / "events.sqlite"))
    # Bare-mode st.session_state persists across tests in one process.
    for key in ("news_alpha_user_events", "news_alpha_mandate"):
        st.session_state.pop(key, None)
    yield store
    for key in ("news_alpha_user_events", "news_alpha_mandate"):
        st.session_state.pop(key, None)


def _seed_fomc(store: NewsStore) -> None:
    store.add_item(
        MarketNewsItem(
            news_id="CACHED-FOMC", headline="Federal Reserve issues FOMC statement", source_name="Federal Reserve",
            source_type=NewsSourceType.OFFICIAL, source_url="https://federalreserve.gov/cached-fomc",
            published_at=_NOW - timedelta(hours=1), retrieved_at=_NOW,
            related_products=mapping.products_for_category(NewsCategory.FOMC_POLICY),
            related_asset_classes=mapping.asset_classes_for_category(NewsCategory.FOMC_POLICY),
            category=NewsCategory.FOMC_POLICY, mapping_reason=mapping.mapping_reason_for_category(NewsCategory.FOMC_POLICY),
        )
    )


def _ask(text: str):
    classification = classify_intent(text, context=_SIG_CTX)
    assert classification.intent is ResearchConversationIntent.OBSERVATION_TO_FACTOR
    return conversation_engine._handle_observation_to_factor(text, classification, _CTX)


def test_described_event_is_scanned_not_an_unrelated_cached_item(_isolated):
    _seed_fomc(_isolated)  # an unrelated cached item that the old handler would have translated
    response = _ask(OPEC_CHAT)
    scan = response.evidence["impact_scan"]
    assert scan["event"]["kind"] == "USER_DESCRIBED"
    assert scan["event"]["headline"] == OPEC_EVENT  # trailing research question stripped
    translation = response.evidence["observation_translation"]
    assert translation["observation"]["event_type"] == "USER_DESCRIBED_EVENT"
    assert translation["observation"]["root_symbol"] == "CL"
    assert "CACHED-FOMC" not in str(response.evidence)
    assert "IMPACT TRIAGE" in response.text and "Futures" in response.text


def test_chat_event_triage_card_and_mechanism_graph_share_one_identity_and_scan():
    response = _ask(OPEC_CHAT)
    event_id = response.evidence["user_event_id"]
    mandate = news_alpha_context.current_mandate()

    card_scans = news_alpha_context.user_event_scans(mandate)  # what the triage section renders
    assert [s.event.event_id for s in card_scans] == [event_id]
    assert card_scans[0].model_dump(mode="json") == response.evidence["impact_scan"]

    graph = build_economic_mechanism_graph(card_scans[0])
    assert response.evidence["mechanism_graph"] == {"event_id": event_id, "fingerprint": graph.fingerprint()}
    assert response.evidence["translation_handoff_key"] == f"user:{event_id}:CL"


def test_repeating_the_event_keeps_one_event_identity():
    first = _ask(OPEC_CHAT).evidence["user_event_id"]
    second = _ask(OPEC_CHAT).evidence["user_event_id"]
    assert first == second
    assert len(st.session_state["news_alpha_user_events"]) == 1


def test_event_typed_in_the_describe_box_then_asked_in_chat_is_the_same_event():
    boxed = news_alpha_context.add_user_event(OPEC_EVENT)
    assert _ask(OPEC_CHAT).evidence["user_event_id"] == boxed.event_id


def test_phase_a_scan_semantics_are_unchanged_by_the_chat_route():
    response = _ask(OPEC_CHAT)
    event = st.session_state["news_alpha_user_events"][0]
    mandate = news_alpha_context.current_mandate()
    direct = scan_initial_impact(event, mandate, universe=news_alpha_context.allowed_universe(mandate))
    assert direct.model_dump(mode="json") == response.evidence["impact_scan"]


def test_cached_news_request_keeps_the_phase_1_cached_item_path(_isolated):
    _seed_fomc(_isolated)
    response = _ask("What could I research from the latest FOMC news?")
    assert "impact_scan" not in response.evidence
    assert response.evidence["observation_translation"]["observation"]["event_type"] == "MARKET_NEWS"
    assert "news_alpha_user_events" not in st.session_state or not st.session_state["news_alpha_user_events"]


def test_message_without_a_detectable_event_keeps_the_cached_path(_isolated):
    assert "don't have a cached" in _ask("What could I research?").text
    _seed_fomc(_isolated)
    response = _ask("ZN what could I research?")
    assert response.evidence["observation_translation"]["observation"]["root_symbol"] == "ZN"
    assert "impact_scan" not in response.evidence


def test_chat_event_route_never_calls_an_llm_or_writes_the_registry(monkeypatch):
    from alpha_agent.ui import translation_context

    def _boom(*a, **kw):
        raise AssertionError("the deterministic chat route must never construct a live LLM client")

    monkeypatch.setattr(translation_context, "AnthropicClient", _boom)
    before = hashlib.sha256(services.REGISTRY_PATH.read_bytes()).hexdigest()
    _ask("The Fed unexpectedly raised rates by 50 basis points. What could I research?")
    assert hashlib.sha256(services.REGISTRY_PATH.read_bytes()).hexdigest() == before


def test_agent_chat_message_appears_on_news_as_the_same_event():
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    # One session: Ask (the Agent chat) first, then News -- the same event, the same id.
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from alpha_agent.ui.views import agent, news\n"
        "(news.render if st.session_state.get('_show_news') else agent.render)()\n"
    )
    at.run(timeout=90)
    at.text_input(key="agent-ask-input").set_value(OPEC_CHAT)
    at.button(key="agent-ask-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    events = at.session_state["news_alpha_user_events"]
    assert [e.text for e in events] == [OPEC_EVENT]
    at.session_state["_show_news"] = True
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    body = " ".join(m.value for m in at.markdown)
    assert OPEC_EVENT in body  # the News card for the chat-described event
    assert any(b.key == f"news-USER_DESCRIBED-{events[0].event_id}-research" for b in at.button)