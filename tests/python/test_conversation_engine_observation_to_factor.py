"""Phase 1 -- `conversation_engine._handle_observation_to_factor` (prompt 1
section 12). Always the offline/deterministic engine (this handler never
calls Claude -- see the module's own section comment); isolates the
`market_intel_context` news/event singletons the same way
`test_translation_context.py` does (both default to a persistent on-disk
store that can carry real data from prior manual sessions in this sandbox).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import alpha_agent.features.compute  # noqa: F401 -- register feature defs
import pytest
from alpha_agent.agents.conversation import (
    ConversationSignalContext,
    ResearchConversationIntent,
    classify_intent,
)
from alpha_agent.market_intel import mapping
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.market_intel.store import EventStore, NewsStore
from alpha_agent.ui import conversation_engine, market_intel_context, services
from alpha_agent.ui.conversation_engine import ConversationBoundContext

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

_NOW = datetime.now(UTC)
_CTX = ConversationBoundContext()
_SIG_CTX = ConversationSignalContext(approved_roots=("ES", "NQ", "CL", "GC", "ZN"))


@pytest.fixture(autouse=True)
def _isolated_stores(monkeypatch, tmp_path):
    store = NewsStore(tmp_path / "news.sqlite")
    event_store = EventStore(tmp_path / "events.sqlite")
    monkeypatch.setattr(market_intel_context, "_store", lambda: store)
    monkeypatch.setattr(market_intel_context, "_event_store", lambda: event_store)
    monkeypatch.setattr(market_intel_context, "_last_refresh_wall", datetime.now(UTC), raising=False)
    yield store


def _seed(store: NewsStore, category: NewsCategory, *, news_id: str = "N1") -> None:
    # Distinct headline/url per item -- NewsStore dedups near-identical
    # headline/url pairs, which would silently collapse two differently
    # categorised test items into one.
    store.add_item(
        MarketNewsItem(
            news_id=news_id, headline=f"A real official {category.value} release headline ({news_id})",
            source_name="Official Source", source_type=NewsSourceType.OFFICIAL,
            source_url=f"https://example.gov/{news_id}",
            published_at=_NOW - timedelta(hours=1), retrieved_at=_NOW,
            related_products=mapping.products_for_category(category),
            related_asset_classes=mapping.asset_classes_for_category(category),
            category=category, mapping_reason=mapping.mapping_reason_for_category(category),
        )
    )


def test_no_cached_observation_gives_an_honest_message_never_a_crash():
    classification = classify_intent("EIA inventories dropped sharply. What could I research?", context=_SIG_CTX)
    response = conversation_engine._handle_observation_to_factor("...", classification, _CTX)
    assert response.intent == ResearchConversationIntent.OBSERVATION_TO_FACTOR
    assert "don't have a cached" in response.text
    assert response.evidence == {}


def test_cached_observation_produces_a_grounded_structured_reply(_isolated_stores):
    _seed(_isolated_stores, NewsCategory.PETROLEUM)
    classification = classify_intent("What could I research from this?", context=_SIG_CTX)
    response = conversation_engine._handle_observation_to_factor("...", classification, _CTX)
    assert "OBSERVATION:" in response.text
    assert "observation_translation" in response.evidence
    translation = response.evidence["observation_translation"]
    assert translation["generated_by"] == "DETERMINISTIC_LIBRARY"
    assert translation["observation"]["root_symbol"] == "CL"


def test_mentioned_root_narrows_which_cached_observation_is_used(_isolated_stores):
    _seed(_isolated_stores, NewsCategory.PETROLEUM, news_id="N-CL")
    _seed(_isolated_stores, NewsCategory.FOMC_POLICY, news_id="N-MACRO")
    classification = classify_intent("ZN what could I research?", context=_SIG_CTX)
    response = conversation_engine._handle_observation_to_factor("...", classification, _CTX)
    translation = response.evidence["observation_translation"]
    assert translation["observation"]["root_symbol"] == "ZN"


def test_handler_never_touches_the_network(_isolated_stores, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("the always-deterministic chat handler must never construct a live LLM client")

    from alpha_agent.ui import translation_context

    monkeypatch.setattr(translation_context, "AnthropicClient", _boom)
    _seed(_isolated_stores, NewsCategory.PETROLEUM)
    classification = classify_intent("What could I research?", context=_SIG_CTX)
    conversation_engine._handle_observation_to_factor("...", classification, _CTX)  # must not raise


def test_registered_in_handlers_table():
    assert (
        conversation_engine._HANDLERS[ResearchConversationIntent.OBSERVATION_TO_FACTOR]
        is conversation_engine._handle_observation_to_factor
    )
