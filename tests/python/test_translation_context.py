"""Phase 1 -- `alpha_agent.ui.translation_context` boundary tests: candidate
observations are built ONLY from already-cached `market_intel_context` reads
(never a live fetch), and `translate_observation` never touches the network
in the default (SCRIPTED_MODE) path.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import alpha_agent.features.compute  # noqa: F401 -- register feature defs
import pytest
from alpha_agent.market_intel import mapping
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.market_intel.store import EventStore, NewsStore
from alpha_agent.ui import market_intel_context, services, translation_context

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

_NOW = datetime.now(UTC)


@pytest.fixture(autouse=True)
def _isolated_news_store(monkeypatch, tmp_path):
    """A fresh, isolated, on-disk `NewsStore`/`EventStore` per test -- never
    the real process-wide singleton (`NewsStore()`/`EventStore()` default to
    a persistent on-disk sqlite path; this sandbox has real data on it from
    prior manual app sessions). Never triggers a real connector refresh."""
    store = NewsStore(tmp_path / "news.sqlite")
    event_store = EventStore(tmp_path / "events.sqlite")
    monkeypatch.setattr(market_intel_context, "_store", lambda: store)
    monkeypatch.setattr(market_intel_context, "_event_store", lambda: event_store)
    monkeypatch.setattr(market_intel_context, "_last_refresh_wall", datetime.now(UTC), raising=False)
    yield store


def _seed_petroleum_item(store: NewsStore, *, news_id: str = "N1") -> MarketNewsItem:
    item = MarketNewsItem(
        news_id=news_id, headline="EIA reports a larger-than-expected crude draw",
        source_name="EIA", source_type=NewsSourceType.OFFICIAL, source_url="https://eia.gov/x",
        published_at=_NOW - timedelta(hours=2), retrieved_at=_NOW,
        related_products=mapping.products_for_category(NewsCategory.PETROLEUM),
        related_asset_classes=mapping.asset_classes_for_category(NewsCategory.PETROLEUM),
        category=NewsCategory.PETROLEUM, mapping_reason=mapping.mapping_reason_for_category(NewsCategory.PETROLEUM),
    )
    store.add_item(item)
    return item


def test_candidate_observations_empty_before_any_cache():
    assert translation_context.candidate_observations() == ()


def test_candidate_observations_only_lists_certified_roots(_isolated_news_store):
    _seed_petroleum_item(_isolated_news_store)
    candidates = translation_context.candidate_observations()
    roots = {c.root_symbol for c in candidates}
    # PETROLEUM maps to CL/MCL/RB/HO -- only CL is certified for Phase 1.
    assert roots == {"CL"}


def test_candidate_observations_never_triggers_a_live_refresh(_isolated_news_store, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("must never call a live refresh from candidate_observations")

    monkeypatch.setattr(market_intel_context, "_refresh_if_stale", _boom)
    translation_context.candidate_observations()  # must not raise


def test_build_observation_from_candidate(_isolated_news_store):
    _seed_petroleum_item(_isolated_news_store)
    candidate = translation_context.candidate_observations()[0]
    obs = translation_context.build_observation(candidate)
    assert obs.root_symbol == "CL"
    assert obs.event_type == "MARKET_NEWS"


def test_translate_observation_scripted_mode_makes_no_network_call(_isolated_news_store, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("SCRIPTED_MODE must never construct a live LLM client")

    monkeypatch.setattr(translation_context, "AnthropicClient", _boom)
    _seed_petroleum_item(_isolated_news_store)
    candidate = translation_context.candidate_observations()[0]
    obs = translation_context.build_observation(candidate)
    result, error = translation_context.translate_observation(obs, mode=translation_context.SCRIPTED_MODE)
    assert error is None
    assert result is not None
    assert result.generated_by == "DETERMINISTIC_LIBRARY"


class _FakeAnthropicClient:
    """A `LLMClient` test double injected in place of the real
    `AnthropicClient` -- mirrors `test_claude_conversation.py`'s own
    `_FakeAnthropicClient` pattern (never let a real network-touching client
    be constructed in a test, even to exercise the error path)."""

    raise_exc: Exception | None = None
    response_json: str | None = None

    def __init__(self, *a, **kw):
        pass

    def complete(self, **kwargs):
        from alpha_agent.agents.llm import LLMResponse

        if self.raise_exc is not None:
            raise self.raise_exc
        return LLMResponse(text=self.response_json, model="claude-sonnet-5-test", output_tokens=10)


def test_translate_observation_live_mode_unavailable_returns_honest_error_never_a_silent_fallback(
    _isolated_news_store, monkeypatch,
):
    from alpha_agent.agents.llm import LLMClientUnavailable

    fake = _FakeAnthropicClient
    fake.raise_exc = LLMClientUnavailable("no anthropic package installed")
    monkeypatch.setattr(translation_context, "AnthropicClient", fake)
    _seed_petroleum_item(_isolated_news_store)
    candidate = translation_context.candidate_observations()[0]
    obs = translation_context.build_observation(candidate)
    result, error = translation_context.translate_observation(obs, mode=translation_context.LIVE_MODE)
    assert result is None
    assert error is not None
    assert "unavailable" in error.lower()


def test_translate_observation_live_mode_with_a_working_client_is_labeled_claude(_isolated_news_store, monkeypatch):
    import json

    fake = _FakeAnthropicClient
    fake.raise_exc = None
    fake.response_json = json.dumps(
        {
            "mechanism_candidates": [
                {"mechanism": "TREND", "explanation": "x", "causal_chain": [], "evidence_basis": "y"},
            ],
            "measurable_variables": [],
            "factor_candidates": [
                {
                    "concept": "c", "mechanism": "TREND", "transform_or_proxy": "t",
                    "proposed_feature_kinds": ["trend_strength"], "required_external_data": [],
                },
            ],
        }
    )
    monkeypatch.setattr(translation_context, "AnthropicClient", fake)
    _seed_petroleum_item(_isolated_news_store)
    candidate = translation_context.candidate_observations()[0]
    obs = translation_context.build_observation(candidate)
    result, error = translation_context.translate_observation(obs, mode=translation_context.LIVE_MODE)
    assert error is None
    assert result is not None
    assert result.generated_by == "CLAUDE"
