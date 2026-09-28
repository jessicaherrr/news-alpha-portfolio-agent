"""Phase 1 -- the Research Translation (prompt 1 sections 12/13/18), now a
Research Thread Signals-step route ("test as a single-market strategy");
"Research This" still seeds the Agent (Ask) composer. Real
`streamlit.testing.v1.AppTest` renders.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from alpha_agent.market_intel import mapping
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.market_intel.store import EventStore, NewsStore
from alpha_agent.ui import market_intel_context, services
from alpha_agent.ui.research_thread import ThreadStep
from news_alpha_thread_support import strategy_route, thread_at

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

_NOW = datetime.now(UTC)


@pytest.fixture(autouse=True)
def _isolated_stores(monkeypatch, tmp_path):
    store = NewsStore(tmp_path / "news.sqlite")
    event_store = EventStore(tmp_path / "events.sqlite")
    monkeypatch.setattr(market_intel_context, "_store", lambda: store)
    monkeypatch.setattr(market_intel_context, "_event_store", lambda: event_store)
    monkeypatch.setattr(market_intel_context, "_last_refresh_wall", datetime.now(UTC), raising=False)
    yield store


def _seed_petroleum(store: NewsStore) -> None:
    store.add_item(
        MarketNewsItem(
            news_id="UI-1", headline="EIA reports a larger-than-expected crude draw",
            source_name="EIA", source_type=NewsSourceType.OFFICIAL, source_url="https://eia.gov/ui-1",
            published_at=_NOW - timedelta(hours=2), retrieved_at=_NOW,
            related_products=mapping.products_for_category(NewsCategory.PETROLEUM),
            related_asset_classes=mapping.asset_classes_for_category(NewsCategory.PETROLEUM),
            category=NewsCategory.PETROLEUM, mapping_reason=mapping.mapping_reason_for_category(NewsCategory.PETROLEUM),
        )
    )


def _petroleum_item() -> MarketNewsItem:
    return MarketNewsItem(
        news_id="UI-1", headline="EIA reports a larger-than-expected crude draw",
        source_name="EIA", source_type=NewsSourceType.OFFICIAL, source_url="https://eia.gov/ui-1",
        published_at=_NOW - timedelta(hours=2), retrieved_at=_NOW,
        related_products=mapping.products_for_category(NewsCategory.PETROLEUM),
        related_asset_classes=mapping.asset_classes_for_category(NewsCategory.PETROLEUM),
        category=NewsCategory.PETROLEUM, mapping_reason=mapping.mapping_reason_for_category(NewsCategory.PETROLEUM),
    )


def _translation_route():
    """A petroleum news item's Research Thread at Signals, with the
    single-market strategy route (the Research Translation) turned on."""
    return strategy_route(thread_at(_petroleum_item(), ThreadStep.SIGNALS))


def _translate(at):
    go = next(b for b in at.button if b.key and b.key.startswith("agent-translate-go-"))
    go.click().run(timeout=120)
    assert not list(at.exception), list(at.exception)
    return at


def test_the_translation_route_is_offered_in_the_thread():
    body = " ".join(m.value for m in _translation_route().markdown)
    assert "RESEARCH TRANSLATION" in body


def test_translate_button_renders_a_card():
    at = _translate(_translation_route())
    body = " ".join(m.value for m in at.markdown)
    assert "OBSERVATION" in body
    assert "POSSIBLE MECHANISMS" in body
    assert "FACTOR CANDIDATES" in body
    assert "HYPOTHESIS" in body


def test_translate_never_calls_a_live_llm_by_default(monkeypatch):
    from alpha_agent.ui import translation_context

    def _boom(*a, **kw):
        raise AssertionError("the default (unchecked) Translate action must never construct a live LLM client")

    monkeypatch.setattr(translation_context, "AnthropicClient", _boom)
    _translate(_translation_route())


def test_research_this_seeds_the_ask_composer():
    at = _translate(_translation_route())
    research_this = next(b for b in at.button if b.key and b.key.endswith("-research-this"))
    research_this.click().run(timeout=120)
    # "Research This" hands off to Ask (st.switch_page); the seeded composer state is what matters here.
    assert "agent_research_origin" in at.session_state
    origin = at.session_state["agent_research_origin"]
    assert origin["source"] == "observation_translation"
    assert origin["root"] == "CL"
    assert origin["holdout_eligible"] is False  # this sandbox's real wall clock is post-2025
    # The PETROLEUM template's first factor candidate (curve-shape
    # confirmation) is NOT_EXECUTABLE -- the hypothesis is actually built
    # from a later, AVAILABLE factor (trend continuation). The hand-off must
    # report THAT factor's researchability, never just factor_candidates[0].
    assert origin["researchability"] == "AVAILABLE"
    assert at.session_state["agent-root"] == "CL"
    # AppTest's `session_state` proxy has no real `.get()` (it treats any
    # attribute access as a widget-key lookup) -- `in` + `[]` is the only
    # safe read, matching every other AppTest-based test in this suite.
    objective = at.session_state["agent-objective"] if "agent-objective" in at.session_state else ""  # noqa: SIM401
    assert "observed_at" not in objective
