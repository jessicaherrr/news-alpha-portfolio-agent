"""News Alpha Phase A -- news impact triage (real `streamlit.testing.v1.AppTest`
renders). Migrated with the Research Thread workspace: the triage feed and
"describe an event" live on News, the mandate editor is the Research Setup
dialog, and the hand-off into the EXISTING Research Translation card is a
thread's Signals-step route ("test as a single-market strategy").
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from alpha_agent.market_intel import mapping
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.market_intel.store import EventStore, NewsStore
from alpha_agent.news_alpha import MandateDomain, ResearchMandate
from alpha_agent.ui import market_intel_context, news_alpha_context, services
from alpha_agent.ui.research_thread import ThreadStep
from news_alpha_thread_support import body, describe_on_news, run_page, thread_at

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
            news_id="TRIAGE-1", headline="EIA reports a larger-than-expected crude draw",
            source_name="EIA", source_type=NewsSourceType.OFFICIAL, source_url="https://eia.gov/triage-1",
            published_at=_NOW - timedelta(hours=2), retrieved_at=_NOW,
            related_products=mapping.products_for_category(NewsCategory.PETROLEUM),
            related_asset_classes=mapping.asset_classes_for_category(NewsCategory.PETROLEUM),
            category=NewsCategory.PETROLEUM, mapping_reason=mapping.mapping_reason_for_category(NewsCategory.PETROLEUM),
        )
    )


def _news():
    return run_page("news")


def _strategy_route(at, thread_id: str):
    """Turn on a thread Signals step's "test as a single-market strategy" route."""
    at.toggle(key=f"thread-strategy-route-{thread_id}").set_value(True).run(timeout=120)
    assert not list(at.exception), list(at.exception)
    return at


def test_news_renders_default_setup_and_empty_state():
    at = _news()
    text = body(at)
    assert "What is worth researching?" in text
    assert "Research Setup" in text and ">default<" in text  # not yet personalized
    assert "Nothing to research in this window yet" in text


def test_cached_news_card_shows_setup_scoped_domain_triage(_isolated_stores):
    _seed_petroleum(_isolated_stores)
    at = _news()
    text = body(at)
    assert "EIA reports a larger-than-expected crude draw" in text
    assert "Futures" in text and "<b>VERY HIGH</b>" in text  # PRIMARY exposure + larger-than-expected surprise
    assert "Options · not in your setup" in text and "Crypto · not in your setup" in text
    assert "Crude oil supply/demand balance" in text
    assert any(b.key and b.key.startswith("news-MARKET_NEWS-") and b.key.endswith("-research") for b in at.button)


def test_cached_items_irrelevant_to_every_allowed_domain_are_not_listed(_isolated_stores):
    _isolated_stores.add_item(
        MarketNewsItem(
            news_id="NOISE-1", headline="Battery storage capacity averaged 70% growth", source_name="EIA",
            source_type=NewsSourceType.OFFICIAL, source_url="https://eia.gov/noise-1",
            published_at=_NOW - timedelta(hours=1), retrieved_at=_NOW, category=NewsCategory.OTHER,
            mapping_reason=mapping.mapping_reason_for_category(NewsCategory.OTHER),
        )
    )
    _seed_petroleum(_isolated_stores)
    text = body(_news())
    assert "EIA reports a larger-than-expected crude draw" in text
    assert "Battery storage capacity" not in text


def test_describe_an_event_scans_it_and_hands_off_to_the_existing_translation(monkeypatch):
    from alpha_agent.ui import translation_context

    def _boom(*a, **kw):
        raise AssertionError("the default Translate action must never construct a live LLM client")

    monkeypatch.setattr(translation_context, "AnthropicClient", _boom)
    text = "OPEC+ agrees a production cut of 1 million barrels per day"
    at = describe_on_news(text)
    shown = body(at)
    assert "Futures" in shown and "<b>HIGH</b>" in shown and "Equity" in shown and "<b>MEDIUM</b>" in shown

    at = thread_at(text, ThreadStep.SIGNALS)
    at = _strategy_route(at, at.thread_id)
    assert "approximation" in body(at)  # channel-routed hand-off is labelled as one
    go = next(b for b in at.button if b.key and b.key.startswith("agent-translate-go-user:") and b.key.endswith(":CL"))
    go.click().run(timeout=120)
    assert not list(at.exception), list(at.exception)
    shown = body(at)
    assert "POSSIBLE MECHANISMS" in shown and "HYPOTHESIS" in shown


def test_a_saved_setup_excluding_futures_removes_futures_handoffs(_isolated_stores):
    news_alpha_context.MANDATE_STORE.save(ResearchMandate(allowed_domains=(MandateDomain.EQUITY,)))
    _seed_petroleum(_isolated_stores)
    text = body(_news())
    assert "Futures · not in your setup" in text
    assert "Equity" in text and "<b>HIGH</b>" in text
    assert ">default<" not in text

    event = "OPEC+ agrees a production cut of 1 million barrels per day"
    at = thread_at(event, ThreadStep.SIGNALS)
    at = _strategy_route(at, at.thread_id)
    assert not any(b.key and b.key.startswith("agent-translate-go-") for b in at.button)


def test_setup_editor_saves_access_constraints_only(monkeypatch, tmp_path):
    from alpha_agent.recommendation.profile import ProfileStore

    monkeypatch.setattr(services, "_PROFILE_STORE", ProfileStore(tmp_path / "profile.json"))
    at = run_page("news", session={"research_setup_editor_open": True})
    at.button_group(key="research-setup-domains").set_value([MandateDomain.FUTURES, MandateDomain.CRYPTO])
    at.toggle(key="research-setup-shorting").set_value(True)
    save = next(b for b in at.button if b.label == "Save setup")
    save.click().run(timeout=120)
    assert not list(at.exception), list(at.exception)
    saved = json.loads(news_alpha_context.MANDATE_STORE.path.read_text())
    assert saved["allowed_domains"] == ["FUTURES", "CRYPTO"]
    assert saved["shorting_allowed"] is True
    assert "risk_profile" not in saved  # the risk half stays in the profile file
    assert not at.session_state["research_setup_editor_open"]
