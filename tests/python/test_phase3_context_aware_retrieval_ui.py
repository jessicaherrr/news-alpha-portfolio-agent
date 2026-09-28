"""Phase 3 (Agentic Alpha Evolution) -- Agent page's "Context-Aware Research
Ranking" section. Real `AppTest` render through the full app entrypoint, same
pattern as `test_translation_alpha_memory_handoff.py` (which seeds the exact
same CL / PETROLEUM real-registry-evidence observation this file reuses).

Runs against the suite's GLOBAL default (Databento disconnected,
`conftest.py`'s autouse isolation fixtures) -- `context_retrieval_context`'s
own cache-first accessors degrade to an honest empty market state
(trend="Insufficient data", no curve, no peer confirmation) rather than
crashing, exactly like every other observation-plane accessor in this
codebase already does when disconnected.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from alpha_agent.market_intel import mapping
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.market_intel.store import EventStore, NewsStore
from alpha_agent.ui import market_intel_context, services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

_NOW = datetime.now(UTC)
APP_ENTRYPOINT = str(services.REPO_ROOT / "python" / "alpha_agent" / "ui" / "app.py")


@pytest.fixture(autouse=True)
def _isolated_stores(monkeypatch, tmp_path):
    store = NewsStore(tmp_path / "news.sqlite")
    event_store = EventStore(tmp_path / "events.sqlite")
    monkeypatch.setattr(market_intel_context, "_store", lambda: store)
    monkeypatch.setattr(market_intel_context, "_event_store", lambda: event_store)
    monkeypatch.setattr(market_intel_context, "_last_refresh_wall", datetime.now(UTC), raising=False)
    yield store


def _seed_petroleum(store: NewsStore, *, news_id: str = "HANDOFF-1", hours_ago: float = 2.0) -> None:
    """CL has real TREND-mapped (tsmom / ma_trend) registry evidence today
    -- same fixture `test_translation_alpha_memory_handoff.py` uses."""
    store.add_item(
        MarketNewsItem(
            news_id=news_id, headline="EIA reports a larger-than-expected crude draw",
            source_name="EIA", source_type=NewsSourceType.OFFICIAL, source_url=f"https://eia.gov/{news_id}",
            published_at=_NOW - timedelta(hours=hours_ago), retrieved_at=_NOW,
            related_products=mapping.products_for_category(NewsCategory.PETROLEUM),
            related_asset_classes=mapping.asset_classes_for_category(NewsCategory.PETROLEUM),
            category=NewsCategory.PETROLEUM, mapping_reason=mapping.mapping_reason_for_category(NewsCategory.PETROLEUM),
        )
    )


def _fresh_app():
    """The first relevant cached item's Research Thread at Signals, with the
    single-market strategy route (the Research Translation) turned on --
    where the translation card lives since the Research Thread workspace."""
    from alpha_agent.ui import news_alpha_context
    from alpha_agent.ui.research_thread import ThreadStep
    from news_alpha_thread_support import current_mandate, strategy_route, thread_at

    scan = news_alpha_context.cached_event_scans(current_mandate())[0]
    return strategy_route(thread_at(scan.event.source, ThreadStep.SIGNALS))


def _translate_first_candidate(at):
    go = next(b for b in at.button if b.key and b.key.startswith("agent-translate-go-"))
    go.click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


def test_context_aware_retrieval_section_renders_after_retrieval(_isolated_stores):
    _seed_petroleum(_isolated_stores)
    at = _fresh_app()
    _translate_first_candidate(at)

    go = next(b for b in at.button if b.key and b.key.endswith("-context-retrieval-go"))
    go.click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    body = " ".join(c.value for c in at.caption) + " " + " ".join(m.value for m in at.markdown)
    assert "CURRENT CONTEXT" in body
    assert "RANKED MECHANISM CANDIDATES" in body
    # CL / PETROLEUM maps to TREND and TERM_STRUCTURE in the deterministic
    # mechanism library -- both must appear, never silently dropped.
    assert "TREND" in body
    assert "TERM_STRUCTURE" in body
    assert "#1 VS #2" in body
    assert "SUGGESTED NEXT EXPERIMENT" in body
    # The page's own disclaimer explicitly disclaims these concepts (see
    # `RETRIEVAL_RANK_POLICY`'s docstring) -- the regression that matters is
    # a NUMBER presented as one, which `test_context_retrieval.py`'s
    # `test_schemas_never_carry_a_forbidden_field` already guards at the
    # schema level (no such field can even exist to render).
    assert "never expected return, probability of success, or trade confidence" in body


def test_context_aware_retrieval_result_is_cached_across_reruns(_isolated_stores):
    """A second, unrelated rerun (e.g. a sidebar navigation click on this
    same page) must not silently re-fetch/re-rank -- the result is cached
    in session state exactly like the Translate result above it."""
    _seed_petroleum(_isolated_stores)
    at = _fresh_app()
    _translate_first_candidate(at)

    go = next(b for b in at.button if b.key and b.key.endswith("-context-retrieval-go"))
    go.click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    stored = at.session_state["agent_context_retrieval"]
    assert stored  # at least one key_prefix entry cached
    for result in stored.values():
        assert result.ranked_candidates


def test_context_aware_retrieval_shows_event_importance_and_objective_magnitude(_isolated_stores):
    """Semantic hardening patch, section 3: category IMPORTANCE and
    objective event MAGNITUDE must render as two separate, honestly-labeled
    rows -- never conflated."""
    _seed_petroleum(_isolated_stores)
    at = _fresh_app()
    _translate_first_candidate(at)

    go = next(b for b in at.button if b.key and b.key.endswith("-context-retrieval-go"))
    go.click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    body = " ".join(c.value for c in at.caption)
    prov = " ".join(str(m.value) for m in at.markdown)
    assert "Event Importance" in prov
    assert "Objective Event Magnitude" in prov
    assert "UNKNOWN" in prov  # no real typed magnitude source is ingested anywhere today
    assert "never derived from event_importance" in body


def test_context_aware_retrieval_labels_related_prior_event_occurrences_honestly(_isolated_stores):
    """Semantic hardening patch, section 5: a genuine second, OLDER same-
    category item must render under "RELATED PRIOR EVENT OCCURRENCES",
    never "RELATED HISTORICAL CONTEXTS" (which would overclaim a matched
    historical market-state fingerprint this phase never reconstructs)."""
    _seed_petroleum(_isolated_stores, news_id="HANDOFF-CURRENT", hours_ago=2.0)
    _seed_petroleum(_isolated_stores, news_id="HANDOFF-PRIOR", hours_ago=48.0)
    at = _fresh_app()
    _translate_first_candidate(at)  # picks the most recent (HANDOFF-CURRENT)

    go = next(b for b in at.button if b.key and b.key.endswith("-context-retrieval-go"))
    go.click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    body = " ".join(str(m.value) for m in at.markdown) + " " + " ".join(c.value for c in at.caption)
    assert "RELATED PRIOR EVENT OCCURRENCES" in body
    assert "RELATED HISTORICAL CONTEXTS" not in body
    assert "matched today's" in body or "never that the historical market context" in body


def test_context_aware_retrieval_alpha_memory_links_navigate(_isolated_stores):
    _seed_petroleum(_isolated_stores)
    at = _fresh_app()
    _translate_first_candidate(at)

    go = next(b for b in at.button if b.key and b.key.endswith("-context-retrieval-go"))
    go.click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    memory_buttons = [b for b in at.button if b.key and "context-retrieval-alpha-" in b.key]
    assert memory_buttons, "TREND has real registry evidence on CL -- expected at least one linked Alpha Memory object"
    memory_buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    # "View Alpha Memory" opens Learn -> My Alpha on that object (Learn consumes the focus key).
    assert at.session_state["learn-tabs"] == "My Alpha"
