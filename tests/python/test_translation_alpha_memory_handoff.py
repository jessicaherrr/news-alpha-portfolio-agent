"""Phase 2 section 17 -- Phase 1 integration: a translated observation's
mechanism candidates can retrieve real Personal Alpha Memory (never Phase 3
context-match ranking). Real `AppTest` render through the full app entrypoint
so `st.switch_page` is exercised exactly as a live session would use it (same
pattern as `test_agent_to_research_navigation.py`).
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


def _seed_petroleum(store: NewsStore) -> None:
    """Mirrors `test_phase1_agent_translation_ui.py::_seed_petroleum` -- a
    PETROLEUM observation maps to CL, which has real TREND-mapped (tsmom /
    ma_trend) registry evidence today."""
    store.add_item(
        MarketNewsItem(
            news_id="HANDOFF-1", headline="EIA reports a larger-than-expected crude draw",
            source_name="EIA", source_type=NewsSourceType.OFFICIAL, source_url="https://eia.gov/handoff-1",
            published_at=_NOW - timedelta(hours=2), retrieved_at=_NOW,
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


def test_personal_alpha_memory_nugget_appears_for_a_researched_mechanism(_isolated_stores):
    """CL has real registry evidence for TREND (tsmom + ma_trend, two real
    separate Factors today) -- the Research Translation panel's "Research
    memory" expander must surface a real Related Mechanism Memory link for
    it, never fabricate one for an unmapped mechanism."""
    _seed_petroleum(_isolated_stores)
    at = _fresh_app()

    go = next(b for b in at.button if b.key and b.key.startswith("agent-translate-go-"))
    go.click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    memory_buttons = [b for b in at.button if b.key and "alpha-memory" in b.key]
    body = " ".join(c.value for c in at.caption)
    assert "Related Mechanism Memory" in body
    assert memory_buttons, "expected at least one 'View Alpha Memory' button for a researched mechanism"


def test_view_alpha_memory_button_navigates_to_the_matching_library_detail(_isolated_stores):
    _seed_petroleum(_isolated_stores)
    at = _fresh_app()
    at.button(key=next(b.key for b in at.button if b.key and b.key.startswith("agent-translate-go-"))).click().run(timeout=90)

    memory_buttons = [b for b in at.button if b.key and "alpha-memory" in b.key]
    assert memory_buttons
    memory_buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    # "View Alpha Memory" opens Learn -> My Alpha on that object (Learn consumes the focus key).
    assert at.session_state["learn-tabs"] == "My Alpha"
    selected_id = at.session_state["alpha_library_selected_id"]
    assert selected_id

    body = " ".join(m.value for m in at.markdown)
    assert '<div class="aa-page-title">Learn</div>' in body  # the switch really landed on Learn
    obj = services.get_alpha_research_object(selected_id)
    assert obj is not None
    assert obj["root_symbol"] == "CL"


def test_no_alpha_memory_nugget_for_an_unmapped_mechanism(monkeypatch, _isolated_stores):
    """CARRY (and every mechanism outside Phase 1's frozen bridge) must never
    get a fabricated Personal Alpha Memory link."""
    from alpha_agent.ui import services as ui_services

    monkeypatch.setattr(
        ui_services, "mechanism_memory_lookup",
        lambda **kw: {"mechanism": kw["mechanism"], "root_symbol": kw["root_symbol"], "match_kind": "NONE", "objects": []},
    )
    _seed_petroleum(_isolated_stores)
    at = _fresh_app()
    go = next(b for b in at.button if b.key and b.key.startswith("agent-translate-go-"))
    go.click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    memory_buttons = [b for b in at.button if b.key and "alpha-memory" in b.key]
    assert not memory_buttons


def test_mechanism_only_hand_off_never_claims_matching_factor_even_with_real_evidence(_isolated_stores):
    """Final Phase 2 semantic fix, section 2: Phase 1's hand-off only ever
    knows the candidate's mechanism, never a concrete strategy_family --
    mechanism-only lookup must render the hedged "Related Mechanism Memory"
    wording, never the confident "matching factor" one, however much real
    evidence exists under the mechanism (CL/TREND has two real Factors
    today)."""
    _seed_petroleum(_isolated_stores)
    at = _fresh_app()
    go = next(b for b in at.button if b.key and b.key.startswith("agent-translate-go-"))
    go.click().run(timeout=90)
    body = " ".join(c.value for c in at.caption)
    assert "Related Mechanism Memory" in body
    assert "Personal Alpha Memory (matching factor)" not in body


def test_multiple_real_factors_under_one_mechanism_each_get_their_own_button(_isolated_stores):
    """CL's real TREND mechanism resolves to two real, separate Factors
    (tsmom, ma_trend) -- the hand-off must show every one of them, never
    silently pick one."""
    _seed_petroleum(_isolated_stores)
    at = _fresh_app()
    go = next(b for b in at.button if b.key and b.key.startswith("agent-translate-go-"))
    go.click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    lookup = services.mechanism_memory_lookup(mechanism="TREND", root_symbol="CL")
    assert lookup["match_kind"] == "RELATED_MECHANISM"
    assert len(lookup["objects"]) == 2
    memory_buttons = [b for b in at.button if b.key and "alpha-memory-TREND" in b.key]
    assert len(memory_buttons) == 2  # one per real Factor, never a single silently-picked one


def test_matching_factor_rendering_path_is_exercised_when_the_service_returns_it(monkeypatch, _isolated_stores):
    """`_render_personal_alpha_memory_links`'s MATCHING_FACTOR branch is
    unreachable from today's real mechanism-only hand-off (see the test
    above) but must still render correctly for a future integration point
    that does supply an exact match -- proven here by making the service
    return MATCHING_FACTOR directly, exactly as `mechanism_memory_lookup`
    would once a caller supplies a strategy_family."""
    from alpha_agent.ui import services as ui_services

    real_obj = services.list_alpha_research_objects("CL")[0]

    def _fake_lookup(**kw):
        return {"mechanism": kw["mechanism"], "root_symbol": kw["root_symbol"], "match_kind": "MATCHING_FACTOR", "objects": [real_obj]}

    monkeypatch.setattr(ui_services, "mechanism_memory_lookup", _fake_lookup)
    _seed_petroleum(_isolated_stores)
    at = _fresh_app()
    go = next(b for b in at.button if b.key and b.key.startswith("agent-translate-go-"))
    go.click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    body = " ".join(c.value for c in at.caption)
    assert "Personal Alpha Memory (matching factor)" in body
    assert "Related Mechanism Memory" not in body


def test_research_translation_never_writes_the_registry_via_the_alpha_memory_hook():
    import re

    from alpha_agent.ui import translation_view

    with open(translation_view.__file__, encoding="utf-8") as fh:
        text = fh.read()
    hook_start = text.index("_render_personal_alpha_memory_links")
    hook_region = text[hook_start:hook_start + 1500]
    for pattern in (r"\.insert_experiment\(", r"\.apply_bundle\(", r"\.record_failure\("):
        assert not re.search(pattern, hook_region)
