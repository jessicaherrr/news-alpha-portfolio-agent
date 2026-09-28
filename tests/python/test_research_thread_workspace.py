"""News Alpha Research Thread workspace (real `streamlit.testing.v1.AppTest`
renders + unit tests of the thread model).

Proves: a thread is created from News and persisted; one thread per event;
the Research Setup persists and a change invalidates exactly the dependent
steps; the stepper's Next / Previous / completed-step / unreachable-step
behaviour; deep links (in-app and URL); following paths and choosing a market
are FOCUS, never a change of the tested family; every value shown matches the
canonical backend object; unsupported asset classes and PIT status are shown
honestly; prices never load on render; and the workspace modules contain no
orchestration or scientific computation of their own.
"""
from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import alpha_agent.features.compute  # noqa: F401 -- register feature defs
import pytest
from alpha_agent.market_intel.store import EventStore, NewsStore
from alpha_agent.news_alpha import MandateDomain, ResearchMandate
from alpha_agent.portfolio import EligibilityMode
from alpha_agent.ui import (
    market_home,
    market_intel_context,
    news_alpha_context,
    portfolio_plan_view,
    research_thread,
    services,
)
from alpha_agent.ui.research_thread import STEPS, ResearchThreadStore, ThreadStep
from news_alpha_thread_support import (
    body,
    current_mandate,
    describe_on_news,
    run_page,
    seed_thread,
    tables,
    thread_at,
)

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

AI = "Microsoft expands AI infrastructure spending"
THREE = ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES, MandateDomain.ETF))
UI = Path(__file__).resolve().parents[2] / "python" / "alpha_agent" / "ui"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    news, events = NewsStore(tmp_path / "news.sqlite"), EventStore(tmp_path / "events.sqlite")
    monkeypatch.setattr(market_intel_context, "_store", lambda: news)
    monkeypatch.setattr(market_intel_context, "_event_store", lambda: events)
    monkeypatch.setattr(market_intel_context, "_last_refresh_wall", datetime.now(UTC), raising=False)
    from alpha_agent.recommendation.profile import ProfileStore

    monkeypatch.setattr(services, "_PROFILE_STORE", ProfileStore(tmp_path / "profile.json"))
    news_alpha_context.MANDATE_STORE.save(THREE)

    def _no_network(*a, **kw):
        raise AssertionError("the workspace must never fetch prices on render")

    monkeypatch.setattr(market_home, "get_ohlcv_cached", _no_network)


def _thread():
    (thread,) = research_thread.THREAD_STORE.list()
    return thread


# ---------------------------------------------------------------------------
# creation, persistence, one thread per event
# ---------------------------------------------------------------------------


def test_research_on_news_creates_and_persists_a_thread_at_the_news_step():
    at = describe_on_news(AI)
    button = next(b for b in at.button if b.key and b.key.startswith("news-USER_DESCRIBED-") and
                  b.key.endswith("-research"))
    button.click().run(timeout=120)
    thread = _thread()
    assert thread.current_step is ThreadStep.NEWS and thread.completed_steps == ()
    assert thread.name == "AI infrastructure investment"  # the graph's own anchor state
    assert thread.event.source_type == "user" and thread.event.headline == AI
    assert at.session_state["research_thread_active_id"] == thread.thread_id
    assert thread.setup_fingerprint == current_mandate().fingerprint()


def test_a_thread_survives_a_new_store_instance_and_rescans_identically():
    thread = seed_thread(AI)
    reloaded = ResearchThreadStore(research_thread.THREAD_STORE.path).get(thread.thread_id)
    assert reloaded == thread
    pipe = research_thread.pipeline_for(reloaded, current_mandate())
    assert pipe.scan.event.event_id == thread.event.event_id  # the stored source reproduces the scan


def test_reopening_the_same_event_returns_the_same_thread_with_its_progress():
    thread = seed_thread(AI)
    research_thread.advance(thread)
    again = research_thread.create_thread(research_thread.pipeline_for(thread, current_mandate()).scan,
                                          current_mandate())
    assert again.thread_id == thread.thread_id and again.current_step is ThreadStep.REASONING


# ---------------------------------------------------------------------------
# stepper, previous / next, reachability, deep links
# ---------------------------------------------------------------------------


def test_next_completes_the_step_and_previous_returns_without_losing_progress():
    at = thread_at(AI)
    at.button(key="thread-next").click().run(timeout=120)
    thread = _thread()
    assert thread.current_step is ThreadStep.REASONING and thread.completed_steps == (ThreadStep.NEWS,)
    at.button(key="thread-prev").click().run(timeout=120)
    thread = _thread()
    assert thread.current_step is ThreadStep.NEWS and ThreadStep.NEWS in thread.completed_steps
    assert thread.furthest_step() is ThreadStep.REASONING  # progress kept: Reasoning stays reachable
    assert thread.is_reachable(ThreadStep.MARKET) and not thread.is_reachable(ThreadStep.SIGNALS)


def test_the_stepper_marks_done_current_todo_and_disables_unreachable_steps():
    at = thread_at(AI, ThreadStep.MARKET)
    buttons = {b.key: b for b in at.button if b.key and b.key.startswith("stepbtn-")}
    assert [k.removeprefix("stepbtn-") for k in buttons] == [s.value for s in STEPS]
    assert not buttons["stepbtn-news"].disabled and not buttons["stepbtn-market"].disabled
    assert not buttons["stepbtn-signals"].disabled  # the next step is reachable
    assert buttons["stepbtn-portfolio"].disabled and buttons["stepbtn-learn"].disabled
    buttons["stepbtn-news"].click().run(timeout=120)
    assert _thread().current_step is ThreadStep.NEWS  # a completed step is one click away
    assert ThreadStep.REASONING in _thread().completed_steps


def test_deep_link_opens_at_a_step_with_the_lineage_marked_complete():
    thread = research_thread.deep_link(seed_thread(AI), ThreadStep.SIGNALS)
    assert thread.current_step is ThreadStep.SIGNALS
    assert thread.completed_steps == (ThreadStep.NEWS, ThreadStep.REASONING, ThreadStep.MARKET)


def test_url_deep_link_opens_the_thread_at_the_step():
    thread = seed_thread(AI)
    from news_alpha_thread_support import app_test

    at = app_test().from_string("from alpha_agent.ui.views.workspace import render\nrender()\n")
    at.query_params["thread"] = thread.thread_id
    at.query_params["step"] = "portfolio"
    at.run(timeout=180)
    assert not list(at.exception), list(at.exception)
    assert "Step 5 of 7" in body(at)
    assert _thread().current_step is ThreadStep.PORTFOLIO


def test_research_without_an_active_thread_lists_threads_and_points_to_news():
    seed_thread(AI)
    at = run_page("workspace")
    text = body(at)
    assert "Your research threads" in text and "AI infrastructure investment" in text
    assert any(b.key == "research-go-news" for b in at.button)


# ---------------------------------------------------------------------------
# Research Setup -- persistent context; a change invalidates dependent steps
# ---------------------------------------------------------------------------


def test_setup_change_invalidates_market_onward_but_keeps_reasoning():
    thread = research_thread.deep_link(seed_thread(AI), ThreadStep.PORTFOLIO)
    news_alpha_context.MANDATE_STORE.save(ResearchMandate(allowed_domains=(MandateDomain.FUTURES,)))
    mandate = current_mandate()
    pipe = research_thread.pipeline_for(research_thread.THREAD_STORE.get(thread.thread_id), mandate)
    updated, notice = research_thread.reconcile_setup(pipe.thread, mandate, pipe.expressions)
    assert notice and "Reasoning is unaffected" in notice
    assert updated.completed_steps == (ThreadStep.NEWS, ThreadStep.REASONING)
    assert updated.current_step is ThreadStep.MARKET
    assert updated.setup_fingerprint == mandate.fingerprint()
    again, second = research_thread.reconcile_setup(updated, mandate, pipe.expressions)
    assert second is None and again.completed_steps == updated.completed_steps


def test_the_setup_change_notice_renders_in_the_thread():
    thread = research_thread.deep_link(seed_thread(AI), ThreadStep.SIGNALS)
    news_alpha_context.MANDATE_STORE.save(ResearchMandate(allowed_domains=(MandateDomain.FUTURES,)))
    from news_alpha_thread_support import app_test

    at = app_test().from_string(
        f"import streamlit as st\nst.session_state.setdefault('research_thread_active_id', {thread.thread_id!r})\n"
        "from alpha_agent.ui.views.workspace import render\nrender()\n"
    )
    at.run(timeout=180)
    assert any("Research Setup changed" in w.value for w in at.warning)
    assert "Step 3 of 7" in body(at)


# ---------------------------------------------------------------------------
# Reasoning / Market -- focus is never a new family
# ---------------------------------------------------------------------------


def test_following_a_path_is_saved_highlighted_and_narrows_the_market_step():
    at = thread_at(AI, ThreadStep.REASONING)
    follow = next(b for b in at.button if b.key and b.key.startswith("follow-sp-"))
    path_id = follow.key.removeprefix("follow-")
    follow.click().run(timeout=120)
    thread = _thread()
    assert thread.followed_path_ids == (path_id,)
    pipe = research_thread.pipeline_for(thread, current_mandate())
    from alpha_agent.ui import mechanism_graph_view

    dot = mechanism_graph_view.graph_dot(pipe.graph, highlight=[p.state_ids for p in pipe.followed_paths()])
    assert "penwidth=2.6" in dot  # the followed path's links are drawn in the accent
    all_markets = len(pipe.market_groups())
    market = thread_at(AI, ThreadStep.MARKET, followed_path_ids=(path_id,))
    assert "you follow" in body(market)
    selectors = [bg for bg in market.button_group if bg.key == f"mx-select-{market.thread_id}"]
    assert (len(selectors[0].options) if selectors else 0) < all_markets
    # focus never changes the tested family: the candidate set is the event's full one
    assert len(pipe.candidates.candidates) == len(research_thread.pipeline_for(
        thread.model_copy(update={"followed_path_ids": ()}), current_mandate()).candidates.candidates)


def test_selecting_a_market_sets_the_thread_focus_and_the_context_panel():
    at = thread_at(AI, ThreadStep.MARKET)
    from alpha_agent.ui.workspace import step_market

    from alpha_agent.ui.workspace import market_data_panel

    groups = {step_market._label(g): g for g in research_thread.pipeline_for(_thread(), current_mandate())
              .market_groups()}
    selector = at.button_group(key=f"mx-select-{at.thread_id}")
    shown = [groups[label] for label in selector.options]
    # a market is always shown: by default the first measurable one with a price feed (NQ here)
    default = next(g for g in shown if market_data_panel.has_price_feed(g))
    assert selector.value == default.key and "NQ" in default.symbols
    second = next(g.key for g in shown if g.key != default.key)
    selector.set_value(second).run(timeout=120)
    assert not list(at.exception), list(at.exception)
    thread = _thread()
    group = research_thread.pipeline_for(thread, current_mandate()).focus_group()
    assert group is not None and group.key == second
    text = body(at)
    assert "Why affected" in text and "Price trend" in text and "Measure it" in text
    assert "Current market" in text and group.concept in text


def test_measure_it_distinguishes_ideal_measurement_actual_data_and_pit_status():
    thread = seed_thread(AI)
    pipe = research_thread.pipeline_for(thread, current_mandate())
    nq = next(g for g in pipe.market_groups() if "NQ" in g.symbols)
    at = thread_at(AI, ThreadStep.MARKET, focus_expression_id=nq.first.expression_id)
    text = body(at)
    for label in ("Ideal measurement", "Actual data", "Point-in-time", "PIT safe", "Missing",
                  "Ready to become candidate signals", "Economic consequence"):
        assert label in text, label
    assert "Load 5D prices" in [b.label for b in at.button]  # prices only behind an explicit action


def test_unsupported_asset_classes_say_so_instead_of_showing_numbers():
    thread = seed_thread(AI)
    pipe = research_thread.pipeline_for(thread, current_mandate())
    etf = next(g for g in pipe.market_groups() if g.domain is MandateDomain.ETF and g.symbols)
    text = body(thread_at(AI, ThreadStep.MARKET, focus_expression_id=etf.first.expression_id))
    assert "Recent ETF prices are not connected in this build" in text and "Not connected" in text
    eq = next(g for g in pipe.market_groups() if g.domain is MandateDomain.EQUITY and g.symbols)
    text = body(thread_at(AI, ThreadStep.MARKET, focus_expression_id=eq.first.expression_id))
    assert "Equity price data has not been acquired yet" in text and "Data not acquired" in text


def test_the_price_button_loads_through_the_existing_cost_checked_path(monkeypatch):
    calls = []
    monkeypatch.setattr(market_home, "get_ohlcv_cached", lambda root, **kw: calls.append((root, kw)))
    thread = seed_thread(AI)
    nq = next(g for g in research_thread.pipeline_for(thread, current_mandate()).market_groups() if "NQ" in g.symbols)
    at = thread_at(AI, ThreadStep.MARKET, focus_expression_id=nq.first.expression_id)
    assert calls == []  # nothing on render
    next(b for b in at.button if b.label == "Load 5D prices").click().run(timeout=120)
    assert calls == [("NQ", {"timeframe": "1h", "lookback_bars": 120})]


# ---------------------------------------------------------------------------
# Signals / Portfolio / Backtest -- every value is the backend's
# ---------------------------------------------------------------------------


def test_signals_step_counts_match_the_canonical_candidate_set():
    at = thread_at(AI, ThreadStep.SIGNALS)
    pipe = research_thread.pipeline_for(_thread(), current_mandate())
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Candidate signals"] == str(len(pipe.candidates.candidates))
    assert metrics["Screened"] == str(len(pipe.screens))
    assert "Rank IC" not in body(at)  # time-series ICs are never called Rank IC


def test_portfolio_and_backtest_steps_render_the_canonical_plan_and_gate():
    thread = seed_thread(AI)
    at = thread_at(thread, ThreadStep.PORTFOLIO)
    pipe = research_thread.pipeline_for(_thread(), current_mandate())
    plan, _missing = portfolio_plan_view.build_plan(pipe.ranked, [pipe.candidates], pipe.mandate,
                                                    EligibilityMode.QUALIFIED)
    assert plan.headline() in body(at)
    backtest = thread_at(_thread(), ThreadStep.BACKTEST)
    from alpha_agent.ui.workspace import step_backtest

    _tone, label, detail = step_backtest.status_label(research_thread.pipeline_for(_thread(), current_mandate()))
    text = body(backtest)
    assert label in text and detail in text and "2025 holdout sealed" in text
    assert "Validation" in text and "Not testable yet" in text  # the context panel agrees with the gate


def test_learn_step_links_to_my_alpha_research_map_and_community():
    at = thread_at(AI, ThreadStep.LEARN)
    text = body(at)
    assert "This thread, end to end" in text and "What this thread taught us" in text
    keys = {b.key for b in at.button}
    assert {"thread-learn-alpha_library", "thread-learn-alpha_graph", "thread-learn-community"} <= keys
    at.button(key="thread-learn-alpha_graph").click().run(timeout=120)
    assert not list(at.exception), list(at.exception)
    assert at.session_state["learn-tabs"] == "Research Map"


def test_my_alpha_lists_threads_and_opens_them_at_their_step():
    thread = research_thread.deep_link(seed_thread(AI), ThreadStep.MARKET)
    at = run_page("learn")
    assert "Research threads" in body(at) and "AI infrastructure investment" in body(at)
    assert f"learn-open-thread-{thread.thread_id}" in {b.key for b in at.button}


def test_thread_references_expose_provenance_progressively():
    text = body(thread_at(AI, ThreadStep.SIGNALS))
    for label in ("Thread", "Event", "Research Setup", "Mechanism graph", "Signal paths", "Asset expressions",
                  "Candidate signals"):
        assert f'<span class="aa-prov-label">{label}</span>' in text, label


def test_the_portfolio_page_combines_every_thread():
    seed_thread(AI)
    seed_thread("OPEC+ agrees a production cut of 1 million barrels per day")
    at = run_page("portfolio")
    (threads,) = [df.value for df in at.dataframe if "Candidate signals" in df.value.columns]
    assert len(threads) == 2
    assert "Portfolio plan" in body(at) and "Validation &amp; research memory" in body(at)
    assert "Stage" in tables(at) or "News" in tables(at)


# ---------------------------------------------------------------------------
# static guards -- no second orchestration path, no UI-side science
# ---------------------------------------------------------------------------

_RENDERERS = sorted((UI / "workspace").glob("*.py")) + [UI / "views" / f for f in ("news.py", "workspace.py",
                                                                                     "portfolio.py")]
_FORBIDDEN_CALLS = {
    "scan_initial_impact", "scan_many", "build_economic_mechanism_graph", "discover_signal_paths", "adjust_impact",
    "build_asset_expressions", "build_candidate_signals", "rank_candidate_signals", "construct_portfolio_plan",
    "gate_portfolio", "validate_portfolio", "screen_candidates", "screen_candidate", "close_research_loop",
    "record_signal_path_evidence", "ExperimentRegistry", "run_deep_research", "ResearchOrchestrator",
}


@pytest.mark.parametrize("path", _RENDERERS, ids=lambda p: p.name)
def test_workspace_renderers_never_compute_research_values_themselves(path):
    source = path.read_text(encoding="utf-8")
    for token in ("sorted(", ".sort(", "max(", "min(", "round("):
        assert token not in source, (path.name, token)
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & _FORBIDDEN_CALLS, (path.name, names & _FORBIDDEN_CALLS)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not imported & _FORBIDDEN_CALLS


def test_the_thread_model_only_reaches_the_backend_through_the_existing_ui_boundaries():
    source = (UI / "research_thread.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not calls & _FORBIDDEN_CALLS
    for boundary in ("news_alpha_context", "candidate_signal_view", "signal_ranking_view", "portfolio_plan_view",
                     "research_loop_view"):
        assert boundary in source
    assert "ExperimentRegistry" not in source and "registry" not in {
        a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}


# ---------------------------------------------------------------------------
# Market -- ONE destination: Affected Markets (the thread's) + Explore All
# ---------------------------------------------------------------------------


def test_market_without_a_thread_opens_the_explorer_with_a_notice():
    at = run_page("market")
    text = body(at)
    assert "No active research thread" in text and "Explore all markets" in text
    assert any(b.key == "market-start-news" for b in at.button)
    assert not [bg for bg in at.button_group if bg.key and bg.key.startswith("mx-select-")]


def test_market_with_an_active_thread_defaults_to_the_same_affected_markets():
    thread = research_thread.deep_link(seed_thread(AI), ThreadStep.MARKET)
    at = run_page("market", session={"research_thread_active_id": thread.thread_id})
    text = body(at)
    assert "Researching" in text and AI in text
    assert at.session_state["market_view"] == "affected"
    market_options = at.button_group(key=f"mx-select-{thread.thread_id}").options
    step = thread_at(thread, ThreadStep.MARKET)
    assert step.button_group(key=f"mx-select-{thread.thread_id}").options == market_options  # one component
    assert "Explore all markets" not in text  # the explorer only renders when chosen


def test_market_explore_view_keeps_the_full_explorer_while_a_thread_is_active():
    thread = seed_thread(AI)
    at = run_page("market", session={"research_thread_active_id": thread.thread_id, "market_view": "explore"})
    text = body(at)
    assert "Explore all markets" in text and "Researching" in text
    assert not [bg for bg in at.button_group if bg.key and bg.key.startswith("mx-select-")]


def test_affected_markets_never_list_unreached_instruments():
    thread = seed_thread(AI)
    pipe = research_thread.pipeline_for(thread, current_mandate())
    from alpha_agent.ui.workspace import step_market

    reached = {step_market._label(g) for g in pipe.market_groups()}
    at = run_page("market", session={"research_thread_active_id": thread.thread_id})
    assert set(at.button_group(key=f"mx-select-{thread.thread_id}").options) <= reached
    assert "CL" not in {s for g in pipe.market_groups() for s in g.symbols}  # crude is not on the AI routes
    assert "CL · " not in " ".join(at.button_group(key=f"mx-select-{thread.thread_id}").options)
