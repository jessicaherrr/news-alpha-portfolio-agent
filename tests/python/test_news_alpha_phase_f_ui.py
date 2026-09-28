"""News Alpha Phase F -- the signal ranking (real `streamlit.testing.v1.AppTest`
renders): per Research Thread (Signals step), across threads (the Portfolio
page), and in the Agent chat. Migrated from the retired Agent-page triage.

The capability snapshot is pinned and the screen cache is a per-test
temporary store (conftest). Stored screens come from the Phase F test
fixtures (in-memory bars labelled REAL); the run-button test screens the
already-acquired ETF bars and skips when the raw store is absent.
"""
from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import alpha_agent.features.compute  # noqa: F401 -- register feature defs
import pytest
import streamlit as st
from alpha_agent.agents.conversation import ConversationSignalContext, classify_intent
from alpha_agent.market_intel.store import EventStore, NewsStore
from alpha_agent.news_alpha import MandateDomain, ResearchMandate
from alpha_agent.recommendation.profile import DEFAULT_PROFILE, HoldingPeriod, InvestorProfile
from alpha_agent.recommendation.signal_ranking import rank_candidate_signals
from alpha_agent.ui import (
    candidate_signal_view,
    conversation_engine,
    market_intel_context,
    news_alpha_context,
    services,
)
from alpha_agent.ui.conversation_engine import ConversationBoundContext
from alpha_agent.ui.research_thread import ThreadStep
from news_alpha_thread_support import run_page, seed_thread, thread_at
from test_news_alpha_phase_f import AI_CAPEX, ALL_ASSESSABLE, CAPS, OPEC_CUT, _candidates, _screens

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VIEW = REPO_ROOT / "python" / "alpha_agent" / "ui" / "signal_ranking_view.py"
_ETF_RAW = REPO_ROOT / "data" / "raw" / "databento" / "ARCX.PILLAR"
ETF_ONLY = ResearchMandate(allowed_domains=(MandateDomain.ETF,))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    store = NewsStore(tmp_path / "news.sqlite")
    event_store = EventStore(tmp_path / "events.sqlite")
    monkeypatch.setattr(market_intel_context, "_store", lambda: store)
    monkeypatch.setattr(market_intel_context, "_event_store", lambda: event_store)
    monkeypatch.setattr(market_intel_context, "_last_refresh_wall", datetime.now(UTC), raising=False)
    monkeypatch.setattr(news_alpha_context, "_capabilities", lambda: CAPS)
    # Never the user's real saved profile (data/user_prefs/) -- fit columns must not depend on it.
    monkeypatch.setattr(services, "load_investor_profile", lambda: DEFAULT_PROFILE)
    for key in ("news_alpha_user_events", "news_alpha_mandate"):
        st.session_state.pop(key, None)
    yield
    for key in ("news_alpha_user_events", "news_alpha_mandate"):
        st.session_state.pop(key, None)


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch):
    from alpha_agent.ui import translation_context

    def _boom(*a, **kw):
        raise AssertionError("the signal ranking must never construct a live LLM client")

    monkeypatch.setattr(translation_context, "AnthropicClient", _boom)


def _describe(text: str):
    """The event's Research Thread at the Signals step, where its ranking lives."""
    return thread_at(text, ThreadStep.SIGNALS)


def _body(at) -> str:
    return " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)


def _tables(at) -> str:
    frames = [df.value for df in at.dataframe] + [t.value for t in at.table]
    return " ".join(str(v) for f in frames for v in f.to_numpy().ravel())


def _store_fixture_screens() -> None:
    for screen in _screens(_candidates()).values():
        candidate_signal_view.SCREEN_STORE.save(screen)


def test_an_unscreened_event_lists_every_signal_as_not_rankable_not_as_low():
    news_alpha_context.MANDATE_STORE.save(ALL_ASSESSABLE)
    at = _describe(AI_CAPEX)
    body = _body(at)
    assert "Signal ranking: Nothing ranked yet -- none of the 8 candidate signal(s) has comparable screening " \
           "evidence (8 not rankable yet). Missing evidence is not low quality." in body
    tables = _tables(at)
    assert "Not screened yet -- no diagnostics to rank on." in tables
    assert "Run the factor diagnostics (2018-2022 discovery data)." in tables
    assert "Not rankable (8)" in [t.label for t in at.tabs]


def test_stored_screens_rank_into_independent_exposures_with_reasons():
    news_alpha_context.MANDATE_STORE.save(ALL_ASSESSABLE)
    _store_fixture_screens()
    at = _describe(AI_CAPEX)
    body = _body(at)
    assert "Signal ranking: 8 ranked signal(s) = 2 independent exposure(s) from 1 event thesis." in body
    assert "How they rank for portfolio consideration" in body
    (ranked,) = [df.value for df in at.dataframe if "Why this rank" in df.value.columns]
    assert list(ranked["Rank"]) == list(range(1, 9))
    assert list(ranked["Role"][:2]) == ["Lead", "Lead"]
    assert all(r.startswith("Alt. of #") for r in ranked["Role"][2:])
    assert "NQ ts_return(close, 20) → 20D" in set(ranked["Signal"])  # the exact formula that runs
    assert set(ranked["Domain"]) == {"Futures", "ETF"}
    assert set(ranked["Your mandate"]) <= {"shorting constrained", "No conflict"}  # default mandate: no shorting
    # "Why this rank?": the six aspects, the uncertainty list and the raw metrics with their normalization.
    tables = _tables(at)
    for aspect in ("Why this rank", "Scientific quality", "Fit to your mandate", "Data quality", "Liquidity & cost",
                   "Redundancy"):
        assert aspect in tables, aspect
    assert "**Uncertainty**" in body and "not validated" in body
    assert "factor-screen-grades/1" in body and "never rescaled" in body
    tabs = [t.label for t in at.tabs]
    assert "Exposures & redundancy (2)" in tabs and "Excluded by mandate (0)" in tabs
    assert "Aligned rank correlation" in " ".join(str(c) for df in at.dataframe for c in df.value.columns)
    assert f"Event thesis -- {AI_CAPEX}: reaches 2 exposure(s)" in body
    assert "not portfolio weights" in body


def _ranked_table(at):
    (ranked,) = [df.value for df in at.dataframe if "Why this rank" in df.value.columns]
    return ranked


def test_the_current_profile_reaches_the_fit_column_without_reordering_merit(monkeypatch):
    news_alpha_context.MANDATE_STORE.save(ALL_ASSESSABLE)
    _store_fixture_screens()
    default = _ranked_table(_describe(AI_CAPEX))
    assert set(default["Your mandate"]) <= {"shorting constrained", "No conflict"}
    monkeypatch.setattr(services, "load_investor_profile",
                        lambda: InvestorProfile(holding_period=HoldingPeriod.ONE_TO_THREE_DAYS))
    personal = _ranked_table(_describe(AI_CAPEX))
    assert all("holding period mismatch" in cell for cell in personal["Your mandate"])  # 20D/60D vs 1-3 days
    assert list(personal["Signal"]) == list(default["Signal"])  # fit is shown beside merit, never reorders it
    assert list(personal["t (n/h)"]) == list(default["t (n/h)"])


def test_two_threads_rank_together_on_the_portfolio_page():
    news_alpha_context.MANDATE_STORE.save(ALL_ASSESSABLE)
    _store_fixture_screens()
    seed_thread(AI_CAPEX)
    seed_thread(OPEC_CUT)
    at = run_page("portfolio")
    body = _body(at)
    assert "Contributing threads" in body
    (threads,) = [df.value for df in at.dataframe if "Candidate signals" in df.value.columns]
    assert len(threads) == 2
    ai, opec = _candidates(), _candidates(OPEC_CUT)
    screens = {**candidate_signal_view.stored_screens(ai), **candidate_signal_view.stored_screens(opec)}
    expected = rank_candidate_signals((ai, opec), screens, ALL_ASSESSABLE)
    # One structural candidate reached by both events (NQ momentum) is one signal, ranked once.
    assert expected.family_size < len(ai.candidates) + len(opec.candidates)
    assert f"Signal ranking: {expected.headline()}" in body


def test_the_ranking_view_never_sorts_scores_or_groups_signals_itself():
    source = VIEW.read_text(encoding="utf-8")
    for token in ("sorted(", ".sort(", "key=lambda", "max(", "min("):
        assert token not in source, token
    imported = {n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom)}
    assert "alpha_agent.recommendation.signal_ranking" in imported
    assert "alpha_agent.screening.candidate_signal_screen" not in imported  # no screening, only the cached store


def test_a_chat_described_event_carries_the_signal_ranking():
    news_alpha_context.MANDATE_STORE.save(ALL_ASSESSABLE)
    _store_fixture_screens()
    text = f"{AI_CAPEX}. What could I research?"
    classification = classify_intent(text, context=ConversationSignalContext(approved_roots=("ES", "NQ", "CL")))
    response = conversation_engine._handle_observation_to_factor(text, classification, ConversationBoundContext())
    assert "SIGNAL RANKING: 8 ranked signal(s) = 2 independent exposure(s)" in response.text
    assert "Independent exposures: #1 " in response.text and "Not portfolio weights." in response.text
    evidence = response.evidence["signal_ranking"]
    assert evidence["independent_exposures"] == 2 and evidence["fingerprint"].startswith("ranked1:")
    assert [s["rank"] for s in evidence["signals"]] == list(range(1, 9))


@pytest.mark.skipif(not _ETF_RAW.exists(), reason="acquired ETF raw bars not present")
def test_running_real_etf_diagnostics_ranks_them_in_the_same_card():
    news_alpha_context.MANDATE_STORE.save(ETF_ONLY)  # ETF bars only: seconds, not minutes
    at = _describe(AI_CAPEX)
    (run,) = [b for b in at.button if b.label.startswith("Run factor diagnostics")]
    run.click().run(timeout=180)
    assert not list(at.exception), list(at.exception)
    body = _body(at)
    assert "Signal ranking: 6 ranked signal(s) = 2 independent exposure(s)" in body
    exposures = [df.value for df in at.dataframe if "Linked by" in df.value.columns]
    assert exposures and {"QQQ · XLK", "XLK · QQQ"} & set(exposures[0]["Exposure"])
