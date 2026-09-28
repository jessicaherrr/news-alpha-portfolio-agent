"""News Alpha Phase E -- candidate signals and factor diagnostics in a Research
Thread's Signals step (real `streamlit.testing.v1.AppTest` render; migrated
from the retired Agent-page triage card), and in the Agent chat's reply for a
described event.

The capability snapshot is pinned. Screens are real: the button and
stored-screen tests run on the already-acquired ETF bars (an ETF-only
mandate keeps them to seconds) and skip when the raw store is absent. The
screen cache is a per-test temporary store (conftest).
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

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
    SignalHorizon,
    UserDescribedEvent,
    build_asset_expressions,
    build_candidate_signals,
    build_economic_mechanism_graph,
    discover_signal_paths,
    resolve_allowed_universe,
    scan_initial_impact,
)
from alpha_agent.screening.candidate_signal_screen import screen_candidate
from alpha_agent.ui import (
    candidate_signal_view,
    conversation_engine,
    market_intel_context,
    news_alpha_context,
    services,
)
from alpha_agent.ui.conversation_engine import ConversationBoundContext
from alpha_agent.ui.research_thread import ThreadStep
from news_alpha_thread_support import body, thread_at

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_ETF_RAW = REPO_ROOT / "data" / "raw" / "databento" / "ARCX.PILLAR"
needs_etf_bars = pytest.mark.skipif(not _ETF_RAW.exists(), reason="acquired ETF raw bars not present")

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
ALL_ASSESSABLE = ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES, MandateDomain.ETF))
ETF_ONLY = ResearchMandate(allowed_domains=(MandateDomain.ETF,))


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
        raise AssertionError("candidate signals must never construct a live LLM client")

    monkeypatch.setattr(translation_context, "AnthropicClient", _boom)


def _signals():
    return thread_at(AI_CAPEX, ThreadStep.SIGNALS)


def _body(at) -> str:
    return body(at)


def _table_values(at) -> str:
    frames = [df.value for df in at.dataframe] + [t.value for t in at.table]
    return " ".join(str(v) for f in frames for v in f.to_numpy().ravel())


def _candidates(mandate: ResearchMandate):
    universe = resolve_allowed_universe(mandate, capabilities=CAPS)
    scan = scan_initial_impact(UserDescribedEvent.create(AI_CAPEX), mandate, universe=universe)
    paths = discover_signal_paths(build_economic_mechanism_graph(scan))
    return build_candidate_signals(build_asset_expressions(paths, universe, mandate), paths)


def test_signals_step_shows_candidates_with_their_exact_formula_and_lineage():
    news_alpha_context.MANDATE_STORE.save(ALL_ASSESSABLE)
    at = _signals()
    body_text = _body(at)
    assert ("Candidate signals: 8 testable factor hypotheses on NQ, QQQ, XLK, XLU (e.g. NQ ts_return(close, 20) → "
            "20D) -- none screened yet on 2018-2022 discovery data.") in body_text
    tables = _table_values(at)
    for value in ("NQ 20-day price momentum → 20D", "ts_return(close, 20)", "ts_return(close, 60)",
                  "Candidate · not screened", "Positive", "Cross-sector", "Supply chain", "Holds exposed names",
                  "registered feature return {'n': 20}", "Expected relationship", "Continuation",
                  "Formation lookback", "Prediction horizon", "NOT from the paths' economic transmission lag",
                  "Confirmation input, not directional"):
        assert value in tables, value
    # "Why this signal?" (each card's Lineage): the full chain, every originating path kept.
    for stage in ("1. **News / event**", "2. **Mechanism**", "3. **Signal paths** (9)", "4. **Economic consequence**",
                  "5. **Asset expression**", "6. **Measurement**", "7. **Factor expression** -- `ts_return(close, 20)`"):
        assert stage in body_text, stage
    # Every card shows the exact formula that runs, as code.
    codes = [c.value for c in at.code]
    assert len(codes) == 8 and set(codes) == {"ts_return(close, 20)", "ts_return(close, 60)"}
    assert "Candidate · not screened yet" in body_text  # user-facing status, never a validation word
    assert any(b.label.startswith("Run factor diagnostics (8 unscreened)") for b in at.button)
    assert "Candidate signals are testable factor hypotheses -- not validated factors" in body_text


@needs_etf_bars
def test_a_stored_real_screen_shows_diagnostics_of_the_same_expression():
    news_alpha_context.MANDATE_STORE.save(ALL_ASSESSABLE)
    cset = _candidates(ALL_ASSESSABLE)
    (xlk,) = [c for c in cset.candidates if c.spec.instrument == "XLK" and c.spec.prediction_horizon is SignalHorizon.D20]
    screen = screen_candidate(xlk)  # real, already-acquired bars
    candidate_signal_view.SCREEN_STORE.save(screen)
    at = _signals()
    # The card's Diagnostics shows the SAME formula the candidate and the stored screen carry.
    assert screen.expression == xlk.expression == screen.diagnostics.expression
    assert [c.value for c in at.code].count(screen.expression) >= 2  # card + its diagnostics
    tables = _table_values(at)
    for value in ("Coverage", "Turnover", "Cost sensitivity", "Group concentration", "Forward return",
                  "factor-screen/1", "ARCX.PILLAR ohlcv-1d"):
        assert value in tables, value
    body_text = _body(at)
    assert "1 of 8 screened as unconditional factors" in body_text and "Not event-conditioned evidence" in body_text
    assert "Scope: unconditional factor · IC kind: time-series" in body_text
    headers = " ".join(str(c) for df in at.dataframe for c in df.value.columns)
    assert "TS Spearman IC" in headers and "TS Pearson IC" in headers and "Rank IC" not in headers
    assert "Rank IC" not in body_text


@needs_etf_bars
def test_the_run_button_screens_real_bars_and_the_result_persists():
    news_alpha_context.MANDATE_STORE.save(ETF_ONLY)  # ETF bars only: seconds, not minutes
    at = _signals()
    (run,) = [b for b in at.button if b.label.startswith("Run factor diagnostics")]
    run.click().run(timeout=180)
    assert not list(at.exception), list(at.exception)
    assert "6 of 6 screened as unconditional factors" in _body(at)
    assert not [b for b in at.button if b.label.startswith("Run factor diagnostics")]
    stored = candidate_signal_view.SCREEN_STORE.load_many(_candidates(ETF_ONLY))
    assert len(stored) == 6 and all(s.series.data_role.value == "REAL" for s in stored.values())
    # Phase F: the screened factors' correlations render in the ranking's redundancy view.
    assert any(t.label.startswith("Exposures & redundancy") for t in at.tabs)
    assert "Common days" in " ".join(str(c) for df in at.dataframe for c in df.value.columns)


def test_an_unseeded_event_shows_no_candidate_section():
    at = thread_at(GEOPOLITICAL, ThreadStep.SIGNALS)
    text = _body(at)
    assert "No candidate signal yet" in text
    assert "Candidate signals:" not in text


def test_a_chat_described_event_carries_the_same_candidate_signals():
    text = f"{AI_CAPEX}. What could I research?"
    classification = classify_intent(text, context=ConversationSignalContext(approved_roots=("ES", "NQ", "CL")))
    response = conversation_engine._handle_observation_to_factor(text, classification, ConversationBoundContext())
    assert ("CANDIDATE SIGNALS: 8 testable factor hypotheses -- NQ ts_return(close, 20) → 20D, NQ ts_return(close, "
            "60) → 60D, QQQ ts_return(close, 20) → 20D (+5 more); 0 screened") in response.text
    evidence = response.evidence["candidate_signals"]
    assert evidence["event_id"] == response.evidence["user_event_id"]
    assert evidence["fingerprint"].startswith("candsig1:") and len(evidence["candidates"]) == 8
    assert all(c["candidate_signal_id"].startswith("csig2:") for c in evidence["candidates"])
