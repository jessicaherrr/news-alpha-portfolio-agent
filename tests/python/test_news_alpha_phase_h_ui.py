"""News Alpha Phase H -- the validation gate and path studies in a Research
Thread's Backtest step, and route memory in its Learn step (real
`streamlit.testing.v1.AppTest` renders; migrated from the retired Agent-page
triage).

Screens and snapshots are the Phase F / G fixtures saved into the per-test
caches (conftest). The registry the page reads is a per-test temporary file,
so recorded memory is exercised without touching the real registry. The page
never starts a validation run and never writes the registry.
"""
from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import alpha_agent.features.compute  # noqa: F401 -- register feature defs
import pytest
from alpha_agent.alpha_memory.signal_path_evidence import EventResearch, build_signal_path_evidence
from alpha_agent.news_alpha import (
    MandateDomain,
    ResearchMandate,
    UserDescribedEvent,
    build_asset_expressions,
    build_candidate_signals,
    build_economic_mechanism_graph,
    discover_signal_paths,
    resolve_allowed_universe,
    scan_initial_impact,
)
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.ui import news_alpha_context, research_loop_view, research_thread, services
from alpha_agent.ui.research_thread import ThreadStep
from news_alpha_thread_support import thread_at
from test_news_alpha_phase_f import AI_CAPEX, ALL_ASSESSABLE, CAPS
from test_news_alpha_phase_f_ui import (  # noqa: F401 -- fixtures
    _body,
    _isolated,
    _no_live_llm,
    _tables,
)
from test_news_alpha_phase_g import CLI
from test_news_alpha_phase_g_ui import _store_screens, _store_snapshots

needs_cpp = pytest.mark.skipif(CLI is None, reason="quant_portfolio_construct_csv is not built")
VIEW = Path(__file__).resolve().parents[2] / "python" / "alpha_agent" / "ui" / "research_loop_view.py"
ETF_SHORTING = ResearchMandate(allowed_domains=(MandateDomain.ETF,), shorting_allowed=True)
MIXED_SHORTING = ALL_ASSESSABLE.model_copy(update={"shorting_allowed": True})


@pytest.fixture(autouse=True)
def _temp_registry(monkeypatch, tmp_path):
    path = tmp_path / "registry.sqlite"
    ExperimentRegistry(path).close()
    monkeypatch.setattr(services, "REGISTRY_PATH", path)
    return path


def _backtest(text: str = AI_CAPEX):
    return thread_at(text, ThreadStep.BACKTEST)


def test_without_screen_support_the_gate_says_no_eligible_portfolio_and_reads_nothing():
    news_alpha_context.MANDATE_STORE.save(ALL_ASSESSABLE)
    _store_screens(supported=False)
    at = _backtest()
    body = _body(at)
    assert "Not testable yet" in body and "2025 holdout sealed" in body
    assert "No eligible portfolio -- The plan was not constructed (NO_ELIGIBLE_SIGNAL)" in body
    tables = _tables(at).lower()
    assert "plan not constructed" in tables and "not yet testable" in tables
    assert "Nothing from the 2023-2024 validation window was read" in body
    assert "No backtest has run for this portfolio" in body  # never an estimated performance

    learn = thread_at(AI_CAPEX, ThreadStep.LEARN)
    text = _body(learn)
    assert "Computed now from this thread -- not recorded" in text
    assert "No run of these routes is recorded yet." in text


def test_the_gate_waits_for_the_qualified_plans_market_data():
    news_alpha_context.MANDATE_STORE.save(ALL_ASSESSABLE)
    _store_screens(supported=True)
    body = _body(_backtest())
    assert "Market data not loaded" in body
    assert "The qualified plan needs market data for 4 instrument(s)" in body


@needs_cpp
def test_an_eligible_single_domain_book_is_run_from_the_command_line_never_the_page():
    news_alpha_context.MANDATE_STORE.save(ETF_SHORTING)
    _store_screens(supported=True)
    _store_snapshots()
    at = _backtest()
    assert "Eligible -- not run yet" in _body(at)
    assert any(research_loop_view.CLI_COMMAND in c.value for c in at.code)
    assert not any("valid" in (b.label or "").lower() for b in at.button)  # no run button on the page


@needs_cpp
def test_a_mixed_book_is_shown_with_its_remaining_trading_day_blocker():
    news_alpha_context.MANDATE_STORE.save(MIXED_SHORTING)
    _store_screens(supported=True)
    _store_snapshots()
    at = _backtest()
    text = _body(at) + _tables(at)
    assert "Execution not supported yet" in text
    assert "trading-day conventions" in text and "one commission per unit" not in text


def test_a_new_event_recalls_the_recorded_memory_of_its_routes(_temp_registry):
    """Memory is recalled by ROUTE, not by event: an earlier occurrence of the
    same event (another timestamp, another event id) is what the Learn step shows."""
    news_alpha_context.MANDATE_STORE.save(ALL_ASSESSABLE)
    _store_screens(supported=False)
    at = thread_at(AI_CAPEX, ThreadStep.LEARN)
    thread = research_thread.THREAD_STORE.get(at.thread_id)
    universe = resolve_allowed_universe(ALL_ASSESSABLE, capabilities=CAPS)
    earlier = UserDescribedEvent.create(AI_CAPEX, described_at=datetime(2026, 1, 5, 9, 0, tzinfo=UTC))
    assert earlier.event_id != thread.event.event_id
    scan = scan_initial_impact(earlier, ALL_ASSESSABLE, universe=universe)
    paths = discover_signal_paths(build_economic_mechanism_graph(scan))
    expressions = build_asset_expressions(paths, universe, ALL_ASSESSABLE)
    research = EventResearch(discovery=paths, expressions=expressions,
                             candidates=build_candidate_signals(expressions, paths))
    records = build_signal_path_evidence([research], ranked=None, screens={}, plan=None, validation=None,
                                         recorded_at="2026-09-27T00:00:00+00:00")
    with ExperimentRegistry(_temp_registry) as reg:
        assert reg.record_signal_path_evidence(records) == len(records) > 0
    at.run(timeout=120)
    assert not list(at.exception)
    assert "No run of these routes is recorded yet." not in _body(at)
    assert "Seen in" in _body(at)  # recorded route cards


def test_the_research_loop_view_is_a_pure_renderer():
    source = VIEW.read_text(encoding="utf-8")
    for token in ("sorted(", ".sort(", "max(", "min(", "round(", "validate_portfolio", "close_research_loop",
                  "record_signal_path_evidence", "apply_bundle", "ExperimentRegistry(", "run_schedule"):
        assert token not in source, token
    tree = ast.parse(source)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Div))]
