"""News Alpha Phase G -- the portfolio plan in a Research Thread's Portfolio
step and across threads on the Portfolio page (real `streamlit.testing.v1.AppTest`
renders; migrated from the retired Agent-page triage).

Screens and market snapshots are the Phase F / G fixtures (in-memory bars
labelled REAL) saved into the per-test temporary caches (conftest). Plans are
sized by the real compiled C++ allocator; tests that need a sized plan skip
when it is not built.
"""
from __future__ import annotations

import ast
from pathlib import Path

import alpha_agent.features.compute  # noqa: F401 -- register feature defs
import pytest
from alpha_agent.news_alpha import MandateDomain
from alpha_agent.portfolio import EligibilityMode
from alpha_agent.screening.candidate_signal_screen import screen_candidate
from alpha_agent.screening.factor_diagnostics import ScreenStatus
from alpha_agent.ui import candidate_signal_view, news_alpha_context, portfolio_plan_view, services
from alpha_agent.ui.research_thread import ThreadStep
from news_alpha_thread_support import run_page, seed_thread, thread_at
from test_news_alpha_phase_f import AI_CAPEX, ALL_ASSESSABLE, OPEC_CUT, _candidates, _pinned
from test_news_alpha_phase_f_ui import (  # noqa: F401 -- fixtures
    _body,
    _isolated,
    _no_live_llm,
    _tables,
)
from test_news_alpha_phase_g import CLI, FRAMES, _snapshot

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)
needs_cpp = pytest.mark.skipif(CLI is None, reason="quant_portfolio_construct_csv is not built")

REPO_ROOT = Path(__file__).resolve().parents[2]
VIEW = REPO_ROOT / "python" / "alpha_agent" / "ui" / "portfolio_plan_view.py"
SHORTING = ALL_ASSESSABLE.model_copy(update={"shorting_allowed": True})


def _store_screens(*, supported: bool, text: str = AI_CAPEX) -> None:
    from test_news_alpha_phase_f import _loader

    loader = _loader(FRAMES)
    for c in _candidates(text).candidates:
        if c.spec.instrument not in FRAMES:
            continue
        screen = screen_candidate(c, loader=loader)
        if supported:
            screen = _pinned(screen, status=ScreenStatus.SCREEN_CONTINUE, t=2.5)
        candidate_signal_view.SCREEN_STORE.save(screen)


def _store_snapshots() -> None:
    for domain, symbol in ((MandateDomain.FUTURES, "NQ"), (MandateDomain.ETF, "QQQ"), (MandateDomain.ETF, "XLK"),
                           (MandateDomain.ETF, "XLU")):
        portfolio_plan_view.SNAPSHOT_STORE.save(_snapshot(domain, symbol))


def _describe(text: str):
    """The event's Research Thread at the Portfolio step."""
    return thread_at(text, ThreadStep.PORTFOLIO)


def test_without_screen_support_the_step_says_no_signal_qualifies():
    news_alpha_context.MANDATE_STORE.save(ALL_ASSESSABLE)
    _store_screens(supported=False)
    at = _describe(AI_CAPEX)
    body = _body(at)
    assert "Qualified plan: no signal can enter the portfolio -- 0 of 8 signal(s) eligible" in body
    assert "None has screening support yet." in body
    assert "No signal qualifies yet" in body
    assert "No screening support -- only signals whose screen says continue are qualified." in _tables(at)


def test_supported_signals_wait_for_market_data_behind_an_explicit_button():
    news_alpha_context.MANDATE_STORE.save(ALL_ASSESSABLE)
    _store_screens(supported=True)
    at = _describe(AI_CAPEX)
    assert "4 eligible instrument(s) need real market data before the plan can be sized" in _body(at)
    button = next(b for b in at.button if b.key and b.key.endswith("-load-go"))
    assert "about 1 minute(s) for futures" in button.label  # NQ is the one futures root


@needs_cpp
def test_cached_market_data_renders_the_cpp_sized_plan():
    news_alpha_context.MANDATE_STORE.save(SHORTING)
    _store_screens(supported=True)
    _store_snapshots()
    at = _describe(AI_CAPEX)
    body = _body(at)
    assert "Qualified plan as of 2022-12-26:" in body
    positions = next(df.value for df in at.dataframe if "Notional ($)" in df.value.columns)  # step + advanced tab
    assert set(positions["Side"]) <= {"Long", "Short"} and "Short" in set(positions["Side"])  # XLU trends down
    assert any(u.endswith("shares") for u in positions["Units"])
    tables = _tables(at)
    assert "Gross leverage" in tables and "default for unstated" in tables  # the mandate states no cap
    (handoff,) = [df.value for df in at.dataframe if "Target units" in df.value.columns]
    assert set(handoff["Root"]) >= {"EQQQ", "EXLU"}  # ETF roots are the synthetic E<ticker> execution roots
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["As of"] == "2022-12-26" and "Ex-ante vol / target" in metrics
    assert "not validation" in body and "Phase H" in body
    assert "Constraints" in body and "✓ Gross leverage" in body  # the checklist beside the tables


@needs_cpp
def test_the_exploratory_preview_is_an_explicit_labelled_choice():
    news_alpha_context.MANDATE_STORE.save(SHORTING)
    from test_news_alpha_phase_f import _loader

    loader = _loader(FRAMES)
    for c in _candidates().candidates:
        screen = _pinned(screen_candidate(c, loader=loader), status=ScreenStatus.NO_SCREEN_SUPPORT, t=0.9)
        candidate_signal_view.SCREEN_STORE.save(screen)
    _store_snapshots()
    at = _describe(AI_CAPEX)
    assert "No signal qualifies yet" in _body(at)
    control = next(c for c in at.segmented_control if c.key.startswith("thread-portfolio-mode-"))
    control.set_value(EligibilityMode.EXPLORATORY).run(timeout=90)
    assert not list(at.exception)
    assert "Exploratory preview as of 2022-12-26" in _body(at)
    assert any("not a qualified portfolio" in w.value for w in at.warning)


def test_two_threads_carry_one_cross_thread_portfolio():
    news_alpha_context.MANDATE_STORE.save(ALL_ASSESSABLE)
    _store_screens(supported=False)
    _store_screens(supported=False, text=OPEC_CUT)
    seed_thread(AI_CAPEX)
    seed_thread(OPEC_CUT)
    at = run_page("portfolio")
    body = _body(at)
    assert "Portfolio plan" in body and "Qualified plan: no signal can enter the portfolio" in body


def test_the_portfolio_view_never_sorts_sizes_or_constrains_itself():
    source = VIEW.read_text(encoding="utf-8")
    for token in ("sorted(", ".sort(", "key=lambda", "max(", "min(", "round("):
        assert token not in source, token
    tree = ast.parse(source)
    products = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult)]
    assert products == []
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "alpha_agent.portfolio" in imported and "alpha_agent.portfolio.allocator" not in imported
