"""Market Intelligence + Futures Universe campaign -- regression test for a
real bug caught while building the Market Scanner (Checkpoint A): the
market-selection validity guard used to check membership in the RESEARCH
universe (`services.approved_universe()`, 5 roots) even though the selected
product is a MARKET UNIVERSE concept the Scanner can set to any of 36+
catalogued products. That mismatch meant selecting a catalogued-but-
uncertified product (e.g. NG) would get silently reset back to a research
root on the very next render -- which the Scanner would then try to
reselect, which the guard would reset again: an actual rerun loop in a real
browser, not just a hypothetical.

Sidebar IA pass (Release UX product-consolidation acceptance pass, task spec
section 1): this guard now lives on `views/market.py`
(`_get_or_init_selected_root`), Market's own local state -- the shared
sidebar no longer has a market selector at all. See that function's
docstring for the fix this test still covers.
"""
from __future__ import annotations

import pytest


def _market_app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")


def test_market_page_does_not_reset_a_valid_non_research_market_universe_root():
    at = _market_app()
    at.session_state["market_selected_root"] = "NG"  # catalogued (Energy), not in the research universe
    at.run(timeout=90)
    assert not list(at.exception)
    assert at.session_state["market_selected_root"] == "NG"


def test_market_page_resets_an_uncatalogued_root_to_a_research_default():
    from alpha_agent.ui import services

    at = _market_app()
    at.session_state["market_selected_root"] = "NOT_A_REAL_PRODUCT"
    at.run(timeout=90)
    assert not list(at.exception)
    assert at.session_state["market_selected_root"] in services.approved_universe()


def test_market_page_still_defaults_a_fresh_session_to_a_research_root():
    from alpha_agent.ui import services

    at = _market_app()
    at.run(timeout=90)
    assert not list(at.exception)
    assert at.session_state["market_selected_root"] in services.approved_universe()
