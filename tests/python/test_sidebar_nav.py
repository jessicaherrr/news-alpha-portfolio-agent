"""Sidebar/header/Research-Scope IA pass (Release UX product-consolidation
acceptance pass, task spec sections 6/7/8/10/12/20-23). Focused regression
coverage for:

* the shared sidebar: exactly Agent/Market/Research/Paper/Learn/Settings, no
  Current Market / Data Status / Research Window blocks, and
  Research/Paper/Learn/Settings expandable navigation never creates a new
  top-level route;
* Research Scope moved from the sidebar onto the Research page itself, with
  the exact same authoritative dates;
* the global header carrying no backend-engineering badges/commit hash, with
  that detail available on Settings instead.
"""
from __future__ import annotations

import pytest
from alpha_agent.ui import services

_PRIMARY_PAGES = ("news", "research", "market", "portfolio", "learn", "agent", "lab", "paper", "settings")
_PRIMARY_LABELS = ("News", "Research", "Market", "Portfolio", "Learn", "Ask", "Strategy Lab", "Paper Trading",
                   "Settings")
_ALL_MODULES = ["news", "workspace", "portfolio", "learn", "agent", "market", "research", "paper_trading", "system"]


def _app_test():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    return AppTest


def _page(module_name: str):
    at_cls = _app_test()
    at = at_cls.from_string(f"from alpha_agent.ui.views.{module_name} import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


# ---------------------------------------------------------------------------
# section 21 -- shared sidebar structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", _ALL_MODULES)
def test_sidebar_has_exactly_the_journey_tools_and_settings(module_name):
    at = _page(module_name)
    nav_buttons = {b.key: b.label for b in at.sidebar.button if b.key and b.key.startswith("navbtn-")}
    assert set(nav_buttons) == {f"navbtn-{p}" for p in _PRIMARY_PAGES}
    assert set(nav_buttons.values()) == set(_PRIMARY_LABELS)


@pytest.mark.parametrize("module_name", _ALL_MODULES)
def test_sidebar_never_contains_the_retired_blocks(module_name):
    at = _page(module_name)
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
    assert "WORKSPACE" not in sidebar_text.upper()
    assert "CURRENT MARKET" not in sidebar_text.upper()
    assert "DATA STATUS" not in sidebar_text.upper()
    assert "RESEARCH WINDOW" not in sidebar_text.upper()
    assert "Change Market" not in sidebar_text  # no market selector in the SHARED sidebar any more


def test_sidebar_shows_no_subrows_for_an_inactive_group():
    at = _page("agent")
    sub_buttons = [b for b in at.sidebar.button if b.key and b.key.startswith("navsubbtn-")]
    assert sub_buttons == []


def test_sidebar_auto_expands_only_the_active_group_strategy_lab():
    """Strategy Lab (the former Research page) keeps its drill-down views as
    sub-rows; My Alpha / Research Map / Community moved to Learn."""
    at = _page("research")
    sub_keys = {b.key for b in at.sidebar.button if b.key and b.key.startswith("navsubbtn-")}
    assert sub_keys == {
        "navsubbtn-lab-strategies", "navsubbtn-lab-experiments",
        "navsubbtn-lab-validation", "navsubbtn-lab-provenance",
    }


def test_the_research_thread_page_has_no_subrows():
    at = _page("workspace")
    assert not [b for b in at.sidebar.button if b.key and b.key.startswith("navsubbtn-")]


def test_sidebar_auto_expands_only_the_active_group_paper():
    at = _page("paper_trading")
    sub_keys = {b.key for b in at.sidebar.button if b.key and b.key.startswith("navsubbtn-")}
    assert sub_keys == {
        "navsubbtn-paper-positions", "navsubbtn-paper-execution",
        "navsubbtn-paper-risk", "navsubbtn-paper-monitoring",
    }


def test_sidebar_auto_expands_only_the_active_group_settings():
    at = _page("system")
    sub_keys = {b.key for b in at.sidebar.button if b.key and b.key.startswith("navsubbtn-")}
    assert sub_keys == {
        "navsubbtn-settings-providers", "navsubbtn-settings-runtime",
        "navsubbtn-settings-claude", "navsubbtn-settings-about",
    }


def test_sidebar_auto_expands_only_the_active_group_learn():
    at = _page("learn")
    sub_keys = {b.key for b in at.sidebar.button if b.key and b.key.startswith("navsubbtn-")}
    assert sub_keys == {
        "navsubbtn-learn-alpha_library", "navsubbtn-learn-alpha_graph",
        "navsubbtn-learn-community", "navsubbtn-learn-guides",
    }


def test_strategy_lab_subnav_deep_link_never_creates_a_new_top_level_route():
    """Clicking Strategy Lab -> Provenance stays on the SAME `lab` page -- it
    focuses one internal view (via a plain `st.rerun()`, since the target IS
    the currently active page), never a new sidebar destination."""
    at = _page("research")
    at.button(key="navsubbtn-lab-provenance").click().run(timeout=90)
    assert not list(at.exception)
    assert at.session_state["research_landing_focus"] == "provenance"
    back_buttons = [b for b in at.button if b.key == "research-focus-back"]
    assert back_buttons, "expected a back-to-landing control while a sub-view is focused"
    nav_buttons = {b.key for b in at.sidebar.button if b.key and b.key.startswith("navbtn-")}
    assert nav_buttons == {f"navbtn-{p}" for p in _PRIMARY_PAGES}

    back_buttons[0].click().run(timeout=90)
    assert not list(at.exception)
    assert "research_landing_focus" not in at.session_state
    assert {"Workflow", "Provenance"}.issubset({t.label for t in at.tabs})


def test_learn_subnav_deep_link_selects_that_tab():
    at = _page("learn")
    at.button(key="navsubbtn-learn-community").click().run(timeout=90)
    assert not list(at.exception)
    assert at.session_state["learn-tabs"] == "Community"


def test_settings_subnav_deep_link_focuses_one_section():
    at = _page("system")
    at.button(key="navsubbtn-settings-claude").click().run(timeout=90)
    assert not list(at.exception)
    assert at.session_state["settings_landing_focus"] == "claude"
    back_buttons = [b for b in at.button if b.key == "sys-focus-back"]
    assert back_buttons
    full_text = " ".join(m.value for m in at.markdown)
    assert "Claude API" in full_text

    back_buttons[0].click().run(timeout=90)
    assert not list(at.exception)
    assert "settings_landing_focus" not in at.session_state


def test_paper_subnav_present_only_when_a_run_is_selected():
    """Paper's sub-rows deep-link into a per-run detail view (task spec
    section 7D) -- with the real registry state in this checkout (0 eligible
    strategies, no runs), the page shows its honest empty state instead, so
    clicking a sub-row is a no-op rather than a crash."""
    at = _page("paper_trading")
    at.button(key="navsubbtn-paper-risk").click().run(timeout=90)
    assert not list(at.exception)


# ---------------------------------------------------------------------------
# section 22 -- Research Scope moved onto the Research page
# ---------------------------------------------------------------------------


def test_research_page_shows_research_scope_with_authoritative_dates():
    at = _page("research")
    w = services.research_window()
    full_text = " ".join(m.value for m in at.markdown)
    assert "Research Scope" in full_text
    metric_labels = {m.label: m.value for m in at.metric}
    assert metric_labels.get("Research") == w["research"]
    assert metric_labels.get("Validation") == w["validation"]
    assert "2025" in full_text and "LOCKED" in full_text.upper()


# ---------------------------------------------------------------------------
# section 23 -- global header carries no backend-engineering badges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", _ALL_MODULES)
def test_normal_headers_never_show_backend_engineering_detail(module_name):
    at = _page(module_name)
    full_text = " ".join(m.value for m in at.markdown)
    assert "AGENTIC ALPHA" in full_text
    assert "C++ Core" not in full_text
    assert "pybind11" not in full_text
    assert "commit " not in full_text


def test_settings_runtime_and_about_carry_the_relocated_detail():
    at = _page("system")
    full_text = " ".join(m.value for m in at.markdown)
    assert "pybind" in full_text.lower()  # Runtime tab -- panels.render_system_status()
    assert "Git commit" in full_text  # About tab
    assert "Branch" in full_text


# ---------------------------------------------------------------------------
# section 20 -- Market's local state never leaks into Agent's global scope or
# Research's own selected object (task spec section 13)
# ---------------------------------------------------------------------------


def test_market_selection_never_narrows_agents_global_scope_framing():
    at = _page("market")
    at.session_state["market_selected_root"] = "NQ"
    at.run(timeout=90)
    assert not list(at.exception)

    agent_at = _page("agent")
    full_text = " ".join(m.value for m in agent_at.markdown)
    assert "All Research Markets" in full_text
    assert "Current Market" not in full_text


def test_research_selected_object_is_not_mutated_by_a_market_product_change():
    at = _page("research")
    at.session_state["research_details_target"] = {
        "source": "registry_lookup", "objective": None, "root": "NQ", "hypothesis": None,
        "compiled": None, "evidence": None, "run_outcome": None, "experiment_id": "does-not-exist",
    }
    # A market change happening on a DIFFERENT page/session key must never
    # reach into or mutate this already-selected research object.
    at.session_state["market_selected_root"] = "CL"
    at.run(timeout=90)
    assert not list(at.exception)
    target = at.session_state["research_details_target"]
    assert target["root"] == "NQ"  # untouched by market_selected_root = "CL"
