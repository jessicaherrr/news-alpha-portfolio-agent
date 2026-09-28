"""Navigation consolidation. Updated for the News Alpha Research Thread
workspace: the research journey is the primary navigation -- News (default),
Research (the thread workspace), Market (Affected Markets + Explore All
Markets -- ONE market destination), Portfolio, Learn -- followed by three
Tools (Ask, Strategy Lab, Paper Trading) and one Settings entry. Every
former top-level page (Discover, Dashboard, Strategies, Backtests,
Validation, Experiment Log, Crypto Lab) must stay reachable -- its module and
`render()` are unchanged -- but never appear as a sidebar nav item.
"""
from __future__ import annotations

import re

import pytest
from alpha_agent.ui import services

APP_PATH = services.REPO_ROOT / "python" / "alpha_agent" / "ui" / "app.py"

_HIDDEN_URL_PATHS = (
    "discover", "dashboard", "strategies", "backtests", "validation", "experiment-log", "crypto-lab",
)
_PRIMARY_URL_PATHS = ("news", "research", "market", "portfolio", "learn")
_TOOL_URL_PATHS = ("agent", "lab", "paper-trading")


def _app_source() -> str:
    return APP_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# static structure -- exact page counts/visibility, no live Streamlit needed
# ---------------------------------------------------------------------------


def test_the_journey_and_tool_destinations_are_registered_visible():
    text = _app_source()
    for url_path in (*_PRIMARY_URL_PATHS, *_TOOL_URL_PATHS):
        pattern = rf'st\.Page\([^)]*url_path="{url_path}"[^)]*\)'
        match = re.search(pattern, text)
        assert match, f"expected a registered st.Page for url_path={url_path!r}"
        assert "visibility=" not in match.group(0), f"{url_path!r} must stay visible (a primary destination)"


def test_settings_is_the_only_other_visible_entry():
    text = _app_source()
    match = re.search(r'st\.Page\([^)]*url_path="system"[^)]*\)', text)
    assert match
    assert "visibility=" not in match.group(0)


def test_legacy_pages_are_registered_hidden_not_deleted():
    text = _app_source()
    for url_path in _HIDDEN_URL_PATHS:
        pattern = rf'st\.Page\([^)]*url_path="{url_path}"[^)]*\)'
        match = re.search(pattern, text)
        assert match, f"expected {url_path!r} to still be a registered (hidden) page, not removed"
        assert 'visibility="hidden"' in match.group(0), f"{url_path!r} must be visibility=hidden, not a nav item"


def test_exactly_five_journey_pages_three_tools_and_settings():
    text = _app_source()
    all_pages = re.findall(r"st\.Page\(([^)]*)\)", text)
    visible = [p for p in all_pages if "visibility=" not in p]
    hidden = [p for p in all_pages if 'visibility="hidden"' in p]
    assert len(visible) == 9, f"expected 9 visible pages (5 journey + 3 tools + Settings), got {len(visible)}"
    assert len(hidden) == len(_HIDDEN_URL_PATHS)


def test_news_is_the_default_landing_page():
    text = _app_source()
    match = re.search(r'st\.Page\([^)]*url_path="news"[^)]*\)', text)
    assert match and "default=True" in match.group(0)
    assert text.count("default=True") == 1


def test_settings_and_primary_pages_share_one_navigation_section():
    """Product UI Polish pass, section 2: a prior version put Settings alone
    in its own dict group keyed "Settings", which rendered a bold section
    header reading "Settings" directly above the Settings page link -- a
    real visible duplicate. The fix keeps Settings in the SAME "Agentic
    Alpha" list as the primary pages (as the last entry -- sixth since Phase
    4 added Learn) so there is only ever one section header in the whole
    sidebar; `layout.py` demotes it visually via `li:last-of-type` CSS. This
    is a static proof that Settings and Agent/Market/Research/Paper/Learn are
    declared in the same list literal, not a spot check of that CSS."""
    text = _app_source()
    agentic_alpha_start = text.index('"Agentic Alpha": [')
    next_key_start = text.index('"": [')
    assert next_key_start > agentic_alpha_start
    primary_block = text[agentic_alpha_start:next_key_start]
    for url_path in (*_PRIMARY_URL_PATHS, *_TOOL_URL_PATHS, "system"):
        assert f'url_path="{url_path}"' in primary_block, (
            f"expected {url_path!r} inside the SAME 'Agentic Alpha' page list as the other primary destinations"
        )
    assert '"Settings": [' not in text, "a second dict key literally named 'Settings' would reintroduce the bug"


# ---------------------------------------------------------------------------
# live app -- exactly one visible "Settings" sidebar entry (section 19)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present")
def test_sidebar_shows_settings_exactly_once():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    # Every st.Page title this app registers, in declaration order -- proves
    # "Settings" is declared exactly once across the whole page set (a
    # second, hidden "Settings" page would also be a bug, not just a second
    # VISIBLE one).
    titles = re.findall(r'title="([^"]+)"', _app_source())
    assert titles.count("Settings") == 1


# ---------------------------------------------------------------------------
# live app -- the real entrypoint boots on Agent by default. Cross-page
# hand-offs (Market's "Discover Strategies" button, Agent's "View Research
# Details", etc.) already have their own dedicated regression tests
# (`test_market_product_detail.py`, `test_agent_to_research_navigation.py`)
# built on this codebase's own established small-multipage-script fixture --
# `streamlit.testing.v1.AppTest.switch_page()` only matches FILE-based
# `st.Page(...)`, and every page in this app is callable-sourced (see
# `test_market_product_detail.py::_multipage_script`'s own docstring), so
# this file does not duplicate that mechanism; it only proves the real
# entrypoint's page LIST matches what the static tests above assert.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present")
def test_app_boots_on_the_news_page_by_default():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    markdown_text = " ".join(md.value for md in at.markdown)
    assert "AGENTIC ALPHA" in markdown_text or "Agentic Alpha" in markdown_text
    assert "What is worth researching?" in markdown_text
