"""Settings terminology (Agent Evidence + Research Provenance acceptance
pass, task spec sections 7/7A/7B/7C/17): the normal Settings/Architecture
surface must read as a professional product, not a development changelog --
no visible "Phase 13" / "Phase 14" / "Phase 16" / "Phase 17" labels. The same
internal names may still exist, but only under a collapsed "Developer
Details" surface.

`st.tabs` cannot be preselected programmatically, so a bare `render()` (no
`settings_landing_focus`) already renders every tab's body into the DOM
regardless of which tab is visually active -- the same property
`system.py`'s own module docstring documents and the pre-existing "2025
STATUS"/"Agent Runtime" regression tests already rely on.
"""
from __future__ import annotations

import pytest

_BANNED_PHASE_LABELS = ("Phase 13", "Phase 14", "Phase 16", "Phase 17")


def _fresh_settings():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.system import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


def _dev_details_values(at) -> set[str]:
    dev_details = next(e for e in at.expander if e.label == "Developer Details")
    return {m.value for m in dev_details.markdown} | {c.value for c in dev_details.caption}


def _normal_text(at) -> str:
    """Everything OUTSIDE the "Developer Details" expander -- excludes by
    exact value match against that expander's own elements (a plain string
    `.replace()` is not reliable here: `at.markdown`/`at.caption` walk the
    WHOLE tree, so the dev-details text is already a substring of the joined
    full text, but re-joining separately can reorder/re-space it)."""
    dev_values = _dev_details_values(at)
    parts = [m.value for m in at.markdown if m.value not in dev_values]
    parts += [c.value for c in at.caption if c.value not in dev_values]
    return " ".join(parts)


def test_normal_settings_surface_has_no_phase_numbers():
    at = _fresh_settings()
    assert any(e.label == "Developer Details" for e in at.expander), (
        "Developer Details expander must exist somewhere on Settings"
    )
    normal_text = _normal_text(at)
    for label in _BANNED_PHASE_LABELS:
        assert label not in normal_text, f"{label!r} leaked into the normal Settings surface"


def test_developer_details_still_carries_the_internal_phase_labels():
    """The internal names are not deleted -- they move under "Developer
    Details" (task spec section 7B), never disappear entirely."""
    at = _fresh_settings()
    dev_text = " ".join(_dev_details_values(at))
    for label in _BANNED_PHASE_LABELS:
        assert label in dev_text


def test_architecture_uses_professional_functional_names():
    at = _fresh_settings()
    normal_text = _normal_text(at)
    for name in (
        "Market Data", "Feature Engine", "Research Planner", "Hypothesis Spec",
        "Strategy Compiler", "Strategy Spec", "Quant Core", "Validation Engine",
        "Experiment Registry",
    ):
        assert name in normal_text
