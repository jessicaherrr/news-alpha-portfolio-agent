"""Phase 4 -- "Learn Why" contextual deep links (task spec section 4): DSR
fail -> Learn DSR; backwardation -> Learn term structure; next-bar execution
-> Learn look-ahead bias, and every other validation gate besides. One small
button on a professional surface that switches to Learn's Concepts tab with
the given concept already focused.

Mirrors `layout.render_sidebar_nav`'s own deep-link mechanism
(`st.session_state["learn_landing_focus"]` / `"learn_focus_concept"`) rather
than inventing a second navigation pattern.
"""
from __future__ import annotations

import re

import streamlit as st

from alpha_agent.ui import services

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _learn_target():
    from alpha_agent.ui.views.learn import render

    return render


def render_learn_why(concept_id: str, *, key: str, label: str = "Learn Why", already_on_learn_page: bool = False) -> None:
    """Renders nothing if ``concept_id`` is not a real, known Concept (never a
    dead link). ``already_on_learn_page=True`` (Learn's own My Research /
    Learning Paths tabs linking to Concepts) uses a plain `st.rerun()`
    instead of `st.switch_page` -- see `layout.render_sidebar_nav`'s own
    docstring: switching to the CURRENTLY active top-level page falls back to
    the app's default page instead of just re-rendering it."""
    if services.learn_concept(concept_id) is None:
        return
    slug = _SLUG_RE.sub("-", key).strip("-").lower()
    if st.button(f"[?] {label}", key=f"learn-why-{slug}", type="tertiary"):
        st.session_state["learn_landing_focus"] = "concepts"
        st.session_state["learn_focus_concept"] = concept_id
        if already_on_learn_page:
            st.rerun()
        else:
            st.switch_page(st.Page(_learn_target(), url_path="learn"))
