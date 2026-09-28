"""Shared AppTest helpers for the News Alpha UI tests: the News page and a
Research Thread rendered at any step (real `streamlit.testing.v1.AppTest`
renders of the real pages). The thread store is isolated per test by
`conftest._research_thread_store_is_isolated`."""
from __future__ import annotations

import pytest
from alpha_agent.news_alpha import UserDescribedEvent
from alpha_agent.ui import news_alpha_context, research_thread, services
from alpha_agent.ui.research_thread import ThreadStep

#: The Research page inside a real `st.navigation` (like `app.py`, with the
#: workspace as the landing page), so in-app hand-offs (`st.switch_page` to
#: Learn, Ask, Markets ...) really switch pages.
_WORKSPACE_SCRIPT = """
import streamlit as st
st.session_state.setdefault("research_thread_active_id", {thread_id!r})
from alpha_agent.ui.views import agent, learn, market, news, paper_trading, portfolio, research, workspace
st.navigation([
    st.Page(workspace.render, title="Research", url_path="research", default=True),
    st.Page(news.render, title="News", url_path="news"),
    st.Page(portfolio.render, title="Portfolio", url_path="portfolio"),
    st.Page(learn.render, title="Learn", url_path="learn"),
    st.Page(agent.render, title="Ask", url_path="agent"),
    st.Page(market.render, title="Markets", url_path="market"),
    st.Page(research.render, title="Strategy Lab", url_path="lab"),
    st.Page(paper_trading.render, title="Paper Trading", url_path="paper-trading"),
], position="hidden").run()
"""


def app_test():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    return AppTest


def run_page(module: str, *, session: dict | None = None):
    at = app_test().from_string(f"from alpha_agent.ui.views.{module} import render\nrender()\n")
    for k, v in (session or {}).items():
        at.session_state[k] = v
    at.run(timeout=120)
    assert not list(at.exception), list(at.exception)
    return at


def describe_on_news(text: str):
    """The News page after describing ``text`` and scanning it."""
    at = run_page("news")
    at.text_input(key="news-describe-text").input(text)
    at.button(key="news-describe-go").click().run(timeout=120)
    assert not list(at.exception), list(at.exception)
    return at


def current_mandate():
    return news_alpha_context.MANDATE_STORE.load(risk_profile=services.load_investor_profile())


def seed_thread(event) -> research_thread.ResearchThread:
    """A saved thread for ``event`` (event text -> a user-described event, or a
    cached news item / scheduled event), created exactly as News creates one
    (`research_thread.create_thread` on the same scan)."""
    mandate = current_mandate()
    source = UserDescribedEvent.create(event) if isinstance(event, str) else event
    scan = news_alpha_context.scan_event_source(source, mandate)
    return research_thread.create_thread(scan, mandate)


def strategy_route(at):
    """Turn on the Signals step's "test as a single-market strategy" route."""
    at.toggle(key=f"thread-strategy-route-{at.thread_id}").set_value(True).run(timeout=120)
    assert not list(at.exception), list(at.exception)
    return at


def thread_at(text, step: ThreadStep = ThreadStep.NEWS, **changes):
    """The Research page with a thread for ``text`` (or an existing
    `ResearchThread`) open at ``step`` (earlier steps marked complete, like a
    deep link)."""
    thread = text if isinstance(text, research_thread.ResearchThread) else seed_thread(text)
    if changes:
        thread = research_thread.update_thread(thread, **changes)
    if step is not ThreadStep.NEWS:
        thread = research_thread.deep_link(thread, step)
    at = app_test().from_string(_WORKSPACE_SCRIPT.format(thread_id=thread.thread_id))
    at.run(timeout=180)
    assert not list(at.exception), list(at.exception)
    at.thread_id = thread.thread_id  # the seeded thread, for key lookups
    return at


def body(at) -> str:
    return " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)


def tables(at) -> str:
    return " ".join(str(v) for df in at.dataframe for v in df.value.to_numpy().ravel())
