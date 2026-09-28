"""Research -- the persistent Research Thread workspace.

One research object, progressively enriched: News -> Reasoning -> Market ->
Signals -> Portfolio -> Backtest -> Learn, with the Research Setup as
persistent context and a Research Context panel beside every step. Not a
rigid wizard: completed steps stay one click away, and a thread can be opened
directly at any step (`?thread=<id>&step=<step>`, or an in-app hand-off via
`research_thread.deep_link`).

Every value shown comes from the canonical backend through
`research_thread.ThreadPipeline`; this page orchestrates navigation only.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.ui import (
    components,
    layout,
    news_alpha_context,
    research_setup_view,
    research_thread,
)
from alpha_agent.ui.research_thread import STEP_LABELS, STEPS, ThreadStep
from alpha_agent.ui.workspace import (
    shell,
    step_backtest,
    step_learn,
    step_market,
    step_news,
    step_portfolio,
    step_reasoning,
    step_signals,
)
from alpha_agent.ui.workspace.common import ago, esc

PAGE_TITLE = "Research"

_STEP_RENDERERS = {
    ThreadStep.NEWS: step_news.render,
    ThreadStep.REASONING: step_reasoning.render,
    ThreadStep.MARKET: step_market.render,
    ThreadStep.SIGNALS: step_signals.render,
    ThreadStep.PORTFOLIO: step_portfolio.render,
    ThreadStep.BACKTEST: step_backtest.render,
    ThreadStep.LEARN: step_learn.render,
}
#: The last (thread, step) this page wrote to / applied from the URL -- so a
#: stale query string never overrides an in-app switch, while a pasted or
#: bookmarked link still opens exactly where it points.
_QP_APPLIED_KEY = "research_thread_qp_applied"


def open_thread(thread_id: str, step: ThreadStep | None = None) -> None:
    """The in-app deep link into a thread (from News, My Alpha, a signal, a
    past experiment): activates it, optionally at ``step`` with the earlier
    lineage marked complete, and opens this page."""
    thread = research_thread.THREAD_STORE.get(thread_id)
    if thread is None:
        return
    research_thread.set_active(thread_id)
    if step is not None and step is not thread.current_step:
        research_thread.deep_link(thread, step)
    st.switch_page(st.Page(render, url_path="research"))


def _apply_query_params() -> None:
    qp_thread = st.query_params.get("thread")
    qp_step = st.query_params.get("step")
    if not qp_thread or st.session_state.get(_QP_APPLIED_KEY) == (qp_thread, qp_step):
        return
    st.session_state[_QP_APPLIED_KEY] = (qp_thread, qp_step)
    thread = research_thread.THREAD_STORE.get(qp_thread)
    if thread is None:
        return
    research_thread.set_active(thread.thread_id)
    step = next((s for s in STEPS if s.value == qp_step), None)
    if step is not None and step is not thread.current_step:
        research_thread.deep_link(thread, step)


def _write_query_params(thread: research_thread.ResearchThread) -> None:
    pair = (thread.thread_id, thread.current_step.value)
    st.session_state[_QP_APPLIED_KEY] = pair
    if (st.query_params.get("thread"), st.query_params.get("step")) != pair:
        st.query_params["thread"], st.query_params["step"] = pair


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active="research")
    layout.render_header(subtitle="One research thread, from news to evidence.")

    mandate = news_alpha_context.current_mandate()
    _apply_query_params()
    thread = research_thread.active_thread()
    if thread is None:
        _render_thread_list()
        research_setup_view.render_setup_dialog_if_open(mandate)
        layout.render_disclaimer()
        return

    pipe = research_thread.pipeline_for(thread, mandate)
    thread, notice = research_thread.reconcile_setup(thread, mandate, pipe.expressions)
    pipe.thread = thread
    _write_query_params(thread)

    shell.render_thread_header(thread, mandate)
    research_setup_view.render_setup_dialog_if_open(mandate)
    if notice:
        st.warning(notice, icon=":material/sync_problem:")
    shell.render_stepper(thread)

    main, rail = layout.main_rail_columns()
    with main:
        _STEP_RENDERERS[thread.current_step](pipe)
        shell.render_step_nav(thread)
    with rail:
        shell.render_context_panel(pipe)
    layout.render_disclaimer()


def _render_thread_list() -> None:
    st.markdown('<div class="aa-page-title">Research</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aa-page-lede">A research thread follows one event from what happened to what it taught us: '
        "the economic reasoning, the markets that could express it, measurable signals, a portfolio, and a real "
        "backtest. Start one from News -- every thread is saved as you go.</div>",
        unsafe_allow_html=True,
    )
    threads = research_thread.THREAD_STORE.list()
    if st.button("Find something to research", key="research-go-news", type="primary", icon=":material/newspaper:",
                 help="Open News: recent releases and events, each read against your Research Setup."):
        from alpha_agent.ui.views import news

        st.switch_page(st.Page(news.render, url_path="news"))
    if not threads:
        components.empty_state(
            "No research threads yet",
            "Pick an event on News and choose Research -- or describe one there. Nothing is researched on its own.",
            key="research-no-threads",
        )
        return
    st.markdown('<div class="aa-subhead">Your research threads</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, t in enumerate(threads):
        with cols[i % 2], components.card(f"research-thread-{t.thread_id}"):
            step_no = STEPS.index(t.current_step) + 1
            st.markdown(
                f'<div class="aa-eyebrow">Step {step_no} of {len(STEPS)} · {esc(STEP_LABELS[t.current_step])}</div>'
                f'<div class="aa-news-headline">{esc(t.name)}</div>'
                f'<div class="aa-news-summary">{esc(t.event.headline)}</div>'
                f'<div class="aa-news-meta" style="margin-top:0.35rem">Updated {esc(ago(t.updated_at))}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Open", key=f"research-open-{t.thread_id}", icon=":material/arrow_forward:",
                         icon_position="right", help="Continue this thread where you left off."):
                research_thread.set_active(t.thread_id)
                st.rerun()
