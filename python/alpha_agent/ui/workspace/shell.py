"""The Research Thread shell: header, progress stepper, the persistent
Research Context panel, and the previous/next continuation bar. Navigation
state only -- every value in the context panel is read from the thread's
canonical pipeline objects."""
from __future__ import annotations

import streamlit as st

from alpha_agent.news_alpha import DOMAIN_LABELS, ResearchMandate
from alpha_agent.portfolio import EligibilityMode, PlanStatus
from alpha_agent.ui import components, research_setup_view, research_thread
from alpha_agent.ui.research_thread import (
    STEP_LABELS,
    STEP_QUESTIONS,
    STEPS,
    ResearchThread,
    ThreadPipeline,
    ThreadStep,
)
from alpha_agent.ui.workspace.common import PATH_TYPE_TEXT, ago, esc, info_html

__all__ = [
    "NEXT_LABEL",
    "render_context_panel",
    "render_step_heading",
    "render_step_nav",
    "render_stepper",
    "render_thread_header",
]

_STEP_ICON = {
    "done": ":material/check_circle:",
    "current": ":material/radio_button_checked:",
    "todo": ":material/radio_button_unchecked:",
}

#: The continuation action at the bottom of each step, and what it promises.
NEXT_LABEL: dict[ThreadStep, tuple[str, str]] = {
    ThreadStep.NEWS: ("See why this matters", ("Next: the economic reasoning -- how this event could propagate. "
                                              "Causal reasoning, not a trade recommendation.")),
    ThreadStep.REASONING: ("Translate to markets", ("Next: which markets could express these consequences, and "
                                                   "what data could measure them.")),
    ThreadStep.MARKET: ("Build signals", ("Next: the quantitative candidate signals built from the measurable "
                                         "markets, and their factor diagnostics.")),
    ThreadStep.SIGNALS: ("Build portfolio", ("Next: how the ranked signals would be combined and sized under your "
                                            "Research Setup.")),
    ThreadStep.PORTFOLIO: ("Continue to backtest", ("Next: whether the portfolio is eligible for validation, and "
                                                   "the recorded C++ backtest and validation results.")),
    ThreadStep.BACKTEST: ("Save & learn", ("Saves this step and moves to what the thread taught us -- your "
                                          "research memory for these routes.")),
    ThreadStep.LEARN: ("Finish thread", ("Marks the thread complete. It stays saved; reopen it any time from News "
                                        "or Research.")),
}

_PLAN_STATUS_TEXT = {
    PlanStatus.CONSTRUCTED: "Built", PlanStatus.NO_EXECUTABLE_POSITION: "No executable position",
    PlanStatus.NO_ELIGIBLE_SIGNAL: "No eligible signal yet", PlanStatus.NO_ALLOCATABLE_SIGNAL: "Nothing allocatable",
    PlanStatus.DEGENERATE_RISK_MODEL: "Exposures hedge exactly", PlanStatus.RISK_INPUTS_UNAVAILABLE:
    "Risk inputs unavailable", PlanStatus.ALLOCATOR_UNAVAILABLE: "C++ allocator not built",
}


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------


def _switch_thread(thread_id: str) -> None:
    research_thread.set_active(thread_id)


def render_thread_header(thread: ResearchThread, mandate: ResearchMandate) -> None:
    left, right = st.columns([5, 1.4], vertical_alignment="bottom")
    with left:
        ev = thread.event
        st.markdown(
            '<div class="aa-eyebrow">Research thread</div>'
            f'<div class="aa-thread-title">{esc(thread.name)}</div>'
            f'<div class="aa-thread-sub">Source event: <b>{esc(ev.headline)}</b></div>',
            unsafe_allow_html=True,
        )
    with right, st.popover("All threads", icon=":material/forum:", width="stretch",
                           help="Switch to another research thread, or start a new one from News."):
        others = [t for t in research_thread.THREAD_STORE.list() if t.thread_id != thread.thread_id]
        if not others:
            st.caption("This is your only thread so far.")
        for t in others:
            if st.button(f"{t.name} · {STEP_LABELS[t.current_step]}", key=f"thread-switch-{t.thread_id}",
                         width="stretch", help=t.event.headline):
                _switch_thread(t.thread_id)
                st.rerun()
        if st.button("Start from News", key="thread-new-from-news", icon=":material/add:", type="tertiary"):
            from alpha_agent.ui.views import news

            st.switch_page(st.Page(news.render, url_path="news"))
    research_setup_view.render_setup_summary(mandate, key=f"thread-{thread.thread_id}")


# ---------------------------------------------------------------------------
# stepper
# ---------------------------------------------------------------------------


def render_stepper(thread: ResearchThread) -> None:
    with st.container(key="thread-stepper"):
        cols = st.columns(len(STEPS), gap="small")
        for i, (col, step) in enumerate(zip(cols, STEPS, strict=True), start=1):
            status = thread.status(step)
            reachable = thread.is_reachable(step)
            tip = f"Step {i} of {len(STEPS)} · {STEP_QUESTIONS[step]}"
            if not reachable:
                tip += " Reach the steps before it first."
            with col, st.container(key=f"step-{status}-{step.value}"):
                if st.button(STEP_LABELS[step], key=f"stepbtn-{step.value}", icon=_STEP_ICON[status],
                             disabled=not reachable, help=tip, width="stretch") and step is not thread.current_step:
                    research_thread.go_to(thread, step)
                    st.rerun()


def render_step_heading(step: ThreadStep, *, help: str | None = None) -> None:
    i = STEPS.index(step) + 1
    st.markdown(
        f'<div class="aa-step-head"><span class="aa-step-num">Step {i} of {len(STEPS)}</span>'
        f'<span class="aa-step-title">{esc(STEP_LABELS[step])}</span>{info_html(help) if help else ""}</div>'
        f'<div class="aa-step-q">{esc(STEP_QUESTIONS[step])}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# previous / next
# ---------------------------------------------------------------------------


def render_step_nav(thread: ResearchThread) -> None:
    i = STEPS.index(thread.current_step)
    with st.container(key="thread-nav", horizontal=True, vertical_alignment="center"):
        if i > 0:
            prev = STEPS[i - 1]
            if st.button(f"{STEP_LABELS[prev]}", key="thread-prev", icon=":material/arrow_back:",
                         help=f"Back to {STEP_LABELS[prev]}: {STEP_QUESTIONS[prev]}"):
                research_thread.go_to(thread, prev)
                st.rerun()
        st.space("stretch")
        label, tip = NEXT_LABEL[thread.current_step]
        done = thread.current_step is ThreadStep.LEARN and ThreadStep.LEARN in thread.completed_steps
        if st.button(label, key="thread-next", type="primary", icon=":material/arrow_forward:", icon_position="right",
                     help=tip, disabled=done):
            research_thread.advance(thread)
            if thread.current_step in (ThreadStep.BACKTEST, ThreadStep.LEARN):
                st.toast("Research thread saved.", icon=":material/check:")
            st.rerun()


# ---------------------------------------------------------------------------
# research context (right rail)
# ---------------------------------------------------------------------------


def _path_text(pipe: ThreadPipeline) -> str:
    followed = pipe.followed_paths()
    if not followed:
        return '<span class="aa-dim">All paths -- none followed yet</span>'
    first = followed[0]
    chain = " → ".join(esc(s) for s in first.state_labels)
    more = f' <span class="aa-dim">(+{len(followed) - 1} more)</span>' if len(followed) > 1 else ""
    return f"{esc(PATH_TYPE_TEXT[first.path_type])}: {chain}{more}"


def _market_text(pipe: ThreadPipeline) -> str:
    e = pipe.focus_expression()
    if e is None:
        return '<span class="aa-dim">Not chosen yet</span>'
    symbols = ", ".join(e.symbols)
    return f"{esc(e.concept)}" + (f" · {esc(symbols)}" if symbols else "") + \
        f' <span class="aa-dim">({esc(DOMAIN_LABELS[e.domain])})</span>'


def _signals_text(pipe: ThreadPipeline) -> str:
    thread = pipe.thread
    if thread.status(ThreadStep.SIGNALS) == "todo" and ThreadStep.SIGNALS not in thread.completed_steps:
        return '<span class="aa-dim">Not built yet</span>'
    n, k = len(pipe.candidates.candidates), len(pipe.screens)
    if n == 0:
        return '<span class="aa-dim">No measurable signal</span>'
    return f"{n} candidate{'s' if n != 1 else ''} · {k} screened"


def _portfolio_text(pipe: ThreadPipeline) -> str:
    if pipe.thread.status(ThreadStep.PORTFOLIO) == "todo":
        return '<span class="aa-dim">Not built yet</span>'
    plan, missing = pipe.plan(EligibilityMode.QUALIFIED)
    if plan is None:
        return f'<span class="aa-dim">Market data for {len(missing)} instrument(s) not loaded</span>'
    return esc(_PLAN_STATUS_TEXT.get(plan.status, plan.status.value))


def _validation_text(pipe: ThreadPipeline) -> str:
    if pipe.thread.status(ThreadStep.BACKTEST) == "todo":
        return '<span class="aa-dim">Not run yet</span>'
    from alpha_agent.ui.workspace import step_backtest

    return esc(step_backtest.status_label(pipe)[1])


def render_context_panel(pipe: ThreadPipeline) -> None:
    thread, mandate = pipe.thread, pipe.mandate
    rows = [
        ("Event", esc(thread.event.headline)),
        ("Assets", esc(" + ".join(DOMAIN_LABELS[d] for d in mandate.allowed_domains))),
        ("Horizon", esc(mandate.investment_horizon.value)),
        ("Selected path", _path_text(pipe)),
        ("Current market", _market_text(pipe)),
        ("Signals", _signals_text(pipe)),
        ("Portfolio", _portfolio_text(pipe)),
        ("Validation", _validation_text(pipe)),
    ]
    with components.card("thread-context"):
        st.markdown('<div class="aa-eyebrow" style="margin-bottom:0.2rem">Research context</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="aa-ctx">' + "".join(
                f'<div class="aa-ctx-row"><div class="aa-ctx-label">{label}</div>'
                f'<div class="aa-ctx-value">{value}</div></div>' for label, value in rows
            ) + "</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Saved automatically · updated {ago(thread.updated_at)}")
    with st.expander("Thread references", icon=":material/fingerprint:"):
        st.caption("The canonical objects this thread is built from. The thread stores references, never copies of "
                   "results.")
        components.provenance_row("Thread", thread.thread_id)
        components.provenance_row("Event", thread.event.event_id)
        components.provenance_row("Research Setup", mandate.fingerprint()[:28] + "…")
        if not pipe.graph.is_empty:
            components.provenance_row("Mechanism graph", pipe.graph.fingerprint()[:28] + "…")
            components.provenance_row("Signal paths", pipe.paths.fingerprint()[:28] + "…")
        if thread.status(ThreadStep.MARKET) != "todo":
            components.provenance_row("Asset expressions", pipe.expressions.fingerprint()[:28] + "…")
        if thread.status(ThreadStep.SIGNALS) != "todo" and not pipe.candidates.is_empty:
            components.provenance_row("Candidate signals", pipe.candidates.fingerprint()[:28] + "…")
