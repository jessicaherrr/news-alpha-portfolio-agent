"""Step 7 -- Learn: "What did we learn?"

The thread's lineage end to end, and its transmission routes' memory: what
this run shows (computed now from the pipeline, labelled as not recorded) and
what the registry has recorded for the same routes from any event (schema v7
signal-path evidence, read-only). Links onward to My Alpha, the Research Map
and Community. Memory never blacklists a route and never becomes a verdict.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.alpha_memory.signal_path_evidence import EventResearch, build_signal_path_evidence
from alpha_agent.alpha_memory.signal_path_memory import (
    GUIDANCE_TEXT,
    SignalPathMemory,
    summarize_path_memory,
)
from alpha_agent.portfolio import EligibilityMode
from alpha_agent.ui import components, services
from alpha_agent.ui.research_thread import ThreadPipeline, ThreadStep
from alpha_agent.ui.workspace import step_backtest
from alpha_agent.ui.workspace.common import PATH_TYPE_TEXT, esc, facts_html, subhead
from alpha_agent.ui.workspace.shell import render_step_heading

__all__ = ["render"]

_LEARN_TABS = (("alpha_library", "My Alpha", ":material/psychology:",
                "Your research memory: threads, recorded routes, factors and strategies."),
               ("alpha_graph", "Research Map", ":material/hub:",
                "What has actually been researched before, on which assets, with what evidence."),
               ("community", "Community", ":material/groups:",
                "Independent replication and reproducible research -- not a leaderboard."))


def _go_learn(focus: str) -> None:
    from alpha_agent.ui.views import learn

    st.session_state["learn_landing_focus"] = focus
    st.switch_page(st.Page(learn.render, url_path="learn"))


def render(pipe: ThreadPipeline) -> None:
    render_step_heading(ThreadStep.LEARN)
    _render_lineage(pipe)
    _render_memory(pipe)
    subhead("Keep going")
    cols = st.columns(len(_LEARN_TABS))
    for col, (focus, label, icon, tip) in zip(cols, _LEARN_TABS, strict=True):
        with col:
            if st.button(f"Open {label}", key=f"thread-learn-{focus}", icon=icon, width="stretch", help=tip):
                _go_learn(focus)


def _render_lineage(pipe: ThreadPipeline) -> None:
    """Event -> Mechanism -> Signal path -> Market -> Measurement -> Signal ->
    Portfolio -> Experiment -> Evidence, each read from its canonical object."""
    thread = pipe.thread
    graph, paths, plan = pipe.graph, pipe.paths, pipe.expressions
    followed = pipe.followed_paths()
    focus = pipe.focus_group()
    counts = paths.count_by_type()
    if followed:
        path_text = "; ".join(f"{PATH_TYPE_TEXT[p.path_type]}: {' → '.join(p.state_labels)}" for p in followed[:2])
    elif paths.paths:
        path_text = f"{len(paths.paths)} paths ({', '.join(f'{n} {PATH_TYPE_TEXT[t].lower()}' for t, n in counts.items() if n)})"
    else:
        path_text = "--"
    usable = plan.usable_measurements()
    screened = len(pipe.screens)
    qplan, _missing = pipe.plan(EligibilityMode.QUALIFIED) if not pipe.candidates.is_empty else (None, [])
    runs = step_backtest.recorded_runs(pipe) if qplan is not None else []
    _tone, status, _detail = step_backtest.status_label(pipe)
    rows = [
        ("Event", thread.event.headline),
        ("Mechanism", f"{graph.anchors[0].label} · {len(graph.graph.states)} states, {len(graph.graph.edges)} links"
         if graph.anchors else "Not seeded for this event"),
        ("Signal path", path_text),
        ("Market", f"{focus.concept} ({', '.join(focus.symbols)})" if focus is not None and focus.symbols
         else focus.concept if focus is not None else f"{len(pipe.market_groups())} markets reached"),
        ("Measurement", (f"{len(usable)} of {len(plan.measurements)} measurements real, point-in-time safe and "
                        "computable")),
        ("Signal", f"{len(pipe.candidates.candidates)} candidates · {screened} screened"),
        ("Portfolio", qplan.headline() if qplan is not None else "Not built"),
        ("Experiment", f"{len(runs)} recorded validation run(s)" if runs else "None recorded yet"),
        ("Evidence", status),
    ]
    subhead("This thread, end to end", "Every stage of the research object, as it stands now.")
    with components.card("thread-lineage"):
        st.markdown('<div class="aa-ctx">' + "".join(
            f'<div class="aa-ctx-row"><div class="aa-ctx-label">{esc(label)}</div>'
            f'<div class="aa-ctx-value">{esc(value)}</div></div>' for label, value in rows
        ) + "</div>", unsafe_allow_html=True)


def _memory_card(m: SignalPathMemory, *, key: str) -> None:
    to = " / ".join(c.replace("_", " ").lower() for c in m.consequence_states) or "--"
    with components.card(key):
        st.markdown(
            f'<div class="aa-eyebrow">{esc((m.path_type or "unclassified").replace("_", " "))} route · depth '
            f"{m.transmission_depth}</div>"
            f'<div class="aa-news-headline" style="font-size:0.95rem;margin-top:0.1rem">To {esc(to)}</div>',
            unsafe_allow_html=True,
        )
        outcomes = ", ".join(f"{k.lower()} {v}" for k, v in m.outcomes.items()) or "--"
        st.markdown(facts_html([
            ("Furthest stage", m.furthest_stage.value.replace("_", " ").capitalize()),
            ("Main reason", m.primary_reason.value.replace("_", " ").capitalize()),
            ("Outcomes", outcomes),
            ("Seen in", f"{len(m.event_ids)} event(s) · {m.research_runs} run(s)"),
        ]), unsafe_allow_html=True)
        guidance = " ".join(GUIDANCE_TEXT[g] for g in m.guidance[:2])
        if guidance:
            st.caption("What would change it: " + guidance)


def _render_memory(pipe: ThreadPipeline) -> None:
    subhead("What this thread taught us",
            "A failure belongs to the route, market, measurement and signal it happened at -- never to a whole "
            "mechanism, and nothing is blacklisted. A screen is in-sample and unconditional; a portfolio result is "
            "about that portfolio only.")
    if pipe.paths.is_empty:
        st.caption("No transmission route to remember: this event has no seeded mechanism yet.")
        return
    qplan = pipe.plan(EligibilityMode.QUALIFIED)[0] if not pipe.candidates.is_empty else None
    event = EventResearch(discovery=pipe.paths, expressions=pipe.expressions, candidates=pipe.candidates)
    current = build_signal_path_evidence([event], ranked=pipe.ranked, screens=pipe.screens, plan=qplan,
                                         validation=pipe.gate if qplan is not None else None, recorded_at="--")
    memories = summarize_path_memory(current)
    tabs = st.tabs([f"This run ({len(memories)})", "Recorded memory"], key=f"thread-memory-tabs-{pipe.thread.thread_id}")
    with tabs[0]:
        st.caption("Computed now from this thread -- not recorded. Recording happens when a validation run records "
                   "its evidence, whatever the outcome.")
        cols = st.columns(2)
        for i, m in enumerate(memories[:8]):
            with cols[i % 2]:
                _memory_card(m, key=f"thread-mem-now-{i}")
        if len(memories) > 8:
            st.caption(f"+{len(memories) - 8} more routes under Learn → My Alpha.")
    with tabs[1]:
        recorded = services.signal_path_memory_for_routes([r.path_signature for r in current if r.path_signature])
        if not recorded:
            st.caption("No run of these routes is recorded yet.")
        cols = st.columns(2)
        for i, m in enumerate(recorded[:8]):
            with cols[i % 2]:
                _memory_card(m, key=f"thread-mem-rec-{i}")
