"""Step 2 -- Reasoning: "Why could this matter economically?"

The event's economic mechanism graph (why an effect may propagate) and its
signal paths (Direct / Supply chain / Cross sector), each a hypothesis to
test. The user FOLLOWS paths to focus the next steps; following never changes
what is tested. No BUY/SELL anywhere: a path ends at an economic
consequence, not a trade.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.news_alpha import PathStatus, PathType, SignalPath
from alpha_agent.ui import components, mechanism_graph_view, research_thread, signal_path_view
from alpha_agent.ui.research_thread import ThreadPipeline, ThreadStep
from alpha_agent.ui.workspace.common import (
    PATH_TYPE_TEXT,
    chain_html,
    esc,
    info_html,
    kind_html,
    meta_html,
    subhead,
)
from alpha_agent.ui.workspace.shell import render_step_heading

__all__ = ["render"]

_FILTER_KEY = "thread-path-filter-{}"
_SHOW_ALL_KEY = "thread-paths-all-{}"
_PAGE = 8
_STATUS_TEXT = {
    PathStatus.RESEARCHABLE: "Researchable", PathStatus.PROPOSED: "Proposed", PathStatus.UNRESOLVED: "Unresolved",
    PathStatus.REJECTED: "Rejected",
}
_STATUS_HELP = (
    "Researchable: a direction, a horizon and reviewed links of at least medium confidence -- ready to take further, "
    "not shown to be true. Proposed: well-formed but missing one of those (often: not every link is backed by a "
    "verified source yet). Unresolved: a conflicting or unsigned link, or an opposing path at the same horizon."
)


_GRAPH_OPEN_KEY = "thread-graph-dialog-open"


def _open_graph() -> None:
    st.session_state[_GRAPH_OPEN_KEY] = True


def _close_graph() -> None:
    st.session_state[_GRAPH_OPEN_KEY] = False


@st.dialog("Economic transmission", width="large", on_dismiss=_close_graph)
def _graph_dialog(dot: str) -> None:
    st.graphviz_chart(dot, width="stretch")
    st.caption("↑/↓ implied direction · solid = backed by a verified source · dashed = reasoned, not yet sourced · "
               "amber = unresolved · blue = the paths you follow.")


def _toggle_follow(pipe: ThreadPipeline, path: SignalPath) -> None:
    thread = pipe.thread
    followed = list(thread.followed_path_ids)
    changes: dict = {}
    if path.path_id in followed:
        followed.remove(path.path_id)
        group = pipe.focus_group()
        if group is not None and followed and not set(group.path_ids) & set(followed):
            changes["focus_expression_id"] = None
            st.toast("Your market focus was on that path, so it was cleared.", icon=":material/info:")
    else:
        followed.append(path.path_id)
    research_thread.update_thread(thread, followed_path_ids=tuple(followed), **changes)


def render(pipe: ThreadPipeline) -> None:
    render_step_heading(ThreadStep.REASONING)
    graph, paths = pipe.graph, pipe.paths
    if graph.is_empty:
        components.empty_state(
            "No economic mechanism is seeded for this event yet",
            " ".join(x for x in (mechanism_graph_view.summary_line(graph),
                                 *(u.note for u in graph.unseeded_channels)) if x) or
            "The event's text did not match an economic channel the reviewed transmission library covers. You can "
            "still describe the event differently on News.",
            key="thread-reasoning-empty",
        )
        return

    anchors = " · ".join(f"{a.label} (via {a.channel_label})" for a in graph.anchors)
    st.markdown(f'<div class="aa-subnote">Starts from <b style="color:var(--aa-text)">{esc(anchors)}</b>. Every link '
                "below is a reviewed economic relationship; solid lines are backed by a verified source, dashed ones "
                "are reasoned but not yet sourced.</div>", unsafe_allow_html=True)

    followed = pipe.followed_paths()
    with components.card("thread-mechanism-graph"):
        st.markdown('<div class="aa-eyebrow">Economic transmission' + info_html(
            "Why an effect may propagate -- a hypothesis graph. What has actually been researched before is a "
            "different map: Learn → Research Map.") + "</div>", unsafe_allow_html=True)
        dot = mechanism_graph_view.graph_dot(graph, highlight=[p.state_ids for p in followed], wrap=16)
        st.graphviz_chart(dot, width="stretch")
        st.button("View larger", key="thread-graph-larger", icon=":material/open_in_full:", type="tertiary",
                  on_click=_open_graph, help="Open the transmission graph at full size.")
        if st.session_state.get(_GRAPH_OPEN_KEY):
            _graph_dialog(dot)
        st.caption(
            "↑/↓ implied direction · ⇅ paths disagree · solid = backed by a verified source · dashed = reasoned, "
            "not yet sourced · amber = unresolved · link labels show sign and typical lag"
            + (" · blue = the paths you follow" if followed else "") + "."
        )

    _render_paths(pipe)

    with st.expander("Advanced details", icon=":material/tune:"):
        for line in (mechanism_graph_view.summary_line(graph), signal_path_view.summary_line(paths)):
            if line:
                st.caption(line)
        tabs = st.tabs(["Consequences", "Links & sources", "Open questions", "All paths", "Mechanism check"])
        with tabs[0]:
            st.dataframe(mechanism_graph_view.consequence_rows(graph), hide_index=True, width="stretch")
            for i in graph.implications:
                if i.note and not i.is_anchor:
                    st.caption(f"{i.label}: {i.note}")
        with tabs[1]:
            st.dataframe(mechanism_graph_view.link_rows(graph), hide_index=True, width="stretch")
            mechanism_graph_view.render_sources(graph)
        with tabs[2]:
            mechanism_graph_view.render_open_questions(graph)
        with tabs[3]:
            st.dataframe(signal_path_view.path_rows(paths.paths), hide_index=True, width="stretch")
            for conflict in paths.conflicts:
                st.caption(f"Opposing paths -- {conflict.note}")
            for f in paths.findings:
                st.caption(f"{f.kind.value.replace('_', ' ').title()}: {f.message}")
        with tabs[4]:
            if pipe.adjusted is not None:
                signal_path_view.render_adjusted_impact(pipe.adjusted)
        st.caption(graph.not_evidence_note)
        st.caption(paths.not_trade_note)


def _render_paths(pipe: ThreadPipeline) -> None:
    paths = pipe.paths
    thread = pipe.thread
    subhead(
        "Signal paths",
        "Each path is one hypothesis: how the event could reach an economic consequence, link by link. Follow the "
        "paths you want to carry forward -- the Market step then starts from them. Depth is recorded, never ranked.",
        help=_STATUS_HELP,
    )
    counts = paths.count_by_type()
    options: list[PathType | str] = ["all", *[t for t in PathType if counts[t]]]
    if paths.of_type(None):
        options.append("unclassified")
    labels = {"all": f"All ({len(paths.paths)})", "unclassified": f"Unclassified ({len(paths.of_type(None))})",
              **{t: f"{PATH_TYPE_TEXT[t]} ({counts[t]})" for t in PathType}}
    choice = st.segmented_control(
        "Path type", options, default="all", format_func=labels.__getitem__,
        key=_FILTER_KEY.format(thread.thread_id), label_visibility="collapsed",
        help="Direct: the event's own first-order consequences. Supply chain: the effect travels to suppliers or "
             "customers two or more steps away. Cross sector: it lands in one sector, then crosses into another.",
    ) or "all"
    if choice == "all":
        shown = paths.paths
    elif choice == "unclassified":
        shown = paths.of_type(None)
    else:
        shown = paths.of_type(choice)
    followed_ids = set(thread.followed_path_ids)
    show_all = st.session_state.get(_SHOW_ALL_KEY.format(thread.thread_id), False)
    visible = shown if show_all else shown[:_PAGE]
    cols = st.columns(2)
    for i, p in enumerate(visible):
        with cols[i % 2]:
            _render_path_card(pipe, p, followed=p.path_id in followed_ids)
    if len(shown) > _PAGE:
        st.toggle(f"Show all {len(shown)} paths", key=_SHOW_ALL_KEY.format(thread.thread_id))
    if followed_ids:
        st.caption(f"Following {len(followed_ids)} path(s). The Market step starts from the markets they reach.")


def _render_path_card(pipe: ThreadPipeline, p: SignalPath, *, followed: bool) -> None:
    slug = (p.path_type.value.lower() if p.path_type else "none")
    state = "followed" if followed else "idle"
    row = signal_path_view.path_rows((p,))[0]
    with components.card(f"path-{slug}-{state}-{p.path_id}"):
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center">{kind_html(p.path_type)}'
            f'<span class="aa-news-meta">{esc(_STATUS_TEXT[p.status])}</span></div>'
            + chain_html(p.state_labels),
            unsafe_allow_html=True,
        )
        st.markdown(meta_html([
            ("Depth", str(p.transmission_depth)),
            ("Horizon", str(row["Horizon"]).title()),
            ("Confidence", str(row["Weakest link"])),
            ("Direction", str(row["Direction"])),
            ("Verified", str(row["Verified links"])),
        ]), unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.6], vertical_alignment="center")
        with c1:
            if st.button("Following" if followed else "Follow", key=f"follow-{p.path_id}",
                         icon=":material/check:" if followed else ":material/add:",
                         type="primary" if followed else "secondary",
                         help="Unfollow this path." if followed else
                              "Carry this path forward: the Market step starts from the markets it reaches. "
                              "Following never changes what is tested."):
                _toggle_follow(pipe, p)
                st.rerun()
        with c2, st.popover("Inspect links", icon=":material/account_tree:", width="stretch"):
            _render_links(pipe, p)


def _render_links(pipe: ThreadPipeline, p: SignalPath) -> None:
    """Each link on the path: meaning, sign, lag, confidence, status, and what
    backs it -- the graph's own link rows, filtered to this path."""
    ids = {link.edge_id for link in p.links}
    rows = [r for e, r in zip(pipe.graph.graph.edges, mechanism_graph_view.link_rows(pipe.graph), strict=True)
            if e.edge_id in ids]
    st.markdown(f"**{esc(' → '.join(p.state_labels))}**")
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_order=("Link", "Sign", "Status", "Lag", "Confidence", "Backed by"))
    if p.status_reasons:
        st.caption("Why not researchable yet: " + str(signal_path_view.path_rows((p,))[0]["Status"]))
    st.caption(p.classification_basis)
