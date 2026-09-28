"""News Alpha Phase B -- renders one event's `EconomicMechanismGraph` inside
a Research Thread's Reasoning step.

A read surface only: it receives an already-built graph from
`news_alpha_context.mechanism_graph` and never builds claims, calls an LLM,
opens the registry, or fetches anything. The map is a left-to-right DOT
layout (`st.graphviz_chart`, rendered client-side -- no Python graphviz
dependency). Line style carries SUPPORT (solid = backed by a verified
source, dashed = reasoned only, amber = unresolved); arrows in node labels
carry the IMPLIED direction given the event. Neither is ever colored as a
verdict (no green/red), because neither is one.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence

import streamlit as st

from alpha_agent.news_alpha import (
    EconomicMechanismGraph,
    EdgeOrigin,
    ImpliedMovement,
    Polarity,
    StateImplication,
    SupportStatus,
    TransmissionLag,
)
from alpha_agent.news_alpha.transmission import GraphIssueKind, IssueSeverity, LoopType, StateKind
from alpha_agent.ui import palette

__all__ = ["consequence_rows", "expander_label", "graph_dot", "link_rows", "render_mechanism_graph",
           "render_open_questions", "render_sources", "summary_line"]


def consequence_rows(mg: EconomicMechanismGraph) -> list[dict[str, object]]:
    return _consequence_rows(mg)


def link_rows(mg: EconomicMechanismGraph) -> list[dict[str, object]]:
    return _link_rows(mg)


def render_sources(mg: EconomicMechanismGraph) -> None:
    _render_sources(mg)


def render_open_questions(mg: EconomicMechanismGraph) -> None:
    _render_open_questions(mg)

_GLYPH = {
    ImpliedMovement.UP: "↑", ImpliedMovement.DOWN: "↓", ImpliedMovement.MIXED: "⇅",
    ImpliedMovement.INDETERMINATE: "?",
}
_RELATIVE_GLYPH = {Polarity.POSITIVE: "+", Polarity.NEGATIVE: "−", Polarity.AMBIGUOUS: "±", Polarity.UNKNOWN: "?"}
_MOVEMENT_WORD = {
    ImpliedMovement.UP: "up", ImpliedMovement.DOWN: "down", ImpliedMovement.MIXED: "mixed",
    ImpliedMovement.INDETERMINATE: "indeterminate",
}
_LAG_TEXT = {
    TransmissionLag.IMMEDIATE: "immediate", TransmissionLag.WEEKS: "weeks", TransmissionLag.MONTHS: "months",
    TransmissionLag.QUARTERS: "quarters", TransmissionLag.YEARS: "years", TransmissionLag.UNKNOWN: "unknown",
}
_ORIGIN_TEXT = {
    EdgeOrigin.MANUALLY_SEEDED: "Reviewed seed", EdgeOrigin.MODEL_PROPOSED: "Model-proposed",
    EdgeOrigin.EXTERNALLY_SOURCED: "External source", EdgeOrigin.OBSERVED: "Observed in data",
}
_SUPPORT_TEXT = {
    SupportStatus.SUPPORTED: "Supported -- verified source",
    SupportStatus.PROPOSED: "Proposed -- reasoned, not yet sourced",
    SupportStatus.UNRESOLVED: "Unresolved -- conflict or missing provenance",
}
_SUPPORTED_EDGE = palette.SEQUENTIAL_BLUE[4]


def _glyph(i: StateImplication) -> str:
    if i.movement is ImpliedMovement.UNANCHORED:
        # the starting state itself has no direction; everything else is
        # shown relative to it (+ with, − against)
        return "•" if i.is_anchor else _RELATIVE_GLYPH[i.relative_sign]
    return _GLYPH[i.movement]


def _movement_text(i: StateImplication) -> str:
    if i.movement is ImpliedMovement.UNANCHORED:
        return {Polarity.POSITIVE: "+ moves with the anchor", Polarity.NEGATIVE: "− moves against the anchor"}.get(
            i.relative_sign, "± relation to the anchor is ambiguous",
        )
    return f"{_GLYPH[i.movement]} {_MOVEMENT_WORD[i.movement]}"


def summary_line(mg: EconomicMechanismGraph) -> str | None:
    """One caption for the collapsed card: where the event starts and what
    it reaches first. ``None`` when no channel was detected at all."""
    if mg.is_empty:
        if not mg.unseeded_channels:
            return None
        names = ", ".join(u.channel_label for u in mg.unseeded_channels)
        return f"Mechanism graph: no economic transmission is seeded yet for {names}."
    anchors = " · ".join(f"{i.label} {_glyph(i)}" for i in mg.implications if i.is_anchor)
    downstream = [i for i in mg.implications if not i.is_anchor]
    if not downstream:
        return f"Mechanism graph: {anchors} -- no downstream links are seeded yet."
    first = " · ".join(f"{i.label} {_glyph(i)}" for i in downstream if i.hops == 1)
    counts = mg.graph.support_counts()
    mixed = sum(i.movement is ImpliedMovement.MIXED for i in downstream)
    tail = f"{len(downstream)} downstream states"
    if mixed:
        tail += f", {mixed} with paths that disagree"
    unanchored = any(i.is_anchor and i.movement is ImpliedMovement.UNANCHORED for i in mg.implications)
    legend = " (no direction in the text: + moves with, − against the starting state)" if unanchored else ""
    return (
        f"Mechanism graph{legend}: {anchors} → {first} … {tail} · {counts[SupportStatus.SUPPORTED]} of "
        f"{len(mg.graph.edges)} links backed by a verified source."
    )


def expander_label(mg: EconomicMechanismGraph) -> str:
    if mg.is_empty:
        return "Economic mechanism graph · not seeded for this event"
    return f"Economic mechanism graph · {len(mg.graph.states)} states · {len(mg.graph.edges)} links"


def _dot_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _wrapped_label(text: str, glyph: str = "", width: int = 22) -> str:
    """A quoted DOT label broken onto lines of about ``width`` characters, so
    the top-to-bottom layout stays narrow enough to read inside a card. The
    direction ``glyph`` stays on the last line; each line is escaped first and
    the DOT line break inserted after."""
    lines: list[str] = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += f" {word}"
        else:
            lines.append(word)
    if glyph:
        lines[-1] += f" {glyph}"
    return "\\n".join(_dot_quote(line)[1:-1] for line in lines).join('""')


def graph_dot(mg: EconomicMechanismGraph, *, highlight: Sequence[Sequence[str]] = (), wrap: int = 22) -> str:
    """The event graph as DOT. Pure (testable without Streamlit).
    ``highlight``: state-id sequences (the signal paths a Research Thread
    follows) drawn in the accent color over the rest of the graph -- a
    display emphasis only; support and direction encodings are unchanged."""
    lit_states = {s for chain in highlight for s in chain}
    lit_edges = {(a, b) for chain in highlight for a, b in itertools.pairwise(chain)}
    by_state = {i.state_id: i for i in mg.implications}
    lines = [
        "digraph mechanism {",
        '  graph [rankdir=TB, bgcolor="transparent", pad=0.15, nodesep=0.25, ranksep=0.38];',
        (
            f'  node [shape=box, style="rounded,filled", fillcolor="{palette.CARD_BG_RAISED}", '
            f'color="{palette.BORDER}", fontcolor="{palette.TEXT_PRIMARY}", fontname="Helvetica", fontsize=13, '
            'margin="0.14,0.07"];'
        ),
        (
            f'  edge [color="{palette.TEXT_MUTED}", fontcolor="{palette.TEXT_SECONDARY}", fontname="Helvetica", '
            "fontsize=10, arrowsize=0.7];"
        ),
    ]
    for state in mg.graph.states:
        imp = by_state.get(state.state_id)
        attrs = [f"label={_wrapped_label(state.label, _glyph(imp) if imp else '', width=wrap)}"]
        if imp is not None and imp.is_anchor:
            attrs += [f'color="{palette.BLUE}"', "penwidth=2", f'fillcolor="{palette.SEQUENTIAL_BLUE[0]}"']
        elif imp is not None and imp.movement is ImpliedMovement.MIXED:
            attrs += [f'color="{palette.AMBER}"', "penwidth=1.6"]
        if state.kind is StateKind.UNSPECIFIED:
            attrs.append('style="rounded,filled,dashed"')
        if state.state_id in lit_states and not (imp is not None and imp.is_anchor):
            attrs += [f'color="{palette.BLUE}"', "penwidth=2.2"]
        lines.append(f"  {_dot_quote(state.state_id)} [{', '.join(attrs)}];")
    for e in mg.graph.edges:
        label = f"{_RELATIVE_GLYPH[e.polarity]} {_LAG_TEXT[e.lag]}"
        attrs = [f"label={_dot_quote(label)}"]
        if e.support is SupportStatus.SUPPORTED:
            attrs += [f'color="{_SUPPORTED_EDGE}"', "penwidth=1.5"]
        elif e.support is SupportStatus.UNRESOLVED:
            attrs += [f'color="{palette.AMBER}"', 'style="dashed"', "penwidth=1.5"]
        else:
            attrs.append('style="dashed"')
        if (e.source, e.target) in lit_edges:
            attrs += [f'color="{palette.BLUE}"', "penwidth=2.6", f'fontcolor="{palette.TEXT_PRIMARY}"']
        lines.append(f"  {_dot_quote(e.source)} -> {_dot_quote(e.target)} [{', '.join(attrs)}];")
    lines.append("}")
    return "\n".join(lines)


def _labels(mg: EconomicMechanismGraph) -> dict[str, str]:
    return {s.state_id: s.label for s in mg.graph.states}


def _render_anchors(mg: EconomicMechanismGraph) -> None:
    for a in mg.anchors:
        imp = mg.implication(a.state_id)
        if imp.movement is ImpliedMovement.UNANCHORED:
            how = (
                "the event text gives no direction, so each state shows whether it moves with (+) or against (−) "
                "this starting point"
            )
        else:
            how = f"direction cue: {(a.shift or '').replace('_', ' ').lower()}"
        st.markdown(f"Starts from **{a.label} {_glyph(imp)}** -- via {a.channel_label} ({how}).")


def _consequence_rows(mg: EconomicMechanismGraph) -> list[dict[str, object]]:
    labels = _labels(mg)
    rows = []
    for i in mg.implications:
        if i.is_anchor:
            continue
        best = i.paths[0]
        rows.append({
            "Economic state": i.label,
            "Implied": _movement_text(i),
            "Slowest link": _LAG_TEXT[best.slowest_lag],
            "Weakest link": best.weakest_support.value.title(),
            "Paths": len(i.paths),
            "Via (shortest chain)": " → ".join(labels[s] for s in best.state_ids[1:-1]) or "direct",
        })
    return rows


def _link_rows(mg: EconomicMechanismGraph) -> list[dict[str, object]]:
    labels = _labels(mg)
    rows = []
    for e in mg.graph.edges:
        backing = [p.title for p in e.sources if p.counts_as_support]
        rows.append({
            "Link": f"{labels[e.source]} → {labels[e.target]}",
            "Sign": _RELATIVE_GLYPH[e.polarity],
            "Status": e.support.value.title(),
            "Lag": _LAG_TEXT[e.lag],
            "Confidence": e.confidence.value.title(),
            "Backed by": "; ".join(backing) or "--",
            "Channel": ", ".join(c.value.replace("_", " ").lower() for c in e.channels),
            "Origin": ", ".join(_ORIGIN_TEXT[o] for o in e.origins),
        })
    return rows


def _render_sources(mg: EconomicMechanismGraph) -> None:
    labels = _labels(mg)
    cited: dict[tuple[str, str], tuple] = {}
    for e in mg.graph.edges:
        for p in e.sources:
            if p.counts_as_support:
                entry = cited.setdefault((p.kind.value, p.reference), (p, []))
                entry[1].append(f"{labels[e.source]} → {labels[e.target]}")
    if not cited:
        st.caption("No link in this graph is backed by a verified source yet.")
        return
    st.markdown("**Verified sources**")
    for p, links in cited.values():
        title = f"[{p.title}]({p.reference})" if p.reference.startswith("http") else p.title
        meta = ", ".join(x for x in (p.publisher, p.published.isoformat() if p.published else None) if x)
        st.markdown(f"- {title}{f' -- {meta}' if meta else ''}")
        st.caption(
            f"States: “{p.claim}” · checked {p.verified_on.isoformat()} · supports {'; '.join(links)}"
            + (f" · {p.note}" if p.note else "")
        )


def _render_open_questions(mg: EconomicMechanismGraph) -> None:
    labels = _labels(mg)
    shown = False
    for i in mg.implications:
        if i.movement is ImpliedMovement.MIXED:
            st.markdown(f"**{i.label} ⇅** -- {i.note}")
            shown = True
    for loop in mg.graph.loops:
        chain = " → ".join(labels[s] for s in (*loop.state_ids, loop.state_ids[0]))
        effect = {
            LoopType.BALANCING: "the loop eventually counteracts the initial move",
            LoopType.REINFORCING: "the loop amplifies the initial move",
            LoopType.INDETERMINATE: "the loop's net sign is unknown",
        }[loop.loop_type]
        st.markdown(f"**{loop.loop_type.value.title()} loop** -- {chain}: {effect}.")
        shown = True
    for issue in mg.issues:
        if issue.kind is GraphIssueKind.FEEDBACK_LOOP:
            continue
        if issue.severity is not IssueSeverity.INFO or issue.kind in (
            GraphIssueKind.HOP_LIMIT_REACHED, GraphIssueKind.DISCONNECTED_CLAIMS, GraphIssueKind.UNVERIFIED_CITATION,
        ):
            st.caption(f"{issue.kind.value.replace('_', ' ').title()}: {issue.message}")
            shown = True
    for u in mg.unseeded_channels:
        st.caption(u.note)
        shown = True
    proposed = mg.graph.support_counts()[SupportStatus.PROPOSED]
    if proposed:
        st.caption(
            f"{proposed} of {len(mg.graph.edges)} links are reasoned but not yet backed by a verified source -- "
            "treat them as hypotheses to check."
        )
        shown = True
    if not shown:
        st.caption("No conflicts, loops or unknowns in this graph.")


def render_mechanism_graph(mg: EconomicMechanismGraph, *, key: str) -> None:
    if mg.is_empty:
        for u in mg.unseeded_channels:
            st.caption(u.note)
        st.caption(mg.not_evidence_note)
        return
    _render_anchors(mg)
    tab_map, tab_states, tab_links, tab_open = st.tabs(
        ["Map", "Consequences", "Links & sources", "Open questions"], key=f"{key}-mechanism-tabs",
    )
    with tab_map:
        st.graphviz_chart(graph_dot(mg), width="stretch")
        st.caption(
            "↑/↓ implied direction given the event · ⇅ paths disagree · solid line = backed by a verified source · "
            "dashed = reasoned, not yet sourced · amber = unresolved · link labels: sign and typical lag."
        )
    with tab_states:
        st.dataframe(_consequence_rows(mg), width="stretch", hide_index=True)
        for i in mg.implications:
            if i.note and not i.is_anchor:
                st.caption(f"{i.label}: {i.note}")
    with tab_links:
        st.dataframe(_link_rows(mg), width="stretch", hide_index=True)
        st.caption(" · ".join(f"{s.value.title()}: {_SUPPORT_TEXT[s].split(' -- ')[1]}" for s in SupportStatus))
        _render_sources(mg)
    with tab_open:
        _render_open_questions(mg)
    st.caption(mg.not_evidence_note)
