"""News Alpha Phase C -- renders one event's signal paths and its
mechanism-adjusted impact in a Research Thread (and the Agent chat).

A read surface only: it receives an already-built `SignalPathDiscovery` /
`MechanismAdjustedImpact` from `news_alpha_context` and never builds paths,
calls an LLM, opens the registry, or fetches anything. Path type, status and
direction are shown as text, never as verdict colors -- none of them is a
verdict, and a signal path is never a trade.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.news_alpha import (
    DOMAIN_LABELS,
    ImpliedMovement,
    MechanismAdjustedImpact,
    PathStatus,
    PathType,
    Polarity,
    SignalPath,
    SignalPathDiscovery,
    StatusReason,
    SupportStatus,
    TransmissionLag,
)
from alpha_agent.news_alpha.channels import channel_definition
from alpha_agent.news_alpha.impact_adjustment import MechanismSupport

__all__ = [
    "adjusted_summary_text",
    "expander_label",
    "initial_badge_row",
    "mechanism_support_row",
    "path_rows",
    "render_adjusted_impact",
    "render_signal_paths",
    "summary_line",
]

_TYPE_TEXT = {
    PathType.DIRECT: "direct",
    PathType.SUPPLY_CHAIN: "supply-chain",
    PathType.CROSS_SECTOR: "cross-sector",
}
_TYPE_TAB = {PathType.DIRECT: "Direct", PathType.SUPPLY_CHAIN: "Supply chain", PathType.CROSS_SECTOR: "Cross-sector"}
_TYPE_MEANING = {
    PathType.DIRECT: "The event's own first-order consequences, and the first supplier or customer its demand or "
                     "cost lands on.",
    PathType.SUPPLY_CHAIN: "The effect travels past that first counterparty to its own suppliers or customers "
                           "(two or more value-chain steps), staying in the sector where it landed.",
    PathType.CROSS_SECTOR: "The effect lands in one sector, then crosses into a different one.",
}
_EMPTY_TYPE = {
    PathType.DIRECT: "No direct path in this graph.",
    PathType.SUPPLY_CHAIN: "No supply-chain path: the seeded transmission for this event never takes a second "
                           "supplier/customer step within one sector.",
    PathType.CROSS_SECTOR: "No cross-sector path: every seeded consequence stays in the sector where the event lands.",
}
_DIRECTION = {
    ImpliedMovement.UP: "↑ up", ImpliedMovement.DOWN: "↓ down", ImpliedMovement.INDETERMINATE: "? indeterminate",
}
_RELATIVE = {Polarity.POSITIVE: "+ with the start", Polarity.NEGATIVE: "− against the start"}
_LAG_TEXT = {
    TransmissionLag.IMMEDIATE: "immediate", TransmissionLag.WEEKS: "weeks", TransmissionLag.MONTHS: "months",
    TransmissionLag.QUARTERS: "quarters", TransmissionLag.YEARS: "years", TransmissionLag.UNKNOWN: "unknown",
}
_STATUS_TEXT = {
    PathStatus.RESEARCHABLE: "Researchable", PathStatus.PROPOSED: "Proposed", PathStatus.UNRESOLVED: "Unresolved",
    PathStatus.REJECTED: "Rejected",
}
_REASON_SHORT = {
    StatusReason.UNRESOLVED_LINK: "unresolved link",
    StatusReason.UNSIGNED_LINK: "unsigned link",
    StatusReason.ANCHOR_CONFLICT: "event direction conflicts",
    StatusReason.UNCLASSIFIED_STATE: "unclassified state",
    StatusReason.OPPOSED_AT_SAME_HORIZON: "opposed at the same horizon",
    StatusReason.NO_EVENT_DIRECTION: "no direction in the text",
    StatusReason.WEAK_LINK: "low-confidence link",
    StatusReason.UNKNOWN_LAG: "unknown lag",
    StatusReason.UNREVIEWED_MODEL_LINK: "unreviewed model link",
}
_SUPPORT_TEXT = {
    MechanismSupport.CORROBORATED: "Corroborated",
    MechanismSupport.PROPOSED_ONLY: "Proposed route only",
    MechanismSupport.NOT_REACHED: "Not reached",
    MechanismSupport.CHANNEL_NOT_SEEDED: "Channel not seeded",
    MechanismSupport.NO_ECONOMIC_BASIS: "No basis recorded",
}
_MOVEMENT_GLYPH = {
    ImpliedMovement.UP: "↑", ImpliedMovement.DOWN: "↓", ImpliedMovement.MIXED: "⇅",
    ImpliedMovement.INDETERMINATE: "?", ImpliedMovement.UNANCHORED: "±",
}
_ROW_LABEL_STYLE = "font-size:0.78rem;color:var(--aa-text-secondary);margin-right:0.35rem;"


def _level(level) -> str:
    return level.value.replace("_", " ")


def _direction(p: SignalPath) -> str:
    if p.expected_direction is ImpliedMovement.UNANCHORED:
        return _RELATIVE.get(p.relative_sign, "± ambiguous")
    return _DIRECTION.get(p.expected_direction, p.expected_direction.value.lower())


# ---------------------------------------------------------------------------
# signal paths
# ---------------------------------------------------------------------------


def summary_line(d: SignalPathDiscovery) -> str | None:
    """One caption for the collapsed card; ``None`` when there is no graph."""
    if d.is_empty:
        return None
    counts = d.count_by_type()
    status = d.count_by_status()
    parts = [f"{counts[t]} {_TYPE_TEXT[t]}" for t in PathType]
    unclassified = len(d.of_type(None))
    if unclassified:
        parts.append(f"{unclassified} unclassified")
    tail = f"{status[PathStatus.RESEARCHABLE]} researchable"
    if status[PathStatus.UNRESOLVED]:
        tail += f", {status[PathStatus.UNRESOLVED]} unresolved"
    opposed = [c for c in d.conflicts if not c.separable_by_horizon]
    if opposed:
        tail += f" · {', '.join(c.consequence_label for c in opposed)}: opposing paths at the same horizon"
    return f"Signal paths: {' · '.join(parts)} -- {tail}."


def expander_label(d: SignalPathDiscovery) -> str:
    return f"Signal paths · {len(d.paths)} hypotheses · {d.count_by_status()[PathStatus.RESEARCHABLE]} researchable"


def _route_hints(paths: tuple[SignalPath, ...]) -> dict[str, str]:
    """For a consequence reached by several routes in one table, the state
    where each route first departs from its siblings -- so two rows for the
    same consequence never look identical in a narrow card."""
    by_consequence: dict[str, list[SignalPath]] = {}
    for p in paths:
        by_consequence.setdefault(p.consequence_state, []).append(p)
    hints: dict[str, str] = {}
    for group in by_consequence.values():
        if len(group) < 2:
            continue
        for p in group:
            others = [q for q in group if q is not p]
            i = next(
                i for i in range(1, len(p.state_ids))
                if any(i >= len(q.state_ids) or q.state_ids[i] != p.state_ids[i] for q in others)
            )
            hints[p.path_id] = (
                f"via {p.state_labels[i]}" if i < len(p.state_ids) - 1 else f"straight from {p.state_labels[i - 1]}"
            )
    return hints


def path_rows(paths: tuple[SignalPath, ...]) -> list[dict[str, object]]:
    """Table rows, consequence and status first so they survive a narrow card.
    A non-researchable status carries its reasons in the same cell, so they
    are never scrolled out of view."""
    hints = _route_hints(paths)
    rows = []
    for p in paths:
        verified = sum(link.support is SupportStatus.SUPPORTED for link in p.links)
        status = " · ".join([_STATUS_TEXT[p.status], *(_REASON_SHORT[r] for r in p.status_reasons)])
        rows.append({
            "Economic consequence": p.consequence_label + (f" · {hints[p.path_id]}" if p.path_id in hints else ""),
            "Direction": _direction(p),
            "Status": status,
            "Depth": p.transmission_depth,
            "Value-chain steps": p.value_chain_steps,
            "Sector crossings": "--" if p.sector_transitions is None else p.sector_transitions,
            "Horizon": _LAG_TEXT[p.expected_horizon],
            "Weakest link": p.confidence.value.title(),
            "Verified links": f"{verified} of {len(p.links)}",
            "Via": " → ".join(p.via_labels) or "direct",
            "Target concept": p.candidate_target_concept or "--",
        })
    return rows


def _render_type(d: SignalPathDiscovery, path_type: PathType | None) -> None:
    paths = d.of_type(path_type)
    if path_type is not None:
        st.caption(_TYPE_MEANING[path_type])
    if not paths:
        st.caption(_EMPTY_TYPE[path_type] if path_type is not None else "")
        return
    st.dataframe(path_rows(paths), width="stretch", hide_index=True)
    ids = {p.path_id for p in paths}
    for conflict in d.conflicts:
        if ids & set(conflict.path_ids):
            st.caption(f"Opposing paths -- {conflict.note}")
    if path_type is None:
        for p in paths:
            st.caption(f"{p.consequence_label}: {p.classification_basis}")


def render_signal_paths(d: SignalPathDiscovery, *, key: str) -> None:
    if d.is_empty:
        st.caption("No signal paths: this event has no seeded mechanism graph yet.")
        st.caption(d.not_trade_note)
        return
    st.caption(
        "Each row is one hypothesis: how this event could reach an economic consequence, link by link. Depth is "
        "recorded, never ranked -- a farther consequence is not assumed to be better research."
    )
    types: list[PathType | None] = list(PathType)
    labels = [f"{_TYPE_TAB[t]} ({len(d.of_type(t))})" for t in PathType]
    if d.of_type(None):
        types.append(None)
        labels.append(f"Unclassified ({len(d.of_type(None))})")
    if d.rejected:
        labels.append(f"Rejected proposals ({len(d.rejected)})")
    tabs = st.tabs(labels, key=f"{key}-signal-path-tabs")
    for tab, path_type in zip(tabs, types, strict=False):
        with tab:
            _render_type(d, path_type)
    if d.rejected:
        with tabs[-1]:
            for r in d.rejected:
                st.markdown(f"**{' → '.join(r.states)}** -- {', '.join(x.value for x in r.reasons)}")
                st.caption(f"{r.proposer}: {r.detail}")
    st.caption(
        "Researchable: a direction, a horizon and reviewed links of at least medium confidence -- ready to take "
        "further, not shown to be true (see Verified links). Proposed: well-formed but missing one of those. "
        "Unresolved: an unresolved or unsigned link, or an opposing path at the same horizon."
    )
    for f in d.findings:
        st.caption(f"{f.kind.value.replace('_', ' ').title()}: {f.message}")
    st.caption(d.not_trade_note)


# ---------------------------------------------------------------------------
# initial vs mechanism-adjusted impact
# ---------------------------------------------------------------------------


def initial_badge_row(badges: list[str]) -> str:
    return f'<span style="{_ROW_LABEL_STYLE}">Initial scan</span>' + " ".join(badges)


def _support_counts(adj: MechanismAdjustedImpact) -> list[str]:
    return [
        f"{DOMAIN_LABELS[a.asset_domain]} {a.corroborated()} of {len(a.revisions)}"
        for a in adj.assessments if a.revisions
    ]


def mechanism_support_row(adj: MechanismAdjustedImpact) -> str:
    """The second pass beside -- never instead of -- the initial row. Shows
    mechanism SUPPORT in plain text (no impact badge), because the mechanism
    establishes a route, not an economic magnitude, and levels stay put."""
    counts = " · ".join(_support_counts(adj)) or "no assessed exposure"
    text = (
        f"levels kept (economic magnitude not established) · mechanism corroborates exposures: {counts}"
    )
    return (
        f'<span style="{_ROW_LABEL_STYLE}">After mechanism graph</span>'
        f'<span style="font-size:0.8rem;color:var(--aa-text-secondary);">{text}</span>'
    )


def adjusted_summary_text(adj: MechanismAdjustedImpact) -> str:
    """Plain-text form for the Agent chat."""
    levels = " · ".join(
        f"{DOMAIN_LABELS[a.asset_domain]} {_level(a.adjusted_level)}" for a in adj.assessments
        if a.adjusted_level is not None
    )
    counts = ", ".join(_support_counts(adj)) or "none"
    return (
        f"levels kept ({levels}) -- economic magnitude not established; mechanism corroborates exposures: {counts}"
    )


def render_adjusted_impact(adj: MechanismAdjustedImpact) -> None:
    """Exposure-by-exposure table, for the "Why these domains?" panel: the
    level and the mechanism evidence in separate columns."""
    rows = []
    for a in adj.assessments:
        for r in a.revisions:
            rows.append({
                "Domain": DOMAIN_LABELS[a.asset_domain],
                "Exposure": r.exposure_label,
                "Initial level": _level(r.initial_level),
                "After graph": _level(r.adjusted_level),
                "Mechanism": _SUPPORT_TEXT[r.mechanism_support],
                "Route confidence": r.mechanism_confidence.value.title() if r.mechanism_confidence else "--",
                "Economic basis": ", ".join(
                    f"{b.label} {_MOVEMENT_GLYPH[b.movement]}" if b.movement else f"{b.label} (not reached)"
                    for b in r.basis
                ) or "--",
                "Channel": channel_definition(r.channel).label,
            })
    st.markdown('<div class="aa-gate-title" style="margin-top:0.6rem;">MECHANISM CHECK (SECOND PASS)</div>',
                unsafe_allow_html=True)
    if not rows:
        st.caption("No assessed exposure to check.")
        return
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        "Mechanism shows whether a reviewed route from the event reaches the exposure's economic basis; route "
        "confidence is how sure we are the route holds -- neither is the size of the effect. Economic magnitude is "
        "not established, so every level stays at the initial scan above. Arrows show the direction of the economic "
        "consequence, not of any price."
    )
