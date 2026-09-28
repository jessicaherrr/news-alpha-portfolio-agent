"""News Alpha Phase D -- renders one event's asset expressions and their
point-in-time measurement resolution inside the Agent page's News Impact
Triage card.

A read surface only: it receives an already-built `AssetExpressionPlan`
from `news_alpha_context` and never resolves data, calls an LLM, opens the
registry, or fetches anything. Statuses are shown as text, never as verdict
colors -- an expression is where research COULD look, not a trade.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.news_alpha import (
    DOMAIN_LABELS,
    AssetExpression,
    AssetExpressionPlan,
    ExpressionFidelity,
    ExpressionForm,
    ExpressionStatus,
    ImpliedMovement,
    MandateDomain,
    Polarity,
    PublicationLag,
    ResolutionStatus,
    TransmissionLag,
)
from alpha_agent.news_alpha.channels import PressureSign

__all__ = [
    "chat_text",
    "domain_rows",
    "expander_label",
    "expression_rows",
    "field_rows",
    "gap_rows",
    "readiness_rows",
    "render_asset_expressions",
    "summary_line",
]

_STATUS_TEXT = {
    ExpressionStatus.CONTINUES: "Continues",
    ExpressionStatus.EXCLUDED_BY_MANDATE: "Excluded by your mandate",
    ExpressionStatus.DOMAIN_UNAVAILABLE: "Domain unavailable",
    ExpressionStatus.NO_INSTRUMENT: "Concept only -- no instrument here",
    ExpressionStatus.REMOVED_BY_CONSTRAINTS: "Removed by your instrument limits",
}
_FIDELITY_TEXT = {
    ExpressionFidelity.DIRECT_UNDERLYING: "Direct underlying",
    ExpressionFidelity.DIRECT_COMPANY: "Direct company",
    ExpressionFidelity.SEGMENT_EXPOSURE: "One segment of the company",
    ExpressionFidelity.CONSTITUENT_EXPOSURE: "Holds exposed names",
    ExpressionFidelity.ECOSYSTEM_PROXY: "Ecosystem proxy",
    ExpressionFidelity.MACRO_PROXY: "Macro proxy",
}
_FORM_TEXT = {
    ExpressionForm.UNDERLYING: "Underlying",
    ExpressionForm.SINGLE_NAME: "Single name",
    ExpressionForm.SECTOR_BASKET: "Sector basket",
    ExpressionForm.BROAD_INDEX: "Broad index",
}
_RELATION_TEXT = {
    Polarity.POSITIVE: "+ moves with it", Polarity.NEGATIVE: "− moves against it", Polarity.AMBIGUOUS: "± either way",
}
_RESOLUTION_TEXT = {
    ResolutionStatus.AVAILABLE: "Available",
    ResolutionStatus.AVAILABLE_WITH_PROXY: "Proxy",
    ResolutionStatus.PARTIAL: "Partial",
    ResolutionStatus.MISSING: "Missing",
    ResolutionStatus.NOT_PIT_SAFE: "Not PIT-safe",
    ResolutionStatus.DOMAIN_UNAVAILABLE: "Domain unavailable",
    ResolutionStatus.NOT_EXECUTABLE: "Not executable",
}
_ARROW = {ImpliedMovement.UP: "↑", ImpliedMovement.DOWN: "↓"}
_PRESSURE = {PressureSign.UP: "↑", PressureSign.DOWN: "↓"}
_LAG_TEXT = {
    TransmissionLag.IMMEDIATE: "immediate", TransmissionLag.WEEKS: "weeks", TransmissionLag.MONTHS: "months",
    TransmissionLag.QUARTERS: "quarters", TransmissionLag.YEARS: "years", TransmissionLag.UNKNOWN: "unknown",
}
_PUBLICATION_TEXT = {
    PublicationLag.AT_BAR_CLOSE: "at bar close",
    PublicationLag.NEXT_SESSION: "next session",
    PublicationLag.WEEKLY_RELEASE: "weekly release (days later)",
    PublicationLag.FILING_ACCEPTANCE: "SEC filing acceptance (weeks)",
    PublicationLag.AT_PUBLICATION: "at publication",
    PublicationLag.NOT_APPLICABLE: "--",
}
_PIT_TEXT = {True: "Yes", False: "No", None: "Unknown"}
_DOMAIN_COLUMNS = {
    "Domain": st.column_config.TextColumn(width="small"),
    "Expressions": st.column_config.TextColumn(width="small"),
    "Continue": st.column_config.TextColumn(width="small"),
    "Runnable today": st.column_config.TextColumn(
        width="small", help="Continuing expressions on an instrument the research stack can simulate today",
    ),
    "Platform": st.column_config.TextColumn(width="large"),
}
_FIELD_COLUMNS = {
    "Instrument": st.column_config.TextColumn(width="small"),
    "Status": st.column_config.TextColumn(width="small"),
    "Source": st.column_config.TextColumn(width="medium"),
    "History": st.column_config.TextColumn(width="medium"),
    "Point-in-time": st.column_config.TextColumn(
        width="small", help="Was each value knowable at its own timestamp? Unknown = no data to judge.",
    ),
    "Why / proxy": st.column_config.TextColumn(width="large"),
}
_EXPRESSION_COLUMNS = {
    "Consequence": st.column_config.TextColumn(width="medium"),
    "Domain": st.column_config.TextColumn(width="small"),
    "Expression": st.column_config.TextColumn(width="medium"),
    "Instruments": st.column_config.TextColumn(width="medium"),
    "Status": st.column_config.TextColumn(width="medium"),
    "Fidelity": st.column_config.TextColumn(width="medium"),
    "Measurable": st.column_config.TextColumn(
        width="small", help="Measurements usable on 2018-2024 history, of those resolved",
    ),
}


def _domain_counts(plan: AssetExpressionPlan) -> str:
    return " · ".join(
        f"{DOMAIN_LABELS[s.domain]} {s.expressions}" for s in plan.domains if s.in_mandate and s.expressions
    )


def _measurable_targets(plan: AssetExpressionPlan) -> list[str]:
    """Instruments with a usable measurement, in domain order."""
    order = {d: i for i, d in enumerate(MandateDomain)}
    usable = sorted(plan.usable_measurements(), key=lambda m: order[m.domain])
    return list(dict.fromkeys(m.target for m in usable))


def summary_line(plan: AssetExpressionPlan) -> str | None:
    """One caption for the collapsed card; ``None`` when no path reaches a
    consequence."""
    if plan.is_empty:
        return None
    continuing = len(plan.with_status(ExpressionStatus.CONTINUES))
    measurable = plan.measurable_expressions()
    targets = _measurable_targets(plan)
    head = f"Asset expression: {len(plan.expressions)} ways to express these consequences ({_domain_counts(plan)})"
    body = f"{continuing} continue under your mandate, {len(measurable)} measurable on 2018-2024 history"
    body += f" via {', '.join(targets)}" if targets else " (none yet)"
    blockers = plan.blockers()
    tail = f" Largest gap: {_gap_phrase(blockers[0])}." if blockers else ""
    return f"{head} -- {body}.{tail}"


def _gap_phrase(b) -> str:
    """A gap with its honest reach: closing it removes one blocker, and only
    unlocks the measurements it was the last blocker for."""
    return (
        f"{b.gap} (primary blocker for {b.primary_for} measurements; the only blocker for {b.only_blocker_for})"
    )


def expander_label(plan: AssetExpressionPlan) -> str:
    return (
        f"Asset expression & measurement · {len(plan.expressions)} expressions · "
        f"{len(plan.measurable_expressions())} measurable today"
    )


def _direction(e: AssetExpression) -> str:
    parts = [
        f"{_ARROW.get(p.consequence_direction, '?')} {_LAG_TEXT[p.horizon]}" for p in e.pressures
    ]
    return " · ".join(dict.fromkeys(parts)) or "--"


def _pressure(e: AssetExpression) -> str:
    parts = [
        f"{_PRESSURE[p.pressure]} {_LAG_TEXT[p.horizon]}" if p.pressure else f"? {_LAG_TEXT[p.horizon]}"
        for p in e.pressures
    ]
    return " · ".join(dict.fromkeys(parts)) or "--"


def expression_rows(plan: AssetExpressionPlan) -> list[dict[str, object]]:
    """One row per expression, consequence and status first so they survive
    a narrow card. A non-continuing status carries its reason in the cell."""
    usable = {m.spec_id for m in plan.usable_measurements()}
    rows = []
    for e in plan.expressions:
        measured = set(e.measurement_ids)
        rows.append({
            "Consequence": e.consequence_label,
            "Domain": DOMAIN_LABELS[e.domain],
            "Expression": e.concept,
            "Fidelity": _FIDELITY_TEXT[e.fidelity],
            "Instruments": ", ".join(e.symbols) or "--",
            "Status": _STATUS_TEXT[e.status],
            "Measurable": f"{len(usable & measured)} of {len(measured)}" if measured else "--",
            "Consequence direction": _direction(e),
            "Relation": _RELATION_TEXT[e.relation],
            "Pressure (hypothesis)": _pressure(e),
            "Form": _FORM_TEXT[e.form],
        })
    return rows


def readiness_rows(plan: AssetExpressionPlan, domain: MandateDomain) -> list[dict[str, str]]:
    """Instruments x measurements for one domain: each cell is the
    resolution status, so a row reads as "what we can measure here"."""
    specs = [m for m in plan.measurements if m.domain is domain]
    labels = list(dict.fromkeys(m.label for m in specs))
    by_target: dict[str, dict[str, str]] = {}
    for m in specs:
        row = by_target.setdefault(m.target, {"Instrument": m.target})
        row[m.label] = _RESOLUTION_TEXT[m.resolution.status]
    return [{"Instrument": r["Instrument"], **{lab: r.get(lab, "--") for lab in labels}} for r in by_target.values()]


def field_rows(plan: AssetExpressionPlan) -> list[dict[str, object]]:
    rows = []
    for m in plan.measurements:
        r = m.resolution
        source = " · ".join(x for x in (r.dataset, r.data_schema, r.field) if x) or "--"
        rows.append({
            "Instrument": m.target,
            "Measurement": m.label,
            "Status": _RESOLUTION_TEXT[r.status],
            "Blocked by": " + ".join(g.label for g in r.gaps) or "--",
            "Source": source,
            "History": r.coverage_note or "--",
            "Point-in-time": _PIT_TEXT[r.pit_safe],
            "Knowable": _PUBLICATION_TEXT[r.publication_lag],
            "Why / proxy": r.missing_reason or r.proxy_note or "--",
            "Frequency": r.frequency.value.replace("_", " ").lower(),
            "Transform": r.transform,
            "Ideal measurement": m.variable.economic_meaning,
        })
    return rows


def gap_rows(plan: AssetExpressionPlan) -> list[dict[str, object]]:
    return [
        {
            "Gap": b.gap,
            "Primary blocker for": b.primary_for,
            "Only blocker for": b.only_blocker_for,
            "Also blocks": b.also_blocks,
            "Resolves as": _RESOLUTION_TEXT[b.status],
            "Instruments": ", ".join(b.targets),
        }
        for b in plan.blockers()
    ]


def domain_rows(plan: AssetExpressionPlan) -> list[dict[str, object]]:
    """One compact row per domain: what is economically relevant there, what
    the mandate lets continue, and what the platform can actually run."""
    return [
        {
            "Domain": DOMAIN_LABELS[s.domain],
            "Expressions": str(s.expressions),
            "Continue": str(s.continuing) if s.in_mandate else "--",
            "Runnable today": str(s.execution_capable) if s.in_mandate else "--",
            "Platform": s.note,
        }
        for s in plan.domains
    ]


def render_asset_expressions(plan: AssetExpressionPlan, *, key: str) -> None:
    if plan.is_empty:
        st.caption("No asset expression: no signal path reaches an economic consequence for this event yet.")
        st.caption(plan.not_trade_note)
        return
    st.caption(
        "Where each consequence could be expressed in a market, whether your mandate lets it continue, and which "
        "measurements actually existed at the time. Economic relevance is listed for every domain; your mandate "
        "only decides what continues."
    )
    st.dataframe(domain_rows(plan), width="stretch", hide_index=True, column_config=_DOMAIN_COLUMNS)
    gaps = gap_rows(plan)
    tabs = st.tabs(
        [f"Expressions ({len(plan.expressions)})", "Measurement readiness",
         f"Field resolution ({len(plan.measurements)})", f"Gaps ({len(gaps)})"],
        key=f"{key}-asset-expression-tabs",
    )
    with tabs[0]:
        st.dataframe(expression_rows(plan), width="stretch", hide_index=True, column_config=_EXPRESSION_COLUMNS)
        notes = sorted({n for e in plan.expressions for n in e.mandate_notes})
        for note in notes:
            st.caption(f"Mandate: {note}")
        st.caption(
            "Fidelity: how directly the instrument carries the consequence -- its own quantity, the company "
            "itself, one segment of a company, a fund holding the exposed names, or only an ecosystem/macro proxy. "
            "A description, never a score. Relation: whether the instrument's value moves with or against the "
            "consequence. Pressure: the first-order push that implies, per path horizon -- a hypothesis to test, "
            "never a forecast."
        )
    with tabs[1]:
        measured_domains = [d for d in MandateDomain if any(m.domain is d for m in plan.measurements)]
        if not measured_domains:
            st.caption("No continuing expression to measure under your mandate.")
        for d in measured_domains:
            st.markdown(f"**{DOMAIN_LABELS[d]}**")
            st.dataframe(readiness_rows(plan, d), width="stretch", hide_index=True)
        st.caption(
            "Available / Proxy / Partial: real, point-in-time safe and computable today. Missing: not acquired or "
            "no source. Not PIT-safe: would leak hindsight even if acquired. Not executable: the data exists but no "
            "registered feature computes it."
        )
    with tabs[2]:
        st.dataframe(field_rows(plan), width="stretch", hide_index=True, column_config=_FIELD_COLUMNS)
        st.caption(
            "A value queried today is not automatically usable historically: only a Point-in-time 'Yes' with a "
            "usable status may feed a historical test. The locked 2025 holdout is never covered."
        )
    with tabs[3]:
        if gaps:
            st.dataframe(gaps, width="stretch", hide_index=True)
            st.caption(
                "A measurement can wait on several gaps. Closing a gap removes that one blocker: it makes usable only "
                "the measurements it is the only blocker for; the rest still wait on another gap (primary = the gap "
                "that decides the status; also blocks = an additional blocker behind another gap). Any acquisition "
                "needs a cost estimate and your approval first -- nothing on this page fetches data."
            )
        else:
            st.caption("No gap: every measurement resolved.")
    st.caption(plan.not_trade_note)


def chat_text(plan: AssetExpressionPlan) -> str | None:
    """Plain-text form for the Agent chat."""
    if plan.is_empty:
        return None
    targets = _measurable_targets(plan)
    blockers = plan.blockers()
    text = (
        f"ASSET EXPRESSION: {len(plan.expressions)} expressions ({_domain_counts(plan)}); "
        f"{len(plan.with_status(ExpressionStatus.CONTINUES))} continue under your mandate, "
        f"{len(plan.measurable_expressions())} measurable on 2018-2024 history"
        + (f" via {', '.join(targets)}" if targets else "")
    )
    if blockers:
        text += f"; largest gap: {_gap_phrase(blockers[0])}"
    return text + "."
