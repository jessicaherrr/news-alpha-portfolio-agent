"""News Alpha Phase F -- renders a `RankedSignalSet`: which candidate signals
deserve portfolio consideration first, for one event card and across every
event on the triage board.

A pure renderer. Every order, grade, group, link and sentence shown here is
computed by the canonical backend (`alpha_agent.recommendation.signal_ranking`)
-- this module never sorts, scores or groups signals itself. Screens are
read from the Phase E operational cache (`candidate_signal_view.SCREEN_STORE`);
nothing here loads bars, so rendering is instant, and a signal without a
screen is shown as NOT RANKABLE with the step that would rank it.
"""
from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from alpha_agent.news_alpha import DOMAIN_LABELS, CandidateSignalSet, ResearchMandate
from alpha_agent.recommendation.signal_ranking import (
    NEXT_STEP_TEXT,
    NOT_RANKABLE_TEXT,
    CostHeadroom,
    EvidenceDirection,
    FitStatus,
    RankedSignal,
    RankedSignalSet,
    SignalRole,
    StabilityGrade,
    rank_candidate_signals,
)
from alpha_agent.screening.factor_diagnostics import ScreenStatus
from alpha_agent.ui import candidate_signal_view

__all__ = [
    "chat_text",
    "expander_label",
    "exposure_rows",
    "rank_for",
    "ranked_rows",
    "render_signal_ranking",
    "summary_line",
    "unranked_rows",
]

_SCREEN = {
    ScreenStatus.SCREEN_CONTINUE: "Continue to validation",
    ScreenStatus.NO_SCREEN_SUPPORT: "No screen support",
    ScreenStatus.CONTRADICTS_EXPECTED_SIGN: "Contradicts its sign",
}
_DIRECTION = {
    EvidenceDirection.SUPPORTS: "Supports", EvidenceDirection.LEANS_EXPECTED: "Leans declared way",
    EvidenceDirection.LEANS_OPPOSITE: "Leans opposite", EvidenceDirection.CONTRADICTS: "Contradicts",
}
_STABILITY = {
    StabilityGrade.CONSISTENT: "Consistent", StabilityGrade.MIXED: "Mixed", StabilityGrade.OPPOSED: "Opposed",
    StabilityGrade.NOT_EVALUABLE: "Not evaluable",
}
_HEADROOM = {
    CostHeadroom.TIMING_EDGE: "Timing edge", CostHeadroom.NO_TIMING_EDGE: "No timing edge",
    CostHeadroom.NOT_ASSESSED: "Not assessed",
}
_RANKED_COLUMNS = {
    "Rank": st.column_config.NumberColumn(width="small", help="Independent exposures first, then their alternates"),
    "Role": st.column_config.TextColumn(
        width="small", help="Lead = the best-merit signal of an exposure; Alt. = another expression of the same one"),
    "Signal": st.column_config.TextColumn(
        width="medium", help="Instrument, the exact factor formula that runs, and the prediction horizon"),
    "t (n/h)": st.column_config.NumberColumn(
        format="%+.2f", help="Effective-sample t of the TS Spearman IC, in the declared direction"),
    "Why this rank": st.column_config.TextColumn(width="large"),
}


def rank_for(candidate_sets: Sequence[CandidateSignalSet], mandate: ResearchMandate) -> RankedSignalSet:
    """Rank candidate sets under the mandate using the cached screens only."""
    screens: dict = {}
    for cset in candidate_sets:
        screens.update(candidate_signal_view.stored_screens(cset))
    return rank_candidate_signals(tuple(candidate_sets), screens, mandate)


def summary_line(ranked: RankedSignalSet) -> str | None:
    return f"Signal ranking: {ranked.headline()}" if ranked.signals else None


def expander_label(ranked: RankedSignalSet) -> str:
    parts = [f"{ranked.independent_exposures} independent exposure(s)", f"{len(ranked.ranked)} ranked"]
    if ranked.not_rankable:
        parts.append(f"{len(ranked.not_rankable)} not rankable")
    if ranked.excluded:
        parts.append(f"{len(ranked.excluded)} excluded")
    return "Signal ranking for portfolio consideration · " + " · ".join(parts)


def _role(s: RankedSignal, ranked: RankedSignalSet) -> str:
    if s.role is SignalRole.LEAD:
        return "Lead"
    return f"Alt. of #{ranked.signal(s.lead_id).rank}"


def _fit_cell(s: RankedSignal) -> str:
    flagged = [c for c in s.user_fit.checks if c.status in (FitStatus.MISMATCH, FitStatus.CONSTRAINED)]
    if not flagged:
        return "Fits" if s.user_fit.personalization_state.value != "NOT_PERSONALIZED" else "No conflict"
    return " · ".join(f"{c.dimension.value.replace('_', ' ').lower()} {c.status.value.lower()}" for c in flagged)


def ranked_rows(ranked: RankedSignalSet) -> list[dict[str, object]]:
    rows = []
    for s in ranked.ranked:
        m, e = s.merit, s.merit.evidence
        rows.append({
            "Rank": s.rank,
            "Role": _role(s, ranked),
            "Signal": f"{s.instrument} {s.expression} → {s.prediction_horizon_days}D",
            "Screen": _SCREEN[m.screen_status],
            "Evidence": _DIRECTION[e.direction],
            "t (n/h)": e.aligned_t,
            "Years": f"{e.metrics.years_with_expected_sign} of {e.metrics.evaluable_years}",
            "Stability": _STABILITY[e.stability],
            "Cost": _HEADROOM[e.cost_headroom],
            "Data": m.data_quality.value.title(),
            "Your mandate": _fit_cell(s),
            "Exposure": ranked.group(s.exposure_group_id).label,
            "Domain": DOMAIN_LABELS[s.domain],
            "Why this rank": s.summaries.reason_for_rank,
        })
    return rows


def unranked_rows(ranked: RankedSignalSet) -> list[dict[str, str]]:
    rows = []
    for s in ranked.not_rankable:
        rows.append({
            "Signal": s.name, "Factor expression": s.expression, "Domain": DOMAIN_LABELS[s.domain],
            "Why not ranked": NOT_RANKABLE_TEXT[s.not_rankable_reason],
            "What would rank it": NEXT_STEP_TEXT[s.not_rankable_reason], "Redundancy": s.summaries.redundancy,
        })
    return rows


def _excluded_rows(ranked: RankedSignalSet) -> list[dict[str, str]]:
    return [{"Signal": s.name, "Factor expression": s.expression, "Domain": DOMAIN_LABELS[s.domain],
             "Excluded because": s.summaries.user_fit, "Research merit (unchanged)": s.summaries.scientific_quality}
            for s in ranked.excluded]


def exposure_rows(ranked: RankedSignalSet) -> list[dict[str, object]]:
    rows = []
    for g in ranked.exposure_groups:
        lead = ranked.signal(g.lead_id)
        rows.append({
            "Exposure": g.label,
            "Lead": f"#{lead.rank} {lead.name}",
            "Signals": len(g.member_ids),
            "Domains": " + ".join(DOMAIN_LABELS[d] for d in g.domains),
            "Linked by": " · ".join(r.value.replace("_", " ").lower() for r in g.relations) or "--",
            "Max |rho|": g.max_abs_correlation,
            "Events": len(g.event_ids),
        })
    return rows


def _correlation_rows(ranked: RankedSignalSet) -> list[dict[str, object]]:
    label = {s.candidate_signal_id: f"#{s.rank} {s.instrument} {s.expression} → {s.prediction_horizon_days}D"
             for s in ranked.ranked}
    return [{"Signal A": label[p.a], "Signal B": label[p.b], "Aligned rank correlation": p.correlation,
             "Common days": p.n_common, "Same exposure": "Yes" if p.links else "No"} for p in ranked.correlations]


def _detail_rows(s: RankedSignal) -> list[dict[str, str]]:
    sm = s.summaries
    return [
        {"Aspect": "Why this rank", "Summary": sm.reason_for_rank},
        {"Aspect": "Scientific quality", "Summary": sm.scientific_quality},
        {"Aspect": "Fit to your mandate", "Summary": sm.user_fit},
        {"Aspect": "Data quality", "Summary": sm.data_quality},
        {"Aspect": "Liquidity & cost", "Summary": sm.liquidity_cost},
        {"Aspect": "Redundancy", "Summary": sm.redundancy},
    ]


def _render_detail(s: RankedSignal, *, key: str) -> None:
    head = f"#{s.rank} · " if s.rank is not None else ""
    # The exact formula inline (the candidate section above already shows it as a code block).
    st.markdown(f"**{head}{s.name}** · `{s.expression}` → {s.prediction_horizon_days}D")
    st.table(_detail_rows(s), hide_index=True, border="horizontal")
    st.markdown("**Uncertainty**\n\n" + "\n".join(f"- {u.text}" for u in s.uncertainty))
    if s.merit is not None:
        e, m = s.merit.evidence, s.merit.evidence.metrics
        st.caption(
            f"Raw metrics at {m.horizon_days}D, as measured (never rescaled): TS Spearman IC "
            f"{m.ts_spearman_ic:+.3f} · t {m.ts_spearman_t:+.2f} on {m.n_pairs} pairs (~{m.n_independent:.0f} "
            f"non-overlapping) · {m.years_with_expected_sign}/{m.evaluable_years} years · coverage {m.coverage:.1%}. "
            f"Normalized by {e.normalization_method} within peer group {e.peer_group.key}: {e.peer_group.label}. "
            f"{e.normalization_note}"
            if m.ts_spearman_ic is not None and m.ts_spearman_t is not None else
            f"Normalized by {e.normalization_method} within peer group {e.peer_group.key}."
        )
    st.dataframe(
        [{"Mandate check": c.dimension.value.replace("_", " ").title(), "Status": c.status.value.replace("_", " ")
          .title(), "Your mandate": c.preference, "This signal": c.evidence} for c in s.user_fit.checks],
        width="stretch", hide_index=True, key=f"{key}-fit",
    )
    mech = s.mechanism
    st.caption(
        f"Provenance: {len(mech.event_ids)} event(s), {mech.n_paths} signal path(s) "
        f"({mech.researchable_paths} researchable), consequences: {', '.join(mech.consequence_labels)}; fidelity "
        f"{', '.join(f.replace('_', ' ').lower() for f in mech.fidelities)}. {mech.note}"
    )


def render_signal_ranking(ranked: RankedSignalSet, *, key: str) -> None:
    st.caption(
        "Which candidate signals deserve portfolio consideration first: research merit from the 2018-2022 screen "
        "(screen outcome → direction of evidence → stability across years → cost headroom → data quality → t), "
        "never changed by your mandate; your mandate can exclude a signal and its fit is shown beside it. Correlated "
        "or same-instrument signals count as one exposure. No weights, sizes or trade instructions."
    )
    if ranked.ranked:
        st.dataframe(ranked_rows(ranked), width="stretch", hide_index=True, column_config=_RANKED_COLUMNS,
                     key=f"{key}-ranked")
    else:
        st.caption("Nothing ranked yet -- run the factor diagnostics in the candidate section to rank these signals.")
    tabs = st.tabs(
        ["Why this rank?", f"Exposures & redundancy ({ranked.independent_exposures})",
         f"Not rankable ({len(ranked.not_rankable)})", f"Excluded by mandate ({len(ranked.excluded)})"],
        key=f"{key}-ranking-tabs",
    )
    with tabs[0]:
        options = [s.candidate_signal_id for s in (*ranked.ranked, *ranked.excluded, *ranked.not_rankable)]
        label = {s.candidate_signal_id: (f"#{s.rank} " if s.rank else "") + f"{s.name} · {s.expression}"
                 for s in ranked.signals}
        chosen = st.selectbox("Signal", options, format_func=label.__getitem__, key=f"{key}-ranking-why")
        _render_detail(ranked.signal(chosen), key=f"{key}-detail")
    with tabs[1]:
        if not ranked.exposure_groups:
            st.caption("No ranked signal yet -- exposures are grouped once signals are screened.")
        else:
            st.dataframe(exposure_rows(ranked), width="stretch", hide_index=True, key=f"{key}-exposures",
                         column_config={"Max |rho|": st.column_config.NumberColumn(format="%.2f")})
            for t in ranked.event_theses:
                shown = ", ".join(t.consequence_labels[:3]) + (
                    f" (+{len(t.consequence_labels) - 3} more)" if len(t.consequence_labels) > 3 else "")
                st.caption(f"Event thesis -- {t.event_headline}: reaches {len(t.exposure_group_ids)} exposure(s) via "
                           f"{shown}. One news hypothesis, not independent bets.")
            if ranked.correlations:
                st.dataframe(_correlation_rows(ranked), width="stretch", hide_index=True, key=f"{key}-corr",
                             column_config={"Aligned rank correlation": st.column_config.NumberColumn(
                                 format="%+.2f")})
            st.caption(
                f"Signals on the same instrument, or whose expected-sign-aligned factors rank-correlate at |rho| >= "
                f"{ranked.policy.redundancy_correlation:g} over >= {ranked.policy.min_common_days} common days, are "
                "one exposure; its best-merit member leads. An opposed (negative) correlation links too: offsetting "
                "positions are not independent opportunities either."
            )
    with tabs[2]:
        rows = unranked_rows(ranked)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True, key=f"{key}-unranked")
        st.caption("Missing evidence is not low quality: these carry no rank rather than a low one.")
    with tabs[3]:
        rows = _excluded_rows(ranked)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True, key=f"{key}-excluded")
        st.caption("Your mandate removes these from portfolio consideration; their research merit is kept, unchanged.")
    st.caption(f"{ranked.not_portfolio_note} Family: {ranked.family_size} candidate(s) -- validation must correct "
               "across all of them.")


def chat_text(ranked: RankedSignalSet) -> str | None:
    if not ranked.signals:
        return None
    text = f"SIGNAL RANKING: {ranked.headline()}"
    if ranked.leads:
        leads = "; ".join(f"#{s.rank} {s.instrument} {s.expression} → {s.prediction_horizon_days}D "
                          f"({_SCREEN[s.merit.screen_status].lower()})" for s in ranked.leads[:3])
        text += f" Independent exposures: {leads}."
    return text + " Not portfolio weights."
