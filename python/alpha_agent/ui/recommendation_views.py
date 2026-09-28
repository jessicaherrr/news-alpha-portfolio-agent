"""Phase B1 -- shared rendering for Scientific Verdict / Research Promise /
User Fit, used by Dashboard, Strategies, Research Details, and Validation so
the three stay visually distinct everywhere (task spec sections 12/19) without
duplicating markup across those four view modules. Every function here only
formats a `ResearchCandidateSummary` (or its component breakdowns) a caller
already has from `alpha_agent.ui.services` -- nothing is computed here.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.recommendation import PersonalizationState, ResearchCandidateSummary
from alpha_agent.ui import components, palette

#: Research Promise / User Fit are deliberately never rendered through
#: `components.badge`/`palette.STATE_COLOR` -- those are reserved for the
#: scientific verdict vocabulary (PASS/REJECT/...) and reusing them here risks
#: a HIGH promise label being visually mistaken for a scientific PASS (task
#: spec sections 9/12: "Never let... User Fit... outrank... PASS").
_LABEL_COLOR = {"HIGH": palette.GREEN, "MEDIUM": palette.AMBER, "LOW": palette.GREY}


def _label_color(label: str) -> str:
    return _LABEL_COLOR.get(label, palette.GREY)


def format_user_fit(candidate: ResearchCandidateSummary) -> str:
    """The one place `user_fit_score` becomes display text (task spec
    sections 6/7): a `NOT_PERSONALIZED` candidate must never show `100%` or
    any other percentage -- there is nothing measured yet. Every other call
    site (Strategies table, Validation caption, this module's own card/metric
    renderers) goes through this function so the rule cannot be bypassed in
    one place and forgotten in another."""
    if candidate.user_fit_personalization_state == PersonalizationState.NOT_PERSONALIZED:
        return "Not personalized"
    return f"{candidate.user_fit_score:.0f}%"


def user_fit_sub_caption(candidate: ResearchCandidateSummary) -> str:
    if candidate.user_fit_personalization_state == PersonalizationState.NOT_PERSONALIZED:
        return "Set your preferences to personalize -- no measurable preference stated yet"
    n = candidate.user_fit_measurable_dimension_count
    plural = "dimension" if n == 1 else "dimensions"
    coverage_note = " (partial -- not every stated preference has evidence yet)" \
        if candidate.user_fit_personalization_state == PersonalizationState.PARTIAL else ""
    return f"Based on {n} measurable {plural}{coverage_note}"


def render_verdict_promise_fit_row(candidate: ResearchCandidateSummary, *, key_prefix: str) -> None:
    """The three-metric row task spec sections 6/12/19 require near the top
    of Research Details and (secondary, alongside the dominant scientific
    result) on Validation: Scientific Verdict, Research Promise, User Fit --
    visually distinct, never conflated."""
    with components.metric_row(key_prefix):
        c1, c2, c3 = st.columns(3)
        with c1:
            components.verdict_metric_card(
                f"{key_prefix}-verdict", "Scientific Verdict", candidate.scientific_verdict
            )
        with c2:
            components.metric_card(
                f"{key_prefix}-promise", "Research Promise",
                f"{candidate.research_promise_label}",
                sub=f"{candidate.research_promise_score:.0f} / 100 -- prioritization only, not a scientific claim",
                accent=_label_color(candidate.research_promise_label),
            )
        with c3:
            not_personalized = candidate.user_fit_personalization_state == PersonalizationState.NOT_PERSONALIZED
            components.metric_card(
                f"{key_prefix}-fit", "User Fit",
                format_user_fit(candidate),
                sub=user_fit_sub_caption(candidate),
                accent=palette.GREY if not_personalized else _label_color(candidate.user_fit_label),
            )


def render_candidate_card(candidate: ResearchCandidateSummary, *, key_prefix: str, rank: int | None = None) -> None:
    """One compact "Most Promising Research Candidates" row (task spec
    section 1's mockup): rank, market/strategy, verdict, promise, fit, and
    the deterministic why/blockers text -- never softened language for a
    REJECT/INCONCLUSIVE verdict (task spec section 9)."""
    with components.card(f"{key_prefix}-candidate-{candidate.experiment_id}"):
        title = f"#{rank} " if rank is not None else ""
        st.markdown(f"**{title}{candidate.root_symbol} {candidate.friendly_strategy_name}**")
        st.markdown(
            components.badge(candidate.scientific_verdict, label=f"Verdict: {candidate.scientific_verdict}")
            + f'&nbsp;&nbsp;<span style="color:{_label_color(candidate.research_promise_label)};'
              f'font-size:0.82rem;font-weight:600;">Promise: {candidate.research_promise_label} '
              f"({candidate.research_promise_score:.0f})</span>"
              f'&nbsp;&nbsp;<span style="color:var(--aa-text-secondary);font-size:0.82rem;">'
              f"Fit for You: {format_user_fit(candidate)}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Why promising: {candidate.why_promising}")
        if candidate.validation_blockers:
            st.caption("What blocks validation: " + " ".join(candidate.validation_blockers))
        st.caption(f"Next research direction: {candidate.next_research_direction}")


def render_validated_section(candidates: list[ResearchCandidateSummary], *, key_prefix: str) -> None:
    """Task spec section 18: PASS only, and a legitimate empty state (never
    an error) when there are none."""
    if not candidates:
        components.empty_state(
            "Validated Strategies",
            "No strategy currently satisfies all scientific validation requirements. "
            "This is a legitimate research result, not an error.",
            key=f"{key_prefix}-validated-empty",
        )
        return
    with components.card(f"{key_prefix}-validated"):
        st.markdown(f"**{len(candidates)} Validated Strateg{'y' if len(candidates) == 1 else 'ies'} (PASS)**")
        for c in candidates:
            sharpe = f"Sharpe {c.annualized_sharpe:.2f}" if c.annualized_sharpe is not None else "Sharpe N/A"
            components.status_row(f"{c.root_symbol} · {c.friendly_strategy_name}", "PASS", sharpe)


def render_promising_section(
    candidates: list[ResearchCandidateSummary], *, key_prefix: str, limit: int = 5
) -> None:
    """Task spec sections 1/17: the "Most Promising Research Candidates"
    list, ranked (by whatever order the caller already sorted `candidates`
    in -- this function never re-sorts)."""
    if not candidates:
        components.empty_state(
            "Promising Research Candidates",
            "No tested candidate carries committed evidence yet.",
            key=f"{key_prefix}-promising-empty",
        )
        return
    for i, c in enumerate(candidates[:limit], start=1):
        render_candidate_card(c, key_prefix=key_prefix, rank=i)
