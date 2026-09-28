"""The Initial Impact Scan, rendered: one badge per asset class and the
"why these asset classes?" detail (Phase A triage), shared by the News feed
and a Research Thread's News step. Formatting only -- every level,
availability, horizon and confidence shown is the scan's own value.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.news_alpha import (
    DOMAIN_LABELS,
    AvailabilityStatus,
    InitialImpactAssessment,
    InitialImpactScan,
)
from alpha_agent.ui import components

__all__ = ["AVAILABILITY_TONE", "HORIZON_TEXT", "impact_badge", "render_triage_details"]

AVAILABILITY_TONE = {
    AvailabilityStatus.RESEARCH_READY: "READY",
    AvailabilityStatus.OBSERVATION_ONLY: "WATCH",
    AvailabilityStatus.DATA_NOT_ACQUIRED: "WARN",
    AvailabilityStatus.FOUNDATION_ONLY: "WARN",
    AvailabilityStatus.SYNTHETIC_ONLY: "WARN",
    AvailabilityStatus.NO_CANDIDATE_MARKETS: "NOT_AVAILABLE",
    AvailabilityStatus.EXCLUDED_BY_MANDATE: "NOT_AVAILABLE",
    AvailabilityStatus.DOMAIN_NOT_SUPPORTED: "NOT_AVAILABLE",
}

HORIZON_TEXT = {
    "INTRADAY_TO_DAYS": "intraday to days",
    "DAYS_TO_WEEKS": "days to weeks",
    "WEEKS_TO_MONTHS": "weeks to months",
}


def impact_badge(assessment: InitialImpactAssessment) -> str:
    label = DOMAIN_LABELS[assessment.asset_domain]
    if assessment.impact_level is None:
        why = "excluded" if assessment.availability is AvailabilityStatus.EXCLUDED_BY_MANDATE else "not supported"
        return components.badge("NOT_AVAILABLE", label=f"{label} · {why}")
    level = assessment.impact_level.value
    return components.badge(f"IMPACT_{level}", label=f"{label} · {level.replace('_', ' ')}")


def render_triage_details(scan: InitialImpactScan) -> None:
    for a in scan.assessments:
        label = DOMAIN_LABELS[a.asset_domain]
        tone = AVAILABILITY_TONE[a.availability]
        availability = components.badge(tone, label=a.availability.value.replace("_", " "))
        if not a.assessed:
            st.markdown(f"**{label}** &middot; {availability}", unsafe_allow_html=True)
            st.caption(a.availability_note)
            continue
        level = a.impact_level.value
        level_badge = components.badge(f"IMPACT_{level}", label=level.replace("_", " "))
        st.markdown(f"**{label}** &middot; {level_badge} {availability}", unsafe_allow_html=True)
        st.caption(a.availability_note)
        if a.candidate_markets:
            shown = [f"{m.symbol}{' ✓' if m.research_ready else ''}" for m in a.candidate_markets[:10]]
            more = f" (+{len(a.candidate_markets) - 10} more)" if len(a.candidate_markets) > 10 else ""
            st.caption(f"Where research could start ({' · '.join(a.candidate_groups)}): {', '.join(shown)}{more}"
                       + (" -- ✓ research-ready" if any(m.research_ready for m in a.candidate_markets) else ""))
        for d in a.possible_direction:
            st.caption(
                f"Possible direction: {d.exposure_label} {d.pressure.value} on a {d.shift.replace('_', ' ').lower()} "
                f"shift -- a hypothesis to test, not a forecast. {d.rationale}"
            )
        if a.expected_horizon is not None:
            alignment = a.horizon_alignment.value.lower() if a.horizon_alignment else "unknown"
            st.caption(
                f"Typical horizon: {HORIZON_TEXT[a.expected_horizon.value]} ({alignment} with your mandate). "
                f"{a.horizon_rationale}"
            )
        if a.confidence is not None:
            st.caption(f"Triage confidence: {a.confidence.value} (confidence in the classification, not in any outcome).")
        for note in a.uncertainty[1:]:
            st.caption(f"Uncertainty: {note}")
        for note in a.mandate_notes:
            st.caption(f"Mandate: {note}")
    st.caption(scan.not_final_note)
    rules = sorted({r for a in scan.assessments if a.assessed for r in a.reasoning_provenance})
    st.caption(f"Provenance: {scan.generated_by} · {', '.join(rules)} · mandate {scan.mandate_fingerprint[:20]}…")
