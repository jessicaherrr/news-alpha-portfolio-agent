"""Step 1 -- News: "What happened?" The selected event in detail and its
Initial Impact Scan. Causal reasoning comes next; nothing here is a trade."""
from __future__ import annotations

import streamlit as st

from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem
from alpha_agent.news_alpha import DOMAIN_LABELS, ImpactLevel
from alpha_agent.ui import components, impact_scan_view, news_alpha_context, signal_path_view
from alpha_agent.ui.research_thread import ThreadPipeline, ThreadStep
from alpha_agent.ui.workspace.common import (
    ago,
    esc,
    facts_html,
    horizon_text,
    impact_row_html,
    level_text,
    subhead,
)
from alpha_agent.ui.workspace.shell import render_step_heading

__all__ = ["render"]

_KIND_TEXT = {"MARKET_NEWS": "News", "SCHEDULED_EVENT": "Scheduled release", "USER_DESCRIBED": "Described by you"}
_CONFIDENCE_TEXT = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}


def _source_url(source) -> str | None:
    if isinstance(source, MarketNewsItem | ScheduledMarketEvent):
        return getattr(source, "source_url", None)
    return None


def render(pipe: ThreadPipeline) -> None:
    render_step_heading(ThreadStep.NEWS)
    scan = pipe.scan
    event = scan.event
    source = pipe.thread.event.source_object()

    with components.card("thread-news-event"):
        category = event.source_category.replace("_", " ").title() if event.source_category else "--"
        st.markdown(
            f'<div class="aa-news-meta"><span>{esc(_KIND_TEXT.get(event.kind.value, event.kind.value))}</span>'
            f'<span class="aa-dot-sep">·</span><span>{esc(event.source_name)}</span>'
            f'<span class="aa-dot-sep">·</span><span>{esc(ago(event.observed_at))}</span></div>'
            f'<div class="aa-thread-title" style="font-size:1.25rem">{esc(event.headline)}</div>'
            + (f'<div class="aa-news-summary">{esc(event.summary)}</div>' if event.summary else ""),
            unsafe_allow_html=True,
        )
        themes = " · ".join(c.channel_label for c in scan.channels) or "No economic channel detected"
        entities = ", ".join(dict.fromkeys(g for a in scan.assessments for g in a.candidate_groups)) or "--"
        st.markdown(facts_html([
            ("Event type", _KIND_TEXT.get(event.kind.value, event.kind.value)),
            ("Source", event.source_name),
            ("Timestamp", event.observed_at.strftime("%Y-%m-%d %H:%M UTC")),
            ("Category", category),
            ("Themes", themes),
        ]), unsafe_allow_html=True)
        if entities != "--":
            st.markdown('<div class="aa-fact-k" style="margin-top:0.5rem">Related markets</div><div class="aa-tag-row">'
                        + "".join(f'<span class="aa-tag">{esc(e)}</span>' for e in entities.split(", ")) + "</div>",
                        unsafe_allow_html=True)
        url = _source_url(source)
        if url:
            st.markdown(f"[Open the original source ↗]({url})")

    subhead("Initial impact", "How relevant this event looks for each asset class, read against your Research Setup. "
            "A research-triage level -- never an expected return, a probability or a trade signal.",
            help="Computed by the deterministic initial impact scan (rule impact-level/1) from the event's economic "
                 "channels and how directly each asset class is exposed. Economic magnitude is not established here.")
    st.markdown(impact_row_html(scan), unsafe_allow_html=True)
    if pipe.adjusted is not None:
        # The mechanism check sits BESIDE the initial scan, never instead of it (Phase C).
        st.markdown(signal_path_view.mechanism_support_row(pipe.adjusted), unsafe_allow_html=True)
    if not any(a.impact_level not in (None, ImpactLevel.NONE) for a in scan.assessments):
        st.info("This event does not look relevant to any asset class in your Research Setup. You can still follow "
                "its reasoning, or widen your setup.", icon=":material/info:")
    rows = []
    for a in scan.assessments:
        if a.impact_level is None:
            continue
        rows.append({
            "Asset class": DOMAIN_LABELS[a.asset_domain],
            "Impact": level_text(a.impact_level),
            "Typical horizon": horizon_text(a.expected_horizon),
            "Triage confidence": _CONFIDENCE_TEXT.get(a.confidence.value, a.confidence.value) if a.confidence else "--",
            "Where research could start": ", ".join(a.candidate_groups[:3]) or "--",
        })
    with st.expander("Why these asset classes?", icon=":material/help:"):
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        impact_scan_view.render_triage_details(scan)

    _render_related_events(pipe)

    with st.expander("Provenance", icon=":material/fingerprint:"):
        components.provenance_row("Event id", event.event_id)
        components.provenance_row("Source type", event.source_type.value.replace("_", " ").title())
        components.provenance_row("Observed at (UTC)", event.observed_at.isoformat())
        components.provenance_row("Scan", scan.generated_by)
        rules = list(dict.fromkeys(r for a in scan.assessments if a.assessed for r in a.reasoning_provenance))
        components.provenance_row("Rules", ", ".join(rules) or "--")
        components.provenance_row("Research Setup", scan.mandate_fingerprint[:28] + "…")
        for ref in event.evidence_refs:
            components.provenance_row("Evidence", ref)


def _render_related_events(pipe: ThreadPipeline) -> None:
    """Other cached events on one of this event's economic channels -- the
    scan's own channel labels, matched as-is (no similarity score)."""
    channels = {c.channel for c in pipe.scan.channels}
    if not channels:
        return
    related = [
        s for s in news_alpha_context.cached_event_scans(pipe.mandate, window_hours=24 * 30, limit=20)
        if s.event.event_id != pipe.scan.event.event_id and channels & {c.channel for c in s.channels}
    ][:3]
    if not related:
        return
    subhead("Related events", "Other recent items on the same economic channel.")
    for s in related:
        st.markdown(
            f'<div class="aa-news-meta"><span>{esc(s.event.source_name)}</span><span class="aa-dot-sep">·</span>'
            f'<span>{esc(ago(s.event.observed_at))}</span></div>'
            f'<div style="font-size:0.88rem;color:var(--aa-text);margin-bottom:0.4rem">{esc(s.event.headline)}</div>',
            unsafe_allow_html=True,
        )
