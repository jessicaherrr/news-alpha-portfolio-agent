"""Market Intelligence Data Completion Pass, Checkpoint F -- Scheduled
Events UI. Mirrors `alpha_agent.ui.market_news`'s pure/render split;
everything renders through `alpha_agent.ui.market_intel_context`.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import streamlit as st

from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.ui import components, market_intel_context

__all__ = ["event_row", "render_events_section", "time_until_label"]


def time_until_label(scheduled_at: datetime, *, now: datetime | None = None) -> str:
    """Timezone-aware only -- Section 26: "Never compare naive datetime
    with aware datetime." Raises if either side is naive rather than
    silently guessing a timezone."""
    now = now or datetime.now(UTC)
    if scheduled_at.tzinfo is None or now.tzinfo is None:
        raise ValueError("time_until_label requires timezone-aware datetimes")
    delta = scheduled_at - now
    if delta.total_seconds() < 0:
        return "past"
    hours = delta.total_seconds() / 3600.0
    if hours < 1:
        return f"{delta.total_seconds() / 60.0:.0f} min"
    if hours < 48:
        return f"{hours:.1f} hours"
    return f"{hours / 24.0:.1f} days"


def event_row(event: ScheduledMarketEvent, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    return {
        "Time (ET)": event.scheduled_at.astimezone(_et()).strftime("%Y-%m-%d %H:%M"),
        "Time Until": time_until_label(event.scheduled_at, now=now),
        "Event": event.name,
        "Importance": event.importance.value,
        "Affected Products": ", ".join(event.affected_products) if event.affected_products else "—",
        "Source": event.source_name,
        "Link": event.source_url,
    }


def _et():
    from zoneinfo import ZoneInfo

    return ZoneInfo("America/New_York")


def render_events_section(root: str, *, key_prefix: str = "market-events") -> None:
    with components.card(f"{key_prefix}-upcoming"):
        st.markdown('<div class="aa-gate-title">UPCOMING EVENTS</div>', unsafe_allow_html=True)
        health = market_intel_context.event_connector_health()
        parts = [f"{name}: {state.value}" for name, state in sorted(health.items())]
        st.caption(f"Official calendars · {' · '.join(parts)}")

        events = market_intel_context.upcoming_events()
        if not events:
            components.empty_state(
                "Scheduled Events", "No real upcoming events are currently available from any connected calendar.",
                key=f"{key_prefix}-empty",
            )
            return

        now = datetime.now(UTC)
        st.dataframe(
            [event_row(e, now=now) for e in events], width="stretch", hide_index=True,
            height=min(400, 44 + 35 * len(events)),
            column_config={"Link": st.column_config.LinkColumn("Source Link", display_text="Open")},
        )

        next_for_root = market_intel_context.next_event_for_product(root)
        next_high_impact = market_intel_context.next_high_impact_event_for_product(root)
        c1, c2 = st.columns(2)
        with c1:
            label = (
                f"{next_for_root.name} ({time_until_label(next_for_root.scheduled_at, now=now)})"
                if next_for_root else "None scheduled"
            )
            st.markdown(f'<div class="aa-metric-label">NEXT EVENT ({root})</div><div class="aa-hero-contract-value">{label}</div>', unsafe_allow_html=True)
        with c2:
            label = (
                f"{next_high_impact.name} ({time_until_label(next_high_impact.scheduled_at, now=now)})"
                if next_high_impact else "None scheduled"
            )
            st.markdown(f'<div class="aa-metric-label">NEXT HIGH-IMPACT EVENT ({root})</div><div class="aa-hero-contract-value">{label}</div>', unsafe_allow_html=True)

        st.caption(
            "Importance is deterministic (see importance_rule per event) -- never LLM-invented. "
            "Times shown in America/New_York; stored internally as UTC."
        )
