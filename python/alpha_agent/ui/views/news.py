"""News -- the default landing and discovery entry: "what is worth
researching?"

Real cached news items, official releases and scheduled events (plus events
the user describes), each with its Initial Impact Scan against the Research
Setup. "Research" opens -- or re-opens -- the event's persistent Research
Thread; nothing here is a BUY/SELL, a return or a verdict.

First paint never waits on the network: the feed reads only what an explicit
refresh already cached (`news_alpha_context.cached_event_scans`); "Refresh
news" is the one explicit action that fetches (free official sources only --
the same `market_intel_context.force_refresh` Markets uses).
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.news_alpha import InitialImpactScan, ResearchMandate
from alpha_agent.ui import (
    components,
    impact_scan_view,
    layout,
    market_intel_context,
    news_alpha_context,
    research_setup_view,
    research_thread,
)
from alpha_agent.ui.research_thread import STEP_LABELS, STEPS
from alpha_agent.ui.workspace.common import ago, esc, impact_row_html, info_html

PAGE_TITLE = "News"

_WINDOWS: dict[str, int] = {"24 hours": 24, "7 days": 24 * 7, "30 days": 24 * 30}
_WINDOW_KEY = "news-window"
_KIND_TEXT = {"MARKET_NEWS": "News", "SCHEDULED_EVENT": "Scheduled release", "USER_DESCRIBED": "Your event"}
_MAX_THREADS_SHOWN = 3


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active="news")
    layout.render_header(subtitle="From market news to tested research.")

    mandate = news_alpha_context.current_mandate()
    st.markdown('<div class="aa-page-title">What is worth researching?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aa-page-lede">Recent news, official releases and scheduled events, each read against your '
        "Research Setup. Open one as a research thread to follow it from economic reasoning to a tested "
        "portfolio. Nothing on this page is a trade recommendation.</div>",
        unsafe_allow_html=True,
    )
    research_setup_view.render_setup_summary(mandate, key="news")
    research_setup_view.render_setup_dialog_if_open(mandate)

    _render_continue_research()
    _render_describe_event()
    _render_feed(mandate)
    layout.render_disclaimer()


# ---------------------------------------------------------------------------
# continue where you left off
# ---------------------------------------------------------------------------


def _open_thread_page() -> None:
    from alpha_agent.ui.views import workspace

    st.switch_page(st.Page(workspace.render, url_path="research"))


def _render_continue_research() -> None:
    threads = research_thread.THREAD_STORE.list()
    if not threads:
        return
    st.markdown('<div class="aa-subhead">Continue your research</div>', unsafe_allow_html=True)
    shown = threads[:_MAX_THREADS_SHOWN]
    cols = st.columns(_MAX_THREADS_SHOWN)
    for col, t in zip(cols, shown, strict=False):
        with col, components.card(f"news-thread-{t.thread_id}"):
            step_no = STEPS.index(t.current_step) + 1
            st.markdown(
                f'<div class="aa-eyebrow">Step {step_no} of {len(STEPS)} · {esc(STEP_LABELS[t.current_step])}</div>'
                f'<div class="aa-news-headline">{esc(t.name)}</div>'
                f'<div class="aa-news-summary">{esc(t.event.headline)}</div>'
                f'<div class="aa-news-meta" style="margin-top:0.35rem">Updated {esc(ago(t.updated_at))}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Continue", key=f"news-continue-{t.thread_id}", icon=":material/arrow_forward:",
                         icon_position="right", help="Reopen this research thread where you left off."):
                research_thread.set_active(t.thread_id)
                _open_thread_page()
    if len(threads) > _MAX_THREADS_SHOWN:
        st.caption(f"{len(threads) - _MAX_THREADS_SHOWN} more thread(s) in Research.")


# ---------------------------------------------------------------------------
# describe an event
# ---------------------------------------------------------------------------


def _render_describe_event() -> None:
    with st.container(horizontal=True, vertical_alignment="bottom", key="news-describe"):
        text = st.text_input(
            "Have an event in mind?", key="news-describe-text",
            placeholder="Describe an event, e.g. “Microsoft expands AI infrastructure spending”",
            help="Anything you read elsewhere. It is scanned against your Research Setup exactly like a cached "
                 "news item, and kept for this session unless you open it as a research thread.",
        )
        if st.button("Scan impact", key="news-describe-go", icon=":material/radar:",
                     help="Runs the initial impact scan: which asset classes this event may deserve research in. "
                          "Instant and offline."):
            if news_alpha_context.add_user_event(text) is None:
                st.warning("Describe the event in a few words first.")
            else:
                st.rerun()


# ---------------------------------------------------------------------------
# the feed
# ---------------------------------------------------------------------------


def _render_feed(mandate: ResearchMandate) -> None:
    st.markdown('<div class="aa-subhead">Latest' + info_html(
        "Each bar shows the initial impact scan for one asset class: how relevant the event looks for research "
        "there, read against your Research Setup. A research-triage level -- never an expected return or a trade "
        "signal.") + "</div>", unsafe_allow_html=True)
    with st.container(horizontal=True, vertical_alignment="center", key="news-feed-controls"):
        window = st.segmented_control(
            "Window", list(_WINDOWS), default="7 days", key=_WINDOW_KEY, label_visibility="collapsed",
            help="How far back to look in the local news cache.",
        ) or "7 days"
        if st.button("Refresh news", key="news-refresh", icon=":material/refresh:", type="tertiary",
                     help="Fetches the latest official releases (Federal Reserve, EIA, BLS, USDA). Free public "
                          "sources; takes a few seconds. Nothing refreshes on its own."):
            with st.spinner("Fetching the latest official releases…"):
                market_intel_context.force_refresh()
            st.rerun()
    last = market_intel_context.last_refresh_at()
    st.caption(
        f"Last refreshed {ago(last)}." if last else
        "Showing what is already cached on this machine. Use Refresh news to fetch the latest releases."
    )

    user_scans = news_alpha_context.user_event_scans(mandate)
    cached = news_alpha_context.cached_event_scans(mandate, window_hours=_WINDOWS[window], limit=12)
    if not user_scans and not cached:
        components.empty_state(
            "Nothing to research in this window yet",
            "No cached news or scheduled event in this window touches an asset class in your Research Setup. "
            "Refresh news, widen the window, or describe an event above.",
            key="news-empty",
        )
        return
    for scan in (*user_scans, *cached):
        _render_event_card(scan, mandate)


def _when(scan: InitialImpactScan) -> list[str]:
    """A scheduled release is dated by when it is scheduled, not when it was
    scanned."""
    source = scan.event.source
    if isinstance(source, ScheduledMarketEvent):
        return [f'<span>Scheduled {esc(source.scheduled_at.strftime("%b %d, %H:%M UTC"))}</span>',
                f"<span>{esc(ago(source.scheduled_at))}</span>"]
    return [f'<span>{esc(scan.event.observed_at.strftime("%b %d, %H:%M UTC"))}</span>',
            f"<span>{esc(ago(scan.event.observed_at))}</span>"]


def _render_event_card(scan: InitialImpactScan, mandate: ResearchMandate) -> None:
    event = scan.event
    key = f"news-{event.kind.value}-{event.event_id}"
    thread_id = research_thread.thread_id_for_event(event.event_id)
    existing = research_thread.THREAD_STORE.get(thread_id)
    with components.card(key):
        meta = [
            f'<span>{esc(_KIND_TEXT.get(event.kind.value, event.kind.value))}</span>',
            f'<span>{esc(event.source_name)}</span>',
            *_when(scan),
        ]
        if event.source_category:
            meta.append(f'<span>{esc(event.source_category.replace("_", " ").title())}</span>')
        if existing is not None:
            meta.append(f'<span class="aa-news-new">In research · {esc(STEP_LABELS[existing.current_step])}</span>')
        st.markdown(
            '<div class="aa-news-meta">' + '<span class="aa-dot-sep">·</span>'.join(meta) + "</div>"
            f'<div class="aa-news-headline">{esc(event.headline)}</div>'
            + (f'<div class="aa-news-summary">{esc(event.summary)}</div>' if event.summary else ""),
            unsafe_allow_html=True,
        )
        st.markdown(impact_row_html(scan), unsafe_allow_html=True)
        if scan.channels:
            st.markdown(
                '<div class="aa-tag-row">' + "".join(
                    f'<span class="aa-tag">{esc(c.channel_label)}</span>' for c in scan.channels) + "</div>",
                unsafe_allow_html=True,
            )
        with st.container(horizontal=True, vertical_alignment="center"):
            label = "Continue research" if existing is not None else "Research"
            if st.button(label, key=f"{key}-research", type="primary", icon=":material/arrow_forward:",
                         icon_position="right",
                         help="Open this event as a research thread: reasoning, affected markets, signals, a "
                              "portfolio and a real backtest -- one step at a time. Your progress is saved."):
                research_thread.open_thread_for_scan(scan, mandate)
                _open_thread_page()
        with st.expander("Explore: why these asset classes?", key=f"{key}-explore"):
            impact_scan_view.render_triage_details(scan)
