"""Portfolio -- signals from ALL research threads, ranked together and
combined under the Research Setup (the cross-event view the Agent page's
retired "portfolio consideration across events" card used to hold).

Ranking (Phase F), the plan (Phase G, sized by the C++ allocator) and the
validation gate + route memory (Phase H) are the canonical backend objects,
rendered with the same renderers a single thread uses. Nothing here is a
trade instruction; validation runs from the command line.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.alpha_memory.signal_path_evidence import EventResearch
from alpha_agent.ui import (
    components,
    layout,
    news_alpha_context,
    portfolio_plan_view,
    research_loop_view,
    research_setup_view,
    research_thread,
    signal_ranking_view,
)
from alpha_agent.ui.research_thread import STEP_LABELS
from alpha_agent.ui.workspace import step_portfolio
from alpha_agent.ui.workspace.common import esc, subhead

PAGE_TITLE = "Portfolio"
_MODE_KEY = "portfolio-page-portfolio-mode"  # the key research_loop_view reads


def _go_news() -> None:
    from alpha_agent.ui.views import news

    st.switch_page(st.Page(news.render, url_path="news"))


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active="portfolio")
    layout.render_header(subtitle="Every research thread's signals, combined under your Research Setup.")

    mandate = news_alpha_context.current_mandate()
    st.markdown('<div class="aa-page-title">Portfolio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aa-page-lede">Signals from all your research threads, ranked together -- correlated signals '
        "count as one exposure -- and sized by the C++ allocator under your Research Setup. A research portfolio, "
        "not a trade instruction.</div>",
        unsafe_allow_html=True,
    )
    research_setup_view.render_setup_summary(mandate, key="portfolio")
    research_setup_view.render_setup_dialog_if_open(mandate)

    threads = research_thread.THREAD_STORE.list()
    pipes = [research_thread.pipeline_for(t, mandate) for t in threads]
    with_signals = [p for p in pipes if not p.candidates.is_empty]
    if not with_signals:
        components.empty_state(
            "No signals to combine yet",
            "A portfolio draws on the candidate signals of your research threads. Open an event from News and take "
            "it through Signals first." if threads else "You have no research threads yet. Start one from News.",
            key="portfolio-empty",
        )
        if st.button("Go to News", key="portfolio-go-news", icon=":material/newspaper:"):
            _go_news()
        _render_paper_note()
        layout.render_disclaimer()
        return

    subhead("Contributing threads")
    st.dataframe([{
        "Thread": p.thread.name, "Event": p.thread.event.headline,
        "Stage": STEP_LABELS[p.thread.current_step], "Candidate signals": len(p.candidates.candidates),
        "Screened": len(p.screens),
    } for p in with_signals], hide_index=True, width="stretch")

    sets = [p.candidates for p in with_signals]
    ranked = signal_ranking_view.rank_for(sets, mandate)
    subhead("Ranked for portfolio consideration")
    line = signal_ranking_view.summary_line(ranked)
    if line:
        st.caption(line)
    if ranked.leads:
        st.markdown("".join(
            f'<div class="aa-ctx-row"><div class="aa-ctx-value"><b>#{s.rank}</b> · {esc(s.instrument)} '
            f'<code>{esc(s.expression)}</code> → {s.prediction_horizon_days}D '
            f'<span class="aa-dim"> · {esc(s.summaries.reason_for_rank)}</span></div></div>'
            for s in ranked.leads), unsafe_allow_html=True)
    with st.expander("Ranking details", icon=":material/leaderboard:"):
        signal_ranking_view.render_signal_ranking(ranked, key="portfolio-page-ranking")

    subhead("Portfolio plan")
    mode = step_portfolio.render_mode_control(_MODE_KEY)
    plan, missing = portfolio_plan_view.build_plan(ranked, sets, mandate, mode)
    step_portfolio.render_plan(plan, missing, key="portfolio-page-plan")

    subhead("Validation & research memory")
    events = [EventResearch(discovery=p.paths, expressions=p.expressions, candidates=p.candidates)
              for p in with_signals]
    research_loop_view.render_research_loop_card(ranked, portfolio_plan_view.screens_for(sets), mandate, events,
                                                 key="portfolio-page")
    _render_paper_note()
    layout.render_disclaimer()


def _render_paper_note() -> None:
    st.caption("Paper Trading (under Tools) replays validated single strategies with real risk limits. A research "
               "portfolio reaches it only after it passes validation.")

