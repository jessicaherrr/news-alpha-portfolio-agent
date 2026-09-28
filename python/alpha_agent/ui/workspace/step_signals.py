"""Step 4 -- Signals: "Is there a quantitative signal?"

The thread's canonical candidate signals (Phase E: one registered factor
expression on one instrument, predicting its own forward return over a
declared horizon), their factor diagnostics on the 2018-2022 discovery
window, and how they rank for portfolio consideration (Phase F). A screen is
never a validation: statuses stay "candidate", "promising" or "screened"
until the scientific engine gives a verdict in Backtest. The candidate set is
always the event's full family -- focus only reorders what is emphasised.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.news_alpha import DOMAIN_LABELS, PathType
from alpha_agent.news_alpha.candidate_signals import CandidateSignal
from alpha_agent.screening.factor_diagnostics import ScreenStatus
from alpha_agent.ui import (
    candidate_signal_view,
    components,
    news_alpha_context,
    signal_ranking_view,
    translation_view,
)
from alpha_agent.ui.research_thread import ThreadPipeline, ThreadStep
from alpha_agent.ui.workspace.common import (
    PATH_TYPE_TEXT,
    chain_html,
    esc,
    info_html,
    meta_html,
    subhead,
)
from alpha_agent.ui.workspace.shell import render_step_heading

__all__ = ["STATUS_TEXT", "render"]

#: User-facing status -- never a validation word.
STATUS_TEXT: dict[ScreenStatus | None, str] = {
    None: "Candidate · not screened yet",
    ScreenStatus.SCREEN_CONTINUE: "Promising · screen supports validation",
    ScreenStatus.NO_SCREEN_SUPPORT: "Screened · no support",
    ScreenStatus.CONTRADICTS_EXPECTED_SIGN: "Screened · contradicts its expected sign",
    ScreenStatus.INSUFFICIENT_DATA: "Screened · insufficient data",
}
_FILTER_KEY = "thread-signal-filter-{}"
_STRATEGY_KEY = "thread-strategy-route-{}"


def _fmt(v: float | None, spec: str) -> str:
    return "--" if v is None else format(v, spec)


def render(pipe: ThreadPipeline) -> None:
    render_step_heading(ThreadStep.SIGNALS)
    cset, screens = pipe.candidates, pipe.screens
    _render_origin(pipe)
    if cset.is_empty:
        components.empty_state(
            "No candidate signal yet",
            "No measurement on this thread's markets is real, point-in-time safe and computable on history yet, so "
            "there is nothing quantitative to test. Market → Measure it shows what is missing.",
            key="thread-signals-empty",
        )
        _render_strategy_route(pipe)
        return

    counts = {s: sum(candidate_signal_view.screen_status_of(c, screens) is s for c in cset.candidates)
              for s in ScreenStatus}
    with components.metric_row("thread-signal-summary"):
        cols = st.columns(4)
        cols[0].metric("Candidate signals", len(cset.candidates),
                       help="Every testable factor hypothesis this event's measurable markets produce -- the full "
                            "family validation must later correct across.")
        cols[1].metric("Screened", len(screens), help="Factor diagnostics computed on 2018-2022 discovery data.")
        cols[2].metric("Promising", counts[ScreenStatus.SCREEN_CONTINUE],
                       help="The screen supports taking these to validation. Not validated.")
        cols[3].metric("Instruments", len(cset.instruments()))

    unscreened = [c for c in cset.candidates if c.candidate_signal_id not in screens]
    if unscreened:
        n_futures = len({c.spec.instrument for c in unscreened if c.spec.domain.value == "FUTURES"})
        wait = f" -- about {n_futures} minute(s) for futures history" if n_futures else ""
        with st.container(horizontal=True, vertical_alignment="center"):
            if st.button(f"Run factor diagnostics ({len(unscreened)} unscreened)", key="thread-run-screens",
                         type="primary", icon=":material/query_stats:",
                         help="Computes time-series ICs, decay, stability by year, coverage, turnover and cost "
                              "headroom on already-acquired 2018-2022 bars. No download, no cost; never reads the "
                              "2023-2024 validation window or the 2025 holdout. Results are cached." + wait):
                candidate_signal_view.run_screens(cset)
                st.rerun()
            st.caption("Unconditional factor screens -- not event-conditioned evidence, and not a validation.")

    _render_cards(pipe)
    _render_ranking(pipe)
    _render_strategy_route(pipe)

    with st.expander("Advanced details", icon=":material/tune:"):
        line = candidate_signal_view.summary_line(cset, screens)
        if line:
            st.caption(line)
        st.dataframe(candidate_signal_view.candidate_rows(cset, screens), hide_index=True, width="stretch")
        rows = candidate_signal_view.diagnostic_rows(cset, screens)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        refused = candidate_signal_view.refusal_rows(cset)
        if refused:
            st.markdown("**Not candidates**")
            st.dataframe(refused, hide_index=True, width="stretch")
        st.caption(cset.not_validated_note)


def _render_origin(pipe: ThreadPipeline) -> None:
    group = pipe.focus_group()
    followed = pipe.followed_paths()
    anchor = pipe.graph.anchors[0].label if pipe.graph.anchors else pipe.scan.event.headline
    if group is not None:
        path = next((p for p in followed if set(group.path_ids) & {p.path_id}), None) or \
            pipe.paths.path(group.path_ids[0])
        labels = list(path.state_labels)
        ptype = PATH_TYPE_TEXT[path.path_type]
        market = f"{group.concept}" + (f" ({', '.join(group.symbols)})" if group.symbols else "")
    elif followed:
        labels, ptype, market = list(followed[0].state_labels), PATH_TYPE_TEXT[followed[0].path_type], None
    else:
        labels, ptype, market = [anchor], "All paths", None
    st.markdown(
        f'<div class="aa-fact-k">Origin · {esc(ptype)}' + info_html(
            "Where these signals come from: event → economic path → market. Every candidate below keeps its full "
            "lineage under Lineage.") + "</div>" + chain_html(labels, market=market),
        unsafe_allow_html=True,
    )


def _render_cards(pipe: ThreadPipeline) -> None:
    cset, screens = pipe.candidates, pipe.screens
    thread = pipe.thread
    subhead("Candidate signals", "Each is one registered factor formula on one instrument, predicting that "
            "instrument's own forward return over a declared horizon with a declared sign. The formula shown is the "
            "one that runs.")
    present = [t for t in PathType if any(t in c.path_types for c in cset.candidates)]
    options = ["all", *present]
    labels = {"all": "All paths", **{t: PATH_TYPE_TEXT[t] for t in PathType}}
    choice = st.segmented_control("Compare by path", options, default="all", format_func=labels.__getitem__,
                                  key=_FILTER_KEY.format(thread.thread_id), label_visibility="collapsed") or "all"
    focus = pipe.focus_group()
    focus_ids = set(focus.expression_ids) if focus is not None else set()
    shown = [c for c in cset.candidates if choice == "all" or choice in c.path_types]
    rows = candidate_signal_view.diagnostic_rows(cset, screens)
    screened = [c for c in cset.candidates if c.candidate_signal_id in screens]
    diag = {c.candidate_signal_id: r for c, r in zip(screened, rows, strict=True)}
    cols = st.columns(2)
    for i, c in enumerate(shown):
        with cols[i % 2]:
            on_focus = bool(focus_ids & {o.expression_id for o in c.origins})
            _render_signal_card(pipe, c, diag.get(c.candidate_signal_id), on_focus=on_focus)


def _render_signal_card(pipe: ThreadPipeline, c: CandidateSignal, diag: dict | None, *, on_focus: bool) -> None:
    status = candidate_signal_view.screen_status_of(c, pipe.screens)
    spec = c.spec
    with components.card(f"signal-{c.candidate_signal_id}"):
        tag = ' · <span class="aa-news-new">On your market</span>' if on_focus else ""
        st.markdown(
            f'<div class="aa-eyebrow">{esc(STATUS_TEXT[status])}{tag}</div>'
            f'<div class="aa-news-headline" style="margin-top:0.1rem">{esc(c.name)}</div>',
            unsafe_allow_html=True,
        )
        st.code(c.expression, language=None)
        st.markdown(meta_html([
            ("Asset class", DOMAIN_LABELS[spec.domain]),
            ("Instrument", spec.instrument),
            ("Horizon", spec.prediction_horizon.value),
            ("Lookback", f"{spec.formation_lookback} bars"),
            ("Path", " + ".join(PATH_TYPE_TEXT[t] for t in c.path_types) or "--"),
            ("Data", f"{spec.data.dataset} · PIT safe" if spec.data.pit_safe else spec.data.dataset),
        ]), unsafe_allow_html=True)
        if diag is not None:
            st.markdown(meta_html([
                ("TS Spearman IC", _fmt(diag["TS Spearman IC"], "+.3f")),
                ("t (n/h)", _fmt(diag["t (n/h)"], "+.2f")),
                ("TS Pearson IC", _fmt(diag["TS Pearson IC"], "+.3f")),
                ("Years w/ sign", str(diag["Years with expected sign"])),
                ("Coverage", _fmt(diag["Coverage"], ".0%")),
                ("Flips / yr", _fmt(diag["Sign flips / yr"], ".1f")),
                ("Break-even cost", "--" if diag["Break-even cost (bp)"] is None
                 else f'{diag["Break-even cost (bp)"]:.1f} bp'),
            ]), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1, st.popover("Lineage", icon=":material/timeline:", width="stretch",
                                  help="Why this signal? Event → mechanism → path → market → measurement → formula."):
            st.markdown(candidate_signal_view.lineage_markdown(c))
            st.table(candidate_signal_view.spec_rows(c), hide_index=True, border="horizontal")
        screen = pipe.screens.get(c.candidate_signal_id)
        with c2:
            if screen is None:
                st.caption("Diagnostics after screening.")
            else:
                with st.popover("Diagnostics", icon=":material/insights:", width="stretch",
                                help="Decay by horizon, stability by year, the factor series, coverage, turnover "
                                     "and cost."):
                    candidate_signal_view.render_diagnostic_detail(c, screen, key=f"thread-diag-{c.candidate_signal_id}")


def _render_ranking(pipe: ThreadPipeline) -> None:
    ranked = pipe.ranked
    subhead("How they rank for portfolio consideration",
            "Research merit from the screen (outcome → direction → stability → cost headroom → data quality), "
            "never changed by your setup. Correlated or same-instrument signals count as one exposure.",
            help="Ranked by the canonical signal-ranking policy. A signal without a screen gets no rank rather than a "
                 "low one. No weights, sizes or trade instructions here.")
    line = signal_ranking_view.summary_line(ranked)
    if line:
        st.caption(line)
    if ranked.leads:
        st.markdown("".join(
            f'<div class="aa-ctx-row"><div class="aa-ctx-value"><b>#{s.rank}</b> · {esc(s.instrument)} '
            f'<code>{esc(s.expression)}</code> → {s.prediction_horizon_days}D '
            f'<span class="aa-dim"> · {esc(s.summaries.reason_for_rank)}</span></div></div>'
            for s in ranked.leads), unsafe_allow_html=True)
    else:
        st.caption("Nothing is ranked until the signals are screened.")
    with st.expander("Ranking details", icon=":material/leaderboard:"):
        signal_ranking_view.render_signal_ranking(ranked, key=f"thread-ranking-{pipe.thread.thread_id}")


def _render_strategy_route(pipe: ThreadPipeline) -> None:
    """The alternative route: turn a certified futures market of this event
    into a single-market strategy hypothesis for Ask / Strategy Lab. A toggle
    rather than an expander -- the translation card has expanders of its own."""
    plan = news_alpha_context.handoff_plan(pipe.scan)
    if not plan.handoffs and not plan.gaps:
        return
    if st.toggle("Also test as a single-market strategy", key=_STRATEGY_KEY.format(pipe.thread.thread_id),
                 help="Translates the event into one falsifiable strategy hypothesis on a certified futures market "
                      "and hands it to Ask, where it is compiled and run through the C++ backtest. A separate route "
                      "from these factor signals."):
        translation_view.render_translation_handoffs(plan, card_key=f"thread-{pipe.thread.thread_id}")
