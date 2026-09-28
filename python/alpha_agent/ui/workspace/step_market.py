"""Step 3 -- Market: "Where is this economic effect showing up in markets --
and can we measure it?"

Impact-driven, never a market browser: only the markets the thread's signal
paths reach appear (the canonical `AssetExpressionPlan`), starting from the
paths the user follows. Each market says WHY it is affected (its economic
lineage), where it sits in the Research Setup, what its price is doing
(observation only), and -- in "Measure it" -- the difference between the
economic concept, the ideal measurement, the actual data field and what can
become a quantitative signal. Unavailable data is always shown as
unavailable. Nothing is ranked here; that belongs to Signals and Portfolio.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.news_alpha import DOMAIN_LABELS, ExpressionStatus, ResolutionStatus
from alpha_agent.ui import asset_expression_view, components, research_thread
from alpha_agent.ui.research_thread import MarketGroup, ThreadPipeline, ThreadStep
from alpha_agent.ui.workspace import market_data_panel
from alpha_agent.ui.workspace.common import (
    PATH_TYPE_TEXT,
    chain_html,
    esc,
    info_html,
    kind_html,
    lag_text,
    level_text,
    subhead,
)
from alpha_agent.ui.workspace.shell import render_step_heading

__all__ = ["render"]

_SHOW_ALL_KEY = "thread-market-all-{}"
_STATUS_TEXT = {
    ExpressionStatus.CONTINUES: "In your setup",
    ExpressionStatus.EXCLUDED_BY_MANDATE: "Outside your setup",
    ExpressionStatus.DOMAIN_UNAVAILABLE: "Asset class not available",
    ExpressionStatus.NO_INSTRUMENT: "Concept only -- no tradable instrument",
    ExpressionStatus.REMOVED_BY_CONSTRAINTS: "Removed by your instrument limits",
}
_FIDELITY_TEXT = {
    "DIRECT_UNDERLYING": "Direct underlying", "DIRECT_COMPANY": "The company itself",
    "SEGMENT_EXPOSURE": "One segment of the company", "CONSTITUENT_EXPOSURE": "Holds exposed names",
    "ECOSYSTEM_PROXY": "Ecosystem proxy", "MACRO_PROXY": "Macro proxy",
}
_DATA_TEXT = {
    ResolutionStatus.AVAILABLE: ("Available", "yes"),
    ResolutionStatus.AVAILABLE_WITH_PROXY: ("Available via proxy", "part"),
    ResolutionStatus.PARTIAL: ("Partial", "part"),
    ResolutionStatus.MISSING: ("Missing", "no"),
    ResolutionStatus.NOT_PIT_SAFE: ("Not point-in-time safe", "part"),
    ResolutionStatus.DOMAIN_UNAVAILABLE: ("Asset class unavailable", "no"),
    ResolutionStatus.NOT_EXECUTABLE: ("Data exists, not computable yet", "part"),
}
_PIT_TEXT = {True: ("PIT safe", "yes"), False: ("Not PIT safe", "part"), None: ("Unknown", "no")}


def _state(text: str, tone: str) -> str:
    return f'<span class="aa-state aa-state-{tone}">{esc(text)}</span>'


def _in_focus(pipe: ThreadPipeline, show_all: bool) -> tuple[tuple[MarketGroup, ...], bool]:
    groups = pipe.market_groups()
    followed = set(pipe.thread.followed_path_ids)
    if not followed or show_all:
        return groups, False
    return tuple(g for g in groups if followed & set(g.path_ids)), True


def _set_focus(pipe: ThreadPipeline, group: MarketGroup | None) -> None:
    research_thread.update_thread(pipe.thread, focus_expression_id=None if group is None else group.first.expression_id)


def render(pipe: ThreadPipeline) -> None:
    """The Research Thread's Market step -- the SAME affected-markets view the
    top-level Market destination shows for the active thread."""
    render_step_heading(ThreadStep.MARKET)
    render_affected_markets(pipe)


def _label(g: MarketGroup) -> str:
    symbols = ", ".join(g.symbols[:3]) + ("…" if len(g.symbols) > 3 else "")
    return f"{symbols} · {g.concept}" if symbols else g.concept


def _on_select(key: str, thread_id: str, expression_ids: dict[str, str]) -> None:
    """Persist the chosen market as the thread's market focus."""
    chosen = st.session_state.get(key)
    thread = research_thread.THREAD_STORE.get(thread_id)
    if thread is not None and chosen in expression_ids:
        research_thread.update_thread(thread, focus_expression_id=expression_ids[chosen])


def render_affected_markets(pipe: ThreadPipeline) -> None:
    """Affected Markets: ONLY the markets this thread's event reaches through
    its mechanism graph, signal paths and asset expressions (never a general
    browser). Market-first: pick a market, see its price, its market data,
    why it is affected and what can measure the effect -- without leaving the
    page. Shared by the Research Thread's Market step and the top-level
    Market destination; the thread supplies the event, followed paths,
    expressions, measurements and Research Setup."""
    plan = pipe.expressions
    if plan.is_empty:
        components.empty_state(
            "No market expresses these consequences yet",
            "No signal path reaches an economic consequence with a reviewed market expression. Reasoning shows "
            "where the transmission stops.",
            key="thread-market-empty",
        )
        return

    show_all_key = _SHOW_ALL_KEY.format(pipe.thread.thread_id)
    groups, narrowed = _in_focus(pipe, st.session_state.get(show_all_key, False))
    in_setup = [g for g in groups if g.status is ExpressionStatus.CONTINUES]
    others = [g for g in groups if g.status is not ExpressionStatus.CONTINUES]
    usable = {m.spec_id for m in plan.usable_measurements()}
    ready = [g for g in in_setup if usable & set(g.measurement_ids)]
    waiting = [g for g in in_setup if not usable & set(g.measurement_ids)]

    scope = (f"reached by the {len(pipe.thread.followed_path_ids)} path(s) you follow" if narrowed
             else "reached by this event's signal paths")
    st.markdown(
        f'<div class="aa-subnote">{len(in_setup)} market{"s" if len(in_setup) != 1 else ""} {esc(scope)} are in your '
        f"Research Setup; <b style='color:var(--aa-text)'>{len(ready)}</b> can be measured on real, point-in-time "
        f"safe history today. {len(others)} more sit outside your setup or have no tradable instrument.</div>",
        unsafe_allow_html=True,
    )
    if pipe.thread.followed_path_ids:
        st.toggle("Show markets from every path", key=show_all_key,
                  help="Off: only the markets your followed paths reach. On: every market this event's paths reach.")

    ordered = [*ready, *waiting]
    if not ordered:
        st.caption("None of the reached markets is in your Research Setup -- see below, or widen your setup.")
    else:
        focus = pipe.focus_group()
        # Default (display only, until the user picks one): the first measurable market with a
        # connected price feed, so the page leads with a chart when one exists.
        priced = next((g for g in ordered if market_data_panel.has_price_feed(g)), ordered[0])
        selected = focus if focus is not None and focus.key in {g.key for g in ordered} else priced
        if focus is None or focus.key != selected.key:
            # the shown market IS the thread's current market (the context panel must agree)
            pipe.thread = research_thread.update_thread(pipe.thread, focus_expression_id=selected.first.expression_id)
        select_key = f"mx-select-{pipe.thread.thread_id}"
        if st.session_state.get(select_key) != selected.key:
            st.session_state[select_key] = selected.key
        labels = {g.key: _label(g) for g in ordered}
        st.pills(
            "Affected markets", [g.key for g in ordered], format_func=labels.__getitem__, key=select_key,
            width="stretch", wrap=True,
            on_change=_on_select, args=(select_key, pipe.thread.thread_id, {g.key: g.first.expression_id
                                                                           for g in ordered}),
            help="Every market this event reaches that your Research Setup covers -- measurable ones first. Pick one "
                 "to see its price, market data, why it is affected and what can measure the effect.",
        )
        _render_market_hero(pipe, selected, usable)
        _render_market_table(pipe, ordered, usable)

    _render_consequence_map(pipe, groups)

    if others:
        with st.expander(f"Other possible exposures ({len(others)})", icon=":material/visibility_off:"):
            st.caption("Markets the economics reach but research cannot continue in: outside your Research Setup, an "
                       "asset class the platform does not support yet, or a concept with no tradable instrument.")
            st.dataframe([{
                "Market": g.concept, "Asset class": DOMAIN_LABELS[g.domain],
                "Instruments": ", ".join(g.symbols) or "--", "Why not": _STATUS_TEXT[g.status],
                "Detail": g.first.status_note or "--",
            } for g in others], hide_index=True, width="stretch")

    with st.expander("Advanced details", icon=":material/tune:"):
        line = asset_expression_view.summary_line(plan)
        if line:
            st.caption(line)
        asset_expression_view.render_asset_expressions(plan, key=f"thread-{pipe.thread.thread_id}")


def _render_market_hero(pipe: ThreadPipeline, g: MarketGroup, usable: set[str]) -> None:
    """The selected market, market-first: name, impact line, lineage, price
    trend + market data, then Measure it."""
    types, horizons, confidence = _route_facts(pipe, g)
    exposure = pipe.initial_exposure(g)
    impact = (components.badge(f"IMPACT_{exposure[1].value}", label=f"{level_text(exposure[1])} impact")
              if exposure else components.badge("NOT_AVAILABLE", label="Reached via the mechanism only"))
    fidelity = " · ".join(dict.fromkeys(_FIDELITY_TEXT.get(e.fidelity.value, e.fidelity.value) for e in g.expressions))
    measured = set(g.measurement_ids)
    with components.card(f"mx-hero-{g.key}"):
        name = f"{g.concept}" + (f" · {', '.join(g.symbols)}" if g.symbols else "")
        st.markdown(
            f'<div class="aa-eyebrow">{esc(DOMAIN_LABELS[g.domain])} · {esc(fidelity)}</div>'
            f'<div class="aa-market-title">{esc(name)}</div>'
            f'<div class="aa-market-line">{impact}<span>{esc(types)}</span><span class="aa-dot-sep">·</span>'
            f"<span>{esc(horizons)}</span><span class=\"aa-dot-sep\">·</span>"
            f"<span>Route confidence {esc(confidence)}</span><span class=\"aa-dot-sep\">·</span>"
            f"<span>{len(usable & measured)} of {len(measured)} measurable</span></div>"
            '<div class="aa-fact-k" style="margin-top:0.6rem">Why affected' + info_html(
                "The economic route from the event to this market -- the signal path you follow, or the first one "
                "that reaches it.") + "</div>" + _why_chain(pipe, g),
            unsafe_allow_html=True,
        )
        more = len(g.consequences) - 1
        if more > 0:
            st.caption(f"Also reached through {more} other consequence{'s' if more != 1 else ''}: "
                       f"{', '.join(g.consequences[1:4])}{'…' if more > 3 else ''}.")
        market_data_panel.render_market_data(g, key=f"mx-{g.key}")
        _render_measure_it(pipe, g)


def _render_market_table(pipe: ThreadPipeline, groups: list[MarketGroup], usable: set[str]) -> None:
    rows = []
    for g in groups:
        types, horizons, _confidence = _route_facts(pipe, g)
        exposure = pipe.initial_exposure(g)
        rows.append({
            "Market": g.concept, "Instruments": ", ".join(g.symbols) or "--", "Asset class": DOMAIN_LABELS[g.domain],
            "Impact": level_text(exposure[1]) if exposure else "Via mechanism only", "Path": types,
            "Horizon": horizons, "Measurable": f"{len(usable & set(g.measurement_ids))} of {len(g.measurement_ids)}",
        })
    with st.expander(f"All affected markets ({len(rows)})", icon=":material/table_rows:"):
        st.dataframe(rows, hide_index=True, width="stretch")


def _route_facts(pipe: ThreadPipeline, g: MarketGroup) -> tuple[str, str, str]:
    paths = [pipe.paths.path(pid) for pid in g.path_ids]
    types = " + ".join(dict.fromkeys(PATH_TYPE_TEXT[p.path_type] for p in paths)) or "--"
    horizons = " · ".join(dict.fromkeys(lag_text(pr.horizon) for e in g.expressions for pr in e.pressures)) or "--"
    confidence = " · ".join(dict.fromkeys(p.confidence.value.title() for p in paths)) or "--"
    return types, horizons, confidence


def _why_chain(pipe: ThreadPipeline, g: MarketGroup) -> str:
    followed = set(pipe.thread.followed_path_ids)
    ids = [pid for pid in g.path_ids if pid in followed] or list(g.path_ids)
    path = pipe.paths.path(ids[0]) if ids else None
    label = f"{g.concept}" + (f" ({', '.join(g.symbols)})" if g.symbols else "")
    if path is None:
        return chain_html([g.first.consequence_label], market=label)
    return chain_html(path.state_labels, market=label)


# ---------------------------------------------------------------------------
# consequence -> market map (the second, lightweight visual)
# ---------------------------------------------------------------------------


def _render_consequence_map(pipe: ThreadPipeline, groups: tuple[MarketGroup, ...]) -> None:
    by_consequence: dict[str, list[MarketGroup]] = {}
    for g in groups:
        for c in g.consequences:
            by_consequence.setdefault(c, []).append(g)
    if not by_consequence:
        return
    with st.expander("How the consequences reach markets", icon=":material/account_tree:", expanded=False):
        st.caption("Each economic consequence, and the markets that could express it. Solid tags are in your "
                   "Research Setup; dashed ones are not.")
        for consequence, members in by_consequence.items():
            exprs = [e for g in members for e in g.expressions if e.consequence_label == consequence]
            path_types = list(dict.fromkeys(pipe.paths.path(pid).path_type for e in exprs for pid in e.path_ids))
            tags = "".join(
                f'<span class="aa-tag" style="{"" if g.status is ExpressionStatus.CONTINUES else "opacity:0.55;border-style:dashed"}">'
                f'{esc(g.concept)}{" · " + esc(", ".join(g.symbols)) if g.symbols else ""}</span>'
                for g in members
            )
            kinds = " ".join(kind_html(t) for t in path_types)
            st.markdown(
                f'<div style="padding:0.4rem 0;border-bottom:1px dashed var(--aa-border-soft)">'
                f'<div style="display:flex;gap:0.6rem;align-items:baseline;flex-wrap:wrap">'
                f'<b style="font-size:0.88rem">{esc(consequence)}</b>{kinds}</div>{tags}</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# inspector: lineage, price, market data, measure it
# ---------------------------------------------------------------------------


def _render_measure_it(pipe: ThreadPipeline, g: MarketGroup) -> None:
    plan = pipe.expressions
    specs = [plan.measurement(mid) for mid in g.measurement_ids]
    subhead("Measure it", "What would test the economic hypothesis here -- and what data actually exists. Market "
            "data above describes the market; only the measurements below can become a signal.",
            help="Economic concept → ideal measurement → actual data field → quantitative signal. A measurement "
                 "can feed a signal only when its data is real, point-in-time safe and computable.")
    consequences = " · ".join(g.consequences)
    st.markdown(f'<div class="aa-fact-k">Economic consequence</div>'
                f'<div class="aa-fact-v" style="margin-bottom:0.5rem">{esc(consequences)}</div>',
                unsafe_allow_html=True)
    if not specs:
        st.caption("No measurement template exists for this market yet.")
        return
    rows = [('<div class="aa-measure aa-measure-head"><span>Ideal measurement</span><span>Actual data</span>'
            "<span>Point-in-time</span></div>")]
    for m in specs:
        r = m.resolution
        data_text, data_tone = _DATA_TEXT[r.status]
        source = " · ".join(x for x in (r.dataset, r.field) if x)
        source_html = f'<br><span class="aa-measure-src">{esc(source)}</span>' if source else ""
        pit_text, pit_tone = _PIT_TEXT[r.pit_safe]
        rows.append(
            f'<div class="aa-measure"><span title="{esc(m.variable.economic_meaning)}"><b>{esc(m.label)}</b>'
            f'<span class="aa-measure-src"> · {esc(m.symbol)}</span></span>'
            f"<span>{_state(data_text, data_tone)}{source_html}</span>"
            f"<span>{_state(pit_text, pit_tone)}</span></div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
    ready = [m for m in specs if m.resolution.historically_usable]
    if ready:
        st.markdown(
            '<div class="aa-fact-k" style="margin-top:0.6rem">Ready to become candidate signals</div>'
            '<div class="aa-tag-row">' + "".join(
                f'<span class="aa-tag">{esc(m.label)} · {esc(m.symbol)}</span>' for m in ready) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No measurement here is real, point-in-time safe and computable yet, so this market cannot feed "
                   "a signal. The largest gaps are listed under Advanced details → Gaps.")
    blockers = list(dict.fromkeys(gap.label for m in specs if not m.resolution.historically_usable
                                  for gap in m.resolution.gaps))
    if blockers:
        st.caption("What is missing: " + "; ".join(blockers[:4]) + ".")
