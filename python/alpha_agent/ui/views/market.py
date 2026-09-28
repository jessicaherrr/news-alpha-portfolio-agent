"""Page -- Market. A MARKET OBSERVATION page (Release UX Part D/F, task spec
sections 21-26/46; MARKET REALITY pass, mission section 9; Market
Intelligence + Futures Universe campaign, Checkpoints A-B: "Market home = ALL
FUTURES scanner" + "Product Detail Page"): current futures market data for
display and conversational context, explicitly never a prediction or trading
page.

The landing view is the ALL FUTURES Market Scanner (`alpha_agent.ui.
market_scanner`) -- every CATALOGUED product (mission Part 4), not just the
five research-universe roots. Selecting a scanner row opens a tabbed Product
Detail view below it (mission Part 7: "Market / <product>", implemented as
in-page sub-navigation -- "Do NOT create more sidebar top-level pages"):
Overview / Contracts / Relative / News & Events / Research. Checkpoint B
builds Overview (the SAME shared `alpha_agent.ui.market_home` rendering the
Agent/Home page uses, now reachable for ANY catalogued product Databento can
resolve -- see `alpha_agent.ui.databento_context.OBSERVABLE_ROOTS`) and
Research (mission Part 30: per-root scientific-status summary, gated to the
certified Research Universe). The Market Intelligence Data Completion Pass
builds Contracts (Checkpoint C: real contract ladder + term structure),
Relative (Checkpoint D: cross-market analytics), and both halves of News &
Events -- NEWS (Checkpoint E: real official market-news ingestion) and
Scheduled Events (Checkpoint F: real official economic-event calendars) --
via `alpha_agent.ui.market_news`/`market_events`/`alpha_agent.market_intel`.
Any real data gap still renders an honest "unavailable" state naming why
(mission Part 37) -- never a silently empty section.

Every number renders through `alpha_agent.ui.databento_context` -- the sole
UI boundary into the read-only Databento observation provider -- and every
value that is not available renders as an honest "unavailable" state, never
a fabricated placeholder (a disconnected feed collapses to ONE concise card,
never a table of N/A). A bounded, explicit "Refresh" action is the only way
to force a fresh network call before `market_home`'s own UI-session cache
window elapses; a bare page render/rerun reuses that cache (no
Streamlit-rerun charge storm).

This page is architecturally incapable of contaminating the scientific plane
-- see `tests/python/test_release_market_plane_isolation.py` and
`databento_context`'s own module docstring; nothing rendered here is passed
to Fast Screen, strict validation, Research Promise, or the registry.
"""
from __future__ import annotations

import html

import streamlit as st

from alpha_agent.marketdata.databento_schemas import UNAVAILABLE_CAPABILITIES
from alpha_agent.ui import (
    charts_market,
    components,
    layout,
    market_contracts,
    market_events,
    market_home,
    market_news,
    market_reaction,
    market_relative,
    market_scanner,
    market_universe,
    services,
)

PAGE_TITLE = "Market"

_TABS = ("Overview", "Contracts", "Relative", "News & Events", "Research")

#: Market's OWN local selected-product state (Sidebar IA pass, task spec
#: section 1D). Historically `selected_root` behaved like a hidden GLOBAL
#: application selector (set by the shared sidebar, read by several
#: unrelated pages); that block is gone from the sidebar entirely now (task
#: spec section 1), and this key is namespaced to make the new contract
#: explicit: Market owns it, Agent's evidence scope never reads it (see
#: `claude_conversation.py`'s own module docstring), and Research/Paper each
#: own their own selection independently (task spec section 13). Compatibility
#: is isolated to the Market UI boundary -- `market_scanner.py`/
#: `market_home.py`/`market_context.py` -- rather than a repository-wide
#: rename of an unrelated concept.
_SELECTED_ROOT_KEY = "market_selected_root"


def _get_or_init_selected_root() -> str:
    """The single canonical selected-product control for Market (task spec
    section 1A/1C): validated against the full MARKET UNIVERSE catalog
    (`market_universe.market_universe_roots()`, e.g. from the Scanner), never
    narrowed to the certified RESEARCH UNIVERSE -- see the prior sidebar
    implementation's own docstring for the rerun-loop bug that guard
    previously caused."""
    universe = services.approved_universe()
    valid_roots = market_universe.market_universe_roots()
    if _SELECTED_ROOT_KEY not in st.session_state or st.session_state[_SELECTED_ROOT_KEY] not in valid_roots:
        st.session_state[_SELECTED_ROOT_KEY] = universe[0] if universe else "NQ"
    return st.session_state[_SELECTED_ROOT_KEY]


#: Which of Market's two views is showing: "affected" (the active research
#: thread's affected markets) or "explore" (every catalogued futures market).
VIEW_KEY = "market_view"
_VIEWS = {"affected": "Affected Markets", "explore": "Explore All Markets"}


def open_explorer(root: str | None = None) -> None:
    """The one hand-off into Explore All Markets (optionally on ``root``):
    used by Ask's "Open Market", the Research Thread's "Full market detail",
    and any other in-app link that means the general explorer -- never the
    thread's affected markets."""
    if root:
        st.session_state[_SELECTED_ROOT_KEY] = root
    st.session_state[VIEW_KEY] = "explore"
    st.switch_page(st.Page(render, url_path="market"))


def _go_news() -> None:
    from alpha_agent.ui.views import news

    st.switch_page(st.Page(news.render, url_path="news"))


def _open_thread() -> None:
    from alpha_agent.ui.views import workspace

    st.switch_page(st.Page(workspace.render, url_path="research"))


def render() -> None:
    """Market -- ONE destination, two views. With an active Research Thread
    it opens on Affected Markets (only what that thread's event reaches --
    the SAME component the thread's Market step renders); Explore All Markets
    is the full futures explorer (scanner, product detail, contracts, term
    structure, relative moves, news & events), unchanged. Without a thread it
    opens on the explorer."""
    from alpha_agent.ui import news_alpha_context, research_thread

    layout.inject_style()
    layout.render_sidebar_nav(active="market")
    layout.render_header(subtitle="Where your research shows up in markets -- and every futures market beside it.")
    st.markdown('<div class="aa-page-title">Market</div>', unsafe_allow_html=True)

    thread = research_thread.active_thread()
    if thread is None:
        with st.container(horizontal=True, vertical_alignment="center", key="market-no-thread"):
            st.markdown(
                '<div class="aa-context-strip">No active research thread. Explore markets below, or start research '
                "from News.</div>",
                unsafe_allow_html=True, width="content",
            )
            if st.button("Start from News", key="market-start-news", icon=":material/newspaper:", type="tertiary",
                         help="Pick an event to research; its affected markets then appear here."):
                _go_news()
        _render_explorer()
        layout.render_disclaimer()
        return

    with st.container(horizontal=True, vertical_alignment="center", key="market-thread-context"):
        st.markdown(
            f'<div class="aa-context-strip"><span class="aa-eyebrow">Researching</span>'
            f"<b>{html.escape(thread.event.headline)}</b></div>",
            unsafe_allow_html=True, width="content",
        )
        if st.button("Open research thread", key="market-open-thread", icon=":material/arrow_forward:",
                     icon_position="right", type="tertiary",
                     help="Back to the thread -- reasoning, signals, portfolio and backtest for this event."):
            _open_thread()
    if st.session_state.get(VIEW_KEY) not in _VIEWS:
        st.session_state[VIEW_KEY] = "affected"
    view = st.segmented_control(
        "View", list(_VIEWS), format_func=_VIEWS.__getitem__, key=VIEW_KEY, label_visibility="collapsed",
        help="Affected Markets: only the markets your research thread's event reaches, with why and how to measure "
             "the effect. Explore All Markets: every catalogued futures market, for general observation.",
    ) or "affected"
    if view == "explore":
        _render_explorer()
    else:
        from alpha_agent.ui.workspace import step_market

        mandate = news_alpha_context.current_mandate()
        pipe = research_thread.pipeline_for(thread, mandate)
        thread, notice = research_thread.reconcile_setup(thread, mandate, pipe.expressions)
        pipe.thread = thread
        if notice:
            st.warning(notice, icon=":material/sync_problem:")
        step_market.render_affected_markets(pipe)
    layout.render_disclaimer()


def _render_explorer() -> None:
    """Explore All Markets -- the full futures explorer, exactly as the
    standalone Markets page rendered it."""
    components.section_header(
        "Explore all markets",
        "The full observable futures universe -- browse and filter every catalogued product below, "
        "then select one for deeper observation. Observation only: nothing here enters the scientific "
        "research or validation plane.",
    )

    market_scanner.render_scanner(key_prefix="market")
    root = _get_or_init_selected_root()

    if root not in market_universe.market_universe_roots():
        # Structurally unreachable through the Scanner (it only ever offers
        # catalogued roots) -- kept as an honest guard rather than assumed.
        components.empty_state(
            "Unknown Product", f"{root!r} is not in the futures product catalog.", key="market-uncatalogued-na",
        )
        return

    _render_product_header_and_selector(root)
    _render_product_detail(root)


# ---------------------------------------------------------------------------
# Product Detail -- canonical product-selection surface (task spec section
# 1A/1B) + tabbed sub-navigation (mission Part 7)
# ---------------------------------------------------------------------------


def _render_product_header_and_selector(root: str) -> None:
    """Turns the Product Detail area's own header into the canonical
    product-selection surface (task spec section 1A): "CL - Crude Oil"
    alongside a compact "Change Market" popover -- integrated into the
    existing header rather than a large new one. Selecting a different
    product here updates the SAME `market_selected_root` state the Scanner
    drives (task spec section 1B/1C: ONE canonical selected product for
    Market), so Overview/Contracts/Relative/News & Events/Research below all
    re-render for the newly selected root on the very next rerun."""
    entry_name = services.market_name(root)
    universe = market_universe.market_universe()
    c1, c2 = st.columns([4, 1.3])
    with c1:
        components.section_header(f"{root} · {entry_name}")
    with c2, st.popover("Change Market ▾", width="stretch"):
        search = st.text_input(
            "Search products...", key="market-selector-search", label_visibility="collapsed",
            placeholder="Search products...",
        )
        lowered = search.strip().lower()
        filtered = [
            e for e in universe
            if not lowered or lowered in e.root_symbol.lower() or lowered in e.display_name.lower()
        ]
        for entry in filtered[:60]:
            is_selected = entry.root_symbol == root
            label = f"{entry.root_symbol}  {entry.display_name}"
            if st.button(
                label, key=f"market-selector-{entry.root_symbol}", width="stretch",
                type=("primary" if is_selected else "secondary"),
            ):
                st.session_state[_SELECTED_ROOT_KEY] = entry.root_symbol
                st.rerun()
        if not filtered:
            st.caption("No catalogued product matches this search.")


def _render_product_detail(root: str) -> None:
    tab_overview, tab_contracts, tab_relative, tab_news, tab_research = st.tabs(list(_TABS))
    with tab_overview:
        _render_overview_tab(root)
    with tab_contracts:
        market_contracts.render_contracts_tab(root)
    with tab_relative:
        market_relative.render_relative_tab(root)
    with tab_news:
        _render_news_and_events_tab(root)
    with tab_research:
        _render_research_tab(root)


def _render_news_and_events_tab(root: str) -> None:
    market_news.render_news_section(root, key_prefix="market-news")
    market_events.render_events_section(root, key_prefix="market-events")


def _render_overview_tab(root: str) -> None:
    health, _ = market_home.get_health_cached()
    if health.capability in UNAVAILABLE_CAPABILITIES:
        market_home.render_unavailable_state(health, key_prefix="market")
        return

    market_home.render_selected_market_hero(root, key_prefix="market")
    market_home.render_chart_and_metrics(root, key_prefix="market")
    market_home.render_market_context(root, key_prefix="market")
    market_reaction.render_chart_with_markers(root, key_prefix="market-overlay")
    market_reaction.render_observed_reaction(root, key_prefix="market-reaction")
    _render_comparison()


def _render_research_tab(root: str) -> None:
    with components.card("market-research-tab"):
        st.markdown('<div class="aa-gate-title">RESEARCH</div>', unsafe_allow_html=True)
        research_universe = market_universe.research_universe()
        if root not in research_universe:
            components.empty_state(
                "Not a Certified Research Root",
                f"{root} is not in the certified Research Universe ({', '.join(research_universe)}). "
                "Contract economics, roll logic, and C++ execution have not been onboarded for this "
                "product (CLAUDE.md Part 38) -- no scientific research exists here, and none of this "
                "platform's ES-style assumptions are silently reused for it.",
                key="market-research-tab-uncertified",
            )
            return

        candidates = [c for c in services.research_candidate_summaries() if c.root_symbol == root]
        if not candidates:
            components.empty_state(
                "No Research Yet", f"No committed research candidate exists for {root} yet.",
                key="market-research-tab-empty",
            )
        else:
            from alpha_agent.ui.recommendation_views import format_user_fit

            families = sorted({c.strategy_family for c in candidates})
            st.caption(f"Tested strategy families: {', '.join(services.strategy_name(f) for f in families)}")
            best = max(candidates, key=lambda c: (c.scientific_verdict == "PASS", c.research_promise_score))
            # Section 30's compact per-product research summary -- every
            # field read straight from the EXISTING registry-backed
            # ResearchCandidateSummary, never a second data source.
            components.provenance_row("Research Enabled", "Yes")
            components.provenance_row("Tested Strategy Families", str(len(families)))
            components.provenance_row("Number of Canonical Experiments", str(len(candidates)))
            components.provenance_row("Best Candidate", services.strategy_name(best.strategy_family))
            components.provenance_row("Scientific Verdict", best.scientific_verdict)
            components.provenance_row("Research Promise", best.research_promise_label)
            components.provenance_row("User Fit", format_user_fit(best))
            components.provenance_row(
                "Best Historical Sharpe",
                f"{best.annualized_sharpe:.2f}" if best.annualized_sharpe is not None else "N/A (not committed)",
            )
            components.provenance_row(
                "Trade Count", str(best.trade_count) if best.trade_count is not None else "N/A (not committed)",
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("View Research", key="market-research-tab-view", width="stretch"):
                best = max(candidates, key=lambda c: (c.scientific_verdict == "PASS", c.research_promise_score)) \
                    if candidates else None
                st.session_state["research_details_target"] = {
                    "source": "market", "objective": None, "root": root,
                    "family": best.strategy_family if best else None,
                    "hypothesis": None, "compiled": None, "evidence": None, "run_outcome": None,
                    "experiment_id": best.experiment_id if best else None,
                }
                from alpha_agent.ui.views import research

                st.switch_page(st.Page(research.render, url_path="lab"))
        with c2:
            if st.button("Discover Strategies", key="market-research-tab-discover", width="stretch"):
                st.session_state["discover_target_root"] = root
                from alpha_agent.ui.views import discover

                st.switch_page(st.Page(discover.render, url_path="discover"))


# ---------------------------------------------------------------------------
# market comparison (mission section 10/25)
# ---------------------------------------------------------------------------


def _render_comparison() -> None:
    universe = list(services.approved_universe())
    with components.card("market-compare"):
        st.markdown('<div class="aa-gate-title">COMPARE MARKETS</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        root_a = cols[0].selectbox("Market A", universe, index=0, key="market-cmp-a")
        root_b = cols[1].selectbox("Market B", universe, index=min(1, len(universe) - 1), key="market-cmp-b")
        if root_a == root_b:
            st.caption("Choose two different markets to compare.")
            return
        series = {}
        for cmp_root in (root_a, root_b):
            result, _ = market_home.get_ohlcv_cached(
                cmp_root, timeframe="1h", lookback_bars=market_home.SNAPSHOT_LOOKBACK_BARS,
            )
            series[cmp_root] = [bar.model_dump(mode="json") for bar in result.bars] if (result and result.fetched) else []
        if not any(series.values()):
            components.empty_state("Compare Markets", "No real bars available for either market.", key="market-cmp-na")
            return
        components.plotly_chart(charts_market.market_comparison_chart(series), key="market-compare-fig")
        for cmp_root, bars in series.items():
            if len(bars) >= 2 and bars[0]["close"]:
                pct = (bars[-1]["close"] / bars[0]["close"] - 1.0) * 100.0
                rv = market_home.realized_volatility_pct(bars)
                rv_text = f", realized vol {rv:.1f}% (annualized)" if rv is not None else ""
                st.caption(f"{cmp_root}: {pct:+.2f}% over the shown window{rv_text}.")

        correlation = None
        if series.get(root_a) and series.get(root_b):
            correlation = _pearson_correlation(
                [row["close"] for row in series[root_a]], [row["close"] for row in series[root_b]],
            )
        if correlation is not None:
            st.caption(
                f"Correlation ({root_a} vs {root_b}, aligned by bar index over the shorter series): "
                f"{correlation:.2f}. No causal interpretation is implied."
            )
        else:
            st.caption("Correlation: insufficient overlapping bars. No causal interpretation is implied either way.")


def _pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """A purely descriptive correlation of two real close-price series over
    their common (shorter) window -- never a causal or predictive claim
    (mission section 10: "No causal interpretation")."""
    n = min(len(x), len(y))
    if n < 3:
        return None
    x, y = x[-n:], y[-n:]
    mean_x, mean_y = sum(x) / n, sum(y) / n
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    var_x = sum((v - mean_x) ** 2 for v in x)
    var_y = sum((v - mean_y) ** 2 for v in y)
    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return None
    return cov / denom
