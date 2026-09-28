"""Page 2 -- Dashboard. Secondary analytics page (the Agent conversation is
the default landing page): one root/family's authoritative registry
evidence, cross-market context, the robustness grid, recent experiments, and
the AI agent rail -- everything read verbatim from `alpha_agent.ui.services`,
nothing recomputed here.

Release UX (agent-first navigation): the registry-wide inventory
visualizations that used to live here moved to where they fit the page's own
narrative better -- Verdict Distribution and Failure Reason Distribution to
Validation, Strategy x Market Matrix to Strategies (`services.py`'s
aggregation functions are unchanged, only which page calls them).
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.ui import (
    charts,
    components,
    layout,
    market_context,
    palette,
    panels,
    recommendation_views,
    services,
)

PAGE_TITLE = "Dashboard"


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active=None)
    layout.render_header(subtitle="Secondary analytics -- one market/family's authoritative registry evidence.")
    root = services.approved_universe()[0]
    market_context.render_quote_strip(root)

    _render_executive_summary()
    _render_validated_and_promising()

    left, right = layout.main_rail_columns(gap="medium")

    with left:
        _render_market_header(root)
        detail, _canonical_row = _selected_experiment(root)
        result = detail["result"] if detail else None

        _render_metric_row(result, detail)
        _render_experiment_series_or_registry_view(detail, st.session_state.get("ov-family"))

        components.section_header(
            "Robustness & Statistical Validation",
            "Explicit committed evidence only -- a gate never shows PASS from an absent reason code.",
        )
        components.render_gate_grid(
            result=result,
            trial_role=(detail["experiment"]["trial_role"] if detail else "CANONICAL"),
            verdict=(result["headline_verdict"] if result else None),
            key_prefix="ov",
        )

        components.section_header("Recent Experiments")
        _render_recent_experiments(root)

    with right:
        panels.render_agent_panel(key_prefix="ov")
        panels.render_activity_feed()
        panels.render_system_status()

    layout.render_disclaimer()


def _render_executive_summary() -> None:
    """Release UX (investor-readability pass): an executive research overview
    -- "what is the current state of the research platform?" -- ABOVE the
    per-root/family detail below. Every number here is a pure tally over
    already-exposed `services.py` reads (`registry_summary`,
    `canonical_verdict_distribution`, `paper_eligible_experiments`,
    `list_experiments`) -- no new statistic, verdict, or threshold is computed
    on this page."""
    components.section_header(
        "Research Overview",
        "The current state of the research platform -- coverage, validated strategies, and "
        "what is eligible for paper trading, before any one market/strategy detail below.",
    )
    universe = services.approved_universe()
    reg = services.registry_summary()
    verdicts = services.canonical_verdict_distribution()
    eligible = services.paper_eligible_experiments()

    with components.metric_row("ov-exec-summary"):
        cols = st.columns(4)
        with cols[0]:
            components.metric_card(
                "ov-exec-coverage", "Research Coverage",
                f"{len(universe)} markets",
                sub=f"{reg['authoritative_statistical_hypotheses']} authoritative hypotheses",
            )
        with cols[1]:
            components.metric_card(
                "ov-exec-pass", "Validated -- PASS",
                str(verdicts.get("PASS", 0)),
                sub=f"REJECT {verdicts.get('REJECT', 0)} · INCONCLUSIVE {verdicts.get('INCONCLUSIVE', 0)}",
            )
        with cols[2]:
            components.metric_card(
                "ov-exec-paper", "Paper-Trading Eligible",
                str(len(eligible)),
                sub="requires a committed PASS with complete evidence",
                accent=palette.GREEN if eligible else None,
            )
        with cols[3]:
            components.metric_card(
                "ov-exec-nextaction", "Next Action",
                "Review Agent" if verdicts.get("PASS", 0) == 0 else "Review eligible strategies",
                sub="propose a new hypothesis" if verdicts.get("PASS", 0) == 0 else "see Paper Trading",
            )

    components.section_header("Recent Research")
    _render_recent_research_list()


def _render_validated_and_promising() -> None:
    """Task spec section 19 ("Add: Validated Strategies, Promising
    Candidates and a compact top-candidate section"): the two lists stay
    architecturally separate -- only a real PASS ever appears in the first
    (task spec section 18)."""
    profile = panels.current_investor_profile()
    validated = services.validated_strategy_summaries(profile)
    promising = services.promising_candidate_summaries(profile, sort_by="promise")

    components.section_header(
        "Validated Strategies",
        "Scientific Verdict, Research Promise, and User Fit are independent -- see Validation for the "
        "full gate-by-gate evidence behind any verdict.",
    )
    recommendation_views.render_validated_section(validated, key_prefix="ov")

    components.section_header("Most Promising Research Candidates")
    recommendation_views.render_promising_section(promising, key_prefix="ov", limit=5)


def _render_recent_research_list(*, limit: int = 6) -> None:
    rows = services.list_experiments(trial_role=None)
    rows = [r for r in rows if r.get("verdict")]
    rows = sorted(rows, key=lambda r: r["experiment_id"], reverse=True)[:limit]
    if not rows:
        components.empty_state(
            "Recent Research", "No adjudicated research result is committed yet.", key="ov-recent-na",
        )
        return
    with components.card("ov-recent"):
        for r in rows:
            components.status_row(
                f"{r['root_symbol']} · {services.strategy_name(r['strategy_family'])}",
                r["verdict"],
                (r.get("reason_codes") or ["--"])[0],
            )


def _render_market_header(root: str) -> None:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            f'<div class="aa-ticker-row"><span class="aa-ticker">{root}</span>'
            f'<span class="aa-ticker-name">{services.market_name(root)} Futures</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("CME Globex (GLBX.MDP3) · 1-minute bars · continuous, back-adjusted for research only")
    with c2:
        families = sorted({f["family_key"] for f in services.family_catalog()})
        st.selectbox("Strategy", families, format_func=services.strategy_name, key="ov-family")


def _selected_experiment(root: str) -> tuple[dict | None, dict | None]:
    family = st.session_state.get("ov-family")
    rows = services.list_experiments(strategy_family=family, root_symbol=root, trial_role=None)
    canonical = next((r for r in rows if r["trial_role"] == "CANONICAL"), None)
    if canonical is None:
        return None, None
    detail = services.get_experiment(canonical["experiment_id"])
    violation = services.check_no_holdout_leak(detail, path="$.overview_detail")
    layout.holdout_leak_banner(violation)
    return detail, canonical


def _render_metric_row(result: dict | None, detail: dict | None) -> None:
    max_dd = services.max_drawdown_usd(
        services.trade_ledger_equity_series(services.find_experiment_bound_trade_ledger(detail) if detail else None)
    )
    with components.metric_row("ov-metrics"):
        cols = st.columns(7)
        with cols[0]:
            components.metric_card(
                "net-pnl", "Net PnL",
                f"${result['net_pnl_usd']:,.0f}" if result and result.get("net_pnl_usd") is not None else "N/A",
            )
        with cols[1]:
            components.metric_card(
                "sharpe", "Annual Sharpe",
                f"{result['annualized_sharpe']:.2f}" if result and result.get("annualized_sharpe") is not None else "N/A",
            )
        with cols[2]:
            if max_dd is not None:
                components.metric_card("drawdown", "Max Drawdown", f"${max_dd:,.0f}", accent=palette.RED)
            else:
                components.unavailable_metric_card(
                    "drawdown", "Max Drawdown",
                    "No committed, experiment-bound trade ledger is bound to this experiment.",
                )
        with cols[3]:
            components.metric_card(
                "trades", "Total Trades",
                str(result["n_trades"]) if result and result.get("n_trades") is not None else "N/A",
            )
        with cols[4]:
            components.unavailable_metric_card(
                "winrate", "Win Rate", "Requires an experiment-bound trade ledger (none committed).",
            )
        with cols[5]:
            components.metric_card(
                "costs", "Costs",
                f"${result['costs_usd']:,.0f}" if result and result.get("costs_usd") is not None else "N/A",
            )
        with cols[6]:
            components.verdict_metric_card("verdict", "Verdict", result["headline_verdict"] if result else None)


def _render_experiment_series_or_registry_view(detail: dict | None, selected_family: str | None) -> None:
    """Section 5/6 of the Release UI polish task: never reserve giant empty
    chart panels for a per-experiment series that is not committed. When one
    IS committed (a verified, experiment-bound trade ledger), plot it for
    real; otherwise show one compact callout and fill the rest of the page
    with REAL, provenance-backed registry-wide evidence instead."""
    ledger = services.find_experiment_bound_trade_ledger(detail) if detail else None
    series = services.trade_ledger_equity_series(ledger)

    if series:
        c1, c2 = st.columns(2)
        with c1, components.card("ov-equity"):
            st.markdown('<div class="aa-gate-title">Equity Curve</div>', unsafe_allow_html=True)
            components.plotly_chart(charts.equity_curve_chart([r["cum_net_pnl_usd"] for r in series]))
        with c2, components.card("ov-drawdown"):
            st.markdown('<div class="aa-gate-title">Drawdown</div>', unsafe_allow_html=True)
            components.plotly_chart(charts.drawdown_chart([r["drawdown_usd"] for r in series]))
        st.caption(f"From the bound ledger `{ledger['source_artifact']}` -- {ledger['n_trades']} real trades.")
    else:
        components.committed_series_notice(key="ov-series-na")

    components.section_header(
        "Portfolio Evidence",
        "Registry-wide, provenance-backed -- every figure below is a direct tally over the committed "
        "ExperimentRegistry, not specific to the single experiment selected above.",
    )
    _render_registry_visuals(selected_family)


def _render_registry_visuals(selected_family: str | None) -> None:
    if selected_family:
        _render_cross_market_strip(selected_family)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        scatter_points = services.canonical_sharpe_vs_fdr()
        with components.card("ov-scatter"):
            st.markdown('<div class="aa-gate-title">Sharpe vs. BH-FDR q-value</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="aa-gate-note">Every canonical trial registry-wide.</div>',
                unsafe_allow_html=True,
            )
            if scatter_points:
                components.plotly_chart(charts.sharpe_fdr_scatter(scatter_points))
            else:
                st.caption("No canonical trial carries both a committed Sharpe and BH-q value yet.")
    with r1c2, components.card("ov-gate-rates"):
        st.markdown('<div class="aa-gate-title">Validation Gate Outcomes Across the Registry</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="aa-gate-note">Per-gate state tally over every canonical trial -- the same '
            'explicit-only resolution shown per-experiment below.</div>',
            unsafe_allow_html=True,
        )
        components.plotly_chart(charts.gate_stack_bar(services.gate_failure_rates()))


def _render_cross_market_strip(family: str) -> None:
    """Section 5's "selected-family cross-market evidence": how the currently
    selected strategy family's canonical trial resolved across every root,
    not just the one selected above -- the full inventory (every family x
    every root) now lives on the Strategies page instead."""
    cells = [c for c in services.strategy_market_matrix() if c["strategy_family"] == family]
    if not cells:
        return
    with components.card("ov-xmarket"):
        st.markdown(
            f'<div class="aa-gate-title">Cross-Market Evidence &mdash; {services.strategy_name(family)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="aa-gate-note">The canonical trial for this family, by root. Full '
            'family &times; market inventory is on the Strategies page.</div>',
            unsafe_allow_html=True,
        )
        with components.metric_row(f"ov-xmarket-{family}"):
            cols = st.columns(len(cells))
            for col, cell in zip(cols, sorted(cells, key=lambda c: c["root_symbol"]), strict=True):
                with col:
                    st.markdown(
                        f'<div style="text-align:center">'
                        f'<div class="aa-metric-label">{cell["root_symbol"]}</div>'
                        f'{components.badge(cell["verdict"])}</div>',
                        unsafe_allow_html=True,
                    )


def _render_recent_experiments(root: str) -> None:
    rows = services.list_experiments(root_symbol=root, trial_role=None)
    rows = sorted(rows, key=lambda r: r["experiment_id"])[:12]
    st.dataframe(
        [
            {
                "Experiment": r["experiment_id"],
                "Strategy": services.strategy_name(r["strategy_family"]),
                "Role": r["trial_role"],
                "Net PnL": r["net_pnl_usd"],
                "Sharpe": r["daily_sharpe"],
                "BH q": r["bh_q"],
                "DSR": r["dsr_probability"],
                "Verdict": r["verdict"],
                "Key Reason": (r["reason_codes"] or ["--"])[0],
            }
            for r in rows
        ],
        width="stretch",
        hide_index=True,
        height=min(380, 44 + 35 * len(rows)),
    )
