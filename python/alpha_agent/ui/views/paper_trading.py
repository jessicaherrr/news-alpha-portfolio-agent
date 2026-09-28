"""Page 7 -- Paper Trading. Phase 21
(`prompts/21_PAPER_TRADING_AND_DRIFT_MONITOR.md`) is implemented:
target-position -> order -> simulated fill -> portfolio flow through the
hard-risk-gated C++ engine (`quant_paper_trading_targets_csv` /
`quant::PortfolioRiskManager`), a persistent paper ledger, drift diagnostics,
and alerts. Every READ on this page calls only a read method on
`alpha_agent.paper.ledger.PaperLedger` or the deterministic
`alpha_agent.paper.eligibility` service (`alpha_agent.ui.services`'s existing
boundary discipline, statically grep-tested to never call a ledger write
method). The one WRITE this page can trigger -- "Start Paper Test" (Product UI
Polish pass, section 11) -- goes through the SEPARATE, explicit
`alpha_agent.ui.paper_actions` boundary, which calls the exact same
`PaperTradingEngine.start_run` `scripts/phase_21_paper_trading.py start`
already uses; nothing here reimplements eligibility, execution, or ledger
writes in Streamlit. Stepping or stopping an existing run stays CLI-only.

Every number on this page is either read verbatim from the committed paper
ledger, or is the deterministic output of `assert_paper_trading_eligible`.
Nothing here fabricates an equity curve, a position, an order, a fill, or a
risk/drift state: if no paper run has been started AND no strategy is
currently eligible, the page shows a clean, honest empty state (section 10)
instead of an implementation-heavy dump.

This is deterministic REPLAY-based paper trading over the already-acquired
2018-2024 research + validation data (see `scripts/phase_21_paper_trading.py`
module docstring) -- there is still no live or paper BROKER connection and no
live-money routing in this system (CLAUDE.md risk rule 2/3).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from alpha_agent.paper.ledger import is_legacy_evidence
from alpha_agent.ui import charts_discovery, components, layout, palette, paper_actions, services

PAGE_TITLE = "Paper Trading"

_STATUS_BADGE = {"ACTIVE": "OK", "HALTED": "WARN", "STOPPED": "OFFLINE"}


def _fmt_usd(v: float | None) -> str:
    return "n/a" if v is None else f"${v:,.0f}"


def _eligible_option_label(e: dict) -> str:
    return f"{e['root_symbol']} · {services.strategy_name(e['strategy_family'])} ({e['experiment_id']})"


def _render_empty_state(*, eligible_count: int, active_runs: int) -> None:
    """Section 10: a clean product state when nothing is currently testable
    -- no run warning wall, no raw risk-manager implementation detail on the
    landing surface. `eligible_count`/`active_runs` are always real, already-
    computed values (never a hardcoded zero) even though this branch only
    renders when both are honestly zero."""
    with components.card("pt-empty"):
        st.markdown(
            '<div class="aa-empty">'
            '<div class="aa-empty-title">No active paper test.</div>'
            '<div class="aa-empty-reason">No strategy currently satisfies the scientific eligibility policy '
            "(a frozen validation PASS, all required gates satisfied, and a StrategySpec this phase can "
            "deterministically rebuild and fingerprint-verify). See the Experiment Log for what has been "
            "tried.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        with components.metric_row("pt-empty"):
            c1, c2, c3 = st.columns(3)
            with c1:
                components.metric_card("pt-empty-active", "Active Runs", str(active_runs))
            with c2:
                components.metric_card("pt-empty-eligible", "Eligible", str(eligible_count))
            with c3:
                components.metric_card("pt-empty-exposure", "Current Exposure", _fmt_usd(0.0))
        st.write("")
        if st.button("Open Research", key="pt-empty-open-research", width="stretch", type="primary"):
            from alpha_agent.ui.views import research

            st.switch_page(st.Page(research.render, url_path="lab"))


def _render_eligible_and_start(eligible: list[dict]) -> None:
    """Section 11: "Choose eligible strategy -> [ Start Paper Test ]" -- the
    ONLY paper-trading write path this UI exposes, and only ever enabled when
    `services.paper_eligible_experiments()` (the same authoritative,
    registry-derived scan `alpha_agent.paper.eligibility` computes) actually
    lists at least one strategy. Starting calls `alpha_agent.ui.paper_actions
    .start_paper_run`, which re-verifies eligibility itself -- this page never
    decides eligibility on its own."""
    components.section_header(
        "Strategies eligible for paper trading",
        "Every AUTHORITATIVE registry experiment with a committed PASS verdict "
        "(all_required_gates_satisfied) whose StrategySpec this phase can "
        "deterministically rebuild and fingerprint-verify.",
    )
    if not eligible:
        st.caption("No experiment is currently eligible for a new paper test -- this is the honest registry state.")
        return

    options = {_eligible_option_label(e): e["experiment_id"] for e in eligible}
    labels = list(options.keys())
    # Research's "Test in Paper" handoff (section 12) preselects the
    # experiment it was clicked from, one time, rather than leaving the user
    # to find it again in the list.
    preselect_id = st.session_state.pop("pt-preselect-experiment-id", None)
    default_index = 0
    if preselect_id is not None:
        for i, (label, experiment_id) in enumerate(options.items()):
            if experiment_id == preselect_id:
                default_index = i
                break
    with components.card("pt-eligible"):
        choice_label = st.selectbox(
            "Choose eligible strategy", labels, index=default_index, key="pt-start-select",
        )
        if st.button("Start Paper Test", key="pt-start-button", type="primary"):
            result = paper_actions.start_paper_run(options[choice_label])
            if result.ok:
                st.session_state["pt-run-select-pending"] = result.run_id
                st.rerun()
            else:
                st.error(f"REFUSED: {result.error}", icon="🛑")

    with st.expander("All eligible strategies (technical detail)", expanded=False):
        df = pd.DataFrame(
            [
                {
                    "Experiment": e["experiment_id"],
                    "Market": e["root_symbol"],
                    "Strategy": services.strategy_name(e["strategy_family"]),
                    "Backtest Daily Sharpe": e["backtest_daily_sharpe"],
                    "Backtest Net PnL ($)": e["backtest_net_pnl_usd"],
                    "Backtest Trades": e["backtest_n_trades"],
                }
                for e in eligible
            ]
        )
        st.dataframe(df, hide_index=True, width="stretch")


def _render_overview_charts(run_id: str, report: dict) -> None:
    """Section 13: a compact visual overview -- paper equity / cumulative PnL
    curve and drawdown -- built ONLY from the authoritative committed
    `PaperStepRow.equity_usd` history already in `report["steps"]` (the SAME
    per-step C++-computed values the Execution tab's own Steps table
    renders). Reuses the EXISTING `charts_discovery.daily_equity_curve_chart`
    / `.daily_drawdown_chart` (built for the Alpha Discovery campaign's daily
    equity trace) rather than a new chart function -- a paper run IS a
    per-trading-day step series, the same shape those functions already
    expect. No metric here is invented: fewer than 2 steps has no drawdown
    baseline to plot, so this renders the existing `committed_series_notice`
    instead of a one-point chart."""
    steps = report.get("steps") or []
    if len(steps) < 2:
        components.committed_series_notice(
            key=f"pt-overview-chart-na-{run_id}",
            text="Fewer than 2 committed steps -- no equity/drawdown trace to plot yet.",
        )
        return
    daily_rows = [
        {
            "equity_usd": s["equity_usd"],
            "trading_day": pd.to_datetime(s["as_of_ts_ns"], unit="ns", utc=True).strftime("%Y-%m-%d"),
        }
        for s in steps
    ]
    with components.card(f"pt-overview-{run_id}"):
        st.markdown('<div class="aa-gate-title">Paper Equity</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            components.plotly_chart(
                charts_discovery.daily_equity_curve_chart(daily_rows), key=f"pt-equity-chart-{run_id}",
            )
        with c2:
            components.plotly_chart(
                charts_discovery.daily_drawdown_chart(daily_rows), key=f"pt-drawdown-chart-{run_id}",
            )


def _render_run_detail(run_id: str) -> None:
    report = services.paper_run_report(run_id)
    if report is None:
        st.error(f"No paper run {run_id!r} in the ledger.", icon="⚠️")
        return
    run = report["run"]
    latest = report["latest_step"]

    with components.metric_row(f"pt-run-{run_id}"):
        cols = st.columns(4)
        with cols[0], components.card(f"pt-run-status-{run_id}"):
            st.markdown('<div class="aa-metric-label">Status</div>', unsafe_allow_html=True)
            st.markdown(
                components.badge(_STATUS_BADGE.get(run["status"], "NOT_AVAILABLE"), label=run["status"]),
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="aa-metric-sub" style="margin-top:0.5rem">{run["root_symbol"]} '
                f'/ {services.strategy_name(run["strategy_family"])}</div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            components.metric_card(
                f"pt-equity-{run_id}", "Equity",
                _fmt_usd(latest["equity_usd"]) if latest else "n/a",
                sub=f"peak {_fmt_usd(latest['peak_equity_usd'])}" if latest else None,
            )
        with cols[2]:
            components.metric_card(
                f"pt-drawdown-{run_id}", "Drawdown",
                f"{latest['drawdown_pct']:.1%}" if latest else "n/a",
                sub=_fmt_usd(latest["drawdown_usd"]) if latest else None,
            )
        with cols[3]:
            components.metric_card(
                f"pt-fills-{run_id}", "Fills / Trades Closed",
                f"{latest['fills_generated']} / {latest['trades_closed']}" if latest else "0 / 0",
                sub=f"{report['n_steps']} step(s), {report['n_alerts']} alert(s)",
            )

    if latest is not None and latest["kill_switch_active"]:
        st.error(
            "Hard-risk kill switch ACTIVE for this run: the deterministic "
            "PortfolioRiskManager is refusing every risk-increasing order. "
            "Existing positions are never force-closed by a kill switch.",
            icon="🛑",
        )
    if latest is not None and not latest["margin_complete"]:
        st.info(
            "No committed per-root margin schedule is wired to this run; margin "
            "utilization is not enforced (see PaperRiskPolicy docstring).",
            icon="ℹ️",
        )

    _render_overview_charts(run_id, report)

    # Product Consolidation + Opportunity V1 campaign, Checkpoint E: FOUR
    # internal views (Positions / Execution / Risk / Monitoring) instead of
    # six flatter tabs -- nothing below was rewritten, only regrouped: Steps
    # + Fills now share one Execution tab, Alerts moved under Risk (they are
    # risk-state-transition records, not execution mechanics), and Drift
    # Monitor + Provenance now share one Monitoring tab.
    #
    # Sidebar IA pass (task spec section 7D): each tab's body is its own
    # function so the shared sidebar's Paper -> {Positions,Execution,Risk,
    # Monitoring} sub-rows can deep-link straight to ONE of them via
    # `st.session_state["paper_landing_focus"]` -- `st.tabs` has no supported
    # way to be preselected programmatically, so a focused view renders that
    # section directly (with a "back to all views" control) instead of the
    # tab strip; the unfocused default path (tabs, byte-for-byte unchanged
    # content) is exactly what every existing test already exercises.
    focus = st.session_state.get("paper_landing_focus")
    sections: dict[str, tuple[str, object, tuple]] = {
        "positions": ("Positions", _render_positions_tab, (report, latest)),
        "execution": ("Execution", _render_execution_tab, (report,)),
        "risk": ("Risk", _render_risk_tab, (run_id, latest, report)),
        "monitoring": ("Monitoring", _render_monitoring_tab, (latest,)),
    }
    if focus in sections:
        label, renderer, args = sections[focus]
        if st.button("< Back to all views", key=f"pt-focus-back-{run_id}"):
            st.session_state.pop("paper_landing_focus", None)
            st.rerun()
        st.markdown(f'<div class="aa-gate-title">{label}</div>', unsafe_allow_html=True)
        renderer(*args)
        return

    tab_positions, tab_execution, tab_risk, tab_monitoring = st.tabs(
        ["Positions", "Execution", "Risk", "Monitoring"]
    )
    with tab_positions:
        _render_positions_tab(report, latest)
    with tab_execution:
        _render_execution_tab(report)
    with tab_risk:
        _render_risk_tab(run_id, latest, report)
    with tab_monitoring:
        _render_monitoring_tab(latest)


def _render_positions_tab(report: dict, latest: dict | None) -> None:
    # Phase 21.1: ONLY the committed C++ position snapshot
    # (BacktestResult::portfolio_at_end.positions) may be shown here --
    # never a position reconstructed in Python from the fill history.
    # Phase 21.1b: an empty list here is ambiguous between "flat" and "this
    # legacy step predates position-snapshot capture" -- disambiguate
    # explicitly rather than silently rendering "no open position".
    positions = report.get("latest_positions") or []
    latest_is_legacy = latest is not None and is_legacy_evidence(latest.get("provenance") or {})
    if latest_is_legacy and not positions:
        st.info(
            "This step was recorded under the Phase 21 (pre-21.1) ledger schema, "
            "before the authoritative position snapshot was captured -- "
            "'no position' is NOT a claim this run was flat, it is simply not "
            "available for this legacy step.",
            icon="ℹ️",
        )
    elif positions:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Instrument": p["raw_symbol"],
                        "Root": p["root_symbol"],
                        "Units": p["units"],
                        "Avg Entry": p["avg_entry_price"],
                        "Mark": p["mark_price"],
                        "Unrealized PnL ($)": p["unrealized_pnl_usd"],
                        "Gross Notional ($)": p["gross_notional_usd"],
                        "Mark Stale": p["mark_is_stale"],
                        "Margin Known": p["margin_known"],
                    }
                    for p in positions
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No open position in the latest committed C++ snapshot.")


def _render_execution_tab(report: dict) -> None:
    components.section_header("Steps", "Run state over time -- one row per replayed step.")
    if report["steps"]:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Step": s["step_ordinal"],
                        "As-of": pd.to_datetime(s["as_of_ts_ns"], unit="ns", utc=True),
                        "Equity ($)": s["equity_usd"],
                        "Net PnL ($)": s["net_pnl_usd"],
                        "Drawdown (%)": s["drawdown_pct"] * 100,
                        "Fills": s["fills_generated"],
                        "Open Positions": s["open_positions"],
                        "Kill Switch": s["kill_switch_active"],
                    }
                    for s in report["steps"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No step has been recorded yet.")
    components.section_header("Fills", "Real simulated fills from the C++ engine -- costs and execution state.")
    if report["fills"]:
        st.dataframe(pd.DataFrame(report["fills"]), hide_index=True, width="stretch")
    else:
        st.caption("No fill has been recorded yet.")


def _render_risk_tab(run_id: str, latest: dict | None, report: dict) -> None:
    components.section_header(
        "Risk", "Authoritative portfolio risk state -- C++ `PortfolioRiskManager` remains the source of truth.",
    )
    with components.metric_row(f"pt-risk-{run_id}"):
        r1, r2, r3 = st.columns(3)
        with r1:
            components.metric_card(
                f"pt-risk-killswitch-{run_id}", "Kill Switch",
                "ACTIVE" if (latest and latest["kill_switch_active"]) else "Inactive",
                accent=palette.RED if (latest and latest["kill_switch_active"]) else None,
            )
        with r2:
            components.metric_card(
                f"pt-risk-drawdown-{run_id}", "Drawdown",
                f"{latest['drawdown_pct']:.1%}" if latest else "n/a",
                sub=_fmt_usd(latest["drawdown_usd"]) if latest else None,
            )
        with r3:
            components.metric_card(
                f"pt-risk-margin-{run_id}", "Margin Schedule",
                "Complete" if (latest and latest["margin_complete"]) else "Not wired",
            )
    components.section_header("Alerts", "Typed, state-transition risk alerts committed with the ledger.")
    if report["alerts"]:
        st.dataframe(pd.DataFrame(report["alerts"]), hide_index=True, width="stretch")
    else:
        st.caption("No alert has been recorded yet.")


def _render_monitoring_tab(latest: dict | None) -> None:
    # Phase 21.1: fill rate / realized slippage / trade frequency / PnL
    # distribution, each computed from official C++ Fill evidence and
    # compared to the committed backtest ONLY where a real, experiment-
    # bound comparable artifact exists (never fabricated -- see
    # comparison_availability). Diagnostic only, never a pass/fail gate.
    components.section_header(
        "Drift Monitor",
        "Is paper behavior materially different from validated backtest evidence?",
    )
    drift = latest.get("drift") if latest else None
    if not drift:
        st.caption("No drift diagnostic has been recorded yet.")
    elif is_legacy_evidence(drift):
        st.info(
            "This step was recorded under the Phase 21 (pre-21.1) ledger schema, "
            "before the completed drift-monitor contract existed -- no drift "
            "evidence was fabricated for it.",
            icon="ℹ️",
        )
    else:
        avail = drift.get("comparison_availability", {})
        with components.metric_row("pt-drift"):
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                components.metric_card(
                    "pt-drift-fillrate", "Fill Rate",
                    f"{drift['fill_rate']:.0%}" if drift.get("fill_rate") is not None else "n/a",
                    sub=f"backtest: {avail.get('fill_rate', 'NOT_AVAILABLE')}",
                )
            with d2:
                slip = drift.get("realized_slippage", {})
                components.metric_card(
                    "pt-drift-slippage", "Realized Slippage",
                    _fmt_usd(slip.get("mean_slippage_usd")) + "/fill" if slip.get("mean_slippage_usd") is not None else "n/a",
                    sub=f"backtest: {avail.get('slippage', 'NOT_AVAILABLE')}",
                )
            with d3:
                freq = drift.get("trade_frequency", {})
                tpd = freq.get("trades_per_day")
                components.metric_card(
                    "pt-drift-freq", "Trade Frequency",
                    f"{tpd:.2f}/day" if tpd is not None else "n/a",
                    sub=f"backtest: {avail.get('trade_frequency', 'NOT_AVAILABLE')}",
                )
            with d4:
                pnl = drift.get("pnl_distribution", {})
                components.metric_card(
                    "pt-drift-pnl", "Mean Daily PnL",
                    _fmt_usd(pnl.get("mean_daily_pnl_usd")),
                    sub=f"backtest: {avail.get('pnl_distribution', 'NOT_AVAILABLE')}",
                )
        if drift.get("flags"):
            st.warning(
                "Drift flags (advisory monitoring only -- never a pass/fail gate): "
                + ", ".join(drift["flags"]),
                icon="📈",
            )
        with components.card("pt-drift-detail"):
            st.json(drift)
    components.section_header("Provenance", "Exact per-step execution provenance, committed with the step.")
    # Phase 21.1: exact per-step execution provenance (parent experiment
    # identity, strategy/risk-policy identities, and SHA-256 of the exact
    # bars/contracts/schedule/validation-days/executable/result bytes this
    # step actually used) -- committed with the step, never recomputed here.
    if latest is not None and is_legacy_evidence(latest.get("provenance") or {}):
        st.info(
            "This step was recorded under the Phase 21 (pre-21.1) ledger schema, "
            "before exact per-step execution provenance existed -- nothing was "
            "fabricated for it.",
            icon="ℹ️",
        )
        st.json(latest["provenance"])
    elif latest is not None and latest.get("provenance"):
        st.caption(
            "Exact provenance for the latest committed step -- every hash below is "
            "of the literal bytes sent to (or produced by) the C++ boundary."
        )
        st.json(latest["provenance"])
    else:
        st.caption("No step has been recorded yet.")


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active="paper", active_sub=st.session_state.get("paper_landing_focus"))
    layout.render_header(subtitle="Deterministic, hard-risk-gated replay -- not a live or paper broker connection.")

    st.markdown("## Paper Trading")
    st.caption(
        "Deterministic, hard-risk-gated replay of already-acquired historical data "
        "through the C++ Quant Core -- not a live or paper broker connection. "
        "No live-money routing exists in this system."
    )

    runs = services.list_paper_runs()
    eligible = services.paper_eligible_experiments()
    active_runs = sum(1 for r in runs if r["status"] == "ACTIVE")

    if not runs and not eligible:
        _render_empty_state(eligible_count=len(eligible), active_runs=active_runs)
    else:
        if runs:
            components.section_header(f"Paper runs ({len(runs)})")
            options = {
                f"{r['run_id']} -- {r['root_symbol']} {services.strategy_name(r['strategy_family'])} "
                f"[{r['status']}]": r["run_id"]
                for r in runs
            }
            labels = list(options.keys())
            # After a fresh "Start Paper Test", default the selector to the
            # run that action just created instead of leaving the user to
            # find it themselves.
            pending_run_id = st.session_state.pop("pt-run-select-pending", None)
            default_index = 0
            if pending_run_id is not None:
                for i, (label, run_id) in enumerate(options.items()):
                    if run_id == pending_run_id:
                        default_index = i
                        break
            choice = st.selectbox("Select a run", labels, index=default_index, key="pt-run-select")
            _render_run_detail(options[choice])

        _render_eligible_and_start(eligible)

    with st.expander("Details / Provenance / Developer", expanded=False):
        st.caption(
            "Phase 13.5C's reference historical research runs execute under a pass-through risk manager, "
            "not the hard risk engine -- paper trading (this page) is the first path in this system that "
            "runs under the real `PortfolioRiskManager` kill switches. Stepping/stopping an existing run "
            "is CLI-only: `python scripts/phase_21_paper_trading.py step|stop --run <run_id>` -- see "
            "docs/PAPER_TRADING_AND_DRIFT_MONITOR.md."
        )
        report = services.load_validation_report("NQ", "tsmom")
        if report:
            st.markdown("**Registry `risk_identity` (verbatim, not recomputed)**")
            st.json(report.get("risk_identity", {}))

    layout.render_disclaimer()
