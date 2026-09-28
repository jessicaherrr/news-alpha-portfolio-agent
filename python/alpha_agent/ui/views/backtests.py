"""Page 4 -- Backtests. A professional tear-sheet view of one experiment's
authoritative registry evidence. Every number is read verbatim from its
`ResultRecord`; a trade ledger is shown ONLY when
`services.find_experiment_bound_trade_ledger` proves it is bound to THIS
experiment (never matched by root symbol) -- otherwise the page says so
explicitly rather than fabricating one.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.ui import charts, charts_discovery, components, layout, palette, services
from alpha_agent.ui.charts import ci_range_chart, cost_stress_line, regime_bar

PAGE_TITLE = "Backtests"


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active=None)
    layout.render_header(subtitle="Backtest tear sheet -- one experiment's authoritative registry evidence.")

    st.markdown("## Backtest Tear Sheet")
    st.caption(
        "Authoritative results from the experiment registry (Phase 14) -- the highest-ordinal "
        "VALID execution attempt for each hypothesis. Reference runs execute under "
        "`PassThroughRiskManager`; hard risk limits are a separate C++ layer."
    )

    families = sorted({f["family_key"] for f in services.family_catalog()})
    roots = services.approved_universe()
    f1, f2, f3, f4 = st.columns(4)
    sel_root = f1.selectbox("Market", list(roots), index=0)
    sel_family = f2.selectbox("Strategy", families, format_func=services.strategy_name)
    rows = services.list_experiments(strategy_family=sel_family, root_symbol=sel_root, trial_role=None)
    role_options = sorted({r["trial_role"] for r in rows}) or ["CANONICAL"]
    sel_role = f3.selectbox("Variant / Role", role_options)
    candidates = [r for r in rows if r["trial_role"] == sel_role]
    exp_ids = [r["experiment_id"] for r in candidates] or ["(none)"]
    sel_exp = f4.selectbox("Experiment", exp_ids)

    if sel_exp == "(none)":
        st.warning("No experiment matches this filter.")
        layout.render_disclaimer()
        return

    detail = services.get_experiment(sel_exp)
    violation = services.check_no_holdout_leak(detail, path="$.backtests_detail")
    layout.holdout_leak_banner(violation)
    result = detail["result"]

    if result is None:
        st.info("This hypothesis has no VALID authoritative result yet (invalid-attempt history only).")
        layout.render_disclaimer()
        return

    with components.metric_row("bt-metrics"):
        m = st.columns(7)
        with m[0]:
            components.metric_card("bt-gross", "Gross PnL", f"${result['gross_pnl_usd']:,.0f}")
        with m[1]:
            components.metric_card("bt-costs", "Costs", f"${result['costs_usd']:,.0f}")
        with m[2]:
            components.metric_card("bt-net", "Net PnL", f"${result['net_pnl_usd']:,.0f}")
        with m[3]:
            components.metric_card("bt-sharpe", "Annual Sharpe", f"{result['annualized_sharpe']:.2f}")
        with m[4]:
            components.metric_card("bt-trades", "Trades", str(result["n_trades"]))
        with m[5]:
            components.metric_card("bt-fills", "Fills", str(result["n_fills"]))
        with m[6]:
            components.verdict_metric_card("bt-verdict", "Verdict", result["headline_verdict"])

    if result["headline_verdict"] != "PASS":
        st.caption(
            "[!] Backtest Completed is not the same as Scientific PASS -- the numbers above are a "
            "completed historical simulation; see the Validation page for why this strategy did or "
            "did not clear the frozen statistical requirements."
        )
    st.caption("Reason codes: " + ("; ".join(result["reason_codes"]) if result["reason_codes"] else "none"))
    st.caption(f"Source artifact: `{result.get('source_artifact') or 'unknown'}`")

    ledger = services.find_experiment_bound_trade_ledger(detail)
    series = services.trade_ledger_equity_series(ledger)
    if series:
        max_dd = services.max_drawdown_usd(series)
        with components.metric_row("bt-dd"):
            components.metric_card(
                "bt-maxdd", "Max Drawdown", f"${max_dd:,.0f}" if max_dd is not None else "N/A",
                accent=palette.RED if max_dd else None,
            )
        c1, c2 = st.columns(2)
        with c1, components.card("bt-equity"):
            st.markdown('<div class="aa-gate-title">Equity Curve</div>', unsafe_allow_html=True)
            components.plotly_chart(charts.equity_curve_chart([r["cum_net_pnl_usd"] for r in series]))
        with c2, components.card("bt-drawdown"):
            st.markdown('<div class="aa-gate-title">Drawdown</div>', unsafe_allow_html=True)
            components.plotly_chart(charts.drawdown_chart([r["drawdown_usd"] for r in series]))
        st.caption(f"From the bound ledger `{ledger['source_artifact']}` -- {ledger['n_trades']} real trades.")
    else:
        components.committed_series_notice(key="bt-series-na")

    render_signals_section(detail, key_prefix="bt")

    components.section_header("Cost Sensitivity", "Directly from this experiment's committed ResultRecord.cost_stress.")
    cost = result.get("cost_stress") or {}
    if cost.get("scenarios"):
        with components.card("bt-cost-chart"):
            components.plotly_chart(cost_stress_line(cost["scenarios"]))
            st.caption(f"Max net-PnL degradation under stress: {cost['max_net_pnl_degradation']:.2%}")
    else:
        components.empty_state("Cost Sensitivity", "No cost-stress evidence committed for this experiment.", key="bt-cost")

    components.section_header("Bootstrap Evidence", "The committed summary CI -- not a fabricated sample distribution.")
    boot = result.get("bootstrap_evidence") or {}
    if boot:
        with components.card("bt-bootstrap"):
            components.plotly_chart(
                ci_range_chart(point=boot["point"], ci_low=boot["ci_low"], ci_high=boot["ci_high"])
            )
            st.caption(f"{boot['ci_level']:.0%} CI: [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}], point {boot['point']:.3f}")

    components.section_header("Regime Evidence")
    regime = result.get("regime_evidence") or {}
    if regime.get("status") == "evaluated" and regime.get("buckets"):
        with components.card("bt-regime"):
            components.plotly_chart(regime_bar(regime["buckets"]))
    else:
        components.empty_state("Regime Evidence", f"Evidence status: {regime.get('status', 'not_evaluated')}", key="bt-regime-na")

    components.section_header("Trade Ledger", "Shown only when verifiably bound to this exact experiment.")
    if ledger:
        with components.card("bt-ledger"):
            st.caption(f"Source: `{ledger['source_artifact']}` (sha256 {ledger['source_artifact_sha256'][:16]}...)")
            st.dataframe(ledger["trades"], width="stretch", hide_index=True, height=320)
    else:
        components.empty_state(
            "Trade Ledger",
            "No committed artifact names a per-trade ledger as THIS experiment's source (never inferred "
            "from a shared root symbol). The reference C++ engine produces a per-trade ledger on every "
            "run; a future phase committing it with explicit experiment provenance will surface it here.",
            key="bt-ledger-na",
        )

    with st.expander("Execution / contract provenance"):
        exp = detail["experiment"]
        components.provenance_row("Strategy fingerprint", exp["strategy_fingerprint"])
        components.provenance_row("Dataset fingerprint", exp["dataset_fingerprint"])
        components.provenance_row("Execution config identity", exp["execution_config_identity"])
        components.provenance_row("Cost config identity", exp["cost_config_identity"])
        components.provenance_row("Risk identity", exp["risk_identity"])
        components.provenance_row("Code commit", exp.get("code_commit") or "--")

    layout.render_disclaimer()


def render_signals_section(detail: dict, *, key_prefix: str = "bt") -> None:
    """Price + real fill markers + actual position (Release UX Part G;
    MARKET REALITY pass mission section 11/12/13) -- ONLY when this exact
    experiment's own committed `ResultRecord.source_artifact` names a
    hash-verified artifact bundle carrying real `price_bars`/`trades`/
    `fills` (`services.find_experiment_bound_artifact_bundle`). Shared by the
    Backtests tear sheet and the Research Details "Signals" tab so both
    render identically from the same evidence -- never reconstructed from
    summary statistics. `key_prefix` keeps Streamlit element keys unique
    when both pages happen to render in the same session."""
    components.section_header("Signals", "Real price, fills, and actual position for one bound trade.")
    bundle = services.find_experiment_bound_artifact_bundle(detail)
    if not bundle or not bundle.get("price_bars") or not bundle.get("trades"):
        components.empty_state(
            "Price + Signal",
            "No committed artifact bundle with real price_bars/trades/fills is bound to this exact "
            "experiment yet -- never reconstructed from summary metrics.",
            key=f"{key_prefix}-signals-na",
        )
        return

    trades = services.trades_rows_from_bundle(bundle) or []
    if not trades:
        components.empty_state("Price + Signal", "The bound bundle carries no trades.", key=f"{key_prefix}-signals-empty")
        return

    options = {
        f"#{t['trade_index']} {t['raw_symbol']} {t['direction']=='1' and 'long' or 'short'} "
        f"net ${float(t['net_pnl_usd']):,.0f}": int(t["trade_index"])
        for t in trades
    }
    with components.card(f"{key_prefix}-signals"):
        label = st.selectbox("Trade", list(options.keys()), key=f"{key_prefix}-signal-trade")
        trade_index = options[label]
        window = services.price_signal_window_from_bundle(bundle, trade_index=trade_index, context_bars=60)
        if not window or not window.get("bars"):
            components.empty_state(
                "Price + Signal", "No real price bars align with this trade.", key=f"{key_prefix}-signals-window-na",
            )
            return
        st.caption("Legend: entries/exits are placed only at this trade's own real fill timestamps/prices "
                   "-- never inferred from a summary statistic. Position below is derived from the same trade.")
        components.plotly_chart(charts_discovery.price_signal_chart(window))
        st.caption(
            f"{len(window['bars'])} real price bar(s), {len(window['fills'])} real fill(s) -- "
            f"source `{bundle['schema_version']}` bundle for `{bundle['experiment_identity']}`."
        )
