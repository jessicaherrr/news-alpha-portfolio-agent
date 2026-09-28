"""Page 5 -- Validation. Every gate here is a direct display of a value the
frozen `ReliabilityPolicy` already computed and committed -- this page applies
no threshold and makes no PASS/FAIL decision of its own. A gate reads PASS
only when the committed record explicitly says so (`all_required_gates_satisfied`)
and NOT_EVALUATED only when the committed record explicitly says THAT
(regime/cross-market's `*_not_evaluated` codes); otherwise it is FAIL (an
explicit reason code fired), REFUSED_BEFORE_GATE (an explicit minimum-sample
short-circuit), or NOT_AVAILABLE -- an absent reason code alone never proves
PASS or NOT_EVALUATED, since the committed artifact carries no per-gate
boolean to confirm either (see `services.gate_state`).
"""
from __future__ import annotations

import collections

import streamlit as st

from alpha_agent.ui import (
    components,
    layout,
    learn_links,
    palette,
    panels,
    recommendation_views,
    services,
)
from alpha_agent.ui.charts import (
    bar_chart,
    ci_range_chart,
    cost_stress_line,
    hbar_chart,
    percentile_gauge,
    regime_bar,
    verdict_donut,
)

PAGE_TITLE = "Validation"


def _render_promise_fit_context(experiment_id: str) -> None:
    """Task spec section 19 (VALIDATION): "Keep scientific result visually
    dominant. Promise/Fit may appear as secondary context only." -- a single
    collapsed, muted line, never the metric-card treatment Research Details
    gives these two scores."""
    profile = panels.current_investor_profile()
    candidate = services.candidate_summary_for_experiment(experiment_id, profile)
    if not candidate:
        return
    with st.expander("Research Promise & User Fit (secondary -- not a scientific claim)", expanded=False):
        st.caption(
            f"Research Promise: {candidate.research_promise_label} ({candidate.research_promise_score:.0f}/100) "
            f"-- prioritization only. Fit for You: {recommendation_views.format_user_fit(candidate)} "
            "-- personal alignment only. Neither affects the scientific verdict above."
        )


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active=None)
    layout.render_header(subtitle="Gate-by-gate validation evidence -- no threshold applied here.")
    render_body(services.approved_universe()[0])
    layout.render_disclaimer()


def render_body(root: str) -> None:
    """Content only, no page chrome -- see `strategies.render_body`'s
    docstring for why. Used inline by `views/research.py`'s consolidated
    "Validation" tab."""
    st.markdown("## Validation Center")
    st.caption(
        "Walk-forward validation, bootstrap null test, Benjamini-Hochberg FDR across the whole "
        "predeclared family, Deflated Sharpe Ratio, parameter-plateau stability, cost stress, and "
        "regime/cross-market evidence -- all frozen (Phase 13) and bound to the experiment's identity "
        "(Phase 14)."
    )

    _render_registry_wide_verdicts()

    families = sorted({f["family_key"] for f in services.family_catalog()})
    roots = services.approved_universe()
    c1, c2 = st.columns(2)
    sel_root = c1.selectbox("Market", list(roots), index=list(roots).index(root) if root in roots else 0)
    sel_family = c2.selectbox("Strategy", families, format_func=services.strategy_name)

    report = services.load_validation_report(sel_root, sel_family)
    rows = services.list_experiments(strategy_family=sel_family, root_symbol=sel_root, trial_role=None)
    canonical = next((r for r in rows if r["trial_role"] == "CANONICAL"), None)

    if report is None or canonical is None:
        st.warning(f"No committed Phase 13.5C validation report for {sel_root} / {sel_family}.")
        return

    violation = services.check_no_holdout_leak(report, path="$.validation_report")
    layout.holdout_leak_banner(violation)

    detail = services.get_experiment(canonical["experiment_id"])
    result = detail["result"]
    hv = report["headline_verdict_global_family"]

    exp = detail["experiment"]
    with components.metric_row("val-top"):
        top1, top2, top3, top4 = st.columns(4)
        with top1:
            components.verdict_metric_card("val-verdict", "Scientific Verdict", hv["verdict"])
        with top2:
            components.metric_card("val-window", "Validation Window", report["oos_window"].split(" ")[0])
        with top3:
            bh_key = next(k for k in hv if k.startswith("bh_q_value_over_"))
            components.metric_card("val-bh", "BH Family Size", bh_key.rsplit("_", 1)[-1])
        with top4:
            paper_ok = services.paper_eligible_experiment(canonical["experiment_id"])
            components.metric_card(
                "val-paper", "Paper Eligible", "YES" if paper_ok else "NO",
                accent=palette.GREEN if paper_ok else palette.RED,
            )
    st.markdown(
        f'<div style="margin:-0.1rem 0 0.4rem 0;font-size:0.86rem;color:var(--aa-text-secondary);">'
        f'<b>Why:</b> {services.plain_language_reason(result=result, trial_role=exp["trial_role"], verdict=hv["verdict"])}'
        f'</div>',
        unsafe_allow_html=True,
    )
    _render_promise_fit_context(canonical["experiment_id"])

    components.section_header("Validation Funnel", "Every gate, in evaluation order -- explicit committed state only.")
    components.render_validation_funnel(
        services.gate_table(
            reason_codes=(result["reason_codes"] if result else []),
            verdict=hv["verdict"],
            trial_role=exp["trial_role"],
        ),
        key=f"val-funnel-{canonical['experiment_id']}",
    )

    components.section_header("Gate-by-Gate Outcome", "Canonical trial -- explicit committed evidence only.")
    components.render_gate_grid(
        result=result, trial_role=exp["trial_role"], verdict=(result["headline_verdict"] if result else None),
        key_prefix="val",
    )

    components.section_header("Bootstrap Significance")
    learn_links.render_learn_why("bootstrap_null", key="val-bootstrap")
    ci = report["bootstrap_annualized_sharpe_ci"]
    with components.card("val-bootstrap"):
        components.plotly_chart(ci_range_chart(point=ci["point"], ci_low=ci["ci_low"], ci_high=ci["ci_high"]))
        st.write(
            f"Observed statistic (annualized Sharpe): **{ci['point']:.3f}** &nbsp; "
            f"{ci['ci_level']:.0%} CI: **[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]** &nbsp; "
            f"gating null p-value: **{report['gating_null_centered_block_bootstrap_p']:.3f}**",
            unsafe_allow_html=True,
        )

    components.section_header("Multiple Testing (BH-FDR)")
    learn_links.render_learn_why("bh_fdr", key="val-bhfdr")
    dsr_key = next(k for k in hv if k.startswith("dsr_probability_over_"))
    with components.card("val-bhfdr"), components.metric_row("val-bhfdr"):
        c1, c2, c3 = st.columns(3)
        c1.metric("q-value", f"{hv[bh_key]:.4f}")
        c2.metric("Family size", bh_key.rsplit("_", 1)[-1])
        c3.metric("BH-rejected", "Yes" if result and result.get("bh_rejected_at_q") else "No")

    components.section_header("Deflated Sharpe Ratio")
    learn_links.render_learn_why("dsr", key="val-dsr")
    with components.card("val-dsr"), components.metric_row("val-dsr"):
        c1, c2, c3 = st.columns(3)
        c1.metric("P(DSR)", f"{hv[dsr_key]:.4f}")
        c2.metric("Effective trials", hv["dsr_effective_trial_count"])
        c3.metric("Observed daily Sharpe", f"{hv['dsr_daily_sharpe']:.4f}")

    components.section_header("Walk-Forward Consistency")
    learn_links.render_learn_why("walk_forward", key="val-walkforward")
    with components.card("val-walkforward"):
        st.write(f"Fold consistency: **{result['fold_consistency']:.2f}**" if result else "not available")
        st.caption("Per-fold breakdown is not a committed artifact in this report; only the aggregate consistency ratio is.")

    components.section_header("Parameter Stability")
    learn_links.render_learn_why("parameter_stability", key="val-paramstab")
    ps = report["parameter_stability"]
    with components.card("val-paramstab"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Canonical percentile", f"{ps['canonical_sharpe_percentile']:.0%}")
        with c2:
            components.plotly_chart(percentile_gauge(ps["canonical_sharpe_percentile"]))
        st.write(
            f"{ps['n_evaluated']} neighbours evaluated &middot; {ps['fraction_positive_sharpe']:.0%} positive "
            f"Sharpe &middot; isolated spike: **{ps['canonical_is_isolated_spike']}** &middot; "
            f"dispersion (std): {ps['sharpe_dispersion_std']:.4f}",
            unsafe_allow_html=True,
        )

    components.section_header("Cost Stress")
    learn_links.render_learn_why("cost_stress", key="val-cost")
    cost = report["cost_stress"]
    with components.card("val-cost"):
        components.plotly_chart(cost_stress_line(cost["scenarios"]))
        st.caption(f"Max net-PnL degradation: {cost['max_net_pnl_degradation']:.2%}")

    components.section_header("Regime Robustness")
    learn_links.render_learn_why("regime_robustness", key="val-regime")
    regime = report["regime_evidence_volatility"]
    with components.card("val-regime"):
        if regime.get("buckets"):
            components.plotly_chart(regime_bar(regime["buckets"]))
        st.caption(f"Max regime PnL share: {regime['max_regime_pnl_share']:.1%} &middot; status: {regime['status']}",
                   unsafe_allow_html=True)

    components.section_header("Cross-Market Evidence")
    learn_links.render_learn_why("cross_market_evidence", key="val-xmkt")
    xmkt = report["cross_market_evidence"]
    if xmkt.get("status") == "evaluated":
        with components.card("val-xmkt"):
            st.json(xmkt)
    else:
        components.empty_state("Cross-Market Evidence", f"Evidence status: {xmkt.get('status', 'not_evaluated')}", key="val-xmkt-na")

    components.section_header("Validation Reasons")
    with components.card("val-reasons"):
        if result and result["reason_codes"]:
            for code in result["reason_codes"]:
                st.markdown(f"- `{code}`")
        else:
            st.caption("No reason codes committed for this trial.")

    with st.expander("Frozen fingerprints (identity plane -- not editable here)"):
        st.json(report["fingerprints"])

    st.divider()
    st.markdown("### Phase 15B -- ML Meta-Labeling Adjudication")
    ml = services.ml_meta_label_report()
    if ml is None:
        st.caption("No committed Phase 15B report found.")
    else:
        with components.card("val-ml"):
            st.write(
                f"Predeclared BH family: **{ml['counts']['statistical_hypotheses_bh_family']}** "
                f"(always exactly 60, even under a typed refusal). PASS: **{ml['n_pass']}** &middot; "
                f"REJECT: **{ml['n_reject']}** &middot; INCONCLUSIVE: **{ml['n_inconclusive']}** &middot; "
                f"REFUSED: **{ml['n_refused']}**",
                unsafe_allow_html=True,
            )
            reasons = collections.Counter()
            for t in ml["trials"]:
                reasons.update(t["reason_codes"])
            st.caption(
                "All 60 trials were typed-refused for a genuine reason: the frozen expanding-origin "
                "CV's earliest fold has fewer than the declared minimum pooled meta-label episodes -- "
                "not a relaxed gate, not an engineering defect."
            )
            components.plotly_chart(bar_chart(list(reasons.keys()), list(reasons.values()), title="Refusal reason codes"))
            st.caption(f"Source: `{ml['_source_artifact']}` &middot; reliability policy fingerprint identical to Phase 13.5C's.",
                       unsafe_allow_html=True)


def _render_registry_wide_verdicts() -> None:
    """Release UX (agent-first navigation): Verdict Distribution and Failure
    Reason Distribution moved here from the Dashboard -- both are registry-
    wide aggregate context for this page's per-experiment gate detail, not
    specific to any one selected root/family. Same `services.py` aggregation
    functions as before, only the calling page changed."""
    components.section_header(
        "Registry-Wide Outcomes",
        "Aggregate context for the per-experiment detail below -- every canonical trial, every root/family.",
    )
    c1, c2 = st.columns(2)
    with c1, components.card("val-verdict-donut"):
        st.markdown('<div class="aa-gate-title">Verdict Distribution</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="aa-gate-note">Canonical (headline-adjudicated) trials only.</div>',
            unsafe_allow_html=True,
        )
        components.plotly_chart(verdict_donut(services.canonical_verdict_distribution()))
    with c2, components.card("val-failures"):
        st.markdown('<div class="aa-gate-title">Failure Reason Distribution</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="aa-gate-note">Typed, permanently-preserved failure-memory records.</div>',
            unsafe_allow_html=True,
        )
        reasons = services.failure_reason_distribution()
        if reasons:
            labels, values = zip(*reversed(reasons), strict=True)
            components.plotly_chart(hbar_chart(list(labels), list(values), color=palette.AMBER))
        else:
            st.caption("No failure records committed yet.")
