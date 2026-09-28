"""Page 3 -- Strategies. The Phase 11 baseline family catalog plus the Phase 17
Strategy Compiler Agent: compiles the hypothesis proposed on the Research page
into a typed `StrategySpec` via the real, unmodified `StrategyCompilerAgent`.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.registry import TrialRole
from alpha_agent.schemas.hypothesis import HypothesisSpec
from alpha_agent.ui import components, layout, llm_demo, panels, recommendation_views, services
from alpha_agent.ui.charts import family_market_heatmap

PAGE_TITLE = "Strategies"


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active=None)
    layout.render_header(subtitle="What has been studied -- every tested strategy family and market.")
    render_body(services.approved_universe()[0])
    layout.render_disclaimer()


def render_body(root: str) -> None:
    """The page's own content, with no page chrome (no `inject_style`/
    `render_sidebar`/`render_header`/`render_disclaimer`) -- factored out so
    `views/research.py`'s consolidated "Strategies" tab can render this
    exact content inline without duplicating the shell. `render()` above
    keeps the module directly runnable/testable on its own (Product
    Consolidation campaign: this page is no longer a primary nav
    destination, but nothing about its own behavior changed)."""
    families = services.family_catalog()
    all_experiments = services.list_experiments()
    canonical = [e for e in all_experiments if e["trial_role"] == "CANONICAL"]
    unique_fingerprints = {e["strategy_fingerprint"] for e in all_experiments}

    with components.metric_row("strategies-metrics"):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            components.metric_card("total-strategies", "Total Strategies", str(len(unique_fingerprints)))
        with m2:
            components.metric_card("families", "Strategy Families", str(len(families)))
        with m3:
            components.metric_card("compiled", "Compiled Specs (session)",
                                    "1" if st.session_state.get("lab_compiled") else "0")
        with m4:
            components.metric_card("canonical", "Canonical Trials", str(len(canonical)))

    left, right = layout.main_rail_columns(gap="medium")

    with left:
        _render_research_status_table()

        components.section_header(
            "Strategy Library",
            "Every baseline strategy family this platform can research (Phase 11) -- economic mechanism, "
            "parameters, and default action, independent of whether it has been tested yet.",
        )
        st.dataframe(
            [
                {
                    "Family": f["family_key"],
                    "Name": f["name"],
                    "Parameters": ", ".join(f["parameters"]),
                    "Default Action": f["default_action"],
                }
                for f in families
            ],
            width="stretch", hide_index=True,
        )
        selected_family = st.selectbox(
            "Inspect strategy", [f["family_key"] for f in families], format_func=services.strategy_name,
        )
        card = next(f for f in families if f["family_key"] == selected_family)
        with components.card("family-detail"):
            st.markdown(f"**{card['name']}**")
            st.write(card["economic_mechanism"])
            st.caption(card["formula_and_timing"])
            st.markdown("**Parameter bounds**")
            st.json(card["param_bounds"])

        components.section_header(
            "Strategy × Market Matrix",
            "Release UX: moved here from the Dashboard -- every canonical trial, every root/family, "
            "real registry evidence. P=PASS · R=REJECT · I=INCONCLUSIVE · NA=NOT ADJUDICATED "
            "· MX=MIXED (canonical variants disagreed). Hover a cell for detail.",
        )
        with components.card("strategies-matrix"):
            components.plotly_chart(family_market_heatmap(services.strategy_market_matrix()))

        components.section_header("Compiled Strategy (this session)")
        proposal = st.session_state.get("ra_proposal")
        compiled = st.session_state.get("lab_compiled")
        if compiled and compiled.get("accepted"):
            with components.card("compiled-detail"):
                with components.metric_row("strategies-compiled"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Build mode", compiled["build_mode"])
                    c2.metric("Market", compiled["root_symbol"] or "-")
                    c3.metric(
                        "Strategy",
                        services.strategy_name(compiled["family_key"]) if compiled["family_key"] else "blueprint",
                    )
                st.code(compiled["strategy_fingerprint"], language="text")
                with st.expander("StrategySpec (typed, closed DSL)"):
                    st.json(compiled["strategy_spec"])
                with st.expander("Execution semantics"):
                    st.json(compiled["execution_semantics"])
                with st.expander("Compiler diagnostics"):
                    st.write(compiled["compile_diagnostics"] or "none")
        else:
            components.empty_state(
                "No compiled StrategySpec this session",
                "Generate a hypothesis on the Research page, then compile it from the panel on the right.",
                key="no-compiled",
            )

    with right:
        components.section_header("Strategy Compiler")
        with components.card("compiler-panel"):
            if not proposal or not proposal.get("accepted"):
                st.caption("No accepted hypothesis yet -- go to Research.")
            else:
                hypothesis = HypothesisSpec.model_validate(proposal["hypothesis"])
                st.markdown(f"**Selected hypothesis:** {hypothesis.title}")
                if st.button("Compile Strategy", type="primary", key="strategies-compile"):
                    universe = services.approved_universe()
                    scenario_key = st.session_state.get("ra_scenario_key")
                    result, error = llm_demo.compile_hypothesis(
                        mode=llm_demo.SCRIPTED_MODE, scenario_key=scenario_key,
                        hypothesis=hypothesis, universe=universe,
                    )
                    st.session_state["lab_compiled_error"] = error
                    st.session_state["lab_compiled"] = result.model_dump(mode="json") if result else None
                    if result and result.accepted:
                        panels.log_activity("OK", "Strategy compiled", result.strategy_fingerprint[:24] + "...")
                    st.rerun()

                lab_error = st.session_state.get("lab_compiled_error")
                if lab_error:
                    st.error(lab_error)

        if compiled and compiled.get("accepted"):
            with components.card("duplicate-evidence"):
                de = compiled["duplicate_evidence"]
                st.markdown("**Duplicate Evidence** (fingerprint only)")
                st.write(f"Seen before: **{de['strategy_fingerprint_seen']}**")
                st.caption(de["advisory"][:220] + "...")

                fc = compiled["feature_coverage"]
                st.markdown("**Feature Fidelity**")
                if fc["missing"] or fc["unapproved"]:
                    st.error(f"Missing: {fc['missing']} / Unapproved: {fc['unapproved']}")
                else:
                    st.success("All required feature kinds represented.")


def _render_research_status_table() -> None:
    """Task spec section 19: Validated / Promising Research / All Tested.
    Every value is a direct read of `services.list_experiments` /
    `services.research_candidate_summaries` / `services.paper_eligible_experiments`
    -- no new statistic or verdict is computed on this page."""
    components.section_header(
        "Strategies",
        "What strategies exist, and what is their research status -- Scientific Verdict, Research "
        "Promise, and User Fit stay independent.",
    )
    profile = panels.current_investor_profile()
    all_candidates = services.research_candidate_summaries(profile)
    by_experiment_id = {c.experiment_id: c for c in all_candidates}

    tab_validated, tab_promising, tab_all = st.tabs(["Validated", "Promising Research", "All Tested"])
    with tab_validated:
        validated = [c for c in all_candidates if c.scientific_verdict == "PASS"]
        recommendation_views.render_validated_section(validated, key_prefix="strat")
    with tab_promising:
        promising = services.promising_candidate_summaries(profile, sort_by="promise")
        recommendation_views.render_promising_section(promising, key_prefix="strat", limit=10)
    with tab_all:
        _render_all_tested_table(by_experiment_id)


def _render_all_tested_table(by_experiment_id: dict[str, object]) -> None:
    """Every approved market x baseline strategy combination, `Tested` only
    when a real canonical trial exists for it, `Untested` otherwise (never a
    fabricated placeholder verdict) -- unchanged from the pre-Phase-B1
    behaviour, plus the Research Promise / User Fit columns task spec
    section 19 requires."""
    families = services.family_catalog()
    universe = services.approved_universe()
    canonical = {
        (r["root_symbol"], r["strategy_family"]): r
        for r in services.list_experiments(trial_role=TrialRole.CANONICAL)
    }
    eligible_ids = {e["experiment_id"] for e in services.paper_eligible_experiments()}

    f1, f2, f3, f4 = st.columns(4)
    market_filter = f1.selectbox("Market", ["(any)", *universe], key="strat-status-market")
    family_filter = f2.selectbox(
        "Strategy Type", ["(any)", *(f["family_key"] for f in families)],
        format_func=lambda v: v if v == "(any)" else services.strategy_name(v), key="strat-status-family",
    )
    verdict_filter = f3.selectbox(
        "Verdict", ["(any)", "PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED", "Untested"],
        key="strat-status-verdict",
    )
    paper_filter = f4.selectbox("Paper Eligible", ["(any)", "YES", "NO"], key="strat-status-paper")

    rows = []
    for f in families:
        for root in universe:
            r = canonical.get((root, f["family_key"]))
            tested = r is not None
            candidate = by_experiment_id.get(r["experiment_id"]) if tested else None
            rows.append(
                {
                    "Strategy": services.strategy_name(f["family_key"]),
                    "_family": f["family_key"],
                    "Market": root,
                    "Research Status": "Tested" if tested else "Untested",
                    "Scientific Verdict": (r["verdict"] or "NOT_ADJUDICATED") if tested else "Untested",
                    "Research Promise": candidate.research_promise_label if candidate else "--",
                    "User Fit": recommendation_views.format_user_fit(candidate) if candidate else "--",
                    "Latest Sharpe": r["daily_sharpe"] if tested else None,
                    "Net PnL": r["net_pnl_usd"] if tested else None,
                    "Trades": r["n_trades"] if tested else None,
                    "Paper Eligible": "YES" if tested and r["experiment_id"] in eligible_ids else "NO",
                    "Last Evaluated": r["experiment_id"] if tested else "--",
                }
            )

    if market_filter != "(any)":
        rows = [x for x in rows if x["Market"] == market_filter]
    if family_filter != "(any)":
        rows = [x for x in rows if x["_family"] == family_filter]
    if verdict_filter != "(any)":
        rows = [x for x in rows if x["Scientific Verdict"] == verdict_filter]
    if paper_filter != "(any)":
        rows = [x for x in rows if x["Paper Eligible"] == paper_filter]
    for x in rows:
        x.pop("_family", None)

    if not rows:
        components.empty_state(
            "Strategies", "No strategies match the selected filters.", key="strat-status-empty",
        )
        return
    st.dataframe(rows, width="stretch", hide_index=True, height=min(420, 44 + 35 * len(rows)))

    tested_rows = [x for x in rows if x["Research Status"] == "Tested"]
    _render_open_detail_control(tested_rows)


def _render_open_detail_control(tested_rows: list[dict]) -> None:
    """"Clicking a strategy opens a detail view" (task spec section 17) --
    a compact select-then-view control, the same two-step pattern
    `views/experiment_log.py`'s "Inspect experiment" selector already uses,
    rather than a native per-row button `st.dataframe` cannot render. Sets
    the SAME `research_details_target` shape every other hand-off in this
    app builds (`views/agent.py::_build_research_details_target`,
    `views/market.py::_render_research_tab`) so `views/research.py` renders
    identically regardless of which page opened the detail view."""
    if not tested_rows:
        return
    options = {f"{r['Market']} -- {r['Strategy']} ({r['Scientific Verdict']})": r for r in tested_rows}
    c1, c2 = st.columns([3, 1])
    with c1:
        choice = st.selectbox("Open strategy detail", list(options.keys()), key="strat-open-detail-select")
    with c2:
        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        open_clicked = st.button("View Detail", key="strat-open-detail-button", width="stretch")
    if open_clicked:
        row = options[choice]
        st.session_state["research_details_target"] = {
            "source": "strategies", "objective": None, "root": row["Market"],
            "hypothesis": None, "compiled": None, "evidence": None, "run_outcome": None,
            "experiment_id": row["Last Evaluated"],
        }
        st.rerun()
