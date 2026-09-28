"""Research Golden Path -- the Research page's own DISCOVER -> HYPOTHESIS ->
COMPILE -> C++ TEST -> VALIDATE -> DECISION BRIEF workflow tab.

This is the ONE new place besides `alpha_agent.ui.deep_research` that a UI
click may trigger a real, registry-writing `ResearchOrchestrator` run --
`_run_historical_test` below calls `deep_research.run_deep_research` verbatim
(the same single boundary `views/agent.py`'s "Run This Hypothesis" already
uses), never a second, duplicated orchestration path. Every rendering step
reuses the SAME real primitives the rest of the app already has: the closed
`FeatureRegistry` projection (`factor_library`), the frozen Phase 13.5C
candidate manifest (`candidate_mechanisms`), the runtime `ResearchAgent` /
`StrategyCompilerAgent` (`llm_demo`), the typed `StrategySpec` DSL
(`strategy_spec_view`), the real C++ execution result + validation gates
(`services`), and Claude only as a bounded synthesis layer
(`decision_brief`) -- never a new engine.

Session state is namespaced `rwf_*` (Research WorkFlow) so it never collides
with the Agent page's `agent_*` / `ra_*` / `lab_*` keys -- a user may have an
independent research thread open on each page at once.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from alpha_agent.schemas.hypothesis import HypothesisSpec
from alpha_agent.ui import (
    candidate_mechanisms,
    charts,
    components,
    decision_brief,
    deep_research,
    execution_provenance,
    factor_library,
    learn_links,
    llm_demo,
    palette,
    services,
    strategy_spec_view,
)
from alpha_agent.ui.views.backtests import render_signals_section

_STEPS = ("Discover", "Hypothesis", "Compile", "C++ Test", "Validate", "Decision Brief")

#: The 4 baseline families with a canned Offline / Deterministic scenario
#: (`llm_demo.SCENARIOS`) -- see that module's docstring for why each scenario
#: is safely re-rooted to any approved market. Silver Bullet has no canned
#: scenario (it is an NQ-only benchmark, not a cross-root convenience family),
#: so a deterministic proposal for it is genuinely unavailable here -- never
#: faked with an unrelated scenario.
_FAMILY_TO_SCENARIO = {
    "tsmom": "tsmom_nq",
    "mean_reversion": "mean_reversion_cl",
    "breakout": "breakout_gc",
    "ma_trend": "ma_trend_zn",
}


def _reset_thread() -> None:
    for key in ("rwf_hypothesis", "rwf_hypothesis_origin", "rwf_scenario_key", "rwf_mode",
                "rwf_compiled", "rwf_run_outcome", "rwf_confirm_pending", "rwf_brief"):
        st.session_state.pop(key, None)


def _generate_hypothesis(*, root: str, mode: str, scenario_key: str, objective: str, origin_source: str) -> None:
    proposal, error, _fm = llm_demo.propose_hypothesis(
        mode=mode, scenario_key=scenario_key, universe=(root,), objective=objective,
    )
    if error or not proposal or not proposal.accepted:
        st.session_state["rwf_error"] = error or (
            f"{proposal.rejection_code}: {proposal.rejection_detail}" if proposal else "no proposal"
        )
        return
    st.session_state.pop("rwf_error", None)
    st.session_state["rwf_hypothesis"] = proposal.hypothesis.model_dump(mode="json")
    st.session_state["rwf_hypothesis_origin"] = {"source": origin_source, "mode": mode}
    st.session_state["rwf_scenario_key"] = scenario_key
    st.session_state["rwf_mode"] = mode
    st.session_state.pop("rwf_compiled", None)
    st.session_state.pop("rwf_run_outcome", None)
    st.session_state.pop("rwf_brief", None)


# ---------------------------------------------------------------------------
# Step 1 -- Discover: Factor Library + Candidate Mechanisms
# ---------------------------------------------------------------------------


def _render_factor_library(root: str) -> None:
    components.section_header(
        "Factor Library",
        "Real registered feature kinds (`alpha_agent.features.REGISTRY`), grouped into mechanism "
        "families -- only a family with real registered features gets a card.",
    )
    cards = factor_library.mechanism_families()
    cols = st.columns(3)
    for i, card in enumerate(cards):
        with cols[i % 3], components.card(f"rwf-factor-{card['family']}"):
            st.markdown(f"**{card['label']}**")
            st.caption(card["description"])
            st.markdown(f"{card['n_features']} available feature(s)")
            components.render_tags(card["feature_kinds"])
            status = card["coverage_status"]
            if status == factor_library.NOT_APPLICABLE:
                st.caption("No strategy family currently built on this feature family yet.")
            else:
                components.render_badge(
                    {"TESTED": "OK", "PARTIALLY_TESTED": "WARN", "RESEARCH_GAP": "NOT_AVAILABLE"}[status],
                    label=status.replace("_", " "),
                )
                st.caption(f"{card['n_experiments']} historical experiment(s) across the approved universe.")
    axes = factor_library.unsupported_axes()
    st.caption(
        "Not yet supported by the current feature schema: "
        + "; ".join(f"{a['label']} ({a['reason']})" for a in axes)
    )


def _render_deterministic_candidates(root: str) -> None:
    st.markdown("**Deterministic (Frozen Candidate Grid)**")
    st.caption(
        "Canonical trials from the frozen Phase 13.5C candidate manifest -- pre-declared before any "
        "performance was observed."
    )
    for cand in candidate_mechanisms.deterministic_candidates(root):
        with components.card(f"rwf-cand-{cand['family_key']}"):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                st.markdown(f"**{cand['strategy_name']}** ({root})")
                st.caption(", ".join(f"{k}={v}" for k, v in cand["canonical_params"].items()))
            with c2:
                if cand["tested"]:
                    components.render_badge(cand["verdict"] or "NOT_ADJUDICATED", label="TESTED")
                else:
                    components.render_badge("NOT_AVAILABLE", label="RESEARCH GAP -- UNTESTED")
            with c3:
                if cand["tested"]:
                    if st.button("View result", key=f"rwf-view-{cand['family_key']}"):
                        st.session_state["research_details_target"] = {
                            "source": "research_workflow", "objective": None, "root": root,
                            "hypothesis": None, "compiled": None, "evidence": None, "run_outcome": None,
                            "experiment_id": cand["experiment_id"],
                        }
                        st.rerun()
                else:
                    scenario_key = _FAMILY_TO_SCENARIO.get(cand["family_key"])
                    disabled = scenario_key is None
                    if st.button(
                        "Propose hypothesis", key=f"rwf-propose-{cand['family_key']}", disabled=disabled,
                        help=None if scenario_key else "No canned Offline scenario for this family/root yet.",
                    ):
                        sc = llm_demo.scenario(scenario_key)
                        _generate_hypothesis(
                            root=root, mode=llm_demo.SCRIPTED_MODE, scenario_key=scenario_key,
                            objective=sc.objective, origin_source="deterministic",
                        )
                        st.rerun()


def _render_research_memory_candidates(root: str) -> None:
    candidates = candidate_mechanisms.research_memory_candidates(root)
    if not candidates:
        return
    st.markdown("**Research Memory**")
    st.caption("An untested candidate whose neighbourhood already carries a real, actionable lesson.")
    for cand in candidates:
        with components.card(f"rwf-mem-cand-{cand['family_key']}"):
            st.markdown(f"**{cand['strategy_name']}** ({root})")
            st.info(cand["lesson"])


def _render_claude_proposal(root: str) -> None:
    st.markdown("**Claude Proposed**")
    st.caption(
        "Claude may propose a hypothesis from the available features, current evidence, and research "
        "gaps -- it is a HYPOTHESIS CANDIDATE, not validated, and it still compiles through the closed "
        "typed Strategy DSL like every other candidate."
    )
    with components.card("rwf-claude-propose"):
        scenario_options = {s.label: s.key for s in llm_demo.SCENARIOS}
        preset_label = st.selectbox(
            "Preset objective (optional starting point)", list(scenario_options.keys()), key="rwf-preset",
        )
        scenario_key = scenario_options[preset_label]
        objective = st.text_area(
            "Research objective", value=llm_demo.scenario(scenario_key).objective, height=80, key="rwf-objective",
        )
        mode_label = st.radio(
            "Research Engine", options=["Offline / Deterministic", "Claude Research"],
            index=0, key="rwf-claude-mode", horizontal=True,
        )
        mode = llm_demo.LIVE_MODE if mode_label == "Claude Research" else llm_demo.SCRIPTED_MODE
        if st.button("Ask Claude to propose a mechanism", key="rwf-claude-go", type="primary"):
            _generate_hypothesis(
                root=root, mode=mode, scenario_key=scenario_key, objective=objective, origin_source="claude",
            )
            st.rerun()


def _render_discover(root: str) -> None:
    st.caption("Choose a market, inspect available factors, and select or generate a mechanism to research.")
    _render_factor_library(root)
    components.section_header("Candidate Mechanisms", "Each candidate declares its own provenance -- never silently mixed.")
    _render_deterministic_candidates(root)
    _render_research_memory_candidates(root)
    _render_claude_proposal(root)
    if st.session_state.get("rwf_error"):
        st.error(st.session_state["rwf_error"])


# ---------------------------------------------------------------------------
# Step 2 -- Hypothesis
# ---------------------------------------------------------------------------


def _render_hypothesis_step(root: str) -> None:
    h = st.session_state.get("rwf_hypothesis")
    if not h:
        components.empty_state(
            "Hypothesis", "No hypothesis yet -- pick or propose a candidate mechanism in Discover.", key="rwf-hyp-na",
        )
        return
    origin = st.session_state.get("rwf_hypothesis_origin") or {}
    with components.card("rwf-hyp"):
        st.markdown(f"### {h['title']}")
        st.caption(
            f"Origin: Historical Research Discovery ({origin.get('source', 'unknown')}) &middot; "
            "Holdout eligibility: **Eligible** -- this thread was seeded from a research question, "
            "never a current-observation seed.",
            unsafe_allow_html=True,
        )
        st.markdown("**Economic mechanism**")
        st.write(h["economic_mechanism"])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Universe / Horizon**")
            st.write(", ".join(h["universe"]))
            st.write(h["horizon"])
        with c2:
            st.markdown("**Expected / Failure regime**")
            st.write(h["expected_regime"])
            st.write(h["failure_regime"])
        with c3:
            st.markdown("**Required features**")
            components.render_tags(h["required_features"])
        st.markdown("**Falsification test**")
        st.write(h["falsification_test"])

    if st.button("Compile Strategy", key="rwf-compile-go", type="primary"):
        mode = st.session_state.get("rwf_mode", llm_demo.SCRIPTED_MODE)
        scenario_key = st.session_state.get("rwf_scenario_key")
        compiled, error = llm_demo.compile_hypothesis(
            mode=mode, scenario_key=scenario_key, hypothesis=HypothesisSpec.model_validate(h), universe=(root,),
        )
        if error or not compiled or not compiled.accepted:
            st.session_state["rwf_error"] = error or (compiled.rejection_detail if compiled else "no compiled spec")
        else:
            st.session_state.pop("rwf_error", None)
            st.session_state["rwf_compiled"] = compiled.model_dump(mode="json")
            st.session_state.pop("rwf_run_outcome", None)
        st.rerun()
    if st.session_state.get("rwf_error"):
        st.error(st.session_state["rwf_error"])


# ---------------------------------------------------------------------------
# Step 3 -- Compile
# ---------------------------------------------------------------------------


def _render_compile_step(root: str) -> None:
    cd = st.session_state.get("rwf_compiled")
    if not cd:
        components.empty_state("Compile", "No compiled StrategySpec yet -- accept a hypothesis first.", key="rwf-cd-na")
        return

    summary = strategy_spec_view.summarize_strategy_spec(cd["strategy_spec"])
    supported = services.execution_supported_family(cd.get("family_key"))
    with components.card("rwf-compiled"):
        st.markdown(f"### {summary['strategy_name']}")
        with components.metric_row("rwf-compiled"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Market", summary["root_symbol"] or "-")
            c2.metric("Build mode", cd["build_mode"])
            c3.metric("Execution support", "SUPPORTED" if supported else "UNSUPPORTED")
        st.markdown("**Features**")
        components.render_tags([f"{f['alias']} ({f['kind']})" for f in summary["features"]])
        st.markdown("**Rules**")
        for r in summary["rules"]:
            st.write(f"- {r['text']}")
            if r["rationale"]:
                st.caption(r["rationale"])
        st.caption(f"Default action: {summary['default_action']}. On missing data: {summary['on_missing']}.")
        st.code(cd["strategy_fingerprint"], language="text")
        with st.expander("View typed spec (raw StrategySpec JSON)", expanded=False):
            st.json(cd["strategy_spec"])

        de = cd.get("duplicate_evidence") or {}
        if de:
            st.markdown("**Strategy fingerprint history** (evidence only)")
            st.caption(de.get("advisory", ""))
        fc = cd.get("feature_coverage") or {}
        if fc.get("missing") or fc.get("unapproved"):
            st.error(f"Feature fidelity -- missing: {fc.get('missing')} / unapproved: {fc.get('unapproved')}")
        else:
            st.success("Feature fidelity: all required feature kinds represented.")

    if not supported:
        st.warning(
            "Execution support: UNSUPPORTED for this build mode/family -- a genuine capability gap, "
            "never a scientific outcome. Run Historical Test is disabled.", icon="\U0001f6a7",
        )
        return

    evidence = services.find_registry_evidence_for_compiled(cd)
    already_tested = bool(evidence and evidence.get("match_type") == "exact_fingerprint" and evidence.get("result"))
    if already_tested:
        st.caption(
            "This exact StrategySpec already has a committed, authoritative result -- see C++ Test / "
            "Validate for the existing evidence. Running again would not change the scientific record."
        )

    run_label = "Run Historical Test" if not already_tested else "Re-run anyway (reproduce evidence)"
    if st.button(run_label, key="rwf-run-go", type="primary"):
        st.session_state["rwf_confirm_pending"] = True
        st.rerun()

    if st.session_state.get("rwf_confirm_pending"):
        with components.card("rwf-run-confirm"):
            st.markdown("**Run scientific experiment?**")
            st.markdown(
                "This action may run the local C++ backtest, run frozen scientific validation, and append "
                "a genuinely novel result to the ExperimentRegistry. It will NOT access the 2025 holdout "
                "or submit broker orders. Typical runtime: roughly 1-3 minutes."
            )
            c1, c2 = st.columns(2)
            cancel = c1.button("Cancel", key="rwf-run-cancel", width="stretch")
            confirm = c2.button("Confirm Run", key="rwf-run-confirm", type="primary", width="stretch")
        if cancel:
            st.session_state["rwf_confirm_pending"] = False
            st.rerun()
        if confirm:
            st.session_state["rwf_confirm_pending"] = False
            _run_historical_test(cd)
            st.rerun()


def _run_historical_test(cd: dict) -> None:
    """Reuses `deep_research.run_deep_research` verbatim -- the ONE UI
    boundary that may write to the registry. `root`/`family_stem` are derived
    from the compiled StrategySpec, never re-entered."""
    mode = st.session_state.get("rwf_mode", llm_demo.SCRIPTED_MODE)
    scenario_key = st.session_state.get("rwf_scenario_key")
    objective = (st.session_state.get("rwf_hypothesis") or {}).get("economic_mechanism", "")
    root = cd["root_symbol"]
    family_stem = f"rwf-{cd.get('family_key') or 'blueprint'}"
    with st.spinner("Planning -> executing the real C++ backtest -> validating -> finalizing..."):
        outcome = deep_research.run_deep_research(
            mode=mode, objective=objective, root=root, family_stem=family_stem,
            scenario_key=scenario_key if mode == llm_demo.SCRIPTED_MODE else None,
        )
    st.session_state["rwf_run_outcome"] = outcome


# ---------------------------------------------------------------------------
# Step 4 -- C++ Test
# ---------------------------------------------------------------------------


def _current_detail() -> dict[str, Any] | None:
    """The experiment detail this workflow's current run actually produced,
    or an already-tested exact-fingerprint match -- never a fabricated one."""
    outcome = st.session_state.get("rwf_run_outcome")
    if outcome is not None and outcome.accepted and outcome.report and outcome.report.member_results:
        exp_id = outcome.report.member_results[0].experiment_id
        if exp_id:
            return services.get_experiment(exp_id)
    cd = st.session_state.get("rwf_compiled")
    if cd:
        evidence = services.find_registry_evidence_for_compiled(cd)
        if evidence and evidence.get("match_type") == "exact_fingerprint":
            return evidence
    return None


def _render_cpp_test_step() -> None:
    outcome = st.session_state.get("rwf_run_outcome")
    if outcome is None:
        components.empty_state("C++ Test", "Run the historical test in Compile first.", key="rwf-cpp-na")
        return
    if not outcome.accepted:
        st.markdown(components.badge("FAIL", label="EXECUTION FAILED"), unsafe_allow_html=True)
        st.error(outcome.error)
        return

    from alpha_agent.ui.views.research import _render_family_status_banner

    _render_family_status_banner(outcome.report)
    if outcome.report.status.value == "INCOMPLETE_NOT_ADJUDICATED":
        return
    mr = outcome.report.member_results[0] if outcome.report.member_results else None
    if mr is None or not mr.experiment_id:
        st.caption("No member result recorded for this run.")
        return

    detail = services.get_experiment(mr.experiment_id)
    result = detail["result"]

    components.section_header("Execution Provenance", "Read from the real run/configuration -- section 8A.")
    learn_links.render_learn_why("look_ahead_bias", key="rwf-exec-prov", label="Learn Why: No Look-Ahead")
    with components.card("rwf-exec-prov"):
        for label, text in execution_provenance.execution_provenance_rows(detail):
            components.provenance_row(label, text)

    components.section_header("C++ Result", "Authoritative -- read verbatim from the committed ResultRecord.")
    with components.metric_row("rwf-cpp-metrics"):
        m = st.columns(6)
        with m[0]:
            components.metric_card("rwf-gross", "Gross PnL", f"${result['gross_pnl_usd']:,.0f}")
        with m[1]:
            components.metric_card("rwf-costs", "Costs", f"${result['costs_usd']:,.0f}")
        with m[2]:
            components.metric_card("rwf-net", "Net PnL", f"${result['net_pnl_usd']:,.0f}")
        with m[3]:
            components.metric_card("rwf-sharpe", "Annual Sharpe", f"{result['annualized_sharpe']:.2f}")
        with m[4]:
            components.metric_card("rwf-trades", "Trades", str(result["n_trades"]))
        with m[5]:
            components.metric_card("rwf-fills", "Fills", str(result["n_fills"]))

    # Preferred source: a hash-verified Part I artifact bundle (real daily
    # equity trace + trades.csv) -- the SAME bundle the Execution Explainer
    # below already binds and verifies. Falls back to the older, narrower
    # `find_experiment_bound_trade_ledger` (a direct `trades.csv` pointer)
    # only when no bundle is bound -- see the matching comment on
    # `views/research.py::_render_backtest_tab`.
    bundle = services.find_experiment_bound_artifact_bundle(detail)
    ledger = None if bundle else services.find_experiment_bound_trade_ledger(detail)
    series = services.daily_equity_series_from_bundle(bundle) if bundle else services.trade_ledger_equity_series(ledger)
    trades_rows = services.trades_rows_from_bundle(bundle) if bundle else (ledger["trades"] if ledger else None)
    if series:
        max_dd = services.max_drawdown_usd(series)
        with components.metric_row("rwf-dd"):
            components.metric_card(
                "rwf-maxdd", "Max Drawdown", f"${max_dd:,.0f}" if max_dd is not None else "N/A",
                accent=palette.RED if max_dd else None,
            )
        c1, c2 = st.columns(2)
        with c1, components.card("rwf-equity"):
            st.markdown('<div class="aa-gate-title">Equity Curve</div>', unsafe_allow_html=True)
            components.plotly_chart(charts.equity_curve_chart([r["cum_net_pnl_usd"] for r in series]))
        with c2, components.card("rwf-drawdown"):
            st.markdown('<div class="aa-gate-title">Drawdown</div>', unsafe_allow_html=True)
            components.plotly_chart(charts.drawdown_chart([r["drawdown_usd"] for r in series]))
        with components.card("rwf-ledger"):
            st.markdown("**Trade Ledger**")
            st.dataframe(trades_rows, width="stretch", hide_index=True, height=280)
    else:
        components.committed_series_notice(key="rwf-series-na")

    components.section_header("Execution Explainer", "Signal -> next executable bar -> fill -> position -> realized result.")
    render_signals_section(detail, key_prefix="rwf")


# ---------------------------------------------------------------------------
# Step 5 -- Validate
# ---------------------------------------------------------------------------


def _render_validate_step() -> None:
    detail = _current_detail()
    if not detail or not detail.get("result"):
        components.empty_state("Validate", "No committed result yet -- run the historical test first.", key="rwf-val-na")
        return
    result = detail["result"]
    exp = detail["experiment"]
    verdict = result["headline_verdict"]
    components.render_badge(verdict or "NOT_ADJUDICATED")
    st.caption(services.plain_language_reason(result=result, trial_role=exp["trial_role"], verdict=verdict))
    components.render_gate_grid(result=result, trial_role=exp["trial_role"], verdict=verdict, key_prefix="rwf-val")
    with components.card("rwf-val-reasons"):
        st.markdown("**Reason codes**")
        for code in (result["reason_codes"] or ["none"]):
            st.markdown(f"- `{code}`")


# ---------------------------------------------------------------------------
# Step 6 -- Decision Brief
# ---------------------------------------------------------------------------


def _render_decision_brief_step() -> None:
    h = st.session_state.get("rwf_hypothesis")
    detail = _current_detail()
    exp_id = detail["experiment"]["experiment_id"] if detail else None

    st.caption(
        "Claude synthesizes the evidence already gathered above into a research conclusion. It never "
        "assigns a scientific verdict and never recommends BUY/SELL."
    )
    mode_label = st.radio(
        "Brief Engine", options=["Offline / Deterministic", "Claude Research"],
        index=0, key="rwf-brief-mode", horizontal=True,
    )
    mode = decision_brief.CLAUDE_MODE if mode_label == "Claude Research" else decision_brief.OFFLINE_MODE
    if st.button("Generate Decision Brief", key="rwf-brief-go", type="primary"):
        # The SAME real, already-refreshed Opportunity snapshot Agent's own
        # "Refresh Opportunities" writes (section 3B) -- read, never fetched
        # here; a session that never clicked Refresh Opportunities still
        # renders an honest "not loaded" CURRENT MARKET section.
        opportunity_snapshot = st.session_state.get(services.OPPORTUNITIES_SNAPSHOT_KEY)
        evidence = decision_brief.build_decision_brief_evidence(
            exp_id, hypothesis=h, opportunity_snapshot=opportunity_snapshot,
        )
        text, error = decision_brief.generate_decision_brief(evidence, mode=mode)
        st.session_state["rwf_brief"] = {"text": text, "error": error}
        st.rerun()

    brief = st.session_state.get("rwf_brief")
    if brief:
        if brief["error"]:
            st.error(brief["error"])
        else:
            with components.card("rwf-brief"):
                st.markdown(brief["text"])

    if exp_id and st.button("View Research Details", key="rwf-view-details"):
        st.session_state["research_details_target"] = {
            "source": "research_workflow", "objective": None, "root": detail["experiment"]["root_symbol"],
            "hypothesis": h, "compiled": st.session_state.get("rwf_compiled"), "evidence": None,
            "run_outcome": st.session_state.get("rwf_run_outcome"), "experiment_id": exp_id,
        }
        st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _stepper_strip(current_index: int) -> None:
    cols = st.columns(len(_STEPS))
    for i, (col, label) in enumerate(zip(cols, _STEPS)):
        state = "OK" if i < current_index else ("WARN" if i == current_index else "NOT_AVAILABLE")
        with col:
            st.markdown(
                f'<div style="text-align:center;font-size:0.72rem;color:{palette.state_color(state)};">'
                f'{i + 1}. {label}</div>', unsafe_allow_html=True,
            )


def _current_step_index() -> int:
    if st.session_state.get("rwf_brief"):
        return 5
    if st.session_state.get("rwf_run_outcome") is not None:
        return 4
    if st.session_state.get("rwf_compiled"):
        return 3
    if st.session_state.get("rwf_hypothesis"):
        return 2
    return 1


def render_body(root: str) -> None:
    components.section_header(
        "Research Workflow",
        "The full research golden path -- Discover a mechanism, generate a hypothesis, compile it, run "
        "the real C++ Quant Core, validate, and ask Claude for a Decision Brief.",
    )
    universe = services.approved_universe()
    if st.session_state.get("rwf_compiled"):
        root = st.session_state["rwf_compiled"]["root_symbol"] or root
    selected_root = st.selectbox("Market", list(universe), index=list(universe).index(root) if root in universe else 0,
                                  key="rwf-root")
    if st.button("Start a new research thread", key="rwf-reset"):
        _reset_thread()
        st.rerun()

    _stepper_strip(_current_step_index())
    tabs = st.tabs([f"{i + 1}. {label}" for i, label in enumerate(_STEPS)])
    with tabs[0]:
        _render_discover(selected_root)
    with tabs[1]:
        _render_hypothesis_step(selected_root)
    with tabs[2]:
        _render_compile_step(selected_root)
    with tabs[3]:
        _render_cpp_test_step()
    with tabs[4]:
        _render_validate_step()
    with tabs[5]:
        _render_decision_brief_step()
