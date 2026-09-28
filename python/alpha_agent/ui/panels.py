"""Phase 20.1 -- shared right-rail panels: the AI Research Agent mini-panel,
the Recent Activity feed, and the System Status card. Used by both the
Overview and Research pages (agent panel) and Overview / System pages (status
card) so they render identically everywhere.

The activity feed is a real, session-local log of actions THIS session
actually took (`log_activity`) plus the registry's own real import provenance
-- never a fabricated "autonomous activity" stream.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.recommendation import (
    HoldingPeriod,
    InvestorProfile,
    MaxDrawdown,
    OvernightPreference,
    RiskStyle,
    StrategyPreference,
    TradingFrequency,
)
from alpha_agent.schemas.hypothesis import HypothesisSpec
from alpha_agent.ui import components, llm_demo, services

_ACTIVITY_KEY = "activity_log"
_PROFILE_KEY = "investor_profile"
_PROFILE_EDIT_KEY = "investor_profile_editing"


def log_activity(icon: str, title: str, detail: str) -> None:
    st.session_state.setdefault(_ACTIVITY_KEY, [])
    st.session_state[_ACTIVITY_KEY].insert(
        0, {"icon": icon, "title": title, "detail": detail, "when": components.now_utc_str()}
    )
    st.session_state[_ACTIVITY_KEY] = st.session_state[_ACTIVITY_KEY][:20]


def _seed_activity_once() -> None:
    if st.session_state.get("_activity_seeded"):
        return
    st.session_state["_activity_seeded"] = True
    st.session_state.setdefault(_ACTIVITY_KEY, [])
    try:
        summary = services.registry_summary()
        imports = sorted(summary.get("imports", ()), key=lambda i: i.get("created_at", ""), reverse=True)
        if imports:
            latest = imports[0]
            log_activity(
                "OK", "Registry loaded",
                f"phase {latest.get('phase', '?')} import, {summary['authoritative_statistical_hypotheses']} "
                "authoritative hypotheses on disk",
            )
    except Exception:  # noqa: BLE001, S110 -- activity seeding is best-effort, never fails the page
        pass
    log_activity("OK", "Session started", "Read-only registry + report access initialised")


def render_activity_feed(limit: int = 6) -> None:
    _seed_activity_once()
    components.section_header("Recent Activity")
    with components.card("activity"):
        for row in st.session_state.get(_ACTIVITY_KEY, [])[:limit]:
            components.activity_row(row["icon"], row["title"], row["detail"], row["when"])


def render_system_status(*, compact: bool = True) -> None:
    engine = services.engine_provenance()
    catalog = services.catalog_summary()
    registry_ok = True
    try:
        services.registry_summary()
    except Exception:  # noqa: BLE001 -- status card is best-effort, never fails the page
        registry_ok = False
    validation_ok = bool(services.available_validation_reports())

    components.section_header("System Status")
    with components.card("system-status"):
        cli_path = engine.get("reference_cli_path")
        components.status_row("C++ Quant Core", "OK" if engine["reference_cli_built"] else "OFFLINE",
                               f"reference CLI: {cli_path.rsplit('/', 1)[-1]}" if cli_path else "reference CLI not built")
        components.status_row("Python / pybind Boundary",
                               "OK" if engine["pybind_fast_boundary_available"] else "OFFLINE",
                               "fast boundary loaded" if engine["pybind_fast_boundary_available"]
                               else "optional; CLI reference path in use")
        components.status_row("Databento Data", "OK" if catalog["n_rows"] > 0 else "OFFLINE",
                               f"{catalog['n_rows']} catalog entries")
        components.status_row("Experiment Registry", "OK" if registry_ok else "OFFLINE",
                               "data/registry/experiments.sqlite")
        components.status_row("Validation Engine", "OK" if validation_ok else "OFFLINE",
                               f"{len(services.available_validation_reports())} committed reports")


def render_agent_runtime_status() -> None:
    """Agent/LLM runtime status -- moved here from the Agent/Research pages
    (product refactor, section 12): Agent keeps only a compact status strip,
    System owns the full infrastructure detail. Claude configuration is
    reported as a boolean presence check ONLY (`ANTHROPIC_API_KEY` in the
    process environment) -- this never reads, parses, or displays the key
    value itself (CLAUDE.md: never store/echo API key material)."""
    import os

    claude_configured = bool(os.environ.get("ANTHROPIC_API_KEY"))
    components.section_header("Agent Runtime")
    with components.card("system-agent-runtime"):
        components.status_row("Research Agent", "OK", "alpha_agent.agents.research_agent.ResearchAgent")
        components.status_row(
            "Strategy Compiler Agent", "OK", "alpha_agent.agents.compiler_agent.StrategyCompilerAgent"
        )
        components.status_row(
            "Execution Validation Service", "OK",
            "alpha_agent.agents.execution_service.ProductionExecutionValidationService",
        )
        components.status_row(
            "Claude API", "OK" if claude_configured else "OFFLINE",
            "ANTHROPIC_API_KEY present in environment (value never read/displayed by this app)"
            if claude_configured else "ANTHROPIC_API_KEY not set -- Claude Research mode will fail honestly",
        )


def render_agent_panel(*, key_prefix: str = "panel") -> None:
    """The compact two-tab (Research / Compile) AI agent panel."""
    components.section_header("AI Research Agent", )
    st.markdown(components.header_pill("Online", "OK"), unsafe_allow_html=True)

    with components.card(f"{key_prefix}-agent"):
        tab_research, tab_compile = st.tabs(["Research", "Compile"])

        with tab_research:
            _render_research_tab(key_prefix)
        with tab_compile:
            _render_compile_tab(key_prefix)


def _render_research_tab(key_prefix: str) -> None:
    scenario_options = {s.label: s.key for s in llm_demo.SCENARIOS}
    chosen_label = st.selectbox("Research objective", list(scenario_options.keys()),
                                 key=f"{key_prefix}-scenario", label_visibility="collapsed")
    scenario_key = scenario_options[chosen_label]
    sc = llm_demo.scenario(scenario_key)
    objective = st.text_area("objective", value=sc.objective, height=68, key=f"{key_prefix}-objective",
                              label_visibility="collapsed")

    if st.button("Generate Hypothesis", type="primary", key=f"{key_prefix}-propose",
                 width="stretch"):
        universe = services.approved_universe()
        proposal, error, _fm = llm_demo.propose_hypothesis(
            mode=llm_demo.SCRIPTED_MODE, scenario_key=scenario_key, universe=universe,
            objective=objective,
        )
        st.session_state["ra_scenario_key"] = scenario_key
        st.session_state["ra_proposal_error"] = error
        st.session_state["ra_proposal"] = proposal.model_dump(mode="json") if proposal else None
        if proposal and proposal.accepted:
            log_activity("OK", "Hypothesis generated", proposal.hypothesis.title)
        elif error:
            log_activity("WARN", "Hypothesis proposal failed", error)

    error = st.session_state.get("ra_proposal_error")
    proposal = st.session_state.get("ra_proposal")
    if error:
        st.error(error)
    elif proposal and proposal.get("accepted"):
        h = proposal["hypothesis"]
        st.markdown(f"**{h['title']}**")
        st.caption(h["economic_mechanism"])
        components.render_tags(h["required_features"])
        with st.expander("Full hypothesis"):
            st.write(f"**Expected regime:** {h['expected_regime']}")
            st.write(f"**Failure regime:** {h['failure_regime']}")
            st.write(f"**Falsification test:** {h['falsification_test']}")


def _render_compile_tab(key_prefix: str) -> None:
    proposal = st.session_state.get("ra_proposal")
    scenario_key = st.session_state.get("ra_scenario_key")
    if not proposal or not proposal.get("accepted"):
        st.caption("Generate a hypothesis in the Research tab first.")
        return
    hypothesis = HypothesisSpec.model_validate(proposal["hypothesis"])
    st.markdown(f"Compiling: **{hypothesis.title}**")
    if st.button("Compile Strategy", type="primary", key=f"{key_prefix}-compile", width="stretch"):
        universe = services.approved_universe()
        compiled, error = llm_demo.compile_hypothesis(
            mode=llm_demo.SCRIPTED_MODE, scenario_key=scenario_key, hypothesis=hypothesis, universe=universe
        )
        st.session_state["lab_compiled_error"] = error
        st.session_state["lab_compiled"] = compiled.model_dump(mode="json") if compiled else None
        if compiled and compiled.accepted:
            log_activity("OK", "Strategy compiled", f"{compiled.build_mode} -> {compiled.strategy_fingerprint[:24]}...")
        elif error:
            log_activity("WARN", "Strategy compilation failed", error)

    compiled = st.session_state.get("lab_compiled")
    error = st.session_state.get("lab_compiled_error")
    if error:
        st.error(error)
    elif compiled and compiled.get("accepted"):
        st.success(f"Build mode: {compiled['build_mode']}")
        st.code(compiled["strategy_fingerprint"], language="text")
        fc = compiled["feature_coverage"]
        if fc["missing"] or fc["unapproved"]:
            st.error(f"Missing: {fc['missing']} / Unapproved: {fc['unapproved']}")
        else:
            st.caption("Feature fidelity: all required kinds represented.")


# ---------------------------------------------------------------------------
# Phase B1 -- investor profile panel (compact summary + inline editor)
#
# Presentation / research-prioritization state only (CLAUDE.md, task spec
# sections 4/5/12/20): loaded from `services.load_investor_profile` (a local
# JSON preference file -- never the ExperimentRegistry) and cached in
# `st.session_state` for the rest of the session. Never a full-screen
# onboarding wizard: a compact summary renders every time, and "Edit Profile"
# opens an inline form in place -- see the module docstring's "shared
# right-rail panels" framing, reused here for a shared cross-page profile
# widget.
# ---------------------------------------------------------------------------


def current_investor_profile() -> InvestorProfile:
    """The active `InvestorProfile` for this session: loaded from the local
    preference file once, then cached in `st.session_state` so every page
    within one session sees the same profile (including one just saved by
    `_save_profile_form`, without a second file read)."""
    if _PROFILE_KEY not in st.session_state:
        st.session_state[_PROFILE_KEY] = services.load_investor_profile()
    return st.session_state[_PROFILE_KEY]


def _save_profile_form(profile: InvestorProfile) -> None:
    services.save_investor_profile(profile)
    st.session_state[_PROFILE_KEY] = profile
    st.session_state[_PROFILE_EDIT_KEY] = False
    log_activity("OK", "Research preferences updated", profile.risk_style.value)


def render_profile_panel(*, key_prefix: str = "profile") -> None:
    """Compact "YOUR PROFILE" summary (task spec section 4/20's mockup) with
    an "Edit Profile" toggle. Never presented as personalized advice when it
    is still the untouched default -- see the caption below. Deliberately
    just one small card, not a full-screen wizard (task spec section 20)."""
    profile = current_investor_profile()
    #: A saved preference file is the only signal this is a real, chosen
    #: profile -- never inferred from the field values themselves (a user
    #: could deliberately choose values that happen to equal the defaults).
    is_default = not services.has_saved_investor_profile()

    with components.card(f"{key_prefix}-profile-summary"):
        st.markdown(f"**{profile.risk_style.value}**")
        st.caption(
            f"{profile.holding_period.value} holding &middot; "
            f"Max comfortable drawdown {profile.max_drawdown.value} &middot; "
            f"Overnight: {profile.overnight.value}"
        )
        st.caption(
            f"Trading frequency: {profile.trading_frequency.value} &middot; "
            f"Strategy preference: {profile.strategy_preference.value}",
            unsafe_allow_html=True,
        )
        if is_default:
            st.caption("Using default preferences -- not yet personalized.")
        editing = st.session_state.get(_PROFILE_EDIT_KEY, False)
        if st.button("Edit Profile" if not editing else "Close", key=f"{key_prefix}-profile-edit-toggle"):
            st.session_state[_PROFILE_EDIT_KEY] = not editing
            st.rerun()

    if st.session_state.get(_PROFILE_EDIT_KEY, False):
        _render_profile_editor(profile, key_prefix)


def _render_profile_editor(profile: InvestorProfile, key_prefix: str) -> None:
    with components.card(f"{key_prefix}-profile-editor"), st.form(key=f"{key_prefix}-profile-form"):
        st.caption("Research / risk preferences -- affects ranking and recommendation only, never scientific results.")
        holding_period = st.selectbox(
            "Preferred Holding Period", list(HoldingPeriod), index=list(HoldingPeriod).index(profile.holding_period),
            format_func=lambda v: v.value,
        )
        risk_style = st.selectbox(
            "Risk Style", list(RiskStyle), index=list(RiskStyle).index(profile.risk_style),
            format_func=lambda v: v.value,
        )
        max_drawdown = st.selectbox(
            "Maximum Comfortable Drawdown", list(MaxDrawdown), index=list(MaxDrawdown).index(profile.max_drawdown),
            format_func=lambda v: v.value,
        )
        trading_frequency = st.selectbox(
            "Trading Frequency Preference", list(TradingFrequency),
            index=list(TradingFrequency).index(profile.trading_frequency), format_func=lambda v: v.value,
        )
        overnight = st.selectbox(
            "Overnight Positions", list(OvernightPreference),
            index=list(OvernightPreference).index(profile.overnight), format_func=lambda v: v.value,
        )
        strategy_preference = st.selectbox(
            "Strategy Preference", list(StrategyPreference),
            index=list(StrategyPreference).index(profile.strategy_preference), format_func=lambda v: v.value,
        )
        saved = st.form_submit_button("Save Profile", type="primary")
        if saved:
            new_profile = InvestorProfile(
                holding_period=holding_period, risk_style=risk_style, max_drawdown=max_drawdown,
                trading_frequency=trading_frequency, overnight=overnight, strategy_preference=strategy_preference,
                approximate_capital_usd=profile.approximate_capital_usd, max_contracts=profile.max_contracts,
                turnover_sensitivity=profile.turnover_sensitivity,
            )
            _save_profile_form(new_profile)
            st.rerun()
