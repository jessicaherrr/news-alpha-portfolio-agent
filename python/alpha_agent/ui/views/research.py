"""Research -- the consolidated primary destination (Product Consolidation +
Opportunity V1 campaign, Checkpoint D): "WHAT DOES THE EVIDENCE SAY?"

Strategy Lab (url `lab`; the former Research page). Two modes, both on this one page/route:

* LANDING (no explicit selection) -- Phase 8's Research UI consolidation
  (prompt 8 section 10): FIVE primary internal views in one `st.tabs` --
  Workflow (the golden-path stepper -- Discover / Hypothesis / Compile /
  C++ Test / Validate / Decision Brief, `views/research_workflow.py`),
  My Alpha (Phase 2 personal research memory, evolved by the Phase 6
  closure into a Factor-first, evidence-oriented
  (VALIDATED/UNDER_RESEARCH/RESEARCH_ARCHIVE) surface, renamed from "Factor
  Library" this phase -- same tab key, same module,
  `views/alpha_library.py::render_body`), Research Map (Phase 7's
  Cross-Asset Alpha Graph, `views/alpha_graph.py::render_body`), Community
  (Phase 8's Community Alpha Network -- reproducible research and
  independent replication, `views/community.py::render_body`), and
  Provenance (the audit trail -- `_render_provenance_tab` below).
  Strategies / Experiments / Validation are NOT equal top-level tabs
  anymore (prompt 8 section 10: "simplify navigation, not research
  capability") -- they remain fully intact, unmodified modules, reachable
  as deeper drill-down views via `_render_more_detail_views` below (same
  `research_landing_focus` deep-link mechanism the sidebar sub-nav already
  used, so nothing that could reach them before has stopped working).

  Research Golden Path campaign: Workflow is a SECOND explicit, user-
  triggered execution surface alongside Agent's "Generate Hypothesis" /
  "Run This Hypothesis" -- both call the exact same single boundary
  (`alpha_agent.ui.deep_research.run_deep_research`), never a duplicated
  orchestration path. This page is no longer execution-free; it is now
  ALSO where a user can run the complete research loop without leaving
  Research, while Agent remains the conversational entry point.
* DETAIL (an explicit selection exists) -- the read-only, tabbed inspector
  over ONE research artifact this page has always been: an Agent
  conversation's proposal/compilation/execution, or an arbitrary registry
  experiment looked up by id. "Clicking a strategy opens a detail view"
  (task spec section 17) -- Strategies' own "Open strategy detail" control,
  Market's "View Research", and Agent's "View Research Details" all set
  `st.session_state["research_details_target"]` and land here; a "Back to
  Research" action clears it and returns to the landing. This is the ONLY
  selection source now (a prior "fall back to the latest authoritative
  registry experiment when nothing is selected" behaviour is retired: the
  Strategies-first landing IS the "nothing is selected yet" state now,
  never an auto-picked Detail view).

Workflow aside, the DETAIL view below remains purely an inspector: it never
proposes a hypothesis or triggers execution itself, only renders what the
Workflow tab (or Agent) already produced.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.agents.orchestrator import FamilyReport, FamilyStatus
from alpha_agent.registry.enums import RegistryVerdict
from alpha_agent.ui import (
    charts,
    components,
    layout,
    palette,
    panels,
    recommendation_views,
    services,
)
from alpha_agent.ui.charts import ci_range_chart, cost_stress_line, regime_bar
from alpha_agent.ui.views import (
    alpha_graph,
    alpha_library,
    community,
    experiment_log,
    research_workflow,
    strategies,
    validation,
)

PAGE_TITLE = "Strategy Lab"

_TABS = (
    "Overview", "Hypothesis", "Performance", "Signals", "Validation", "Sources", "Provenance",
    "Memory & Lineage", "Raw / Logs",
)

#: The frozen policy's own reason codes for "a required evidence category was
#: never evaluated" (validation-safety fix, df861c1). Duplicated as plain
#: strings, not imported, to keep this presentation module import-light and
#: because it is not a registry-semantics authority -- mirrors
#: `alpha_agent.paper.eligibility._INCOMPLETE_EVIDENCE_CODES`.
_INCOMPLETE_EVIDENCE_CODES = frozenset(
    {"parameter_stability_not_evaluated", "ablation_not_evaluated"}
)


def _render_family_status_banner(report: FamilyReport) -> None:
    """EXECUTION COMPLETED must never be painted as SCIENTIFIC SUCCESS
    (df861c1 / validation-safety fix): the banner's tone comes ONLY from the
    actual committed verdict(s)/reason codes, never from "the orchestrator
    finished without raising" alone. `st.success` (green) is reserved for a
    genuine authoritative PASS; REJECT is a normal, successful system outcome
    (never an error) but gets a neutral tone, not green; incomplete required
    evidence and INCONCLUSIVE get a warning tone; an unresolved
    INVALID_EXECUTION member (INCOMPLETE_NOT_ADJUDICATED) gets an error
    tone -- it is an engineering defect, not a scientific verdict."""
    if report.status is FamilyStatus.INCOMPLETE_NOT_ADJUDICATED:
        st.error(
            "EXECUTION INCOMPLETE -- an unresolved engineering/execution defect left "
            f"this family not adjudicated (family status: {report.status.value}). "
            "This is NOT a scientific verdict, and nothing was written to the registry."
        )
        return
    if report.status is not FamilyStatus.FINALIZED:
        st.info(f"Family status: {report.status.value} -- no scientific verdict yet.")
        return

    verdicts = {mr.final_verdict for mr in report.member_results if mr.final_verdict is not None}
    incomplete_evidence = any(
        _INCOMPLETE_EVIDENCE_CODES.intersection(mr.reason_codes) for mr in report.member_results
    )
    if incomplete_evidence:
        st.warning(
            "EXECUTION COMPLETE -- PARTIAL SCIENTIFIC EVIDENCE: a required evidence "
            "category (parameter stability / ablation) was not evaluated, so this "
            "cannot be an authoritative PASS regardless of the other gates."
        )
    elif verdicts and verdicts == {RegistryVerdict.PASS}:
        st.success(
            f"EXECUTION COMPLETE -- authoritative SCIENTIFIC PASS (family status: "
            f"{report.status.value})."
        )
    elif RegistryVerdict.REJECT in verdicts:
        st.info(
            f"EXECUTION COMPLETE -- scientific verdict: REJECT (family status: "
            f"{report.status.value}). REJECT is a normal, successful system outcome, "
            "not an error."
        )
    else:
        st.warning(
            f"EXECUTION COMPLETE -- scientific verdict: INCONCLUSIVE (family status: "
            f"{report.status.value})."
        )


def _selection_experiment_id(selection: dict) -> str | None:
    """The one experiment id this selection is actually about, preferring
    the FRESHEST evidence: a just-executed run's own member result, then a
    prior-registry-evidence lookup, then an explicit lookup-by-id."""
    outcome = selection.get("run_outcome")
    if outcome is not None and getattr(outcome, "accepted", False) and outcome.report and outcome.report.member_results:
        return outcome.report.member_results[0].experiment_id
    evidence = selection.get("evidence")
    if evidence and evidence.get("result"):
        return evidence["experiment"]["experiment_id"]
    return selection.get("experiment_id")


def _selection_detail(selection: dict) -> dict | None:
    """The full `services.get_experiment(...)` record for this selection, or
    `None` if there is nothing committed to look up yet -- never fabricated."""
    exp_id = _selection_experiment_id(selection)
    if not exp_id:
        return None
    try:
        return services.get_experiment(exp_id)
    except Exception:  # noqa: BLE001 -- an unknown/stale id is an honest "not available", not a crash
        return None


def _execution_status_label(selection: dict) -> str:
    """VALID execution is never the same as scientific PASS (df861c1's UI-
    honesty rules, preserved here): this label is purely about whether an
    attempt ran cleanly, never about the scientific outcome."""
    outcome = selection.get("run_outcome")
    if outcome is None:
        return "NOT EXECUTED (proposal only)" if selection.get("compiled") else "NOT AVAILABLE"
    if not outcome.accepted:
        return "EXECUTION FAILED"
    if outcome.report and outcome.report.member_results:
        return outcome.report.member_results[0].attempt_status.value
    if outcome.report is not None and outcome.report.status.value == "INCOMPLETE_NOT_ADJUDICATED":
        return "INVALID_EXECUTION"
    return "NOT AVAILABLE"


def _render_lookup_expander(selection: dict | None) -> None:
    """Compact experiment picker: a one-line "Experiment / [selected]"
    readout, with the raw experiment-ID search form tucked into a collapsed
    expander below it -- not a large, always-open lookup form."""
    exp_id = _selection_experiment_id(selection) if selection else None
    st.markdown(
        f'<div style="margin-top:0.2rem;font-size:0.86rem;color:var(--aa-text-secondary);">'
        f'<b>Experiment</b><br>{f"`{exp_id}`" if exp_id else "No experiment selected yet."}'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Search by experiment ID", expanded=selection is None):
        st.caption(
            "Inspect any committed registry experiment directly -- this does not require an Agent "
            "conversation to have produced it."
        )
        exp_id_input = st.text_input("Experiment ID", key="research-details-lookup")
        if st.button("Load", key="research-details-lookup-go") and exp_id_input.strip():
            try:
                services.get_experiment(exp_id_input.strip())
            except Exception as exc:  # noqa: BLE001 -- honest lookup failure, never a crash
                st.error(f"No experiment found for {exp_id_input!r}: {exc}")
            else:
                st.session_state["research_details_target"] = {
                    "source": "registry_lookup",
                    "objective": None, "root": None, "hypothesis": None, "compiled": None,
                    "evidence": None, "run_outcome": None,
                    "experiment_id": exp_id_input.strip(),
                }
                st.rerun()


def render() -> None:
    layout.inject_style()
    focus = st.session_state.get("research_landing_focus")
    layout.render_sidebar_nav(active="lab", active_sub=focus)
    layout.render_header(subtitle="Single-strategy research -- hypothesis, C++ backtest, validation and the full "
                                  "experiment record.")

    selection = st.session_state.get("research_details_target")
    if selection:
        _render_detail_view(selection)
    else:
        _render_landing(services.approved_universe()[0], focus=focus)

    layout.render_disclaimer()


# ---------------------------------------------------------------------------
# Research Scope -- moved here from the shared sidebar (task spec section
# 8/8A): this is scientific Research context, not global navigation, and
# belongs visually next to Strategies/Experiments/Validation/Provenance, not
# competing with page-to-page wayfinding. Dates/holdout policy unchanged --
# see `services.research_window`/`holdout_status`, the exact same read this
# block always used.
# ---------------------------------------------------------------------------


def _render_research_scope() -> None:
    w = services.research_window()
    with components.card("research-scope"):
        st.markdown('<div class="aa-gate-title">Research Scope</div>', unsafe_allow_html=True)
        with components.metric_row("research-scope"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Research", w["research"])
            c2.metric("Validation", w["validation"])
            with c3:
                st.markdown(
                    '<div class="aa-metric-label">Holdout</div>'
                    f'<div class="aa-metric-value" style="color:{palette.GREEN}">2025 &middot; LOCKED</div>',
                    unsafe_allow_html=True,
                )
        st.caption("No 2025 data is used in research. Deeper holdout-lifecycle/audit detail: Settings -> Runtime.")


# ---------------------------------------------------------------------------
# LANDING -- Strategies / Experiments / Validation / Provenance
# ---------------------------------------------------------------------------

#: Phase 8 Research UI consolidation (prompt section 10): five primary
#: top-level tabs, "Prefer: Workflow / My Alpha / Research Map / Community /
#: Provenance". Strategies/Experiments/Validation are deliberately absent --
#: they remain full, unmodified detail/drill-down views, one click away via
#: `_render_more_detail_views`, never deleted or weakened.
_LANDING_TABS = ("Workflow", "Provenance")
#: Every view `research_landing_focus` can deep-link to -- the five primary
#: tabs above PLUS the three drill-down-only views. A focus value is set
#: either by `_render_more_detail_views`' own buttons below or (legacy,
#: still honored) a sidebar sub-nav row / any other in-app hand-off.
_LANDING_FOCUS_LABEL = {"workflow": "Workflow", "alpha_library": "My Alpha",
                         "alpha_graph": "Research Map", "community": "Community",
                         "provenance": "Provenance", "strategies": "Strategies",
                         "experiments": "Experiments", "validation": "Validation"}
#: The three drill-down-only views (prompt section 10) -- rendered as a
#: compact "More detail views" row on the landing page rather than as
#: top-level tabs or sidebar sub-nav rows. Order matches their historical
#: tab order.
_MORE_DETAIL_VIEWS = (("strategies", "Strategies"), ("experiments", "Experiments"), ("validation", "Validation"))


def _render_landing(root: str, *, focus: str | None = None) -> None:
    st.markdown("## Strategy Lab")
    st.caption(
        "Test one strategy hypothesis end to end -- propose, compile, run the C++ backtest, validate -- and inspect "
        "the full experiment record behind it. Your personal research memory, the Research Map and Community now "
        "live under Learn."
    )
    _render_research_scope()
    _render_lookup_expander(None)

    # Sidebar IA pass (task spec section 7C): the shared sidebar's Research
    # -> {Strategies,Experiments,Validation,Provenance} sub-rows deep-link
    # here via `st.session_state["research_landing_focus"]` -- `st.tabs` has
    # no supported way to be preselected programmatically, so a focused view
    # renders that ONE internal view directly (with a "back to all" control)
    # instead of the tab strip; the unfocused default path below is
    # byte-for-byte unchanged.
    if focus in _LANDING_FOCUS_LABEL:
        if st.button("< Back to all views", key="research-focus-back"):
            st.session_state.pop("research_landing_focus", None)
            st.rerun()
        st.markdown(f'<div class="aa-gate-title">{_LANDING_FOCUS_LABEL[focus]}</div>', unsafe_allow_html=True)
        {
            "workflow": lambda: research_workflow.render_body(root),
            "strategies": lambda: strategies.render_body(root),
            "experiments": experiment_log.render_body,
            "validation": lambda: validation.render_body(root),
            "alpha_library": lambda: alpha_library.render_body(root),
            "alpha_graph": lambda: alpha_graph.render_body(root),
            "community": lambda: community.render_body(root),
            "provenance": _render_landing_provenance_tab,
        }[focus]()
        return

    tabs = st.tabs(list(_LANDING_TABS))
    with tabs[0]:
        research_workflow.render_body(root)
    with tabs[1]:
        _render_landing_provenance_tab()

    _render_more_detail_views()


def _render_more_detail_views() -> None:
    """Prompt 8 section 10: "Strategies, Experiments, and Validation should
    remain available as deeper detail/drill-down views rather than equal
    top-level Research tabs." Rendered once, below the primary five tabs
    (not inside any one of them) -- clicking a button here reuses the exact
    same `research_landing_focus` deep-link mechanism the sidebar sub-nav
    already used, so nothing that could reach these views before has
    stopped working; they are simply no longer competing with the five
    primary tabs above for a reader's first click."""
    with st.expander("More detail views: Strategies / Experiments / Validation"):
        st.caption(
            "Full, unmodified technical detail -- the same views this page has always had, one click "
            "further from the primary tabs above."
        )
        cols = st.columns(len(_MORE_DETAIL_VIEWS))
        for col, (focus_key, label) in zip(cols, _MORE_DETAIL_VIEWS, strict=True):
            with col:
                if st.button(label, key=f"research-more-detail-{focus_key}", width="stretch"):
                    st.session_state["research_landing_focus"] = focus_key
                    st.rerun()


def _render_landing_provenance_tab() -> None:
    """The deeper audit trail (task spec section 21): data provenance,
    experiment lineage, and artifact hashes, kept off the Research landing's
    primary surface (Strategies) but still one click away. Academic/GitHub/
    practitioner source citations stay per-hypothesis, on that hypothesis's
    own Detail view Sources tab (`_render_sources_tab`) -- this tab is the
    registry-/dataset-wide provenance, never a duplicate research-knowledge
    browser."""
    components.section_header(
        "Provenance",
        "Registry-wide audit trail -- data catalog, import lineage, and runtime provenance. Per-hypothesis "
        "research sources are on that strategy's own Detail view.",
    )
    catalog = services.catalog_summary()
    git = services.git_provenance()
    engine = services.engine_provenance()
    reg = services.registry_summary()
    with components.card("rd-landing-prov-data"):
        st.markdown('<div class="aa-gate-title">Data</div>', unsafe_allow_html=True)
        components.provenance_row("Dataset", "Databento GLBX.MDP3")
        components.provenance_row("Roots", ", ".join(catalog["roots"]))
        components.provenance_row("Catalog entries", str(catalog["n_rows"]))
    with components.card("rd-landing-prov-runtime"):
        st.markdown('<div class="aa-gate-title">Runtime</div>', unsafe_allow_html=True)
        components.provenance_row("Git commit", git.get("commit"))
        components.provenance_row("Branch", git.get("branch"))
        components.provenance_row("Reference C++ CLI", engine.get("reference_cli_path") or "not built")
        components.provenance_row("Registry schema version", str(reg["schema_version"]))
        components.provenance_row("Registry content digest", reg["content_digest"])
    with components.card("rd-landing-prov-imports"):
        st.markdown('<div class="aa-gate-title">Import Lineage</div>', unsafe_allow_html=True)
        st.caption("Append-only provenance of every batch write to the registry.")
        st.json(reg["imports"])


# ---------------------------------------------------------------------------
# DETAIL -- one selected research artifact
# ---------------------------------------------------------------------------


def _render_detail_view(selection: dict) -> None:
    if st.button("← Back to Strategy Lab", key="rd-back-to-landing"):
        st.session_state.pop("research_details_target", None)
        st.rerun()

    st.markdown("## Research Details")
    st.caption("Scientific evidence, execution provenance, and experiment lineage.")

    _render_lookup_expander(selection)
    selection = st.session_state.get("research_details_target") or selection

    _render_viewing_experiment_banner(selection)
    _render_research_classification(selection)
    _render_verdict_promise_fit(selection)

    tabs = st.tabs(list(_TABS))
    with tabs[0]:
        _render_overview_tab(selection)
    with tabs[1]:
        _render_hypothesis_tab(selection)
    with tabs[2]:
        _render_backtest_tab(selection)
    with tabs[3]:
        _render_signals_tab(selection)
    with tabs[4]:
        _render_validation_tab(selection)
    with tabs[5]:
        _render_sources_tab(selection)
    with tabs[6]:
        _render_provenance_tab(selection)
    with tabs[7]:
        _render_memory_tab(selection)
    with tabs[8]:
        _render_raw_tab(selection)


# ---------------------------------------------------------------------------
# Viewing Experiment -- root/family/verdict/id readout, ALWAYS visible above
# the tabs, and ALWAYS derived from the selected experiment itself
# ---------------------------------------------------------------------------


def _selection_root_family(selection: dict) -> tuple[str | None, str | None]:
    """The (root, family) this selection is ACTUALLY about.

    There are three concepts on this page that can legitimately differ:
    Market's own local selected product (`market_selected_root`), the Agent
    page's "Research Root" selector, and the root of whichever experiment
    this page is showing. This selection's own root/family is authoritative
    for Research Details and is never mutated to match Market's selection --
    doing so would silently swap which experiment's evidence a gate/tab
    reports on.
    Shared by the "Viewing Experiment" banner and every tab below so none of
    them can disagree with each other."""
    compiled = selection.get("compiled")
    root = selection.get("root") or (compiled or {}).get("root_symbol")
    family = (compiled or {}).get("family_key") or selection.get("family")
    return root, family


def _render_viewing_experiment_banner(selection: dict) -> None:
    """Prominent, always-visible-above-the-tabs summary of what this page is
    ACTUALLY showing -- so a reader is never left guessing whether "Root"
    here means the sidebar's market browser or the selected experiment.
    Release UX (investor-readability pass): leads with Market / Strategy /
    Result / Performance / Why -- the same investment-facing question order
    as every other page -- with the technical identity fields (Experiment ID)
    kept, just after the plain-language summary."""
    root, family = _selection_root_family(selection)
    detail = _selection_detail(selection)
    result = detail["result"] if detail else None
    verdict = result["headline_verdict"] if result else None
    exp_id = _selection_experiment_id(selection)
    exp = detail["experiment"] if detail else None
    paper_ok = services.paper_eligible_experiment(exp_id) if exp_id else False

    with components.card("rd-viewing-experiment"):
        market_label = f"{root} {services.market_name(root)} Futures" if root else "--"
        st.markdown(
            '<div class="aa-metric-label">VIEWING EXPERIMENT</div>'
            f'<div class="aa-gate-title">{market_label}</div>'
            f'<div style="font-size:0.95rem;font-weight:700;color:var(--aa-text);margin-bottom:0.3rem;">'
            f'{services.strategy_name(family) if family else "No strategy selected"}</div>',
            unsafe_allow_html=True,
        )
        with components.metric_row("rd-viewing-experiment"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                components.verdict_metric_card("rd-view-verdict", "Result", verdict)
            with c2:
                components.metric_card(
                    "rd-view-pnl", "Net PnL",
                    f"${result['net_pnl_usd']:,.0f}" if result and result.get("net_pnl_usd") is not None else "N/A",
                )
            with c3:
                components.metric_card(
                    "rd-view-sharpe", "Sharpe",
                    f"{result['annualized_sharpe']:.2f}" if result and result.get("annualized_sharpe") is not None else "N/A",
                )
            with c4:
                components.metric_card(
                    "rd-view-paper", "Paper Eligible", "YES" if paper_ok else "NO",
                    accent=palette.GREEN if paper_ok else palette.RED,
                )
        # Section 12: "Test in Paper" -- enabled ONLY when the authoritative,
        # registry-derived `services.paper_eligible_experiment` check (the
        # SAME one `alpha_agent.paper.eligibility` computes) says YES. This
        # page never decides eligibility itself; a disabled button always
        # states the honest reason instead of silently hiding the action.
        if st.button(
            "Test in Paper", key="rd-test-in-paper", disabled=not paper_ok,
            help=None if paper_ok else "Not currently paper-eligible -- see Paper Eligible above.",
        ):
            st.session_state["pt-preselect-experiment-id"] = exp_id
            from alpha_agent.ui.views import paper_trading

            st.switch_page(st.Page(paper_trading.render, url_path="paper-trading"))
        why = services.plain_language_reason(
            result=result, trial_role=(exp["trial_role"] if exp else "CANONICAL"), verdict=verdict
        )
        st.markdown(
            f'<div style="margin-top:0.3rem;font-size:0.86rem;color:var(--aa-text-secondary);">'
            f'<b>Why:</b> {why}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Technical identity", expanded=False):
            components.provenance_row("Experiment ID", exp_id or "--")
            components.provenance_row("Root symbol (raw)", root or "--")
            components.provenance_row("Strategy family (raw)", family or "--")
        st.caption(
            "This selected experiment's own market is authoritative for Research Details -- "
            "independent of the sidebar's Market Browser."
        )


def _render_research_classification(selection: dict) -> None:
    """Research Golden Path V1 acceptance pass, section 1C: an experiment is
    either part of a frozen, predeclared research program (Phase 13.5C /
    Phase 15B) or a later standalone research-session append -- these are
    real, already-committed, mechanically-derived facts
    (`services.research_classification`), never inferred from `TrialRole`/
    `Authority` (which answer different questions -- see that function's own
    docstring). Deliberately separate from "Scientific Verdict, Research
    Promise & User Fit" below: classification/origin/holdout-eligibility are
    provenance, not a verdict, a promise score, or a personal-fit measure."""
    detail = _selection_detail(selection)
    if not detail or not detail.get("experiment"):
        return
    rc = services.research_classification(detail["experiment"])
    result = detail.get("result")
    execution_evidence = (
        f"{detail.get('authority', 'UNKNOWN')} real C++ execution" if result else "No committed execution yet"
    )
    holdout_eligible = (result or {}).get("holdout_eligible", False)
    components.section_header(
        "Research Classification",
        "Frozen scientific program vs. a later standalone research-session append -- provenance, not a verdict.",
    )
    with components.card("rd-research-classification"), components.metric_row("rd-research-classification"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            components.metric_card("rd-rc-class", "Research Classification", rc["label"])
        with c2:
            st.markdown(
                '<div class="aa-metric-label">ORIGIN</div>'
                f'<div style="font-size:0.85rem;color:var(--aa-text-secondary);">{rc["origin"]}</div>',
                unsafe_allow_html=True,
            )
        with c3:
            components.metric_card("rd-rc-exec", "Execution Evidence", execution_evidence)
        with c4:
            components.metric_card(
                "rd-rc-holdout", "2025 Holdout Eligibility",
                "ELIGIBLE" if holdout_eligible else "NOT ELIGIBLE",
                accent=palette.GREEN if holdout_eligible else palette.RED,
            )


def _render_verdict_promise_fit(selection: dict) -> None:
    """Task spec section 19 (RESEARCH DETAILS): Scientific Verdict, Research
    Promise, and User Fit near the top, clearly separated from each other and
    from the tabs below."""
    exp_id = _selection_experiment_id(selection)
    if not exp_id:
        return
    profile = panels.current_investor_profile()
    candidate = services.candidate_summary_for_experiment(exp_id, profile)
    if not candidate:
        return
    components.section_header(
        "Scientific Verdict, Research Promise & User Fit",
        "Independent dimensions: Research Promise prioritizes further research and User Fit measures "
        "personal alignment -- neither is a scientific claim, and neither can change the verdict.",
    )
    recommendation_views.render_verdict_promise_fit_row(candidate, key_prefix="rd-vpf")


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def _render_overview_tab(selection: dict) -> None:
    root, family = _selection_root_family(selection)
    detail = _selection_detail(selection)
    result = detail["result"] if detail else None
    verdict = result["headline_verdict"] if result else None
    exp_id = _selection_experiment_id(selection)

    with components.card("rd-overview"):
        components.provenance_row("Objective", selection.get("objective") or "--")
        components.provenance_row("Result origin", selection.get("source") or "--")
        components.provenance_row("Market", f"{root} {services.market_name(root)}" if root else "--")
        components.provenance_row("Strategy", services.strategy_name(family) if family else "--")
        components.provenance_row("Execution status", _execution_status_label(selection))
        components.provenance_row("Experiment ID", exp_id or "--")

    outcome = selection.get("run_outcome")
    if outcome is not None and outcome.accepted and outcome.report is not None:
        _render_family_status_banner(outcome.report)
    elif outcome is not None and not outcome.accepted:
        st.error(outcome.error)

    paper_ok = services.paper_eligible_experiment(exp_id) if exp_id else False
    with components.metric_row("rd-overview"):
        c1, c2 = st.columns(2)
        with c1:
            components.verdict_metric_card("rd-overview-verdict", "Scientific Verdict", verdict)
        with c2:
            components.metric_card(
                "rd-overview-paper", "Paper Eligible", "YES" if paper_ok else "NO",
                accent=palette.GREEN if paper_ok else palette.RED,
            )

    if not detail:
        components.committed_series_notice(
            key="rd-overview-na",
            text="No committed registry experiment is bound to this selection yet -- a proposal-only "
                 "result has no scientific evidence to show beyond the Hypothesis tab.",
        )


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


def _render_hypothesis_tab(selection: dict) -> None:
    h = selection.get("hypothesis")
    if not h:
        components.empty_state(
            "Hypothesis", "No HypothesisSpec is available for this selection.", key="rd-hyp-na",
        )
        return
    with components.card("rd-hyp"):
        st.markdown(f"### {h['title']}")
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
        if h.get("novelty_notes"):
            st.markdown("**Novelty notes**")
            st.write(h["novelty_notes"])


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def _render_backtest_tab(selection: dict) -> None:
    detail = _selection_detail(selection)
    if not detail or not detail.get("result"):
        components.empty_state(
            "Backtest", "No committed authoritative execution result for this selection yet.",
            key="rd-bt-na",
        )
        return
    result = detail["result"]
    exp = detail["experiment"]

    with components.metric_row("rd-bt"):
        cols = st.columns(6)
        with cols[0]:
            components.metric_card("rd-bt-gross", "Gross PnL",
                                    f"${result['gross_pnl_usd']:,.0f}" if result.get("gross_pnl_usd") is not None else "N/A")
        with cols[1]:
            components.metric_card("rd-bt-net", "Net PnL",
                                    f"${result['net_pnl_usd']:,.0f}" if result.get("net_pnl_usd") is not None else "N/A")
        with cols[2]:
            components.metric_card("rd-bt-costs", "Costs",
                                    f"${result['costs_usd']:,.0f}" if result.get("costs_usd") is not None else "N/A")
        with cols[3]:
            components.metric_card("rd-bt-sharpe", "Sharpe",
                                    f"{result['annualized_sharpe']:.2f}" if result.get("annualized_sharpe") is not None else "N/A")
        with cols[4]:
            components.metric_card("rd-bt-trades", "Trades", str(result.get("n_trades", "N/A")))
        with cols[5]:
            components.metric_card("rd-bt-fills", "Fills", str(result.get("n_fills", "N/A")))

    with components.card("rd-bt-detail"):
        components.provenance_row("Source artifact", result.get("source_artifact") or "--")
        components.provenance_row("Code commit", exp.get("code_commit") or "--")

    # Preferred source: a hash-verified Part I artifact bundle (real daily
    # equity trace + trades.csv) -- the SAME bundle the Signals tab already
    # binds and verifies. Only if this experiment's own `source_artifact`
    # does not name such a bundle do we fall back to the older, narrower
    # `find_experiment_bound_trade_ledger` (a direct `trades.csv` pointer --
    # no currently-committed ResultRecord uses that shape, but it is kept for
    # any future/legacy result that does). Neither path ever matches by root
    # symbol; both refuse and render NOT AVAILABLE rather than guess.
    bundle = services.find_experiment_bound_artifact_bundle(detail)
    ledger = None if bundle else services.find_experiment_bound_trade_ledger(detail)
    series = services.daily_equity_series_from_bundle(bundle) if bundle else services.trade_ledger_equity_series(ledger)
    trades_rows = services.trades_rows_from_bundle(bundle) if bundle else (ledger["trades"] if ledger else None)
    if series:
        max_dd = services.max_drawdown_usd(series)
        with components.metric_row("rd-bt-dd"):
            components.metric_card(
                "rd-bt-maxdd", "Max Drawdown", f"${max_dd:,.0f}" if max_dd is not None else "N/A",
                accent=palette.RED if max_dd else None,
            )
        c1, c2 = st.columns(2)
        with c1, components.card("rd-bt-equity"):
            st.markdown('<div class="aa-gate-title">Equity Curve</div>', unsafe_allow_html=True)
            components.plotly_chart(charts.equity_curve_chart([r["cum_net_pnl_usd"] for r in series]))
        with c2, components.card("rd-bt-drawdown"):
            st.markdown('<div class="aa-gate-title">Drawdown</div>', unsafe_allow_html=True)
            components.plotly_chart(charts.drawdown_chart([r["drawdown_usd"] for r in series]))
        if bundle:
            st.caption(
                f"Equity from the bound artifact bundle's real daily equity trace "
                f"({bundle['daily_equity']['n_rows']} day(s)) -- `{bundle['daily_equity']['path']}`."
            )
        else:
            st.caption(f"From the bound ledger `{ledger['source_artifact']}` -- {ledger['n_trades']} real trades.")
    else:
        components.committed_series_notice(key="rd-bt-series-na")

    components.section_header("Cost Sensitivity", "Directly from this experiment's committed ResultRecord.cost_stress.")
    cost = result.get("cost_stress") or {}
    if cost.get("scenarios"):
        with components.card("rd-bt-cost-chart"):
            components.plotly_chart(cost_stress_line(cost["scenarios"]))
            st.caption(f"Max net-PnL degradation under stress: {cost['max_net_pnl_degradation']:.2%}")
    else:
        components.empty_state("Cost Sensitivity", "No cost-stress evidence committed for this experiment.", key="rd-bt-cost")

    components.section_header("Bootstrap Evidence", "The committed summary CI -- not a fabricated sample distribution.")
    boot = result.get("bootstrap_evidence") or {}
    if boot:
        with components.card("rd-bt-bootstrap"):
            components.plotly_chart(
                ci_range_chart(point=boot["point"], ci_low=boot["ci_low"], ci_high=boot["ci_high"])
            )
            st.caption(f"{boot['ci_level']:.0%} CI: [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}], point {boot['point']:.3f}")

    components.section_header("Regime Evidence")
    regime = result.get("regime_evidence") or {}
    if regime.get("status") == "evaluated" and regime.get("buckets"):
        with components.card("rd-bt-regime"):
            components.plotly_chart(regime_bar(regime["buckets"]))
    else:
        components.empty_state("Regime Evidence", f"Evidence status: {regime.get('status', 'not_evaluated')}", key="rd-bt-regime-na")

    components.section_header("Trade Ledger", "Shown only when verifiably bound to this exact experiment.")
    if trades_rows:
        with components.card("rd-bt-ledger"):
            if bundle:
                st.caption(
                    f"Source: `{bundle['trades']['path']}` (sha256 {bundle['trades']['sha256'][:16]}..., "
                    f"artifact bundle {bundle['bundle_sha256'][:16]}...)"
                )
            else:
                st.caption(f"Source: `{ledger['source_artifact']}` (sha256 {ledger['source_artifact_sha256'][:16]}...)")
            st.dataframe(trades_rows, width="stretch", hide_index=True, height=320)
    else:
        components.empty_state(
            "Trade Ledger",
            "No committed artifact names a per-trade ledger as THIS experiment's source (never inferred "
            "from a shared root symbol).",
            key="rd-bt-ledger-na",
        )


# ---------------------------------------------------------------------------
# Signals -- price + real fills + actual position (MARKET REALITY pass,
# mission sections 11-13). Shared with the Backtests tear sheet so both
# render identically from the same verifiably-bound artifact bundle.
# ---------------------------------------------------------------------------


def _render_signals_tab(selection: dict) -> None:
    detail = _selection_detail(selection)
    if not detail:
        components.empty_state(
            "Signals", "No committed result for this selection yet.", key="rd-sig-na",
        )
        return
    from alpha_agent.ui.views.backtests import render_signals_section

    render_signals_section(detail, key_prefix="rd-sig")


# ---------------------------------------------------------------------------
# Sources -- the real, live research knowledge this hypothesis cited, if any
# (mission section 11's "Sources" tab). Plain lineage only -- these strings
# never feed novelty detection, deduplication, or any scientific gate (see
# `HypothesisSpec.source_inspirations`'s own docstring).
# ---------------------------------------------------------------------------


def _render_sources_tab(selection: dict) -> None:
    h = selection.get("hypothesis")
    inspirations = (h or {}).get("source_inspirations") or []
    if not inspirations:
        components.empty_state(
            "Sources",
            "No external research source was cited for this hypothesis -- either none was available "
            "at proposal time, or this selection has no HypothesisSpec attached.",
            key="rd-src-na",
        )
        return
    with components.card("rd-sources"):
        st.markdown(f"**{len(inspirations)} cited research source(s)**")
        st.caption(
            "Informational lineage only -- these are the research-knowledge-base item(s) that inspired "
            "this hypothesis, never used for novelty detection, deduplication, or any scientific gate."
        )
        for text in inspirations:
            st.markdown(f"- {text}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _registry_selection(experiment_id: str) -> dict:
    return {"source": "registry_lookup", "objective": None, "root": None, "hypothesis": None, "compiled": None,
            "evidence": None, "run_outcome": None, "experiment_id": experiment_id}


def render_experiment_performance(experiment_id: str) -> None:
    """The committed C++ result of one registry experiment (PnL, costs,
    Sharpe, trades, equity trace when bound) -- the SAME renderer as this
    page's Performance tab, for a Research Thread's Backtest step."""
    _render_backtest_tab(_registry_selection(experiment_id))


def render_experiment_validation(experiment_id: str) -> None:
    """The committed verdict, gate grid and reason codes of one registry
    experiment -- the SAME renderer as this page's Validation tab."""
    _render_validation_tab(_registry_selection(experiment_id))


def _render_validation_tab(selection: dict) -> None:
    detail = _selection_detail(selection)
    if not detail:
        components.empty_state(
            "Validation", "No committed result for this selection yet.", key="rd-val-na",
        )
        return
    result = detail["result"]
    exp = detail["experiment"]
    verdict = result["headline_verdict"] if result else None

    components.render_badge(verdict or "NOT_ADJUDICATED")
    components.render_gate_grid(result=result, trial_role=exp["trial_role"], verdict=verdict, key_prefix="rd-val")

    with components.card("rd-val-reasons"):
        st.markdown("**Reason codes**")
        if result and result["reason_codes"]:
            for code in result["reason_codes"]:
                st.markdown(f"- `{code}`")
        else:
            st.caption("No reason codes committed for this trial.")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _render_provenance_tab(selection: dict) -> None:
    detail = _selection_detail(selection)
    if not detail:
        components.empty_state(
            "Provenance", "No committed experiment record for this selection yet.", key="rd-prov-na",
        )
        return
    exp = detail["experiment"]
    result = detail["result"]
    with components.card("rd-prov"):
        for label, value in [
            ("Experiment Identity", exp["experiment_identity"]),
            ("Strategy Fingerprint", exp["strategy_fingerprint"]),
            ("Dataset Fingerprint", exp["dataset_fingerprint"]),
            ("Split Identity", exp["split_identity"]),
            ("Validation Spec Fingerprint", exp["validation_spec_fingerprint"]),
            ("Reliability Policy Fingerprint", exp["reliability_policy_fingerprint"]),
            ("Execution Config Identity", exp["execution_config_identity"]),
            ("Cost Config Identity", exp["cost_config_identity"]),
            ("Risk Identity", exp["risk_identity"]),
            ("Source Artifact", (result or {}).get("source_artifact") or "--"),
            ("Authoritative Attempt", detail.get("authoritative_attempt_id") or "--"),
        ]:
            components.provenance_row(label, value)


# ---------------------------------------------------------------------------
# Memory & Lineage
# ---------------------------------------------------------------------------


def _render_memory_tab(selection: dict) -> None:
    root, family = _selection_root_family(selection)
    compiled = selection.get("compiled")

    if family and root:
        fm = services.failure_memory_lookup(strategy_family=family, root_symbol=root)
        with components.card("rd-mem"):
            st.write(
                f"Execution attempts: {fm['execution_attempts_total']} total "
                f"({fm['valid_execution_attempts']} valid / {fm['invalid_execution_attempts']} invalid). "
                f"Verdict counts: {fm['verdict_counts']}"
            )
            for lesson in fm["lessons"]:
                st.info(lesson)
        if fm["prior_experiments"]:
            with components.card("rd-mem-prior"):
                st.markdown("**Prior Experiments**")
                st.dataframe(fm["prior_experiments"], width="stretch", hide_index=True)
        if fm["related_experiments"]:
            with components.card("rd-mem-related"):
                st.markdown("**Related Experiments** (similarity-ranked)")
                st.dataframe(
                    [
                        {"Experiment": x["experiment_id"], "Root": x["root_symbol"],
                         "Verdict": x["headline_verdict"], "Similarity": round(x["similarity"]["score"], 3),
                         "Reasons": "; ".join(x["similarity"]["reasons"])}
                        for x in fm["related_experiments"]
                    ],
                    width="stretch", hide_index=True,
                )
    else:
        components.empty_state(
            "Memory & Lineage", "No family/root context is available for this selection.", key="rd-mem-na",
        )

    de = (compiled or {}).get("duplicate_evidence") if compiled else None
    if de:
        with components.card("rd-mem-dup"):
            st.markdown("**Strategy fingerprint history** (evidence only -- never a re-execution decision)")
            st.write(f"Seen before: **{de.get('strategy_fingerprint_seen')}**")
            if de.get("advisory"):
                st.caption(de["advisory"])


# ---------------------------------------------------------------------------
# Raw / Logs
# ---------------------------------------------------------------------------


def _render_raw_tab(selection: dict) -> None:
    """Default-collapsed, advanced-only detail (product refactor section 15:
    Agent stays compact; full JSON/attempt lineage lives only here). Never
    exposes API secrets -- nothing here ever touches an API key."""
    with st.expander("HypothesisSpec JSON", expanded=False):
        h = selection.get("hypothesis")
        st.json(h) if h else st.caption("Not available for this selection.")
    with st.expander("StrategySpec JSON", expanded=False):
        compiled = selection.get("compiled")
        st.json(compiled["strategy_spec"]) if compiled else st.caption("Not available for this selection.")
    with st.expander("Registry record", expanded=False):
        detail = _selection_detail(selection)
        st.json(detail) if detail else st.caption("Not available for this selection.")
    outcome = selection.get("run_outcome")
    if outcome is not None:
        with st.expander("Run outcome (execution attempt)", expanded=False):
            if outcome.accepted and outcome.report:
                st.json(outcome.report.model_dump(mode="json"))
            else:
                st.error(outcome.error or "Execution did not complete.")
