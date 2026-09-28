"""News Alpha Phase H -- renders the validation gate, signal-path memory and the
path-level study hooks under the portfolio card.

A pure renderer. The gate (`portfolio.validation.gate_portfolio`) reads no
market data and runs on every render; a validation run itself -- C++ replays
over 2018-2024 plus a registry write -- is never started from the page (the UI
never writes the registry). It runs from the command line, exactly like paper
trading. Memory comes read-only from the registry through `services`; the
"this run" view is computed now from the pipeline objects and is labelled as
not recorded. Nothing here sorts evidence into a ranking or a score.
"""
from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from alpha_agent.alpha_memory.signal_path_evidence import EventResearch, build_signal_path_evidence
from alpha_agent.alpha_memory.signal_path_memory import (
    GUIDANCE_TEXT,
    SignalPathMemory,
    summarize_path_memory,
)
from alpha_agent.news_alpha.mandate import ResearchMandate
from alpha_agent.news_alpha.study_design import (
    StudyReadiness,
    StudyTestability,
    assess_study,
    default_designs,
    probe_study_capabilities,
)
from alpha_agent.portfolio import (
    EligibilityMode,
    PortfolioConstructionPolicy,
    PortfolioPlan,
    construct_portfolio_plan,
)
from alpha_agent.portfolio.validation import (
    PortfolioValidationReport,
    PortfolioValidationStatus,
    gate_portfolio,
)
from alpha_agent.recommendation.signal_ranking import RankedSignalSet
from alpha_agent.ui import components, portfolio_plan_view, services

__all__ = [
    "CLI_COMMAND",
    "gate_for",
    "memory_rows",
    "render_research_loop_card",
    "study_rows",
    "summary_line",
]

CLI_COMMAND = "PYTHONPATH=python python scripts/news_alpha_phase_h_validation.py"
_STATUS_TONE = {
    PortfolioValidationStatus.VALIDATED: ("PASS", "Validated (validation window)"),
    PortfolioValidationStatus.REJECTED: ("REJECT", "Rejected"),
    PortfolioValidationStatus.INSUFFICIENT_EVIDENCE: ("INCONCLUSIVE", "Insufficient evidence"),
    PortfolioValidationStatus.NO_ELIGIBLE_PORTFOLIO: ("NOT_EVALUATED", "No eligible portfolio"),
    PortfolioValidationStatus.COST_MODEL_INCOMPLETE: ("NOT_AVAILABLE", "Cost model incomplete"),
    PortfolioValidationStatus.EXECUTION_UNSUPPORTED: ("NOT_AVAILABLE", "Execution unsupported"),
    PortfolioValidationStatus.PRIOR_RESULT_CITED: ("NOT_ADJUDICATED", "Prior result cited"),
}
_MEMORY_COLUMNS = {"What would change it": st.column_config.TextColumn(width="large"),
                   "To": st.column_config.TextColumn(width="medium")}


def qualified_plan(ranked: RankedSignalSet, screens: dict, mandate: ResearchMandate) -> PortfolioPlan | None:
    """The QUALIFIED plan -- the only one the loop acts on -- from cached
    snapshots, or ``None`` while market data for an eligible signal is not
    loaded (the portfolio card loads it)."""
    policy = PortfolioConstructionPolicy(eligibility=EligibilityMode.QUALIFIED)
    if portfolio_plan_view.missing_snapshots(ranked, policy):
        return None
    return construct_portfolio_plan(ranked, screens, mandate, snapshots=portfolio_plan_view.cached_snapshot,
                                    policy=policy)


def gate_for(plan: PortfolioPlan, ranked: RankedSignalSet, screens: dict) -> PortfolioValidationReport | None:
    return gate_portfolio(plan, ranked, screens)


def summary_line(plan: PortfolioPlan | None, gate: PortfolioValidationReport | None) -> str:
    if plan is None:
        return "Validation (Phase H): load the market data above to build the qualified plan first."
    if gate is None:
        return ("Validation (Phase H): the qualified portfolio is eligible -- validation runs from the command line "
                "and records its result in the experiment registry.")
    return (f"Validation (Phase H): {gate.headline()} The 2023-2024 validation window was not read; the 2025 holdout "
            "stays sealed.")


def memory_rows(memories: Sequence[SignalPathMemory]) -> list[dict[str, object]]:
    rows = []
    for m in memories:
        rows.append({
            "Route": f"{m.path_type or 'UNCLASSIFIED'} · depth {m.transmission_depth}",
            "To": " / ".join(c.replace("_", " ").lower() for c in m.consequence_states),
            "Furthest stage": m.furthest_stage.value.replace("_", " ").lower(),
            "Main reason": m.primary_reason.value.replace("_", " ").lower(),
            "What would change it": " ".join(GUIDANCE_TEXT[g] for g in m.guidance[:2]),
            "Outcomes": ", ".join(f"{k.lower()} {v}" for k, v in m.outcomes.items()),
            "Events · runs · experiments": f"{len(m.event_ids)} · {m.research_runs} · {len(m.experiment_identities)}",
        })
    return rows


def study_rows(studies: Sequence[StudyTestability]) -> list[dict[str, object]]:
    return [{
        "Question": s.design.question.value.replace("_", " ").lower(),
        "Arms": ", ".join(a.value.split("_", 1)[0] for a in s.design.arms),
        "Cells (one BH family)": s.design.declared_family_size,
        "Status": s.status.value.replace("_", " ").lower(),
        "Missing": "; ".join(b.requirement.value.replace("_", " ").lower() for b in s.blockers) or "--",
    } for s in studies]


def _render_gate(plan: PortfolioPlan | None, gate: PortfolioValidationReport | None, *, exploratory: bool,
                 key: str) -> None:
    if plan is None:
        st.caption("The qualified plan needs market data that is not loaded yet -- use the portfolio card above.")
        return
    if exploratory:
        st.info("You are viewing the exploratory preview. Previews never enter validation; the gate below judges "
                "the qualified plan.", icon=":material/science:")
    status = gate.status if gate else None
    tone, label = _STATUS_TONE.get(status, ("READY", "Eligible for validation"))
    st.markdown(components.badge(tone, label=label) + " &nbsp; " + components.badge("LOCKED", label="2025 holdout "
                                                                                                 "sealed"),
                unsafe_allow_html=True)
    if gate is None:
        st.caption("Every member's discovery screen supports continuing, and the engine can execute and cost the "
                   "book. Validation replays the frozen portfolio through the C++ engine (2018-2022 walk-forward "
                   "folds, the 2023-2024 validation window, cost stress, rebalance neighbours) and records the result "
                   "in the experiment registry. It is started from the command line, never from this page:")
        st.code(CLI_COMMAND, language="bash")
        prior = services.portfolio_validations(plan.fingerprint())
        if prior:
            st.dataframe(prior, width="stretch", hide_index=True, key=f"{key}-loop-prior")
        return
    st.caption(gate.headline())
    st.dataframe([{"Reason": f.reason.value.replace("_", " ").lower(), "Detail": f.detail,
                   "Members": len(f.candidate_signal_ids) or "--"} for f in gate.eligibility.findings]
                 or [{"Reason": "execution", "Detail": gate.status_detail, "Members": "--"}],
                 width="stretch", hide_index=True, key=f"{key}-loop-findings")
    st.caption("Eligibility authority: " + gate.eligibility.authority + " Nothing from the 2023-2024 validation "
               "window was read, so nothing was spent from it.")


def render_research_loop_card(ranked: RankedSignalSet, screens: dict, mandate: ResearchMandate,
                              events: Sequence[EventResearch], *, key: str) -> None:
    """Caption + expander under the portfolio card."""
    plan = qualified_plan(ranked, screens, mandate)
    gate = gate_for(plan, ranked, screens) if plan is not None else None
    st.caption(summary_line(plan, gate))
    status = ("market data to load" if plan is None else "eligible" if gate is None
              else _STATUS_TONE[gate.status][1].lower())
    with st.expander(f"Validation & research memory · {status}", expanded=False, key=f"{key}-loop-expander"):
        tabs = st.tabs(["Validation gate", "Research memory", "Path studies"], key=f"{key}-loop-tabs")
        exploratory = st.session_state.get(f"{key}-portfolio-mode") is EligibilityMode.EXPLORATORY
        with tabs[0]:
            _render_gate(plan, gate, exploratory=exploratory, key=key)
        with tabs[1]:
            current = build_signal_path_evidence(events, ranked=ranked, screens=screens, plan=plan,
                                                 validation=gate, recorded_at="--")
            st.markdown("**This run** -- computed now, not recorded")
            st.dataframe(memory_rows(summarize_path_memory(current)), width="stretch", hide_index=True,
                         key=f"{key}-loop-current", column_config=_MEMORY_COLUMNS)
            recorded = services.signal_path_memory_for_routes([r.path_signature for r in current if r.path_signature])
            st.markdown("**Recorded memory** -- every recorded run of these routes, from any event")
            if recorded:
                st.dataframe(memory_rows(recorded), width="stretch", hide_index=True, key=f"{key}-loop-recorded",
                             column_config=_MEMORY_COLUMNS)
            else:
                st.caption("No run of these routes is recorded yet. Recording happens with the validation command "
                           "(it records the evidence whatever the outcome).")
            st.caption("A failure belongs to the route, expression, measurement and candidate it happened at -- never "
                       "to a whole transmission family or mechanism, and nothing is blacklisted. A screen is in-sample "
                       "and unconditional; a portfolio result is about that portfolio only.")
        with tabs[2]:
            studies = [assess_study(d, probe_study_capabilities()) for d in default_designs()]
            st.dataframe(study_rows(studies), width="stretch", hide_index=True, key=f"{key}-loop-studies")
            blocked = next((s for s in studies if s.status is StudyReadiness.NOT_YET_TESTABLE), None)
            if blocked is not None:
                st.caption("Not yet testable, and not assumed either way. " + " ".join(
                    f"{b.requirement.value.replace('_', ' ').lower()}: {b.basis}." for b in blocked.blockers[:3]))
            st.caption("Direct vs supply-chain vs cross-sector, decay by depth, and mechanism vs sentiment need a "
                       "dated, point-in-time historical event sample. The unconditional factor screens above are not "
                       "event-conditioned evidence and are never used as such.")
