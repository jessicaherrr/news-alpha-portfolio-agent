"""Step 5 -- Portfolio: "How would these signals form a portfolio?"

Renders the canonical Phase G `PortfolioPlan` for the thread's ranked signals
under the Research Setup. Selection, weights, risk shares and every limit
come from `alpha_agent.portfolio` (sizing from the C++ allocator); this
module computes no weight. Qualified is the default and the only plan
validation ever judges; the exploratory preview is labelled as such.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.news_alpha import DOMAIN_LABELS, MandateDomain
from alpha_agent.portfolio import EligibilityMode, PlanStatus
from alpha_agent.portfolio.policy import ELIGIBILITY_TEXT
from alpha_agent.ui import components, portfolio_plan_view
from alpha_agent.ui.research_thread import ThreadPipeline, ThreadStep
from alpha_agent.ui.workspace.common import esc, facts_html, subhead
from alpha_agent.ui.workspace.shell import render_step_heading

__all__ = ["mode_key", "render", "render_mode_control", "render_plan"]

_MODE_LABEL = {EligibilityMode.QUALIFIED: "Qualified", EligibilityMode.EXPLORATORY: "Exploratory preview"}
_CONSTRAINT_STATUS = {"Satisfied": "within the limit", "Binding": "the limit applied"}
_EMPTY_TEXT = {
    PlanStatus.NO_ELIGIBLE_SIGNAL: (
        "No signal qualifies yet",
        ("A qualified portfolio only takes signals whose discovery screen supports continuing to validation, and "
        "none of these signals does yet. Screen them in Signals -- or switch to the exploratory preview "
        "to see how the allocator would treat signals whose evidence merely leans the declared way (a preview "
        "can never be validated)."),
    ),
    PlanStatus.NO_ALLOCATABLE_SIGNAL: ("Nothing could be allocated", ("Every eligible signal was removed on the "
                                       "as-of date -- see 'Not in the portfolio' below.")),
    PlanStatus.NO_EXECUTABLE_POSITION: ("No whole position fits", ("The sized targets round to zero whole contracts "
                                        "or shares under your capital -- see 'Not in the portfolio'.")),
    PlanStatus.DEGENERATE_RISK_MODEL: ("The exposures hedge each other exactly", ("The risk model cannot size "
                                       "offsetting exposures; the plan stops rather than guess.")),
    PlanStatus.RISK_INPUTS_UNAVAILABLE: ("Risk inputs unavailable", ("Market data for an eligible instrument could "
                                         "not be loaded.")),
    PlanStatus.ALLOCATOR_UNAVAILABLE: ("The C++ allocator is not built", ("Build the C++ engine to size portfolios "
                                       "(cmake --build build/cpp).")),
}


def mode_key(pipe: ThreadPipeline) -> str:
    return f"thread-portfolio-mode-{pipe.thread.thread_id}"


def render(pipe: ThreadPipeline) -> None:
    render_step_heading(ThreadStep.PORTFOLIO)
    if pipe.candidates.is_empty:
        components.empty_state("No signals to combine", "This thread has no candidate signal, so there is no "
                               "portfolio to build.", key="thread-portfolio-empty")
        return
    mode = render_mode_control(mode_key(pipe))
    plan, missing = pipe.plan(mode)
    render_plan(plan, missing, key=f"thread-plan-{pipe.thread.thread_id}")


def render_mode_control(key: str) -> EligibilityMode:
    return st.segmented_control(
        "Eligibility", list(EligibilityMode), default=EligibilityMode.QUALIFIED, key=key,
        format_func=_MODE_LABEL.__getitem__,
        help="Qualified: only signals whose screen supports continuing to validation -- the only portfolio "
             "validation ever judges. Exploratory preview: signals whose evidence merely leans the declared way, to "
             "see how the allocator treats them. Never validated.",
    ) or EligibilityMode.QUALIFIED


def render_plan(plan, missing, *, key: str) -> None:
    """One canonical plan (or its missing-data state), shared by a thread's
    Portfolio step and the cross-thread Portfolio page."""
    if plan is None:
        futures = sum(d is MandateDomain.FUTURES for d, _ in missing)
        wait = f" -- about {futures} minute(s) for futures" if futures else ""
        with components.card(f"{key}-load"):
            st.markdown(
                f'<div class="aa-subnote" style="margin:0">{len(missing)} eligible instrument(s) need real market '
                "data before the plan can be sized: " + esc(", ".join(f"{DOMAIN_LABELS[d]} {s}" for d, s in missing))
                + ".</div>", unsafe_allow_html=True,
            )
            if st.button(f"Load market data and build the plan{wait}", key=f"{key}-load-go", type="primary",
                         icon=":material/account_balance:",
                         help="Reads already-acquired 2018-2022 bars only (no download, no cost), never the 2023-2024 "
                              "validation window or the 2025 holdout. Cached for every later plan."):
                portfolio_plan_view.load_missing(missing)
                st.rerun()
        return

    if plan.is_exploratory:
        st.warning(ELIGIBILITY_TEXT[EligibilityMode.EXPLORATORY] + ".", icon=":material/science:")
    st.caption(plan.headline())
    if plan.status is not PlanStatus.CONSTRUCTED and plan.allocation is None:
        title, text = _EMPTY_TEXT.get(plan.status, ("No portfolio", plan.status_detail))
        components.empty_state(title, text, key=f"{key}-status")
        _render_rejected(plan)
        return

    a = plan.allocation
    with components.metric_row(f"{key}-metrics"):
        cols = st.columns(5)
        cols[0].metric("As of", str(plan.as_of), help="The last common trading day of the 2018-2022 discovery data.")
        cols[1].metric("Capital", f"${plan.constraints.capital_usd:,.0f}",
                       help=next(x.note for x in plan.constraints.limits if x.name == "Capital"))
        cols[2].metric("Ex-ante vol / target", f"{a.executable.annual_vol:.1%} / {a.target_annual_vol:.1%}",
                       help="Annualized volatility of the held whole-unit book vs the target it was sized to.")
        cols[3].metric("Gross / net", f"{a.executable.gross_exposure:.2f}x / {a.executable.net_exposure:+.2f}x")
        cols[4].metric("Positions", len(plan.positions), help="Instruments actually held after whole-unit rounding.")

    subhead("Selected signals", "The signals the plan takes, grouped into independent exposures.")
    selected = portfolio_plan_view.selected_rows(plan)
    if selected:
        st.dataframe(selected, hide_index=True, width="stretch",
                     column_config={"Factor on as-of": st.column_config.NumberColumn(format="%+.4f")})

    subhead("Portfolio", "Whole contracts or shares, sized by the C++ allocator (equal risk across independent "
            "exposures, scaled down by every limit).")
    rows = portfolio_plan_view.position_rows(plan)
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch",
                     column_order=("Instrument", "Domain", "Side", "Units", "Weight", "Risk share", "Volatility",
                                   "Liquidity"))
    else:
        st.caption("No position is held.")

    _render_constraints(plan)
    _render_rejected(plan)

    with st.expander("Advanced details", icon=":material/tune:"):
        portfolio_plan_view.render_plan_tabs(plan, key=key)


def _render_constraints(plan) -> None:
    rows = portfolio_plan_view.constraint_rows(plan)
    if not rows:
        return
    subhead("Constraints", "Every limit is enforced by the allocator; a binding limit scaled the book down.",
            help="Set by your Research Setup, your profile, or the portfolio policy -- the source is shown under "
                 "Advanced details → Constraints.")
    items = []
    for r in rows:
        note = _CONSTRAINT_STATUS.get(r["Status"], r["Status"].lower())
        scope = "" if r["Scope"] in ("portfolio", "--") else f" · {r['Scope']}"
        items.append((f"✓ {r['Constraint']}{scope}", f"{note} ({r['After']} vs limit {r['Limit']})"))
    st.markdown(facts_html(items), unsafe_allow_html=True)


def _render_rejected(plan) -> None:
    rejected = portfolio_plan_view.rejection_rows(plan)
    unheld = portfolio_plan_view.unheld_rows(plan)
    if not rejected and not unheld:
        return
    with st.expander(f"Not in the portfolio ({len(rejected) + len(unheld)})", icon=":material/block:",
                     expanded=not plan.positions):
        if rejected:
            st.dataframe([{"Signal": r["Signal"], "Why": r["Reason"].capitalize(), "Detail": r["Detail"]}
                          for r in rejected], hide_index=True, width="stretch")
        if unheld:
            st.dataframe(unheld, hide_index=True, width="stretch")
