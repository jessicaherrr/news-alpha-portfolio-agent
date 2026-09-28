"""Step 6 -- Backtest: "Does it survive a real backtest and scientific
validation?"

Performance and Validation render ONLY authoritative outputs: the Phase H
validation gate on the qualified plan (`portfolio.validation.gate_portfolio`,
reads no market data) and the registry experiments a validation run recorded
for this exact plan (C++ Fill-derived PnL, the frozen policy's verdict and
gate states). A validation run itself replays the frozen portfolio through
the C++ engine and writes the registry, so it runs from the command line --
the UI never writes the registry and never derives a verdict of its own.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.news_alpha.study_design import (
    assess_study,
    default_designs,
    probe_study_capabilities,
)
from alpha_agent.portfolio import EligibilityMode
from alpha_agent.portfolio.validation import PortfolioValidationStatus
from alpha_agent.ui import components, research_loop_view, services
from alpha_agent.ui.research_thread import ThreadPipeline, ThreadStep
from alpha_agent.ui.workspace.common import esc, info_html, subhead
from alpha_agent.ui.workspace.shell import render_step_heading

__all__ = ["recorded_runs", "render", "status_label"]

#: Backend status -> (badge tone, user-facing scientific status).
_GATE_STATUS = {
    PortfolioValidationStatus.VALIDATED: ("PASS", "Validated"),
    PortfolioValidationStatus.REJECTED: ("REJECT", "Failed validation"),
    PortfolioValidationStatus.INSUFFICIENT_EVIDENCE: ("INCONCLUSIVE", "Insufficient evidence"),
    PortfolioValidationStatus.NO_ELIGIBLE_PORTFOLIO: ("NOT_EVALUATED", "Not testable yet"),
    PortfolioValidationStatus.COST_MODEL_INCOMPLETE: ("NOT_AVAILABLE", "Cost model incomplete"),
    PortfolioValidationStatus.EXECUTION_UNSUPPORTED: ("NOT_AVAILABLE", "Execution not supported yet"),
    PortfolioValidationStatus.PRIOR_RESULT_CITED: ("NOT_ADJUDICATED", "Prior result cited"),
}
#: Registry verdict of a recorded run -> (badge tone, user-facing status).
_VERDICT_STATUS = {
    "PASS": ("PASS", "Validated"), "REJECT": ("REJECT", "Failed validation"),
    "INCONCLUSIVE": ("INCONCLUSIVE", "Insufficient evidence"),
}
_PROTOCOL = (
    ("Walk-forward folds", "Out-of-sample folds across the 2018-2022 discovery data."),
    ("Validation window", "The frozen portfolio replayed on 2023-2024 -- never seen during discovery."),
    ("Cost sensitivity", "The same book under stressed commissions and slippage."),
    ("Robustness", "Neighbouring rebalance schedules, so one lucky calendar cannot carry the result."),
    ("Multiple testing", "Corrected across the report's own family of tested hypotheses."),
    ("Locked holdout", "2025 stays sealed -- no step reads it."),
)


def recorded_runs(pipe: ThreadPipeline) -> list[dict]:
    """Registry experiments that validated this thread's qualified plan."""
    key = "recorded-runs"
    if key not in pipe._memo:
        plan, _missing = pipe.plan(EligibilityMode.QUALIFIED)
        pipe._memo[key] = [] if plan is None else services.portfolio_validations(plan.fingerprint())
    return pipe._memo[key]


def status_label(pipe: ThreadPipeline) -> tuple[str, str, str]:
    """``(badge tone, scientific status, detail)`` -- every value from the
    gate or a recorded registry verdict, never inferred here."""
    if pipe.candidates.is_empty:
        return "NOT_AVAILABLE", "Nothing to test", "This thread has no candidate signal."
    plan, missing = pipe.plan(EligibilityMode.QUALIFIED)
    if plan is None:
        return ("NOT_AVAILABLE", "Market data not loaded",
                f"The qualified plan needs market data for {len(missing)} instrument(s) -- load it on Portfolio.")
    gate = pipe.gate
    if gate is not None:
        tone, label = _GATE_STATUS[gate.status]
        return tone, label, gate.headline()
    runs = recorded_runs(pipe)
    if runs:
        tone, label = _VERDICT_STATUS.get(runs[-1]["verdict"] or "", ("NOT_ADJUDICATED", "Not adjudicated"))
        return tone, label, f"{len(runs)} validation run(s) recorded for this portfolio."
    return ("READY", "Eligible -- not run yet",
            "Every member's screen supports continuing and the engine can execute and cost the book.")


def render(pipe: ThreadPipeline) -> None:
    render_step_heading(ThreadStep.BACKTEST)
    tone, label, detail = status_label(pipe)
    st.markdown(
        '<div class="aa-fact-k">Scientific status' + info_html(
            "Set only by the validation gate or a recorded registry verdict -- this page never derives one.")
        + "</div>" + components.badge(tone, label=label) + " &nbsp; "
        + components.badge("LOCKED", label="2025 holdout sealed"),
        unsafe_allow_html=True,
    )
    st.caption(detail)

    tabs = st.tabs(["Performance", "Validation"], key=f"thread-backtest-tabs-{pipe.thread.thread_id}")
    runs = recorded_runs(pipe)
    with tabs[0]:
        _render_performance(pipe, runs)
    with tabs[1]:
        _render_validation(pipe, runs)


def _pick_run(runs: list[dict], key: str) -> str:
    if len(runs) == 1:
        return runs[0]["experiment_id"]
    ids = [r["experiment_id"] for r in runs]
    return st.selectbox("Recorded run", ids, index=len(ids) - 1, key=key,
                        format_func=lambda i: f"{i[:18]}… · {next(r['verdict'] for r in runs if r['experiment_id'] == i)}")


def _render_how_to_run() -> None:
    st.markdown("**Run the validation**")
    st.caption("A validation run replays the frozen portfolio through the C++ engine and records the result in the "
               "experiment registry, whatever the outcome -- so it runs from the command line, never from this page:")
    st.code(research_loop_view.CLI_COMMAND, language="bash")


def _render_performance(pipe: ThreadPipeline, runs: list[dict]) -> None:
    if runs:
        from alpha_agent.ui.views import research

        exp_id = _pick_run(runs, f"thread-bt-run-{pipe.thread.thread_id}")
        research.render_experiment_performance(exp_id)
        st.caption("Authoritative C++ results (Fill-derived PnL, costs, trades) of the recorded validation run.")
        return
    components.empty_state(
        "No backtest has run for this portfolio",
        "Performance appears here once a validation run has replayed the qualified portfolio through the C++ engine: "
        "PnL, Sharpe, drawdown, turnover, costs, exposure and trades -- never estimated by this page.",
        key="thread-bt-none",
    )
    if pipe.gate is None and pipe.plan(EligibilityMode.QUALIFIED)[0] is not None:
        _render_how_to_run()


def _render_validation(pipe: ThreadPipeline, runs: list[dict]) -> None:
    gate = pipe.gate
    if gate is not None:
        subhead("Why it cannot be validated yet")
        st.dataframe(
            [{"Reason": f.reason.value.replace("_", " ").capitalize(), "Detail": f.detail,
              "Signals": len(f.candidate_signal_ids) or "--"} for f in gate.eligibility.findings]
            or [{"Reason": "Execution", "Detail": gate.status_detail, "Signals": "--"}],
            hide_index=True, width="stretch",
        )
        st.caption("Eligibility authority: " + gate.eligibility.authority + " Nothing from the 2023-2024 validation "
                   "window was read, so nothing was spent from it.")
    if runs:
        from alpha_agent.ui.views import research

        subhead("Recorded validation")
        exp_id = _pick_run(runs, f"thread-val-run-{pipe.thread.thread_id}")
        research.render_experiment_validation(exp_id)
    elif gate is None and pipe.plan(EligibilityMode.QUALIFIED)[0] is not None:
        _render_how_to_run()

    subhead("What a validation run checks")
    st.markdown("".join(
        f'<div class="aa-ctx-row"><div class="aa-ctx-label">{esc(name)}</div>'
        f'<div class="aa-ctx-value">{esc(text)}</div></div>' for name, text in _PROTOCOL
    ), unsafe_allow_html=True)

    with st.expander("Path-level studies", icon=":material/science:"):
        studies = [assess_study(d, probe_study_capabilities()) for d in default_designs()]
        st.dataframe(research_loop_view.study_rows(studies), hide_index=True, width="stretch")
        st.caption("Direct vs supply-chain vs cross-sector, decay by depth, and mechanism vs sentiment need a dated, "
                   "point-in-time sample of comparable events. Unconditional factor screens are never used as such.")
