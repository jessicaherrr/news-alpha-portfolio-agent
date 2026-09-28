"""Phase B1 -- deterministic, reason-code-driven prose.

Every string below is built from already-committed fields (a `ResultRecord`'s
`reason_codes`, or a `ResearchPromiseBreakdown`) via a fixed lookup table or
rule -- never a second judgment, never an LLM call, and never stored as
authoritative scientific evidence (task spec section 13: "Do NOT store
generated prose as authoritative scientific evidence"). The exact machine
reason codes always remain available alongside this text (task spec section
15: "Preserve raw reason codes in technical details").
"""
from __future__ import annotations

from typing import Any

from alpha_agent.recommendation.promise import ResearchPromiseBreakdown

# ---------------------------------------------------------------------------
# reason code -> investor-readable sentence (task spec section 15)
# ---------------------------------------------------------------------------

_REASON_TEXT: dict[str, str] = {
    "null_hypothesis_not_rejected": "Statistical evidence was not strong enough to reject the null hypothesis.",
    "fdr_qvalue_above_threshold": "The result did not survive multiple-testing adjustment.",
    "deflated_sharpe_below_threshold": (
        "Risk-adjusted performance was not strong enough after accounting for selection bias."
    ),
    "negative_oos_net_pnl": "Out-of-sample net PnL was negative.",
    "fold_consistency_below_threshold": "Performance was not consistent enough across validation folds.",
    "parameter_neighbourhood_unstable": "Performance was not stable across nearby parameter choices.",
    "cost_stress_degradation_exceeds_limit": (
        "Performance degraded too much under higher trading-cost assumptions."
    ),
    "performance_concentrated_in_one_regime": "Performance was concentrated in a single market regime.",
    "performance_concentrated_in_one_root": "Performance was concentrated in a single market.",
    "insufficient_trades": "The available sample contains too few trades for reliable validation.",
    "insufficient_oos_days": "The available out-of-sample period is too short for reliable validation.",
    "insufficient_folds": "Too few walk-forward folds are available for reliable validation.",
    "insufficient_nonzero_observations": (
        "Too few non-zero return observations are available for reliable validation."
    ),
    "no_daily_trace": "No daily return trace is available for this trial.",
    "PHASE_15_TRIAL_REFUSED": "Execution was refused before producing a scientific result.",
    "INSUFFICIENT_TRAIN_EVENTS": "The training corpus did not contain enough events to fit this hypothesis.",
}


def _explain_code(code: str) -> str:
    return _REASON_TEXT.get(code, code.replace("_", " ").strip().capitalize() + ".")


def validation_blockers(result: dict[str, Any] | None, verdict: str) -> tuple[str, ...]:
    """Investor-readable translation of every committed reason code -- empty
    for a PASS (task spec section 15/9: never soften a PASS, and a PASS has
    no blocker)."""
    if verdict == "PASS":
        return ()
    codes = (result or {}).get("reason_codes") or ()
    return tuple(_explain_code(c) for c in codes)


# ---------------------------------------------------------------------------
# "Why promising" (task spec section 14)
# ---------------------------------------------------------------------------


def why_promising(result: dict[str, Any] | None, promise: ResearchPromiseBreakdown) -> str:
    if not result or "result" in promise.missing_inputs:
        return "This hypothesis has not yet produced enough committed evidence to assess research promise."
    if promise.total <= 0:
        return "Committed evidence for this hypothesis is too limited to indicate research promise."

    highlights: list[str] = []
    sharpe = result.get("annualized_sharpe")
    if sharpe is not None and sharpe > 0:
        highlights.append(f"a historical annualized Sharpe of {sharpe:.2f}")
    net_pnl = result.get("net_pnl_usd")
    if net_pnl is not None and net_pnl > 0:
        highlights.append("positive net PnL")
    frac_positive = (result.get("parameter_stability") or {}).get("fraction_positive_sharpe")
    if frac_positive is not None and frac_positive >= 0.6:
        highlights.append("good parameter stability")
    fold = result.get("fold_consistency")
    if fold is not None and fold >= 0.5:
        highlights.append("consistent performance across validation folds")

    text = "Shows " + ", ".join(highlights) if highlights else "Historical evidence is weak or mixed"

    n_trades = result.get("n_trades")
    if n_trades is not None and n_trades < 20:
        text += f", though the sample is limited to {n_trades} trades"
    return text + "."


# ---------------------------------------------------------------------------
# "Next research direction" (task spec section 16) -- conservative, rule-
# based guidance only; never automatically executed.
# ---------------------------------------------------------------------------


def _has_low_trades(result: dict[str, Any]) -> bool:
    n_trades = result.get("n_trades")
    return n_trades is not None and n_trades < 20


def _refused_for_train_events(result: dict[str, Any]) -> bool:
    return "INSUFFICIENT_TRAIN_EVENTS" in (result.get("reason_codes") or ())


def _regime_concentrated(result: dict[str, Any]) -> bool:
    regime = result.get("regime_evidence") or {}
    return regime.get("status") == "evaluated" and (regime.get("max_regime_pnl_share") or 0) > 0.6


def _parameter_unstable(result: dict[str, Any]) -> bool:
    frac_positive = (result.get("parameter_stability") or {}).get("fraction_positive_sharpe")
    return frac_positive is not None and frac_positive < 0.6


def _weak_statistical_evidence(result: dict[str, Any]) -> bool:
    dsr = result.get("dsr_probability")
    bh_q = result.get("bh_q")
    return (dsr is not None and dsr < 0.5) or (bh_q is not None and bh_q > 0.10)


_DIRECTION_RULES: tuple[tuple[Any, str], ...] = (
    (
        _has_low_trades,
        (
            "Test a mechanism with more independent trading opportunities, or obtain more history, "
            "before drawing further conclusions from this small a sample."
        ),
    ),
    (
        _refused_for_train_events,
        "Obtain a longer or more complete training corpus before this hypothesis can be adjudicated.",
    ),
    (
        _regime_concentrated,
        "Investigate a regime-conditioned successor hypothesis -- performance is concentrated in one regime.",
    ),
    (
        _parameter_unstable,
        "Test a broader predeclared parameter neighbourhood before relying on this result.",
    ),
    (
        _weak_statistical_evidence,
        "Seek stronger independent evidence (a new market, period, or mechanism) rather than retuning thresholds.",
    ),
)


def next_research_direction(result: dict[str, Any] | None) -> str:
    if not result:
        return "No committed evidence exists yet -- this remains an open research question."
    for predicate, text in _DIRECTION_RULES:
        if predicate(result):
            return text
    return "No specific follow-up is indicated by the committed evidence at this time."
