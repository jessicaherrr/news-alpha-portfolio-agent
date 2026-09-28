"""Phase B1 -- the deterministic Research Promise Score.

Purpose (task spec section 7): prioritize which already-tested candidates
deserve further research attention. This is explicitly NOT a claim of
scientific validity -- `RegistryVerdict` (PASS / REJECT / INCONCLUSIVE /
NOT_ADJUDICATED) remains the only authoritative scientific outcome, computed
entirely inside `alpha_agent.validation` / the C++ engine and never touched
here. A REJECT candidate can legitimately score HIGH; a PASS candidate's
promise score never changes its PASS status.

Design principles enforced by this module (task spec section 8):

* Deterministic -- a pure function of the committed `ResultRecord` fields
  already exposed by `alpha_agent.ui.services.list_experiments`.
* Bounded 0-100 by construction (every sub-score is `clamp(..., 0, 1)` times
  its fixed weight share).
* Missing-data-safe -- a `None`/absent field contributes exactly 0 to its
  sub-score, never a neutral or positive default. Missing evidence can only
  ever push the score DOWN, never up (task spec section 8: "missing
  parameter stability != parameter stability PASS").
* Absolute, not rank-based -- one candidate's score never changes because
  another candidate was added to or removed from the registry (task spec
  section 8's "avoid a ranking transform" rule). Leaderboard order is a
  separate concern (`alpha_agent.recommendation.candidates`).
* Independent of the LLM and of `InvestorProfile` -- nothing here reads a
  user preference; see `alpha_agent.recommendation.fit` for the (separate,
  profile-dependent) User Fit Score.

Reference ranges/thresholds are fixed, documented constants below. Several are
read (never modified, never used to gate anything) from the frozen
`ReliabilityPolicy` defaults so this score's notion of "close to the
scientific bar" tracks the SAME bar the real validation pipeline uses,
without this module importing, calling, or influencing that pipeline's
verdict logic in any way.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from alpha_agent.validation.policy import ReliabilityPolicy

#: Read-only reference to the frozen default policy thresholds. Never
#: constructed with custom values, never used to adjudicate a verdict --
#: purely a source of fixed reference numbers for the transforms below.
_POLICY = ReliabilityPolicy()

# ---------------------------------------------------------------------------
# fixed reference ranges (documented, not re-derived from the data scored)
# ---------------------------------------------------------------------------

#: Annualized Sharpe reference range for the performance component. -1.0 is a
#: clearly poor result, +2.0 a clearly excellent one for a daily-bar futures
#: strategy over a multi-year OOS window; chosen as a fixed, wide, symmetric-
#: enough band so a single lucky trial cannot saturate the component, not
#: derived from the current registry's own distribution (task spec section
#: 8's "avoid a ranking transform" rule applies to reference ranges too).
SHARPE_FLOOR = -1.0
SHARPE_CEIL = 2.0

#: Trade-count reference range for the evidence-quantity component.
#: `TRADE_FLOOR` is the frozen policy's own minimum-sample bar
#: (`MinimumSampleRequirements.min_trades`) -- a trial AT the floor earns zero
#: credit here (it only just cleared the bar to be evaluated at all).
#: `TRADE_CEIL` is a fixed, documented "well-powered" reference sample size;
#: it is not itself a policy threshold (no such richness threshold exists),
#: chosen conservatively low relative to typical multi-year daily-bar trade
#: counts so a real candidate can plausibly reach it.
TRADE_FLOOR = float(_POLICY.minimum_sample.min_trades)
TRADE_CEIL = 60.0

#: Cost-stress and regime-concentration ceilings are the frozen policy's own
#: gate thresholds (`max_cost_degradation`, `regime_max_pnl_share`) -- a
#: trial AT the threshold earns zero credit on that sub-score.
COST_STRESS_CEIL = _POLICY.max_cost_degradation
REGIME_SHARE_CEIL = _POLICY.regime_max_pnl_share

#: Statistical-significance reference points, read from the frozen policy.
DSR_REFERENCE = _POLICY.dsr_min

#: Hardening (this campaign's Part B, section 10 -- fixes a real bug): the
#: null-gate p-value and BH-FDR q-value threshold `ReliabilityPolicy` itself
#: uses to gate PASS/REJECT. Read-only reference numbers, never used to
#: adjudicate anything here.
NULL_P_THRESHOLD = _POLICY.null_p_value_max
FDR_Q_THRESHOLD = _POLICY.fdr_q_threshold

#: Floor for a threshold-relative ratio so a value of exactly 0.0 never
#: divides by zero (it still saturates at the full-credit cap of 1.0).
_EPS = 1e-9

# ---------------------------------------------------------------------------
# weights -- a small, interpretable, additive model (task spec section 8)
# ---------------------------------------------------------------------------

WEIGHT_PERFORMANCE = 25.0
WEIGHT_STATISTICAL_EVIDENCE = 30.0
WEIGHT_STABILITY = 20.0
WEIGHT_ROBUSTNESS = 15.0
WEIGHT_EVIDENCE_QUALITY = 10.0

_TOTAL_WEIGHT = (
    WEIGHT_PERFORMANCE
    + WEIGHT_STATISTICAL_EVIDENCE
    + WEIGHT_STABILITY
    + WEIGHT_ROBUSTNESS
    + WEIGHT_EVIDENCE_QUALITY
)
assert abs(_TOTAL_WEIGHT - 100.0) < 1e-9, "Research Promise component weights must sum to 100"

#: Statistical evidence splits evenly across DSR / BH-q / gating-null-p.
_STAT_SUB_WEIGHT = WEIGHT_STATISTICAL_EVIDENCE / 3.0
#: Stability splits evenly across fold consistency / parameter stability.
_STABILITY_SUB_WEIGHT = WEIGHT_STABILITY / 2.0
#: Robustness: cost stress carries 2/3, regime concentration 1/3 -- cost
#: stress is evaluated for every full-evidence trial today, regime evidence
#: only when `regime_evidence.status == "evaluated"` (task spec section 7F:
#: "only if existing committed regime evidence supports deterministic
#: scoring"); cross-market evidence is excluded entirely because it is
#: `not_evaluated` for every canonical trial in the registry today (there is
#: nothing committed to score).
_COST_STRESS_WEIGHT = WEIGHT_ROBUSTNESS * (2.0 / 3.0)
_REGIME_WEIGHT = WEIGHT_ROBUSTNESS * (1.0 / 3.0)
#: Evidence quantity/quality: trade count carries 0.7, evidence_completeness
#: 0.3 -- a "full" evidence trial with very few trades (task spec section 22:
#: "a 10-trade high-Sharpe experiment should receive an explicit evidence-
#: quality penalty") still loses most of this component.
_TRADE_COUNT_WEIGHT = WEIGHT_EVIDENCE_QUALITY * 0.7
_COMPLETENESS_WEIGHT = WEIGHT_EVIDENCE_QUALITY * 0.3

HIGH_THRESHOLD = 80.0
MEDIUM_THRESHOLD = 60.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _threshold_relative_score(value: float, threshold: float) -> float:
    """`min(1, threshold / max(value, epsilon))` (task spec section 10).

    Fixes a real bug in the original Phase B1 formula, which used the naive
    `1 - value` transform for a p-value / BH-q-value. That transform gives
    `p = 0.50` roughly HALF credit against a `p <= 0.05` scientific bar --
    exactly the "large statistical credit for evidence nowhere near the
    threshold" the task spec calls out. This transform instead gives full
    credit AT OR BETTER than the threshold (`value <= threshold`) and decays
    monotonically, and steeply, for anything worse -- `p = 0.50` against a
    `0.05` threshold now earns `0.05/0.50 = 0.10` credit, not `0.50`.

    `value >= 1.0` (the literal worst possible point for a bounded [0, 1]
    p-value / q-value -- and the exact conservative sentinel the registry
    records for a typed pre-adjudication refusal, e.g. Phase 15's "enters BH
    at p = 1") is pinned to exactly `0.0`, never the small residual
    `threshold / 1.0` the bare ratio would otherwise give. A worst-case
    placeholder must not be able to buy positive Research Promise credit.
    """
    if value >= 1.0:
        return 0.0
    return _clamp01(threshold / max(value, _EPS))


def _label(total: float) -> str:
    if total >= HIGH_THRESHOLD:
        return "HIGH"
    if total >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


class ResearchPromiseBreakdown(BaseModel):
    """Deterministic, auditable output: every component score plus which
    committed inputs were missing (and therefore contributed zero, never a
    positive default)."""

    model_config = {"frozen": True, "extra": "forbid"}

    performance: float
    statistical_evidence: float
    stability: float
    robustness: float
    evidence_quality: float
    total: float
    label: str
    missing_inputs: tuple[str, ...] = ()


def score_research_promise(result: dict[str, Any] | None) -> ResearchPromiseBreakdown:
    """Score one candidate's `ResultRecord` (as returned by
    `alpha_agent.ui.services.list_experiments`/`get_experiment`, or an
    equivalent plain dict with the same keys). `result=None` (no committed
    result at all -- e.g. a hypothesis that has never been executed) scores
    0/LOW, matching the missing-data-safe rule.
    """
    if not result:
        return ResearchPromiseBreakdown(
            performance=0.0, statistical_evidence=0.0, stability=0.0, robustness=0.0,
            evidence_quality=0.0, total=0.0, label="LOW", missing_inputs=("result",),
        )

    missing: list[str] = []

    # -- A. performance (annualized Sharpe) ---------------------------------
    ann_sharpe = result.get("annualized_sharpe")
    if ann_sharpe is None:
        performance = 0.0
        missing.append("annualized_sharpe")
    else:
        performance = _clamp01((ann_sharpe - SHARPE_FLOOR) / (SHARPE_CEIL - SHARPE_FLOOR)) * WEIGHT_PERFORMANCE

    # -- B. statistical evidence quality -------------------------------------
    dsr = result.get("dsr_probability")
    if dsr is None:
        dsr_sub = 0.0
        missing.append("dsr_probability")
    else:
        dsr_sub = _clamp01(dsr / DSR_REFERENCE) * _STAT_SUB_WEIGHT

    bh_q = result.get("bh_q")
    if bh_q is None:
        bhq_sub = 0.0
        missing.append("bh_q")
    else:
        bhq_sub = _threshold_relative_score(bh_q, FDR_Q_THRESHOLD) * _STAT_SUB_WEIGHT

    gating_p = result.get("gating_null_p")
    if gating_p is None:
        gating_sub = 0.0
        missing.append("gating_null_p")
    else:
        gating_sub = _threshold_relative_score(gating_p, NULL_P_THRESHOLD) * _STAT_SUB_WEIGHT

    statistical_evidence = dsr_sub + bhq_sub + gating_sub

    # -- C. stability ---------------------------------------------------------
    fold = result.get("fold_consistency")
    if fold is None:
        fold_sub = 0.0
        missing.append("fold_consistency")
    else:
        fold_sub = _clamp01(fold) * _STABILITY_SUB_WEIGHT

    param_stability = result.get("parameter_stability") or {}
    frac_positive = param_stability.get("fraction_positive_sharpe")
    if frac_positive is None:
        param_sub = 0.0
        missing.append("parameter_stability")
    else:
        param_sub = _clamp01(frac_positive) * _STABILITY_SUB_WEIGHT

    stability = fold_sub + param_sub

    # -- D. practical robustness (cost stress + regime concentration) --------
    cost_stress = result.get("cost_stress") or {}
    max_degradation = cost_stress.get("max_net_pnl_degradation")
    if max_degradation is None:
        cost_sub = 0.0
        missing.append("cost_stress")
    else:
        cost_sub = _clamp01(1.0 - max_degradation / COST_STRESS_CEIL) * _COST_STRESS_WEIGHT

    # Sign-safety hardening (this campaign's Part B, section 11 -- fixes a
    # real bug): `max_regime_pnl_share` measures concentration only (a
    # sign-stripped ratio of PnL across regime buckets) -- it says nothing
    # about whether the candidate is a net winner. Without a net-positive
    # OOS PnL gate, a candidate that loses money EVENLY across every regime
    # (a low concentration ratio) previously earned a strong robustness
    # sub-score for being "un-concentrated", rewarding a losing strategy. The
    # gate below is the sign-safe fix: regime-concentration credit is awarded
    # only when the candidate is a net winner over the OOS window; otherwise
    # it earns zero credit here, same as evidence that was never evaluated.
    regime_evidence = result.get("regime_evidence") or {}
    max_regime_share = regime_evidence.get("max_regime_pnl_share")
    net_pnl_usd = result.get("net_pnl_usd")
    if (
        regime_evidence.get("status") == "evaluated"
        and max_regime_share is not None
        and net_pnl_usd is not None
        and net_pnl_usd > 0.0
    ):
        regime_sub = _clamp01(1.0 - max_regime_share / REGIME_SHARE_CEIL) * _REGIME_WEIGHT
    else:
        regime_sub = 0.0
        missing.append("regime_evidence")

    robustness = cost_sub + regime_sub

    # -- E. evidence quantity / quality ---------------------------------------
    n_trades = result.get("n_trades")
    if n_trades is None:
        trade_sub = 0.0
        missing.append("n_trades")
    else:
        trade_sub = _clamp01((n_trades - TRADE_FLOOR) / (TRADE_CEIL - TRADE_FLOOR)) * _TRADE_COUNT_WEIGHT

    completeness = result.get("evidence_completeness")
    if completeness == "full":
        completeness_sub = _COMPLETENESS_WEIGHT
    else:
        completeness_sub = 0.0
        missing.append("evidence_completeness")

    evidence_quality = trade_sub + completeness_sub

    total = performance + statistical_evidence + stability + robustness + evidence_quality
    total = round(_clamp01(total / 100.0) * 100.0, 1)

    return ResearchPromiseBreakdown(
        performance=round(performance, 2),
        statistical_evidence=round(statistical_evidence, 2),
        stability=round(stability, 2),
        robustness=round(robustness, 2),
        evidence_quality=round(evidence_quality, 2),
        total=total,
        label=_label(total),
        missing_inputs=tuple(missing),
    )
