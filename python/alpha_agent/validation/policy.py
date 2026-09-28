"""The frozen typed ReliabilityPolicy and the CODE-BASED verdict (sections 8 & 19).

The policy is fixed BEFORE any validation / holdout outcome is seen (enforced by
:mod:`.holdout`). It encodes no hidden subjective finance opinion -- every gate
is an explicit, named threshold. The verdict is one of ``PASS`` / ``REJECT`` /
``INCONCLUSIVE`` with typed reason codes. No LLM may override it.

Verdict order:
    1. minimum evidence not met            -> INCONCLUSIVE
    2. any clear statistical-failure gate   -> REJECT
    3. a required evidence family not run   -> INCONCLUSIVE
    4. all required gates satisfied         -> PASS
"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field

from alpha_agent.validation.ablation import AblationReport
from alpha_agent.validation.cost_stress import CostStressReport
from alpha_agent.validation.crossmarket import CrossMarketEvidence
from alpha_agent.validation.dsr import DeflatedSharpeResult
from alpha_agent.validation.enums import EvidenceStatus, ReasonCode, Verdict
from alpha_agent.validation.fdr import FdrResult
from alpha_agent.validation.fingerprint import fingerprint
from alpha_agent.validation.metrics import OosMetrics
from alpha_agent.validation.nulls import NullTestResult
from alpha_agent.validation.regime import RegimeStabilityResult
from alpha_agent.validation.stability import ParameterStabilityResult
from alpha_agent.validation.walkforward import WalkForwardResult


class MinimumSampleRequirements(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    min_oos_trading_days: int = Field(default=60, ge=1)
    min_fills: int = Field(default=20, ge=0)
    min_trades: int = Field(default=10, ge=0)
    min_walk_forward_folds: int = Field(default=3, ge=1)
    min_nonzero_return_days: int = Field(default=20, ge=0)

    def identity(self) -> str:
        return fingerprint("valminsample1", self.model_dump(mode="json"))


class ReliabilityPolicy(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "reliability-policy/1"
    policy_name: str = "default"               # cosmetic -- NOT in the fingerprint
    minimum_sample: MinimumSampleRequirements = MinimumSampleRequirements()

    null_p_value_max: float = Field(default=0.05, gt=0.0, lt=1.0)
    fdr_q_threshold: float = Field(default=0.10, gt=0.0, lt=1.0)
    fdr_require_canonical_rejected: bool = True

    dsr_min: float = Field(default=0.95, ge=0.0, le=1.0)

    max_cost_degradation: float = Field(default=0.5, ge=0.0, le=10.0)

    param_stability_min_fraction_positive_sharpe: float = Field(default=0.6, ge=0.0, le=1.0)
    param_stability_reject_isolated_spike: bool = True

    fold_consistency_min: float = Field(default=0.5, ge=0.0, le=1.0)
    require_positive_oos_net_pnl: bool = True

    require_regime_stability: bool = False
    regime_max_pnl_share: float = Field(default=0.9, gt=0.0, le=1.0)

    require_cross_market: bool = False
    cross_market_max_root_share: float = Field(default=0.9, gt=0.0, le=1.0)

    require_ablation_mechanism_value: bool = False

    def identity(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("policy_name", None)       # cosmetic name never moves the fingerprint
        return fingerprint("valreliabilitypolicy1", payload)


class PolicyOutcome(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    verdict: Verdict
    reason_codes: tuple[ReasonCode, ...]
    policy_fingerprint: str
    gate_results: dict                          # {gate_name: bool passed} -- audit, non-semantic


def _null_gate_p(null_results: list[NullTestResult], null_p_value_max: float) -> float:
    """The p-value the null gate uses. ONLY ``NullRole.GATING`` nulls (the
    centered block bootstrap) count -- the schedule time-shift / circular
    permutation nulls are diagnostic-only and never gate PASS/REJECT
    (Phase 13.2 section 4). Of the gating nulls with enough replicates to resolve
    ``null_p_value_max`` the strategy must beat the WORST (max). If none is
    adequately powered the gate cannot be satisfied (returns 1.0)."""
    from alpha_agent.validation.enums import NullRole

    min_samples = max(19, math.ceil(1.0 / null_p_value_max) - 1)
    gating = [r for r in null_results if r.role == NullRole.GATING]
    powered = [r.p_value for r in gating if r.n_null_samples >= min_samples]
    if powered:
        return max(powered)
    any_run = [r.p_value for r in gating if r.n_null_samples > 0]
    return max(any_run) if any_run else 1.0


def evaluate_policy(
    policy: ReliabilityPolicy,
    *,
    oos_metrics: OosMetrics,
    walk_forward: WalkForwardResult,
    null_results: list[NullTestResult],
    fdr_result: FdrResult,
    canonical_trial_index: int,
    dsr_result: DeflatedSharpeResult,
    cost_report: CostStressReport | None,
    parameter_stability: ParameterStabilityResult | None,
    ablation_report: AblationReport | None,
    regime_result: RegimeStabilityResult,
    cross_market: CrossMarketEvidence,
    parameter_stability_required: bool = False,
) -> PolicyOutcome:
    """``parameter_stability_required`` (validation-safety fix, opt-in,
    default False): a call site that CAN and normally DOES have a predeclared
    parameter neighbourhood for its canonical trial (the Phase 13.5C matrix's
    frozen candidate family, and now the runtime Agent's single-strategy
    execution path) should pass ``True`` -- a missing neighbourhood there is
    genuinely missing REQUIRED evidence, never a silent pass-through (see
    step 3 below). The default False preserves, byte-for-byte, every existing
    caller's behaviour, in particular the Phase 15 ML meta-labeling
    adjudication path (`alpha_agent.ml.family_adjudication`), which passes
    `parameter_stability=None` UNCONDITIONALLY and by design -- its own frozen
    docstring (`alpha_agent.ml.adjudication`) explains there is no per-
    hyperparameter-point OOS economics to build a `ParameterStabilityResult`
    from at all, so "missing" there means "not applicable to this trial type",
    never "evidence someone forgot to compute". This flag is a plain function
    argument, not a `ReliabilityPolicy` field -- it never changes
    `policy.identity()`, so no experiment_identity, evaluation-plane check, or
    Phase 15 "fingerprints identically to Phase 13.5C's" assertion is
    affected."""
    ms = policy.minimum_sample
    gates: dict[str, bool] = {}
    reasons: list[ReasonCode] = []

    # -- 1. minimum evidence -------------------------------------------------
    if oos_metrics.n_trading_days == 0:
        return PolicyOutcome(
            verdict=Verdict.INCONCLUSIVE,
            reason_codes=(ReasonCode.NO_DAILY_TRACE,),
            policy_fingerprint=policy.identity(),
            gate_results={"has_daily_trace": False},
        )
    insufficient: list[ReasonCode] = []
    if oos_metrics.n_trading_days < ms.min_oos_trading_days:
        insufficient.append(ReasonCode.INSUFFICIENT_OOS_DAYS)
    if oos_metrics.n_fills < ms.min_fills:
        insufficient.append(ReasonCode.INSUFFICIENT_TRADES)
    if oos_metrics.n_trades < ms.min_trades:
        insufficient.append(ReasonCode.INSUFFICIENT_TRADES)
    if walk_forward.n_folds_evaluated < ms.min_walk_forward_folds:
        insufficient.append(ReasonCode.INSUFFICIENT_FOLDS)
    if oos_metrics.n_nonzero_return_days < ms.min_nonzero_return_days:
        insufficient.append(ReasonCode.INSUFFICIENT_NONZERO_OBS)
    gates["minimum_sample"] = not insufficient
    if insufficient:
        # de-dup, preserve order
        seen: set[ReasonCode] = set()
        codes = tuple(c for c in insufficient if not (c in seen or seen.add(c)))
        return PolicyOutcome(
            verdict=Verdict.INCONCLUSIVE,
            reason_codes=codes,
            policy_fingerprint=policy.identity(),
            gate_results=gates,
        )

    # -- 2. clear statistical failure -> REJECT ----------------------------
    gate_p = _null_gate_p(null_results, policy.null_p_value_max)
    gates["null_rejected"] = gate_p <= policy.null_p_value_max
    if not gates["null_rejected"]:
        reasons.append(ReasonCode.NULL_NOT_REJECTED)

    canon_rejected = (
        0 <= canonical_trial_index < len(fdr_result.decisions)
        and fdr_result.decisions[canonical_trial_index].rejected
    )
    gates["fdr_canonical_rejected"] = canon_rejected or not policy.fdr_require_canonical_rejected
    if policy.fdr_require_canonical_rejected and not canon_rejected:
        reasons.append(ReasonCode.FDR_NOT_SIGNIFICANT)

    gates["dsr"] = dsr_result.is_valid and dsr_result.deflated_sharpe_ratio >= policy.dsr_min
    if not gates["dsr"]:
        reasons.append(ReasonCode.DSR_BELOW_THRESHOLD)

    if policy.require_positive_oos_net_pnl:
        gates["positive_oos_net_pnl"] = oos_metrics.oos_net_pnl_usd > 0.0
        if not gates["positive_oos_net_pnl"]:
            reasons.append(ReasonCode.NEGATIVE_OOS_NET_PNL)

    gates["fold_consistency"] = walk_forward.fold_consistency >= policy.fold_consistency_min
    if not gates["fold_consistency"]:
        reasons.append(ReasonCode.FOLD_CONSISTENCY_FAILED)

    if cost_report is not None:
        gates["cost_stress"] = cost_report.max_net_pnl_degradation <= policy.max_cost_degradation
        if not gates["cost_stress"]:
            reasons.append(ReasonCode.COST_STRESS_FAILED)

    if parameter_stability is not None:
        stable = (
            parameter_stability.fraction_positive_sharpe
            >= policy.param_stability_min_fraction_positive_sharpe
        )
        if policy.param_stability_reject_isolated_spike and parameter_stability.canonical_is_isolated_spike:
            stable = False
        gates["parameter_stability"] = stable
        if not stable:
            reasons.append(ReasonCode.PARAMETER_UNSTABLE)
    # else: `parameter_stability is None` -- handled in step 3 below via the
    # `parameter_stability_required` flag (validation-safety fix). When the
    # caller declares this evidence required (a genuinely fresh hypothesis
    # with no predeclared neighbourhood, e.g. a runtime-agent proposal) a
    # `None` here must never be silently treated as "gate not applicable" --
    # that previously let a trial reach PASS / ALL_GATES_SATISFIED with no
    # parameter-stability evidence at all.

    if policy.require_regime_stability and regime_result.status == EvidenceStatus.EVALUATED:
        gates["regime_stability"] = regime_result.max_regime_pnl_share <= policy.regime_max_pnl_share
        if not gates["regime_stability"]:
            reasons.append(ReasonCode.REGIME_CONCENTRATED)

    if policy.require_cross_market and cross_market.status == EvidenceStatus.EVALUATED:
        gates["cross_market"] = (
            cross_market.max_single_root_pnl_share <= policy.cross_market_max_root_share
        )
        if not gates["cross_market"]:
            reasons.append(ReasonCode.CROSS_MARKET_CONCENTRATED)

    if policy.require_ablation_mechanism_value and ablation_report is not None:
        gates["ablation_mechanism_value"] = ablation_report.mechanism_adds_value
        if not ablation_report.mechanism_adds_value:
            reasons.append(ReasonCode.PARAMETER_UNSTABLE)  # mechanism not distinguishable
    # else (require_ablation_mechanism_value and ablation_report is None):
    # handled in step 3 below -- ablation is OPTIONAL by default
    # (require_ablation_mechanism_value=False, unchanged here), but a policy
    # that DOES require it must not silently skip a missing ablation report
    # the same way it must not for parameter stability.

    if reasons:
        seen2: set[ReasonCode] = set()
        codes = tuple(c for c in reasons if not (c in seen2 or seen2.add(c)))
        return PolicyOutcome(
            verdict=Verdict.REJECT,
            reason_codes=codes,
            policy_fingerprint=policy.identity(),
            gate_results=gates,
        )

    # -- 3. required evidence family not evaluated -> INCONCLUSIVE ---------
    not_run: list[ReasonCode] = []
    # Parameter stability: required only when the caller says so
    # (`parameter_stability_required`, see the function docstring) -- its
    # absence there is missing REQUIRED evidence, never a silent pass-through
    # and never a fabricated REJECT. When not required (the default, and the
    # Phase 15 ML adjudication path's explicit, documented, unchanged case),
    # a `None` here contributes nothing, exactly as before this fix.
    if parameter_stability_required and parameter_stability is None:
        not_run.append(ReasonCode.PARAMETER_STABILITY_NOT_EVALUATED)
    # Ablation is required only when the policy says so (default: optional).
    # When required, a missing report is the same "evidence not evaluated"
    # shape as regime/cross-market, never a silent PASS.
    if policy.require_ablation_mechanism_value and ablation_report is None:
        not_run.append(ReasonCode.ABLATION_NOT_EVALUATED)
    if policy.require_regime_stability and regime_result.status != EvidenceStatus.EVALUATED:
        not_run.append(ReasonCode.REGIME_NOT_EVALUATED)
    if policy.require_cross_market and cross_market.status != EvidenceStatus.EVALUATED:
        not_run.append(ReasonCode.CROSS_MARKET_NOT_EVALUATED)
    if not_run:
        return PolicyOutcome(
            verdict=Verdict.INCONCLUSIVE,
            reason_codes=tuple(not_run),
            policy_fingerprint=policy.identity(),
            gate_results=gates,
        )

    # -- 4. PASS ----------------------------------------------------------
    return PolicyOutcome(
        verdict=Verdict.PASS,
        reason_codes=(ReasonCode.ALL_GATES_SATISFIED,),
        policy_fingerprint=policy.identity(),
        gate_results=gates,
    )


# A permissive policy for the synthetic statistical fixtures (section 23): it
# exercises the PASS path on a deliberately strong synthetic signal without
# requiring a large real sample.
TEST_FIXTURE_POLICY = ReliabilityPolicy(
    policy_name="test_fixture",
    minimum_sample=MinimumSampleRequirements(
        min_oos_trading_days=40,
        min_fills=0,
        min_trades=0,
        min_walk_forward_folds=2,
        min_nonzero_return_days=20,
    ),
    null_p_value_max=0.10,
    fdr_q_threshold=0.10,
    fdr_require_canonical_rejected=True,
    dsr_min=0.90,
    max_cost_degradation=0.6,
    param_stability_min_fraction_positive_sharpe=0.5,
    fold_consistency_min=0.5,
    require_positive_oos_net_pnl=True,
)
