"""The Phase 15 VALIDATION and ADJUDICATION planes (Phase 15A.3).

Phase 15A.2 could prove that the 60 predeclared trials map to 60 distinct
*discriminators*, but not to 60 complete ``experiment_identity`` values: two of
the six identity planes -- the Phase 15 ``ValidationSpec`` and
``ReliabilityPolicy`` -- did not exist, and inventing them would have been
fabricated provenance. This module creates them, and it creates nothing else.

The rule this module obeys
--------------------------
**Every gate is either explicitly predeclared by the frozen Phase 15 protocol,
or inherited unchanged from the frozen Phase 13 default that Phase 15 says it
does not replace.** Nothing is chosen here, and in particular nothing is chosen
from ML performance -- no model has been fitted, so there is no performance to
choose from, which is exactly why the planes are frozen now rather than later.

The protocol is explicit that Phase 15 replaces the reliability machinery with
nothing (``docs/ML_META_LABELING_AND_REGIME.md`` s.0: "It replaces nothing: not
the ``StrategySpec``, not C++ execution, not Fill accounting, not risk, not the
``ReliabilityPolicy``, not the experiment registry"). So the Phase 15
``ReliabilityPolicy`` **is** the frozen Phase 13 default policy, and
:func:`assert_reliability_policy_inherited_unchanged` proves it by fingerprint
equality against the value committed in all 21 Phase 13.5C validation reports.
The one gate Phase 15 does restate -- BH ``q = 0.10`` -- is checked to *agree*
with the inherited value rather than to override it; a disagreement would be an
unresolved protocol conflict and raises.

What is genuinely Phase 15's own is the ``ValidationSpec``, because a validation
spec names the corpus, the split, the cost surface, the null plan and the
multiple-testing family, and all five of those are Phase 15 quantities. It is a
new typed object rather than a reuse of
:class:`alpha_agent.validation.spec.ValidationSpec` for one structural reason:
the Phase 13 spec is per-strategy (it carries a ``strategy_fingerprint`` and a
``ParameterNeighbourhood``), while a Phase 15 validation protocol is declared
once for the whole 60-trial family. Keeping it family-wide is what makes the six
identity planes genuinely shared, which is the premise the Phase 15A.2
discriminator-uniqueness proof rests on.

Two things a reader should check deliberately
---------------------------------------------
*The null plan carries only the gating null.* The frozen protocol declares one
null -- the centred block bootstrap, "requiring no extra C++ runs" (s.15) -- and
the frozen compute estimate budgets **240** C++ evaluations. The Phase 13
schedule time-shift null runs one C++ backtest per replicate, so including it
would need order 12,000 more runs and would contradict the frozen accounting.
:func:`assert_null_plan_fits_frozen_compute_budget` enforces that link rather
than leaving it as a comment.

*There is no economic hyperparameter-plateau gate, and there could not be one.*
The frozen budget evaluates economics for the selected pipeline only (60 primary
baselines + 180 meta-labeled runs), so no per-hyperparameter-point out-of-sample
economics exist to build a Phase 13 ``ParameterStabilityResult`` from. The
protocol lists a "hyperparameter plateau check" under adjudication but declares
no threshold for one, so it is recorded here as what it can be -- a diagnostic
over inner-fold objective values -- and ``parameter_stability`` is passed to
``evaluate_policy`` as ``None``, which is the frozen Phase 13 behaviour for
absent evidence. The policy's own ``param_stability_*`` thresholds are kept at
their inherited Phase 13 values, so if a later predeclared phase does produce
that evidence it is judged by a threshold nobody invented after the fact.
"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

from alpha_agent.ml.guards import (
    DEVELOPMENT_CORPUS_END_NS,
    DEVELOPMENT_CORPUS_START_NS,
    HOLDOUT_START_NS,
    HoldoutAccessError,
    assert_exclusive_corpus_bound,
)
from alpha_agent.ml.manifest import (
    COST_SCENARIOS,
    NESTED_CV,
    PLACEBO_DRAWS,
    PLACEBO_POLICY,
    MLCandidateManifest,
)
from alpha_agent.ml.models import FROZEN_TAKE_THRESHOLD
from alpha_agent.ml.splits import NestedCVSpec
from alpha_agent.validation.bootstrap import BootstrapConfig
from alpha_agent.validation.cost_stress import (
    REFERENCE_COMMISSION_USD,
    REFERENCE_SLIPPAGE_TICKS,
    REFERENCE_SPREAD_TICKS,
    CostScenario,
    CostStressPlan,
)
from alpha_agent.validation.enums import CostScenarioKind, NullMethod, Verdict
from alpha_agent.validation.fingerprint import VALIDATION_FRAMEWORK_VERSION, fingerprint
from alpha_agent.validation.nulls import NullTestConfig
from alpha_agent.validation.policy import (
    MinimumSampleRequirements,
    PolicyOutcome,
    ReliabilityPolicy,
)

#: Phase 15B bumps three schemas: two Phase-15-specific adjudication rules are
#: added (the fixed 60-hypothesis BH family with a conservative refusal p, and
#: the ML-value-add PASS gate). They are new *semantics*, so the schema strings
#: move and every downstream fingerprint -- ``MLValidationSpec.validation_fingerprint``,
#: the ``validation_spec_fingerprint`` identity plane, and all 60 pre-run
#: ``experiment_identity`` values -- moves with them. The Phase 15A.3 plane is
#: preserved as history; nothing is edited in place.
PHASE_15_VALIDATION_SPEC_SCHEMA = "ml-validation-spec/2"
PHASE_15_MULTIPLE_TESTING_SCHEMA = "ml-multiple-testing-plan/2"
PHASE_15_PLACEBO_SCHEMA = "ml-placebo-plan/2"
PHASE_15_BASELINE_COMPARISON_SCHEMA = "ml-baseline-comparison/1"

#: Phase 15B adjudication rule 1 -- the predeclared BH family is ALWAYS exactly
#: this many hypotheses. A trial that cannot produce a valid statistical result
#: because of a typed refusal/failure is never silently dropped from the family:
#: it enters BH accounting at the conservative ``p = 1.0`` and its actual typed
#: refusal/failure is preserved in the registry.
PHASE_15_BH_FAMILY_SIZE = 60
PHASE_15_REFUSED_TRIAL_BH_P_VALUE = 1.0

#: Phase 15B adjudication rule 2 -- a Phase 15 PASS must demonstrate that the ML
#: filter ADDS VALUE to its primary, not merely that the filtered strategy is
#: profitable standalone. On top of the inherited ``ReliabilityPolicy`` gates:
#:   * ``meta daily Sharpe > primary daily Sharpe`` on exactly aligned support;
#:   * the predeclared 20-draw TAKE-rate-matched placebo satisfies
#:     ``empirical placebo p <= 0.05`` with daily Sharpe as the comparison
#:     statistic and ``p = (1 + #{placebo >= observed}) / (draws + 1)``.
#: The placebo stays a null/control and never becomes a BH hypothesis.
PHASE_15_PLACEBO_COMPARISON_STATISTIC = "daily_sharpe"
PHASE_15_PLACEBO_EMPIRICAL_P_MAX = 0.05
PHASE_15_PLACEBO_EMPIRICAL_P_FORMULA = "(1 + count(placebo_stat >= observed_stat)) / (draws + 1)"
PHASE_15_META_VS_PRIMARY_SHARPE_SUPPORT = "exactly_aligned"

#: Phase-15-local reason codes. They are NOT added to the frozen Phase 13
#: ``ReasonCode`` enum -- these are Phase-15-specific adjudication semantics that
#: layer on top of the inherited policy without modifying it.
PHASE_15_REASON_TYPED_REFUSAL = "PHASE_15_TRIAL_REFUSED"
PHASE_15_REASON_META_NOT_BEAT_PRIMARY = "PHASE_15_META_DOES_NOT_BEAT_PRIMARY_SHARPE"
PHASE_15_REASON_PLACEBO_NOT_SIGNIFICANT = "PHASE_15_PLACEBO_NOT_SIGNIFICANT"
PHASE_15_REASON_ML_VALUE_ADD_CONFIRMED = "PHASE_15_ML_VALUE_ADD_CONFIRMED"

#: Cosmetic. ``ReliabilityPolicy.identity()`` drops ``policy_name`` explicitly,
#: which is what lets the Phase 15 policy be provably the Phase 13 one.
PHASE_15_POLICY_NAME = "phase_15_ml"

#: The ``reliability_policy_fingerprint`` recorded in every one of the 21 frozen
#: Phase 13.5C validation reports. The Phase 15 policy must equal it: "Phase 15
#: replaces the ReliabilityPolicy with nothing" is a checkable claim, not a
#: sentence in a document.
PHASE_13_5C_RELIABILITY_POLICY_FINGERPRINT = (
    "valreliabilitypolicy1:b0682e86917c7b84550dc4d578926b8724fc95e18edb035242241ea008ec3705"
)

#: Frozen Phase 13.5C capital base (``phase_13_5c_matrix.CAPITAL_BASE_USD``).
#: Inherited unchanged: the ML comparison must sit on the primary's own scale.
CAPITAL_BASE_USD = 100_000.0

#: Predeclared ML diagnostics (protocol s.9 / ``configs/phase_15_ml.yaml``).
#: DIAGNOSTIC ONLY -- none of them gates a verdict, and plain accuracy is never
#: the objective.
ML_DIAGNOSTIC_METRICS: tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "log_loss",
    "brier",
    "calibration_curve",
    "precision",
    "recall",
    "class_balance",
    "prediction_coverage",
    "take_rate",
)

#: Per-trial economics the protocol predeclares as REPORTED (protocol s.10).
REPORTED_ECONOMIC_METRICS: tuple[str, ...] = (
    "net_pnl_usd",
    "daily_sharpe",
    "n_trades",
    "n_fills",
    "turnover",
    "cost_stress",
    "max_drawdown_usd",
    "take_rate",
    "economic_delta_vs_primary",
)

#: Where every ``ReliabilityPolicy`` gate comes from. ``PROTOCOL`` = restated by
#: the frozen Phase 15 protocol; ``INHERITED`` = the frozen Phase 13 default,
#: unchanged, because Phase 15 replaces the reliability policy with nothing.
#: The map is exhaustive by assertion: a field added to ``ReliabilityPolicy``
#: later cannot slip into the Phase 15 fingerprint unsourced.
RELIABILITY_GATE_SOURCES: dict[str, str] = {
    "schema_version": "INHERITED: reliability-policy/1, the frozen Phase 13 schema",
    "policy_name": "COSMETIC: excluded from ReliabilityPolicy.identity() by construction",
    "minimum_sample": (
        "INHERITED: frozen Phase 13 MinimumSampleRequirements (60 OOS trading days / 20 "
        "fills / 10 trades / 3 folds / 20 non-zero return days). Phase 15's own event "
        "gates (min_train_events=100, min_test_events=20) are SPLIT gates and live in "
        "NestedCVSpec; they are a different quantity and neither replaces the other."
    ),
    "null_p_value_max": "INHERITED: frozen Phase 13 default 0.05; Phase 15 restates no alpha",
    "fdr_q_threshold": (
        "PROTOCOL: multiple_testing.bh_q = 0.10 (configs/phase_15_ml.yaml s.10, manifest "
        "multiple_testing_family). Equals the inherited Phase 13 default, checked."
    ),
    "fdr_require_canonical_rejected": "INHERITED: frozen Phase 13 default True",
    "dsr_min": "INHERITED: frozen Phase 13 default 0.95; Phase 15 restates no DSR threshold",
    "max_cost_degradation": (
        "INHERITED: frozen Phase 13 default 0.5, applied to the 1.5x / 2.0x reruns of the "
        "unchanged Phase 13.5C cost plan that Phase 15 explicitly inherits (protocol s.10)"
    ),
    "param_stability_min_fraction_positive_sharpe": (
        "INHERITED: frozen Phase 13 default 0.6. Not exercised in Phase 15 -- the frozen "
        "compute budget produces no per-hyperparameter-point OOS economics -- but kept at "
        "the inherited value so no threshold is ever invented after the fact."
    ),
    "param_stability_reject_isolated_spike": "INHERITED: frozen Phase 13 default True",
    "fold_consistency_min": (
        "INHERITED: frozen Phase 13 default 0.5, evaluated over the five predeclared "
        "nested-CV outer test blocks (2020-2024)"
    ),
    "require_positive_oos_net_pnl": "INHERITED: frozen Phase 13 default True",
    "require_regime_stability": (
        "INHERITED: frozen Phase 13 default False -- regime evidence is descriptive, as in "
        "Phase 13.5C. The Phase 15 regime ABLATION is a separate BH trial, not a gate."
    ),
    "regime_max_pnl_share": "INHERITED: frozen Phase 13 default 0.9",
    "require_cross_market": (
        "INHERITED: frozen Phase 13 default False; the protocol declares cross-root "
        "summaries descriptive and out of the BH denominator (s.11)"
    ),
    "cross_market_max_root_share": "INHERITED: frozen Phase 13 default 0.9",
    "require_ablation_mechanism_value": (
        "INHERITED: frozen Phase 13 default False; the Phase 15 regime ablation is its own "
        "predeclared BH trial, never an ablation gate on another trial"
    ),
}


# --------------------------------------------------------------------------
# the ReliabilityPolicy plane
# --------------------------------------------------------------------------
def phase_15_reliability_policy() -> ReliabilityPolicy:
    """The Phase 15 :class:`ReliabilityPolicy`: the frozen Phase 13 one, unchanged.

    Only the cosmetic ``policy_name`` is set, and ``identity()`` drops it, so
    this object fingerprints identically to the policy that adjudicated all 107
    Phase 13.5C hypotheses.
    """
    return ReliabilityPolicy(
        policy_name=PHASE_15_POLICY_NAME,
        minimum_sample=MinimumSampleRequirements(),
    )


def assert_reliability_policy_gates_are_predeclared(policy: ReliabilityPolicy) -> dict[str, str]:
    """Every gate on the policy is sourced, and every source names a real gate.

    An allowlist, not a blacklist: a field added to ``ReliabilityPolicy`` in a
    later phase fails here rather than silently entering the Phase 15
    reliability plane with no stated provenance.
    """
    declared = set(RELIABILITY_GATE_SOURCES)
    actual = set(type(policy).model_fields)
    if declared != actual:
        raise ValueError(
            "the Phase 15 reliability-gate provenance map and ReliabilityPolicy disagree; "
            f"unsourced gates {sorted(actual - declared)}, stale sources "
            f"{sorted(declared - actual)}. Every Phase 15 gate must name the protocol "
            "clause or the frozen Phase 13 default it comes from."
        )
    return dict(RELIABILITY_GATE_SOURCES)


def assert_reliability_policy_inherited_unchanged(policy: ReliabilityPolicy) -> str:
    """The Phase 15 policy IS the frozen Phase 13.5C policy (protocol s.0)."""
    got = policy.identity()
    if got != PHASE_13_5C_RELIABILITY_POLICY_FINGERPRINT:
        raise ValueError(
            f"the Phase 15 ReliabilityPolicy fingerprints {got}, not the frozen Phase 13.5C "
            f"{PHASE_13_5C_RELIABILITY_POLICY_FINGERPRINT}. Phase 15 replaces the "
            "ReliabilityPolicy with nothing; a different policy is a new frozen research "
            "semantic and needs explicit approval, not a code change."
        )
    return got


def assert_protocol_and_policy_agree(
    policy: ReliabilityPolicy, manifest: MLCandidateManifest
) -> None:
    """The one gate the protocol restates must AGREE with the inherited one.

    If the frozen Phase 15 BH ``q`` and the inherited Phase 13 ``fdr_q_threshold``
    ever disagreed, that would be a genuine unresolved protocol conflict -- not
    something to silently resolve in either direction.
    """
    bh_q = manifest.multiple_testing_family.get("bh_q")
    if bh_q is None:
        raise ValueError("the frozen manifest declares no BH q for the Phase 15 family")
    if float(bh_q) != float(policy.fdr_q_threshold):
        raise ValueError(
            f"the frozen Phase 15 protocol declares BH q={bh_q} but the inherited Phase 13 "
            f"ReliabilityPolicy gates at fdr_q_threshold={policy.fdr_q_threshold}. These are "
            "the same quantity; the disagreement is a protocol conflict that must be "
            "resolved by a human, never by picking one."
        )


# --------------------------------------------------------------------------
# the plan objects the ValidationSpec is built from
# --------------------------------------------------------------------------
class MLMultipleTestingPlan(BaseModel):
    """The Phase 15 multiple-testing family, derived from the frozen manifest.

    Deliberately a SEPARATE family from ``phase_13_5c.all``: the 107 Phase 13.5C
    trials are historical and unchanged and are never folded into this
    denominator. BH counts inspected out-of-sample economic RESULTS (60); DSR
    counts inspected model CONFIGURATIONS (380 family-wide, 3 or 8 per trial),
    because DSR is precisely the tool that deflates for search breadth.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = PHASE_15_MULTIPLE_TESTING_SCHEMA
    family_id: str
    n_trials: int = Field(ge=1)
    n_headline_trials: int = Field(ge=0)
    n_ablation_trials: int = Field(ge=0)
    bh_q: float = Field(gt=0.0, lt=1.0)
    dsr_effective_trial_count: int = Field(ge=1)
    dsr_effective_trial_count_by_model_family: dict[str, int]
    placebo_in_denominator: bool = False
    cross_root_summaries_in_denominator: bool = False
    diagnostic_reruns_in_denominator: bool = False
    phase_13_5c_trials_folded_in: bool = False
    phase_13_5c_n_trials_unchanged: int = 107

    #: Phase 15B adjudication rule 1. The predeclared family size is fixed: a
    #: trial that a typed refusal/failure keeps from producing a valid
    #: statistical result is NOT dropped -- it enters BH accounting at the
    #: conservative ``refused_trial_bh_p_value`` (1.0, never significant) and its
    #: real typed refusal is preserved in the registry.
    family_size_is_fixed: bool = True
    refused_trial_bh_p_value: float = Field(default=PHASE_15_REFUSED_TRIAL_BH_P_VALUE, ge=1.0, le=1.0)

    @model_validator(mode="after")
    def _consistent(self) -> MLMultipleTestingPlan:
        if self.n_headline_trials + self.n_ablation_trials != self.n_trials:
            raise ValueError(
                "headline + ablation trials must equal the BH denominator; a trial that is "
                "in neither role is a trial nobody declared"
            )
        if not self.family_size_is_fixed:
            raise ValueError(
                "the Phase 15 predeclared BH family size is FIXED (rule 1): a typed "
                "refusal/failure never removes a trial from the denominator; it enters at "
                "the conservative p = 1.0 with its real refusal preserved in the registry"
            )
        if self.refused_trial_bh_p_value != PHASE_15_REFUSED_TRIAL_BH_P_VALUE:
            raise ValueError(
                "a refused Phase 15 trial enters BH at the conservative p = 1.0; any other "
                "value would let a refusal influence significance"
            )
        if any(
            getattr(self, flag)
            for flag in (
                "placebo_in_denominator",
                "cross_root_summaries_in_denominator",
                "diagnostic_reruns_in_denominator",
                "phase_13_5c_trials_folded_in",
            )
        ):
            raise ValueError(
                "a null, a descriptive cross-root summary, a diagnostic rerun and the "
                "historical Phase 13.5C family are none of them Phase 15 trials and never "
                "enter this denominator"
            )
        return self

    def identity(self) -> str:
        return fingerprint(
            "mlmultitest1",
            {
                "schema_version": self.schema_version,
                "family_id": self.family_id,
                "n_trials": self.n_trials,
                "n_headline_trials": self.n_headline_trials,
                "n_ablation_trials": self.n_ablation_trials,
                "bh_q": self.bh_q,
                "dsr_effective_trial_count": self.dsr_effective_trial_count,
                "dsr_effective_trial_count_by_model_family": {
                    k: self.dsr_effective_trial_count_by_model_family[k]
                    for k in sorted(self.dsr_effective_trial_count_by_model_family)
                },
                "placebo_in_denominator": self.placebo_in_denominator,
                "cross_root_summaries_in_denominator": self.cross_root_summaries_in_denominator,
                "diagnostic_reruns_in_denominator": self.diagnostic_reruns_in_denominator,
                "phase_13_5c_trials_folded_in": self.phase_13_5c_trials_folded_in,
                "phase_13_5c_n_trials_unchanged": self.phase_13_5c_n_trials_unchanged,
                "family_size_is_fixed": self.family_size_is_fixed,
                "refused_trial_bh_p_value": self.refused_trial_bh_p_value,
            },
        )


class MLPlaceboPlan(BaseModel):
    """The TAKE-rate-matched placebo control (protocol s.15).

    It separates "the model chose well" from "the model simply traded less" --
    the specific failure mode a filter that reduces trade count invites. It is a
    NULL, so it never competes for significance and never enters the BH
    denominator, and it runs only for a trial that already rejected its gating
    null.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = PHASE_15_PLACEBO_SCHEMA
    draws: int = Field(default=PLACEBO_DRAWS, ge=1)
    policy: str = PLACEBO_POLICY
    matched_on: str = "realised_take_rate"
    cost_scenario_label: str = "baseline_1_0x"
    executed_by: str = "cpp_quant_core"
    in_bh_denominator: bool = False

    #: Phase 15B adjudication rule 2 -- the placebo's decision rule, predeclared.
    #: The comparison statistic is the daily Sharpe; the empirical p-value is
    #: ``(1 + #{placebo >= observed}) / (draws + 1)`` and a Phase 15 PASS needs
    #: ``p <= empirical_p_max``. Still a control, never a BH hypothesis.
    comparison_statistic: str = PHASE_15_PLACEBO_COMPARISON_STATISTIC
    empirical_p_max: float = Field(default=PHASE_15_PLACEBO_EMPIRICAL_P_MAX, gt=0.0, le=1.0)
    empirical_p_formula: str = PHASE_15_PLACEBO_EMPIRICAL_P_FORMULA

    @model_validator(mode="after")
    def _never_a_trial(self) -> MLPlaceboPlan:
        if self.in_bh_denominator:
            raise ValueError("a null never competes for significance")
        if self.comparison_statistic != PHASE_15_PLACEBO_COMPARISON_STATISTIC:
            raise ValueError(
                f"the Phase 15 placebo comparison statistic is frozen at "
                f"{PHASE_15_PLACEBO_COMPARISON_STATISTIC!r} (daily Sharpe); a different "
                "statistic is a different predeclared control"
            )
        if self.empirical_p_max > PHASE_15_PLACEBO_EMPIRICAL_P_MAX:
            raise ValueError(
                f"the Phase 15 placebo significance threshold is frozen at "
                f"p <= {PHASE_15_PLACEBO_EMPIRICAL_P_MAX}; it is never loosened after a result"
            )
        return self

    def identity(self) -> str:
        return fingerprint("mlplacebo1", self.model_dump(mode="json"))


class MLBaselineComparisonPlan(BaseModel):
    """PRIMARY vs META-LABELED, on exactly aligned evaluation support.

    The baseline LABEL is the deterministic primary itself, evaluated on the same
    rows through the same C++ path -- not a rescaled or reconstructed reference.
    The cost surface is the unchanged frozen Phase 13.5C plan, so the downstream
    1.5x / 2.0x stress applies to BOTH arms and the gate that reads it
    (``max_cost_degradation``) means the same thing it meant in Phase 13.5C.

    ``economic_delta_vs_primary`` is REPORTED, never gated: the frozen protocol
    lists it under "reported per trial" and declares no threshold for it, and a
    threshold nobody predeclared is not one this module may invent.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = PHASE_15_BASELINE_COMPARISON_SCHEMA
    baseline_arm_label: str = "PRIMARY"
    treatment_arm_label: str = "META_LABELED"
    alignment: str = (
        "identical row timestamps, root, strategy fingerprint, warm-up and absent-row "
        "policy; only target values differ"
    )
    baseline_cost_scenario_label: str = "baseline_1_0x"
    stress_cost_scenario_labels: tuple[str, ...] = ("stress_1_5x", "stress_2_0x")
    reported_metrics: tuple[str, ...] = REPORTED_ECONOMIC_METRICS
    delta_vs_primary_is_gated: bool = False

    @model_validator(mode="after")
    def _delta_is_reported(self) -> MLBaselineComparisonPlan:
        if self.delta_vs_primary_is_gated:
            raise ValueError(
                "the frozen protocol reports economic_delta_vs_primary and declares no "
                "PASS/REJECT threshold for it; gating on it would be an invented rule"
            )
        return self

    def identity(self) -> str:
        return fingerprint("mlbaselinecmp1", self.model_dump(mode="json"))


def phase_15_cost_stress_plan() -> CostStressPlan:
    """The frozen Phase 13.5C cost plan, unchanged (protocol s.10).

    Built from the manifest's own ``COST_SCENARIOS`` so the ValidationSpec and
    the frozen manifest cannot drift apart; a test pins it to
    ``phase_13_5c_matrix.frozen_cost_plan()`` by identity.
    """
    return CostStressPlan(
        base_commission_per_contract_usd=REFERENCE_COMMISSION_USD,
        base_slippage_ticks=REFERENCE_SLIPPAGE_TICKS,
        base_spread_ticks=REFERENCE_SPREAD_TICKS,
        scenarios=tuple(
            CostScenario(
                label=str(s["label"]),
                kind=CostScenarioKind(str(s["kind"])),
                multiplier=float(s["multiplier"]),
            )
            for s in COST_SCENARIOS
        ),
    )


def phase_15_null_test_config() -> NullTestConfig:
    """The Phase 15 null plan: the GATING centred block bootstrap, and nothing else.

    Frozen Phase 13 machinery with its frozen parameters (199 replicates, block
    length 20, seed 0, one-sided upper tail on the daily Sharpe). The Phase 13
    schedule time-shift null is absent because it costs one C++ backtest per
    replicate and the frozen Phase 15 compute estimate budgets 240 C++
    evaluations in total -- see
    :func:`assert_null_plan_fits_frozen_compute_budget`.
    """
    return NullTestConfig(methods=(NullMethod.CENTERED_BLOCK_BOOTSTRAP,))


#: Null methods whose evaluation reruns the C++ engine once per replicate.
CPP_RERUN_NULL_METHODS = frozenset(
    {NullMethod.SCHEDULE_TIME_SHIFT, NullMethod.CIRCULAR_SCHEDULE_PERMUTATION}
)


def assert_null_plan_fits_frozen_compute_budget(
    null_test: NullTestConfig, manifest: MLCandidateManifest
) -> None:
    """A null that needs C++ reruns cannot fit the frozen 240-evaluation budget.

    This is the link between the protocol's prose ("requiring no extra C++ runs")
    and its arithmetic. Adding a schedule-based null would silently multiply the
    real compute by the replicate count, so it fails here instead.
    """
    offenders = sorted(m.value for m in null_test.methods if m in CPP_RERUN_NULL_METHODS)
    if offenders:
        budget = manifest.compute_estimate.n_cpp_evaluations_total_excluding_placebo
        raise ValueError(
            f"Phase 15 null methods {offenders} rerun the C++ engine once per replicate "
            f"({null_test.n_null_samples} each), which the frozen compute estimate of "
            f"{budget} C++ evaluations does not budget for. The frozen protocol declares "
            "one null -- the centred block bootstrap on the OOS daily net return series, "
            "requiring no extra C++ runs."
        )
    if NullMethod.CENTERED_BLOCK_BOOTSTRAP not in null_test.methods:
        raise ValueError(
            "the Phase 15 gating null is the centred block bootstrap; without it no trial "
            "can satisfy the null gate and the verdict machinery is not the frozen one"
        )


def multiple_testing_plan_from_manifest(manifest: MLCandidateManifest) -> MLMultipleTestingPlan:
    """Derive the multiple-testing plan from the frozen manifest, by hand nowhere."""
    mt = manifest.multiple_testing_family
    est = manifest.compute_estimate
    return MLMultipleTestingPlan(
        family_id=str(mt["family_id"]),
        n_trials=int(mt["n_trials"]),
        n_headline_trials=int(mt["headline"]),
        n_ablation_trials=int(mt["ablation"]),
        bh_q=float(mt["bh_q"]),
        dsr_effective_trial_count=int(est.n_inspected_configurations_dsr),
        dsr_effective_trial_count_by_model_family=dict(
            est.dsr_effective_trial_count_by_model_family
        ),
        phase_13_5c_n_trials_unchanged=int(mt["phase_13_5c_n_trials_unchanged"]),
    )


# --------------------------------------------------------------------------
# the ValidationSpec plane
# --------------------------------------------------------------------------
class MLValidationSpec(BaseModel):
    """The Phase 15 validation protocol, declared once for the whole family.

    Family-wide on purpose. The Phase 13 :class:`ValidationSpec` is per-strategy
    -- it carries a ``strategy_fingerprint`` and a ``ParameterNeighbourhood`` --
    and a per-trial validation fingerprint would break the property the Phase
    15A.2 identity proof rests on: that all 60 trials share the same six identity
    planes, so discriminator uniqueness implies identity uniqueness.

    Every field is a semantic input and enters the fingerprint. ``label`` is
    cosmetic and does not.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = PHASE_15_VALIDATION_SPEC_SCHEMA
    framework_version: str = VALIDATION_FRAMEWORK_VERSION
    label: str = ""                                   # cosmetic -- never fingerprinted

    # -- data roles (protocol s.1) -----------------------------------------
    development_corpus_start_ts_ns: int = DEVELOPMENT_CORPUS_START_NS
    #: EXCLUSIVE. ``2025-01-01`` here is a half-open upper bound for data ending
    #: 2024-12-31; it does not mean holdout data exists or was loaded.
    development_corpus_end_ts_ns: int = DEVELOPMENT_CORPUS_END_NS
    locked_final_holdout_start_ts_ns: int = HOLDOUT_START_NS
    locked_final_holdout_accessed: bool = False

    # -- split plane (protocol s.8) ----------------------------------------
    nested_cv: NestedCVSpec = NESTED_CV
    #: Which folds the frozen ``fold_consistency_min`` gate reads. Phase 15 has
    #: no Phase 13 walk-forward; its folds ARE the five outer test blocks.
    fold_evidence_source: str = "nested_cv_outer_test_blocks"

    # -- economics (protocol s.10) -----------------------------------------
    capital_base_usd: float = Field(default=CAPITAL_BASE_USD, gt=0.0)
    cost_stress: CostStressPlan
    baseline_comparison: MLBaselineComparisonPlan = MLBaselineComparisonPlan()

    # -- statistics (protocol s.11, s.15) ----------------------------------
    null_test: NullTestConfig
    bootstrap: BootstrapConfig = BootstrapConfig()
    multiple_testing: MLMultipleTestingPlan
    placebo: MLPlaceboPlan = MLPlaceboPlan()

    # -- adjudication ------------------------------------------------------
    minimum_sample: MinimumSampleRequirements = MinimumSampleRequirements()
    reliability_policy_fingerprint: str = Field(min_length=1)
    #: The TAKE threshold is a decision rule of the composite strategy, and it is
    #: frozen. Recorded here too so the validation plane states the rule the
    #: economics it adjudicates were produced under.
    take_threshold: float = FROZEN_TAKE_THRESHOLD
    #: A plateau over the frozen grid can be inspected on inner-fold objective
    #: values; it cannot be inspected economically, because the frozen budget
    #: produces no per-configuration OOS economics. Diagnostic, and predeclared
    #: as diagnostic BEFORE any fit, so it cannot become a gate after a result.
    hyperparameter_plateau_evidence: str = "diagnostic_inner_objective_only"
    parameter_stability_gate_applies: bool = False
    ml_diagnostics: tuple[str, ...] = ML_DIAGNOSTIC_METRICS
    ml_diagnostics_gate_verdict: bool = False

    #: Phase 15B adjudication rule 2. A Phase 15 PASS must demonstrate the ML
    #: filter ADDS VALUE, not merely that the filtered strategy is profitable
    #: standalone. On top of the inherited ``ReliabilityPolicy`` PASS:
    #:   * ``meta daily Sharpe > primary daily Sharpe`` on exactly aligned
    #:     support (``meta_vs_primary_sharpe_support``);
    #:   * the 20-draw TAKE-rate-matched placebo (``self.placebo``) satisfies
    #:     ``empirical placebo p <= placebo_empirical_p_max``.
    #: Neither is a tunable: both are frozen here BEFORE any model is fitted.
    require_meta_beats_primary_daily_sharpe: bool = True
    meta_vs_primary_sharpe_support: str = PHASE_15_META_VS_PRIMARY_SHARPE_SUPPORT
    placebo_empirical_p_max: float = Field(
        default=PHASE_15_PLACEBO_EMPIRICAL_P_MAX, gt=0.0, le=1.0
    )

    @model_validator(mode="after")
    def _check(self) -> MLValidationSpec:
        assert_exclusive_corpus_bound(
            self.development_corpus_end_ts_ns,
            what="MLValidationSpec.development_corpus_end_ts_ns",
        )
        if self.locked_final_holdout_accessed:
            raise HoldoutAccessError(
                "the Phase 15 LOCKED_FINAL_HOLDOUT (2025) is never downloaded, queried, "
                "cost-fetched, loaded, featured, labelled, trained on, scored or inspected"
            )
        if self.locked_final_holdout_start_ts_ns != HOLDOUT_START_NS:
            raise HoldoutAccessError(
                "the locked final holdout begins exactly at the frozen boundary "
                f"{HOLDOUT_START_NS}; {self.locked_final_holdout_start_ts_ns} is not it"
            )
        if self.development_corpus_start_ts_ns >= self.development_corpus_end_ts_ns:
            raise ValueError("the development corpus window is empty or reversed")
        if self.nested_cv.corpus_start_ts_ns != self.development_corpus_start_ts_ns or (
            self.nested_cv.corpus_end_ts_ns != self.development_corpus_end_ts_ns
        ):
            raise ValueError(
                "the declared development corpus and the nested-CV corpus must be the same "
                "window; two corpora would mean two different out-of-sample claims"
            )
        if (self.nested_cv.min_train_events, self.nested_cv.min_test_events) != (100, 20):
            raise ValueError(
                "the predeclared Phase 15 event gates are FROZEN at min_train_events=100 / "
                f"min_test_events=20; found {self.nested_cv.min_train_events} / "
                f"{self.nested_cv.min_test_events}. They are never lowered after real "
                "counts are seen -- an insufficient-evidence refusal is a legitimate "
                "predeclared outcome."
            )
        if self.take_threshold != FROZEN_TAKE_THRESHOLD:
            raise ValueError(
                f"the TAKE threshold is FROZEN at {FROZEN_TAKE_THRESHOLD}; a different "
                "threshold is a new predeclared decision-policy hypothesis in a new search "
                "family, never a free parameter of this one"
            )
        if self.parameter_stability_gate_applies:
            raise ValueError(
                "the frozen Phase 15 compute budget produces no per-hyperparameter-point "
                "out-of-sample economics, so no Phase 13 ParameterStabilityResult exists "
                "to gate on; a plateau GATE would need its own predeclared threshold"
            )
        if self.ml_diagnostics_gate_verdict:
            raise ValueError(
                "ML classification metrics are diagnostic only; improvement is never "
                "claimed from them and they never gate a verdict (protocol s.9)"
            )
        if not self.require_meta_beats_primary_daily_sharpe:
            raise ValueError(
                "Phase 15B adjudication rule 2: a Phase 15 PASS must demonstrate the ML "
                "filter adds value to its primary. The meta-beats-primary daily-Sharpe "
                "requirement is a frozen Phase 15 semantic, not a switch to turn off"
            )
        if self.meta_vs_primary_sharpe_support != PHASE_15_META_VS_PRIMARY_SHARPE_SUPPORT:
            raise ValueError(
                "the primary-vs-meta daily Sharpe comparison is on EXACTLY aligned support "
                "(same bars, warm-up and absent-row policy); any other support would make "
                "the delta un-attributable to the TAKE/SKIP decisions"
            )
        if self.placebo_empirical_p_max > PHASE_15_PLACEBO_EMPIRICAL_P_MAX:
            raise ValueError(
                f"the Phase 15 placebo significance threshold is frozen at p <= "
                f"{PHASE_15_PLACEBO_EMPIRICAL_P_MAX}; it is never loosened after a result"
            )
        if self.placebo.empirical_p_max != self.placebo_empirical_p_max:
            raise ValueError(
                "the placebo plan and the validation spec disagree on the empirical-p "
                "threshold; they are the same frozen quantity"
            )
        if self.placebo.draws != PLACEBO_DRAWS:
            raise ValueError(
                f"the Phase 15 placebo is the predeclared {PLACEBO_DRAWS}-draw control; "
                f"{self.placebo.draws} draws is a different plan"
            )
        if self.cost_stress.baseline_label() != (
            self.baseline_comparison.baseline_cost_scenario_label
        ):
            raise ValueError(
                "the baseline cost scenario named by the comparison plan is not the 1.0x "
                "baseline of the cost-stress plan; the two arms must be compared on the "
                "same reference cost surface"
            )
        return self

    def validation_fingerprint(self) -> str:
        return fingerprint(
            "mlvalidationspec1",
            {
                "schema_version": self.schema_version,
                "framework_version": self.framework_version,
                "development_corpus": [
                    self.development_corpus_start_ts_ns,
                    self.development_corpus_end_ts_ns,
                ],
                "locked_final_holdout_start_ts_ns": self.locked_final_holdout_start_ts_ns,
                "locked_final_holdout_accessed": self.locked_final_holdout_accessed,
                "nested_cv": self.nested_cv.identity(),
                "fold_evidence_source": self.fold_evidence_source,
                "capital_base_usd": self.capital_base_usd,
                "cost_stress": self.cost_stress.identity(),
                "baseline_comparison": self.baseline_comparison.identity(),
                "null_test": self.null_test.identity(),
                "bootstrap": self.bootstrap.identity(),
                "multiple_testing": self.multiple_testing.identity(),
                "placebo": self.placebo.identity(),
                "minimum_sample": self.minimum_sample.identity(),
                "reliability_policy_fingerprint": self.reliability_policy_fingerprint,
                "take_threshold": self.take_threshold,
                "hyperparameter_plateau_evidence": self.hyperparameter_plateau_evidence,
                "parameter_stability_gate_applies": self.parameter_stability_gate_applies,
                "ml_diagnostics": list(self.ml_diagnostics),
                "ml_diagnostics_gate_verdict": self.ml_diagnostics_gate_verdict,
                "require_meta_beats_primary_daily_sharpe": (
                    self.require_meta_beats_primary_daily_sharpe
                ),
                "meta_vs_primary_sharpe_support": self.meta_vs_primary_sharpe_support,
                "placebo_empirical_p_max": self.placebo_empirical_p_max,
            },
        )


def build_phase_15_validation_spec(
    manifest: MLCandidateManifest | None = None,
    *,
    policy: ReliabilityPolicy | None = None,
) -> MLValidationSpec:
    """The frozen Phase 15 ``ValidationSpec``, derived from the frozen manifest.

    Reads no market data, fits no model, inspects no performance. Every value is
    either taken from the frozen manifest or inherited unchanged from Phase 13 /
    13.5C, and both facts are asserted rather than asserted-in-a-comment.
    """
    from alpha_agent.ml.manifest import build_candidate_manifest

    manifest = manifest if manifest is not None else build_candidate_manifest()
    policy = policy if policy is not None else phase_15_reliability_policy()

    assert_reliability_policy_gates_are_predeclared(policy)
    assert_reliability_policy_inherited_unchanged(policy)
    assert_protocol_and_policy_agree(policy, manifest)

    null_test = phase_15_null_test_config()
    assert_null_plan_fits_frozen_compute_budget(null_test, manifest)

    return MLValidationSpec(
        label="phase_15_ml_meta_labeling",
        nested_cv=manifest.nested_cv,
        cost_stress=phase_15_cost_stress_plan(),
        null_test=null_test,
        multiple_testing=multiple_testing_plan_from_manifest(manifest),
        placebo=MLPlaceboPlan(draws=manifest.placebo_draws, policy=manifest.placebo_policy),
        minimum_sample=policy.minimum_sample,
        reliability_policy_fingerprint=policy.identity(),
        take_threshold=manifest.take_threshold,
    )


# --------------------------------------------------------------------------
# Phase 15B adjudication rule 1 -- the BH family is ALWAYS exactly 60
# --------------------------------------------------------------------------
def phase_15_bh_family_p_values(
    trial_labels: tuple[str, ...],
    *,
    p_value_by_trial: dict[str, float],
    typed_refusal_by_trial: dict[str, str],
    plan: MLMultipleTestingPlan,
) -> tuple[tuple[str, float, str | None], ...]:
    """One ``(label, p_value, typed_refusal)`` row per predeclared trial.

    Phase 15B adjudication rule 1. The returned tuple has EXACTLY
    ``plan.n_trials`` entries, in ``trial_labels`` order. A trial that a typed
    refusal/failure kept from producing a valid statistical result is not
    dropped: it enters at ``plan.refused_trial_bh_p_value`` (1.0) and its typed
    refusal string is carried alongside so the caller can preserve it in the
    registry. A trial may not be both refused and have a real p-value.
    """
    if len(set(trial_labels)) != len(trial_labels):
        raise ValueError("trial_labels contains duplicates")
    if len(trial_labels) != plan.n_trials:
        raise ValueError(
            f"the Phase 15 BH family is fixed at {plan.n_trials} predeclared hypotheses; "
            f"{len(trial_labels)} labels were supplied -- a refusal never changes the count"
        )
    rows: list[tuple[str, float, str | None]] = []
    for label in trial_labels:
        refusal = typed_refusal_by_trial.get(label)
        if refusal is not None:
            if label in p_value_by_trial:
                raise ValueError(
                    f"trial {label!r} is marked refused ({refusal}) but also carries a "
                    "statistical p-value; a refused trial has no valid statistical result"
                )
            rows.append((label, plan.refused_trial_bh_p_value, refusal))
            continue
        if label not in p_value_by_trial:
            raise ValueError(
                f"trial {label!r} has neither a p-value nor a typed refusal; every "
                "predeclared trial must resolve to one or the other"
            )
        p = float(p_value_by_trial[label])
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"trial {label!r} p-value {p} is outside [0, 1]")
        rows.append((label, p, None))
    return tuple(rows)


# --------------------------------------------------------------------------
# Phase 15B adjudication rule 2 -- the ML-value-add PASS gate
# --------------------------------------------------------------------------
def placebo_empirical_p(
    observed_statistic: float,
    placebo_statistics: tuple[float, ...],
    *,
    draws: int,
) -> float:
    """``p = (1 + #{placebo_stat >= observed_stat}) / (draws + 1)``.

    The one-sided empirical p-value of the observed statistic (daily Sharpe)
    against the TAKE-rate-matched placebo draws. ``placebo_statistics`` must hold
    exactly ``draws`` finite values.
    """
    stats = tuple(float(s) for s in placebo_statistics)
    if len(stats) != draws:
        raise ValueError(
            f"the Phase 15 placebo is a {draws}-draw control; {len(stats)} placebo "
            "statistics were supplied"
        )
    if any(math.isnan(s) for s in stats):
        raise ValueError("a placebo statistic is NaN; every placebo draw must resolve")
    n_ge = sum(1 for s in stats if s >= float(observed_statistic))
    return (1 + n_ge) / (draws + 1)


class Phase15TrialAdjudication(BaseModel):
    """The Phase-15-specific verdict for one predeclared trial.

    It LAYERS on the inherited :class:`~alpha_agent.validation.policy.PolicyOutcome`
    without modifying the Phase 13 ``ReliabilityPolicy``. A base ``PASS`` is only
    kept if the ML filter also demonstrably adds value (rule 2); a typed
    refusal/failure yields ``REFUSED`` and never leaves the BH family (rule 1).
    """

    model_config = {"frozen": True, "extra": "forbid"}

    trial_label: str
    root_symbol: str
    primary_family: str
    model_family: str
    regime_kind: str
    trial_role: str

    #: What entered the BH family for this trial (1.0 for a refusal).
    bh_p_value: float
    typed_refusal: str | None = None

    base_reliability_verdict: str | None = None
    base_reliability_reason_codes: tuple[str, ...] = ()

    meta_daily_sharpe: float | None = None
    primary_daily_sharpe: float | None = None
    meta_beats_primary: bool | None = None

    placebo_ran: bool = False
    placebo_statistic_name: str = PHASE_15_PLACEBO_COMPARISON_STATISTIC
    placebo_observed_statistic: float | None = None
    placebo_p_value: float | None = None
    placebo_significant: bool | None = None

    verdict: str
    reason_codes: tuple[str, ...]


def adjudicate_phase_15_trial(
    spec: MLValidationSpec,
    *,
    trial_label: str,
    root_symbol: str,
    primary_family: str,
    model_family: str,
    regime_kind: str,
    trial_role: str,
    bh_p_value: float,
    base_outcome: PolicyOutcome | None = None,
    typed_refusal: str | None = None,
    meta_daily_sharpe: float | None = None,
    primary_daily_sharpe: float | None = None,
    placebo_observed_statistic: float | None = None,
    placebo_statistics: tuple[float, ...] | None = None,
) -> Phase15TrialAdjudication:
    """Apply the two Phase-15-specific adjudication rules to one trial.

    * ``typed_refusal`` set  -> ``REFUSED``; ``bh_p_value`` is forced to 1.0 and
      the trial stays in the 60-hypothesis family (rule 1).
    * base ``PASS`` -> kept only if ``meta_daily_sharpe > primary_daily_sharpe``
      on aligned support AND the placebo empirical p <= the frozen threshold
      (rule 2). Otherwise ``REJECT`` with a Phase-15 reason code.
    * base ``REJECT`` / ``INCONCLUSIVE`` -> carried through unchanged.
    """
    if typed_refusal is not None:
        return Phase15TrialAdjudication(
            trial_label=trial_label,
            root_symbol=root_symbol,
            primary_family=primary_family,
            model_family=model_family,
            regime_kind=regime_kind,
            trial_role=trial_role,
            bh_p_value=PHASE_15_REFUSED_TRIAL_BH_P_VALUE,
            typed_refusal=typed_refusal,
            verdict="REFUSED",
            reason_codes=(PHASE_15_REASON_TYPED_REFUSAL, typed_refusal),
        )

    if base_outcome is None:
        raise ValueError(
            f"trial {trial_label!r} has no typed refusal, so a base ReliabilityPolicy "
            "outcome is required to adjudicate it"
        )

    base_verdict = base_outcome.verdict
    base_reasons = tuple(c.value for c in base_outcome.reason_codes)

    common = {
        "trial_label": trial_label,
        "root_symbol": root_symbol,
        "primary_family": primary_family,
        "model_family": model_family,
        "regime_kind": regime_kind,
        "trial_role": trial_role,
        "bh_p_value": float(bh_p_value),
        "base_reliability_verdict": base_verdict.value,
        "base_reliability_reason_codes": base_reasons,
        "meta_daily_sharpe": meta_daily_sharpe,
        "primary_daily_sharpe": primary_daily_sharpe,
    }

    if base_verdict is not Verdict.PASS:
        return Phase15TrialAdjudication(
            **common,
            verdict=base_verdict.value,
            reason_codes=base_reasons,
        )

    # -- base PASS: rule 2, the ML must demonstrably add value ---------------
    reasons: list[str] = []

    meta_beats = (
        meta_daily_sharpe is not None
        and primary_daily_sharpe is not None
        and float(meta_daily_sharpe) > float(primary_daily_sharpe)
    )
    if spec.require_meta_beats_primary_daily_sharpe and not meta_beats:
        reasons.append(PHASE_15_REASON_META_NOT_BEAT_PRIMARY)

    if placebo_observed_statistic is None:
        raise ValueError(
            f"trial {trial_label!r} reached a base PASS, so the conditional placebo must "
            "have run; its observed statistic is required to adjudicate rule 2"
        )
    if not placebo_statistics:
        # the placebo was triggered but a typed refusal prevented its draws (e.g.
        # no eval-window TAKEs to match). It cannot demonstrate value-add, so the
        # trial cannot PASS.
        p_placebo = 1.0
        placebo_significant = False
        reasons.append(PHASE_15_REASON_PLACEBO_NOT_SIGNIFICANT)
    else:
        p_placebo = placebo_empirical_p(
            placebo_observed_statistic, placebo_statistics, draws=spec.placebo.draws
        )
        placebo_significant = p_placebo <= spec.placebo_empirical_p_max
        if not placebo_significant:
            reasons.append(PHASE_15_REASON_PLACEBO_NOT_SIGNIFICANT)

    if reasons:
        verdict, out_reasons = "REJECT", tuple(reasons)
    else:
        verdict = "PASS"
        out_reasons = (*base_reasons, PHASE_15_REASON_ML_VALUE_ADD_CONFIRMED)

    return Phase15TrialAdjudication(
        **common,
        meta_beats_primary=bool(meta_beats),
        placebo_ran=True,
        placebo_observed_statistic=float(placebo_observed_statistic),
        placebo_p_value=float(p_placebo),
        placebo_significant=bool(placebo_significant),
        verdict=verdict,
        reason_codes=out_reasons,
    )


def assert_phase_15_adjudication_rules_frozen(spec: MLValidationSpec) -> dict[str, object]:
    """The two Phase-15-specific adjudication rules are present and frozen.

    Called by the pre-run freeze BEFORE any market data or model fit. Returns a
    small audit payload for the freeze artifact.
    """
    mt = spec.multiple_testing
    pl = spec.placebo
    problems: list[str] = []
    if not mt.family_size_is_fixed:
        problems.append("multiple_testing.family_size_is_fixed is not True")
    if mt.n_trials != PHASE_15_BH_FAMILY_SIZE:
        problems.append(f"BH family is {mt.n_trials}, not the frozen {PHASE_15_BH_FAMILY_SIZE}")
    if mt.refused_trial_bh_p_value != PHASE_15_REFUSED_TRIAL_BH_P_VALUE:
        problems.append("refused_trial_bh_p_value is not the conservative 1.0")
    if mt.placebo_in_denominator or pl.in_bh_denominator:
        problems.append("the placebo entered the BH denominator")
    if not spec.require_meta_beats_primary_daily_sharpe:
        problems.append("require_meta_beats_primary_daily_sharpe is not True")
    if spec.meta_vs_primary_sharpe_support != PHASE_15_META_VS_PRIMARY_SHARPE_SUPPORT:
        problems.append("meta-vs-primary Sharpe support is not 'exactly_aligned'")
    if pl.draws != PLACEBO_DRAWS:
        problems.append(f"placebo draws is {pl.draws}, not the predeclared {PLACEBO_DRAWS}")
    if pl.comparison_statistic != PHASE_15_PLACEBO_COMPARISON_STATISTIC:
        problems.append("placebo comparison statistic is not the daily Sharpe")
    if pl.empirical_p_max != PHASE_15_PLACEBO_EMPIRICAL_P_MAX:
        problems.append("placebo empirical-p threshold is not the frozen 0.05")
    if spec.placebo_empirical_p_max != PHASE_15_PLACEBO_EMPIRICAL_P_MAX:
        problems.append("validation-spec placebo empirical-p threshold is not the frozen 0.05")
    if problems:
        raise ValueError(
            "the Phase 15B adjudication rules are not frozen as declared: " + "; ".join(problems)
        )
    return {
        "rule_1_bh_family_size_fixed": {
            "n_predeclared_hypotheses": mt.n_trials,
            "family_size_is_fixed": mt.family_size_is_fixed,
            "refused_trial_bh_p_value": mt.refused_trial_bh_p_value,
            "typed_refusal_preserved_in_registry": True,
            "placebo_in_bh_denominator": pl.in_bh_denominator,
        },
        "rule_2_ml_value_add_pass_gate": {
            "require_meta_beats_primary_daily_sharpe": (
                spec.require_meta_beats_primary_daily_sharpe
            ),
            "meta_vs_primary_sharpe_support": spec.meta_vs_primary_sharpe_support,
            "placebo_draws": pl.draws,
            "placebo_comparison_statistic": pl.comparison_statistic,
            "placebo_empirical_p_formula": pl.empirical_p_formula,
            "placebo_empirical_p_max": spec.placebo_empirical_p_max,
            "placebo_remains_a_control_not_a_bh_hypothesis": not pl.in_bh_denominator,
        },
        "core_phase_13_reliability_policy_modified": False,
    }
