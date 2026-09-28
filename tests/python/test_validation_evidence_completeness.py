"""Validation-safety fix (approved narrow scientific-semantics change):
missing REQUIRED scientific evidence must never produce PASS / paper
eligibility. This file reproduces and pins the original audit finding and
proves the fix, without touching any numeric threshold, statistic, BH/FDR
method, DSR formula, split, cost model, C++ execution semantics, dataset
boundary, or holdout rule.

The verified blocker (empirically reproduced by the prior audit):

    parameter_stability = None
    ablation_report = None
        =>  verdict = PASS
            reason_codes = (ALL_GATES_SATISFIED,)

Fix (`alpha_agent.validation.policy.evaluate_policy`): a new, opt-in, PLAIN
FUNCTION ARGUMENT `parameter_stability_required` (default False -- NOT a
`ReliabilityPolicy` field, so `policy.identity()` and every fingerprint/
evaluation-plane check derived from it are untouched). When a caller passes
`True`, a `None` parameter_stability routes through the SAME "required
evidence family not evaluated -> INCONCLUSIVE" step the frozen engine already
uses for regime/cross-market, via a new
`ReasonCode.PARAMETER_STABILITY_NOT_EVALUATED`, and headline PASS becomes
unreachable. Only ONE real call site opts in:
`alpha_agent.agents.execution_service._non_family_outcome` -- the runtime
Agent's single-strategy execution path the original audit actually
reproduced the bug against.

This scoping is deliberate and load-bearing, not incidental: the Phase 15 ML
meta-labeling adjudication path (`alpha_agent.ml.family_adjudication`) passes
`parameter_stability=None` UNCONDITIONALLY, by design -- its own frozen
docstring (`alpha_agent.ml.adjudication`) explains there is no per-
hyperparameter-point OOS economics to build a `ParameterStabilityResult` from
for a trained classifier at all, so "missing" there means "not applicable",
never "forgotten". Making the requirement unconditional inside
`evaluate_policy` itself (an earlier, REJECTED draft of this fix) would have
silently made Phase 15 PASS unreachable too -- a real regression against an
already-frozen, already-audited scientific pathway this task does not
authorize touching. `test_D2_phase_13_5c_and_phase_15_ml_paths_are_byte_
unchanged` pins the corrected scoping: the exact original reproduction shape
still reaches PASS when the new flag is left at its default.

Ablation gets the symmetric fix for when a policy sets
`require_ablation_mechanism_value=True` (default stays False / optional,
unchanged, including for Phase 15 -- see its own `RELIABILITY_GATE_SOURCES`
provenance) via `ReasonCode.ABLATION_NOT_EVALUATED`.

Everything else is untouched: `ReliabilityPolicy.identity()` is byte-identical
(no new Pydantic field was added -- see `test_reliability_policy_fingerprint_
is_byte_identical_after_the_fix`), the verdict order (INCONCLUSIVE -> REJECT
-> INCONCLUSIVE -> PASS) is unchanged, and every existing threshold/formula in
`evaluate_policy` is untouched.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import test_reliability_validation as trv
import validation_fixtures as vf
from alpha_agent.core.instrument import AssetDomain
from alpha_agent.paper.eligibility import assert_paper_trading_eligible
from alpha_agent.paper.errors import PaperTradingEligibilityError
from alpha_agent.registry.enums import ExperimentStatus, RegistryVerdict, TrialRole
from alpha_agent.registry.models import ExperimentRecord, MarketWindow, ResultRecord
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.strategy import strategy_fingerprint
from alpha_agent.strategy.candidates_phase_13_5c import spec_for_params
from alpha_agent.ui import services
from alpha_agent.validation import ParameterNeighbourhood, ReasonCode, ReliabilityPolicy, Verdict
from alpha_agent.validation.policy import TEST_FIXTURE_POLICY

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(),
    reason="Phase 14 registry sqlite not present in this checkout",
)


def _strong_signal_report(*, neighbourhood=None, neighbour_results=None):
    """The exact shape the original audit reproduction used: a strong
    synthetic canonical signal (every OTHER required gate genuinely passes)
    with an optional parameter neighbourhood. `parameter_stability_required=
    True` mirrors the real call site this whole file audits
    (`alpha_agent.agents.execution_service._non_family_outcome`, the runtime
    Agent's single-strategy path) -- NOT the frozen Phase 13.5C matrix or the
    Phase 15 ML adjudication path, both of which keep the default False and
    are proven byte-unchanged elsewhere in this suite."""
    pol = TEST_FIXTURE_POLICY
    spec = trv._spec(pol, neighbourhood=neighbourhood)
    strong = vf.fixture_B_strong_signal(180, mu=140.0, sigma=180.0, seed=1)
    return trv._assemble_synth(
        spec, pol, strong,
        cost_results=trv._cost_results(base_net=float(strong.sum()), degrade_2x_to=float(strong.sum()) * 0.7),
        neighbour_results=neighbour_results,
        n_folds=4,
        parameter_stability_required=True,
    )


# ---------------------------------------------------------------------------
# A/B -- the original blocker: missing parameter_stability must never PASS
# ---------------------------------------------------------------------------
def test_A_missing_parameter_stability_on_a_strong_signal_is_not_pass():
    rep = _strong_signal_report()  # neighbourhood=None -> parameter_stability=None
    assert rep.parameter_stability is None
    assert rep.ablation_report is None
    assert rep.verdict != Verdict.PASS, (
        "REGRESSION: missing required parameter-stability evidence reached PASS"
    )
    assert rep.verdict == Verdict.INCONCLUSIVE


def test_B_missing_parameter_stability_never_emits_all_gates_satisfied():
    rep = _strong_signal_report()
    assert ReasonCode.ALL_GATES_SATISFIED not in rep.reason_codes
    assert ReasonCode.PARAMETER_STABILITY_NOT_EVALUATED in rep.reason_codes


# ---------------------------------------------------------------------------
# C -- paper eligibility defense in depth
# ---------------------------------------------------------------------------
_PARAMS = {"root_symbol": "NQ", "fast_horizon": 21, "slow_horizon": 120, "size": 1}


def _tsmom_fingerprint() -> str:
    return strategy_fingerprint(spec_for_params("tsmom", _PARAMS))


_BASE_IDENTITY_FIELDS = {
    "dataset_fingerprint": "valdataset2:test",
    "split_identity": "splitplan1:test",
    "validation_spec_fingerprint": "validationspec1:test",
    "reliability_policy_fingerprint": ReliabilityPolicy().identity(),
    "execution_config_identity": "execconfig1:test",
    "cost_config_identity": "costconfig1:test",
    "risk_identity": "riskconfig1:test",
    "feature_spec_fingerprint": "featset1:test",
}


def _experiment_record(*, identity: str, strategy_fp: str) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_identity=identity,
        experiment_id="NQ__TSMOM__CANONICAL__EVIDENCE_TEST",
        display_name="NQ tsmom (evidence-completeness test fixture)",
        created_at="2026-01-01T00:00:00+00:00",
        phase="TEST",
        status=ExperimentStatus.COMPLETED,
        root_symbol="NQ",
        asset_domain=AssetDomain.FUTURES,
        strategy_family="tsmom",
        strategy_fingerprint=strategy_fp,
        strategy_id="TEST-TSMOM",
        strategy_spec_json={
            "schema": "registry-strategy-spec/1",
            "strategy_family": "tsmom",
            "params": _PARAMS,
            "parameter_names": sorted(_PARAMS),
            "feature_fingerprints": ["feat1:x"],
        },
        market_window=MarketWindow(label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31"),
        trial_role=TrialRole.CANONICAL,
        parameter_variant_identity="paramvariant1:test",
        parameter_variant_label="canonical",
        **_BASE_IDENTITY_FIELDS,
    )


@pytest.fixture
def isolated_registry(tmp_path):
    with ExperimentRegistry(tmp_path / "evidence_completeness_test.sqlite") as reg:
        yield reg


def test_C_paper_eligibility_refuses_a_pass_record_with_incomplete_evidence(isolated_registry):
    """Defense in depth (never rely on the upstream fix alone): even a
    hand-constructed registry row that claims PASS + ALL_GATES_SATISFIED --
    something `evaluate_policy` can no longer itself produce -- must still be
    refused by `assert_paper_trading_eligible` if its OWN committed
    reason_codes also carry a required-evidence-not-evaluated code."""
    fp = _tsmom_fingerprint()
    identity = "experiment1:" + "a" * 64
    record = _experiment_record(identity=identity, strategy_fp=fp)
    result = ResultRecord(
        experiment_identity=identity,
        headline_verdict=RegistryVerdict.PASS,
        reason_codes=("all_required_gates_satisfied", "parameter_stability_not_evaluated"),
        net_pnl_usd=1234.0,
        daily_sharpe=0.05,
        annualized_sharpe=0.8,
        n_trades=10,
    )
    isolated_registry.insert_experiment(record, result)
    with pytest.raises(PaperTradingEligibilityError, match="required scientific evidence was not evaluated"):
        assert_paper_trading_eligible(isolated_registry, identity)


def test_C2_paper_eligibility_still_accepts_a_genuinely_complete_pass(isolated_registry):
    """The defensive check must not become a new false-negative: a genuine,
    complete PASS (no incomplete-evidence code) remains eligible."""
    fp = _tsmom_fingerprint()
    identity = "experiment1:" + "b" * 64
    record = _experiment_record(identity=identity, strategy_fp=fp)
    result = ResultRecord(
        experiment_identity=identity,
        headline_verdict=RegistryVerdict.PASS,
        reason_codes=("all_required_gates_satisfied",),
        net_pnl_usd=1234.0,
        daily_sharpe=0.05,
        annualized_sharpe=0.8,
        n_trades=10,
    )
    isolated_registry.insert_experiment(record, result)
    elig = assert_paper_trading_eligible(isolated_registry, identity)
    assert elig.strategy_family == "tsmom"


# ---------------------------------------------------------------------------
# D/E -- PASS remains possible with real evidence; REJECT-on-instability unchanged
# ---------------------------------------------------------------------------
def test_D_pass_remains_possible_with_a_real_predeclared_neighbourhood():
    nb = ParameterNeighbourhood(strategy_key="synthetic", canonical_params={"w": 20},
                                neighbour_params=tuple({"w": 20 + d} for d in range(1, 7)))
    fx = vf.fixture_D_broad_plateau(seed=3)
    strong = vf.fixture_B_strong_signal(180, mu=140.0, sigma=180.0, seed=1)
    neigh = trv._neigh_results(strong, fx["neighbours"])
    rep = _strong_signal_report(neighbourhood=nb, neighbour_results=neigh)
    assert rep.verdict == Verdict.PASS, rep.reason_codes
    assert rep.reason_codes == (ReasonCode.ALL_GATES_SATISFIED,)
    assert rep.parameter_stability is not None


def test_D2_phase_13_5c_and_phase_15_ml_paths_are_byte_unchanged():
    """`parameter_stability_required` defaults to False -- the frozen Phase
    13.5C matrix (`validation.phase_13_5c_matrix`) and the Phase 15 ML
    adjudication path (`alpha_agent.ml.family_adjudication`, whose own
    docstring documents `parameter_stability=None` as intentional and
    unconditional) never pass it, so this exact original-audit scenario
    (missing parameter stability on an otherwise-strong signal) MUST still
    reach PASS for them -- proving the validation-safety fix is scoped
    exclusively to the runtime Agent's execution_service.py call site, never
    a global behavior change."""
    pol = TEST_FIXTURE_POLICY
    spec = trv._spec(pol)  # neighbourhood=None
    strong = vf.fixture_B_strong_signal(180, mu=140.0, sigma=180.0, seed=1)
    rep = trv._assemble_synth(
        spec, pol, strong,
        cost_results=trv._cost_results(base_net=float(strong.sum()), degrade_2x_to=float(strong.sum()) * 0.7),
        n_folds=4,
        # parameter_stability_required omitted -> default False, exactly what
        # phase_13_5c_matrix.py and ml/family_adjudication.py both do today.
    )
    assert rep.parameter_stability is None
    assert rep.verdict == Verdict.PASS, rep.reason_codes
    assert rep.reason_codes == (ReasonCode.ALL_GATES_SATISFIED,)


def test_E_evaluated_unstable_parameter_neighbourhood_is_reject_not_inconclusive():
    """An EVALUATED (not missing) parameter-stability gate that fails must
    still REJECT with the pre-existing PARAMETER_UNSTABLE code -- the fix only
    changes what happens when the evidence is MISSING, never what happens
    when it is present and fails. Reuses the same unstable-spike fixture
    `test_S_isolated_spike_flagged_unstable` already proves is unstable."""
    fx = vf.fixture_C_unstable_spike()
    nb = ParameterNeighbourhood(strategy_key="synthetic", canonical_params={"w": 20},
                                neighbour_params=tuple({"w": 20 + d} for d in range(1, 9)))
    neigh = trv._neigh_results(fx["canonical"], fx["neighbours"])
    pol = TEST_FIXTURE_POLICY
    spec = trv._spec(pol, neighbourhood=nb)
    canonical_pnl = fx["canonical"]
    total = float(np.asarray(canonical_pnl, dtype=float).sum())
    rep = trv._assemble_synth(
        spec, pol, canonical_pnl,
        cost_results=trv._cost_results(base_net=total, degrade_2x_to=total * 0.7),
        neighbour_results=neigh, n_folds=4,
    )
    assert rep.parameter_stability is not None
    assert rep.parameter_stability.canonical_is_isolated_spike
    assert rep.verdict == Verdict.REJECT
    assert ReasonCode.PARAMETER_UNSTABLE in rep.reason_codes
    assert ReasonCode.PARAMETER_STABILITY_NOT_EVALUATED not in rep.reason_codes


# ---------------------------------------------------------------------------
# F -- optional evidence (ablation, default policy) missing is not a fabricated FAIL
# ---------------------------------------------------------------------------
def _real_neighbourhood_report(policy):
    """The D-style shape: a real predeclared parameter neighbourhood (so
    parameter_stability is populated and never the confounding factor),
    isolating whatever `policy` says about ablation."""
    nb = ParameterNeighbourhood(strategy_key="synthetic", canonical_params={"w": 20},
                                neighbour_params=tuple({"w": 20 + d} for d in range(1, 7)))
    fx = vf.fixture_D_broad_plateau(seed=3)
    strong = vf.fixture_B_strong_signal(180, mu=140.0, sigma=180.0, seed=1)
    neigh = trv._neigh_results(strong, fx["neighbours"])
    spec = trv._spec(policy, neighbourhood=nb)
    return trv._assemble_synth(
        spec, policy, strong,
        cost_results=trv._cost_results(base_net=float(strong.sum()), degrade_2x_to=float(strong.sum()) * 0.7),
        neighbour_results=neigh, n_folds=4,
    )


def test_F_optional_ablation_missing_is_neither_fail_nor_not_evaluated_by_default():
    """`require_ablation_mechanism_value` defaults to False (unchanged): a
    missing ablation report under the DEFAULT policy must contribute nothing
    -- no REJECT reason, no NOT_EVALUATED code, PASS remains reachable."""
    pol = TEST_FIXTURE_POLICY
    assert pol.require_ablation_mechanism_value is False
    rep = _real_neighbourhood_report(pol)
    assert rep.ablation_report is None
    assert ReasonCode.ABLATION_NOT_EVALUATED not in rep.reason_codes
    assert rep.verdict == Verdict.PASS, rep.reason_codes
    assert rep.reason_codes == (ReasonCode.ALL_GATES_SATISFIED,)


def test_F2_required_ablation_missing_is_inconclusive_not_a_fabricated_fail():
    """If a policy DOES require ablation (opt-in, default unchanged), a
    missing report must be INCONCLUSIVE/NOT_EVALUATED, never a fabricated
    REJECT, and never PASS."""
    pol = TEST_FIXTURE_POLICY.model_copy(update={"require_ablation_mechanism_value": True})
    rep = _real_neighbourhood_report(pol)
    assert rep.ablation_report is None
    assert rep.verdict == Verdict.INCONCLUSIVE
    assert ReasonCode.ABLATION_NOT_EVALUATED in rep.reason_codes
    assert ReasonCode.ALL_GATES_SATISFIED not in rep.reason_codes
    assert ReasonCode.PARAMETER_UNSTABLE not in rep.reason_codes  # not a fabricated FAIL


# ---------------------------------------------------------------------------
# G -- UI gate badge honesty
# ---------------------------------------------------------------------------
def test_G_parameter_stability_badge_is_not_evaluated_for_the_real_inconclusive_shape():
    state = services.gate_state(
        fail_code="parameter_neighbourhood_unstable",
        reason_codes=["parameter_stability_not_evaluated"],
        verdict="INCONCLUSIVE",
        trial_role="CANONICAL",
    )
    assert state == "NOT_EVALUATED"


def test_G2_parameter_stability_badge_never_shows_pass_even_if_adversarially_paired():
    """Defense in depth at the presentation layer too: even if a "not
    evaluated" code were adversarially paired with a headline PASS +
    ALL_GATES_SATISFIED (a shape `evaluate_policy` can no longer itself
    produce), the UI must still never render PASS for this gate."""
    state = services.gate_state(
        fail_code="parameter_neighbourhood_unstable",
        reason_codes=["all_required_gates_satisfied", "parameter_stability_not_evaluated"],
        verdict="PASS",
        trial_role="CANONICAL",
    )
    assert state == "NOT_EVALUATED"
    assert state != "PASS"


def test_G3_parameter_stability_badge_still_shows_pass_for_a_genuine_complete_pass():
    """No regression: a genuine complete PASS with no not-evaluated code
    still renders PASS for this gate."""
    state = services.gate_state(
        fail_code="parameter_neighbourhood_unstable",
        reason_codes=["all_required_gates_satisfied"],
        verdict="PASS",
        trial_role="CANONICAL",
    )
    assert state == "PASS"


# ---------------------------------------------------------------------------
# H/I -- Research page execution-completion banner honesty
# ---------------------------------------------------------------------------
def _family_report(*, status, member_verdict, reason_codes=()):
    from alpha_agent.agents.orchestrator import (
        FamilyMemberResult,
        FamilyReport,
        MemberDisposition,
    )
    from alpha_agent.registry.enums import AttemptStatus, TrialRole

    member = FamilyMemberResult(
        ordinal=0,
        experiment_identity="experiment1:" + "c" * 64,
        experiment_id="TEST-EXP-1",
        trial_role=TrialRole.CANONICAL,
        disposition=MemberDisposition.VALID_EVIDENCE,
        attempt_status=AttemptStatus.VALID,
        attempt_ordinal=1,
        n_execution_attempts=1,
        final_verdict=member_verdict,
        reason_codes=reason_codes,
    )
    return FamilyReport(
        family_id="fam:test", family_stem="test", generation=0, parent_family_id=None,
        manifest_fingerprint="manifest1:test", status=status, predeclared_family_size=1,
        fdr_q_threshold=0.10, member_results=(member,),
        unresolved_invalid_identities=(), passing_experiment_ids=(),
    )


def test_H_reject_banner_uses_neutral_not_success_wording():
    from alpha_agent.agents.orchestrator import FamilyStatus
    from alpha_agent.registry.enums import RegistryVerdict
    from alpha_agent.ui.views import research as research_view

    report = _family_report(status=FamilyStatus.FINALIZED, member_verdict=RegistryVerdict.REJECT,
                            reason_codes=("fdr_qvalue_above_threshold",))
    with patch.object(research_view.st, "success") as success, \
         patch.object(research_view.st, "info") as info, \
         patch.object(research_view.st, "warning") as warning, \
         patch.object(research_view.st, "error") as error:
        research_view._render_family_status_banner(report)
    success.assert_not_called()
    info.assert_called_once()
    warning.assert_not_called()
    error.assert_not_called()


def test_I_incomplete_evidence_banner_is_a_warning_not_a_pass_look():
    from alpha_agent.agents.orchestrator import FamilyStatus
    from alpha_agent.registry.enums import RegistryVerdict
    from alpha_agent.ui.views import research as research_view

    report = _family_report(status=FamilyStatus.FINALIZED, member_verdict=RegistryVerdict.INCONCLUSIVE,
                            reason_codes=("parameter_stability_not_evaluated",))
    with patch.object(research_view.st, "success") as success, \
         patch.object(research_view.st, "warning") as warning:
        research_view._render_family_status_banner(report)
    success.assert_not_called()
    warning.assert_called_once()


def test_I2_genuine_pass_banner_is_still_green():
    from alpha_agent.agents.orchestrator import FamilyStatus
    from alpha_agent.registry.enums import RegistryVerdict
    from alpha_agent.ui.views import research as research_view

    report = _family_report(status=FamilyStatus.FINALIZED, member_verdict=RegistryVerdict.PASS,
                            reason_codes=("all_required_gates_satisfied",))
    with patch.object(research_view.st, "success") as success:
        research_view._render_family_status_banner(report)
    success.assert_called_once()


def test_I3_incomplete_not_adjudicated_is_an_error_not_a_scientific_verdict():
    from alpha_agent.agents.orchestrator import FamilyStatus
    from alpha_agent.ui.views import research as research_view

    report = _family_report(status=FamilyStatus.INCOMPLETE_NOT_ADJUDICATED, member_verdict=None)
    with patch.object(research_view.st, "error") as error, \
         patch.object(research_view.st, "success") as success:
        research_view._render_family_status_banner(report)
    error.assert_called_once()
    success.assert_not_called()


# ---------------------------------------------------------------------------
# J -- the real committed NQ TSMOM REJECT is untouched
# ---------------------------------------------------------------------------
def test_J_real_committed_nq_tsmom_canonical_reject_is_unchanged():
    rows = services.list_experiments(strategy_family="tsmom", root_symbol="NQ", trial_role=None)
    canonical = [r for r in rows if r["trial_role"] == "CANONICAL"]
    assert canonical, "expected at least one committed NQ tsmom canonical row"
    assert all(r["verdict"] == "REJECT" for r in canonical)


def test_reliability_policy_fingerprint_is_byte_identical_after_the_fix():
    """No new Pydantic field was added to ReliabilityPolicy -- the fix lives
    entirely in evaluate_policy's control flow -- so the policy's own
    identity (embedded in `experiment_identity` and the "one evaluation
    plane" check) is unchanged for every existing and future caller."""
    assert (
        ReliabilityPolicy().identity()
        == "valreliabilitypolicy1:b0682e86917c7b84550dc4d578926b8724fc95e18edb035242241ea008ec3705"
    )
