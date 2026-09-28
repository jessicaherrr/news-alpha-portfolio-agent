"""Registry smoke tests.

Phase 14 replaced the pre-existing flat ``INSERT OR REPLACE`` table with the
append-only normalized schema. This file keeps the cheap end-to-end checks that
a rejected experiment is *recorded and kept*; the full Phase 14 invariant suite
lives in ``test_phase_14_registry.py``.
"""
from __future__ import annotations

from alpha_agent.core.instrument import AssetDomain
from alpha_agent.registry import (
    ExperimentRegistry,
    ExperimentStatus,
    FailureClass,
    MarketWindow,
    RegistryVerdict,
    TrialRole,
)
from alpha_agent.registry.enums import FailureScope
from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
from alpha_agent.registry.models import ExperimentRecord, FailureRecord, ResultRecord

_PARAMS = {"zscore_window": 20, "entry_z": 2.0, "exit_z": 0.5, "size": 1}
_FINGERPRINTS = {
    "dataset_fingerprint": "valdataset2:abc",
    "split_identity": "splitplan1:abc",
    "validation_spec_fingerprint": "validationspec1:abc",
    "reliability_policy_fingerprint": "valreliabilitypolicy1:abc",
    "execution_config_identity": "execconfig1:abc",
    "cost_config_identity": "costconfig1:abc",
    "risk_identity": "riskconfig1:abc",
}


def _record() -> ExperimentRecord:
    variant = parameter_variant_identity(_PARAMS)
    identity = experiment_identity(
        strategy_fingerprint="stratdsl1:mr",
        strategy_family="mean_reversion",
        root_symbol="ES",
        parameter_variant_identity=variant,
        **_FINGERPRINTS,
    )
    return ExperimentRecord(
        experiment_identity=identity,
        experiment_id="ES__MEAN_REVERSION__CANONICAL__TEST",
        display_name="ES / MEAN_REVERSION / canonical / TEST",
        created_at="2026-01-01T00:00:00+00:00",
        phase="TEST",
        status=ExperimentStatus.COMPLETED,
        root_symbol="ES",
        asset_domain=AssetDomain.FUTURES,
        strategy_family="mean_reversion",
        strategy_fingerprint="stratdsl1:mr",
        strategy_id="TEST-MR",
        strategy_spec_json={"params": _PARAMS},
        market_window=MarketWindow(
            label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31"
        ),
        trial_role=TrialRole.CANONICAL,
        parameter_variant_identity=variant,
        parameter_variant_label="canonical",
        **_FINGERPRINTS,
    )


def test_registry_records_and_keeps_a_rejected_experiment(tmp_path):
    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reg:
        exp = _record()
        reg.insert_experiment(
            exp,
            ResultRecord(
                experiment_identity=exp.experiment_identity,
                headline_verdict=RegistryVerdict.REJECT,
                reason_codes=("cost_stress_degradation_exceeds_limit",),
                daily_sharpe=-0.2,
            ),
        )
        reg.record_failure(
            FailureRecord(
                failure_id="F__ES__MEAN_REVERSION__COST_FRAGILITY",
                scope=FailureScope.EXPERIMENT,
                experiment_identity=exp.experiment_identity,
                failure_class=FailureClass.COST_FRAGILITY,
                failure_code="cost_stress_degradation_exceeds_limit",
                summary="fails cost stress",
                mechanism="net PnL degrades beyond the policy limit at 2.0x costs",
                created_at="2026-01-01T00:00:00+00:00",
                root_symbol="ES",
                strategy_family="mean_reversion",
            )
        )

        assert reg.count_experiments() == 1
        view = reg.get("ES__MEAN_REVERSION__CANONICAL__TEST")
        assert view.verdict is RegistryVerdict.REJECT
        assert reg.failures(failure_class=FailureClass.COST_FRAGILITY)[0].summary == (
            "fails cost stress"
        )
        # a rejected experiment is evidence: still there on reopen
        digest = reg.content_digest()

    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reopened:
        assert reopened.count_experiments() == 1
        assert reopened.content_digest() == digest
