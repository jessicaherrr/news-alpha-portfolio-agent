"""Shared fixture-builder helpers for the Phase 8 Community Alpha Network
test files. Mirrors `test_phase6_factor_library.py`'s own
`_identity`/`_experiment`/`_result` pattern -- real typed
`ExperimentRecord`/`ResultRecord` objects, real `experiment_identity()`
computation, inserted via `ExperimentRegistry.insert_experiment` into a
throwaway `tmp_path` registry. Never the real production registry.

`make_pair` builds TWO experiments for the SAME (root, family) with
different ``dataset_vintage`` tags (folded into the dataset/split
fingerprints) -- a fixture stand-in for "two independent researchers each
built their own dataset snapshot and ran the same hypothesis", used by the
Phase 8 replication tests. This is disclosed, fixture-only data: this
repository's real registry has exactly one CANONICAL trial per (root,
family), so no genuine second-researcher execution exists yet to replicate
against.
"""
from __future__ import annotations

from alpha_agent.registry.enums import AssetDomain, ExperimentStatus, RegistryVerdict, TrialRole
from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
from alpha_agent.registry.models import ExperimentRecord, MarketWindow, ResultRecord
from alpha_agent.registry.sqlite_registry import ExperimentRegistry

WINDOW = MarketWindow(label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31")


def _fingerprints(vintage: str) -> dict:
    return {
        "dataset_fingerprint": f"valdataset2:ds-{vintage}",
        "split_identity": f"splitplan1:sp-{vintage}",
        "validation_spec_fingerprint": "validationspec1:vs",
        "reliability_policy_fingerprint": "valreliabilitypolicy1:rp",
        "execution_config_identity": "execconfig1:ex",
        "cost_config_identity": "costconfig1:co",
        "risk_identity": "riskconfig1:ri",
        "feature_spec_fingerprint": "featset1:fs",
    }


def make_experiment(
    *,
    root: str,
    family: str,
    params: dict,
    vintage: str = "a",
    asset_domain: AssetDomain = AssetDomain.FUTURES,
    trial_role: TrialRole = TrialRole.CANONICAL,
    label: str = "canonical",
    window: MarketWindow = WINDOW,
) -> ExperimentRecord:
    fp = _fingerprints(vintage)
    identity = experiment_identity(
        strategy_fingerprint=f"stratdsl1:{family}:{root}",
        strategy_family=family,
        root_symbol=root,
        parameter_variant_identity=parameter_variant_identity(params),
        **fp,
    )
    return ExperimentRecord(
        experiment_identity=identity,
        experiment_id=f"{root}__{family.upper()}__{label.upper()}__{vintage.upper()}__TEST",
        display_name=f"{root} / {family} / {label} / vintage {vintage} / TEST",
        created_at="2026-01-01T00:00:00+00:00",
        phase="TEST",
        status=ExperimentStatus.COMPLETED,
        root_symbol=root,
        asset_domain=asset_domain,
        strategy_family=family,
        strategy_fingerprint=f"stratdsl1:{family}:{root}",
        strategy_id="TEST-STRAT",
        strategy_spec_json={
            "params": params,
            "signal_cadence": "daily_trading_day",
            "execution_cadence": "native_1m_raw_contract",
        },
        market_window=window,
        trial_role=trial_role,
        parameter_variant_identity=parameter_variant_identity(params),
        parameter_variant_label=label,
        **fp,
    )


def make_result(
    identity: str, *, verdict: RegistryVerdict, reason_codes: tuple[str, ...] = ("fdr_qvalue_above_threshold",),
    holdout_eligible: bool = False,
) -> ResultRecord:
    return ResultRecord(
        experiment_identity=identity, headline_verdict=verdict, reason_codes=reason_codes,
        net_pnl_usd=1.0, daily_sharpe=0.01, holdout_eligible=holdout_eligible,
    )


def insert(
    registry: ExperimentRegistry,
    *,
    root: str,
    family: str,
    params: dict,
    verdict: RegistryVerdict,
    vintage: str = "a",
    asset_domain: AssetDomain = AssetDomain.FUTURES,
    trial_role: TrialRole = TrialRole.CANONICAL,
    label: str = "canonical",
    reason_codes: tuple[str, ...] = ("fdr_qvalue_above_threshold",),
) -> ExperimentRecord:
    """Build + insert one fixture experiment/result pair; returns the
    inserted `ExperimentRecord` (its `.experiment_id` is what a Community
    contribution/replication cites)."""
    exp = make_experiment(
        root=root, family=family, params=params, vintage=vintage,
        asset_domain=asset_domain, trial_role=trial_role, label=label,
    )
    registry.insert_experiment(exp, make_result(exp.experiment_identity, verdict=verdict, reason_codes=reason_codes))
    return exp
