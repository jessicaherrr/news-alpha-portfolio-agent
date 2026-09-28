"""Deterministic semantic identity for a registry experiment (section 8).

An ``experiment_identity`` is a fingerprint over the *meaningful scientific
inputs only*. It deliberately excludes wall-clock time, random UUIDs, the code
commit, the human display name, the trial ``role`` label and the database row
id: two runs of the same hypothesis under the same configuration are the same
experiment however they were labelled or when they were executed.

**Identity is a PRE-RUN quantity** (schema v2). Every component is known from
the hypothesis and its configuration before a single bar is executed, so a
future Research Agent can compute a proposal's identity and ask "has this
already been run?" without compiling a schedule or running a backtest.

Two fields that Phase 14 v1 conflated with identity are therefore *provenance*,
not identity:

``target_schedule_hash``
    post-compilation audit evidence -- what the compiled schedule actually was.
``report_fingerprint``
    result-artifact provenance -- which ``ValidationReport`` produced the row.

Both are recorded on the experiment and both participate in the *content*
fingerprint (so a conflicting value fails loudly, section 5), but neither
changes what experiment this is. Including ``target_schedule_hash`` in the
identity would let the same hypothesis, recompiled, masquerade as a new
experiment and slip past exact-duplicate detection.

Three layers are kept strictly separate:

``experiment_identity``
    the SHA-256 scientific identity (this module);
``experiment_id``
    a stable, deterministic, human-readable slug (this module) -- convenient,
    never authoritative;
``row_id``
    the SQLite autoincrement primary key -- storage only.
"""
from __future__ import annotations

import re

from alpha_agent.validation.fingerprint import fingerprint

IDENTITY_SCHEMA = "experiment-identity/2"
#: v1 kept identity == pre-run inputs PLUS ``target_schedule_hash``. Superseded
#: by Phase 14.1; recognised only so a stale row can be rejected loudly.
IDENTITY_SCHEMA_V1 = "experiment-identity/1"
PARAMETER_VARIANT_SCHEMA = "parameter-variant/1"
EXECUTION_CONFIG_SCHEMA = "execution-config-identity/1"
COST_CONFIG_SCHEMA = "cost-config-identity/1"
RISK_CONFIG_SCHEMA = "risk-config-identity/1"

_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def parameter_variant_identity(params: dict) -> str:
    """Identity of one concrete parameter set (canonical or neighbour)."""
    return fingerprint(
        "paramvariant1", {"schema": PARAMETER_VARIANT_SCHEMA, "params": params}
    )


def execution_config_identity(
    *,
    execution_assumptions: dict,
    contract_economics: dict,
    economics_rule: str,
) -> str:
    """Identity of the execution plane a result was produced under.

    ``contract_economics`` covers **every root in the multiple-testing family**,
    not just this experiment's root. That is not decoration: BH q-values and the
    DSR benchmark are computed *jointly* over the whole 107-trial family, so a
    mis-derived point value on any single root changes the global statistics of
    every trial in the family (Phase 13.5C ``CONTRACT_ECONOMICS_CORRECTION.json``
    ``global_statistics_superseded``). Including the family-wide economics is
    what makes the pre-correction and corrected experiments distinct identities
    rather than a silent overwrite of the same one.
    """
    return fingerprint(
        "execconfig1",
        {
            "schema": EXECUTION_CONFIG_SCHEMA,
            "execution_assumptions": execution_assumptions,
            "family_contract_economics": contract_economics,
            "economics_rule": economics_rule,
        },
    )


def cost_config_identity(*, base_commission_per_contract_usd: float,
                         base_slippage_ticks: float, base_spread_ticks: float,
                         scenarios: list[dict], commission_schedule: list[dict] | None = None) -> str:
    """``commission_schedule`` (additive, News Alpha Phase H acceptance patch):
    the explicit per-root commission rates with their provenance. ``None`` --
    every pre-existing caller -- leaves the identity byte-identical."""
    payload = {
        "schema": COST_CONFIG_SCHEMA,
        "base_commission_per_contract_usd": base_commission_per_contract_usd,
        "base_slippage_ticks": base_slippage_ticks,
        "base_spread_ticks": base_spread_ticks,
        "scenarios": scenarios,
    }
    if commission_schedule is not None:
        payload["commission_schedule"] = commission_schedule
    return fingerprint("costconfig1", payload)


def risk_config_identity(risk_assumptions: dict) -> str:
    return fingerprint(
        "riskconfig1", {"schema": RISK_CONFIG_SCHEMA, "risk": risk_assumptions}
    )


def experiment_identity(
    *,
    strategy_fingerprint: str,
    strategy_family: str,
    root_symbol: str,
    parameter_variant_identity: str,
    dataset_fingerprint: str,
    split_identity: str,
    validation_spec_fingerprint: str,
    reliability_policy_fingerprint: str,
    execution_config_identity: str,
    cost_config_identity: str,
    risk_identity: str,
    feature_spec_fingerprint: str | None = None,
) -> str:
    """The canonical, PRE-RUN semantic identity of one experiment.

    Deliberately absent: ``phase``, ``created_at``, ``code_commit``,
    ``trial_role``, ``display_name`` -- those describe *when / how / under what
    label* the hypothesis was run, not *which* hypothesis under *which*
    configuration was run -- and, since schema v2, ``target_schedule_hash`` and
    ``report_fingerprint``, which are execution/result provenance and only exist
    *after* the run.
    """
    return fingerprint(
        "experiment1",
        {
            "schema": IDENTITY_SCHEMA,
            "strategy_fingerprint": strategy_fingerprint,
            "strategy_family": strategy_family,
            "root_symbol": root_symbol,
            "parameter_variant_identity": parameter_variant_identity,
            "dataset_fingerprint": dataset_fingerprint,
            "split_identity": split_identity,
            "validation_spec_fingerprint": validation_spec_fingerprint,
            "reliability_policy_fingerprint": reliability_policy_fingerprint,
            "execution_config_identity": execution_config_identity,
            "cost_config_identity": cost_config_identity,
            "risk_identity": risk_identity,
            "feature_spec_fingerprint": feature_spec_fingerprint or "",
        },
    )


def _slug(text: str) -> str:
    return _SLUG_RE.sub("_", str(text)).strip("_").upper()


def friendly_experiment_id(
    *,
    root_symbol: str,
    strategy_family: str,
    variant_label: str,
    split_label: str,
    lineage_tag: str = "",
) -> str:
    """Deterministic human-readable id, e.g.
    ``NQ__TSMOM__CANONICAL__VALIDATION_2023_2024``.

    Convenient for humans and CLI lookup. It is **not** the semantic identity
    and never participates in duplicate detection.
    """
    parts = [_slug(root_symbol), _slug(strategy_family), _slug(variant_label),
             _slug(split_label)]
    if lineage_tag:
        parts.append(_slug(lineage_tag))
    return "__".join(p for p in parts if p)
