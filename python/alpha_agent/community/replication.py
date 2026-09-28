"""Phase 8 -- "Replicate", not "Like" (prompt section 4).

The community's one core interaction: an independent researcher reproduces a
:class:`~alpha_agent.community.schemas.Contribution`'s claim with their OWN
registry evidence, and the comparison preserves every dimension the prompt
names -- contributor, experiment identity, dataset/universe, context,
factor/strategy identity, result, validation, provenance -- as separate,
individually inspectable facts (:class:`~alpha_agent.community.schemas.ReplicationComparison`),
never collapsed into one "reproduced: yes/no" boolean.

Threat-model defense (prompt section 7, "copied strategies"):
:func:`create_replication` enforces TWO independence conditions, either of
which refuses the replication (:class:`NotIndependentReplicationError`):
citing the EXACT same ``experiment_identity`` as the original (re-citing the
same run, not an independent reproduction), or coming from the SAME
(self-attested) contributor as the original contribution -- under this MVP's
identity model (no authentication anywhere in this platform), a contributor
cannot "independently replicate" their own work merely by pointing at a
second experiment_id.

:func:`evidence_maturity` is the ONE place `EvidenceMaturity` is ever computed
-- a conservative, documented, monotone-with-evidence rule (prompt section 3:
"use stronger labels only if scientific policy supports them"). It mirrors
the registry's own ``RegistryVerdict`` for the un-replicated/REJECT/
INCONCLUSIVE cases exactly, and only ever upgrades to ``EMERGING_EVIDENCE``
on >=2 independently CONFIRMING replications -- this is the real data source
Phase 3's `IndependentReplicationLevel.NOT_AVAILABLE` had no source for.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from alpha_agent.community.contributions import build_evidence_reference, contributor_ref
from alpha_agent.community.schemas import (
    COMMUNITY_SCHEMA,
    Contribution,
    EvidenceMaturity,
    EvidenceReference,
    ReplicationComparison,
    ReplicationOutcome,
    ReplicationRecord,
)
from alpha_agent.community.store import CommunityStore
from alpha_agent.registry.enums import RegistryVerdict
from alpha_agent.registry.sqlite_registry import ExperimentRegistry

__all__ = [
    "NotIndependentReplicationError",
    "compare_replication",
    "create_replication",
    "evidence_maturity",
    "list_replications",
]


class NotIndependentReplicationError(RuntimeError):
    """A replication failed one of the two independence conditions
    :func:`create_replication` enforces: same registry ``experiment_identity``
    as the original (re-citing a run, not independently reproducing it), or
    same (self-attested) ``contributor_id`` as the original contribution's own
    contributor (the original author "replicating" their own work) --
    prompt section 7's "threat-model ... copied strategies" covers both."""


def compare_replication(contribution: Contribution, evidence: EvidenceReference) -> ReplicationComparison:
    """Build the full, transparent, never-collapsed comparison between a
    contribution's original evidence and one replicator's own evidence."""
    orig = contribution.evidence
    same_root = orig.root_symbol == evidence.root_symbol
    same_domain = orig.asset_domain == evidence.asset_domain
    same_family = orig.strategy_family == evidence.strategy_family
    same_window = (
        (orig.market_window_start, orig.market_window_end)
        == (evidence.market_window_start, evidence.market_window_end)
    )
    same_policy = bool(orig.reliability_policy_fingerprint) and (
        orig.reliability_policy_fingerprint == evidence.reliability_policy_fingerprint
    )
    same_dataset = bool(orig.dataset_fingerprint) and orig.dataset_fingerprint == evidence.dataset_fingerprint

    same_factor: bool | None = None
    if contribution.mechanism is not None:
        if not same_family:
            same_factor = False
        else:
            from alpha_agent.alpha_memory.factor_identity import compute_factor_identity

            same_factor = (
                compute_factor_identity(contribution.mechanism, (orig.strategy_family,)).factor_identity
                == compute_factor_identity(contribution.mechanism, (evidence.strategy_family,)).factor_identity
            )

    if not same_root or not same_family:
        outcome = ReplicationOutcome.NOT_COMPARABLE
        notes = (
            "Different root symbol or strategy family -- this is not an independent replication of the "
            "same claim, and is preserved as its own distinct evidence rather than merged."
        )
    elif orig.headline_verdict is None or evidence.headline_verdict is None:
        outcome = ReplicationOutcome.PARTIAL
        notes = "At least one side has no adjudicated headline verdict yet."
    elif orig.headline_verdict == evidence.headline_verdict:
        outcome = ReplicationOutcome.CONFIRMS
        notes = f"Both independent executions reached the same headline verdict: {orig.headline_verdict.value}."
    else:
        outcome = ReplicationOutcome.CONFLICTS
        notes = (
            f"Original reached {orig.headline_verdict.value}; replication reached "
            f"{evidence.headline_verdict.value}. Both are preserved -- neither is discarded."
        )

    return ReplicationComparison(
        same_root_symbol=same_root,
        same_asset_domain=same_domain,
        same_strategy_family=same_family,
        same_factor_identity=same_factor,
        same_market_window=same_window,
        same_reliability_policy=same_policy,
        same_dataset_fingerprint=same_dataset,
        original_verdict=orig.headline_verdict,
        replication_verdict=evidence.headline_verdict,
        outcome=outcome,
        notes=notes,
    )


def create_replication(
    store: CommunityStore,
    registry: ExperimentRegistry,
    *,
    contribution: Contribution,
    replicator_display_name: str,
    experiment_id: str,
    notes: str = "",
) -> ReplicationRecord:
    replicator = contributor_ref(replicator_display_name)
    if replicator.contributor_id == contribution.contributor.contributor_id:
        raise NotIndependentReplicationError(
            f"{replicator.display_name!r} is the SAME (self-attested) contributor as the original "
            "contribution's own contributor -- an independent replication must come from a different "
            "contributor identity, never the original author replicating their own work."
        )
    evidence = build_evidence_reference(registry, experiment_id)
    if evidence.experiment_identity == contribution.evidence.experiment_identity:
        raise NotIndependentReplicationError(
            f"{experiment_id} is the SAME registry experiment_identity as the original contribution's own "
            "evidence -- an independent replication must supply the replicator's own, separately executed "
            "evidence, never re-cite the original run."
        )
    comparison = compare_replication(contribution, evidence)
    record = ReplicationRecord(
        schema_version=COMMUNITY_SCHEMA,
        replication_id=f"replic-{uuid.uuid4().hex[:16]}",
        contribution_id=contribution.contribution_id,
        replicator=replicator,
        evidence=evidence,
        comparison=comparison,
        notes=notes,
        created_at=datetime.now(UTC).isoformat(),
    )
    store.insert_replication(
        {
            "replication_id": record.replication_id,
            "contribution_id": record.contribution_id,
            "schema_version": record.schema_version,
            "replicator_id": record.replicator.contributor_id,
            "replicator_display_name": record.replicator.display_name,
            "evidence": record.evidence.model_dump(mode="json"),
            "comparison": record.comparison.model_dump(mode="json"),
            "notes": record.notes,
            "created_at": record.created_at,
        }
    )
    return record


def list_replications(store: CommunityStore, contribution_id: str | None = None) -> list[ReplicationRecord]:
    from alpha_agent.community.schemas import ContributorRef

    out = []
    for row in store.list_replications(contribution_id):
        out.append(
            ReplicationRecord(
                schema_version=row["schema_version"],
                replication_id=row["replication_id"],
                contribution_id=row["contribution_id"],
                replicator=ContributorRef(
                    contributor_id=row["replicator_id"], display_name=row["replicator_display_name"],
                ),
                evidence=EvidenceReference.model_validate(row["evidence"]),
                comparison=ReplicationComparison.model_validate(row["comparison"]),
                notes=row["notes"],
                created_at=row["created_at"],
            )
        )
    return out


def evidence_maturity(
    contribution: Contribution, replications: Sequence[ReplicationRecord]
) -> tuple[EvidenceMaturity, str]:
    """The one, documented, conservative maturity rule (see module docstring).
    Order of precedence matters: the original's own registry verdict and any
    conflicting replication both take priority over a confirming-replication
    count, so a REJECT/INCONCLUSIVE claim can never be laundered into
    EMERGING_EVIDENCE merely by accumulating confirmations of something else."""
    verdict = contribution.evidence.headline_verdict
    if verdict is None:
        return (
            EvidenceMaturity.PROPOSED,
            "The original contribution's own registry evidence has no adjudicated headline verdict yet.",
        )
    if verdict == RegistryVerdict.INCONCLUSIVE:
        return EvidenceMaturity.INCONCLUSIVE, "The original contribution's own registry verdict is INCONCLUSIVE."
    if any(r.comparison.outcome == ReplicationOutcome.CONFLICTS for r in replications):
        return (
            EvidenceMaturity.INCONCLUSIVE,
            "At least one independent replication reached a conflicting headline verdict.",
        )
    if verdict == RegistryVerdict.REJECT:
        return (
            EvidenceMaturity.REJECTED,
            "The original contribution's own registry verdict is REJECT, and no replication overturned it.",
        )
    confirms = [r for r in replications if r.comparison.outcome == ReplicationOutcome.CONFIRMS]
    if len(confirms) >= 2:
        return (
            EvidenceMaturity.EMERGING_EVIDENCE,
            f"{len(confirms)} independent replications confirm the original {verdict.value} result.",
        )
    if len(confirms) == 1:
        return (
            EvidenceMaturity.REPLICATING,
            (
                f"One independent replication confirms the original {verdict.value} result -- not yet "
                "emerging evidence."
            ),
        )
    return EvidenceMaturity.PROPOSED, "No independent replication confirms this result yet."
