"""Phase 8 -- creating and reading :class:`~alpha_agent.community.schemas.Contribution` objects.

Every contribution is anchored to one real, already-committed registry
experiment (:class:`~alpha_agent.registry.sqlite_registry.ExperimentView`) --
this module never accepts a free-text verdict, PnL, or provenance claim.
Two structural threat-model defenses (prompt section 7) live here, at
creation time, so nothing downstream needs to re-check them:

* **Cherry-picking / hidden parameter search** -- :func:`build_evidence_reference`
  refuses any ``experiment_id`` whose ``trial_role`` is not ``CANONICAL``. A
  contributor cannot cite a hand-picked best-of-N parameter neighbour or
  ablation as if it were the family's own headline result.
* **Leakage** -- every :class:`~alpha_agent.community.schemas.EvidenceReference`
  is swept by :func:`alpha_agent.registry.holdout_guard.assert_no_holdout_market_data`
  before it is ever persisted, and its ``market_window_end`` is independently
  checked against the same 2025-01-01 boundary CLAUDE.md fixes everywhere
  else -- defense in depth on top of the registry's own write-time guard.

Duplicate / novelty transparency (also prompt section 7) is a
COMMUNITY-level check (:func:`_detect_duplicate_contribution`), deliberately
NOT :func:`alpha_agent.research_recycling.assess_research_recycling`: that
function answers "has this SCIENTIFIC hypothesis already been RUN", and a
contribution by construction always cites an experiment that has already been
run (it would trivially "duplicate itself" every time). What Community needs
to detect instead is "has this evidence, or this (mechanism, root, strategy
family), already been SHARED to the community before" -- a duplicate
CONTRIBUTION, not a duplicate experiment. A near-duplicate contribution is
never silently blocked, only flagged on the record
(``Contribution.recycling_decision``/``recycling_explanation``).
"""
from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

from alpha_agent.community.schemas import (
    COMMUNITY_SCHEMA,
    Contribution,
    ContributionKind,
    ContributorRef,
    EvidenceReference,
    VisibilityLevel,
)
from alpha_agent.community.store import CommunityStore
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.enums import TrialRole
from alpha_agent.registry.holdout_guard import assert_no_holdout_market_data
from alpha_agent.registry.sqlite_registry import ExperimentRegistry

__all__ = [
    "CherryPickedTrialError",
    "CommunityHoldoutError",
    "build_evidence_reference",
    "contributor_ref",
    "create_contribution",
    "get_contribution",
    "list_contributions",
]

#: The same locked-holdout boundary CLAUDE.md fixes everywhere else -- an
#: EXCLUSIVE end date; a market window ending on or after this date is
#: refused, never silently truncated.
_HOLDOUT_BOUNDARY = "2025-01-01"


class CherryPickedTrialError(RuntimeError):
    """A contribution tried to cite a non-CANONICAL trial (prompt section 7:
    cherry-picking / hidden parameter search)."""


class CommunityHoldoutError(RuntimeError):
    """A contribution's cited evidence touches the locked 2025 holdout
    (CLAUDE.md STOP condition 2; prompt section 7: leakage)."""


def contributor_ref(display_name: str) -> ContributorRef:
    """A stable, normalized contributor identity. There is no authentication
    anywhere in this local tool -- this is a SELF-ATTESTED identity (two
    contributions typed under the same display name resolve to the same
    contributor); ``identity_verified`` stays ``False``."""
    slug = re.sub(r"[^a-z0-9]+", "-", display_name.strip().lower()).strip("-")
    name = display_name.strip() or "Anonymous"
    return ContributorRef(contributor_id=slug or "anonymous", display_name=name)


def build_evidence_reference(registry: ExperimentRegistry, experiment_id: str) -> EvidenceReference:
    """Look up ``experiment_id`` in the real registry and build the ONE typed
    evidence object a contribution/replication is ever allowed to carry.
    Raises :class:`CherryPickedTrialError` for a non-CANONICAL trial and
    :class:`CommunityHoldoutError` for any evidence touching the locked
    holdout -- both fail loudly rather than silently coercing the input."""
    view = registry.get(experiment_id)
    if view.experiment.trial_role != TrialRole.CANONICAL:
        raise CherryPickedTrialError(
            f"{experiment_id} is a {view.experiment.trial_role.value} trial. A community contribution must "
            "cite the strategy family's own CANONICAL trial, never a hand-picked parameter neighbour, "
            "ablation, or variant (prompt 8 section 7: cherry-picking / hidden parameter search)."
        )
    mw = view.experiment.market_window
    ref = EvidenceReference(
        experiment_id=view.experiment_id,
        experiment_identity=view.experiment_identity,
        root_symbol=view.experiment.root_symbol,
        asset_domain=view.experiment.asset_domain,
        strategy_family=view.experiment.strategy_family,
        trial_role=view.experiment.trial_role,
        headline_verdict=view.verdict,
        reason_codes=view.result.reason_codes if view.result else (),
        holdout_eligible=bool(view.result.holdout_eligible) if view.result else False,
        market_window_label=mw.label,
        market_window_start=mw.start_date,
        market_window_end=mw.end_date,
        reliability_policy_fingerprint=view.experiment.reliability_policy_fingerprint,
        dataset_fingerprint=view.experiment.dataset_fingerprint,
        code_commit=view.experiment.code_commit,
        strategy_params=dict(view.experiment.strategy_spec_json.get("params", {})),
    )
    # Explicit, targeted check FIRST (a clear, community-specific message for
    # the expected case), then the generic recursive guard as a backstop over
    # the WHOLE payload -- including `strategy_params`, which this explicit
    # check never inspects.
    if ref.market_window_end and ref.market_window_end >= _HOLDOUT_BOUNDARY:
        raise CommunityHoldoutError(
            f"{experiment_id}'s market window ends {ref.market_window_end}, on or after the locked "
            f"{_HOLDOUT_BOUNDARY} holdout boundary -- refusing to build community evidence from it."
        )
    assert_no_holdout_market_data(ref.model_dump(mode="json"))
    return ref


def _detect_duplicate_contribution(
    store: CommunityStore,
    *,
    mechanism: EconomicMechanism,
    root_symbol: str,
    strategy_family: str,
    experiment_identity: str,
    novelty_notes: str,
) -> tuple[str, str]:
    """Has this evidence, or this (mechanism, root, strategy family), already
    been shared to the community before? Scans every EXISTING contribution
    in the store (any visibility -- a contributor re-sharing their own prior
    private work as if it were new is caught the same way as two different
    contributors converging on the same idea). Never blocks -- see module
    docstring."""
    existing = [
        c for c in list_contributions(store)
        if c.mechanism == mechanism and c.root_symbol == root_symbol and c.evidence.strategy_family == strategy_family
    ]
    if not existing:
        return (
            "RECONSIDER",
            (
                "No prior community contribution exists for this mechanism/root/strategy family -- novel "
                "to the community."
            ),
        )
    exact = next((c for c in existing if c.evidence.experiment_identity == experiment_identity), None)
    if exact is not None and not novelty_notes.strip():
        return (
            "DEPRIORITIZE_NO_NOVELTY",
            (
                f"This exact registry evidence has already been contributed as {exact.contribution_id} -- "
                "consider replicating it instead of re-contributing the same evidence. This is a "
                "transparency flag, not a block."
            ),
        )
    return (
        "RECONSIDER",
        (
            f"{len(existing)} prior community contribution(s) exist for this mechanism/root/strategy "
            "family, but this one cites different evidence (or states novelty) -- allowed, and preserved "
            "as its own distinct record."
        ),
    )


def create_contribution(
    store: CommunityStore,
    registry: ExperimentRegistry,
    *,
    kind: ContributionKind,
    visibility: VisibilityLevel,
    contributor_display_name: str,
    experiment_id: str,
    title: str,
    notes: str = "",
    mechanism: EconomicMechanism | None = None,
    novelty_notes: str = "",
) -> Contribution:
    """Create and persist one :class:`Contribution`. ``mechanism`` is
    optional (a ``RESEARCH_NOTES``/``EVIDENCE_BUNDLE`` contribution may not
    claim one), but the community duplicate-contribution check only runs
    when it is supplied -- ``mechanism`` is exactly what
    :func:`_detect_duplicate_contribution` needs to identify the hypothesis
    being (re-)shared."""
    evidence = build_evidence_reference(registry, experiment_id)

    recycling_decision = "RECONSIDER"
    recycling_explanation = (
        "No economic mechanism was declared for this contribution -- the community duplicate-contribution "
        "check needs a mechanism to identify the hypothesis, so none was run."
    )
    if mechanism is not None:
        recycling_decision, recycling_explanation = _detect_duplicate_contribution(
            store, mechanism=mechanism, root_symbol=evidence.root_symbol,
            strategy_family=evidence.strategy_family, experiment_identity=evidence.experiment_identity,
            novelty_notes=novelty_notes,
        )

    contribution = Contribution(
        schema_version=COMMUNITY_SCHEMA,
        contribution_id=f"contrib-{uuid.uuid4().hex[:16]}",
        kind=kind,
        visibility=visibility,
        contributor=contributor_ref(contributor_display_name),
        mechanism=mechanism,
        root_symbol=evidence.root_symbol,
        title=title.strip() or "Untitled contribution",
        notes=notes,
        novelty_notes=novelty_notes,
        evidence=evidence,
        created_at=datetime.now(UTC).isoformat(),
        recycling_decision=recycling_decision,
        recycling_explanation=recycling_explanation,
    )
    store.insert_contribution(
        {
            "contribution_id": contribution.contribution_id,
            "schema_version": contribution.schema_version,
            "kind": contribution.kind.value,
            "visibility": contribution.visibility.value,
            "contributor_id": contribution.contributor.contributor_id,
            "contributor_display_name": contribution.contributor.display_name,
            "mechanism": contribution.mechanism.value if contribution.mechanism else None,
            "root_symbol": contribution.root_symbol,
            "title": contribution.title,
            "notes": contribution.notes,
            "novelty_notes": contribution.novelty_notes,
            "evidence": contribution.evidence.model_dump(mode="json"),
            "created_at": contribution.created_at,
            "recycling_decision": contribution.recycling_decision,
            "recycling_explanation": contribution.recycling_explanation,
        }
    )
    return contribution


def _row_to_contribution(row: dict) -> Contribution:
    return Contribution(
        schema_version=row["schema_version"],
        contribution_id=row["contribution_id"],
        kind=ContributionKind(row["kind"]),
        visibility=VisibilityLevel(row["visibility"]),
        contributor=ContributorRef(
            contributor_id=row["contributor_id"], display_name=row["contributor_display_name"],
        ),
        mechanism=EconomicMechanism(row["mechanism"]) if row["mechanism"] else None,
        root_symbol=row["root_symbol"],
        title=row["title"],
        notes=row["notes"],
        novelty_notes=row["novelty_notes"],
        evidence=EvidenceReference.model_validate(row["evidence"]),
        created_at=row["created_at"],
        recycling_decision=row["recycling_decision"],
        recycling_explanation=row["recycling_explanation"],
    )


def get_contribution(store: CommunityStore, contribution_id: str) -> Contribution | None:
    row = store.get_contribution(contribution_id)
    return _row_to_contribution(row) if row is not None else None


def list_contributions(store: CommunityStore) -> list[Contribution]:
    """Every contribution, every visibility level, unfiltered -- the raw
    typed store contents. Privacy-aware views (what a given viewer is
    actually allowed to see) live in :mod:`alpha_agent.community.visibility`;
    this function is its one real data source, never duplicated."""
    return [_row_to_contribution(row) for row in store.list_contributions()]
