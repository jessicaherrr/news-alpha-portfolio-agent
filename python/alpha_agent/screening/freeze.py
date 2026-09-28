"""Alpha Discovery campaign, Part G -- Freeze before strict validation
(task spec sections 42-44).

    candidate generation (Part D)
        -> Fast Screen (Part F)
        -> screen ranking
        -> Top K selection
        -> FREEZE  <-- this module
        -> strict validation (unchanged Phase 18
           ResearchOrchestrator.execute_family / finalize_family)

`freeze_top_k` builds a brand-new, from-scratch `FamilyManifest` (the SAME
frozen, content-hashed, immutable type Phase 18 already uses -- imported, not
re-implemented) containing ONLY the Top-K fast-screened survivors, with
`declared_target_size == K` so `FamilyManifest.planning_complete` is true and
the family is a legitimate, fully-predeclared multiple-testing family of size
K -- never a "family of one" evading FDR (task spec section 43), and never
missing members silently downgraded to a smaller family after the fact
(`FamilyManifest.assert_consistent` / its content-hash `family_id` already
make a post-hoc member change fail loudly; this module changes nothing about
that machinery).

`write_frozen_manifest` persists the immutable choice to disk BEFORE any
strict-validation call is made, so "what was frozen" is independently
auditable even if the process handling execution restarts. Frozen manifests
are operational discovery artifacts (like the paper-trading ledger), not
source or registry truth -- `data/discovery/freezes/` is gitignored.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from alpha_agent.agents.orchestrator import FamilyManifest, IdentityPlanes, _compute_family_id
from alpha_agent.discovery.candidate_pool import CandidatePoolResult
from alpha_agent.registry.models import MarketWindow
from alpha_agent.screening.fast_screen import FastScreenTrial, select_top_k

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FREEZE_DIR = REPO_ROOT / "data" / "discovery" / "freezes"


class FreezeError(RuntimeError):
    """A frozen candidate set could not be built -- e.g. the fast-screen
    trials and the candidate pool disagree about membership. Never silently
    proceeds with a mismatched set."""


class FrozenCandidateSet(BaseModel):
    """The auditable record of one freeze decision: which K members were
    selected, from which pool, with which screen scores, at what generation.
    `manifest` is the actual `FamilyManifest` strict validation will execute;
    everything else here is provenance for the freeze decision itself."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "frozen-candidate-set/1"
    market: str
    ideas_considered: int
    mechanisms_supported: int
    fast_screened: int
    frozen_count: int
    screen_scores: dict[str, float]  # experiment_identity -> ResearchScreenScore.total
    manifest: FamilyManifest
    frozen_at: str = ""


def freeze_top_k(
    *,
    pool: CandidatePoolResult,
    trials: list[FastScreenTrial],
    k: int,
    family_stem: str,
    planes: IdentityPlanes,
    market_window: MarketWindow,
    fdr_q_threshold: float = 0.10,
    generation: int = 0,
    parent_family_id: str | None = None,
) -> FrozenCandidateSet:
    """Select the Top `k` fast-screened survivors from `pool` and freeze them
    into a brand-new, from-scratch `FamilyManifest` of declared size `k` (or
    fewer, if fewer than `k` members were actually SCREENED -- the family is
    then declared at that smaller REAL size, never padded).

    `trials` must be exactly the Fast Screen outcomes for `pool.members`
    (task spec section 42: freezing must operate on a real 1:1 mapping, never
    a mismatched or partial one) -- checked explicitly.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    pool_ids = {m.experiment_identity for m in pool.members}
    trial_ids = {t.experiment_identity for t in trials}
    if trial_ids != pool_ids:
        raise FreezeError(
            "fast-screen trials do not exactly match the candidate pool's members "
            f"(pool has {len(pool_ids)}, trials have {len(trial_ids)}, "
            f"symmetric difference {pool_ids ^ trial_ids}) -- refusing to freeze a "
            "mismatched candidate set"
        )

    survivors = select_top_k(trials, k)
    if not survivors:
        raise FreezeError("no candidate survived the fast screen -- nothing to freeze")

    by_id = {m.experiment_identity: m for m in pool.members}
    frozen_members = tuple(by_id[t.experiment_identity] for t in survivors)
    reordinaled = tuple(m.model_copy(update={"ordinal": i}) for i, m in enumerate(frozen_members))

    declared_size = len(reordinaled)
    policy_identity = planes.reliability_policy_fingerprint
    family_id = _compute_family_id(
        family_stem=family_stem, generation=generation, parent_family_id=parent_family_id,
        planes=planes, market_window=market_window, fdr_q_threshold=fdr_q_threshold,
        policy_identity=policy_identity, declared_target_size=declared_size, members=reordinaled,
    )
    from datetime import UTC, datetime

    manifest = FamilyManifest(
        family_id=family_id, family_stem=family_stem, generation=generation,
        parent_family_id=parent_family_id, planes=planes, market_window=market_window,
        fdr_q_threshold=fdr_q_threshold, policy_identity=policy_identity,
        declared_target_size=declared_size, members=reordinaled,
        created_at=datetime.now(UTC).isoformat(),
    )
    manifest.assert_consistent()
    if not manifest.planning_complete:
        raise FreezeError("frozen manifest failed its own planning_complete invariant")  # pragma: no cover

    scores = {t.experiment_identity: t.score.total for t in survivors if t.score is not None}
    return FrozenCandidateSet(
        market=pool.market, ideas_considered=pool.ideas_considered,
        mechanisms_supported=pool.mechanisms_supported, fast_screened=len(trials),
        frozen_count=declared_size, screen_scores=scores, manifest=manifest,
        frozen_at=manifest.created_at,
    )


def write_frozen_manifest(frozen: FrozenCandidateSet, *, out_dir: Path | None = None) -> Path:
    """Persist the freeze decision to an immutable, hash-named JSON file
    BEFORE strict validation runs -- audit evidence independent of the
    calling process. Never overwrites an existing freeze for the same
    family_id (a `FamilyManifest`'s content hash IS its identity; two freezes
    of the same content are the same file, and a DIFFERENT content always
    gets a different file name)."""
    out_dir = out_dir or DEFAULT_FREEZE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{frozen.manifest.family_id.replace(':', '_')}.json"
    if not path.exists():
        path.write_text(json.dumps(frozen.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_frozen_manifest(path: Path) -> FrozenCandidateSet:
    return FrozenCandidateSet.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def adopt_frozen_candidate_set(orchestrator, frozen: FrozenCandidateSet) -> FamilyManifest:
    """Safely hand a `FrozenCandidateSet` (this module's own freeze-decision
    record -- e.g. round-tripped through `write_frozen_manifest` /
    `read_frozen_manifest`, so it can come from an entirely separate process)
    to a fresh `ResearchOrchestrator` for strict validation.

    Thin, deliberate convenience wrapper over
    `ResearchOrchestrator.adopt_frozen_manifest` -- it lives here (not on
    `ResearchOrchestrator` itself) because `FrozenCandidateSet` is a
    Fast-Screen/Freeze-layer type and `alpha_agent.agents.orchestrator` must
    not depend on `alpha_agent.screening` (this module already depends on
    `orchestrator`, never the reverse). `orchestrator` is typed loosely
    (avoids importing `ResearchOrchestrator` here purely for an annotation and
    creating that reverse dependency); it must be a
    `alpha_agent.agents.orchestrator.ResearchOrchestrator` in practice, and
    `adopt_frozen_manifest` type-checks it at the `FamilyManifest` level
    regardless of what calls it.
    """
    return orchestrator.adopt_frozen_manifest(
        frozen.manifest, fast_screen_provenance=frozen.screen_scores
    )
