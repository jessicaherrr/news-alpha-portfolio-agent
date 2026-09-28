"""Phase 8 acceptance (prompt section 9): one private contribution, one
shared/public contribution, one independent replication, conflicting/
rejected evidence handling, privacy boundaries, and Agent retrieval -- all in
one real, end-to-end scenario against a throwaway registry/store (never the
real production registry -- this repository's real registry has exactly one
CANONICAL trial per (root, strategy family), so no genuine second-researcher
execution of the identical hypothesis exists yet to replicate against; the
fixture's second "researcher" is disclosed, not claimed as real).

Also covers Community Memory's research-prioritization note: a repo-wide-
style banned-trade-vocabulary sweep, mirroring Phase 7's own
`test_phase7_alpha_graph_retrieval.py` precedent for `alpha_graph`'s
cross-asset synthesis text.
"""
from __future__ import annotations

import re

import pytest
from _community_fixtures import insert
from alpha_agent.community import contributions, memory, replication, reputation, visibility
from alpha_agent.community.schemas import (
    ContributionKind,
    EvidenceMaturity,
    ReplicationOutcome,
    VisibilityLevel,
)
from alpha_agent.community.store import CommunityStore
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.enums import RegistryVerdict
from alpha_agent.registry.sqlite_registry import ExperimentRegistry

_BANNED_TRADE_WORDS = re.compile(
    r"\b(buy|sell|go long|go short|should trade|guaranteed|recommend(ed|s)?)\b", re.IGNORECASE
)


def _sweep(note: str) -> re.Match | None:
    """Scan `note` for banned trade-instruction vocabulary, excluding the
    one fixed, reviewed, ALLOW-LISTED disclaimer sentence
    (`memory.DISCLAIMER`) that legitimately contains "guaranteed"/
    "recommendation" in a negating context -- mirrors Phase 7's own
    `alpha_graph` banned-vocabulary sweep convention."""
    return _BANNED_TRADE_WORDS.search(note.replace(memory.DISCLAIMER, ""))


@pytest.fixture
def registry(tmp_path):
    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reg:
        yield reg


@pytest.fixture
def store(tmp_path):
    with CommunityStore(tmp_path / "community.sqlite") as s:
        yield s


def test_phase8_full_acceptance_scenario(registry, store):
    # -- real registry evidence, three distinct hypotheses -----------------
    cl_original = insert(
        registry, root="CL", family="tsmom", params={"fast_horizon": 21, "slow_horizon": 120, "size": 1},
        verdict=RegistryVerdict.REJECT, vintage="researcher-a",
    )
    cl_second_run = insert(
        registry, root="CL", family="tsmom", params={"fast_horizon": 21, "slow_horizon": 120, "size": 1},
        verdict=RegistryVerdict.PASS, vintage="researcher-b",
    )
    nq_private_evidence = insert(
        registry, root="NQ", family="ma_trend", params={"fast_horizon": 10, "slow_horizon": 50, "size": 1},
        verdict=RegistryVerdict.REJECT, vintage="researcher-a",
    )

    # -- 1. one PRIVATE contribution -----------------------------------------
    private = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PRIVATE,
        contributor_display_name="Alice", experiment_id=nq_private_evidence.experiment_id,
        title="NQ moving-average trend (personal, not ready to share)",
        mechanism=EconomicMechanism.TREND,
    )
    assert private.visibility is VisibilityLevel.PRIVATE

    # -- 2. one SHARED/PUBLIC contribution ------------------------------------
    public = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=cl_original.experiment_id,
        title="CL time-series momentum(21,120)", mechanism=EconomicMechanism.TREND,
        notes="Classic TSMOM on crude -- sharing the real, rejected registry evidence as-is.",
    )
    assert public.visibility is VisibilityLevel.PUBLIC
    assert public.evidence.headline_verdict is RegistryVerdict.REJECT

    # -- 3. one independent replication --------------------------------------
    rep = replication.create_replication(
        store, registry, contribution=public, replicator_display_name="Bob",
        experiment_id=cl_second_run.experiment_id,
        notes="Re-ran the same hypothesis on my own dataset vintage.",
    )
    assert rep.replicator.contributor_id == "bob"
    assert rep.evidence.experiment_identity != public.evidence.experiment_identity

    # -- 4. conflicting/rejected evidence handling ---------------------------
    # the original REJECTed; the replication PASSed -- a real disagreement,
    # preserved distinctly, never merged into one verdict.
    assert rep.comparison.outcome is ReplicationOutcome.CONFLICTS
    assert rep.comparison.original_verdict is RegistryVerdict.REJECT
    assert rep.comparison.replication_verdict is RegistryVerdict.PASS
    maturity, rationale = replication.evidence_maturity(public, [rep])
    assert maturity is EvidenceMaturity.INCONCLUSIVE
    assert "conflict" in rationale.lower()
    # the REJECTed private contribution's own evidence is preserved verbatim too
    assert private.evidence.headline_verdict is RegistryVerdict.REJECT

    # -- 5. privacy boundaries ------------------------------------------------
    feed = visibility.community_feed(store, viewer_contributor_id="a-random-visitor")
    feed_ids = {c.contribution_id for c in feed}
    assert public.contribution_id in feed_ids
    assert private.contribution_id not in feed_ids  # never leaks, any viewer
    mine = visibility.my_contributions(store, "alice")
    assert {private.contribution_id, public.contribution_id} <= {c.contribution_id for c in mine}

    # -- 6. Agent retrieval (Community Memory) --------------------------------
    lookup = memory.community_memory_lookup(store, registry, mechanism=EconomicMechanism.TREND, root_symbol="CL")
    assert lookup.community is not None
    assert lookup.community.n_public_contributions == 1
    assert lookup.community.n_replications == 1
    note = lookup.research_prioritization_note
    assert "never a guaranteed trade recommendation" in note
    assert not _sweep(note)

    # -- reputation, both sides ------------------------------------------------
    alice = reputation.reputation_profile(store, "alice")
    bob = reputation.reputation_profile(store, "bob")
    assert alice.n_contributions == 2  # private + public, counted equally
    assert alice.n_conflicting_replications_received == 1
    assert bob.n_replications_performed == 1


def test_community_memory_note_never_uses_trade_instruction_vocabulary_across_mechanisms(registry, store):
    """Sweep every mechanism this fixture can produce a note for -- mirrors
    Phase 7's own repo-wide banned-vocabulary sweep, scoped to what this
    test can construct rather than the live production registry."""
    exp = insert(registry, root="GC", family="breakout", params={"lookback": 20, "size": 1}, verdict=RegistryVerdict.INCONCLUSIVE)
    contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.SHARED,
        contributor_display_name="Dana", experiment_id=exp.experiment_id, title="GC breakout",
        mechanism=EconomicMechanism.BREAKOUT,
    )
    for mechanism in EconomicMechanism:
        lookup = memory.community_memory_lookup(store, registry, mechanism=mechanism, root_symbol="GC")
        assert not _sweep(lookup.research_prioritization_note), mechanism


def test_community_memory_reports_no_evidence_honestly_when_nothing_shared(registry, store):
    lookup = memory.community_memory_lookup(store, registry, mechanism=EconomicMechanism.CARRY, root_symbol="ZN")
    assert lookup.community is None
    assert "No shared or public community evidence exists yet" in lookup.research_prioritization_note
