"""Phase 8 -- Community Alpha Network core logic (prompt 8).

Covers: contribution creation + threat-model defenses (cherry-picking,
leakage, copied-strategy replication, community-level duplicate detection),
replication comparison + evidence-maturity ladder, reputation (never a
blended score, rejected work not penalized), and privacy boundaries
(PRIVATE never leaks, SHARED redacts strategy params, aggregates never
counted from PRIVATE). All against throwaway `tmp_path` registries/stores --
never the real production registry.
"""
from __future__ import annotations

import pytest
from _community_fixtures import insert
from alpha_agent.community import contributions, replication, reputation, visibility
from alpha_agent.community.schemas import (
    ContributionKind,
    EvidenceMaturity,
    ReplicationOutcome,
    VisibilityLevel,
)
from alpha_agent.community.store import CommunityStore
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.enums import RegistryVerdict, TrialRole
from alpha_agent.registry.sqlite_registry import ExperimentRegistry


@pytest.fixture
def registry(tmp_path):
    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reg:
        yield reg


@pytest.fixture
def store(tmp_path):
    with CommunityStore(tmp_path / "community.sqlite") as s:
        yield s


TSMOM_PARAMS = {"fast_horizon": 21, "slow_horizon": 120, "size": 1}


# ==========================================================================
# 1. Contribution kinds never include an unsupported trade call
# ==========================================================================


def test_contribution_kind_has_no_buy_sell_call_representation():
    values = {k.value for k in ContributionKind}
    assert values == {
        "HYPOTHESIS", "MECHANISM", "FACTOR_DEFINITION", "STRATEGY_SPEC",
        "EVIDENCE_BUNDLE", "RESEARCH_NOTES",
    }
    assert not any("BUY" in v or "SELL" in v or "CALL" in v for v in values)


# ==========================================================================
# 2. Evidence reference is built ONLY from real registry data
# ==========================================================================


def test_evidence_reference_is_read_verbatim_from_the_registry(registry):
    exp = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)
    ref = contributions.build_evidence_reference(registry, exp.experiment_id)
    assert ref.experiment_id == exp.experiment_id
    assert ref.experiment_identity == exp.experiment_identity
    assert ref.root_symbol == "CL"
    assert ref.strategy_family == "tsmom"
    assert ref.headline_verdict == RegistryVerdict.REJECT
    assert ref.strategy_params == TSMOM_PARAMS


def test_evidence_reference_refuses_an_unknown_experiment_id(registry):
    with pytest.raises(Exception):  # noqa: B017 -- registry's own UnknownExperiment
        contributions.build_evidence_reference(registry, "does-not-exist")


# ==========================================================================
# 3. Threat model: cherry-picking / hidden parameter search
# ==========================================================================


def test_a_non_canonical_trial_is_refused_as_contribution_evidence(registry):
    neighbour = insert(
        registry, root="CL", family="tsmom", params={"fast_horizon": 30, "slow_horizon": 90, "size": 1},
        verdict=RegistryVerdict.REJECT, trial_role=TrialRole.NEIGHBOUR, label="neighbour",
    )
    with pytest.raises(contributions.CherryPickedTrialError):
        contributions.build_evidence_reference(registry, neighbour.experiment_id)


# ==========================================================================
# 4. Threat model: leakage / locked holdout
# ==========================================================================


def test_a_market_window_touching_the_2025_holdout_is_refused(registry, monkeypatch):
    """`build_evidence_reference`'s own explicit `market_window_end` check
    fires first (a clear, community-specific message); the generic
    recursive `assert_no_holdout_market_data` guard is a second, independent
    backstop over the whole payload (defense in depth)."""
    exp = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)

    from alpha_agent.registry.models import MarketWindow

    class _BadView:
        def __init__(self, view):
            self._view = view
            self.experiment = view.experiment.model_copy(
                update={"market_window": MarketWindow(label="VALIDATION", start_date="2024-01-01", end_date="2025-06-30")}
            )
            self.result = view.result
            self.experiment_id = view.experiment_id
            self.experiment_identity = view.experiment_identity
            self.verdict = view.verdict

    real_get = registry.get

    def _patched_get(key):
        return _BadView(real_get(key))

    monkeypatch.setattr(registry, "get", _patched_get)
    with pytest.raises(contributions.CommunityHoldoutError):
        contributions.build_evidence_reference(registry, exp.experiment_id)


def test_a_holdout_date_leaking_only_through_strategy_params_is_still_caught(registry, monkeypatch):
    """A value the explicit `market_window_end` check never inspects
    (`strategy_params`) is still caught by the generic recursive guard --
    proving the second defense is not dead code."""
    exp = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)

    from alpha_agent.registry.holdout_guard import HoldoutAccessError

    class _BadView:
        def __init__(self, view):
            self.experiment = view.experiment.model_copy(
                update={"strategy_spec_json": {"params": {**TSMOM_PARAMS, "note": "as of 2025-03-01"}}}
            )
            self.result = view.result
            self.experiment_id = view.experiment_id
            self.experiment_identity = view.experiment_identity
            self.verdict = view.verdict

    real_get = registry.get
    monkeypatch.setattr(registry, "get", lambda key: _BadView(real_get(key)))
    with pytest.raises(HoldoutAccessError):
        contributions.build_evidence_reference(registry, exp.experiment_id)


# ==========================================================================
# 5. Community-level duplicate-contribution detection (never a hard block)
# ==========================================================================


def test_first_contribution_for_a_hypothesis_is_never_flagged_as_duplicate(registry, store):
    exp = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)
    c = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=exp.experiment_id, title="CL TSMOM",
        mechanism=EconomicMechanism.TREND,
    )
    assert c.recycling_decision == "RECONSIDER"


def test_re_contributing_the_exact_same_evidence_is_flagged_but_never_blocked(registry, store):
    exp = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)
    contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=exp.experiment_id, title="CL TSMOM",
        mechanism=EconomicMechanism.TREND,
    )
    dupe = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Carol", experiment_id=exp.experiment_id, title="CL TSMOM (again)",
        mechanism=EconomicMechanism.TREND,
    )
    assert dupe.recycling_decision == "DEPRIORITIZE_NO_NOVELTY"
    # never a hard block -- the contribution is still created and persisted
    assert contributions.get_contribution(store, dupe.contribution_id) is not None


def test_stated_novelty_reconsiders_even_an_exact_evidence_repeat(registry, store):
    exp = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)
    contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=exp.experiment_id, title="CL TSMOM",
        mechanism=EconomicMechanism.TREND,
    )
    novel = contributions.create_contribution(
        store, registry, kind=ContributionKind.RESEARCH_NOTES, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Carol", experiment_id=exp.experiment_id, title="CL TSMOM -- new regime read",
        mechanism=EconomicMechanism.TREND, novelty_notes="Re-reading this under the 2024 rate-cutting cycle.",
    )
    assert novel.recycling_decision == "RECONSIDER"


# ==========================================================================
# 6. Threat model: copied strategies (replication independence)
# ==========================================================================


def test_replication_refuses_citing_the_original_experiment_identity(registry, store):
    exp = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)
    c = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=exp.experiment_id, title="CL TSMOM",
    )
    with pytest.raises(replication.NotIndependentReplicationError):
        replication.create_replication(
            store, registry, contribution=c, replicator_display_name="Bob", experiment_id=exp.experiment_id,
        )


def test_replication_refuses_the_same_contributor_even_with_different_evidence(registry, store):
    """Independence has TWO conditions, either one sufficient to refuse:
    same experiment_identity (proven above), or same (self-attested)
    contributor_id -- Alice cannot "independently replicate" her own
    contribution merely by pointing at a second, genuinely different
    experiment. Bob, a different contributor, citing that same second
    experiment remains allowed."""
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT, vintage="a")
    b = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT, vintage="b")
    c = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM",
    )
    with pytest.raises(replication.NotIndependentReplicationError):
        replication.create_replication(
            store, registry, contribution=c, replicator_display_name="Alice", experiment_id=b.experiment_id,
        )
    # a different contributor citing the SAME (different-from-original) evidence remains allowed
    rep = replication.create_replication(
        store, registry, contribution=c, replicator_display_name="Bob", experiment_id=b.experiment_id,
    )
    assert rep.replicator.contributor_id == "bob"


# ==========================================================================
# 7. Replication comparison preserves every dimension, never collapses
# ==========================================================================


def test_replication_confirms_when_verdicts_match_on_the_same_claim(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT, vintage="a")
    b = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT, vintage="b")
    c = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM",
        mechanism=EconomicMechanism.TREND,
    )
    rep = replication.create_replication(
        store, registry, contribution=c, replicator_display_name="Bob", experiment_id=b.experiment_id,
    )
    cmp = rep.comparison
    assert cmp.outcome == ReplicationOutcome.CONFIRMS
    assert cmp.same_root_symbol and cmp.same_strategy_family and cmp.same_asset_domain
    assert cmp.same_factor_identity is True
    # different vintages -> different dataset fingerprints, correctly NOT collapsed
    assert cmp.same_dataset_fingerprint is False
    assert cmp.original_verdict == RegistryVerdict.REJECT
    assert cmp.replication_verdict == RegistryVerdict.REJECT


def test_replication_conflicts_when_verdicts_disagree_and_is_preserved(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT, vintage="a")
    b = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.PASS, vintage="b")
    c = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM",
        mechanism=EconomicMechanism.TREND,
    )
    rep = replication.create_replication(
        store, registry, contribution=c, replicator_display_name="Bob", experiment_id=b.experiment_id,
    )
    assert rep.comparison.outcome == ReplicationOutcome.CONFLICTS
    # both verdicts stay individually visible -- never merged into one field
    assert rep.comparison.original_verdict != rep.comparison.replication_verdict
    stored = replication.list_replications(store, c.contribution_id)
    assert len(stored) == 1
    assert stored[0].comparison.outcome == ReplicationOutcome.CONFLICTS


def test_replication_on_a_different_root_is_not_comparable_not_merged(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)
    b = insert(registry, root="ES", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)
    c = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM",
    )
    rep = replication.create_replication(
        store, registry, contribution=c, replicator_display_name="Bob", experiment_id=b.experiment_id,
    )
    assert rep.comparison.outcome == ReplicationOutcome.NOT_COMPARABLE
    assert rep.comparison.same_root_symbol is False


# ==========================================================================
# 8. Evidence maturity: conservative, documented, never inflated
# ==========================================================================


def test_maturity_is_proposed_with_no_replication(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.PASS)
    c = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM",
    )
    maturity, _ = replication.evidence_maturity(c, [])
    assert maturity is EvidenceMaturity.PROPOSED


def test_maturity_is_rejected_when_original_verdict_is_reject_regardless_of_replication_count(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT, vintage="a")
    b = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT, vintage="b")
    c = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM",
        mechanism=EconomicMechanism.TREND,
    )
    replication.create_replication(store, registry, contribution=c, replicator_display_name="Bob", experiment_id=b.experiment_id)
    maturity, _ = replication.evidence_maturity(c, replication.list_replications(store, c.contribution_id))
    assert maturity is EvidenceMaturity.REJECTED


def test_maturity_is_inconclusive_when_replications_conflict(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.PASS, vintage="a")
    b = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT, vintage="b")
    c = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM",
        mechanism=EconomicMechanism.TREND,
    )
    replication.create_replication(store, registry, contribution=c, replicator_display_name="Bob", experiment_id=b.experiment_id)
    maturity, _ = replication.evidence_maturity(c, replication.list_replications(store, c.contribution_id))
    assert maturity is EvidenceMaturity.INCONCLUSIVE


def test_maturity_needs_two_confirming_replications_for_emerging_evidence(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.PASS, vintage="a")
    b = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.PASS, vintage="b")
    d = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.PASS, vintage="c")
    c = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM",
        mechanism=EconomicMechanism.TREND,
    )
    replication.create_replication(store, registry, contribution=c, replicator_display_name="Bob", experiment_id=b.experiment_id)
    reps = replication.list_replications(store, c.contribution_id)
    maturity, _ = replication.evidence_maturity(c, reps)
    assert maturity is EvidenceMaturity.REPLICATING

    replication.create_replication(store, registry, contribution=c, replicator_display_name="Carol", experiment_id=d.experiment_id)
    reps = replication.list_replications(store, c.contribution_id)
    maturity, _ = replication.evidence_maturity(c, reps)
    assert maturity is EvidenceMaturity.EMERGING_EVIDENCE


# ==========================================================================
# 9. Privacy boundaries (prompt section 2)
# ==========================================================================


def test_private_contribution_never_appears_in_the_community_feed(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)
    priv = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PRIVATE,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM (private)",
    )
    feed = visibility.community_feed(store)
    assert priv.contribution_id not in {c.contribution_id for c in feed}
    # not even visible to its own contributor through the feed path
    feed_as_owner = visibility.community_feed(store, viewer_contributor_id="alice")
    assert priv.contribution_id not in {c.contribution_id for c in feed_as_owner}
    # but IS visible through the owner's own "my contributions" view
    mine = visibility.my_contributions(store, "alice")
    assert priv.contribution_id in {c.contribution_id for c in mine}


def test_shared_contribution_redacts_strategy_params_for_non_owners_only(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)
    shared = contributions.create_contribution(
        store, registry, kind=ContributionKind.STRATEGY_SPEC, visibility=VisibilityLevel.SHARED,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM (shared)",
    )
    feed_stranger = visibility.community_feed(store, viewer_contributor_id="mallory")
    seen = next(c for c in feed_stranger if c.contribution_id == shared.contribution_id)
    assert seen.evidence.strategy_params is None
    # the headline verdict and mechanism-level facts still benefit the aggregate
    assert seen.evidence.headline_verdict == RegistryVerdict.REJECT

    feed_owner = visibility.community_feed(store, viewer_contributor_id="alice")
    seen_owner = next(c for c in feed_owner if c.contribution_id == shared.contribution_id)
    assert seen_owner.evidence.strategy_params == TSMOM_PARAMS


def test_public_contribution_is_never_redacted(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)
    pub = contributions.create_contribution(
        store, registry, kind=ContributionKind.STRATEGY_SPEC, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM (public)",
    )
    feed_stranger = visibility.community_feed(store, viewer_contributor_id="mallory")
    seen = next(c for c in feed_stranger if c.contribution_id == pub.contribution_id)
    assert seen.evidence.strategy_params == TSMOM_PARAMS


def test_private_contribution_never_enters_any_aggregate_not_even_as_a_count(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)
    contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PRIVATE,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM (private)",
        mechanism=EconomicMechanism.TREND,
    )
    agg = visibility.aggregate_for_mechanism_root(store, mechanism=EconomicMechanism.TREND, root_symbol="CL")
    assert agg is None  # absence, never a negative or zero-with-detail signal


def test_aggregate_counts_shared_and_public_contributions_separately(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT, vintage="a")
    b = insert(registry, root="CL", family="ma_trend", params={"fast_horizon": 10, "slow_horizon": 50, "size": 1}, verdict=RegistryVerdict.PASS, vintage="b")
    contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM",
        mechanism=EconomicMechanism.TREND,
    )
    contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.SHARED,
        contributor_display_name="Bob", experiment_id=b.experiment_id, title="CL MA Trend",
        mechanism=EconomicMechanism.TREND,
    )
    agg = visibility.aggregate_for_mechanism_root(store, mechanism=EconomicMechanism.TREND, root_symbol="CL")
    assert agg.n_public_contributions == 1
    assert agg.n_shared_contributions == 1
    assert agg.n_contributors == 2
    assert agg.verdict_distribution == {"REJECT": 1, "PASS": 1}  # never collapsed into one verdict


# ==========================================================================
# 10. Reputation: never a blended score, rejected work not penalized
# ==========================================================================


def test_reputation_counts_reject_contributions_equally(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT)
    b = insert(registry, root="ES", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.PASS)
    contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM (rejected)",
    )
    contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=b.experiment_id, title="ES TSMOM (passed)",
    )
    profile = reputation.reputation_profile(store, "alice")
    assert profile.n_contributions == 2
    assert profile.n_public_contributions == 2


def test_reputation_profile_has_no_single_blended_score_field():
    from alpha_agent.community.schemas import ReputationProfile

    model_fields = set(ReputationProfile.model_fields)
    forbidden = {"score", "reputation_score", "rank", "alpha_score", "confidence"}
    assert not (model_fields & forbidden)


def test_reputation_tracks_replications_performed_and_received_separately(registry, store):
    a = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT, vintage="a")
    b = insert(registry, root="CL", family="tsmom", params=TSMOM_PARAMS, verdict=RegistryVerdict.REJECT, vintage="b")
    c = contributions.create_contribution(
        store, registry, kind=ContributionKind.HYPOTHESIS, visibility=VisibilityLevel.PUBLIC,
        contributor_display_name="Alice", experiment_id=a.experiment_id, title="CL TSMOM",
        mechanism=EconomicMechanism.TREND,
    )
    replication.create_replication(store, registry, contribution=c, replicator_display_name="Bob", experiment_id=b.experiment_id)

    alice = reputation.reputation_profile(store, "alice")
    bob = reputation.reputation_profile(store, "bob")
    assert alice.n_times_own_work_replicated == 1
    assert alice.n_replications_performed == 0
    assert bob.n_replications_performed == 1
    assert bob.n_times_own_work_replicated == 0


# ==========================================================================
# 11. No "Like" interaction exists anywhere in this package (popularity bias)
# ==========================================================================


def test_no_like_or_upvote_interaction_exists_in_the_community_package():
    import alpha_agent.community as pkg

    banned = ("like", "upvote", "star_count", "favorite")
    names = " ".join(pkg.__all__).lower()
    assert not any(word in names for word in banned)
