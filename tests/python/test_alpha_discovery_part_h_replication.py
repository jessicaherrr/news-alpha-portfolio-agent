"""Alpha Discovery campaign, Part H -- multi-period/multi-market robustness
tests (task spec section 45-47; test coverage implied by the general
"honest capability, no brute-force mining" requirements)."""
from __future__ import annotations

from alpha_agent.agents.orchestrator import (
    FamilyManifest,
    FamilyMember,
    IdentityPlanes,
    _compute_family_id,
)
from alpha_agent.discovery.replication import (
    RELATED_MARKET_GROUPS,
    build_replication_suggestions,
    propose_replication_targets,
)
from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
from alpha_agent.registry.models import MarketWindow
from alpha_agent.strategy import strategy_fingerprint
from alpha_agent.strategy.baselines.factories import make_tsmom_spec
from alpha_agent.strategy.baselines.params import TsmomParams
from alpha_agent.validation.policy import ReliabilityPolicy

POLICY = ReliabilityPolicy()
PLANES = IdentityPlanes(
    dataset_fingerprint="valdataset2:ds", split_identity="split1:sp",
    validation_spec_fingerprint="validationspec1:vs",
    reliability_policy_fingerprint=POLICY.identity(),
    execution_config_identity="execconfig1:ex", cost_config_identity="costconfig1:co",
    risk_identity="riskconfig1:ri",
)
MW = MarketWindow(label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31")


def test_index_futures_are_a_related_group():
    assert ("ES", "NQ") in RELATED_MARKET_GROUPS


def test_propose_replication_targets_stays_within_the_same_economic_group():
    assert propose_replication_targets("NQ") == ("ES",)
    assert propose_replication_targets("ES") == ("NQ",)


def test_propose_replication_targets_never_crosses_unrelated_groups():
    """The exact CLAUDE.md principle: never force a crude-specific mechanism
    onto NQ (or vice versa) just to increase sample count."""
    assert propose_replication_targets("CL") == ()
    assert propose_replication_targets("GC") == ()
    assert propose_replication_targets("ZN") == ()
    assert "CL" not in propose_replication_targets("NQ")
    assert "NQ" not in propose_replication_targets("CL")


def test_propose_replication_targets_filtered_to_approved_universe():
    assert propose_replication_targets("NQ", universe=("NQ", "CL")) == ()  # ES not approved
    assert propose_replication_targets("NQ", universe=("NQ", "ES", "CL")) == ("ES",)


def _member(root: str) -> FamilyMember:
    params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1, "root_symbol": root}
    spec = make_tsmom_spec(TsmomParams(**params))
    fp = strategy_fingerprint(spec)
    pvi = parameter_variant_identity(params)
    identity = experiment_identity(
        strategy_fingerprint=fp, strategy_family="tsmom", root_symbol=root,
        parameter_variant_identity=pvi, dataset_fingerprint=PLANES.dataset_fingerprint,
        split_identity=PLANES.split_identity, validation_spec_fingerprint=PLANES.validation_spec_fingerprint,
        reliability_policy_fingerprint=PLANES.reliability_policy_fingerprint,
        execution_config_identity=PLANES.execution_config_identity,
        cost_config_identity=PLANES.cost_config_identity, risk_identity=PLANES.risk_identity,
        feature_spec_fingerprint="featset1:test",
    )
    return FamilyMember(
        ordinal=0, experiment_identity=identity, strategy_fingerprint=fp, strategy_id=spec.strategy_id,
        strategy_family="tsmom", root_symbol=root, params=params, parameter_variant_identity=pvi,
        feature_spec_fingerprint="featset1:test", hypothesis_id="H-0", hypothesis_title="t",
        strategy_spec=spec, strategy_spec_json={"schema": "registry-strategy-spec/1"},
    )


def _manifest(members: tuple[FamilyMember, ...]) -> FamilyManifest:
    members = tuple(m.model_copy(update={"ordinal": i}) for i, m in enumerate(members))
    fam_id = _compute_family_id(
        family_stem="test_replication", generation=0, parent_family_id=None, planes=PLANES,
        market_window=MW, fdr_q_threshold=0.10, policy_identity=PLANES.reliability_policy_fingerprint,
        declared_target_size=len(members), members=members,
    )
    return FamilyManifest(
        family_id=fam_id, family_stem="test_replication", generation=0, planes=PLANES, market_window=MW,
        fdr_q_threshold=0.10, policy_identity=PLANES.reliability_policy_fingerprint,
        declared_target_size=len(members), members=members,
    )


def test_build_replication_suggestions_for_a_frozen_nq_member():
    manifest = _manifest((_member("NQ"),))
    suggestions = build_replication_suggestions(manifest, universe=("ES", "NQ", "CL", "GC", "ZN"))
    assert len(suggestions) == 1
    assert suggestions[0].suggested_roots == ("ES",)
    assert suggestions[0].source_root == "NQ"


def test_build_replication_suggestions_skips_already_covered_roots():
    manifest = _manifest((_member("NQ"), _member("ES")))
    suggestions = build_replication_suggestions(manifest, universe=("ES", "NQ"))
    assert suggestions == ()  # both index-futures roots already in the family


def test_build_replication_suggestions_empty_for_singleton_groups():
    manifest = _manifest((_member("CL"),))
    assert build_replication_suggestions(manifest, universe=("ES", "NQ", "CL", "GC", "ZN")) == ()


def test_replication_never_recomputes_or_writes_experiment_identity():
    """A suggestion carries the SOURCE identity for reference; it must never
    invent or claim a NEW experiment_identity of its own (that only exists
    once a real hypothesis is compiled through the normal Part D/E path)."""
    manifest = _manifest((_member("NQ"),))
    suggestions = build_replication_suggestions(manifest, universe=("ES", "NQ"))
    dumped = suggestions[0].model_dump()
    assert "experiment_identity" not in dumped or dumped.get("source_experiment_identity")
    assert not hasattr(suggestions[0], "new_experiment_identity")
