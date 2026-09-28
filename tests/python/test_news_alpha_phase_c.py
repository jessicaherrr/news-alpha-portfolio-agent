"""News Alpha Phase C -- Economic Mechanism Graph -> SIGNAL PATH DISCOVERY, and
the mechanism-adjusted impact assessment (second triage pass).

Pure, offline tests: fixed `DomainCapabilities`, fixed timestamps, no network,
no LLM. Covers DIRECT / SUPPLY_CHAIN / CROSS_SECTOR classification, ordered
paths, depth as a recorded (never ranked) variable, conflicting paths,
duplicate detection, model path proposals, mechanism-graph integration,
initial vs mechanism-adjusted impact, provenance, and the no-trade boundary.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alpha_agent.news_alpha import (
    DomainCapabilities,
    EconomicMechanismGraph,
    ImpactLevel,
    InitialImpactScan,
    LevelChange,
    MandateDomain,
    MechanismSupport,
    Movement,
    PathStatus,
    PathType,
    Polarity,
    ResearchMandate,
    SignalPathDiscovery,
    SignalPathProposal,
    StatusReason,
    TransmissionLibrary,
    TransmissionProposal,
    UserDescribedEvent,
    adjust_impact,
    build_economic_mechanism_graph,
    claims_from_proposal,
    discover_signal_paths,
    scan_initial_impact,
)
from alpha_agent.news_alpha.channels import EconomicChannel
from alpha_agent.news_alpha.impact_adjustment import MAGNITUDE_NOTE, scan_fingerprint
from alpha_agent.news_alpha.signal_path_library import (
    CHANNEL_ROLES,
    DEFAULT_PATH_LIBRARY,
    EXPOSURE_BASIS,
    STATE_PROFILES,
    EconomicSector,
    ExposureBasis,
    SignalPathLibrary,
    StateProfile,
    _validate,
)
from alpha_agent.news_alpha.signal_paths import (
    PathFindingKind,
    PathOrigin,
    ProposedSignalPath,
    RejectionReason,
    SignalPath,
)
from alpha_agent.news_alpha.transmission import (
    EdgeOrigin,
    LinkConfidence,
    ProvenanceKind,
    ProvenanceSource,
    StateKind,
    TransmissionChannel,
    TransmissionClaim,
    TransmissionLag,
)
from alpha_agent.news_alpha.transmission_library import AI_INFRASTRUCTURE_SEED, AnchorRule
from alpha_agent.ui import palette, signal_path_view
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "python" / "alpha_agent" / "news_alpha"
PHASE_C_SOURCES = [
    PACKAGE_DIR / "signal_paths.py", PACKAGE_DIR / "signal_path_library.py", PACKAGE_DIR / "impact_adjustment.py",
    REPO_ROOT / "python" / "alpha_agent" / "ui" / "signal_path_view.py",
]
REGISTRY_PATH = REPO_ROOT / "data" / "registry" / "experiments.sqlite"

CAPS = DomainCapabilities(etf_market_data_acquired=True, equity_market_data_acquired=False)
AT = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
ALL_ASSESSABLE = ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES, MandateDomain.ETF))

AI_CAPEX = "Hyperscalers announce a major AI infrastructure spending increase, lifting data center capex guidance"
AI_CAPEX_PARAPHRASE = "Cloud giants boost spending on AI infrastructure and data centers, raising capex guidance"
AI_NO_DIRECTION = "Hyperscalers discuss AI infrastructure and data center capex plans"
OPEC_CUT = "OPEC+ agrees a production cut of 1 million barrels per day"
RATE_SURPRISE = "Rate-policy surprise: the Fed unexpectedly raised rates by 50 basis points"
GEOPOLITICAL = "Missile strikes and a naval blockade raise geopolitical escalation fears"


def _scan(text: str, mandate: ResearchMandate = ALL_ASSESSABLE) -> InitialImpactScan:
    return scan_initial_impact(UserDescribedEvent.create(text, described_at=AT), mandate, capabilities=CAPS)


def _graph(text: str, mandate: ResearchMandate = ALL_ASSESSABLE, **kwargs) -> EconomicMechanismGraph:
    return build_economic_mechanism_graph(_scan(text, mandate), **kwargs)


def _paths(text: str, *, graph_kwargs: dict | None = None, **kwargs) -> SignalPathDiscovery:
    return discover_signal_paths(_graph(text, **(graph_kwargs or {})), **kwargs)


def _types(d: SignalPathDiscovery, state_id: str) -> set[PathType | None]:
    return {p.path_type for p in d.to(state_id)}


def _only(d: SignalPathDiscovery, *state_ids: str) -> SignalPath:
    (p,) = [p for p in d.paths if p.state_ids[1:] == state_ids]
    return p


def _model_claim(source: str, target: str, polarity: Polarity, **overrides) -> TransmissionClaim:
    proposal = TransmissionProposal(
        proposer="scripted-fixture", prompt_fingerprint="test",
        links=({"source": source, "target": target, "polarity": polarity.value,
                "rationale": "fixture model link", **overrides},),
    )
    return claims_from_proposal(proposal)[0]


def _path_proposal(*paths: tuple[tuple[str, ...], str]) -> SignalPathProposal:
    return SignalPathProposal(
        proposer="scripted-fixture", prompt_fingerprint="test",
        paths=tuple({"states": list(states), "rationale": why} for states, why in paths),
    )


# ---------------------------------------------------------------------------
# classification: DIRECT / SUPPLY_CHAIN / CROSS_SECTOR
# ---------------------------------------------------------------------------


def test_ai_event_yields_all_three_path_types():
    d = _paths(AI_CAPEX)
    assert d.count_by_type() == {PathType.DIRECT: 6, PathType.SUPPLY_CHAIN: 7, PathType.CROSS_SECTOR: 7}
    for state in ("compute_demand", "accelerator_demand", "data_center_construction", "ai_spender_free_cash_flow"):
        assert _types(d, state) == {PathType.DIRECT}, state
    for state in ("hbm_demand", "advanced_foundry_utilization", "foundry_capacity_expansion",
                  "semiconductor_equipment_demand"):
        assert _types(d, state) == {PathType.SUPPLY_CHAIN}, state
    for state in ("electricity_demand", "grid_investment", "power_equipment_demand", "gas_power_generation_demand",
                  "copper_demand"):
        assert _types(d, state) == {PathType.CROSS_SECTOR}, state


def test_direct_reaches_the_first_counterparty_and_supply_chain_goes_past_it():
    d = _paths(AI_CAPEX)
    accelerator = _only(d, "compute_demand", "accelerator_demand")
    assert (accelerator.path_type, accelerator.value_chain_steps) == (PathType.DIRECT, 1)
    assert "first supplier/customer" in accelerator.classification_basis
    hbm = _only(d, "compute_demand", "accelerator_demand", "hbm_demand")
    assert (hbm.path_type, hbm.value_chain_steps) == (PathType.SUPPLY_CHAIN, 2)
    assert "AI accelerator demand → HBM demand" in hbm.classification_basis
    assert set(hbm.sectors) == {EconomicSector.TECHNOLOGY}


def test_cross_sector_is_measured_from_where_the_effect_first_lands():
    d = _paths(AI_CAPEX)
    electricity = _only(d, "data_center_construction", "electricity_demand")
    assert electricity.path_type is PathType.CROSS_SECTOR
    assert electricity.classification_basis == (
        "Lands in technology at Data-center construction, then crosses into utilities at Electricity demand."
    )
    rates = _paths(RATE_SURPRISE)
    # the rate path lives in the rates sphere, bank margins in financials: a
    # first-order target is DIRECT wherever it lands, and bank earnings stay
    # with the same banks
    assert _only(rates, "bank_net_interest_margin").path_type is PathType.DIRECT
    assert _only(rates, "bank_net_interest_margin", "bank_earnings").path_type is PathType.DIRECT
    assert _only(rates, "treasury_yields", "equity_valuation_multiples").path_type is PathType.CROSS_SECTOR
    assert _only(rates, "borrowing_costs", "housing_activity").path_type is PathType.CROSS_SECTOR


def test_oil_has_customer_steps_and_an_honestly_empty_supply_chain():
    d = _paths(OPEC_CUT)
    refined = _only(d, "crude_oil_price", "refined_product_prices")
    assert (refined.path_type, refined.links[-1].value_chain_role.value) == (PathType.DIRECT, "CUSTOMER")
    assert _only(d, "crude_oil_price", "refined_product_prices", "transport_fuel_costs").path_type is (
        PathType.CROSS_SECTOR
    )
    assert _only(d, "crude_oil_price", "headline_inflation").path_type is PathType.CROSS_SECTOR
    assert d.count_by_type()[PathType.SUPPLY_CHAIN] == 0
    assert signal_path_view._EMPTY_TYPE[PathType.SUPPLY_CHAIN].startswith("No supply-chain path")


def test_classification_is_driven_by_the_reviewed_tables_not_by_labels():
    copper_in_tech = tuple(
        dataclasses.replace(p, sector=EconomicSector.TECHNOLOGY) if p.state_id == "copper_demand" else p
        for p in STATE_PROFILES
    )
    roles = {**CHANNEL_ROLES, TransmissionChannel.INPUT_DEMAND: None}
    library = SignalPathLibrary(profiles=copper_in_tech, channel_roles=roles, exposure_basis=EXPOSURE_BASIS)
    d = _paths(AI_CAPEX, library=library)
    assert _only(d, "data_center_construction", "copper_demand").path_type is PathType.DIRECT
    # INPUT_DEMAND no longer a value-chain step: accelerator -> HBM is one step (capacity) short of two
    assert _only(d, "compute_demand", "accelerator_demand", "hbm_demand").path_type is PathType.DIRECT


def test_path_type_is_a_primary_label_over_two_independent_characteristics():
    ai, oil = _paths(AI_CAPEX), _paths(OPEC_CUT)
    # a CROSS_SECTOR label never erases supply-chain depth ...
    gas = _only(ai, "data_center_construction", "electricity_demand", "gas_power_generation_demand")
    assert (gas.path_type, gas.value_chain_steps, gas.sector_transitions) == (PathType.CROSS_SECTOR, 2, 2)
    assert "also takes 2 supplier/customer steps" in gas.classification_basis
    fuel = _only(oil, "crude_oil_price", "refined_product_prices", "transport_fuel_costs")
    assert (fuel.path_type, fuel.value_chain_steps, fuel.sector_transitions) == (PathType.CROSS_SECTOR, 2, 1)
    # ... a SUPPLY_CHAIN path never crosses, and a DIRECT path may still take one step
    assert all(p.sector_transitions == 0 and p.value_chain_steps >= 2 for p in ai.of_type(PathType.SUPPLY_CHAIN))
    assert {p.value_chain_steps for p in ai.of_type(PathType.DIRECT)} == {0, 1}
    # both characteristics vary independently across the AI paths
    assert {(p.value_chain_steps >= 2, bool(p.sector_transitions)) for p in ai.paths} >= {
        (True, True), (True, False), (False, True), (False, False),
    }
    data = gas.model_dump()
    with pytest.raises(ValidationError, match="sector transition"):
        SignalPath(**{**data, "sector_transitions": 0})
    rows = signal_path_view.path_rows((gas,))
    assert (rows[0]["Value-chain steps"], rows[0]["Sector crossings"]) == (2, 2)


def test_a_state_outside_the_sector_table_is_unclassified_never_guessed():
    extra = [_model_claim("HBM demand", "advanced packaging equipment demand", Polarity.POSITIVE)]
    d = _paths(AI_CAPEX, graph_kwargs={"extra_claims": extra})
    (p,) = d.of_type(None)
    assert p.consequence_state == "advanced_packaging_equipment_demand"
    assert p.sectors[-1] is None and p.candidate_target_concept is None and p.sector_transitions is None
    assert p.status is PathStatus.UNRESOLVED
    assert StatusReason.UNCLASSIFIED_STATE in p.status_reasons
    assert "not in the reviewed sector table" in p.classification_basis


# ---------------------------------------------------------------------------
# ordered paths, depth, integration with the mechanism graph
# ---------------------------------------------------------------------------


def test_every_graph_path_becomes_exactly_one_ordered_signal_path():
    mg = _graph(AI_CAPEX)
    d = discover_signal_paths(mg)
    graph_paths = {(tp.state_ids, tp.edge_ids) for i in mg.implications for tp in i.paths}
    assert {(p.state_ids, tuple(link.edge_id for link in p.links)) for p in d.paths} == graph_paths
    assert len({p.signature for p in d.paths}) == len(d.paths) == len(graph_paths)
    assert {p.mechanism_graph_id for p in d.paths} == {mg.fingerprint()} == {d.mechanism_graph_id}
    edge_ids = {e.edge_id for e in mg.graph.edges}
    for p in d.paths:
        assert p.origin_event_id == mg.event_id and p.anchor_state == p.state_ids[0]
        assert p.anchor_channel is EconomicChannel.TECHNOLOGY_INVESTMENT
        assert all(link.edge_id in edge_ids for link in p.links)
        for i, link in enumerate(p.links):
            assert (link.source, link.target) == (p.state_ids[i], p.state_ids[i + 1])
        assert p.state_labels == tuple(mg.graph.state(s).label for s in p.state_ids)


def test_a_shuffled_or_misaligned_path_fails_validation():
    p = _only(_paths(AI_CAPEX), "compute_demand", "accelerator_demand", "hbm_demand")
    data = p.model_dump()
    with pytest.raises(ValidationError, match="chain"):
        SignalPath(**{**data, "links": tuple(reversed(data["links"]))})
    with pytest.raises(ValidationError, match="depth"):
        SignalPath(**{**data, "transmission_depth": 2})
    with pytest.raises(ValidationError, match="does not follow"):
        SignalPath(**{**data, "status": PathStatus.PROPOSED})


def test_order_survives_a_json_round_trip_and_fingerprints_are_deterministic():
    d = _paths(AI_CAPEX)
    restored = SignalPathDiscovery.model_validate_json(d.model_dump_json())
    assert restored == d
    assert [p.state_ids for p in restored.paths] == [p.state_ids for p in d.paths]
    assert _paths(AI_CAPEX).fingerprint() == d.fingerprint()
    assert d.plane == "HYPOTHESIS"
    with pytest.raises(ValidationError):
        SignalPathDiscovery(**{**d.model_dump(), "plane": "EVIDENCE"})


def test_depth_is_recorded_never_ranked():
    d = _paths(AI_CAPEX)
    for p in d.paths:
        assert p.transmission_depth == len(p.links) == len(p.state_ids) - 1
    shallow = _only(d, "ai_services_revenue")
    deep = _only(d, "compute_demand", "accelerator_demand", "hbm_demand", "foundry_capacity_expansion",
                 "semiconductor_equipment_demand")
    # a one-link path can be merely PROPOSED while a five-link path is RESEARCHABLE: status never reads depth
    assert (shallow.transmission_depth, shallow.status) == (1, PathStatus.PROPOSED)
    assert (deep.transmission_depth, deep.status) == (5, PathStatus.RESEARCHABLE)
    # and the second pass never reads depth either: a 4-link route and a 2-link route leave levels identical
    scan = _scan(AI_CAPEX)
    adj = adjust_impact(scan, d)
    assert all(r.adjusted_level is r.initial_level for a in adj.assessments for r in a.revisions)


def test_signal_paths_are_mandate_independent():
    mandates = [
        ALL_ASSESSABLE, ResearchMandate(allowed_domains=(MandateDomain.EQUITY,)),
        ResearchMandate(allowed_domains=(MandateDomain.FUTURES,), shorting_allowed=False),
        ResearchMandate(allowed_domains=(MandateDomain.ETF,), max_gross_leverage=1.0),
    ]
    for text in (AI_CAPEX, OPEC_CUT, RATE_SURPRISE):
        assert len({_paths(text, graph_kwargs={}).fingerprint()} | {
            discover_signal_paths(_graph(text, m)).fingerprint() for m in mandates
        }) == 1


def test_an_unseeded_event_has_no_paths_and_names_its_channel():
    d = _paths(GEOPOLITICAL)
    assert d.is_empty and EconomicChannel.GEOPOLITICAL_RISK in d.unseeded_channels
    assert signal_path_view.summary_line(d) is None


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_follows_typed_reasons():
    d = _paths(OPEC_CUT)
    assert _only(d, "crude_oil_price").status is PathStatus.RESEARCHABLE
    via_policy = _only(d, "crude_oil_price", "headline_inflation", "policy_rate_path")
    assert (via_policy.status, via_policy.status_reasons) == (PathStatus.PROPOSED, (StatusReason.WEAK_LINK,))
    for p in d.paths:
        assert (p.status is PathStatus.RESEARCHABLE) == (not p.status_reasons)


def test_no_event_direction_keeps_every_path_proposed_with_its_relative_sign():
    d = _paths(AI_NO_DIRECTION)
    assert d.paths and all(p.anchor_movement is Movement.UNKNOWN for p in d.paths)
    assert all(StatusReason.NO_EVENT_DIRECTION in p.status_reasons for p in d.paths)
    assert {p.status for p in d.paths} <= {PathStatus.PROPOSED, PathStatus.UNRESOLVED}
    fcf = [p for p in d.to("ai_spender_free_cash_flow")]
    assert {p.relative_sign for p in fcf} == {Polarity.NEGATIVE, Polarity.POSITIVE}


def test_an_anchor_conflict_leaves_its_paths_unresolved():
    from alpha_agent.news_alpha.transmission import EconomicState

    states = tuple(EconomicState(state_id=s, label=s, kind=StateKind.DEMAND) for s in ("a", "b"))
    rules = (
        AnchorRule(EconomicChannel.MONETARY_POLICY, "test-rule/1", "a",
                   (("TIGHTENING", Movement.UP), ("EASING", Movement.DOWN)), "test"),
        AnchorRule(EconomicChannel.CRUDE_OIL_SUPPLY, "test-rule/1", "a",
                   (("TIGHTER_BALANCE", Movement.DOWN), ("LOOSER_BALANCE", Movement.UP)), "test"),
    )
    claim = TransmissionClaim(
        source="a", target="b", channel=TransmissionChannel.DEMAND_PULL, polarity=Polarity.POSITIVE,
        lag=TransmissionLag.MONTHS, confidence=LinkConfidence.HIGH, origin=EdgeOrigin.MANUALLY_SEEDED,
        rationale="test link", provenance=(ProvenanceSource(kind=ProvenanceKind.PLATFORM_RULE, reference="t",
                                                            title="t"),),
    )
    library = TransmissionLibrary(states=states, claims=(claim,), anchors=rules)
    path_library = SignalPathLibrary(
        profiles=(StateProfile("a", EconomicSector.ENERGY, "a"), StateProfile("b", EconomicSector.ENERGY, "b")),
        channel_roles=CHANNEL_ROLES, exposure_basis=(),
    )
    d = _paths("The Fed raises rates as OPEC agrees a production cut", graph_kwargs={"library": library},
               library=path_library)
    (p,) = d.paths
    assert p.status is PathStatus.UNRESOLVED and StatusReason.ANCHOR_CONFLICT in p.status_reasons
    assert StatusReason.NO_EVENT_DIRECTION not in p.status_reasons


# ---------------------------------------------------------------------------
# conflicting paths
# ---------------------------------------------------------------------------


def test_opposing_paths_at_different_horizons_both_survive_cross_referenced():
    d = _paths(AI_CAPEX)
    down = _only(d, "ai_spender_free_cash_flow")
    up = _only(d, "ai_services_revenue", "ai_spender_free_cash_flow")
    assert (down.expected_direction.value, down.expected_horizon) == ("DOWN", TransmissionLag.MONTHS)
    assert (up.expected_direction.value, up.expected_horizon) == ("UP", TransmissionLag.YEARS)
    assert down.status is PathStatus.RESEARCHABLE
    assert StatusReason.OPPOSED_AT_SAME_HORIZON not in up.status_reasons  # only its LOW link keeps it PROPOSED
    assert down.opposing_path_ids == (up.path_id,) and up.opposing_path_ids == (down.path_id,)
    (conflict,) = [c for c in d.conflicts if c.consequence_state == "ai_spender_free_cash_flow"]
    assert conflict.separable_by_horizon and "different horizons" in conflict.note


def test_opposing_paths_at_the_same_horizon_are_both_unresolved():
    d = _paths(RATE_SURPRISE)
    earnings = d.to("bank_earnings")
    assert {p.expected_direction.value for p in earnings} == {"UP", "DOWN"}
    assert {p.expected_horizon for p in earnings} == {TransmissionLag.QUARTERS}
    assert all(p.status is PathStatus.UNRESOLVED for p in earnings)
    assert all(StatusReason.OPPOSED_AT_SAME_HORIZON in p.status_reasons for p in earnings)
    (conflict,) = d.conflicts
    assert not conflict.separable_by_horizon and "unresolved" in conflict.note
    # nothing is averaged: both directions stay on the record
    assert sorted(conflict.directions) == ["DOWN", "UP"]


def test_a_disputed_link_does_not_unresolve_independent_paths_beside_it():
    contradiction = _model_claim("electricity demand", "grid investment", Polarity.NEGATIVE)
    d = _paths(AI_CAPEX, graph_kwargs={"extra_claims": [contradiction]})
    independent = _only(d, "data_center_construction", "power_equipment_demand")
    assert independent.status is PathStatus.RESEARCHABLE and independent.opposing_path_ids
    through_dispute = [p for p in d.to("power_equipment_demand") if "grid_investment" in p.state_ids]
    assert through_dispute and all(StatusReason.UNRESOLVED_LINK in p.status_reasons for p in through_dispute)
    (conflict,) = [c for c in d.conflicts if c.consequence_state == "power_equipment_demand"]
    assert conflict.separable_by_horizon and "disputed" in conflict.note


# ---------------------------------------------------------------------------
# duplicates and model path proposals
# ---------------------------------------------------------------------------


def test_a_model_path_repeating_an_extracted_path_is_recorded_not_added():
    base = _paths(AI_CAPEX)
    proposal = _path_proposal((("AI capex", "compute demand", "GPU demand", "HBM"), "memory leg"))
    d = _paths(AI_CAPEX, proposals=[proposal])
    assert len(d.paths) == len(base.paths) and not d.rejected
    hbm = _only(d, "compute_demand", "accelerator_demand", "hbm_demand")
    assert hbm.provenance.origin is PathOrigin.GRAPH_EXTRACTED
    assert hbm.provenance.model_proposals == (proposal.reference,)
    assert hbm.status is _only(base, "compute_demand", "accelerator_demand", "hbm_demand").status
    (finding,) = d.findings
    assert finding.kind is PathFindingKind.DUPLICATE_PROPOSAL and finding.path_ids == (hbm.path_id,)


def test_the_same_route_from_two_events_shares_a_signature_not_a_path_id():
    a, b = _paths(AI_CAPEX), _paths(AI_CAPEX_PARAPHRASE)
    assert a.event_id != b.event_id
    assert {p.signature for p in a.paths} == {p.signature for p in b.paths}
    assert not {p.path_id for p in a.paths} & {p.path_id for p in b.paths}


def test_a_valid_walk_beyond_the_enumerated_paths_is_added_as_model_proposed():
    d = _paths(AI_CAPEX, graph_kwargs={"max_hops": 1},
               proposals=[_path_proposal((("AI capex", "AI services revenue", "hyperscaler free cash flow"),
                                          "revenue eventually lands in cash flow"))])
    (added,) = [p for p in d.paths if p.provenance.origin is PathOrigin.MODEL_PROPOSED]
    assert added.state_ids == ("ai_infrastructure_investment", "ai_services_revenue", "ai_spender_free_cash_flow")
    assert added.path_type is PathType.DIRECT and added.provenance.model_proposals
    assert any(f.kind is PathFindingKind.MODEL_PATH_ADDED for f in d.findings)


@pytest.mark.parametrize(("states", "rationale", "reason"), [
    (("AI infrastructure investment", "data center buildout", "NVDA"), "ticker", RejectionReason.UNKNOWN_STATE),
    (("AI infrastructure investment", "Copper futures"), "an instrument", RejectionReason.UNKNOWN_STATE),
    (("compute demand", "accelerator demand"), "not an anchor", RejectionReason.NOT_FROM_AN_ANCHOR),
    (("AI capex", "HBM demand"), "skips links", RejectionReason.NOT_A_GRAPH_LINK),
    (("AI capex", "compute demand", "AI capex"), "loops", RejectionReason.REPEATS_A_STATE),
    (("AI capex", "compute demand"), "BUY the spenders", RejectionReason.TRADE_EXPRESSION),
    (("AI capex", "compute demand"), "go long accelerators via call options", RejectionReason.TRADE_EXPRESSION),
])
def test_invalid_model_paths_are_rejected_with_typed_reasons(states, rationale, reason):
    d = _paths(AI_CAPEX, proposals=[_path_proposal((states, rationale))])
    (rejected,) = d.rejected
    assert reason in rejected.reasons and rejected.status is PathStatus.REJECTED
    assert rejected.states == states and rejected.detail.startswith("Rejected:")
    assert len(d.paths) == len(_paths(AI_CAPEX).paths)
    assert d.count_by_status()[PathStatus.REJECTED] == 1


def test_an_economic_rationale_that_says_sell_is_not_mistaken_for_a_trade():
    d = _paths(AI_CAPEX, proposals=[_path_proposal(
        (("AI capex", "AI services revenue"), "the capacity is built to sell AI services to customers who buy them"),
    )])
    assert not d.rejected


@pytest.mark.parametrize("smuggled", ["path_type", "status", "direction", "symbol", "ticker", "instrument", "trade"])
def test_a_path_proposal_cannot_carry_a_type_status_direction_or_instrument(smuggled):
    with pytest.raises(ValidationError):
        ProposedSignalPath(states=("AI capex", "compute demand"), rationale="fixture", **{smuggled: "X"})


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def test_path_provenance_is_its_links_provenance():
    d = _paths(AI_CAPEX)
    electricity = _only(d, "data_center_construction", "electricity_demand")
    prov = electricity.provenance
    assert prov.origin is PathOrigin.GRAPH_EXTRACTED and prov.seed_rules == (AI_INFRASTRUCTURE_SEED,)
    assert any("Energy and AI" in t for t in prov.verified_sources)
    assert any("DOE" in t for t in prov.verified_sources)
    assert {"path-extraction/1", "path-type/1", "path-status/1", "sign-propagation/1"} <= set(prov.rules)
    compute = _only(d, "compute_demand")
    assert compute.provenance.verified_sources == () and compute.provenance.model_proposals == ()


def test_model_links_carry_their_model_output_into_path_provenance():
    extra = [_model_claim("HBM demand", "advanced packaging equipment demand", Polarity.POSITIVE,
                          lag="QUARTERS", confidence="HIGH")]
    d = _paths(AI_CAPEX, graph_kwargs={"extra_claims": extra})
    (p,) = d.of_type(None)
    assert p.provenance.model_proposals == ("scripted-fixture#test",)
    assert StatusReason.UNREVIEWED_MODEL_LINK in p.status_reasons
    assert not p.links[-1].reviewed and p.links[0].reviewed


def test_consequence_fields_come_from_the_graph_and_the_reviewed_profiles():
    d = _paths(AI_CAPEX)
    copper = _only(d, "data_center_construction", "electricity_demand", "grid_investment", "copper_demand")
    assert (copper.consequence_label, copper.consequence_kind) == ("Copper demand", StateKind.DEMAND)
    assert copper.candidate_target_concept == "Copper (industrial-metal demand)"
    assert copper.expected_horizon is TransmissionLag.YEARS and copper.confidence is LinkConfidence.MEDIUM
    assert copper.expected_direction.value == "UP" and copper.relative_sign is Polarity.POSITIVE


# ---------------------------------------------------------------------------
# initial vs mechanism-adjusted impact
# ---------------------------------------------------------------------------


def _adjust(text: str, mandate: ResearchMandate = ALL_ASSESSABLE):
    scan = _scan(text, mandate)
    return scan, adjust_impact(scan, discover_signal_paths(build_economic_mechanism_graph(scan)))


def _revision(adj, domain: MandateDomain, label: str):
    return next(r for r in adj.assessment(domain).revisions if r.exposure_label == label)


def test_the_initial_scan_is_never_modified_and_both_levels_are_kept():
    for text in (AI_CAPEX, OPEC_CUT, RATE_SURPRISE, GEOPOLITICAL):
        scan = _scan(text)
        before = scan.model_dump_json()
        adj = adjust_impact(scan, discover_signal_paths(build_economic_mechanism_graph(scan)))
        assert scan.model_dump_json() == before
        assert adj.initial_scan_fingerprint == scan_fingerprint(scan)
        assert scan.assessment_pass == "INITIAL" and adj.assessment_pass == "MECHANISM_ADJUSTED"
        for initial in scan.assessments:
            second = adj.assessment(initial.asset_domain)
            assert second.initial_level is initial.impact_level
            assert [r.initial_level for r in second.revisions] == [c.impact_level for c in initial.contributions]


def test_initial_contributions_record_the_impact_level_rule_per_exposure():
    scan = _scan(AI_CAPEX)
    for a in scan.assessments:
        if a.impact_level is not None and a.contributions:
            assert max(c.impact_level.rank for c in a.contributions) == a.impact_level.rank
    copper = next(c for c in scan.assessment(MandateDomain.FUTURES).contributions if c.exposure_label == "Copper futures")
    assert (copper.symbols, copper.impact_level) == (("HG",), ImpactLevel.LOW)


def test_link_confidence_never_becomes_exposure_strength_or_moves_a_level():
    for text in (AI_CAPEX, OPEC_CUT, RATE_SURPRISE):
        _, adj = _adjust(text)
        assert not adj.changed
        for a in adj.assessments:
            assert a.change in (LevelChange.UNCHANGED, LevelChange.NOT_ASSESSED)
            assert a.adjusted_level is a.initial_level
            for r in a.revisions:
                assert r.adjusted_level is r.initial_level and not r.magnitude_established
                assert r.uncertainty == (MAGNITUDE_NOTE,)
    # the leak: an all-HIGH-confidence route to a TERTIARY exposure used to raise it; now the level holds
    _, adj = _adjust(AI_CAPEX)
    utilities = _revision(adj, MandateDomain.ETF, "Utilities sector ETF")
    assert (utilities.mechanism_support, utilities.mechanism_confidence) == (
        MechanismSupport.CORROBORATED, LinkConfidence.HIGH,
    )
    assert (utilities.exposure_strength.value, utilities.initial_level, utilities.adjusted_level) == (
        "TERTIARY", ImpactLevel.LOW, ImpactLevel.LOW,
    )
    # and no field on a revision carries a strength derived from the mechanism
    from alpha_agent.news_alpha.impact_adjustment import ExposureRevision

    assert not {"path_strength", "adjusted_strength", "initial_strength"} & set(ExposureRevision.model_fields)


def test_mechanism_support_is_reported_beside_the_level():
    _, ai = _adjust(AI_CAPEX)
    copper = _revision(ai, MandateDomain.FUTURES, "Copper futures")
    assert (copper.mechanism_support, copper.mechanism_confidence, copper.initial_level, copper.adjusted_level) == (
        MechanismSupport.CORROBORATED, LinkConfidence.MEDIUM, ImpactLevel.LOW, ImpactLevel.LOW,
    )
    assert [(b.label, b.movement.value) for b in copper.basis] == [("Copper demand", "UP")]
    assert ai.assessment(MandateDomain.FUTURES).corroborated() == 3
    # the cited route prefers researchable and unopposed, never the contested FCF path
    nasdaq = _revision(ai, MandateDomain.FUTURES, "Nasdaq-100 index futures")
    assert "AI accelerator demand" in nasdaq.rationale and "free cash flow" not in nasdaq.rationale
    _, fed = _adjust(RATE_SURPRISE)
    gold = _revision(fed, MandateDomain.FUTURES, "Gold futures")
    assert (gold.mechanism_support, gold.mechanism_confidence) == (MechanismSupport.PROPOSED_ONLY, LinkConfidence.LOW)
    assert (gold.initial_level, gold.adjusted_level) == (ImpactLevel.HIGH, ImpactLevel.HIGH)


def test_an_unseeded_channel_leaves_every_initial_level_standing():
    _, adj = _adjust(GEOPOLITICAL)
    for a in adj.assessments:
        assert a.change in (LevelChange.UNCHANGED, LevelChange.NOT_ASSESSED)
        assert all(r.mechanism_support is MechanismSupport.CHANNEL_NOT_SEEDED for r in a.revisions)
        assert all(r.initial_level is r.adjusted_level and r.cited_path_id is None for r in a.revisions)
    assert not adj.changed


def test_an_unreached_basis_is_a_library_gap_not_evidence_against():
    rebased = tuple(
        ExposureBasis(b.channel, b.exposure_label, ("bank_earnings",)) if b.exposure_label == "Copper futures" else b
        for b in EXPOSURE_BASIS
    )
    library = SignalPathLibrary(profiles=STATE_PROFILES, channel_roles=CHANNEL_ROLES, exposure_basis=rebased)
    scan = _scan(AI_CAPEX)
    adj = adjust_impact(scan, discover_signal_paths(build_economic_mechanism_graph(scan)), library=library)
    copper = _revision(adj, MandateDomain.FUTURES, "Copper futures")
    assert copper.mechanism_support is MechanismSupport.NOT_REACHED and copper.mechanism_confidence is None
    assert copper.adjusted_level is copper.initial_level is ImpactLevel.LOW
    assert copper.basis[0].movement is None and "library gap" in copper.rationale


def test_each_exposure_is_revised_only_by_its_own_channels_paths():
    text = "The Fed unexpectedly raised rates as OPEC+ agreed a production cut"
    scan = _scan(text)
    d = discover_signal_paths(build_economic_mechanism_graph(scan))
    adj = adjust_impact(scan, d)
    for a in adj.assessments:
        for r in a.revisions:
            if r.cited_path_id is not None:
                assert d.path(r.cited_path_id).anchor_channel is r.channel


def test_excluded_domains_stay_not_assessed_and_mismatched_inputs_are_refused():
    scan, adj = _adjust(AI_CAPEX, ResearchMandate(allowed_domains=(MandateDomain.EQUITY,)))
    assert adj.assessment(MandateDomain.FUTURES).change is LevelChange.NOT_ASSESSED
    assert adj.assessment(MandateDomain.FUTURES).adjusted_level is None
    with pytest.raises(ValueError, match="different event"):
        adjust_impact(scan, _paths(OPEC_CUT))


# ---------------------------------------------------------------------------
# reviewed tables
# ---------------------------------------------------------------------------


def test_the_reviewed_tables_cover_the_catalog_and_every_seeded_exposure():
    _validate(DEFAULT_PATH_LIBRARY)
    assert set(CHANNEL_ROLES) == set(TransmissionChannel)
    with pytest.raises(ValueError, match="drifted"):
        _validate(dataclasses.replace(DEFAULT_PATH_LIBRARY, profiles=STATE_PROFILES[1:]))
    with pytest.raises(ValueError, match="no economic basis"):
        _validate(dataclasses.replace(DEFAULT_PATH_LIBRARY, exposure_basis=EXPOSURE_BASIS[1:]))
    with pytest.raises(ValueError, match="does not exist"):
        _validate(dataclasses.replace(DEFAULT_PATH_LIBRARY, exposure_basis=(
            *EXPOSURE_BASIS, ExposureBasis(EconomicChannel.TECHNOLOGY_INVESTMENT, "Semiconductor ETF", ("hbm_demand",)),
        )))


def test_the_sector_table_is_documented_as_an_internal_transmission_taxonomy():
    from alpha_agent.news_alpha import signal_path_library

    doc = " ".join(signal_path_library.__doc__.split())
    assert "INTERNAL TRANSMISSION TAXONOMY" in doc
    assert "NOT GICS, NAICS, SIC" in doc and "NOT an Asset Expression mapping" in doc
    doc_md = (REPO_ROOT / "docs" / "NEWS_ALPHA_PHASE_C_SIGNAL_PATH_DISCOVERY.md").read_text(encoding="utf-8")
    assert "internal transmission taxonomy" in doc_md


def test_target_concepts_are_concepts_not_instruments():
    import re

    from alpha_agent.news_alpha import CHANNELS, resolve_allowed_universe

    universe = resolve_allowed_universe(ResearchMandate(allowed_domains=tuple(MandateDomain)), capabilities=CAPS)
    symbols = {i.symbol for scope in universe.scopes for i in scope.instruments}
    symbols |= {sym for c in CHANNELS for e in c.exposures for sym in e.symbols}
    assert len(symbols) > 20
    for p in STATE_PROFILES:
        words = set(re.findall(r"[A-Za-z]+", p.target_concept))
        assert not words & symbols, (p.state_id, words & symbols)
        # word-bounded and not part of a hyphenated compound ("short-rate" is a rates term, not a trade)
        assert not re.search(r"(?i)\b(futures|etf|stock|shares|contract|long|short|buy|sell)\b(?!-)", p.target_concept), p


# ---------------------------------------------------------------------------
# boundaries: no trade recommendation, no registry, plane separation
# ---------------------------------------------------------------------------


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


@pytest.mark.parametrize("path", PHASE_C_SOURCES, ids=lambda p: p.name)
def test_phase_c_never_imports_the_evidence_plane_or_an_llm(path):
    forbidden = (
        "alpha_agent.alpha_graph", "alpha_agent.alpha_memory", "alpha_agent.registry", "alpha_agent.knowledge",
        "alpha_agent.translation", "alpha_agent.validation", "alpha_agent.agents", "anthropic",
    )
    for imported in _imports(path):
        assert not any(imported == f or imported.startswith(f + ".") for f in forbidden), (path.name, imported)
    source = path.read_text(encoding="utf-8")
    for token in ("ExperimentRegistry", "insert_", "INSERT", "UPDATE ", "requests.", "urllib", "httpx", "open("):
        assert token not in source, (path.name, token)


def test_no_schema_carries_a_trade_instrument_return_or_verdict_field():
    from alpha_agent.news_alpha import impact_adjustment, signal_paths

    forbidden = {
        "expected_return", "return", "probability", "probability_of_profit", "p_value", "sharpe", "score",
        "weight", "portfolio_weight", "position_size", "buy", "sell", "action", "trade", "side", "verdict",
        "scientific_verdict", "experiment_id", "experiment_identity", "alpha", "forecast", "target_price",
        "symbol", "ticker", "root", "instrument", "contract", "asset", "order", "rank",
    }
    models = [
        signal_paths.SignalPath, signal_paths.PathLink, signal_paths.PathProvenance, signal_paths.PathConflict,
        signal_paths.RejectedPathProposal, signal_paths.PathFinding, signal_paths.SignalPathDiscovery,
        signal_paths.ProposedSignalPath, signal_paths.SignalPathProposal,
        impact_adjustment.BasisConsequence, impact_adjustment.MechanismAdjustedImpact,
        impact_adjustment.MechanismAdjustedImpactAssessment,
    ]
    for model in models:
        assert not (set(model.model_fields) & forbidden), model.__name__
    # ExposureRevision carries the Phase A candidate symbols it revises -- and nothing a trade needs
    assert not (set(impact_adjustment.ExposureRevision.model_fields) & (forbidden - {"symbol"}))
    for text in (AI_CAPEX, OPEC_CUT, RATE_SURPRISE):
        _, adj = _adjust(text)
        rendered = (_paths(text).model_dump_json() + adj.model_dump_json()).upper()
        for word in ('"BUY"', '"SELL"', '"LONG"', '"SHORT"', '"PASS"', '"FAIL"', "STRONG BUY", "GUARANTEED"):
            assert word not in rendered


def test_discovery_and_adjustment_never_touch_the_registry():
    if not REGISTRY_PATH.exists():
        pytest.skip("Phase 14 registry sqlite not present in this checkout")
    before = hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
    for text in (AI_CAPEX, OPEC_CUT, RATE_SURPRISE, GEOPOLITICAL):
        _adjust(text)
    assert hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest() == before


# ---------------------------------------------------------------------------
# view helpers (pure)
# ---------------------------------------------------------------------------


def test_view_rows_lead_with_consequence_and_status_and_never_use_verdict_colors():
    d = _paths(AI_CAPEX)
    assert signal_path_view.summary_line(d) == (
        "Signal paths: 6 direct · 7 supply-chain · 7 cross-sector -- 17 researchable."
    )
    rows = signal_path_view.path_rows(d.of_type(PathType.SUPPLY_CHAIN))
    assert list(rows[0])[:3] == ["Economic consequence", "Direction", "Status"]
    assert rows[0]["Economic consequence"] == "HBM demand" and rows[0]["Verified links"] == "1 of 3"
    _, adj = _adjust(AI_CAPEX)
    row = signal_path_view.mechanism_support_row(adj)
    assert "After mechanism graph" in row and "levels kept (economic magnitude not established)" in row
    assert "Futures 3 of 3 · ETF 2 of 2 · Equity 2 of 2" in row
    assert "▲" not in row and "IMPACT_" not in row
    assert palette.GREEN not in row and palette.RED not in row
    assert signal_path_view.adjusted_summary_text(adj) == (
        "levels kept (Futures MEDIUM · ETF MEDIUM · Equity HIGH) -- economic magnitude not established; mechanism "
        "corroborates exposures: Futures 3 of 3, ETF 2 of 2, Equity 2 of 2"
    )
    rates = _paths(RATE_SURPRISE)
    assert "Bank earnings: opposing paths at the same horizon" in signal_path_view.summary_line(rates)
    json.dumps(rows)  # plain, serializable table values
