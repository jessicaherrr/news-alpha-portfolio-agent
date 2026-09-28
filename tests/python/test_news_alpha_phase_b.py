"""News Alpha Phase B -- Initial Impact Scan -> ECONOMIC MECHANISM GRAPH.

Pure, offline tests: fixed `DomainCapabilities`, fixed timestamps, no
network, no LLM. Covers the typed graph, graph safety (duplicates,
conflicts, cycles, unknowns, missing provenance), the provenance rule
(a model proposal never promotes itself), serialization, the Phase A
integration, mandate independence, and the plane boundaries.
"""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.market_intel import mapping
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.news_alpha import (
    DomainCapabilities,
    EconomicMechanismGraph,
    EconomicState,
    EdgeOrigin,
    ImpliedMovement,
    InitialImpactScan,
    LinkConfidence,
    MandateDomain,
    Movement,
    Polarity,
    ProvenanceSource,
    ResearchMandate,
    SupportStatus,
    TransmissionChannel,
    TransmissionClaim,
    TransmissionLag,
    TransmissionLibrary,
    TransmissionProposal,
    UserDescribedEvent,
    assemble_graph,
    build_economic_mechanism_graph,
    claims_from_proposal,
    scan_initial_impact,
)
from alpha_agent.news_alpha.channels import EconomicChannel
from alpha_agent.news_alpha.transmission import (
    GraphIssueKind,
    LoopType,
    ProvenanceKind,
    StateKind,
    Verification,
    canonical_state_id,
)
from alpha_agent.news_alpha.transmission_library import (
    ANCHOR_RULES,
    DEFAULT_LIBRARY,
    SEEDED_CLAIMS,
    STATE_CATALOG,
    AnchorRule,
    _validate,
)
from alpha_agent.ui import mechanism_graph_view, palette
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "python" / "alpha_agent" / "news_alpha"
PHASE_B_SOURCES = [
    PACKAGE_DIR / "transmission.py", PACKAGE_DIR / "transmission_library.py", PACKAGE_DIR / "mechanism_graph.py",
    REPO_ROOT / "python" / "alpha_agent" / "ui" / "mechanism_graph_view.py",
]
REGISTRY_PATH = REPO_ROOT / "data" / "registry" / "experiments.sqlite"

CAPS = DomainCapabilities(etf_market_data_acquired=True, equity_market_data_acquired=False)
AT = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
EQUITY_FUTURES = ResearchMandate(allowed_domains=(MandateDomain.EQUITY, MandateDomain.FUTURES))

AI_CAPEX = "Hyperscalers announce a major AI infrastructure spending increase, lifting data center capex guidance"
AI_CAPEX_CUT = "Hyperscalers cut spending on AI data centers as capex guidance falls"
OPEC_CUT = "OPEC+ agrees a production cut of 1 million barrels per day"
RATE_SURPRISE = "Rate-policy surprise: the Fed unexpectedly raised rates by 50 basis points"


def _scan(text: str, mandate: ResearchMandate = EQUITY_FUTURES) -> InitialImpactScan:
    return scan_initial_impact(UserDescribedEvent.create(text, described_at=AT), mandate, capabilities=CAPS)


def _graph(text: str, mandate: ResearchMandate = EQUITY_FUTURES, **kwargs) -> EconomicMechanismGraph:
    return build_economic_mechanism_graph(_scan(text, mandate), **kwargs)


def _rule(ref: str = "test-rule/1") -> ProvenanceSource:
    return ProvenanceSource(kind=ProvenanceKind.PLATFORM_RULE, reference=ref, title="test rule")


def _claim(source: str, target: str, polarity: Polarity = Polarity.POSITIVE, **overrides) -> TransmissionClaim:
    fields = {
        "source": source, "target": target, "channel": TransmissionChannel.DEMAND_PULL, "polarity": polarity,
        "lag": TransmissionLag.MONTHS, "confidence": LinkConfidence.MEDIUM, "origin": EdgeOrigin.MANUALLY_SEEDED,
        "rationale": "test relationship", "provenance": (_rule(),),
    }
    fields.update(overrides)
    return TransmissionClaim(**fields)


def _verified_doc(ref: str = "https://example.org/report") -> ProvenanceSource:
    return ProvenanceSource(
        kind=ProvenanceKind.EXTERNAL_DOCUMENT, reference=ref, title="A report", claim="X raises Y.",
        verification=Verification.VERIFIED, verified_on=date(2026, 9, 25),
    )


def _states(*ids: str) -> tuple[EconomicState, ...]:
    return tuple(EconomicState(state_id=i, label=i.replace("_", " "), kind=StateKind.DEMAND) for i in ids)


# ---------------------------------------------------------------------------
# typed nodes / edges / provenance
# ---------------------------------------------------------------------------


def test_state_ids_are_canonical_and_states_forbid_unknown_fields():
    with pytest.raises(ValidationError):
        EconomicState(state_id="HBM Demand", label="HBM demand", kind=StateKind.DEMAND)
    with pytest.raises(ValidationError):
        EconomicState(state_id="hbm_demand", label="HBM", kind=StateKind.DEMAND, symbol="MU")
    assert canonical_state_id(" HBM  Demand ") == canonical_state_id("hbm-demand") == "hbm_demand"
    assert canonical_state_id("2025 capex") == "s_2025_capex"


def test_economic_state_has_no_tradable_instrument_field():
    tradable = {"symbol", "root", "root_symbol", "ticker", "instrument", "asset_domain", "domain", "exchange"}
    assert not (set(EconomicState.model_fields) & tradable)


def test_verification_is_earned_never_self_declared():
    with pytest.raises(ValidationError, match="verified_on"):
        ProvenanceSource(kind=ProvenanceKind.EXTERNAL_DOCUMENT, reference="u", title="t", claim="c",
                         verification=Verification.VERIFIED)
    for kind in (ProvenanceKind.MODEL_OUTPUT, ProvenanceKind.PLATFORM_RULE):
        with pytest.raises(ValidationError, match="never verified evidence"):
            ProvenanceSource(kind=kind, reference="u", title="t", claim="c", verification=Verification.VERIFIED,
                             verified_on=date(2026, 9, 25))
    with pytest.raises(ValidationError, match="not VERIFIED"):
        ProvenanceSource(kind=ProvenanceKind.EXTERNAL_DOCUMENT, reference="u", title="t", verified_on=date(2026, 9, 25))
    assert _verified_doc().counts_as_support
    assert not _rule().counts_as_support


def test_a_claim_has_no_status_field_to_assert_itself_supported():
    assert not ({"support", "status", "support_status", "verified"} & set(TransmissionClaim.model_fields))
    with pytest.raises(ValidationError):
        _claim("a", "b", support=SupportStatus.SUPPORTED)


def test_transmission_vocabulary_is_disjoint_from_the_quant_mechanism_taxonomy():
    quant = {m.value for m in EconomicMechanism}
    assert not ({c.value for c in TransmissionChannel} & quant)
    assert not ({k.value for k in StateKind} & quant)


# ---------------------------------------------------------------------------
# graph construction from the Phase A scan (the AI-infrastructure slice)
# ---------------------------------------------------------------------------


def test_ai_infrastructure_event_builds_both_prompt_chains_upward():
    mg = _graph(AI_CAPEX)
    assert [(a.state_id, a.movement) for a in mg.anchors] == [("ai_infrastructure_investment", Movement.UP)]
    for sid in ("compute_demand", "accelerator_demand", "hbm_demand", "foundry_capacity_expansion",
                "semiconductor_equipment_demand", "data_center_construction", "electricity_demand",
                "grid_investment", "power_equipment_demand"):
        assert mg.implication(sid).movement is ImpliedMovement.UP, sid

    def has_path(target: str, chain: tuple[str, ...]) -> bool:
        return any(p.state_ids == chain for p in mg.implication(target).paths)

    assert has_path("hbm_demand", ("ai_infrastructure_investment", "compute_demand", "accelerator_demand", "hbm_demand"))
    assert has_path("semiconductor_equipment_demand", (
        "ai_infrastructure_investment", "compute_demand", "accelerator_demand", "advanced_foundry_utilization",
        "foundry_capacity_expansion", "semiconductor_equipment_demand",
    ))
    assert has_path("power_equipment_demand", (
        "ai_infrastructure_investment", "data_center_construction", "electricity_demand", "grid_investment",
        "power_equipment_demand",
    ))


def test_a_spending_decrease_flips_every_implied_direction_on_the_same_graph():
    up, down = _graph(AI_CAPEX), _graph(AI_CAPEX_CUT)
    assert down.anchors[0].movement is Movement.DOWN
    assert {e.edge_id for e in up.graph.edges} == {e.edge_id for e in down.graph.edges}
    flip = {ImpliedMovement.UP: ImpliedMovement.DOWN, ImpliedMovement.DOWN: ImpliedMovement.UP,
            ImpliedMovement.MIXED: ImpliedMovement.MIXED}
    for imp in up.implications:
        assert down.implication(imp.state_id).movement is flip[imp.movement], imp.state_id


def test_the_graph_is_library_driven_not_hard_coded():
    library = TransmissionLibrary(
        states=_states("ai_infrastructure_investment", "robotics_demand"),
        claims=(_claim("ai_infrastructure_investment", "robotics_demand"),),
        anchors=(AnchorRule(EconomicChannel.TECHNOLOGY_INVESTMENT, "test-rule/1", "ai_infrastructure_investment",
                            (("SPENDING_INCREASE", Movement.UP), ("SPENDING_DECREASE", Movement.DOWN)), "test"),),
    )
    mg = _graph(AI_CAPEX, library=library)
    assert [s.state_id for s in mg.graph.states] == ["ai_infrastructure_investment", "robotics_demand"]
    assert mg.implication("robotics_demand").movement is ImpliedMovement.UP


def test_seeded_links_are_supported_only_where_a_verified_source_exists():
    mg = _graph(AI_CAPEX)
    by_pair = {(e.source, e.target): e for e in mg.graph.edges}
    for pair in (("accelerator_demand", "hbm_demand"), ("data_center_construction", "electricity_demand"),
                 ("foundry_capacity_expansion", "semiconductor_equipment_demand"),
                 ("ai_infrastructure_investment", "ai_spender_free_cash_flow")):
        assert by_pair[pair].support is SupportStatus.SUPPORTED, pair
    # no verified source was found for the grid link -- it stays a hypothesis
    assert by_pair[("electricity_demand", "grid_investment")].support is SupportStatus.PROPOSED
    assert by_pair[("ai_infrastructure_investment", "compute_demand")].support is SupportStatus.PROPOSED


# ---------------------------------------------------------------------------
# duplicate normalization
# ---------------------------------------------------------------------------


def test_duplicate_claims_merge_across_aliases_preserving_every_claim_and_source():
    model = _claim("AI Accelerator Demand", "high bandwidth memory demand", origin=EdgeOrigin.MODEL_PROPOSED,
                   provenance=(ProvenanceSource(kind=ProvenanceKind.MODEL_OUTPUT, reference="m#1", title="m"),),
                   confidence=LinkConfidence.LOW)
    seeded = next(c for c in SEEDED_CLAIMS if (c.source, c.target) == ("accelerator_demand", "hbm_demand"))
    graph = assemble_graph([seeded, model], catalog=STATE_CATALOG)
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert (edge.source, edge.target) == ("accelerator_demand", "hbm_demand")
    assert edge.claims == (seeded, model)
    assert edge.origins == (EdgeOrigin.MANUALLY_SEEDED, EdgeOrigin.MODEL_PROPOSED)
    assert edge.confidence is LinkConfidence.LOW  # most conservative stated
    assert edge.support is SupportStatus.SUPPORTED  # the seed's verified source still backs it
    assert graph.issues_of(GraphIssueKind.DUPLICATE_MERGED)


def test_lag_disagreement_is_reported_not_silently_resolved():
    graph = assemble_graph(
        [_claim("a", "b", lag=TransmissionLag.WEEKS), _claim("a", "b", lag=TransmissionLag.YEARS)],
        catalog=_states("a", "b"),
    )
    assert graph.edges[0].lag is TransmissionLag.UNKNOWN
    assert {c.lag for c in graph.edges[0].claims} == {TransmissionLag.WEEKS, TransmissionLag.YEARS}
    assert graph.issues_of(GraphIssueKind.LAG_DISAGREEMENT)


def test_conflicting_state_declarations_and_ambiguous_names_are_reported():
    catalog = (
        *_states("a", "b"),
        EconomicState(state_id="a", label="a", kind=StateKind.PRICE),
        EconomicState(state_id="c", label="c", kind=StateKind.DEMAND, aliases=("b",)),
    )
    graph = assemble_graph([_claim("a", "b")], catalog=catalog)
    assert len(graph.issues_of(GraphIssueKind.STATE_DEFINITION_CONFLICT)) == 2
    assert graph.state("a").kind is StateKind.DEMAND  # the first declaration wins, visibly


def test_undeclared_states_are_admitted_as_unspecified_and_flagged():
    graph = assemble_graph([_claim("hbm_demand", "Undersea cable demand")], catalog=STATE_CATALOG)
    new = graph.state("undersea_cable_demand")
    assert new.kind is StateKind.UNSPECIFIED and not new.declared
    assert graph.issues_of(GraphIssueKind.UNDECLARED_STATE)


# ---------------------------------------------------------------------------
# conflict preservation
# ---------------------------------------------------------------------------


def test_opposite_sign_claims_are_both_kept_and_unresolved():
    graph = assemble_graph([_claim("a", "b"), _claim("a", "b", Polarity.NEGATIVE)], catalog=_states("a", "b"))
    assert {e.polarity for e in graph.edges} == {Polarity.POSITIVE, Polarity.NEGATIVE}
    assert all(e.support is SupportStatus.UNRESOLVED for e in graph.edges)
    (issue,) = graph.issues_of(GraphIssueKind.DIRECTION_CONFLICT)
    assert set(issue.edge_ids) == {e.edge_id for e in graph.edges}


def test_a_direction_conflict_propagates_as_mixed_not_averaged():
    library = TransmissionLibrary(
        states=_states("ai_infrastructure_investment", "b"),
        claims=(_claim("ai_infrastructure_investment", "b"),
                _claim("ai_infrastructure_investment", "b", Polarity.NEGATIVE)),
        anchors=DEFAULT_LIBRARY.anchors,
    )
    mg = _graph(AI_CAPEX, library=library)
    assert mg.implication("b").movement is ImpliedMovement.MIXED
    assert all(p.weakest_support is SupportStatus.UNRESOLVED for p in mg.implication("b").paths)


def test_spender_side_disagreement_is_exposed_with_both_paths():
    fcf = _graph(AI_CAPEX).implication("ai_spender_free_cash_flow")
    assert fcf.movement is ImpliedMovement.MIXED
    by_move = {p.movement: p for p in fcf.paths}
    assert by_move[ImpliedMovement.DOWN].slowest_lag is TransmissionLag.MONTHS
    assert by_move[ImpliedMovement.UP].slowest_lag is TransmissionLag.YEARS
    assert by_move[ImpliedMovement.UP].weakest_confidence is LinkConfidence.LOW
    assert "Paths disagree" in fcf.note


def test_rate_tightening_leaves_bank_earnings_mixed():
    mg = _graph(RATE_SURPRISE)
    assert mg.anchors[0].movement is Movement.UP
    assert mg.implication("treasury_yields").movement is ImpliedMovement.UP
    assert mg.implication("equity_valuation_multiples").movement is ImpliedMovement.DOWN
    assert mg.implication("bank_earnings").movement is ImpliedMovement.MIXED


def test_channels_that_move_one_root_in_opposite_directions_leave_it_unknown():
    root_rule = AnchorRule(EconomicChannel.MONETARY_POLICY, "test-rule/1", "a",
                           (("TIGHTENING", Movement.UP), ("EASING", Movement.DOWN)), "test")
    oil_rule = AnchorRule(EconomicChannel.CRUDE_OIL_SUPPLY, "test-rule/1", "a",
                          (("TIGHTER_BALANCE", Movement.DOWN), ("LOOSER_BALANCE", Movement.UP)), "test")
    library = TransmissionLibrary(states=_states("a", "b"), claims=(_claim("a", "b"),), anchors=(root_rule, oil_rule))
    mg = _graph("The Fed raises rates as OPEC agrees a production cut", library=library)
    assert [a.movement for a in mg.anchors] == [Movement.UNKNOWN]
    assert any(i.kind is GraphIssueKind.ANCHOR_CONFLICT for i in mg.stage_issues)


# ---------------------------------------------------------------------------
# provenance: unknown / missing / model-proposed
# ---------------------------------------------------------------------------


def test_missing_provenance_is_unresolved():
    bare = _claim("a", "b", provenance=())
    external_without_document = _claim("b", "c", origin=EdgeOrigin.EXTERNALLY_SOURCED)
    graph = assemble_graph([bare, external_without_document], catalog=_states("a", "b", "c"))
    assert all(e.support is SupportStatus.UNRESOLVED for e in graph.edges)
    (issue,) = graph.issues_of(GraphIssueKind.MISSING_PROVENANCE)
    assert len(issue.edge_ids) == 2


def test_observed_origin_needs_observed_data():
    ok = _claim("a", "b", origin=EdgeOrigin.OBSERVED, provenance=(ProvenanceSource(
        kind=ProvenanceKind.OBSERVED_DATA, reference="dataset:x", title="x", claim="co-movement measured",
        verification=Verification.VERIFIED, verified_on=date(2026, 9, 25)),))
    graph = assemble_graph([ok, _claim("b", "c", origin=EdgeOrigin.OBSERVED)], catalog=_states("a", "b", "c"))
    support = {(e.source, e.target): e.support for e in graph.edges}
    assert support == {("a", "b"): SupportStatus.SUPPORTED, ("b", "c"): SupportStatus.UNRESOLVED}


def _proposal(*links: dict) -> TransmissionProposal:
    return TransmissionProposal.model_validate({
        "proposer": "claude-test", "prompt_fingerprint": "sha256:abc", "links": list(links),
    })


def test_a_model_proposal_stays_proposed_even_when_it_cites_a_source():
    proposal = _proposal({
        "source": "HBM demand", "target": "advanced packaging equipment demand", "polarity": "POSITIVE",
        "rationale": "HBM stacking needs TSV and bonding tools",
        "citations": [{"title": "Some report", "reference": "https://example.org/r", "claim": "HBM needs TSV"}],
    })
    claims = claims_from_proposal(proposal)
    assert claims[0].origin is EdgeOrigin.MODEL_PROPOSED
    kinds = {p.kind: p for p in claims[0].provenance}
    assert kinds[ProvenanceKind.EXTERNAL_DOCUMENT].verification is Verification.UNVERIFIED
    graph = assemble_graph(claims, catalog=STATE_CATALOG)
    assert graph.edges[0].support is SupportStatus.PROPOSED
    assert graph.issues_of(GraphIssueKind.UNVERIFIED_CITATION)
    assert graph.issues_of(GraphIssueKind.UNKNOWN_LAG) and graph.issues_of(GraphIssueKind.UNKNOWN_CONFIDENCE)


@pytest.mark.parametrize("smuggled", [
    {"support": "SUPPORTED"}, {"status": "SUPPORTED"}, {"origin": "MANUALLY_SEEDED"}, {"verified": True},
])
def test_a_model_output_cannot_assert_its_own_status_or_origin(smuggled):
    link = {"source": "a", "target": "b", "polarity": "POSITIVE", "rationale": "because", **smuggled}
    with pytest.raises(ValidationError):
        _proposal(link)


def test_a_model_citation_cannot_claim_verification():
    link = {"source": "a", "target": "b", "polarity": "POSITIVE", "rationale": "because",
            "citations": [{"title": "t", "reference": "u", "verification": "VERIFIED"}]}
    with pytest.raises(ValidationError):
        _proposal(link)


def test_only_an_independently_verified_source_promotes_a_model_link():
    proposed = claims_from_proposal(_proposal({"source": "a", "target": "b", "polarity": "POSITIVE",
                                               "rationale": "because"}))[0]
    promoted = proposed.model_copy(update={"provenance": (*proposed.provenance, _verified_doc())})
    assert assemble_graph([proposed], catalog=_states("a", "b")).edges[0].support is SupportStatus.PROPOSED
    assert assemble_graph([promoted], catalog=_states("a", "b")).edges[0].support is SupportStatus.SUPPORTED


def test_proposals_join_the_event_graph_with_their_own_provenance():
    proposal = _proposal(
        {"source": "hbm_demand", "target": "advanced packaging equipment demand", "polarity": "POSITIVE",
         "lag": "QUARTERS", "confidence": "MEDIUM", "rationale": "HBM stacking needs TSV and bonding tools"},
        {"source": "cocoa supply", "target": "chocolate prices", "polarity": "NEGATIVE", "rationale": "unrelated"},
    )
    mg = _graph(AI_CAPEX, extra_claims=claims_from_proposal(proposal))
    new = mg.implication("advanced_packaging_equipment_demand")
    assert new.movement is ImpliedMovement.UP and new.kind is StateKind.UNSPECIFIED
    assert new.paths[0].weakest_support is SupportStatus.PROPOSED
    assert "MODEL_PROPOSAL:claude-test#sha256:abc" in mg.generated_by
    assert not mg.graph.has_state("chocolate_prices")
    assert any(i.kind is GraphIssueKind.DISCONNECTED_CLAIMS for i in mg.stage_issues)
    # seeded claims are untouched by a proposal
    assert _graph(AI_CAPEX).graph.edges == tuple(e for e in mg.graph.edges if e.target != new.state_id)


def test_unknown_lag_and_confidence_propagate_as_the_weakest_link():
    library = TransmissionLibrary(
        states=_states("ai_infrastructure_investment", "b", "c"),
        claims=(_claim("ai_infrastructure_investment", "b"),
                _claim("b", "c", lag=TransmissionLag.UNKNOWN, confidence=LinkConfidence.UNKNOWN)),
        anchors=DEFAULT_LIBRARY.anchors,
    )
    path = _graph(AI_CAPEX, library=library).implication("c").paths[0]
    assert path.slowest_lag is TransmissionLag.UNKNOWN and path.weakest_confidence is LinkConfidence.UNKNOWN


def test_an_unsigned_link_makes_downstream_direction_indeterminate():
    library = TransmissionLibrary(
        states=_states("ai_infrastructure_investment", "b", "c"),
        claims=(_claim("ai_infrastructure_investment", "b", Polarity.AMBIGUOUS), _claim("b", "c")),
        anchors=DEFAULT_LIBRARY.anchors,
    )
    mg = _graph(AI_CAPEX, library=library)
    assert mg.implication("c").movement is ImpliedMovement.INDETERMINATE
    assert mg.graph.issues_of(GraphIssueKind.UNSIGNED_LINK)


# ---------------------------------------------------------------------------
# cycles
# ---------------------------------------------------------------------------


def test_seeded_balancing_loops_are_detected_and_classified():
    ai = _graph(AI_CAPEX).graph.loops
    assert [(loop.state_ids, loop.loop_type) for loop in ai] == [
        (("advanced_foundry_utilization", "foundry_capacity_expansion"), LoopType.BALANCING),
    ]
    oil = _graph(OPEC_CUT).graph.loops
    assert [(loop.state_ids, loop.loop_type) for loop in oil] == [
        (("crude_oil_supply", "crude_oil_price", "shale_drilling_activity"), LoopType.BALANCING),
    ]


def test_loop_types_and_self_loops():
    reinforcing = assemble_graph([_claim("a", "b"), _claim("b", "a")], catalog=_states("a", "b"))
    assert reinforcing.loops[0].loop_type is LoopType.REINFORCING
    unsigned = assemble_graph([_claim("a", "b"), _claim("b", "a", Polarity.UNKNOWN)], catalog=_states("a", "b"))
    assert unsigned.loops[0].loop_type is LoopType.INDETERMINATE
    parallel = assemble_graph([_claim("a", "b"), _claim("a", "b", Polarity.NEGATIVE), _claim("b", "a")],
                              catalog=_states("a", "b"))
    assert [loop.loop_type for loop in parallel.loops] == [LoopType.INDETERMINATE]
    selfie = assemble_graph([_claim("a", "A")], catalog=_states("a"))
    assert not selfie.edges and len(selfie.rejected_claims) == 1
    assert selfie.issues_of(GraphIssueKind.SELF_LOOP_REJECTED)


def test_propagation_terminates_on_cycles_and_walks_only_simple_paths():
    ring = [_claim(a, b) for a, b in (("ai_infrastructure_investment", "b"), ("b", "c"), ("c", "d"), ("d", "b"),
                                      ("c", "ai_infrastructure_investment"))]
    library = TransmissionLibrary(states=_states("ai_infrastructure_investment", "b", "c", "d"), claims=tuple(ring),
                                  anchors=DEFAULT_LIBRARY.anchors)
    mg = _graph(AI_CAPEX, library=library)
    for imp in mg.implications:
        for path in imp.paths:
            assert len(set(path.state_ids)) == len(path.state_ids)
    assert mg.implication("ai_infrastructure_investment").is_anchor  # not re-reached through the loop
    assert len(mg.graph.loops) == 2


def test_a_balancing_path_through_new_capacity_is_a_path_not_a_loop_artifact():
    util = _graph(AI_CAPEX).implication("advanced_foundry_utilization")
    assert util.movement is ImpliedMovement.MIXED
    down = next(p for p in util.paths if p.movement is ImpliedMovement.DOWN)
    assert "foundry_capacity_expansion" in down.state_ids and down.slowest_lag is TransmissionLag.YEARS


# ---------------------------------------------------------------------------
# serialization / determinism
# ---------------------------------------------------------------------------


def test_graph_round_trips_through_json_and_fingerprints_deterministically():
    mg = _graph(AI_CAPEX)
    again = EconomicMechanismGraph.model_validate_json(mg.model_dump_json())
    assert again == mg and again.fingerprint() == mg.fingerprint()
    assert _graph(AI_CAPEX).fingerprint() == mg.fingerprint()
    assert mg.graph.fingerprint().startswith("txgraph1:")
    payload = json.loads(mg.model_dump_json())
    assert payload["plane"] == "HYPOTHESIS" and payload["schema_version"] == "economic-mechanism-graph/1"


def test_the_plane_cannot_be_relabelled_evidence():
    payload = json.loads(_graph(AI_CAPEX).model_dump_json())
    payload["plane"] = "EVIDENCE"
    with pytest.raises(ValidationError):
        EconomicMechanismGraph.model_validate(payload)


def test_a_raw_json_model_proposal_parses_into_the_closed_schema():
    raw = json.dumps({"proposer": "claude-test", "prompt_fingerprint": "sha256:1", "links": [
        {"source": "electricity demand", "target": "grid investment", "polarity": "POSITIVE",
         "channel": "INVESTMENT_RESPONSE", "rationale": "load growth needs interconnection"}]})
    proposal = TransmissionProposal.model_validate_json(raw)
    edge = assemble_graph(claims_from_proposal(proposal), catalog=STATE_CATALOG).edges[0]
    assert (edge.source, edge.target) == ("electricity_demand", "grid_investment")


# ---------------------------------------------------------------------------
# Phase A integration + mandate independence
# ---------------------------------------------------------------------------


def test_the_three_phase_a_examples_each_get_an_anchored_graph():
    anchors = {text: [(a.state_id, a.movement) for a in _graph(text).anchors]
               for text in (AI_CAPEX, OPEC_CUT, RATE_SURPRISE)}
    assert anchors == {
        AI_CAPEX: [("ai_infrastructure_investment", Movement.UP)],
        OPEC_CUT: [("crude_oil_supply", Movement.DOWN)],
        RATE_SURPRISE: [("policy_rate_path", Movement.UP)],
    }
    oil = _graph(OPEC_CUT)
    assert oil.implication("crude_oil_price").movement is ImpliedMovement.UP
    assert oil.implication("refined_product_prices").movement is ImpliedMovement.UP


def test_a_paraphrase_triages_to_the_same_anchor_and_graph():
    paraphrase = _graph("Microsoft raised capex again to build more AI data centers")
    assert paraphrase.anchors[0].movement is Movement.UP
    assert {e.edge_id for e in paraphrase.graph.edges} == {e.edge_id for e in _graph(AI_CAPEX).graph.edges}


def test_no_direction_cue_gives_a_relative_graph():
    mg = _graph("Analysts debate the AI data center buildout and semiconductor supply")
    assert mg.anchors[0].movement is Movement.UNKNOWN
    downstream = [i for i in mg.implications if not i.is_anchor]
    assert all(i.movement is ImpliedMovement.UNANCHORED for i in downstream)
    assert mg.implication("hbm_demand").relative_sign is Polarity.POSITIVE
    assert mg.implication("ai_spender_free_cash_flow").relative_sign is Polarity.AMBIGUOUS


def test_an_unseeded_channel_is_reported_not_rendered_as_an_empty_finding():
    mg = _graph("Missile strikes and a naval blockade raise geopolitical escalation fears")
    assert mg.is_empty and mg.unseeded_channels
    assert mg.unseeded_channels[0].channel is EconomicChannel.GEOPOLITICAL_RISK
    assert "no economic transmission is seeded yet" in mechanism_graph_view.summary_line(mg)
    unrelated = _graph("A local bakery won a regional award")
    assert unrelated.is_empty and not unrelated.unseeded_channels
    assert mechanism_graph_view.summary_line(unrelated) is None


def test_an_anchor_with_no_seeded_links_says_so():
    library = TransmissionLibrary(states=DEFAULT_LIBRARY.states, claims=(), anchors=DEFAULT_LIBRARY.anchors)
    mg = _graph(AI_CAPEX, library=library)
    assert not mg.is_empty and [s.state_id for s in mg.graph.states] == ["ai_infrastructure_investment"]
    assert "no downstream links are seeded yet" in mechanism_graph_view.summary_line(mg)


def test_a_cached_official_news_item_flows_through_to_the_graph():
    item = MarketNewsItem(
        news_id="N-1", headline="EIA: crude inventory draw larger than expected", source_name="EIA",
        source_type=NewsSourceType.OFFICIAL, source_url="https://eia.gov/n-1", published_at=AT - timedelta(hours=2),
        retrieved_at=AT, related_products=mapping.products_for_category(NewsCategory.PETROLEUM),
        related_asset_classes=mapping.asset_classes_for_category(NewsCategory.PETROLEUM),
        category=NewsCategory.PETROLEUM, mapping_reason=mapping.mapping_reason_for_category(NewsCategory.PETROLEUM),
    )
    mg = build_economic_mechanism_graph(scan_initial_impact(item, EQUITY_FUTURES, capabilities=CAPS))
    assert mg.event_id == "N-1"
    assert [(a.state_id, a.movement) for a in mg.anchors] == [("crude_oil_supply", Movement.DOWN)]


@pytest.mark.parametrize("text", [AI_CAPEX, OPEC_CUT, RATE_SURPRISE])
def test_the_mandate_never_shapes_the_economic_graph(text):
    mandates = [
        ResearchMandate(allowed_domains=(MandateDomain.EQUITY,)),
        ResearchMandate(allowed_domains=(MandateDomain.FUTURES,)),
        ResearchMandate(allowed_domains=tuple(MandateDomain)),
        ResearchMandate(allowed_domains=(MandateDomain.ETF,), shorting_allowed=True),
    ]
    prints = {_graph(text, m).fingerprint() for m in mandates}
    assert len(prints) == 1
    scan = _scan(text)
    assert build_economic_mechanism_graph(scan.model_copy(update={"assessments": ()})).fingerprint() in prints


def test_an_equity_only_mandate_still_sees_non_tradable_economic_states():
    equity_only = ResearchMandate(allowed_domains=(MandateDomain.EQUITY,))
    ai = _graph(AI_CAPEX, equity_only)
    for sid in ("electricity_demand", "copper_demand", "gas_power_generation_demand", "grid_investment"):
        assert ai.graph.has_state(sid), sid
    oil = _graph(OPEC_CUT, equity_only)
    for sid in ("headline_inflation", "policy_rate_path", "treasury_yields"):
        assert oil.graph.has_state(sid), sid


def test_the_hop_limit_is_explicit():
    mg = _graph(OPEC_CUT, max_hops=2)
    assert not mg.graph.has_state("treasury_yields")
    assert any(i.kind is GraphIssueKind.HOP_LIMIT_REACHED for i in mg.stage_issues)
    with pytest.raises(ValueError):
        _graph(OPEC_CUT, max_hops=0)


# ---------------------------------------------------------------------------
# library integrity
# ---------------------------------------------------------------------------


def test_every_seeded_claim_and_anchor_is_well_formed():
    declared = {s.state_id for s in STATE_CATALOG}
    for claim in SEEDED_CLAIMS:
        assert claim.origin is EdgeOrigin.MANUALLY_SEEDED
        assert {claim.source, claim.target} <= declared
        assert claim.has_required_provenance
        for p in claim.provenance:
            if p.counts_as_support and p.kind is ProvenanceKind.EXTERNAL_DOCUMENT:
                assert p.reference.startswith("https://") and p.published and p.claim and p.verified_on
    assert {a.channel for a in ANCHOR_RULES} == {
        EconomicChannel.TECHNOLOGY_INVESTMENT, EconomicChannel.CRUDE_OIL_SUPPLY, EconomicChannel.MONETARY_POLICY,
    }


def test_library_validation_refuses_anchor_rules_that_drift_from_phase_a_shifts():
    bad = AnchorRule(EconomicChannel.MONETARY_POLICY, "x", "policy_rate_path", (("TIGHTENING", Movement.UP),), "x")
    with pytest.raises(ValueError, match="must map exactly the shifts"):
        _validate(TransmissionLibrary(states=STATE_CATALOG, claims=SEEDED_CLAIMS, anchors=(bad,)))
    stray = _claim("policy_rate_path", "unknown_state")
    with pytest.raises(ValueError, match="undeclared state"):
        _validate(TransmissionLibrary(states=STATE_CATALOG, claims=(stray,), anchors=()))


# ---------------------------------------------------------------------------
# boundaries: no verdict, no registry, plane separation
# ---------------------------------------------------------------------------


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


@pytest.mark.parametrize("path", PHASE_B_SOURCES, ids=lambda p: p.name)
def test_the_hypothesis_plane_never_imports_the_evidence_plane_or_an_llm(path):
    forbidden = (
        "alpha_agent.alpha_graph", "alpha_agent.alpha_memory", "alpha_agent.registry", "alpha_agent.knowledge",
        "alpha_agent.translation", "alpha_agent.validation", "alpha_agent.agents", "anthropic",
    )
    for imported in _imports(path):
        assert not any(imported == f or imported.startswith(f + ".") for f in forbidden), (path.name, imported)
    source = path.read_text(encoding="utf-8")
    for token in ("ExperimentRegistry", "insert_", "INSERT", "UPDATE ", "requests.", "urllib", "httpx", "open("):
        assert token not in source, (path.name, token)


def test_the_evidence_plane_and_market_intel_never_import_the_hypothesis_plane():
    root = REPO_ROOT / "python" / "alpha_agent"
    for pkg in ("alpha_graph", "market_intel", "translation", "core"):
        for path in (root / pkg).rglob("*.py"):
            assert not any(i.startswith("alpha_agent.news_alpha") for i in _imports(path)), path


def test_no_schema_carries_a_return_probability_trade_weight_or_verdict_field():
    from alpha_agent.news_alpha import mechanism_graph, transmission

    forbidden = {
        "expected_return", "return", "probability", "probability_of_profit", "p_value", "sharpe", "score",
        "weight", "portfolio_weight", "position_size", "buy", "sell", "action", "signal", "verdict",
        "scientific_verdict", "experiment_id", "experiment_identity", "alpha", "forecast", "target_price",
    }
    models = [
        transmission.EconomicState, transmission.TransmissionClaim, transmission.TransmissionEdge,
        transmission.TransmissionGraph, transmission.ProvenanceSource, transmission.GraphIssue,
        transmission.FeedbackLoop, transmission.ProposedLink, transmission.ProposedCitation,
        transmission.TransmissionProposal, mechanism_graph.EconomicMechanismGraph, mechanism_graph.GraphAnchor,
        mechanism_graph.StateImplication, mechanism_graph.TransmissionPath, mechanism_graph.UnseededChannel,
    ]
    for model in models:
        assert not (set(model.model_fields) & forbidden), model.__name__
    for text in (AI_CAPEX, OPEC_CUT, RATE_SURPRISE):
        rendered = json.dumps(json.loads(_graph(text).model_dump_json())).upper()
        for word in ('"BUY"', '"SELL"', '"PASS"', '"FAIL"', "STRONG BUY", "GUARANTEED"):
            assert word not in rendered


def test_building_graphs_never_touches_the_registry():
    if not REGISTRY_PATH.exists():
        pytest.skip("Phase 14 registry sqlite not present in this checkout")
    before = hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
    for text in (AI_CAPEX, AI_CAPEX_CUT, OPEC_CUT, RATE_SURPRISE):
        _graph(text)
    assert hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest() == before


def test_the_map_never_uses_verdict_colors_and_draws_every_state_and_link():
    mg = _graph(AI_CAPEX)
    dot = mechanism_graph_view.graph_dot(mg)
    assert dot.startswith("digraph mechanism {") and dot.rstrip().endswith("}")
    assert palette.GREEN not in dot and palette.RED not in dot
    assert dot.count(" -> ") == len(mg.graph.edges)
    for state in mg.graph.states:
        assert f'"{state.state_id}" [' in dot
    assert dot.count('style="dashed"') == mg.graph.support_counts()[SupportStatus.PROPOSED]
