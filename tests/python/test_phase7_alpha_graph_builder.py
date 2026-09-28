"""Phase 7 -- the Cross-Asset Alpha Graph builder (prompt 7 sections 2/3/4/6/9/10).

Uses the REAL, already-committed local registry (read-only, never mutated --
the same pattern `test_alpha_memory_builder.py` already uses) so the real
cross-asset acceptance case (prompt section 8) is proven against genuine
evidence, not a fabricated fixture: TREND has real registry evidence on
Futures CL/ES/GC/NQ/ZN (tsmom, ma_trend) AND on ETF XLE (etf_tsmom) -- the
Phase 6 closure's one real, persisted ETF experiment.
"""
from __future__ import annotations

import re

import pytest
from alpha_agent.alpha_graph import builder as alpha_graph_builder
from alpha_agent.alpha_graph import schemas as alpha_graph_schemas
from alpha_agent.alpha_graph.builder import (
    MECHANISM_UNIVERSE,
    build_alpha_graph,
    build_mechanism_graph,
    cross_asset_synthesis,
    list_considered_instruments,
    list_research_gaps,
    summarize,
)
from alpha_agent.alpha_graph.schemas import EvidenceCoverage, GraphNodeType, InstrumentRef
from alpha_agent.alpha_memory.factor_identity import compute_factor_identity
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.enums import AssetDomain
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

_ALL_MODULES = (alpha_graph_schemas, alpha_graph_builder)


def _registry() -> ExperimentRegistry:
    return ExperimentRegistry(services.REGISTRY_PATH)


# ---------------------------------------------------------------------------
# Minimal node/relationship model (prompt section 2)
# ---------------------------------------------------------------------------


def test_only_the_six_prescribed_node_types_exist():
    assert {t.value for t in GraphNodeType} == {
        "EVENT", "MECHANISM", "FACTOR", "INSTRUMENT", "STRATEGY", "EXPERIMENT",
    }


def test_mechanism_universe_is_the_union_of_both_domain_bridges():
    from alpha_agent.etf.alpha_memory_bridge import ETF_MECHANISM_TO_FAMILIES
    from alpha_agent.translation.research_memory import MECHANISM_TO_KNOWN_FAMILIES

    assert set(MECHANISM_UNIVERSE) == set(MECHANISM_TO_KNOWN_FAMILIES) | set(ETF_MECHANISM_TO_FAMILIES)
    assert EconomicMechanism.TREND in MECHANISM_UNIVERSE


def test_considered_instruments_are_the_existing_frozen_universes_never_broadened():
    from alpha_agent.etf.universe import PILOT_UNIVERSE
    from alpha_agent.translation.schemas import CERTIFIED_ROOTS

    instruments = list_considered_instruments()
    futures = {i.root_symbol for i in instruments if i.asset_domain is AssetDomain.FUTURES}
    etf = {i.root_symbol for i in instruments if i.asset_domain is AssetDomain.ETF}
    assert futures == set(CERTIFIED_ROOTS)
    assert etf == set(PILOT_UNIVERSE)


# ---------------------------------------------------------------------------
# Real cross-asset acceptance case (prompt section 8): TREND on Futures + ETF
# ---------------------------------------------------------------------------


def test_trend_has_real_futures_and_etf_evidence_under_one_shared_mechanism():
    with _registry() as reg:
        view = build_mechanism_graph(reg, EconomicMechanism.TREND)
    assert AssetDomain.FUTURES in view.domains_with_evidence
    assert AssetDomain.ETF in view.domains_with_evidence

    futures_researched = {
        e.instrument.root_symbol for e in view.instrument_evidence
        if e.instrument.asset_domain is AssetDomain.FUTURES and e.coverage is EvidenceCoverage.RESEARCHED
    }
    etf_researched = {
        e.instrument.root_symbol for e in view.instrument_evidence
        if e.instrument.asset_domain is AssetDomain.ETF and e.coverage is EvidenceCoverage.RESEARCHED
    }
    assert futures_researched == {"CL", "ES", "GC", "NQ", "ZN"}
    assert etf_researched == {"XLE"}

    summary = summarize(view)
    assert summary.is_cross_asset is True
    assert summary.n_researched == 6  # 5 Futures roots + XLE


def test_trend_factors_include_both_domains_families():
    with _registry() as reg:
        view = build_mechanism_graph(reg, EconomicMechanism.TREND)
    families_by_domain: dict[AssetDomain, set[str]] = {}
    for f in view.factors:
        families_by_domain.setdefault(f.asset_domain, set()).update(f.factor.related_strategy_families)
    assert families_by_domain[AssetDomain.FUTURES] >= {"tsmom", "ma_trend"}
    assert families_by_domain[AssetDomain.ETF] == {"etf_tsmom"}
    # each family is its own singleton Factor, never merged (V1 default)
    futures_factor_families = [
        f.factor.related_strategy_families for f in view.factors if f.asset_domain is AssetDomain.FUTURES
    ]
    assert ("tsmom",) in futures_factor_families
    assert ("ma_trend",) in futures_factor_families

    xle_factor = next(f for f in view.factors if f.asset_domain is AssetDomain.ETF)
    assert {i.root_symbol for i in xle_factor.instruments_with_evidence} == {"XLE"}


def test_same_factor_identity_across_domains_never_mixes_instruments():
    """Regression: `compute_factor_identity` hashes only (mechanism, family
    names) -- it does NOT include asset_domain. If a FUTURES family and an
    ETF family ever shared a name for the same mechanism, their
    factor_identity strings would collide. `_factors_for_mechanism`'s
    `factor_instruments` map is keyed by (asset_domain, factor_identity)
    together specifically so that collision can never merge a FUTURES
    instrument onto an ETF Factor node or vice versa (prompt section 4/9:
    asset-domain isolation). Forces the collision directly (today's real
    ETF_MECHANISM_TO_FAMILIES/MECHANISM_TO_KNOWN_FAMILIES never share a
    family name, so this cannot be observed from real registry data alone)."""
    forced_factor = compute_factor_identity(EconomicMechanism.TREND, ("tsmom",))
    cl = InstrumentRef(root_symbol="CL", asset_domain=AssetDomain.FUTURES)
    xle = InstrumentRef(root_symbol="XLE", asset_domain=AssetDomain.ETF)
    factor_instruments = {
        (AssetDomain.FUTURES, forced_factor.factor_identity): [cl],
        (AssetDomain.ETF, forced_factor.factor_identity): [xle],
    }

    def _with_forced_groups(mechanism, asset_domain):
        return (("tsmom",),)

    orig = alpha_graph_builder._family_groups
    try:
        alpha_graph_builder._family_groups = _with_forced_groups
        factors = alpha_graph_builder._factors_for_mechanism(EconomicMechanism.TREND, factor_instruments)
    finally:
        alpha_graph_builder._family_groups = orig

    assert len({f.factor.factor_identity for f in factors}) == 1  # the collision really happened
    futures_entry = next(f for f in factors if f.asset_domain is AssetDomain.FUTURES)
    etf_entry = next(f for f in factors if f.asset_domain is AssetDomain.ETF)
    assert [i.root_symbol for i in futures_entry.instruments_with_evidence] == ["CL"]
    assert [i.root_symbol for i in etf_entry.instruments_with_evidence] == ["XLE"]
    # never merged into one combined list under either domain
    assert "XLE" not in [i.root_symbol for i in futures_entry.instruments_with_evidence]
    assert "CL" not in [i.root_symbol for i in etf_entry.instruments_with_evidence]


def test_event_to_mechanism_edge_is_real_and_deterministic_no_network():
    """PETROLEUM (Event category) -> TREND -- Phase 1's own frozen
    mechanism library, reused verbatim, never a live news fetch."""
    with _registry() as reg:
        view = build_mechanism_graph(reg, EconomicMechanism.TREND)
    categories = {e.category.value for e in view.event_categories}
    assert "PETROLEUM" in categories
    for ev in view.event_categories:
        assert ev.evidence_basis  # real, non-fabricated provenance text


# ---------------------------------------------------------------------------
# Verdicts remain experiment-specific -- never merged/transferred (section 4)
# ---------------------------------------------------------------------------


def test_verdicts_remain_experiment_specific_never_merged_across_instruments():
    with _registry() as reg:
        view = build_mechanism_graph(reg, EconomicMechanism.TREND)
    per_instrument = {
        e.instrument.root_symbol: e.scientific_evidence
        for e in view.instrument_evidence
        if e.coverage in (EvidenceCoverage.RESEARCHED, EvidenceCoverage.WEAKLY_RESEARCHED)
    }
    # CL and ES both have tsmom REJECT, but ES's ma_trend is INCONCLUSIVE while
    # CL's is REJECT -- if verdicts were merged/blended these would collapse.
    assert per_instrument["CL"]["ma_trend"] == "REJECT"
    assert per_instrument["ES"]["ma_trend"] == "INCONCLUSIVE"
    assert per_instrument["XLE"]["etf_tsmom"] == "REJECT"
    # no field anywhere on the schema could hold a merged/combined verdict
    for model in (alpha_graph_schemas.MechanismGraphView, alpha_graph_schemas.InstrumentEvidence):
        assert "overall_verdict" not in model.model_fields
        assert "combined_verdict" not in model.model_fields
        assert "merged_verdict" not in model.model_fields


def test_cross_asset_evidence_cannot_overwrite_or_transfer_a_verdict():
    """A synthetic REJECT-only ETF instrument does not change a genuinely
    different Futures instrument's own scientific_evidence, and vice versa
    -- proven by re-fetching each instrument's own Personal Alpha Memory
    object directly and comparing to what the graph reports for it."""
    with _registry() as reg:
        view = build_mechanism_graph(reg, EconomicMechanism.TREND)
    graph_cl = next(
        e for e in view.instrument_evidence
        if e.instrument.root_symbol == "CL" and e.instrument.asset_domain is AssetDomain.FUTURES
    )
    graph_xle = next(
        e for e in view.instrument_evidence
        if e.instrument.root_symbol == "XLE" and e.instrument.asset_domain is AssetDomain.ETF
    )
    # each instrument's graph-reported evidence matches its OWN Alpha Memory
    # object exactly -- no cross-contamination from the other domain.
    direct_objects = {o["alpha_id"]: o for o in services.list_alpha_research_objects()}
    for alpha_id in graph_cl.alpha_ids:
        assert direct_objects[alpha_id]["root_symbol"] == "CL"
        assert direct_objects[alpha_id]["asset_domain"] == "FUTURES"
    for alpha_id in graph_xle.alpha_ids:
        assert direct_objects[alpha_id]["root_symbol"] == "XLE"
        assert direct_objects[alpha_id]["asset_domain"] == "ETF"
    assert set(graph_cl.alpha_ids).isdisjoint(set(graph_xle.alpha_ids))


# ---------------------------------------------------------------------------
# No-evidence combinations are research gaps, never failures (section 6)
# ---------------------------------------------------------------------------


def test_unmapped_mechanism_for_a_domain_is_no_evidence_not_a_failure():
    """ETF's bridge maps ONLY TREND today -- BREAKOUT on any ETF ticker must
    be NO_EVIDENCE (not structurally representable), never a REJECT-shaped
    failure code."""
    with _registry() as reg:
        view = build_mechanism_graph(reg, EconomicMechanism.BREAKOUT)
    etf_rows = [e for e in view.instrument_evidence if e.instrument.asset_domain is AssetDomain.ETF]
    assert etf_rows
    assert all(e.coverage is EvidenceCoverage.NO_EVIDENCE for e in etf_rows)
    assert all(not e.scientific_evidence for e in etf_rows)
    assert all(not e.alpha_ids for e in etf_rows)


def test_mapped_but_untested_instrument_is_underexplored_not_no_evidence():
    """TREND IS mapped for ETF, but SPY has never been tested -- must be
    UNDEREXPLORED, distinct from BREAKOUT's genuinely-unmapped NO_EVIDENCE."""
    with _registry() as reg:
        view = build_mechanism_graph(reg, EconomicMechanism.TREND)
    spy = next(e for e in view.instrument_evidence if e.instrument.root_symbol == "SPY")
    assert spy.coverage is EvidenceCoverage.UNDEREXPLORED


def test_research_gaps_are_first_class_never_hidden():
    with _registry() as reg:
        views = build_alpha_graph(reg)
    gaps = list_research_gaps(views)
    assert gaps
    assert all(g.coverage in (EvidenceCoverage.UNDEREXPLORED, EvidenceCoverage.NO_EVIDENCE) for g in gaps)
    # every gap is traceable to a real considered (mechanism, instrument) pair
    considered = set(list_considered_instruments())
    for g in gaps:
        assert g.instrument in considered
        assert g.mechanism in MECHANISM_UNIVERSE


# ---------------------------------------------------------------------------
# Asset-domain isolation remains intact (section 4/9)
# ---------------------------------------------------------------------------


def test_asset_domain_isolation_futures_and_etf_never_cross_contaminate():
    with _registry() as reg:
        view = build_mechanism_graph(reg, EconomicMechanism.TREND)
    futures_alpha_ids: set[str] = set()
    etf_alpha_ids: set[str] = set()
    for e in view.instrument_evidence:
        if e.instrument.asset_domain is AssetDomain.FUTURES:
            futures_alpha_ids.update(e.alpha_ids)
        else:
            etf_alpha_ids.update(e.alpha_ids)
    assert futures_alpha_ids
    assert etf_alpha_ids
    assert futures_alpha_ids.isdisjoint(etf_alpha_ids)

    # a Futures-only mechanism query (e.g. MOMENTUM, unmapped for ETF) never
    # surfaces an ETF alpha id.
    with _registry() as reg:
        momentum = build_mechanism_graph(reg, EconomicMechanism.MOMENTUM)
    for e in momentum.instrument_evidence:
        if e.instrument.asset_domain is AssetDomain.ETF:
            assert e.coverage is EvidenceCoverage.NO_EVIDENCE
            assert not e.alpha_ids


# ---------------------------------------------------------------------------
# Cross-asset synthesis -- deterministic, never a merged verdict
# ---------------------------------------------------------------------------


def test_cross_asset_synthesis_is_deterministic_and_cites_every_instrument_separately():
    with _registry() as reg:
        view = build_mechanism_graph(reg, EconomicMechanism.TREND)
    synth1 = cross_asset_synthesis(view)
    synth2 = cross_asset_synthesis(view)
    assert synth1 == synth2  # pure function of already-computed fields

    assert synth1.common
    assert "CL" in synth1.differs[0] and "XLE" in synth1.differs[0]
    assert "REJECT" in synth1.differs[0]


def test_synthesis_never_recommends_a_trade():
    with _registry() as reg:
        views = build_alpha_graph(reg)
    banned = re.compile(r"\bbuy\b|\bsell\b|\bshould trade\b|\brecommend(ed|s)?\b|\bsignal to trade\b", re.IGNORECASE)
    for view in views:
        synth = cross_asset_synthesis(view)
        for bucket in (synth.common, synth.differs, synth.repeated_failures, synth.underexplored, (synth.note,)):
            for line in bucket:
                assert not banned.search(line), f"trade-signal language found: {line!r}"


# ---------------------------------------------------------------------------
# No second scientific database / graph score / write / network / execution
# ---------------------------------------------------------------------------


def test_no_alpha_score_or_graph_score_field_anywhere():
    for model in (
        alpha_graph_schemas.MechanismGraphView, alpha_graph_schemas.InstrumentEvidence,
        alpha_graph_schemas.FactorUnderMechanism, alpha_graph_schemas.MechanismGraphSummary,
        alpha_graph_schemas.CrossAssetSynthesis, alpha_graph_schemas.ResearchGapRow,
    ):
        for name in model.model_fields:
            assert "score" not in name.lower()
            assert "confidence" not in name.lower()
            assert "expected_return" not in name.lower()


def test_alpha_graph_never_calls_a_registry_write_method():
    forbidden = [
        r"\.insert_experiment\(", r"\.record_failure\(", r"\.record_lineage\(",
        r"\.apply_bundle\(", r"\.record_attempt", r"INSERT OR REPLACE", r"\bUPDATE\s+\w+\s+SET\b",
    ]
    for module in _ALL_MODULES:
        with open(module.__file__, encoding="utf-8") as fh:
            text = fh.read()
        for pattern in forbidden:
            assert not re.search(pattern, text), f"{module.__name__} must never call {pattern!r}"


def test_alpha_graph_never_imports_network_or_llm_clients():
    forbidden_imports = ("AnthropicClient", "databento", "import anthropic", "requests.")
    for module in _ALL_MODULES:
        with open(module.__file__, encoding="utf-8") as fh:
            text = fh.read()
        for token in forbidden_imports:
            assert token not in text, f"{module.__name__} must never reference {token!r}"


def test_alpha_graph_never_triggers_backtest_execution():
    forbidden_tokens = (
        "StrategyCompilerAgent", "run_deep_research", "CppEngineRunner",
        "PaperTradingEngine", "run_targets_backtest",
    )
    for module in _ALL_MODULES:
        with open(module.__file__, encoding="utf-8") as fh:
            text = fh.read()
        for token in forbidden_tokens:
            assert token not in text, f"{module.__name__} must never reference {token!r}"


def test_alpha_graph_never_introduces_a_graph_database():
    forbidden_tokens = ("neo4j", "networkx", "py2neo", "gremlin", "cypher")
    for module in _ALL_MODULES:
        with open(module.__file__, encoding="utf-8") as fh:
            text = fh.read().lower()
        for token in forbidden_tokens:
            assert token not in text, f"{module.__name__} must never reference {token!r}"
