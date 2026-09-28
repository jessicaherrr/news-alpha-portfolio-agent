"""Phase 7 -- the small, deterministic Agent retrieval layer (prompt section
5/10: "Agent retrieval returns related research without generating trade
signals").

Uses the REAL local registry, same pattern as
`test_phase7_alpha_graph_builder.py`.
"""
from __future__ import annotations

import re

import pytest
from alpha_agent.alpha_graph import retrieval as alpha_graph_retrieval
from alpha_agent.alpha_graph.retrieval import (
    analogous_research,
    cross_asset_confirmation,
    repeated_failure_patterns,
    underexplored_combinations,
)
from alpha_agent.alpha_graph.schemas import EvidenceCoverage
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.enums import AssetDomain
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

_BANNED = re.compile(
    r"\bbuy\b|\bsell\b|\bshould trade\b|\brecommend(ed|s)?\b|\bsignal to trade\b|\bgo long\b|\bgo short\b",
    re.IGNORECASE,
)


def _registry() -> ExperimentRegistry:
    return ExperimentRegistry(services.REGISTRY_PATH)


def test_analogous_research_finds_xle_from_a_futures_root_and_vice_versa():
    with _registry() as reg:
        from_cl = analogous_research(reg, mechanism=EconomicMechanism.TREND, root_symbol="CL")
        from_xle = analogous_research(reg, mechanism=EconomicMechanism.TREND, root_symbol="XLE")
    assert any(e.instrument.root_symbol == "XLE" for e in from_cl)
    assert any(e.instrument.asset_domain is AssetDomain.FUTURES for e in from_xle)
    # the query instrument itself is always excluded
    assert all(e.instrument.root_symbol != "CL" for e in from_cl)
    assert all(e.instrument.root_symbol != "XLE" for e in from_xle)


def test_analogous_research_is_empty_not_an_error_on_a_fresh_registry(tmp_path):
    with ExperimentRegistry(tmp_path / "empty.sqlite") as reg:
        related = analogous_research(reg, mechanism=EconomicMechanism.TREND, root_symbol="CL")
    assert related == ()


def test_analogous_research_excludes_only_the_query_instrument():
    """Querying a root that is NOT itself a considered instrument (so the
    exclusion filter never matches anything) still returns every real
    researched instrument -- proves the exclusion is exact-match, not a
    silent broad filter."""
    with _registry() as reg:
        related = analogous_research(reg, mechanism=EconomicMechanism.TREND, root_symbol="NOT_A_REAL_ROOT")
    assert {e.instrument.root_symbol for e in related} == {"CL", "ES", "GC", "NQ", "ZN", "XLE"}


def test_cross_asset_confirmation_matches_the_builder_synthesis_exactly():
    from alpha_agent.alpha_graph.builder import build_mechanism_graph, cross_asset_synthesis

    with _registry() as reg:
        via_retrieval = cross_asset_confirmation(reg, EconomicMechanism.TREND)
    with _registry() as reg2:
        via_builder = cross_asset_synthesis(build_mechanism_graph(reg2, EconomicMechanism.TREND))
    assert via_retrieval == via_builder


def test_repeated_failure_patterns_matches_builder_aggregate():
    from alpha_agent.alpha_graph.builder import build_mechanism_graph

    with _registry() as reg:
        via_retrieval = repeated_failure_patterns(reg, EconomicMechanism.TREND)
    with _registry() as reg2:
        via_builder = dict(build_mechanism_graph(reg2, EconomicMechanism.TREND).repeated_failure_reason_codes)
    assert via_retrieval == via_builder
    assert via_retrieval  # real, non-empty reason codes exist for TREND


def test_underexplored_combinations_never_includes_a_researched_pair():
    with _registry() as reg:
        gaps = underexplored_combinations(reg)
    researched_pairs = {("TREND", "CL"), ("TREND", "ES"), ("TREND", "GC"), ("TREND", "NQ"), ("TREND", "ZN"), ("TREND", "XLE")}
    for g in gaps:
        assert (g.mechanism.value, g.instrument.root_symbol) not in researched_pairs
        assert g.coverage in (EvidenceCoverage.UNDEREXPLORED, EvidenceCoverage.NO_EVIDENCE)


def test_underexplored_combinations_can_be_scoped_to_one_mechanism():
    with _registry() as reg:
        all_gaps = underexplored_combinations(reg)
        trend_only = underexplored_combinations(reg, mechanisms=(EconomicMechanism.TREND,))
    assert len(trend_only) < len(all_gaps)
    assert all(g.mechanism is EconomicMechanism.TREND for g in trend_only)


def test_retrieval_is_deterministic():
    with _registry() as reg:
        a1 = analogous_research(reg, mechanism=EconomicMechanism.TREND, root_symbol="CL")
        a2 = analogous_research(reg, mechanism=EconomicMechanism.TREND, root_symbol="CL")
    assert a1 == a2


def test_retrieval_never_generates_a_trade_signal():
    """Sweeps every string this module can produce (real registry, every
    mechanism) for banned trade-instruction vocabulary -- the exact failure
    mode prompt section 5 calls out: "because it worked in NQ, buy XLE"."""
    from alpha_agent.alpha_graph.builder import MECHANISM_UNIVERSE

    with _registry() as reg:
        for mechanism in MECHANISM_UNIVERSE:
            synth = cross_asset_confirmation(reg, mechanism)
            for bucket in (synth.common, synth.differs, synth.repeated_failures, synth.underexplored, (synth.note,)):
                for line in bucket:
                    assert not _BANNED.search(line), f"{mechanism}: trade-signal language found: {line!r}"
            for e in analogous_research(reg, mechanism=mechanism, root_symbol="ZZZ_NONE"):
                assert not _BANNED.search(str(e.scientific_evidence))


def test_retrieval_module_never_calls_a_registry_write_method_or_network():
    with open(alpha_graph_retrieval.__file__, encoding="utf-8") as fh:
        text = fh.read()
    for pattern in (r"\.insert_experiment\(", r"\.record_failure\(", r"\.apply_bundle\("):
        assert not re.search(pattern, text)
    for token in ("AnthropicClient", "databento", "requests."):
        assert token not in text
