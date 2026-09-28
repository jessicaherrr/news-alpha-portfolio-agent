"""Phase 7 -- the small, deterministic Agent retrieval layer (prompt section
5) over the Cross-Asset Alpha Graph.

Research prioritization only. Every function here is a pure read over
:mod:`alpha_agent.alpha_graph.builder`'s output -- no live LLM call, no
network call, no registry write. The Agent MAY say "this Mechanism has
related evidence in NQ and XLE"; it must NEVER say "because it worked in NQ,
buy XLE" (prompt section 5) -- related evidence is never an unsupported trade
signal. ``test_phase7_alpha_graph_retrieval.py`` enforces this with a banned-
vocabulary sweep over every string this module can produce.
"""
from __future__ import annotations

from alpha_agent.alpha_graph.builder import (
    build_alpha_graph,
    build_mechanism_graph,
    cross_asset_synthesis,
    list_research_gaps,
)
from alpha_agent.alpha_graph.schemas import (
    CrossAssetSynthesis,
    EvidenceCoverage,
    InstrumentEvidence,
    ResearchGapRow,
)
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.sqlite_registry import ExperimentRegistry

__all__ = [
    "analogous_research",
    "cross_asset_confirmation",
    "repeated_failure_patterns",
    "underexplored_combinations",
]


def analogous_research(
    registry: ExperimentRegistry, *, mechanism: EconomicMechanism, root_symbol: str,
) -> tuple[InstrumentEvidence, ...]:
    """"Where else has this Mechanism been researched?" -- every OTHER
    instrument (any asset domain) with real evidence (RESEARCHED or
    WEAKLY_RESEARCHED), excluding ``root_symbol`` itself. An empty result is
    an honest "no analogous research yet", never an error."""
    view = build_mechanism_graph(registry, mechanism)
    return tuple(
        e for e in view.instrument_evidence
        if e.instrument.root_symbol != root_symbol
        and e.coverage in (EvidenceCoverage.RESEARCHED, EvidenceCoverage.WEAKLY_RESEARCHED)
    )


def cross_asset_confirmation(registry: ExperimentRegistry, mechanism: EconomicMechanism) -> CrossAssetSynthesis:
    """"What is common, what differs, repeated failures, what remains
    underexplored?" for one Mechanism, across every asset domain with real
    evidence (prompt section 8). Never transfers a verdict -- see
    :func:`alpha_agent.alpha_graph.builder.cross_asset_synthesis`."""
    return cross_asset_synthesis(build_mechanism_graph(registry, mechanism))


def repeated_failure_patterns(registry: ExperimentRegistry, mechanism: EconomicMechanism) -> dict[str, int]:
    """Reason code -> count, aggregated across every instrument researched
    under this Mechanism (any asset domain) -- always traceable back to
    `build_mechanism_graph`'s own per-instrument breakdown, never presented
    as a standalone number."""
    return dict(build_mechanism_graph(registry, mechanism).repeated_failure_reason_codes)


def underexplored_combinations(
    registry: ExperimentRegistry, *, mechanisms: tuple[EconomicMechanism, ...] | None = None,
) -> tuple[ResearchGapRow, ...]:
    """Every (Mechanism, Instrument) combination with UNDEREXPLORED or
    NO_EVIDENCE coverage -- first-class research gaps, never failures
    (prompt section 6)."""
    return list_research_gaps(build_alpha_graph(registry, mechanisms))
