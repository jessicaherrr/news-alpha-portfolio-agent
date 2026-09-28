"""Phase 7 -- the Cross-Asset Alpha Graph (prompt 7).

A RESEARCH RELATIONSHIP / NAVIGATION layer connecting existing Futures + ETF
research memory: where a Mechanism has been researched, across which assets,
with which Factors/Strategies, what evidence exists, what repeatedly failed,
and what remains underexplored.

A typed, read-only aggregation over :mod:`alpha_agent.registry` (scientific
truth), :mod:`alpha_agent.alpha_memory` (Futures Personal Alpha Memory),
:mod:`alpha_agent.etf.alpha_memory_bridge` (ETF Personal Alpha Memory), and
:mod:`alpha_agent.translation` (Mechanism/Event vocabulary). This package
never writes to the registry, never calls Claude, never calls a market-data
API, never runs a backtest, and never merges or transfers a verdict across
instruments or asset domains -- see ``builder.py`` and ``retrieval.py``
module docstrings.
"""
from __future__ import annotations

from alpha_agent.alpha_graph.builder import (
    MECHANISM_UNIVERSE,
    build_alpha_graph,
    build_mechanism_graph,
    cross_asset_synthesis,
    list_considered_instruments,
    list_research_gaps,
    summarize,
)
from alpha_agent.alpha_graph.retrieval import (
    analogous_research,
    cross_asset_confirmation,
    repeated_failure_patterns,
    underexplored_combinations,
)
from alpha_agent.alpha_graph.schemas import (
    ALPHA_GRAPH_SCHEMA,
    CrossAssetSynthesis,
    EventCategoryRef,
    EvidenceCoverage,
    FactorUnderMechanism,
    GraphNodeType,
    InstrumentEvidence,
    InstrumentRef,
    MechanismGraphSummary,
    MechanismGraphView,
    ResearchGapRow,
)

__all__ = [
    "ALPHA_GRAPH_SCHEMA",
    "MECHANISM_UNIVERSE",
    "CrossAssetSynthesis",
    "EventCategoryRef",
    "EvidenceCoverage",
    "FactorUnderMechanism",
    "GraphNodeType",
    "InstrumentEvidence",
    "InstrumentRef",
    "MechanismGraphSummary",
    "MechanismGraphView",
    "ResearchGapRow",
    "analogous_research",
    "build_alpha_graph",
    "build_mechanism_graph",
    "cross_asset_confirmation",
    "cross_asset_synthesis",
    "list_considered_instruments",
    "list_research_gaps",
    "repeated_failure_patterns",
    "summarize",
    "underexplored_combinations",
]
