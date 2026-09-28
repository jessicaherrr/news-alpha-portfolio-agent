"""Phase 6 closure: the ETF-domain mirror of
:data:`alpha_agent.translation.research_memory.MECHANISM_TO_KNOWN_FAMILIES` --
a conservative, documented bridge from an ``EconomicMechanism`` (shared
vocabulary) to the ETF strategy family key(s) that structurally implement it
TODAY (a separate, ETF-specific mapping -- Phase 6 kickoff's own principle:
"shared research concepts do NOT imply shared scientific evidence").

Deliberately tiny: exactly one real ETF strategy family exists today
(``etf_tsmom``, :mod:`alpha_agent.etf.strategy_family`). Extending this table
is a real research-scope decision, not a routine addition.
"""
from __future__ import annotations

from alpha_agent.alpha_memory.builder import list_alpha_research_objects
from alpha_agent.alpha_memory.schemas import AlphaResearchObject
from alpha_agent.core.instrument import AssetDomain
from alpha_agent.etf.strategy_family import ETF_TSMOM_FAMILY
from alpha_agent.etf.universe import PILOT_UNIVERSE
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.sqlite_registry import ExperimentRegistry

__all__ = ["ETF_MECHANISM_TO_FAMILIES", "list_etf_alpha_research_objects"]

#: XLE tsmom's real economic framing this session (EIA petroleum/energy
#: supply-demand shocks -> trend continuation in energy-linked instruments --
#: the same TREND mechanism Futures' CL/XLE-analog research already uses;
#: shared MECHANISM vocabulary, never shared evidence).
ETF_MECHANISM_TO_FAMILIES: dict[EconomicMechanism, tuple[str, ...]] = {
    EconomicMechanism.TREND: (ETF_TSMOM_FAMILY,),
}


def list_etf_alpha_research_objects(registry: ExperimentRegistry) -> tuple[AlphaResearchObject, ...]:
    """Every real, evidence-grounded ETF AlphaResearchObject -- reuses
    `alpha_memory.builder`'s entire aggregation machinery unchanged, supplying
    only ETF-specific inputs (asset_domain, the pilot universe as the root
    list, and this module's own mechanism->family map instead of Futures'
    frozen bridge). Returns () today unless a real ETF experiment exists in
    the registry -- never a fabricated empty-but-present object."""
    return list_alpha_research_objects(
        registry, asset_domain=AssetDomain.ETF, roots_override=PILOT_UNIVERSE,
        mechanism_family_map=ETF_MECHANISM_TO_FAMILIES,
    )
