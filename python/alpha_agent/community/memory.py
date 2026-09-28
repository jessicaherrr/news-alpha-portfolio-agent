"""Phase 8 -- Community Memory (prompt section 8).

Event/Context -> Personal + Community Memory -> factors/strategies/
replications -> research prioritization. This module composes, without
duplicating, two already-real read models:

* Personal memory: :func:`alpha_agent.alpha_memory.mechanism_memory_lookup`
  (Phase 1/2), reused VERBATIM -- never a second personal-memory computation.
* Community memory: :func:`alpha_agent.community.visibility.aggregate_for_mechanism_root`
  (this phase), which already enforces the privacy boundary (PRIVATE
  contributions never enter it).

:func:`community_memory_lookup` is the one Agent-facing entry point
(:mod:`alpha_agent.ui.services`'s ``community_memory_lookup`` wraps it for the
UI). Its ``research_prioritization_note`` is deterministic free text, always
ending with an explicit disclaimer -- "never a guaranteed trade
recommendation" (prompt section 8) -- and is swept by a dedicated repo-wide
test (mirroring Phase 7's own banned-trade-vocabulary sweep) for
buy/sell/recommend/guaranteed-style language.
"""
from __future__ import annotations

from alpha_agent.alpha_memory import mechanism_memory_lookup as _personal_mechanism_lookup
from alpha_agent.community.schemas import CommunityMemoryLookup
from alpha_agent.community.store import CommunityStore
from alpha_agent.community.visibility import aggregate_for_mechanism_root
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.sqlite_registry import ExperimentRegistry

__all__ = ["DISCLAIMER", "community_memory_lookup"]

#: The one fixed, reviewed closing sentence every note ends with -- allow-
#: listed by name so a banned-trade-vocabulary sweep (mirroring Phase 7's
#: own `alpha_graph` sweep) can exclude this SANCTIONED disclaimer sentence
#: (which legitimately contains "guaranteed"/"recommendation" in a negating
#: context) while still scanning every OTHER word this module ever emits.
DISCLAIMER = (
    "This is context for research prioritization only -- never a guaranteed trade recommendation, "
    "and never a signal to act on directly."
)


def _research_prioritization_note(personal, community) -> str:
    parts: list[str] = []
    if personal.match_kind.value == "NONE":
        parts.append("No personal research memory exists for this mechanism/market yet.")
    else:
        parts.append(
            f"Personal memory: {personal.match_kind.value} across {len(personal.objects)} research object(s)."
        )
    if community is None:
        parts.append("No shared or public community evidence exists yet for this mechanism/market.")
    else:
        parts.append(
            f"Community: {community.n_public_contributions} public and {community.n_shared_contributions} "
            f"shared contribution(s) from {community.n_contributors} contributor(s), "
            f"{community.n_replications} independent replication(s). Evidence maturity distribution: "
            f"{community.evidence_maturity_distribution}."
        )
    parts.append(DISCLAIMER)
    return " ".join(parts)


def community_memory_lookup(
    store: CommunityStore, registry: ExperimentRegistry, *, mechanism: EconomicMechanism, root_symbol: str
) -> CommunityMemoryLookup:
    personal = _personal_mechanism_lookup(registry, mechanism=mechanism, root_symbol=root_symbol)
    community = aggregate_for_mechanism_root(store, mechanism=mechanism, root_symbol=root_symbol)
    return CommunityMemoryLookup(
        mechanism=mechanism,
        root_symbol=root_symbol,
        personal=personal.model_dump(mode="json"),
        community=community,
        research_prioritization_note=_research_prioritization_note(personal, community),
    )
