"""The UI's ONLY boundary into `alpha_agent.news_alpha` (mirrors
`alpha_agent.ui.translation_context`'s role for Phase 1).

Reads news/events ONLY through `market_intel_context`'s cached (never
live-fetching) accessors, never calls an LLM, never opens the registry. The
Phase B mechanism graph is built from the scan alone (never the mandate); the
Phase C signal paths from that graph alone; the mechanism-adjusted impact
from the scan plus those paths, as a separate object beside the scan; the
Phase D asset expressions from those paths under the current mandate; the
Phase E candidate signals from those expressions and paths (through the
Phase D gate). Factor screening is NOT reached from here -- it is the
screening plane (`ui.candidate_signal_view`). The
mandate's access/constraint half persists via `MandateStore` (gitignored
`data/user_prefs/`); its risk half is always the session's current
`InvestorProfile`, so the two can never disagree.
"""
from __future__ import annotations

from functools import lru_cache

import streamlit as st

from alpha_agent.news_alpha import (
    AllowedAssetUniverse,
    AssetExpressionPlan,
    CandidateSignalSet,
    DomainCapabilities,
    EconomicMechanismGraph,
    ImpactLevel,
    InitialImpactScan,
    MandateStore,
    MechanismAdjustedImpact,
    ResearchMandate,
    SignalPathDiscovery,
    TranslationHandoffPlan,
    UserDescribedEvent,
    adjust_impact,
    build_asset_expressions,
    build_candidate_signals,
    build_economic_mechanism_graph,
    build_translation_handoffs,
    discover_signal_paths,
    probe_domain_capabilities,
    resolve_allowed_universe,
    scan_initial_impact,
    scan_many,
)
from alpha_agent.ui import market_intel_context, panels

__all__ = [
    "MANDATE_STORE",
    "add_user_event",
    "adjusted_impact",
    "allowed_universe",
    "asset_expressions",
    "cached_event_scans",
    "candidate_signals",
    "current_mandate",
    "handoff_plan",
    "has_saved_mandate",
    "mechanism_graph",
    "save_mandate",
    "scan_event_source",
    "scan_user_event",
    "signal_paths",
    "user_event_scans",
]

#: Module-level so tests (and `tests/python/conftest.py`'s autouse isolation)
#: can point it at a temporary path -- never the user's real preference file.
MANDATE_STORE = MandateStore()

_MANDATE_KEY = "news_alpha_mandate"
_USER_EVENTS_KEY = "news_alpha_user_events"
_MAX_USER_EVENTS = 5
_MAX_CACHED_SCANS = 6


@lru_cache(maxsize=1)
def _capabilities() -> DomainCapabilities:
    """One cheap raw-store path probe per process."""
    return probe_domain_capabilities()


def current_mandate() -> ResearchMandate:
    profile = panels.current_investor_profile()
    cached: ResearchMandate | None = st.session_state.get(_MANDATE_KEY)
    if cached is None or cached.risk_profile != profile:
        cached = MANDATE_STORE.load(risk_profile=profile)
        st.session_state[_MANDATE_KEY] = cached
    return cached


def save_mandate(mandate: ResearchMandate) -> None:
    MANDATE_STORE.save(mandate)
    st.session_state[_MANDATE_KEY] = mandate


def has_saved_mandate() -> bool:
    return MANDATE_STORE.exists()


def allowed_universe(mandate: ResearchMandate) -> AllowedAssetUniverse:
    return resolve_allowed_universe(mandate, capabilities=_capabilities())


def cached_event_scans(
    mandate: ResearchMandate, *, limit: int = _MAX_CACHED_SCANS, window_hours: int | None = None,
) -> tuple[InitialImpactScan, ...]:
    """Initial impact scans for the already-cached news/events that touch at
    least one allowed domain, most relevant first. Items the scan finds
    irrelevant everywhere (NONE / not assessed) are left out of the list --
    noise, not evidence. Empty before any explicit Market refresh -- the
    honest answer. (User-described events are always shown; see
    `user_event_scans`.)"""
    news = (market_intel_context.cached_recent_news(limit=50) if window_hours is None
            else market_intel_context.cached_recent_news(limit=50, window_hours=window_hours))
    items = (*news, *market_intel_context.cached_upcoming_events())
    if not items:
        return ()
    scans = scan_many(items, mandate, universe=allowed_universe(mandate))
    relevant = [s for s in scans if s.top_level is not None and s.top_level is not ImpactLevel.NONE]
    return tuple(relevant[:limit])


def add_user_event(text: str) -> UserDescribedEvent | None:
    """Session-only: a user-described event is never written to NewsStore.

    IDENTITY CONTINUITY: re-describing the same text (from the "Describe a
    market event" box OR the Agent chat) returns the SAME `UserDescribedEvent`
    -- same ``event_id``, same ``described_at`` -- moved to the front, never a
    second event for one thing the user said. The chat reply, the triage card,
    and the mechanism graph therefore always refer to one event."""
    text = " ".join((text or "").split())
    if len(text) < 3:
        return None
    existing = st.session_state.get(_USER_EVENTS_KEY, [])
    event = next((e for e in existing if e.text == text), None) or UserDescribedEvent.create(text)
    others = [e for e in existing if e.event_id != event.event_id]
    st.session_state[_USER_EVENTS_KEY] = [event, *others][:_MAX_USER_EVENTS]
    return event


def scan_user_event(event: UserDescribedEvent, mandate: ResearchMandate) -> InitialImpactScan:
    """The ONE Phase A scan path for a user-described event (`scan_initial_impact`
    under the mandate-scoped universe) -- shared by the triage cards and the
    Agent chat so both see the identical scan."""
    return scan_initial_impact(event, mandate, universe=allowed_universe(mandate))


def scan_event_source(source, mandate: ResearchMandate) -> InitialImpactScan:
    """The same Phase A scan for any event source a Research Thread holds (a
    cached news item, a scheduled event, or a user-described event) -- one
    path, so a thread and the News feed always see the identical scan."""
    return scan_initial_impact(source, mandate, universe=allowed_universe(mandate))


def user_event_scans(mandate: ResearchMandate) -> tuple[InitialImpactScan, ...]:
    events = st.session_state.get(_USER_EVENTS_KEY, [])
    return tuple(scan_user_event(e, mandate) for e in events)


def handoff_plan(scan: InitialImpactScan) -> TranslationHandoffPlan:
    return build_translation_handoffs(scan)


def mechanism_graph(scan: InitialImpactScan) -> EconomicMechanismGraph:
    """The event's economic mechanism graph -- pure and cheap (a few dozen
    seeded claims), so it is rebuilt per render rather than cached."""
    return build_economic_mechanism_graph(scan)


def signal_paths(graph: EconomicMechanismGraph) -> SignalPathDiscovery:
    """Signal paths for one event's mechanism graph -- read from the graph
    alone (so never shaped by the mandate); pure, rebuilt per render."""
    return discover_signal_paths(graph)


def adjusted_impact(scan: InitialImpactScan, paths: SignalPathDiscovery) -> MechanismAdjustedImpact:
    """The mechanism-adjusted second pass. A separate object: ``scan`` (the
    initial assessment) is read, never modified."""
    return adjust_impact(scan, paths)


def asset_expressions(paths: SignalPathDiscovery, mandate: ResearchMandate) -> AssetExpressionPlan:
    """Where each consequence could be expressed, and what the platform can
    measure there. Economic relevance comes from the paths alone; only the
    admission of each expression reads the mandate. Pure, rebuilt per render
    (a few milliseconds), against the process's one capability snapshot."""
    return build_asset_expressions(paths, allowed_universe(mandate), mandate)


def candidate_signals(expressions: AssetExpressionPlan, paths: SignalPathDiscovery) -> CandidateSignalSet:
    """Testable factor hypotheses for the expressions' usable measurements --
    pure and cheap (no data is loaded), rebuilt per render."""
    return build_candidate_signals(expressions, paths)
