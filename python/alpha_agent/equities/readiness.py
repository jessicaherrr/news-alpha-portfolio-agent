"""A single, deterministic "Research Readiness" summary for the Phase 9.1
Equities foundation -- the human concept the approved Phase 9 proposal's UI
principle names (goal 6: show "Equities / Data Coverage / Corporate Actions /
Historical Universe / Research Readiness," never raw enum/schema names).

Phase 9.1 does not wire this into any UI page (no Equities page exists yet,
and none should per the approved proposal -- Equities surfaces inside
Research's existing sub-tabs starting in Phase 9.3). This module exists so
that WHEN a future phase does surface it, the human-readable computation
already exists, is tested, and reads only real declared/typed facts -- never
a fabricated "green checkmark."
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from alpha_agent.equities import corporate_actions as ca
from alpha_agent.equities import data_availability as da
from alpha_agent.equities import earnings
from alpha_agent.equities import membership as mem
from alpha_agent.equities.data_source import load_primary_listing_bars
from alpha_agent.equities.schemas import CorporateActionCoverageStatus, UniverseMembershipStatus
from alpha_agent.equities.universe import EQUITY_UNIVERSE, UNIVERSE_DECLARED_AT

__all__ = ["EquitiesFoundationReadiness", "equities_foundation_readiness"]


class EquitiesFoundationReadiness(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    universe_size: int
    universe_declared_at: date
    active_members: int
    delisted_members: int
    market_data_acquired: bool
    splits_sourced_count: int
    splits_not_yet_investigated_count: int
    earnings_events_sourced_count: int
    pit_fundamentals_availability: str
    pit_analyst_consensus_availability: str
    pead_admissible_for_phase_9_2: bool


def _market_data_acquired() -> bool:
    try:
        load_primary_listing_bars(EQUITY_UNIVERSE[0])
    except FileNotFoundError:
        return False
    return True


def equities_foundation_readiness() -> EquitiesFoundationReadiness:
    memberships = [mem.membership(t) for t in EQUITY_UNIVERSE]
    coverages = [ca.coverage(t) for t in EQUITY_UNIVERSE]
    return EquitiesFoundationReadiness(
        universe_size=len(EQUITY_UNIVERSE),
        universe_declared_at=UNIVERSE_DECLARED_AT,
        active_members=sum(1 for m in memberships if m.status == UniverseMembershipStatus.ACTIVE),
        delisted_members=sum(1 for m in memberships if m.status == UniverseMembershipStatus.DELISTED),
        market_data_acquired=_market_data_acquired(),
        splits_sourced_count=sum(
            1 for c in coverages if c.splits_status == CorporateActionCoverageStatus.SOURCED_FROM_OFFICIAL_RECORD
        ),
        splits_not_yet_investigated_count=sum(
            1 for c in coverages if c.splits_status == CorporateActionCoverageStatus.NOT_YET_INVESTIGATED
        ),
        earnings_events_sourced_count=len(earnings.KNOWN_EARNINGS_EVENTS),
        pit_fundamentals_availability=da.finding("point_in_time_fundamentals").availability.value,
        pit_analyst_consensus_availability=da.finding("analyst_consensus_expectations").availability.value,
        pead_admissible_for_phase_9_2=da.pead_admissible_for_phase_9_2(),
    )
