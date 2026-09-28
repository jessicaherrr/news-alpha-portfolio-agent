"""Phase 9.1 earnings-announcement timestamp store.

Scope, per the approved Phase 9 proposal: point-in-time announcement
DATE/TIME only, sourced and typed with the same
``point_in_time_available: bool | None`` honesty pattern Phase 1's
``MeasurableVariable`` already uses -- NOT point-in-time fundamentals, and
NOT analyst consensus/expectations (see
:mod:`alpha_agent.equities.data_availability` for that investigation's
finding, which is what gates 9.2's PEAD candidacy).

One real demonstration entry is populated (Apple's FY2023 Q4 earnings),
sourced from a real SEC EDGAR Form 8-K Item 2.02 filing -- proving the
mechanism (an Item 2.02 8-K is itself a timestamped, same-day, self-serve,
free filing, so its filing date genuinely IS a point-in-time-honest earnings
announcement date). No other ticker/quarter has been populated; an empty
``known_earnings(...)`` result must never be read as "this ticker never
reported earnings" -- it means "not yet sourced here."
"""
from __future__ import annotations

from datetime import UTC, date, datetime

from alpha_agent.equities.schemas import (
    CorporateActionSource,
    EarningsTiming,
    EquityEarningsEvent,
)

__all__ = ["KNOWN_EARNINGS_EVENTS", "known_earnings"]

_AAPL_Q4_FY2023_SOURCE = CorporateActionSource(
    name=(
        "SEC EDGAR Form 8-K Item 2.02 Ex-99.1, Apple Inc. (CIK 0000320193), filed 2023-11-02 "
        "(accession 0000320193-23-000104): 'Apple reports fourth quarter results,' quarter "
        "ended 2023-09-30, EPS $1.46."
    ),
    url="https://www.sec.gov/Archives/edgar/data/320193/000032019323000104/a8-kex991q4202309302023.htm",
    retrieved_at=datetime(2026, 9, 25, tzinfo=UTC),
)

#: A single real demonstration record. An Item 2.02 8-K's SEC filing
#: timestamp is itself the point-in-time evidence -- the announcement cannot
#: have been known any earlier than the moment the filing was accepted by
#: EDGAR, and the filing date recorded here is that same day.
KNOWN_EARNINGS_EVENTS: tuple[EquityEarningsEvent, ...] = (
    EquityEarningsEvent(
        raw_symbol="AAPL",
        fiscal_period="FY2023 Q4",
        announcement_date=date(2023, 11, 2),
        timing=EarningsTiming.AFTER_MARKET_CLOSE,
        point_in_time_available=True,
        point_in_time_note=(
            "Confirmed via a real, dated SEC EDGAR Form 8-K Item 2.02 filing -- the filing's "
            "own acceptance timestamp is the point-in-time evidence, not a third-party "
            "earnings-calendar aggregator's un-cited date."
        ),
        source=_AAPL_Q4_FY2023_SOURCE,
    ),
)


def known_earnings(raw_symbol: str) -> tuple[EquityEarningsEvent, ...]:
    """Empty for every ticker/quarter not yet sourced -- never read as "no
    earnings were ever announced" for that ticker."""
    return tuple(e for e in KNOWN_EARNINGS_EVENTS if e.raw_symbol == raw_symbol)
