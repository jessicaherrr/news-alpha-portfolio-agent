"""Phase 9.1's typed record of a real data-feasibility investigation --
mirrors the honesty discipline the Phase 6 ETF pilot used for dividends
(``alpha_agent.etf.corporate_actions`` module docstring: "found no free,
self-serve source... reported, not silently worked around").

The approved Phase 9 proposal requires this investigation to happen BEFORE
9.2 picks a candidate factor family: "This step must also verify whether a
reliable point-in-time expectations/consensus reference exists alongside the
announcement timestamp (PEAD needs both); 9.2's candidate-family choice is
gated on that verified finding, not assumed here." This module IS that gate,
expressed as typed, importable facts rather than prose a future phase could
silently disagree with.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class DataSourceAvailability(str, Enum):
    """How available a data concept genuinely is -- never optimistically
    assumed."""

    #: A real, free, self-serve source was found and is usable as-is.
    AVAILABLE_FREE_SELF_SERVE = "AVAILABLE_FREE_SELF_SERVE"
    #: A real source exists but requires a paid subscription/license --
    #: a MONEY decision, not an engineering one.
    AVAILABLE_PAID_ONLY = "AVAILABLE_PAID_ONLY"
    #: Investigated; no usable source (free or paid) was identified.
    NOT_AVAILABLE = "NOT_AVAILABLE"


class DataConceptFinding(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    concept: str
    availability: DataSourceAvailability
    finding: str
    investigated_at: str  # ISO date; kept as str to avoid importing datetime just for one field


#: Real investigation findings, 2026-09-25 sourcing pass (see
#: ``alpha_agent.equities.earnings`` for the concrete AAPL demonstration this
#: first finding is based on, and the Phase 9 proposal's own analyst-vendor
#: table for the second/third).
FINDINGS: tuple[DataConceptFinding, ...] = (
    DataConceptFinding(
        concept="earnings_announcement_timestamp",
        availability=DataSourceAvailability.AVAILABLE_FREE_SELF_SERVE,
        finding=(
            "SEC EDGAR Form 8-K Item 2.02 filings are a genuinely free, self-serve, "
            "point-in-time source for earnings-announcement date/time for any US-listed "
            "issuer: the filing itself is timestamped at SEC acceptance, and EDGAR's "
            "full-text search (efts.sec.gov/LATEST/search-index) can locate them per "
            "issuer/period. Demonstrated for real on Apple's FY2023 Q4 filing "
            "(alpha_agent.equities.earnings.KNOWN_EARNINGS_EVENTS). Not yet built into a "
            "systematic per-ticker/per-quarter connector for the whole universe -- one "
            "real demonstration record exists, not full coverage."
        ),
        investigated_at="2026-09-25",
    ),
    DataConceptFinding(
        concept="analyst_consensus_expectations",
        availability=DataSourceAvailability.AVAILABLE_PAID_ONLY,
        finding=(
            "No free, self-serve source with reconstructable point-in-time history was "
            "found. IBES (LSEG/Refinitiv), Visible Alpha (S&P), FactSet Estimates, and "
            "Zacks are the standard vendors and all require a paid license. Free sources "
            "(Yahoo Finance, Finnhub, Nasdaq Data Link) expose only a CURRENT consensus "
            "number, not an auditable 'what was consensus as of date X, before later "
            "revisions' series -- that reconstruction fidelity is exactly the paid "
            "vendors' product. Consequence for 9.2: PEAD (post-earnings-announcement "
            "drift), which needs both the announcement timestamp AND a defensible "
            "surprise/expectations reference, is NOT admissible as a candidate factor "
            "family without an explicit, separately-approved paid-data decision -- this "
            "finding is what forces that gate, not a policy choice made here."
        ),
        investigated_at="2026-09-25",
    ),
    DataConceptFinding(
        concept="point_in_time_fundamentals",
        availability=DataSourceAvailability.AVAILABLE_PAID_ONLY,
        finding=(
            "Databento does not carry fundamentals data (it is a market-data vendor, "
            "same finding the Phase 6 ETF pilot already established for corporate "
            "actions). SEC EDGAR's XBRL structured financial data "
            "(data.sec.gov/api/xbrl/) is a real, free, genuinely point-in-time-honest "
            "candidate -- each filing's own SEC acceptance timestamp is a true PIT "
            "anchor, unlike a restated/as-of-today fundamentals snapshot -- but no "
            "connector has been built against it yet; this is a real, named, deferred "
            "engineering task, not a data-availability dead end the way analyst "
            "consensus is. Phase 9.1 does not populate any fundamentals schema (per the "
            "approved proposal's 'only if actually available' instruction) rather than "
            "ship a half-built, never-populated placeholder type."
        ),
        investigated_at="2026-09-25",
    ),
)


def finding(concept: str) -> DataConceptFinding:
    for f in FINDINGS:
        if f.concept == concept:
            return f
    raise KeyError(f"no data-availability finding recorded for concept {concept!r}")


def pead_admissible_for_phase_9_2() -> bool:
    """The single boolean 9.2 should consult before treating PEAD as a
    candidate factor family -- computed from the recorded finding, never
    re-asserted independently."""
    return finding("analyst_consensus_expectations").availability == DataSourceAvailability.AVAILABLE_FREE_SELF_SERVE
