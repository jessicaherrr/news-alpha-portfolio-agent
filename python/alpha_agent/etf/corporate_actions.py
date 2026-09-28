"""The Phase 6 pilot's known-corporate-actions store.

Every entry here is either a real, attributable, cited record (never
inferred from a price jump -- Phase 6 instruction 3) or an explicit
"not yet sourced" fact (never a silent empty-means-nothing-happened
assumption -- these are all real, established, distribution-paying funds).

Splits: the Phase 6 acquisition audit ran a free day-over-day close-ratio
scan (>=45% move) across the whole 14-ticker, 2018-05-01..2024-12-30 window
and found exactly one candidate: USO on 2020-04-29. That scan is a
heuristic SCREEN, not a source -- the actual recorded action below is
sourced from USO's real SEC EDGAR Form 8-K filing, found by a separate,
targeted search, not derived from the price data at all.

Distributions: Phase 6 instruction 7's feasibility investigation (real web
research, not assumption) found no free, self-serve source with 2018-2024
depth for this universe -- Databento does not carry corporate actions
(confirmed empirically, not assumed: the full real schema list across all
5 acquired datasets carries no dividend/split schema); Databento's own new
"Corporate Actions" product exists but pricing is sales-gated, not a
self-serve cost estimate; free tiers of third-party vendors (e.g. EODHD)
cap free historical depth at ~1 year, far short of this window; official
fund-sponsor "distribution history" pages (State Street/BlackRock/Invesco)
are the most promising free/official avenue but were not yet built into a
connector. This is reported, not silently worked around -- see the Phase 6
roadmap-state doc and the session's report to the user. No distribution
record is populated below; do not add one without a real citation.
"""
from __future__ import annotations

from datetime import UTC, date, datetime

from alpha_agent.etf.schemas import (
    CorporateActionCoverage,
    CorporateActionCoverageStatus,
    CorporateActionSource,
    EtfCashDistribution,
    EtfSplitAction,
)
from alpha_agent.etf.universe import PILOT_UNIVERSE

__all__ = [
    "COVERAGE",
    "KNOWN_DISTRIBUTIONS",
    "KNOWN_SPLITS",
    "coverage",
    "known_distributions",
    "known_splits",
]

_USO_8K_SOURCE = CorporateActionSource(
    name="SEC EDGAR Form 8-K, United States Oil Fund LP (CIK 1327068), filed 2026-09-23 lookup of a 2020-04-22 filing",
    url="https://www.sec.gov/Archives/edgar/data/1327068/000117120020000271/i20263_uso-8k.htm",
    retrieved_at=datetime(2026, 9, 23, tzinfo=UTC),
)

#: The one real, sourced split in the whole pilot universe/window.
KNOWN_SPLITS: tuple[EtfSplitAction, ...] = (
    EtfSplitAction(
        raw_symbol="USO",
        effective_date=date(2020, 4, 29),
        ratio=0.125,  # 1-for-8 reverse split
        source=_USO_8K_SOURCE,
    ),
)

#: Not yet sourced -- see module docstring. Deliberately empty.
KNOWN_DISTRIBUTIONS: tuple[EtfCashDistribution, ...] = ()

_SCREEN_NOTE = (
    "Phase 6 audit: a day-over-day close-ratio scan (>=45% move) across "
    "2018-05-01..2024-12-30 found no candidate event for this ticker. A "
    "heuristic screen only -- weak evidence of absence, not an official "
    "confirmation that no split occurred."
)
_USO_SPLIT_NOTE = (
    "Sourced from SEC EDGAR Form 8-K (United States Oil Fund LP, CIK 1327068), "
    "not inferred from the price scan that originally flagged the date as a "
    "candidate worth investigating."
)
_DISTRIBUTIONS_NOTE = (
    "No free/self-serve source with 2018-2024 depth was found for this "
    "universe (Phase 6 instruction 7 feasibility investigation). Not "
    "populated pending an explicit data-source decision from the user."
)

COVERAGE: dict[str, CorporateActionCoverage] = {
    ticker: CorporateActionCoverage(
        raw_symbol=ticker,
        splits_status=(
            CorporateActionCoverageStatus.SOURCED_FROM_OFFICIAL_RECORD
            if ticker == "USO"
            else CorporateActionCoverageStatus.SCREENED_NO_EVENT_DETECTED
        ),
        splits_note=_USO_SPLIT_NOTE if ticker == "USO" else _SCREEN_NOTE,
        distributions_status=CorporateActionCoverageStatus.NOT_YET_INVESTIGATED,
        distributions_note=_DISTRIBUTIONS_NOTE,
    )
    for ticker in PILOT_UNIVERSE
}


def known_splits(raw_symbol: str) -> tuple[EtfSplitAction, ...]:
    return tuple(s for s in KNOWN_SPLITS if s.raw_symbol == raw_symbol)


def known_distributions(raw_symbol: str) -> tuple[EtfCashDistribution, ...]:
    return tuple(d for d in KNOWN_DISTRIBUTIONS if d.raw_symbol == raw_symbol)


def coverage(raw_symbol: str) -> CorporateActionCoverage:
    if raw_symbol not in COVERAGE:
        raise KeyError(
            f"{raw_symbol!r} is not in the Phase 6 pilot universe; no coverage record exists"
        )
    return COVERAGE[raw_symbol]
