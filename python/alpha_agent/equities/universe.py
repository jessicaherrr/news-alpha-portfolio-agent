"""The Phase 9.1 limited liquid-equities universe and its real, empirically-
resolved dataset facts.

Every fact here is either a user/product decision (the ticker list, frozen at
a stated declaration date) or something measured from a live Databento
capability probe (``scripts/phase9_1_equity_capability_probe.py``) -- never a
guess. See ``docs/PHASE_9_1_EQUITIES_FOUNDATION.md`` for the full audit this
module encodes.

Survivorship-bias caveat (explicit, not hidden): this V1 universe was chosen
FROM 2026 hindsight as 20 large-cap names that were all still actively
trading at the time of this declaration. That is itself a survivorship-biased
selection process -- a name that would have delisted or been acquired inside
the 2018-2024 research window and is therefore no longer prominent today is
systematically less likely to have been picked. Phase 9.1 does not attempt
point-in-time historical index reconstruction (out of scope per the approved
Phase 9 proposal); it only guarantees that *within* this fixed, dated
universe, any future delisting event is handled as an explicit typed event
(see ``alpha_agent.equities.membership``), never silently gap-filled. The
survivorship limitation is in how the universe was CHOSEN, not in how a
delisting occurring after choice would be processed.
"""
from __future__ import annotations

from datetime import date

#: Frozen, dated declaration -- this exact 20-name list must never be
#: silently edited in place. Any change to membership is a new, separately
#: dated universe (mirrors the ETF pilot's own "finalized Phase 6 pilot
#: universe (user-approved 2026-09-22)" precedent).
UNIVERSE_DECLARED_AT: date = date(2026, 9, 25)

#: Phase 9.1 limited liquid-equities universe: 20 large-cap, high-ADV single
#: names spanning multiple sectors -- same order of magnitude as the ETF
#: pilot's 14, deliberately not a broad index (per the approved Phase 9
#: proposal's "15-25 large-cap, high-ADV single names" instruction).
EQUITY_UNIVERSE: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "BAC",
    "JNJ", "UNH", "PFE",
    "XOM", "CVX",
    "WMT", "PG", "KO",
    "HD", "DIS", "CAT",
)

#: Sector label per ticker -- documentation/provenance metadata only, never
#: consulted by any statistical/identity path in Phase 9.1 (no cross-sectional
#: research exists yet).
SECTOR: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Communication Services", "META": "Communication Services", "DIS": "Communication Services",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "JPM": "Financials", "BAC": "Financials",
    "JNJ": "Health Care", "UNH": "Health Care", "PFE": "Health Care",
    "XOM": "Energy", "CVX": "Energy",
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "CAT": "Industrials",
}

#: Real primary-listing exchange per ticker. Unlike the ETF pilot (where
#: `symbology.resolve` presence does not prove primary listing under Reg
#: NMS), a single-name equity's IPO/listing exchange is public record; still
#: to be cross-checked against the real acquired `definition` schema's
#: `exchange` field once data is acquired (Phase 9.1 does not acquire data --
#: this dict is the pre-acquisition declared expectation, not yet verified
#: against a real definition record the way the ETF pilot's was).
PRIMARY_LISTING_EXCHANGE: dict[str, str] = {
    "AAPL": "XNAS", "MSFT": "XNAS", "NVDA": "XNAS", "GOOGL": "XNAS", "AMZN": "XNAS",
    "META": "XNAS", "TSLA": "XNAS",
    "JPM": "XNYS", "BAC": "XNYS", "JNJ": "XNYS", "UNH": "XNYS", "PFE": "XNYS",
    "XOM": "XNYS", "CVX": "XNYS", "WMT": "XNYS", "PG": "XNYS", "KO": "XNYS",
    "HD": "XNYS", "DIS": "XNYS", "CAT": "XNYS",
}

#: Databento dataset code for each real exchange MIC used above -- the exact
#: same dataset codes the ETF pilot already acquired and proved (Phase 6
#: audit: byte-identical `exchange` field across independently-acquired
#: datasets), reused here rather than guessed.
EXCHANGE_TO_PRIMARY_DATASET: dict[str, str] = {
    "XNAS": "XNAS.ITCH",
    "XNYS": "XNYS.PILLAR",
}

#: (dataset, start, end) -- populated from
#: scripts/phase9_1_equity_capability_probe.py's real, live
#: `metadata.get_dataset_range` result (never derived from datetime.now()).
#: Bounded above by 2024-12-31, one day before the locked 2025-01-01 holdout,
#: matching the ETF pilot's own window exactly -- no acquisition has happened
#: yet, so this is the CANDIDATE research window this foundation targets, not
#: a claim that any of it has been downloaded.
DATASET_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("XNAS.ITCH", "2018-05-01", "2024-12-31"),
    ("XNYS.PILLAR", "2018-05-01", "2024-12-31"),
)


def primary_listing_dataset(raw_symbol: str) -> str:
    """The Databento dataset code for `raw_symbol`'s declared primary listing
    exchange. Raises KeyError for a ticker outside the Phase 9.1 universe --
    never guesses a fallback."""
    exchange = PRIMARY_LISTING_EXCHANGE[raw_symbol]
    return EXCHANGE_TO_PRIMARY_DATASET[exchange]


def equity_dsl_root_symbol(raw_symbol: str) -> str:
    """The synthetic "virtual root" symbol used ONLY inside the StrategySpec
    DSL / target schedule / ContractSpec.root_symbol -- never in any
    human-facing label (registry root_symbol, dataset identity, friendly ids
    all use the real ticker).

    Mirrors ``alpha_agent.etf.universe.etf_dsl_root_symbol`` exactly, with a
    distinct ``"S"`` (Share) prefix -- deliberately different from ETF's
    ``"E"`` prefix so an Equity and an ETF sharing a raw ticker string (not
    expected, but not structurally prevented) can never collide on the same
    DSL virtual root."""
    return f"S{raw_symbol}"
