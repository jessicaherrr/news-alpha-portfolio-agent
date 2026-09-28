"""The Phase 6 pilot's universe and real, empirically-resolved dataset facts.

Every fact here is either a user decision (the ticker list) or something
measured from the real acquired Databento bytes (venue, windows, row
structure) -- never a guess. See ``docs/AGENTIC_ALPHA_ROADMAP_STATE.md``'s
Phase 6 row for the full audit this module encodes.
"""
from __future__ import annotations

#: Finalized Phase 6 pilot universe (user-approved 2026-09-22): broad equity,
#: Nasdaq/growth, small caps, 4 sectors, full rates curve, credit, gold,
#: oil/energy.
PILOT_UNIVERSE: tuple[str, ...] = (
    "SPY", "QQQ", "IWM",
    "XLK", "XLF", "XLE", "XLU",
    "SHY", "IEF", "TLT",
    "LQD", "HYG",
    "GLD", "USO",
)

#: Exposure label per ticker -- documentation/provenance metadata only
#: (mirrors `alpha_agent.equities.universe.SECTOR`), naming what each fund
#: holds exactly as the finalized universe comment above groups them. Never
#: consulted by any statistical/identity path; read by the News Alpha
#: universe resolver for display grouping only.
EXPOSURE: dict[str, str] = {
    "SPY": "Broad US Equity", "QQQ": "Nasdaq-100 Growth Equity", "IWM": "US Small-Cap Equity",
    "XLK": "Technology Sector", "XLF": "Financials Sector", "XLE": "Energy Sector", "XLU": "Utilities Sector",
    "SHY": "Short-Term Treasuries", "IEF": "Intermediate Treasuries", "TLT": "Long-Term Treasuries",
    "LQD": "Investment-Grade Credit", "HYG": "High-Yield Credit",
    "GLD": "Gold", "USO": "Crude Oil",
}

#: Real primary-listing exchange per ticker, resolved from the `exchange`
#: field of the real acquired `definition` schema records -- byte-identical
#: across all 5 independently-acquired datasets (ARCX.PILLAR, XNAS.ITCH,
#: XNYS.PILLAR, DBEQ.BASIC, EQUS.SUMMARY), not inferred from memory or from
#: `symbology.resolve` (which only proves venue PRESENCE, not primary
#: listing -- every ticker resolves on every venue under Reg NMS).
PRIMARY_LISTING_EXCHANGE: dict[str, str] = {
    "SPY": "ARCX", "IWM": "ARCX", "GLD": "ARCX", "HYG": "ARCX", "LQD": "ARCX",
    "USO": "ARCX", "XLE": "ARCX", "XLF": "ARCX", "XLK": "ARCX", "XLU": "ARCX",
    "QQQ": "XNAS", "IEF": "XNAS", "SHY": "XNAS", "TLT": "XNAS",
}

#: Databento dataset code for each real exchange MIC used above.
EXCHANGE_TO_PRIMARY_DATASET: dict[str, str] = {
    "ARCX": "ARCX.PILLAR",
    "XNAS": "XNAS.ITCH",
}

#: (dataset, start, end) -- each window is that dataset's own real
#: pre-holdout availability, verified live via
#: scripts/databento_equity_capability_probe.py on 2026-09-22. Never derived
#: from datetime.now().
DATASET_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("ARCX.PILLAR", "2018-05-01", "2024-12-31"),
    ("XNAS.ITCH", "2018-05-01", "2024-12-31"),
    ("XNYS.PILLAR", "2018-05-01", "2024-12-31"),
    ("DBEQ.BASIC", "2023-03-28", "2024-12-31"),
    ("EQUS.SUMMARY", "2024-07-01", "2024-12-31"),
)

#: Real acquisition-time cost estimates (USD), for reference/audit only --
#: not authoritative (the spend ledger at
#: data/manifests/phase_6_etf/spend_ledger.json is authoritative).
ACQUIRED_TOTAL_COST_USD = 0.60532


def primary_listing_dataset(raw_symbol: str) -> str:
    """The Databento dataset code for `raw_symbol`'s real primary listing
    exchange. Raises KeyError for a ticker outside the pilot universe --
    never guesses a fallback."""
    exchange = PRIMARY_LISTING_EXCHANGE[raw_symbol]
    return EXCHANGE_TO_PRIMARY_DATASET[exchange]


def etf_dsl_root_symbol(raw_symbol: str) -> str:
    """The synthetic "virtual root" symbol used ONLY inside the StrategySpec
    DSL / target schedule / ContractSpec.root_symbol -- never in any
    human-facing label (registry root_symbol, dataset identity, friendly ids
    all use the real ticker).

    The reference CLI's target-schedule path requires a target's
    `root_symbol` to be DISTINCT from any real tradable `raw_symbol` in the
    registry (`quant_backtest_targets_csv`: "targets csv: root_symbol '...'
    is a real raw contract in the registry") -- a Futures root ("NQ") is
    always a virtual label distinct from its resolved contract ("NQZ6"). An
    ETF's `raw_symbol` IS its ticker (no roll, one instrument forever), so
    `root_symbol == raw_symbol` would collide with that check; this
    deterministic "E" prefix keeps root_symbol distinct while remaining
    within StrategySpec's `^[A-Z0-9]{1,12}$` pattern (no dot allowed, unlike
    a Databento continuous symbol)."""
    return f"E{raw_symbol}"
