"""The extensible futures PRODUCT CATALOG -- Market Intelligence + Futures
Universe campaign, Parts 2-4.

This module is pure, static, offline data: no network call, no registry
read, no Databento client. It exists to answer "what futures products does
this platform know about at all" -- the MARKET UNIVERSE -- which is
deliberately a much broader and looser concept than the RESEARCH UNIVERSE
(CLAUDE.md's certified ES/NQ/CL/GC/ZN) or the TRADING UNIVERSE (whatever a
strategy has actually been validated and made paper-eligible for). See the
module docstring of `alpha_agent.ui.market_universe` for how those three
concepts are composed together for the UI; this module only ever answers
"what is CATALOGUED", never "what is DATA VERIFIED" (mission Part 4) -- that
distinction is load-bearing and must never collapse into a single boolean.

Every root below is a real CME Globex product (dataset ``GLBX.MDP3`` covers
CME, CBOT, NYMEX and COMEX, all on one Globex platform) -- chosen from
mission Part 4's own worked category lists, not invented. Being catalogued
here is NOT a claim that Databento's continuous-front-month proxy actually
resolves for a given root today; that is a runtime fact, checked live and
reported honestly by `alpha_agent.ui.market_universe.capability_for`, never
assumed from this static list (mission Part 4: "Do not blindly hard-code
contracts that Databento cannot actually resolve").

CLAUDE.md Part 38 ("current core research universe must not expand
silently"): `RESEARCH_UNIVERSE` here is a re-export of
`alpha_agent.marketdata.contracts.APPROVED_ROOTS` -- the SAME single frozen
constant, never a second independently-typed list that could drift from it
(see `test_market_product_catalog.py::test_research_universe_matches_the_one_frozen_constant`).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from alpha_agent.marketdata.contracts import APPROVED_ROOTS


class AssetClass(str, Enum):
    """Mission Part 4's seven initial catalog categories."""

    EQUITY_INDEX = "EQUITY_INDEX"
    ENERGY = "ENERGY"
    METALS = "METALS"
    RATES = "RATES"
    FX = "FX"
    AGRICULTURE = "AGRICULTURE"
    CRYPTO = "CRYPTO"


#: Human-readable category labels, in the display order mission Part 5's
#: filter pills use ("All" is a UI-only synthetic option, not a member here).
ASSET_CLASS_LABELS: dict[AssetClass, str] = {
    AssetClass.EQUITY_INDEX: "Equity Index",
    AssetClass.ENERGY: "Energy",
    AssetClass.METALS: "Metals",
    AssetClass.RATES: "Rates",
    AssetClass.FX: "FX",
    AssetClass.AGRICULTURE: "Agriculture",
    AssetClass.CRYPTO: "Crypto",
}


class ProductCatalogEntry(BaseModel):
    """One CATALOGUED futures product -- static metadata only. Never carries
    a price, a capability flag, or anything else that could go stale; see
    `alpha_agent.marketdata.capability.FuturesProductCapability` for the
    typed runtime-status object this entry feeds into."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "futures-product-catalog-entry/1"
    root_symbol: str
    display_name: str
    asset_class: AssetClass
    #: The exchange this root actually lists on (CME / CBOT / NYMEX / COMEX)
    #: -- distinct from `dataset`, which is the single Databento feed
    #: (GLBX.MDP3) that carries all four.
    venue: str
    dataset: str = "GLBX.MDP3"
    #: Databento continuous front-month display proxy convention this
    #: platform already uses for the research universe
    #: (`alpha_agent.marketdata.databento_provider.DISPLAY_SYMBOL`) --
    #: carried here so a broader provider wiring (mission Part 4/41) has a
    #: single, catalog-derived source for it rather than a second hardcoded
    #: `f"{root}.v.0"` literal.
    continuous_display_symbol: str


def _entry(root: str, display_name: str, asset_class: AssetClass, venue: str) -> ProductCatalogEntry:
    return ProductCatalogEntry(
        root_symbol=root, display_name=display_name, asset_class=asset_class, venue=venue,
        continuous_display_symbol=f"{root}.v.0",
    )


#: The MARKET UNIVERSE (mission Part 2/4) -- every product this platform
#: knows about for OBSERVATION purposes. Declaration order follows mission
#: Part 4's own category listing; `market_universe_roots()` below returns a
#: deterministic sorted view for callers that do not care about that order.
PRODUCT_CATALOG: tuple[ProductCatalogEntry, ...] = (
    # -- Equity Index (CME) ------------------------------------------------
    _entry("ES", "E-mini S&P 500", AssetClass.EQUITY_INDEX, "CME"),
    _entry("MES", "Micro E-mini S&P 500", AssetClass.EQUITY_INDEX, "CME"),
    _entry("NQ", "E-mini Nasdaq-100", AssetClass.EQUITY_INDEX, "CME"),
    _entry("MNQ", "Micro E-mini Nasdaq-100", AssetClass.EQUITY_INDEX, "CME"),
    _entry("YM", "E-mini Dow ($5)", AssetClass.EQUITY_INDEX, "CBOT"),
    _entry("MYM", "Micro E-mini Dow", AssetClass.EQUITY_INDEX, "CBOT"),
    _entry("RTY", "E-mini Russell 2000", AssetClass.EQUITY_INDEX, "CME"),
    _entry("M2K", "Micro E-mini Russell 2000", AssetClass.EQUITY_INDEX, "CME"),
    # -- Energy (NYMEX) ------------------------------------------------------
    _entry("CL", "WTI Crude Oil", AssetClass.ENERGY, "NYMEX"),
    _entry("MCL", "Micro WTI Crude Oil", AssetClass.ENERGY, "NYMEX"),
    _entry("NG", "Henry Hub Natural Gas", AssetClass.ENERGY, "NYMEX"),
    _entry("RB", "RBOB Gasoline", AssetClass.ENERGY, "NYMEX"),
    _entry("HO", "NY Harbor ULSD (Heating Oil)", AssetClass.ENERGY, "NYMEX"),
    # -- Metals (COMEX) -------------------------------------------------------
    _entry("GC", "Gold", AssetClass.METALS, "COMEX"),
    _entry("MGC", "Micro Gold", AssetClass.METALS, "COMEX"),
    _entry("SI", "Silver", AssetClass.METALS, "COMEX"),
    _entry("HG", "Copper", AssetClass.METALS, "COMEX"),
    # -- Interest Rates (CBOT) -------------------------------------------------
    _entry("ZT", "2-Year T-Note", AssetClass.RATES, "CBOT"),
    _entry("ZF", "5-Year T-Note", AssetClass.RATES, "CBOT"),
    _entry("ZN", "10-Year T-Note", AssetClass.RATES, "CBOT"),
    _entry("ZB", "30-Year T-Bond", AssetClass.RATES, "CBOT"),
    _entry("UB", "Ultra T-Bond", AssetClass.RATES, "CBOT"),
    # -- FX (CME) ---------------------------------------------------------
    _entry("6E", "Euro FX", AssetClass.FX, "CME"),
    _entry("6J", "Japanese Yen", AssetClass.FX, "CME"),
    _entry("6B", "British Pound", AssetClass.FX, "CME"),
    _entry("6A", "Australian Dollar", AssetClass.FX, "CME"),
    _entry("6C", "Canadian Dollar", AssetClass.FX, "CME"),
    _entry("6S", "Swiss Franc", AssetClass.FX, "CME"),
    # -- Agriculture (CBOT) ---------------------------------------------------
    _entry("ZC", "Corn", AssetClass.AGRICULTURE, "CBOT"),
    _entry("ZW", "Wheat (Chicago SRW)", AssetClass.AGRICULTURE, "CBOT"),
    _entry("ZS", "Soybeans", AssetClass.AGRICULTURE, "CBOT"),
    _entry("ZM", "Soybean Meal", AssetClass.AGRICULTURE, "CBOT"),
    _entry("ZL", "Soybean Oil", AssetClass.AGRICULTURE, "CBOT"),
    # -- Crypto Futures (CME) -------------------------------------------------
    _entry("BTC", "Bitcoin", AssetClass.CRYPTO, "CME"),
    _entry("MBT", "Micro Bitcoin", AssetClass.CRYPTO, "CME"),
    _entry("ETH", "Ether", AssetClass.CRYPTO, "CME"),
)


def _validate_catalog(catalog: tuple[ProductCatalogEntry, ...]) -> None:
    seen: set[str] = set()
    for entry in catalog:
        if entry.root_symbol in seen:
            raise ValueError(f"duplicate root_symbol in PRODUCT_CATALOG: {entry.root_symbol!r}")
        seen.add(entry.root_symbol)


_validate_catalog(PRODUCT_CATALOG)

#: RESEARCH UNIVERSE (mission Part 2) -- the certified scientific research
#: roots. A re-export, deliberately not a redeclaration -- see module
#: docstring.
RESEARCH_UNIVERSE: tuple[str, ...] = APPROVED_ROOTS

_BY_ROOT: dict[str, ProductCatalogEntry] = {e.root_symbol: e for e in PRODUCT_CATALOG}


def catalog_entry(root_symbol: str | None) -> ProductCatalogEntry | None:
    if not root_symbol:
        return None
    return _BY_ROOT.get(root_symbol.upper())


def is_catalogued(root_symbol: str | None) -> bool:
    return catalog_entry(root_symbol) is not None


def market_universe_roots() -> tuple[str, ...]:
    """Every catalogued root, sorted -- the deterministic MARKET UNIVERSE
    membership list (mission Part 2)."""
    return tuple(sorted(_BY_ROOT))


def catalog_by_asset_class(asset_class: AssetClass) -> tuple[ProductCatalogEntry, ...]:
    return tuple(e for e in PRODUCT_CATALOG if e.asset_class is asset_class)


def catalog_ordered_by_category() -> tuple[ProductCatalogEntry, ...]:
    """`PRODUCT_CATALOG` is already declared in this order; this accessor
    exists so callers state the dependency explicitly rather than relying on
    tuple declaration order never changing silently."""
    return PRODUCT_CATALOG


__all__ = [
    "ASSET_CLASS_LABELS",
    "PRODUCT_CATALOG",
    "RESEARCH_UNIVERSE",
    "AssetClass",
    "ProductCatalogEntry",
    "catalog_by_asset_class",
    "catalog_entry",
    "catalog_ordered_by_category",
    "is_catalogued",
    "market_universe_roots",
]
