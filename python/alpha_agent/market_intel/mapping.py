"""Deterministic MARKET NEWS product mapping -- Section 21. No LLM decides a
mapping in this checkpoint; every rule is a fixed, inspectable table with an
explicit ``mapping_rule_id`` and human-readable ``mapping_reason``. Shared by
every news connector so a category always maps to the same products
everywhere it appears (also reused for scheduled-event mapping, Checkpoint F
-- the products a FOMC decision affects are the same products a CPI print
affects, and both are macro categories).
"""
from __future__ import annotations

from alpha_agent.market_intel.news_schemas import NewsCategory

#: Section 21's worked mapping table, verbatim category -> affected roots.
#: FX is expressed as the actual catalogued 6-letter futures roots (mission
#: Part 4's FX group), never the word "FX" standing in for an unresolved set.
_FX_ROOTS: tuple[str, ...] = ("6E", "6J", "6B", "6A", "6C", "6S")
_RATES_ROOTS: tuple[str, ...] = ("ZT", "ZF", "ZN", "ZB", "UB")
_EQUITY_INDEX_ROOTS: tuple[str, ...] = ("ES", "NQ", "YM", "RTY")

CATEGORY_PRODUCTS: dict[NewsCategory, tuple[str, ...]] = {
    NewsCategory.FOMC_POLICY: (*_EQUITY_INDEX_ROOTS, *_RATES_ROOTS, *_FX_ROOTS, "GC"),
    NewsCategory.CPI_PPI_EMPLOYMENT: (*_EQUITY_INDEX_ROOTS, *_RATES_ROOTS, *_FX_ROOTS, "GC"),
    NewsCategory.PETROLEUM: ("CL", "MCL", "RB", "HO"),
    NewsCategory.NATURAL_GAS: ("NG",),
    NewsCategory.USDA_GRAIN_OILSEED: ("ZC", "ZW", "ZS", "ZM", "ZL"),
    NewsCategory.OTHER: (),
}

CATEGORY_ASSET_CLASSES: dict[NewsCategory, tuple[str, ...]] = {
    NewsCategory.FOMC_POLICY: ("EQUITY_INDEX", "RATES", "FX", "METALS"),
    NewsCategory.CPI_PPI_EMPLOYMENT: ("EQUITY_INDEX", "RATES", "FX", "METALS"),
    NewsCategory.PETROLEUM: ("ENERGY",),
    NewsCategory.NATURAL_GAS: ("ENERGY",),
    NewsCategory.USDA_GRAIN_OILSEED: ("AGRICULTURE",),
    NewsCategory.OTHER: (),
}

#: A stable id per category-mapping rule -- inspectable provenance for "why
#: was this item mapped to these products" (Section 21: "Store: mapping_rule_id,
#: mapping_reason").
CATEGORY_MAPPING_RULE_ID: dict[NewsCategory, str] = {
    NewsCategory.FOMC_POLICY: "fomc-policy-macro/1",
    NewsCategory.CPI_PPI_EMPLOYMENT: "cpi-ppi-employment-macro/1",
    NewsCategory.PETROLEUM: "eia-petroleum/1",
    NewsCategory.NATURAL_GAS: "eia-natural-gas/1",
    NewsCategory.USDA_GRAIN_OILSEED: "usda-grain-oilseed/1",
    NewsCategory.OTHER: "uncategorized/1",
}

_CATEGORY_REASON: dict[NewsCategory, str] = {
    NewsCategory.FOMC_POLICY: (
        "FOMC monetary-policy actions move broad equity index, U.S. Treasury futures, G10 FX, and gold pricing."
    ),
    NewsCategory.CPI_PPI_EMPLOYMENT: (
        "CPI/PPI/employment releases are macro inputs to Fed policy expectations -- same downstream products as FOMC."
    ),
    NewsCategory.PETROLEUM: "EIA petroleum releases are the official weekly U.S. crude/refined-product supply data.",
    NewsCategory.NATURAL_GAS: "EIA natural gas storage releases are the official weekly U.S. storage data for NG.",
    NewsCategory.USDA_GRAIN_OILSEED: "USDA WASDE/crop reports are the official U.S. grain/oilseed supply-demand data.",
    NewsCategory.OTHER: "No deterministic category rule matched this item -- no product mapping is claimed.",
}


def products_for_category(category: NewsCategory) -> tuple[str, ...]:
    return CATEGORY_PRODUCTS.get(category, ())


def asset_classes_for_category(category: NewsCategory) -> tuple[str, ...]:
    return CATEGORY_ASSET_CLASSES.get(category, ())


def mapping_reason_for_category(category: NewsCategory) -> str:
    return _CATEGORY_REASON.get(category, "No deterministic category rule matched this item.")


def mapping_rule_id_for_category(category: NewsCategory) -> str:
    return CATEGORY_MAPPING_RULE_ID.get(category, "uncategorized/1")


__all__ = [
    "CATEGORY_ASSET_CLASSES",
    "CATEGORY_MAPPING_RULE_ID",
    "CATEGORY_PRODUCTS",
    "asset_classes_for_category",
    "mapping_reason_for_category",
    "mapping_rule_id_for_category",
    "products_for_category",
]
