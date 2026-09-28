"""News Alpha Phase G -- the RISK TAXONOMY portfolio constraints are written in.

``risk-classification/1`` names, for every instrument the platform can size,
an asset class and a sector: the buckets the allocator's asset-class and
sector gross caps apply to. It is an internal, reviewed risk taxonomy --
not GICS, not NAICS, and not the Phase C transmission taxonomy
(`news_alpha.signal_paths.EconomicSector`, which describes where an economic
consequence lands, never what a position holds).

Sectors are global names (``US_ENERGY_EQUITY`` is not ``ENERGY``): energy
equities and crude oil are different holdings even when a news event moves
both. What they share is measured by the correlation matrix, never assumed
here.

Futures roots come from the product catalog's own asset class
(`marketdata.product_catalog`), refined to a sector; ETFs from the pilot's
reviewed exposure labels (`etf.universe.EXPOSURE`). An instrument missing
from this table is UNCLASSIFIED and cannot be sized -- never guessed.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from alpha_agent.news_alpha.mandate import MandateDomain

__all__ = [
    "RISK_CLASSIFICATION_RULE",
    "RiskAssetClass",
    "RiskClassification",
    "classify_instrument",
]

RISK_CLASSIFICATION_RULE = "risk-classification/1"


class RiskAssetClass(str, Enum):
    EQUITY = "EQUITY"
    RATES = "RATES"
    CREDIT = "CREDIT"
    COMMODITY = "COMMODITY"
    FX = "FX"


class RiskClassification(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    asset_class: RiskAssetClass
    sector: str
    rule: str = RISK_CLASSIFICATION_RULE


_E, _R, _C, _M, _X = (RiskAssetClass.EQUITY, RiskAssetClass.RATES, RiskAssetClass.CREDIT,
                      RiskAssetClass.COMMODITY, RiskAssetClass.FX)

_FUTURES: dict[str, tuple[RiskAssetClass, str]] = {
    "ES": (_E, "US_BROAD_EQUITY"), "MES": (_E, "US_BROAD_EQUITY"),
    "YM": (_E, "US_BROAD_EQUITY"), "MYM": (_E, "US_BROAD_EQUITY"),
    "NQ": (_E, "US_GROWTH_EQUITY"), "MNQ": (_E, "US_GROWTH_EQUITY"),
    "RTY": (_E, "US_SMALL_CAP_EQUITY"), "M2K": (_E, "US_SMALL_CAP_EQUITY"),
    "CL": (_M, "ENERGY"), "MCL": (_M, "ENERGY"), "NG": (_M, "ENERGY"), "RB": (_M, "ENERGY"), "HO": (_M, "ENERGY"),
    "GC": (_M, "PRECIOUS_METALS"), "MGC": (_M, "PRECIOUS_METALS"), "SI": (_M, "PRECIOUS_METALS"),
    "HG": (_M, "INDUSTRIAL_METALS"),
    "ZC": (_M, "GRAINS"), "ZW": (_M, "GRAINS"),
    "ZS": (_M, "OILSEEDS"), "ZM": (_M, "OILSEEDS"), "ZL": (_M, "OILSEEDS"),
    "ZT": (_R, "US_TREASURY_SHORT"),
    "ZF": (_R, "US_TREASURY_INTERMEDIATE"), "ZN": (_R, "US_TREASURY_INTERMEDIATE"),
    "ZB": (_R, "US_TREASURY_LONG"), "UB": (_R, "US_TREASURY_LONG"),
    "6E": (_X, "FX_EUR"), "6J": (_X, "FX_JPY"), "6B": (_X, "FX_GBP"),
    "6A": (_X, "FX_AUD"), "6C": (_X, "FX_CAD"), "6S": (_X, "FX_CHF"),
}

_ETF: dict[str, tuple[RiskAssetClass, str]] = {
    "SPY": (_E, "US_BROAD_EQUITY"), "QQQ": (_E, "US_GROWTH_EQUITY"), "IWM": (_E, "US_SMALL_CAP_EQUITY"),
    "XLK": (_E, "US_TECHNOLOGY_EQUITY"), "XLF": (_E, "US_FINANCIALS_EQUITY"),
    "XLE": (_E, "US_ENERGY_EQUITY"), "XLU": (_E, "US_UTILITIES_EQUITY"),
    "SHY": (_R, "US_TREASURY_SHORT"), "IEF": (_R, "US_TREASURY_INTERMEDIATE"), "TLT": (_R, "US_TREASURY_LONG"),
    "LQD": (_C, "US_INVESTMENT_GRADE_CREDIT"), "HYG": (_C, "US_HIGH_YIELD_CREDIT"),
    "GLD": (_M, "PRECIOUS_METALS"), "USO": (_M, "ENERGY"),
}


def classify_instrument(domain: MandateDomain, symbol: str) -> RiskClassification | None:
    """The instrument's risk bucket, or ``None`` when it is unclassified (and
    therefore cannot be sized)."""
    table = {MandateDomain.FUTURES: _FUTURES, MandateDomain.ETF: _ETF}.get(domain, {})
    found = table.get(symbol)
    return None if found is None else RiskClassification(asset_class=found[0], sector=found[1])
