"""Typed `MarketNewsItem` -- Market Intelligence Data Completion Pass,
Checkpoint E, Section 15. Current market information, deliberately a
DIFFERENT concept from `alpha_agent.knowledge.models.StrategyKnowledgeItem`
(`ResearchSource`) -- see the package docstring.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, model_validator


class NewsSourceType(str, Enum):
    """Section 19: official government feeds are the only AUTHORITATIVE,
    implemented connectors this checkpoint ships; a broad commercial
    provider is designed for but stays DISABLED/NOT_CONFIGURED."""

    OFFICIAL = "OFFICIAL"
    COMMERCIAL = "COMMERCIAL"


class NewsCategory(str, Enum):
    """Section 21's mapping categories, verbatim, plus OTHER for an item a
    connector returned that does not deterministically match any of them
    (never forced into a category it does not belong to)."""

    FOMC_POLICY = "FOMC_POLICY"
    CPI_PPI_EMPLOYMENT = "CPI_PPI_EMPLOYMENT"
    PETROLEUM = "PETROLEUM"
    NATURAL_GAS = "NATURAL_GAS"
    USDA_GRAIN_OILSEED = "USDA_GRAIN_OILSEED"
    OTHER = "OTHER"


class MarketNewsItem(BaseModel):
    """Section 15's required/optional field list, verbatim.
    ``published_at != retrieved_at`` is enforced at construction (Section
    15: "Never fabricate published_at") -- a connector that cannot determine
    a real published time for an item must not manufacture one by reusing
    the retrieval time."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "market-news-item/1"
    news_id: str
    headline: str
    source_name: str
    source_type: NewsSourceType
    source_url: str

    published_at: datetime
    retrieved_at: datetime

    related_products: tuple[str, ...] = ()
    related_asset_classes: tuple[str, ...] = ()

    category: NewsCategory
    mapping_reason: str

    summary: str | None = None
    event_type: str | None = None

    @model_validator(mode="after")
    def _published_before_or_at_retrieved(self) -> MarketNewsItem:
        if self.published_at == self.retrieved_at:
            raise ValueError("published_at must never equal retrieved_at -- never fabricate a published time")
        if self.published_at > self.retrieved_at:
            raise ValueError("published_at cannot be after retrieved_at")
        return self


__all__ = ["MarketNewsItem", "NewsCategory", "NewsSourceType"]
