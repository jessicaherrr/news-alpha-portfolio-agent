"""News Alpha Phase A -- the NEWS / EVENT input to the impact scan.

There is ONE news/event store: `alpha_agent.market_intel` (`NewsStore` /
`EventStore`, observation plane). This module adds no second one. It defines:

* `UserDescribedEvent` -- an event the user typed ("OPEC+ announced a
  production cut"). The official connectors only cover Fed/BLS/EIA/USDA, so
  this is how a user raises anything else. It is session state only: never
  written to `NewsStore` (it is not a sourced news item) and never evidence.
* `ImpactEvent` -- a read-only VIEW over exactly one of `MarketNewsItem`,
  `ScheduledMarketEvent`, or `UserDescribedEvent`, normalizing the fields the
  scan reads. It carries the original object verbatim (``source``) so every
  downstream stage can cite what was actually read.

News is never scientific evidence: nothing here feeds the registry,
`FailureMemory`, validation, or any verdict.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem

__all__ = ["ImpactEvent", "ImpactEventKind", "ImpactSourceType", "UserDescribedEvent", "impact_event_from"]


class ImpactEventKind(str, Enum):
    MARKET_NEWS = "MARKET_NEWS"
    SCHEDULED_EVENT = "SCHEDULED_EVENT"
    USER_DESCRIBED = "USER_DESCRIBED"


class ImpactSourceType(str, Enum):
    OFFICIAL = "OFFICIAL"
    COMMERCIAL = "COMMERCIAL"
    USER = "USER"


class UserDescribedEvent(BaseModel):
    """Free text the user supplied about a market event. Its id is a content
    hash of (text, described_at) -- deterministic, never a random UUID."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "user-described-event/1"
    event_id: str
    text: str = Field(min_length=3, max_length=2000)
    described_at: datetime

    @model_validator(mode="after")
    def _utc(self) -> UserDescribedEvent:
        if self.described_at.tzinfo is None:
            raise ValueError("described_at must be UTC-aware -- never a naive datetime")
        return self

    @classmethod
    def create(cls, text: str, *, described_at: datetime | None = None) -> UserDescribedEvent:
        text = " ".join(text.split())
        at = described_at or datetime.now(UTC)
        digest = hashlib.sha256(f"{text}\n{at.isoformat()}".encode()).hexdigest()[:16]
        return cls(event_id=f"user-{digest}", text=text, described_at=at)


class ImpactEvent(BaseModel):
    """Normalized, read-only view of one news/event item."""

    model_config = {"frozen": True, "extra": "forbid"}

    event_id: str
    kind: ImpactEventKind
    headline: str
    summary: str | None = None
    source_name: str
    source_type: ImpactSourceType
    #: The market_intel category the item's own connector assigned
    #: (`NewsCategory` value / event category string), or ``None`` for a
    #: user-described event, which has no sourced category.
    source_category: str | None = None
    #: Products the item's own connector mapped it to -- preserved so the
    #: translation hand-off can reuse the existing constructors verbatim.
    source_mapped_products: tuple[str, ...] = ()
    observed_at: datetime
    evidence_refs: tuple[str, ...] = ()
    source: MarketNewsItem | ScheduledMarketEvent | UserDescribedEvent

    @property
    def text(self) -> str:
        """Everything the lexicon may read: headline plus summary."""
        return f"{self.headline}. {self.summary}" if self.summary else self.headline


def impact_event_from(item: MarketNewsItem | ScheduledMarketEvent | UserDescribedEvent) -> ImpactEvent:
    if isinstance(item, MarketNewsItem):
        return ImpactEvent(
            event_id=item.news_id, kind=ImpactEventKind.MARKET_NEWS, headline=item.headline, summary=item.summary,
            source_name=item.source_name, source_type=ImpactSourceType(item.source_type.value),
            source_category=item.category.value, source_mapped_products=item.related_products,
            observed_at=item.published_at,
            evidence_refs=(f"news_id={item.news_id}", f"source_url={item.source_url}", f"category={item.category.value}"),
            source=item,
        )
    if isinstance(item, ScheduledMarketEvent):
        return ImpactEvent(
            event_id=item.event_id, kind=ImpactEventKind.SCHEDULED_EVENT, headline=item.name, summary=None,
            source_name=item.source_name, source_type=ImpactSourceType.OFFICIAL,
            source_category=item.category, source_mapped_products=item.affected_products,
            observed_at=item.retrieved_at,
            evidence_refs=(
                f"event_id={item.event_id}", f"source_url={item.source_url}", f"category={item.category}",
                f"scheduled_at={item.scheduled_at.isoformat()}",
            ),
            source=item,
        )
    if isinstance(item, UserDescribedEvent):
        return ImpactEvent(
            event_id=item.event_id, kind=ImpactEventKind.USER_DESCRIBED, headline=item.text, summary=None,
            source_name="User-described event", source_type=ImpactSourceType.USER, source_category=None,
            observed_at=item.described_at, evidence_refs=(f"user_event_id={item.event_id}",), source=item,
        )
    raise TypeError(f"unsupported event type {type(item).__name__}")
