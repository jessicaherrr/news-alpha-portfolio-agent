"""Typed `ScheduledMarketEvent` -- Market Intelligence Data Completion
Pass, Checkpoint F, Section 23.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, model_validator


class EventImportance(str, Enum):
    """Section 25 -- deterministic and inspectable, never LLM-invented."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ScheduledMarketEvent(BaseModel):
    """Section 23's required/optional field list, verbatim.
    ``actual_release_at`` stays `None` until a REAL release timestamp is
    observed (Section 23: "Do not fabricate actual-release timestamps")."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "scheduled-market-event/1"
    event_id: str
    name: str
    source_name: str
    source_url: str

    scheduled_at: datetime  # UTC-aware internally (Section 26)
    timezone: str  # display timezone label, e.g. "America/New_York"

    category: str
    importance: EventImportance
    importance_rule: str

    affected_products: tuple[str, ...] = ()
    mapping_reason: str

    retrieved_at: datetime
    actual_release_at: datetime | None = None

    @model_validator(mode="after")
    def _scheduled_at_is_utc_aware(self) -> ScheduledMarketEvent:
        if self.scheduled_at.tzinfo is None:
            raise ValueError("scheduled_at must be UTC-aware -- never a naive datetime (Section 26)")
        if self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be UTC-aware -- never a naive datetime")
        if self.actual_release_at is not None and self.actual_release_at.tzinfo is None:
            raise ValueError("actual_release_at must be UTC-aware -- never a naive datetime")
        return self


__all__ = ["EventImportance", "ScheduledMarketEvent"]
