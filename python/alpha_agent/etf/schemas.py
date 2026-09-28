"""Typed ETF corporate-action, data-source-provenance, and coverage schemas.

Three Phase 6 instructions this module exists to satisfy structurally, not by
prose convention:

1. "Corporate-action records must come from a real attributable source with
   provenance. Do not infer actions from price jumps." -- every
   :class:`EtfSplitAction` / :class:`EtfCashDistribution` REQUIRES a
   :class:`CorporateActionSource` citation; there is no constructor path that
   omits one.
2. "Its provenance must explicitly say that it is a primary-listing-venue
   dataset... Do not use primary-venue volume as a proxy for whole-market
   volume." -- :class:`EtfDataSourceProvenance` refuses to be constructed with
   ``volume_is_whole_market=True`` for any single-venue role; only
   ``CONSOLIDATED_EOD_SUMMARY`` (the one Phase 6 dataset -- EQUS.SUMMARY --
   empirically confirmed to be a real single pre-aggregated bar per
   instrument/day, see the Phase 6 roadmap-state audit) may claim it, and even
   then only explicitly, never by default.
3. An empty distributions list must never be silently read as "this ETF paid
   no distributions" -- :class:`CorporateActionCoverageStatus` makes the
   sourcing state of every ticker an explicit, typed fact instead of an
   inferred one.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "CorporateActionCoverage",
    "CorporateActionCoverageStatus",
    "CorporateActionSource",
    "EtfCashDistribution",
    "EtfDataSourceProvenance",
    "EtfDataSourceRole",
    "EtfSplitAction",
]


class CorporateActionSource(BaseModel):
    """A real, attributable citation. Required on every corporate-action
    record -- never optional, never a default."""

    model_config = {"frozen": True, "extra": "forbid"}

    name: str = Field(min_length=1)          # e.g. "SEC EDGAR Form 8-K, United States Oil Fund LP (CIK 1327068)"
    url: str = Field(min_length=1)
    retrieved_at: datetime


class EtfSplitAction(BaseModel):
    """A real share split (or reverse split), sourced from an attributable
    record -- the Python-side typed mirror of the C++
    ``quant::SplitAction`` this feeds (``cpp/include/quant_core/
    corporate_actions.hpp``), plus the provenance the C++ layer has no
    concept of."""

    model_config = {"frozen": True, "extra": "forbid"}

    raw_symbol: str = Field(min_length=1)
    effective_date: date
    ratio: float                              # new_units = old_units * ratio; 0.125 == 1-for-8 reverse
    source: CorporateActionSource

    @model_validator(mode="after")
    def _check_ratio(self) -> EtfSplitAction:
        if self.ratio <= 0 or self.ratio == 1.0:
            raise ValueError(f"split ratio must be > 0 and != 1.0, got {self.ratio}")
        return self


class EtfCashDistribution(BaseModel):
    """A real cash distribution (dividend), sourced from an attributable
    record -- the Python-side typed mirror of the C++
    ``quant::CashDistribution``, plus provenance."""

    model_config = {"frozen": True, "extra": "forbid"}

    raw_symbol: str = Field(min_length=1)
    ex_date: date
    pay_date: date
    amount_per_share_usd: float
    source: CorporateActionSource

    @model_validator(mode="after")
    def _check_dates_and_amount(self) -> EtfCashDistribution:
        if self.pay_date < self.ex_date:
            raise ValueError(
                f"pay_date {self.pay_date} is before ex_date {self.ex_date}"
            )
        if self.amount_per_share_usd <= 0:
            raise ValueError(
                f"amount_per_share_usd must be > 0, got {self.amount_per_share_usd}"
            )
        return self


class CorporateActionCoverageStatus(str, Enum):
    """How a ticker's corporate-action history was established -- an
    explicit, typed honesty signal, never left implicit."""

    #: A real record exists, sourced from an official/attributable filing.
    SOURCED_FROM_OFFICIAL_RECORD = "SOURCED_FROM_OFFICIAL_RECORD"
    #: A heuristic screen (e.g. a day-over-day price-discontinuity scan) found
    #: no candidate event -- weak evidence of absence, NOT an official
    #: confirmation. Never treated as equivalent to SOURCED.
    SCREENED_NO_EVENT_DETECTED = "SCREENED_NO_EVENT_DETECTED"
    #: No investigation has populated this ticker's history at all. An empty
    #: action list under this status must never be read as "no actions
    #: occurred."
    NOT_YET_INVESTIGATED = "NOT_YET_INVESTIGATED"


class CorporateActionCoverage(BaseModel):
    """Per-ticker, per-action-type sourcing honesty record."""

    model_config = {"frozen": True, "extra": "forbid"}

    raw_symbol: str = Field(min_length=1)
    splits_status: CorporateActionCoverageStatus
    splits_note: str
    distributions_status: CorporateActionCoverageStatus
    distributions_note: str


class EtfDataSourceRole(str, Enum):
    """What a Databento product actually IS, empirically -- not what it is
    conventionally assumed to be. Set from the real Phase 6 acquisition
    audit (``docs/AGENTIC_ALPHA_ROADMAP_STATE.md``'s Phase 6 row), never
    guessed."""

    #: A single exchange's own OHLCV prints for a ticker that IS primary-
    #: listed there (real evidence: the `exchange` field, byte-identical
    #: across all 5 acquired datasets' definition records).
    PRIMARY_LISTING_VENUE = "PRIMARY_LISTING_VENUE"
    #: A single exchange's own OHLCV prints for a ticker that is NOT primary-
    #: listed there.
    OTHER_SINGLE_VENUE = "OTHER_SINGLE_VENUE"
    #: EQUS.SUMMARY -- empirically confirmed (Phase 6 audit) to be one real
    #: pre-aggregated bar per instrument/day (0% duplicate rows, single
    #: publisher_id). The only role permitted to claim whole-market volume.
    CONSOLIDATED_EOD_SUMMARY = "CONSOLIDATED_EOD_SUMMARY"
    #: DBEQ.BASIC as acquired -- empirically confirmed (Phase 6 audit) to be
    #: 4 distinct publisher_id components per instrument/day (71.4%
    #: duplicate (instrument, ts) rate), NOT one consolidated bar. Must not
    #: be read as a usable price/volume series without a real cross-venue
    #: aggregation methodology, which Phase 6 explicitly declined to invent
    #: ad hoc.
    RAW_MULTI_PUBLISHER_UNAGGREGATED = "RAW_MULTI_PUBLISHER_UNAGGREGATED"


class EtfDataSourceProvenance(BaseModel):
    """Attached to every loaded ETF bar series. Structurally forbids the
    exact mislabeling Phase 6 instruction 1 called out: "Do not call this:
    consolidated / SIP truth / whole-market volume / authoritative market
    close" for a primary-listing-venue series."""

    model_config = {"frozen": True, "extra": "forbid"}

    dataset: str = Field(min_length=1)               # e.g. "ARCX.PILLAR"
    role: EtfDataSourceRole
    listing_exchange: str | None = None               # e.g. "ARCX" -- from the real definition record's `exchange` field
    volume_is_whole_market: bool = False
    #: Measured Phase 6 divergence evidence, carried forward so a consumer
    #: never has to re-derive "how much should I trust this" from scratch.
    #: None only for CONSOLIDATED_EOD_SUMMARY (it IS the reference the
    #: divergence was measured against).
    measured_mean_close_divergence_bps: float | None = None
    measured_mean_volume_capture_fraction: float | None = None

    @model_validator(mode="after")
    def _check_volume_claim(self) -> EtfDataSourceProvenance:
        if self.role != EtfDataSourceRole.CONSOLIDATED_EOD_SUMMARY and self.volume_is_whole_market:
            raise ValueError(
                f"{self.role.value} must never claim volume_is_whole_market=True -- "
                "measured Phase 6 evidence: a single-venue feed captures a mean of "
                "only ~27% of EQUS.SUMMARY's true consolidated volume "
                "(range ~20-34% across the pilot universe)"
            )
        return self
