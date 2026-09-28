"""Typed Equities corporate-action, data-source-provenance, point-in-time
universe-membership, and earnings-timestamp schemas -- Phase 9.1's mirror of
:mod:`alpha_agent.etf.schemas`, plus the genuinely new PIT-integrity concepts
the approved Phase 9 proposal flagged as "zero hits anywhere in the repo" for
either Futures or ETF: survivorship/delisting and earnings-timestamp honesty.

Deliberately a SEPARATE module from ``alpha_agent.etf.schemas`` rather than a
shared import, even though several types (``CorporateActionSource``,
``CorporateActionCoverageStatus``) would be byte-identical -- structural
domain isolation (goal 2 of the Phase 9.1 prompt: "future Equity experiments...
remain structurally isolated from FUTURES and ETF evidence") is easier to
audit and impossible to accidentally weaken when nothing here imports from
``alpha_agent.etf`` at all.

Four Phase 9.1 instructions this module exists to satisfy structurally, not
by prose convention:

1. "Corporate-action records must come from a real attributable source with
   provenance. Do not infer actions from price jumps." -- every
   :class:`EquitySplitAction` / :class:`EquityCashDistribution` REQUIRES a
   :class:`CorporateActionSource` citation; there is no constructor path that
   omits one. (Same rule as Phase 6 ETF.)
2. "Survivorship-bias protection... delisting handling." -- an empty
   :class:`EquityUniverseMembership` list must never be silently read as
   "this ticker never delisted" -- ``status`` is an explicit, typed fact, and
   a ``DELISTED`` status structurally REQUIRES a populated
   :class:`DelistingEvent` (never a bare enum flip with no evidence).
3. "Earnings/event timestamps... never inferred." -- :class:`EquityEarningsEvent`
   uses the same ``point_in_time_available: bool | None`` honesty pattern
   Phase 1's ``MeasurableVariable`` already established: ``None`` (UNKNOWN)
   unless a typed repository capability actually confirms it.
4. "Point-in-time fundamentals only if actually available" -- no fundamentals
   schema is defined here at all (Phase 9.1 investigated and found none
   verified free/self-serve; see ``alpha_agent.equities.data_availability``)
   rather than a half-built, never-populated placeholder type.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "CorporateActionCoverage",
    "CorporateActionCoverageStatus",
    "CorporateActionSource",
    "DelistingEvent",
    "DelistingReason",
    "EarningsTiming",
    "EquityCashDistribution",
    "EquityDataSourceProvenance",
    "EquityDataSourceRole",
    "EquityEarningsEvent",
    "EquitySplitAction",
    "EquityUniverseMembership",
    "UniverseMembershipStatus",
]


class CorporateActionSource(BaseModel):
    """A real, attributable citation. Required on every corporate-action /
    delisting / earnings-timestamp record here -- never optional, never a
    default."""

    model_config = {"frozen": True, "extra": "forbid"}

    name: str = Field(min_length=1)          # e.g. "SEC EDGAR Form 8-K, Apple Inc. (CIK 0000320193)"
    url: str = Field(min_length=1)
    retrieved_at: datetime


class EquitySplitAction(BaseModel):
    """A real share split (or reverse split), sourced from an attributable
    record -- the Python-side typed mirror of the C++ ``quant::SplitAction``
    this feeds (``cpp/include/quant_core/corporate_actions.hpp``, reused here
    completely unchanged), plus the provenance the C++ layer has no concept
    of."""

    model_config = {"frozen": True, "extra": "forbid"}

    raw_symbol: str = Field(min_length=1)
    effective_date: date
    ratio: float                              # new_units = old_units * ratio; 4.0 == 4-for-1
    source: CorporateActionSource

    @model_validator(mode="after")
    def _check_ratio(self) -> EquitySplitAction:
        if self.ratio <= 0 or self.ratio == 1.0:
            raise ValueError(f"split ratio must be > 0 and != 1.0, got {self.ratio}")
        return self


class EquityCashDistribution(BaseModel):
    """A real cash distribution (dividend), sourced from an attributable
    record -- the Python-side typed mirror of the C++ ``quant::CashDistribution``,
    plus provenance."""

    model_config = {"frozen": True, "extra": "forbid"}

    raw_symbol: str = Field(min_length=1)
    ex_date: date
    pay_date: date
    amount_per_share_usd: float
    source: CorporateActionSource

    @model_validator(mode="after")
    def _check_dates_and_amount(self) -> EquityCashDistribution:
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


class EquityDataSourceRole(str, Enum):
    """What a Databento product actually IS, empirically -- not what it is
    conventionally assumed to be. Same taxonomy as
    ``alpha_agent.etf.schemas.EtfDataSourceRole`` (the underlying Databento US
    equities dataset catalog is identical for a single-name equity and an
    ETF); Phase 9.1 has not yet run the acquisition-time audit the ETF pilot
    did, so this taxonomy is declared/expected, not yet re-verified against a
    real definition record for THIS universe."""

    #: A single exchange's own OHLCV prints for a ticker that IS primary-
    #: listed there.
    PRIMARY_LISTING_VENUE = "PRIMARY_LISTING_VENUE"
    #: A single exchange's own OHLCV prints for a ticker that is NOT primary-
    #: listed there.
    OTHER_SINGLE_VENUE = "OTHER_SINGLE_VENUE"
    #: EQUS.SUMMARY -- confirmed by the Phase 6 audit (for the ETF universe)
    #: to be one real pre-aggregated bar per instrument/day. The only role
    #: permitted to claim whole-market volume.
    CONSOLIDATED_EOD_SUMMARY = "CONSOLIDATED_EOD_SUMMARY"
    #: DBEQ.BASIC as acquired for the ETF pilot -- confirmed NOT to be one
    #: consolidated bar (4 distinct publisher_id components/day). Must not be
    #: read as a usable price/volume series without a real cross-venue
    #: aggregation methodology.
    RAW_MULTI_PUBLISHER_UNAGGREGATED = "RAW_MULTI_PUBLISHER_UNAGGREGATED"


class EquityDataSourceProvenance(BaseModel):
    """Attached to every loaded Equity bar series once acquired. Structurally
    forbids the exact mislabeling Phase 6 instruction 1 called out for ETFs:
    "Do not call this: consolidated / SIP truth / whole-market volume /
    authoritative market close" for a primary-listing-venue series."""

    model_config = {"frozen": True, "extra": "forbid"}

    dataset: str = Field(min_length=1)               # e.g. "XNAS.ITCH"
    role: EquityDataSourceRole
    listing_exchange: str | None = None               # e.g. "XNAS" -- from the real definition record's `exchange` field, once acquired
    volume_is_whole_market: bool = False
    #: Measured divergence evidence, once a real cross-source comparison
    #: exists for THIS universe. None until that comparison has been run --
    #: unlike the ETF pilot's carried-forward Phase 6 measurement, Phase 9.1
    #: has not acquired any data yet, so there is nothing real to carry.
    measured_mean_close_divergence_bps: float | None = None
    measured_mean_volume_capture_fraction: float | None = None

    @model_validator(mode="after")
    def _check_volume_claim(self) -> EquityDataSourceProvenance:
        if self.role != EquityDataSourceRole.CONSOLIDATED_EOD_SUMMARY and self.volume_is_whole_market:
            raise ValueError(
                f"{self.role.value} must never claim volume_is_whole_market=True -- "
                "a single-venue feed does not capture whole-market volume "
                "(measured for the ETF pilot's universe at ~27% mean capture; "
                "not yet re-measured for this universe, but the structural "
                "rule holds regardless of the exact figure)"
            )
        return self


# ---------------------------------------------------------------------------
# Point-in-time universe membership / survivorship / delisting -- genuinely
# new for this repo (the approved Phase 9 proposal found "zero hits anywhere"
# for this concept in either Futures or ETF).
# ---------------------------------------------------------------------------


class DelistingReason(str, Enum):
    """Why a name left the universe -- typed, never a free-text guess."""

    ACQUIRED_OR_MERGED = "ACQUIRED_OR_MERGED"
    BANKRUPTCY = "BANKRUPTCY"
    VOLUNTARY_DELISTING = "VOLUNTARY_DELISTING"
    EXCHANGE_COMPELLED_DELISTING = "EXCHANGE_COMPELLED_DELISTING"
    OTHER = "OTHER"


class DelistingEvent(BaseModel):
    """A real, attributable delisting record. Required whenever
    :class:`EquityUniverseMembership.status` is ``DELISTED`` -- never a bare
    status flip with no evidence, and never inferred from a price series
    (e.g. a name's bars simply stopping) without a real citation."""

    model_config = {"frozen": True, "extra": "forbid"}

    raw_symbol: str = Field(min_length=1)
    last_trade_date: date
    delisting_date: date
    reason: DelistingReason
    source: CorporateActionSource

    @model_validator(mode="after")
    def _check_dates(self) -> DelistingEvent:
        if self.delisting_date < self.last_trade_date:
            raise ValueError(
                f"delisting_date {self.delisting_date} is before last_trade_date {self.last_trade_date}"
            )
        return self


class UniverseMembershipStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DELISTED = "DELISTED"


class EquityUniverseMembership(BaseModel):
    """The Phase 9.1 minimum-viable point-in-time membership record: a fixed,
    dated universe (``alpha_agent.equities.universe.EQUITY_UNIVERSE``,
    declared at ``UNIVERSE_DECLARED_AT``) plus an explicit typed status per
    name. This is NOT full point-in-time index reconstruction (out of scope
    per the approved Phase 9 proposal) -- it only guarantees that within this
    fixed list, a delisting occurring after declaration is never silently
    gap-filled: ``status == DELISTED`` structurally REQUIRES a populated
    :class:`DelistingEvent`, and ``status == ACTIVE`` structurally forbids
    one (no partial/contradictory state is representable)."""

    model_config = {"frozen": True, "extra": "forbid"}

    raw_symbol: str = Field(min_length=1)
    declared_at: date
    status: UniverseMembershipStatus
    delisting: DelistingEvent | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> EquityUniverseMembership:
        if self.status == UniverseMembershipStatus.DELISTED and self.delisting is None:
            raise ValueError(
                f"{self.raw_symbol!r} is marked DELISTED but has no DelistingEvent -- "
                "a delisting must never be recorded without a real, cited event"
            )
        if self.status == UniverseMembershipStatus.ACTIVE and self.delisting is not None:
            raise ValueError(
                f"{self.raw_symbol!r} is marked ACTIVE but carries a DelistingEvent -- "
                "contradictory membership state"
            )
        if self.delisting is not None and self.delisting.raw_symbol != self.raw_symbol:
            raise ValueError(
                f"DelistingEvent.raw_symbol {self.delisting.raw_symbol!r} does not match "
                f"EquityUniverseMembership.raw_symbol {self.raw_symbol!r}"
            )
        return self


# ---------------------------------------------------------------------------
# Earnings-event timestamps -- PIT announcement timing only (NOT PIT
# fundamentals/consensus; see alpha_agent.equities.data_availability for why).
# ---------------------------------------------------------------------------


class EarningsTiming(str, Enum):
    """When, within the trading day, an earnings announcement was released --
    genuinely load-bearing for "point-in-time" (a BMO release is tradable
    that same session; an AMC release is not tradable until next session)."""

    BEFORE_MARKET_OPEN = "BEFORE_MARKET_OPEN"
    AFTER_MARKET_CLOSE = "AFTER_MARKET_CLOSE"
    DURING_MARKET_HOURS = "DURING_MARKET_HOURS"
    UNKNOWN = "UNKNOWN"


class EquityEarningsEvent(BaseModel):
    """A real earnings-announcement timestamp. ``point_in_time_available``
    follows the exact honesty pattern
    ``alpha_agent.translation.schemas.MeasurableVariable`` already
    established: ``True`` only when a typed, cited source confirms the
    announcement was genuinely knowable at that instant (e.g. a same-day SEC
    EDGAR Form 8-K Item 2.02 filing, which is itself timestamped at
    submission); ``None`` (UNKNOWN) otherwise -- NEVER inferred from a
    calendar convention or a third party's un-cited "estimated" date."""

    model_config = {"frozen": True, "extra": "forbid"}

    raw_symbol: str = Field(min_length=1)
    fiscal_period: str = Field(min_length=1)   # e.g. "FY2023 Q4"
    announcement_date: date
    timing: EarningsTiming
    point_in_time_available: bool | None
    point_in_time_note: str
    source: CorporateActionSource | None = None

    @model_validator(mode="after")
    def _check_pit_claim_has_a_source(self) -> EquityEarningsEvent:
        if self.point_in_time_available is True and self.source is None:
            raise ValueError(
                "point_in_time_available=True requires a real cited source -- "
                "PIT confirmation is never asserted without one"
            )
        return self
