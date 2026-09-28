"""Typed product-capability model -- Market Intelligence + Futures Universe
campaign, Part 3.

Pure schema, no logic: `alpha_agent.ui.market_universe.capability_for` is
what actually DERIVES a `FuturesProductCapability` from the static catalog,
a live (or cached) Databento observation probe, and read-only registry
evidence. Kept separate from that derivation on purpose -- this module has
no network dependency and no registry dependency, so it can be imported (and
its shape asserted) from anywhere without pulling in either plane.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from alpha_agent.marketdata.product_catalog import AssetClass


class CapabilityState(str, Enum):
    """Mission Part 3: "Do not use booleans alone when an UNKNOWN / NOT_TESTED
    state is scientifically important." Used for every capability field where
    the honest answer can be "we don't yet know" rather than a forced
    available/unavailable binary.

    Market Intelligence Data Completion Pass, Section 1 sharpens the
    NOT_TESTED/UNAVAILABLE split with two more explicit states -- a blank
    price must never ambiguously mean either "we never asked" or "the vendor
    has nothing":

    * ``NOT_LOADED`` -- the product is catalogued and observation support is
      believed to exist (it is in a currently "wired"/lazy-loaded scope), but
      no network request has actually been made for it in this session/cache
      window yet. Distinct from ``NOT_TESTED`` ("do we even believe this root
      is supported at all") -- ``NOT_LOADED`` only ever describes "supported,
      not yet fetched".
    * ``NOT_CONNECTED`` -- the PROVIDER itself (not this one product) is
      unavailable, e.g. a health probe reports Databento is disconnected, or
      a news/event connector has no credential configured. Reported instead
      of a per-product ``UNAVAILABLE`` so a global outage never reads as 36
      independent per-product failures.
    """

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_TESTED = "NOT_TESTED"
    NOT_LOADED = "NOT_LOADED"
    NOT_CONNECTED = "NOT_CONNECTED"
    DISABLED = "DISABLED"
    DEGRADED = "DEGRADED"


class FuturesProductCapability(BaseModel):
    """One product's full, typed capability readout (mission Part 3's field
    list, verbatim). `research_enabled` / `paper_trading_enabled` /
    `challenge_eligible` stay plain booleans -- unlike the ``*_verified``/
    ``*_available`` fields, these three are ALWAYS decidable from committed
    evidence (in the RESEARCH_UNIVERSE constant, or not; has a paper-eligible
    registry experiment, or not) with no third "unknown" state, so a
    `CapabilityState` would only invite a meaningless DISABLED/NOT_TESTED
    reading on a fact that is actually just False today.

    Never constructed directly outside `alpha_agent.ui.market_universe` --
    see that module for the one sanctioned derivation path (mission Part 32:
    "create clear service/store abstractions" rather than scattering ad hoc
    capability logic across UI pages)."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "futures-product-capability/1"

    root_symbol: str
    display_name: str
    asset_class: AssetClass
    venue: str
    dataset: str

    market_data_available: CapabilityState
    historical_data_available: CapabilityState
    news_available: CapabilityState
    event_calendar_available: CapabilityState

    research_enabled: bool
    paper_trading_enabled: bool
    challenge_eligible: bool

    contract_economics_verified: CapabilityState
    roll_logic_verified: CapabilityState
    cpp_execution_verified: CapabilityState

    #: Always populated -- a human-inspectable reason for the combination of
    #: states above (mission Part 4: "transparent and inspectable" mapping
    #: logic applies just as much to capability derivation).
    capability_reason: str
    #: When a live probe actually produced `market_data_available`, the
    #: probe's own timestamp; `None` when every field here was derived from
    #: static/declarative facts only (never a fabricated "just checked" time).
    verified_at: datetime | None = None


__all__ = ["CapabilityState", "FuturesProductCapability"]
