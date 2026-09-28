"""Typed schemas for the Databento market-OBSERVATION plane (mission Part D).

Everything here describes a CURRENT/RECENT observation for display and
conversational context -- never scientific evidence. See
``alpha_agent.marketdata.databento_provider``'s module docstring for the full
boundary. Pydantic per CLAUDE.md's cross-component-schema preference; frozen,
``extra="forbid"`` like every other typed schema in this codebase.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class DatabentoCapability(str, Enum):
    """What the configured key/entitlement actually proves -- never assumed.
    ``LIVE`` and ``DELAYED`` are named for the typed contract's completeness
    (mirrors ``alpha_agent.marketdata.quote.FeedState``) but nothing in this
    module ever emits them: this release only ever proves a HISTORICAL-API
    capability (an entitled dataset whose most recent available data is
    either near-real-time or lagged), never a live/delayed streaming
    gateway. Claiming either would violate CLAUDE.md's "never hard-code a
    delay if the provider does not prove it"."""

    LIVE = "LIVE"
    DELAYED = "DELAYED"
    LATEST_AVAILABLE = "LATEST_AVAILABLE"
    HISTORICAL_ONLY = "HISTORICAL_ONLY"
    NOT_CONNECTED = "NOT_CONNECTED"
    RATE_LIMITED = "RATE_LIMITED"
    ERROR = "ERROR"


#: Capabilities that mean "no usable data can be returned right now" -- callers
#: use this to decide whether to even attempt a snapshot/OHLCV call.
UNAVAILABLE_CAPABILITIES = frozenset(
    {DatabentoCapability.NOT_CONNECTED, DatabentoCapability.RATE_LIMITED, DatabentoCapability.ERROR}
)


class DatabentoHealth(BaseModel):
    """The result of one capability probe -- built ONLY from free Databento
    metadata endpoints (``metadata.get_dataset_range`` / ``list_datasets``),
    never from a billable timeseries request."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "databento-health/1"
    capability: DatabentoCapability
    dataset: str
    checked_at: datetime
    latest_available_ts: datetime | None = None
    lag_seconds: float | None = None
    detail: str = ""


class ContractResolution(BaseModel):
    """Root -> continuous display proxy -> resolved raw contract, kept
    explicitly distinct (mission section 15: "do not make a continuous
    synthetic symbol look like a directly tradable contract"). ``expiry`` is
    the resolved contract's OWN expiration from the Databento definition
    record -- never guessed."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "databento-contract-resolution/1"
    root_symbol: str
    display_symbol: str  # e.g. "NQ.v.0" -- a continuous front-month PROXY, not a tradable symbol
    resolved_raw_symbol: str | None = None  # e.g. "NQU6"
    resolved_instrument_id: int | None = None
    expiry: datetime | None = None
    exchange: str | None = None
    resolved_at: datetime
    stype_in: str = "continuous"
    dataset: str = "GLBX.MDP3"
    note: str = (
        "Continuous front-month proxy for display only -- not a statement about which raw "
        "contract any historical backtest actually traded on a given date."
    )


class ContractEconomicsView(BaseModel):
    """Display-only projection of ``alpha_agent.data.contract_economics.
    ContractEconomics`` -- DERIVED from the same Databento definition record
    ``resolve_display_contract`` already fetched, via the SAME shared,
    never-hardcoded derivation the scientific pipeline uses (CLAUDE.md
    contract-economics rules). Never a second, independent guess."""

    model_config = {"frozen": True, "extra": "forbid"}

    quote_tick_size: float
    contract_size: float
    unit_of_measure: str
    point_value_usd: float
    tick_value_usd: float
    quote_convention: str
    source: str


class OhlcvBar(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    ts_event: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class RecentOhlcvResult(BaseModel):
    """``get_recent_ohlcv``'s outcome. ``fetched=False`` means the bounded
    cost estimate exceeded the configured auto-fetch ceiling -- the caller
    gets the estimate back and must explicitly approve a larger fetch
    (mission section 16); this module NEVER downloads past that ceiling on
    its own."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "databento-recent-ohlcv/1"
    root_symbol: str
    timeframe: str
    bars: tuple[OhlcvBar, ...] = ()
    as_of: datetime
    capability: DatabentoCapability
    estimated_cost_usd: float | None = None
    fetched: bool
    detail: str = ""


class SessionSummary(BaseModel):
    """Descriptive UTC-calendar-day aggregate -- deliberately NOT presented as
    the CME's official session boundary (17:00 CT roll), which this module
    does not implement; ``session_label`` says so explicitly so the UI never
    implies more precision than is real."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "databento-session-summary/1"
    root_symbol: str
    session_date: str  # ISO date, UTC calendar day this summary actually covers
    session_label: str = "UTC calendar day (not the official CME 17:00 CT session roll)"
    session_open: float | None = None
    session_high: float | None = None
    session_low: float | None = None
    session_close: float | None = None
    volume: float | None = None
    n_bars: int = 0


class MarketSnapshot(BaseModel):
    """One point-in-time OBSERVATION-plane snapshot for display and
    conversational context. Every field is either read verbatim from a
    Databento response or a locally computed descriptive statistic over
    already-fetched real bars -- nothing here is a trading signal, and
    nothing here may enter a feature, a backtest, or validation (see the
    provider module's boundary docstring and
    ``tests/python/test_release_market_plane_isolation.py``)."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "databento-market-snapshot/1"
    root_symbol: str
    contract: ContractResolution
    last: float | None = None
    change: float | None = None
    change_pct: float | None = None
    session: SessionSummary | None = None
    prior_close: float | None = None
    as_of: datetime
    source: str = "DATABENTO"
    dataset: str = "GLBX.MDP3"
    capability: DatabentoCapability
    freshness_seconds: float | None = None


class FuturesContract(BaseModel):
    """One real, expiry-ordered outright futures instrument from a Databento
    ``definition`` (``stype_in="parent"``) row -- Market Intelligence
    Checkpoint C, Section 4. Never a futures SPREAD (``instrument_class``
    other than the outright-future code is filtered out before this is ever
    built -- see ``DatabentoMarketDataProvider._outright_definitions``).

    ``instrument_name`` stays ``None`` for GLBX.MDP3 (Databento's own
    definition schema carries no such field for CME/CBOT/NYMEX/COMEX futures
    -- Section 4: "Required contract fields WHERE ACTUALLY SUPPLIED"); never
    a synthesized label dressed up as vendor data. ``open_interest`` /
    ``settlement_price`` are populated only from a real Databento
    ``statistics`` row (``stat_type`` 9 / 3 respectively) and stay ``None``
    otherwise -- Section 6: "Never fabricate open interest ... If not
    available: N/A, not 0."
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "futures-contract/1"
    root_symbol: str
    raw_symbol: str
    instrument_id: int
    instrument_name: str | None = None
    activation: datetime | None = None
    expiration: datetime
    min_price_increment: float | None = None
    display_factor: float | None = None
    venue: str | None = None  # Databento `exchange` (e.g. "XNYM") -- the real venue/publisher code
    is_front_month: bool = False
    #: True when this raw symbol matches the SAME contract
    #: `resolve_display_contract` resolves the continuous front-month proxy
    #: to -- may legitimately differ from `is_front_month` (Databento's own
    #: continuous-roll methodology is not always "first unexpired expiry").
    is_display_contract: bool = False
    last: float | None = None
    settlement_price: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    days_to_expiry: int | None = None


class ContractLadderResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "futures-contract-ladder/1"
    root_symbol: str
    contracts: tuple[FuturesContract, ...] = ()
    as_of: datetime
    capability: DatabentoCapability
    estimated_cost_usd: float | None = None
    fetched: bool
    detail: str = ""


class CurveShape(str, Enum):
    """Term-structure regime, classified with an explicit tolerance
    (Checkpoint C Section 8) -- never from raw noise. ``INSUFFICIENT_DATA``
    is a first-class outcome (fewer than 2 real priced legs), not an error."""

    CONTANGO = "CONTANGO"
    BACKWARDATION = "BACKWARDATION"
    FLAT = "FLAT"
    MIXED = "MIXED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class TermStructurePoint(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    raw_symbol: str
    expiration: datetime
    price: float | None = None  # settlement price when available, else the OHLCV close
    price_source: str = "unavailable"  # "settlement" | "last" | "unavailable"
    volume: float | None = None
    open_interest: float | None = None


class TermStructureResult(BaseModel):
    """Built from `ContractLadderResult`'s OWN expiry-ordered outright
    contracts (Section 8) -- never from a continuous ``ROOT.v.N`` volume-rank
    symbol, which Databento documents as volume-ranked mapping behavior, not
    chronological expiry order (Section 8's mandatory distinction)."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "futures-term-structure/1"
    root_symbol: str
    points: tuple[TermStructurePoint, ...] = ()
    front_second_spread: float | None = None
    front_third_spread: float | None = None
    curve_slope: float | None = None  # mean per-leg spread across all real priced consecutive pairs
    curve_shape: CurveShape
    tolerance_pct: float
    as_of: datetime
    capability: DatabentoCapability
    detail: str = ""


__all__ = [
    "UNAVAILABLE_CAPABILITIES",
    "ContractEconomicsView",
    "ContractLadderResult",
    "ContractResolution",
    "CurveShape",
    "DatabentoCapability",
    "DatabentoHealth",
    "FuturesContract",
    "MarketSnapshot",
    "OhlcvBar",
    "RecentOhlcvResult",
    "SessionSummary",
    "TermStructurePoint",
    "TermStructureResult",
]
