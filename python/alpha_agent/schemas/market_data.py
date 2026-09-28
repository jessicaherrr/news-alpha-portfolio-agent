"""Canonical futures market-data schemas and the Python -> C++ boundary contract.

Phase 02.5 froze these. See docs/BOUNDARY_CONTRACT.md for the rationale and the
seven-layer separation (raw vendor / canonical raw-contract bars / unadjusted
continuous / back-adjusted / contract metadata / roll map / fills).

Nothing here downloads data or talks to a vendor.
"""
from __future__ import annotations

import math
from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, model_validator

# Bumped whenever CANONICAL_BAR_COLUMNS / semantics change. Recorded in every
# processed lineage sidecar.
CANONICAL_SCHEMA_VERSION = "1.0"
# Layer 3 / 4 / 6 schema versions (Phase 04).
CONTINUOUS_SCHEMA_VERSION = "1.0"
BACKADJUSTED_SCHEMA_VERSION = "1.0"
ROLL_SCHEMA_VERSION = "1.0"

# Databento DBN fixed-point conventions.
PRICE_SCALE = 1_000_000_000            # DBN prices are int64 * 1e-9
UNDEF_PRICE = 9_223_372_036_854_775_807  # INT64_MAX -- DBN "no price" sentinel

# Normalized raw-contract / raw-continuous price invariant (mirrors the C++
# core's ``kMaxPlausibleRawPrice``). A normalized futures price may be positive,
# ZERO, or NEGATIVE -- WTI crude (CL) front-month settled at -$37.63 on
# 2020-04-20. The only price-domain guard is: finite (not NaN/inf) and magnitude
# below this fixed-point tripwire. Missing data is NEVER encoded as a price
# value -- it is the UNDEF_PRICE sentinel / an explicit rejection, upstream.
NORMALIZED_PRICE_ABS_LIMIT = 1.0e9


def is_normalized_price(x: float) -> bool:
    """True iff ``x`` is a valid normalized raw-contract price: finite and with a
    magnitude below the fixed-point tripwire. Sign is unconstrained."""
    return math.isfinite(x) and abs(x) < NORMALIZED_PRICE_ABS_LIMIT

# Minimum fields the canonical normalizer needs from a vendor OHLCV record set
# (a superset column layout, e.g. Databento ``DBNStore.to_df(price_type="fixed",
# pretty_ts=False)`` for ``ohlcv-1m``, is fine -- extra columns are ignored).
VENDOR_OHLCV_COLUMNS: tuple[str, ...] = (
    "ts_event",       # int64  UTC ns since epoch (never rewritten)
    "instrument_id",  # uint32
    "open", "high", "low", "close",  # int64 fixed-point (PRICE_SCALE), UNDEF_PRICE == no price
    "volume",         # int64/uint64
    "symbol",         # str    vendor-reported symbol for the record
)

# Fields the definition parser reads from a vendor instrument-definition record
# set (Databento ``InstrumentDefMsg`` via ``to_df()`` default float price_type).
# Contract economics (tick_size / point value) are DERIVED from these -- see
# alpha_agent.data.contract_economics -- not taken from ``contract_multiplier``
# (a sentinel on GLBX.MDP3).
VENDOR_DEFINITION_COLUMNS: tuple[str, ...] = (
    "instrument_id",               # uint32
    "raw_symbol",                  # str
    "asset",                       # str  -- root symbol (authoritative), e.g. "NQ"
    "exchange",                    # str  -- ISO 10383 MIC, e.g. "XCME"
    "instrument_class",            # str  -- "F" future / "S" spread / ...
    "min_price_increment",         # float -- quoted-price tick, e.g. 0.25
    "min_price_increment_amount",  # float
    "display_factor",              # float
    "unit_of_measure",             # str  -- e.g. "IPNT"
    "unit_of_measure_qty",         # float -- contract size, e.g. 20.0
    "main_fraction",               # float or "" (255 sentinel -> "")
    "contract_multiplier",         # raw vendor value, kept for the record; NOT used
    "activation",                  # int64 ns
    "expiration",                  # int64 ns
    "first_notice",                # "" -- Databento has no such field
    "last_trade",                  # "" -- Databento has no such field
)


# --- column orders (the on-disk / on-wire contracts) ------------------------

# Layer 2: canonical raw-contract bars stored as Parquet under data/processed/bars/.
CANONICAL_BAR_COLUMNS: tuple[str, ...] = (
    "ts_event_ns",     # int64  UTC ns since epoch, interval start (source time, never rewritten)
    "instrument_id",   # uint32 Databento per-dataset id -- join key to contract metadata
    "raw_symbol",      # str    e.g. "NQZ6" (denormalized for auditing)
    "root_symbol",     # str    e.g. "NQ" (denormalized convenience / partition)
    "open", "high", "low", "close",  # float64 actual traded prices of that contract
    "volume",          # int64  contracts traded in the interval
    "trading_day",     # date   CME session date (derived label; does not replace ts_event_ns)
    "session",         # str    RTH / ETH / MAINTENANCE / CLOSED (derived label)
)

# Layer 5: contract metadata. Also the exact header the C++ core parses
# (cpp/src/contract_io.cpp).
CONTRACT_COLUMNS: tuple[str, ...] = (
    "instrument_id", "raw_symbol", "root_symbol", "exchange",
    "tick_size", "multiplier", "activation_ns", "expiration_ns",
    "first_notice_ns", "last_trade_ns",
)

# The exact bar columns Python writes across the CLI boundary to the C++ core
# (a subset of the canonical bar -- the core resolves the rest from the contract
# registry it is given alongside).
BOUNDARY_BAR_COLUMNS: tuple[str, ...] = (
    "ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume",
)

# Layer 3: unadjusted continuous series (data/processed/continuous/).
CONTINUOUS_BAR_COLUMNS: tuple[str, ...] = (
    "ts_event_ns", "continuous_symbol", "active_instrument_id", "active_raw_symbol",
    "open", "high", "low", "close", "volume", "trading_day", "session", "is_roll_boundary",
)

# Layer 4: research-only back-adjusted series (data/processed/backadjusted/).
# No instrument_id / raw_symbol -- the adjusted price never traded.
BACKADJUSTED_BAR_COLUMNS: tuple[str, ...] = (
    "ts_event_ns", "continuous_symbol", "open", "high", "low", "close", "volume",
    "adjustment_method", "adjustment_mode", "cumulative_adjustment", "adjusted_through_ts_ns",
)

# Layer 6: roll map (data/processed/rolls/).
ROLL_EVENT_COLUMNS: tuple[str, ...] = (
    "continuous_symbol", "effective_ts_ns", "from_instrument_id", "to_instrument_id",
    "from_raw_symbol", "to_raw_symbol", "rule", "price_policy", "used_fallback",
    "from_ts_ns", "to_ts_ns", "from_price", "to_price", "additive_gap",
)


class Session(str, Enum):
    RTH = "RTH"
    ETH = "ETH"
    MAINTENANCE = "MAINTENANCE"
    CLOSED = "CLOSED"


class PriceDomain(str, Enum):
    """Which price space a series lives in. Only RAW_CONTRACT (Futures) or RAW
    (ETF) may back a fill.

    ``RAW`` / ``SPLIT_ADJUSTED`` / ``TOTAL_RETURN`` were added in Phase 6 (ETF
    Research Pilot) as the ETF-domain mirror of Futures' existing
    RAW_CONTRACT/BACK_ADJUSTED distinction -- the SAME shared, asset-neutral
    mechanism (``FeatureSpec.required_price_domain``, ``EXECUTION_PRICE_DOMAINS``
    in ``alpha_agent.features.source``), never a second, ETF-only concept
    bolted onto ``StrategySpec`` itself: ``StrategySpec``'s DSL fingerprint is
    frozen (identity-bearing for every existing Futures experiment) and is
    deliberately NOT touched by this addition. ``RAW``: unadjusted share price
    (the only domain a Fill may use). ``SPLIT_ADJUSTED``: back-adjusted for
    splits only, dividends NOT removed -- safe for a pure price/trend signal
    across a split boundary, still NOT total return. ``TOTAL_RETURN``: splits
    AND reinvested cash distributions both removed -- a genuinely different
    quantitative series from RAW/SPLIT_ADJUSTED, never silently substituted for
    either (a price-momentum factor must not silently become a total-return
    factor -- Phase 6 instruction 6)."""

    RAW_CONTRACT = "raw_contract"
    RAW_CONTINUOUS = "raw_continuous"
    BACK_ADJUSTED = "back_adjusted"
    RAW = "raw"
    SPLIT_ADJUSTED = "split_adjusted"
    TOTAL_RETURN = "total_return"


_CONTINUOUS_RULES = frozenset({"c", "v", "n"})


def is_continuous_symbol(symbol: str) -> bool:
    """True for Databento continuous symbology, e.g. ``NQ.v.0`` / ``ES.c.1``."""
    if any(tok in symbol for tok in (".c.", ".v.", ".n.")):
        return True
    parts = symbol.split(".")
    return (
        len(parts) == 3
        and len(parts[1]) == 1
        and parts[1] in _CONTINUOUS_RULES
        and parts[2].isdigit()
    )


def is_tradable_contract_symbol(symbol: str) -> bool:
    """A real contract symbol is non-empty and dot-free (``NQZ6``), never a
    continuous (``NQ.v.0``) or parent (``NQ.FUT``) symbol."""
    if not symbol or "." in symbol:
        return False
    return not is_continuous_symbol(symbol)


class ContractSpecModel(BaseModel):
    """Mirror of the C++ ``quant::ContractSpec`` (cpp/include/quant_core/contract.hpp)."""

    instrument_id: int = Field(gt=0)
    raw_symbol: str = Field(min_length=1)
    root_symbol: str = Field(min_length=1)
    exchange: str = Field(min_length=1)
    tick_size: float = Field(gt=0)
    multiplier: float = Field(gt=0)
    activation_ns: int = Field(gt=0)
    expiration_ns: int = Field(gt=0)
    first_notice_ns: int | None = None
    last_trade_ns: int | None = None

    @model_validator(mode="after")
    def _check(self) -> ContractSpecModel:
        if not is_tradable_contract_symbol(self.raw_symbol):
            raise ValueError(
                f"raw_symbol must be a real tradable contract, got {self.raw_symbol!r}"
            )
        if self.expiration_ns <= self.activation_ns:
            raise ValueError("expiration_ns must be after activation_ns")
        return self


class CanonicalBar(BaseModel):
    """One row of a layer-2 canonical raw-contract bar series."""

    ts_event_ns: int = Field(gt=0)
    instrument_id: int = Field(gt=0)
    raw_symbol: str = Field(min_length=1)
    root_symbol: str = Field(min_length=1)
    open: float
    high: float
    low: float
    close: float
    volume: int = Field(ge=0)
    trading_day: date
    session: Session
    price_domain: PriceDomain = PriceDomain.RAW_CONTRACT

    @model_validator(mode="after")
    def _check(self) -> CanonicalBar:
        if self.price_domain is not PriceDomain.RAW_CONTRACT:
            raise ValueError("a canonical raw-contract bar must be PriceDomain.RAW_CONTRACT")
        if not is_tradable_contract_symbol(self.raw_symbol):
            raise ValueError(f"raw_symbol must be a real contract, got {self.raw_symbol!r}")
        # Signed prices are valid (historical negative CL). The guard is finite +
        # magnitude only -- NaN/inf or an un-normalized fixed-point value fails.
        for name in ("open", "high", "low", "close"):
            if not is_normalized_price(getattr(self, name)):
                raise ValueError(f"{name} is not a finite normalized raw-contract price")
        # OHLC ordering is arithmetic and holds for signed values.
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high is below open/close/low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low is above open/close/high")
        return self


class ContinuousBar(BaseModel):
    """One row of a layer-3 unadjusted continuous series. Prices are the raw
    prices of ``active_instrument_id`` and are discontinuous at rolls."""

    ts_event_ns: int = Field(gt=0)
    continuous_symbol: str = Field(min_length=1)
    active_instrument_id: int = Field(gt=0)
    active_raw_symbol: str = Field(min_length=1)
    open: float
    high: float
    low: float
    close: float
    volume: int = Field(ge=0)
    trading_day: date | None = None       # derived label, carried through from layer 2
    session: Session | None = None
    is_roll_boundary: bool = False
    price_domain: PriceDomain = PriceDomain.RAW_CONTINUOUS

    @model_validator(mode="after")
    def _check(self) -> ContinuousBar:
        if not is_continuous_symbol(self.continuous_symbol):
            raise ValueError(f"continuous_symbol must be continuous, got {self.continuous_symbol!r}")
        if not is_tradable_contract_symbol(self.active_raw_symbol):
            raise ValueError("active_raw_symbol must be a real contract")
        if self.price_domain is not PriceDomain.RAW_CONTINUOUS:
            raise ValueError("a continuous bar must be PriceDomain.RAW_CONTINUOUS")
        # Raw prices of the active contract -- signed is valid (negative CL).
        for name in ("open", "high", "low", "close"):
            if not is_normalized_price(getattr(self, name)):
                raise ValueError(f"{name} is not a finite normalized price")
        return self


class BackAdjustedBar(BaseModel):
    """One row of a layer-4 research-only back-adjusted series.

    Carries no ``raw_symbol`` / ``instrument_id`` on the price row: the adjusted
    price never traded, so it must never be mistaken for something fillable.
    """

    ts_event_ns: int = Field(gt=0)
    continuous_symbol: str = Field(min_length=1)
    open: float
    high: float
    low: float
    close: float
    volume: int = Field(ge=0)
    adjustment_method: str = "additive"
    # "retrospective_research" (offline full-history) | "point_in_time" (walk-forward safe)
    adjustment_mode: str = "retrospective_research"
    cumulative_adjustment: float = 0.0
    adjusted_through_ts_ns: int = Field(gt=0)
    price_domain: PriceDomain = PriceDomain.BACK_ADJUSTED

    @model_validator(mode="after")
    def _check(self) -> BackAdjustedBar:
        if self.price_domain is not PriceDomain.BACK_ADJUSTED:
            raise ValueError("a back-adjusted bar must be PriceDomain.BACK_ADJUSTED")
        if self.adjustment_mode not in ("retrospective_research", "point_in_time"):
            raise ValueError(f"unknown adjustment_mode {self.adjustment_mode!r}")
        # Adjusted prices are already signed by design; still reject NaN/inf and
        # an un-normalized fixed-point magnitude.
        for name in ("open", "high", "low", "close"):
            if not is_normalized_price(getattr(self, name)):
                raise ValueError(f"{name} is not a finite normalized price")
        return self


class RollEvent(BaseModel):
    """One row of a layer-6 roll map: an auditable continuous -> contract change.

    Authoritative trigger is the ``instrument_id`` transition of the continuous
    series, never a contract-month string.
    """

    continuous_symbol: str = Field(min_length=1)
    effective_ts_ns: int = Field(gt=0)     # ts of the first bar on the new contract
    from_instrument_id: int = Field(gt=0)
    to_instrument_id: int = Field(gt=0)
    from_raw_symbol: str = Field(min_length=1)
    to_raw_symbol: str = Field(min_length=1)
    rule: str = "instrument_id_transition"  # what triggered the roll (observed, not computed)
    # preferred: same_timestamp_close_close > same_timestamp_open_open ;
    # prev_close_new_open only as an explicitly marked fallback.
    price_policy: str = "same_timestamp_close_close"
    used_fallback: bool = False             # True => the aligned basis was unavailable
    from_ts_ns: int | None = None           # ts of the reference bar on the OLD contract
    to_ts_ns: int | None = None             # ts of the reference bar on the NEW contract
    from_price: float | None = None
    to_price: float | None = None
    additive_gap: float | None = None       # to_price - from_price

    @model_validator(mode="after")
    def _check(self) -> RollEvent:
        if self.from_instrument_id == self.to_instrument_id:
            raise ValueError("a roll must change the instrument_id")
        if not is_continuous_symbol(self.continuous_symbol):
            raise ValueError("continuous_symbol must be continuous")
        for s in (self.from_raw_symbol, self.to_raw_symbol):
            if not is_tradable_contract_symbol(s):
                raise ValueError(f"roll endpoint {s!r} must be a real contract")
        return self
