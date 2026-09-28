"""Typed quote schema for the optional, read-only market-context feed.

Pydantic per CLAUDE.md's cross-component-schema preference. This is
presentation/conversational context only -- see the package docstring for
the full boundary this deliberately never crosses.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class FeedState(str, Enum):
    """The state of a market-data feed. This release's IBKR adapter only
    ever produces ``DELAYED`` or ``DISCONNECTED`` -- ``LIVE`` and ``FROZEN``
    are named for the typed contract's completeness (and so a future,
    explicitly-approved real-time integration has somewhere to report into)
    but nothing in this codebase requests or emits them today."""

    DELAYED = "DELAYED"
    LIVE = "LIVE"
    FROZEN = "FROZEN"
    DISCONNECTED = "DISCONNECTED"


class Quote(BaseModel):
    """One point-in-time quote snapshot. Every field is either read verbatim
    off the provider's own callback data or a local wall-clock timestamp --
    nothing here is computed, inferred, or backfilled."""

    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str
    contract: str
    last: float | None = None
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    last_size: float | None = None
    volume: float | None = None
    exchange_timestamp: datetime | None = None
    received_timestamp: datetime
    provider: str
    feed_state: FeedState

    def is_stale(self, *, max_age_seconds: float = 60.0) -> bool:
        """Purely a display hint (e.g. dim an old quote) -- never used to
        fabricate a fresher value."""
        age = datetime.now(self.received_timestamp.tzinfo) - self.received_timestamp
        return age.total_seconds() > max_age_seconds


class MarketDataUnavailable(RuntimeError):
    """The provider could not produce a quote (not connected, request timed
    out, symbol not found, etc.) -- callers must treat this the same as
    ``None``: never render a fabricated value in its place."""


class QuoteAvailability(str, Enum):
    """Why `quote_status()` does or doesn't carry a `Quote` -- kept
    deliberately separate from `FeedState`/connection status (release-review
    finding: the UI was collapsing "socket connected, no tick yet" and
    "socket disconnected" into the same "DISCONNECTED" banner, which is
    false whenever the socket is actually up). For PRESENTATION ONLY -- this
    is never persisted to `ExperimentRegistry` or any scientific artifact."""

    AVAILABLE = "AVAILABLE"                #: a real quote exists
    WAITING = "WAITING"                    #: connected + subscribed, no tick yet, no error
    NO_PERMISSION = "NO_PERMISSION"        #: TWS reported a market-data permission/subscription error
    CONTRACT_ERROR = "CONTRACT_ERROR"      #: TWS reported a contract-resolution/validation error
    UNKNOWN_UNAVAILABLE = "UNKNOWN_UNAVAILABLE"  #: an unrecognized TWS error code -- raw code/message still shown
    DISCONNECTED = "DISCONNECTED"          #: the provider itself is not connected


class QuoteStatus(BaseModel):
    """Connection/quote-availability diagnosis for one root, richer than a
    bare `Quote | None` -- lets the UI say "connected, still waiting" or
    "connected, no permission for this product" instead of always falling
    back to a blanket, and sometimes false, "disconnected" message.
    `diagnostic_code`/`diagnostic_message` are the raw TWS `error()` values
    verbatim (never invented) when TWS reported one; both `None` otherwise."""

    model_config = {"frozen": True, "extra": "forbid"}

    availability: QuoteAvailability
    quote: Quote | None = None
    diagnostic_code: int | None = None
    diagnostic_message: str | None = None
