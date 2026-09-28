"""The read-only market-data provider boundary every adapter implements.

No implementation of this protocol may expose an order method, an account/
trading method, or a registry write method -- see the package docstring.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from alpha_agent.marketdata.quote import FeedState, Quote, QuoteStatus


@runtime_checkable
class MarketDataProvider(Protocol):
    """Read-only market-data access. Implementations: `IBKRDelayedMarketDataProvider`.

    Named ``MarketDataProvider`` rather than ``LiveMarketDataProvider`` so a
    provider that is explicitly delayed-only (this release's only
    implementation) is never mis-described as a live feed by its own type
    name.
    """

    def connect(self) -> bool:
        """Attempt to connect, honoring the provider's own configured
        timeout. Returns whether a usable connection resulted; never
        raises for an ordinary "the other side isn't running" failure."""
        ...

    def is_connected(self) -> bool:
        ...

    def feed_state(self) -> FeedState:
        ...

    def get_quote(self, root_symbol: str) -> Quote | None:
        """Returns the latest known quote for ``root_symbol``, or ``None``
        if unavailable (not connected, symbol not supported, no tick
        received yet). Never fabricates a value."""
        ...

    def quote_status(self, root_symbol: str) -> QuoteStatus:
        """Richer than `get_quote`: distinguishes WHY no quote exists yet
        (still connected and waiting, a TWS permission/subscription error, a
        contract-resolution error) from the provider being disconnected
        outright -- so the UI never tells a user to "start TWS" when TWS is
        already connected. Presentation-only; never raises."""
        ...

    def disconnect(self) -> None:
        ...
