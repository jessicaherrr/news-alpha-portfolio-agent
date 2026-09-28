"""Release UX -- optional, read-only, delayed market-data context.

Deliberately a SEPARATE top-level package from `alpha_agent.data` (the
historical Databento research pipeline). Nothing here is a market-data
source for research: no feature, no backtest, no validation, no
`experiment_identity`, and no paper-trading eligibility check may import from
this package. It exists only to give the UI (and, as one tagged
OBSERVATIONAL_CONTEXT_ONLY `knowledge_base` string, the conversational
Research Agent) an optional live-ish quote for display/ideation.

CLAUDE.md boundaries this package holds itself to:

* read-only -- no order method, no account/trading method, no registry write
  method exists anywhere in this package;
* delayed only -- `IBKRDelayedMarketDataProvider` always requests IBKR TWS
  API market data type 3 (delayed, ``reqMarketDataType(3)``); it never
  requests or claims real-time/live data, and a quote's `feed_state` is
  never rendered as `FeedState.LIVE` by this adapter;
* no credentials -- `IBKRConnectionConfig` carries only host/port/client_id/
  enabled/timeout; authentication happens entirely inside a locally-running
  Trader Workstation / IB Gateway that this code merely connects to as a
  local TCP client, exactly like any other TWS API consumer;
* offline-safe -- disabled by default (`configs/market_data.yaml`); when
  enabled but TWS/IB Gateway is not reachable, every call fails closed to
  `FeedState.DISCONNECTED` / `None`, never a fabricated quote, and never
  blocks the caller for longer than the configured connect timeout.
"""
from __future__ import annotations

from alpha_agent.marketdata.provider import MarketDataProvider
from alpha_agent.marketdata.quote import FeedState, Quote, QuoteAvailability, QuoteStatus

__all__ = ["FeedState", "MarketDataProvider", "Quote", "QuoteAvailability", "QuoteStatus"]
