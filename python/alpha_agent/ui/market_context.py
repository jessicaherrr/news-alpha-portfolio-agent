"""The UI's ONLY boundary into `alpha_agent.marketdata` (mirrors
`services.py`'s role as the sole boundary into the registry). Every page
that wants a delayed quote or the market-context strip goes through this
module, never `alpha_agent.marketdata` directly -- so there is exactly one
place that constructs a provider, one place that decides what "unavailable"
looks like, and one place the never-fabricate-a-value invariant is enforced.

Nothing here ever calls an order method, an account method, or a registry
write method (there are none in `alpha_agent.marketdata` to call). A quote's
`feed_state` is rendered exactly as the provider reports it (DELAYED or
DISCONNECTED, this release) -- never re-labeled as LIVE.
"""
from __future__ import annotations

from functools import lru_cache

import streamlit as st

from alpha_agent.marketdata.config import load_config
from alpha_agent.marketdata.contracts import contract_for
from alpha_agent.marketdata.ibkr_provider import IBKRDelayedMarketDataProvider
from alpha_agent.marketdata.provider import MarketDataProvider
from alpha_agent.marketdata.quote import FeedState, Quote, QuoteAvailability, QuoteStatus


@lru_cache(maxsize=1)
def _provider() -> MarketDataProvider:
    """One provider instance for the process's lifetime. Constructing it
    never connects -- `connect()` is a separate, explicit, best-effort call
    made lazily by `get_quote`/`render_quote_strip`, so importing this
    module (or running the test suite, which never calls `get_quote`
    against a real provider) makes no network/socket call by itself."""
    return IBKRDelayedMarketDataProvider(load_config())


def _ensure_connected() -> MarketDataProvider:
    provider = _provider()
    if not provider.is_connected():
        try:
            provider.connect()
        except Exception:  # noqa: BLE001, S110 -- market context is best-effort; never fails a page render
            pass
    return provider


def get_quote(root_symbol: str | None) -> Quote | None:
    """The latest known delayed quote for `root_symbol`, or `None` if the
    provider is disabled/unreachable/has no tick yet. Never raises, never
    fabricates a value, never blocks longer than the provider's own
    configured connect timeout."""
    if not root_symbol or not contract_for(root_symbol):
        return None
    try:
        provider = _ensure_connected()
        return provider.get_quote(root_symbol)
    except Exception:  # noqa: BLE001 -- same defensive boundary as the rest of the UI layer
        return None


def is_connected() -> bool:
    """Whether the underlying TWS/IB Gateway SOCKET is currently connected --
    deliberately separate from `get_quote`'s "is there a usable quote"
    question. A release-review finding: the quote strip used to collapse
    "connected, no tick yet" into the same DISCONNECTED banner as "socket
    never connected," which told a user to start TWS while TWS was already
    connected. Never raises."""
    try:
        return _ensure_connected().is_connected()
    except Exception:  # noqa: BLE001 -- same defensive boundary as the rest of the UI layer
        return False


def quote_status(root_symbol: str | None) -> QuoteStatus | None:
    """The richer connection/availability diagnosis behind `render_quote_strip`
    -- `None` only when `root_symbol` is missing or not one of the approved
    roots (the caller has nothing to ask about); never raises otherwise."""
    if not root_symbol or not contract_for(root_symbol):
        return None
    try:
        provider = _ensure_connected()
        return provider.quote_status(root_symbol)
    except Exception:  # noqa: BLE001 -- same defensive boundary as the rest of the UI layer
        return None


def is_market_context_enabled() -> bool:
    """Whether IBKR delayed context is configured on at all -- used to
    decide whether to show the "disconnected" banner or nothing."""
    try:
        return load_config().enabled
    except Exception:  # noqa: BLE001
        return False


def render_quote_strip(root_symbol: str | None = None) -> None:
    """The market-context strip shown on Agent/Dashboard. Renders nothing at
    all when the feed is disabled (the default), per the release instruction
    not to clutter the offline default experience. Otherwise renders exactly
    one of three states -- CONNECTION STATE (connected/disconnected) is kept
    strictly separate from QUOTE AVAILABILITY (available/waiting/
    unavailable): a release-review finding was that a connected-but-
    quoteless root rendered the same "DISCONNECTED / start TWS" banner as an
    actually-disconnected socket, which is false whenever TWS is really up."""
    if not is_market_context_enabled():
        return

    if not is_connected():
        _render_disconnected_banner()
        return

    root = root_symbol or st.session_state.get("market_selected_root")
    status = quote_status(root) if root else None
    if status is None or status.availability != QuoteAvailability.AVAILABLE or status.quote is None:
        _render_connected_no_quote_banner(status)
        return

    _render_quote(status.quote)


def _render_disconnected_banner() -> None:
    st.markdown(
        '<div class="aa-marketstrip aa-marketstrip-disconnected">'
        '<span class="aa-badge" style="color:#6b7280;border-color:#6b728066;background:#6b72801f;">'
        "&#9679; IBKR DELAYED FEED &middot; DISCONNECTED</span>"
        '<span class="aa-marketstrip-hint">Start TWS or IB Gateway (API access enabled, delayed data ok) '
        "to enable market context. The rest of this app works fully offline without it.</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_connected_no_quote_banner(status: QuoteStatus | None) -> None:
    """TWS IS connected -- never tell the user to start it. Distinguishes
    "still waiting for the first tick" (blue, transient) from a TWS-reported
    permission/contract/unknown error (amber, carries the real diagnostic
    code/message verbatim, never invented)."""
    availability = status.availability if status is not None else QuoteAvailability.WAITING
    if availability == QuoteAvailability.WAITING:
        st.markdown(
            '<div class="aa-marketstrip aa-marketstrip-waiting">'
            '<span class="aa-badge" style="color:#3b82f6;border-color:#3b82f666;background:#3b82f61f;">'
            "&#9679; IBKR &middot; DELAYED</span>"
            '<span class="aa-marketstrip-hint">CONNECTED &middot; WAITING FOR QUOTE -- no tick received for this '
            "root yet.</span></div>",
            unsafe_allow_html=True,
        )
        return

    detail = ""
    if status is not None and status.diagnostic_code is not None:
        detail = f" (TWS {status.diagnostic_code}: {status.diagnostic_message})"
    st.markdown(
        '<div class="aa-marketstrip aa-marketstrip-unavailable">'
        '<span class="aa-badge" style="color:#f59e0b;border-color:#f59e0b66;background:#f59e0b1f;">'
        "&#9679; IBKR &middot; DELAYED</span>"
        f'<span class="aa-marketstrip-hint">CONNECTED &middot; QUOTE UNAVAILABLE{detail}</span></div>',
        unsafe_allow_html=True,
    )


def _render_quote(quote: Quote) -> None:
    parts = [f'<span class="aa-marketstrip-ticker">{quote.root_symbol}</span>']
    if quote.last is not None:
        parts.append(f'<span class="aa-marketstrip-item">Last <b>{quote.last:g}</b></span>')
    if quote.bid is not None:
        parts.append(f'<span class="aa-marketstrip-item">Bid <b>{quote.bid:g}</b></span>')
    if quote.ask is not None:
        parts.append(f'<span class="aa-marketstrip-item">Ask <b>{quote.ask:g}</b></span>')
    if quote.volume is not None:
        parts.append(f'<span class="aa-marketstrip-item">Vol <b>{quote.volume:g}</b></span>')
    parts.append(
        f'<span class="aa-badge" style="color:#f59e0b;border-color:#f59e0b66;background:#f59e0b1f;">'
        f"&#9679; {quote.provider} &middot; DELAYED ~15-20m</span>"
    )
    parts.append(f'<span class="aa-marketstrip-ts">as of {quote.received_timestamp:%H:%M:%S} UTC</span>')
    st.markdown(f'<div class="aa-marketstrip">{"".join(parts)}</div>', unsafe_allow_html=True)


def observational_context_note(root_symbol: str | None) -> str | None:
    """A single, clearly-tagged free-text string for the Research Agent's
    `ResearchContext.knowledge_base` -- the ONE sanctioned extension point
    for background prose (see `llm_demo.build_context_for_objective`).
    Explicitly labeled OBSERVATIONAL_CONTEXT_ONLY so nothing downstream can
    mistake this for scientific evidence; it never enters a feature, a
    backtest, validation, `experiment_identity`, or paper eligibility.
    Returns `None` when no quote is available -- the agent then proceeds
    exactly as it always has, with no market-context line at all."""
    quote = get_quote(root_symbol) if root_symbol else None
    if quote is None or quote.feed_state != FeedState.DELAYED:
        return None
    last = f"{quote.last:g}" if quote.last is not None else "n/a"
    return (
        f"OBSERVATIONAL_CONTEXT_ONLY provider={quote.provider} feed={quote.feed_state.value} "
        f"root={quote.root_symbol} last={last} as_of={quote.received_timestamp.isoformat()} "
        "-- delayed market color for conversational context only; not scientific evidence, "
        "not a feature, not used in any validation or paper-eligibility decision."
    )
