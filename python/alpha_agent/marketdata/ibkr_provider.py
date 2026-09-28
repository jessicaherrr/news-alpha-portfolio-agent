"""IBKR TWS API delayed-market-data adapter.

READ-ONLY: no order method, no account/trading method, and no registry write
method exists anywhere in this module. Always requests delayed data
(``EClient.reqMarketDataType(3)``, the official TWS API semantics for
delayed quotes) immediately after connecting, and this adapter never claims
``FeedState.LIVE``.

Requires the optional `ibapi` package (``pip install -e '.[ibkr]'``) and a
locally-running, already-logged-in Trader Workstation or IB Gateway with API
access enabled -- this module never collects a username/password; TWS/IB
Gateway owns authentication entirely. Neither `ibapi` nor a running
TWS/Gateway is required to import this module or run this repository's test
suite: `ibapi` is imported lazily inside `connect()`, and every public
method fails closed (``False`` / ``None`` / ``FeedState.DISCONNECTED``)
rather than raising or fabricating a value when the vendor package is
absent, the config is disabled, or TWS/IB Gateway is unreachable.
"""
from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import Any

from alpha_agent.marketdata.config import IBKRConnectionConfig, load_config
from alpha_agent.marketdata.contracts import APPROVED_ROOTS, contract_for
from alpha_agent.marketdata.quote import FeedState, Quote, QuoteAvailability, QuoteStatus

#: TWS API delayed market-data type. See EClient.reqMarketDataType: 1=live,
#: 2=frozen, 3=delayed, 4=delayed-frozen. This adapter always requests 3.
_DELAYED_MARKET_DATA_TYPE = 3

# tickType codes this adapter reads (mirrors ibapi.ticktype.TickTypeEnum),
# duplicated as plain ints so this module carries no import-time dependency
# on the optional `ibapi` package. Pinned against a real installed
# `ibapi.ticktype.TickTypeEnum` (ibapi 9.81.1): BID_SIZE=0, BID=1, ASK=2,
# ASK_SIZE=3, LAST=4, LAST_SIZE=5, VOLUME=8; DELAYED_BID=66, DELAYED_ASK=67,
# DELAYED_LAST=68, DELAYED_BID_SIZE=69, DELAYED_ASK_SIZE=70,
# DELAYED_LAST_SIZE=71, DELAYED_VOLUME=74.
_TICK_BID, _TICK_ASK, _TICK_LAST, _TICK_VOLUME = 1, 2, 4, 8
_TICK_DELAYED_BID, _TICK_DELAYED_ASK = 66, 67
_TICK_DELAYED_LAST, _TICK_DELAYED_VOLUME = 68, 74
_SIZE_TICK_BID, _SIZE_TICK_ASK, _SIZE_TICK_LAST = 0, 3, 5
_SIZE_TICK_DELAYED_BID, _SIZE_TICK_DELAYED_ASK, _SIZE_TICK_DELAYED_LAST = 69, 70, 71

# TWS `error()` codes this adapter classifies for PRESENTATION-ONLY
# diagnostics (see `quote_status`) -- never persisted, never used to change
# scientific state. Anything not in either set still surfaces its raw
# code/message under `QuoteAvailability.UNKNOWN_UNAVAILABLE` rather than
# being silently dropped or guessed at.
_NO_PERMISSION_CODES = {354, 10090, 10091, 10168, 10197}
_CONTRACT_ERROR_CODES = {200, 321, 361, 362}
# Connection-housekeeping / "delayed data is being shown" notices -- real
# TWS messages observed during live smokes, informational, never an error
# state for `quote_status` to report.
_INFORMATIONAL_CODES = {2104, 2106, 2108, 2158, 10167}


def _quote_kwargs(existing: Quote | None, root: str, contract_symbol: str) -> dict[str, Any]:
    """Merge one new field into the existing cached quote (or start fresh) --
    every field not explicitly updated below is carried over verbatim."""
    if existing is not None:
        kwargs = existing.model_dump()
    else:
        kwargs = {
            "root_symbol": root, "contract": contract_symbol, "last": None, "bid": None,
            "ask": None, "bid_size": None, "ask_size": None, "last_size": None, "volume": None,
            "exchange_timestamp": None, "provider": "IBKR", "feed_state": FeedState.DELAYED,
        }
    kwargs["received_timestamp"] = datetime.now(UTC)
    return kwargs


class IBKRDelayedMarketDataProvider:
    """One TWS API socket connection, one background message-processing
    thread, a small per-root quote cache updated by wrapper callbacks.
    Deliberately minimal: 5 fixed roots, delayed quotes only -- no
    historical data request, no order, no account data."""

    def __init__(self, config: IBKRConnectionConfig | None = None) -> None:
        self._config = config or load_config()
        self._client: Any = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._quotes: dict[str, Quote] = {}
        self._req_id_to_root: dict[int, str] = {}
        self._local_symbol: dict[str, str] = {}
        self._quote_errors: dict[str, tuple[int, str]] = {}
        self._contract_errors: dict[str, tuple[int, str]] = {}
        self._next_req_id = 9000
        self._connected = False

    # -- connection lifecycle ------------------------------------------------

    def connect(self) -> bool:
        if not self._config.enabled or self._connected:
            return self._connected
        try:
            from ibapi.client import EClient
            from ibapi.wrapper import EWrapper
        except ImportError:
            return False

        ready, quotes, req_map, lock = self._ready, self._quotes, self._req_id_to_root, self._lock
        local_symbol = self._local_symbol
        quote_errors, contract_errors = self._quote_errors, self._contract_errors
        resolved_expiry: dict[str, str] = {}
        detail_req_to_root: dict[int, str] = {}

        class _Wrapper(EWrapper):
            def nextValidId(self_inner, order_id: int) -> None:  # ibapi callback signature is fixed by the vendor
                ready.set()

            def error(self_inner, req_id, error_code, error_string, advanced_order_reject_json=""):
                # A connection-level failure surfaces as `connect()` timing
                # out and returning False -- nothing here raises, retries an
                # order, or fabricates a quote. Real per-request TWS errors
                # ARE captured (keyed by root, never by req_id, since the
                # caller only ever asks about a root) for `quote_status`'s
                # presentation-only diagnostics -- purely informational
                # codes (farm-connection notices, the "displaying delayed
                # market data" notice) are not errors and are not recorded.
                if error_code in _INFORMATIONAL_CODES:
                    return
                root = req_map.get(req_id)
                if root:
                    with lock:
                        quote_errors[root] = (error_code, error_string)
                    return
                root = detail_req_to_root.get(req_id)
                if root:
                    with lock:
                        contract_errors[root] = (error_code, error_string)

            def contractDetails(self_inner, req_id, contract_details) -> None:
                # Resolves a root's CONTFUT to its current real front-month
                # FUT contract (see `_subscribe_all` for why this step is
                # required before `reqMktData`). Read-only metadata, no
                # market-data line, no order, no account/position data.
                root = detail_req_to_root.get(req_id)
                expiry = contract_details.contract.lastTradeDateOrContractMonth
                if root and expiry:
                    with lock:
                        resolved_expiry[root] = expiry
                        if contract_details.contract.localSymbol:
                            local_symbol[root] = contract_details.contract.localSymbol

            def contractDetailsEnd(self_inner, req_id) -> None:
                # Resolution completeness is bounded by a timeout in
                # `_subscribe_all` instead of tracked per-request here --
                # a root that never calls back (or errors instead) is simply
                # never subscribed, never fabricated.
                pass

            def tickPrice(self_inner, req_id, tick_type, price, attrib) -> None:
                root = req_map.get(req_id)
                if not root or price is None or price <= 0:
                    return
                spec = contract_for(root)
                with lock:
                    symbol = local_symbol.get(root) or (spec.ibkr_symbol if spec else root)
                    kwargs = _quote_kwargs(quotes.get(root), root, symbol)
                    if tick_type in (_TICK_LAST, _TICK_DELAYED_LAST):
                        kwargs["last"] = float(price)
                    elif tick_type in (_TICK_BID, _TICK_DELAYED_BID):
                        kwargs["bid"] = float(price)
                    elif tick_type in (_TICK_ASK, _TICK_DELAYED_ASK):
                        kwargs["ask"] = float(price)
                    else:
                        return
                    quotes[root] = Quote(**kwargs)

            def tickSize(self_inner, req_id, tick_type, size) -> None:
                root = req_map.get(req_id)
                if not root:
                    return
                spec = contract_for(root)
                with lock:
                    symbol = local_symbol.get(root) or (spec.ibkr_symbol if spec else root)
                    kwargs = _quote_kwargs(quotes.get(root), root, symbol)
                    # `size` may be a plain int/float or (IBKR's 2026-era
                    # fractional-share/decimal-quantity API) a
                    # `decimal.Decimal` -- `float()` accepts either.
                    if tick_type in (_SIZE_TICK_BID, _SIZE_TICK_DELAYED_BID):
                        kwargs["bid_size"] = float(size)
                    elif tick_type in (_SIZE_TICK_ASK, _SIZE_TICK_DELAYED_ASK):
                        kwargs["ask_size"] = float(size)
                    elif tick_type in (_SIZE_TICK_LAST, _SIZE_TICK_DELAYED_LAST):
                        kwargs["last_size"] = float(size)
                    elif tick_type in (_TICK_VOLUME, _TICK_DELAYED_VOLUME):
                        kwargs["volume"] = float(size)
                    else:
                        return
                    quotes[root] = Quote(**kwargs)

        client = EClient(_Wrapper())
        try:
            client.connect(self._config.host, self._config.port, self._config.client_id)
        except OSError:
            return False

        thread = threading.Thread(target=client.run, daemon=True, name="ibkr-market-data")
        thread.start()
        got_ready = ready.wait(timeout=self._config.connect_timeout_seconds)
        if not got_ready or not client.isConnected():
            try:
                client.disconnect()
            except Exception:  # noqa: BLE001, S110 -- best-effort teardown of a connection that never fully came up
                pass
            return False

        client.reqMarketDataType(_DELAYED_MARKET_DATA_TYPE)  # official TWS API: 3 == delayed
        self._client = client
        self._thread = thread
        self._connected = True
        self._subscribe_all(detail_req_to_root, resolved_expiry)
        return True

    def _subscribe_all(self, detail_req_to_root: dict[int, str], resolved_expiry: dict[str, str]) -> None:
        """Subscribe to a delayed quote for every approved root.

        `reqMktData` REJECTS a `CONTFUT` contract outright -- confirmed
        against a real IBKR TWS Demo instance with TWS error 321 ("Error
        validating request... Please enter a valid security type"). `CONTFUT`
        is only a valid `secType` for `reqContractDetails` /
        `reqHistoricalData`; a live/delayed streaming subscription needs a
        real `FUT` contract naming a specific `lastTradeDateOrContractMonth`.
        So this resolves each root's CONTFUT to its current real front-month
        expiry first (`reqContractDetails` -- read-only metadata, no market-
        data line, no order, no account/position data), THEN subscribes
        `reqMktData` on the resolved `FUT` contract. A root that doesn't
        resolve within `request_timeout_seconds` (or that TWS can't resolve
        at all) is simply never subscribed -- no synthetic/guessed expiry,
        no fabricated quote.
        """
        from ibapi.contract import Contract

        detail_roots: list[str] = []
        for root in APPROVED_ROOTS:
            spec = contract_for(root)
            if not spec:
                continue
            contract = Contract()
            contract.symbol = spec.ibkr_symbol
            contract.secType = spec.sec_type  # CONTFUT -- resolution only, never subscribed directly
            contract.exchange = spec.exchange
            contract.currency = spec.currency
            req_id = self._next_req_id
            self._next_req_id += 1
            detail_req_to_root[req_id] = root
            detail_roots.append(root)
            self._client.reqContractDetails(req_id, contract)

        deadline = time.monotonic() + self._config.request_timeout_seconds
        while time.monotonic() < deadline and len(resolved_expiry) < len(detail_roots):
            time.sleep(0.05)

        for root in detail_roots:
            expiry = resolved_expiry.get(root)
            if not expiry:
                continue
            spec = contract_for(root)
            contract = Contract()
            contract.symbol = spec.ibkr_symbol
            contract.secType = "FUT"
            contract.exchange = spec.exchange
            contract.currency = spec.currency
            contract.lastTradeDateOrContractMonth = expiry
            req_id = self._next_req_id
            self._next_req_id += 1
            self._req_id_to_root[req_id] = root
            self._client.reqMktData(req_id, contract, "", False, False, [])

    def is_connected(self) -> bool:
        return self._connected and self._client is not None and self._client.isConnected()

    def feed_state(self) -> FeedState:
        return FeedState.DELAYED if self.is_connected() else FeedState.DISCONNECTED

    def get_quote(self, root_symbol: str) -> Quote | None:
        if not self.is_connected():
            return None
        with self._lock:
            return self._quotes.get(root_symbol.upper())

    def quote_status(self, root_symbol: str) -> QuoteStatus:
        """The connection-state-aware diagnosis `get_quote` can't express on
        its own: a disconnected socket, a connected socket still waiting for
        its first tick, a TWS permission/subscription error, or a contract-
        resolution error each get their own `QuoteAvailability` -- so the UI
        never says "start TWS" while TWS is actually connected (release-
        review finding). Presentation-only: nothing here is persisted."""
        root = root_symbol.upper()
        if not self.is_connected():
            return QuoteStatus(availability=QuoteAvailability.DISCONNECTED)
        with self._lock:
            quote = self._quotes.get(root)
            contract_error = self._contract_errors.get(root)
            quote_error = self._quote_errors.get(root)
        if quote is not None:
            return QuoteStatus(availability=QuoteAvailability.AVAILABLE, quote=quote)
        if contract_error is not None:
            code, message = contract_error
            return QuoteStatus(
                availability=QuoteAvailability.CONTRACT_ERROR, diagnostic_code=code, diagnostic_message=message
            )
        if quote_error is not None:
            code, message = quote_error
            availability = (
                QuoteAvailability.NO_PERMISSION if code in _NO_PERMISSION_CODES
                else QuoteAvailability.CONTRACT_ERROR if code in _CONTRACT_ERROR_CODES
                else QuoteAvailability.UNKNOWN_UNAVAILABLE
            )
            return QuoteStatus(availability=availability, diagnostic_code=code, diagnostic_message=message)
        return QuoteStatus(availability=QuoteAvailability.WAITING)

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:  # noqa: BLE001, S110 -- teardown must never raise into the caller
                pass
        self._connected = False
