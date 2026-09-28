"""Release UX -- agent-first navigation + IBKR delayed market-context tests.

No test in this module makes a network call. `alpha_agent.marketdata` is
disabled by default (`configs/market_data.yaml`); every test that needs a
"connected" provider uses a small in-process fake implementing the
`MarketDataProvider` protocol, never the real `IBKRDelayedMarketDataProvider`
against an actual socket -- a real IBKR connection is never required for
this suite.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from alpha_agent.marketdata.config import IBKRConnectionConfig, load_config
from alpha_agent.marketdata.contracts import APPROVED_ROOTS, contract_for
from alpha_agent.marketdata.ibkr_provider import IBKRDelayedMarketDataProvider
from alpha_agent.marketdata.quote import FeedState, Quote, QuoteAvailability, QuoteStatus

REPO_ROOT = Path(__file__).resolve().parents[2]

_FORBIDDEN_TRADING_API_SUBSTRINGS = (
    "placeorder", "cancelorder", "modifyorder", "reqaccountupdates",
    "reqaccountsummary", "reqpositions", "reqopenorders", "reqallopenorders",
    "reqexecutions", "reqglobalcancel", "execdetails", "reqaccountupdatesmulti",
)
_FORBIDDEN_REGISTRY_WRITE_SUBSTRINGS = (
    "insert_experiment", "record_failure", "record_lineage", "apply_bundle",
    "record_attempt", "insert or replace",
)


# ---------------------------------------------------------------------------
# config / offline safety -- no credentials, disabled by default, fails
# closed with no network call
# ---------------------------------------------------------------------------


def test_default_config_is_disabled_and_carries_no_credential_field():
    cfg = load_config()
    assert cfg.enabled is False
    for bad_field in ("username", "password", "api_key", "token", "secret"):
        assert not hasattr(cfg, bad_field)


def test_committed_config_file_is_disabled_and_carries_no_credentials():
    raw = (REPO_ROOT / "configs" / "market_data.yaml").read_text(encoding="utf-8").lower()
    assert "enabled: false" in raw
    for bad in ("password", "username", "secret", "api_key", "token"):
        assert bad not in raw


def test_disabled_provider_never_connects_and_returns_no_quote():
    provider = IBKRDelayedMarketDataProvider(IBKRConnectionConfig(enabled=False))
    assert provider.connect() is False
    assert provider.is_connected() is False
    assert provider.feed_state() is FeedState.DISCONNECTED
    assert provider.get_quote("CL") is None


def test_enabled_provider_fails_closed_when_ibapi_import_is_unavailable(monkeypatch):
    """Environment-INDEPENDENT proof of the graceful fallback -- passes
    identically whether or not `ibapi` is physically installed in the
    current venv. A prior version of this test asserted `ibapi` is not
    installed at all, which broke as soon as a workstation legitimately
    installs the optional `[ibkr]` extra for real TWS integration work (this
    repo's own dev workflow now does exactly that -- see
    `docs/STREAMLIT_RESEARCH_INTERFACE.md`); a correct optional-dependency
    test must simulate the dependency being unavailable, not require the
    developer's real environment to uninstall it.

    Simulates the import failure via Python's own import machinery:
    `sys.modules[name] = None` makes any `import`/`from ... import` of that
    exact name raise `ImportError` deterministically (a documented CPython
    behavior, not a hack on `alpha_agent` code) -- `connect()`'s lazy
    `from ibapi.client import EClient` / `from ibapi.wrapper import
    EWrapper` therefore fails exactly as it would if `ibapi` were genuinely
    absent, with zero change to `alpha_agent.marketdata` production code."""
    import sys
    monkeypatch.setitem(sys.modules, "ibapi.client", None)
    monkeypatch.setitem(sys.modules, "ibapi.wrapper", None)
    with pytest.raises(ImportError):
        import ibapi.client  # noqa: F401

    provider = IBKRDelayedMarketDataProvider(
        IBKRConnectionConfig(enabled=True, connect_timeout_seconds=0.2)
    )
    assert provider.connect() is False
    assert provider.is_connected() is False
    assert provider.feed_state() is FeedState.DISCONNECTED
    assert provider.get_quote("CL") is None
    assert provider.quote_status("CL").availability == QuoteAvailability.DISCONNECTED


def test_all_five_approved_roots_map_to_a_market_data_only_contract():
    for root in APPROVED_ROOTS:
        spec = contract_for(root)
        assert spec is not None
        # CONTFUT is IBKR's market-data-only continuous-futures proxy -- it
        # carries no conId for order routing, matching this package having
        # no order method to route one through in the first place.
        assert spec.sec_type == "CONTFUT"


def test_unknown_root_has_no_contract_mapping():
    assert contract_for("ZZ") is None


def test_ibkr_delayed_market_data_type_constant_is_the_documented_value():
    """TWS API: reqMarketDataType(3) == delayed. Pinned as a named constant
    so a future edit can't silently drift to 1 (live) or 4 (delayed-frozen)
    without this test failing."""
    from alpha_agent.marketdata.ibkr_provider import _DELAYED_MARKET_DATA_TYPE
    assert _DELAYED_MARKET_DATA_TYPE == 3


def test_subscribe_all_never_passes_contfut_to_reqmktdata(monkeypatch):
    """Regression test for a real bug found against a live IBKR TWS Demo
    instance (server version 157): `reqMktData` REJECTS a `CONTFUT` contract
    outright with TWS error 321 ("Error validating request... Please enter a
    valid security type") -- CONTFUT is only a valid `secType` for
    `reqContractDetails`/`reqHistoricalData`, never for a live/delayed
    streaming subscription. The provider must resolve each root's CONTFUT to
    its real current front-month FUT contract via `reqContractDetails`
    FIRST, then `reqMktData` only that resolved FUT contract. A root that
    never resolves must be skipped, never subscribed with a guessed/stale
    expiry (no fabrication).

    Drives the provider's REAL `connect()`/`_subscribe_all()` through a fake
    `ibapi.client.EClient` (no real socket) so this runs deterministically
    offline; only `ibapi` itself must be installed (the optional `[ibkr]`
    extra) for the real `EWrapper` base class the provider subclasses --
    skips gracefully otherwise, same convention as every other
    optional-dependency-gated test in this suite."""
    pytest.importorskip("ibapi")
    from types import SimpleNamespace

    import ibapi.client
    from alpha_agent.marketdata.contracts import APPROVED_ROOTS

    reqcontractdetails_calls: list[tuple[int, str, str]] = []
    reqmktdata_calls: list[tuple[int, str, str, str]] = []

    class _FakeClient:
        def __init__(self, wrapper) -> None:
            self._wrapper = wrapper
            self._connected = False

        def connect(self, host, port, client_id) -> None:
            self._connected = True
            self._wrapper.nextValidId(1)  # simulate an instant handshake

        def run(self) -> None:
            pass  # no real message loop; every callback above is driven synchronously

        def isConnected(self) -> bool:
            return self._connected

        def reqMarketDataType(self, market_data_type: int) -> None:
            pass

        def reqContractDetails(self, req_id, contract) -> None:
            reqcontractdetails_calls.append((req_id, contract.symbol, contract.secType))
            # Simulate TWS resolving every root except ZN, to prove an
            # unresolved root is skipped rather than fabricated below.
            if contract.symbol != "ZN":
                cd = SimpleNamespace(contract=SimpleNamespace(
                    lastTradeDateOrContractMonth="20261218", localSymbol=f"{contract.symbol}Z6",
                ))
                self._wrapper.contractDetails(req_id, cd)
            self._wrapper.contractDetailsEnd(req_id)

        def reqMktData(self, req_id, contract, *_args) -> None:
            reqmktdata_calls.append(
                (req_id, contract.symbol, contract.secType, contract.lastTradeDateOrContractMonth)
            )

        def disconnect(self) -> None:
            self._connected = False

    monkeypatch.setattr(ibapi.client, "EClient", _FakeClient)

    config = IBKRConnectionConfig(enabled=True, request_timeout_seconds=0.5)
    provider = IBKRDelayedMarketDataProvider(config)
    assert provider.connect() is True
    assert provider.is_connected() is True
    assert provider.feed_state() is FeedState.DELAYED

    # Resolution phase: every approved root queried via CONTFUT (unchanged).
    assert {r for _, _, sec_type in reqcontractdetails_calls for r in [sec_type]} == {"CONTFUT"}
    assert {sym for _, sym, _ in reqcontractdetails_calls} == set(APPROVED_ROOTS)

    # Subscription phase: CONTFUT must NEVER reach reqMktData -- only a real
    # FUT contract carrying the resolved expiry.
    assert reqmktdata_calls, "reqMktData was never called at all"
    for _req_id, symbol, sec_type, expiry in reqmktdata_calls:
        assert sec_type == "FUT", f"{symbol} subscribed with secType={sec_type!r}, not FUT"
        assert sec_type != "CONTFUT"
        assert expiry, f"{symbol} subscribed with no lastTradeDateOrContractMonth"

    # ZN never resolved in this simulation -- it must be skipped entirely,
    # never subscribed with a guessed/fallback expiry.
    subscribed_symbols = {symbol for _, symbol, _, _ in reqmktdata_calls}
    assert "ZN" not in subscribed_symbols
    assert subscribed_symbols == set(APPROVED_ROOTS) - {"ZN"}


def _connect_provider_with_fake_client(monkeypatch):
    """Shared harness for the two tests below: drives the REAL
    `IBKRDelayedMarketDataProvider.connect()`/`_subscribe_all()` through a
    fake `ibapi.client.EClient` (no real socket, every approved root
    resolves), then hands back the constructed real `EWrapper` instance so a
    test can drive `tickPrice`/`tickSize`/`error` directly -- proving the
    provider's OWN callback-handling code, never a hand-built `Quote`."""
    pytest.importorskip("ibapi")
    from types import SimpleNamespace

    import ibapi.client

    captured: dict[str, object] = {}
    reqmktdata_calls: list[tuple[int, str, str, str]] = []

    class _FakeClient:
        def __init__(self, wrapper) -> None:
            self._wrapper = wrapper
            captured["wrapper"] = wrapper
            self._connected = False

        def connect(self, host, port, client_id) -> None:
            self._connected = True
            self._wrapper.nextValidId(1)

        def run(self) -> None:
            pass

        def isConnected(self) -> bool:
            return self._connected

        def reqMarketDataType(self, market_data_type: int) -> None:
            pass

        def reqContractDetails(self, req_id, contract) -> None:
            cd = SimpleNamespace(contract=SimpleNamespace(
                lastTradeDateOrContractMonth="20261218", localSymbol=f"{contract.symbol}Z6",
            ))
            self._wrapper.contractDetails(req_id, cd)
            self._wrapper.contractDetailsEnd(req_id)

        def reqMktData(self, req_id, contract, *_args) -> None:
            reqmktdata_calls.append(
                (req_id, contract.symbol, contract.secType, contract.lastTradeDateOrContractMonth)
            )

        def disconnect(self) -> None:
            self._connected = False

    monkeypatch.setattr(ibapi.client, "EClient", _FakeClient)
    config = IBKRConnectionConfig(enabled=True, request_timeout_seconds=0.5)
    provider = IBKRDelayedMarketDataProvider(config)
    assert provider.connect() is True
    req_id_by_root = {symbol: req_id for req_id, symbol, _, _ in reqmktdata_calls}
    return provider, captured["wrapper"], req_id_by_root


def test_delayed_tick_callbacks_populate_quote_fields_correctly(monkeypatch):
    """Regression test: audits the real delayed tick-type mapping (IBKR
    DELAYED tick ids 66-76, `reqMarketDataType(3)`) by driving the
    provider's REAL `tickPrice`/`tickSize` callbacks -- not by instantiating
    a mocked `Quote` directly -- with exactly the tick ids/values a real TWS
    Demo was observed to send (AAPL smoke: price 66/67/68, size 69/70/71/74).
    Also proves IBKR's 2026-era `decimal.Decimal` size quantities (the
    fractional-share/decimal-size API) don't break the typed `float`
    conversion: DELAYED_ASK_SIZE is fed a `Decimal` below."""
    from decimal import Decimal

    provider, wrapper, req_id_by_root = _connect_provider_with_fake_client(monkeypatch)
    req_id = req_id_by_root["ES"]

    wrapper.tickPrice(req_id, 66, 71.20, None)     # DELAYED_BID
    wrapper.tickPrice(req_id, 67, 71.25, None)     # DELAYED_ASK
    wrapper.tickPrice(req_id, 68, 71.23, None)     # DELAYED_LAST
    wrapper.tickSize(req_id, 69, 5)                # DELAYED_BID_SIZE
    wrapper.tickSize(req_id, 70, Decimal(7))     # DELAYED_ASK_SIZE -- Decimal input
    wrapper.tickSize(req_id, 71, 3)                # DELAYED_LAST_SIZE
    wrapper.tickSize(req_id, 74, 12345)            # DELAYED_VOLUME

    quote = provider.get_quote("ES")
    assert quote is not None
    assert quote.bid == 71.20
    assert quote.ask == 71.25
    assert quote.last == 71.23
    assert quote.bid_size == 5.0
    assert quote.ask_size == 7.0  # Decimal("7") converted cleanly to float
    assert quote.last_size == 3.0
    assert quote.volume == 12345.0
    assert quote.feed_state is FeedState.DELAYED
    assert quote.provider == "IBKR"

    status = provider.quote_status("ES")
    assert status.availability == QuoteAvailability.AVAILABLE
    assert status.quote == quote


def test_quote_status_classifies_waiting_permission_and_contract_errors(monkeypatch):
    """Regression test for the connection-state-vs-quote-availability
    correction: drives the provider's REAL `error()` callback with the exact
    TWS codes observed on a live smoke (200 = contract/security-definition
    error, 354 = market-data permission/subscription error, 2104 = a purely
    informational farm-connection notice) and proves each buckets into the
    right `QuoteAvailability` for presentation, that a purely informational
    code never flips a root into an error state, and that a real tick always
    takes priority over an earlier recorded error."""
    provider, wrapper, req_id_by_root = _connect_provider_with_fake_client(monkeypatch)

    # No tick, no error yet -- CONNECTED and waiting, never DISCONNECTED.
    status = provider.quote_status("CL")
    assert status.availability == QuoteAvailability.WAITING
    assert status.quote is None
    assert status.diagnostic_code is None

    # A real market-data permission/subscription error.
    wrapper.error(req_id_by_root["NQ"], 354, "Requested market data is not subscribed.")
    status = provider.quote_status("NQ")
    assert status.availability == QuoteAvailability.NO_PERMISSION
    assert status.diagnostic_code == 354
    assert status.diagnostic_message == "Requested market data is not subscribed."
    assert status.quote is None

    # A real contract/security-definition error on the market-data request.
    wrapper.error(req_id_by_root["GC"], 200, "No security definition has been found for the request")
    status = provider.quote_status("GC")
    assert status.availability == QuoteAvailability.CONTRACT_ERROR
    assert status.diagnostic_code == 200

    # A purely informational farm-connection notice must NOT flip ES into
    # any error state.
    wrapper.error(req_id_by_root["ES"], 2104, "Market data farm connection is OK:usfarm")
    status = provider.quote_status("ES")
    assert status.availability == QuoteAvailability.WAITING
    assert status.diagnostic_code is None

    # A real tick always wins over an earlier recorded error for the SAME root.
    wrapper.error(req_id_by_root["ZN"], 354, "Requested market data is not subscribed.")
    assert provider.quote_status("ZN").availability == QuoteAvailability.NO_PERMISSION
    wrapper.tickPrice(req_id_by_root["ZN"], 68, 108.5, None)
    status = provider.quote_status("ZN")
    assert status.availability == QuoteAvailability.AVAILABLE
    assert status.quote is not None
    assert status.quote.last == 108.5


# ---------------------------------------------------------------------------
# read-only boundary -- no order/account method, no registry write method,
# anywhere in the new package or its UI boundary
# ---------------------------------------------------------------------------


def test_marketdata_package_has_no_order_or_account_trading_method():
    pkg_dir = REPO_ROOT / "python" / "alpha_agent" / "marketdata"
    for path in sorted(pkg_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for bad in _FORBIDDEN_TRADING_API_SUBSTRINGS:
            assert bad not in text, f"{path.name}: forbidden trading/account API reference {bad!r}"


def test_market_context_module_never_calls_a_registry_write_method():
    text = (REPO_ROOT / "python" / "alpha_agent" / "ui" / "market_context.py").read_text(encoding="utf-8").lower()
    for bad in _FORBIDDEN_REGISTRY_WRITE_SUBSTRINGS:
        assert bad not in text
    for bad in _FORBIDDEN_TRADING_API_SUBSTRINGS:
        assert bad not in text


def test_agent_view_never_calls_a_registry_write_method():
    text = (REPO_ROOT / "python" / "alpha_agent" / "ui" / "views" / "agent.py").read_text(encoding="utf-8").lower()
    for bad in _FORBIDDEN_REGISTRY_WRITE_SUBSTRINGS:
        assert bad not in text


def test_marketdata_provider_protocol_has_no_order_or_write_method():
    import inspect

    from alpha_agent.marketdata.provider import MarketDataProvider
    members = [name for name, _ in inspect.getmembers(MarketDataProvider) if not name.startswith("_")]
    for name in members:
        lowered = name.lower()
        assert "order" not in lowered
        assert "account" not in lowered
        assert "write" not in lowered and "insert" not in lowered


# ---------------------------------------------------------------------------
# quote strip rendering -- mocked provider, DELAYED never rendered as LIVE
# ---------------------------------------------------------------------------


def _sample_quote(root: str = "CL") -> Quote:
    return Quote(
        root_symbol=root, contract=root, last=71.23, bid=71.20, ask=71.25,
        bid_size=5, ask_size=7, volume=12345, exchange_timestamp=None,
        received_timestamp=datetime.now(UTC), provider="IBKR", feed_state=FeedState.DELAYED,
    )


class _FakeConnectedProvider:
    def __init__(self, quote: Quote) -> None:
        self._quote = quote

    def connect(self) -> bool:
        return True

    def is_connected(self) -> bool:
        return True

    def feed_state(self) -> FeedState:
        return FeedState.DELAYED

    def get_quote(self, root_symbol: str) -> Quote | None:
        return self._quote if root_symbol == self._quote.root_symbol else None

    def quote_status(self, root_symbol: str) -> QuoteStatus:
        quote = self.get_quote(root_symbol)
        if quote is not None:
            return QuoteStatus(availability=QuoteAvailability.AVAILABLE, quote=quote)
        return QuoteStatus(availability=QuoteAvailability.WAITING)

    def disconnect(self) -> None:
        pass


class _FakeStatusProvider:
    """A connected fake exposing an explicit `QuoteStatus` -- for testing
    `render_quote_strip`'s CONNECTED-but-no-quote states (WAITING /
    NO_PERMISSION / CONTRACT_ERROR / UNKNOWN_UNAVAILABLE) distinctly from
    both DISCONNECTED and a real available quote."""

    def __init__(self, status: QuoteStatus) -> None:
        self._status = status

    def connect(self) -> bool:
        return True

    def is_connected(self) -> bool:
        return True

    def feed_state(self) -> FeedState:
        return FeedState.DELAYED

    def get_quote(self, root_symbol: str) -> Quote | None:
        return self._status.quote

    def quote_status(self, root_symbol: str) -> QuoteStatus:
        return self._status

    def disconnect(self) -> None:
        pass


def test_quote_strip_renders_delayed_quote_and_never_says_live(monkeypatch):
    pytest.importorskip("streamlit")
    from alpha_agent.ui import market_context
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(market_context, "is_market_context_enabled", lambda: True)
    monkeypatch.setattr(market_context, "_provider", lambda: _FakeConnectedProvider(_sample_quote("CL")))

    at = AppTest.from_string(
        "from alpha_agent.ui import market_context\n"
        "import streamlit as st\n"
        "st.session_state['market_selected_root'] = 'CL'\n"
        "market_context.render_quote_strip('CL')\n"
    )
    at.run(timeout=30)
    assert not list(at.exception)
    html = " ".join(md.value for md in at.markdown)
    assert "DELAYED" in html
    assert "IBKR" in html
    assert "71.23" in html
    assert "LIVE" not in html


def test_quote_strip_shows_honest_disconnected_state_never_a_fabricated_quote(monkeypatch):
    pytest.importorskip("streamlit")
    from alpha_agent.marketdata.ibkr_provider import IBKRDelayedMarketDataProvider
    from alpha_agent.ui import market_context
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(market_context, "is_market_context_enabled", lambda: True)
    monkeypatch.setattr(
        market_context, "_provider",
        lambda: IBKRDelayedMarketDataProvider(IBKRConnectionConfig(enabled=False)),
    )

    at = AppTest.from_string(
        "from alpha_agent.ui import market_context\n"
        "market_context.render_quote_strip('CL')\n"
    )
    at.run(timeout=30)
    assert not list(at.exception)
    html = " ".join(md.value for md in at.markdown)
    assert "DISCONNECTED" in html
    assert "Start TWS or IB Gateway" in html
    assert "LIVE" not in html
    # Release-review correction: DISCONNECTED must never overlap with either
    # CONNECTED-but-no-quote state below -- they are mutually exclusive.
    # (Note: "CONNECTED" is deliberately not asserted absent here -- it is a
    # substring of "DISCONNECTED" itself.)
    assert "WAITING FOR QUOTE" not in html
    assert "QUOTE UNAVAILABLE" not in html


def test_quote_strip_shows_connected_waiting_state_never_says_start_tws(monkeypatch):
    """Release-review correction: a socket that IS connected but has not yet
    received a tick for the selected root must say so plainly -- it must
    NEVER render the same "start TWS" banner as an actually-disconnected
    socket, and must never fabricate a quote in the meantime."""
    pytest.importorskip("streamlit")
    from alpha_agent.ui import market_context
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(market_context, "is_market_context_enabled", lambda: True)
    monkeypatch.setattr(
        market_context, "_provider",
        lambda: _FakeStatusProvider(QuoteStatus(availability=QuoteAvailability.WAITING)),
    )

    at = AppTest.from_string(
        "from alpha_agent.ui import market_context\n"
        "market_context.render_quote_strip('CL')\n"
    )
    at.run(timeout=30)
    assert not list(at.exception)
    html = " ".join(md.value for md in at.markdown)
    assert "CONNECTED" in html
    assert "WAITING FOR QUOTE" in html
    assert "DISCONNECTED" not in html
    assert "Start TWS" not in html
    assert "LIVE" not in html


def test_quote_strip_shows_connected_quote_unavailable_with_real_diagnostic(monkeypatch):
    """Release-review correction: a TWS-reported permission/subscription
    error must surface as CONNECTED + QUOTE UNAVAILABLE with the exact TWS
    code/message -- never as DISCONNECTED, never a fabricated quote, and the
    diagnostic is never invented (it is whatever the fake -- standing in for
    a real `error()` callback -- actually supplied)."""
    pytest.importorskip("streamlit")
    from alpha_agent.ui import market_context
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(market_context, "is_market_context_enabled", lambda: True)
    monkeypatch.setattr(
        market_context, "_provider",
        lambda: _FakeStatusProvider(QuoteStatus(
            availability=QuoteAvailability.NO_PERMISSION, diagnostic_code=354,
            diagnostic_message="Requested market data is not subscribed.",
        )),
    )

    at = AppTest.from_string(
        "from alpha_agent.ui import market_context\n"
        "market_context.render_quote_strip('CL')\n"
    )
    at.run(timeout=30)
    assert not list(at.exception)
    html = " ".join(md.value for md in at.markdown)
    assert "CONNECTED" in html
    assert "QUOTE UNAVAILABLE" in html
    assert "354" in html
    assert "not subscribed" in html
    assert "DISCONNECTED" not in html
    assert "Start TWS" not in html
    assert "LIVE" not in html


def test_quote_strip_renders_nothing_when_market_context_disabled(monkeypatch):
    pytest.importorskip("streamlit")
    from alpha_agent.ui import market_context
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(market_context, "is_market_context_enabled", lambda: False)

    at = AppTest.from_string(
        "from alpha_agent.ui import market_context\n"
        "market_context.render_quote_strip('CL')\n"
    )
    at.run(timeout=30)
    assert not list(at.exception)
    assert not list(at.markdown)  # nothing rendered at all -- the offline default stays uncluttered


def test_observational_context_note_is_tagged_and_never_scientific(monkeypatch):
    from alpha_agent.ui import market_context
    monkeypatch.setattr(market_context, "get_quote", lambda root: _sample_quote(root))
    note = market_context.observational_context_note("CL")
    assert note is not None
    assert note.startswith("OBSERVATIONAL_CONTEXT_ONLY")
    assert "provider=IBKR" in note
    assert "feed=DELAYED" in note
    assert "not scientific evidence" in note


def test_observational_context_note_is_none_without_a_quote(monkeypatch):
    from alpha_agent.ui import market_context
    monkeypatch.setattr(market_context, "get_quote", lambda root: None)
    assert market_context.observational_context_note("CL") is None


def test_no_test_in_this_module_touches_the_real_ibkr_provider_singleton():
    """Sanity check on the fixtures above: `market_context._provider` was
    monkeypatched in every test that needed a "connected" state, so the
    real, module-level cached `IBKRDelayedMarketDataProvider` singleton was
    never constructed with `enabled=True` by this test module."""
    from alpha_agent.ui import market_context
    # calling with the real (disabled) config must still be side-effect-free
    assert market_context.get_quote("CL") is None


# ---------------------------------------------------------------------------
# Agent page -- default page, real pipeline, typed transcript cards
# ---------------------------------------------------------------------------


def test_agent_page_renders_empty_state_with_no_conversation(tmp_path, monkeypatch):
    """Agent Experience Consolidation campaign (section 13): a fresh Agent
    page renders no giant empty "AI Research Agent" / "No conversation yet"
    card -- the page should feel clean, not like an idle engineering module.
    The composer's own placeholder text invites the first question instead."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    html = " ".join(md.value for md in at.markdown)
    assert "No conversation yet" not in html
    assert "AI Research Agent" not in html
    assert not at.session_state["agent_transcript"]
    text_inputs = {ti.key: ti.placeholder for ti in at.text_input}
    assert text_inputs["agent-ask-input"] == "Ask about current opportunities, markets, or research..."


def test_agent_page_send_runs_the_real_pipeline_and_produces_typed_cards():
    """Runs the real offline pipeline (ResearchAgent -> failure-memory check
    -> StrategyCompilerAgent -> registry evidence lookup) exactly as
    research.py/strategies.py already do, and checks every typed transcript
    card actually rendered. `tsmom_nq` is one of the four demo scenarios with
    a real, existing registry verdict (a known prior REJECT), so this also
    exercises the exact-fingerprint registry-evidence match end to end."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    at.selectbox(key="agent-scenario-select").select(
        "Multi-week time-series momentum in NQ"
    ).run(timeout=90)
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception)

    transcript = at.session_state["agent_transcript"]
    types = [t["type"] for t in transcript]
    # runtime-integration release, section 3: failure memory is looked up
    # BEFORE the hypothesis is proposed (it is placed into the ResearchContext
    # the agent actually sees), not after. True Inline Research Workspace
    # pass (task spec section 3): a `research_workspace` marker turn is
    # appended right after the user's objective -- the transcript position
    # the Research Workspace artifact (composer/actions/result) renders at.
    assert types == [
        "user", "research_workspace", "failure_memory", "hypothesis", "compiled", "evidence", "suggestion",
    ]
    fm_turn = next(t for t in transcript if t["type"] == "failure_memory")
    assert fm_turn["pre_proposal"] is True
    assert any(
        e["query"].get("strategy_family") == "tsmom" and e["query"].get("root_symbol") == "NQ"
        for e in fm_turn["data"]
    )

    html = " ".join(md.value for md in at.markdown)
    assert "HYPOTHESIS" in html
    # Compact-by-default (product refactor): the primary transcript shows the
    # "PRIOR RESEARCH MEMORY" digest, never the old full-dump heading.
    assert "PRIOR RESEARCH MEMORY" in html
    assert "FAILURE MEMORY" not in html
    assert "COMPILATION" in html
    assert "EXECUTION TARGET" in html
    assert "PRIOR REGISTRY EVIDENCE" in html

    evidence_turn = next(t for t in transcript if t["type"] == "evidence")
    assert evidence_turn["data"]["match_type"] == "exact_fingerprint"
    assert evidence_turn["data"]["result"]["headline_verdict"] == "REJECT"


def test_agent_page_novel_blueprint_scenario_shows_honest_not_yet_evaluated():
    """The one demo scenario with no Phase 11 template (a novel blueprint)
    has no prior registry evidence -- the evidence card must say so plainly,
    never fabricate a backtest result."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    at.selectbox(key="agent-scenario-select").select(
        "Trend conditioned on a volatility regime in NQ (novel blueprint)"
    ).run(timeout=90)
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception)

    transcript = at.session_state["agent_transcript"]
    evidence_turn = next(t for t in transcript if t["type"] == "evidence")
    assert evidence_turn["data"] is None
    html = " ".join(md.value for md in at.markdown)
    assert "NOT YET EVALUATED" in html


def test_agent_clear_conversation_resets_the_transcript():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    at.button(key="agent-send").click().run(timeout=90)
    assert at.session_state["agent_transcript"]
    at.button(key="agent-clear").click().run(timeout=90)
    assert at.session_state["agent_transcript"] == []


def test_agent_page_still_works_with_ibkr_disabled_and_no_network(monkeypatch):
    """Section 5/9's "the rest of the application must continue working
    offline" -- IBKR disabled (the default) end to end through the Agent
    page's real pipeline run."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from alpha_agent.ui import market_context
    from streamlit.testing.v1 import AppTest
    assert market_context.is_market_context_enabled() is False

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception)


# ---------------------------------------------------------------------------
# navigation / rename
# ---------------------------------------------------------------------------


def test_dashboard_module_exists_and_renders(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.dashboard import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)


def test_app_navigation_lists_the_research_journey_first_then_ask_then_dashboard():
    """Updated for the Research Thread workspace: News is the default landing,
    the Agent conversation is the Ask tool, and Dashboard stays a hidden page."""
    text = Path(REPO_ROOT / "python" / "alpha_agent" / "ui" / "app.py").read_text(encoding="utf-8")
    news_idx = text.index('title="News"')
    ask_idx = text.index('title="Ask"')
    dashboard_idx = text.index('title="Dashboard"')
    assert news_idx < ask_idx < dashboard_idx
    assert 'st.Page(agent.render, title="Ask"' in text
    assert "default=True" in text.split('title="News"')[1].split("\n")[0]