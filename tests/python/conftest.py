"""Repo-wide pytest fixtures for `tests/python` (`testpaths` in
`pyproject.toml`).

MARKET REALITY pass safety net. The Agent (Home) page now renders a real
Databento market strip by default (`alpha_agent.ui.market_home`, wired into
`alpha_agent.ui.views.agent.render`). Dozens of pre-existing tests exercise
that page -- directly, or via the full `app.py` entrypoint -- without any
awareness of Databento, because they were written before the Agent page made
any such call.

`alpha_agent.knowledge.connector_support.load_dotenv_if_available` is called
at IMPORT time by several test modules (e.g.
`test_databento_market_data_provider_live.py`). Whenever a local `.env`
carries a real `DATABENTO_API_KEY`, that call leaks the key into
`os.environ` for the REST of the pytest session -- not just that one file
-- because `python-dotenv`'s `load_dotenv()` only ever ADDS variables that
are not already set, and pytest imports every test module during collection
before any test body runs. Without a fixture like this one, that leak turns
every Agent/`app.py`-rendering test that does not explicitly mock
`databento_context` into an unmocked, REAL network call (CLAUDE.md: "no
billable call in unit tests" -- a real run of this repo's own suite measured
single real Databento calls taking 4-11 seconds each, and one that never
returned inside a 20-minute window, most likely several such calls serialized
across many unmocked page renders).

This autouse fixture gives EVERY test a safe, fast, deterministic
NOT_CONNECTED default for `alpha_agent.ui.databento_context` -- the exact
honest-degradation path the app already renders correctly everywhere (N/A,
never a crash; see `test_market_page_handles_not_connected_gracefully`).
Any test that wants real-shaped (fake) market data -- `test_market_ui.py`,
`test_market_home.py` -- defines its OWN, more specific
`monkeypatch.setattr(databento_context, ...)` in its own (function-scoped,
autouse) fixture; pytest instantiates a conftest-level autouse fixture
BEFORE a same-scoped autouse fixture defined in the test module itself, so
that module's assignment simply overrides this default.

This fixture deliberately does NOT request the `monkeypatch` fixture (it
constructs its own `pytest.MonkeyPatch()` and undoes it in a `finally`
instead). Requesting `monkeypatch` here would make IT a dependency of every
test's fixture graph and change fixture setup/teardown ORDER repo-wide --
concretely, it broke `test_databento_ui_context.py`'s own
`_reset_provider_cache` fixture, whose teardown expects `databento_context.
_provider` to already be restored by the TIME it runs (LIFO teardown, and
that ordering shifted once `monkeypatch` became this fixture's dependency
too). Constructing `pytest.MonkeyPatch()` directly sidesteps that entirely.

Tests that construct `DatabentoMarketDataProvider` directly
(`test_databento_market_data_provider.py`,
`test_databento_market_data_provider_live.py`) never go through
`databento_context` at all and are entirely unaffected -- the live file's
own `@pytest.mark.skipif(not _HAS_KEY, ...)` gating still runs for real
exactly as designed whenever a key is genuinely configured.

One file is a deliberate, narrower exception: `test_databento_ui_context.py`
tests `databento_context`'s OWN implementation (that `health`/
`market_snapshot`/etc. correctly delegate to `_provider()` and degrade
honestly on a real exception) by monkeypatching the internal `_provider`
factory, not the public functions this fixture replaces -- applying this
default there would mock away the very code under test. It opts out with
`pytestmark = pytest.mark.real_databento_context` (registered in
`pyproject.toml`), which this fixture checks for and skips.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest


@pytest.fixture(autouse=True)
def _default_databento_context_is_not_connected(request):
    if request.node.get_closest_marker("real_databento_context"):
        yield
        return

    from alpha_agent.marketdata.databento_schemas import DatabentoCapability, DatabentoHealth
    from alpha_agent.ui import databento_context

    def _not_connected_health() -> DatabentoHealth:
        return DatabentoHealth(
            capability=DatabentoCapability.NOT_CONNECTED, dataset="GLBX.MDP3", checked_at=datetime.now(UTC),
            detail=(
                "test default (tests/python/conftest.py): alpha_agent.ui.databento_context is mocked "
                "NOT_CONNECTED for every test unless that test's own fixture overrides it -- a real "
                "DATABENTO_API_KEY present in the test process environment never reaches this default."
            ),
        )

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(databento_context, "health", _not_connected_health)
        mp.setattr(databento_context, "market_snapshot", lambda root: None)
        mp.setattr(databento_context, "resolve_display_contract", lambda root: None)
        mp.setattr(databento_context, "contract_metadata", lambda root: None)
        mp.setattr(databento_context, "recent_ohlcv", lambda root, **kw: None)
        # Checkpoint C (Market Intelligence Data Completion Pass): the
        # Contracts tab renders unconditionally as part of the Market page,
        # so these need the SAME safe default as every other
        # `databento_context` function above -- omitting them reopened
        # exactly the leak this fixture exists to close (a real,
        # ambient DATABENTO_API_KEY from `load_dotenv_if_available()`
        # reaching an unmocked call).
        mp.setattr(databento_context, "contract_ladder", lambda root, **kw: None)
        mp.setattr(databento_context, "term_structure", lambda root, **kw: None)
        mp.setattr(databento_context, "clear_cache", lambda: None)
        yield
    finally:
        mp.undo()


@pytest.fixture(autouse=True)
def _default_anthropic_api_key_is_unset():
    """SAME leak class as `_default_databento_context_is_not_connected`,
    for `ANTHROPIC_API_KEY`: `test_alpha_discovery_live_c5_github_connector.py`
    and `test_databento_market_data_provider_live.py` both call the real
    `load_dotenv_if_available()` at MODULE IMPORT time (needed so their own
    `skipif` conditions see a real key), and pytest imports every test
    module during collection before any test body runs -- so once a local
    `.env` carries a real `ANTHROPIC_API_KEY` (CLAUDE LIVE INTEGRATION V0),
    that key leaks into `os.environ` for the REST of the session, not just
    those two files. Before the `anthropic` package was installed this was
    harmless (`AnthropicClient._ensure_client()` always raised
    `LLMClientUnavailable` first); with it installed, any test that reaches
    live/"Connected Research" mode by default (`alpha_agent.ui.views.
    discover`'s `_anthropic_configured()` check) makes a REAL, billable
    Anthropic API call -- a real run of this repo's own suite hit exactly
    this (a genuine `anthropic.BadRequestError` from the real API). This
    autouse fixture gives every test a safe, deterministic "key absent"
    default; a test that wants to simulate a configured key still can with
    its own `monkeypatch.setenv("ANTHROPIC_API_KEY", ...)`, applied after
    this fixture runs."""
    mp = pytest.MonkeyPatch()
    try:
        mp.delenv("ANTHROPIC_API_KEY", raising=False)
        yield
    finally:
        mp.undo()


@pytest.fixture(autouse=True)
def _default_market_intel_context_is_not_connected(request):
    """SAME safety net as `_default_databento_context_is_not_connected`,
    for Checkpoint E's News tab: `alpha_agent.ui.market_intel_context`
    otherwise hits 4-5 REAL official government endpoints on every unmocked
    Market page render. No `.env` key is needed for these (they are all
    keyless public feeds), so without this default EVERY test rendering the
    Market page would make real network calls."""
    if request.node.get_closest_marker("real_databento_context"):
        yield
        return

    from alpha_agent.ui import market_intel_context

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(market_intel_context, "connector_health", dict)
        mp.setattr(market_intel_context, "recent_news", lambda **kw: ())
        mp.setattr(market_intel_context, "news_count_for_product", lambda root, **kw: 0)
        mp.setattr(market_intel_context, "force_refresh", lambda: None)
        # Checkpoint F: the Scheduled Events section renders unconditionally
        # alongside News -- same safety net.
        mp.setattr(market_intel_context, "event_connector_health", dict)
        mp.setattr(market_intel_context, "upcoming_events", lambda **kw: ())
        mp.setattr(market_intel_context, "next_event_for_product", lambda root, **kw: None)
        mp.setattr(market_intel_context, "next_high_impact_event_for_product", lambda root, **kw: None)
        yield
    finally:
        mp.undo()


@pytest.fixture(autouse=True)
def _research_mandate_store_is_isolated():
    """News Alpha Phase A: the Agent page reads the user's saved
    `ResearchMandate` (`alpha_agent.ui.news_alpha_context.MANDATE_STORE`,
    `data/user_prefs/research_mandate.json`). Without this default, every
    test rendering the Agent page would (a) depend on whatever mandate the
    developer last saved locally -- a Futures-excluding mandate would hide
    every Research Translation button and silently break unrelated tests --
    and (b) could overwrite that real preference file. Every test gets its
    own empty temporary store (= `DEFAULT_MANDATE`); a test that needs a
    specific mandate saves one into it. Same no-`monkeypatch`-dependency
    construction as the fixtures above."""
    import tempfile
    from pathlib import Path

    from alpha_agent.news_alpha.mandate import MandateStore
    from alpha_agent.ui import news_alpha_context

    mp = pytest.MonkeyPatch()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            mp.setattr(news_alpha_context, "MANDATE_STORE", MandateStore(Path(tmp) / "research_mandate.json"))
            yield
        finally:
            mp.undo()


@pytest.fixture(autouse=True)
def _factor_screen_store_is_isolated():
    """News Alpha Phase E: the Agent card reads cached factor screens from
    `candidate_signal_view.SCREEN_STORE` (`data/news_alpha/factor_screens/`).
    Every test gets its own empty temporary store, so a test neither depends
    on screens the developer ran locally nor writes into that cache."""
    import tempfile
    from pathlib import Path

    from alpha_agent.screening.candidate_signal_screen import FactorScreenStore
    from alpha_agent.ui import candidate_signal_view

    mp = pytest.MonkeyPatch()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            mp.setattr(candidate_signal_view, "SCREEN_STORE", FactorScreenStore(Path(tmp) / "factor_screens"))
            yield
        finally:
            mp.undo()


@pytest.fixture(autouse=True)
def _market_snapshot_store_is_isolated():
    """News Alpha Phase G: the Agent card sizes portfolios from cached market
    snapshots (`portfolio_plan_view.SNAPSHOT_STORE`,
    `data/news_alpha/market_snapshots/`). Every test gets its own empty
    temporary store, so a plan never depends on data the developer loaded
    locally and a test never writes into that cache."""
    import tempfile
    from pathlib import Path

    from alpha_agent.portfolio.risk_model import MarketSnapshotStore
    from alpha_agent.ui import portfolio_plan_view

    mp = pytest.MonkeyPatch()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            mp.setattr(portfolio_plan_view, "SNAPSHOT_STORE", MarketSnapshotStore(Path(tmp) / "market_snapshots"))
            yield
        finally:
            mp.undo()


@pytest.fixture(autouse=True)
def _research_thread_store_is_isolated():
    """News Alpha Research Thread workspace: threads persist to
    `data/user_prefs/research_threads.json`. Every test gets its own empty
    temporary store, so a test never depends on (or overwrites) the
    developer's real research threads."""
    import tempfile
    from pathlib import Path

    from alpha_agent.ui import research_thread

    mp = pytest.MonkeyPatch()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            mp.setattr(research_thread, "THREAD_STORE",
                       research_thread.ResearchThreadStore(Path(tmp) / "research_threads.json"))
            yield
        finally:
            mp.undo()


@pytest.fixture(autouse=True)
def _market_intel_cache_is_isolated():
    """Cache hygiene: `market_intel_context._store()` / `_event_store()`
    default to the developer's REAL observational cache
    (`data/market_intelligence/news/*.sqlite`). A test that renders a page or
    seeds an item without its own isolation would otherwise read that real
    cache (making results depend on local state) or WRITE to it -- which is
    exactly how a fixture item (`news_id="N1"`, `https://eia.gov/x`, from an
    earlier, pre-isolation `test_translation_context.py`) ended up in a real
    local cache and then surfaced in the live Agent page. Every test now gets
    its own empty temp stores; a test module's own
    `monkeypatch.setattr(market_intel_context, "_store", ...)` still
    overrides this (module fixtures run after conftest ones)."""
    import tempfile
    from pathlib import Path

    from alpha_agent.market_intel.store import EventStore, NewsStore
    from alpha_agent.ui import market_intel_context

    mp = pytest.MonkeyPatch()
    with tempfile.TemporaryDirectory() as tmp:
        news, events = NewsStore(Path(tmp) / "news.sqlite"), EventStore(Path(tmp) / "events.sqlite")
        try:
            mp.setattr(market_intel_context, "_store", lambda: news)
            mp.setattr(market_intel_context, "_event_store", lambda: events)
            yield
        finally:
            mp.undo()
