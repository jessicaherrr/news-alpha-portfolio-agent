"""HOME TERMINAL + RUNTIME CONNECTIVITY pass -- shared, market-OBSERVATION-
plane rendering for the "real futures market" experience on the Agent/Home
and Market pages: a compact ticker-card strip, a Selected Market hero, a
price/volume chart paired with a Core Metrics panel, and a visual Market
Context row. Extracted so neither page re-implements the same rendering
(and so the two can never silently disagree about what "freshness" or
"realized volatility" means).

Every value here is read through `alpha_agent.ui.databento_context` -- the
sole UI boundary into the read-only Databento provider -- or computed as a
pure descriptive statistic over already-fetched real bars. Nothing here is a
trading signal, a feature, or scientific evidence; nothing here enters Fast
Screen, strict validation, Research Promise, a scientific verdict, or the
2025 holdout (see `databento_context`'s own boundary docstring and
`tests/python/test_release_market_plane_isolation.py`).

FRESHNESS TRUTH: this module never claims "Live from Databento" or
"15-minute delayed" -- it only ever renders whatever
`alpha_agent.marketdata.databento_schemas.DatabentoCapability` the
provider's own free `metadata` probe actually proved
(`databento_context.health()`), together with the exact latest-observation
timestamp and a computed age. "Last refreshed" (this session's own cache
time), "Latest observation" (the data's own timestamp), and "Data age" stay
three distinct, separately labeled concepts.

CACHE-FIRST FIRST LOAD: a UI-level session cache sits in front of
`databento_context` so a bare Streamlit rerun never re-issues a provider
call for every approved root on every script execution -- only once the
cached entry has aged past `DatabentoMarketDataConfig.
ui_refresh_interval_seconds` does the next render call through to the
provider again. This is layered ON TOP OF, not instead of, the provider's
own request-level TTL cache (`databento_provider._TTLCache`).

PRIORITIZED FIRST PAINT (HOME TERMINAL pass, mission section 15):
`render_home_terminal` fetches the SELECTED root's own snapshot/OHLCV
first -- before the other four -- and reserves the ticker strip's position
at the TOP of the page via `st.container()` while filling it LAST, after
the hero/chart/metrics have already been written. Streamlit streams each
element to the browser as soon as it is produced regardless of which
container it belongs to, so this changes WHEN things appear (selected
market first) without changing WHERE they appear (strip still on top).

PRODUCT DISPLAY CACHE (mission section 14): deliberately NOT implemented as
a new on-disk cache. The session-level cache above plus the prioritized
fetch order already bring a cold Agent-page render's useful content
(selected market hero + chart) to the screen within one real round trip
(~4-11s measured against the real entitlement), with the remaining four
ticker cards streaming in shortly after (~15-30s for the whole page,
concurrent). A persistent `data/cache/market_ui/` would only shave that
same handful of seconds off an occasional cold process restart, at the
cost of real complexity this pass chose not to take on: multi-session
concurrent-write safety, a second staleness concept to keep visually
distinct from "current," and a new artifact surface to keep scientifically
quarantined. If a future pass finds the cold-load latency still
objectionable, add it then -- session-only caching stays the simpler,
safer default until it is proven necessary.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

import streamlit as st

from alpha_agent.marketdata.databento_config import load_databento_config
from alpha_agent.marketdata.databento_schemas import (
    UNAVAILABLE_CAPABILITIES,
    DatabentoHealth,
    MarketSnapshot,
    RecentOhlcvResult,
)
from alpha_agent.ui import charts, charts_market, components, databento_context, palette, services

TIMEFRAMES = ("1m", "5m", "15m", "1h", "1d")

#: The SAME 1h lookback `DatabentoMarketDataProvider.get_market_snapshot`
#: fetches internally to compute session/change -- every OTHER "1h" real-bar
#: read on this page (realized volatility, sparklines, Market Context)
#: intentionally asks for exactly this many bars too, so it lands on the
#: SAME provider request-level cache key `market_snapshot` already
#: populated (never re-fetch what a page already just fetched, real cost or
#: not). Changing this number re-introduces a duplicate real fetch.
SNAPSHOT_LOOKBACK_BARS = 72

#: How many of the most recent 1h closes feed a ticker card's sparkline --
#: bounded well below SNAPSHOT_LOOKBACK_BARS so the shape stays legible at
#: card size; never a second, separately-fetched window.
_SPARKLINE_BARS = 24

#: Timeframes that are a real, honestly-labeled RESAMPLE of 1-minute bars,
#: never a separate vendor grain (see `databento_provider._TIMEFRAME_SCHEMA`).
_RESAMPLED_TIMEFRAMES = frozenset({"5m", "15m"})

_CACHE_KEY = "mh-cache"  # session_state[_CACHE_KEY][cache_key] = (monotonic_ts, wall_clock_dt, value)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# UI-session cache-first plumbing
# ---------------------------------------------------------------------------


def _refresh_interval_seconds() -> float:
    try:
        return load_databento_config().ui_refresh_interval_seconds
    except Exception:  # noqa: BLE001 -- a bad/missing config must never crash the page
        return 90.0


def _cached(cache_key: str, fetch_fn: Callable[[], T], *, ttl_seconds: float | None = None) -> tuple[T, datetime]:
    """Return `(value, last_refreshed_at)`, reusing a still-fresh session-level
    entry when one exists -- see module docstring. `fetch_fn` is called at
    most once per render, and only when the cache is cold or stale.
    ``ttl_seconds`` overrides the default UI-refresh interval -- Section 34
    (Checkpoint C) uses a much longer one for contract definitions, which
    change far more slowly than a price."""
    store: dict[str, tuple[float, datetime, Any]] = st.session_state.setdefault(_CACHE_KEY, {})
    now_mono = time.monotonic()
    interval = ttl_seconds if ttl_seconds is not None else _refresh_interval_seconds()
    hit = store.get(cache_key)
    if hit is not None and (now_mono - hit[0]) <= interval:
        return hit[2], hit[1]
    value = fetch_fn()
    wall_clock = datetime.now(UTC)
    store[cache_key] = (now_mono, wall_clock, value)
    return value, wall_clock


def force_refresh() -> None:
    """The explicit "Refresh" action -- clears BOTH this UI-level session
    cache and the provider's own request cache, so the very next render is
    guaranteed to re-query (subject to the provider's own cost gate). Never
    called automatically on a bare page render/rerun."""
    st.session_state.pop(_CACHE_KEY, None)
    databento_context.clear_cache()


def get_health_cached() -> tuple[DatabentoHealth, datetime]:
    return _cached("health", databento_context.health)


def get_snapshot_cached(root: str) -> tuple[MarketSnapshot | None, datetime]:
    return _cached(f"snapshot:{root}", lambda: databento_context.market_snapshot(root))


def get_snapshots_for_universe(universe: tuple[str, ...]) -> dict[str, MarketSnapshot | None]:
    """The ticker strip's batch read -- every root's snapshot, fetching only
    whatever is NOT already cache-fresh, and fetching those CONCURRENTLY.
    Each root's snapshot is a fully independent real provider round trip
    (definition/symbology resolution + an OHLCV fetch); a naive serial loop
    over 5 roots pays Databento's own per-request latency 5 times over. When
    `render_home_terminal` has already fetched the selected root (via
    `get_snapshot_cached`) before calling this, that one root is already
    cache-fresh here and only the remaining four are fetched -- the actual
    mechanism behind "selected market first" (mission section 15).

    Streamlit's `session_state` is a main-thread concept and is touched ONLY
    on the main thread here -- worker threads call the plain,
    Streamlit-free `databento_context.market_snapshot` directly and return
    a plain value; this function merges results back into the session cache
    itself, after the pool has already joined. Callers MUST have already
    called `get_health_cached()` (or anything else that triggers a
    provider call) before this, once, on the main thread -- the provider's
    SDK client is constructed lazily on first use and that construction is
    not itself thread-safe; every caller of this function already renders
    the freshness header (which calls `get_health_cached()`) first."""
    store: dict[str, tuple[float, datetime, Any]] = st.session_state.setdefault(_CACHE_KEY, {})
    now_mono = time.monotonic()
    interval = _refresh_interval_seconds()

    out: dict[str, MarketSnapshot | None] = {}
    stale: list[str] = []
    for root in universe:
        hit = store.get(f"snapshot:{root}")
        if hit is not None and (now_mono - hit[0]) <= interval:
            out[root] = hit[2]
        else:
            stale.append(root)

    if stale:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(stale)) as pool:
            results = list(pool.map(databento_context.market_snapshot, stale))
        fetch_mono = time.monotonic()
        wall_clock = datetime.now(UTC)
        for root, snap in zip(stale, results, strict=True):
            store[f"snapshot:{root}"] = (fetch_mono, wall_clock, snap)
            out[root] = snap

    return out


def get_ohlcv_cached(root: str, *, timeframe: str, lookback_bars: int) -> tuple[RecentOhlcvResult | None, datetime]:
    return _cached(
        f"ohlcv:{root}:{timeframe}:{lookback_bars}",
        lambda: databento_context.recent_ohlcv(root, timeframe=timeframe, lookback_bars=lookback_bars),
    )


def peek_ohlcv_cached(root: str, *, timeframe: str, lookback_bars: int) -> tuple[RecentOhlcvResult | None, datetime] | None:
    """What `get_ohlcv_cached` would return from this session's cache, or
    ``None`` when nothing is cached -- NEVER fetches. Lets a page show
    already-loaded prices on first paint and keep any network call behind an
    explicit action (a Research Thread's Market step)."""
    hit = st.session_state.get(_CACHE_KEY, {}).get(f"ohlcv:{root}:{timeframe}:{lookback_bars}")
    return None if hit is None else (hit[2], hit[1])


def window_change_pct(bars: list[dict[str, Any]]) -> float | None:
    """First-to-last close change (%) over an already-fetched real bar
    window -- descriptive only, never a forecast or a signal."""
    if len(bars) < 2 or not bars[0]["close"]:
        return None
    return (bars[-1]["close"] / bars[0]["close"] - 1.0) * 100.0


def get_ohlcv_for_universe(
    universe: tuple[str, ...], *, timeframe: str, lookback_bars: int,
) -> dict[str, RecentOhlcvResult | None]:
    """Batch counterpart to `get_ohlcv_cached` -- SAME cache-first-then-
    concurrent-fetch mechanism as `get_snapshots_for_universe`, for the
    Market Scanner's own multi-root OHLCV read (Section 7 performance
    acceptance: a serial per-root loop here previously turned an N-root
    eager-fetch set into N sequential real Databento round trips instead of
    one bounded concurrent batch -- the exact class of bug `771ed6b` already
    fixed once for the Relative tab's peer-group fetch). Worker threads call
    the plain, Streamlit-free `databento_context.recent_ohlcv` directly;
    `st.session_state` is touched only on the main thread, before and after
    the pool joins."""
    store: dict[str, tuple[float, datetime, Any]] = st.session_state.setdefault(_CACHE_KEY, {})
    now_mono = time.monotonic()
    interval = _refresh_interval_seconds()

    out: dict[str, RecentOhlcvResult | None] = {}
    stale: list[str] = []
    for root in universe:
        cache_key = f"ohlcv:{root}:{timeframe}:{lookback_bars}"
        hit = store.get(cache_key)
        if hit is not None and (now_mono - hit[0]) <= interval:
            out[root] = hit[2]
        else:
            stale.append(root)

    if stale:
        from concurrent.futures import ThreadPoolExecutor

        def _fetch(root: str) -> RecentOhlcvResult | None:
            return databento_context.recent_ohlcv(root, timeframe=timeframe, lookback_bars=lookback_bars)

        with ThreadPoolExecutor(max_workers=len(stale)) as pool:
            results = list(pool.map(_fetch, stale))
        fetch_mono = time.monotonic()
        wall_clock = datetime.now(UTC)
        for root, result in zip(stale, results, strict=True):
            store[f"ohlcv:{root}:{timeframe}:{lookback_bars}"] = (fetch_mono, wall_clock, result)
            out[root] = result

    return out


def get_contract_cached(root: str):
    return _cached(f"contract:{root}", lambda: databento_context.resolve_display_contract(root))


def get_contract_metadata_cached(root: str):
    return _cached(f"contract-econ:{root}", lambda: databento_context.contract_metadata(root))


#: Checkpoint C, Section 34: contract ladder/term-structure definitions
#: change far more slowly than a price -- a much longer UI-session cache
#: window than the default `ui_refresh_interval_seconds` (90s), independent
#: of the provider's own long-TTL definitions cache underneath.
CONTRACT_LADDER_UI_TTL_SECONDS = 900.0


def get_contract_ladder_cached(root: str, *, max_contracts: int = 12):
    return _cached(
        f"ladder:{root}:{max_contracts}",
        lambda: databento_context.contract_ladder(root, max_contracts=max_contracts),
        ttl_seconds=CONTRACT_LADDER_UI_TTL_SECONDS,
    )


def get_term_structure_cached(root: str, *, max_contracts: int = 12):
    return _cached(
        f"term-structure:{root}:{max_contracts}",
        lambda: databento_context.term_structure(root, max_contracts=max_contracts),
        ttl_seconds=CONTRACT_LADDER_UI_TTL_SECONDS,
    )


# ---------------------------------------------------------------------------
# pure descriptive statistics (no Streamlit) -- MARKET CONTEXT, never a signal
# ---------------------------------------------------------------------------


def age_label(freshness_seconds: float | None) -> str:
    if freshness_seconds is None:
        return "N/A"
    if freshness_seconds < 0:
        freshness_seconds = 0.0
    hours = freshness_seconds / 3600.0
    if hours < 1:
        return f"{freshness_seconds / 60.0:.0f} min"
    if hours < 48:
        return f"{hours:.1f} hours"
    return f"{hours / 24.0:.1f} days"


def trend_state(bars: list[dict[str, Any]]) -> str:
    """Descriptive only -- never SIGNAL/BUY/SELL."""
    if len(bars) < 2:
        return "Insufficient data"
    change = bars[-1]["close"] - bars[0]["close"]
    pct = change / bars[0]["close"] if bars[0]["close"] else 0.0
    if pct > 0.01:
        return "Up"
    if pct < -0.01:
        return "Down"
    return "Sideways"


def _log_returns(bars: list[dict[str, Any]]) -> list[float]:
    closes = [b["close"] for b in bars]
    return [(closes[i] / closes[i - 1] - 1.0) for i in range(1, len(closes)) if closes[i - 1]]


def volatility_state(bars: list[dict[str, Any]]) -> str:
    rets = _log_returns(bars)
    if len(rets) < 2:
        return "Insufficient data"
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    stdev = var ** 0.5
    if stdev > 0.01:
        return "High"
    if stdev > 0.003:
        return "Moderate"
    return "Low"


def realized_volatility_pct(bars: list[dict[str, Any]], *, bars_per_year: float = 24 * 252) -> float | None:
    """Annualized realized volatility (%) from the stdev of per-bar returns
    over an already-fetched real bar window -- a DESCRIPTIVE statistic, never
    a forecast or a trading signal. `bars_per_year` defaults to an approximate
    24 (roughly round-the-clock futures session) x 252 trading days; the
    caller states the true bar cadence via this parameter when it differs.
    `None` when there are too few bars to compute a meaningful stdev."""
    rets = _log_returns(bars)
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    stdev = var ** 0.5
    return stdev * (bars_per_year ** 0.5) * 100.0


def session_position_state(bars: list[dict[str, Any]]) -> str:
    """Where the last close sits within the shown window's own high/low --
    5-way, symmetric around the middle. Purely positional, never a
    directional judgment (being "Near High" is not labeled as good)."""
    if not bars:
        return "Insufficient data"
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    high, low = max(highs), min(lows)
    last = bars[-1]["close"]
    if high == low:
        return "Middle"
    pct = (last - low) / (high - low)
    if pct >= 0.8:
        return "Near High"
    if pct >= 0.6:
        return "Upper Half"
    if pct > 0.4:
        return "Middle"
    if pct > 0.2:
        return "Lower Half"
    return "Near Low"


def volume_context_state(bars: list[dict[str, Any]]) -> str:
    """Latest bar's volume vs. the window's own average -- descriptive only,
    and honestly `Unavailable` when the schema/provider carried no volume
    (never zero-filled -- mirrors `charts_market.market_volume_chart`)."""
    vols = [b.get("volume") for b in bars]
    if not vols or any(v is None for v in vols) or len(vols) < 2:
        return "Unavailable"
    prior = vols[:-1]
    avg = sum(prior) / len(prior) if prior else 0.0
    if avg <= 0:
        return "Unavailable"
    ratio = vols[-1] / avg
    if ratio >= 1.5:
        return "Elevated"
    if ratio <= 0.5:
        return "Quiet"
    return "Normal"


def session_range_pct(session_high: float | None, session_low: float | None) -> float | None:
    """(High - Low) / Low, as a percent -- how wide the session has been."""
    if session_high is None or session_low is None or session_low <= 0:
        return None
    return (session_high - session_low) / session_low * 100.0


def distance_from_high_pct(last: float | None, session_high: float | None) -> float | None:
    """(Last - High) / High, as a percent -- normally <= 0."""
    if last is None or session_high is None or session_high <= 0:
        return None
    return (last - session_high) / session_high * 100.0


def distance_from_low_pct(last: float | None, session_low: float | None) -> float | None:
    """(Last - Low) / Low, as a percent -- normally >= 0."""
    if last is None or session_low is None or session_low <= 0:
        return None
    return (last - session_low) / session_low * 100.0


# ---------------------------------------------------------------------------
# formatting + descriptive-context color helpers
# ---------------------------------------------------------------------------


def _fmt(value: float | None, *, digits: int = 2) -> str:
    return f"{value:,.{digits}f}" if value is not None else "N/A"


def _fmt_signed(value: float | None, *, digits: int = 2) -> str:
    return f"{value:+,.{digits}f}" if value is not None else "N/A"


def _fmt_pct(value: float | None) -> str:
    return f"{value:+.2f}%" if value is not None else "N/A"


def _change_color(change_pct: float | None) -> str | None:
    if change_pct is None:
        return None
    return palette.GREEN if change_pct >= 0 else palette.RED


#: Descriptive-context labels -> a tone color, purely for fast visual
#: scanning -- NEVER a BUY/SELL/PASS/REJECT vocabulary (kept deliberately
#: separate from `palette.STATE_COLOR`, which IS that vocabulary). Session
#: Position is intentionally neutral (blue/grey only) -- "near high" is a
#: location, not a judgment.
_CONTEXT_TONE: dict[str, str] = {
    "Up": palette.GREEN, "Down": palette.RED, "Sideways": palette.GREY,
    "Low": palette.GREEN, "Moderate": palette.AMBER, "High": palette.RED,
    "Near High": palette.BLUE, "Upper Half": palette.BLUE, "Middle": palette.GREY,
    "Lower Half": palette.BLUE, "Near Low": palette.BLUE,
    "Elevated": palette.AMBER, "Normal": palette.GREY, "Quiet": palette.GREY,
    "Insufficient data": palette.GREY, "Unavailable": palette.GREY,
}


def _context_pill(text: str) -> str:
    color = _CONTEXT_TONE.get(text, palette.GREY)
    return f'<span class="aa-badge" style="color:{color};border-color:{color}66;background:{color}1f">{text}</span>'


# ---------------------------------------------------------------------------
# disconnected state -- ONE concise card, never a table of N/A
# ---------------------------------------------------------------------------


def render_unavailable_state(health: DatabentoHealth, *, key_prefix: str = "home") -> None:
    with components.card(f"{key_prefix}-unavailable"):
        st.markdown(
            '<div class="aa-empty">'
            '<div class="aa-empty-title">MARKET DATA UNAVAILABLE</div>'
            f'<div class="aa-empty-badge">{components.badge("FAIL", label="DATABENTO · " + health.capability.value)}</div>'
            '<div class="aa-empty-reason">Databento is not currently connected -- no real futures data can be '
            'shown right now. Nothing below is fabricated.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption(health.detail)
        if st.button("Retry", key=f"{key_prefix}-retry", type="primary"):
            force_refresh()
            st.rerun()


# ---------------------------------------------------------------------------
# ticker strip (compact clickable cards, each with a real sparkline)
# ---------------------------------------------------------------------------


def _sparkline_closes(root: str) -> list[float]:
    """The last `_SPARKLINE_BARS` real closes for `root`, reusing the SAME
    72-bar 1h window every other panel on this page asks for -- this is a
    cache hit against the provider's own request-level TTL cache in the
    common case (the snapshot fetch already populated it), never a new
    request issued solely for a decorative sparkline."""
    result, _ = get_ohlcv_cached(root, timeframe="1h", lookback_bars=SNAPSHOT_LOOKBACK_BARS)
    if not result or not result.fetched or not result.bars:
        return []
    return [b.close for b in result.bars[-_SPARKLINE_BARS:]]


def _render_ticker_card(root: str, snap: MarketSnapshot | None, *, is_selected: bool, key_prefix: str) -> None:
    with st.container(border=True):
        color = _change_color(snap.change_pct if snap else None) or palette.TEXT_MUTED
        st.markdown(
            f'<div class="aa-ticker-card-root">{root}</div>'
            f'<div class="aa-ticker-card-last">{_fmt(snap.last if snap else None)}</div>'
            f'<div class="aa-ticker-card-change" style="color:{color}">'
            f'{_fmt_pct(snap.change_pct if snap else None)}</div>',
            unsafe_allow_html=True,
        )
        closes = _sparkline_closes(root) if snap else []
        if len(closes) >= 2:
            spark_color = palette.GREEN if closes[-1] >= closes[0] else palette.RED
            components.plotly_chart(
                charts.sparkline(closes, color=spark_color, height=32), key=f"{key_prefix}-spark-{root}",
            )
        label = "Selected" if is_selected else "Select"
        if st.button(label, key=f"{key_prefix}-select-{root}", type="primary" if is_selected else "secondary",
                     width="stretch"):
            st.session_state["market_selected_root"] = root
            st.rerun()


def render_ticker_strip(*, key_prefix: str = "home") -> None:
    """The compact, five-card "REAL FUTURES MARKET" strip -- last price,
    change %, and a real sparkline per root, never a large primary table.
    Clicking a card's Select button updates the SAME `market_selected_root`
    session key every other Market-local selector on this page drives --
    never a second, competing selection (Sidebar IA pass, task spec section
    1D: this state is Market-page-local, never a hidden global selector)."""
    universe = services.approved_universe()
    health, health_refreshed = get_health_cached()

    c_status, c_refresh = st.columns([6, 1])
    with c_status:
        st.caption(
            f"Market data from Databento (GLBX.MDP3) · {health.capability.value} · "
            f"updated {health_refreshed.strftime('%H:%M:%S UTC')}"
        )
    with c_refresh:
        if st.button("↻ Refresh", key=f"{key_prefix}-refresh", width="stretch"):
            force_refresh()
            st.rerun()

    snapshots = get_snapshots_for_universe(tuple(universe))
    selected_root = st.session_state.get("market_selected_root")
    cols = st.columns(len(universe))
    for col, root in zip(cols, universe, strict=True):
        with col:
            _render_ticker_card(root, snapshots.get(root), is_selected=(selected_root == root), key_prefix=key_prefix)


# ---------------------------------------------------------------------------
# Selected Market hero -- investor-readable primary display
# ---------------------------------------------------------------------------


def render_selected_market_hero(root: str, *, key_prefix: str = "home") -> None:
    contract, _ = get_contract_cached(root)
    snap, snap_refreshed = get_snapshot_cached(root)
    color = _change_color(snap.change_pct if snap else None) or palette.TEXT_MUTED

    with components.card(f"{key_prefix}-hero"):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(
                f'<div class="aa-hero-market">{root} · {services.market_name(root)} Futures</div>'
                f'<div class="aa-hero-price">{_fmt(snap.last if snap else None)}</div>'
                f'<div class="aa-hero-change" style="color:{color}">'
                f'{_fmt_signed(snap.change if snap else None)}&nbsp;&nbsp;{_fmt_pct(snap.change_pct if snap else None)}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div style="text-align:right">'
                '<div class="aa-hero-contract-label">DISPLAY CONTRACT</div>'
                f'<div class="aa-hero-contract-value">'
                f'{contract.resolved_raw_symbol if (contract and contract.resolved_raw_symbol) else "N/A"}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        obs = snap.as_of.strftime("%Y-%m-%d %H:%M UTC") if snap and snap.last is not None else "N/A"
        age = age_label(snap.freshness_seconds if snap else None)
        st.caption(f"Latest available data · observed {obs} ({age} old) · refreshed {snap_refreshed.strftime('%H:%M:%S UTC')}")

        with st.expander("Data Details", expanded=False):
            st.caption(
                "Research Root is this platform's canonical symbol; Display Contract is the specific "
                "expiring future currently resolved from Databento's continuous front-month proxy for "
                "display only -- not a statement about which raw contract any historical backtest "
                "actually traded on a given date."
            )
            components.provenance_row("Research Root", root)
            components.provenance_row(
                "Display Contract",
                contract.resolved_raw_symbol if (contract and contract.resolved_raw_symbol) else "N/A",
            )
            components.provenance_row(
                "Expiry", contract.expiry.strftime("%Y-%m-%d") if (contract and contract.expiry) else "N/A",
            )
            components.provenance_row("Latest observation", obs)
            components.provenance_row("Observation age", age)
            components.provenance_row("Last refreshed (this session)", snap_refreshed.strftime("%Y-%m-%d %H:%M:%S UTC"))
            econ, _ = get_contract_metadata_cached(root)
            if econ:
                components.provenance_row(
                    "Contract economics",
                    f"tick size {econ.quote_tick_size:g} · point value ${econ.point_value_usd:,.2f} · "
                    f"tick value ${econ.tick_value_usd:,.2f} ({econ.quote_convention})",
                )


# ---------------------------------------------------------------------------
# Core Metrics panel (compact, label:value rows)
# ---------------------------------------------------------------------------


def render_core_metrics_panel(root: str, *, key_prefix: str = "home") -> None:
    snap, _ = get_snapshot_cached(root)
    ohlcv, _ = get_ohlcv_cached(root, timeframe="1h", lookback_bars=SNAPSHOT_LOOKBACK_BARS)
    bars = [b.model_dump(mode="json") for b in ohlcv.bars] if (ohlcv and ohlcv.fetched) else []
    rv = realized_volatility_pct(bars) if bars else None
    high = snap.session.session_high if (snap and snap.session) else None
    low = snap.session.session_low if (snap and snap.session) else None
    last = snap.last if snap else None

    with components.card(f"{key_prefix}-metrics"):
        st.markdown('<div class="aa-gate-title">CORE METRICS</div>', unsafe_allow_html=True)
        components.provenance_row("Last", _fmt(last))
        components.provenance_row(
            "Change", f"{_fmt_signed(snap.change if snap else None)}  ({_fmt_pct(snap.change_pct if snap else None)})",
        )
        components.provenance_row("Session High", _fmt(high))
        components.provenance_row("Session Low", _fmt(low))
        components.provenance_row("Volume", _fmt(snap.session.volume if (snap and snap.session) else None, digits=0))
        components.provenance_row("Realized Volatility", f"{rv:.1f}%" if rv is not None else "N/A")
        rng = session_range_pct(high, low)
        components.provenance_row("Session Range", f"{rng:.2f}%" if rng is not None else "N/A")
        dfh = distance_from_high_pct(last, high)
        components.provenance_row("From Session High", f"{dfh:+.2f}%" if dfh is not None else "N/A")
        dfl = distance_from_low_pct(last, low)
        components.provenance_row("From Session Low", f"{dfl:+.2f}%" if dfl is not None else "N/A")


# ---------------------------------------------------------------------------
# Price + Volume chart, paired with Core Metrics (2-column, above the fold)
# ---------------------------------------------------------------------------


def render_chart_and_metrics(root: str, *, key_prefix: str = "home", default_timeframe_index: int = 3) -> None:
    col_chart, col_metrics = st.columns([3, 1])
    with col_chart, components.card(f"{key_prefix}-chart"):
        timeframe = st.radio(
            "Timeframe", TIMEFRAMES, index=default_timeframe_index, horizontal=True, key=f"{key_prefix}-timeframe",
        )
        result, refreshed = get_ohlcv_cached(root, timeframe=timeframe, lookback_bars=96)
        if result is None or not result.fetched or not result.bars:
            reason = result.detail if result else "provider unavailable"
            components.empty_state("Price", f"No real bars available: {reason}", key=f"{key_prefix}-price-na")
        else:
            if timeframe in _RESAMPLED_TIMEFRAMES:
                st.caption(f"{timeframe} bars are a resample of real 1-minute Databento bars -- not a separate vendor grain.")
            bars = [b.model_dump(mode="json") for b in result.bars]
            components.plotly_chart(charts_market.market_price_chart(bars, height=360), key=f"{key_prefix}-price-fig")
            components.plotly_chart(charts_market.market_volume_chart(bars, height=110), key=f"{key_prefix}-vol-fig")
            st.caption(f"{result.detail} · Last refreshed: {refreshed.strftime('%H:%M:%S UTC')}")
    with col_metrics:
        render_core_metrics_panel(root, key_prefix=key_prefix)


# ---------------------------------------------------------------------------
# MARKET CONTEXT -- short, visual, descriptive only, never a signal
# ---------------------------------------------------------------------------


def render_market_context(root: str, *, key_prefix: str = "home") -> None:
    result, _ = get_ohlcv_cached(root, timeframe="1h", lookback_bars=SNAPSHOT_LOOKBACK_BARS)
    with components.card(f"{key_prefix}-context"):
        st.markdown('<div class="aa-gate-title">MARKET CONTEXT</div>', unsafe_allow_html=True)
        if not result or not result.fetched or not result.bars:
            components.empty_state("Market Context", "No real bars available.", key=f"{key_prefix}-context-na")
            return
        bars = [b.model_dump(mode="json") for b in result.bars]
        trend, vol, session, volume = (
            trend_state(bars), volatility_state(bars), session_position_state(bars), volume_context_state(bars),
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="aa-metric-label">TREND</div>{_context_pill(trend)}', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="aa-metric-label">VOLATILITY</div>{_context_pill(vol)}', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="aa-metric-label">SESSION POSITION</div>{_context_pill(session)}', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="aa-metric-label">VOLUME</div>{_context_pill(volume)}', unsafe_allow_html=True)
        st.caption("Descriptive statistics only -- never a trading signal.")
        with st.expander("Calculation notes", expanded=False):
            st.caption(
                "Trend: net close-to-close change over the shown window, thresholded at ±1%. "
                "Volatility: stdev of per-bar returns, thresholded at 0.3%/1.0%. "
                "Session Position: where the last close sits within the window's own high-low range. "
                "Volume: latest bar's volume vs. the window's own average, thresholded at 0.5x/1.5x."
            )


# ---------------------------------------------------------------------------
# HOME TERMINAL -- the Agent/Home page's whole market surface, one call
# ---------------------------------------------------------------------------


def render_home_terminal(root: str, *, key_prefix: str = "home") -> None:
    """Renders, in this exact visual order: ticker strip -> Selected Market
    hero -> chart + Core Metrics -> Market Context. See the module docstring
    ("PRIORITIZED FIRST PAINT") for why the strip's `st.container()` is
    created first but filled last. A disconnected/unavailable Databento
    collapses to ONE concise state card (`render_unavailable_state`), never
    a table of N/A."""
    health, _ = get_health_cached()
    strip_slot = st.container()

    if health.capability in UNAVAILABLE_CAPABILITIES:
        with strip_slot:
            render_unavailable_state(health, key_prefix=key_prefix)
        return

    # Priority fetch: the selected root's own snapshot, resolved and cached
    # BEFORE the ticker strip (below) fetches the other four.
    get_snapshot_cached(root)

    render_selected_market_hero(root, key_prefix=key_prefix)
    render_chart_and_metrics(root, key_prefix=key_prefix)
    render_market_context(root, key_prefix=key_prefix)

    with strip_slot:
        render_ticker_strip(key_prefix=key_prefix)


__all__ = [
    "CONTRACT_LADDER_UI_TTL_SECONDS",
    "SNAPSHOT_LOOKBACK_BARS",
    "TIMEFRAMES",
    "age_label",
    "distance_from_high_pct",
    "distance_from_low_pct",
    "force_refresh",
    "get_contract_cached",
    "get_contract_ladder_cached",
    "get_contract_metadata_cached",
    "get_health_cached",
    "get_ohlcv_cached",
    "get_snapshot_cached",
    "get_snapshots_for_universe",
    "get_term_structure_cached",
    "peek_ohlcv_cached",
    "realized_volatility_pct",
    "render_chart_and_metrics",
    "render_core_metrics_panel",
    "render_home_terminal",
    "render_market_context",
    "render_selected_market_hero",
    "render_ticker_strip",
    "render_unavailable_state",
    "session_position_state",
    "session_range_pct",
    "trend_state",
    "volatility_state",
    "volume_context_state",
    "window_change_pct",
]
