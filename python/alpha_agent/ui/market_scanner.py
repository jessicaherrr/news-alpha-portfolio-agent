"""MARKET SCANNER -- Market Intelligence + Futures Universe campaign, Parts
5-6. The "ALL FUTURES" landing view for the Market page: every catalogued
product, with deterministic typed fields (mission Part 6) that will later be
handed to an Agent/Claude layer as-is -- "Do NOT have Claude generate these
values."

Row construction (`build_scanner_rows`) and sorting/filtering
(`sort_rows`/`filter_rows`) are pure functions with no Streamlit and no
network dependency -- fully unit-testable with injected snapshots/bars/
capabilities (mission Part 40: "uses real/fake injected provider values").
`render_scanner` is the only Streamlit entry point and is a thin, bounded
wiring layer over `alpha_agent.ui.market_home` (real per-root fetches,
already TTL-cached and concurrency-bounded) and
`alpha_agent.ui.market_universe` (typed capability derivation) -- it never
issues a live request for a product that is not in the wired research
universe (mission Part 44: never load 50 products serially before first
paint).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

import streamlit as st
from pydantic import BaseModel

from alpha_agent.market_intel.mapping import CATEGORY_PRODUCTS
from alpha_agent.marketdata.capability import CapabilityState, FuturesProductCapability
from alpha_agent.marketdata.databento_schemas import MarketSnapshot
from alpha_agent.marketdata.product_catalog import (
    ASSET_CLASS_LABELS,
    AssetClass,
    ProductCatalogEntry,
)
from alpha_agent.ui import components, market_home, market_intel_context, market_universe, services

#: Section 32/33: every root Section 21's mapping table can ever reach --
#: a root outside this set has no deterministic news-mapping rule at all
#: (e.g. SI/HG/crypto today), so it stays honestly NOT_TESTED rather than
#: a fabricated AVAILABLE-with-zero or a misleading NOT_CONNECTED (which
#: would imply a real connector attempt was made and failed).
_NEWS_MAPPED_PRODUCTS: frozenset[str] = frozenset(p for products in CATEGORY_PRODUCTS.values() for p in products)

__all__ = [
    "FEATURED_COVERAGE",
    "SORT_OPTIONS",
    "MarketScannerRow",
    "ResearchStatus",
    "TradingStatus",
    "build_scanner_rows",
    "filter_rows",
    "render_scanner",
    "root_for_selection",
    "sort_rows",
    "split_primary_secondary",
]


class ResearchStatus(str, Enum):
    RESEARCH_ENABLED = "RESEARCH_ENABLED"
    NOT_CERTIFIED = "NOT_CERTIFIED"


class TradingStatus(str, Enum):
    PAPER_ELIGIBLE = "PAPER_ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"


class MarketScannerRow(BaseModel):
    """One scanner row -- mission Part 5's column list plus Part 6's
    deterministic descriptive fields.

    Market Intelligence Data Completion Pass, Section 2: ``news_count_recent``
    is `None` unless ``news_state`` is actually `AVAILABLE` -- a numeric ``0``
    must never be indistinguishable from "the news system was never queried".
    Same idea for ``event_state``/``next_high_impact_event`` (Section 32).
    Both default to `NOT_CONNECTED` until Checkpoints E/F wire a real
    NewsStore/event calendar; nothing here is a fabricated placeholder."""

    model_config = {"frozen": True, "extra": "forbid"}
    schema_version: str = "market-scanner-row/2"

    root_symbol: str
    display_name: str
    asset_class: AssetClass
    market_data_state: CapabilityState

    last: float | None = None
    change_pct: float | None = None
    volume: float | None = None
    realized_volatility_pct: float | None = None
    trend_state: str | None = None
    session_position_state: str | None = None
    volume_state: str | None = None

    news_state: CapabilityState = CapabilityState.NOT_CONNECTED
    news_count_recent: int | None = None
    news_window_label: str = "24h"

    event_state: CapabilityState = CapabilityState.NOT_CONNECTED
    next_high_impact_event: str | None = None
    high_impact_event_soon: bool = False

    research_status: ResearchStatus
    trading_status: TradingStatus
    best_research_promise_label: str | None = None
    scientific_verdict: str | None = None

    as_of: datetime | None = None
    detail: str = ""


# ---------------------------------------------------------------------------
# pure row construction
# ---------------------------------------------------------------------------


def build_scanner_rows(
    catalog: tuple[ProductCatalogEntry, ...],
    *,
    capabilities: dict[str, FuturesProductCapability],
    snapshots: dict[str, MarketSnapshot | None] | None = None,
    bars_by_root: dict[str, list[dict[str, Any]]] | None = None,
    promise_by_root: dict[str, str | None] | None = None,
    verdict_by_root: dict[str, str | None] | None = None,
    news_by_root: dict[str, tuple[CapabilityState, int | None]] | None = None,
    event_by_root: dict[str, tuple[CapabilityState, str | None]] | None = None,
) -> tuple[MarketScannerRow, ...]:
    """``news_by_root``/``event_by_root`` map root -> (state, value); a root
    absent from either dict stays honestly `NOT_CONNECTED` (Section 2/32) --
    never a fabricated ``0`` or ``False``."""
    snapshots = snapshots or {}
    bars_by_root = bars_by_root or {}
    promise_by_root = promise_by_root or {}
    verdict_by_root = verdict_by_root or {}
    news_by_root = news_by_root or {}
    event_by_root = event_by_root or {}

    rows: list[MarketScannerRow] = []
    for entry in catalog:
        root = entry.root_symbol
        cap = capabilities[root]
        snap = snapshots.get(root)
        bars = bars_by_root.get(root) or []
        news_state, news_count = news_by_root.get(root, (CapabilityState.NOT_CONNECTED, None))
        event_state, next_event = event_by_root.get(root, (CapabilityState.NOT_CONNECTED, None))
        rows.append(
            MarketScannerRow(
                root_symbol=root,
                display_name=entry.display_name,
                asset_class=entry.asset_class,
                market_data_state=cap.market_data_available,
                last=snap.last if snap else None,
                change_pct=snap.change_pct if snap else None,
                volume=(snap.session.volume if (snap and snap.session) else None),
                realized_volatility_pct=market_home.realized_volatility_pct(bars) if bars else None,
                trend_state=market_home.trend_state(bars) if bars else None,
                session_position_state=market_home.session_position_state(bars) if bars else None,
                volume_state=market_home.volume_context_state(bars) if bars else None,
                news_state=news_state,
                news_count_recent=news_count if news_state is CapabilityState.AVAILABLE else None,
                event_state=event_state,
                next_high_impact_event=next_event if event_state is CapabilityState.AVAILABLE else None,
                high_impact_event_soon=bool(next_event) if event_state is CapabilityState.AVAILABLE else False,
                research_status=ResearchStatus.RESEARCH_ENABLED if cap.research_enabled else ResearchStatus.NOT_CERTIFIED,
                trading_status=TradingStatus.PAPER_ELIGIBLE if cap.paper_trading_enabled else TradingStatus.NOT_ELIGIBLE,
                best_research_promise_label=promise_by_root.get(root),
                scientific_verdict=verdict_by_root.get(root),
                as_of=snap.as_of if snap else None,
                detail=cap.capability_reason,
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# pure sort / filter
# ---------------------------------------------------------------------------


def _desc(value: float | None) -> tuple[int, float]:
    """Sort key for "highest first, missing values last" -- independent of
    Python's default None-comparison behavior (which raises)."""
    return (1, 0.0) if value is None else (0, -value)


def _trend_rank(state: str | None) -> int:
    if state in ("Up", "Down"):
        return 0
    if state == "Sideways":
        return 1
    return 2  # "Insufficient data" or None


SORT_OPTIONS: tuple[str, ...] = (
    "Symbol",
    "Top Movers",
    "Highest Volume",
    "Highest Volatility",
    "Strongest Trend",
    "Most News Activity",
    "Research Enabled",
    "Paper-Trading Eligible",
)

_SORT_KEYS: dict[str, Any] = {
    "Symbol": lambda r: r.root_symbol,
    "Top Movers": lambda r: _desc(abs(r.change_pct) if r.change_pct is not None else None),
    "Highest Volume": lambda r: _desc(r.volume),
    "Highest Volatility": lambda r: _desc(r.realized_volatility_pct),
    "Strongest Trend": lambda r: (_trend_rank(r.trend_state), _desc(abs(r.change_pct) if r.change_pct is not None else None)),
    "Most News Activity": lambda r: _desc(float(r.news_count_recent) if r.news_count_recent is not None else None),
    "Research Enabled": lambda r: r.research_status is not ResearchStatus.RESEARCH_ENABLED,
    "Paper-Trading Eligible": lambda r: r.trading_status is not TradingStatus.PAPER_ELIGIBLE,
}


def sort_rows(rows: tuple[MarketScannerRow, ...], sort_by: str) -> tuple[MarketScannerRow, ...]:
    """Deterministic, stable sort -- rows that tie on the chosen key keep
    their catalog order (mission Part 40: "same input => same output")."""
    key_fn = _SORT_KEYS.get(sort_by, _SORT_KEYS["Symbol"])
    return tuple(sorted(rows, key=key_fn))


def root_for_selection(visible_rows: tuple[MarketScannerRow, ...], selected_row_indices: list[int]) -> str | None:
    """Pure translation from a `st.dataframe(on_select=...)` event's
    `selection.rows` (integer positions into the CURRENTLY VISIBLE,
    already-sorted-and-filtered table) to the root symbol that row
    represents -- `None` when nothing is selected. Kept separate from
    `render_scanner`'s Streamlit call so the click-to-select behavior is
    unit-testable without simulating a real frontend dataframe interaction
    (`streamlit.testing.v1.AppTest` has no such simulation)."""
    if not selected_row_indices:
        return None
    index = selected_row_indices[0]
    if index < 0 or index >= len(visible_rows):
        return None
    return visible_rows[index].root_symbol


#: Product UI Polish pass, section 6 -- a SMALL, curated "featured" basket per
#: asset class (~3-4 products), so the primary scanner never leaves an entire
#: asset class looking empty while another is crowded with real rows. Every
#: root below was verified live against the REAL configured Databento
#: entitlement (`alpha_agent.ui.databento_context.market_snapshot`, the exact
#: same bounded-concurrency mechanism this module's own eager-fetch path
#: already uses -- verified 2026-09-18): each one returned a real last price
#: and a resolved raw contract symbol. This supersedes the narrower Checkpoint
#: B extension (`MNQ, NG, SI, HG, ZB, 6E`, mission Part 41), which covered
#: only four of the seven asset classes. Featured membership is a
#: PRESENTATION curation only -- it implies nothing about the RESEARCH or
#: TRADING universes (mission Part 2) and is never used to decide research or
#: paper eligibility.
FEATURED_COVERAGE: dict[AssetClass, tuple[str, ...]] = {
    AssetClass.EQUITY_INDEX: ("ES", "NQ", "RTY", "YM"),
    AssetClass.ENERGY: ("CL", "NG", "RB", "HO"),
    AssetClass.METALS: ("GC", "SI", "HG"),
    AssetClass.RATES: ("ZN", "ZB", "ZF", "ZT"),
    AssetClass.FX: ("6E", "6J", "6B", "6A"),
    AssetClass.AGRICULTURE: ("ZC", "ZS", "ZW", "ZM"),
    AssetClass.CRYPTO: ("BTC", "ETH", "MBT"),
}


def filter_rows(
    rows: tuple[MarketScannerRow, ...], *, asset_class: AssetClass | None = None, search: str | None = None,
) -> tuple[MarketScannerRow, ...]:
    out = rows
    if asset_class is not None:
        out = tuple(r for r in out if r.asset_class is asset_class)
    query = (search or "").strip().upper()
    if query:
        out = tuple(r for r in out if query in r.root_symbol.upper() or query in r.display_name.upper())
    return out


# ---------------------------------------------------------------------------
# supporting reads (bounded, cached where the underlying provider already is)
# ---------------------------------------------------------------------------


def _best_candidate_per_root() -> dict[str, tuple[str | None, str | None]]:
    """root -> (best_research_promise_label, scientific_verdict) from the
    real registry, PASS ranked above any promise score, else highest
    `research_promise_score`. Empty/absent roots are simply not keyed --
    callers treat a missing root as "no candidate research yet", never a
    fabricated placeholder."""
    best: dict[str, Any] = {}
    for summary in services.research_candidate_summaries():
        root = summary.root_symbol
        current = best.get(root)
        is_pass = summary.scientific_verdict == "PASS"
        rank = (1 if is_pass else 0, summary.research_promise_score)
        if current is None or rank > current[0]:
            best[root] = (rank, summary.research_promise_label, summary.scientific_verdict)
    return {root: (v[1], v[2]) for root, v in best.items()}


def _fetch_wired_market_state(
    research_roots: tuple[str, ...],
) -> tuple[dict[str, MarketSnapshot | None], dict[str, list[dict[str, Any]]]]:
    """Section 7 (performance acceptance): both the snapshot batch and the
    OHLCV batch below are fetched CONCURRENTLY across `research_roots` --
    never a serial per-root loop, which would turn an N-root eager-fetch set
    into N sequential real Databento round trips (each independently
    measured at several seconds to under a minute in this environment)."""
    snapshots = market_home.get_snapshots_for_universe(research_roots)
    ohlcv_by_root = market_home.get_ohlcv_for_universe(
        research_roots, timeframe="1h", lookback_bars=market_home.SNAPSHOT_LOOKBACK_BARS,
    )
    bars_by_root: dict[str, list[dict[str, Any]]] = {
        root: ([b.model_dump(mode="json") for b in result.bars] if (result and result.fetched) else [])
        for root, result in ohlcv_by_root.items()
    }
    return snapshots, bars_by_root


def _news_state_for_universe(
    catalog: tuple[ProductCatalogEntry, ...],
) -> dict[str, tuple[CapabilityState, int | None]]:
    """Section 32/33 -- real NewsStore-backed state per catalogued root.
    NewsStore reads are local SQLite (cheap for the whole catalog, unlike a
    Databento price fetch), so this is never bounded/lazy the way
    `_fetch_wired_market_state` is."""
    health = market_intel_context.connector_health()
    any_available = any(state is CapabilityState.AVAILABLE for state in health.values())
    out: dict[str, tuple[CapabilityState, int | None]] = {}
    for entry in catalog:
        root = entry.root_symbol
        if root not in _NEWS_MAPPED_PRODUCTS:
            out[root] = (CapabilityState.NOT_TESTED, None)
        elif any_available:
            out[root] = (CapabilityState.AVAILABLE, market_intel_context.news_count_for_product(root))
        else:
            out[root] = (CapabilityState.NOT_CONNECTED, None)
    return out


def _event_state_for_universe(
    catalog: tuple[ProductCatalogEntry, ...],
) -> dict[str, tuple[CapabilityState, str | None]]:
    """Section 32/33 -- real EventStore-backed high-impact-event state per
    catalogued root, mirroring `_news_state_for_universe`. The value is a
    short human label for the row (never the bool `high_impact_event_soon`
    alone -- `build_scanner_rows` derives that from AVAILABLE + a non-None
    label)."""
    health = market_intel_context.event_connector_health()
    any_available = any(state is CapabilityState.AVAILABLE for state in health.values())
    out: dict[str, tuple[CapabilityState, str | None]] = {}
    for entry in catalog:
        root = entry.root_symbol
        if root not in _NEWS_MAPPED_PRODUCTS:
            out[root] = (CapabilityState.NOT_TESTED, None)
            continue
        if not any_available:
            out[root] = (CapabilityState.NOT_CONNECTED, None)
            continue
        event = market_intel_context.next_high_impact_event_for_product(root)
        label = f"{event.name} ({event.scheduled_at:%Y-%m-%d %H:%M} UTC)" if event else None
        out[root] = (CapabilityState.AVAILABLE, label)
    return out


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------


#: Section 1/35: a blank price must never render as a bare "—" when a more
#: informative capability state exists -- these are the states where "—" IS
#: the whole honest answer (nothing has ever been attempted, or the provider
#: itself is down); every other state still shows the state NAME alongside
#: any real partial value (Section 35F: DEGRADED still shows a real price).
_PRICE_BLANK_STATES = frozenset({CapabilityState.NOT_TESTED, CapabilityState.NOT_LOADED, CapabilityState.NOT_CONNECTED})


def _price_cell(row: MarketScannerRow) -> str:
    if row.last is not None:
        return f"{row.last:,.2f}"
    if row.market_data_state in _PRICE_BLANK_STATES:
        return f"— ({row.market_data_state.value.replace('_', ' ')})"
    return row.market_data_state.value.replace("_", " ")  # UNAVAILABLE with no price


def _news_cell(row: MarketScannerRow) -> str:
    if row.news_state is CapabilityState.AVAILABLE:
        return str(row.news_count_recent if row.news_count_recent is not None else 0)
    return row.news_state.value.replace("_", " ")


def _event_cell(row: MarketScannerRow) -> str:
    if row.event_state is CapabilityState.AVAILABLE:
        return row.next_high_impact_event or "None scheduled"
    return row.event_state.value.replace("_", " ")


def _row_to_table_dict(row: MarketScannerRow) -> dict[str, Any]:
    """The PRIMARY table's row shape (Section 3/4): real observational
    columns only. Per-row Research/Trading capability labels were removed
    here -- that explanation now renders ONCE, above the table (see
    `_render_capability_summary`), never repeated on every row."""
    return {
        "Symbol": row.root_symbol,
        "Product": row.display_name,
        "Asset Class": ASSET_CLASS_LABELS[row.asset_class],
        "Last": _price_cell(row),
        "Change %": f"{row.change_pct:+.2f}%" if row.change_pct is not None else "—",
        "Volume": f"{row.volume:,.0f}" if row.volume is not None else "—",
        "Realized Vol": f"{row.realized_volatility_pct:.1f}%" if row.realized_volatility_pct is not None else "—",
        "Trend": row.trend_state or "—",
        f"News {row.news_window_label}": _news_cell(row),
        "Next High-Impact Event": _event_cell(row),
    }


#: Section 5's own three canonical honest reasons, plus one more for a real
#: attempted-but-empty fetch -- never the long, multi-sentence
#: `MarketScannerRow.detail` capability paragraph (that stays available in
#: the row's own tooltip-free `detail` field for anyone reading the model
#: directly, just never crammed into a compact secondary-table cell).
_SHORT_REASON_BY_STATE: dict[CapabilityState, str] = {
    CapabilityState.NOT_TESTED: "Not currently observed",
    CapabilityState.NOT_LOADED: "Not loaded",
    CapabilityState.NOT_CONNECTED: "Provider unavailable",
    CapabilityState.UNAVAILABLE: "No usable price data",
}


def _short_unavailable_reason(row: MarketScannerRow) -> str:
    return _SHORT_REASON_BY_STATE.get(row.market_data_state, row.market_data_state.value.replace("_", " ").title())


def _row_to_secondary_dict(row: MarketScannerRow) -> dict[str, Any]:
    """The secondary "Data Details" table's row shape (Section 5): compact
    and honest -- Symbol, Product, Reason -- never a repeat of the primary
    table's full column set."""
    return {
        "Symbol": row.root_symbol,
        "Product": row.display_name,
        "Asset Class": ASSET_CLASS_LABELS[row.asset_class],
        "Reason": _short_unavailable_reason(row),
    }


def split_primary_secondary(
    rows: tuple[MarketScannerRow, ...],
) -> tuple[tuple[MarketScannerRow, ...], tuple[MarketScannerRow, ...]]:
    """Section 3/5: the PRIMARY table prioritizes products with real usable
    observational data; everything else (NOT_TESTED / NOT_LOADED /
    NOT_CONNECTED / UNAVAILABLE -- none of which carry a real price) moves to
    a secondary "Data Details" view instead of diluting the primary table
    with repeated unavailable-state text. A row with a real price (AVAILABLE
    or DEGRADED -- DEGRADED still has a real observed price, just no resolved
    raw contract symbol) is primary; a row with no real price is secondary.
    Pure and order-preserving so it composes directly with `sort_rows`/
    `filter_rows`'s own output."""
    primary = tuple(r for r in rows if r.last is not None)
    secondary = tuple(r for r in rows if r.last is None)
    return primary, secondary


def _render_capability_summary(*, catalog_size: int, n_observed_total: int, research_roots: tuple[str, ...]) -> None:
    """Section 4: research/trading capability explained ONCE, above the
    table, never repeated as a per-row column. Every value is read live from
    the same `alpha_agent.ui.market_universe` composition the per-row
    capability derivation already uses -- never a second, independently
    guessed summary -- and the three universes (Market/Research/Trading)
    stay scientifically distinct (CLAUDE.md / mission Part 2)."""
    trading_roots = market_universe.trading_universe()
    trading_text = ", ".join(trading_roots) if trading_roots else "None currently"
    st.caption(
        f"Research Universe: {', '.join(research_roots)}  ·  "
        f"Scientific Trading Eligibility: {trading_text}  ·  "
        f"{catalog_size} products catalogued, {n_observed_total} currently observable"
    )


# ---------------------------------------------------------------------------
# category-scoped lazy loading (Section 3/36) -- monotonic, session-scoped
# ---------------------------------------------------------------------------

#: A single bounded concurrent-fetch batch never exceeds this many roots,
#: independent of how large a category or the on-demand set happens to be --
#: defense in depth alongside `market_home`'s own per-root TTL cache.
MAX_CONCURRENT_FETCH = 8


def _loaded_roots_key(key_prefix: str) -> str:
    return f"{key_prefix}-scanner-loaded-roots"


def _fetched_once_key(key_prefix: str) -> str:
    return f"{key_prefix}-scanner-fetched-once"


def render_scanner(*, key_prefix: str = "market") -> None:
    """The ALL FUTURES scanner (mission Part 5). Clicking a row selects it
    (the same `st.session_state["market_selected_root"]` key every other
    Market-local selector on this platform already drives -- Sidebar IA pass,
    task spec section 1D: this state is Market-page-local, never a hidden
    global selector) -- it does not by itself
    imply that root has observation wired; the caller decides what to
    render below based on the row's own `market_data_state`. This release
    never claims a LIVE/streaming capability (see
    `alpha_agent.marketdata.databento_schemas.DatabentoCapability`'s
    docstring) -- the word "live" is deliberately avoided everywhere on this
    page; use "actively observed" / "connected" instead."""
    catalog = market_universe.market_universe()
    research_roots = market_universe.research_universe()
    trading_roots = market_universe.trading_universe()
    historical_roots = frozenset(services.catalog_summary()["roots"])

    # The SAME UI-session cache-first health read every other panel on this
    # platform uses (`market_home.get_health_cached`) -- never a second,
    # uncached `databento_context.health()` call that would defeat the "no
    # repeated provider calls across a bare rerun" invariant.
    health, health_refreshed = market_home.get_health_cached()
    research_frozen = frozenset(research_roots)
    # The Scanner's eager-fetch DEFAULT: the certified research universe PLUS
    # exactly one FEATURED_COVERAGE representative for every asset class the
    # research universe does not already touch -- deliberately NOT the same
    # set as `research_roots` (that stays scientific-certification-only,
    # mission Part 38), and deliberately NOT every featured root across every
    # category (Section 7: opening Market must never fetch more than a
    # handful of products). This guarantees every asset class has at least
    # one real observed row on the very first paint, without eagerly fetching
    # a whole category's featured basket before the user has asked for it.
    #
    # CATEGORY-SCOPED LAZY LOADING (Section 3/7): `loaded_roots` is a
    # monotonically-growing, session-scoped set of roots this session has
    # ever asked to observe -- the DEFAULT set, plus that category's own
    # FEATURED_COVERAGE basket once the user selects it (never the WHOLE
    # category -- a category can hold micro/ultra variants this pass
    # deliberately does not feature), plus any individually clicked row.
    # `fetched_once` (a SEPARATE session set) tracks which of those have
    # actually completed at least one real fetch; a root freshly ADDED to
    # `loaded_roots` this exact render is deliberately excluded from this
    # render's fetch/capability pass -- the table below paints its honest
    # NOT_LOADED row FIRST, then a bounded fetch runs and a single
    # `st.rerun()` (mirrors the existing click-to-select pattern) shows the
    # resolved state on the very next paint. The shell/table itself is never
    # blocked on the whole newly-selected category finishing.
    landing_extra = tuple(
        roots[0] for roots in FEATURED_COVERAGE.values() if not (research_frozen & frozenset(roots))
    )
    default_wired = research_frozen | frozenset(landing_extra)
    is_first_render_this_session = _loaded_roots_key(key_prefix) not in st.session_state
    loaded_roots: set[str] = st.session_state.setdefault(_loaded_roots_key(key_prefix), set(default_wired))
    fetched_once: set[str] = st.session_state.setdefault(_fetched_once_key(key_prefix), set())
    if is_first_render_this_session:
        # Fetched inline on first paint, exactly like the prior Checkpoint B
        # default did. Lazy, shell-first-then-rerun loading (below) applies
        # only to roots added LATER by a category selection or an on-demand
        # row click, which is what mission Part 3's worked examples ("User
        # selects Equity Index then...") are actually describing.
        fetched_once.update(default_wired)

    labels = ["All", *ASSET_CLASS_LABELS.values()]
    chosen = st.session_state.get(f"{key_prefix}-scanner-category", "All")
    asset_class = None
    if chosen != "All" and chosen in labels:
        asset_class = next(ac for ac, label in ASSET_CLASS_LABELS.items() if label == chosen)
        loaded_roots |= set(FEATURED_COVERAGE.get(asset_class, ()))

    attempt_now = frozenset(loaded_roots) & fetched_once
    pending_first_load = frozenset(loaded_roots) - fetched_once

    # Section 33: computed BEFORE capabilities so `capability_for`'s own
    # news_available/event_calendar_available fields reflect the SAME real
    # NewsStore/EventStore state as the row's news_state/event_state --
    # never a second, independently-derived guess.
    news_by_root = _news_state_for_universe(catalog)
    event_by_root = _event_state_for_universe(catalog)

    snapshots, bars_by_root = _fetch_wired_market_state(tuple(sorted(attempt_now))) if attempt_now else ({}, {})
    capabilities = {
        entry.root_symbol: market_universe.capability_for(
            entry.root_symbol, health=health, wired_roots=frozenset(loaded_roots),
            research_roots=research_frozen, trading_roots=frozenset(trading_roots), historical_roots=historical_roots,
            attempted=(entry.root_symbol in attempt_now), snapshot=snapshots.get(entry.root_symbol),
            news_state=news_by_root[entry.root_symbol][0], event_calendar_state=event_by_root[entry.root_symbol][0],
        )
        for entry in catalog
    }
    promise_verdict = _best_candidate_per_root()
    promise_by_root = {root: v[0] for root, v in promise_verdict.items()}
    verdict_by_root = {root: v[1] for root, v in promise_verdict.items()}

    rows = build_scanner_rows(
        catalog, capabilities=capabilities, snapshots=snapshots, bars_by_root=bars_by_root,
        promise_by_root=promise_by_root, verdict_by_root=verdict_by_root, news_by_root=news_by_root,
        event_by_root=event_by_root,
    )

    with components.card(f"{key_prefix}-scanner"):
        st.markdown('<div class="aa-gate-title">ALL FUTURES</div>', unsafe_allow_html=True)
        st.caption(
            f"Market data from Databento (GLBX.MDP3) · {health.capability.value} · "
            f"updated {health_refreshed.strftime('%H:%M:%S UTC')}"
        )
        c_search, c_sort = st.columns([2, 1])
        with c_search:
            search = st.text_input("Search products...", key=f"{key_prefix}-scanner-search", label_visibility="collapsed",
                                    placeholder="Search products...")
        with c_sort:
            sort_by = st.selectbox("Sort", SORT_OPTIONS, key=f"{key_prefix}-scanner-sort", label_visibility="collapsed")

        st.radio(
            "Category", labels, horizontal=True, key=f"{key_prefix}-scanner-category", label_visibility="collapsed",
        )

        visible = sort_rows(filter_rows(rows, asset_class=asset_class, search=search), sort_by)
        if sort_by == "Most News Activity" and not any(r.news_state is CapabilityState.AVAILABLE for r in visible):
            st.caption("News is NOT CONNECTED for every visible product -- this sort has nothing real to rank by yet.")
        if not visible:
            components.empty_state("Market Scanner", "No catalogued product matches this search/filter.",
                                    key=f"{key_prefix}-scanner-na")
            return

        primary, secondary = split_primary_secondary(visible)
        n_pending = len(pending_first_load & {r.root_symbol for r in visible})
        pending_note = f" · {n_pending} loading…" if n_pending else ""
        st.caption(f"{len(primary)} actively observed · {len(secondary)} not yet available{pending_note}.")
        _render_capability_summary(catalog_size=len(catalog), n_observed_total=sum(
            1 for r in rows if r.last is not None
        ), research_roots=research_roots)

        if primary:
            event = st.dataframe(
                [_row_to_table_dict(r) for r in primary],
                width="stretch", hide_index=True, height=min(560, 44 + 35 * len(primary)),
                on_select="rerun", selection_mode="single-row", key=f"{key_prefix}-scanner-table",
            )
            selected_rows = (event.selection.rows if event and hasattr(event, "selection") else []) or []
            picked_root = root_for_selection(primary, selected_rows)
            if picked_root is not None:
                # Section 3: "Clicking a NOT_LOADED row should trigger an
                # on-demand fetch for that product" -- a single-root fetch is
                # cheap enough to do synchronously, in the SAME rerun that
                # already fires below for the selection change.
                if picked_root not in fetched_once:
                    loaded_roots.add(picked_root)
                    _fetch_wired_market_state((picked_root,))
                    fetched_once.add(picked_root)
                if st.session_state.get("market_selected_root") != picked_root:
                    st.session_state["market_selected_root"] = picked_root
                    st.rerun()
        else:
            st.caption("No actively observed product matches this view yet -- see Data Details below.")

        with st.expander(f"Data Details ({len(secondary)})", expanded=False):
            st.caption(
                "CATALOGUED means this platform knows the product exists (real CME Globex/GLBX.MDP3 root); "
                "it does not by itself mean active observation, research certification, or paper-trading "
                "eligibility are available. NOT_LOADED means this root is supported but has not been fetched in "
                "this session yet (pick its category above to load its featured products, or search for it by "
                "name); NOT_TESTED means provider support for it has never been verified at all."
            )
            if secondary:
                st.dataframe(
                    [_row_to_secondary_dict(r) for r in secondary],
                    width="stretch", hide_index=True, height=min(360, 44 + 35 * len(secondary)),
                    key=f"{key_prefix}-scanner-table-secondary",
                )
            else:
                st.caption("Every catalogued product in this view is already actively observed.")
            for root, (label, verdict) in sorted(promise_verdict.items()):
                st.caption(f"{root}: best research candidate promise={label or '—'}, verdict={verdict or '—'}")

    # Committing the bounded fetch for a newly-selected category happens
    # LAST, after the table above has already rendered the honest NOT_LOADED
    # shell for those roots (Section 3: "the scanner shell must render
    # before every product finishes loading"). One rerun then shows the
    # resolved AVAILABLE/DEGRADED/UNAVAILABLE state -- never a polling loop,
    # never re-triggered by an unrelated rerun (`pending_first_load` is only
    # ever non-empty the render immediately after a NEW category/root is
    # added to `loaded_roots`).
    if pending_first_load:
        bounded = tuple(sorted(pending_first_load))[:MAX_CONCURRENT_FETCH]
        _fetch_wired_market_state(bounded)
        fetched_once.update(bounded)
        st.rerun()
