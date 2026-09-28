"""Market Intelligence Data Completion Pass, Checkpoint D -- Relative
Markets UI. Composes the pure `alpha_agent.marketdata.relative_markets`
registry/metrics with real, already-cached OHLCV bars from
`alpha_agent.ui.market_home` -- no new API, no new provider (Section 12).

Row/table building is a pure function (`snapshot_row`) for the same
testability reason as `alpha_agent.ui.market_scanner`; `render_relative_tab`
is the only Streamlit entry point.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import streamlit as st

from alpha_agent.marketdata.relative_markets import (
    RelativeMarketSnapshot,
    build_relative_snapshot,
    related_markets_for,
)
from alpha_agent.ui import charts_market, components, databento_context

__all__ = ["build_snapshots_for_root", "fetch_relative_bars", "render_relative_tab", "selected_root_row", "snapshot_row"]

#: Section 36 -- never fetch a related-market group serially. A group has at
#: most 6 members (FX) plus the selected root, well within one bounded batch.
MAX_CONCURRENT_PEER_FETCHES = 8


def fetch_relative_bars(
    roots: tuple[str, ...], *, timeframe: str = "1h", lookback_bars: int = 96,
) -> dict[str, list[dict[str, Any]]]:
    """Real, BOUNDED CONCURRENT fetch for every root in `roots` (the
    selected root plus its whole related-market group) -- Section 36: a
    serial loop over up to 7 real Databento round trips would make the
    Relative tab feel like it hung. Reuses the plain `databento_context`
    functions (not `market_home`'s session-cache wrapper) from worker
    threads -- `alpha_agent.ui.market_home.get_snapshots_for_universe`'s own
    docstring explains why `st.session_state` is never touched off the main
    thread; the provider's OWN process-lifetime TTL cache underneath still
    prevents a repeat real network call within its window regardless."""
    def _fetch(root: str) -> tuple[str, list[dict[str, Any]]]:
        result = databento_context.recent_ohlcv(root, timeframe=timeframe, lookback_bars=lookback_bars)
        bars = [b.model_dump(mode="python") for b in result.bars] if (result and result.fetched) else []
        return root, bars

    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_PEER_FETCHES, len(roots))) as pool:
        results = list(pool.map(_fetch, roots))
    return dict(results)


def build_snapshots_for_root(
    root: str, *, timeframe: str = "1h", lookback_bars: int = 96,
    bars_by_root: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[RelativeMarketSnapshot, ...]:
    """Real, bounded fetch (Section 12: SAME timeframe, SAME bounded window
    for every peer). ``bars_by_root`` lets a caller that already fetched the
    whole group (e.g. `render_relative_tab`, so the chart and the table
    share ONE real fetch pass) pass it straight through; omitting it fetches
    fresh via `fetch_relative_bars`."""
    peers = related_markets_for(root)
    if not peers:
        return ()

    if bars_by_root is None:
        all_roots = (root, *(p.peer_root_symbol for p in peers))
        bars_by_root = fetch_relative_bars(all_roots, timeframe=timeframe, lookback_bars=lookback_bars)

    selected_bars = bars_by_root.get(root, [])
    snapshots = []
    for peer in peers:
        peer_bars = bars_by_root.get(peer.peer_root_symbol, [])
        snapshots.append(
            build_relative_snapshot(
                root, peer.peer_root_symbol, selected_bars=selected_bars, peer_bars=peer_bars,
                relation_type=peer.relation_type, mapping_reason=peer.mapping_reason,
            )
        )
    return tuple(snapshots)


# ---------------------------------------------------------------------------
# pure row building
# ---------------------------------------------------------------------------


def selected_root_row(root: str, snapshots: tuple[RelativeMarketSnapshot, ...]) -> dict[str, Any]:
    """Section 14's worked example table always leads with the SELECTED
    root's own row (e.g. "For NQ: NQ, ES, YM, RTY") -- its own window
    return/realized vol come from any one non-insufficient snapshot (they
    are computed identically regardless of which peer it was paired with);
    correlation-to-self and relative-performance-to-self are trivial and
    rendered as such, never computed."""
    usable = next((s for s in snapshots if not s.insufficient_data), None)
    window_return = f"{usable.window_return_pct:+.2f}%" if usable else "N/A"
    vol = f"{usable.realized_volatility_pct:.1f}%" if usable else "N/A"
    return {
        "Product": root, "Window Return": window_return, "Realized Vol": vol,
        "Correlation": "1.00 (self)", "Relative Performance": "0.00% (self)",
        "Common Obs.": usable.common_observations if usable else 0,
    }


def snapshot_row(snapshot: RelativeMarketSnapshot) -> dict[str, Any]:
    if snapshot.insufficient_data:
        return {
            "Product": snapshot.peer_root_symbol,
            "Window Return": "INSUFFICIENT DATA",
            "Realized Vol": "INSUFFICIENT DATA",
            "Correlation": "INSUFFICIENT DATA",
            "Relative Performance": "INSUFFICIENT DATA",
            "Common Obs.": snapshot.common_observations,
        }
    return {
        "Product": snapshot.peer_root_symbol,
        "Window Return": f"{snapshot.peer_window_return_pct:+.2f}%" if snapshot.peer_window_return_pct is not None else "N/A",
        "Realized Vol": f"{snapshot.peer_realized_volatility_pct:.1f}%" if snapshot.peer_realized_volatility_pct is not None else "N/A",
        "Correlation": f"{snapshot.correlation:.2f}" if snapshot.correlation is not None else "N/A",
        # Section 13: "selected normalized return minus peer normalized
        # return" -- `relative_performance_pct` is already selected-minus-
        # peer, framed around the SELECTED root the whole tab is about.
        "Relative Performance": f"{snapshot.relative_performance_pct:+.2f}%" if snapshot.relative_performance_pct is not None else "N/A",
        "Common Obs.": snapshot.common_observations,
    }


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------


def render_relative_tab(root: str) -> None:
    """Product Detail -> Relative (Checkpoint D)."""
    peers = related_markets_for(root)
    if not peers:
        components.empty_state(
            "Relative", f"{root} has no declared related-market group yet -- no comparison peers configured.",
            key="market-relative-na",
        )
        return

    # ONE bounded concurrent fetch for the whole group -- shared by the
    # chart and the comparison table below (never two separate real fetch
    # passes for the same roots, Section 36).
    all_roots = (root, *(p.peer_root_symbol for p in peers))
    bars_by_root = fetch_relative_bars(all_roots)
    snapshots = build_snapshots_for_root(root, bars_by_root=bars_by_root)

    with components.card("market-relative"):
        st.markdown('<div class="aa-gate-title">RELATIVE MARKETS</div>', unsafe_allow_html=True)
        st.caption(
            f"{root} vs. its related-market group -- normalized returns and descriptive statistics only, "
            "never a trading signal. Grouping does not imply economic equivalence (see Data Details)."
        )

        if any(bars_by_root.values()):
            components.plotly_chart(charts_market.market_comparison_chart(bars_by_root), key="market-relative-fig")

        rows = [selected_root_row(root, snapshots), *[snapshot_row(s) for s in snapshots]]
        st.dataframe(rows, width="stretch", hide_index=True)

        with st.expander("Data Details -- mapping reasons", expanded=False):
            for snap in snapshots:
                components.provenance_row(
                    f"{root} vs {snap.peer_root_symbol} ({snap.relation_type.value})", snap.mapping_reason,
                )
                if not snap.insufficient_data and snap.price_ratio is not None:
                    st.caption(f"{root}/{snap.peer_root_symbol} price ratio: {snap.price_ratio:.4f}")
