"""Opportunity V1 -- real evidence assembly (Product Consolidation +
Opportunity V1 campaign, Checkpoint B).

The ONE place that turns EXISTING observation-plane evidence
(`alpha_agent.ui.market_home`, `market_relative`, `market_intel_context`) and
read-only registry evidence (`alpha_agent.ui.services`) into typed
`alpha_agent.opportunity.schemas.OpportunityInputs` for the certified
Research Universe. No new data pipeline and no new API are introduced here
(task spec section 8) -- every value below is a call into an accessor this
platform already had before this campaign.

The actual "what does this evidence mean" decision logic
(setup/why_now/risks/catalyst/strengthen/invalidate/research_status/
next_action, plus the internal ranking) lives in the pure, Streamlit-free
`alpha_agent.opportunity.schemas` module and is unit-tested there with no
live data; this module's own job is ONLY assembly.

PERFORMANCE (real-live-browser verification finding, Checkpoint F): a first
version fetched each certified root's own bars via `market_home`'s
UI-session-cached wrapper, THEN separately re-fetched (a different lookback,
a real cache miss) that SAME root's bars again inside relative-market peer
confirmation -- serially, root by root. Against a real Databento
entitlement that measured 80+ seconds to paint a single "Top Opportunities"
section, which fails this campaign's own "simple, calm" acceptance bar.
`top_opportunities` now fetches every root's bars EXACTLY ONCE, for the
union of the certified roots and their declared relative-market peers, in
one bounded-concurrency pass (mirrors `market_home.get_snapshots_for_universe`
/ `market_relative.fetch_relative_bars`'s own already-established pattern:
worker threads call the plain `databento_context` functions directly,
`st.session_state` is touched only on the main thread, before/after the
pool). Term structure is fetched the same way, concurrently across roots.

SCIENTIFIC ISOLATION: every function below calls a READ-ONLY accessor --
`services.research_candidate_summaries` (services.py's own read-only
boundary), `market_home`/`market_relative`/`databento_context` (observation
plane, live/recent market data), `market_intel_context` (observation plane,
news/events). Nothing here writes to the `ExperimentRegistry`, mutates a
scientific verdict or Research Promise, or can reach the sealed 2025
holdout -- the providers this module calls read live/recent market data and
the already-frozen 2018-2024 registry only, exactly like every other
Market/Agent-page accessor.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from alpha_agent.marketdata.databento_schemas import (
    UNAVAILABLE_CAPABILITIES,
    CurveShape,
    TermStructureResult,
)
from alpha_agent.opportunity.schemas import (
    CatalystInfo,
    OpportunityInputs,
    OpportunitySnapshot,
    TrendDirection,
    rank_opportunities,
)
from alpha_agent.ui import (
    databento_context,
    market_home,
    market_intel_context,
    market_relative,
    market_universe,
    services,
)

__all__ = ["top_opportunities"]

#: The single OHLCV timeframe every root's bars (and the window
#: strengthen/invalidate levels derived from them) are fetched at -- carried
#: onto `OpportunityInputs.window_timeframe` so the UI can state the real
#: window size accurately instead of implying a technical model that does
#: not exist (Product Acceptance Fix Pass, item 4).
_OHLCV_TIMEFRAME = "1h"

_TREND_LABELS = {
    "Up": TrendDirection.UP, "Down": TrendDirection.DOWN, "Sideways": TrendDirection.SIDEWAYS,
}

_CURVE_SHAPE_LABEL = {
    CurveShape.CONTANGO: "Contango",
    CurveShape.BACKWARDATION: "Backwardation",
    CurveShape.FLAT: "Flat",
    CurveShape.MIXED: "Mixed",
    CurveShape.INSUFFICIENT_DATA: "Insufficient Data",
}

#: Never fetch more than this many roots' term structure concurrently in one
#: pass -- mirrors `market_relative.MAX_CONCURRENT_PEER_FETCHES`'s own bound.
_MAX_CONCURRENT_TERM_STRUCTURE_FETCHES = 8


def _trend_direction(bars: list[dict]) -> TrendDirection:
    return _TREND_LABELS.get(market_home.trend_state(bars), TrendDirection.UNKNOWN)


def _fetch_term_structures(roots: tuple[str, ...]) -> dict[str, TermStructureResult | None]:
    """One term-structure lookup per root, concurrently -- worker threads
    call `databento_context.term_structure` directly (never the
    `market_home` session-cache wrapper, which is main-thread only; see
    `market_relative.fetch_relative_bars`'s own docstring for why)."""
    def _fetch(root: str) -> tuple[str, TermStructureResult | None]:
        try:
            return root, databento_context.term_structure(root, max_contracts=12)
        except Exception:  # noqa: BLE001 -- one root's failure must never block the others
            return root, None

    with ThreadPoolExecutor(max_workers=min(_MAX_CONCURRENT_TERM_STRUCTURE_FETCHES, len(roots))) as pool:
        results = list(pool.map(_fetch, roots))
    return dict(results)


def _peer_confirmation(
    root: str, bars: list[dict], peers: tuple, bars_by_root: dict[str, list[dict]],
) -> tuple[int, int]:
    """How many of `root`'s declared related-market peers moved the SAME
    direction as `root` over the same observed window, using bars already
    fetched by `top_opportunities` (never a second fetch). Deliberately
    simple and auditable: same-sign window return only -- never a fabricated
    peer, and 0/0 (not "confirming") when `root` itself has no clear trend
    or has no declared peer group."""
    trend = market_home.trend_state(bars)
    if trend not in ("Up", "Down") or not peers:
        return 0, 0
    snapshots = [
        market_relative.build_relative_snapshot(
            root, peer.peer_root_symbol, selected_bars=bars, peer_bars=bars_by_root.get(peer.peer_root_symbol, []),
            relation_type=peer.relation_type, mapping_reason=peer.mapping_reason,
        )
        for peer in peers
    ]
    usable = [s for s in snapshots if not s.insufficient_data and s.peer_window_return_pct is not None]
    if not usable:
        return 0, len(snapshots)
    root_sign = 1 if trend == "Up" else -1
    confirming = sum(1 for s in usable if (1 if s.peer_window_return_pct > 0 else -1) == root_sign)
    return confirming, len(snapshots)


def _catalyst_for(root: str) -> CatalystInfo | None:
    event = market_intel_context.next_high_impact_event_for_product(root)
    if event is None:
        return None
    return CatalystInfo(
        name=event.name, source_name=event.source_name, scheduled_at=event.scheduled_at,
        importance=event.importance.value,
    )


def _research_evidence(root: str, candidates: list) -> tuple[str | None, str | None, str | None]:
    """The single best existing candidate for this root -- PASS-first, then
    highest Research Promise -- or `(None, None, None)` when nothing has
    been tested here yet (never fabricated)."""
    own = [c for c in candidates if c.root_symbol == root]
    if not own:
        return None, None, None
    best = max(own, key=lambda c: (c.scientific_verdict == "PASS", c.research_promise_score))
    return best.scientific_verdict, best.research_promise_label, best.strategy_family


def _build_opportunity_inputs(
    root: str,
    *,
    bars_by_root: dict[str, list[dict]],
    peers_by_root: dict[str, tuple],
    term_structures: dict[str, TermStructureResult | None],
    candidates: list,
    observed_at: datetime,
) -> OpportunityInputs | None:
    """Assemble one root's `OpportunityInputs` from already-fetched evidence
    -- `None` only when this root's own bars never actually resolved (never
    a fabricated snapshot, mirroring every other Market-page accessor)."""
    bars = bars_by_root.get(root) or []
    if len(bars) < 2:
        return None

    window_low = min(b["low"] for b in bars)
    window_high = max(b["high"] for b in bars)

    curve_shape: str | None = None
    term_structure = term_structures.get(root)
    if term_structure is not None and len([p for p in term_structure.points if p.price is not None]) >= 2:
        curve_shape = _CURVE_SHAPE_LABEL.get(term_structure.curve_shape, str(term_structure.curve_shape.value))

    peer_confirming, peer_total = _peer_confirmation(root, bars, peers_by_root.get(root, ()), bars_by_root)
    news_count = market_intel_context.news_count_for_product(root)
    catalyst = _catalyst_for(root)
    verdict, promise_label, family = _research_evidence(root, candidates)

    entry = next((e for e in market_universe.market_universe() if e.root_symbol == root), None)
    display_name = entry.display_name if entry is not None else root

    return OpportunityInputs(
        root_symbol=root,
        display_name=display_name,
        trend=_trend_direction(bars),
        volatility=market_home.volatility_state(bars),
        volume_context=market_home.volume_context_state(bars),
        session_position=market_home.session_position_state(bars),
        window_low=window_low,
        window_high=window_high,
        window_bars=len(bars),
        window_timeframe=_OHLCV_TIMEFRAME,
        curve_shape=curve_shape,
        peer_confirming_count=peer_confirming,
        peer_total_count=peer_total,
        news_count_24h=news_count,
        catalyst=catalyst,
        research_verdict=verdict,
        research_promise_label=promise_label,
        best_strategy_family=family,
        observed_at=observed_at,
    )


def top_opportunities(
    *, limit: int = 3, universe: tuple[str, ...] | None = None,
) -> list[OpportunitySnapshot]:
    """The Agent page's "TOP OPPORTUNITIES" list -- the certified Research
    Universe by default (task spec section 8's evidence inputs, notably
    "existing research status"/"Research Promise", are only meaningful
    there); a caller may narrow `universe` explicitly (e.g. to Market's own
    single selected product) for a one-product read. Returns fewer than `limit`
    (down to an empty list) whenever the observation plane cannot honestly
    say anything about the remaining roots -- never padded with a
    fabricated entry.

    Every root's bars are fetched EXACTLY ONCE (see this module's docstring
    for the real-latency bug this fixed): one bounded-concurrency pass over
    the union of `roots` and their declared relative-market peers."""
    roots = universe if universe is not None else market_universe.research_universe()
    if not roots:
        return []

    health, _ = market_home.get_health_cached()
    if health.capability in UNAVAILABLE_CAPABILITIES:
        return []

    peers_by_root = {root: market_relative.related_markets_for(root) for root in roots}
    all_roots: set[str] = set(roots)
    for peers in peers_by_root.values():
        all_roots.update(p.peer_root_symbol for p in peers)

    bars_by_root: dict[str, list[dict[str, Any]]] = market_relative.fetch_relative_bars(
        tuple(all_roots), timeframe=_OHLCV_TIMEFRAME, lookback_bars=market_home.SNAPSHOT_LOOKBACK_BARS,
    )
    term_structures = _fetch_term_structures(roots)
    candidates = list(services.research_candidate_summaries())
    observed_at = datetime.now(UTC)

    all_inputs = [
        x for x in (
            _build_opportunity_inputs(
                root, bars_by_root=bars_by_root, peers_by_root=peers_by_root,
                term_structures=term_structures, candidates=candidates, observed_at=observed_at,
            )
            for root in roots
        )
        if x is not None
    ]
    return rank_opportunities(all_inputs)[:limit]
