"""The UI's real-evidence assembly boundary for
:mod:`alpha_agent.context_retrieval` (Phase 3) -- mirrors
:mod:`alpha_agent.ui.opportunity_context`'s role for Opportunity V1 and
:mod:`alpha_agent.ui.translation_context`'s role for Phase 1: the ONE place
that turns already-established observation-plane accessors
(``alpha_agent.ui.market_home``, ``alpha_agent.marketdata.relative_markets``,
``alpha_agent.ui.market_intel_context``) into the typed inputs
:mod:`alpha_agent.context_retrieval` needs. The package itself stays pure and
network-free; every real read happens here.

CACHE-FIRST, NO NEW NETWORK POLICY: the live market-state reads
(trend/volatility/curve/related-market confirmation) go through
``market_home``'s existing cache-first accessors -- the SAME ones
``render_market_context``/``opportunity_context`` already call elsewhere on
this exact page, so this module introduces no new auto-fetch behavior.
Related prior event occurrences go through ``market_intel_context
.cached_recent_news`` specifically (never ``recent_news``, which can trigger
a live refresh) -- mirroring ``translation_context.candidate_observations``'s
own "no auto network from chat" discipline for this data source. Those
occurrences prove only root+event-category recurrence, never a
reconstructed historical trend/volatility/curve fingerprint at that past
timestamp (semantic hardening patch, section 5) -- no historical Databento
context reconstruction exists in this module.
"""
from __future__ import annotations

from datetime import UTC, datetime

from alpha_agent.context_retrieval.fingerprint import build_market_context_fingerprint
from alpha_agent.context_retrieval.retrieval import build_context_retrieval
from alpha_agent.context_retrieval.schemas import (
    ContextRetrievalResult,
    MarketContextFingerprint,
    RelatedPriorEventOccurrence,
)
from alpha_agent.marketdata.databento_schemas import CurveShape
from alpha_agent.marketdata.relative_markets import build_relative_snapshot, related_markets_for
from alpha_agent.recommendation.profile import InvestorProfile
from alpha_agent.translation.schemas import Observation
from alpha_agent.ui import market_home, market_intel_context, services

__all__ = ["build_context_for_observation", "related_prior_event_occurrences_for", "retrieve_for_observation"]

#: The SAME 1h/72-bar window `market_home`'s own Market Context /
#: Opportunity sections already use -- a real fetch here lands on the
#: provider's own request-level cache key those sections already populated
#: (never a second, differently-windowed fetch for the same root).
_LOOKBACK_BARS = market_home.SNAPSHOT_LOOKBACK_BARS

#: How far back `related_prior_event_occurrences_for` looks through the
#: ALREADY cached news store -- wide enough to surface a genuinely recurring
#: official-release context (e.g. weekly EIA petroleum reports) without
#: ever triggering a live fetch itself.
_HISTORICAL_WINDOW_HOURS = 24 * 180


def _fetch_bars(root: str) -> list[dict]:
    result, _ = market_home.get_ohlcv_cached(root, timeframe="1h", lookback_bars=_LOOKBACK_BARS)
    if not result or not result.fetched or not result.bars:
        return []
    return [b.model_dump(mode="json") for b in result.bars]


def _peer_confirmation(root: str, bars: list[dict]) -> tuple[int, int]:
    """Same same-sign-window-return rule as
    ``alpha_agent.ui.opportunity_context._peer_confirmation`` -- a small,
    independent implementation rather than a shared generic helper, matching
    how ``market_relative``/``opportunity_context`` already each keep their
    own focused peer-fetch logic in this codebase. `(0, 0)` when the root has
    no clear trend or no declared peer group."""
    trend = market_home.trend_state(bars)
    peers = related_markets_for(root)
    if trend not in ("Up", "Down") or not peers:
        return 0, 0
    snapshots = [
        build_relative_snapshot(
            root, peer.peer_root_symbol, selected_bars=bars, peer_bars=_fetch_bars(peer.peer_root_symbol),
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


def _curve_state(root: str) -> str | None:
    term_structure, _ = market_home.get_term_structure_cached(root)
    if term_structure is None:
        return None
    priced = [p for p in term_structure.points if p.price is not None]
    if len(priced) < 2 or term_structure.curve_shape is CurveShape.INSUFFICIENT_DATA:
        return None
    return term_structure.curve_shape.value


def build_context_for_observation(
    observation: Observation, *, evaluated_at: datetime | None = None,
) -> MarketContextFingerprint:
    """A real `MarketContextFingerprint` for `observation`, from
    already-established cache-first observation-plane accessors."""
    root = observation.root_symbol
    bars = _fetch_bars(root)
    trend = market_home.trend_state(bars)
    volatility = market_home.volatility_state(bars)
    curve_state = _curve_state(root)
    confirming, total = _peer_confirmation(root, bars)
    return build_market_context_fingerprint(
        root_symbol=root,
        event_type=observation.event_type,
        event_category=observation.structured_attributes.get("category", ""),
        affected_products=observation.affected_products,
        observed_at=observation.observed_at,
        evaluated_at=evaluated_at or datetime.now(UTC),
        trend=trend,
        volatility=volatility,
        curve_state=curve_state,
        related_market_confirming=confirming,
        related_market_total=total,
        structured_attributes=observation.structured_attributes,
    )


def related_prior_event_occurrences_for(
    observation: Observation, *, limit: int = 6,
) -> tuple[RelatedPriorEventOccurrence, ...]:
    """Real prior events of the SAME (root, event category) -- proving this
    KIND of event genuinely recurs -- from already-cached news only (see
    module docstring). Excludes the current observation's own news item when
    it came from one. NEVER a claim that the historical market context
    (trend/volatility/curve) at those past timestamps matched today's --
    only root+category recurrence is proven here (semantic hardening patch,
    section 5)."""
    category = observation.structured_attributes.get("category", "")
    own_news_id = observation.structured_attributes.get("news_id")
    items = market_intel_context.cached_recent_news(
        related_product=observation.root_symbol, window_hours=_HISTORICAL_WINDOW_HOURS, limit=200,
    )
    out: list[RelatedPriorEventOccurrence] = []
    for item in items:
        if item.category.value != category:
            continue
        if own_news_id is not None and item.news_id == own_news_id:
            continue
        out.append(
            RelatedPriorEventOccurrence(
                source="news",
                reference_id=item.news_id,
                headline=item.headline,
                observed_at=item.published_at,
                match_reason=f"category={item.category.value}, related_product={observation.root_symbol}",
            )
        )
    out.sort(key=lambda o: o.observed_at, reverse=True)
    return tuple(out[:limit])


def retrieve_for_observation(
    observation: Observation,
    *,
    investor_profile: InvestorProfile | None = None,
    evaluated_at: datetime | None = None,
) -> ContextRetrievalResult:
    """The one Agent-page entry point: real context + real related prior
    event occurrences, then delegates all scoring/ranking to the pure
    ``alpha_agent.context_retrieval.build_context_retrieval`` (never a
    registry write, never a network call beyond the cache-first reads
    above)."""
    context = build_context_for_observation(observation, evaluated_at=evaluated_at)
    occurrences = related_prior_event_occurrences_for(observation)
    with services.open_registry() as reg:
        return build_context_retrieval(
            observation,
            context=context,
            registry=reg,
            related_prior_event_occurrences=occurrences,
            investor_profile=investor_profile,
        )
