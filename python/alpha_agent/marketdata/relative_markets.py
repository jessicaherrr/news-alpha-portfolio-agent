"""The RELATED-MARKET REGISTRY -- Market Intelligence Data Completion Pass,
Checkpoint D, Section 11. Pure, static, offline data (mirrors
`alpha_agent.marketdata.product_catalog`'s own "no network, no registry"
posture): declares which catalogued roots are grouped together for cross-
market comparison, and WHY, as an explicit, inspectable relation type --
never a bare "same group" implying economic equivalence that is not real
(Section 11: "Do not imply economic equivalence merely because products
share a group.").
"""
from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class RelationType(str, Enum):
    """Section 11's worked relation types, verbatim -- a transparent,
    inspectable reason a pair of roots is grouped for comparison, never a
    claim of tradable equivalence."""

    SAME_ASSET_CLASS = "SAME_ASSET_CLASS"
    REFINED_PRODUCT_RELATIONSHIP = "REFINED_PRODUCT_RELATIONSHIP"
    CURVE_NEIGHBOR = "CURVE_NEIGHBOR"
    EQUITY_INDEX_PEER = "EQUITY_INDEX_PEER"


class RelatedMarketEntry(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "related-market-entry/1"
    root_symbol: str
    peer_root_symbol: str
    relation_type: RelationType
    mapping_reason: str


#: Section 11's suggested initial groups, verbatim.
RELATED_MARKET_GROUPS: dict[str, tuple[str, ...]] = {
    "EQUITY_INDEX": ("ES", "NQ", "YM", "RTY"),
    "ENERGY": ("CL", "RB", "HO", "NG"),
    "METALS": ("GC", "SI", "HG"),
    "RATES": ("ZT", "ZF", "ZN", "ZB", "UB"),
    "FX": ("6E", "6J", "6B", "6A", "6C", "6S"),
    "AGRICULTURE": ("ZC", "ZW", "ZS", "ZM", "ZL"),
}

_GROUP_DEFAULT_RELATION: dict[str, RelationType] = {
    "EQUITY_INDEX": RelationType.EQUITY_INDEX_PEER,
    "ENERGY": RelationType.SAME_ASSET_CLASS,
    "METALS": RelationType.SAME_ASSET_CLASS,
    "RATES": RelationType.CURVE_NEIGHBOR,
    "FX": RelationType.SAME_ASSET_CLASS,
    "AGRICULTURE": RelationType.SAME_ASSET_CLASS,
}

_GROUP_DEFAULT_REASON: dict[str, str] = {
    "EQUITY_INDEX": "Both are broad U.S. equity index futures -- a comparison group only, no economic equivalence implied.",
    "ENERGY": "Both are CME/NYMEX energy-complex futures -- a comparison group only, no economic equivalence implied.",
    "METALS": "Both are COMEX metals futures -- a comparison group only, no economic equivalence implied.",
    "RATES": "Adjacent points on the U.S. Treasury futures curve (CURVE_NEIGHBOR) -- not interchangeable durations.",
    "FX": "Both are CME G10 FX futures -- a comparison group only, no economic equivalence implied.",
    "AGRICULTURE": "Both are CBOT grain/oilseed-complex futures -- a comparison group only, no economic equivalence implied.",
}

#: Pair-specific overrides where a MORE SPECIFIC, real economic relationship
#: exists beyond "same group" -- e.g. RBOB/Heating Oil are refined FROM
#: crude oil (the real-world "crack spread"); soybean meal/oil are
#: processed FROM soybeans (the real-world "crush spread"). Keyed by a
#: symmetric frozenset so lookup direction never matters.
_PAIR_OVERRIDES: dict[frozenset[str], tuple[RelationType, str]] = {
    frozenset({"CL", "RB"}): (
        RelationType.REFINED_PRODUCT_RELATIONSHIP,
        "RB (RBOB gasoline) is refined from crude oil (CL) -- the real-world 'crack spread' relationship.",
    ),
    frozenset({"CL", "HO"}): (
        RelationType.REFINED_PRODUCT_RELATIONSHIP,
        "HO (heating oil) is refined from crude oil (CL) -- the real-world 'crack spread' relationship.",
    ),
    frozenset({"ZS", "ZM"}): (
        RelationType.REFINED_PRODUCT_RELATIONSHIP,
        "ZM (soybean meal) is processed from soybeans (ZS) -- the real-world soybean 'crush spread' relationship.",
    ),
    frozenset({"ZS", "ZL"}): (
        RelationType.REFINED_PRODUCT_RELATIONSHIP,
        "ZL (soybean oil) is processed from soybeans (ZS) -- the real-world soybean 'crush spread' relationship.",
    ),
}


def group_for(root: str) -> str | None:
    root = root.upper()
    for group, roots in RELATED_MARKET_GROUPS.items():
        if root in roots:
            return group
    return None


def related_markets_for(root: str) -> tuple[RelatedMarketEntry, ...]:
    """Every OTHER root in `root`'s group(s), each with a transparent,
    inspectable relation type and reason. Empty for a root with no declared
    group (e.g. BTC/ETH crypto futures are not yet grouped) -- never a
    fabricated/guessed relation."""
    root = root.upper()
    group = group_for(root)
    if group is None:
        return ()
    entries = []
    for peer in RELATED_MARKET_GROUPS[group]:
        if peer == root:
            continue
        override = _PAIR_OVERRIDES.get(frozenset({root, peer}))
        if override is not None:
            relation, reason = override
        else:
            relation, reason = _GROUP_DEFAULT_RELATION[group], _GROUP_DEFAULT_REASON[group]
        entries.append(
            RelatedMarketEntry(root_symbol=root, peer_root_symbol=peer, relation_type=relation, mapping_reason=reason)
        )
    return tuple(entries)


# ---------------------------------------------------------------------------
# Section 12/13 -- deterministic cross-market metrics over already-fetched
# real OHLCV bars. Pure: no network, no Streamlit -- callers (`alpha_agent.
# ui.market_relative`) supply real bars fetched through the existing
# Databento observation path.
# ---------------------------------------------------------------------------

#: Section 13: "If sample too small: INSUFFICIENT DATA. Do not print a
#: meaningless correlation from a tiny sample." Chosen well above the bare
#: minimum (3) `_pearson_correlation` elsewhere on this page already uses,
#: since a *relative-markets* claim is a stronger, more visible statement.
MIN_COMMON_OBSERVATIONS = 10


class RelativeMarketSnapshot(BaseModel):
    """One selected-root/peer-root comparison, over the intersection of
    real, actually-common timestamps only -- Section 12: "Do not
    forward-fill missing price bars across closed markets simply to improve
    correlation sample size." `insufficient_data=True` means every
    statistic below is `None` except `common_observations` -- never a
    computed-but-meaningless value."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "relative-market-snapshot/1"
    root_symbol: str
    peer_root_symbol: str
    relation_type: RelationType
    mapping_reason: str
    common_observations: int
    insufficient_data: bool
    window_return_pct: float | None = None
    peer_window_return_pct: float | None = None
    realized_volatility_pct: float | None = None
    peer_realized_volatility_pct: float | None = None
    correlation: float | None = None
    relative_performance_pct: float | None = None
    price_ratio: float | None = None
    as_of: datetime


def align_by_timestamp(
    series_a: list[dict[str, Any]], series_b: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Section 12's mandatory alignment rule: intersection of real,
    actually-present timestamps only. Two closed-market gaps of different
    shapes (e.g. a holiday observed by one exchange, not another) simply
    drop those bars from BOTH series -- never forward-filled to pad the
    sample."""
    index_b = {bar["ts_event"]: bar for bar in series_b}
    aligned_a: list[dict[str, Any]] = []
    aligned_b: list[dict[str, Any]] = []
    for bar in series_a:
        match = index_b.get(bar["ts_event"])
        if match is not None:
            aligned_a.append(bar)
            aligned_b.append(match)
    return aligned_a, aligned_b


def normalized_returns(bars: list[dict[str, Any]]) -> list[float]:
    """``close_t / close_0 - 1`` per bar (Section 13, verbatim formula)."""
    if not bars or not bars[0].get("close"):
        return []
    base = bars[0]["close"]
    return [bar["close"] / base - 1.0 for bar in bars]


def log_returns(bars: list[dict[str, Any]]) -> list[float]:
    closes = [bar["close"] for bar in bars]
    return [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] and closes[i] and closes[i - 1] > 0 and closes[i] > 0
    ]


def realized_volatility_pct(bars: list[dict[str, Any]], *, bars_per_year: float = 24 * 252) -> float | None:
    rets = log_returns(bars)
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return (var ** 0.5) * (bars_per_year ** 0.5) * 100.0


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Descriptive only -- no causal or predictive claim (Section 13/14)."""
    n = min(len(x), len(y))
    if n < 2:
        return None
    x, y = x[-n:], y[-n:]
    mean_x, mean_y = sum(x) / n, sum(y) / n
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    var_x = sum((v - mean_x) ** 2 for v in x)
    var_y = sum((v - mean_y) ** 2 for v in y)
    denom = (var_x * var_y) ** 0.5
    return cov / denom if denom > 0 else None


def build_relative_snapshot(
    root: str, peer_root: str, *, selected_bars: list[dict[str, Any]], peer_bars: list[dict[str, Any]],
    relation_type: RelationType, mapping_reason: str, bars_per_year: float = 24 * 252,
) -> RelativeMarketSnapshot:
    aligned_sel, aligned_peer = align_by_timestamp(selected_bars, peer_bars)
    n = len(aligned_sel)
    now = datetime.now(UTC)
    if n < MIN_COMMON_OBSERVATIONS:
        return RelativeMarketSnapshot(
            root_symbol=root, peer_root_symbol=peer_root, relation_type=relation_type, mapping_reason=mapping_reason,
            common_observations=n, insufficient_data=True, as_of=now,
        )

    sel_closes = [bar["close"] for bar in aligned_sel]
    peer_closes = [bar["close"] for bar in aligned_peer]
    sel_norm = normalized_returns(aligned_sel)
    peer_norm = normalized_returns(aligned_peer)
    sel_window_return = sel_norm[-1] * 100.0 if sel_norm else None
    peer_window_return = peer_norm[-1] * 100.0 if peer_norm else None
    relative_performance = (
        sel_window_return - peer_window_return if (sel_window_return is not None and peer_window_return is not None)
        else None
    )
    price_ratio = (
        sel_closes[-1] / peer_closes[-1] if (sel_closes and peer_closes and peer_closes[-1]) else None
    )

    return RelativeMarketSnapshot(
        root_symbol=root, peer_root_symbol=peer_root, relation_type=relation_type, mapping_reason=mapping_reason,
        common_observations=n, insufficient_data=False,
        window_return_pct=sel_window_return, peer_window_return_pct=peer_window_return,
        realized_volatility_pct=realized_volatility_pct(aligned_sel, bars_per_year=bars_per_year),
        peer_realized_volatility_pct=realized_volatility_pct(aligned_peer, bars_per_year=bars_per_year),
        correlation=pearson_correlation(sel_closes, peer_closes),
        relative_performance_pct=relative_performance, price_ratio=price_ratio, as_of=now,
    )


__all__ = [
    "MIN_COMMON_OBSERVATIONS",
    "RELATED_MARKET_GROUPS",
    "RelatedMarketEntry",
    "RelationType",
    "RelativeMarketSnapshot",
    "align_by_timestamp",
    "build_relative_snapshot",
    "group_for",
    "log_returns",
    "normalized_returns",
    "pearson_correlation",
    "realized_volatility_pct",
    "related_markets_for",
]
