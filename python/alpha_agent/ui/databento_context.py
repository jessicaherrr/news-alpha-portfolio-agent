"""The UI's ONLY boundary into `alpha_agent.marketdata.databento_provider`
(mirrors `alpha_agent.ui.market_context`'s role for the IBKR provider, and
`alpha_agent.ui.services`'s role for the registry). Every page that wants a
Databento snapshot, recent OHLCV, or contract/economics metadata goes through
this module, never the provider class directly -- so there is exactly one
place that constructs a provider, one place that decides caching/refresh
behaviour for Streamlit, and one place the never-fabricate-a-value invariant
is enforced for this data source.

OBSERVATION-plane boundary (mission Part E): nothing here writes to
`data/raw/`/`data/processed/`, computes a feature, or touches the
`ExperimentRegistry` / `experiment_identity`. The one sanctioned extension
into scientific-adjacent territory is `observational_context_note`, which
builds the SAME closed, regex-matched `OBSERVATIONAL_CONTEXT_ONLY` string
template `alpha_agent.ui.market_context.observational_context_note` already
uses (see `alpha_agent.registry.holdout_guard`'s allowlisted providers/feeds)
-- never free text, never a feature, never validation evidence.
"""
from __future__ import annotations

from functools import lru_cache

from alpha_agent.marketdata.databento_provider import DatabentoMarketDataProvider
from alpha_agent.marketdata.databento_schemas import (
    UNAVAILABLE_CAPABILITIES,
    ContractEconomicsView,
    ContractLadderResult,
    ContractResolution,
    DatabentoHealth,
    MarketSnapshot,
    RecentOhlcvResult,
    TermStructureResult,
)
from alpha_agent.marketdata.product_catalog import market_universe_roots

#: Market Intelligence campaign Checkpoint B: every function below is
#: reachable for any CATALOGUED product (mission Part 4), not narrowed to
#: the certified RESEARCH UNIVERSE -- deliberately named apart from
#: "approved"/"research" so this boundary can never again be mistaken for a
#: scientific-certification gate (see `alpha_agent.ui.market_universe`'s
#: module docstring for the three-universe distinction this separation
#: exists to protect). A root outside this set returns `None` immediately,
#: with no provider/network call; a root inside it still gets no fabricated
#: value -- the provider itself fails closed per-root exactly as before.
OBSERVABLE_ROOTS: tuple[str, ...] = market_universe_roots()

__all__ = [
    "OBSERVABLE_ROOTS",
    "clear_cache",
    "contract_ladder",
    "contract_metadata",
    "health",
    "market_snapshot",
    "observational_context_note",
    "recent_ohlcv",
    "resolve_display_contract",
    "term_structure",
]


@lru_cache(maxsize=1)
def _provider() -> DatabentoMarketDataProvider:
    """One provider instance for the process's lifetime -- constructing it
    makes no network call by itself (the SDK client is built lazily on first
    real call); importing this module (or running the test suite, which
    never calls these functions against a real provider) makes no network
    call either."""
    return DatabentoMarketDataProvider()


def health() -> DatabentoHealth:
    try:
        return _provider().health()
    except Exception:  # noqa: BLE001 -- same defensive boundary as market_context.py
        from datetime import UTC, datetime

        from alpha_agent.marketdata.databento_schemas import DatabentoCapability

        return DatabentoHealth(
            capability=DatabentoCapability.ERROR, dataset="GLBX.MDP3", checked_at=datetime.now(UTC),
            detail="databento_context boundary caught an unexpected error",
        )


def market_snapshot(root: str | None) -> MarketSnapshot | None:
    if not root or root.upper() not in OBSERVABLE_ROOTS:
        return None
    try:
        return _provider().get_market_snapshot(root)
    except Exception:  # noqa: BLE001
        return None


def recent_ohlcv(root: str | None, *, timeframe: str = "1h", lookback_bars: int = 48) -> RecentOhlcvResult | None:
    if not root or root.upper() not in OBSERVABLE_ROOTS:
        return None
    try:
        return _provider().get_recent_ohlcv(root, timeframe=timeframe, lookback_bars=lookback_bars)
    except Exception:  # noqa: BLE001
        return None


def resolve_display_contract(root: str | None) -> ContractResolution | None:
    if not root or root.upper() not in OBSERVABLE_ROOTS:
        return None
    try:
        return _provider().resolve_display_contract(root)
    except Exception:  # noqa: BLE001
        return None


def contract_metadata(root: str | None) -> ContractEconomicsView | None:
    if not root or root.upper() not in OBSERVABLE_ROOTS:
        return None
    try:
        return _provider().get_contract_metadata(root)
    except Exception:  # noqa: BLE001
        return None


def contract_ladder(root: str | None, *, max_contracts: int = 12) -> ContractLadderResult | None:
    """Checkpoint C -- the real, expiry-ordered outright-contract ladder.
    `None` for an uncatalogued root (never a fabricated empty ladder)."""
    if not root or root.upper() not in OBSERVABLE_ROOTS:
        return None
    try:
        return _provider().get_contract_ladder(root, max_contracts=max_contracts)
    except Exception:  # noqa: BLE001
        return None


def term_structure(root: str | None, *, max_contracts: int = 12, tolerance_pct: float = 0.05) -> TermStructureResult | None:
    if not root or root.upper() not in OBSERVABLE_ROOTS:
        return None
    try:
        return _provider().get_term_structure(root, max_contracts=max_contracts, tolerance_pct=tolerance_pct)
    except Exception:  # noqa: BLE001
        return None


def clear_cache() -> None:
    """The explicit "Refresh Market Data" UI action (mission section 16) --
    never invoked automatically on a bare page render/rerun."""
    try:
        _provider().clear_cache()
    except Exception:  # noqa: BLE001, S110 -- best-effort refresh, never fails a page render
        pass


def observational_context_note(root: str | None) -> str | None:
    """A single, clearly-tagged free-text string for the Research Agent's
    `ResearchContext.knowledge_base` -- the SAME sanctioned extension point
    `alpha_agent.ui.market_context.observational_context_note` uses for IBKR.
    Explicitly labeled `OBSERVATIONAL_CONTEXT_ONLY` so nothing downstream can
    mistake this for scientific evidence; it never enters a feature, a
    backtest, validation, `experiment_identity`, or paper eligibility.
    Returns `None` whenever no snapshot/price is available -- the agent then
    proceeds exactly as it always has, with no market-context line at all."""
    snap = market_snapshot(root)
    if snap is None or snap.last is None or snap.capability in UNAVAILABLE_CAPABILITIES:
        return None
    return (
        f"OBSERVATIONAL_CONTEXT_ONLY provider=DATABENTO feed={snap.capability.value} "
        f"root={snap.root_symbol} last={snap.last:g} as_of={snap.as_of.isoformat()} "
        "-- delayed market color for conversational context only; not scientific "
        "evidence, not a feature, not used in any validation or paper-eligibility "
        "decision."
    )
