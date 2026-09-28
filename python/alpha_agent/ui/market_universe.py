"""The THREE UNIVERSES (mission Part 2) + typed capability derivation
(mission Part 3) -- Market Intelligence + Futures Universe campaign,
Checkpoint A.

This is a UI-layer composition module, not an observation-plane module: it
is the one place that combines the static MARKET UNIVERSE catalog
(`alpha_agent.marketdata.product_catalog`, no network/registry) with the
live Databento OBSERVATION plane (`alpha_agent.ui.databento_context`) and
read-only SCIENTIFIC-plane evidence (`alpha_agent.ui.services`, which already
reads the registry for display) into one typed
`alpha_agent.marketdata.capability.FuturesProductCapability` per product.
Exactly like `alpha_agent.ui.dashboard`/`alpha_agent.ui.panels` already
compose observation + registry reads for other pages -- this module is not
new architectural territory, only a new, explicit, reusable name for a
composition every page needs (mission Part 32: "create clear service/store
abstractions").

MANDATORY DISTINCTION (mission Part 2), enforced by construction here, never
merely documented:

* ``market_universe()`` -- every CATALOGUED product. Observation only; implies
  nothing about research or trading readiness.
* ``research_universe()`` -- `alpha_agent.marketdata.product_catalog.
  RESEARCH_UNIVERSE`, the certified ES/NQ/CL/GC/ZN roots (CLAUDE.md Part 38:
  must never expand silently). Re-exported here, not recomputed, so this
  module can never itself become a second source of truth.
* ``trading_universe()`` -- computed FRESH from
  `alpha_agent.ui.services.paper_eligible_experiments()` every call: the
  roots that currently carry at least one paper-eligible (PASS +
  reconstructable) registry experiment. Real registry state today is 0/N, so
  this legitimately returns an empty tuple -- never inferred from
  `research_universe()` membership (mission Part 2: "Do not infer: market
  data available => trading enabled").
"""
from __future__ import annotations

from alpha_agent.marketdata.capability import CapabilityState, FuturesProductCapability
from alpha_agent.marketdata.databento_schemas import (
    UNAVAILABLE_CAPABILITIES,
    DatabentoHealth,
    MarketSnapshot,
)
from alpha_agent.marketdata.product_catalog import (
    PRODUCT_CATALOG,
    RESEARCH_UNIVERSE,
    ProductCatalogEntry,
    catalog_entry,
    market_universe_roots,
)
from alpha_agent.ui import services

__all__ = [
    "capability_for",
    "market_universe",
    "market_universe_roots",
    "research_universe",
    "trading_universe",
]


def market_universe() -> tuple[ProductCatalogEntry, ...]:
    """Every catalogued product, in the catalog's own declared category
    order (mission Part 4)."""
    return PRODUCT_CATALOG


def research_universe() -> tuple[str, ...]:
    """The certified scientific research roots -- a straight re-export of
    the one frozen constant (see module docstring)."""
    return RESEARCH_UNIVERSE


def trading_universe() -> tuple[str, ...]:
    """Roots with at least one currently paper-eligible registry experiment,
    sorted and de-duplicated. Computed fresh every call from
    `services.paper_eligible_experiments()` -- never cached here, since a
    stale "eligible" answer for a trading gate would be a real integrity
    risk; `services.py` already applies its own registry-level caching."""
    roots = {row["root_symbol"] for row in services.paper_eligible_experiments()}
    return tuple(sorted(roots))


def _historical_data_state(root: str, catalog_roots_with_data: frozenset[str]) -> CapabilityState:
    return CapabilityState.AVAILABLE if root in catalog_roots_with_data else CapabilityState.NOT_TESTED


#: Market Intelligence Data Completion Pass, Section 1 -- explicit,
#: non-ambiguous observation states. NOT_TESTED ("do we even believe
#: Databento supports this root") is deliberately distinct from NOT_LOADED
#: ("supported, but no fetch has happened yet this session/cache window"):
#: a product a user has never navigated to must never look identical to one
#: this platform has actually tried and failed to observe.
def _market_data_state(
    root: str,
    *,
    wired_roots: frozenset[str],
    health: DatabentoHealth | None,
    attempted: bool = False,
    snapshot: MarketSnapshot | None = None,
) -> tuple[CapabilityState, str]:
    if root not in wired_roots:
        return (
            CapabilityState.NOT_TESTED,
            (
                "Catalogued, but provider support for this root has not yet been verified "
                "(no continuous-contract resolution has ever been attempted for it)."
            ),
        )
    if health is None or health.capability in UNAVAILABLE_CAPABILITIES:
        return (
            CapabilityState.NOT_CONNECTED,
            "Databento itself is not currently connected -- this is a provider-wide outage, not a per-product gap.",
        )
    if not attempted:
        return (
            CapabilityState.NOT_LOADED,
            "Provider support is verified for this root, but no fetch has been made in the current cache window yet.",
        )
    if snapshot is None or snapshot.last is None:
        return (
            CapabilityState.UNAVAILABLE,
            "A real fetch was attempted for this root and Databento reported no usable price data.",
        )
    if snapshot.contract.resolved_raw_symbol is None:
        return (
            CapabilityState.DEGRADED,
            (
                "Price and contract economics are available, but the raw display-contract symbol did not "
                "resolve for this root in the probed window -- partial observation, not a failed one."
            ),
        )
    return CapabilityState.AVAILABLE, f"Actively observed via Databento ({health.capability.value})."


def capability_for(
    root: str,
    *,
    health: DatabentoHealth | None = None,
    wired_roots: frozenset[str] | None = None,
    research_roots: frozenset[str] | None = None,
    trading_roots: frozenset[str] | None = None,
    historical_roots: frozenset[str] | None = None,
    attempted: bool = False,
    snapshot: MarketSnapshot | None = None,
    news_state: CapabilityState | None = None,
    event_calendar_state: CapabilityState | None = None,
) -> FuturesProductCapability:
    """Build one product's typed capability readout (mission Part 3).

    Every keyword argument is an injectable override so callers that already
    fetched a health probe / registry summary for a whole scanner render
    (mission Part 44: bounded concurrency, one fetch, not N) pass it in
    once; omitting all of them makes exactly one real call to each of
    `alpha_agent.ui.databento_context.health`, `research_universe()`,
    `trading_universe()`, and `services.catalog_summary()` -- safe for a
    single product lookup, wasteful for a 36-row scanner.

    ``attempted``/``snapshot`` (Market Intelligence Data Completion Pass,
    Section 1) let a caller that already fetched (or deliberately has NOT
    yet fetched) this root's snapshot report the honest NOT_LOADED /
    UNAVAILABLE / DEGRADED / AVAILABLE distinction -- ``attempted=False``
    (the default) means "no fetch was made this call", never "fetch failed".

    ``news_state``/``event_calendar_state`` (Section 33) let a caller that
    already computed real per-root NewsStore/EventStore state (e.g. the
    Scanner's own ``_news_state_for_universe``/``_event_state_for_universe``)
    pass it straight through -- omitting either keeps the honest DISABLED
    default (this module never imports ``market_intel_context`` itself, to
    avoid a real network-adjacent dependency inside a "pure composition"
    function that many callers invoke per-row).

    Raises ``ValueError`` for an uncatalogued root -- every caller is
    expected to iterate `market_universe()` / `market_universe_roots()`
    first, so reaching this with an unknown root is a caller bug, not a
    runtime "unavailable" state.
    """
    entry = catalog_entry(root)
    if entry is None:
        raise ValueError(f"{root!r} is not in the futures product catalog")

    if wired_roots is None:
        wired_roots = frozenset(research_roots or research_universe())
    if health is None:
        from alpha_agent.ui import databento_context

        health = databento_context.health()
    if research_roots is None:
        research_roots = frozenset(research_universe())
    if trading_roots is None:
        trading_roots = frozenset(trading_universe())
    if historical_roots is None:
        historical_roots = frozenset(services.catalog_summary()["roots"])

    market_data_state, market_reason = _market_data_state(
        entry.root_symbol, wired_roots=wired_roots, health=health, attempted=attempted, snapshot=snapshot,
    )
    historical_state = _historical_data_state(entry.root_symbol, historical_roots)
    is_research = entry.root_symbol in research_roots
    is_trading = entry.root_symbol in trading_roots

    # Contract economics / roll logic / C++ execution are all verified as one
    # bundle by the frozen Phase 13.5C onboarding process (CLAUDE.md
    # "Contract economics" + "roll-close marks" sections) -- there is no
    # partial state for a certified research root, and no root outside
    # RESEARCH_UNIVERSE has had this onboarding done at all (mission Part 38:
    # never silently reuse ES economics/assumptions for an uncertified root).
    onboarding_state = CapabilityState.AVAILABLE if is_research else CapabilityState.NOT_TESTED

    reasons = [market_reason]
    if is_research:
        reasons.append("Certified RESEARCH UNIVERSE root (CLAUDE.md-frozen contract economics, roll logic, C++ execution).")
    else:
        reasons.append(
            "Not a certified research root -- contract economics/roll logic/execution have not been onboarded; "
            "never assume ES-style economics for this product."
        )
    if is_trading:
        reasons.append("At least one registry experiment for this root is currently paper-trading eligible.")
    else:
        reasons.append("No registry experiment for this root is currently paper-trading eligible.")

    # Only ever timestamp a REAL fetch attempt -- membership in `wired_roots`
    # alone is "support believed available" (NOT_LOADED until proven),
    # never itself a "just checked" claim (Section 1: a blank/NOT_LOADED row
    # must never be dressed up with a fabricated verification timestamp).
    verified_at = health.checked_at if (attempted and health is not None) else None

    return FuturesProductCapability(
        root_symbol=entry.root_symbol,
        display_name=entry.display_name,
        asset_class=entry.asset_class,
        venue=entry.venue,
        dataset=entry.dataset,
        market_data_available=market_data_state,
        historical_data_available=historical_state,
        news_available=news_state if news_state is not None else CapabilityState.DISABLED,
        event_calendar_available=event_calendar_state if event_calendar_state is not None else CapabilityState.DISABLED,
        research_enabled=is_research,
        paper_trading_enabled=is_trading,
        challenge_eligible=False,
        contract_economics_verified=onboarding_state,
        roll_logic_verified=onboarding_state,
        cpp_execution_verified=onboarding_state,
        capability_reason=" ".join(reasons),
        verified_at=verified_at,
    )
