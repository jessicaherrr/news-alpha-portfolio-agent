"""Phase 5 -- the one genuinely NEW concept this phase adds: a domain-
qualified instrument identity.

Every existing schema (``StrategySpec.root_symbol``, ``Observation.root_symbol``,
``AlphaResearchObject.root_symbol``, every registry table) keeps its own
``root_symbol: str`` field exactly as-is -- Phase 5 instruction 5 forbids
rewriting historical identities, so nothing here renames, migrates, or
replaces those fields. :class:`InstrumentIdentity` is an additive, on-demand
VIEW over a root symbol, not a new source of truth: it exists so a future
generation (Phase 6 ETF Pilot onward) has a real typed shape to construct its
own instrument identities with, without implying today's Futures instruments
were ever anything but Futures.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# AssetDomain's canonical home is alpha_agent.registry.enums, NOT this module
# -- re-exported here unchanged (`is`-identical, the same zero-cost-aliasing
# pattern every other `core` re-export uses) so downstream/UI-facing code can
# keep reading it as `core.instrument.AssetDomain`. The registry package sits
# BELOW `alpha_agent.core` in the import graph (many `core` re-exports pull
# FROM `registry.enums` already: RegistryVerdict/Authority/TrialRole), so a
# type the registry itself needs to store in a column (Phase 6's
# `experiments.asset_domain`) cannot be DEFINED here without a real circular
# import (`registry -> core -> alpha_memory -> agents -> registry`, hit for
# real 2026-09-23 building the Phase 6 registry namespacing work) -- see
# `registry.enums.AssetDomain`'s own docstring for the full explanation.
from alpha_agent.registry.enums import AssetDomain

__all__ = [
    "ROOT_SYMBOL_PATTERN",
    "AssetDomain",
    "InstrumentIdentity",
    "equity_instrument_identity",
    "etf_instrument_identity",
    "futures_instrument_identity",
]

#: Byte-identical to ``alpha_agent.strategy.spec.StrategySpec.root_symbol``'s
#: own pattern (proven, never merely asserted similar, by
#: ``test_phase5_asset_neutral_core.py``) -- Phase 5 does not loosen or
#: reinterpret the Futures root-symbol shape.
ROOT_SYMBOL_PATTERN = r"^[A-Z0-9]{1,12}$"


class InstrumentIdentity(BaseModel):
    """A domain-qualified instrument identity -- the asset-neutral SHAPE a
    future generation will also need. Wraps today's Futures root symbol
    without replacing it anywhere it already lives."""

    model_config = {"frozen": True, "extra": "forbid"}

    asset_domain: AssetDomain
    symbol: str = Field(pattern=ROOT_SYMBOL_PATTERN)


def futures_instrument_identity(root_symbol: str) -> InstrumentIdentity:
    """The only constructor Phase 5 provides: wraps an existing Futures
    ``root_symbol`` as a domain-qualified :class:`InstrumentIdentity`. Pure
    construction -- it reads nothing, writes nothing, and is not called by
    any existing Futures code path (additive only)."""

    return InstrumentIdentity(asset_domain=AssetDomain.FUTURES, symbol=root_symbol)


def etf_instrument_identity(symbol: str) -> InstrumentIdentity:
    """Phase 6's mirror of :func:`futures_instrument_identity`: wraps an ETF
    ticker (e.g. ``"SPY"``) as a domain-qualified :class:`InstrumentIdentity`.
    Pure construction -- reads nothing, writes nothing."""

    return InstrumentIdentity(asset_domain=AssetDomain.ETF, symbol=symbol)


def equity_instrument_identity(symbol: str) -> InstrumentIdentity:
    """Phase 9.1's mirror of :func:`etf_instrument_identity`: wraps a single-
    name equity ticker (e.g. ``"AAPL"``) as a domain-qualified
    :class:`InstrumentIdentity`. Pure construction -- reads nothing, writes
    nothing. ``AssetDomain.EQUITY != AssetDomain.ETF``, so an equity and an
    ETF sharing a symbol string (not expected in practice, but not
    structurally prevented) still resolve to distinct identities."""

    return InstrumentIdentity(asset_domain=AssetDomain.EQUITY, symbol=symbol)
