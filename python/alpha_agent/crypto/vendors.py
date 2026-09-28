"""Vendor-neutral typed adapter interfaces for crypto derivatives / on-chain data.

Prompt 22 names four external data surfaces: perpetual funding/open-interest/
liquidations "from approved vendors", exchange flows, MVRV/SOPR/active-address
style on-chain features, and CME BTC/ETH futures basis. No vendor is named or
approved anywhere in this repository today (checked at Phase 22 kickoff), and
connecting one is a
MONEY/NETWORK decision CLAUDE.md's autonomous-execution section reserves for
the user. These :class:`typing.Protocol` classes are the STABLE, vendor-neutral
contract a real adapter would implement later; adding a real vendor means
writing one class that satisfies a Protocol here, with **zero** change to the
downstream feature / StrategySpec / C++ / validation semantics that consume its
output.

Today exactly one family of implementations exists --
:mod:`alpha_agent.crypto.synthetic_fixtures` -- and every one of them is class-
tagged ``provenance_role = DataProvenanceRole.SYNTHETIC``.
:func:`assert_vendor_is_synthetic_only` is the mechanical guard that a real,
un-vetted adapter can never be wired into the Phase 22 scaffold path by
accident: it is the same shape as
:func:`alpha_agent.ml.engine_io.assert_synthetic_runner_forbidden_for_real`
(Phase 15B.1), applied in the opposite direction (this scaffold FORBIDS any
adapter that is not tagged synthetic, rather than forbidding synthetic in a
real run).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from alpha_agent.crypto.provenance import DataProvenanceRole, RealVendorNotConnectedError
from alpha_agent.crypto.schemas import (
    CmeCryptoBasisBar,
    ExchangeFlowBar,
    LiquidationEvent,
    OnChainMetricBar,
    PerpFundingBar,
)


@runtime_checkable
class PerpFundingVendorAdapter(Protocol):
    """A vendor of perpetual funding / open-interest data."""

    provenance_role: DataProvenanceRole

    def fetch_funding(
        self, *, symbol: str, start_ts_ns: int, end_ts_ns: int
    ) -> list[PerpFundingBar]: ...


@runtime_checkable
class LiquidationVendorAdapter(Protocol):
    """A vendor of aggregated perpetual liquidation prints."""

    provenance_role: DataProvenanceRole

    def fetch_liquidations(
        self, *, symbol: str, start_ts_ns: int, end_ts_ns: int
    ) -> list[LiquidationEvent]: ...


@runtime_checkable
class ExchangeFlowVendorAdapter(Protocol):
    """A vendor of on-chain exchange inflow/outflow attribution."""

    provenance_role: DataProvenanceRole

    def fetch_flows(
        self, *, asset: str, start_ts_ns: int, end_ts_ns: int
    ) -> list[ExchangeFlowBar]: ...


@runtime_checkable
class OnChainVendorAdapter(Protocol):
    """A vendor of on-chain metrics (MVRV / SOPR / active addresses / ...)."""

    provenance_role: DataProvenanceRole

    def fetch_metrics(
        self, *, asset: str, start_ts_ns: int, end_ts_ns: int
    ) -> list[OnChainMetricBar]: ...


@runtime_checkable
class CmeCryptoBasisSource(Protocol):
    """A source of CME-listed crypto future vs. spot/index basis.

    ``futures_price_usd`` in the returned rows is expected to originate from
    the existing, unmodified Databento ``GLBX.MDP3`` adapter -- CME BTC/ETH
    futures already trade on that dataset (CLAUDE.md market-data rule 1); only
    the spot/index leg is a genuinely new input.
    """

    provenance_role: DataProvenanceRole

    def fetch_basis(
        self, *, root_symbol: str, start_ts_ns: int, end_ts_ns: int
    ) -> list[CmeCryptoBasisBar]: ...


def assert_vendor_is_synthetic_only(adapter: object) -> None:
    """The Phase 22 scaffold's one choke point: every adapter it drives MUST be
    tagged ``provenance_role == SYNTHETIC``. Raises
    :class:`~alpha_agent.crypto.provenance.RealVendorNotConnectedError`
    otherwise -- including when the attribute is simply missing, which fails
    closed rather than assuming synthetic."""
    role = getattr(adapter, "provenance_role", None)
    if role is not DataProvenanceRole.SYNTHETIC:
        raise RealVendorNotConnectedError(
            f"{type(adapter).__name__} is not a Phase 22 synthetic adapter "
            f"(provenance_role={role!r}); no real crypto vendor is connected -- "
            "see docs/CRYPTO_ONCHAIN_EXTENSION.md."
        )
