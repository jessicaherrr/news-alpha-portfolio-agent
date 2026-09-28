"""Root-symbol -> IBKR contract routing for the delayed market-context feed.

``CONTFUT`` is IBKR's own continuous-futures security type: a market-data-only
proxy contract IBKR resolves to whichever expiry it currently treats as
front-month. It carries no `conId` for order routing and this package never
places an order, so that limitation is irrelevant here -- it exists purely so
`reqMktData` has a stable, expiry-independent symbol to request quotes for.
This is a display/context convenience, not a statement about which contract
the frozen research corpus's `RegistryActiveContractResolver` selected for a
given historical date; those two contract-selection mechanisms are
unrelated.
"""
from __future__ import annotations

from dataclasses import dataclass

APPROVED_ROOTS: tuple[str, ...] = ("ES", "NQ", "CL", "GC", "ZN")


@dataclass(frozen=True)
class IBKRContractSpec:
    root_symbol: str
    ibkr_symbol: str
    exchange: str
    currency: str = "USD"
    sec_type: str = "CONTFUT"


_CONTRACTS: dict[str, IBKRContractSpec] = {
    "ES": IBKRContractSpec("ES", "ES", "CME"),
    "NQ": IBKRContractSpec("NQ", "NQ", "CME"),
    "CL": IBKRContractSpec("CL", "CL", "NYMEX"),
    "GC": IBKRContractSpec("GC", "GC", "COMEX"),
    "ZN": IBKRContractSpec("ZN", "ZN", "CBOT"),
}


def contract_for(root_symbol: str) -> IBKRContractSpec | None:
    return _CONTRACTS.get(root_symbol.upper())
