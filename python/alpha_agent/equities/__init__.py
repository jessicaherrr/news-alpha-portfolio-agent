"""Phase 9.1 -- Equities Foundation.

Mirrors :mod:`alpha_agent.etf` structurally (own universe, own corporate-
action store, own instrument-economics adapter, own calendar) so Equity
evidence stays namespaced under ``AssetDomain.EQUITY`` and never merges with
FUTURES or ETF evidence merely because a mechanism label or a strategy family
key happens to match (the same principle Phase 6/7/8 already established).

This package deliberately stops at the DATA-INTEGRITY FOUNDATION layer:
point-in-time universe membership, survivorship/delisting handling, corporate
actions, and earnings-timestamp provenance. It contains no strategy family,
no factor research, and no registry-append path -- those are Phase 9.2/9.3
scope. No historical Equity market data has been acquired or downloaded by
this package; every data-source loader here fails loudly (``FileNotFoundError``)
until a real, user-approved, cost-capped acquisition happens.
"""
from __future__ import annotations
