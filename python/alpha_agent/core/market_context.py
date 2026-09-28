"""Phase 5 -- Market Context. ``MarketContextFingerprint``
(``alpha_agent.context_retrieval.schemas``) keeps two Futures-specific fields
inline (``curve_state``: term-structure shape; ``related_market_confirming``/
``related_market_total``: cross-market confirmation) alongside otherwise
asset-neutral fields (trend, volatility, event importance, freshness). Phase
5 does not split the schema -- that would touch a FROZEN
``CONTEXT_FINGERPRINT_SCHEMA`` fingerprint plane for no behavioural gain
before a second domain exists to prove which fields it would actually share.
Re-exported verbatim; a future generation either reuses this shape whole or
defines its own -- Phase 5 does not decide that here.
"""
from __future__ import annotations

from alpha_agent.context_retrieval.schemas import (
    MarketContextFingerprint,
)

__all__ = ["MarketContextFingerprint"]
