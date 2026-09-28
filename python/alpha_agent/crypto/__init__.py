"""Phase 22 -- optional crypto / on-chain extension (prompts/22).

Synthetic-scaffold-only (Phase 22 review-gate decision, 2026-09-11): zero
network calls, zero paid API use, zero real vendor connections. See
``docs/CRYPTO_ONCHAIN_EXTENSION.md``.
"""
from __future__ import annotations

from alpha_agent.crypto.provenance import (
    SYNTHETIC_BANNER,
    CryptoDataProvenance,
    DataProvenanceRole,
    RealVendorNotConnectedError,
)

__all__ = [
    "SYNTHETIC_BANNER",
    "CryptoDataProvenance",
    "DataProvenanceRole",
    "RealVendorNotConnectedError",
]
