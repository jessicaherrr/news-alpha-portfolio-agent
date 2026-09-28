"""Phase 5 -- Factor metadata. ``FactorCandidate`` (Phase 1) and
``FactorIdentity`` (Phase 2) are both already asset-neutral in shape: neither
embeds a multiplier, tick size, or any other Futures-only field, only a
``mechanism`` and (for ``FactorIdentity``) the strategy-family set it
bundles. Re-exported verbatim -- see their own modules
(``alpha_agent.translation.schemas`` / ``alpha_agent.alpha_memory.schemas``)
for the identity/provenance discipline neither of these types' shape
changes.
"""
from __future__ import annotations

from alpha_agent.alpha_memory.schemas import FactorIdentity
from alpha_agent.translation.schemas import FactorCandidate

__all__ = ["FactorCandidate", "FactorIdentity"]
