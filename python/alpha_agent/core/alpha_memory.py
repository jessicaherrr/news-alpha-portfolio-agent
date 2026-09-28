"""Phase 5 -- Alpha Memory. ``AlphaResearchObject`` / ``EvidenceProfile`` /
``ResearchMaturity`` (``alpha_agent.alpha_memory.schemas``) are already a
generic read-model shape (root symbol + Factor identity + evidence, no
multiplier/tick/roll field anywhere). Re-exported verbatim, unchanged.

Every ``AlphaResearchObject`` materialized by today's ``alpha_agent.alpha_memory``
package is implicitly Futures-domain evidence -- the only domain
``alpha_agent.core.instrument.AssetDomain`` supports. This module
deliberately does NOT add a stored ``asset_domain`` field: doing so would
either require backfilling every historical registry-derived object (Phase 5
instruction 5 forbids rewriting old rows) or default silently, which is worse
-- an explicit, honest gap. Once a second domain exists, a caller comparing
evidence across domains must add its own real, typed domain check; nothing
here should ever be read as implying today's Futures evidence transfers to
another domain (Phase 5 instruction 6).
"""
from __future__ import annotations

from alpha_agent.alpha_memory.schemas import AlphaResearchObject, EvidenceProfile, ResearchMaturity

__all__ = ["AlphaResearchObject", "EvidenceProfile", "ResearchMaturity"]
