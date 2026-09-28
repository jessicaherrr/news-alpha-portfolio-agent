"""Phase 5 (Agentic Alpha Evolution) -- the asset-neutral research spine.

Architecture migration ONLY (per this phase's own instructions): no ETF data,
no Futures-engine rewrite, no new registry write path, no change to any
frozen research identity. Every symbol re-exported from this package's
submodules is the SAME object already defined elsewhere in this repository
-- ``core`` adds an import surface a future asset generation (Phase 6 ETF
Pilot onward) can depend on without reaching into Futures-specific packages
by name; it does not fork, copy, or duplicate any existing type.

Genuinely shared (this package, generalized no earlier than Phase 5 -- see
``docs/AGENTIC_ALPHA_EVOLUTION_PLAN.md`` section 8.1):

* Instrument identity  -- :mod:`alpha_agent.core.instrument`   (NEW: wraps a root symbol)
* Event / Observation  -- :mod:`alpha_agent.core.observation`
* Market Context       -- :mod:`alpha_agent.core.market_context`
* Mechanism            -- :mod:`alpha_agent.core.mechanism`
* Factor metadata      -- :mod:`alpha_agent.core.factor`
* Hypothesis           -- :mod:`alpha_agent.core.hypothesis`
* Alpha Memory         -- :mod:`alpha_agent.core.alpha_memory`
* Validation evidence  -- :mod:`alpha_agent.core.validation_evidence`
* Learn linkage        -- :mod:`alpha_agent.core.learn_linkage`

Stays Futures-specific, deliberately NOT extracted here (evolution plan
section 8.1's "Asset-Specific Mechanics" column): contract chains, roll,
expiry, multiplier, tick size/value, futures margin, term structure/carry
*computation*, and Futures execution rules -- ``alpha_agent.data``,
``alpha_agent.features``, ``alpha_agent.strategy.candidates_phase_13_5c``,
and the whole ``cpp/`` engine are untouched by this phase.

Registry-storage decision (evolution plan section 7) -- BEGUN, not decided,
here: two options remain open (A: one database per asset generation, the
existing pattern; B: one shared store with a structurally enforced
namespace/discriminator). Phase 5 makes neither choice; it is deferred to
Phase 6 with a real second generation's requirements in hand. Nothing in
this package writes ``data/registry/experiments.sqlite`` or any other
registry table (see ``test_phase5_asset_neutral_core.py``'s static guards).
"""
from __future__ import annotations

from alpha_agent.core.alpha_memory import AlphaResearchObject, EvidenceProfile, ResearchMaturity
from alpha_agent.core.factor import FactorCandidate, FactorIdentity
from alpha_agent.core.hypothesis import HypothesisSpec
from alpha_agent.core.instrument import (
    ROOT_SYMBOL_PATTERN,
    AssetDomain,
    InstrumentIdentity,
    futures_instrument_identity,
)
from alpha_agent.core.learn_linkage import Concept, get_concept
from alpha_agent.core.market_context import MarketContextFingerprint
from alpha_agent.core.mechanism import EconomicMechanism
from alpha_agent.core.observation import MarketNewsItem, Observation, ScheduledMarketEvent
from alpha_agent.core.validation_evidence import (
    Authority,
    ExperimentEvidenceRef,
    RegistryVerdict,
    TrialRole,
)

__all__ = [
    "ROOT_SYMBOL_PATTERN",
    "AlphaResearchObject",
    "AssetDomain",
    "Authority",
    "Concept",
    "EconomicMechanism",
    "EvidenceProfile",
    "ExperimentEvidenceRef",
    "FactorCandidate",
    "FactorIdentity",
    "HypothesisSpec",
    "InstrumentIdentity",
    "MarketContextFingerprint",
    "MarketNewsItem",
    "Observation",
    "RegistryVerdict",
    "ResearchMaturity",
    "ScheduledMarketEvent",
    "TrialRole",
    "futures_instrument_identity",
    "get_concept",
]
