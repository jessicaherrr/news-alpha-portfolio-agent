"""Phase 3 (Agentic Alpha Evolution) -- Context-Aware Alpha Retrieval +
Transparent Ranking, FUTURES ONLY.

When a similar market/news context recurs, this package retrieves the
mechanisms, factors, strategy evidence, repeated failures, and research gaps
already known about it, and ranks them for RESEARCH PRIORITIZATION ONLY --
never as expected return, probability of success, or trade confidence (see
``schemas.RETRIEVAL_RANK_POLICY``).

This is a read model, exactly like :mod:`alpha_agent.alpha_memory`: nothing
here writes the ``ExperimentRegistry``, calls Claude, calls a market-data
API, or runs a backtest. It composes three already-frozen sources of truth
rather than inventing parallel ones:

* :mod:`alpha_agent.translation` (Phase 1) -- the deterministic mechanism/
  factor library and researchability classification.
* :mod:`alpha_agent.alpha_memory` (Phase 2) -- Personal Alpha Memory
  (``AlphaResearchObject``, ``mechanism_memory_lookup``).
* :mod:`alpha_agent.recommendation` (Phase B1) -- Research Promise / User
  Fit, reused verbatim rather than re-scored.

The one genuinely new concept this package adds is
:class:`schemas.MarketContextFingerprint` -- a transparent, deterministic
snapshot of "what does the current observation plane look like right now"
(event type/category/importance/freshness plus the SAME descriptive market
statistics :mod:`alpha_agent.ui.market_home`/``market_relative`` already
compute) -- and the Context Match dimension that measures how well a
candidate mechanism's own declared context signals (:mod:`mechanism_signals`)
align with it.
"""
from __future__ import annotations

from alpha_agent.context_retrieval.fingerprint import build_market_context_fingerprint
from alpha_agent.context_retrieval.retrieval import build_context_retrieval
from alpha_agent.context_retrieval.schemas import (
    CONTEXT_FINGERPRINT_SCHEMA,
    RETRIEVAL_RANK_POLICY,
    RETRIEVAL_RESULT_SCHEMA,
    UNKNOWN_OBJECTIVE_MAGNITUDE,
    ContextMatchLevel,
    ContextRetrievalResult,
    FreshnessBucket,
    IndependentReplicationLevel,
    MarketContextFingerprint,
    NextExperimentSuggestion,
    RankDimensionScores,
    RankedResearchCandidate,
    RelatedPriorEventOccurrence,
    ResearchCoverageLevel,
    ScientificReliabilityLevel,
)

__all__ = [
    "CONTEXT_FINGERPRINT_SCHEMA",
    "RETRIEVAL_RANK_POLICY",
    "RETRIEVAL_RESULT_SCHEMA",
    "UNKNOWN_OBJECTIVE_MAGNITUDE",
    "ContextMatchLevel",
    "ContextRetrievalResult",
    "FreshnessBucket",
    "IndependentReplicationLevel",
    "MarketContextFingerprint",
    "NextExperimentSuggestion",
    "RankDimensionScores",
    "RankedResearchCandidate",
    "RelatedPriorEventOccurrence",
    "ResearchCoverageLevel",
    "ScientificReliabilityLevel",
    "build_context_retrieval",
    "build_market_context_fingerprint",
]
