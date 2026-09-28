"""Phase B1 -- user profile, Research Promise, User Fit, and candidate
ranking. Presentation / research-prioritization only: nothing in this package
computes PnL, fills, risk, or a scientific verdict, writes the
`ExperimentRegistry`, or influences `experiment_identity` / `ReliabilityPolicy`
/ BH-FDR / DSR. See `alpha_agent.recommendation.profile`/`.promise`/`.fit`/
`.candidates` for the module-level boundary docstrings.

News Alpha Phase F extends this layer to a second ranked object: candidate
signals + their screening diagnostics, ranked for portfolio consideration
(`alpha_agent.recommendation.signal_ranking`). It reuses the same merit /
fit split and is imported by module path, not re-exported here, because it
reads the News Alpha hypothesis plane and the screening plane.
"""
from __future__ import annotations

from alpha_agent.recommendation.candidates import (
    ResearchCandidateSummary,
    build_candidate_summaries,
    build_candidate_summary,
    promising_candidates,
    validated_strategies,
)
from alpha_agent.recommendation.fit import PersonalizationState, UserFitBreakdown, score_user_fit
from alpha_agent.recommendation.profile import (
    DEFAULT_PROFILE,
    HoldingPeriod,
    InvestorProfile,
    MaxDrawdown,
    OvernightPreference,
    ProfileStore,
    RiskStyle,
    StrategyPreference,
    TradingFrequency,
    TurnoverSensitivity,
)
from alpha_agent.recommendation.promise import ResearchPromiseBreakdown, score_research_promise

__all__ = [
    "DEFAULT_PROFILE",
    "HoldingPeriod",
    "InvestorProfile",
    "MaxDrawdown",
    "OvernightPreference",
    "PersonalizationState",
    "ProfileStore",
    "ResearchCandidateSummary",
    "ResearchPromiseBreakdown",
    "RiskStyle",
    "StrategyPreference",
    "TradingFrequency",
    "TurnoverSensitivity",
    "UserFitBreakdown",
    "build_candidate_summaries",
    "build_candidate_summary",
    "promising_candidates",
    "score_research_promise",
    "score_user_fit",
    "validated_strategies",
]
