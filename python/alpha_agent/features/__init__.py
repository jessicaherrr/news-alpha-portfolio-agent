"""Phase 09 -- deterministic, point-in-time futures feature engine.

    FeatureSpec  ->  FeatureRegistry  ->  FeatureComputer  ->  FeatureFrame

The engine transforms approved historical market data (canonical raw-contract
bars, unadjusted continuous series, or -- point-in-time or retrospective --
back-adjusted series) into point-in-time features. It does not generate Orders,
compute official PnL, or execute strategies.

Three explicit safety concepts (:mod:`alpha_agent.features.safety`):
``point_in_time_safe`` / ``signal_safe`` / ``execution_price_safe``. A
point-in-time back-adjusted trend feature is signal-safe (Phase 10 DSL may use
it) but never execution-price safe (a feature value is never a Fill price).

Importing this package registers every built-in feature ``kind`` (returns,
trend, mean-reversion, volatility, volume, futures-roll). Carry / term-structure
and COT / macro alignment are separate typed interfaces (``features.carry`` /
``features.macro``); cross-market support is ``features.crossmarket``.
"""
from __future__ import annotations

# Registering side-effect imports -- keep after the core imports above.
from alpha_agent.features import (  # noqa: F401
    market_structure,
    returns,
    reversion,
    roll_features,
    trend,
    volatility,
    volume,
)
from alpha_agent.features.compute import (
    FeatureComputer,
    FeatureNameCollision,
    FeaturePriceDomainError,
    LookaheadUnsafeError,
    compute_features,
)
from alpha_agent.features.enums import FeatureFamily, RollFeatureMode, SessionPolicy
from alpha_agent.features.frame import FeatureFrame, FeatureLineage, feature_cache_key

# Legacy Phase-01 helper, kept for the smoke script.
from alpha_agent.features.price import add_price_features
from alpha_agent.features.qa import FeatureIssueKind, FeatureQAReport
from alpha_agent.features.registry import (
    FEATURE_ENGINE_VERSION,
    REGISTRY,
    FeatureRegistry,
    feature,
    require_lt,
)
from alpha_agent.features.safety import (
    FeatureSafety,
    FeatureSafetyError,
    SourceSafety,
)
from alpha_agent.features.source import SourceSeries
from alpha_agent.features.spec import FeatureMetadata, FeatureSpec

__all__ = [
    "FEATURE_ENGINE_VERSION",
    "REGISTRY",
    "FeatureComputer",
    "FeatureFamily",
    "FeatureFrame",
    "FeatureIssueKind",
    "FeatureLineage",
    "FeatureMetadata",
    "FeatureNameCollision",
    "FeaturePriceDomainError",
    "FeatureQAReport",
    "FeatureRegistry",
    "FeatureSafety",
    "FeatureSafetyError",
    "FeatureSpec",
    "LookaheadUnsafeError",
    "RollFeatureMode",
    "SessionPolicy",
    "SourceSafety",
    "SourceSeries",
    "add_price_features",
    "compute_features",
    "feature",
    "feature_cache_key",
    "require_lt",
]
