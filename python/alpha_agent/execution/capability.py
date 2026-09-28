"""Alpha Discovery campaign, Part E -- capability-based execution gating
(task spec sections 30-35).

Replaces the family-NAME allowlist `alpha_agent.agents.execution_service`
used to have (`_SUPPORTED_FAMILIES: dict[family_key, cadence]`) with a
capability question: is `signal_cadence` one this release's C++ execution
path actually knows how to schedule, are every one of the strategy's
required feature kinds registered, and (dynamically, once a schedule build
is actually attempted) does the feature/schedule pipeline accept it?

This is INTENTIONALLY NOT a bigger change than that. `signal_cadence` for a
full BLUEPRINT compile is still the placeholder string
`alpha_agent.agents.compiler_agent._execution_semantics` sets when
`build_mode != "template"` ("set by the orchestrator backtest configuration
...") -- not a real cadence -- so a blueprint honestly still reports
`UNSUPPORTED_CADENCE` today. Making a blueprint's OWN declared cadence a real,
checkable pre-run quantity is future work (would need a compiler-schema
change), not something this module fakes. What this module DOES achieve: the
five frozen legacy families are now accepted because their cadence is
supported, not because their NAME is on a list -- so any FUTURE family whose
compiler build correctly declares one of `SUPPORTED_CADENCES` executes
through the exact same generic path with zero further gating changes needed.

Every reason code below is one of task spec section 31's typed list. These
are TECHNICAL RESEARCH CONSTRAINTS -- never a scientific REJECT (task spec
section 31's own words), and this module never assigns REJECT/PASS/
INCONCLUSIVE.
"""
from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel

from alpha_agent.strategy.candidates_phase_13_5c import SIGNAL_CADENCE

#: The `signal_cadence` values this release's execution path can actually
#: schedule -- derived from the SAME frozen `SIGNAL_CADENCE` mapping the
#: Phase 13.5C matrix and the compiler agent's TEMPLATE mode already use, so
#: the five legacy families are supported for the identical reason they
#: always were (their real signal cadence), not a re-derived coincidence.
SUPPORTED_CADENCES: frozenset[str] = frozenset(SIGNAL_CADENCE.values())


class CapabilityReason(str, Enum):
    """Task spec section 31's exact typed taxonomy."""

    UNSUPPORTED_FEATURE = "UNSUPPORTED_FEATURE"
    NON_CAUSAL_FEATURE = "NON_CAUSAL_FEATURE"
    UNSUPPORTED_CADENCE = "UNSUPPORTED_CADENCE"
    MISSING_DATA_LAYER = "MISSING_DATA_LAYER"
    UNSUPPORTED_CROSS_MARKET_INPUT = "UNSUPPORTED_CROSS_MARKET_INPUT"
    UNSUPPORTED_STATEFUL_EXIT = "UNSUPPORTED_STATEFUL_EXIT"
    INVALID_STRATEGY_SPEC = "INVALID_STRATEGY_SPEC"
    DATA_ALIGNMENT_UNSUPPORTED = "DATA_ALIGNMENT_UNSUPPORTED"
    EXECUTION_CAPABILITY_MISSING = "EXECUTION_CAPABILITY_MISSING"


class CapabilityAssessment(BaseModel):
    """Never a scientific verdict (task spec section 31) -- `supported=False`
    means "cannot be executed by this release today", full stop."""

    model_config = {"frozen": True, "extra": "forbid"}

    supported: bool
    reason: CapabilityReason | None = None
    detail: str = ""


def resolve_cadence(*, signal_cadence: str, strategy_family: str) -> str:
    """The cadence a member will actually be scheduled with. Prefers the
    member's OWN declared `signal_cadence` (the genuinely capability-based
    path -- works for any future family whose compiler build correctly
    declares a real cadence); falls back to the frozen legacy family-name
    mapping only when `signal_cadence` was left empty/unrecognized (an
    older/hand-built caller that never populated it). Shared by
    `alpha_agent.agents.execution_service` and
    `alpha_agent.screening.fast_screen` so the two execution paths can never
    silently disagree about what cadence a member runs under."""
    if signal_cadence in SUPPORTED_CADENCES:
        return signal_cadence
    return SIGNAL_CADENCE.get(strategy_family, signal_cadence)


def assess_cadence(signal_cadence: str) -> CapabilityReason | None:
    if signal_cadence not in SUPPORTED_CADENCES:
        return CapabilityReason.UNSUPPORTED_CADENCE
    return None


def assess_required_features(
    required_features: Iterable[str], *, feature_registry: object | None = None
) -> CapabilityReason | None:
    """Every required feature `kind` must be a REGISTERED kind (task spec
    section 30: "Are its features registered?"). Causal safety
    (`NON_CAUSAL_FEATURE`) is a DYNAMIC property of a real source series --
    see `classify_schedule_exception` below, which classifies the outcome of
    an actual schedule-build attempt rather than guessing statically."""
    if feature_registry is None:
        import alpha_agent.features.compute  # noqa: F401 - register defs
        from alpha_agent.features import REGISTRY as feature_registry

    known = set(feature_registry.kinds())  # type: ignore[union-attr]
    missing = [f for f in required_features if f not in known]
    if missing:
        return CapabilityReason.UNSUPPORTED_FEATURE
    return None


def assess_static_capability(
    *, signal_cadence: str, required_features: Iterable[str], feature_registry: object | None = None
) -> CapabilityAssessment:
    """The cheap, PRE-RUN capability gate -- no market data touched, no
    schedule built. Checks cadence first (cheapest, most common today's
    blueprint-mode gap) then feature registration."""
    reason = assess_cadence(signal_cadence)
    if reason is not None:
        return CapabilityAssessment(
            supported=False, reason=reason,
            detail=f"signal_cadence {signal_cadence!r} is not one of {sorted(SUPPORTED_CADENCES)}",
        )
    reason = assess_required_features(required_features, feature_registry=feature_registry)
    if reason is not None:
        return CapabilityAssessment(
            supported=False, reason=reason,
            detail=f"one or more required features are not registered: {list(required_features)}",
        )
    return CapabilityAssessment(supported=True)


def classify_schedule_exception(exc: Exception) -> CapabilityReason:
    """Map an exception raised while actually attempting to build a target
    schedule (feature computation + `build_target_schedule`) to a typed
    capability reason. Reuses the EXISTING frozen exception types from
    `alpha_agent.features` / `alpha_agent.strategy` -- never re-derives
    causal-safety logic; the frozen `compute_features(...,
    require_point_in_time=True)` path IS the causal-safety authority."""
    from alpha_agent.features import (
        FeatureNameCollision,
        FeaturePriceDomainError,
        LookaheadUnsafeError,
    )
    from alpha_agent.features.safety import FeatureSafetyError

    if isinstance(exc, (LookaheadUnsafeError, FeatureSafetyError)):
        return CapabilityReason.NON_CAUSAL_FEATURE
    if isinstance(exc, (FeaturePriceDomainError, FeatureNameCollision)):
        return CapabilityReason.DATA_ALIGNMENT_UNSUPPORTED
    if isinstance(exc, KeyError):
        return CapabilityReason.MISSING_DATA_LAYER
    return CapabilityReason.EXECUTION_CAPABILITY_MISSING
