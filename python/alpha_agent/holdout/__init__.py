"""Alpha Discovery campaign, Part L -- the 2025 holdout lifecycle. Pure
bookkeeping, no market-data-loading code at all -- see
`alpha_agent.holdout.lifecycle` for the full boundary docstring.
"""
from __future__ import annotations

from alpha_agent.holdout.lifecycle import (
    AUTHORIZATION_ENV_VAR,
    DEFAULT_LIFECYCLE_DIR,
    FinalHoldoutManifest,
    HoldoutAuthorizationError,
    HoldoutEpochRecord,
    HoldoutLifecycle,
    HoldoutLifecycleError,
    HoldoutState,
)

__all__ = [
    "AUTHORIZATION_ENV_VAR",
    "DEFAULT_LIFECYCLE_DIR",
    "FinalHoldoutManifest",
    "HoldoutAuthorizationError",
    "HoldoutEpochRecord",
    "HoldoutLifecycle",
    "HoldoutLifecycleError",
    "HoldoutState",
]
