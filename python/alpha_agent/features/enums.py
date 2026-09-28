"""Enums shared across the Phase 09 feature engine."""
from __future__ import annotations

from enum import Enum


class SessionPolicy(str, Enum):
    """How rolling/return windows behave around session boundaries and data gaps.

    ``CONTINUOUS``            -- one unbroken series; a window may span a gap
                                (QA still records the gap / staleness).
    ``RESET_ON_SESSION``      -- a new ``trading_day`` (or ``session`` label change)
                                starts a fresh window; the first ``min_observations``
                                rows of the new segment are explicitly missing.
    ``RESET_ON_GAP``          -- a missing-bar gap larger than
                                ``max_gap_intervals`` * ``interval_ns`` starts a
                                fresh segment.
    ``RESET_ON_SESSION_AND_GAP`` -- both of the above.
    """

    CONTINUOUS = "continuous"
    RESET_ON_SESSION = "reset_on_session"
    RESET_ON_GAP = "reset_on_gap"
    RESET_ON_SESSION_AND_GAP = "reset_on_session_and_gap"


class RollFeatureMode(str, Enum):
    """Point-in-time-legitimate roll features vs retrospective research ones.

    ``POINT_IN_TIME``   -- only information observable at or before ``T``
                           (a roll that already happened).
    ``RETROSPECTIVE``   -- may look forward across the whole sample
                           (e.g. bars-until-next-roll); offline research only.
    """

    POINT_IN_TIME = "point_in_time"
    RETROSPECTIVE = "retrospective"


class FeatureFamily(str, Enum):
    RETURN = "return"
    TREND = "trend"
    REVERSION = "reversion"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    ROLL = "roll"
    CARRY = "carry"
    MACRO = "macro"
    CROSS_MARKET = "cross_market"
    MARKET_STRUCTURE = "market_structure"
