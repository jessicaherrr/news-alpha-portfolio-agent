"""Typed, validated parameter models for the Phase 11 baseline factories.

Only economically-meaningful parameters are exposed. Invalid combinations are
rejected at construction (``fast >= slow``, ``entry_z <= exit_z``, non-positive
windows, a zero position size). There is deliberately **no optimizer, grid
search or walk-forward selection here** -- the documented parameter grid ranges
live in :mod:`alpha_agent.strategy.baselines.families` for later phases.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

# A baseline never sizes by leverage / margin / risk budget. It emits a small
# integer target position; Phase 08 Hard Risk is authoritative for what actually
# trades. This ceiling also stays within the Phase 10 compiler's default
# ``max_abs_target_units`` (10).
MAX_BASELINE_SIZE = 5


class _BaseParams(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str = Field(pattern=r"^[A-Z0-9]{1,12}$")
    # A small, explicitly-configured integer target magnitude. Longs target
    # ``+size``, shorts ``-size``. Not a leverage or notional figure.
    size: int = Field(default=1, ge=1, le=MAX_BASELINE_SIZE)
    price_field: str = "close"


class TsmomParams(_BaseParams):
    """Multi-horizon time-series momentum / trend.

    Uses arithmetic price *differences* over two horizons (signed-price safe --
    valid for CL through zero, section 20). Long only when both horizons agree
    up, short only when both agree down, otherwise flat.
    """

    fast_horizon: int = Field(ge=1, le=100_000)
    slow_horizon: int = Field(ge=1, le=100_000)

    @model_validator(mode="after")
    def _check(self) -> TsmomParams:
        if self.fast_horizon >= self.slow_horizon:
            raise ValueError(
                f"fast_horizon ({self.fast_horizon}) must be strictly less than "
                f"slow_horizon ({self.slow_horizon})"
            )
        return self


class MaTrendParams(_BaseParams):
    """Fast vs slow simple moving-average trend."""

    fast_window: int = Field(ge=1, le=100_000)
    slow_window: int = Field(ge=2, le=100_000)

    @model_validator(mode="after")
    def _check(self) -> MaTrendParams:
        if self.fast_window >= self.slow_window:
            raise ValueError(
                f"fast_window ({self.fast_window}) must be strictly less than "
                f"slow_window ({self.slow_window})"
            )
        return self


class BreakoutParams(_BaseParams):
    """Donchian / channel breakout on the PRIOR window's rolling high / low.

    The breakout reference (``breakout_up`` / ``breakout_down``) is
    ``close - shift(rolling_high_or_low, 1)`` -- it never includes the current
    bar or any future bar (section C). Position is held until the opposite
    breakout (``default_action = keep_previous_target``).
    """

    lookback: int = Field(ge=2, le=100_000)


class MeanReversionParams(_BaseParams):
    """Short-horizon z-score mean reversion with explicit entry / exit bands.

    Ordered DSL rules give unambiguous entry / exit semantics:
      ``z < -entry_z`` -> long ; ``z > +entry_z`` -> short ;
      ``|z| < exit_z`` -> flat ; otherwise keep the previous target.
    Uses a price-level z-score (``(price - rolling mean) / rolling std``), which
    is signed-price safe (no log, no return across zero -- section 20).
    """

    zscore_window: int = Field(ge=2, le=100_000)
    entry_z: float = Field(gt=0.0, le=20.0)
    exit_z: float = Field(ge=0.0, le=20.0)

    @model_validator(mode="after")
    def _check(self) -> MeanReversionParams:
        if self.entry_z <= self.exit_z:
            raise ValueError(
                f"entry_z ({self.entry_z}) must be strictly greater than exit_z "
                f"({self.exit_z}); the entry band must sit outside the exit band"
            )
        return self
