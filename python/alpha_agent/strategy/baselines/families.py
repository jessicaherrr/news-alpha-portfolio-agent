"""Documentation metadata for the Phase 11 baseline families.

Economic mechanism, exact formula/timing, required data, the parameter grid
ranges declared **before** looking at any results (prompt 11), and the known
failure regimes -- as typed data, so Phase 13 reliability validation and the
experiment registry can consume it without re-parsing prose. This module builds
**no** optimizer and does no search.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class BaselineFamilyDoc(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    key: str
    name: str
    factory: str
    economic_mechanism: str
    formula_and_timing: str
    required_data: list[str]
    parameters: list[str]
    # Declared BEFORE results are inspected. Ranges only -- not a grid to search.
    param_grid_ranges: dict[str, str]
    failure_regimes: list[str]
    default_action: str
    is_claimed_alpha: bool = Field(default=False)


BASELINE_FAMILIES: tuple[BaselineFamilyDoc, ...] = (
    BaselineFamilyDoc(
        key="tsmom",
        name="Time-series momentum / trend (multi-horizon)",
        factory="make_tsmom_spec",
        economic_mechanism=(
            "Futures prices exhibit persistent directional drift (the trend / momentum "
            "premium documented across asset classes). Requiring two horizons to agree "
            "filters noise-driven single-horizon flips."
        ),
        formula_and_timing=(
            "d_fast = price[T] - price[T-fast_horizon]; d_slow = price[T] - price[T-slow_horizon]. "
            "Both > 0 -> +size; both < 0 -> -size; else flat. Decided on bar T information; "
            "executed on the engine's next eligible bar."
        ),
        required_data=["one continuous or raw OHLC price series per root"],
        parameters=["fast_horizon", "slow_horizon", "size"],
        param_grid_ranges={
            "fast_horizon": "5 .. 60 bars",
            "slow_horizon": "20 .. 250 bars (must exceed fast_horizon)",
            "size": "1 .. 3",
        },
        failure_regimes=[
            "choppy / mean-reverting regimes (whipsaw, repeated small losses)",
            "sharp reversals after extended trends (gives back open profit)",
            "low-volatility drift where costs dominate the small edge",
        ],
        default_action="flat",
    ),
    BaselineFamilyDoc(
        key="ma_trend",
        name="Moving-average trend (fast vs slow)",
        factory="make_ma_trend_spec",
        economic_mechanism=(
            "Same trend premium, expressed as a fast/slow simple-moving-average "
            "crossover -- a smoother, more persistent trend proxy than a raw price change."
        ),
        formula_and_timing=(
            "fast_ma > slow_ma -> +size; fast_ma < slow_ma -> -size; else flat. "
            "Both MAs are backward-looking; decision on bar T, execution next eligible bar."
        ),
        required_data=["one continuous or raw OHLC price series per root"],
        parameters=["fast_window", "slow_window", "size"],
        param_grid_ranges={
            "fast_window": "5 .. 50 bars",
            "slow_window": "20 .. 200 bars (must exceed fast_window)",
            "size": "1 .. 3",
        },
        failure_regimes=[
            "range-bound markets (crossover whipsaw)",
            "lag at turning points (late entries and exits)",
            "parameter sensitivity near the fast/slow ratio",
        ],
        default_action="flat",
    ),
    BaselineFamilyDoc(
        key="breakout",
        name="Donchian / channel breakout",
        factory="make_breakout_spec",
        economic_mechanism=(
            "A move beyond a recent trading range often marks the start of a new trend "
            "as constrained participants are forced to adjust."
        ),
        formula_and_timing=(
            "bo_up = close[T] - max(high[T-lookback .. T-1]); bo_down = close[T] - "
            "min(low[T-lookback .. T-1]). bo_up > 0 -> +size; bo_down < 0 -> -size; "
            "else keep the previous target. The channel is the PRIOR window only -- no "
            "look-ahead into bar T's own extreme."
        ),
        required_data=["one continuous or raw OHLC price series per root (needs high/low)"],
        parameters=["lookback", "size"],
        param_grid_ranges={
            "lookback": "10 .. 100 bars",
            "size": "1 .. 3",
        },
        failure_regimes=[
            "false breakouts in choppy ranges",
            "gap openings beyond the channel (adverse fill vs the signal level)",
            "sustained low volatility (few breakouts, stale positions held via keep-previous)",
        ],
        default_action="keep_previous_target",
    ),
    BaselineFamilyDoc(
        key="mean_reversion",
        name="Short-horizon z-score mean reversion",
        factory="make_mean_reversion_spec",
        economic_mechanism=(
            "Short-horizon price overshoots relative to a trailing mean partially revert "
            "as liquidity providers are compensated for absorbing flow."
        ),
        formula_and_timing=(
            "z = (price[T] - mean_w(price)) / std_w(price). Ordered rules: z < -entry_z -> "
            "+size; z > +entry_z -> -size; |z| < exit_z -> 0; else keep previous target. "
            "entry_z > exit_z enforced. Decision on bar T, execution next eligible bar."
        ),
        required_data=["one continuous or raw OHLC price series per root"],
        parameters=["zscore_window", "entry_z", "exit_z", "size"],
        param_grid_ranges={
            "zscore_window": "10 .. 120 bars",
            "entry_z": "1.5 .. 3.0",
            "exit_z": "0.0 .. 1.0 (must be < entry_z)",
            "size": "1 .. 3",
        },
        failure_regimes=[
            "strong trends (fading a persistent move -> compounding losses)",
            "volatility regime shifts (the trailing std lags, z-scores mis-scaled)",
            "structural breaks / news gaps (mean is no longer the right anchor)",
        ],
        default_action="keep_previous_target",
    ),
)


def baseline_family_docs() -> tuple[BaselineFamilyDoc, ...]:
    return BASELINE_FAMILIES
