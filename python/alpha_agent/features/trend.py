"""Trend / time-series-momentum features (section 6). Feature only -- no strategy.

Prefers signed-price-safe arithmetic (moving averages, price differences,
breakout distances). Volatility scaling uses the standard deviation of 1-bar
*price differences*, which is defined across a sign change, rather than a
percentage return.
"""
from __future__ import annotations

import numpy as np

from alpha_agent.features.enums import FeatureFamily, SessionPolicy
from alpha_agent.features.registry import FeatureComputeContext, ParamRule, feature, require_lt
from alpha_agent.features.windows import (
    safe_ratio,
    seg_diff,
    seg_ewm_mean,
    seg_rolling,
    seg_shift,
)

_W = {"window": ParamRule(int, min=1, max=100_000)}
_FASTSLOW = {
    "fast": ParamRule(int, min=1, max=100_000),
    "slow": ParamRule(int, min=2, max=100_000),
}


@feature("ma", family=FeatureFamily.TREND, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"], description="simple backward moving average")
def _ma(ctx: FeatureComputeContext):
    return seg_rolling(ctx.price(), ctx.segment_id, ctx.params["window"], ctx.min_obs, "mean")


@feature("ema", family=FeatureFamily.TREND, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"], description="exponential moving average (adjust=False)")
def _ema(ctx: FeatureComputeContext):
    return seg_ewm_mean(ctx.price(), ctx.segment_id, ctx.params["window"], ctx.min_obs)


@feature("ma_spread", family=FeatureFamily.TREND, param_rules=_FASTSLOW,
         param_order=("fast", "slow"), lookback=lambda p: p["slow"],
         param_constraints=(require_lt("fast", "slow"),),
         description="fast MA minus slow MA (absolute price units; signed-safe)")
def _ma_spread(ctx: FeatureComputeContext):
    p = ctx.price()
    fast = seg_rolling(p, ctx.segment_id, ctx.params["fast"], ctx.params["fast"], "mean")
    slow = seg_rolling(p, ctx.segment_id, ctx.params["slow"], ctx.params["slow"], "mean")
    return fast - slow


@feature("trend_strength", family=FeatureFamily.TREND, param_rules=_FASTSLOW,
         param_order=("fast", "slow"), lookback=lambda p: p["slow"] + 1,
         param_constraints=(require_lt("fast", "slow"),),
         description="normalized trend: (fast MA - slow MA) / rolling std of 1-bar price diff")
def _trend_strength(ctx: FeatureComputeContext):
    p = ctx.price()
    fast = seg_rolling(p, ctx.segment_id, ctx.params["fast"], ctx.params["fast"], "mean")
    slow = seg_rolling(p, ctx.segment_id, ctx.params["slow"], ctx.params["slow"], "mean")
    d1 = seg_diff(p, ctx.segment_id, 1)
    sd = seg_rolling(d1, ctx.segment_id, ctx.params["slow"], max(2, ctx.params["slow"] - 1), "std")
    val, _ = safe_ratio(fast - slow, sd)
    return val


@feature("rolling_high", family=FeatureFamily.TREND, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"], description="backward rolling max of the high")
def _rolling_high(ctx: FeatureComputeContext):
    field = "high" if "high" in ctx.frame.columns else ctx.price_field
    return seg_rolling(ctx.price(field), ctx.segment_id, ctx.params["window"],
                       ctx.params["window"], "max")


@feature("rolling_low", family=FeatureFamily.TREND, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"], description="backward rolling min of the low")
def _rolling_low(ctx: FeatureComputeContext):
    field = "low" if "low" in ctx.frame.columns else ctx.price_field
    return seg_rolling(ctx.price(field), ctx.segment_id, ctx.params["window"],
                       ctx.params["window"], "min")


@feature("breakout_up", family=FeatureFamily.TREND, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"] + 1,
         description="close minus the PRIOR window's rolling high (>0 => new-high breakout)")
def _breakout_up(ctx: FeatureComputeContext):
    high = ctx.price("high" if "high" in ctx.frame.columns else ctx.price_field)
    prior_high = seg_shift(
        seg_rolling(high, ctx.segment_id, ctx.params["window"], ctx.params["window"], "max"),
        ctx.segment_id, 1,
    )
    return ctx.price("close") - prior_high


@feature("breakout_down", family=FeatureFamily.TREND, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"] + 1,
         description="close minus the PRIOR window's rolling low (<0 => new-low breakout)")
def _breakout_down(ctx: FeatureComputeContext):
    low = ctx.price("low" if "low" in ctx.frame.columns else ctx.price_field)
    prior_low = seg_shift(
        seg_rolling(low, ctx.segment_id, ctx.params["window"], ctx.params["window"], "min"),
        ctx.segment_id, 1,
    )
    return ctx.price("close") - prior_low


@feature("donchian_pos", family=FeatureFamily.TREND, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"],
         description="position of close within the rolling [low, high] channel, 0..1")
def _donchian_pos(ctx: FeatureComputeContext):
    high = ctx.price("high" if "high" in ctx.frame.columns else ctx.price_field)
    low = ctx.price("low" if "low" in ctx.frame.columns else ctx.price_field)
    hi = seg_rolling(high, ctx.segment_id, ctx.params["window"], ctx.params["window"], "max")
    lo = seg_rolling(low, ctx.segment_id, ctx.params["window"], ctx.params["window"], "min")
    val, _ = safe_ratio(ctx.price("close") - lo, hi - lo)
    return val


@feature("vol_scaled_trend", family=FeatureFamily.TREND,
         param_rules={"n": ParamRule(int, min=1, max=100_000),
                      "vol_window": ParamRule(int, min=2, max=100_000)},
         param_order=("n", "vol_window"),
         default_session_policy=SessionPolicy.RESET_ON_GAP,
         lookback=lambda p: p["n"] + p["vol_window"],
         description="n-bar price change scaled by rolling std of 1-bar diff * sqrt(n)")
def _vol_scaled_trend(ctx: FeatureComputeContext):
    p = ctx.price()
    dn = seg_diff(p, ctx.segment_id, ctx.params["n"])
    d1 = seg_diff(p, ctx.segment_id, 1)
    sd = seg_rolling(d1, ctx.segment_id, ctx.params["vol_window"],
                     max(2, ctx.params["vol_window"] - 1), "std")
    val, _ = safe_ratio(dn, sd * np.sqrt(ctx.params["n"]))
    return val
