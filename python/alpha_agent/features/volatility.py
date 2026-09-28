"""Volatility features (section 8). The regime model is NOT built here -- these
are the point-in-time inputs a later regime model would consume.

``vol_percentile`` is a trailing rank (fraction of the trailing window <= the
current value): it uses only past observations and is point-in-time safe.
"""
from __future__ import annotations

import numpy as np

from alpha_agent.features.enums import FeatureFamily, SessionPolicy
from alpha_agent.features.qa import FeatureIssueKind
from alpha_agent.features.registry import FeatureComputeContext, ParamRule, feature
from alpha_agent.features.windows import (
    rolling_rank_pct,
    seg_diff,
    seg_log_return,
    seg_pct_change,
    seg_rolling,
    seg_rolling_apply,
    true_range,
)

_W = {"window": ParamRule(int, min=2, max=100_000)}


def _simple_returns(ctx: FeatureComputeContext):
    r1, base = seg_pct_change(ctx.price(), ctx.segment_id, 1)
    return r1.mask(base.notna() & (base <= 0.0))


@feature("volatility", family=FeatureFamily.VOLATILITY, param_rules=_W, param_order=("window",),
         default_session_policy=SessionPolicy.RESET_ON_GAP, lookback=lambda p: p["window"],
         description="rolling std (ddof=1) of 1-bar simple returns")
def _volatility(ctx: FeatureComputeContext):
    r1 = _simple_returns(ctx)
    return seg_rolling(r1, ctx.segment_id, ctx.params["window"], max(2, ctx.min_obs), "std")


@feature("realized_vol", family=FeatureFamily.VOLATILITY, param_rules=_W, param_order=("window",),
         default_session_policy=SessionPolicy.RESET_ON_GAP, lookback=lambda p: p["window"],
         description="sqrt of the rolling sum of squared 1-bar log returns (positive prices only)")
def _realized_vol(ctx: FeatureComputeContext):
    lr, invalid = seg_log_return(ctx.price(), ctx.segment_id, 1)
    if invalid.any():
        ctx.qa.add(FeatureIssueKind.NON_POSITIVE_LOG_INPUT, ctx.name,
                   f"{int(invalid.sum())} bar(s) excluded from realized vol (non-positive price)",
                   count=int(invalid.sum()))
    sq = lr * lr
    var = seg_rolling(sq, ctx.segment_id, ctx.params["window"], max(2, ctx.min_obs), "sum")
    return np.sqrt(var)


@feature("atr", family=FeatureFamily.VOLATILITY, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"],
         description="average true range (rolling mean of true range)")
def _atr(ctx: FeatureComputeContext):
    high = ctx.price("high" if "high" in ctx.frame.columns else ctx.price_field)
    low = ctx.price("low" if "low" in ctx.frame.columns else ctx.price_field)
    close = ctx.price("close")
    tr = true_range(high, low, close, ctx.segment_id)
    return seg_rolling(tr, ctx.segment_id, ctx.params["window"], ctx.min_obs, "mean")


@feature("range_vol", family=FeatureFamily.VOLATILITY, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"],
         description="rolling mean of the bar high-low range (absolute price units)")
def _range_vol(ctx: FeatureComputeContext):
    high = ctx.price("high" if "high" in ctx.frame.columns else ctx.price_field)
    low = ctx.price("low" if "low" in ctx.frame.columns else ctx.price_field)
    return seg_rolling(high - low, ctx.segment_id, ctx.params["window"], ctx.min_obs, "mean")


@feature("vol_change", family=FeatureFamily.VOLATILITY, param_rules=_W, param_order=("window",),
         default_session_policy=SessionPolicy.RESET_ON_GAP,
         lookback=lambda p: 2 * p["window"] + 1,
         description="change in rolling return-vol over the last `window` bars")
def _vol_change(ctx: FeatureComputeContext):
    r1 = _simple_returns(ctx)
    vol = seg_rolling(r1, ctx.segment_id, ctx.params["window"],
                      max(2, ctx.params["window"] - 1), "std")
    return seg_diff(vol, ctx.segment_id, ctx.params["window"])


@feature("vol_percentile", family=FeatureFamily.VOLATILITY,
         param_rules={"window": ParamRule(int, min=2, max=100_000),
                      "lookback": ParamRule(int, min=2, max=100_000)},
         param_order=("window", "lookback"),
         default_session_policy=SessionPolicy.RESET_ON_GAP,
         lookback=lambda p: p["window"] + p["lookback"],
         description="trailing percentile rank (0..1) of current return-vol within `lookback` bars")
def _vol_percentile(ctx: FeatureComputeContext):
    r1 = _simple_returns(ctx)
    vol = seg_rolling(r1, ctx.segment_id, ctx.params["window"],
                      max(2, ctx.params["window"] - 1), "std")
    return seg_rolling_apply(vol, ctx.segment_id, ctx.params["lookback"],
                             ctx.params["lookback"], rolling_rank_pct)
