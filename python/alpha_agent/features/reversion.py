"""Mean-reversion features (section 7).

Every feature requires a minimum number of observations and guards against an
unstable division when the rolling standard deviation is near zero (the value
becomes an explicit missing + a ZERO_DENOMINATOR issue, never ``inf``).
"""
from __future__ import annotations

from alpha_agent.features.enums import FeatureFamily
from alpha_agent.features.qa import FeatureIssueKind
from alpha_agent.features.registry import FeatureComputeContext, ParamRule, feature
from alpha_agent.features.windows import safe_ratio, seg_diff, seg_rolling

_W = {"window": ParamRule(int, min=2, max=100_000)}


def _mean_std(ctx: FeatureComputeContext):
    p = ctx.price()
    w = ctx.params["window"]
    mean = seg_rolling(p, ctx.segment_id, w, ctx.min_obs, "mean")
    std = seg_rolling(p, ctx.segment_id, w, max(2, ctx.min_obs), "std")
    return p, mean, std


@feature("rolling_mean", family=FeatureFamily.REVERSION, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"], description="backward rolling mean of the price")
def _rolling_mean(ctx: FeatureComputeContext):
    return seg_rolling(ctx.price(), ctx.segment_id, ctx.params["window"], ctx.min_obs, "mean")


@feature("rolling_std", family=FeatureFamily.REVERSION, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"], description="backward rolling std (ddof=1) of the price")
def _rolling_std(ctx: FeatureComputeContext):
    return seg_rolling(ctx.price(), ctx.segment_id, ctx.params["window"],
                       max(2, ctx.min_obs), "std")


@feature("zscore", family=FeatureFamily.REVERSION, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"],
         description="(price - rolling mean) / rolling std; missing when std ~ 0")
def _zscore(ctx: FeatureComputeContext):
    p, mean, std = _mean_std(ctx)
    val, zero = safe_ratio(p - mean, std)
    if zero.any():
        ctx.qa.add(FeatureIssueKind.ZERO_DENOMINATOR, ctx.name,
                   f"{int(zero.sum())} row(s) had rolling std ~ 0 -> z-score missing",
                   count=int(zero.sum()))
    return val


@feature("dist_from_ma", family=FeatureFamily.REVERSION, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"],
         description="price minus rolling mean (absolute price units; signed-safe)")
def _dist_from_ma(ctx: FeatureComputeContext):
    p, mean, _ = _mean_std(ctx)
    return p - mean


@feature("norm_dev", family=FeatureFamily.REVERSION, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"],
         description="(price - rolling mean) / rolling mean; missing when the mean ~ 0")
def _norm_dev(ctx: FeatureComputeContext):
    p, mean, _ = _mean_std(ctx)
    val, zero = safe_ratio(p - mean, mean)
    if zero.any():
        ctx.qa.add(FeatureIssueKind.ZERO_DENOMINATOR, ctx.name,
                   f"{int(zero.sum())} row(s) had a rolling mean ~ 0 -> norm. deviation missing",
                   count=int(zero.sum()))
    return val


@feature("reversal", family=FeatureFamily.REVERSION,
         param_rules={"n": ParamRule(int, min=1, max=100_000)}, param_order=("n",),
         lookback=lambda p: p["n"],
         description="negative n-bar price change (short-horizon reversal signal; signed-safe)")
def _reversal(ctx: FeatureComputeContext):
    return -seg_diff(ctx.price(), ctx.segment_id, ctx.params["n"])
