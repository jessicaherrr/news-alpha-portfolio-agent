"""Core return features (section 5).

Signed-price aware. ``diff_*`` is pure arithmetic and valid for zero / negative
prices. ``return_*`` (percentage) is defined only where the base price is
strictly positive; ``log_return_*`` only where both endpoints are positive.
Undefined points are explicit missing values plus a recorded QA issue -- never a
silent ``NaN``/``inf``.
"""
from __future__ import annotations

from alpha_agent.features.enums import FeatureFamily, SessionPolicy
from alpha_agent.features.qa import FeatureIssueKind
from alpha_agent.features.registry import FeatureComputeContext, ParamRule, feature
from alpha_agent.features.windows import (
    seg_diff,
    seg_log_return,
    seg_pct_change,
    seg_rolling,
)

_N = {"n": ParamRule(int, min=1, max=100_000)}
_W = {"window": ParamRule(int, min=1, max=100_000)}


@feature("diff", family=FeatureFamily.RETURN, param_rules=_N, param_order=("n",),
         default_session_policy=SessionPolicy.RESET_ON_GAP, lookback=lambda p: p["n"],
         description="arithmetic price change over n bars; valid for signed/zero prices")
def _diff(ctx: FeatureComputeContext):
    return seg_diff(ctx.price(), ctx.segment_id, ctx.params["n"])


@feature("return", family=FeatureFamily.RETURN, param_rules=_N, param_order=("n",),
         default_session_policy=SessionPolicy.RESET_ON_GAP, lookback=lambda p: p["n"],
         description="simple percentage return over n bars; missing where base price <= 0")
def _return(ctx: FeatureComputeContext):
    s = ctx.price()
    out, base = seg_pct_change(s, ctx.segment_id, ctx.params["n"])
    zero = base.notna() & s.notna() & (base == 0.0)
    if zero.any():
        ctx.qa.add(FeatureIssueKind.NON_POSITIVE_RETURN_BASE, ctx.name,
                   f"{int(zero.sum())} row(s) have a zero base price -> return undefined",
                   count=int(zero.sum()))
    neg = base.notna() & (base < 0.0)
    if neg.any():
        # arithmetic is defined but the sign convention is unusual: suppress and flag
        out = out.mask(neg)
        ctx.qa.add(FeatureIssueKind.NEGATIVE_RETURN_BASE, ctx.name,
                   f"{int(neg.sum())} row(s) have a negative base price -> percentage return "
                   f"suppressed (use diff_{ctx.params['n']} instead)",
                   count=int(neg.sum()))
    return out


@feature("log_return", family=FeatureFamily.RETURN, param_rules=_N, param_order=("n",),
         default_session_policy=SessionPolicy.RESET_ON_GAP, lookback=lambda p: p["n"],
         description="log return over n bars; defined only where both endpoints > 0")
def _log_return(ctx: FeatureComputeContext):
    out, invalid = seg_log_return(ctx.price(), ctx.segment_id, ctx.params["n"])
    if invalid.any():
        ctx.qa.add(FeatureIssueKind.NON_POSITIVE_LOG_INPUT, ctx.name,
                   f"{int(invalid.sum())} row(s) have a non-positive price -> log return missing",
                   count=int(invalid.sum()))
    return out


@feature("mean_return", family=FeatureFamily.RETURN, param_rules=_W, param_order=("window",),
         default_session_policy=SessionPolicy.RESET_ON_GAP, lookback=lambda p: p["window"],
         description="rolling mean of 1-bar simple returns")
def _mean_return(ctx: FeatureComputeContext):
    r1, base = seg_pct_change(ctx.price(), ctx.segment_id, 1)
    r1 = r1.mask(base.notna() & (base <= 0.0))
    return seg_rolling(r1, ctx.segment_id, ctx.params["window"], ctx.min_obs, "mean")


@feature("cum_log_return", family=FeatureFamily.RETURN, param_rules=_W, param_order=("window",),
         default_session_policy=SessionPolicy.RESET_ON_GAP, lookback=lambda p: p["window"],
         description="rolling sum of 1-bar log returns over the window (positive prices only)")
def _cum_log_return(ctx: FeatureComputeContext):
    lr, invalid = seg_log_return(ctx.price(), ctx.segment_id, 1)
    if invalid.any():
        ctx.qa.add(FeatureIssueKind.NON_POSITIVE_LOG_INPUT, ctx.name,
                   f"{int(invalid.sum())} bar(s) excluded (non-positive price)",
                   count=int(invalid.sum()))
    return seg_rolling(lr, ctx.segment_id, ctx.params["window"], ctx.min_obs, "sum")
