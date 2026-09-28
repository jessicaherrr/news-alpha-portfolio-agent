"""Volume / liquidity features (section 9).

Only what OHLCV genuinely supports. These are NOT order-book imbalance, queue
position or bid/ask depth -- those need richer market data in a later phase.
"""
from __future__ import annotations

from alpha_agent.features.enums import FeatureFamily, SessionPolicy
from alpha_agent.features.qa import FeatureIssueKind
from alpha_agent.features.registry import FeatureComputeContext, ParamRule, feature
from alpha_agent.features.windows import safe_ratio, seg_pct_change, seg_rolling, seg_shift

_W = {"window": ParamRule(int, min=2, max=100_000)}


def _volume(ctx: FeatureComputeContext):
    if "volume" not in ctx.frame.columns:
        raise ValueError("volume features require a 'volume' column in the source frame")
    return ctx.column("volume")


@feature("volume", family=FeatureFamily.VOLUME, param_rules={}, param_order=(),
         lookback=1, description="raw bar volume (pass-through)")
def _volume_passthrough(ctx: FeatureComputeContext):
    return _volume(ctx)


@feature("avg_volume", family=FeatureFamily.VOLUME, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"], description="rolling mean bar volume")
def _avg_volume(ctx: FeatureComputeContext):
    return seg_rolling(_volume(ctx), ctx.segment_id, ctx.params["window"], ctx.min_obs, "mean")


@feature("rel_volume", family=FeatureFamily.VOLUME, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"] + 1,
         description="volume / PRIOR rolling-mean volume (>1 => above-average activity)")
def _rel_volume(ctx: FeatureComputeContext):
    v = _volume(ctx)
    prior_avg = seg_shift(
        seg_rolling(v, ctx.segment_id, ctx.params["window"], ctx.params["window"], "mean"),
        ctx.segment_id, 1,
    )
    val, zero = safe_ratio(v, prior_avg, eps=1e-9)
    if zero.any():
        ctx.qa.add(FeatureIssueKind.ZERO_DENOMINATOR, ctx.name,
                   f"{int(zero.sum())} row(s) had ~0 average volume -> relative volume missing",
                   count=int(zero.sum()))
    return val


@feature("volume_zscore", family=FeatureFamily.VOLUME, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"],
         description="(volume - rolling mean) / rolling std of volume; missing when std ~ 0")
def _volume_zscore(ctx: FeatureComputeContext):
    v = _volume(ctx)
    w = ctx.params["window"]
    mean = seg_rolling(v, ctx.segment_id, w, ctx.min_obs, "mean")
    std = seg_rolling(v, ctx.segment_id, w, max(2, ctx.min_obs), "std")
    val, zero = safe_ratio(v - mean, std, eps=1e-9)
    if zero.any():
        ctx.qa.add(FeatureIssueKind.ZERO_DENOMINATOR, ctx.name,
                   f"{int(zero.sum())} row(s) had ~0 volume std -> volume z-score missing",
                   count=int(zero.sum()))
    return val


@feature("range_volume_ratio", family=FeatureFamily.VOLUME, param_rules=_W, param_order=("window",),
         lookback=lambda p: p["window"],
         description="rolling mean (high-low) / rolling mean volume (a crude price-impact proxy)")
def _range_volume_ratio(ctx: FeatureComputeContext):
    high = ctx.price("high" if "high" in ctx.frame.columns else "close")
    low = ctx.price("low" if "low" in ctx.frame.columns else "close")
    w = ctx.params["window"]
    rng = seg_rolling(high - low, ctx.segment_id, w, w, "mean")
    vol = seg_rolling(_volume(ctx), ctx.segment_id, w, w, "mean")
    val, zero = safe_ratio(rng, vol, eps=1e-9)
    if zero.any():
        ctx.qa.add(FeatureIssueKind.ZERO_DENOMINATOR, ctx.name,
                   f"{int(zero.sum())} row(s) had ~0 average volume -> range/volume ratio missing",
                   count=int(zero.sum()))
    return val


@feature("amihud_illiq", family=FeatureFamily.VOLUME, param_rules=_W, param_order=("window",),
         default_session_policy=SessionPolicy.RESET_ON_GAP, lookback=lambda p: p["window"] + 1,
         description="rolling mean of |1-bar return| / volume (Amihud illiquidity proxy)")
def _amihud(ctx: FeatureComputeContext):
    r1, base = seg_pct_change(ctx.price("close"), ctx.segment_id, 1)
    r1 = r1.mask(base.notna() & (base <= 0.0))
    per_bar, zero = safe_ratio(r1.abs(), _volume(ctx), eps=1e-9)
    if zero.any():
        ctx.qa.add(FeatureIssueKind.ZERO_DENOMINATOR, ctx.name,
                   f"{int(zero.sum())} bar(s) had 0 volume -> excluded from Amihud",
                   count=int(zero.sum()))
    return seg_rolling(per_bar, ctx.segment_id, ctx.params["window"],
                       max(1, ctx.params["window"] - 1), "mean")
