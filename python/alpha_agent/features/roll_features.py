"""Futures roll features (section 10).

Point-in-time-legitimate features (``is_roll_day``, ``bars_since_roll``,
``days_since_roll``) use only the observed ``active_instrument_id`` transitions
at or before ``T``. Retrospective features (``bars_until_next_roll``,
``contract_transition_indicator``) look forward across the sample and are **not**
point-in-time safe -- the computer refuses them unless
``require_point_in_time=False`` is passed for explicit offline research.

Retrospective roll knowledge must never leak into a point-in-time feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_agent.features.enums import FeatureFamily, SessionPolicy
from alpha_agent.features.registry import FeatureComputeContext, ParamRule, feature


def _boundary_mask(ctx: FeatureComputeContext) -> pd.Series:
    f = ctx.frame
    if "active_instrument_id" in f.columns:
        aid = f["active_instrument_id"]
        b = aid.ne(aid.shift())
    elif "instrument_id" in f.columns:
        aid = f["instrument_id"]
        b = aid.ne(aid.shift())
    elif "is_roll_boundary" in f.columns:
        b = f["is_roll_boundary"].astype(bool)
    else:
        raise ValueError(
            "roll features need 'active_instrument_id', 'instrument_id' or 'is_roll_boundary'"
        )
    b = b.copy()
    b.iloc[0] = False  # the first observed contract is not itself a roll
    if "is_roll_boundary" in f.columns:
        extra = f["is_roll_boundary"].astype(bool).copy()
        extra.iloc[0] = False
        b = b | extra
    return b.fillna(False).astype(bool)


@feature("is_roll_day", family=FeatureFamily.ROLL, param_rules={}, param_order=(),
         default_session_policy=SessionPolicy.CONTINUOUS, lookback=1,
         description="1.0 on the first bar of a new active contract (observed), else 0.0")
def _is_roll_day(ctx: FeatureComputeContext):
    return _boundary_mask(ctx).astype("float64")


@feature("bars_since_roll", family=FeatureFamily.ROLL, param_rules={}, param_order=(),
         default_session_policy=SessionPolicy.CONTINUOUS, lookback=1,
         description="bars elapsed since the most recent observed roll (0 on the roll bar)")
def _bars_since_roll(ctx: FeatureComputeContext):
    b = _boundary_mask(ctx)
    grp = b.cumsum()
    return grp.groupby(grp).cumcount().astype("float64")


@feature("days_since_roll", family=FeatureFamily.ROLL, param_rules={}, param_order=(),
         default_session_policy=SessionPolicy.CONTINUOUS, lookback=1,
         description="trading days since the most recent observed roll (needs 'trading_day')")
def _days_since_roll(ctx: FeatureComputeContext):
    f = ctx.frame
    if "trading_day" not in f.columns:
        raise ValueError("days_since_roll needs a 'trading_day' column")
    b = _boundary_mask(ctx).to_numpy()
    day = pd.to_datetime(f["trading_day"]).to_numpy().astype("datetime64[D]").astype("int64")
    last = np.empty(len(day), dtype="int64")
    cur = day[0] if len(day) else 0
    for i in range(len(day)):
        if b[i]:
            cur = day[i]
        last[i] = cur
    return pd.Series((day - last).astype("float64"), index=f.index)


@feature("bars_until_next_roll", family=FeatureFamily.ROLL, param_rules={}, param_order=(),
         default_session_policy=SessionPolicy.CONTINUOUS, lookback=1,
         point_in_time_safe=False,
         description="RETROSPECTIVE: bars until the next observed roll (looks forward)")
def _bars_until_next_roll(ctx: FeatureComputeContext):
    b = _boundary_mask(ctx)
    rev = b.iloc[::-1]
    grp = rev.cumsum()
    cnt = grp.groupby(grp).cumcount()
    out = cnt.iloc[::-1]
    # rows at or after the last roll have no "next roll" -> missing
    grp_fwd = grp.iloc[::-1]
    out = out.where(grp_fwd > 0)
    return out.astype("float64")


@feature("contract_transition_indicator", family=FeatureFamily.ROLL,
         param_rules={"k": ParamRule(int, min=0, max=10_000)}, param_order=("k",),
         default_session_policy=SessionPolicy.CONTINUOUS, lookback=lambda p: p["k"] + 1,
         point_in_time_safe=False,
         description="RETROSPECTIVE: 1.0 within k bars either side of an observed roll")
def _contract_transition_indicator(ctx: FeatureComputeContext):
    b = _boundary_mask(ctx)
    k = ctx.params["k"]
    near = b.copy()
    for shift in range(1, k + 1):
        near = near | b.shift(shift, fill_value=False) | b.shift(-shift, fill_value=False)
    return near.astype("float64")
