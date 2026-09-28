"""Phase 04 -- research-only back-adjusted series (Layer 4) + point-in-time safety."""
from __future__ import annotations

import pandas as pd
import pytest
from alpha_agent.data.backadjust import (
    AdjustmentMode,
    assert_no_executable_identity,
    build_back_adjusted_series,
    reject_adjusted_execution,
)
from alpha_agent.data.continuous import build_continuous_series
from alpha_agent.data.diagnostics import DiagnosticsReport, PipelineError
from alpha_agent.data.rolls import build_roll_events
from alpha_agent.schemas.market_data import BACKADJUSTED_BAR_COLUMNS
from futures_fixtures import (
    MIN_NS,
    NQH7,
    NQU6,
    NQZ6,
    canonical_bars,
    registry,
    two_contract_history,
)


def _cont_and_rolls(bars):
    reg = registry()
    rolls, _ = build_roll_events(bars, continuous_symbol="NQ.v.0", registry=reg)
    cont, _ = build_continuous_series(bars, continuous_symbol="NQ.v.0", registry=reg, rolls=rolls)
    return cont, rolls


def test_additive_backadjust_removes_discontinuity_keeps_newest():
    bars = two_contract_history(u6_close=29000.0, z6_open=29100.0)
    cont, rolls = _cont_and_rolls(bars)
    adj, report = build_back_adjusted_series(cont, rolls, continuous_symbol="NQ.v.0")

    assert list(adj.columns) == list(BACKADJUSTED_BAR_COLUMNS)
    # newest contract (post-roll) unchanged
    post = adj[adj["ts_event_ns"] >= rolls[0].effective_ts_ns]
    raw_post = cont[cont["ts_event_ns"] >= rolls[0].effective_ts_ns]
    assert list(post["close"]) == list(raw_post["close"])
    assert (post["cumulative_adjustment"] == 0.0).all()
    # pre-roll history shifted up by +100 -> the seam disappears
    pre = adj[adj["ts_event_ns"] < rolls[0].effective_ts_ns]
    assert (pre["cumulative_adjustment"] == 100.0).all()
    assert pre["close"].iloc[-1] == 29100.0            # was 29000, now continuous with NQZ6
    assert report.ok


def test_cumulative_multiple_rolls():
    # A(100) -> B(105, gap +5) -> C(110, gap +5)
    bars = canonical_bars([
        {"spec": NQU6, "n": 3, "base_price": 100.0, "step": 0.0},
        {"spec": NQZ6, "n": 3, "base_price": 105.0, "step": 0.0},
        {"spec": NQH7, "n": 3, "base_price": 110.0, "step": 0.0},
    ])
    cont, rolls = _cont_and_rolls(bars)
    assert [r.additive_gap for r in rolls] == [5.0, 5.0]
    adj, _ = build_back_adjusted_series(cont, rolls, continuous_symbol="NQ.v.0")
    seg = dict(zip(adj["ts_event_ns"], adj["cumulative_adjustment"]))
    a_ts = cont[cont["active_raw_symbol"] == "NQU6"]["ts_event_ns"]
    b_ts = cont[cont["active_raw_symbol"] == "NQZ6"]["ts_event_ns"]
    c_ts = cont[cont["active_raw_symbol"] == "NQH7"]["ts_event_ns"]
    assert all(seg[t] == 10.0 for t in a_ts)   # both future gaps
    assert all(seg[t] == 5.0 for t in b_ts)    # only the BC gap
    assert all(seg[t] == 0.0 for t in c_ts)    # newest, unadjusted


def test_back_adjusted_bar_has_no_executable_identity():
    bars = two_contract_history()
    cont, rolls = _cont_and_rolls(bars)
    adj, _ = build_back_adjusted_series(cont, rolls, continuous_symbol="NQ.v.0")
    for col in ("instrument_id", "raw_symbol", "active_instrument_id", "active_raw_symbol"):
        assert col not in adj.columns
    # and the guard trips if identity is spliced back in
    leaked = adj.assign(instrument_id=101)
    with pytest.raises(PipelineError, match="executable contract identity"):
        assert_no_executable_identity(leaked, DiagnosticsReport())


def test_execution_path_rejects_non_raw_contract_domains():
    for domain in ("back_adjusted", "raw_continuous"):
        with pytest.raises(PipelineError, match="execution path"):
            reject_adjusted_execution(domain)
    reject_adjusted_execution("raw_contract")  # allowed


def test_point_in_time_cannot_see_a_future_roll():
    # A(100) then B(105) then C(110). Two rolls.
    bars = canonical_bars([
        {"spec": NQU6, "n": 4, "base_price": 100.0, "step": 0.0},
        {"spec": NQZ6, "n": 4, "base_price": 105.0, "step": 0.0},
        {"spec": NQH7, "n": 4, "base_price": 110.0, "step": 0.0},
    ])
    cont, rolls = _cont_and_rolls(bars)
    roll2 = rolls[1].effective_ts_ns
    t0 = int(cont["ts_event_ns"].iloc[0])   # a bar in segment A

    retro, _ = build_back_adjusted_series(
        cont, rolls, continuous_symbol="NQ.v.0", mode=AdjustmentMode.RETROSPECTIVE_RESEARCH)
    # as of just after roll1 but before roll2: only roll1 is known
    as_of = roll2 - MIN_NS
    pit, _ = build_back_adjusted_series(
        cont, rolls, continuous_symbol="NQ.v.0",
        mode=AdjustmentMode.POINT_IN_TIME, as_of_ts_ns=as_of)

    retro_t0 = float(retro.loc[retro["ts_event_ns"] == t0, "cumulative_adjustment"].iloc[0])
    pit_t0 = float(pit.loc[pit["ts_event_ns"] == t0, "cumulative_adjustment"].iloc[0])

    assert retro_t0 == 10.0                 # sees both future rolls
    assert pit_t0 == 5.0                    # sees ONLY roll1 (known by as_of)
    assert pit["ts_event_ns"].max() <= as_of  # series stops at as_of
    assert (pit["adjustment_mode"] == "point_in_time").all()
    assert (pit["adjusted_through_ts_ns"] == as_of).all()
    assert not pd.Series(retro["ts_event_ns"] > roll2).empty
