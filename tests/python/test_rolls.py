"""Phase 04 -- roll map (Layer 6)."""
from __future__ import annotations

import pytest
from alpha_agent.data.diagnostics import DiagnosticKind, PipelineError
from alpha_agent.data.rolls import RollPricePolicy, build_roll_events
from futures_fixtures import (
    BASE_NS,
    MIN_NS,
    NQH7,
    NQU6,
    NQZ6,
    canonical_bars,
    registry,
    two_contract_history,
)


def test_two_contract_history_makes_exactly_one_roll():
    bars = two_contract_history(u6_close=29000.0, z6_open=29100.0)
    rolls, report = build_roll_events(bars, continuous_symbol="NQ.v.0", registry=registry())
    assert len(rolls) == 1
    r = rolls[0]
    assert (r.from_instrument_id, r.to_instrument_id) == (101, 102)
    assert (r.from_raw_symbol, r.to_raw_symbol) == ("NQU6", "NQZ6")
    assert r.from_price == 29000.0 and r.to_price == 29100.0
    assert r.additive_gap == 100.0
    assert r.from_ts_ns is not None and r.to_ts_ns == r.effective_ts_ns
    assert report.ok


def test_roll_trigger_is_instrument_id_not_month_string():
    # same contract month label but a real instrument_id change still rolls;
    # here we just assert the transition drives it (no month parsing anywhere).
    bars = canonical_bars([
        {"spec": NQU6, "n": 3, "base_price": 100.0},
        {"spec": NQZ6, "n": 3, "base_price": 105.0},
        {"spec": NQH7, "n": 3, "base_price": 110.0},
    ])
    rolls, _ = build_roll_events(bars, continuous_symbol="NQ.v.0", registry=registry())
    assert [(x.from_instrument_id, x.to_instrument_id) for x in rolls] == [(101, 102), (102, 103)]


def test_price_policy_recorded_and_configurable():
    bars = two_contract_history()
    r_open = build_roll_events(bars, continuous_symbol="NQ.v.0", registry=registry(),
                               price_policy=RollPricePolicy.PREV_CLOSE_NEW_OPEN)[0][0]
    r_close = build_roll_events(bars, continuous_symbol="NQ.v.0", registry=registry(),
                                price_policy=RollPricePolicy.PREV_CLOSE_NEW_CLOSE)[0][0]
    assert r_open.price_policy == "prev_close_new_open"
    assert r_close.price_policy == "prev_close_new_close"


def test_backward_roll_timestamp_is_fatal():
    bars = two_contract_history()
    bars.loc[bars.index[-1], "ts_event_ns"] = BASE_NS - MIN_NS  # NQZ6 bar earlier than NQU6
    with pytest.raises(PipelineError, match="monotonic"):
        build_roll_events(bars, continuous_symbol="NQ.v.0", registry=registry())


def test_root_mismatch_is_fatal():
    from futures_fixtures import ESZ6

    bars = canonical_bars([
        {"spec": NQU6, "n": 3, "base_price": 100.0},
        {"spec": ESZ6, "n": 3, "base_price": 5000.0},
    ])
    with pytest.raises(PipelineError, match="roots"):
        build_roll_events(bars, continuous_symbol="NQ.v.0", registry=registry(NQU6, ESZ6))


def test_unknown_instrument_during_roll_is_fatal():
    bars = two_contract_history()
    with pytest.raises(PipelineError, match="unknown instrument"):
        build_roll_events(bars, continuous_symbol="NQ.v.0", registry=registry(NQU6))  # no NQZ6


def test_missing_reference_price_downgrades_gap_to_none():
    bars = two_contract_history()
    # blank the new contract's opening price. Missing data is NaN, never 0.0 --
    # a zero (or negative) price is valid data (historical CL), not "missing".
    roll_bar = bars.index[5]
    bars.loc[roll_bar, ["open", "high", "low", "close"]] = float("nan")
    rolls, report = build_roll_events(bars, continuous_symbol="NQ.v.0", registry=registry())
    assert rolls[0].additive_gap is None
    assert report.has_kind(DiagnosticKind.MISSING_ALIGNED_ROLL_REFERENCE_PRICE)


def test_roll_endpoint_cannot_be_continuous():
    from alpha_agent.schemas.market_data import RollEvent

    with pytest.raises(ValueError):
        RollEvent(continuous_symbol="NQ.v.0", effective_ts_ns=BASE_NS,
                  from_instrument_id=1, to_instrument_id=2,
                  from_raw_symbol="NQ.v.0", to_raw_symbol="NQZ6")
