"""Phase 09 -- carry / term-structure interface with synthetic overlapping
contracts (checklist Q)."""
from __future__ import annotations

import math

import pytest
from alpha_agent.features.carry import (
    CarryError,
    ContractQuote,
    CurveObservation,
    annualized_carry,
    assert_carry_domain,
    build_curve_observations,
    calendar_spread,
    carry_frame,
    synthetic_curve,
)
from alpha_agent.schemas.market_data import PriceDomain


def _quote(iid, sym, exp_ns, price):
    return ContractQuote(instrument_id=iid, raw_symbol=sym, expiration_ns=exp_ns, price=price)


DAY = 86_400_000_000_000


def test_calendar_spread_and_annualized_carry_are_hand_computable():
    obs = synthetic_curve(1_000_000_000_000, front_price=80.0, next_price=78.0, spacing_days=30)
    assert calendar_spread(obs.front.price, obs.nxt.price) == -2.0     # next - front
    # backwardation: front above next -> positive carry for a long front
    ann = annualized_carry(80.0, 78.0, 30.0)
    assert ann == pytest.approx((80 - 78) / 78 * (365 / 30))
    assert ann > 0


def test_contango_gives_negative_carry():
    ann = annualized_carry(78.0, 80.0, 30.0)
    assert ann == pytest.approx((78 - 80) / 80 * (365 / 30))
    assert ann < 0


def test_carry_is_undefined_for_a_non_positive_far_price_not_inf():
    assert annualized_carry(-5.0, 0.0, 30.0) is None
    assert annualized_carry(-5.0, -20.0, 30.0) is None      # percentage not meaningful
    # calendar spread is still defined across the sign change
    assert calendar_spread(-5.0, -20.0) == -15.0


def test_carry_must_come_from_raw_contracts_not_back_adjusted():
    with pytest.raises(CarryError):
        assert_carry_domain(PriceDomain.BACK_ADJUSTED)
    with pytest.raises(CarryError):
        assert_carry_domain(PriceDomain.RAW_CONTINUOUS)
    assert_carry_domain(PriceDomain.RAW_CONTRACT)  # ok


def test_build_curve_observations_needs_two_overlapping_contracts():
    base = 1_700_000_000_000_000_000
    quotes = {
        base: [_quote(1, "CLF6", base + 10 * DAY, 70.0)],                       # only front
        base + 60_000_000_000: [
            _quote(1, "CLF6", base + 10 * DAY, 70.5),
            _quote(2, "CLG6", base + 40 * DAY, 71.0),
        ],
    }
    obs, warnings = build_curve_observations(quotes, root_symbol="CL")
    assert len(obs) == 1                                    # the single-contract ts is skipped
    assert any("only 1 contract" in w for w in warnings)
    frame = carry_frame(obs)
    row = frame.iloc[0]
    assert row["front_raw_symbol"] == "CLF6" and row["next_raw_symbol"] == "CLG6"
    assert row["calendar_spread"] == pytest.approx(0.5)
    assert row["maturity_spacing_days"] == pytest.approx(30.0)
    assert row["annualized_carry"] == pytest.approx((70.5 - 71.0) / 71.0 * (365 / 30))


def test_curve_observation_rejects_unordered_maturities():
    ts = 1_700_000_000_000_000_000
    with pytest.raises(ValueError):  # CarryError, re-raised by pydantic as ValidationError
        CurveObservation(
            ts_event_ns=ts, root_symbol="CL",
            quotes=(_quote(1, "CLG6", ts + 40 * DAY, 71.0),
                    _quote(2, "CLF6", ts + 10 * DAY, 70.0)),
        )


def test_empty_curve_is_reported_not_fabricated():
    obs, warnings = build_curve_observations({}, root_symbol="CL")
    assert obs == []
    assert any("empty" in w for w in warnings)


def test_carry_frame_is_point_in_time_each_row_uses_only_its_own_curve():
    o1 = synthetic_curve(10, front_price=80.0, next_price=79.0, spacing_days=30)
    o2 = synthetic_curve(20, front_price=90.0, next_price=95.0, spacing_days=30)
    frame = carry_frame([o1, o2])
    assert list(frame["calendar_spread"]) == [-1.0, 5.0]
    assert not math.isnan(frame["annualized_carry"].iloc[0])
