"""Phase 04 -- fractional quoting / Treasury economics (typed architecture).

Real ZN Databento field interpretation is verified in Phase 04.5; these tests
prove the representation does not silently borrow NQ decimal assumptions.
"""
from __future__ import annotations

import pytest
from alpha_agent.data.contract_economics import EconomicsError, derive_contract_economics
from alpha_agent.data.fractional_price import (
    QuoteConvention,
    decimal_to_display,
    display_to_decimal,
    normalized_tick_size,
)

# A deterministic ZN-shaped fixture (10-Year T-Note). Half-tick = 1/64 point;
# $100,000 face; 1/64 point == $15.625. Values are illustrative, NOT scraped.
ZN = {
    "min_price_increment": 1.0 / 64,        # 0.015625 normalized points
    "min_price_increment_amount": 15.625,   # USD per half-32nd
    "display_factor": 1.0,
    "unit_of_measure": "DLR",
    "unit_of_measure_qty": 100_000.0,
    "main_fraction": 4,                      # non-sentinel -> fractionally quoted
}


def test_fractional_is_refused_without_opt_in():
    with pytest.raises(EconomicsError, match="fractionally-quoted"):
        derive_contract_economics(**ZN)


def test_fractional_economics_do_not_use_nq_assumptions():
    econ = derive_contract_economics(**ZN, allow_fractional=True)
    assert econ.quote_convention is QuoteConvention.FRACTIONAL_32
    assert econ.quote_tick_size == pytest.approx(1.0 / 64)
    # tick value from the definition, convention-independent
    assert econ.tick_value_usd == pytest.approx(15.625)
    assert econ.point_value_usd == pytest.approx(1000.0)   # $1000 per full point on $100k face
    assert econ.cross_check()
    # NQ's answer ($5 tick, $20 point) is nowhere near
    assert econ.tick_value_usd != 5.0 and econ.point_value_usd != 20.0


def test_display_conversion_roundtrips():
    assert display_to_decimal(110, 16, tick_fraction=32) == pytest.approx(110.5)
    assert display_to_decimal(110, 17, tick_fraction=64) == pytest.approx(110 + 17 / 64)
    assert decimal_to_display(110.5, tick_fraction=32) == "110'160"
    assert normalized_tick_size(64) == pytest.approx(0.015625)
    with pytest.raises(ValueError):
        display_to_decimal(110, 40, tick_fraction=32)
