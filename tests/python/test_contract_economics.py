"""Phase 03.5 -- contract economics from the REAL observed NQ definition."""
from __future__ import annotations

import pytest
from alpha_agent.data.contract_economics import (
    EconomicsError,
    derive_contract_economics,
    usable_number,
)

# Exact values observed in the live GLBX.MDP3 NQU6 (instrument_id 42004177) record.
NQ = {
    "min_price_increment": 0.25,
    "min_price_increment_amount": 0.05,
    "display_factor": 0.01,
    "unit_of_measure_qty": 20.0,
    "unit_of_measure": "IPNT",
    "main_fraction": None,             # real field is the 255 sentinel
    "contract_multiplier": 2147483647,  # INT32_MAX sentinel
}


def test_real_nq_derivation():
    econ = derive_contract_economics(**NQ)
    assert econ.quote_tick_size == 0.25
    assert econ.contract_size == 20.0
    assert econ.price_scale == 1.0
    assert econ.point_value_usd == 20.0
    assert econ.tick_value_usd == 5.0
    assert econ.cross_check()  # tick_size * point_value == tick_value


def test_contract_multiplier_sentinel_is_ignored():
    # even with an absurd contract_multiplier, the result is driven by
    # unit_of_measure_qty + the price-scale derivation.
    econ = derive_contract_economics(**{**NQ, "contract_multiplier": 999999})
    assert econ.point_value_usd == 20.0
    assert usable_number(2147483647) is None
    assert usable_number(255) is None
    assert usable_number(-1) is None
    assert usable_number(20.0) == 20.0


def test_missing_unit_of_measure_qty_raises_not_guesses():
    with pytest.raises(EconomicsError, match="unit_of_measure_qty"):
        derive_contract_economics(**{**NQ, "unit_of_measure_qty": None})
    with pytest.raises(EconomicsError):
        derive_contract_economics(**{**NQ, "unit_of_measure_qty": 4294967295})


def test_missing_tick_value_inputs_raise():
    with pytest.raises(EconomicsError, match="tick value"):
        derive_contract_economics(**{**NQ, "min_price_increment_amount": None})
    with pytest.raises(EconomicsError, match="tick value"):
        derive_contract_economics(**{**NQ, "display_factor": None})


def test_fractional_product_is_refused_for_now():
    # non-sentinel main_fraction => CBOT-Treasury-style fractional quoting
    with pytest.raises(EconomicsError, match="fractionally-quoted"):
        derive_contract_economics(**{**NQ, "main_fraction": 4})


def test_non_unit_price_scale_is_supported():
    # a fabricated cent-quoted product: quoted in cents, tick 1 cent,
    # min_price_increment_amount 12.5, display_factor 1.0, size 1000
    # tick_value_from_def = 12.5 / 1.0 = 12.5
    # price_scale = 12.5 / (0.01 * 1000) = 1.25
    # point_value = 1000 * 1.25 = 1250 ; tick_value = 0.01 * 1250 = 12.5
    econ = derive_contract_economics(
        min_price_increment=0.01,
        min_price_increment_amount=12.5,
        display_factor=1.0,
        unit_of_measure_qty=1000.0,
        unit_of_measure="LBS",
    )
    assert econ.price_scale == 1.25
    assert econ.point_value_usd == 1250.0
    assert econ.tick_value_usd == pytest.approx(12.5)
    assert econ.cross_check()


# ==========================================================================
# Phase 13.5C correction -- FRACTIONALLY-QUOTED (percent-of-par) products.
#
# Real observed GLBX.MDP3 definition fields for the five Phase 13.5B roots. The
# expected USD economics are the PUBLISHED CME contract specifications -- they
# are never edited to match whatever the derivation happens to produce.
# ==========================================================================
REAL_DEFINITION_FIELDS = {
    # root: (mpi, mpia, display_factor, uomq, main_fraction)
    "NQ": (0.25, 0.05, 0.01, 20.0, None),
    "ES": (0.25, 0.125, 0.01, 50.0, None),
    "GC": (0.1, 1.0, 0.1, 100.0, None),
    "CL": (0.01, 0.1, 0.01, 1000.0, None),
    "ZN": (0.015625, 0.001, 1.0, 100000.0, 32.0),   # fractional: 1/32nds of a percent of par
}
PUBLISHED_CME = {
    # root: (point_value_usd, tick_size, tick_value_usd)
    "NQ": (20.0, 0.25, 5.00),
    "ES": (50.0, 0.25, 12.50),
    "GC": (100.0, 0.1, 10.00),
    "CL": (1000.0, 0.01, 10.00),
    "ZN": (1000.0, 0.015625, 15.625),
}


def _econ(root: str, **over):
    mpi, mpia, df, uomq, mf = REAL_DEFINITION_FIELDS[root]
    kw = {
        "min_price_increment": mpi,
        "min_price_increment_amount": mpia,
        "display_factor": df,
        "unit_of_measure_qty": uomq,
        "unit_of_measure": "USD" if root == "ZN" else "IPNT",
        "main_fraction": mf,
        "contract_multiplier": 2147483647,   # INT32_MAX sentinel on every real record
        "allow_fractional": root == "ZN",
    }
    kw.update(over)
    return derive_contract_economics(**kw)


def test_zn_fractional_percent_of_par_matches_published_cme_spec():
    from alpha_agent.data.fractional_price import QuoteConvention

    econ = _econ("ZN")
    assert econ.quote_convention is QuoteConvention.FRACTIONAL_32
    assert econ.price_scale == 0.01                 # percent of par -- NOT from mpia/display_factor
    assert econ.contract_size == 100000.0           # FACE value
    assert econ.quote_tick_size == 0.015625
    assert econ.point_value_usd == 1000.0
    assert econ.tick_value_usd == pytest.approx(15.625)
    assert econ.cross_check()
    # the misleading field is deliberately not consulted on this path
    assert "min_price_increment_amount not used" in econ.source


def test_zn_derivation_ignores_min_price_increment_amount_and_display_factor():
    base = _econ("ZN")
    # perturbing / removing the fields that were mis-scaled changes nothing
    for over in ({"min_price_increment_amount": 999.0},
                 {"display_factor": 0.03125},
                 {"min_price_increment_amount": None},
                 {"display_factor": None}):
        assert _econ("ZN", **over).point_value_usd == base.point_value_usd == 1000.0


def test_zn_contract_multiplier_int32_sentinel_still_ignored():
    assert _econ("ZN", contract_multiplier=2147483647).point_value_usd == 1000.0
    assert _econ("ZN", contract_multiplier=999999).point_value_usd == 1000.0
    assert usable_number(2147483647) is None


def test_fractional_product_still_fails_loudly_without_allow_fractional():
    with pytest.raises(EconomicsError, match="fractionally-quoted"):
        _econ("ZN", allow_fractional=False)


@pytest.mark.parametrize("root", ["NQ", "ES", "GC", "CL"])
def test_decimal_roots_are_exactly_unchanged_by_the_fractional_correction(root):
    from alpha_agent.data.fractional_price import QuoteConvention

    pv, tick, tv = PUBLISHED_CME[root]
    econ = _econ(root)
    assert econ.quote_convention is QuoteConvention.DECIMAL
    assert econ.quote_tick_size == tick
    assert econ.point_value_usd == pv
    assert econ.tick_value_usd == pytest.approx(tv)
    assert econ.price_scale == 1.0
    assert econ.cross_check()
    # the decimal path still requires its inputs
    with pytest.raises(EconomicsError, match="tick value"):
        _econ(root, min_price_increment_amount=None)


def test_all_five_real_roots_match_the_published_cme_specification():
    for root, (pv, tick, tv) in PUBLISHED_CME.items():
        econ = _econ(root)
        assert (econ.point_value_usd, econ.quote_tick_size) == (pv, tick), root
        assert econ.tick_value_usd == pytest.approx(tv), root


def test_signed_and_negative_prices_are_unaffected_by_economics():
    """CL traded below zero on 2020-04-20. Contract economics are a pure scale --
    they carry no sign assumption and are unchanged by this correction."""
    from alpha_agent.schemas.market_data import is_normalized_price

    econ = _econ("CL")
    assert econ.point_value_usd == 1000.0
    for px in (-40.32, -0.01, 0.0, 61.5):
        assert is_normalized_price(px)
        # PnL is a linear function of the price move, valid through zero
        assert (px - 10.0) * econ.point_value_usd == pytest.approx((px - 10.0) * 1000.0)
