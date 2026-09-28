"""Pure formatting/row-builder tests for `alpha_agent.ui.market_contracts`
(Checkpoint C) -- no Streamlit runtime needed for these; the Streamlit-level
integration is covered in `test_market_product_detail.py`.
"""
from __future__ import annotations

from datetime import UTC, datetime

from alpha_agent.marketdata.databento_schemas import (
    ContractEconomicsView,
    FuturesContract,
    TermStructurePoint,
)
from alpha_agent.ui import market_contracts as mc


def _econ(*, quote_convention: str, quote_tick_size: float) -> ContractEconomicsView:
    return ContractEconomicsView(
        quote_tick_size=quote_tick_size, contract_size=1000.0, unit_of_measure="USD",
        point_value_usd=1000.0, tick_value_usd=quote_tick_size * 1000.0, quote_convention=quote_convention,
        source="derived",
    )


def test_format_display_price_na_for_missing_price():
    display, detail = mc.format_display_price(None, None)
    assert display == "N/A"
    assert detail is None


def test_format_display_price_plain_decimal_for_decimal_quoted_product():
    display, detail = mc.format_display_price(100.25, _econ(quote_convention="decimal", quote_tick_size=0.25))
    assert display == "100.25"
    assert detail is None


def test_format_display_price_uses_existing_fractional_utility_for_treasuries():
    """ZN half-32nd (1/64) tick -- reuses `alpha_agent.data.fractional_price.
    decimal_to_display` verbatim, never a new conversion (CLAUDE.md/Section 10)."""
    econ = _econ(quote_convention="fractional_32", quote_tick_size=1 / 64)
    display, detail = mc.format_display_price(106.203125, econ)
    assert "'" in display  # CME 32nds notation
    assert detail is not None and "106.203125" in detail


def test_contract_row_marks_front_and_display_independently():
    front_only = FuturesContract(
        root_symbol="ZN", raw_symbol="ZNU6", instrument_id=1, expiration=datetime(2026, 9, 21, tzinfo=UTC),
        is_front_month=True, is_display_contract=False, last=106.2, days_to_expiry=7,
    )
    display_only = FuturesContract(
        root_symbol="ZN", raw_symbol="ZNZ6", instrument_id=2, expiration=datetime(2026, 12, 21, tzinfo=UTC),
        is_front_month=False, is_display_contract=True, last=106.0, days_to_expiry=98,
    )
    row1 = mc.contract_row(front_only, as_of=datetime.now(UTC))
    row2 = mc.contract_row(display_only, as_of=datetime.now(UTC))
    assert row1["Status"] == "FRONT"
    assert row2["Status"] == "DISPLAY"


def test_contract_row_never_shows_zero_for_missing_open_interest():
    contract = FuturesContract(
        root_symbol="CL", raw_symbol="CLK7", instrument_id=3, expiration=datetime(2027, 4, 20, tzinfo=UTC),
        last=78.0, open_interest=None, volume=None,
    )
    row = mc.contract_row(contract, as_of=datetime.now(UTC))
    assert row["OI"] == "N/A"
    assert row["Volume"] == "N/A"


def test_contract_row_shows_real_open_interest_when_present():
    contract = FuturesContract(
        root_symbol="CL", raw_symbol="CLV6", instrument_id=4, expiration=datetime(2026, 9, 22, tzinfo=UTC),
        last=100.0, open_interest=175963.0, volume=262508.0,
    )
    row = mc.contract_row(contract, as_of=datetime.now(UTC))
    assert row["OI"] == "175,963"
    assert row["Volume"] == "262,508"


def test_term_structure_row_carries_price_source_provenance():
    point = TermStructurePoint(
        raw_symbol="CLV6", expiration=datetime(2026, 9, 22, tzinfo=UTC), price=100.05, price_source="settlement",
    )
    row = mc.term_structure_row(point)
    assert row["Source"] == "settlement"
    assert row["Price"] == "100.0500"
