"""Phase 02.5 -- the frozen futures data / C++ boundary contract (Python side)."""
from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from alpha_agent.adapters.cpp_cli import write_boundary_bundle
from alpha_agent.schemas.market_data import (
    BOUNDARY_BAR_COLUMNS,
    CONTRACT_COLUMNS,
    BackAdjustedBar,
    CanonicalBar,
    ContinuousBar,
    ContractSpecModel,
    PriceDomain,
    RollEvent,
    Session,
    is_continuous_symbol,
    is_tradable_contract_symbol,
)

_NS = 1_600_000_000_000_000_000


def test_symbol_classification():
    for sym in ("NQ.v.0", "ES.c.1", "CL.n.0", "6E.v.0"):
        assert is_continuous_symbol(sym)
        assert not is_tradable_contract_symbol(sym)
    for sym in ("NQZ6", "ESH25", "GCJ7"):
        assert not is_continuous_symbol(sym)
        assert is_tradable_contract_symbol(sym)
    # parent + empty
    assert not is_tradable_contract_symbol("NQ.FUT")
    assert not is_tradable_contract_symbol("")


def test_boundary_column_contracts_are_frozen():
    assert BOUNDARY_BAR_COLUMNS == (
        "ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume",
    )
    assert CONTRACT_COLUMNS == (
        "instrument_id", "raw_symbol", "root_symbol", "exchange",
        "tick_size", "multiplier", "activation_ns", "expiration_ns",
        "first_notice_ns", "last_trade_ns",
    )


def _contract(**over) -> dict:
    base = {
        "instrument_id": 101, "raw_symbol": "NQZ6", "root_symbol": "NQ", "exchange": "XCME",
        "tick_size": 0.25, "multiplier": 20.0, "activation_ns": _NS,
        "expiration_ns": _NS + 10**15,
    }
    base.update(over)
    return base


def test_contract_spec_model():
    ContractSpecModel(**_contract())
    with pytest.raises(ValueError):
        ContractSpecModel(**_contract(raw_symbol="NQ.v.0"))
    with pytest.raises(ValueError):
        ContractSpecModel(**_contract(tick_size=0.0))
    with pytest.raises(ValueError):
        ContractSpecModel(**_contract(activation_ns=_NS + 10**15, expiration_ns=_NS))


def _bar(**over) -> dict:
    base = {
        "ts_event_ns": _NS, "instrument_id": 101, "raw_symbol": "NQZ6", "root_symbol": "NQ",
        "open": 20000.0, "high": 20010.0, "low": 19990.0, "close": 20005.0, "volume": 1234,
        "trading_day": date(2020, 9, 14), "session": Session.RTH,
    }
    base.update(over)
    return base


def test_canonical_bar():
    CanonicalBar(**_bar())
    with pytest.raises(ValueError):
        CanonicalBar(**_bar(raw_symbol="NQ.v.0"))
    with pytest.raises(ValueError):
        CanonicalBar(**_bar(price_domain=PriceDomain.BACK_ADJUSTED))
    with pytest.raises(ValueError):
        CanonicalBar(**_bar(high=19999.0))  # high below close

    # signed prices are valid (historical negative CL): open -10, high -5,
    # low -25, close -20 satisfies low <= open/close <= high.
    CanonicalBar(**_bar(raw_symbol="CLK0", root_symbol="CL",
                        open=-10.0, high=-5.0, low=-25.0, close=-20.0))
    CanonicalBar(**_bar(open=5.0, high=6.0, low=-6.0, close=-5.0))  # straddles zero
    # ... but non-finite / un-normalized fixed-point still fail
    with pytest.raises(ValueError):
        CanonicalBar(**_bar(open=float("nan")))
    with pytest.raises(ValueError):
        CanonicalBar(**_bar(close=float("inf")))
    with pytest.raises(ValueError):
        CanonicalBar(**_bar(open=2.0e13, high=2.1e13, low=1.9e13, close=2.0e13))


def test_continuous_and_back_adjusted_bars():
    ContinuousBar(
        ts_event_ns=_NS, continuous_symbol="NQ.v.0", active_instrument_id=101,
        active_raw_symbol="NQZ6", open=1.0, high=2.0, low=0.5, close=1.5, volume=10,
    )
    with pytest.raises(ValueError):
        ContinuousBar(
            ts_event_ns=_NS, continuous_symbol="NQZ6", active_instrument_id=101,
            active_raw_symbol="NQZ6", open=1.0, high=2.0, low=0.5, close=1.5, volume=10,
        )
    BackAdjustedBar(
        ts_event_ns=_NS, continuous_symbol="NQ.v.0", open=-5.0, high=1.0, low=-9.0,
        close=0.0, volume=10, adjusted_through_ts_ns=_NS,
    )
    with pytest.raises(ValueError):
        BackAdjustedBar(
            ts_event_ns=_NS, continuous_symbol="NQ.v.0", open=1.0, high=1.0, low=1.0,
            close=1.0, volume=1, adjusted_through_ts_ns=_NS,
            price_domain=PriceDomain.RAW_CONTRACT,
        )


def test_roll_event_requires_contract_change():
    RollEvent(
        continuous_symbol="NQ.v.0", effective_ts_ns=_NS, from_instrument_id=101,
        to_instrument_id=102, from_raw_symbol="NQZ6", to_raw_symbol="NQH6",
    )
    with pytest.raises(ValueError):
        RollEvent(
            continuous_symbol="NQ.v.0", effective_ts_ns=_NS, from_instrument_id=101,
            to_instrument_id=101, from_raw_symbol="NQZ6", to_raw_symbol="NQZ6",
        )


def test_write_boundary_bundle(tmp_path):
    bars = pd.DataFrame({
        "ts_event_ns": [1, 2, 3], "instrument_id": [101, 101, 102],
        "open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0], "low": [1.0, 2.0, 3.0],
        "close": [1.0, 2.0, 3.0], "volume": [1, 1, 1],
    })
    contracts = pd.DataFrame([
        _contract(instrument_id=101, raw_symbol="NQZ6"),
        _contract(instrument_id=102, raw_symbol="NQH6"),
    ])
    contracts["first_notice_ns"] = ""
    contracts["last_trade_ns"] = ""

    bars_path, contracts_path = write_boundary_bundle(bars, contracts, tmp_path)
    assert list(pd.read_csv(bars_path).columns) == list(BOUNDARY_BAR_COLUMNS)
    assert list(pd.read_csv(contracts_path).columns) == list(CONTRACT_COLUMNS)

    # A bar with no matching contract row is refused.
    bad_bars = bars.assign(instrument_id=[101, 101, 999])
    with pytest.raises(ValueError, match="no contract row"):
        write_boundary_bundle(bad_bars, contracts, tmp_path)


def test_hybrid_bundle_resolves_in_cpp(tmp_path):
    """End-to-end: the frozen bundle is accepted by the C++ CLI and every bar
    resolves to a real contract."""
    exe = Path("build/cpp/cpp/quant_backtest_csv")
    if not exe.exists():
        pytest.skip("C++ core not built")

    n = 60
    bars = pd.DataFrame({
        "ts_event_ns": [(i + 1) * 60_000_000_000 for i in range(n)],
        "instrument_id": [2001] * n,
        "open": [100.0 + i for i in range(n)],
        "high": [100.5 + i for i in range(n)],
        "low": [99.5 + i for i in range(n)],
        "close": [100.0 + i for i in range(n)],
        "volume": [10] * n,
    })
    contracts = pd.DataFrame([{
        "instrument_id": 2001, "raw_symbol": "NQH6", "root_symbol": "NQ",
        "exchange": "XCME", "tick_size": 0.25, "multiplier": 20.0,
        "activation_ns": 1, "expiration_ns": bars["ts_event_ns"].iloc[-1] + 10**12,
        "first_notice_ns": "", "last_trade_ns": "",
    }])
    bars_path, contracts_path = write_boundary_bundle(bars, contracts, tmp_path)
    proc = subprocess.run(
        [str(exe), str(bars_path), str(contracts_path), "5", "0.0"],
        check=True, capture_output=True, text=True,
    )
    out = json.loads(proc.stdout)
    assert out["bars"] == n
    assert out["contracts_resolved"] == 1

    # Phase 08 portfolio view is surfaced additively on the frozen CLI JSON.
    for key in (
        "starting_capital_usd", "cash_usd", "equity_usd",
        "gross_exposure_usd", "net_exposure_usd", "gross_leverage",
        "initial_margin_usd", "margin_utilization_pct", "margin_complete",
        "valuation_complete",
        "peak_equity_usd", "portfolio_drawdown_usd", "portfolio_drawdown_pct",
        "risk_rejects", "risk_resizes",
    ):
        assert key in out, key
    # reporting-only path: PassThrough risk never rejects or resizes
    assert out["risk_rejects"] == 0
    assert out["risk_resizes"] == 0
    # equity == cash + unrealized; force-liquidate default -> flat, so equity == cash
    assert out["equity_usd"] == out["cash_usd"]
    assert out["starting_capital_usd"] == 100000


def test_negative_cl_price_through_cli(tmp_path):
    """Phase 08.2: a CL bundle whose prices move through zero into negatives is
    accepted end-to-end by the C++ CLI -- no layer rejects a bar for price <= 0."""
    exe = Path("build/cpp/cpp/quant_backtest_csv")
    if not exe.exists():
        pytest.skip("C++ core not built")

    opens = [5.0, 1.0, 0.0, -5.0, -20.0, -10.0, -10.0, -8.0]
    n = len(opens)
    bars = pd.DataFrame({
        "ts_event_ns": [(i + 1) * 60_000_000_000 for i in range(n)],
        "instrument_id": [3001] * n,
        "open": opens,
        "high": [o + 1.0 for o in opens],
        "low": [o - 1.0 for o in opens],
        "close": opens,
        "volume": [50] * n,
    })
    contracts = pd.DataFrame([{
        "instrument_id": 3001, "raw_symbol": "CLK0", "root_symbol": "CL",
        "exchange": "XNYM", "tick_size": 0.01, "multiplier": 1000.0,
        "activation_ns": 1, "expiration_ns": bars["ts_event_ns"].iloc[-1] + 10**12,
        "first_notice_ns": "", "last_trade_ns": "",
    }])
    bars_path, contracts_path = write_boundary_bundle(bars, contracts, tmp_path)
    proc = subprocess.run(
        [str(exe), str(bars_path), str(contracts_path), "3", "0.0"],
        check=True, capture_output=True, text=True,
    )
    out = json.loads(proc.stdout)
    assert out["bars"] == n
    assert out["contracts_resolved"] == 1
    assert out["fills"] >= 1                 # the momentum strategy traded across zero
    # official realized PnL is a finite number derived from Fills only
    assert isinstance(out["net_pnl_usd"], (int, float))
