"""Phase 04 -- unadjusted continuous series (Layer 3)."""
from __future__ import annotations

import pandas as pd
import pytest
from alpha_agent.data.continuous import build_continuous_series
from alpha_agent.data.diagnostics import PipelineError
from alpha_agent.data.rolls import build_roll_events
from alpha_agent.schemas.market_data import CONTINUOUS_BAR_COLUMNS, PriceDomain
from futures_fixtures import NQU6, NQZ6, canonical_bars, registry, two_contract_history


def test_instrument_id_transition_resolves_two_contracts():
    bars = two_contract_history()
    rolls, _ = build_roll_events(bars, continuous_symbol="NQ.v.0", registry=registry())
    cont, _report = build_continuous_series(
        bars, continuous_symbol="NQ.v.0", registry=registry(), rolls=rolls,
    )
    assert list(cont.columns) == list(CONTINUOUS_BAR_COLUMNS)
    mapping = dict(zip(cont["active_instrument_id"], cont["active_raw_symbol"]))
    assert mapping == {101: "NQU6", 102: "NQZ6"}
    assert cont["continuous_symbol"].eq("NQ.v.0").all()
    # NQ.v.0 is never a tradable contract -- it only appears as the label
    assert "NQ.v.0" not in set(cont["active_raw_symbol"])
    assert int(cont["is_roll_boundary"].sum()) == 1


def test_unadjusted_series_preserves_the_raw_roll_gap():
    bars = two_contract_history(u6_close=29000.0, z6_open=29100.0)
    cont, _ = build_continuous_series(bars, continuous_symbol="NQ.v.0", registry=registry())
    last_u6 = cont[cont["active_raw_symbol"] == "NQU6"]["close"].iloc[-1]
    first_z6 = cont[cont["active_raw_symbol"] == "NQZ6"]["open"].iloc[0]
    assert last_u6 == 29000.0
    assert first_z6 == 29100.0
    assert first_z6 - last_u6 == 100.0          # the jump is NOT smoothed


def test_continuous_bar_domain_is_raw_continuous():
    from alpha_agent.schemas.market_data import ContinuousBar

    b = ContinuousBar(
        ts_event_ns=1, continuous_symbol="NQ.v.0", active_instrument_id=101,
        active_raw_symbol="NQU6", open=1.0, high=2.0, low=0.5, close=1.5, volume=1,
    )
    assert b.price_domain is PriceDomain.RAW_CONTINUOUS


def test_missing_active_contract_is_fatal():
    bars = two_contract_history()
    with pytest.raises(PipelineError, match="unknown active contract"):
        build_continuous_series(bars, continuous_symbol="NQ.v.0", registry=registry(NQU6))


def test_overlapping_active_contracts_is_fatal():
    bars = canonical_bars([{"spec": NQU6, "n": 2, "base_price": 100.0}])
    dup = bars.iloc[[0]].copy()
    dup["instrument_id"] = NQZ6.instrument_id
    dup["raw_symbol"] = "NQZ6"
    clash = pd.concat([bars, dup], ignore_index=True)
    with pytest.raises(PipelineError, match="overlapping"):
        build_continuous_series(clash, continuous_symbol="NQ.v.0", registry=registry())
