"""Phase 03 -- contract-definition parsing + the Python contract registry."""
from __future__ import annotations

import pandas as pd
import pytest
from alpha_agent.data.definitions import (
    DefinitionRegistry,
    contracts_frame,
    parse_definition_frame,
    write_contracts_parquet,
)
from alpha_agent.schemas.market_data import CONTRACT_COLUMNS
from vendor_fixtures import vendor_definition_frame


def test_parse_definition_frame_basic():
    specs = parse_definition_frame(vendor_definition_frame())
    assert len(specs) == 1
    s = specs[0]
    assert s.raw_symbol == "NQM5"
    assert s.root_symbol == "NQ"
    assert s.tick_size == 0.25       # derived: quote_tick_size
    assert s.multiplier == 20.0      # derived: point value (NOT contract_multiplier sentinel)
    assert s.tick_size * s.multiplier == pytest.approx(5.0)  # USD tick value
    assert s.first_notice_ns is None


def test_parse_ignores_contract_multiplier_sentinel():
    # fixture default contract_multiplier is INT32_MAX; must not leak into multiplier
    s = parse_definition_frame(vendor_definition_frame())[0]
    assert s.multiplier == 20.0


def test_parse_fails_loudly_when_economics_cannot_be_derived():
    from alpha_agent.data.contract_economics import EconomicsError

    frame = vendor_definition_frame([{"unit_of_measure_qty": 4294967295}])  # sentinel
    with pytest.raises(EconomicsError):
        parse_definition_frame(frame)


def test_registry_lookups_and_membership():
    reg = DefinitionRegistry(parse_definition_frame(vendor_definition_frame()))
    assert len(reg) == 1
    assert 4021 in reg
    assert reg.is_tradable(4021)
    assert not reg.is_tradable(999)
    assert reg.require(4021).raw_symbol == "NQM5"
    assert reg.by_raw_symbol("NQM5").instrument_id == 4021
    with pytest.raises(KeyError):
        reg.require(999)


def test_root_must_be_determinable():
    frame = vendor_definition_frame([{"asset": "", "raw_symbol": "NQM5"}])
    with pytest.raises(ValueError, match="cannot determine root"):
        parse_definition_frame(frame)
    # explicit root_map rescues it
    specs = parse_definition_frame(frame, root_map={"NQM5": "NQ"})
    assert specs[0].root_symbol == "NQ"


def test_continuous_symbol_rejected_in_definitions():
    frame = vendor_definition_frame([{"raw_symbol": "NQ.v.0"}])
    with pytest.raises(ValueError):
        parse_definition_frame(frame)


def test_write_contracts_parquet_layout(tmp_path):
    specs = parse_definition_frame(
        vendor_definition_frame(
            [
                {"instrument_id": 1, "raw_symbol": "NQM5", "asset": "NQ"},
                {"instrument_id": 2, "raw_symbol": "ESM5", "asset": "ES"},
            ]
        )
    )
    paths = write_contracts_parquet(specs, tmp_path)
    assert {p.name for p in paths} == {"NQ.parquet", "ES.parquet"}
    nq = pd.read_parquet(tmp_path / "NQ.parquet")
    assert list(nq.columns) == list(CONTRACT_COLUMNS)
    assert list(contracts_frame(specs).columns) == list(CONTRACT_COLUMNS)
