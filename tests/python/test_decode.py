"""Phase 03.5 -- real-DBN column mapping (no DBN file needed: synthetic frames
carry the real Databento column names)."""
from __future__ import annotations

import pandas as pd
import pytest
from alpha_agent.data import decode as dec
from alpha_agent.schemas.market_data import VENDOR_DEFINITION_COLUMNS, VENDOR_OHLCV_COLUMNS

_NS = 1_600_000_000_000_000_000


def _real_ohlcv_df() -> pd.DataFrame:
    # what DBNStore.to_df(price_type="fixed", pretty_ts=False, map_symbols=True)
    # yields for ohlcv-1m, with ts_event as a column after reset_index()
    return pd.DataFrame(
        {
            "ts_event": [_NS, _NS + 60_000_000_000],
            "rtype": [34, 34],
            "publisher_id": [1, 1],
            "instrument_id": [42, 42],
            "open": [20000_000_000_000, 20001_000_000_000],
            "high": [20005_000_000_000, 20006_000_000_000],
            "low": [19999_000_000_000, 20000_000_000_000],
            "close": [20002_000_000_000, 20003_000_000_000],
            "volume": [1200, 1350],
            "symbol": ["NQZ5", "NQZ5"],
        }
    )


def _real_def_df(**over) -> pd.DataFrame:
    # real GLBX.MDP3 InstrumentDefMsg to_df() shape (float price_type)
    base = {
        "ts_event": [_NS],
        "instrument_id": [42],
        "raw_symbol": ["NQZ5"],
        "asset": ["NQ"],
        "exchange": ["XCME"],
        "instrument_class": ["F"],
        "min_price_increment": [0.25],
        "min_price_increment_amount": [0.05],
        "display_factor": [0.01],
        "unit_of_measure": ["IPNT"],
        "unit_of_measure_qty": [20.0],
        "main_fraction": [255],                 # uint8 sentinel
        "contract_multiplier": [2147483647],    # int32 sentinel
        "activation": [_NS - 10**15],
        "expiration": [_NS + 10**15],
        "price_ratio": [0.0],
        "contract_multiplier_unit": [127],
    }
    base.update({k: [v] for k, v in over.items()})
    return pd.DataFrame(base)


def test_ohlcv_mapping_keeps_fixed_point_and_ns():
    out = dec.ohlcv_df_to_vendor_frame(_real_ohlcv_df())
    assert list(out.columns) == list(VENDOR_OHLCV_COLUMNS)
    assert out["ts_event"].iloc[0] == _NS
    assert out["open"].iloc[0] == 20000_000_000_000
    assert out["volume"].dtype == "int64"


def test_ohlcv_mapping_symbol_is_optional_and_a_label():
    # a real continuous response labels rows "NQ.v.0"; with stype_out=instrument_id
    # there may be no symbol column at all -- both are fine.
    with_label = _real_ohlcv_df()
    with_label["symbol"] = "NQ.v.0"
    assert dec.ohlcv_df_to_vendor_frame(with_label)["symbol"].iloc[0] == "NQ.v.0"

    no_symbol = _real_ohlcv_df().drop(columns=["symbol"])
    out = dec.ohlcv_df_to_vendor_frame(no_symbol)
    assert (out["symbol"] == "").all()


def test_ohlcv_mapping_missing_required_column_raises():
    df = _real_ohlcv_df().drop(columns=["close"])
    with pytest.raises(dec.SchemaMismatch):
        dec.ohlcv_df_to_vendor_frame(df)


def test_definition_frame_passes_raw_fields_through():
    out = dec.definition_df_to_vendor_frame(_real_def_df(), keep_instrument_ids={42})
    assert list(out.columns) == list(VENDOR_DEFINITION_COLUMNS)
    row = out.iloc[0]
    assert row["min_price_increment"] == 0.25
    assert row["min_price_increment_amount"] == 0.05
    assert row["display_factor"] == 0.01
    assert row["unit_of_measure_qty"] == 20.0
    assert row["main_fraction"] == ""              # 255 sentinel -> unset
    assert row["first_notice"] == "" and row["last_trade"] == ""


def test_definition_frame_then_parse_derives_real_nq_economics():
    from alpha_agent.data.definitions import parse_definition_frame

    vendor = dec.definition_df_to_vendor_frame(_real_def_df(), keep_instrument_ids={42})
    spec = parse_definition_frame(vendor)[0]
    assert spec.tick_size == 0.25
    assert spec.multiplier == 20.0
    assert spec.tick_size * spec.multiplier == pytest.approx(5.0)


def test_definition_join_is_on_instrument_id_and_drops_spreads():
    spread = _real_def_df(instrument_id=99, raw_symbol="NQZ5-NQH6", instrument_class="S")
    other = _real_def_df(instrument_id=43, raw_symbol="NQH5")
    combined = pd.concat([_real_def_df(), other, spread], ignore_index=True)
    out = dec.definition_df_to_vendor_frame(combined, keep_instrument_ids={42})
    assert list(out["instrument_id"]) == [42]
    assert list(out["raw_symbol"]) == ["NQZ5"]
