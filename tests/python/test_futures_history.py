"""Phase 04 -- futures history orchestrator (layers 3/4/6 + lineage)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from alpha_agent.data.futures_history import build_futures_history
from alpha_agent.data.lineage import load_lineage
from alpha_agent.schemas.market_data import (
    BACKADJUSTED_BAR_COLUMNS,
    CONTINUOUS_BAR_COLUMNS,
    ROLL_EVENT_COLUMNS,
)
from futures_fixtures import NQU6, canonical_bars, registry, two_contract_history


def test_writes_three_layers_with_lineage(tmp_path):
    bars = two_contract_history(u6_close=29000.0, z6_open=29100.0)
    res = build_futures_history(
        bars, continuous_symbol="NQ.v.0", registry=registry(),
        processed_root=tmp_path,
    )
    assert res.paths["rolls"] == tmp_path / "rolls" / "NQ.v.0.parquet"
    assert res.paths["continuous"] == tmp_path / "continuous" / "NQ.v.0.parquet"
    assert res.paths["backadjusted"] == tmp_path / "backadjusted" / "NQ.v.0.parquet"

    assert list(pd.read_parquet(res.paths["rolls"]).columns) == list(ROLL_EVENT_COLUMNS)
    assert list(pd.read_parquet(res.paths["continuous"]).columns) == list(CONTINUOUS_BAR_COLUMNS)
    assert list(pd.read_parquet(res.paths["backadjusted"]).columns) == list(BACKADJUSTED_BAR_COLUMNS)

    lin = load_lineage(res.paths["backadjusted"])
    assert lin.artifact_kind == "backadjusted"
    assert lin.price_domain == "back_adjusted"
    assert lin.adjustment_method == "additive"
    assert lin.adjustment_mode == "retrospective_research"
    assert lin.roll_rule == "instrument_id_transition"
    assert lin.roll_price_policy == "same_timestamp_close_close"   # preferred (Phase 04.5)
    assert lin.extra["n_rolls_used_fallback"] == 1                 # no overlap data -> fallback
    assert lin.first_notice_policy["safety_buffer_business_days"] == 0
    assert lin.calendar_version
    assert lin.timezone == "UTC"

    cont_lin = load_lineage(res.paths["continuous"])
    assert cont_lin.price_domain == "raw_continuous"
    roll_lin = load_lineage(res.paths["rolls"])
    assert roll_lin.artifact_kind == "rolls"
    assert bool(pd.read_parquet(res.paths["rolls"])["used_fallback"].iloc[0]) is True


def test_single_contract_history_has_zero_rolls(tmp_path):
    bars = canonical_bars([{"spec": NQU6, "n": 20, "base_price": 100.0}])
    res = build_futures_history(bars, continuous_symbol="NQ.v.0", registry=registry(),
                                processed_root=tmp_path)
    assert res.rolls == []
    # back-adjusted == unadjusted when there are no rolls
    assert (res.back_adjusted["cumulative_adjustment"] == 0.0).all()
    assert list(res.continuous["active_raw_symbol"].unique()) == ["NQU6"]


def test_real_nqu6_replay_produces_a_clean_single_contract_history(tmp_path):
    dbn = Path("data/raw/databento/GLBX.MDP3/ohlcv-1m/ad665732c52dfea5/ohlcv-1m.dbn.zst")
    if not dbn.exists():
        pytest.skip("real NQU6 replay dataset not present")

    from alpha_agent.data import decode as dec
    from alpha_agent.data.calendars import default_calendar
    from alpha_agent.data.canonicalize import canonicalize
    from alpha_agent.data.definitions import DefinitionRegistry, parse_definition_frame

    def_dbn = Path("data/raw/databento/GLBX.MDP3/definition/24a980b7c26bfc89/definition.dbn.zst")
    bars_df = dec.ohlcv_df_to_vendor_frame(
        dec.load_dbn(dbn).to_df(price_type="fixed", pretty_ts=False, map_symbols=True).reset_index()
    )
    def_df = dec.load_dbn(def_dbn).to_df(pretty_ts=False, map_symbols=False).reset_index()
    seen = {int(i) for i in bars_df["instrument_id"].unique()}
    specs = parse_definition_frame(dec.definition_df_to_vendor_frame(def_df, keep_instrument_ids=seen))
    reg = DefinitionRegistry(specs)
    canonical, _ = canonicalize(bars_df, reg, default_calendar())

    res = build_futures_history(canonical, continuous_symbol="NQ.v.0", registry=reg,
                                processed_root=tmp_path)
    assert res.rolls == []                                   # one trading day, one contract
    assert set(res.continuous["active_raw_symbol"]) == {"NQU6"}
    assert len(res.continuous) == 1380
    assert (res.back_adjusted["cumulative_adjustment"] == 0.0).all()
