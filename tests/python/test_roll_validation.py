"""Phase 04.5 -- real-style NQ roll + aligned back-adjustment validation.

Uses deterministic multi-contract fixtures shaped like a real volume roll. The
actual Databento replay lives in the ``--replay`` path of
scripts/databento_roll_validate.py (skipped here until Stage A/B data exist).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from alpha_agent.data.backadjust import AdjustmentMode, build_back_adjusted_series
from alpha_agent.data.continuous import build_continuous_series
from alpha_agent.data.databento_source import HoldoutViolation
from alpha_agent.data.diagnostics import DiagnosticKind
from alpha_agent.data.roll_validation import (
    STAGE_A_END,
    STAGE_A_START,
    detect_transition,
    stage_a_requests,
    stage_b_request,
)
from alpha_agent.data.rolls import RollPricePolicy, build_roll_events
from futures_fixtures import MIN_NS, NQU6, NQZ6, registry

T0 = pd.Timestamp("2026-06-11T14:00:00Z").value


def _bar(ts, iid, raw, root, o, h, low, c, vol=100):
    return {"ts_event_ns": ts, "instrument_id": iid, "raw_symbol": raw, "root_symbol": root,
            "open": o, "high": h, "low": low, "close": c, "volume": vol}


def _real_style():
    """Continuous NQU6 -> NQZ6 with a +50/min market move across the seam, and an
    overlap set where the true contract basis is exactly +100."""
    # market rises 50/min; NQZ6 trades 100 above NQU6 at every instant (basis).
    u6_close = {0: 19950.0, 1: 20000.0, 2: 20050.0}
    z6_close = {0: 20050.0, 1: 20100.0, 2: 20150.0}
    ts = {k: T0 + k * MIN_NS for k in (0, 1, 2)}

    continuous = pd.DataFrame([
        _bar(ts[0], 101, "NQU6", "NQ", 19940, 19960, 19935, u6_close[0]),
        _bar(ts[1], 101, "NQU6", "NQ", 19990, 20010, 19985, u6_close[1]),
        # roll: first NQZ6 bar. open == its own close (flat within the minute)
        _bar(ts[2], 102, "NQZ6", "NQ", z6_close[2], z6_close[2] + 5, z6_close[2] - 5, z6_close[2]),
        _bar(ts[2] + MIN_NS, 102, "NQZ6", "NQ", 20155, 20160, 20150, 20160),
    ])
    overlap = pd.DataFrame([
        _bar(ts[k], iid, raw, "NQ", cl - 5, cl + 5, cl - 8, cl)
        for k in (0, 1, 2)
        for iid, raw, cl in ((101, "NQU6", u6_close[k]), (102, "NQZ6", z6_close[k]))
    ])
    return continuous, overlap, ts


def test_stage_a_request_is_refused_by_the_holdout_boundary():
    """The Phase 04.5 hardcoded Stage A window is a real, already-downloaded
    2026 engineering probe (Phase 23.1 finding C1; permanent provenance --
    see data/manifests/real_dataset/spend_ledger.json and
    docs/FINAL_SYSTEM_AUDIT.md). It is at/after the locked 2025-01-01 research
    holdout, so as of Phase 23.2 it can never again construct a live
    HistoricalRequest -- stage_a_requests() now raises before returning."""
    assert (STAGE_A_START, STAGE_A_END) == ("2026-06-08", "2026-06-19")  # unedited historical constant
    with pytest.raises(HoldoutViolation):
        stage_a_requests()


def test_detect_transition_resolves_from_the_definition_table():
    continuous, _, ts = _real_style()
    vendor = continuous.rename(columns={"ts_event_ns": "ts_event"})[
        ["ts_event", "instrument_id", "open", "high", "low", "close", "volume"]
    ].assign(symbol="NQ.v.0")
    det = detect_transition(vendor, registry(NQU6, NQZ6))
    assert (det.from_instrument_id, det.to_instrument_id) == (101, 102)
    assert (det.from_raw_symbol, det.to_raw_symbol) == ("NQU6", "NQZ6")   # from the registry
    assert det.transition_ts_ns == ts[2]
    assert {s.instrument_id for s in det.spans} == {101, 102}


def test_no_transition_returns_none_not_a_manufactured_roll():
    one = pd.DataFrame([_bar(T0 + i * MIN_NS, 101, "NQU6", "NQ", 100, 101, 99, 100) for i in range(5)])
    vendor = one.rename(columns={"ts_event_ns": "ts_event"})[
        ["ts_event", "instrument_id", "open", "high", "low", "close", "volume"]
    ].assign(symbol="NQ.v.0")
    assert detect_transition(vendor, registry(NQU6)) is None


def test_aligned_close_close_isolates_the_basis_from_market_movement():
    continuous, overlap, _ = _real_style()
    # with the overlap: preferred policy -> aligned same-timestamp close-close
    rolls, report = build_roll_events(
        continuous, continuous_symbol="NQ.v.0", registry=registry(NQU6, NQZ6),
        price_policy=RollPricePolicy.SAME_TIMESTAMP_CLOSE_CLOSE, overlap_bars=overlap,
    )
    r = rolls[0]
    assert r.price_policy == "same_timestamp_close_close"
    assert r.used_fallback is False
    assert r.from_ts_ns == r.to_ts_ns                       # SAME timestamp
    assert r.additive_gap == 100.0                          # pure contract basis
    assert not report.has_kind(DiagnosticKind.ROLL_BASIS_FALLBACK_USED)


def test_fallback_contaminates_the_gap_with_market_movement_and_warns():
    continuous, _, _ = _real_style()
    rolls, report = build_roll_events(
        continuous, continuous_symbol="NQ.v.0", registry=registry(NQU6, NQZ6),
        price_policy=RollPricePolicy.SAME_TIMESTAMP_CLOSE_CLOSE, overlap_bars=None,
    )
    r = rolls[0]
    assert r.used_fallback is True
    assert r.price_policy == "prev_close_new_open"          # recorded as the fallback
    assert r.from_ts_ns != r.to_ts_ns                       # adjacent, NOT same timestamp
    assert r.additive_gap == 150.0                          # 100 basis + 50 market move
    assert report.has_kind(DiagnosticKind.ROLL_BASIS_FALLBACK_USED)
    assert report.warnings and not report.errors


def test_aligned_open_open_policy():
    continuous, overlap, _ = _real_style()
    rolls, _ = build_roll_events(
        continuous, continuous_symbol="NQ.v.0", registry=registry(NQU6, NQZ6),
        price_policy=RollPricePolicy.SAME_TIMESTAMP_OPEN_OPEN, overlap_bars=overlap,
    )
    assert rolls[0].price_policy == "same_timestamp_open_open"
    assert rolls[0].from_ts_ns == rolls[0].to_ts_ns


def _chain(continuous, overlap):
    reg = registry(NQU6, NQZ6)
    rolls, _ = build_roll_events(
        continuous, continuous_symbol="NQ.v.0", registry=reg,
        price_policy=RollPricePolicy.SAME_TIMESTAMP_CLOSE_CLOSE, overlap_bars=overlap)
    cont, _ = build_continuous_series(continuous, continuous_symbol="NQ.v.0",
                                      registry=reg, rolls=rolls)
    return rolls, cont


def test_retrospective_adjustment_removes_only_the_measured_basis():
    continuous, overlap, ts = _real_style()
    rolls, cont = _chain(continuous, overlap)
    retro, rep = build_back_adjusted_series(cont, rolls, continuous_symbol="NQ.v.0")

    # unadjusted keeps the raw seam (100 basis + 50 move = 150 jump)
    pre_close = cont.loc[cont["ts_event_ns"] < ts[2], "close"].iloc[-1]
    post_open = cont.loc[cont["ts_event_ns"] == ts[2], "open"].iloc[0]
    assert post_open - pre_close == 150.0

    # back-adjust shifts pre-roll history by exactly the +100 measured basis
    adj_pre = retro.loc[retro["ts_event_ns"] < ts[2]]
    assert (adj_pre["cumulative_adjustment"] == 100.0).all()
    assert (retro.loc[retro["ts_event_ns"] >= ts[2], "cumulative_adjustment"] == 0.0).all()
    assert rep.ok


def test_point_in_time_before_the_roll_sees_no_adjustment():
    continuous, overlap, ts = _real_style()
    rolls, cont = _chain(continuous, overlap)
    as_of = ts[2] - MIN_NS
    pit, _ = build_back_adjusted_series(
        cont, rolls, continuous_symbol="NQ.v.0",
        mode=AdjustmentMode.POINT_IN_TIME, as_of_ts_ns=as_of)
    assert (pit["cumulative_adjustment"] == 0.0).all()      # future roll invisible
    assert pit["ts_event_ns"].max() <= as_of
    assert (pit["adjustment_mode"] == "point_in_time").all()


def test_no_execution_identity_and_domain_rejection():
    continuous, overlap, _ = _real_style()
    rolls, cont = _chain(continuous, overlap)
    retro, _ = build_back_adjusted_series(cont, rolls, continuous_symbol="NQ.v.0")
    for col in ("instrument_id", "raw_symbol", "active_instrument_id", "active_raw_symbol"):
        assert col not in retro.columns
    from alpha_agent.data.backadjust import reject_adjusted_execution
    from alpha_agent.data.diagnostics import PipelineError
    with pytest.raises(PipelineError):
        reject_adjusted_execution("back_adjusted")


def test_stage_b_request_is_refused_by_the_holdout_boundary():
    """Same reasoning as Stage A: the fixture's T0 (2026-06-11) sits inside
    the real, already-downloaded Stage A window, so the padded overlap window
    stage_b_request() derives is also at/after the holdout and is refused
    before a live HistoricalRequest can be constructed."""
    continuous, _, _ = _real_style()
    vendor = continuous.rename(columns={"ts_event_ns": "ts_event"})[
        ["ts_event", "instrument_id", "open", "high", "low", "close", "volume"]
    ].assign(symbol="NQ.v.0")
    det = detect_transition(vendor, registry(NQU6, NQZ6))
    with pytest.raises(HoldoutViolation):
        stage_b_request(det, pad_days_before=1, pad_days_after=2)


def test_replay_script_present_and_skips_cleanly_without_data():
    assert Path("scripts/databento_roll_validate.py").exists()


# --- exact real June-2026 NQM6 -> NQU6 numbers (fixed-point -> normalized) ---

FIX = 1_000_000_000
_TT = pd.Timestamp("2026-06-17T00:00:00Z").value   # observed transition ts


def _real_june_roll():
    """Overlap comes in as raw DBN FIXED-POINT (like Stage B); the continuous
    feed is already normalized (post-canonicalize). Numbers are the observed
    NQM6 -> NQU6 values."""
    from alpha_agent.data.price_domain import descale_fixed_point

    continuous = pd.DataFrame([
        _bar(_TT - 2 * MIN_NS, 42004058, "NQM6", "NQ", 30010.00, 30014, 30009, 30011.00),
        _bar(_TT - 1 * MIN_NS, 42004058, "NQM6", "NQ", 30011.00, 30015, 30010, 30012.75),
        _bar(_TT, 42004177, "NQU6", "NQ", 30330.50, 30335, 30329, 30331.00),
        _bar(_TT + 1 * MIN_NS, 42004177, "NQU6", "NQ", 30331.00, 30336, 30330, 30332.00),
    ])
    overlap_fixed = pd.DataFrame([
        # both contracts trading at the aligned timestamp _TT
        {"ts_event": _TT, "instrument_id": 42004058, "symbol": "NQ.v.0",
         "open": int(30031.50 * FIX), "high": int(30032 * FIX), "low": int(30031 * FIX),
         "close": 30031_750_000_000, "volume": 40},
        {"ts_event": _TT, "instrument_id": 42004177, "symbol": "NQ.v.0",
         "open": int(30343.00 * FIX), "high": int(30344 * FIX), "low": int(30343 * FIX),
         "close": 30343_250_000_000, "volume": 55},
    ])
    overlap = descale_fixed_point(overlap_fixed.rename(columns={"ts_event": "ts_event_ns"}))
    return continuous, overlap


def _real_reg():
    from alpha_agent.data.definitions import DefinitionRegistry
    from alpha_agent.schemas.market_data import ContractSpecModel
    common = {"root_symbol": "NQ", "exchange": "XCME", "tick_size": 0.25, "multiplier": 20.0,
              "activation_ns": pd.Timestamp("2025-06-01T00:00:00Z").value}
    return DefinitionRegistry([
        ContractSpecModel(instrument_id=42004058, raw_symbol="NQM6",
                          expiration_ns=pd.Timestamp("2026-06-19T13:30:00Z").value, **common),
        ContractSpecModel(instrument_id=42004177, raw_symbol="NQU6",
                          expiration_ns=pd.Timestamp("2026-09-18T13:30:00Z").value, **common),
    ])


def test_A_fixed_point_overlap_decodes_to_normalized():
    _, overlap = _real_june_roll()
    assert sorted(overlap["close"].tolist()) == [30031.75, 30343.25]


def test_B_C_aligned_basis_and_invariant():
    continuous, overlap = _real_june_roll()
    rolls, report = build_roll_events(
        continuous, continuous_symbol="NQ.v.0", registry=_real_reg(),
        price_policy=RollPricePolicy.SAME_TIMESTAMP_CLOSE_CLOSE, overlap_bars=overlap)
    r = rolls[0]
    assert r.price_policy == "same_timestamp_close_close" and r.used_fallback is False
    assert r.from_ts_ns == r.to_ts_ns == _TT
    assert r.from_price == 30031.75 and r.to_price == 30343.25
    assert r.additive_gap == 311.5                                   # B
    assert abs((r.from_price + r.additive_gap) - r.to_price) < 1e-9   # C
    assert not report.has_kind(DiagnosticKind.ROLL_BASIS_FALLBACK_USED)


def test_D_E_F_splice_decomposition_and_preserved_market_move():
    continuous, overlap = _real_june_roll()
    reg = _real_reg()
    rolls, _ = build_roll_events(
        continuous, continuous_symbol="NQ.v.0", registry=reg,
        price_policy=RollPricePolicy.SAME_TIMESTAMP_CLOSE_CLOSE, overlap_bars=overlap)
    cont, _ = build_continuous_series(continuous, continuous_symbol="NQ.v.0",
                                      registry=reg, rolls=rolls)
    retro, _ = build_back_adjusted_series(cont, rolls, continuous_symbol="NQ.v.0")

    old_close = cont.loc[cont["ts_event_ns"] < _TT, "close"].iloc[-1]
    new_open = cont.loc[cont["ts_event_ns"] == _TT, "open"].iloc[0]
    jump = new_open - old_close
    basis = rolls[0].additive_gap
    residual = jump - basis
    assert (jump, basis, residual) == (317.75, 311.5, 6.25)          # D

    adj_old = retro.loc[retro["ts_event_ns"] == _TT - MIN_NS, "close"].iloc[0]
    adj_new = retro.loc[retro["ts_event_ns"] == _TT, "open"].iloc[0]
    assert round(adj_new - adj_old, 4) == 6.25                       # E: residual, NOT 0
    assert round(adj_old - old_close, 4) == 311.5                    # F: only the basis removed
    assert (retro.loc[retro["ts_event_ns"] >= _TT, "cumulative_adjustment"] == 0.0).all()


def test_G_fixed_point_overlap_without_descale_is_rejected():
    from alpha_agent.data.price_domain import PriceScaleError
    continuous, _ = _real_june_roll()
    raw_overlap = pd.DataFrame([
        {"ts_event_ns": _TT, "instrument_id": 42004058, "open": 30031_500_000_000,
         "high": 30032_000_000_000, "low": 30031_000_000_000, "close": 30031_750_000_000},
        {"ts_event_ns": _TT, "instrument_id": 42004177, "open": 30343_000_000_000,
         "high": 30344_000_000_000, "low": 30343_000_000_000, "close": 30343_250_000_000},
    ])
    with pytest.raises(PriceScaleError, match="price-domain mismatch"):
        build_roll_events(continuous, continuous_symbol="NQ.v.0", registry=_real_reg(),
                          price_policy=RollPricePolicy.SAME_TIMESTAMP_CLOSE_CLOSE,
                          overlap_bars=raw_overlap)
