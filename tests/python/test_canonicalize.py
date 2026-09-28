"""Phase 03 -- canonical normalizer QA rules."""
from __future__ import annotations

import pandas as pd
import pytest
from alpha_agent.data.calendars import default_calendar
from alpha_agent.data.canonicalize import NormalizationPolicy, canonicalize
from alpha_agent.data.definitions import DefinitionRegistry, parse_definition_frame
from alpha_agent.data.diagnostics import DiagnosticKind, PipelineError
from alpha_agent.schemas.market_data import CANONICAL_BAR_COLUMNS, UNDEF_PRICE
from vendor_fixtures import (
    MIN_NS,
    SESSION_START_NS,
    to_fixed,
    vendor_definition_frame,
    vendor_ohlcv_frame,
)


def _registry():
    return DefinitionRegistry(parse_definition_frame(vendor_definition_frame()))


def _run(vendor_df, *, policy=None):
    return canonicalize(vendor_df, _registry(), default_calendar(), policy=policy)


def test_clean_frame_normalizes():
    canonical, report = _run(vendor_ohlcv_frame(n_bars=30))
    assert list(canonical.columns) == list(CANONICAL_BAR_COLUMNS)
    assert len(canonical) == 30
    assert report.n_rejected_rows == 0
    assert report.ok
    # ts preserved exactly, prices de-scaled, symbol from the registry
    assert canonical["ts_event_ns"].iloc[0] == SESSION_START_NS
    assert canonical["open"].iloc[0] == pytest.approx(20000.0)
    assert (canonical["raw_symbol"] == "NQM5").all()
    assert (canonical["root_symbol"] == "NQ").all()
    assert (canonical["session"] == "RTH").all()


def test_duplicate_bar_is_fatal():
    df = vendor_ohlcv_frame(n_bars=5)
    df = pd.concat([df, df.iloc[[2]]], ignore_index=True)
    with pytest.raises(PipelineError) as ei:
        _run(df)
    assert ei.value.report.has_kind(DiagnosticKind.DUPLICATE_BAR)


def test_non_monotonic_fails_by_default_or_sorts_with_diagnostic():
    df = vendor_ohlcv_frame(n_bars=5)
    df = df.iloc[[0, 1, 3, 2, 4]].reset_index(drop=True)
    with pytest.raises(PipelineError):
        _run(df)
    canonical, report = _run(df, policy=NormalizationPolicy(on_non_monotonic="sort"))
    assert report.has_kind(DiagnosticKind.NON_MONOTONIC_TIMESTAMP)
    assert report.has_kind(DiagnosticKind.TIMESTAMPS_REORDERED)
    assert canonical["ts_event_ns"].is_monotonic_increasing


def test_invalid_ohlc_row_rejected():
    bad = {"ts_event": SESSION_START_NS + 100 * MIN_NS, "high": to_fixed(1.0), "low": to_fixed(9.0),
           "open": to_fixed(5.0), "close": to_fixed(5.0)}
    canonical, report = _run(vendor_ohlcv_frame(n_bars=5, rows_override=[bad]))
    assert len(canonical) == 5
    assert report.n_rejected_rows == 1
    assert report.has_kind(DiagnosticKind.INVALID_OHLC)


def test_implausible_normalized_price_rejected():
    # An un-normalized fixed-point magnitude (someone skipped the descale) is the
    # real failure mode -- not a non-positive sign.
    bad = {"ts_event": SESSION_START_NS + 100 * MIN_NS, "open": to_fixed(2.0e9),
           "high": to_fixed(2.1e9), "low": to_fixed(1.9e9), "close": to_fixed(2.0e9)}
    canonical, report = _run(vendor_ohlcv_frame(n_bars=5, rows_override=[bad]))
    assert report.n_rejected_rows == 1
    assert report.has_kind(DiagnosticKind.IMPLAUSIBLE_NORMALIZED_PRICE)
    assert len(canonical) == 5


def test_zero_and_negative_normalized_prices_are_kept():
    # WTI crude traded through zero into negatives on 2020-04-20. A zero or
    # negative normalized price is valid data, not a QA failure.
    rows = [
        {"ts_event": SESSION_START_NS + 100 * MIN_NS, "open": to_fixed(0.0),
         "high": to_fixed(0.0), "low": to_fixed(-5.0), "close": to_fixed(-2.0)},
        {"ts_event": SESSION_START_NS + 101 * MIN_NS, "open": to_fixed(-10.0),
         "high": to_fixed(-5.0), "low": to_fixed(-25.0), "close": to_fixed(-20.0)},
    ]
    canonical, report = _run(vendor_ohlcv_frame(n_bars=5, rows_override=rows))
    assert report.n_rejected_rows == 0
    assert not report.has_kind(DiagnosticKind.IMPLAUSIBLE_NORMALIZED_PRICE)
    assert not report.has_kind(DiagnosticKind.NON_POSITIVE_PRICE)
    assert len(canonical) == 5 + len(rows)   # every row kept, including the negatives
    assert canonical["close"].min() == pytest.approx(-20.0)


def test_negative_volume_rejected():
    bad = {"ts_event": SESSION_START_NS + 100 * MIN_NS, "volume": -3}
    _, report = _run(vendor_ohlcv_frame(n_bars=5, rows_override=[bad]))
    assert report.n_rejected_rows == 1
    assert report.has_kind(DiagnosticKind.NEGATIVE_VOLUME)


def test_continuous_request_symbol_is_not_rejected():
    # Whole frame carries the smart-symbol label "NQ.v.0" -- this is what a real
    # continuous OHLCV response looks like and must NOT cause rejection.
    canonical, report = _run(vendor_ohlcv_frame(n_bars=8, symbol="NQ.v.0"))
    assert len(canonical) == 8
    assert report.n_rejected_rows == 0
    assert not report.has_kind(DiagnosticKind.CONTINUOUS_SYMBOL_AS_RAW)
    assert not report.has_kind(DiagnosticKind.RAW_SYMBOL_MISMATCH)
    assert (canonical["raw_symbol"] == "NQM5").all()   # from the registry, not the label


def test_raw_symbol_always_comes_from_the_registry():
    canonical, _ = _run(vendor_ohlcv_frame(n_bars=3, symbol="NQ.v.0"))
    assert set(canonical["raw_symbol"]) == {"NQM5"}
    assert "NQ.v.0" not in set(canonical["raw_symbol"])


def test_unknown_instrument_id_rejected():
    bad = {"ts_event": SESSION_START_NS + 100 * MIN_NS, "instrument_id": 999999}
    _, report = _run(vendor_ohlcv_frame(n_bars=5, rows_override=[bad]))
    assert report.has_kind(DiagnosticKind.UNKNOWN_INSTRUMENT_ID)
    assert report.n_rejected_rows == 1


def test_independently_resolved_contract_symbol_mismatch_is_flagged():
    # If the vendor gives a *real contract* symbol that disagrees with the
    # definition, that is a genuine conflict (continuous labels are exempt).
    bad = {"ts_event": SESSION_START_NS + 100 * MIN_NS, "symbol": "NQU5"}
    _, report = _run(vendor_ohlcv_frame(n_bars=5, rows_override=[bad]))
    assert report.has_kind(DiagnosticKind.RAW_SYMBOL_MISMATCH)
    assert report.n_rejected_rows == 1


def test_roll_resolves_two_distinct_contracts():
    reg = DefinitionRegistry(
        parse_definition_frame(
            vendor_definition_frame(
                [
                    {"instrument_id": 4021, "raw_symbol": "NQH5", "asset": "NQ"},
                    {"instrument_id": 4022, "raw_symbol": "NQM5", "asset": "NQ"},
                ]
            )
        )
    )
    first = vendor_ohlcv_frame(n_bars=4, instrument_id=4021, symbol="NQ.v.0")
    second = vendor_ohlcv_frame(
        n_bars=4, instrument_id=4022, symbol="NQ.v.0",
        start_ns=SESSION_START_NS + 4 * MIN_NS,
    )
    frame = pd.concat([first, second], ignore_index=True)
    canonical, report = canonicalize(frame, reg, default_calendar())
    assert report.n_rejected_rows == 0
    mapping = dict(zip(canonical["instrument_id"], canonical["raw_symbol"]))
    assert mapping == {4021: "NQH5", 4022: "NQM5"}


def test_undefined_price_sentinel_handled():
    bad = {"ts_event": SESSION_START_NS + 100 * MIN_NS, "close": UNDEF_PRICE}
    canonical, report = _run(vendor_ohlcv_frame(n_bars=5, rows_override=[bad]))
    assert report.has_kind(DiagnosticKind.UNDEFINED_PRICE_SENTINEL)
    assert report.n_rejected_rows == 1
    assert len(canonical) == 5
    with pytest.raises(PipelineError):
        _run(vendor_ohlcv_frame(n_bars=5, rows_override=[bad]),
             policy=NormalizationPolicy(on_undefined_price="fail"))


def test_missing_minute_gap_recorded_not_filled():
    df = vendor_ohlcv_frame(n_bars=10)
    # drop bars 4 and 5 -> a ~3-minute gap in RTH
    df = df.drop(index=[4, 5]).reset_index(drop=True)
    canonical, report = _run(df)
    assert len(canonical) == 8                      # NOT forward-filled
    assert report.has_kind(DiagnosticKind.MISSING_MINUTE_GAP)
    gap = next(d for d in report.diagnostics if d.kind is DiagnosticKind.MISSING_MINUTE_GAP)
    assert gap.severity.value == "warning"
    assert gap.context["missing_bars"] == 2


def test_utc_nanosecond_timestamp_preserved_exactly():
    odd_ns = SESSION_START_NS + 123_456_789  # sub-minute ns must survive verbatim
    df = vendor_ohlcv_frame(n_bars=1)
    df.loc[0, "ts_event"] = odd_ns
    canonical, _ = _run(df)
    assert canonical["ts_event_ns"].iloc[0] == odd_ns
    assert canonical["ts_event_ns"].dtype == "int64"
