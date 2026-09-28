"""Phase 04 -- contract lifecycle + external first-notice join."""
from __future__ import annotations

import json

import pandas as pd
import pytest
from alpha_agent.data.contract_lifecycle import (
    ContractLifecycleMeta,
    FirstNoticePolicy,
    check_lifecycle_ordering,
    enrich_specs,
    load_lifecycle_meta,
    tradable_until_ns,
)
from alpha_agent.data.diagnostics import DiagnosticKind, DiagnosticsReport
from alpha_agent.schemas.market_data import ContractSpecModel

_A = pd.Timestamp("2026-01-01T00:00:00Z").value
_E = pd.Timestamp("2026-12-18T00:00:00Z").value
_FN = pd.Timestamp("2026-11-25T00:00:00Z").value


def _spec(**over) -> ContractSpecModel:
    base = {
        "instrument_id": 1, "raw_symbol": "CLZ6", "root_symbol": "CL", "exchange": "XNYM",
        "tick_size": 0.01, "multiplier": 1000.0, "activation_ns": _A, "expiration_ns": _E,
    }
    base.update(over)
    return ContractSpecModel(**base)


def test_tradable_until_takes_the_minimum():
    assert tradable_until_ns(_spec()) == _E                       # only expiration
    assert tradable_until_ns(_spec(last_trade_ns=_E - 10**12)) == _E - 10**12
    s = _spec(first_notice_ns=_FN)
    assert tradable_until_ns(s) == _FN                            # FND is earliest
    # safety buffer rolls even earlier
    buffered = tradable_until_ns(s, FirstNoticePolicy(safety_buffer_days=2))
    assert buffered == _FN - 2 * 86_400_000_000_000
    # policy can ignore FND
    assert tradable_until_ns(s, FirstNoticePolicy(apply_first_notice=False)) == _E


def test_lifecycle_ordering_is_validated():
    assert check_lifecycle_ordering(_spec()) is None
    assert check_lifecycle_ordering(_spec(first_notice_ns=_A - 1)) is not None   # FN <= activation
    assert check_lifecycle_ordering(_spec(first_notice_ns=_E + 10**12)) is not None  # FN > expiration
    assert check_lifecycle_ordering(_spec(last_trade_ns=_A - 1)) is not None


def test_missing_first_notice_never_manufactured_only_warns():
    report = DiagnosticsReport()
    out = enrich_specs([_spec()], meta={}, report=report)   # CL is physically delivered
    assert out[0].first_notice_ns is None                   # NOT invented
    assert report.has_kind(DiagnosticKind.MISSING_FIRST_NOTICE_FOR_DELIVERABLE)
    assert tradable_until_ns(out[0]) == _E                   # falls back to expiration


def test_external_meta_join(tmp_path):
    p = tmp_path / "lifecycle.json"
    p.write_text(json.dumps([
        {"raw_symbol": "CLZ6", "first_notice_ns": _FN, "last_trade_ns": _E - 10**12,
         "source": "cme_calendar"},
    ]))
    meta = load_lifecycle_meta(p)
    assert isinstance(meta["CLZ6"], ContractLifecycleMeta)
    out = enrich_specs([_spec()], meta=meta, report=DiagnosticsReport())
    assert out[0].first_notice_ns == _FN
    assert out[0].last_trade_ns == _E - 10**12
    assert load_lifecycle_meta(tmp_path / "nope.json") == {}   # missing file -> empty


def test_from_dates_helper():
    m = ContractLifecycleMeta.from_dates("CLZ6", source="manual",
                                         first_notice="2026-11-25", last_trade="2026-12-19")
    assert m.first_notice_ns is not None and m.last_trade_ns is not None


def test_invalid_activation_expiration_ordering_fails_at_model():
    with pytest.raises(ValueError):
        _spec(activation_ns=_E, expiration_ns=_A)


def test_business_day_first_notice_buffer():
    # first notice on Wed 2026-11-25; a 2-business-day buffer -> Mon 2026-11-23.
    fn = pd.Timestamp("2026-11-25T00:00:00Z").value
    s = _spec(first_notice_ns=fn)
    got = tradable_until_ns(s, FirstNoticePolicy(safety_buffer_business_days=2))
    assert pd.Timestamp(got, unit="ns", tz="UTC").date().isoformat() == "2026-11-23"
    # calendar-day buffer of 2 would land on Sun 2026-11-23 too here, but a
    # 4-business-day buffer skips the weekend: Wed -> Tue -> Mon -> Fri -> Thu
    got4 = tradable_until_ns(s, FirstNoticePolicy(safety_buffer_business_days=4))
    assert pd.Timestamp(got4, unit="ns", tz="UTC").date().isoformat() == "2026-11-19"
