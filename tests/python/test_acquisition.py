"""Phase 13.5B -- acquisition request planning (pure, no network)."""
from __future__ import annotations

import pandas as pd
import pytest
from alpha_agent.data.acquisition import (
    HOLDOUT_START,
    AcquisitionBudget,
    BudgetExceeded,
    HoldoutViolation,
    continuous_request,
    definition_snapshot_requests,
    detect_all_transitions,
    guard_no_holdout,
    roll_overlap_requests,
)
from alpha_agent.data.databento_source import HistoricalRequest
from alpha_agent.data.roll_validation import InstrumentSpan, RollDetection

DAY_NS = 86_400_000_000_000


def test_holdout_guard_refuses_anything_reaching_2025():
    with pytest.raises(HoldoutViolation):
        continuous_request("NQ", "2024-06-01", "2025-06-01")
    with pytest.raises(HoldoutViolation):
        continuous_request("NQ", "2025-01-01", "2025-02-01")
    with pytest.raises(HoldoutViolation):
        guard_no_holdout(HistoricalRequest(
            dataset="GLBX.MDP3", symbols=("NQ.FUT",), schema="definition",
            start="2024-12-15", end="2025-01-02", stype_in="parent",
        ))
    ok = continuous_request("NQ", "2024-12-01", HOLDOUT_START)   # end == holdout start is allowed
    assert ok.end == "2025-01-01"


def test_budget_blocks_a_download_that_would_exceed_the_cap():
    b = AcquisitionBudget(cap_usd=65.0)
    b.charge("continuous x5", 44.34)
    b.charge("defs", 0.45)
    b.charge("overlaps", 5.27)
    assert b.remaining() == pytest.approx(65.0 - 50.06)
    b.charge("headroom", 14.0)
    with pytest.raises(BudgetExceeded):
        b.charge("one more", 2.0)


def test_definition_snapshots_are_short_periodic_and_never_touch_holdout():
    # quarterly default: Jan/Apr/Jul/Oct 1, a few days each, aligned to Jan.
    # The window is >1 day so a snapshot whose 1st is a market holiday (Jan 1)
    # still catches a trading day -- an empty parent set left ZNM2 undefined.
    reqs = definition_snapshot_requests("CL", "2023-01-01", "2025-01-01")
    assert [(r.start, r.end) for r in reqs] == [
        ("2023-01-01", "2023-01-05"), ("2023-04-01", "2023-04-05"),
        ("2023-07-01", "2023-07-05"), ("2023-10-01", "2023-10-05"),
        ("2024-01-01", "2024-01-05"), ("2024-04-01", "2024-04-05"),
        ("2024-07-01", "2024-07-05"), ("2024-10-01", "2024-10-05"),
    ]
    assert all(r.schema == "definition" and r.stype_in == "parent" for r in reqs)
    assert all(r.end <= HOLDOUT_START for r in reqs)
    # the trailing window is clipped to the holdout boundary, never past it
    tail = definition_snapshot_requests("NQ", "2024-10-01", "2025-01-01")
    assert [(r.start, r.end) for r in tail] == [("2024-10-01", "2024-10-05")]
    # a partial leading window still aligns to the period grid
    q = definition_snapshot_requests("NQ", "2024-03-15", "2025-01-01", months=3)
    assert [r.start for r in q] == ["2024-04-01", "2024-07-01", "2024-10-01"]


def test_roll_overlap_requests_come_only_from_observed_transitions():
    t = RollDetection(
        continuous_symbol="NQ.v.0", from_instrument_id=1, to_instrument_id=2,
        from_raw_symbol="NQH4", to_raw_symbol="NQM4",
        transition_ts_ns=int(pd.Timestamp("2024-03-08T22:00:00Z").value),
        spans=(InstrumentSpan(1, "NQH4", 0, 1, 1), InstrumentSpan(2, "NQM4", 2, 3, 1)),
    )
    reqs = roll_overlap_requests([t], window_start="2024-01-01", window_end="2024-04-01")
    assert len(reqs) == 1
    assert reqs[0].symbols == ("NQH4", "NQM4")         # BOTH contracts (roll-basis needs it)
    assert reqs[0].stype_in == "raw_symbol"
    assert reqs[0].start < "2024-03-08" < reqs[0].end
    assert reqs[0].end <= "2024-04-01" and reqs[0].end <= HOLDOUT_START
    # a transition whose window would cross the holdout is clipped
    t2 = t.__class__(**{**t.to_dict(),
                        "transition_ts_ns": int(pd.Timestamp("2024-12-30T22:00:00Z").value),
                        "spans": t.spans})
    r2 = roll_overlap_requests([t2], window_start="2024-01-01", window_end="2025-01-01")
    assert all(r.end <= HOLDOUT_START for r in r2)


class _Reg:
    def __init__(self, m):
        self._m = m

    def get(self, i):
        return self._m.get(i)


def test_detect_all_transitions_finds_every_roll_not_just_the_first():
    class S:
        def __init__(self, rs):
            self.raw_symbol = rs

    reg = _Reg({1: S("NQH4"), 2: S("NQM4"), 3: S("NQU4")})
    n = 30
    ts = [i * 60_000_000_000 for i in range(n)]
    iid = [1] * 10 + [2] * 10 + [3] * 10
    vendor = pd.DataFrame({"ts_event": ts, "instrument_id": iid,
                           "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1})
    trans = detect_all_transitions(vendor, reg, continuous_symbol="NQ.v.0")
    assert [(t.from_raw_symbol, t.to_raw_symbol) for t in trans] == [
        ("NQH4", "NQM4"), ("NQM4", "NQU4")
    ]
    assert trans[0].transition_ts_ns == ts[10]
