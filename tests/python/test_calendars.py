"""Phase 03 -- config-driven session / trading-day derivation."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from alpha_agent.data.calendars import SessionCalendar, SessionCalendarMissing, default_calendar
from alpha_agent.schemas.market_data import Session


def _ns(iso: str) -> int:
    return pd.Timestamp(iso).value


def test_rth_bar_classifies():
    cal = default_calendar()
    # 13:30 UTC 2025-03-17 == 08:30 America/Chicago (CDT) -> RTH open
    day, sess = cal.classify(_ns("2025-03-17T13:30:00Z"), "NQ")
    assert day == date(2025, 3, 17)
    assert sess is Session.RTH


def test_overnight_rolls_to_next_trading_day():
    cal = default_calendar()
    # 23:00 UTC == 18:00 CDT -> past the 17:00 boundary -> next day's ETH
    day, sess = cal.classify(_ns("2025-03-17T23:00:00Z"), "NQ")
    assert day == date(2025, 3, 18)
    assert sess is Session.ETH


def test_maintenance_and_weekend_closed():
    cal = default_calendar()
    # 21:30 UTC == 16:30 CDT -> maintenance break
    _, sess = cal.classify(_ns("2025-03-17T21:30:00Z"), "NQ")
    assert sess is Session.MAINTENANCE
    # Saturday
    _, sat = cal.classify(_ns("2025-03-15T18:00:00Z"), "NQ")
    assert sat is Session.CLOSED


def test_dst_is_handled_by_zoneinfo():
    cal = default_calendar()
    # 14:30 UTC in January == 08:30 CST (RTH); in July 13:30 UTC == 08:30 CDT (RTH)
    _, jan = cal.classify(_ns("2025-01-15T14:30:00Z"), "NQ")
    _, jul = cal.classify(_ns("2025-07-15T13:30:00Z"), "NQ")
    assert jan is Session.RTH and jul is Session.RTH


def test_unknown_root_raises():
    cal = default_calendar()
    with pytest.raises(SessionCalendarMissing):
        cal.classify(_ns("2025-03-17T13:30:00Z"), "ZZ")


def test_classify_series_matches_scalar():
    cal = SessionCalendar.from_config()
    ts = pd.Series([_ns("2025-03-17T13:30:00Z"), _ns("2025-03-17T13:31:00Z")])
    days, sessions = cal.classify_series(ts, "NQ")
    assert list(sessions) == ["RTH", "RTH"]
    assert list(days) == [date(2025, 3, 17), date(2025, 3, 17)]


# --- Phase 04 -----------------------------------------------------------

def test_products_use_distinct_calendar_config():
    cal = default_calendar()
    names = {r: cal.calendar_name_for(r) for r in ("ES", "NQ", "CL", "GC", "ZN")}
    assert names == {
        "ES": "cme_equity_index", "NQ": "cme_equity_index",
        "CL": "nymex_energy", "GC": "comex_metals", "ZN": "cbot_rates",
    }
    assert cal.version  # calendar_version is surfaced for lineage


def test_product_rth_windows_differ():
    cal = default_calendar()
    # 07:30 CT (13:30 UTC in July): RTH for GC/ZN (07:20 open) but not NQ (08:30)
    ts = _ns("2025-07-15T12:30:00Z")   # 07:30 CT
    assert cal.classify(ts, "NQ")[1] is Session.ETH
    assert cal.classify(ts, "GC")[1] is Session.RTH
    assert cal.classify(ts, "ZN")[1] is Session.RTH
    # 14:30 CT: RTH still for CL? no (13:30 close). RTH for ZN (14:00 close)? no.
    ts2 = _ns("2025-07-15T19:30:00Z")  # 14:30 CT
    assert cal.classify(ts2, "CL")[1] is Session.ETH
    assert cal.classify(ts2, "NQ")[1] is Session.RTH   # NQ RTH to 15:00 CT


def test_holiday_and_early_close_architecture():
    cfg = {
        "version": "test",
        "calendars": {"x": {
            "timezone": "America/Chicago", "trading_day_boundary_local": "17:00",
            "sessions": [{"session": "RTH", "start": "08:30", "end": "15:00"},
                         {"session": "ETH", "start": "00:00", "end": "08:30"}],
            "holidays": ["2026-07-03"],
            "early_closes": {"2026-11-27": "12:00"},
        }},
        "roots": {"XX": "x"},
    }
    cal = SessionCalendar(cfg)
    assert cal.classify(_ns("2026-07-03T14:00:00Z"), "XX")[1] is Session.CLOSED  # holiday
    # 2026-11-27 13:00 CT (19:00 UTC, CST) -> after the 12:00 early close
    assert cal.classify(_ns("2026-11-27T19:00:00Z"), "XX")[1] is Session.CLOSED
    # 2026-11-27 10:00 CT -> still open
    assert cal.classify(_ns("2026-11-27T16:00:00Z"), "XX")[1] is Session.RTH


def test_dst_winter_vs_summer_transition_days():
    cal = default_calendar()
    # 2025 US DST: spring forward 2025-03-09, fall back 2025-11-02.
    # 08:30 local NQ RTH open: CST = 14:30 UTC, CDT = 13:30 UTC.
    assert cal.classify(_ns("2025-03-07T14:30:00Z"), "NQ")[1] is Session.RTH  # Fri, CST
    assert cal.classify(_ns("2025-03-10T13:30:00Z"), "NQ")[1] is Session.RTH  # Mon, CDT
    assert cal.classify(_ns("2025-11-03T14:30:00Z"), "NQ")[1] is Session.RTH  # Mon, back to CST
    # the wrong-season UTC offset lands OUTSIDE RTH
    assert cal.classify(_ns("2025-11-03T13:30:00Z"), "NQ")[1] is Session.ETH  # 07:30 CST, pre-open
