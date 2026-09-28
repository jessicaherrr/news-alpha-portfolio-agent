"""A US-equity :class:`~alpha_agent.data.calendars.SessionCalendar` for the
Phase 9.1 equities foundation -- byte-identical shape to
``alpha_agent.etf.calendar.etf_calendar`` (same market, same daily-bar
granularity, same weekday-closed rule), declared separately per this
package's domain-isolation principle rather than imported from ``etf``.

Built in-memory and passed explicitly via ``ValidationEngine(..., calendar=...)``
when a future phase runs one -- never registered into the shared, CME-specific
``configs/calendars/cme.yaml``, so it cannot perturb any real Futures root's
calendar.
"""
from __future__ import annotations

from alpha_agent.data.calendars import SessionCalendar
from alpha_agent.equities.universe import EQUITY_UNIVERSE

_EQUITY_CALENDAR_CONFIG = {
    "version": "phase9-1-equities/1",
    "calendars": {
        "us_equity_daily": {
            "timezone": "America/New_York",
            "trading_day_boundary_local": "00:00",
            "sessions": [{"start": "00:00", "end": "24:00", "session": "RTH"}],
        },
    },
    "roots": {ticker: "us_equity_daily" for ticker in EQUITY_UNIVERSE},
}


def equity_calendar() -> SessionCalendar:
    return SessionCalendar(_EQUITY_CALENDAR_CONFIG)
