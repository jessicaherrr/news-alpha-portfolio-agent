"""A US-equity :class:`~alpha_agent.data.calendars.SessionCalendar` for the
Phase 6 ETF pilot, mirroring the precedent
``alpha_agent.crypto.calendar.crypto_calendar`` already established for a
different new asset domain: every ``ValidationEngine`` run needs a session
calendar to derive its canonical ``trading_day`` series, CME or not.

This is a daily-bar-only calendar -- the pilot's acquired granularity is
``ohlcv-1d``, so the exact NYSE intraday open/close instant does not affect
``trading_day``/weekly-closed classification, only the date boundary and
weekday-closed rule do. It exists only so the engine has a calendar to
consult, is built in-memory, and is passed explicitly via
``ValidationEngine(..., calendar=...)`` -- never registered into the shared,
CME-specific ``configs/calendars/cme.yaml``, so it cannot perturb any real
Futures root's calendar. Unlike crypto's 24/7 market, this deliberately does
NOT override ``weekly_closed`` -- ``SessionCalendar``'s default
(``weekday >= 5`` -> closed Saturday/Sunday) is the CORRECT rule for a real
US equity market, not a CME-oriented approximation.
"""
from __future__ import annotations

from alpha_agent.data.calendars import SessionCalendar
from alpha_agent.etf.universe import PILOT_UNIVERSE

_ETF_CALENDAR_CONFIG = {
    "version": "phase6-etf/1",
    "calendars": {
        "us_equity_daily": {
            "timezone": "America/New_York",
            "trading_day_boundary_local": "00:00",
            # One all-day session labelled RTH -- daily bars carry no
            # intraday resolution, so the exact 09:30-16:00 boundary is not
            # load-bearing for trading_day/session classification.
            "sessions": [{"start": "00:00", "end": "24:00", "session": "RTH"}],
        },
    },
    "roots": {ticker: "us_equity_daily" for ticker in PILOT_UNIVERSE},
}


def etf_calendar() -> SessionCalendar:
    return SessionCalendar(_ETF_CALENDAR_CONFIG)
