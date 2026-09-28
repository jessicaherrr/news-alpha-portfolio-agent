"""A continuous, 24/7 :class:`~alpha_agent.data.calendars.SessionCalendar` for
the Phase 22 crypto scaffold.

Every :class:`~alpha_agent.validation.engine.ValidationEngine` run needs a
session calendar to derive its canonical ``trading_day`` series -- CME or not.
Real CME-listed BTC/ETH futures actually trade nearly around the clock with a
short daily maintenance break; this in-memory calendar does not attempt to
reproduce that schedule precisely (it would be guesswork with no committed
CME specification behind it). It exists only so the engine has a calendar to
consult over a UTC-midnight-aligned synthetic daily bar series, is built
in-memory, and is passed explicitly via ``ValidationEngine(..., calendar=...)``
-- it is never registered into the shared, CME-specific
``configs/calendars/cme.yaml`` the real futures family reads, so it cannot
perturb any real root's calendar.
"""
from __future__ import annotations

from alpha_agent.data.calendars import SessionCalendar

#: A `weekly_closed` window whose `from` and `to` are the identical
#: weekday+time is a zero-width interval (`X <= cur < X` is never true), so
#: `_is_weekly_closed` always returns False -- crypto trades every day of the
#: week. This is deliberately explicit (an *empty* configured window) rather
#: than omitting `weekly_closed`, which would fall back to
#: `SessionCalendar`'s CME-oriented default (`weekday >= 5` -> closed
#: Sat/Sun) -- wrong for a 24/7 market.
_CRYPTO_CALENDAR_CONFIG = {
    "version": "phase22-crypto-synthetic/1",
    "calendars": {
        "crypto_continuous": {
            "timezone": "UTC",
            "trading_day_boundary_local": "00:00",
            # One all-day session labelled ETH (electronic trading hours) --
            # the closest existing Session value to a continuously-traded
            # market; not a claim about real CME BTC/ETH session structure.
            "sessions": [{"start": "00:00", "end": "24:00", "session": "ETH"}],
            "weekly_closed": {
                "from": {"weekday": "monday", "time": "00:00"},
                "to": {"weekday": "monday", "time": "00:00"},
            },
        },
    },
    "roots": {"BTC": "crypto_continuous", "ETH": "crypto_continuous"},
}


def crypto_calendar() -> SessionCalendar:
    return SessionCalendar(_CRYPTO_CALENDAR_CONFIG)
