"""Config-driven, product-aware session / trading-day derivation.

`configs/calendars/cme.yaml` defines one calendar per product family
(equity index, energy, metals, rates) and maps roots to them. No product logic
lives outside the config -- this module only *applies* a calendar: timezone
conversion (DST via zoneinfo), the trading-day boundary, session windows, the
weekly halt, and (architecture in place) per-date holidays / early closes.

UTC timestamps are never modified -- ``trading_day`` / ``session`` are separate
derived values.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from alpha_agent.schemas.market_data import Session

DEFAULT_CALENDAR_CONFIG = Path("configs/calendars/cme.yaml")

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _minutes(hhmm: str) -> int:
    if hhmm == "24:00":
        return 1440
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def parse_local_hhmm(hhmm: str) -> int:
    """``"HH:MM"`` exchange-local wall-clock -> minute-of-day (0..1440)."""
    try:
        return _minutes(hhmm)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid local time {hhmm!r}; expected 'HH:MM'") from exc


class SessionCalendarMissing(KeyError):
    """No calendar is configured for the requested root."""


class _Calendar:
    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.timezone_name = str(cfg["timezone"])
        self.trading_day_boundary_local = str(cfg["trading_day_boundary_local"])
        self.tz = ZoneInfo(cfg["timezone"])
        self.boundary_min = _minutes(cfg["trading_day_boundary_local"])
        self.sessions: list[tuple[int, int, Session]] = [
            (_minutes(s["start"]), _minutes(s["end"]), Session(s["session"]))
            for s in cfg["sessions"]
        ]
        wc = cfg.get("weekly_closed")
        self.closed_from = (
            (_WEEKDAYS[wc["from"]["weekday"]], _minutes(wc["from"]["time"])) if wc else None
        )
        self.closed_to = (
            (_WEEKDAYS[wc["to"]["weekday"]], _minutes(wc["to"]["time"])) if wc else None
        )
        # holiday / early-close architecture (data typically empty this phase)
        self.holidays: frozenset[date] = frozenset(
            date.fromisoformat(d) for d in cfg.get("holidays", [])
        )
        self.early_closes: dict[date, int] = {
            date.fromisoformat(d): _minutes(t) for d, t in (cfg.get("early_closes") or {}).items()
        }

    def _is_weekly_closed(self, weekday: int, minute_of_day: int) -> bool:
        if self.closed_from is None:
            return weekday >= 5
        (fw, fm), (tw, tm) = self.closed_from, self.closed_to
        cur = weekday * 1440 + minute_of_day
        return fw * 1440 + fm <= cur < tw * 1440 + tm

    def _session_at(self, minute_of_day: int) -> Session:
        for start, end, sess in self.sessions:
            if start <= minute_of_day < end:
                return sess
        return Session.CLOSED

    def local_minute_of_day(self, ts_event_ns: int) -> int:
        """Exchange-local minute-of-day (DST-safe via zoneinfo). UTC is untouched."""
        local = pd.Timestamp(int(ts_event_ns), unit="ns", tz="UTC").tz_convert(self.tz)
        return local.hour * 60 + local.minute

    def in_local_window(self, ts_event_ns: int, start_min: int, end_min: int) -> bool:
        """True when the bar's exchange-local wall-clock time is in
        ``[start_min, end_min)``. DST is handled by the timezone conversion, not by
        a hard-coded UTC offset. ``start_min < end_min`` (no midnight wrap)."""
        m = self.local_minute_of_day(ts_event_ns)
        return start_min <= m < end_min

    def classify(self, ts_event_ns: int) -> tuple[date, Session]:
        local = pd.Timestamp(int(ts_event_ns), unit="ns", tz="UTC").tz_convert(self.tz)
        minute_of_day = local.hour * 60 + local.minute
        weekday = local.weekday()
        local_date = local.date()

        trading_day = (
            (local + pd.Timedelta(days=1)).date()
            if minute_of_day >= self.boundary_min else local_date
        )

        if local_date in self.holidays:
            return trading_day, Session.CLOSED
        ec = self.early_closes.get(local_date)
        if ec is not None and minute_of_day >= ec:
            return trading_day, Session.CLOSED
        if self._is_weekly_closed(weekday, minute_of_day):
            return trading_day, Session.CLOSED
        return trading_day, self._session_at(minute_of_day)


class SessionCalendar:
    def __init__(self, config: dict):
        self.version: str = str(config.get("version", "unversioned"))
        self._calendars = {
            name: _Calendar(name, cfg) for name, cfg in config["calendars"].items()
        }
        self._roots: dict[str, str] = dict(config.get("roots", {}))

    @classmethod
    def from_config(cls, path: str | Path = DEFAULT_CALENDAR_CONFIG) -> SessionCalendar:
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def has_root(self, root: str) -> bool:
        return root in self._roots

    def calendar_name_for(self, root: str) -> str:
        name = self._roots.get(root)
        if name is None:
            raise SessionCalendarMissing(f"no session calendar configured for root {root!r}")
        return name

    def classify(self, ts_event_ns: int, root: str) -> tuple[date, Session]:
        return self._calendars[self.calendar_name_for(root)].classify(ts_event_ns)

    def trading_day_convention(self, root: str) -> dict:
        """The trading-day convention for ``root`` -- calendar identity + the
        exchange-local day boundary + timezone. Read-only; recorded in Phase 13
        validation lineage so two day conventions never share one fingerprint."""
        cal = self._calendars[self.calendar_name_for(root)]
        return {
            "calendar_name": cal.name,
            "calendar_version": self.version,
            "timezone": cal.timezone_name,
            "trading_day_boundary_local": cal.trading_day_boundary_local,
        }

    def in_local_window(
        self, ts_event_ns: int, root: str, start_local: str, end_local: str
    ) -> bool:
        """True when the bar's exchange-local wall-clock time (for ``root``'s
        calendar) falls in ``[start_local, end_local)``. DST-safe. Raises if
        ``end_local <= start_local`` (a Silver-Bullet-style window never wraps
        midnight) or the root has no configured calendar."""
        start_min = parse_local_hhmm(start_local)
        end_min = parse_local_hhmm(end_local)
        if end_min <= start_min:
            raise ValueError(
                f"session window end {end_local!r} must be after start {start_local!r}"
            )
        return self._calendars[self.calendar_name_for(root)].in_local_window(
            ts_event_ns, start_min, end_min
        )

    def classify_series(self, ts_ns: pd.Series, root: str) -> tuple[pd.Series, pd.Series]:
        cal = self._calendars[self.calendar_name_for(root)]
        pairs = [cal.classify(int(v)) for v in ts_ns]
        return (
            pd.Series([p[0] for p in pairs], index=ts_ns.index, dtype="object"),
            pd.Series([p[1].value for p in pairs], index=ts_ns.index, dtype="object"),
        )


_DEFAULT_CALENDAR_CACHE: dict[str, SessionCalendar] = {}


def default_calendar(config: str | Path = DEFAULT_CALENDAR_CONFIG) -> SessionCalendar:
    """Load (and cache) the config-driven session calendar. Cached by path so a
    feature computation does not re-parse the YAML on every call."""
    key = str(config)
    cal = _DEFAULT_CALENDAR_CACHE.get(key)
    if cal is None:
        cal = SessionCalendar.from_config(config)
        _DEFAULT_CALENDAR_CACHE[key] = cal
    return cal
