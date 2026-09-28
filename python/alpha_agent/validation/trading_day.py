"""Canonical futures trading-day boundaries for the validation return series
(Phase 13.1).

A CME exchange trading day (e.g. 17:00 -> 17:00 America/Chicago, DST-safe via the
existing ``SessionCalendar``) spans a UTC midnight, so UTC calendar dates are
**not** an acceptable canonical daily bucket. This module derives, from the bars
about to be sent to the C++ engine, one boundary timestamp per observed
``trading_day`` -- the ts of that day's last eligible event. The C++ engine emits
the ``PortfolioAccountant`` equity at exactly those instants
(``EngineConfig::validation_day_boundaries_ns``); Python never reconstructs
equity.

The trading-day convention (calendar name + version + timezone + local boundary)
enters validation lineage and the ``ValidationSpec`` fingerprint.
"""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from alpha_agent.data.calendars import SessionCalendar, default_calendar
from alpha_agent.validation.fingerprint import fingerprint


class TradingDayConvention(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    calendar_name: str
    calendar_version: str
    timezone: str
    trading_day_boundary_local: str

    def identity(self) -> str:
        return fingerprint("valtradingdayconv1", self.model_dump(mode="json"))

    @classmethod
    def for_root(cls, root_symbol: str, calendar: SessionCalendar | None = None) -> TradingDayConvention:
        cal = calendar or default_calendar()
        return cls(**cal.trading_day_convention(root_symbol))


class ValidationDay(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    trading_day: str                     # ISO date of the canonical futures session
    first_ts_ns: int
    boundary_ts_ns: int                  # ts of this trading_day's last eligible event
    n_bars: int = Field(ge=1)


class ValidationDayPlan(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "validation-day-plan/1"
    root_symbol: str
    convention: TradingDayConvention
    days: tuple[ValidationDay, ...]

    @property
    def n_days(self) -> int:
        return len(self.days)

    def boundary_ts_list(self) -> list[int]:
        return [d.boundary_ts_ns for d in self.days]

    def trading_day_labels(self) -> list[str]:
        return [d.trading_day for d in self.days]

    def identity(self) -> str:
        return fingerprint(
            "valdayplan1",
            {
                "schema_version": self.schema_version,
                "root_symbol": self.root_symbol,
                "convention": self.convention.identity(),
                "days": [[d.trading_day, d.boundary_ts_ns] for d in self.days],
            },
        )

    def write_csv(self, path) -> object:
        import pathlib

        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("boundary_ts_ns\n" + "\n".join(str(t) for t in self.boundary_ts_list()) + "\n")
        return p


def build_validation_day_plan(
    bars: pd.DataFrame,
    *,
    root_symbol: str,
    calendar: SessionCalendar | None = None,
) -> ValidationDayPlan:
    """Group ``bars`` (must have ``ts_event_ns``) into canonical futures
    trading days via the ``SessionCalendar`` and record each day's last-event
    boundary. Bars are used chronologically; no bar is dropped or forward-filled.
    """
    if "ts_event_ns" not in bars.columns:
        raise ValueError("bars frame needs a ts_event_ns column")
    cal = calendar or default_calendar()
    df = bars[["ts_event_ns"]].sort_values("ts_event_ns").reset_index(drop=True)
    if df.empty:
        return ValidationDayPlan(
            root_symbol=root_symbol,
            convention=TradingDayConvention.for_root(root_symbol, cal),
            days=(),
        )
    tdays, _sessions = cal.classify_series(df["ts_event_ns"], root_symbol)
    labels = [str(v) for v in tdays]

    days: list[ValidationDay] = []
    run_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[run_start]:
            first = int(df["ts_event_ns"].iloc[run_start])
            last = int(df["ts_event_ns"].iloc[i - 1])
            days.append(
                ValidationDay(
                    trading_day=labels[run_start],
                    first_ts_ns=first,
                    boundary_ts_ns=last,
                    n_bars=i - run_start,
                )
            )
            run_start = i

    # boundaries must be strictly ascending for the C++ reader
    from itertools import pairwise

    for a, b in pairwise(days):
        if b.boundary_ts_ns <= a.boundary_ts_ns:
            raise ValueError("trading-day boundaries are not strictly ascending")
    return ValidationDayPlan(
        root_symbol=root_symbol,
        convention=TradingDayConvention.for_root(root_symbol, cal),
        days=tuple(days),
    )
