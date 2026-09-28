"""News Alpha Phase H -- a frozen portfolio's SIGNAL INPUTS through the
validation window.

Phase G sized a plan from the discovery window (2018-2022) only. Testing the
portfolio that plan defines means replaying it on data the plan never read,
so each member signal's own factor has to exist past 2022:

    CandidateSignal (its FactorExpression, its measurement's bars)
      -> the SAME Phase E loader and registered feature engine
      -> bars from the measurement's coverage start up to 2024-12-31
      -> EvaluationFactorSeries (days, values)

The factor is the one the screen diagnosed -- same expression, same bars, same
engine -- just evaluated further. Two guards keep that honest:

* CAUSAL PREFIX. Every factor in the vocabulary is a trailing operator, so a
  value on a discovery day cannot change when later bars arrive. When the
  Phase E screen is supplied, its series must equal the evaluated series on
  every discovery day, value for value; any difference is a look-ahead defect
  (`LookAheadDefect`), never a rounding issue to tolerate.
* WINDOW. Bars end at 2025-01-01 (exclusive); every timestamp passes the
  dataset holdout guard. The 2025 locked holdout is never loaded.

An evaluation series is not screening evidence (it covers the validation
window) and is never cached with the screens.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from alpha_agent.crypto.provenance import DataProvenanceRole
from alpha_agent.data.real_market_dataset import assert_no_holdout_ts
from alpha_agent.features.expression import evaluate_expression
from alpha_agent.news_alpha.candidate_signals import CandidateSignal
from alpha_agent.portfolio.risk_model import EVALUATION_END
from alpha_agent.screening.candidate_signal_screen import (
    CandidateScreen,
    DailyBarLoader,
    discovery_window,
    load_daily_bars,
)

__all__ = [
    "EvaluationFactorSeries",
    "LookAheadDefect",
    "evaluate_factor_through_validation",
]

EVALUATION_SERIES_SCHEMA = "evaluation-factor-series/1"
_BAR_COLUMNS = ("trading_day", "ts_event_ns", "open", "high", "low", "close", "volume")


class LookAheadDefect(RuntimeError):
    """Evaluating a factor further changed one of its past values."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class EvaluationFactorSeries(BaseModel):
    """One member signal's factor from its coverage start through the
    validation window -- the direction source when a frozen portfolio is
    re-sized at a later date."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = EVALUATION_SERIES_SCHEMA
    candidate_signal_id: str
    factor_identity: str
    instrument: str
    expression: str
    data_role: DataProvenanceRole
    window_start: date
    window_end_exclusive: date
    days: tuple[str, ...]
    #: ``None`` where the window is not yet full or an input is invalid -- never filled.
    values: tuple[float | None, ...]
    data_fingerprint: str
    values_fingerprint: str
    #: Discovery days checked equal to the Phase E screen (0 = not checked).
    prefix_days_verified: int = 0

    @model_validator(mode="after")
    def _bounded(self) -> EvaluationFactorSeries:
        if len(self.days) != len(self.values):
            raise ValueError("one value per trading day")
        if self.window_end_exclusive > EVALUATION_END:
            raise ValueError("an evaluation series never reaches the 2025 holdout")
        if self.days and date.fromisoformat(self.days[-1]) >= self.window_end_exclusive:
            raise ValueError("a factor value beyond its window")
        return self

    @property
    def series(self) -> EvaluationFactorSeries:
        """`portfolio.selection.FactorSource`: the series is itself."""
        return self


def _verify_prefix(days: list[str], values: list[float | None], screen: CandidateScreen) -> int:
    evaluated = dict(zip(days, values, strict=True))
    for day, screened in zip(screen.series.days, screen.series.values, strict=True):
        if day not in evaluated:
            raise LookAheadDefect(f"{screen.candidate_signal_id}: discovery day {day} missing from the evaluation bars")
        now = evaluated[day]
        if (screened is None) != (now is None) or (screened is not None and screened != now):
            raise LookAheadDefect(
                f"{screen.candidate_signal_id}: the factor on {day} was {screened!r} in the discovery screen but "
                f"{now!r} when evaluated through the validation window -- a later bar changed a past value")
    return len(screen.series.days)


def evaluate_factor_through_validation(
    candidate: CandidateSignal,
    *,
    screen: CandidateScreen | None = None,
    loader: DailyBarLoader = load_daily_bars,
) -> EvaluationFactorSeries:
    """The candidate's factor from its discovery-window start to the end of
    the 2023-2024 validation window (or its measurement's coverage end)."""
    spec = candidate.spec
    start, _ = discovery_window(candidate)
    end = min(EVALUATION_END, spec.data.coverage_end_exclusive)
    bars = loader(candidate)
    frame = bars.frame.sort_values("trading_day", kind="stable").reset_index(drop=True)
    day = pd.to_datetime(frame["trading_day"]).dt.date
    frame = frame[(day >= start) & (day < end)].reset_index(drop=True)
    if frame["trading_day"].duplicated().any():
        raise ValueError("more than one bar per trading day")
    assert_no_holdout_ts(frame["ts_event_ns"])
    evaluated = evaluate_expression(
        spec.expression, frame, instrument=spec.instrument, price_domain=bars.price_domain,
        adjustment_mode=bars.adjustment_mode,
    )
    days = list(frame["trading_day"])
    values = [None if not np.isfinite(v) else float(v) for v in evaluated.values]
    verified = 0
    if screen is not None:
        if screen.candidate_signal_id != candidate.candidate_signal_id or screen.expression != candidate.expression:
            raise ValueError("the screen belongs to another candidate")
        verified = _verify_prefix(days, values, screen)
    return EvaluationFactorSeries(
        candidate_signal_id=candidate.candidate_signal_id, factor_identity=candidate.factor_identity,
        instrument=spec.instrument, expression=evaluated.expression, data_role=bars.data_role, window_start=start,
        window_end_exclusive=end, days=tuple(days), values=tuple(values),
        data_fingerprint="bars1:" + _sha(frame[list(_BAR_COLUMNS)].to_csv(index=False)),
        values_fingerprint="factorvals1:" + _sha(json.dumps([None if v is None else repr(v) for v in values])),
        prefix_days_verified=verified,
    )
