"""COT / macro point-in-time alignment interface (section 12).

No CFTC / FRED data is acquired here. This module only defines the typed
extension point so the feature engine can later align:

* CFTC Commitments of Traders (report date vs release date),
* FRED / ALFRED macro series (observation date vs vintage/publication date),
* scheduled macro releases (surprise = actual - consensus),

using **two** timestamps per datum: the period it describes (``reference_ts_ns``)
and the instant it became public (``published_ts_ns``). A feature at ``T`` may
only use a datum whose ``published_ts_ns <= T`` -- future publication data can
never leak backward.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


class PointInTimeDatum(BaseModel):
    model_config = {"frozen": True}

    reference_ts_ns: int = Field(gt=0)   # the period/date the value describes
    published_ts_ns: int = Field(gt=0)   # when the value first became public
    value: float

    @model_validator(mode="after")
    def _check(self) -> PointInTimeDatum:
        if self.published_ts_ns < self.reference_ts_ns:
            raise ValueError(
                "published_ts_ns precedes reference_ts_ns -- a release cannot predate its "
                "own reference period"
            )
        return self


class PointInTimeSeries(BaseModel):
    """An ordered set of point-in-time data for one external series."""

    series_id: str = Field(min_length=1)
    kind: str = "generic"                 # "cot" | "fred" | "macro_release" | "generic"
    data: tuple[PointInTimeDatum, ...] = ()

    def sorted_by_publication(self) -> list[PointInTimeDatum]:
        return sorted(self.data, key=lambda d: (d.published_ts_ns, d.reference_ts_ns))


class COTReport(BaseModel):
    """One CFTC COT row. ``report_date_ns`` is the Tuesday snapshot; releases
    ~3 days later (``release_ts_ns``)."""

    report_date_ns: int = Field(gt=0)
    release_ts_ns: int = Field(gt=0)
    net_position: float

    def to_datum(self) -> PointInTimeDatum:
        return PointInTimeDatum(
            reference_ts_ns=self.report_date_ns,
            published_ts_ns=self.release_ts_ns,
            value=self.net_position,
        )


class MacroRelease(BaseModel):
    """One scheduled macro release (e.g. CPI, NFP)."""

    reference_period_ns: int = Field(gt=0)
    release_ts_ns: int = Field(gt=0)
    actual: float
    consensus: float | None = None

    def surprise(self) -> float | None:
        return None if self.consensus is None else self.actual - self.consensus

    def to_datum(self, *, use_surprise: bool = False) -> PointInTimeDatum:
        v = self.surprise() if use_surprise else self.actual
        if v is None:
            raise ValueError("surprise requested but no consensus on the release")
        return PointInTimeDatum(
            reference_ts_ns=self.reference_period_ns,
            published_ts_ns=self.release_ts_ns,
            value=float(v),
        )


def align_as_of(
    feature_ts: Iterable[int],
    series: PointInTimeSeries,
    *,
    as_of_ts_ns: int | None = None,
    max_staleness_ns: int | None = None,
) -> pd.Series:
    """For each feature timestamp ``T`` return the value of the latest datum
    whose ``published_ts_ns <= min(T, as_of_ts_ns)``. Never uses a datum
    published after ``T``. Optionally drops values older than
    ``max_staleness_ns`` (by publication time)."""
    ts_index = pd.Index([int(t) for t in feature_ts], name="ts_event_ns")
    pub = np.array([d.published_ts_ns for d in series.sorted_by_publication()], dtype="int64")
    val = np.array([d.value for d in series.sorted_by_publication()], dtype="float64")
    out = np.full(len(ts_index), np.nan, dtype="float64")
    if pub.size:
        for i, t in enumerate(ts_index):
            cutoff = t if as_of_ts_ns is None else min(int(t), int(as_of_ts_ns))
            j = int(np.searchsorted(pub, cutoff, side="right")) - 1
            if j < 0:
                continue
            if max_staleness_ns is not None and (cutoff - pub[j]) > max_staleness_ns:
                continue
            out[i] = val[j]
    return pd.Series(out, index=ts_index, name=series.series_id)
