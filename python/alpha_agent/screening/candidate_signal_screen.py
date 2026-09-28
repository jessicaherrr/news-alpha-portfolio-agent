"""News Alpha Phase E -- the factor SERIES and SCREEN of a candidate signal.

    CandidateSignal (news_alpha, hypothesis plane)
      -> point-in-time daily bars from the dataset its measurement resolved to
      -> the DISCOVERY window only (2018-01-01 -> 2023-01-01, `RESEARCH_WINDOW`)
      -> its FactorExpression, through the registered feature engine = FactorSeries
      -> `factor_diagnostics.diagnose_factor`                        = FactorDiagnostics
      -> CandidateScreen, kept in an operational cache (never the registry)

ONE EXPRESSION. The series is computed from `candidate.spec.expression` and
the diagnostics are handed that expression's `render()`; `CandidateScreen`
refuses to exist unless candidate, series and diagnostics name the same
formula and the same structural identity.

WINDOWS. Screening reads 2018-2022 only, so the 2023-2024 validation window
stays unused by candidate selection and the locked 2025 holdout is never
reached: the frame is sliced by trading day, then checked by the Fast
Screen's own `assert_research_window_only` and the dataset layer's
`assert_no_holdout_ts` before any feature is computed.

DATA. Futures: the roll-resolved forward-adjusted continuous front
(GLBX.MDP3 1-minute, rebuilt offline from the raw store) collapsed to one
bar per CME trading day by `daily_signal_series` -- the research series
signals may use; fills would resolve to raw contracts in the C++ engine,
which this module never calls. ETFs: the primary-listing daily bars, with
the only SOURCED split in the pilot (USO, 2020-04-29, SEC 8-K) applied
backward; distributions are not sourced, so returns stay price-only (the
measurement already says so as a proxy). No bar is fetched from a vendor.

A screen is SCREENING evidence: no registry write, no verdict, no alpha
memory promotion. Screens computed from anything but REAL data cannot be
stored.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from alpha_agent.crypto.provenance import DataProvenanceRole
from alpha_agent.data.real_market_dataset import RESEARCH_WINDOW, assert_no_holdout_ts
from alpha_agent.features.expression import evaluate_expression
from alpha_agent.news_alpha.candidate_signals import CandidateSignal, CandidateSignalSet
from alpha_agent.news_alpha.mandate import MandateDomain
from alpha_agent.schemas.market_data import PriceDomain
from alpha_agent.screening.factor_diagnostics import (
    FACTOR_DIAGNOSTICS_SCHEMA,
    FACTOR_SCREEN_RULE,
    DiagnosticScope,
    FactorDiagnostics,
    FactorScreenPolicy,
    SignalCorrelation,
    diagnose_factor,
    signal_correlations,
)
from alpha_agent.screening.fast_screen import assert_research_window_only

__all__ = [
    "CANDIDATE_SCREEN_SCHEMA",
    "DEFAULT_SCREEN_DIR",
    "SCREENABLE_DOMAINS",
    "CandidateScreen",
    "CandidateScreenRun",
    "DailyBars",
    "FactorScreenStore",
    "FactorSeries",
    "discovery_window",
    "load_daily_bars",
    "screen_candidate",
    "screen_candidates",
]

CANDIDATE_SCREEN_SCHEMA = "candidate-screen/2"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCREEN_DIR = REPO_ROOT / "data" / "news_alpha" / "factor_screens"
_RESEARCH_START = date.fromisoformat(RESEARCH_WINDOW[0])
_RESEARCH_END = date.fromisoformat(RESEARCH_WINDOW[1])  # exclusive == validation start
_BAR_COLUMNS = ("trading_day", "ts_event_ns", "open", "high", "low", "close", "volume")
#: The domains `load_daily_bars` has real point-in-time bars for. A candidate
#: in any other domain has no diagnostic method yet (never a low-quality one).
SCREENABLE_DOMAINS: frozenset[MandateDomain] = frozenset({MandateDomain.FUTURES, MandateDomain.ETF})


@dataclass(frozen=True)
class DailyBars:
    """One instrument's one-row-per-trading-day bars (``trading_day`` ISO,
    ascending) and how they were built."""

    frame: pd.DataFrame
    price_domain: PriceDomain
    adjustment_mode: str | None
    data_role: DataProvenanceRole
    provenance: tuple[str, ...]


DailyBarLoader = Callable[[CandidateSignal], DailyBars]


# ---------------------------------------------------------------------------
# real point-in-time loaders
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _futures_daily(root: str) -> pd.DataFrame:
    """~1 minute per root (a full offline rebuild from the raw store), so
    cached for the process."""
    from alpha_agent.data.calendars import default_calendar
    from alpha_agent.data.real_market_dataset import daily_signal_series, reconstitute_root

    recon = reconstitute_root(root)
    return daily_signal_series(recon.forward_adjusted, root, calendar=default_calendar())


def _etf_daily(symbol: str) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    from alpha_agent.etf.corporate_actions import known_splits
    from alpha_agent.etf.data_source import load_primary_listing_bars

    raw, source = load_primary_listing_bars(symbol)
    frame = pd.DataFrame({
        "trading_day": raw["day"].astype(str), "ts_event_ns": raw["ts_event"].astype("int64"),
        "open": raw["open"], "high": raw["high"], "low": raw["low"], "close": raw["close"],
        "volume": raw["volume"].astype("float64"),
    })
    notes = [f"{source.dataset} ohlcv-1d, primary listing ({source.listing_exchange}); price return only"]
    for split in known_splits(symbol):
        before = pd.to_datetime(frame["trading_day"]).dt.date < split.effective_date
        for c in ("open", "high", "low", "close"):
            frame.loc[before, c] = frame.loc[before, c] / split.ratio
        frame.loc[before, "volume"] = frame.loc[before, "volume"] * split.ratio
        notes.append(f"split-adjusted backward for {split.ratio:g} on {split.effective_date} ({split.source.name})")
    return frame, source.dataset, tuple(notes)


def load_daily_bars(candidate: CandidateSignal) -> DailyBars:
    """The real point-in-time bars the candidate's measurement resolved to."""
    spec = candidate.spec
    if spec.domain is MandateDomain.FUTURES:
        frame = _futures_daily(spec.instrument)[list(_BAR_COLUMNS)].copy()
        return DailyBars(
            frame=frame, price_domain=PriceDomain.BACK_ADJUSTED, adjustment_mode="forward_adjusted",
            data_role=DataProvenanceRole.REAL,
            provenance=((
                f"{spec.data.dataset} {spec.data.data_schema} {spec.instrument}.v.0 continuous front, forward-adjusted "
                "at each observed roll, one bar per CME trading day (data.real_market_dataset.daily_signal_series)"
            ),),
        )
    if spec.domain is MandateDomain.ETF:
        frame, dataset, notes = _etf_daily(spec.instrument)
        if dataset != spec.data.dataset:
            raise ValueError(f"{spec.instrument}: loaded {dataset}, but the measurement resolved {spec.data.dataset}")
        adjusted = any("split-adjusted" in n for n in notes)
        return DailyBars(
            frame=frame, price_domain=PriceDomain.SPLIT_ADJUSTED if adjusted else PriceDomain.RAW,
            adjustment_mode=None, data_role=DataProvenanceRole.REAL, provenance=notes,
        )
    assert spec.domain not in SCREENABLE_DOMAINS, "every screenable domain has a loader above"
    raise ValueError(f"no point-in-time daily bar loader for {spec.domain.value} (no {spec.domain.value} bars exist)")


# ---------------------------------------------------------------------------
# series, screen, run
# ---------------------------------------------------------------------------


def discovery_window(candidate: CandidateSignal) -> tuple[date, date]:
    """``[start, end)``: the research (discovery) window intersected with the
    measurement's own coverage."""
    data = candidate.spec.data
    return max(_RESEARCH_START, data.coverage_start), min(_RESEARCH_END, data.coverage_end_exclusive)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class FactorSeries(BaseModel):
    """``factor_value(i, t)`` for one candidate over its discovery window --
    enough to reproduce and audit every value."""

    model_config = {"frozen": True, "extra": "forbid"}

    candidate_signal_id: str
    factor_identity: str
    instrument: str
    expression: str
    feature_name: str
    feature_engine_version: str
    price_domain: PriceDomain
    adjustment_mode: str | None
    data_role: DataProvenanceRole
    window_start: date
    window_end_exclusive: date
    days: tuple[str, ...]
    #: ``None`` where the window is not yet full or an input is invalid --
    #: never filled.
    values: tuple[float | None, ...]
    #: SHA-256 of the exact bars the factor was computed from.
    data_fingerprint: str
    values_fingerprint: str
    source_fingerprint: str
    input_provenance: tuple[str, ...]
    qa_issues: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _aligned(self) -> FactorSeries:
        if len(self.days) != len(self.values):
            raise ValueError("one value per trading day")
        if self.days and date.fromisoformat(self.days[-1]) >= self.window_end_exclusive:
            raise ValueError("a factor value beyond its window")
        return self

    def as_series(self) -> pd.Series:
        return pd.Series([np.nan if v is None else v for v in self.values], index=list(self.days), dtype="float64")


class CandidateScreen(BaseModel):
    """One candidate's factor series and screening diagnostics -- screening
    evidence, never a verdict."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = CANDIDATE_SCREEN_SCHEMA
    plane: Literal["SCREENING"] = "SCREENING"
    candidate_signal_id: str
    factor_identity: str
    expression: str
    series: FactorSeries
    diagnostics: FactorDiagnostics
    screened_at: datetime

    @model_validator(mode="after")
    def _one_expression(self) -> CandidateScreen:
        if not self.expression == self.series.expression == self.diagnostics.expression:
            raise ValueError("the screened series and diagnostics must be of the candidate's own expression")
        if (self.series.candidate_signal_id, self.series.factor_identity) != (
            self.candidate_signal_id, self.factor_identity,
        ):
            raise ValueError("the series belongs to another candidate")
        return self

    def fingerprint(self) -> str:
        """Over the evidence only (not ``screened_at``): the same inputs give
        the same fingerprint."""
        payload = self.model_dump(mode="json", exclude={"screened_at"})
        return "candscreen1:" + _sha(json.dumps(payload, sort_keys=True))


def _slice(bars: DailyBars, start: date, end: date) -> pd.DataFrame:
    f = bars.frame.sort_values("trading_day", kind="stable").reset_index(drop=True)
    days = pd.to_datetime(f["trading_day"]).dt.date
    f = f[(days >= start) & (days < end)].reset_index(drop=True)
    if f["trading_day"].duplicated().any():
        raise ValueError("more than one bar per trading day")
    assert_research_window_only(f)  # the Fast Screen's own 2023-01-01 boundary
    assert_no_holdout_ts(f["ts_event_ns"])
    return f


def screen_candidate(
    candidate: CandidateSignal,
    *,
    loader: DailyBarLoader = load_daily_bars,
    policy: FactorScreenPolicy | None = None,
    now: datetime | None = None,
) -> CandidateScreen:
    spec = candidate.spec
    start, end = discovery_window(candidate)
    bars = loader(candidate)
    frame = _slice(bars, start, end)
    evaluated = evaluate_expression(
        spec.expression, frame, instrument=spec.instrument, price_domain=bars.price_domain,
        adjustment_mode=bars.adjustment_mode,
    )
    values = evaluated.values
    series = FactorSeries(
        candidate_signal_id=candidate.candidate_signal_id, factor_identity=candidate.factor_identity,
        instrument=spec.instrument, expression=evaluated.expression, feature_name=evaluated.feature_name,
        feature_engine_version=evaluated.metadata.version, price_domain=bars.price_domain,
        adjustment_mode=bars.adjustment_mode, data_role=bars.data_role, window_start=start,
        window_end_exclusive=end, days=tuple(frame["trading_day"]),
        values=tuple(None if not np.isfinite(v) else float(v) for v in values),
        data_fingerprint="bars1:" + _sha(frame[list(_BAR_COLUMNS)].to_csv(index=False)),
        values_fingerprint="factorvals1:" + _sha(json.dumps([None if not np.isfinite(v) else repr(float(v))
                                                               for v in values])),
        source_fingerprint=evaluated.source_fingerprint, input_provenance=bars.provenance,
        qa_issues=evaluated.qa_issues,
    )
    diagnostics = diagnose_factor(
        days=frame["trading_day"], factor=values, open_=frame["open"].to_numpy(), close=frame["close"].to_numpy(),
        expression=spec.expression.render(), expected_sign=spec.expected_relationship.sign,
        declared_horizon_days=spec.prediction_horizon.days, window_start=start, window_end_exclusive=end,
        scope=DiagnosticScope.UNCONDITIONAL_FACTOR, policy=policy,
    )
    return CandidateScreen(
        candidate_signal_id=candidate.candidate_signal_id, factor_identity=candidate.factor_identity,
        expression=candidate.expression, series=series, diagnostics=diagnostics,
        screened_at=now or datetime.now(UTC),
    )


class CandidateScreenRun(BaseModel):
    """Candidates screened together: how many (the multiple-testing family
    validation must correct across) and how correlated their factors are."""

    model_config = {"frozen": True, "extra": "forbid"}

    candidate_set_fingerprint: str
    candidate_signal_ids: tuple[str, ...]
    family_size: int
    correlations: tuple[SignalCorrelation, ...]
    family_note: str = (
        "Screening applies no multiple-testing correction. Every candidate screened here counts toward the family "
        "that validation must correct across -- including those that did not pass the screen."
    )


def correlations_for(screens: list[CandidateScreen], names: dict[str, str]) -> tuple[SignalCorrelation, ...]:
    return signal_correlations({names[s.candidate_signal_id]: s.series.as_series() for s in screens})


def screen_candidates(
    candidates: CandidateSignalSet,
    *,
    loader: DailyBarLoader = load_daily_bars,
    policy: FactorScreenPolicy | None = None,
    store: FactorScreenStore | None = None,
    progress: Callable[[CandidateSignal], None] | None = None,
) -> tuple[tuple[CandidateScreen, ...], CandidateScreenRun]:
    """Screen every candidate of a set (reusing a stored screen of the same
    structural candidate when one exists)."""
    screens = []
    for c in candidates.candidates:
        cached = store.load(c) if store is not None else None
        if cached is None:
            if progress is not None:
                progress(c)
            cached = screen_candidate(c, loader=loader, policy=policy)
            if store is not None and cached.series.data_role is DataProvenanceRole.REAL:
                store.save(cached)
        screens.append(cached)
    names = {c.candidate_signal_id: f"{c.spec.instrument} {c.expression} → {c.spec.prediction_horizon.value}"
             for c in candidates.candidates}
    run = CandidateScreenRun(
        candidate_set_fingerprint=candidates.fingerprint(),
        candidate_signal_ids=tuple(c.candidate_signal_id for c in candidates.candidates),
        family_size=len(candidates.candidates), correlations=correlations_for(screens, names),
    )
    return tuple(screens), run


# ---------------------------------------------------------------------------
# operational cache
# ---------------------------------------------------------------------------


class FactorScreenStore:
    """JSON screens keyed by structural candidate id (gitignored,
    `data/news_alpha/factor_screens/`). An operational cache: safe to delete,
    never registry truth, never alpha memory. Only REAL-data screens are
    stored."""

    def __init__(self, root: Path = DEFAULT_SCREEN_DIR) -> None:
        self.root = Path(root)

    def _path(self, candidate_signal_id: str) -> Path:
        return self.root / (candidate_signal_id.replace(":", "_") + ".json")

    def save(self, screen: CandidateScreen) -> Path:
        if screen.series.data_role is not DataProvenanceRole.REAL:
            raise ValueError("only a screen of REAL data may be stored -- synthetic data is never evidence")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(screen.candidate_signal_id)
        path.write_text(screen.model_dump_json(indent=1), encoding="utf-8")
        return path

    def load(self, candidate: CandidateSignal) -> CandidateScreen | None:
        """The stored screen of exactly this candidate and expression under
        the CURRENT screening method, or ``None`` (missing, unreadable, other
        schema, another formula, or screened under another policy/rule --
        stale evidence is recomputed, never shown as current)."""
        path = self._path(candidate.candidate_signal_id)
        if not path.exists():
            return None
        try:
            screen = CandidateScreen.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            return None
        if screen.expression != candidate.expression or screen.factor_identity != candidate.factor_identity:
            return None
        d = screen.diagnostics
        if screen.schema_version != CANDIDATE_SCREEN_SCHEMA or (d.schema_version, d.rule, d.policy_fingerprint) != (
            FACTOR_DIAGNOSTICS_SCHEMA, FACTOR_SCREEN_RULE, FactorScreenPolicy().fingerprint(),
        ):
            return None
        return screen

    def load_many(self, candidates: CandidateSignalSet) -> dict[str, CandidateScreen]:
        out = {}
        for c in candidates.candidates:
            s = self.load(c)
            if s is not None:
                out[c.candidate_signal_id] = s
        return out
