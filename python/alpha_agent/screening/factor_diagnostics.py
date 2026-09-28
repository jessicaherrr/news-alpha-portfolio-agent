"""Factor diagnostics -- SCREENING evidence for one factor expression on one
instrument's history. Generic: it knows nothing about news, paths or
candidates; it receives factor values, the prices they are tested against,
and the predeclared sign and horizon.

SCOPE (`DiagnosticScope`). Every result says what it is evidence OF. This
engine produces only UNCONDITIONAL_FACTOR evidence: the factor on every day
of the window, with no knowledge of any event. EVENT_CONDITIONED evidence
(the factor's behaviour after comparable events) needs an event history and
is refused here, never approximated.

IC KIND (`ICKind`). For one instrument every IC here is a TIME-SERIES
correlation -- the factor against the SAME instrument's own forward return,
across dates. It is not a cross-sectional Rank IC (factor ranks against
forward-return ranks across instruments on one date, averaged over dates),
which needs a universe and is reserved, never computed here. Field names
say so: ``ts_spearman_ic`` / ``ts_pearson_ic``.

What it answers, for a single-instrument TIME-SERIES factor:

* TS Pearson / TS Spearman IC -- the correlation over time between the
  factor at day t and the forward return that starts at the NEXT permitted execution
  point: ``fwd_h(t) = close[t+h] / open[t+1] - 1`` (a signal read at bar t's
  close is acted on no earlier than bar t+1's open);
* t -- the TS Spearman IC's t on its EFFECTIVE sample: overlapping h-day forward
  returns carry about ``n / h`` independent windows, so
  ``t = r * sqrt(n/h - 2) / sqrt(1 - r^2)``. Under no relationship the
  variance of r is about ``(1/n) * sum_k rho_x(k) rho_y(k)``, and with
  ``|rho_x| <= 1`` that is at most ``h / n`` -- so this t is conservative
  whatever the factor's own persistence (a naive t on n daily pairs
  overstates the evidence about sqrt(h) times). Newey-West with a Bartlett
  kernel at lag h was tried first and REJECTED: its weights recover only
  about 2/3 of the long-run variance of overlapping returns, and on real
  XLU bars it turned a 60-day TS Spearman IC of -0.30 resting on ~18 independent
  windows into t = -3.2 (this t: -1.25);
* decay -- the same at 1 / 5 / 20 / 60 days;
* subperiod stability -- TS Spearman IC per calendar year, the share of
  years with the expected sign, and their mean/std across years (NOT a
  cross-sectional ICIR, which averages per-date ICs across a universe);
* coverage -- defined days, warm-up, and any undefined day after it (never
  filled);
* turnover -- factor autocorrelation, sign flips per year;
* cost sensitivity -- for a +/-1 sign position rebalanced every h days: the
  gross edge, the part of it that is mere drift (the average position times
  the average forward return -- a mostly-long position on a rising market
  earns it whatever the factor says), the TIMING edge left, and the one-way
  break-even cost that would consume that timing edge. Descriptive only --
  not a backtest, not PnL (official PnL and fills belong to the C++ engine).

Group / sector concentration is NOT APPLICABLE to a single-instrument
factor (there is no cross-section) and is reported as such, never faked.

SCREEN (``factor-screen/1``, `FactorScreenPolicy`, predeclared): a screen
says whether a candidate deserves deeper validation -- never that it is
valid. SCREEN_CONTINUE needs the TS Spearman IC's t at the declared horizon to
clear the threshold IN THE PREDECLARED DIRECTION and most yearly subperiods
to agree; an equally strong opposite sign is CONTRADICTS_EXPECTED_SIGN (the
factor is never re-signed); too little data is INSUFFICIENT_DATA. No
p-value, q-value, DSR, Sharpe verdict or registry write exists here;
multiple-testing correction happens at validation. A diagnostic is keyed
to one factor, not to the run that computed it, so the number of candidates
screened together is recorded by the caller's run, not here.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from alpha_agent.registry.holdout_guard import HOLDOUT_START, HoldoutAccessError

__all__ = [
    "DIAGNOSTIC_HORIZONS",
    "FACTOR_DIAGNOSTICS_SCHEMA",
    "FACTOR_SCREEN_RULE",
    "FORWARD_RETURN_CONVENTION",
    "CostSensitivity",
    "CoverageDiagnostics",
    "DiagnosticScope",
    "FactorDiagnostics",
    "FactorScreenPolicy",
    "HorizonDiagnostics",
    "ICKind",
    "ScreenReason",
    "ScreenStatus",
    "SignalCorrelation",
    "SubperiodDiagnostics",
    "TurnoverDiagnostics",
    "diagnose_factor",
    "forward_returns",
    "signal_correlations",
]

FACTOR_DIAGNOSTICS_SCHEMA = "factor-diagnostics/2"
FACTOR_SCREEN_RULE = "factor-screen/1"
DIAGNOSTIC_HORIZONS: tuple[int, ...] = (1, 5, 20, 60)
FORWARD_RETURN_CONVENTION = (
    "fwd_h(t) = close[t+h] / open[t+1] - 1: the factor read at bar t's close is acted on no earlier than bar t+1's "
    "open and held h bars. Pairs whose forward window leaves the evaluation window are dropped, never truncated."
)
_TRADING_DAYS_PER_YEAR = 252
_MAX_ABS_R = 1 - 1e-9
_HOLDOUT = date.fromisoformat(HOLDOUT_START)
_NOT_VALIDATION = (
    "Screening diagnostics on the discovery window only -- not a validation, verdict, expected return or trade. "
    "Validation (walk-forward, nulls, multiple-testing correction) belongs to the registry's validation authority."
)


class FactorScreenPolicy(BaseModel):
    """``factor-screen/1`` thresholds -- fixed before any diagnostic runs."""

    model_config = {"frozen": True, "extra": "forbid"}

    min_pairs: int = 250
    min_subperiod_pairs: int = 60
    min_evaluable_subperiods: int = 3
    t_threshold: float = 2.0
    min_subperiod_sign_share: float = 0.6

    def fingerprint(self) -> str:
        return "screenpolicy1:" + hashlib.sha256(
            json.dumps({"rule": FACTOR_SCREEN_RULE, **self.model_dump()}, sort_keys=True).encode()
        ).hexdigest()


class DiagnosticScope(str, Enum):
    """What the evidence is evidence OF."""

    #: The factor on every day of the window -- knows nothing of any event.
    UNCONDITIONAL_FACTOR = "UNCONDITIONAL_FACTOR"
    #: The factor after comparable events only. Needs an event history; not
    #: produced by this engine.
    EVENT_CONDITIONED = "EVENT_CONDITIONED"


class ICKind(str, Enum):
    #: One instrument's factor vs its own forward return, across dates.
    TIME_SERIES = "TIME_SERIES"
    #: Factor ranks vs forward-return ranks across instruments per date.
    #: Needs a universe; reserved, never computed here.
    CROSS_SECTIONAL = "CROSS_SECTIONAL"


class ScreenStatus(str, Enum):
    #: Enough screening support to continue into validation -- NOT validated.
    SCREEN_CONTINUE = "SCREEN_CONTINUE"
    NO_SCREEN_SUPPORT = "NO_SCREEN_SUPPORT"
    CONTRADICTS_EXPECTED_SIGN = "CONTRADICTS_EXPECTED_SIGN"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ScreenReason(str, Enum):
    TOO_FEW_PAIRS = "TOO_FEW_PAIRS"
    TOO_FEW_SUBPERIODS = "TOO_FEW_SUBPERIODS"
    UNDEFINED_STATISTIC = "UNDEFINED_STATISTIC"
    EXPECTED_SIGN_CLEARS_THRESHOLD = "EXPECTED_SIGN_CLEARS_THRESHOLD"
    OPPOSITE_SIGN_CLEARS_THRESHOLD = "OPPOSITE_SIGN_CLEARS_THRESHOLD"
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    SUBPERIODS_DISAGREE = "SUBPERIODS_DISAGREE"


class HorizonDiagnostics(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    horizon_days: int
    n_pairs: int
    #: ``n_pairs / horizon`` -- roughly how many non-overlapping windows the
    #: evidence rests on.
    n_independent: float
    #: Pearson correlation over time with the same instrument's forward return.
    ts_pearson_ic: float | None
    #: Spearman correlation over time -- NOT a cross-sectional Rank IC.
    ts_spearman_ic: float | None
    #: t of the TS Spearman IC on ``n_independent`` effective observations.
    ts_spearman_t: float | None


class SubperiodDiagnostics(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    label: str
    n_pairs: int
    ts_spearman_ic: float | None
    evaluable: bool


class CoverageDiagnostics(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    window_start: date
    window_end_exclusive: date
    n_days: int
    n_factor_defined: int
    coverage: float
    first_defined_day: date | None
    last_day: date | None
    #: Days with no factor value AFTER the first defined one -- an invalid
    #: input, never forward-filled.
    undefined_after_warmup: int


class TurnoverDiagnostics(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    autocorrelation_1d: float | None
    sign_flips_per_year: float | None
    #: Mean |change| of the +/-1 sign position per day (0 = never flips, 2 =
    #: flips every day).
    mean_daily_position_change: float | None


class CostSensitivity(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    horizon_days: int
    #: Mean forward return of the +/-1 position (expected-sign aligned), bps
    #: per holding period, before any cost. Includes drift: a mostly-long
    #: position on a rising market earns it whatever the factor says.
    gross_edge_bps: float | None
    #: The part of the gross edge an unconditional position of the same
    #: average size would earn (mean position x mean forward return).
    drift_bps: float | None
    #: Gross minus drift -- what the factor's switching adds.
    timing_edge_bps: float | None
    #: Mean |position change| per h-day rebalance (0..2).
    position_change_per_rebalance: float | None
    #: One-way cost (bps of notional) that would consume the TIMING edge;
    #: ``None`` when there is no positive timing edge or no turnover.
    break_even_cost_bps: float | None
    note: str


class SignalCorrelation(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    a: str
    b: str
    n_common: int
    correlation: float | None


class FactorDiagnostics(BaseModel):
    """Screening diagnostics for the EXACT expression named in `expression`."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = FACTOR_DIAGNOSTICS_SCHEMA
    rule: str = FACTOR_SCREEN_RULE
    plane: Literal["SCREENING"] = "SCREENING"
    scope: Literal[DiagnosticScope.UNCONDITIONAL_FACTOR] = DiagnosticScope.UNCONDITIONAL_FACTOR
    ic_kind: Literal[ICKind.TIME_SERIES] = ICKind.TIME_SERIES
    scope_note: str = (
        "Unconditional factor evidence: the factor on every day of the window, not its behaviour after this or "
        "any comparable event. Not event-conditioned evidence."
    )
    policy: FactorScreenPolicy
    policy_fingerprint: str
    expression: str
    expected_sign: Literal[1, -1]
    declared_horizon_days: int
    forward_return_convention: str = FORWARD_RETURN_CONVENTION
    coverage: CoverageDiagnostics
    horizons: tuple[HorizonDiagnostics, ...]
    subperiods: tuple[SubperiodDiagnostics, ...]
    #: Mean / std of the yearly TS Spearman ICs (not a cross-sectional ICIR).
    ts_spearman_ir_yearly: float | None
    subperiod_sign_share: float | None
    turnover: TurnoverDiagnostics
    cost: CostSensitivity
    concentration_note: str = (
        "Not applicable: a single-instrument time-series factor has no cross-section, so there is no group or "
        "sector concentration to measure."
    )
    status: ScreenStatus
    reasons: tuple[ScreenReason, ...]
    status_note: str
    not_validation_note: str = _NOT_VALIDATION

    @model_validator(mode="after")
    def _declared_is_diagnosed(self) -> FactorDiagnostics:
        if self.declared_horizon_days not in {h.horizon_days for h in self.horizons}:
            raise ValueError("the declared horizon must be among the diagnosed horizons")
        if self.coverage.window_end_exclusive > _HOLDOUT:
            raise ValueError("diagnostics may never cover the locked holdout")
        return self

    @property
    def declared(self) -> HorizonDiagnostics:
        return next(h for h in self.horizons if h.horizon_days == self.declared_horizon_days)


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


def forward_returns(open_: np.ndarray, close: np.ndarray, horizon: int) -> np.ndarray:
    """``close[t+h] / open[t+1] - 1``; NaN where the window leaves the data or
    the entry price is not positive."""
    n = len(close)
    out = np.full(n, np.nan)
    if n > horizon:
        entry = open_[1:n - horizon + 1]
        exit_ = close[horizon:]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[: n - horizon] = np.where(entry > 0, exit_ / entry - 1.0, np.nan)
    return out


def _finite(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return _finite(np.corrcoef(x, y)[0, 1])


def _ranks(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy()


def _effective_t(r: float | None, n_pairs: int, h: int) -> float | None:
    """The correlation t on ``n_pairs / h`` effective observations --
    conservative for overlapping h-day forward returns (module docstring)."""
    n_eff = n_pairs / h
    if r is None or n_eff <= 2:
        return None
    r = max(-_MAX_ABS_R, min(_MAX_ABS_R, r))  # a perfect correlation clears any threshold, finitely
    return _finite(r * math.sqrt(n_eff - 2) / math.sqrt(1 - r * r))


def _pairs(factor: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    return np.isfinite(factor) & np.isfinite(fwd)


def _horizon(factor: np.ndarray, fwd: np.ndarray, h: int) -> HorizonDiagnostics:
    ok = _pairs(factor, fwd)
    x, y = factor[ok], fwd[ok]
    rx, ry = _ranks(x), _ranks(y)
    return HorizonDiagnostics(
        horizon_days=h, n_pairs=int(ok.sum()), n_independent=round(ok.sum() / h, 2),
        ts_pearson_ic=_pearson(x, y), ts_spearman_ic=(rho := _pearson(rx, ry)),
        ts_spearman_t=_effective_t(rho, int(ok.sum()), h),
    )


def _subperiods(
    days: pd.Series, factor: np.ndarray, fwd: np.ndarray, policy: FactorScreenPolicy,
) -> tuple[SubperiodDiagnostics, ...]:
    ok = _pairs(factor, fwd)
    years = pd.to_datetime(days).dt.year.to_numpy()
    out = []
    for year in sorted(set(years.tolist())):
        sel = ok & (years == year)
        n = int(sel.sum())
        rho = _pearson(_ranks(factor[sel]), _ranks(fwd[sel])) if n >= 3 else None
        out.append(SubperiodDiagnostics(
            label=str(year), n_pairs=n, ts_spearman_ic=rho,
            evaluable=n >= policy.min_subperiod_pairs and rho is not None,
        ))
    return tuple(out)


def _positions(factor: np.ndarray, sign: int) -> np.ndarray:
    """The expected-sign-aligned +/-1 position (0 at an exact zero, NaN when
    undefined)."""
    return np.where(np.isfinite(factor), sign * np.sign(factor), np.nan)


def _turnover(factor: np.ndarray, sign: int) -> TurnoverDiagnostics:
    ok = np.isfinite(factor[1:]) & np.isfinite(factor[:-1])
    if ok.sum() < 3:
        return TurnoverDiagnostics(autocorrelation_1d=None, sign_flips_per_year=None, mean_daily_position_change=None)
    pos = _positions(factor, sign)
    change = np.abs(pos[1:] - pos[:-1])[ok]
    return TurnoverDiagnostics(
        autocorrelation_1d=_pearson(factor[1:][ok], factor[:-1][ok]),
        sign_flips_per_year=round(float((change > 0).sum()) / (ok.sum() / _TRADING_DAYS_PER_YEAR), 2),
        mean_daily_position_change=round(float(change.mean()), 4),
    )


def _cost(factor: np.ndarray, fwd: np.ndarray, sign: int, h: int) -> CostSensitivity:
    pos = _positions(factor, sign)
    ok = _pairs(factor, fwd)
    both = np.isfinite(pos[h:]) & np.isfinite(pos[:-h]) if len(pos) > h else np.zeros(0, bool)
    note = (
        f"A +/-1 position in the expected direction, rebalanced every {h} day(s). Drift (what holding the average "
        "position would earn anyway) is split out; the break-even is the one-way cost that consumes the timing "
        "edge. Descriptive -- not a backtest, fills or PnL."
    )
    if ok.sum() < 3 or not both.sum():
        return CostSensitivity(horizon_days=h, gross_edge_bps=None, drift_bps=None, timing_edge_bps=None,
                               position_change_per_rebalance=None, break_even_cost_bps=None,
                               note=note + " Not enough data.")
    p, y = pos[ok], fwd[ok]
    gross = float(np.mean(p * y)) * 1e4
    drift = float(np.mean(p) * np.mean(y)) * 1e4
    timing = gross - drift
    change = float(np.mean(np.abs(pos[h:] - pos[:-h])[both]))
    if timing <= 0:
        be, note = None, note + " No positive timing edge in the expected direction, so no cost can be afforded."
    elif change == 0:
        be, note = None, note + " The position never changes over the window, so turnover cost is not the constraint."
    else:
        be = timing / change
    return CostSensitivity(
        horizon_days=h, gross_edge_bps=round(gross, 3), drift_bps=round(drift, 3), timing_edge_bps=round(timing, 3),
        position_change_per_rebalance=round(change, 4), break_even_cost_bps=round(be, 3) if be is not None else None,
        note=note,
    )


def _status(
    declared: HorizonDiagnostics, subperiods: tuple[SubperiodDiagnostics, ...], sign: int, share: float | None,
    policy: FactorScreenPolicy,
) -> tuple[ScreenStatus, tuple[ScreenReason, ...], str]:
    evaluable = sum(s.evaluable for s in subperiods)
    reasons = []
    if declared.n_pairs < policy.min_pairs:
        reasons.append(ScreenReason.TOO_FEW_PAIRS)
    if evaluable < policy.min_evaluable_subperiods:
        reasons.append(ScreenReason.TOO_FEW_SUBPERIODS)
    if declared.ts_spearman_t is None:
        reasons.append(ScreenReason.UNDEFINED_STATISTIC)
    if reasons:
        return ScreenStatus.INSUFFICIENT_DATA, tuple(reasons), (
            f"{declared.n_pairs} factor/forward-return pairs and {evaluable} evaluable yearly subperiods at "
            f"{declared.horizon_days}D (needs {policy.min_pairs} and {policy.min_evaluable_subperiods})."
        )
    t = declared.ts_spearman_t * sign
    thr = policy.t_threshold
    if t <= -thr:
        return ScreenStatus.CONTRADICTS_EXPECTED_SIGN, (ScreenReason.OPPOSITE_SIGN_CLEARS_THRESHOLD,), (
            f"TS Spearman IC {declared.ts_spearman_ic:+.3f} (t {declared.ts_spearman_t:+.2f}) at {declared.horizon_days}D is "
            "the OPPOSITE of the predeclared sign -- the hypothesis as stated is contradicted; the factor is not "
            "re-signed."
        )
    agree = share is not None and share >= policy.min_subperiod_sign_share
    if t >= thr and agree:
        return ScreenStatus.SCREEN_CONTINUE, (ScreenReason.EXPECTED_SIGN_CLEARS_THRESHOLD,), (
            f"TS Spearman IC {declared.ts_spearman_ic:+.3f} (t {declared.ts_spearman_t:+.2f}) at {declared.horizon_days}D in "
            f"the predeclared direction, and {share:.0%} of yearly subperiods agree -- worth deeper validation, "
            "not validated."
        )
    if t < thr:
        reasons.append(ScreenReason.BELOW_THRESHOLD)
    if not agree:
        reasons.append(ScreenReason.SUBPERIODS_DISAGREE)
    return ScreenStatus.NO_SCREEN_SUPPORT, tuple(reasons), (
        f"TS Spearman IC {declared.ts_spearman_ic:+.3f} (t {declared.ts_spearman_t:+.2f}) at {declared.horizon_days}D"
        + (f"; {share:.0%} of yearly subperiods share the predeclared sign" if share is not None else "")
        + f" -- below the screen (t >= {thr:g} in the predeclared direction, >= "
        f"{policy.min_subperiod_sign_share:.0%} of years agreeing)."
    )


def diagnose_factor(
    *,
    days: pd.Series,
    factor: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    expression: str,
    expected_sign: int,
    declared_horizon_days: int,
    window_start: date,
    window_end_exclusive: date,
    scope: DiagnosticScope = DiagnosticScope.UNCONDITIONAL_FACTOR,
    policy: FactorScreenPolicy | None = None,
    horizons: tuple[int, ...] = DIAGNOSTIC_HORIZONS,
) -> FactorDiagnostics:
    """Diagnose one factor series. ``days`` are the trading days (ascending,
    unique) the arrays are aligned to; every one must lie inside
    ``[window_start, window_end_exclusive)`` -- a day at or past the window
    end (or in the holdout) raises rather than being silently dropped."""
    if scope is not DiagnosticScope.UNCONDITIONAL_FACTOR:
        raise ValueError(
            f"{scope.value} diagnostics need a history of comparable events; this engine only produces "
            "UNCONDITIONAL_FACTOR evidence and never approximates event-conditioned evidence with it"
        )
    policy = policy or FactorScreenPolicy()
    d = pd.to_datetime(pd.Series(days)).dt.date.reset_index(drop=True)
    if len(d) and (d.min() < window_start or d.max() >= window_end_exclusive):
        raise ValueError(f"days span {d.min()}..{d.max()}, outside the window [{window_start}, {window_end_exclusive})")
    if len(d) and d.max() >= _HOLDOUT:
        raise HoldoutAccessError(f"{d.max()} is inside the locked holdout (>= {HOLDOUT_START})")
    if not d.is_monotonic_increasing or d.duplicated().any():
        raise ValueError("days must be ascending and unique")
    if not len(d) == len(factor) == len(open_) == len(close):
        raise ValueError("days, factor, open and close must align")
    if expected_sign not in (1, -1):
        raise ValueError("expected_sign is +1 or -1")
    factor = np.asarray(factor, dtype="float64")
    open_, close = np.asarray(open_, dtype="float64"), np.asarray(close, dtype="float64")
    hs = tuple(sorted(set(horizons) | {declared_horizon_days}))

    fwds = {h: forward_returns(open_, close, h) for h in hs}
    horizon_diags = tuple(_horizon(factor, fwds[h], h) for h in hs)
    declared = next(h for h in horizon_diags if h.horizon_days == declared_horizon_days)
    subs = _subperiods(d, factor, fwds[declared_horizon_days], policy)
    ics = [s.ts_spearman_ic for s in subs if s.evaluable]
    share = (sum(np.sign(ic) == expected_sign for ic in ics) / len(ics)) if ics else None
    icir = _finite(np.mean(ics) / np.std(ics, ddof=1)) if len(ics) >= 2 and np.std(ics, ddof=1) > 0 else None
    status, reasons, note = _status(declared, subs, expected_sign, share, policy)

    defined = np.isfinite(factor)
    first = int(np.argmax(defined)) if defined.any() else None
    return FactorDiagnostics(
        policy=policy, policy_fingerprint=policy.fingerprint(), expression=expression, expected_sign=expected_sign,
        declared_horizon_days=declared_horizon_days,
        coverage=CoverageDiagnostics(
            window_start=window_start, window_end_exclusive=window_end_exclusive, n_days=len(d),
            n_factor_defined=int(defined.sum()), coverage=round(float(defined.mean()), 4) if len(d) else 0.0,
            first_defined_day=d.iloc[first] if first is not None else None,
            last_day=d.iloc[-1] if len(d) else None,
            undefined_after_warmup=int((~defined[first:]).sum()) if first is not None else 0,
        ),
        horizons=horizon_diags, subperiods=subs,
        ts_spearman_ir_yearly=round(icir, 4) if icir is not None else None,
        subperiod_sign_share=round(share, 4) if share is not None else None,
        turnover=_turnover(factor, expected_sign),
        cost=_cost(factor, fwds[declared_horizon_days], expected_sign, declared_horizon_days),
        status=status, reasons=reasons, status_note=note,
    )


def signal_correlations(
    series: dict[str, pd.Series], *, method: Literal["pearson", "spearman"] = "pearson",
) -> tuple[SignalCorrelation, ...]:
    """Pairwise correlation of factor values on their common days (series
    indexed by trading day). Tells how many of a set's candidates are really
    distinct signals. ``spearman`` ranks each pair on its common days first:
    robust to a factor's outliers (CL's 20-day return reached +326% off the
    April 2020 near-zero prices, which alone pulled its Pearson correlation
    with USO's to 0.74 against a rank correlation of 0.97)."""
    keys = list(series)
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            joined = pd.concat([series[a], series[b]], axis=1, join="inner").dropna()
            x, y = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
            if method == "spearman":
                x, y = _ranks(x), _ranks(y)
            out.append(SignalCorrelation(a=a, b=b, n_common=len(joined), correlation=_pearson(x, y)))
    return tuple(out)
