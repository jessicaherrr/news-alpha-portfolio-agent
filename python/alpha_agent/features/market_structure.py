"""Phase 12 -- deterministic market-structure detector (ICT "Silver Bullet").

This module turns an informal discretionary trading idea into an explicit,
closed, point-in-time quantitative definition:

    liquidity reference -> liquidity sweep -> displacement -> fair value gap
        -> retracement into the FVG -> target-position intent

Every concept has a deterministic mathematical definition (sections 4-10 of
``prompts/12``). There is **no discretionary visual judgement**, no arbitrary
strategy code, and no look-ahead. The detector is a small closed state machine
whose per-bar outputs are registered :class:`FeatureFrame` columns; the closed
Phase 10 Strategy DSL only thresholds those columns into target intent.

Phase 12 makes **no** claim that this pattern is profitable, robust, or has
alpha -- Phase 13 reliability validation decides that.

Bar indexing / event ordering (section 10):

* ``T0`` -- a liquidity sweep bar closes back across the swept level.
* ``Td`` -- a displacement bar, **strictly after** ``T0`` (one OHLC bar cannot
  prove the sweep low happened before the displacement close -- section 16).
* ``Tf = Td + 1`` -- the first bar at which the 3-bar FVG ``(Td-1, Td, Td+1)``
  is fully observable. The feature is stamped at ``Tf``, never ``Td``.
* ``Tr`` -- a retracement bar, **strictly after** ``Tf``: price trades back to
  the configured depth inside the FVG. This is the StrategyDecision bar.
* C++ execution happens at the engine's next eligible bar after ``Tr``.

Signed prices (section 17): every comparison is relational and every magnitude is
an arithmetic difference / true range -- valid for zero and negative prices
(a CL path through zero does not break the detector). No log / percentage return.

Rolls (section 18): the detector consumes a research price domain. On a raw
*continuous* feed it resets its state at an ``is_roll_boundary`` row so a roll
gap never manufactures a sweep or displacement. Execution still resolves to the
real ``RawContract`` in C++ (``make_fill`` guarantee, unchanged).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from alpha_agent.features.enums import FeatureFamily, SessionPolicy
from alpha_agent.features.qa import FeatureIssueKind
from alpha_agent.features.registry import FeatureComputeContext, ParamRule, feature
from alpha_agent.features.windows import seg_rolling, seg_shift, true_range


# ==========================================================================
# State machine
# ==========================================================================
class SilverBulletState(IntEnum):
    IDLE = 0
    SWEEP_OBSERVED = 1
    DISPLACEMENT_CONFIRMED = 2
    FVG_ACTIVE = 3
    RETRACEMENT_TRIGGERED = 4  # == in a position, until a typed exit returns to IDLE
    EXPIRED = 5                # transient: a stale setup that reset to IDLE this bar


SILVER_BULLET_STATE_CODES: dict[str, int] = {s.name: int(s) for s in SilverBulletState}

# Closed distance-unit vocabulary for every Phase 12 price-distance threshold
# (sweep_penetration, fvg_min_width). There is NO silent fallback: TICKS requires
# a finite tick_size > 0 obtained from typed metadata, PRICE_UNITS needs none.
DISTANCE_UNITS = ("price_units", "ticks")

# rolling_nbar_extreme -- prior N *bars*; prior_session_extreme -- prior
# *trading day* (requires reliable session/trading-day labels, no gap-segment
# substitution).
LIQUIDITY_REFERENCES = ("rolling_nbar_extreme", "prior_session_extreme")


# ==========================================================================
# Typed parameter contract
# ==========================================================================
@dataclass(frozen=True)
class SilverBulletDetectorParams:
    """Fully-resolved, validated scalar parameters for one detector run.

    Distance thresholds (``sweep_penetration`` / ``fvg_min_width``) carry an
    **explicit** unit (``distance_unit``):

    * ``price_units`` -- the value is already in normalized price units; no tick
      conversion, no tick metadata needed.
    * ``ticks`` -- the value is a multiple of ``tick_size``, which **must** be a
      finite number > 0 supplied via typed metadata. A missing / non-finite /
      non-positive ``tick_size`` is a loud error -- it is never silently treated
      as ``1.0``.

    The ``ContractSpec`` remains the sole authority for real execution economics;
    ``tick_size`` here is causal research metadata only.
    """

    liquidity_lookback: int
    liquidity_reference: str
    sweep_penetration: float
    distance_unit: str
    tick_size: float
    displacement_atr_window: int
    displacement_atr_multiple: float
    displacement_close_loc: float
    fvg_min_width: float
    retracement_fraction: float
    setup_expiry_bars: int
    max_holding_bars: int
    window_start_local: str
    window_end_local: str
    window_calendar: str

    def __post_init__(self) -> None:
        msg = distance_unit_error(self.distance_unit, self.tick_size)
        if msg:
            raise ValueError(msg)

    @property
    def unit_price(self) -> float:
        """Price-unit value of one threshold unit. ``ticks`` -> ``tick_size``
        (validated > 0); ``price_units`` -> ``1.0`` (identity, by definition --
        not a fallback)."""
        if self.distance_unit == "ticks":
            return float(self.tick_size)
        return 1.0

    @property
    def sweep_penetration_price(self) -> float:
        return self.sweep_penetration * self.unit_price

    @property
    def fvg_min_width_price(self) -> float:
        return self.fvg_min_width * self.unit_price

    @property
    def has_window(self) -> bool:
        return bool(self.window_start_local and self.window_end_local)

    def fingerprint(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return "sbparams1:" + hashlib.sha256(payload.encode()).hexdigest()[:32]


def distance_unit_error(distance_unit: str, tick_size: float) -> str | None:
    """Validate the explicit distance-unit contract. Returns an error message or
    ``None``. Used by both the registry parameter contract and the dataclass."""
    if distance_unit not in DISTANCE_UNITS:
        return f"distance_unit {distance_unit!r} must be one of {list(DISTANCE_UNITS)}"
    if distance_unit == "ticks":
        if tick_size is None or not math.isfinite(tick_size) or tick_size <= 0.0:
            return (
                "distance_unit='ticks' requires a finite tick_size > 0 (from typed "
                f"contract/tick metadata); got tick_size={tick_size!r}. Tick size is "
                "never assumed -- pass distance_unit='price_units' to express "
                "sweep_penetration / fvg_min_width directly in normalized price units."
            )
    elif tick_size not in (0.0, None):
        return (
            f"distance_unit='price_units' does not use tick_size (got {tick_size!r}); "
            "leave it unset, or use distance_unit='ticks'"
        )
    return None


_SB_PARAM_RULES: dict[str, ParamRule] = {
    "liquidity_lookback": ParamRule(int, min=2, max=100_000),
    "liquidity_reference": ParamRule(str, required=False, default="rolling_nbar_extreme"),
    "sweep_penetration": ParamRule(float, min=0.0, max=1.0e12, required=False, default=0.0),
    "distance_unit": ParamRule(str, required=False, default="price_units"),
    "tick_size": ParamRule(float, min=0.0, max=1.0e9, required=False, default=0.0),
    "displacement_atr_window": ParamRule(int, min=2, max=100_000),
    "displacement_atr_multiple": ParamRule(float, min=0.0, max=1.0e6),
    "displacement_close_loc": ParamRule(float, min=0.0, max=1.0),
    "fvg_min_width": ParamRule(float, min=0.0, max=1.0e12, required=False, default=0.0),
    "retracement_fraction": ParamRule(float, min=0.0, max=1.0),
    "setup_expiry_bars": ParamRule(int, min=1, max=100_000),
    "max_holding_bars": ParamRule(int, min=1, max=100_000),
    "window_start_local": ParamRule(str, required=False, default=""),
    "window_end_local": ParamRule(str, required=False, default=""),
    "window_calendar": ParamRule(str, required=False, default=""),
}

_SB_PARAM_ORDER: tuple[str, ...] = tuple(_SB_PARAM_RULES)


def _sb_constraints(p: dict) -> str | None:
    if p["liquidity_reference"] not in LIQUIDITY_REFERENCES:
        return (
            f"liquidity_reference {p['liquidity_reference']!r} must be one of "
            f"{list(LIQUIDITY_REFERENCES)}"
        )
    unit_msg = distance_unit_error(p["distance_unit"], p["tick_size"])
    if unit_msg:
        return unit_msg
    if p["displacement_atr_multiple"] <= 0.0:
        return "displacement_atr_multiple must be > 0"
    starts, ends = p["window_start_local"], p["window_end_local"]
    if bool(starts) != bool(ends):
        return "window_start_local and window_end_local must be set together (or both empty)"
    return None


def _resolved(params: dict) -> SilverBulletDetectorParams:
    return SilverBulletDetectorParams(**{k: params[k] for k in _SB_PARAM_ORDER})


def _sb_lookback(p: dict) -> int:
    # bars until the detector can first produce output: a prior liquidity
    # reference plus an ATR window. The full setup lifecycle after that is signal
    # latency, not warm-up.
    return int(p["liquidity_lookback"]) + int(p["displacement_atr_window"]) + 2


# ==========================================================================
# Audit record (section 20) -- no Fill price, no RiskDecision.
# ==========================================================================
class SilverBulletAudit(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str
    setup_direction: str  # "long" | "short"
    liquidity_reference_type: str
    liquidity_level: float
    distance_unit: str            # "price_units" | "ticks"
    distance_unit_price: float    # price-unit value of one threshold unit
    sweep_ts_ns: int
    displacement_ts_ns: int | None = None
    fvg_created_ts_ns: int | None = None
    fvg_lower: float | None = None
    fvg_upper: float | None = None
    retracement_ts_ns: int | None = None
    decision_ts_ns: int | None = None
    expiry_ts_ns: int | None = None
    expiry_reason: str = ""
    parameter_fingerprint: str
    diagnostics: list[str] = Field(default_factory=list)


@dataclass
class SilverBulletResult:
    ts_event_ns: np.ndarray
    intent: np.ndarray           # -1.0 / 0.0 / +1.0 -- the DSL-facing target sign
    state: np.ndarray            # SilverBulletState code per bar (diagnostic)
    long_entry: np.ndarray       # 1.0 on a bullish decision bar, else 0.0
    short_entry: np.ndarray      # 1.0 on a bearish decision bar, else 0.0
    exit_event: np.ndarray       # 1.0 on a bar an active setup exited, else 0.0
    audits: list[SilverBulletAudit] = field(default_factory=list)
    diagnostics: Counter = field(default_factory=Counter)

    def column(self, name: str) -> np.ndarray:
        return {
            "silver_bullet_intent": self.intent,
            "silver_bullet_state": self.state,
            "silver_bullet_long_entry": self.long_entry,
            "silver_bullet_short_entry": self.short_entry,
            "silver_bullet_exit": self.exit_event,
        }[name]


# ==========================================================================
# The detector
# ==========================================================================
_LONG, _SHORT = 1, -1


def detect_silver_bullet(
    frame: pd.DataFrame,
    segment_id: pd.Series | np.ndarray,
    params: SilverBulletDetectorParams,
    *,
    root_symbol: str,
    in_window: np.ndarray | None = None,
) -> SilverBulletResult:
    """Causal, deterministic Silver-Bullet state machine over one OHLC series.

    ``in_window[i]`` (optional) is True where bar ``i``'s exchange-local time is
    inside the configured Silver-Bullet trading window. When given, a retracement
    entry is only taken inside the window and an open position is flattened once
    the window closes.
    """
    for col in ("open", "high", "low", "close"):
        if col not in frame.columns:
            raise ValueError(f"silver_bullet detector needs an OHLC '{col}' column")

    n = len(frame)
    ts = pd.to_numeric(frame["ts_event_ns"], errors="raise").to_numpy("int64")
    o = pd.to_numeric(frame["open"], errors="coerce").to_numpy("float64")
    h = pd.to_numeric(frame["high"], errors="coerce").to_numpy("float64")
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy("float64")
    c = pd.to_numeric(frame["close"], errors="coerce").to_numpy("float64")
    seg = np.asarray(segment_id, dtype="int64")

    roll_reset = np.zeros(n, dtype=bool)
    if "is_roll_boundary" in frame.columns:
        roll_reset = frame["is_roll_boundary"].fillna(False).astype(bool).to_numpy()

    if in_window is None:
        in_window = np.ones(n, dtype=bool)
    else:
        in_window = np.asarray(in_window, dtype=bool)

    # -- vectorised, strictly-backward references -------------------------------
    hs = pd.Series(h)
    ls = pd.Series(low)
    cs = pd.Series(c)
    L = params.liquidity_lookback
    if params.liquidity_reference == "prior_session_extreme":
        ref_hi, ref_lo = _prior_session_extreme(frame, h, low)
    else:
        ref_hi = seg_shift(seg_rolling(hs, seg, L, L, "max"), seg, 1).to_numpy("float64")
        ref_lo = seg_shift(seg_rolling(ls, seg, L, L, "min"), seg, 1).to_numpy("float64")

    w = params.displacement_atr_window
    tr = true_range(hs, ls, cs, seg)
    atr = seg_rolling(tr, seg, w, w, "mean").to_numpy("float64")

    rng = h - low
    with np.errstate(invalid="ignore", divide="ignore"):
        close_loc = np.where(rng > 0.0, (c - low) / rng, 0.5)

    intent = np.zeros(n, dtype="float64")
    state = np.full(n, int(SilverBulletState.IDLE), dtype="float64")
    long_entry = np.zeros(n, dtype="float64")
    short_entry = np.zeros(n, dtype="float64")
    exit_event = np.zeros(n, dtype="float64")
    audits: list[SilverBulletAudit] = []
    diag: Counter = Counter()

    fp = params.fingerprint()
    pen = params.sweep_penetration_price
    min_w = params.fvg_min_width_price

    # -- state -----------------------------------------------------------------
    st = SilverBulletState.IDLE
    direction = 0
    sweep_level = np.nan
    sweep_bar = -1
    disp_bar = -1
    fvg_lo = fvg_hi = np.nan
    fvg_bar = -1
    pos_dir = 0
    entry_bar = -1
    cur: dict | None = None  # audit-in-progress

    def _reset(reason: str, i: int) -> None:
        nonlocal st, direction, sweep_level, sweep_bar, disp_bar, fvg_lo, fvg_hi, fvg_bar, cur
        if cur is not None and cur.get("decision_ts_ns") is None:
            cur["expiry_ts_ns"] = int(ts[i])
            cur["expiry_reason"] = reason
            cur["diagnostics"].append(reason)
            audits.append(SilverBulletAudit(**cur))
        st = SilverBulletState.IDLE
        direction = 0
        sweep_level = np.nan
        sweep_bar = disp_bar = fvg_bar = -1
        fvg_lo = fvg_hi = np.nan
        cur = None

    def _seg_last(i: int) -> bool:
        return i == n - 1 or seg[i + 1] != seg[i]

    for i in range(n):
        seg_start = i == 0 or seg[i] != seg[i - 1]

        # -- 1. hard resets: new session / data gap / roll boundary -----------
        if seg_start or roll_reset[i]:
            if pos_dir != 0:
                intent[i] = 0.0
                exit_event[i] = 1.0
                reason = "roll_reset" if (roll_reset[i] and not seg_start) else "session_reset"
                diag[f"exit_{reason}"] += 1
                if cur is not None:
                    cur["expiry_ts_ns"] = int(ts[i])
                    cur["expiry_reason"] = reason
                    cur["diagnostics"].append(reason)
                    audits.append(SilverBulletAudit(**cur))
                cur = None
                pos_dir = 0
                entry_bar = -1
            else:
                _reset("session_reset" if seg_start else "roll_reset", i)
                # no liquidity reference exists yet in the new segment -> NO
                # DECISION (an explicit flat only follows a real setup).
                intent[i] = np.nan
                state[i] = np.nan
            diag["roll_resets" if (roll_reset[i] and not seg_start) else "session_resets"] += 1
            st = SilverBulletState.IDLE
            # keep waiting for the next bar; a reset bar never opens a setup.
            continue

        # -- 2. in a position: only typed exits apply ------------------------
        if pos_dir != 0:
            held = i - entry_bar
            reason = ""
            if held >= params.max_holding_bars:
                reason = "max_holding_bars"
            elif params.has_window and not in_window[i]:
                reason = "window_end"
            elif _seg_last(i):
                reason = "session_end"
            elif pos_dir == _LONG and c[i] < sweep_level or pos_dir == _SHORT and c[i] > sweep_level:
                reason = "setup_invalidation"
            if reason:
                intent[i] = 0.0
                exit_event[i] = 1.0
                diag[f"exit_{reason}"] += 1
                if cur is not None:
                    cur["expiry_ts_ns"] = int(ts[i])
                    cur["expiry_reason"] = reason
                    audits.append(SilverBulletAudit(**cur))
                cur = None
                pos_dir = 0
                entry_bar = -1
                st = SilverBulletState.IDLE
            else:
                intent[i] = float(pos_dir)
                st = SilverBulletState.RETRACEMENT_TRIGGERED
            state[i] = int(st)
            continue

        # -- 3. setup state machine (flat) ----------------------------------
        if st == SilverBulletState.IDLE:
            have_ref = np.isfinite(ref_lo[i]) and np.isfinite(ref_hi[i])
            if not have_ref:
                # genuine insufficient history -> NO DECISION (NaN), not a flat.
                diag["bars_without_liquidity_reference"] += 1
                intent[i] = np.nan
                state[i] = np.nan
                continue
            sell_side = (low[i] < ref_lo[i] - pen) and (c[i] > ref_lo[i])   # -> bullish setup
            buy_side = (h[i] > ref_hi[i] + pen) and (c[i] < ref_hi[i])      # -> bearish setup
            if sell_side and buy_side:
                diag["ambiguous_intrabar_sweep"] += 1
                state[i] = int(st)
                continue
            if sell_side or buy_side:
                direction = _LONG if sell_side else _SHORT
                sweep_level = ref_lo[i] if sell_side else ref_hi[i]
                sweep_bar = i
                st = SilverBulletState.SWEEP_OBSERVED
                diag["setups_started"] += 1
                cur = {
                    "root_symbol": root_symbol,
                    "setup_direction": "long" if direction == _LONG else "short",
                    "liquidity_reference_type": params.liquidity_reference,
                    "liquidity_level": float(sweep_level),
                    "distance_unit": params.distance_unit,
                    "distance_unit_price": float(params.unit_price),
                    "sweep_ts_ns": int(ts[i]),
                    "parameter_fingerprint": fp,
                    "diagnostics": [],
                }
            state[i] = int(st)
            continue

        # past this point a setup is in progress -- check expiry first
        if i - sweep_bar > params.setup_expiry_bars and st in (
            SilverBulletState.SWEEP_OBSERVED,
            SilverBulletState.DISPLACEMENT_CONFIRMED,
        ):
            diag["expired_before_fvg"] += 1
            _reset("setup_expired_before_fvg", i)
            state[i] = int(SilverBulletState.EXPIRED)
            continue

        if st == SilverBulletState.SWEEP_OBSERVED:
            # displacement must be STRICTLY after the sweep bar (section 16)
            if i > sweep_bar and np.isfinite(atr[i]) and atr[i] > 0.0:
                body = c[i] - o[i]
                if direction == _LONG:
                    ok = (
                        body > 0.0
                        and body / atr[i] >= params.displacement_atr_multiple
                        and close_loc[i] >= params.displacement_close_loc
                    )
                else:
                    ok = (
                        -body > 0.0
                        and (-body) / atr[i] >= params.displacement_atr_multiple
                        and close_loc[i] <= 1.0 - params.displacement_close_loc
                    )
                if ok:
                    disp_bar = i
                    st = SilverBulletState.DISPLACEMENT_CONFIRMED
                    diag["reached_displacement"] += 1
                    if cur is not None:
                        cur["displacement_ts_ns"] = int(ts[i])
                else:
                    diag["displacement_bar_failed"] += 1
            state[i] = int(st)
            continue

        if st == SilverBulletState.DISPLACEMENT_CONFIRMED:
            # the 3-bar FVG (disp_bar-1, disp_bar, disp_bar+1) is first knowable
            # at bar disp_bar+1 -- stamp it there, never at disp_bar.
            if i == disp_bar + 1 and disp_bar - 1 >= 0 and seg[disp_bar - 1] == seg[i]:
                if direction == _LONG:
                    lo_b, hi_b = h[disp_bar - 1], low[i]
                else:
                    lo_b, hi_b = h[i], low[disp_bar - 1]
                width = hi_b - lo_b
                if width <= 0.0:
                    diag["fvg_absent"] += 1
                    _reset("fvg_absent", i)
                elif width < min_w:
                    diag["fvg_too_small"] += 1
                    _reset("fvg_too_small", i)
                else:
                    fvg_lo, fvg_hi, fvg_bar = float(lo_b), float(hi_b), i
                    st = SilverBulletState.FVG_ACTIVE
                    diag["reached_fvg"] += 1
                    if cur is not None:
                        cur["fvg_created_ts_ns"] = int(ts[i])
                        cur["fvg_lower"] = fvg_lo
                        cur["fvg_upper"] = fvg_hi
            elif i > disp_bar + 1:
                diag["fvg_absent"] += 1
                _reset("fvg_absent", i)
            state[i] = int(st)
            continue

        if st == SilverBulletState.FVG_ACTIVE:
            if i - fvg_bar > params.setup_expiry_bars:
                diag["expired_no_retracement"] += 1
                _reset("retracement_not_reached", i)
                state[i] = int(SilverBulletState.EXPIRED)
                continue
            # retracement STRICTLY after the FVG-confirm bar (section 8, checklist I)
            span = fvg_hi - fvg_lo
            if direction == _LONG:
                level = fvg_hi - params.retracement_fraction * span
                touched = low[i] <= level
            else:
                level = fvg_lo + params.retracement_fraction * span
                touched = h[i] >= level
            if touched:
                if params.has_window and not in_window[i]:
                    diag["retracement_blocked_by_window"] += 1
                    state[i] = int(st)
                    continue
                pos_dir = direction
                entry_bar = i
                intent[i] = float(pos_dir)
                st = SilverBulletState.RETRACEMENT_TRIGGERED
                diag["entries"] += 1
                diag["entries_long" if pos_dir == _LONG else "entries_short"] += 1
                if pos_dir == _LONG:
                    long_entry[i] = 1.0
                else:
                    short_entry[i] = 1.0
                if cur is not None:
                    cur["retracement_ts_ns"] = int(ts[i])
                    cur["decision_ts_ns"] = int(ts[i])
            state[i] = int(st)
            continue

        state[i] = int(st)

    # -- close a still-open audit at end of data -------------------------------
    if cur is not None:
        if cur.get("decision_ts_ns") is not None:
            cur["expiry_ts_ns"] = int(ts[-1])
            cur["expiry_reason"] = cur.get("expiry_reason") or "end_of_data"
        else:
            cur["expiry_ts_ns"] = int(ts[-1])
            cur["expiry_reason"] = "end_of_data_incomplete"
            cur["diagnostics"].append("insufficient_history")
        audits.append(SilverBulletAudit(**cur))

    return SilverBulletResult(
        ts_event_ns=ts,
        intent=intent,
        state=state,
        long_entry=long_entry,
        short_entry=short_entry,
        exit_event=exit_event,
        audits=audits,
        diagnostics=diag,
    )


class SilverBulletSessionLabelsMissing(ValueError):
    """``prior_session_extreme`` was requested but the source frame carries no
    reliable session / trading-day labels from the calendar architecture."""


def _prior_session_extreme(
    frame: pd.DataFrame, h: np.ndarray, low: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """High / low of the immediately-preceding **trading day**, carried forward
    inside the current day. Strictly causal: a bar on day ``D`` sees only the
    completed extremes of the most recent day ``D' < D``.

    Requires a reliable ``trading_day`` (or ``session``+``trading_day``) label
    column produced by the Phase 03/04 calendar architecture. There is **no**
    silent substitution of a gap-delimited segment -- a missing label is a loud
    error (``prompts/12.1`` section 5).
    """
    label_col = "trading_day" if "trading_day" in frame.columns else None
    if label_col is None:
        raise SilverBulletSessionLabelsMissing(
            "liquidity_reference='prior_session_extreme' requires a 'trading_day' "
            "label column (derived by the Phase 03/04 SessionCalendar). The source "
            "frame has none; a gap-delimited segment is NOT a valid substitute. "
            "Provide calendar-labelled bars, or use "
            "liquidity_reference='rolling_nbar_extreme'."
        )
    day = frame[label_col].astype("object").to_numpy()
    if pd.isna(day).any():
        raise SilverBulletSessionLabelsMissing(
            "liquidity_reference='prior_session_extreme': the 'trading_day' column "
            "has missing values; every bar needs a reliable session label."
        )
    n = len(day)
    ref_hi = np.full(n, np.nan)
    ref_lo = np.full(n, np.nan)
    prev_hi = prev_lo = np.nan
    cur_day = day[0] if n else None
    cur_hi = -np.inf
    cur_lo = np.inf
    for i in range(n):
        if day[i] != cur_day:
            prev_hi, prev_lo = cur_hi, cur_lo
            cur_day = day[i]
            cur_hi, cur_lo = -np.inf, np.inf
        if np.isfinite(prev_hi):
            ref_hi[i] = prev_hi
            ref_lo[i] = prev_lo
        cur_hi = max(cur_hi, h[i])
        cur_lo = min(cur_lo, low[i])
    return ref_hi, ref_lo


# ==========================================================================
# Feature-engine integration
# ==========================================================================
_DETECTOR_CACHE: dict[str, SilverBulletResult] = {}


def _cache_key(source_fingerprint: str, p: SilverBulletDetectorParams, root: str) -> str:
    return hashlib.sha256(
        (source_fingerprint + "|" + p.fingerprint() + "|" + root).encode()
    ).hexdigest()


def _window_mask(ctx: FeatureComputeContext, p: SilverBulletDetectorParams, root: str) -> np.ndarray:
    n = len(ctx.frame)
    if not p.has_window:
        return np.ones(n, dtype=bool)
    from alpha_agent.data.calendars import SessionCalendarMissing, default_calendar

    cal_root = p.window_calendar or root
    if not cal_root:
        raise ValueError(
            "silver_bullet session window requested but no root symbol is available; "
            "set the 'window_calendar' parameter to the calendar root"
        )
    try:
        cal = default_calendar()
        return np.array(
            [
                cal.in_local_window(int(t), cal_root, p.window_start_local, p.window_end_local)
                for t in ctx.frame["ts_event_ns"].to_numpy("int64")
            ],
            dtype=bool,
        )
    except (SessionCalendarMissing, FileNotFoundError, KeyError) as exc:
        raise ValueError(
            f"silver_bullet session window requested for calendar root {cal_root!r} but the "
            f"session/calendar architecture cannot express it ({exc}); report the limitation "
            "rather than inventing timestamps (prompt 12 section 11)"
        ) from exc


def _root_symbol(ctx: FeatureComputeContext) -> str:
    f = ctx.frame
    if "root_symbol" in f.columns:
        vals = {str(x) for x in f["root_symbol"].dropna().unique()}
        if len(vals) == 1:
            return next(iter(vals))
    return str(ctx.source.identity.get("root_symbol", ""))


def _run_detector(ctx: FeatureComputeContext) -> SilverBulletResult:
    p = _resolved(ctx.params)
    root = _root_symbol(ctx)
    key = _cache_key(ctx.source.fingerprint(), p, root)
    cached = _DETECTOR_CACHE.get(key)
    if cached is not None:
        return cached

    in_window = _window_mask(ctx, p, root)
    result = detect_silver_bullet(
        ctx.frame, ctx.segment_id, p, root_symbol=root, in_window=in_window
    )
    # Frequency diagnostics (section 19): record WHY setups did not become
    # entries rather than silently emitting a flat feature. The full per-setup
    # reason list is on the audit records; here we surface an aggregate note.
    d = result.diagnostics
    started = int(d.get("setups_started", 0))
    entries = int(d.get("entries", 0))
    if started and not entries:
        reasons = {
            k: int(v)
            for k, v in d.items()
            if k
            in (
                "displacement_bar_failed",
                "fvg_absent",
                "fvg_too_small",
                "expired_before_fvg",
                "expired_no_retracement",
                "retracement_blocked_by_window",
                "ambiguous_intrabar_sweep",
            )
            and v
        }
        ctx.qa.add(
            FeatureIssueKind.INSUFFICIENT_LOOKBACK,
            ctx.name,
            f"{started} Silver Bullet setup(s) started but 0 reached an entry; "
            f"rejection reasons={reasons}",
            count=started,
        )

    if len(_DETECTOR_CACHE) > 64:
        _DETECTOR_CACHE.clear()
    _DETECTOR_CACHE[key] = result
    return result


_MS_DOMAINS = None  # use the @feature default (RawContract / RawContinuous / BackAdjusted)


@feature(
    "silver_bullet_intent",
    family=FeatureFamily.MARKET_STRUCTURE,
    param_rules=_SB_PARAM_RULES,
    param_order=_SB_PARAM_ORDER,
    param_constraints=(_sb_constraints,),
    default_session_policy=SessionPolicy.RESET_ON_SESSION_AND_GAP,
    lookback=_sb_lookback,
    point_in_time_safe=True,
    description=(
        "ICT Silver Bullet market-structure benchmark: signed target-position intent "
        "(-1 / 0 / +1) from a causal sweep -> displacement -> FVG -> retracement state "
        "machine. Phase 12 benchmark, NOT claimed alpha."
    ),
)
def _silver_bullet_intent(ctx: FeatureComputeContext):
    return _run_detector(ctx).column("silver_bullet_intent")


@feature(
    "silver_bullet_state",
    family=FeatureFamily.MARKET_STRUCTURE,
    param_rules=_SB_PARAM_RULES,
    param_order=_SB_PARAM_ORDER,
    param_constraints=(_sb_constraints,),
    default_session_policy=SessionPolicy.RESET_ON_SESSION_AND_GAP,
    lookback=_sb_lookback,
    point_in_time_safe=True,
    description=(
        "ICT Silver Bullet setup lifecycle state code per bar "
        "(IDLE=0, SWEEP_OBSERVED=1, DISPLACEMENT_CONFIRMED=2, FVG_ACTIVE=3, "
        "RETRACEMENT_TRIGGERED=4, EXPIRED=5). Diagnostic; not a trading signal."
    ),
)
def _silver_bullet_state(ctx: FeatureComputeContext):
    return _run_detector(ctx).column("silver_bullet_state")


def silver_bullet_result(
    source, params: dict | SilverBulletDetectorParams
) -> SilverBulletResult:
    """Run the detector directly on a ``SourceSeries`` and return the full typed
    result (per-bar columns + audit records + frequency diagnostics). The DSL
    still consumes only the registered feature columns via ``compute_features``;
    this is for the Phase 12 report, benchmark diagnostics and tests.
    """
    from alpha_agent.features.registry import REGISTRY

    if isinstance(params, SilverBulletDetectorParams):
        p = params
    else:
        p = _resolved(REGISTRY.get("silver_bullet_intent").resolve_params(dict(params)))
    root = str(getattr(source, "identity", {}).get("root_symbol", ""))
    key = _cache_key(source.fingerprint(), p, root)
    cached = _DETECTOR_CACHE.get(key)
    if cached is not None:
        return cached
    seg = source.segment_ids(SessionPolicy.RESET_ON_SESSION_AND_GAP)
    in_window: np.ndarray | None = None
    if p.has_window:
        from alpha_agent.data.calendars import default_calendar

        cal = default_calendar()
        cal_root = p.window_calendar or root
        in_window = np.array(
            [
                cal.in_local_window(int(t), cal_root, p.window_start_local, p.window_end_local)
                for t in source.frame["ts_event_ns"].to_numpy("int64")
            ],
            dtype=bool,
        )
    return detect_silver_bullet(
        source.frame, seg, p, root_symbol=root, in_window=in_window
    )
