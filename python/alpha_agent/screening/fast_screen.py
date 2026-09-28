"""Alpha Discovery campaign, Part F -- the research-only C++ Fast Screen
(task spec sections 36-41).

"Fast" means a CHEAPER scientific workload, never fake PnL (task spec
section 38): every fast-screen trial is ONE real C++ backtest
(`quant_backtest_targets_csv`, the same frozen CLI / roll-aware runner /
target-schedule builders `alpha_agent.agents.execution_service` uses for
strict validation), restricted to the frozen RESEARCH window
(2018-01-01 - 2023-01-01, `alpha_agent.data.real_market_dataset.
RESEARCH_WINDOW`) ONLY. It never runs a walk-forward fold, a bootstrap null,
a BH/FDR correction, or a DSR calculation -- there is no p-value, q-value, or
scientific verdict anywhere in this module, by construction (task spec
section 39: "Do NOT use 2023-2024 p-values/BH q-values/DSR/scientific
verdict during same-run candidate selection").

`assert_research_window_only` is a hard, second, independent boundary check
(mirroring `alpha_agent.registry.holdout_guard`'s pattern for the 2025
holdout, one partition earlier): every bar frame this module ever hands to
the C++ runner is asserted to carry no timestamp on or after
`VALIDATION_WINDOW[0]` (2023-01-01) before the backtest runs, not just
"trust the slice already did it right".

`ResearchScreenScore` is DELIBERATELY separate from
`alpha_agent.recommendation.promise.ResearchPromiseBreakdown` (different
module, no shared import) -- Research Promise is a POST-VALIDATION score
that reads DSR/BH-q/p-values; a Fast Screen trial never has those fields to
read from. Reusing the name or the formula would blur a distinction the task
spec is explicit about (section 39).
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from alpha_agent.agents.orchestrator import FamilyMember
from alpha_agent.data.calendars import SessionCalendar, default_calendar
from alpha_agent.data.real_market_dataset import (
    RESEARCH_WINDOW,
    VALIDATION_WINDOW,
    ReconstitutedRoot,
    _iso_ns,
    daily_signal_series,
    reconstitute_root,
    roll_close_marks,
    write_roll_close_marks_csv,
)
from alpha_agent.execution.capability import (
    CapabilityReason,
    assess_static_capability,
    classify_schedule_exception,
    resolve_cadence,
)
from alpha_agent.validation.phase_13_5c_matrix import (
    CAPITAL_BASE_USD,
    RollAwareCliRunner,
    daily_baseline_schedule,
    native_1m_schedule,
)
from alpha_agent.validation.runner import BacktestRun, CliBacktestRunner
from alpha_agent.validation.trading_day import build_validation_day_plan

_RESEARCH_LO_NS = _iso_ns(RESEARCH_WINDOW[0])
_RESEARCH_HI_NS = _iso_ns(RESEARCH_WINDOW[1])  # EXCLUSIVE -- == VALIDATION_WINDOW[0]
assert _RESEARCH_HI_NS == _iso_ns(VALIDATION_WINDOW[0]), "research/validation boundary must be contiguous"


class ResearchWindowViolation(RuntimeError):
    """A bar/timestamp on or after the 2023-01-01 validation boundary reached
    the Fast Screen. This must never happen -- the Fast Screen is a
    2018-2022-only research tool (task spec section 39)."""


def assert_research_window_only(frame: pd.DataFrame, *, ts_column: str = "ts_event_ns") -> None:
    if frame.empty:
        return
    max_ts = int(frame[ts_column].max())
    if max_ts >= _RESEARCH_HI_NS:
        raise ResearchWindowViolation(
            f"Fast Screen received a timestamp {max_ts} >= the validation boundary "
            f"{_RESEARCH_HI_NS} ({VALIDATION_WINDOW[0]}) -- the 2023-2024 strict-validation "
            "window (and the 2025 holdout beyond it) must never influence fast-screen selection"
        )


class FastScreenStatus(str, Enum):
    SCREENED = "SCREENED"           # a real trial ran and produced a score
    UNSUPPORTED = "UNSUPPORTED"     # a typed capability gap (never a scientific verdict)
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # too little research-window data to schedule


# ---------------------------------------------------------------------------
# ResearchScreenScore -- bounded 0-100, research-window descriptive metrics
# only. Fixed, documented weights; missing metrics contribute exactly zero
# (same missing-data-safety discipline as alpha_agent.recommendation.promise,
# independently re-derived here since this module must not import that one).
# ---------------------------------------------------------------------------

WEIGHT_SHARPE = 40.0
WEIGHT_COST_BURDEN = 25.0
WEIGHT_DRAWDOWN = 20.0
WEIGHT_ACTIVITY = 15.0
_TOTAL_SCREEN_WEIGHT = WEIGHT_SHARPE + WEIGHT_COST_BURDEN + WEIGHT_DRAWDOWN + WEIGHT_ACTIVITY
assert abs(_TOTAL_SCREEN_WEIGHT - 100.0) < 1e-9

#: Sharpe reference range -- same fixed, documented, non-data-derived choice
#: `alpha_agent.recommendation.promise` uses for its own performance
#: sub-score (not imported from there; independently declared so the two
#: modules never accidentally share mutable state).
_SHARPE_FLOOR, _SHARPE_CEIL = -1.0, 2.0
#: A cost burden (total cost / gross PnL magnitude) at or above this fraction
#: earns zero credit; at 0 it earns full credit.
_COST_BURDEN_CEIL = 1.0
#: A max drawdown (as a fraction of the starting/peak equity) at or above
#: this earns zero credit.
_DRAWDOWN_CEIL = 0.5
#: An "active day" (a day with |return| > 0) rate at or below this earns zero
#: activity credit -- a strategy that essentially never trades is not usefully
#: screenable, independent of whatever Sharpe a handful of days produced.
_ACTIVE_DAY_FLOOR = 0.02
_ACTIVE_DAY_CEIL = 0.35


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class ResearchScreenScore(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    sharpe_component: float
    cost_component: float
    drawdown_component: float
    activity_component: float
    total: float
    missing_inputs: tuple[str, ...] = ()


def _max_drawdown_fraction(equity: np.ndarray) -> float | None:
    if equity.size == 0:
        return None
    peak = np.maximum.accumulate(equity)
    peak = np.where(peak <= 0, np.nan, peak)
    drawdown = (peak - equity) / peak
    if np.all(np.isnan(drawdown)):
        return None
    return float(np.nanmax(drawdown))


def score_fast_screen_trial(run: BacktestRun) -> ResearchScreenScore:
    """Pure function of one real `BacktestRun` (task spec section 39's
    allowed inputs: Sharpe, net PnL / cost burden, max drawdown, trade/active-
    day count -- ONLY from real artifacts). No p-value, q-value, DSR, or
    verdict is ever read here -- the type simply has none."""
    missing: list[str] = []
    daily_sharpe = run.daily_sharpe
    if not np.isfinite(daily_sharpe):
        sharpe_sub = 0.0
        missing.append("daily_sharpe")
    else:
        ann = daily_sharpe * np.sqrt(252.0)
        sharpe_sub = _clamp01((ann - _SHARPE_FLOOR) / (_SHARPE_CEIL - _SHARPE_FLOOR)) * WEIGHT_SHARPE

    gross_abs = abs(run.gross_pnl_usd)
    if gross_abs <= 0:
        cost_sub = 0.0
        missing.append("cost_burden")
    else:
        burden = _clamp01(run.costs_usd / gross_abs)
        cost_sub = _clamp01(1.0 - burden / _COST_BURDEN_CEIL) * WEIGHT_COST_BURDEN

    equity = np.asarray(run.daily.equity_usd, dtype=float)
    mdd = _max_drawdown_fraction(equity)
    if mdd is None:
        dd_sub = 0.0
        missing.append("max_drawdown")
    else:
        dd_sub = _clamp01(1.0 - mdd / _DRAWDOWN_CEIL) * WEIGHT_DRAWDOWN

    n_days = run.daily.n_days
    if n_days <= 0:
        activity_sub = 0.0
        missing.append("active_days")
    else:
        active_rate = run.daily.n_nonzero_return_days / n_days
        activity_sub = (
            _clamp01((active_rate - _ACTIVE_DAY_FLOOR) / (_ACTIVE_DAY_CEIL - _ACTIVE_DAY_FLOOR))
            * WEIGHT_ACTIVITY
        )

    total = round(_clamp01((sharpe_sub + cost_sub + dd_sub + activity_sub) / 100.0) * 100.0, 1)
    return ResearchScreenScore(
        sharpe_component=round(sharpe_sub, 2), cost_component=round(cost_sub, 2),
        drawdown_component=round(dd_sub, 2), activity_component=round(activity_sub, 2),
        total=total, missing_inputs=tuple(missing),
    )


class FastScreenTrial(BaseModel):
    """One member's fast-screen outcome. `score` is `None` whenever `status`
    is not `SCREENED` -- an unsupported/insufficient-data member never gets a
    fabricated score."""

    model_config = {"frozen": True, "extra": "forbid"}

    experiment_identity: str
    strategy_family: str
    root_symbol: str
    status: FastScreenStatus
    capability_reason: CapabilityReason | None = None
    detail: str = ""
    score: ResearchScreenScore | None = None
    metrics: dict = {}


def run_fast_screen(
    *,
    member: FamilyMember,
    cli_executable: str | Path,
    work_dir: str | Path,
    calendar: SessionCalendar | None = None,
    recon: ReconstitutedRoot | None = None,
    artifact_dir: str | Path | None = None,
) -> FastScreenTrial:
    """Run ONE real C++ backtest for `member`, restricted to the frozen
    2018-2022 RESEARCH window. `recon` lets a caller share one already-
    reconstituted root's data across many members (real market data
    reconstitution is the expensive part of this call) -- built fresh via
    `alpha_agent.data.real_market_dataset.reconstitute_root` otherwise.

    `artifact_dir` (Alpha Discovery Part I, opt-in, default `None` ==
    unchanged prior behaviour): when set, persists a REAL fills/trades/daily-
    equity artifact bundle for this trial via
    `alpha_agent.artifacts.store.persist_run_artifacts` and records its
    manifest path + hash on the returned `FastScreenTrial.metrics`
    (`artifact_manifest` / `artifact_bundle_sha256`) -- unlike the deep,
    multi-call strict-validation engine, this function makes exactly ONE
    runner call, so requesting a real per-fill/per-trade export here touches
    no frozen internals at all.
    """
    calendar = calendar or default_calendar()
    cadence = resolve_cadence(signal_cadence=member.signal_cadence, strategy_family=member.strategy_family)
    required_feature_kinds = [f.spec.kind for f in member.strategy_spec.features]
    capability = assess_static_capability(signal_cadence=cadence, required_features=required_feature_kinds)
    base = {
        "experiment_identity": member.experiment_identity, "strategy_family": member.strategy_family,
        "root_symbol": member.root_symbol,
    }
    if not capability.supported:
        return FastScreenTrial(
            **base, status=FastScreenStatus.UNSUPPORTED,
            capability_reason=capability.reason, detail=capability.detail,
        )

    recon = recon or reconstitute_root(member.root_symbol, calendar=calendar)

    exec_bars = recon.canonical_bars
    exec_bars = exec_bars[
        (exec_bars["ts_event_ns"] >= _RESEARCH_LO_NS) & (exec_bars["ts_event_ns"] < _RESEARCH_HI_NS)
    ][["ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume"]]
    exec_bars = exec_bars.sort_values("ts_event_ns").reset_index(drop=True)
    assert_research_window_only(exec_bars)
    if len(exec_bars) < 3:
        return FastScreenTrial(**base, status=FastScreenStatus.INSUFFICIENT_DATA, detail="too few research-window bars")

    sig_1m = recon.forward_adjusted
    sig_1m = sig_1m[
        (sig_1m["ts_event_ns"] >= _RESEARCH_LO_NS) & (sig_1m["ts_event_ns"] < _RESEARCH_HI_NS)
    ].sort_values("ts_event_ns").reset_index(drop=True)
    assert_research_window_only(sig_1m)

    run_dir = Path(work_dir) / f"{member.experiment_identity.split(':')[-1][:16]}__fastscreen"
    run_dir.mkdir(parents=True, exist_ok=True)

    marks = roll_close_marks(recon, _RESEARCH_LO_NS, _RESEARCH_HI_NS)
    marks_path, _marks_sha = write_roll_close_marks_csv(marks, run_dir / "roll_close_marks.csv")
    roll_ts = tuple(
        int(rl.effective_ts_ns) for rl in recon.rolls
        if _RESEARCH_LO_NS <= int(rl.effective_ts_ns) < _RESEARCH_HI_NS
    )

    lo = int(exec_bars["ts_event_ns"].iloc[0])
    hi = int(exec_bars["ts_event_ns"].iloc[-1])
    from alpha_agent.schemas.market_data import PriceDomain

    try:
        if cadence == "native_1m":
            tday, sess = calendar.classify_series(sig_1m["ts_event_ns"], member.root_symbol)
            sig = sig_1m.assign(
                trading_day=tday.astype(str), session=sess.astype(str)
            ).reset_index(drop=True)
            schedule = native_1m_schedule(
                member.strategy_spec, sig, lo, hi, lo, root=member.root_symbol, price_domain=PriceDomain,
            )
        else:
            daily = daily_signal_series(sig_1m, member.root_symbol, calendar=calendar)
            schedule = daily_baseline_schedule(
                member.strategy_spec, daily, lo, hi, lo, root=member.root_symbol, price_domain=PriceDomain,
            )
    except Exception as exc:  # noqa: BLE001 -- classified, never a crash of the whole screen
        return FastScreenTrial(
            **base, status=FastScreenStatus.UNSUPPORTED,
            capability_reason=classify_schedule_exception(exc), detail=f"{type(exc).__name__}: {exc}",
        )

    if schedule is None:
        return FastScreenTrial(
            **base, status=FastScreenStatus.INSUFFICIENT_DATA,
            capability_reason=CapabilityReason.MISSING_DATA_LAYER,
            detail="target-schedule construction returned no schedule for the research window",
        )

    # A `ValidationDayPlan` (arg 8 in the CLI's positional contract) is
    # required to even POSITION the roll-close-marks path (arg 9) -- without
    # it the C++ engine has no auxiliary same-timestamp outgoing-contract
    # close to resolve a roll cleanly and refuses with a retroactive-expiry
    # error. Built exactly the way `ValidationEngine` already does for every
    # strict-validation run, over the SAME research-window `exec_bars`.
    validation_day_plan = build_validation_day_plan(exec_bars, root_symbol=member.root_symbol, calendar=calendar)
    runner = RollAwareCliRunner(
        CliBacktestRunner(str(cli_executable), run_dir, roll_close_marks_path=marks_path), roll_ts,
    )
    trades_out = fills_out = None
    if artifact_dir is not None:
        trades_out = run_dir / "trades.csv"
        fills_out = run_dir / "fills.csv"
    run = runner.run(
        schedule=schedule, bars=exec_bars, contracts=recon.contracts, capital_base_usd=CAPITAL_BASE_USD,
        validation_day_plan=validation_day_plan, trades_out_path=trades_out, fills_out_path=fills_out,
    )
    assert_research_window_only(exec_bars)  # re-asserted post-run: the frame handed to C++ never mutated

    score = score_fast_screen_trial(run)
    # Release UX bugfix pass, issue 5: the candidate card wants a real,
    # displayable max-drawdown PERCENTAGE, not just `score.drawdown_component`
    # (a 0-20 weighted sub-score `score_fast_screen_trial` already computes
    # internally and does not expose). Re-derived here via the exact same
    # `_max_drawdown_fraction` helper, from the same real daily equity series
    # -- purely additive to `metrics` (a loose dict, no schema/identity
    # impact), never a new formula or threshold.
    max_drawdown_fraction = _max_drawdown_fraction(np.asarray(run.daily.equity_usd, dtype=float))
    metrics = {
        "gross_pnl_usd": run.gross_pnl_usd, "costs_usd": run.costs_usd, "net_pnl_usd": run.net_pnl_usd,
        "daily_sharpe": run.daily_sharpe, "annualized_sharpe": run.annualized_sharpe,
        "n_trades": run.n_trades, "n_fills": run.n_fills, "n_days": run.daily.n_days,
        "max_drawdown_fraction": max_drawdown_fraction,
    }
    if artifact_dir is not None:
        from alpha_agent.artifacts.store import manifest_path_for, persist_run_artifacts

        bundle = persist_run_artifacts(
            experiment_identity=member.experiment_identity, attempt_ordinal=1,
            strategy_fingerprint=member.strategy_fingerprint, daily=run.daily,
            fills_csv_path=fills_out, trades_csv_path=trades_out, price_bars=exec_bars,
            out_dir=Path(artifact_dir),
        )
        metrics["artifact_manifest"] = str(manifest_path_for(bundle, out_dir=Path(artifact_dir)))
        metrics["artifact_bundle_sha256"] = bundle.bundle_sha256

    return FastScreenTrial(**base, status=FastScreenStatus.SCREENED, score=score, metrics=metrics)


def rank_fast_screen_trials(trials: list[FastScreenTrial]) -> list[FastScreenTrial]:
    """Deterministic ranking: SCREENED trials by score (desc), tie-broken by
    `experiment_identity`; every non-SCREENED trial sorts after every SCREENED
    one (never silently promoted for lacking a score)."""

    def key(t: FastScreenTrial) -> tuple[int, float, str]:
        if t.status is FastScreenStatus.SCREENED and t.score is not None:
            return (0, -t.score.total, t.experiment_identity)
        return (1, 0.0, t.experiment_identity)

    return sorted(trials, key=key)


def rank_pool_members(members: list[FamilyMember], trials: list[FastScreenTrial]) -> list[FamilyMember]:
    """Candidate-pool members in `rank_fast_screen_trials` order: members
    with a SCREENED trial first, by that ranking; every other member after
    them, in pool order (never promoted or demoted for lacking a score). The
    one ordering rule the Discover page shows -- it ranks nothing itself."""
    screened = [t for t in rank_fast_screen_trials(trials)
                if t.status is FastScreenStatus.SCREENED and t.score is not None]
    position = {t.experiment_identity: i for i, t in enumerate(screened)}
    return sorted(members, key=lambda m: (0, position[m.experiment_identity]) if m.experiment_identity in position
                  else (1, 0))


def select_top_k(trials: list[FastScreenTrial], k: int) -> list[FastScreenTrial]:
    """Top `k` SCREENED trials only (task spec section 42: freezing a
    candidate that was never actually screenable would freeze nothing real).
    May return fewer than `k` if fewer than `k` trials were SCREENED."""
    if k < 1:
        raise ValueError("k must be >= 1")
    ranked = rank_fast_screen_trials(trials)
    screened = [t for t in ranked if t.status is FastScreenStatus.SCREENED]
    return screened[:k]
