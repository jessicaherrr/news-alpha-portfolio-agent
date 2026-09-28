"""PRIMARY vs META-LABELED economics, on exactly aligned evaluation support.

For every economic trial:

1. split the pooled OOF predictions back to per-root TAKE/SKIP actions;
2. build the meta-labeled :class:`TargetSchedule` per root
   (:func:`build_meta_labeled_schedule`) -- identical row timestamps, only
   target values change;
3. run PRIMARY and META through the economics engine at 1.0x / 1.5x / 2.0x cost;
4. restrict BOTH daily return series to the OOF evaluation window (the union of
   the nested-CV outer test blocks) so the comparison is on exactly aligned
   support -- ``assert_aligned_evaluation_support`` is a hard invariant first;
5. report net PnL, daily Sharpe, trades, turnover, drawdown, TAKE rate and the
   economic delta vs primary, per cost scenario.

Official numbers are Fill-derived from the engine. Python restricts a day series
and computes descriptive statistics on it; it never reprices.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from alpha_agent.backtest.targets import TargetSchedule
from alpha_agent.ml.engine_io import EngineRunResult, MetaEngineRunner
from alpha_agent.ml.enums import MetaLabelAction
from alpha_agent.ml.episodes import extract_primary_episodes
from alpha_agent.ml.execution import (
    MetaLabeledScheduleDiff,
    assert_aligned_evaluation_support,
    assert_closed_action_space,
    build_meta_labeled_schedule,
)
from alpha_agent.ml.manifest import COST_SCENARIOS, OUTER_TEST_BLOCKS
from alpha_agent.ml.predictions import MLPredictionFrame
from alpha_agent.validation.metrics import annualized_sharpe, daily_sharpe

#: The OOF evaluation window: the union of the frozen nested-CV outer test blocks.
EVAL_WINDOW_NS: tuple[int, int] = (OUTER_TEST_BLOCKS[0][0], OUTER_TEST_BLOCKS[-1][1])


def _max_drawdown_usd(daily_pnl: np.ndarray) -> float:
    equity = np.cumsum(daily_pnl)
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity)) if equity.size else 0.0


class ArmScenarioEconomics(BaseModel):
    """One arm (PRIMARY or META) at one cost scenario, on the eval window."""

    model_config = {"frozen": True, "extra": "forbid"}

    arm: str
    cost_scenario_label: str
    cost_multiplier: float
    n_eval_days: int
    net_pnl_usd: float
    gross_pnl_usd: float
    costs_usd: float
    daily_sharpe: float
    annualized_sharpe: float
    max_drawdown_usd: float
    n_trades: int
    n_fills: int
    turnover: float
    schedule_hash: str


class TrialEconomics(BaseModel):
    """The full PRIMARY vs META economic comparison for one economic trial."""

    model_config = {"frozen": True, "extra": "forbid"}

    trial_label: str
    root_symbol: str
    primary_family: str
    take_rate: float
    n_episodes_in_eval: int
    n_take_in_eval: int
    aligned_support_days: int
    schedule_diff: MetaLabeledScheduleDiff

    primary: tuple[ArmScenarioEconomics, ...]
    meta: tuple[ArmScenarioEconomics, ...]

    #: per cost scenario: meta.net - primary.net
    economic_delta_vs_primary_usd: dict[str, float]
    #: worst fractional net-PnL degradation of META across the >1x scenarios
    meta_max_cost_degradation: float

    baseline_primary_daily_sharpe: float
    baseline_meta_daily_sharpe: float
    meta_beats_primary_daily_sharpe: bool

    #: the META baseline daily net return series over the eval window (for the
    #: gating null and DSR downstream), with its day timestamps so fold
    #: consistency can split on the real calendar-year outer-block boundaries
    meta_baseline_daily_returns: tuple[float, ...]
    primary_baseline_daily_returns: tuple[float, ...]
    meta_baseline_daily_ts_ns: tuple[int, ...]


def restrict_to_eval_window(
    run: EngineRunResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(daily_returns, daily_pnl, day_ts_ns)`` restricted to :data:`EVAL_WINDOW_NS`."""
    ts = np.asarray(run.daily.ts_ns, dtype="int64")
    rets = run.daily.returns_array()
    pnl = run.daily.pnl_array()
    lo, hi = EVAL_WINDOW_NS
    mask = (ts >= lo) & (ts < hi)
    return rets[mask], pnl[mask], ts[mask]


def _arm_scenario(
    arm: str, run: EngineRunResult, *, capital_base_usd: float
) -> ArmScenarioEconomics:
    rets, pnl, _ts = restrict_to_eval_window(run)
    return ArmScenarioEconomics(
        arm=arm,
        cost_scenario_label=run.cost_scenario_label,
        cost_multiplier=run.cost_multiplier,
        n_eval_days=int(rets.size),
        net_pnl_usd=float(pnl.sum()),
        gross_pnl_usd=run.gross_pnl_usd,
        costs_usd=run.costs_usd,
        daily_sharpe=daily_sharpe(rets) if rets.size else float("nan"),
        annualized_sharpe=annualized_sharpe(rets) if rets.size else float("nan"),
        max_drawdown_usd=_max_drawdown_usd(pnl),
        n_trades=run.n_trades,
        n_fills=run.n_fills,
        turnover=float(run.n_fills) / max(1, rets.size),
        schedule_hash=run.schedule_hash,
    )


def actions_by_episode_for_root(
    frame: MLPredictionFrame, root_symbol: str
) -> dict[int, MetaLabelAction]:
    """Map this root's OOF predictions to ``episode_index -> action``.

    The event id is ``{root}__{family}__EP{index:05d}``.
    """
    out: dict[int, MetaLabelAction] = {}
    for p in frame.predictions:
        if p.root_symbol != root_symbol:
            continue
        try:
            ep_index = int(p.event_id.rsplit("EP", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"cannot parse episode index from event id {p.event_id!r}") from exc
        out[ep_index] = p.action
    return out


def evaluate_trial_economics(
    *,
    trial_label: str,
    root_symbol: str,
    primary_family: str,
    primary_schedule: TargetSchedule,
    oof_frame: MLPredictionFrame,
    engine: MetaEngineRunner,
    capital_base_usd: float,
) -> TrialEconomics:
    """Run and compare PRIMARY vs META for one economic trial."""
    episodes = extract_primary_episodes(primary_schedule)
    actions = actions_by_episode_for_root(oof_frame, root_symbol)
    unknown = set(actions) - {e.episode_index for e in episodes}
    if unknown:
        raise ValueError(f"OOF actions reference unknown episodes for {root_symbol}: {sorted(unknown)}")

    meta_schedule, diff = build_meta_labeled_schedule(
        primary_schedule, episodes, actions, default_action=MetaLabelAction.TAKE
    )
    assert_aligned_evaluation_support(primary_schedule, meta_schedule)
    assert_closed_action_space(meta_schedule, primary_schedule)

    primary_runs: list[ArmScenarioEconomics] = []
    meta_runs: list[ArmScenarioEconomics] = []
    delta: dict[str, float] = {}
    prim_base_rets: np.ndarray = np.empty(0)
    meta_base_rets: np.ndarray = np.empty(0)
    meta_base_ts: np.ndarray = np.empty(0, dtype="int64")
    base_label = str(COST_SCENARIOS[0]["label"])

    for scenario in COST_SCENARIOS:
        label = str(scenario["label"])
        mult = float(scenario["multiplier"])
        pr = engine.run(
            primary_schedule, cost_scenario_label=label, cost_multiplier=mult,
            want_audit_trails=False,
        )
        mr = engine.run(
            meta_schedule, cost_scenario_label=label, cost_multiplier=mult,
            want_audit_trails=False,
        )
        p_arm = _arm_scenario("PRIMARY", pr, capital_base_usd=capital_base_usd)
        m_arm = _arm_scenario("META", mr, capital_base_usd=capital_base_usd)
        if p_arm.n_eval_days != m_arm.n_eval_days:
            raise ValueError(
                f"{trial_label}: PRIMARY has {p_arm.n_eval_days} eval days but META has "
                f"{m_arm.n_eval_days}; the comparison support is not aligned"
            )
        primary_runs.append(p_arm)
        meta_runs.append(m_arm)
        delta[label] = m_arm.net_pnl_usd - p_arm.net_pnl_usd
        if label == base_label:
            prim_base_rets = restrict_to_eval_window(pr)[0]
            meta_base_rets, _mp, meta_base_ts = restrict_to_eval_window(mr)

    # META cost degradation vs its own baseline
    base_meta = next(m for m in meta_runs if m.cost_scenario_label == base_label)
    worst = 0.0
    for m in meta_runs:
        if m.cost_scenario_label == base_label:
            continue
        if abs(base_meta.net_pnl_usd) < 1e-9:
            degr = 0.0 if m.net_pnl_usd >= 0 else 1.0
        else:
            degr = 1.0 - (m.net_pnl_usd / base_meta.net_pnl_usd)
        worst = max(worst, degr)

    n_take_eval = sum(
        1 for e in episodes
        if EVAL_WINDOW_NS[0] <= e.entry_decision_ts_ns < EVAL_WINDOW_NS[1]
        and actions.get(e.episode_index, MetaLabelAction.TAKE) is MetaLabelAction.TAKE
    )
    n_ep_eval = sum(
        1 for e in episodes
        if EVAL_WINDOW_NS[0] <= e.entry_decision_ts_ns < EVAL_WINDOW_NS[1]
    )
    take_rate_eval = n_take_eval / n_ep_eval if n_ep_eval else 0.0

    p_base_sharpe = next(
        p.daily_sharpe for p in primary_runs if p.cost_scenario_label == base_label
    )
    m_base_sharpe = base_meta.daily_sharpe

    return TrialEconomics(
        trial_label=trial_label,
        root_symbol=root_symbol,
        primary_family=primary_family,
        take_rate=take_rate_eval,
        n_episodes_in_eval=n_ep_eval,
        n_take_in_eval=n_take_eval,
        aligned_support_days=primary_runs[0].n_eval_days,
        schedule_diff=diff,
        primary=tuple(primary_runs),
        meta=tuple(meta_runs),
        economic_delta_vs_primary_usd=delta,
        meta_max_cost_degradation=float(worst),
        baseline_primary_daily_sharpe=float(p_base_sharpe) if np.isfinite(p_base_sharpe) else 0.0,
        baseline_meta_daily_sharpe=float(m_base_sharpe) if np.isfinite(m_base_sharpe) else 0.0,
        meta_beats_primary_daily_sharpe=bool(
            np.isfinite(m_base_sharpe)
            and np.isfinite(p_base_sharpe)
            and m_base_sharpe > p_base_sharpe
        ),
        meta_baseline_daily_returns=tuple(float(r) for r in meta_base_rets),
        primary_baseline_daily_returns=tuple(float(r) for r in prim_base_rets),
        meta_baseline_daily_ts_ns=tuple(int(t) for t in meta_base_ts),
    )
