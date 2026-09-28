"""The TAKE-rate-matched placebo control (prompt 15A s.15, Phase 15B rule 2).

It separates "the model chose well" from "the model simply traded less". For a
trial that rejected its gating null, the placebo runs the frozen 20 draws:

* each draw is a RANDOM meta-labeler that TAKEs a random subset of the SAME
  eval-window episodes the model saw, with the SAME realised TAKE count -- so it
  trades exactly as much as the model, just not selectively;
* it is C++-executed at baseline cost (never Python repricing);
* the comparison statistic is the eval-window daily Sharpe;
* ``p = (1 + #{placebo Sharpe >= observed}) / (20 + 1)``; a Phase 15 PASS needs
  ``p <= 0.05``.

The RNG for each draw is derived deterministically from the trial's pre-run
``experiment_identity`` and the draw index -- never the ambient/global RNG -- so
the whole placebo is reproducible. The placebo is a NULL: it never enters the BH
denominator.
"""
from __future__ import annotations

import hashlib

import numpy as np
from pydantic import BaseModel

from alpha_agent.backtest.targets import TargetSchedule
from alpha_agent.ml.adjudication import placebo_empirical_p
from alpha_agent.ml.economics import EVAL_WINDOW_NS, restrict_to_eval_window
from alpha_agent.ml.engine_io import MetaEngineRunner
from alpha_agent.ml.enums import MetaLabelAction
from alpha_agent.ml.episodes import extract_primary_episodes
from alpha_agent.ml.execution import (
    assert_aligned_evaluation_support,
    assert_closed_action_space,
    build_meta_labeled_schedule,
)
from alpha_agent.ml.manifest import COST_SCENARIOS, PLACEBO_DRAWS
from alpha_agent.ml.predictions import MLPredictionFrame
from alpha_agent.validation.metrics import daily_sharpe


def _draw_rng(experiment_identity: str, draw_index: int) -> tuple[np.random.Generator, int]:
    digest = hashlib.sha256(
        f"phase_15_placebo|{experiment_identity}|draw={draw_index}".encode()
    ).digest()
    seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(seed), seed


class PlaceboDraw(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    draw_index: int
    rng_seed: int
    n_take: int
    n_eval_episodes: int
    selected_episode_indices: tuple[int, ...]
    selection_hash: str
    meta_labeled_schedule_hash: str
    daily_sharpe: float
    net_pnl_usd: float


class PlaceboResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    trial_label: str
    experiment_identity: str
    ran: bool
    reason_not_run: str = ""
    draws: tuple[PlaceboDraw, ...] = ()
    observed_meta_daily_sharpe: float = 0.0
    comparison_statistic: str = "daily_sharpe"
    n_draws: int = 0
    n_placebo_ge_observed: int = 0
    empirical_p_value: float = 1.0
    empirical_p_formula: str = "(1 + count(placebo_stat >= observed_stat)) / (draws + 1)"
    empirical_p_max: float = 0.05
    significant: bool = False
    in_bh_denominator: bool = False


def run_placebo_for_trial(
    *,
    trial_label: str,
    experiment_identity: str,
    root_symbol: str,
    primary_schedule: TargetSchedule,
    oof_frame: MLPredictionFrame,
    observed_meta_daily_sharpe: float,
    engine: MetaEngineRunner,
    should_run: bool,
    reason_not_run: str = "",
    draws: int = PLACEBO_DRAWS,
    empirical_p_max: float = 0.05,
) -> PlaceboResult:
    """Run (or record as skipped) the conditional placebo for one trial."""
    if not should_run:
        return PlaceboResult(
            trial_label=trial_label,
            experiment_identity=experiment_identity,
            ran=False,
            reason_not_run=reason_not_run or "gating null not rejected",
            observed_meta_daily_sharpe=float(observed_meta_daily_sharpe),
            n_draws=0,
            empirical_p_value=1.0,
            empirical_p_max=empirical_p_max,
        )

    episodes = extract_primary_episodes(primary_schedule)
    from alpha_agent.ml.economics import actions_by_episode_for_root

    model_actions = actions_by_episode_for_root(oof_frame, root_symbol)
    eval_episode_indices = [
        e.episode_index
        for e in episodes
        if EVAL_WINDOW_NS[0] <= e.entry_decision_ts_ns < EVAL_WINDOW_NS[1]
        and e.episode_index in model_actions
    ]
    n_eval = len(eval_episode_indices)
    n_take = sum(
        1 for i in eval_episode_indices if model_actions[i] is MetaLabelAction.TAKE
    )
    base_label = str(COST_SCENARIOS[0]["label"])

    # A base PASS implies the meta arm traded, so this should never fire; if it
    # does, no draw can be matched on a non-existent TAKE rate -- that is a typed
    # refusal to preserve, not 20 fabricated no-op draws.
    if n_eval == 0 or n_take == 0:
        return PlaceboResult(
            trial_label=trial_label,
            experiment_identity=experiment_identity,
            ran=True,
            reason_not_run=(
                f"NO_EVAL_WINDOW_TAKES: n_eval_episodes={n_eval}, n_take={n_take}; "
                "the matched-TAKE-rate placebo has nothing to match"
            ),
            draws=(),
            observed_meta_daily_sharpe=float(observed_meta_daily_sharpe),
            n_draws=draws,
            n_placebo_ge_observed=0,
            empirical_p_value=1.0,
            empirical_p_max=empirical_p_max,
            significant=False,
        )

    out_draws: list[PlaceboDraw] = []
    for d in range(draws):
        rng, seed = _draw_rng(experiment_identity, d)
        take_set = (
            set(rng.choice(eval_episode_indices, size=n_take, replace=False).tolist())
            if 0 < n_take < n_eval
            else (set(eval_episode_indices) if n_take >= n_eval else set())
        )
        placebo_actions = {
            i: (MetaLabelAction.TAKE if i in take_set else MetaLabelAction.SKIP)
            for i in eval_episode_indices
        }
        meta_sched, _diff = build_meta_labeled_schedule(
            primary_schedule, episodes, placebo_actions, default_action=MetaLabelAction.TAKE
        )
        assert_aligned_evaluation_support(primary_schedule, meta_sched)
        assert_closed_action_space(meta_sched, primary_schedule)
        run = engine.run(
            meta_sched, cost_scenario_label=base_label, cost_multiplier=1.0,
            want_audit_trails=False,
        )
        rets, pnl, _ts = restrict_to_eval_window(run)
        sel = tuple(sorted(take_set))
        out_draws.append(
            PlaceboDraw(
                draw_index=d,
                rng_seed=seed,
                n_take=len(take_set),
                n_eval_episodes=n_eval,
                selected_episode_indices=sel,
                selection_hash=hashlib.sha256(str(sel).encode()).hexdigest(),
                meta_labeled_schedule_hash=meta_sched.schedule_hash(),
                daily_sharpe=float(daily_sharpe(rets)) if rets.size else 0.0,
                net_pnl_usd=float(pnl.sum()),
            )
        )

    stats = tuple(dr.daily_sharpe for dr in out_draws)
    p = placebo_empirical_p(observed_meta_daily_sharpe, stats, draws=draws)
    n_ge = sum(1 for s in stats if s >= observed_meta_daily_sharpe)
    return PlaceboResult(
        trial_label=trial_label,
        experiment_identity=experiment_identity,
        ran=True,
        draws=tuple(out_draws),
        observed_meta_daily_sharpe=float(observed_meta_daily_sharpe),
        n_draws=draws,
        n_placebo_ge_observed=n_ge,
        empirical_p_value=float(p),
        empirical_p_max=empirical_p_max,
        significant=bool(p <= empirical_p_max),
    )
