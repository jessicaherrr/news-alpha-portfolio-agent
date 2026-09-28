"""TAKE/SKIP -> a CLOSED :class:`TargetSchedule` for the C++ Quant Core.

This is where the ML layer stops. Downstream of this module the official numbers
are produced exactly as they were in Phase 13: one target schedule, one
``quant_backtest_targets_csv`` run per cost scenario, Fill-derived PnL. Python
never computes an official PnL, and a probability never becomes a position size.

The frozen meta-label event semantics (prompt 15A s.4)
-----------------------------------------------------
The meta-labeled schedule has EXACTLY the same row timestamps as the primary
schedule. Only target values change, and only inside episodes:

* ``TAKE``  -> the episode's rows keep the primary strategy's own targets;
* ``SKIP``  -> the episode's rows are rewritten to the neutral target ``0``;
* rows outside any episode are untouched;
* the closing row of an episode belongs to whatever comes next and is untouched
  (on a reversal it is the next episode's entry row and that episode has its own
  decision).

Keeping the row timestamps identical is what makes the primary-vs-meta-labeled
comparison run on *exactly aligned evaluation support* (prompt 15A s.12): same
bars, same schedule policy, same warm-up, same absent-row semantics. Any
difference in the result is then attributable to the TAKE/SKIP decisions and
nothing else.

One decision per episode
------------------------
The MVP decides once, at the entry, and does not re-decide mid-episode. That is
a deliberate restriction, not an oversight: a mid-episode re-decision needs its
own label definition (the outcome of a *partial* episode), which would be a new
frozen research semantic.

A note on the schedule fingerprint
----------------------------------
``targets.csv`` carries a ``stratdsl1:`` strategy fingerprint, validated by the
C++ parser. A meta-labeled schedule therefore keeps the PRIMARY spec's
fingerprint -- the DSL boundary is frozen and Phase 15 does not touch it. The
two schedules are still distinguishable, and provably so: ``schedule_hash()``
covers the target values, so it differs as soon as one episode is skipped. The
ML identity lives in the ML artifacts and in the registry, never in the
target-schedule fingerprint.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from alpha_agent.backtest.targets import TargetSchedule, TargetScheduleRow
from alpha_agent.ml.enums import MetaLabelAction
from alpha_agent.ml.episodes import PrimaryEpisode
from alpha_agent.ml.errors import MLProtocolError

#: The neutral target a SKIP resolves to. Closed: the model cannot pick another.
NEUTRAL_TARGET_UNITS = 0


class MetaLabeledScheduleDiff(BaseModel):
    """What the meta-label actually changed. Auditable, and reported every run."""

    model_config = {"frozen": True, "extra": "forbid"}

    n_rows: int = Field(ge=0)
    n_episodes: int = Field(ge=0)
    n_take: int = Field(ge=0)
    n_skip: int = Field(ge=0)
    n_rows_changed: int = Field(ge=0)
    primary_schedule_hash: str
    meta_labeled_schedule_hash: str

    @property
    def take_rate(self) -> float:
        return (self.n_take / self.n_episodes) if self.n_episodes else 0.0


def build_meta_labeled_schedule(
    primary: TargetSchedule,
    episodes: tuple[PrimaryEpisode, ...],
    actions: dict[int, MetaLabelAction],
    *,
    default_action: MetaLabelAction = MetaLabelAction.TAKE,
) -> tuple[TargetSchedule, MetaLabeledScheduleDiff]:
    """Apply per-episode TAKE/SKIP decisions to a primary target schedule.

    ``actions`` maps ``episode_index`` -> action. An episode with no prediction
    falls back to ``default_action``; ``TAKE`` is the honest default because it
    leaves the primary strategy untouched wherever the model abstained, so an
    unlabelable episode never silently becomes a bet that the primary was wrong.
    """
    if any(e.strategy_fingerprint != primary.strategy_fingerprint for e in episodes):
        raise MLProtocolError(
            "episodes were extracted from a different strategy than this schedule"
        )
    unknown = set(actions) - {e.episode_index for e in episodes}
    if unknown:
        raise MLProtocolError(f"actions reference unknown episode indices: {sorted(unknown)}")

    targets = [r.target_units for r in primary.rows]
    n_take = n_skip = 0
    for ep in episodes:
        action = actions.get(ep.episode_index, default_action)
        if action is MetaLabelAction.TAKE:
            n_take += 1
            continue
        n_skip += 1
        stop = ep.exit_row_index if ep.exit_row_index is not None else len(targets)
        for i in range(ep.entry_row_index, stop):
            targets[i] = NEUTRAL_TARGET_UNITS

    changed = sum(1 for a, r in zip(targets, primary.rows) if a != r.target_units)
    rows = tuple(
        TargetScheduleRow(
            ts_event_ns=r.ts_event_ns,
            root_symbol=r.root_symbol,
            target_units=t,
            strategy_fingerprint=r.strategy_fingerprint,
            matched_rule_id=r.matched_rule_id,
        )
        for r, t in zip(primary.rows, targets)
    )
    meta = primary.model_copy(update={"rows": rows})
    return meta, MetaLabeledScheduleDiff(
        n_rows=len(rows),
        n_episodes=len(episodes),
        n_take=n_take,
        n_skip=n_skip,
        n_rows_changed=changed,
        primary_schedule_hash=primary.schedule_hash(),
        meta_labeled_schedule_hash=meta.schedule_hash(),
    )


def assert_aligned_evaluation_support(
    primary: TargetSchedule, meta_labeled: TargetSchedule
) -> None:
    """The two schedules must differ ONLY in target values.

    Same root, same fingerprint, same row count, same timestamps. Anything else
    would make the primary-vs-meta-labeled comparison an unfair one.
    """
    if primary.root_symbol != meta_labeled.root_symbol:
        raise MLProtocolError("meta-labeled schedule changed the root symbol")
    if primary.strategy_fingerprint != meta_labeled.strategy_fingerprint:
        raise MLProtocolError(
            "meta-labeled schedule changed the primary strategy fingerprint; the meta-label is "
            "a filter around a frozen primary, never a redefinition of it"
        )
    if len(primary.rows) != len(meta_labeled.rows):
        raise MLProtocolError(
            f"meta-labeled schedule has {len(meta_labeled.rows)} rows vs the primary's "
            f"{len(primary.rows)}; the comparison must run on identical evaluation support"
        )
    for a, b in zip(primary.rows, meta_labeled.rows):
        if a.ts_event_ns != b.ts_event_ns:
            raise MLProtocolError(
                f"meta-labeled schedule moved a decision timestamp ({a.ts_event_ns} -> "
                f"{b.ts_event_ns}); only target VALUES may change"
            )


def assert_closed_action_space(meta_labeled: TargetSchedule, primary: TargetSchedule) -> None:
    """Every row is either the primary's own target or the neutral target.

    This is the structural proof that the model never invented a position size
    (prompt 15A s.4).
    """
    for a, b in zip(primary.rows, meta_labeled.rows):
        if b.target_units not in (a.target_units, NEUTRAL_TARGET_UNITS):
            raise MLProtocolError(
                f"meta-labeled target {b.target_units} at ts {b.ts_event_ns} is neither the "
                f"primary target {a.target_units} nor the neutral target "
                f"{NEUTRAL_TARGET_UNITS}; the MVP action space is closed to TAKE/SKIP and a "
                "model probability never becomes continuous leverage"
            )
