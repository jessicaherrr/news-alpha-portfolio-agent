"""Subprocess bridge to the C++ ``quant_backtest_targets_csv`` reference CLI.

Writes the research bundle -- the frozen Phase 02.5 ``bars.csv`` / ``contracts.csv``
plus the **separate** Phase 11 ``targets.csv`` -- into a work directory and runs
the C++ engine over it. The C++ side owns contract resolution, the mandatory
risk gate, execution simulation, fills and official PnL.

pybind11 replaces this transport in Phase 19; the CLI stays the reference path.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from alpha_agent.adapters.cpp_cli import write_boundary_bundle
from alpha_agent.backtest.targets import TargetSchedule


def write_targets_bundle(
    bars: pd.DataFrame,
    contracts: pd.DataFrame,
    schedule: TargetSchedule,
    out_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Write ``bars.csv`` + ``contracts.csv`` (frozen) + ``targets.csv`` (new,
    separate). Returns the three paths."""
    bars_path, contracts_path = write_boundary_bundle(bars, contracts, out_dir)
    targets_path = schedule.write_csv(Path(out_dir) / "targets.csv")
    return bars_path, contracts_path, targets_path


def run_targets_backtest_cli(
    bars: pd.DataFrame,
    contracts: pd.DataFrame,
    schedule: TargetSchedule,
    *,
    executable: str | Path,
    work_dir: str | Path,
    schedule_policy: str = "no_decision",
    commission_per_contract_usd: float | None = None,
    slippage_ticks: float | None = None,
    spread_ticks: float | None = None,
    validation_days_path: str | Path | None = None,
    roll_close_marks_path: str | Path | None = None,
    trades_out_path: str | Path | None = None,
    fills_out_path: str | Path | None = None,
) -> dict:
    """Run the reference C++ scheduled-target backtest over a research bundle and
    return its one-line JSON result as a dict.

    ``schedule_policy`` is the C++ absent-row policy: ``no_decision`` (default --
    no row means NO NEW STRATEGY DECISION), ``flat`` (a missing row is an explicit
    target 0), ``require_row`` (strict), or ``reemit_previous`` (explicit opt-in
    that re-emits the previous target as a NEW Signal; can retry a rejected
    intent -- never the reference path).

    ``commission_per_contract_usd`` / ``slippage_ticks`` / ``spread_ticks`` are
    the optional Phase 13 cost-stress overrides; ``None`` -> the frozen reference
    assumptions (2.0, 0.0, 0.0). ``validation_days_path`` (Phase 13.1) points at a
    ``boundary_ts_ns`` CSV so ``daily_equity`` becomes a canonical futures
    ``trading_day`` series; when it is given the three cost args are always
    emitted (defaulted if unset) to keep the positional CLI order unambiguous.

    ``roll_close_marks_path`` (Phase 13.5C) points at an
    ``instrument_id,ts_event_ns,close`` CSV of auxiliary same-timestamp
    OUTGOING-contract closes consulted ONLY by the C++ roll close-leg (never a
    ``MarketEvent``); it requires ``validation_days_path`` to also be set. Omitted
    => the frozen ``RejectDefer`` roll path is byte-identical.

    ``trades_out_path`` (Phase 15A) is the ADDITIVE, READ-ONLY closed-trade audit
    export. It is passed as the named flag ``--trades-out=<path>``, which the C++
    CLI strips from ``argv`` before positional parsing, so it never disturbs the
    positional contract above and changes no execution/accounting semantics and no
    JSON field. It is how Phase 15 gets Fill-derived per-episode economics for a
    meta-label without Python ever computing PnL.

    ``fills_out_path`` (Phase 15A) is the matching fill audit export. It is needed
    alongside the trade export because ``ClosedTrade::costs_usd`` carries the
    CLOSING fill's commission only, so an episode's full round-turn cost can only
    be attributed by summing the fill trail."""
    bars_path, contracts_path, targets_path = write_targets_bundle(
        bars, contracts, schedule, work_dir
    )
    argv = [
        str(executable),
        str(bars_path),
        str(contracts_path),
        str(targets_path),
        schedule_policy,
    ]
    cost = (commission_per_contract_usd, slippage_ticks, spread_ticks)
    need_cost_args = (
        any(c is not None for c in cost)
        or validation_days_path is not None
        or roll_close_marks_path is not None
    )
    if need_cost_args:
        if any(c is None for c in cost):
            cost = (
                2.0 if cost[0] is None else cost[0],
                0.0 if cost[1] is None else cost[1],
                0.0 if cost[2] is None else cost[2],
            )
        argv += [repr(float(cost[0])), repr(float(cost[1])), repr(float(cost[2]))]
    # roll_close_marks.csv (Phase 13.5C) is positional arg 9 -- it requires the
    # validation_days.csv slot (arg 8) to be filled so the order is unambiguous.
    if validation_days_path is not None or roll_close_marks_path is not None:
        if validation_days_path is None:
            raise ValueError(
                "roll_close_marks_path requires validation_days_path to also be set "
                "(the C++ CLI positional order needs arg 8 filled)"
            )
        argv.append(str(validation_days_path))
    if roll_close_marks_path is not None:
        argv.append(str(roll_close_marks_path))
    # ADDITIVE (Phase 15A): a NAMED flag, deliberately not positional.
    if trades_out_path is not None:
        argv.append(f"--trades-out={trades_out_path}")
    if fills_out_path is not None:
        argv.append(f"--fills-out={fills_out_path}")
    proc = subprocess.run(argv, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout.strip())
