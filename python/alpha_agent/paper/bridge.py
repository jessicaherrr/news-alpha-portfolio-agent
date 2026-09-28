"""Subprocess bridge to the Phase 21 ``quant_paper_trading_targets_csv`` CLI.

Mirrors :mod:`alpha_agent.adapters.targets_bridge` exactly, adding only the
``risk_config.csv`` (mandatory) and ``--margin-config=`` (optional) inputs the
new CLI needs. The C++ side owns contract resolution, the mandatory HARD risk
gate (``PortfolioRiskManager``, not pass-through), execution simulation, fills
and official PnL -- this module writes CSV, shells the CLI, and parses its one
JSON line back. It computes nothing.

Phase 21.1 splits argv-building (:func:`build_paper_trading_argv`) from the
subprocess call (:func:`invoke_paper_trading_cli`, which also returns the raw
stdout bytes) so :mod:`alpha_agent.paper.engine` can hash the EXACT file
bytes it wrote and the EXACT stdout the CLI produced for
:class:`~alpha_agent.paper.provenance.PaperStepProvenance`, rather than
re-deriving them from a re-serialization.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from alpha_agent.adapters.cpp_cli import write_boundary_bundle
from alpha_agent.backtest.targets import TargetSchedule
from alpha_agent.paper.risk_policy import PaperRiskPolicy


def write_paper_trading_bundle(
    bars: pd.DataFrame,
    contracts: pd.DataFrame,
    schedule: TargetSchedule,
    risk_policy: PaperRiskPolicy,
    out_dir: str | Path,
) -> tuple[Path, Path, Path, Path]:
    """Write ``bars.csv`` + ``contracts.csv`` (frozen) + ``targets.csv`` +
    ``risk_config.csv`` (Phase 21). Returns the four paths -- the EXACT files
    the CLI will read."""
    bars_path, contracts_path = write_boundary_bundle(bars, contracts, out_dir)
    targets_path = schedule.write_csv(Path(out_dir) / "targets.csv")
    risk_config_path = risk_policy.write_csv(Path(out_dir) / "risk_config.csv")
    return bars_path, contracts_path, targets_path, risk_config_path


def build_paper_trading_argv(
    *,
    bars_path: str | Path,
    contracts_path: str | Path,
    targets_path: str | Path,
    risk_config_path: str | Path,
    executable: str | Path,
    schedule_policy: str = "no_decision",
    commission_per_contract_usd: float | None = None,
    slippage_ticks: float | None = None,
    spread_ticks: float | None = None,
    validation_days_path: str | Path | None = None,
    roll_close_marks_path: str | Path | None = None,
    end_of_test: str = "leave_open",
    margin_config_path: str | Path | None = None,
    trades_out_path: str | Path | None = None,
    fills_out_path: str | Path | None = None,
) -> list[str]:
    """Build the exact argv for one ``quant_paper_trading_targets_csv``
    invocation over an already-written bundle. Pure -- no I/O, no subprocess."""
    if end_of_test not in ("leave_open", "force_liquidate"):
        raise ValueError(f"end_of_test must be 'leave_open' or 'force_liquidate', got {end_of_test!r}")

    argv = [
        str(executable),
        str(bars_path),
        str(contracts_path),
        str(targets_path),
        str(risk_config_path),
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
    if validation_days_path is not None or roll_close_marks_path is not None:
        if validation_days_path is None:
            raise ValueError(
                "roll_close_marks_path requires validation_days_path to also be set "
                "(the C++ CLI positional order needs the validation-days slot filled)"
            )
        argv.append(str(validation_days_path))
    if roll_close_marks_path is not None:
        argv.append(str(roll_close_marks_path))
    if margin_config_path is not None:
        argv.append(f"--margin-config={margin_config_path}")
    if trades_out_path is not None:
        argv.append(f"--trades-out={trades_out_path}")
    if fills_out_path is not None:
        argv.append(f"--fills-out={fills_out_path}")
    argv.append(f"--end-of-test={end_of_test}")
    return argv


def invoke_paper_trading_cli(argv: list[str]) -> tuple[dict, str]:
    """Run one already-built argv and return ``(parsed_json, raw_stdout)``.
    ``raw_stdout`` is the literal text the CLI process emitted -- hash it
    directly for provenance rather than re-serializing the parsed dict."""
    proc = subprocess.run(argv, check=True, capture_output=True, text=True)
    raw = proc.stdout.strip()
    return json.loads(raw), raw


def run_paper_trading_backtest_cli(
    bars: pd.DataFrame,
    contracts: pd.DataFrame,
    schedule: TargetSchedule,
    risk_policy: PaperRiskPolicy,
    *,
    executable: str | Path,
    work_dir: str | Path,
    schedule_policy: str = "no_decision",
    commission_per_contract_usd: float | None = None,
    slippage_ticks: float | None = None,
    spread_ticks: float | None = None,
    validation_days_path: str | Path | None = None,
    roll_close_marks_path: str | Path | None = None,
    end_of_test: str = "leave_open",
    margin_config_path: str | Path | None = None,
    trades_out_path: str | Path | None = None,
    fills_out_path: str | Path | None = None,
) -> dict:
    """Convenience wrapper: write the bundle, build argv, invoke, return the
    parsed JSON only. :mod:`alpha_agent.paper.engine` calls the lower-level
    functions directly instead, so it can hash the exact bundle files and
    exact stdout for step provenance; this wrapper remains for any simpler
    caller (mirrors :func:`alpha_agent.adapters.targets_bridge
    .run_targets_backtest_cli`'s shape)."""
    bars_path, contracts_path, targets_path, risk_config_path = write_paper_trading_bundle(
        bars, contracts, schedule, risk_policy, work_dir
    )
    argv = build_paper_trading_argv(
        bars_path=bars_path,
        contracts_path=contracts_path,
        targets_path=targets_path,
        risk_config_path=risk_config_path,
        executable=executable,
        schedule_policy=schedule_policy,
        commission_per_contract_usd=commission_per_contract_usd,
        slippage_ticks=slippage_ticks,
        spread_ticks=spread_ticks,
        validation_days_path=validation_days_path,
        roll_close_marks_path=roll_close_marks_path,
        end_of_test=end_of_test,
        margin_config_path=margin_config_path,
        trades_out_path=trades_out_path,
        fills_out_path=fills_out_path,
    )
    result, _raw = invoke_paper_trading_cli(argv)
    return result
