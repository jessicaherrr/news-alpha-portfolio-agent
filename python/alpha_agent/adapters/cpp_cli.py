"""CLI boundary between the Python research layer and the C++ quant core.

This is deliberately the temporary transport (CSV + subprocess). pybind11 comes
later, only once the engine API has stabilised. The bar columns and the contract
columns written here are the frozen boundary contract (docs/BOUNDARY_CONTRACT.md
section G).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from alpha_agent.schemas.market_data import (
    BOUNDARY_BAR_COLUMNS,
    CANONICAL_BAR_COLUMNS,
    CONTRACT_COLUMNS,
)


def canonical_bars_to_boundary(canonical: pd.DataFrame) -> pd.DataFrame:
    """Project canonical bars (``CANONICAL_BAR_COLUMNS``) down to the frozen CLI
    bar columns without touching values. The dropped columns (raw_symbol,
    root_symbol, trading_day, session) are recoverable: raw_symbol/root from the
    contracts file via instrument_id, trading_day/session from the calendar.
    """
    missing = [c for c in CANONICAL_BAR_COLUMNS if c not in canonical.columns]
    if missing:
        raise ValueError(f"not a canonical bar frame, missing: {missing}")
    return canonical.loc[:, list(BOUNDARY_BAR_COLUMNS)].reset_index(drop=True)


def write_boundary_bundle(
    bars: pd.DataFrame,
    contracts: pd.DataFrame,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write ``bars.csv`` + ``contracts.csv`` in the frozen column order.

    Every ``instrument_id`` in ``bars`` must be present in ``contracts`` -- the
    C++ core rejects a bundle where a bar cannot be resolved to a real contract.
    """
    missing_bar = [c for c in BOUNDARY_BAR_COLUMNS if c not in bars.columns]
    if missing_bar:
        raise ValueError(f"bars is missing boundary columns: {missing_bar}")
    missing_contract = [c for c in CONTRACT_COLUMNS if c not in contracts.columns]
    if missing_contract:
        raise ValueError(f"contracts is missing columns: {missing_contract}")

    bar_ids = set(bars["instrument_id"].unique())
    known_ids = set(contracts["instrument_id"].unique())
    unresolved = bar_ids - known_ids
    if unresolved:
        raise ValueError(f"bars reference instrument_id(s) with no contract row: {sorted(unresolved)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars_path = out_dir / "bars.csv"
    contracts_path = out_dir / "contracts.csv"
    bars.loc[:, list(BOUNDARY_BAR_COLUMNS)].to_csv(bars_path, index=False)
    contracts.loc[:, list(CONTRACT_COLUMNS)].to_csv(contracts_path, index=False)
    return bars_path, contracts_path


def run_momentum_backtest_cli(
    bars: pd.DataFrame,
    contracts: pd.DataFrame,
    *,
    executable: str | Path,
    lookback: int,
    threshold_return: float,
    work_dir: str | Path,
) -> dict:
    """Run the reference C++ momentum backtest over a boundary bundle.

    The C++ side loads the contract registry, verifies every bar resolves to a
    real contract that was live at the bar timestamp, then runs the backtest.
    """
    bars_path, contracts_path = write_boundary_bundle(bars, contracts, work_dir)
    proc = subprocess.run(
        [
            str(executable),
            str(bars_path),
            str(contracts_path),
            str(lookback),
            str(threshold_return),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout.strip())
