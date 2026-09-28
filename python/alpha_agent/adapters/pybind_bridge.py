"""Phase 19 -- in-process pybind11 boundary to the C++ quant core.

The high-volume transport that replaces the CSV-bundle + subprocess + JSON round
trip of :mod:`alpha_agent.adapters.targets_bridge` for research that runs the
scheduled-target engine thousands of times (the Phase 13.5C matrix, Phase 15B
economics trials, sensitivity sweeps).

**The CLI (``quant_backtest_targets_csv`` /
:func:`alpha_agent.adapters.targets_bridge.run_targets_backtest_cli`) remains the
frozen reference implementation and is unchanged.** This path uses the thin C++
helper ``quant::run_targets_backtest`` (``quant_core/targets_run.hpp``), whose
engine wiring MIRRORS the CLI's -- the same frozen boundary parsers
(``load_contract_registry``, ``parse_targets_csv``), the same deterministic
``BacktestEngine`` / ``PassThroughRiskManager`` /
``EndOfTestPolicy::ForceLiquidateFinalClose``. The helper and the CLI are wired
independently (the CLI is not refactored onto the helper), so semantic
equivalence is an ENFORCED invariant, not a structural one: the deterministic
CLI-vs-pybind parity tests (``tests/python/test_phase_19_pybind.py``) run both
transports on the same fixtures and assert every result field agrees, and
:func:`assert_boundary_parity` is the check they use.

Only the bar array crosses as typed NumPy columns instead of ``bars.csv``.
Official PnL / fill / risk / accounting authority stays entirely in the C++ core;
Python computes none of it and only reads what the engine already decided
(CLAUDE.md architecture boundary 3). :func:`run_targets_backtest_pybind` is a
drop-in for ``run_targets_backtest_cli``.

PROVENANCE. :func:`pybind_boundary_provenance` returns the transport tag, ABI
string and the compiled module's SHA-256 so any registry evidence produced over
this boundary records *which artifact* the economics came from -- exactly as the
CLI path records ``source_artifact`` / ``source_artifact_sha256``. This boundary
does not carry, weaken or bypass the ``ValidationSpec`` /
``ReliabilityPolicy`` fingerprint-equality checks: those live above the engine
call (``alpha_agent.validation.engine`` / ``.holdout``) and apply unchanged
whichever transport produced the daily-return series. It is deliberately NOT
wired into the frozen Phase 15B ``CppEngineRunner`` real path (Phase 15B.2, still
gated); it serves offline research, which is evidence, never a BH/FDR trial.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_agent.backtest.targets import TargetSchedule
from alpha_agent.schemas.market_data import BOUNDARY_BAR_COLUMNS, CONTRACT_COLUMNS

MODULE_NAME = "quant_core_py"
BOUNDARY_TRANSPORT = "cpp_pybind__quant_core_py"
REFERENCE_TRANSPORT = "cpp_cli__quant_backtest_targets_csv"

#: Default locations a locally built module may sit in (scripts/build_pybind.sh
#: writes the first). Overridable with $QUANT_CORE_PY_DIR.
_DEFAULT_BUILD_DIRS = ("build/cpp", "build/cpp/cpp", "build")

# The result keys quant_backtest_targets_csv emits on its one JSON line, in
# order. run_targets_backtest_pybind returns exactly this set so it is a literal
# drop-in for run_targets_backtest_cli (docs/BOUNDARY_CONTRACT.md section G).
_COUNT_KEYS = (
    "target_rows", "target_rows_applied", "trades", "events", "signals", "orders",
    "fills", "rolls", "rolls_priced_contemporaneous", "rolls_priced_auxiliary_marks",
    "rolls_priced_stale", "rolls_deferred", "non_fills", "eot_liquidations",
    "unique_contracts", "open_positions", "risk_rejects", "risk_resizes", "bars",
    "contracts_resolved",
)
_FLOAT_KEYS = (
    "gross_pnl_usd", "costs_usd", "net_pnl_usd", "max_drawdown_usd", "unrealized_pnl_usd",
    "slippage_ticks", "commission_per_contract_usd", "starting_capital_usd", "cash_usd",
    "equity_usd", "gross_exposure_usd", "net_exposure_usd", "gross_leverage",
    "peak_equity_usd", "portfolio_drawdown_usd", "portfolio_drawdown_pct",
    "net_equity_usd_at_end",
)
_STR_KEYS = ("strategy_fingerprint", "schedule_policy", "daily_equity_basis")


# --------------------------------------------------------------------------
# module discovery
# --------------------------------------------------------------------------
def _candidate_dirs() -> list[Path]:
    dirs: list[str] = []
    env = os.environ.get("QUANT_CORE_PY_DIR")
    if env:
        dirs.append(env)
    dirs.extend(_DEFAULT_BUILD_DIRS)
    return [Path(d) for d in dirs]


@lru_cache(maxsize=1)
def _load_module():
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError:
        pass
    for d in _candidate_dirs():
        for suffix in (".so", ".pyd", ".dylib"):
            for hit in sorted(d.glob(f"{MODULE_NAME}*{suffix}")):
                spec = importlib.util.spec_from_file_location(MODULE_NAME, hit)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[MODULE_NAME] = mod
                    spec.loader.exec_module(mod)
                    mod.__quant_core_py_path__ = str(hit.resolve())
                    return mod
    raise ModuleNotFoundError(
        f"the optional pybind module '{MODULE_NAME}' is not importable. Build it with "
        "`scripts/build_pybind.sh` (needs `pip install pybind11`), or set "
        "$QUANT_CORE_PY_DIR to the directory holding the compiled module. The CLI "
        "path (targets_bridge.run_targets_backtest_cli) needs no build."
    )


def is_pybind_available() -> bool:
    """True iff :data:`MODULE_NAME` can be imported (does not raise)."""
    try:
        _load_module()
        return True
    except ModuleNotFoundError:
        return False


def pybind_module():
    """The imported pybind module, or raise ``ModuleNotFoundError`` with build help."""
    return _load_module()


def _module_file(mod) -> Path:
    return Path(getattr(mod, "__quant_core_py_path__", None) or mod.__file__)


def pybind_boundary_provenance() -> dict:
    """Transport identity for anything that records pybind-sourced economics.

    ``module_sha256`` is the SHA-256 of the compiled shared object -- the
    artifact that actually produced the numbers. Recorded next to a result the
    way the CLI path records the executable / bundle it used; never part of a
    scientific ``experiment_identity``.
    """
    mod = _load_module()
    f = _module_file(mod)
    digest = hashlib.sha256(f.read_bytes()).hexdigest()
    return {
        "transport": BOUNDARY_TRANSPORT,
        "reference_transport": REFERENCE_TRANSPORT,
        "boundary_abi": str(mod.BOUNDARY_ABI),
        "engine_path": str(mod.ENGINE_PATH),
        "module_file": str(f),
        "module_sha256": digest,
    }


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def _write_contracts_targets(
    contracts: pd.DataFrame, schedule: TargetSchedule, out_dir: Path
) -> tuple[Path, Path]:
    missing = [c for c in CONTRACT_COLUMNS if c not in contracts.columns]
    if missing:
        raise ValueError(f"contracts is missing columns: {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    contracts_path = out_dir / "contracts.csv"
    contracts.loc[:, list(CONTRACT_COLUMNS)].to_csv(contracts_path, index=False)
    targets_path = schedule.write_csv(out_dir / "targets.csv")
    return contracts_path, Path(targets_path)


def _read_validation_days(path: str | Path) -> list[int]:
    df = pd.read_csv(path)
    if list(df.columns) != ["boundary_ts_ns"]:
        raise ValueError("validation-days csv header must be exactly 'boundary_ts_ns'")
    return [int(v) for v in df["boundary_ts_ns"].tolist()]


def _read_roll_close_marks(path: str | Path) -> list[tuple[int, int, float]]:
    df = pd.read_csv(path)
    if list(df.columns) != ["instrument_id", "ts_event_ns", "close"]:
        raise ValueError(
            "roll-close-marks csv header must be exactly 'instrument_id,ts_event_ns,close'"
        )
    return [
        (int(r.instrument_id), int(r.ts_event_ns), float(r.close))
        for r in df.itertuples(index=False)
    ]


def run_targets_backtest_pybind(
    bars: pd.DataFrame,
    contracts: pd.DataFrame,
    schedule: TargetSchedule,
    *,
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
    """In-process drop-in for
    :func:`alpha_agent.adapters.targets_bridge.run_targets_backtest_cli` (the
    frozen reference).

    Same keyword contract (minus ``executable``), same return dict. ``bars`` must
    carry :data:`BOUNDARY_BAR_COLUMNS`; it is passed to C++ as typed columns, not
    written to disk. ``contracts`` and the (small) target schedule are written
    into ``work_dir`` so the frozen C++ boundary parsers read them verbatim.

    The three cost values default to the frozen reference assumptions
    ``(2.0, 0.0, 0.0)`` when ``None`` -- matching the CLI bridge.

    This calls the thin ``quant::run_targets_backtest`` helper, whose wiring
    mirrors the CLI's; the two are held in agreement by the CLI-vs-pybind parity
    tests, not by shared execution code. Use ``run_targets_backtest_cli`` when
    you need the frozen reference path itself.
    """
    missing = [c for c in BOUNDARY_BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing boundary columns: {missing}")

    mod = _load_module()
    work_dir = Path(work_dir)
    contracts_path, targets_path = _write_contracts_targets(contracts, schedule, work_dir)

    commission = 2.0 if commission_per_contract_usd is None else float(commission_per_contract_usd)
    slippage = 0.0 if slippage_ticks is None else float(slippage_ticks)
    spread = 0.0 if spread_ticks is None else float(spread_ticks)

    if roll_close_marks_path is not None and validation_days_path is None:
        raise ValueError(
            "roll_close_marks_path requires validation_days_path to also be set "
            "(matches the C++ CLI positional-order rule)"
        )
    vdays: list[int] = (
        _read_validation_days(validation_days_path) if validation_days_path is not None else []
    )
    marks: list[tuple[int, int, float]] = (
        _read_roll_close_marks(roll_close_marks_path)
        if roll_close_marks_path is not None
        else []
    )

    b = bars.loc[:, list(BOUNDARY_BAR_COLUMNS)]
    result = mod.run_scheduled_targets(
        np.ascontiguousarray(b["ts_event_ns"].to_numpy(dtype=np.int64)),
        np.ascontiguousarray(b["instrument_id"].to_numpy(dtype=np.int64)),
        np.ascontiguousarray(b["open"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(b["high"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(b["low"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(b["close"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(b["volume"].to_numpy(dtype=np.int64)),
        str(contracts_path),
        str(targets_path),
        schedule_policy,
        commission,
        slippage,
        spread,
        vdays,
        marks,
    )

    if trades_out_path is not None:
        result.write_trades_csv(str(trades_out_path))
    if fills_out_path is not None:
        result.write_fills_csv(str(fills_out_path))

    out: dict = {}
    for k in _STR_KEYS:
        out[k] = str(getattr(result, k))
    for k in _COUNT_KEYS:
        out[k] = int(getattr(result, k))
    for k in _FLOAT_KEYS:
        out[k] = float(getattr(result, k))
    out["daily_equity"] = [list(row) for row in result.daily_equity]
    return out


# --------------------------------------------------------------------------
# parity
# --------------------------------------------------------------------------
def assert_boundary_parity(
    cli_result: dict,
    pybind_result: dict,
    *,
    float_rtol: float = 2e-5,
    float_atol: float = 5e-4,
) -> None:
    """Assert a CLI result dict and a pybind result dict describe the same run.

    This is the deterministic check that enforces CLI/pybind equivalence (the
    two transports are wired independently -- see :mod:`~alpha_agent.adapters.pybind_bridge`).
    Counts / strings must match exactly. Floats are compared with a tolerance
    that absorbs only the CLI's ``std::ostream`` default-precision serialisation
    (6 significant digits) vs the pybind path's full ``f64``; any residual beyond
    that is a real disagreement between the two wiring sites and fails here.
    """
    shared = set(cli_result) & set(pybind_result)
    assert shared, "no overlapping keys between CLI and pybind results"

    for k in sorted(shared):
        a, b = cli_result[k], pybind_result[k]
        if k == "daily_equity":
            assert len(a) == len(b), f"daily_equity length {len(a)} != {len(b)}"
            for i, (ra, rb) in enumerate(zip(a, b)):
                assert len(ra) == len(rb) == 8, f"daily_equity[{i}] wrong width"
                # cols 0,1,6,7 are integers; 2..5 are money
                for j in (0, 1, 6, 7):
                    assert int(ra[j]) == int(rb[j]), f"daily_equity[{i}][{j}] {ra[j]} != {rb[j]}"
                for j in (2, 3, 4, 5):
                    assert np.isclose(ra[j], rb[j], rtol=float_rtol, atol=float_atol), (
                        f"daily_equity[{i}][{j}] {ra[j]} != {rb[j]}"
                    )
            continue
        if isinstance(a, float) or isinstance(b, float):
            assert np.isclose(a, b, rtol=float_rtol, atol=float_atol), f"{k}: {a} != {b}"
        else:
            assert a == b, f"{k}: {a!r} != {b!r}"
