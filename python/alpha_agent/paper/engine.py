"""Phase 21 / 21.1 -- the PaperTradingEngine.

target-position -> order -> simulated fill -> portfolio flow, entirely inside
the deterministic C++ core (cpp/include/quant_core/paper_trading_run.hpp /
``quant_paper_trading_targets_csv``). One "step" REPLAYS THE WHOLE WINDOW
``[window_start_ns, new_watermark_ns]`` from scratch through the C++ engine --
never an incremental / mutated Python-side state -- so every persisted number
is independently reproducible from (bars, schedule, risk policy) alone, and
Python never accumulates state the C++ core did not itself just compute. A
step's only Python-side work is: advance the watermark, rebuild the target
schedule for the (now longer) window, run the CLI, PROVE the new replay's fill
prefix matches everything already committed, diff the new fills since the
previous step, and persist -- ALL FOUR of the step snapshot, its new fills,
its authoritative C++ position snapshot, and its alerts, plus the run's
watermark/status, in ONE atomic ledger transaction
(:meth:`~alpha_agent.paper.ledger.PaperLedger.commit_step`). No PnL, fill,
position, or risk decision is computed in Python (CLAUDE.md architecture
boundary 1/3).

Phase 21.1 hardening on top of the Phase 21 flow:

* **exact per-step provenance** (:class:`~alpha_agent.paper.provenance
  .PaperStepProvenance`) -- SHA-256 of the literal bars/contracts/targets/
  validation-days bytes sent to the C++ boundary, the literal C++ executable,
  and the literal result payload, plus the parent experiment identity /
  strategy fingerprint / risk-policy identity / cost config this step ran
  under. Computed BEFORE any ledger write and stored WITH the step.
* **replay-prefix consistency** (:mod:`alpha_agent.paper.replay_integrity`) --
  before anything is written, the new replay's fills must begin with an exact
  copy of every fill already committed for this run. A shorter or disagreeing
  replay raises :class:`~alpha_agent.paper.errors.PaperReplayDivergence` and
  writes NOTHING: no step, no fill, no position, no alert, no watermark move.
* **authoritative position snapshot** -- the C++ CLI's own
  ``portfolio_at_end.positions`` (Phase 08 ``PositionExposure``, already
  computed by the C++ PortfolioAccountant) is persisted verbatim per step;
  this module never reconstructs a position from fills.
"""
from __future__ import annotations

import csv
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from alpha_agent.paper.alerts import Alert, generate_step_alerts, kill_switch_active
from alpha_agent.paper.bridge import (
    build_paper_trading_argv,
    invoke_paper_trading_cli,
    write_paper_trading_bundle,
)
from alpha_agent.paper.drift import DriftReport, build_drift_report, paper_backtest_run
from alpha_agent.paper.eligibility import assert_paper_trading_eligible
from alpha_agent.paper.errors import PaperDataWindowExhausted, PaperRunStateError
from alpha_agent.paper.ledger import (
    PaperAlertRow,
    PaperFillRow,
    PaperLedger,
    PaperPositionRow,
    PaperRunRow,
    PaperStepRow,
)
from alpha_agent.paper.provenance import PaperStepProvenance, sha256_bytes, sha256_file
from alpha_agent.paper.replay_integrity import assert_replay_prefix_consistent
from alpha_agent.paper.risk_policy import PaperRiskPolicy
from alpha_agent.paper.schedule_window import MarketWindowProvider, build_schedule_window
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.strategy.candidates_phase_13_5c import spec_for_params
from alpha_agent.validation.trading_day import build_validation_day_plan

DEFAULT_PAPER_TRADING_CLI = Path("build/cpp/cpp/quant_paper_trading_targets_csv")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StepSnapshot(NamedTuple):
    run_id: str
    step_ordinal: int
    as_of_ts_ns: int
    status: str
    kill_switch_active: bool
    drift: DriftReport
    alerts: tuple[Alert, ...]
    provenance: PaperStepProvenance


@contextmanager
def _work_dir(base: Path | None) -> Iterator[Path]:
    if base is None:
        with tempfile.TemporaryDirectory(prefix="paper_step_") as d:
            yield Path(d)
    else:
        base.mkdir(parents=True, exist_ok=True)
        yield base


def _read_fills_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _position_rows(run_id: str, step_ordinal: int, cpp_json: dict) -> list[PaperPositionRow]:
    """Parse the CLI's ``"positions"`` array (a verbatim serialization of
    ``BacktestResult::portfolio_at_end.positions``) into typed rows. Copies
    every field as given -- computes nothing."""
    out: list[PaperPositionRow] = []
    for p in cpp_json.get("positions", []):
        out.append(
            PaperPositionRow(
                run_id=run_id,
                step_ordinal=step_ordinal,
                instrument_id=int(p["instrument_id"]),
                raw_symbol=str(p["raw_symbol"]),
                root_symbol=str(p["root_symbol"]),
                units=int(p["units"]),
                avg_entry_price=float(p["avg_entry_price"]),
                multiplier=float(p["multiplier"]),
                mark_price=float(p["mark_price"]),
                mark_ts_ns=int(p["mark_ts_ns"]),
                mark_age_ns=int(p["mark_age_ns"]),
                mark_present=bool(p["mark_present"]),
                mark_is_stale=bool(p["mark_is_stale"]),
                valuation_is_estimated=bool(p["valuation_is_estimated"]),
                gross_notional_usd=float(p["gross_notional_usd"]),
                signed_notional_usd=float(p["signed_notional_usd"]),
                unrealized_pnl_usd=float(p["unrealized_pnl_usd"]),
                initial_margin_usd=float(p["initial_margin_usd"]),
                maintenance_margin_usd=float(p["maintenance_margin_usd"]),
                margin_known=bool(p["margin_known"]),
            )
        )
    return out


class PaperTradingEngine:
    """Orchestrates deterministic paper-trading steps for runs held in
    ``ledger``. Reads the experiment registry ONLY at :meth:`start_run` --
    every later step replays purely from the immutable snapshot taken then, so
    a step never depends on the registry being reachable or unchanged."""

    def __init__(
        self,
        *,
        registry: ExperimentRegistry,
        ledger: PaperLedger,
        provider: MarketWindowProvider,
        executable: str | Path = DEFAULT_PAPER_TRADING_CLI,
        work_dir: str | Path | None = None,
        data_source_label: str = "real_2018_2024_replay",
    ):
        self._registry = registry
        self._ledger = ledger
        self._provider = provider
        self._executable = Path(executable)
        self._work_dir = Path(work_dir) if work_dir is not None else None
        self._data_source_label = data_source_label

    # -- lifecycle --------------------------------------------------------
    def start_run(
        self,
        *,
        experiment_key: str,
        window_start_ns: int,
        window_end_ns: int,
        risk_policy: PaperRiskPolicy | None = None,
        schedule_policy: str = "no_decision",
        commission_per_contract_usd: float = 2.0,
        slippage_ticks: float = 0.0,
        spread_ticks: float = 0.0,
    ) -> str:
        if window_end_ns <= window_start_ns:
            raise ValueError("window_end_ns must be after window_start_ns")
        eligibility = assert_paper_trading_eligible(self._registry, experiment_key)
        policy = risk_policy or PaperRiskPolicy()

        run_id = self._ledger.create_run(
            experiment_id=eligibility.experiment_id,
            experiment_identity=eligibility.experiment_identity,
            strategy_family=eligibility.strategy_family,
            strategy_id=eligibility.strategy_id,
            strategy_fingerprint=eligibility.strategy_fingerprint,
            root_symbol=eligibility.root_symbol,
            params=eligibility.params,
            risk_policy=policy.model_dump(mode="json"),
            risk_policy_identity=policy.identity(),
            schedule_policy=schedule_policy,
            commission_per_contract_usd=commission_per_contract_usd,
            slippage_ticks=slippage_ticks,
            spread_ticks=spread_ticks,
            window_start_ns=window_start_ns,
            window_end_ns=window_end_ns,
            data_source=self._data_source_label,
            validation_result={
                "backtest_daily_sharpe": eligibility.backtest_daily_sharpe,
                "backtest_annualized_sharpe": eligibility.backtest_annualized_sharpe,
                "backtest_net_pnl_usd": eligibility.backtest_net_pnl_usd,
                "backtest_n_trades": eligibility.backtest_n_trades,
                "reason_codes": list(eligibility.validation_report_reason_codes),
            },
        )
        self._ledger.append_alerts(
            [
                PaperAlertRow(
                    run_id=run_id,
                    alert_seq=0,
                    step_ordinal=-1,
                    ts_ns=window_start_ns,
                    created_at=_now(),
                    alert_type="RUN_STARTED",
                    severity="INFO",
                    message=(
                        f"Paper run started for {eligibility.experiment_id} "
                        f"({eligibility.root_symbol}, {eligibility.strategy_family})."
                    ),
                    detail={"experiment_identity": eligibility.experiment_identity},
                )
            ]
        )
        return run_id

    def step(self, run_id: str, *, n_trading_days: int = 1) -> StepSnapshot:
        """Advance the run's watermark by up to ``n_trading_days`` trading days
        and replay. Raises :class:`PaperDataWindowExhausted` if the window has
        no further trading day to advance into."""
        if n_trading_days < 1:
            raise ValueError("n_trading_days must be >= 1")
        run = self._ledger.get_run(run_id)
        if run.status not in ("ACTIVE", "HALTED"):
            raise PaperRunStateError(f"paper run {run_id} is {run.status}; cannot step it")

        full_bars = self._provider.execution_bars(
            run.root_symbol, run.window_start_ns, run.window_end_ns
        )
        plan = build_validation_day_plan(full_bars, root_symbol=run.root_symbol)
        boundaries = plan.boundary_ts_list()
        remaining = [b for b in boundaries if b > run.watermark_ns]
        if not remaining:
            raise PaperDataWindowExhausted(
                f"paper run {run_id} has no further trading day beyond watermark "
                f"{run.watermark_ns} in window [{run.window_start_ns}, {run.window_end_ns}]"
            )
        new_watermark = remaining[min(n_trading_days, len(remaining)) - 1]
        window_exhausted = new_watermark >= boundaries[-1]
        return self._run_step(
            run, new_watermark_ns=new_watermark, force_liquidate=False,
            window_exhausted=window_exhausted,
        )

    def stop_run(self, run_id: str) -> StepSnapshot:
        """One final replay through the run's CURRENT watermark with
        ``end_of_test=force_liquidate`` (flatten every open position), then
        marks the run STOPPED. Does not require any further trading day."""
        run = self._ledger.get_run(run_id)
        if run.status == "STOPPED":
            raise PaperRunStateError(f"paper run {run_id} is already STOPPED")
        return self._run_step(
            run, new_watermark_ns=run.watermark_ns, force_liquidate=True, window_exhausted=False
        )

    # -- internals ----------------------------------------------------------
    def _run_step(
        self, run: PaperRunRow, *, new_watermark_ns: int, force_liquidate: bool, window_exhausted: bool
    ) -> StepSnapshot:
        run_id = run.run_id
        strategy_spec = spec_for_params(run.strategy_family, run.params)
        window = build_schedule_window(
            strategy_family=run.strategy_family,
            root_symbol=run.root_symbol,
            strategy_spec=strategy_spec,
            provider=self._provider,
            window_start_ns=run.window_start_ns,
            watermark_ns=new_watermark_ns,
            emit_from_ts_ns=run.window_start_ns,
        )
        if window.schedule is None:
            raise PaperDataWindowExhausted(
                f"paper run {run_id}: the signal window through {new_watermark_ns} is "
                "still too short (<3 daily rows) for any strategy decision"
            )
        window_day_plan = build_validation_day_plan(window.bars, root_symbol=run.root_symbol)
        policy = PaperRiskPolicy.model_validate(run.risk_policy)
        step_ordinal = self._ledger.next_step_ordinal(run_id)
        step_dir = None if self._work_dir is None else self._work_dir / run_id / f"step_{step_ordinal:05d}"
        end_of_test = "force_liquidate" if force_liquidate else "leave_open"

        with _work_dir(step_dir) as wd:
            trades_out = wd / "trades.csv"
            fills_out = wd / "fills.csv"
            vdays_path = window_day_plan.write_csv(wd / "validation_days.csv")
            bars_path, contracts_path, targets_path, risk_config_path = write_paper_trading_bundle(
                window.bars, window.contracts, window.schedule, policy, wd
            )
            argv = build_paper_trading_argv(
                bars_path=bars_path,
                contracts_path=contracts_path,
                targets_path=targets_path,
                risk_config_path=risk_config_path,
                executable=self._executable,
                schedule_policy=run.schedule_policy,
                commission_per_contract_usd=run.commission_per_contract_usd,
                slippage_ticks=run.slippage_ticks,
                spread_ticks=run.spread_ticks,
                validation_days_path=vdays_path,
                end_of_test=end_of_test,
                trades_out_path=trades_out,
                fills_out_path=fills_out,
            )
            cpp_json, raw_stdout = invoke_paper_trading_cli(argv)
            fill_rows = _read_fills_csv(fills_out)

            # Exact per-step provenance: hashed BEFORE any ledger write, from
            # the literal bytes actually sent to (and produced by) the C++
            # boundary this step (Phase 21.1).
            provenance = PaperStepProvenance(
                parent_experiment_identity=run.experiment_identity,
                strategy_fingerprint=run.strategy_fingerprint,
                risk_policy_identity=run.risk_policy_identity,
                schedule_policy=run.schedule_policy,
                commission_per_contract_usd=run.commission_per_contract_usd,
                slippage_ticks=run.slippage_ticks,
                spread_ticks=run.spread_ticks,
                end_of_test=end_of_test,
                bars_sha256=sha256_file(bars_path),
                contracts_sha256=sha256_file(contracts_path),
                targets_sha256=sha256_file(targets_path),
                target_schedule_hash=window.schedule.schedule_hash(),
                validation_days_sha256=sha256_file(vdays_path),
                validation_day_plan_identity=window_day_plan.identity(),
                roll_close_marks_sha256=None,
                executable_path=str(self._executable),
                executable_sha256=sha256_file(self._executable),
                result_payload_sha256=sha256_bytes(raw_stdout.encode("utf-8")),
            )

        # Replay-prefix consistency: proven BEFORE any ledger write. A
        # divergence raises and NOTHING below runs -- no step, no fill, no
        # position, no alert, no watermark/status move (Phase 21.1).
        committed_fills = self._ledger.list_fills(run_id)
        assert_replay_prefix_consistent(
            run_id=run_id, committed=committed_fills, replayed_csv_rows=fill_rows
        )
        prev_fill_count = len(committed_fills)
        new_fills = [
            PaperFillRow(
                run_id=run_id,
                fill_seq=prev_fill_count + i,
                step_ordinal=step_ordinal,
                fill_id=int(row["fill_id"]),
                order_id=int(row["order_id"]),
                ts_fill_ns=int(row["ts_fill_ns"]),
                instrument_id=int(row["instrument_id"]),
                raw_symbol=row["raw_symbol"],
                side=row["side"],
                quantity=int(row["quantity"]),
                fill_price=float(row["fill_price"]),
                commission_usd=float(row["commission_usd"]),
                slippage_ticks=float(row["slippage_ticks"]),
            )
            for i, row in enumerate(fill_rows[prev_fill_count:])
        ]

        margin_complete = bool(cpp_json.get("margin_complete", False))
        kill_now = kill_switch_active(run.risk_policy, cpp_json)
        prev_step = self._ledger.latest_step(run_id)
        kill_before = bool(prev_step.kill_switch_active) if prev_step is not None else False

        paper_run = paper_backtest_run(
            cpp_json,
            capital_base_usd=policy.starting_capital_usd,
            validation_day_plan=window_day_plan,
        )
        drift = build_drift_report(
            run_id=run_id,
            as_of_ts_ns=new_watermark_ns,
            cpp_json=cpp_json,
            paper_run=paper_run,
            fills=list(committed_fills) + new_fills,
            contracts=window.contracts,
            elapsed_trading_days=window_day_plan.n_days,
            backtest_daily_sharpe=run.validation_result.get("backtest_daily_sharpe"),
            backtest_annualized_sharpe=run.validation_result.get("backtest_annualized_sharpe"),
            backtest_net_pnl_usd=run.validation_result.get("backtest_net_pnl_usd"),
            backtest_n_trades=run.validation_result.get("backtest_n_trades"),
        )

        positions = _position_rows(run_id, step_ordinal, cpp_json)

        step_row = PaperStepRow(
            run_id=run_id,
            step_ordinal=step_ordinal,
            as_of_ts_ns=new_watermark_ns,
            created_at=_now(),
            bars_processed=int(cpp_json.get("bars", 0)),
            orders_generated=int(cpp_json.get("orders", 0)),
            fills_generated=int(cpp_json.get("fills", 0)),
            trades_closed=int(cpp_json.get("trades", 0)),
            gross_pnl_usd=float(cpp_json.get("gross_pnl_usd", 0.0)),
            costs_usd=float(cpp_json.get("costs_usd", 0.0)),
            net_pnl_usd=float(cpp_json.get("net_pnl_usd", 0.0)),
            unrealized_pnl_usd=float(cpp_json.get("unrealized_pnl_usd", 0.0)),
            equity_usd=float(cpp_json.get("equity_usd", 0.0)),
            peak_equity_usd=float(cpp_json.get("peak_equity_usd", 0.0)),
            drawdown_usd=float(cpp_json.get("portfolio_drawdown_usd", 0.0)),
            drawdown_pct=float(cpp_json.get("portfolio_drawdown_pct", 0.0)),
            day_realized_pnl_usd=float(cpp_json.get("day_realized_pnl_usd", 0.0)),
            risk_rejects=int(cpp_json.get("risk_rejects", 0)),
            risk_resizes=int(cpp_json.get("risk_resizes", 0)),
            kill_switch_active=kill_now,
            margin_complete=margin_complete,
            open_positions=int(cpp_json.get("open_positions", 0)),
            elapsed_trading_days=window_day_plan.n_days,
            daily_sharpe=drift.paper_daily_sharpe,
            annualized_sharpe=drift.paper_annualized_sharpe,
            provenance=provenance.model_dump(mode="json"),
            drift=drift.model_dump(mode="json"),
            raw_result=cpp_json,
        )

        alerts = generate_step_alerts(
            kill_switch_now=kill_now,
            kill_switch_before=kill_before,
            margin_complete=margin_complete,
            window_exhausted=window_exhausted,
            drift_flags=drift.flags,
        )
        next_seq = self._ledger.next_alert_seq(run_id)
        alert_rows = [
            PaperAlertRow(
                run_id=run_id,
                alert_seq=next_seq + i,
                step_ordinal=step_ordinal,
                ts_ns=new_watermark_ns,
                created_at=_now(),
                alert_type=a.alert_type.value,
                severity=a.severity.value,
                message=a.message,
                detail=a.detail,
            )
            for i, a in enumerate(alerts)
        ]

        new_status = "STOPPED" if force_liquidate else ("HALTED" if kill_now else "ACTIVE")
        # ATOMIC: step + fills + positions + alerts + run watermark/status all
        # commit together or none do (Phase 21.1).
        self._ledger.commit_step(
            run_id=run_id,
            step=step_row,
            fills=new_fills,
            positions=positions,
            alerts=alert_rows,
            watermark_ns=new_watermark_ns,
            status=new_status,
            stopped_at=_now() if force_liquidate else None,
        )

        return StepSnapshot(
            run_id=run_id,
            step_ordinal=step_ordinal,
            as_of_ts_ns=new_watermark_ns,
            status=new_status,
            kill_switch_active=kill_now,
            drift=drift,
            alerts=tuple(alerts),
            provenance=provenance,
        )
