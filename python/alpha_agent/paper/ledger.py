"""Phase 21 -- the persistent paper-trading ledger.

A separate, append-oriented SQLite store
(``data/paper_trading/paper_ledger.sqlite`` by default) -- deliberately NOT the
Phase 14 experiment registry (``data/registry/experiments.sqlite``). Paper
trading is operational monitoring of an already-approved strategy, not a new
statistical hypothesis: it must never weaken, extend, or be confused with the
registry's scientific identity / BH-FDR / failure-memory semantics
(CLAUDE.md). This module reads the registry (via
:mod:`alpha_agent.paper.eligibility`) but never writes to it.

Every number stored here is copied VERBATIM from a C++
``quant_paper_trading_targets_csv`` JSON result -- this module computes no
PnL, fill, or risk decision. ``paper_runs`` / ``paper_steps`` / ``paper_fills``
/ ``paper_positions`` / ``paper_alerts`` are append-only except for the small,
explicitly mutable run-state columns on ``paper_runs`` (``watermark_ns``,
``status``, ``stopped_at``) -- operational state, not a scientific result.

Phase 21.1 adds atomic multi-table commits (:meth:`PaperLedger.commit_step`):
a step's ``PaperStepRow`` snapshot, its new fills, its authoritative C++
position snapshot, and its alerts must all commit together or none do, and the
run's watermark/status must move in the SAME transaction -- never separately,
so a crash between them can never leave a step half-recorded. Every public
single-purpose ``append_*`` / ``update_run_state`` method still exists
(used directly by tests and by anything that only needs one write), each
wrapped in its own one-operation transaction; ``commit_step`` composes the
same private, transaction-free primitives inside ONE transaction instead.

Phase 21.1b -- VERSIONED LEDGER SCHEMA MIGRATION. Phase 21 (commit 1087724)
shipped an unversioned ``paper_steps`` shape (no ``elapsed_trading_days`` /
``provenance_json`` / ``drift_json`` columns, no ``paper_positions`` table).
``CREATE TABLE IF NOT EXISTS`` alone cannot add columns to an
already-existing table, so a real ledger created by 1087724 would silently
keep its OLD ``paper_steps`` shape forever, and every 21.1 write naming the
new columns would fail against it. This module now tracks an explicit
``CURRENT_SCHEMA_VERSION`` ("paper-ledger/2") via ``PRAGMA user_version`` and
runs :func:`_migrate` on every open: a brand-new database is created
directly at the latest version; an existing pre-21.1 (``user_version == 0``)
database is migrated in place, in one transaction, deterministically and
idempotently -- re-opening an already-migrated database changes nothing.
Migrated legacy rows never get a fabricated provenance/drift value: their
``provenance_json`` / ``drift_json`` are set to an explicit
``LEGACY_PROVENANCE`` / ``LEGACY_DRIFT`` marker (``status ==
"NOT_AVAILABLE"``, a `"schema_version"` ending in ``"/legacy"``) and their
``elapsed_trading_days`` is left ``NULL`` -- never ``0`` or ``"{}"``, which
would read as "recorded, and empty" rather than "not recorded at all".
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field

DEFAULT_LEDGER_PATH = Path("data/paper_trading/paper_ledger.sqlite")

#: the whole-database schema shape version, tracked via ``PRAGMA user_version``
#: (an integer stored in the SQLite file header -- not a table, so it survives
#: even a schema that has no rows yet). Phase 21 (1087724) never set this, so
#: any pre-21.1 database reads back 0 (SQLite's own default). Distinct from
#: ``paper_runs.schema_version`` ("paper-run/1"), which is a per-ROW shape
#: marker for ``PaperRunRow`` and has not changed.
CURRENT_SCHEMA_VERSION = 2
LEDGER_SCHEMA_LABEL = "paper-ledger/2"

#: explicit, non-fabricated markers for a step migrated from a pre-21.1
#: (schema version < 2) database, where per-step provenance / the completed
#: drift-monitor contract did not exist yet. Never used for a step actually
#: committed under the current engine -- those always carry real data.
LEGACY_PROVENANCE: dict = {
    "schema_version": "paper-step-provenance/legacy",
    "status": "NOT_AVAILABLE",
    "note": (
        "This step was committed under the Phase 21 (pre-21.1) paper-ledger "
        "schema, before exact per-step execution provenance existed. Nothing "
        "was fabricated for it."
    ),
}
LEGACY_DRIFT: dict = {
    "schema_version": "paper-drift-report/legacy",
    "status": "NOT_AVAILABLE",
    "note": (
        "This step was committed under the Phase 21 (pre-21.1) paper-ledger "
        "schema, before the completed drift-monitor contract (realized "
        "slippage / trade frequency / PnL distribution / comparison "
        "availability) existed. Nothing was fabricated for it."
    ),
}


def is_legacy_evidence(payload: dict) -> bool:
    """True iff ``payload`` (a step's ``provenance`` or ``drift`` dict) is one
    of the explicit migration markers above -- i.e. this evidence was never
    recorded, not merely empty."""
    return isinstance(payload, dict) and str(payload.get("schema_version", "")).endswith("/legacy")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_runs (
    run_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    experiment_identity TEXT NOT NULL,
    strategy_family TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    strategy_fingerprint TEXT NOT NULL,
    root_symbol TEXT NOT NULL,
    params_json TEXT NOT NULL,
    risk_policy_json TEXT NOT NULL,
    risk_policy_identity TEXT NOT NULL,
    schedule_policy TEXT NOT NULL,
    commission_per_contract_usd REAL NOT NULL,
    slippage_ticks REAL NOT NULL,
    spread_ticks REAL NOT NULL,
    window_start_ns INTEGER NOT NULL,
    window_end_ns INTEGER NOT NULL,
    watermark_ns INTEGER NOT NULL,
    status TEXT NOT NULL,
    data_source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    stopped_at TEXT,
    validation_result_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_steps (
    run_id TEXT NOT NULL,
    step_ordinal INTEGER NOT NULL,
    as_of_ts_ns INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    bars_processed INTEGER NOT NULL,
    orders_generated INTEGER NOT NULL,
    fills_generated INTEGER NOT NULL,
    trades_closed INTEGER NOT NULL,
    gross_pnl_usd REAL NOT NULL,
    costs_usd REAL NOT NULL,
    net_pnl_usd REAL NOT NULL,
    unrealized_pnl_usd REAL NOT NULL,
    equity_usd REAL NOT NULL,
    peak_equity_usd REAL NOT NULL,
    drawdown_usd REAL NOT NULL,
    drawdown_pct REAL NOT NULL,
    day_realized_pnl_usd REAL NOT NULL,
    risk_rejects INTEGER NOT NULL,
    risk_resizes INTEGER NOT NULL,
    kill_switch_active INTEGER NOT NULL,
    margin_complete INTEGER NOT NULL,
    open_positions INTEGER NOT NULL,
    elapsed_trading_days INTEGER,
    daily_sharpe REAL,
    annualized_sharpe REAL,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    drift_json TEXT NOT NULL DEFAULT '{}',
    raw_result_json TEXT NOT NULL,
    PRIMARY KEY (run_id, step_ordinal),
    FOREIGN KEY (run_id) REFERENCES paper_runs(run_id)
);

CREATE TABLE IF NOT EXISTS paper_fills (
    run_id TEXT NOT NULL,
    fill_seq INTEGER NOT NULL,
    step_ordinal INTEGER NOT NULL,
    fill_id INTEGER NOT NULL,
    order_id INTEGER NOT NULL,
    ts_fill_ns INTEGER NOT NULL,
    instrument_id INTEGER NOT NULL,
    raw_symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    fill_price REAL NOT NULL,
    commission_usd REAL NOT NULL,
    slippage_ticks REAL NOT NULL,
    PRIMARY KEY (run_id, fill_seq),
    FOREIGN KEY (run_id) REFERENCES paper_runs(run_id)
);

CREATE TABLE IF NOT EXISTS paper_positions (
    run_id TEXT NOT NULL,
    step_ordinal INTEGER NOT NULL,
    instrument_id INTEGER NOT NULL,
    raw_symbol TEXT NOT NULL,
    root_symbol TEXT NOT NULL,
    units INTEGER NOT NULL,
    avg_entry_price REAL NOT NULL,
    multiplier REAL NOT NULL,
    mark_price REAL NOT NULL,
    mark_ts_ns INTEGER NOT NULL,
    mark_age_ns INTEGER NOT NULL,
    mark_present INTEGER NOT NULL,
    mark_is_stale INTEGER NOT NULL,
    valuation_is_estimated INTEGER NOT NULL,
    gross_notional_usd REAL NOT NULL,
    signed_notional_usd REAL NOT NULL,
    unrealized_pnl_usd REAL NOT NULL,
    initial_margin_usd REAL NOT NULL,
    maintenance_margin_usd REAL NOT NULL,
    margin_known INTEGER NOT NULL,
    PRIMARY KEY (run_id, step_ordinal, instrument_id),
    FOREIGN KEY (run_id) REFERENCES paper_runs(run_id)
);

CREATE TABLE IF NOT EXISTS paper_alerts (
    run_id TEXT NOT NULL,
    alert_seq INTEGER NOT NULL,
    step_ordinal INTEGER NOT NULL,
    ts_ns INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    PRIMARY KEY (run_id, alert_seq),
    FOREIGN KEY (run_id) REFERENCES paper_runs(run_id)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PaperRunRow(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    run_id: str
    schema_version: str
    experiment_id: str
    experiment_identity: str
    strategy_family: str
    strategy_id: str
    strategy_fingerprint: str
    root_symbol: str
    params: dict
    risk_policy: dict
    risk_policy_identity: str
    schedule_policy: str
    commission_per_contract_usd: float
    slippage_ticks: float
    spread_ticks: float
    window_start_ns: int
    window_end_ns: int
    watermark_ns: int
    status: str
    data_source: str
    created_at: str
    stopped_at: str | None = None
    validation_result: dict = Field(default_factory=dict)


class PaperStepRow(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    run_id: str
    step_ordinal: int
    as_of_ts_ns: int
    created_at: str
    bars_processed: int
    orders_generated: int
    fills_generated: int
    trades_closed: int
    gross_pnl_usd: float
    costs_usd: float
    net_pnl_usd: float
    unrealized_pnl_usd: float
    equity_usd: float
    peak_equity_usd: float
    drawdown_usd: float
    drawdown_pct: float
    day_realized_pnl_usd: float
    risk_rejects: int
    risk_resizes: int
    kill_switch_active: bool
    margin_complete: bool
    open_positions: int
    #: None means "not recorded" (a Phase 21 legacy row migrated by Phase
    #: 21.1b) -- a real step always sets a concrete int (>= 1 in practice).
    elapsed_trading_days: int | None = None
    daily_sharpe: float | None
    annualized_sharpe: float | None
    #: alpha_agent.paper.provenance.PaperStepProvenance.model_dump(mode="json")
    provenance: dict = Field(default_factory=dict)
    #: alpha_agent.paper.drift.DriftReport.model_dump(mode="json") -- the full
    #: fill-rate / realized-slippage / trade-frequency / PnL-distribution /
    #: backtest-comparison-availability evidence for this step, persisted so
    #: it is retrievable later (report / UI), not only at step() call time.
    drift: dict = Field(default_factory=dict)
    raw_result: dict


class PaperFillRow(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    run_id: str
    fill_seq: int
    step_ordinal: int
    fill_id: int
    order_id: int
    ts_fill_ns: int
    instrument_id: int
    raw_symbol: str
    side: str
    quantity: int
    fill_price: float
    commission_usd: float
    slippage_ticks: float


class PaperPositionRow(BaseModel):
    """One authoritative C++ ``PositionExposure`` (portfolio.hpp), verbatim,
    for one instrument at one step. Never reconstructed in Python -- every
    field here is copied straight from the CLI's ``"positions"`` JSON array,
    itself a direct serialization of ``BacktestResult::portfolio_at_end
    .positions`` (Phase 21.1, cpp/apps/paper_trading_targets_csv.cpp)."""

    model_config = {"frozen": True, "extra": "forbid"}

    run_id: str
    step_ordinal: int
    instrument_id: int
    raw_symbol: str
    root_symbol: str
    units: int
    avg_entry_price: float
    multiplier: float
    mark_price: float
    mark_ts_ns: int
    mark_age_ns: int
    mark_present: bool
    mark_is_stale: bool
    valuation_is_estimated: bool
    gross_notional_usd: float
    signed_notional_usd: float
    unrealized_pnl_usd: float
    initial_margin_usd: float
    maintenance_margin_usd: float
    margin_known: bool


class PaperAlertRow(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    run_id: str
    alert_seq: int
    step_ordinal: int
    ts_ns: int
    created_at: str
    alert_type: str
    severity: str
    message: str
    detail: dict = Field(default_factory=dict)


class PaperLedgerError(RuntimeError):
    pass


class UnknownPaperRun(PaperLedgerError):
    pass


def _migrate(conn: sqlite3.Connection) -> None:
    """Deterministic, transactional, idempotent migration to
    ``CURRENT_SCHEMA_VERSION``. Called once per :class:`PaperLedger` open,
    inside the SAME transaction ``__init__`` already holds. A database
    already at ``CURRENT_SCHEMA_VERSION`` (including a brand-new one, whose
    ``_SCHEMA`` already created every table/column at the latest shape) is
    read once (one ``PRAGMA user_version`` + one ``PRAGMA table_info``) and
    left untouched -- re-running this on an already-migrated database issues
    no ``ALTER`` / ``UPDATE`` statement at all.

    Only ``paper_steps`` needs an ``ALTER TABLE`` here: ``paper_positions``
    is a wholly NEW table, so ``_SCHEMA``'s own ``CREATE TABLE IF NOT
    EXISTS`` (already executed by ``__init__`` before this call) creates it
    correctly on a pre-21.1 database with no extra code. ``paper_runs`` /
    ``paper_fills`` / ``paper_alerts`` are unchanged since Phase 21 and are
    never touched.
    """
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version >= CURRENT_SCHEMA_VERSION:
        return

    cols = {row[1] for row in conn.execute("PRAGMA table_info(paper_steps)")}
    added_new_columns = False
    if "elapsed_trading_days" not in cols:
        conn.execute("ALTER TABLE paper_steps ADD COLUMN elapsed_trading_days INTEGER")
        added_new_columns = True
    if "provenance_json" not in cols:
        conn.execute("ALTER TABLE paper_steps ADD COLUMN provenance_json TEXT NOT NULL DEFAULT '{}'")
        added_new_columns = True
    if "drift_json" not in cols:
        conn.execute("ALTER TABLE paper_steps ADD COLUMN drift_json TEXT NOT NULL DEFAULT '{}'")
        added_new_columns = True

    if added_new_columns:
        # Every paper_steps row present RIGHT NOW predates this migration --
        # _migrate runs synchronously inside PaperLedger.__init__, before the
        # caller can possibly have written a new step through this
        # connection, so there is no race with a concurrent commit_step.
        # Mark them explicitly LEGACY / NOT_AVAILABLE (never invent a value).
        conn.execute(
            "UPDATE paper_steps SET provenance_json = ?, drift_json = ?, elapsed_trading_days = NULL",
            (json.dumps(LEGACY_PROVENANCE, sort_keys=True), json.dumps(LEGACY_DRIFT, sort_keys=True)),
        )

    conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")


class PaperLedger:
    """Thin, typed wrapper over the append-only paper-trading SQLite store."""

    def __init__(self, path: str | Path = DEFAULT_LEDGER_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._conn:
            self._conn.executescript(_SCHEMA)
            _migrate(self._conn)

    @property
    def schema_version(self) -> int:
        """The ledger file's current ``PRAGMA user_version``. Always
        ``CURRENT_SCHEMA_VERSION`` once ``__init__`` has run."""
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- runs -------------------------------------------------------------
    def create_run(
        self,
        *,
        experiment_id: str,
        experiment_identity: str,
        strategy_family: str,
        strategy_id: str,
        strategy_fingerprint: str,
        root_symbol: str,
        params: dict,
        risk_policy: dict,
        risk_policy_identity: str,
        schedule_policy: str,
        commission_per_contract_usd: float,
        slippage_ticks: float,
        spread_ticks: float,
        window_start_ns: int,
        window_end_ns: int,
        data_source: str,
        validation_result: dict,
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or f"paper-{uuid.uuid4().hex[:16]}"
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO paper_runs (
                    run_id, schema_version, experiment_id, experiment_identity,
                    strategy_family, strategy_id, strategy_fingerprint, root_symbol,
                    params_json, risk_policy_json, risk_policy_identity, schedule_policy,
                    commission_per_contract_usd, slippage_ticks, spread_ticks,
                    window_start_ns, window_end_ns, watermark_ns, status, data_source,
                    created_at, stopped_at, validation_result_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id, "paper-run/1", experiment_id, experiment_identity,
                    strategy_family, strategy_id, strategy_fingerprint, root_symbol,
                    json.dumps(params, sort_keys=True), json.dumps(risk_policy, sort_keys=True),
                    risk_policy_identity, schedule_policy,
                    commission_per_contract_usd, slippage_ticks, spread_ticks,
                    window_start_ns, window_end_ns, window_start_ns, "ACTIVE", data_source,
                    _now(), None, json.dumps(validation_result, sort_keys=True),
                ),
            )
        return run_id

    def get_run(self, run_id: str) -> PaperRunRow:
        row = self._conn.execute(
            "SELECT * FROM paper_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise UnknownPaperRun(f"no paper run {run_id!r}")
        return _row_to_run(row)

    def list_runs(self) -> tuple[PaperRunRow, ...]:
        rows = self._conn.execute(
            "SELECT * FROM paper_runs ORDER BY created_at ASC"
        ).fetchall()
        return tuple(_row_to_run(r) for r in rows)

    def _update_run_state(
        self, run_id: str, *, watermark_ns: int, status: str, stopped_at: str | None = None
    ) -> None:
        cur = self._conn.execute(
            "UPDATE paper_runs SET watermark_ns = ?, status = ?, stopped_at = ? "
            "WHERE run_id = ?",
            (watermark_ns, status, stopped_at, run_id),
        )
        if cur.rowcount == 0:
            raise UnknownPaperRun(f"no paper run {run_id!r}")

    def update_run_state(
        self, run_id: str, *, watermark_ns: int, status: str, stopped_at: str | None = None
    ) -> None:
        with self._conn:
            self._update_run_state(run_id, watermark_ns=watermark_ns, status=status, stopped_at=stopped_at)

    # -- steps --------------------------------------------------------------
    def next_step_ordinal(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(step_ordinal) FROM paper_steps WHERE run_id = ?", (run_id,)
        ).fetchone()
        return int(row[0]) + 1 if row[0] is not None else 0

    def _insert_step(self, step: PaperStepRow) -> None:
        self._conn.execute(
            """
            INSERT INTO paper_steps (
                run_id, step_ordinal, as_of_ts_ns, created_at, bars_processed,
                orders_generated, fills_generated, trades_closed, gross_pnl_usd,
                costs_usd, net_pnl_usd, unrealized_pnl_usd, equity_usd,
                peak_equity_usd, drawdown_usd, drawdown_pct, day_realized_pnl_usd,
                risk_rejects, risk_resizes, kill_switch_active, margin_complete,
                open_positions, elapsed_trading_days, daily_sharpe, annualized_sharpe,
                provenance_json, drift_json, raw_result_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                step.run_id, step.step_ordinal, step.as_of_ts_ns, step.created_at,
                step.bars_processed, step.orders_generated, step.fills_generated,
                step.trades_closed, step.gross_pnl_usd, step.costs_usd, step.net_pnl_usd,
                step.unrealized_pnl_usd, step.equity_usd, step.peak_equity_usd,
                step.drawdown_usd, step.drawdown_pct, step.day_realized_pnl_usd,
                step.risk_rejects, step.risk_resizes, int(step.kill_switch_active),
                int(step.margin_complete), step.open_positions, step.elapsed_trading_days,
                step.daily_sharpe, step.annualized_sharpe,
                json.dumps(step.provenance, sort_keys=True),
                json.dumps(step.drift, sort_keys=True),
                json.dumps(step.raw_result, sort_keys=True),
            ),
        )

    def append_step(self, step: PaperStepRow) -> None:
        with self._conn:
            self._insert_step(step)

    def list_steps(self, run_id: str) -> tuple[PaperStepRow, ...]:
        rows = self._conn.execute(
            "SELECT * FROM paper_steps WHERE run_id = ? ORDER BY step_ordinal ASC", (run_id,)
        ).fetchall()
        return tuple(_row_to_step(r) for r in rows)

    def latest_step(self, run_id: str) -> PaperStepRow | None:
        row = self._conn.execute(
            "SELECT * FROM paper_steps WHERE run_id = ? ORDER BY step_ordinal DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return _row_to_step(row) if row is not None else None

    # -- fills ----------------------------------------------------------
    def cumulative_fill_count(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM paper_fills WHERE run_id = ?", (run_id,)
        ).fetchone()
        return int(row[0])

    def _insert_fills(self, fills: Sequence[PaperFillRow]) -> None:
        if not fills:
            return
        self._conn.executemany(
            """
            INSERT INTO paper_fills (
                run_id, fill_seq, step_ordinal, fill_id, order_id, ts_fill_ns,
                instrument_id, raw_symbol, side, quantity, fill_price,
                commission_usd, slippage_ticks
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    f.run_id, f.fill_seq, f.step_ordinal, f.fill_id, f.order_id,
                    f.ts_fill_ns, f.instrument_id, f.raw_symbol, f.side, f.quantity,
                    f.fill_price, f.commission_usd, f.slippage_ticks,
                )
                for f in fills
            ],
        )

    def append_fills(self, fills: Sequence[PaperFillRow]) -> None:
        if not fills:
            return
        with self._conn:
            self._insert_fills(fills)

    def list_fills(self, run_id: str) -> tuple[PaperFillRow, ...]:
        rows = self._conn.execute(
            "SELECT * FROM paper_fills WHERE run_id = ? ORDER BY fill_seq ASC", (run_id,)
        ).fetchall()
        return tuple(
            PaperFillRow(
                run_id=r["run_id"], fill_seq=r["fill_seq"], step_ordinal=r["step_ordinal"],
                fill_id=r["fill_id"], order_id=r["order_id"], ts_fill_ns=r["ts_fill_ns"],
                instrument_id=r["instrument_id"], raw_symbol=r["raw_symbol"], side=r["side"],
                quantity=r["quantity"], fill_price=r["fill_price"],
                commission_usd=r["commission_usd"], slippage_ticks=r["slippage_ticks"],
            )
            for r in rows
        )

    # -- positions (Phase 21.1) ------------------------------------------
    def _insert_positions(self, positions: Sequence[PaperPositionRow]) -> None:
        if not positions:
            return
        self._conn.executemany(
            """
            INSERT INTO paper_positions (
                run_id, step_ordinal, instrument_id, raw_symbol, root_symbol, units,
                avg_entry_price, multiplier, mark_price, mark_ts_ns, mark_age_ns,
                mark_present, mark_is_stale, valuation_is_estimated, gross_notional_usd,
                signed_notional_usd, unrealized_pnl_usd, initial_margin_usd,
                maintenance_margin_usd, margin_known
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    p.run_id, p.step_ordinal, p.instrument_id, p.raw_symbol, p.root_symbol,
                    p.units, p.avg_entry_price, p.multiplier, p.mark_price, p.mark_ts_ns,
                    p.mark_age_ns, int(p.mark_present), int(p.mark_is_stale),
                    int(p.valuation_is_estimated), p.gross_notional_usd, p.signed_notional_usd,
                    p.unrealized_pnl_usd, p.initial_margin_usd, p.maintenance_margin_usd,
                    int(p.margin_known),
                )
                for p in positions
            ],
        )

    def append_positions(self, positions: Sequence[PaperPositionRow]) -> None:
        if not positions:
            return
        with self._conn:
            self._insert_positions(positions)

    def list_positions(self, run_id: str, step_ordinal: int | None = None) -> tuple[PaperPositionRow, ...]:
        """All position rows for ``run_id``, optionally filtered to one
        ``step_ordinal``. With no filter, rows for every step are returned
        (full audit history); pass the run's latest step_ordinal for
        "current holdings"."""
        if step_ordinal is None:
            rows = self._conn.execute(
                "SELECT * FROM paper_positions WHERE run_id = ? "
                "ORDER BY step_ordinal ASC, instrument_id ASC",
                (run_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM paper_positions WHERE run_id = ? AND step_ordinal = ? "
                "ORDER BY instrument_id ASC",
                (run_id, step_ordinal),
            ).fetchall()
        return tuple(_row_to_position(r) for r in rows)

    # -- alerts -----------------------------------------------------------
    def next_alert_seq(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(alert_seq) FROM paper_alerts WHERE run_id = ?", (run_id,)
        ).fetchone()
        return int(row[0]) + 1 if row[0] is not None else 0

    def _insert_alerts(self, alerts: Sequence[PaperAlertRow]) -> None:
        if not alerts:
            return
        self._conn.executemany(
            """
            INSERT INTO paper_alerts (
                run_id, alert_seq, step_ordinal, ts_ns, created_at, alert_type,
                severity, message, detail_json
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    a.run_id, a.alert_seq, a.step_ordinal, a.ts_ns, a.created_at,
                    a.alert_type, a.severity, a.message, json.dumps(a.detail, sort_keys=True),
                )
                for a in alerts
            ],
        )

    def append_alerts(self, alerts: Sequence[PaperAlertRow]) -> None:
        if not alerts:
            return
        with self._conn:
            self._insert_alerts(alerts)

    def list_alerts(self, run_id: str) -> tuple[PaperAlertRow, ...]:
        rows = self._conn.execute(
            "SELECT * FROM paper_alerts WHERE run_id = ? ORDER BY alert_seq ASC", (run_id,)
        ).fetchall()
        return tuple(
            PaperAlertRow(
                run_id=r["run_id"], alert_seq=r["alert_seq"], step_ordinal=r["step_ordinal"],
                ts_ns=r["ts_ns"], created_at=r["created_at"], alert_type=r["alert_type"],
                severity=r["severity"], message=r["message"],
                detail=json.loads(r["detail_json"]),
            )
            for r in rows
        )

    # -- atomic step commit (Phase 21.1) ---------------------------------
    def commit_step(
        self,
        *,
        run_id: str,
        step: PaperStepRow,
        fills: Sequence[PaperFillRow],
        positions: Sequence[PaperPositionRow],
        alerts: Sequence[PaperAlertRow],
        watermark_ns: int,
        status: str,
        stopped_at: str | None = None,
    ) -> None:
        """Append one step's snapshot, its new fills, its authoritative
        position snapshot, and its alerts, and advance the run's
        watermark/status/stopped_at -- ALL inside ONE SQLite transaction.

        A raised exception at any point rolls back every write issued in this
        call (sqlite3's own transaction semantics on ``self._conn``): a step
        is either fully recorded or the ledger is left exactly as it was
        before this call, never partially committed.
        """
        with self._conn:
            self._insert_step(step)
            self._insert_fills(fills)
            self._insert_positions(positions)
            self._insert_alerts(alerts)
            self._update_run_state(run_id, watermark_ns=watermark_ns, status=status, stopped_at=stopped_at)


def _row_to_run(r: sqlite3.Row) -> PaperRunRow:
    return PaperRunRow(
        run_id=r["run_id"], schema_version=r["schema_version"], experiment_id=r["experiment_id"],
        experiment_identity=r["experiment_identity"], strategy_family=r["strategy_family"],
        strategy_id=r["strategy_id"], strategy_fingerprint=r["strategy_fingerprint"],
        root_symbol=r["root_symbol"], params=json.loads(r["params_json"]),
        risk_policy=json.loads(r["risk_policy_json"]), risk_policy_identity=r["risk_policy_identity"],
        schedule_policy=r["schedule_policy"],
        commission_per_contract_usd=r["commission_per_contract_usd"],
        slippage_ticks=r["slippage_ticks"], spread_ticks=r["spread_ticks"],
        window_start_ns=r["window_start_ns"], window_end_ns=r["window_end_ns"],
        watermark_ns=r["watermark_ns"], status=r["status"], data_source=r["data_source"],
        created_at=r["created_at"], stopped_at=r["stopped_at"],
        validation_result=json.loads(r["validation_result_json"]),
    )


def _row_to_step(r: sqlite3.Row) -> PaperStepRow:
    # PaperLedger.__init__ always runs _migrate before any row can be read,
    # so every column below is guaranteed present -- no defensive "in keys"
    # fallback needed (and none that could silently mask a migration bug).
    return PaperStepRow(
        run_id=r["run_id"], step_ordinal=r["step_ordinal"], as_of_ts_ns=r["as_of_ts_ns"],
        created_at=r["created_at"], bars_processed=r["bars_processed"],
        orders_generated=r["orders_generated"], fills_generated=r["fills_generated"],
        trades_closed=r["trades_closed"], gross_pnl_usd=r["gross_pnl_usd"],
        costs_usd=r["costs_usd"], net_pnl_usd=r["net_pnl_usd"],
        unrealized_pnl_usd=r["unrealized_pnl_usd"], equity_usd=r["equity_usd"],
        peak_equity_usd=r["peak_equity_usd"], drawdown_usd=r["drawdown_usd"],
        drawdown_pct=r["drawdown_pct"], day_realized_pnl_usd=r["day_realized_pnl_usd"],
        risk_rejects=r["risk_rejects"], risk_resizes=r["risk_resizes"],
        kill_switch_active=bool(r["kill_switch_active"]), margin_complete=bool(r["margin_complete"]),
        open_positions=r["open_positions"], elapsed_trading_days=r["elapsed_trading_days"],
        daily_sharpe=r["daily_sharpe"], annualized_sharpe=r["annualized_sharpe"],
        provenance=json.loads(r["provenance_json"]), drift=json.loads(r["drift_json"]),
        raw_result=json.loads(r["raw_result_json"]),
    )


def _row_to_position(r: sqlite3.Row) -> PaperPositionRow:
    return PaperPositionRow(
        run_id=r["run_id"], step_ordinal=r["step_ordinal"], instrument_id=r["instrument_id"],
        raw_symbol=r["raw_symbol"], root_symbol=r["root_symbol"], units=r["units"],
        avg_entry_price=r["avg_entry_price"], multiplier=r["multiplier"], mark_price=r["mark_price"],
        mark_ts_ns=r["mark_ts_ns"], mark_age_ns=r["mark_age_ns"],
        mark_present=bool(r["mark_present"]), mark_is_stale=bool(r["mark_is_stale"]),
        valuation_is_estimated=bool(r["valuation_is_estimated"]),
        gross_notional_usd=r["gross_notional_usd"], signed_notional_usd=r["signed_notional_usd"],
        unrealized_pnl_usd=r["unrealized_pnl_usd"], initial_margin_usd=r["initial_margin_usd"],
        maintenance_margin_usd=r["maintenance_margin_usd"], margin_known=bool(r["margin_known"]),
    )
