# Paper Trading and Drift Monitor (Phase 21 / 21.1 / 21.1b)

Phase 21.1 hardens the Phase 21 flow (commit `1087724`) on four points a
review found incomplete: the declared drift-monitor contract, exact per-step
execution provenance, replay-prefix consistency + atomic step commits, and a
persisted authoritative C++ position snapshot. See "Phase 21.1 hardening"
below; everything under "Phase 21" describes the unchanged base architecture.

Phase 21.1b fixes two blockers found reviewing 21.1 itself: (1) 21.1's new
`paper_steps` columns / `paper_positions` table were only ever created via
`CREATE TABLE IF NOT EXISTS`, which cannot add a column to a table that
already exists -- a real ledger created by 1087724 would silently keep its
OLD shape forever and every 21.1 write would fail against it; (2) the
replay-prefix check's float comparison used a `1e-9` tolerance, which is not
"exact" and could in principle mask a genuine tiny divergence. See "Phase
21.1b hardening" below.

`prompts/21_PAPER_TRADING_AND_DRIFT_MONITOR.md`:

> after a strategy passes frozen validation, route it only to a simulated
> paper portfolio. Implement target-position -> order -> simulated fill ->
> portfolio flow, a persistent paper ledger, live/paper vs backtest drift
> metrics, fill-rate/slippage/trade-frequency/PnL distribution monitoring, a
> kill switch using deterministic risk rules, and alert/report interfaces. No
> live-money broker routing in this phase.

## What "paper trading" means in this system today

There is no live market-data feed and no broker connection anywhere in this
codebase (CLAUDE.md: MVP is historical research + paper trading only; no
live-money routing without an explicit future phase). Phase 21 does not add
one. "Paper trading" here is **deterministic replay-based paper trading**: it
repeatedly runs the exact same frozen research pipeline (feature engine ->
`StrategyCompiler` -> target schedule -> C++ engine) forward through
already-acquired historical CME data
(`alpha_agent.data.real_market_dataset`, `[2018-01-01, 2025-01-01)` --
never the locked 2025 holdout, never a new paid Databento request), advancing
a persistent watermark one or more trading days at a time, and persisting the
result to a ledger. This is the plumbing a future live-data phase plugs a real
feed into (`MarketWindowProvider`) without changing anything downstream of it
-- and it is explicitly labelled as replay everywhere it is surfaced (CLI,
docs, UI), never presented as a live account.

## Architecture boundary (unchanged, extended additively)

* **C++ Quant Core remains authoritative** for fills, execution, accounting,
  PnL, positions, and risk. Paper trading adds one new, additive C++ path --
  it does not modify the frozen historical research reference path.
* **Python orchestrates, monitors, persists, and presents** -- it never
  recomputes a fill, a PnL number, or a risk decision. Every number in the
  paper ledger is copied verbatim from the C++ CLI's JSON result.
* **The Streamlit UI's read path is read-only presentation.** It can only
  call a read method on the paper ledger or the deterministic eligibility
  service (mirrors how the registry has no write path through the UI
  either). Product UI Polish pass, section 11: the ONE exception is
  "Start Paper Test" -- a button that calls the single, explicit
  `alpha_agent.ui.paper_actions.start_paper_run` boundary, which invokes the
  exact same `PaperTradingEngine.start_run` `scripts/phase_21_paper_trading
  .py start` already uses (eligibility is re-verified inside `start_run`
  itself, never decided by the UI). Stepping and stopping an existing run
  are still CLI-only, via `scripts/phase_21_paper_trading.py`.
* Phase 13 statistical semantics, Phase 14 registry identity /
  failure-memory semantics, and the 2025 locked holdout are all untouched.

## The hard-risk C++ path (additive, mirrors Phase 19's pattern)

The historical research reference path
(`apps/backtest_targets_csv.cpp` / `quant_core/targets_run.hpp`) is wired to
`PassThroughRiskManager` and stays that way -- it is the frozen reference
path. Paper trading needed a REAL kill switch, so Phase 21 adds a **new,
additive** wiring, exactly the way Phase 13.5C added auxiliary roll-close
marks and Phase 19 added the pybind fast boundary: a new header + a new CLI
that share the same `BacktestEngine`, `ScheduledTargetStrategy`, and
`Fill`/`PositionLedger`/`PortfolioAccountant` machinery, but gate every Order
through the Phase 08 `quant::PortfolioRiskManager` instead:

* `cpp/include/quant_core/paper_trading_run.hpp` -- `run_paper_trading_backtest`,
  the in-process helper (mirrors `targets_run.hpp`'s shape).
* `cpp/apps/paper_trading_targets_csv.cpp` -- the new reference CLI
  (`quant_paper_trading_targets_csv`), positional-compatible with
  `quant_backtest_targets_csv` plus a mandatory `risk_config.csv` and an
  optional `--margin-config=` / `--end-of-test=`.
* `cpp/tests/test_paper_trading_run.cpp` -- pins: equivalence to a hand-wired
  engine when nothing is breached; the SAME losing schedule/bars that runs to
  completion under the pass-through reference path trips the drawdown kill
  switch here (opening orders rejected, the existing position is never
  force-closed); `EndOfTestPolicy` defaults to `LeaveOpen` (a paper position
  survives into the next step's replay; `ForceLiquidateFinalClose` is the
  explicit "flatten and stop" case); the contract-resolution hard check;
  config guards; determinism.

Phase 21.1 additive change: the CLI's JSON output gained a `"positions"`
array, a verbatim serialization of `BacktestResult::portfolio_at_end
.positions` (instrument/raw/root symbol, signed units, avg entry, mark price
/ age / staleness, gross/signed notional, unrealized PnL, margin fields) --
no new execution/accounting/risk semantics, purely an additive read of a
value the C++ `PortfolioAccountant` already computed.

`EndOfTestPolicy::LeaveOpen` is the key design point: one paper-trading
**step** replays the WHOLE window `[window_start, new_watermark]` from
scratch through the engine (never an incremental / mutated Python-side
state), so forcing a liquidation at the end of every step would fabricate a
round-trip trade that never happened. Positions only get force-flattened on
the one deliberate `stop_run` step.

### Margin

CLAUDE.md forbids guessing a margin figure from notional, and this system has
no committed, dated, sourced per-root CME margin schedule yet
(`quant::MarginModel` / `--margin-config=`). Rather than fabricate one,
`PaperRiskPolicy` defaults to `missing_margin_policy = "treat_as_zero"` with
`max_margin_utilization_pct = 0` (disabled) -- margin utilization is simply
not enforced, and the gap is surfaced verbatim as
`portfolio_at_end.margin_complete == false`, never silently treated as
satisfied. Every OTHER hard-risk check is fully active: position caps
(per-symbol / per-root / gross), gross exposure / leverage caps, the
$1,500 daily-loss kill switch, the 10% drawdown kill switch, and stale-mark
protection (all `quant::RiskConfig`'s own Phase 08 defaults, unchanged).

## Python package (`alpha_agent.paper`)

| Module | Responsibility |
|---|---|
| `eligibility.py` | The ONLY gate between the registry and paper trading. Requires an AUTHORITATIVE experiment with a VALID attempt, headline verdict `PASS` with `all_required_gates_satisfied` present (the same positive-proof rule `alpha_agent.ui.services.gate_state` uses), a strategy family this MVP can deterministically rebuild (`candidates_phase_13_5c.spec_for_params` -- the four Phase 13.5C daily baselines: `tsmom`, `ma_trend`, `breakout`, `mean_reversion`), and a rebuilt `StrategySpec` whose fingerprint matches the committed one. Refuses loudly otherwise; approximates nothing. |
| `risk_policy.py` | `PaperRiskPolicy` -- the typed, versioned, human-set hard-risk configuration (mirrors `quant::RiskConfig` field-for-field) + its CSV serialisation for the C++ CLI. Immutable for the life of a run. |
| `schedule_window.py` | Rebuilds bars/contracts/target-schedule for one replay window via the SAME Phase 09-11 feature/DSL/evaluator machinery the research path uses, behind a `MarketWindowProvider` protocol so a real data source (`RealMarketWindowProvider`, wrapping `alpha_agent.data.real_market_dataset`) and a synthetic test fixture share one code path. |
| `bridge.py` | Subprocess bridge to `quant_paper_trading_targets_csv` (mirrors `alpha_agent.adapters.targets_bridge`). |
| `ledger.py` | The persistent, append-oriented SQLite paper ledger (`data/paper_trading/paper_ledger.sqlite`) -- `paper_runs` / `paper_steps` / `paper_fills` / `paper_positions` (Phase 21.1) / `paper_alerts`. A SEPARATE store from the Phase 14 experiment registry: paper trading is operational monitoring of an already-approved strategy, not a new statistical hypothesis, and must never be confused with the registry's BH-FDR / failure-memory semantics. `commit_step` (Phase 21.1) writes a step's snapshot, fills, positions, and alerts, plus the run's watermark/status, in ONE atomic transaction. `CURRENT_SCHEMA_VERSION` / `_migrate()` (Phase 21.1b) version the whole database via `PRAGMA user_version` and migrate a pre-21.1 database in place, marking any preexisting step's provenance/drift with an explicit `LEGACY_*` marker (`is_legacy_evidence`) rather than a fabricated value. |
| `engine.py` | `PaperTradingEngine` -- `start_run` (reads the registry once), `step` (advances the watermark by N trading days and replays), `stop_run` (one final `force_liquidate` replay). Builds exact step provenance, proves replay-prefix consistency, then persists the step/fills/positions/alerts/watermark in ONE atomic ledger transaction (Phase 21.1). Never recomputes a fill, a position, or a PnL number. |
| `provenance.py` | `PaperStepProvenance` (Phase 21.1) -- SHA-256 of the literal bars/contracts/targets/validation-days bytes sent to the C++ boundary, the literal C++ executable, and the literal result payload, plus the parent experiment identity / strategy fingerprint / risk-policy identity / cost config a step ran under. `.identity()` is a single fingerprint that changes iff any canonical input changes. |
| `replay_integrity.py` | `assert_replay_prefix_consistent` (Phase 21.1) -- proves a step's full-window replay begins with an EXACT, field-for-field copy of every fill the ledger already committed for that run; raises `PaperReplayDivergence` otherwise. Run BEFORE any ledger write. |
| `drift.py` | `DriftReport` -- reuses the SAME frozen `alpha_agent.validation.runner.backtest_run_from_cpp_json` helper the Phase 13 validation framework uses to turn a step's official `daily_equity` trace into daily Sharpe / PnL / trade-count stats, then compares them descriptively to the experiment's own committed `ResultRecord`. Phase 21.1 completes the declared monitoring contract: `RealizedSlippageEvidence` (from the official per-fill audit trail x the contract's own tick size/multiplier), `TradeFrequencyEvidence` (trades normalized by `ValidationDayPlan.n_days`), `PnlDistributionEvidence` (descriptive stats over the official daily PnL series), and `comparison_availability` (AVAILABLE only where a real, experiment-bound backtest artifact exists -- `daily_sharpe`/`net_pnl_usd`/`n_trades`; ALWAYS `NOT_AVAILABLE` for `fill_rate`/`slippage`/`trade_frequency`/`pnl_distribution`, since no committed `ResultRecord` carries those series for any experiment today). Diagnostic only -- defines no new statistical test, threshold, or BH/FDR family; never a pass/fail authority. |
| `alerts.py` | Typed `Alert`s on STATE TRANSITIONS only (kill-switch tripped/cleared, incomplete margin data, exhausted replay window, a drift flag) -- never a spam-every-step re-alert. `kill_switch_active()` restates, for display, the same threshold comparison the C++ gate already enforced; it makes no risk decision itself. |
| `report.py` | `build_paper_run_report` -- a read-only JSON report of one run's full ledger state, including the latest step's authoritative position snapshot (`latest_positions`), provenance, and drift report. |

## Phase 21.1 hardening

Four integrity blockers found on review of Phase 21, fixed without touching
any Phase 21 architecture decision (the additive C++ hard-risk path stays the
paper risk authority; Python stays orchestration/persistence/diagnostics
only; the UI stays read-only; deterministic replay is unchanged; 2025 stays
locked; the real 0/167-eligible state is unchanged):

1. **Completed the declared drift-monitor contract.** `DriftReport` (schema
   `paper-drift-report/2`) now carries realized slippage (from the official
   per-fill audit trail, never the configured cost assumption), trade
   frequency (normalized by elapsed trading days, never a raw count), and PnL
   distribution (descriptive stats over the official daily PnL series), each
   with its own `comparison_availability` -- `NOT_AVAILABLE` rather than
   fabricated wherever no committed backtest artifact exists.
2. **Exact per-step execution provenance.** `PaperStepProvenance` (schema
   `paper-step-provenance/1`) is built BEFORE any ledger write from the
   literal SHA-256 of every canonical input a step's C++ invocation actually
   used (bars/contracts/targets/validation-days files, the C++ executable,
   the result payload) plus the parent experiment identity / strategy
   fingerprint / risk-policy identity / cost config. Persisted with the step
   (`paper_steps.provenance_json`), never a free-form label.
3. **Replay-prefix consistency + atomic step commit.** Before any write,
   `alpha_agent.paper.replay_integrity.assert_replay_prefix_consistent`
   proves the step's freshly-replayed fills begin with an EXACT copy of every
   fill the ledger already committed for that run; a shorter or disagreeing
   replay raises `PaperReplayDivergence` and nothing is written. The step
   snapshot, its new fills, its position snapshot, its alerts, and the run's
   watermark/status now all commit inside ONE SQLite transaction
   (`PaperLedger.commit_step`) -- a crash or exception midway rolls back
   everything, never leaving a step half-recorded.
4. **Persisted authoritative C++ position snapshot.** The additive CLI now
   serializes `BacktestResult::portfolio_at_end.positions` (Phase 08
   `PositionExposure`, already computed by the C++ `PortfolioAccountant`) as a
   `"positions"` JSON array; Python copies it verbatim into a new
   `paper_positions` ledger table (`PaperPositionRow`) -- one row per held
   instrument per step. The Paper Trading UI's Positions tab renders ONLY
   this committed snapshot; nothing here reconstructs a position from fills.

Verified against BOTH a synthetic fixture and the real, already-acquired NQ
2018-2024 archive (`scripts/phase_21_paper_trading.py start` /`step` /
`status` / `stop` against a real NQ `mean_reversion` experiment): a real
`NQH8` position entered, a real committed position snapshot, real provenance
hashes, and `stop` correctly flattening the position (`end_of_test =
force_liquidate`) -- with zero new market-data cost (the offline
`RealMarketWindowProvider` reads only the already-acquired 2018-2024 raw
store).

## Phase 21.1b hardening

1. **Versioned paper-ledger schema migration.** The ledger file now tracks an
   explicit whole-database shape version via `PRAGMA user_version`
   (`CURRENT_SCHEMA_VERSION = 2`, label `"paper-ledger/2"` --
   `alpha_agent.paper.ledger.CURRENT_SCHEMA_VERSION` /
   `LEDGER_SCHEMA_LABEL`). Phase 21 (1087724) never set this, so it reads
   back 0 on any database it created. `PaperLedger.__init__` runs
   `_migrate()` on every open, inside its own transaction: a database
   already at `CURRENT_SCHEMA_VERSION` (including a brand-new one, created
   directly at the latest shape) is read once and left untouched; a pre-21.1
   (`user_version == 0`) database gets `ALTER TABLE paper_steps ADD COLUMN`
   for `elapsed_trading_days` / `provenance_json` / `drift_json` (the three
   columns `CREATE TABLE IF NOT EXISTS` cannot retrofit onto an existing
   table), then every row already in `paper_steps` -- by construction, every
   row that predates this call -- is set to the explicit `LEGACY_PROVENANCE`
   / `LEGACY_DRIFT` markers (`status: "NOT_AVAILABLE"`, `schema_version`
   ending `/legacy`) with `elapsed_trading_days = NULL`, never a fabricated
   value. `paper_positions` (a wholly new table) is already created
   correctly by the earlier `CREATE TABLE IF NOT EXISTS` with no extra code.
   `paper_runs` / `paper_fills` / `paper_alerts` are untouched (unchanged
   since Phase 21). Migration is one transaction and checks its own
   precondition first (`version >= CURRENT_SCHEMA_VERSION: return`), so
   re-opening an already-migrated database is a no-op -- verified by
   re-opening the SAME migrated file three times in a row and by a
   byte-identical re-read of the legacy row.
   `alpha_agent.paper.ledger.is_legacy_evidence(payload)` lets a caller (the
   UI) tell "not recorded" apart from "recorded, and empty" for a migrated
   step's `provenance` / `drift`; the Paper Trading page's Positions / Drift
   Monitor / Provenance tabs now show an explicit "this step predates
   Phase 21.1" notice instead of silently rendering "no position" / "n/a"
   for a legacy step.
2. **Exact replay-prefix comparison.** `assert_replay_prefix_consistent`
   dropped the `math.isclose(rel_tol=1e-9, abs_tol=1e-9)` float tolerance
   entirely; every field, including `fill_price` / `commission_usd` /
   `slippage_ticks`, is now compared with plain `==`. This is safe, not just
   stricter: the C++ side always formats these floats with a fixed `%.12g`
   (`detail::trade_g`, trade_export.hpp) before Python ever sees them, so two
   replays of IDENTICAL inputs through the deterministic engine produce the
   same double, the same decimal text, and (`float()` parsing being
   deterministic) the same bit pattern; SQLite's `REAL` column stores and
   returns that double losslessly. There is no "harmless serialization
   noise" a tolerance could ever need to absorb here, only a real divergence
   a tolerance could hide.

Audited while making this patch: `paper_steps.drift_json` already was (since
21.1) a real, dedicated SQLite column persisting
`DriftReport.model_dump(mode="json")` with the step -- `ledger.py`,
`PaperStepRow.drift`, the docs, and `outputs/phase_21/
PHASE_21_PAPER_TRADING.json` all already agreed on this; no design change
was needed here, only the schema-version audit above. The one real
documentation gap found: `outputs/phase_21/PHASE_21_PAPER_TRADING.json`'s
`ledger.schema_version` field named `"paper-run/1"` -- that is
`paper_runs.schema_version`, a per-ROW marker for `PaperRunRow`'s shape
(unrelated to, and unchanged by, this patch), not the whole ledger file's
schema version. It is now split into `row_schema_version` ("paper-run/1")
and `ledger_schema_version` ("paper-ledger/2", the new `PRAGMA user_version`
value) so the two distinct concepts stop sharing one ambiguous key.

## CLI

```
python scripts/phase_21_paper_trading.py list-eligible
python scripts/phase_21_paper_trading.py start --experiment <experiment_id_or_identity> \
    [--capital 100000] [--max-drawdown-pct 0.10] [--max-daily-loss-usd 1500] [--commission 2.0]
python scripts/phase_21_paper_trading.py step --run <run_id> [--days N]
python scripts/phase_21_paper_trading.py stop --run <run_id>
python scripts/phase_21_paper_trading.py status --run <run_id> [--out outputs/phase_21/<run_id>.json]
python scripts/phase_21_paper_trading.py list-runs
```

Every command is strictly offline (`[2018-01-01, 2025-01-01)` only). `start`
refuses any experiment that has not passed frozen validation via
`alpha_agent.paper.eligibility`.

## Streamlit UI (Paper Trading page)

Read-only: lists eligible strategies (`alpha_agent.paper.eligibility`) and, if
any paper run exists in the ledger, its real committed state (status badge,
equity / drawdown / fills metrics, kill-switch / margin-gap banners, and
Steps / Positions / Fills / Alerts / Drift Monitor / Provenance tabs). No
button on this page starts, steps, or stops a run. The Positions tab renders
ONLY the committed C++ position snapshot; the Drift Monitor tab shows fill
rate, realized slippage, trade frequency, PnL distribution and each
dimension's backtest-comparison availability (never a pass/fail badge); the
Provenance tab shows the latest step's exact SHA-256 provenance.
`alpha_agent.ui.services`'s Phase 21 additions (`paper_eligible_experiments`,
`list_paper_runs`, `paper_run_report`) never call a `PaperLedger` write
method (`test_paper_services_module_never_calls_a_ledger_write_method`,
extended in Phase 21.1 to also cover `commit_step` / `append_positions`,
mirroring the Phase 20 registry-boundary guard).

## Real registry state today

The real committed registry has **0 authoritative PASS experiments**
(107 Phase 13.5C hypotheses, 0 PASS; Phase 15B ML meta-labeling, 0 PASS -- see
`docs/EXPERIMENT_REGISTRY.md` / phase-status notes). `list-eligible` and the
Paper Trading page therefore both honestly report 0 strategies eligible for
paper trading today. This is not a placeholder -- it is exactly what the
frozen evidence says, and the full mechanism above is real, tested (C++
integration tests against the compiled CLI, a Python end-to-end test that
starts a run, steps through an engineered price crash, and proves the kill
switch trips and halts the run, all through the REAL compiled
`quant_paper_trading_targets_csv`, PLUS a Phase 21.1 CLI smoke test run
against the REAL already-acquired NQ 2018-2024 archive with a temporary
PASS-seeded registry: start / step / status / stop all worked, produced a
real `NQH8` position, and `stop` correctly flattened it) and ready for the
day a strategy does pass.

## Scope / known limitations (documented, not hidden)

* Only the four Phase 13.5C daily-signal baseline families are eligible
  (`tsmom`, `ma_trend`, `breakout`, `mean_reversion`); `silver_bullet`
  (native 1m) and any Phase 17-compiled LLM blueprint need their own
  schedule-window builder / full-`StrategySpec` persistence, not implemented
  here.
* No committed per-root margin schedule; margin utilization is not enforced
  (see "Margin" above) -- visible via `margin_complete == false`.
* `stop_run` on a run with zero prior steps re-replays the (too-short)
  `[window_start, window_start]` window and raises
  `PaperDataWindowExhausted` rather than a no-op success; a run that never
  stepped has nothing to flatten anyway.
* No pybind fast boundary for the paper-trading CLI (Phase 19 added one only
  for the historical research path); paper-trading steps are infrequent
  (daily-cadence) so subprocess overhead is immaterial.
* No committed backtest artifact carries a fill-rate, a slippage
  distribution, an elapsed-day-normalized trade-frequency, or a daily-PnL
  distribution for any experiment today -- those four `comparison_availability`
  entries are always `NOT_AVAILABLE` until a future phase commits one
  (Phase 21.1; only `daily_sharpe` / `net_pnl_usd` / `n_trades` -- already on
  `ResultRecord` -- can ever read `AVAILABLE`).
* Roll continuation is not yet wired into paper trading
  (`PaperStepProvenance.roll_close_marks_sha256` is always `None` today).
* A step migrated from a pre-21.1 database has no `paper_positions` rows at
  all (that table did not exist yet) -- indistinguishable from "flat" except
  via `is_legacy_evidence(step.provenance)`, which the UI now checks
  explicitly before rendering the Positions tab.

## Tests

* `cpp/tests/test_paper_trading_run.cpp` (27 checks, `quant_paper_trading_run_tests`,
  part of the 13/13 ctest suite, unchanged by 21.1b) -- C++ wiring, kill
  switch, `EndOfTestPolicy`, determinism.
* `tests/python/test_phase_21_paper_trading.py` (40 tests, was 35) -- risk
  policy; eligibility (accept + 4 refusal paths); ledger round-trip; the
  engine end-to-end through the real compiled CLI (fills, kill switch,
  watermark accumulation, data-window exhaustion, stop_run); alerts; the
  completed drift contract (realized slippage from fill evidence,
  elapsed-day-normalized trade frequency, PnL-distribution stats,
  comparison-availability); EXACT replay-prefix consistency (exact-prefix
  pass, shorter-replay refusal, field-mismatch refusal, a deliberately TINY
  -- `1e-10` -- float perturbation the OLD `1e-9` tolerance would have
  silently accepted, now correctly raising `PaperReplayDivergence` both at
  the unit level and end-to-end through the real engine with proof that
  NOTHING is written, plus the original engine-level test that mutates an
  earlier replay input); atomic step commit (success + a failure-injection
  test proving a mid-transaction exception rolls back the step/fills/alerts
  and leaves the watermark untouched); the authoritative position snapshot
  round-tripping into the ledger and matching the step's own unrealized PnL;
  a direct proof that a contrived, internally-inconsistent position value
  survives verbatim (Python never recomputes it); step-provenance identity
  stability under identical inputs and change under a different risk policy;
  provenance + drift retrievable later via the report; a static guard that
  `_run_step` only ever persists through `commit_step`; **versioned ledger
  migration** (Phase 21.1b) -- migrating a byte-exact synthetic Phase-21
  (1087724) schema fixture preserves every existing run/step/fill/alert row
  unchanged, marks legacy provenance/drift honestly (never fabricated),
  allows a new Phase-21.1 step to commit atomically afterward on the SAME
  database, is idempotent across three repeated opens, a brand-new database
  is created directly at the latest schema, and -- when a real local
  pre-21.1 `data/paper_trading/paper_ledger.sqlite` exists on the machine --
  a non-destructive smoke test against a COPY of it (skipped otherwise).
* `tests/python/test_phase_21_streamlit_ui.py` (8 tests) -- services
  read-only boundary guard (extended to `commit_step` / `append_positions`),
  real-registry eligibility state, page rendering (empty state, a seeded real
  run, and the Positions / Drift Monitor / Provenance tabs with a seeded
  position + drift + provenance fixture); the empty-state render test now
  isolates its own ledger path (Phase 21.1b: `PaperLedger.__init__` migrates
  whatever it opens, so a UI smoke test must never point at a real machine's
  ledger by accident -- `tests/python/test_phase_20_streamlit_ui.py`'s
  `test_all_eight_pages_render_without_exceptions` was fixed the same way).

Negative-controlled during review, both in Phase 21.1 and again for 21.1b:
disabling the replay-prefix check, splitting `commit_step` into separate
per-table transactions, and (21.1b) reverting the exact-comparison fix back
to the `1e-9` tolerance were each independently confirmed to make their
corresponding regression test fail before the fix was in place.

## Launch

```
cmake --build build/cpp --target quant_paper_trading_targets_csv quant_paper_trading_run_tests
ctest --test-dir build/cpp
PYTHONPATH=python pytest tests/python/test_phase_21_paper_trading.py tests/python/test_phase_21_streamlit_ui.py
streamlit run python/alpha_agent/ui/app.py   # Paper Trading page
```
