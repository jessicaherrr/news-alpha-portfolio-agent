# Baseline Strategy Library + Python→C++ Target Bridge (Phase 11)

The first end-to-end alpha-strategy phase. It proves the complete research path
without introducing any new alpha claim:

```
historical futures data
  -> Feature Engine (Phase 09)
  -> closed StrategySpec (Phase 10 DSL)
  -> CompiledStrategyPlan
  -> ReferenceEvaluator -> StrategyDecision (target position)
  -> TargetSchedule (targets.csv -- deterministic research bridge)
  -> C++ ScheduledTargetStrategy
  -> BacktestEngine -> Hard Risk -> ExecutionSimulator -> Fill -> Portfolio
  -> official BacktestResult
```

**Official fills / realized PnL / costs / positions / drawdown / portfolio output
remain C++ `Fill`-derived.** Python only writes the target-intent bundle and reads
the C++ result into a typed descriptive summary + a run manifest. There is no
alternate Python PnL calculator.

> These are **research baselines / benchmarks**. No claim of profitability,
> robustness, statistical significance or production readiness — those belong to
> Phase 13 reliability validation.

---

## 1. Baseline families

`python/alpha_agent/strategy/baselines/` — every family is expressed **only**
through the closed Phase 10 DSL. Full economic mechanism, exact formula/timing,
required data, pre-declared parameter grid ranges, and failure regimes are typed
data in `families.py` (`BASELINE_FAMILIES`).

| key | factory | mechanism | rules (ordered) | default action |
|---|---|---|---|---|
| `tsmom` | `make_tsmom_spec` | multi-horizon trend premium | both `diff_fast>0` & `diff_slow>0` → `+size`; both `<0` → `-size` | `flat` |
| `ma_trend` | `make_ma_trend_spec` | trend premium via MA crossover | `fast_ma>slow_ma` → `+size`; `fast_ma<slow_ma` → `-size` | `flat` |
| `breakout` | `make_breakout_spec` | range-breakout continuation | `breakout_up>0` → `+size`; `breakout_down<0` → `-size` | `keep_previous_target` |
| `mean_reversion` | `make_mean_reversion_spec` | short-horizon overshoot reversion | `z<-entry_z`→`+size`; `z>+entry_z`→`-size`; `|z|<exit_z`→`0` | `keep_previous_target` |

All timing: the decision uses only information through bar `T`; the C++ engine
executes at its next eligible bar (no look-ahead, no double lag — see §5).

## 2. StrategySpec factories

`baseline config → make_*_spec(...) → StrategySpec`. The result passes the
standard Phase 10 `StrategyCompiler` unchanged — no hidden execution logic, no
per-strategy code path. One `root_symbol` per spec; the same factory yields an
`ES`, `NQ`, `CL`, `GC` or `ZN` spec purely by parameter. No instrument economics
(tick / multiplier / margin) appear in strategy logic; the C++ `ContractSpec`
supplies those at execution time.

## 3. Parameter rules (`params.py`, typed & frozen)

Rejected at construction: `fast_horizon >= slow_horizon`, `fast_window >=
slow_window`, `entry_z <= exit_z`, non-positive windows, `size` outside
`1..5` (also within the compiler's `max_abs_target_units = 10`), a `root_symbol`
that is not `^[A-Z0-9]{1,12}$`. **No optimizer, grid search or walk-forward
selection** is built — the grid *ranges* are declared (before results) in
`families.py` for Phase 13+.

## 4. Target bridge schema — `targets.csv`

A **separate** research boundary next to the frozen Phase 02.5 `bars.csv` /
`contracts.csv` (never overloaded onto them). Header **exactly**:

```
ts_event_ns,root_symbol,target_units,strategy_fingerprint,matched_rule_id
```

* carries **target-position intent only** — `target_units` is a signed integer
  target position, never an order quantity;
* **never** a fill/execution/reference price, `raw_symbol`, `instrument_id`,
  `contract_month`, slippage, spread, commission, latency, margin, or a risk
  decision (both the Python `TargetSchedule` and the C++ `parse_targets_csv`
  reject any such column via the strict header + a denylist);
* `matched_rule_id` is a **non-executable audit field** and is **not** part of
  the schedule hash;
* one `strategy_fingerprint` for the whole file — **mixed fingerprints are
  rejected** on both sides;
* `root_symbol` must be a **bare root** — `NQU6` / `NQ.v.0` are rejected; the C++
  engine resolves the real execution contract through `MarketState →
  ActiveContractResolver`.

`TargetSchedule.schedule_hash()` = `targsched1:` + SHA-256 over the executable
intent only (`ts_event_ns`, `root_symbol`, `target_units`) + fingerprint + DSL /
feature-engine versions — deterministic and stable across processes.

### Schedule semantics — NO DECISION vs FLAT (Phase 11.1)

`build_target_schedule(plan, frame)` runs the Phase 10 reference evaluator and
emits a row for bar `T` **only** when every referenced feature value was present
and every rule was evaluable — i.e. the strategy genuinely had enough
information.

* **No row at `T`** ⇒ *no new strategy decision*. Warm-up bars and bars inside a
  data gap produce no row. The C++ `ScheduledTargetStrategy` returns **NO
  DECISION** (`std::optional<Signal>` = `std::nullopt`, `AbsentPolicy::NoDecision`
  default): the engine allocates no `signal_id`, counts nothing, queues nothing,
  and **does not cancel a pending earlier intent or change any target state**. In
  particular a target that was **risk-rejected on an earlier bar is never
  automatically retried** because a later bar has no row. Absence is **not** an
  implicit flat and **not** an implicit re-emission.
* **A row with `target_units = 0`** ⇒ an explicit **FLAT** decision (a real
  `Signal(0)` → order delta → flatten through Risk/Execution).

`KEEP_PREVIOUS_TARGET` (Phase 10) stays Python semantics: the reference evaluator
resolves it and writes an **explicit** row at `T` (`previous target +1` → row
`target_units = +1` → the C++ side emits a new `Signal +1`). `KEEP_PREVIOUS_TARGET
≠ NO DECISION` — they are never merged.

## 5. `ScheduledTargetStrategy` (C++) design — optional intent

`cpp/include/quant_core/scheduled_target_strategy.hpp`. `Strategy::decide` now
returns `std::optional<Signal>` (a correctness-driven Phase 11.1 change to the
frozen core — every other strategy/test was updated mechanically, semantics
unchanged; the engine simply skips id/count/queue when it gets `std::nullopt`).

At `MarketEvent T` `ScheduledTargetStrategy` looks up `(root_symbol, T)`:

* row present → `Signal(target_units)` exactly as decided in Python;
* row absent → the `AbsentPolicy`:
  * `NoDecision` (**default, normal Phase 11 path**) → `std::nullopt`;
  * `Flat` → explicit `Signal(0)`;
  * `RequireRow` → throw (dense-schedule guard);
  * `ReemitPreviousTarget` → **explicit opt-in, testing/research only** —
    re-emits the previous target as a **brand-new `Signal`** (this *does* retry a
    rejected intent); never the reference path.

In the normal path it stores **no** previous-target state. It **cannot**: know a
fill price, select a contract, read the `RiskManager` / `Portfolio`, or compute
PnL. Reference CLI:
`quant_backtest_targets_csv <bars.csv> <contracts.csv> <targets.csv> [no_decision|flat|require_row|reemit_previous]`.

## 6. Decision-time vs execution-time

A Python `StrategyDecision` at bar `T` is decision-time intent, stamped at `T` in
`targets.csv`. `ScheduledTargetStrategy` emits that `Signal` when the engine
reaches decision bar `T`; the engine executes it only at the **next eligible
execution event** per its existing latency semantics (`i + 1 + latency_bars`).
Targets are **not** pre-shifted in Python — no accidental double lag. Regression:
`test_J_*` (Python) and `test_replay_and_execution_timing` (C++) assert a
decision at `T` fills at `T+1`, never `T`.

**Independent pending intents (Phase 11.2).** With `latency_bars > 0` every
emitted `Signal` is an independent intent in a per-root FIFO queue — a later
`Signal` never overwrites an earlier not-yet-due one, and `order_delta` /
contract resolution stay execution-time. The engine already implemented this as
a `std::deque<PendingIntent>` (verified — not a single replaceable slot);
Phase 11.2 added `cpp/tests/test_latency_queue.cpp` (`quant_latency_queue_tests`,
checklist A–J) and the invariant text in `docs/EXECUTION_MODEL.md` §Latency.

## 7. Lineage / run manifest

`BaselineRunManifest` (`backtest/manifest.py`, `schema_version = "baseline-run/1"`)
records: source-data lineage (price domain, adjustment mode, source fingerprint,
paths, row count, `as_of`), the `FeatureSpec`s + feature-engine version, the
canonical `StrategySpec` + fingerprint + DSL version + warm-up, the target
schedule hash + row count + date range, the execution-config identity, the
risk-config identity, and the bundle row counts. This is **not** the Phase 14
experiment registry — a simple typed manifest.

## 8. Official PnL path

`summarize_cpp_result()` wraps one line of `quant_backtest_targets_csv` JSON in a
typed `BaselineRunSummary`. Every number is read **verbatim** from the C++
`BacktestResult`; the module does no PnL arithmetic and computes **no** Sharpe /
significance (that would duplicate Phase 13). `official_pnl_source =
"cpp_fill_events"`.

## 9. Back-adjusted futures regression

`test_R_*`: with a synthetic single roll of basis ≈ +10 —
* the **unadjusted continuous** series shows an artificial +10 jump in the 1-bar
  change straddling the roll;
* the **point-in-time back-adjusted** series (as-of the pre-roll bar) shows no
  such jump — it never uses the future roll basis;
* a **retrospective** back-adjusted frame is rejected by the evaluator gate
  (`FeatureFrame.assert_signal_safe`) — `build_target_schedule` raises;
* a point-in-time adjusted trend feature may generate a `Signal`; execution still
  occurs in the real `RawContract` at raw prices — the adjusted feature value is
  never a `Fill` price (`make_fill` + `PriceDomain::RawContract` guarantee it).

## 10. Signed-price compatibility (CL)

Every baseline references only signed-price-safe feature kinds — `diff`, `ma`,
`breakout_up`/`breakout_down`, `zscore` (a price-level z-score). No percentage
return, no log return across zero. `SIGNED_PRICE_SAFE_KINDS` is asserted in the
factories. Mean reversion runs end-to-end on a CL path through zero into negative
territory (`test_S_*`). A hand-built spec that (wrongly) uses `return` on signed
CL data simply has no signal where the base price ≤ 0 (the feature suppresses it
+ records QA) rather than emitting garbage.

## 11. Carry status — **deferred**

The closed DSL can only reference registered single-series `FeatureSpec` columns.
A production carry signal needs an overlapping front/next raw-contract curve,
which is a separate typed interface (`alpha_agent.features.carry`) with no
registered feature `kind`. Wiring it in cleanly would require new DSL primitives
(a curve / relative-contract operand, an annualized-carry feature kind) — and
prompt 11 says not to redesign Phase 10 for carry. Real curve data is also not
yet ingested. `carry_baseline_status()` returns the typed deferral note.

## 12. Tests

* Python `tests/python/test_baseline_strategies.py` — 32 cases, checklist
  A–T + V + the Phase 11.1 NO-DECISION bridge default + Phase 11.2 independent
  consecutive intents + carry/family docs.
* C++ `cpp/tests/test_latency_queue.cpp` (`ctest` target
  `quant_latency_queue_tests`) — the Phase 11.2 pending-intent queue invariant:
  three consecutive distinct Signals with `latency 2` all survive and execute in
  decision order; a later Signal does not overwrite an earlier pending one;
  `NoDecision` between queued Signals does not cancel them; signal ids stay
  decision-order monotonic; `order_delta` uses the execution-time position; a
  pending Signal crossing a roll resolves the execution-time contract; a risk
  rejection of one queued intent does not drop later ones; deterministic replay;
  `latency_bars == 0` unchanged.
* C++ `cpp/tests/test_scheduled_target.cpp` (`ctest` target
  `quant_scheduled_target_tests`) — schedule replay; decision-time ≠
  execution-time; **absent row → NO DECISION** (not flat, not retry); explicit
  `target_units = 0` → flat; **NO DECISION consumes no `signal_id` and does not
  increment `signals_generated`**; **a risk-rejected target is not auto-retried
  after an absent row**; an **explicit repeated target row does** create a new
  intent; a **pending latency-delayed intent is not cancelled by NO DECISION**;
  the mandatory `RiskManager` gate still applies (`MaxContractsRiskManager`
  RESIZE); official realized PnL from `Fill` events only (hand-checked);
  `parse_targets_csv` guards; `RequireRow` throws; `ReemitPreviousTarget` is an
  explicit opt-in that *does* retry; deterministic replay.

## 13. Frozen C++ core / boundary

**Correctness-driven change (Phase 11.1, approved):** `Strategy::decide` now
returns `std::optional<Signal>` — `std::nullopt` is a first-class **NO DECISION**
outcome, structurally distinct from a flat `Signal`. The engine, when it gets
`std::nullopt`, allocates no `signal_id`, increments no counter, validates/queues
nothing, and leaves pending intent and target state untouched. Every existing
strategy and test was updated **mechanically** (return type only; bodies and
semantics unchanged) — `TimeSeriesMomentum` still always emits.

**Otherwise unchanged**: `MarketEvent`/`Signal`/`Order`/`Fill`/`RiskDecision`,
the `BacktestEngine` execution/roll/EOT semantics, `ExecutionSimulator`,
`PortfolioAccountant`, `RiskManager`, `make_fill`, and the Phase 02.5 `bars.csv`
/ `contracts.csv` wire formats. Phase 11 **adds** `ScheduledTargetStrategy`,
`parse_targets_csv`, and the `quant_backtest_targets_csv` CLI; `targets.csv` is a
new **separate** research boundary.

## 14. What Phase 12 (Silver Bullet) receives

* the baseline factories + `families.py` docs as comparison benchmarks;
* the deterministic `TargetSchedule` bridge + `schedule_hash` and the
  `quant_backtest_targets_csv` reference path — any strategy that produces a
  Phase 10 `StrategySpec` now has an end-to-end C++-official backtest;
* `BaselineRunManifest` / `BaselineRunSummary` typed lineage & descriptive
  metrics (no Sharpe / significance — reserved for Phase 13).
