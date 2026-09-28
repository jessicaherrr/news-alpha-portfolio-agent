# Phase 13.5C -- Real-Market Research + Validation

The first phase that runs the predeclared strategy hypotheses through the frozen
Phase 13 reliability machinery on **real** CME research/validation data. A
strategy may `PASS`, `REJECT`, or `INCONCLUSIVE`. Bad real-market results are
valid system outputs -- there is no attempt to make strategies pass.

**Status: the full real-market matrix has been run, then RERUN after a
contract-economics correction found during that first run (see below).** The
pre-correction checkpoint is commit `52222c3`; it is superseded for performance
interpretation and retained in git history.

**Status detail:** Every canonical trial went
through the frozen Phase 13 / 13.1 / 13.2 machinery on the real 2018-2024 CME
data; the corrected global 107-hypothesis BH/FDR family and DSR effective-trial
count drive the headline verdicts; cross-market evidence and the
degraded-vendor-days sensitivity pass are computed but never enter the 107.
Artifacts: `outputs/phase_13_5c/REAL_MARKET_REPORT.md` (+ `.json`),
`<ROOT>__<FAMILY>__validation_report.json`, `VERDICT_TABLE.csv`,
`TRIAL_FAMILY.csv`, `CROSS_MARKET_SUMMARY.csv`, `DATA_QUALITY_SENSITIVITY.csv`.
Nothing in this phase downloads, queries, cost-fetches, loads, features, selects
on, or reports the 2025 `LOCKED_HOLDOUT`.

Driver: `python scripts/phase_13_5c_research.py --run`
(`alpha_agent.validation.phase_13_5c_matrix` + `.phase_13_5c_report`). Roll
plumbing QA: `python scripts/phase_13_5c_roll_qa.py` -> `outputs/phase_13_5c/roll_qa.json`.

### Contract-economics defect found here, and CORRECTED (supersedes commit 52222c3)

The first real-market matrix run (commit `52222c3`) surfaced a Phase 04.5
contract-economics defect. `scripts/phase_13_5c_contract_economics_qa.py` audits
each root's DERIVED USD point value (`ContractSpec.multiplier`) against the
published CME contract specification. Pre-correction:

| root | derived point value | published | derived tick value | published tick value | status |
|---|---|---|---|---|---|
| ES | 50.0 | 50 | $12.50 | $12.50 | PASS |
| NQ | 20.0 | 20 | $5.00 | $5.00 | PASS |
| CL | 1000.0 | 1000 | $10.00 | $10.00 | PASS |
| GC | 100.0 | 100 | $10.00 | $10.00 | PASS |
| **ZN** | **0.064** | **1000** | **$0.001** | **$15.625** | **DATA_QUALITY_FAILURE** |

ZN was wrong by exactly **15,625x**. Root cause, from the stored ZN definition
records (`ZNH8`, verified):

```
min_price_increment        = 0.015625      (1/64 -- correct)
min_price_increment_amount = 0.001         <-- NOT the USD tick value
display_factor             = 1.0
unit_of_measure_qty        = 100000.0      ($100,000 FACE value, not the point value)
main_fraction              = 32.0          (quoted in 1/32nds of a percent of par)
contract_multiplier        = 2147483647    (INT32 sentinel -- correctly distrusted)
```

Cross-root contrast of the same stored fields (the clinching evidence):

| root | min_price_increment | min_price_increment_amount | display_factor | mpia/df | true tick value | unit_of_measure_qty |
|---|---|---|---|---|---|---|
| NQ | 0.25 | 0.05 | 0.01 | **5.0** | $5.00 | 20 (= point value) |
| ES | 0.25 | 0.125 | 0.01 | **12.5** | $12.50 | 50 (= point value) |
| GC | 0.1 | 1.0 | 0.1 | **10.0** | $10.00 | 100 (= point value) |
| CL | 0.01 | 0.1 | 0.01 | **10.0** | $10.00 | 1000 (= point value) |
| **ZN** | 0.015625 | **0.001** | **1.0** | **0.001** | **$15.625** | 100000 (= FACE value) |

For the four decimal roots `min_price_increment_amount / display_factor` IS the
USD tick value and `unit_of_measure_qty` IS the point value, so `price_scale`
resolves to 1.0 and the old derivation was right. For ZN neither holds.

**Correction (generic, quote-convention driven -- no root special case).**
`derive_contract_economics` now branches on the quote convention: a non-sentinel
`main_fraction` means the product is quoted as a **percent of par**, so

```
price_scale     = 0.01
point_value_usd = unit_of_measure_qty * 0.01      -> ZN: 100000 * 0.01 = $1,000
tick_value_usd  = min_price_increment * point_value_usd  -> 0.015625 * 1000 = $15.625
```

and `min_price_increment_amount / display_factor` is never consulted on that
path. Decimal products keep the previous derivation exactly.
`allow_fractional=False` still refuses fractional products outright, and
`contract_multiplier` is still never trusted. **All five roots now PASS the
contract-economics QA gate.**

**Research-integrity invariance** (`scripts/phase_13_5c_economics_invariance.py`,
artifact `CONTRACT_ECONOMICS_CORRECTION.json`). The correction changes ONLY
contract economics / Fill-derived USD numbers. Proved empirically by re-running
the original defective derivation and comparing: every ZN canonical and
predeclared-neighbour `TargetSchedule` semantic hash is **identical** pre/post
(the strategy makes exactly the same decisions), the ZN daily signal series is
identical, ZN's `tick_size` is unchanged (only the point value moves
0.064 -> 1000), and ES/NQ/CL/GC contract economics are unchanged. The frozen
candidate-manifest fingerprint, the 107-hypothesis family composition
fingerprint, every `StrategySpec` fingerprint, the canonical parameters, the
neighbourhood identities and the `ReliabilityPolicy` fingerprint are all
unchanged.

**Commit `52222c3` is SUPERSEDED for performance interpretation** and retained
unmodified in git history. Because 20 of the 107 hypotheses were ZN hypotheses
and entered the global multiple-testing family, its global BH q-values and DSR
are superseded too. The whole matrix was rerun from the corrected code; no
result was reused, rescued or tuned.

### Roll execution on real multi-contract data (auxiliary marks)

The frozen C++ roll close-leg prices a position held across a roll from a real
**same-timestamp** outgoing-contract close (`RejectDefer` default: no stale
price, no retroactive fill). The clean continuous-front execution feed never
carries that bar. Phase 13.5C adds `EngineConfig::roll.close_marks` -- an
out-of-band `(instrument_id, ts) -> close` map, built deterministically from the
already-acquired roll-overlap raw bars (`data.real_market_dataset.roll_close_marks`),
consulted **only** by the close-leg. It is never a `MarketEvent`: no effect on
`MarketState`, `bars_seen`, `Strategy::decide`, `BarHistory`, latency, or
daily-equity sampling; empty => byte-identical frozen path;
`rolls_priced_auxiliary_marks <= rolls_priced_contemporaneous`. Thin-market rolls
(GC / ZN overnight) can resolve a bounded number of bars after the nominal
instant -- still a real same-timestamp close, recorded as `rolls_deferred`.

---

## 1. Access-discipline manifest (`data/manifests/phase_13_5c/access_discipline.json`)

| window | role | engineering access | performance access |
|---|---|---|---|
| 2018-01-01 .. 2022-12-31 | RESEARCH | yes | yes |
| 2023-01-01 .. 2024-12-31 | VALIDATION | yes | **released only for this Phase 13.5C evaluation** |
| 2025-01-01 .. 2025-12-31 | LOCKED_HOLDOUT | **no** -- `downloaded = false` | **no** |

The 2023-2024 validation raw market data was previously inspected in Phase 13.5B
for **data-engineering QA only** (the 13-check acquisition QA gate), **not** for
strategy-performance selection. The latest allowed performance date is
**2024-12-31**. Any code path requesting 2025 market payload fails loudly
(`assert_offline_window` / `assert_no_holdout_ts` / `acquisition.guard_no_holdout`).

## 2. Data path (offline reconstitution)

Phase 13.5B wrote the derived layer-3/4/6 parquet as single files per root, so
the 2023-2024 run overwrote the 2018-2022 layers; only the immutable raw DBN
artifacts survived intact. `python/alpha_agent/data/real_market_dataset.py`
rebuilds the full 2018-2024 layers per root **strictly offline**:

* `load_stored_ohlcv_or_fail` / `load_stored_definitions_or_fail` resolve a
  `HistoricalRequest` to its write-once artifact, verify the SHA-256, and decode
  it locally (`databento.DBNStore.from_file` only -- no client, no network).
  A missing artifact raises `MissingRawArtifact` naming the request. There is
  **no** download fallback, no `metadata.get_cost`, no `get_range` payload call,
  no new spend.
* `reconstitute_root` replays `canonicalize` -> `build_futures_history` over the
  union of the research and validation windows.
* `scripts/phase_13_5c_build_data.py --verify` runs a 13.5B-style QA gate
  (no canonical errors, strictly increasing ts/contract, no dup bars, every bar
  resolves to a live contract, trading_day/session labels, 0 roll fallback,
  **no ts >= 2025-01-01**, forward-adjust causal, one daily-signal row per
  trading day).

### Signal vs execution cadence

| family | SIGNAL | EXECUTION |
|---|---|---|
| TSMOM, MA Trend, Breakout, Mean Reversion | **daily**: causal `trading_day` aggregation of the 1m forward-adjusted continuous -> one bar per `(root, trading_day)` -> daily `StrategyDecision` stamped at that trading day's last eligible 1m event | **native 1-minute** `RawContract` bars -> next eligible 1m event -> Risk -> ExecutionSimulator -> Fill -> Portfolio |
| Silver Bullet | native 1-minute | native 1-minute |

Daily cadence applies **only** to the feature/signal path. No daily execution
bars exist anywhere. `targets.csv` carries the sparse daily intent; the C++
engine executes on the native 1m raw-contract bars.

### Forward-adjusted continuous (causal signal)

`build_forward_adjusted_series` (`data/backadjust.py`,
`AdjustmentMode.FORWARD_ADJUSTED`): the first contract segment is unadjusted and
`adjustment(T) = -sum(gap for rolls with effective_ts_ns <= T)`. A bar at `T`
therefore reflects **only** rolls that have already happened by `T` -- no future
roll basis ever touches a feature computed at `T` (prefix-invariance test), and
the roll-jump artifact of the raw continuous series is removed. The frame carries
no executable identity (`price_domain = BACK_ADJUSTED`) and is rejected by the
execution guard -- it is never a Fill price. Backward point-in-time
back-adjustment was rejected: it bakes in-window roll gaps as constant offsets
(bounded look-ahead) and truncates at `as_of`.

### Data-quality sensitivity (degraded vendor days)

Databento flagged 2020-02-27, 2020-07-01, 2021-12-05, 2022-01-02, 2024-09-18 as
`degraded`. Canonical analysis uses them **as delivered** (no silent
forward-fill). One predeclared **data-quality sensitivity** analysis re-runs the
identical frozen process on the sample with the affected `trading_day`s excluded
(`degraded_trading_days` -- deterministic UTC-date -> trading_day map per root)
and reports whether the verdict/evidence materially changes. It is a robustness
analysis of the **same** hypothesis -- **not** a new trial, and its reruns/
p-values are **not** added to the BH/FDR denominator. Output is stored separately
under `outputs/phase_13_5c/data_quality_sensitivity/`. Never used for
strategy/parameter selection.

## 3. Frozen candidate manifest (`data/manifests/phase_13_5c/candidate_manifest.json`)

`python/alpha_agent/strategy/candidates_phase_13_5c.py`, committed **before** any
real-market performance. `manifest_fingerprint` (sha256 over canonical JSON) is
deterministic. A test asserts no performance field can appear.

**21 canonical trials** = 20 baseline (4 families x 5 roots) + 1 Silver Bullet
(NQ). Per trial: canonical `StrategySpec` fingerprint, canonical params,
predeclared parameter neighbourhood (+/-1 grid step per parameter, all inside the
declared `param_grid_ranges`), feature fingerprints, root applicability,
execution assumptions, risk assumptions, cost scenarios, multiple-testing family
id.

Textbook canonical params (baseline horizons/windows are in **trading days**):

| family | canonical |
|---|---|
| TSMOM | fast_horizon 20, slow_horizon 120, size 1 |
| MA Trend | fast_window 50, slow_window 200, size 1 |
| Breakout | lookback 55, size 1 (Donchian/Turtle) |
| Mean Reversion | zscore_window 20, entry_z 2.0, exit_z 0.5, size 1 |
| Silver Bullet (NQ) | `SILVER_BULLET_BENCHMARK` exactly as frozen in Phase 12/12.1 |

### Silver Bullet -- NQ only

`SILVER_BULLET_BENCHMARK` was frozen in Phase 12 as an **NQ hypothesis**,
including its NQ session/window semantics. It is evaluated on NQ **exactly as
frozen** (no window / sweep / displacement / FVG / retracement / exit change).
ES/CL/GC/ZN are recorded `NOT_EVALUATED` with typed reason
`benchmark_not_predeclared_for_root`; they are not canonical trials and produce
no p-value. A root-specific Silver Bullet window would be a NEW hypothesis with
its own `StrategySpec` fingerprint / trial identity, for a later phase.

## 4. Validation configuration

* **Split**: TRAIN 2018-2022 / VALIDATION 2023-2024 / LOCKED_HOLDOUT
  2025-01-01..2026-01-01 (**metadata only -- never loaded**).
* **Headline OOS = the VALIDATION window** via the additive, opt-in
  `ValidationEngine(oos_split_role=SplitRole.VALIDATION)` parameter (13.1/13.2
  additive-only precedent; `None` keeps the frozen Phase 13 behaviour). The
  headline OOS run, cost stress, parameter neighbourhood and schedule-shift null
  all run over 2023-2024; the walk-forward folds are built **inside TRAIN
  (2018-2022) only**. The holdout window is never selectable
  (`oos_split_role == LOCKED_HOLDOUT` raises).
* **Frozen Phase 13 machinery, unchanged**: `cost_stress` 1.0x / 1.5x / 2.0x
  (C++ reruns, never a Python subtraction); `SCHEDULE_TIME_SHIFT` null =
  **DIAGNOSTIC** (never gates); `CENTERED_BLOCK_BOOTSTRAP` = **GATING**;
  daily-frequency DSR; BH/FDR over the full inspected trial set; bootstrap CIs;
  `causal_volatility_regime_labels` (train-derived cut points); trend regime
  `NOT_EVALUATED` (no frozen definition).
* **Global multiple testing (Part 1.1 -- unique trial identity)**: the BH/FDR
  family contains each inspected performance **hypothesis exactly once**.
  `validation/phase_13_5c_trials.py`:
  * `trial_semantic_identity(strategy_fingerprint, root, schedule_hash,
    dataset)` -- deterministic, **role-free**. The same canonical strategy-root
    result never appears once as `role="canonical"` and again as
    `role="cross_market"`.
  * `build_global_trial_family(records)` -- de-duplicates by that identity,
    **refuses** `role="cross_market"` records (`CrossMarketIsNotATrial`), and
    raises `DuplicateTrialIdentity` on a genuine conflict.
  * The 20 baseline canonical trials **are** the per-root results (4 families x
    5 roots). `CrossMarketEvidence` is a **descriptive summary** of those
    existing trial results -- it references their identities and is **not** a BH
    observation. Data-quality-sensitivity reruns point back to the same
    hypothesis identities and are also excluded.
  * **Unique BH/FDR trial family = 107** = 21 canonical + 86 predeclared
    neighbours + 0 ablations. (The earlier "127" wrongly added 20 cross-market
    roots.) BH/FDR, q-values, the DSR effective-trial count, the
    `MultipleTestingFamily` fingerprint and the validation-report lineage all
    use 107. The `--run` path passes `cross_market_runs=None` to
    `ValidationEngine` so no `role="cross_market"` `TrialRecord` is ever created;
    `CrossMarketEvidence` is computed separately from the per-root
    `RootRunSummary` objects.
  * The frozen manifest records this as `bh_fdr_trial_family`
    (`unique_trial_count`, `family_composition_fingerprint`); it is recomputed
    from the semantic identities and asserted unique on every build.

## 5. Risk identity (truthful)

The frozen reference CLI `quant_backtest_targets_csv`
(`cpp/apps/backtest_targets_csv.cpp`) constructs `quant::PassThroughRiskManager`
-- *"frozen reference path (Phase 19 wires RiskConfig)"*. **Phase 08 hard risk
limits are NOT enforced on the Phase 13 / 13.5C execution path.** Every target is
passed through unchanged; the candidate `size` is a small integer (1) by
construction. The manifest records `risk_manager = "PassThroughRiskManager"`,
`hard_risk_limits_enforced = false`, and this limitation verbatim. No
risk-system scope is added in Phase 13.5C.

## 6. Compute estimate (`outputs/phase_13_5c/run_estimate.json`)

`python scripts/phase_13_5c_research.py --estimate-only` executes **zero**
strategy backtests and inspects **no** real PnL / Sharpe / signal. It
reconstitutes the data only to count exact bars, enumerates the run matrix from
the frozen manifest, and estimates compute conservatively from
`runs x bars / assumed_throughput` (a **stated assumption**, not a measurement).

**A. unique statistical trials** (BH/FDR denominator, DSR effective-trial count):
**107** = 21 canonical + 86 neighbours + 0 ablations. Cross-market roots and the
data-quality sensitivity pass are **not** trials.

**B. C++ backtest executions** (compute, not trials) -- by purpose, per pass:

| purpose | statistical? | runs |
|---|---|---|
| canonical headline OOS (validation 2023-2024) | yes -- 1 per canonical trial | 21 |
| walk-forward folds (TRAIN 2018-2022) | diagnostic evidence | ~84 |
| parameter-neighbour runs | yes -- 1 per neighbour trial | 86 |
| cost-stress reruns (1.5x, 2.0x) | robustness | 42 |
| schedule-shift **diagnostic** null (Silver Bullet 1m only) | diagnostic, never gates | 9 |
| centered-block bootstrap **gating** null | gating | **0 C++** -- Python-only on the daily series |
| data-quality sensitivity | robustness rerun of the same hypotheses | = the whole inventory again |

Current estimate: ~242 C++ runs per pass x 2 passes = **~484 C++ executions**,
~249M processed 1m rows, ~28 min wall clock at an assumed (stated, not measured)
150k rows/s. A diagnostic or sensitivity rerun raises B without raising A. No
Databento spend; no 2025 request. `--estimate-only` runs **zero** real strategy
backtests.

## 7. Artifacts

* `data/manifests/phase_13_5c/candidate_manifest.json` -- frozen, committed
* `data/manifests/phase_13_5c/access_discipline.json` -- committed
* `outputs/phase_13_5c/run_estimate.json` -- run/trial/bar counts + estimate
* (deferred to the run) per-trial `ValidationReport` + `report_fingerprint`,
  every semantic fingerprint, global `MultipleTestingFamily`, cross-market
  evidence, verdict table, `REAL_MARKET_REPORT.md` + `.json`.

## 8. Implementation changes

Additive only:

* `data/backadjust.py`: `AdjustmentMode.FORWARD_ADJUSTED` +
  `build_forward_adjusted_series` (causal forward-adjust helper).
* `features/source.py`: `"forward_adjusted"` added to the valid back-adjustment
  modes (signal-safe, like point-in-time).
* `validation/engine.py`: `ValidationEngine(oos_split_role=...)` opt-in
  parameter. `None` reproduces the frozen Phase 13 behaviour byte-for-byte.
* `validation/phase_13_5c_trials.py` (Part 1.1): `trial_semantic_identity` +
  `build_global_trial_family` + `unique_trial_identities` -- de-duplicated
  BH/FDR family; `MultipleTestingFamily` itself is unchanged (frozen).

No change to the frozen C++ core (the Phase 13 `daily_equity` trace + optional
cost args already suffice), the Fill/accounting formulas, the Phase 10 DSL, or
the Phase 13/13.1/13.2 statistics.
