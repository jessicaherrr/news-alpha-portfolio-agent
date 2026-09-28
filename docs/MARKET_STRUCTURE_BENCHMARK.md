# Quantified ICT Silver Bullet / Market-Structure Benchmark (Phase 12)

An informal trader-folklore idea turned into an explicit, closed, causal
quantitative definition and run through the **exact same** Phase 11 research
bridge as the academic baselines:

```
informal discretionary idea
  -> explicit typed definitions        (this doc)
  -> causal registered features        alpha_agent.features.market_structure
  -> closed StrategySpec               make_silver_bullet_spec  (Phase 10 DSL)
  -> StrategyDecision                  ReferenceEvaluator
  -> targets.csv                       TargetSchedule (Phase 11 boundary)
  -> C++ BacktestEngine                official Fill-derived PnL
```

> **Phase 12 makes no claim of profitability, alpha, robustness, or statistical
> significance.** It is a *formalised hypothesis* and a *benchmark*. Whether there
> is any statistical evidence is decided by Phase 13 reliability validation.
> Phase 12 asserts **mechanics** and reports **descriptive frequency** only.

The deterministic C++ Quant Core is **unchanged** by this phase.

**Phase 12.1** (semantic-identity fixes, no behaviour change to sweep /
displacement / FVG / retracement / state machine / window / exits / DSL / bridge):
(1) distance thresholds carry an **explicit** `distance_unit` enum — no silent
`tick_size → 1.0` fallback (§3); (2) `prior_session_extreme` requires real
calendar `trading_day` labels and fails loudly otherwise — no gap-segment
aliasing (§2). Both are now part of the parameter fingerprint.

---

## 1. Formal definition (the sequence)

```
liquidity reference  ->  liquidity sweep  ->  displacement  ->  fair value gap
    ->  retracement into the FVG  ->  target-position intent
```

Every step has a deterministic mathematical definition below; every step is
strictly causal (uses only bars completed at or before the bar it is evaluated
on). Bullish and bearish are exact mirrors. There is **no discretionary visual
judgement**, no `eval`/`exec`, no handwritten dataframe logic outside the closed
Feature Engine, and no direct `Order` generation.

The state machine, its parameters, and its per-bar outputs live entirely in the
closed `FeatureRegistry` as two registered kinds with a typed scalar parameter
contract:

| kind | column | meaning |
|---|---|---|
| `silver_bullet_intent` | signed `-1 / 0 / +1` | the DSL-facing target-position sign (lifecycle-managed) |
| `silver_bullet_state`  | `0..5` | setup lifecycle state code (diagnostic, not a signal) |

`state` codes: `IDLE=0`, `SWEEP_OBSERVED=1`, `DISPLACEMENT_CONFIRMED=2`,
`FVG_ACTIVE=3`, `RETRACEMENT_TRIGGERED=4` (== in a position until a typed exit),
`EXPIRED=5` (transient — a stale setup that reset to `IDLE` this bar).

---

## 2. Liquidity reference (`prompts/12` §4)

Two typed, causal reference families (`liquidity_reference` parameter):

* **`rolling_nbar_extreme`** (default) — for a sweep evaluated on bar `T`:
  `ref_low[T]  = min(low[T-L .. T-1])`,
  `ref_high[T] = max(high[T-L .. T-1])`,
  where `L = liquidity_lookback`. The window is the **prior `L` bars**; it never
  includes bar `T` itself or any later bar (`seg_shift(seg_rolling(..), 1)` — the
  same construction as the Phase 09 `breakout_up` feature).
* **`prior_session_extreme`** — the completed high / low of the **immediately
  preceding trading day**, carried forward inside the current day. A bar on day
  `D` sees only the finished extremes of the most recent day `D' < D`. This
  requires a reliable `trading_day` label column from the Phase 03/04
  `SessionCalendar`; a missing/incomplete label is a **loud error**
  (`SilverBulletSessionLabelsMissing`), never silently reinterpreted as a
  gap-delimited segment (Phase 12.1 §5; `test_12_1_G` / `test_12_1_H` /
  `test_12_1_no_silent_gap_segment_substitution`). If there is no prior day there
  is no reference → NO DECISION.

**No centered swing calculations.** A reference at `T` is bit-identical whether
the series ends at `T` or continues far past it (Phase 09 prefix-invariance;
regression `test_A_C_liquidity_reference_is_prior_bars_only`).

---

## 3. Liquidity sweep (`prompts/12` §5) & distance-unit semantics (Phase 12.1 §1–4)

Every price-distance threshold (`sweep_penetration`, `fvg_min_width`) carries an
**explicit** `distance_unit` — a closed enum, fixed per run, part of the
fingerprint:

| `distance_unit` | meaning | `tick_size` |
|---|---|---|
| `price_units` (default) | the value is already in normalized price units | must be unset — a non-zero `tick_size` here is a loud error |
| `ticks` | the value is a multiple of `tick_size` | **required**: a finite number `> 0` from typed contract/tick metadata |

`unit_price = tick_size` for `ticks` (validated `> 0`), `1.0` for `price_units`
(the identity for price units — **not** a fallback). There is **no**
`tick_size → 1.0` silent substitution: `distance_unit='ticks'` with a missing /
non-finite / non-positive `tick_size` raises at parameter construction, at the
registry parameter-contract check, and in the detector (`test_12_1_A–D`). So
`NQ` `sweep_penetration = 1` tick means `0.25`, `CL` `1` tick means `0.01`, never
`1.0`. `1 tick` and `1.0 price unit` produce **different** semantic fingerprints
(`test_12_1_E`). Tick metadata is never hard-coded and never guessed; the
`ContractSpec` stays authoritative for execution economics.

With penetration tolerance `pen = sweep_penetration * unit_price`:

* **sell-side sweep** (precedes a **bullish** setup):
  `low[T] < ref_low[T] - pen`  **and**  `close[T] > ref_low[T]`
  (price took out the prior low and *closed back above* it).
* **buy-side sweep** (precedes a **bearish** setup):
  `high[T] > ref_high[T] + pen`  **and**  `close[T] < ref_high[T]`.

If both fire on one OHLC bar → `ambiguous_intrabar_sweep`, no setup. No future
confirmation is used. The swept level is recorded as the setup's `liquidity_level`.

---

## 4. Displacement (`prompts/12` §6)

Evaluated on a bar `Td` **strictly after** the sweep bar (§7 below). With
`atr[Td]` = rolling mean of true range over `displacement_atr_window` bars
(`true_range` is signed-price safe), `body = close - open`, and
`close_loc = (close - low) / (high - low)` (`0.5` when `high == low`):

* **bullish**: `body > 0` **and** `body / atr >= displacement_atr_multiple`
  **and** `close_loc >= displacement_close_loc`.
* **bearish**: `-body > 0` **and** `-body / atr >= displacement_atr_multiple`
  **and** `close_loc <= 1 - displacement_close_loc`.

No subjective words ("strong", "impulsive") — every threshold is a typed
parameter.

---

## 5. Fair value gap (`prompts/12` §7)

A causal three-bar FVG on bars `(Td-1, Td, Td+1)` where `Td` is the displacement
bar. **It is first observable at the completion of bar `Tf = Td + 1`** and is
timestamped there — never at `Td`.

* **bullish FVG**: `low[Tf] > high[Td-1]` →
  `lower = high[Td-1]`, `upper = low[Tf]`.
* **bearish FVG**: `high[Tf] < low[Td-1]` →
  `lower = high[Tf]`, `upper = low[Td-1]`.

Requires `upper - lower > 0` and `>= fvg_min_width * unit_price` (same explicit
`distance_unit` semantics as the sweep — §3; `test_12_1_F`), else the setup is
dropped (`fvg_absent` / `fvg_too_small`). Recorded: direction, `lower`, `upper`,
width, `fvg_created_ts_ns`. If bar `Td+1` does not form a gap, the setup expires
(the FVG is tied to the displacement bar — the smallest deterministic rule).

---

## 6. Retracement (`prompts/12` §8)

Evaluated on a bar `Tr` **strictly after** `Tf` (a retracement can never be
observed before the FVG exists — `test_H_I` / `test_I`). With
`span = upper - lower` and `f = retracement_fraction` (`0.0` = near edge,
`0.5` = midpoint, `1.0` = far edge):

* **bullish**: `retr_level = upper - f * span`; triggered when `low[Tr] <= retr_level`.
* **bearish**: `retr_level = lower + f * span`; triggered when `high[Tr] >= retr_level`.

Only the bar's own low/high is used — no claim about intrabar ordering (§9). The
detector never uses future knowledge of whether the FVG *eventually* fills. If
the FVG is not retraced within `setup_expiry_bars` bars of `Tf`, it expires
(`retracement_not_reached`).

`Tr` is the **StrategyDecision bar** (`decision_ts_ns`).

---

## 7. State machine / event ordering (`prompts/12` §9, §10)

Smallest closed deterministic machine. State is **typed, bounded, deterministic,
resettable, auditable** and lives only in the Feature Engine.

| bar | what becomes observable |
|---|---|
| `T0` | liquidity sweep (`SWEEP_OBSERVED`) |
| `Td > T0` | displacement (`DISPLACEMENT_CONFIRMED`) |
| `Tf = Td + 1` | FVG (`FVG_ACTIVE`) |
| `Tr > Tf` | retracement → `RETRACEMENT_TRIGGERED`, `intent = ±1` |
| next eligible C++ bar after `Tr` | execution (engine latency semantics) |

`Td` is required to be **strictly after** `T0`, and `Tr` strictly after `Tf`:
one OHLC bar cannot prove its low happened before its close, so
sweep+displacement (or FVG+retracement) never collapse into a single bar
(`prompts/12` §16; `test_D_displacement_must_follow_the_sweep_bar`,
`test_N_single_bar_cannot_produce_a_setup`). While a setup is in progress the
machine is single-track; while in a position only typed exits are evaluated.

**No-decision vs flat**: before a liquidity reference exists in a segment the
`intent` feature is `NaN` → the Phase 11 bridge emits **no row** (NO DECISION).
Once a reference exists, `0.0` is a real *flat* decision.

---

## 8. Session / time window (`prompts/12` §11) & resets (§12)

The window is expressed as **exchange-local wall-clock** (`window_start_local` /
`window_end_local`, `"HH:MM"`, `start < end`, no midnight wrap) interpreted
through the existing Phase 03/04 `SessionCalendar` (`window_calendar` = calendar
root). Timezone and **DST are handled by `zoneinfo`** — there is no hard-coded
UTC offset (`SessionCalendar.in_local_window`; `test_M`). A retracement entry is
only taken **inside** the window; an open position is flattened once the window
closes.

The canonical benchmark uses `09:00–10:00 America/Chicago` for `NQ`, which is the
CME-local representation of the "as taught" NY-AM Silver Bullet window
(≈ 10:00–11:00 America/New_York). This is a **documented convention, not a claim
that this window is optimal** — London / NY-AM / NY-PM are all research variants.

Market-structure state does **not** leak across:

* a new trading session — `default_session_policy = RESET_ON_SESSION_AND_GAP`;
  the detector iterates per segment and forces `intent = 0` + `session_reset` on
  a segment boundary (`test_K`);
* a large data gap — same segment mechanism;
* an `is_roll_boundary` row — hard reset (`roll_reset`), see §10;
* an expired setup / a completed setup — back to `IDLE`.

---

## 9. Exit semantics (`prompts/12` §15)

The smallest deterministic benchmark behaviour — **target-position decisions
only**, no stop-loss / take-profit, no fake same-bar Python PnL. An open position
returns to flat (`intent = 0`, an explicit `silver_bullet_exit` event) on the
first of:

1. `max_holding_bars` bars held;
2. the trading window closes (or, with no window, the session ends);
3. the session resets / roll boundary;
4. **setup invalidation** — `close` moves back through the swept `liquidity_level`
   against the position.

No dynamic leverage, martingale, averaging down, or risk-based sizing.
`target_units` is `±size` (`1..5`) or `0`. Hard Risk stays authoritative in C++.

---

## 10. Signed prices (§17) & rolls (§18)

**Signed prices**: every comparison is relational and every magnitude is an
arithmetic difference or true range — valid for zero and negative prices. No log
return, no percentage return. A CL path straight through zero produces a normal
setup (`test_O`).

**Rolls**: the detector consumes a research price domain
(`RawContract` / `RawContinuous` / point-in-time `BackAdjusted` — a retrospective
back-adjusted source is rejected by the Phase 09/10 signal-safety gate,
`test_R`). For directional pattern research prefer a causal adjusted domain. On a
raw *continuous* feed the detector resets its state at every `is_roll_boundary`
row, so a large roll basis never manufactures a sweep or displacement
(`test_P_large_roll_basis_does_not_manufacture_a_setup`). Execution still
resolves to the real `RawContract` in C++ (`make_fill` + `PriceDomain::RawContract`;
Phase 11 guarantee, unchanged).

---

## 11. `StrategySpec` factory (`prompts/12` §13)

`alpha_agent.strategy.baselines.silver_bullet`:

```python
make_silver_bullet_spec(SilverBulletParams(
    root_symbol, size,
    liquidity_lookback, liquidity_reference,
    sweep_penetration, distance_unit, tick_size,
    displacement_atr_window, displacement_atr_multiple, displacement_close_loc,
    fvg_min_width,
    retracement_fraction,
    setup_expiry_bars, max_holding_bars,
    window_start_local, window_end_local, window_calendar,
)) -> StrategySpec        # the authoritative Phase 10 schema
```

The returned spec compiles through the **standard** `StrategyCompiler` with no
special handling. Its only feature is the `silver_bullet_intent` column; its
rules are:

```
sb > 0.5   -> target +size
sb < -0.5  -> target -size
default    -> flat
```

`intent` already encodes the full held state every bar, so no `keep_previous`
is needed. Invalid parameter combinations are rejected at construction
(`test_S_invalid_params_rejected`).

---

## 12. Audit record (`prompts/12` §20)

`SilverBulletAudit` (frozen, `extra="forbid"`), one per setup that reached
`SWEEP_OBSERVED`:

`root_symbol`, `setup_direction`, `liquidity_reference_type`, `liquidity_level`,
`distance_unit`, `distance_unit_price`, `sweep_ts_ns`, `displacement_ts_ns`,
`fvg_created_ts_ns`, `fvg_lower`, `fvg_upper`, `retracement_ts_ns`,
`decision_ts_ns`, `expiry_ts_ns`, `expiry_reason`, `parameter_fingerprint`
(`sbparams1:<sha256[:32]>` — now includes `distance_unit` + `tick_size`),
`diagnostics`.

It contains **no `Fill` price and no `RiskDecision`**
(`test_audit_record_has_no_execution_fields`).

## 13. Failure / ambiguity diagnostics (`prompts/12` §19)

`SilverBulletResult.diagnostics` is a `Counter` — never a silent feature `0`.
Keys include: `bars_without_liquidity_reference`, `setups_started`,
`reached_displacement`, `reached_fvg`, `entries` / `entries_long` /
`entries_short`, `displacement_bar_failed`, `fvg_absent`, `fvg_too_small`,
`expired_before_fvg`, `expired_no_retracement`, `retracement_blocked_by_window`,
`ambiguous_intrabar_sweep`, `session_resets`, `roll_resets`,
`exit_max_holding_bars` / `exit_window_end` / `exit_session_end` /
`exit_setup_invalidation`. When setups start but none reach an entry, an
aggregate note is also pushed to the `FeatureQAReport`.

---

## 14. Same pipeline as the baselines (`prompts/12` §21)

No second backtester. `FeatureFrame -> StrategySpec -> StrategyCompiler ->
ReferenceEvaluator -> TargetSchedule -> ScheduledTargetStrategy ->
quant_backtest_targets_csv`. Official PnL is C++ `Fill`-derived
(`summarize_cpp_result`, `official_pnl_source = "cpp_fill_events"`). The Phase 11
`BaselineRunManifest` / `BaselineRunSummary` describe a Silver Bullet run exactly
as they describe a baseline run.

---

## 15. Descriptive benchmark comparison (`prompts/12` §22)

Allowed Phase 12 outputs (all read verbatim from the C++ `BacktestResult` or
counted by the detector): number of setups, long / short setups, signals, fills,
contracts traded, gross PnL, costs, net PnL, max drawdown, holding-duration
counts. **Not** computed here: statistical significance, Deflated Sharpe,
bootstrap CIs, FDR, robustness, parameter superiority — those are Phase 13.

---

## 16. Parameter discipline (`prompts/12` §23)

One canonical "as taught" set: `SILVER_BULLET_BENCHMARK`. A small **pre-declared**
research grid *range* (not a grid to search): `SILVER_BULLET_GRID_RANGES`. No
optimizer, no walk-forward selection, no search over the synthetic fixture is
built in Phase 12.

---

## 17. Data (`prompts/12` §24)

Built and validated entirely on deterministic synthetic fixtures
(`tests/python/silver_bullet_fixtures.py`). **No Databento data was downloaded.**
A single small real-data slice (e.g. one month of `GLBX.MDP3` `NQ` front-month
`ohlcv-1m` around a known NY-AM session, ~1 GB-class request — call
`metadata.get_cost()` first) would sharpen the displacement/ATR and
window-population calibration, but is deferred and must be explicitly
cost-approved before any download.

---

## 18. Limitations / ambiguous cases

* FVG is tied to `Td+1` only — a gap that forms 2–3 bars after displacement is
  not captured (deliberate: smallest rule).
* Single-track state machine — no simultaneous long & short setup tracking, and
  no new setup detection while in a position.
* Exit is target-position only; a true protective stop would need C++
  `ExecutionSimulator` bracket orders (out of Phase 12 scope).
* `prior_session_extreme` **requires** a reliable `trading_day` label column
  (Phase 03/04 calendar); a bare frame with no labels is a loud error — there is
  no gap-segment substitute. The gap-delimited segment behaviour is available
  only under `rolling_nbar_extreme` (which is bar-based, not session-based).
* `distance_unit='ticks'` needs `tick_size` supplied explicitly (from typed
  metadata); Phase 12 does not itself look tick size up from a `ContractSpec`
  path — a caller wanting tick-based geometry passes the causal tick metadata.
* Synthetic-fixture geometry is hand-tuned to *exercise* each branch; it is **not**
  evidence about real markets.
* Intrabar ordering is never inferred — where a sequence would require it, the
  detector requires bar separation instead.

---

## 19. What Phase 13 receives

* `make_silver_bullet_spec` + `SILVER_BULLET_BENCHMARK` + `SILVER_BULLET_GRID_RANGES`
  + `SILVER_BULLET_FAMILY` (typed, `is_claimed_alpha = False`);
* the `silver_bullet_intent` / `silver_bullet_state` registered feature kinds and
  the `SilverBulletAudit` / diagnostics records;
* an end-to-end C++-official backtest path identical to the baselines, with the
  deterministic `strategy_fingerprint` and `schedule_hash` for the experiment
  registry / multiple-testing-aware validation.

Phase 13 decides whether the Silver Bullet hypothesis has any statistical
evidence. Phase 12 asserts only that it is now a precise, causal, reproducible
definition.
