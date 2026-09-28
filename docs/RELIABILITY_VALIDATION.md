# Reliability-Aware Strategy Validation Framework (Phase 13)

Phase 13 answers **"does this strategy have evidence that survives realistic
validation?"** -- not merely "did the historical backtest make money?". It
evaluates Phase 11 baselines, the Phase 12 Silver Bullet, and any future
`StrategySpec` through the **same** official C++ execution / Fill / Portfolio
path, and returns one of:

```
PASS          REJECT          INCONCLUSIVE
```

**REJECT is a normal, successful system outcome.** The verdict is code-based and
typed (`Verdict` + `ReasonCode`); no LLM may override it.

Package: `python/alpha_agent/validation/`. C++ change: one additive export
(`BacktestResult::daily_equity`), nothing else.

---

## 1. Official performance source

Official economics stay C++-sourced. The framework never builds a second Python
PnL engine and never reconstructs fills, positions, commissions, execution
prices or equity.

`BacktestResult` previously exposed only `portfolio_at_end` -- a single final
snapshot. Phase 13 adds the **smallest additive typed export**:

```cpp
struct DailyEquityPoint {           // one per validation-day boundary (13.1) or UTC bucket
    std::int64_t session_day_index; // boundary ordinal, or UTC day index (legacy)
    std::int64_t ts_ns;            // engine event this snapshot was sampled at
    double equity_usd;             // starting_capital + net_realized + unrealized
    double net_realized_pnl_usd;   // Fill-derived
    double unrealized_pnl_usd;     // mark-to-market
    double costs_usd;              // cumulative commissions
    std::size_t fills_cumulative, bars_cumulative;
};
std::vector<DailyEquityPoint> BacktestResult::daily_equity;
std::vector<std::int64_t>     EngineConfig::validation_day_boundaries_ns;  // 13.1
```

Each point is the equity value `PortfolioAccountant::snapshot()` **already**
computes. **Phase 13.1** makes the day bucket the canonical CME futures
`trading_day`, not the UTC date: Python derives one boundary timestamp per
observed `trading_day` (that day's last eligible event, from the
`SessionCalendar`) and passes them to `EngineConfig::validation_day_boundaries_ns`
(CLI: an 8th positional `validation_days.csv` with header `boundary_ts_ns`); the
engine emits one snapshot per boundary, at the first event that reaches it. With
no boundaries it falls back to legacy UTC-midnight buckets, tagged
`daily_equity_basis: "utc_day"` -- **diagnostics only, never canonical for CME
futures** (the framework always passes boundaries). The final day's point is
re-sampled after end-of-test liquidation. **No accounting formula changed**
(`cpp/tests/test_validation_trace.cpp` pins both bases: one point per
day/boundary, final point == `portfolio_at_end`, differencing reproduces total
PnL, a flat day still emits a point, one boundary spanning a UTC midnight is one
observation, the two bases allocate the same total PnL to a different number of
day observations, deterministic replay).

Python (`validation/returns.py`) **aggregates** these official snapshots -- it
does no PnL arithmetic of its own.

## 2. ValidationSpec (section 4)

`validation/spec.py` -- frozen, `extra="forbid"`, no callables / expressions.
Fields: `schema_version`, `framework_version`, `strategy_fingerprint`,
`strategy_key`, `dataset` (`DatasetIdentity`), `split_plan`, `walk_forward`,
`capital_base_usd`, `cost_stress`, `null_test`, `bootstrap`, `trial_family_id`,
`parameter_neighbourhood`, `minimum_sample`, `reliability_policy_fingerprint`,
`random_seed`. `validation_fingerprint()` folds **every** semantic choice
(each sub-config's own `identity()` / fingerprint) and nothing cosmetic
(`label`).

## 3. Validation return series (sections 2 & 3, hardened in 13.1)

Canonical series -- one observation per **CME exchange `trading_day`**
(`validation/trading_day.py` + `validation/returns.py`):

```
for each observed trading_day D:
  EOD_equity[D]        = official C++ portfolio equity at D's last eligible event
  daily_pnl[D]         = EOD_equity[D] - previous official EOD equity
                         (first day of an independently flat-started fold:
                          EOD_equity[D] - explicit initial_equity)
  validation_return[D] = daily_pnl[D] / validation_capital_base   (explicit, finite, > 0, fixed)
```

`ValidationDayPlan` (`build_validation_day_plan`) groups the exec-window bars by
`SessionCalendar.classify` `trading_day` (17:00 America/Chicago boundary, DST-safe
via `zoneinfo` -- no hard-coded UTC offset) and records each day's boundary
event; the C++ engine emits its equity at exactly those instants. One trading day
whose bars straddle a UTC midnight is **one** observation; no UTC-midnight
pseudo-day is ever inserted. A `trading_day` with no engine event produces no
observation -- a missing day never silently becomes zero. A day observed with no
economic PnL is a real `0.0`, kept. Every daily observation carries `trading_day`,
the EOD ts, official equity, daily PnL and validation return; `day_basis ==
"trading_day"` and `day_plan_identity` name the convention. The trading-day
convention (calendar name + version + timezone + local boundary) is in
`DatasetIdentity` -> `validation_fingerprint()`, so two day conventions never
share a `ValidationSpec` identity. Sharpe (`sqrt(252)` only), bootstrap, DSR,
minimum-day gates and regime summaries all consume this canonical series;
`concat_oos_series` appends per-fold series for the walk-forward combined view.
The legacy UTC-day trace remains available (`day_basis == "utc_day"`) for
diagnostics and never feeds the `ReliabilityPolicy`.

## 4. Chronological split / holdout (sections 5, 20)

`validation/splits.py` -- `SplitPlan` of contiguous `SplitWindow`s with roles
`TRAIN` / `VALIDATION` / `LOCKED_HOLDOUT` (closed enum). Validators enforce:
no overlap, strictly chronological, **exactly one** holdout and it is last, and
`holdout.start - research.end >= embargo_days` (embargo, section 7). No random
split. There is **no** `select_best(holdout_results)` API.

`validation/holdout.py` -- two-step lifecycle. `freeze_research(spec, policy,
research_report)` -> `FrozenResearchBundle` (strategy / spec / policy / split /
dataset / trial-family fingerprints + research verdict). `HoldoutRelease(bundle)`
is single-use: `evaluate_once(engine)` verifies every frozen fingerprint still
matches (any change -> `HoldoutDisciplineError`: a new hypothesis needs a new
future holdout) and refuses a second call.

## 5. Walk-forward (section 6) + boundary safety (section 7)

`validation/walkforward.py` -- causal `train_i -> test_i` folds inside the
research span (never the holdout). A **predeclared fixed** `StrategySpec` is
evaluated across folds; no parameter optimisation inside a fold. Each OOS fold:
features + schedule are computed over `[warmup_start, test_end)` (feature warm-up
only), the strategy emits no decision before `test_start`, and the C++ engine
executes over `[test_start, test_end)` -- so **no warm-up bar carries an economic
position or a return observation**, and no future bar closes a training trade
that is then counted as test performance. `embargo_days` separates train from
test. `PortfolioStatePolicy.FLAT_START` (documented default) -- no position / PnL
inherited from training.

## 6. Minimum-sample gates (section 8)

`MinimumSampleRequirements` (predeclared, in the policy): `min_oos_trading_days`,
`min_fills`, `min_trades`, `min_walk_forward_folds`, `min_nonzero_return_days`.
Not met -> **INCONCLUSIVE** (never PASS, never automatic REJECT).

## 7. OOS metrics (section 9)

`validation/metrics.py`, all from the daily series: mean daily return, daily
volatility, daily & **annualised** Sharpe (`* sqrt(252)` -- the series is daily,
the only annualisation constant in the framework; no `sqrt(252*390)`), Sortino,
downside deviation, max-drawdown of the cumulative-return path, skewness,
(non-excess) kurtosis, OOS gross / costs / net PnL, trading days, fills, trades,
positive-fold count.

## 8. Block / stationary bootstrap (section 10)

`validation/bootstrap.py` -- deterministic (`numpy.random.default_rng(seed)`),
circular. `BootstrapMethod.MOVING_BLOCK` (fixed length) or `STATIONARY`
(Politis-Romano geometric block length). Naive IID is **not** a main-path option.
Returns percentile confidence intervals for the **mean daily return** and the
**annualised Sharpe**. Block length is a typed config, never tuned to taste.

## 9. Null hypothesis test (section 11; 13.1 boundaries, 13.2 roles + common support)

`validation/nulls.py`. Each `NullTestResult` carries a `NullRole`:

**`NullMethod.CENTERED_BLOCK_BOOTSTRAP` -- `NullRole.GATING`.** Block bootstrap of
the mean-removed daily returns (dependence kept, edge removed); always
`n_null_samples` replicates. **This is the only null the `ReliabilityPolicy`
gates on** (`policy._null_gate_p` considers `role == GATING` only, taking the max
p-value over the adequately-powered gating nulls). Observed statistic = the
full-sample canonical OOS statistic.

**`NullMethod.SCHEDULE_TIME_SHIFT` -- `NullRole.DIAGNOSTIC` (never gates,
Phase 13.2 section 4).** It uses a proper per-segment **common support** so its
p-value is still statistically comparable:

* A causal segment (`causal_segment_ids`) is a maximal bar run with one root, one
  contract (no roll), one `trading_day`, one `session`, no data gap.
* A **fixed predeclared** family max forward shift `K = max_shift_bars` and
  latency `L = execution_latency_bars` (0 on the frozen CLI). The
  **common-support control schedule** (`common_support_control_schedule`) keeps a
  target row at segment position `p` only if `p + K + 1 + L <= segment_end` -- so
  even shifted by the largest `k <= K` its next-bar(+latency) execution still
  lands in the segment.
* Every replicate shifts the **same** control rows forward by `k` in
  `[min_shift_bars, K]`. By construction none overflows, none is dropped, none
  wraps -- the control and all replicates share **one opportunity set**; no
  replicate can cherry-pick a favourable support.
* The observed statistic for the p-value is the **common-support control run's**
  statistic (`null_test_observed_statistic` on `NullTestResult.observed_stat`),
  **never** the full-sample canonical statistic
  (`NullTestResult.canonical_oos_statistic`, recorded for contrast only --
  section 3).
* On one-bar-per-`trading_day` data each bar is its own segment and the control
  is empty -> the schedule-shift null is simply not run.

**`NullMethod.CIRCULAR_SCHEDULE_PERMUTATION` -- `NullRole.DIAGNOSTIC`, research
only, never default.** A circular rotation that **does** wrap late rows to earlier
timestamps -- an explicitly named permutation null, never conflated with the
causal shift.

Every schedule null reruns the full C++ path (`targets ->
ScheduledTargetStrategy -> Risk -> ExecutionSimulator -> Fill ->
PortfolioAccountant -> official validation equity`); no Python PnL shortcut.
One-sided upper-tail empirical p-value with the finite-sample correction

```
p = (1 + #{ null_stat >= observed_stat }) / (1 + n_null)     -- never 0
```

## 10. Multiple testing / BH-FDR (section 12)

`validation/multiple_testing.py` + `validation/fdr.py`. `MultipleTestingFamily`
records `family_id`, every `TrialRecord` (canonical + every neighbour + every
ablation + every cross-market root), tested fingerprints, and each trial's
p-value. `family_fingerprint()` covers `(role, strategy_fingerprint,
schedule_hash)` sorted -- not cosmetic labels. `benjamini_hochberg_decisions`
returns deterministic q-values (lexsort `(p, index)` ties, monotone step-up) and
rejection decisions. **Failed trials stay in the denominator** -- a trial that
was inspected but never null-tested gets `p = 1.0`.

## 11. Deflated Sharpe Ratio (section 13; 13.2 frequency audit)

`validation/dsr.py` -- Bailey & Lopez de Prado (2014). Exact equations:

```
PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(N - 1)
                / sqrt( 1 - g3*SR_hat + (g4 - 1)/4 * SR_hat^2 ) )

DSR   = PSR(SR*_0)
SR*_0 = sqrt(Var[{SR_n}]) * ( (1 - gamma) * Z^-1(1 - 1/T) + gamma * Z^-1(1 - 1/(T e)) )
```

**Frequency consistency (Phase 13.2 sections 7-8).** Every quantity in the
inference is on the **same per-observation frequency -- DAILY**: `SR_hat =
mean(daily_returns)/std(daily_returns)`, `N` = daily observation count, `g3`/`g4`
= skewness / (non-excess) kurtosis of those same daily returns, and `SR*_0` is a
**daily-scale** benchmark built from `Var[{SR_n}]` over the trials' **daily**
Sharpe estimates. So the observed Sharpe and the expected-maximum benchmark
entering the DSR probability are on the same scale. The annualised Sharpe
`SR_hat * sqrt(252)` is `observed_annualized_sharpe`, a **reporting metric only**
-- it never enters any formula, and `DeflatedSharpeResult.inference_frequency ==
"daily"` records this. Edge cases: N<2 / zero variance / denominator <= 0 ->
`is_valid = False`; `T<=1` or zero trial variance -> `SR*_0 = 0`. DSR is a
probability in `[0,1]`; it does not, alone, prove alpha. Tests: hand-checkable
PSR; **annualised rescaling does not change the DSR probability**; internal
z-statistic reconstructed from daily SR/N/skew/kurt; more trials lower
reliability; stronger daily signal raises DSR; null-like daily signal stays weak;
heavy-tailed (Student-t) edges finite/typed.

## 12. Parameter stability (section 14)

`validation/stability.py` -- `ParameterNeighbourhood` (canonical + **predeclared**
neighbour param sets, never grown after results). Each neighbour is its own
`StrategySpec` fingerprint and its own trial. Evidence: fraction of neighbours
with positive OOS net / positive OOS Sharpe, Sharpe & net dispersion, canonical
rank / percentile, and `canonical_is_isolated_spike` (documented heuristic; the
PASS/REJECT threshold is in the policy). The best neighbour is never adopted.

## 13. Cost stress (section 15)

`validation/cost_stress.py` -- `CostStressPlan` of predeclared `CostScenario`s
(multiplier or absolute), must include a `1.0x` baseline. **Each scenario reruns
the C++ engine** with scaled `[commission] [slippage_ticks] [spread_ticks]` CLI
args -- never a Python post-hoc subtraction. All non-cost semantics held
identical. `CostStressReport.max_net_pnl_degradation` is the worst fractional
net-PnL loss vs baseline across stressed scenarios.

## 14. Ablations (section 16)

`validation/ablation.py` -- predeclared typed `AblationSpec`s (kind +
`param_overrides`). Each produces a distinct `StrategySpec` fingerprint (never a
dynamic mutation of the canonical strategy) and counts as a trial. `AblationReport`
reports whether the canonical Sharpe beats every single-component ablation
(`mechanism_adds_value`). The best ablation is not adopted in Phase 13.

## 15. Regime & cross-market (sections 17, 18)

`validation/regime.py` -- predefined **causal** regime labels
(`causal_volatility_regime_labels` buckets by tertile cut points estimated only
from train data; callers may supply causal trend/session labels). Reports PnL
concentration by regime; no labels -> `NOT_EVALUATED`. Regimes are never defined
after seeing which partition flatters the strategy.

`validation/crossmarket.py` -- the same family hypothesis applied **separately**
per root (ES/NQ/CL/GC/ZN); one `StrategySpec` stays one-root scoped, raw prices
never pooled. `CrossMarketEvidence` summarises independent per-root runs and
whether performance is concentrated in one root. Not every market must be
profitable; there is no universal "N of M" rule unless the policy states one.

## 16. ReliabilityPolicy + verdict (sections 8, 19)

`validation/policy.py` -- frozen `ReliabilityPolicy`, fixed **before** any
validation / holdout outcome (`configs/validation.yaml` mirrors the defaults).
Gates: minimum sample, null p-value max (**GATING null only** -- the centered
block bootstrap; the schedule time-shift null is diagnostic and never gates,
Phase 13.2), BH q-threshold + canonical-rejected, DSR min, max cost degradation,
parameter-stability fraction + isolated-spike, fold-consistency min, positive OOS
net PnL, optional regime / cross-market / ablation. `identity()` excludes the
cosmetic `policy_name`.

`evaluate_policy(...)` order:

```
1. minimum evidence not met            -> INCONCLUSIVE  (INSUFFICIENT_* codes)
2. any clear statistical-failure gate   -> REJECT        (NULL_NOT_REJECTED, FDR_NOT_SIGNIFICANT,
                                                          DSR_BELOW_THRESHOLD, NEGATIVE_OOS_NET_PNL,
                                                          FOLD_CONSISTENCY_FAILED, COST_STRESS_FAILED,
                                                          PARAMETER_UNSTABLE, REGIME_CONCENTRATED,
                                                          CROSS_MARKET_CONCENTRATED)
3. a required evidence family not run   -> INCONCLUSIVE  (REGIME/CROSS_MARKET_NOT_EVALUATED)
4. all required gates satisfied         -> PASS          (ALL_GATES_SATISFIED)
```

## 17. ValidationReport (section 22)

`validation/report.py` -- frozen. Identity block (strategy / validation / dataset
/ split / policy / trial-family fingerprints, target schedule hash, holdout
flag), split windows, fold summaries + `WalkForwardResult`, OOS daily series +
`OosMetrics`, `BootstrapResult`, null results, `FdrResult` + canonical index,
`DeflatedSharpeResult`, cost / parameter-stability / ablation / regime /
cross-market results, `SampleDiagnostics`, `verdict` + `reason_codes` +
`gate_results`. `explanation` is explicitly non-semantic. `report_fingerprint()`
is deterministic (non-finite statistics hash as a stable sentinel).

## 18. Trial identity / reproducibility (section 21)

Deterministic `<prefix>:sha256(canonical_json)` fingerprints for every semantic
object: `ValidationSpec`, `SplitPlan`, `WalkForwardConfig`, `BootstrapConfig`,
`NullTestConfig`, `CostStressPlan`, `ParameterNeighbourhood`,
`MinimumSampleRequirements`, `ReliabilityPolicy`, `DatasetIdentity`,
`MultipleTestingFamily`, `ValidationReport`, `FrozenResearchBundle`. Cosmetic
names / timestamps are excluded. Same inputs -> byte-identical report fingerprint
(`test_AG`, `test_e2e_deterministic_replay`).

## 19. Orchestration

`ValidationEngine` (`validation/engine.py`) drives the real path: the headline
OOS run, `k` walk-forward folds, the cost-stress reruns, the parameter
neighbourhood, the ablations, the schedule time-shift null -- each a C++ backtest
via `CliBacktestRunner`. Feature computation + DSL compilation + target-schedule
construction is delegated to a `StrategyFamilyAdapter` (`DslFamilyAdapter` works
for any Phase 10/11/12 DSL strategy). `assemble_report(...)` is the single pure
function turning per-run economic evidence into statistics + verdict; the
section-23 fixtures feed it synthetic runs.

## 20. Synthetic statistical fixtures (section 23)

`tests/python/validation_fixtures.py` -- **software / statistical tests only, not
market results**: A zero-alpha, B strong signal, C unstable spike, D broad
plateau, E cost-fragile, F too-small sample, G regime-concentrated. A null
synthetic strategy does not reliably PASS (`test_null_synthetic_strategy_does_not
_reliably_pass`, 0/8 seeds); a deliberately strong signal PASSes under an
explicitly configured test policy (`test_AC`).

## 21. Tests / results

* C++ `cpp/tests/test_validation_trace.cpp` (`quant_validation_trace_tests`) --
  6 checks A-F on the additive daily-equity trace. **10/10 ctest targets pass.**
* Python `tests/python/test_reliability_validation.py` -- **45 tests**, checklist
  A-AH: chronological-split-only + no random-split API; embargo enforced; folds
  causal with embargo & no future leak; validation series from the C++ trace
  only; daily aggregation + capital base; `sqrt(252)` annualisation, no intraday
  factor; deterministic moving-block / stationary bootstrap that preserves block
  structure; deterministic minimum-shift null schedules; finite-correction
  p-value never 0; BH known example + deterministic ties; failed trials kept in
  the denominator; DSR hand-check + trial-count monotonicity + edge cases;
  isolated spike flagged / plateau stable / neighbours are trials; cost-fragile
  synthetic strategy degrades and REJECTs; ablations have separate fingerprints;
  regime labels causal + concentration detected; cross-market runs independent by
  root; insufficient sample -> INCONCLUSIVE; zero-alpha -> REJECT; strong signal
  -> PASS; policy fingerprint stable under a cosmetic name; policy change moves
  validation identity; holdout result cannot mutate the policy and is single-use;
  a strategy change after freeze needs a new holdout; deterministic replay of the
  report; end-to-end research + holdout on the real C++ path; cost stress reruns
  C++ (2x commission -> exactly 2x official costs), not Python subtraction.
* **Phase 13.1** adds tests: `build_validation_day_plan` groups bars by
  `trading_day`; one CME session spanning a UTC midnight is one observation; DST
  does not split/duplicate a `trading_day` (local boundary shifts one UTC hour);
  first-fold-day PnL uses the explicit initial equity; the trading-day convention
  is in the validation fingerprint; the default null never wraps future rows to
  earlier timestamps and never crosses a causal segment; the common-support
  policy is deterministic and symmetric; the circular permutation is a separate
  named method; the CLI trading-day trace reruns C++, follows `trading_day` (not
  UTC), and allocates the same total PnL to a different number of day
  observations than the UTC-bucket method; the engine null stays within the fold
  and never touches the holdout.
* **Phase 13.2** adds tests: per-segment common support is symmetric and
  comparable (the control and every shifted replicate share one source-row set,
  no overflow, no wrap); no late-segment overflow target enters the comparison;
  the schedule-shift null's observed statistic is the common-support control's,
  distinct from the recorded full-sample canonical statistic; the schedule-shift
  null is `NullRole.DIAGNOSTIC` and `_null_gate_p` ignores it; DSR annualised
  rescaling does not change the probability, its z-statistic reconstructs from
  daily SR/N/skew/kurt, more trials lower reliability, stronger daily signal
  raises DSR, null-like stays weak, Student-t edges stay finite/typed; a C++
  hand-check that a Fill and a mark on a trading_day's boundary bar land in
  **that** day's EOD equity (differencing attributes the effect to D, not D+1).
* Full suite: **Python 420 passed** (Phase 13: +45; 13.1: +11; 13.2: +7 net),
  **C++ 10/10 ctest**. All Phase 09-13.1 tests remain green. `ruff` clean.

## 22. C++ change

Phase 13: `BacktestResult::daily_equity` (additive typed export of existing
accounting snapshots) + its CLI emission + three optional cost-override CLI args.
Phase 13.1: `EngineConfig::validation_day_boundaries_ns` (additive; empty =
unchanged legacy behaviour) + the 8th optional CLI arg `validation_days.csv` +
the `daily_equity_basis` JSON field. **Phase 13.2: no C++ change** -- an audit
confirmed the boundary snapshot is already taken after all economic processing
for that engine event (pending-intent execution -> RiskDecision -> Fill ->
`PortfolioAccountant` update -> mark-to-close), before the next-bar decision; a
regression test pins it. **No** Fill mechanics, `ExecutionSimulator`,
`ActiveContractResolver`, `RiskManager`, `PortfolioAccountant` formula, roll /
latency / strategy semantics changed. Every strategy/test compiled unchanged; the
frozen reference path (no boundaries, reference costs) is byte-identical.

## 23. Data cost

**No Databento data downloaded.** Built and proven entirely on deterministic
synthetic fixtures (daily-bar fixtures for the real C++ path, PnL-vector fixtures
for the statistics). No `metadata.get_cost()` call was required. Meaningful
real-data validation of a specific hypothesis would need a cost-approved
historical slice (dataset `GLBX.MDP3`, the strategy's root, the split date range,
`ohlcv-1d` or `ohlcv-1m`) -- report the estimate via `metadata.get_cost()` and
**stop before downloading**.

## 24. Limitations

* The walk-forward evaluates a **predeclared fixed** spec across folds;
  in-fold parameter selection is out of scope (section 6).
* The canonical daily bucket is the CME `trading_day` from the config
  `SessionCalendar` (13.1); holidays / early closes are architecturally supported
  but the Phase-04 config ships empty tables. Python derives the boundaries and
  the C++ core stays tz-database-free.
* The schedule time-shift null is **diagnostic-only** (never gates PASS/REJECT,
  Phase 13.2 section 4): with a proper per-segment common support it is
  statistically comparable, but on one-bar-per-`trading_day` data its control is
  empty and it does not run. The centered block bootstrap is the sole gating
  null. The circular permutation null is research-only, never default.
* `NullTestConfig.execution_latency_bars` in the common-support rule must equal
  the engine's `EngineConfig::latency_bars` (0 on the frozen reference CLI).
* Cross-market and regime evidence are consumed as caller-supplied per-root runs
  / causal label arrays; the engine does not itself fan out across roots.
* Real market claims are out of scope -- Phase 13 proves the *framework*.

## 25. What Phase 14 receives

* `ValidationReport` + `report_fingerprint()` -- a frozen, typed, reproducible
  record per (strategy, validation spec, policy) with a code-based verdict.
* `MultipleTestingFamily` / `family_fingerprint()` + every `TrialRecord` -- the
  full inspected-variant ledger for the experiment registry.
* `FrozenResearchBundle` / `HoldoutEvaluation` -- the freeze -> single-holdout
  lifecycle records.
* every semantic fingerprint (spec / split / policy / dataset / neighbourhood /
  bootstrap / null) for near-duplicate detection and failure memory.
* `REJECT` and `INCONCLUSIVE` reports are **first-class data** and must not be
  deleted.

Phase 14 (experiment registry + failure memory) is **not** started here.
