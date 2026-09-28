# Futures Feature Engine (Phase 09 / 09.1)

The deterministic, point-in-time Python feature layer. It transforms **approved
historical market data** (layers 2/3/4) into point-in-time features for baseline
strategies, later ML, reliability validation and the future Research Agent.

It does **not** generate Orders, compute official PnL, or execute strategies.
The C++ Quant Core is unchanged by this phase.

```
FeatureSpec            typed request, no code
    |
FeatureRegistry        kind -> registered FeatureDef (Python source only)
    |
FeatureComputer        SourceSeries + [FeatureSpec] -> FeatureFrame
    |
FeatureFrame           identifiers | features | mask | availability | lineage | safety | qa
```

Module map (`python/alpha_agent/features/`): `enums`, `safety`, `source`, `spec`,
`qa`, `windows`, `registry`, `frame`, `compute`; feature families `returns`,
`trend`, `reversion`, `volatility`, `volume`, `roll_features`; standalone typed
interfaces `carry`, `macro`, `crossmarket`.

**Phase 09.1** replaced the single `research_only` flag with explicit safety
concepts and propagated the Phase 04 back-adjustment mode. **Phase 09.2**
separates SOURCE price safety from DERIVED-FEATURE safety: a feature value is a
transform, never an execution price, so `FeatureSafety.execution_price_safe` is
always `False` even for a `RawContract` source (see section 1). No feature
formula, source price domain, back-adjustment mode, window, session/gap,
signed-price, carry, macro or FeatureFrame-row behaviour changed.

---

## 1. Price domains, SOURCE safety, and DERIVED-FEATURE safety (Phase 09.1 / 09.2)

Every feature declares the price domains it may consume
(`FeatureSpec.required_price_domain`, defaulting to the `FeatureDef` default).
`FeatureComputer` raises `FeaturePriceDomainError` if the source domain is not
permitted. Feature *columns* never carry a fillable contract identity
(`assert_no_execution_identity_leak`); `instrument_id` / `raw_symbol` stay as
**identifiers** only.

The single `research_only` flag is gone. `alpha_agent.features.safety` separates
two levels:

### SourceSafety -- what a *price series* confers

`SourceSeries.safety() -> SourceSafety`, also on `FeatureFrame.source_safety` and
in the lineage. Here `execution_price_safe` is a genuine property.

| source | point_in_time | signal_safe | execution_price_safe |
|--------|---------------|-------------|----------------------|
| `RawContract` (causal) | yes | yes | **yes** |
| `RawContinuous` (causal) | yes | yes | no (discontinuous at rolls) |
| `BackAdjusted`, `adjustment_mode="point_in_time"` | yes | yes | no (never traded) |
| `BackAdjusted`, `adjustment_mode="retrospective_research"` | no | no | no |

A `back_adjusted` `SourceSeries` **must** declare its Phase 04 `adjustment_mode`
(explicitly or via an `adjustment_mode` frame column) -- it is never assumed.

### FeatureSafety -- what a *computed feature column* confers

Per feature on `FeatureFrame.safety`, aggregated by `FeatureFrame.frame_safety`.

* **`point_in_time_safe`** = `source.point_in_time_safe AND feature.point_in_time_safe`
  (a retrospective roll feature or a retrospective back-adjusted source makes it
  `False`).
* **`signal_safe`** = `source.signal_safe AND feature.point_in_time_safe` -- the
  value may drive a strategy `Signal` (Phase 10 DSL contract).
* **`execution_price_safe`** = **always `False`.** A moving average, a return, an
  ATR, a z-score, a rolling high, a breakout distance, a carry number -- every
  feature value is a *transform*, not a price. It does **not** inherit the
  source's execution-price safety. Features drive `Signal`s, not `Fill`s. No
  `FeatureFrame` the `FeatureRegistry` can produce is execution-price safe.

`FeatureMetadata.execution_price_safe` is likewise always `False`;
`point_in_time_safe` / `signal_safe` from `REGISTRY.metadata_for(spec)` are the
feature's own contribution, on a `FeatureFrame` the effective combined values.

### Assertions (`FeatureFrame`)

* `assert_signal_safe()` -- **Phase 10's DSL compiler calls this.** A point-in-time
  back-adjusted trend/momentum feature passes; a retrospective back-adjusted
  source or a retrospective roll feature raises `FeatureSafetyError`.
* `assert_point_in_time_safe()` -- the raw causal check.
* `assert_execution_price_safe()` -- **always raises `FeatureSafetyError`.** It
  documents the invariant: no feature value is ever an execution reference / Fill
  / slippage / contract execution price. Execution prices are selected only by
  the deterministic C++ path from `RawContract` market data (BOUNDARY_CONTRACT E;
  `make_fill` is the final guard). For whether the *source feed* was a real
  contract, read `FeatureFrame.source_safety.execution_price_safe`.

### Feature → Signal vs RawContract → Fill

```
FeatureFrame  ->  Strategy  ->  Signal            (research / signal plane)
RawContract MarketEvent  ->  ExecutionSimulator  ->  Fill   (execution plane)
```

The `ExecutionSimulator` never consumes a `FeatureFrame` numerical feature as an
execution reference; the C++ `make_fill` is the independent final guard
(rejects `BackAdjusted` / `RawContinuous`, tick-aligns to the `ContractSpec`).

---

## 2. No look-ahead

* The source is truncated to `as_of_ts_ns` **before** any computation
  (`SourceSeries.truncate_as_of`).
* Every window primitive in `windows.py` is backward-looking: pandas `rolling`
  at the right edge, `shift(n>0)`, EWM recursing from the segment start. No
  `center=True`, no full-sample `expanding`, no full-sample mean/std
  normalization (z-score and volatility-percentile use trailing stats only).
* Rolling calculations run **within a segment** (`SourceSeries.segment_ids`), so
  a window never reaches across a session boundary or a data gap.
* **Guarantee (tested for 22 features):** the value at timestamp `T` is
  bit-identical whether the series ends at `T` or continues far past it
  (`test_features_lookahead.py`). This is also deterministic replay (checklist O).
* A **retrospective_research** back-adjusted source applies every roll gap,
  including rolls after `T`; the value at `T` then depends on the future. The
  regression test (`test_features_safety.py`) proves a retrospective back-adjusted
  MA at a pre-roll bar is shifted by the *future* roll basis while the
  point-in-time back-adjusted MA equals the unadjusted feed.

Retrospective (look-ahead) roll features (`bars_until_next_roll`,
`contract_transition_indicator`) are `point_in_time_safe=False`. The computer
raises `LookaheadUnsafeError` -- for a retrospective feature **or** a
retrospective back-adjusted source -- unless `require_point_in_time=False` is
passed for explicit offline research; `assert_signal_safe()` then still refuses
the frame.

---

## 3. Sessions & missing data

`SessionPolicy` (per spec; each family has a sensible default):

| policy | window resets on |
|--------|------------------|
| `CONTINUOUS` | never (a window may span a gap; QA still records staleness) |
| `RESET_ON_SESSION` | a new `trading_day` / `session` label |
| `RESET_ON_GAP` | a gap > `max_gap_intervals` * `interval_ns` (default 1.5) |
| `RESET_ON_SESSION_AND_GAP` | both |

Return / volatility families default to `RESET_ON_GAP` so a "1-bar return" is
never silently computed across a multi-hour hole. **A missing bar is never a zero
return** and market prices are **never forward-filled** -- the row after a hole is
an explicit missing value, not a fabricated 0. Rows are preserved 1:1 with the
source (no silent drops).

---

## 4. Signed / zero prices (Phase 08.2)

Normalized futures prices may be positive, zero, or negative (CL, 2020-04-20).

| feature style | behaviour around 0 / negative |
|---------------|-------------------------------|
| `diff_n`, `ma`, `ma_spread`, `atr`, `range_vol`, `reversal`, `dist_from_ma`, calendar spread | pure arithmetic -- valid for any finite signed price |
| `return_n` (percentage) | base `== 0` -> explicit missing + `NON_POSITIVE_RETURN_BASE`; base `< 0` -> suppressed + `NEGATIVE_RETURN_BASE` (use `diff_n`) |
| `log_return_n`, `realized_vol` | defined only where both endpoints `> 0`; else missing + `NON_POSITIVE_LOG_INPUT` |
| any ratio (`zscore`, `norm_dev`, `donchian_pos`, `rel_volume`, carry) | denominator `~0` -> missing + `ZERO_DENOMINATOR`; never `inf` |

`inf` produced anywhere is coerced to missing and recorded (`INF_IN_OUTPUT`).

---

## 5. Feature metadata & naming

`FeatureRegistry.metadata_for(spec) -> FeatureMetadata`: `feature_name`,
`version`, `family`, `kind`, `required_price_domain`, `lookback`,
`minimum_observations`, `session_policy`, `point_in_time_safe`, `signal_safe`,
`execution_price_safe`, `price_field`, `parameters`, `description`.

Canonical name = `<stem>` + params joined by `_` in a fixed per-kind order
(`return_20`, `volatility_60`, `zscore_30`, `ma_spread_20_100`); a non-default
`price_field` appends `_<field>`; a retrospective roll variant appends `_retro`.
The same spec always yields the same name. Two different specs that resolve to the
same name in one request is a hard `FeatureNameCollision`.

---

## 6. Registry & parameter contracts (no code execution)

A `FeatureDef` is registered in Python source via the `@feature(...)` decorator.
A `FeatureSpec` may only reference a registered `kind` and pass **scalar**
parameters. There is no path to inject a callable, an expression, or an import
name.

`FeatureRegistry.validate_spec(spec)` is the full typed parameter-contract check
(**Phase 10's compiler calls this before requesting a feature**):

* unknown `kind` -> `KeyError`; `params` not a dict / non-scalar value -> `ValueError`;
* unknown parameter name -> `ValueError` (lists the allowed names);
* missing required parameter -> `ValueError`;
* wrong type -> `ValueError` (a bool is not an int; `"5"` and `1.5` are not
  windows);
* out of range -> `ValueError` (`ParamRule(int, min=..., max=...)`; windows are
  positive integers, variance windows `>= 2`);
* cross-parameter constraints -> `ValueError` (`require_lt("fast", "slow")` on
  `ma_spread` / `trend_strength`);
* at registration, two defs whose `(stem, param_order)` would collide -> `ValueError`.

Parameters are never silently ignored or coerced into different semantics.

---

## 7. FeatureFrame

Keyed by `ts_event_ns` (+ instrument / root / continuous-symbol identity).
`identifiers` (labels) are kept strictly separate from `features` (float64
columns). Companions: `mask` (bool, True where present), `availability` (first
`ts_event_ns` each feature is present), `metadata`, `specs`, `lineage`,
`safety` (per-feature `FeatureSafety`), `qa`. A value that cannot be computed for
insufficient history is `NaN` with `mask == False` -- never a dropped row.

---

## 8. Futures roll / carry / macro interfaces

**Roll** (`roll_features.py`, source = continuous feed): point-in-time
`is_roll_day`, `bars_since_roll`, `days_since_roll` (observed
`active_instrument_id` transitions only); retrospective `bars_until_next_roll`,
`contract_transition_indicator` (gated, see 2).

**Carry** (`carry.py`): `ContractQuote` / `CurveObservation` / `CarrySnapshot`
typed on **individual raw contracts** -- `assert_carry_domain` refuses
`back_adjusted` / `raw_continuous`. `calendar_spread` (signed-safe),
`annualized_carry` (`None` when the far price `<= 0`), `normalized_carry`,
`carry_frame` (point-in-time: each row uses only its own curve). When the dataset
lacks overlapping curve observations `build_curve_observations` returns what it
has plus a warning -- it never fabricates a contract. `synthetic_curve` is a
test-only helper.

**COT / macro** (`macro.py`): no data is acquired. `PointInTimeDatum` carries
`reference_ts_ns` (period described) **and** `published_ts_ns` (became public).
`align_as_of(feature_ts, series, as_of_ts_ns=None)` gives each `T` the latest
datum with `published_ts_ns <= min(T, as_of)` -- future publication data can
never leak backward. `COTReport` (report vs release date), `MacroRelease`
(actual / consensus / surprise).

**Cross-market** (`crossmarket.py`): `align_backward` = `merge_asof` direction
`backward` only; a future other-market row is never used. MVP kinds: `spread`,
`ratio`, `return_corr`.

---

## 9. QA (`FeatureQAReport`)

Records, never silently "fixes": `NAN_IN_OUTPUT`, `INF_IN_OUTPUT`,
`INSUFFICIENT_LOOKBACK`, `ZERO_DENOMINATOR`, `NON_POSITIVE_LOG_INPUT`,
`NEGATIVE_RETURN_BASE` / `NON_POSITIVE_RETURN_BASE`, `STALE_INPUT`,
`MISSING_BAR_GAP`, `SESSION_BOUNDARY_RESET`, `EXTREME_VALUE` (kept, never
clipped), `PRICE_DOMAIN_MISMATCH`, `LOOKAHEAD_UNSAFE_REQUEST`.

---

## 10. Reproducibility

`FEATURE_ENGINE_VERSION = "0.9.2"`. `SourceSeries.fingerprint()` is a SHA-256 of
the source content (values + columns + dtypes + domain + `adjustment_mode` +
`causal`). `feature_cache_key(source_fingerprint, spec, engine_version,
as_of_ts_ns)` -- same source lineage + same spec + same engine version => same
key and the same values. An optional `MutableMapping` cache may be passed to
`FeatureComputer.compute`.

`FeatureLineage` (on every `FeatureFrame`): engine version, source price domain,
`source_adjustment_mode`, paths / fingerprint / row count, `as_of_ts_ns`,
`source_point_in_time_safe` / `source_signal_safe` / `source_execution_price_safe`,
`code_commit`, `generated_at`, feature names + canonical specs, identity.

---

## What Phase 10 (Strategy DSL) receives

* `FeatureSpec` / `FeatureMetadata` / `FeatureRegistry` -- request features by
  typed spec; `REGISTRY.kinds()` is the closed allow-list and
  `REGISTRY.validate_spec(spec)` the full parameter-contract gate a compiler
  runs first (unknown / missing / mistyped / out-of-range params, `fast<slow`,
  name collisions -- section 6).
* `FeatureFrame` -- point-in-time features + `mask` + `availability` + `lineage`
  for backtest dataset records (CLAUDE backtest rule 5).
* **Compiler rule: `FeatureFrame.assert_signal_safe()` is the gate.** A
  `StrategySpec` / `FeatureRef` may reference **signal-safe feature values**: a
  point-in-time back-adjusted trend/momentum feature qualifies. It may **not**
  reference a retrospective back-adjusted feature or a retrospective future-roll
  feature.
* **A `FeatureRef` may never specify a fill price, execution reference price,
  slippage price or contract execution price.** No feature value is ever an
  execution price (`FeatureSafety.execution_price_safe` is always `False`;
  `assert_execution_price_safe()` always raises). Execution prices are selected
  only by the deterministic C++ execution path from `RawContract` market data;
  `make_fill` is the independent final guard.
* The carry / macro typed interfaces for later Carry / COT strategy families.
