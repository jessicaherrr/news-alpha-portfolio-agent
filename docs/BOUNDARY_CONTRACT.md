# Futures Data / C++ Boundary Contract (frozen in Phase 02.5)

This document freezes the canonical futures data model and the Python -> C++
boundary **before** the full data pipeline (Phase 03/04) and the event engine
(Phase 05-08) are built. Everything downstream must obey it.

Prime directive: **a multi-contract futures backtest must always be able to
resolve every simulated fill to the real tradable contract that was active at
that timestamp** (CLAUDE.md market-data rules 6-8).

---

## 1. Seven-layer separation

Each layer is a distinct representation with its own storage path and its own
type. They are never merged into one table.

| # | Layer | Storage | Represented by | Used for |
|---|-------|---------|----------------|----------|
| 1 | Raw vendor records | `data/raw/databento/<dataset>/<schema>/<request_hash>/` (immutable) + `*.manifest.json` | bytes as returned | provenance, re-derivation |
| 2 | Canonical raw-contract bars | `data/processed/bars/<root>/<raw_symbol>.parquet` + `_lineage.json` | `CanonicalBar` / `quant::MarketBar` + `ContractRegistry` | **the only prices fills may touch** |
| 3 | Unadjusted continuous series | `data/processed/continuous/<continuous_symbol>.parquet` | `ContinuousBar` (Phase 04, `alpha_agent.data.continuous`) | inspection, feature substrate |
| 4 | Research-only back-adjusted series | `data/processed/backadjusted/<continuous_symbol>.parquet` | `BackAdjustedBar` (Phase 04, `alpha_agent.data.backadjust` -- Python only) | **signals / research only, never fills** |
| 5 | Contract metadata | `data/processed/contracts/<root>.parquet` / `contracts.csv` | `ContractSpecModel` / `quant::ContractSpec` | tick, multiplier, life window, fill resolution |
| 6 | Roll mappings | `data/processed/rolls/<continuous_symbol>.parquet` | `RollEvent` (Phase 04, `alpha_agent.data.rolls`) | auditable roll calendar |
| 7 | Execution / fill data | backtest outputs only | `quant::Fill` (+ `Order`, `RiskDecision`) | realised PnL, drift analysis |

Raw (layer 1) is written once and never modified. Layers 2-6 are derived and
regenerable; each processed file carries a `_lineage.json` sidecar recording the
raw manifest SHA-256, dataset version, code commit, generation time, `tz: "UTC"`,
and the price scale applied.

---

## A. Canonical futures bar schema (layer 2)

`CANONICAL_BAR_COLUMNS` (`python/alpha_agent/schemas/market_data.py`):

| column | type | in the bar? | rationale |
|--------|------|-------------|-----------|
| `ts_event_ns` | int64 | **yes** | Source event time, UTC ns since epoch, **never rewritten** (CLAUDE rule 5). Interval **start**. |
| `instrument_id` | uint32 | **yes** | Databento per-dataset id. The join key to layer 5. This is what makes fill resolution always possible. |
| `raw_symbol` | str | **yes (denormalized)** | Functionally determined by `instrument_id`, but carrying it makes bars and fills auditable without a registry round-trip. Cheap on disk. |
| `root_symbol` | str | **yes (denormalized / partition)** | Queries filter by root constantly; stable; also the Parquet partition key. |
| `open/high/low/close` | float64 | **yes** | Actual traded prices of that one contract (`PriceDomain.RAW_CONTRACT`), de-scaled from Databento 1e-9 fixed point. |
| `volume` | int64 | **yes** | Integer contracts traded in the interval. |
| `trading_day` | date | **yes (derived label)** | CME **session** date (the Globex session that opens ~17:00 CT the prior calendar day belongs to the next trading day). A derived column, added alongside `ts_event_ns`, not replacing it. |
| `session` | enum `RTH/ETH/MAINTENANCE/CLOSED` | **yes (derived label)** | Needed for session-aware features and execution rules; cheap; derived from `ts_event_ns` + the session calendar. |
| `is_roll_day` | bool | **NO -> layers 3 & 6** | A raw contract does not "roll"; the *continuous series* rolls. Putting this on a raw-contract bar conflates layers 2 and 3. It lives on `ContinuousBar.is_roll_boundary` and in the `RollEvent` table. |
| `source` | str | **NO -> lineage sidecar** | Constant per dataset today. Becomes a per-row column only if a canonical file is ever built by merging multiple vendors. |
| `dataset_version` | str | **NO -> lineage sidecar + backtest record** | CLAUDE backtest rule 5 requires the *backtest* to record it; storing ~1400 identical strings per contract-day in the bar table is waste. If a processed file is ever built from multiple raw pulls, add a small integer `raw_batch_id` instead. |

The **in-memory C++ `quant::MarketBar`** is a deliberately smaller value type:
`ts_event_ns, instrument_id, open, high, low, close, volume`. `raw_symbol`,
tick size and multiplier are resolved from the `ContractRegistry` via
`instrument_id`, so the bar stays trivially copyable and cache-friendly.

### Layer 3 `ContinuousBar`
`ts_event_ns, continuous_symbol, active_instrument_id, active_raw_symbol,
open/high/low/close (raw prices of the active contract), volume,
is_roll_boundary, price_domain = raw_continuous`. Prices are discontinuous at
rolls -- that is correct and intentional.

### Layer 4 `BackAdjustedBar`
`ts_event_ns, continuous_symbol, open/high/low/close (ADJUSTED), volume,
adjustment_method, cumulative_adjustment, adjusted_through_ts_ns,
price_domain = back_adjusted`. **No `raw_symbol`, no `instrument_id` on the
price row** -- the adjusted price never traded, so nothing fillable may point at
it.

### Layer 6 `RollEvent`
`continuous_symbol, effective_ts_ns, from_instrument_id, to_instrument_id,
from_raw_symbol, to_raw_symbol, rule, from_price, to_price, additive_gap`.

---

## B. `ContractSpec` (layer 5) -- `cpp/include/quant_core/contract.hpp`

Minimum deterministic metadata:

| field | type | notes |
|-------|------|-------|
| `instrument_id` | uint32 | primary key; `0` == invalid |
| `raw_symbol` | string | real exchange symbol, e.g. `NQZ6`; never continuous/parent |
| `root_symbol` | string | e.g. `NQ` |
| `exchange` | string | ISO 10383 MIC, e.g. `XCME` (or `GLBX`) |
| `tick_size` | double > 0 | minimum price increment |
| `multiplier` | double > 0 | contract point value in USD |
| `activation_ns` | int64 > 0 | first tradable instant, UTC ns |
| `expiration_ns` | int64 > activation | nominal expiry instant |
| `first_notice_ns` | optional int64 | physically-delivered products (CL, GC, ZN); drives roll timing |
| `last_trade_ns` | optional int64 | when distinct from `expiration_ns`; `tradable_until_ns()` uses it |

`is_valid()` checks structure only. `is_live_at(ts)` checks
`activation_ns <= ts <= tradable_until_ns()`.

**Rejected from `ContractSpec`:** currency (assume USD for the CME MVP),
margin/initial-margin (risk layer, changes daily, not deterministic contract
metadata), settlement type, price display format, session-calendar id (the
session calendar is keyed by root/exchange in config, not duplicated per
contract).

`ContractRegistry` (`contract_registry.hpp`): `by_instrument_id`,
`find_by_instrument_id`, `find_by_raw_symbol`, and
`latest_live_contract(root, ts_ns)` -- a **metadata query only** (newest
activation among contracts of that root live at `ts_ns`), for diagnostics /
coverage, **never** to route an Order: around a roll several contracts of one
root are simultaneously live, so root + timestamp cannot say which one a
continuous feed is trading. The active contract comes from current market state
via `ActiveContractResolver` (section C). `add()` rejects invalid specs,
continuous symbols, and duplicate ids/symbols.

---

## C. Domain event types (frozen shapes) -- `cpp/include/quant_core/events.hpp`

Frozen in Phase 05; the event loop / execution sim / portfolio / real risk rules
are Phases 06-08. Structural validators: `domain_model.hpp`
(`validate_market_event`, `validate_signal`, `validate_order`,
`validate_risk_decision`). Domain errors: `domain_errors.hpp`
(`DomainError` base + `InvalidMarketEvent`, `InvalidSignal`, `InvalidOrder`,
`ContractResolutionError`, `RiskInvariantError`, `ExecutionDomainError`).

**Event flow:** `MarketEvent -> Strategy -> Signal -> [ActiveContractResolver] ->
Order -> [RiskManager] -> RiskDecision -> [ExecutionSim] -> Fill -> Portfolio
update`.

- **`MarketEvent`** — `type {Bar}` (**Phase 05 implements only `Bar`**;
  `SessionOpen` / `SessionClose` / `Roll` and quote / trade / settlement events
  are reserved for Phase 06+, added together with their typed payload and their
  `validate_market_event` branch so no un-checkable event can pass validation),
  `ts_event_ns` (authoritative UTC ns), `seq` (deterministic ordering
  tie-breaker), `instrument_id` (must resolve through `ContractRegistry`; also
  the current continuous-market state), + Bar payload
  (`open/high/low/close/volume`). Future payloads attach as extra optional
  members or a payload variant without breaking a consumer that reads only
  `{type, ts_event_ns, seq, instrument_id}`.
- **`Signal`** — strategy intent against a **root**: `signal_id` (engine-stamped),
  `ts_decision_ns`, `root_symbol` (bare root, e.g. `"NQ"`), `target_units`
  (**signed integer contract target**, carried in a `double` for a stable shape;
  `validate_signal` rejects a non-integral value within `1e-6` so `0.51`
  contracts can never silently become `1` — MVP invariant B),
  `rationale_code`. `direction()` derives `{Flat, Long, Short}`. **No raw
  contract, no price, no fill/PnL fields.** Signal intent and execution are
  separate layers.
- **`Order`** — engine-produced, contract-resolved: `order_id`, `signal_id`
  (traceability), `ts_created_ns`, `instrument_id`, `raw_symbol` (== registry
  entry), `side` (`Buy/Sell`), `quantity > 0`, `order_type {Market,Limit,Stop}`,
  `optional<double> limit_price` (required for Limit), `optional<double>
  stop_price` (required for Stop), `tif`. Continuous / back-adjusted identities
  are rejected.
- **`Fill`** — deterministic outcome, **always `PriceDomain::RawContract`**:
  `fill_id`, `order_id` (traceability), `instrument_id` (never 0), `raw_symbol`
  (never empty/continuous), `side`, `quantity`, `fill_price` (raw, tick-aligned),
  `tick_size`, `multiplier` (both copied from `ContractSpec`), `commission_usd`,
  `slippage_ticks`. Built only by `make_fill()` (section E).
- **`RiskDecision`** — immutable audit record: `ts_decision_ns`, `order_id`,
  `instrument_id`, `verdict {Approve, Resize, Reject}`, `requested_quantity`,
  `approved_quantity`, `reason_code`. Produced only by a C++ `RiskManager`
  (`risk_manager.hpp`); `Order -> RiskManager -> RiskDecision` is mandatory
  before execution and an LLM can never override it. Invariants: APPROVE
  `approved == requested > 0`; RESIZE `0 < approved < requested`; REJECT
  `approved == 0`.

**Deterministic IDs / ordering** (`ids.hpp`, `domain_model.hpp`): `signal_id`,
`order_id`, `fill_id` are plain monotonic integers from `MonotonicId` — no UUIDs,
clocks, or external services — giving a reproducible `S<n> -> O<n> -> F<n>` chain.
Events sort by `(ts_event_ns, seq)` via `event_before` / `MarketEventOrder`;
`seq` breaks ties for equal timestamps so ordering never depends on
unordered-container iteration.

**Price domains** (`domain_model.hpp`): `MarketEvent`/bar prices and `Order`/
`Fill` prices are normalized raw-contract price units (e.g. NQ ≈ `30000.00`). A
`Signal` carries no price. `multiplier` is USD PnL per `1.0` price-unit move.
Databento fixed-point (1e-9) vendor prices are normalized at the Python boundary
(`alpha_agent.data.price_domain`) and must never reach C++.

**Signed prices (Phase 08.2).** A normalized futures price may be **positive,
zero, or negative** — WTI crude (CL) front-month settled at −$37.63 on
2020-04-20 and traded well below zero intraday. `is_plausible_raw_price` is a
**magnitude** guard only: `finite` (not NaN/±inf) **and** `|price| < 1e9`
(`kMaxPlausibleRawPrice`, the un-normalized fixed-point tripwire). It is used by
`validate_market_event` (per OHLC value), `validate_order` (limit/stop price),
the `ExecutionSimulator` reference-price check, and `make_fill`. OHLC ordering
(`low ≤ open/close ≤ high`) is arithmetic and holds for signed values. **Missing
data is never encoded as a price value** — zero is a real price, not a sentinel;
absent data travels the `UNDEF_PRICE` sentinel / optional / rejection path
upstream. The Python `CanonicalBar` / `ContinuousBar` / `BackAdjustedBar`
validators and the canonicalize QA gate use the same finite+magnitude rule
(`is_normalized_price`, `NORMALIZED_PRICE_ABS_LIMIT`); this is a semantic
validation change only — the frozen column tuples and CSV wire format are
unchanged.

Strong enums: `Side`, `OrderType`, `TimeInForce`, `MarketEventType`,
`PriceDomain`, `SignalDirection`, `RiskVerdict`.

**Active-contract selection** (`contract_selector.hpp`): a `Signal` is root-level
intent; the concrete raw contract is chosen from the **current market state**,
not from activation dates. `MarketState {active_instrument_id, as_of_ts_ns}` is
the engine's view of which real contract the continuous feed is on right now
(Phase 04.5: the roll is the `instrument_id` transition of the ordered feed,
never a contract-month parse). `ActiveContractResolver::resolve(root_symbol,
const MarketState&) -> const ContractSpec&` (throwing `ContractResolutionError`)
takes that instrument_id and validates it against the registry: instrument
exists, `root_symbol` matches (a Signal for `NQ` fails if the feed is on an `ES`
contract), contract live at `as_of_ts_ns`. The MVP
`RegistryActiveContractResolver` does exactly this and no activation-date
inference. `make_order_from_signal(sig, resolver, market_state, order_id,
ts_created_ns)` is the only sanctioned `Signal -> Order` path.

---

## D. Databento mapping design (Phase 03/04)

Databento does **not** support `continuous -> raw_symbol` or
`parent -> raw_symbol` (422 `symbology_invalid_request`). **`instrument_id` is
the authoritative join key.** See `docs/DATABENTO_REALITY.md`.

| need | request | gives |
|------|---------|-------|
| layer 2 bars | `schema=ohlcv-1m`, `stype_in=continuous`, `symbols=["NQ.v.0"]`, **`stype_out=instrument_id`** | each bar carries `instrument_id` (the real contract, changes at rolls). The `symbol` column is the smart-symbol label `NQ.v.0`, not a contract. |
| layer 5 metadata | `schema=definition`, `stype_in=parent`, `symbols=["NQ.FUT"]`, **`stype_out=instrument_id`**, same range | records carry `instrument_id`, `raw_symbol`, `asset` (root), `exchange`, `activation`, `expiration`, `min_price_increment` (tick), multiplier fields, `instrument_class` |
| join | `bar.instrument_id` → definition record | `raw_symbol`, `root_symbol`, `tick_size`, `multiplier`, `activation_ns`, `expiration_ns` |
| layer 6 roll map (optional cross-check) | `symbology.resolve` `continuous → instrument_id` and `instrument_id → raw_symbol` (free) | audit cross-check against observed `instrument_id` changes in the bars; not mandatory |

**Continuous symbols are never tradable contracts.** The canonical `raw_symbol`
always comes from the definition-table / `ContractRegistry` lookup by
`instrument_id` -- never from the OHLCV `symbol` column. A continuous value in
that column is expected and never a rejection cause. A `CanonicalBar` /
`ContractSpecModel` with a continuous `raw_symbol` is still rejected by pydantic,
and `ContractRegistry::add` / `make_fill` reject it in C++. `NQ.v.0` may appear
as a label on OHLCV rows and as `continuous_symbol` in layers 3, 4, 6.

Cost rule unchanged: `metadata.get_cost()` before every request, hard USD cap,
no download without explicit approval.

---

## E. Execution invariant

The C++ backtester must **never** fill against a back-adjusted price or a
synthetic continuous symbol. A fill must reference the actual raw contract, the
actual raw price, and the `tick_size` / `multiplier` valid for that contract at
that timestamp.

Enforced by `quant::make_fill(registry, FillRequest)` -- the **only** sanctioned
way to build a `Fill`. It throws `FillResolutionError` unless:

1. `price_domain == PriceDomain::RawContract` (rejects `BackAdjusted`, `RawContinuous`);
2. `instrument_id != 0` and present in the registry;
3. the registry entry is a real tradable contract (`is_tradable_contract_symbol`);
4. `expected_raw_symbol` (if given) matches the registry;
5. the contract was live at `ts_fill_ns` (`is_live_at`);
6. `quantity > 0`; `price` is a valid normalized value — `finite` and
   `|price| < 1e9` (`is_plausible_raw_price`; **sign unconstrained** — a zero or
   negative execution price is valid, e.g. negative CL, Phase 08.2);
7. `tick_size` / `multiplier` are taken **from the `ContractSpec`**, never the caller;
8. `fill_price` is rounded to the contract tick grid — `std::round(price /
   tick_size) * tick_size`, symmetric about zero (round-half-away-from-zero), so
   a negative price aligns by the **same** rule (`-20.007 / 0.01 → -2001 →
   -20.01`).

Phase 06 replaced the placeholder `backtest.cpp` ad-hoc entry/exit math with the
deterministic event loop in `engine.hpp` (`BacktestEngine`); every simulated fill
is routed through `make_fill`, and official realized PnL derives **only** from
those `Fill` events via the `ContractSpec` tick/multiplier -- never from a bar
close, a config multiplier, or strategy output. The retired `run_backtest` /
`signal_to_position` / `BacktestConfig` are gone.

**Phase 07** inserts a clean `ExecutionSimulator` (`execution_simulator.hpp`,
section J + `docs/EXECUTION_MODEL.md`) between the risk-approved Order and the
ledger: the engine submits Orders, the simulator returns a validated `Fill` (via
`make_fill`) or a deterministic non-fill. It owns the *model* -- market / limit /
stop semantics, slippage & spread in `ContractSpec` ticks, gap-through, the
intrabar stop/target conservative rule, N-bar latency -- and the strategy never
chooses a fill price. The Phase 06 roll close-leg (old contract closed at its
last *observed* bar, a retroactive fill) is replaced by execution-time
raw-contract pricing. **Phase 07.1 causality invariant:** every Fill produced
while processing engine event time `T` has `ts_fill_ns >= T` (`== T` for the
bar-based model) -- the engine throws `ExecutionDomainError` otherwise. The roll
fallback defaults to `RejectDefer` (missing overlap data is not approximated);
`StaleObservedClose` is an explicit opt-in that records a `RollFallbackAudit`. A
held contract that is already non-tradable at `T` with no earlier close is a
**loud failure**, never a manufactured past fill.

---

## F. Look-ahead-safe strategy interface

`Strategy::target_position(index, const std::vector<MarketBar>&)` exposed the
whole vector and allowed `bars[index + k]`. Replaced (Phase 05) with:

```cpp
virtual Signal Strategy::decide(const StrategyContext& ctx) const = 0;
```

`BarHistoryView` (`bar_history.hpp`) holds `{const MarketBar* data, size_t size}`
where `size` is clamped by the engine to `decision_index + 1`. Every accessor
(`at`, `latest`, `ago`) is bounds-checked and throws `std::out_of_range` instead
of reading past the visible range. There is no method that can reach
`data[size]`. The engine builds the view per decision bar; the fill still
happens at `index + 1`.

`StrategyContext` (`strategy_context.hpp`) is the **only** thing a strategy sees:
`decision_ts_ns`, the current `MarketEvent`, the bounded `BarHistoryView`,
`root_symbol`, and a read-only `const ContractSpec*` snapshot of the contract
live at the decision timestamp (may be null). It is non-copyable and holds **no**
registry handle, mutable state, future roll info, or execution/portfolio
internals — so a strategy cannot resolve a future active contract or read past
the decision bar. `decide` returns a root-level `Signal` (section C); the engine
does contract selection, risk review and fill simulation.

---

## G. Boundary serialization (reference CLI)

Python writes a **bundle** into a work directory:

**`bars.csv`** — header exactly:
```
ts_event_ns,instrument_id,open,high,low,close,volume
```
(`BOUNDARY_BAR_COLUMNS`). A subset of the canonical bar; the core resolves
`raw_symbol` / tick / multiplier from the contracts file.

**`contracts.csv`** — header exactly:
```
instrument_id,raw_symbol,root_symbol,exchange,tick_size,multiplier,activation_ns,expiration_ns,first_notice_ns,last_trade_ns
```
(`CONTRACT_COLUMNS`; last two may be empty). Parsed by
`quant::parse_contracts_csv` / `load_contract_registry`.

**Invocation** (frozen):
```
quant_backtest_csv <bars.csv> <contracts.csv> <lookback> <threshold_return>
```
The core loads the registry, verifies **every** bar resolves to a real contract
live at its timestamp (hard error otherwise), then replays the bars through the
`BacktestEngine` (bars are validated, ordered canonically by
`(ts_event_ns, instrument_id)`, then stamped with a global monotonic `seq` --
input / file order is irrelevant; a duplicate `(instrument_id, ts_event_ns)` is a
hard error). Output is one JSON line:
`bars`, `contracts_resolved`, `events`, `signals`, `orders`, `fills`, `rolls`,
`trades` (closed-position events), `gross_pnl_usd` / `costs_usd` / `net_pnl_usd`
(realized, from `Fill`s), `unrealized_pnl_usd`, `max_drawdown_usd`,
`unique_contracts`, `open_positions`, and (Phase 07)
`rolls_priced_contemporaneous`, `rolls_priced_stale`, `rolls_deferred`,
`roll_fallback_audit_records`, `non_fills`, `eot_liquidations`,
`eot_positions_left_open_stale`, `slippage_ticks`, `commission_per_contract_usd`.
Adding keys is backward-compatible; the Python adapter reads the JSON as a dict.
The frozen CLI signature is unchanged -- it runs at a flat commission with no
slippage (a Market order at the execution bar's open); `ExecutionConfig` is
exposed directly at the pybind boundary (Phase 19). The legacy PnL keys are
retained but derived from the engine, not the retired ad-hoc math.

A run manifest JSON (dataset version, raw SHA-256, params, cost assumptions) is
written next to the bundle by Phase 03 and recorded with the experiment
(CLAUDE backtest rule 5).

---

## G.2 Fast boundary (pybind11, Phase 19)

`bindings/pybind_module.cpp` builds an optional Python module `quant_core_py`
(`cmake -DBUILD_PYBIND=ON`, `scripts/build_pybind.sh`). It is the **high-volume**
transport for the scheduled-target engine (the Phase 13.5C matrix, Phase 15B
economics trials, sensitivity sweeps).

**The section-G CLI (`apps/backtest_targets_csv.cpp`) remains the frozen
reference implementation and is unchanged.** The pybind path calls a thin C++
helper, `quant::run_targets_backtest` (`quant_core/targets_run.hpp`), whose
engine wiring **mirrors** the CLI's — the same `BacktestEngine`,
`PassThroughRiskManager`, `RegistryActiveContractResolver`,
`EngineConfig{ ForceLiquidateFinalClose, commission/slippage/spread }`,
contract-resolution hard check and validation-day / roll-close-mark wiring — and
re-uses the frozen boundary parsers verbatim (`load_contract_registry`,
`parse_targets_csv`). The CLI is **not** refactored onto this helper (that would
touch the frozen official C++ execution path); the two are wired independently.

**Semantic equivalence is an enforced invariant, not a structural guarantee.**
The deterministic CLI-vs-pybind parity tests
(`tests/python/test_phase_19_pybind.py`, via `assert_boundary_parity`) run both
transports on the same fixtures and assert every result field agrees;
`cpp/tests/test_targets_run.cpp` additionally pins the helper against a
hand-built `BacktestEngine` run. If the two wiring sites ever diverge, those
tests fail.

**What crosses:** only `bars` — the one high-volume payload — moves as typed
NumPy columns (`ts_event_ns` int64, `instrument_id` int64→uint32, OHLC float64,
`volume` int64), forcecast and copied once into `std::vector<quant::MarketBar>`.
`contracts.csv` and `targets.csv` (small) are still written into the work dir and
read by the same parsers. Validation-day boundaries and roll-close marks cross as
Python lists. The result is a typed `TargetsRunResult` whose read-only properties
mirror the section-G JSON keys 1:1, plus `write_trades_csv` / `write_fills_csv`
(byte-identical to `--trades-out` / `--fills-out`).

**Nothing moves into Python.** Official PnL / fill / risk / contract resolution /
roll pricing / portfolio accounting / `daily_equity` trace authority stays
entirely in the C++ core (CLAUDE architecture boundary 3). No mutable risk
internals are exposed: `RiskConfig`, the `RiskManager` and the
`PortfolioAccountant` are not bound; this path runs `PassThroughRiskManager`
exactly as the frozen CLI does.

**Python adapter:** `alpha_agent.adapters.pybind_bridge.run_targets_backtest_pybind`
is a drop-in for `targets_bridge.run_targets_backtest_cli` (same kwargs minus
`executable`, same result dict). `assert_boundary_parity` is the equivalence
check — counts/strings exact, floats within a tolerance that absorbs only the
CLI's 6-significant-digit `std::ostream` serialisation vs the pybind path's full
`f64`; any residual beyond that fails as a real disagreement.

**Provenance.** `pybind_boundary_provenance()` returns
`{transport, boundary_abi, engine_path, module_file, module_sha256}`. Any
registry evidence produced over this boundary records `module_sha256` — the
artifact that actually produced the economics — the way the CLI path records its
executable / bundle (`source_artifact` / `source_artifact_sha256`, CLAUDE
"experiment registry"). The boundary does **not** carry, weaken or bypass the
`ValidationSpec.validation_fingerprint()` / `ReliabilityPolicy.identity()`
equality checks: those live above the engine call
(`alpha_agent.validation.engine`, `.holdout`) and apply unchanged whichever
transport produced the daily-return series. `BOUNDARY_ABI` (`phase19-targets/1`)
is transport identity only and is bumped when the `run_scheduled_targets`
argument/result surface changes — it is never part of a scientific
`experiment_identity`.

**Not wired into the frozen Phase 15B real path.** `CppEngineRunner`
(`--run-real`, Phase 15B.2, still gated) keeps using the CLI. The pybind boundary
serves offline research, which is evidence, never a BH/FDR trial.

**Tests:** `cpp/tests/test_targets_run.cpp` (`quant_targets_run_tests`) pins the
helper against a hand-built engine run; `tests/python/test_phase_19_pybind.py`
runs the CLI and the pybind transport on the same fixtures and asserts parity
(plain run, cost overrides, validation-day plan, roll-close marks, audit
exports, determinism, error parity). Benchmark: `scripts/phase_19_pybind_benchmark.py`
(~3x on a single-contract daily fixture; the gap is the CSV write + subprocess,
and grows with bar count).

---

## H. Tests

C++ (`cpp/tests/test_domain_model.cpp`, `ctest` target `quant_domain_model_tests`)
— the Phase 05 domain model: MarketEvent validation (bad instrument rejected,
timestamp preserved, non-`Bar` type rejected; NaN / ±inf / un-normalized
fixed-point price rejected either sign; **signed OHLC accepted** — a negative-
price CL-style bar and a bar straddling zero both pass, OHLC ordering still
enforced; Phase 08.2); Order limit/stop price (present + finite + magnitude;
**negative and zero limit/stop prices accepted**);
StrategyContext no-lookahead (peeking strategy throws, no future container
escape); Signal validation (bad id/ts/root rejected; fractional `target_units`
rejected — MVP invariant B); deterministic active-contract selection (current
`MarketState` instrument_id → real `instrument_id`/`raw_symbol`; empty/unknown
state, root mismatch, not-live all → `ContractResolutionError`); real roll
semantics (same `Signal(root="NQ")` → `Order(NQM6)` before the roll,
`Order(NQU6)` after, driven only by the feed's `instrument_id` while both
contracts are live); Order invariants (quantity, continuous symbol, id/symbol
mismatch, bad limit/stop); RiskDecision APPROVE/RESIZE/REJECT invariants +
mandatory
`Order → RiskManager → RiskDecision → apply` path; Fill model (raw contract only,
tick/multiplier from `ContractSpec`, PnL conversion); deterministic event
ordering `(ts_event_ns, seq)` stable under shuffle; the `S1 → O1 → F1`
traceability chain.

C++ (`cpp/tests/test_contract_boundary.cpp`, `ctest` target
`quant_contract_boundary_tests`):

1. two different raw contracts in one continuous history — both resolve, both live;
2. contract lookup by `instrument_id` and by `(root, timestamp)` via `latest_live_contract` (a metadata query, not the active selector), incl. the overlap heuristic and misses;
3. a roll transition — exactly one boundary, both sides resolve, additive gap well-defined, fills map to the correct raw contract;
4. fill rejection — `instrument_id == 0`, unknown id, ts outside contract life, `quantity <= 0`, non-finite / un-normalized fixed-point price (either sign), symbol mismatch; a **normalized signed price fills** — zero and negative both align to the tick grid by the same round-half-away-from-zero rule (Phase 08.2);
5. adjusted/continuous prices cannot enter execution — `BackAdjusted` and `RawContinuous` domains rejected; symbol classification; registry refuses a continuous `raw_symbol`;
6. strategy cannot read a future bar — `at(size)` / `ago(size)` / far-future all throw; a peeking strategy throws; a well-behaved one is unaffected;
7. contracts CSV round-trips; a malformed header is rejected.

C++ (`cpp/tests/test_engine.cpp`, `ctest` target `quant_engine_tests`) — the
Phase 06 event-driven engine: deterministic event ordering + byte-identical
replay; **multi-root same-timestamp ingestion is permutation-invariant** (ES / NQ
/ CL bars at one timestamp -> identical MarketEvent order, `seq`, signals, orders,
fills, PnL, trace ids and `BacktestResult` for every permutation of the same bar
set; a duplicate `(instrument_id, ts_event_ns)` is rejected); no same-bar
execution (a signal on bar *i* fills at bar *i+1*'s open);
strategy look-ahead throws through the loop; target-position delta semantics
(`0→+1`, `+1→+1`, `+1→-1`, `-1→-2`, `-2→0`); execution-time active-contract
selection; real-style roll (NQM6 decision bar → NQU6 execution bar; a held NQM6
position is explicitly closed as NQM6 -- Phase 07: at the execution timestamp,
opt-in `StaleObservedClose` last-observed price flagged `rolls_priced_stale` +
`RollFallbackAudit` -- and re-opened in NQU6, never relabelled); mandatory
`RiskDecision` gate (RESIZE honoured before the fill);
every `Fill` traces to an `Order` + a `ContractSpec` tick/multiplier; monotonic
`order_id` / `fill_id`; both end-of-test policies (force-liquidate vs leave-open
+ unrealized); official PnL from `Fill`s only (not bar closes); **negative CL
price end-to-end (Phase 08.2)** — bars moving `+5 → +1 → 0 → −5 → −20 → −10`
flow `MarketBar → … → BacktestResult` with no layer rejecting a bar for
`price <= 0`; realized PnL is `(exit − entry) · multiplier · signed_qty` across
zero (no `abs()`), e.g. long from −20 exit −15 → +$5,000; long from +5 exit −5 →
−$10,000; a bar opening exactly at 0.0 still fills.

C++ (`cpp/tests/test_execution_simulator.cpp`, `ctest` target
`quant_execution_simulator_tests`) -- the Phase 07 execution model: market
buy/sell at the reference price; tick slippage + half-spread, adverse by side,
tick from the `ContractSpec`; commission per contract per fill; limit fill /
no-fill / gap-through / fill price; stop trigger / no-trigger / gap-through /
slipped execution price; the intrabar stop-vs-target conservative rule (Stop
wins when both are reachable, even when Target would be more profitable -- never
resolved by PnL); raw-contract-only domain + `ContractSpec` tick/multiplier
authority + the un-normalized fixed-point guard; the Phase 02.5
`BackAdjusted` / `RawContinuous` `make_fill` guard unchanged; **signed / negative
prices (Phase 08.2)** — Market buy/sell, Limit (fill / gap-through / unreachable),
Stop (trigger / slip / no-trigger) and a bar straddling zero all work with
negative CL prices; adverse slippage stays adverse; negative tick-grid rounding
uses the same rule; an NQM6→NQU6 roll
priced from a **contemporaneous** raw bar (closed as NQM6, at NQM6's real price,
at the execution timestamp -- no relabel, no retroactive timestamp, no
back-adjusted price); an explicit `StaleObservedClose` recorded + audited
(`reference_price_ts_ns != fill.ts_fill_ns`); `RejectDefer` is the default and
never fills at a stale price; **no Fill precedes its engine event time**; **an
expired held contract fails loudly** (`ExecutionDomainError`), never a
retroactive fill, even with `StaleObservedClose` set; a multi-root ES/NQ/CL EOT
fixture where roots stop at different times -- the default leaves stale positions
open with mark provenance (`OpenPositionMark`), `Fail` refuses,
`ForceApproximateStaleClose` liquidates flagged + audited, the freshness
tolerance is configurable, expired-contract EOT never manufactures a fill;
deterministic N-bar latency + replay; the mandatory risk gate with the simulator
in the path; official PnL from fill prices including slippage; injected-simulator
determinism / dense ids.

`cpp/tests/test_core.cpp` also runs a momentum backtest through the engine.

C++ (`cpp/tests/test_portfolio.cpp`, `ctest` target `quant_portfolio_tests`) --
Phase 08 portfolio accounting: cash/equity identities; realized vs unrealized
separation; long + short; multi-contract / multi-root; gross/net/long/short
exposure via `ContractSpec.multiplier` (not `quantity × price`); mark age /
staleness / unmarked-position handling (`valuation_is_estimated` /
`valuation_complete`); a **negative CL mark** → positive gross exposure, position
still classified LONG; `Fill`-only realized PnL (favourable marks create none;
the closing `Fill` price is the realized number); peak-equity / drawdown;
deterministic daily-loss buckets across a day boundary; deterministic replay of
the engine `portfolio_at_end`.

C++ (`cpp/tests/test_risk_manager.cpp`, `ctest` target `quant_risk_manager_tests`)
-- Phase 08 / 08.1 hard risk: APPROVE/RESIZE/REJECT invariants on every returned
decision; per-symbol cap RESIZE then REJECT; per-root and gross-contracts caps;
leverage + gross-exposure rejection (proving the multiplier economics);
margin-utilisation rejection + RESIZE; missing-margin policy (Reject vs
TreatAsZero); **flip decomposition A–I** (`+2 SELL 1/2/3`, `+2 SELL 5` → RESIZE
`SELL 3` when one new short fits, mirrored short → long, and drawdown /
daily-loss / stale-mark / missing-margin each allowing the flatten leg while
RESIZE-ing away the new opposite exposure); **net-leverage magnitude** (large
positive *and* large negative net leverage REJECT); **negative-price risk
exposure** (`max_gross_exposure_usd` sees `+|price|·mult·contracts`);
`max_order_contracts` + the context-free `review(order)`; the drawdown kill
switch and a **flip that RESIZEs to flatten** through a full `BacktestEngine` run
with deterministic replay.

Python (`tests/python/test_boundary_schema.py`): symbol classification, frozen
column tuples, `ContractSpecModel` / `CanonicalBar` / `ContinuousBar` /
`BackAdjustedBar` / `RollEvent` validation, `write_boundary_bundle` column order
+ unresolved-instrument rejection, an end-to-end bundle accepted by the C++ CLI,
and the additive Phase 08 portfolio keys on the CLI JSON.

---

## I. Event-driven engine (Phase 06-07) -- `cpp/include/quant_core/engine.hpp`

**Deterministic ingestion** (`order_bar_events`): every input bar is validated
(`instrument_id` resolves through the registry, `ts_event_ns > 0`); a duplicate
`(instrument_id, ts_event_ns)` is a hard `InvalidMarketEvent`, never resolved by
input order; bars are ordered **canonically** by `(ts_event_ns, instrument_id)`
-- a total order for a Bar-only stream, independent of the caller's vector order,
file order, or any hash iteration; then a **global monotonic `seq`** (`0, 1, 2,
…`) is stamped. `MarketEventOrder` stays `(ts_event_ns, seq)` and, post-ingestion,
agrees with the canonical order. Equal-timestamp ES / NQ / CL events therefore
always process in the same order for any permutation of the same bar set.

`BacktestEngine::run(bars, strategy)` is deterministic: no step depends on hash /
pointer / filesystem / input-vector order. Per **Bar** event (only `Bar` in Phase
06):

1. update the **per-root** `MarketState` from the event's `instrument_id`
   (root resolved through the registry) -- never one global active contract;
2. **execute** the intent queued at an *earlier* bar of this root, at *this*
   event: (a) if the held contract differs from the execution-time active
   contract, close the old raw contract with a real Order/Fill and re-establish
   the target in the new one -- **Phase 07** prices this close-leg from
   execution-time raw data for that contract: a contemporaneous old-contract bar
   (`roll`); else, per `RollPriceFallback`, defer the roll to a later bar
   (`RejectDefer`, **default**) or use the last observed price stamped at the
   execution timestamp with a `RollFallbackAudit` (`StaleObservedClose`, opt-in);
   a held contract already non-tradable at `T` with no earlier close is a **loud
   `ExecutionDomainError`** -- never a retroactive fill, never a silent stale
   price; (b) resolve the active contract from the execution-time `MarketState`;
   (c) `order_delta = target_units - current_position_units`;
   (d) Order → `RiskManager` → `RiskDecision` → apply → **`ExecutionSimulator`**
   (section J) → `Fill` (with `ts_fill_ns == T`) or a deterministic non-fill;
3. append the bar to the bounded per-root history;
4. build the look-ahead-safe `StrategyContext` (history ends at this bar) and
   call `Strategy::decide`;
5. the **engine** stamps `signal_id` (monotonic, restarts each `run`), sets the
   authoritative `ts_decision_ns` / `root_symbol`, `validate_signal`s, and queues
   the Signal for bar `i + 1 + EngineConfig.latency_bars` of the root (Phase 07;
   `latency_bars == 0` == the next bar).

**Decision time ≠ execution time.** A Signal formed on bar *i* stays root-level
intent; it is resolved to a raw contract only at bar *i+1* using *i+1*'s market
state -- so a signal decided on an NQM6 bar and executed after the feed rolls
trades **NQU6**.

**Position ledger** (`position_ledger.hpp`) is the minimal deterministic state
for official PnL: per raw contract, signed units + average entry + open ts.
Realized PnL = `(exit − entry) × multiplier × signed_units_closed` from `Fill`s
only. **No** cash / margin / collateral / multi-currency / MTM risk in the ledger
itself -- Phase 08's `PortfolioAccountant` (section K) wraps it and adds that.

**End-of-test policy** (`EndOfTestPolicy`): `LeaveOpen` (keep every open position;
report `unrealized_pnl_usd_at_end` + `final_position_marks`) or
`ForceLiquidateFinalClose` (default). Force-liquidation happens **at the final
engine event time `final_ts`** and stamps every EOT Fill there -- never at an
earlier per-contract bar. A position is force-liquidated only when its last
observed bar is a **contemporaneous executable price** (`final_ts - last_bar_ts
<= eot.contemporaneous_tolerance_ns` (Phase 07.2, default 0, no product-specific
threshold) AND the contract is tradable at `final_ts`). Otherwise `EotStalePolicy`
decides: `LeaveOpen` (**default** -- keep it open, report its `OpenPositionMark`),
`Fail` (loud `ExecutionDomainError`), or `ForceApproximateStaleClose` (opt-in --
liquidate at the last observed close, stamped at `final_ts`, flagged
`is_approximation` in `eot_liquidations`; throws if the contract is not tradable
at `final_ts`). An old observed close is never silently used as a current
executable price. Open positions are never silently dropped.

`BacktestResult` carries counts (`events/bars/signals/orders/fills/closed_trades/
rolls`), the Phase 07 roll-pricing counters (`rolls_priced_contemporaneous`,
`rolls_priced_stale`, `rolls_deferred`, `non_fills`,
`eot_positions_left_open_stale`) + `roll_fallback_audit` + `eot_liquidations`
(one `EotLiquidationAudit` per forced EOT fill: `execution_ts_ns`,
`reference_price_ts_ns`, `reference_age_ns`, `reference_price`,
`is_contemporaneous`, `is_approximation`), realized `gross/costs/net`,
`max_drawdown_usd` (on the realized+unrealized equity curve),
`unrealized_pnl_usd_at_end`, the full `orders` / `risk_decisions` / `fills` /
`trades` audit vectors, `final_positions` + `final_position_marks` (mark price,
`mark_ts_ns`, `mark_age_ns`, `is_stale`, per-position unrealized -- for Phase 08
risk), and the sorted unique `contracts_traded`. A **trade** == one `ClosedTrade`
(a close/reduce/flip on one raw contract); a flip is one closing trade plus a new
open.

---

## J. Execution simulator (Phase 07) -- `cpp/include/quant_core/execution_simulator.hpp`

Full semantics: `docs/EXECUTION_MODEL.md`. Summary:

`ExecutionSimulator::execute(order, ExecutionRequest, fill_id) -> ExecReport`
(`Filled` with a `Fill` | `NoFill` | `Rejected`). The engine submits every
risk-approved Order -- strategy legs, roll close-legs, end-of-test liquidations.
`BarExecutionSimulator` is the MVP OHLCV model; the interface is the sanctioned
seam for a later tick / MBP model with no strategy rewrite.

* **strategy never prices a fill.** The simulator derives the price from raw
  market data + `ExecutionConfig`; `make_fill()` stays the only Fill constructor
  (raw-contract domain, `ContractSpec` tick/multiplier, tick-grid rounding). The
  simulator also rejects an un-normalized fixed-point reference price.
* **`ExecutionConfig`**: `slippage_ticks`, `spread_ticks` (both in `ContractSpec`
  ticks, summed, applied adverse-by-side to *marketable* orders only),
  `commission_per_contract_usd` (every fill). `ExecutionConfig{}` == Phase 06
  (fill at the reference price, no cost). All-or-none fills; a volume-cap partial
  model is a later, explicitly-approximate schema.
* **Market**: reference = execution bar `open` (or the engine's forced-close
  `ref_price_override`); `Buy +adverse·tick`, `Sell −adverse·tick`.
* **Limit**: fills iff reachable (`Buy: low ≤ L`; `Sell: high ≥ L`); fill at `L`,
  or at the bar `open` on a gap-through (price improvement); never slips.
* **Stop**: triggers (`Buy: high ≥ S`; `Sell: low ≤ S`); fills at `S` (or the
  `open` on a gap-through) then adverse slippage -- a triggered stop is marketable.
* **Intrabar stop/target ambiguity**: `resolve_intrabar_bracket()` -- open
  through the stop → Stop; else open through the target → Target; else both
  reachable → **Stop** (worst case); else the single reachable one, else None.
  **Never** resolved by realised PnL.
* **Latency**: `EngineConfig.latency_bars` -- whole-bar deterministic delay, no
  jitter, no RNG.
* **causality (Phase 07.1)**: the engine asserts `fill.ts_fill_ns >= T` for
  every Fill it accepts while processing engine event time `T` (`== T` for the
  bar-based model) and throws `ExecutionDomainError` otherwise -- no strategy
  leg, roll fallback, or EOT liquidation may backdate a Fill. EOT liquidations
  are stamped at the final engine event time and never use a stale mark as a
  current executable price without explicit opt-in (Phase 07.2, below).
* **determinism**: same Order + same `ExecutionRequest` + same `ExecutionConfig`
  ⇒ identical `ExecReport` (price, ts, ids, commission). `fill_id` stays dense
  (only consumed on an actual fill).

**Roll fallback (Phase 07.1)**: `RollExecutionPolicy.fallback` defaults to
`RejectDefer` -- when there is no contemporaneous old-contract bar the roll is
postponed, not approximated. `StaleObservedClose` is an explicit opt-in that
still stamps the fill at `T` and appends a `RollFallbackAudit` (`execution_ts_ns`,
`reference_price_ts_ns`, `reference_age_ns`, `reference_price`, `fallback_policy`)
so the stale source is never lost. A held contract already non-tradable at `T`
with no earlier close → loud `ExecutionDomainError` (no manufactured expiry
fill); scheduled expiry / settlement needs typed lifecycle events (Phase 08+).

**EOT liquidation (Phase 07.2)**: `EotExecutionPolicy` (`EngineConfig.eot`) --
`stale` defaults to `EotStalePolicy::LeaveOpen` (a stale-marked position is kept
open with an `OpenPositionMark`, not liquidated from an old close);
`contemporaneous_tolerance_ns` defaults to 0. `Fail` / `ForceApproximateStaleClose`
are the opt-ins. Every forced EOT fill is stamped at the final engine event time
and recorded in `eot_liquidations`.

**`EngineConfig`** carries `end_of_test`, `eot` (`EotExecutionPolicy`), `roll`
(`RollExecutionPolicy`: `fallback` (default `RejectDefer`) +
`contemporaneous_tolerance_ns`), `latency_bars`, and `execution` (the
`ExecutionConfig` for the convenience constructor). The
`BacktestEngine(registry, resolver, risk, execution, config)` overload injects a
simulator explicitly; the 4-arg form builds a `BarExecutionSimulator` from
`config.execution`.

---

## K. Portfolio accounting + hard risk (Phase 08) -- `portfolio.hpp` / `margin.hpp` / `risk_config.hpp` / `risk_manager.hpp`

Full detail: `docs/PORTFOLIO_AND_RISK.md`. Summary:

**`PortfolioAccountant`** wraps a `PositionLedger`, consuming **only validated
`Fill`s** (`apply_fill`) and **valuation marks** (`observe_mark` -- a bar open or
close, never a `Fill`, never an execution price). `snapshot(as_of_ts_ns)`
produces a deterministic **`PortfolioState`**: `cash_usd` (`starting_capital +
net_realized`), `equity_usd` (`cash + unrealized`), realized / unrealized kept
strictly separate, `gross/net/long/short_exposure_usd`, `gross/net_leverage`,
`initial/maintenance_margin_usd`, `margin_utilization_pct`, `margin_complete`,
`peak_equity_usd` / `drawdown_usd` / `drawdown_pct` (equity curve, peak seeded at
`starting_capital`), deterministic daily-loss buckets
(`session_day_index`, `day_start_equity_usd`, `day_equity_change_usd`), and per
position `mark_ts_ns` / `mark_age_ns` / `mark_is_stale` + portfolio
`has_stale_mark` / `valuation_complete`. **Futures exposure**:
`gross_notional = |units · price · ContractSpec.multiplier|` (>= 0, valid for a
**negative** futures price), `signed_notional = sign(units) · gross_notional` --
long/short is by **position direction, never the sign of the price**; never
`quantity × price`. An unmarked position uses avg-entry notional flagged
`valuation_is_estimated`. `correlation_adjusted_gross_exposure_usd` == gross
(SPAN / cross-margin is an explicit phase-09+ TODO).

Signed-price note: as of **Phase 08.2** the price-domain guard
(`is_plausible_raw_price`) is finite + magnitude only, so a zero or negative
normalized price flows end-to-end — `MarketEvent → Order → ExecutionSimulator →
Fill → PositionLedger → PortfolioAccountant → BacktestResult`. Portfolio/risk
exposure was already correct for signed prices (Phase 08.1); official realized
PnL is `(exit − entry) · multiplier · signed_qty` with no `abs()`.

**Margin** (`margin.hpp`) is a typed, dated, per-root, per-contract input
(`MarginRequirement { root_symbol, initial_margin_usd, maintenance_margin_usd,
source, as_of_ns }`), **never** derived from Databento or guessed from notional.
`MarginModel` + `load_margin_model("root_symbol,initial_margin_usd,maintenance_margin_usd,source,as_of_ns")`.
Missing data is visible: `MissingMarginPolicy::Reject` (default -- a
risk-increasing order in an un-margined root is REJECTed, `margin_complete` goes
false) or `TreatAsZero` (opt-in, still flagged).

**`RiskManager`** gains a context overload `review(order, RiskReviewContext)` (the
default forwards to `review(order)`, so Phase 05/06/07 managers are unchanged)
plus `risk_config()` / `margin_model()` / `portfolio_config()` hooks the engine
reads. **`PortfolioRiskManager`** implements the deterministic hard hierarchy.

**Flip decomposition (Phase 08.1)**: an order opposing the current position is
split into `close_qty = min(requested, |current_units|)` (risk-**reducing** --
**always approved**, kill switches included) and `requested_open = requested −
close_qty` (opens new opposite-side exposure -- the only part gated). `|post| ≤
|pre|` is **not** a blanket approve: `+2, SELL 3` contains a `SELL 1` that opens
a short and is risk-checked. Opening-leg hierarchy (first decisive wins):
drawdown kill switch → daily-loss limit → stale-mark → missing-margin → position
caps (per-symbol / per-root / gross) → exposure / leverage / margin-utilisation.
The first four **block the opening leg** for a risk-increasing order (a flip then
RESIZEs to `close_qty`, a pure new position REJECTs); caps and exposure RESIZE
via a downward integer scan for the largest feasible opening quantity evaluated
as the full post-trade state (`max_order_contracts` caps the opening leg only).
`approved = close_qty + feasible_open`. So a kill switch **permits closing
existing risk but prohibits new opposite exposure** — `current +2, drawdown
active, SELL 5` → **RESIZE to SELL 2**, never APPROVE 5, never REJECT 5. Mirror
for short → long. `max_net_leverage` compares `|net_exposure_after| / equity`
(magnitude -- a net-short book cannot pass on sign). `APPROVE approved ==
requested > 0`; `RESIZE 0 < approved < requested`; `REJECT approved == 0` --
unchanged, every decision passes `validate_risk_decision`. An LLM can never
override any of this (CLAUDE risk rule 1).

**Engine**: `BacktestEngine::run` maintains a `PortfolioAccountant` beside the
ledger, marks each instrument to this bar's open **before** executing intent (no
look-ahead) and to its close after, feeds every accepted `Fill`, and builds the
`RiskReviewContext` from the **pre-trade** snapshot for the mandatory gate.
`BacktestResult` gains `portfolio_at_end` (a `PortfolioState`), `risk_rejects`,
`risk_resizes`. The reference CLI JSON gains `equity_usd`, `cash_usd`,
`gross_exposure_usd`, `net_exposure_usd`, `gross_leverage`, `initial_margin_usd`,
`margin_utilization_pct`, `margin_complete`, `peak_equity_usd`,
`portfolio_drawdown_usd`, `portfolio_drawdown_pct`, `risk_rejects`,
`risk_resizes` (additive; the frozen CLI still runs `PassThroughRiskManager`, so
this is reporting-only there -- `RiskConfig` / `MarginModel` are wired at the
pybind boundary, Phase 19). Paper trading only (CLAUDE risk rule 2).

---

## What Phase 03 must obey

1. Write raw vendor data to `data/raw/` **once**; never modify it. Derived data
   goes only to `data/processed/`.
2. Preserve `ts_event_ns` exactly (UTC). `trading_day` / `session` are **new
   derived columns**, never overwrites.
3. Pull bars with `stype_out=instrument_id` (continuous → raw_symbol is
   unsupported) and pull the `definition` schema, so layer 2 and layer 5 are
   produced together. Resolve every bar's `raw_symbol` / tick / multiplier by
   joining `bar.instrument_id` to the definition table. Emit `bars.parquet` in
   `CANONICAL_BAR_COLUMNS` order and `contracts.csv` in `CONTRACT_COLUMNS` order.
4. Every canonical bar has a non-zero `instrument_id` present in the contracts
   file; QA fails the build otherwise.
5. The OHLCV `symbol` label may be continuous — that is not an error. The
   canonical `raw_symbol` (from the registry) that is continuous or parent *is*
   a hard error (`is_tradable_contract_symbol`).
6. Do **not** build the back-adjusted series yet (Phase 04). When built, it goes
   to `data/processed/backadjusted/` as `BackAdjustedBar` with no
   contract identity on the price row.
7. Roll detection uses observed `instrument_id` changes in the unadjusted
   continuous series, cross-checked against `symbology.resolve`; disagreements
   are logged, not silently resolved.
8. No silent forward-fill across missing minutes; gaps are recorded in a
   diagnostics table.
9. Write a `_lineage.json` sidecar per processed file (raw manifest SHA-256,
   dataset version, code commit, `tz: "UTC"`, price scale, generated_at).
10. `metadata.get_cost()` before any request; stop at the approved cap.
