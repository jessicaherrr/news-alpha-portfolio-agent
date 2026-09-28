# Deterministic Futures Execution Model (Phase 07)

`cpp/include/quant_core/execution_simulator.hpp` + `src/execution_simulator.cpp`.

Phase 06 filled every Market order at the execution bar's open with a flat
commission. Phase 07 replaces that approximation with an explicit, configurable,
**deterministic** execution model and a clean interface the engine submits
risk-approved orders to.

```
Signal ─▶ [ActiveContractResolver] ─▶ Order ─▶ [RiskManager] ─▶ RiskDecision
       ─▶ [ExecutionSimulator] ─▶ Fill | deterministic non-fill ─▶ PositionLedger
```

## Non-negotiables

1. **The strategy never chooses the official fill price.** It emits a root-level
   `Signal` (a target position). The engine resolves the contract and quantity;
   the `ExecutionSimulator` owns the price.
2. **Every Fill is built only through `make_fill()`** — so only
   `PriceDomain::RawContract` prices, `tick_size` / `multiplier` copied from the
   `ContractSpec`, tick-grid rounding, contract-live-at-`ts` check. The simulator
   additionally rejects an un-normalized (Databento fixed-point ~1e13) reference
   price up front (`is_plausible_raw_price`). **A normalized price may be
   positive, zero, or negative (Phase 08.2)** — the guard is finite + `|price| <
   1e9` only; negative CL prices fill normally, all Market/Limit/Stop comparisons
   are relational and hold across zero, adverse slippage stays adverse, and
   `std::round(price/tick)*tick` aligns negatives by the same rule.
3. **No RNG.** Same `Order` + same `ExecutionRequest` + same `ExecutionConfig` ⇒
   byte-identical `ExecReport` (price, timestamp, ids, commission). No random
   latency, ever.
4. **OHLCV bars are never given invented depth.** Slippage and spread are
   explicit configured tick offsets, not a synthesised bid/ask or order book.
5. **The risk gate stays mandatory** and upstream of the simulator. `order.quantity`
   reaching the simulator is already the risk-approved quantity.
6. **Causal fills only (Phase 07.1).** A Fill produced while processing engine
   event time `T` is stamped at `T` (`>= T` is the asserted invariant). No path
   backdates a Fill; missing execution-time data fails or defers, it is never
   approximated with a past timestamp.

## `ExecutionConfig`

| field | meaning | default |
|---|---|---|
| `slippage_ticks` | adverse ticks applied to a *marketable* order (Market, or a Stop once triggered) | `0` |
| `spread_ticks` | extra half-spread ticks, summed with `slippage_ticks` before applying; separate only for attribution | `0` |
| `commission_per_contract_usd` | charged per contract per fill, on every fill the simulator produces | `0` |

Tick size is **always** `ContractSpec.tick_size` — never a field here, never a
product constant. `ExecutionConfig{}` reproduces Phase 06 (fill at the reference
price, no cost).

Partial fills: Phase 07 MVP is **all-or-none**. A volume-cap partial-fill model
(`fill_qty = min(order_qty, floor(bar.volume * f))`) is a documented approximation
deferred to a richer market-data schema — it is not faked from OHLCV.

## Market orders

Reference price = the execution bar's `open` (or an engine-supplied
`ref_price_override` for an administrative close). Then:

```
adverse = slippage_ticks + spread_ticks           (0 for administrative closes)
Buy  fill = reference + adverse * tick_size
Sell fill = reference − adverse * tick_size
```

`make_fill()` then rounds to the contract tick grid. `Fill.slippage_ticks`
records the adverse ticks applied; `Fill.fill_price` already includes them.

## Limit orders

A limit order fills only when the bar's range makes the price **reachable**. No
queue-priority assumption is made (OHLCV cannot prove one), so the fill is never
better than the limit **except** on a gap, where the bar's `open` is the genuine
first print.

| side | fills when | fill price |
|---|---|---|
| **Buy limit** `L` | `bar.low ≤ L` | `bar.open` if `bar.open ≤ L` (gap-through, price improvement), else `L` |
| **Sell limit** `L` | `bar.high ≥ L` | `bar.open` if `bar.open ≥ L` (gap-through), else `L` |

A limit order **never** pays slippage or spread — you named your price.
Unreachable ⇒ `ExecOutcome::NoFill` (`detail = "limit_unreachable"`).

## Stop orders

A stop triggers when the bar trades to/through the stop price; once triggered it
is marketable and takes adverse slippage.

| side | triggers when | execution price |
|---|---|---|
| **Buy stop** `S` | `bar.high ≥ S` | `bar.open` if `bar.open ≥ S` (gap-through), else `S`; then `+ adverse * tick` |
| **Sell stop** `S` | `bar.low ≤ S` | `bar.open` if `bar.open ≤ S` (gap-through), else `S`; then `− adverse * tick` |

Not triggered ⇒ `ExecOutcome::NoFill` (`detail = "stop_not_triggered"`).

## Intrabar stop / target ambiguity

An OHLC bar does not reveal the path between `open`, `high`, `low`, `close`. When
an open position's protective **stop** and its profit **target** are both
potentially reachable inside one bar, `resolve_intrabar_bracket()` applies a
fixed conservative rule — **never** the outcome with the better PnL:

1. bar opens already through the stop → **Stop** (gap)
2. else bar opens already through the target → **Target** (gap)
3. else both reachable in the bar → **Stop** (worst case for the position)
4. else whichever single one is reachable, else **None**

`exit_side` is the side of the closing order: `Sell` exits a long (stop below,
target above), `Buy` exits a short (stop above, target below). Bracket-order
*strategies* arrive with the strategy DSL (Phase 10); Phase 07 ships and tests
the rule.

## Latency

`EngineConfig.latency_bars` (`N ≥ 0`): a `Signal` decided on bar `i` of a root
executes on bar `i + 1 + N` of that root, at that bar's open. `N = 0` is the
Phase 06 next-bar behaviour. The model is a whole-bar delay only — no fractional
latency, no random jitter. Multi-bar *working orders* (cancel/replace, session
expiry) are a strategy-DSL concern, not modelled here; a Day order that does not
fill on its execution bar is reported `NoFill` and dropped.

**Independent pending intents (invariant, tested `quant_latency_queue_tests`).**
Each root keeps a **FIFO `std::deque<PendingIntent>`** — `{Signal, due-bar
index}` only, never an execution price, contract identity or risk state. A
`Signal` is `push_back`-ed at decision time and `pop_front`-ed when due, so with
`N > 0` several intents coexist and a later `Signal` **never overwrites** an
earlier not-yet-due one: three consecutive decisions at `T0,T1,T2` execute at
`T0+1+N, T1+1+N, T2+1+N` in decision order. Contract resolution and
`order_delta = target_units − current_position` are computed at **execution**
time (an earlier pending intent may have moved the position first). A `NoDecision`
enqueues nothing and cancels nothing (Phase 11.1). A risk-rejected intent does
not delete or mutate later queued intents. The **only** path that collapses the
queue is a `RejectDefer` roll that cannot be priced — documented above under
*Roll execution*: only the most recent target is kept, by design, because the
deferred intents would net to the same target anyway.

## Causality invariant (Phase 07.1)

**Every Fill produced while the engine is processing event time `T` satisfies
`fill.ts_fill_ns >= T`** — for the current bar-based model, `== T`. No path — a
strategy leg, a roll close-leg, a roll fallback, an end-of-test liquidation, or
the simulator — may backdate a Fill. The engine checks this on every fill it
accepts and throws `ExecutionDomainError` otherwise. A Fill created while
processing `T` is an event *caused by* `T`; it cannot have happened before it.

## Roll execution — no retroactive fills

When the continuous feed has moved to the new contract and the engine holds the
old one, the **old raw contract is closed explicitly** (real Order → Fill, never
relabelled) and the target is re-established in the **new raw contract** — both
priced from execution-time raw data for their own contract.

Pricing the close-leg:

| case | close price | close `ts` | tag | counter |
|---|---|---|---|---|
| a **contemporaneous** old-contract bar exists (age ≤ `roll.contemporaneous_tolerance_ns`, default 0 = exact-timestamp overlap, the Phase 04.5 `same_timestamp_close_close` basis) | that bar's close | `T` | `roll` | `rolls_priced_contemporaneous` |
| an auxiliary **`roll.close_marks`** entry `(held_id, T)` exists (Phase 13.5C) — a real same-timestamp outgoing-contract close supplied out of band, never as a `MarketEvent` | that close | `T` | `roll` | `rolls_priced_contemporaneous` **and** `rolls_priced_auxiliary_marks` (a subset) |
| old contract still live at `T`, no fresh bar, no aux mark, fallback = **`RejectDefer` (default)** | — no fill — the roll is postponed to the next bar of the root; only the most recent target is kept | — | — | `rolls_deferred` |
| old contract still live at `T`, no fresh bar, fallback = **`StaleObservedClose`** (explicit opt-in approximation) | last **observed** old-contract close | `T` (never retroactive) | `roll_stale_close` | `rolls_priced_stale` + one `RollFallbackAudit` |
| old contract **not tradable** at `T` (would-be expiry-forced), no earlier event closed it | — | — | **`ExecutionDomainError` — the run fails loudly** | — |

The default is `RejectDefer`: in a reliability-oriented research engine, missing
execution-time raw-contract data is **not approximated by default**. The correct
fix is to provide the old contract's real same-timestamp close around the roll
(the Phase 04.5 Stage-B roll-overlap download), which yields the `roll` path —
either as a genuine contemporaneous `MarketEvent` at `T`, or, when the primary
feed is a clean single-contract continuous front (Phase 13.5C), via
`EngineConfig::roll.close_marks`: an out-of-band map `(instrument_id, ts) ->
close` consulted **only** by the close-leg. Those entries are never
`MarketEvent`s — they do not touch `MarketState.active_instrument_id`,
`bars_seen`, `Strategy::decide`, `BarHistory`, latency, or daily-equity
sampling; an empty map is byte-identical to the frozen path. A thin-market roll
whose outgoing contract has no print at the first candidate execution instant
still defers (`rolls_deferred`) until a real same-timestamp close exists a
bounded number of bars later — never a stale price, never a retroactive fill.

**Expiry-forced rolls no longer manufacture a fill.** The previous behaviour
discovered at `T` that the old contract was already dead and created a Fill
stamped at `old.tradable_until_ns()` — retroactive, causality-breaking. It is
removed. If a held contract is non-tradable at the current execution event and
no earlier event closed it, the engine **fails loudly** (`ExecutionDomainError`).
Proper scheduled expiry / settlement needs typed lifecycle events and arrives in
a later phase. This is a deliberate fail-safe: a wrong number is worse than a
stopped backtest.

### `RollFallbackAudit` (provenance for `StaleObservedClose`)

The frozen `Fill` schema is not overloaded. Each `StaleObservedClose` close-leg
appends one record to `BacktestResult.roll_fallback_audit`:

| field | meaning |
|---|---|
| `instrument_id` / `raw_symbol` | the old raw contract that was closed |
| `execution_ts_ns` | engine event time `T` == `fill.ts_fill_ns` |
| `reference_price_ts_ns` | timestamp of the bar the stale price came from (`< T`) |
| `reference_age_ns` | `execution_ts_ns − reference_price_ts_ns` (> 0) |
| `reference_price` | the stale close used |
| `fallback_policy` | `StaleObservedClose` |

`reference_price_ts_ns != execution_ts_ns` — the approximation is never presented
as contemporaneous.

## End-of-test liquidation (Phase 07.2)

`ForceLiquidateFinalClose` is processed after the final event, so it happens **at
the final engine event time `final_ts`**. Every EOT Fill is stamped at `final_ts`
— never at an earlier per-contract bar. A position is force-liquidated **only
when its last observed raw-contract bar is a contemporaneous executable price**:

```
mark_age = final_ts − position.last_observed_bar_ts
contemporaneous  ⟺  contract tradable at final_ts
                    AND  0 ≤ mark_age ≤ eot.contemporaneous_tolerance_ns
```

`eot.contemporaneous_tolerance_ns` defaults to `0` (the position's last bar must
be exactly the final engine event). No product-specific hard-coded threshold.

When a position's mark is **not** contemporaneous, `EotStalePolicy` decides — an
old observed close is never silently used as a current executable price:

| `EotStalePolicy` | behaviour |
|---|---|
| **`LeaveOpen`** (default) | Keep the position open. It stays in `final_positions` and gets an `OpenPositionMark` (mark price, `mark_ts_ns`, `mark_age_ns`, `is_stale`, per-position unrealized). Its unrealized PnL is reported separately and **never** folded into realized PnL. `eot_positions_left_open_stale` counts these. |
| `Fail` | Throw `ExecutionDomainError` — refuse the forced liquidation. |
| `ForceApproximateStaleClose` | Explicit opt-in: liquidate at the last observed close, still stamped at `final_ts`, recorded in `eot_liquidations` with `is_contemporaneous = false` / `is_approximation = true`. Refused (throws) if the contract is not tradable at `final_ts`. |

### `EotLiquidationAudit` — every forced EOT liquidation

`BacktestResult.eot_liquidations` gets one record per EOT Fill: `instrument_id` /
`raw_symbol`, `units`, `execution_ts_ns` (== `fill.ts_fill_ns` == `final_ts`),
`reference_price_ts_ns`, `reference_age_ns`, `reference_price`,
`is_contemporaneous`, `is_approximation`. A contemporaneous liquidation has
`reference_age_ns ≤ tolerance`; an approximation has `reference_price_ts_ns !=
execution_ts_ns` and `reference_age_ns > 0` — the staleness is on the record, not
hidden.

### Mark reporting for surviving positions

Whether a position survives because of `EndOfTestPolicy::LeaveOpen` or because it
was stale under `EotStalePolicy::LeaveOpen`, it appears in `final_positions` and
in `final_position_marks` with mark price, mark timestamp and mark age — the raw
material Phase 08 risk needs. Realized PnL (`gross/net_realized_pnl_usd`) is from
`Fill`s only; unrealized (`unrealized_pnl_usd_at_end`) is separate.

The new-contract roll leg and ordinary signal legs fill as Market orders at the
execution bar's open. Back-adjusted / synthetic-continuous prices can never enter
any of this (`make_fill` + `price_domain`).

## Costs

`commission_per_contract_usd` is applied to every fill (strategy legs, roll
close-legs, end-of-test liquidations) and flows to `Fill.commission_usd →
PositionLedger.costs_usd`. `BacktestResult` still derives official realized PnL
**only** from `Fill` events via the `ContractSpec` tick/multiplier —
`net = gross − costs`.

## Tests

`cpp/tests/test_execution_simulator.cpp` (`ctest` target
`quant_execution_simulator_tests`): market buy/sell; tick slippage + spread,
adverse by side; commission; limit fill / no-fill / gap-through; stop trigger /
no-trigger / gap-through / slipped price; the intrabar stop-vs-target rule (incl.
a case where Target is more profitable but Stop is still chosen); raw-contract
domain + `ContractSpec` tick/multiplier authority; the Phase 02.5
`BackAdjusted` / `RawContinuous` guard; **signed / negative prices (Phase 08.2)**
— Market buy/sell + adverse slippage, Limit (at-price / gap-through / unreachable),
Stop (trigger / slip / no-trigger), a bar straddling zero, negative tick-grid
rounding, all with negative CL prices; an NQM6→NQU6 roll priced from a
contemporaneous raw bar (no relabel, no retroactive timestamp, no back-adjusted
price); an explicit `StaleObservedClose` flagged + audited (`reference_price_ts_ns
!= fill.ts_fill_ns`); `RejectDefer` is the default and never fills at a stale
price; **no Fill precedes its engine event time**; **an expired held contract
fails loudly, never a retroactive fill** (even with `StaleObservedClose` set);
a multi-root ES/NQ/CL fixture where roots stop at different times — the default
leaves stale positions open with mark provenance, `Fail` refuses,
`ForceApproximateStaleClose` liquidates flagged + audited, a freshness tolerance
is configurable; expired-contract EOT never manufactures a fill; deterministic
N-bar latency; the mandatory risk gate with the
simulator in the path; official PnL from fill prices (including slippage);
injected-simulator determinism / dense ids.
