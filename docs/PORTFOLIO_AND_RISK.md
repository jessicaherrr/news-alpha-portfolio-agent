# Portfolio Accounting and Hard Risk Management (Phase 08)

C++ only. `portfolio.hpp` / `margin.hpp` / `risk_config.hpp` / `risk_manager.hpp`
(+ the matching `src/*.cpp`), wired into `engine.hpp`.

Phase 06's `PositionLedger` is the minimal state for official realized PnL from
`Fill` events. Phase 08 adds a deterministic portfolio valuation layer and turns
the frozen `RiskManager` interface into real hard risk controls that an LLM
cannot bypass or override (CLAUDE.md risk rule 1).

```
Signal -> [ActiveContractResolver] -> Order -> [RiskManager.review(order, ctx)]
       -> RiskDecision -> [ExecutionSimulator] -> Fill -> PositionLedger
                                                       -> PortfolioAccountant
```

---

## 1. PortfolioState -- the deterministic snapshot

`PortfolioAccountant` wraps a `PositionLedger`. It consumes **only validated
`Fill` events** (`apply_fill`) and **valuation marks** (`observe_mark`, a bar
open or close) and produces a `PortfolioState` for any instant (`snapshot`).

| group | fields |
|---|---|
| cash / PnL / equity | `starting_capital_usd`, `gross_realized_pnl_usd`, `costs_usd`, `net_realized_pnl_usd`, `unrealized_pnl_usd`, `cash_usd`, `equity_usd` |
| exposure | `gross_exposure_usd`, `net_exposure_usd`, `long_exposure_usd`, `short_exposure_usd`, `gross_contracts`, `net_contracts` |
| leverage | `gross_leverage`, `net_leverage` |
| margin | `initial_margin_usd`, `maintenance_margin_usd`, `margin_utilization_pct`, `margin_complete` |
| peak / drawdown | `peak_equity_usd`, `drawdown_usd`, `drawdown_pct` |
| daily loss | `session_day_index`, `day_start_equity_usd`, `day_start_net_realized_usd`, `day_realized_pnl_usd`, `day_equity_change_usd` |
| marks / staleness | per position: `mark_price`, `mark_ts_ns`, `mark_age_ns`, `mark_present`, `mark_is_stale`; portfolio: `has_stale_mark`, `worst_mark_age_ns` |
| correlated exposure | `correlation_adjusted_gross_exposure_usd` (== `gross_exposure_usd`; **TODO phase-09+**) |

Determinism: `std::map` iteration only, no clock, no RNG, no hash-order
dependence. `snapshot` is a pure function of the fills + marks applied so far.

### cash / equity model (explicit)

Futures post margin; they do not pay cash at entry. The model is a standard
futures account:

```
cash_usd   = starting_capital + net_realized_pnl        (variation margin on CLOSED trades)
equity_usd = cash_usd + unrealized_pnl                  (a.k.a. liquidating value / NLV)
```

Unrealized PnL is open-trade equity. It is reported but **never** folded into
realized PnL or into cash (requirement 9). Realized PnL is `Fill`-derived only
(requirement 8): a mark can never become a realized number and a mark price can
never become a `Fill` price (requirement 2 -- enforced structurally, `make_fill`
is still the only `Fill` constructor and marks never flow to it).

---

## 2. Futures exposure -- ContractSpec economics

Exposure is **never** `quantity x price`. For each position:

```
gross_notional_usd  = |units * valuation_price * ContractSpec.multiplier|   (>= 0 always)
signed_notional_usd = sign(units) * gross_notional_usd                      (direction from units)
```

`multiplier` is the point value from the `ContractSpec` (NQ 20, ES 50, CL 1000,
GC 100, ZN 1000 in the test fixtures). `gross_exposure_usd = Σ gross_notional`
(a sum of magnitudes -- **never negative**); `long_exposure_usd` /
`short_exposure_usd` sum `gross_notional` over `units > 0` / `units < 0`;
`net_exposure_usd = long_exposure - short_exposure = Σ signed_notional`.

**Negative futures prices** (CL has traded below zero). `gross_notional` uses
`|units| · |price| · multiplier` so it stays non-negative, and long/short is
determined by **position direction (`sign(units)`), never the sign of the
price**. `units = +1`, `CL mark = −20`, `multiplier = 1000` → `gross_notional =
20,000`, `signed_notional = +20,000`, classified **LONG**. (`unrealized_pnl` is
still `(mark − entry) · multiplier · units`, which is correctly negative here.)
The risk manager's post-trade exposure/leverage math mirrors this: notional
candidates use `|price|` and `sign(post_units)`.

**Phase 08.2:** the price-domain guard (`is_plausible_raw_price` /
`validate_market_event` / `validate_order` / `make_fill`) is now finite +
magnitude only — a positive, zero, or negative normalized price flows
end-to-end. The portfolio/risk exposure arithmetic above (Phase 08.1) already
handled signed prices; a negative CL mark now also reaches it through the normal
engine path, not only via a direct `observe_mark` call.

**Unmarked open position** — its notional falls back to the average **entry**
price. This is an **estimate**, not a current market value: `PositionExposure
.valuation_is_estimated = true` and `PortfolioState.valuation_complete = false`.
It is not presented as a live exposure — under the default stale-mark policy a
risk-increasing order is blocked while `valuation_complete` is false (see §5).

---

## 3. Margin model -- typed, explicit, never guessed

`margin.hpp`. Exchange margin is a **separate, human-supplied, dated** input --
CLAUDE.md market-data rule 5 forbids inventing it from Databento, and Phase 08
requirement 5 forbids guessing it from notional.

`MarginRequirement { root_symbol, initial_margin_usd, maintenance_margin_usd,
source, as_of_ns }` -- **per contract, per root**. `source` + `as_of_ns` make
every figure auditable (margins change often; a backtest records which schedule
it used, like it records the dataset version).

`MarginModel` is a `std::map<root, MarginRequirement>`. `load_margin_model(path)`
parses `configs/margin.csv` (header
`root_symbol,initial_margin_usd,maintenance_margin_usd,source,as_of_ns`). The
shipped file is a **placeholder** (`source = PLACEHOLDER_user_must_verify`) --
the user supplies real margins.

**Missing margin is visible, not silent** (`MissingMarginPolicy`):

| policy | held position with no requirement | risk-increasing order in that root |
|---|---|---|
| `Reject` (default) | `PortfolioState.margin_complete = false` | **REJECT** `missing_margin_metadata` |
| `TreatAsZero` (opt-in, research) | `margin_complete = false` | proceeds with 0 margin |

`margin_utilization_pct = initial_margin / equity`. When `equity <= 0` and margin
is required it is `kUtilizationNoEquity` (`1e9`) -- a finite, comparable,
JSON-safe stand-in for "infinite".

---

## 4. Risk hierarchy (`PortfolioRiskManager`)

`review(const Order&, const RiskReviewContext&)`. The engine builds the context
from **pre-trade** state.

### Flip decomposition (never trap risk reduction; never let a flip bypass risk)

An order that opposes the current position is split:

```
close_qty      = min(requested, |current_units|)   when the order opposes the position, else 0
requested_open = requested - close_qty
```

* `close_qty` — contracts that merely reduce the position toward flat. **Always
  approved**, whatever the portfolio state (kill switches included).
* `requested_open` — contracts that open **new opposite-side exposure** once the
  old position is gone (or an add on the existing side). This is the only part
  that faces the hierarchy below.

`|post| <= |pre|` is **no longer** a blanket approve — a flip through zero
(`+2, SELL 3` → `-1`) contains a `SELL 1` that opens a new short and must be
risk-gated.

| order | close leg | opening leg | outcome |
|---|---|---|---|
| `+2, SELL 1` | 1 | 0 | APPROVE 1 |
| `+2, SELL 2` | 2 | 0 | APPROVE 2 (flat) |
| `+2, SELL 3` | 2 | 1 | short-1 risk-gated → APPROVE 3 or RESIZE 2 |
| `+2, SELL 5`, only 1 new short fits | 2 | 3 → 1 | **RESIZE to SELL 3** (close 2 + open 1) |
| `+2, SELL 5`, kill switch active | 2 | 3 → 0 | **RESIZE to SELL 2** (flatten only) — not APPROVE 5, not REJECT 5 |

Mirror for short → long. A flip is **never** REJECTed outright — the close leg
always yields at least `close_qty > 0`, so the worst case is RESIZE-to-flatten.
Only a **pure new position** (`close_qty == 0`) can be REJECTed.

### Opening-leg hierarchy (first decisive rule wins)

| # | rule | effect on the opening leg |
|---|---|---|
| 1 | drawdown kill switch (`drawdown_pct >= max_drawdown_pct` or `drawdown_usd >= max_drawdown_usd`) | blocked (`drawdown_kill_switch`) |
| 2 | daily-loss limit (`day_start_equity - equity >= max_daily_loss_usd`) | blocked (`daily_loss_limit`) |
| 3 | stale-mark protection (ref price stale/missing, or `has_stale_mark`) under `RejectRiskIncreasing` | blocked (`stale_mark`) |
| 4 | missing margin metadata under `MissingMarginPolicy::Reject` | blocked (`missing_margin_metadata`) |
| 5 | position caps: `max_contracts_per_symbol`, `max_contracts_per_root`, `max_gross_contracts` (post-trade counts) | RESIZE, else block |
| 6 | `max_gross_exposure_usd`, `max_gross_leverage`, `max_net_leverage`, `max_margin_utilization_pct` (post-trade) | RESIZE, else block |
| 7 | otherwise | APPROVE |

"Blocked" ⇒ feasible opening quantity is 0. The opening quantity is found by a
downward integer scan (`1 … requested_open`, capped by `max_order_contracts`
which now limits the **opening leg only**, never the close); each candidate is
evaluated as the **full post-trade portfolio state** for `close_qty + open_qty`.
`approved = close_qty + feasible_open`; `approved == requested` → APPROVE,
`approved == 0` → REJECT (pure-open only), else RESIZE.

### Kill-switch flip behaviour (review point 3, preferred behaviour "A")

A kill switch (drawdown / daily loss) — and likewise stale-mark and
missing-margin blocks — **permits closing existing risk but prohibits any new
opposite exposure**:

```
current +2, drawdown kill switch active, SELL 5
  -> close 2 allowed, new short 3 prohibited
  -> RESIZE to SELL 2   (not APPROVE 5, not REJECT 5)
```

### Net-leverage sign (review point 3)

`max_net_leverage` compares a **magnitude**:
`|net_exposure_after| / equity <= max_net_leverage`. A large net-**short** book
(negative `net_exposure`) cannot slip under the limit on sign. `PortfolioState
.net_leverage` is the signed ratio for reporting; the *check* uses `std::fabs`.
Regression tests cover large positive and large negative net leverage.

### APPROVE / RESIZE / REJECT semantics (unchanged, enforced)

```
APPROVE   approved_quantity == requested_quantity  (> 0)
RESIZE    0 < approved_quantity < requested_quantity
REJECT    approved_quantity == 0
```

Every returned `RiskDecision` passes `validate_risk_decision` (tested). The
mandatory `Order -> RiskManager -> RiskDecision -> apply_risk_decision` path is
unchanged; `apply_risk_decision` still throws on a REJECT.

`review(const Order&)` (no context) applies only structural + `max_order_contracts`
(as a whole-order cap, since the position is unknown) -- full enforcement needs
the context overload the engine uses.

### APPROVE / RESIZE / REJECT semantics (unchanged, enforced)

```
APPROVE   approved_quantity == requested_quantity  (> 0)
RESIZE    0 < approved_quantity < requested_quantity
REJECT    approved_quantity == 0
```

Every returned `RiskDecision` passes `validate_risk_decision` (tested). The
mandatory `Order -> RiskManager -> RiskDecision -> apply_risk_decision` path is
unchanged; `apply_risk_decision` still throws on a REJECT.

`review(const Order&)` (no context) applies only structural + `max_order_contracts`
-- full enforcement needs the context overload the engine uses.

### Correlated exposure -- explicit TODO

The MVP is **additive by root**: no netting between correlated roots (ES/NQ), no
SPAN scanning, no correlation haircut. `PortfolioState.correlation_adjusted_gross_exposure_usd`
equals `gross_exposure_usd` until a real cross-margin / correlation model exists
(phase-09+). This is deliberately **not** guessed.

---

## 5. Stale-mark policy (requirement 7, feeds from Phase 07.2)

A mark carries `mark_ts_ns`, `mark_age_ns` (`as_of - mark_ts`, floored at 0) and
`mark_is_stale` (`!present || age > PortfolioConfig.mark_staleness_tolerance_ns`).
`PortfolioState.has_stale_mark` is true if **any** held position has a stale or
missing mark. `PortfolioState.valuation_complete` is its complement — a named
"are the exposure numbers a true current-market picture?" flag. Per position,
`valuation_is_estimated` marks a leg whose notional came from the average **entry**
price because no mark exists (an estimate — see §2).

The engine marks each instrument to **this bar's open before executing intent**
(a real print at `ts_event_ns` -- no look-ahead) so a normal held position is
never falsely stale; positions in instruments that have stopped printing (a root
that ended earlier -- the Phase 07.2 multi-root EOT case) genuinely age and trip
`has_stale_mark` / clear `valuation_complete`.

Under `StaleMarkPolicy::RejectRiskIncreasing` (default) the **opening leg** of a
risk-increasing order is blocked when it or the portfolio depends on a stale /
missing / estimated mark -- because the leverage and margin-utilisation checks
divide by an equity computed from those marks. A flip then RESIZEs to its close
quantity; a pure new position is REJECTed. Risk-reducing (close) legs are never
blocked. `Ignore` disables the check.

**A stale / estimated mark never becomes a `Fill`.** Marks live only in
`PortfolioAccountant` (`observe_mark`); the execution path (Phase 07) is
untouched and `make_fill` remains the only `Fill` constructor. Average-entry
fallback notional is a risk/valuation estimate only — it never flows to
execution.

---

## 6. Daily-loss and drawdown -- deterministic definitions

**Drawdown** is on the `(net_realized + unrealized)` equity curve with
`peak_equity` seeded at `starting_capital`:

```
peak_equity = max over all observations of equity     (seed = starting_capital)
drawdown_usd = max(0, peak_equity - equity)
drawdown_pct = drawdown_usd / peak_equity             (0 when peak <= 0)
```

(`BacktestResult.max_drawdown_usd` from Phase 07 is unchanged -- it seeds its
peak at 0 and is a separate reported number.)

**Daily loss** buckets time by
`session_day_index = floor((ts_ns - day_boundary_offset_ns) / 86_400e9)` (floor
toward -inf). `day_boundary_offset_ns` defaults to 0 (UTC midnight); set it to
the CME session reset (~17:00 America/Chicago) upstream where the session
calendar is known -- the deterministic C++ core carries no tz database. On the
first observation of a new bucket, `day_start_equity_usd` and
`day_start_net_realized_usd` re-seed to the current values. The limit trips when
`day_start_equity_usd - equity_usd >= max_daily_loss_usd` (mark-to-market,
including open-trade equity). `day_realized_pnl_usd` is reported separately for a
realized-only view.

---

## 7. Multi-asset (ES / NQ / CL / GC / ZN)

The engine was already per-root (per-root `MarketState`). `MarginModel` is keyed
by root and `configs/margin.csv` carries all five. `PortfolioState` aggregates
per-root exposure, per-root position caps (`max_contracts_per_root`), and a
portfolio-wide gross cap. Tests cover a five-root registry.

---

## 8. Engine integration

`BacktestEngine::run` keeps its `PositionLedger` and adds a `PortfolioAccountant`
(`portfolio_config()` / `margin_model()` come from the `RiskManager` -- a
`PortfolioRiskManager` opts the whole run into hard risk; `PassThroughRiskManager`
/ `MaxContractsRiskManager` leave the portfolio view reporting-only and behave
exactly as in Phase 07).

- every accepted `Fill` -> `portfolio.apply_fill` (Fill-only accounting)
- every bar open (step 1) and close (step 3) -> `portfolio.observe_mark`
- before the mandatory risk gate -> a `RiskReviewContext` from the pre-trade
  `snapshot(current_event_ts_ns)` + the order instrument's `ContractSpec` + mark
  freshness + current per-instrument / per-root units
- `BacktestResult.portfolio_at_end = portfolio.snapshot(final_ts)`,
  `risk_rejects`, `risk_resizes`

`BacktestResult` and the reference CLI JSON gain the portfolio fields additively
(`equity_usd`, `gross_exposure_usd`, `net_exposure_usd`, `gross_leverage`,
`initial_margin_usd`, `margin_utilization_pct`, `margin_complete`,
`peak_equity_usd`, `portfolio_drawdown_usd`, `portfolio_drawdown_pct`,
`risk_rejects`, `risk_resizes`). The frozen CLI signature is unchanged and still
runs `PassThroughRiskManager`; `RiskConfig` / `MarginModel` are wired at the
pybind boundary (Phase 19).

---

## 9. Tests

`cpp/tests/test_portfolio.cpp` (`quant_portfolio_tests`): cash/equity identities;
realized vs unrealized separation; long + short; multi-contract / multi-root;
gross/net/long/short exposure via `multiplier` (not `quantity x price`); mark
age / staleness / unmarked-position handling + `valuation_is_estimated` /
`valuation_complete`; **negative CL mark → positive gross exposure, position
still classified LONG, direction not inverted by the price sign**; `Fill`-only
realized PnL (a wall of favourable marks creates no realized PnL; the closing
`Fill` price is the realized number); peak-equity / drawdown; deterministic
daily-loss buckets across a day boundary; deterministic replay of the engine
`portfolio_at_end`.

`cpp/tests/test_risk_manager.cpp` (`quant_risk_manager_tests`): APPROVE / RESIZE /
REJECT invariants on every returned decision; per-symbol cap RESIZE then REJECT;
per-root cap; gross-contracts cap; leverage + gross-exposure rejection (proving
the multiplier economics); margin-utilisation rejection + RESIZE; missing-margin
policy (Reject vs TreatAsZero); **flip decomposition A–I** — `+2 SELL 1/2/3`,
`+2 SELL 5` RESIZE-to-`SELL 3` when one new short fits, mirrored short → long,
and drawdown / daily-loss / stale-mark / missing-margin each allowing the flatten
leg while RESIZE-ing away the new opposite exposure; **net-leverage magnitude**
(large positive *and* large negative net leverage both REJECT); **negative-price
risk exposure** (`max_gross_exposure_usd` sees `+|price|·mult·contracts`);
`max_order_contracts` and the context-free `review(order)`; the drawdown kill
switch and a **flip that RESIZEs to flatten** through a full `BacktestEngine` run
with deterministic replay.

Paper trading only (CLAUDE.md risk rule 2). No live routing exists to gate.
