# News Alpha Phase G -- Multi-Asset Portfolio Construction

    ... -> Candidate Signals -> Factor Diagnostics          (Phase E)
        -> Multi-Asset Signal Ranking (RankedSignalSet)      (Phase F)
        -> PORTFOLIO CONSTRUCTION (PortfolioPlan)            (Phase G, this document)
        -> existing C++ execution/backtest boundary          (handoff proven here)
        -> scientific validation of the portfolio            (Phase H -- not here)

A PortfolioPlan says how capital WOULD be allocated to a set of signals under
a mandate. It is not evidence that the portfolio works: nothing in Phase G is
validated, and no plan is a verdict.

## 1. Audit (verified before designing)

| Area | What exists | Allocator? |
|---|---|---|
| `cpp/include/quant_core/portfolio.hpp` | `PortfolioAccountant`: Fill-only positions/realized PnL, mark-based exposure, leverage, margin, drawdown, daily loss. Additive by root; `correlation_adjusted_gross_exposure_usd` is a documented TODO equal to gross. | No -- backtest accounting |
| `risk_manager.hpp` / `risk_config.hpp` | `PortfolioRiskManager`: per-ORDER hard gate (APPROVE/RESIZE/REJECT) -- kill switches, unit caps, USD exposure/leverage, margin utilization, stale marks. Unit caps count a futures contract and an ETF share identically. | No -- admissibility of one order |
| `margin.hpp` | Typed per-root margin table; no committed CME schedule exists (paper trading runs `treat_as_zero`, visibly). | -- |
| `scheduled_target_strategy.hpp`, `targets_run.hpp`, `paper_trading_run.hpp` | Replay integer target units per ROOT through `BacktestEngine`; multi-root by construction (`by_root_ts_`). `parse_targets_csv` requires a `stratdsl1:` fingerprint (frozen Phase 11 parser). | No -- execution of given targets |
| Python `alpha_agent.backtest.targets` | `TargetSchedule` = one root, one DSL strategy. | No |
| Python `alpha_agent.paper.risk_policy` | `PaperRiskPolicy` = typed mirror of `RiskConfig` + CSV writer. | No |
| Strategy DSL (`strategy/spec.py`) | `target_units` are fixed integers per rule (e.g. +/-1 contract). | No |
| `recommendation.signal_ranking` | Research prioritization; explicitly "not weights, sizes or allocations". | No |
| UI | No allocation math (only `paper_trading.py` renders the C++ `gross_notional_usd`). | No |
| ETF economics | `etf.data_source`: multiplier 1.0 (definitional), tick 0.01, synthetic never-expiring contract, root `E<ticker>`. | -- |
| Futures economics | `ContractSpec.multiplier` = point value derived from definitions (`reconstitute_root`, ~60 s per root, offline). Forward adjustment is ADDITIVE. | -- |

Conclusion: the platform has accounting and a per-order risk gate, but no
component turns several signals into a sized multi-asset book. Nothing to
merge with; no duplicate to retire (see section 8).

## 2. Design -- five separated components

| Component | Owner | Owns | Never does |
|---|---|---|---|
| Signal Selection | Python `alpha_agent.portfolio.selection` | Which RANKED signals may enter (eligibility policy, mandate, executable domain), the signal's CURRENT direction (sign of its own factor at the as-of date x declared sign), shorting admissibility, exposure-cluster membership | Sizing, weights |
| Risk Model (inputs) | Python `alpha_agent.portfolio.risk_model` | Statistics from real point-in-time bars: annualized vol, joint-panel correlation, median dollar volume, raw reference price, contract economics | Any decision |
| Portfolio Allocation + Risk Budgeting | C++ `quant::construct_portfolio` (`portfolio_construction.hpp`) | Equal risk budget across clusters (ERC on covariance), inverse-vol within a cluster, target-vol scaling, every hard constraint, unit conversion (contracts / shares), post-rounding repair, risk contributions | Choosing signals, estimating statistics, fills |
| Execution Targets | Python `alpha_agent.portfolio.handoff` + C++ `quant_portfolio_plan_replay_csv` | Plan -> integer target units per ROOT at the decision timestamp + a `RiskConfig` from the same constraints -> replay through the unchanged `BacktestEngine` under `PortfolioRiskManager` | New execution semantics |
| Accounting | Existing C++ `PortfolioAccountant` (unchanged) | Fills, positions, exposure | -- |

Python never multiplies a price by a multiplier or rounds a unit: every
notional, unit, exposure and risk contribution in a `PortfolioPlan` is copied
verbatim from the C++ output. A C++ test pins that the allocator's executable
exposure equals the `PortfolioAccountant`'s exposure for the same units and
prices; a Python test replays a plan through the real engine and checks the
same.

## 3. Method `portfolio-construction/1` (Baseline V1)

1. **Select** (Python). RANKED signals only. Eligibility policy:
   `QUALIFIED` (default) = screen outcome `SCREEN_CONTINUE`;
   `EXPLORATORY` (explicit opt-in, labelled on every surface) = not
   `CONTRADICTS_EXPECTED_SIGN` and evidence leaning the declared way
   (aligned t > 0). Ranking decides PRIORITY (which exposures / members
   enter first under the count limits) -- never weight.
2. **Direction only.** At the as-of date (the last common trading day of
   the discovery window), direction = sign(factor value) x declared sign.
   Undefined -> `NO_CURRENT_SIGNAL`; zero -> `SIGNAL_FLAT`; short under a
   no-shorting mandate -> `SHORT_NOT_PERMITTED` (its long/flat form is flat
   today). Screening statistics never become an expected return.
3. **Cluster** = the Phase F exposure group. Alternates share their lead's
   budget; they never get one of their own.
4. **Equal risk budget across clusters** -- equal risk contribution (ERC)
   on the cluster-composite covariance, solved by deterministic cyclical
   coordinate descent.
5. **Within a cluster** each member leg carries an equal standalone risk
   (1/n of the composite's unit).
6. **Volatility normalization** -- a leg's notional is direction / (n x vol).
7. **Covariance** enters twice: ERC across clusters, and the portfolio's
   ex-ante volatility scaled to the target.
8. **Notionals** = weight x capital, per instrument, legs netted.
9. **Constraints** (all scale-downs, hard limits before turnover): shorting,
   unit cap, instrument concentration, liquidity participation, sector gross,
   asset-class gross, gross leverage, net exposure, per-exposure risk share,
   volatility ceiling, then turnover.
10. **Units** -- truncate toward zero (futures contracts, ETF shares), then
    repair any limit truncation can break (net exposure, risk share, vol
    ceiling) by removing one unit at a time from the largest contributor.

Every unit of risk is measured in the same currency (annualized USD
volatility of notional), so a futures contract and a share are never
"one unit" of anything.

## 4. Constraints and where each comes from

| Constraint | Source |
|---|---|
| Domain permission, shorting | Mandate |
| Gross leverage | Mandate `max_gross_leverage`; none stated -> 1.0x (unlevered), labelled |
| Capital | Profile `approximate_capital_usd`; none -> reference capital, labelled |
| Unit cap (futures) | Profile `max_contracts` |
| Liquidity floor | Mandate `liquidity_requirement` -> minimum median dollar volume |
| Target volatility | Policy table by risk style (Conservative / Balanced / Aggressive) |
| Net exposure, instrument / sector / asset-class gross shares, per-exposure risk share, ADV participation, turnover | Policy defaults |

## 5. C++ boundary

Additive only: `portfolio_construction.hpp/.cpp`, CLI
`quant_portfolio_construct_csv <input_dir>` (four CSVs in, one JSON line
out), CLI `quant_portfolio_plan_replay_csv` (plan targets with a
`portplan1:` fingerprint, wired exactly like the Phase 21 paper-trading
helper). The frozen reference CLI, the Phase 11 targets parser, the engine,
execution, Fill and accounting paths are untouched.

## 6. Windows

Everything reads the 2018-2022 discovery window. The as-of date is its last
common trading day; risk statistics use the trailing 252 trading days up to
it. Executing the plan needs the NEXT bar, which lies in the 2023-2024
validation window -- that replay belongs to Phase H, so the real-data run
stops at the handoff files. The handoff itself is proven on synthetic bars.

## 7. Implementation

| Piece | Where |
|---|---|
| Allocator (ERC, limits, units, repair) | `cpp/include/quant_core/portfolio_construction.hpp`, `cpp/src/portfolio_construction.cpp` |
| Allocator CLI (4 CSVs in, one `%.17g` JSON line out) | `cpp/apps/portfolio_construct_csv.cpp` -> `quant_portfolio_construct_csv` |
| Plan replay CLI (`portplan1:` targets + Phase 21 `RiskConfig` row) | `cpp/apps/portfolio_plan_replay_csv.cpp` -> `quant_portfolio_plan_replay_csv` |
| Risk taxonomy (`risk-classification/1`) | `python/alpha_agent/portfolio/classification.py` |
| Policy + constraint resolution with sources | `portfolio/policy.py` |
| Market snapshots, cache, estimator (`trailing-joint-panel/1`) | `portfolio/risk_model.py` |
| Selection (eligibility, direction, priority) | `portfolio/selection.py` |
| C++ bridge (typed JSON mirror, `extra="forbid"`) | `portfolio/allocator.py` |
| `PortfolioPlan` + orchestrator | `portfolio/plan.py` |
| Execution handoff + replay | `portfolio/handoff.py` |
| Agent page card (per event and across events) | `python/alpha_agent/ui/portfolio_plan_view.py`, `ui/views/agent.py` |
| Real slice | `scripts/news_alpha_phase_g_portfolio.py`; demo stage in `scripts/news_alpha_pipeline_demo.py` |

### Plan statuses (every empty outcome is typed)

`CONSTRUCTED` · `NO_EXECUTABLE_POSITION` (every target is below one unit) ·
`NO_ELIGIBLE_SIGNAL` (with counts per reason) · `NO_ALLOCATABLE_SIGNAL` ·
`DEGENERATE_RISK_MODEL` (two exposures hedge each other exactly, so no
equal-risk-contribution solution exists -- reachable since the concurrent
Phase F patch `f495d13` keeps two signal FAMILIES on one instrument in
separate exposure groups; opposite families on one instrument are exactly
such a hedge) ·
`RISK_INPUTS_UNAVAILABLE` (too little joint history, no bar on the as-of
date) · `ALLOCATOR_UNAVAILABLE` (C++ not built -- there is deliberately no
Python fallback). A malformed allocator input raises: it is a pipeline bug,
never a property of the portfolio.

### Rejection reasons (every ranked-set signal is accounted for)

Selection: `EXCLUDED_BY_MANDATE`, `NOT_RANKABLE`, `NO_SCREEN_SUPPORT`,
`CONTRADICTS_DECLARED_SIGN`, `EVIDENCE_AGAINST_DECLARED_SIGN`,
`ALTERNATES_EXCLUDED`, `DOMAIN_NOT_EXECUTABLE`, `NO_RISK_CLASSIFICATION`,
`NO_MARKET_DATA`, `NO_SCREEN`, `NO_CURRENT_SIGNAL`, `SIGNAL_FLAT`,
`SHORT_NOT_PERMITTED`, `EXPOSURE_LIMIT`, `EXPOSURE_MEMBER_LIMIT`.
Allocation (C++): `INSTRUMENT_NOT_ADMITTED` (with the instrument's own reason:
`DOMAIN_NOT_ALLOWED`, `NON_POSITIVE_PRICE`, `INVALID_ECONOMICS`,
`NO_VOLATILITY`, `LIQUIDITY_NOT_MEASURED`, `BELOW_MIN_LIQUIDITY`),
`CLUSTER_NETS_TO_ZERO` (one exposure's own signals cancel on one
instrument), `DEGENERATE_RISK_MODEL`.

### Decisions worth reviewing

1. **Default eligibility is QUALIFIED** (screen outcome `SCREEN_CONTINUE`).
   EXPLORATORY is an explicit, labelled preview (aligned t > 0, never a
   contradicted signal) so the allocator can be seen on real data while no
   signal has screening support. Its identity is part of the plan.
2. **A short under a no-shorting mandate is flat** for that date (its
   long/flat form), not sized and not redistributed.
3. **Freed risk is not redistributed.** A clipped position's budget is not
   handed to the others; the ex-ante volatility then sits below target and
   the plan says so.
4. **Per-exposure risk cap is absolute** (share x target volatility), so a
   one-exposure book runs at 50% of the target by default -- concentration
   costs risk budget rather than being silently accepted.
5. **Truncation toward zero** for units, plus a one-unit-at-a-time repair for
   the three limits truncation can break. A futures contract that is larger
   than its target is reported (`below_one_unit`, with one unit's risk) rather
   than rounded up.
6. **Liquidity fails closed**: an instrument whose traded notional cannot be
   measured is not admitted when a liquidity rule applies. ETF volume is the
   primary venue's only (a lower bound, ~27% of consolidated).
7. **Unit caps in the execution gate** are set to the plan's own largest
   position: `RiskConfig`'s unit caps count a contract and a share alike, so
   the gate's real limits are the USD-based gross/net leverage.
8. **The as-of date is the last common discovery day**; risk statistics use
   the trailing 252 joint trading days to it. The handoff is written for real
   plans but not replayed (the next bar is validation data).

## 8. Cleanup audit (section 10 of the prompt)

| Looked for | Found | Action |
|---|---|---|
| Position-sizing helpers | None: DSL `target_units` are fixed integers; `ml.execution` states a probability never becomes a size | none |
| Portfolio wrappers | `paper.risk_policy.PaperRiskPolicy` (typed `RiskConfig`) -- REUSED by the handoff, not duplicated; `paper.engine` runs registry-eligible experiments, a different object | none |
| UI-side allocation math | None; `ui/views/paper_trading.py` only renders the C++ `gross_notional_usd`. The new view is statically guarded (no sort/min/max/round, no multiplication) | none |
| Experimental scripts | None size positions (`phase6_uso_split_demo`, `phase9_1_equity_split_synthetic_demo` mention notional only to explain split continuity) | none |
| C++ accounting | `PortfolioState.correlation_adjusted_gross_exposure_usd` is still a documented TODO equal to gross; correlation now lives in the allocator's risk model, not in accounting | kept (frozen accounting path) |

No proven dead duplicate exists, so nothing was deleted. Execution and
accounting code used by historical strategies is untouched.

## 9. Tests

- C++ `quant_portfolio_construction_tests` (23 cases): futures multiplier vs
  ETF shares, truncation and below-one-unit, inverse-vol legs, ERC
  (uncorrelated and correlated, contributions equal to 1e-9), short
  restriction, gross leverage, instrument concentration, liquidity floor /
  unmeasured / participation,
  sector and asset-class caps, per-exposure risk cap, unit cap (exact at the
  cap), volatility ceiling after a broken hedge, domain permission, net
  exposure rounding repair, turnover (binding and overridden by a hard
  limit), nets-to-zero refusal, malformed inputs (not PSD, missing pair,
  direction, unknown instrument, duplicate key, no leverage cap, rho > 1),
  a degenerate risk model (incl. two families on one instrument) as a typed
  status, permutation invariance, Euler contributions summing to
  volatility, and the allocator's exposure == `PortfolioAccountant`'s.
- Python `test_news_alpha_phase_g.py`: constraint sources, taxonomy, empty
  qualified plan, exploratory labelling, direction from the factor sign on the
  as-of date (and a negative declared sign), no-shorting => flat, rank =
  priority not weight, alternates share budgets / equal exposure risk, units
  per asset class, byte-identical C++ re-run, a static guard that no plan or
  handoff module multiplies by a price/multiplier/capital, liquidity refusal,
  determinism, missing allocator, mandate mismatch, exact futures returns
  across a roll, risk-model refusals (validation window, too few
  observations, synthetic data), discovery-only snapshots, the snapshot
  cache, and the replay through the real engine (units and exposure equal to
  the plan; a tighter gate resizes).
- UI `test_news_alpha_phase_g_ui.py`: the empty qualified card, the explicit
  load button, a C++-sized plan rendered from cached snapshots, the
  exploratory switch, cards per event and across events, and a static guard on
  the view.

## 10. Real vertical slice (Phase F's three events, $0)

`PYTHONPATH=python python scripts/news_alpha_phase_g_portfolio.py` -> 
`outputs/news_alpha/phase_g/THREE_EVENTS__portfolio_plan.json` (+ handoff CSVs
under `handoff/`). Already-acquired bytes only; snapshots cached after the
first run (182 s cold, 2 s warm). Default profile: Balanced (10% target), no
shorting, no leverage cap stated (-> 1.0x), no capital stated (-> $1M
reference). As of 2022-12-30; risk window 2021-12-31 -> 2022-12-30 (252 joint
returns).

| Plan | Outcome |
|---|---|
| Qualified | `NO_ELIGIBLE_SIGNAL` -- 0 of 38 (37 no screen support, 1 contradicts its declared sign). The honest default. |
| Exploratory, no shorting | 6 signals, 4 exposures -> XLF 3,688 sh, LQD 1,828 sh, GLD 566 sh, ZN 1 contract (ZNH3), SHY 3,918 sh. Ex-ante vol 6.4% vs 10% target, gross 0.85x. Binding: instrument concentration (SHY 61% -> 40% of capital) and gross leverage (1.26x -> 1.0x). GC dropped: its $105k target is below one $183k contract, so gold is held through GLD only. 9 signals flat because they point short. |
| Exploratory, shorting allowed | 11 signals, 4 exposures -> XLF long, TLT short 1,321 sh, ZN short 2, USO short 832 sh, GLD long. Vol 5.1%, gross 0.73x. LQD 20D (short) and LQD 60D (long) cancel -> `CLUSTER_NETS_TO_ZERO`; SHY 60D/20D also net to zero inside the rates exposure; CL below one contract. |

Reference prices match the real 2022-12-30 closes (XLF 34.29, TLT 99.62,
ZNH3 112.20, GLD 169.70). What the slice teaches: an unlevered mandate cannot
reach a 10% target with bond-heavy exposures (the leverage cap binds first),
and at $1M one futures contract is coarse -- the plan says so per instrument
("one unit alone would carry 2.8% of capital in annual volatility") instead of
rounding up.

## 11. Verification

- C++: `quant_portfolio_construction_tests` 128 checks; ctest 16/16.
  12 negative controls (round-to-nearest, ERC stopped early, no net repair,
  no volatility ceiling, no liquidity floor, no unit nudge, notional without
  multiplier, no canonical order, no short admission, no sector caps, no
  volatility normalization, no instrument concentration) -- all caught, each
  at its own assertion. The last one was NOT caught at first (no test
  exercised the concentration cap) -> `test_instrument_concentration` added.
  Mutation runs force an object rebuild: GNU Make 3.81 compares whole-second
  mtimes and silently reused stale objects in a first attempt.
- Python: 12 negative controls (declared-sign flip, handoff units, liquidity
  floor, roll-exact returns, short restriction, qualified gate, leverage
  default, union-of-days panel, forward-filled panel, allocator rejections not
  mapped, ...) -- all caught; the union/forward-fill mutations were NOT caught
  until `test_a_day_one_venue_missed_is_absorbed_never_filled` was added.
- Live UI: a real `streamlit run` (API keys blanked -- no paid call possible)
  + Playwright screenshots of every tab on the real cached data; fixed from
  them: misleading delta arrows on non-delta metrics, "Us" casing, a truncated
  liquidity column, "0.000%" for ZN's participation, GC shown "Allocated"
  while not held.
