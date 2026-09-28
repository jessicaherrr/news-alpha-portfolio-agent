# News Alpha Phase H -- Backtest, Scientific Validation, and Signal-Path Memory

    ... -> Multi-Asset Signal Ranking (RankedSignalSet)        (Phase F)
        -> Portfolio Construction (PortfolioPlan)               (Phase G)
        -> VALIDATION GATE                                       (Phase H, this document)
        -> frozen portfolio strategy -> causal rebalance schedule
        -> C++ backtest (the unchanged engine, hard risk gate)
        -> the existing validation plane + frozen ReliabilityPolicy
        -> registry: experiments + signal-path evidence (schema v7)
        -> Alpha Memory (read model) + Hypothesis<->Evidence plane links

Phase H closes the research loop. It adds no backtester, no validation
framework, no registry and no memory database: every stage reuses the
existing one and extends it only where the portfolio shape required it.

## 1. Reuse audit

| Existing system | Reused as | Extended? |
|---|---|---|
| C++ `BacktestEngine` / execution / Fill / `PortfolioAccountant` | every economic number of every run | no |
| `PortfolioRiskManager` (hard gate), `quant::run_paper_trading_backtest` | the gate every order passes | `PaperTradingRunConfig.corporate_actions` pass-through (empty = byte-identical) |
| `quant_portfolio_plan_replay_csv` (Phase G) | the replay CLI | optional validation-day / roll-mark / splits files, `--end-of-test`, the reference CLI's PnL fields + `daily_equity` trace |
| Phase G `construct_portfolio_plan`, selection, risk model, C++ allocator | each rebalance IS a Phase G plan as of that day | risk model / snapshot `window` (DISCOVERY default, EVALUATION for Phase H); `FactorSource` protocol |
| Phase E loaders + `evaluate_expression` | member factors through 2024 | none (new caller) |
| `validation.assemble_report`, `evaluate_policy`, CBB null, bootstrap, BH/FDR, DSR, walk-forward, regime labeller | the validation plane | none |
| Phase 13.5C frozen split / walk-forward / null / bootstrap / policy | the protocol | none |
| `ExperimentRegistry` | experiments of validated portfolios; hypothesis evidence | schema v7: one additive table `signal_path_evidence` |
| `alpha_memory` (read model) | route memory | `signal_path_evidence` builder + `signal_path_memory` read model |
| `alpha_graph` (evidence plane) | Instrument nodes | `hypothesis_links` (ids only) |
| `holdout.lifecycle` | holdout status | read-only |

## 2. The tested portfolio

A `PortfolioPlan` is one allocation on one day. Holding it for two years would
test a basket (and produce a handful of fills). Phase H tests the portfolio
the plan DEFINES (`portfolio.strategy`):

* **Frozen before any validation data is read** (`PortfolioStrategySpec`,
  `portstrat1:`): members = the ranked set's eligible signals under the plan's
  policy (Phase G stage 1, never re-selected later), the plan's policy,
  constraints and mandate, and `RebalanceRule` (every 20 common trading days
  -- a convention, fingerprinted).
* **Re-derived causally at each rebalance day t**: Phase G
  `construct_portfolio_plan(as_of=t)` -- direction from each member's own
  factor on t, the risk model from the trailing window ending t, sizing by
  the C++ allocator. A day nothing can be sized is a typed decision to be
  flat. A test proves the decision on t is unchanged when every input after t
  is removed.
* **Member factors through 2024** (`portfolio.evaluation`): the same
  expression through the same Phase E loader and feature engine. Its
  discovery prefix must equal the Phase E screen value for value
  (`LookAheadDefect` otherwise); the series is cut at 2025-01-01 even if a
  measurement's coverage runs later.

## 3. The input gate (a plan is not eligibility)

`portfolio.validation.gate_portfolio` -- reads no market data, safe on every
render. The authority is the existing one: Phase E screen `SCREEN_CONTINUE`,
carried by the Phase F ranked set and Phase G's QUALIFIED policy, re-checked
against every member's actual screen. Typed refusals (all of them, not the
first): `PLAN_NOT_CONSTRUCTED`, `EXPLORATORY_PLAN`,
`MEMBER_WITHOUT_SCREEN_SUPPORT`, `MEMBER_SCREEN_MISSING`,
`NO_ELIGIBLE_MEMBER` -> `NO_ELIGIBLE_PORTFOLIO`. An exploratory preview is
never eligible. Then the execution gate (below) -> `EXECUTION_UNSUPPORTED`.
A refused portfolio's loaders are never called (tested with loaders that
raise).

## 4. Execution (the unchanged C++ engine)

`portfolio.execution` builds the engine inputs and replays each schedule:
futures = native 1-minute raw-contract GLBX bars + definitions + roll close
marks (fills resolve to the contract traded); ETFs = primary-listing daily
raw bars + the one-share contract + sourced splits (applied by the engine).
The engine force-liquidates at the end, emits the trading-day equity trace,
and the SAME `validation.runner.backtest_run_from_cpp_json` turns it into
the daily return series. Python never computes a fill, cost or PnL (the
Phase G AST guard covers every Phase H module).

* **Costs** (acceptance patch, section 12). Every root carries its own
  commission, charged by the C++ engine
  (`ExecutionConfig::commission_per_contract_usd_by_root`): FUTURES $2.00 per
  contract, ETF $0.005 per share -- both DECLARED research assumptions, not
  sourced broker schedules. Stressed x1.5 / x2.0. A root without a declared
  rate refuses the portfolio (`COST_MODEL_INCOMPLETE`); a book whose roots
  follow different trading-day conventions (CME session vs US-equity day) is
  `EXECUTION_UNSUPPORTED`.
* **ETF identity.** XNAS.ITCH reassigns an instrument id (a locate code)
  almost daily -- QQQ, SHY, IEF and TLT carry 574-839 vendor ids over
  2018-2024 (found on real bytes; the Phase 6 pilot only ran single-id XLE).
  The ticker is the identity: each ETF is one deterministic engine id
  (`etf_engine_instrument_id`, 4,000,000,000 + index); raw prices untouched.
* **Trading days.** One boundary per session day across roots: the latest
  root's last bar (`merged_day_plan`); roots with different conventions are
  refused; a label inside the holdout raises.
* **Hard gate.** The replay's `RiskConfig` comes from the same constraints
  (`handoff.hard_gate_policy`, now shared by Phase G's handoff and Phase H):
  unit caps non-binding by construction, USD gross / net leverage and the
  drawdown kill switch binding.

## 5. Validation (`portfolio-validation/1`)

The frozen Phase 13.5C protocol wherever it applies: TRAIN 2018-2022 /
VALIDATION 2023-2024 / LOCKED_HOLDOUT 2025 (never loaded); headline = the
validation window; 4 expanding walk-forward folds inside TRAIN; costs x1.0 /
x1.5 / x2.0; gating null = centered block bootstrap; bootstrap CIs; a
predeclared rebalance neighbourhood (10 / 40 days around 20), each its own
trial; BH over the report's own trial family (canonical + predeclared
neighbours -- the Validation Engine's canonical per-report family); DSR over
the same family; causal
volatility regimes of the portfolio universe (TRAIN-only cut points,
reported, not gating); verdict = the frozen policy's `evaluate_policy` with
parameter stability required. `VALIDATED` / `REJECTED` /
`INSUFFICIENT_EVIDENCE`; `PRIOR_RESULT_CITED` when the registry already holds
a valid result for the identity (not re-run). The holdout stage is always
`HOLDOUT_NOT_RELEASED` -- a final evaluation goes only through the existing
holdout-release authority.

**Scope.** A result applies to the tested portfolio only. The report, the
strategy spec and every memory record say so; nothing marks a member signal
validated, a path proven or a mechanism established.

## 6. Registry and memory

* **Experiments** (`portfolio.registry_record`): a validated portfolio
  records its canonical trial + neighbours through `apply_bundle`, identity
  by the existing formula: `strategy_family = news_alpha_portfolio`,
  `root_symbol` = the members joined by `+`, the portfolio's one
  `asset_domain`.
* **Schema v7** adds `signal_path_evidence` -- PURELY ADDITIVE (nothing
  renamed, copied or altered; an empty table keeps the v6 content digest).
  One typed record per hypothesis of a research run at the furthest stage it
  reached: `stage_reached` (SIGNAL_PATH ... VALIDATION), `outcome`
  (FAILURE / SUCCESS / UNRESOLVED), `evidence_scope` (HYPOTHESIS / SCREENING /
  PORTFOLIO), a typed `reason_code`, the full lineage ids (event, mechanism
  graph, path + signature + type + depth, expression + fidelity + domain,
  measurement + status, candidate + factor identity, plan + strategy
  fingerprints, experiment identity). `SUCCESS` exists only at PORTFOLIO scope
  (the model refuses anything else); content-fingerprint ids make recording
  idempotent; append-only; holdout-guarded; the UI boundary is statically
  guarded never to write it.
* **Alpha Memory** (`alpha_memory.signal_path_memory`, a read model): per
  route signature, across every event and run -- furthest stage, outcomes,
  scopes, reasons, split by expression domain, linked experiments, and typed
  `MemoryGuidance` (RETRY_WHEN_DATA_EXISTS, NEEDS_EVENT_CONDITIONING,
  SCREEN_EVIDENCE_AGAINST, AWAITING_VALIDATION, PORTFOLIO_EVIDENCE_FOR/AGAINST,
  ...). `blocked` is always False: a failure belongs to the route, expression,
  measurement and candidate it happened at, never to a family, event type or
  mechanism. `lineage(record)` renders the Event -> ... -> Evidence chain.
  The Agent page recalls memory by the current event's ROUTE signatures, so a
  newly described event sees every recorded run of its routes, from any event.
* **Alpha Graph** (`alpha_graph.hypothesis_links`): the Hypothesis Plane
  links to Evidence-Plane Instrument nodes and registry experiments by id
  only, in both directions -- no verdict, coverage or maturity crosses.

## 7. Path-level research questions and baselines A-E

`news_alpha.study_design` declares the questions (DIRECT_PRICES_FASTER,
INDIRECT_DECAY, MECHANISM_VS_SENTIMENT), the arms A-E (raw sentiment, direct
LLM call, event x exposure, + confirmation, full transmission) compared under
one protocol and one predeclared BH family per design, and each arm's typed
requirements. `probe_study_capabilities` reads the real stores and the
pipeline's own conditioning. Today every design is **NOT YET TESTABLE**: 0
dated events or news items inside 2018-2024 (the market-intelligence stores
hold 2026 items), exposure magnitude UNKNOWN, confirmation is the
unconditional price response, transmission weight NOT_DERIVED -- and arm B
has its own blocker: the platform's LLM was trained after 2024, so its call
on 2018-2024 news can encode the outcome (testable only prospectively).
`EventConditionedEvidenceRef` refuses unconditional factor diagnostics as
event-conditioned evidence. No answer is assumed.

## 8. Real vertical slice (Phase G's three events, $0)

`PYTHONPATH=python python scripts/news_alpha_phase_h_validation.py` ->
`outputs/news_alpha/phase_h/THREE_EVENTS__validation_and_memory.json`.

| Stage | Outcome |
|---|---|
| Qualified plan | `NO_ELIGIBLE_SIGNAL` -- 0 of 38 signals (37 no screen support, 1 contradicts its sign) |
| Validation gate (qualified) | **`NO_ELIGIBLE_PORTFOLIO`** (`PLAN_NOT_CONSTRUCTED`, `NO_ELIGIBLE_MEMBER`). No validation-window value read. |
| Validation gate (exploratory preview) | refused: `EXPLORATORY_PLAN`, `MEMBER_WITHOUT_SCREEN_SUPPORT` (it also mixes ETFs with ZN -> would be `EXECUTION_UNSUPPORTED`) |
| Holdout | `HOLDOUT_NOT_RELEASED` (lifecycle SEALED, accessed: no) |
| Evidence recorded (registry v7) | 264 records, one run: FAILURE 192 / UNRESOLVED 72; scopes HYPOTHESIS 120 / SCREENING 144; reasons SCREEN_NO_SUPPORT 140, CONFIRMATION_ONLY 72, MEASUREMENT_UNAVAILABLE 29, NO_INSTRUMENT 19, SCREEN_CONTRADICTS_SIGN 4 |
| Route memory | 47 routes (25 cross-sector, 15 direct, 7 supply-chain); furthest stage SCREEN 31, MEASUREMENT 11, ASSET_EXPRESSION 5. On the same routes ETF (31) and futures (16) expressions reached screening while EQUITY expressions stopped at measurement (21) or expression (14) -- missing equity bars, recorded as RETRY_WHEN_DATA_EXISTS, not as evidence against them |
| Path studies | all three NOT YET TESTABLE |
| Engineering replay (2022 discovery bars, ETF-only exploratory book -- NOT evidence) | 13 rebalances, 91/91 targets applied, 55 fills, 28 closed trades, costs $587.32, net -$6,216.54 on the $1M reference, 251 aligned days; the hard gate resized 3 and rejected 2 orders (all `max_gross_leverage`: the allocator sizes at the 1.0x cap at the close, the engine re-checks at the next open); final equity = start + net PnL exactly |

The honest scientific outcome: no portfolio from these events has earned the
right to spend the validation window. The first thing that would change it is
screening support for a signal; the second, real equity bars for the equity
expressions; the third, an event-conditioned history for the path studies.

## 9. Cleanup audit

| Looked for | Found | Action |
|---|---|---|
| Duplicate boundary-file readers (C++) | `read_bars` / validation-days / roll-marks / splits / risk-config copied in 3 CLIs | Phase G replay CLI now uses the new shared `detail/replay_csv.hpp`; the frozen reference CLI and the Phase 21 CLI keep their byte-stable copies (debt, below) |
| Duplicate hard-gate construction | Phase G `build_handoff` and the new schedule replay | one `handoff.hard_gate_policy` (Phase G handoff files verified byte-identical) |
| Replay wrapper | `replay_handoff` | kept as a thin wrapper over the new `run_replay_cli` |
| UI-side calculations | the memory card's "main reason" | moved to `SignalPathMemory.primary_reason`; the view is statically guarded (no sort/min/max/arithmetic, no run, no write) |
| Private cross-module helper | `portfolio_plan_view._cached_only` | public `cached_snapshot` |
| Dead code (scan of every top-level def in news_alpha / portfolio / screening / ranking / alpha_memory / alpha_graph / UI views) | `news_alpha.universe.domain_label` -- a one-line wrapper over `DOMAIN_LABELS`, zero references in code, tests or docs | deleted |
| Superseded proposal | `docs/PHASE_10_SIGNAL_PATH_AND_PORTFOLIO_CONSTRUCTION_PROPOSAL.md` (untracked, cited by the accepted Phase A doc) | archived in place with a SUPERSEDED banner, body verbatim (moving it would break the accepted doc's link) |
| Stale scripts | the Phase E-H scripts and the demo are all current | none |
| Duplicate schemas A-G | `MandateDomain` vs registry `AssetDomain`, transmission `EconomicSector` vs risk sectors, two `handoff` modules -- deliberately separate concepts | none |

Nothing touching accepted scientific history, registry provenance or
accepted docs was deleted.

## 10. Decisions worth reviewing

1. **The tested object** is the plan's portfolio STRATEGY (frozen eligible
   members + policy + a 20-day rebalance rule), not the single-date book.
2. **Mixed-unit books**: resolved by the acceptance patch (section 12) --
   per-root commissions in the C++ execution config. What still blocks a
   mixed futures + ETF VALIDATION is the trading-day convention: one daily
   return across a CME session and a US-equity day is a validation decision
   not made here (typed `EXECUTION_UNSUPPORTED`).
3. **Registry representation**: a portfolio experiment's `root_symbol` is the
   `+`-joined member label and its domain the book's one domain; no real
   portfolio experiment exists yet.
4. **ETF commission** $0.005 per share is a declared convention (the Phase 6
   pilot's $2 per unit is not a cost model for thousands of shares).
5. **BH family** = the report's own trials (canonical + predeclared
   neighbours), the Validation Engine's canonical per-report rule. The
   original Phase H pooling of every recorded portfolio sharing a validation
   configuration was a new rule, removed by the acceptance patch; family
   membership across research generations stays owned by the Validation /
   Registry research-generation semantics.
6. **Schedule-shift / circular-permutation nulls** (diagnostic-only in the
   framework) are not run for portfolio schedules; the gating null runs.
7. **Evidence outcome taxonomy**: data gaps are HYPOTHESIS-scope FAILURE
   (retry when data exists); mandate exclusions, missing signal rules and
   confirmation-only measurements are UNRESOLVED.
8. **Schema v7** additive migration; tests pinning the version moved 6 -> 7.

## 11. Tests and verification

* C++: `quant_paper_trading_run_tests` +8 checks (split pass-through equals
  the hand-wired engine; empty config unchanged) -> 35; ctest 16/16.
* Python: `test_news_alpha_phase_h.py` (27) + `test_news_alpha_phase_h_ui.py`
  (6): gate refusals never load data, causality by truncation, no-short
  schedules, holdout cut and look-ahead prefix, ETF ids, merged day plans,
  off-bar targets, hard-gate caps, a full validation through the real
  binaries on 2018-2024 fixture bars (C++-sourced daily PnL, costs, folds,
  neighbours, BH/DSR, regime, holdout untouched, determinism), registry
  recording + PRIOR_RESULT_CITED, schema v7 additivity + digest, evidence
  accounting, portfolio-scope-only success, no blacklisting, cross-event
  memory, plane links, study readiness, and the page (gate, memory, studies,
  no run button, pure renderer).
* 13 negative controls: 12 caught; the one not caught
  (`parameter_stability_required=False`) is unobservable by design -- a
  neighbourhood is always supplied -- and is kept as defence in depth. Three
  were initially NOT caught and led to stronger tests (the 2025 cap shadowed
  by coverage; the day-plan max hidden by equal timestamps). The mutation
  loop itself exposed a bytecode-cache trap (a same-second, same-size restore
  kept the mutated `.pyc` running) -- now run with
  `PYTHONDONTWRITEBYTECODE=1`.
* Real data: the vertical slice above; the engineering replay on real bytes
  found and fixed the XNAS id issue and an ETF calendar key mismatch before
  any test did.
* Full suite (sharded per file, C++ built): 3,138 passed, 23 skipped (the
  optional pybind module and data-dependent ETF files, as before), 2 failed --
  one a schema-version pin (6 -> 7) missed by the first sweep, fixed and
  re-run green; the other the known pre-existing
  `test_phase_b1_streamlit_ui::test_research_details_shows_verdict_promise_fit_row`.
* Live UI: a real `streamlit run` (API keys blanked) driven by Playwright at
  desktop and phone width, zero exceptions. It found two UX defects the
  AppTest suite could not: memory was recalled by event id (a newly
  described event saw nothing) -- now by route signature; and the guidance
  column led with another reason's advice -- now the primary reason's.
* Registry hygiene: the first real recording used a doubled id prefix
  (`spevidence1:spevidence1:`); with the user's approval those 264 rows (one
  run, the table's only rows) were deleted and the run re-recorded with the
  corrected ids; a regression test pins the format.

## 12. Acceptance patch -- explicit multi-asset commission schedules

The authoritative engine charged ONE scalar commission per unit for every
instrument -- right for a book of one unit kind, wrong for a book of futures
contracts and ETF shares. The fix extends the existing execution authority;
there is no Python commission calculator and no second execution path.

* **C++ (additive).** `ExecutionConfig::commission_per_contract_usd_by_root`
  (`std::map<root, USD per unit>`), resolved in the ONE place a commission is
  computed (`BarExecutionSimulator::execute`), through the contract's own
  `ContractSpec::root_symbol`: every raw contract of a futures root -- entry,
  roll close-leg, re-open -- shares one rate; an ETF is its `E<ticker>` root and
  a unit is one share. A root not in the map is charged the scalar. EMPTY map
  (every pre-existing caller) => the legacy expression is evaluated unchanged.
  `PaperTradingRunConfig` passes the map through (finite, non-negative);
  `quant_portfolio_plan_replay_csv --commission-schedule=<csv>`
  (`root_symbol,commission_per_unit_usd`) REFUSES a schedule missing any traded
  root and echoes the schedule it charged. The frozen reference CLI,
  `targets_run` and pybind stay scalar-only and untouched.
* **Python.** `portfolio.execution` holds a per-domain `CommissionConvention`
  with provenance (`CommissionBasis.DECLARED_RESEARCH_ASSUMPTION` vs
  `SOURCED_EXECUTION_METADATA`); `cost_plan_for` builds one explicit
  `RootCommission` per root into the existing `CostStressPlan`
  (`commission_schedule`, stressed per MULTIPLIER scenario). A root without a
  convention => `COST_MODEL_INCOMPLETE`, before any data is read. Provenance:
  the schedule enters the cost plan's identity (and so the ValidationSpec
  fingerprint), the registry `cost_config_identity`, the report
  (`commission_schedule`, with basis and source), and every run summary
  (the rates the engine echoed back).
* **Legacy byte-equivalence.** Both new fields are dropped from their
  fingerprint payloads when absent: `DEFAULT_COST_STRESS_PLAN` / the frozen
  13.5C plan and the scalar `cost_config_identity` are pinned to their
  pre-patch hashes. A schedule charging every root the scalar reproduces the
  legacy replay result exactly; all 16 pre-existing ctest suites unchanged.
* **Mixed example** (Phase G fixture book, NQ + three ETFs, real replay CLI):
  NQ $2.00/contract, ETFs $0.005/share -> $195.55 of commission; the old
  scalar $2.00 charged the same book $11,986.00 (every share at a contract
  rate).
* **Not changed:** slippage, spread, fill timing, rolls, corporate actions,
  allocation, risk limits, validation / registry verdict / memory / holdout
  semantics. The 264 recorded signal-path evidence rows are unaffected (the
  real run was `NO_ELIGIBLE_PORTFOLIO`; re-running records nothing new).
* **Semantic review.** (A) `EligibilityMode.QUALIFIED` is screen-supported
  eligibility only: the gate passes such a plan on, and until a validation
  runs its members stay UNRESOLVED screening evidence -- tested. (B) the BH
  family is the Validation Engine's per-report family; the Phase H pooling
  rule was removed (decision 5).
* **Tests.** C++ `quant_commission_schedule_tests` (23 checks: legacy
  equivalence, per contract, per share, unlisted root, mixed engine run, roll,
  determinism, negative rate); ctest 17/17. Python: typed refusals, the mixed
  book through the replay CLI, uniform-schedule parity, the CLI's refusal of an
  incomplete schedule, pinned legacy identities, labelled assumptions, a static
  guard that no portfolio module multiplies by a commission, the schedule in
  the recorded cost identity, and the per-report BH family. 9 negative
  controls, all caught.

