# News Alpha Portfolio Agent — Phase D: Asset Expression + Measurement Resolution

Status: **IMPLEMENTED, TESTED, acceptance patch applied (§11) — stopped at review gate.** Quantitative Candidate Signals are not started.

Phase D extends the pipeline by three stages:

```
... → Economic Mechanism Graph → Signal Paths → Economic Consequence          (Phases B–C)
    → Asset Expression → Measurement Spec → PIT Data Field Resolution          (Phase D, new)
```

It answers three questions for every consequence a signal path reaches:

1. **Where** could it be expressed in a tradable market?
2. **How** would each expression be measured?
3. **Which** of those measurements did we actually have, and could we have known them, at the historical point in time?

Run it:

```
PYTHONPATH=python python scripts/news_alpha_pipeline_demo.py --no-cached      # offline vertical slice
streamlit run python/alpha_agent/ui/app.py                                    # Agent → News Impact Triage → a card
```

Nothing in this phase names a weight, size, return, probability or trade. A pressure sign on an expression is a hypothesis to test.

## 1. Reuse audit

| Searched for | Found | Decision |
|---|---|---|
| Registry `AssetDomain` | FUTURES / ETF / EQUITY evidence namespace | Untouched (widening it is a frozen-semantics change). Phase D uses Phase A's `MandateDomain`, which maps onto it by value. |
| Domain support | `news_alpha.universe.DomainSupport` + `AllowedAssetUniverse` (Phase A) | **Reused** as the only domain-capability authority. Options = `NOT_SUPPORTED`, Crypto = `SYNTHETIC_ONLY`, Equity = `DATA_NOT_ACQUIRED`. |
| Filesystem capability snapshot | Phase A `DomainCapabilities` (ETF / Equity "acquired" flags) | **Extended additively** with `futures_bars: tuple[FuturesBarCoverage]`, read from the committed `data/catalog/real_cme_dataset.json`. One snapshot for the whole pipeline, not a second one. |
| `translation.MeasurableVariable` | Phase 1's ideal measurable variable (`point_in_time_available: bool \| None`) | **Reused verbatim** as `MeasurementSpec.variable`. A validator forces its PIT fields to equal the resolution's. |
| `translation.researchability.classify_factor` | The authority for "does the FeatureRegistry express this?" | **Reused** to decide executability, against the live `FeatureRegistry` (plus each kind's own `point_in_time_safe` flag). |
| `equities.data_availability` | Phase 9.1 findings for PIT fundamentals and analyst consensus | **Reused**: fundamentals and estimate-revision measurements resolve from the recorded finding's availability enum, never a re-assertion. |
| ETF / Equity dataset windows | `etf.universe.DATASET_WINDOWS`, `equities.universe.DATASET_WINDOWS` + `primary_listing_dataset` | **Reused** for dataset, window and venue. |
| `crypto.provenance.DataProvenanceRole` | REAL / SYNTHETIC closed enum | **Reused** as the resolution's `data_role`; SYNTHETIC can only resolve as `DOMAIN_UNAVAILABLE`. |
| `features.macro`, `features.carry`, `features.crossmarket` | Typed PIT-alignment / carry / cross-market interfaces with no registered kind | Cited in PIT rules and transforms; they are why COT is "connector not built", and why term structure and relative strength are "not executable". |
| `marketdata.capability.CapabilityState` | Live observation state per futures product (UI) | Different question (live quotes, not history). Untouched. |
| Phase C `EXPOSURE_BASIS` | Exposure → economic states bridge | Not duplicated. The new expression rules are **checked against it at import**: every ETF/Futures exposure's instruments must be expressed by a state its basis rests on. |

**Deleted: nothing.** The search found no dead duplicate. Each availability vocabulary answers a different question: triage availability (`AvailabilityStatus`), platform support (`DomainSupport`), live observation (`CapabilityState`), external-source investigation (`DataSourceAvailability`), and factor expressibility (`ResearchabilityStatus`). Phase D composes them rather than adding a sixth.

## 2. Domain honesty (discovered from the repo)

| Domain | What exists | How Phase D treats it |
|---|---|---|
| Futures | 36 catalogued roots; research bars acquired for **ES, NQ, CL, GC, ZN** only (GLBX.MDP3 ohlcv-1m, 2018-01-01 → 2024-12-31, plus raw contracts around each roll) | Real. HG, NG, MNQ, ZT… are catalogued and expressible but resolve `MISSING` (not acquired). |
| ETF | 14 pilot tickers, daily primary-listing bars acquired (2018-05-01 → 2024-12-30) | Real. Returns are price-only (no sourced distributions), volume is one venue's. Both are recorded as proxies. |
| Equity | 20 declared names; **no bars acquired** | Expressions continue, but every measurement is `MISSING`. None is runnable. |
| Options | No universe, chain or pricing data | `DOMAIN_UNAVAILABLE`. Never runnable. |
| Crypto | Synthetic scaffold only (CME crypto futures are catalogued under Futures, never acquired) | `DOMAIN_UNAVAILABLE`, `data_role=SYNTHETIC`. Never runnable, never promoted. |

## 3. Asset Expression (`news_alpha/asset_expression.py`, `expression-admission/1`)

`AssetExpression` = one way one consequence could be expressed in one domain. It carries:

- the consequence and every path reaching it;
- the domain, the expression concept ("AI accelerator designers"), and its form (`UNDERLYING` / `SINGLE_NAME` / `SECTOR_BASKET` / `BROAD_INDEX`);
- the **relation**: does the instrument move with (+) or against (−) the consequence, or `AMBIGUOUS` (capacity spending is the supplier's revenue and the spender's cost);
- per-horizon **pressure**: consequence direction × relation, for example Treasury yields ↑ × (−) → pressure ↓ on Treasury futures;
- a status, its in-scope instruments, `execution_capable`, and the measurements behind it.

**Economic relevance is not a user-allowed expression.** The expression set comes from the reviewed library alone, so its ids are identical under every mandate (tested). The mandate only decides the status:

| Status | Meaning |
|---|---|
| `CONTINUES` | At least one in-scope instrument. Its measurements are resolved. |
| `EXCLUDED_BY_MANDATE` | The domain is not allowed. The row stays visible, but no instrument is enumerated and nothing is measured. |
| `DOMAIN_UNAVAILABLE` | Allowed, but the platform has no real data (Options, Crypto). Measured, so you can see how it *would* be measured, and every measurement is `DOMAIN_UNAVAILABLE`. |
| `NO_INSTRUMENT` | A real tradable concept with no member in the declared universe (HBM makers, copper miners). Never filled with a nearest-looking ticker. |
| `REMOVED_BY_CONSTRAINTS` | The mandate's allow/deny list removed every instrument that carries it. |

`execution_capable` is true only for a continuing expression with a certified Futures root or an acquired ETF. A validator enforces this, so no Options, Crypto or Equity-foundation expression can ever claim it.

**Shorting.** A downward pressure under a long-only mandate is **noted, never dropped**: "can be researched but not held as a net short". This is consistent with Phase A's mandate note, which it refines per expression and per horizon.

The reviewed rules (`news_alpha/expression_library.py`) cover every one of the 31 states a seeded claim can reach. The default rules name no Options/Crypto expression: nothing in the seeded consequences needs one, and the prompt asked not to force symmetry. The architecture supports them, and tests prove the unavailable path with fixture rules.

## 4. Measurement specs (`MEASUREMENT_TEMPLATES`)

These are candidate measurement languages. Each names the ideal measurement and the data it requires; none claims the data exists.

| Domain | Generic templates | Concept-specific (named by a rule) |
|---|---|---|
| Futures | price momentum, trading activity, term structure & carry, open interest, COT positioning | spot-futures basis, physical inventories (commodity underlyings) |
| ETF | price momentum, relative strength vs SPY*, activity, fund flows, breadth*, factor exposure* | — |
| Equity | price momentum, relative strength vs SPY, activity, estimate revisions | revenue growth, gross margin, inventory growth, capex, free cash flow, net interest margin |
| Options | ATM IV, skew, vol term structure, expected move | — |
| Crypto | funding, basis, open interest, liquidations, on-chain | — |

\* Sector baskets only; breadth also applies to broad indices.

Measurements are **per instrument and shared** across expressions: NQ momentum is one spec serving six AI expressions.

## 5. PIT data field resolution (`news_alpha/measurement.py`, `field-resolution/1`)

Every `DataFieldResolution` records: status, typed issues, dataset / schema / field, frequency, coverage window (exclusive end), `pit_safe`, the PIT rule, publication lag, transform, feature kinds, `executable`, proxy note, missing reason, the named **gap**, `data_role`, and provenance.

Status follows from typed issues by fixed precedence:

**DOMAIN_UNAVAILABLE > NOT_PIT_SAFE > MISSING > NOT_EXECUTABLE > PARTIAL > AVAILABLE_WITH_PROXY > AVAILABLE**

A structural PIT violation outranks missing data, because acquiring the data would not fix it. A validator refuses any status that does not follow from the issues.

**PIT rules.**

- `pit_safe` is `True` only under a typed rule, `False` for a known violation, and `None` (UNKNOWN) when there is no data to judge.
- **Bars:** `ts_event` is the interval **start** (`schemas.market_data`). A bar's values are knowable at its end, so bar *t* drives nothing before the next permitted execution point.
- **COT:** Tuesday positions are released Friday. Values are keyed to the release timestamp (`features.macro.PointInTimeDatum`).
- **Fundamentals:** keyed to the SEC acceptance timestamp. Restated snapshots are never PIT.
- **A real `NOT_PIT_SAFE`:** ETF breadth. The only constituent list on the platform is the equity universe declared 2026-09-25 from hindsight, so this is unsafe even before any data is acquired.
- **Retrospective features** (`bars_until_next_roll`) are `NOT_PIT_SAFE` via the registry's own flag.

`historically_usable` requires a data-present status, `pit_safe=True`, `executable`, and `data_role=REAL`, all validator-enforced. `available_on(day)` answers "did we have it then?" and raises `HoldoutAccessError` for any date ≥ 2025-01-01. Coverage can never reach the holdout: both the resolver and the validator refuse it, and the catalog reader fails loudly on a row past it.

**Partial history.** A futures root acquired over a shorter span than the platform's other roots resolves as `PARTIAL`, usable only inside its own window. The rule is data-driven, with no threshold. Term structure also carries `PARTIAL_HISTORY`, because curve observations exist only around rolls; it is `NOT_EXECUTABLE` in any case, since no registered feature reads two contracts.

**Found by checking the real bytes:** the ETF window "2024-12-31" in `DATASET_WINDOWS` is the vendor request's **exclusive** end. The last acquired bar is 2024-12-30, and coverage is recorded that way.

## 6. The vertical slice: AI infrastructure (default mandate: Futures + ETF + Equity)

38 expressions: 29 continue, 9 are concept-only. 90 measurements: 1 available, 7 proxy, 5 not executable, 3 not PIT-safe, 74 missing. **20 expressions are measurable on 2018-2024 history, through NQ, XLK, QQQ and XLU.**

| Consequence | Equity | ETF | Futures |
|---|---|---|---|
| AI accelerator demand | NVDA (revenue, margin, inventory, momentum …: all missing, no equity bars) | XLK, QQQ (momentum: proxy) | NQ (momentum: **available**), MNQ (not acquired) |
| HBM / foundry / equipment demand | concept only (no memory, foundry or equipment maker in the universe) | XLK, QQQ | NQ, MNQ |
| AI spenders' FCF (↓ months · ↑ years) | MSFT, GOOGL, AMZN, META (capex, FCF) | XLK, QQQ | NQ, MNQ |
| Electricity / grid demand | concept only | XLU (momentum: proxy; relative strength: not executable) | — |
| Power-equipment demand | CAT (one segment of its business) | — | — |
| Copper demand | concept only (miners) | concept only (miner funds) | HG (catalogued, **not acquired**) |
| Gas for power | XOM, CVX | — | NG (not acquired) |

The Futures expressions exist only where a connected, catalogued contract exists. Only the NQ-backed ones are runnable today.

**Largest gaps** (the Gaps tab). Each count is primary blocker for / the only blocker for / also blocks behind another gap:

| Gap | Primary | Only blocker | Also blocks |
|---|---|---|---|
| Equity daily bars not acquired | 24 | 16 | 3 |
| PIT fundamentals: paid vendors only | 18 | 0 | 0 |
| Futures history not acquired (MNQ, HG, NG) | 9 | 6 | 0 |
| Analyst estimates: paid only | 8 | 0 | 0 |
| No cross-instrument features | 5 | 5 | 11 |
| Open interest not acquired | 4 | 0 | 0 |
| No COT connector | 4 | 0 | 0 |
| No registered feature computes it | 0 | 0 | 44 |

Acquiring equity bars therefore makes 16 measurements usable (momentum and activity), not 24. Relative strength would still need a cross-instrument feature. No fundamentals measurement becomes usable from a connector alone, because no registered feature consumes fundamentals yet.

Other events, same mandate:

- **OPEC+ cut:** 30 expressions, 17 measurable (CL, ZN, ES, NQ, GC and 13 ETFs).
- **Fed hike:** 19 expressions, 12 measurable.

## 7. UX (Agent → News Impact Triage card)

- **A summary line under the signal paths:** "Asset expression: 38 ways to express these consequences (Futures 8 · ETF 15 · Equity 15) — 29 continue under your mandate, 20 measurable on 2018-2024 history via NQ, XLK, QQQ, XLU. Largest gap: Equity daily bars not acquired (primary blocker for 24 measurements; the only blocker for 16)."
- **An "Asset expression & measurement" expander**, containing:
  - a domain table (expressions, how many continue, how many are runnable today, platform note; excluded domains stay listed);
  - four tabs, one per question:
    - **Expressions:** consequence, domain, expression, instruments, status, measurable n of m, direction, pressure, relation, form;
    - **Measurement readiness:** instrument × measurement matrix, per domain;
    - **Field resolution:** source, history, point-in-time, when knowable, why/proxy, frequency, transform, ideal measurement;
    - **Gaps:** per gap, where it is the primary blocker, the only blocker, and an additional blocker, noting that any acquisition needs a cost estimate and your approval.
- **The Agent chat** adds one `ASSET EXPRESSION:` line for a described event, with the plan's fingerprint and counts in `evidence`.

This was verified live with `streamlit run` and Playwright at 1440px and 420px, with no page-level horizontal overflow; wide tables scroll inside the card, as in Phases B–C. The first pass led to two changes: the five domain captions became a compact table, and the expression and field tables were re-ordered and sized so status and source stay in view.

## 8. Tests

`tests/python/test_news_alpha_phase_d.py` (42 tests) covers:

- the AI slice, and its resolution against the real data layout;
- the real catalog probe (roots = certified universe, before the holdout), and a catalog row inside the holdout failing loudly;
- multiple expressions per path;
- concept-only rows;
- an identical expression set under every mandate, with only admission changing;
- instrument constraints;
- a mismatched universe refused;
- pressure = direction × relation per horizon;
- the shorting note, never used as a filter;
- Options/Crypto reported and measured as unavailable, and never runnable under any mandate (including CME crypto futures);
- PIT safety:
  - the bar rule;
  - hindsight constituents;
  - a retrospective registered feature;
  - missing data leaving PIT unknown;
  - `available_on` refusing the holdout;
  - coverage never reaching the holdout;
- partial history;
- proxies;
- missing fields naming their gap;
- fundamentals following the recorded finding (monkeypatched);
- executability read from the live registry;
- status precedence;
- no synthetic-to-real promotion (validator and resolver);
- usable ⇒ real + PIT-safe + executable, swept over three events;
- `MeasurableVariable` reuse;
- provenance citations;
- determinism and round trip;
- blocker grouping;
- an unseeded event;
- an unmapped consequence;
- library validation, including drift against Phase C's `EXPOSURE_BASIS`;
- import and schema boundaries, and an unchanged registry hash;
- the view helpers.

`tests/python/test_news_alpha_phase_d_ui.py` (4 tests) covers:

- the full card in AppTest;
- an excluded domain staying visible but never enumerated;
- an unseeded card;
- the chat evidence.

The capability snapshot is pinned, and a live LLM client fails the test. Phase A's package-wide static guards (no registry writes, no network, no forbidden imports) automatically cover the three new modules.

**Negative controls.** Each was run on a scratch copy; each broke exactly its own test, then was restored:

- removing the synthetic-promotion validator broke the no-promotion test;
- treating hindsight constituents as PIT-unknown broke the breadth test;
- ranking MISSING above NOT_PIT_SAFE broke the precedence test;
- answering a holdout date broke the `available_on` test;
- admitting an excluded domain broke the economic-relevance test;
- letting `execution_capable` ignore research readiness broke the AI-slice test (through the validator).

## 9. Files

| New | Purpose |
|---|---|
| `news_alpha/expression_library.py` | Reviewed expression rules (31 states), measurement templates, import-time validation |
| `news_alpha/measurement.py` | `MeasurementSpec`, `DataFieldResolution`, resolvers, PIT rules |
| `news_alpha/asset_expression.py` | `AssetExpression`, `AssetExpressionPlan`, admission, pressure, gaps |
| `ui/asset_expression_view.py` | Summary, domain table, four tabs, chat text |
| `tests/python/test_news_alpha_phase_d.py`, `test_news_alpha_phase_d_ui.py` | Tests |

| Changed | Why |
|---|---|
| `news_alpha/universe.py` | Additive `FuturesBarCoverage` + `DomainCapabilities.futures_bars` + catalog reader (holdout-guarded) |
| `news_alpha/__init__.py` | Pipeline docstring, exports |
| `ui/news_alpha_context.py`, `ui/views/agent.py`, `ui/conversation_engine.py` | Accessor, card section, chat line |
| `scripts/news_alpha_pipeline_demo.py` | Asset-expression stage; economic-relevance-vs-admission check |

There was no C++, registry, identity, BH-FDR, `ReliabilityPolicy` or `AssetDomain` change, no holdout access, and no paid or network call. No data was acquired.

## 10. Not done, and choices to review

- **Quantitative Candidate Signals:** not started, by design.
- **Expression judgments** (reviewed at acceptance; kept, with their fidelity labelled, §11):
  - CAT for power-generation equipment: SEGMENT_EXPOSURE;
  - XLK/QQQ/NQ for foundry utilization: ECOSYSTEM_PROXY;
  - AAPL/MSFT/NVDA as "long-duration technology" for multiples: MACRO_PROXY;
  - HD for housing activity: ECOSYSTEM_PROXY.
- **Futures partial-history rule:** partial is measured against the span the platform holds for its other roots, not against a fixed window.
- **What would unlock the most research** (each needs your approval; nothing was acquired):
  - Equity daily bars (≈ $0.44 per the Phase 9.1 doc);
  - a free SEC XBRL fundamentals connector;
  - a free CFTC COT connector;
  - GLBX.MDP3 history for HG/NG, and the `statistics` schema (open interest).
- **Cross-instrument features** (relative strength, curve slope, beta): the data exists for ETFs and certified futures, but no registered feature reads two series. That is a FeatureRegistry/StrategySpec extension, not a data gap.

## 11. Acceptance patch

Three changes after review. There was no change to PIT rules, the holdout boundary, mandate-independence, concept-only behaviour, Options/Crypto honesty, the registry, C++ or scientific semantics, or data acquisition.

**1. Expression fidelity (`ExpressionFidelity`, library `/2`).** Every rule now states how directly its instrument carries the consequence. The fidelity sits beside `form`, which says what the instrument *is*.

| Fidelity | Meaning | Examples |
|---|---|---|
| `DIRECT_UNDERLYING` | Prices the consequence's own quantity | HG / NG / CL, Treasury futures, gold, broad indices for multiples |
| `DIRECT_COMPANY` | The company's own economics are the concept | NVDA for accelerators, hyperscalers for their own free cash flow, banks for NIM |
| `SEGMENT_EXPOSURE` | Moves one segment of a company | CAT (power generation), XOM/CVX for gas-for-power, cloud platforms for compute, hyperscalers for AI services revenue |
| `CONSTITUENT_EXPOSURE` | A fund or index holding the exposed companies | XLK/QQQ/NQ for accelerators, HBM, equipment and spenders; XLU; XLE; XLF |
| `ECOSYSTEM_PROXY` | The exposed actors are not inside, only adjacent businesses | XLK/QQQ/NQ for leading-edge foundry utilization, HD for housing |
| `MACRO_PROXY` | Tracks the consequence through a shared macro sensitivity | AAPL/MSFT/NVDA and XLU for equity valuation multiples |

The fidelity is a category, never a score: a test replaces every fidelity and shows that statuses, order, execution capability, measurements and gaps are unchanged. The import-time check requires every rule to state one and to fit its form (company fidelities need a single name; a constituent exposure cannot be one). The one rule that mixed two fidelities (AI compute providers: NVDA plus cloud platforms) is split into `AI accelerator designers` (DIRECT_COMPANY) and `Cloud platforms` (SEGMENT_EXPOSURE). That changes the AI slice from 37 to 38 expressions; the measurable count is unchanged at 20.

**2. Gaps are per blocker, not per measurement.** A `DataFieldResolution` now keeps `gaps`: every blocking issue named as a typed `ResolutionGap`, primary first (the one that decides the status), then by status precedence. The validator refuses a dropped or reordered blocker. `missing_reason` ends with "Also blocked by: …". `blockers()` reports, per gap:

- `primary_for`;
- `only_blocker_for` (what closing it alone would make usable);
- `also_blocks`.

The UI's Gaps tab, field table ("Blocked by"), summary and chat now say this. The claim "closing one gap unlocks every measurement it blocks" is gone, and a test checks the view source for it.

**3. Phase E guard.** `execution_capable` is documented as a platform fact: the instrument can be researched and simulated. It is not candidate readiness. `AssetExpressionPlan.candidate_signal_basis(expression)` is the gate a Candidate Signal stage must pass. It returns the measurements that are real, point-in-time safe, executable and historically usable, on a research-ready instrument of a continuing expression; otherwise it raises `CandidateSignalPrerequisiteError`. It is tested with an empty feature registry: NQ stays execution-capable but the gate refuses it.

Tests: `test_news_alpha_phase_d.py` gained 6 tests and rewrote the blocker test (48 collected): fidelity, multi-blocker gaps, the gate. The focused regressions pass: Phases A–D, the chat, translation, and the Agent/Phase 20 UI suites (408 tests). The negative controls broke their own tests: a gate that skips the measurement check, and "only blocker" counted as every primary.
