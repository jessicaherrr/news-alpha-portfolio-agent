# News Alpha Portfolio Agent — Phase E: Candidate Signals + Factor Diagnostics

Status: **IMPLEMENTED, TESTED, acceptance patch applied (§12) — stopped at review gate.** Cross-asset ranking and portfolio construction are not started.

Phase E extends the pipeline by four stages:

```
... → Asset Expression → Measurement Spec → PIT Data Field Resolution                     (Phase D)
    → Candidate Signal Spec → Factor Expression → Factor Series → Factor Diagnostics   (Phase E, new)
```

A **candidate signal** is a testable quantitative hypothesis: one registered factor expression on one instrument, predicting that instrument's forward return over a declared horizon with a declared sign, plus the full economic lineage that says why it exists. It is not a validated factor, an alpha, a recommendation, a trade, a weight or a verdict.

Run it:

```
PYTHONPATH=python python scripts/news_alpha_pipeline_demo.py --no-cached         # offline: candidates for 3 events
PYTHONPATH=python python scripts/news_alpha_phase_e_vertical_slice.py            # real bars: screens the AI slice (~1 min)
streamlit run python/alpha_agent/ui/app.py                                       # Agent → News Impact Triage → a card
```

## 1. Reuse audit

| Searched for | Found | Decision |
|---|---|---|
| IC / ICIR / decay / turnover utilities | **None** anywhere in the repo (`validation/` has Sharpe, DSR, BH, bootstrap; `screening/fast_screen` runs C++ backtests) | New, generic engine in `screening/factor_diagnostics.py`. Not a duplicate: no factor-level diagnostic existed. |
| Feature computation | `features.registry` + `compute_features` + `features.daily.compute_contiguous_daily_features` (the frozen gap-free daily technique) | **Reused unchanged.** Every factor operator *is* a registered feature kind; the series is computed by the same engine. |
| A typed expression language | `FeatureSpec` (one kind + params); no composition layer | New thin `features/expression.py`: a closed vocabulary where each operator is bound to exactly one registered kind (checked at import). It is not a second feature system. |
| `translation.FactorCandidate` | Phase 1's narrative factor candidate for a futures root: quant `EconomicMechanism`, proposed kinds, researchability. No instrument/expression/horizon plane | Not reused as the candidate type. Wrapping it would require assigning a quant mechanism (TREND…) to an economic-transmission hypothesis, a new claim that Phase B deliberately kept separate. It is left unchanged. |
| `alpha_memory.FactorIdentity` | Identity of an economic idea (quant mechanism × strategy family); no expression, instrument or parameter plane | Not reusable as an executable-factor identity (it cannot distinguish `ts_return(close, 20)` on NQ from XLK). Phase E's structural identity sits at the expression level. No candidate is mapped onto a strategy family, because the V1 rule forbids merging without typed proof. |
| Screening authority | `screening/fast_screen` (C++ strategy screen, 2018-2022 only) + its `assert_research_window_only` | Same package; the Fast Screen's own window guard is **reused** as the boundary check. |
| Research window | `data.real_market_dataset.RESEARCH_WINDOW` (2018-01-01 → 2023-01-01) | **Reused.** Screening never reads 2023-2024 (validation) or 2025 (holdout). |
| PIT daily bars | Futures: `reconstitute_root` + `daily_signal_series` (forward-adjusted). ETF: `etf.data_source.load_primary_listing_bars` | **Reused.** The only new code is the loader glue. It applies the one *sourced* ETF split (USO 1-for-8, SEC 8-K, `etf.corporate_actions.KNOWN_SPLITS`), which no existing helper applied. |
| Strategy DSL / `StrategySpec` | Frozen strategy compiler | Not touched. A candidate is not a strategy; compiling one into a strategy is a later stage. |
| Registry / validation | Scientific authority | Not touched. There are no writes, no verdicts, and no alpha-memory promotion. |

**Deleted: nothing.** No duplicated responsibility was found. `FactorCandidate`, `FactorIdentity`, `discovery.candidate_pool` and the Fast Screen each answer a different question from a candidate signal.

**Boundary.** Phase A's test-enforced guard forbids `news_alpha` (and its UI context) from importing `validation`, `screening` or `strategy`. So the hypothesis objects live in `news_alpha/candidate_signals.py`, while loading bars and computing diagnostics live in `screening/`. The guard now sweeps the new module too.

## 2. The Phase D gate

Candidates are built **only** from `AssetExpressionPlan.candidate_signal_basis(expression)`. That gate returns only measurements that are real, point-in-time safe, executable and historically usable, on a research-ready instrument of a continuing expression. Nothing here reads news text, an impact scan, a graph or a path on its own.

`SignalDataBinding` re-checks the gate's result: status must be AVAILABLE, PROXY or PARTIAL; `pit_safe` must be True; `data_role` must be REAL; coverage must end before 2025-01-01. It is tested against MISSING, NOT_PIT_SAFE, SYNTHETIC, `pit_safe=False` and holdout coverage. With an empty feature registry, NQ stays `execution_capable` but yields **no** candidate.

## 3. Candidate rules (`candidate-signal-rules/2`, predeclared, never fitted)

| Measurement | Candidate | Why |
|---|---|---|
| Price momentum (futures / ETF / equity) | `ts_return(close, L)`, expected relationship **POSITIVE** | Continuation: a consequence that diffuses into price gradually keeps moving it the same way. Reversal would be a different, negative-sign hypothesis. The sign is never flipped after diagnostics. |
| Trading activity | **none** — typed refusal `CONFIRMATION_ONLY_MEASUREMENT` | Volume says the market is paying attention, not which way. It is kept for a later event-conditioned test. |
| Anything else usable | typed refusal `NO_SIGNAL_RULE` | No rule, no formula. |

**Transmission horizon is not prediction horizon.** A path's economic lag ("AI capex reaches HBM demand over months") says when the economics arrive; prices can react long before or after. No lag is converted into a market horizon, and there is no "years → 60D" cap. The lag stays on every origin as `transmission_lag` (provenance), and an UNKNOWN lag refuses nothing.

**Generation grid (`candidate-generation/1`).** Every directional measurement gets the same predeclared (formation lookback, prediction horizon) pairs:

| Formation lookback | Prediction horizon |
|---|---|
| 20 bars | 20D |
| 60 bars | 60D |

The diagnostics still report 1D / 5D / 20D / 60D decay for every candidate.

**`formation_lookback` and `prediction_horizon` are independent fields.** The grid starts with them matched, but that is an initial generation policy, not a semantic rule. A (60 bars, 20D) spec is valid and has its own candidate id (tested, via a custom grid). The only coupling a validator enforces is that `formation_lookback` equals the window of the expression that actually executes. Changing the grid is a new policy with new identities.

**Why not `rank(ts_return(close, 20))`?** `rank()` is cross-sectional. NQ alone has no cross-section, so its rank is a constant: a formula that looks like a factor and carries no information. The vocabulary includes cross-sectional rank so that its refusal is typed: `evaluate` raises `CrossSectionalOperatorError`, and `SignalSpec` refuses it. The prompt's example is therefore not used as written. Phase E declares the time-series form it actually tests.

**Event conditioning (`EventConditioning`).** Each candidate records where it sits in the conceptual model *Signal = EventStrength × Exposure × Confirmation × TransmissionWeight*. Each term is kept as a separately testable ablation dimension:

| Term | Status |
|---|---|
| Confirmation | **The factor itself** (the market's own price response) |
| EventStrength | **UNKNOWN_MAGNITUDE** (the mechanism pass reports `magnitude_established=False`) |
| Exposure | **UNKNOWN_MAGNITUDE** (fidelity is a category, not a size) |
| TransmissionWeight | **NOT_DERIVED** (never read off link confidence) |

A validator refuses any invented magnitude. The diagnostics are therefore *unconditional*: they test the factor on every day of 2018-2022, not the reaction to this event. An event-conditioned test needs a history of comparable events.

## 4. One expression, end to end

`FactorExpression` (`features/expression.py`, `factor-expression/1`) is one typed object with three views:

```
render()        -> "ts_return(close, 20)"                what the UI shows
feature_spec()  -> FeatureSpec(kind="return", n=20)     what executes (registered feature engine)
evaluate(bars)  -> factor values                         what the diagnostics test
```

`CandidateScreen` refuses to exist unless the candidate, the series and the diagnostics name the same formula and the same structural identity. A test recomputes the series independently, both through the registered engine directly and by hand as `close.pct_change(20)`, and matches it.

The operator vocabulary is closed: `ts_return`, `ts_log_return`, `ts_delta`, `ts_zscore` (close or volume) and `ts_trend_strength`. Each is bound to its registered kind with that kind's own parameter rules. Parameters are strict integers (a `True` is never read as 1).

## 5. Identity: structure vs provenance

| Identity | Includes | Excludes |
|---|---|---|
| `factor_identity` (`factor1:`) | instrument, dataset, schema, canonical expression (registered FeatureSpec + feature-engine version) | events, paths, timestamps, labels |
| `candidate_signal_id` (`csig2:`) | factor identity + formation lookback, prediction horizon, expected relationship, execution lag, measurement, rule version, generation policy | the same, and the economic transmission lag |
| `origins` | every originating event → path → consequence → asset expression → measurement | — |

Several paths reaching NQ `ts_return(close, 20)` → 20D are **one** candidate with several origins, and so are several events. The AI and OPEC events share `csig2:9a63e932…` (NQ 20/20D). Across the three demo events, 74 candidates collapse to 38 distinct structural candidates.

Changing any of the following changes the identity:

- the formation lookback (a new factor);
- the prediction horizon (same factor, new candidate);
- the sign (a re-signed factor is a new hypothesis);
- the instrument or data (a new factor).

## 6. Factor series and diagnostics

**Data.** All bars are real and already acquired.

- **Futures:** the forward-adjusted continuous front (GLBX.MDP3 1-minute, rebuilt offline from the raw store), one bar per CME trading day. This is the research series that signals may use. Fills would resolve to raw contracts in the C++ engine, which is never called here.
- **ETFs:** primary-listing daily bars, split-adjusted only where a sourced split exists. Returns are price-only, as Phase D already flags.

No download was made, the data cost was $0, and no network call was made.

**Window.** Only the discovery window, 2018-01-01 → 2023-01-01 intersected with each measurement's coverage, is used. The frame is sliced by trading day and then checked by the Fast Screen's own boundary guard and the holdout guard before any feature is computed. Forward-return pairs whose window would leave the discovery window are dropped, not truncated.

**Forward return.** `fwd_h(t) = close[t+h] / open[t+1] − 1`: the factor read at bar t's close is acted on no earlier than bar t+1's open.

The diagnostics (`screening/factor_diagnostics.py`, `factor-diagnostics/1`) are:

| Diagnostic | Definition |
|---|---|
| Scope (`DiagnosticScope`) | Every result is **UNCONDITIONAL_FACTOR**: the factor on every day of the window, with no knowledge of any event. `EVENT_CONDITIONED` exists as a scope but is refused by the engine (and by the model), never approximated with unconditional evidence. |
| TS Pearson / TS Spearman IC (`ICKind.TIME_SERIES`) | The correlation **over time** between one instrument's factor and its **own** forward return. This is not a cross-sectional Rank IC (factor ranks against forward-return ranks across instruments on each date), which needs a universe; `CROSS_SECTIONAL` is a reserved IC kind nothing computes yet. Fields say so: `ts_pearson_ic`, `ts_spearman_ic`, `ts_spearman_t`. |
| t (n/h) | The TS Spearman IC's t on the **effective** sample of n/h non-overlapping windows. It is conservative: under no relationship, var(r) ≈ (1/n)·Σρₓρ_y ≤ h/n because \|ρₓ\| ≤ 1. |
| Decay | 1D / 5D / 20D / 60D (plus the declared horizon) |
| Subperiod stability | TS Spearman IC per calendar year, the share of years with the predeclared sign, and their mean / std across years (`ts_spearman_ir_yearly`). This is not a cross-sectional ICIR. |
| Coverage | defined days, warm-up, and undefined days after warm-up (never filled) |
| Turnover | 1-day autocorrelation, sign flips per year |
| Cost sensitivity | For a ±1 sign position rebalanced every h days: gross edge = drift + timing edge. The break-even one-way cost is computed on the **timing** edge only. Descriptive; not a backtest or PnL. |
| Group concentration | "Not applicable" (single-instrument factor), stated rather than faked |
| Signal correlation | pairwise, per screened set |

**Screen (`factor-screen/1`, predeclared, fingerprinted `FactorScreenPolicy`):**

| Status | Rule |
|---|---|
| `SCREEN_CONTINUE` | t ≥ 2 in the predeclared direction, and ≥ 60% of evaluable years agreeing. Worth deeper validation; not validated. |
| `CONTRADICTS_EXPECTED_SIGN` | t ≤ −2. The factor is never re-signed into a pass. |
| `INSUFFICIENT_DATA` | fewer than 250 pairs, fewer than 3 evaluable years, or an undefined statistic |
| `NO_SCREEN_SUPPORT` | otherwise |

Screening applies no multiple-testing correction. The run records `family_size` (every candidate screened counts toward the family that validation must correct across). No p-value, q-value, DSR or verdict exists here, and there is no registry write.

### Two findings made on real data

1. **Newey-West was rejected.** The first version used a Newey-West (Bartlett, lag h) t. On real XLU bars it turned a 60-day TS Spearman IC of −0.30, resting on about 18 independent windows, into t = −3.2 (`CONTRADICTS_EXPECTED_SIGN`). Non-overlapping subsamples at five offsets gave TS Spearman ICs from −0.05 to −0.57. The Bartlett kernel recovers only about ⅔ of the long-run variance of overlapping returns. The screen now uses the conservative n/h t (−1.25 → `NO_SCREEN_SUPPORT`). A test checks calibration: across 60 independent null worlds with a persistent 60-day factor, |t| > 2 in at most 10% of them.
2. **Gross edge was mostly drift.** The first cost diagnostic reported NQ 60D momentum with a *positive* gross edge (+9.7 bp) despite a negative IC. A mostly-long position on a rising index earns the drift whatever the factor says. The diagnostic now splits drift from timing, and the break-even uses timing only.

## 7. The real vertical slice (AI infrastructure; Futures + ETF + Equity mandate)

`outputs/news_alpha/phase_e/AI_INFRASTRUCTURE__candidate_signal_screens.json` is written by `scripts/news_alpha_phase_e_vertical_slice.py`. It records the fingerprints of every stage, the code commit, the $0 data cost, and the diagnostics of each candidate.

Pipeline: 20 signal paths → 38 asset expressions → 8 usable measurements → **8 candidates** (NQ, QQQ, XLK, XLU × the 20/20 and 60/60 grid points). There are 38 refusals: 18 not candidate-ready (Phase D gate), 20 activity measurements kept as confirmation inputs. There is **no equity candidate**: equity bars are not acquired, and none was fabricated.

All results are scope **UNCONDITIONAL_FACTOR** with a **time-series** IC.

| Candidate (2018-2022) | TS Spearman IC | t (n/h) | Years with + sign | Timing edge | Screen |
|---|---|---|---|---|---|
| NQ `ts_return(close, 20)` → 20D | −0.056 | −0.44 | 0 / 5 | −29.7 bp | no screen support |
| NQ `ts_return(close, 60)` → 60D | −0.074 | −0.31 | 0 / 5 | −104.3 bp | no screen support |
| QQQ `ts_return(close, 20)` → 20D | −0.051 | −0.38 | 1 / 5 | −32.4 bp | no screen support |
| QQQ `ts_return(close, 60)` → 60D | −0.042 | −0.17 | 0 / 5 | −91.8 bp | no screen support |
| XLK `ts_return(close, 20)` → 20D | −0.040 | −0.29 | 1 / 5 | −19.3 bp | no screen support |
| XLK `ts_return(close, 60)` → 60D | −0.111 | −0.44 | 0 / 5 | −101.9 bp | no screen support |
| XLU `ts_return(close, 20)` → 20D | −0.293 | −2.27 | 0 / 5 | −108.7 bp | **contradicts expected sign** |
| XLU `ts_return(close, 60)` → 60D | −0.302 | −1.25 | 0 / 5 | −118.6 bp | no screen support |

Reading it:

- Momentum continuation on the AI-exposed instruments has **no screening support** on 2018-2022. Every TS Spearman IC is negative.
- **XLU 20D contradicts the predeclared continuation sign** (t −2.27). This candidate exists only since the acceptance patch; the old lag mapping gave XLU only a 60D horizon. It is a screen outcome for one of 8 candidates with no multiple-testing correction. It is not validated, never re-signed into a reversal factor, and says nothing about the AI event itself, since the scope is unconditional.
- NQ, QQQ and XLK momentum correlate at **+0.96 to +0.98**. They are close to one signal counted three times, and the Signal correlation tab says so.
- Nothing here is validated, and no 2023-2024 or 2025 data was read.

Other events, same mandate:

- **OPEC+ cut:** 36 candidates on 18 instruments.
- **Fed hike:** 30 candidates on 15 instruments.
- Across the three events, 74 candidates collapse to 38 distinct structural candidates.

## 8. UX (Agent → News Impact Triage card)

- **A summary line under Asset expression:** "Candidate signals: 7 testable factor hypotheses on NQ, QQQ, XLK, XLU (e.g. NQ ts_return(close, 20) → 20D) — 7 of 7 screened: 7 no support. Not validated."
- **A "Candidate signals & factor diagnostics" expander**, containing:
  - **A candidates table:** signal, the exact factor expression, status, instrument, domain, prediction horizon, formation lookback, expected relationship, consequences, path types, depth, fidelity (provenance), measurement, transform, PIT, data window, notes.
  - **A run button:** "Run factor diagnostics on 2018-2022 (n unscreened)". It is explicit and never automatic, and it says a futures root takes about a minute. It reads acquired bars only. Results are cached (gitignored `data/news_alpha/factor_screens/`) by structural id, so another event reaching the same candidate shows them immediately. A screen made under a different screening policy or rule is treated as stale.
  - **Why this signal?:** the formula as code, then the numbered chain News → Mechanism → Signal paths (every path) → Economic consequence → Asset expression (every origin, with fidelity and pressure) → Measurement → Factor expression. Below it sits a wrapped Field / Value / Why table (transform, the registered feature it executes, formation lookback, normalization, sign, prediction horizon — stated as not derived from the economic lag, execution lag, data, PIT rule, identity) and the event-conditioning statement.
  - **Diagnostics:** a per-candidate table (TS Spearman IC, t (n/h), TS Pearson IC, years with the expected sign, TS IC mean/std across years, coverage, flips per year, break-even, screen). Selecting a candidate shows the same formula as code, a "Scope: unconditional factor · IC kind: time-series" line, the screen note, the decay and yearly charts, the factor series, and wrapped coverage / turnover / cost / concentration / forward-return / data / rule rows.
  - **Signal correlation** and **Not candidates** (typed refusals).
- **The Agent chat** adds a `CANDIDATE SIGNALS:` line, with ids, identities and expressions in `evidence`.

This was verified live with `streamlit run` and Playwright at 1440px and 420px, with no page-level horizontal overflow. The first pass showed long lineage and rationale text truncated in dataframe cells and a raw graph hash in the lineage. These were changed to a Markdown chain and wrapped `st.table`s.

## 9. Tests

| File | Tests | Covers |
|---|---|---|
| `test_news_alpha_phase_e.py` | 31 | the AI slice; every candidate through the gate (3 events, every domain incl. Options/Crypto fixtures); no candidate from MISSING / NOT_PIT_SAFE / NOT_EXECUTABLE / DOMAIN_UNAVAILABLE / SYNTHETIC (the sweep asserts it met each); the data binding refusing unusable/holdout data; `execution_capable` alone yields nothing; activity refused as confirmation-only; the transmission lag never decides the prediction horizon (every lag rewritten, incl. UNKNOWN); lookback and horizon independent; full lineage per origin; PathType + depth preserved; fidelity provenance-only (flattening every fidelity changes nothing else); no invented magnitude; determinism + round trip; parameter-change identity rules; shared structural identity across paths and events; the closed vocabulary; deterministic compilation; cross-sectional rank refused; UI expression = executed = diagnosed, recomputed independently; same bars → same fingerprints; discovery window only with bars running into 2025; store keeps REAL screens only, reuses across events, refuses stale-policy screens; one real-bytes ETF screen; import/token boundaries; no return/weight/verdict fields; registry hash unchanged |
| `test_factor_diagnostics.py` | 15 | forward-return convention; an oracle factor screens and its negative contradicts (never re-signed); noise and too-little-data; the n/h t (exact formula, ≪ naive t); **null calibration** across 60 worlds; decay horizons; yearly subperiods and ICIR; coverage without filling; turnover; the drift/timing split; correlations; window/holdout/order guards; policy fingerprint; every result scoped UNCONDITIONAL_FACTOR and EVENT_CONDITIONED refused; ICs typed TIME_SERIES with no `rank_ic`/`ic` field names |
| `test_news_alpha_phase_e_ui.py` | 5 | the card (summary, expander, formula as code, full chain, run button); a stored **real** XLK screen rendered with the same expression; the run button screening real ETF bars and persisting; an unseeded event; chat evidence |

Phase A's package-wide static guards now also sweep `candidate_signals.py`.

**Regression.** The 75 test files touching the Agent page, the chat, features, screening, charts or News Alpha: 1,428 passed, 51 skipped, 2 failed. Both failures are known and predate Phase E:

- `test_agent_product_refactor::test_G` needs the C++ engine, and this checkout has no `build/`;
- `test_phase_b1_streamlit_ui::test_research_details_shows_verdict_promise_fit_row` was already failing.

**Negative controls.** Each control broke one guarantee in the live file, ran the test meant to catch it, and restored the file byte-for-byte. All eleven were caught (the last two were added by the acceptance patch):

| Broken guarantee | Caught by |
|---|---|
| gate bypassed (any measurement of the expression) | `test_no_candidate_from_missing_…` |
| economic lag decides the prediction horizon | `test_the_transmission_lag_never_decides_the_prediction_horizon` |
| event-conditioned scope silently accepted | `test_every_diagnostic_is_scoped_unconditional_and_event_conditioned_is_refused` |
| hidden ×1.0001 transform on the executed series | `test_ui_expression_equals_the_executed_and_diagnosed_expression…` |
| fidelity used as a filter | `test_fidelity_is_provenance_only…` |
| naive t on overlapping daily pairs | `test_the_effective_sample_t_is_calibrated…` |
| forward return entered at the signal bar's own close | `test_forward_returns_enter_at_the_next_open…` |
| a contradicting factor re-signed into a pass | `test_a_factor_that_knows_the_forward_return…` |
| synthetic screen allowed into the store | `test_the_screen_store_keeps_real_screens_only…` |
| stale-policy screen served as current | same |
| cross-sectional rank evaluated on one instrument | `test_a_cross_sectional_rank_is_refused…` |

## 10. Files

| New | Purpose |
|---|---|
| `features/expression.py` | Closed factor-operator vocabulary over registered features |
| `news_alpha/candidate_signals.py` | `SignalSpec`, `CandidateSignal`, lineage, rules, identity, gate |
| `screening/factor_diagnostics.py` | Generic IC / decay / stability / turnover / cost diagnostics and `factor-screen/1` |
| `screening/candidate_signal_screen.py` | PIT bar loaders, discovery-window slicing, `FactorSeries`, `CandidateScreen`, the screen cache |
| `ui/candidate_signal_view.py` | Card section, tabs, charts, chat line |
| `scripts/news_alpha_phase_e_vertical_slice.py` | Real end-to-end slice and provenance artifact |
| `outputs/news_alpha/phase_e/AI_INFRASTRUCTURE__candidate_signal_screens.json` | Vertical-slice evidence |
| three test files | above |

| Changed | Why |
|---|---|
| `news_alpha/__init__.py` | Pipeline docstring, exports |
| `ui/news_alpha_context.py`, `ui/views/agent.py`, `ui/conversation_engine.py` | Accessor, card section, chat line |
| `ui/charts.py` | `signed_bar_chart`, `factor_series_chart` |
| `scripts/news_alpha_pipeline_demo.py` | Candidate stage + structural-identity check |
| `tests/python/conftest.py`, `.gitignore` | Screen-cache isolation; the cache is gitignored |

There was no C++, registry, experiment-identity, BH-FDR, `ReliabilityPolicy`, `AssetDomain` or Phase A–D semantic change, no holdout access, and no paid or network call.

## 11. Not done, and choices to review

- **Not started, by design:** cross-asset ranking, portfolio construction, validation, and alpha-memory promotion.
- **Choices to review:**
  - the generation grid (20/20D and 60/60D);
  - activity as confirmation-only;
  - the screen thresholds (t ≥ 2, 60% of years, 250 pairs);
  - the n/h t;
  - calendar-year subperiods.
- **Next useful steps:**
  - An **event-conditioned** test: event history × this factor. That is where EventStrength / Exposure / Confirmation ablations (A–E in the prompt) become measurable.
  - **Cross-instrument features** (relative strength vs SPY, curve slope). The data exists; the FeatureRegistry lacks two-series kinds. Adding them would unlock Phase D's NOT_EXECUTABLE measurements.
  - **Equity daily bars** (≈ $0.44 per the Phase 9.1 doc; needs your approval) would unlock single-name momentum candidates, e.g. NVDA.

## 12. Acceptance patch

Four changes after review, with no change to the gate, the data, the windows, the registry or C++:

1. **Transmission horizon ≠ prediction horizon.**
   - `signal-horizon/1` (lag → horizon, including the "years → 60D" cap) is removed, along with `HORIZON_CAPPED` and `HORIZON_UNKNOWN`.
   - Prediction horizons come from the predeclared generation grid `candidate-generation/1`.
   - The lag survives only as `CandidateOrigin.transmission_lag`.
   - A test rewrites every path's lag (immediate, years, unknown) and shows no candidate changes.
2. **Independent fields.**
   - `SignalSpec.formation_lookback` and `SignalSpec.prediction_horizon` replace `lookback` / `horizon`, plus `generation_policy`.
   - Matching them is the grid's initial policy only; a (60 bars, 20D) spec is valid and distinct.
   - Identity is `csig2:`, and the rules are `candidate-signal-rules/2`.
3. **`DiagnosticScope`.**
   - Every diagnostic carries `scope = UNCONDITIONAL_FACTOR` and a scope note.
   - `EVENT_CONDITIONED` is refused by the engine and by the model.
   - The UI and chat say "screened as unconditional factors … not event-conditioned evidence".
4. **Time-series IC, named as such.**
   - `ic` / `rank_ic` / `rank_ic_t` / `icir_subperiod` became `ts_pearson_ic` / `ts_spearman_ic` / `ts_spearman_t` / `ts_spearman_ir_yearly`, with `ic_kind = TIME_SERIES`.
   - `CROSS_SECTIONAL` is reserved for a future universe-level Rank IC, and the model refuses it today.
   - UI columns read "TS Spearman IC" / "TS Pearson IC".

Schemas moved to `candidate-signal-set/2`, `factor-diagnostics/2` and `candidate-screen/2`, so previously cached screens are stale by construction and recomputed.
