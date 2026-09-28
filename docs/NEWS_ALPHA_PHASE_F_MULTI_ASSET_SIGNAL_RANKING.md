# News Alpha Portfolio Agent — Phase F: Multi-Asset Signal Ranking

Status: **IMPLEMENTED, TESTED — stopped at review gate.** Portfolio construction is not started, and no weights exist anywhere.

Phase F adds one stage to the pipeline:

```
... → Candidate Signal Spec → Factor Expression → Factor Series → Factor Diagnostics   (Phase E)
    → MULTI-ASSET SIGNAL RANKING (portfolio consideration, under one mandate)          (Phase F, new)
    → Portfolio construction                                                          (later)
```

The question it answers: **given one research mandate and a set of candidate signals with their diagnostics, which signals deserve portfolio consideration first?**

It does not produce:

- portfolio weights, sizes or allocations;
- BUY/SELL decisions;
- expected returns or probabilities;
- a validation or a verdict.

Run it:

```
PYTHONPATH=python python scripts/news_alpha_phase_f_signal_ranking.py                # real bars, 3 events (~4 min first time)
PYTHONPATH=python python scripts/news_alpha_phase_f_signal_ranking.py --cached-only  # rank what the screen cache holds
PYTHONPATH=python python scripts/news_alpha_pipeline_demo.py --no-cached             # offline demo, ranking stage at the end
streamlit run python/alpha_agent/ui/app.py                                           # Agent → News Impact Triage
```

## 1. Reuse audit and cleanup

**One canonical owner.** The recommendation layer is the platform's research-prioritization owner. Phase F extends it with `recommendation/signal_ranking.py` rather than starting another engine. Each existing ranker orders a different object:

| Ranker | Ranks | Reused? |
|---|---|---|
| `recommendation.promise` / `.fit` / `.candidates` | Registry results (Sharpe, DSR, BH q) of tested strategies, plus User Fit | **Reused:** the merit/fit split, `PersonalizationState`, and the trading-frequency bands (`fit._frequency_band` is now the public `frequency_band`, shared by both). **Not reused:** `score_research_promise`. A screened candidate has none of its inputs, and its "missing input scores 0" rule contradicts "missing evidence is not low quality". |
| `context_retrieval.ranking` | Quant mechanisms for a market context | The pattern was reused (ordinal dimensions compared lexicographically in a declared order, never blended; "why #1 over #2" names the first differing dimension). The object is different. |
| `opportunity.schemas` | Current market conditions (WATCH/INVESTIGATE/WAIT) | Not applicable: this is the observation plane. |
| `screening.fast_screen` | Strategy backtest screens | Not applicable: a different object. It stays the owner of its own ordering (see the cleanup below). |
| `screening.factor_diagnostics.signal_correlations` | — | **Reused** for redundancy. It gained an additive `method="spearman"` option (section 5). |

**Cleanup: UI-side ranking removed or migrated.**

| Where | What it did | Now |
|---|---|---|
| `ui/views/discover.py` `_rank_key` | Re-implemented `rank_fast_screen_trials` inline to order pool members | **Deleted.** It now calls the new `screening.fast_screen.rank_pool_members`, which reuses that module's rule. A regression test covers it. |
| `ui/candidate_signal_view.py` "Signal correlation" tab | Computed pairwise correlations and sorted them by \|ρ\| in the view | **Removed.** Correlation and redundancy now come from the backend (`RankedSignalSet.correlations`, already ordered). They render in the ranking's "Exposures & redundancy" tab. Two Phase E UI assertions were migrated. |
| `ui/signal_ranking_view.py` (new) | — | A pure renderer. A static test fails it if it ever contains `sorted(`, `.sort(`, `key=lambda`, `min(` or `max(`. |

`ui/market_scanner.sort_rows` was reviewed and left alone. It is a user-chosen table sort over observation rows, not a ranking of research objects.

**Boundary.** Phase A's guard forbids `news_alpha` (and its UI context) from importing `screening`. The ranker needs the screening plane, so it lives in `recommendation/` and is imported by module path. It is not re-exported from `recommendation/__init__`, which avoids an import cycle through `news_alpha.mandate`.

## 2. What is ranked: candidate + diagnostics, in three tiers

A naked candidate is never ranked. Every signal lands in exactly one tier:

| Tier | When | Carries |
|---|---|---|
| `RANKED` | A real, comparable, sufficient screen, and admitted by the mandate | rank, role, merit, fit |
| `EXCLUDED_BY_MANDATE` | The mandate's access constraints (domain, instrument allowlist or denylist) exclude it | Its merit, **unchanged**, beside the exclusion reason |
| `NOT_RANKABLE` | `NOT_SCREENED`, `INSUFFICIENT_EVIDENCE`, `NO_DIAGNOSTIC_METHOD` (no loader for the domain, e.g. Equity, Options, Crypto today), `NON_COMPARABLE_EVIDENCE` (another method, scope or policy), or `NOT_REAL_EVIDENCE` | No rank, and "what would rank it" |

**Missing evidence is not low quality.** A NOT_RANKABLE signal has no rank at all rather than a low one, and it sits in its own list. It never appears below a contradicted signal as if it were worse. Adding unscreened candidates never changes the ranks of screened ones (test-enforced).

## 3. Research merit: no universal score

**Peer group = diagnostic method.** A futures Sharpe, an equity IC and a crypto funding signal are not interchangeable. Evidence enters through a normalizer keyed by the method that produced it (`PeerGroup`: diagnostics schema, screen rule, scope, IC kind, policy fingerprint, forward-return convention).

- Today there is one method: Phase E's unconditional, single-instrument, time-series factor screen. It is identical for futures and ETFs.
- A futures and an ETF candidate are comparable **because they were measured the same way**, not because their numbers were forced onto one scale.
- A screen from any other method, scope or screening policy is `NON_COMPARABLE_EVIDENCE` until a normalizer for it exists.

**Normalization (`factor-screen-grades/1`).** Raw metrics are kept exactly as measured: TS Spearman/Pearson IC, t, pairs, windows, years, coverage, turnover, gross/drift/timing edge, break-even. They are graded **with the screen policy's own thresholds**; no new cut-off is introduced:

| Dimension | Grade | Rule |
|---|---|---|
| Screen outcome | continue / no support / contradicts | Phase E's `factor-screen/1` status |
| Direction of evidence | supports / leans declared / leans opposite / contradicts | The effective-sample t × declared sign, against the screen's t threshold and 0 |
| Subperiod stability | consistent / mixed / opposed (not evaluable = neutral) | Share of years with the declared sign, against the screen's 60% and its mirror 40% |
| Cost headroom | timing edge / no timing edge (not assessed = neutral) | The sign of the drift-free timing edge |
| Data quality | clean / qualified / flagged | Proxy or partial history = qualified; feature QA issues or undefined values after warm-up = flagged |

**Order (`signal-ranking/1`).** The comparison is lexicographic:

> screen outcome → direction → stability → cost headroom → data quality → effective-sample t

Nothing is summed. The t is dimensionless and sample-size adjusted, so it is the one continuous metric comparable across instruments, domains and horizons within the peer group. "Why this rank" names the first dimension on which a signal beats the next one, for example "above #2 USO on direction of evidence: supports the declared sign (t +2.01) vs leans the declared way (t +0.68)".

**Never ranking strength.** Mechanism confidence, path status and expression fidelity travel as `MechanismContext` (provenance) and are never read by any key. The test gives opposite fidelities and path statuses to different instruments in turn; ranks, merit and groups stay identical.

## 4. User fit: separate from merit

`UserFit` reads only the mandate. `ResearchMerit` never reads it: the merit JSON is byte-identical across five different mandates (test-enforced).

| Dimension | Kind | Rule |
|---|---|---|
| Domain access, instrument allowlist/denylist | **Excludes** | The signal becomes EXCLUDED_BY_MANDATE, with merit kept |
| Shorting | Constraint | Measured from the factor: the share of days the ±1 expected-sign position is short. Under a no-shorting mandate this is CONSTRAINED: "only a long/flat form would be admissible, and that form was not measured". It is not an exclusion, because the default mandate disallows shorting and would otherwise empty every list. |
| Holding period | Soft | A prediction horizon inside the profile's window (1-3 Days = 1–3, Several Days = 2–10, Weeks = 5–30 trading days; Intraday fits no daily signal; Flexible = no preference) |
| Trading frequency | Soft | Position flips per year under the diagnostics' own rebalance-every-h model, banded by the shared `recommendation.fit.frequency_band` |
| Liquidity, drawdown | Not measured | Shown honestly: screens record no traded notional, and drawdown needs sizing plus a backtest |

Soft fit **only breaks exact merit ties**. A test covers a mandate whose fit differs across signals (Weeks: 20D fits, 60D does not), and the order does not change. `PersonalizationState` follows `recommendation.fit` exactly.

## 5. Redundancy: exposures, not repeated opportunities

1. **Structural dedup first.** One structural candidate (`csig2:` id) reached by several events is one signal, with every event kept in its mechanism provenance.
2. **Links.** Two ranked signals are one exposure when they are:
   - the **same structural signal family on the same instrument** — same instrument *and* the same `signal_rule_id` (the generation rule; a different lookback or horizon is still the same family); or
   - **correlated factors**, meaning the expected-sign-aligned factor series have |rank correlation| ≥ 0.8 over ≥ 250 common days — regardless of instrument or family (`SignalRankingPolicy`, fingerprinted).

   **Same instrument is not automatically same exposure.** A momentum candidate and a (once it exists) mean-reversion candidate on one instrument are different structural families and are never merged by co-location alone — only actual correlation can link them. Today's rule set has one family, so this has no effect on today's real-data groupings; it becomes load-bearing the moment a second family is added, and is test-enforced now with a synthetic second family (section 10.5).

   An opposed correlation links too: offsetting positions are not independent opportunities either.
3. **Exposure groups.** These are the connected components of the links. A signal may join through lower-ranked members; the explanation names the actual link. Each group's **lead** is its best-merit member.
4. **Order.** Leads (the independent exposures) come first, then alternates, each saying "Alt. of #k" and why.
5. **Event theses.** Exposures reached by the same originating event share that event's thesis: if the reading of the news is wrong, all of them are. An exposure reached by several events belongs to each thesis. Theses are never chained through shared consequences (section 7).

## 6. Output: `RankedSignalSet` (`ranked-signal-set/1`)

Every signal carries:

- `tier`, `rank`, `role` (LEAD/ALTERNATE), `exposure_group_id`, `lead_id`;
- `merit` (raw metrics, normalization method, peer group, grades);
- `user_fit` (checks, admission, personalization);
- `mechanism` (provenance) and `shares_thesis_with`;
- one-line summaries: **scientific quality, user fit, data quality, liquidity & cost, redundancy, reason for rank**;
- **uncertainty**: typed notes, including:
  - screening only;
  - unconditional evidence;
  - the effective number of windows;
  - unstable years;
  - qualified data;
  - unresolved mechanism;
  - redundant exposure;
  - shared thesis;
  - multiple testing (one of N);
  - and, for unranked signals, "standing unknown, not low".

The set carries:

- the exposure groups and event theses;
- aligned correlations, ordered by the backend;
- the policy fingerprint and the mandate fingerprint;
- the family size;
- `headline()`. The headline always says whether **anything** has screening support. When nothing does: *"the order shows which hypotheses are least contradicted by 2018-2022 data, not which are promising."*

There is no field for weight, allocation, size, score, probability, expected return or verdict. A test walks the whole JSON to check this.

## 7. Real data: three events, 38 signals

`outputs/news_alpha/phase_f/THREE_EVENTS__signal_ranking.json` is written by `scripts/news_alpha_phase_f_signal_ranking.py`.

- **Events:** the AI capex, OPEC+ cut and Fed surprise events.
- **Mandate:** Futures + ETF + Equity.
- **Screening:** all 38 distinct candidates were screened on real, already-acquired bars in 252 s (5 futures roots, 14 ETFs; 2018-2022 only). The committed artifact is the `--cached-only` re-rank of those same screens.
- **Cost and scope:** $0; no network, no registry write, no 2023+ data.

Headline: **38 ranked signals = 7 independent exposures from 3 event theses, 5 of them reached by more than one event. None has screening support.**

| # | Lead (independent exposure) | Exposure group | Screen | t (declared direction) | Years | Cost | Data |
|---|---|---|---|---|---|---|---|
| 1 | SHY `ts_return(close, 60)` → 60D | SHY · ZN · TLT · IEF (8 signals, ETF + Futures, ρ ≤ 0.98, 2 events) | no support | +2.01 | 2/5 | timing edge | qualified |
| 2 | USO `ts_return(close, 20)` → 20D | USO · CL (4, ETF + Futures, ρ ≤ 0.97) | no support | +0.68 | 3/5 | timing edge | qualified |
| 3 | GC `ts_return(close, 20)` → 20D | GC · GLD (4, Futures + ETF, ρ ≤ 0.98, 2 events) | no support | +0.01 | 2/5 | timing edge | clean |
| 4 | LQD `ts_return(close, 20)` → 20D | LQD (2) | no support | +0.81 | 1/5 | timing edge | qualified |
| 5 | XLF `ts_return(close, 60)` → 60D | XLF · ES · NQ · IWM · HYG · SPY · QQQ · XLK (16, ETF + Futures, 3 events) | no support | +0.18 | 1/5 | no timing edge | qualified |
| 6 | XLE `ts_return(close, 20)` → 20D | XLE (2) | no support | −0.05 | 1/5 | no timing edge | qualified |
| 7 | XLU `ts_return(close, 60)` → 60D | XLU (2, 3 events) | no support | −1.25 | 0/5 | no timing edge | qualified |

**Reading the table.**

- **Futures and their ETFs are one exposure each:** ZN with IEF/TLT/SHY, CL with USO, GC with GLD, and ES/NQ with SPY/QQQ/XLK/IWM (and XLF and HYG). Thirty-eight hypotheses are really seven bets, and one equity-beta bet absorbs 16 of them.
- **Concentration:** five of the seven exposures are reached by more than one event. The equity complex and XLU are reached by all three.
- **SHY 60D leads on direction of evidence** (t +2.01, above the screen's threshold) but has **no screen support**: only 2 of 5 years agree. The ranking puts it first *and* says so.
- **Rank 3 vs 4:** GC (t +0.01) ranks above LQD (t +0.81) because they tie on every grade and GC's futures data is clean while LQD's ETF momentum is a price-only proxy. This is a review choice (section 11).
- **Futures-only mandate:** 28 signals are excluded, 4 exposures remain, and merit is **unchanged for every signal**.
- **Per event:** AI 8 → 2 exposures; OPEC 36 → 7; Fed 30 → 5.

**Two findings made on real data** (both covered by tests and negative controls):

1. **Pearson was rejected for redundancy.** CL's 20-day trailing return reached **+326%** in the rebound from April 2020's near-zero prices. That single outlier pulled its Pearson correlation with USO's to **0.74**; the rank correlation is **0.97**. With Pearson, CL and USO were two "independent" exposures. CL 20D vs 60D was 0.17 under Pearson and 0.56 by rank. A ±1 position follows the factor's sign and order, not its outliers, so redundancy now uses rank correlation (`correlation_method="spearman"`, in the policy fingerprint).
2. **Theses must not chain.** Linking theses through any shared consequence (AI and Fed both touch equity valuations; OPEC and Fed both touch Treasury yields) merged three unrelated news stories into "1 economic thesis". A thesis is now the originating event, and multi-event reach is reported per exposure instead.

A bug found by the live UI test on real ETF bars: an alternate can join its exposure only through lower-ranked members (QQQ 20D linked to QQQ 60D and XLK 20D, but not to its lead). Building the explanation crashed. It is fixed and covered by a regression test with a negative control.

## 8. UX (Agent → News Impact Triage)

- **Each event card**, under the candidate section:
  - a caption with the set's headline;
  - a **"Signal ranking for portfolio consideration · k independent exposure(s) · n ranked · m not rankable"** expander.
- **Inside the expander:**
  - the ranked table: rank, role (Lead / Alt. of #k), the exact formula → horizon, screen, evidence, t, years, stability, cost, data, *your mandate*, exposure, domain, why this rank;
  - tab **Why this rank?**: the six summaries, the uncertainty list, the raw metrics with their normalization method and peer group, a mandate-check table and mechanism provenance;
  - tab **Exposures & redundancy**: exposure groups (with the number of events reaching each), event theses, and aligned rank correlations (moved here from Phase E);
  - tabs **Not rankable** (why, and what would rank it) and **Excluded by mandate** (merit shown unchanged).
- **"Portfolio consideration across N events"**: a card above the event cards when two or more events carry candidates. It shows the same ranking over their union, where structural duplicates merge across events.
- **Chat:** the Agent chat adds a `SIGNAL RANKING:` line, plus `signal_ranking` evidence (fingerprint, tiers, ranks, roles, groups).
- **Rendering cost:** rendering reads cached screens only (no bars), so it is instant. Each event's pipeline is computed once per render and shared by the cross-event card and the event card.
- **Visual check:** verified live with `streamlit run` and Playwright at 1440 px and 420 px, with no page-level horizontal overflow. The first pass showed a truncated duplicate "Signal"/"Factor expression" pair; these became one exact-formula column.
- **Your saved profile applies live.** With a 1-3 Days horizon, High frequency and no shorting, every 20D/60D signal shows a holding-period and frequency mismatch and a shorting constraint. Phase E's generation grid has no short horizon, and the ranking says so rather than hiding it.

## 9. Tests

| File | Tests | Covers |
|---|---|---|
| `test_news_alpha_phase_f.py` | 44 | **Tiers:** missing ≠ low; adding unscreened candidates never moves ranks; insufficient data; non-real, unsupported-domain and non-comparable evidence; a screen filed under the wrong candidate. **Normalization:** one peer group for futures and ETFs, raw metrics preserved, grade boundaries at the policy's own thresholds, other policy or schema → non-comparable, ETF proxy = qualified. **Merit vs fit:** merit identical across five mandates, soft fit only breaks exact ties, exclusions keep merit, each fit dimension, shorting measured as a constraint. **Redundancy:** a cross-domain exposure with leads first; same instrument; offsetting; min common days; one outlier cannot split an exposure (rank correlation); lower-ranked link naming; structural merge across events; event theses never chained. **Ordering:** each merit dimension decides in its declared order; cost from timing, not drift. **Determinism:** independent of event and candidate order. **Provenance:** fidelity and path status never rank. **Honesty:** the headline never calls unsupported signals promising. **Boundaries:** no sizing, score or verdict field anywhere; pure module (no IO, registry, validation, LLM or Streamlit); registry bytes unchanged. **Real bytes:** QQQ and XLK are one exposure on real ETF bars. |
| `test_news_alpha_phase_f_ui.py` | 7 | Unscreened → every signal NOT RANKABLE (not low); stored screens → table, leads/alternates, the six aspects, uncertainty, normalization, exposures, theses; the current profile reaches the fit column without reordering; the cross-event card equals the backend ranking; the view never sorts; chat line + evidence; the real ETF run button ranks in the same card |
| `test_alpha_discovery_part_f_fast_screen.py` | +1 | The Discover page's migrated ordering (`rank_pool_members`) |
| `test_news_alpha_phase_e_ui.py` | 2 migrated | Correlation lives in the ranking view now |

The UI tests pin the investor profile. Without that, they read the user's real saved profile from `data/user_prefs/`, which surfaced during this phase.

**Negative controls.** Each control broke one guarantee in the live file, ran the test meant to catch it, and restored the file byte-for-byte. **All 15 were caught:**

| Broken guarantee | Caught by |
|---|---|
| Merit reads the mandate | `test_research_merit_never_reads_the_mandate` |
| Fit compared before merit | `test_soft_preferences_only_ever_break_exact_merit_ties` |
| Fidelity used to rank | `test_mechanism_confidence_and_fidelity_never_change_the_ranking` |
| Unsupported domain treated as merely unscreened | `test_unsupported_domains_…` |
| Correlation never links | `test_correlated_expressions_across_domains_…` |
| Offsetting correlation ignored | `test_offsetting_expressions_are_linked_too` |
| Normalizer ignores the screen policy | `test_evidence_from_another_method_or_policy_…` |
| Lead is not the best-merit member | `test_correlated_expressions_across_domains_…` |
| Headline overclaims support | `test_the_headline_never_calls_unsupported_signals_promising` |
| Excluded signal loses its merit | `test_the_mandate_excludes_without_rewriting_merit` |
| The UI sorts signals itself | `test_the_ranking_view_never_sorts_…` |
| Pearson redundancy | `test_redundancy_uses_rank_correlation_…` |
| Theses chained through consequences | `test_event_theses_are_never_chained_…` |
| Alternate linked only via lower-ranked members (crash) | `test_an_alternate_linked_only_through_lower_ranked_members_…` |
| Same instrument alone merges different factor families | `test_same_instrument_different_factor_family_is_not_automatically_one_exposure` |

## 10. Files

| New | Purpose |
|---|---|
| `recommendation/signal_ranking.py` | The canonical ranker: tiers, normalization, merit, fit, redundancy, theses, summaries |
| `ui/signal_ranking_view.py` | A pure renderer for event cards and the cross-event card, plus the chat line |
| `scripts/news_alpha_phase_f_signal_ranking.py` | Real three-event screening + ranking, and the provenance artifact |
| `outputs/news_alpha/phase_f/THREE_EVENTS__signal_ranking.json` | Real-data evidence |
| Two test files | As above |

| Changed | Why |
|---|---|
| `recommendation/fit.py` | `frequency_band` made public (one band definition, two users) |
| `screening/candidate_signal_screen.py` | `SCREENABLE_DOMAINS` (which domains have a real PIT loader) |
| `screening/factor_diagnostics.py` | Additive `signal_correlations(method=...)` |
| `screening/fast_screen.py`, `ui/views/discover.py` | Canonical `rank_pool_members`; the UI sort key was deleted |
| `ui/candidate_signal_view.py` | UI-side correlation tab removed (it moved to the ranking) |
| `ui/views/agent.py` | One pipeline per event; the ranking section per card; the cross-event card |
| `ui/conversation_engine.py` | Chat line and evidence |
| `scripts/news_alpha_pipeline_demo.py` | The ranking stage |
| `news_alpha/__init__.py`, `recommendation/__init__.py` | Pipeline and owner docstrings only |

There was no change to C++, the registry, experiment identity, BH-FDR, `ReliabilityPolicy`, `AssetDomain`, or Phase A–E semantics. There was no holdout access and no paid or network call.

## 10.5 Acceptance patch (user review)

Three corrections, made before Phase F was accepted, none touching real-data outcomes today (one signal family exists, so today's 7 exposures are unchanged):

1. **Ranked ≠ portfolio-eligible.** `RankedSignalSet`'s docstring, `not_portfolio_note`, and the module docstring now say explicitly that a rank is never an eligibility decision. A future portfolio-construction phase needs its **own** allocation/eligibility gate (capital, margin, position-count caps, risk budget) — this schema has no field for it and none should be inferred from `rank`/`tier`.
2. **Same instrument is not permanently same exposure — fixed, a real defect.** `RedundancyRelation.SAME_INSTRUMENT` unconditionally linked any two signals sharing an instrument, regardless of what generated them. With today's single signal family this never showed up (every same-instrument pair happens to share it), but it would have silently merged, say, a momentum candidate and an unrelated mean-reversion candidate on NQ the moment a second family existed — never because they were actually related. Renamed to `RedundancyRelation.SAME_STRUCTURAL_FAMILY`: the link now requires the same instrument **and** the same `signal_rule_id` (the generation rule); a different family only links through actual correlation. Regression: `test_same_instrument_different_factor_family_is_not_automatically_one_exposure` builds a synthetic second family on NQ with deliberately uncorrelated data and asserts the two do **not** share an exposure group; caught by the 15th negative control.
3. **The redundancy threshold is a versioned policy, not a scientific finding — strengthened, already true.** `SignalRankingPolicy.redundancy_correlation` (and `.correlation_method`) were already a fingerprinted, fixed-before-ranking field (`policy_fingerprint`, exercised by `test_a_correlation_on_too_few_common_days_never_links`) and were already shown in the UI caption rather than asserted as truth. The docstrings on `SignalRankingPolicy` and the module now say this explicitly, so it cannot be read as a scientific cut-off by a later reader.

## 11. Not done, and choices to review

- **Not started, by design:** portfolio weights and construction, validation, alpha-memory promotion.
- **Choices to review:**
  - the merit order: screen → direction → stability → cost → data quality → t. In particular, data quality comes before t, which is why GC outranks LQD;
  - redundancy at |rank ρ| ≥ 0.8 over ≥ 250 days, with single linkage. This errs toward fewer claimed independent exposures;
  - same-instrument signals are always one exposure;
  - shorting is a constraint, not an exclusion;
  - the holding-period windows;
  - flips per year from the rebalance-every-h model.
- **Next useful steps:**
  - **Measured liquidity** (median traded notional over the screen window), so the mandate's liquidity requirement can finally apply;
  - **a long/flat variant** for no-shorting mandates, as its own predeclared candidate;
  - **a shorter-horizon generation point** if short holding periods matter to you: your saved 1-3 Days profile mismatches every current signal;
  - portfolio construction over the independent exposures, with each exposure's lead as the default expression.
