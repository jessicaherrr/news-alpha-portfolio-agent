# News Alpha -- Persistent Research Thread Workspace (UI phase)

Status: implemented and tested, stopped for review (not frozen). This document is both the pre-build
audit/design (section 29 of the phase prompt: written BEFORE any UI was
rewritten or removed) and the record of what was built.

The backend stays layered (Mandate -> Information -> Economic Reasoning ->
Market Translation -> Measurement -> Quant Research -> Portfolio -> Scientific
Engine -> Learning). The UI compresses those layers into ONE persistent
Research Thread that a user enriches step by step:

    Research Setup (persistent context, not a step)
    News -> Reasoning -> Market -> Signals -> Portfolio -> Backtest -> Learn

---------------------------------------------------------------------------

## 1. Current UI inventory (before this phase)

Visible sidebar pages (`ui/app.py`, custom nav in `ui/layout.py`):

| Page | Module | What it holds today |
|---|---|---|
| Agent (default) | `views/agent.py` (2.5k lines) | hero + Top Opportunities (futures triage), **News Impact Triage** (the entire News Alpha A-H pipeline: mandate strip/editor, "describe an event", one stacked card per event with nested expanders for graph, paths, expressions, candidates, ranking, portfolio, validation loop, plus the Phase 1 translation hand-off), Claude conversation, advanced hypothesis composer |
| Market | `views/market.py` | all-futures scanner + product detail (overview, contracts, relative, news & events, research) |
| Research | `views/research.py` | landing tabs Workflow / My Alpha / Research Map / Community / Provenance, drill-downs Strategies / Experiments / Validation, and the experiment Detail view (hand-off target via `research_details_target`) |
| Paper | `views/paper_trading.py` | replay paper trading (registry strategies) |
| Learn | `views/learn.py` | educational layer: Concepts / Strategies / My Research (failure autopsy) / Learning Paths |
| Settings | `views/system.py` | providers, runtime, Claude API, about |

Hidden pages (reachable by hand-off only): Discover, Dashboard, Strategies,
Backtests, Validation, Experiment Log, Crypto Lab.

News Alpha stage renderers (all pure renderers over canonical backend
objects, currently only mounted inside Agent's triage card):
`news_alpha_context` (the one UI boundary into `alpha_agent.news_alpha`),
`mechanism_graph_view` (B), `signal_path_view` (C), `asset_expression_view`
(D), `candidate_signal_view` (E, + the explicit screening action),
`signal_ranking_view` (F), `portfolio_plan_view` (G, + explicit snapshot
loading), `research_loop_view` (H: gate, memory, path studies).

Shared shell: `layout` (CSS, header, sidebar), `components` (card, badge,
metric/gate rows, empty state, provenance rows, plotly wrapper), `palette`,
`charts`, `charts_market`, `market_home` (session-cached observation data).

Persistence that already exists: `MandateStore` (access half of the mandate,
`data/user_prefs/research_mandate.json`), `ProfileStore` (risk half,
`investor_profile.json`), session-only user-described events
(`news_alpha_context`), operational caches (`FactorScreenStore`,
`MarketSnapshotStore`), and the registry (read-only from the UI).

## 2. Old -> new mapping

| Old surface | New home | Treatment |
|---|---|---|
| Agent -> News Impact Triage (feed, describe event) | **News** page | MERGE (moved; Agent keeps the conversation) |
| Agent -> mandate strip + editor | **Research Setup** (one dialog, shown on News and in every thread) | MERGE |
| Agent triage card "Why these domains?" | Research -> **News** step (initial impact detail) | REUSE INSIDE WORKSPACE |
| `mechanism_graph_view` | Research -> **Reasoning** (graph is the primary visual; full tabs under Advanced) | REUSE (+ path highlight) |
| `signal_path_view` | Reasoning path cards + filters; full tables under Advanced | REUSE + new cards |
| `asset_expression_view` | Research -> **Market** (one card per market, "Measure it"); full tables under Advanced | REUSE + new cards |
| Market page price chart / `market_home` cache | Market step price trend (futures, explicit load, observation only) | REUSE |
| `candidate_signal_view` | Research -> **Signals** (cards, decay chart, explicit screening) | REUSE + new cards |
| `signal_ranking_view` | Signals (ranking) + **Portfolio** page (across threads) | REUSE |
| `portfolio_plan_view` | Research -> **Portfolio** step; **Portfolio** page across threads | REUSE (+ public `build_plan`) |
| `research_loop_view` | Research -> **Backtest** (gate, recorded runs) and **Learn** step (route memory) | REUSE |
| Agent translation cards (Phase 1 -> strategy hypothesis) | Signals step, "Test as a single-market strategy" (hands off to Ask) | MERGE (code moved to `ui/translation_view.py`) |
| Research landing My Alpha / Research Map / Community | **Learn** page tabs | MERGE |
| Research landing Workflow / Provenance / drill-downs / Detail | **Strategy Lab** (tool page, url `lab`) | KEEP (renamed) |
| Learn: Concepts / Strategies / My Research / Paths | Learn -> **Guides** tab | MERGE |
| Market | **Markets** tool (unchanged) | KEEP |
| Agent (conversation, opportunities, composer) | **Ask** tool | KEEP (triage removed) |
| Paper | **Paper trading** tool | KEEP |
| Hidden legacy pages | unchanged, still hidden | HIDE FROM PRIMARY NAV (already) |

## 3. Navigation

Primary (the research journey):

    News (default landing)  ·  Research  ·  Portfolio  ·  Learn

Tools (smaller, below a divider): Ask · Markets · Strategy Lab · Paper trading.
Settings last. Research Setup is not a destination: it is a persistent
control on News and in every thread.

URL paths: `news`, `research` (thread workspace, deep-linkable with
`?thread=<id>&step=<step>`), `portfolio`, `learn`, `agent`, `market`, `lab`
(the former Research page), `paper-trading`, `system`.

## 4. Research Thread state model

A thread stores REFERENCES and the user's navigation/focus -- never a copy of
a scientific result. Everything else is recomputed from the canonical backend
on render (the pipeline stages are pure and take milliseconds; screens,
snapshots and registry evidence come from their existing stores).

```
ResearchThread (research-thread/1, data/user_prefs/research_threads.json, gitignored)
  thread_id            "thread-<sha256(event_id)[:12]>" -- one thread per event
  name                 human name (the mechanism graph's anchor state, else the headline)
  event                ThreadEventRef: event_id, kind, headline, and the source
                       object (MarketNewsItem | ScheduledMarketEvent |
                       UserDescribedEvent) so the scan is reproducible after
                       the news cache rotates or the session ends
  current_step         News | Reasoning | Market | Signals | Portfolio | Backtest | Learn
  completed_steps      explicit progression (Next marks a step done)
  setup_fingerprint    ResearchMandate.fingerprint() the Market-onward choices
                       were made under (Research Setup id)
  followed_path_ids    Reasoning focus (SignalPath.path_id)
  focus_expression_id  Market focus (AssetExpression.expression_id)
  mechanism_graph_id   last seen graph fingerprint (drift notice only)
  reached_step         furthest step opened (stepping back never makes it unreachable)
  created_at / updated_at (UTC)
```

Resolved live, shown under Advanced details: impact scan, mechanism graph
id, signal-path id, asset-expression id, candidate-signal ids, ranked-set
fingerprint, portfolio plan fingerprint, validation experiment ids.

Research Setup = the user's ONE `ResearchMandate` (access half via
`MandateStore`, risk half via the saved `InvestorProfile`) -- deliberately not
a per-thread copy: `MandateStore` exists so the two halves can never drift
into disagreeing copies. Each thread records the setup fingerprint its
Market-onward choices were made under; when the setup changes, the thread
warns, clears focus that no longer continues, and marks Market onward as
needing review (Reasoning is mandate-independent by construction and stays).

Selections are FOCUS, never a change to the tested family: candidate signals,
ranking and the portfolio are always built from the event's full canonical
candidate set. A user deselecting paths or markets after seeing screen
results would otherwise shrink the multiple-testing family post hoc.

## 5. Files reused unchanged

`news_alpha_context` (+1 additive helper), `mechanism_graph_view` (+1 optional
highlight argument), `signal_path_view`, `asset_expression_view`,
`candidate_signal_view`, `signal_ranking_view`, `research_loop_view`,
`components`, `palette`, `charts`, `charts_market`, `market_home` (+1 read-only
peek), every hidden legacy page, `alpha_library` / `alpha_graph` / `community`
bodies (only their deep-link keys retargeted to Learn).

## 6. Files extended

`app.py` (pages), `layout.py` (nav + workspace CSS), `views/agent.py`
(triage section and translation cards moved out), `views/research.py`
(becomes Strategy Lab), `views/learn.py` (My Alpha / Research Map /
Community / Guides), `portfolio_plan_view.py` (public `build_plan`),
`services.py` (read-only route-memory listing).

## 7. Files added

`ui/research_thread.py` (state, store, pipeline resolution),
`ui/research_setup_view.py`, `ui/translation_view.py`, `ui/impact_scan_view.py`,
`ui/workspace/` (`common`, `shell`, `step_news`, `step_reasoning`,
`step_market`, `market_data_panel`, `step_signals`, `step_portfolio`,
`step_backtest`, `step_learn`), `views/news.py`, `views/workspace.py`
(Research), `views/portfolio.py`.

## 8. Delete candidates (NOT deleted in this phase)

None deleted. Candidates for a later cleanup once the new surfaces have been
reviewed: the Agent-only `_render_portfolio_consideration` / triage helpers
(now moved), and the per-event `expander_label` / `summary_line` caption
helpers in the stage renderers if nothing but the retired stacked card uses
them. Every hidden legacy page stays: each is still reachable by an existing
hand-off and covered by its own tests.

## 9. Migration risks

1. Tests that pin the old IA (six visible pages, Agent default, `research`
   url for the experiment Detail view, triage rendered on Agent). Migrated
   deliberately, not deleted: each assertion is retargeted to the surface the
   behaviour moved to.
2. Hand-offs into the experiment Detail view (`research_details_target` +
   `switch_page(url_path="research")`) -> retargeted to `lab`.
3. My Alpha / Research Map / Community deep links used
   `research_landing_focus`; they now use `learn_landing_focus`.
4. First paint: nothing on News or Research may block on network or a
   screen; recent prices, screening and snapshot loading stay explicit
   buttons (the Opportunity first-paint lesson).
5. The market-observation plane (delayed Databento prices) must never feed
   a signal or a validation -- it is display-only in the Market step.

## 10. Implementation sequence

A audit (this document) -> B thread state + navigation shell -> C Research
Setup -> D News entry -> E Reasoning -> F Market + Measurement -> G Signals ->
H Portfolio -> I Backtest + Validation -> J Learn (My Alpha / Research Map /
Community) -> K end-to-end UI test + live browser pass -> L remove only
proven-dead code.

---------------------------------------------------------------------------

## As built

### Screens

* **News** (default landing, `/`): Research Setup strip + editor dialog,
  "continue your research" cards, describe-an-event, a 24h / 7d / 30d feed of
  cached news, releases (dated by their SCHEDULED time) and your own events,
  each with a per-asset-class impact meter, its economic channels, a primary
  **Research** action and an "Explore" explainer. **Refresh news** is the one
  explicit fetch (free official sources, the same action as Markets).
* **Research** (`/research`, deep-linkable `?thread=<id>&step=<step>`):
  thread header + thread switcher, Setup strip, a seven-step stepper (✓ done,
  ● current, ○ not yet; unreachable steps disabled with a tooltip), the step,
  Previous / Next, and a Research Context rail (event, assets, horizon,
  selected path, current market, signals, portfolio, validation, thread
  references). Without an active thread: the list of your threads.
  1. **News** -- event detail, related markets, initial impact meters with
     the mechanism check beside them, explainer, related events, provenance.
  2. **Reasoning** -- the transmission graph (followed paths drawn in the
     accent, "view larger" dialog), Direct / Supply chain / Cross sector
     filter, compact path cards with **Follow** and **Inspect links**;
     Advanced: consequences, links & sources, open questions, all paths,
     mechanism check.
  3. **Market** -- one card per market (the AI slice reaches NQ through six
     consequences -- it is one card, listing them), split into "Measurable
     today" and "Affected, not measurable yet"; each with the initial scan's
     own exposure level, path types, horizon, route confidence, measurable
     count and its economic lineage. **Inspect** opens the price trend
     (futures: delayed CME bars behind an explicit, cost-checked load; ETF /
     equity: honestly "not connected" / "not acquired"), asset-class market
     data, and **Measure it** (economic consequence -> ideal measurement ->
     actual data field -> PIT status -> ready to become signals -> what is
     missing). Other exposures and the consequence->market map are collapsed.
  4. **Signals** -- origin lineage, counts, the explicit screening action,
     candidate cards (formula, one-line facts, diagnostics, Lineage and
     Diagnostics popovers), the ranking leads, and the single-market strategy
     route (the Phase 1 translation, handing off to Ask).
  5. **Portfolio** -- Qualified / Exploratory preview, market-data load
     action, metrics, selected signals, holdings, a constraint checklist,
     "not in the portfolio" with reasons; Advanced: the full plan tabs.
  6. **Backtest** -- scientific status (gate or recorded registry verdict
     only) + holdout badge; Performance (the recorded run's C++ results via
     Strategy Lab's own renderer, else an honest empty state + the CLI
     command) and Validation (why not yet, recorded gate grid, what a run
     checks, path-level studies).
  7. **Learn** -- the thread end to end (event -> ... -> evidence), this
     run's route memory vs the recorded memory, links to My Alpha / Research
     Map / Community.
* **Portfolio** (`/portfolio`): contributing threads, the ranking across all
  of them, the plan, and the validation / memory card.
* **Learn** (`/learn`): My Alpha (threads, every recorded transmission route,
  then the factor / strategy library), Research Map, Community, Guides
  (Concepts, Strategies, Failure autopsy, Learning paths).
* **Tools**: Ask (the Agent page without the triage), Markets, Strategy Lab
  (the former Research page at `lab`), Paper Trading. Settings unchanged.

### Design decisions worth reviewing

1. **Focus, not family.** Following paths and inspecting markets narrows
   what is shown; candidate signals, ranking and the portfolio are always the
   event's full canonical set, so a post-hoc deselection can never shrink the
   multiple-testing family.
2. **One Research Setup.** The Setup edits the user's single mandate +
   profile (their existing single sources of truth); threads record the
   setup fingerprint and, on a change, reset Market onward to "needs review"
   (current step moves to Market) with a notice. Reasoning is kept.
3. **Market impact per market** is the Initial Impact Scan's OWN exposure
   level for the exposure the scan put that market in (a lookup, first match
   in the scan's order); markets the scan never named say "via mechanism
   only". No impact number is computed in the UI.
4. **Prices are observation-plane only** and only load on an explicit click
   through the existing cost-guarded provider (1D / 5D / 1M -- the delayed
   feed's 30-day ceiling rules out 3M / 1Y rather than offering and refusing
   them).
5. **Validation runs stay on the command line** (they write the registry);
   the Backtest step shows the command, never a run button.
6. **One thread per event** (`thread-<sha256(event_id)[:12]>`), persisted to
   gitignored `data/user_prefs/research_threads.json` with the event's source
   object, so a thread survives the session and the news cache.

### Old view classification (cleanup audit)

| View / module | Classification |
|---|---|
| `views/agent.py` | KEEP (Ask) -- triage + translation moved out |
| `views/research.py` | KEEP as Strategy Lab (`lab`) |
| `views/research_workflow.py`, `strategies.py`, `experiment_log.py`, `validation.py`, `backtests.py` | REUSE INSIDE Strategy Lab (unchanged) |
| `views/alpha_library.py`, `alpha_graph.py`, `community.py` | REUSE INSIDE Learn (only deep-link keys changed) |
| `views/learn.py` | MERGE (four tabs; Phase 4 content under Guides) |
| `views/market.py` | KEEP (Markets tool; also the Market step's "Open in Markets") |
| `views/paper_trading.py`, `system.py` | KEEP |
| `views/discover.py`, `dashboard.py`, `crypto_lab.py` | HIDE FROM PRIMARY NAV (already hidden, still hand-off targets) |
| News Alpha stage renderers | REUSE INSIDE WORKSPACE (Advanced details + shared row builders) |
| `signal_path_view.initial_badge_row`, `*.expander_label` helpers | DELETE CANDIDATE -- only the retired stacked triage card used them; kept for now (tested, harmless) |

Nothing was deleted: every module is still imported by a live page or by its
tests. The retired Agent triage code lives on in git history (2cd296b..).

### Tests

New: `tests/python/test_research_thread_workspace.py` (36) + the shared
`tests/python/news_alpha_thread_support.py`. Migrated (behaviour retargeted
to where it moved, never dropped): News Alpha Phase A-H UI tests, chat
continuity, Phase 1 translation, Phase 3 context retrieval, translation ->
memory hand-off, navigation consolidation, sidebar, Learn, Research Map,
Community, My Alpha, Ask -> Strategy Lab navigation, opportunity cards,
research defaults, product refactor, Phase 20 entrypoint, release nav order.

### Remaining UI technical debt

* The stepper shows a visited-but-not-continued step as "not yet" (it stays
  reachable); a distinct "visited" state may read better.
* ETF / equity price trends need a recent-data source (none connected).
* The Research Setup strip wraps "Edit setup" onto its own line on narrower
  screens; a compact summary popover could replace the chip row there.
* The mechanism graph is still small inside the main column on laptop widths
  -- the "View larger" dialog is the workaround.
* Delete candidates above, once reviewed.

---------------------------------------------------------------------------

## Follow-up: Market UX consolidation (one Market)

"Market" (the thread step) and "Markets" (a tool) were two user-facing
concepts. They are now ONE destination, **Market** (url `market`), in the
research journey: News · Research · Market · Portfolio · Learn; Tools are Ask,
Strategy Lab, Paper Trading.

* **Two views of one product area** (`views/market.py`, segmented control,
  session key `market_view`): **Affected Markets** -- the active thread's
  markets only -- and **Explore All Markets** -- the existing futures explorer
  (scanner, search, filters, product detail, contracts, term structure,
  relative, news & events, research tab), unchanged and only rendered when
  chosen. With an active thread: "Researching: <headline>" + Open research
  thread, default Affected. Without: a notice + Start from News, explorer.
* **One implementation**: `workspace/step_market.render_affected_markets(pipe)`
  is rendered by BOTH the thread's Market step and the Market destination;
  the thread supplies the event, followed paths, expressions, measurements
  and Research Setup through its `ThreadPipeline`.
* **Market-first layout**: a wrapping pill selector of every affected market
  (measurable first; default = the first measurable market with a price feed,
  persisted as the thread's current market), then the selected market: name,
  impact / path / horizon / route confidence line, why affected, price trend
  (1D / 5D / 1M, explicit load) beside the asset-class market data, Measure
  it. All affected markets (table), the consequence map, other exposures and
  Advanced details are collapsed below. The old per-market cards + Inspect
  button are replaced by the selector.
* **Hand-offs** that mean the general explorer (Ask's "Open Market", the
  panel's "Full market detail") go through `market.open_explorer(root)`,
  which selects the Explore view -- never the thread's affected markets.
* Layout fixes found in the live pass: `st.pills` placed directly in a column
  scrolls on one row unless `wrap=True`; the main column of the main/rail
  split now has `min-width: 0`, so a wide child can never push the page wider
  than the viewport and drop the context rail below.
