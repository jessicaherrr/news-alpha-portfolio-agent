# Streamlit Research Interface (Phase 20 / 20.1 / 20.1b / Release UX)

A presentation/control layer over the existing deterministic system
(CLAUDE.md "Architecture boundaries"). It renders what the C++ quant core and
the Python research/registry layers already computed; it never owns or
recomputes PnL, fills, risk, validation, scientific verdicts, experiment
identity, or holdout decisions, and it exposes no control that bypasses frozen
validation, risk, duplicate authority, multiple-testing control, or the 2025
holdout isolation.

Phase 20.1 redesigned the interface into a dark, institutional quant-terminal
look (a specific reference screenshot was the visual spec) and fixed two
integrity blockers found in Phase 20 -- see "Phase 20.1 integrity fixes"
below. Phase 20.1b is a small, presentation-only correction to the second
fix's epistemic states (no UI/visual change): `NOT_EVALUATED` was itself
being inferred from an absent reason code, which needed its own committed
proof exactly like `PASS` does.

**Release UX** made the AI Research Agent conversation the default landing
page (previously the analytics dashboard) and added an optional, read-only,
DELAYED-only IBKR TWS API market-context quote strip -- see "Agent-first
navigation" and "IBKR delayed market context" below.

## Launch

```
pip install -e '.[ui]'          # streamlit>=1.38, plotly>=5.24
streamlit run python/alpha_agent/ui/app.py
```

Run from the repository root. `.streamlit/config.toml` supplies the dark
theme (native `st.dataframe`/chart theming); no API key is required for the
default, fully offline demo mode. To enable the optional live-LLM mode on the
Research / Strategies pages, set `ANTHROPIC_API_KEY` in the environment
*before* launching Streamlit -- see "LLM modes" below.

## Pages

Ten pages share one shell (dark sidebar with brand/nav/Markets/Data
Status/Research Window, a custom top status bar, and the same card/badge/chart
vocabulary throughout). **Agent is the default landing page** (Release UX --
AGENTIC ALPHA is an AI-native research system, not an analytics dashboard
first):

| # | Page | What it shows | What it never does |
|---|------|----------------|---------------------|
| 1 | **Agent** | The default landing page: a conversational research workspace over the real, frozen Phase 16/17/18 pipeline -- typed HYPOTHESIS / FAILURE MEMORY / COMPILED STRATEGY / BACKTEST RESULT / VALIDATION cards and a bounded next-research suggestion, plus a Research Context rail (market, dataset windows, current hypothesis, agent state, activity feed). | Decide PnL, fills, risk, verdict, FDR, DSR, validation, or paper eligibility -- the conversation only orchestrates and explains the deterministic system. |
| 2 | **Dashboard** | Secondary analytics page (formerly "Overview"): one root/family's authoritative registry evidence (metric row, robustness grid, recent experiments), cross-market evidence for the selected family, Sharpe-vs-BH-FDR and Validation Gate Outcomes registry-wide, plus the AI agent rail (propose/compile, activity feed, system status). | Compute a verdict, PnL, or duplicate decision. |
| 3 | **Research** | The full Phase 16 `ResearchAgent.propose(...)` workflow: the complete `HypothesisSpec`, the registry evidence consulted, prompt/provenance log. | Decide anything about the hypothesis's merit. |
| 4 | **Strategies** | The Phase 11 baseline family catalog, the Phase 17 `StrategyCompilerAgent.compile_hypothesis(...)` workflow, and the registry-wide Strategy x Market Matrix (every canonical trial, every root/family). | Build a scientific `experiment_identity` or decide re-execution. |
| 5 | **Backtests** | A tear-sheet for one experiment: authoritative `ResultRecord` metrics, cost-stress / bootstrap / regime charts (all directly from that experiment's own committed evidence), and a trade ledger **shown only when a committed record verifiably binds it to that exact experiment**. | Derive an "equity curve" or "drawdown" by summing a same-root file, or attach any ledger by root symbol alone. |
| 6 | **Validation** | Every reliability gate (Bootstrap, BH-FDR, DSR, Walk-Forward, Parameter Stability, Cost Stress, Regime, Cross-Market) plus the Phase 15B ML meta-labeling adjudication and the registry-wide Verdict Distribution / Failure Reason Distribution. Each gate badge is **PASS or NOT_EVALUATED only on their own explicit committed proof**, otherwise FAIL / REFUSED_BEFORE_GATE / NOT_AVAILABLE. | Apply a threshold, run a statistical test, or infer PASS *or* NOT_EVALUATED from an absent reason code. |
| 7 | **Experiment Log** | Registry summary, a filterable browser over every experiment, `FailureMemory.lookup(...)`, the engineering-vs-scientific failure split, import/lineage provenance. | Insert, update, or delete a registry row. |
| 8 | **Paper Trading** | A `Phase 21 -- Not Enabled` banner and six empty, explicitly `PHASE 21 REQUIRED` placeholder cards. | Show a single fabricated number, position, order, or kill-switch state. |
| 9 | **Crypto Lab** | The Phase 22 fully-synthetic on-chain/crypto scaffold: causal envelope, synthetic registry evidence. | Represent synthetic crypto evidence as real market data or blend it with the futures registry. |
| 10 | **System** | Real system/data/research-safety/provenance status (git commit, engine build state, catalog contents, research/validation/holdout windows, registry fingerprints) and the architecture diagram. | Fabricate a "connected" indicator. |

Every page ends with the disclaimer: *"Historical backtest and validation
evidence. Past simulated performance is not a guarantee, projection, or
promise of future returns."*

The Agent and Dashboard pages additionally render a market-context quote
strip (see "IBKR delayed market context" below) when it is enabled and a
locally-running TWS/IB Gateway is reachable; it renders nothing when disabled
and an honest disconnected banner when enabled but unreachable.

## Market Reality / HOME TERMINAL (MARKET REALITY + VISUAL PRODUCT PASS,
## then HOME TERMINAL + RUNTIME CONNECTIVITY FINAL PASS)

A normal `streamlit run python/alpha_agent/ui/app.py` discovers a local
`.env`'s `DATABENTO_API_KEY` automatically -- `app.main()` calls
`bootstrap_local_runtime_environment()` once, explicitly, at real startup
(never at arbitrary module import time, which is what previously leaked a
key into the pytest process and required a manual `source .env` before
launch; see that function's own docstring and `tests/python/conftest.py`).

The Agent (Home) and Market pages both open on real Databento futures data,
via one shared module, `alpha_agent.ui.market_home`, styled as a compact
market terminal rather than a developer status page:

- **Ticker strip** (`render_ticker_strip`): five compact, bordered CLICKABLE
  cards (ES/NQ/CL/GC/ZN) -- root, last, change %, and a real sparkline (
  `charts.sparkline`, reusing the SAME cached bars every other panel already
  fetched, never a decorative-only request) -- not a large primary table.
  Each card's Select button drives the SAME `selected_root` session key the
  sidebar's Market Browser uses -- one authoritative selection, never
  duplicated controls.
- **Selected Market hero** (`render_selected_market_hero`): large,
  investor-readable price + change, with Root/Display Contract distinction
  and every scientific caveat moved into a "Data Details" expander rather
  than sitting in the primary flow.
- **Chart + Core Metrics** (`render_chart_and_metrics` /
  `render_core_metrics_panel`), side by side, ABOVE THE FOLD: candlestick +
  volume (1m/5m/15m/1h/1d; 5m/15m are an honestly-labeled resample of real
  1m bars) paired with a compact Last/Change/Session High/Session
  Low/Volume/Realized Volatility/Session Range/Distance-from-High/
  Distance-from-Low list.
- **Market Context** (`render_market_context`): four short, colored pills --
  Trend/Volatility/Session Position/Volume -- descriptive only, never
  labeled SIGNAL/BUY/SELL (session position coloring is deliberately neutral
  blue/grey; "near high" is a location, not a judgment).
- **Disconnected state** (`render_unavailable_state`): ONE concise
  "MARKET DATA UNAVAILABLE" card with a Retry button, never a table of N/A.
- **Compare Markets** (Market page only -- Home stays the fast overview,
  Market is deeper exploration): normalized return overlay, per-market
  realized volatility, and a descriptive Pearson correlation ("no causal
  interpretation is implied").
- **Research on `<root>`** (Agent/Home only, `agent._render_research_snapshot`):
  Validated/Promising candidates FILTERED to the selected root -- never raw
  experiment rows, fingerprints, or JSON.
- **Compact header** (`layout.render_header(compact=True)`, Agent/Home
  only): three product-facing pills (Market Data/Engine/Research) replace
  the five developer ones, which move into a "System status" expander along
  with the commit hash -- every OTHER page keeps the full header unchanged.

Never "Live from Databento" -- only whatever `DatabentoCapability` the
provider's free `metadata` probe actually proved, with the exact
observation timestamp and computed age.

**Cache-first, no rerun storm**: a UI-session cache
(`DatabentoMarketDataConfig.ui_refresh_interval_seconds`, default 90s) sits
in front of the provider's own request-level TTL cache -- a bare Streamlit
rerun never re-issues a network call; "Last refreshed" (this session's own
cache time), "Latest observation" (the data's own timestamp), and "Data age"
are always labeled as three distinct concepts (short human-readable phrase
on the primary surface, exact field-labeled values in Data Details).
Deliberately NOT backed by a new on-disk cache (`market_home`'s own module
docstring records why, and when to revisit that call).

**First paint, prioritized**: `render_home_terminal` fetches the SELECTED
root's own snapshot/OHLCV FIRST, then reserves the ticker strip's position
at the TOP of the page via `st.container()` while filling it LAST, after
the hero/chart/metrics/context have already been written -- Streamlit
streams each element to the browser as it is produced regardless of which
container it belongs to, so the important content (what the user actually
asked to see) appears first without changing where the strip sits visually.
Measured against the real configured entitlement, one root's
resolve-display-contract + OHLCV round trip took 4-11 real seconds; the
other four roots are fetched CONCURRENTLY
(`market_home.get_snapshots_for_universe`, a `ThreadPoolExecutor` fan-out
over the plain, Streamlit-free `databento_context.market_snapshot`, merged
back into session state on the main thread only), bringing a full cold
Agent-page render from ~70s (serial) to ~15-30s, with the hero/chart
themselves visible well before that. This is a UI-layer-only change -- it
never alters what is fetched, cost-gated, or cached, only how many
independent real round trips overlap and in what order they stream.

**Strategy Details tabs** (`views/research.py`, "Research Details"): Overview
/ Hypothesis / **Performance** (renamed from "Backtest"; equity, drawdown,
Max Drawdown) / **Signals** (real price + fill markers + actual position,
shared with the Backtests tear sheet via
`views.backtests.render_signals_section`, honestly NOT_AVAILABLE unless a
committed artifact bundle is verifiably bound to that exact experiment) /
Validation / **Sources** (a hypothesis's cited `source_inspirations`, if
any) / Provenance / Memory & Lineage / Raw / Logs.

**Test safety**: `tests/python/conftest.py` gives every test in the suite a
fast, deterministic NOT_CONNECTED default for `databento_context` -- several
test modules call `load_dotenv_if_available()` at import time, which (when a
local `.env` carries a real `DATABENTO_API_KEY`) leaks that key into
`os.environ` for the rest of the pytest session; without this fixture, any
Agent/`app.py`-rendering test that does not explicitly mock
`databento_context` would make a real, unbounded-latency network call.
`test_databento_ui_context.py` opts out (`pytest.mark.real_databento_context`)
because it tests that module's own delegation/degradation behavior.

## Phase 20.1 integrity fixes

**1. No more root-matched trade ledgers.** Phase 20 attached
`outputs/phase_15/_run_real_work/<ROOT>/run_*/trades.csv` to whatever
experiment shared that root, and plotted a UI-side cumulative sum of its
`net_pnl_usd` column as an "Equity Curve"/"Drawdown". Both are wrong: those
`run_000N` directories are the Phase 15B ML-engine-call cache's
execution-*order* scratch output (`alpha_agent.ml.engine_io`'s internal
counter) -- nothing on disk records which of 12 pipelines x 5 cost scenarios
produced a given `run_0001`, and in any case they belong to the Phase 15B ML
family, never the Phase 13.5C canonical trial a user is usually inspecting.
`services.find_experiment_bound_trade_ledger(detail)` replaces both: it looks
ONLY at the selected experiment's own `ResultRecord.source_artifact`, requires
it to literally name a `trades.csv`, and (when `source_artifact_sha256` is
recorded) verifies the file's hash before ever reading it. Every committed
Phase 13.5C/15B result's `source_artifact` is a `validation_report.json` /
JSON report today, so this correctly returns `None` for all 167 registry
experiments -- proven by
`test_no_experiment_in_the_registry_has_a_bound_trade_ledger_today`. The
Backtests page shows the authoritative summary metrics (from `ResultRecord`
directly) plus an explicit "NOT AVAILABLE" card instead of a chart.

**2. No more inferred gate PASS (Phase 20.1) -- then no more inferred
NOT_EVALUATED either (Phase 20.1b).** Phase 20's gate table read PASS
whenever a gate's failure reason code was *absent*. That is unsound: the
committed Phase 13.5C report never serialises the frozen policy's actual
per-gate boolean (`alpha_agent.validation.policy.PolicyOutcome.gate_results`
is computed in memory and dropped by the report writer), and an absent code
is equally consistent with "never evaluated" (an earlier minimum-sample
refusal short-circuits every later gate) as with "passed". Phase 20.1's first
pass fixed the PASS half of this but still labeled every other absent-code
case `NOT_EVALUATED` -- itself an unjustified factual claim, since
"NOT_EVALUATED" asserts something as specific as "PASS" does and needs its
own committed proof. `services.gate_state(...)` now resolves strictly by
priority, each step requiring its own explicit committed evidence:

1. this gate's own fail code present -> **FAIL**
2. `all_required_gates_satisfied` present (verdict PASS) -> **PASS**
3. this gate's own paired `*_not_evaluated` code present -- only Regime
   Robustness (`regime_not_evaluated`) and Cross-Market Evidence
   (`cross_market_not_evaluated`) have one -> **NOT_EVALUATED**
4. a minimum-sample code present (Section 8's early return, which provably
   short-circuits every later gate) -> **REFUSED_BEFORE_GATE**
5. otherwise -- including every REJECT-verdict gate whose own fail code
   never fired, and the other 7 gates under an evidence-family INCONCLUSIVE
   -- the committed record cannot distinguish "passed" from "never
   evaluated" -> **NOT_AVAILABLE**

`NOT_ADJUDICATED` still applies unconditionally to a neighbour trial (the
frozen policy only headline-adjudicates the canonical trial). Since the real
registry has 0/107 PASS verdicts,
`test_real_registry_never_shows_a_gate_pass_without_the_family_wide_pass`
sweeps every one of the 167 registry experiments and asserts no gate ever
renders PASS; the corrected NOT_EVALUATED/NOT_AVAILABLE split has its own
dedicated positive (Regime/Cross-Market with the paired code present) and
negative (every other gate, and a REJECT verdict) cases.

## LLM modes (Research / Strategies)

* **Scripted demo (default).** `ScriptedLLMClient` replays one of five
  hand-built, schema-valid canned responses through the real, unmodified
  agent -- no network call, no cost, fully deterministic. Four scenarios map
  onto the four Phase 11 convenience families (`tsmom`/NQ, `mean_reversion`/CL,
  `breakout`/GC, `ma_trend`/ZN) so a proposed-and-compiled demo hypothesis
  lines up with a real, already-adjudicated registry entry the user can then
  inspect on the Backtests / Validation pages. The fifth is a novel blueprint
  (not a template) to exercise the full closed Strategy DSL.
* **Live Claude API.** Builds `AnthropicClient()` with **no key argument** --
  the Anthropic SDK reads `ANTHROPIC_API_KEY` from the process environment
  itself. Neither `alpha_agent.ui` nor this page ever reads, parses, stores,
  logs, or displays that key; there is deliberately no key-entry widget on any
  page. If the SDK is not installed or no key is present, the page shows a
  plain typed error message and nothing else.

## Read-only boundary

`alpha_agent.ui.services` is the only module the UI uses to reach the rest of
the system, and it never calls an `ExperimentRegistry` write method (no
`insert_experiment`, `record_failure`, `record_lineage`, `apply_bundle`) --
`test_services_module_never_calls_a_registry_write_method` statically greps
the module's own source for those calls as a durable regression guard. Every
number shown is either:

1. read verbatim from the registry (`data/registry/experiments.sqlite`) or a
   committed Phase 13.5C / 15B artifact under `outputs/`, or
2. the direct, typed output of an already-approved service call
   (`ResearchAgent.propose`, `StrategyCompilerAgent.compile_hypothesis`,
   `FailureMemory.lookup`) with a scripted or live LLM -- the LLM only
   proposes; the deterministic Phase 10 compiler and the frozen
   `ReliabilityPolicy` still decide everything they always decided.

`alpha_agent.ui.services.check_no_holdout_leak` re-runs the Phase 14
locked-holdout guard (`assert_no_holdout_market_data`) on every registry/report
payload a page is about to render, purely as a defensive, belt-and-suspenders
check -- it should never fire given the upstream guarantees, and every page
calls `layout.holdout_leak_banner(...)` to fail loud (not silently render a
2025 value) if it ever does.

## Testing

```
PYTHONPATH=python python -m pytest tests/python/test_phase_20_streamlit_ui.py -q
PYTHONPATH=python python -m pytest tests/python/test_release_agent_and_marketdata.py -q
```

40 tests in `test_phase_20_streamlit_ui.py`: the read-only `services` layer
(including the Phase 20.1 trade-ledger fix and the Phase 20.1/20.1b
gate-state correction, each with dedicated positive/negative cases and a
full-registry sweep), all five demo scenarios round-tripping through the real
Phase 16/17 agents with the scripted LLM client, the holdout-guard wiring,
sidebar market selection, the Research -> Strategies hand-off, a (root x
family) sweep over every Backtests/Validation selectbox combination, a check
that Paper Trading renders zero metrics/dataframes, and (via Streamlit's
`AppTest`, skipped automatically if `streamlit`/`plotly` are not installed)
all ten pages rendering with zero exceptions, Agent first and default.

24 more tests in `test_release_agent_and_marketdata.py` (Release UX): the
Agent page's typed transcript cards through the real pipeline (including the
honest "not yet evaluated" path for a novel blueprint and a real
exact-fingerprint REJECT for a known one), the IBKR marketdata package's
offline-safety and read-only boundary (disabled by default, no credential
field, fails closed with no `ibapi` installed, no order/account method
anywhere in the package or its UI boundary, no registry write method), and
the quote strip's three states (delayed-connected, honestly disconnected,
rendered-as-nothing when disabled) -- always via an in-process fake provider,
never a real IBKR connection.

## Module layout

```
python/alpha_agent/ui/
  app.py            entrypoint: st.set_page_config + st.navigation over 10 pages
                    (Agent first and default)
  layout.py         injected CSS (dark shell) + custom header bar + sidebar
  palette.py        shared, validated dark color tokens (status colors reserved)
  charts.py         one shared dark Plotly template + chart builders
  components.py     reusable render pieces: card, badge, metric card, gate grid,
                    empty state, provenance row, activity row, status row
  panels.py         shared right-rail panels: AI agent mini-panel, activity feed,
                    system status card
  services.py       read-only data-access + pure gate/ledger logic (no `streamlit`
                    import; pure & testable)
  market_context.py sole UI boundary into `alpha_agent.marketdata` -- read-only
                    IBKR delayed quote strip + OBSERVATIONAL_CONTEXT_ONLY note
  databento_context.py sole UI boundary into `alpha_agent.marketdata.
                    databento_provider` -- read-only Databento market
                    OBSERVATION plane (health/snapshot/OHLCV/contract)
  market_home.py    MARKET REALITY pass (see below): shared "REAL FUTURES
                    MARKET" strip / Selected Market panel / price+volume
                    chart / Market Context, used by BOTH the Agent (Home)
                    and Market pages, with a UI-session cache-first layer
                    in front of `databento_context`
  llm_demo.py       curated scripted scenarios + the only place an LLM client is built
  views/
    agent.py            default landing page: Real Futures Market strip +
                        Selected Market + Research summary, THEN the
                        conversational research workspace
    dashboard.py        secondary analytics page (formerly "overview.py")
    market.py           dedicated Market Observation page (market_home +
                        Compare Markets)
    research.py
    strategies.py
    backtests.py
    validation.py
    experiment_log.py
    paper_trading.py
    crypto_lab.py
    system.py
```

`alpha_agent/marketdata/` (Release UX, optional `[ibkr]` extra, not required
to run the test suite) is the read-only IBKR TWS API delayed-market-data
package: `quote.py` (`Quote`, `FeedState`), `provider.py`
(`MarketDataProvider` protocol), `contracts.py` (approved-root ->
market-data-only `CONTFUT` contract), `config.py` (`IBKRConnectionConfig` --
host/port/client_id only, no credentials, disabled by default), and
`ibkr_provider.py` (`IBKRDelayedMarketDataProvider`, `reqMarketDataType(3)`
for delayed data, lazy `ibapi` import, no order or account method). See the
package docstring for the full CLAUDE.md boundary rationale.

The page-module directory is named `views/`, not `pages/` -- Streamlit's
legacy multipage auto-discovery activates on any literal `pages/` directory
next to the entrypoint regardless of an explicit `st.navigation(...)` call,
which silently replaced the custom-titled/iconed pages with auto-discovered,
unstyled ones when a page's URL was opened directly. `views/` avoids the
collision entirely.
