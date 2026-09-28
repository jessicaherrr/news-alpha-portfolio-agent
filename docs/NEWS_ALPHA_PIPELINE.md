# News Alpha Pipeline

The canonical research workflow that turns a news or economic event into a
validated, executable portfolio hypothesis. Each stage below is a real
module in `python/alpha_agent/news_alpha/`, `python/alpha_agent/screening/`,
`python/alpha_agent/recommendation/`, and `python/alpha_agent/portfolio/`.
Detailed design notes for each stage live in `docs/NEWS_ALPHA_PHASE_*.md`.

```
Research Mandate
  → News / Event
  → Initial Impact Scan
  → Economic Mechanism Graph
  → Signal Paths
  → Asset Expressions
  → PIT Measurements
  → Candidate Signals
  → Factor Diagnostics
  → Multi-Asset Ranking
  → Portfolio Construction
  → C++ Backtest / Execution
  → Scientific Validation
  → Registry / Alpha Memory / Failure Memory
```

## 1. Research Mandate + Initial Impact Scan

A `ResearchMandate` declares the allowed asset universe (Futures, ETF, Equity)
and scope before any event is examined. An incoming news/economic event is
triaged for *initial* economic impact plausibility — this is a coarse filter,
not a trading decision. See `docs/NEWS_ALPHA_PHASE_A_MANDATE_AND_IMPACT_TRIAGE.md`.

## 2. Economic Mechanism Graph

The event is decomposed into a typed economic transmission model — the causal
"why" a price should move, not just "what" moved. Every edge in the graph
carries an explicit provenance/support status; the model separates "what is
known" from "what is proposed but unresolved." See
`docs/NEWS_ALPHA_PHASE_B_ECONOMIC_MECHANISM_GRAPH.md`.

## 3. Signal Path Discovery

From the mechanism graph, the system enumerates candidate signal paths —
direct, supply-chain, and cross-sector transmission chains from the event to
a measurable market consequence. Path depth is treated as a research
variable, not a quality score: a longer chain is not automatically weaker
evidence, but it does carry more assumptions. See
`docs/NEWS_ALPHA_PHASE_C_SIGNAL_PATH_DISCOVERY.md`.

## 4. Asset Expression + Measurement Resolution

Each signal path is expressed against a specific, tradeable instrument with an
explicit measurement specification, then resolved to a point-in-time (PIT)
safe data field. This is where "the event should move oil demand" becomes
"measure the CL front-month return over this specific, publication-lag-aware
window." See `docs/NEWS_ALPHA_PHASE_D_ASSET_EXPRESSION_AND_MEASUREMENT.md` and
`docs/DATA_AND_PIT.md`.

## 5. Candidate Signals + Factor Diagnostics

Measurement specs become closed-form, predeclared candidate signal rules —
never fitted to the outcome they are meant to predict. Each candidate is run
through real factor diagnostics (information coefficient, decay, basic
stability) before it is allowed to compete for portfolio inclusion. See
`docs/NEWS_ALPHA_PHASE_E_CANDIDATE_SIGNALS_AND_FACTOR_DIAGNOSTICS.md`.

## 6. Multi-Asset Signal Ranking

Signals from potentially many events/paths are ranked together, not scored in
isolation. Ranking considers research merit (diagnostic strength), user/
mandate fit, and redundancy — correlated signals are treated as overlapping
exposures, not independent opportunities. Output is a typed `RankedSignalSet`.
See `docs/NEWS_ALPHA_PHASE_F_MULTI_ASSET_SIGNAL_RANKING.md`.

## 7. Portfolio Construction

Deterministic, C++-backed portfolio construction turns the ranked signal set
into position sizes under explicit constraints (risk budget, per-asset caps,
correlation limits). This logic lives in the C++ quant core — never the LLM.
See `docs/NEWS_ALPHA_PHASE_G_PORTFOLIO_CONSTRUCTION.md`.

## 8. C++ Backtest / Execution

The constructed portfolio plan is executed through the same deterministic
C++ engine used for single-asset futures research: fills resolve to actual
raw contracts traded on real dates, with explicit commissions, tick size, and
roll handling. No look-ahead by construction. See `docs/EXECUTION_MODEL.md`.

## 9. Scientific Validation

Before any result is trusted, it passes through the shared validation gate:
walk-forward OOS evaluation, multiple-testing correction, cost sensitivity,
and the locked holdout boundary. "No eligible portfolio" and "insufficient
evidence" are valid, first-class outcomes — not failures to hide. See
`docs/SCIENTIFIC_VALIDATION.md` and
`docs/NEWS_ALPHA_PHASE_H_VALIDATION_AND_MEMORY.md`.

## 10. Registry / Alpha Memory / Failure Memory

Every hypothesis — validated, rejected, or invalidated by an engineering
defect — is permanently recorded. Nothing is deleted. Re-running the same
scientific hypothesis is either blocked (if a valid authoritative result
already exists) or recorded as a new execution attempt under the same
identity (if the only prior attempts were invalid). See
`docs/EXPERIMENT_REGISTRY.md`.

## Interactive exploration

The Streamlit research workspace (`python/alpha_agent/ui/`) exposes every
stage above as an interactive step: News, Mechanism Graph, Signal Paths,
Candidate Signals, Portfolio, Validation, and Experiment Log. See
`docs/STREAMLIT_RESEARCH_INTERFACE.md` and
`docs/NEWS_ALPHA_RESEARCH_THREAD_WORKSPACE.md`.

## Runnable example

```bash
PYTHONPATH=python python scripts/news_alpha_pipeline_demo.py
```

Runs the pipeline end to end, offline and deterministic, against cached
example events (`--scripted-proposal` adds clearly-labelled fixture LLM
output with no live model call). See also
`scripts/news_alpha_phase_e_vertical_slice.py` through
`scripts/news_alpha_phase_h_validation.py` for the stage-by-stage runnable
chain, and `examples/sample_event.json` for the input shape.
