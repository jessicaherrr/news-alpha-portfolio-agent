<div align="center">

<img src="media/news_alpha_logo.png" alt="News Alpha Portfolio Agent" width="70%"/>


### From real-world news to testable multi-asset research

A quantitative research platform that turns market news and events into economic mechanisms, measurable signals, portfolio hypotheses, authoritative C++ backtests, and reusable scientific evidence.

<p>
  <img src="https://img.shields.io/badge/Python-research%20layer-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/C%2B%2B-execution%20engine-00599C?logo=cplusplus&logoColor=white" alt="C++"/>
  <img src="https://img.shields.io/badge/Streamlit-research%20workspace-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/github/actions/workflow/status/jessicaherrr/news-alpha-portfolio-agent/ci.yml?label=tests" alt="Tests"/>
  <img src="https://img.shields.io/github/license/jessicaherrr/news-alpha-portfolio-agent" alt="License"/>
  <img src="https://img.shields.io/badge/status-research%20only-6C757D" alt="Research only"/>
</p>


</div>

---

## The Research Pipeline

<p align="center">
  <img src="media/news_alpha_overview.png" alt="News Alpha Portfolio Agent overview" width="100%"/>
</p>


## How It Works

The user-facing workflow follows one persistent research thread:

**News → Reasoning → Market → Signals → Portfolio → Backtest → Learn**

Research Setup remains persistent context for the thread rather than a separate workflow step.

### 1. News
#### Turn a real-world event into a research case

Each research run begins with a dated market event and preserves its source, timestamp,
entities, freshness, and provenance.

The first pass asks which market domains may be relevant. It does **not** convert a headline
directly into a BUY or SELL call.


### 2. Reasoning
#### Explain why the event could matter economically

The system turns the event into an explicit economic mechanism and traces possible transmission paths.

Those paths may be:

- **Direct** — the effect reaches the same market or sector relatively quickly.
- **Supply Chain** — the effect propagates through suppliers, customers, or production inputs.
- **Cross Sector** — the effect spills into another industry or asset class.

For example, higher AI infrastructure spending may propagate through compute demand,
accelerators, HBM, foundry utilization, data-center construction, electricity demand,
and grid investment.

The mechanism graph is a **research hypothesis**, not proof of causality.

### 3. Market
#### Translate economic consequences into market expressions

An economic consequence can appear in more than one market.

The system asks:

- Which instruments can express the hypothesis?
- Are they allowed by the Research Setup?
- Do we have real historical data?
- Is the measurement point-in-time safe?
- Is the instrument a direct expression or only a proxy?

A market expression can remain visible even when the data needed to test it is missing.
Missing data is recorded as a research limitation rather than replaced with synthetic truth.


### 4. Signals
#### Turn measurable market effects into testable quantitative factors

A hypothesis only becomes a CandidateSignal when it can be written as an explicit,
executable quantitative expression.

For example:

```text
ts_return(close, 20)
```

The expression shown in the research workspace is the same expression used to construct
the factor series.

Signal diagnostics ask whether the relationship is statistically meaningful, stable through time,
sufficiently covered by real data, robust to trading costs, and distinct from redundant exposures.

Signals from different markets are compared on evidence quality, data integrity, cost, liquidity,
and redundancy before portfolio construction.

Screening helps decide what deserves deeper research. It is **not** a scientific validation verdict.



### 5. Portfolio
#### Convert independent signals into a feasible risk-constrained portfolio

Portfolio construction starts from eligible, non-redundant exposures rather than raw candidate count.

The allocator considers risk budgets, volatility, covariance, leverage, liquidity, concentration,
shorting rules, transaction costs, and instrument-specific tradable units.

Research rank does not become expected return, and rank does not directly become portfolio weight.

Portfolio sizing and accounting are authoritative in C++.


### 6. Backtest
#### Execute deterministically and test whether the evidence survives

The C++ engine owns mechanics that should not depend on an LLM:

- fills and position changes,
- futures multipliers and share units,
- transaction costs,
- portfolio accounting,
- hard risk constraints,
- and deterministic replay.

Python coordinates the research process and prepares the inputs. It does not maintain
a second execution engine.

A successful backtest is still not automatically a validated strategy.

| Validation check | What it asks |
|---|---|
| **Out-of-Sample** | Does the result survive outside the discovery window? |
| **Cost Sensitivity** | Does the result survive realistic trading friction? |
| **Bootstrap / Null Tests** | Is the observed result unusual relative to an appropriate null? |
| **Stability** | Does the result persist across periods or regimes? |
| **Multiple Testing** | Does it survive repeated research trials? |
| **Holdout** | Does it survive the final locked evaluation boundary? |

Evidence remains attached to the level that was actually tested. A portfolio-level result
does not automatically validate every signal, mechanism, or Signal Path that contributed to it.


### 7. Learn
#### Turn both success and failure into reusable research evidence

The research loop preserves what happened after the experiment.

A research record can remain:

- **Supported**
- **Failed**
- **Unresolved**
- **Waiting for better data**
- **Rejected after costs**
- **Infeasible under portfolio constraints**

The important point is that failure is not discarded.



## Point-in-Time Integrity

Point-in-time integrity applies across the entire pipeline.

Historical research may only use information that would actually have been available at the time.

| Data type | Research rule |
|---|---|
| **Market bars** | Used only after the bar is complete |
| **Fundamentals** | Keyed to filing or acceptance timing |
| **Macro releases** | Keyed to the actual release timestamp |
| **Restatements** | Never substituted backward into history |
| **Futures contracts** | Historical contract and roll semantics are preserved |
| **Missing data** | Remains missing unless a valid historical source exists |
| **Synthetic data** | Never promoted to real scientific evidence |

See [Data and Point-in-Time Integrity](docs/DATA_AND_PIT.md).



## Architecture

<p align="center">
  <img src="media/system_architecture.png" alt="News Alpha Portfolio Agent system architecture" width="96%"/>
</p>

For implementation boundaries and ownership, see [Architecture](docs/ARCHITECTURE.md).



## Repository Map

| Path | Purpose |
|---|---|
| `python/alpha_agent/news_alpha` | News, mechanisms, signal paths, market expressions |
| `python/alpha_agent/features` | Executable factor definitions |
| `python/alpha_agent/screening` | Factor diagnostics and candidate screening |
| `python/alpha_agent/recommendation` | Multi-asset research prioritization |
| `python/alpha_agent/portfolio` | Portfolio construction orchestration |
| `python/alpha_agent/validation` | Scientific validation |
| `python/alpha_agent/registry` | Experiment and evidence Registry |
| `python/alpha_agent/alpha_memory` | Research memory |
| `python/alpha_agent/ui` | Unified research workspace |
| `cpp/` | Deterministic execution, risk, accounting, and backtesting |
| `tests/` | PIT, determinism, scientific-integrity, and regression tests |



## Quick Start

### Clone

```bash
git clone https://github.com/jessicaherrr/news-alpha-portfolio-agent.git
cd news-alpha-portfolio-agent
```

### Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure optional providers

```bash
cp .env.example .env
```

Only add credentials for providers you intend to use. Never commit `.env`.

### Build the C++ engine

```bash
cmake -S . -B build
cmake --build build
```

### Run tests

```bash
pytest tests/python
ctest --test-dir build
```

### Launch the workspace

Use the Streamlit entry point documented in the current release and [News Alpha Pipeline](docs/NEWS_ALPHA_PIPELINE.md).



## Example Research Case

A complete research case moves through the same canonical backend used by the UI.

| Step | Output |
|---|---|
| **Event** | Dated news with source and provenance |
| **Impact** | Affected market domains and initial relevance |
| **Mechanism** | Economic transmission hypothesis |
| **Expression** | Candidate instruments and measurements |
| **Signal** | Exact executable factor expression |
| **Diagnostics** | Screening evidence and redundancy |
| **Ranking** | Independent research exposures |
| **Portfolio** | Deterministic constrained target plan |
| **Backtest** | Authoritative C++ execution result |
| **Validation** | Scientific evidence or explicit rejection |
| **Memory** | Success, failure, or unresolved research record |

The pipeline is allowed to stop honestly at any stage.



## Current Scientific Status

This repository does not claim a validated profitable strategy.

The current framework can distinguish between supported, weak, contradictory, unavailable, and untestable research states. In the current accepted real-data News Alpha run, no signal had enough screening support to produce a qualified portfolio.

That result is recorded rather than hidden.



## Current Limitations

| Area | Current limitation |
|---|---|
| **Live trading** | No live-money broker routing |
| **Validated alpha** | No claim of an approved profitable strategy |
| **Historical events** | Event-conditioned diffusion studies need a larger PIT-safe event history |
| **Data coverage** | Some economic measurements remain unavailable or proxy-only |
| **Asset coverage** | Some domains have less complete history than the futures research core |
| **Mixed-market validation** | Futures + ETF scientific validation still needs a canonical cross-market valuation clock |
| **External providers** | Some sources require user-provided credentials |

The system is designed to fail explicitly rather than substitute unsupported assumptions.



## Data and Reproducibility

Large market datasets, runtime Registries, cached outputs, and private research state are intentionally excluded from the public repository.

The public repository contains source code, tests, safe configuration examples, scientific documentation, and small reproducible fixtures.



## Tests

| Test area | What it protects |
|---|---|
| **PIT integrity** | No look-ahead through data availability |
| **Factor identity** | Same executable hypothesis produces deterministic identity |
| **Signal lineage** | CandidateSignals retain their research origin |
| **Screening** | Diagnostics stay separate from scientific validation |
| **Ranking** | Research merit stays separate from user fit |
| **Portfolio** | Constraints, unit conversion, multipliers, and covariance |
| **Execution** | Costs, fills, accounting, deterministic replay |
| **Validation** | OOS, holdout, multiple testing, verdict authority |
| **Memory** | Evidence is stored at the scope actually tested |
| **UI** | Displayed results stay consistent with backend authority |

Public CI is designed not to depend on private market data or paid API calls.



## Documentation

| Document | Contents |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | System layers, ownership, Python/C++ boundaries |
| [News Alpha Pipeline](docs/NEWS_ALPHA_PIPELINE.md) | End-to-end research workflow |
| [Scientific Validation](docs/SCIENTIFIC_VALIDATION.md) | Screening, OOS, multiple testing, holdout |
| [Data and PIT](docs/DATA_AND_PIT.md) | Historical availability, provenance, no look-ahead |



## Disclaimer

News Alpha Portfolio Agent is intended for quantitative research and educational use.

Nothing in this repository is financial, investment, or trading advice. Historical research results and backtests do not guarantee future performance.



## License

See [LICENSE](LICENSE).


<p align="center">
  <img src="media/news_alpha_closing_banner.png"
       alt="News Alpha Portfolio Agent research philosophy"
       width="100%"/>
</p>
