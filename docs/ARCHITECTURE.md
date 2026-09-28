# Architecture

Updated for the frozen state at the end of Phase 23.2. See
[`docs/FINAL_SYSTEM_AUDIT.md`](FINAL_SYSTEM_AUDIT.md) for the fully
evidence-cited version of this diagram (every arrow traced to a line of code)
and the top-level [`README.md`](../README.md) for the portfolio-level summary.

## Separation of concerns

```text
                          USER / STREAMLIT UI (read-only)
                                |
                                v
+------------------- PYTHON AI RESEARCH LAYER --------------------+
| Research Agent -> HypothesisSpec -> Compiler Agent -> StrategySpec |
| Data/Features | ML meta-labeling | Validation | Registry | UI    |
| The LLM never sees or sets PnL, fills, risk, cost, or thresholds. |
+-------------------------------+----------------------------------+
                                |
                     typed boundary (CSV/JSON, reference;
                     optional pybind11 fast path, Phase 19)
                                |
+--------------------- C++20 QUANT CORE ----------------------------+
| Contract metadata | Event-driven backtest engine                 |
| Execution / fill simulator (raw contract only, never adjusted)   |
| Portfolio accounting | Hard risk manager (LLM cannot override)   |
+-------------------------------+----------------------------------+
                                |
                                v
                       deterministic results
                                |
                                v
       Reliability Validation -> PASS / REJECT / INCONCLUSIVE
        (BH/FDR, deflated Sharpe, null tests, cost sensitivity)
                                |
                                v
              Experiment Registry (append-only, schema v5)
           every attempt kept, including every failure
                                |
                    +---------- feedback -----------+
                    |                                |
                    v                                v
             Research Agent                 Paper Trading (replay,
          (registry-derived context,          risk-gated, real kill
           holdout-guarded)                    switches; 0 eligible
                                                strategies today)
```

## Why hybrid

Python is the research/control plane: vendor APIs, LLM/ML libraries,
statistics, dashboards. C++ is the execution/accounting plane: deterministic,
speed-sensitive market mechanics that must never be second-guessed by a
model. The split is a hard architectural boundary, not a convenience — see
`CLAUDE.md`'s "Architecture boundaries" section, and
`docs/FINAL_SYSTEM_AUDIT.md`'s confirmation that no Python module duplicates
PnL/fill/slippage/commission logic anywhere in the codebase.

## The Python/C++ boundary, then and now

Phase 00-06 deliberately started with a plain CLI boundary (Python writes
canonical bars/contracts to CSV, calls a compiled C++ executable, reads back
JSON) rather than a binding, so C++ build/debug issues would never entangle
early agent-research work. That CLI (`quant_backtest_csv` /
`quant_backtest_targets_csv`) remains the **frozen reference implementation**
today — every later phase's Python bridge (`adapters/targets_bridge.py`,
`adapters/cpp_cli.py`) still calls it, and it is what every backtest, ML
corpus build, and paper-trading step actually runs.

Phase 19 added an **optional** pybind11 fast boundary
(`bindings/pybind_module.cpp` -> `quant_core_py`, built only with
`-DBUILD_PYBIND=ON`) that crosses bars as typed NumPy arrays instead of a CSV
round-trip (~2.8x faster on the benchmark in `scripts/phase_19_pybind_benchmark.py`).
It is a **thin, independently-wired mirror** of the same engine logic, not a
shared code path with the CLI — parity between the two is enforced by
dedicated CLI-vs-pybind tests (`tests/python/test_phase_19_pybind.py`), not by
construction. Official PnL/fill/risk/accounting authority stays with the C++
engine either way; the pybind path is not wired into the production research
run (`--run-real`) and serves high-volume offline research only.

## The LLM's actual surface area

Two LLM-backed components exist, and both have a closed, typed output schema
enforced before anything downstream ever runs:

- **Research Agent** (`alpha_agent.agents.research_agent`) — sees a frozen
  `ResearchContext` (universe, feature catalog, registry digest, failure
  memory, validation feedback; holdout-guarded at construction) and returns a
  `HypothesisSpec`. It cannot see raw market data, PnL, or a fingerprint.
- **Strategy Compiler Agent** (`alpha_agent.agents.compiler_agent`) — turns a
  `HypothesisSpec` into either a `template` (a Phase 11 baseline family with
  typed-bounded parameters) or a full `blueprint` (a closed mirror of the
  entire Phase 10 strategy DSL), or declares the hypothesis inexpressible. It
  compiles through the unchanged, from-Phase-10 `StrategyCompiler` — the real
  safety gate — and its output carries a fixed `ExecutionSemantics` block
  (next-bar entry, no same-bar fills, flat commission) that is stated, never
  LLM-chosen.

The **Research Orchestrator** (`alpha_agent.agents.orchestrator`) sequences
both agents, builds the pre-run scientific `experiment_identity`, and owns
the family-wide BH/FDR gate and final verdict via a deterministic
`PolicyFamilyAdjudicator` — never the LLM. This is the component the LLM
literally cannot reach: it has no path to influence a PASS/REJECT decision or
a p-value.

## Data lifecycle

`vendor request -> immutable raw store -> derived catalog -> features ->
research`, with the >= 2025-01-01 holdout enforced at the lowest shared
acquisition boundary (`HistoricalRequest.__post_init__`, Phase 23.2) so no
caller can construct a request that reaches it. See
`docs/DATA_PIPELINE.md` and `docs/REAL_DATASET_ACQUISITION.md` for the full
acquisition/canonicalization chain, and `CLAUDE.md`'s "Market-data rules" for
the invariants that chain must satisfy.
