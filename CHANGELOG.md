# Changelog

## v0.1.0 — Initial Public Release (2026-09-27)

### Features
- **News-to-mechanism research workflow** — economic transmission modeling from news events and economic data
- **Signal path reasoning** — causal chain definition from mechanism to market measurements
- **Point-in-time safe measurements** — publication-lag-aware historical resolution; no look-ahead bias
- **Executable candidate signals** — closed-form factor definitions with explicit measurement availability
- **Factor diagnostics** — information coefficient, regime stability, correlation analysis, cross-market deduplication
- **Cross-asset signal ranking** — multi-asset validation, signal competition, portfolio contribution assessment
- **Deterministic C++ portfolio construction** — repeatable, auditable allocation logic
- **Authoritative execution and accounting** — fills resolve to actual raw contracts; commissions, ticks, rolls are explicit
- **Walk-forward validation** — training/validation/holdout splits with locked 2025+ holdout
- **Multiple-testing correction** — Benjamini–Hochberg / FDR family-wide gates
- **Scientific validation gates** — regime stability, cross-market evidence, meta daily Sharpe, trade frequency
- **Alpha Memory and Failure Memory** — permanent, append-only registry of hypotheses, results, failures, and invalid execution attempts
- **Replay-based paper trading** — deterministic historical playback with real risk gates (kill switches)
- **Streamlit research workspace** — interactive exploration of news, mechanisms, signals, diagnostics, validation, experiment log, paper trading state
- **LLM research agents** — Claude-powered hypothesis generation, strategy compilation, orchestration
- **Experiment registry** — queryable scientific identity + execution attempt history with typed verdicts
- **Real market data** — 2018–2024 CME Globex futures (ES, NQ, CL, GC, ZN) via Databento

### Architecture
- **Python research plane** — data ingestion, feature engineering, hypothesis generation, validation
- **C++ execution plane** — deterministic backtesting, execution simulation, risk management, position accounting
- **Validation plane** — scientific gates, holdout enforcement, evidence verification
- **Registry and memory** — permanent append-only scientific evidence store

### Documentation
- `docs/ARCHITECTURE.md` — system design and ownership boundaries
- `docs/NEWS_ALPHA_PIPELINE.md` — research workflow and methodology
- `docs/SCIENTIFIC_VALIDATION.md` — validation gates and statistical rigor
- `docs/DATA_AND_PIT.md` — point-in-time measurement and data integrity
- `docs/FINAL_SYSTEM_AUDIT.md` — complete acceptance audit

### Current Results
- **170 experiment identities** in the registry
- **0 strategies with authoritative PASS** verdict
- **0 paper-trading-eligible strategies** as of 2026-09-27
- **Real BH/FDR multiple-testing correction** applied; conservative gates; honest verdicts

### Known Limitations
- No live-money trading; research and replay-based paper trading only
- No approved strategies currently; insufficient statistical evidence
- Futures-only in primary registry; ETF and crypto extensions are experimental
- 2025+ data locked; no future-dated research in validation corpus

### Testing
- Python tests for PIT safety, validation integrity, registry mechanics
- C++ tests for determinism, accounting correctness, risk gate mechanics
- End-to-end integration tests
- Replay-based paper trading verified with real kill-switch trip

See `docs/FINAL_SYSTEM_AUDIT.md` for the complete acceptance audit and open items.
