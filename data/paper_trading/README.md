# `data/paper_trading/` — the persistent paper-trading ledger

`paper_ledger.sqlite` (Phase 21) is the persistent, append-oriented record of
every paper-trading run: its immutable configuration (which registry
experiment, which `PaperRiskPolicy`), every step's committed C++ snapshot,
every fill, and every alert. See `docs/PAPER_TRADING_AND_DRIFT_MONITOR.md` and
`python/alpha_agent/paper/ledger.py`.

The `.sqlite` file itself is **not committed** (same reasoning as
`data/registry/README.md`: a SQLite binary is not byte-reproducible across
machines). It is created automatically, empty, the first time anything opens
it (`alpha_agent.paper.ledger.PaperLedger(path)` — no separate "init" step),
and it is a SEPARATE store from `data/registry/experiments.sqlite`: paper
trading is operational monitoring of an already-approved strategy, never a
new statistical hypothesis, and this ledger must never be confused with the
registry's scientific identity / BH-FDR / failure-memory semantics.

Inspect without SQL:

```bash
python scripts/phase_21_paper_trading.py list-eligible
python scripts/phase_21_paper_trading.py list-runs
python scripts/phase_21_paper_trading.py status --run <run_id>
```

There is no live broker or live market-data feed behind this ledger. Every
run it records is a deterministic replay of already-acquired historical CME
data (`[2018-01-01, 2025-01-01)`, never the locked 2025 holdout) through the
hard-risk-gated C++ engine.
