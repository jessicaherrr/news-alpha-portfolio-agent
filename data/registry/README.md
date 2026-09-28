# `data/registry/` — the experiment registry database

`experiments.sqlite` is the canonical Phase 14 experiment registry and failure
memory: every research hypothesis this platform has run, its immutable result,
its typed failures, and its supersession lineage.

The `.sqlite` file itself is **not committed**. It is a deterministic derived
artifact — a SQLite binary is not byte-reproducible across machines, so
committing it would add noise without adding provenance. What *is* committed is
everything needed to rebuild it exactly:

- the importer, `scripts/phase_14_import_phase_13_5c.py`;
- the immutable Phase 13.5C source artifacts under `outputs/phase_13_5c/` and
  `data/manifests/phase_13_5c/`;
- the human-readable exports under `outputs/phase_14/`.

Rebuild (offline, no data spend, idempotent — running it twice produces the same
database and the same content digest):

```bash
python scripts/phase_14_import_phase_13_5c.py
```

Inspect without SQL:

```bash
python scripts/phase_14_registry.py summary
python scripts/phase_14_registry.py list --family tsmom
python scripts/phase_14_registry.py show NQ__TSMOM__CANONICAL__VALIDATION_2023_2024
python scripts/phase_14_registry.py find-related --family tsmom --root NQ \
    --param fast_horizon=21 --param slow_horizon=120
python scripts/phase_14_registry.py failures --class CONTRACT_ECONOMICS_FAILURE
python scripts/phase_14_registry.py history \
    ZN__TSMOM__CANONICAL__VALIDATION_2023_2024__PRECORRECTION_52222C3
```

The registry is append-oriented. Records are never deleted or replaced in place;
a correction is a new experiment plus an explicit `SUPERSEDES` / `CORRECTS`
edge. See `docs/EXPERIMENT_REGISTRY.md`.
