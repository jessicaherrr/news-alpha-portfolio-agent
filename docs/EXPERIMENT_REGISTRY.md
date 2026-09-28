# Experiment Registry and Failure Memory (Phase 14)

The persistent, deterministic, **offline** memory of every research attempt this
platform has made. It exists so that a future Research Agent can ask *"has this
already been tried, and what did we learn?"* **before** spending compute, and so
that a failed experiment stays first-class evidence instead of being quietly
forgotten.

Phase 14 is infrastructure and memory. It generates no alpha, tunes no
parameters, changes no frozen statistics, and never touches the 2025 locked
holdout.

- database: `data/registry/experiments.sqlite` (schema version 5, experiment
  identity schema v2; not committed —
  rebuilt deterministically, see [`data/registry/README.md`](../data/registry/README.md))
- package: [`python/alpha_agent/registry/`](../python/alpha_agent/registry/)
- importer: [`scripts/phase_14_import_phase_13_5c.py`](../scripts/phase_14_import_phase_13_5c.py)
- researcher CLI: [`scripts/phase_14_registry.py`](../scripts/phase_14_registry.py)
  (or `python -m alpha_agent.registry.cli`)
- artifacts: `outputs/phase_14/`

## 1. Architectural role

```
Research Agent
    |
    v
HypothesisSpec
    |
    v
Failure Memory / Experiment Registry
    |
    +-- exact duplicate?  -> reference the prior experiment; do not re-run
    |
    +-- near duplicate?   -> warn + return prior evidence (agent policy decides)
    |
    v
Compiler Agent -> StrategySpec -> C++ Quant Core -> Reliability Validation
    |
    v
Registry append (never a silent rerun, never an overwrite)
```

Phase 14 implements only the registry, the failure memory, the deterministic
retrieval interfaces, and the import of the already-completed Phase 13.5C work.
There is no runtime Research Agent yet.

## 2. Entity model

| table | holds |
|---|---|
| `experiments` | one statistical hypothesis under one full configuration |
| `results` | its immutable adjudicated outcome |
| `failures` | typed lessons: scientific rejections and engineering/data defects |
| `lineage` | `source <RELATION> target` edges |
| `sensitivity_evidence` | robustness reruns of an existing hypothesis — **not** trials |
| `cross_market_evidence` | descriptive aggregation over per-root trials — **not** experiments |
| `imports` | provenance of each deterministic import |
| `registry_meta` | `schema_version` |

Relations: `SUPERSEDES`, `CORRECTS`, `REPRODUCES`, `PARAMETER_NEIGHBOUR_OF`,
`SAME_HYPOTHESIS_SENSITIVITY_OF`.

## 3. Identity

Three identifiers are kept strictly apart:

| layer | example | role |
|---|---|---|
| `experiment_identity` | `experiment1:<sha256>` | the scientific identity; the only thing duplicate detection uses |
| `experiment_id` | `NQ__TSMOM__CANONICAL__VALIDATION_2023_2024` | deterministic human-readable label |
| `row_id` | `42` | SQLite storage only |

`experiment_identity` = SHA-256 over canonical JSON of: strategy fingerprint,
family, root, parameter variant, dataset, split, `ValidationSpec`,
`ReliabilityPolicy`, execution config, cost config, risk identity, feature set.
It excludes wall-clock time, UUIDs, code commit, display name and the trial
`role` label.

**Identity is a pre-run quantity** (identity schema v2, Phase 14.1). Every
component is known from the hypothesis and its configuration before a single bar
executes, so an agent can compute a proposal's identity and ask "has this already
been run?" without compiling a schedule or running a backtest. Two fields are
therefore *provenance*, not identity:

| field | meaning | when NULL |
|---|---|---|
| `experiment_identity` | pre-run semantic identity | never |
| `target_schedule_hash` | post-compilation audit evidence | no committed artifact records this variant's compiled schedule |
| `report_fingerprint` | result-artifact provenance | no per-variant `ValidationReport` exists |

Both provenance fields still enter the **content** fingerprint. So the same
identity reported with a different compiled schedule raises
`ScheduleProvenanceConflict` — an integrity condition to investigate, never a
second experiment that slips past duplicate detection. Identity schema v1
(which folded `target_schedule_hash` into identity) is recognised only so a
stale record is rejected loudly with `IdentitySchemaMismatch`.

**Provenance is never fabricated.** Where no committed artifact supports a
value it is `NULL` and the row's `evidence_completeness` says so, with
`source_artifact` / `source_artifact_sha256` naming what the statistics were
actually read from.

**Identity uses the ACTUAL proposed StrategySpec's feature semantics.**
`feature_spec_fingerprint` is a legitimate pre-run identity input, so it must be
the variant's own — never the canonical trial's. A parameter neighbour usually
depends on different `FeatureSpec`s (a TSMOM neighbour with `fast_horizon=10`
uses a different feature than the canonical `fast_horizon=20`), and copying the
canonical set would both misstate provenance and give the neighbour a wrong
identity that its own pre-run computation would never match.

The importer therefore reconstructs each variant's `StrategySpec` from its
frozen parameters with the frozen factories
(`alpha_agent.strategy.candidates_phase_13_5c.spec_for_params`), **proves** the
reconstruction is faithful by requiring its fingerprint to equal the committed
`neighbour_fingerprints` entry, and derives that spec's own feature fingerprints
and `strategy_id`. Reconstruction reads no market data and runs no backtest.

Across the 86 neighbours, 66 have a feature set genuinely different from their
canonical trial and 20 (the mean-reversion `entry_z` / `exit_z` variants) are
legitimately identical because those parameters move DSL thresholds, not
`FeatureSpec`s. That 20 is derived, not assumed.

**The execution-config component carries the derived contract economics of every
root in the multiple-testing family**, not just this experiment's root. BH
q-values and the DSR benchmark are computed jointly over all 107 trials, so a
mis-derived point value on one root changes the global statistics of every
trial. That is what makes the Phase 13.5C pre-correction and corrected runs
genuinely different experiments — and therefore what lets the registry keep both
rather than overwrite one with the other.

## 4. Immutability

- Idempotent: the same identity with byte-equivalent content returns the
  existing row.
- Loud: the same identity with *conflicting* scientific content raises
  `ExperimentConflict` and the transaction rolls back.
- Immutable: a recorded execution attempt's result cannot be changed;
  `ImmutableResultError`.
- No `INSERT OR REPLACE` on any registry table; no `UPDATE` of any row.
- Authority (`AUTHORITATIVE` / `SUPERSEDED`) is **derived** from the lineage
  graph at query time, so marking something superseded never rewrites its row.

Imports are transactional: the bundle is validated in full before the
transaction opens, and a mid-import failure rolls back completely.

## 4a. Execution attempts (schema v5)

`experiment_identity` is the pre-run **scientific hypothesis**; an **execution
attempt** is one run of it (`execution_attempts` / `attempt_results` /
`attempt_lineage`). The same identity may carry several immutable attempts.

- `AttemptStatus.VALID` — a genuine execution. Its result is eligible to be the
  authoritative result.
- `AttemptStatus.INVALID_EXECUTION` — an **engineering / data-pipeline /
  software / infrastructure** defect (`InvalidationClass`), never a scientific
  reason. Kept verbatim as legacy evidence; **never** supplies an authoritative
  result and **never** enters a BH/FDR family.
- **Authoritative attempt** = the highest-ordinal `VALID` attempt. If a
  hypothesis has only invalid attempts it has *no* authoritative result yet
  (`summary().canonical_without_valid_attempt`).
- **Re-execution** after an invalid attempt is a NEW attempt under the SAME
  identity — it does not create a new hypothesis, does not change the identity
  formula, and the executing commit / report / schedule fingerprints are attempt
  provenance that never enter `experiment_identity`.
- `find_exact_duplicate` distinguishes: identity + VALID authoritative result →
  `blocks_reexecution` (cite the prior result); identity + invalid-only history
  → re-execution permitted.
- Schema v3/v4 identity supersession (a genuine semantic-plane change → a *new*
  identity + a `SUPERSEDES`/`CORRECTS` edge) is unchanged. The attempt layer is
  for the different case: **same identity, invalid execution, re-run**.

Migration v4→v5 is preservation-only: every v4 table is renamed to `*_v4_legacy`
and forward-copied. `scripts/registry_migrate_v5.py`;
`outputs/phase_14/REGISTRY_SCHEMA_V5_MIGRATION.json`.

## 5. Retrieval

```python
from alpha_agent.registry import ExperimentRegistry
from alpha_agent.registry.failure_memory import FailureMemory

reg = ExperimentRegistry("data/registry/experiments.sqlite")

reg.find_exact_duplicate(identity)          # has this exact run happened?
reg.find_related(strategy_family="tsmom", root_symbol="NQ",
                 params={"fast_horizon": 21, "slow_horizon": 120})
reg.resolve_authoritative(identity)         # follow corrections to today's answer
reg.experiments(include_superseded=True)    # historical + corrected
FailureMemory(reg).lookup(strategy_family="tsmom", root_symbol="NQ",
                          strategy_spec={"params": {...}})
```

Near-duplicate similarity is deterministic and transparent — no embeddings, no
vector DB, no LLM. It is a fixed weighted sum:

| component | weight |
|---|---|
| `same_family` | 0.35 |
| `same_root` | 0.20 |
| `same_structure` | 0.15 |
| `same_feature_set` | 0.10 |
| `parameter_proximity` | 0.20 |

Parameter distance is normalised by the **declared** `param_grid_ranges` — the
ranges frozen before any result was inspected. Every hit carries the reasons
that produced its score. This is memory and warning infrastructure: it never
changes a `ReliabilityPolicy` and never rejects a hypothesis on its own.

## 6. What Phase 13.5C imported

| quantity | value |
|---|---|
| unique statistical hypotheses (authoritative) | **107** |
| canonical (headline-adjudicated) | 21 |
| predeclared parameter neighbours | 86 |
| canonical verdicts | PASS 0, REJECT 19, INCONCLUSIVE 2 |
| holdout-eligible candidates | 0 |
| superseded pre-correction rows retained | 21 |
| failure records | 106 |
| completed C++ executions (compute, **not** hypotheses) | 528 |
| data-quality sensitivity reruns (**not** BH trials) | 21 |
| cross-market summaries (**not** experiments) | 5 |

These are deliberately distinct numbers. "How many experiments?" has no single
answer, and the registry refuses to collapse them into one.

- authoritative research commit: `d40050b`
- superseded checkpoint, retained in history: `52222c3`
- provenance stamp: `bdc231a`
- 2025 locked holdout: never downloaded, queried, cost-fetched, loaded,
  featured or evaluated.

## 7. Lessons the failure memory remembers

**Roll execution data plumbing** (`ROLL_DATA_FAILURE`). The clean
continuous-front primary `MarketEvent` stream carried no same-timestamp close
for the *outgoing* contract, so the C++ `RejectDefer` roll close-leg had nothing
real to price against. Feeding the overlap bars as ordinary `MarketEvent`s was
**rejected**: equal-timestamp ordering by `instrument_id` could change
active-contract state, `bars_seen` and latency. The accepted correction is
auxiliary roll-close marks keyed by `(instrument_id, ts_event_ns)`, consulted
only by the roll close-leg and never entering the event stream.

**Fractional Treasury contract economics** (`CONTRACT_ECONOMICS_FAILURE`).
`min_price_increment_amount / display_factor` was treated as the USD tick value
for fractionally quoted CBOT Treasuries, giving ZN a point value of `0.064` and
a tick value of `0.001` instead of `1000` and `15.625` — 15,625x too small. The
fix is a generic percent-of-par quote-convention rule (`price_scale = 0.01`),
with no root-specific hardcoding. 20 of the 107 hypotheses were affected and
they entered the joint BH/DSR family, so the pre-correction global statistics of
*every* trial are superseded.

**Pre-correction matrix supersession** (`SUPERSEDED_RESULT`). `52222c3` stays in
git history and in the registry as historical lineage; `d40050b` is
authoritative. Nothing is deleted.

## 8. Limits

- Similarity is structural, not semantic. It does not understand economic
  mechanism and is not presented as if it does.
- The 86 neighbours are recorded with their trial statistics (p-value, BH
  q-value, rejection at q), sourced from `TRIAL_FAMILY.csv`. Per-neighbour
  PnL/Sharpe were summarised into each canonical trial's parameter-stability
  block and not retained per trial; there is no per-neighbour `ValidationReport`
  and no committed per-neighbour schedule hash, so both provenance fields are
  `NULL` and `evidence_completeness` says why.
- `CONTRACT_ECONOMICS_CORRECTION.json` does list 20 ZN schedule hashes
  (4 canonical + 16 neighbours, pre and post), and they look like the missing
  neighbour provenance. They are not: the invariance script emits them from
  2018-01-01 across the whole 2018–2024 span, while a `ValidationReport`
  fingerprints the headline VALIDATION-window schedule. The importer *proves*
  the two differ rather than assuming it, and refuses to back-fill from them.
- The pre-correction lineage is imported at canonical granularity (21 rows) —
  the retained correction artifact records the superseded verdict, net PnL, BH
  q-value and DSR per canonical trial, but not pre-correction reason codes,
  report fingerprints, schedule hashes or the 86 pre-correction neighbour
  statistics. Those rows carry `NULL` provenance rather than the corrected
  run's.
- The registry stores no market data and no 2025 timestamp of any kind. Any
  market-data or performance value at or after 2025-01-01 raises
  `HoldoutAccessError` at the write boundary.
