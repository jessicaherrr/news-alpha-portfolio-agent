# Real CME Futures Dataset -- Acquisition Record (Phase 13.5B)

**Approved:** Plan B research/validation, 2018-2024 only, combined spend cap
**USD 65.00**. The 2025 `LOCKED_HOLDOUT` is prohibited in this phase and is
refused structurally (`acquisition.guard_no_holdout` -- no override flag).

Driver: `scripts/databento_acquire.py` (`--pilot` / `--plan-b` / `--qa` /
`--replay`). Planners: `python/alpha_agent/data/acquisition.py`. Frozen plan:
`configs/real_dataset.yaml` (Plan B). Design rationale:
`docs/REAL_DATASET_PLAN.md`.

---

## 1. Guardrails enforced by the driver

| rule | mechanism |
|---|---|
| nothing may touch 2025 | `guard_no_holdout(req)` raises `HoldoutViolation` if `req.end > "2025-01-01"` or `req.start >= "2025-01-01"`. Called on **every** request. No CLI flag can bypass it. |
| combined spend <= $65 | `AcquisitionBudget.charge()` refuses a download whose `metadata.get_cost` estimate would push cumulative spend past the cap. Cumulative is seeded from the permanent ledger. |
| cost-first | every request is `metadata.get_cost`-estimated and printed before any byte is streamed (CLAUDE market-data rule 2). |
| immutable raw | `raw_store.store_raw` is write-once; a re-run **reuses** the stored `.dbn.zst` (verified by SHA-256) and never re-charges. The driver is fully resumable. |
| roll symbols never guessed | `roll_overlap_requests` builds one `raw_symbol` request per **observed** `<ROOT>.v.0` `instrument_id` transition, for **both** the outgoing and incoming contract (the frozen `SAME_TIMESTAMP_CLOSE_CLOSE` basis needs a contemporaneous bar of both). |
| definitions cheap | `definition_snapshot_requests` = one short (4-day) `<ROOT>.FUT` parent snapshot per quarter boundary (definition cost scales with the requested span, so ~20 snapshots/root cost a few cents total). Quarterly + a multi-day window because a semi-annual 1-day snapshot that landed on a market holiday (Jan 1) returned an empty parent set and left `ZNM2`'s 2022 front-month tenure undefined. Range definitions remain the documented fallback only if snapshot coverage still proves insufficient. |
| transient-error resilient | every `metadata.get_cost` and every streaming download is wrapped in `_retry` (exp. backoff, 6 attempts) against Databento read-timeout / 5xx blips; the download streams to a tempdir and commits to the write-once store only on success, so a retry never double-writes or double-charges. |

Spend ledger: `data/manifests/real_dataset/spend_ledger.json` -- keyed by request
hash, records every download's real `metadata.get_cost` and cumulative total.
Per-component `RealDatasetManifest` under `data/manifests/real_dataset/B/`.

---

## 2. NQ Q1-2024 pilot -- PASSED

`NQ.v.0 ohlcv-1m`, `2024-01-01 .. 2024-04-01` (inside Plan B's validation
window; engineering validation only, not for parameter selection).

| | |
|---|---|
| continuous rows | 86,547 (`NQH4` 69,987 + `NQM4` 16,560) |
| canonical diagnostics | **0 errors, 0 rejected rows** |
| roll transitions | 1: `NQH4 -> NQM4` at 2024-03-13 00:00 UTC |
| roll basis | `used_fallback = False`, `additive_gap = 246.25` (real contemporaneous close-close from the both-contracts overlap `NQH4,NQM4` 2024-03-09..2024-03-21) |
| trading days | **64 `trading_day`(s) across 77 UTC date(s)** -- `trading_day != UTC date`, as required |
| definition snapshots | 1 quarterly `NQ.FUT` parent snapshot (2024-01-01, 4-day) -- the pilot's Q1-2024 span is also fully inside the NQ validation window below |
| economics | `NQH4` / `NQM4`: tick 0.25, point value $20, tick value $5 (no sentinel multiplier) |

**QA + lineage gate: 13 / 13 PASS**

```
[PASS] no canonical diagnostic errors
[PASS] timestamps strictly increasing per contract
[PASS] no duplicate (instrument_id, ts)
[PASS] every bar resolves to a contract in the registry
[PASS] OHLC invariants (low <= min(o,c) <= max(o,c) <= high)
[PASS] session labels present
[PASS] trading_day labels present
[PASS] roll transitions detected match instrument changes
[PASS] no roll used a fallback (contemporaneous overlap present)
[PASS] plausible normalized price scale (median|close| = 17821.25)
[PASS] layer 3/4/6 lineage sidecars written
[PASS] every bar's contract was tradable at that timestamp
[PASS] C++ replays every canonical bar (empty schedule, 0 fills, 2 contracts resolved)
```

`--replay` verifies all raw SHA-256s and every `RealDatasetManifest.semantic_identity()`.

Pilot NEW spend ≈ **$0.40** (continuous $0.31596 + 3 def snapshots $0.00006 +
overlap `NQH4,NQM4` $0.06286). A first-run config bug also spent **$0.02211** on a
one-contract `NQH4`-only overlap that the corrected code does not use; the
artifact is immutable, left in place, and counted. The ledger's cumulative
**$0.4785** additionally counts the pre-existing ~$0.10 mid-2026 NQ Phase
03.5/04.5 test slice (conservative -- not part of Plan B).

`data/manifests/real_dataset/pilot_passed.json` gates the Plan B run.

---

## 3. Plan B full acquisition -- COMPLETE

Acquired 2026-09-08 on branch `phase-13-5-real-dataset`. **10 root × window runs,
13 / 13 QA + lineage checks each, 0 failures, 0 roll fallbacks, 0 canonical
diagnostic errors.** Total Databento spend to date **$57.29 / $65.00 cap**
(headroom $7.71). The 2025 `LOCKED_HOLDOUT` was never requested.

Windows (end-exclusive, `configs/real_dataset.yaml` plan B):

| role | window | years |
|---|---|---|
| research | 2018-01-01 .. 2023-01-01 | 5 |
| validation | 2023-01-01 .. 2025-01-01 | 2 |

Per root × window: 1 `<ROOT>.v.0` continuous + quarterly `<ROOT>.FUT` definition
snapshots (4-day windows) + one both-contracts `raw_symbol` overlap per observed
`.v.0` roll transition.

### 3.1 Downloads + spend

| root | window | continuous rows | roll transitions | def snapshots | overlap reqs | component spend |
|---|---|---:|---:|---:|---:|---:|
| ES | research | 1,756,356 | 20 | 20 | 20 | $7.79 |
| NQ | research | 1,756,201 | 20 | 20 | 20 | $7.72 |
| CL | research | 1,757,020 | 60 | 20 | 60 | $10.11 |
| GC | research | 1,743,315 | 25 | 20 | 25 | $7.80 |
| ZN | research | 1,649,734 | 20 | 20 | 20 | $7.17 |
| ES | validation | 707,953 | 8 | 8 | 8 | $3.09 |
| NQ | validation | 708,312 | 8 | 8 | 8 | $3.09 |
| CL | validation | 698,025 | 24 | 8 | 24 | $4.03 |
| GC | validation | 697,994 | 10 | 8 | 10 | $3.12 |
| ZN | validation | 669,313 | 8 | 8 | 8 | $2.94 |
| | | | | | **subtotal** | **$56.84** |

353 component manifests (10 continuous + 140 definition snapshots + 203 roll
overlaps), 433 immutable raw DBN artifacts. One `raw_symbol` overlap request per
observed `.v.0` transition (both contracts). CL rolls monthly (front-month WTI),
hence 60 / 24 transitions vs 20 / 8 for the quarterly roots.

The ledger cumulative **$57.29** exceeds the $56.84 component subtotal by ~$0.45:
the NQ Q1-2024 pilot (~$0.40), the pilot's superseded one-contract overlap
($0.022), superseded semi-annual 1-day definition snapshots from an earlier run
attempt (~$0.001 total -- immutable, left in place, counted), and the
pre-existing ~$0.10 mid-2026 NQ Phase 03.5/04.5 engineering slice.

Two transient failures during acquisition were absorbed by `_retry`
(exp. backoff + 240 s `SIGALRM` hard deadline per attempt): Databento streaming
read-timeouts, and one multi-hour streaming stall on 2026-09-07 (the API status
page stayed green; a re-probe the same evening completed in 6 s). No request
exceeded its budget gate; the run is fully resumable and idempotent.

Databento flagged 5 source days as reduced-quality (`degraded`): 2020-02-27,
2020-07-01, 2021-12-05, 2022-01-02, 2024-09-18. Bars for those days are stored
**as delivered** -- no silent forward-fill (backtest-integrity rule 4); the
`degraded` marker is carried in the raw DBN and surfaced by Databento's
`metadata.get_dataset_condition`.

### 3.2 QA + lineage gate per root

All 10 runs: **13 / 13 PASS**. The gate (`qa_gate` in `scripts/databento_acquire.py`):

```
no canonical diagnostic errors            every bar's contract tradable at its ts
timestamps strictly increasing/contract   plausible normalized price scale
no duplicate (instrument_id, ts)          layer 3/4/6 lineage sidecars written
every bar resolves to a registry contract roll transitions match instrument changes
OHLC invariants                           no roll used a fallback (contemporaneous overlap)
session + trading_day labels present      C++ replays every canonical bar (0 fills)
```

`trading_day != UTC date` confirmed every run -- research ≈ 1,292 `trading_day`s
across ≈ 1,556 UTC dates; validation ≈ 517 across ≈ 623. Normalized median
`|close|`: ES 3282 / 4792, NQ 10530 / 16894, CL 63.3 / 76.7, GC 1710 / 2039,
ZN 128.2 / 111.1 (research / validation) -- ZN via the Phase 04.5
convention-independent fractional (32nds) economics path.

### 3.3 Local footprint

`data/raw` 433 immutable DBN artifacts (`--replay` verifies every SHA-256 and
every `RealDatasetManifest.semantic_identity()`). 353 component manifests under
`data/manifests/real_dataset/B/`. Derived layer 2/3/4/6 parquet + `.lineage.json`
under `data/processed/{bars,continuous,backadjusted,rolls,contracts}/`. `data/raw`,
`data/processed`, `data/staging` are gitignored; only `data/manifests/` is committed.

---

## 4. Directory / artifact map

```
data/raw/databento/GLBX.MDP3/{ohlcv-1m,definition}/<request-hash>/*.dbn.zst  + .manifest.json   (immutable, gitignored)
data/processed/
  contracts/<ROOT>.parquet                 + .lineage.json
  bars/<ROOT>/<RAW_SYMBOL>.parquet          + .lineage.json    layer 2  RAW_CONTRACT (execution)
  continuous/<ROOT>.v.0.parquet             + .lineage.json    layer 3  RAW_CONTINUOUS (signal)
  backadjusted/<ROOT>.v.0.parquet           + .lineage.json    layer 4  BACK_ADJUSTED (research view)
  rolls/<ROOT>.v.0.parquet                  + .lineage.json    layer 6
data/manifests/real_dataset/
  spend_ledger.json                                            permanent, keyed by request hash
  pilot_passed.json                                            Plan B gate
  B/<ROOT>__<component>__<start>_<end>.json                    RealDatasetManifest per component
```

`data/raw` / `data/processed` / `data/staging` are gitignored; only the
`data/manifests/` metadata is committed (no licensed bytes, no secrets).

---

## 5. What Phase 13 validation receives

The frozen `SIGNAL` / `EXECUTION` split (`docs/RELIABILITY_VALIDATION.md`):

* **execution**: layer-2 `RAW_CONTRACT` per-contract bars + layer-6 roll map →
  `ActiveContractResolver` → `ExecutionSimulator` → `Fill` (`make_fill` still
  rejects any non-`RAW_CONTRACT` price).
* **signal**: layer-3 `RAW_CONTINUOUS` (or a per-as-of point-in-time
  back-adjust the Feature Engine builds) → Feature Engine → `StrategySpec`.
* the canonical futures `trading_day` (from the `SessionCalendar`, Phase 13.1)
  drives the daily validation return series.

The 2025 `LOCKED_HOLDOUT` is acquired only after the strategy / `ValidationSpec` /
`ReliabilityPolicy` / code commit are frozen -- a separate future approval.
