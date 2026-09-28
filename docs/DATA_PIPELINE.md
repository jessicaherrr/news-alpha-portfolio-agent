# Canonical Futures Data Pipeline (Phase 03)

Implements layers 1, 2 and 5 of [`BOUNDARY_CONTRACT.md`](BOUNDARY_CONTRACT.md).
No network calls, no back-adjustment, no roll math, no full CME calendar -- those
are Phase 04+.

```
raw OHLCV artifact  +  raw definition artifact        (layer 1, immutable)
        -> verify_manifest (SHA-256, fail loudly)
        -> parse_definition_frame -> DefinitionRegistry -> contracts.parquet   (layer 5)
        -> canonicalize -> canonical bars .parquet per contract                (layer 2)
        -> lineage sidecar per processed file
```

Entry point: `alpha_agent.data.pipeline.run_canonical_pipeline`.

---

## 1. Raw storage layout

```
data/raw/<vendor>/<dataset>/<schema>/<request_hash>/
    <schema>.<format>                     # bytes exactly as delivered
    <schema>.<format>.manifest.json       # sidecar
```

`request_hash` = first 16 hex of `sha256({dataset, schema, stype_in, stype_out,
sorted(symbols), start, end})`. The same request lands in the same directory, so
a re-download trips the write-once guard instead of silently diverging.

`store_raw()` writes atomically (`.tmp` then `replace`) and **raises
`FileExistsError`** if the artifact or its manifest already exists. Processed
writers never touch these files.

### Raw manifest fields

`vendor, dataset, schema, stype_in, stype_out, symbols[], start, end,
artifact_format, artifact_filename, sha256, row_count?, created_at (UTC ISO),
extra{}`.

`verify_manifest(path)` recomputes the SHA-256 and raises `ManifestMismatchError`
on any mismatch, `FileNotFoundError` if either file is missing.

---

## 2. Canonical bar schema (layer 2)

Written to `data/processed/bars/<root>/<raw_symbol>.parquet`, columns in exactly
this order (`CANONICAL_BAR_COLUMNS`, `CANONICAL_SCHEMA_VERSION = "1.0"`):

| column | dtype | notes |
|---|---|---|
| `ts_event_ns` | int64 | UTC ns, **preserved byte-for-byte** from the vendor record |
| `instrument_id` | int64 | non-zero; resolves in the `DefinitionRegistry` |
| `raw_symbol` | str | **authoritative value from the contract definition**, not the vendor symbol column |
| `root_symbol` | str | from the definition's `asset` field (or an explicit `root_map`) |
| `open/high/low/close` | float64 | de-scaled: `fixed_point / 1_000_000_000` |
| `volume` | int64 | contracts in the interval |
| `trading_day` | date | derived from the session calendar |
| `session` | str | `RTH / ETH / MAINTENANCE / CLOSED`, derived |

---

## 3. Contract-definition schema (layer 5)

Written to `data/processed/contracts/<root>.parquet` in `CONTRACT_COLUMNS` order
-- the same header the C++ core parses (`cpp/src/contract_io.cpp`):

`instrument_id, raw_symbol, root_symbol, exchange, tick_size, multiplier,
activation_ns, expiration_ns, first_notice_ns, last_trade_ns` (last two `""` when
absent).

`parse_definition_frame` maps vendor `InstrumentDefMsg`-style records
(`VENDOR_DEFINITION_COLUMNS`) and **derives** `tick_size` and `multiplier` via
`alpha_agent.data.contract_economics` (`docs/CONTRACT_ECONOMICS.md`):
`tick_size = min_price_increment` (quoted units), `multiplier = point_value_usd`
(from `unit_of_measure_qty` + the documented price-scale derivation, **never**
`contract_multiplier`, which is an INT32_MAX sentinel on GLBX.MDP3). An
`EconomicsError` aborts the parse rather than guessing. Every row is validated
through `ContractSpecModel`.

Verified on real NQU6: `tick_size=0.25`, `multiplier=20.0`,
`tick_size*multiplier == 5.0` == USD tick value.

> **Phase 04 must** add the fractional-quoting derivation (CBOT Treasuries;
> `main_fraction`) and source `first_notice_ns` for physically-delivered
> products (CL, GC, ZN) -- Databento definitions carry no such field.

**Authority:** a symbol is tradable **iff** it is a member of the
`DefinitionRegistry`. Syntax (`is_tradable_contract_symbol`) is a necessary
pre-filter, never sufficient.

---

## 4. Root-symbol derivation

No regex futures parser. In order:

1. the vendor definition `asset` field (authoritative);
2. an explicit `root_map` (`raw_symbol -> root`) passed to the pipeline;
3. otherwise the row is **rejected** (`parse_definition_frame` raises;
   `ROOT_UNDETERMINED` diagnostic at the bar level).

---

## 5. Normalization rules (`canonicalize`)

Applied in order; `NormalizationPolicy` chooses fail-vs-reject-row per class.

| rule | default action |
|---|---|
| `ts_event` preserved exactly (UTC), renamed `ts_event_ns` | always |
| duplicate `(instrument_id, ts_event_ns)` | **always fatal** (`PipelineError`) |
| DBN `UNDEF_PRICE` sentinel in any OHLC field | reject row (`on_undefined_price`) |
| `instrument_id` not in registry | reject row (`on_unknown_instrument`) |
| `raw_symbol` resolved from the registry by `instrument_id` | always |
| vendor `symbol` column is a **label** (may be `NQ.v.0`) | never a rejection cause |
| vendor `symbol` is a *real contract* that disagrees with the registry | reject row (`RAW_SYMBOL_MISMATCH`) |
| price de-scale `fixed / PRICE_SCALE -> float64` | always |
| any OHLC `<= 0` | reject row (`on_invalid_row`) |
| `high < max(o,c,l)` or `low > min(o,c,h)` | reject row |
| `volume < 0` | reject row |
| non-monotonic `ts_event_ns` within an instrument (arrival order) | **fail** (`on_non_monotonic="fail"`); `"sort"` sorts + emits `TIMESTAMPS_REORDERED` |
| missing-minute gaps | **record only, never forward-fill** (`on_gaps`) |
| no session calendar for a root | fail (`on_missing_calendar`) |

Rejected rows are dropped from the output and counted in
`report.n_rejected_rows`. **No silent forward-fill. No silent reordering.**

---

## 6. QA / diagnostics

`DiagnosticsReport` (`alpha_agent.data.diagnostics`). `DiagnosticKind`:
`duplicate_bar, non_monotonic_timestamp, timestamps_reordered, invalid_ohlc,
non_positive_price, negative_volume, unknown_instrument_id, raw_symbol_mismatch,
undefined_price_sentinel, missing_minute_gap, root_undetermined,
session_calendar_missing` (`continuous_symbol_as_raw` is retained in the enum but
no longer emitted -- a continuous OHLCV `symbol` label is expected).

Each `Diagnostic` carries `kind, severity (error|warning), message, count,
instrument_id?, ts_event_ns?, context{}`. `report.summary()` (input/output/
rejected row counts + `by_kind`) is embedded in the bar lineage sidecar.

`FATAL_KINDS = {duplicate_bar}` -> `report.ok` is False and the pipeline raises.
`run_canonical_pipeline(require_clean=True)` additionally raises on any
error-severity diagnostic.

Missing minutes: for each intra-instrument gap the missing slots are classified
by the session calendar; only slots in a **tradeable** session (RTH/ETH) are
counted -- a gap fully inside MAINTENANCE / CLOSED (daily or weekly halt) is
**not** flagged. Genuinely-missing minutes are recorded (instrument, gap
start/end ns, count), **not** filled, **not** an automatic dataset rejection
unless `on_gaps="fail"`. (Real NQ 2026-09-03: 0 gaps -- the 60-minute maintenance
break was correctly recognised, not reported.)

---

## 7. Lineage format

`<processed_file>.lineage.json` (`alpha_agent.data.lineage.Lineage`):

`artifact_kind (canonical_bars|contracts), source_raw_sha256, source_manifest_path,
dataset, requested_symbols[], schema, stype_in, stype_out, code_commit,
timezone ("UTC"), price_scale_policy, normalization_policy{}, canonical_schema_version,
generated_at (UTC ISO), diagnostics_summary{}, extra{}`.

`source_raw_sha256` equals the SHA-256 in the source raw manifest -- the audit
chain from a processed bar back to the immutable vendor bytes.

---

## 8. Databento adapter (implemented, not executed)

`HistoricalRequest` defaults `stype_out="instrument_id"` (Databento does not
support `continuous/parent -> raw_symbol`; the builder raises `ValueError` on
that combination). `.definition_request()` derives the `parent` `definition`
companion request, also `instrument_id` output. `fetch_and_store_raw()` does:
`estimate_cost_usd` -> enforce `max_cost_usd` cap -> `get_range(path=<tmp .dbn.zst>)`
streaming the **original DBN bytes** to disk -> `store_raw` (raw artifact =
`<schema>.dbn.zst`). Decode / staging is `alpha_agent.data.decode`. No test
invokes the download; it needs a live key + network.
`scripts/databento_validate.py` orchestrates the one-day validation:
`--execute` downloads once; **`--replay` re-runs the whole chain from the stored
raw DBN with no network and no cost** (Phase 03.5 passed via `--replay`).

---

## Phase 04 (done -- see docs/FUTURES_HISTORY.md)

- Layers 3 (unadjusted continuous) + 4 (research-only back-adjusted) + 6 (roll
  map). Roll trigger = observed `instrument_id` transition.
- Product-aware CME calendars (`configs/calendars/cme.yaml`: equity index /
  energy / metals / rates), holiday + early-close architecture, DST.
- Point-in-time vs retrospective back-adjustment modes.
- Contract lifecycle + external first-notice join interface (dates never
  manufactured).
- Fractional / Treasury economics -- typed architecture + fixtures; real ZN
  verification deferred to Phase 04.5.
- C++ `additive_back_adjust` deleted (back-adjustment is Python-only).

## Still deferred

- Real ZN definition-field verification (Phase 04.5).
- `symbology.resolve` roll cross-check (optional; the instrument_id transition is
  authoritative).
- A real historical CME holiday database.
- `trading_day` stored as a clean `date` (currently coerced to `Timestamp`;
  boundary unaffected).
- Vectorised QA for very large datasets (row-wise is fine for samples).
