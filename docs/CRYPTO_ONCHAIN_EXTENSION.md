# Phase 22 / 22.1 / 22.1b / 22.1c -- Crypto / On-Chain Extension (Synthetic Scaffold)

`prompts/22_CRYPTO_ONCHAIN_OPTIONAL_EXTENSION.md`:

> Add a separate alternative-data module for BTC/ETH derivatives and on-chain
> features: CME BTC futures basis, perpetual funding/open interest/liquidations
> from approved vendors, exchange flows, MVRV/SOPR/active-address style
> features where provenance is clear. Reuse the same HypothesisSpec,
> StrategySpec, deterministic backtest, validation, registry, and risk
> principles. Do not mix crypto venue assumptions with CME futures execution
> semantics.

## Scope decision (2026-09-11 review gate; hardened 2026-09-12 as Phase 22.1)

Implementing the prompt literally requires connecting to new external crypto
vendors (perpetual funding/OI/liquidations, exchange flows, on-chain
MVRV/SOPR/active-address data) that this repository has never named,
evaluated, or paid for. CLAUDE.md's autonomous-execution section reserves
exactly that decision (**MONEY/NETWORK** -- "any new paid API use ... or new
external market-data spend") for the user, and "approved vendors" names no
vendor anywhere in this codebase.

The user's decision: **synthetic-scaffold-only**. Build the full module --
typed schemas, vendor-neutral adapter interfaces, a strategy family, the real
validation engine wiring, an isolated results store, and a UI page -- against
deterministic synthetic fixtures. Zero network calls, zero paid API use, zero
vendor lock-in, no 2025 holdout access. Real vendor integration is a distinct,
separately-approved future step (mirrors the Phase 15B.1 -> 15B.2 synthetic-
then-real split: 15B.1 built the whole ML DAG on deterministic fixtures before
15B.2's real run touched real data, gated by an explicit `allow_real=True`).

**Phase 22.1** closed four review findings on that scaffold, without
redesigning the approved architecture: (1) the alternative-data surface was
incomplete (no exchange flows, liquidation missing from the unified dataset);
(2) nothing distinguished when a value happened from when it became knowable
to a strategy; (3) provenance was collapsed to a single record instead of
preserved per input series; (4) docs/UI wording overclaimed real CME
execution. See "Phase 22.1 hardening" below for each fix.

Hard constraints from both review gates, all enforced structurally, not just
documented:

* Phase 22 synthetic experiments must never receive an authoritative
  scientific PASS, become paper-trading eligible, mix with the real Phase
  13.5C+ statistical family, appear in the UI as real historical performance,
  or contaminate real-data failure memory.
* Every dataset / feature frame / backtest result / validation report / chart
  / registry row carries explicit `SYNTHETIC` provenance, preserved per
  series, never collapsed.
* Vendor adapters are vendor-neutral typed interfaces so a real vendor can be
  added later with no change to downstream feature / StrategySpec / C++ /
  validation semantics.
* Every alternative-data record distinguishes an observation timestamp from
  an availability timestamp, and the dataset assembler enforces a real causal
  as-of join.
* Presentation is accurate everywhere: the C++ engine, DSL/compiler, and
  validation implementation are REAL; the market data, alternative data,
  contract-economics fixture, and calendar are SYNTHETIC, and so is the
  resulting research evidence.

## What "do not mix crypto venue assumptions with CME futures execution
## semantics" means here

The executed instrument is always a **synthetic CME-futures-shaped execution
fixture** -- the exact same C++ `quant_backtest_targets_csv` engine,
`ContractSpec`, roll/session/risk machinery every other root in this
repository uses, run over synthetic OHLCV rather than real CME BTC/ETH
history. Crypto-derivatives and on-chain series (perpetual funding rate, open
interest, liquidations, exchange flows, MVRV, SOPR, active addresses,
CME-vs-spot basis) are **inputs to the strategy signal only** -- they never
touch the C++ Fill/execution/risk boundary, and the perpetual venue itself is
never traded. This resolves the constraint by construction rather than by a
runtime check: `alpha_agent.crypto.strategy_family._compute_frame` only ever
exposes the crypto feature columns to the Phase 09 feature engine, never the
OHLCV columns the C++ boundary reads (`BOUNDARY_BAR_COLUMNS`), and vice versa.

**Real vs. synthetic, stated plainly (Phase 22.1 presentation fix):**

| | Real | Synthetic |
| --- | --- | --- |
| C++ Quant Core (Fill/roll/risk/accounting) | ✅ | |
| Phase 10 DSL / compiler | ✅ | |
| Phase 13 validation implementation (`ValidationEngine`, `ReliabilityPolicy`) | ✅ | |
| Market data (OHLCV) | | ✅ |
| Alternative data (funding, OI, liquidations, exchange flows, MVRV, SOPR, active addresses, basis) | | ✅ |
| Contract economics (tick size, multiplier) | | ✅ (schema-shape placeholder) |
| Calendar | | ✅ (in-memory 24/7) |
| Resulting research evidence | | ✅ (never authoritative) |

## Package layout (`python/alpha_agent/crypto/`)

| Module | Purpose |
| --- | --- |
| `provenance.py` | `DataProvenanceRole` (`REAL` / `SYNTHETIC`), `CryptoDataProvenance` (+ `content_identity()`, wall-clock independent), `RealVendorNotConnectedError`, `SYNTHETIC_BANNER`. The one choke point every constructor in this package calls. |
| `schemas.py` | `_AltDataRecord` base (`ts_event_ns` / `available_ts_ns >= ts_event_ns` invariant) + typed row schemas a real vendor adapter would eventually populate: `PerpFundingBar`, `LiquidationEvent`, `OnChainMetricBar`, `ExchangeFlowBar`, `CmeCryptoBasisBar`. |
| `vendors.py` | Vendor-neutral `typing.Protocol` interfaces (`PerpFundingVendorAdapter`, `LiquidationVendorAdapter`, `OnChainVendorAdapter`, `ExchangeFlowVendorAdapter`, `CmeCryptoBasisSource`) + `assert_vendor_is_synthetic_only`, which fails closed on anything not tagged `provenance_role == SYNTHETIC`. |
| `synthetic_fixtures.py` | Deterministic, seeded generators for every series above, plus the merged daily OHLCV+crypto-feature bars frame (`synthetic_crypto_bars`, causally as-of joined via `dataset.py`), `synthetic_crypto_generator_identity` (a stable, content-based fingerprint over the generator's own parameters), and a fabricated, clearly-labelled placeholder contract row (`synthetic_contract_row` -- schema-shape only, never a real CME specification, never routed through `alpha_agent.data.contract_economics`). Also the `Synthetic*Adapter` Protocol implementations. |
| `dataset.py` | **Phase 22.1.** The ONLY causal as-of join: `causal_as_of_join` (`pandas.merge_asof`, `direction="backward"`), `aggregate_daily_liquidations`, `daily_series_from_exchange_flow_bars`, `build_causal_feature_columns`. A value is visible to bar `T` iff `available_ts_ns <= T` -- never earlier, regardless of how informative or recent it is. |
| `calendar.py` | An in-memory, 24/7 `SessionCalendar` for `ValidationEngine`'s trading-day plan, passed explicitly (`calendar=crypto_calendar()`) -- never registered into the shared `configs/calendars/cme.yaml` the real futures family reads. |
| `hypotheses.py` | Example `HypothesisSpec` instances (the real, unmodified Phase 16 schema -- no crypto-specific extension needed). |
| `strategy_family.py` | `FundingContrarianParams` + `make_funding_contrarian_spec` -- a real, unmodified `StrategySpec` built from the real Phase 10 DSL, reusing the existing, unmodified `zscore` feature kind over a `price_field` that happens to be a synthetic funding-rate column. `CRYPTO_FUNDING_CONTRARIAN_FAMILY` is deliberately not one of `candidates_phase_13_5c.BASELINE_FAMILIES`. `CRYPTO_FEATURE_PRICE_FIELDS` exposes every declared surface to the DSL, not just the one column this demonstration strategy reads. |
| `envelope.py` | **Phase 22.1.** `SyntheticCryptoValidationArtifact` -- wraps the untouched, frozen Phase 13 `ValidationReport` with `data_role`, `banner`, complete `provenance_by_series`, `dataset_identity`, `generator_identity`, fingerprints, and `content_identity()` (a stable identity excluding wall-clock `generated_at_utc`). `SYNTHETIC_POLICY_OUTCOME_LABEL` is the presentation label every surface must use instead of a bare "Verdict". |
| `research.py` | Wires all of the above into the real `ValidationEngine`, the frozen `ReliabilityPolicy` (`phase_13_5c_matrix.frozen_policy()`, reused unmodified) and null/bootstrap/cost-stress configuration, and -- when the compiled C++ core is present -- the real `CliBacktestRunner` subprocess boundary. `run_crypto_research()` refuses unless every input series' provenance is verifiably `SYNTHETIC`, and returns a `SyntheticCryptoValidationArtifact`. |
| `synthetic_registry.py` | An isolated, append-only sqlite store (`data/crypto_synthetic/registry.sqlite`, schema v2) persisting the whole envelope, with its own schema (`data_role` under a `CHECK` constraint), structurally separate from the real Phase 14 registry. **Phase 22.1b:** detects a legacy v1 table (or anything unrecognized) via `PRAGMA user_version` + real column shape and raises `SyntheticRegistrySchemaError` before any v2 query runs -- never fabricates the missing v2 provenance fields, never leaks a raw `sqlite3.OperationalError`. **Phase 22.1c:** `reset_synthetic_store(confirm=True)` archives (renames) the isolated file so a fresh v2 database is created next, and refuses -- via `SyntheticRegistryPathError`, before any rename/unlink -- unless the resolved target is exactly the configured canonical synthetic registry path, so a `--db` override can never point a reset at an arbitrary sqlite file, the real registry, or the paper ledger. |

## Point-in-time availability + the causal as-of join (Phase 22.1)

Every alternative-data row distinguishes:

* `ts_event_ns` -- when the thing happened / was observed;
* `available_ts_ns` -- when the value became knowable to a strategy, with the
  invariant `available_ts_ns >= ts_event_ns` (equality allowed for a
  genuinely immediate synthetic event, e.g. a funding-rate settlement).

`alpha_agent.crypto.dataset` is the only place a row turns into a feature
column, via `causal_as_of_join` (`pandas.merge_asof`, `direction="backward"`):
a value is visible to bar `T` iff `available_ts_ns <= T`. This is a
point-in-time AS-OF join, not a market-data forward-fill -- CLAUDE.md's "no
silent forward-filling across missing market data" forbids papering over a
GAP in a market-data series; an as-of join over a slowly-updating external
series is the correct point-in-time semantics for "what was actually known at
`T`", and before the first publication this module returns an explicit `NaN`,
never an invented value.

Publication delays in the synthetic generator (illustrative, not researched
vendor SLAs):

| Surface | Delay | Why |
| --- | --- | --- |
| Funding rate / open interest | 0 (immediate) | Settles and is known instantly |
| CME-vs-spot basis | 0 (immediate) | Built from two live prices |
| Individual liquidation print | 0 (immediate) | Near-real-time feed |
| Daily liquidation aggregate | day end + 15 min | A day's total isn't known until the day is over |
| On-chain MVRV / SOPR / active addresses | day end + 6 h | Block confirmation + provider processing |
| Exchange flow (inflow/outflow/netflow) | day end + 4 h | Confirmation + address-clustering attribution |

The daily-aggregate rule matters most: prompt 22 warns that "a full-day
liquidation, flow, or on-chain value cannot be treated as available at the
beginning of that same day" -- every one of those three surfaces here becomes
available only after `day_end + delay`, never at `day_start`
(`tests/python/test_phase_22_crypto_extension.py::test_L6`).

## Why results can never leak into the real system

1. **Different file, different schema.** `data/crypto_synthetic/registry.sqlite`
   shares no table, no identity function, and no code path with
   `data/registry/experiments.sqlite`.
2. **No `Authority` concept.** The synthetic schema has no
   `SUPERSEDES`/`AUTHORITATIVE` notion at all -- a row here is never the
   "current scientific answer" to anything (verified by introspecting the
   live schema, `test_F5`).
3. **Structurally unrebuildable for paper trading, regardless of verdict.**
   `alpha_agent.paper.eligibility` only ever opens the real
   `ExperimentRegistry` and rebuilds a strategy via
   `candidates_phase_13_5c.spec_for_params`, keyed on
   `BASELINE_FAMILIES`. `crypto_funding_contrarian` is not a member, so even
   a mistaken cross-wire cannot produce a paper-eligible strategy from a
   synthetic row -- proven even for a forced synthetic PASS (`test_G4`).
4. **CHECK constraint at the schema level.** Every row's `data_role` column is
   `CHECK (data_role = 'SYNTHETIC')` -- belt-and-suspenders on top of the typed
   `CryptoDataProvenance` guard already enforced before a row is built.
5. **UI is read-only, always shows the banner, and never a bare "Verdict".**
   `alpha_agent/ui/views/crypto_lab.py` reads only the isolated store via
   `alpha_agent.ui.services` functions, renders `SYNTHETIC_BANNER`
   unconditionally, and labels every outcome
   `SYNTHETIC_POLICY_OUTCOME_LABEL` ("Synthetic Policy Outcome
   (NON-AUTHORITATIVE)") -- no other page references the crypto package at
   all (statically tested).
6. **Holdout guard applied defensively.** `record_synthetic_experiment` sweeps
   every payload through the real Phase 14 `assert_no_holdout_market_data`
   guard before writing, even though the fixture calendar (2016-era,
   `SYNTHETIC_BASE_NS`) is already years away from the real 2025 locked
   holdout.

## Legacy synthetic registry schema guard (Phase 22.1b)

Phase 22.1 changed the isolated store's row shape from schema v1 (a single
collapsed `provenance` record, `bdafd12`) to schema v2 (the complete
`provenance_by_series` envelope). There is **no migration** from v1 to v2: a
v1 row never captured the complete provenance the v2 envelope requires, and
inventing those missing fields for an old row would put fabricated content
into a scientific-looking artifact -- not allowed even though this store is
non-authoritative, synthetic, and never part of the real registry or BH/FDR
family.

Instead, every open of an *existing* database detects its real on-disk shape
before any v2 query runs:

* `PRAGMA user_version` plus the actual column set (never trusted blindly --
  a database written by the Phase 22.1 code before this guard existed has real
  v2 columns but `user_version` still `0`; it is recognized by its columns and
  stamped `2` now, with no row touched);
* a table matching the exact Phase 22 (`bdafd12`) v1 column set raises
  `SyntheticRegistrySchemaError` with a message explaining what happened and
  how to reset;
* anything else unrecognized fails the same way rather than guessing.

`alpha_agent/ui/views/crypto_lab.py` catches the typed error and renders an
honest "LEGACY SYNTHETIC STORE" state -- it never crashes and never displays a
v1 row as if it were a v2 validation artifact. `scripts/phase_22_crypto_research.py
run|list|show` catch it too and print `SCHEMA_INCOMPATIBLE: ...` to stderr with
a non-zero exit, never a raw traceback or `sqlite3.OperationalError`.

`reset_synthetic_store(confirm=True)` (CLI: `reset-store --confirm`) archives
(renames) the isolated file so a fresh v2 database is created next -- it never
deletes data and never runs without explicit confirmation. **Phase 22.1c:**
this is a DESTRUCTIVE action, so it is restricted at the function boundary
itself, not merely by the fact that this module doesn't import the real
registry: the resolved target must equal the currently-configured canonical
synthetic registry path (`DEFAULT_DB_PATH`, read fresh at call time so a test
can `monkeypatch` it to a tmp directory), or the call raises
`SyntheticRegistryPathError` before any rename/unlink -- `--db` freely
overrides the target for `run`/`list`/`show` (read/append), but passing it to
`reset-store` with any other value is refused with a typed `PATH_REFUSED`
message and a non-zero exit. This is proven against the real
`data/registry/experiments.sqlite` (byte-hash unchanged, no `.legacy-*`
backup created) and against the Phase 21 paper ledger path, not just asserted.

## Running it

```
# run one synthetic hypothesis end-to-end and record it (OFFLINE, free)
python scripts/phase_22_crypto_research.py run
python scripts/phase_22_crypto_research.py list
python scripts/phase_22_crypto_research.py show --experiment <experiment_id>

# only if a legacy v1 store is detected (SCHEMA_INCOMPATIBLE) and you want to
# start recording v2 results again -- archives, never deletes:
python scripts/phase_22_crypto_research.py reset-store --confirm
```

The CLI prints `Synthetic Policy Outcome (NON-AUTHORITATIVE): <verdict>` to
stderr, never a bare "Verdict:" line. Streamlit: the "Crypto Lab" page
(`streamlit run python/alpha_agent/ui/app.py`) renders whatever has been
recorded, read-only, with the synthetic banner always shown and the same
non-authoritative label -- or the legacy-store state if schema v1 is detected.

## Adding a real vendor later

Write one class per `alpha_agent.crypto.vendors` Protocol
(`PerpFundingVendorAdapter`, `LiquidationVendorAdapter`, `OnChainVendorAdapter`,
`ExchangeFlowVendorAdapter`, `CmeCryptoBasisSource`) tagged
`provenance_role = DataProvenanceRole.REAL`, and supply a REAL, vendor-
documented `available_ts_ns` per record (never copy the synthetic delay
constants). Nothing downstream (feature computation, `StrategySpec`, the
causal join, the C++ boundary, the validation engine) needs to change --
only:

1. a new MONEY/NETWORK approval (Databento `metadata.get_cost()` first, an
   explicit user-provided maximum cost, CLAUDE.md market-data rules);
2. a real registry write path parallel to `synthetic_registry.py` (or a
   promotion path with its own explicit `SUPERSEDES`/`CORRECTS` edge -- never
   an in-place upgrade of a synthetic row to real);
3. removing the `assert_vendor_is_synthetic_only` gate from whatever new
   "real" entrypoint is built alongside (never from `run_crypto_research`
   itself, which stays the synthetic-scaffold path).

## Tests

`tests/python/test_phase_22_crypto_extension.py` -- provenance enforcement
(including wall-clock-independent `content_identity`), vendor Protocol
conformance for every surface (including exchange flows), the
`available_ts_ns >= ts_event_ns` invariant, fixture determinism,
zero-network-call static guard, real DSL/feature-registry/validation-engine
reuse across the FULL alternative-data surface (including a full run through
the real compiled C++ engine when present, with deterministic replay),
isolated-store append-only + `CHECK`-constraint + no-authority-column
behaviour, structural paper-ineligibility (including under a forced synthetic
PASS), the causal as-of join (no forward-looking leak, delayed-publication
safety, prefix invariance, the full-day-aggregate rule), complete
per-series-provenance survival end to end, and a static wording guard that
scans every Phase 22 source file, the UI page, the CLI, and this document for
language claiming a real, historically-traded CME contract is what actually
executes. Section O (Phase 22.1b) constructs the EXACT Phase 22 (`bdafd12`)
v1 table shape in a temporary sqlite database and proves: a fresh database
initializes as v2 and persists the version; a v2 database predating the
version-pragma stamp is recognized by its columns, not rejected; an existing
v2 store round-trips normally; a seeded v1 store is detected before any v2
`SELECT`/`INSERT`; `list`/`get`/`record` on v1 never leak a raw
`sqlite3.OperationalError`; no v1 row is silently transformed into a v2
artifact; a malformed/unknown schema fails closed; the Crypto Lab page renders
the legacy-store state without exception (and never shows the v1 row's own
content); the CLI returns a typed `SCHEMA_INCOMPATIBLE` message and non-zero
exit, never a traceback; `reset-store` requires `--confirm`; and the real
Phase 14 registry / Phase 21 paper eligibility remain provably unaffected
throughout. Section P (Phase 22.1c) proves the destructive-reset path
restriction directly: `reset_synthetic_store` refuses an arbitrary path at
the function boundary before any content is even read; the real CLI mutation
path (`--db <path> reset-store --confirm`) refuses an arbitrary sqlite file,
a fake Phase 21 paper-ledger path, AND -- the mandatory negative control --
the real `data/registry/experiments.sqlite` itself, verified by a SHA-256
hash of that file taken before and after the attempt (unchanged), the file's
parent directory listing before and after (unchanged -- no `.legacy-*`
backup created), a non-zero exit code, and a `PATH_REFUSED` message on
stderr with no traceback; `list`/`show` are confirmed to keep accepting an
arbitrary `--db` (only `reset-store` is restricted); and the restriction is
proven to be a live, call-time lookup of the configured canonical path (not a
value bound once at function-definition time), so a test can `monkeypatch`
it to a tmp directory and reset still works normally against that path.
