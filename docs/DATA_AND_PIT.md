# Data and Point-in-Time (PIT) Integrity

How market data enters the system safely, and why "the data I can see today"
is not the same as "the data that was actually available at decision time."
Detailed supporting documents: `docs/BOUNDARY_CONTRACT.md` (Python/C++ data
model), `docs/DATA_PIPELINE.md` (ingestion layers), `docs/FUTURES_HISTORY.md`
(rolls and back-adjustment), `docs/CONTRACT_ECONOMICS.md` (quote conventions
and point value), `docs/REAL_DATASET_ACQUISITION.md` (the real acquisition
record).

## Point-in-time safety

A signal computed using bar `t` information may execute no earlier than the
next permitted execution point, unless a strategy explicitly and
transparently models intrabar information with data that actually supports
it. News/economic event measurements respect publication lag — the system
never uses a value before the timestamp at which it was actually knowable.

## Publication lag and measurement availability

Every measurement spec in the News Alpha pipeline carries an explicit field
resolution step (`news_alpha/measurement.py`) that maps a desired economic
quantity to a real, PIT-safe data field with a known availability lag. When a
required field cannot be resolved safely for a given date/symbol, the
candidate is excluded via a typed reason — never silently zero-filled or
forward-filled.

## Bars and no silent forward-filling

Missing market data is never silently forward-filled across a gap. A missing
bar is a missing bar; strategies and features that depend on continuity
either handle the gap explicitly (see the regime-transformation contiguous-
index handling in the ML pipeline) or the affected window is excluded.

## Futures rolls

Roll events are explicit, auditable records (`ROLL_EVENT_COLUMNS`), not an
implicit side effect of a continuous-price series. A documented
`RollPricePolicy` governs how the roll reference price is chosen, and the
back-adjustment splice seam is deliberately **not** zero — collapsing it
would silently mix a roll-basis effect into the measured price return.

## Back-adjusted vs. raw execution prices

Back-adjusted continuous prices are useful for signal research (they give a
clean, splice-free series for feature computation) but are **never** used to
simulate a fill. Every simulated fill resolves to the actual raw contract
that was traded on that real date — the C++ execution engine has no notion of
a "continuous" instrument.

## Contract economics: quote conventions

`ContractSpec.multiplier` (point value in USD) is *derived* from the
instrument definition record, never guessed and never taken from the
`contract_multiplier` sentinel field (which is routinely a meaningless
`2147483647` placeholder in raw vendor data).

- **Decimal-quoted** products (ES, NQ, CL, GC, …): point value is derived from
  `unit_of_measure_qty` and the vendor's price scale, which resolves to 1.0
  for these products.
- **Fractionally-quoted** products (CBOT Treasuries, e.g. ZN): the price is a
  **percent of par**, so point value = face value × 0.01. The vendor's raw
  tick-amount field does **not** carry the correct USD tick value for these
  products and must never be used directly — a naive derivation understates
  ZN's point value by roughly 15,000×.

Every new root is validated against the published CME contract specification
before use; a mismatch is a typed data-quality failure that blocks the
research matrix, because commissions are a flat USD-per-contract charge that
does **not** rescale with a wrong point value.

## Real vs. synthetic provenance

Every dataset used in an official result carries explicit provenance: real
vendor data (Databento `GLBX.MDP3`) is content-hashed and stored immutably;
any synthetic fixture used for testing or engineering smoke checks is clearly
labeled as such and is structurally barred from producing an official,
registry-authoritative result (`--run-real` vs. the synthetic test runner are
distinct, non-interchangeable code paths).

## Immutable raw store vs. derived discovery layer

Raw downloaded data is stored immutably, content-hashed, and never modified
in place. A separate, regenerable, human-readable catalog/index layer
provides friendly names and pointers into the raw store — it is a discovery
convenience, never a second source of truth, and never holds the actual
market-data bytes.

## Data-quality gaps

Not every economic measurement is available for every date/symbol. Where a
measurement cannot be safely resolved, the system reports the gap explicitly
(a typed exclusion reason) rather than filling it with an interpolated or
assumed value. A candidate signal with unresolvable measurements for its
event window is excluded from that trial, not silently degraded.

## Acquisition discipline

Before any historical data request that may incur a charge, the system calls
the vendor's cost-estimation endpoint first and requires an explicit,
user-approved spend cap. The real dataset backing the primary validated
research program (2018–2024, ES/NQ/CL/GC/ZN) was acquired for **$57.29**
against a **$65.00** approved cap — see `docs/REAL_DATASET_ACQUISITION.md`
for the full record.
