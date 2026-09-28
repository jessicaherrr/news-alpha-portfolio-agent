# Contract economics (Phase 03.5)

How Databento instrument-definition fields become the USD quantities a backtest
needs. Implemented in `alpha_agent.data.contract_economics`.

## Concepts (kept explicit and separate)

| name | meaning | NQ |
|---|---|---|
| `quote_tick_size` | `min_price_increment` -- minimum move in the quoted price | 0.25 |
| `contract_size` | `unit_of_measure_qty` -- size in the unit of measure | 20.0 (IPNT) |
| `price_scale` | USD per (1 quoted unit * 1 contract-size unit) | 1.0 |
| `point_value_usd` | USD PnL per 1.0 move in the quoted price, per contract | 20.0 |
| `tick_value_usd` | USD per tick | 5.0 |

The frozen field `ContractSpec.multiplier` / `ContractSpecModel.multiplier`
carries **`point_value_usd`** (a.k.a. pnl_multiplier / point value). The word
"multiplier" is not used inside the derivation to avoid ambiguity.

## Derivation (non-fractional products)

```
tick_value_from_definition = min_price_increment_amount / display_factor
price_scale                = tick_value_from_definition / (min_price_increment * unit_of_measure_qty)
point_value_usd            = unit_of_measure_qty * price_scale
tick_value_usd             = min_price_increment * point_value_usd
```

`price_scale` is rounded to 12 dp, the USD values to 6 dp, to absorb binary
float noise (`0.05/0.01` is not exactly `5.0`).

### Observed real NQ (GLBX.MDP3, NQU6, instrument_id 42004177)

```
min_price_increment        = 0.25
min_price_increment_amount = 0.05
display_factor             = 0.01
unit_of_measure            = IPNT
unit_of_measure_qty        = 20.0

tick_value_from_definition = 0.05 / 0.01           = 5.0
price_scale                = 5.0 / (0.25 * 20.0)   = 1.0
point_value_usd            = 20.0 * 1.0            = 20.0
tick_value_usd             = 0.25 * 20.0           = 5.0
```

Cross-check: `tick_size * multiplier = 0.25 * 20.0 = 5.0 = tick_value_usd`. ✓
CME independently confirms NQ: $20/point, 0.25 tick, $5.00 tick value.

## Sentinels -- never guessed from

GLBX.MDP3 definition records carry integer "unset" sentinels. For NQU6:

| field | value | meaning |
|---|---|---|
| `contract_multiplier` | 2147483647 | INT32_MAX -- **not** the point value |
| `contract_multiplier_unit` | 127 | INT8_MAX |
| `main_fraction` | 255 | UINT8_MAX |
| `decay_quantity`, `original_contract_size` | 2147483647 | INT32_MAX |

`usable_number()` rejects `{2^7-1, 2^8-1, 2^15-1, 2^16-1, 2^31-1, 2^32-1,
2^63-1, 2^64-1}`, non-positive, and non-finite values. `contract_multiplier` is
**never** used as the point value. If `unit_of_measure_qty` /
`min_price_increment_amount` / `display_factor` are missing or sentinels,
`derive_contract_economics` raises `EconomicsError` -- the pipeline stops rather
than guessing.

## Other product families

`price_scale` is **not** assumed to be 1. The derivation is generic:

- equity index (NQ, ES): `IPNT`, scale 1
- energy / metals / FX: different `unit_of_measure` + `unit_of_measure_qty`; the
  same `min_price_increment_amount / display_factor` path applies
- **fractionally-quoted** products (CBOT Treasuries in 32nds/64ths): a
  non-sentinel `main_fraction` is present -- the non-fractional path is **not**
  proven for these, so `derive_contract_economics` **raises** rather than
  mis-derive. Phase 04 adds the fractional derivation.
