# Closed Typed Strategy DSL and Compiler (Phase 10 / 10.1)

The safe language through which the AI Research Layer expresses **strategy logic
only**: it transforms signal-safe feature values into target-position intent. It
does not control execution.

```
FeatureSpec  ->  FeatureFrame  ->  StrategySpec  ->  StrategyCompiler
    ->  CompiledStrategyPlan  ->  ReferenceEvaluator  ->  StrategyDecision
```

Package: `python/alpha_agent/strategy/` (`enums`, `errors`, `spec`,
`fingerprint`, `plan`, `compiler`, `evaluator`, `decision`, `examples`). The
C++ Quant Core is unchanged by this phase.

## 0. The one authoritative StrategySpec (Phase 10.1)

There is exactly one executable strategy schema. The canonical import — the path
future Research / Strategy-Compiler agents must use — is:

```python
from alpha_agent.strategy import StrategySpec        # == alpha_agent.strategy.spec.StrategySpec
```

The Phase-01 placeholder `alpha_agent.schemas.strategy` (a free-text
`entry_rule` / `exit_rule` / `cost_model` model) has been **deleted**. Importing
it now raises `ModuleNotFoundError`; `alpha_agent.schemas` exposes no
`StrategySpec`. A legacy-shaped payload cannot enter any path: `StrategySpec`
forbids unknown fields and deep-scans for execution-plane names, so
`compile_strategy({... "entry_rule": ..., "cost_model": {"slippage_ticks": ...}})`
raises `StrategyCompileError` (`illegal_dsl_field`). A regression test
(`test_strategy_dsl_hardening.py`) asserts no production module imports a
`schemas.strategy` path and that the DSL `StrategySpec` is the single object
reachable as `StrategySpec`.

---

## 1. Closed DSL — no arbitrary code

Forbidden and structurally impossible: arbitrary Python/C++, `eval`/`exec`,
import names, lambdas, user callables, free-form expressions, SQL/code snippets,
dynamically loaded operators. Every legal behaviour comes from a closed
allow-list enumerated in `enums.py`:

| enum | values |
|------|--------|
| `Comparator` | `gt`, `gte`, `lt`, `lte` |
| `BooleanOp` | `all`, `any` (+ a dedicated `not` node) |
| `OperandType` | `feature`, `lag`, `const` |
| `DefaultAction` | `flat`, `keep_previous_target` |
| `MissingRulePolicy` | `skip_rule`, `hold` |

`eq` / `ne` are deliberately **not** implemented — exact float equality is not a
meaningful trading operator and invites cross-platform non-determinism. Use a
banded `gt` / `lt` pair.

Conditions support `FeatureRef vs const` and `FeatureRef vs FeatureRef`. There is
no arithmetic on operands.

## 2. StrategySpec (schema)

Frozen Pydantic, `extra="forbid"` on every model, one `root_symbol` per spec
(section 20; cross-market features may still be *inputs*).

```
StrategySpec
  schema_version : "strategy-dsl/1"
  strategy_name  : str            # identity only — not in the fingerprint
  strategy_id    : str            # identity only — not in the fingerprint
  root_symbol    : str            # ^[A-Z0-9]{1,12}$
  features       : [FeatureDeclaration]   # >= 1
  rules          : [Rule]                 # >= 1, ordered
  default_action : DefaultAction          # explicit, never implicit
  on_missing     : MissingRulePolicy = skip_rule
  rationale      : str  = ""      # free text — never affects behaviour or hash
  metadata       : {str: str} = {}  # labels — never affects behaviour or hash

FeatureDeclaration { alias: ^[A-Za-z][A-Za-z0-9_]*$ , spec: FeatureSpec }   # Phase 09 spec
Rule { rule_id: str , when: ConditionNode , action: TargetAction , rationale: str = "" }
TargetAction { target_units: int }         # TARGET POSITION only, never order quantity
```

### Condition tree (recursively typed, finite depth)

Discriminated union on `type`:

* `comparison` — `{ op: Comparator, left: Operand, right: Operand }`; at least
  one operand must be a feature.
* `boolean` — `{ op: all|any, nodes: [ConditionNode] }` (>= 1 child).
* `not` — `{ node: ConditionNode }`.

Operands (`Operand`, discriminated on `type`):

* `feature` — `{ feature: <alias> }` — current point-in-time value.
* `lag` — `{ feature: <alias>, periods: int >= 1 }` — value `periods` bars ago.
  **Strictly causal.** `periods == 0` is ambiguous and rejected; there is no
  negative lag, `lead`, `future`, or centered window in the vocabulary
  (section 9). `cross_above` / `cross_below` are deferred — not required by
  `prompts/10`.
* `const` — `{ value: int | float }` — see strict typing below.

### Strict scalar types (Phase 10.1) — no silent Pydantic coercion

| field | accepts | rejects (never coerced) |
|-------|---------|-------------------------|
| `TargetAction.target_units` | a true `int` (`1`, `0`, `-2`); compiler bounds `|units|` | `1.0`, `1.5`, `"1"`, `True`, `False`, `None` |
| `LagOperand.periods` | a true `int` `>= 1` (`1`, `5`) | `0`, `-1`, `1.0`, `"5"`, `True` |
| `ConstOperand.value` | a numeric `int` or `float`, finite | `"0.5"` and other strings, `True` / `False`, `NaN`, `+inf`, `-inf` |

Enforced by `mode="before"` field validators (`spec._strict_int` /
`spec._strict_number`) that run ahead of Pydantic's lax coercion. A violation is
a `ValidationError` at parse and a `StrategyCompileError`
(`invalid_target_units` / `invalid_constant` / `future_temporal_reference`) at
compile.

## 3. Feature-safety compilation gate

For every declared `FeatureSpec` the compiler runs
`REGISTRY.validate_spec(spec)` (Phase 09 §6 parameter contract: unknown kind,
unknown / missing / mistyped / out-of-range parameter, `fast < slow`
constraints, canonical-name collisions) and then requires the spec-level
`FeatureMetadata` to be **`point_in_time_safe` and `signal_safe`**. A
retrospective future-roll feature (`bars_until_next_roll`,
`contract_transition_indicator`, `roll_mode=retrospective`) is rejected here.

The **source-level** case — a retrospective back-adjusted price series makes
*every* derived feature look-ahead-unsafe — cannot be seen at compile time (the
spec names no source). It is caught by the evaluator, which calls
`FeatureFrame.assert_signal_safe()` before doing anything else. Allowed:
causal RawContract-derived, causal RawContinuous-derived, point-in-time
BackAdjusted-derived. Rejected: retrospective BackAdjusted / retrospective
future-roll / any look-ahead-unsafe feature.

Feature execution-price safety is irrelevant here: **no feature value is ever an
execution price** (`FeatureSafety.execution_price_safe` is always `False`).
Features drive `Signal`s only.

## 4. Execution plane is inaccessible

No DSL model has a field for `fill_price`, `execution_price`, `reference_price`,
`raw_symbol`, `instrument_id`, `contract_month`, `slippage`, `spread`,
`commission`, `latency`, `margin`, risk overrides, `RiskDecision` or `Fill`.
`extra="forbid"` rejects any such field, and `StrategySpec` additionally
deep-scans the incoming payload for a denylist of execution-plane names
(`spec.FORBIDDEN_EXECUTION_FIELDS`) and raises a legible error. A `FeatureRef`
can never be used as an execution price. The `strategy` package imports nothing
from the execution / risk / portfolio / adapter layers (enforced by a test).

Execution stays entirely in C++:
`RawContract MarketEvent -> Order -> RiskManager -> ExecutionSimulator -> Fill`.

Strategy does **not** see portfolio / risk state (`cash`, `margin`, `drawdown`,
`risk_limit`) — those belong to the deterministic Hard Risk layer (section 21).

## 5. Missing-value semantics (three-valued logic)

A condition evaluates to `TRUE` / `FALSE` / `UNKNOWN`. `NaN` / missing never
silently becomes a boolean.

* comparison with a missing operand → `UNKNOWN`.
* `NOT UNKNOWN = UNKNOWN`.
* `ALL`: `FALSE` if any child `FALSE`, else `UNKNOWN` if any `UNKNOWN`, else `TRUE`.
* `ANY`: `TRUE` if any child `TRUE`, else `UNKNOWN` if any `UNKNOWN`, else `FALSE`.

A rule matches iff its condition is `TRUE`. A rule that evaluates `UNKNOWN` is
**not evaluable**: under `skip_rule` (default) it is skipped and evaluation
continues (“first matching *evaluable* rule wins”); under `hold` the first
non-evaluable rule stops the scan for that bar. When no rule matches, the
`default_action` applies. Missing values are never converted to zero; every
non-evaluable rule and every missing feature is recorded on the
`StrategyDecision`.

## 6. Target / default semantics

Actions specify **target position** only (`target_units`, integral, no
fractional futures, `|units|` bounded by the compiler — default 10). C++ remains
authoritative for `order_delta = target_position − current_position`.

`default_action`:

* `flat` → `target_units = 0`.
* `keep_previous_target` → re-emit the last emitted target. This is
  strategy-local state only; the evaluator keeps exactly one integer (the last
  emitted target, seeded at 0). It never reads portfolio/execution internals.

Rule precedence: **ordered, first matching evaluable rule wins**. The order is
part of the canonical spec; nothing depends on dict/hash iteration order.

## 7. Canonicalization / fingerprint

`strategy_fingerprint(spec)` = `stratdsl1:` + SHA-256 of the canonical semantic
payload (`fingerprint.canonical_strategy_payload`). The same *semantic* spec
always produces the same fingerprint regardless of:

* `strategy_name` / `strategy_id`
* `rule_id` labels, per-rule and top-level `rationale`, `metadata`
* feature **alias** spelling — features are keyed by their canonical
  `FeatureSpec` (`FeatureSpec.canonical_json()`), not the local alias
* child order inside `ALL` / `ANY` (commutative — children are sorted)
* dict / mapping iteration order

A semantic change always moves the fingerprint: different comparator, constant,
feature parameter, `target_units`, `NOT` wrapper, **rule order**,
`default_action`, or `on_missing`. This fingerprint later backs the experiment
registry, failure memory, near-duplicate detection and reproducibility.

## 8. CompiledStrategyPlan (closed typed IR)

No source code is generated. `compiler.compile(spec) -> CompiledStrategyPlan`:

* `feature_bindings` — `alias → canonical_name + FeatureSpec + FeatureMetadata`
* `declared_features` / `required_features` — canonical names
* `rules` — `CompiledRule { rule_id, condition (CompiledCondition, aliases
  resolved to canonical names), target_units, rationale_label }`
* `default_action`, `on_missing`
* `warmup_bars` — `max(feature.minimum_observations + lag_periods)` over
  referenced operands
* `safety` — `SafetyRequirements { require_signal_safe, require_point_in_time_safe,
  forbid_execution_price_use }`
* `canonical_payload`, `fingerprint`

Compiling the same spec twice yields an identical plan (no timestamps / paths /
random ids in the plan).

### Compile-time rejections (typed `StrategyDiagnostic` list)

`unknown_feature`, `invalid_feature_params`, `unsafe_feature`,
`duplicate_feature_alias`, `canonical_name_collision`, `unknown_feature_ref`,
`unknown_operator`, `unknown_action`, `malformed_condition`, `invalid_constant`,
`invalid_target_units`, `illegal_dsl_field`, `condition_too_deep`,
`condition_too_many_nodes`, `future_temporal_reference`, `duplicate_rule_id`,
`too_many_rules`, `too_many_features`. `CompileLimits` (all configurable):
`max_condition_depth=8`, `max_condition_nodes=64`, `max_rules=64`,
`max_features=32`, `max_abs_target_units=10`.

## 9. Reference evaluator

`ReferenceEvaluator(plan).evaluate_frame(frame) -> [StrategyDecision]`. It is for
DSL semantic validation, **not** production trading. Official execution and PnL
remain in the C++ Quant Core.

It consumes only: the plan, a point-in-time `FeatureFrame`
(timestamps, identifiers, feature values + masks), and one integer of
strategy-local state (last emitted target). It never consumes `Fill`, `Order`,
`RiskDecision`, an execution price, portfolio cash/equity, or a future row —
value lookups only ever index row `i` or an earlier row `i − periods`.

Gate order: `frame.assert_signal_safe()` → root-symbol match (when the frame
carries a `root_symbol` identifier) → every `required_features` column present
(missing column ⇒ `StrategyEvaluationError`, the explicit policy).

**Prefix invariance (section 19):** because every lookup is backward-looking and
Phase 09 guarantees each feature value at `T` is bit-identical whether the frame
ends at `T` or later, `Decision(T)` is identical for a frame ending at `T` and a
longer frame through `T+N` — including `lag` operands and
`keep_previous_target` state. Deterministic replay: same plan + same frame ⇒
identical decisions.

## 10. Decision audit record

`StrategyDecision` (frozen, `extra="forbid"`): `ts_event_ns`,
`strategy_fingerprint`, `root_symbol`, `target_units`, `matched_rule_id | None`,
`default_applied`, `referenced_feature_values` (`{canonical_name |
"<name>@lag<N>": float | None}` — every value actually read this bar),
`missing_features`, `not_evaluable_rule_ids`, `rationale_label | None`.

**No execution or fill price.** No order quantity. No risk field.

## 10a. Serialization safety (Phase 10.1)

`strategy_from_yaml` uses `yaml.safe_load` **only** — an
`!!python/object/apply:` or `!!python/name:` tag raises rather than constructing
an object. `strategy_to_yaml` uses `yaml.safe_dump`. JSON and YAML round-trips
preserve semantics, the fingerprint, and produce byte-stable text (regression
test `test_strategy_dsl_hardening.py`).

## 11. Tests

`tests/python/test_strategy_dsl.py` — 38 cases covering checklist A–V:
valid compile; unknown feature / unknown param; retrospective roll feature
rejected at compile and retrospective back-adjusted source rejected at
evaluation; point-in-time back-adjusted feature accepted; FeatureRef cannot
carry an execution price and execution fields rejected anywhere in the payload;
unknown DSL field / operator / action rejected; `ALL`/`ANY`/`NOT` three-valued
truth tables; ordered first-match precedence; missing feature ⇒ not-evaluable +
`hold` policy; integral / bounded `target_units`; `keep_previous_target` state;
NaN/inf constants rejected; JSON + YAML round-trip (semantics + fingerprint +
byte-stable text); fingerprint stable under cosmetic change; rationale does not
move the fingerprint; semantic change does; prefix-invariant decision at `T`
(with `lag` + `keep_previous_target`); no future temporal primitive; feature-
value audit; the `strategy` package imports no execution-plane module and the
evaluator output is only target intent; deterministic replay; plus condition-
tree bounds, unknown feature-ref, duplicate rule id / alias, canonical-name
collision, feature-vs-feature comparison, root mismatch, missing required column.

`tests/python/test_strategy_dsl_hardening.py` — 16 Phase-10.1 cases: single
authoritative `StrategySpec` + canonical import; legacy schema module gone; no
production module imports a `schemas.strategy` path; legacy free-text payload
rejected by compiler and schema; strict `target_units` / `lag periods` /
comparison-constant typing (bool, `1.0`, `"1"`, `"0.5"`, `NaN`/`inf` all
rejected); YAML rejects `!!python/object` / `!!python/name` tags; round-trip
fingerprint unchanged; `strategy_from_yaml` uses `safe_load` only.

Full Python suite: **287 passed**. `ruff` clean on the `strategy` package and
both test files.

## 12. C++ Quant Core

**Unchanged.** All 7 `ctest` targets pass. Phases 10 and 10.1 are Python-only.

## 13. What Phase 11 receives

* `StrategySpec` — a closed, versioned, frozen, fingerprinted logic schema that
  future Claude agents emit as data (never code).
* `StrategyCompiler` / `CompiledStrategyPlan` — the validated typed IR plus
  warm-up and safety requirements, ready to hand to a backtest driver that
  routes decisions through the C++ execution path.
* `ReferenceEvaluator` / `StrategyDecision` — deterministic target-position
  intent for DSL semantic checks and prefix-invariance regression.
* the deterministic fingerprint, for the experiment registry / failure memory /
  near-duplicate detection in Phases 13–14.

Phase 11 builds the baseline strategy *library* (trend / carry / mean-reversion /
COT) on top of this DSL. It is **not** implemented here.
