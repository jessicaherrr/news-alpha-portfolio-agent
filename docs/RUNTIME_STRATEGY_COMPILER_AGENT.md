# Runtime Strategy Compiler Agent (Phase 17, corrected in 17.1 / 17.2)

The Strategy Compiler Agent is the second component wired to Claude. It turns one
**already-approved** `HypothesisSpec` (Phase 16 output) into a safe,
schema-valid `StrategySpec` (frozen Phase 10 DSL) — or a typed rejection. It does
not sequence tools, run backtests, or decide anything; that is the orchestrator
(Phase 18).

## Boundary

```
approved HypothesisSpec ─┐
                         ├─► StrategyCompilerAgent ─► CompiledStrategyProposal
CompilerContext ─────────┘      (LLM fills a closed     (StrategySpec + Phase 10
  approved universe               typed blueprint;       CompiledStrategyPlan
  feature catalog                 deterministic code     summary + feature
  template catalog + typed        builds + compiles)     coverage + duplicate
    param bounds                                         evidence, or a typed
  known strategies (+authority)                          rejection_code)
```

The LLM proposes a **closed typed blueprint**; deterministic code builds the spec
and the **unchanged Phase 10 `StrategyCompiler`** is the safety gate. The agent
never produces PnL, fills, verdicts, p-values, validation thresholds, or a
multiple-testing family definition, and it holds no filesystem / registry /
broker handle.

## Full DSL expressivity — not just four families

The model's output is a `StrategyPlanRequest`. When `expressible`, it carries
**exactly one** of:

### `template` — a Phase 11 convenience family

`{family_key, root_symbol, params}` for `tsmom` / `ma_trend` / `breakout` /
`mean_reversion`. Numbers are checked against `PHASE_11_PARAM_BOUNDS`, a **typed**
`ScalarBound` table transcribed from the frozen Phase 11 declared grid ranges —
no prose parsing on the policy path. `assert_phase_11_bounds_match_declared_ranges()`
proves the table reproduces the numeric endpoints declared in
`alpha_agent.strategy.baselines.families`. The spec is rebuilt by the frozen
`spec_for_params` factory, so a template fingerprints **byte-identically** to the
Phase 13.5C canonical/neighbour trials.

### `blueprint` — a full closed-DSL composition

For genuinely new hypotheses. A closed typed mirror of the Phase 10 DSL,
translated node-by-node into a real `StrategySpec`:

| blueprint piece | closed to |
|---|---|
| `features[].kind` | a registered feature `kind` from the catalog (else `UNKNOWN_FEATURE_KIND`) |
| `features[].params` | scalar number / string / bool only — never nested, never an expression |
| `rules[].when` | `comparison` (`op` ∈ gt/gte/lt/lte), `boolean` (`op` ∈ all/any), `not` |
| operands | `feature` (declared alias), `lag` (declared alias, `periods ≥ 1`), `const` (finite number) |
| `rules[].target_units` | a plain integer (the Phase 10 compiler enforces the tight `±10` limit) |
| `default_action` / `on_missing` | `flat`/`keep_previous_target`, `skip_rule`/`hold` |

`extra="forbid"` on every blueprint model + numeric-only validators mean the LLM
cannot emit code, an expression, a callable, an import, a new operator, or a new
feature kind. A blueprint that trips a Phase 10 rule (unsafe feature, non-causal
reference, unbounded tree, execution-plane field) is rejected with
`STRATEGY_COMPILE_REJECTED` and the Phase 10 diagnostics.

If neither a template nor a blueprint can faithfully represent the mechanism,
`{"expressible": false, "not_expressible_reason": "..."}` →
`HYPOTHESIS_NOT_EXPRESSIBLE`. Inventing a strategy is forbidden.

## HypothesisSpec fidelity (17.2)

The compiler *composes* the approved hypothesis; it never silently introduces a
new research feature. After compilation the set of feature **kinds** in the
compiled spec must **equal** `hypothesis.required_features`:

| finding | code |
|---|---|
| a required kind not referenced by any rule (checked against `CompiledStrategyPlan.required_features`, the canonical names actually read while evaluating rules) | `REQUIRED_FEATURE_NOT_REPRESENTED` |
| a feature kind present in the compiled spec (referenced **or** merely declared — a declared feature enters the strategy fingerprint) that the hypothesis did not approve | `UNAPPROVED_FEATURE_ADDED` |

Several `FeatureSpec`s of the same approved kind with different parameters, plus
lags, constants and boolean composition, are all allowed — only a **new kind** is
a fidelity violation. This keeps the division clean: the ResearchAgent decides
what mechanism is tested; the CompilerAgent only translates it.

## Duplicate semantics (17.2) — evidence only, never a block

A matching `strategy_fingerprint` is **pure prior evidence**. Phase 17 does not
construct the complete scientific `experiment_identity` (`ValidationSpec` /
`ReliabilityPolicy` / split / cost + risk config / feature set), so a fingerprint
— even one carrying prior VALID authoritative results — **never rejects
compilation**. A successfully compiled spec is always returned `accepted=True`
with `DuplicateEvidence` attached.

`CompilerContext.known_strategies` is a list of `KnownStrategyRecord` (`source`,
`experiment_id`, `experiment_identity`, `has_valid_authoritative_result`,
`valid_execution_attempts`, `invalid_execution_attempts`). The frozen candidate
manifest is folded in with `source="frozen_candidate_manifest"` and no authority
claim; Phase 18 supplies real records from the live schema-v5 registry.

`DuplicateEvidence` carries **no decision** — only:
`strategy_fingerprint_seen`, `matches`, `prior_valid_authoritative_results`,
`prior_valid_execution_attempts`, `prior_invalid_execution_attempts`,
`prior_experiment_identities`, `prior_experiment_ids`, and a fixed `advisory`.

**Phase 18** builds the full scientific identity and queries the schema-v5
registry:

| registry state | decision (Phase 18) |
|---|---|
| exact `experiment_identity` + VALID authoritative result | block the rerun |
| exact `experiment_identity` + INVALID_EXECUTION-only history | permit another attempt under the same identity |
| same strategy fingerprint, **different** scientific identity | permit as a distinct experiment |

## Execution semantics (stated, not chosen)

Every proposal carries an `ExecutionSemantics` block — the LLM sets none of it:
next-eligible-bar entry, **same-bar fills prohibited** (the DSL has no lead /
future primitive; lag `periods ≥ 1`), target/default-only exits with no
stop/target/bracket orders, and the frozen `$2.00`/contract commission,
`slippage_ticks=0`, `spread_ticks=0`, `latency_bars=0`.

## Retry vs. reject

- **Retry (schema failure only):** unparseable JSON, or JSON that fails
  `StrategyPlanRequest` validation. Fed back as a correction turn, up to
  `max_schema_retries` (default 2). Exhausted → `CompilerSchemaRetryExhausted`.
- **Reject (never retried):** a schema-valid plan that fails a deterministic
  guardrail.

## Provenance

`StrategyCompilerAgent` logs a `PromptLog` per call: `prompt_version`,
`system_prompt_sha256`, `context_sha256` (context **+** hypothesis), model,
temperature, token totals, attempt count. Frozen snapshot:
`scripts/phase_17_strategy_compiler_provenance.py` →
`outputs/phase_17/PHASE_17_STRATEGY_COMPILER_AGENT.json`.

Tests: `tests/python/test_phase_17_strategy_compiler_agent.py` (mocked LLM only,
no network).
