# Runtime Research Agent (Phase 16)

The Research Agent is the first component wired to Claude. It runs only now that
every deterministic tool it depends on exists: the feature registry, the
reliability validator, the experiment registry, failure memory, and duplicate
prevention. Its single job is to turn a frozen, typed context into one
schema-valid `HypothesisSpec`. It does **not** sequence tools, run backtests, or
decide anything — that is the orchestrator (Phase 18).

## Boundary

```
deterministic layers ──► ResearchContext ──► ResearchAgent ──► ResearchProposal
  feature registry          (frozen,           (LLM +            (HypothesisSpec
  universe.yaml              holdout-guarded)    guardrails)       or typed
  registry summary                                                rejection)
  failure memory
  validation feedback
```

The LLM proposes; deterministic code decides admissibility. The agent never
produces PnL, fills, verdicts, p-values, validation thresholds, or a
multiple-testing family definition.

## Inputs the agent is allowed to see (`ResearchContext`)

| field | source | notes |
|---|---|---|
| `objective` | caller | plain-text research goal |
| `market_universe` | `configs/universe.yaml` | approved CME roots |
| `feature_catalog` | `alpha_agent.features.REGISTRY` | closed `kind` labels + declared params, no callables |
| `knowledge_base` | caller / literature notes | free-text summaries |
| `registry_digest` | `ExperimentRegistry.summary()` | counts only, holdout-excluded |
| `failure_memory` | `FailureMemory.lookup(...)` | per-neighbourhood digest |
| `validation_feedback` | prior validation reports | free-text critic summaries |
| `avoid_repeating` | orchestrator | titles/mechanisms already rejected |

Everything else is invisible: no filesystem, no registry handle, no market data,
no ability to run a backtest, no 2025.

`ResearchContext` is a frozen Pydantic model. On construction, every string and
integer in the serialised bundle passes `assert_no_holdout_market_data` — a
2025-or-later market-data/performance value anywhere in the context is a hard
`HoldoutAccessError`.

## Output

One `HypothesisSpec` (unchanged schema): `economic_mechanism`, `universe`,
`horizon`, `required_features`, `signal_description`, `expected_regime`,
`failure_regime`, `falsification_test`, `novelty_notes`.

`ResearchProposal` wraps the result: either `accepted=True` with a `hypothesis`,
or `accepted=False` with a typed `rejection_code`. It always carries the
`PromptLog` and every `AttemptRecord`.

## Retry vs. reject

- **Retry (schema failure only):** the model returned unparseable JSON, or JSON
  that fails `HypothesisSpec` validation. The error text is fed back as a
  correction turn, up to `max_schema_retries` (default 2). Exhausted →
  `SchemaRetryExhausted` carrying all attempts.
- **Reject (non-retryable):** a schema-valid spec that violates a deterministic
  guardrail —
  - `MARKET_OUTSIDE_UNIVERSE` — a root not in the approved universe
  - `UNKNOWN_FEATURE` — a `required_features` entry not in the catalog
  - `EMPTY_UNIVERSE` / `NO_REQUIRED_FEATURES`

  These are **not** retried: silent retries would multiply tests without
  registry records (Phase 18 rule). The rejection is returned for the
  orchestrator to record.
- **Fail loud:** a locked-holdout reference in the produced spec raises
  `HoldoutAccessError`, never a soft rejection.

## Budgets

`max_schema_retries` caps the turn count. `token_budget` (default 40k) caps
cumulative input+output tokens across attempts; exceeding it raises
`BudgetExceeded` before the next call.

## Prompt / version logging

`PromptLog` records `prompt_version` (16-hex of the template), `prompt_path`,
`system_prompt_sha256`, `context_sha256`, model, temperature, token totals, and
attempt count. An optional `log_sink` callback receives it; the agent itself
never writes a file.

## LLM transport

`alpha_agent.agents.llm`:

- `LLMClient` — structural protocol, one `complete()` call.
- `AnthropicClient` — lazy adapter. `import anthropic` happens inside the call,
  never at import, so the core suite runs without the `ai` extra. An API key is
  never read from source; the SDK reads `ANTHROPIC_API_KEY`, or the caller passes
  a key it pulled from the environment.
- `ScriptedLLMClient` — deterministic test double. Every Phase 16 test uses it;
  no network.

## Phase 15B as failure memory (and the Phase 16.1 abstraction)

`FailureMemoryDigest` keeps three kinds of count strictly separate — they are
computed in `FailureMemory.lookup` from schema-v5 execution attempts, never
inferred from failure-record counts:

| kind | fields | counts |
|---|---|---|
| execution attempts | `invalid_execution_attempts`, `valid_execution_attempts`, `invalidation_class_counts` | **unique runs** of a hypothesis. An `INVALID_EXECUTION` attempt is an engineering / data-pipeline defect — **not** a refutation; re-execution under the same identity is permitted and the agent is told to say so. |
| valid scientific outcomes | `valid_scientific_outcomes`, `valid_scientific_refusals`, `scientific_verdict_counts` | **unique results** from a `VALID` attempt, at most one per experiment. A *refusal* (protocol declined to adjudicate for lack of evidence) is a real outcome; a repeat proposal **must** state what is materially different. |
| typed record histograms | `engineering_failure_records`, `scientific_failure_records` | typed `failures` **records**. Several can attach to one attempt or outcome, so these are evidence detail — never a count of attempts or hypotheses. |

**Invariant:** a failure-record count is never described as a number of attempts
or hypotheses.

Phase 15B is the canonical case. 60 ML meta-labeling experiments each have:

- 1 `INVALID_EXECUTION` attempt from a feature-pipeline defect (a31f571),
- 1 `VALID` attempt (corrected, 2e81b36) whose result is a typed refusal for
  insufficient pooled event density,
- 2 `STATISTICAL_INCONCLUSIVE` failure *records*.

The agent-facing digest therefore conveys **60 invalid execution attempts** and
**60 valid scientific refusals**, while `scientific_failure_records` still shows
the **120** underlying `STATISTICAL_INCONCLUSIVE` records. Proven end-to-end
against the committed `data/registry/experiments.sqlite` in
`test_real_schema_v5_phase_15b_history_conveys_60_and_60`.

## Provenance

`scripts/phase_16_research_agent_provenance.py` →
`outputs/phase_16/PHASE_16_RESEARCH_AGENT.json` (prompt version, prompt/schema
hashes, feature-catalog snapshot, defaults, guarantees). Deterministic, offline,
spends nothing.

## Tests

`tests/python/test_phase_16_research_agent.py` — 21 tests. Mostly
`ScriptedLLMClient` (no network): structured-output validation, fenced/prose
JSON extraction, retry-only-on-schema-failure, non-retryable guardrail
rejections, holdout guard on context and output, token budget, prompt logging
determinism, the attempts-vs-outcomes-vs-records split (including a proof
against the committed schema-v5 registry that Phase 15B conveys 60 invalid
attempts / 60 valid refusals / 120 records), and assertions that the agent
exposes no filesystem/registry/subprocess/order-routing surface. Two tests use a
mocked Anthropic SDK object to check `AnthropicClient` response normalization and
`messages.create` call shape — still no network or API spend.
