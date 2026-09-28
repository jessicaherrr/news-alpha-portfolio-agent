# Research Orchestrator (Phase 18; 18.1 predeclares the family; 18.2 closes cross-generation adaptive testing)

`alpha_agent.agents.orchestrator.ResearchOrchestrator` is the controlled loop
that **sequences** the deterministic research services built in Phases 13–17.
It is a plain state machine. It owns no science.

## Why 18.1

Phase 18 (`7b4abe3`, kept as the initial implementation) executed a hypothesis,
**reflected on its outcome**, then generated the next member of the *same*
BH/FDR family. Family membership was therefore outcome-adaptive — a
multiple-testing violation that fixing `predeclared_family_size` alone does not
address.

18.1 makes one multiple-testing family **fully predeclared before any member
executes**, in three strict phases plus post-finalization reflection.

```
PLAN  (executes nothing)
  planning context  (a pure function of FINALIZED history + inter-generation feedback)
    -> ResearchAgent          (Phase 16)  HypothesisSpec
    -> StrategyCompilerAgent  (Phase 17)  StrategySpec  (frozen Phase 10 compiler is the gate)
    -> complete PRE-RUN experiment_identity  (Phase 14 formula)
    -> schema-v5 find_exact_duplicate + attempt authority; near-duplicate gate
  ... repeat for every predeclared member ...
  -> freeze an immutable FamilyManifest   (family_id = content hash)

EXECUTE  (only after the manifest is frozen; no reflection, no new members)
  for each frozen member -> ExecutionValidationService.run -> TrialEvidence  (raw)
  an INVALID_EXECUTION writes an ExecutionAttemptRecord + a typed FailureRecord
  and NOTHING scientific.

FINALIZE  (only after the frozen family is complete)
  family-wide BH/FDR over every member's trial p-value
  -> final per-member reliability verdict
  -> ONE atomic ImportBundle: Experiment + ExecutionAttempt + Result + Failure records
  A family with any unresolved INVALID_EXECUTION member is
  INCOMPLETE_NOT_ADJUDICATED: no BH, no verdicts, no PASS, and the BH
  denominator is never silently shrunk.

REFLECT  (only after a family is finalized)
  may propose a NEW family / generation with its own predeclared manifest and a
  new family_id. It never mutates a finalized family.
```

`plan_family` → `execute_family` → `finalize_family` is the unit of work.
`run()` plans, executes and finalizes **exactly one** outcome-adaptive family on
this evaluation plane, then (optionally) *plans* one successor family it does not
execute.

## Why 18.2

18.1's `run()` could finalize a BH family, reflect, and then execute another
newly-generated family on the **same** dataset / split / validation plane with a
fresh BH q-budget — automated multiple-testing inflation across generations.
18.2 adds three integrity guarantees:

### 1. One evaluation plane, one outcome-adaptive campaign

`evaluation_plane_id(planes)` hashes the tuple
`(dataset_fingerprint, split_identity, validation_spec_fingerprint,
reliability_policy_fingerprint)`. `execute_family` refuses a second family on a
plane that already carries an executed research family — checked against both
this run **and** the registry (`_planes_already_used`), raising
`AdaptiveTestingError`. Post-finalization reflection may `plan_family` a
successor, but `run()` returns it as `FamilyStatus.PLANNED_NOT_EXECUTED`. A
successor can be scientifically executed only under a genuinely new predeclared
evaluation plane or a separately frozen sequential-testing procedure (neither
exists in the MVP — we do **not** invent online FDR / alpha-investing).

### 2. Adjudication is the frozen policy's

The orchestrator is constructed with the actual frozen
`ReliabilityPolicy` (or an explicit `FamilyAdjudicationService`) and asserts
`adjudicator.policy_identity() == planes.reliability_policy_fingerprint`. The BH
q-threshold is read from that policy — `FamilyPlanSpec` no longer carries an
independent `fdr_q_threshold`. `PolicyFamilyAdjudicator` reuses the shared
`benjamini_hochberg_decisions` and reads only two declared policy fields
(`fdr_q_threshold`, `fdr_require_canonical_rejected`); it forks no other Phase-13
gate — every non-family gate arrives as a `NonFamilyVerdict` from the execution
service. `FamilyAdjudication` records `policy_identity` and `fdr_q_threshold`;
`TrialEvidence.produced_under_*` fingerprints, when set, are asserted against the
manifest's planes (`EvidenceProvenanceError`).

### 3. Manifest immutability is verified, not assumed

A frozen pydantic model still has nested mutable dicts (`FamilyMember.params`,
`strategy_spec_json`, `StrategySpec.metadata`). `plan_family` stores an
**independent** canonical snapshot — strategy fingerprint, feature-set
fingerprint, `parameter_variant_identity`, full `experiment_identity`, and a full
`StrategySpec` dump — each **re-derived** from the actual executable member.
`_verify_manifest` re-derives it fresh from the passed manifest before every
execution / finalization and compares fingerprints; any drift (a mutated param,
a mutated spec, a mutated `strategy_spec_json`) raises `FamilyManifestError`. It
never compares two references to the same mutable object.

`FamilyPlanSpec.target_family_size` is the **scientific** family size: if
planning cannot fill it the family is `FamilyStatus.PLANNING_INCOMPLETE` and is
never executed as a smaller BH family.

## Statistical-integrity invariants (enforced by construction)

* **The whole family is predeclared.** `FamilyManifest` is a frozen pydantic
  model whose `family_id` is `fingerprint(stem, generation, parent, planes,
  market_window, fdr_q, [member identities in order])`. Adding, removing, or
  altering a member changes `family_id`; `assert_consistent()` (called at every
  execution/finalization boundary) catches a `model_copy`-tampered manifest, and
  the orchestrator refuses a manifest it did not plan.
* **No within-family outcome adaptivity.** `_planning_context()` is a pure
  function of *finalized* families plus inter-generation feedback. It is built
  once per family and is byte-identical for every planning slot — it can carry no
  per-member p-value, verdict, or metric because none exist yet.
* **BH and final verdicts only at finalization.** `execute_family` writes zero
  `ResultRecord`s and zero BH q-values. `finalize_family` runs
  `benjamini_hochberg_decisions` over the complete family, then writes every
  member's result in one `apply_bundle` transaction.
* **No partial family → PASS.** An unresolved `INVALID_EXECUTION` member makes
  the whole family `INCOMPLETE_NOT_ADJUDICATED`; not one `ResultRecord` is
  written, not even for a would-be-PASS sibling.
* **INVALID_EXECUTION ≠ scientific result.** An invalid attempt is an
  `ExecutionAttemptRecord` (status `INVALID_EXECUTION`, typed `InvalidationClass`)
  + an `INVALID_EXECUTION_ATTEMPT` `FailureRecord`, and never a `ResultRecord`. A
  corrected VALID re-execution of the same `experiment_identity` (bounded by
  `max_execution_attempts_per_identity`) may later create the authoritative
  result on attempt 2, with an auto `CORRECTS_ATTEMPT` edge.
* **Strategy fingerprint is evidence only** (Phase 17.2). A member is admitted or
  refused on the full pre-run `experiment_identity` + schema-v5 attempt
  authority. The same strategy under a different execution/validation plane is a
  distinct experiment.
* **The LLM cannot set or see** PnL / fills / risk / costs / validation
  thresholds / the BH-FDR family / the DSR benchmark. Those are frozen
  `IdentityPlanes` / `FamilyPlanSpec`, or the injected service's concern.
* **No 2025.** `OrchestratorConfig`, `TrialEvidence`, and the holdout output are
  all `assert_no_holdout_market_data`-guarded. `run_holdout` needs an explicitly
  passed `HoldoutEvaluationService`, the candidate's `StrategySpec`, and
  `allow_holdout=True` — arguments the iteration loop never has.

## The service boundary

`ExecutionValidationService.run(member, manifest, attempt_ordinal, holdout) ->
TrialEvidence` returns **raw** per-trial evidence:

* `status = VALID` with a `trial_p_value` (the one-sided p-value that enters the
  BH family), a `non_family_verdict ∈ {ELIGIBLE, REJECT, INCONCLUSIVE}` (the
  frozen `ReliabilityPolicy`'s verdict *ignoring* the family BH gate), raw
  `metrics`, and provenance; **or**
* `status = INVALID_EXECUTION` with a typed `InvalidationClass`.

The orchestrator overlays **only** the family layer at finalization:
`ELIGIBLE` + canonical + BH-rejected → `PASS`; `ELIGIBLE` + canonical + not
BH-rejected → `REJECT (FDR_NOT_SIGNIFICANT)`; `REJECT`/`INCONCLUSIVE` pass
through; a non-canonical member records its bh_q as `NOT_ADJUDICATED`.

The BH q-threshold comes from the bound frozen `ReliabilityPolicy` (18.2), not
from `FamilyPlanSpec`. The manifest records `fdr_q_threshold` and
`policy_identity` (`== planes.reliability_policy_fingerprint`) for transparency.
The real C++ Quant Core + validation engine wiring behind
`ExecutionValidationService` remains deferred; `ScriptedExecutionValidationService`
is the deterministic test double. `FamilyAdjudicationService` (default
`PolicyFamilyAdjudicator`) is likewise pluggable but bound to the same frozen
policy identity.

## Budgets and termination

`OrchestratorBudget`: `max_planning_attempts_per_family`,
`max_execution_attempts_per_identity`, `agent_schema_retries`,
`plan_successor_family_after_finalization`.
`OrchestrationReport.termination_reason` is one of `family:finalized`,
`family:incomplete_not_adjudicated`, `planning:no_admissible_members`,
`planning:underfilled_family`,
`reflection:successor_family_planned_not_executed`.

`FamilyStatus`: `FINALIZED`, `INCOMPLETE_NOT_ADJUDICATED`, `EMPTY`,
`PLANNING_INCOMPLETE`, `PLANNED_NOT_EXECUTED`.

## Provenance

`scripts/phase_18_research_orchestrator_provenance.py` →
`outputs/phase_18/PHASE_18_RESEARCH_ORCHESTRATOR.json` (module SHA-256, the two
agent prompt versions, the default frozen `ReliabilityPolicy` identity, the
status/disposition vocab, and the 12 enforced invariants). Tests:
`tests/python/test_phase_18_research_orchestrator.py` (mocked LLM + scripted
execution, fully offline) — the eight 18.1 proofs plus the 18.2
adaptive-testing, policy-binding, manifest-immutability and underfill proofs.
