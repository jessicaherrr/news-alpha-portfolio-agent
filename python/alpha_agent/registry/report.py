"""``REGISTRY_REPORT.md`` -- the researcher-facing explanation of the registry.

Written to be legible to a quant researcher, an MFE admissions reviewer and a
software engineer at once: what the registry is, how identity and immutability
work, what the current authoritative research result is, and the two
engineering lessons that shaped it. Nothing here oversells the system -- the
registry is deterministic bookkeeping, not intelligence.
"""
from __future__ import annotations

from alpha_agent.registry.enums import FailureScope, TrialRole
from alpha_agent.registry.exports import display_path
from alpha_agent.registry.similarity import WEIGHTS
from alpha_agent.registry.sqlite_registry import ExperimentRegistry


def _table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(out)


def render_registry_report(registry: ExperimentRegistry) -> str:
    s = registry.summary()
    canonical = registry.experiments(trial_role=TrialRole.CANONICAL)
    imports = s.imports[0] if s.imports else {"metadata": {}}
    meta = imports.get("metadata", {})

    verdict_rows = [
        [
            v.experiment.root_symbol,
            v.experiment.strategy_family,
            v.verdict.value if v.verdict else "-",
            ";".join(v.result.reason_codes) if v.result else "-",
        ]
        for v in sorted(canonical, key=lambda x: (x.experiment.strategy_family,
                                                  x.experiment.root_symbol))
    ]

    system_failures = [
        f for f in registry.failures() if f.scope is FailureScope.SYSTEM
    ]

    lines: list[str] = []
    add = lines.append

    add("# Phase 14 -- Experiment Registry and Failure Memory")
    add("")
    add("## 1. What this is")
    add("")
    add(
        "A persistent, deterministic, offline record of every research attempt this "
        "platform has made: what was tested, on what market, under which configuration, "
        "what happened, and why. It exists so that a future Research Agent can ask "
        "*\"has this already been tried, and what did we learn?\"* **before** spending "
        "compute -- and so that a failed experiment is kept as evidence rather than "
        "quietly forgotten."
    )
    add("")
    add(
        "It is bookkeeping, not intelligence. No language model decides what is a "
        "duplicate, which result is authoritative, whether an experiment failed, or "
        "what its reason codes are. An LLM may *read* this registry later; it can "
        "never define registry truth."
    )
    add("")
    add(f"- registry database: `{display_path(s.registry_path)}`")
    add(f"- schema version: `{s.schema_version}` "
        f"(experiment identity schema `{s.identity_schema}`)")
    add(f"- content digest: `{s.content_digest}`")
    add("")

    add("## 2. Experiment identity")
    add("")
    add(
        "Three identifiers are kept strictly apart, because conflating them is how "
        "registries silently double-count:"
    )
    add("")
    add(
        "| layer | example | role |\n|---|---|---|\n"
        "| `experiment_identity` | `experiment1:<sha256>` | the scientific identity; "
        "the only thing duplicate detection uses |\n"
        "| `experiment_id` | `NQ__TSMOM__CANONICAL__VALIDATION_2023_2024` | a "
        "deterministic human-readable label |\n"
        "| `row_id` | `42` | SQLite storage only |"
    )
    add("")
    add(
        "The identity is a SHA-256 over canonical JSON of the *meaningful, PRE-RUN* "
        "inputs only: strategy fingerprint, family, root, parameter variant, dataset, "
        "split, `ValidationSpec`, `ReliabilityPolicy`, execution config, cost config, "
        "risk identity and feature set. It deliberately excludes wall-clock time, "
        "random UUIDs, the code commit, the display name and the trial role label."
    )
    add("")
    add(
        "One consequence is worth stating explicitly. The execution-config component "
        "carries the derived contract economics of **every root in the multiple-testing "
        "family**, not just this experiment's root. BH q-values and the DSR benchmark "
        "are computed jointly over all 107 trials, so a mis-derived point value on one "
        "root changes the global statistics of every trial. Including family-wide "
        "economics in the identity is what makes the pre-correction and corrected runs "
        "genuinely *different experiments* -- which is why the registry can keep both "
        "instead of overwriting one with the other."
    )
    add("")

    add(
        "Identity is deliberately computable **before** a backtest runs, so a future "
        "agent can ask \"has this already been tried?\" without compiling a schedule. "
        "Two fields are therefore provenance, not identity:"
    )
    add("")
    add(
        "| field | meaning |\n|---|---|\n"
        "| `experiment_identity` | pre-run semantic identity |\n"
        "| `target_schedule_hash` | post-compilation audit evidence |\n"
        "| `report_fingerprint` | result-artifact provenance |"
    )
    add("")
    add(
        "Both provenance fields still enter the *content* fingerprint, so the same "
        "identity reported with a different compiled schedule fails loudly as an "
        "integrity condition (`ScheduleProvenanceConflict`) rather than quietly "
        "becoming a second experiment that evades duplicate detection. And neither is "
        "ever fabricated: where no committed artifact records a value, it is `NULL` and "
        "the row's `evidence_completeness` says so."
    )
    add("")
    add("## 3. Append-only, and what happens on a conflict")
    add("")
    add(
        "There is no code path that updates a completed experiment's scientific result. "
        "Inserting the same `experiment_identity` with byte-equivalent content returns "
        "the existing row (idempotent). Inserting it with *conflicting* scientific "
        "content raises `ExperimentConflict` and the transaction rolls back -- the "
        "registry refuses to guess which version is true. Correcting a result means "
        "recording a **new** experiment plus an explicit `SUPERSEDES` / `CORRECTS` edge."
    )
    add("")
    add(
        "\"Is this the current answer?\" is therefore never a stored flag that could "
        "drift: it is derived from the lineage graph at query time. An experiment is "
        "`SUPERSEDED` exactly when some other experiment has a `SUPERSEDES` or "
        "`CORRECTS` edge pointing at it."
    )
    add("")
    add("```")
    add("registry.get(id)                          -> the row as recorded")
    add("registry.find_exact_duplicate(identity)   -> has this exact run happened?")
    add("registry.resolve_authoritative(identity)  -> follow corrections to today's answer")
    add("registry.experiments(include_superseded=True) -> historical + corrected")
    add("```")
    add("")

    add("## 4. Near-duplicate retrieval (deterministic, no embeddings)")
    add("")
    add(
        "A proposal of *TSMOM fast=21 slow=120 on NQ* should surface the existing "
        "*TSMOM fast=20 slow=120 on NQ* and its evidence. The score is a fixed weighted "
        "sum of transparent components, and every hit carries the reasons that produced "
        "it:"
    )
    add("")
    add(_table(
        [[k, f"{v:.2f}"] for k, v in sorted(WEIGHTS.items())],
        ["component", "weight"],
    ))
    add("")
    add(
        "Parameter distance is normalised by the **declared** `param_grid_ranges` -- the "
        "ranges frozen before any result was inspected -- so \"close\" means close inside "
        "the predeclared neighbourhood, not close in raw units. This is memory and "
        "warning infrastructure: it never changes a `ReliabilityPolicy` and never "
        "rejects a hypothesis on its own. What to do with the evidence is future agent "
        "policy."
    )
    add("")

    add("## 5. Failure taxonomy")
    add("")
    add(
        "Phase 13's typed `ReasonCode`s stay authoritative. Phase 14 classifies them "
        "into a closed `FailureClass` set so failures are queryable by kind, and adds "
        "the classes that describe engineering and procedural defects. Free text "
        "supplements a typed code; it is never the identity of a failure."
    )
    add("")
    add(_table(
        [[k, str(v)] for k, v in sorted(s.failure_class_counts.items())],
        ["failure_class", "records"],
    ))
    add("")

    add("## 6. Current authoritative result (Phase 13.5C)")
    add("")
    add(
        f"- unique statistical hypotheses: **{s.authoritative_statistical_hypotheses}** "
        f"({s.canonical} canonical + {s.neighbour} predeclared parameter neighbours)"
    )
    add(f"- completed C++ executions: **{s.completed_cpp_executions}** (compute, not hypotheses)")
    add(
        f"- data-quality sensitivity reruns: **{s.sensitivity_evidence}** "
        "(robustness evidence, NOT in the BH/FDR denominator)"
    )
    add(
        f"- cross-market summaries: **{s.cross_market_evidence}** "
        "(descriptive aggregation of existing per-root trials, NOT experiments)"
    )
    add(
        "- canonical headline verdicts: "
        + ", ".join(f"**{k} = {v}**" for k, v in sorted(s.canonical_verdict_counts.items()))
    )
    add(f"- candidates eligible for the locked holdout: **{s.holdout_eligible}**")
    add(f"- superseded experiment rows retained: **{s.superseded_experiments}**")
    add("")
    add(
        "These counts are deliberately kept distinct. \"How many experiments?\" is not a "
        "single number: 107 statistical hypotheses, 528 C++ executions, 21 canonical "
        "headline trials, 86 neighbours and 21 sensitivity reruns each answer a "
        "different question."
    )
    add("")
    add(f"- authoritative research commit: `{meta.get('corrected_commit', '')}`")
    add(f"- superseded checkpoint (retained): `{meta.get('superseded_commit', '')}`")
    add(f"- frozen candidate manifest: `{meta.get('candidate_manifest_fingerprint', '')}`")
    add(f"- BH/FDR family fingerprint: `{meta.get('bh_fdr_family_fingerprint', '')}`")
    add(
        f"- locked holdout 2025: {meta.get('locked_holdout_2025', 'never accessed')}"
    )
    add("")
    add("### Canonical headline verdicts")
    add("")
    add(_table(verdict_rows, ["root", "family", "verdict", "reason codes"]))
    add("")

    add("## 7. Engineering lessons the registry remembers")
    add("")
    for f in system_failures:
        add(f"### {f.failure_class.value} -- `{f.failure_code}`")
        add("")
        add(f.summary)
        add("")
        add(f"**Mechanism.** {f.mechanism}")
        add("")
        if f.action_taken:
            add(f"**Correction.** {f.action_taken}")
            add("")
        if f.resolution_commit:
            add(f"**Resolved in.** `{f.resolution_commit}`")
            add("")

    add("## 8. How a future Research Agent uses this")
    add("")
    add("```")
    add("HypothesisSpec")
    add("     |")
    add("     v")
    add("failure_memory.lookup(strategy_family=..., root_symbol=..., strategy_spec=...)")
    add("     |")
    add("     +-- exact_duplicate.exists      -> do not re-run; cite the prior result")
    add("     +-- related_experiments[...]    -> prior nearby evidence + reasons")
    add("     +-- engineering_failures[...]   -> data/execution traps already paid for")
    add("     |")
    add("     v")
    add("(agent policy decides)  ->  Compiler Agent  ->  C++ quant core  ->  validation")
    add("     |")
    add("     v")
    add("registry.insert_experiment(...)  (append-only; never a silent rerun)")
    add("```")
    add("")
    add(
        "The response is a typed `FailureMemoryResponse`: an agent consumes it "
        "directly and never parses Markdown or re-reads a Phase 13 report."
    )
    add("")

    add("## 9. Limits")
    add("")
    add(
        "- Similarity is structural, not semantic. It compares families, roots, "
        "structural shape, feature sets and normalised parameter distance. It does not "
        "understand economic mechanism, and it is not presented as if it does."
    )
    add(
        "- The 86 predeclared neighbours are recorded with their trial statistics "
        "(p-value, BH q-value, rejection at q). Per-neighbour PnL and Sharpe were "
        "summarised into each canonical trial's parameter-stability block and not "
        "retained per trial in a committed artifact; those rows say so in "
        "`evidence_completeness`."
    )
    add(
        "- The pre-correction lineage is imported at canonical granularity (21 rows). "
        "The retained correction artifact records the superseded verdict, net PnL, BH "
        "q-value and DSR per canonical trial; pre-correction reason codes and the 86 "
        "pre-correction neighbour statistics were not retained."
    )
    add(
        "- The registry stores no market data and no 2025 timestamp of any kind. Any "
        "market-data or performance value at or after 2025-01-01 is refused at the "
        "write boundary."
    )
    add("")
    return "\n".join(lines) + "\n"
