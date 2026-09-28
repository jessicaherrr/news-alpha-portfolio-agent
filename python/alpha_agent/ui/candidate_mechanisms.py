"""Candidate Mechanisms (Research Golden Path, section 4/4A) -- a compact,
provenance-tagged discovery surface built ONLY from existing engines:

* ``PROVENANCE_DETERMINISTIC`` -- a canonical trial from the frozen Phase
  13.5C candidate manifest (`alpha_agent.strategy.candidates_phase_13_5c`)
  that has no committed CANONICAL registry result for the selected root yet.
  This is a real, pre-declared research gap, never a fabricated suggestion.
* ``PROVENANCE_RESEARCH_MEMORY`` -- a candidate whose prior registry history
  (`FailureMemory.lookup`) carries an actual lesson worth surfacing before
  re-proposing the same mechanism.
* ``PROVENANCE_CLAUDE`` -- NOT precomputed here. The caller invokes the
  existing `llm_demo.propose_hypothesis` pipeline directly (Research Agent,
  Phase 16/17) and tags the result with this provenance; this module never
  calls an LLM itself.

These three sources are never merged into one undifferentiated list -- each
candidate always carries its own `provenance` field so the UI can render them
in clearly separated sections (section 4A: "Do not silently mix the
sources").
"""
from __future__ import annotations

from typing import Any

from alpha_agent.registry import TrialRole
from alpha_agent.strategy.candidates_phase_13_5c import build_candidate_manifest
from alpha_agent.ui import services

PROVENANCE_DETERMINISTIC = "DETERMINISTIC"
PROVENANCE_RESEARCH_MEMORY = "RESEARCH_MEMORY"
PROVENANCE_CLAUDE = "CLAUDE_PROPOSED"


def deterministic_candidates(root: str) -> list[dict[str, Any]]:
    """Every canonical trial in the frozen candidate manifest applicable to
    `root`, tagged TESTED (with its real verdict) or UNTESTED (a genuine,
    pre-declared research gap) by cross-referencing the live registry's
    CANONICAL trials for the same root/family -- never by re-deriving a
    verdict here."""
    manifest = build_candidate_manifest()
    canonical_rows = {
        (r["root_symbol"], r["strategy_family"]): r
        for r in services.list_experiments(root_symbol=root, trial_role=TrialRole.CANONICAL)
    }
    out: list[dict[str, Any]] = []
    for trial in manifest.trials:
        if trial.root_symbol != root:
            continue
        row = canonical_rows.get((root, trial.family_key))
        out.append(
            {
                "provenance": PROVENANCE_DETERMINISTIC,
                "family_key": trial.family_key,
                "strategy_name": services.strategy_name(trial.family_key),
                "root_symbol": root,
                "canonical_params": dict(trial.canonical_params),
                "strategy_fingerprint": trial.strategy_fingerprint,
                "tested": row is not None,
                "experiment_id": row["experiment_id"] if row else None,
                "verdict": row["verdict"] if row else None,
            }
        )
    return out


def research_memory_candidates(root: str, *, deterministic: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """For each UNTESTED deterministic candidate, a real `FailureMemory`
    lesson if one exists for this exact family/root neighbourhood (e.g. a
    related root's REJECT with an actionable note) -- omitted entirely when
    there is no real lesson to show, never padded with a generic caption."""
    deterministic = deterministic if deterministic is not None else deterministic_candidates(root)
    out: list[dict[str, Any]] = []
    for cand in deterministic:
        if cand["tested"]:
            continue
        fm = services.failure_memory_lookup(
            strategy_family=cand["family_key"], root_symbol=root, params=cand["canonical_params"],
        )
        if not fm.get("lessons"):
            continue
        out.append(
            {
                "provenance": PROVENANCE_RESEARCH_MEMORY,
                "family_key": cand["family_key"],
                "strategy_name": cand["strategy_name"],
                "root_symbol": root,
                "lesson": fm["lessons"][0],
                "prior_experiments": fm.get("prior_experiments", []),
            }
        )
    return out
