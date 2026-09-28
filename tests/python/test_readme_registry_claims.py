"""Documentation-accuracy regression: the headline registry numbers quoted in
README.md must always match the committed, frozen release snapshot -- and
that snapshot must in turn match the live registry whenever one is present.

This is a documentation guard, not a scientific test: it computes nothing new
and asserts nothing about registry semantics, PASS/REJECT/BH-FDR meaning, or
validation behavior. It exists because README.md's "Current honest results"
section once quoted an internally-inconsistent breakdown (61 canonical + 86
neighbour was presented as if it summed to the registry's true total of 167,
silently omitting a third role) before a documentation-only correction fixed
it; this test makes that class of drift fail loudly instead of silently.

Two layers, deliberately different skip behavior (see the release-packaging
task this was written for):

* ``test_readme_headline_registry_numbers_match_the_frozen_release_snapshot``
  -- ALWAYS runs, never skips. It compares README.md against
  ``outputs/release/FROZEN_REGISTRY_SUMMARY.json``, a small, committed,
  derived-metadata-only artifact -- so a clean public clone (which never
  contains the gitignored, operational ``data/registry/experiments.sqlite``)
  still mechanically verifies README's claims against something real and
  committed, with no network call, no Databento spend, and no 2025+ access.
* ``test_frozen_release_snapshot_matches_the_live_registry`` -- an optional,
  STRONGER local cross-check: if the operational registry sqlite happens to
  be present (a local research checkout), it re-derives the same breakdown
  live and confirms the frozen snapshot hasn't drifted from it. Skips
  cleanly when the sqlite is absent.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from alpha_agent.registry.enums import RegistryVerdict, TrialRole
from alpha_agent.registry.sqlite_registry import ExperimentRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data" / "registry" / "experiments.sqlite"
README = REPO_ROOT / "README.md"
FROZEN_SNAPSHOT = REPO_ROOT / "outputs" / "release" / "FROZEN_REGISTRY_SUMMARY.json"


def _live_breakdown() -> dict[str, int]:
    with ExperimentRegistry(REGISTRY_PATH) as reg:
        auth = reg.experiments(authoritative_only=True)
        by_role: dict[TrialRole, int] = dict.fromkeys(TrialRole, 0)
        for v in auth:
            by_role[v.experiment.trial_role] += 1
        canonical = [v for v in auth if v.experiment.trial_role is TrialRole.CANONICAL]
        verdicts: dict[str, int] = dict.fromkeys(
            (RegistryVerdict.PASS.value, RegistryVerdict.REJECT.value,
             RegistryVerdict.INCONCLUSIVE.value, "NOT_ADJUDICATED"), 0,
        )
        for v in canonical:
            key = v.verdict.value if v.verdict is not None else "NOT_ADJUDICATED"
            verdicts[key] = verdicts.get(key, 0) + 1
        failure_records = reg._conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
        return {
            "total": len(auth),
            "canonical": by_role[TrialRole.CANONICAL],
            "neighbour": by_role[TrialRole.NEIGHBOUR],
            "ablation": by_role[TrialRole.ABLATION],
            "variant": by_role[TrialRole.VARIANT],
            "canonical_pass": verdicts[RegistryVerdict.PASS.value],
            "canonical_reject": verdicts[RegistryVerdict.REJECT.value],
            "canonical_inconclusive": verdicts[RegistryVerdict.INCONCLUSIVE.value],
            "canonical_not_adjudicated": verdicts["NOT_ADJUDICATED"],
            "failure_records": int(failure_records),
        }


def _frozen_snapshot_breakdown() -> dict[str, int]:
    d = json.loads(FROZEN_SNAPSHOT.read_text(encoding="utf-8"))
    roles = d["trial_role_breakdown"]
    verdicts = d["canonical_headline_verdict_breakdown"]
    return {
        "total": d["authoritative_experiment_identities"],
        "canonical": roles["CANONICAL"],
        "neighbour": roles["NEIGHBOUR"],
        "ablation": roles["ABLATION"],
        "variant": roles["VARIANT"],
        "canonical_pass": verdicts["PASS"],
        "canonical_reject": verdicts["REJECT"],
        "canonical_inconclusive": verdicts["INCONCLUSIVE"],
        "canonical_not_adjudicated": verdicts["NOT_ADJUDICATED"],
        "failure_records": d["failure_records"],
    }


def _readme_figures(text: str) -> dict[str, int]:
    def one(pattern: str) -> int:
        m = re.search(pattern, text)
        assert m, f"README.md is missing the expected pattern: {pattern!r}"
        return int(m.group(1))

    return {
        "total": one(r"(\d+) total experiment identities"),
        "canonical": one(r"=\s*(\d+)\s+CANONICAL"),
        "neighbour": one(r"\+\s*(\d+)\s+NEIGHBOUR"),
        "ablation": one(r"\+\s*(\d+)\s+ABLATION"),
        "canonical_pass": one(r"\n\s*(\d+)\s+PASS\n"),
        "canonical_reject": one(r"\n\s*(\d+)\s+REJECT\n"),
        "canonical_inconclusive": one(r"\n\s*(\d+)\s+INCONCLUSIVE\n"),
        "canonical_not_adjudicated": one(r"\n\s*(\d+)\s+NOT_ADJUDICATED\s"),
        "failure_records": one(r"(\d+)\s+typed failure/rejection records"),
    }


def test_frozen_release_snapshot_reconciles_arithmetically():
    """Sanity check on the fixture every other test in this module relies on."""
    b = _frozen_snapshot_breakdown()
    assert b["canonical"] + b["neighbour"] + b["ablation"] + b["variant"] == b["total"]
    assert (
        b["canonical_pass"] + b["canonical_reject"] + b["canonical_inconclusive"]
        + b["canonical_not_adjudicated"]
    ) == b["canonical"]


def test_readme_headline_registry_numbers_match_the_frozen_release_snapshot():
    """ALWAYS runs -- never skips. A clean public clone never contains the
    gitignored, operational data/registry/experiments.sqlite, but it does
    contain outputs/release/FROZEN_REGISTRY_SUMMARY.json, so this is the
    check that actually protects every reader of a fresh clone, not just a
    local research checkout."""
    frozen = _frozen_snapshot_breakdown()
    quoted = _readme_figures(README.read_text(encoding="utf-8"))
    for key, frozen_value in frozen.items():
        if key not in quoted:
            continue
        assert quoted[key] == frozen_value, (
            f"README.md quotes {key}={quoted[key]} but the committed "
            f"outputs/release/FROZEN_REGISTRY_SUMMARY.json reports "
            f"{key}={frozen_value} -- correct README.md's 'Current honest "
            f"results' section (or regenerate the frozen snapshot if the "
            f"registry itself legitimately changed, which requires the "
            f"same evidence discipline as any other registry-affecting change)"
        )


def test_frozen_release_snapshot_matches_the_live_registry():
    """OPTIONAL, stronger local cross-check -- skips cleanly if the
    operational registry sqlite is absent (e.g. a clean public clone)."""
    if not REGISTRY_PATH.exists():
        pytest.skip("no local registry sqlite present -- clean-clone check above already ran")
    live = _live_breakdown()
    frozen = _frozen_snapshot_breakdown()
    assert live == frozen, (
        "the live registry has drifted from outputs/release/FROZEN_REGISTRY_SUMMARY.json -- "
        "regenerate the frozen snapshot (see its own 'generation_method' field) if this "
        "reflects a legitimate, evidence-backed registry change; otherwise investigate "
        "why the operational registry no longer matches its own committed release snapshot"
    )
