"""Phase 21.1 -- exact per-step execution provenance.

CLAUDE.md: "every backtest records dataset version/hash, strategy hash, code
commit if available, parameters, and cost assumptions." A paper-trading step
is a backtest replay and needs the same discipline, PER STEP: an independent
auditor must be able to verify EXACTLY which bytes were sent to the C++
boundary and which C++ build produced the result, without trusting a
descriptive label such as ``"real_2018_2024_replay"``.

:class:`PaperStepProvenance` is immutable, versioned, and never enters the
Phase 14 experiment_identity formula -- it is attempt/step provenance for an
ALREADY-approved parent experiment
(``parent_experiment_identity``), exactly the same relationship
``target_schedule_hash`` / ``report_fingerprint`` have to a registry
``ExperimentRecord``: recorded, hashed, never used to define which hypothesis
this is.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

from alpha_agent.validation.fingerprint import fingerprint


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """SHA-256 of the exact bytes on disk at ``path`` -- the literal file
    handed to (or produced by) the C++ boundary, not a re-serialization."""
    return sha256_bytes(Path(path).read_bytes())


class PaperStepProvenance(BaseModel):
    """Exact provenance for ONE paper-trading step's C++ invocation.

    Every field here is either copied from the run's own immutable
    configuration (set once at ``start_run``, CLAUDE.md risk rule 1) or is a
    SHA-256 of the literal bytes actually written to / read from disk for
    this specific step's C++ call. Nothing here is a free-form label.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "paper-step-provenance/1"

    #: the registry experiment this paper run was started from -- preserved,
    #: never redefined; this is NOT a new Phase 14 experiment_identity.
    parent_experiment_identity: str
    #: the committed StrategySpec fingerprint verified at eligibility time.
    strategy_fingerprint: str
    #: PaperRiskPolicy.identity() -- the hard-risk configuration this step ran under.
    risk_policy_identity: str

    #: execution / cost configuration actually passed to the CLI this step.
    schedule_policy: str
    commission_per_contract_usd: float
    slippage_ticks: float
    spread_ticks: float
    end_of_test: str

    #: SHA-256 of the exact bars.csv / contracts.csv / targets.csv /
    #: validation_days.csv bytes actually submitted to the C++ boundary.
    bars_sha256: str
    contracts_sha256: str
    targets_sha256: str
    validation_days_sha256: str
    #: canonical semantic identities of the same two artifacts (additional,
    #: reproducible without a byte-exact file -- TargetSchedule.schedule_hash()
    #: / ValidationDayPlan.identity(), both pre-existing frozen fingerprints).
    target_schedule_hash: str
    validation_day_plan_identity: str
    #: SHA-256 of roll_close_marks.csv, when the step actually used one; None
    #: otherwise (Phase 21.1 does not yet wire roll continuation into paper
    #: trading -- see docs/PAPER_TRADING_AND_DRIFT_MONITOR.md limitations).
    roll_close_marks_sha256: str | None = None

    #: identifies the exact C++ build that produced this step's result.
    executable_path: str
    executable_sha256: str

    #: SHA-256 of the raw stdout bytes the CLI process actually emitted.
    result_payload_sha256: str

    def identity(self) -> str:
        """A single fingerprint over every field above: changes if ANY
        canonical step input changes, stable for identical inputs."""
        return fingerprint("paperstepprov1", self.model_dump(mode="json"))
