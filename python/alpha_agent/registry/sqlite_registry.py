"""The Phase 14 experiment registry: append-oriented, typed, offline SQLite.

Design rules this module enforces mechanically, not by convention:

* **append-only** -- there is no code path that updates a completed experiment's
  scientific result. Re-running the same hypothesis under a corrected
  configuration produces a NEW experiment plus an explicit ``SUPERSEDES`` /
  ``CORRECTS`` edge (section 7).
* **idempotent** -- inserting a byte-equivalent record twice returns the
  existing row. Inserting a *conflicting* record under the same
  ``experiment_identity`` raises :class:`ExperimentConflict` (section 7).
* **derived authority** -- "is this the current answer?" is computed from the
  lineage graph at query time, so marking something superseded never rewrites
  the superseded row (section 12).
* **deterministic** -- no LLM, no network, no clock-dependent identity
  (sections 23, 25).
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

from alpha_agent.registry.enums import (
    AssetDomain,
    AttemptRelation,
    AttemptStatus,
    Authority,
    ExperimentStatus,
    FailureClass,
    RegistryVerdict,
    RelationType,
    TrialRole,
)
from alpha_agent.registry.identity import IDENTITY_SCHEMA
from alpha_agent.registry.models import (
    AttemptLineageEdge,
    CrossMarketEvidenceRecord,
    ExecutionAttemptRecord,
    ExperimentRecord,
    FailureRecord,
    ImportBundle,
    LineageEdge,
    MarketWindow,
    ResultRecord,
    SensitivityEvidence,
    SignalPathEvidenceRecord,
)
from alpha_agent.registry.schema import SCHEMA_VERSION, ensure_schema, read_schema_version
from alpha_agent.registry.similarity import (
    SimilarityBreakdown,
    rank_key,
    similarity,
    structural_shape,
)
from alpha_agent.validation.fingerprint import fingerprint

DEFAULT_REGISTRY_PATH = Path("data/registry/experiments.sqlite")


class RegistryError(RuntimeError):
    """Base class for registry integrity failures."""


class ExperimentConflict(RegistryError):
    """The same ``experiment_identity`` was offered twice with *different*
    scientific content. The registry refuses to guess which one is true."""


class ScheduleProvenanceConflict(ExperimentConflict):
    """Two runs share every pre-run semantic input but report different compiled
    ``target_schedule_hash`` values.

    Since identity schema v2 the schedule hash is provenance, not identity, so
    this cannot silently become a "new" experiment that evades duplicate
    detection. It is an integrity condition to investigate: identical inputs
    should compile to an identical schedule.
    """


class IdentitySchemaMismatch(RegistryError):
    """A record was built under a different experiment-identity schema than this
    build computes. Storing it would mix two incompatible notions of "the same
    experiment"."""


class ImmutableResultError(RegistryError):
    """An attempt to change the recorded scientific result of a completed
    execution attempt. A completed attempt is immutable; record a NEW attempt
    (schema v5) -- never a new ``experiment_identity``."""


class AttemptConflict(RegistryError):
    """The same execution attempt was offered twice with different content, or
    an ordinal / status invariant was violated."""


class UnknownExperiment(RegistryError):
    """No experiment matches the given identity / friendly id."""


class AssetDomainMismatch(RegistryError):
    """A query scoped to one :class:`AssetDomain` found an ``experiment_identity``
    that exists, but under a DIFFERENT domain.

    ``experiment_identity`` is a SHA-256 over structurally domain-distinct
    inputs (dataset fingerprint, execution-config identity, ...), so this
    should be cryptographically impossible in practice. It is checked anyway,
    as defense in depth (Phase 6 ETF Research Pilot): a same-identity hit
    under the wrong domain is a genuine integrity condition to investigate,
    never something to paper over by silently returning "not found".
    """


# --------------------------------------------------------------------------
# query result models
# --------------------------------------------------------------------------
class AttemptView(BaseModel):
    """One execution attempt of an experiment plus its result (if any)."""

    model_config = {"frozen": True, "extra": "forbid"}

    attempt_id: str
    experiment_identity: str
    attempt_ordinal: int
    attempt_status: AttemptStatus
    invalidation_class: str | None = None
    invalidation_detail: str = ""
    defect_resolution_commit: str | None = None
    code_commit: str = ""
    engine: str = ""
    report_fingerprint: str | None = None
    target_schedule_hash: str | None = None
    created_at: str = ""
    notes: str = ""
    result: ResultRecord | None = None

    @property
    def is_valid(self) -> bool:
        return self.attempt_status is AttemptStatus.VALID


class ExperimentView(BaseModel):
    """One experiment plus its AUTHORITATIVE result (the highest-ordinal VALID
    execution attempt's result) and derived identity-level authority."""

    model_config = {"frozen": True, "extra": "forbid"}

    experiment: ExperimentRecord
    result: ResultRecord | None = None
    authority: Authority = Authority.AUTHORITATIVE
    superseded_by: tuple[str, ...] = ()
    attempts: tuple[AttemptView, ...] = ()
    authoritative_attempt_id: str | None = None

    @property
    def experiment_id(self) -> str:
        return self.experiment.experiment_id

    @property
    def experiment_identity(self) -> str:
        return self.experiment.experiment_identity

    @property
    def verdict(self) -> RegistryVerdict | None:
        return self.result.headline_verdict if self.result else None

    @property
    def has_valid_authoritative_result(self) -> bool:
        return self.result is not None

    @property
    def n_valid_attempts(self) -> int:
        return sum(1 for a in self.attempts if a.is_valid)

    @property
    def n_invalid_attempts(self) -> int:
        return sum(1 for a in self.attempts if not a.is_valid)


class ExactDuplicate(BaseModel):
    """The answer to "has this exact experiment already been run?"."""

    model_config = {"frozen": True, "extra": "forbid"}

    exists: bool
    experiment_identity: str
    experiment_id: str | None = None
    display_name: str | None = None
    phase: str | None = None
    headline_verdict: RegistryVerdict | None = None
    reason_codes: tuple[str, ...] = ()
    report_fingerprint: str | None = None
    authority: Authority | None = None
    superseded_by: tuple[str, ...] = ()
    authoritative_replacement: str | None = None
    #: schema v5 -- does a VALID execution attempt supply an authoritative result?
    #: ``exists and not has_valid_authoritative_result`` == "known hypothesis with
    #: only invalid attempts" -> re-execution is allowed, not a duplicate.
    has_valid_authoritative_result: bool = False
    n_attempts: int = 0
    n_valid_attempts: int = 0
    n_invalid_attempts: int = 0
    latest_attempt_status: AttemptStatus | None = None

    @property
    def blocks_reexecution(self) -> bool:
        """A prior run blocks an accidental rerun only if it left a VALID
        authoritative result. Invalid-only history permits re-execution."""
        return self.exists and self.has_valid_authoritative_result


class RelatedExperiment(BaseModel):
    """One near-duplicate hit with its transparent similarity explanation."""

    model_config = {"frozen": True, "extra": "forbid"}

    experiment_id: str
    experiment_identity: str
    display_name: str
    strategy_family: str
    root_symbol: str
    trial_role: TrialRole
    params: dict = Field(default_factory=dict)
    headline_verdict: RegistryVerdict | None = None
    reason_codes: tuple[str, ...] = ()
    authority: Authority = Authority.AUTHORITATIVE
    similarity: SimilarityBreakdown


class RegistrySummary(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: int
    identity_schema: str = IDENTITY_SCHEMA
    registry_path: str
    authoritative_statistical_hypotheses: int
    canonical: int
    neighbour: int
    #: TrialRole.ABLATION / .VARIANT authoritative experiment counts.
    #: canonical + neighbour + ablation + variant == authoritative_statistical_hypotheses
    #: always, by construction (every TrialRole value is covered) -- added
    #: because a summary that reports canonical+neighbour alone silently
    #: implies they exhaust the total when a third/fourth role is present
    #: (see docs/FINAL_SYSTEM_AUDIT.md and the release-packaging correction
    #: that found README repeating this same incomplete partition).
    ablation: int = 0
    variant: int = 0
    superseded_experiments: int
    total_experiment_rows: int
    canonical_verdict_counts: dict[str, int] = Field(default_factory=dict)
    canonical_without_valid_attempt: int = 0
    execution_attempts: int = 0
    valid_execution_attempts: int = 0
    invalid_execution_attempts: int = 0
    holdout_eligible: int = 0
    failure_records: int = 0
    failure_class_counts: dict[str, int] = Field(default_factory=dict)
    lineage_edges: int = 0
    sensitivity_evidence: int = 0
    cross_market_evidence: int = 0
    completed_cpp_executions: int = 0
    imports: tuple[dict, ...] = ()
    content_digest: str = ""


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class ExperimentRegistry:
    """Persistent, offline experiment registry + failure memory."""

    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self.schema_version = ensure_schema(self._conn)
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def on_disk_schema_version(self) -> int | None:
        return read_schema_version(self._conn)

    # ------------------------------------------------------------------
    # writes -- append-only
    # ------------------------------------------------------------------
    def insert_experiment(
        self,
        record: ExperimentRecord,
        result: ResultRecord | None = None,
        *,
        attempt: ExecutionAttemptRecord | None = None,
    ) -> ExperimentView:
        """Append one experiment (idempotently) inside its own transaction.

        Schema v5: a ``result`` (with no explicit ``attempt``) is recorded on a
        fresh ``VALID`` execution attempt of ``record.experiment_identity``.
        """
        with self._conn:
            view = self._insert_experiment(record, result, attempt=attempt)
        return view

    def _insert_experiment(
        self,
        record: ExperimentRecord,
        result: ResultRecord | None,
        *,
        attempt: ExecutionAttemptRecord | None = None,
    ) -> ExperimentView:
        record.assert_no_holdout()
        if record.identity_schema != IDENTITY_SCHEMA:
            raise IdentitySchemaMismatch(
                f"experiment {record.experiment_id} was built under identity schema "
                f"{record.identity_schema!r}, but this build computes "
                f"{IDENTITY_SCHEMA!r}; rebuild the record rather than mixing schemas"
            )
        if result is not None:
            if result.experiment_identity != record.experiment_identity:
                raise RegistryError(
                    "result.experiment_identity does not match the experiment record"
                )
            result.assert_no_holdout()

        existing = self._conn.execute(
            "SELECT content_fingerprint, experiment_id, target_schedule_hash, record_json "
            "FROM experiments WHERE experiment_identity = ?",
            (record.experiment_identity,),
        ).fetchone()

        if existing is not None:
            # v5: the experiment row is the PRE-RUN hypothesis.
            stored = ExperimentRecord.model_validate(json.loads(existing["record_json"]))
            if stored.hypothesis_fingerprint() != record.hypothesis_fingerprint():
                # a genuinely different hypothesis claiming the same identity
                self._raise_content_conflict(record, existing)
            elif (
                attempt is None
                and stored.content_fingerprint() != record.content_fingerprint()
                and self._authoritative_attempt_row(record.experiment_identity) is not None
            ):
                # same hypothesis, different ATTEMPT provenance offered by a bare
                # re-insert while a VALID authoritative attempt already exists ->
                # "changing a completed experiment in place". A real re-execution
                # after an INVALID attempt passes an explicit ExecutionAttemptRecord,
                # or (invalid-only history) is allowed a fresh VALID attempt below.
                self._raise_content_conflict(record, existing)
        else:
            self._conn.execute(
                """
                INSERT INTO experiments (
                    experiment_identity, identity_schema,
                    experiment_id, display_name, created_at, phase,
                    status, code_commit, root_symbol, asset_domain, strategy_family,
                    strategy_fingerprint,
                    strategy_id, strategy_spec_json, feature_spec_fingerprint,
                    target_schedule_hash, candidate_manifest_fingerprint, dataset_fingerprint,
                    split_identity, market_window_json, validation_spec_fingerprint,
                    reliability_policy_fingerprint, execution_config_identity,
                    cost_config_identity, risk_identity, trial_role, canonical_or_neighbour,
                    parameter_variant_identity, parameter_variant_label,
                    parent_experiment_identity, report_fingerprint, notes,
                    content_fingerprint, record_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.experiment_identity, record.identity_schema,
                    record.experiment_id, record.display_name,
                    record.created_at, record.phase, record.status.value, record.code_commit,
                    record.root_symbol, record.asset_domain.value, record.strategy_family,
                    record.strategy_fingerprint,
                    record.strategy_id, _dumps(record.strategy_spec_json),
                    record.feature_spec_fingerprint, record.target_schedule_hash,
                    record.candidate_manifest_fingerprint, record.dataset_fingerprint,
                    record.split_identity, _dumps(record.market_window.model_dump(mode="json")),
                    record.validation_spec_fingerprint, record.reliability_policy_fingerprint,
                    record.execution_config_identity, record.cost_config_identity,
                    record.risk_identity, record.trial_role.value,
                    record.canonical_or_neighbour, record.parameter_variant_identity,
                    record.parameter_variant_label, record.parent_experiment_identity,
                    record.report_fingerprint, record.notes, record.content_fingerprint(),
                    _dumps(record.model_dump(mode="json")),
                ),
            )

        if result is not None or attempt is not None:
            self._record_attempt_and_result(
                record.experiment_identity, attempt=attempt, result=result,
                fallback_provenance=record,
            )
        return self.get(record.experiment_identity)

    # ------------------------------------------------------------------
    # execution attempts (schema v5)
    # ------------------------------------------------------------------
    def _next_attempt_ordinal(self, identity: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(attempt_ordinal) FROM execution_attempts WHERE experiment_identity = ?",
            (identity,),
        ).fetchone()
        return int(row[0] or 0) + 1

    def _attempt_rows(self, identity: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM execution_attempts WHERE experiment_identity = ? "
            "ORDER BY attempt_ordinal",
            (identity,),
        ).fetchall()

    def _authoritative_attempt_row(self, identity: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM execution_attempts WHERE experiment_identity = ? "
            "AND attempt_status = ? ORDER BY attempt_ordinal DESC LIMIT 1",
            (identity, AttemptStatus.VALID.value),
        ).fetchone()

    def _record_attempt_and_result(
        self,
        identity: str,
        *,
        attempt: ExecutionAttemptRecord | None,
        result: ResultRecord | None,
        fallback_provenance: ExperimentRecord | None = None,
    ) -> str:
        """Insert one immutable execution attempt (+ its result) for ``identity``.

        A bare ``result`` with no ``attempt`` is recorded on a fresh ``VALID``
        attempt -- this keeps every pre-v5 importer working unchanged.
        """
        if attempt is not None and attempt.experiment_identity != identity:
            raise RegistryError("attempt.experiment_identity does not match the experiment")
        if attempt is not None:
            attempt.assert_no_holdout()

        if attempt is None:
            # A bare result (no explicit ExecutionAttemptRecord) is only allowed
            # to create a NEW attempt when the hypothesis has no VALID
            # authoritative result yet. Once a VALID result exists, adding
            # another attempt must be an EXPLICIT, deliberate re-execution.
            auth = self._authoritative_attempt_row(identity)
            if auth is not None and result is not None:
                current = self._conn.execute(
                    "SELECT result_fingerprint FROM attempt_results WHERE attempt_id = ?",
                    (auth["attempt_id"],),
                ).fetchone()
                if current is not None and current["result_fingerprint"] == (
                    result.result_fingerprint()
                ):
                    return auth["attempt_id"]           # idempotent no-op
                raise ImmutableResultError(
                    f"{identity} already has a VALID authoritative result "
                    f"(attempt #{auth['attempt_ordinal']}); a completed result is "
                    "immutable. To re-execute after an invalid attempt, pass an "
                    "explicit ExecutionAttemptRecord."
                )
            fp = fallback_provenance
            attempt = ExecutionAttemptRecord(
                experiment_identity=identity,
                attempt_status=AttemptStatus.VALID,
                code_commit=(fp.code_commit if fp else ""),
                target_schedule_hash=(fp.target_schedule_hash if fp else None),
                report_fingerprint=(fp.report_fingerprint if fp else None),
                source_artifact=(result.source_artifact if result else None),
                source_artifact_sha256=(result.source_artifact_sha256 if result else None),
                created_at=(fp.created_at if fp else datetime.now(UTC).isoformat()),
            )
        ordinal = attempt.attempt_ordinal or self._next_attempt_ordinal(identity)
        attempt_id = attempt.attempt_id(ordinal)

        existing = self._conn.execute(
            "SELECT content_fingerprint, attempt_status FROM execution_attempts "
            "WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if existing is not None:
            if existing["content_fingerprint"] != attempt.content_fingerprint():
                raise AttemptConflict(
                    f"execution attempt {attempt_id} already exists with different "
                    "content; a completed attempt is immutable"
                )
        else:
            if self._conn.execute(
                "SELECT 1 FROM execution_attempts WHERE experiment_identity = ? "
                "AND attempt_ordinal = ?",
                (identity, ordinal),
            ).fetchone() is not None:
                raise AttemptConflict(
                    f"{identity} already has an attempt at ordinal {ordinal}"
                )
            rec_json = attempt.model_dump(mode="json")
            rec_json["attempt_ordinal"] = ordinal
            rec_json["resolved_attempt_id"] = attempt_id
            self._conn.execute(
                "INSERT INTO execution_attempts ("
                " attempt_id, experiment_identity, identity_schema, attempt_ordinal,"
                " attempt_status, invalidation_class, invalidation_detail,"
                " invalidation_evidence_json, defect_resolution_commit, code_commit, engine,"
                " target_schedule_hash, report_fingerprint, source_artifact,"
                " source_artifact_sha256, created_at, notes, content_fingerprint, record_json"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt_id, identity, attempt.identity_schema, ordinal,
                    attempt.attempt_status.value,
                    attempt.invalidation_class.value if attempt.invalidation_class else None,
                    attempt.invalidation_detail,
                    _dumps(attempt.invalidation_evidence),
                    attempt.defect_resolution_commit, attempt.code_commit, attempt.engine,
                    attempt.target_schedule_hash, attempt.report_fingerprint,
                    attempt.source_artifact, attempt.source_artifact_sha256,
                    attempt.created_at, attempt.notes, attempt.content_fingerprint(),
                    _dumps(rec_json),
                ),
            )
            # derived-authority audit edge: a new attempt supersedes/corrects the
            # previous ordinal for the same identity (never the authority itself).
            if ordinal > 1:
                prev = self._conn.execute(
                    "SELECT attempt_id, attempt_status FROM execution_attempts "
                    "WHERE experiment_identity = ? AND attempt_ordinal = ?",
                    (identity, ordinal - 1),
                ).fetchone()
                if prev is not None:
                    rel = (
                        AttemptRelation.CORRECTS_ATTEMPT
                        if prev["attempt_status"] == AttemptStatus.INVALID_EXECUTION.value
                        else AttemptRelation.SUPERSEDES_ATTEMPT
                    )
                    self._conn.execute(
                        "INSERT OR IGNORE INTO attempt_lineage "
                        "(source_attempt_id, target_attempt_id, relation_type, note) "
                        "VALUES (?,?,?,?)",
                        (attempt_id, prev["attempt_id"], rel.value,
                         f"attempt {ordinal} follows attempt {ordinal - 1}"),
                    )

        if result is not None:
            self._record_attempt_result(result, attempt_id, ordinal, identity)
        return attempt_id

    def _record_attempt_result(
        self, result: ResultRecord, attempt_id: str, ordinal: int, identity: str
    ) -> None:
        existing = self._conn.execute(
            "SELECT result_fingerprint FROM attempt_results WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if existing is not None:
            if existing["result_fingerprint"] != result.result_fingerprint():
                raise ImmutableResultError(
                    f"execution attempt {attempt_id} already has a recorded scientific "
                    "result; a completed attempt is immutable. Record a NEW attempt."
                )
            return
        rec = result.model_dump(mode="json")
        rec["attempt_id"] = attempt_id
        self._conn.execute(
            """
            INSERT INTO attempt_results (
                attempt_id, experiment_identity, attempt_ordinal, headline_verdict,
                reason_codes_json, gross_pnl_usd, costs_usd, net_pnl_usd, daily_sharpe,
                annualized_sharpe, gating_null_p, bh_p_value, bh_q, bh_rejected_at_q,
                dsr_probability, fold_consistency, n_trades, n_fills, n_oos_days,
                holdout_eligible, evidence_completeness, source_artifact,
                source_artifact_sha256, result_fingerprint, record_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                attempt_id, identity, ordinal, result.headline_verdict.value,
                _dumps(list(result.reason_codes)), result.gross_pnl_usd, result.costs_usd,
                result.net_pnl_usd, result.daily_sharpe, result.annualized_sharpe,
                result.gating_null_p, result.bh_p_value, result.bh_q,
                None if result.bh_rejected_at_q is None else int(result.bh_rejected_at_q),
                result.dsr_probability, result.fold_consistency, result.n_trades,
                result.n_fills, result.n_oos_days, int(result.holdout_eligible),
                result.evidence_completeness, result.source_artifact,
                result.source_artifact_sha256, result.result_fingerprint(),
                _dumps(rec),
            ),
        )

    def _raise_content_conflict(self, record: ExperimentRecord, existing) -> None:
        """Classify the conflict before failing, so the message says what to
        investigate rather than just "different content"."""
        stored_hash = existing["target_schedule_hash"]
        offered_hash = record.target_schedule_hash
        if stored_hash != offered_hash:
            if stored_hash is not None and offered_hash is not None:
                raise ScheduleProvenanceConflict(
                    f"experiment_identity {record.experiment_identity} already exists as "
                    f"{existing['experiment_id']} with target_schedule_hash "
                    f"{stored_hash!r}, but a run with identical pre-run semantic inputs "
                    f"reports {offered_hash!r}. Identical inputs must compile to an "
                    "identical schedule: investigate the compiler / data, do not record "
                    "this as a separate experiment."
                )
            raise ScheduleProvenanceConflict(
                f"experiment_identity {record.experiment_identity} already exists as "
                f"{existing['experiment_id']} with target_schedule_hash {stored_hash!r}; "
                f"offered {offered_hash!r}. Provenance is never backfilled or removed in "
                "place on a completed experiment."
            )
        raise ExperimentConflict(
            f"experiment_identity {record.experiment_identity} already exists as "
            f"{existing['experiment_id']} with different scientific content. "
            "A completed experiment is never replaced in place -- record a new "
            "experiment and an explicit CORRECTS / SUPERSEDES relation."
        )

    def record_failure(self, failure: FailureRecord) -> None:
        with self._conn:
            self._record_failure(failure)

    def _record_failure(self, failure: FailureRecord) -> None:
        failure.assert_no_holdout()
        existing = self._conn.execute(
            "SELECT content_fingerprint FROM failures WHERE failure_id = ?",
            (failure.failure_id,),
        ).fetchone()
        if existing is not None:
            if existing["content_fingerprint"] != failure.content_fingerprint():
                raise ExperimentConflict(
                    f"failure_id {failure.failure_id} already exists with different "
                    "content; failures are append-only"
                )
            return
        attempt_id = failure.attempt_id
        if attempt_id is None and failure.experiment_identity is not None:
            # bind to the identity's latest attempt (schema v5)
            latest = self._conn.execute(
                "SELECT attempt_id FROM execution_attempts WHERE experiment_identity = ? "
                "ORDER BY attempt_ordinal DESC LIMIT 1",
                (failure.experiment_identity,),
            ).fetchone()
            attempt_id = latest["attempt_id"] if latest else None
        self._conn.execute(
            """
            INSERT INTO failures (
                failure_id, scope, experiment_identity, attempt_id, failure_class,
                failure_code, summary, mechanism, evidence_json, action_taken, resolved,
                resolution_commit, superseded_by, created_at, root_symbol,
                strategy_family, content_fingerprint
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                failure.failure_id, failure.scope.value, failure.experiment_identity,
                attempt_id, failure.failure_class.value, failure.failure_code,
                failure.summary, failure.mechanism, _dumps(failure.evidence),
                failure.action_taken, int(failure.resolved), failure.resolution_commit,
                failure.superseded_by, failure.created_at, failure.root_symbol,
                failure.strategy_family, failure.content_fingerprint(),
            ),
        )

    # -- News Alpha Phase H: signal-path evidence (schema v7) ---------------
    def record_signal_path_evidence(self, records: Sequence[SignalPathEvidenceRecord]) -> int:
        """Append typed hypothesis-plane evidence, all or nothing. A record
        already present with the same ``evidence_id`` (its content
        fingerprint) is skipped -- recording one run twice writes nothing
        new. Returns the number of rows inserted. A linked
        ``experiment_identity`` must already exist in the registry."""
        for r in records:
            r.assert_consistent()
            r.assert_no_holdout()
        inserted = 0
        with self._conn:
            for r in records:
                if self._conn.execute("SELECT 1 FROM signal_path_evidence WHERE evidence_id = ?",
                                      (r.evidence_id,)).fetchone() is not None:
                    continue
                if r.experiment_identity is not None and not self._exists(r.experiment_identity):
                    raise UnknownExperiment(
                        f"signal-path evidence {r.evidence_id} links unknown experiment {r.experiment_identity}")
                self._conn.execute(
                    "INSERT INTO signal_path_evidence (evidence_id, research_run_id, recorded_at, event_id, "
                    "path_signature, path_type, candidate_signal_id, experiment_identity, stage_reached, outcome, "
                    "evidence_scope, reason_code, record_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r.evidence_id, r.research_run_id, r.recorded_at, r.event_id, r.path_signature, r.path_type,
                     r.candidate_signal_id, r.experiment_identity, r.stage_reached.value, r.outcome.value,
                     r.evidence_scope.value, r.reason_code.value, r.model_dump_json()),
                )
                inserted += 1
        return inserted

    def signal_path_evidence(
        self,
        *,
        path_signature: str | None = None,
        event_id: str | None = None,
        candidate_signal_id: str | None = None,
        research_run_id: str | None = None,
        experiment_identity: str | None = None,
    ) -> tuple[SignalPathEvidenceRecord, ...]:
        """Recorded signal-path evidence, oldest first, filtered by any id."""
        clauses, args = [], []
        for column, value in (("path_signature", path_signature), ("event_id", event_id),
                              ("candidate_signal_id", candidate_signal_id), ("research_run_id", research_run_id),
                              ("experiment_identity", experiment_identity)):
            if value is not None:
                clauses.append(f"{column} = ?")
                args.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(f"SELECT record_json FROM signal_path_evidence{where} ORDER BY row_id", args)
        return tuple(SignalPathEvidenceRecord.model_validate_json(r["record_json"]) for r in rows)

    def record_lineage(self, edge: LineageEdge) -> None:
        with self._conn:
            self._record_lineage(edge)

    def _record_lineage(self, edge: LineageEdge) -> None:
        for ident in (edge.source_experiment_identity, edge.target_experiment_identity):
            if not self._exists(ident):
                raise UnknownExperiment(
                    f"lineage edge references unknown experiment {ident}"
                )
        self._conn.execute(
            "INSERT OR IGNORE INTO lineage "
            "(source_experiment_identity, target_experiment_identity, relation_type, note) "
            "VALUES (?,?,?,?)",
            (edge.source_experiment_identity, edge.target_experiment_identity,
             edge.relation_type.value, edge.note),
        )

    def _attempt_id_at(self, identity: str, ordinal: int) -> str:
        row = self._conn.execute(
            "SELECT attempt_id FROM execution_attempts WHERE experiment_identity = ? "
            "AND attempt_ordinal = ?",
            (identity, ordinal),
        ).fetchone()
        if row is None:
            raise UnknownExperiment(
                f"no execution attempt {identity} #{ordinal} for the lineage edge"
            )
        return row["attempt_id"]

    def _record_attempt_lineage(self, edge: AttemptLineageEdge) -> None:
        src = self._attempt_id_at(edge.source_experiment_identity, edge.source_attempt_ordinal)
        tgt = self._attempt_id_at(edge.target_experiment_identity, edge.target_attempt_ordinal)
        self._conn.execute(
            "INSERT OR IGNORE INTO attempt_lineage "
            "(source_attempt_id, target_attempt_id, relation_type, note) VALUES (?,?,?,?)",
            (src, tgt, edge.relation_type.value, edge.note),
        )

    def apply_bundle(self, bundle: ImportBundle) -> dict[str, int]:
        """Apply a whole import atomically (section 22).

        The bundle is fully validated (holdout guard, identity/lineage
        consistency) *before* the transaction opens, and every insert happens
        inside a single transaction: a partial failure leaves no half-populated
        authoritative research state.
        """
        self._validate_bundle(bundle)
        counts = {
            "experiments": len(bundle.experiments),
            "execution_attempts": (
                len(bundle.execution_attempts) or len(bundle.results) or len(bundle.experiments)
            ),
            "results": len(bundle.results),
            "failures": len(bundle.failures),
            "lineage": len(bundle.lineage),
            "attempt_lineage": len(bundle.attempt_lineage),
            "sensitivity": len(bundle.sensitivity),
            "cross_market": len(bundle.cross_market),
        }
        results_by_identity = {r.experiment_identity: r for r in bundle.results}
        attempts_by_identity: dict[str, ExecutionAttemptRecord] = {
            a.experiment_identity: a for a in bundle.execution_attempts
        }
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            edges_before = self._conn.execute(
                "SELECT COUNT(*) FROM attempt_lineage"
            ).fetchone()[0]
            for exp in bundle.experiments:
                ident = exp.experiment_identity
                self._insert_experiment(
                    exp,
                    results_by_identity.get(ident),
                    attempt=attempts_by_identity.get(ident),
                )
            # explicit attempts for identities NOT in bundle.experiments (rare)
            for a in bundle.execution_attempts:
                if a.experiment_identity not in {e.experiment_identity for e in bundle.experiments}:
                    self._record_attempt_and_result(
                        a.experiment_identity, attempt=a,
                        result=results_by_identity.get(a.experiment_identity),
                    )
            for edge in bundle.lineage:
                self._record_lineage(edge)
            for edge in bundle.attempt_lineage:
                self._record_attempt_lineage(edge)
            for failure in bundle.failures:
                self._record_failure(failure)
            for sens in bundle.sensitivity:
                self._record_sensitivity(sens)
            for cm in bundle.cross_market:
                self._record_cross_market(cm)
            edges_after = self._conn.execute(
                "SELECT COUNT(*) FROM attempt_lineage"
            ).fetchone()[0]
            counts["attempt_lineage"] = int(edges_after - edges_before)
            self._conn.execute(
                "INSERT OR IGNORE INTO imports "
                "(import_id, phase, source_fingerprint, created_at, counts_json, metadata_json) "
                "VALUES (?,?,?,?,?,?)",
                (bundle.import_id, bundle.phase, bundle.source_fingerprint,
                 bundle.created_at, _dumps(counts), _dumps(bundle.metadata)),
            )
        except Exception:
            self._conn.rollback()
            raise
        self._conn.commit()
        return counts

    def _validate_bundle(self, bundle: ImportBundle) -> None:
        identities = {e.experiment_identity for e in bundle.experiments}
        if len(identities) != len(bundle.experiments):
            raise ExperimentConflict("import bundle contains duplicate experiment identities")
        friendly = {e.experiment_id for e in bundle.experiments}
        if len(friendly) != len(bundle.experiments):
            raise ExperimentConflict("import bundle contains duplicate friendly experiment ids")
        for exp in bundle.experiments:
            exp.assert_no_holdout()
        for att in bundle.execution_attempts:
            att.assert_no_holdout()
            if att.attempt_status is AttemptStatus.VALID and att.invalidation_class is not None:
                raise AttemptConflict("a VALID attempt cannot carry an invalidation_class")
        for res in bundle.results:
            res.assert_no_holdout()
            if res.experiment_identity not in identities and not self._exists(
                res.experiment_identity
            ):
                raise UnknownExperiment(
                    f"result references unknown experiment {res.experiment_identity}"
                )
        for f in bundle.failures:
            f.assert_no_holdout()
        known = identities | {r["experiment_identity"] for r in self._conn.execute(
            "SELECT experiment_identity FROM experiments"
        )}
        for edge in bundle.lineage:
            for ident in (edge.source_experiment_identity, edge.target_experiment_identity):
                if ident not in known:
                    raise UnknownExperiment(
                        f"lineage edge references unknown experiment {ident}"
                    )
        for s in bundle.sensitivity:
            if s.experiment_identity not in known:
                raise UnknownExperiment(
                    f"sensitivity evidence references unknown experiment {s.experiment_identity}"
                )

    def _record_sensitivity(self, s: SensitivityEvidence) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO sensitivity_evidence "
            "(evidence_id, experiment_identity, relation_type, kind, baseline_verdict, "
            " rerun_verdict, verdict_changed, metrics_json, in_bh_fdr_denominator) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (s.evidence_id, s.experiment_identity, s.relation_type.value, s.kind,
             s.baseline_verdict.value, s.rerun_verdict.value, int(s.verdict_changed),
             _dumps(s.metrics), int(s.in_bh_fdr_denominator)),
        )

    def _record_cross_market(self, cm: CrossMarketEvidenceRecord) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO cross_market_evidence "
            "(evidence_id, strategy_family, status, n_roots, n_positive_roots, "
            " max_single_root_pnl_share, herfindahl_pnl, concentrated_in_one_root, "
            " referenced_identities_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (cm.evidence_id, cm.strategy_family, cm.status, cm.n_roots,
             cm.n_positive_roots, cm.max_single_root_pnl_share, cm.herfindahl_pnl,
             int(cm.concentrated_in_one_root), _dumps(list(cm.referenced_experiment_identities))),
        )

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def _exists(self, identity: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM experiments WHERE experiment_identity = ?", (identity,)
        ).fetchone() is not None

    def _row(self, key: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM experiments WHERE experiment_identity = ? OR experiment_id = ?",
            (key, key),
        ).fetchone()

    def _attempt_result(self, attempt_id: str) -> ResultRecord | None:
        rres = self._conn.execute(
            "SELECT record_json FROM attempt_results WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        return ResultRecord.model_validate(json.loads(rres["record_json"])) if rres else None

    def _attempt_view(self, r: sqlite3.Row) -> AttemptView:
        return AttemptView(
            attempt_id=r["attempt_id"],
            experiment_identity=r["experiment_identity"],
            attempt_ordinal=int(r["attempt_ordinal"]),
            attempt_status=AttemptStatus(r["attempt_status"]),
            invalidation_class=r["invalidation_class"],
            invalidation_detail=r["invalidation_detail"] or "",
            defect_resolution_commit=r["defect_resolution_commit"],
            code_commit=r["code_commit"] or "",
            engine=r["engine"] or "",
            report_fingerprint=r["report_fingerprint"],
            target_schedule_hash=r["target_schedule_hash"],
            created_at=r["created_at"] or "",
            notes=r["notes"] or "",
            result=self._attempt_result(r["attempt_id"]),
        )

    def _view(self, row: sqlite3.Row) -> ExperimentView:
        exp = ExperimentRecord.model_validate(json.loads(row["record_json"]))
        attempt_rows = self._attempt_rows(exp.experiment_identity)
        attempts = tuple(self._attempt_view(r) for r in attempt_rows)
        auth_attempt = self._authoritative_attempt_row(exp.experiment_identity)
        result = None
        auth_attempt_id = None
        if auth_attempt is not None:
            auth_attempt_id = auth_attempt["attempt_id"]
            result = self._attempt_result(auth_attempt_id)
        superseded_by = self.superseded_by(exp.experiment_identity)
        return ExperimentView(
            experiment=exp,
            result=result,
            authority=Authority.SUPERSEDED if superseded_by else Authority.AUTHORITATIVE,
            superseded_by=superseded_by,
            attempts=attempts,
            authoritative_attempt_id=auth_attempt_id,
        )

    # -- execution attempts (schema v5) ---------------------------------
    def attempts(self, key: str) -> tuple[AttemptView, ...]:
        """Every execution attempt of an experiment, oldest ordinal first."""
        view = self.get(key)
        return view.attempts

    def authoritative_attempt(self, key: str) -> AttemptView | None:
        """The attempt that supplies the authoritative scientific result -- the
        highest-ordinal VALID attempt, or ``None`` if the hypothesis has only
        invalid attempts (or none)."""
        view = self.get(key)
        if view.authoritative_attempt_id is None:
            return None
        return next(a for a in view.attempts if a.attempt_id == view.authoritative_attempt_id)

    def get(self, key: str) -> ExperimentView:
        """Look up by ``experiment_identity`` or friendly ``experiment_id``."""
        row = self._row(key)
        if row is None:
            raise UnknownExperiment(f"no experiment with identity or id {key!r}")
        return self._view(row)

    def superseded_by(self, identity: str) -> tuple[str, ...]:
        """Identities of experiments that SUPERSEDE / CORRECT ``identity``."""
        rows = self._conn.execute(
            "SELECT source_experiment_identity FROM lineage "
            "WHERE target_experiment_identity = ? AND relation_type IN (?, ?) "
            "ORDER BY source_experiment_identity",
            (identity, RelationType.SUPERSEDES.value, RelationType.CORRECTS.value),
        ).fetchall()
        return tuple(dict.fromkeys(r[0] for r in rows))

    def authority_of(self, identity: str) -> Authority:
        return Authority.SUPERSEDED if self.superseded_by(identity) else Authority.AUTHORITATIVE

    def resolve_authoritative(self, identity: str) -> ExperimentView:
        """Follow SUPERSEDES / CORRECTS edges to the current answer (section 12).

        Cycles are impossible to follow blindly, so the walk is bounded by the
        number of experiment rows and raises if it does not terminate.
        """
        seen = {identity}
        current = identity
        limit = self.count_experiments() + 1
        for _ in range(limit):
            nxt = self.superseded_by(current)
            if not nxt:
                return self.get(current)
            current = min(nxt)
            if current in seen:
                raise RegistryError(f"supersession cycle detected at {current}")
            seen.add(current)
        raise RegistryError(f"supersession chain from {identity} did not terminate")

    def lineage_edges(self, identity: str | None = None) -> tuple[LineageEdge, ...]:
        if identity is None:
            rows = self._conn.execute(
                "SELECT * FROM lineage ORDER BY relation_type, source_experiment_identity, "
                "target_experiment_identity"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM lineage WHERE source_experiment_identity = ? "
                "OR target_experiment_identity = ? "
                "ORDER BY relation_type, source_experiment_identity, target_experiment_identity",
                (identity, identity),
            ).fetchall()
        return tuple(
            LineageEdge(
                source_experiment_identity=r["source_experiment_identity"],
                target_experiment_identity=r["target_experiment_identity"],
                relation_type=RelationType(r["relation_type"]),
                note=r["note"],
            )
            for r in rows
        )

    def experiments(
        self,
        *,
        strategy_family: str | None = None,
        root_symbol: str | None = None,
        asset_domain: AssetDomain | None = None,
        trial_role: TrialRole | None = None,
        phase: str | None = None,
        verdict: RegistryVerdict | None = None,
        reason_code: str | None = None,
        authoritative_only: bool = True,
        include_superseded: bool = False,
    ) -> tuple[ExperimentView, ...]:
        """Filtered experiment query.

        ``authoritative_only=True`` (the default) hides superseded lineage;
        ``include_superseded=True`` returns both historical and corrected rows.

        ``asset_domain`` is an OPTIONAL structural filter (Phase 6 ETF Research
        Pilot): ``None`` (the default) is domain-blind -- correct for a genuine
        whole-registry browse/export/report, never for a query that pools
        evidence by ``strategy_family``/``root_symbol`` (BH/FDR family
        membership, near-duplicate retrieval, failure-memory lookup). Those
        callers (``find_related``, ``FailureMemory.lookup``, ...) always pass
        it explicitly -- see ``alpha_agent.registry.failure_memory`` and
        ``find_related`` below, which require it rather than default it.
        """
        sql = ["SELECT e.* FROM experiments e"]
        where: list[str] = []
        args: list[Any] = []
        if verdict is not None or reason_code is not None:
            # the AUTHORITATIVE result = the highest-ordinal VALID attempt's
            # attempt_results row. INVALID attempts never satisfy a verdict filter.
            sql.append(
                "JOIN attempt_results r ON r.attempt_id = ("
                " SELECT ea.attempt_id FROM execution_attempts ea"
                " WHERE ea.experiment_identity = e.experiment_identity"
                " AND ea.attempt_status = 'VALID'"
                " ORDER BY ea.attempt_ordinal DESC LIMIT 1)"
            )
        if strategy_family is not None:
            where.append("e.strategy_family = ?")
            args.append(strategy_family)
        if root_symbol is not None:
            where.append("e.root_symbol = ?")
            args.append(root_symbol)
        if asset_domain is not None:
            where.append("e.asset_domain = ?")
            args.append(asset_domain.value)
        if trial_role is not None:
            where.append("e.trial_role = ?")
            args.append(trial_role.value)
        if phase is not None:
            where.append("e.phase = ?")
            args.append(phase)
        if verdict is not None:
            where.append("r.headline_verdict = ?")
            args.append(verdict.value)
        if reason_code is not None:
            where.append("r.reason_codes_json LIKE ?")
            args.append(f'%"{reason_code}"%')
        if where:
            sql.append("WHERE " + " AND ".join(where))
        sql.append("ORDER BY e.experiment_id")
        rows = self._conn.execute(" ".join(sql), args).fetchall()
        views = [self._view(r) for r in rows]
        if include_superseded:
            return tuple(views)
        if authoritative_only:
            return tuple(v for v in views if v.authority is Authority.AUTHORITATIVE)
        return tuple(views)

    def count_experiments(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])

    # -- section 9: exact duplicate ------------------------------------
    def find_exact_duplicate(
        self, experiment_identity: str, *, asset_domain: AssetDomain
    ) -> ExactDuplicate:
        """Has this exact experiment already been run? (no backtest required)

        ``asset_domain`` is REQUIRED (Phase 6 ETF Research Pilot), not
        inferred: defense in depth over the identity formula's own structural
        domain distinctness (dataset / execution-config fingerprints already
        differ between domains). A hit under a DIFFERENT domain than requested
        raises :class:`AssetDomainMismatch` -- a genuine integrity condition,
        never silently treated as "not found".
        """
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE experiment_identity = ?",
            (experiment_identity,),
        ).fetchone()
        if row is None:
            return ExactDuplicate(exists=False, experiment_identity=experiment_identity)
        if row["asset_domain"] != asset_domain.value:
            raise AssetDomainMismatch(
                f"experiment_identity {experiment_identity} exists under asset_domain "
                f"{row['asset_domain']!r}, not the requested {asset_domain.value!r}; "
                "identity collisions across domains should be structurally impossible "
                "-- investigate rather than proceeding"
            )
        view = self._view(row)
        superseded = view.superseded_by
        replacement = None
        if superseded:
            replacement = self.resolve_authoritative(experiment_identity).experiment_id
        latest_status = view.attempts[-1].attempt_status if view.attempts else None
        return ExactDuplicate(
            exists=True,
            experiment_identity=experiment_identity,
            experiment_id=view.experiment_id,
            display_name=view.experiment.display_name,
            phase=view.experiment.phase,
            headline_verdict=view.verdict,
            reason_codes=view.result.reason_codes if view.result else (),
            report_fingerprint=view.experiment.report_fingerprint,
            authority=view.authority,
            superseded_by=superseded,
            authoritative_replacement=replacement,
            has_valid_authoritative_result=view.has_valid_authoritative_result,
            n_attempts=len(view.attempts),
            n_valid_attempts=view.n_valid_attempts,
            n_invalid_attempts=view.n_invalid_attempts,
            latest_attempt_status=latest_status,
        )

    # -- section 10: near duplicate ------------------------------------
    def find_related(
        self,
        *,
        strategy_family: str,
        root_symbol: str,
        asset_domain: AssetDomain,
        params: dict,
        signal_cadence: str = "",
        execution_cadence: str = "",
        feature_fingerprints: Sequence[str] = (),
        top_k: int = 10,
        min_score: float = 0.0,
        authoritative_only: bool = True,
    ) -> tuple[RelatedExperiment, ...]:
        """Deterministic near-duplicate retrieval with explicit reasons.

        Memory / warning infrastructure only: it returns prior evidence, it
        never rejects a hypothesis and never touches a ``ReliabilityPolicy``.

        ``asset_domain`` is REQUIRED (Phase 6 ETF Research Pilot): the
        candidate pool is scoped to it BEFORE any similarity score is
        computed, so a same-named ``strategy_family`` under a different domain
        can never surface here -- structural isolation, not incidental (see
        CLAUDE.md's Experiment registry section). ETF and Futures may share
        Mechanism vocabulary; they must never share near-duplicate evidence.
        """
        query_shape = structural_shape(
            strategy_family=strategy_family, params=params,
            signal_cadence=signal_cadence, execution_cadence=execution_cadence,
        )
        scored: list[tuple[SimilarityBreakdown, str, ExperimentView]] = []
        for view in self.experiments(
            authoritative_only=authoritative_only, asset_domain=asset_domain
        ):
            spec = view.experiment.strategy_spec_json
            cand_params = dict(spec.get("params", {}))
            cand_shape = structural_shape(
                strategy_family=view.experiment.strategy_family,
                params=cand_params,
                signal_cadence=str(spec.get("signal_cadence", "")),
                execution_cadence=str(spec.get("execution_cadence", "")),
            )
            breakdown = similarity(
                query_family=strategy_family, query_root=root_symbol,
                query_params=params, query_shape=query_shape,
                query_features=tuple(feature_fingerprints),
                candidate_family=view.experiment.strategy_family,
                candidate_root=view.experiment.root_symbol,
                candidate_params=cand_params, candidate_shape=cand_shape,
                candidate_features=tuple(spec.get("feature_fingerprints", ())),
            )
            if breakdown.score >= min_score:
                scored.append((breakdown, view.experiment_id, view))
        scored.sort(key=lambda t: rank_key((t[0], t[1])))
        out = []
        for breakdown, _, view in scored[:top_k]:
            out.append(
                RelatedExperiment(
                    experiment_id=view.experiment_id,
                    experiment_identity=view.experiment_identity,
                    display_name=view.experiment.display_name,
                    strategy_family=view.experiment.strategy_family,
                    root_symbol=view.experiment.root_symbol,
                    trial_role=view.experiment.trial_role,
                    params=dict(view.experiment.strategy_spec_json.get("params", {})),
                    headline_verdict=view.verdict,
                    reason_codes=view.result.reason_codes if view.result else (),
                    authority=view.authority,
                    similarity=breakdown,
                )
            )
        return tuple(out)

    # -- failures --------------------------------------------------------
    def failures(
        self,
        *,
        failure_class: FailureClass | None = None,
        experiment_identity: str | None = None,
        strategy_family: str | None = None,
        root_symbol: str | None = None,
        resolved: bool | None = None,
    ) -> tuple[FailureRecord, ...]:
        where: list[str] = []
        args: list[Any] = []
        if failure_class is not None:
            where.append("failure_class = ?")
            args.append(failure_class.value)
        if experiment_identity is not None:
            where.append("experiment_identity = ?")
            args.append(experiment_identity)
        if strategy_family is not None:
            where.append("strategy_family = ?")
            args.append(strategy_family)
        if root_symbol is not None:
            where.append("root_symbol = ?")
            args.append(root_symbol)
        if resolved is not None:
            where.append("resolved = ?")
            args.append(int(resolved))
        sql = "SELECT * FROM failures"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY failure_class, failure_id"
        return tuple(_failure_from_row(r) for r in self._conn.execute(sql, args).fetchall())

    def sensitivity_evidence(
        self, experiment_identity: str | None = None
    ) -> tuple[SensitivityEvidence, ...]:
        sql = "SELECT * FROM sensitivity_evidence"
        args: tuple = ()
        if experiment_identity is not None:
            sql += " WHERE experiment_identity = ?"
            args = (experiment_identity,)
        sql += " ORDER BY evidence_id"
        return tuple(
            SensitivityEvidence(
                evidence_id=r["evidence_id"],
                experiment_identity=r["experiment_identity"],
                relation_type=RelationType(r["relation_type"]),
                kind=r["kind"],
                baseline_verdict=RegistryVerdict(r["baseline_verdict"]),
                rerun_verdict=RegistryVerdict(r["rerun_verdict"]),
                verdict_changed=bool(r["verdict_changed"]),
                metrics=json.loads(r["metrics_json"]),
                in_bh_fdr_denominator=bool(r["in_bh_fdr_denominator"]),
            )
            for r in self._conn.execute(sql, args).fetchall()
        )

    def cross_market_evidence(self) -> tuple[CrossMarketEvidenceRecord, ...]:
        return tuple(
            CrossMarketEvidenceRecord(
                evidence_id=r["evidence_id"],
                strategy_family=r["strategy_family"],
                status=r["status"],
                n_roots=r["n_roots"],
                n_positive_roots=r["n_positive_roots"],
                max_single_root_pnl_share=r["max_single_root_pnl_share"],
                herfindahl_pnl=r["herfindahl_pnl"],
                concentrated_in_one_root=bool(r["concentrated_in_one_root"]),
                referenced_experiment_identities=tuple(
                    json.loads(r["referenced_identities_json"])
                ),
            )
            for r in self._conn.execute(
                "SELECT * FROM cross_market_evidence ORDER BY strategy_family"
            ).fetchall()
        )

    def imports(self) -> tuple[dict, ...]:
        return tuple(
            {
                "import_id": r["import_id"],
                "phase": r["phase"],
                "source_fingerprint": r["source_fingerprint"],
                "created_at": r["created_at"],
                "counts": json.loads(r["counts_json"]),
                "metadata": json.loads(r["metadata_json"]),
            }
            for r in self._conn.execute("SELECT * FROM imports ORDER BY import_id").fetchall()
        )

    # -- summary / digest -------------------------------------------------
    def content_digest(self) -> str:
        """Deterministic digest of every semantic row.

        Two runs of the same deterministic import must produce the same digest;
        this is the machine-checkable form of "the importer is idempotent".
        """
        payload = {
            "experiments": sorted(
                r[0] for r in self._conn.execute(
                    "SELECT content_fingerprint FROM experiments"
                )
            ),
            "execution_attempts": sorted(
                "|".join((r[0], r[1], str(r[2]), r[3])) for r in self._conn.execute(
                    "SELECT experiment_identity, attempt_status, attempt_ordinal, "
                    "content_fingerprint FROM execution_attempts"
                )
            ),
            "results": sorted(
                r[0] for r in self._conn.execute(
                    "SELECT result_fingerprint FROM attempt_results"
                )
            ),
            "attempt_lineage": sorted(
                "|".join(r) for r in self._conn.execute(
                    "SELECT source_attempt_id, target_attempt_id, relation_type "
                    "FROM attempt_lineage"
                )
            ),
            "failures": sorted(
                r[0] for r in self._conn.execute("SELECT content_fingerprint FROM failures")
            ),
            "lineage": sorted(
                "|".join(r) for r in self._conn.execute(
                    "SELECT source_experiment_identity, target_experiment_identity, "
                    "relation_type FROM lineage"
                )
            ),
            "sensitivity": sorted(
                r[0] for r in self._conn.execute("SELECT evidence_id FROM sensitivity_evidence")
            ),
            "cross_market": sorted(
                r[0] for r in self._conn.execute("SELECT evidence_id FROM cross_market_evidence")
            ),
        }
        # schema v7: included only when present, so a registry without
        # signal-path evidence keeps its v6 digest byte for byte
        signal_paths = sorted(r[0] for r in self._conn.execute("SELECT evidence_id FROM signal_path_evidence"))
        if signal_paths:
            payload["signal_path_evidence"] = signal_paths
        return fingerprint("registrydigest1", payload)

    def summary(self) -> RegistrySummary:
        auth = self.experiments(authoritative_only=True)
        canonical = [v for v in auth if v.experiment.trial_role is TrialRole.CANONICAL]
        neighbour = [v for v in auth if v.experiment.trial_role is TrialRole.NEIGHBOUR]
        ablation = [v for v in auth if v.experiment.trial_role is TrialRole.ABLATION]
        variant = [v for v in auth if v.experiment.trial_role is TrialRole.VARIANT]
        # every adjudicable verdict is always present, so a reader can see
        # "PASS: 0" rather than having to notice a missing key. NOT_ADJUDICATED
        # is deliberately NOT pre-seeded here (existing callers/tests compare
        # this dict for exact equality against the 3 headline-adjudicated
        # verdicts); it still appears via .get(...)+1 below whenever a
        # canonical trial actually resolves to it, exactly as before this
        # reporting fix. The CLI printer (registry/cli.py) instead reads this
        # dict with .get(key, 0), so it always shows all four keys regardless
        # of whether this dict happens to contain NOT_ADJUDICATED.
        verdicts: dict[str, int] = {
            RegistryVerdict.PASS.value: 0,
            RegistryVerdict.REJECT.value: 0,
            RegistryVerdict.INCONCLUSIVE.value: 0,
        }
        canonical_without_valid_attempt = 0
        for v in canonical:
            if v.verdict is not None:
                verdicts[v.verdict.value] = verdicts.get(v.verdict.value, 0) + 1
            elif not v.has_valid_authoritative_result:
                canonical_without_valid_attempt += 1
        classes: dict[str, int] = {}
        for r in self._conn.execute(
            "SELECT failure_class, COUNT(*) FROM failures GROUP BY failure_class"
        ):
            classes[r[0]] = int(r[1])
        total = self.count_experiments()
        cpp = 0
        for imp in self.imports():
            cpp += int(imp["metadata"].get("completed_cpp_execution_count", 0) or 0)
        attempts_total, attempts_valid, attempts_invalid = self._conn.execute(
            "SELECT COUNT(*), "
            " SUM(CASE WHEN attempt_status='VALID' THEN 1 ELSE 0 END), "
            " SUM(CASE WHEN attempt_status='INVALID_EXECUTION' THEN 1 ELSE 0 END) "
            "FROM execution_attempts"
        ).fetchone()
        return RegistrySummary(
            schema_version=self.schema_version,
            identity_schema=IDENTITY_SCHEMA,
            registry_path=str(self.path),
            authoritative_statistical_hypotheses=len(auth),
            canonical=len(canonical),
            neighbour=len(neighbour),
            ablation=len(ablation),
            variant=len(variant),
            superseded_experiments=total - len(auth),
            total_experiment_rows=total,
            canonical_verdict_counts=verdicts,
            canonical_without_valid_attempt=canonical_without_valid_attempt,
            execution_attempts=int(attempts_total or 0),
            valid_execution_attempts=int(attempts_valid or 0),
            invalid_execution_attempts=int(attempts_invalid or 0),
            holdout_eligible=int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM attempt_results ar JOIN execution_attempts ea "
                    "ON ar.attempt_id = ea.attempt_id "
                    "WHERE ar.holdout_eligible = 1 AND ea.attempt_status = 'VALID'"
                ).fetchone()[0]
            ),
            failure_records=int(
                self._conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
            ),
            failure_class_counts=classes,
            lineage_edges=int(
                self._conn.execute("SELECT COUNT(*) FROM lineage").fetchone()[0]
            ),
            sensitivity_evidence=int(
                self._conn.execute("SELECT COUNT(*) FROM sensitivity_evidence").fetchone()[0]
            ),
            cross_market_evidence=int(
                self._conn.execute("SELECT COUNT(*) FROM cross_market_evidence").fetchone()[0]
            ),
            completed_cpp_executions=cpp,
            imports=self.imports(),
            content_digest=self.content_digest(),
        )


def _failure_from_row(r: sqlite3.Row) -> FailureRecord:
    from alpha_agent.registry.enums import FailureScope

    return FailureRecord(
        failure_id=r["failure_id"],
        scope=FailureScope(r["scope"]),
        experiment_identity=r["experiment_identity"],
        failure_class=FailureClass(r["failure_class"]),
        failure_code=r["failure_code"],
        summary=r["summary"],
        mechanism=r["mechanism"],
        evidence=json.loads(r["evidence_json"]),
        action_taken=r["action_taken"],
        resolved=bool(r["resolved"]),
        resolution_commit=r["resolution_commit"],
        superseded_by=r["superseded_by"],
        created_at=r["created_at"],
        root_symbol=r["root_symbol"],
        strategy_family=r["strategy_family"],
    )


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "SCHEMA_VERSION",
    "AssetDomainMismatch",
    "ExactDuplicate",
    "ExperimentConflict",
    "ExperimentRegistry",
    "ExperimentStatus",
    "ExperimentView",
    "IdentitySchemaMismatch",
    "ImmutableResultError",
    "MarketWindow",
    "RegistryError",
    "RegistrySummary",
    "RelatedExperiment",
    "ScheduleProvenanceConflict",
    "UnknownExperiment",
]
