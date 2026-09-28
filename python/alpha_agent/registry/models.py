"""Typed cross-component schemas for the Phase 14 registry (section 4).

Pydantic models are the only way records enter or leave the registry; the
SQLite layer never accepts a loose ``dict``. Every model forbids unknown fields
so a mis-spelled scientific field fails loudly at the boundary instead of being
silently dropped into a JSON blob.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from alpha_agent.registry.enums import (
    AssetDomain,
    AttemptRelation,
    AttemptStatus,
    EvidenceScope,
    ExperimentStatus,
    FailureClass,
    FailureScope,
    HypothesisStage,
    InvalidationClass,
    PathEvidenceOutcome,
    PathEvidenceReason,
    RegistryVerdict,
    RelationType,
    TrialRole,
)
from alpha_agent.registry.holdout_guard import assert_no_holdout_market_data
from alpha_agent.registry.identity import IDENTITY_SCHEMA
from alpha_agent.validation.fingerprint import fingerprint


class MarketWindow(BaseModel):
    """The market-data window a result was measured on, stored inclusively.

    Every field here is guarded against the locked holdout.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    label: str                                   # e.g. "VALIDATION"
    start_date: str                              # inclusive, YYYY-MM-DD
    end_date: str                                # inclusive, YYYY-MM-DD

    def identity(self) -> str:
        return fingerprint("marketwindow1", self.model_dump(mode="json"))


class ExperimentRecord(BaseModel):
    """One statistical hypothesis, executed under one full configuration.

    ``experiment_identity`` is a PRE-RUN quantity (see
    :mod:`alpha_agent.registry.identity`). ``target_schedule_hash`` and
    ``report_fingerprint`` are post-run provenance: they are recorded and they
    are part of the content fingerprint, so a conflicting value under the same
    identity fails loudly -- but they never define which experiment this is,
    and they are ``None`` whenever no committed artifact supports them.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    experiment_identity: str
    identity_schema: str = IDENTITY_SCHEMA
    experiment_id: str                           # deterministic friendly slug
    display_name: str
    created_at: str                              # provenance of the SOURCE artifact
    phase: str
    status: ExperimentStatus
    code_commit: str = ""

    root_symbol: str
    #: Phase 6 (ETF Research Pilot): a storage/query-layer domain tag -- NEVER
    #: an input to ``experiment_identity()`` (see ``alpha_agent.registry.identity``).
    #: No default -- every construction site must say explicitly which domain
    #: a record belongs to, never inferred.
    asset_domain: AssetDomain
    strategy_family: str
    strategy_fingerprint: str
    strategy_id: str
    strategy_spec_json: dict                     # canonical/neighbour params + shape
    feature_spec_fingerprint: str | None = None
    #: post-compilation audit evidence, NOT part of ``experiment_identity``;
    #: ``None`` when no committed artifact records this variant's schedule.
    target_schedule_hash: str | None = None

    candidate_manifest_fingerprint: str | None = None
    dataset_fingerprint: str
    split_identity: str
    market_window: MarketWindow
    validation_spec_fingerprint: str
    reliability_policy_fingerprint: str
    execution_config_identity: str
    cost_config_identity: str
    risk_identity: str

    trial_role: TrialRole
    parameter_variant_identity: str
    parameter_variant_label: str                 # "canonical" | "neighbour_0" | ...
    parent_experiment_identity: str | None = None
    #: post-run provenance; ``None`` when no committed artifact records it.
    report_fingerprint: str | None = None
    notes: str = ""

    @property
    def canonical_or_neighbour(self) -> str:
        """Section 4A's ``canonical_or_neighbour`` field, derived from the role
        so the two can never disagree."""
        return "canonical" if self.trial_role is TrialRole.CANONICAL else "neighbour"

    def scientific_payload(self) -> dict:
        """The immutable scientific content of this experiment.

        Excludes ``display_name``, ``notes`` and ``created_at`` -- cosmetic /
        provenance fields whose change is not a scientific conflict.
        """
        d = self.model_dump(mode="json")
        for cosmetic in ("display_name", "notes", "created_at"):
            d.pop(cosmetic, None)
        return d

    def hypothesis_payload(self) -> dict:
        """The PRE-RUN scientific hypothesis -- everything that defines *which*
        experiment this is, with every EXECUTION-ATTEMPT provenance field removed
        (schema v5). Two records with the same ``hypothesis_payload`` are the same
        hypothesis executed under (possibly) different attempts; that is not a
        conflict, it is a re-execution.
        """
        d = self.scientific_payload()
        for attempt_provenance in (
            "report_fingerprint", "target_schedule_hash", "status", "code_commit"
        ):
            d.pop(attempt_provenance, None)
        return d

    def hypothesis_fingerprint(self) -> str:
        return fingerprint("exphypothesis1", self.hypothesis_payload())

    def content_fingerprint(self) -> str:
        return fingerprint("expcontent1", self.scientific_payload())

    def assert_no_holdout(self) -> None:
        assert_no_holdout_market_data(self.market_window.model_dump(mode="json"),
                                      path=f"experiment[{self.experiment_id}].market_window")
        assert_no_holdout_market_data(self.strategy_spec_json,
                                      path=f"experiment[{self.experiment_id}].strategy_spec_json")


class ResultRecord(BaseModel):
    """The immutable adjudicated outcome of one EXECUTION ATTEMPT (section 4B).

    Schema v5: a result belongs to an execution attempt, not directly to an
    identity. ``attempt_id`` is ``None`` in a bundle whose importer does not
    manage attempts explicitly -- :meth:`ExperimentRegistry.apply_bundle` then
    binds the result to a fresh ``VALID`` attempt of ``experiment_identity``.
    ``attempt_id`` is deliberately excluded from :meth:`result_fingerprint` so
    the SAME scientific result fingerprints identically no matter which attempt
    carried it (a migrated v4 result keeps its fingerprint).
    """

    model_config = {"frozen": True, "extra": "forbid"}

    experiment_identity: str
    attempt_id: str | None = None
    headline_verdict: RegistryVerdict
    reason_codes: tuple[str, ...] = ()

    gross_pnl_usd: float | None = None
    costs_usd: float | None = None
    net_pnl_usd: float | None = None
    daily_sharpe: float | None = None
    annualized_sharpe: float | None = None

    gating_null_p: float | None = None
    bh_p_value: float | None = None
    bh_q: float | None = None
    bh_rejected_at_q: bool | None = None
    dsr_probability: float | None = None
    fold_consistency: float | None = None

    n_trades: int | None = None
    n_fills: int | None = None
    n_oos_days: int | None = None

    parameter_stability: dict = Field(default_factory=dict)
    cost_stress: dict = Field(default_factory=dict)
    bootstrap_evidence: dict = Field(default_factory=dict)
    regime_evidence: dict = Field(default_factory=dict)
    cross_market_reference: dict = Field(default_factory=dict)

    holdout_eligible: bool = False
    evidence_completeness: str = "full"          # "full" | "partial:<reason>"
    #: the committed artifact these statistics were actually read from, and its
    #: SHA-256. For a predeclared neighbour this is the trial-family table, NOT
    #: a per-neighbour ValidationReport -- there is no such report.
    source_artifact: str | None = None
    source_artifact_sha256: str | None = None

    def scientific_payload(self) -> dict:
        d = self.model_dump(mode="json")
        d.pop("attempt_id", None)                 # attempt binding is not scientific content
        return d

    def result_fingerprint(self) -> str:
        return fingerprint("expresult1", self.scientific_payload(), allow_non_finite=True)

    def assert_no_holdout(self) -> None:
        payload = self.model_dump(mode="json")
        assert_no_holdout_market_data(
            payload, path=f"result[{self.experiment_identity[:20]}]"
        )


class FailureRecord(BaseModel):
    """One typed lesson: a scientific rejection, or an engineering / data
    integrity defect and its correction (sections 4C, 5, 6)."""

    model_config = {"frozen": True, "extra": "forbid"}

    failure_id: str
    scope: FailureScope
    experiment_identity: str | None = None
    #: schema v5: the exact execution attempt this failure belongs to. ``None``
    #: for a family/system-scope failure, or when the importer does not manage
    #: attempts -- ``apply_bundle`` then binds it to the identity's latest attempt.
    attempt_id: str | None = None
    failure_class: FailureClass
    failure_code: str                            # authoritative typed code
    summary: str
    mechanism: str
    evidence: dict = Field(default_factory=dict)
    action_taken: str = ""
    resolved: bool = False
    resolution_commit: str | None = None
    superseded_by: str | None = None
    created_at: str
    root_symbol: str | None = None
    strategy_family: str | None = None

    def scientific_payload(self) -> dict:
        d = self.model_dump(mode="json")
        d.pop("created_at", None)
        d.pop("attempt_id", None)                 # attempt binding is not scientific content
        return d

    def content_fingerprint(self) -> str:
        return fingerprint("failcontent1", self.scientific_payload())

    def assert_no_holdout(self) -> None:
        assert_no_holdout_market_data(self.evidence,
                                      path=f"failure[{self.failure_id}].evidence")


class LineageEdge(BaseModel):
    """``source <relation_type> target``, e.g. corrected SUPERSEDES defective."""

    model_config = {"frozen": True, "extra": "forbid"}

    source_experiment_identity: str
    target_experiment_identity: str
    relation_type: RelationType
    note: str = ""


class ExecutionAttemptRecord(BaseModel):
    """Schema v5: one immutable execution of a pre-run ``experiment_identity``.

    The identity is the scientific hypothesis; the attempt is one run of it. An
    ``INVALID_EXECUTION`` attempt (an engineering / data-pipeline / software /
    infrastructure defect) is kept verbatim and never supplies an authoritative
    result. Execution commit / report / schedule fingerprints are ATTEMPT
    provenance and never enter ``experiment_identity``.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    experiment_identity: str
    identity_schema: str = IDENTITY_SCHEMA
    #: 1-based, dense per identity. ``None`` -> ``apply_bundle`` assigns the next.
    attempt_ordinal: int | None = None
    attempt_status: AttemptStatus
    invalidation_class: InvalidationClass | None = None
    invalidation_detail: str = ""
    invalidation_evidence: dict = Field(default_factory=dict)
    defect_resolution_commit: str | None = None
    #: attempt provenance -- never identity
    code_commit: str = ""
    engine: str = ""
    target_schedule_hash: str | None = None
    report_fingerprint: str | None = None
    source_artifact: str | None = None
    source_artifact_sha256: str | None = None
    created_at: str
    notes: str = ""

    def model_post_init(self, context: object, /) -> None:
        valid = self.attempt_status is AttemptStatus.VALID
        if valid and self.invalidation_class is not None:
            raise ValueError("a VALID attempt must not carry an invalidation_class")
        if not valid and self.invalidation_class is None:
            raise ValueError(
                "an INVALID_EXECUTION attempt must name a typed invalidation_class"
            )

    def scientific_payload(self) -> dict:
        d = self.model_dump(mode="json")
        for provenance in ("created_at", "notes"):
            d.pop(provenance, None)
        return d

    def content_fingerprint(self) -> str:
        return fingerprint("execattempt1", self.scientific_payload())

    def attempt_id(self, ordinal: int) -> str:
        """Deterministic id: identity + ordinal + attempt provenance."""
        return "attempt1:" + fingerprint(
            "execattemptid1",
            {
                "experiment_identity": self.experiment_identity,
                "attempt_ordinal": ordinal,
                "attempt_status": self.attempt_status.value,
                "code_commit": self.code_commit,
                "engine": self.engine,
                "target_schedule_hash": self.target_schedule_hash,
                "report_fingerprint": self.report_fingerprint,
                "source_artifact_sha256": self.source_artifact_sha256,
            },
        )

    def assert_no_holdout(self) -> None:
        assert_no_holdout_market_data(
            self.invalidation_evidence,
            path=f"attempt[{self.experiment_identity[:20]}].invalidation_evidence",
        )


class AttemptLineageEdge(BaseModel):
    """Attempt-layer ``source <relation> target`` (e.g. corrected attempt 2
    CORRECTS_ATTEMPT invalid attempt 1). Audit provenance only -- attempt
    authority is derived from (ordinal, status)."""

    model_config = {"frozen": True, "extra": "forbid"}

    source_experiment_identity: str
    source_attempt_ordinal: int
    target_experiment_identity: str
    target_attempt_ordinal: int
    relation_type: AttemptRelation
    note: str = ""


class SensitivityEvidence(BaseModel):
    """A robustness rerun of an EXISTING hypothesis (section 14).

    Deliberately *not* an experiment: it raises compute, never the statistical
    hypothesis count, and never enters the BH/FDR denominator.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    evidence_id: str
    experiment_identity: str
    relation_type: RelationType = RelationType.SAME_HYPOTHESIS_SENSITIVITY_OF
    kind: str                                    # e.g. "degraded_vendor_days_excluded"
    baseline_verdict: RegistryVerdict
    rerun_verdict: RegistryVerdict
    verdict_changed: bool
    metrics: dict = Field(default_factory=dict)
    in_bh_fdr_denominator: bool = False


class CrossMarketEvidenceRecord(BaseModel):
    """Descriptive aggregation over per-root canonical results (section 4D).

    Never an experiment: the roots it summarises ARE the canonical trials.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    evidence_id: str
    strategy_family: str
    status: str
    n_roots: int
    n_positive_roots: int
    max_single_root_pnl_share: float
    herfindahl_pnl: float
    concentrated_in_one_root: bool
    referenced_experiment_identities: tuple[str, ...] = ()


class ImportBundle(BaseModel):
    """Everything one deterministic import wants to write, validated as a set
    before a single row is inserted (section 22)."""

    model_config = {"frozen": True, "extra": "forbid"}

    import_id: str
    phase: str
    source_fingerprint: str
    created_at: str
    experiments: tuple[ExperimentRecord, ...] = ()
    #: schema v5: explicit execution attempts. When empty, every ``ResultRecord``
    #: in ``results`` is bound to a fresh VALID attempt of its identity.
    execution_attempts: tuple[ExecutionAttemptRecord, ...] = ()
    attempt_lineage: tuple[AttemptLineageEdge, ...] = ()
    results: tuple[ResultRecord, ...] = ()
    failures: tuple[FailureRecord, ...] = ()
    lineage: tuple[LineageEdge, ...] = ()
    sensitivity: tuple[SensitivityEvidence, ...] = ()
    cross_market: tuple[CrossMarketEvidenceRecord, ...] = ()
    metadata: dict = Field(default_factory=dict)


class SignalPathEvidenceRecord(BaseModel):
    """News Alpha Phase H (schema v7): one typed outcome of one news-alpha
    hypothesis in one research run.

    The HYPOTHESIS plane's lineage -- event -> transmission path (type, depth)
    -> asset expression -> measurement -> candidate signal -> portfolio --
    recorded as evidence and linked BY ID to the EVIDENCE plane
    (``experiment_identity``, only when a portfolio containing the signal was
    actually validated). Append-only; ``evidence_id`` is a content
    fingerprint, so recording the same run twice writes nothing new.

    Never a verdict about a mechanism: ``evidence_scope`` says what an outcome
    is about, and a PORTFOLIO-scoped SUCCESS is a fact about that portfolio
    only. Hypothesis-plane values (path types, fidelities, domains) are stored
    as their plain string values -- the registry never imports the news-alpha
    package.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    evidence_id: str
    research_run_id: str
    recorded_at: str
    event_id: str
    event_headline: str
    mechanism_graph_id: str | None = None
    path_id: str | None = None
    path_signature: str | None = None
    path_type: str | None = None
    transmission_depth: int | None = None
    consequence_state: str | None = None
    expression_id: str | None = None
    expression_concept: str | None = None
    expression_domain: str | None = None
    expression_fidelity: str | None = None
    instrument: str | None = None
    measurement_id: str | None = None
    measurement_status: str | None = None
    candidate_signal_id: str | None = None
    factor_identity: str | None = None
    portfolio_plan_fingerprint: str | None = None
    portfolio_strategy_fingerprint: str | None = None
    experiment_identity: str | None = None
    stage_reached: HypothesisStage
    outcome: PathEvidenceOutcome
    evidence_scope: EvidenceScope
    reason_code: PathEvidenceReason
    detail: str
    rule_versions: tuple[str, ...] = ()
    evidence: dict = Field(default_factory=dict)

    @staticmethod
    def content_id(payload: dict) -> str:
        body = {k: v for k, v in payload.items() if k not in ("evidence_id", "recorded_at")}
        return fingerprint("spevidence1", body)

    @classmethod
    def create(cls, **fields) -> SignalPathEvidenceRecord:
        """Build a record whose ``evidence_id`` is its own content fingerprint."""
        draft = cls(evidence_id="pending", **fields)
        return draft.model_copy(update={"evidence_id": cls.content_id(draft.model_dump(mode="json"))})

    def assert_consistent(self) -> None:
        if self.evidence_id != self.content_id(self.model_dump(mode="json")):
            raise ValueError(f"signal-path evidence {self.evidence_id} does not match its content")
        if self.outcome is PathEvidenceOutcome.SUCCESS and self.evidence_scope is not EvidenceScope.PORTFOLIO:
            raise ValueError("only a validated portfolio is a success -- never a hypothesis or a screen")
        if self.evidence_scope is EvidenceScope.PORTFOLIO and self.portfolio_strategy_fingerprint is None:
            raise ValueError("portfolio-scoped evidence names the portfolio strategy it is about")

    def assert_no_holdout(self) -> None:
        assert_no_holdout_market_data(self.evidence, path=f"signal_path_evidence[{self.evidence_id[:24]}].evidence")
