"""Runtime Research Orchestrator (Phase 18; 18.1 predeclares the family; 18.2
closes cross-generation adaptive multiple testing + binds adjudication to the
frozen ReliabilityPolicy + hardens manifest immutability).

The controlled loop that *sequences* the deterministic research services built in
Phases 13--17. It is a plain, deterministic state machine -- it owns no science.

18.1 fixes an outcome-adaptive-family defect: the whole multiple-testing family
is **fully predeclared before any member executes**.

18.2 adds three integrity guarantees:

* **One evaluation plane, one outcome-adaptive campaign.** A fixed
  ``(dataset_fingerprint, split_identity, validation_spec_fingerprint,
  reliability_policy_fingerprint)`` tuple may have only ONE executed
  research family. Post-finalization reflection may *plan* a successor family,
  but the orchestrator will not automatically *execute* it on the same
  validation evidence -- that needs a genuinely new predeclared evaluation plane
  or a separately frozen sequential-testing procedure (neither in the MVP).
* **Adjudication is the frozen policy's.** The family-wide BH threshold and the
  final verdict come from a :class:`FamilyAdjudicationService` bound to the
  actual frozen :class:`~alpha_agent.validation.policy.ReliabilityPolicy`; the
  orchestrator asserts ``policy.identity() ==
  planes.reliability_policy_fingerprint`` and never forks Phase-13 semantics.
* **Manifest immutability is verified, not assumed.** A canonical snapshot
  (strategy fingerprint, feature-set fingerprint, parameter-variant identity,
  full experiment_identity -- each *re-derived* from the actual executable
  member) is taken at planning time and re-derived before execution and
  finalization; any drift fails loudly. Underfilled families are a typed
  ``PLANNING_INCOMPLETE`` state -- never a silently-smaller tested BH family.

    PLAN  (no execution):
        research context (finalized history only)
          -> ResearchAgent            (Phase 16)  proposes a HypothesisSpec
          -> StrategyCompilerAgent    (Phase 17)  fills a closed blueprint
          -> StrategySpec                         frozen Phase 10 compiler is the gate
          -> complete PRE-RUN experiment_identity (Phase 14 formula)
          -> schema-v5 exact-duplicate / attempt-authority check
        ... repeat for every predeclared member ...
        -> freeze an immutable FamilyManifest (family_id from its content)

    EXECUTE  (only after the manifest is frozen; no reflection, no new members):
        for each frozen member -> ExecutionValidationService.run -> TrialEvidence
        raw per-trial evidence only; an INVALID_EXECUTION writes an
        ExecutionAttemptRecord + a typed FailureRecord and NOTHING scientific.

    FINALIZE  (only after the frozen family is complete):
        family-wide BH/FDR over every member's trial p-value, then the final
        per-member reliability verdict, written atomically. A family with any
        unresolved INVALID_EXECUTION member is INCOMPLETE_NOT_ADJUDICATED: no BH,
        no verdicts, no PASS, and the BH denominator is never silently shrunk.

    REFLECT  (only after a family is finalized):
        may propose a NEW family / generation with its own predeclared manifest
        and a new family_id. It never mutates a finalized family.

Invariants enforced by construction, not prompt wording:

* no result from one family member influences the selection, StrategySpec,
  identity, or configuration of another member of the same family -- the planning
  context is a pure function of *finalized* history plus inter-generation feedback;
* the strategy fingerprint is evidence only (Phase 17.2); the duplicate decision
  uses the full pre-run ``experiment_identity`` + schema-v5 attempt authority;
* an ``INVALID_EXECUTION`` attempt never creates a scientific ``ResultRecord``; a
  corrected VALID re-execution of the SAME identity may later create it;
* the LLM cannot set or see PnL / fills / risk / costs / validation thresholds /
  the BH-FDR family / the DSR benchmark;
* no arbitrary code; strategy logic only via the closed Phase 10 DSL;
* no access to 2025 -- config, evidence, and the holdout output are all guarded;
  the locked-holdout evaluation is a separate opt-in method the loop never reaches;
* every agent loop has an explicit budget and a deterministic termination reason.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from alpha_agent.agents.compiler_agent import (
    CompiledStrategyProposal,
    CompilerBudgetExceeded,
    CompilerContext,
    CompilerSchemaRetryExhausted,
    StrategyCompilerAgent,
)
from alpha_agent.agents.context import ResearchContext, build_research_context
from alpha_agent.agents.research_agent import (
    BudgetExceeded,
    ResearchAgent,
    ResearchProposal,
    SchemaRetryExhausted,
)
from alpha_agent.execution.capability import assess_cadence, resolve_cadence
from alpha_agent.registry.enums import (
    AssetDomain,
    AttemptStatus,
    ExperimentStatus,
    FailureClass,
    FailureScope,
    InvalidationClass,
    RegistryVerdict,
    TrialRole,
)
from alpha_agent.registry.failure_memory import FailureMemory, relevant_failure_memory
from alpha_agent.registry.holdout_guard import assert_no_holdout_market_data
from alpha_agent.registry.identity import (
    IDENTITY_SCHEMA,
    experiment_identity,
    friendly_experiment_id,
    parameter_variant_identity,
)
from alpha_agent.registry.models import (
    ExecutionAttemptRecord,
    ExperimentRecord,
    FailureRecord,
    ImportBundle,
    MarketWindow,
    ResultRecord,
)
from alpha_agent.registry.sqlite_registry import ExperimentRegistry, UnknownExperiment
from alpha_agent.strategy import StrategySpec, strategy_fingerprint
from alpha_agent.strategy.candidates_phase_13_5c import feature_fingerprints
from alpha_agent.validation.fdr import benjamini_hochberg_decisions
from alpha_agent.validation.fingerprint import fingerprint
from alpha_agent.validation.policy import ReliabilityPolicy

__all__ = [
    "AdaptiveTestingError",
    "EvidenceProvenanceError",
    "ExecutionValidationService",
    "FamilyAdjudication",
    "FamilyAdjudicationService",
    "FamilyManifest",
    "FamilyManifestError",
    "FamilyMember",
    "FamilyMemberResult",
    "FamilyPlanSpec",
    "FamilyReport",
    "FamilyStatus",
    "HoldoutEvaluationService",
    "HoldoutIsolationError",
    "IdentityPlanes",
    "MemberAdjudication",
    "MemberDisposition",
    "NonFamilyVerdict",
    "OrchestrationError",
    "OrchestrationReport",
    "OrchestratorBudget",
    "OrchestratorConfig",
    "PlanningDisposition",
    "PlanningSlotOutcome",
    "PolicyFamilyAdjudicator",
    "ResearchOrchestrator",
    "ScriptedExecutionValidationService",
    "TrialEvidence",
    "evaluation_plane_id",
    "feature_set_fingerprint",
]


# ==========================================================================
# errors
# ==========================================================================
class OrchestrationError(RuntimeError):
    """Base class for orchestrator control-flow failures."""


class FamilyManifestError(OrchestrationError):
    """A frozen :class:`FamilyManifest` was tampered with -- a member was added,
    removed, or altered after the family was predeclared, or a manifest this
    orchestrator never planned was handed to execution / finalization."""


class HoldoutIsolationError(OrchestrationError):
    """The locked 2025 holdout was reached without an explicit, separately-passed
    holdout service and ``allow_holdout=True``. The iteration loop can never
    raise this because it never has the arguments that would."""


class AdaptiveTestingError(OrchestrationError):
    """A second outcome-adaptive research family was about to be executed on a
    validation plane that already had one. Cross-generation adaptive
    multiple-testing inflation is refused: a successor family needs a genuinely
    new predeclared evaluation plane or a separately frozen sequential-testing
    procedure."""


class EvidenceProvenanceError(OrchestrationError):
    """A :class:`TrialEvidence` states it was produced under a validation-spec or
    reliability-policy fingerprint that is not the one the frozen manifest
    predeclared."""


# ==========================================================================
# service return: RAW per-trial evidence (no final verdict, no BH-q)
# ==========================================================================
class NonFamilyVerdict(str, Enum):
    """The verdict the frozen per-trial ``ReliabilityPolicy`` reaches for this
    trial IGNORING the family-level BH/FDR gate. The orchestrator overlays only
    the family decision at finalization -- it never re-derives the per-trial
    reliability gates."""

    ELIGIBLE = "ELIGIBLE"          # every non-family gate passed; PASS iff BH-rejected
    REJECT = "REJECT"             # a non-family statistical gate failed
    INCONCLUSIVE = "INCONCLUSIVE"  # insufficient evidence to adjudicate


class TrialEvidence(BaseModel):
    """The raw result of ONE execution attempt of one family member.

    Either the C++ Quant Core produced the official Fills / PnL and the frozen
    per-trial ``ReliabilityPolicy`` gates were evaluated (``status == VALID``),
    or the attempt was an ``INVALID_EXECUTION`` (an engineering / data-pipeline /
    software / infrastructure defect, never a scientific reason).

    It carries NO family-wide BH q-value and NO final PASS/REJECT -- those are
    computed only at family finalization, over the whole frozen family.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    status: AttemptStatus

    # -- INVALID_EXECUTION only --
    invalidation_class: InvalidationClass | None = None
    invalidation_detail: str = ""
    invalidation_evidence: dict = Field(default_factory=dict)
    defect_resolution_commit: str | None = None

    # -- VALID only: raw per-trial evidence --
    #: the one-sided p-value for positive performance that enters the BH family.
    trial_p_value: float | None = None
    #: the frozen policy's verdict IGNORING the BH/FDR gate.
    non_family_verdict: NonFamilyVerdict | None = None
    non_family_reason_codes: tuple[str, ...] = ()
    metrics: dict = Field(default_factory=dict)

    # -- attempt provenance (never enters experiment_identity) --
    engine: str = ""
    target_schedule_hash: str | None = None
    report_fingerprint: str | None = None
    source_artifact: str | None = None
    source_artifact_sha256: str | None = None
    evidence_completeness: str = "full"
    #: which validation / policy fingerprints actually produced this evidence.
    #: When set, ``execute_family`` asserts they equal the frozen manifest's
    #: planes; a real service should always stamp them.
    produced_under_validation_spec_fingerprint: str | None = None
    produced_under_reliability_policy_fingerprint: str | None = None

    @model_validator(mode="after")
    def _coherent(self) -> TrialEvidence:
        if self.status is AttemptStatus.VALID:
            if self.invalidation_class is not None:
                raise ValueError("a VALID TrialEvidence must not carry an invalidation_class")
            if self.trial_p_value is None or not (0.0 <= self.trial_p_value <= 1.0):
                raise ValueError("a VALID TrialEvidence needs a trial_p_value in [0, 1]")
            if self.non_family_verdict is None:
                raise ValueError("a VALID TrialEvidence needs a non_family_verdict")
        else:
            if self.invalidation_class is None:
                raise ValueError(
                    "an INVALID_EXECUTION TrialEvidence must name a typed invalidation_class"
                )
            if self.trial_p_value is not None or self.non_family_verdict is not None:
                raise ValueError(
                    "an INVALID_EXECUTION TrialEvidence carries no scientific evidence"
                )
        assert_no_holdout_market_data(self.model_dump(mode="json"), path="$.trial_evidence")
        return self


@runtime_checkable
class ExecutionValidationService(Protocol):
    """The deterministic "C++ Quant Core + frozen per-trial reliability gates"
    step. It returns RAW evidence for one member; it never computes the
    family-wide BH q-value or the final PASS/REJECT."""

    def run(
        self,
        *,
        member: FamilyMember,
        manifest: FamilyManifest,
        attempt_ordinal: int,
        holdout: bool,
    ) -> TrialEvidence: ...


@runtime_checkable
class HoldoutEvaluationService(Protocol):
    """Separate service for the one-time locked-holdout evaluation. Deliberately
    NOT the same shape as :class:`ExecutionValidationService` so it cannot be
    passed where the iteration loop would use it."""

    def evaluate_holdout(
        self, *, strategy_spec: StrategySpec, experiment_identity: str, root_symbol: str
    ) -> TrialEvidence: ...


class ScriptedExecutionValidationService:
    """Deterministic test double. Returns one canned :class:`TrialEvidence` per
    call -- by call order, by a per-identity mapping (value may be a list popped
    per call, e.g. INVALID then a corrected VALID), or from a callable."""

    def __init__(
        self,
        evidence: Sequence[TrialEvidence]
        | dict[str, TrialEvidence | Sequence[TrialEvidence]]
        | Callable[[dict], TrialEvidence],
    ):
        self._evidence = evidence
        self.calls: list[dict] = []
        self._cursor = 0
        self._per_identity: dict[str, int] = {}

    def run(
        self,
        *,
        member: FamilyMember,
        manifest: FamilyManifest,
        attempt_ordinal: int,
        holdout: bool,
    ) -> TrialEvidence:
        call = {
            "experiment_identity": member.experiment_identity,
            "family_id": manifest.family_id,
            "root_symbol": member.root_symbol,
            "strategy_family": member.strategy_family,
            "params": dict(member.params),
            "attempt_ordinal": attempt_ordinal,
            "holdout": holdout,
            "strategy_fingerprint": member.strategy_fingerprint,
        }
        self.calls.append(call)
        if holdout:  # pragma: no cover - the loop never sets this
            raise HoldoutIsolationError("the scripted execution service is not a holdout service")
        if callable(self._evidence):
            return self._evidence(call)
        if isinstance(self._evidence, dict):
            item = self._evidence[member.experiment_identity]
            if isinstance(item, TrialEvidence):
                return item
            i = self._per_identity.get(member.experiment_identity, 0)
            self._per_identity[member.experiment_identity] = i + 1
            return item[min(i, len(item) - 1)]
        if self._cursor >= len(self._evidence):
            raise AssertionError(
                f"ScriptedExecutionValidationService exhausted after {self._cursor} "
                "call(s); the orchestrator executed more attempts than scripted"
            )
        item = self._evidence[self._cursor]
        self._cursor += 1
        return item

    @property
    def call_count(self) -> int:
        return len(self.calls)


# ==========================================================================
# family adjudication -- owned by the frozen ReliabilityPolicy, not the orchestrator
# ==========================================================================
class MemberAdjudication(BaseModel):
    """The family-layer decision for one member: its BH q-value and the final
    verdict the frozen policy reaches once the family BH gate is applied."""

    model_config = {"frozen": True, "extra": "forbid"}

    experiment_identity: str
    trial_role: TrialRole
    bh_p_value: float
    bh_q_value: float
    bh_rejected_at_q: bool
    final_verdict: RegistryVerdict
    reason_codes: tuple[str, ...] = ()


class FamilyAdjudication(BaseModel):
    """The complete family-wide adjudication. Records which policy identity and
    which q-threshold produced it -- provenance the orchestrator cannot fake."""

    model_config = {"frozen": True, "extra": "forbid"}

    policy_identity: str
    fdr_q_threshold: float
    n_members: int
    n_bh_rejected: int
    member_adjudications: tuple[MemberAdjudication, ...]

    def by_identity(self) -> dict[str, MemberAdjudication]:
        return {a.experiment_identity: a for a in self.member_adjudications}


@runtime_checkable
class FamilyAdjudicationService(Protocol):
    """Owns the family-wide BH/FDR + final-verdict boundary. The orchestrator
    sequences it; it never reimplements the policy."""

    def policy_identity(self) -> str: ...

    def fdr_q_threshold(self) -> float: ...

    def adjudicate_family(
        self, *, members: Sequence[FamilyMember], latest_evidence: dict[str, TrialEvidence]
    ) -> FamilyAdjudication: ...


class PolicyFamilyAdjudicator:
    """The default :class:`FamilyAdjudicationService`, bound to the actual frozen
    :class:`~alpha_agent.validation.policy.ReliabilityPolicy`.

    It reads two declared thresholds from the policy -- ``fdr_q_threshold`` and
    ``fdr_require_canonical_rejected`` -- and runs the SHARED
    :func:`benjamini_hochberg_decisions`. It does not fork any other Phase-13
    gate: every non-family gate was already evaluated by the execution service
    and arrives as :class:`NonFamilyVerdict`.
    """

    def __init__(self, policy: ReliabilityPolicy):
        if not isinstance(policy, ReliabilityPolicy):
            raise TypeError("policy must be a ReliabilityPolicy")
        self._policy = policy

    def policy_identity(self) -> str:
        return self._policy.identity()

    def fdr_q_threshold(self) -> float:
        return float(self._policy.fdr_q_threshold)

    def adjudicate_family(
        self, *, members: Sequence[FamilyMember], latest_evidence: dict[str, TrialEvidence]
    ) -> FamilyAdjudication:
        ordered = list(members)
        p_values = [float(latest_evidence[m.experiment_identity].trial_p_value) for m in ordered]
        fdr = benjamini_hochberg_decisions(
            p_values,
            self._policy.fdr_q_threshold,
            labels=[m.experiment_identity for m in ordered],
        )
        adj: list[MemberAdjudication] = []
        for m, decision in zip(ordered, fdr.decisions, strict=True):
            ev = latest_evidence[m.experiment_identity]
            verdict, reasons = self._overlay(m, ev, bh_rejected=decision.rejected)
            adj.append(
                MemberAdjudication(
                    experiment_identity=m.experiment_identity,
                    trial_role=m.trial_role,
                    bh_p_value=decision.p_value,
                    bh_q_value=decision.q_value,
                    bh_rejected_at_q=decision.rejected,
                    final_verdict=verdict,
                    reason_codes=reasons,
                )
            )
        return FamilyAdjudication(
            policy_identity=self._policy.identity(),
            fdr_q_threshold=float(self._policy.fdr_q_threshold),
            n_members=len(ordered),
            n_bh_rejected=fdr.n_rejected,
            member_adjudications=tuple(adj),
        )

    def _overlay(
        self, member: FamilyMember, ev: TrialEvidence, *, bh_rejected: bool
    ) -> tuple[RegistryVerdict, tuple[str, ...]]:
        if ev.non_family_verdict is NonFamilyVerdict.INCONCLUSIVE:
            return RegistryVerdict.INCONCLUSIVE, (
                ev.non_family_reason_codes or ("INSUFFICIENT_EVIDENCE",)
            )
        if ev.non_family_verdict is NonFamilyVerdict.REJECT:
            return RegistryVerdict.REJECT, (
                ev.non_family_reason_codes or ("NON_FAMILY_GATE_FAILED",)
            )
        # ELIGIBLE -- the family BH gate decides
        if member.trial_role is not TrialRole.CANONICAL:
            return RegistryVerdict.NOT_ADJUDICATED, ()
        if self._policy.fdr_require_canonical_rejected and not bh_rejected:
            return RegistryVerdict.REJECT, ("FDR_NOT_SIGNIFICANT",)
        return RegistryVerdict.PASS, ("ALL_GATES_SATISFIED",)


# ==========================================================================
# frozen configuration
# ==========================================================================
class IdentityPlanes(BaseModel):
    """The pre-run identity planes the orchestrator does NOT get to choose. Each
    is frozen by an earlier phase (dataset acquisition, split plan,
    ``ValidationSpec``, ``ReliabilityPolicy``, the family-wide execution / cost /
    risk configuration). The orchestrator passes them straight into the
    ``experiment_identity`` formula -- an LLM never touches them."""

    model_config = {"frozen": True, "extra": "forbid"}

    dataset_fingerprint: str
    split_identity: str
    validation_spec_fingerprint: str
    reliability_policy_fingerprint: str
    execution_config_identity: str
    cost_config_identity: str
    risk_identity: str

    def canonical(self) -> dict:
        return self.model_dump(mode="json")

    def evaluation_plane(self) -> dict:
        """The four fingerprints that define one outcome-adaptive testing plane."""
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_identity": self.split_identity,
            "validation_spec_fingerprint": self.validation_spec_fingerprint,
            "reliability_policy_fingerprint": self.reliability_policy_fingerprint,
        }


def evaluation_plane_id(planes: IdentityPlanes) -> str:
    """A fixed ``(dataset, split, validation_spec, reliability_policy)`` tuple may
    have only ONE outcome-adaptive executed research family."""
    return fingerprint("evalplane1", planes.evaluation_plane())


class FamilyPlanSpec(BaseModel):
    """How one predeclared multiple-testing family is planned. Every field is
    fixed before any member executes. The BH q-threshold is NOT here -- it is
    derived from the frozen :class:`ReliabilityPolicy` the orchestrator is bound
    to, so there is exactly one source of truth."""

    model_config = {"frozen": True, "extra": "forbid"}

    family_stem: str
    #: the predeclared BH-FDR family size. If planning cannot fill it the family
    #: is a typed PLANNING_INCOMPLETE state -- never a silently smaller family.
    target_family_size: int = Field(ge=1)
    #: a prior REJECT/INCONCLUSIVE at/above this transparent similarity score,
    #: re-proposed with no stated novelty, is not admitted to the family.
    near_duplicate_block_threshold: float = Field(default=0.88, ge=0.0, le=1.0)


class OrchestratorBudget(BaseModel):
    """Explicit termination budgets. Every agent loop is bounded."""

    model_config = {"frozen": True, "extra": "forbid"}

    #: ResearchAgent.propose calls allowed while planning ONE family (>= target
    #: size, to tolerate rejected / duplicate slots).
    max_planning_attempts_per_family: int = Field(default=12, ge=1)
    #: execution attempts allowed for one identity. Re-execution is only ever
    #: after an INVALID_EXECUTION attempt. 2 => one run + one engineering re-run.
    max_execution_attempts_per_identity: int = Field(default=2, ge=1)
    #: schema-retry budget handed to the two LLM agents.
    agent_schema_retries: int = Field(default=2, ge=0)
    #: after the first family is finalized, PLAN (never execute) one successor
    #: family so a human can see what reflection proposes. It is returned as
    #: PLANNED_NOT_EXECUTED and can be scientifically executed only under a new
    #: evaluation plane.
    plan_successor_family_after_finalization: bool = False


class OrchestratorConfig(BaseModel):
    """The complete, frozen configuration for one orchestration run."""

    model_config = {"frozen": True, "extra": "forbid"}

    objective: str
    market_universe: tuple[str, ...] = Field(min_length=1)
    market_window: MarketWindow
    planes: IdentityPlanes
    family_plan: FamilyPlanSpec
    budget: OrchestratorBudget = OrchestratorBudget()
    knowledge_base: tuple[str, ...] = ()
    phase: str = "18.1"
    code_commit: str = ""

    @model_validator(mode="after")
    def _guard(self) -> OrchestratorConfig:
        assert_no_holdout_market_data(
            self.model_dump(mode="json"),
            path="$.orchestrator_config",
            observational_context_paths=frozenset({"$.orchestrator_config.knowledge_base"}),
        )
        return self


# ==========================================================================
# predeclared family -- the immutable manifest
# ==========================================================================
class FamilyMember(BaseModel):
    """One predeclared member of a multiple-testing family. Every field is a
    PRE-RUN quantity -- known before the member (or any sibling) executes."""

    model_config = {"frozen": True, "extra": "forbid"}

    ordinal: int = Field(ge=0)
    experiment_identity: str
    strategy_fingerprint: str
    strategy_id: str
    strategy_family: str
    root_symbol: str
    params: dict = Field(default_factory=dict)
    parameter_variant_identity: str
    feature_spec_fingerprint: str
    trial_role: TrialRole = TrialRole.CANONICAL
    hypothesis_id: str
    hypothesis_title: str
    strategy_spec: StrategySpec
    strategy_spec_json: dict
    signal_cadence: str = ""
    execution_cadence: str = ""
    #: Alpha Discovery live-research campaign, Checkpoint 9 -- pure lineage
    #: (which `research_knowledge_base` snippets the ResearchAgent said
    #: inspired this proposal, `HypothesisSpec.source_inspirations` passed
    #: through unchanged). NOT part of `identity_tuple()` / `_manifest_
    #: snapshot()` -- it is display-only provenance, never a scientific
    #: identity component, and mutating it is not tamper-checked (unlike
    #: every field that already IS covered there).
    source_inspirations: tuple[str, ...] = ()

    def identity_tuple(self) -> list:
        return [
            self.ordinal,
            self.experiment_identity,
            self.strategy_fingerprint,
            self.parameter_variant_identity,
            self.trial_role.value,
            self.root_symbol,
        ]


class FamilyStatus(str, Enum):
    PLANNED = "PLANNED"
    PLANNING_INCOMPLETE = "PLANNING_INCOMPLETE"   # could not fill target_family_size
    PLANNED_NOT_EXECUTED = "PLANNED_NOT_EXECUTED"  # a reflection successor; plane already used
    EXECUTING = "EXECUTING"
    FINALIZED = "FINALIZED"
    INCOMPLETE_NOT_ADJUDICATED = "INCOMPLETE_NOT_ADJUDICATED"
    EMPTY = "EMPTY"


def _compute_family_id(
    *,
    family_stem: str,
    generation: int,
    parent_family_id: str | None,
    planes: IdentityPlanes,
    market_window: MarketWindow,
    fdr_q_threshold: float,
    policy_identity: str,
    declared_target_size: int,
    members: Sequence[FamilyMember],
) -> str:
    return fingerprint(
        "familymanifest1",
        {
            "schema": "family-manifest/2",
            "family_stem": family_stem,
            "generation": generation,
            "parent_family_id": parent_family_id or "",
            "planes": planes.canonical(),
            "market_window": market_window.model_dump(mode="json"),
            "fdr_q_threshold": fdr_q_threshold,
            "policy_identity": policy_identity,
            "declared_target_size": declared_target_size,
            "members": [m.identity_tuple() for m in members],
        },
    )


class FamilyManifest(BaseModel):
    """The immutable, fully-predeclared multiple-testing family. Frozen BEFORE
    any member executes. ``family_id`` is a content hash: a member added,
    removed, or altered after freezing changes it, and
    :meth:`assert_consistent` catches a tampered copy."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "family-manifest/2"
    family_id: str
    family_stem: str
    generation: int = Field(ge=0)
    parent_family_id: str | None = None
    planes: IdentityPlanes
    market_window: MarketWindow
    fdr_q_threshold: float = Field(gt=0.0, lt=1.0)
    #: == planes.reliability_policy_fingerprint == the bound policy's identity().
    policy_identity: str
    #: the scientifically declared family size (FamilyPlanSpec.target_family_size).
    declared_target_size: int = Field(ge=1)
    members: tuple[FamilyMember, ...]
    planning_log: tuple[PlanningSlotOutcome, ...] = ()
    created_at: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> FamilyManifest:
        expected = _compute_family_id(
            family_stem=self.family_stem,
            generation=self.generation,
            parent_family_id=self.parent_family_id,
            planes=self.planes,
            market_window=self.market_window,
            fdr_q_threshold=self.fdr_q_threshold,
            policy_identity=self.policy_identity,
            declared_target_size=self.declared_target_size,
            members=self.members,
        )
        if self.family_id != expected:
            raise FamilyManifestError(
                f"family_id {self.family_id} does not match the manifest content "
                f"({expected}); a member was added, removed, or altered"
            )
        if self.policy_identity != self.planes.reliability_policy_fingerprint:
            raise FamilyManifestError(
                "manifest policy_identity does not equal planes.reliability_policy_fingerprint"
            )
        if [m.ordinal for m in self.members] != list(range(len(self.members))):
            raise FamilyManifestError("family members must be densely ordinal-numbered from 0")
        if len({m.experiment_identity for m in self.members}) != len(self.members):
            raise FamilyManifestError("a family may not contain the same experiment_identity twice")
        if len(self.members) > self.declared_target_size:
            raise FamilyManifestError("a family may not exceed its declared target size")
        # `created_at` is ordinary wall-clock bookkeeping (when this manifest was
        # built), never market/performance data; the guard's default
        # `BOOKKEEPING_TIMESTAMP_KEYS` already excludes it (see holdout_guard's
        # module docstring, defect 3) -- without that, this would reject every
        # manifest built on or after 2025-01-01 wall-clock, which is not what
        # it is for.
        assert_no_holdout_market_data(self.model_dump(mode="json"), path="$.family_manifest")
        return self

    @property
    def predeclared_family_size(self) -> int:
        return len(self.members)

    @property
    def planning_complete(self) -> bool:
        """True iff planning filled the scientifically declared family size. An
        underfilled family is never executed as a smaller BH family."""
        return len(self.members) == self.declared_target_size

    def assert_consistent(self) -> None:
        """Re-run the content check. A ``model_copy(update=...)`` skips validation,
        so this is the guard called at every execution / finalization boundary."""
        FamilyManifest.model_validate(self.model_dump(mode="json"))

    def manifest_fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("created_at", None)
        payload.pop("planning_log", None)
        return fingerprint("familymanifestcontent1", payload)

    def member_identities(self) -> frozenset[str]:
        return frozenset(m.experiment_identity for m in self.members)

    def evaluation_plane_id(self) -> str:
        return evaluation_plane_id(self.planes)


# ==========================================================================
# planning + execution + finalization outcomes
# ==========================================================================
class PlanningDisposition(str, Enum):
    MEMBER_ADDED = "MEMBER_ADDED"
    RESEARCH_SCHEMA_EXHAUSTED = "RESEARCH_SCHEMA_EXHAUSTED"
    RESEARCH_BUDGET_EXCEEDED = "RESEARCH_BUDGET_EXCEEDED"
    RESEARCH_GUARDRAIL_REJECTED = "RESEARCH_GUARDRAIL_REJECTED"
    COMPILER_SCHEMA_EXHAUSTED = "COMPILER_SCHEMA_EXHAUSTED"
    COMPILER_BUDGET_EXCEEDED = "COMPILER_BUDGET_EXCEEDED"
    COMPILER_REJECTED = "COMPILER_REJECTED"
    DUPLICATE_AUTHORITATIVE_RESULT = "DUPLICATE_AUTHORITATIVE_RESULT"
    DUPLICATE_WITHIN_MANIFEST = "DUPLICATE_WITHIN_MANIFEST"
    NEAR_DUPLICATE_NO_NOVELTY = "NEAR_DUPLICATE_NO_NOVELTY"


class MemberDisposition(str, Enum):
    VALID_EVIDENCE = "VALID_EVIDENCE"
    INVALID_EXECUTION_UNRESOLVED = "INVALID_EXECUTION_UNRESOLVED"
    INVALID_THEN_CORRECTED = "INVALID_THEN_CORRECTED"


class PlanningSlotOutcome(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    slot: int
    disposition: PlanningDisposition
    detail: str = ""
    hypothesis_id: str | None = None
    experiment_identity: str | None = None
    strategy_fingerprint: str | None = None
    research_attempts: int = 0
    compiler_attempts: int = 0


class FamilyMemberResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    ordinal: int
    experiment_identity: str
    experiment_id: str
    trial_role: TrialRole
    disposition: MemberDisposition
    attempt_status: AttemptStatus
    attempt_ordinal: int
    n_execution_attempts: int
    trial_p_value: float | None = None
    bh_q_value: float | None = None
    bh_rejected_at_q: bool | None = None
    final_verdict: RegistryVerdict | None = None
    reason_codes: tuple[str, ...] = ()
    holdout_eligible: bool = False


class FamilyReport(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "family-report/2"
    family_id: str
    family_stem: str
    generation: int
    parent_family_id: str | None = None
    manifest_fingerprint: str
    status: FamilyStatus
    predeclared_family_size: int
    declared_target_size: int = 0
    fdr_q_threshold: float
    policy_identity: str = ""
    evaluation_plane_id: str = ""
    validation_spec_fingerprint: str = ""
    n_bh_rejected: int = 0
    verdict_counts: dict[str, int] = Field(default_factory=dict)
    member_results: tuple[FamilyMemberResult, ...] = ()
    unresolved_invalid_identities: tuple[str, ...] = ()
    passing_experiment_ids: tuple[str, ...] = ()
    planning_dispositions: dict[str, int] = Field(default_factory=dict)
    detail: str = ""

    @property
    def is_finalized(self) -> bool:
        return self.status is FamilyStatus.FINALIZED


class OrchestrationReport(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "orchestration-report/2"
    objective: str
    phase: str
    identity_schema: str = IDENTITY_SCHEMA
    generations: tuple[FamilyReport, ...]
    n_generations: int
    total_members_planned: int
    total_members_finalized: int
    termination_reason: str
    passing_experiment_ids: tuple[str, ...] = ()
    registry_content_digest_before: str = ""
    registry_content_digest_after: str = ""

    @property
    def pass_candidates_for_holdout(self) -> tuple[str, ...]:
        return self.passing_experiment_ids


# ==========================================================================
# helpers
# ==========================================================================
def feature_set_fingerprint(spec: StrategySpec) -> str:
    """The variant's OWN feature-set fingerprint, using the frozen Phase 13.5C
    per-feature rule wrapped exactly as the registry importer wraps it."""
    return fingerprint("featset1", {"features": list(feature_fingerprints(spec))})


def _strategy_family_label(compiled: CompiledStrategyProposal) -> str:
    if compiled.family_key:
        return compiled.family_key
    kinds = compiled.blueprint_feature_kinds or (
        tuple(compiled.feature_coverage.referenced_kinds)
        if compiled.feature_coverage
        else ()
    )
    body = "+".join(sorted(kinds)) if kinds else "composite"
    # Alpha Discovery live-research campaign continuation: a blueprint now
    # declares its OWN real `signal_cadence` (compiler_agent.StrategyBlueprint,
    # replacing the old placeholder string every non-template build used to
    # get). Cadence changes which schedule builder runs (daily aggregation vs
    # native 1-minute) and therefore changes realized trades/PnL for an
    # otherwise-identical feature composition. `strategy_family` already flows
    # into `experiment_identity` (`alpha_agent.registry.identity.
    # experiment_identity`), so cadence must be part of THIS label -- two
    # blueprints with the same features but a different cadence must never
    # collide into one scientific identity. The TEMPLATE branch above is
    # returned unconditionally before this point, so every already-committed
    # registry `experiment_identity` for tsmom/ma_trend/breakout/
    # mean_reversion/silver_bullet is byte-identical; no blueprint has ever
    # produced a VALID authoritative registry result under the old label
    # (every blueprint died at UNSUPPORTED_CADENCE before this fix), so this
    # is additive, not a retroactive change to any real historical row.
    cadence = compiled.execution_semantics.signal_cadence if compiled.execution_semantics else ""
    return f"dsl:{cadence or 'unknown_cadence'}:{body}"


def _registry_strategy_spec_json(
    *, compiled: CompiledStrategyProposal, spec: StrategySpec, family: str, params: dict
) -> dict:
    sem = compiled.execution_semantics
    return {
        "schema": "registry-strategy-spec/1",
        "strategy_family": family,
        "strategy_id": spec.strategy_id,
        "build_mode": compiled.build_mode,
        "params": dict(params),
        "parameter_names": sorted(str(k) for k in params),
        "feature_fingerprints": list(feature_fingerprints(spec)),
        "signal_cadence": sem.signal_cadence if sem else "",
        "execution_cadence": sem.execution_cadence if sem else "",
    }


def _normalize_params(params: dict) -> dict:
    """Canonicalise a template's numeric params to the same shape the frozen
    registry importer uses: an integer-valued float becomes an ``int`` so the
    ``parameter_variant_identity`` fingerprint matches a deterministic rebuild."""
    out: dict = {}
    for k, v in params.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, float) and v.is_integer():
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _flt(v: object) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _int(v: object) -> int | None:
    return int(v) if isinstance(v, int) and not isinstance(v, bool) else None


_SCIENTIFIC_FAILURE_CLASS = {
    RegistryVerdict.REJECT: FailureClass.SCIENTIFIC_REJECTION,
    RegistryVerdict.INCONCLUSIVE: FailureClass.STATISTICAL_INCONCLUSIVE,
}


# ==========================================================================
# the orchestrator
# ==========================================================================
class ResearchOrchestrator:
    """Deterministic sequencer for the Phase 13--17 research services.

    ``plan_family`` -> ``execute_family`` -> ``finalize_family`` is the unit of
    work; ``run`` drives one or more generations of it with inter-generation
    reflection.
    """

    def __init__(
        self,
        *,
        registry: ExperimentRegistry,
        research_agent: ResearchAgent,
        compiler_agent: StrategyCompilerAgent,
        execution_service: ExecutionValidationService,
        config: OrchestratorConfig,
        reliability_policy: ReliabilityPolicy | None = None,
        adjudicator: FamilyAdjudicationService | None = None,
        feature_registry: object | None = None,
    ):
        if not isinstance(registry, ExperimentRegistry):
            raise TypeError("registry must be an ExperimentRegistry")
        if not isinstance(research_agent, ResearchAgent):
            raise TypeError("research_agent must be a ResearchAgent")
        if not isinstance(compiler_agent, StrategyCompilerAgent):
            raise TypeError("compiler_agent must be a StrategyCompilerAgent")
        if not isinstance(execution_service, ExecutionValidationService):
            raise TypeError(
                "execution_service must implement the ExecutionValidationService protocol"
            )
        if isinstance(execution_service, HoldoutEvaluationService):
            raise TypeError(
                "the iteration execution_service must not also be a HoldoutEvaluationService"
            )

        # family adjudication is owned by the frozen ReliabilityPolicy.
        if adjudicator is None:
            if reliability_policy is None:
                raise TypeError(
                    "pass reliability_policy (a frozen ReliabilityPolicy) or an explicit "
                    "FamilyAdjudicationService"
                )
            adjudicator = PolicyFamilyAdjudicator(reliability_policy)
        elif reliability_policy is not None:
            raise TypeError("pass exactly one of reliability_policy / adjudicator")
        if not isinstance(adjudicator, FamilyAdjudicationService):
            raise TypeError("adjudicator must implement the FamilyAdjudicationService protocol")
        if adjudicator.policy_identity() != config.planes.reliability_policy_fingerprint:
            raise OrchestrationError(
                "the family adjudicator's policy identity "
                f"{adjudicator.policy_identity()} does not equal "
                f"planes.reliability_policy_fingerprint "
                f"{config.planes.reliability_policy_fingerprint}; the FDR q-threshold and "
                "the final verdict must come from the frozen policy the identity names"
            )

        self._registry = registry
        self._failure_memory = FailureMemory(registry)
        self._research = research_agent
        self._compiler = compiler_agent
        self._exec = execution_service
        self._adjudicator = adjudicator
        self._config = config
        self._feature_registry = feature_registry

        self._planes_fp = fingerprint("orchplanes1", config.planes.canonical())
        self._plan_fp = fingerprint("orchplan1", config.family_plan.model_dump(mode="json"))
        self._evaluation_plane_id = evaluation_plane_id(config.planes)

        # run-scoped, deterministic given the scripted inputs
        self._planned: dict[str, FamilyManifest] = {}
        self._planned_snapshots: dict[str, dict] = {}
        self._executing: set[str] = set()
        self._finalized: dict[str, FamilyReport] = {}
        self._touched: list[tuple[str, str]] = []
        self._generation_feedback: list[str] = []
        self._generation_avoid: list[str] = []
        # one outcome-adaptive executed research family per evaluation plane.
        self._executed_planes: set[str] = self._planes_already_used()

    def _planes_already_used(self) -> set[str]:
        """Evaluation planes that already carry an executed research family in the
        registry -- a second adaptive campaign on any of them is refused."""
        used: set[str] = set()
        target = self._config.planes.evaluation_plane()
        for view in self._registry.experiments(authoritative_only=False):
            e = view.experiment
            plane = {
                "dataset_fingerprint": e.dataset_fingerprint,
                "split_identity": e.split_identity,
                "validation_spec_fingerprint": e.validation_spec_fingerprint,
                "reliability_policy_fingerprint": e.reliability_policy_fingerprint,
            }
            if plane == target:
                used.add(fingerprint("evalplane1", plane))
        return used

    # -- public API ------------------------------------------------------
    @property
    def config(self) -> OrchestratorConfig:
        return self._config

    def run(self) -> OrchestrationReport:
        """Plan, execute and finalize exactly ONE outcome-adaptive research family
        on this evaluation plane. Post-finalization reflection may PLAN one
        successor family (never executed here)."""
        cfg = self._config
        digest_before = self._registry.content_digest()
        reports: list[FamilyReport] = []

        self._assert_frozen_planes()
        manifest = self.plan_family(generation=0, parent_family_id=None)

        if not manifest.members:
            reports.append(_empty_family_report(manifest))
            termination = "planning:no_admissible_members"
        elif not manifest.planning_complete:
            reports.append(_planning_incomplete_report(manifest))
            termination = "planning:underfilled_family"
        else:
            evidence = self.execute_family(manifest)
            report = self.finalize_family(manifest, evidence)
            reports.append(report)

            if report.status is FamilyStatus.INCOMPLETE_NOT_ADJUDICATED:
                termination = "family:incomplete_not_adjudicated"
            elif cfg.budget.plan_successor_family_after_finalization:
                self._reflect_after_finalization(report)
                successor = self.plan_family(
                    generation=1, parent_family_id=manifest.family_id
                )
                if not successor.members:
                    reports.append(_empty_family_report(successor))
                elif not successor.planning_complete:
                    reports.append(_planning_incomplete_report(successor))
                else:
                    reports.append(_planned_not_executed_report(successor))
                termination = "reflection:successor_family_planned_not_executed"
            else:
                termination = "family:finalized"

        digest_after = self._registry.content_digest()
        finalized = [r for r in reports if r.status is FamilyStatus.FINALIZED]
        passing = tuple(pid for r in finalized for pid in r.passing_experiment_ids)
        return OrchestrationReport(
            objective=cfg.objective,
            phase=cfg.phase,
            generations=tuple(reports),
            n_generations=len(reports),
            total_members_planned=sum(r.predeclared_family_size for r in reports),
            total_members_finalized=sum(r.predeclared_family_size for r in finalized),
            termination_reason=termination,
            passing_experiment_ids=passing,
            registry_content_digest_before=digest_before,
            registry_content_digest_after=digest_after,
        )

    def planned_family(self, family_id: str) -> FamilyManifest:
        """The frozen manifest for a family this orchestrator planned."""
        m = self._planned.get(family_id)
        if m is None:
            raise FamilyManifestError(f"no family {family_id} was planned by this orchestrator")
        return m

    # -- PLAN -----------------------------------------------------------
    def plan_family(
        self, *, generation: int = 0, parent_family_id: str | None = None
    ) -> FamilyManifest:
        """Predeclare a whole multiple-testing family. Runs the ResearchAgent and
        the StrategyCompilerAgent and assembles a complete pre-run
        ``experiment_identity`` for every member. Executes NOTHING."""
        self._assert_frozen_planes()
        cfg = self._config
        spec = cfg.family_plan
        members: list[FamilyMember] = []
        log: list[PlanningSlotOutcome] = []
        seen: set[str] = set()

        # the planning context is a PURE FUNCTION of finalized history + the
        # inter-generation feedback: it is built once and never re-derived from a
        # within-family outcome (there are none).
        base_context = self._planning_context()

        for slot in range(cfg.budget.max_planning_attempts_per_family):
            if len(members) >= spec.target_family_size:
                break
            outcome, member = self._plan_one_slot(base_context, slot=slot, seen=seen)
            log.append(outcome)
            if member is not None:
                members.append(member)
                seen.add(member.experiment_identity)

        ordered = tuple(
            m.model_copy(update={"ordinal": i}) for i, m in enumerate(members)
        )
        fdr_q = self._adjudicator.fdr_q_threshold()
        policy_id = self._adjudicator.policy_identity()
        family_id = _compute_family_id(
            family_stem=spec.family_stem,
            generation=generation,
            parent_family_id=parent_family_id,
            planes=cfg.planes,
            market_window=cfg.market_window,
            fdr_q_threshold=fdr_q,
            policy_identity=policy_id,
            declared_target_size=spec.target_family_size,
            members=ordered,
        )
        manifest = FamilyManifest(
            family_id=family_id,
            family_stem=spec.family_stem,
            generation=generation,
            parent_family_id=parent_family_id,
            planes=cfg.planes,
            market_window=cfg.market_window,
            fdr_q_threshold=fdr_q,
            policy_identity=policy_id,
            declared_target_size=spec.target_family_size,
            members=ordered,
            planning_log=tuple(log),
            created_at=datetime.now(UTC).isoformat(),
        )
        self._planned[manifest.family_id] = manifest
        # an INDEPENDENT canonical snapshot, re-derived from each executable
        # member -- not a reference to the (nested-mutable) manifest object.
        self._planned_snapshots[manifest.family_id] = self._manifest_snapshot(manifest)
        return manifest

    def _plan_one_slot(
        self, context: ResearchContext, *, slot: int, seen: set[str]
    ) -> tuple[PlanningSlotOutcome, FamilyMember | None]:
        def slot_out(disp: PlanningDisposition, **kw) -> PlanningSlotOutcome:
            return PlanningSlotOutcome(slot=slot, disposition=disp, **kw)

        try:
            proposal: ResearchProposal = self._research.propose(context)
        except SchemaRetryExhausted as exc:
            return slot_out(
                PlanningDisposition.RESEARCH_SCHEMA_EXHAUSTED,
                detail=str(exc),
                research_attempts=len(exc.attempts),
            ), None
        except BudgetExceeded as exc:
            return slot_out(PlanningDisposition.RESEARCH_BUDGET_EXCEEDED, detail=str(exc)), None

        research_attempts = proposal.prompt_log.n_attempts
        if not proposal.accepted or proposal.hypothesis is None:
            return slot_out(
                PlanningDisposition.RESEARCH_GUARDRAIL_REJECTED,
                detail=f"{proposal.rejection_code}: {proposal.rejection_detail}",
                research_attempts=research_attempts,
            ), None
        hypothesis = proposal.hypothesis

        compiler_context = self._build_compiler_context()
        try:
            compiled = self._compiler.compile_hypothesis(hypothesis, compiler_context)
        except CompilerSchemaRetryExhausted as exc:
            return slot_out(
                PlanningDisposition.COMPILER_SCHEMA_EXHAUSTED,
                detail=str(exc),
                hypothesis_id=hypothesis.hypothesis_id,
                research_attempts=research_attempts,
                compiler_attempts=len(exc.attempts),
            ), None
        except CompilerBudgetExceeded as exc:
            return slot_out(
                PlanningDisposition.COMPILER_BUDGET_EXCEEDED,
                detail=str(exc),
                hypothesis_id=hypothesis.hypothesis_id,
                research_attempts=research_attempts,
            ), None

        compiler_attempts = compiled.prompt_log.n_attempts
        base = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "research_attempts": research_attempts,
            "compiler_attempts": compiler_attempts,
        }
        if not compiled.accepted or compiled.strategy_spec is None:
            return slot_out(
                PlanningDisposition.COMPILER_REJECTED,
                detail=f"{compiled.rejection_code}: {compiled.rejection_detail}",
                **base,
            ), None

        spec = compiled.strategy_spec
        family = _strategy_family_label(compiled)
        root = compiled.root_symbol or spec.root_symbol
        params = _normalize_params(compiled.resolved_params)
        feature_fp = feature_set_fingerprint(spec)
        pvi = parameter_variant_identity(params)
        identity = experiment_identity(
            strategy_fingerprint=compiled.strategy_fingerprint,
            strategy_family=family,
            root_symbol=root,
            parameter_variant_identity=pvi,
            dataset_fingerprint=self._config.planes.dataset_fingerprint,
            split_identity=self._config.planes.split_identity,
            validation_spec_fingerprint=self._config.planes.validation_spec_fingerprint,
            reliability_policy_fingerprint=self._config.planes.reliability_policy_fingerprint,
            execution_config_identity=self._config.planes.execution_config_identity,
            cost_config_identity=self._config.planes.cost_config_identity,
            risk_identity=self._config.planes.risk_identity,
            feature_spec_fingerprint=feature_fp,
        )
        base.update(experiment_identity=identity, strategy_fingerprint=compiled.strategy_fingerprint)

        if identity in seen:
            return slot_out(PlanningDisposition.DUPLICATE_WITHIN_MANIFEST, **base), None

        # the runtime orchestrator only plans Futures research today (Phase 6
        # ETF Research Pilot's orchestration wiring is a separate, later phase).
        dup = self._registry.find_exact_duplicate(identity, asset_domain=AssetDomain.FUTURES)
        if dup.blocks_reexecution:
            return slot_out(
                PlanningDisposition.DUPLICATE_AUTHORITATIVE_RESULT,
                detail=(
                    f"identity already has a VALID authoritative "
                    f"{dup.headline_verdict.value if dup.headline_verdict else '?'} "
                    f"result ({dup.experiment_id})"
                ),
                **base,
            ), None

        sem = compiled.execution_semantics
        related = self._registry.find_related(
            strategy_family=family,
            root_symbol=root,
            asset_domain=AssetDomain.FUTURES,
            params=params,
            signal_cadence=sem.signal_cadence if sem else "",
            execution_cadence=sem.execution_cadence if sem else "",
            feature_fingerprints=list(feature_fingerprints(spec)),
            top_k=3,
        )
        if related:
            top = related[0]
            if (
                top.similarity.score >= self._config.family_plan.near_duplicate_block_threshold
                and top.headline_verdict in (RegistryVerdict.REJECT, RegistryVerdict.INCONCLUSIVE)
                and not hypothesis.novelty_notes.strip()
            ):
                return slot_out(
                    PlanningDisposition.NEAR_DUPLICATE_NO_NOVELTY,
                    detail=(
                        f"near-duplicate of {top.experiment_id} "
                        f"(score {top.similarity.score:.3f}, prior {top.headline_verdict.value}) "
                        "with no stated novelty"
                    ),
                    **base,
                ), None

        member = FamilyMember(
            ordinal=len(seen),
            experiment_identity=identity,
            strategy_fingerprint=compiled.strategy_fingerprint,
            strategy_id=spec.strategy_id,
            strategy_family=family,
            root_symbol=root,
            params=params,
            parameter_variant_identity=pvi,
            feature_spec_fingerprint=feature_fp,
            trial_role=TrialRole.CANONICAL,
            hypothesis_id=hypothesis.hypothesis_id,
            hypothesis_title=hypothesis.title,
            source_inspirations=tuple(hypothesis.source_inspirations),
            strategy_spec=spec,
            strategy_spec_json=_registry_strategy_spec_json(
                compiled=compiled, spec=spec, family=family, params=params
            ),
            signal_cadence=sem.signal_cadence if sem else "",
            execution_cadence=sem.execution_cadence if sem else "",
        )
        return slot_out(PlanningDisposition.MEMBER_ADDED, **base), member

    # -- ADOPT (safe frozen-manifest adoption) ---------------------------
    def adopt_frozen_manifest(
        self,
        manifest: FamilyManifest,
        *,
        fast_screen_provenance: Mapping[str, float] | None = None,
    ) -> FamilyManifest:
        """Safely adopt an EXTERNALLY-frozen `FamilyManifest` -- e.g. the
        output of `alpha_agent.screening.freeze.freeze_top_k` -- into THIS
        orchestrator instance's own bookkeeping, so `execute_family` /
        `finalize_family` can run it exactly as if this orchestrator had
        planned it itself.

        This does NOT weaken the Phase 18 anti-tampering invariant
        (`_verify_manifest` still requires `manifest.family_id` to be a key of
        `self._planned_snapshots`, completely unchanged -- see module
        docstring history in `docs/ALPHA_DISCOVERY_CAMPAIGN.md`). It satisfies
        that invariant LEGITIMATELY: everything a locally-planned manifest
        gets "for free" (because `plan_family` builds it FROM `self._config`
        in the first place) is verified explicitly here for a manifest this
        orchestrator did NOT build, before it is trusted:

        * the manifest's own content-hash self-consistency
          (`FamilyManifest.assert_consistent`);
        * every member's `StrategySpec` / feature-set / parameter-variant /
          experiment identity RE-DERIVES to what the manifest claims
          (`_manifest_snapshot` -- the exact re-derivation `_verify_manifest`
          already runs for a locally-planned family, called here up front so a
          tampered member is refused at adoption, not silently deferred to a
          later execution-boundary re-check);
        * the family is fully predeclared (`planning_complete`) -- an
          underfilled family is never adopted as a smaller BH family;
        * the manifest was frozen under the IDENTICAL evaluation plane
          (dataset / split / validation-spec / ReliabilityPolicy / execution /
          cost / risk identity) this orchestrator is configured for -- never a
          different plane smuggled in under a familiar-looking family_id;
        * the SAME BH/FDR q-threshold and policy identity this orchestrator's
          adjudicator uses;
        * the declared market window, and every member's root is inside this
          orchestrator's approved market universe;
        * every member's signal cadence is one this release's execution path
          can actually schedule (refusing the whole adoption up front, rather
          than adopting a family that would later die member-by-member as
          typed `INVALID_EXECUTION`);
        * when supplied, `fast_screen_provenance` (e.g.
          `FrozenCandidateSet.screen_scores`) names EXACTLY this manifest's
          members -- proving candidate-identity continuity from Fast Screen
          through Freeze, never a member swapped in after screening;
        * this exact `family_id` was never already planned/adopted by this
          orchestrator instance, and its evaluation plane never already
          carries an executed family (the identical `AdaptiveTestingError`
          `execute_family` already enforces for a locally-planned family).

        Adoption may only happen BEFORE `execute_family` reveals any
        validation evidence for this family: each `family_id` may be adopted
        at most once, so there is no "adopt, inspect an outcome, re-adopt a
        replacement" path -- and none is needed, since adoption itself never
        looks at 2023-2024 validation evidence (it only re-derives PRE-RUN
        identity and plane fields).
        """
        if not isinstance(manifest, FamilyManifest):
            raise TypeError("manifest must be a FamilyManifest")
        self._assert_frozen_planes()

        if manifest.schema_version != "family-manifest/2":
            raise FamilyManifestError(
                f"family {manifest.family_id}: unrecognized manifest "
                f"schema_version {manifest.schema_version!r}; expected "
                "'family-manifest/2'"
            )

        # (1) the manifest's own internal content-hash self-consistency.
        manifest.assert_consistent()

        # (2) every member RE-DERIVES to what the manifest claims.
        snapshot = self._manifest_snapshot(manifest)

        # (3) fully predeclared -- never a silently-shrunk family.
        if not manifest.planning_complete:
            raise FamilyManifestError(
                f"family {manifest.family_id} is not fully predeclared "
                f"({manifest.predeclared_family_size} of "
                f"{manifest.declared_target_size} members); an underfilled "
                "family may never be adopted as a smaller BH family"
            )

        # (4) the IDENTICAL evaluation plane this orchestrator is configured for.
        manifest_planes_fp = fingerprint("orchplanes1", manifest.planes.canonical())
        if manifest_planes_fp != self._planes_fp:
            raise FamilyManifestError(
                f"family {manifest.family_id} was frozen under a different "
                f"evaluation plane ({manifest_planes_fp}) than this "
                f"orchestrator is configured for ({self._planes_fp}); refusing "
                "to adopt a manifest planned/screened against a different "
                "dataset / split / validation-spec / ReliabilityPolicy / "
                "execution / cost / risk identity"
            )

        # (5) the SAME BH/FDR q-threshold and policy identity.
        adjudicator_q = self._adjudicator.fdr_q_threshold()
        if manifest.fdr_q_threshold != adjudicator_q:
            raise FamilyManifestError(
                f"family {manifest.family_id} declares fdr_q_threshold "
                f"{manifest.fdr_q_threshold} but this orchestrator's "
                f"adjudicator uses {adjudicator_q}"
            )
        adjudicator_policy = self._adjudicator.policy_identity()
        if manifest.policy_identity != adjudicator_policy:
            raise FamilyManifestError(
                f"family {manifest.family_id} declares policy_identity "
                f"{manifest.policy_identity} but this orchestrator's "
                f"adjudicator uses {adjudicator_policy}"
            )

        # (6) the declared market window.
        if manifest.market_window != self._config.market_window:
            raise FamilyManifestError(
                f"family {manifest.family_id} declares market_window "
                f"{manifest.market_window!r} but this orchestrator is "
                f"configured for {self._config.market_window!r}"
            )

        # (7) every member's root is inside the approved universe, and its
        # signal cadence is one this release can actually execute.
        approved = set(self._config.market_universe)
        bad_roots = sorted({m.root_symbol for m in manifest.members} - approved)
        if bad_roots:
            raise FamilyManifestError(
                f"family {manifest.family_id} contains root(s) outside the "
                f"approved market universe {sorted(approved)}: {bad_roots}"
            )
        for m in manifest.members:
            cadence = resolve_cadence(
                signal_cadence=m.signal_cadence, strategy_family=m.strategy_family
            )
            if assess_cadence(cadence) is not None:
                raise FamilyManifestError(
                    f"family {manifest.family_id} member {m.ordinal} "
                    f"({m.experiment_identity}) declares signal_cadence "
                    f"{m.signal_cadence!r} (resolved {cadence!r}), which this "
                    "release's execution path cannot schedule; refusing to "
                    "adopt a family that cannot be safely executed"
                )

        # (8) Fast Screen provenance, when supplied: names EXACTLY this
        # manifest's members -- no candidate swapped in after screening.
        if fast_screen_provenance is not None:
            provided_ids = set(fast_screen_provenance)
            member_ids = manifest.member_identities()
            if provided_ids != member_ids:
                raise FamilyManifestError(
                    f"family {manifest.family_id}: fast_screen_provenance "
                    f"names {len(provided_ids)} experiment identities but the "
                    f"manifest has {len(member_ids)}; symmetric difference "
                    f"{sorted(provided_ids ^ member_ids)}"
                )

        # (9) never already adopted/planned, and never on an already-executed
        # evaluation plane.
        if manifest.family_id in self._planned:
            raise FamilyManifestError(
                f"family {manifest.family_id} was already planned/adopted by "
                "this orchestrator instance"
            )
        plane = manifest.evaluation_plane_id()
        if plane in self._executed_planes:
            raise AdaptiveTestingError(
                f"the evaluation plane {plane} already carries an executed "
                "research family; a frozen manifest may not be adopted onto "
                "an evaluation plane that has already produced an executed "
                "family"
            )

        # Adopted. Registered EXACTLY as `plan_family` would -- `execute_family`
        # / `finalize_family` need no changes at all.
        self._planned[manifest.family_id] = manifest
        self._planned_snapshots[manifest.family_id] = snapshot
        return manifest

    # -- EXECUTE ------------------------------------------------------
    def execute_family(self, manifest: FamilyManifest) -> dict[str, list[TrialEvidence]]:
        """Run every predeclared member. Collects raw :class:`TrialEvidence`.

        An ``INVALID_EXECUTION`` writes an ``ExecutionAttemptRecord`` + a typed
        ``FailureRecord`` immediately (engineering evidence, preserved verbatim);
        it never writes a scientific ``ResultRecord``. No BH q-value, no verdict,
        and no ``ResultRecord`` for any member is written here.
        """
        self._verify_manifest(manifest)
        if manifest.family_id in self._finalized:
            raise FamilyManifestError(f"family {manifest.family_id} is already finalized")
        if not manifest.planning_complete:
            raise FamilyManifestError(
                f"family {manifest.family_id} planning is incomplete "
                f"({manifest.predeclared_family_size} of {manifest.declared_target_size} "
                "members); an underfilled family is never executed as a smaller BH family"
            )
        plane = manifest.evaluation_plane_id()
        if plane in self._executed_planes:
            raise AdaptiveTestingError(
                f"the evaluation plane {plane} already carries an executed research "
                "family; a successor family may be executed only under a genuinely new "
                "predeclared evaluation plane or a separately frozen sequential-testing "
                "procedure (neither exists in the MVP)"
            )
        self._executed_planes.add(plane)
        self._executing.add(manifest.family_id)
        cfg = self._config
        evidence: dict[str, list[TrialEvidence]] = {}

        for member in manifest.members:
            self._verify_manifest(manifest)  # re-check on every member boundary
            trail: list[TrialEvidence] = []
            for _attempt in range(1, cfg.budget.max_execution_attempts_per_identity + 1):
                ev = self._exec.run(
                    member=member,
                    manifest=manifest,
                    attempt_ordinal=self._next_ordinal(member.experiment_identity),
                    holdout=False,
                )
                if not isinstance(ev, TrialEvidence):
                    raise OrchestrationError("execution_service.run did not return a TrialEvidence")
                self._assert_evidence_provenance(manifest, member, ev)
                trail.append(ev)
                if ev.status is AttemptStatus.INVALID_EXECUTION:
                    self._record_invalid_attempt(manifest, member, ev)
                    continue
                break
            evidence[member.experiment_identity] = trail
            self._touched.append((member.strategy_family, member.root_symbol))
        return evidence

    def _assert_evidence_provenance(
        self, manifest: FamilyManifest, member: FamilyMember, ev: TrialEvidence
    ) -> None:
        vspec = ev.produced_under_validation_spec_fingerprint
        pol = ev.produced_under_reliability_policy_fingerprint
        if vspec is not None and vspec != manifest.planes.validation_spec_fingerprint:
            raise EvidenceProvenanceError(
                f"member {member.ordinal} evidence was produced under validation spec "
                f"{vspec}, not the manifest's {manifest.planes.validation_spec_fingerprint}"
            )
        if pol is not None and pol != manifest.planes.reliability_policy_fingerprint:
            raise EvidenceProvenanceError(
                f"member {member.ordinal} evidence was produced under reliability policy "
                f"{pol}, not the manifest's {manifest.planes.reliability_policy_fingerprint}"
            )

    # -- FINALIZE ---------------------------------------------------
    def finalize_family(
        self, manifest: FamilyManifest, evidence: dict[str, list[TrialEvidence]]
    ) -> FamilyReport:
        """Family-wide BH/FDR over every member's trial p-value, then the final
        per-member verdict -- written to the registry atomically. A family with
        any unresolved ``INVALID_EXECUTION`` member is not adjudicated at all."""
        self._verify_manifest(manifest)
        planning = _disposition_counts(o.disposition.value for o in manifest.planning_log)

        latest = {ident: trail[-1] for ident, trail in evidence.items()}
        unresolved = tuple(
            m.experiment_identity
            for m in manifest.members
            if latest.get(m.experiment_identity) is None
            or latest[m.experiment_identity].status is AttemptStatus.INVALID_EXECUTION
        )
        if unresolved:
            # do NOT shrink the BH denominator, do NOT manufacture evidence.
            self._executing.discard(manifest.family_id)
            report = self._family_report_shell(
                manifest,
                status=FamilyStatus.INCOMPLETE_NOT_ADJUDICATED,
                planning=planning,
                unresolved_invalid_identities=unresolved,
                member_results=tuple(
                    _incomplete_member_result(m, evidence.get(m.experiment_identity, []))
                    for m in manifest.members
                ),
            )
            self._finalized[manifest.family_id] = report
            return report

        # every member has VALID evidence -> the frozen policy adjudicates the
        # whole family (BH + final verdict). The orchestrator does not fork it.
        latest_valid = {ident: ev for ident, ev in latest.items()}
        adjudication = self._adjudicator.adjudicate_family(
            members=manifest.members, latest_evidence=latest_valid
        )
        if adjudication.policy_identity != manifest.policy_identity:
            raise OrchestrationError(
                "the adjudicator returned a policy identity that does not match the "
                "frozen manifest"
            )
        by_ident = adjudication.by_identity()

        now = datetime.now(UTC).isoformat()
        run_tag = manifest.family_id.split(":")[-1][:12]
        experiments: list[ExperimentRecord] = []
        attempts: list[ExecutionAttemptRecord] = []
        results: list[ResultRecord] = []
        failures: list[FailureRecord] = []
        member_results: list[FamilyMemberResult] = []

        for m in manifest.members:
            decision = by_ident[m.experiment_identity]
            ev = latest[m.experiment_identity]
            verdict, reasons = decision.final_verdict, decision.reason_codes
            had_invalid = any(
                e.status is AttemptStatus.INVALID_EXECUTION
                for e in evidence[m.experiment_identity]
            )
            exp_id = friendly_experiment_id(
                root_symbol=m.root_symbol,
                strategy_family=m.strategy_family,
                variant_label="canonical",
                split_label=manifest.market_window.label,
                lineage_tag=m.experiment_identity.split(":")[-1][:10],
            )
            experiments.append(self._experiment_record(manifest, m, ev, exp_id, now, verdict))
            attempts.append(
                ExecutionAttemptRecord(
                    experiment_identity=m.experiment_identity,
                    identity_schema=IDENTITY_SCHEMA,
                    attempt_ordinal=None,
                    attempt_status=AttemptStatus.VALID,
                    code_commit=self._config.code_commit,
                    engine=ev.engine,
                    target_schedule_hash=ev.target_schedule_hash,
                    report_fingerprint=ev.report_fingerprint,
                    source_artifact=ev.source_artifact,
                    source_artifact_sha256=ev.source_artifact_sha256,
                    created_at=now,
                    notes=(
                        f"family {manifest.family_id} finalization; verdict {verdict.value}"
                    ),
                )
            )
            results.append(
                self._result_record(m, ev, verdict, reasons, decision.bh_p_value,
                                    decision.bh_q_value, decision.bh_rejected_at_q,
                                    verdict is RegistryVerdict.PASS)
            )
            if verdict in _SCIENTIFIC_FAILURE_CLASS:
                failures.append(
                    FailureRecord(
                        failure_id=f"phase_{self._config.phase}_{verdict.value.lower()}__{run_tag}__{m.ordinal}",
                        scope=FailureScope.EXPERIMENT,
                        experiment_identity=m.experiment_identity,
                        failure_class=_SCIENTIFIC_FAILURE_CLASS[verdict],
                        failure_code="|".join(reasons) or verdict.value,
                        summary=f"family {manifest.family_id} member {m.ordinal}: {verdict.value}",
                        mechanism="; ".join(reasons) or "family-wide BH + frozen per-trial gates",
                        evidence={"reason_codes": list(reasons), "bh_q_value": decision.bh_q_value},
                        action_taken="recorded as first-class evidence; never deleted",
                        created_at=now,
                        root_symbol=m.root_symbol,
                        strategy_family=m.strategy_family,
                    )
                )
            member_results.append(
                FamilyMemberResult(
                    ordinal=m.ordinal,
                    experiment_identity=m.experiment_identity,
                    experiment_id=exp_id,
                    trial_role=m.trial_role,
                    disposition=(
                        MemberDisposition.INVALID_THEN_CORRECTED
                        if had_invalid
                        else MemberDisposition.VALID_EVIDENCE
                    ),
                    attempt_status=AttemptStatus.VALID,
                    attempt_ordinal=len(evidence[m.experiment_identity]),
                    n_execution_attempts=len(evidence[m.experiment_identity]),
                    trial_p_value=decision.bh_p_value,
                    bh_q_value=decision.bh_q_value,
                    bh_rejected_at_q=decision.bh_rejected_at_q,
                    final_verdict=verdict,
                    reason_codes=tuple(reasons),
                    holdout_eligible=verdict is RegistryVerdict.PASS,
                )
            )

        bundle = ImportBundle(
            import_id=f"phase_{self._config.phase}_family__{run_tag}",
            phase=self._config.phase,
            source_fingerprint=manifest.manifest_fingerprint(),
            created_at=now,
            experiments=tuple(experiments),
            execution_attempts=tuple(attempts),
            results=tuple(results),
            failures=tuple(failures),
            metadata={
                "family_id": manifest.family_id,
                "bh_family_size": manifest.predeclared_family_size,
                "bh_q_threshold": manifest.fdr_q_threshold,
                "n_bh_rejected": adjudication.n_bh_rejected,
                "policy_identity": adjudication.policy_identity,
                "evaluation_plane_id": manifest.evaluation_plane_id(),
            },
        )
        self._registry.apply_bundle(bundle)
        self._executing.discard(manifest.family_id)

        verdict_counts = _disposition_counts(
            r.final_verdict.value for r in member_results if r.final_verdict
        )
        passing = tuple(r.experiment_id for r in member_results if r.final_verdict is RegistryVerdict.PASS)
        report = self._family_report_shell(
            manifest,
            status=FamilyStatus.FINALIZED,
            planning=planning,
            n_bh_rejected=adjudication.n_bh_rejected,
            verdict_counts=verdict_counts,
            member_results=tuple(member_results),
            passing_experiment_ids=passing,
        )
        self._finalized[manifest.family_id] = report
        return report

    def _family_report_shell(
        self, manifest: FamilyManifest, *, status: FamilyStatus, planning: dict, **kw
    ) -> FamilyReport:
        return FamilyReport(
            family_id=manifest.family_id,
            family_stem=manifest.family_stem,
            generation=manifest.generation,
            parent_family_id=manifest.parent_family_id,
            manifest_fingerprint=manifest.manifest_fingerprint(),
            status=status,
            predeclared_family_size=manifest.predeclared_family_size,
            declared_target_size=manifest.declared_target_size,
            fdr_q_threshold=manifest.fdr_q_threshold,
            policy_identity=manifest.policy_identity,
            evaluation_plane_id=manifest.evaluation_plane_id(),
            validation_spec_fingerprint=manifest.planes.validation_spec_fingerprint,
            planning_dispositions=planning,
            **kw,
        )

    # -- HOLDOUT (isolated) ----------------------------------------
    def run_holdout(
        self,
        experiment_identity: str,
        *,
        strategy_spec: StrategySpec,
        holdout_service: HoldoutEvaluationService,
        allow_holdout: bool = False,
    ) -> TrialEvidence:
        """The one-time locked-holdout evaluation -- isolated from the loop.

        Needs an *explicitly passed* :class:`HoldoutEvaluationService`, the
        candidate's ``StrategySpec``, and ``allow_holdout=True`` -- arguments the
        iteration loop never has. The candidate's family must already be
        finalized with a VALID authoritative result for this identity."""
        if not allow_holdout or holdout_service is None:
            raise HoldoutIsolationError(
                "run_holdout requires allow_holdout=True and an explicit "
                "HoldoutEvaluationService; the locked 2025 holdout is never touched "
                "from the research iteration loop"
            )
        if not isinstance(holdout_service, HoldoutEvaluationService):
            raise TypeError("holdout_service must implement HoldoutEvaluationService")
        if isinstance(holdout_service, ExecutionValidationService):
            raise TypeError(
                "the holdout service must not also be the iteration ExecutionValidationService"
            )
        view = self._registry.get(experiment_identity)
        if not view.has_valid_authoritative_result:
            raise HoldoutIsolationError(
                f"{experiment_identity} has no VALID authoritative research result; the "
                "holdout is evaluated only for a finalized-family candidate"
            )
        if strategy_fingerprint(strategy_spec) != view.experiment.strategy_fingerprint:
            raise OrchestrationError(
                "strategy_spec does not match the stored research strategy_fingerprint"
            )
        outcome = holdout_service.evaluate_holdout(
            strategy_spec=strategy_spec,
            experiment_identity=experiment_identity,
            root_symbol=view.experiment.root_symbol,
        )
        assert_no_holdout_market_data(outcome.model_dump(mode="json"), path="$.holdout_outcome")
        return outcome

    # -- internal: registry records --------------------------------
    def _experiment_record(
        self,
        manifest: FamilyManifest,
        m: FamilyMember,
        ev: TrialEvidence,
        exp_id: str,
        now: str,
        verdict: RegistryVerdict,
    ) -> ExperimentRecord:
        cfg = self._config
        return ExperimentRecord(
            experiment_identity=m.experiment_identity,
            identity_schema=IDENTITY_SCHEMA,
            experiment_id=exp_id,
            display_name=f"{m.hypothesis_title} [{m.hypothesis_id}]",
            created_at=now,
            phase=cfg.phase,
            status=ExperimentStatus.COMPLETED,
            code_commit=cfg.code_commit,
            root_symbol=m.root_symbol,
            # the runtime Research Orchestrator only researches Futures roots
            # today -- a stated fact, not an inference (Phase 6 ETF Research
            # Pilot's own orchestration wiring is a separate, later phase).
            asset_domain=AssetDomain.FUTURES,
            strategy_family=m.strategy_family,
            strategy_fingerprint=m.strategy_fingerprint,
            strategy_id=m.strategy_id,
            strategy_spec_json=m.strategy_spec_json,
            feature_spec_fingerprint=m.feature_spec_fingerprint,
            target_schedule_hash=ev.target_schedule_hash,
            candidate_manifest_fingerprint=manifest.manifest_fingerprint(),
            dataset_fingerprint=cfg.planes.dataset_fingerprint,
            split_identity=cfg.planes.split_identity,
            market_window=manifest.market_window,
            validation_spec_fingerprint=cfg.planes.validation_spec_fingerprint,
            reliability_policy_fingerprint=cfg.planes.reliability_policy_fingerprint,
            execution_config_identity=cfg.planes.execution_config_identity,
            cost_config_identity=cfg.planes.cost_config_identity,
            risk_identity=cfg.planes.risk_identity,
            trial_role=m.trial_role,
            parameter_variant_identity=m.parameter_variant_identity,
            parameter_variant_label="canonical",
            report_fingerprint=ev.report_fingerprint,
            notes=(
                f"Phase {cfg.phase} predeclared family {manifest.family_id} "
                f"(size {manifest.predeclared_family_size}); verdict {verdict.value}"
            ),
        )

    def _result_record(
        self,
        m: FamilyMember,
        ev: TrialEvidence,
        verdict: RegistryVerdict,
        reasons: Sequence[str],
        bh_p: float,
        bh_q: float,
        bh_rejected: bool,
        holdout_eligible: bool,
    ) -> ResultRecord:
        mm = ev.metrics
        return ResultRecord(
            experiment_identity=m.experiment_identity,
            headline_verdict=verdict,
            reason_codes=tuple(reasons),
            gross_pnl_usd=_flt(mm.get("gross_pnl_usd")),
            costs_usd=_flt(mm.get("costs_usd")),
            net_pnl_usd=_flt(mm.get("net_pnl_usd")),
            daily_sharpe=_flt(mm.get("daily_sharpe")),
            annualized_sharpe=_flt(mm.get("annualized_sharpe")),
            gating_null_p=_flt(mm.get("gating_null_p")),
            bh_p_value=bh_p,
            bh_q=bh_q,
            bh_rejected_at_q=bh_rejected,
            dsr_probability=_flt(mm.get("dsr_probability")),
            fold_consistency=_flt(mm.get("fold_consistency")),
            n_trades=_int(mm.get("n_trades")),
            n_fills=_int(mm.get("n_fills")),
            n_oos_days=_int(mm.get("n_oos_days")),
            holdout_eligible=holdout_eligible,
            evidence_completeness=ev.evidence_completeness,
            source_artifact=ev.source_artifact,
            source_artifact_sha256=ev.source_artifact_sha256,
        )

    def _record_invalid_attempt(
        self, manifest: FamilyManifest, m: FamilyMember, ev: TrialEvidence
    ) -> None:
        """An INVALID_EXECUTION attempt: an ExecutionAttemptRecord + a typed
        FailureRecord, and NOTHING scientific."""
        now = datetime.now(UTC).isoformat()
        exp_id = friendly_experiment_id(
            root_symbol=m.root_symbol,
            strategy_family=m.strategy_family,
            variant_label="canonical",
            split_label=manifest.market_window.label,
            lineage_tag=m.experiment_identity.split(":")[-1][:10],
        )
        record = self._experiment_record(manifest, m, ev, exp_id, now, RegistryVerdict.NOT_ADJUDICATED)
        record = record.model_copy(update={"status": ExperimentStatus.FAILED})
        attempt = ExecutionAttemptRecord(
            experiment_identity=m.experiment_identity,
            identity_schema=IDENTITY_SCHEMA,
            attempt_ordinal=None,
            attempt_status=AttemptStatus.INVALID_EXECUTION,
            invalidation_class=ev.invalidation_class,
            invalidation_detail=ev.invalidation_detail,
            invalidation_evidence=dict(ev.invalidation_evidence),
            defect_resolution_commit=ev.defect_resolution_commit,
            code_commit=self._config.code_commit,
            engine=ev.engine,
            report_fingerprint=ev.report_fingerprint,
            source_artifact=ev.source_artifact,
            source_artifact_sha256=ev.source_artifact_sha256,
            created_at=now,
            notes=(
                f"family {manifest.family_id} member {m.ordinal}: INVALID_EXECUTION "
                f"({ev.invalidation_class.value if ev.invalidation_class else '?'})"
            ),
        )
        # NO ResultRecord for an INVALID_EXECUTION attempt.
        self._registry.insert_experiment(record, None, attempt=attempt)
        tag = m.experiment_identity.split(":")[-1][:12]
        self._registry.record_failure(
            FailureRecord(
                failure_id=f"phase_{self._config.phase}_invalid__{tag}__{self._next_ordinal(m.experiment_identity) - 1}",
                scope=FailureScope.EXPERIMENT,
                experiment_identity=m.experiment_identity,
                failure_class=FailureClass.INVALID_EXECUTION_ATTEMPT,
                failure_code=(
                    ev.invalidation_class.value if ev.invalidation_class else "INVALID_EXECUTION"
                ),
                summary=f"family {manifest.family_id} member {m.ordinal}: INVALID_EXECUTION",
                mechanism=ev.invalidation_detail
                or "engineering / data-pipeline defect; not a scientific outcome",
                evidence=dict(ev.invalidation_evidence),
                action_taken="recorded as an INVALID_EXECUTION attempt; re-execution permitted",
                created_at=now,
                root_symbol=m.root_symbol,
                strategy_family=m.strategy_family,
            )
        )

    # -- internal: context + reflection ---------------------------
    def _planning_context(self) -> ResearchContext:
        """The research context for planning a family. A pure function of
        FINALIZED history + inter-generation feedback -- it never carries an
        in-flight or within-family outcome.

        PRE-PROPOSAL failure memory (Agent runtime-integration release,
        section 3): a bounded sweep of every (strategy_family, root_symbol)
        combination already tested for a root in the approved
        `market_universe`, via the shared
        `alpha_agent.registry.failure_memory.relevant_failure_memory` (the
        SAME sweep the lightweight UI propose/inspect path uses). Without
        this, GENERATION 0's very first family would see no failure memory at
        all -- `self._touched` starts empty and previously only accumulated
        AFTER a member executed, so a fresh orchestrator run's H1 could never
        know about prior evidence for its own market before proposing.
        `self._touched` is still unioned in (a strict superset once any
        member has executed) so later generations lose nothing."""
        universe_wide = relevant_failure_memory(
            self._registry, market_universe=self._config.market_universe,
            asset_domain=AssetDomain.FUTURES,
        )
        touched = [
            self._failure_memory.lookup(
                strategy_family=family, root_symbol=root, asset_domain=AssetDomain.FUTURES
            )
            for family, root in dict.fromkeys(self._touched)
        ]
        seen_combos = {
            (fm.query.get("strategy_family"), fm.query.get("root_symbol")) for fm in universe_wide
        }
        seen = list(universe_wide) + [
            fm for fm in touched
            if (fm.query.get("strategy_family"), fm.query.get("root_symbol")) not in seen_combos
        ]
        return build_research_context(
            objective=self._config.objective,
            market_universe=self._config.market_universe,
            knowledge_base=self._config.knowledge_base,
            registry_summary=self._registry.summary(),
            failure_memory=tuple(seen),
            validation_feedback=tuple(self._generation_feedback),
            avoid_repeating=tuple(self._generation_avoid),
            feature_registry=self._feature_registry,
        )

    def _build_compiler_context(self) -> CompilerContext:
        from alpha_agent.agents.compiler_agent import KnownStrategyRecord, build_compiler_context

        known: list[KnownStrategyRecord] = []
        for view in self._registry.experiments(authoritative_only=False):
            fp = view.experiment.strategy_fingerprint
            if not fp:
                continue
            known.append(
                KnownStrategyRecord(
                    source="live_experiment_registry",
                    strategy_fingerprint=fp,
                    experiment_id=view.experiment_id,
                    experiment_identity=view.experiment_identity,
                    has_valid_authoritative_result=view.has_valid_authoritative_result,
                    valid_execution_attempts=view.n_valid_attempts,
                    invalid_execution_attempts=view.n_invalid_attempts,
                )
            )
        return build_compiler_context(
            approved_universe=self._config.market_universe,
            feature_registry=self._feature_registry,
            known_strategies=tuple(known),
        )

    def _reflect_after_finalization(self, report: FamilyReport) -> None:
        """Turn a FINALIZED family's typed results into feedback for the NEXT
        generation. It never mutates the finalized family; the next family gets a
        new predeclared manifest and a new family_id."""
        for r in report.member_results:
            loc = ""
            if r.final_verdict is None:
                continue
            if r.final_verdict is RegistryVerdict.PASS:
                self._generation_feedback.append(
                    f"family {report.family_id} member {r.ordinal}: PASS -- a holdout "
                    "candidate. Prefer a distinct mechanism in the next family."
                )
                self._generation_avoid.append(
                    f"{loc}experiment {r.experiment_id}: already PASSED research."
                )
            elif r.final_verdict is RegistryVerdict.REJECT:
                self._generation_feedback.append(
                    f"family {report.family_id} member {r.ordinal}: REJECT "
                    f"({list(r.reason_codes)})."
                )
                self._generation_avoid.append(
                    f"experiment {r.experiment_id}: REJECT with a VALID authoritative "
                    "result -- do not repropose without a materially different mechanism."
                )
            elif r.final_verdict is RegistryVerdict.INCONCLUSIVE:
                self._generation_feedback.append(
                    f"family {report.family_id} member {r.ordinal}: INCONCLUSIVE "
                    f"({list(r.reason_codes)}) -- insufficient evidence."
                )
        if report.status is FamilyStatus.INCOMPLETE_NOT_ADJUDICATED:
            self._generation_feedback.append(
                f"family {report.family_id} was INCOMPLETE_NOT_ADJUDICATED "
                "(unresolved engineering defect); its hypotheses are neither "
                "confirmed nor refuted."
            )

    # -- internal: guards ----------------------------------------
    def _verify_manifest(self, manifest: FamilyManifest) -> None:
        if not isinstance(manifest, FamilyManifest):
            raise TypeError("manifest must be a FamilyManifest")
        manifest.assert_consistent()
        stored = self._planned_snapshots.get(manifest.family_id)
        if stored is None:
            raise FamilyManifestError(
                f"family {manifest.family_id} was not planned by this orchestrator"
            )
        # re-derive the canonical snapshot from the (possibly nested-mutated)
        # manifest and compare against the INDEPENDENT planning-time snapshot.
        current = self._manifest_snapshot(manifest)
        if current["snapshot_fingerprint"] != stored["snapshot_fingerprint"]:
            raise FamilyManifestError(
                f"family {manifest.family_id} content changed after it was predeclared: "
                "a member's params, StrategySpec, strategy_spec_json, or identity no "
                "longer re-derives to the frozen snapshot"
            )
        self._assert_frozen_planes()

    def _manifest_snapshot(self, manifest: FamilyManifest) -> dict:
        """An INDEPENDENT canonical fingerprint of the family, every field
        RE-DERIVED from the actual executable member -- not read back from the
        (nested-mutable) manifest object. Raises if a stored member field no
        longer agrees with a fresh re-derivation."""
        planes = manifest.planes
        rows: list[dict] = []
        for m in manifest.members:
            spec = m.strategy_spec
            sfp = strategy_fingerprint(spec)
            ffp = feature_set_fingerprint(spec)
            pvi = parameter_variant_identity(_normalize_params(m.params))
            ident = experiment_identity(
                strategy_fingerprint=sfp,
                strategy_family=m.strategy_family,
                root_symbol=m.root_symbol,
                parameter_variant_identity=pvi,
                dataset_fingerprint=planes.dataset_fingerprint,
                split_identity=planes.split_identity,
                validation_spec_fingerprint=planes.validation_spec_fingerprint,
                reliability_policy_fingerprint=planes.reliability_policy_fingerprint,
                execution_config_identity=planes.execution_config_identity,
                cost_config_identity=planes.cost_config_identity,
                risk_identity=planes.risk_identity,
                feature_spec_fingerprint=ffp,
            )
            for field, rederived, stored_value in (
                ("strategy_fingerprint", sfp, m.strategy_fingerprint),
                ("feature_spec_fingerprint", ffp, m.feature_spec_fingerprint),
                ("parameter_variant_identity", pvi, m.parameter_variant_identity),
                ("experiment_identity", ident, m.experiment_identity),
            ):
                if rederived != stored_value:
                    raise FamilyManifestError(
                        f"family {manifest.family_id} member {m.ordinal}: {field} "
                        f"re-derives to {rederived} but the manifest stores {stored_value}"
                    )
            rows.append(
                {
                    "ordinal": m.ordinal,
                    "strategy_fingerprint": sfp,
                    "feature_spec_fingerprint": ffp,
                    "parameter_variant_identity": pvi,
                    "experiment_identity": ident,
                    "strategy_family": m.strategy_family,
                    "root_symbol": m.root_symbol,
                    "trial_role": m.trial_role.value,
                    "hypothesis_id": m.hypothesis_id,
                    "signal_cadence": m.signal_cadence,
                    "execution_cadence": m.execution_cadence,
                    "params": _normalize_params(m.params),
                    "strategy_spec_json": fingerprint("regstratspecjson1", m.strategy_spec_json),
                    # full spec dump: catches even a cosmetic (metadata) mutation
                    "strategy_spec_full": fingerprint(
                        "stratspecfull1", spec.model_dump(mode="json"), allow_non_finite=True
                    ),
                }
            )
        snap = {
            "family_id": manifest.family_id,
            "policy_identity": manifest.policy_identity,
            "fdr_q_threshold": manifest.fdr_q_threshold,
            "declared_target_size": manifest.declared_target_size,
            "planes": planes.canonical(),
            "market_window": manifest.market_window.model_dump(mode="json"),
            "members": rows,
        }
        snap["snapshot_fingerprint"] = fingerprint("familysnapshot1", snap)
        return snap

    def _assert_frozen_planes(self) -> None:
        if fingerprint("orchplanes1", self._config.planes.canonical()) != self._planes_fp:
            raise OrchestrationError("identity planes changed during the run")
        if (
            fingerprint("orchplan1", self._config.family_plan.model_dump(mode="json"))
            != self._plan_fp
        ):
            raise OrchestrationError(
                "the family plan changed during the run -- it is predeclared and never "
                "adapted after outcomes"
            )

    def _next_ordinal(self, identity: str) -> int:
        try:
            return len(self._registry.get(identity).attempts) + 1
        except UnknownExperiment:
            return 1


# ==========================================================================
# module-level helpers
# ==========================================================================
def _incomplete_member_result(
    m: FamilyMember, trail: Sequence[TrialEvidence]
) -> FamilyMemberResult:
    latest = trail[-1] if trail else None
    invalid = latest is None or latest.status is AttemptStatus.INVALID_EXECUTION
    return FamilyMemberResult(
        ordinal=m.ordinal,
        experiment_identity=m.experiment_identity,
        experiment_id="",
        trial_role=m.trial_role,
        disposition=(
            MemberDisposition.INVALID_EXECUTION_UNRESOLVED
            if invalid
            else MemberDisposition.VALID_EVIDENCE
        ),
        attempt_status=(
            AttemptStatus.INVALID_EXECUTION if invalid else AttemptStatus.VALID
        ),
        attempt_ordinal=max(1, len(trail)),
        n_execution_attempts=len(trail),
        trial_p_value=None if invalid else latest.trial_p_value,
        final_verdict=None,
        reason_codes=("FAMILY_NOT_ADJUDICATED",),
    )


def _family_report(manifest: FamilyManifest, status: FamilyStatus, detail: str = "") -> FamilyReport:
    return FamilyReport(
        family_id=manifest.family_id,
        family_stem=manifest.family_stem,
        generation=manifest.generation,
        parent_family_id=manifest.parent_family_id,
        manifest_fingerprint=manifest.manifest_fingerprint(),
        status=status,
        predeclared_family_size=manifest.predeclared_family_size,
        declared_target_size=manifest.declared_target_size,
        fdr_q_threshold=manifest.fdr_q_threshold,
        policy_identity=manifest.policy_identity,
        evaluation_plane_id=manifest.evaluation_plane_id(),
        validation_spec_fingerprint=manifest.planes.validation_spec_fingerprint,
        detail=detail,
        planning_dispositions=_disposition_counts(
            o.disposition.value for o in manifest.planning_log
        ),
    )


def _empty_family_report(manifest: FamilyManifest) -> FamilyReport:
    return _family_report(manifest, FamilyStatus.EMPTY, "planning admitted no members")


def _planning_incomplete_report(manifest: FamilyManifest) -> FamilyReport:
    return _family_report(
        manifest,
        FamilyStatus.PLANNING_INCOMPLETE,
        f"planning filled {manifest.predeclared_family_size} of the declared "
        f"{manifest.declared_target_size} members; not executed as a smaller BH family",
    )


def _planned_not_executed_report(manifest: FamilyManifest) -> FamilyReport:
    return _family_report(
        manifest,
        FamilyStatus.PLANNED_NOT_EXECUTED,
        "a reflection successor family; its evaluation plane already carried an "
        "executed research family, so it is planned but not scientifically executed",
    )


def _disposition_counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
