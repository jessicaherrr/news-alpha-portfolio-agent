"""Closed enumerations for the Phase 14 experiment registry / failure memory.

Registry truth is deterministic and typed: every status, role, relation and
failure identity below is a closed set. Free text may *supplement* a typed code
(``summary``, ``mechanism``) but never *is* the identity of a record, and an LLM
never decides any value here (prompt 14 section 23).
"""
from __future__ import annotations

from enum import Enum


class ExperimentStatus(str, Enum):
    """Lifecycle of one registry experiment row."""

    COMPLETED = "COMPLETED"          # executed and adjudicated; result is immutable
    FAILED = "FAILED"                # engineering/data failure, no interpretable result
    WITHDRAWN = "WITHDRAWN"          # recorded then withdrawn (never deleted)


class Authority(str, Enum):
    """Whether a completed experiment is the *current* scientific answer.

    Never stored as a column -- it is DERIVED from the lineage graph, so no
    completed experiment row is ever updated in place (section 7).
    """

    AUTHORITATIVE = "AUTHORITATIVE"
    SUPERSEDED = "SUPERSEDED"


class AttemptStatus(str, Enum):
    """Schema v5: the outcome of ONE execution attempt of an experiment.

    ``experiment_identity`` is the pre-run scientific hypothesis; an *attempt* is
    one execution of it. The same identity may have several immutable attempts
    when an earlier one was invalid for an engineering / data-pipeline /
    software / infrastructure reason. Only a ``VALID`` attempt may supply the
    authoritative scientific result; ``INVALID_EXECUTION`` attempts are kept
    verbatim as legacy evidence and never enter a BH/FDR family.
    """

    VALID = "VALID"
    INVALID_EXECUTION = "INVALID_EXECUTION"


class InvalidationClass(str, Enum):
    """Why an execution attempt was invalid -- never a scientific reason."""

    FEATURE_PIPELINE_DEFECT = "FEATURE_PIPELINE_DEFECT"
    DATA_PIPELINE_DEFECT = "DATA_PIPELINE_DEFECT"
    SOFTWARE_DEFECT = "SOFTWARE_DEFECT"
    INFRASTRUCTURE_DEFECT = "INFRASTRUCTURE_DEFECT"


class AttemptRelation(str, Enum):
    """Attempt-layer lineage, read ``source <RELATION> target``. Attempt
    authority is still DERIVED (highest-ordinal VALID attempt wins); an edge is
    audit provenance, never the authority mechanism itself."""

    SUPERSEDES_ATTEMPT = "SUPERSEDES_ATTEMPT"
    CORRECTS_ATTEMPT = "CORRECTS_ATTEMPT"


class TrialRole(str, Enum):
    """The role a registry experiment plays in its multiple-testing family.

    Both roles are genuine statistical hypotheses and both stay in the BH/FDR
    denominator. ``CANONICAL`` is additionally *headline adjudicated*.
    """

    CANONICAL = "CANONICAL"
    NEIGHBOUR = "NEIGHBOUR"
    ABLATION = "ABLATION"
    VARIANT = "VARIANT"


class RegistryVerdict(str, Enum):
    """Headline adjudication outcome.

    ``PASS`` / ``REJECT`` / ``INCONCLUSIVE`` mirror
    :class:`alpha_agent.validation.enums.Verdict` exactly. ``NOT_ADJUDICATED``
    is the registry-only value for a predeclared parameter neighbour: it is a
    real BH observation with its own p-value and q-value, but the frozen
    Phase 13 policy adjudicates a headline verdict only for the canonical trial.
    """

    PASS = "PASS"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_ADJUDICATED = "NOT_ADJUDICATED"


class FailureClass(str, Enum):
    """Typed failure taxonomy (section 5). A failure is first-class data."""

    # -- scientific / statistical --
    SCIENTIFIC_REJECTION = "SCIENTIFIC_REJECTION"
    STATISTICAL_INCONCLUSIVE = "STATISTICAL_INCONCLUSIVE"
    INSUFFICIENT_TRADES = "INSUFFICIENT_TRADES"
    NULL_NOT_REJECTED = "NULL_NOT_REJECTED"
    FDR_NOT_PASSED = "FDR_NOT_PASSED"
    DSR_NOT_PASSED = "DSR_NOT_PASSED"
    PARAMETER_INSTABILITY = "PARAMETER_INSTABILITY"
    COST_FRAGILITY = "COST_FRAGILITY"
    CROSS_MARKET_WEAKNESS = "CROSS_MARKET_WEAKNESS"

    # -- data --
    DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
    CONTRACT_ECONOMICS_FAILURE = "CONTRACT_ECONOMICS_FAILURE"
    ROLL_DATA_FAILURE = "ROLL_DATA_FAILURE"

    # -- engineering --
    EXECUTION_INTEGRITY_FAILURE = "EXECUTION_INTEGRITY_FAILURE"
    SOFTWARE_FAILURE = "SOFTWARE_FAILURE"
    FEATURE_PIPELINE_FAILURE = "FEATURE_PIPELINE_FAILURE"

    # -- procedural --
    SUPERSEDED_RESULT = "SUPERSEDED_RESULT"
    INVALID_EXECUTION_ATTEMPT = "INVALID_EXECUTION_ATTEMPT"


class FailureScope(str, Enum):
    """Whether a failure attaches to one experiment or to the system."""

    EXPERIMENT = "EXPERIMENT"
    SYSTEM = "SYSTEM"


class RelationType(str, Enum):
    """Lineage edge, read as ``source <RELATION> target`` (section 4D)."""

    SUPERSEDES = "SUPERSEDES"
    CORRECTS = "CORRECTS"
    REPRODUCES = "REPRODUCES"
    PARAMETER_NEIGHBOUR_OF = "PARAMETER_NEIGHBOUR_OF"
    SAME_HYPOTHESIS_SENSITIVITY_OF = "SAME_HYPOTHESIS_SENSITIVITY_OF"


#: Relations whose *target* is no longer the current scientific answer.
SUPERSEDING_RELATIONS: frozenset[RelationType] = frozenset(
    {RelationType.SUPERSEDES, RelationType.CORRECTS}
)


class AssetDomain(str, Enum):
    """The research domains this platform supports. Canonical home: this
    module, not ``alpha_agent.core.instrument`` -- the registry (this
    package) sits BELOW ``alpha_agent.core`` in the import graph (many
    `core` re-exports pull FROM `registry.enums` already, e.g.
    `RegistryVerdict`/`Authority`/`TrialRole`), so a type the registry
    itself needs to store (Phase 6's `experiments.asset_domain` column)
    cannot live in `core` without a circular import
    (`registry -> core -> alpha_memory -> agents -> registry`, hit for
    real 2026-09-23 building the Phase 6 registry namespacing work).
    `core.instrument.AssetDomain` re-exports this object unchanged
    (`is`-identical), matching the zero-cost-aliasing pattern every other
    `core` re-export already uses.

    ``ETF`` was added in Phase 6 (ETF Research Pilot) under the user's
    explicit approval of that phase's "shared physical Registry,
    structurally namespaced scientific evidence" instruction -- not a
    routine implementation choice, and not a silent side effect of any
    other change (``FUTURES`` alone was frozen by Phase 5's own review
    gate). Adding a THIRD member (Equity, Options, ...) still needs the
    same kind of explicit approval -- CLAUDE.md's FROZEN RESEARCH
    SEMANTICS stop condition covers any change that would let a new
    domain's evidence enter another domain's statistical family.

    ``EQUITY`` was added in Phase 9.1 (Equities Foundation), explicitly
    named as goal 2 of the user-approved Phase 9.1 prompt on top of the
    already-approved Phase 9 sub-roadmap proposal
    (``docs/PHASE_9_EQUITIES_SPECIALIZED_LABS_PROPOSAL.md``
    section 4.2 named this exact addition as the FROZEN RESEARCH SEMANTICS
    item needing sign-off before implementation) -- again not a routine
    choice. As of Phase 9.1, no Equity `experiment_identity` has been
    created and `alpha_graph.builder`'s domain loop / `ui/services.py`'s
    FUTURES+ETF tuple are UNCHANGED (widening them is explicitly Phase
    9.3 scope) -- adding this member alone cannot yet let Equity evidence
    enter any existing statistical family."""

    FUTURES = "FUTURES"
    ETF = "ETF"
    EQUITY = "EQUITY"


# ---------------------------------------------------------------------------
# News Alpha Phase H -- signal-path evidence (schema v7). The HYPOTHESIS plane's
# lineage (event -> transmission path -> expression -> measurement -> candidate
# signal -> portfolio) recorded as typed evidence, linked by id to the EVIDENCE
# plane (an experiment identity, when one exists). Closed vocabularies: an LLM
# never decides any of these values.
# ---------------------------------------------------------------------------


class HypothesisStage(str, Enum):
    """How far one news-alpha hypothesis travelled down the pipeline."""

    SIGNAL_PATH = "SIGNAL_PATH"
    ASSET_EXPRESSION = "ASSET_EXPRESSION"
    MEASUREMENT = "MEASUREMENT"
    CANDIDATE_SIGNAL = "CANDIDATE_SIGNAL"
    SCREEN = "SCREEN"
    RANKING = "RANKING"
    PORTFOLIO = "PORTFOLIO"
    VALIDATION = "VALIDATION"


class PathEvidenceOutcome(str, Enum):
    """Failure / success / unresolved -- always read together with the
    record's ``evidence_scope``: a SUCCESS at PORTFOLIO scope says a portfolio
    containing this signal passed validation, never that the signal, its path
    or its mechanism is established."""

    FAILURE = "FAILURE"
    SUCCESS = "SUCCESS"
    UNRESOLVED = "UNRESOLVED"


class EvidenceScope(str, Enum):
    """What the outcome is evidence ABOUT."""

    #: A fact about the hypothesis pipeline itself: data, mandate, method.
    HYPOTHESIS = "HYPOTHESIS"
    #: In-sample discovery-window screening of the signal's own factor.
    SCREENING = "SCREENING"
    #: Out-of-sample evidence about the PORTFOLIO the signal was a member of.
    PORTFOLIO = "PORTFOLIO"


class PathEvidenceReason(str, Enum):
    # hypothesis plane
    PATH_UNRESOLVED = "PATH_UNRESOLVED"
    PATH_NOT_EXPRESSED = "PATH_NOT_EXPRESSED"
    EXPRESSION_EXCLUDED_BY_MANDATE = "EXPRESSION_EXCLUDED_BY_MANDATE"
    DOMAIN_UNAVAILABLE = "DOMAIN_UNAVAILABLE"
    NO_INSTRUMENT = "NO_INSTRUMENT"
    REMOVED_BY_CONSTRAINTS = "REMOVED_BY_CONSTRAINTS"
    MEASUREMENT_UNAVAILABLE = "MEASUREMENT_UNAVAILABLE"
    CONFIRMATION_ONLY_MEASUREMENT = "CONFIRMATION_ONLY_MEASUREMENT"
    NO_SIGNAL_RULE = "NO_SIGNAL_RULE"
    # screening / ranking
    NOT_SCREENED = "NOT_SCREENED"
    NO_DIAGNOSTIC_METHOD = "NO_DIAGNOSTIC_METHOD"
    SCREEN_NO_SUPPORT = "SCREEN_NO_SUPPORT"
    SCREEN_CONTRADICTS_SIGN = "SCREEN_CONTRADICTS_SIGN"
    SCREEN_INSUFFICIENT_DATA = "SCREEN_INSUFFICIENT_DATA"
    NOT_RANKABLE = "NOT_RANKABLE"
    SIGNAL_EXCLUDED_BY_MANDATE = "SIGNAL_EXCLUDED_BY_MANDATE"
    # portfolio gate
    NOT_ELIGIBLE_FOR_VALIDATION = "NOT_ELIGIBLE_FOR_VALIDATION"
    EXECUTION_UNSUPPORTED = "EXECUTION_UNSUPPORTED"
    COST_MODEL_INCOMPLETE = "COST_MODEL_INCOMPLETE"
    # validation of the portfolio (portfolio-scoped)
    PORTFOLIO_REJECTED = "PORTFOLIO_REJECTED"
    PORTFOLIO_COST_FRAGILE = "PORTFOLIO_COST_FRAGILE"
    PORTFOLIO_INCONCLUSIVE = "PORTFOLIO_INCONCLUSIVE"
    PORTFOLIO_VALIDATED = "PORTFOLIO_VALIDATED"
    PORTFOLIO_PRIOR_RESULT_CITED = "PORTFOLIO_PRIOR_RESULT_CITED"
