"""Phase 2 -- the typed Personal Alpha Memory read model (prompt 2).

Turns isolated registry experiments into reusable PERSONAL RESEARCH MEMORY.
Every object here is a **read model** over existing sources of truth
(:mod:`alpha_agent.registry`, :mod:`alpha_agent.translation`) -- it is never a
second scientific database (prompt 2 section 4). Nothing in this package
writes to the registry, calls Claude, calls a market-data API, or runs a
backtest.

``AlphaResearchObject`` deliberately does NOT mean "proven profitable alpha".
It is a RESEARCH OBJECT: an item may be REJECT, INCONCLUSIVE, NOT_ADJUDICATED
or exploratory and still belong in research memory (prompt 2 section 1). This
module never computes an Alpha Score, an expected return, a probability of
success, or any single combined ranking number (section 10/21) -- those
belong to a later phase this file must never anticipate.

MECHANISM != FACTOR != STRATEGY != EXPERIMENT (section 2) is structural
here, not cosmetic:

* An ``EconomicMechanism`` (Phase 1's closed enum) is a PARENT semantic
  dimension, never the Factor identity itself (identity-hardening patch,
  section 1) -- two strategy families that both implement the same
  mechanism are NOT automatically the same Factor.
* :class:`FactorIdentity` -- mechanism PLUS a real, typed structural
  signature (never root- or parameter-scoped). One Factor may still be
  implemented by several Strategy variants, but only when their structural
  signatures actually agree -- see :mod:`alpha_agent.alpha_memory.factor_identity`.
* :class:`StrategyVariantEvidence` -- one ``strategy_family`` (e.g.
  ``"tsmom"``), root-scoped. A Strategy may carry several Experiments (e.g.
  the same family tested across parameter neighbours, splits, or
  re-executions).
* :class:`ExperimentEvidenceRef` -- one real registry ``experiment_identity``.
  Never fabricated, never re-derived: every field here is read verbatim from
  the registry (section 9).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from alpha_agent.agents.context import FailureMemoryDigest
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.enums import Authority, RegistryVerdict, TrialRole

__all__ = [
    "ALPHA_OBJECT_SCHEMA",
    "FACTOR_IDENTITY_SCHEMA",
    "AlphaResearchObject",
    "EvidenceProfile",
    "ExperimentEvidenceRef",
    "FactorIdentity",
    "MechanismMemoryLookup",
    "MechanismMemoryMatchKind",
    "ResearchMaturity",
    "StrategyVariantEvidence",
    "WhatChanged",
]

#: v2 (identity-hardening patch): factor identity is mechanism + real
#: evidence, never mechanism alone. v3 (final Phase 2 semantic fix): that
#: evidence is now the EXACT strategy-family set a Factor bundles, never a
#: required_data-based signature match -- equal input-data requirements
#: prove only shared data requirements, not equal quantitative factor
#: representation (alpha_agent.alpha_memory.factor_identity). No schema
#: value here is forward-compatible with an earlier one; nothing persists
#: these values outside one call, so no migration is needed.
FACTOR_IDENTITY_SCHEMA = "factor-identity/3"
#: v3: identity is root + FactorIdentity.factor_identity, never root + mechanism.
ALPHA_OBJECT_SCHEMA = "alpha-research-object/3"


class ResearchMaturity(str, Enum):
    """How much research exists -- NEVER how profitable or promising it is
    (prompt 2 section 11). Many REJECT experiments can be ADJUDICATED (high
    maturity) while the scientific evidence stays REJECT; a mechanism with no
    candidate-family representation at all stays IDEA regardless of how
    interesting the narrative is.

    Monotonic ladder, each rung strictly more accumulated evidence than the
    last: IDEA (no representation) < FORMALIZED (representable, never run)
    < TESTED (>=1 real execution attempt) < REPLICATED (>=2 real experiments)
    < ADJUDICATED (>=1 experiment reached a genuine headline verdict).
    """

    IDEA = "IDEA"
    FORMALIZED = "FORMALIZED"
    TESTED = "TESTED"
    REPLICATED = "REPLICATED"
    ADJUDICATED = "ADJUDICATED"


class FactorIdentity(BaseModel):
    """WHICH economic idea this is -- NEVER root- or parameter-scoped, but
    also NEVER just the mechanism (identity-hardening patch, section 1:
    "Factor must not equal Mechanism"). One Factor may still be implemented
    by several Strategy variants (section 8), but only when a caller holds
    genuinely typed proof they compute the same factor.

    Conservative V1 rule (final Phase 2 semantic fix, section 1): identity is
    a fingerprint over the closed ``EconomicMechanism`` value PLUS the EXACT
    set of strategy families this Factor bundles --
    :mod:`alpha_agent.alpha_memory.factor_identity` bundles exactly ONE
    family per Factor by default. Equal declared ``required_data`` (the data
    SHAPE a family needs) is NOT sufficient proof of equivalence: tsmom and
    ma_trend both only need "one continuous or raw OHLC price series per
    root", yet they are mathematically different transforms (price
    differencing vs moving-average crossover), so V1 keeps them as separate
    Factors -- "false duplication is preferable to false merging". A future
    phase with genuine typed transform/feature-kind equivalence evidence
    could justify a multi-family Factor; none exists in this repository
    today. ``structural_signature`` (``required_data``) is exposed purely as
    INFORMATIONAL context about the (single, in every real case) family's
    own data-shape requirement -- it plays no role in identity computation.
    ``related_strategy_families`` is the exact family set this Factor was
    built from, never the mechanism's whole mapped family list.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = FACTOR_IDENTITY_SCHEMA
    mechanism: EconomicMechanism
    factor_identity: str
    #: INFORMATIONAL ONLY (sorted ``required_data`` strings) -- what market
    #: data this Factor's (single, in every real V1 case) family needs. NOT
    #: part of the identity fingerprint and NOT proof two families could
    #: share a Factor; two families needing identical data can still be
    #: mathematically different transforms (see class docstring).
    structural_signature: tuple[str, ...] = ()
    related_strategy_families: tuple[str, ...] = ()
    provenance_note: str


class StrategyVariantEvidence(BaseModel):
    """WHICH concrete implementation of the factor this is -- one
    ``strategy_family`` (e.g. ``"tsmom"``), root-scoped.

    ``canonical_params`` is the small (few numeric fields) params dict of the
    CANONICAL trial, when one exists -- never a large artifact payload
    (section 5: "do not duplicate large artifact payloads"), and used only to
    support the "what changed" parameter comparison (section 15).
    """

    model_config = {"frozen": True, "extra": "forbid"}

    strategy_family: str
    strategy_display_name: str
    n_experiments: int = 0
    canonical_experiment_ids: tuple[str, ...] = ()
    strategy_fingerprints: tuple[str, ...] = ()
    canonical_params: dict | None = None
    digest: FailureMemoryDigest


class ExperimentEvidenceRef(BaseModel):
    """One real registry experiment, referenced -- never recomputed.

    Every field is read verbatim from :class:`alpha_agent.registry.sqlite_registry.ExperimentView`.
    No PnL, fill, or other large official-metric payload is duplicated here;
    a consumer that needs those reads the registry directly by
    ``experiment_id`` (CLAUDE.md: the LLM/UI is never the source of official
    PnL/fill/pass-fail truth).
    """

    model_config = {"frozen": True, "extra": "forbid"}

    experiment_id: str
    experiment_identity: str
    strategy_family: str
    trial_role: TrialRole
    parameter_variant_label: str
    headline_verdict: RegistryVerdict | None = None
    reason_codes: tuple[str, ...] = ()
    n_valid_attempts: int = 0
    n_invalid_attempts: int = 0
    has_valid_authoritative_result: bool = False
    holdout_eligible: bool = False
    authority: Authority = Authority.AUTHORITATIVE
    market_window: str = ""


class EvidenceProfile(BaseModel):
    """Multidimensional, categorical, explainable evidence -- NEVER combined
    into one number (prompt 2 section 10: no Alpha Score, no Confidence %,
    no Expected Return, no Probability of Success).

    Every dimension is read directly off committed ``ResultRecord`` fields
    (``cost_stress`` / ``parameter_stability`` / ``regime_evidence``) using
    the same explicit-only-evidence discipline
    :func:`alpha_agent.ui.services.gate_state` already established: a
    dimension reads a definite state only from an explicit committed marker
    (a reason code or a non-empty evidence dict), never inferred from
    absence.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    #: strategy_family -> canonical-trial verdict label, or "NO_CANONICAL_TRIAL"
    #: when no canonical trial exists for that family yet, or
    #: "MIXED(...)" when this root has more than one canonical row under the
    #: same family disagreeing (never silently collapsed to one).
    scientific_evidence: dict[str, str] = Field(default_factory=dict)
    #: count of real registry experiment IDENTITIES related to this Factor
    #: (canonical + neighbour + ablation + variant rows alike) -- NOT
    #: independent replication (identity-hardening patch, section 6: a
    #: neighbour/ablation/re-execution is coverage of the SAME hypothesis
    #: family, never a separate independent reproduction of the result).
    #: ``independent_replication`` is reserved terminology for a future
    #: phase where genuinely independent replications (e.g. a different
    #: researcher/dataset vintage reproducing the same result) exist.
    related_experiment_count: int = 0
    related_experiment_label: str = ""
    cost_robustness: str = "NOT_EVALUATED"
    parameter_stability: str = "NOT_EVALUATED"
    regime_breadth: str = "NOT_EVALUATED"


class WhatChanged(BaseModel):
    """"Is this genuinely a different idea, or another implementation /
    parameter variant?" (prompt 2 section 15). Only dimensions the actual
    structured evidence can compare are populated -- never inferred from free
    text (section 15: "do not claim novelty from free text")."""

    model_config = {"frozen": True, "extra": "forbid"}

    same: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    parameter_differences: dict[str, tuple[float, float]] = Field(default_factory=dict)
    notes: str = ""


class AlphaResearchObject(BaseModel):
    """The Phase 2 research memory unit: one (root, Factor) pair with real,
    registry-grounded evidence.

    Deliberately NOT named ``VerifiedAlpha`` / ``WinningAlpha`` / ``ProvenAlpha``
    (section 1) -- an object here may carry an all-REJECT evidence profile and
    still be a first-class, useful piece of research memory. Identity is a
    deterministic composition of stable identities only (``root_symbol`` +
    :attr:`FactorIdentity.factor_identity`), never an ``experiment_id`` and
    never ``root_symbol`` + ``mechanism`` alone (identity-hardening patch,
    section 2) -- ``mechanism`` stays on this object as a PARENT semantic
    dimension for display/filtering, but two objects with the same mechanism
    on the same root are only the SAME object when their Factor identity
    also agrees.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = ALPHA_OBJECT_SCHEMA
    alpha_id: str
    root_symbol: str
    mechanism: EconomicMechanism
    mechanism_provenance: str
    factor: FactorIdentity
    strategy_variants: tuple[StrategyVariantEvidence, ...] = ()
    experiments: tuple[ExperimentEvidenceRef, ...] = ()
    evidence_profile: EvidenceProfile
    research_maturity: ResearchMaturity
    repeated_failure_reason_codes: dict[str, int] = Field(default_factory=dict)
    repeated_failure_classes: dict[str, int] = Field(default_factory=dict)
    engineering_lessons: tuple[str, ...] = ()
    #: other AlphaResearchObjects on the SAME root whose mapped strategy
    #: families overlap this one's -- e.g. BREAKOUT / FAILED_BREAKOUT /
    #: VOLATILITY_BREAKOUT all currently resolve to the single "breakout"
    #: family, an honest fact worth surfacing, never silently merged into one
    #: object (section 7: "false merging is worse than duplication").
    related_alpha_ids: tuple[str, ...] = ()
    summary: str = ""


class MechanismMemoryMatchKind(str, Enum):
    """How confidently a Phase 1 candidate resolves to Personal Alpha Memory
    (identity-hardening patch, section 5; hardened further by the final
    Phase 2 semantic fix, section 2). Mechanism alone can NEVER establish an
    exact Factor match: Phase 1 does not yet compute a candidate's own
    ``strategy_family``, so a mechanism-only lookup can claim at most
    RELATED_MECHANISM, however few (even exactly one) historical Factor
    objects exist under that mechanism -- cardinality is not proof.
    MATCHING_FACTOR requires an explicit candidate ``strategy_family`` whose
    own deterministic ``FactorIdentity`` (computed the same way any stored
    object's was) equals a stored object's ``factor_identity`` exactly."""

    #: no AlphaResearchObject exists for this (root, mechanism) at all.
    NONE = "NONE"
    #: a candidate strategy_family was supplied, its own deterministic
    #: FactorIdentity was computed, and it equals a stored object's exactly.
    MATCHING_FACTOR = "MATCHING_FACTOR"
    #: real evidence exists under this mechanism, but no candidate Factor
    #: was supplied (or none of its exact matches were found) -- mechanism
    #: matches, Factor equality cannot be established.
    RELATED_MECHANISM = "RELATED_MECHANISM"


class MechanismMemoryLookup(BaseModel):
    """The typed answer to "have I researched this mechanism before?" for a
    Phase 1 candidate (section 5/17) -- never a silent single-object pick,
    and never MATCHING_FACTOR without an explicit, exactly-equal candidate
    FactorIdentity. It is acceptable for a mechanism-only lookup (no
    ``strategy_family`` known yet, today's real Phase 1 integration point)
    to return only RELATED_MECHANISM or NONE."""

    model_config = {"frozen": True, "extra": "forbid"}

    mechanism: EconomicMechanism
    root_symbol: str
    match_kind: MechanismMemoryMatchKind
    objects: tuple[AlphaResearchObject, ...] = ()
