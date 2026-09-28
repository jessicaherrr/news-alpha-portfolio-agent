"""The Phase 3 orchestrating pipeline: Observation + MarketContextFingerprint
+ ExperimentRegistry -> Related Mechanisms + Relevant Factors + Strategy
Evidence + Repeated Failures + Research Gaps + a deterministic, explainable
rank + a Suggested Next Experiment (prompt section 6).

Pure and network-free, exactly like
``alpha_agent.translation.pipeline.build_observation_translation``: the only
network-shaped input this function accepts is an already-built ``Observation``
and an already-assembled ``MarketContextFingerprint`` /
``related_prior_event_occurrences`` tuple -- the real-data READ boundary is
``alpha_agent.ui.context_retrieval_context``, mirroring
``alpha_agent.ui.translation_context``'s own role for Phase 1.

Composes THREE already-frozen sources of truth rather than re-deriving any of
them (module docstring in ``__init__.py``):

* ``alpha_agent.translation`` (Phase 1) -- which mechanisms/factors this
  observation's category proposes, and researchability, re-derived here
  against live ``FeatureRegistry`` state exactly like Phase 1's own pipeline
  does (never a stale cached claim).
* ``alpha_agent.alpha_memory`` (Phase 2) -- real Strategy Evidence
  (``AlphaResearchObject``) via the SAME mechanism-only lookup Phase 1's own
  hand-off uses (``mechanism_memory_lookup``, never a stronger match claim
  than that function itself allows).
* ``alpha_agent.recommendation.fit`` (Phase B1) -- User Fit, reused verbatim
  against each mapped family's own canonical trial.

Deliberately DETERMINISTIC-LIBRARY ONLY (never Claude-sourced): Phase 3 does
not re-propose mechanisms, it retrieves and ranks the ones
``mechanism_library``'s deterministic, ALWAYS-AVAILABLE catalog already
declares for this observation's category -- consistent with prompt section 5
("a deterministic/versioned/explainable retrieval rank").
"""
from __future__ import annotations

from alpha_agent.alpha_memory.lookup import mechanism_memory_lookup
from alpha_agent.context_retrieval.mechanism_signals import context_match_for
from alpha_agent.context_retrieval.ranking import (
    explain_rank,
    rank_candidates,
    suggest_next_experiment,
)
from alpha_agent.context_retrieval.schemas import (
    ContextRetrievalResult,
    IndependentReplicationLevel,
    MarketContextFingerprint,
    RankDimensionScores,
    RankedResearchCandidate,
    RelatedPriorEventOccurrence,
    ResearchCoverageLevel,
    ScientificReliabilityLevel,
)
from alpha_agent.features.registry import REGISTRY as FEATURE_REGISTRY
from alpha_agent.features.registry import FeatureRegistry
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.market_intel.news_schemas import NewsCategory
from alpha_agent.recommendation.fit import score_user_fit
from alpha_agent.recommendation.profile import DEFAULT_PROFILE, InvestorProfile
from alpha_agent.registry.enums import AssetDomain, TrialRole
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.translation.mechanism_library import (
    CATEGORY_MECHANISM_TEMPLATES,
    FactorTemplate,
    MechanismTemplate,
)
from alpha_agent.translation.research_memory import (
    MECHANISM_TO_KNOWN_FAMILIES,
    ResearchMemoryNote,
    mechanism_research_memory,
)
from alpha_agent.translation.researchability import classify_factor
from alpha_agent.translation.schemas import (
    FactorCandidate,
    FactorProvenance,
    MechanismCandidate,
    MechanismProvenance,
    Observation,
    ResearchabilityStatus,
)

__all__ = ["build_context_retrieval"]


def _category_of(value: str) -> NewsCategory:
    try:
        return NewsCategory(value)
    except ValueError:
        return NewsCategory.OTHER


def _classify(raw: FactorTemplate, feature_registry: FeatureRegistry) -> FactorCandidate:
    """Deterministic-library counterpart to
    ``alpha_agent.translation.pipeline._classify_factor`` with
    ``claude_sourced`` always False (see module docstring) -- reimplemented
    here (rather than importing that private function across packages) since
    it is a thin ~10-line wrapper over the public ``classify_factor``."""
    status, reason, available, missing = classify_factor(
        proposed_feature_kinds=raw.proposed_feature_kinds,
        required_external_data=raw.required_external_data,
        structurally_expressible=raw.structurally_expressible,
        registry=feature_registry,
    )
    if status in (ResearchabilityStatus.DATA_MISSING, ResearchabilityStatus.NOT_EXECUTABLE):
        provenance = FactorProvenance.UNSUPPORTED_IDEA
    elif len(available) > 1:
        provenance = FactorProvenance.DERIVED_FROM_REGISTERED_FEATURES
    else:
        provenance = FactorProvenance.REGISTERED_FEATURE
    return FactorCandidate(
        concept=raw.concept, mechanism=raw.mechanism, transform_or_proxy=raw.transform_or_proxy,
        proposed_feature_kinds=raw.proposed_feature_kinds, available_feature_kinds=available,
        required_external_data=raw.required_external_data, researchability=status,
        researchability_reason=reason, missing_requirements=missing, provenance=provenance,
    )


#: The only three real, committed adjudicated-verdict strings
#: ``alpha_memory.EvidenceProfile.scientific_evidence`` ever carries for one
#: family (mirrors ``RegistryVerdict`` minus ``NOT_ADJUDICATED``, which is
#: represented by that family's key being ABSENT from ``real_verdicts``
#: below, or by the literal strings ``"NOT_ADJUDICATED"``/
#: ``"NO_CANONICAL_TRIAL"``).
_REAL_VERDICTS = frozenset({"PASS", "REJECT", "INCONCLUSIVE"})


def _scientific_reliability(evidence: dict[str, str]) -> ScientificReliabilityLevel:
    """A CONSERVATIVE, meaning-preserving reduction of the real, committed
    per-family verdicts (``alpha_memory.EvidenceProfile.scientific_evidence``)
    -- see ``ScientificReliabilityLevel``'s own docstring for the semantic
    hardening patch this fixes (never fold INCONCLUSIVE into REJECTED, never
    cherry-pick the most favorable verdict when families disagree).

    A family's own already-mixed string (``"MIXED(...)"``, meaning that
    ONE family's canonical rows already disagree) forces the overall result
    to ``MIXED`` immediately, same as more than one distinct real verdict
    being found across families -- both are genuine disagreement, never
    resolved by picking a side."""
    if not evidence:
        return ScientificReliabilityLevel.NO_EVIDENCE
    values = set(evidence.values())
    if any(v.startswith("MIXED(") for v in values):
        return ScientificReliabilityLevel.MIXED
    real_verdicts = values & _REAL_VERDICTS
    if len(real_verdicts) > 1:
        return ScientificReliabilityLevel.MIXED
    if real_verdicts == {"PASS"}:
        return ScientificReliabilityLevel.PASSED
    if real_verdicts == {"REJECT"}:
        return ScientificReliabilityLevel.REJECTED
    if real_verdicts == {"INCONCLUSIVE"}:
        return ScientificReliabilityLevel.INCONCLUSIVE
    return ScientificReliabilityLevel.NOT_ADJUDICATED


def _independent_replication() -> IndependentReplicationLevel:
    """Always ``NOT_AVAILABLE`` in V1 -- see
    ``IndependentReplicationLevel``'s own docstring and the module
    docstring's "INDEPENDENT REPLICATION != RELATED EXPERIMENT COUNT"
    section. A real per-family related-experiment count is still gathered
    and surfaced (see ``related_experiment_counts`` in ``_build_candidate``),
    just never relabeled as replication."""
    return IndependentReplicationLevel.NOT_AVAILABLE


def _research_coverage(note: ResearchMemoryNote) -> tuple[ResearchCoverageLevel, str]:
    if note.mapped_family_count == 0:
        return (
            ResearchCoverageLevel.NOT_MAPPED,
            "No strategy family in the current candidate manifest maps to this mechanism yet.",
        )
    if note.tested_family_count == 0:
        return (
            ResearchCoverageLevel.MAPPED_UNTESTED,
            f"{note.mapped_family_count} mapped family(ies), none tested yet on this root.",
        )
    if note.tested_family_count < note.mapped_family_count:
        return (
            ResearchCoverageLevel.PARTIALLY_TESTED,
            f"{note.tested_family_count}/{note.mapped_family_count} mapped family(ies) tested on this root.",
        )
    return (
        ResearchCoverageLevel.FULLY_TESTED,
        f"{note.tested_family_count}/{note.mapped_family_count} mapped family(ies) tested on this root.",
    )


def _user_fit_detail(
    strategy_families: tuple[str, ...], *, root_symbol: str, registry: ExperimentRegistry, profile: InvestorProfile,
) -> dict[str, str]:
    """One ``alpha_agent.recommendation.fit.score_user_fit`` call per mapped
    family, against that family's own canonical trial (never a fabricated
    one) -- reused verbatim, never re-scored. A family with no canonical
    trial yet is simply absent from the result (never a guessed neutral
    entry)."""
    out: dict[str, str] = {}
    for family in strategy_families:
        canonical = [
            v for v in registry.experiments(
                strategy_family=family, root_symbol=root_symbol,
                asset_domain=AssetDomain.FUTURES, authoritative_only=True,
            )
            if v.experiment.trial_role is TrialRole.CANONICAL
        ]
        if not canonical:
            continue
        result = canonical[0].result
        result_dict = {"n_trades": result.n_trades, "n_oos_days": result.n_oos_days} if result is not None else None
        fit = score_user_fit(strategy_family=family, result=result_dict, profile=profile)
        out[family] = f"{fit.personalization_state.value}:{fit.label}"
    return out


def _build_candidate(
    mechanism: EconomicMechanism,
    templates: tuple[MechanismTemplate, ...],
    *,
    root_symbol: str,
    registry: ExperimentRegistry,
    feature_registry: FeatureRegistry,
    context: MarketContextFingerprint,
    profile: InvestorProfile,
) -> RankedResearchCandidate:
    first = templates[0]
    mechanism_candidate = MechanismCandidate(
        mechanism=mechanism, explanation=first.explanation, causal_chain=first.causal_chain,
        evidence_basis=first.evidence_basis, provenance=MechanismProvenance.DETERMINISTIC_LIBRARY,
    )
    factors = tuple(_classify(f, feature_registry) for t in templates for f in t.factors)

    lookup = mechanism_memory_lookup(registry, mechanism=mechanism, root_symbol=root_symbol, strategy_family=None)
    objects = lookup.objects

    scientific_evidence: dict[str, str] = {}
    robustness_detail: dict[str, str] = {}
    related_experiment_counts: dict[str, int] = {}
    alpha_ids: list[str] = []
    strategy_evidence_summary: list[str] = []
    reason_codes: dict[str, int] = {}
    failure_classes: dict[str, int] = {}
    for obj in objects:
        alpha_ids.append(obj.alpha_id)
        strategy_evidence_summary.append(obj.summary)
        scientific_evidence.update(obj.evidence_profile.scientific_evidence)
        for family in obj.evidence_profile.scientific_evidence:
            robustness_detail[family] = obj.evidence_profile.cost_robustness
            related_experiment_counts[family] = obj.evidence_profile.related_experiment_count
        for code, n in obj.repeated_failure_reason_codes.items():
            reason_codes[code] = reason_codes.get(code, 0) + n
        for cls, n in obj.repeated_failure_classes.items():
            failure_classes[cls] = failure_classes.get(cls, 0) + n

    strategy_families = tuple(sorted(scientific_evidence)) or tuple(
        sorted(MECHANISM_TO_KNOWN_FAMILIES.get(mechanism, ()))
    )
    note = mechanism_research_memory(mechanisms=(mechanism,), root_symbol=root_symbol, registry=registry)
    user_fit_detail = _user_fit_detail(
        strategy_families, root_symbol=root_symbol, registry=registry, profile=profile,
    )

    context_match, context_match_reasons = context_match_for(mechanism, context)
    scientific_reliability = _scientific_reliability(scientific_evidence)
    coverage_level, coverage_detail = _research_coverage(note)

    research_gap_notes = tuple(
        f"{f.concept}: {f.researchability.value} ({f.researchability_reason})"
        for f in factors if f.researchability != ResearchabilityStatus.AVAILABLE
    )
    if note.underexplored and not research_gap_notes:
        research_gap_notes = (note.summary,)

    scores = RankDimensionScores(
        context_match=context_match,
        context_match_reasons=context_match_reasons,
        scientific_reliability=scientific_reliability,
        scientific_reliability_detail=scientific_evidence,
        execution_robustness_detail=robustness_detail,
        independent_replication=_independent_replication(),
        related_experiment_counts=related_experiment_counts,
        user_fit_detail=user_fit_detail,
        research_coverage=coverage_level,
        research_coverage_detail=coverage_detail,
    )

    return RankedResearchCandidate(
        mechanism=mechanism,
        mechanism_candidate=mechanism_candidate,
        factors=factors,
        strategy_families=strategy_families,
        alpha_ids=tuple(alpha_ids),
        strategy_evidence_summary=tuple(strategy_evidence_summary),
        repeated_failure_reason_codes=reason_codes,
        repeated_failure_classes=failure_classes,
        research_gap_notes=research_gap_notes,
        scores=scores,
        rank_key_components=(),
        retrieval_rank=0,
    )


def build_context_retrieval(
    observation: Observation,
    *,
    context: MarketContextFingerprint,
    registry: ExperimentRegistry,
    feature_registry: FeatureRegistry | None = None,
    related_prior_event_occurrences: tuple[RelatedPriorEventOccurrence, ...] = (),
    investor_profile: InvestorProfile | None = None,
) -> ContextRetrievalResult:
    """The one Phase 3 entry point. ``context`` must already be built for
    ``observation.root_symbol`` (mismatches are the caller's responsibility,
    exactly like ``build_observation_translation`` trusts its own
    ``Observation`` argument)."""
    feature_registry = feature_registry or FEATURE_REGISTRY
    profile = investor_profile or DEFAULT_PROFILE
    category = _category_of(observation.structured_attributes.get("category", ""))
    templates = CATEGORY_MECHANISM_TEMPLATES.get(category, ())

    grouped: dict[EconomicMechanism, list[MechanismTemplate]] = {}
    for template in templates:
        grouped.setdefault(template.mechanism, []).append(template)

    candidates = tuple(
        _build_candidate(
            mechanism, tuple(group), root_symbol=observation.root_symbol, registry=registry,
            feature_registry=feature_registry, context=context, profile=profile,
        )
        for mechanism, group in grouped.items()
    )
    ranked = rank_candidates(candidates)
    evidence_gaps = tuple(sorted({note for c in ranked for note in c.research_gap_notes}))

    return ContextRetrievalResult(
        context=context,
        related_prior_event_occurrences=related_prior_event_occurrences,
        ranked_candidates=ranked,
        rank_explanation=explain_rank(ranked),
        evidence_gaps=evidence_gaps,
        suggested_next_experiment=suggest_next_experiment(ranked),
    )
