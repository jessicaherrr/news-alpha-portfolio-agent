"""Deterministic, versioned, explainable ranking over the six kept-separate
dimensions (prompt sections 4/5) -- never a blended score. This module is
pure: it operates only on already-built :class:`RankedResearchCandidate`
values, mirroring ``alpha_agent.opportunity.schemas``'s own "pure decision
logic, no Streamlit, no network, no registry" discipline.

Every ordinal reduction below is a DOCUMENTED, one-way "higher is more
research-relevant right now" mapping used ONLY to produce a reproducible sort
order (``RETRIEVAL_RANK_POLICY``) -- never itself rendered as a score, and
never a claim about expected return, probability of success, or trade
confidence (prompt section 5). The full, un-reduced evidence
(``RankDimensionScores``'s per-family detail dicts) stays on every candidate
so a reader can always check the reduction rather than trust it blindly.
"""
from __future__ import annotations

from alpha_agent.context_retrieval.schemas import (
    RETRIEVAL_RANK_POLICY,
    ContextMatchLevel,
    IndependentReplicationLevel,
    NextExperimentSuggestion,
    RankedResearchCandidate,
    ResearchCoverageLevel,
    ScientificReliabilityLevel,
)

__all__ = ["explain_rank", "rank_candidates", "suggest_next_experiment"]

#: RETRIEVAL_RANK_POLICY's own declared priority order -- index-aligned with
#: `RankedResearchCandidate.rank_key_components`.
_DIMENSION_LABELS: tuple[str, ...] = (
    "Context Match", "Scientific Reliability", "Research Coverage", "Execution Robustness",
    "Independent Replication", "User Fit",
)

_CONTEXT_MATCH_ORDER: dict[ContextMatchLevel, int] = {
    ContextMatchLevel.NONE: 0, ContextMatchLevel.WEAK: 1,
    ContextMatchLevel.MODERATE: 2, ContextMatchLevel.STRONG: 3,
}
#: Conservative, MEANING-PRESERVING ordinal (semantic hardening patch,
#: section 1): INCONCLUSIVE is its own rung, strictly between "no real
#: verdict yet" and "a definitive REJECT" -- never folded into REJECTED.
#: MIXED (genuine disagreement across families) sits ABOVE every single-
#: verdict outcome except PASSED, since disagreement is real, informative
#: evidence, never resolved by picking the most favorable side.
_SCIENTIFIC_RELIABILITY_ORDER: dict[ScientificReliabilityLevel, int] = {
    ScientificReliabilityLevel.NO_EVIDENCE: 0, ScientificReliabilityLevel.NOT_ADJUDICATED: 1,
    ScientificReliabilityLevel.INCONCLUSIVE: 2, ScientificReliabilityLevel.REJECTED: 3,
    ScientificReliabilityLevel.MIXED: 4, ScientificReliabilityLevel.PASSED: 5,
}
_RESEARCH_COVERAGE_ORDER: dict[ResearchCoverageLevel, int] = {
    ResearchCoverageLevel.NOT_MAPPED: 0, ResearchCoverageLevel.MAPPED_UNTESTED: 1,
    ResearchCoverageLevel.PARTIALLY_TESTED: 2, ResearchCoverageLevel.FULLY_TESTED: 3,
}
#: A known survivor ranks above unknown, which ranks above a known failure --
#: an unevaluated dimension is neutral, never worse than a documented failure.
_ROBUSTNESS_ORDER: dict[str, int] = {"FAILED_STRESS": 0, "NOT_EVALUATED": 1, "EVALUATED": 2}
#: Always a single value in V1 (`IndependentReplicationLevel.NOT_AVAILABLE`)
#: -- see that enum's own docstring. This ordinal is therefore a constant
#: today and never discriminates between candidates; it stays a real,
#: separate lookup (rather than a hardcoded `0` inline) so a future phase
#: that adds a real independent-replication level only needs to extend this
#: table, never touch the ranking logic itself.
_INDEPENDENT_REPLICATION_ORDER: dict[IndependentReplicationLevel, int] = {
    IndependentReplicationLevel.NOT_AVAILABLE: 0,
}
_USER_FIT_LABEL_ORDER: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _execution_robustness_ordinal(detail: dict[str, str]) -> int:
    if not detail:
        return _ROBUSTNESS_ORDER["NOT_EVALUATED"]
    return max(_ROBUSTNESS_ORDER.get(v, _ROBUSTNESS_ORDER["NOT_EVALUATED"]) for v in detail.values())


def _user_fit_ordinal(detail: dict[str, str]) -> int:
    """``detail`` values are ``"{personalization_state}:{label}"`` (see
    ``retrieval.py``). A ``NOT_PERSONALIZED`` entry is neutral, never a
    penalty -- ``alpha_agent.recommendation.fit``'s own docstring: its
    `label` is `"LOW"` only because nothing was scored, never a real
    measurement of poor fit, and this module must not re-introduce the exact
    bug that discipline was written to prevent.

    Semantic hardening patch, section 4: the neutral ``MEDIUM`` fallback
    applies ONLY when there is no real personalized evidence at all (an
    empty ``detail``, or every entry ``NOT_PERSONALIZED``). Once at least
    one genuinely personalized entry exists, its real label -- including
    ``LOW`` -- must be preserved, never silently floored up to ``MEDIUM``
    (the previous bug: starting the running-best at ``MEDIUM`` made a
    real, measured ``LOW`` unreachable)."""
    personalized_scores: list[int] = []
    for token in detail.values():
        state, _, label = token.partition(":")
        if state == "NOT_PERSONALIZED":
            continue
        personalized_scores.append(_USER_FIT_LABEL_ORDER.get(label, _USER_FIT_LABEL_ORDER["MEDIUM"]))
    if not personalized_scores:
        return _USER_FIT_LABEL_ORDER["MEDIUM"]
    return max(personalized_scores)


def _rank_key_components(candidate: RankedResearchCandidate) -> tuple[int, ...]:
    scores = candidate.scores
    return (
        _CONTEXT_MATCH_ORDER[scores.context_match],
        _SCIENTIFIC_RELIABILITY_ORDER[scores.scientific_reliability],
        _RESEARCH_COVERAGE_ORDER[scores.research_coverage],
        _execution_robustness_ordinal(scores.execution_robustness_detail),
        _INDEPENDENT_REPLICATION_ORDER[scores.independent_replication],
        _user_fit_ordinal(scores.user_fit_detail),
    )


def rank_candidates(candidates: tuple[RankedResearchCandidate, ...]) -> tuple[RankedResearchCandidate, ...]:
    """Sorts by :data:`RETRIEVAL_RANK_POLICY`'s declared priority order
    (descending on every dimension), with the mechanism's own name as the
    final deterministic tiebreak so repeated calls over identical input
    always return an identical order. Returns NEW candidate instances with
    ``rank_key_components``/``retrieval_rank`` filled in (the input
    candidates are frozen and do not yet know their sort position)."""
    decorated = [(_rank_key_components(c), c) for c in candidates]
    decorated.sort(key=lambda pair: (*(-x for x in pair[0]), pair[1].mechanism.value))
    return tuple(
        c.model_copy(update={"rank_key_components": components, "retrieval_rank": position})
        for position, (components, c) in enumerate(decorated, start=1)
    )


def explain_rank(ranked: tuple[RankedResearchCandidate, ...]) -> str:
    """Explains #1 vs #2 (prompt section 7's acceptance requirement):
    states the FIRST dimension (in declared priority order) where the two
    top candidates actually differ -- that dimension is exactly what decided
    the order, since every dimension ahead of it was tied."""
    if not ranked:
        return "No relevant mechanism candidates were retrieved for this context."
    if len(ranked) == 1:
        return (
            f"Only one relevant mechanism candidate ({ranked[0].mechanism.value}) was retrieved for this "
            "context -- no #1 vs #2 comparison applies."
        )
    a, b = ranked[0], ranked[1]
    for i, label in enumerate(_DIMENSION_LABELS):
        if a.rank_key_components[i] != b.rank_key_components[i]:
            return (
                f"{a.mechanism.value} ranks #1 over {b.mechanism.value} primarily on {label}: "
                f"{a.mechanism.value}={a.rank_key_components[i]} vs {b.mechanism.value}={b.rank_key_components[i]} "
                f"(ordinal, {RETRIEVAL_RANK_POLICY} priority order -- research prioritization only, never expected "
                "return or trade confidence)."
            )
    return (
        f"{a.mechanism.value} and {b.mechanism.value} tied on every ranked dimension; final order was broken by "
        "mechanism name alone."
    )


#: Coverage levels "suggest next experiment" treats as genuinely underexplored.
_UNDEREXPLORED_COVERAGE = (ResearchCoverageLevel.NOT_MAPPED, ResearchCoverageLevel.MAPPED_UNTESTED)


def suggest_next_experiment(ranked: tuple[RankedResearchCandidate, ...]) -> NextExperimentSuggestion | None:
    """Highest INFORMATION/RESEARCH VALUE, never predicted profit (prompt
    section 6): prefers a candidate whose research coverage is genuinely
    thin (never mapped, or mapped but never tested) among the ones this
    context actually matches -- re-confirming an already-adjudicated
    hypothesis (high Scientific Reliability) is explicitly NOT what this
    picks, since that adds little new information. Falls back to the
    best-context-match candidate overall when every retrieved candidate
    already has some coverage, with an honest note that nothing here is
    clearly underexplored."""
    if not ranked:
        return None

    underexplored = [c for c in ranked if c.scores.research_coverage in _UNDEREXPLORED_COVERAGE]
    pool = underexplored or list(ranked)
    best = min(
        pool,
        key=lambda c: (
            -_CONTEXT_MATCH_ORDER[c.scores.context_match],
            _RESEARCH_COVERAGE_ORDER[c.scores.research_coverage],
            c.mechanism.value,
        ),
    )

    concept = next((f.concept for f in best.factors if f.researchability.value == "AVAILABLE"), None)
    if concept is None and best.factors:
        concept = best.factors[0].concept

    if underexplored:
        rationale = (
            f"{best.mechanism.value} has the strongest Context Match ({best.scores.context_match.value}) among "
            f"mechanisms with little or no existing research coverage ({best.scores.research_coverage.value}) "
            "for this context."
        )
        information_value_reason = (
            "Testing an underexplored, context-relevant mechanism adds genuinely new information; re-running an "
            "already-adjudicated hypothesis mostly re-confirms what is already known."
        )
    else:
        rationale = (
            f"Every retrieved mechanism already has some research coverage; {best.mechanism.value} has the "
            f"strongest Context Match ({best.scores.context_match.value}) among them, making it the best "
            "incremental follow-up for this context."
        )
        information_value_reason = (
            "No clearly underexplored mechanism candidate exists for this context today -- this suggestion "
            "prioritizes context relevance among already-covered mechanisms instead."
        )

    return NextExperimentSuggestion(
        mechanism=best.mechanism, concept=concept, rationale=rationale,
        information_value_reason=information_value_reason,
    )
