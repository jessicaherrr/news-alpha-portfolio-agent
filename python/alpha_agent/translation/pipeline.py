"""Phase 1's deterministic translation pipeline: Observation -> Mechanism
Candidates -> Measurable Variables -> Factor Candidates -> Researchability ->
Research Memory -> Falsifiable Hypothesis (or an honest Research Gap).

``build_observation_translation`` is a pure function of an already-built
``Observation``, an open ``ExperimentRegistry``, and an OPTIONAL, already-
validated Claude proposal (``alpha_agent.agents.mechanism_agent
.MechanismTranslationProposal``). It never touches the network itself -- the
only place an LLM is called is ``alpha_agent.agents.mechanism_agent``, and
only when a caller explicitly asks for it (prompt 1 section 14: no automatic
paid network calls).

Claude may propose mechanism/factor narrative text; it never gets the final
word on researchability, feature availability, or point-in-time status --
every ``FactorCandidate``/``MeasurableVariable`` field beyond the narrative
ones is (re)computed here against live repository state regardless of
``generated_by`` (prompt 1 section 6).
"""
from __future__ import annotations

from alpha_agent.features.registry import REGISTRY as FEATURE_REGISTRY
from alpha_agent.features.registry import FeatureRegistry
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.schemas.hypothesis import HypothesisSpec
from alpha_agent.translation.mechanism_library import CATEGORY_MECHANISM_TEMPLATES, FactorTemplate
from alpha_agent.translation.research_memory import mechanism_research_memory
from alpha_agent.translation.researchability import classify_factor
from alpha_agent.translation.schemas import (
    CERTIFIED_ROOTS,
    FactorCandidate,
    FactorProvenance,
    MeasurableVariable,
    MechanismCandidate,
    MechanismProvenance,
    Observation,
    ObservationTranslation,
    ResearchabilityStatus,
    origin_vintage_fields,
)

__all__ = [
    "build_observation_translation",
    "observation_from_event",
    "observation_from_news",
    "render_summary_text",
]

_GENERATED_BY_DETERMINISTIC = "DETERMINISTIC_LIBRARY"
_GENERATED_BY_CLAUDE = "CLAUDE"

_REASONING_PROVENANCE_NOTE = (
    "Mechanism/factor narrative text is proposed reasoning (deterministic template library, or Claude -- "
    "see generated_by); researchability, feature availability, and research-memory counts are always "
    "computed deterministically from the live FeatureRegistry and ExperimentRegistry and are never taken "
    "from that narrative."
)


# ---------------------------------------------------------------------------
# Observation construction -- Observation Plane data in, typed Observation out
# ---------------------------------------------------------------------------


def _category_of(value: str) -> NewsCategory:
    try:
        return NewsCategory(value)
    except ValueError:
        return NewsCategory.OTHER


def observation_from_news(item: MarketNewsItem, *, root_symbol: str) -> Observation:
    """One certified-root Observation from a real, already-cached
    ``MarketNewsItem``. ``root_symbol`` must be one of the item's own
    deterministically mapped ``related_products`` -- never a root the
    category mapping did not itself produce."""
    if root_symbol not in CERTIFIED_ROOTS:
        raise ValueError(f"root_symbol {root_symbol!r} is outside the certified Phase 1 universe {CERTIFIED_ROOTS}")
    if root_symbol not in item.related_products:
        raise ValueError(
            f"{root_symbol!r} is not among {item.news_id!r}'s mapped related_products {item.related_products}"
        )
    return Observation(
        event_type="MARKET_NEWS",
        root_symbol=root_symbol,
        affected_products=item.related_products,
        observed_at=item.published_at,
        source=item.source_name,
        summary=item.summary or item.headline,
        structured_attributes={
            "news_id": item.news_id,
            "category": item.category.value,
            "mapping_reason": item.mapping_reason,
            "headline": item.headline,
        },
        evidence_refs=(
            f"news_id={item.news_id}",
            f"source_url={item.source_url}",
            f"category={item.category.value}",
        ),
        **origin_vintage_fields(item.published_at),
    )


def observation_from_event(event: ScheduledMarketEvent, *, root_symbol: str) -> Observation:
    """One certified-root Observation from a real, already-cached
    ``ScheduledMarketEvent`` (a scheduled release, not yet realized -- the
    mechanism/factor reasoning below is identical; only the causal framing
    differs slightly in rendering)."""
    if root_symbol not in CERTIFIED_ROOTS:
        raise ValueError(f"root_symbol {root_symbol!r} is outside the certified Phase 1 universe {CERTIFIED_ROOTS}")
    if root_symbol not in event.affected_products:
        raise ValueError(
            f"{root_symbol!r} is not among {event.event_id!r}'s mapped affected_products {event.affected_products}"
        )
    return Observation(
        event_type="SCHEDULED_EVENT",
        root_symbol=root_symbol,
        affected_products=event.affected_products,
        observed_at=event.retrieved_at,
        source=event.source_name,
        summary=event.name,
        structured_attributes={
            "event_id": event.event_id,
            "category": event.category,
            "mapping_reason": event.mapping_reason,
            "importance": event.importance.value,
            "scheduled_at": event.scheduled_at.isoformat(),
        },
        evidence_refs=(
            f"event_id={event.event_id}",
            f"source_url={event.source_url}",
            f"category={event.category}",
        ),
        **origin_vintage_fields(event.retrieved_at),
    )


# ---------------------------------------------------------------------------
# mechanism / factor assembly -- deterministic library, or a validated Claude
# proposal (alpha_agent.agents.mechanism_agent.MechanismTranslationProposal)
# ---------------------------------------------------------------------------


def _from_templates(
    templates: tuple,
) -> tuple[tuple[MechanismCandidate, ...], tuple[MeasurableVariable, ...], tuple[FactorTemplate, ...]]:
    mechanisms: list[MechanismCandidate] = []
    variables: dict[str, MeasurableVariable] = {}
    raw_factors: list[FactorTemplate] = []
    for tmpl in templates:
        mechanisms.append(
            MechanismCandidate(
                mechanism=tmpl.mechanism,
                explanation=tmpl.explanation,
                causal_chain=tmpl.causal_chain,
                evidence_basis=tmpl.evidence_basis,
                provenance=MechanismProvenance.DETERMINISTIC_LIBRARY,
            )
        )
        for v in tmpl.measurable_variables:
            variables.setdefault(v.name, v)
        raw_factors.extend(tmpl.factors)
    return tuple(mechanisms), tuple(variables.values()), tuple(raw_factors)


#: Phase 1 acceptance patch, section 1: free text can never establish
#: capability, no matter how it reads. A Claude-proposed measurable
#: variable's `required_source` is narrative only -- there is no typed
#: repository capability (an exact registered feature/data-source identity)
#: that a free-text string can be checked against, so every Claude-proposed
#: variable's availability is UNKNOWN (`None`), full stop. This is distinct
#: from `FactorCandidate.available_feature_kinds`, which IS a real
#: capability check -- it compares a `proposed_feature_kinds` list against
#: `alpha_agent.features.REGISTRY.kinds()` by exact kind identity, never by
#: scanning prose.
_UNKNOWN_AVAILABILITY_NOTE = (
    "Not independently verified from repository capability metadata -- this variable was proposed as free "
    "text, and prose is never scanned for availability markers (only an exact registered feature/data-source "
    "identity, such as a FactorCandidate's proposed_feature_kinds, can establish capability)."
)


def _from_proposal(
    proposal,
) -> tuple[tuple[MechanismCandidate, ...], tuple[MeasurableVariable, ...], tuple[FactorTemplate, ...]]:
    mechanisms = tuple(
        MechanismCandidate(
            mechanism=item.mechanism,
            explanation=item.explanation,
            causal_chain=tuple(item.causal_chain),
            evidence_basis=item.evidence_basis or "Claude-proposed reasoning; not independently verified.",
            provenance=MechanismProvenance.CLAUDE_PROPOSED,
        )
        for item in proposal.mechanism_candidates
    )
    variables = [
        MeasurableVariable(
            name=v.name,
            economic_meaning=v.economic_meaning,
            required_source=v.required_source,
            point_in_time_available=None,
            point_in_time_note=_UNKNOWN_AVAILABILITY_NOTE,
        )
        for v in proposal.measurable_variables
    ]
    raw_factors = tuple(
        FactorTemplate(
            concept=f.concept,
            mechanism=f.mechanism,
            transform_or_proxy=f.transform_or_proxy,
            proposed_feature_kinds=tuple(f.proposed_feature_kinds),
            required_external_data=tuple(f.required_external_data),
        )
        for f in proposal.factor_candidates
    )
    return mechanisms, tuple(variables), raw_factors


def _classify_factor(raw: FactorTemplate, feature_registry: FeatureRegistry, *, claude_sourced: bool) -> FactorCandidate:
    status, reason, available, missing = classify_factor(
        proposed_feature_kinds=raw.proposed_feature_kinds,
        required_external_data=raw.required_external_data,
        structurally_expressible=raw.structurally_expressible,
        registry=feature_registry,
    )
    if claude_sourced:
        provenance = FactorProvenance.CLAUDE_PROPOSED_PROXY
    elif status in (ResearchabilityStatus.DATA_MISSING, ResearchabilityStatus.NOT_EXECUTABLE):
        provenance = FactorProvenance.UNSUPPORTED_IDEA
    elif len(available) > 1:
        provenance = FactorProvenance.DERIVED_FROM_REGISTERED_FEATURES
    else:
        provenance = FactorProvenance.REGISTERED_FEATURE
    return FactorCandidate(
        concept=raw.concept,
        mechanism=raw.mechanism,
        transform_or_proxy=raw.transform_or_proxy,
        proposed_feature_kinds=raw.proposed_feature_kinds,
        available_feature_kinds=available,
        required_external_data=raw.required_external_data,
        researchability=status,
        researchability_reason=reason,
        missing_requirements=missing,
        provenance=provenance,
    )


def _best_factor(factors: tuple[FactorCandidate, ...]) -> FactorCandidate | None:
    """Phase 1 acceptance patch, section 2: ONLY `AVAILABLE` may
    automatically produce a hypothesis. `PARTIALLY_AVAILABLE` remains a real,
    visible research idea (see `build_observation_translation`'s
    PARTIAL SUPPORT note below) but must never silently become a narrower
    proxy factor presented under the original concept's title -- "never
    silently substitute a weaker proxy and call it the same factor." Ties
    among several AVAILABLE candidates keep list order (deterministic:
    template order, or the proposer's own declared order)."""
    for f in factors:
        if f.researchability is ResearchabilityStatus.AVAILABLE:
            return f
    return None


def _build_hypothesis(observation: Observation, mechanism: MechanismCandidate, factor: FactorCandidate) -> HypothesisSpec:
    assert factor.researchability is ResearchabilityStatus.AVAILABLE  # _best_factor's own contract
    features = list(factor.available_feature_kinds or factor.proposed_feature_kinds)
    category = observation.structured_attributes.get("category", observation.event_type)
    return HypothesisSpec(
        hypothesis_id=f"H-OBS-{observation.root_symbol}-{mechanism.mechanism.value}",
        title=f"{factor.concept} in {observation.root_symbol}",
        economic_mechanism=mechanism.explanation,
        universe=[observation.root_symbol],
        horizon="short-horizon (bars in the window following the observation)",
        required_features=features,
        signal_description=factor.transform_or_proxy,
        expected_regime=f"following a {category}-category observation for {observation.root_symbol}",
        failure_regime="ordinary sessions with no comparable observation",
        falsification_test=(
            f"No statistically significant difference in {factor.concept} between the observation window and "
            "a matched non-event baseline, after multiple-testing correction and cost sensitivity."
        ),
        source_inspirations=[],
    )


def build_observation_translation(
    observation: Observation,
    *,
    registry: ExperimentRegistry,
    feature_registry: FeatureRegistry | None = None,
    mechanism_proposal=None,
) -> ObservationTranslation:
    """The full translation for one Observation. ``mechanism_proposal``, when
    given, is an already schema-validated
    ``alpha_agent.agents.mechanism_agent.MechanismTranslationProposal`` --
    this function never calls an LLM itself."""
    feature_registry = feature_registry or FEATURE_REGISTRY
    claude_sourced = mechanism_proposal is not None

    if claude_sourced:
        mechanism_candidates, measurable_variables, raw_factors = _from_proposal(mechanism_proposal)
        generated_by = _GENERATED_BY_CLAUDE
    else:
        category = _category_of(observation.structured_attributes.get("category", ""))
        templates = CATEGORY_MECHANISM_TEMPLATES.get(category, ())
        mechanism_candidates, measurable_variables, raw_factors = _from_templates(templates)
        generated_by = _GENERATED_BY_DETERMINISTIC

    factor_candidates = tuple(
        _classify_factor(raw, feature_registry, claude_sourced=claude_sourced) for raw in raw_factors
    )

    mechanisms_present: tuple[EconomicMechanism, ...] = tuple(dict.fromkeys(m.mechanism for m in mechanism_candidates))
    memory = mechanism_research_memory(mechanisms=mechanisms_present, root_symbol=observation.root_symbol, registry=registry)

    best = _best_factor(factor_candidates)
    hypothesis: HypothesisSpec | None = None
    research_gap_note: str | None = None
    if best is not None:
        mechanism = next((m for m in mechanism_candidates if m.mechanism == best.mechanism), mechanism_candidates[0])
        hypothesis = _build_hypothesis(observation, mechanism, best)
    elif not factor_candidates:
        research_gap_note = (
            "RESEARCH GAP: no factor candidates were proposed for this observation's category -- no "
            "falsifiable hypothesis can be constructed yet."
        )
    elif any(f.researchability is ResearchabilityStatus.PARTIALLY_AVAILABLE for f in factor_candidates):
        # Phase 1 acceptance patch, section 2: a PARTIALLY_AVAILABLE factor
        # is real, visible research signal -- never silently promoted into a
        # narrower proxy hypothesis presented under the original concept's
        # title. State the gap plainly instead.
        reasons = "; ".join(
            f"{f.concept}: {f.researchability.value} ({f.researchability_reason})" for f in factor_candidates
        )
        research_gap_note = (
            "PARTIAL SUPPORT: some components are available, but the complete factor is not currently "
            f"testable faithfully -- no automatic hypothesis is constructed from a partial match. {reasons}"
        )
    else:
        reasons = "; ".join(
            f"{f.concept}: {f.researchability.value} ({f.researchability_reason})" for f in factor_candidates
        )
        research_gap_note = (
            "RESEARCH GAP: no factor candidate is currently researchable enough to support a falsifiable "
            f"hypothesis. {reasons}"
        )

    return ObservationTranslation(
        observation=observation,
        mechanism_candidates=mechanism_candidates,
        measurable_variables=measurable_variables,
        factor_candidates=factor_candidates,
        research_memory=memory,
        hypothesis=hypothesis,
        research_gap_note=research_gap_note,
        generated_by=generated_by,
        reasoning_provenance_note=_REASONING_PROVENANCE_NOTE,
    )


def render_summary_text(result: ObservationTranslation) -> str:
    """A compact, chat-sized textual digest -- used by the deterministic
    conversation engine's ``OBSERVATION_TO_FACTOR`` handler. The full,
    progressively-disclosed card lives in the Agent page's Research
    Translation panel; this is deliberately shorter (prompt 1 section 18:
    "do not overwhelm... use progressive disclosure")."""
    obs = result.observation
    lines = [
        f"OBSERVATION: {obs.summary} ({obs.source}, {obs.root_symbol}).",
    ]
    if result.mechanism_candidates:
        mech = result.mechanism_candidates[0]
        lines.append(f"POSSIBLE MECHANISM: {mech.mechanism.value} -- {mech.explanation}")
    if result.hypothesis is not None:
        h = result.hypothesis
        lines.append(f"RESEARCHABLE FACTOR: {h.title} (required features: {', '.join(h.required_features) or 'none'}).")
        lines.append(f"FALSIFICATION: {h.falsification_test}")
    elif result.research_gap_note:
        lines.append(result.research_gap_note)
    lines.append(f"RESEARCH MEMORY: {result.research_memory.summary}")
    lines.append(
        "Open the Research Translation panel below for the full mechanism/factor breakdown, or click "
        "Research This to hand this off to the research composer."
    )
    return "\n".join(lines)
