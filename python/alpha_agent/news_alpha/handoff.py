"""News Alpha Phase A -- the hand-off from the Initial Impact Scan into the
EXISTING translation boundary (`alpha_agent.translation`, Phase 1).

The scan decides WHERE research should start; the translation layer decides
WHAT is researchable. This module only builds the `Observation` the existing
`translation.pipeline.build_observation_translation` entry point already
accepts -- it adds no second translation path, calls no LLM, and touches no
registry (the caller runs the translation, which reads the registry
read-only for research memory).

HOW AN OBSERVATION IS BUILT, in order of fidelity (`CategoryBasis`):

1. SOURCE_CATEGORY -- the item is a real market_intel item whose own category
   and connector mapping already cover this root: the existing
   `observation_from_news` / `observation_from_event` constructor is used
   verbatim (impact-scan provenance is appended to ``evidence_refs``).
2. IMPACT_CHANNEL -- no source category covers it (a Fed speech filed as
   OTHER, a user-described OPEC decision): the scan's channel routes it to the
   Phase 1 templates that model the same economic channel. Those templates
   were written for official releases, so the hand-off carries an explicit
   fidelity note -- an approximation, labelled as one.
3. NO_TEMPLATE -- the channel has no deterministic template yet (technology
   investment, trade policy, ...). The translation then reports its own
   honest RESEARCH GAP; the optional Claude mode may propose mechanisms, with
   researchability still decided deterministically.

Only instruments the Phase 1 `Observation` validator accepts (the certified
Futures roots) can be handed off today. Every other domain that deserves
research is reported as a typed `TranslationGap`, never silently dropped and
never forced through a Futures-only schema.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory
from alpha_agent.news_alpha.channels import EconomicChannel, channel_definition
from alpha_agent.news_alpha.events import ImpactEventKind
from alpha_agent.news_alpha.mandate import DOMAIN_LABELS, MandateDomain
from alpha_agent.news_alpha.schemas import (
    CandidateMarket,
    ImpactLevel,
    InitialImpactAssessment,
    InitialImpactScan,
)
from alpha_agent.translation.pipeline import observation_from_event, observation_from_news
from alpha_agent.translation.schemas import CERTIFIED_ROOTS, Observation, origin_vintage_fields

__all__ = [
    "CategoryBasis",
    "TranslationGap",
    "TranslationHandoff",
    "TranslationHandoffPlan",
    "build_translation_handoffs",
]

_KEY_PREFIX = {
    ImpactEventKind.MARKET_NEWS: "news",
    ImpactEventKind.SCHEDULED_EVENT: "event",
    ImpactEventKind.USER_DESCRIBED: "user",
}
_EVENT_TYPE = {
    ImpactEventKind.MARKET_NEWS: "MARKET_NEWS",
    ImpactEventKind.SCHEDULED_EVENT: "SCHEDULED_EVENT",
    ImpactEventKind.USER_DESCRIBED: "USER_DESCRIBED_EVENT",
}
_ID_ATTRIBUTE = {
    ImpactEventKind.MARKET_NEWS: "news_id",
    ImpactEventKind.SCHEDULED_EVENT: "event_id",
    ImpactEventKind.USER_DESCRIBED: "user_event_id",
}


class CategoryBasis(str, Enum):
    SOURCE_CATEGORY = "SOURCE_CATEGORY"
    IMPACT_CHANNEL = "IMPACT_CHANNEL"
    NO_TEMPLATE = "NO_TEMPLATE"


class TranslationHandoff(BaseModel):
    """One (event, certified root) ready for the existing translation entry
    point. ``key`` matches `alpha_agent.ui.translation_context
    .CandidateObservation.key` for the same (news item, root), so a UI can
    keep one translation cache across both entry points."""

    model_config = {"frozen": True, "extra": "forbid"}

    key: str
    event_id: str
    asset_domain: MandateDomain
    root_symbol: str
    impact_level: ImpactLevel
    channel: EconomicChannel
    category_basis: CategoryBasis
    translation_category: str | None
    fidelity_note: str
    observation: Observation


class TranslationGap(BaseModel):
    """A domain that deserves research but has no translation path today."""

    model_config = {"frozen": True, "extra": "forbid"}

    asset_domain: MandateDomain
    impact_level: ImpactLevel
    candidate_symbols: tuple[str, ...]
    reason: str


class TranslationHandoffPlan(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    event_id: str
    handoffs: tuple[TranslationHandoff, ...] = ()
    gaps: tuple[TranslationGap, ...] = ()


def _translatable(domain: MandateDomain, market: CandidateMarket) -> bool:
    """Exactly what the Phase 1 `Observation` validator accepts."""
    return domain is MandateDomain.FUTURES and market.symbol in CERTIFIED_ROOTS


def _impact_refs(scan: InitialImpactScan, market: CandidateMarket, basis: CategoryBasis) -> tuple[str, ...]:
    return (
        f"impact_channel={market.channel.value}",
        f"impact_channel_rule={channel_definition(market.channel).rule_id}",
        f"category_basis={basis.value}",
        f"mandate={scan.mandate_fingerprint}",
    )


def _source_category_observation(scan: InitialImpactScan, root: str) -> Observation | None:
    src = scan.event.source
    if isinstance(src, MarketNewsItem) and src.category is not NewsCategory.OTHER and root in src.related_products:
        return observation_from_news(src, root_symbol=root)
    if isinstance(src, ScheduledMarketEvent) and src.category != NewsCategory.OTHER.value and root in src.affected_products:
        return observation_from_event(src, root_symbol=root)
    return None


def _channel_observation(
    scan: InitialImpactScan, assessment: InitialImpactAssessment, market: CandidateMarket, basis: CategoryBasis,
) -> Observation:
    ev = scan.event
    defn = channel_definition(market.channel)
    category = defn.translation_category.value if defn.translation_category else NewsCategory.OTHER.value
    return Observation(
        event_type=_EVENT_TYPE[ev.kind],
        root_symbol=market.symbol,
        affected_products=tuple(m.symbol for m in assessment.candidate_markets),
        observed_at=ev.observed_at,
        source=ev.source_name,
        summary=ev.summary or ev.headline,
        structured_attributes={
            _ID_ATTRIBUTE[ev.kind]: ev.event_id,
            "category": category,
            "source_category": ev.source_category or "NONE",
            "category_basis": basis.value,
            "impact_channel": market.channel.value,
            "headline": ev.headline,
        },
        evidence_refs=(*ev.evidence_refs, *_impact_refs(scan, market, basis)),
        **origin_vintage_fields(ev.observed_at),
    )


def _fidelity_note(scan: InitialImpactScan, market: CandidateMarket, basis: CategoryBasis) -> str:
    defn = channel_definition(market.channel)
    if basis is CategoryBasis.SOURCE_CATEGORY:
        return (
            f"The item's own market_intel category ({scan.event.source_category}) selects the Phase 1 mechanism "
            "templates -- the existing translation path, unchanged."
        )
    if basis is CategoryBasis.IMPACT_CHANNEL:
        origin = "a user-described event" if scan.event.kind is ImpactEventKind.USER_DESCRIBED else (
            f"source category {scan.event.source_category}"
        )
        return (
            f"No source category covers this ({origin}); the scan's '{defn.label}' channel routes it to the "
            f"{defn.translation_category.value} mechanism templates, which were written for official releases -- "
            "treat their mechanism text as an approximation until the Mechanism Graph stage."
        )
    return (
        f"No deterministic mechanism template models the '{defn.label}' channel yet -- the deterministic "
        "translation will report an honest research gap; the optional Claude mode can propose mechanisms "
        "(researchability is still decided deterministically)."
    )


def build_translation_handoffs(scan: InitialImpactScan) -> TranslationHandoffPlan:
    """Every certified-root hand-off the scan supports (domains with an
    assessed impact above NONE, most relevant first), plus a typed gap for
    every relevant domain the translation layer cannot accept yet."""
    handoffs: list[TranslationHandoff] = []
    gaps: list[TranslationGap] = []
    relevant = sorted(
        (a for a in scan.assessments if a.impact_level is not None and a.impact_level is not ImpactLevel.NONE),
        key=lambda a: -a.impact_level.rank,
    )
    for assessment in relevant:
        translatable = [m for m in assessment.candidate_markets if _translatable(assessment.asset_domain, m)]
        if not translatable:
            label = DOMAIN_LABELS[assessment.asset_domain]
            symbols = tuple(m.symbol for m in assessment.candidate_markets)
            if assessment.asset_domain is MandateDomain.FUTURES:
                reason = (
                    f"Futures candidates ({', '.join(symbols)}) are outside the certified research roots "
                    f"({', '.join(CERTIFIED_ROOTS)}) the translation layer accepts."
                )
            else:
                reason = (
                    f"The Phase 1 translation layer accepts certified Futures roots only; {label} research "
                    "continues at the Mechanism Graph / Asset Expression stages."
                )
            gaps.append(
                TranslationGap(
                    asset_domain=assessment.asset_domain, impact_level=assessment.impact_level,
                    candidate_symbols=symbols, reason=reason,
                )
            )
            continue
        for market in translatable:
            observation = _source_category_observation(scan, market.symbol)
            if observation is not None:
                basis = CategoryBasis.SOURCE_CATEGORY
                observation = observation.model_copy(
                    update={"evidence_refs": (*observation.evidence_refs, *_impact_refs(scan, market, basis))}
                )
            else:
                defn = channel_definition(market.channel)
                basis = CategoryBasis.IMPACT_CHANNEL if defn.translation_category else CategoryBasis.NO_TEMPLATE
                observation = _channel_observation(scan, assessment, market, basis)
            handoffs.append(
                TranslationHandoff(
                    key=f"{_KEY_PREFIX[scan.event.kind]}:{scan.event.event_id}:{market.symbol}",
                    event_id=scan.event.event_id, asset_domain=assessment.asset_domain, root_symbol=market.symbol,
                    impact_level=assessment.impact_level, channel=market.channel, category_basis=basis,
                    translation_category=observation.structured_attributes.get("category"),
                    fidelity_note=_fidelity_note(scan, market, basis), observation=observation,
                )
            )
    return TranslationHandoffPlan(event_id=scan.event.event_id, handoffs=tuple(handoffs), gaps=tuple(gaps))
