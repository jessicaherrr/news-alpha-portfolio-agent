"""News Alpha Phase A -- the INITIAL IMPACT SCAN.

``scan_initial_impact(event, mandate)`` is a pure, deterministic function:
event text/category + the static channel table (`news_alpha.channels`) + the
mandate-scoped universe (`news_alpha.universe`) -> one
`InitialImpactAssessment` per `MandateDomain`. No LLM, no network, no
registry access, no filesystem access beyond an optional one-time capability
probe when the caller passes no universe.

THE IMPACT-LEVEL RULE (``impact-level/1``), per (detected channel, exposure
with at least one candidate market in the scoped universe):

* base from exposure strength: PRIMARY -> HIGH, SECONDARY -> MEDIUM,
  TERTIARY -> LOW;
* -1 level when detection is WEAK (a single cue: topic plausible, not
  established);
* +1 level when detection is at least MODERATE and the event is SALIENT
  (a surprise/magnitude term, or a scheduled event market_intel's
  importance rule marks HIGH) -- surprise is what moves prices; a routine or
  explanatory item on the same topic is not;
* clamped to LOW..VERY_HIGH; the domain level is the maximum over its
  contributions. No contribution -> NONE.

THE MANDATE SCOPES, IT NEVER SCORES. Domain access and instrument
constraints decide WHICH candidates exist (an excluded domain is simply not
assessed). Risk style, drawdown, leverage, shorting and horizon never change
an impact level -- they only add `mandate_notes` (e.g. "downward pressure
cannot be expressed as a net short under this mandate").
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

from alpha_agent.market_intel.event_schemas import EventImportance, ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem
from alpha_agent.news_alpha.channels import (
    CHANNELS,
    SALIENCE_TERMS,
    ChannelDefinition,
    ExposureStrength,
    ImpactHorizon,
    MarketExposure,
    PressureSign,
    channel_definition,
    normalize_text,
)
from alpha_agent.news_alpha.events import ImpactEvent, UserDescribedEvent, impact_event_from
from alpha_agent.news_alpha.mandate import DOMAIN_LABELS, ResearchMandate
from alpha_agent.news_alpha.schemas import (
    AvailabilityStatus,
    CandidateMarket,
    ChannelEvidence,
    DetectionBasis,
    DetectionStrength,
    DirectionalPressure,
    ExposureContribution,
    HorizonAlignment,
    ImpactLevel,
    InitialImpactAssessment,
    InitialImpactScan,
    TriageConfidence,
)
from alpha_agent.news_alpha.universe import (
    AllowedAssetUniverse,
    DomainCapabilities,
    DomainScope,
    DomainSupport,
    UniverseInstrument,
    resolve_allowed_universe,
)
from alpha_agent.recommendation.profile import HoldingPeriod

__all__ = ["IMPACT_LEVEL_RULE", "SALIENCE_RULE", "impact_level_for", "scan_initial_impact", "scan_many"]

IMPACT_LEVEL_RULE = "impact-level/1"
SALIENCE_RULE = "salience-lexicon/1"

_BASE_RANK = {
    ExposureStrength.PRIMARY: ImpactLevel.HIGH.rank,
    ExposureStrength.SECONDARY: ImpactLevel.MEDIUM.rank,
    ExposureStrength.TERTIARY: ImpactLevel.LOW.rank,
}
_STRENGTH_ORDER = {ExposureStrength.PRIMARY: 0, ExposureStrength.SECONDARY: 1, ExposureStrength.TERTIARY: 2}
_DETECTION_ORDER = {DetectionStrength.STRONG: 0, DetectionStrength.MODERATE: 1, DetectionStrength.WEAK: 2}
_CONFIDENCE = {
    DetectionStrength.STRONG: TriageConfidence.HIGH,
    DetectionStrength.MODERATE: TriageConfidence.MEDIUM,
    DetectionStrength.WEAK: TriageConfidence.LOW,
}
_HORIZON_COMPATIBLE: dict[ImpactHorizon, frozenset[HoldingPeriod]] = {
    ImpactHorizon.INTRADAY_TO_DAYS: frozenset(
        {HoldingPeriod.INTRADAY, HoldingPeriod.ONE_TO_THREE_DAYS, HoldingPeriod.SEVERAL_DAYS}
    ),
    ImpactHorizon.DAYS_TO_WEEKS: frozenset(
        {HoldingPeriod.ONE_TO_THREE_DAYS, HoldingPeriod.SEVERAL_DAYS, HoldingPeriod.WEEKS}
    ),
    ImpactHorizon.WEEKS_TO_MONTHS: frozenset({HoldingPeriod.WEEKS}),
}
_HORIZON_LABEL = {
    ImpactHorizon.INTRADAY_TO_DAYS: "intraday to days",
    ImpactHorizon.DAYS_TO_WEEKS: "days to weeks",
    ImpactHorizon.WEEKS_TO_MONTHS: "weeks to months",
}
_INITIAL_SCAN_NOTE = "Initial scan: channel/keyword triage before any mechanism reasoning -- not final."


# ---------------------------------------------------------------------------
# text matching -- word-bounded, overlap-aware (so "crude oil" counts once,
# not as both "crude" and "crude oil")
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z0-9]){re.escape(normalize_text(term))}(?![a-z0-9])")


def _match_terms(text_norm: str, terms: Iterable[str]) -> tuple[str, ...]:
    spans = [(m.start(), m.end(), t) for t in terms for m in _pattern(t).finditer(text_norm)]
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    accepted: list[tuple[int, int, str]] = []
    for s in spans:
        if all(s[1] <= a[0] or s[0] >= a[1] for a in accepted):
            accepted.append(s)
    hits: list[str] = []
    for _start, _end, term in sorted(accepted):
        if term not in hits:
            hits.append(term)
    return tuple(hits)


# ---------------------------------------------------------------------------
# channel detection
# ---------------------------------------------------------------------------


def _salience(event: ImpactEvent, text_norm: str) -> tuple[str, ...]:
    terms = list(_match_terms(text_norm, SALIENCE_TERMS))
    if isinstance(event.source, ScheduledMarketEvent) and event.source.importance is EventImportance.HIGH:
        terms.append(f"scheduled HIGH-importance event ({event.source.importance_rule})")
    return tuple(terms)


def _detect(defn: ChannelDefinition, event: ImpactEvent, text_norm: str, salience: tuple[str, ...]) -> ChannelEvidence | None:
    bases: list[DetectionBasis] = []
    if event.source_category is not None and event.source_category in defn.source_categories:
        bases.append(DetectionBasis.SOURCE_CATEGORY)
    hits = _match_terms(text_norm, defn.lexicon)
    if hits:
        bases.append(DetectionBasis.LEXICON)
    source_norm = normalize_text(event.source_name)
    issuer = any(_pattern(cue).search(source_norm) for cue in defn.issuer_cues)
    if issuer:
        bases.append(DetectionBasis.ISSUER)
    if not bases:
        return None

    if DetectionBasis.SOURCE_CATEGORY in bases:
        strength = DetectionStrength.STRONG
    elif len(hits) + int(issuer) >= 2:
        strength = DetectionStrength.MODERATE
    else:
        strength = DetectionStrength.WEAK

    shifts = [cue.shift for cue in defn.direction_cues if _match_terms(text_norm, cue.terms)]
    return ChannelEvidence(
        channel=defn.channel, channel_label=defn.label, rule_id=defn.rule_id, bases=tuple(bases), strength=strength,
        matched_terms=hits, shift=shifts[0] if len(shifts) == 1 else None,
        conflicting_shift_cues=len(shifts) > 1, salience_terms=salience,
    )


def _detect_channels(event: ImpactEvent) -> tuple[ChannelEvidence, ...]:
    text_norm = normalize_text(event.text)
    salience = _salience(event, text_norm)
    found = [ev for defn in CHANNELS if (ev := _detect(defn, event, text_norm, salience)) is not None]
    order = {c.channel: i for i, c in enumerate(CHANNELS)}
    return tuple(sorted(found, key=lambda ev: (_DETECTION_ORDER[ev.strength], order[ev.channel])))


# ---------------------------------------------------------------------------
# per-domain assessment
# ---------------------------------------------------------------------------


def _matches(exposure: MarketExposure, inst: UniverseInstrument) -> bool:
    return inst.symbol in exposure.symbols or inst.group in exposure.groups


def impact_level_for(strength: ExposureStrength, detection: DetectionStrength, salient: bool) -> ImpactLevel:
    """``impact-level/1`` for one exposure -- the single implementation both
    the initial scan and the mechanism-adjusted pass apply."""
    rank = _BASE_RANK[strength]
    if detection is DetectionStrength.WEAK:
        rank -= 1
    elif salient:
        rank += 1
    return list(ImpactLevel)[max(ImpactLevel.LOW.rank, min(ImpactLevel.VERY_HIGH.rank, rank))]


def _level_rank(exposure: MarketExposure, ev: ChannelEvidence) -> int:
    return impact_level_for(exposure.strength, ev.strength, bool(ev.salience_terms)).rank


def _not_assessed(event: ImpactEvent, scope: DomainScope, availability: AvailabilityStatus, note: str) -> InitialImpactAssessment:
    return InitialImpactAssessment(
        event_id=event.event_id, asset_domain=scope.domain, impact_level=None, availability=availability,
        availability_note=note, reasoning_provenance=scope.sources,
    )


def _availability(scope: DomainScope, markets: list[CandidateMarket]) -> tuple[AvailabilityStatus, str]:
    label = DOMAIN_LABELS[scope.domain]
    if scope.support is DomainSupport.SYNTHETIC_ONLY:
        return AvailabilityStatus.SYNTHETIC_ONLY, scope.support_note
    if any(m.research_ready for m in markets):
        ready = [m.symbol for m in markets if m.research_ready]
        return AvailabilityStatus.RESEARCH_READY, f"Research-ready {label} candidates: {', '.join(ready)}."
    if scope.support is DomainSupport.RESEARCH_READY:
        return (
            AvailabilityStatus.OBSERVATION_ONLY,
            f"Every {label} candidate is observation-only (catalogued, outside the certified research universe).",
        )
    if scope.support is DomainSupport.FOUNDATION_ONLY:
        return AvailabilityStatus.FOUNDATION_ONLY, scope.support_note
    return AvailabilityStatus.DATA_NOT_ACQUIRED, scope.support_note


def _horizon_alignment(horizon: ImpactHorizon, mandate: ResearchMandate) -> HorizonAlignment:
    if mandate.investment_horizon is HoldingPeriod.FLEXIBLE:
        return HorizonAlignment.UNCONSTRAINED
    return (
        HorizonAlignment.ALIGNED if mandate.investment_horizon in _HORIZON_COMPATIBLE[horizon]
        else HorizonAlignment.MISALIGNED
    )


def _assess_domain(
    scope: DomainScope, channels: tuple[ChannelEvidence, ...], event: ImpactEvent, mandate: ResearchMandate,
) -> InitialImpactAssessment:
    label = DOMAIN_LABELS[scope.domain]
    if not scope.in_mandate:
        return _not_assessed(
            event, scope, AvailabilityStatus.EXCLUDED_BY_MANDATE,
            f"{label} is excluded by your research mandate -- not investigated.",
        )
    if scope.support is DomainSupport.NOT_SUPPORTED:
        return _not_assessed(event, scope, AvailabilityStatus.DOMAIN_NOT_SUPPORTED, scope.support_note)

    # (rank, channel evidence, exposure, candidate instruments)
    contributions: list[tuple[int, ChannelEvidence, MarketExposure, tuple[UniverseInstrument, ...]]] = []
    filtered_out: list[str] = []
    for ev in channels:
        for exposure in channel_definition(ev.channel).exposures:
            if exposure.domain is not scope.domain:
                continue
            candidates = tuple(i for i in scope.instruments if _matches(exposure, i))
            if scope.support is not DomainSupport.SYNTHETIC_ONLY and not candidates:
                removed = [i.symbol for i in scope.removed_by_constraints if _matches(exposure, i)]
                if removed:
                    filtered_out.append(f"{exposure.label} ({', '.join(removed)})")
                continue
            contributions.append((_level_rank(exposure, ev), ev, exposure, candidates))

    provenance_base = (*scope.sources,)
    if not contributions:
        if filtered_out:
            note = (
                "The detected channels expose this domain, but your instrument constraints exclude every "
                f"candidate: {'; '.join(filtered_out)}."
            )
        elif channels:
            note = f"The detected channels ({', '.join(ev.channel_label for ev in channels)}) have no mapped {label} exposure."
        else:
            note = "No economic channel was detected in this event's category or text."
        return InitialImpactAssessment(
            event_id=event.event_id, asset_domain=scope.domain, impact_level=ImpactLevel.NONE,
            availability=AvailabilityStatus.NO_CANDIDATE_MARKETS, availability_note=note,
            uncertainty=(_INITIAL_SCAN_NOTE,), reasoning_provenance=(IMPACT_LEVEL_RULE, *provenance_base),
        )

    channel_order = {ev.channel: i for i, ev in enumerate(channels)}
    contributions.sort(key=lambda c: (-c[0], _STRENGTH_ORDER[c[2].strength], channel_order[c[1].channel]))
    top_rank, top_ev, _top_exposure, _ = contributions[0]
    level = list(ImpactLevel)[top_rank]
    top_detection = min(
        (c[1].strength for c in contributions if c[0] == top_rank), key=lambda s: _DETECTION_ORDER[s],
    )

    markets: dict[str, CandidateMarket] = {}
    groups: list[str] = []
    channels_used: list = []
    directions: list[DirectionalPressure] = []
    for _rank, ev, exposure, candidates in contributions:
        if ev.channel not in channels_used:
            channels_used.append(ev.channel)
        if exposure.label not in groups:
            groups.append(exposure.label)
        for inst in candidates:
            markets.setdefault(
                inst.symbol,
                CandidateMarket(
                    symbol=inst.symbol, display_name=inst.display_name, group_label=inst.group_label,
                    research_ready=inst.research_ready, channel=ev.channel, exposure_label=exposure.label,
                    exposure_strength=exposure.strength, rationale=exposure.rationale,
                ),
            )
        sign = dict(exposure.pressure).get(ev.shift) if ev.shift else None
        if sign is not None:
            directions.append(
                DirectionalPressure(
                    channel=ev.channel, shift=ev.shift, exposure_label=exposure.label,
                    symbols=tuple(i.symbol for i in candidates), pressure=sign, rationale=exposure.rationale,
                )
            )

    # A shift is "unsigned" for this domain only when NONE of the domain's
    # exposures to that channel carries a textbook sign -- one signed group
    # (e.g. Energy) plus one unsigned tertiary group (Industrials) is a signed
    # direction, not a contradiction.
    signed_channels = {d.channel for d in directions}
    unsigned_shifts = list(dict.fromkeys(
        f"{ev.channel_label}: {ev.shift}"
        for _rank, ev, _exposure, _c in contributions
        if ev.shift and ev.channel not in signed_channels
    ))

    ordered_markets = sorted(
        markets.values(), key=lambda m: (_STRENGTH_ORDER[m.exposure_strength], not m.research_ready),
    )
    availability, availability_note = _availability(scope, ordered_markets)

    top_defn = channel_definition(top_ev.channel)
    horizon = top_defn.typical_horizon
    alignment = _horizon_alignment(horizon, mandate)

    uncertainty = [_INITIAL_SCAN_NOTE]
    if top_detection is DetectionStrength.WEAK:
        uncertainty.append(
            f"Only a single cue links this event to {top_ev.channel_label}; the topic itself is not established."
        )
    for ev in channels:
        if ev.conflicting_shift_cues and ev.channel in channels_used:
            uncertainty.append(f"Opposing direction cues matched for {ev.channel_label}; direction left undetermined.")
    for shift in unsigned_shifts:
        uncertainty.append(
            f"{shift} was detected, but no textbook first-order sign applies to these {label} groups -- the "
            "Mechanism Graph stage resolves it."
        )
    if not directions:
        uncertainty.append("No first-order price direction is justified from the event text at this stage.")

    mandate_notes: list[str] = []
    if not mandate.shorting_allowed:
        for d in directions:
            if d.pressure is PressureSign.DOWN:
                mandate_notes.append(
                    f"Hypothesized downward pressure on {d.exposure_label} cannot be expressed as a net short under "
                    "this mandate; it can still be researched (e.g. as an avoid/underweight or hedge signal)."
                )
    if alignment is HorizonAlignment.MISALIGNED:
        mandate_notes.append(
            f"This channel typically plays out over {_HORIZON_LABEL[horizon]}, unlike your "
            f"{mandate.investment_horizon.value} horizon -- research may proceed, but expression timing needs care."
        )
    if filtered_out:
        mandate_notes.append(f"Your instrument constraints removed some candidates: {'; '.join(filtered_out)}.")

    provenance = list(dict.fromkeys(channel_definition(c).rule_id for c in channels_used))
    provenance.append(IMPACT_LEVEL_RULE)
    if any(c[1].salience_terms for c in contributions):
        provenance.append(SALIENCE_RULE)
    provenance.extend(provenance_base)

    recorded = tuple(
        ExposureContribution(
            channel=ev.channel, exposure_label=exposure.label, exposure_strength=exposure.strength,
            detection_strength=ev.strength, salient=bool(ev.salience_terms), impact_level=list(ImpactLevel)[rank],
            symbols=tuple(i.symbol for i in candidates),
        )
        for rank, ev, exposure, candidates in contributions
    )
    return InitialImpactAssessment(
        event_id=event.event_id, asset_domain=scope.domain, impact_level=level, availability=availability,
        availability_note=availability_note, contributions=recorded, candidate_channels=tuple(channels_used),
        candidate_groups=tuple(groups),
        candidate_markets=tuple(ordered_markets), possible_direction=tuple(directions), expected_horizon=horizon,
        horizon_rationale=top_defn.horizon_rationale, horizon_alignment=alignment,
        confidence=_CONFIDENCE[top_detection], uncertainty=tuple(uncertainty), mandate_notes=tuple(mandate_notes),
        reasoning_provenance=tuple(provenance),
    )


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

EventLike = MarketNewsItem | ScheduledMarketEvent | UserDescribedEvent | ImpactEvent


def scan_initial_impact(
    event: EventLike,
    mandate: ResearchMandate,
    *,
    universe: AllowedAssetUniverse | None = None,
    capabilities: DomainCapabilities | None = None,
) -> InitialImpactScan:
    """The initial impact scan for one event under one mandate. Pass a
    pre-resolved ``universe`` when scanning many events (it must have been
    resolved from this exact mandate)."""
    view = event if isinstance(event, ImpactEvent) else impact_event_from(event)
    universe = universe or resolve_allowed_universe(mandate, capabilities=capabilities)
    if universe.mandate_fingerprint != mandate.fingerprint():
        raise ValueError("universe was resolved from a different mandate than the one being scanned")
    channels = _detect_channels(view)
    return InitialImpactScan(
        event=view, mandate_fingerprint=universe.mandate_fingerprint, mandate_summary=mandate.summary(),
        channels=channels,
        assessments=tuple(_assess_domain(scope, channels, view, mandate) for scope in universe.scopes),
    )


def scan_many(
    events: Iterable[EventLike], mandate: ResearchMandate, *, universe: AllowedAssetUniverse | None = None,
) -> tuple[InitialImpactScan, ...]:
    """Scan several events under one resolved universe, most relevant first
    (highest assessed level, then most recent). Ordering is presentation only."""
    universe = universe or resolve_allowed_universe(mandate)
    scans = [scan_initial_impact(e, mandate, universe=universe) for e in events]
    return tuple(
        sorted(
            scans,
            key=lambda s: (-(s.top_level.rank if s.top_level else -1), -s.event.observed_at.timestamp()),
        )
    )
