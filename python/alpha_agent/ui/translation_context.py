"""The UI's ONLY boundary into `alpha_agent.translation` (mirrors
`alpha_agent.ui.opportunity_context`'s role for Opportunity V1).

Candidate observations are built ONLY from `market_intel_context`'s
already-cached (never-refreshing) news/event reads -- this module never
triggers a live news/event fetch, and never calls Claude unless the caller
explicitly asks for `mode="live"` (prompt 1 section 14: no automatic paid
network calls).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alpha_agent.agents.context import build_feature_catalog
from alpha_agent.agents.llm import AnthropicClient, LLMClientUnavailable
from alpha_agent.agents.mechanism_agent import MechanismAgent
from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem
from alpha_agent.translation.pipeline import (
    build_observation_translation,
    observation_from_event,
    observation_from_news,
)
from alpha_agent.translation.schemas import CERTIFIED_ROOTS, Observation, ObservationTranslation
from alpha_agent.ui import market_intel_context, services

__all__ = [
    "LIVE_MODE",
    "SCRIPTED_MODE",
    "CandidateObservation",
    "build_observation",
    "candidate_observations",
    "translate_observation",
]

#: Mirrors `alpha_agent.ui.llm_demo.LIVE_MODE` / `SCRIPTED_MODE` exactly -- the
#: same two engine labels the rest of the Agent page already renders (the
#: existing Research Engine selector), so this module never introduces a
#: third, differently-spelled mode vocabulary.
SCRIPTED_MODE = "scripted"
LIVE_MODE = "live"

_MAX_CANDIDATES = 6


@dataclass(frozen=True)
class CandidateObservation:
    """One (cached news/event item, certified root) pair a user can translate
    -- never itself scientific evidence, see `alpha_agent.translation.schemas
    .Observation`'s own module docstring."""

    kind: Literal["news", "event"]
    key: str
    label: str
    root_symbol: str
    item: MarketNewsItem | ScheduledMarketEvent


def candidate_observations(*, limit: int = _MAX_CANDIDATES) -> tuple[CandidateObservation, ...]:
    """Every (cached item, certified root) pair currently available to
    translate -- built ONLY from `market_intel_context`'s cached (never
    live-fetching) reads. Empty before any Market/Agent news refresh has run
    in this process, which is the honest answer, never a fabricated one."""
    out: list[CandidateObservation] = []
    seen: set[tuple[str, str]] = set()

    for item in market_intel_context.cached_recent_news(limit=50):
        for root in item.related_products:
            if root not in CERTIFIED_ROOTS:
                continue
            key = (item.news_id, root)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                CandidateObservation(
                    kind="news", key=f"news:{item.news_id}:{root}",
                    label=f"{root} · {item.headline}", root_symbol=root, item=item,
                )
            )

    for event in market_intel_context.cached_upcoming_events():
        for root in event.affected_products:
            if root not in CERTIFIED_ROOTS:
                continue
            key = (event.event_id, root)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                CandidateObservation(
                    kind="event", key=f"event:{event.event_id}:{root}",
                    label=f"{root} · {event.name} (scheduled)", root_symbol=root, item=event,
                )
            )

    out.sort(key=lambda c: getattr(c.item, "published_at", None) or getattr(c.item, "scheduled_at", None), reverse=True)
    return tuple(out[:limit])


def build_observation(candidate: CandidateObservation) -> Observation:
    if candidate.kind == "news":
        return observation_from_news(candidate.item, root_symbol=candidate.root_symbol)
    return observation_from_event(candidate.item, root_symbol=candidate.root_symbol)


def translate_observation(observation: Observation, *, mode: str = SCRIPTED_MODE) -> tuple[ObservationTranslation | None, str | None]:
    """Returns `(translation, error)`. `mode=SCRIPTED_MODE` (default) never
    touches the network -- the deterministic mechanism library proposes.
    `mode=LIVE_MODE` makes one bounded `AnthropicClient()` call (the SAME
    transport `alpha_agent.ui.llm_demo`/`decision_brief` already use) and
    falls back to an honest error string, never a silently downgraded
    deterministic answer presented as if Claude produced it."""
    mechanism_proposal = None
    if mode == LIVE_MODE:
        try:
            feature_catalog = build_feature_catalog()
            client = AnthropicClient()
            result = MechanismAgent(client).propose(observation, feature_catalog=feature_catalog)
        except LLMClientUnavailable as exc:
            return None, f"Claude is unavailable: {exc}"
        except Exception as exc:  # noqa: BLE001 -- defensive UI boundary, never a raw stack trace or key material
            return None, f"{type(exc).__name__}: {exc}"
        if not result.accepted:
            return None, result.rejection_detail or "Claude's proposal was not accepted."
        mechanism_proposal = result.proposal

    with services.open_registry() as reg:
        translation = build_observation_translation(observation, registry=reg, mechanism_proposal=mechanism_proposal)
    return translation, None
