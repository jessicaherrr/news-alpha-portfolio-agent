"""Phase 1 -- NEWS/MARKET OBSERVATION -> MECHANISM -> FACTOR translation
layer.

Deterministic, offline, dependency-light: no Streamlit import, no network
call, no registry write. See ``alpha_agent.translation.schemas`` for the
typed model and ``alpha_agent.translation.pipeline.build_observation_translation``
for the entry point. Optional Claude-backed mechanism/factor reasoning lives
in ``alpha_agent.agents.mechanism_agent`` (a separate, LLM-touching module --
this package never imports it).
"""
from __future__ import annotations

from alpha_agent.translation.mechanism_library import CATEGORY_MECHANISM_TEMPLATES
from alpha_agent.translation.pipeline import (
    build_observation_translation,
    observation_from_event,
    observation_from_news,
    render_summary_text,
)
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
    ResearchMemoryNote,
    origin_vintage_fields,
)

__all__ = [
    "CATEGORY_MECHANISM_TEMPLATES",
    "CERTIFIED_ROOTS",
    "FactorCandidate",
    "FactorProvenance",
    "MeasurableVariable",
    "MechanismCandidate",
    "MechanismProvenance",
    "Observation",
    "ObservationTranslation",
    "ResearchMemoryNote",
    "ResearchabilityStatus",
    "build_observation_translation",
    "classify_factor",
    "mechanism_research_memory",
    "observation_from_event",
    "observation_from_news",
    "origin_vintage_fields",
    "render_summary_text",
]
