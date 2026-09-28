from __future__ import annotations

from pydantic import BaseModel, Field


class HypothesisSpec(BaseModel):
    hypothesis_id: str
    title: str
    economic_mechanism: str
    universe: list[str]
    horizon: str
    required_features: list[str]
    signal_description: str
    expected_regime: str
    failure_regime: str
    falsification_test: str
    novelty_notes: str = ""
    evidence_level: str = Field(default="experimental")
    #: Alpha Discovery live-research campaign, Checkpoint 9 -- a short list of
    #: which `research_knowledge_base` snippet(s) (matched by title/mechanism
    #: text, never a structural id the agent could not see) inspired this
    #: hypothesis, if any. Purely informational lineage for the investor-
    #: facing "Research Inspiration" view (task spec sections 47/48/63) --
    #: NEVER read by novelty detection, deduplication, or any scientific
    #: gate, which all key on `experiment_identity` / `strategy_fingerprint`
    #: exactly as before this field existed. An empty list is the honest
    #: default for a hypothesis with no external inspiration (e.g. every
    #: existing scripted test fixture).
    source_inspirations: list[str] = Field(default_factory=list)
