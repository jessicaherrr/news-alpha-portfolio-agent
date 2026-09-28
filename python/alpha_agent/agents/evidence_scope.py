"""Deterministic evidence-scope selection for the Agent conversation (Agent
Experience Consolidation campaign).

The Agent answers "what deserves attention across markets?" -- so its Claude
context must never be silently narrowed to whatever ONE market happens to be
selected on Market's own local product selector (a LOCAL, Market-page-only
concept -- see `alpha_agent.ui.views.market`'s `market_selected_root`).
Before this module existed, `alpha_agent.ui.claude_conversation` filtered the
cached Opportunity snapshot down to a single `root` taken from that local
selection, so a cross-market question ("why CL instead of NQ?") asked while
Market happened to be pointed at CL would silently drop NQ's evidence from
Claude's context entirely -- not because NQ lacked evidence, but because an
unrelated page's LOCAL selection leaked into a GLOBAL conversation.

This module is the one small, typed, deterministic decision of WHICH cached
evidence a given question should see -- never an LLM router, never a network
fetch, never a scientific judgment. Four closed scope kinds:

- GLOBAL: "what's interesting right now?" -- every currently cached
  Opportunity observation.
- ROOT_SET: one or more markets named explicitly in the question (a single-
  market question, or an explicit cross-market comparison/contrast, use the
  SAME rule: whatever roots are named).
- CATEGORY: an asset-class question ("what's interesting in energy?"),
  resolved only against roots the caller already has cached evidence for --
  this module never fetches the rest of a category.
- RESEARCH_OBJECT: a question fundamentally about the currently bound
  research artifact/experiment itself ("why did it fail?", "what did we test
  before?"), scoped to that object's own root rather than a market-wide dump.

The resolver takes only what is already in hand -- the question text, the
intent router's own structured signals (`alpha_agent.agents.conversation`),
the roots present in the caller's already-cached Opportunity snapshot, and a
root->category map the caller already has from the static market-universe
catalog. It never imports `alpha_agent.ui` (UI-free, exactly like
`alpha_agent.agents.conversation`) and never performs I/O -- resolving a scope
can never itself trigger a Databento fetch, a registry read, or a network
call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel

from alpha_agent.agents.conversation import ResearchConversationIntent

__all__ = [
    "EvidenceScope",
    "EvidenceScopeKind",
    "EvidenceScopeSignals",
    "resolve_evidence_scope",
]


class EvidenceScopeKind(str, Enum):
    """Closed scope catalog -- see module docstring."""

    GLOBAL = "GLOBAL"
    ROOT_SET = "ROOT_SET"
    CATEGORY = "CATEGORY"
    RESEARCH_OBJECT = "RESEARCH_OBJECT"


class EvidenceScope(BaseModel):
    """The resolved scope for one conversational turn. `roots` is always the
    FINAL set of roots to show evidence for -- already resolved against the
    caller's cached universe for CATEGORY, verbatim for ROOT_SET, the bound
    object's own root (if any) for RESEARCH_OBJECT, and every cached root for
    GLOBAL -- so a caller never re-derives it."""

    model_config = {"frozen": True, "extra": "forbid"}

    kind: EvidenceScopeKind
    roots: tuple[str, ...] = ()
    category: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class EvidenceScopeSignals:
    """Everything the resolver is allowed to know -- deliberately small, like
    `alpha_agent.agents.conversation.ConversationSignalContext`. Never a full
    transcript, never session internals."""

    text: str
    mentioned_roots: tuple[str, ...] = ()
    intent: ResearchConversationIntent | None = None
    bound_root: str | None = None
    cached_roots: tuple[str, ...] = ()
    root_category: dict[str, str] = field(default_factory=dict)


#: Intents that are fundamentally ABOUT the currently bound research object
#: (a hypothesis/strategy/experiment already on screen) rather than about a
#: market -- "why did it fail?", "what did we test before?", "have we studied
#: something similar?" (task spec section 9). Only consulted when the
#: question names no explicit market symbol -- `resolve_evidence_scope`
#: always lets an explicit root mention win first.
_RESEARCH_OBJECT_INTENTS: frozenset[ResearchConversationIntent] = frozenset(
    {
        ResearchConversationIntent.VALIDATION_EXPLANATION,
        ResearchConversationIntent.STRATEGY_EXPLANATION,
        ResearchConversationIntent.CANDIDATE_COMPARISON,
        ResearchConversationIntent.SOURCE_QUESTION,
        ResearchConversationIntent.RESEARCH_DISCOVERY,
        ResearchConversationIntent.RESEARCH_FOLLOWUP,
        ResearchConversationIntent.WHAT_IF_STRATEGY,
    }
)

#: Category keyword -> category label. Labels match
#: `alpha_agent.marketdata.product_catalog.AssetClass` values so a caller's
#: `root_category` map (built from that same catalog) lines up directly --
#: this module itself never imports the catalog, keeping it UI/registry-free
#: and independently unit-testable with a plain dict.
_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("EQUITY_INDEX", ("equity index", "equity-index", "equity indices", "equities", "stock index", "stock indices")),
    ("ENERGY", ("energy", "crude", "oil")),
    ("METALS", ("metals", "precious metals", "gold and silver")),
    ("RATES", ("rates", "treasuries", "treasury", "bonds", "fixed income")),
    ("FX", ("fx", "currencies", "currency", "forex")),
    ("AGRICULTURE", ("agriculture", "agricultural", "grains", "softs")),
    ("CRYPTO", ("crypto", "cryptocurrency", "digital assets")),
)


def _match_category(text: str) -> str | None:
    lowered = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(k in lowered for k in keywords):
            return category
    return None


def resolve_evidence_scope(signals: EvidenceScopeSignals) -> EvidenceScope:
    """Deterministic, ordered resolution -- never an LLM call, never I/O.

    Order (most explicit wins first):

    1. An explicit market symbol named in the question -> ROOT_SET, exactly
       the roots named. One rule covers both a single-market question
       ("why is CL interesting?") and an explicit cross-market comparison/
       contrast ("why CL instead of NQ?", "compare ES and NQ") -- whatever
       roots the question names, all of them reach the evidence.
    2. Else, a question fundamentally about the bound research object ->
       RESEARCH_OBJECT, scoped to that object's own root (never a market-wide
       dump, and never overridden by an unrelated page's local selection).
    3. Else, a recognised asset-class/category word -> CATEGORY, resolved
       only against roots the caller already has cached evidence for -- never
       a prompt to fetch the rest of the category.
    4. Else -> GLOBAL: every currently cached root. This is the safe default
       for "what's interesting right now?"-shaped questions, and for any
       question a Market/Research page's own LOCAL selection must never be
       allowed to narrow.
    """
    if signals.mentioned_roots:
        return EvidenceScope(
            kind=EvidenceScopeKind.ROOT_SET, roots=signals.mentioned_roots,
            reason="explicit market symbol(s) named in the question",
        )
    if signals.intent in _RESEARCH_OBJECT_INTENTS:
        roots = (signals.bound_root,) if signals.bound_root else ()
        return EvidenceScope(
            kind=EvidenceScopeKind.RESEARCH_OBJECT, roots=roots,
            reason="question is about the currently bound research object, not a market scan",
        )
    category = _match_category(signals.text)
    if category is not None:
        roots = tuple(r for r in signals.cached_roots if signals.root_category.get(r) == category)
        return EvidenceScope(
            kind=EvidenceScopeKind.CATEGORY, roots=roots, category=category,
            reason=f"category question ({category})",
        )
    return EvidenceScope(
        kind=EvidenceScopeKind.GLOBAL, roots=signals.cached_roots,
        reason="no explicit market or category named -- global attention view",
    )
