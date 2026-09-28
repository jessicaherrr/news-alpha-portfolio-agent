"""Conversational intent routing (Release UX Part A / task spec sections 1-2).

The Agent page's primary interaction becomes free-text conversation, but a
casual question must never silently trigger a scientific action -- so the
FIRST thing a free-text turn goes through is a typed, closed
:class:`ResearchConversationIntent`, decided by structured feature extraction
over the text (:func:`_extract_signals`) plus an ordered table of named rules
(:data:`_RULES`), never a single flat "if word in text" chain. This keeps the
router auditable and testable rule-by-rule, and keeps the door open for a
future model-assisted classifier (:class:`IntentClassifier` is a Protocol;
:func:`classify_intent` is the default, zero-dependency, deterministic
implementation every test in this codebase uses) without changing the
downstream contract.

This module is intentionally UI- and registry-free: it takes text and a small
:class:`ConversationSignalContext` (what roots/strategies exist to recognise,
whether a strategy is currently bound in the conversation, whether there is
prior conversation history) and returns a typed :class:`IntentClassification`
-- nothing here touches `alpha_agent.ui` or `alpha_agent.registry`. Evidence
binding and response composition live in `alpha_agent.ui.conversation_engine`,
the same layering `alpha_agent.agents.research_agent` (pure) /
`alpha_agent.ui.llm_demo` (orchestration) already uses.

ACTIONS ARE NEVER EXECUTED HERE (Part A section 4): `RUN_FAST_SCREEN`,
`FREEZE_CANDIDATES`, and `RUN_STRICT_VALIDATION` are recognised intents, not
verbs -- classifying a turn as one of them produces a label and nothing else;
the caller decides, separately, whether to render an explicit action button.
No function in this module can run a backtest, freeze a manifest, or write
the registry -- there is no such capability in its import graph at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel


class ResearchConversationIntent(str, Enum):
    """Closed intent catalog (task spec section 2). `GENERAL_QUANT_QUESTION`
    is the safe fallback -- never a made-up "I don't know" catch-all outside
    the enum, and never silently reclassified as an action."""

    MARKET_QUESTION = "MARKET_QUESTION"
    STRATEGY_EXPLANATION = "STRATEGY_EXPLANATION"
    CANDIDATE_COMPARISON = "CANDIDATE_COMPARISON"
    VALIDATION_EXPLANATION = "VALIDATION_EXPLANATION"
    SOURCE_QUESTION = "SOURCE_QUESTION"
    PROFILE_QUESTION = "PROFILE_QUESTION"
    RESEARCH_DISCOVERY = "RESEARCH_DISCOVERY"
    RESEARCH_FOLLOWUP = "RESEARCH_FOLLOWUP"
    OBSERVATION_TO_FACTOR = "OBSERVATION_TO_FACTOR"
    WHAT_IF_PROFILE = "WHAT_IF_PROFILE"
    WHAT_IF_STRATEGY = "WHAT_IF_STRATEGY"
    MARKET_COMPARISON = "MARKET_COMPARISON"
    RUN_FAST_SCREEN = "RUN_FAST_SCREEN"
    FREEZE_CANDIDATES = "FREEZE_CANDIDATES"
    RUN_STRICT_VALIDATION = "RUN_STRICT_VALIDATION"
    GENERAL_QUANT_QUESTION = "GENERAL_QUANT_QUESTION"
    UNSUPPORTED_CURRENT_EVENT_CAUSAL_QUESTION = "UNSUPPORTED_CURRENT_EVENT_CAUSAL_QUESTION"


#: Intents that name a state-transition ACTION rather than a question. Every
#: caller (the UI, tests) uses this closed set to decide whether a response
#: must be presented as "I can test that -- [Run ...]" rather than an answer.
ACTION_INTENTS: frozenset[ResearchConversationIntent] = frozenset(
    {
        ResearchConversationIntent.RUN_FAST_SCREEN,
        ResearchConversationIntent.FREEZE_CANDIDATES,
        ResearchConversationIntent.RUN_STRICT_VALIDATION,
    }
)


@dataclass(frozen=True)
class ConversationSignalContext:
    """What the classifier is allowed to know about the surrounding
    conversation -- deliberately small and UI-free. `approved_roots` lets
    root-symbol mentions be recognised without hardcoding a market list here;
    `has_bound_strategy` / `has_history` are single booleans, never a full
    transcript, so this module cannot accidentally depend on session
    internals."""

    approved_roots: tuple[str, ...] = ()
    has_bound_strategy: bool = False
    has_history: bool = False


@dataclass(frozen=True)
class IntentSignals:
    """Structured features extracted from raw text -- the router's rules
    reference these NAMED fields, never raw substring checks scattered
    ad hoc through the rule table (task spec section 2: "do not rely purely
    on string matching")."""

    mentioned_roots: tuple[str, ...] = ()
    has_compare_word: bool = False
    has_why_word: bool = False
    has_today_word: bool = False
    has_move_word: bool = False  # fell/rose/rallied/crashed/dropped/surged
    has_whatif_word: bool = False
    has_profile_vocab: bool = False
    has_strategy_vocab: bool = False
    has_validation_vocab: bool = False
    has_source_vocab: bool = False
    has_discovery_vocab: bool = False
    has_followup_word: bool = False
    has_observation_to_factor_vocab: bool = False
    has_run_fast_screen_phrase: bool = False
    has_freeze_phrase: bool = False
    has_run_validation_phrase: bool = False
    has_general_quant_vocab: bool = False


class IntentClassification(BaseModel):
    """The router's output. `matched_rule` names WHICH rule fired -- purely
    diagnostic (tests assert on it), never part of any downstream contract."""

    model_config = {"frozen": True, "extra": "forbid"}

    intent: ResearchConversationIntent
    mentioned_roots: tuple[str, ...] = ()
    matched_rule: str = "fallback"
    is_action: bool = False


# ---------------------------------------------------------------------------
# signal extraction -- one small, named, regex-backed check per concept
# ---------------------------------------------------------------------------

_WORD = r"(?<![A-Za-z0-9]){}(?![A-Za-z0-9])"


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(re.search(_WORD.format(re.escape(w)), text) for w in words)


_COMPARE_WORDS = ("compare", "versus", "vs", "vs.", "relative to", "different from", "how different")
_WHY_WORDS = ("why",)
_TODAY_WORDS = ("today", "now", "currently", "right now", "this morning", "this session")
_MOVE_WORDS = ("fell", "fall", "falling", "rose", "rising", "rallied", "rally", "crashed", "crash",
               "dropped", "drop", "surged", "surge", "tanked", "spiked", "plunged", "plunge")
_WHATIF_WORDS = ("what if", "suppose", "if i")
_PROFILE_VOCAB = ("drawdown", "risk tolerance", "risk profile", "my profile", "overnight", "holding period",
                  "trading frequency", "turnover", "tolerate", "willing to")
_STRATEGY_VOCAB = ("strategy", "candidate", "this trade", "the trade", "position", "backtest")
_VALIDATION_VOCAB = ("inconclusive", "reject", "rejected", "fail validation", "failed validation", "pass",
                     "passed", "p-value", "pvalue", "fdr", "dsr", "gate", "bh-fdr", "multiple testing",
                     "significance", "significant", "verdict")
_SOURCE_VOCAB = ("source", "academic", "paper", "research paper", "inspired", "inspiration", "github",
                 "repo", "repository", "where did this come from", "citation", "cite")
_DISCOVERY_VOCAB = ("find me", "find a", "find another", "look for", "research whether", "new idea",
                     "alternative", "propose", "search for", "discover", "come up with",
                     "what should we test", "test next", "what next", "what's next")
_FOLLOWUP_WORDS = ("also", "and what about", "what about", "another one", "one more", "again")
#: Phase 1 (prompt 1 section 12): "EIA inventories dropped sharply. What
#: could I research?" -- a distinct question shape from ordinary discovery
#: vocab (`_DISCOVERY_VOCAB`'s "find me"/"propose"/"discover"): the user is
#: asking for the Observation -> Mechanism -> Factor translation for
#: something that just happened/was observed, not a free-standing new idea.
_OBSERVATION_TO_FACTOR_VOCAB = (
    "what could i research", "what should i research", "what can i research",
    "turn this into a hypothesis", "translate this into a factor", "translate this into a hypothesis",
    "what's the mechanism", "what is the mechanism", "how does this become a hypothesis",
    "what factor could this be", "what could this become",
)
_RUN_FAST_SCREEN_PHRASES = ("run fast screen", "run the fast screen", "fast-screen", "run a fast screen")
_FREEZE_PHRASES = ("freeze candidates", "freeze the candidates", "freeze the top", "freeze this", "freeze it")
_RUN_VALIDATION_PHRASES = ("run strict validation", "run validation", "run the validation",
                            "strict validation", "run out-of-sample validation")
_GENERAL_QUANT_VOCAB = ("what is", "what does", "explain", "define", "how does", "how do you compute")


#: Deterministic product-alias resolver (Release UX product-consolidation
#: acceptance pass, task spec section 4): common natural-language names for a
#: certified root that are NOT the ticker symbol itself. Checked as part of
#: `_mentioned_roots`, so it runs BEFORE `alpha_agent.agents.evidence_scope`'s
#: broad asset-class CATEGORY matching (that module's rule 1 fires whenever
#: `mentioned_roots` is non-empty) -- "what's happening in crude?" resolves to
#: CL rather than the whole ENERGY category. An explicit ticker symbol always
#: wins first (see `_mentioned_roots` below); this is a small, fixed lookup
#: table, never an LLM router. Longer/more specific phrases are listed before
#: their shorter substrings purely for readability -- matching itself is
#: whole-phrase (word-boundary), never a bare substring check, so ordering
#: does not affect correctness.
_PRODUCT_ALIASES: tuple[tuple[str, str], ...] = (
    ("e-mini nasdaq 100", "NQ"), ("e-mini nasdaq", "NQ"), ("nasdaq 100", "NQ"), ("nasdaq", "NQ"),
    ("e-mini s&p 500", "ES"), ("e-mini s&p", "ES"), ("s&p 500", "ES"), ("s&p", "ES"),
    ("crude oil", "CL"), ("crude", "CL"), ("wti", "CL"),
    ("gold", "GC"),
    ("10-year treasury", "ZN"), ("10 year treasury", "ZN"), ("10y treasury", "ZN"),
)


def _aliased_roots(text: str, approved_roots: tuple[str, ...]) -> tuple[str, ...]:
    """Alias hits for roots actually in the caller's approved universe --
    order-preserving, de-duplicated. Never resolves an alias for a root the
    caller does not recognise (mirrors `_mentioned_roots`'s own symbol check)."""
    lowered = text.lower()
    found: list[str] = []
    for phrase, root in _PRODUCT_ALIASES:
        if root not in approved_roots or root in found:
            continue
        if re.search(_WORD.format(re.escape(phrase)), lowered):
            found.append(root)
    return tuple(found)


def _mentioned_roots(text: str, approved_roots: tuple[str, ...]) -> tuple[str, ...]:
    """Explicit futures ticker symbols first (highest priority -- task spec
    section 4A), then any product-alias hits not already covered by an
    explicit symbol. A question naming both ("CL crude oil") is not double-
    counted."""
    upper = text.upper()
    explicit = [r for r in approved_roots if re.search(_WORD.format(re.escape(r)), upper)]
    aliased = [r for r in _aliased_roots(text, approved_roots) if r not in explicit]
    return tuple(explicit + aliased)


def _extract_signals(text: str, context: ConversationSignalContext) -> IntentSignals:
    lowered = text.lower()
    return IntentSignals(
        mentioned_roots=_mentioned_roots(text, context.approved_roots),
        has_compare_word=_has_any(lowered, _COMPARE_WORDS),
        has_why_word=_has_any(lowered, _WHY_WORDS),
        has_today_word=_has_any(lowered, _TODAY_WORDS),
        has_move_word=_has_any(lowered, _MOVE_WORDS),
        has_whatif_word=any(w in lowered for w in _WHATIF_WORDS),
        has_profile_vocab=any(w in lowered for w in _PROFILE_VOCAB),
        has_strategy_vocab=any(w in lowered for w in _STRATEGY_VOCAB),
        has_validation_vocab=any(w in lowered for w in _VALIDATION_VOCAB),
        has_source_vocab=any(w in lowered for w in _SOURCE_VOCAB),
        has_discovery_vocab=any(w in lowered for w in _DISCOVERY_VOCAB),
        has_followup_word=any(w in lowered for w in _FOLLOWUP_WORDS),
        has_observation_to_factor_vocab=any(w in lowered for w in _OBSERVATION_TO_FACTOR_VOCAB),
        has_run_fast_screen_phrase=any(w in lowered for w in _RUN_FAST_SCREEN_PHRASES),
        has_freeze_phrase=any(w in lowered for w in _FREEZE_PHRASES),
        has_run_validation_phrase=any(w in lowered for w in _RUN_VALIDATION_PHRASES),
        has_general_quant_vocab=any(w in lowered for w in _GENERAL_QUANT_VOCAB),
    )


# ---------------------------------------------------------------------------
# ordered rule table -- most specific first. Each rule is a small, named,
# pure predicate over (signals, context) -- auditable and independently
# testable, never a monolithic if/elif chain over raw text.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Rule:
    name: str
    intent: ResearchConversationIntent
    predicate: object = field(repr=False)  # Callable[[IntentSignals, ConversationSignalContext], bool]


_RULES: tuple[_Rule, ...] = (
    _Rule("run_fast_screen_phrase", ResearchConversationIntent.RUN_FAST_SCREEN,
          lambda s, c: s.has_run_fast_screen_phrase),
    _Rule("freeze_phrase", ResearchConversationIntent.FREEZE_CANDIDATES,
          lambda s, c: s.has_freeze_phrase),
    _Rule("run_validation_phrase", ResearchConversationIntent.RUN_STRICT_VALIDATION,
          lambda s, c: s.has_run_validation_phrase),
    _Rule("whatif_profile_vocab", ResearchConversationIntent.WHAT_IF_PROFILE,
          lambda s, c: s.has_whatif_word and s.has_profile_vocab),
    _Rule("whatif_strategy_vocab", ResearchConversationIntent.WHAT_IF_STRATEGY,
          lambda s, c: s.has_whatif_word and (s.has_strategy_vocab or c.has_bound_strategy)),
    _Rule("compare_two_roots", ResearchConversationIntent.MARKET_COMPARISON,
          lambda s, c: s.has_compare_word and len(s.mentioned_roots) >= 2),
    _Rule("compare_strategy_vocab", ResearchConversationIntent.CANDIDATE_COMPARISON,
          lambda s, c: s.has_compare_word and (s.has_strategy_vocab or c.has_bound_strategy)),
    _Rule("why_validation_vocab", ResearchConversationIntent.VALIDATION_EXPLANATION,
          lambda s, c: s.has_why_word and s.has_validation_vocab),
    _Rule("why_move_today_no_evidence", ResearchConversationIntent.UNSUPPORTED_CURRENT_EVENT_CAUSAL_QUESTION,
          lambda s, c: s.has_why_word and s.has_move_word and (s.has_today_word or bool(s.mentioned_roots))),
    _Rule("observation_to_factor_vocab", ResearchConversationIntent.OBSERVATION_TO_FACTOR,
          lambda s, c: s.has_observation_to_factor_vocab),
    # An explicit discovery VERB ("research whether", "find me", "look for",
    # "what should we test next") wins over a bare mention of "academic" /
    # "paper" -- initiating new research is the stronger, more actionable
    # signal even when the request names a source-flavoured topic.
    _Rule("followup_with_history", ResearchConversationIntent.RESEARCH_FOLLOWUP,
          lambda s, c: c.has_history and (s.has_followup_word or s.has_discovery_vocab)),
    _Rule("discovery_vocab", ResearchConversationIntent.RESEARCH_DISCOVERY,
          lambda s, c: s.has_discovery_vocab),
    _Rule("source_vocab", ResearchConversationIntent.SOURCE_QUESTION,
          lambda s, c: s.has_source_vocab),
    _Rule("why_strategy_vocab", ResearchConversationIntent.STRATEGY_EXPLANATION,
          lambda s, c: (s.has_why_word or s.has_strategy_vocab) and (s.has_strategy_vocab or c.has_bound_strategy)
          and not s.has_validation_vocab),
    _Rule("profile_vocab", ResearchConversationIntent.PROFILE_QUESTION,
          lambda s, c: s.has_profile_vocab and not s.has_whatif_word),
    _Rule("market_today_root", ResearchConversationIntent.MARKET_QUESTION,
          lambda s, c: bool(s.mentioned_roots) and (s.has_today_word or not s.has_general_quant_vocab)),
    _Rule("general_quant_vocab", ResearchConversationIntent.GENERAL_QUANT_QUESTION,
          lambda s, c: s.has_general_quant_vocab),
)


def classify_intent(
    text: str, *, context: ConversationSignalContext | None = None,
) -> IntentClassification:
    """Deterministic, structured intent routing. Never raises on empty/odd
    input -- an unrecognised turn falls back to `GENERAL_QUANT_QUESTION`,
    never to an action intent (the safe direction to fail in)."""
    context = context or ConversationSignalContext()
    if not text or not text.strip():
        return IntentClassification(intent=ResearchConversationIntent.GENERAL_QUANT_QUESTION, matched_rule="empty_input")

    signals = _extract_signals(text, context)
    for rule in _RULES:
        if rule.predicate(signals, context):
            intent = rule.intent
            return IntentClassification(
                intent=intent, mentioned_roots=signals.mentioned_roots, matched_rule=rule.name,
                is_action=intent in ACTION_INTENTS,
            )
    return IntentClassification(
        intent=ResearchConversationIntent.GENERAL_QUANT_QUESTION,
        mentioned_roots=signals.mentioned_roots, matched_rule="fallback",
    )


__all__ = [
    "ACTION_INTENTS",
    "ConversationSignalContext",
    "IntentClassification",
    "IntentSignals",
    "ResearchConversationIntent",
    "classify_intent",
]
