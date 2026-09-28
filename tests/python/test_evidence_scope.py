"""Tests for `alpha_agent.agents.evidence_scope` -- the deterministic,
UI-free scope resolver behind the Agent Experience Consolidation campaign's
question-scoped Claude context (see `claude_conversation.py`'s own module
docstring for the bug this fixes: a global question silently narrowed to
whatever ONE market a different page's local selection happened to have
picked). Pure unit tests -- no Streamlit, no network, no registry.
"""
from __future__ import annotations

from alpha_agent.agents.conversation import ResearchConversationIntent
from alpha_agent.agents.evidence_scope import (
    EvidenceScopeKind,
    EvidenceScopeSignals,
    resolve_evidence_scope,
)


def _signals(**overrides) -> EvidenceScopeSignals:
    base = {
        "text": "", "mentioned_roots": (), "intent": ResearchConversationIntent.GENERAL_QUANT_QUESTION,
        "bound_root": None, "cached_roots": (), "root_category": {},
    }
    base.update(overrides)
    return EvidenceScopeSignals(**base)


def test_global_question_with_no_roots_named_resolves_to_global():
    scope = resolve_evidence_scope(_signals(text="What's interesting right now?", cached_roots=("ES", "NQ", "CL")))
    assert scope.kind == EvidenceScopeKind.GLOBAL
    assert scope.roots == ("ES", "NQ", "CL")


def test_single_named_root_resolves_to_root_set_of_one():
    scope = resolve_evidence_scope(_signals(text="Why is CL interesting?", mentioned_roots=("CL",)))
    assert scope.kind == EvidenceScopeKind.ROOT_SET
    assert scope.roots == ("CL",)


def test_two_named_roots_resolve_to_root_set_of_both():
    scope = resolve_evidence_scope(_signals(text="Why CL instead of NQ?", mentioned_roots=("NQ", "CL")))
    assert scope.kind == EvidenceScopeKind.ROOT_SET
    assert set(scope.roots) == {"NQ", "CL"}


def test_named_root_always_wins_over_a_research_object_intent():
    """An explicit market mention wins even when the question also classifies
    as a research-object intent -- ROOT_SET is checked first."""
    scope = resolve_evidence_scope(
        _signals(
            text="Why did the CL strategy fail?", mentioned_roots=("CL",),
            intent=ResearchConversationIntent.STRATEGY_EXPLANATION, bound_root="NQ",
        )
    )
    assert scope.kind == EvidenceScopeKind.ROOT_SET
    assert scope.roots == ("CL",)


def test_research_object_intent_with_no_named_root_scopes_to_bound_root():
    scope = resolve_evidence_scope(
        _signals(text="Why did it fail?", intent=ResearchConversationIntent.STRATEGY_EXPLANATION, bound_root="NQ")
    )
    assert scope.kind == EvidenceScopeKind.RESEARCH_OBJECT
    assert scope.roots == ("NQ",)


def test_research_object_intent_with_no_bound_root_yields_empty_roots():
    """Never fabricates a root -- an unbound research-object question with
    nothing to scope to yields an empty root set, not a guess."""
    scope = resolve_evidence_scope(
        _signals(text="What did we test before?", intent=ResearchConversationIntent.RESEARCH_DISCOVERY, bound_root=None)
    )
    assert scope.kind == EvidenceScopeKind.RESEARCH_OBJECT
    assert scope.roots == ()


def test_category_question_resolves_only_against_cached_roots_in_that_category():
    scope = resolve_evidence_scope(
        _signals(
            text="What's interesting in energy?", cached_roots=("ES", "CL"),
            root_category={"ES": "EQUITY_INDEX", "CL": "ENERGY", "GC": "METALS"},
        )
    )
    assert scope.kind == EvidenceScopeKind.CATEGORY
    assert scope.category == "ENERGY"
    assert scope.roots == ("CL",)


def test_category_question_never_pulls_in_an_uncached_root():
    """GC is in the METALS category but not in the cached snapshot -- a
    category scope must never claim evidence for it (task spec section 8:
    "do not fetch missing category data automatically")."""
    scope = resolve_evidence_scope(
        _signals(
            text="Compare the metals opportunities.", cached_roots=("CL",),
            root_category={"CL": "ENERGY", "GC": "METALS"},
        )
    )
    assert scope.kind == EvidenceScopeKind.CATEGORY
    assert scope.roots == ()


def test_a_market_pages_local_selection_never_enters_the_signals_at_all():
    """There is no field on `EvidenceScopeSignals` for a page's local
    selection (e.g. the sidebar's Workspace Market) -- this is a structural
    guarantee, not just a behavioural one: the resolver cannot see it even if
    a caller wanted to pass it."""
    assert "selected_root" not in EvidenceScopeSignals.__dataclass_fields__
    assert set(EvidenceScopeSignals.__dataclass_fields__) == {
        "text", "mentioned_roots", "intent", "bound_root", "cached_roots", "root_category",
    }
