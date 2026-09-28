"""Product-alias resolution BEFORE category matching (Release UX product-
consolidation acceptance pass, task spec sections 4/25). "crude"/"WTI"/
"gold"/"Nasdaq" etc. must resolve to their certified root -- a ROOT_SET
evidence scope of exactly one -- rather than falling through to a broad
asset-class CATEGORY scope. Explicit futures symbols remain highest priority
and broad category words (e.g. "energy") remain CATEGORY, never narrowed to
one product by this alias table.
"""
from __future__ import annotations

import pytest
from alpha_agent.agents.conversation import ConversationSignalContext, classify_intent
from alpha_agent.agents.evidence_scope import (
    EvidenceScopeKind,
    EvidenceScopeSignals,
    resolve_evidence_scope,
)

ROOTS = ("ES", "NQ", "CL", "GC", "ZN")
CTX = ConversationSignalContext(approved_roots=ROOTS)

_ROOT_CATEGORY = {"ES": "EQUITY_INDEX", "NQ": "EQUITY_INDEX", "CL": "ENERGY", "GC": "METALS", "ZN": "RATES"}


def _resolved_roots(text: str, *, cached_roots: tuple[str, ...] = ROOTS) -> tuple[str, ...]:
    classification = classify_intent(text, context=CTX)
    scope = resolve_evidence_scope(
        EvidenceScopeSignals(
            text=text, mentioned_roots=classification.mentioned_roots, intent=classification.intent,
            bound_root=None, cached_roots=cached_roots, root_category=_ROOT_CATEGORY,
        )
    )
    return scope


@pytest.mark.parametrize(
    "text,expected_root",
    [
        ("What's happening in crude?", "CL"),
        ("What's happening in crude oil?", "CL"),
        ("What's happening in WTI?", "CL"),
        ("What's happening in gold?", "GC"),
        ("What's happening in Nasdaq?", "NQ"),
        ("What's happening in the Nasdaq 100?", "NQ"),
        ("What's happening in E-mini Nasdaq?", "NQ"),
        ("What's happening in the S&P?", "ES"),
        ("What's happening in the S&P 500?", "ES"),
        ("What's happening in E-mini S&P?", "ES"),
        ("What's happening in the 10-year Treasury?", "ZN"),
        ("What's happening in the 10 year Treasury?", "ZN"),
    ],
)
def test_product_alias_resolves_to_root_set_of_one(text, expected_root):
    scope = _resolved_roots(text)
    assert scope.kind == EvidenceScopeKind.ROOT_SET, (text, scope)
    assert scope.roots == (expected_root,), (text, scope)


def test_explicit_symbol_still_wins_when_both_symbol_and_alias_present():
    scope = _resolved_roots("Why is CL (crude oil) worth watching?")
    assert scope.kind == EvidenceScopeKind.ROOT_SET
    assert scope.roots == ("CL",)


def test_broad_category_word_stays_category_not_narrowed_by_alias_table():
    scope = _resolved_roots("What's interesting in energy?")
    assert scope.kind == EvidenceScopeKind.CATEGORY
    assert scope.category == "ENERGY"
    assert scope.roots == ("CL",)  # only cached root in ENERGY -- category resolution, not alias resolution


def test_global_question_with_no_product_or_category_word_stays_global():
    scope = _resolved_roots("What deserves attention right now?")
    assert scope.kind == EvidenceScopeKind.GLOBAL
