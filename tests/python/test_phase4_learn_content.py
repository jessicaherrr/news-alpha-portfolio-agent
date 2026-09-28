"""Phase 4 -- Learn content: Concepts, Strategy explainers, Learning Paths.

Fixed content, but every real-code `pointers` entry is checked to actually
resolve (never a stale/fabricated reference), and every cross-reference
between modules (concept<->gate, path step<->concept/family) is checked to
resolve to something real.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from alpha_agent.learn.autopsy import _LABEL_TO_FAIL_CODE
from alpha_agent.learn.concepts import (
    CONCEPT_CATEGORIES,
    CONCEPT_LIBRARY,
    GATE_FAIL_CODE_TO_CONCEPT,
    get_concept,
    list_concepts,
)
from alpha_agent.learn.learning_paths import list_learning_paths
from alpha_agent.learn.strategy_explainers import build_strategy_explainer, list_strategy_explainers
from alpha_agent.ui.services import GATE_DEFINITIONS

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_pointer(pointer: str) -> None:
    """A pointer is one of three real, checked forms: a relative file path
    (contains ``/`` -- a C++ header or script), ``module.path:attribute``
    (must import and resolve the attribute), or a bare importable module
    path (neither -- must import cleanly)."""
    if "/" in pointer:
        path = REPO_ROOT / pointer
        assert path.exists(), f"{pointer!r}: no such file at {path}"
    elif ":" in pointer:
        module_path, _, attr = pointer.partition(":")
        mod = importlib.import_module(module_path)
        assert hasattr(mod, attr), f"{pointer!r}: {module_path} has no attribute {attr!r}"
    else:
        importlib.import_module(pointer)


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------


def test_every_concept_has_a_unique_id_and_valid_category():
    concepts = list_concepts()
    ids = [c.concept_id for c in concepts]
    assert len(ids) == len(set(ids))
    assert len(concepts) >= 10
    for c in concepts:
        assert c.category in CONCEPT_CATEGORIES
        assert c.intuition and c.quant and c.implementation and c.one_line


def test_every_concept_pointer_resolves_to_something_real():
    for c in list_concepts():
        for pointer in c.pointers:
            _resolve_pointer(pointer)


def test_get_concept_matches_library_and_returns_none_for_unknown():
    for c in list_concepts():
        assert get_concept(c.concept_id) == c
    assert get_concept("not_a_real_concept") is None
    assert CONCEPT_LIBRARY == {c.concept_id: c for c in list_concepts()}


def test_gate_fail_code_to_concept_covers_exactly_the_real_nine_gates():
    """Every `GATE_DEFINITIONS` fail code (the real, already-tested gate
    table -- `test_gate_definitions_match_the_real_reason_code_enum` proves
    IT is correct against the ReasonCode enum) must have a Concept, and no
    concept claims a fail code that isn't one of these nine."""
    real_fail_codes = {code for _, code in GATE_DEFINITIONS}
    assert set(GATE_FAIL_CODE_TO_CONCEPT) == real_fail_codes
    for concept_id in GATE_FAIL_CODE_TO_CONCEPT.values():
        assert concept_id in CONCEPT_LIBRARY


def test_autopsy_label_to_fail_code_is_byte_identical_to_gate_definitions():
    """`alpha_agent.learn.autopsy._LABEL_TO_FAIL_CODE` is a deliberate,
    documented duplication of `services.GATE_DEFINITIONS` (kept so the
    backend `learn` package never imports the `ui` layer) -- this proves the
    duplication has not drifted."""
    assert _LABEL_TO_FAIL_CODE == dict(GATE_DEFINITIONS)


# ---------------------------------------------------------------------------
# Strategy explainers
# ---------------------------------------------------------------------------


def test_strategy_explainers_cover_exactly_the_compiler_supported_families():
    from alpha_agent.agents.compiler_agent import COMPILER_FAMILY_KEYS

    explainers = list_strategy_explainers()
    assert {e.family_key for e in explainers} == set(COMPILER_FAMILY_KEYS)


def test_strategy_explainer_text_is_projected_verbatim_from_the_frozen_family_doc():
    from alpha_agent.agents.compiler_agent import build_family_catalog

    cards = {c.family_key: c for c in build_family_catalog()}
    for exp in list_strategy_explainers():
        card = cards[exp.family_key]
        assert exp.intuition == card.economic_mechanism
        assert exp.quant == card.formula_and_timing


def test_strategy_explainer_chain_has_five_stages_in_order():
    expected = ("Economic idea", "Equations & timing", "Python feature", "StrategySpec", "C++ execution")
    for exp in list_strategy_explainers():
        assert tuple(s.stage for s in exp.chain) == expected


def test_strategy_explainer_pointers_resolve():
    for exp in list_strategy_explainers():
        for step in exp.chain:
            for pointer in step.pointers:
                _resolve_pointer(pointer)


def test_build_strategy_explainer_raises_key_error_for_unknown_family():
    with pytest.raises(KeyError):
        build_strategy_explainer("not_a_real_family")


# ---------------------------------------------------------------------------
# Learning paths
# ---------------------------------------------------------------------------


def test_every_learning_path_step_resolves_to_a_real_concept_or_family():
    from alpha_agent.agents.compiler_agent import COMPILER_FAMILY_KEYS

    for path in list_learning_paths():
        assert path.steps
        for step in path.steps:
            if step.kind == "concept":
                assert step.ref_id in CONCEPT_LIBRARY, f"{path.path_id}: unknown concept {step.ref_id!r}"
            elif step.kind == "strategy":
                assert step.ref_id in COMPILER_FAMILY_KEYS, f"{path.path_id}: unknown family {step.ref_id!r}"
            else:
                raise AssertionError(f"unknown step kind {step.kind!r}")
