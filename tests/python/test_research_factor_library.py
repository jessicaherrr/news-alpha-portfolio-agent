"""Research Golden Path -- Factor Library / Candidate Mechanisms tests
(task spec section 18): the Research UI must derive factors from the REAL
feature catalog, factor counts must be real, no fictitious factor family may
be shown, and every candidate mechanism must reference only supported
features/families.
"""
from __future__ import annotations

import alpha_agent.features.compute  # noqa: F401 -- registers feature defs
import pytest
from alpha_agent.features import REGISTRY
from alpha_agent.registry import TrialRole
from alpha_agent.strategy.candidates_phase_13_5c import BASELINE_FAMILIES, build_candidate_manifest
from alpha_agent.ui import candidate_mechanisms, factor_library, services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(),
    reason="Phase 14 registry sqlite not present in this checkout",
)


def test_mechanism_families_only_reports_real_populated_feature_families():
    real_families = {getattr(k, "value", k) for k in {REGISTRY.get(kind).family for kind in REGISTRY.kinds()}}
    cards = factor_library.mechanism_families()
    reported = {c["family"] for c in cards}
    assert reported, "at least one real mechanism family must be reported"
    assert reported <= real_families, f"fictitious family reported: {reported - real_families}"


def test_mechanism_family_feature_counts_are_real():
    """Every card's `n_features`/`feature_kinds` must exactly match a live
    count from `services.feature_catalog()` -- never estimated or padded."""
    entries = services.feature_catalog()
    by_family: dict[str, list[str]] = {}
    for e in entries:
        by_family.setdefault(e["family"], []).append(e["kind"])
    for card in factor_library.mechanism_families():
        real_kinds = sorted(by_family[card["family"]])
        assert card["feature_kinds"] == real_kinds
        assert card["n_features"] == len(real_kinds)


def test_unsupported_axes_never_appear_as_populated_cards():
    """Carry/macro/cross_market (declared, zero-feature) and the task-spec-only
    vocabulary (Event/Catalyst, Seasonality) must never be presented as if
    they had real feature/coverage counts."""
    card_labels = {c["label"] for c in factor_library.mechanism_families()}
    for axis in factor_library.unsupported_axes():
        assert axis["label"] not in card_labels


def test_coverage_status_is_derived_from_real_registry_counts():
    """A family with zero mapped baseline strategies is NOT_APPLICABLE; a
    mapped family's `n_experiments` must equal the real sum of
    `services.list_experiments` rows for its mapped baseline families."""
    cards = {c["family"]: c for c in factor_library.mechanism_families()}
    for family, baseline_families in factor_library.FEATURE_FAMILY_TO_BASELINE_FAMILIES.items():
        if family not in cards:
            continue
        card = cards[family]
        if not baseline_families:
            assert card["coverage_status"] == factor_library.NOT_APPLICABLE
            continue
        expected = sum(
            len(services.list_experiments(strategy_family=f, trial_role=TrialRole.CANONICAL))
            for f in baseline_families
        )
        assert card["n_experiments"] == expected


def test_deterministic_candidates_reference_only_real_baseline_families():
    real_families = set(BASELINE_FAMILIES) | {"silver_bullet"}
    for root in services.approved_universe():
        for cand in candidate_mechanisms.deterministic_candidates(root):
            assert cand["family_key"] in real_families
            assert cand["provenance"] == candidate_mechanisms.PROVENANCE_DETERMINISTIC


def test_deterministic_candidate_fingerprints_match_the_frozen_manifest():
    manifest = build_candidate_manifest()
    by_key = {(t.root_symbol, t.family_key): t.strategy_fingerprint for t in manifest.trials}
    for root in services.approved_universe():
        for cand in candidate_mechanisms.deterministic_candidates(root):
            assert cand["strategy_fingerprint"] == by_key[(root, cand["family_key"])]


def test_deterministic_candidate_tested_flag_matches_real_registry_verdict():
    for root in services.approved_universe():
        for cand in candidate_mechanisms.deterministic_candidates(root):
            rows = services.list_experiments(
                root_symbol=root, strategy_family=cand["family_key"],
            )
            canonical = [r for r in rows if r["trial_role"] == "CANONICAL"]
            if cand["tested"]:
                assert canonical, "candidate marked tested but no CANONICAL row exists"
                assert cand["verdict"] == canonical[0]["verdict"]
            else:
                assert not canonical
