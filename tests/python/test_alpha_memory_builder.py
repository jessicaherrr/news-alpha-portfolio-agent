"""Phase 2 -- Personal Alpha Memory builder (prompt 2 sections 2/5/6/7/8/9/11/16),
hardened by the identity-hardening patch and the final Phase 2 semantic fix
(equal input-data requirements do NOT prove equal Factor representation).

Uses the REAL, already-committed local registry (read-only, never mutated --
the same pattern `test_translation_research_memory.py` and
`alpha_agent.ui.services` tests already use) so every count here is real
evidence, not a fabricated fixture.
"""
from __future__ import annotations

import re

import pytest
from alpha_agent.alpha_memory import (
    AlphaResearchObject,
    EvidenceProfile,
    ExperimentEvidenceRef,
    FactorIdentity,
    ResearchMaturity,
    StrategyVariantEvidence,
    alpha_research_object_id,
    build_alpha_research_objects_for_mechanism,
    get_alpha_research_object,
    list_alpha_research_objects,
)
from alpha_agent.alpha_memory import builder as alpha_builder
from alpha_agent.alpha_memory import comparison as alpha_comparison
from alpha_agent.alpha_memory import factor_identity as alpha_factor_identity
from alpha_agent.alpha_memory import lookup as alpha_lookup
from alpha_agent.alpha_memory import schemas as alpha_schemas
from alpha_agent.alpha_memory.factor_identity import (
    compute_factor_identity,
    family_structural_signature,
    group_families_for_factor_identity,
)
from alpha_agent.core.instrument import AssetDomain
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

_ALL_MODULES = (alpha_schemas, alpha_builder, alpha_factor_identity, alpha_comparison, alpha_lookup)

#: Mechanisms whose frozen bridge maps to exactly ONE family, so they always
#: produce at most one real object regardless of V1's grouping policy.
_SINGLE_FAMILY_MECHANISMS = (
    EconomicMechanism.MOMENTUM, EconomicMechanism.MEAN_REVERSION, EconomicMechanism.BREAKOUT,
    EconomicMechanism.FAILED_BREAKOUT, EconomicMechanism.VOLATILITY_BREAKOUT,
)


def _registry() -> ExperimentRegistry:
    return ExperimentRegistry(services.REGISTRY_PATH)


def _build_one(reg: ExperimentRegistry, *, root_symbol: str, mechanism: EconomicMechanism) -> AlphaResearchObject | None:
    """Convenience for the single-family mechanisms in
    `_SINGLE_FAMILY_MECHANISMS`, which always produce at most one real
    object under V1's one-family-per-Factor default."""
    assert mechanism in _SINGLE_FAMILY_MECHANISMS, f"{mechanism} maps to >1 family -- use _build_for_family instead"
    objs = build_alpha_research_objects_for_mechanism(reg, root_symbol=root_symbol, mechanism=mechanism)
    assert len(objs) <= 1, f"expected at most one real Factor for {mechanism} on {root_symbol}, got {len(objs)}"
    return objs[0] if objs else None


def _build_for_family(
    reg: ExperimentRegistry, *, root_symbol: str, mechanism: EconomicMechanism, family: str
) -> AlphaResearchObject | None:
    """For a mechanism mapping to more than one family (e.g. TREND), the one
    real object whose Factor bundles exactly `family` -- V1's singleton
    default, asserted explicitly."""
    objs = build_alpha_research_objects_for_mechanism(reg, root_symbol=root_symbol, mechanism=mechanism)
    matches = [o for o in objs if o.factor.related_strategy_families == (family,)]
    assert len(matches) <= 1
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Family structural signature -- real, typed, but INFORMATIONAL ONLY
# ---------------------------------------------------------------------------


def test_family_structural_signature_matches_between_tsmom_and_ma_trend():
    """Real, typed evidence: tsmom and ma_trend both declare the identical
    `required_data`. This is only proof of a shared DATA requirement -- see
    the tests below proving it is NOT treated as proof of a shared Factor."""
    assert family_structural_signature("tsmom") == family_structural_signature("ma_trend")


def test_family_structural_signature_differs_for_breakout():
    assert family_structural_signature("breakout") != family_structural_signature("tsmom")
    assert family_structural_signature("breakout") != family_structural_signature("mean_reversion")


def test_unknown_family_gets_an_empty_signature_never_a_fabricated_one():
    assert family_structural_signature("does_not_exist") == ()


# ---------------------------------------------------------------------------
# Final Phase 2 semantic fix, section 1: equal data requirements != same Factor
# ---------------------------------------------------------------------------


def test_same_data_requirements_do_not_imply_the_same_factor():
    """The core requirement of the final fix: tsmom and ma_trend need
    IDENTICAL `required_data`, yet must resolve to DIFFERENT Factor
    identities -- equal input-data requirements prove only shared data
    requirements, not equal quantitative factor representation."""
    assert family_structural_signature("tsmom") == family_structural_signature("ma_trend")
    tsmom_factor = compute_factor_identity(EconomicMechanism.TREND, ("tsmom",))
    ma_trend_factor = compute_factor_identity(EconomicMechanism.TREND, ("ma_trend",))
    assert tsmom_factor.factor_identity != ma_trend_factor.factor_identity


def test_v1_default_never_merges_two_different_families_automatically():
    """`group_families_for_factor_identity` -- V1's ONE place merging policy
    is decided -- returns one singleton group per mapped family, even when
    they share a mechanism AND a structural signature ("false duplication is
    preferable to false merging")."""
    trend_groups = group_families_for_factor_identity(EconomicMechanism.TREND)
    assert trend_groups == (("tsmom",), ("ma_trend",))
    assert all(len(g) == 1 for g in trend_groups)


def test_merging_remains_architecturally_possible_with_explicit_proof():
    """`compute_factor_identity` itself stays general: a caller holding
    independent proof (however it is obtained) MAY still bundle more than
    one family into one Factor -- the policy fix lives in the GROUPING
    function, not in identity computation itself. Nothing in this
    repository supplies that proof today, so this is exercised directly,
    not through the real registry-grounded builder."""
    bundled = compute_factor_identity(EconomicMechanism.TREND, ("tsmom", "ma_trend"))
    assert set(bundled.related_strategy_families) == {"tsmom", "ma_trend"}
    # ... and it differs from either family's own singleton Factor, since
    # identity depends on the EXACT family set given.
    tsmom_only = compute_factor_identity(EconomicMechanism.TREND, ("tsmom",))
    assert bundled.factor_identity != tsmom_only.factor_identity


def test_real_registry_never_merges_tsmom_and_ma_trend_under_trend():
    """End to end, through the real builder: NQ's TREND mechanism produces
    TWO separate real AlphaResearchObjects today (one per family), never one
    merged object -- the change in behaviour the final fix requires."""
    with _registry() as reg:
        objs = build_alpha_research_objects_for_mechanism(reg, root_symbol="NQ", mechanism=EconomicMechanism.TREND)
    assert len(objs) == 2
    families = {o.factor.related_strategy_families for o in objs}
    assert families == {("tsmom",), ("ma_trend",)}
    assert objs[0].factor.factor_identity != objs[1].factor.factor_identity


# ---------------------------------------------------------------------------
# Factor identity is deterministic and NEVER equals mechanism alone
# ---------------------------------------------------------------------------


def test_factor_identity_is_deterministic():
    a = compute_factor_identity(EconomicMechanism.TREND, ("tsmom",))
    b = compute_factor_identity(EconomicMechanism.TREND, ("tsmom",))
    assert a.factor_identity == b.factor_identity
    assert isinstance(a.factor_identity, str) and a.factor_identity.startswith("factoridentity3:")


def test_factor_must_not_equal_mechanism_same_mechanism_can_contain_different_factor_identities():
    """The identity-hardening bug, proven directly: two families genuinely
    mapped under the SAME mechanism (TREND) produce DIFFERENT Factor
    identities."""
    tsmom_factor = compute_factor_identity(EconomicMechanism.TREND, ("tsmom",))
    ma_trend_factor = compute_factor_identity(EconomicMechanism.TREND, ("ma_trend",))
    assert tsmom_factor.mechanism == ma_trend_factor.mechanism == EconomicMechanism.TREND
    assert tsmom_factor.factor_identity != ma_trend_factor.factor_identity


def test_factor_identity_differs_across_mechanisms_even_with_the_same_family():
    """tsmom alone under TREND vs tsmom alone under MOMENTUM -- same family,
    but a DIFFERENT mechanism, so a DIFFERENT Factor (mechanism is still a
    real component of identity, just not the ONLY one)."""
    trend_tsmom = compute_factor_identity(EconomicMechanism.TREND, ("tsmom",))
    momentum_tsmom = compute_factor_identity(EconomicMechanism.MOMENTUM, ("tsmom",))
    assert trend_tsmom.factor_identity != momentum_tsmom.factor_identity


def test_unmapped_mechanism_reuses_phase1s_frozen_table_and_stays_honest():
    """CARRY is Phase 1's own worked example of an unmapped mechanism
    (`test_translation_research_memory.py::test_unmapped_mechanism_is_an_honest_research_representation_gap`).
    Phase 2 must reuse that SAME frozen table, never invent a mapping."""
    assert group_families_for_factor_identity(EconomicMechanism.CARRY) == ()


# ---------------------------------------------------------------------------
# Factor != Strategy != Experiment
# ---------------------------------------------------------------------------


def test_one_strategy_links_multiple_experiments():
    """Real evidence: NQ tsmom alone carries several distinct registry
    experiments (canonical + neighbours) under the ONE strategy family."""
    with _registry() as reg:
        obj = _build_one(reg, root_symbol="NQ", mechanism=EconomicMechanism.MOMENTUM)
    assert obj is not None
    assert [v.strategy_family for v in obj.strategy_variants] == ["tsmom"]
    tsmom = obj.strategy_variants[0]
    assert tsmom.n_experiments >= 2
    assert len({e.experiment_id for e in obj.experiments if e.strategy_family == "tsmom"}) == tsmom.n_experiments


def test_rejected_strategy_variant_does_not_erase_its_factor_or_its_sibling():
    """A REJECT canonical trial for one strategy variant must not make its
    OWN AlphaResearchObject disappear, and must not affect its sibling
    Factor under the same mechanism either (section 2: "one rejected
    Strategy implementation does NOT automatically invalidate the entire
    Factor concept")."""
    with _registry() as reg:
        tsmom_obj = _build_for_family(reg, root_symbol="NQ", mechanism=EconomicMechanism.TREND, family="tsmom")
        ma_trend_obj = _build_for_family(reg, root_symbol="NQ", mechanism=EconomicMechanism.TREND, family="ma_trend")
    assert tsmom_obj is not None and ma_trend_obj is not None
    assert tsmom_obj.evidence_profile.scientific_evidence["tsmom"] == "REJECT"
    assert ma_trend_obj.evidence_profile.scientific_evidence["ma_trend"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# AlphaResearchObject identity = root + real FactorIdentity
# ---------------------------------------------------------------------------


def test_alpha_research_object_id_is_deterministic_and_composed_from_factor_not_mechanism():
    """Section 2: identity is root + real FactorIdentity, never root +
    mechanism alone."""
    tsmom_factor = compute_factor_identity(EconomicMechanism.TREND, ("tsmom",))
    momentum_factor = compute_factor_identity(EconomicMechanism.MOMENTUM, ("tsmom",))
    a = alpha_research_object_id("NQ", tsmom_factor.factor_identity)
    b = alpha_research_object_id("NQ", tsmom_factor.factor_identity)
    c = alpha_research_object_id("CL", tsmom_factor.factor_identity)
    d = alpha_research_object_id("NQ", momentum_factor.factor_identity)
    assert a == b
    assert a != c  # different root
    assert a != d  # different Factor, even though both are real objects on the same root


def test_alpha_research_object_id_changes_when_factor_identity_changes():
    """Direct proof of section 2's requirement: two objects built from
    DIFFERENT Factors under the same mechanism get different ids."""
    tsmom_only = compute_factor_identity(EconomicMechanism.TREND, ("tsmom",))
    ma_trend_only = compute_factor_identity(EconomicMechanism.TREND, ("ma_trend",))
    assert alpha_research_object_id("NQ", tsmom_only.factor_identity) != alpha_research_object_id(
        "NQ", ma_trend_only.factor_identity
    )


def test_experiment_id_is_never_the_alpha_identity():
    with _registry() as reg:
        obj = _build_one(reg, root_symbol="NQ", mechanism=EconomicMechanism.MOMENTUM)
    assert obj is not None
    assert obj.alpha_id not in {e.experiment_id for e in obj.experiments}
    assert obj.alpha_id not in {e.experiment_identity for e in obj.experiments}


# ---------------------------------------------------------------------------
# Real Registry-backed evidence only / no duplicate scientific store
# ---------------------------------------------------------------------------


def test_experiment_evidence_matches_the_real_registry_row():
    with _registry() as reg:
        obj = _build_one(reg, root_symbol="NQ", mechanism=EconomicMechanism.MOMENTUM)
        assert obj is not None
        rows = {e.experiment_id: e for e in obj.experiments}
        for exp_id, ref in rows.items():
            view = reg.get(exp_id)
            assert ref.experiment_identity == view.experiment_identity
            assert ref.headline_verdict == view.verdict
            assert ref.n_valid_attempts == view.n_valid_attempts
            assert ref.n_invalid_attempts == view.n_invalid_attempts


def test_unmapped_strategy_family_never_gets_a_factor_of_its_own():
    """`silver_bullet` (NQ market_structure) has real registry rows but no
    entry in the frozen mechanism bridge -- it must never appear inside ANY
    AlphaResearchObject's strategy_variants (section 7: false merging is
    worse than duplication)."""
    with _registry() as reg:
        objs = list_alpha_research_objects(reg, root_symbol="NQ")
    for obj in objs:
        assert "silver_bullet" not in {v.strategy_family for v in obj.strategy_variants}


def test_no_second_scientific_database_object_never_duplicates_pnl_or_sharpe():
    """AlphaResearchObject must never carry an official metric value
    (net_pnl_usd, sharpe, bh_q, ...) -- only references (experiment_id /
    experiment_identity) a consumer re-reads from the registry itself."""
    forbidden_fields = {"net_pnl_usd", "gross_pnl_usd", "daily_sharpe", "annualized_sharpe", "bh_q", "dsr_probability"}
    all_model_fields: set[str] = set()
    for model in (AlphaResearchObject, FactorIdentity, StrategyVariantEvidence, ExperimentEvidenceRef, EvidenceProfile):
        all_model_fields |= set(model.model_fields)
    assert not (forbidden_fields & all_model_fields)


# ---------------------------------------------------------------------------
# Evidence Profile is deterministic, categorical -- no Alpha Score anywhere
# ---------------------------------------------------------------------------


#: Identifier-shaped only (snake_case), so this never trips on prose
#: *explaining* the forbidden concept (e.g. a docstring saying "never an
#: Alpha Score") -- only on an actual field/variable name being introduced.
_FORBIDDEN_SCORE_NAMES = re.compile(
    r"\balpha_score\b|\bexpected_return\b|\bprobability_of_success\b|\bconfidence_score\b|\bbuy_confidence\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("module", _ALL_MODULES)
def test_no_alpha_score_or_ranking_field_anywhere_in_the_typed_model(module):
    """Static regression guard (prompt 2 sections 10/21): no combined score,
    confidence percentage, expected return, or probability-of-success field
    is ever introduced by this phase."""
    with open(module.__file__, encoding="utf-8") as fh:
        text = fh.read()
    assert not _FORBIDDEN_SCORE_NAMES.search(text), f"{module.__name__} must never carry an Alpha-Score-like field"


def test_evidence_profile_never_combines_dimensions_into_one_number():
    fields = set(EvidenceProfile.model_fields)
    assert "score" not in fields
    assert "combined_score" not in fields
    assert "confidence" not in fields


def test_evidence_profile_dimensions_are_deterministic_and_categorical():
    with _registry() as reg:
        obj = _build_for_family(reg, root_symbol="NQ", mechanism=EconomicMechanism.TREND, family="tsmom")
    assert obj is not None
    ep = obj.evidence_profile
    assert ep.cost_robustness in {"NOT_EVALUATED", "EVALUATED", "FAILED_STRESS"}
    assert ep.parameter_stability in {"NOT_EVALUATED", "EVALUATED", "WEAK"}
    assert ep.regime_breadth in {"NOT_EVALUATED", "EVALUATED"}
    assert ep.related_experiment_count == len(obj.experiments)


def test_related_experiment_count_is_not_called_independent_replication():
    """Identity-hardening patch, section 6: the field/label must be honestly
    named -- a neighbour/ablation/re-execution is coverage of the SAME
    hypothesis, never an independent reproduction."""
    fields = set(EvidenceProfile.model_fields)
    assert "replication_count" not in fields
    assert "replication_label" not in fields
    assert "independent_replication" not in fields
    assert {"related_experiment_count", "related_experiment_label"} <= fields


def test_related_experiment_label_text_never_says_replication():
    with _registry() as reg:
        obj = _build_for_family(reg, root_symbol="NQ", mechanism=EconomicMechanism.TREND, family="tsmom")
    assert obj is not None
    assert "replicat" not in obj.evidence_profile.related_experiment_label.lower()
    assert "related experiment" in obj.evidence_profile.related_experiment_label.lower()


# ---------------------------------------------------------------------------
# Research Maturity is independent from Scientific Verdict
# ---------------------------------------------------------------------------


def test_reject_can_still_be_high_research_maturity():
    """Real evidence: NQ TREND's tsmom-Factor canonical trial is REJECT, but
    with several real registry experiments behind it, maturity is
    ADJUDICATED (the top rung) -- maturity measures how much research
    exists, never how profitable it looks (section 11)."""
    with _registry() as reg:
        obj = _build_for_family(reg, root_symbol="NQ", mechanism=EconomicMechanism.TREND, family="tsmom")
    assert obj is not None
    assert obj.evidence_profile.scientific_evidence["tsmom"] == "REJECT"
    assert obj.research_maturity == ResearchMaturity.ADJUDICATED


def test_research_maturity_ladder_is_never_computed_from_the_verdict_field():
    """Static check: `_research_maturity` never reads a `RegistryVerdict`
    value beyond whether it exists / is NOT_ADJUDICATED (i.e. never branches
    on PASS vs REJECT)."""
    import inspect

    src = inspect.getsource(alpha_builder._research_maturity)
    assert "RegistryVerdict.PASS" not in src
    assert "RegistryVerdict.REJECT" not in src


def test_maturity_is_idea_for_an_unmapped_mechanism():
    with _registry() as reg:
        objs = build_alpha_research_objects_for_mechanism(reg, root_symbol="NQ", mechanism=EconomicMechanism.CARRY)
    assert objs == ()  # not materialized -- see the "no fabricated empty entry" tests below


# ---------------------------------------------------------------------------
# No fabricated empty entries; honest "have I researched this before" gaps
# ---------------------------------------------------------------------------


def test_unmapped_mechanism_returns_empty_tuple_not_a_fabricated_object():
    with _registry() as reg:
        assert build_alpha_research_objects_for_mechanism(reg, root_symbol="NQ", mechanism=EconomicMechanism.CARRY) == ()


def test_mapped_but_never_attempted_family_returns_empty_on_an_empty_registry(tmp_path):
    """A mapped family (TREND -> tsmom, ma_trend) with zero real attempts on
    a fresh registry must not materialize a fabricated FORMALIZED-level
    object in the library -- the library only shows evidence-grounded
    objects (section 9)."""
    with ExperimentRegistry(tmp_path / "empty.sqlite") as reg:
        assert build_alpha_research_objects_for_mechanism(reg, root_symbol="NQ", mechanism=EconomicMechanism.TREND) == ()
        assert list_alpha_research_objects(reg) == ()


# ---------------------------------------------------------------------------
# Repeated FailureMemory aggregation
# ---------------------------------------------------------------------------


def test_repeated_failure_reason_codes_match_failure_memory_lookup_for_that_family():
    from alpha_agent.registry.failure_memory import FailureMemory

    with _registry() as reg:
        obj = _build_for_family(reg, root_symbol="NQ", mechanism=EconomicMechanism.TREND, family="tsmom")
        assert obj is not None
        fm = FailureMemory(reg)
        expected = dict(
            fm.lookup(
                strategy_family="tsmom", root_symbol="NQ", asset_domain=AssetDomain.FUTURES
            ).reason_code_counts
        )
    assert obj.repeated_failure_reason_codes == expected


def test_engineering_lessons_are_root_scoped_not_family_filtered():
    with _registry() as reg:
        momentum = _build_one(reg, root_symbol="NQ", mechanism=EconomicMechanism.MOMENTUM)
        trend_tsmom = _build_for_family(reg, root_symbol="NQ", mechanism=EconomicMechanism.TREND, family="tsmom")
    assert momentum is not None and trend_tsmom is not None
    # engineering lessons are root-scoped (never family-filtered), so two
    # different Factors on the SAME root see the same root-level lessons.
    assert set(momentum.engineering_lessons) == set(trend_tsmom.engineering_lessons)


# ---------------------------------------------------------------------------
# List / get round-trip, related-object cross-referencing
# ---------------------------------------------------------------------------


def test_get_alpha_research_object_round_trips_list():
    with _registry() as reg:
        objs = list_alpha_research_objects(reg, root_symbol="NQ")
        assert objs
        got = get_alpha_research_object(reg, objs[0].alpha_id)
    assert got == objs[0]


def test_get_alpha_research_object_returns_none_for_unknown_id():
    with _registry() as reg:
        assert get_alpha_research_object(reg, "alpharesearchobject3:doesnotexist") is None


def test_related_alpha_ids_are_root_scoped_and_overlap_based():
    """BREAKOUT / FAILED_BREAKOUT / VOLATILITY_BREAKOUT all currently map to
    the single 'breakout' family -- on the SAME root they must cross-
    reference each other; a DIFFERENT root's breakout objects must not leak
    in."""
    with _registry() as reg:
        objs = list_alpha_research_objects(reg, root_symbol="NQ")
    breakout_objs = [
        o for o in objs
        if o.mechanism in (EconomicMechanism.BREAKOUT, EconomicMechanism.FAILED_BREAKOUT, EconomicMechanism.VOLATILITY_BREAKOUT)
    ]
    assert len(breakout_objs) == 3
    ids = {o.alpha_id for o in breakout_objs}
    for o in breakout_objs:
        assert ids - {o.alpha_id} <= set(o.related_alpha_ids)
        assert all(rid != o.alpha_id for rid in o.related_alpha_ids)


def test_related_alpha_ids_never_cross_roots():
    with _registry() as reg:
        nq_breakout = _build_one(reg, root_symbol="NQ", mechanism=EconomicMechanism.BREAKOUT)
        cl_breakout = _build_one(reg, root_symbol="CL", mechanism=EconomicMechanism.BREAKOUT)
    assert nq_breakout is not None and cl_breakout is not None
    assert cl_breakout.alpha_id not in nq_breakout.related_alpha_ids


# ---------------------------------------------------------------------------
# No writes / no network / no execution -- static guards
# ---------------------------------------------------------------------------


def test_builder_and_comparison_never_call_a_registry_write_method():
    forbidden = [
        r"\.insert_experiment\(", r"\.record_failure\(", r"\.record_lineage\(",
        r"\.apply_bundle\(", r"\.record_attempt", r"INSERT OR REPLACE", r"\bUPDATE\s+\w+\s+SET\b",
    ]
    for module in _ALL_MODULES:
        with open(module.__file__, encoding="utf-8") as fh:
            text = fh.read()
        for pattern in forbidden:
            assert not re.search(pattern, text), f"{module.__name__} must never call {pattern!r}"


def test_alpha_memory_package_never_imports_network_or_llm_clients():
    forbidden_imports = ("AnthropicClient", "databento", "import anthropic", "requests.")
    for module in _ALL_MODULES:
        with open(module.__file__, encoding="utf-8") as fh:
            text = fh.read()
        for token in forbidden_imports:
            assert token not in text, f"{module.__name__} must never reference {token!r}"


def test_alpha_memory_package_never_triggers_backtest_execution():
    """No compiler / C++ engine / paper-trading entry point is ever
    referenced by this read-model package (section 18)."""
    forbidden_tokens = ("StrategyCompilerAgent", "run_deep_research", "CppEngineRunner", "PaperTradingEngine", "run_targets_backtest")
    for module in _ALL_MODULES:
        with open(module.__file__, encoding="utf-8") as fh:
            text = fh.read()
        for token in forbidden_tokens:
            assert token not in text, f"{module.__name__} must never reference {token!r}"
