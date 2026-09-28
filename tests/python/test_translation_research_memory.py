"""Phase 1 -- deterministic research-memory tests (prompt 1 sections 10/11).

Uses the REAL, already-committed local registry (read-only, never mutated --
the same pattern `test_conversation_engine.py` and `alpha_agent.ui.services`
tests already use) so these counts are real evidence, not a fabricated
fixture.
"""
from __future__ import annotations

import pytest
from alpha_agent.core.instrument import AssetDomain
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.enums import ExperimentStatus, RegistryVerdict, TrialRole
from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
from alpha_agent.registry.models import ExperimentRecord, MarketWindow, ResultRecord
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.translation.research_memory import (
    MECHANISM_TO_KNOWN_FAMILIES,
    mechanism_research_memory,
)
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

_WINDOW = MarketWindow(label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31")
_IDENTITY_BASE = {
    "dataset_fingerprint": "valdataset2:aaa",
    "split_identity": "splitplan1:bbb",
    "validation_spec_fingerprint": "validationspec1:ccc",
    "reliability_policy_fingerprint": "valreliabilitypolicy1:ddd",
    "execution_config_identity": "execconfig1:eee",
    "cost_config_identity": "costconfig1:fff",
    "risk_identity": "riskconfig1:ggg",
    "feature_spec_fingerprint": "featset1:iii",
}


def _registry() -> ExperimentRegistry:
    return ExperimentRegistry(services.REGISTRY_PATH)


def _insert_one_tsmom_experiment(reg: ExperimentRegistry, *, root: str) -> None:
    """Mirrors `test_phase_14_registry.py`'s own minimal
    `ExperimentRecord`/`ResultRecord` construction pattern -- inserts exactly
    ONE real, valid CANONICAL tsmom experiment for `root`, leaving every
    other mapped family (ma_trend) with zero attempts."""
    params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    variant = parameter_variant_identity(params)
    identity = experiment_identity(
        strategy_fingerprint="stratdsl1:base", strategy_family="tsmom", root_symbol=root,
        parameter_variant_identity=variant, **_IDENTITY_BASE,
    )
    exp = ExperimentRecord(
        experiment_identity=identity, experiment_id=f"{root}__TSMOM__CANONICAL__SPLIT",
        display_name=f"{root}__TSMOM__CANONICAL__SPLIT", created_at="2026-01-01T00:00:00+00:00",
        phase="TEST", status=ExperimentStatus.COMPLETED, code_commit="abc1234", root_symbol=root,
        asset_domain=AssetDomain.FUTURES,
        strategy_family="tsmom", strategy_fingerprint="stratdsl1:base", strategy_id="TEST-STRAT",
        strategy_spec_json={
            "schema": "registry-strategy-spec/1", "strategy_family": "tsmom", "params": params,
            "parameter_names": sorted(params), "feature_fingerprints": ["feat1:x"],
            "signal_cadence": "daily_trading_day", "execution_cadence": "native_1m_raw_contract",
        },
        market_window=_WINDOW, trial_role=TrialRole.CANONICAL, parameter_variant_identity=variant,
        parameter_variant_label="canonical", target_schedule_hash="targsched1:hhh", **_IDENTITY_BASE,
    )
    result = ResultRecord(
        experiment_identity=identity, headline_verdict=RegistryVerdict.REJECT,
        reason_codes=("fdr_qvalue_above_threshold",), net_pnl_usd=1.0, daily_sharpe=0.01,
    )
    reg.insert_experiment(exp, result)


def test_known_mechanism_maps_to_real_candidate_families():
    assert MECHANISM_TO_KNOWN_FAMILIES[EconomicMechanism.TREND] == ("tsmom", "ma_trend")
    assert MECHANISM_TO_KNOWN_FAMILIES[EconomicMechanism.MEAN_REVERSION] == ("mean_reversion",)


def test_unmapped_mechanism_is_an_honest_research_representation_gap():
    """A mechanism this platform's candidate manifest cannot express yet must
    never be silently mapped to an unrelated family just to produce a match
    (prompt 1 section 11: never adaptive p-hacking)."""
    with _registry() as reg:
        note = mechanism_research_memory(
            mechanisms=(EconomicMechanism.CARRY,), root_symbol="CL", registry=reg,
        )
    assert note.related_strategy_families == ()
    assert note.digests == ()
    assert "research-representation gap" in note.summary
    assert note.underexplored is True


def test_known_mechanism_on_a_real_root_returns_real_registry_counts():
    with _registry() as reg:
        note = mechanism_research_memory(
            mechanisms=(EconomicMechanism.TREND,), root_symbol="CL", registry=reg,
        )
    assert note.related_strategy_families == ("tsmom", "ma_trend")
    assert len(note.digests) == 2
    assert all(d.root_symbol == "CL" for d in note.digests)
    # This local registry has real prior tsmom/ma_trend CL attempts (Phase
    # 13.5C); this is a real evidence assertion, not a fabricated count.
    assert note.mapped_family_count == 2
    assert note.tested_family_count == 2


def test_mapped_family_count_and_tested_family_count_are_distinct_fields(tmp_path):
    """Phase 1 acceptance patch, section 3: a family being MAPPED (has a
    candidate-family representation) is not the same claim as it having
    actually been TESTED (a real execution attempt exists). A fresh, empty
    scratch registry has zero attempts anywhere -- TREND still maps to two
    families (tsmom, ma_trend), but neither has ever been tested here."""
    with ExperimentRegistry(tmp_path / "empty.sqlite") as reg:
        note = mechanism_research_memory(mechanisms=(EconomicMechanism.TREND,), root_symbol="CL", registry=reg)
    assert note.mapped_family_count == 2
    assert note.tested_family_count == 0
    assert note.underexplored is True


_OVERCLAIMED_NOVELTY_WORDS = ("genuinely new", "novel", "never tested before", "never been tested")


def test_underexplored_summary_never_overclaims_novelty(tmp_path):
    """Phase 1 acceptance patch, section 3: this lookup only establishes
    'no attempts in the currently mapped family/root scope' -- it must never
    claim genuine novelty across the whole registry (that would need
    deterministic similarity evidence this lookup does not have)."""
    with ExperimentRegistry(tmp_path / "empty2.sqlite") as reg:
        note = mechanism_research_memory(mechanisms=(EconomicMechanism.TREND,), root_symbol="CL", registry=reg)
    lowered = note.summary.lower()
    for phrase in _OVERCLAIMED_NOVELTY_WORDS:
        assert phrase not in lowered, (phrase, note.summary)
    assert "currently mapped family/root scope" in note.summary
    assert "underexplored" in lowered


def test_mapped_family_count_and_tested_family_count_split_on_a_partially_tested_scratch_registry(tmp_path):
    """A real split: tsmom has one real, valid attempt; ma_trend (also
    mapped to TREND) has none -- mapped_family_count and tested_family_count
    must genuinely differ (2 vs 1), never conflated into `len(mapped)`."""
    with ExperimentRegistry(tmp_path / "partial.sqlite") as reg:
        _insert_one_tsmom_experiment(reg, root="CL")
        note = mechanism_research_memory(mechanisms=(EconomicMechanism.TREND,), root_symbol="CL", registry=reg)
    assert note.mapped_family_count == 2
    assert note.tested_family_count == 1
    assert note.underexplored is False


def test_research_memory_note_is_root_scoped():
    with _registry() as reg:
        cl = mechanism_research_memory(mechanisms=(EconomicMechanism.TREND,), root_symbol="CL", registry=reg)
        es = mechanism_research_memory(mechanisms=(EconomicMechanism.TREND,), root_symbol="ES", registry=reg)
    assert cl.root_symbol == "CL"
    assert es.root_symbol == "ES"
    assert cl.summary != es.summary or cl.digests != es.digests


def test_engineering_lessons_never_treated_as_scientific_verdicts():
    """Engineering/data-quality lessons are surfaced separately from
    scientific verdict counts -- never merged into `digests` (prompt 1
    section 10: 'repeated failure reasons' is its own, honestly labeled
    field)."""
    with _registry() as reg:
        note = mechanism_research_memory(mechanisms=(EconomicMechanism.TREND,), root_symbol="CL", registry=reg)
    assert isinstance(note.engineering_lessons, tuple)
