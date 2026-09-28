"""Phase 6 CLOSURE -- Factor Library evidence-oriented status, real ETF
Registry persistence, and research recycling.

Complements (never duplicates) `test_alpha_library_ui.py` (the renamed Phase
2 rendering tests: lives inside Research, never calls Claude/market
data/registry-write, progressive disclosure, no Alpha Score). This file
covers the closure's own new behaviors:

1. Factor status (VALIDATED / UNDER_RESEARCH / RESEARCH_ARCHIVE) is computed
   purely from deterministic evidence -- PASS never fabricated, REJECT never
   collapsed into "the whole Factor is false".
2. The real Phase 6 XLE ETF experiment is persisted as `AssetDomain.ETF`
   through the normal authoritative Registry path and is structurally
   isolated from Futures evidence.
3. Research recycling (`alpha_agent.research_recycling`) reuses the runtime
   orchestrator's own near-duplicate decision and Phase 2's `what_changed`
   to deprioritize an unchanged repeat while never blacklisting a Factor,
   and allows reconsideration on a meaningful change.
4. No Alpha Score / expected-return / success-probability field anywhere in
   the new modules; Factor Library browsing never touches research
   recycling or triggers research.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from alpha_agent.alpha_memory.builder import build_alpha_research_objects_for_mechanism
from alpha_agent.alpha_memory.factor_status import FactorStatus, factor_status
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.enums import AssetDomain, ExperimentStatus, RegistryVerdict, TrialRole
from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
from alpha_agent.registry.models import ExperimentRecord, MarketWindow, ResultRecord
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.research_recycling import RecyclingDecision, assess_research_recycling
from alpha_agent.ui import services

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_REGISTRY = REPO_ROOT / "data" / "registry" / "experiments.sqlite"

WINDOW = MarketWindow(label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31")

_FINGERPRINTS = {
    "dataset_fingerprint": "valdataset2:ds",
    "split_identity": "splitplan1:sp",
    "validation_spec_fingerprint": "validationspec1:vs",
    "reliability_policy_fingerprint": "valreliabilitypolicy1:rp",
    "execution_config_identity": "execconfig1:ex",
    "cost_config_identity": "costconfig1:co",
    "risk_identity": "riskconfig1:ri",
    "feature_spec_fingerprint": "featset1:fs",
}


def _identity(*, root: str, family: str, params: dict) -> str:
    return experiment_identity(
        strategy_fingerprint=f"stratdsl1:{family}:{root}",
        strategy_family=family,
        root_symbol=root,
        parameter_variant_identity=parameter_variant_identity(params),
        **_FINGERPRINTS,
    )


def _experiment(
    *, root: str, family: str, params: dict, asset_domain: AssetDomain = AssetDomain.FUTURES,
    label: str = "canonical",
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_identity=_identity(root=root, family=family, params=params),
        experiment_id=f"{root}__{family.upper()}__{label.upper()}__TEST",
        display_name=f"{root} / {family} / {label} / TEST",
        created_at="2026-01-01T00:00:00+00:00",
        phase="TEST",
        status=ExperimentStatus.COMPLETED,
        root_symbol=root,
        asset_domain=asset_domain,
        strategy_family=family,
        strategy_fingerprint=f"stratdsl1:{family}:{root}",
        strategy_id="TEST-STRAT",
        strategy_spec_json={
            "params": params,
            "signal_cadence": "daily_trading_day",
            "execution_cadence": "native_1m_raw_contract",
        },
        market_window=WINDOW,
        trial_role=TrialRole.CANONICAL,
        parameter_variant_identity=parameter_variant_identity(params),
        parameter_variant_label=label,
        **_FINGERPRINTS,
    )


def _result(identity: str, *, verdict: RegistryVerdict, reason_codes=("fdr_qvalue_above_threshold",)) -> ResultRecord:
    return ResultRecord(
        experiment_identity=identity, headline_verdict=verdict,
        reason_codes=reason_codes, net_pnl_usd=1.0, daily_sharpe=0.01,
    )


@pytest.fixture
def registry(tmp_path):
    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reg:
        yield reg


# ==========================================================================
# 1. Factor status: deterministic, evidence-based, never a boolean
# ==========================================================================


def test_factor_status_is_validated_only_when_a_canonical_trial_genuinely_passed(registry):
    params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    exp = _experiment(root="NQ", family="tsmom", params=params)
    registry.insert_experiment(exp, _result(exp.experiment_identity, verdict=RegistryVerdict.PASS))

    objs = build_alpha_research_objects_for_mechanism(
        registry, root_symbol="NQ", mechanism=EconomicMechanism.TREND,
        family_groups_override=(("tsmom",),),
    )
    assert len(objs) == 1
    assert factor_status(objs[0]) is FactorStatus.VALIDATED


def test_factor_status_is_research_archive_when_every_canonical_trial_rejected_but_factor_not_declared_false(registry):
    params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    exp = _experiment(root="NQ", family="tsmom", params=params)
    registry.insert_experiment(exp, _result(exp.experiment_identity, verdict=RegistryVerdict.REJECT))

    objs = build_alpha_research_objects_for_mechanism(
        registry, root_symbol="NQ", mechanism=EconomicMechanism.TREND,
        family_groups_override=(("tsmom",),),
    )
    assert len(objs) == 1
    obj = objs[0]
    assert factor_status(obj) is FactorStatus.RESEARCH_ARCHIVE
    # REJECT does not mean the entire Factor is scientifically false: the
    # object still carries its full structured evidence, not a boolean.
    assert obj.evidence_profile.scientific_evidence["tsmom"] == "REJECT"
    assert obj.mechanism is EconomicMechanism.TREND  # the Mechanism itself is untouched by this verdict
    assert obj.research_maturity is not None


def test_factor_status_is_under_research_for_inconclusive_or_mixed_evidence(registry):
    params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    exp = _experiment(root="NQ", family="tsmom", params=params)
    registry.insert_experiment(exp, _result(exp.experiment_identity, verdict=RegistryVerdict.INCONCLUSIVE))

    objs = build_alpha_research_objects_for_mechanism(
        registry, root_symbol="NQ", mechanism=EconomicMechanism.TREND,
        family_groups_override=(("tsmom",),),
    )
    assert factor_status(objs[0]) is FactorStatus.UNDER_RESEARCH


def test_a_rejected_strategyspec_does_not_invalidate_a_factor_with_another_passing_variant(registry):
    """Mechanism != Factor != Strategy != Experiment: one REJECTed
    implementation under a Factor must not suppress a genuine PASS recorded
    under a *different* strategy variant of the same Factor group."""
    rej_params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    pass_params = {"fast_horizon": 5, "slow_horizon": 30, "size": 1}
    rej = _experiment(root="NQ", family="tsmom", params=rej_params, label="canonical-a")
    passed = _experiment(root="NQ", family="tsmom", params=pass_params, label="canonical-b")
    registry.insert_experiment(rej, _result(rej.experiment_identity, verdict=RegistryVerdict.REJECT))
    registry.insert_experiment(passed, _result(passed.experiment_identity, verdict=RegistryVerdict.PASS))

    objs = build_alpha_research_objects_for_mechanism(
        registry, root_symbol="NQ", mechanism=EconomicMechanism.TREND,
        family_groups_override=(("tsmom",),),
    )
    assert len(objs) == 1
    # both experiments are on the SAME (root, Factor) -- one family, so this
    # is deliberately a MIXED same-family case; the object status must not
    # be dragged down to RESEARCH_ARCHIVE by the REJECT sibling.
    status = factor_status(objs[0])
    assert status in (FactorStatus.VALIDATED, FactorStatus.UNDER_RESEARCH)
    assert status is not FactorStatus.RESEARCH_ARCHIVE


# ==========================================================================
# 2. Research recycling: evidence, never a blacklist
# ==========================================================================


def test_unchanged_repeat_with_no_novelty_is_deprioritized(registry):
    params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    exp = _experiment(root="NQ", family="tsmom", params=params)
    registry.insert_experiment(exp, _result(exp.experiment_identity, verdict=RegistryVerdict.REJECT))

    assessment = assess_research_recycling(
        registry, mechanism=EconomicMechanism.TREND, strategy_family="tsmom",
        root_symbol="NQ", asset_domain=AssetDomain.FUTURES, params=params,
        signal_cadence="daily_trading_day", execution_cadence="native_1m_raw_contract",
    )
    assert assessment.decision is RecyclingDecision.DEPRIORITIZE_NO_NOVELTY
    assert assessment.nearest_related is not None
    assert assessment.nearest_related.headline_verdict is RegistryVerdict.REJECT
    # the explanation cites concrete prior evidence, not a bare verdict.
    assert exp.experiment_id in assessment.explanation


def test_stated_novelty_allows_reconsideration_despite_a_prior_reject(registry):
    params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    exp = _experiment(root="NQ", family="tsmom", params=params)
    registry.insert_experiment(exp, _result(exp.experiment_identity, verdict=RegistryVerdict.REJECT))

    assessment = assess_research_recycling(
        registry, mechanism=EconomicMechanism.TREND, strategy_family="tsmom",
        root_symbol="NQ", asset_domain=AssetDomain.FUTURES, params=params,
        signal_cadence="daily_trading_day", execution_cadence="native_1m_raw_contract",
        novelty_notes="new CFTC positioning data suggests a regime shift since the prior test",
    )
    assert assessment.decision is RecyclingDecision.RECONSIDER


def test_meaningfully_different_parameters_are_not_deprioritized(registry):
    prior_params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    exp = _experiment(root="NQ", family="tsmom", params=prior_params)
    registry.insert_experiment(exp, _result(exp.experiment_identity, verdict=RegistryVerdict.REJECT))

    far_params = {"fast_horizon": 3, "slow_horizon": 250, "size": 1}
    assessment = assess_research_recycling(
        registry, mechanism=EconomicMechanism.TREND, strategy_family="tsmom",
        root_symbol="NQ", asset_domain=AssetDomain.FUTURES, params=far_params,
        signal_cadence="daily_trading_day", execution_cadence="native_1m_raw_contract",
    )
    assert assessment.decision is RecyclingDecision.RECONSIDER


def test_no_prior_evidence_at_all_is_always_reconsider(registry):
    assessment = assess_research_recycling(
        registry, mechanism=EconomicMechanism.TREND, strategy_family="tsmom",
        root_symbol="ZZ", asset_domain=AssetDomain.FUTURES, params={"fast_horizon": 20, "slow_horizon": 120},
    )
    assert assessment.decision is RecyclingDecision.RECONSIDER
    assert assessment.nearest_related is None


def test_a_prior_pass_is_never_deprioritized(registry):
    """A near-identical PROPOSAL against a prior PASS must never be labeled
    a deprioritized near-duplicate -- only REJECT/INCONCLUSIVE evidence
    triggers the "no novelty" caution; a PASS is duplicate-blocked upstream
    by the registry's own `find_exact_duplicate`, not by this advisory."""
    params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    exp = _experiment(root="NQ", family="tsmom", params=params)
    registry.insert_experiment(exp, _result(exp.experiment_identity, verdict=RegistryVerdict.PASS))

    assessment = assess_research_recycling(
        registry, mechanism=EconomicMechanism.TREND, strategy_family="tsmom",
        root_symbol="NQ", asset_domain=AssetDomain.FUTURES, params=params,
        signal_cadence="daily_trading_day", execution_cadence="native_1m_raw_contract",
    )
    assert assessment.decision is RecyclingDecision.RECONSIDER


def test_research_recycling_schema_carries_no_alpha_score_or_expected_return_field():
    """The assessment's own top-level fields, not its nested (pre-existing,
    already-approved) `nearest_related.similarity.score` -- that structural
    near-duplicate retrieval score is reused verbatim from the runtime
    orchestrator's own `RelatedExperiment`/`SimilarityBreakdown` schema and
    is a transparent, decomposed distance metric, never a combined
    predictive Alpha Score / expected-return / success-probability field."""
    from alpha_agent.research_recycling import ResearchRecyclingAssessment

    field_names = " ".join(ResearchRecyclingAssessment.model_fields.keys()).lower()
    for pattern in ("score", "confidence", "expected_return", "probability"):
        assert pattern not in field_names, f"ResearchRecyclingAssessment must never carry a {pattern!r} field"


def test_research_recycling_never_writes_the_registry():
    import alpha_agent.research_recycling as mod

    forbidden = [
        r"\.insert_experiment\(", r"\.record_failure\(", r"\.record_lineage\(",
        r"\.apply_bundle\(", r"\.record_attempt", r"INSERT OR REPLACE",
    ]
    with open(mod.__file__, encoding="utf-8") as fh:
        text = fh.read()
    for pattern in forbidden:
        assert not re.search(pattern, text), f"research_recycling.py must never call {pattern!r}"


def test_factor_library_never_imports_research_recycling():
    """Passive Factor Library browsing must never run a recycling assessment
    -- that is a proposal-time capability for the Agent, not something
    triggered by viewing existing evidence."""
    from alpha_agent.ui.views import alpha_library

    with open(alpha_library.__file__, encoding="utf-8") as fh:
        text = fh.read()
    assert "research_recycling" not in text


# ==========================================================================
# 3. Real ETF (XLE) Registry persistence, and cross-domain isolation
# ==========================================================================

pytestmark_real_registry = pytest.mark.skipif(
    not REAL_REGISTRY.exists(), reason="no local data/registry/experiments.sqlite in this checkout",
)


@pytestmark_real_registry
def test_real_xle_etf_experiment_is_persisted_as_asset_domain_etf():
    with ExperimentRegistry(REAL_REGISTRY) as reg:
        rows = reg._conn.execute(
            "SELECT experiment_id, asset_domain, root_symbol, strategy_family, status "
            "FROM experiments WHERE asset_domain = 'ETF'"
        ).fetchall()
    assert len(rows) == 1, "expected exactly the one real Phase 6 ETF experiment"
    _experiment_id, asset_domain, root_symbol, strategy_family, status = rows[0]
    assert asset_domain == "ETF"
    assert root_symbol == "XLE"
    assert strategy_family == "etf_tsmom"
    assert status == "COMPLETED"


@pytestmark_real_registry
def test_real_xle_etf_experiment_surfaces_through_alpha_memory_and_factor_library_services():
    objs = services.list_alpha_research_objects(root_symbol="XLE")
    etf_objs = [o for o in objs if o["asset_domain"] == "ETF"]
    assert len(etf_objs) == 1
    obj = etf_objs[0]
    assert obj["root_symbol"] == "XLE"
    assert obj["factor_status"] in (fs.value for fs in FactorStatus)

    detail = services.get_alpha_research_object(obj["alpha_id"])
    assert detail is not None
    assert detail["asset_domain"] == "ETF"
    assert detail["factor_status"] == obj["factor_status"]


@pytestmark_real_registry
def test_etf_evidence_does_not_contaminate_futures_evidence():
    all_objs = services.list_alpha_research_objects()
    assert any(o["asset_domain"] == "ETF" and o["root_symbol"] == "XLE" for o in all_objs)
    # the combined list legitimately contains the one ETF object; the real
    # isolation claim is that no FUTURES-domain object references XLE, and
    # the ETF object's own experiments never appear under a FUTURES query.
    futures_only = [o for o in all_objs if o["asset_domain"] == "FUTURES"]
    assert all(o["root_symbol"] != "XLE" for o in futures_only)

    with ExperimentRegistry(REAL_REGISTRY) as reg:
        related = reg.find_related(
            strategy_family="etf_tsmom", root_symbol="XLE", asset_domain=AssetDomain.FUTURES,
            params={"fast_horizon": 21, "slow_horizon": 120}, top_k=10,
        )
    # find_related returns the top-K nearest candidates WITHIN the requested
    # domain regardless of absolute score (a ranked list, not a threshold
    # filter), so unrelated low-similarity Futures rows legitimately appear
    # here. The real isolation claim is narrower and structural: the
    # ETF/XLE row itself (which does not exist under asset_domain=FUTURES)
    # can never be one of them.
    assert all(r.root_symbol != "XLE" for r in related), (
        "the real ETF/XLE experiment must never surface under a FUTURES-scoped query"
    )


@pytestmark_real_registry
def test_research_recycling_respects_asset_domain_isolation_for_the_real_xle_experiment():
    real_params = {"fast_horizon": 21, "price_field": "close", "size": 1, "slow_horizon": 120}
    with ExperimentRegistry(REAL_REGISTRY) as reg:
        futures_view = assess_research_recycling(
            reg, mechanism=EconomicMechanism.TREND, strategy_family="etf_tsmom",
            root_symbol="XLE", asset_domain=AssetDomain.FUTURES, params=real_params,
        )
        etf_view = assess_research_recycling(
            reg, mechanism=EconomicMechanism.TREND, strategy_family="etf_tsmom",
            root_symbol="XLE", asset_domain=AssetDomain.ETF, params=real_params,
        )
    # a FUTURES-scoped query can legitimately surface unrelated low-similarity
    # Futures rows (find_related is a ranked top-K, not a threshold filter),
    # but the real XLE/ETF row itself must never be among them.
    assert futures_view.nearest_related is None or futures_view.nearest_related.root_symbol != "XLE"
    assert etf_view.nearest_related is not None
    assert etf_view.nearest_related.root_symbol == "XLE"
    assert etf_view.nearest_related.experiment_id == "XLE__ETF_TSMOM__CANONICAL__VALIDATION__285C2E4D47"


# ==========================================================================
# 4. Micro-fix: exact structured repeat (real XLE etf_tsmom, real params)
# ==========================================================================

_REAL_XLE_PARAMS = {"fast_horizon": 21, "price_field": "close", "size": 1, "slow_horizon": 120}


@pytestmark_real_registry
def test_real_xle_exact_repeat_is_deprioritized_no_novelty():
    """Generic similarity alone caps an exact etf_tsmom repeat at 0.70 (no
    declared parameter-grid range for this ETF-specific family), below the
    shared 0.88 near-duplicate threshold -- the exact-structured-repeat
    check must catch this case directly off registry evidence instead."""
    with ExperimentRegistry(REAL_REGISTRY) as reg:
        assessment = assess_research_recycling(
            reg, mechanism=EconomicMechanism.TREND, strategy_family="etf_tsmom",
            root_symbol="XLE", asset_domain=AssetDomain.ETF, params=_REAL_XLE_PARAMS,
        )
    assert assessment.decision is RecyclingDecision.DEPRIORITIZE_NO_NOVELTY
    assert assessment.nearest_related is not None
    assert assessment.nearest_related.experiment_id == "XLE__ETF_TSMOM__CANONICAL__VALIDATION__285C2E4D47"
    assert assessment.nearest_related.headline_verdict is RegistryVerdict.REJECT


@pytestmark_real_registry
def test_real_xle_exact_repeat_with_stated_novelty_is_reconsider():
    """A rejected experiment is evidence, not a blacklist: stated novelty
    always allows reconsideration, even for an exact structured repeat."""
    with ExperimentRegistry(REAL_REGISTRY) as reg:
        assessment = assess_research_recycling(
            reg, mechanism=EconomicMechanism.TREND, strategy_family="etf_tsmom",
            root_symbol="XLE", asset_domain=AssetDomain.ETF, params=_REAL_XLE_PARAMS,
            novelty_notes="new sector-rotation regime evidence since the prior validation window",
        )
    assert assessment.decision is RecyclingDecision.RECONSIDER


@pytestmark_real_registry
def test_futures_scoped_query_does_not_see_the_real_etf_xle_evidence():
    """The exact-structured-repeat check is asset_domain-scoped exactly like
    the generic path -- a FUTURES query for the same family/root/params must
    never surface the real ETF/XLE row."""
    with ExperimentRegistry(REAL_REGISTRY) as reg:
        assessment = assess_research_recycling(
            reg, mechanism=EconomicMechanism.TREND, strategy_family="etf_tsmom",
            root_symbol="XLE", asset_domain=AssetDomain.FUTURES, params=_REAL_XLE_PARAMS,
        )
    assert assessment.nearest_related is None or assessment.nearest_related.root_symbol != "XLE"
