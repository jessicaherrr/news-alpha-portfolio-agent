"""Phase 14 -- experiment registry / failure memory (prompt 14 section 19).

Two layers are tested:

* the registry mechanics (identity, idempotency, conflict, immutability,
  duplicate detection, similarity, supersession) against small synthetic
  records, so the invariants are exercised in isolation;
* the real corrected Phase 13.5C import, against the committed artifacts.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from alpha_agent.core.instrument import AssetDomain
from alpha_agent.registry import (
    IDENTITY_SCHEMA,
    ExperimentConflict,
    ExperimentRegistry,
    FailureClass,
    IdentitySchemaMismatch,
    ImmutableResultError,
    RegistryVerdict,
    RelationType,
    ScheduleProvenanceConflict,
    TrialRole,
)
from alpha_agent.registry.enums import Authority, ExperimentStatus, FailureScope
from alpha_agent.registry.exports import write_all
from alpha_agent.registry.failure_memory import FailureMemory
from alpha_agent.registry.holdout_guard import (
    HOLDOUT_START_NS,
    HoldoutAccessError,
    assert_no_holdout_market_data,
)
from alpha_agent.registry.identity import (
    experiment_identity,
    friendly_experiment_id,
    parameter_variant_identity,
)
from alpha_agent.registry.models import (
    ExperimentRecord,
    FailureRecord,
    ImportBundle,
    LineageEdge,
    MarketWindow,
    ResultRecord,
    SensitivityEvidence,
)
from alpha_agent.registry.phase_13_5c_import import (
    CORRECTED_COMMIT,
    EXPECTED_CANONICAL,
    EXPECTED_NEIGHBOURS,
    EXPECTED_UNIQUE_TRIALS,
    EXPECTED_VERDICTS,
    PHASE,
    PRECORRECTION_PHASE,
    SUPERSEDED_COMMIT,
    Phase135cArtifactError,
    Phase135cSources,
    _config_identities,
    build_bundle,
    import_phase_13_5c,
    reconstruct_variant,
)
from alpha_agent.registry.schema import SCHEMA_VERSION, UnknownSchemaVersion, ensure_schema
from alpha_agent.registry.similarity import (
    declared_param_ranges,
    parameter_distance,
    structural_shape,
)
from alpha_agent.strategy import strategy_fingerprint
from alpha_agent.strategy.candidates_phase_13_5c import (
    feature_fingerprints,
    spec_for_params,
)
from alpha_agent.validation.fingerprint import fingerprint

REPO = Path(__file__).resolve().parents[2]
WINDOW = MarketWindow(label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31")

_BASE = {
    "dataset_fingerprint": "valdataset2:aaa",
    "split_identity": "splitplan1:bbb",
    "validation_spec_fingerprint": "validationspec1:ccc",
    "reliability_policy_fingerprint": "valreliabilitypolicy1:ddd",
    "execution_config_identity": "execconfig1:eee",
    "cost_config_identity": "costconfig1:fff",
    "risk_identity": "riskconfig1:ggg",
    "feature_spec_fingerprint": "featset1:iii",
}


def _identity(**overrides) -> str:
    kwargs = {
        "strategy_fingerprint": "stratdsl1:base",
        "strategy_family": "tsmom",
        "root_symbol": "NQ",
        "parameter_variant_identity": parameter_variant_identity(
            {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
        ),
        **_BASE,
    }
    kwargs.update(overrides)
    return experiment_identity(**kwargs)


def _experiment(
    *,
    identity: str | None = None,
    experiment_id: str = "NQ__TSMOM__CANONICAL__V",
    family: str = "tsmom",
    root: str = "NQ",
    params: dict | None = None,
    strategy_fingerprint: str = "stratdsl1:base",
    role: TrialRole = TrialRole.CANONICAL,
    phase: str = "TEST",
    notes: str = "",
    parent: str | None = None,
    target_schedule_hash: str | None = "targsched1:hhh",
    report_fingerprint: str | None = None,
) -> ExperimentRecord:
    params = params or {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    variant = parameter_variant_identity(params)
    ident = identity or _identity(
        strategy_fingerprint=strategy_fingerprint,
        strategy_family=family,
        root_symbol=root,
        parameter_variant_identity=variant,
    )
    return ExperimentRecord(
        experiment_identity=ident,
        experiment_id=experiment_id,
        display_name=experiment_id,
        created_at="2026-01-01T00:00:00+00:00",
        phase=phase,
        status=ExperimentStatus.COMPLETED,
        code_commit="abc1234",
        root_symbol=root,
        asset_domain=AssetDomain.FUTURES,
        strategy_family=family,
        strategy_fingerprint=strategy_fingerprint,
        strategy_id="TEST-STRAT",
        strategy_spec_json={
            "schema": "registry-strategy-spec/1",
            "strategy_family": family,
            "params": params,
            "parameter_names": sorted(params),
            "feature_fingerprints": ["feat1:x"],
            "signal_cadence": "daily_trading_day",
            "execution_cadence": "native_1m_raw_contract",
        },
        market_window=WINDOW,
        trial_role=role,
        parameter_variant_identity=variant,
        parameter_variant_label="canonical" if role is TrialRole.CANONICAL else "neighbour_0",
        parent_experiment_identity=parent,
        target_schedule_hash=target_schedule_hash,
        report_fingerprint=report_fingerprint,
        notes=notes,
        **_BASE,
    )


def _result(identity: str, *, verdict=RegistryVerdict.REJECT, net=1.0) -> ResultRecord:
    return ResultRecord(
        experiment_identity=identity,
        headline_verdict=verdict,
        reason_codes=("fdr_qvalue_above_threshold",),
        net_pnl_usd=net,
        daily_sharpe=0.01,
    )


@pytest.fixture
def registry(tmp_path):
    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reg:
        yield reg


# --------------------------------------------------------------------------
# A / B -- deterministic identity, and semantic change changes it
# --------------------------------------------------------------------------
def test_A_same_semantic_experiment_same_identity():
    assert _identity() == _identity()
    # cosmetic fields are not part of the identity at all
    a = _experiment(experiment_id="A", notes="one")
    b = _experiment(experiment_id="B", notes="two")
    assert a.experiment_identity == b.experiment_identity


@pytest.mark.parametrize(
    "override",
    [
        {"strategy_fingerprint": "stratdsl1:other"},
        {"strategy_family": "ma_trend"},
        {"root_symbol": "ES"},
        {"parameter_variant_identity": parameter_variant_identity({"fast_horizon": 21})},
        {"dataset_fingerprint": "valdataset2:other"},
        {"split_identity": "splitplan1:other"},
        {"validation_spec_fingerprint": "validationspec1:other"},
        {"reliability_policy_fingerprint": "valreliabilitypolicy1:other"},
        {"execution_config_identity": "execconfig1:other"},
        {"cost_config_identity": "costconfig1:other"},
        {"risk_identity": "riskconfig1:other"},
        {"feature_spec_fingerprint": "featset1:other"},
    ],
)
def test_B_meaningful_change_changes_identity(override):
    assert _identity(**override) != _identity()


def test_B_friendly_id_is_not_the_semantic_identity():
    fid = friendly_experiment_id(
        root_symbol="NQ", strategy_family="tsmom",
        variant_label="canonical", split_label="VALIDATION_2023_2024",
    )
    assert fid == "NQ__TSMOM__CANONICAL__VALIDATION_2023_2024"
    assert not fid.startswith("experiment1:")


# --------------------------------------------------------------------------
# C / D / E -- idempotency, conflict, immutability
# --------------------------------------------------------------------------
def test_C_idempotent_insert_produces_one_row(registry):
    exp = _experiment()
    registry.insert_experiment(exp, _result(exp.experiment_identity))
    registry.insert_experiment(exp, _result(exp.experiment_identity))
    assert registry.count_experiments() == 1


def test_D_conflicting_scientific_payload_fails_loudly(registry):
    exp = _experiment()
    registry.insert_experiment(exp, _result(exp.experiment_identity))
    conflicting = exp.model_copy(update={"strategy_id": "DIFFERENT-STRAT"})
    with pytest.raises(ExperimentConflict):
        registry.insert_experiment(conflicting)


def test_E_completed_result_cannot_be_silently_replaced(registry):
    exp = _experiment()
    registry.insert_experiment(exp, _result(exp.experiment_identity, net=1.0))
    with pytest.raises(ImmutableResultError):
        registry.insert_experiment(
            exp, _result(exp.experiment_identity, verdict=RegistryVerdict.PASS, net=999.0)
        )
    stored = registry.get(exp.experiment_identity)
    assert stored.result.net_pnl_usd == 1.0
    assert stored.verdict is RegistryVerdict.REJECT


def test_E_registry_source_contains_no_insert_or_replace_on_experiments():
    src = (REPO / "python/alpha_agent/registry/sqlite_registry.py").read_text()
    assert "INSERT OR REPLACE" not in src.upper()
    assert "UPDATE experiments" not in src
    assert "UPDATE results" not in src


# --------------------------------------------------------------------------
# F -- exact duplicate query
# --------------------------------------------------------------------------
def test_F_find_exact_duplicate(registry):
    exp = _experiment()
    registry.insert_experiment(exp, _result(exp.experiment_identity))
    hit = registry.find_exact_duplicate(exp.experiment_identity, asset_domain=AssetDomain.FUTURES)
    assert hit.exists
    assert hit.experiment_id == exp.experiment_id
    assert hit.headline_verdict is RegistryVerdict.REJECT
    assert "fdr_qvalue_above_threshold" in hit.reason_codes
    assert hit.authority is Authority.AUTHORITATIVE
    assert registry.find_exact_duplicate(
        _identity(root_symbol="CL"), asset_domain=AssetDomain.FUTURES
    ).exists is False


# --------------------------------------------------------------------------
# G / H -- near-duplicate retrieval and its explanation
# --------------------------------------------------------------------------
def test_G_near_duplicate_ranks_above_unrelated(registry):
    near = _experiment(
        experiment_id="NQ__TSMOM__NEAR", params={"fast_horizon": 20, "slow_horizon": 120,
                                                 "size": 1},
        strategy_fingerprint="stratdsl1:near",
    )
    other_root = _experiment(
        experiment_id="ES__TSMOM__FAR", root="ES",
        params={"fast_horizon": 20, "slow_horizon": 120, "size": 1},
        strategy_fingerprint="stratdsl1:esfar",
    )
    other_family = _experiment(
        experiment_id="NQ__BREAKOUT", family="breakout",
        params={"lookback": 55, "size": 1}, strategy_fingerprint="stratdsl1:bo",
    )
    distant = _experiment(
        experiment_id="NQ__TSMOM__DISTANT",
        params={"fast_horizon": 55, "slow_horizon": 245, "size": 1},
        strategy_fingerprint="stratdsl1:distant",
    )
    for e in (near, other_root, other_family, distant):
        registry.insert_experiment(e, _result(e.experiment_identity))

    hits = registry.find_related(
        strategy_family="tsmom", root_symbol="NQ", asset_domain=AssetDomain.FUTURES,
        params={"fast_horizon": 21, "slow_horizon": 120, "size": 1},
        signal_cadence="daily_trading_day", execution_cadence="native_1m_raw_contract",
    )
    ids = [h.experiment_id for h in hits]
    assert ids[0] == "NQ__TSMOM__NEAR"
    assert ids.index("NQ__TSMOM__DISTANT") < ids.index("ES__TSMOM__FAR")
    assert ids.index("ES__TSMOM__FAR") < ids.index("NQ__BREAKOUT")


def test_H_related_result_carries_transparent_reasons(registry):
    exp = _experiment(experiment_id="NQ__TSMOM__X")
    registry.insert_experiment(exp, _result(exp.experiment_identity))
    hit = registry.find_related(
        strategy_family="tsmom", root_symbol="NQ", asset_domain=AssetDomain.FUTURES,
        params={"fast_horizon": 21, "slow_horizon": 120, "size": 1},
        signal_cadence="daily_trading_day", execution_cadence="native_1m_raw_contract",
    )[0]
    reasons = hit.similarity.explain()
    assert "same_family=tsmom" in reasons
    assert "same_root=NQ" in reasons
    assert "same_structure" in reasons
    assert "parameter_distance=" in reasons
    assert hit.similarity.per_parameter_distance["fast_horizon"] > 0
    assert hit.similarity.per_parameter_distance["slow_horizon"] == 0
    # the weighted components sum to the score -- nothing hidden
    assert hit.similarity.score == pytest.approx(sum(hit.similarity.components.values()))


def test_H_parameter_distance_uses_declared_grid_ranges():
    ranges = declared_param_ranges("tsmom")
    assert ranges["fast_horizon"] == (5.0, 60.0)
    assert ranges["slow_horizon"] == (20.0, 250.0)
    dist, per = parameter_distance(
        "tsmom", {"fast_horizon": 20}, {"fast_horizon": 30}
    )
    assert per["fast_horizon"] == pytest.approx(10 / 55)
    assert dist == pytest.approx(10 / 55)
    # a non-numeric declaration contributes nothing rather than a fabricated 0
    assert parameter_distance("tsmom", {"root_symbol": "NQ"}, {"root_symbol": "ES"}) == (
        None, {}
    )


def test_H_structural_shape_ignores_administrative_root_symbol_key():
    """Agent runtime-integration release regression: two write paths disagree
    about whether `root_symbol` lives inside the stored `params` dict (the
    Phase 14 historical import's `reconstruct_variant` includes it; the
    runtime `StrategyCompilerAgent`'s `resolved_params` deliberately does
    not, since it is tracked separately as the template's own field). That
    administrative difference alone must never zero out `same_structure` for
    two otherwise-identical strategies -- discovered when a fresh
    orchestrator-driven proposal failed to register as a near-duplicate of an
    identical, already-imported historical experiment."""
    with_root = structural_shape(
        strategy_family="tsmom",
        params={"fast_horizon": 20, "slow_horizon": 120, "size": 1, "root_symbol": "NQ"},
        signal_cadence="daily_trading_day", execution_cadence="native_1m_raw_contract",
    )
    without_root = structural_shape(
        strategy_family="tsmom",
        params={"fast_horizon": 20, "slow_horizon": 120, "size": 1},
        signal_cadence="daily_trading_day", execution_cadence="native_1m_raw_contract",
    )
    assert with_root == without_root
    # a REAL structural difference (a different parameter set) still counts.
    different = structural_shape(
        strategy_family="tsmom",
        params={"fast_horizon": 20, "slow_horizon": 120},
        signal_cadence="daily_trading_day", execution_cadence="native_1m_raw_contract",
    )
    assert different != without_root


def test_H_similarity_never_mutates_policy_or_rejects(registry):
    """Near-duplicate retrieval is memory, not adjudication: it returns evidence
    and carries no verdict authority of its own."""
    exp = _experiment()
    registry.insert_experiment(exp, _result(exp.experiment_identity))
    hits = registry.find_related(
        strategy_family="tsmom", root_symbol="NQ", asset_domain=AssetDomain.FUTURES,
        params={"fast_horizon": 20},
    )
    assert hits[0].headline_verdict is RegistryVerdict.REJECT
    # unchanged stored result -- retrieval wrote nothing
    assert registry.get(exp.experiment_identity).result.net_pnl_usd == 1.0


# --------------------------------------------------------------------------
# I -- supersession
# --------------------------------------------------------------------------
def test_I_supersession_preserves_history_and_resolves_authoritative(registry):
    old = _experiment(experiment_id="OLD", identity=_identity(
        execution_config_identity="execconfig1:defective"))
    new = _experiment(experiment_id="NEW")
    registry.insert_experiment(old, _result(old.experiment_identity, net=5.0))
    registry.insert_experiment(new, _result(new.experiment_identity, net=7.0))
    registry.record_lineage(LineageEdge(
        source_experiment_identity=new.experiment_identity,
        target_experiment_identity=old.experiment_identity,
        relation_type=RelationType.SUPERSEDES,
    ))

    assert registry.authority_of(old.experiment_identity) is Authority.SUPERSEDED
    assert registry.authority_of(new.experiment_identity) is Authority.AUTHORITATIVE
    assert registry.resolve_authoritative(
        old.experiment_identity
    ).experiment_id == "NEW"

    only_auth = registry.experiments(authoritative_only=True)
    assert [v.experiment_id for v in only_auth] == ["NEW"]
    both = registry.experiments(include_superseded=True)
    assert {v.experiment_id for v in both} == {"OLD", "NEW"}
    # the historical row is intact, not deleted or rewritten
    assert registry.get("OLD").result.net_pnl_usd == 5.0


# --------------------------------------------------------------------------
# holdout guard (used by O below, checked in isolation here)
# --------------------------------------------------------------------------
def test_holdout_guard_rejects_2025_values():
    with pytest.raises(HoldoutAccessError):
        assert_no_holdout_market_data({"oos_window": "2025-01-01..2025-12-31"})
    with pytest.raises(HoldoutAccessError):
        assert_no_holdout_market_data({"ts": HOLDOUT_START_NS})
    # in-range market data and non-timestamp integers pass
    assert_no_holdout_market_data(
        {"window": "2023-01-01..2024-12-31", "ts": HOLDOUT_START_NS - 1, "n_trades": 31}
    )


def test_holdout_guard_blocks_a_2025_experiment(registry):
    bad = _experiment().model_copy(update={
        "market_window": MarketWindow(
            label="VALIDATION", start_date="2024-01-01", end_date="2025-06-30"
        )
    })
    with pytest.raises(HoldoutAccessError):
        registry.insert_experiment(bad)
    assert registry.count_experiments() == 0


# --------------------------------------------------------------------------
# schema / migrations
# --------------------------------------------------------------------------
def test_schema_version_is_recorded(registry):
    assert registry.schema_version == SCHEMA_VERSION
    assert registry.on_disk_schema_version() == SCHEMA_VERSION


def test_unknown_schema_version_fails_loudly(tmp_path):
    path = tmp_path / "future.sqlite"
    conn = sqlite3.connect(path)
    ensure_schema(conn)
    conn.execute("UPDATE registry_meta SET value='999' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(UnknownSchemaVersion):
        ExperimentRegistry(path)


def test_legacy_v1_database_is_migrated_not_dropped(tmp_path):
    path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE experiments (experiment_id TEXT PRIMARY KEY, status TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO experiments VALUES ('E1', 'REJECTED')")
    conn.commit()
    conn.close()
    with ExperimentRegistry(path) as reg:
        assert reg.schema_version == SCHEMA_VERSION
        rows = reg._conn.execute("SELECT * FROM experiments_v1_legacy").fetchall()
    assert [tuple(r) for r in rows] == [("E1", "REJECTED")]


# --------------------------------------------------------------------------
# transactional integrity
# --------------------------------------------------------------------------
def test_failed_import_leaves_no_half_populated_state(registry, monkeypatch):
    good = _experiment(experiment_id="GOOD")
    bundle = ImportBundle(
        import_id="imp1", phase="TEST", source_fingerprint="src1",
        created_at="2026-01-01T00:00:00+00:00",
        experiments=(good,), results=(_result(good.experiment_identity),),
    )
    boom = RuntimeError("simulated mid-import failure")

    def explode(self, s):
        raise boom

    monkeypatch.setattr(ExperimentRegistry, "_record_sensitivity", explode)
    bundle = bundle.model_copy(update={"sensitivity": (
        SensitivityEvidence(
            evidence_id="S1", experiment_identity=good.experiment_identity,
            kind="k", baseline_verdict=RegistryVerdict.REJECT,
            rerun_verdict=RegistryVerdict.REJECT, verdict_changed=False,
        ),
    )})
    with pytest.raises(RuntimeError):
        registry.apply_bundle(bundle)
    assert registry.count_experiments() == 0
    assert registry.imports() == ()


# ==========================================================================
# The real corrected Phase 13.5C import
# ==========================================================================
@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    path = tmp_path_factory.mktemp("phase14") / "experiments.sqlite"
    reg = ExperimentRegistry(path)
    bundle, counts = import_phase_13_5c(reg, REPO)
    yield reg, bundle, counts
    reg.close()


# -- J / L / M / N: the count distinctions -----------------------------
def test_J_107_unique_statistical_hypotheses_imported_once(imported):
    reg, _, _ = imported
    s = reg.summary()
    assert s.authoritative_statistical_hypotheses == EXPECTED_UNIQUE_TRIALS == 107
    identities = {
        v.experiment_identity for v in reg.experiments(authoritative_only=True)
    }
    assert len(identities) == 107


def test_J_importer_is_idempotent(tmp_path):
    with ExperimentRegistry(tmp_path / "twice.sqlite") as reg:
        import_phase_13_5c(reg, REPO)
        first = reg.summary()
        import_phase_13_5c(reg, REPO)
        second = reg.summary()
    assert first.content_digest == second.content_digest
    assert first.total_experiment_rows == second.total_experiment_rows
    assert first.failure_records == second.failure_records
    assert first.lineage_edges == second.lineage_edges
    assert second.authoritative_statistical_hypotheses == EXPECTED_UNIQUE_TRIALS


def test_K_corrected_canonical_verdict_counts(imported):
    reg, _, _ = imported
    assert reg.summary().canonical_verdict_counts == EXPECTED_VERDICTS
    assert EXPECTED_VERDICTS == {"PASS": 0, "REJECT": 19, "INCONCLUSIVE": 2}
    assert reg.summary().holdout_eligible == 0
    assert reg.experiments(verdict=RegistryVerdict.PASS) == ()


def test_K_canonical_count_is_21(imported):
    reg, _, _ = imported
    canonical = reg.experiments(trial_role=TrialRole.CANONICAL)
    assert len(canonical) == EXPECTED_CANONICAL == 21


def test_L_neighbour_count_is_86(imported):
    reg, _, _ = imported
    neighbours = reg.experiments(trial_role=TrialRole.NEIGHBOUR)
    assert len(neighbours) == EXPECTED_NEIGHBOURS == 86
    for v in neighbours:
        assert v.verdict is RegistryVerdict.NOT_ADJUDICATED
        assert v.experiment.parent_experiment_identity is not None


def test_M_cross_market_aggregation_creates_no_experiments(imported):
    reg, bundle, _ = imported
    evidence = reg.cross_market_evidence()
    assert len(evidence) == 5
    assert {e.strategy_family for e in evidence} == {
        "breakout", "ma_trend", "mean_reversion", "silver_bullet", "tsmom"
    }
    # they reference existing canonical trials rather than adding new ones
    canonical_ids = {
        v.experiment_identity for v in reg.experiments(trial_role=TrialRole.CANONICAL)
    }
    for e in evidence:
        assert set(e.referenced_experiment_identities) <= canonical_ids
    assert reg.summary().authoritative_statistical_hypotheses == EXPECTED_UNIQUE_TRIALS
    assert all(e.experiment_id for e in bundle.experiments)


def test_N_sensitivity_reruns_do_not_raise_the_hypothesis_count(imported):
    reg, _, _ = imported
    sens = reg.sensitivity_evidence()
    assert len(sens) == 21
    for s in sens:
        assert s.in_bh_fdr_denominator is False
        assert s.relation_type is RelationType.SAME_HYPOTHESIS_SENSITIVITY_OF
        assert s.verdict_changed is False
    assert reg.summary().authoritative_statistical_hypotheses == EXPECTED_UNIQUE_TRIALS


def test_N_count_distinctions_are_recorded_separately(imported):
    reg, bundle, _ = imported
    s = reg.summary()
    meta = bundle.metadata
    assert meta["unique_statistical_trial_count"] == 107
    assert meta["completed_cpp_execution_count"] == 528
    assert meta["canonical_trial_count"] == 21
    assert meta["neighbour_trial_count"] == 86
    assert meta["data_quality_sensitivity_reruns"] == 21
    assert meta["data_quality_sensitivity_in_bh_denominator"] is False
    assert meta["cross_market_summaries"] == 5
    assert s.completed_cpp_executions == 528 != s.authoritative_statistical_hypotheses


# -- O / P: holdout and offline ---------------------------------------
def test_O_no_2025_market_data_anywhere_in_the_import(imported):
    reg, bundle, _ = imported
    for exp in bundle.experiments:
        exp.assert_no_holdout()
        assert exp.market_window.end_date == "2024-12-31"
    for res in bundle.results:
        res.assert_no_holdout()
    for f in bundle.failures:
        f.assert_no_holdout()
    for s in bundle.sensitivity:
        assert_no_holdout_market_data(s.metrics)
    # the only mention of the holdout is the declarative never-accessed label
    assert bundle.metadata["holdout_2025_never_accessed"] is True
    assert "NEVER" in bundle.metadata["locked_holdout_2025"]
    for row in reg._conn.execute("SELECT record_json FROM attempt_results"):
        assert_no_holdout_market_data(json.loads(row[0]))


#: modules that could reach the network or a paid vendor endpoint
_FORBIDDEN_MODULES = {
    "databento", "httpx", "requests", "urllib", "socket", "http", "ftplib",
    "aiohttp", "boto3",
}
#: vendor call names that would incur data spend
_FORBIDDEN_CALLS = {"get_cost", "get_range", "Historical", "timeseries"}


def _phase_14_source_files() -> list[Path]:
    files = sorted((REPO / "python/alpha_agent/registry").glob("*.py"))
    files.append(REPO / "scripts/phase_14_import_phase_13_5c.py")
    files.append(REPO / "scripts/phase_14_registry.py")
    return files


def test_P_phase_14_code_contains_no_network_or_acquisition_path():
    """Parsed, not grepped: a prose mention of ``metadata.get_cost`` in a
    docstring is documentation; an *import* or *call* would be an acquisition
    path."""
    for path in _phase_14_source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in _FORBIDDEN_MODULES, f"{path.name} imports {root}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in _FORBIDDEN_MODULES, f"{path.name} imports {root}"
            elif isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name)
                    else ""
                )
                assert name not in _FORBIDDEN_CALLS, f"{path.name} calls {name}"
                if name == "__import__" and node.args:
                    arg = node.args[0]
                    assert not (
                        isinstance(arg, ast.Constant)
                        and str(arg.value).split(".")[0] in _FORBIDDEN_MODULES
                    ), f"{path.name} dynamically imports a network module"


def test_P_import_runs_without_a_databento_client(imported):
    _, bundle, _ = imported
    assert bundle.metadata["no_new_databento_spend"] is True


# -- Q / R: failure memory + superseded checkpoint ---------------------
def test_Q_roll_data_failure_is_queryable(imported):
    reg, _, _ = imported
    rolls = reg.failures(failure_class=FailureClass.ROLL_DATA_FAILURE)
    assert len(rolls) == 1
    f = rolls[0]
    assert f.scope is FailureScope.SYSTEM
    assert "outgoing" in f.summary.lower()
    assert "REJECTED" in f.evidence["rejected_solution"]
    assert "auxiliary roll-close marks" in f.evidence["accepted_correction"]
    assert any("never MarketEvents" in inv for inv in f.evidence["invariants"])
    assert f.resolved is True and f.resolution_commit


def test_Q_zn_economics_failure_is_queryable(imported):
    reg, _, _ = imported
    econ = reg.failures(failure_class=FailureClass.CONTRACT_ECONOMICS_FAILURE)
    system = [f for f in econ if f.scope is FailureScope.SYSTEM]
    assert len(system) == 1
    f = system[0]
    assert f.evidence["defective_economics"]["point_value_usd"] == 0.064
    assert f.evidence["defective_economics"]["tick_value_usd"] == 0.001
    assert f.evidence["corrected_economics"]["point_value_usd"] == 1000.0
    assert f.evidence["corrected_economics"]["tick_value_usd"] == 15.625
    assert f.evidence["corrected_economics"]["price_scale"] == 0.01
    assert f.evidence["n_affected_of_107"] == 20
    assert f.evidence["no_root_specific_hardcoding"] is True
    assert f.resolution_commit == CORRECTED_COMMIT
    # and it is attached to the 4 defective ZN canonical experiments too
    zn = [f for f in econ if f.scope is FailureScope.EXPERIMENT]
    assert len(zn) == 4
    assert {f.root_symbol for f in zn} == {"ZN"}


def test_Q_failure_memory_lookup_returns_structured_evidence(imported):
    reg, _, _ = imported
    resp = FailureMemory(reg).lookup(
        strategy_family="mean_reversion", root_symbol="ES", asset_domain=AssetDomain.FUTURES,
        strategy_spec={"params": {"zscore_window": 20, "entry_z": 2.0, "exit_z": 0.5,
                                  "size": 1}},
        signal_cadence="daily_trading_day",
        execution_cadence="native_1m_raw_contract",
    )
    assert resp.markets_tested == ("ES",)
    assert resp.verdict_counts["REJECT"] >= 1
    assert "fdr_qvalue_above_threshold" in resp.reason_code_counts
    assert len(resp.parameter_variants_tested) == 7      # canonical + 6 neighbours
    assert resp.related_experiments[0].root_symbol == "ES"
    assert resp.related_experiments[0].strategy_family == "mean_reversion"
    assert any("fractional" in lesson.lower() or "roll" in lesson.lower()
               for lesson in resp.lessons)
    # consumable without parsing Markdown
    assert json.loads(resp.model_dump_json())["schema_version"]


def test_R_superseded_checkpoint_is_discoverable_but_not_authoritative(imported):
    reg, _, _ = imported
    superseded = reg.experiments(phase=PRECORRECTION_PHASE, include_superseded=True)
    assert len(superseded) == 21
    for v in superseded:
        assert v.authority is Authority.SUPERSEDED
        assert v.experiment.code_commit == SUPERSEDED_COMMIT
        assert v.superseded_by
        auth = reg.resolve_authoritative(v.experiment_identity)
        assert auth.experiment.code_commit == CORRECTED_COMMIT
        assert auth.experiment.phase == PHASE
        assert auth.authority is Authority.AUTHORITATIVE

    # the corrected lineage is what a default query returns
    assert all(
        v.experiment.phase == PHASE
        for v in reg.experiments(authoritative_only=True)
    )
    # nothing was deleted
    assert reg.summary().total_experiment_rows == 128


def test_R_zn_correction_history_is_traceable(imported):
    reg, _, _ = imported
    corrected = reg.get("ZN__TSMOM__CANONICAL__VALIDATION_2023_2024")
    old = reg.get("ZN__TSMOM__CANONICAL__VALIDATION_2023_2024__PRECORRECTION_52222C3")
    relations = {
        e.relation_type
        for e in reg.lineage_edges(old.experiment_identity)
        if e.source_experiment_identity == corrected.experiment_identity
    }
    assert relations == {RelationType.SUPERSEDES, RelationType.CORRECTS}
    # the defective numbers are preserved, not overwritten
    assert old.result.dsr_probability == 0.0
    assert corrected.result.dsr_probability != 0.0
    assert old.authority is Authority.SUPERSEDED


def test_R_superseded_result_failures_recorded_for_every_canonical_trial(imported):
    reg, _, _ = imported
    sup = reg.failures(failure_class=FailureClass.SUPERSEDED_RESULT)
    system = [f for f in sup if f.scope is FailureScope.SYSTEM]
    per_experiment = [f for f in sup if f.scope is FailureScope.EXPERIMENT]
    assert len(system) == 1
    assert system[0].evidence["superseded_commit"] == SUPERSEDED_COMMIT
    assert system[0].evidence["corrected_commit"] == CORRECTED_COMMIT
    assert len(per_experiment) == 21


# -- imported data fidelity -------------------------------------------
def test_imported_canonical_metrics_match_the_validation_reports(imported):
    reg, _, _ = imported
    report = json.loads(
        (REPO / "outputs/phase_13_5c/ES__TSMOM__validation_report.json").read_text()
    )
    v = reg.get("ES__TSMOM__CANONICAL__VALIDATION_2023_2024")
    assert v.result.net_pnl_usd == report["oos_metrics"]["net_pnl_usd"]
    assert v.result.gross_pnl_usd == report["oos_metrics"]["gross_pnl_usd"]
    assert v.result.costs_usd == report["oos_metrics"]["costs_usd"]
    assert v.result.daily_sharpe == report["oos_metrics"]["daily_sharpe"]
    assert v.result.bh_q == report["headline_verdict_global_family"]["bh_q_value_over_107"]
    assert v.result.dsr_probability == (
        report["headline_verdict_global_family"][
            "dsr_probability_over_107_effective_trials"
        ]
    )
    assert v.experiment.report_fingerprint == report["fingerprints"]["report_fingerprint"]
    assert v.result.fold_consistency == report["walk_forward"]["fold_consistency"]


def test_every_canonical_rejection_has_typed_failure_records(imported):
    reg, _, _ = imported
    for v in reg.experiments(trial_role=TrialRole.CANONICAL, phase=PHASE):
        failures = reg.failures(experiment_identity=v.experiment_identity)
        assert failures, f"{v.experiment_id} has no failure record"
        classes = {f.failure_class for f in failures}
        assert FailureClass.SUPERSEDED_RESULT not in classes
        for f in failures:
            assert f.failure_code   # typed code, not free prose alone
            assert f.summary


def test_failure_classes_cover_the_required_taxonomy():
    required = {
        "SCIENTIFIC_REJECTION", "STATISTICAL_INCONCLUSIVE", "INSUFFICIENT_TRADES",
        "NULL_NOT_REJECTED", "FDR_NOT_PASSED", "DSR_NOT_PASSED",
        "PARAMETER_INSTABILITY", "COST_FRAGILITY", "CROSS_MARKET_WEAKNESS",
        "DATA_QUALITY_FAILURE", "CONTRACT_ECONOMICS_FAILURE", "ROLL_DATA_FAILURE",
        "EXECUTION_INTEGRITY_FAILURE", "SOFTWARE_FAILURE", "SUPERSEDED_RESULT",
    }
    assert required <= {c.value for c in FailureClass}


def test_relation_types_cover_the_required_set():
    required = {
        "SUPERSEDES", "CORRECTS", "REPRODUCES", "PARAMETER_NEIGHBOUR_OF",
        "SAME_HYPOTHESIS_SENSITIVITY_OF",
    }
    assert required == {r.value for r in RelationType}


def test_a_reproduction_is_a_typed_relation_not_a_silent_rerun(registry):
    """REPRODUCES exists so an independent re-run is recorded as such rather
    than colliding with, or silently replacing, the original."""
    original = _experiment(experiment_id="ORIG")
    repro = _experiment(
        experiment_id="REPRO",
        identity=_identity(execution_config_identity="execconfig1:rebuilt"),
    )
    registry.insert_experiment(original, _result(original.experiment_identity))
    registry.insert_experiment(repro, _result(repro.experiment_identity))
    registry.record_lineage(LineageEdge(
        source_experiment_identity=repro.experiment_identity,
        target_experiment_identity=original.experiment_identity,
        relation_type=RelationType.REPRODUCES,
    ))
    # REPRODUCES does not supersede
    assert registry.authority_of(original.experiment_identity) is Authority.AUTHORITATIVE
    assert len(registry.experiments(authoritative_only=True)) == 2


def test_system_failure_records_have_no_experiment_scope(registry):
    f = FailureRecord(
        failure_id="F__SYSTEM__X", scope=FailureScope.SYSTEM,
        failure_class=FailureClass.SOFTWARE_FAILURE, failure_code="x",
        summary="s", mechanism="m", created_at="2026-01-01T00:00:00+00:00",
    )
    registry.record_failure(f)
    assert registry.failures(failure_class=FailureClass.SOFTWARE_FAILURE)[0].summary == "s"


# -- exports ------------------------------------------------------------
def test_exports_are_deterministic_and_friendly_named(imported, tmp_path):
    reg, _, _ = imported
    a, b = tmp_path / "a", tmp_path / "b"
    written = write_all(reg, a)
    write_all(reg, b)
    assert set(written) == {
        "EXPERIMENT_INDEX.csv", "FAILURE_MEMORY.csv", "SUPERSESSION_GRAPH.json",
        "REGISTRY_SUMMARY.json", "REGISTRY_REPORT.md",
    }
    for name in written:
        assert (a / name).read_bytes() == (b / name).read_bytes()

    index = (a / "EXPERIMENT_INDEX.csv").read_text().splitlines()
    assert len(index) == 129                      # header + 128 rows
    assert "NQ__TSMOM__CANONICAL__VALIDATION_2023_2024" in index[0] or any(
        "NQ__TSMOM__CANONICAL__VALIDATION_2023_2024" in line for line in index
    )
    graph = json.loads((a / "SUPERSESSION_GRAPH.json").read_text())
    assert graph["n_superseded"] == 21
    report = (a / "REGISTRY_REPORT.md").read_text()
    for expected in ("Experiment Registry", "append", "PASS = 0", "REJECT = 19",
                     "INCONCLUSIVE = 2", "ROLL_DATA_FAILURE",
                     "CONTRACT_ECONOMICS_FAILURE", "107", "528"):
        assert expected in report


def test_build_bundle_is_deterministic():
    a = build_bundle(REPO)
    b = build_bundle(REPO)
    assert a.model_dump_json() == b.model_dump_json()
    assert a.import_id == b.import_id
    assert a.source_fingerprint == b.source_fingerprint


# ==========================================================================
# Phase 14.1 -- identity schema v2 and neighbour provenance
# ==========================================================================
def test_14_1_I_changing_only_the_schedule_hash_does_not_change_identity():
    """The Phase 14.1 blocker in one assertion: a schedule hash is compiled
    AFTER the hypothesis is chosen, so it cannot be part of what makes an
    experiment that experiment."""
    a = _experiment(target_schedule_hash="targsched1:aaa")
    b = _experiment(target_schedule_hash="targsched1:bbb")
    c = _experiment(target_schedule_hash=None)
    assert a.experiment_identity == b.experiment_identity == c.experiment_identity
    # ...and neither does the report fingerprint
    d = _experiment(report_fingerprint="validationreport1:zzz")
    assert d.experiment_identity == a.experiment_identity


def test_14_1_identity_is_computable_before_the_run():
    """Identity takes only pre-run semantic inputs -- no schedule, no report."""
    import inspect

    params = set(inspect.signature(experiment_identity).parameters)
    assert "target_schedule_hash" not in params
    assert "report_fingerprint" not in params
    assert params == {
        "strategy_fingerprint", "strategy_family", "root_symbol",
        "parameter_variant_identity", "dataset_fingerprint", "split_identity",
        "validation_spec_fingerprint", "reliability_policy_fingerprint",
        "execution_config_identity", "cost_config_identity", "risk_identity",
        "feature_spec_fingerprint",
    }
    assert IDENTITY_SCHEMA == "experiment-identity/2"


def test_14_1_J_conflicting_schedule_hash_fails_loudly(registry):
    """Same semantic inputs, different compiled schedule -> integrity error,
    NOT a second experiment."""
    first = _experiment(target_schedule_hash="targsched1:aaa")
    registry.insert_experiment(first, _result(first.experiment_identity))
    second = _experiment(target_schedule_hash="targsched1:bbb")
    assert second.experiment_identity == first.experiment_identity
    with pytest.raises(ScheduleProvenanceConflict) as exc:
        registry.insert_experiment(second)
    assert "targsched1:aaa" in str(exc.value)
    assert "targsched1:bbb" in str(exc.value)
    # a ScheduleProvenanceConflict is still a conflicting payload
    assert isinstance(exc.value, ExperimentConflict)
    assert registry.count_experiments() == 1
    assert registry.get(first.experiment_identity).experiment.target_schedule_hash == (
        "targsched1:aaa"
    )


def test_14_1_J_provenance_is_not_backfilled_in_place(registry):
    exp = _experiment(target_schedule_hash=None)
    registry.insert_experiment(exp, _result(exp.experiment_identity))
    with pytest.raises(ScheduleProvenanceConflict):
        registry.insert_experiment(_experiment(target_schedule_hash="targsched1:new"))
    assert registry.get(exp.experiment_identity).experiment.target_schedule_hash is None


def test_14_1_identity_schema_mismatch_is_refused(registry):
    stale = _experiment().model_copy(update={"identity_schema": "experiment-identity/1"})
    with pytest.raises(IdentitySchemaMismatch):
        registry.insert_experiment(stale)
    assert registry.count_experiments() == 0


def test_14_1_migration_preserves_the_v2_representation(tmp_path):
    """A v2 database migrates through the chain to the current version, keeping
    the whole old representation rather than dropping it."""
    path = tmp_path / "v2.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE experiments (experiment_identity TEXT PRIMARY KEY,"
        " experiment_id TEXT, content_fingerprint TEXT);"
        "CREATE TABLE results (experiment_identity TEXT PRIMARY KEY);"
        "INSERT INTO registry_meta VALUES ('schema_version', '2');"
        "INSERT INTO experiments VALUES ('experiment1:old', 'OLD_V2_ROW', 'fp');"
    )
    conn.commit()
    conn.close()

    with ExperimentRegistry(path) as reg:
        assert reg.schema_version == SCHEMA_VERSION
        preserved = reg._conn.execute(
            "SELECT experiment_id FROM experiments_v2_legacy"
        ).fetchall()
        assert [r[0] for r in preserved] == ["OLD_V2_ROW"]
        # the current table exists, is empty, and carries the Phase 14.1 column
        cols = {
            r[1] for r in reg._conn.execute("PRAGMA table_info(experiments)").fetchall()
        }
        assert "identity_schema" in cols
        assert reg.count_experiments() == 0


def test_14_1_A_canonical_records_carry_their_supported_provenance(imported):
    reg, _, _ = imported
    canonical = reg.experiments(trial_role=TrialRole.CANONICAL, phase=PHASE)
    assert len(canonical) == 21
    for v in canonical:
        report = json.loads(
            (
                REPO / "outputs/phase_13_5c"
                / f"{v.experiment.root_symbol}__"
                  f"{v.experiment.strategy_family.upper()}__validation_report.json"
            ).read_text()
        )
        fp = report["fingerprints"]
        assert v.experiment.target_schedule_hash == fp["target_schedule_hash"]
        assert v.experiment.report_fingerprint == fp["report_fingerprint"]
        assert v.result.evidence_completeness == "full"


def test_14_1_B_no_neighbour_schedule_hash_is_fabricated(imported):
    reg, _, _ = imported
    neighbours = reg.experiments(trial_role=TrialRole.NEIGHBOUR)
    assert len(neighbours) == 86
    with_hash = [v for v in neighbours if v.experiment.target_schedule_hash]
    assert with_hash == [], (
        "no committed Phase 13.5C artifact records a neighbour's headline-window "
        "schedule hash"
    )


def test_14_1_C_no_neighbour_report_fingerprint_is_fabricated(imported):
    reg, _, _ = imported
    for v in reg.experiments(trial_role=TrialRole.NEIGHBOUR):
        assert v.experiment.report_fingerprint is None


def test_14_1_D_neighbour_evidence_points_at_the_real_source(imported):
    reg, _, _ = imported
    expected_sha = hashlib.sha256(
        (REPO / "outputs/phase_13_5c/TRIAL_FAMILY.csv").read_bytes()
    ).hexdigest()
    for v in reg.experiments(trial_role=TrialRole.NEIGHBOUR):
        assert v.result.source_artifact == "outputs/phase_13_5c/TRIAL_FAMILY.csv"
        assert v.result.source_artifact_sha256 == expected_sha
        assert "no ValidationReport of its own" in v.result.evidence_completeness
        assert "deliberately NULL" in v.result.evidence_completeness
        # the statistics it DOES claim are real
        assert v.result.bh_p_value is not None
        assert v.result.bh_q is not None
        assert v.result.bh_rejected_at_q is not None


def test_14_1_E_no_neighbour_inherits_canonical_provenance(imported):
    reg, _, _ = imported
    canonical_hashes = {
        v.experiment.target_schedule_hash
        for v in reg.experiments(trial_role=TrialRole.CANONICAL, phase=PHASE)
    }
    canonical_reports = {
        v.experiment.report_fingerprint
        for v in reg.experiments(trial_role=TrialRole.CANONICAL, phase=PHASE)
    }
    for v in reg.experiments(trial_role=TrialRole.NEIGHBOUR):
        assert v.experiment.target_schedule_hash not in canonical_hashes
        assert v.experiment.report_fingerprint not in canonical_reports


def test_14_1_precorrection_rows_do_not_carry_corrected_provenance(imported):
    reg, _, _ = imported
    rows = reg.experiments(phase=PRECORRECTION_PHASE, include_superseded=True)
    assert len(rows) == 21
    for v in rows:
        assert v.experiment.target_schedule_hash is None
        assert v.experiment.report_fingerprint is None
        assert v.result.source_artifact == (
            "outputs/phase_13_5c/CONTRACT_ECONOMICS_CORRECTION.json"
        )


def test_14_1_F_canonical_exact_duplicate_still_works_from_pre_run_inputs(imported):
    """Rebuild a canonical identity from the frozen manifest + committed
    fingerprints only -- no schedule, no report -- and find the imported row."""
    reg, _, _ = imported
    identity = _independent_identity(root="ES", family="tsmom", variant="canonical")
    hit = reg.find_exact_duplicate(identity, asset_domain=AssetDomain.FUTURES)
    assert hit.exists
    assert hit.experiment_id == "ES__TSMOM__CANONICAL__VALIDATION_2023_2024"
    assert hit.headline_verdict is RegistryVerdict.REJECT


def test_14_1_G_neighbour_exact_duplicate_works_without_a_schedule(imported):
    """THE Phase 14.1 regression test.

    An agent proposing the already-tested NQ TSMOM neighbour (fast_horizon=10,
    slow_horizon=120) computes its identity from pre-run semantic inputs alone
    and must be told the experiment already exists -- with no schedule
    compilation and no backtest.
    """
    reg, _, _ = imported
    manifest = json.loads(
        (REPO / "data/manifests/phase_13_5c/candidate_manifest.json").read_text()
    )["manifest"]
    trial = next(
        t for t in manifest["trials"]
        if t["root_symbol"] == "NQ" and t["family_key"] == "tsmom"
    )
    index = 0
    proposed_params = trial["neighbour_params"][index]
    assert proposed_params["fast_horizon"] == 10
    assert proposed_params["slow_horizon"] == 120

    identity = _independent_identity(
        root="NQ", family="tsmom", variant=f"neighbour_{index}",
        params=proposed_params,
    )
    hit = reg.find_exact_duplicate(identity, asset_domain=AssetDomain.FUTURES)
    assert hit.exists, "an already-tested neighbour must be found without a schedule"
    assert hit.experiment_id == "NQ__TSMOM__NEIGHBOUR_0__VALIDATION_2023_2024"
    assert hit.headline_verdict is RegistryVerdict.NOT_ADJUDICATED
    assert hit.authority is Authority.AUTHORITATIVE
    # and the registry stored no schedule hash for it, so none was needed
    assert reg.get(hit.experiment_id).experiment.target_schedule_hash is None


def test_14_1_H_semantic_change_still_yields_a_different_identity(imported):
    """Same neighbour, one meaningful change each time -> a new identity that is
    NOT a duplicate."""
    reg, _, _ = imported
    base = _independent_identity(root="NQ", family="tsmom", variant="neighbour_0")
    assert reg.find_exact_duplicate(base, asset_domain=AssetDomain.FUTURES).exists

    for override in (
        {"strategy_fingerprint": "stratdsl1:something-else"},
        {"root_symbol": "ES"},
        {"parameter_variant_identity": parameter_variant_identity(
            {"fast_horizon": 11, "slow_horizon": 120, "size": 1, "root_symbol": "NQ"}
        )},
        {"dataset_fingerprint": "valdataset2:rebuilt"},
        {"validation_spec_fingerprint": "validationspec1:changed"},
        {"execution_config_identity": "execconfig1:changed"},
        {"cost_config_identity": "costconfig1:changed"},
        {"risk_identity": "riskconfig1:changed"},
    ):
        changed = _independent_identity(
            root="NQ", family="tsmom", variant="neighbour_0", **override
        )
        assert changed != base
        assert not reg.find_exact_duplicate(changed, asset_domain=AssetDomain.FUTURES).exists


# --------------------------------------------------------------------------
# Independent identity construction (Phase 14.2 section 9).
#
# Nothing here reads the target row. Configuration identities come from the
# committed artifacts; strategy fingerprint, parameter variant and feature-spec
# identity are DERIVED from the reconstructed StrategySpec. This is what a
# future Research Agent actually has before it runs anything.
# --------------------------------------------------------------------------
def _independent_identity(
    *, root: str, family: str, variant: str, params: dict | None = None, **overrides
) -> str:
    sources = Phase135cSources(REPO)
    ids = _config_identities(sources)
    fp = sources.validation_report(root, family)["fingerprints"]
    manifest_trial = next(
        t for t in sources.manifest["trials"]
        if t["root_symbol"] == root and t["family_key"] == family
    )
    if variant == "canonical":
        frozen_params = manifest_trial["canonical_params"]
        expected_fp = manifest_trial["strategy_fingerprint"]
    else:
        index = int(variant.rsplit("_", 1)[1])
        frozen_params = manifest_trial["neighbour_params"][index]
        expected_fp = manifest_trial["neighbour_fingerprints"][index]
    proposed = params if params is not None else frozen_params

    spec = spec_for_params(family, proposed)
    derived_strategy_fp = strategy_fingerprint(spec)
    if params is None:
        assert derived_strategy_fp == expected_fp
    kwargs = {
        "strategy_fingerprint": derived_strategy_fp,
        "strategy_family": family,
        "root_symbol": root,
        "parameter_variant_identity": parameter_variant_identity(proposed),
        "dataset_fingerprint": fp["dataset_fingerprint"],
        "split_identity": fp["split_fingerprint"],
        "validation_spec_fingerprint": fp["validation_fingerprint"],
        "reliability_policy_fingerprint": fp["reliability_policy_fingerprint"],
        "execution_config_identity": ids["execution_corrected"],
        "cost_config_identity": ids["cost"],
        "risk_identity": ids["risk"],
        "feature_spec_fingerprint": fingerprint(
            "featset1", {"features": list(feature_fingerprints(spec))}
        ),
    }
    kwargs.update(overrides)
    return experiment_identity(**kwargs)


# ==========================================================================
# Phase 14.2 -- neighbour feature identity and provenance
# ==========================================================================
def _manifest() -> dict:
    return json.loads(
        (REPO / "data/manifests/phase_13_5c/candidate_manifest.json").read_text()
    )["manifest"]


def _frozen_variants() -> list[tuple[str, str, int, dict, str]]:
    """(family, root, index, params, committed_neighbour_fingerprint) x 86."""
    out = []
    for t in _manifest()["trials"]:
        for i, (params, fp) in enumerate(
            zip(t["neighbour_params"], t["neighbour_fingerprints"], strict=True)
        ):
            out.append((t["family_key"], t["root_symbol"], i, params, fp))
    return out


def test_14_2_A_all_86_neighbour_specs_reconstruct_faithfully():
    """Configuration reconstruction only -- no market data, no backtest."""
    variants = _frozen_variants()
    assert len(variants) == 86
    for family, root, i, params, committed in variants:
        rebuilt = reconstruct_variant(
            family_key=family, root_symbol=root, variant_label=f"neighbour_{i}",
            params=params, expected_strategy_fingerprint=committed,
        )
        assert rebuilt.strategy_fingerprint == committed


def test_14_2_A_a_bad_reconstruction_is_a_blocker():
    family, root, i, params, _committed = _frozen_variants()[0]
    with pytest.raises(Phase135cArtifactError, match="refusing to derive feature"):
        reconstruct_variant(
            family_key=family, root_symbol=root, variant_label=f"neighbour_{i}",
            params=params, expected_strategy_fingerprint="stratdsl1:not-the-frozen-one",
        )


def test_14_2_B_stored_feature_fingerprints_are_the_neighbours_own(imported):
    reg, _, _ = imported
    for family, root, i, params, committed in _frozen_variants():
        rebuilt = reconstruct_variant(
            family_key=family, root_symbol=root, variant_label=f"neighbour_{i}",
            params=params, expected_strategy_fingerprint=committed,
        )
        stored = reg.get(
            friendly_experiment_id(
                root_symbol=root, strategy_family=family,
                variant_label=f"neighbour_{i}", split_label="VALIDATION_2023_2024",
            )
        ).experiment
        assert tuple(stored.strategy_spec_json["feature_fingerprints"]) == (
            rebuilt.feature_fingerprints
        )


def test_14_2_C_feature_spec_fingerprint_matches_those_features(imported):
    reg, _, _ = imported
    for v in reg.experiments(include_superseded=True):
        derived = fingerprint(
            "featset1",
            {"features": list(v.experiment.strategy_spec_json["feature_fingerprints"])},
        )
        assert v.experiment.feature_spec_fingerprint == derived, v.experiment_id


def test_14_2_D_no_neighbour_takes_canonical_features_unless_proven_equal(imported):
    """66 of the 86 neighbours genuinely differ from their canonical trial; the
    20 that match do so because reconstruction proves it, not by inheritance."""
    reg, _, _ = imported
    same = diff = 0
    for v in reg.experiments(trial_role=TrialRole.NEIGHBOUR):
        canonical = reg.get(v.experiment.parent_experiment_identity).experiment
        neighbour_feats = tuple(v.experiment.strategy_spec_json["feature_fingerprints"])
        canonical_feats = tuple(canonical.strategy_spec_json["feature_fingerprints"])
        rebuilt = reconstruct_variant(
            family_key=v.experiment.strategy_family,
            root_symbol=v.experiment.root_symbol,
            variant_label=v.experiment.parameter_variant_label,
            params=v.experiment.strategy_spec_json["params"],
            expected_strategy_fingerprint=v.experiment.strategy_fingerprint,
        )
        assert neighbour_feats == rebuilt.feature_fingerprints
        if neighbour_feats == canonical_feats:
            same += 1
        else:
            diff += 1
    assert (same, diff) == (20, 66)


def test_14_2_E_horizon_and_window_neighbours_differ_from_canonical(imported):
    """Known feature-moving examples, one per family."""
    reg, _, _ = imported
    cases = [
        ("NQ", "tsmom", "neighbour_0", {"fast_horizon": 10, "slow_horizon": 120}),
        ("ES", "ma_trend", "neighbour_0", {"fast_window": 25}),
        ("CL", "breakout", "neighbour_0", {"lookback": 35}),
        ("GC", "mean_reversion", "neighbour_0", {"zscore_window": 15}),
        ("NQ", "silver_bullet", "neighbour_0", {"displacement_atr_multiple": 1.25}),
    ]
    for root, family, label, expect_params in cases:
        v = reg.get(
            friendly_experiment_id(
                root_symbol=root, strategy_family=family, variant_label=label,
                split_label="VALIDATION_2023_2024",
            )
        )
        params = v.experiment.strategy_spec_json["params"]
        for key, value in expect_params.items():
            assert params[key] == value, (root, family, key)
        canonical = reg.get(v.experiment.parent_experiment_identity).experiment
        assert (
            v.experiment.strategy_spec_json["feature_fingerprints"]
            != canonical.strategy_spec_json["feature_fingerprints"]
        ), f"{root}/{family}/{label} should have its own feature set"
        assert (
            v.experiment.feature_spec_fingerprint != canonical.feature_spec_fingerprint
        )


def test_14_2_F_threshold_only_neighbours_keep_an_identical_feature_set(imported):
    """entry_z / exit_z move DSL thresholds, not FeatureSpecs. An identical
    feature set is correct here -- derived, not forced."""
    reg, _, _ = imported
    checked = 0
    for v in reg.experiments(strategy_family="mean_reversion",
                             trial_role=TrialRole.NEIGHBOUR):
        params = v.experiment.strategy_spec_json["params"]
        canonical = reg.get(v.experiment.parent_experiment_identity).experiment
        canonical_params = canonical.strategy_spec_json["params"]
        changed = {k for k in params if params[k] != canonical_params.get(k)}
        if changed <= {"entry_z", "exit_z"}:
            assert (
                v.experiment.strategy_spec_json["feature_fingerprints"]
                == canonical.strategy_spec_json["feature_fingerprints"]
            )
            assert (
                v.experiment.feature_spec_fingerprint
                == canonical.feature_spec_fingerprint
            )
            checked += 1
        else:
            assert changed == {"zscore_window"}
            assert (
                v.experiment.strategy_spec_json["feature_fingerprints"]
                != canonical.strategy_spec_json["feature_fingerprints"]
            )
    assert checked == 20


def test_14_2_G_similarity_uses_the_corrected_feature_sets(imported):
    """same_feature_set fires for a threshold-only neighbour and not for a
    horizon-changing one."""
    reg, _, _ = imported
    canonical = reg.get("GC__MEAN_REVERSION__CANONICAL__VALIDATION_2023_2024").experiment
    canonical_feats = tuple(canonical.strategy_spec_json["feature_fingerprints"])

    hits = reg.find_related(
        strategy_family="mean_reversion", root_symbol="GC", asset_domain=AssetDomain.FUTURES,
        params=canonical.strategy_spec_json["params"],
        signal_cadence="daily_trading_day",
        execution_cadence="native_1m_raw_contract",
        feature_fingerprints=canonical_feats, top_k=40,
    )
    by_id = {h.experiment_id: h for h in hits}
    threshold_only = by_id["GC__MEAN_REVERSION__NEIGHBOUR_2__VALIDATION_2023_2024"]
    window_changing = by_id["GC__MEAN_REVERSION__NEIGHBOUR_0__VALIDATION_2023_2024"]
    assert threshold_only.params["entry_z"] != canonical.strategy_spec_json["params"][
        "entry_z"
    ]
    assert window_changing.params["zscore_window"] != canonical.strategy_spec_json[
        "params"
    ]["zscore_window"]
    assert "same_feature_set" in threshold_only.similarity.explain()
    assert "same_feature_set" not in window_changing.similarity.explain()
    assert threshold_only.similarity.components["same_feature_set"] > 0
    assert window_changing.similarity.components["same_feature_set"] == 0


def test_14_2_H_changing_feature_semantics_changes_identity(imported):
    reg, _, _ = imported
    base = _independent_identity(root="NQ", family="tsmom", variant="neighbour_0")
    assert reg.find_exact_duplicate(base, asset_domain=AssetDomain.FUTURES).exists
    moved = _independent_identity(
        root="NQ", family="tsmom", variant="neighbour_0",
        feature_spec_fingerprint="featset1:different-features",
    )
    assert moved != base
    assert not reg.find_exact_duplicate(moved, asset_domain=AssetDomain.FUTURES).exists


def test_14_2_I_canonical_records_are_unchanged_by_the_correction(imported):
    """Canonical rows keep the manifest's feature set and strategy id: the
    correction only removed a value neighbours should never have had."""
    reg, _, _ = imported
    for t in _manifest()["trials"]:
        v = reg.get(
            friendly_experiment_id(
                root_symbol=t["root_symbol"], strategy_family=t["family_key"],
                variant_label="canonical", split_label="VALIDATION_2023_2024",
            )
        ).experiment
        assert list(v.strategy_spec_json["feature_fingerprints"]) == list(
            t["feature_fingerprints"]
        )
        assert v.strategy_id == t["strategy_id"]
        assert v.strategy_fingerprint == t["strategy_fingerprint"]


def test_14_2_J_precorrection_lineage_still_intact(imported):
    reg, _, _ = imported
    rows = reg.experiments(phase=PRECORRECTION_PHASE, include_superseded=True)
    assert len(rows) == 21
    for v in rows:
        assert v.authority is Authority.SUPERSEDED
        assert v.experiment.code_commit == SUPERSEDED_COMMIT
        assert v.experiment.target_schedule_hash is None
        assert v.experiment.report_fingerprint is None
        assert reg.resolve_authoritative(
            v.experiment_identity
        ).experiment.code_commit == CORRECTED_COMMIT


def test_14_2_5_strategy_id_is_the_reconstructed_spec_id(imported):
    """No synthetic '<canonical>-NEIGHBOUR-<i>' id masquerading as a real
    StrategySpec.strategy_id; the registry label lives in
    parameter_variant_label."""
    reg, _, _ = imported
    for v in reg.experiments(trial_role=TrialRole.NEIGHBOUR):
        rebuilt = reconstruct_variant(
            family_key=v.experiment.strategy_family,
            root_symbol=v.experiment.root_symbol,
            variant_label=v.experiment.parameter_variant_label,
            params=v.experiment.strategy_spec_json["params"],
            expected_strategy_fingerprint=v.experiment.strategy_fingerprint,
        )
        assert v.experiment.strategy_id == rebuilt.strategy_id
        assert "-NEIGHBOUR-" not in v.experiment.strategy_id
        assert v.experiment.parameter_variant_label.startswith("neighbour_")


def test_14_2_9_neighbour_duplicate_detection_is_independent(imported):
    """THE Phase 14.2 regression test.

    The proposal identity is built from the frozen params + reconstructed
    StrategySpec + committed configuration fingerprints. It never reads the
    target row's stored feature_spec_fingerprint, so it cannot be circular.
    """
    reg, _, _ = imported
    trial = next(
        t for t in _manifest()["trials"]
        if t["root_symbol"] == "NQ" and t["family_key"] == "tsmom"
    )
    params = trial["neighbour_params"][0]
    assert params["fast_horizon"] == 10 and params["slow_horizon"] == 120

    identity = _independent_identity(root="NQ", family="tsmom", variant="neighbour_0")
    hit = reg.find_exact_duplicate(identity, asset_domain=AssetDomain.FUTURES)
    assert hit.exists
    assert hit.experiment_id == "NQ__TSMOM__NEIGHBOUR_0__VALIDATION_2023_2024"
    assert hit.headline_verdict is RegistryVerdict.NOT_ADJUDICATED

    # the identity used the neighbour's OWN feature set, which differs from the
    # canonical trial's -- the Phase 14.2 bug would have failed this lookup.
    stored = reg.get(hit.experiment_id).experiment
    canonical = reg.get(stored.parent_experiment_identity).experiment
    assert stored.feature_spec_fingerprint != canonical.feature_spec_fingerprint
    wrong = _independent_identity(
        root="NQ", family="tsmom", variant="neighbour_0",
        feature_spec_fingerprint=canonical.feature_spec_fingerprint,
    )
    assert not reg.find_exact_duplicate(wrong, asset_domain=AssetDomain.FUTURES).exists


def test_14_2_9_canonical_duplicate_detection_is_independent(imported):
    reg, _, _ = imported
    identity = _independent_identity(root="ES", family="tsmom", variant="canonical")
    hit = reg.find_exact_duplicate(identity, asset_domain=AssetDomain.FUTURES)
    assert hit.exists
    assert hit.experiment_id == "ES__TSMOM__CANONICAL__VALIDATION_2023_2024"
    assert hit.headline_verdict is RegistryVerdict.REJECT


def test_14_2_schema_v4_migration_preserves_v3(tmp_path):
    path = tmp_path / "v3.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE experiments (experiment_identity TEXT PRIMARY KEY,"
        " experiment_id TEXT, content_fingerprint TEXT);"
        "CREATE TABLE results (experiment_identity TEXT PRIMARY KEY);"
        "INSERT INTO registry_meta VALUES ('schema_version', '3');"
        "INSERT INTO experiments VALUES ('experiment1:v3row', 'OLD_V3_ROW', 'fp');"
    )
    conn.commit()
    conn.close()
    with ExperimentRegistry(path) as reg:
        assert reg.schema_version == SCHEMA_VERSION == 7
        # the v3 representation is preserved through every migration in the chain
        assert [
            r[0] for r in reg._conn.execute(
                "SELECT experiment_id FROM experiments_v3_legacy"
            )
        ] == ["OLD_V3_ROW"]
        assert reg.count_experiments() == 0
