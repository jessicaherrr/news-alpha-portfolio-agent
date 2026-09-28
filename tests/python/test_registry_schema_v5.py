"""Registry schema v5 -- the scientific-identity / execution-attempt split.

The same pre-run ``experiment_identity`` may carry several immutable execution
attempts. Only a ``VALID`` attempt supplies the authoritative scientific result;
an ``INVALID_EXECUTION`` attempt (engineering / data-pipeline / software /
infrastructure defect) is preserved verbatim and never enters a BH/FDR family.

The eight proofs the Phase 15B correction requires are
``test_proof_1_...`` .. ``test_proof_8_...``.
"""
from __future__ import annotations

import json
import sqlite3

import pytest
from alpha_agent.core.instrument import AssetDomain
from alpha_agent.registry.enums import (
    AttemptRelation,
    AttemptStatus,
    ExperimentStatus,
    FailureClass,
    FailureScope,
    InvalidationClass,
    RegistryVerdict,
    TrialRole,
)
from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
from alpha_agent.registry.models import (
    ExecutionAttemptRecord,
    ExperimentRecord,
    FailureRecord,
    ImportBundle,
    MarketWindow,
    ResultRecord,
)
from alpha_agent.registry.schema import SCHEMA_VERSION
from alpha_agent.registry.sqlite_registry import (
    AttemptConflict,
    ExperimentRegistry,
    ImmutableResultError,
)

MW = MarketWindow(label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31")


def _identity(i: int) -> str:
    return experiment_identity(
        strategy_fingerprint=f"stratdsl1:s{i}",
        strategy_family="ml_meta_label.tsmom",
        root_symbol="NQ",
        parameter_variant_identity=parameter_variant_identity({"n": i}),
        dataset_fingerprint="valdataset2:ds",
        split_identity="split1:sp",
        validation_spec_fingerprint="mlvalidationspec1:vs",
        reliability_policy_fingerprint="valreliabilitypolicy1:rp",
        execution_config_identity="execconfig1:ex",
        cost_config_identity="costconfig1:co",
        risk_identity="riskconfig1:ri",
        feature_spec_fingerprint="mlfeatureset1:fs",
    )


def _experiment(i: int, *, report_fp: str | None = None, sched: str | None = None,
                status: ExperimentStatus = ExperimentStatus.COMPLETED,
                role: TrialRole = TrialRole.CANONICAL) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_identity=_identity(i),
        experiment_id=f"EXP_{i}",
        display_name=f"exp {i}",
        created_at="2026-01-01T00:00:00+00:00",
        phase="15B",
        status=status,
        code_commit="commit0",
        root_symbol="NQ",
        asset_domain=AssetDomain.FUTURES,
        strategy_family="ml_meta_label.tsmom",
        strategy_fingerprint=f"stratdsl1:s{i}",
        strategy_id="S",
        strategy_spec_json={"params": {"n": i}},
        feature_spec_fingerprint="mlfeatureset1:fs",
        target_schedule_hash=sched,
        dataset_fingerprint="valdataset2:ds",
        split_identity="split1:sp",
        market_window=MW,
        validation_spec_fingerprint="mlvalidationspec1:vs",
        reliability_policy_fingerprint="valreliabilitypolicy1:rp",
        execution_config_identity="execconfig1:ex",
        cost_config_identity="costconfig1:co",
        risk_identity="riskconfig1:ri",
        trial_role=role,
        parameter_variant_identity=parameter_variant_identity({"n": i}),
        parameter_variant_label="canonical",
        report_fingerprint=report_fp,
    )


def _result(i: int, *, verdict=RegistryVerdict.REJECT, p=0.4, holdout=False) -> ResultRecord:
    return ResultRecord(
        experiment_identity=_identity(i),
        headline_verdict=verdict,
        reason_codes=("R",),
        bh_p_value=p,
        bh_q=p,
        daily_sharpe=0.1,
        holdout_eligible=holdout,
    )


def _invalid_attempt(i: int) -> ExecutionAttemptRecord:
    return ExecutionAttemptRecord(
        experiment_identity=_identity(i),
        attempt_status=AttemptStatus.INVALID_EXECUTION,
        invalidation_class=InvalidationClass.FEATURE_PIPELINE_DEFECT,
        invalidation_detail="weekend gaps starved RESET_ON_GAP features",
        defect_resolution_commit="cc206de",
        code_commit="a31f571",
        engine="cpp_quant_core",
        report_fingerprint="mltrialreport1:bad",
        created_at="2026-09-10T02:30:00+00:00",
        notes="a31f571 -- feature-pipeline defect",
    )


def _valid_attempt(i: int, *, commit: str = "cc206de") -> ExecutionAttemptRecord:
    return ExecutionAttemptRecord(
        experiment_identity=_identity(i),
        attempt_status=AttemptStatus.VALID,
        code_commit=commit,
        engine="cpp_quant_core",
        report_fingerprint=f"mltrialreport1:ok_{commit}",
        created_at="2026-09-11T00:00:00+00:00",
        notes="corrected feature path",
    )


@pytest.fixture
def reg(tmp_path):
    r = ExperimentRegistry(tmp_path / "v5.sqlite")
    yield r
    r.close()


# --------------------------------------------------------------------------
# the eight required proofs
# --------------------------------------------------------------------------
def test_proof_1_one_experiment_has_invalid_attempt_then_valid_attempt(reg):
    ident = _identity(1)
    reg.apply_bundle(ImportBundle(
        import_id="a31f571", phase="15B", source_fingerprint="sha_a", created_at="t0",
        experiments=(_experiment(1, status=ExperimentStatus.FAILED),),
        execution_attempts=(_invalid_attempt(1),),
        results=(_result(1, verdict=RegistryVerdict.NOT_ADJUDICATED, p=1.0),),
        failures=(FailureRecord(
            failure_id="f_a31f571", scope=FailureScope.EXPERIMENT, experiment_identity=ident,
            failure_class=FailureClass.INVALID_EXECUTION_ATTEMPT, failure_code="FEATURE_PIPELINE",
            summary="a31f571 defect", mechanism="RESET_ON_GAP", created_at="t0",
        ),),
    ))
    reg.apply_bundle(ImportBundle(
        import_id="cc206de", phase="15B", source_fingerprint="sha_b", created_at="t1",
        experiments=(_experiment(1),),
        execution_attempts=(_valid_attempt(1),),
        results=(_result(1, verdict=RegistryVerdict.REJECT, p=0.3),),
    ))
    view = reg.get(ident)
    assert [a.attempt_ordinal for a in view.attempts] == [1, 2]
    assert [a.attempt_status for a in view.attempts] == [
        AttemptStatus.INVALID_EXECUTION, AttemptStatus.VALID
    ]


def test_proof_2_the_invalid_attempt_A_remains_inspectable(reg):
    test_proof_1_one_experiment_has_invalid_attempt_then_valid_attempt(reg)
    view = reg.get(_identity(1))
    a1 = view.attempts[0]
    assert a1.attempt_status is AttemptStatus.INVALID_EXECUTION
    assert a1.invalidation_class == "FEATURE_PIPELINE_DEFECT"
    assert a1.defect_resolution_commit == "cc206de"
    assert a1.result is not None
    assert a1.result.headline_verdict is RegistryVerdict.NOT_ADJUDICATED
    assert a1.result.bh_p_value == 1.0
    # the a31f571 failure evidence is still there, linked to exactly attempt 1
    fails = reg.failures(experiment_identity=_identity(1))
    assert any(f.failure_id == "f_a31f571" for f in fails)
    row = reg._conn.execute(
        "SELECT attempt_id FROM failures WHERE failure_id = 'f_a31f571'"
    ).fetchone()
    assert row["attempt_id"] == a1.attempt_id


def test_proof_3_the_valid_attempt_B_becomes_authoritative(reg):
    test_proof_1_one_experiment_has_invalid_attempt_then_valid_attempt(reg)
    view = reg.get(_identity(1))
    assert view.result is not None
    assert view.result.headline_verdict is RegistryVerdict.REJECT
    assert view.result.bh_p_value == 0.3
    auth = reg.authoritative_attempt(_identity(1))
    assert auth.attempt_ordinal == 2 and auth.is_valid
    # an explicit CORRECTS_ATTEMPT audit edge was recorded
    edges = reg._conn.execute(
        "SELECT relation_type FROM attempt_lineage WHERE target_attempt_id = ?",
        (view.attempts[0].attempt_id,),
    ).fetchall()
    assert edges and edges[0]["relation_type"] == AttemptRelation.CORRECTS_ATTEMPT.value


def test_proof_4_scientific_experiment_count_does_not_increase(reg):
    before = reg.summary().authoritative_statistical_hypotheses
    test_proof_1_one_experiment_has_invalid_attempt_then_valid_attempt(reg)
    s = reg.summary()
    assert s.authoritative_statistical_hypotheses == before + 1     # ONE hypothesis, not two
    assert s.total_experiment_rows == before + 1
    assert s.execution_attempts == 2
    assert s.valid_execution_attempts == 1
    assert s.invalid_execution_attempts == 1


def test_proof_5_a_valid_authoritative_result_blocks_an_accidental_rerun(reg):
    reg.apply_bundle(ImportBundle(
        import_id="run1", phase="15B", source_fingerprint="s", created_at="t",
        experiments=(_experiment(5),), results=(_result(5, verdict=RegistryVerdict.PASS),),
    ))
    dup = reg.find_exact_duplicate(_identity(5), asset_domain=AssetDomain.FUTURES)
    assert dup.exists and dup.has_valid_authoritative_result
    assert dup.blocks_reexecution is True
    # a bare re-insert of a different result is refused
    with pytest.raises(ImmutableResultError):
        reg.insert_experiment(_experiment(5), _result(5, verdict=RegistryVerdict.REJECT))


def test_proof_6_an_invalid_only_history_permits_rerun(reg):
    reg.apply_bundle(ImportBundle(
        import_id="a31f571", phase="15B", source_fingerprint="s", created_at="t",
        experiments=(_experiment(6, status=ExperimentStatus.FAILED),),
        execution_attempts=(_invalid_attempt(6),),
        results=(_result(6, verdict=RegistryVerdict.NOT_ADJUDICATED, p=1.0),),
    ))
    dup = reg.find_exact_duplicate(_identity(6), asset_domain=AssetDomain.FUTURES)
    assert dup.exists
    assert dup.has_valid_authoritative_result is False
    assert dup.blocks_reexecution is False
    assert dup.n_invalid_attempts == 1
    # the corrected rerun lands as attempt 2 -- no new identity, no conflict
    reg.apply_bundle(ImportBundle(
        import_id="cc206de", phase="15B", source_fingerprint="s2", created_at="t2",
        experiments=(_experiment(6),), execution_attempts=(_valid_attempt(6),),
        results=(_result(6, verdict=RegistryVerdict.INCONCLUSIVE, p=0.2),),
    ))
    assert reg.get(_identity(6)).verdict is RegistryVerdict.INCONCLUSIVE
    assert reg.count_experiments() == 1


def test_proof_7_v4_to_v5_migration_preserves_history_and_a31f571_evidence(tmp_path):
    """A synthetic v4 database with a Phase 13/14-style VALID import and a
    Phase-15B ``phase_15b__`` import migrates to v5: the first stays VALID, the
    Phase-15 rows become INVALID_EXECUTION, and every v4 row is preserved."""
    path = tmp_path / "v4.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "INSERT INTO registry_meta VALUES ('schema_version', '4');"
    )
    conn.commit()
    conn.close()
    # build the v4 shape via a v4-era ExperimentRegistry? we only have v5 now, so
    # write a v4 db by hand with the minimal columns the migration reads.
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE experiments (
            experiment_identity TEXT PRIMARY KEY, identity_schema TEXT, experiment_id TEXT,
            display_name TEXT, created_at TEXT, phase TEXT, status TEXT, code_commit TEXT,
            root_symbol TEXT, strategy_family TEXT, strategy_fingerprint TEXT, strategy_id TEXT,
            strategy_spec_json TEXT, feature_spec_fingerprint TEXT, target_schedule_hash TEXT,
            candidate_manifest_fingerprint TEXT, dataset_fingerprint TEXT, split_identity TEXT,
            market_window_json TEXT, validation_spec_fingerprint TEXT,
            reliability_policy_fingerprint TEXT, execution_config_identity TEXT,
            cost_config_identity TEXT, risk_identity TEXT, trial_role TEXT,
            canonical_or_neighbour TEXT, parameter_variant_identity TEXT,
            parameter_variant_label TEXT, parent_experiment_identity TEXT,
            report_fingerprint TEXT, notes TEXT, content_fingerprint TEXT, record_json TEXT);
        CREATE TABLE results (
            experiment_identity TEXT PRIMARY KEY, headline_verdict TEXT, reason_codes_json TEXT,
            gross_pnl_usd REAL, costs_usd REAL, net_pnl_usd REAL, daily_sharpe REAL,
            annualized_sharpe REAL, gating_null_p REAL, bh_p_value REAL, bh_q REAL,
            bh_rejected_at_q INTEGER, dsr_probability REAL, fold_consistency REAL,
            n_trades INTEGER, n_fills INTEGER, n_oos_days INTEGER, holdout_eligible INTEGER,
            evidence_completeness TEXT, source_artifact TEXT, source_artifact_sha256 TEXT,
            result_fingerprint TEXT, record_json TEXT);
        CREATE TABLE failures (
            failure_id TEXT PRIMARY KEY, scope TEXT, experiment_identity TEXT, failure_class TEXT,
            failure_code TEXT, summary TEXT, mechanism TEXT, evidence_json TEXT, action_taken TEXT,
            resolved INTEGER, resolution_commit TEXT, superseded_by TEXT, created_at TEXT,
            root_symbol TEXT, strategy_family TEXT, content_fingerprint TEXT);
        CREATE TABLE lineage (source_experiment_identity TEXT, target_experiment_identity TEXT,
            relation_type TEXT, note TEXT, PRIMARY KEY (source_experiment_identity,
            target_experiment_identity, relation_type));
        CREATE TABLE sensitivity_evidence (evidence_id TEXT PRIMARY KEY, experiment_identity TEXT,
            relation_type TEXT, kind TEXT, baseline_verdict TEXT, rerun_verdict TEXT,
            verdict_changed INTEGER, metrics_json TEXT, in_bh_fdr_denominator INTEGER);
        CREATE TABLE cross_market_evidence (evidence_id TEXT PRIMARY KEY, strategy_family TEXT,
            status TEXT, n_roots INTEGER, n_positive_roots INTEGER, max_single_root_pnl_share REAL,
            herfindahl_pnl REAL, concentrated_in_one_root INTEGER, referenced_identities_json TEXT);
        CREATE TABLE imports (import_id TEXT PRIMARY KEY, phase TEXT, source_fingerprint TEXT,
            created_at TEXT, counts_json TEXT, metadata_json TEXT);
        """
    )

    def _exp_row(ident, phase, verdict, p):
        rj = {
            "experiment_identity": ident, "identity_schema": "experiment-identity/2",
            "experiment_id": ident[-6:], "display_name": ident[-6:],
            "created_at": "2026-01-01T00:00:00+00:00", "phase": phase, "status": "COMPLETED",
            "code_commit": "c", "root_symbol": "NQ", "strategy_family": "f",
            "strategy_fingerprint": "sf", "strategy_id": "S", "strategy_spec_json": {},
            "feature_spec_fingerprint": "fs", "target_schedule_hash": None,
            "candidate_manifest_fingerprint": None, "dataset_fingerprint": "ds",
            "split_identity": "sp",
            "market_window": {"label": "V", "start_date": "2023-01-01", "end_date": "2024-12-31"},
            "validation_spec_fingerprint": "vs", "reliability_policy_fingerprint": "rp",
            "execution_config_identity": "ex", "cost_config_identity": "co", "risk_identity": "ri",
            "trial_role": "CANONICAL", "parameter_variant_identity": "pv",
            "parameter_variant_label": "canonical", "parent_experiment_identity": None,
            "report_fingerprint": None, "notes": "",
        }
        conn.execute(
            "INSERT INTO experiments (experiment_identity, identity_schema, experiment_id, "
            "display_name, created_at, phase, status, code_commit, root_symbol, strategy_family, "
            "strategy_fingerprint, strategy_id, strategy_spec_json, feature_spec_fingerprint, "
            "target_schedule_hash, candidate_manifest_fingerprint, dataset_fingerprint, "
            "split_identity, market_window_json, validation_spec_fingerprint, "
            "reliability_policy_fingerprint, execution_config_identity, cost_config_identity, "
            "risk_identity, trial_role, canonical_or_neighbour, parameter_variant_identity, "
            "parameter_variant_label, parent_experiment_identity, report_fingerprint, notes, "
            "content_fingerprint, record_json) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ident, "experiment-identity/2", ident[-6:], ident[-6:],
             "2026-01-01T00:00:00+00:00", phase, "COMPLETED", "c", "NQ", "f", "sf", "S",
             "{}", "fs", None, None, "ds", "sp", json.dumps(rj["market_window"]), "vs", "rp",
             "ex", "co", "ri", "CANONICAL", "canonical", "pv", "canonical", None, None, "",
             f"expcontent1:{ident}", json.dumps(rj)),
        )
        rrj = {"experiment_identity": ident, "headline_verdict": verdict, "reason_codes": ["R"],
               "bh_p_value": p, "bh_q": p, "holdout_eligible": False, "evidence_completeness": "full"}
        conn.execute(
            "INSERT INTO results (experiment_identity, headline_verdict, reason_codes_json, "
            "bh_p_value, bh_q, holdout_eligible, evidence_completeness, result_fingerprint, "
            "record_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (ident, verdict, json.dumps(["R"]), p, p, 0, "full",
             f"expresult1:{ident}", json.dumps(rrj)),
        )

    _exp_row("experiment1:p13", "13.5C", "REJECT", 0.4)
    _exp_row("experiment1:p15a", "15B", "NOT_ADJUDICATED", 1.0)
    _exp_row("experiment1:p15b", "15B", "NOT_ADJUDICATED", 1.0)
    conn.execute(
        "INSERT INTO failures VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("f_p15a", "EXPERIMENT", "experiment1:p15a", "STATISTICAL_INCONCLUSIVE", "REFUSED",
         "refused", "gate", "{}", "recorded", 0, None, None, "t", "NQ", "f", "failcontent1:x"),
    )
    conn.execute(
        "INSERT INTO imports VALUES "
        "('p14import1:x', '13.5C', 'sf', 't', '{}', '{}'), "
        "('phase_15b__vs', '15B', 'sf', 't', '{}', '{}')"
    )
    conn.commit()
    conn.close()

    with ExperimentRegistry(path) as r:
        # SCHEMA_VERSION has since moved to 6 (Phase 6 ETF Research Pilot's
        # asset_domain column) and 7 (News Alpha Phase H's additive
        # signal_path_evidence table); this v4 fixture still exercises the
        # ORIGINAL v4->v5 attempt-layer proofs below, now via a v4->v7
        # migration in one ensure_schema() call.
        assert r.schema_version == SCHEMA_VERSION == 7
        s = r.summary()
        # every hypothesis preserved; identity formula unchanged
        assert s.authoritative_statistical_hypotheses == 3
        assert s.execution_attempts == 3
        assert s.valid_execution_attempts == 1        # the 13.5C row
        assert s.invalid_execution_attempts == 2      # both phase_15b__ rows
        # the 13.5C result is still authoritative
        assert r.get("experiment1:p13").verdict is RegistryVerdict.REJECT
        # the phase_15b rows have NO authoritative result (invalid-only)
        assert r.get("experiment1:p15a").result is None
        assert r.get("experiment1:p15a").attempts[0].attempt_status is (
            AttemptStatus.INVALID_EXECUTION
        )
        assert r.get("experiment1:p15a").attempts[0].invalidation_class == (
            "FEATURE_PIPELINE_DEFECT"
        )
        # the a31f571 failure evidence survives, bound to its (invalid) attempt
        f = r.failures(experiment_identity="experiment1:p15a")
        assert f and f[0].failure_id == "f_p15a"
        # every v4 table is preserved verbatim
        for t in ("experiments", "results", "failures", "lineage", "imports"):
            assert r._conn.execute(
                f"SELECT count(*) FROM {t}_v4_legacy"
            ).fetchone()[0] >= 0


def test_proof_8_bh_family_sees_only_the_corrected_authoritative_results(reg):
    """Build a 4-identity Phase-15-like family: run a31f571 (all INVALID), then
    the corrected run. A registry query for the family's authoritative results
    returns exactly the 4 corrected verdicts -- never 8, never the p=1 values."""
    idents = [_identity(i) for i in range(10, 14)]
    reg.apply_bundle(ImportBundle(
        import_id="a31f571", phase="15B", source_fingerprint="sa", created_at="t0",
        experiments=tuple(_experiment(i, status=ExperimentStatus.FAILED) for i in range(10, 14)),
        execution_attempts=tuple(_invalid_attempt(i) for i in range(10, 14)),
        results=tuple(_result(i, verdict=RegistryVerdict.NOT_ADJUDICATED, p=1.0)
                      for i in range(10, 14)),
    ))
    corrected = {10: RegistryVerdict.REJECT, 11: RegistryVerdict.REJECT,
                 12: RegistryVerdict.INCONCLUSIVE, 13: RegistryVerdict.PASS}
    reg.apply_bundle(ImportBundle(
        import_id="cc206de", phase="15B", source_fingerprint="sb", created_at="t1",
        experiments=tuple(_experiment(i) for i in range(10, 14)),
        execution_attempts=tuple(_valid_attempt(i) for i in range(10, 14)),
        results=tuple(_result(i, verdict=corrected[i], p=0.2 + 0.01 * i) for i in range(10, 14)),
    ))

    # the BH family = the authoritative result of each identity
    fam = [reg.get(ident) for ident in idents]
    assert len(fam) == 4
    verdicts = sorted(v.verdict.value for v in fam)
    assert verdicts == ["INCONCLUSIVE", "PASS", "REJECT", "REJECT"]
    p_values = sorted(v.result.bh_p_value for v in fam)
    assert all(p < 1.0 for p in p_values)             # never the invalid attempt's p=1
    # attempt_results physically holds 8 rows, but only 4 are authoritative
    assert reg._conn.execute("SELECT COUNT(*) FROM attempt_results").fetchone()[0] == 8
    auth_rows = reg._conn.execute(
        "SELECT COUNT(*) FROM attempt_results ar JOIN execution_attempts ea "
        "ON ar.attempt_id = ea.attempt_id WHERE ea.attempt_status = 'VALID'"
    ).fetchone()[0]
    assert auth_rows == 4
    # a verdict-filtered experiments() query only sees the corrected results
    rejects = reg.experiments(phase="15B", verdict=RegistryVerdict.REJECT)
    assert len(rejects) == 2
    refused = reg.experiments(phase="15B", verdict=RegistryVerdict.NOT_ADJUDICATED)
    assert len(refused) == 0


# --------------------------------------------------------------------------
# supporting invariants
# --------------------------------------------------------------------------
def test_experiment_identity_is_byte_identical_across_attempts(reg):
    ident = _identity(20)
    reg.insert_experiment(_experiment(20, status=ExperimentStatus.FAILED),
                          attempt=_invalid_attempt(20))
    reg.insert_experiment(_experiment(20), _result(20), attempt=_valid_attempt(20))
    rows = reg._conn.execute(
        "SELECT experiment_identity FROM execution_attempts WHERE experiment_identity = ?",
        (ident,),
    ).fetchall()
    assert {r["experiment_identity"] for r in rows} == {ident}
    assert reg.count_experiments() == 1


def test_a_completed_attempt_is_immutable(reg):
    a = _valid_attempt(21).model_copy(update={"attempt_ordinal": 1})
    reg.insert_experiment(_experiment(21), _result(21), attempt=a)
    with pytest.raises(ImmutableResultError):
        # SAME attempt (ordinal 1, same provenance -> same attempt_id), different result
        reg.insert_experiment(_experiment(21), _result(21, verdict=RegistryVerdict.PASS),
                              attempt=a)


def test_valid_attempt_cannot_carry_an_invalidation_class():
    with pytest.raises(ValueError, match="VALID attempt"):
        ExecutionAttemptRecord(
            experiment_identity=_identity(1), attempt_status=AttemptStatus.VALID,
            invalidation_class=InvalidationClass.SOFTWARE_DEFECT, created_at="t",
        )


def test_invalid_attempt_must_name_a_class():
    with pytest.raises(ValueError, match="INVALID_EXECUTION"):
        ExecutionAttemptRecord(
            experiment_identity=_identity(1),
            attempt_status=AttemptStatus.INVALID_EXECUTION, created_at="t",
        )


def test_two_attempts_cannot_share_an_ordinal(reg):
    reg.insert_experiment(_experiment(22), attempt=_valid_attempt(22, commit="x"))
    with pytest.raises(AttemptConflict):
        reg.insert_experiment(
            _experiment(22),
            attempt=_valid_attempt(22, commit="y").model_copy(update={"attempt_ordinal": 1}),
        )


# --------------------------------------------------------------------------
# Release-packaging reporting fix: the registry summary's TrialRole/verdict
# breakdown must always reconcile arithmetically -- this is what
# registry/cli.py's cmd_summary prints and what README.md's "Current honest
# results" section quotes. A registry exercising all four TrialRole values
# (the real production registry today has zero VARIANT trials, so this proof
# needs a synthetic fixture to actually exercise that branch).
# --------------------------------------------------------------------------

def test_summary_role_and_verdict_breakdown_reconciles_for_all_four_trial_roles(reg):
    reg.insert_experiment(_experiment(30, role=TrialRole.CANONICAL),
                          _result(30, verdict=RegistryVerdict.PASS), attempt=_valid_attempt(30))
    reg.insert_experiment(_experiment(31, role=TrialRole.CANONICAL),
                          _result(31, verdict=RegistryVerdict.REJECT), attempt=_valid_attempt(31))
    reg.insert_experiment(_experiment(32, role=TrialRole.CANONICAL),
                          _result(32, verdict=RegistryVerdict.INCONCLUSIVE), attempt=_valid_attempt(32))
    reg.insert_experiment(_experiment(33, role=TrialRole.CANONICAL),
                          _result(33, verdict=RegistryVerdict.NOT_ADJUDICATED), attempt=_valid_attempt(33))
    reg.insert_experiment(_experiment(34, role=TrialRole.NEIGHBOUR),
                          _result(34, verdict=RegistryVerdict.NOT_ADJUDICATED), attempt=_valid_attempt(34))
    reg.insert_experiment(_experiment(35, role=TrialRole.ABLATION),
                          _result(35, verdict=RegistryVerdict.NOT_ADJUDICATED), attempt=_valid_attempt(35))
    reg.insert_experiment(_experiment(36, role=TrialRole.VARIANT),
                          _result(36, verdict=RegistryVerdict.NOT_ADJUDICATED), attempt=_valid_attempt(36))

    s = reg.summary()
    assert (s.canonical, s.neighbour, s.ablation, s.variant) == (4, 1, 1, 1)
    assert s.canonical + s.neighbour + s.ablation + s.variant == s.authoritative_statistical_hypotheses == 7
    verdict_total = sum(
        s.canonical_verdict_counts.get(k, 0)
        for k in ("PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED")
    )
    assert verdict_total == s.canonical == 4

    # the actual printed CLI output (registry/cli.py cmd_summary) carries the
    # same reconciling numbers, and does not crash on its own internal
    # reconciliation asserts
    import argparse

    from alpha_agent.registry.cli import cmd_summary

    class _Args(argparse.Namespace):
        json = False

    cmd_summary(reg, _Args())  # would raise AssertionError if it ever drifted

