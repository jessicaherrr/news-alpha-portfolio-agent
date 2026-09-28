"""Phase 6 (ETF Research Pilot) -- schema v6's ``asset_domain`` column.

Covers the user's explicit instruction: "shared physical Registry,
structurally namespaced scientific evidence. Add: AssetDomain.ETF without
rewriting historical Futures rows or identities. Legacy Futures evidence must
remain identity-stable. ETF and Futures: may share Mechanism vocabulary, may
share Factor concepts, but must NOT share: experiment identity namespace,
BH/FDR families, validation families, scientific evidence counts merely
because labels match."

Proofs in this file:

* P1 -- ``ExperimentRecord`` requires ``asset_domain`` (no default; a Pydantic
  ``ValidationError`` without it).
* P2 -- ``experiment_identity()`` is completely domain-blind: identical
  pre-run inputs produce the identical identity regardless of which domain the
  caller happens to be building a record for, and the function's own
  parameter list carries no ``asset_domain`` input at all.
* P3 -- structural cross-domain isolation: a same-named ``strategy_family``
  under ``FUTURES`` and under ``ETF`` can never be pooled by
  ``ExperimentRegistry.experiments()``, ``find_related`` or
  ``FailureMemory.lookup`` -- not merely scored low, never returned at all.
* P4 -- ``find_exact_duplicate`` raises :class:`AssetDomainMismatch` when a
  hit exists but under a different domain than requested (defense in depth;
  cryptographically should never happen given the identity formula's own
  structural domain distinctness).
* P5 -- schema v5 -> v6 migration on a synthetic fixture: every
  ``experiment_identity`` value is byte-for-byte unchanged, every row gets
  ``asset_domain='FUTURES'``, row counts match exactly (nothing dropped,
  nothing duplicated), the v5 representation is preserved verbatim as
  ``*_v5_legacy``, and -- the specific FK-cascade bug this phase's own
  migration work found and fixed -- a BRAND NEW experiment (and its execution
  attempt) can be inserted AFTER the migration with no foreign-key failure.
* P6 -- the REAL committed local registry, if present in this checkout
  (``data/registry/experiments.sqlite``), is migrated on a COPY only (the
  original is opened strictly read-only and is never touched); every one of
  its real ``experiment_identity`` values is proven byte-identical before and
  after.
"""
from __future__ import annotations

import inspect
import shutil
import sqlite3
from pathlib import Path

import pytest
from alpha_agent.core.instrument import AssetDomain
from alpha_agent.registry.enums import ExperimentStatus, RegistryVerdict, TrialRole
from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
from alpha_agent.registry.models import ExperimentRecord, MarketWindow, ResultRecord
from alpha_agent.registry.schema import SCHEMA_VERSION
from alpha_agent.registry.sqlite_registry import AssetDomainMismatch, ExperimentRegistry
from pydantic import ValidationError

REPO = Path(__file__).resolve().parents[2]
REAL_REGISTRY = REPO / "data" / "registry" / "experiments.sqlite"

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


def _identity(*, root: str, family: str = "tsmom", params: dict | None = None) -> str:
    params = params or {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    return experiment_identity(
        strategy_fingerprint=f"stratdsl1:{family}:{root}",
        strategy_family=family,
        root_symbol=root,
        parameter_variant_identity=parameter_variant_identity(params),
        **_FINGERPRINTS,
    )


def _experiment(
    *, root: str, asset_domain: AssetDomain, family: str = "tsmom",
    params: dict | None = None, experiment_id: str | None = None,
) -> ExperimentRecord:
    params = params or {"fast_horizon": 20, "slow_horizon": 120, "size": 1}
    return ExperimentRecord(
        experiment_identity=_identity(root=root, family=family, params=params),
        experiment_id=experiment_id or f"{root}__{family.upper()}__CANONICAL__TEST",
        display_name=f"{root} / {family} / canonical / TEST",
        created_at="2026-01-01T00:00:00+00:00",
        phase="TEST",
        status=ExperimentStatus.COMPLETED,
        root_symbol=root,
        asset_domain=asset_domain,
        strategy_family=family,
        strategy_fingerprint=f"stratdsl1:{family}:{root}",
        strategy_id="TEST-STRAT",
        strategy_spec_json={"params": params},
        market_window=WINDOW,
        trial_role=TrialRole.CANONICAL,
        parameter_variant_identity=parameter_variant_identity(params),
        parameter_variant_label="canonical",
        **_FINGERPRINTS,
    )


def _result(identity: str, *, verdict: RegistryVerdict = RegistryVerdict.REJECT) -> ResultRecord:
    return ResultRecord(
        experiment_identity=identity, headline_verdict=verdict,
        reason_codes=("fdr_qvalue_above_threshold",), net_pnl_usd=1.0, daily_sharpe=0.01,
    )


@pytest.fixture
def registry(tmp_path):
    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reg:
        yield reg


# ==========================================================================
# P1 -- ExperimentRecord requires asset_domain
# ==========================================================================
def test_P1_experiment_record_requires_asset_domain():
    fields = {
        "experiment_identity": _identity(root="NQ"),
        "experiment_id": "NQ__TSMOM__CANONICAL__TEST",
        "display_name": "x",
        "created_at": "2026-01-01T00:00:00+00:00",
        "phase": "TEST",
        "status": ExperimentStatus.COMPLETED,
        "root_symbol": "NQ",
        "strategy_family": "tsmom",
        "strategy_fingerprint": "stratdsl1:tsmom:NQ",
        "strategy_id": "S",
        "strategy_spec_json": {"params": {}},
        "market_window": WINDOW,
        "trial_role": TrialRole.CANONICAL,
        "parameter_variant_identity": "paramvariant1:x",
        "parameter_variant_label": "canonical",
        **_FINGERPRINTS,
    }
    # no asset_domain at all -> a Pydantic validation error, not a default
    with pytest.raises(ValidationError):
        ExperimentRecord(**fields)
    # with it, construction succeeds
    ExperimentRecord(asset_domain=AssetDomain.FUTURES, **fields)


def test_P1_asset_domain_field_has_no_default():
    """The field itself carries no default -- every call site must say it."""
    field = ExperimentRecord.model_fields["asset_domain"]
    assert field.is_required(), "asset_domain must be a REQUIRED field, never defaulted"


# ==========================================================================
# P2 -- experiment_identity() is domain-blind
# ==========================================================================
def test_P2_identity_formula_has_no_asset_domain_parameter():
    sig = inspect.signature(experiment_identity)
    assert "asset_domain" not in sig.parameters, (
        "experiment_identity() must never take asset_domain -- domain "
        "namespacing is a storage/query concern, never an identity input"
    )


def test_P2_identical_prerun_inputs_yield_identical_identity_regardless_of_domain():
    """Two callers building records for DIFFERENT domains, but with every one
    of experiment_identity()'s actual parameters identical, get the SAME
    identity -- proving the function itself does not know or care about
    domain. (In practice ETF and Futures records never share every one of
    these inputs -- the dataset/execution-config fingerprints differ
    structurally -- but the identity FORMULA's blindness to domain is what
    this proves.)"""
    kwargs = dict(
        strategy_fingerprint="stratdsl1:shared",
        strategy_family="tsmom",
        root_symbol="SHARED",
        parameter_variant_identity=parameter_variant_identity({"n": 1}),
        **_FINGERPRINTS,
    )
    a = experiment_identity(**kwargs)
    b = experiment_identity(**kwargs)
    assert a == b
    # and building a FUTURES vs an ETF ExperimentRecord around that SAME
    # identity does not change the stored identity value either.
    futures_rec = ExperimentRecord(
        experiment_identity=a, experiment_id="A", display_name="a",
        created_at="2026-01-01T00:00:00+00:00", phase="TEST",
        status=ExperimentStatus.COMPLETED, root_symbol="SHARED",
        asset_domain=AssetDomain.FUTURES, strategy_family="tsmom",
        strategy_fingerprint="stratdsl1:shared", strategy_id="S",
        strategy_spec_json={"params": {"n": 1}}, market_window=WINDOW,
        trial_role=TrialRole.CANONICAL,
        parameter_variant_identity=parameter_variant_identity({"n": 1}),
        parameter_variant_label="canonical", **_FINGERPRINTS,
    )
    etf_rec = futures_rec.model_copy(update={
        "asset_domain": AssetDomain.ETF, "experiment_id": "B",
    })
    assert futures_rec.experiment_identity == etf_rec.experiment_identity == a


def test_P2_identity_module_source_has_no_diff():
    """Belt and suspenders: identity.py is imported unmodified (checked by
    the caller's `git diff`, this just proves the schema constant is intact)."""
    from alpha_agent.registry.identity import IDENTITY_SCHEMA
    from alpha_agent.registry.schema import IDENTITY_SCHEMA_VERSION

    assert IDENTITY_SCHEMA_VERSION == 2
    assert IDENTITY_SCHEMA == "experiment-identity/2"


# ==========================================================================
# P3 -- structural cross-domain isolation
# ==========================================================================
def test_P3_experiments_query_never_pools_domains_on_a_shared_family_label(registry):
    fut = _experiment(root="NQ", asset_domain=AssetDomain.FUTURES, family="tsmom")
    etf = _experiment(root="QQQ", asset_domain=AssetDomain.ETF, family="tsmom")
    registry.insert_experiment(fut, _result(fut.experiment_identity))
    registry.insert_experiment(etf, _result(etf.experiment_identity))

    both = registry.experiments(strategy_family="tsmom", authoritative_only=True)
    assert {v.experiment.root_symbol for v in both} == {"NQ", "QQQ"}

    futures_only = registry.experiments(
        strategy_family="tsmom", asset_domain=AssetDomain.FUTURES, authoritative_only=True
    )
    assert {v.experiment.root_symbol for v in futures_only} == {"NQ"}
    assert all(v.experiment.asset_domain is AssetDomain.FUTURES for v in futures_only)

    etf_only = registry.experiments(
        strategy_family="tsmom", asset_domain=AssetDomain.ETF, authoritative_only=True
    )
    assert {v.experiment.root_symbol for v in etf_only} == {"QQQ"}
    assert all(v.experiment.asset_domain is AssetDomain.ETF for v in etf_only)


def test_P3_find_related_never_returns_the_other_domains_evidence(registry):
    """The concrete proof the task demands: a synthetic same-strategy_family
    experiment under FUTURES and one under ETF; find_related queried for one
    domain must NEVER surface the other -- structural, not incidental."""
    fut = _experiment(
        root="NQ", asset_domain=AssetDomain.FUTURES, family="tsmom",
        params={"fast_horizon": 20, "slow_horizon": 120, "size": 1},
    )
    etf = _experiment(
        root="QQQ", asset_domain=AssetDomain.ETF, family="tsmom",
        # deliberately near-identical params -- if isolation were merely a
        # similarity-score effect rather than structural, this would still
        # score highly and could leak through.
        params={"fast_horizon": 21, "slow_horizon": 120, "size": 1},
    )
    registry.insert_experiment(fut, _result(fut.experiment_identity))
    registry.insert_experiment(etf, _result(etf.experiment_identity))

    hits_futures = registry.find_related(
        strategy_family="tsmom", root_symbol="NQ", asset_domain=AssetDomain.FUTURES,
        params={"fast_horizon": 20, "slow_horizon": 120, "size": 1},
    )
    ids = {h.experiment_id for h in hits_futures}
    assert fut.experiment_id in ids
    assert etf.experiment_id not in ids, "ETF evidence leaked into a FUTURES-scoped query"

    hits_etf = registry.find_related(
        strategy_family="tsmom", root_symbol="QQQ", asset_domain=AssetDomain.ETF,
        params={"fast_horizon": 21, "slow_horizon": 120, "size": 1},
    )
    ids = {h.experiment_id for h in hits_etf}
    assert etf.experiment_id in ids
    assert fut.experiment_id not in ids, "FUTURES evidence leaked into an ETF-scoped query"


def test_P3_failure_memory_lookup_never_pools_domains(registry):
    from alpha_agent.registry.failure_memory import FailureMemory

    fut = _experiment(root="NQ", asset_domain=AssetDomain.FUTURES, family="tsmom")
    etf = _experiment(root="QQQ", asset_domain=AssetDomain.ETF, family="tsmom")
    registry.insert_experiment(fut, _result(fut.experiment_identity, verdict=RegistryVerdict.REJECT))
    registry.insert_experiment(etf, _result(etf.experiment_identity, verdict=RegistryVerdict.PASS))

    fm = FailureMemory(registry)
    futures_resp = fm.lookup(
        strategy_family="tsmom", root_symbol="NQ", asset_domain=AssetDomain.FUTURES
    )
    assert {p.experiment_id for p in futures_resp.prior_experiments} == {fut.experiment_id}
    assert futures_resp.verdict_counts == {"REJECT": 1}

    etf_resp = fm.lookup(
        strategy_family="tsmom", root_symbol="QQQ", asset_domain=AssetDomain.ETF
    )
    assert {p.experiment_id for p in etf_resp.prior_experiments} == {etf.experiment_id}
    assert etf_resp.verdict_counts == {"PASS": 1}


def test_P3_relevant_failure_memory_sweep_is_domain_scoped(registry):
    from alpha_agent.registry.failure_memory import relevant_failure_memory

    # Same root string AND family label on purpose (the scenario this test
    # exists to prove isolates correctly) -- but experiment_identity does NOT
    # include asset_domain by design (see registry.enums.AssetDomain's
    # docstring), so two REAL experiments sharing root+family must still
    # differ in some real config input, exactly like real Futures vs ETF
    # configs always do (different dataset/execution assumptions). A
    # distinguishing params key stands in for that here.
    fut = _experiment(root="NQ", asset_domain=AssetDomain.FUTURES, family="tsmom")
    etf = _experiment(
        root="NQ", asset_domain=AssetDomain.ETF, family="tsmom",
        params={"fast_horizon": 20, "slow_horizon": 120, "size": 1, "etf_marker": 1},
        experiment_id="NQ__TSMOM__CANONICAL__TEST__ETF",
    )
    registry.insert_experiment(fut, _result(fut.experiment_identity))
    registry.insert_experiment(etf, _result(etf.experiment_identity))

    futures_sweep = relevant_failure_memory(
        registry, market_universe=("NQ",), asset_domain=AssetDomain.FUTURES
    )
    assert len(futures_sweep) == 1
    assert {p.experiment_id for p in futures_sweep[0].prior_experiments} == {fut.experiment_id}

    etf_sweep = relevant_failure_memory(
        registry, market_universe=("NQ",), asset_domain=AssetDomain.ETF
    )
    assert len(etf_sweep) == 1
    assert {p.experiment_id for p in etf_sweep[0].prior_experiments} == {etf.experiment_id}


# ==========================================================================
# P4 -- find_exact_duplicate cross-domain defense in depth
# ==========================================================================
def test_P4_find_exact_duplicate_raises_on_a_cross_domain_hit(registry):
    fut = _experiment(root="NQ", asset_domain=AssetDomain.FUTURES)
    registry.insert_experiment(fut, _result(fut.experiment_identity))

    # the correct domain finds it normally
    hit = registry.find_exact_duplicate(fut.experiment_identity, asset_domain=AssetDomain.FUTURES)
    assert hit.exists

    # the SAME identity, queried under the WRONG domain, is a loud integrity
    # error -- never silently "not found" and never silently returned.
    with pytest.raises(AssetDomainMismatch):
        registry.find_exact_duplicate(fut.experiment_identity, asset_domain=AssetDomain.ETF)


# ==========================================================================
# P5 -- schema v5 -> v6 migration (synthetic fixture)
# ==========================================================================
_V5_DDL = """
CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE experiments (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_identity TEXT NOT NULL UNIQUE, identity_schema TEXT NOT NULL,
    experiment_id TEXT NOT NULL UNIQUE, display_name TEXT NOT NULL,
    created_at TEXT NOT NULL, phase TEXT NOT NULL, status TEXT NOT NULL,
    code_commit TEXT NOT NULL DEFAULT '', root_symbol TEXT NOT NULL,
    strategy_family TEXT NOT NULL, strategy_fingerprint TEXT NOT NULL,
    strategy_id TEXT NOT NULL, strategy_spec_json TEXT NOT NULL,
    feature_spec_fingerprint TEXT, target_schedule_hash TEXT,
    candidate_manifest_fingerprint TEXT, dataset_fingerprint TEXT NOT NULL,
    split_identity TEXT NOT NULL, market_window_json TEXT NOT NULL,
    validation_spec_fingerprint TEXT NOT NULL,
    reliability_policy_fingerprint TEXT NOT NULL,
    execution_config_identity TEXT NOT NULL, cost_config_identity TEXT NOT NULL,
    risk_identity TEXT NOT NULL, trial_role TEXT NOT NULL,
    canonical_or_neighbour TEXT NOT NULL, parameter_variant_identity TEXT NOT NULL,
    parameter_variant_label TEXT NOT NULL, parent_experiment_identity TEXT,
    report_fingerprint TEXT, notes TEXT NOT NULL DEFAULT '',
    content_fingerprint TEXT NOT NULL, record_json TEXT NOT NULL
);
CREATE INDEX ix_experiments_family_root ON experiments (strategy_family, root_symbol);
CREATE INDEX ix_experiments_strategy_fp ON experiments (strategy_fingerprint);

CREATE TABLE execution_attempts (
    attempt_id TEXT PRIMARY KEY,
    experiment_identity TEXT NOT NULL REFERENCES experiments (experiment_identity),
    identity_schema TEXT NOT NULL, attempt_ordinal INTEGER NOT NULL,
    attempt_status TEXT NOT NULL, invalidation_class TEXT,
    invalidation_detail TEXT NOT NULL DEFAULT '',
    invalidation_evidence_json TEXT NOT NULL DEFAULT '{}',
    defect_resolution_commit TEXT, code_commit TEXT NOT NULL DEFAULT '',
    engine TEXT NOT NULL DEFAULT '', target_schedule_hash TEXT,
    report_fingerprint TEXT, source_artifact TEXT, source_artifact_sha256 TEXT,
    created_at TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '',
    content_fingerprint TEXT NOT NULL, record_json TEXT NOT NULL,
    UNIQUE (experiment_identity, attempt_ordinal)
);
CREATE INDEX ix_attempts_identity ON execution_attempts (experiment_identity);
CREATE INDEX ix_attempts_status ON execution_attempts (attempt_status);

CREATE TABLE attempt_results (
    attempt_id TEXT PRIMARY KEY REFERENCES execution_attempts (attempt_id),
    experiment_identity TEXT NOT NULL REFERENCES experiments (experiment_identity),
    attempt_ordinal INTEGER NOT NULL, headline_verdict TEXT NOT NULL,
    reason_codes_json TEXT NOT NULL, gross_pnl_usd REAL, costs_usd REAL,
    net_pnl_usd REAL, daily_sharpe REAL, annualized_sharpe REAL,
    gating_null_p REAL, bh_p_value REAL, bh_q REAL, bh_rejected_at_q INTEGER,
    dsr_probability REAL, fold_consistency REAL, n_trades INTEGER,
    n_fills INTEGER, n_oos_days INTEGER, holdout_eligible INTEGER NOT NULL,
    evidence_completeness TEXT NOT NULL, source_artifact TEXT,
    source_artifact_sha256 TEXT, result_fingerprint TEXT NOT NULL,
    record_json TEXT NOT NULL
);
CREATE INDEX ix_attempt_results_identity ON attempt_results (experiment_identity);

CREATE TABLE attempt_lineage (
    source_attempt_id TEXT NOT NULL REFERENCES execution_attempts (attempt_id),
    target_attempt_id TEXT NOT NULL REFERENCES execution_attempts (attempt_id),
    relation_type TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (source_attempt_id, target_attempt_id, relation_type)
);
CREATE INDEX ix_attempt_lineage_target ON attempt_lineage (target_attempt_id);

CREATE TABLE failures (
    failure_id TEXT PRIMARY KEY, scope TEXT NOT NULL,
    experiment_identity TEXT REFERENCES experiments (experiment_identity),
    attempt_id TEXT REFERENCES execution_attempts (attempt_id),
    failure_class TEXT NOT NULL, failure_code TEXT NOT NULL, summary TEXT NOT NULL,
    mechanism TEXT NOT NULL, evidence_json TEXT NOT NULL, action_taken TEXT NOT NULL,
    resolved INTEGER NOT NULL, resolution_commit TEXT, superseded_by TEXT,
    created_at TEXT NOT NULL, root_symbol TEXT, strategy_family TEXT,
    content_fingerprint TEXT NOT NULL
);
CREATE INDEX ix_failures_class ON failures (failure_class);
CREATE INDEX ix_failures_experiment ON failures (experiment_identity);
CREATE INDEX ix_failures_attempt ON failures (attempt_id);

CREATE TABLE lineage (
    source_experiment_identity TEXT NOT NULL REFERENCES experiments (experiment_identity),
    target_experiment_identity TEXT NOT NULL REFERENCES experiments (experiment_identity),
    relation_type TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (source_experiment_identity, target_experiment_identity, relation_type)
);
CREATE INDEX ix_lineage_target ON lineage (target_experiment_identity);

CREATE TABLE sensitivity_evidence (
    evidence_id TEXT PRIMARY KEY,
    experiment_identity TEXT NOT NULL REFERENCES experiments (experiment_identity),
    relation_type TEXT NOT NULL, kind TEXT NOT NULL, baseline_verdict TEXT NOT NULL,
    rerun_verdict TEXT NOT NULL, verdict_changed INTEGER NOT NULL,
    metrics_json TEXT NOT NULL, in_bh_fdr_denominator INTEGER NOT NULL
);

CREATE TABLE cross_market_evidence (
    evidence_id TEXT PRIMARY KEY, strategy_family TEXT NOT NULL, status TEXT NOT NULL,
    n_roots INTEGER NOT NULL, n_positive_roots INTEGER NOT NULL,
    max_single_root_pnl_share REAL NOT NULL, herfindahl_pnl REAL NOT NULL,
    concentrated_in_one_root INTEGER NOT NULL, referenced_identities_json TEXT NOT NULL
);

CREATE TABLE imports (
    import_id TEXT PRIMARY KEY, phase TEXT NOT NULL, source_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL, counts_json TEXT NOT NULL, metadata_json TEXT NOT NULL
);
"""


def _build_v5_fixture(path: Path, *, n: int = 3) -> list[str]:
    """A hand-built, genuinely v5-shaped registry (no asset_domain column),
    stamped schema_version=5 -- exactly the shape a real pre-Phase-6
    ``data/registry/experiments.sqlite`` has today. Returns the identities
    inserted."""
    import json as _json

    conn = sqlite3.connect(path)
    conn.executescript(_V5_DDL)
    conn.execute("INSERT INTO registry_meta VALUES ('schema_version', '5')")

    identities: list[str] = []
    for i in range(n):
        root = f"R{i}"
        ident = _identity(root=root)
        exp = _experiment(root=root, asset_domain=AssetDomain.FUTURES)
        # a v5 row's record_json has NO asset_domain key at all.
        record = exp.model_dump(mode="json")
        record.pop("asset_domain")
        conn.execute(
            "INSERT INTO experiments (experiment_identity, identity_schema, experiment_id, "
            "display_name, created_at, phase, status, code_commit, root_symbol, "
            "strategy_family, strategy_fingerprint, strategy_id, strategy_spec_json, "
            "feature_spec_fingerprint, target_schedule_hash, candidate_manifest_fingerprint, "
            "dataset_fingerprint, split_identity, market_window_json, "
            "validation_spec_fingerprint, reliability_policy_fingerprint, "
            "execution_config_identity, cost_config_identity, risk_identity, trial_role, "
            "canonical_or_neighbour, parameter_variant_identity, parameter_variant_label, "
            "parent_experiment_identity, report_fingerprint, notes, content_fingerprint, "
            "record_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ident, exp.identity_schema, exp.experiment_id, exp.display_name,
                exp.created_at, exp.phase, exp.status.value, exp.code_commit, root,
                exp.strategy_family, exp.strategy_fingerprint, exp.strategy_id,
                _json.dumps(exp.strategy_spec_json), exp.feature_spec_fingerprint,
                exp.target_schedule_hash, exp.candidate_manifest_fingerprint,
                exp.dataset_fingerprint, exp.split_identity,
                _json.dumps(exp.market_window.model_dump(mode="json")),
                exp.validation_spec_fingerprint, exp.reliability_policy_fingerprint,
                exp.execution_config_identity, exp.cost_config_identity, exp.risk_identity,
                exp.trial_role.value, exp.canonical_or_neighbour,
                exp.parameter_variant_identity, exp.parameter_variant_label,
                exp.parent_experiment_identity, exp.report_fingerprint, exp.notes,
                f"expcontent1:fixture:{ident}", _json.dumps(record),
            ),
        )
        attempt_id = f"attempt1:fixture:{ident}"
        attempt_record = {
            "attempt_id": attempt_id, "experiment_identity": ident,
            "identity_schema": exp.identity_schema, "attempt_ordinal": 1,
            "attempt_status": "VALID", "code_commit": "", "engine": "",
            "created_at": exp.created_at,
        }
        conn.execute(
            "INSERT INTO execution_attempts (attempt_id, experiment_identity, "
            "identity_schema, attempt_ordinal, attempt_status, code_commit, engine, "
            "created_at, content_fingerprint, record_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (attempt_id, ident, exp.identity_schema, 1, "VALID", "", "",
             exp.created_at, f"execattempt1:fixture:{ident}", _json.dumps(attempt_record)),
        )
        result = _result(ident)
        conn.execute(
            "INSERT INTO attempt_results (attempt_id, experiment_identity, attempt_ordinal, "
            "headline_verdict, reason_codes_json, net_pnl_usd, daily_sharpe, "
            "holdout_eligible, evidence_completeness, result_fingerprint, record_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (attempt_id, ident, 1, result.headline_verdict.value,
             _json.dumps(list(result.reason_codes)), result.net_pnl_usd,
             result.daily_sharpe, 0, "full", result.result_fingerprint(),
             _json.dumps(result.model_dump(mode="json"))),
        )
        identities.append(ident)
    conn.commit()
    conn.close()
    return identities


def test_P5_migration_preserves_identities_and_sets_asset_domain(tmp_path):
    db = tmp_path / "v5_fixture.sqlite"
    identities = _build_v5_fixture(db, n=3)

    with ExperimentRegistry(db) as reg:
        assert reg.schema_version == SCHEMA_VERSION == 7
        # 1. every experiment_identity preserved byte-for-byte; nothing
        #    dropped, nothing duplicated.
        rows = reg._conn.execute(
            "SELECT experiment_identity, asset_domain FROM experiments"
        ).fetchall()
        assert len(rows) == len(identities) == 3
        assert {r[0] for r in rows} == set(identities)
        # 2. every migrated row's asset_domain is the hardcoded fact.
        assert {r[1] for r in rows} == {"FUTURES"}
        # 3. every row is readable as a real ExperimentRecord (record_json
        #    round-trips through the model that now requires asset_domain).
        for ident in identities:
            view = reg.get(ident)
            assert view.experiment.asset_domain is AssetDomain.FUTURES
            assert view.experiment.experiment_identity == ident
        # 4. the v5 representation is preserved verbatim, not dropped.
        legacy_rows = reg._conn.execute(
            "SELECT experiment_identity FROM experiments_v5_legacy"
        ).fetchall()
        assert {r[0] for r in legacy_rows} == set(identities)

        # 5. THE FK-CASCADE REGRESSION: a brand-new experiment (never in the
        #    v5 legacy snapshot) must be insertable, WITH an execution attempt,
        #    with no foreign-key failure. This is exactly the bug a naive
        #    "just rename experiments, leave execution_attempts alone"
        #    migration would introduce (SQLite's ALTER TABLE RENAME silently
        #    rewrites execution_attempts's REFERENCES clause to follow the
        #    renamed experiments_v5_legacy table).
        new_exp = _experiment(root="NEWROOT", asset_domain=AssetDomain.FUTURES)
        view = reg.insert_experiment(new_exp, _result(new_exp.experiment_identity))
        assert view.result is not None
        assert view.result.headline_verdict is RegistryVerdict.REJECT
        # and its FK actually resolves against the LIVE experiments table, not
        # the legacy snapshot (a stray FK would have raised on insert already,
        # but re-assert this explicitly via a direct fk_check).
        violations = reg._conn.execute("PRAGMA foreign_key_check").fetchall()
        assert violations == []


def test_P5_migration_from_v5_is_a_no_op_when_already_current(tmp_path):
    """Opening an already-v6 database a second time changes nothing."""
    db = tmp_path / "v5_fixture.sqlite"
    identities = _build_v5_fixture(db, n=2)
    with ExperimentRegistry(db):
        pass
    with ExperimentRegistry(db) as reg:
        assert reg.schema_version == 7
        rows = reg._conn.execute("SELECT experiment_identity FROM experiments").fetchall()
        assert {r[0] for r in rows} == set(identities)
        assert len(rows) == 2  # not duplicated by the second open


# ==========================================================================
# P6 -- the REAL committed local registry (copy only, original untouched)
# ==========================================================================
@pytest.mark.skipif(
    not REAL_REGISTRY.exists(), reason="no local data/registry/experiments.sqlite in this checkout"
)
def test_P6_real_registry_migrates_with_every_identity_unchanged(tmp_path):
    """Copy the REAL registry, migrate the COPY, and diff experiment_identity
    values old vs new. The original file is opened strictly read-only here
    and is never written to by this test."""
    # 1. read the pre-migration identity set from the ORIGINAL, read-only.
    ro_conn = sqlite3.connect(f"file:{REAL_REGISTRY}?mode=ro", uri=True)
    try:
        pre_version = ro_conn.execute(
            "SELECT value FROM registry_meta WHERE key='schema_version'"
        ).fetchone()[0]
        pre_identities = {
            r[0] for r in ro_conn.execute("SELECT experiment_identity FROM experiments")
        }
        pre_domains = dict(
            ro_conn.execute(
                "SELECT asset_domain, COUNT(*) FROM experiments GROUP BY asset_domain"
            ).fetchall()
        )
    finally:
        ro_conn.close()
    assert pre_identities, "the real registry has no experiment rows to check"

    # 2. migrate a COPY, never the original.
    copy_path = tmp_path / "real_registry_copy.sqlite"
    shutil.copyfile(REAL_REGISTRY, copy_path)
    with ExperimentRegistry(copy_path) as reg:
        assert reg.schema_version == SCHEMA_VERSION == 7
        post_identities = {
            r[0] for r in reg._conn.execute("SELECT experiment_identity FROM experiments")
        }
        domains = dict(
            reg._conn.execute(
                "SELECT asset_domain, COUNT(*) FROM experiments GROUP BY asset_domain"
            ).fetchall()
        )

    # 3. the single most important invariant: byte-for-byte identical.
    assert post_identities == pre_identities, (
        f"identity set changed across migration: "
        f"{len(pre_identities - post_identities)} dropped, "
        f"{len(post_identities - pre_identities)} added"
    )
    # asset_domain composition is untouched by migration -- whatever real
    # experiments (Futures and/or ETF) exist pre-migration must exist
    # post-migration in the same counts. Not hardcoded to Futures-only: the
    # real registry legitimately gained ETF rows in Phase 6 and may gain more
    # asset domains later without this invariant becoming stale.
    assert domains == pre_domains

    # 4. the ORIGINAL file is untouched: still stamped at its pre-migration
    #    version, same row count.
    ro_conn = sqlite3.connect(f"file:{REAL_REGISTRY}?mode=ro", uri=True)
    try:
        still_version = ro_conn.execute(
            "SELECT value FROM registry_meta WHERE key='schema_version'"
        ).fetchone()[0]
        still_identities = {
            r[0] for r in ro_conn.execute("SELECT experiment_identity FROM experiments")
        }
    finally:
        ro_conn.close()
    assert still_version == pre_version, "the REAL file must never be migrated by a test"
    assert still_identities == pre_identities
