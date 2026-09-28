"""Explicit, versioned SQLite schema for the experiment registry (section 21).

Migrations are ordered and deterministic. A database stamped with a version this
code does not know about is a hard error -- the registry never silently drops or
recreates a researcher's experiment history.

Schema versions
---------------
1
    the pre-Phase-14 flat ``experiments`` table (single table, ``INSERT OR
    REPLACE``). Recognised only so an existing file can be migrated; the
    replace-in-place behaviour is gone.
2
    Phase 14: normalized experiment / result / failure / lineage model with
    append-only semantics and derived authority. Experiment identity schema v1
    (identity included ``target_schedule_hash``).
3
    Phase 14.1: experiment identity schema **v2** -- identity is a PRE-RUN
    quantity, so ``target_schedule_hash`` and ``report_fingerprint`` moved out
    of identity and are recorded as provenance. Adds
    ``experiments.identity_schema`` and ``results.source_artifact*``.
4
    Phase 14.2: a parameter neighbour's ``feature_spec_fingerprint`` and stored
    feature fingerprints are its OWN reconstructed ``StrategySpec``'s, not the
    canonical trial's. The identity *formula* is unchanged (identity schema
    stays v2); the imported feature-spec input was wrong, so the stored
    identities were wrong. Table shape is unchanged from v3.
6
    Phase 6 (ETF Research Pilot): adds ``experiments.asset_domain`` (a
    :class:`alpha_agent.core.instrument.AssetDomain` value, e.g. ``"FUTURES"``
    or ``"ETF"``) plus a composite index ``(asset_domain, strategy_family,
    root_symbol)``. This is a STORAGE/QUERY-layer guard only -- it is never an
    input to ``experiment_identity()`` (that formula is completely unchanged;
    the dataset/execution-config fingerprints already differ structurally
    between domains). Every one of today's 167 real rows is a known Futures
    experiment, so the v5->v6 forward copy hardcodes ``asset_domain='FUTURES'``
    for every migrated row -- a stated fact about what is being migrated, not
    an inference. Table shape is otherwise unchanged from v5.
7
    News Alpha Phase H: adds the append-only ``signal_path_evidence`` table --
    typed hypothesis-plane outcomes (event -> transmission path -> expression
    -> measurement -> candidate signal -> portfolio), linked by id to an
    experiment when one exists (Alpha Memory reads it). PURELY ADDITIVE: no
    existing table is renamed, copied or altered, no experiment identity or
    result changes, and the content digest of a registry whose new table is
    empty is byte-identical to v6's.

Every migration in this chain is **preservation only**: it renames the outgoing
representation to ``<name>_v<n>_legacy`` and creates nothing. The current DDL is
then applied (every statement is ``IF NOT EXISTS``), and the deterministic
importer repopulates from committed artifacts. Nothing is ever dropped, so a
superseded representation -- including the defects it contained -- stays
inspectable.
"""
from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 7
LEGACY_SCHEMA_VERSION = 1
#: The identity FORMULA version. Unchanged by schema v4/v5/v6 -- v5 added an
#: execution-ATTEMPT layer beneath the identity and v6 adds a storage-layer
#: ``asset_domain`` column; the identity formula and every stored
#: ``experiment_identity`` value are byte-for-byte unchanged by either.
IDENTITY_SCHEMA_VERSION = 2

#: Registry tables preserved verbatim by each migration. ``execution_attempts``
#: / ``attempt_results`` / ``attempt_lineage`` (the schema-v5 attempt layer)
#: are included from v5 onward -- NOT because their own DDL shape changes in
#: v6, but because SQLite's ``ALTER TABLE ... RENAME`` rewrites any OTHER
#: table's ``REFERENCES`` clause to follow the renamed table (proven in this
#: phase's migration tests). If ``experiments`` is renamed aside while
#: ``execution_attempts`` is left in place, ``execution_attempts``'s FK is
#: silently rewritten to point at ``experiments_v5_legacy`` -- so a fresh
#: experiment inserted into the NEW ``experiments`` table after the migration
#: could never get a valid ``execution_attempts`` row (FK constraint failure).
#: Renaming all four together and letting the current DDL recreate them fresh
#: is the only way every FK ends up pointing at the live tables.
_REGISTRY_TABLES = (
    "experiments", "results", "failures", "lineage", "sensitivity_evidence",
    "cross_market_evidence", "imports",
    "execution_attempts", "attempt_results", "attempt_lineage",
)
#: Index names that follow their table through a rename and must be freed
#: before the current DDL can recreate them.
_INDEX_NAMES = (
    "ix_experiments_family_root", "ix_experiments_domain_family_root",
    "ix_experiments_strategy_fp",
    "ix_results_verdict", "ix_failures_class", "ix_failures_experiment",
    "ix_failures_attempt", "ix_lineage_target",
    "ix_attempts_identity", "ix_attempts_status", "ix_attempt_results_identity",
    "ix_attempt_lineage_target",
)

_CURRENT_DDL = """
CREATE TABLE IF NOT EXISTS registry_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    row_id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_identity           TEXT NOT NULL UNIQUE,
    identity_schema               TEXT NOT NULL,
    experiment_id                 TEXT NOT NULL UNIQUE,
    display_name                  TEXT NOT NULL,
    created_at                    TEXT NOT NULL,
    phase                         TEXT NOT NULL,
    status                        TEXT NOT NULL,
    code_commit                   TEXT NOT NULL DEFAULT '',
    root_symbol                   TEXT NOT NULL,
    asset_domain                  TEXT NOT NULL,
    strategy_family               TEXT NOT NULL,
    strategy_fingerprint          TEXT NOT NULL,
    strategy_id                   TEXT NOT NULL,
    strategy_spec_json            TEXT NOT NULL,
    feature_spec_fingerprint      TEXT,
    target_schedule_hash          TEXT,
    candidate_manifest_fingerprint TEXT,
    dataset_fingerprint           TEXT NOT NULL,
    split_identity                TEXT NOT NULL,
    market_window_json            TEXT NOT NULL,
    validation_spec_fingerprint   TEXT NOT NULL,
    reliability_policy_fingerprint TEXT NOT NULL,
    execution_config_identity     TEXT NOT NULL,
    cost_config_identity          TEXT NOT NULL,
    risk_identity                 TEXT NOT NULL,
    trial_role                    TEXT NOT NULL,
    canonical_or_neighbour        TEXT NOT NULL,
    parameter_variant_identity    TEXT NOT NULL,
    parameter_variant_label       TEXT NOT NULL,
    parent_experiment_identity    TEXT,
    report_fingerprint            TEXT,
    notes                         TEXT NOT NULL DEFAULT '',
    content_fingerprint           TEXT NOT NULL,
    record_json                   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_experiments_family_root
    ON experiments (strategy_family, root_symbol);
-- schema v6: the structural query-time domain guard. Every family-grouping /
-- relatedness / BH-FDR-family query should key off this composite, not
-- (strategy_family, root_symbol) alone -- ix_experiments_family_root is kept
-- (additive only) for any existing domain-blind lookup that still wants it.
CREATE INDEX IF NOT EXISTS ix_experiments_domain_family_root
    ON experiments (asset_domain, strategy_family, root_symbol);
CREATE INDEX IF NOT EXISTS ix_experiments_strategy_fp
    ON experiments (strategy_fingerprint);

-- schema v5: one immutable row per EXECUTION ATTEMPT of an experiment_identity.
-- The same identity may have several attempts when an earlier one was invalid
-- for an engineering / data-pipeline / software / infrastructure reason.
CREATE TABLE IF NOT EXISTS execution_attempts (
    attempt_id                 TEXT PRIMARY KEY,
    experiment_identity        TEXT NOT NULL REFERENCES experiments (experiment_identity),
    identity_schema            TEXT NOT NULL,
    attempt_ordinal            INTEGER NOT NULL,
    attempt_status             TEXT NOT NULL,          -- 'VALID' | 'INVALID_EXECUTION'
    invalidation_class         TEXT,                   -- NULL iff VALID
    invalidation_detail        TEXT NOT NULL DEFAULT '',
    invalidation_evidence_json TEXT NOT NULL DEFAULT '{}',
    defect_resolution_commit   TEXT,
    code_commit                TEXT NOT NULL DEFAULT '',
    engine                     TEXT NOT NULL DEFAULT '',
    target_schedule_hash       TEXT,
    report_fingerprint         TEXT,
    source_artifact            TEXT,
    source_artifact_sha256     TEXT,
    created_at                 TEXT NOT NULL,
    notes                      TEXT NOT NULL DEFAULT '',
    content_fingerprint        TEXT NOT NULL,
    record_json                TEXT NOT NULL,
    UNIQUE (experiment_identity, attempt_ordinal)
);
CREATE INDEX IF NOT EXISTS ix_attempts_identity ON execution_attempts (experiment_identity);
CREATE INDEX IF NOT EXISTS ix_attempts_status ON execution_attempts (attempt_status);

-- the adjudicated scientific outcome of ONE attempt. Only a VALID attempt's
-- row is ever the authoritative result of its identity.
CREATE TABLE IF NOT EXISTS attempt_results (
    attempt_id            TEXT PRIMARY KEY REFERENCES execution_attempts (attempt_id),
    experiment_identity   TEXT NOT NULL REFERENCES experiments (experiment_identity),
    attempt_ordinal       INTEGER NOT NULL,
    headline_verdict      TEXT NOT NULL,
    reason_codes_json     TEXT NOT NULL,
    gross_pnl_usd         REAL,
    costs_usd             REAL,
    net_pnl_usd           REAL,
    daily_sharpe          REAL,
    annualized_sharpe     REAL,
    gating_null_p         REAL,
    bh_p_value            REAL,
    bh_q                  REAL,
    bh_rejected_at_q      INTEGER,
    dsr_probability       REAL,
    fold_consistency      REAL,
    n_trades              INTEGER,
    n_fills               INTEGER,
    n_oos_days            INTEGER,
    holdout_eligible      INTEGER NOT NULL,
    evidence_completeness TEXT NOT NULL,
    source_artifact       TEXT,
    source_artifact_sha256 TEXT,
    result_fingerprint    TEXT NOT NULL,
    record_json           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_attempt_results_identity ON attempt_results (experiment_identity);

CREATE TABLE IF NOT EXISTS attempt_lineage (
    source_attempt_id  TEXT NOT NULL REFERENCES execution_attempts (attempt_id),
    target_attempt_id  TEXT NOT NULL REFERENCES execution_attempts (attempt_id),
    relation_type      TEXT NOT NULL,
    note               TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (source_attempt_id, target_attempt_id, relation_type)
);
CREATE INDEX IF NOT EXISTS ix_attempt_lineage_target ON attempt_lineage (target_attempt_id);

CREATE TABLE IF NOT EXISTS failures (
    failure_id           TEXT PRIMARY KEY,
    scope                TEXT NOT NULL,
    experiment_identity  TEXT REFERENCES experiments (experiment_identity),
    attempt_id           TEXT REFERENCES execution_attempts (attempt_id),
    failure_class        TEXT NOT NULL,
    failure_code         TEXT NOT NULL,
    summary              TEXT NOT NULL,
    mechanism            TEXT NOT NULL,
    evidence_json        TEXT NOT NULL,
    action_taken         TEXT NOT NULL,
    resolved             INTEGER NOT NULL,
    resolution_commit    TEXT,
    superseded_by        TEXT,
    created_at           TEXT NOT NULL,
    root_symbol          TEXT,
    strategy_family      TEXT,
    content_fingerprint  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_failures_class ON failures (failure_class);
CREATE INDEX IF NOT EXISTS ix_failures_experiment ON failures (experiment_identity);
CREATE INDEX IF NOT EXISTS ix_failures_attempt ON failures (attempt_id);

CREATE TABLE IF NOT EXISTS lineage (
    source_experiment_identity TEXT NOT NULL
        REFERENCES experiments (experiment_identity),
    target_experiment_identity TEXT NOT NULL
        REFERENCES experiments (experiment_identity),
    relation_type              TEXT NOT NULL,
    note                       TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (source_experiment_identity, target_experiment_identity, relation_type)
);
CREATE INDEX IF NOT EXISTS ix_lineage_target ON lineage (target_experiment_identity);

CREATE TABLE IF NOT EXISTS sensitivity_evidence (
    evidence_id           TEXT PRIMARY KEY,
    experiment_identity   TEXT NOT NULL REFERENCES experiments (experiment_identity),
    relation_type         TEXT NOT NULL,
    kind                  TEXT NOT NULL,
    baseline_verdict      TEXT NOT NULL,
    rerun_verdict         TEXT NOT NULL,
    verdict_changed       INTEGER NOT NULL,
    metrics_json          TEXT NOT NULL,
    in_bh_fdr_denominator INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS cross_market_evidence (
    evidence_id                 TEXT PRIMARY KEY,
    strategy_family             TEXT NOT NULL,
    status                      TEXT NOT NULL,
    n_roots                     INTEGER NOT NULL,
    n_positive_roots            INTEGER NOT NULL,
    max_single_root_pnl_share   REAL NOT NULL,
    herfindahl_pnl              REAL NOT NULL,
    concentrated_in_one_root    INTEGER NOT NULL,
    referenced_identities_json  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS imports (
    import_id           TEXT PRIMARY KEY,
    phase               TEXT NOT NULL,
    source_fingerprint  TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    counts_json         TEXT NOT NULL,
    metadata_json       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signal_path_evidence (
    row_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_id           TEXT NOT NULL UNIQUE,
    research_run_id       TEXT NOT NULL,
    recorded_at           TEXT NOT NULL,
    event_id              TEXT NOT NULL,
    path_signature        TEXT,
    path_type             TEXT,
    candidate_signal_id   TEXT,
    experiment_identity   TEXT REFERENCES experiments(experiment_identity),
    stage_reached         TEXT NOT NULL,
    outcome               TEXT NOT NULL,
    evidence_scope        TEXT NOT NULL,
    reason_code           TEXT NOT NULL,
    record_json           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_spe_path ON signal_path_evidence(path_signature);
CREATE INDEX IF NOT EXISTS ix_spe_candidate ON signal_path_evidence(candidate_signal_id);
CREATE INDEX IF NOT EXISTS ix_spe_event ON signal_path_evidence(event_id);
CREATE INDEX IF NOT EXISTS ix_spe_run ON signal_path_evidence(research_run_id);
"""


class UnknownSchemaVersion(RuntimeError):
    """The database on disk is stamped with a schema version this build does not
    know. Refuse to touch it rather than guess a migration."""


def read_schema_version(conn: sqlite3.Connection) -> int | None:
    """Version stamped in the file, or ``None`` for an empty/new database."""
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "registry_meta" in tables:
        row = conn.execute(
            "SELECT value FROM registry_meta WHERE key='schema_version'"
        ).fetchone()
        if row is not None:
            return int(row[0])
    if "experiments" in tables:
        # a pre-Phase-14 database: the flat v1 table with no meta row.
        return LEGACY_SCHEMA_VERSION
    if not tables:
        return None
    raise UnknownSchemaVersion(
        f"registry database contains unrecognised tables {sorted(tables)} and no "
        "schema_version stamp; refusing to migrate"
    )


def _preserve(conn: sqlite3.Connection, suffix: str, tables: tuple[str, ...]) -> None:
    """Rename an outgoing representation aside, creating nothing.

    Indexes follow their table through ``ALTER TABLE ... RENAME``, so their
    names are freed here for the current DDL to recreate.
    """
    existing = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    for name in tables:
        if name in existing:
            conn.execute(f"ALTER TABLE {name} RENAME TO {name}_{suffix}")
    for idx in _INDEX_NAMES:
        conn.execute(f"DROP INDEX IF EXISTS {idx}")


def _migrate_1_to_2(conn: sqlite3.Connection) -> None:
    """Preserve the pre-Phase-14 flat table as ``experiments_v1_legacy``.

    Its shape (``INSERT OR REPLACE``, free-text status) cannot be mapped onto a
    typed append-only identity without inventing scientific content, so it is
    retained, not converted.
    """
    _preserve(conn, "v1_legacy", ("experiments",))


def _migrate_2_to_3(conn: sqlite3.Connection) -> None:
    """Preserve the whole v2 representation as ``*_v2_legacy``.

    v3 redefines what an ``experiment_identity`` *means* (schema v2: pre-run
    inputs only), so a v2 row's identity is not a v3 row's and cannot be
    rewritten in place without silently restating a scientific claim.
    """
    _preserve(conn, "v2_legacy", _REGISTRY_TABLES)


def _migrate_3_to_4(conn: sqlite3.Connection) -> None:
    """Preserve the whole v3 representation as ``*_v3_legacy``.

    v3 rows carry each neighbour's *canonical* feature-spec fingerprint, so
    their stored ``experiment_identity`` is not the neighbour's true identity.
    Recomputing in place would rewrite completed rows; the importer rebuilds
    instead, and the defective representation stays inspectable.
    """
    _preserve(conn, "v3_legacy", _REGISTRY_TABLES)


def _migrate_4_to_5(conn: sqlite3.Connection) -> None:
    """Preserve the whole v4 representation as ``*_v4_legacy``.

    v5 adds an execution-ATTEMPT layer beneath ``experiment_identity``: the same
    pre-run identity may carry several immutable execution attempts, and only a
    ``VALID`` attempt supplies the authoritative scientific result. The identity
    formula and every stored ``experiment_identity`` value are unchanged. The v4
    ``experiments`` / ``results`` / ``failures`` rows are re-projected into the
    v5 tables by :func:`_populate_v5_from_v4_legacy` AFTER the current DDL runs
    -- a deterministic, preservation-only forward copy (nothing is dropped or
    rewritten; the ``*_v4_legacy`` tables stay inspectable).
    """
    _preserve(conn, "v4_legacy", _REGISTRY_TABLES)


#: Every real row migrated by v5->v6 is a known Futures experiment -- a stated
#: fact about the reality being migrated (100% of the 167 real rows today), not
#: an inference from any other column. Mirrors how ``_populate_v5_from_v4_legacy``
#: hardcodes VALID/INVALID_EXECUTION attempt status from the known
#: ``phase_15b__`` import prefix rather than deriving it.
_MIGRATED_V5_ASSET_DOMAIN = "FUTURES"


def _migrate_5_to_6(conn: sqlite3.Connection) -> None:
    """Preserve the whole v5 representation as ``*_v5_legacy``.

    v6 adds ``experiments.asset_domain`` (Phase 6 ETF Research Pilot): a
    storage/query-layer column so ETF and Futures evidence can never be pooled
    by a family/relatedness/BH-FDR query merely because a strategy-family label
    matches. It is NOT an identity input -- ``experiment_identity()`` and every
    stored identity value are unchanged.

    ``_REGISTRY_TABLES`` now includes the schema-v5 attempt layer
    (``execution_attempts`` / ``attempt_results`` / ``attempt_lineage``), so
    this rename also carries them aside even though their own DDL shape is
    unchanged. That is required, not optional: SQLite's
    ``ALTER TABLE ... RENAME`` rewrites any OTHER table's ``REFERENCES`` clause
    to follow the renamed table, so renaming ``experiments`` alone would leave
    ``execution_attempts`` (etc.) pointing at ``experiments_v5_legacy`` forever
    -- silently breaking every FUTURE experiment's attempt insert after this
    migration. Renaming all four together and letting the current DDL recreate
    them fresh (with the FK correctly bound to the live tables) avoids that.
    The v5 rows are re-projected into the v6 tables by
    :func:`_populate_v6_from_v5_legacy` AFTER the current DDL runs -- a
    deterministic, preservation-only forward copy (nothing is dropped or
    rewritten; the ``*_v5_legacy`` tables stay inspectable).
    """
    _preserve(conn, "v5_legacy", _REGISTRY_TABLES)


#: A phase-15B production import whose 60 Phase-15 records were invalidated by an
#: engineering feature-pipeline defect (weekend/holiday wall-clock gaps tripped
#: SessionPolicy.RESET_ON_GAP, starving every window->5d feature to 100% NaN, so
#: every primary episode was dropped as FEATURES_MISSING_AT_DECISION and all 60
#: trials refused). Corrected by the cc206de feature path. On the v4->v5
#: forward-copy these attempts are recorded as INVALID_EXECUTION, not VALID.
_INVALIDATED_V4_IMPORT_PREFIX = "phase_15b__"
_INVALIDATION_CLASS = "FEATURE_PIPELINE_DEFECT"
_INVALIDATION_DETAIL = (
    "The real daily forward-adjusted signal series carried ~370 weekend/holiday "
    "wall-clock gaps; SessionPolicy.RESET_ON_GAP fragmented it into ~1-trading-week "
    "segments, so cum_log_return_20/120, realized_vol_20/60 and vol_percentile_20_252 "
    "were 100% non-finite. Every primary episode was dropped as "
    "FEATURES_MISSING_AT_DECISION and all 60 trials refused with "
    "INSUFFICIENT_TRAIN_EVENTS. Not a scientific result. Corrected feature path: "
    "alpha_agent.features.daily.compute_contiguous_daily_features."
)
_DEFECT_RESOLUTION_COMMIT = "cc206de"


#: version -> migration into version+1 (run BEFORE the current DDL)
def _migrate_6_to_7(conn: sqlite3.Connection) -> None:
    """Nothing to preserve: v7 only ADDS ``signal_path_evidence`` (created by
    the current DDL, ``IF NOT EXISTS``). Every v6 table and row is untouched."""


_MIGRATIONS = {
    LEGACY_SCHEMA_VERSION: _migrate_1_to_2,
    2: _migrate_2_to_3,
    3: _migrate_3_to_4,
    4: _migrate_4_to_5,
    5: _migrate_5_to_6,
    6: _migrate_6_to_7,
}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _copy_experiments_with_asset_domain(
    conn: sqlite3.Connection, *, legacy_table: str, asset_domain: str
) -> None:
    """Forward-copy every row of a LEGACY ``experiments`` representation into
    the CURRENT ``experiments`` table (schema v6+), injecting ``asset_domain``
    into both the new SQL column and the embedded ``record_json``.

    Shared by every populate function that lands rows on the current
    ``experiments`` shape, however many schema hops back its legacy table is
    (``experiments_v4_legacy`` in one ``ensure_schema()`` call that walks
    straight from v4 to v6, ``experiments_v5_legacy`` in the ordinary v5->v6
    step): ``_CURRENT_DDL`` always reflects the LATEST schema, so whichever
    populate function actually finds rows to copy (each guards on its own
    specific ``_v<n>_legacy`` table existing; in a single ``ensure_schema()``
    call exactly one such table is ever created for a given starting version,
    since an intermediate ``_migrate_X_to_Y`` preservation step is a no-op when
    there is nothing currently named ``experiments`` to rename) must write a
    row shaped for whatever ``experiments`` looks like NOW, not for the schema
    version that function was originally written against. Omitting this here
    is exactly the bug this phase's migration tests caught: a generic
    column-list verbatim copy into a table with a new NOT NULL column raises
    ``NOT NULL constraint failed: experiments.asset_domain``.

    ``record_json`` / ``content_fingerprint`` are regenerated from the SAME
    rebuilt :class:`~alpha_agent.registry.models.ExperimentRecord` (now
    carrying ``asset_domain``) so the stored JSON blob and its content
    fingerprint stay internally consistent with the current model shape --
    ``experiment_identity`` itself is never recomputed, only copied verbatim.
    """
    import json as _json

    from alpha_agent.registry.models import ExperimentRecord

    if not _table_exists(conn, legacy_table):
        return
    for row in conn.execute(
        f"SELECT * FROM {legacy_table} ORDER BY experiment_identity"
    ).fetchall():
        record = _json.loads(row["record_json"])
        record["asset_domain"] = asset_domain
        rebuilt = ExperimentRecord.model_validate(record)
        conn.execute(
            "INSERT INTO experiments ("
            " experiment_identity, identity_schema, experiment_id, display_name, created_at,"
            " phase, status, code_commit, root_symbol, asset_domain, strategy_family,"
            " strategy_fingerprint, strategy_id, strategy_spec_json, feature_spec_fingerprint,"
            " target_schedule_hash, candidate_manifest_fingerprint, dataset_fingerprint,"
            " split_identity, market_window_json, validation_spec_fingerprint,"
            " reliability_policy_fingerprint, execution_config_identity, cost_config_identity,"
            " risk_identity, trial_role, canonical_or_neighbour, parameter_variant_identity,"
            " parameter_variant_label, parent_experiment_identity, report_fingerprint, notes,"
            " content_fingerprint, record_json"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["experiment_identity"], row["identity_schema"], row["experiment_id"],
                row["display_name"], row["created_at"], row["phase"], row["status"],
                row["code_commit"], row["root_symbol"], asset_domain,
                row["strategy_family"], row["strategy_fingerprint"], row["strategy_id"],
                row["strategy_spec_json"], row["feature_spec_fingerprint"],
                row["target_schedule_hash"], row["candidate_manifest_fingerprint"],
                row["dataset_fingerprint"], row["split_identity"], row["market_window_json"],
                row["validation_spec_fingerprint"], row["reliability_policy_fingerprint"],
                row["execution_config_identity"], row["cost_config_identity"],
                row["risk_identity"], row["trial_role"], row["canonical_or_neighbour"],
                row["parameter_variant_identity"], row["parameter_variant_label"],
                row["parent_experiment_identity"], row["report_fingerprint"], row["notes"],
                rebuilt.content_fingerprint(), _json.dumps(record, sort_keys=True),
            ),
        )


def _populate_v5_from_v4_legacy(conn: sqlite3.Connection) -> None:
    """Deterministic, preservation-only forward copy of the v4 rows into v5.

    * ``experiments`` -> re-projected via :func:`_copy_experiments_with_asset_domain`
      (schema v6's ``asset_domain`` column did not exist when this function was
      first written, but ``_CURRENT_DDL`` always reflects the latest schema, so
      a v4->v6 ``ensure_schema()`` call needs this function to land rows on the
      CURRENT ``experiments`` shape, not the v5 shape it was originally targeting).
    * ``lineage`` / ``sensitivity_evidence`` / ``cross_market_evidence`` /
      ``imports`` -> copied verbatim.
    * every v4 experiment gets one ``execution_attempts`` row (``attempt_ordinal
      = 1``). It is ``VALID`` unless its import is a known engineering-invalidated
      one (see ``_INVALIDATED_V4_IMPORT_PREFIX``), in which case it is
      ``INVALID_EXECUTION`` with a typed ``invalidation_class``.
    * every v4 ``results`` row -> ``attempt_results`` under that identity's
      attempt 1 (fingerprint preserved).
    * every v4 ``failures`` row -> ``failures`` with ``attempt_id`` set to that
      identity's attempt 1 (or NULL for a system-scope failure).
    """
    import json as _json

    if not _table_exists(conn, "experiments_v4_legacy"):
        return

    # which identities came from an engineering-invalidated import?
    invalid_identities: set[str] = set()
    if _table_exists(conn, "imports_v4_legacy"):
        bad_imports = [
            r[0] for r in conn.execute("SELECT import_id FROM imports_v4_legacy")
            if str(r[0]).startswith(_INVALIDATED_V4_IMPORT_PREFIX)
        ]
        # the a31f571 Phase-15B import wrote every '15B'-phase experiment
        if bad_imports:
            invalid_identities = {
                r[0] for r in conn.execute(
                    "SELECT experiment_identity FROM experiments_v4_legacy WHERE phase = '15B'"
                )
            }

    # 1. verbatim table copies -- explicit column lists so the copy is robust to
    #    a v4 table that lacks the auto-increment row_id.
    def _copy_verbatim(table: str) -> None:
        legacy = f"{table}_v4_legacy"
        if not _table_exists(conn, legacy):
            return
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({legacy})") if r[1] != "row_id"]
        col_sql = ", ".join(cols)
        conn.execute(f"INSERT INTO {table} ({col_sql}) SELECT {col_sql} FROM {legacy}")

    for tbl in ("lineage", "sensitivity_evidence", "cross_market_evidence", "imports"):
        _copy_verbatim(tbl)
    _copy_experiments_with_asset_domain(
        conn, legacy_table="experiments_v4_legacy",
        asset_domain=_MIGRATED_V5_ASSET_DOMAIN,
    )

    # 2. one attempt-1 row per experiment
    attempt_id_by_identity: dict[str, str] = {}
    for row in conn.execute(
        "SELECT experiment_identity, identity_schema, code_commit, "
        "       target_schedule_hash, report_fingerprint, created_at "
        "FROM experiments_v4_legacy ORDER BY experiment_identity"
    ).fetchall():
        ident = row["experiment_identity"]
        invalid = ident in invalid_identities
        status = "INVALID_EXECUTION" if invalid else "VALID"
        prov = {
            "experiment_identity": ident,
            "attempt_ordinal": 1,
            "attempt_status": status,
            "code_commit": row["code_commit"] or "",
            "engine": "",
            "target_schedule_hash": row["target_schedule_hash"],
            "report_fingerprint": row["report_fingerprint"],
            "source_artifact_sha256": None,
        }
        attempt_id = "attempt1:" + _fp("execattemptid1", prov)
        attempt_id_by_identity[ident] = attempt_id
        scientific = {
            "experiment_identity": ident,
            "identity_schema": row["identity_schema"],
            "attempt_ordinal": 1,
            "attempt_status": status,
            "invalidation_class": _INVALIDATION_CLASS if invalid else None,
            "invalidation_detail": _INVALIDATION_DETAIL if invalid else "",
            "invalidation_evidence": (
                {"invalidated_import_prefix": _INVALIDATED_V4_IMPORT_PREFIX,
                 "phase": "15B"} if invalid else {}
            ),
            "defect_resolution_commit": _DEFECT_RESOLUTION_COMMIT if invalid else None,
            "code_commit": row["code_commit"] or "",
            "engine": "",
            "target_schedule_hash": row["target_schedule_hash"],
            "report_fingerprint": row["report_fingerprint"],
            "source_artifact": None,
            "source_artifact_sha256": None,
        }
        record_json = {
            **scientific,
            "created_at": row["created_at"],
            "notes": (
                "v4->v5 forward copy; original single-attempt execution"
                if not invalid else
                "v4->v5 forward copy; INVALID_EXECUTION -- engineering feature-pipeline defect"
            ),
        }
        conn.execute(
            "INSERT INTO execution_attempts ("
            " attempt_id, experiment_identity, identity_schema, attempt_ordinal,"
            " attempt_status, invalidation_class, invalidation_detail,"
            " invalidation_evidence_json, defect_resolution_commit, code_commit, engine,"
            " target_schedule_hash, report_fingerprint, source_artifact,"
            " source_artifact_sha256, created_at, notes, content_fingerprint, record_json"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                attempt_id, ident, row["identity_schema"], 1, status,
                _INVALIDATION_CLASS if invalid else None,
                _INVALIDATION_DETAIL if invalid else "",
                _json.dumps(record_json["invalidation_evidence"], sort_keys=True),
                _DEFECT_RESOLUTION_COMMIT if invalid else None,
                row["code_commit"] or "", "", row["target_schedule_hash"],
                row["report_fingerprint"], None, None, row["created_at"],
                record_json["notes"], _fp("execattempt1", scientific),
                _json.dumps(record_json, sort_keys=True),
            ),
        )

    # 3. v4 results -> attempt_results (fingerprint preserved verbatim)
    if _table_exists(conn, "results_v4_legacy"):
        for row in conn.execute("SELECT * FROM results_v4_legacy").fetchall():
            ident = row["experiment_identity"]
            attempt_id = attempt_id_by_identity[ident]
            cols = row.keys()
            record = _json.loads(row["record_json"])
            record["attempt_id"] = attempt_id
            conn.execute(
                "INSERT INTO attempt_results ("
                " attempt_id, experiment_identity, attempt_ordinal, headline_verdict,"
                " reason_codes_json, gross_pnl_usd, costs_usd, net_pnl_usd, daily_sharpe,"
                " annualized_sharpe, gating_null_p, bh_p_value, bh_q, bh_rejected_at_q,"
                " dsr_probability, fold_consistency, n_trades, n_fills, n_oos_days,"
                " holdout_eligible, evidence_completeness, source_artifact,"
                " source_artifact_sha256, result_fingerprint, record_json"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt_id, ident, 1, row["headline_verdict"], row["reason_codes_json"],
                    row["gross_pnl_usd"], row["costs_usd"], row["net_pnl_usd"],
                    row["daily_sharpe"], row["annualized_sharpe"], row["gating_null_p"],
                    row["bh_p_value"], row["bh_q"], row["bh_rejected_at_q"],
                    row["dsr_probability"],
                    row["fold_consistency"] if "fold_consistency" in cols else None,
                    row["n_trades"], row["n_fills"], row["n_oos_days"],
                    row["holdout_eligible"], row["evidence_completeness"],
                    row["source_artifact"], row["source_artifact_sha256"],
                    row["result_fingerprint"], _json.dumps(record, sort_keys=True),
                ),
            )

    # 4. v4 failures -> failures + attempt_id
    if _table_exists(conn, "failures_v4_legacy"):
        for row in conn.execute("SELECT * FROM failures_v4_legacy").fetchall():
            ident = row["experiment_identity"]
            attempt_id = attempt_id_by_identity.get(ident) if ident else None
            conn.execute(
                "INSERT INTO failures ("
                " failure_id, scope, experiment_identity, attempt_id, failure_class,"
                " failure_code, summary, mechanism, evidence_json, action_taken, resolved,"
                " resolution_commit, superseded_by, created_at, root_symbol, strategy_family,"
                " content_fingerprint"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row["failure_id"], row["scope"], ident, attempt_id, row["failure_class"],
                    row["failure_code"], row["summary"], row["mechanism"], row["evidence_json"],
                    row["action_taken"], row["resolved"], row["resolution_commit"],
                    row["superseded_by"], row["created_at"], row["root_symbol"],
                    row["strategy_family"], row["content_fingerprint"],
                ),
            )


def _populate_v6_from_v5_legacy(conn: sqlite3.Connection) -> None:
    """Deterministic, preservation-only forward copy of the v5 rows into v6.

    * ``experiments`` -> re-projected via :func:`_copy_experiments_with_asset_domain`
      (``asset_domain`` hardcoded to ``_MIGRATED_V5_ASSET_DOMAIN`` = 'FUTURES'
      for every row -- a stated fact about today's real data, not an
      inference; ``experiment_identity`` itself is never recomputed).
    * ``execution_attempts`` / ``attempt_results`` / ``attempt_lineage`` /
      ``failures`` / ``lineage`` / ``sensitivity_evidence`` /
      ``cross_market_evidence`` / ``imports`` -> copied verbatim (unchanged
      shape), in FK-dependency order (``experiments`` and ``execution_attempts``
      before anything that references them).
    """
    if not _table_exists(conn, "experiments_v5_legacy"):
        return

    _copy_experiments_with_asset_domain(
        conn, legacy_table="experiments_v5_legacy",
        asset_domain=_MIGRATED_V5_ASSET_DOMAIN,
    )

    def _copy_verbatim(table: str) -> None:
        legacy = f"{table}_v5_legacy"
        if not _table_exists(conn, legacy):
            return
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({legacy})") if r[1] != "row_id"]
        col_sql = ", ".join(cols)
        conn.execute(f"INSERT INTO {table} ({col_sql}) SELECT {col_sql} FROM {legacy}")

    # FK-dependency order: execution_attempts before attempt_results /
    # attempt_lineage / failures (which reference it); experiments (already
    # populated above) before lineage / sensitivity_evidence (which reference
    # it); cross_market_evidence / imports have no FK dependency.
    for tbl in (
        "execution_attempts", "attempt_results", "attempt_lineage",
        "failures", "lineage", "sensitivity_evidence", "cross_market_evidence", "imports",
    ):
        _copy_verbatim(tbl)


def _fp(domain: str, payload: object) -> str:
    from alpha_agent.validation.fingerprint import fingerprint
    return fingerprint(domain, payload)


def ensure_schema(conn: sqlite3.Connection) -> int:
    """Create or migrate the database to :data:`SCHEMA_VERSION`, loudly.

    Returns the resulting schema version.
    """
    conn.execute("PRAGMA foreign_keys = ON")
    version = read_schema_version(conn)
    start_version = version
    if version is None:
        version = SCHEMA_VERSION
    elif version > SCHEMA_VERSION:
        raise UnknownSchemaVersion(
            f"registry database schema_version={version} is newer than this build "
            f"({SCHEMA_VERSION}); upgrade the code -- never downgrade the data"
        )
    else:
        while version < SCHEMA_VERSION:
            migrate = _MIGRATIONS.get(version)
            if migrate is None:
                raise UnknownSchemaVersion(
                    f"no migration registered from schema_version={version}"
                )
            migrate(conn)
            version += 1
    # every statement is IF NOT EXISTS: creates a fresh database, or the tables
    # a preservation migration just renamed aside.
    conn.executescript(_CURRENT_DDL)
    # v4->v5: forward-copy the preserved v4 rows into the new attempt layer.
    if start_version is not None and start_version <= 4:
        _populate_v5_from_v4_legacy(conn)
    # v5->v6: forward-copy the preserved v5 rows, adding asset_domain. Guarded
    # separately (not elif) because a single ensure_schema() call may walk
    # straight from v4 (or earlier) to v6 -- in that case ONLY the v4_legacy
    # populate step actually finds rows (the v5_legacy rename in between is a
    # no-op, see _migrate_5_to_6's docstring), so this call's own
    # experiments_v5_legacy existence guard correctly no-ops rather than
    # double-inserting.
    if start_version is not None and start_version <= 5:
        _populate_v6_from_v5_legacy(conn)
    conn.execute(
        "INSERT INTO registry_meta (key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    return SCHEMA_VERSION
