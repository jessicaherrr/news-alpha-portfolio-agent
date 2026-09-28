"""Isolated, append-only store for Phase 22 SYNTHETIC crypto research results.

Structurally separate from ``data/registry/experiments.sqlite`` (the real
Phase 14 experiment registry, schema v5) -- a different file, a different
schema, no shared table, no shared identity function, no code path that writes
into both. That separation is what makes the Phase 22 review-gate constraints
true BY CONSTRUCTION rather than by policy:

* never mixed with the real Phase 13.5C+ statistical family -- there is no
  table here a BH/FDR family computation could join against, and nothing in
  :mod:`alpha_agent.validation.multiple_testing` or the real registry ever
  reads this file;
* never authoritative -- this schema has no ``Authority`` / ``SUPERSEDES``
  concept at all; a row here is never presented as the current scientific
  answer to anything, and its committed ``verdict`` is always presented as a
  "Synthetic Policy Outcome (NON-AUTHORITATIVE)"
  (:data:`alpha_agent.crypto.envelope.SYNTHETIC_POLICY_OUTCOME_LABEL`), never
  a plain "Verdict";
* never paper-trading eligible -- :mod:`alpha_agent.paper.eligibility` only
  ever opens the real ``ExperimentRegistry`` at
  ``data/registry/experiments.sqlite`` and rebuilds a strategy only from
  ``candidates_phase_13_5c.BASELINE_FAMILIES``, which
  ``CRYPTO_FUNDING_CONTRARIAN_FAMILY`` is deliberately not a member of
  (belt-and-suspenders: even a mistaken cross-wire could not rebuild a
  paper-eligible strategy from a row here);
* every row is stamped ``data_role='SYNTHETIC'`` under a ``CHECK`` constraint
  at the schema level, on top of the typed
  :class:`~alpha_agent.crypto.provenance.CryptoDataProvenance` guard already
  enforced before a row is ever built.

Schema v2 (Phase 22.1): a row persists the WHOLE
:class:`~alpha_agent.crypto.envelope.SyntheticCryptoValidationArtifact` --
complete per-series ``provenance_by_series`` (never collapsed to one record),
the dataset/generator identity, and a stable ``content_identity`` that does
NOT depend on any provenance record's wall-clock ``generated_at_utc`` (two
recordings of byte-identical synthetic inputs have the same
``content_identity`` even if made on different days).

Phase 22.1b -- legacy schema guard: there is no migration from schema v1 to
v2 (a v1 row never carried the complete provenance envelope v2 requires, and
fabricating those missing fields for a v1 row would put invented content into
a scientific-looking artifact -- not allowed even though this store is
non-authoritative). Instead, every open of an EXISTING database detects its
real shape (``PRAGMA user_version`` plus the actual column set -- never
trusted blindly) before any v2 ``SELECT``/``INSERT`` runs, and raises
:class:`SyntheticRegistrySchemaError` for a legacy v1 table or anything
unrecognized. A fresh database is created as v2 and stamped
``PRAGMA user_version = 2``; an existing v2 database that predates this stamp
(same column shape, ``user_version`` still 0) is recognized by its columns and
stamped on this open -- metadata only, no row is read, fabricated, or
rewritten. This is the same class of schema-evolution problem Phase 21.1b
hardened for the paper ledger, applied here to a store that is local,
gitignored, isolated dev/test state nothing outside this package has ever
depended on across a commit -- so "no migration" is the correct scientific
answer, and "never silently misread it" is the correct engineering one.

Phase 22.1c -- reset-path restriction: :func:`reset_synthetic_store` is the
only DESTRUCTIVE operation this module offers (it renames a file off disk),
and it may only ever target the currently-configured canonical isolated
synthetic registry path. This is enforced inside the function itself, not by
documentation or by CLI plumbing: a resolved target that is not exactly the
canonical path raises :class:`SyntheticRegistryPathError` before any
rename/unlink is attempted -- so a ``--db`` override can never point the
reset command at an arbitrary sqlite file, the real Phase 14 registry, or the
Phase 21 paper ledger. ``list``/``get``/``record`` remain free to accept any
``db_path`` (read/append only, never destructive).

Read-only accessors here are the ONLY sanctioned way
:mod:`alpha_agent.ui.services` may surface Phase 22 results (mirrors the real
registry's read-only UI boundary, Phase 20's ``services.py``); both they and
the CLI (``scripts/phase_22_crypto_research.py``) must catch
:class:`SyntheticRegistrySchemaError` and :class:`SyntheticRegistryPathError`
explicitly rather than let a raw ``sqlite3.OperationalError`` (or a
mis-targeted reset) reach a user.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from alpha_agent.crypto.envelope import SyntheticCryptoValidationArtifact
from alpha_agent.crypto.provenance import DataProvenanceRole
from alpha_agent.registry.holdout_guard import assert_no_holdout_market_data

DEFAULT_DB_PATH = Path("data/crypto_synthetic/registry.sqlite")

_TABLE_NAME = "crypto_synthetic_experiments"

SCHEMA_VERSION = 2

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL UNIQUE,
    data_role TEXT NOT NULL CHECK (data_role = 'SYNTHETIC'),
    banner TEXT NOT NULL,
    strategy_family TEXT NOT NULL,
    root_symbol TEXT NOT NULL,
    params_json TEXT NOT NULL,
    strategy_fingerprint TEXT NOT NULL,
    validation_fingerprint TEXT NOT NULL,
    report_fingerprint TEXT NOT NULL,
    content_identity TEXT NOT NULL,
    generator_identity TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reason_codes_json TEXT NOT NULL,
    fixture_seed INTEGER NOT NULL,
    provenance_by_series_json TEXT NOT NULL,
    dataset_identity_json TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);
"""

#: Columns that exist ONLY in the Phase 22 (bdafd12) v1 table shape, never in
#: v2 -- a real, structural marker, not a guess.
_V1_ONLY_COLUMNS = frozenset({"validation_spec_fingerprint", "provenance_json"})
#: Columns v2 requires that v1 never had -- the complete Phase 22.1
#: provenance envelope.
_V2_REQUIRED_COLUMNS = frozenset(
    {
        "content_identity", "generator_identity", "provenance_by_series_json",
        "dataset_identity_json", "validation_fingerprint",
    }
)


class SyntheticRegistrySchemaError(RuntimeError):
    """The isolated synthetic store at a given path is not a valid, current
    (schema v2) Phase 22 database -- a legacy v1 table, or something
    unrecognized. Raised BEFORE any v2 ``SELECT``/``INSERT`` is attempted, so
    a caller never sees a raw ``sqlite3.OperationalError`` ("no such
    column") instead."""


_LEGACY_V1_MESSAGE = (
    "Legacy Phase 22 synthetic registry schema v1 detected at {db_path!r}.\n"
    "It contains non-authoritative synthetic fixture results that do not carry "
    "the complete Phase 22.1 provenance envelope (provenance_by_series, "
    "dataset_identity, generator_identity, content_identity).\n"
    "Archive/reset this synthetic-only store and rerun the Phase 22 synthetic "
    "fixture if you want v2 results -- e.g. "
    "`python scripts/phase_22_crypto_research.py reset-store --confirm`.\n"
    "This store is SYNTHETIC, non-authoritative, and never part of the real "
    "ExperimentRegistry or BH/FDR family, so preserving a v1 row as scientific "
    "evidence is not required; nothing here fabricates the missing v2 fields."
)

_UNKNOWN_SCHEMA_MESSAGE = (
    "Unrecognized synthetic registry schema at {db_path!r} "
    "(PRAGMA user_version={user_version}, columns={columns}). Refusing to "
    "guess -- archive/reset this synthetic-only store rather than risk "
    "silently misreading it."
)


def _table_columns(conn: sqlite3.Connection, *, table: str = _TABLE_NAME) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the isolated synthetic store, detecting its
    real on-disk shape before returning a connection any v2 query can trust.

    * no table yet (a brand-new or empty file) -> create the v2 schema, stamp
      ``PRAGMA user_version = 2``;
    * an existing table whose columns already match v2 -> use it as-is,
      stamping the version pragma now if an earlier build never set it (pure
      metadata, no row touched);
    * an existing table matching the known Phase 22 (bdafd12) v1 shape ->
      raise :class:`SyntheticRegistrySchemaError`, never silently adapt;
    * anything else -> raise :class:`SyntheticRegistrySchemaError` too --
      fail closed rather than guess.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    cols = _table_columns(conn)
    if not cols:
        conn.execute(_SCHEMA_SQL)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        return conn

    user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    if _V2_REQUIRED_COLUMNS <= cols:
        # Real v2 shape, whether or not an earlier build already stamped the
        # pragma. Stamping it now changes no row -- pure metadata.
        if user_version != SCHEMA_VERSION:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        return conn

    conn.close()
    if _V1_ONLY_COLUMNS <= cols:
        raise SyntheticRegistrySchemaError(_LEGACY_V1_MESSAGE.format(db_path=str(path)))
    raise SyntheticRegistrySchemaError(
        _UNKNOWN_SCHEMA_MESSAGE.format(db_path=str(path), user_version=user_version, columns=sorted(cols))
    )


class SyntheticRegistryPathError(RuntimeError):
    """A destructive archive/reset action was asked to target a path other
    than the currently-configured canonical isolated synthetic registry
    location. Refused BEFORE any rename/unlink is attempted -- enforced here,
    at the mutation boundary itself, not left to documentation or to the
    (also true) fact that this module never imports the real registry: a
    ``--db`` override must never let a mutating operation reach an arbitrary
    sqlite file, the real Phase 14 registry, or the Phase 21 paper ledger."""


def reset_synthetic_store(db_path: str | Path | None = None, *, confirm: bool = False) -> Path | None:
    """Archive (rename) the ISOLATED synthetic store, and ONLY it, so a fresh
    v2 database is created on next use. Never reinterprets old rows as v2
    scientific evidence: the archived file is left byte-for-byte, just moved
    out of the way. Requires ``confirm=True`` -- there is no default reset.

    ``db_path`` is accepted for test/configuration purposes only: if given, it
    must resolve to exactly the currently-configured canonical synthetic
    registry path (the module's ``DEFAULT_DB_PATH`` read fresh at call time,
    so a test may ``monkeypatch`` that attribute to a tmp directory and this
    function still behaves normally against it). Anything else -- the real
    Phase 14 registry, the Phase 21 paper ledger, or any other sqlite file,
    however supplied -- raises :class:`SyntheticRegistryPathError` before any
    filesystem mutation. There is no way, from the CLI or from this function,
    to reset an arbitrary ``--db`` value; ``list``/``get``/``record`` keep
    accepting one freely (read/append operations, never destructive)."""
    if not confirm:
        raise ValueError("reset_synthetic_store refuses without confirm=True (no default/implicit reset)")
    canonical = Path(DEFAULT_DB_PATH)
    target = Path(db_path) if db_path is not None else canonical
    if target.resolve() != canonical.resolve():
        raise SyntheticRegistryPathError(
            f"reset_synthetic_store refuses to touch {str(target)!r}: it does not "
            f"resolve to the configured isolated synthetic registry path "
            f"{str(canonical)!r}. A destructive archive/reset action may only ever "
            "target the isolated Phase 22 synthetic store -- never an arbitrary "
            "sqlite path, the real Phase 14 registry, or the Phase 21 paper ledger."
        )
    if not target.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archived = target.with_name(f"{target.name}.legacy-{stamp}.bak")
    target.rename(archived)
    return archived


def _experiment_id(artifact: SyntheticCryptoValidationArtifact) -> str:
    short = artifact.content_identity().split(":")[-1][:16]
    return f"CRYPTO-SYNTH__{artifact.strategy_family}__{artifact.dataset_identity.root_symbol}__{short}".upper()


def record_synthetic_experiment(
    artifact: SyntheticCryptoValidationArtifact, *, db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    """Append one SYNTHETIC research result. A plain ``INSERT`` only -- never an
    upsert or an in-place update; a duplicate ``experiment_id`` (same content
    identity) raises rather than silently overwriting, matching the real
    registry's append-only discipline (CLAUDE.md experiment-registry
    section 5). ``experiment_id`` is derived from
    :meth:`SyntheticCryptoValidationArtifact.content_identity`, which is itself
    wall-clock independent -- rerunning byte-identical synthetic inputs on a
    different day is detected as the SAME experiment, not a new one."""
    for series_name, prov in artifact.provenance_by_series.items():
        try:
            prov.assert_synthetic()
        except Exception as exc:
            raise type(exc)(f"series {series_name!r}: {exc}") from exc
    # Defense in depth: even though the Phase 22 fixture calendar is fixed
    # years before 2025 (alpha_agent.crypto.synthetic_fixtures.SYNTHETIC_BASE_NS),
    # every payload is still swept by the same real Phase 14 holdout guard
    # every registry write uses, so a future change to the fixture calendar
    # cannot silently smuggle a >= 2025 value into this store either.
    assert_no_holdout_market_data(artifact.report.model_dump(mode="json"))
    experiment_id = _experiment_id(artifact)
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO crypto_synthetic_experiments (
                experiment_id, data_role, banner, strategy_family, root_symbol,
                params_json, strategy_fingerprint, validation_fingerprint,
                report_fingerprint, content_identity, generator_identity,
                verdict, reason_codes_json, fixture_seed,
                provenance_by_series_json, dataset_identity_json, report_json,
                created_at_utc
            ) VALUES (?, 'SYNTHETIC', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                artifact.banner,
                artifact.strategy_family,
                artifact.dataset_identity.root_symbol,
                json.dumps(artifact.params, sort_keys=True),
                artifact.strategy_fingerprint,
                artifact.validation_fingerprint,
                artifact.report.report_fingerprint(),
                artifact.content_identity(),
                artifact.generator_identity,
                artifact.report.verdict.value,
                json.dumps(sorted(c.value for c in artifact.report.reason_codes)),
                artifact.fixture_seed,
                json.dumps(
                    {k: v.model_dump(mode="json") for k, v in artifact.provenance_by_series.items()},
                    sort_keys=True,
                ),
                json.dumps(artifact.dataset_identity.model_dump(mode="json"), sort_keys=True),
                artifact.report.model_dump_json(),
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            f"a Phase 22 synthetic experiment {experiment_id!r} is already recorded "
            "(append-only store; re-running the identical synthetic spec is not a "
            "new row -- content_identity matched an existing row)"
        ) from exc
    finally:
        conn.close()
    return experiment_id


def list_synthetic_experiments(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    """Read-only. Every row's ``data_role`` is always ``'SYNTHETIC'`` (schema
    ``CHECK`` constraint); callers must never infer otherwise."""
    path = Path(db_path)
    if not path.exists():
        return []
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT experiment_id, data_role, banner, strategy_family, root_symbol, "
            "params_json, strategy_fingerprint, content_identity, verdict, fixture_seed, "
            "created_at_utc FROM crypto_synthetic_experiments ORDER BY row_id DESC"
        )
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
        for r in rows:
            r["params"] = json.loads(r.pop("params_json"))
            assert r["data_role"] == DataProvenanceRole.SYNTHETIC.value
        return rows
    finally:
        conn.close()


def get_synthetic_experiment(experiment_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> dict | None:
    path = Path(db_path)
    if not path.exists():
        return None
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT * FROM crypto_synthetic_experiments WHERE experiment_id = ?", (experiment_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [c[0] for c in cur.description]
        out = dict(zip(cols, row, strict=True))
        out["params"] = json.loads(out.pop("params_json"))
        out["reason_codes"] = json.loads(out.pop("reason_codes_json"))
        out["provenance_by_series"] = json.loads(out.pop("provenance_by_series_json"))
        out["dataset_identity"] = json.loads(out.pop("dataset_identity_json"))
        out["report"] = json.loads(out.pop("report_json"))
        assert out["data_role"] == DataProvenanceRole.SYNTHETIC.value
        return out
    finally:
        conn.close()
