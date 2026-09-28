"""Phase 14 -- import the corrected Phase 13.5C research history into the registry.

    python scripts/phase_14_import_phase_13_5c.py                 # import + export
    python scripts/phase_14_import_phase_13_5c.py --dry-run       # build + validate only
    python scripts/phase_14_import_phase_13_5c.py --db <path>     # alternate database

STRICTLY OFFLINE. This script reads committed Phase 13.5C artifacts and writes
SQLite rows. It constructs no Databento client, makes no network call, incurs no
data spend, recomputes no performance number, and refuses any market-data or
performance timestamp at or after 2025-01-01.

The import is transactional and idempotent: running it twice produces the same
database, the same 107 statistical hypotheses, the same identities and no
duplicate rows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from alpha_agent.registry.exports import write_all
from alpha_agent.registry.phase_13_5c_import import (
    EXPECTED_CANONICAL,
    EXPECTED_NEIGHBOURS,
    EXPECTED_UNIQUE_TRIALS,
    EXPECTED_VERDICTS,
    build_bundle,
    import_phase_13_5c,
)
from alpha_agent.registry.sqlite_registry import ExperimentRegistry

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO / "data" / "registry" / "experiments.sqlite"
DEFAULT_OUT = REPO / "outputs" / "phase_14"


def _assert_offline() -> list[str]:
    """The importer must never construct a market-data client."""
    notes = []
    for module in ("databento", "httpx", "requests"):
        if module in sys.modules:
            notes.append(f"WARNING: {module} is imported in this process")
    notes.append("no Databento Historical client constructed, no metadata.get_cost, "
                 "no get_range, no new spend")
    notes.append("all inputs are committed local artifacts under outputs/phase_13_5c/ "
                 "and data/manifests/phase_13_5c/")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--dry-run", action="store_true",
                    help="build and validate the bundle; write nothing")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("PHASE 14 -- EXPERIMENT REGISTRY IMPORT OF CORRECTED PHASE 13.5C")
    print("=" * 72)

    bundle = build_bundle(REPO)
    print(f"source fingerprint   : {bundle.source_fingerprint}")
    print(f"import id            : {bundle.import_id}")
    print(f"experiments in bundle: {len(bundle.experiments)}")
    print(f"  authoritative      : {EXPECTED_UNIQUE_TRIALS} "
          f"({EXPECTED_CANONICAL} canonical + {EXPECTED_NEIGHBOURS} neighbours)")
    print(f"  superseded lineage : {len(bundle.experiments) - EXPECTED_UNIQUE_TRIALS}")
    print(f"failure records      : {len(bundle.failures)}")
    print(f"lineage edges        : {len(bundle.lineage)}")
    print(f"sensitivity evidence : {len(bundle.sensitivity)} (NOT statistical trials)")
    print(f"cross-market summaries: {len(bundle.cross_market)} (NOT experiments)")

    if args.dry_run:
        print("\n--dry-run: bundle validated, nothing written")
        return 0

    db = Path(args.db)
    with ExperimentRegistry(db) as registry:
        _, counts = import_phase_13_5c(registry, REPO)
        summary = registry.summary()
        print(f"\nrows applied         : {counts}")
        print(f"registry             : {db}")
        print(f"schema version       : {summary.schema_version}")
        print(f"unique hypotheses    : {summary.authoritative_statistical_hypotheses} "
              f"(expected {EXPECTED_UNIQUE_TRIALS})")
        print(f"canonical / neighbour: {summary.canonical} / {summary.neighbour}")
        print(f"canonical verdicts   : {summary.canonical_verdict_counts} "
              f"(expected {EXPECTED_VERDICTS})")
        print(f"superseded rows      : {summary.superseded_experiments}")
        print(f"holdout eligible     : {summary.holdout_eligible}")
        print(f"failure records      : {summary.failure_records}")
        print(f"content digest       : {summary.content_digest}")

        problems = []
        if summary.authoritative_statistical_hypotheses != EXPECTED_UNIQUE_TRIALS:
            problems.append("unique statistical hypothesis count")
        if summary.canonical != EXPECTED_CANONICAL:
            problems.append("canonical count")
        if summary.neighbour != EXPECTED_NEIGHBOURS:
            problems.append("neighbour count")
        if summary.canonical_verdict_counts != EXPECTED_VERDICTS:
            problems.append("canonical verdict counts")
        if summary.holdout_eligible != 0:
            problems.append("holdout-eligible candidates must be 0")
        if problems:
            print("\nFINAL VALIDATION FAILED: " + "; ".join(problems))
            return 1

        if not args.no_export:
            out_dir = Path(args.out_dir)
            for name, path in sorted(write_all(registry, out_dir).items()):
                print(f"written -> {path}")

    print("\nHOLDOUT / SPEND SAFETY:")
    for note in _assert_offline():
        print(f"  - {note}")
    print("  - every market-data / performance value >= 2025-01-01 is refused at the "
          "registry write boundary (HoldoutAccessError)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
