"""Phase 22 -- crypto / on-chain extension CLI (synthetic scaffold only).

    python scripts/phase_22_crypto_research.py run [--root BTC] [--window 14]
        [--threshold 1.5] [--size 1] [--seed 22]
    python scripts/phase_22_crypto_research.py list
    python scripts/phase_22_crypto_research.py show --experiment <experiment_id>
    python scripts/phase_22_crypto_research.py reset-store --confirm

Every command is OFFLINE, deterministic, and touches SYNTHETIC data only
(``alpha_agent.crypto.synthetic_fixtures`` -- a seeded fixture generator, never
a network call, never a paid vendor). ``run`` drives the exact same
:class:`~alpha_agent.validation.engine.ValidationEngine` / frozen
:class:`~alpha_agent.validation.policy.ReliabilityPolicy` and, when the
compiled C++ core is present, the real ``quant_backtest_targets_csv`` engine,
every other family in this repository uses -- then records the result to the
ISOLATED ``data/crypto_synthetic/registry.sqlite`` store (never the real Phase
14 registry, never paper-trading eligible; see
``alpha_agent.crypto.synthetic_registry``).

Mutating writes to this isolated synthetic store are CLI-only, mirroring
Phase 21's paper-trading convention (``alpha_agent.ui.services`` and every
Streamlit view stay strictly read-only).

Phase 22.1b: ``run``/``list``/``show`` refuse a legacy Phase 22 (v1) synthetic
database with a typed ``SCHEMA_INCOMPATIBLE`` message on stderr and a non-zero
exit -- never a raw traceback. ``reset-store --confirm`` archives (renames)
the isolated synthetic store so a fresh v2 database is created on next use.

Phase 22.1c: ``--db`` freely overrides the target for ``run``/``list``/``show``
(read/append operations), but ``reset-store`` is a DESTRUCTIVE action and may
only ever target the configured isolated synthetic registry path -- passing
``--db`` with any other value to ``reset-store`` is refused with a typed
``PATH_REFUSED`` message and a non-zero exit, enforced inside
:func:`alpha_agent.crypto.synthetic_registry.reset_synthetic_store` itself, not
just here. There is no way to reset the real Phase 14 registry or the Phase 21
paper ledger through this command.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from alpha_agent.crypto.envelope import SYNTHETIC_POLICY_OUTCOME_LABEL
from alpha_agent.crypto.provenance import RealVendorNotConnectedError
from alpha_agent.crypto.research import DEFAULT_CLI, run_crypto_research
from alpha_agent.crypto.synthetic_registry import (
    DEFAULT_DB_PATH,
    SyntheticRegistryPathError,
    SyntheticRegistrySchemaError,
    get_synthetic_experiment,
    list_synthetic_experiments,
    record_synthetic_experiment,
    reset_synthetic_store,
)

REPO = Path(__file__).resolve().parents[1]


def cmd_run(args: argparse.Namespace) -> int:
    try:
        artifact = run_crypto_research(
            root_symbol=args.root, window=args.window, threshold=args.threshold,
            size=args.size, seed=args.seed, executable=REPO / args.cli,
            work_dir=REPO / "outputs" / "phase_22" / "_runs",
        )
    except RealVendorNotConnectedError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    experiment_id = record_synthetic_experiment(artifact, db_path=REPO / args.db)
    print(f"{SYNTHETIC_POLICY_OUTCOME_LABEL}: {artifact.report.verdict.value}", file=sys.stderr)
    print(json.dumps(
        {
            "experiment_id": experiment_id,
            "data_role": "SYNTHETIC",
            "banner": artifact.banner,
            "synthetic_policy_outcome": artifact.report.verdict.value,
            "reason_codes": sorted(c.value for c in artifact.report.reason_codes),
            "report_fingerprint": artifact.report.report_fingerprint(),
            "content_identity": artifact.content_identity(),
        },
        indent=2,
    ))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = list_synthetic_experiments(db_path=REPO / args.db)
    print(json.dumps(rows, indent=2, default=str))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    row = get_synthetic_experiment(args.experiment, db_path=REPO / args.db)
    if row is None:
        print(f"no such experiment: {args.experiment}", file=sys.stderr)
        return 1
    print(json.dumps(row, indent=2, default=str))
    return 0


def cmd_reset_store(args: argparse.Namespace) -> int:
    if not args.confirm:
        print(
            "REFUSED: reset-store requires --confirm (no default/implicit reset). "
            "This only ever touches the isolated synthetic store, never the real "
            "Phase 14 registry.",
            file=sys.stderr,
        )
        return 2
    archived = reset_synthetic_store(db_path=REPO / args.db, confirm=True)
    if archived is None:
        print(json.dumps({"status": "no_store_present", "db": str(REPO / args.db)}, indent=2))
        return 0
    print(json.dumps({"status": "archived", "archived_to": str(archived)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(DEFAULT_DB_PATH))
    sub = p.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one synthetic crypto hypothesis and record it")
    p_run.add_argument("--root", default="BTC")
    p_run.add_argument("--window", type=int, default=14)
    p_run.add_argument("--threshold", type=float, default=1.5)
    p_run.add_argument("--size", type=int, default=1)
    p_run.add_argument("--seed", type=int, default=22)
    p_run.add_argument("--cli", default=str(DEFAULT_CLI))
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="list recorded SYNTHETIC crypto experiments")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="show one recorded SYNTHETIC crypto experiment")
    p_show.add_argument("--experiment", required=True)
    p_show.set_defaults(func=cmd_show)

    p_reset = sub.add_parser(
        "reset-store",
        help="archive the isolated synthetic store (e.g. a legacy v1 database) so a fresh v2 one is created next",
    )
    p_reset.add_argument("--confirm", action="store_true", help="required -- there is no default reset")
    p_reset.set_defaults(func=cmd_reset_store)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except SyntheticRegistrySchemaError as exc:
        print(f"SCHEMA_INCOMPATIBLE: {exc}", file=sys.stderr)
        return 3
    except SyntheticRegistryPathError as exc:
        print(f"PATH_REFUSED: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
