"""Researcher-facing registry CLI (section 16).

Normal inspection never requires raw SQL::

    python -m alpha_agent.registry.cli summary
    python -m alpha_agent.registry.cli show NQ__TSMOM__CANONICAL__VALIDATION_2023_2024
    python -m alpha_agent.registry.cli find-exact --identity experiment1:<sha256>
    python -m alpha_agent.registry.cli find-related --family tsmom --root NQ \
        --param fast_horizon=21 --param slow_horizon=120
    python -m alpha_agent.registry.cli failures --class FDR_NOT_PASSED
    python -m alpha_agent.registry.cli history NQ__TSMOM__CANONICAL__VALIDATION_2023_2024
    python -m alpha_agent.registry.cli list --family mean_reversion --root ES
    python -m alpha_agent.registry.cli export --out-dir outputs/phase_14

Every command is a read; nothing here mutates the registry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alpha_agent.registry.enums import (
    AssetDomain,
    Authority,
    FailureClass,
    RegistryVerdict,
    TrialRole,
)
from alpha_agent.registry.failure_memory import FailureMemory
from alpha_agent.registry.sqlite_registry import (
    DEFAULT_REGISTRY_PATH,
    ExperimentRegistry,
    UnknownExperiment,
)


def _parse_params(items: list[str]) -> dict:
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--param expects name=value, got {item!r}")
        key, _, raw = item.partition("=")
        try:
            out[key] = int(raw)
        except ValueError:
            try:
                out[key] = float(raw)
            except ValueError:
                out[key] = raw
    return out


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def cmd_summary(reg: ExperimentRegistry, args: argparse.Namespace) -> int:
    s = reg.summary()
    if args.json:
        _print_json(s.model_dump(mode="json"))
        return 0
    print(f"registry            : {s.registry_path}")
    print(f"schema version      : {s.schema_version} (identity {s.identity_schema})")
    print()
    print(f"authoritative experiment identities : {s.authoritative_statistical_hypotheses}")
    print(f"  CANONICAL                          : {s.canonical}")
    print(f"  NEIGHBOUR                          : {s.neighbour}")
    print(f"  ABLATION                           : {s.ablation}")
    print(f"  VARIANT                            : {s.variant}")
    role_total = s.canonical + s.neighbour + s.ablation + s.variant
    assert role_total == s.authoritative_statistical_hypotheses, (
        f"CANONICAL+NEIGHBOUR+ABLATION+VARIANT ({role_total}) != "
        f"authoritative experiment identities ({s.authoritative_statistical_hypotheses}) "
        "-- a TrialRole value is missing from this breakdown"
    )
    print()
    print("canonical/headline verdicts (only CANONICAL trials are individually")
    print("PASS/REJECT/INCONCLUSIVE-adjudicated; NEIGHBOUR/ABLATION are BH/FDR")
    print("denominator evidence, not headline-adjudicated -- see docs/EXPERIMENT_REGISTRY.md):")
    verdict_order = ("PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED")
    for k in verdict_order:
        print(f"  {k:<35}: {s.canonical_verdict_counts.get(k, 0)}")
    verdict_total = sum(s.canonical_verdict_counts.get(k, 0) for k in verdict_order)
    print(f"  {'total':<35}: {verdict_total}")
    assert verdict_total == s.canonical, (
        f"PASS+REJECT+INCONCLUSIVE+NOT_ADJUDICATED ({verdict_total}) != "
        f"CANONICAL ({s.canonical})"
    )
    print()
    print(f"superseded rows      : {s.superseded_experiments}")
    print(f"total experiment rows: {s.total_experiment_rows}")
    print(f"execution attempts   : {s.execution_attempts} "
          f"({s.valid_execution_attempts} VALID + {s.invalid_execution_attempts} "
          f"INVALID_EXECUTION)")
    if s.canonical_without_valid_attempt:
        print(f"canonical w/o valid attempt: {s.canonical_without_valid_attempt} "
              f"(hypothesis declared; awaiting a corrected execution)")
    print(f"holdout eligible     : {s.holdout_eligible} "
          f"(attempt_results.holdout_eligible -- a validation-methodology flag, "
          f"NOT paper-trading eligibility; see "
          f"'python scripts/phase_21_paper_trading.py list-eligible' for that)")
    print(f"C++ executions       : {s.completed_cpp_executions} (compute, not hypotheses)")
    print(f"sensitivity reruns   : {s.sensitivity_evidence} (not BH trials)")
    print(f"cross-market summaries: {s.cross_market_evidence} (not experiments)")
    print(f"failure records      : {s.failure_records}")
    for k, v in sorted(s.failure_class_counts.items()):
        print(f"    {k:<32} {v}")
    print(f"lineage edges       : {s.lineage_edges}")
    print(f"content digest      : {s.content_digest}")
    return 0


def cmd_show(reg: ExperimentRegistry, args: argparse.Namespace) -> int:
    view = reg.get(args.experiment)
    if args.json:
        _print_json({
            "experiment": view.experiment.model_dump(mode="json"),
            "result": view.result.model_dump(mode="json") if view.result else None,
            "authority": view.authority.value,
            "superseded_by": list(view.superseded_by),
        })
        return 0
    e, r = view.experiment, view.result
    print(f"{e.experiment_id}   [{view.authority.value}]")
    print(f"  display name    : {e.display_name}")
    print(f"  identity        : {e.experiment_identity}")
    print(f"  phase / commit  : {e.phase} / {e.code_commit or '-'}")
    print(f"  root / family   : {e.root_symbol} / {e.strategy_family}")
    print(f"  trial role      : {e.trial_role.value} ({e.parameter_variant_label})")
    params = e.strategy_spec_json.get("params", {})
    print("  params          : " + ", ".join(f"{k}={params[k]}" for k in sorted(params)))
    print(f"  market window   : {e.market_window.start_date}..{e.market_window.end_date}")
    print(f"  strategy fp     : {e.strategy_fingerprint}")
    print(f"  schedule hash   : {e.target_schedule_hash or '- (not retained)'}")
    print(f"  report fp       : {e.report_fingerprint or '- (no per-variant report)'}")
    if r is not None:
        print(f"  verdict         : {r.headline_verdict.value}")
        print(f"  reason codes    : {', '.join(r.reason_codes) or '-'}")
        print(f"  net PnL / dSR   : {r.net_pnl_usd} / {r.daily_sharpe}")
        print(f"  null p / BH q   : {r.gating_null_p} / {r.bh_q}")
        print(f"  DSR             : {r.dsr_probability}")
        print(f"  trades / days   : {r.n_trades} / {r.n_oos_days}")
        print(f"  holdout eligible: {r.holdout_eligible}")
        print(f"  evidence        : {r.evidence_completeness}")
        if r.source_artifact:
            print(f"  evidence source : {r.source_artifact}")
    if view.superseded_by:
        auth = reg.resolve_authoritative(e.experiment_identity)
        print(f"  SUPERSEDED BY   : {auth.experiment_id}")
    for f in reg.failures(experiment_identity=e.experiment_identity):
        print(f"  failure         : {f.failure_class.value} ({f.failure_code})")
    for s in reg.sensitivity_evidence(e.experiment_identity):
        print(f"  sensitivity     : {s.kind} verdict_changed={s.verdict_changed} "
              f"(not a BH trial)")
    if e.notes:
        print(f"  notes           : {e.notes}")
    return 0


def cmd_find_exact(reg: ExperimentRegistry, args: argparse.Namespace) -> int:
    dup = reg.find_exact_duplicate(
        args.identity, asset_domain=AssetDomain(args.asset_domain)
    )
    if args.json:
        _print_json(dup.model_dump(mode="json"))
        return 0
    if not dup.exists:
        print(f"no experiment with identity {args.identity}")
        return 0
    print(f"EXISTS: {dup.experiment_id}  [{dup.authority.value if dup.authority else '-'}]")
    print(f"  display name : {dup.display_name}")
    print(f"  phase        : {dup.phase}")
    print(f"  verdict      : {dup.headline_verdict.value if dup.headline_verdict else '-'}")
    print(f"  reason codes : {', '.join(dup.reason_codes) or '-'}")
    print(f"  report fp    : {dup.report_fingerprint or '-'}")
    if dup.authoritative_replacement:
        print(f"  superseded by: {dup.authoritative_replacement}")
    return 0


def cmd_find_related(reg: ExperimentRegistry, args: argparse.Namespace) -> int:
    hits = reg.find_related(
        strategy_family=args.family,
        root_symbol=args.root,
        asset_domain=AssetDomain(args.asset_domain),
        params=_parse_params(args.param),
        signal_cadence=args.signal_cadence,
        execution_cadence=args.execution_cadence,
        top_k=args.top_k,
        authoritative_only=not args.include_superseded,
    )
    if args.json:
        _print_json([h.model_dump(mode="json") for h in hits])
        return 0
    for h in hits:
        params = ", ".join(f"{k}={h.params[k]}" for k in sorted(h.params))
        print(f"{h.similarity.score:.4f}  {h.experiment_id}  "
              f"[{h.trial_role.value}/{h.authority.value}]")
        print(f"         params : {params}")
        print(f"         verdict: {h.headline_verdict.value if h.headline_verdict else '-'}"
              f"  {', '.join(h.reason_codes)}")
        print(f"         reasons: {h.similarity.explain()}")
    if not hits:
        print("no related experiments")
    return 0


def cmd_failures(reg: ExperimentRegistry, args: argparse.Namespace) -> int:
    cls = FailureClass(args.failure_class) if args.failure_class else None
    records = reg.failures(
        failure_class=cls, strategy_family=args.family, root_symbol=args.root
    )
    if args.json:
        _print_json([f.model_dump(mode="json") for f in records])
        return 0
    for f in records:
        scope = f.scope.value
        where = f.experiment_identity[:26] + "..." if f.experiment_identity else "-"
        print(f"[{f.failure_class.value}] {f.failure_id}")
        print(f"    scope   : {scope}  experiment: {where}")
        print(f"    code    : {f.failure_code}")
        print(f"    summary : {f.summary}")
        if f.resolution_commit:
            print(f"    resolved: {f.resolved} in {f.resolution_commit}")
    if not records:
        print("no failure records match")
    return 0


def cmd_history(reg: ExperimentRegistry, args: argparse.Namespace) -> int:
    view = reg.get(args.experiment)
    identity = view.experiment_identity
    edges = reg.lineage_edges(identity)
    by_identity = {
        v.experiment_identity: v.experiment_id
        for v in reg.experiments(include_superseded=True)
    }
    if args.json:
        _print_json({
            "experiment_id": view.experiment_id,
            "authority": view.authority.value,
            "authoritative": reg.resolve_authoritative(identity).experiment_id,
            "edges": [
                {
                    "source": by_identity.get(e.source_experiment_identity),
                    "relation": e.relation_type.value,
                    "target": by_identity.get(e.target_experiment_identity),
                    "note": e.note,
                }
                for e in edges
            ],
        })
        return 0
    print(f"{view.experiment_id}  [{view.authority.value}]")
    print(f"  authoritative answer: {reg.resolve_authoritative(identity).experiment_id}")
    for e in edges:
        print(f"  {by_identity.get(e.source_experiment_identity)} "
              f"--{e.relation_type.value}--> "
              f"{by_identity.get(e.target_experiment_identity)}")
    return 0


def cmd_list(reg: ExperimentRegistry, args: argparse.Namespace) -> int:
    views = reg.experiments(
        strategy_family=args.family,
        root_symbol=args.root,
        trial_role=TrialRole(args.role) if args.role else None,
        verdict=RegistryVerdict(args.verdict) if args.verdict else None,
        reason_code=args.reason_code,
        authoritative_only=not args.include_superseded,
        include_superseded=args.include_superseded,
    )
    if args.json:
        _print_json([
            {
                "experiment_id": v.experiment_id,
                "authority": v.authority.value,
                "verdict": v.verdict.value if v.verdict else None,
                "reason_codes": list(v.result.reason_codes) if v.result else [],
            }
            for v in views
        ])
        return 0
    for v in views:
        flag = "" if v.authority is Authority.AUTHORITATIVE else "  (SUPERSEDED)"
        verdict = v.verdict.value if v.verdict else "-"
        print(f"{v.experiment_id:<58} {verdict:<16}"
              f"{', '.join(v.result.reason_codes) if v.result else ''}{flag}")
    print(f"-- {len(views)} experiments")
    return 0


def cmd_memory(reg: ExperimentRegistry, args: argparse.Namespace) -> int:
    resp = FailureMemory(reg).lookup(
        strategy_family=args.family,
        root_symbol=args.root,
        asset_domain=AssetDomain(args.asset_domain),
        strategy_spec={"params": _parse_params(args.param)},
        top_k_related=args.top_k,
    )
    _print_json(resp.model_dump(mode="json"))
    return 0


def cmd_export(reg: ExperimentRegistry, args: argparse.Namespace) -> int:
    from alpha_agent.registry.exports import write_all

    written = write_all(reg, Path(args.out_dir))
    for name, path in sorted(written.items()):
        print(f"written -> {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m alpha_agent.registry.cli",
        description="Researcher-facing experiment registry / failure memory queries",
    )
    ap.add_argument("--db", default=str(DEFAULT_REGISTRY_PATH),
                    help=f"registry SQLite path (default {DEFAULT_REGISTRY_PATH})")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("summary", help="counts, verdicts, failures, digest")

    p = sub.add_parser("show", help="one experiment by friendly id or identity")
    p.add_argument("experiment")

    p = sub.add_parser("find-exact", help="has this exact experiment already been run?")
    p.add_argument("--identity", required=True)
    p.add_argument("--asset-domain", choices=[d.value for d in AssetDomain],
                    default=AssetDomain.FUTURES.value,
                    help="structural domain scope (Phase 6 ETF Research Pilot); "
                         "every real experiment today is FUTURES")

    p = sub.add_parser("find-related", help="deterministic near-duplicate retrieval")
    p.add_argument("--family", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--asset-domain", choices=[d.value for d in AssetDomain],
                    default=AssetDomain.FUTURES.value,
                    help="structural domain scope (Phase 6 ETF Research Pilot); "
                         "every real experiment today is FUTURES")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--signal-cadence", default="",
                   help="e.g. daily_trading_day -- enables the same_structure component")
    p.add_argument("--execution-cadence", default="",
                   help="e.g. native_1m_raw_contract")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--include-superseded", action="store_true")

    p = sub.add_parser("failures", help="typed failure memory")
    p.add_argument("--class", dest="failure_class",
                   choices=[c.value for c in FailureClass])
    p.add_argument("--family")
    p.add_argument("--root")

    p = sub.add_parser("history", help="lineage / supersession of one experiment")
    p.add_argument("experiment")

    p = sub.add_parser("list", help="filtered experiment list")
    p.add_argument("--family")
    p.add_argument("--root")
    p.add_argument("--role", choices=[r.value for r in TrialRole])
    p.add_argument("--verdict", choices=[v.value for v in RegistryVerdict])
    p.add_argument("--reason-code")
    p.add_argument("--include-superseded", action="store_true")

    p = sub.add_parser("memory", help="typed FailureMemoryResponse for a proposal")
    p.add_argument("--family", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--asset-domain", choices=[d.value for d in AssetDomain],
                    default=AssetDomain.FUTURES.value,
                    help="structural domain scope (Phase 6 ETF Research Pilot); "
                         "every real experiment today is FUTURES")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--top-k", type=int, default=10)

    p = sub.add_parser("export", help="write the Phase 14 artifacts")
    p.add_argument("--out-dir", default="outputs/phase_14")
    return ap


_COMMANDS = {
    "summary": cmd_summary,
    "show": cmd_show,
    "find-exact": cmd_find_exact,
    "find-related": cmd_find_related,
    "failures": cmd_failures,
    "history": cmd_history,
    "list": cmd_list,
    "memory": cmd_memory,
    "export": cmd_export,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db = Path(args.db)
    if not db.exists():
        print(f"no registry at {db}; run "
              "`python scripts/phase_14_import_phase_13_5c.py` first", file=sys.stderr)
        return 2
    with ExperimentRegistry(db) as reg:
        try:
            return _COMMANDS[args.command](reg, args)
        except UnknownExperiment as exc:
            print(str(exc), file=sys.stderr)
            return 2


if __name__ == "__main__":
    sys.exit(main())
