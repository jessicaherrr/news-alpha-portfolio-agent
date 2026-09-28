"""Deterministic human-readable exports of the registry (section 17).

Friendly names in the visible columns; fingerprints live in the metadata
columns / blocks, never as the only human-visible identity. Nothing here
duplicates a large Phase 13 artifact -- the exports reference them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from alpha_agent.registry.enums import Authority, TrialRole
from alpha_agent.registry.sqlite_registry import ExperimentRegistry

EXPERIMENT_INDEX_COLUMNS = [
    "experiment_id", "display_name", "phase", "authority", "root_symbol",
    "strategy_family", "trial_role", "parameter_variant_label", "params",
    "headline_verdict", "reason_codes", "net_pnl_usd", "daily_sharpe",
    "annualized_sharpe", "gating_null_p", "bh_p_value", "bh_q", "dsr_probability",
    "fold_consistency", "n_trades", "n_oos_days", "holdout_eligible",
    "market_window", "code_commit", "evidence_completeness",
    "source_artifact", "identity_schema", "experiment_identity",
    "strategy_fingerprint", "target_schedule_hash", "report_fingerprint",
]

FAILURE_MEMORY_COLUMNS = [
    "failure_id", "failure_class", "failure_code", "scope", "experiment_id",
    "root_symbol", "strategy_family", "summary", "resolved", "resolution_commit",
    "superseded_by", "mechanism", "experiment_identity",
]


def _params(view) -> str:
    p = view.experiment.strategy_spec_json.get("params", {})
    return ";".join(f"{k}={p[k]}" for k in sorted(p))


def write_experiment_index(registry: ExperimentRegistry, path: Path) -> int:
    views = registry.experiments(include_superseded=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(EXPERIMENT_INDEX_COLUMNS)
        for v in views:
            e, r = v.experiment, v.result
            w.writerow([
                e.experiment_id, e.display_name, e.phase, v.authority.value,
                e.root_symbol, e.strategy_family, e.trial_role.value,
                e.parameter_variant_label, _params(v),
                r.headline_verdict.value if r else "",
                ";".join(r.reason_codes) if r else "",
                r.net_pnl_usd if r else "", r.daily_sharpe if r else "",
                r.annualized_sharpe if r else "", r.gating_null_p if r else "",
                r.bh_p_value if r else "", r.bh_q if r else "",
                r.dsr_probability if r else "", r.fold_consistency if r else "",
                r.n_trades if r else "", r.n_oos_days if r else "",
                int(r.holdout_eligible) if r else "",
                f"{e.market_window.start_date}..{e.market_window.end_date}",
                e.code_commit, r.evidence_completeness if r else "",
                (r.source_artifact or "") if r else "", e.identity_schema,
                e.experiment_identity, e.strategy_fingerprint,
                e.target_schedule_hash or "", e.report_fingerprint or "",
            ])
    return len(views)


def write_failure_memory(registry: ExperimentRegistry, path: Path) -> int:
    failures = registry.failures()
    id_by_identity = {
        v.experiment_identity: v.experiment_id
        for v in registry.experiments(include_superseded=True)
    }
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FAILURE_MEMORY_COLUMNS)
        for f in failures:
            w.writerow([
                f.failure_id, f.failure_class.value, f.failure_code, f.scope.value,
                id_by_identity.get(f.experiment_identity or "", ""),
                f.root_symbol or "", f.strategy_family or "", f.summary,
                int(f.resolved), f.resolution_commit or "", f.superseded_by or "",
                f.mechanism, f.experiment_identity or "",
            ])
    return len(failures)


def write_supersession_graph(registry: ExperimentRegistry, path: Path) -> dict:
    views = {v.experiment_identity: v for v in registry.experiments(include_superseded=True)}
    nodes = [
        {
            "experiment_id": v.experiment_id,
            "display_name": v.experiment.display_name,
            "phase": v.experiment.phase,
            "code_commit": v.experiment.code_commit,
            "authority": v.authority.value,
            "headline_verdict": v.verdict.value if v.verdict else None,
            "experiment_identity": v.experiment_identity,
        }
        for v in sorted(views.values(), key=lambda x: x.experiment_id)
        if v.experiment.trial_role is TrialRole.CANONICAL
    ]
    edges = [
        {
            "source_experiment_id": views[e.source_experiment_identity].experiment_id,
            "target_experiment_id": views[e.target_experiment_identity].experiment_id,
            "relation_type": e.relation_type.value,
            "note": e.note,
            "source_experiment_identity": e.source_experiment_identity,
            "target_experiment_identity": e.target_experiment_identity,
        }
        for e in registry.lineage_edges()
    ]
    graph = {
        "schema": "registry-supersession-graph/1",
        "canonical_nodes": nodes,
        "edges": edges,
        "n_superseded": sum(1 for n in nodes if n["authority"] == Authority.SUPERSEDED.value),
    }
    path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return graph


def display_path(path: str | Path) -> str:
    """Repo-relative when possible, so a committed artifact is not stamped with
    one machine's absolute home directory."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def write_registry_summary(registry: ExperimentRegistry, path: Path) -> dict:
    summary = registry.summary().model_dump(mode="json")
    summary["registry_path"] = display_path(summary["registry_path"])
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_registry_report(registry: ExperimentRegistry, path: Path) -> None:
    from alpha_agent.registry.report import render_registry_report

    path.write_text(render_registry_report(registry), encoding="utf-8")


def write_all(registry: ExperimentRegistry, out_dir: Path) -> dict[str, str]:
    """Write every Phase 14 artifact; returns ``{artifact: path}``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_index(registry, out_dir / "EXPERIMENT_INDEX.csv")
    write_failure_memory(registry, out_dir / "FAILURE_MEMORY.csv")
    write_supersession_graph(registry, out_dir / "SUPERSESSION_GRAPH.json")
    write_registry_summary(registry, out_dir / "REGISTRY_SUMMARY.json")
    write_registry_report(registry, out_dir / "REGISTRY_REPORT.md")
    return {
        "EXPERIMENT_INDEX.csv": str(out_dir / "EXPERIMENT_INDEX.csv"),
        "FAILURE_MEMORY.csv": str(out_dir / "FAILURE_MEMORY.csv"),
        "SUPERSESSION_GRAPH.json": str(out_dir / "SUPERSESSION_GRAPH.json"),
        "REGISTRY_SUMMARY.json": str(out_dir / "REGISTRY_SUMMARY.json"),
        "REGISTRY_REPORT.md": str(out_dir / "REGISTRY_REPORT.md"),
    }
