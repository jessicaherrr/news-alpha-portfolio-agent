"""Human-readable Phase 15B report from an adjudicated family.

Writes a JSON record and a Markdown summary. The counts are laid out so they are
impossible to confuse (prompt 15B.1 s.18): 60 statistical hypotheses, 12 pooled
fitted pipeline definitions, 1,200 model fits when every fold clears its gates,
240 planned non-placebo C++ economic evaluations, conditional placebo executions
separately, typed refusals separately, and no holdout candidates opened.

Tested with synthetic results in Phase 15B.1; run on the real family in 15B.2.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from alpha_agent.ml.family_adjudication import Phase15FamilyResult
from alpha_agent.ml.manifest import MLCandidateManifest


def _round(x: float | None, n: int = 4) -> float | None:
    return None if x is None else round(float(x), n)


def build_report_payload(
    result: Phase15FamilyResult,
    *,
    manifest: MLCandidateManifest,
    mode: str,
    code_commit: str = "",
    environment: dict | None = None,
) -> dict:
    est = manifest.compute_estimate
    n_model_fits = sum(
        p.training_report.n_model_fits
        for p in result.pipelines
        if p.training_report is not None
    )
    n_pipelines_trained = sum(1 for p in result.pipelines if not p.refused)
    # engine calls SERVED (PRIMARY + META, 3 costs each, per trial) vs EXECUTED
    # (PRIMARY schedules repeat across pipelines and are cached, so the executed
    # count converges to the frozen 240 economic budget for a full real run).
    n_cpp_econ_executed = result.n_engine_runs_executed if result.engine.startswith("cpp") else 0

    trials = []
    for t in result.trials:
        econ = t.economics
        base_meta = (
            next(m for m in econ.meta if m.cost_scenario_label == "baseline_1_0x")
            if econ else None
        )
        base_prim = (
            next(m for m in econ.primary if m.cost_scenario_label == "baseline_1_0x")
            if econ else None
        )
        trials.append(
            {
                "trial_label": t.trial_label,
                "experiment_identity": t.experiment_identity,
                "friendly_experiment_id": t.friendly_experiment_id,
                "root_symbol": t.root_symbol,
                "primary_family": t.primary_family,
                "model_family": t.model_family,
                "regime_kind": t.regime_kind,
                "trial_role": t.trial_role,
                "verdict": t.adjudication.verdict,
                "reason_codes": list(t.adjudication.reason_codes),
                "refused": t.refused,
                "typed_refusal": t.typed_refusal,
                "bh_p_value": _round(t.bh_p_value),
                "bh_q_value": _round(t.bh_q_value),
                "bh_rejected_at_q": t.bh_rejected_at_q,
                "gating_null_p": _round(t.gating_null.p_value) if t.gating_null else None,
                "dsr_probability": _round(t.dsr.deflated_sharpe_ratio) if t.dsr else None,
                "dsr_n_trials": t.dsr.n_trials if t.dsr else None,
                "dsr_family_380": _round(t.dsr_family_380.deflated_sharpe_ratio)
                if t.dsr_family_380 else None,
                "base_reliability_verdict": t.adjudication.base_reliability_verdict,
                "meta_daily_sharpe": _round(t.adjudication.meta_daily_sharpe),
                "primary_daily_sharpe": _round(t.adjudication.primary_daily_sharpe),
                "meta_beats_primary_daily_sharpe": t.adjudication.meta_beats_primary,
                "placebo_ran": t.placebo.ran if t.placebo else False,
                "placebo_empirical_p": _round(t.placebo.empirical_p_value) if t.placebo else None,
                "placebo_significant": t.placebo.significant if t.placebo else None,
                "take_rate": _round(econ.take_rate) if econ else None,
                "aligned_support_days": econ.aligned_support_days if econ else None,
                "meta_net_pnl_usd": _round(base_meta.net_pnl_usd, 2) if base_meta else None,
                "primary_net_pnl_usd": _round(base_prim.net_pnl_usd, 2) if base_prim else None,
                "economic_delta_vs_primary_usd": (
                    {k: _round(v, 2) for k, v in econ.economic_delta_vs_primary_usd.items()}
                    if econ else None
                ),
                "diagnostics": (
                    {
                        "roc_auc": _round(t.diagnostics.roc_auc),
                        "pr_auc": _round(t.diagnostics.pr_auc),
                        "log_loss": _round(t.diagnostics.log_loss),
                        "brier_score": _round(t.diagnostics.brier_score),
                        "calibration_mae": _round(t.diagnostics.calibration_mae),
                        "precision": _round(t.diagnostics.precision),
                        "recall": _round(t.diagnostics.recall),
                        "note": "DIAGNOSTIC ONLY -- never gates a verdict",
                    }
                    if t.diagnostics else None
                ),
            }
        )

    return {
        "phase": "15B",
        "mode": mode,
        "generated_at": datetime.now(UTC).isoformat(),
        "code_commit": code_commit,
        "engine": result.engine,
        "is_full_family": result.is_full_family,
        "environment": environment or {},
        "manifest_fingerprint": result.manifest_fingerprint,
        "validation_spec_fingerprint": result.validation_spec_fingerprint,
        "reliability_policy_fingerprint": result.reliability_policy_fingerprint,
        "counts": {
            "statistical_hypotheses_bh_family": (
                60 if result.is_full_family else result.bh_family_size
            ),
            "predeclared_bh_family_always": 60,
            "headline_trials": result.n_headline,
            "regime_ablation_trials": result.n_ablation,
            "pooled_fitted_pipeline_definitions": est.n_distinct_fitted_pipelines,
            "pipelines_run": result.n_pipelines_run,
            "pipelines_trained_not_refused": n_pipelines_trained,
            "model_fits_when_all_folds_clear_gates": est.n_model_fits_total,
            "model_fits_this_run": n_model_fits,
            "planned_non_placebo_cpp_economic_evaluations": (
                est.n_cpp_evaluations_total_excluding_placebo
            ),
            "cpp_economic_evaluations_this_run": n_cpp_econ_executed,
            "engine_calls_served": result.n_engine_calls,
            "engine_runs_executed": result.n_engine_runs_executed,
            "engine_cache_hits": result.n_engine_cache_hits,
            "placebo_triggered_trials": result.placebo_triggered_trials,
            "placebo_cpp_draw_evaluations": result.placebo_cpp_draw_evaluations,
            "placebo_draws_per_triggered_trial": 20,
            "placebo_draws_prevented_by_typed_refusal": dict(result.placebo_draws_prevented),
            "placebo_cpp_draw_evaluations_ceiling": est.n_cpp_evaluations_placebo_max,
            "placebo_in_bh_denominator": False,
            "typed_refusals": result.n_refused,
            "dsr_family_effective_trial_count": result.dsr_family_effective_trial_count,
            "dsr_per_trial_breadth": dict(est.dsr_effective_trial_count_by_model_family),
        },
        "verdict_counts": result.verdict_counts,
        "n_pass": result.n_pass,
        "n_reject": result.n_reject,
        "n_inconclusive": result.n_inconclusive,
        "n_refused": result.n_refused,
        "holdout_candidates_auto_opened": [],
        "holdout_pass_candidates": list(result.holdout_candidates),
        "holdout_note": (
            "2025 remains LOCKED. A PASS here is a CANDIDATE for a later, separately "
            "approved and separately frozen holdout evaluation -- it is never auto-opened."
        ),
        "pipelines": [
            {
                "label": p.key.label,
                "primary_family": p.key.primary_family,
                "model_family": p.key.model_family,
                "regime_kind": p.key.regime_kind,
                "refused": p.refused,
                "refusal_reason": p.refusal_reason,
                "n_pooled_events": p.n_pooled_events,
                "per_root_event_counts": p.per_root_event_counts,
                "n_oof_predictions": p.n_oof_predictions,
                "feature_missing_events": p.feature_missing_events,
                "corpus_reconciles": p.corpus_reconciles,
                "oof_prediction_hash": (
                    p.training_report.oof_prediction_hash if p.training_report else None
                ),
                "report_fingerprint": (
                    p.training_report.report_fingerprint() if p.training_report else None
                ),
            }
            for p in result.pipelines
        ],
        "trials": trials,
    }


def write_report(payload: dict, *, out_dir: str | Path, stem: str = "PHASE_15B_REPORT") -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def _markdown(p: dict) -> str:
    c = p["counts"]
    synthetic = str(p.get("engine", "")).startswith("synthetic")
    lines = [
        f"# Phase 15B report ({p['mode']})",
        "",
    ]
    if synthetic:
        banner = (
            "> **SYNTHETIC INTEGRATION FIXTURE -- NOT A RESEARCH RESULT.** Every number "
            "below comes from a deterministic synthetic engine and synthetic features. No "
            "real 2018-2024 event corpus was built, no real model was fitted, and no "
            "production registry row was written. The verdict counts prove the "
            "adjudication DAG is wired, not that any strategy has alpha. The real result "
            "is Phase 15B.2 (`--run-real`)."
        )
        lines += [banner, ""]
    lines += [
        f"- generated: {p['generated_at']}",
        f"- engine: `{p['engine']}`  ({'FULL 60-family' if p['is_full_family'] else 'SCOPED integration'})",
        f"- manifest: `{p['manifest_fingerprint']}`",
        f"- validation plane: `{p['validation_spec_fingerprint']}`",
        f"- reliability policy: `{p['reliability_policy_fingerprint']}` (inherited, unchanged)",
        "",
        "## Counts (each a different thing)",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| statistical hypotheses (BH family) | **{c['statistical_hypotheses_bh_family']}** |",
        f"| predeclared BH family, always | {c['predeclared_bh_family_always']} |",
        f"| headline / regime-ablation trials | {c['headline_trials']} / {c['regime_ablation_trials']} |",
        f"| pooled fitted pipeline definitions | {c['pooled_fitted_pipeline_definitions']} |",
        f"| model fits when all folds clear gates | {c['model_fits_when_all_folds_clear_gates']} |",
        f"| model fits this run | {c['model_fits_this_run']} |",
        f"| planned non-placebo C++ economic evaluations | {c['planned_non_placebo_cpp_economic_evaluations']} |",
        f"| economics-engine calls served / runs executed / cache hits | {c['engine_calls_served']} / {c['engine_runs_executed']} / {c['engine_cache_hits']} |",
        f"| placebo triggered trials | {c['placebo_triggered_trials']} |",
        f"| placebo C++ draw evaluations (20 per triggered trial) | {c['placebo_cpp_draw_evaluations']} (ceiling {c['placebo_cpp_draw_evaluations_ceiling']}) |",
        f"| typed refusals | {c['typed_refusals']} |",
        f"| DSR family effective trial count | {c['dsr_family_effective_trial_count']} |",
        f"| DSR per-trial breadth | {c['dsr_per_trial_breadth']} |",
        "",
        "## Verdicts",
        "",
        f"- PASS: **{p['n_pass']}**",
        f"- REJECT: {p['n_reject']}",
        f"- INCONCLUSIVE: {p['n_inconclusive']}",
        f"- REFUSED (typed, first-class evidence): {p['n_refused']}",
        "",
        f"Holdout candidates auto-opened: **{len(p['holdout_candidates_auto_opened'])}** (2025 stays locked).",
        "",
        "## Trials",
        "",
        "| trial | verdict | base | bh p | bh q | DSR | meta>prim | placebo p | reasons |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in p["trials"]:
        lines.append(
            f"| {t['trial_label']} | {t['verdict']} | "
            f"{t['base_reliability_verdict'] or '-'} | {t['bh_p_value']} | {t['bh_q_value']} | "
            f"{t['dsr_probability']} | {t['meta_beats_primary_daily_sharpe']} | "
            f"{t['placebo_empirical_p']} | {'; '.join(t['reason_codes'])} |"
        )

    if "orchestration_dag" in p:
        lines += ["", "## Orchestration DAG", ""]
        lines += [f"{i + 1}. {s.split('. ', 1)[-1]}" for i, s in enumerate(p["orchestration_dag"])]
    if "integration_checklist" in p:
        lines += ["", "## Pre-run integration checklist", "", "| check | result |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in p["integration_checklist"].items()]
    if "no_real_performance" in p:
        lines += ["", "## No real performance", ""]
        lines += [f"- {k}: {v}" for k, v in p["no_real_performance"].items()]
    if "remaining_blockers_to_mechanical_run_real" in p:
        lines += ["", "## Remaining blockers to a mechanical `--run-real`", ""]
        for b in p["remaining_blockers_to_mechanical_run_real"]:
            lines += [
                f"- **{b['blocker']}** ({b['severity']}): {b['detail']}",
                f"  - resolution for 15B.2: {b['resolution_for_15b_2']}",
            ]
    lines.append("")
    return "\n".join(lines)
