"""Phase 13.5C -- deterministic human-readable artifacts for the real-market
research + validation matrix (sections 22, 24, 25).

Consumes a :class:`~alpha_agent.validation.phase_13_5c_matrix.MatrixResult` and
writes, under ``outputs/phase_13_5c/``:

* ``<ROOT>__<FAMILY>__validation_report.json`` -- per canonical trial: the full
  frozen :class:`ValidationReport` plus the CORRECTED global BH q-value / global
  DSR / re-derived headline verdict, the regime evidence and the roll-close-mark
  provenance;
* ``VERDICT_TABLE.csv`` / ``TRIAL_FAMILY.csv`` / ``CROSS_MARKET_SUMMARY.csv`` /
  ``DATA_QUALITY_SENSITIVITY.csv``;
* ``REAL_MARKET_REPORT.md`` + ``REAL_MARKET_REPORT.json`` -- the section-25 final
  report.

Every file is deterministic (sorted keys, fixed column order). Failures are
preserved as typed entries, never dropped.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from alpha_agent.validation.enums import NullMethod, Verdict
from alpha_agent.validation.phase_13_5c_matrix import (
    GlobalFamily,
    MatrixResult,
    TrialRun,
    _global_dsr,
    global_verdict,
)


def _fmt(x: float | None, n: int = 6) -> float | None:
    if x is None:
        return None
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def _gating_null_p(run: TrialRun) -> float | None:
    for nr in run.report.null_results:
        if nr.method == NullMethod.CENTERED_BLOCK_BOOTSTRAP:
            return _fmt(nr.p_value)
    return None


def _cost_ratio(run: TrialRun, label: str) -> float | None:
    if run.report.cost_stress is None:
        return None
    for s in run.report.cost_stress.scenarios:
        if s.label == label:
            return _fmt(s.net_pnl_ratio_vs_baseline)
    return None


def _contract_economics_qa(out_dir: Path) -> dict:
    """Embed the typed contract-economics QA (``CONTRACT_ECONOMICS_QA.json``,
    written by ``scripts/phase_13_5c_contract_economics_qa.py``) if present.

    A root whose DERIVED USD point value does not match its published CME
    contract specification produces mis-scaled Fill economics: its PnL, cost
    ratio, Sharpe and verdict are NOT economically interpretable and are
    reported as ``DATA_QUALITY_FAILURE`` rather than as research results.
    """
    p = out_dir / "CONTRACT_ECONOMICS_QA.json"
    if not p.is_file():
        return {"status": "NOT_RUN",
                "note": "run scripts/phase_13_5c_contract_economics_qa.py"}
    blob = json.loads(p.read_text())
    failed = [r["root"] for r in blob.get("roots", []) if r.get("status") != "PASS"]
    return {
        "status": "PASS" if blob.get("pass") else "DATA_QUALITY_FAILURE",
        "failed_roots": sorted(failed),
        "artifact": "CONTRACT_ECONOMICS_QA.json",
        "roots": blob.get("roots", []),
        "impact_note": blob.get("impact_note", ""),
    }


@dataclass
class _TrialView:
    run: TrialRun
    verdict: str
    reason_codes: list[str]
    global_q: float | None
    global_dsr: float | None


def _views(runs: list[TrialRun], gf: GlobalFamily, policy) -> list[_TrialView]:
    out: list[_TrialView] = []
    for r in runs:
        outcome, q, dsr = global_verdict(r, gf, policy)
        out.append(_TrialView(
            run=r,
            verdict=outcome.verdict.value,
            reason_codes=[c.value for c in outcome.reason_codes],
            global_q=_fmt(q),
            global_dsr=_fmt(dsr.deflated_sharpe_ratio),
        ))
    return out


def _per_trial_json(v: _TrialView, gf: GlobalFamily, policy) -> dict:
    r = v.run
    rep = r.report
    dsr = _global_dsr(r, gf)
    return {
        "root_symbol": r.root_symbol,
        "family_key": r.family_key,
        "strategy_id": r.strategy_id,
        "strategy_fingerprint": r.strategy_fingerprint,
        "canonical_params": r.canonical_params,
        "headline_verdict_global_family": {
            "verdict": v.verdict,
            "reason_codes": v.reason_codes,
            "bh_q_value_over_107": v.global_q,
            "dsr_probability_over_107_effective_trials": v.global_dsr,
            "dsr_is_valid": dsr.is_valid,
            "dsr_daily_sharpe": _fmt(dsr.observed_daily_sharpe),
            "dsr_annualized_sharpe": _fmt(dsr.observed_annualized_sharpe),
            "dsr_skew": _fmt(dsr.return_skewness),
            "dsr_kurtosis": _fmt(dsr.return_kurtosis),
            "dsr_effective_trial_count": dsr.n_trials,
        },
        "per_family_local_verdict": {
            "verdict": rep.verdict.value,
            "reason_codes": [c.value for c in rep.reason_codes],
        },
        "oos_window": "2023-01-01..2024-12-31 (VALIDATION)",
        "oos_metrics": {
            "n_trading_days": rep.oos_metrics.n_trading_days,
            "n_fills": rep.oos_metrics.n_fills,
            "n_trades": rep.oos_metrics.n_trades,
            "gross_pnl_usd": _fmt(rep.oos_metrics.oos_gross_pnl_usd, 2),
            "costs_usd": _fmt(rep.oos_metrics.oos_costs_usd, 2),
            "net_pnl_usd": _fmt(rep.oos_metrics.oos_net_pnl_usd, 2),
            "daily_sharpe": _fmt(rep.oos_metrics.daily_sharpe),
            "annualized_sharpe": _fmt(rep.oos_metrics.annualized_sharpe),
        },
        "bootstrap_annualized_sharpe_ci": {
            "point": _fmt(rep.bootstrap.annualized_sharpe.point_estimate),
            "ci_low": _fmt(rep.bootstrap.annualized_sharpe.ci_low),
            "ci_high": _fmt(rep.bootstrap.annualized_sharpe.ci_high),
            "ci_level": rep.bootstrap.annualized_sharpe.ci_level,
        },
        "gating_null_centered_block_bootstrap_p": _gating_null_p(r),
        "null_results": [
            {
                "method": nr.method.value, "role": nr.role.value,
                "n_null_samples": nr.n_null_samples, "p_value": _fmt(nr.p_value),
                "observed_stat": _fmt(nr.observed_stat),
            }
            for nr in rep.null_results
        ],
        "walk_forward": {
            "n_folds_evaluated": rep.walk_forward.n_folds_evaluated,
            "positive_fold_count": rep.walk_forward.positive_fold_count,
            "fold_consistency": _fmt(rep.walk_forward.fold_consistency),
            "combined_oos_net_pnl_usd": _fmt(rep.walk_forward.combined_oos_net_pnl_usd, 2),
            "combined_oos_daily_sharpe": _fmt(rep.walk_forward.combined_oos_daily_sharpe),
            "folds": [
                {
                    "fold_index": f.fold_index,
                    "test_start_ts_ns": f.test_start_ts_ns,
                    "test_end_ts_ns": f.test_end_ts_ns,
                    "n_trading_days": f.n_trading_days,
                    "n_fills": f.n_fills, "n_trades": f.n_trades,
                    "net_pnl_usd": _fmt(f.oos_net_pnl_usd, 2),
                    "daily_sharpe": _fmt(f.daily_sharpe),
                    "is_positive": f.is_positive,
                }
                for f in rep.fold_summaries
            ],
        },
        "cost_stress": (
            {
                "scenarios": [
                    {
                        "label": s.label,
                        "commission_per_contract_usd": s.commission_per_contract_usd,
                        "net_pnl_usd": _fmt(s.oos_net_pnl_usd, 2),
                        "daily_sharpe": _fmt(s.daily_sharpe),
                        "net_pnl_ratio_vs_baseline": _fmt(s.net_pnl_ratio_vs_baseline),
                    }
                    for s in rep.cost_stress.scenarios
                ],
                "max_net_pnl_degradation": _fmt(rep.cost_stress.max_net_pnl_degradation),
            }
            if rep.cost_stress else None
        ),
        "parameter_stability": (
            {
                "n_evaluated": rep.parameter_stability.n_evaluated,
                "fraction_positive_sharpe": _fmt(rep.parameter_stability.fraction_positive_sharpe),
                "fraction_positive_net": _fmt(rep.parameter_stability.fraction_positive_net),
                "sharpe_dispersion_std": _fmt(rep.parameter_stability.sharpe_dispersion_std),
                "canonical_daily_sharpe": _fmt(rep.parameter_stability.canonical_daily_sharpe),
                "canonical_sharpe_percentile": _fmt(rep.parameter_stability.canonical_sharpe_percentile),
                "canonical_is_isolated_spike": rep.parameter_stability.canonical_is_isolated_spike,
            }
            if rep.parameter_stability else None
        ),
        "regime_evidence_volatility": {
            "status": r.regime.status.value,
            "buckets": [
                {"label": b.label, "n_days": b.n_days, "daily_sharpe": _fmt(b.daily_sharpe),
                 "net_pnl_usd": _fmt(b.net_pnl_usd, 2), "pnl_share": _fmt(b.pnl_share)}
                for b in r.regime.buckets
            ],
            "max_regime_pnl_share": _fmt(r.regime.max_regime_pnl_share),
            "trend_regime": "NOT_EVALUATED (no frozen trend-regime definition)",
        },
        "cross_market_evidence": rep.cross_market.model_dump(mode="json"),
        "risk_identity": {
            "risk_manager": "PassThroughRiskManager",
            "hard_risk_limits_enforced": False,
            "limitation": (
                "Phase 13.5C results are reference execution results under "
                "PassThroughRiskManager. Phase 08 production hard-risk limits are "
                "not enforced on the current CLI execution path."
            ),
        },
        "fingerprints": {
            "validation_fingerprint": rep.validation_fingerprint,
            "dataset_fingerprint": rep.dataset_fingerprint,
            "split_fingerprint": rep.split_fingerprint,
            "reliability_policy_fingerprint": rep.reliability_policy_fingerprint,
            "per_family_trial_family_fingerprint": rep.trial_family_fingerprint,
            "target_schedule_hash": rep.target_schedule_hash,
            "report_fingerprint": rep.report_fingerprint(),
            "global_bh_fdr_family_fingerprint": gf.family.family_fingerprint(),
        },
        "roll_close_marks": {
            "n_marks": r.n_roll_close_marks,
            "sha256": r.roll_close_marks_sha256,
            "note": (
                "auxiliary same-timestamp OUTGOING-contract closes for the C++ "
                "roll close-leg only; never a MarketEvent (Phase 13.5C)."
            ),
        },
        "cpp_runs": r.n_cpp_runs,
        "runtime_seconds": _fmt(r.runtime_s, 1),
    }


def write_all(result: MatrixResult, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = result.policy
    gf = result.global_family
    views = _views(result.canonical_runs, gf, policy)
    fdr = gf.fdr()

    # -- per-trial reports --
    per_trial_dir = out_dir
    for v in views:
        blob = _per_trial_json(v, gf, policy)
        name = f"{v.run.root_symbol}__{v.run.family_key.upper()}__validation_report.json"
        (per_trial_dir / name).write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")

    # -- VERDICT_TABLE.csv --
    vt_cols = [
        "root", "family", "headline_verdict", "reason_codes", "net_pnl_usd",
        "daily_sharpe", "annualized_sharpe", "bootstrap_aSR_ci_low",
        "bootstrap_aSR_ci_high", "gating_null_p", "bh_q_over_107", "dsr_over_107",
        "fold_consistency", "param_frac_pos_sharpe", "cost_1_5x_ratio",
        "cost_2_0x_ratio", "n_fills", "n_trades", "n_oos_days",
    ]
    with (out_dir / "VERDICT_TABLE.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(vt_cols)
        for v in sorted(views, key=lambda x: (x.run.family_key, x.run.root_symbol)):
            r, rep = v.run, v.run.report
            w.writerow([
                r.root_symbol, r.family_key, v.verdict, ";".join(v.reason_codes),
                _fmt(rep.oos_metrics.oos_net_pnl_usd, 2), _fmt(rep.oos_metrics.daily_sharpe),
                _fmt(rep.oos_metrics.annualized_sharpe),
                _fmt(rep.bootstrap.annualized_sharpe.ci_low),
                _fmt(rep.bootstrap.annualized_sharpe.ci_high),
                _gating_null_p(r), v.global_q, v.global_dsr,
                _fmt(rep.walk_forward.fold_consistency),
                _fmt(rep.parameter_stability.fraction_positive_sharpe) if rep.parameter_stability else None,
                _cost_ratio(r, "stress_1_5x"), _cost_ratio(r, "stress_2_0x"),
                rep.oos_metrics.n_fills, rep.oos_metrics.n_trades, rep.oos_metrics.n_trading_days,
            ])

    # -- TRIAL_FAMILY.csv (the 107 unique BH/FDR trials) --
    with (out_dir / "TRIAL_FAMILY.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial_index", "label", "role", "strategy_fingerprint",
                    "p_value", "bh_q_value", "rejected_at_q"])
        for d in fdr.decisions:
            t = gf.family.trials[d.trial_index]
            w.writerow([d.trial_index, t.label, t.role, t.strategy_fingerprint,
                        _fmt(d.p_value), _fmt(d.q_value), d.rejected])

    # -- CROSS_MARKET_SUMMARY.csv --
    with (out_dir / "CROSS_MARKET_SUMMARY.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "status", "n_roots", "n_positive_roots", "n_negative_roots",
                    "max_single_root_pnl_share", "herfindahl_pnl", "concentrated_in_one_root"])
        for fam, ev in sorted(result.cross_market.items()):
            w.writerow([
                fam, ev.status.value, ev.n_roots, ev.n_positive_roots,
                ev.n_roots - ev.n_positive_roots, _fmt(ev.max_single_root_pnl_share),
                _fmt(ev.herfindahl_pnl), ev.is_concentrated_in_one_root,
            ])

    # -- DATA_QUALITY_SENSITIVITY.csv --
    sens_by_key = {(r.family_key, r.root_symbol): r for r in result.sensitivity_runs}
    sens_gf = result.sensitivity_global_family
    with (out_dir / "DATA_QUALITY_SENSITIVITY.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["root", "family", "canonical_verdict", "canonical_daily_sharpe",
                    "canonical_net_pnl_usd", "excluded_verdict", "excluded_daily_sharpe",
                    "excluded_net_pnl_usd", "daily_sharpe_delta", "net_pnl_delta",
                    "verdict_changed"])
        for v in sorted(views, key=lambda x: (x.run.family_key, x.run.root_symbol)):
            key = (v.run.family_key, v.run.root_symbol)
            s = sens_by_key.get(key)
            c_sr = v.run.report.oos_metrics.daily_sharpe
            c_np = v.run.report.oos_metrics.oos_net_pnl_usd
            if s is None or sens_gf is None:
                w.writerow([v.run.root_symbol, v.run.family_key, v.verdict, _fmt(c_sr),
                            _fmt(c_np, 2), "NOT_RUN", None, None, None, None, None])
                continue
            s_outcome, _sq, _sd = global_verdict(s, sens_gf, policy)
            s_sr = s.report.oos_metrics.daily_sharpe
            s_np = s.report.oos_metrics.oos_net_pnl_usd
            w.writerow([
                v.run.root_symbol, v.run.family_key, v.verdict, _fmt(c_sr), _fmt(c_np, 2),
                s_outcome.verdict.value, _fmt(s_sr), _fmt(s_np, 2),
                _fmt((s_sr or 0) - (c_sr or 0)), _fmt((s_np or 0) - (c_np or 0), 2),
                s_outcome.verdict.value != v.verdict,
            ])

    # -- REAL_MARKET_REPORT.json + .md --
    final = _final_report(result, views, gf, fdr, out_dir)
    (out_dir / "REAL_MARKET_REPORT.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    (out_dir / "REAL_MARKET_REPORT.md").write_text(_final_md(final))
    return final


def _final_report(result: MatrixResult, views: list[_TrialView], gf: GlobalFamily, fdr,
                  out_dir: Path) -> dict:
    m = result.manifest
    by_family: dict[str, list[dict]] = {}
    for v in views:
        rep = v.run.report
        by_family.setdefault(v.run.family_key, []).append({
            "root": v.run.root_symbol,
            "headline_verdict": v.verdict,
            "reason_codes": v.reason_codes,
            "net_pnl_usd": _fmt(rep.oos_metrics.oos_net_pnl_usd, 2),
            "daily_sharpe": _fmt(rep.oos_metrics.daily_sharpe),
            "annualized_sharpe": _fmt(rep.oos_metrics.annualized_sharpe),
            "n_fills": rep.oos_metrics.n_fills,
            "n_trades": rep.oos_metrics.n_trades,
            "gating_null_p": _gating_null_p(v.run),
            "bh_q_over_107": v.global_q,
            "dsr_over_107": v.global_dsr,
            "fold_consistency": _fmt(rep.walk_forward.fold_consistency),
            "cost_1_5x_ratio": _cost_ratio(v.run, "stress_1_5x"),
            "cost_2_0x_ratio": _cost_ratio(v.run, "stress_2_0x"),
        })

    verdict_counts: dict[str, int] = {}
    for v in views:
        verdict_counts[v.verdict] = verdict_counts.get(v.verdict, 0) + 1

    econ_qa = _contract_economics_qa(out_dir)
    invalid_roots = set(econ_qa.get("failed_roots") or [])
    for fam_rows in by_family.values():
        for row in fam_rows:
            if row["root"] in invalid_roots:
                row["data_quality"] = "DATA_QUALITY_FAILURE"
                row["data_quality_reason"] = (
                    "derived contract point value does not match the published CME "
                    "specification -- economics mis-scaled, result NOT interpretable"
                )

    # a root whose economics are mis-scaled can never be holdout-eligible
    eligible = sorted(
        f"{v.run.root_symbol}/{v.run.family_key}"
        for v in views
        if v.verdict == Verdict.PASS.value and v.run.root_symbol not in invalid_roots
    )

    return {
        "phase": "13.5C",
        "generated_at": result.finished_at,
        "contract_economics_qa": econ_qa,
        "roots_with_uninterpretable_economics": sorted(invalid_roots),
        "started_at": result.started_at,
        "date_roles": {
            "research_train": "2018-01-01..2022-12-31",
            "headline_validation": "2023-01-01..2024-12-31",
            "locked_holdout_2025": "NEVER downloaded / queried / cost-fetched / loaded / featured / evaluated",
        },
        "frozen_candidate_manifest_fingerprint": m.manifest_fingerprint(),
        "bh_fdr_family_fingerprint": m.bh_fdr_trial_family.get("family_composition_fingerprint"),
        "global_bh_fdr_family_fingerprint_runtime": gf.family.family_fingerprint(),
        "unique_statistical_trial_count": gf.family.n_trials(),
        "unique_statistical_trial_count_expected": 107,
        "completed_cpp_execution_count": result.total_cpp_runs,
        "runtime_wall_seconds": _fmt(
            sum(r.runtime_s for r in result.canonical_runs)
            + sum(r.runtime_s for r in result.sensitivity_runs), 1
        ),
        "real_data_qa": "see outputs/phase_13_5c/roll_qa.json and scripts/phase_13_5c_build_data.py --verify",
        "results_by_family": by_family,
        "silver_bullet_nq": next(
            (b for b in by_family.get("silver_bullet", [])), None
        ),
        "cross_market_evidence": {
            fam: ev.model_dump(mode="json") for fam, ev in result.cross_market.items()
        },
        "verdict_counts": verdict_counts,
        "bh_fdr_n_rejected_at_q_0_10": fdr.n_rejected,
        "reliability_policy": "frozen Phase 13 default (configs/validation.yaml); no post-result threshold change",
        "risk_limitation": (
            "Phase 13.5C results are reference execution results under "
            "PassThroughRiskManager. Phase 08 production hard-risk limits are not "
            "enforced on the current CLI execution path."
        ),
        "roll_execution": (
            "positions held across real futures rolls priced on the Phase 04.5 "
            "same-timestamp basis via auxiliary EngineConfig::roll.close_marks "
            "(the acquired roll-overlap raw bars); frozen RejectDefer unchanged "
            "when no mark is present. rolls_priced_auxiliary_marks <= "
            "rolls_priced_contemporaneous."
        ),
        "data_quality_sensitivity": {
            "degraded_vendor_dates": ["2020-02-27", "2020-07-01", "2021-12-05", "2022-01-02", "2024-09-18"],
            "in_bh_fdr_denominator": False,
            "n_sensitivity_runs": len(result.sensitivity_runs),
            "verdict_changes": sorted(
                f"{s.run.root_symbol}/{s.run.family_key}"
                for s in _sens_verdict_changes(result, views)
            ),
        },
        "no_new_databento_spend": True,
        "holdout_2025_never_requested_or_accessed": True,
        "candidates_eligible_for_later_separately_approved_holdout": eligible,
        "software_or_data_failures": result.failures,
        "human_readable_catalog": "data/catalog/real_cme_dataset.csv (+ .json); data/by_name/<ROOT>/<ROLE>/*.json",
        "human_readable_outputs": "outputs/phase_13_5c/<ROOT>__<FAMILY>__validation_report.json + VERDICT_TABLE.csv etc.",
    }


def _sens_verdict_changes(result: MatrixResult, views: list[_TrialView]) -> list[_TrialView]:
    if result.sensitivity_global_family is None:
        return []
    by_key = {(v.run.family_key, v.run.root_symbol): v for v in views}
    changed = []
    for s in result.sensitivity_runs:
        base = by_key.get((s.family_key, s.root_symbol))
        if base is None:
            continue
        outcome, _q, _d = global_verdict(s, result.sensitivity_global_family, result.policy)
        if outcome.verdict.value != base.verdict:
            # reuse the sensitivity run's view slot for the label
            changed.append(_TrialView(run=s, verdict=outcome.verdict.value,
                                      reason_codes=[], global_q=None, global_dsr=None))
    return changed


def _final_md(final: dict) -> str:
    lines = [
        "# Phase 13.5C -- Real-Market Research + Validation Matrix",
        "",
        f"_Generated {final['generated_at']}. STRICTLY OFFLINE -- no Databento spend, 2025 holdout never touched._",
        "",
        "## Date roles (frozen)",
        f"- RESEARCH / TRAIN: {final['date_roles']['research_train']}",
        f"- HEADLINE VALIDATION (OOS): {final['date_roles']['headline_validation']}",
        f"- LOCKED HOLDOUT 2025: {final['date_roles']['locked_holdout_2025']}",
        "",
        "## Identity",
        f"- frozen candidate manifest fingerprint: `{final['frozen_candidate_manifest_fingerprint']}`",
        f"- BH/FDR family composition fingerprint: `{final['bh_fdr_family_fingerprint']}`",
        (
            f"- unique statistical trials: **{final['unique_statistical_trial_count']}** "
            f"(expected {final['unique_statistical_trial_count_expected']})"
        ),
        f"- completed C++ executions: **{final['completed_cpp_execution_count']}**",
        f"- wall time: {final['runtime_wall_seconds']} s",
        "",
        "## Verdict counts (headline, frozen policy over the corrected 107-trial family)",
    ]
    for k, v in sorted(final["verdict_counts"].items()):
        lines.append(f"- {k}: {v}")
    eq = final.get("contract_economics_qa", {})
    if eq.get("status") == "DATA_QUALITY_FAILURE":
        bad = ", ".join(eq.get("failed_roots", []))
        lines += [
            "",
            "## DATA QUALITY FAILURE -- contract economics",
            "",
            (
                f"Roots whose DERIVED contract point value does not match the published CME "
                f"specification: **{bad}**. Every Fill-derived economic number for those roots "
                f"(gross / net PnL, cost ratio, daily and annualized Sharpe) is mis-scaled, so "
                f"their rows below are **NOT economically interpretable** and are not eligible "
                f"for any later holdout evaluation. See `CONTRACT_ECONOMICS_QA.json`."
            ),
            "",
            "| root | derived point value | published | error factor |",
            "|---|---|---|---|",
        ]
        for r in eq.get("roots", []):
            if r.get("status") != "PASS":
                lines.append(
                    f"| {r['root']} | {r['derived_point_value_usd']} | "
                    f"{r['reference_point_value_usd']} | {r['point_value_error_factor']}x |"
                )
        lines.append("")
    lines += ["", "## Results by family", ""]
    for fam in ("tsmom", "ma_trend", "breakout", "mean_reversion", "silver_bullet"):
        rows = final["results_by_family"].get(fam, [])
        if not rows:
            continue
        lines.append(f"### {fam}")
        lines.append("")
        lines.append("| root | verdict | net PnL $ | daily SR | ann SR | null p | BH q (107) | DSR (107) | fold cons | 1.5x | 2.0x |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda x: x["root"]):
            lines.append(
                f"| {r['root']} | {r['headline_verdict']} | {r['net_pnl_usd']} | {r['daily_sharpe']} | "
                f"{r['annualized_sharpe']} | {r['gating_null_p']} | {r['bh_q_over_107']} | {r['dsr_over_107']} | "
                f"{r['fold_consistency']} | {r['cost_1_5x_ratio']} | {r['cost_2_0x_ratio']} |"
            )
        lines.append("")
    lines += [
        "## Cross-market evidence (descriptive -- NOT additional trials)",
        "",
        "| family | status | +roots | -roots | max root PnL share | Herfindahl | one-root? |",
        "|---|---|---|---|---|---|---|",
    ]
    for fam, ev in sorted(final["cross_market_evidence"].items()):
        lines.append(
            f"| {fam} | {ev['status']} | {ev['n_positive_roots']} | "
            f"{ev['n_roots'] - ev['n_positive_roots']} | {round(ev['max_single_root_pnl_share'], 3)} | "
            f"{round(ev['herfindahl_pnl'], 3)} | {ev['is_concentrated_in_one_root']} |"
        )
    lines += [
        "",
        "## Data-quality sensitivity (degraded vendor days excluded -- robustness of the SAME hypotheses)",
        f"- degraded dates: {final['data_quality_sensitivity']['degraded_vendor_dates']}",
        f"- in BH/FDR denominator: {final['data_quality_sensitivity']['in_bh_fdr_denominator']}",
        f"- sensitivity runs: {final['data_quality_sensitivity']['n_sensitivity_runs']}",
        f"- verdict changes: {final['data_quality_sensitivity']['verdict_changes'] or 'none'}",
        "",
        "## Risk identity",
        f"- {final['risk_limitation']}",
        "",
        "## Roll execution",
        f"- {final['roll_execution']}",
        "",
        "## Holdout / spend discipline",
        f"- no new Databento spend: {final['no_new_databento_spend']}",
        f"- 2025 holdout never requested or accessed: {final['holdout_2025_never_requested_or_accessed']}",
        "",
        "## Candidates eligible for a later, separately-approved one-time holdout evaluation",
        f"- {final['candidates_eligible_for_later_separately_approved_holdout'] or 'none'}",
        "",
        "## Software / data failures",
        f"- {final['software_or_data_failures'] or 'none'}",
        "",
        "## Human-readable layers",
        f"- catalog: {final['human_readable_catalog']}",
        f"- outputs: {final['human_readable_outputs']}",
        "",
    ]
    return "\n".join(lines) + "\n"
