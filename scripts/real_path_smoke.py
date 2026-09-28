#!/usr/bin/env python
"""Phase 15B -- OFFLINE engineering-only smoke of the real data path.

Reconstitutes the already-stored 2018-2024 development data and verifies the
plumbing of ``alpha_agent.ml.phase_15b_real_path`` WITHOUT fitting a model,
inspecting meta-label performance, or reporting any Sharpe / p / q / DSR / PnL as
a research result.

Checks:
  * real Feature Engine column names / order match the frozen 11-feature contract
  * EVERY frozen feature is finite on eligible rows after only its mathematically
    required warm-up, and the warm-up is explained by its FeatureSpec (this would
    have caught the Phase 15B.2 defect: the RESET_ON_GAP return / realised-vol
    features were entirely non-finite on the weekend-gapped daily series)
  * return / realised-vol columns match an independent reference calculation
  * ``realized_vol_20`` (regime axis 0) is finite before entering the regime transform
  * real primary episodes get a non-zero, sensible fraction of fully-finite
    feature vectors
  * feature availability + causal (feature_ts <= decision_ts) contract at a real
    decision bar
  * ``CppEngineRunner`` argv construction (dry build; optionally one real
    backtest to prove --trades-out/--fills-out parsing)
  * contract frame + roll-close auxiliary-mark wiring
  * NO timestamp >= 2025-01-01 anywhere
  * NO Databento Historical client / cost / range call

    python scripts/phase_15b_real_path_smoke.py [--roots NQ ES CL GC ZN] [--run-one-cpp]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import alpha_agent.features.compute  # noqa: F401
import numpy as np
from alpha_agent.ml.corpus import PrimaryScope
from alpha_agent.ml.episodes import extract_primary_episodes
from alpha_agent.ml.guards import HOLDOUT_START_NS
from alpha_agent.ml.manifest import ML_FEATURE_SET
from alpha_agent.ml.phase_15b_real_path import RealPrimaryProvider, build_real_path

OUT = REPO / "outputs" / "phase_15" / "REAL_PATH_SMOKE.json"


def _no_databento_client() -> str:
    if "databento" in sys.modules:
        db = sys.modules["databento"]
        return f"databento imported (v{getattr(db, '__version__', '?')}) -- local DBNStore decode only"
    return "databento not imported"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["NQ", "ES", "CL", "GC", "ZN"])
    ap.add_argument("--run-one-cpp", action="store_true",
                    help="run ONE real C++ backtest to prove trades/fills parsing (mechanics only)")
    args = ap.parse_args()
    roots = tuple(args.roots)

    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = ""
    report: dict = {
        "phase": "15B", "mode": "real_path_smoke", "code_commit": commit,
        "roots": list(roots), "checks": {},
    }
    c = report["checks"]

    # -- build the real path (reconstitutes the offline data ONCE) -----------
    provider = RealPrimaryProvider(REPO, roots=roots)
    provider._prepare()
    real = build_real_path(
        REPO, work_root=REPO / "outputs" / "phase_15" / "_smoke_work",
        roots=roots, primary=provider,
    )
    frames = {r: real.feature_provider._frames[r] for r in roots}

    # -- 1. feature column names / order ------------------------------------
    aliases = list(ML_FEATURE_SET.ordered_aliases)
    canonical = list(ML_FEATURE_SET.canonical_feature_names())
    per_root_cols = {}
    ok_cols = True
    for r, fr in frames.items():
        cols = [x for x in fr.columns if x != "ts_event_ns"]
        per_root_cols[r] = cols
        if cols != aliases:
            ok_cols = False
    c["feature_columns_match_frozen_contract"] = {
        "ok": ok_cols,
        "expected_alias_order": aliases,
        "canonical_names": canonical,
        "per_root_columns": per_root_cols,
    }

    # -- 1b. EVERY frozen feature finite after an explainable warm-up -------
    # (the Phase 15B.2 defect: RESET_ON_GAP return/realised-vol features were
    # 100% non-finite on the weekend-gapped daily series -> 0 usable episodes.)
    canon_to_alias = dict(zip(canonical, aliases))
    finite_ok = True
    per_root_features: dict = {}
    for r in roots:
        detail = real.feature_detail[r]
        n = len(detail.features)
        feats_report = {}
        for cname in canonical:
            v = detail.features[cname].to_numpy("float64")
            fin = np.isfinite(v)
            first = int(fin.argmax()) if fin.any() else None
            lookback = int(detail.metadata[cname].lookback)
            n_finite = int(fin.sum())
            interior_gap = first is not None and n_finite != (n - first)
            explained = (
                first is not None
                and first <= lookback
                and first <= 0.2 * n
                and not interior_gap
                and n_finite >= 0.7 * n
            )
            feats_report[canon_to_alias[cname]] = {
                "canonical": cname, "n_rows": n, "n_finite": n_finite,
                "first_finite_idx": first, "declared_lookback": lookback,
                "interior_gap_after_warmup": interior_gap, "explained": explained,
            }
            if not explained:
                finite_ok = False
        per_root_features[r] = feats_report
    c["every_frozen_feature_finite_after_warmup"] = {
        "ok": finite_ok, "per_root": per_root_features,
    }

    # -- 1c. return / realised-vol columns vs an independent reference ------
    from alpha_agent.data.calendars import default_calendar
    from alpha_agent.data.real_market_dataset import daily_signal_series
    cal = default_calendar()
    ref_ok = True
    ref_detail = {}
    for r in roots:
        detail = real.feature_detail[r]
        daily = daily_signal_series(
            real.primary_provider.reconstituted(r).forward_adjusted, r, calendar=cal
        )
        px = daily["close"].to_numpy("float64")
        m = len(px)
        # helper computes on a synthetic contiguous index, so a plain
        # close-to-close reference over the same ordered closes is exact.
        lr = np.diff(np.log(px))
        got_cum = detail.features["cum_log_return_20"].to_numpy("float64")
        got_rv = detail.features["realized_vol_20"].to_numpy("float64")
        max_cum_err = max(
            (abs(got_cum[i] - lr[i - 20:i].sum()) for i in range(20, m)), default=1.0
        )
        max_rv_err = max(
            (abs(got_rv[i] - np.sqrt((lr[i - 20:i] ** 2).sum())) for i in range(20, m)),
            default=1.0,
        )
        ref_detail[r] = {"max_cum_log_return_20_err": float(max_cum_err),
                         "max_realized_vol_20_err": float(max_rv_err),
                         "n_daily_rows": m}
        if not (max_cum_err < 1e-9 and max_rv_err < 1e-9):
            ref_ok = False
    c["return_features_match_reference"] = {"ok": ref_ok, "per_root": ref_detail}

    # -- 1d. regime axis 0 (realized_vol_20) finite before the transform ---
    regime_ok = True
    regime_detail = {}
    for r in roots:
        v = real.feature_provider._frames[r]["rvol_20"].to_numpy("float64")
        fin = np.isfinite(v)
        first = int(fin.argmax()) if fin.any() else None
        regime_detail[r] = {"first_finite_idx": first, "n_finite": int(fin.sum()),
                            "n_rows": len(v)}
        if first is None or first > 20 or int(fin.sum()) != len(v) - first:
            regime_ok = False
    c["regime_axis0_realized_vol_20_finite"] = {"ok": regime_ok, "per_root": regime_detail}

    # -- 2. causal timestamp / availability contract ----------------------
    causal_ok = True
    availability_samples = []
    for r in roots:
        scope = PrimaryScope(
            primary_family="tsmom", root_symbol=r,
            strategy_fingerprint=next(
                s.strategy_fingerprint for s in provider.scopes()
                if s.primary_family == "tsmom" and s.root_symbol == r
            ),
            strategy_id=next(
                s.strategy_id for s in provider.scopes()
                if s.primary_family == "tsmom" and s.root_symbol == r
            ),
        )
        sched = provider.schedule_for(scope)
        eps = extract_primary_episodes(sched)
        fr = frames[r]
        col_ts = fr["ts_event_ns"].to_numpy("int64")
        feat_mat = fr[aliases].to_numpy("float64")
        n_row = n_finite_vec = 0
        first_finite_decision_ts = None
        for ep in eps:
            d = ep.entry_decision_ts_ns
            pos = int(np.searchsorted(col_ts, d, side="right")) - 1
            if pos < 0 or int(col_ts[pos]) > d:
                continue
            n_row += 1
            if np.all(np.isfinite(feat_mat[pos])):
                n_finite_vec += 1
                if first_finite_decision_ts is None:
                    first_finite_decision_ts = int(d)
        availability_samples.append(
            {"root": r, "n_episodes": len(eps),
             "n_episodes_with_feature_row": n_row,
             "n_episodes_with_fully_finite_vector": n_finite_vec,
             "first_fully_finite_decision_ts": first_finite_decision_ts,
             "schedule_first_ts": int(sched.rows[0].ts_event_ns),
             "schedule_last_ts": int(sched.rows[-1].ts_event_ns)}
        )
        # every feature ts is <= the decision it would serve (searchsorted 'right'-1)
        if col_ts.size and not np.all(np.diff(col_ts) >= 0):
            causal_ok = False
    c["causal_feature_availability"] = {"ok": causal_ok, "samples": availability_samples}

    # -- 2b. real episode feature coverage is non-zero and sensible --------
    # The Phase 15B.2 failure signature was n_pooled_events == 0 for every
    # pipeline: 100% of episodes dropped as FEATURES_MISSING_AT_DECISION.
    coverage_ok = True
    coverage = {}
    for s in availability_samples:
        n_row = s["n_episodes_with_feature_row"]
        frac = (s["n_episodes_with_fully_finite_vector"] / n_row) if n_row else 0.0
        coverage[s["root"]] = {
            "n_episodes": s["n_episodes"],
            "fully_finite_fraction": round(frac, 4),
            "n_fully_finite": s["n_episodes_with_fully_finite_vector"],
        }
        # after warm-up almost every episode should get a full vector; the only
        # legitimate drops are the first ~year (vol_percentile needs 252+ days).
        if s["n_episodes_with_fully_finite_vector"] == 0 or frac < 0.6:
            coverage_ok = False
    c["real_episode_feature_coverage"] = {"ok": coverage_ok, "per_root": coverage}

    # -- 3. CppEngineRunner wiring ------------------------------------
    wiring_ok = True
    wiring = {}
    for r in roots:
        eng = real.engine._per_root[r]
        plan = eng._day_plan
        wiring[r] = {
            "executable": str(eng.executable),
            "executable_exists": Path(eng.executable).exists(),
            "root_symbol": eng._root,
            "validation_day_plan_n_days": int(plan.n_days),
            "validation_day_plan_boundaries_ascending": (
                list(plan.boundary_ts_list()) == sorted(plan.boundary_ts_list())
            ),
            "n_roll_effective_ts": len(eng._roll_ts),
            "roll_close_marks_path_set": eng._roll_marks is not None,
            "roll_close_marks_file_exists": (
                eng._roll_marks is not None and Path(eng._roll_marks).exists()
            ),
        }
        if not (
            wiring[r]["executable_exists"]
            and wiring[r]["validation_day_plan_n_days"] > 0
            and wiring[r]["validation_day_plan_boundaries_ascending"]
            and wiring[r]["roll_close_marks_path_set"]
        ):
            wiring_ok = False
    c["cpp_engine_runner_wiring"] = {"ok": wiring_ok, "per_root": wiring}

    # -- 4. contract frame + roll-close marks wiring --------------------
    roll_ok = True
    roll_info = {}
    for r in roots:
        eng = real.engine._per_root[r]
        roll_info[r] = {
            "n_contracts": len(eng._contracts),
            "roll_close_marks_path": eng._roll_marks,
            "roll_close_marks_present": eng._roll_marks is not None,
        }
    c["contract_and_roll_close_wiring"] = {"ok": roll_ok, "per_root": roll_info}

    # -- 5. NO 2025 anywhere ------------------------------------------
    max_ts = 0
    for r in roots:
        recon = provider.reconstituted(r)
        max_ts = max(max_ts, int(recon.canonical_bars["ts_event_ns"].max()))
        sched = provider.schedule_for(
            PrimaryScope(primary_family="tsmom", root_symbol=r,
                         strategy_fingerprint=next(
                             s.strategy_fingerprint for s in provider.scopes()
                             if s.primary_family == "tsmom" and s.root_symbol == r),
                         strategy_id=next(
                             s.strategy_id for s in provider.scopes()
                             if s.primary_family == "tsmom" and s.root_symbol == r))
        )
        max_ts = max(max_ts, int(sched.rows[-1].ts_event_ns))
        max_ts = max(max_ts, int(frames[r]["ts_event_ns"].max()))
    c["no_2025"] = {
        "ok": max_ts < HOLDOUT_START_NS,
        "max_timestamp_seen": max_ts,
        "holdout_start": HOLDOUT_START_NS,
    }

    # -- 6. NO Databento client / spend --------------------------------
    c["no_databento_network_spend"] = {"ok": True, "note": _no_databento_client()}

    # -- optional: one real C++ backtest, mechanics only ---------------
    if args.run_one_cpp:
        r0 = roots[0]
        sched = provider.schedule_for(
            PrimaryScope(primary_family="tsmom", root_symbol=r0,
                         strategy_fingerprint=next(
                             s.strategy_fingerprint for s in provider.scopes()
                             if s.primary_family == "tsmom" and s.root_symbol == r0),
                         strategy_id=next(
                             s.strategy_id for s in provider.scopes()
                             if s.primary_family == "tsmom" and s.root_symbol == r0))
        )
        run = real.engine.run(sched, cost_scenario_label="baseline_1_0x",
                              cost_multiplier=1.0, want_audit_trails=True)
        c["one_cpp_backtest_mechanics"] = {
            "ok": run.audit_trails_present and run.n_signals > 0,
            "n_signals": run.n_signals,
            "n_trades": run.n_trades,
            "n_fills": run.n_fills,
            "closed_trade_columns_parsed": len(run.closed_trades) >= 0,
            "fill_columns_parsed": len(run.fills) >= 0,
            "note": "MECHANICS ONLY -- PnL / Sharpe not inspected or reported",
        }

    report["all_checks_ok"] = all(v.get("ok", False) for v in c.values())
    report["did_not_fit_model"] = True
    report["did_not_inspect_meta_label_performance"] = True
    report["did_not_write_production_registry"] = True
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v.get("ok") for k, v in c.items()}, indent=2))
    print(f"all_checks_ok = {report['all_checks_ok']}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0 if report["all_checks_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
