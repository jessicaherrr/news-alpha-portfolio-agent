"""Phase 13.5A -- real CME dataset acquisition COST QUOTE (estimate only).

    python scripts/databento_dataset_quote.py                 # full quote, no download
    python scripts/databento_dataset_quote.py --json out.json # + machine-readable dump

Calls ``metadata.get_cost`` / ``get_record_count`` / ``get_billable_size`` only
(free metadata endpoints). There is deliberately NO ``--execute`` and NO code
path that streams a data payload. Never prints the API key.

Design lives in ``python/alpha_agent/data/real_dataset.py`` +
``configs/real_dataset.yaml`` + ``docs/REAL_DATASET_PLAN.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from alpha_agent.data.databento_source import HistoricalRequest, estimate_cost_usd
from alpha_agent.data.real_dataset import (
    DATASET,
    RealDatasetPlan,
    SplitRole,
    load_plan,
)

CONFIG = "configs/real_dataset.yaml"
TRADING_DAYS_PER_YEAR = 253.0
# GLBX.MDP3 ohlcv-1m dbn.zst compressed density and canonical parquet density,
# measured from the existing Phase 03.5 / 04.5 real slices (docs/REAL_DATASET_PLAN.md).
BYTES_PER_ROW_DBN_ZST = 22.0
BYTES_PER_ROW_CANONICAL_PARQUET = 30.0


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _client_metadata():
    try:
        import databento as db
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install -e '.[data]'") from exc
    key = os.getenv("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("DATABENTO_API_KEY not set (local .env). Estimate-only, still needs a key.")
    return db.Historical(key).metadata


def _years(window) -> float:
    import datetime as dt

    a = dt.date.fromisoformat(window.start)
    b = dt.date.fromisoformat(window.end)
    return (b - a).days / 365.25


def _validate_request(req: HistoricalRequest) -> list[str]:
    """Semantic checks BEFORE trusting a numeric quote (13.5A section 10)."""
    errs: list[str] = []
    if req.dataset != DATASET:
        errs.append(f"dataset {req.dataset!r} != {DATASET}")
    if req.schema not in ("ohlcv-1m", "definition"):
        errs.append(f"schema {req.schema!r} not in the 13.5A allow-list")
    if (req.stype_in, req.stype_out) == ("continuous", "raw_symbol"):
        errs.append("continuous->raw_symbol is unsupported (DATABENTO_REALITY)")
    if req.end <= req.start:
        errs.append(f"end {req.end} <= start {req.start}")
    return errs


def quote_plan(md, plan: RealDatasetPlan, *, include_holdout: bool) -> dict:
    out: dict = {
        "plan_id": plan.plan_id,
        "plan_fingerprint": plan.plan_fingerprint(),
        "windows": [
            {"role": w.role.value, "start": w.start, "end": w.end} for w in plan.windows
        ],
        "per_window": {},
        "totals": {},
    }
    grand = {"continuous_front": 0.0, "definitions_range": 0.0, "roll_overlap_est": 0.0}
    total_rows = 0
    total_billable = 0
    for w in plan.windows:
        if w.role == SplitRole.LOCKED_HOLDOUT and not include_holdout:
            continue
        yrs = _years(w)
        wkey = w.role.value
        out["per_window"][wkey] = {"years": round(yrs, 3), "roots": {}}
        for root in plan.roots:
            cont = HistoricalRequest(
                dataset=DATASET, symbols=(f"{root}.v.0",), schema=plan.research_schema,
                start=w.start, end=w.end, stype_in="continuous", stype_out="instrument_id",
            )
            defs = HistoricalRequest(
                dataset=DATASET, symbols=(f"{root}.FUT",), schema="definition",
                start=w.start, end=w.end, stype_in="parent", stype_out="instrument_id",
            )
            verrs = _validate_request(cont) + _validate_request(defs)
            cont_cost = estimate_cost_usd(cont)
            defs_cost = estimate_cost_usd(defs)
            rows = int(md.get_record_count(
                dataset=DATASET, symbols=[f"{root}.v.0"], schema=plan.research_schema,
                stype_in="continuous", start=w.start, end=w.end,
            ))
            billable = int(md.get_billable_size(
                dataset=DATASET, symbols=[f"{root}.v.0"], schema=plan.research_schema,
                stype_in="continuous", start=w.start, end=w.end,
            ))
            # roll-overlap estimate: rolls * (per-window fraction of the annual
            # continuous cost). One outgoing contract per roll, overlap window
            # `roll_overlap_trading_days` (13.5A section 7).
            annual_cont = cont_cost / max(yrs, 1e-9)
            per_window = annual_cont * (plan.roll_overlap_trading_days / TRADING_DAYS_PER_YEAR)
            n_rolls = plan.rolls_per_year[root] * yrs
            overlap_est = per_window * n_rolls

            out["per_window"][wkey]["roots"][root] = {
                "continuous_front_ohlcv1m": round(cont_cost, 6),
                "definitions_parent_range": round(defs_cost, 6),
                "roll_overlap_raw_est": round(overlap_est, 6),
                "n_rolls_est": round(n_rolls, 2),
                "record_count": rows,
                "billable_size_bytes": billable,
                "validation_errors": verrs,
            }
            grand["continuous_front"] += cont_cost
            grand["definitions_range"] += defs_cost
            grand["roll_overlap_est"] += overlap_est
            total_rows += rows
            total_billable += billable

    grand_total_range_defs = sum(grand.values())
    out["totals"] = {
        "continuous_front_ohlcv1m": round(grand["continuous_front"], 4),
        "definitions_parent_range": round(grand["definitions_range"], 4),
        "roll_overlap_raw_est": round(grand["roll_overlap_est"], 4),
        "GRAND_TOTAL_with_range_definitions": round(grand_total_range_defs, 4),
        "record_count_continuous": total_rows,
        "billable_size_bytes_continuous": total_billable,
        "storage_est_dbn_zst_mb": round(total_rows * BYTES_PER_ROW_DBN_ZST / 1e6, 1),
        "storage_est_canonical_parquet_mb": round(
            total_rows * BYTES_PER_ROW_CANONICAL_PARQUET / 1e6, 1
        ),
    }
    return out


def quote_pilot(md, cfg_path: str) -> dict:
    import yaml

    p = yaml.safe_load(Path(cfg_path).read_text())["nq_pilot"]
    cont = HistoricalRequest(
        dataset=DATASET, symbols=("NQ.v.0",), schema=p["schema"],
        start=p["start"], end=p["end"], stype_in="continuous", stype_out="instrument_id",
    )
    defs = HistoricalRequest(
        dataset=DATASET, symbols=("NQ.FUT",), schema="definition",
        start=p["start"], end=p["end"], stype_in="parent", stype_out="instrument_id",
    )
    cont_cost, defs_cost = estimate_cost_usd(cont), estimate_cost_usd(defs)
    rows = int(md.get_record_count(dataset=DATASET, symbols=["NQ.v.0"], schema=p["schema"],
                                  stype_in="continuous", start=p["start"], end=p["end"]))
    # Q1 2024 -> exactly one NQ quarterly roll (March 2024)
    overlap_est = (cont_cost / (89 / TRADING_DAYS_PER_YEAR)) * (5 / TRADING_DAYS_PER_YEAR) * 1
    return {
        "root": "NQ", "start": p["start"], "end": p["end"],
        "continuous_front_ohlcv1m": round(cont_cost, 6),
        "definitions_parent": round(defs_cost, 6),
        "roll_overlap_raw_est_1_roll": round(overlap_est, 6),
        "record_count": rows,
        "total_usd": round(cont_cost + defs_cost + overlap_est, 6),
    }


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Phase 13.5A real-dataset cost quote (estimate only)")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--json", default=None, help="also write the full quote as JSON here")
    ap.add_argument("--include-holdout", action="store_true",
                    help="also quote the 2025 LOCKED_HOLDOUT (cost only -- never downloaded here)")
    args = ap.parse_args()

    md = _client_metadata()
    rng = md.get_dataset_range(dataset=DATASET)
    print(f"== {DATASET} availability ==\n  {rng.get('start')} .. {rng.get('end')}")

    report = {"dataset_range": rng, "plans": {}, "nq_pilot": quote_pilot(md, args.config)}
    for plan_id in ("A", "B"):
        plan = load_plan(args.config, plan_id)
        report["plans"][plan_id] = quote_plan(md, plan, include_holdout=args.include_holdout)

    for plan_id, q in report["plans"].items():
        print(f"\n== PLAN {plan_id} ({q['plan_fingerprint']}) ==")
        for wkey, wd in q["per_window"].items():
            print(f"  [{wkey}] {wd['years']}y")
            for root, rd in wd["roots"].items():
                verr = f"  !! {rd['validation_errors']}" if rd["validation_errors"] else ""
                print(f"    {root}: continuous ${rd['continuous_front_ohlcv1m']:.4f}  "
                      f"defs(range) ${rd['definitions_parent_range']:.4f}  "
                      f"roll-overlap~${rd['roll_overlap_raw_est']:.4f}  "
                      f"rows {rd['record_count']:,}{verr}")
        print(f"  TOTALS: {json.dumps(q['totals'], indent=2)}")

    print(f"\n== NQ PILOT ==\n  {json.dumps(report['nq_pilot'], indent=2)}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {args.json}")
    print("\nESTIMATE ONLY -- no data downloaded. Await approval for Phase 13.5B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
