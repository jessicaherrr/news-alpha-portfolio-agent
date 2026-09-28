"""Phase 6 ETF pilot -- real Databento acquisition driver.

Acquires exactly the 10-request bundle the user approved (2026-09-22): daily
OHLCV + instrument definitions for the finalized 14-ETF universe, on 5
datasets, each over its own real pre-holdout availability window. Real cost
estimate at approval time: $0.605 total, against a $5 hard cap.

This is DELIBERATELY separate from ``scripts/databento_acquire.py`` (the
Futures Plan B driver): ETFs have no continuous symbol, no roll, no roll
overlap -- ``stype_in="raw_symbol"`` directly for both schemas. Reuses only
the generic, dataset-agnostic primitives (``HistoricalRequest``,
``estimate_cost_usd``, ``fetch_and_store_raw``, ``AcquisitionBudget``,
``raw_artifact_dir``, ``verify_manifest``) -- no Futures-specific
canonicalization/roll module is imported. Per CLAUDE.md market-data rules and
Phase 6 instruction 3 ("Never mix with old Futures families"), this pilot's
raw artifacts land in the SAME write-once ``data/raw/databento/`` hash store
(the store is vendor/dataset/schema/request-hash keyed, so an ETF dataset can
never collide with ``GLBX.MDP3``) but its own spend ledger and manifests are
kept in a separate ``phase_6_etf`` namespace.

Every request is cost-estimated (metadata.get_cost) and budget-capped BEFORE
any download; a stored artifact is reused free instead of re-downloaded. The
2025-01-01 locked research holdout is enforced by ``HistoricalRequest``
itself (raises ``HoldoutViolation`` at construction, before any network
call) -- there is no flag here to override it.

    python scripts/phase6_etf_acquire.py                  # estimate only
    python scripts/phase6_etf_acquire.py --acquire         # real download, capped at $5
    python scripts/phase6_etf_acquire.py --replay          # verify stored artifacts, no network
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401  # adds repo python/ to sys.path; must precede alpha_agent
from alpha_agent.data.acquisition import AcquisitionBudget, BudgetExceeded
from alpha_agent.data.databento_source import (
    HistoricalRequest,
    estimate_cost_usd,
    fetch_and_store_raw,
)
from alpha_agent.data.raw_store import raw_artifact_dir, verify_manifest
from alpha_agent.etf.universe import DATASET_WINDOWS as _DATASET_WINDOWS_TUPLE
from alpha_agent.etf.universe import PILOT_UNIVERSE

REPO = Path(__file__).resolve().parents[1]
RAW_ROOT = "data/raw"
MANIFEST_ROOT = Path("data/manifests/phase_6_etf")
LEDGER = MANIFEST_ROOT / "spend_ledger.json"

# Canonical source: alpha_agent.etf.universe (also used by the corporate-action
# and data-source-provenance modules, so the acquired artifacts and the typed
# ETF package always agree on universe/windows). Symbol order doesn't affect
# the raw-artifact hash (raw_store._request_hash sorts symbols), so importing
# the canonical tuple here is guaranteed to resolve to the same already-
# downloaded artifacts.
UNIVERSE = list(PILOT_UNIVERSE)
DATASET_WINDOWS = list(_DATASET_WINDOWS_TUPLE)

SCHEMAS = ["ohlcv-1d", "definition"]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO / ".env")


def _requests() -> list[HistoricalRequest]:
    reqs = []
    for dataset, start, end in DATASET_WINDOWS:
        for schema in SCHEMAS:
            reqs.append(
                HistoricalRequest(
                    dataset=dataset, symbols=tuple(UNIVERSE), schema=schema,
                    start=start, end=end, stype_in="raw_symbol", stype_out="instrument_id",
                )
            )
    return reqs


def _artifact_exists(req: HistoricalRequest) -> Path | None:
    d = raw_artifact_dir(
        vendor="databento", dataset=req.dataset, schema=req.schema,
        stype_in=req.stype_in, stype_out=req.stype_out, symbols=list(req.symbols),
        start=req.start, end=req.end, root_dir=RAW_ROOT,
    )
    for ext in ("dbn.zst", "dbn"):
        p = d / f"{req.schema}.{ext}"
        if p.exists():
            return p
    return None


def _ledger_load() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"schema": "phase6-etf-spend-ledger/1", "downloads": {}}


def _ledger_total(led: dict) -> float:
    return round(sum(float(v["cost_usd"]) for v in led.get("downloads", {}).values()), 6)


def _acquire(req: HistoricalRequest, budget: AcquisitionBudget, *, acquire: bool) -> dict:
    label = f"{req.dataset}/{req.schema}"
    existing = _artifact_exists(req)
    if existing is not None:
        art = verify_manifest(existing)
        cost = float(art.manifest.extra.get("estimated_cost_usd", 0) or 0)
        print(f"  [reuse] {label}: {existing}  (${cost:.5f}, already paid)")
        return {"label": label, "status": "reuse", "path": str(art.path), "cost_usd": cost,
                "sha256": art.manifest.sha256, "row_count": art.manifest.row_count}

    cost = estimate_cost_usd(req)
    print(f"  [quote] {label}: ${cost:.5f}  ({len(req.symbols)} symbols, {req.start}..{req.end})")
    if not acquire:
        return {"label": label, "status": "estimate_only", "cost_usd": cost}

    budget.charge(label, cost)  # raises BudgetExceeded before any download if this would exceed the cap
    art = fetch_and_store_raw(req, max_cost_usd=budget.remaining() + cost, root_dir=RAW_ROOT)
    led = _ledger_load()
    led["downloads"][Path(art.path).parent.name] = {
        "cost_usd": cost, "dataset": req.dataset, "schema": req.schema, "symbols": list(req.symbols),
        "start": req.start, "end": req.end, "sha256": art.manifest.sha256,
        "row_count": art.manifest.row_count, "label": label,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    led["cumulative_spent_usd"] = _ledger_total(led)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=2))
    print(f"  [dl]    {label}: {art.path}  (cumulative ${led['cumulative_spent_usd']:.4f})")
    return {"label": label, "status": "downloaded", "path": str(art.path), "cost_usd": cost,
            "sha256": art.manifest.sha256, "row_count": art.manifest.row_count}


def run(*, acquire: bool, cap_usd: float) -> int:
    _load_dotenv()
    led = _ledger_load()
    budget = AcquisitionBudget(cap_usd=cap_usd)
    budget.spent_usd = _ledger_total(led)
    print(f"== Phase 6 ETF acquisition  ({'ACQUIRE' if acquire else 'ESTIMATE ONLY'})  "
          f"cap ${cap_usd:.2f}, already spent ${budget.spent_usd:.4f} ==")

    results = []
    try:
        for req in _requests():
            results.append(_acquire(req, budget, acquire=acquire))
    except BudgetExceeded as exc:
        print(f"\nREFUSING: {exc}")
        return 2

    total = sum(r["cost_usd"] for r in results)
    print(f"\n== total across {len(results)} requests: ${total:.5f}  "
          f"(cap ${cap_usd:.2f}) ==")
    if not acquire:
        print("ESTIMATE ONLY. Re-run with --acquire to download.")
    return 0


def run_replay() -> int:
    print("== REPLAY -- verify every Phase 6 ETF raw manifest (no network, no cost) ==")
    n = 0
    for req in _requests():
        p = _artifact_exists(req)
        if p is None:
            print(f"  [missing] {req.dataset}/{req.schema}")
            continue
        art = verify_manifest(p)
        print(f"  [ok] {req.dataset}/{req.schema}: {art.manifest.sha256[:16]}...  "
              f"rows={art.manifest.row_count}")
        n += 1
    print(f"  {n}/{len(_requests())} artifact(s) verified OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6 ETF pilot real Databento acquisition")
    ap.add_argument("--acquire", action="store_true", help="perform downloads (else estimate only)")
    ap.add_argument("--cap-usd", type=float, default=5.00)
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args()
    if args.replay:
        return run_replay()
    return run(acquire=args.acquire, cap_usd=args.cap_usd)


if __name__ == "__main__":
    sys.exit(main())
