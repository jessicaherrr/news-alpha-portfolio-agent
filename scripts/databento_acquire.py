"""Phase 13.5B -- real CME acquisition driver (Plan B, 2018-2024, NO holdout).

    python scripts/databento_acquire.py --pilot                     # estimate only
    python scripts/databento_acquire.py --pilot   --acquire --cap-usd 65
    python scripts/databento_acquire.py --plan-b  --acquire --cap-usd 65   # only after pilot passes
    python scripts/databento_acquire.py --qa                        # re-run QA on acquired data
    python scripts/databento_acquire.py --replay                    # reproduce from stored bytes, no net

Every request is cost-estimated (metadata.get_cost) and budget-capped BEFORE any
download. Raw DBN bytes are write-once; a re-run reuses the stored artifact
(verify_manifest) instead of paying twice. The 2025 LOCKED_HOLDOUT is refused by
``acquisition.guard_no_holdout`` -- there is no flag to override it here.
Never prints the API key.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from alpha_agent.adapters.cpp_cli import canonical_bars_to_boundary, write_boundary_bundle
from alpha_agent.data import decode as dec
from alpha_agent.data.acquisition import (
    AcquisitionBudget,
    continuous_request,
    definition_snapshot_requests,
    detect_all_transitions,
    roll_overlap_requests,
)
from alpha_agent.data.calendars import default_calendar
from alpha_agent.data.canonicalize import NormalizationPolicy, canonicalize
from alpha_agent.data.databento_source import (
    HistoricalRequest,
    estimate_cost_usd,
    fetch_and_store_raw,
)
from alpha_agent.data.definitions import DefinitionRegistry, contracts_frame, parse_definition_frame
from alpha_agent.data.futures_history import build_futures_history
from alpha_agent.data.lineage import code_commit
from alpha_agent.data.price_domain import descale_fixed_point, median_abs_price
from alpha_agent.data.raw_store import raw_artifact_dir, verify_manifest
from alpha_agent.data.real_dataset import (
    ComponentKind,
    DateWindow,
    RealDatasetComponent,
    RealDatasetManifest,
    SplitRole,
    load_plan,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = "configs/real_dataset.yaml"
RAW_ROOT = "data/raw"
PROCESSED_ROOT = "data/processed"
MANIFEST_ROOT = Path("data/manifests/real_dataset")
LEDGER = MANIFEST_ROOT / "spend_ledger.json"
PILOT_MARKER = MANIFEST_ROOT / "pilot_passed.json"
INGESTION_VERSION = "databento-source/1"
CANON_VERSION = "1.0"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO / ".env")


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
    return {"schema": "spend-ledger/1", "downloads": {}}


def _ledger_total(led: dict | None = None) -> float:
    led = led or _ledger_load()
    return round(sum(float(v["cost_usd"]) for v in led.get("downloads", {}).values()), 6)


def _ledger_backfill() -> dict:
    """Record every raw databento artifact already on disk (permanent spend
    record). Idempotent -- keyed by request hash directory."""
    led = _ledger_load()
    for mp in Path(RAW_ROOT).rglob("*.manifest.json"):
        try:
            m = json.loads(mp.read_text())
        except (json.JSONDecodeError, OSError):  # a malformed sidecar is skipped, not fatal
            continue
        key = mp.parent.name
        if key in led["downloads"]:
            continue
        led["downloads"][key] = {
            "cost_usd": float(m.get("extra", {}).get("estimated_cost_usd", 0.0) or 0.0),
            "schema": m.get("schema"), "symbols": m.get("symbols"),
            "start": m.get("start"), "end": m.get("end"), "sha256": m.get("sha256"),
            "recorded_at": datetime.now(UTC).isoformat(),
        }
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    led["cumulative_spent_usd"] = _ledger_total(led)
    LEDGER.write_text(json.dumps(led, indent=2))
    return led


class _HardTimeout(RuntimeError):
    """A single Databento call exceeded its wall-clock deadline."""


def _retry(fn, what: str, *, attempts: int = 8, deadline_s: int = 240):
    """Run ``fn`` with exponential backoff on transient Databento HTTP errors
    (Read timed out / connection reset / 5xx). Used for both ``metadata.get_cost``
    and the streaming download -- the download streams to a tempdir and is only
    committed to the write-once store on success, so a retry never double-writes.

    Each attempt also gets a hard ``SIGALRM`` wall-clock deadline: a degraded
    Databento stream can trickle bytes slowly enough to keep the socket alive
    (defeating the requests read-timeout) for hours -- observed 2026-09-07. The
    alarm converts that into a normal retry.
    """
    import signal
    import time

    def _on_alarm(_signum, _frame):
        raise _HardTimeout(f"{what}: exceeded {deadline_s}s wall-clock deadline")

    for i in range(1, attempts + 1):
        prev = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(deadline_s)
        try:
            return fn()
        except Exception as exc:  # narrow by message; databento wraps transient errors as BentoError
            msg = str(exc).lower()
            transient = isinstance(exc, _HardTimeout) or any(
                s in msg for s in ("timed out", "timeout", "connection", "reset",
                                   "streaming response", "temporarily", "502", "503",
                                   "504", "gateway", "deadline")
            )
            if not transient or i == attempts:
                raise
            wait = min(90, 5 * 2 ** (i - 1))
            print(f"  [retry] {what} attempt {i}/{attempts} failed ({exc}); sleeping {wait}s")
            time.sleep(wait)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev)
    raise RuntimeError("unreachable")


def _acquire(req: HistoricalRequest, budget: AcquisitionBudget, label: str, *, acquire: bool):
    """Idempotent. A stored artifact is reused free (already in the ledger seed);
    a new download is cost-gated, budget-gated, and appended to the ledger."""
    existing = _artifact_exists(req)
    if existing is not None:
        art = verify_manifest(existing)
        print(f"  [reuse] {label}: {existing}  "
              f"(${float(art.manifest.extra.get('estimated_cost_usd', 0) or 0):.5f}, already paid)")
        return art
    cost = _retry(lambda: estimate_cost_usd(req), f"get_cost {label}")
    print(f"  [quote] {label}: ${cost:.5f}  ({list(req.symbols)} {req.schema} {req.start}..{req.end})")
    if not acquire:
        return None
    budget.charge(label, cost)          # refuses if it would exceed the cap
    art = _retry(
        lambda: fetch_and_store_raw(req, max_cost_usd=budget.remaining() + cost, root_dir=RAW_ROOT),
        f"download {label}",
    )
    led = _ledger_load()
    led["downloads"][Path(art.path).parent.name] = {
        "cost_usd": cost, "schema": req.schema, "symbols": list(req.symbols),
        "start": req.start, "end": req.end, "sha256": art.manifest.sha256,
        "label": label, "recorded_at": datetime.now(UTC).isoformat(),
    }
    led["cumulative_spent_usd"] = _ledger_total(led)
    LEDGER.write_text(json.dumps(led, indent=2))
    print(f"  [dl]    {label}: {art.path}  (cumulative ${led['cumulative_spent_usd']:.4f})")
    return art


def _ohlcv_vendor(path: Path) -> pd.DataFrame:
    df = dec.load_dbn(path).to_df(price_type="fixed", pretty_ts=False, map_symbols=True).reset_index()
    return dec.ohlcv_df_to_vendor_frame(df)


def _def_vendor(paths: list[Path], keep_ids: set[int]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = dec.load_dbn(p).to_df(pretty_ts=False, map_symbols=False).reset_index()
        frames.append(dec.definition_df_to_vendor_frame(df, keep_instrument_ids=keep_ids))
    merged = pd.concat(frames, ignore_index=True).drop_duplicates("instrument_id", keep="last")
    return merged.reset_index(drop=True)


def _write_manifest(
    plan_id: str, kind: ComponentKind, root: str, role: SplitRole,
    req: HistoricalRequest, art, *, price_domain: str, role_label: str,
    def_identity: str, cost: float | None,
) -> Path:
    comp = RealDatasetComponent(
        kind=kind, root_symbol=root, symbols=tuple(req.symbols) or (),
        schema=req.schema, stype_in=req.stype_in, stype_out=req.stype_out,
        window=DateWindow(role=role, start=req.start, end=req.end),
        resolved_from_transitions=(kind == ComponentKind.ROLL_OVERLAP_RAW),
    )
    m = RealDatasetManifest(
        databento_schema=req.schema, component_kind=kind, root_symbol=root, split_role=role,
        symbol_request=tuple(req.symbols), stype_in=req.stype_in, stype_out=req.stype_out,
        start=req.start, end=req.end, query_identity=comp.query_identity(),
        price_domain=price_domain, raw_or_continuous_role=role_label,
        raw_artifact_relpath=(
            str(Path(art.path)) if art and not Path(art.path).is_absolute()
            else (str(Path(art.path).relative_to(REPO)) if art else "(not downloaded)")
        ),
        raw_sha256=art.manifest.sha256 if art else "",
        row_count=art.manifest.row_count if art else None,
        calendar_identity=default_calendar().version,
        ingestion_version=INGESTION_VERSION, canonicalization_version=CANON_VERSION,
        contract_definition_identity=def_identity,
        code_commit=code_commit(), download_cost_usd=cost,
    )
    outdir = MANIFEST_ROOT / plan_id
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{root}__{kind.value}__{req.start}_{req.end}.json"
    out.write_text(m.to_json())
    return out


def acquire_root(
    root: str, start: str, end: str, *, plan_id: str, role: SplitRole,
    budget: AcquisitionBudget, acquire: bool,
) -> dict:
    print(f"\n=== {root}  {start}..{end}  [{role.value}] ===")
    calendar = default_calendar()
    cont_sym = f"{root}.v.0"

    # 1. continuous front
    cont_req = continuous_request(root, start, end)
    cont_art = _acquire(cont_req, budget, f"{root} continuous", acquire=acquire)
    if cont_art is None:
        return {"root": root, "estimated_only": True}
    vendor = _ohlcv_vendor(Path(cont_art.path))
    seen_ids = {int(i) for i in vendor["instrument_id"].unique()}

    # 2. definition snapshots (monthly)
    def_reqs = definition_snapshot_requests(root, start, end)
    def_arts = []
    for i, dr in enumerate(def_reqs):
        a = _acquire(dr, budget, f"{root} defs {dr.start}", acquire=acquire)
        if a is not None:
            def_arts.append(Path(a.path))
    if not def_arts:
        raise SystemExit(f"{root}: no definition snapshots acquired")
    def_vendor = _def_vendor(def_arts, seen_ids)
    # CBOT Treasuries (ZN/ZB/ZF/ZT) quote in 32nds -> the convention-independent
    # fractional economics derivation (Phase 04.5). Decimal roots are unaffected.
    allow_fractional = root in {"ZN", "ZB", "ZF", "ZT", "UB", "TN"}
    specs = parse_definition_frame(def_vendor, allow_fractional=allow_fractional)
    registry = DefinitionRegistry(specs)
    import hashlib
    def_identity = "defs:" + hashlib.sha256(
        json.dumps(sorted(s.raw_symbol for s in specs)).encode()).hexdigest()[:16]

    unresolved = seen_ids - {s.instrument_id for s in specs}
    if unresolved:
        raise SystemExit(f"{root}: bars reference instrument_id(s) with no definition: {sorted(unresolved)}")

    # 3. roll overlaps -- resolved ONLY from observed transitions
    transitions = detect_all_transitions(vendor, registry, continuous_symbol=cont_sym)
    print(f"  {len(transitions)} roll transition(s) observed")
    overlap_reqs = roll_overlap_requests(transitions, window_start=start, window_end=end)
    overlap_frames = []
    for orq in overlap_reqs:
        a = _acquire(orq, budget, f"{root} overlap {orq.symbols[0]}", acquire=acquire)
        if a is not None:
            of = descale_fixed_point(
                _ohlcv_vendor(Path(a.path)).rename(columns={"ts_event": "ts_event_ns"})
            )
            overlap_frames.append(of)
    overlap = pd.concat(overlap_frames, ignore_index=True) if overlap_frames else None

    # 4. canonicalize (layer 2). canonicalize sorts by (instrument_id, ts) which
    #    is not global time order over many years (instrument_id order != listing
    #    order); build_roll_events / the continuous feed need GLOBAL ts order.
    canonical, report = canonicalize(vendor, registry, calendar, policy=NormalizationPolicy())
    canonical = canonical.sort_values("ts_event_ns", kind="stable").reset_index(drop=True)
    proc = Path(PROCESSED_ROOT)
    (proc / "contracts").mkdir(parents=True, exist_ok=True)
    contracts_frame(specs).to_parquet(proc / "contracts" / f"{root}.parquet", index=False)
    for raw_symbol, grp in canonical.groupby("raw_symbol"):
        d = proc / "bars" / root
        d.mkdir(parents=True, exist_ok=True)
        grp.reset_index(drop=True).to_parquet(d / f"{raw_symbol}.parquet", index=False)

    # 5. layers 3/4/6
    fh = build_futures_history(
        canonical, continuous_symbol=cont_sym, registry=registry, calendar=calendar,
        overlap_bars=overlap, processed_root=PROCESSED_ROOT, write=True,
    )

    # manifests
    _write_manifest(plan_id, ComponentKind.CONTINUOUS_FRONT, root, role, cont_req, cont_art,
                    price_domain="raw_contract", role_label="continuous_front",
                    def_identity=def_identity,
                    cost=cont_art.manifest.extra.get("estimated_cost_usd"))
    for dr, dp in zip(def_reqs, def_arts):
        da = verify_manifest(dp)
        _write_manifest(plan_id, ComponentKind.DEFINITIONS, root, role, dr, da,
                        price_domain="n/a", role_label="definitions", def_identity=def_identity,
                        cost=da.manifest.extra.get("estimated_cost_usd"))
    for orq in overlap_reqs:
        p = _artifact_exists(orq)
        if p:
            oa = verify_manifest(p)
            _write_manifest(plan_id, ComponentKind.ROLL_OVERLAP_RAW, root, role, orq, oa,
                            price_domain="raw_contract", role_label="roll_overlap_raw",
                            def_identity=def_identity,
                            cost=oa.manifest.extra.get("estimated_cost_usd"))

    return {
        "root": root, "canonical_rows": len(canonical), "rejected": report.n_rejected_rows,
        "n_errors": len(report.errors), "n_transitions": len(transitions),
        "n_overlaps": len(overlap_reqs), "n_rolls_fallback": sum(1 for r in fh.rolls if r.used_fallback),
        "diagnostics": report.summary(),
        "roll_diagnostics": fh.diagnostics["rolls"].summary(),
        "canonical": canonical, "registry": registry, "continuous": fh.continuous,
        "report": report, "fh": fh, "seen_ids": sorted(seen_ids),
    }


def qa_gate(res: dict) -> tuple[bool, list[str]]:
    """Complete QA + lineage gate (docs/REAL_DATASET_PLAN.md section 11)."""
    root = res["root"]
    calendar = default_calendar()
    canonical: pd.DataFrame = res["canonical"]
    lines: list[str] = []
    ok = True

    def chk(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        lines.append(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
        ok = ok and cond

    chk("no canonical diagnostic errors", res["n_errors"] == 0,
        f"{res['n_errors']} errors: {[d.kind.value for d in res['report'].errors][:5]}")
    chk("timestamps strictly increasing per contract",
        bool(canonical.sort_values("ts_event_ns").groupby("raw_symbol")["ts_event_ns"]
              .apply(lambda s: s.is_monotonic_increasing).all()))
    chk("no duplicate (instrument_id, ts)",
        not canonical.duplicated(["instrument_id", "ts_event_ns"]).any())
    chk("every bar resolves to a contract in the registry",
        {int(i) for i in canonical["instrument_id"].unique()}
        <= {s.instrument_id for s in res["registry"].specs()})
    chk("OHLC invariants (low <= min(o,c) <= max(o,c) <= high)",
        bool(((canonical["low"] <= canonical[["open", "close"]].min(axis=1)) &
              (canonical[["open", "close"]].max(axis=1) <= canonical["high"])).all()))
    # session / trading_day / DST: one CME trading_day may span a UTC midnight
    tdays = set(canonical["trading_day"].astype(str))
    utc_dates = {pd.Timestamp(int(t), unit="ns", tz="UTC").date().isoformat()
                 for t in canonical["ts_event_ns"]}
    chk("session labels present", canonical["session"].astype(str).ne("").all())
    chk("trading_day labels present", len(tdays) >= 1)
    lines.append(f"  [info] {len(tdays)} trading_day(s) across {len(utc_dates)} UTC date(s)")
    # roll boundary detection
    chk("roll transitions detected match instrument changes",
        res["n_transitions"] == max(0, canonical["instrument_id"]
                                    .ne(canonical["instrument_id"].shift()).sum() - 1))
    chk("no roll used a fallback (contemporaneous overlap present)",
        res["n_rolls_fallback"] == 0, f"{res['n_rolls_fallback']} fallback rolls")
    # price-scale sanity
    med = median_abs_price(canonical, "close")
    chk("plausible normalized price scale", 0.01 < med < 1e6, f"median|close|={med:.3f}")
    # lineage hash stability
    lin_ok = True
    for p in res["fh"].paths.values():
        lin_ok = lin_ok and Path(str(p) + ".lineage.json").exists()
    chk("layer 3/4/6 lineage sidecars written", lin_ok)

    # every canonical bar's instrument_id was live at that bar's timestamp
    specs = {s.instrument_id: s for s in res["registry"].specs()}
    from alpha_agent.data.contract_lifecycle import tradable_until_ns as _tuntil

    live_ok = True
    for iid, g in canonical.groupby("instrument_id"):
        s = specs.get(int(iid))
        lo, hi = int(s.activation_ns), int(_tuntil(s))
        if not ((g["ts_event_ns"].astype("int64") >= lo)
                & (g["ts_event_ns"].astype("int64") <= hi)).all():
            live_ok = False
    chk("every bar's contract was tradable at that timestamp", live_ok)

    # C++ boundary smoke: replay the canonical bars through the frozen engine with
    # an EMPTY target schedule (no trades) -- verifies contract resolution + bar
    # replay, without a strategy hitting roll edge cases.
    exe = REPO / "build/cpp/cpp/quant_backtest_targets_csv"
    if exe.exists():
        stg = Path("data/staging/boundary_qa") / root
        bcsv, ccsv = write_boundary_bundle(
            canonical_bars_to_boundary(canonical), contracts_frame(res["registry"].specs()), stg
        )
        tcsv = Path(stg) / "targets.csv"
        tcsv.write_text("ts_event_ns,root_symbol,target_units,strategy_fingerprint,matched_rule_id\n")
        out = subprocess.run([str(exe), str(bcsv), str(ccsv), str(tcsv), "no_decision"],
                             capture_output=True, text=True, check=False)
        try:
            cli = json.loads(out.stdout)
            chk("C++ replays every canonical bar (empty schedule, 0 fills)",
                cli.get("bars", 0) == len(canonical) and cli.get("fills", -1) == 0
                and cli.get("contracts_resolved", 0) == canonical["instrument_id"].nunique(),
                f"bars {cli.get('bars')}/{len(canonical)}, resolved {cli.get('contracts_resolved')}, "
                f"fills {cli.get('fills')}")
        except json.JSONDecodeError:
            chk("C++ boundary replay", False, out.stderr.strip()[:150])
    else:
        lines.append("  [skip] C++ core not built -- boundary replay skipped")
    del calendar
    return ok, lines


def _seed_budget(cap_usd: float) -> AcquisitionBudget:
    led = _ledger_backfill()
    b = AcquisitionBudget(cap_usd=cap_usd)
    b.spent_usd = _ledger_total(led)
    print(f"== cumulative spend so far ${b.spent_usd:.4f} / cap ${cap_usd:.2f} "
          f"(${b.remaining():.4f} left) ==")
    return b


def run_pilot(*, acquire: bool, cap_usd: float) -> int:
    import yaml
    cfg = yaml.safe_load(Path(CONFIG).read_text())["nq_pilot"]
    budget = _seed_budget(cap_usd)
    print(f"== PILOT: NQ {cfg['start']}..{cfg['end']}  ({'ACQUIRE' if acquire else 'ESTIMATE ONLY'}) ==")
    res = acquire_root("NQ", cfg["start"], cfg["end"], plan_id="B", role=SplitRole.VALIDATION,
                       budget=budget, acquire=acquire)
    if res.get("estimated_only"):
        print("\nESTIMATE ONLY. Re-run with --acquire to download.")
        return 0
    ok, lines = qa_gate(res)
    print("\n== PILOT QA + LINEAGE GATE ==")
    print("\n".join(lines))
    print(f"\ncumulative spend: ${_ledger_total():.4f} / cap ${cap_usd:.2f}")
    if ok:
        PILOT_MARKER.write_text(json.dumps({
            "at": datetime.now(UTC).isoformat(), "root": "NQ",
            "window": [cfg["start"], cfg["end"]], "code_commit": code_commit(),
            "canonical_rows": res["canonical_rows"], "n_transitions": res["n_transitions"],
        }, indent=2))
        print(f"\nPILOT PASSED. marker -> {PILOT_MARKER}")
        print("Proceed: python scripts/databento_acquire.py --plan-b --acquire --cap-usd 65")
        return 0
    print("\nPILOT FAILED QA -- not proceeding to Plan B. Review diagnostics above.")
    return 1


def run_plan_b(*, acquire: bool, cap_usd: float) -> int:
    if acquire and not PILOT_MARKER.exists():
        print("Refusing: the NQ pilot must pass QA first (no data/manifests/real_dataset/"
              "pilot_passed.json). Run --pilot --acquire.")
        return 2
    plan = load_plan(CONFIG, "B")
    budget = _seed_budget(cap_usd)

    windows = [w for w in plan.windows if w.role != SplitRole.LOCKED_HOLDOUT]
    all_ok = True
    for w in windows:
        for root in plan.roots:
            res = acquire_root(root, w.start, w.end, plan_id="B", role=w.role,
                               budget=budget, acquire=acquire)
            if res.get("estimated_only"):
                continue
            ok, lines = qa_gate(res)
            print(f"\n-- QA {root} [{w.role.value}] --\n" + "\n".join(lines))
            all_ok = all_ok and ok
    print(f"\n== PLAN B {'ACQUIRE' if acquire else 'ESTIMATE'} "
          f"{'OK' if all_ok else 'HAS QA FAILURES'}  "
          f"cumulative ${_ledger_total():.4f} / cap ${cap_usd:.2f} ==")
    return 0 if all_ok else 1


def run_replay() -> int:
    print("== REPLAY -- verify every stored raw manifest (no network, no cost) ==")
    n = 0
    for mp in Path(RAW_ROOT).rglob("*.manifest.json"):
        verify_manifest(mp)
        n += 1
    print(f"  {n} raw artifact(s) verified OK")
    for mp in MANIFEST_ROOT.rglob("*.json"):
        if mp.name in ("spend_ledger.json", "pilot_passed.json"):
            continue
        m = RealDatasetManifest.model_validate_json(mp.read_text())
        print(f"  {mp.name}: {m.semantic_identity()[:32]}...")
    return 0


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Phase 13.5B real CME acquisition (Plan B, no holdout)")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--plan-b", action="store_true")
    ap.add_argument("--qa", action="store_true", help="replay + re-run QA on acquired data")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--acquire", action="store_true", help="perform downloads (else estimate only)")
    ap.add_argument("--cap-usd", type=float, default=65.00)
    args = ap.parse_args()

    if args.replay or args.qa:
        return run_replay()
    if args.pilot:
        return run_pilot(acquire=args.acquire, cap_usd=args.cap_usd)
    if args.plan_b:
        return run_plan_b(acquire=args.acquire, cap_usd=args.cap_usd)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
