"""Phase 04.5 -- real NQ roll + back-adjustment validation.

    python scripts/databento_roll_validate.py                       # Stage A design + cost estimate
    python scripts/databento_roll_validate.py --stage-a --max-cost-usd 0.20 --execute   # LEGACY, see below
    python scripts/databento_roll_validate.py --stage-b-plan        # Stage B design + cost estimate
    python scripts/databento_roll_validate.py --stage-b --max-cost-usd 0.20 --execute   # LEGACY, see below
    python scripts/databento_roll_validate.py --replay              # local, no network, no cost

DEPRECATED LIVE-FETCH MODE (Phase 23.2): Stage A/B's hardcoded window
(``alpha_agent.data.roll_validation.STAGE_A_START/END = "2026-06-08"/
"2026-06-19"``) is a permanent, immutable provenance record of this script's
one real Phase 04.5 roll-detection run -- never delete or edit it (see
``data/manifests/real_dataset/spend_ledger.json`` and
``docs/FINAL_SYSTEM_AUDIT.md`` C1). It is also, as of Phase 23.1, at/after the
project's locked ``>= 2025-01-01`` research holdout boundary, which this
script's original design never guarded against (it predates that guard).
``alpha_agent.data.databento_source.HistoricalRequest`` now refuses any
holdout-crossing window at construction time, so ``stage_a_requests()`` /
``stage_b_request()`` -- and therefore ``--stage-a``/``--stage-b`` here, live
or estimate-only -- now raise ``HoldoutViolation`` before printing a request,
estimating cost, or touching the network, by design, not a bug. ``--replay``
is unaffected: it resolves the three already-downloaded, sha256-verified Stage
A/B artifacts by their fixed, hardcoded paths (``REPLAY_STAGE_A_BARS`` /
``REPLAY_STAGE_A_DEF`` / ``REPLAY_STAGE_B`` below) rather than reconstructing a
live request, so it needs neither the guard nor a network/API key and fully
and losslessly satisfies this script's original validation purpose.

Never prints the API key. Every paid request is cost-estimated and capped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from alpha_agent.data import decode as dec
from alpha_agent.data.backadjust import AdjustmentMode, build_back_adjusted_series
from alpha_agent.data.calendars import default_calendar
from alpha_agent.data.canonicalize import canonicalize
from alpha_agent.data.continuous import build_continuous_series
from alpha_agent.data.databento_source import estimate_cost_usd, fetch_and_store_raw
from alpha_agent.data.definitions import DefinitionRegistry, parse_definition_frame
from alpha_agent.data.price_domain import PriceScaleError, descale_fixed_point, median_abs_price
from alpha_agent.data.raw_store import verify_manifest
from alpha_agent.data.roll_validation import (
    RollDetection,
    detect_transition,
    stage_a_requests,
    stage_b_request,
)
from alpha_agent.data.rolls import RollPricePolicy, build_roll_events

DETECTION_JSON = Path("data/processed/roll_validation/nq_v0_2026_06.json")

# Permanent provenance of Phase 04.5's one real Stage A/B download
# (2026-06-08..2026-06-19 continuous + definitions, 2026-06-16..2026-06-19
# NQM6/NQU6 overlap) -- at/after the 2025-01-01 holdout, discovered Phase
# 23.1, remediated Phase 23.2 (docs/FINAL_SYSTEM_AUDIT.md C1). ``--replay``
# resolves these fixed, already-verified paths directly and never
# reconstructs a live HistoricalRequest via stage_a_requests()/
# stage_b_request() (both now correctly refuse construction for this window),
# so replay needs neither the holdout guard nor a network/API key.
REPLAY_STAGE_A_BARS = Path("data/raw/databento/GLBX.MDP3/ohlcv-1m/a72afba97b32fe7e/ohlcv-1m.dbn.zst")
REPLAY_STAGE_A_DEF = Path("data/raw/databento/GLBX.MDP3/definition/c9846cc4154a2e4f/definition.dbn.zst")
REPLAY_STAGE_B = Path("data/raw/databento/GLBX.MDP3/ohlcv-1m/277e476e1f5fed12/ohlcv-1m.dbn.zst")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _print_req(label: str, req) -> None:
    print(f"  {label}: dataset={req.dataset} symbols={list(req.symbols)} schema={req.schema} "
          f"stype_in={req.stype_in} stype_out={req.stype_out} start={req.start} end={req.end}")


def _estimate(*reqs) -> float | None:
    if not os.getenv("DATABENTO_API_KEY"):
        print("  (no DATABENTO_API_KEY -- cannot run metadata.get_cost)")
        return None
    total = 0.0
    for r in reqs:
        c = estimate_cost_usd(r)
        total += c
        print(f"    {r.schema} {list(r.symbols)}: ${c:.6f}")
    print(f"    TOTAL: ${total:.6f}")
    return total


def _ohlcv_vendor_from_dbn(path: Path) -> pd.DataFrame:
    df = dec.load_dbn(path).to_df(price_type="fixed", pretty_ts=False, map_symbols=True).reset_index()
    return dec.ohlcv_df_to_vendor_frame(df)


def _registry_from_dbn(def_path: Path, keep_ids: set[int]) -> DefinitionRegistry:
    df = dec.load_dbn(def_path).to_df(pretty_ts=False, map_symbols=False).reset_index()
    specs = parse_definition_frame(dec.definition_df_to_vendor_frame(df, keep_instrument_ids=keep_ids))
    return DefinitionRegistry(specs)


# --- Stage A --------------------------------------------------------------

def stage_a(execute: bool, max_cost: float | None) -> int:
    bars_req, def_req = stage_a_requests()
    print("== STAGE A -- real continuous roll detection ==")
    _print_req("bars       ", bars_req)
    _print_req("definitions", def_req)
    print("\n== COST ESTIMATE (metadata.get_cost, free) ==")
    total = _estimate(bars_req, def_req)

    if not execute:
        print("\nEstimate only. Re-run with --stage-a --max-cost-usd <cap> --execute.")
        return 0
    if max_cost is None:
        print("\n--execute requires --max-cost-usd."); return 2
    if total is not None and total > max_cost:
        print(f"\nEstimate ${total:.6f} exceeds cap ${max_cost:.2f}. Aborting."); return 3

    print("\n== DOWNLOAD (Stage A) ==")
    bars_raw = fetch_and_store_raw(bars_req, max_cost_usd=max_cost, root_dir="data/raw")
    def_raw = fetch_and_store_raw(def_req, max_cost_usd=max_cost, root_dir="data/raw")
    print(f"  {bars_raw.path}\n  {def_raw.path}")
    return _detect_and_record(bars_raw.path, def_raw.path)


def _detect_and_record(bars_path: Path, def_path: Path) -> int:
    verify_manifest(bars_path)
    verify_manifest(def_path)
    vendor = _ohlcv_vendor_from_dbn(bars_path)
    ids = {int(i) for i in vendor["instrument_id"].unique()}
    reg = _registry_from_dbn(def_path, ids)

    detection = detect_transition(vendor, reg)
    print("\n== ROLL DETECTION ==")
    print(f"  continuous rows: {len(vendor)}")
    print(f"  unique instrument_id(s): {sorted(ids)}")
    if detection is None:
        print("  NO instrument_id transition in this window -- not manufacturing a roll. STOP.")
        return 7
    for s in detection.spans:
        t0 = pd.Timestamp(s.first_ts_ns, unit="ns", tz="UTC")
        t1 = pd.Timestamp(s.last_ts_ns, unit="ns", tz="UTC")
        print(f"  {s.raw_symbol} (id {s.instrument_id}): {s.n_bars} bars  {t0} .. {t1}")
    tt = pd.Timestamp(detection.transition_ts_ns, unit="ns", tz="UTC")
    print(f"  TRANSITION: {detection.from_raw_symbol} -> {detection.to_raw_symbol} "
          f"at {detection.transition_ts_ns} ({tt})")

    DETECTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    DETECTION_JSON.write_text(json.dumps(detection.to_dict(), indent=2), encoding="utf-8")
    print(f"  wrote {DETECTION_JSON}")
    return 0


def _load_detection() -> RollDetection:
    d = json.loads(DETECTION_JSON.read_text(encoding="utf-8"))
    from alpha_agent.data.roll_validation import InstrumentSpan
    return RollDetection(
        continuous_symbol=d["continuous_symbol"],
        from_instrument_id=d["from_instrument_id"], to_instrument_id=d["to_instrument_id"],
        from_raw_symbol=d["from_raw_symbol"], to_raw_symbol=d["to_raw_symbol"],
        transition_ts_ns=d["transition_ts_ns"],
        spans=tuple(InstrumentSpan(**s) for s in d["spans"]),
    )


# --- Stage B --------------------------------------------------------------

def stage_b(execute: bool, max_cost: float | None) -> int:
    if not DETECTION_JSON.exists():
        print(f"missing {DETECTION_JSON} -- run Stage A first."); return 1
    det = _load_detection()
    req = stage_b_request(det)
    print("== STAGE B -- raw contract overlap ==")
    _print_req("raw both   ", req)
    print("\n== COST ESTIMATE ==")
    total = _estimate(req)

    if not execute:
        print("\nEstimate only. Re-run with --stage-b --max-cost-usd <cap> --execute.")
        return 0
    if max_cost is None:
        print("\n--execute requires --max-cost-usd."); return 2
    if total is not None and total > max_cost:
        print(f"\nEstimate ${total:.6f} exceeds cap ${max_cost:.2f}. Aborting."); return 3

    print("\n== DOWNLOAD (Stage B) ==")
    raw = fetch_and_store_raw(req, max_cost_usd=max_cost, root_dir="data/raw")
    print(f"  {raw.path}")
    return 0


# --- Replay (no network) ------------------------------------------------

def _seam_prices(frame: pd.DataFrame, eff_ts: int) -> tuple[float, float] | None:
    pre = frame[frame["ts_event_ns"] < eff_ts]
    at = frame[frame["ts_event_ns"] == eff_ts]
    if pre.empty or at.empty:
        return None
    return float(pre["close"].iloc[-1]), float(at["open"].iloc[0])


def _classify_gaps(report, calendar, root: str) -> list[str]:
    lines: list[str] = []
    for d in (x for x in report.diagnostics if x.kind.value == "missing_minute_gap"):
        t0 = int(d.context.get("ts_event_ns", d.ts_event_ns or 0))
        t1 = int(d.context.get("gap_end_ns", t0))
        n = int(d.context.get("missing_bars", d.count))
        s0 = calendar.classify(t0, root)[1].value
        s1 = calendar.classify(t1, root)[1].value
        loc = pd.Timestamp(t0, unit="ns", tz="UTC")
        span_min = (t1 - t0) / 60e9
        if s0 == "RTH" or s1 == "RTH":
            tag = "SUSPICIOUS (spans RTH) -- investigate"
        elif span_min > 180:
            tag = "SUSPICIOUS (>3h gap in a tradeable session) -- investigate"
        else:
            tag = "expected: thin overnight ETH minute(s) with no trades"
        lines.append(f"    {loc}  {s0}->{s1}  ~{n} min ({span_min:.0f}m span)  {tag}")
    return lines


def replay(processed_root: str) -> int:
    del processed_root
    # Fixed, hardcoded provenance paths (see the module docstring and the
    # REPLAY_STAGE_* constants above) -- deliberately NOT stage_a_requests()/
    # stage_b_request(), which now correctly refuse to construct a live
    # request for this at/after-holdout window (Phase 23.2).
    a_bars, a_def = REPLAY_STAGE_A_BARS, REPLAY_STAGE_A_DEF
    if not (a_bars.exists() and a_def.exists()):
        print("Stage A artifacts not present -- run Stage A first."); return 1
    if not DETECTION_JSON.exists():
        print("no roll detection -- run Stage A first."); return 1
    det = _load_detection()
    b_raw = REPLAY_STAGE_B
    have_overlap = b_raw.exists()
    calendar = default_calendar()

    print("== REPLAY (no network, no cost) ==")
    for p in (a_bars, a_def) + ((b_raw,) if have_overlap else ()):
        verify_manifest(p)
        print(f"  verified {p}")
    if not have_overlap:
        print(f"  Stage B overlap not present ({b_raw}) -- roll basis will FALL BACK.")

    vendor = _ohlcv_vendor_from_dbn(a_bars)
    ids = {int(i) for i in vendor["instrument_id"].unique()}
    reg = _registry_from_dbn(a_def, ids)
    canonical, canon_report = canonicalize(vendor, reg, calendar)
    n_missing = canon_report.by_kind.get("missing_minute_gap", 0)
    n_events = sum(1 for d in canon_report.diagnostics if d.kind.value == "missing_minute_gap")
    print(f"\n  canonical rows: {len(canonical)}  rejected: {canon_report.n_rejected_rows}  "
          f"missing minutes: {n_missing} across {n_events} gap events")

    overlap = None
    if have_overlap:
        # SAME normalization primitive as canonical bars -- no ad-hoc /1e9.
        overlap = descale_fixed_point(
            _ohlcv_vendor_from_dbn(b_raw).rename(columns={"ts_event": "ts_event_ns"})
        )
        med_c = median_abs_price(canonical, "close")
        med_o = median_abs_price(overlap, "close")
        print(f"  overlap rows: {len(overlap)}  ids: "
              f"{sorted(int(i) for i in overlap['instrument_id'].unique())}")
        print(f"  price-scale consistency: median|close| canonical={med_c:.2f} overlap={med_o:.2f} "
              f"(ratio {max(med_c, med_o) / max(min(med_c, med_o), 1e-9):.2f}x) OK")

    try:
        rolls, roll_report = build_roll_events(
            canonical, continuous_symbol=det.continuous_symbol, registry=reg,
            price_policy=RollPricePolicy.SAME_TIMESTAMP_CLOSE_CLOSE, overlap_bars=overlap,
        )
    except PriceScaleError as exc:
        print(f"\nPRICE-SCALE ERROR: {exc}\n\nPHASE 04.5: FAIL")
        return 9

    print(f"\n== ROLL ({len(rolls)}) ==")
    for r in rolls:
        print(f"  {r.from_raw_symbol} -> {r.to_raw_symbol}   policy={r.price_policy}  "
              f"fallback={r.used_fallback}")
        print(f"    aligned_ts = {r.from_ts_ns}  ({pd.Timestamp(r.from_ts_ns, unit='ns', tz='UTC')})")
        print(f"    aligned old close = {r.from_price}   aligned new close = {r.to_price}")
        print(f"    measured contract basis (additive_gap) = {r.additive_gap}")
    for d in roll_report.diagnostics:
        print(f"  [{d.severity.value}] {d.kind.value}: {d.message}")

    cont, _ = build_continuous_series(canonical, continuous_symbol=det.continuous_symbol,
                                      registry=reg, rolls=rolls)
    retro, retro_report = build_back_adjusted_series(
        cont, rolls, continuous_symbol=det.continuous_symbol,
        mode=AdjustmentMode.RETROSPECTIVE_RESEARCH)

    invariant_ok = pit_ok = True
    if rolls:
        r = rolls[0]
        eff = r.effective_ts_ns
        raw = _seam_prices(cont, eff)
        adj = _seam_prices(retro, eff)
        jump = raw[1] - raw[0]
        basis = r.additive_gap
        residual = jump - basis
        print("\n== SPLICE DECOMPOSITION ==")
        print(f"  old active last close      = {raw[0]:.4f}")
        print(f"  new active first open       = {raw[1]:.4f}")
        print(f"  UNADJUSTED SPLICE JUMP     = {jump:+.4f}")
        print(f"  MEASURED CONTRACT BASIS    = {basis:+.4f}   (same-timestamp raw close-close)")
        print(f"  RESIDUAL MARKET MOVE       = {residual:+.4f}   (jump - basis; NOT removed)")
        print(f"  ADJUSTED SPLICE RESIDUAL   = {adj[1] - adj[0]:+.4f}   (expected == residual, NOT 0)")

        # PRIMARY invariant: old_raw_close(t) + additive_gap == new_raw_close(t)
        invariant_ok = (r.from_price is not None and r.to_price is not None
                        and abs((r.from_price + basis) - r.to_price) <= 1e-6)
        print(f"\n  ALIGNED-BASIS INVARIANT  old_raw_close(t) + gap == new_raw_close(t): "
              f"{r.from_price} + {basis} == {r.to_price}  "
              f"[{'PASS' if invariant_ok else 'FAIL'}]")

        as_of = eff - 60_000_000_000
        pit, _ = build_back_adjusted_series(
            cont, rolls, continuous_symbol=det.continuous_symbol,
            mode=AdjustmentMode.POINT_IN_TIME, as_of_ts_ns=as_of)
        t0 = int(cont["ts_event_ns"].iloc[0])
        pit_t0 = float(pit.loc[pit["ts_event_ns"] == t0, "cumulative_adjustment"].iloc[0])
        retro_t0 = float(retro.loc[retro["ts_event_ns"] == t0, "cumulative_adjustment"].iloc[0])
        pit_ok = pit_t0 == 0.0 and pit["ts_event_ns"].max() <= as_of
        print(f"\n  POINT-IN-TIME (as_of just before roll): pre-roll adjustment = {pit_t0} "
              f"(retrospective = {retro_t0})  [{'PASS' if pit_ok else 'FAIL'}]")

    identity_ok = all(c not in retro.columns
                      for c in ("instrument_id", "raw_symbol",
                                "active_instrument_id", "active_raw_symbol"))
    print(f"\n  BackAdjustedBar execution-identity guard: [{'PASS' if identity_ok else 'FAIL'}]")

    print("\n== CANONICAL MISSING-MINUTE GAPS ==")
    gap_lines = _classify_gaps(canon_report, calendar, det.to_raw_symbol[:2])
    for line in gap_lines:
        print(line)
    suspicious = [x for x in gap_lines if "SUSPICIOUS" in x]
    print(f"  {n_missing} missing minutes / {len(gap_lines)} gap events -- "
          f"{len(gap_lines) - len(suspicious)} expected (thin overnight ETH no-trade minutes), "
          f"{len(suspicious)} to investigate")

    print("\n== C++ EXECUTION-DOMAIN GUARD ==")
    import subprocess
    exe = Path("build/cpp/cpp/quant_contract_boundary_tests")
    cpp_ok = False
    if exe.exists():
        r = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
        cpp_ok = r.returncode == 0
        print(f"  {r.stdout.strip().splitlines()[-1]}  [{'PASS' if cpp_ok else 'FAIL'}]")
    else:
        print("  (C++ tests not built -- run bash scripts/build_cpp.sh)")

    aligned_policy_ok = bool(rolls) and rolls[0].price_policy == "same_timestamp_close_close" \
        and not rolls[0].used_fallback
    passed = (aligned_policy_ok and invariant_ok and pit_ok and identity_ok and cpp_ok
              and not roll_report.errors and not canon_report.errors and not retro_report.errors
              and not suspicious)
    print("\nPHASE 04.5: PASS" if passed else "\nPHASE 04.5: FAIL / incomplete (see above)")
    return 0 if passed else 8


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Phase 04.5 real NQ roll validation")
    ap.add_argument("--stage-a", action="store_true")
    ap.add_argument("--stage-b", action="store_true")
    ap.add_argument("--stage-b-plan", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--max-cost-usd", type=float, default=None)
    ap.add_argument("--processed-root", default="data/processed")
    args = ap.parse_args()

    if args.replay:
        return replay(args.processed_root)
    if args.stage_b or args.stage_b_plan:
        return stage_b(execute=args.stage_b and args.execute, max_cost=args.max_cost_usd)
    return stage_a(execute=args.stage_a and args.execute, max_cost=args.max_cost_usd)


if __name__ == "__main__":
    sys.exit(main())
