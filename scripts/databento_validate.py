"""Phase 03.5 -- real Databento integration validation.

ONE tiny authenticated request (one trading day of NQ.v.0 ohlcv-1m + the
matching instrument definitions), cost-gated, then the full
raw -> canonical -> C++ CLI chain against real bytes.

    python scripts/databento_validate.py                      # design + cost estimate, no download
    python scripts/databento_validate.py --max-cost-usd 1.00 --execute   # LEGACY, see below
    python scripts/databento_validate.py --replay            # reuse the stored raw DBN, NO network

DEPRECATED LIVE-FETCH MODE (Phase 23.2): the hardcoded window below
(``START, END = "2026-09-03", "2026-09-04"``) is a permanent, immutable
provenance record of this script's one real Phase 03.5 integration run --
never delete or edit it (see ``data/manifests/real_dataset/spend_ledger.json``
and ``docs/FINAL_SYSTEM_AUDIT.md`` C1). It is also, as of Phase 23.1, at/after
the project's locked ``>= 2025-01-01`` research holdout boundary, which this
script's original design never guarded against (it predates that guard).
``alpha_agent.data.databento_source.HistoricalRequest`` now refuses any
holdout-crossing window at construction time (the universal, lowest-shared-
boundary fix), so both the design/estimate path and ``--execute`` now raise
``HoldoutViolation`` before printing the request, checking for an API key, or
touching the network -- by design, not a bug. This script's original
validation purpose (prove the raw -> canonical -> C++ CLI chain works against
real bytes) is fully and losslessly satisfied by ``--replay`` against the
already-downloaded, sha256-verified artifacts; the live-fetch mode is kept
only so the historical request shape stays documented and is not expected to
ever run again.

Never prints the API key. Aborts if the estimate exceeds --max-cost-usd.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  # repo python/ on sys.path
import pandas as pd
from alpha_agent.adapters.cpp_cli import canonical_bars_to_boundary, write_boundary_bundle
from alpha_agent.data import decode as dec
from alpha_agent.data.calendars import default_calendar
from alpha_agent.data.canonicalize import NormalizationPolicy, canonicalize
from alpha_agent.data.contract_economics import EconomicsError
from alpha_agent.data.databento_source import (
    HistoricalRequest,
    estimate_cost_usd,
    fetch_and_store_raw,
)
from alpha_agent.data.definitions import (
    DefinitionRegistry,
    contract_economics_for,
    contracts_frame,
    parse_definition_frame,
)
from alpha_agent.data.raw_store import RawArtifact, verify_manifest
from alpha_agent.schemas.market_data import VENDOR_DEFINITION_COLUMNS, VENDOR_OHLCV_COLUMNS

DATASET, SYMBOL, SCHEMA = "GLBX.MDP3", "NQ.v.0", "ohlcv-1m"
START, END = "2026-09-03", "2026-09-04"          # one trading day, end exclusive

# The already-downloaded immutable raw artifacts (Phase 03.5 first live run).
REPLAY_OHLCV = "data/raw/databento/GLBX.MDP3/ohlcv-1m/ad665732c52dfea5/ohlcv-1m.dbn.zst"
REPLAY_DEF = "data/raw/databento/GLBX.MDP3/definition/24a980b7c26bfc89/definition.dbn.zst"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _bars_request() -> HistoricalRequest:
    return HistoricalRequest(
        dataset=DATASET, symbols=(SYMBOL,), schema=SCHEMA,
        start=START, end=END, stype_in="continuous", stype_out="instrument_id",
    )


def _print_request(req: HistoricalRequest) -> None:
    d = req.definition_request()
    print("== EXACT REQUEST ==")
    print(f"  bars:        {req.dataset} {list(req.symbols)} {req.schema} "
          f"stype_in={req.stype_in} stype_out={req.stype_out} {req.start}..{req.end}")
    print(f"  definitions: {d.dataset} {list(d.symbols)} {d.schema} "
          f"stype_in={d.stype_in} stype_out={d.stype_out} {d.start}..{d.end}")


def _describe(df: pd.DataFrame, name: str) -> None:
    print(f"\n== REAL {name} SCHEMA ==")
    print(f"  rows: {len(df)}  columns ({len(df.columns)}): {list(df.columns)}")
    for c, t in df.dtypes.items():
        print(f"    {c}: {t}")
    if len(df):
        print("  first row:", {c: repr(v) for c, v in df.iloc[0].items()})


def _compare_columns(actual: list[str], expected: tuple[str, ...], name: str) -> None:
    missing = [c for c in expected if c not in actual]
    print(f"\n== {name} vs fixture ==")
    print("  MISSING in real response:" if missing else "  all fixture columns present:",
          missing or "")


def _decode_and_validate(bars_raw: RawArtifact, def_raw: RawArtifact, args) -> int:
    verify_manifest(bars_raw.path)
    verify_manifest(def_raw.path)
    print(f"  raw bars:        {bars_raw.path}  sha256={bars_raw.manifest.sha256[:12]}...")
    print(f"  raw definitions: {def_raw.path}  sha256={def_raw.manifest.sha256[:12]}...")
    print("  manifest verification: OK")

    bars_store = dec.load_dbn(bars_raw.path)
    def_store = dec.load_dbn(def_raw.path)
    bars_df_raw = bars_store.to_df(price_type="fixed", pretty_ts=False, map_symbols=True).reset_index()
    def_df_raw = def_store.to_df(pretty_ts=False, map_symbols=False).reset_index()
    _describe(bars_df_raw, "OHLCV")
    _describe(def_df_raw, "DEFINITION")
    _compare_columns(list(bars_df_raw.columns), VENDOR_OHLCV_COLUMNS, "OHLCV")
    _compare_columns(list(def_df_raw.columns), VENDOR_DEFINITION_COLUMNS, "DEFINITION")

    ohlcv_vendor = dec.ohlcv_df_to_vendor_frame(bars_df_raw)
    seen_ids = {int(i) for i in ohlcv_vendor["instrument_id"].unique()}
    print(f"\n== BARS ==\n  requested-symbol label(s): {sorted(set(ohlcv_vendor['symbol'].unique()))}")
    print(f"  instrument_id(s) (authoritative key): {sorted(seen_ids)}")

    try:
        def_vendor = dec.definition_df_to_vendor_frame(def_df_raw, keep_instrument_ids=seen_ids)
        econ_by_id = contract_economics_for(def_vendor)
        specs = parse_definition_frame(def_vendor)
    except EconomicsError as exc:
        print(f"\nECONOMICS -- stopping rather than guessing:\n  {exc}")
        return 4
    except dec.SchemaMismatch as exc:
        print(f"\nSCHEMA MISMATCH -- stopping:\n  {exc}")
        return 5

    unresolved = seen_ids - {s.instrument_id for s in specs}
    if unresolved:
        print(f"\nUNRESOLVED -- bars reference instrument_id(s) with no definition: {sorted(unresolved)}")
        return 6

    print("\n== CONTRACT ECONOMICS ==")
    for s in specs:
        e = econ_by_id[s.instrument_id]
        print(f"  {s.raw_symbol} (id {s.instrument_id}) root={s.root_symbol} exch={s.exchange}")
        print(f"    quote_tick_size = {e.quote_tick_size}   contract_size = {e.contract_size} "
              f"{e.unit_of_measure}")
        print(f"    price_scale     = {e.price_scale}")
        print(f"    point_value_usd = {e.point_value_usd}   (= ContractSpec.multiplier)")
        print(f"    tick_value_usd  = {e.tick_value_usd}")
        print(f"    cross-check tick_size * multiplier == tick_value: "
              f"{s.tick_size * s.multiplier} == {e.tick_value_usd}  "
              f"[{'OK' if s.tick_size * s.multiplier == e.tick_value_usd else 'FAIL'}]")
        print(f"    activation={s.activation_ns} expiration={s.expiration_ns} "
              f"first_notice={s.first_notice_ns or 'N/A'} last_trade={s.last_trade_ns or 'N/A'}")

    registry = DefinitionRegistry(specs)
    canonical, report = canonicalize(
        ohlcv_vendor, registry, default_calendar(), policy=NormalizationPolicy()
    )
    print("\n== CANONICALIZE DIAGNOSTICS ==")
    print(json.dumps(report.summary(), indent=2))
    for d in report.diagnostics:
        print(f"  [{d.severity.value}] {d.kind.value}: {d.message}")
    if not canonical.empty:
        print(f"  ts coverage: {int(canonical['ts_event_ns'].min())} .. "
              f"{int(canonical['ts_event_ns'].max())}")
        print(f"  trading_day(s): {sorted(set(canonical['trading_day']))}")
        print(f"  session counts: {canonical['session'].value_counts().to_dict()}")
        print(f"  raw contract(s): {sorted(set(canonical['raw_symbol']))}")

    proc = Path(args.processed_root)
    (proc / "bars" / "NQ").mkdir(parents=True, exist_ok=True)
    (proc / "contracts").mkdir(parents=True, exist_ok=True)
    contracts_frame(specs).to_parquet(proc / "contracts" / "NQ.parquet", index=False)
    for raw_symbol, grp in canonical.groupby("raw_symbol"):
        grp.reset_index(drop=True).to_parquet(proc / "bars" / "NQ" / f"{raw_symbol}.parquet", index=False)

    exe = Path("build/cpp/cpp/quant_backtest_csv")
    if not exe.exists():
        print("\nC++ core not built; skipping CLI step.")
        return 0
    bars_csv, contracts_csv = write_boundary_bundle(
        canonical_bars_to_boundary(canonical), contracts_frame(specs),
        Path(args.staging_root) / "boundary",
    )
    out = subprocess.run(
        [str(exe), str(bars_csv), str(contracts_csv), "5", "0.0"],
        check=True, capture_output=True, text=True,
    )
    cli = json.loads(out.stdout)
    print(f"\n== PYTHON -> C++ CLI ==\n  {out.stdout.strip()}")
    print(f"  contracts_resolved: {cli['contracts_resolved']} / bars: {cli['bars']}")
    print("\nPHASE 03.5: PASS" if cli["contracts_resolved"] >= 1 and not report.errors
          else "\nPHASE 03.5: review diagnostics")
    return 0


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Phase 03.5 real Databento validation")
    ap.add_argument("--max-cost-usd", type=float, default=None)
    ap.add_argument("--execute", action="store_true", help="perform the live download")
    ap.add_argument("--replay", action="store_true",
                    help="reuse the already-stored raw DBN files; no network, no cost")
    ap.add_argument("--replay-ohlcv", default=REPLAY_OHLCV)
    ap.add_argument("--replay-def", default=REPLAY_DEF)
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--processed-root", default="data/processed")
    ap.add_argument("--staging-root", default="data/staging")
    args = ap.parse_args()

    if args.replay:
        print("== REPLAY (no network, no cost) ==")
        for p in (args.replay_ohlcv, args.replay_def):
            if not Path(p).exists():
                print(f"missing stored raw artifact: {p}")
                return 1
        bars_raw = verify_manifest(args.replay_ohlcv)
        def_raw = verify_manifest(args.replay_def)
        return _decode_and_validate(bars_raw, def_raw, args)

    req = _bars_request()
    _print_request(req)
    if not os.getenv("DATABENTO_API_KEY"):
        print("\nNO USABLE DATABENTO_API_KEY. Put it in local .env (not .env.example).")
        print("Or use --replay to run against the stored raw DBN.")
        return 0

    def_req = req.definition_request()
    try:
        bars_cost, def_cost = estimate_cost_usd(req), estimate_cost_usd(def_req)
    except Exception as exc:  # noqa: BLE001
        print(f"\nCost estimate failed: {type(exc).__name__}: {exc}")
        return 1
    total = bars_cost + def_cost
    print(f"\n== COST ESTIMATE ==\n  bars ${bars_cost:.6f} + definitions ${def_cost:.6f} "
          f"= ${total:.6f}")

    if not args.execute:
        print("\nEstimate only. Add --max-cost-usd <cap> --execute to download, or --replay.")
        return 0
    if args.max_cost_usd is None:
        print("\n--execute requires --max-cost-usd. Aborting.")
        return 2
    if total > args.max_cost_usd:
        print(f"\nEstimate ${total:.6f} exceeds cap ${args.max_cost_usd:.2f}. Aborting.")
        return 3

    print("\n== DOWNLOAD ==")
    bars_raw = fetch_and_store_raw(req, max_cost_usd=args.max_cost_usd, root_dir=args.raw_root)
    def_raw = fetch_and_store_raw(def_req, max_cost_usd=args.max_cost_usd, root_dir=args.raw_root)
    return _decode_and_validate(bars_raw, def_raw, args)


if __name__ == "__main__":
    sys.exit(main())
