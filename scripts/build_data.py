"""Phase 13.5C -- STRICTLY OFFLINE reconstitution of the real 2018-2024 CME
research / validation dataset from the already-acquired immutable raw DBN
artifacts.

    python scripts/phase_13_5c_build_data.py --verify            # QA gate, no writes
    python scripts/phase_13_5c_build_data.py --verify --write    # + persist derived parquet

No network. No Databento download. No metadata.get_cost / get_range payload call.
A missing raw artifact FAILS LOUDLY (MissingRawArtifact) naming the request.
Nothing on/after 2025-01-01 is ever loaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from alpha_agent.data.calendars import default_calendar
from alpha_agent.data.real_market_dataset import (
    HOLDOUT_START_NS,
    ROOTS,
    MissingRawArtifact,
    ReconstitutedRoot,
    daily_signal_series,
    degraded_trading_days,
    reconstitute_root,
)

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "data" / "processed" / "phase_13_5c"


def qa_gate(recon: ReconstitutedRoot) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True

    def chk(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        lines.append(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    b = recon.canonical_bars
    chk("canonical bars present", len(b) > 0, f"{len(b):,} rows")
    chk("ts strictly increasing per contract",
        all(g["ts_event_ns"].is_monotonic_increasing and not g["ts_event_ns"].duplicated().any()
            for _, g in b.groupby("instrument_id", sort=False)))
    chk("no duplicate (instrument_id, ts)", not b.duplicated(["instrument_id", "ts_event_ns"]).any())
    known = set(recon.contracts["instrument_id"])
    chk("every bar resolves to a registered contract",
        set(b["instrument_id"]).issubset(known),
        f"{len(set(b['instrument_id']) - known)} unresolved")
    chk("OHLC invariants",
        bool(((b["low"] <= b[["open", "close"]].min(axis=1))
              & (b[["open", "close"]].max(axis=1) <= b["high"])).all()))
    chk("trading_day + session labels present",
        {"trading_day", "session"}.issubset(b.columns)
        and not b["trading_day"].isna().any() and not b["session"].isna().any())
    chk("no ts on/after 2025-01-01 (holdout untouched)",
        int(b["ts_event_ns"].max()) < HOLDOUT_START_NS,
        f"max ts {int(b['ts_event_ns'].max())}")
    chk("roll transitions == roll-map rows",
        recon.n_transitions == len(recon.rolls),
        f"{recon.n_transitions} transitions / {len(recon.rolls)} rolls")
    chk("no roll used a fallback basis",
        all(not r.used_fallback for r in recon.rolls),
        f"{sum(r.used_fallback for r in recon.rolls)} fallbacks")
    chk("every roll has an additive gap",
        all(r.additive_gap is not None for r in recon.rolls))
    fwd = recon.forward_adjusted
    chk("forward-adjusted signal series built (causal)",
        len(fwd) == len(recon.continuous) and (fwd["adjustment_mode"] == "forward_adjusted").all())
    # forward-adjust removes the roll jump: no cumulative_adjustment change is a
    # future-roll basis (each change coincides with a roll effective_ts)
    roll_ts = {int(r.effective_ts_ns) for r in recon.rolls}
    changes = fwd.loc[fwd["cumulative_adjustment"].diff().fillna(0) != 0, "ts_event_ns"]
    change_ts = {int(t) for t in changes}
    chk("adjustment steps land only on roll effective timestamps",
        change_ts.issubset(roll_ts),
        f"{len(change_ts - roll_ts)} stray steps")
    cal = default_calendar()
    ds = daily_signal_series(fwd, recon.root, calendar=cal)
    chk("daily signal = one row per (root, trading_day)",
        len(ds) == ds["trading_day"].nunique() and len(ds) > 1000,
        f"{len(ds)} rows / {ds['trading_day'].nunique()} trading days")
    chk("daily signal stamps are real 1m bar timestamps",
        set(ds["ts_event_ns"]).issubset(set(fwd["ts_event_ns"])))
    return ok, lines


def _persist(recon: ReconstitutedRoot, cal) -> None:
    for sub in ("continuous", "forward_adjusted", "daily_signal", "contracts", "bars"):
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
    recon.continuous.to_parquet(OUT_ROOT / "continuous" / f"{recon.root}.v.0.parquet", index=False)
    recon.forward_adjusted.to_parquet(
        OUT_ROOT / "forward_adjusted" / f"{recon.root}.v.0.parquet", index=False
    )
    daily_signal_series(recon.forward_adjusted, recon.root, calendar=cal).to_parquet(
        OUT_ROOT / "daily_signal" / f"{recon.root}.parquet", index=False
    )
    recon.contracts.to_parquet(OUT_ROOT / "contracts" / f"{recon.root}.parquet", index=False)
    d = OUT_ROOT / "bars" / recon.root
    d.mkdir(parents=True, exist_ok=True)
    for raw_symbol, grp in recon.canonical_bars.groupby("raw_symbol"):
        grp.reset_index(drop=True).to_parquet(d / f"{raw_symbol}.parquet", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 13.5C offline data reconstitution")
    ap.add_argument("--verify", action="store_true", help="run the QA gate")
    ap.add_argument("--write", action="store_true", help="persist derived parquet under data/processed/phase_13_5c/")
    ap.add_argument("--roots", nargs="*", default=list(ROOTS))
    args = ap.parse_args()

    cal = default_calendar()
    all_ok = True
    for root in args.roots:
        print(f"\n=== {root} (offline reconstitution, 2018-2024) ===")
        try:
            recon = reconstitute_root(root, calendar=cal)
        except MissingRawArtifact as e:
            print(f"  MISSING RAW ARTIFACT -- offline reconstitution cannot proceed:\n  {e}")
            return 3
        print(f"  canonical bars {len(recon.canonical_bars):,} | contracts {len(recon.contracts)} | "
              f"rolls {len(recon.rolls)} | transitions {recon.n_transitions}")
        print(f"  degraded trading_days: {degraded_trading_days(root, calendar=cal)}")
        if args.verify:
            ok, lines = qa_gate(recon)
            print("\n".join(lines))
            all_ok = all_ok and ok
        if args.write:
            _persist(recon, cal)
            print(f"  persisted -> {OUT_ROOT}")

    print(f"\n== Phase 13.5C data reconstitution {'OK' if all_ok else 'HAS QA FAILURES'} ==")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
