"""Phase 13.5C -- contract-economics QA for the five real CME roots.

Audits the DERIVED USD point value / tick value of each root's contracts
(``ContractSpec.multiplier``, produced by the Phase 03.5/04.5
``contract_economics`` derivation from the stored Databento definition records)
against the published CME contract specifications.

A mismatch is a typed ``DATA_QUALITY_FAILURE``: every economic number the C++
engine produces for that root (Fill PnL, costs ratio, Sharpe, verdict) is
mis-scaled and MUST NOT be interpreted as a research result.

STRICTLY OFFLINE. No Databento. No 2025.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from alpha_agent.data.calendars import default_calendar
from alpha_agent.data.real_market_dataset import ROOTS, reconstitute_root

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "phase_13_5c" / "CONTRACT_ECONOMICS_QA.json"

# Published CME contract specifications (USD PnL per 1.0 move in the quoted
# price, and the quoted minimum price increment of the front month).
REFERENCE = {
    "ES": {"name": "E-mini S&P 500", "point_value_usd": 50.0, "tick_size": 0.25},
    "NQ": {"name": "E-mini Nasdaq-100", "point_value_usd": 20.0, "tick_size": 0.25},
    "CL": {"name": "WTI Crude Oil", "point_value_usd": 1000.0, "tick_size": 0.01},
    "GC": {"name": "Gold", "point_value_usd": 100.0, "tick_size": 0.10},
    "ZN": {"name": "10-Year T-Note (fractional, 1/32nds)", "point_value_usd": 1000.0,
           "tick_size": 0.015625},
}
TOL = 1e-6


def audit_root(root: str, calendar) -> dict:
    recon = reconstitute_root(root, calendar=calendar)
    ref = REFERENCE[root]
    mults = sorted({float(m) for m in recon.contracts["multiplier"]})
    ticks = sorted({float(t) for t in recon.contracts["tick_size"]})
    derived_mult = mults[0] if len(mults) == 1 else None
    derived_tick = ticks[0] if len(ticks) == 1 else None
    ok_mult = derived_mult is not None and abs(derived_mult - ref["point_value_usd"]) <= TOL
    ok_tick = derived_tick is not None and abs(derived_tick - ref["tick_size"]) <= TOL
    return {
        "root": root,
        "contract": ref["name"],
        "n_contracts": len(recon.contracts),
        "derived_point_value_usd": derived_mult,
        "reference_point_value_usd": ref["point_value_usd"],
        "derived_tick_size": derived_tick,
        "reference_tick_size": ref["tick_size"],
        "derived_tick_value_usd": (
            round(derived_mult * derived_tick, 9)
            if derived_mult is not None and derived_tick is not None else None
        ),
        "reference_tick_value_usd": round(ref["point_value_usd"] * ref["tick_size"], 9),
        "point_value_error_factor": (
            round(ref["point_value_usd"] / derived_mult, 6)
            if derived_mult not in (None, 0.0) else None
        ),
        "uniform_across_contracts": len(mults) == 1 and len(ticks) == 1,
        "status": "PASS" if (ok_mult and ok_tick and len(mults) == 1) else "DATA_QUALITY_FAILURE",
        "reason_code": (
            "" if (ok_mult and ok_tick and len(mults) == 1)
            else ("derived_point_value_usd_does_not_match_published_contract_specification"
                  if not ok_mult else "derived_tick_size_does_not_match_published_specification")
        ),
    }


def main() -> int:
    cal = default_calendar()
    roots = sys.argv[1:] or list(ROOTS)
    results = [audit_root(r, cal) for r in roots]
    failures = [r for r in results if r["status"] != "PASS"]
    for r in results:
        print(f"  [{r['status']}] {r['root']}: derived point value "
              f"{r['derived_point_value_usd']} vs published {r['reference_point_value_usd']}"
              + (f"  (off by {r['point_value_error_factor']}x)" if r["status"] != "PASS" else ""))
    payload = {
        "phase": "13.5C",
        "check": "derived contract economics vs published CME contract specifications",
        "source": ("ContractSpec.multiplier / tick_size derived by "
                   "alpha_agent.data.contract_economics from the stored Databento "
                   "definition records (Phase 03.5 / 04.5 path)"),
        "pass": not failures,
        "n_failures": len(failures),
        "roots": sorted(results, key=lambda r: r["root"]),
        "impact_note": (
            "A mis-derived point value mis-scales every Fill-derived economic number "
            "for that root (gross PnL, net PnL, cost ratio, daily/annualized Sharpe) "
            "and therefore its verdict. Commissions are a flat USD-per-contract charge "
            "and are NOT rescaled, so the gross-to-cost ratio is corrupted and the "
            "result is not economically interpretable."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\n== contract-economics QA {'OK' if not failures else 'HAS FAILURES'} -> {OUT} ==")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
