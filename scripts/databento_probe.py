"""Cost-first Databento probe -- ESTIMATE ONLY.

Downloads moved to ``scripts/databento_validate.py`` (Phase 03.5), which owns the
raw DBN store, decode, canonicalize and C++ boundary chain.

    python scripts/databento_probe.py --symbols NQ.v.0 --start 2026-09-03 --end 2026-09-04
"""
from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401  # adds repo python/ to sys.path; must precede alpha_agent
from alpha_agent.data.databento_source import HistoricalRequest, estimate_cost_usd


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cost-first Databento probe (estimate only)")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--dataset", default="GLBX.MDP3")
    p.add_argument("--schema", default="ohlcv-1m")
    p.add_argument("--stype-in", default="continuous")
    # Databento does not support continuous/parent -> raw_symbol; instrument_id
    # is the authoritative join key (see docs/DATABENTO_REALITY.md).
    p.add_argument("--stype-out", default="instrument_id")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    return p.parse_args()


def main() -> None:
    load_dotenv_if_available()
    args = parse_args()
    if not os.getenv("DATABENTO_API_KEY"):
        raise SystemExit("DATABENTO_API_KEY is not set. Put it in local .env, not source code.")

    req = HistoricalRequest(
        dataset=args.dataset,
        symbols=tuple(args.symbols),
        schema=args.schema,
        stype_in=args.stype_in,
        stype_out=args.stype_out,
        start=args.start,
        end=args.end,
    )
    cost = estimate_cost_usd(req)
    print(f"Estimated Databento cost: ${cost:.6f}")
    print("Request:", req)
    print("Estimate only. Use scripts/databento_validate.py --execute to download.")


if __name__ == "__main__":
    main()
