"""Read-only Databento CAPABILITY probe for the Phase 6 ETF pilot's data-
feasibility gate (Phase 6 instruction 1: "inspect provider/data feasibility...
Stop for approval before paid data spend").

Uses ONLY free Databento metadata endpoints -- ``metadata.list_datasets``,
``metadata.get_dataset_range`` (no date arguments -- a dataset's own overall
bounds, unavoidable and non-price-bearing metadata), and ``metadata.get_cost``
(an ESTIMATE call, never a download) -- so running this script never
downloads a byte of market data and never incurs a charge on its own.

HOLDOUT DISCIPLINE (fixed after a real incident): this script's whole purpose
is to inform a SCIENTIFIC RESEARCH acquisition decision -- unlike
``scripts/databento_market_capability_probe.py`` (which legitimately checks
"today" for the separate, non-research MARKET-OBSERVATION plane), every date
this script sends to Databento MUST stay strictly before the locked research
holdout. An earlier version of this script called ``get_dataset_condition``
and a "tiny recent" ``get_cost`` using an unguarded ``datetime.now(UTC)``,
which reached into 2026 -- a real violation of CLAUDE.md's "any code path
touching ts >= 2025-01-01 must fail loudly" rule (metadata/cost-estimate
calls only; no price data was ever downloaded, so no scientific
contamination occurred, but the bright line was still crossed). Fixed here:
every ``get_cost`` window is bounded by the same ``HOLDOUT_START`` constant
the frozen research pipeline (``alpha_agent.data.databento_source``) already
enforces at construction time, via ``_assert_safe_window`` below, and
``get_dataset_condition`` (which only ever makes sense for a "how fresh is
today's data" question, not a research-acquisition question) has been
dropped from this script entirely.

NEVER prints, logs, or persists ``DATABENTO_API_KEY``.

    python scripts/databento_equity_capability_probe.py
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import _bootstrap  # noqa: F401  # adds repo python/ to sys.path; must precede alpha_agent
from alpha_agent.data.databento_source import HOLDOUT_START, HoldoutViolation

# Candidate US equities datasets -- checked against whatever this account is
# actually entitled to, never assumed. A single ETF (e.g. SPY on NYSE Arca,
# QQQ on Nasdaq) trades across many venues, so a single-exchange feed
# (XNAS.ITCH, ARCX.PILLAR) reflects only that venue's own prints, not the
# full consolidated tape; EQUS.SUMMARY and the DBEQ.* family are Databento's
# consolidated-across-venues EOD products and are the natural cross-check
# candidates, but this script still probes single-venue datasets too so the
# real entitlement list is the thing that decides, not a guess.
CANDIDATE_DATASETS = [
    "EQUS.SUMMARY",
    "DBEQ.BASIC",
    "DBEQ.PLUS",
    "DBEQ.MAX",
    "XNAS.ITCH",
    "ARCX.PILLAR",
    "XNYS.PILLAR",
    "IEXG.TOPS",
]

# Phase 6 finalized 14-ETF pilot universe (user-approved 2026-09-22):
# broad equity, Nasdaq/growth, small caps, 4 sectors, full rates curve,
# credit, gold, oil/energy.
FULL_CANDIDATE_UNIVERSE = [
    "SPY", "QQQ", "IWM",
    "XLK", "XLF", "XLE", "XLU",
    "SHY", "IEF", "TLT",
    "LQD", "HYG",
    "GLD", "USO",
]

# Hardcoded, safe research window -- NEVER derived from datetime.now(). The
# upper bound is 2024-12-31, one day before HOLDOUT_START (2025-01-01) --
# the tightest boundary that still cannot reach it, not a wide margin.
RESEARCH_WINDOW_START = "2018-05-01"
RESEARCH_WINDOW_END = "2024-12-31"


def _assert_safe_window(start: str, end: str) -> None:
    """Defense in depth, local to this script: refuses to even construct a
    Databento call whose window reaches the locked research holdout, mirroring
    ``alpha_agent.data.databento_source._assert_before_holdout``."""
    if start >= HOLDOUT_START or end >= HOLDOUT_START:
        raise HoldoutViolation(
            f"refusing to query Databento for [{start}, {end}) -- reaches the "
            f"locked research holdout (>= {HOLDOUT_START})"
        )


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def main() -> None:
    _assert_safe_window(RESEARCH_WINDOW_START, RESEARCH_WINDOW_END)  # fail loudly before any network call

    load_dotenv_if_available()
    key = os.getenv("DATABENTO_API_KEY")
    report: dict = {"checked_at": datetime.now(UTC).isoformat(), "research_window": [RESEARCH_WINDOW_START, RESEARCH_WINDOW_END]}

    if not key:
        report["capability"] = "NOT_CONNECTED"
        report["detail"] = "DATABENTO_API_KEY not set"
        print(json.dumps(report, indent=2, default=str))
        return

    try:
        import databento as db
    except ImportError:
        report["capability"] = "NOT_CONNECTED"
        report["detail"] = "databento SDK not installed"
        print(json.dumps(report, indent=2, default=str))
        return

    try:
        client = db.Historical(key)  # never logs/prints the key (SDK's own docstring guarantee)
    except Exception as exc:  # noqa: BLE001
        report["capability"] = "ERROR"
        report["detail"] = f"client construction failed: {type(exc).__name__}"
        print(json.dumps(report, indent=2, default=str))
        return

    # -- 1. list_datasets: the authoritative entitlement list for THIS key --
    try:
        all_datasets = client.metadata.list_datasets()
        report["n_datasets_entitled_total"] = len(all_datasets)
        report["candidate_equities_datasets_entitled"] = [
            d for d in CANDIDATE_DATASETS if d in all_datasets
        ]
        report["candidate_equities_datasets_not_entitled"] = [
            d for d in CANDIDATE_DATASETS if d not in all_datasets
        ]
    except Exception as exc:  # noqa: BLE001
        status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
        report["capability"] = "RATE_LIMITED" if status == 429 else "ERROR"
        report["detail"] = f"list_datasets failed: {type(exc).__name__}: {exc}"
        print(json.dumps(report, indent=2, default=str))
        return

    entitled = report["candidate_equities_datasets_entitled"]
    if not entitled:
        report["capability"] = "NO_EQUITIES_DATASET_ENTITLED"
        report["note"] = (
            "This Databento account is not entitled to any candidate equities "
            "dataset checked above -- acquiring one is a MONEY/NETWORK decision "
            "(new dataset entitlement, possibly a new subscription cost) that "
            "must go back to the user before any further Phase 6 data work."
        )
        print(json.dumps(report, indent=2, default=str))
        return

    # -- 2. get_dataset_range per entitled candidate: free metadata call, NO --
    # date arguments -- returns the dataset's own overall bounds. Reveals no
    # price information; this is the same call the already-accepted futures
    # probe makes for GLBX.MDP3.
    report["dataset_range"] = {}
    for ds in entitled:
        try:
            report["dataset_range"][ds] = client.metadata.get_dataset_range(dataset=ds)
        except Exception as exc:  # noqa: BLE001
            report["dataset_range"][ds] = f"ERROR: {type(exc).__name__}: {exc}"

    # -- 3. get_cost for the finalized 14-ETF research window, daily bars, ---
    # per entitled candidate -- an ESTIMATE only, never a download, never a
    # charge. Uses raw_symbol stype_in: ETFs have a single fixed ticker, no
    # continuous/parent roll symbology (unlike futures). Every window is the
    # same hardcoded, pre-holdout RESEARCH_WINDOW_* pair, re-asserted safe
    # immediately before each call.
    report["research_window_ohlcv_estimate_usd"] = {}
    for ds in entitled:
        try:
            _assert_safe_window(RESEARCH_WINDOW_START, RESEARCH_WINDOW_END)
            cost = client.metadata.get_cost(
                dataset=ds, symbols=FULL_CANDIDATE_UNIVERSE, schema="ohlcv-1d",
                stype_in="raw_symbol", start=RESEARCH_WINDOW_START, end=RESEARCH_WINDOW_END,
            )
            report["research_window_ohlcv_estimate_usd"][ds] = cost
        except HoldoutViolation:
            raise
        except Exception as exc:  # noqa: BLE001
            report["research_window_ohlcv_estimate_usd"][ds] = f"ERROR: {type(exc).__name__}: {exc}"

    # -- 4. definition schema cost (needed for corporate-action / listing ---
    # metadata inspection) for the same universe/window, entitled datasets.
    report["definition_estimate_usd"] = {}
    for ds in entitled:
        try:
            _assert_safe_window(RESEARCH_WINDOW_START, RESEARCH_WINDOW_END)
            cost = client.metadata.get_cost(
                dataset=ds, symbols=FULL_CANDIDATE_UNIVERSE, schema="definition",
                stype_in="raw_symbol", start=RESEARCH_WINDOW_START, end=RESEARCH_WINDOW_END,
            )
            report["definition_estimate_usd"][ds] = cost
        except HoldoutViolation:
            raise
        except Exception as exc:  # noqa: BLE001
            report["definition_estimate_usd"][ds] = f"ERROR: {type(exc).__name__}: {exc}"

    report["capability"] = "EQUITIES_DATASET_ENTITLED"
    report["note"] = (
        "Corporate actions (dividends/splits) are NOT covered by any Databento "
        "schema probed here -- Databento is a market-data vendor, not a "
        "corporate-actions vendor. See the accompanying Phase 6 feasibility "
        "report for how the candidate universe's split/dividend history is "
        "handled."
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
