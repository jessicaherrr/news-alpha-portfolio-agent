"""Read-only Databento CAPABILITY probe for the Phase 9.1 Equities
foundation's data-feasibility gate (approved Phase 9 proposal, goal 5:
"Inspect the approved proposal and current Databento capabilities/cost
path... stop for approval before purchasing").

Uses ONLY free Databento metadata endpoints -- ``metadata.list_datasets``,
``metadata.get_dataset_range`` (no date arguments -- a dataset's own overall
bounds, unavoidable and non-price-bearing metadata), and ``metadata.get_cost``
(an ESTIMATE call, never a download) -- so running this script never
downloads a byte of market data and never incurs a charge on its own. This is
a direct Phase 9.1 mirror of ``scripts/databento_equity_capability_probe.py``
(Phase 6 ETF pilot), reused as a pattern rather than edited in place, since
that script's own findings remain the ETF pilot's committed evidence.

HOLDOUT DISCIPLINE: every date this script sends to Databento is a hardcoded,
pre-holdout constant -- never derived from ``datetime.now()`` -- and is
re-asserted safe immediately before every ``get_cost`` call via
``_assert_safe_window``, mirroring
``alpha_agent.data.databento_source._assert_before_holdout``.

NEVER prints, logs, or persists ``DATABENTO_API_KEY``.

    python scripts/phase9_1_equity_capability_probe.py
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import _bootstrap  # noqa: F401  # adds repo python/ to sys.path; must precede alpha_agent
from alpha_agent.data.databento_source import HOLDOUT_START, HoldoutViolation
from alpha_agent.equities.universe import EQUITY_UNIVERSE

# Same candidate US equities datasets the Phase 6 ETF probe already checked --
# the underlying Databento product catalog for a single-name equity and an
# ETF is identical; this script still probes the full candidate list rather
# than assuming only XNAS.ITCH/XNYS.PILLAR (this universe's declared primary
# venues) are relevant, so the real entitlement list is the thing that
# decides, not a guess.
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

# Hardcoded, safe research window -- NEVER derived from datetime.now(). The
# upper bound is 2024-12-31, one day before HOLDOUT_START (2025-01-01) --
# the tightest boundary that still cannot reach it, matching the ETF probe.
RESEARCH_WINDOW_START = "2018-05-01"
RESEARCH_WINDOW_END = "2024-12-31"


def _assert_safe_window(start: str, end: str) -> None:
    """Defense in depth, local to this script: refuses to even construct a
    Databento call whose window reaches the locked research holdout."""
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
    report: dict = {
        "checked_at": datetime.now(UTC).isoformat(),
        "research_window": [RESEARCH_WINDOW_START, RESEARCH_WINDOW_END],
        "universe": list(EQUITY_UNIVERSE),
        "universe_size": len(EQUITY_UNIVERSE),
    }

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
            "This Databento account is not entitled to any candidate equities dataset "
            "checked above -- acquiring one is a MONEY/NETWORK decision that must go back "
            "to the user before any further Phase 9.1 data work."
        )
        print(json.dumps(report, indent=2, default=str))
        return

    # -- 2. get_dataset_range per entitled candidate: free metadata call, ---
    # no date arguments -- returns the dataset's own overall bounds.
    report["dataset_range"] = {}
    for ds in entitled:
        try:
            report["dataset_range"][ds] = client.metadata.get_dataset_range(dataset=ds)
        except Exception as exc:  # noqa: BLE001
            report["dataset_range"][ds] = f"ERROR: {type(exc).__name__}: {exc}"

    # -- 3. get_cost for the proposed 20-name research window, daily bars, --
    # per entitled candidate -- an ESTIMATE only, never a download, never a
    # charge.
    report["research_window_ohlcv_estimate_usd"] = {}
    for ds in entitled:
        try:
            _assert_safe_window(RESEARCH_WINDOW_START, RESEARCH_WINDOW_END)
            cost = client.metadata.get_cost(
                dataset=ds, symbols=list(EQUITY_UNIVERSE), schema="ohlcv-1d",
                stype_in="raw_symbol", start=RESEARCH_WINDOW_START, end=RESEARCH_WINDOW_END,
            )
            report["research_window_ohlcv_estimate_usd"][ds] = cost
        except HoldoutViolation:
            raise
        except Exception as exc:  # noqa: BLE001
            report["research_window_ohlcv_estimate_usd"][ds] = f"ERROR: {type(exc).__name__}: {exc}"

    # -- 4. definition schema cost (needed for listing-exchange / corporate- --
    # action-adjacent metadata inspection) for the same universe/window.
    report["definition_estimate_usd"] = {}
    for ds in entitled:
        try:
            _assert_safe_window(RESEARCH_WINDOW_START, RESEARCH_WINDOW_END)
            cost = client.metadata.get_cost(
                dataset=ds, symbols=list(EQUITY_UNIVERSE), schema="definition",
                stype_in="raw_symbol", start=RESEARCH_WINDOW_START, end=RESEARCH_WINDOW_END,
            )
            report["definition_estimate_usd"][ds] = cost
        except HoldoutViolation:
            raise
        except Exception as exc:  # noqa: BLE001
            report["definition_estimate_usd"][ds] = f"ERROR: {type(exc).__name__}: {exc}"

    report["capability"] = "EQUITIES_DATASET_ENTITLED"
    report["note"] = (
        "Corporate actions (dividends/splits) are NOT covered by any Databento schema "
        "probed here -- Databento is a market-data vendor, not a corporate-actions vendor. "
        "See alpha_agent.equities.corporate_actions / .earnings / .data_availability for "
        "how this universe's split/dividend/earnings-timestamp history is handled. This "
        "script performs NO acquisition -- the actual download still requires a separate, "
        "explicit, user-approved maximum-cost decision."
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
