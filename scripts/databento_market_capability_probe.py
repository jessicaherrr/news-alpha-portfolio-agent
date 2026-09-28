"""Read-only Databento CAPABILITY probe for the market-OBSERVATION plane
(mission Part D/O). Uses ONLY free Databento metadata/symbology endpoints --
``metadata.list_datasets``, ``metadata.get_dataset_range``,
``metadata.get_dataset_condition``, ``metadata.get_cost`` (an estimate call,
never a download), and ``symbology.resolve`` -- so running this script never
downloads a byte of market data and never incurs a charge on its own.

This is DELIBERATELY separate from ``alpha_agent.data.databento_source``
(the frozen, holdout-guarded scientific research pipeline): that module's
``HistoricalRequest`` refuses any window reaching 2025-01-01 by construction,
which is correct for research but would make it impossible to even ask "what
does the provider see for TODAY" for market-observation purposes. This script
(and the ``alpha_agent.marketdata.databento_provider`` module it validates)
never write to ``data/raw/``, never enter ``alpha_agent.data``/
``alpha_agent.discovery``/``alpha_agent.screening``/``alpha_agent.validation``/
``alpha_agent.registry``, and never touch ``experiment_identity``.

NEVER prints, logs, or persists ``DATABENTO_API_KEY``.

    python scripts/databento_market_capability_probe.py
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta

import _bootstrap  # noqa: F401  # adds repo python/ to sys.path; must precede alpha_agent

DATASET = "GLBX.MDP3"


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def main() -> None:
    load_dotenv_if_available()
    key = os.getenv("DATABENTO_API_KEY")
    report: dict = {"checked_at": datetime.now(UTC).isoformat()}

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

    # -- 1. list_datasets: confirms auth + entitlement, free metadata call ---
    try:
        datasets = client.metadata.list_datasets()
        report["glbx_entitled"] = DATASET in datasets
        report["n_datasets_entitled"] = len(datasets)
    except Exception as exc:  # noqa: BLE001
        status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
        report["capability"] = "RATE_LIMITED" if status == 429 else "ERROR"
        report["detail"] = f"list_datasets failed: {type(exc).__name__}: {exc}"
        print(json.dumps(report, indent=2, default=str))
        return

    # -- 2. get_dataset_range: free metadata call, no bytes downloaded -------
    try:
        rng = client.metadata.get_dataset_range(dataset=DATASET)
        report["dataset_range"] = rng
    except Exception as exc:  # noqa: BLE001
        report["dataset_range_error"] = f"{type(exc).__name__}: {exc}"
        rng = {}

    # -- 3. get_dataset_condition for the last few days: free metadata call --
    try:
        today = datetime.now(UTC).date()
        cond = client.metadata.get_dataset_condition(
            dataset=DATASET,
            start_date=(today - timedelta(days=7)).isoformat(),
            end_date=today.isoformat(),
        )
        report["recent_condition"] = cond
    except Exception as exc:  # noqa: BLE001
        report["recent_condition_error"] = f"{type(exc).__name__}: {exc}"

    # -- 4. symbology.resolve for NQ continuous front month: free metadata --
    try:
        today = datetime.now(UTC).date()
        resolved = client.symbology.resolve(
            dataset=DATASET, symbols=["NQ.v.0"], stype_in="continuous", stype_out="raw_symbol",
            start_date=(today - timedelta(days=5)).isoformat(), end_date=today.isoformat(),
        )
        report["nq_front_month_resolution"] = resolved
    except Exception as exc:  # noqa: BLE001
        report["nq_front_month_resolution_error"] = f"{type(exc).__name__}: {exc}"

    # -- 5. get_cost for a TINY hypothetical recent OHLCV request -- an
    # ESTIMATE only, never a download, never a charge. -----------------------
    try:
        end = datetime.now(UTC)
        start = end - timedelta(hours=6)
        cost = client.metadata.get_cost(
            dataset=DATASET, symbols=["NQ.v.0"], schema="ohlcv-1m", stype_in="continuous",
            start=start.isoformat(), end=end.isoformat(),
        )
        report["tiny_ohlcv_estimate_usd"] = cost
    except Exception as exc:  # noqa: BLE001
        report["tiny_ohlcv_estimate_error"] = f"{type(exc).__name__}: {exc}"

    # -- capability classification (honest, never fabricated) ---------------
    end_raw = None
    if isinstance(rng, dict):
        end_raw = rng.get("end") or (rng.get("value") or {}).get("end") if isinstance(rng.get("value"), dict) else rng.get("end")
    latest = None
    if end_raw:
        try:
            latest = datetime.fromisoformat(str(end_raw))
        except ValueError:
            latest = None
    if latest is not None:
        lag = (datetime.now(UTC) - latest).total_seconds()
        report["latest_available_ts"] = latest.isoformat()
        report["lag_seconds"] = lag
        report["capability"] = "LATEST_AVAILABLE" if lag <= 24 * 3600 else "HISTORICAL_ONLY"
    else:
        report["capability"] = "HISTORICAL_ONLY"
    report["note"] = (
        "No Live/streaming gateway was probed (a separate, subscription-gated "
        "entitlement) -- LIVE/DELAYED are never claimed without proving them."
    )

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
