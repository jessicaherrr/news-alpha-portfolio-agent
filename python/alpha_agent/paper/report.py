"""Phase 21 / 21.1 -- a read-only JSON report for one paper run, built entirely
from committed ledger rows. Used by the CLI script (for a human-readable
artifact under ``outputs/phase_21/``) and by the Streamlit UI's read-only
services boundary. Computes nothing new -- every field is copied from
:class:`~alpha_agent.paper.ledger.PaperLedger` rows or their typed row models.
"""
from __future__ import annotations

from typing import Any

from alpha_agent.paper.ledger import PaperLedger


def build_paper_run_report(ledger: PaperLedger, run_id: str) -> dict[str, Any]:
    run = ledger.get_run(run_id)
    steps = ledger.list_steps(run_id)
    fills = ledger.list_fills(run_id)
    alerts = ledger.list_alerts(run_id)
    latest = steps[-1] if steps else None
    #: the authoritative C++ position snapshot for the LATEST step only --
    #: "current holdings" for display. Full per-step history is available via
    #: PaperLedger.list_positions(run_id) (every step, for audit) but is not
    #: bundled here by default to keep this report proportionate to a
    #: typical run's step count.
    latest_positions = ledger.list_positions(run_id, latest.step_ordinal) if latest else ()

    return {
        "schema_version": "paper-run-report/2",
        "run": run.model_dump(mode="json"),
        "n_steps": len(steps),
        "n_fills": len(fills),
        "n_alerts": len(alerts),
        "latest_step": latest.model_dump(mode="json") if latest else None,
        "latest_positions": [p.model_dump(mode="json") for p in latest_positions],
        "steps": [s.model_dump(mode="json") for s in steps],
        "fills": [f.model_dump(mode="json") for f in fills],
        "alerts": [a.model_dump(mode="json") for a in alerts],
    }
