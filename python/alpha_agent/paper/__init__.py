"""Phase 21 -- paper trading and drift monitor.

    ExperimentRegistry (PASS only) -> StrategySpec rebuild + fingerprint check
        -> target schedule (frozen Phase 09-11 machinery)
        -> C++ quant_paper_trading_targets_csv (hard PortfolioRiskManager)
        -> persistent paper ledger -> drift diagnostics -> alerts

No live-money broker routing. No competing execution/accounting
implementation: every fill, PnL, and risk decision is read verbatim from the
C++ ``BacktestResult`` JSON. See ``docs/PAPER_TRADING_AND_DRIFT_MONITOR.md``.
"""
from __future__ import annotations

from alpha_agent.paper.alerts import Alert, AlertSeverity, AlertType
from alpha_agent.paper.drift import DriftReport
from alpha_agent.paper.eligibility import PaperEligibility, assert_paper_trading_eligible
from alpha_agent.paper.engine import PaperTradingEngine, StepSnapshot
from alpha_agent.paper.errors import (
    PaperDataWindowExhausted,
    PaperReplayDivergence,
    PaperRunStateError,
    PaperTradingEligibilityError,
    PaperTradingError,
)
from alpha_agent.paper.ledger import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_LEDGER_PATH,
    LEDGER_SCHEMA_LABEL,
    LEGACY_DRIFT,
    LEGACY_PROVENANCE,
    PaperAlertRow,
    PaperFillRow,
    PaperLedger,
    PaperPositionRow,
    PaperRunRow,
    PaperStepRow,
    is_legacy_evidence,
)
from alpha_agent.paper.provenance import PaperStepProvenance
from alpha_agent.paper.risk_policy import PaperRiskPolicy

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "DEFAULT_LEDGER_PATH",
    "LEDGER_SCHEMA_LABEL",
    "LEGACY_DRIFT",
    "LEGACY_PROVENANCE",
    "Alert",
    "AlertSeverity",
    "AlertType",
    "DriftReport",
    "PaperAlertRow",
    "PaperDataWindowExhausted",
    "PaperEligibility",
    "PaperFillRow",
    "PaperLedger",
    "PaperPositionRow",
    "PaperReplayDivergence",
    "PaperRiskPolicy",
    "PaperRunRow",
    "PaperRunStateError",
    "PaperStepProvenance",
    "PaperStepRow",
    "PaperTradingEligibilityError",
    "PaperTradingEngine",
    "PaperTradingError",
    "StepSnapshot",
    "assert_paper_trading_eligible",
    "is_legacy_evidence",
]
