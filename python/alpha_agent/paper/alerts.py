"""Phase 21 -- typed paper-trading alerts.

Every alert is a label on state the C++ engine ALREADY computed (a risk
decision, a drift statistic, a data-window boundary) -- this module never
makes a risk or execution decision itself; ``kill_switch_active`` in
particular only restates, for display/alerting, the same threshold comparison
against the SAME ``portfolio_at_end`` numbers the mandatory C++ risk gate
already enforced before any order using this step's data was ever admitted or
blocked (cpp/include/quant_core/paper_trading_run.hpp).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AlertType(str, Enum):
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"
    KILL_SWITCH_CLEARED = "KILL_SWITCH_CLEARED"
    DRIFT_WARNING = "DRIFT_WARNING"
    DATA_WINDOW_EXHAUSTED = "DATA_WINDOW_EXHAUSTED"
    MARGIN_DATA_INCOMPLETE = "MARGIN_DATA_INCOMPLETE"
    RUN_STARTED = "RUN_STARTED"
    RUN_STOPPED = "RUN_STOPPED"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Alert(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    alert_type: AlertType
    severity: AlertSeverity
    message: str
    detail: dict = Field(default_factory=dict)


def kill_switch_active(risk_policy: dict, portfolio_at_end: dict) -> bool:
    """Restate, for display, the SAME threshold comparison the C++ hard-risk
    gate already enforced against ``portfolio_at_end`` (verbatim from the CLI
    JSON) using the SAME ``risk_policy`` this run was configured with. This
    never blocks or approves an order itself -- that already happened inside
    the mandatory C++ risk gate; it only labels the resulting state."""
    daily_loss_limit = float(risk_policy.get("max_daily_loss_usd", 0.0))
    if daily_loss_limit > 0.0 and float(portfolio_at_end.get("day_realized_pnl_usd", 0.0)) <= -daily_loss_limit:
        return True
    drawdown_pct_limit = float(risk_policy.get("max_drawdown_pct", 0.0))
    if drawdown_pct_limit > 0.0 and float(portfolio_at_end.get("portfolio_drawdown_pct", 0.0)) >= drawdown_pct_limit:
        return True
    drawdown_usd_limit = float(risk_policy.get("max_drawdown_usd", 0.0))
    return (
        drawdown_usd_limit > 0.0
        and float(portfolio_at_end.get("portfolio_drawdown_usd", 0.0)) >= drawdown_usd_limit
    )


def generate_step_alerts(
    *,
    kill_switch_now: bool,
    kill_switch_before: bool,
    margin_complete: bool,
    window_exhausted: bool,
    drift_flags: tuple[str, ...] = (),
) -> list[Alert]:
    """State-TRANSITION alerts only (never re-alert every step on a state that
    has not changed, so the alert log stays a meaningful timeline, not spam)."""
    alerts: list[Alert] = []
    if kill_switch_now and not kill_switch_before:
        alerts.append(
            Alert(
                alert_type=AlertType.KILL_SWITCH_ACTIVE,
                severity=AlertSeverity.CRITICAL,
                message=(
                    "A deterministic hard-risk kill switch (drawdown or daily-loss "
                    "limit) has tripped; the C++ risk gate is now refusing every "
                    "risk-increasing order for this run. Existing positions are "
                    "never force-closed by a kill switch."
                ),
            )
        )
    elif kill_switch_before and not kill_switch_now:
        alerts.append(
            Alert(
                alert_type=AlertType.KILL_SWITCH_CLEARED,
                severity=AlertSeverity.INFO,
                message="The hard-risk kill switch condition has cleared for this run.",
            )
        )
    if not margin_complete:
        alerts.append(
            Alert(
                alert_type=AlertType.MARGIN_DATA_INCOMPLETE,
                severity=AlertSeverity.INFO,
                message=(
                    "No committed per-root margin schedule is wired to this run; "
                    "margin utilization is not enforced (see PaperRiskPolicy docstring). "
                    "This is a visible data gap, not a risk-free portfolio."
                ),
            )
        )
    if window_exhausted:
        alerts.append(
            Alert(
                alert_type=AlertType.DATA_WINDOW_EXHAUSTED,
                severity=AlertSeverity.WARNING,
                message=(
                    "This run's predeclared replay window has no further trading "
                    "days; it cannot advance further without a new run over a new "
                    "window."
                ),
            )
        )
    for flag in drift_flags:
        alerts.append(
            Alert(
                alert_type=AlertType.DRIFT_WARNING,
                severity=AlertSeverity.WARNING,
                message=f"Live/paper vs backtest drift flag: {flag}",
                detail={"flag": flag},
            )
        )
    return alerts
