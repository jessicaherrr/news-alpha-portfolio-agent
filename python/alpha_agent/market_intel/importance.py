"""Deterministic SCHEDULED EVENT importance -- Section 25. A fixed,
inspectable rule table; no LLM ever invents an importance level. Every
`ScheduledMarketEvent` stores the exact `importance_rule` id this module
assigned it.
"""
from __future__ import annotations

from alpha_agent.market_intel.event_schemas import EventImportance

#: Section 25's worked HIGH-importance categories, verbatim.
_HIGH_IMPORTANCE_RULES: dict[str, str] = {
    "FOMC_POLICY": "fomc-policy-decision-always-high/1",
    "CPI_PPI_EMPLOYMENT": "cpi-ppi-employment-release-always-high/1",
    "PETROLEUM": "eia-weekly-petroleum-status-high-for-energy-roots/1",
    "NATURAL_GAS": "eia-natural-gas-storage-high-for-ng/1",
    "USDA_GRAIN_OILSEED": "usda-wasde-high-for-mapped-agriculture-roots/1",
}

_DEFAULT_RULE = "unmapped-category-default-medium/1"


def importance_for_category(category: str) -> tuple[EventImportance, str]:
    """Returns ``(importance, importance_rule)`` -- HIGH only for Section
    25's explicit worked categories; everything else defaults to MEDIUM
    under one documented default rule (never LOW-by-omission, and never an
    LLM judgment call)."""
    if category in _HIGH_IMPORTANCE_RULES:
        return EventImportance.HIGH, _HIGH_IMPORTANCE_RULES[category]
    return EventImportance.MEDIUM, _DEFAULT_RULE


__all__ = ["importance_for_category"]
