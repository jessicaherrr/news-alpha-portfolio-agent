"""Pure, deterministic ``MarketContextFingerprint`` construction.

``build_market_context_fingerprint`` takes only already-typed values -- no
Streamlit, no network call, no implicit ``datetime.now()`` (``evaluated_at``
is always an explicit caller-supplied argument, mirroring
``alpha_agent.opportunity.schemas.OpportunityInputs.observed_at``'s own
discipline) -- so it is reproducible in a test and safe to call from a pure
retrieval pipeline. The real observation-plane READ (fetching bars/term
structure/peer confirmation) lives in ``alpha_agent.ui.context_retrieval_context``,
exactly like ``opportunity_context.py`` is the read boundary for
``opportunity.schemas.build_opportunity``.

EVENT IMPORTANCE != OBJECTIVE EVENT MAGNITUDE (semantic hardening patch,
section 3): ``importance_for_category`` returns a CATEGORY-level
classification, never a measured surprise -- see
``schemas.MarketContextFingerprint``'s own field docs. This module never
invents a magnitude by parsing ``event_category``/headline text; it only
ever reads a real, typed, structured value from
``Observation.structured_attributes`` when one is actually present (see
``_OBJECTIVE_MAGNITUDE_KEYS`` below), which is never today (no current
connector populates one) -- so ``objective_event_magnitude`` is always
``UNKNOWN_OBJECTIVE_MAGNITUDE`` in V1.
"""
from __future__ import annotations

from datetime import datetime

from alpha_agent.context_retrieval.schemas import (
    CONTEXT_FINGERPRINT_SCHEMA,
    UNKNOWN_OBJECTIVE_MAGNITUDE,
    FreshnessBucket,
    MarketContextFingerprint,
)
from alpha_agent.market_intel.importance import importance_for_category
from alpha_agent.market_intel.news_schemas import NewsCategory
from alpha_agent.validation.fingerprint import fingerprint

__all__ = ["build_market_context_fingerprint"]

_FRESH_MAX_HOURS = 6.0
_RECENT_MAX_HOURS = 72.0
_STALE_MAX_HOURS = 24.0 * 30.0

#: The closed set of REAL, TYPED structured-magnitude keys this module knows
#: how to read from ``Observation.structured_attributes`` -- never a
#: free-text scan (module docstring: "never fabricated by parsing free-text
#: headlines"). No connector or ``Observation`` builder in this repository
#: populates any of these today (``alpha_agent.translation.mechanism_library``'s
#: own ``_RELEASE_SURPRISE_VAR``/``_CONSENSUS_SURPRISE_VAR`` are both
#: ``point_in_time_available=False``); this table exists purely so a future
#: phase that DOES ingest one of these only needs to add a key here, never
#: touch the reduction logic itself. Adding a new provider is explicitly OUT
#: of scope for this patch.
_OBJECTIVE_MAGNITUDE_KEYS: tuple[str, ...] = (
    "inventory_change_surprise",
    "cpi_surprise",
    "payroll_surprise",
)


def _freshness_bucket(freshness_seconds: float) -> FreshnessBucket:
    hours = freshness_seconds / 3600.0
    if hours <= _FRESH_MAX_HOURS:
        return FreshnessBucket.FRESH
    if hours <= _RECENT_MAX_HOURS:
        return FreshnessBucket.RECENT
    if hours <= _STALE_MAX_HOURS:
        return FreshnessBucket.STALE
    return FreshnessBucket.OLD


def _category_enum(value: str) -> NewsCategory:
    try:
        return NewsCategory(value)
    except ValueError:
        return NewsCategory.OTHER


def _objective_event_magnitude(structured_attributes: dict[str, str]) -> tuple[str, str]:
    """Returns ``(objective_event_magnitude, detail)``. Honestly
    ``UNKNOWN_OBJECTIVE_MAGNITUDE`` unless one of the closed, real, typed
    keys in ``_OBJECTIVE_MAGNITUDE_KEYS`` is actually present -- never
    derived from ``event_importance`` (a category classification, not a
    magnitude) and never parsed from ``headline``/``summary`` free text."""
    for key in _OBJECTIVE_MAGNITUDE_KEYS:
        value = structured_attributes.get(key)
        if value:
            return value, f"real typed structured_attributes[{key!r}]"
    detail = (
        "No real typed structured magnitude value (e.g. an inventory-change surprise, a CPI surprise, or a "
        "payroll surprise) is currently ingested for this observation -- never derived from event_importance "
        "and never parsed from free text."
    )
    return UNKNOWN_OBJECTIVE_MAGNITUDE, detail


def build_market_context_fingerprint(
    *,
    root_symbol: str,
    event_type: str,
    event_category: str,
    affected_products: tuple[str, ...],
    observed_at: datetime,
    evaluated_at: datetime,
    trend: str,
    volatility: str,
    curve_state: str | None,
    related_market_confirming: int,
    related_market_total: int,
    structured_attributes: dict[str, str] | None = None,
) -> MarketContextFingerprint:
    """Every argument is a direct read of an already-established accessor --
    see the module docstring and ``schemas.MarketContextFingerprint``'s own
    field docs for exactly which one. ``structured_attributes`` is the
    observation's own typed attribute dict (see ``_objective_event_magnitude``);
    omitting it is equivalent to an empty dict (always
    ``UNKNOWN_OBJECTIVE_MAGNITUDE`` today). Raises ``ValueError`` on a naive
    datetime (mirrors ``alpha_agent.translation.schemas.Observation``'s own
    UTC-aware check) or ``evaluated_at`` preceding ``observed_at`` (a
    fingerprint can never be evaluated before the thing it observes)."""
    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be UTC-aware -- never a naive datetime")
    if evaluated_at.tzinfo is None:
        raise ValueError("evaluated_at must be UTC-aware -- never a naive datetime")
    if evaluated_at < observed_at:
        raise ValueError("evaluated_at cannot precede observed_at")

    freshness_seconds = (evaluated_at - observed_at).total_seconds()
    bucket = _freshness_bucket(freshness_seconds)
    importance, importance_rule = importance_for_category(_category_enum(event_category))
    objective_magnitude, objective_magnitude_detail = _objective_event_magnitude(structured_attributes or {})

    #: Only DISCRETE, reproducible fields enter the hash -- raw
    #: ``freshness_seconds`` is continuous and would make the hash different
    #: on every call; the bucket is the reproducible, still-meaningful proxy
    #: (mirrors ``alpha_agent.validation.fingerprint``'s own "cosmetic
    #: fields... are never included" rule, generalised to "continuous
    #: fields").
    payload = {
        "schema": CONTEXT_FINGERPRINT_SCHEMA,
        "root_symbol": root_symbol,
        "event_type": event_type,
        "event_category": event_category,
        "event_importance": importance.value,
        "objective_event_magnitude": objective_magnitude,
        "trend": trend,
        "volatility": volatility,
        "curve_state": curve_state or "N/A",
        "related_market_confirmation": f"{related_market_confirming}/{related_market_total}",
        "freshness_bucket": bucket.value,
    }
    fingerprint_hash = fingerprint("marketcontextfingerprint2", payload)
    component_detail = tuple(f"{k}={v}" for k, v in payload.items() if k != "schema")

    return MarketContextFingerprint(
        root_symbol=root_symbol,
        event_type=event_type,
        event_category=event_category,
        affected_products=affected_products,
        event_importance=importance.value,
        event_importance_rule=importance_rule,
        objective_event_magnitude=objective_magnitude,
        objective_event_magnitude_detail=objective_magnitude_detail,
        trend=trend,
        volatility=volatility,
        curve_state=curve_state,
        related_market_confirming=related_market_confirming,
        related_market_total=related_market_total,
        freshness_seconds=freshness_seconds,
        freshness_bucket=bucket,
        observed_at=observed_at,
        evaluated_at=evaluated_at,
        fingerprint_hash=fingerprint_hash,
        component_detail=component_detail,
    )
