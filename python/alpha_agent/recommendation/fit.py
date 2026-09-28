"""Phase B1 (hardened -- Alpha Discovery campaign, Part A) -- the
deterministic User Fit Score.

Measures how closely a candidate's MEASURED historical characteristics match
the user's stated `InvestorProfile` (task spec section 11). It does NOT
measure expected future return, scientific truth, or probability of PASS --
`alpha_agent.recommendation.promise.score_research_promise` is the (fully
independent, profile-free) Research Promise Score, and `RegistryVerdict`
remains the only scientific outcome. Changing the profile can change a
candidate's User Fit Score; it can never change its Research Promise Score or
its scientific verdict (see the Phase B1 tests).

Honesty about evidence (task spec section 11): today's committed
`ResultRecord` does NOT carry authoritative drawdown, trade-duration/holding-
period, or overnight-exposure evidence for any candidate. This module never
fabricates those dimensions. Only two dimensions are ever scored because only
two are backed by real committed evidence:

* **Strategy mechanism preference** -- an exact, documented
  `strategy_family -> mechanism category` mapping (measured: which family the
  committed experiment actually is).
* **Trading frequency compatibility** -- `n_trades / (n_oos_days / 252)`
  (trades per year), computed from the committed `ResultRecord.n_trades` /
  `n_oos_days` -- real measured evidence, not an inference from the family
  name.

Every other conceptual dimension the task spec lists (overnight, holding
period, drawdown, turnover, capital/contract practicality) is reported in
`evidence_coverage` as explicitly unavailable rather than scored -- "Fit
based on available evidence" (task spec section 11's own phrase), never a
silently neutral weight for a dimension that was actually never measured.

Hardening (this campaign's Part A, sections 6/7 -- fixes a real bug in the
original Phase B1 scoring): "No Preference" (mechanism: `MIXED`; trading
frequency: `NO_PREFERENCE`) is a statement that the dimension carries NO
SCORING CONSTRAINT, not a claim of perfect compatibility. The previous
implementation gave such a dimension full credit (`score = 1.0`) AND counted
it in the score denominator, so the untouched `DEFAULT_PROFILE` (which is
all "no preference" on the two scored dimensions) always produced `total ==
100.0` / `label == "HIGH"` -- a personalization score for a profile that
states no preference at all. A "no preference" dimension is now EXCLUDED
from both the weighted numerator and the denominator entirely (never scored,
never counted as measured). Three explicit `PersonalizationState` values
replace the previous "always show a percentage" behaviour:

* `NOT_PERSONALIZED` -- no scoreable dimension carries an actual stated
  preference (e.g. the untouched default profile). The UI must show
  "Not personalized", never a percentage or a HIGH/MEDIUM/LOW label -- see
  `alpha_agent.ui.recommendation_views.format_user_fit`.
* `PARTIAL` -- at least one dimension has a real stated preference, but not
  every such dimension has committed evidence to measure it against (e.g. the
  user set a trading-frequency preference but the candidate's `n_trades`/
  `n_oos_days` are missing).
* `MEASURED` -- every dimension that has a real stated preference also has
  committed evidence, so the reported percentage reflects full coverage of
  what the user actually asked to be measured on.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel

from alpha_agent.recommendation.profile import (
    HoldingPeriod,
    InvestorProfile,
    OvernightPreference,
    StrategyPreference,
    TradingFrequency,
)

# ---------------------------------------------------------------------------
# strategy-mechanism category mapping -- documented display/scoring metadata,
# never a measured behavioural statistic. A family absent from this table
# (e.g. `silver_bullet`, a session/time-based ICT mechanism that does not map
# cleanly onto the DSL's Trend/Mean-Reversion/Breakout/Relative-Value
# categories) is scored NEUTRAL, never guessed into the wrong bucket.
# ---------------------------------------------------------------------------

_FAMILY_MECHANISM: dict[str, str] = {
    "tsmom": "trend",
    "ma_trend": "trend",
    "breakout": "breakout",
    "mean_reversion": "mean_reversion",
}
_ML_META_LABEL_PREFIX = "ml_meta_label."

_PREFERENCE_MECHANISM: dict[StrategyPreference, str | None] = {
    StrategyPreference.TREND: "trend",
    StrategyPreference.MEAN_REVERSION: "mean_reversion",
    StrategyPreference.BREAKOUT: "breakout",
    StrategyPreference.RELATIVE_VALUE: "relative_value",
    StrategyPreference.MIXED: None,  # "Mixed / No Preference" -- always full credit
}

WEIGHT_MECHANISM = 60.0
WEIGHT_TRADING_FREQUENCY = 40.0

#: Trading-frequency reference bands (fixed, documented heuristic -- trades
#: per year, not derived from the current registry's own distribution).
TRADES_PER_YEAR_LOW_MAX = 15.0
TRADES_PER_YEAR_HIGH_MIN = 40.0
_TRADING_DAYS_PER_YEAR = 252.0

HIGH_THRESHOLD = 80.0
MEDIUM_THRESHOLD = 50.0


class PersonalizationState(str, Enum):
    """Whether `UserFitBreakdown.total` reflects any actual stated
    preference at all (section 6). The UI branches on this, never on
    `total`/`label` alone -- see `alpha_agent.ui.recommendation_views.
    format_user_fit`."""

    NOT_PERSONALIZED = "NOT_PERSONALIZED"
    PARTIAL = "PARTIAL"
    MEASURED = "MEASURED"


def _mechanism_for_family(family: str | None) -> str | None:
    if not family:
        return None
    if family in _FAMILY_MECHANISM:
        return _FAMILY_MECHANISM[family]
    if family.startswith(_ML_META_LABEL_PREFIX):
        return _FAMILY_MECHANISM.get(family[len(_ML_META_LABEL_PREFIX) :])
    return None


def frequency_band(trades_per_year: float) -> str:
    """The `TradingFrequency` band (its ``value``) a trades-per-year rate falls
    in -- the one band definition shared by every User Fit that scores trading
    frequency (strategies here, candidate signals in
    `alpha_agent.recommendation.signal_ranking`)."""
    if trades_per_year <= TRADES_PER_YEAR_LOW_MAX:
        return TradingFrequency.LOW.value
    if trades_per_year < TRADES_PER_YEAR_HIGH_MIN:
        return TradingFrequency.MEDIUM.value
    return TradingFrequency.HIGH.value


def _label(total: float) -> str:
    if total >= HIGH_THRESHOLD:
        return "HIGH"
    if total >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


class UserFitBreakdown(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    mechanism: float | None
    trading_frequency: float | None
    total: float
    label: str
    evidence_coverage: dict[str, str]

    #: Section 6. Drives whether the UI may show a percentage/label at all --
    #: `NOT_PERSONALIZED` means `total`/`label` must NOT be rendered (they are
    #: `0.0`/`"LOW"` only because there was nothing to score, never a real
    #: measurement of poor fit).
    personalization_state: PersonalizationState

    #: Section 7. How many of the (at most two) scoreable dimensions actually
    #: carried both a stated preference AND committed evidence -- "Based on N
    #: measurable dimensions" in the UI.
    measurable_dimension_count: int


def score_user_fit(
    *, strategy_family: str | None, result: dict[str, Any] | None, profile: InvestorProfile
) -> UserFitBreakdown:
    """Score one candidate against `profile`. `result` is the same
    `ResultRecord`-shaped dict `score_research_promise` accepts (only
    `n_trades` / `n_oos_days` are read from it).

    A "no preference" dimension (mechanism: `StrategyPreference.MIXED`;
    frequency: `TradingFrequency.NO_PREFERENCE`) is excluded entirely from the
    score -- it contributes to neither the weighted numerator nor the
    denominator (section 6). Only a dimension with an actual stated
    preference can ever be "measured".
    """
    coverage: dict[str, str] = {}
    weighted_score = 0.0
    weight_total = 0.0
    meaningful_dims = 0
    measured_dims = 0

    # -- strategy mechanism preference ---------------------------------------
    preferred_mechanism = _PREFERENCE_MECHANISM.get(profile.strategy_preference)
    mechanism_component: float | None = None
    if preferred_mechanism is None:
        coverage["strategy_mechanism"] = "no preference set -- excluded from Fit score"
    else:
        meaningful_dims += 1
        mechanism = _mechanism_for_family(strategy_family)
        if mechanism is None:
            mechanism_score = 0.5  # mechanism category not established for this family -- neutral, never guessed
            coverage["strategy_mechanism"] = (
                "partial -- this family has no documented mechanism-category mapping; scored neutrally"
            )
        else:
            mechanism_score = 1.0 if mechanism == preferred_mechanism else 0.0
            coverage["strategy_mechanism"] = "available (category metadata, not a measured trade-path statistic)"
        weighted_score += mechanism_score * WEIGHT_MECHANISM
        weight_total += WEIGHT_MECHANISM
        mechanism_component = round(mechanism_score * WEIGHT_MECHANISM, 2)
        measured_dims += 1

    # -- trading frequency compatibility --------------------------------------
    frequency_component: float | None = None
    if profile.trading_frequency == TradingFrequency.NO_PREFERENCE:
        coverage["trading_frequency"] = "no preference set -- excluded from Fit score"
    else:
        meaningful_dims += 1
        n_trades = (result or {}).get("n_trades")
        n_oos_days = (result or {}).get("n_oos_days")
        if n_trades is not None and n_oos_days:
            trades_per_year = n_trades / (n_oos_days / _TRADING_DAYS_PER_YEAR)
            band = frequency_band(trades_per_year)
            frequency_score = 1.0 if band == profile.trading_frequency.value else 0.0
            weighted_score += frequency_score * WEIGHT_TRADING_FREQUENCY
            weight_total += WEIGHT_TRADING_FREQUENCY
            frequency_component = round(frequency_score * WEIGHT_TRADING_FREQUENCY, 2)
            coverage["trading_frequency"] = "available (measured: trades / OOS-year)"
            measured_dims += 1
        else:
            coverage["trading_frequency"] = (
                "saved preference, evidence not available -- missing committed trade count or OOS day count"
            )

    # -- dimensions the task spec lists but this module never scores ---------
    # (no committed `ResultRecord` field backs any of these today -- section
    # 11/section 8). Still distinguish "user asked, we can't measure it" from
    # "user never asked" so the UI's evidence-coverage panel (section 7) is
    # honest either way.
    coverage["overnight_positions"] = (
        "saved preference, evidence not available -- no committed intraday/overnight holding evidence"
        if profile.overnight != OvernightPreference.NO_PREFERENCE
        else "no preference set"
    )
    coverage["holding_period"] = (
        "saved preference, evidence not available -- no committed trade-duration evidence"
        if profile.holding_period != HoldingPeriod.FLEXIBLE
        else "no preference set"
    )
    coverage["drawdown"] = "unavailable -- no committed drawdown artifact"

    if meaningful_dims == 0:
        state = PersonalizationState.NOT_PERSONALIZED
    elif measured_dims == meaningful_dims:
        state = PersonalizationState.MEASURED
    else:
        state = PersonalizationState.PARTIAL

    total = round((weighted_score / weight_total) * 100.0, 1) if weight_total > 0 else 0.0

    return UserFitBreakdown(
        mechanism=mechanism_component,
        trading_frequency=frequency_component,
        total=total,
        label=_label(total),
        evidence_coverage=coverage,
        personalization_state=state,
        measurable_dimension_count=measured_dims,
    )
