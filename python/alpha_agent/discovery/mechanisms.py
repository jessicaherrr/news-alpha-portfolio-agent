"""Alpha Discovery campaign, Part D -- mechanism-diverse generation ordering
(task spec sections 3 / 26-29).

The default mechanism order is fixed, documented, and identical for every
run; an `InvestorProfile` may only ever REORDER it (move the user's
preferred mechanism family to the front) -- it never adds, removes, or
weights a mechanism based on anything scientific, and it can change which
hypothesis gets PROPOSED FIRST, never which one is admissible, valid, or how
it is scientifically adjudicated (task spec section 5's boundary, reused
here for generation exactly as it already applies to `alpha_agent.
recommendation`).
"""
from __future__ import annotations

from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.recommendation.profile import InvestorProfile, StrategyPreference

#: Fixed default mechanism-diversity order (task spec section 3's list,
#: closed-DSL-representable mechanisms prioritized first so a bounded budget
#: reaches something executable before spending slots on mechanisms this
#: platform cannot yet compile at all).
DEFAULT_MECHANISM_ORDER: tuple[EconomicMechanism, ...] = (
    EconomicMechanism.TREND,
    EconomicMechanism.MOMENTUM,
    EconomicMechanism.MEAN_REVERSION,
    EconomicMechanism.BREAKOUT,
    EconomicMechanism.VOLATILITY_BREAKOUT,
    EconomicMechanism.REGIME_CONDITIONED_TREND,
    EconomicMechanism.REGIME_CONDITIONED_MEAN_REVERSION,
    EconomicMechanism.OPENING_RANGE,
    EconomicMechanism.FAILED_BREAKOUT,
    EconomicMechanism.VOLATILITY_TRANSITION,
    EconomicMechanism.HYBRID_TREND_REVERSAL,
    EconomicMechanism.OVERNIGHT_GAP,
    EconomicMechanism.SESSION_EFFECTS,
    EconomicMechanism.VOLUME_LIQUIDITY,
    EconomicMechanism.MULTI_SIGNAL_ENSEMBLE,
    EconomicMechanism.CROSS_MARKET_LEAD_LAG,
    EconomicMechanism.RELATIVE_VALUE,
    EconomicMechanism.CORRELATION_SPREAD,
    EconomicMechanism.CARRY,
    EconomicMechanism.TERM_STRUCTURE,
    EconomicMechanism.ML_META_LABELING,
)

#: `InvestorProfile.strategy_preference` -> the mechanism it should bring to
#: the front. `MIXED` ("No Preference") never reorders anything.
_PREFERENCE_MECHANISM: dict[StrategyPreference, EconomicMechanism | None] = {
    StrategyPreference.TREND: EconomicMechanism.TREND,
    StrategyPreference.MEAN_REVERSION: EconomicMechanism.MEAN_REVERSION,
    StrategyPreference.BREAKOUT: EconomicMechanism.BREAKOUT,
    StrategyPreference.RELATIVE_VALUE: EconomicMechanism.RELATIVE_VALUE,
    StrategyPreference.MIXED: None,
}


def prioritize_mechanisms(
    profile: InvestorProfile | None = None,
    *,
    order: tuple[EconomicMechanism, ...] = DEFAULT_MECHANISM_ORDER,
) -> tuple[EconomicMechanism, ...]:
    """Task spec section 26/72: "user profile may affect generation priority
    [but] cannot affect scientific rules." Returns a PERMUTATION of `order` --
    same set, same length, only the profile-preferred mechanism (if any) is
    moved to the front. Never used to decide validity or verdict."""
    if profile is None:
        return order
    preferred = _PREFERENCE_MECHANISM.get(profile.strategy_preference)
    if preferred is None or preferred not in order:
        return order
    rest = tuple(m for m in order if m != preferred)
    return (preferred,) + rest
