"""Which ``MarketContextFingerprint`` dimensions are economically relevant
evidence for each mechanism -- a small, closed, hand-reviewed table in
exactly the same spirit as
``alpha_agent.translation.mechanism_library.CATEGORY_MECHANISM_TEMPLATES``
and ``alpha_agent.market_intel.mapping.CATEGORY_PRODUCTS``: a fixed,
inspectable rule, never an LLM judgment, and deliberately NOT exhaustive --
a mechanism absent from (or only partially covered by)
:data:`MECHANISM_CONTEXT_SIGNALS` is a real, honest gap in today's Context
Match model, never something this table papers over by guessing.

Context Match answers ONE question: of the fingerprint dimensions this
mechanism's own economic story says should matter, how many are actually
informative right now? It is never a probability, never a claim about
whether the mechanism will "work" -- see
``alpha_agent.context_retrieval.schemas.ContextMatchLevel``.
"""
from __future__ import annotations

from alpha_agent.context_retrieval.schemas import (
    ContextMatchLevel,
    FreshnessBucket,
    MarketContextFingerprint,
)
from alpha_agent.knowledge.models import EconomicMechanism

__all__ = ["MECHANISM_CONTEXT_SIGNALS", "context_match_for"]

#: mechanism -> the fingerprint dimensions its own causal story (see
#: ``mechanism_library.py``'s ``MechanismTemplate.explanation``/
#: ``causal_chain``) says are relevant evidence. Only the mechanisms today's
#: deterministic library actually proposes
#: (``mechanism_library.CATEGORY_MECHANISM_TEMPLATES``) are covered; a
#: mechanism outside that set (e.g. a future Claude-proposed mechanism) has
#: no declared signal yet and honestly scores ``ContextMatchLevel.NONE``.
MECHANISM_CONTEXT_SIGNALS: dict[EconomicMechanism, tuple[str, ...]] = {
    # A supply/demand-surprise-driven trend needs an actual observed
    # direction, and matters most while the observation is still fresh --
    # mechanism_library.py's own causal chain: "surprise -> directional
    # positioning -> price trends... over the following sessions".
    EconomicMechanism.TREND: ("trend", "freshness"),
    # Term-structure confirmation is literally about the curve's own shape;
    # trend/volatility say nothing about whether near vs. deferred contracts
    # actually moved apart.
    EconomicMechanism.TERM_STRUCTURE: ("curve_state",),
    # A macro release's signature is a step change in realized volatility
    # around the release -- mechanism_library.py: "realized volatility
    # expands sharply around the release, then decays".
    EconomicMechanism.VOLATILITY_BREAKOUT: ("volatility", "freshness"),
    # Breadth of confirmation across the mapped related-market group IS the
    # mechanism (mechanism_library.py: "breadth of joint confirmation
    # signals conviction").
    EconomicMechanism.CROSS_MARKET_LEAD_LAG: ("related_market_confirmation",),
}

_SUPPORTED_SIGNALS = frozenset({"trend", "freshness", "curve_state", "volatility", "related_market_confirmation"})


def _signal_informative(signal: str, fp: MarketContextFingerprint) -> tuple[bool, str]:
    """Returns ``(is_informative, reason)`` for one declared signal against
    the real fingerprint -- every branch states the actual observed value,
    never just true/false."""
    if signal == "trend":
        informative = fp.trend in ("Up", "Down")
        return informative, f"trend={fp.trend}"
    if signal == "freshness":
        informative = fp.freshness_bucket in (FreshnessBucket.FRESH, FreshnessBucket.RECENT)
        return informative, f"freshness={fp.freshness_bucket.value}"
    if signal == "curve_state":
        informative = fp.curve_state is not None
        return informative, f"curve_state={fp.curve_state or 'N/A'}"
    if signal == "volatility":
        informative = fp.volatility in ("High", "Moderate")
        return informative, f"volatility={fp.volatility}"
    if signal == "related_market_confirmation":
        informative = fp.related_market_total > 0
        return informative, f"related_market_confirmation={fp.related_market_confirming}/{fp.related_market_total}"
    raise ValueError(f"unknown context signal {signal!r} -- not in {sorted(_SUPPORTED_SIGNALS)}")


def context_match_for(
    mechanism: EconomicMechanism, fp: MarketContextFingerprint,
) -> tuple[ContextMatchLevel, tuple[str, ...]]:
    """Fraction of this mechanism's declared signals that are informative in
    ``fp``, bucketed into the closed :class:`ContextMatchLevel` vocabulary.
    ``reasons`` always states every declared signal's real observed value,
    whether or not it counted -- full transparency, never a hidden tally."""
    signals = MECHANISM_CONTEXT_SIGNALS.get(mechanism, ())
    if not signals:
        return ContextMatchLevel.NONE, (f"No declared context signal exists for {mechanism.value} yet.",)

    reasons: list[str] = []
    informative_count = 0
    for signal in signals:
        is_informative, reason = _signal_informative(signal, fp)
        reasons.append(reason)
        if is_informative:
            informative_count += 1

    fraction = informative_count / len(signals)
    if fraction >= 1.0:
        level = ContextMatchLevel.STRONG
    elif fraction >= 0.5:
        level = ContextMatchLevel.MODERATE
    elif fraction > 0.0:
        level = ContextMatchLevel.WEAK
    else:
        level = ContextMatchLevel.NONE
    return level, tuple(reasons)
