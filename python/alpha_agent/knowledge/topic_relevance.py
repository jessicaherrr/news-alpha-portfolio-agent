"""Topic-relevance gate for external research connectors (Release UX Part K,
task spec section 43).

`alpha_agent.knowledge.connector_support.classify_mechanism` already refuses
a source with NO mechanism-keyword signal at all -- but its own keyword list
necessarily includes bare mechanism NAMES ("mean reversion", "momentum")
that are also ordinary English/economics vocabulary outside trading (a
real-estate paper's "...Mean Reversion versus the Usual Suspects" title, the
documented false positive in `practitioner_connector.py`'s own docstring).
Requiring a mechanism-name match ALONE cannot fix that -- the false-positive
title contains a genuine mechanism keyword.

This module adds the missing, orthogonal signal: co-occurrence with an
ACTIVITY word that names systematic trading/quant research itself (task spec
section 43's list), never a mechanism name. A source must match BOTH a
mechanism keyword AND a trading-activity word to be treated as a strategy
reference -- :func:`classify_relevant_mechanism` is the one function
connectors call instead of `classify_mechanism` directly.
"""
from __future__ import annotations

import re

from alpha_agent.knowledge.connector_support import classify_mechanism
from alpha_agent.knowledge.models import EconomicMechanism

#: Task spec section 43's list, plus the obvious lexical variants a real
#: title/venue string uses for the same activity. Deliberately NOT a
#: mechanism name (those are already `connector_support.MECHANISM_KEYWORDS`)
#: -- this is "is anyone talking about SYSTEMATIC TRADING/QUANT RESEARCH at
#: all", independent of which specific mechanism.
TRADING_ACTIVITY_KEYWORDS: tuple[str, ...] = (
    "systematic trading", "futures", "trading strategy", "trading strategies", "quant", "quantitative",
    "portfolio construction", "market structure", "execution", "backtest", "backtesting", "algorithmic trading",
    "hedge fund", "cta", "managed futures", "trader", "trend following", "carry trade", "risk premia",
    "systematic strategy", "alpha strategy", "asset pricing", "factor investing",
)

_WORD_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _matches(text: str, phrase: str) -> bool:
    pat = _WORD_RE_CACHE.get(phrase)
    if pat is None:
        pat = re.compile(re.escape(phrase))
        _WORD_RE_CACHE[phrase] = pat
    return bool(pat.search(text))


def is_trading_context(text: str) -> bool:
    """True iff `text` names systematic-trading/quant-research ACTIVITY --
    never a mechanism name alone (see module docstring)."""
    if not text:
        return False
    lowered = text.lower()
    return any(_matches(lowered, kw) for kw in TRADING_ACTIVITY_KEYWORDS)


def classify_relevant_mechanism(text: str) -> EconomicMechanism | None:
    """Drop-in replacement for `connector_support.classify_mechanism`:
    returns the classified mechanism ONLY when `text` also carries a
    trading-activity signal. `None` otherwise -- callers already treat
    `classify_mechanism`'s `None` as "skip this source", so this function's
    contract is identical, just stricter."""
    mechanism = classify_mechanism(text)
    if mechanism is None:
        return None
    if not is_trading_context(text):
        return None
    return mechanism


__all__ = ["TRADING_ACTIVITY_KEYWORDS", "classify_relevant_mechanism", "is_trading_context"]
