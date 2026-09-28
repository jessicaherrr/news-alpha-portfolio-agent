"""Tests for `alpha_agent.knowledge.topic_relevance` (Release UX Part K,
task spec section 43).
"""
from __future__ import annotations

from alpha_agent.knowledge.connector_support import classify_mechanism
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.knowledge.topic_relevance import classify_relevant_mechanism, is_trading_context


def test_documented_real_estate_false_positive_is_now_rejected():
    """The exact case `practitioner_connector.py` used to document as a
    known limitation: a real-estate paper's title contains the MEAN_REVERSION
    keyword but names no trading activity."""
    text = "Mean Reversion versus the Usual Suspects: Rent Dynamics in Metro Housing Markets"
    assert classify_mechanism(text) == EconomicMechanism.MEAN_REVERSION  # the raw keyword still matches
    assert classify_relevant_mechanism(text) is None  # but the relevance gate correctly excludes it


def test_genuine_institutional_futures_research_still_passes():
    text = "Momentum Strategy Performance in Commodity Futures Trading (Federal Reserve Bank of San Francisco)"
    assert classify_mechanism(text) == EconomicMechanism.MOMENTUM
    assert classify_relevant_mechanism(text) == EconomicMechanism.MOMENTUM


def test_genuine_trading_strategy_post_still_passes():
    text = "Simple Mean Reversion Trading Strategy in Python"
    assert classify_relevant_mechanism(text) == EconomicMechanism.MEAN_REVERSION


def test_no_mechanism_at_all_is_still_none():
    assert classify_relevant_mechanism("Show HN: my new budgeting app") is None


def test_mechanism_without_any_activity_context_is_excluded():
    # Matches the MOMENTUM keyword "momentum factor" but names no trading activity.
    text = "Momentum factor performance in cross-sectional equity returns"
    assert classify_mechanism(text) == EconomicMechanism.MOMENTUM
    assert classify_relevant_mechanism(text) is None


def test_is_trading_context_requires_an_activity_word_not_a_mechanism_name():
    assert is_trading_context("a systematic trading strategy for futures") is True
    assert is_trading_context("mean reversion in housing prices") is False
    assert is_trading_context("") is False


def test_relevant_classifier_never_more_permissive_than_the_base_classifier():
    """Structural: classify_relevant_mechanism can only NARROW, never widen,
    what classify_mechanism already accepts."""
    samples = [
        "Time Series Momentum Trading Strategy",
        "A hedge fund's carry trade approach to futures",
        "Random unrelated text about gardening",
        "Volatility breakout systematic strategy for CTAs",
        "Portfolio construction and factor investing in quant funds",
    ]
    for text in samples:
        base = classify_mechanism(text)
        strict = classify_relevant_mechanism(text)
        if strict is not None:
            assert strict == base
