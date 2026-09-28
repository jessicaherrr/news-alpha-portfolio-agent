"""Example Phase 22 crypto hypotheses, reusing :class:`HypothesisSpec` unchanged.

Prompt 22: "Reuse the same HypothesisSpec, StrategySpec, deterministic
backtest, validation, registry, and risk principles." Nothing in
:class:`alpha_agent.schemas.hypothesis.HypothesisSpec` is futures-specific, so
these are ordinary instances of the exact same frozen Phase 16 type -- proof
that the schema needed no crypto-specific extension.

Every hypothesis here trades a synthetic CME-futures-shaped execution fixture
(never a perpetual, never a spot/on-chain venue) using crypto-derivatives /
on-chain features as SIGNAL INPUTS only -- CLAUDE.md's "do not mix crypto
venue assumptions with CME futures execution semantics" resolved by
construction: the executed instrument's Fill/roll/risk/accounting path is the
ordinary, unmodified C++ Quant Core used by every other root in this
repository, run today over SYNTHETIC market data, never real CME BTC/ETH
history (Phase 22.1 presentation-semantics fix -- see
docs/CRYPTO_ONCHAIN_EXTENSION.md).
"""
from __future__ import annotations

from alpha_agent.schemas.hypothesis import HypothesisSpec

FUNDING_CROWDING_CONTRARIAN = HypothesisSpec(
    hypothesis_id="crypto.funding_crowding_contrarian.v1",
    title="Perp funding-rate crowding predicts CME BTC future mean reversion",
    economic_mechanism=(
        "A persistently extreme perpetual funding rate reflects one-sided "
        "leveraged positioning (crowded longs paying shorts, or vice versa). "
        "That positioning imbalance is a cross-venue crowding signal: when it "
        "is extreme, the crowded side is more likely to be forced to unwind, "
        "producing short-horizon mean reversion in the CME-futures-shaped "
        "instrument this hypothesis targets -- a cheaper, regulated way to "
        "express the resulting view without ever touching the perpetual "
        "venue itself. (Phase 22 tests this only against a synthetic "
        "execution fixture; it is not a claim about real CME BTC futures.)"
    ),
    universe=["BTC"],
    horizon="1-10 trading days",
    required_features=["funding_rate_zscore"],
    signal_description=(
        "Rolling z-score of the perpetual funding rate; go short the CME "
        "future when the z-score is extremely positive (crowded longs), long "
        "when extremely negative (crowded shorts), flat otherwise."
    ),
    expected_regime="range-bound / mean-reverting funding regimes",
    failure_regime="a sustained one-directional trend where funding stays "
    "extreme without reverting (the crowding never unwinds)",
    falsification_test=(
        "predeclared parameter neighbourhood over the z-score window + "
        "threshold; a time-shift and block-bootstrap null on the daily PnL "
        "series; cost-stress at 2x commission"
    ),
    novelty_notes="Phase 22 synthetic scaffold example -- SYNTHETIC evidence only, "
    "not a claim about real funding-rate data.",
    evidence_level="experimental",
)

ONCHAIN_MVRV_MEAN_REVERSION = HypothesisSpec(
    hypothesis_id="crypto.onchain_mvrv_mean_reversion.v1",
    title="On-chain MVRV extremes predict CME BTC future mean reversion",
    economic_mechanism=(
        "MVRV (market value to realized value) extremes indicate the average "
        "holder is deeply in profit or loss; large aggregate unrealized "
        "gains/losses historically precede profit-taking or capitulation-driven "
        "reversals. The CME future is used to express the resulting view."
    ),
    universe=["BTC"],
    horizon="5-20 trading days",
    required_features=["mvrv_zscore"],
    signal_description=(
        "Rolling z-score of the on-chain MVRV ratio; fade extreme readings on "
        "the CME future."
    ),
    expected_regime="post-cycle-extreme reversal regimes",
    failure_regime="a structural, sustained on-chain re-rating (MVRV drifts to a "
    "new persistent level rather than reverting)",
    falsification_test=(
        "predeclared parameter neighbourhood over the z-score window; null / "
        "bootstrap / cost-stress identical to the funding-crowding hypothesis"
    ),
    novelty_notes="Phase 22 synthetic scaffold example -- SYNTHETIC evidence only.",
    evidence_level="experimental",
)

CRYPTO_HYPOTHESES: tuple[HypothesisSpec, ...] = (
    FUNDING_CROWDING_CONTRARIAN,
    ONCHAIN_MVRV_MEAN_REVERSION,
)
