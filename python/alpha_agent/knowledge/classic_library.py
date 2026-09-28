"""Alpha Discovery campaign, Part C -- the curated classic strategy library
(task spec section 16).

Every item here is CLASSIC: a historically established systematic-trading
research idea, documented as such. "Classic" never means "profitable" -- none
of these carry a `reported_results` claim, because there is no single
canonical backtest to attribute one to; each is independently proposed,
compiled, and tested by this platform's own pipeline like any other
hypothesis (task spec section 16's closing line).

Deterministic and offline: no network call, no external file read. Seeding
this library is itself the "implement the architecture + fixtures" fallback
task spec section 23 asks for when live ingestion is unavailable (see
`alpha_agent.knowledge.external_sources`).
"""
from __future__ import annotations

import hashlib
import json

from alpha_agent.knowledge.models import (
    EconomicMechanism,
    IngestionStatus,
    SourceQualityTier,
    SourceType,
    StrategyKnowledgeItem,
)


def _hash(item_id: str, payload: dict) -> str:
    canon = json.dumps({"id": item_id, **payload}, sort_keys=True, separators=(",", ":"))
    return "classic1:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:32]


def _classic(
    knowledge_id: str,
    *,
    title: str,
    mechanism: EconomicMechanism,
    time_horizon: str,
    entry: str,
    exit_: str,
    risk: str,
    params: str,
    template: str | None,
    required_features: tuple[str, ...] = (),
    limitations: str = "",
    notes: str = "",
) -> StrategyKnowledgeItem:
    payload = {
        "title": title, "mechanism": mechanism.value, "entry": entry, "exit": exit_,
    }
    return StrategyKnowledgeItem(
        knowledge_id=knowledge_id,
        source_type=SourceType.CLASSIC,
        source_quality=SourceQualityTier.TIER_B,
        ingestion_status=IngestionStatus.SEEDED,
        title=title,
        markets=("ES", "NQ", "CL", "GC", "ZN"),
        asset_classes=("futures",),
        time_horizon=time_horizon,
        economic_mechanism=mechanism,
        required_features=required_features,
        entry_logic_summary=entry,
        exit_logic_summary=exit_,
        risk_logic_summary=risk,
        parameter_summary=params,
        reported_results=None,
        implementation_notes=notes,
        limitations=limitations or "Historically established idea, not a validated result on this platform.",
        internal_supported_capabilities=(
            f"maps to the existing '{template}' Phase 11 template family" if template
            else "no existing Phase 11 template; would need a full closed-DSL blueprint"
        ),
        candidate_dsl_template=template,
        provenance_hash=_hash(knowledge_id, payload),
    )


def build_classic_library() -> tuple[StrategyKnowledgeItem, ...]:
    """Deterministic, fixed ordering -- callers that need a stable index
    (e.g. `dedup`) may rely on ordering being reproducible run to run."""
    return (
        _classic(
            "classic-dual-moving-average",
            title="Dual Moving Average", mechanism=EconomicMechanism.TREND,
            time_horizon="days-to-weeks",
            entry="Go long when the fast SMA is above the slow SMA, short when below.",
            exit_="Signal reversal (fast SMA crosses back).",
            risk="No embedded stop; position sizing external to the signal.",
            params="fast_window, slow_window (e.g. 20/50, 50/200).",
            template="ma_trend", required_features=("sma", "sma"),
            limitations="Whipsaws in range-bound / choppy regimes.",
        ),
        _classic(
            "classic-ma-crossover",
            title="Moving-Average Crossover (single fast/slow pair)",
            mechanism=EconomicMechanism.TREND, time_horizon="days-to-weeks",
            entry="Cross-above/cross-below of one fast MA against one slow MA.",
            exit_="Opposite cross.", risk="None embedded.",
            params="fast_window, slow_window.", template="ma_trend",
            required_features=("sma",),
            notes="A restricted special case of Dual Moving Average; kept as its own "
            "reference because it is the most commonly cited standalone idea.",
        ),
        _classic(
            "classic-time-series-momentum",
            title="Time-Series Momentum (TSMOM)",
            mechanism=EconomicMechanism.MOMENTUM, time_horizon="weeks-to-months",
            entry="Long if trailing N-period return is positive, short if negative.",
            exit_="Signal reversal at the next rebalance.",
            risk="Often paired with volatility-target sizing in the literature.",
            params="lookback_horizon (e.g. 90-252 trading days).",
            template="tsmom", required_features=("rolling_return",),
            limitations="Well-documented crowding / mean-reversion-after-momentum risk in some regimes.",
        ),
        _classic(
            "classic-donchian-trend",
            title="Donchian / Price Channel Trend",
            mechanism=EconomicMechanism.TREND, time_horizon="weeks-to-months",
            entry="Long on a new N-day high, short on a new N-day low.",
            exit_="Opposite channel breach, or a shorter exit channel (dual-channel Turtle style).",
            risk="Classic Turtle-style ATR-based position sizing in the original rules.",
            params="entry_channel_days, exit_channel_days.",
            template="ma_trend", required_features=("rolling_high", "rolling_low"),
        ),
        _classic(
            "classic-donchian-breakout",
            title="Donchian Breakout",
            mechanism=EconomicMechanism.BREAKOUT, time_horizon="days-to-weeks",
            entry="Enter on a close beyond the N-day high/low channel.",
            exit_="Time stop or opposite channel breach.",
            risk="Typically paired with an ATR stop in practitioner writeups.",
            params="channel_days.", template="breakout",
            required_features=("rolling_high", "rolling_low"),
            limitations="Prone to false breakouts in low-volume / low-volatility conditions.",
        ),
        _classic(
            "classic-atr-volatility-breakout",
            title="ATR / Volatility Breakout",
            mechanism=EconomicMechanism.VOLATILITY_BREAKOUT, time_horizon="intraday-to-days",
            entry="Enter when price moves more than k*ATR from a reference level (prior close/open).",
            exit_="Time stop, or reversal below/above the same ATR band.",
            risk="ATR-scaled stop is a common companion, not a fixed dollar stop.",
            params="atr_window, k (ATR multiple).",
            template=None, required_features=("atr", "true_range"),
            limitations="No existing Phase 11 template compiles this directly today -- a "
            "full closed-DSL blueprint would be required.",
        ),
        _classic(
            "classic-opening-range-breakout",
            title="Opening Range Breakout (ORB)",
            mechanism=EconomicMechanism.OPENING_RANGE, time_horizon="intraday",
            entry="Enter on a breakout beyond the high/low of the first N minutes of the session.",
            exit_="End of session flat, or an ATR/time stop.",
            risk="Session-bounded; typically flat overnight.",
            params="opening_range_minutes.", template="silver_bullet",
            required_features=("session", "opening_range_high", "opening_range_low"),
            notes="Economically closest existing template is the native-1m Silver Bullet "
            "session-level family; not a byte-identical match.",
        ),
        _classic(
            "classic-ma-deviation-reversion",
            title="Moving-Average Deviation Mean Reversion",
            mechanism=EconomicMechanism.MEAN_REVERSION, time_horizon="days",
            entry="Enter counter-trend when price deviates more than k standard deviations "
            "from its moving average.",
            exit_="Reversion back to the moving average, or a time stop.",
            risk="Vulnerable to strong trend regimes (deviation keeps widening).",
            params="ma_window, k (std-dev multiple).",
            template="mean_reversion", required_features=("sma", "rolling_std"),
        ),
        _classic(
            "classic-bollinger-zscore-reversion",
            title="Bollinger / Z-Score Mean Reversion",
            mechanism=EconomicMechanism.MEAN_REVERSION, time_horizon="days",
            entry="Enter when a rolling z-score of price exceeds +/-k.",
            exit_="Z-score reverts toward zero.",
            risk="Same trend-regime vulnerability as MA-deviation reversion.",
            params="window, k.", template="mean_reversion",
            required_features=("zscore", "rolling_mean", "rolling_std"),
        ),
        _classic(
            "classic-rsi-reversal",
            title="RSI-Style Reversal",
            mechanism=EconomicMechanism.MEAN_REVERSION, time_horizon="days",
            entry="Enter long when a bounded oscillator (RSI-style) is deeply oversold, "
            "short when deeply overbought.",
            exit_="Oscillator reverts to a neutral band.",
            risk="No embedded stop in the classic formulation.",
            params="oscillator_window, oversold_level, overbought_level.",
            template=None,
            limitations="No RSI-style bounded-oscillator feature is registered in the current "
            "feature catalog; would need a new causal feature primitive before this compiles.",
        ),
        _classic(
            "classic-prior-day-high-low",
            title="Prior-Day High/Low Level Reaction",
            mechanism=EconomicMechanism.SESSION_EFFECTS, time_horizon="intraday",
            entry="Enter on a breakout through, or a rejection at, the prior session's "
            "high/low.",
            exit_="End of session, or a fixed target/stop off the level.",
            risk="Session-bounded.",
            params="none beyond the level definition itself.",
            template=None, required_features=("session",),
        ),
        _classic(
            "classic-overnight-gap",
            title="Overnight Gap",
            mechanism=EconomicMechanism.OVERNIGHT_GAP, time_horizon="intraday",
            entry="Trade the direction (fade or follow) of the gap between the prior close "
            "and the current session's open.",
            exit_="Gap-fill (fade variant) or session-end (follow variant).",
            risk="Session-bounded; a follow variant risks the full overnight gap.",
            params="minimum_gap_size.", template=None, required_features=("session",),
            limitations="An explicit overnight-gap feature primitive is not yet registered; "
            "the closest current proxy is a session-open-vs-prior-close return.",
        ),
        _classic(
            "classic-four-price-source-a",
            title="Four-Price / 菲阿里四价 (Fei-Ali-style), community variant A",
            mechanism=EconomicMechanism.SESSION_EFFECTS, time_horizon="days",
            entry="Compare the current bar's open against the prior bar's high/low/close "
            "(four reference prices) to classify a breakout-continuation vs. reversal "
            "state, then take the continuation side.",
            exit_="Next-bar signal re-evaluation (no embedded stop in this variant).",
            risk="No embedded stop; purely a directional filter in this source's description.",
            params="none beyond the four reference prices themselves.",
            template=None, required_features=("rolling_high", "rolling_low"),
            limitations="Source-specific interpretation A (breakout-continuation reading). "
            "Task spec section 17: preserved as ITS OWN item, distinct from variant B -- "
            "different community sources disagree on which side of the four-price "
            "relationship is the tradeable signal.",
            notes="Provenance: community-sourced strategy family; no single canonical "
            "reference implementation. Represented here as an economic-mechanism "
            "description only, never as copied source code (task spec section 19).",
        ),
        _classic(
            "classic-four-price-source-b",
            title="Four-Price / 菲阿里四价 (Fei-Ali-style), community variant B",
            mechanism=EconomicMechanism.MEAN_REVERSION, time_horizon="days",
            entry="Same four reference prices (current open vs. prior high/low/close), but "
            "this variant fades the breakout side rather than following it.",
            exit_="Next-bar signal re-evaluation.",
            risk="No embedded stop in this variant either.",
            params="none beyond the four reference prices.",
            template=None, required_features=("rolling_high", "rolling_low"),
            limitations="Source-specific interpretation B (fade/reversal reading) -- "
            "deliberately kept separate from variant A rather than merged (task spec "
            "section 17: 'do not pretend all four-price strategies are identical').",
        ),
        _classic(
            "classic-volatility-contraction-expansion",
            title="Volatility Contraction -> Expansion",
            mechanism=EconomicMechanism.VOLATILITY_TRANSITION, time_horizon="days-to-weeks",
            entry="Wait for a realized-volatility percentile to compress to a low regime, "
            "then take the breakout direction once volatility expands again.",
            exit_="Volatility regime reverts to compressed, or a time stop.",
            risk="Regime-conditioned; the entry itself is the risk filter.",
            params="vol_window, low_percentile_threshold.",
            template=None, required_features=("realized_volatility",),
            limitations="No existing Phase 11 template combines a volatility-regime gate with "
            "a breakout trigger; a full closed-DSL blueprint would be required.",
        ),
        _classic(
            "classic-regime-conditioned-trend",
            title="Regime-Conditioned Trend",
            mechanism=EconomicMechanism.REGIME_CONDITIONED_TREND, time_horizon="weeks",
            entry="Take a trend-following signal only inside a favorable (e.g. mid/low "
            "volatility, non-choppy) regime bucket; stand aside otherwise.",
            exit_="Signal reversal or regime exit.",
            risk="The regime filter IS the primary risk control.",
            params="trend lookback, regime lookback/threshold.",
            template=None, required_features=("sma", "realized_volatility"),
            notes="Directly informed by the platform's own frozen "
            "independent_axis_tertiles_cartesian_product regime transformation "
            "(Phase 15B.1b) -- an internal capability, not an external source.",
        ),
        _classic(
            "classic-regime-conditioned-mean-reversion",
            title="Regime-Conditioned Mean Reversion",
            mechanism=EconomicMechanism.REGIME_CONDITIONED_MEAN_REVERSION, time_horizon="days",
            entry="Take a mean-reversion signal only inside a favorable (e.g. range-bound, "
            "low-trend-strength) regime bucket.",
            exit_="Reversion to the mean or regime exit.",
            risk="The regime filter is the primary risk control.",
            params="reversion lookback, regime lookback/threshold.",
            template=None, required_features=("zscore", "realized_volatility"),
        ),
        _classic(
            "classic-cross-market-lead-lag",
            title="Cross-Market Lead-Lag",
            mechanism=EconomicMechanism.CROSS_MARKET_LEAD_LAG, time_horizon="days",
            entry="Trade one market's move as a lagged response to a correlated market's "
            "prior move (e.g. an equity-index futures pair).",
            exit_="Lagged-response window elapses or reverses.",
            risk="Correlation-regime dependent; can decay silently.",
            params="lag_bars, correlated_root.",
            template=None, required_features=("lagged_return", "rolling_correlation"),
            limitations="No cross-market feature primitives are wired into the current "
            "single-root execution path; genuinely a capability gap today, honestly "
            "reported as UNSUPPORTED_CROSS_MARKET_INPUT rather than silently ignored.",
        ),
        _classic(
            "classic-spread-reversion",
            title="Spread / Relative-Value Reversion",
            mechanism=EconomicMechanism.CORRELATION_SPREAD, time_horizon="days-to-weeks",
            entry="Trade the reversion of a price spread (or rolling-beta residual) between "
            "two economically related instruments back toward its historical mean.",
            exit_="Spread reverts to its rolling mean, or a time stop.",
            risk="Structural-break risk if the underlying relationship shifts.",
            params="spread_window, entry_zscore.",
            template=None, required_features=("rolling_correlation", "zscore"),
            limitations="Same cross-market capability gap as lead-lag.",
        ),
    )
