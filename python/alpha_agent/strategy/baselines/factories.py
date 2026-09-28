"""Baseline ``StrategySpec`` factories (Phase 11 section 4).

Each factory returns exactly one :class:`StrategySpec`, scoped to one root
symbol, that compiles through the standard Phase 10 :class:`StrategyCompiler`
with no special handling. The same factory produces an ``ES`` spec or a ``CL``
spec purely by its ``root_symbol`` parameter -- no instrument economics
(tick / multiplier / margin) appear in strategy logic; the C++ ``ContractSpec``
supplies those at execution time (section 19).

Signed-price compatibility (section 20): every feature these factories reference
is signed-price safe -- arithmetic price differences, simple moving averages,
prior-window breakout distances, and a price-level z-score. None require a log
return or a percentage return across zero, so the baselines are valid for CL
(which traded below zero on 2020-04-20).
"""
from __future__ import annotations

from alpha_agent.features.spec import FeatureSpec
from alpha_agent.strategy.baselines.params import (
    BreakoutParams,
    MaTrendParams,
    MeanReversionParams,
    TsmomParams,
)
from alpha_agent.strategy.enums import BooleanOp, Comparator, DefaultAction
from alpha_agent.strategy.spec import (
    BooleanNode,
    ComparisonNode,
    ConstOperand,
    FeatureDeclaration,
    FeatureOperand,
    Rule,
    StrategySpec,
    TargetAction,
)

# Feature kinds the baselines are allowed to reference. Every one is defined by
# signed-price-safe arithmetic (no division by a possibly-zero/negative price).
SIGNED_PRICE_SAFE_KINDS = frozenset(
    {"diff", "ma", "breakout_up", "breakout_down", "zscore"}
)


def _assert_signed_price_safe(*kinds: str) -> None:
    bad = sorted(k for k in kinds if k not in SIGNED_PRICE_SAFE_KINDS)
    if bad:  # pragma: no cover - guard on internal misuse
        raise ValueError(
            f"baseline factory tried to use non-signed-price-safe feature kind(s) {bad}; "
            "baselines must stay valid for signed / zero prices (section 20)"
        )


def _feat(alias: str) -> FeatureOperand:
    return FeatureOperand(feature=alias)


def _const(v: float) -> ConstOperand:
    return ConstOperand(value=float(v))


# ==========================================================================
# A. multi-horizon time-series momentum / trend
# ==========================================================================
def make_tsmom_spec(params: TsmomParams) -> StrategySpec:
    """Economic mechanism: persistent directional drift in futures prices
    (documented trend premium). Both a fast and a slow horizon of arithmetic
    price change must agree in sign before taking a position.

    Formula / timing (bar ``T``):
      ``d_fast = price[T] - price[T - fast_horizon]``
      ``d_slow = price[T] - price[T - slow_horizon]``
      ``d_fast > 0 and d_slow > 0``  -> target ``+size``
      ``d_fast < 0 and d_slow < 0``  -> target ``-size``
      otherwise                     -> flat
    Decision uses only information through bar ``T``; execution is the engine's
    next eligible bar (no look-ahead).
    """
    _assert_signed_price_safe("diff")
    fast = FeatureSpec(
        kind="diff", params={"n": params.fast_horizon}, price_field=params.price_field
    )
    slow = FeatureSpec(
        kind="diff", params={"n": params.slow_horizon}, price_field=params.price_field
    )
    return StrategySpec(
        strategy_name=f"baseline tsmom {params.root_symbol} {params.fast_horizon}/{params.slow_horizon}",
        strategy_id=f"BASE-TSMOM-{params.root_symbol}-{params.fast_horizon}-{params.slow_horizon}-{params.size}",
        root_symbol=params.root_symbol,
        rationale=(
            "Phase 11 research baseline -- multi-horizon time-series momentum. "
            "Not claimed alpha; no profitability / robustness / significance claim."
        ),
        features=[
            FeatureDeclaration(alias="d_fast", spec=fast),
            FeatureDeclaration(alias="d_slow", spec=slow),
        ],
        rules=[
            Rule(
                rule_id="tsmom_long",
                when=BooleanNode(
                    op=BooleanOp.ALL,
                    nodes=[
                        ComparisonNode(op=Comparator.GT, left=_feat("d_fast"), right=_const(0.0)),
                        ComparisonNode(op=Comparator.GT, left=_feat("d_slow"), right=_const(0.0)),
                    ],
                ),
                action=TargetAction(target_units=params.size),
                rationale="both horizons trending up",
            ),
            Rule(
                rule_id="tsmom_short",
                when=BooleanNode(
                    op=BooleanOp.ALL,
                    nodes=[
                        ComparisonNode(op=Comparator.LT, left=_feat("d_fast"), right=_const(0.0)),
                        ComparisonNode(op=Comparator.LT, left=_feat("d_slow"), right=_const(0.0)),
                    ],
                ),
                action=TargetAction(target_units=-params.size),
                rationale="both horizons trending down",
            ),
        ],
        default_action=DefaultAction.FLAT,
    )


# ==========================================================================
# B. fast vs slow moving-average trend
# ==========================================================================
def make_ma_trend_spec(params: MaTrendParams) -> StrategySpec:
    """Economic mechanism: same trend premium, expressed as a moving-average
    crossover.

    Formula / timing (bar ``T``, both MAs backward-looking):
      ``fast_ma > slow_ma`` -> target ``+size``
      ``fast_ma < slow_ma`` -> target ``-size``
      equal / insufficient history -> flat
    """
    _assert_signed_price_safe("ma")
    fast = FeatureSpec(
        kind="ma", params={"window": params.fast_window}, price_field=params.price_field
    )
    slow = FeatureSpec(
        kind="ma", params={"window": params.slow_window}, price_field=params.price_field
    )
    return StrategySpec(
        strategy_name=f"baseline ma-trend {params.root_symbol} {params.fast_window}/{params.slow_window}",
        strategy_id=f"BASE-MATREND-{params.root_symbol}-{params.fast_window}-{params.slow_window}-{params.size}",
        root_symbol=params.root_symbol,
        rationale=(
            "Phase 11 research baseline -- moving-average trend. Not claimed alpha."
        ),
        features=[
            FeatureDeclaration(alias="fast_ma", spec=fast),
            FeatureDeclaration(alias="slow_ma", spec=slow),
        ],
        rules=[
            Rule(
                rule_id="ma_long",
                when=ComparisonNode(op=Comparator.GT, left=_feat("fast_ma"), right=_feat("slow_ma")),
                action=TargetAction(target_units=params.size),
                rationale="fast MA above slow MA",
            ),
            Rule(
                rule_id="ma_short",
                when=ComparisonNode(op=Comparator.LT, left=_feat("fast_ma"), right=_feat("slow_ma")),
                action=TargetAction(target_units=-params.size),
                rationale="fast MA below slow MA",
            ),
        ],
        default_action=DefaultAction.FLAT,
    )


# ==========================================================================
# C. Donchian / channel breakout
# ==========================================================================
def make_breakout_spec(params: BreakoutParams) -> StrategySpec:
    """Economic mechanism: breakouts from a consolidation range tend to
    continue (trend initiation).

    Formula / timing (bar ``T``):
      ``bo_up   = close[T] - max(high[T-lookback .. T-1])``   (> 0 => new high)
      ``bo_down = close[T] - min(low[T-lookback .. T-1])``     (< 0 => new low)
      ``bo_up   > 0`` -> target ``+size``
      ``bo_down < 0`` -> target ``-size``
      otherwise      -> keep the previous target (stay in until the opposite
                        breakout -- ``default_action = keep_previous_target``,
                        resolved by the Python reference evaluator before the
                        target schedule is produced).
    The channel reference is the PRIOR window only -- it never includes bar
    ``T`` or any later bar.
    """
    _assert_signed_price_safe("breakout_up", "breakout_down")
    up = FeatureSpec(kind="breakout_up", params={"window": params.lookback})
    down = FeatureSpec(kind="breakout_down", params={"window": params.lookback})
    return StrategySpec(
        strategy_name=f"baseline breakout {params.root_symbol} {params.lookback}",
        strategy_id=f"BASE-BREAKOUT-{params.root_symbol}-{params.lookback}-{params.size}",
        root_symbol=params.root_symbol,
        rationale=(
            "Phase 11 research baseline -- Donchian breakout. Not claimed alpha."
        ),
        features=[
            FeatureDeclaration(alias="bo_up", spec=up),
            FeatureDeclaration(alias="bo_down", spec=down),
        ],
        rules=[
            Rule(
                rule_id="breakout_long",
                when=ComparisonNode(op=Comparator.GT, left=_feat("bo_up"), right=_const(0.0)),
                action=TargetAction(target_units=params.size),
                rationale="close above the prior-window high",
            ),
            Rule(
                rule_id="breakout_short",
                when=ComparisonNode(op=Comparator.LT, left=_feat("bo_down"), right=_const(0.0)),
                action=TargetAction(target_units=-params.size),
                rationale="close below the prior-window low",
            ),
        ],
        default_action=DefaultAction.KEEP_PREVIOUS_TARGET,
    )


# ==========================================================================
# D. z-score mean reversion with explicit entry / exit bands
# ==========================================================================
def make_mean_reversion_spec(params: MeanReversionParams) -> StrategySpec:
    """Economic mechanism: short-horizon overshoots of price relative to a
    trailing mean tend to partially revert.

    Formula / timing (bar ``T``, all stats backward-looking):
      ``z = (price[T] - mean_w(price)) / std_w(price)``
      ordered rules:
        1. ``z < -entry_z``          -> target ``+size``  (fade the down-move)
        2. ``z > +entry_z``          -> target ``-size``  (fade the up-move)
        3. ``-exit_z < z < +exit_z`` -> target ``0``      (inside band -> flat)
        4. otherwise                 -> keep the previous target (still in the
                                        trade, between the exit and entry bands)
    ``entry_z > exit_z`` is enforced by :class:`MeanReversionParams`.
    """
    _assert_signed_price_safe("zscore")
    z = FeatureSpec(
        kind="zscore", params={"window": params.zscore_window}, price_field=params.price_field
    )
    return StrategySpec(
        strategy_name=f"baseline mean-reversion {params.root_symbol} z{params.zscore_window}",
        strategy_id=(
            f"BASE-MEANREV-{params.root_symbol}-{params.zscore_window}-"
            f"{params.entry_z:g}-{params.exit_z:g}-{params.size}"
        ),
        root_symbol=params.root_symbol,
        rationale=(
            "Phase 11 research baseline -- z-score mean reversion. Not claimed alpha."
        ),
        features=[FeatureDeclaration(alias="z", spec=z)],
        rules=[
            Rule(
                rule_id="meanrev_long",
                when=ComparisonNode(op=Comparator.LT, left=_feat("z"), right=_const(-params.entry_z)),
                action=TargetAction(target_units=params.size),
                rationale="price well below the trailing mean",
            ),
            Rule(
                rule_id="meanrev_short",
                when=ComparisonNode(op=Comparator.GT, left=_feat("z"), right=_const(params.entry_z)),
                action=TargetAction(target_units=-params.size),
                rationale="price well above the trailing mean",
            ),
            Rule(
                rule_id="meanrev_exit_band",
                when=BooleanNode(
                    op=BooleanOp.ALL,
                    nodes=[
                        ComparisonNode(op=Comparator.GT, left=_feat("z"), right=_const(-params.exit_z)),
                        ComparisonNode(op=Comparator.LT, left=_feat("z"), right=_const(params.exit_z)),
                    ],
                ),
                action=TargetAction(target_units=0),
                rationale="reverted into the exit band -> flat",
            ),
        ],
        default_action=DefaultAction.KEEP_PREVIOUS_TARGET,
    )
