"""Small synthetic DSL demonstrations (section 23).

These exist only to exercise the DSL surface. They are **not** the Phase 11
baseline strategy library and make no claim to be tradable alpha.
"""
from __future__ import annotations

from alpha_agent.features.spec import FeatureSpec
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


def _feat(alias: str) -> FeatureOperand:
    return FeatureOperand(feature=alias)


def _const(v: float) -> ConstOperand:
    return ConstOperand(value=v)


def trend_example_spec(root_symbol: str = "NQ") -> StrategySpec:
    """ma_spread_20_100 > 0 AND volatility_60 < 0.02   -> target +1
    ma_spread_20_100 < 0                               -> target -1
    otherwise                                          -> flat
    """
    return StrategySpec(
        strategy_name="synthetic trend demo",
        strategy_id="DSL-DEMO-TREND-001",
        root_symbol=root_symbol,
        rationale="DSL demonstration only -- not a Phase 11 strategy",
        features=[
            FeatureDeclaration(
                alias="ma_spread",
                spec=FeatureSpec(kind="ma_spread", params={"fast": 20, "slow": 100}),
            ),
            FeatureDeclaration(
                alias="vol",
                spec=FeatureSpec(kind="volatility", params={"window": 60}),
            ),
        ],
        rules=[
            Rule(
                rule_id="trend_long",
                when=BooleanNode(
                    op=BooleanOp.ALL,
                    nodes=[
                        ComparisonNode(op=Comparator.GT, left=_feat("ma_spread"), right=_const(0.0)),
                        ComparisonNode(op=Comparator.LT, left=_feat("vol"), right=_const(0.02)),
                    ],
                ),
                action=TargetAction(target_units=1),
                rationale="uptrend confirmed, volatility contained",
            ),
            Rule(
                rule_id="trend_short",
                when=ComparisonNode(op=Comparator.LT, left=_feat("ma_spread"), right=_const(0.0)),
                action=TargetAction(target_units=-1),
                rationale="downtrend",
            ),
        ],
        default_action=DefaultAction.FLAT,
    )
