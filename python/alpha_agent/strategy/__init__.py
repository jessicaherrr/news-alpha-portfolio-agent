"""Phase 10 -- Closed, typed, deterministic Strategy DSL and compiler.

    FeatureSpec -> FeatureFrame -> StrategySpec -> StrategyCompiler
        -> CompiledStrategyPlan -> ReferenceEvaluator -> StrategyDecision

The DSL represents strategy **logic only**: it transforms signal-safe feature
values into target-position intent. It has no field for, and no path to, any
execution-plane concept (fill / execution / reference price, raw symbol,
instrument id, contract month, slippage, spread, commission, latency, margin,
risk decision, fill). Execution stays entirely in the deterministic C++ Quant
Core:

    RawContract MarketEvent -> Order -> RiskManager -> ExecutionSimulator -> Fill

All legal behaviour comes from a closed allow-list implemented in source
(``alpha_agent.strategy.enums``). There is no arbitrary code, ``eval``/``exec``,
lambda, import name, or free-form expression anywhere in the pipeline. Future
Claude agents generate :class:`StrategySpec` objects, never executable code.
"""
from __future__ import annotations

from alpha_agent.strategy.compiler import (
    DEFAULT_COMPILE_LIMITS,
    CompileLimits,
    StrategyCompiler,
    compile_strategy,
)
from alpha_agent.strategy.decision import StrategyDecision
from alpha_agent.strategy.enums import (
    BooleanOp,
    Comparator,
    DefaultAction,
    MissingRulePolicy,
    NodeType,
    OperandType,
)
from alpha_agent.strategy.errors import (
    StrategyCompileError,
    StrategyDiagnostic,
    StrategyEvaluationError,
)
from alpha_agent.strategy.evaluator import ReferenceEvaluator, evaluate_strategy
from alpha_agent.strategy.fingerprint import (
    STRATEGY_DSL_VERSION,
    canonical_strategy_payload,
    strategy_fingerprint,
)
from alpha_agent.strategy.plan import (
    CompiledCondition,
    CompiledRule,
    CompiledStrategyPlan,
    FeatureBinding,
    SafetyRequirements,
)
from alpha_agent.strategy.spec import (
    BooleanNode,
    ComparisonNode,
    ConstOperand,
    FeatureDeclaration,
    FeatureOperand,
    LagOperand,
    NotNode,
    Rule,
    StrategySpec,
    TargetAction,
    strategy_from_json,
    strategy_from_yaml,
    strategy_to_json,
    strategy_to_yaml,
)

__all__ = [
    "DEFAULT_COMPILE_LIMITS",
    "STRATEGY_DSL_VERSION",
    "BooleanNode",
    "BooleanOp",
    "Comparator",
    "ComparisonNode",
    "CompileLimits",
    "CompiledCondition",
    "CompiledRule",
    "CompiledStrategyPlan",
    "ConstOperand",
    "DefaultAction",
    "FeatureBinding",
    "FeatureDeclaration",
    "FeatureOperand",
    "LagOperand",
    "MissingRulePolicy",
    "NodeType",
    "NotNode",
    "OperandType",
    "ReferenceEvaluator",
    "Rule",
    "SafetyRequirements",
    "StrategyCompileError",
    "StrategyCompiler",
    "StrategyDecision",
    "StrategyDiagnostic",
    "StrategyEvaluationError",
    "StrategySpec",
    "TargetAction",
    "canonical_strategy_payload",
    "compile_strategy",
    "evaluate_strategy",
    "strategy_fingerprint",
    "strategy_from_json",
    "strategy_from_yaml",
    "strategy_to_json",
    "strategy_to_yaml",
]
