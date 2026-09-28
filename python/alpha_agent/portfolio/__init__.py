"""News Alpha Phase G -- deterministic multi-asset portfolio construction.

    RankedSignalSet -> selection -> risk model -> C++ allocator -> PortfolioPlan
                    -> execution handoff -> existing C++ engine (replay)

Five separated components: signal SELECTION (`selection`, Python), the RISK
MODEL (`risk_model`, Python statistics), ALLOCATION + RISK BUDGETING (C++
`quant::construct_portfolio`, bridged by `allocator`), EXECUTION TARGETS
(`handoff` + the C++ replay CLI) and ACCOUNTING (the unchanged C++
PortfolioAccountant). See docs/NEWS_ALPHA_PHASE_G_PORTFOLIO_CONSTRUCTION.md.

Construction is not validation: no plan carries a verdict or an expected
return.

News Alpha Phase H closes the loop on top (docs/NEWS_ALPHA_PHASE_H_VALIDATION_AND_MEMORY.md):
`validation` (the input and execution gates, then the existing validation
plane), `strategy` (the frozen portfolio and its causal rebalance schedule),
`evaluation` (member factors through the validation window), `execution`
(engine inputs + one C++ replay per span), `registry_record` (validated
portfolios as registry experiments) and `research_loop` (one call from a plan
to recorded evidence). Import them from their modules.
"""
from alpha_agent.portfolio.classification import (
    RiskAssetClass,
    RiskClassification,
    classify_instrument,
)
from alpha_agent.portfolio.plan import PlanStatus, PortfolioPlan, construct_portfolio_plan
from alpha_agent.portfolio.policy import (
    EligibilityMode,
    PortfolioConstraints,
    PortfolioConstructionPolicy,
    resolve_constraints,
)

__all__ = [
    "EligibilityMode",
    "PlanStatus",
    "PortfolioConstraints",
    "PortfolioConstructionPolicy",
    "PortfolioPlan",
    "RiskAssetClass",
    "RiskClassification",
    "classify_instrument",
    "construct_portfolio_plan",
    "resolve_constraints",
]
