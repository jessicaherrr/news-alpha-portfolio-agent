#pragma once

#include "quant_core/events.hpp"
#include "quant_core/margin.hpp"
#include "quant_core/risk.hpp"
#include "quant_core/risk_config.hpp"

namespace quant {

// Mandatory risk gate. Every Order MUST pass through a RiskManager and receive a
// RiskDecision before it can be executed. No Strategy or Agent may bypass this
// and no LLM may override the result (CLAUDE.md risk rule 1).
//
// Phase 05 froze the order-only interface. Phase 08 adds the context overload
// the BacktestEngine uses: hard portfolio risk (daily loss, drawdown, leverage,
// margin utilisation, gross/root/symbol caps, stale-mark protection) needs the
// PRE-trade PortfolioState, the order instrument's ContractSpec economics, and
// mark freshness. The default context overload ignores the context and forwards
// to review(order), so every Phase 05/06/07 RiskManager keeps working unchanged.
class RiskManager {
public:
    virtual ~RiskManager() = default;

    virtual RiskDecision review(const Order& order) const = 0;

    virtual RiskDecision review(const Order& order, const RiskReviewContext& ctx) const {
        (void)ctx;
        return review(order);
    }

    // Optional hooks the engine reads to configure its PortfolioAccountant. A
    // manager that returns non-null here opts the whole run into hard portfolio
    // risk / margin accounting.
    virtual const RiskConfig*      risk_config() const noexcept { return nullptr; }
    virtual const MarginModel*     margin_model() const noexcept { return nullptr; }
    virtual const PortfolioConfig* portfolio_config() const noexcept { return nullptr; }
};

// Baseline: APPROVE every well-formed order unchanged; REJECT only a
// non-positive quantity. Lets the Signal -> Order -> RiskDecision -> Fill
// pipeline be tested end to end without real limits.
class PassThroughRiskManager final : public RiskManager {
public:
    RiskDecision review(const Order& order) const override;
};

// Minimal size cap: RESIZE down to max_contracts_per_symbol, REJECT a
// non-positive quantity or a non-positive cap. Not the full risk engine -- just
// enough to exercise APPROVE / RESIZE / REJECT.
class MaxContractsRiskManager final : public RiskManager {
public:
    explicit MaxContractsRiskManager(RiskLimits limits) noexcept : limits_(limits) {}
    RiskDecision review(const Order& order) const override;

private:
    RiskLimits limits_;
};

// ============================================================================
// Phase 08: deterministic hard portfolio risk
// ============================================================================
//
// FLIP DECOMPOSITION. An order that opposes the current position is split into a
// risk-REDUCING close leg (contracts that move the position toward flat) and a
// risk-INCREASING opening leg (the remainder, which opens new opposite-side
// exposure once the old position is gone). The close leg is ALWAYS approved --
// kill switches and limits may never trap risk reduction. Only the opening leg
// faces the hierarchy below; if it is fully blocked the decision RESIZEs down to
// the close quantity (flatten only), it is never rejected outright.
//   +2, SELL 1  -> close 1              -> APPROVE
//   +2, SELL 2  -> close 2 (flat)       -> APPROVE
//   +2, SELL 3  -> close 2 + open -1    -> open leg risk-gated (APPROVE 3 or RESIZE 2)
//   +2, SELL 5, only 1 new short fits   -> RESIZE to SELL 3 (close 2 + open 1)
//   +2, SELL 5, kill switch active      -> RESIZE to SELL 2 (flatten only)
// (mirror for short -> long.)
//
// Evaluation hierarchy for the OPENING leg (first decisive rule wins):
//   0. structural            -- quantity <= 0                     -> REJECT
//   1. drawdown kill switch  -- drawdown_pct/usd breached         -> block opening leg
//   2. daily-loss limit      -- day equity drop >= limit          -> block opening leg
//   3. stale-mark protection -- ref price or portfolio mark stale -> block opening leg
//   4. missing margin        -- no MarginRequirement, policy=Reject -> block opening leg
//   5. position caps         -- per-symbol / per-root / gross      -> RESIZE, else block
//   6. exposure / leverage / margin utilisation                    -> RESIZE, else block
//   7. otherwise                                                   -> APPROVE
//
// Steps 1-4 "block the opening leg": feasible opening quantity is 0, so a flip
// RESIZEs to the close quantity and a pure new position is REJECTed. Steps 5-6
// RESIZE via a downward integer scan for the largest opening quantity whose
// post-trade portfolio state satisfies every limit (contract counts are tiny).
// max_order_contracts caps the OPENING leg only (never the close).
//
// review(order) with no context applies only the checks that need no portfolio
// state: non-positive quantity and, if configured, max_order_contracts (as a
// whole-order cap, since the position is unknown). Full enforcement requires the
// context overload the engine uses.
class PortfolioRiskManager final : public RiskManager {
public:
    PortfolioRiskManager(RiskConfig config, const MarginModel* margin /* nullable */) noexcept
        : config_(config), margin_(margin) {}

    RiskDecision review(const Order& order) const override;
    RiskDecision review(const Order& order, const RiskReviewContext& ctx) const override;

    const RiskConfig*      risk_config() const noexcept override { return &config_; }
    const MarginModel*     margin_model() const noexcept override { return margin_; }
    const PortfolioConfig* portfolio_config() const noexcept override { return &config_.portfolio; }

private:
    RiskConfig         config_;
    const MarginModel* margin_{nullptr};
};

// Apply a decision to its order: returns the order with quantity ==
// approved_quantity. Throws RiskInvariantError if the decision does not belong
// to the order (id / instrument mismatch), violates its own invariants, or the
// verdict is REJECT. This is the sanctioned "risk-approved order" step.
Order apply_risk_decision(const Order& order, const RiskDecision& decision);

}  // namespace quant
