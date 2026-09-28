#include "quant_core/risk_manager.hpp"

#include "quant_core/domain_errors.hpp"
#include "quant_core/domain_model.hpp"
#include "quant_core/ids.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>

namespace quant {
namespace {

RiskDecision base_decision(const Order& order) {
    RiskDecision d;
    d.ts_decision_ns     = order.ts_created_ns;
    d.order_id           = order.order_id;
    d.instrument_id      = order.instrument_id;
    d.requested_quantity = order.quantity;
    d.approved_quantity  = 0;
    return d;
}

}  // namespace

RiskDecision PassThroughRiskManager::review(const Order& order) const {
    RiskDecision d = base_decision(order);
    if (order.quantity <= 0) {
        d.verdict          = RiskVerdict::Reject;
        d.approved_quantity = 0;
        d.reason_code      = "non_positive_quantity";
        return d;
    }
    d.verdict           = RiskVerdict::Approve;
    d.approved_quantity = order.quantity;
    d.reason_code       = "ok";
    return d;
}

RiskDecision MaxContractsRiskManager::review(const Order& order) const {
    RiskDecision d = base_decision(order);
    const int cap = limits_.max_contracts_per_symbol;

    if (order.quantity <= 0) {
        d.verdict     = RiskVerdict::Reject;
        d.reason_code = "non_positive_quantity";
    } else if (cap <= 0) {
        d.verdict     = RiskVerdict::Reject;
        d.reason_code = "max_contracts_per_symbol";
    } else if (order.quantity > cap) {
        d.verdict           = RiskVerdict::Resize;
        d.approved_quantity = cap;
        d.reason_code       = "max_contracts_per_symbol";
    } else {
        d.verdict           = RiskVerdict::Approve;
        d.approved_quantity = order.quantity;
        d.reason_code       = "ok";
    }
    return d;
}

// ============================================================================
// PortfolioRiskManager -- deterministic hard portfolio risk (Phase 08)
// ============================================================================

namespace {

// Per-instrument figures already counted in the pre-trade PortfolioState.
struct InstrumentInPortfolio {
    double gross_notional_usd{0.0};
    double signed_notional_usd{0.0};
    double initial_margin_usd{0.0};
};

InstrumentInPortfolio find_in_portfolio(const PortfolioState& pf, std::uint32_t instrument_id) {
    for (const PositionExposure& e : pf.positions) {
        if (e.instrument_id == instrument_id) {
            return {e.gross_notional_usd, e.signed_notional_usd, e.initial_margin_usd};
        }
    }
    return {};
}

// Evaluate every post-trade limit for a candidate signed delta `sd` (sd != 0).
// Returns an empty string when admissible, else the binding reason_code.
std::string check_post_trade_limits(const Order& order, const RiskReviewContext& ctx,
                                    const RiskConfig& cfg, const MarginModel* margin, int sd) {
    const PortfolioState& pf = *ctx.portfolio;
    const int cur      = ctx.current_position_units;
    const int post     = cur + sd;
    const int abs_post  = std::abs(post);
    const int abs_cur   = std::abs(cur);

    // ---- 5. position caps (post-trade counts) ----
    if (cfg.max_contracts_per_symbol > 0 && abs_post > cfg.max_contracts_per_symbol) {
        return "max_contracts_per_symbol";
    }
    if (cfg.max_contracts_per_root > 0 &&
        std::abs(ctx.current_root_units + sd) > cfg.max_contracts_per_root) {
        return "max_contracts_per_root";
    }
    const int gross_contracts_after = pf.gross_contracts - abs_cur + abs_post;
    if (cfg.max_gross_contracts > 0 && gross_contracts_after > cfg.max_gross_contracts) {
        return "max_gross_contracts";
    }

    // ---- 6. exposure / leverage / margin utilisation (ContractSpec economics) ----
    //
    // Notional is |contracts| x |price| x multiplier so it stays non-negative
    // even for a negative futures price (CL below zero). Direction follows the
    // POSITION sign, never the price sign.
    const double px_abs = std::fabs(ctx.reference_price);
    const double mult   = ctx.spec->multiplier;
    const int    post_sign = (post > 0) - (post < 0);
    const InstrumentInPortfolio prior = find_in_portfolio(pf, order.instrument_id);

    const double new_gross_notional = static_cast<double>(abs_post) * px_abs * mult;
    const double gross_exposure_after =
        pf.gross_exposure_usd - prior.gross_notional_usd + new_gross_notional;
    if (cfg.max_gross_exposure_usd > 0.0 && gross_exposure_after > cfg.max_gross_exposure_usd) {
        return "max_gross_exposure_usd";
    }

    const double equity = pf.equity_usd;

    if (cfg.max_gross_leverage > 0.0) {
        if (equity <= 0.0) return "max_gross_leverage";
        if (gross_exposure_after / equity > cfg.max_gross_leverage) return "max_gross_leverage";
    }
    if (cfg.max_net_leverage > 0.0) {
        const double new_signed_notional = post_sign * new_gross_notional;
        const double net_exposure_after =
            pf.net_exposure_usd - prior.signed_notional_usd + new_signed_notional;
        if (equity <= 0.0) return "max_net_leverage";
        // Symmetric absolute limit: a large net-SHORT book (net_exposure_after
        // negative) must not slip under the limit on sign.
        if (std::fabs(net_exposure_after) / equity > cfg.max_net_leverage) return "max_net_leverage";
    }

    if (cfg.max_margin_utilization_pct > 0.0) {
        const std::optional<double> per_contract =
            margin ? margin->initial_per_contract(ctx.spec->root_symbol) : std::optional<double>{};
        // A missing requirement under MissingMarginPolicy::Reject is handled
        // before the scan; here (TreatAsZero, or a known figure) 0 is safe.
        const double add_per_contract = per_contract.value_or(0.0);
        const double initial_margin_after =
            pf.initial_margin_usd - prior.initial_margin_usd + add_per_contract * abs_post;
        const double util = equity > 0.0
                                ? initial_margin_after / equity
                                : (initial_margin_after > 0.0 ? kUtilizationNoEquity : 0.0);
        if (util > cfg.max_margin_utilization_pct) return "max_margin_utilization_pct";
    }

    return {};
}

}  // namespace

RiskDecision PortfolioRiskManager::review(const Order& order) const {
    RiskDecision d = base_decision(order);
    if (order.quantity <= 0) {
        d.verdict           = RiskVerdict::Reject;
        d.approved_quantity = 0;
        d.reason_code       = "non_positive_quantity";
        return d;
    }
    // Context-free path: only the checks that need no portfolio / market state.
    if (config_.max_order_contracts > 0 && order.quantity > config_.max_order_contracts) {
        d.verdict           = RiskVerdict::Resize;
        d.approved_quantity = config_.max_order_contracts;
        d.reason_code       = "max_order_contracts";
        return d;
    }
    d.verdict           = RiskVerdict::Approve;
    d.approved_quantity = order.quantity;
    d.reason_code       = "ok";
    return d;
}

RiskDecision PortfolioRiskManager::review(const Order& order, const RiskReviewContext& ctx) const {
    RiskDecision d = base_decision(order);

    // ---- 0. structural ----
    if (order.quantity <= 0) {
        d.verdict           = RiskVerdict::Reject;
        d.approved_quantity = 0;
        d.reason_code       = "non_positive_quantity";
        return d;
    }
    // The engine always supplies portfolio + spec. A caller that does not gets
    // the safe context-free path rather than a half-checked decision.
    if (ctx.portfolio == nullptr || ctx.spec == nullptr) {
        return review(order);
    }
    if (ctx.as_of_ts_ns > 0) d.ts_decision_ns = ctx.as_of_ts_ns;

    const int sign      = (order.side == Side::Buy) ? 1 : -1;
    const int cur       = ctx.current_position_units;
    const int cur_sign  = (cur > 0) - (cur < 0);
    const int requested = order.quantity;

    // ---- FLIP DECOMPOSITION -------------------------------------------------
    //
    // Split the order into:
    //   close_qty       -- contracts that merely reduce the existing position
    //                      toward flat (only when the order opposes the position)
    //   requested_open  -- contracts that open NEW exposure (same side as the
    //                      order once the old position is gone, or an add on the
    //                      existing side)
    //
    // The close leg is ALWAYS approved -- never trap risk reduction. Only the
    // opening leg faces the hard risk hierarchy. When requested_open > 0 the
    // order fully consumes the old position, so the opening leg starts from flat
    // and post-trade units == sign * (close_qty + open_qty) + ... == the value
    // check_post_trade_limits computes from `cur + sign * q_total`.
    const int close_qty =
        (cur != 0 && sign == -cur_sign) ? std::min(requested, std::abs(cur)) : 0;
    const int requested_open = requested - close_qty;

    // Pure reduction / exact flatten: no new exposure at all -> APPROVE, whatever
    // the portfolio state (kill switches included).
    if (requested_open == 0) {
        d.verdict           = RiskVerdict::Approve;
        d.approved_quantity = requested;
        d.reason_code       = "ok";
        return d;
    }

    const PortfolioState& pf = *ctx.portfolio;

    // ---- kill switches / stale-mark / missing-margin: block the OPENING leg
    //      only. A flip still keeps its close leg (RESIZE down to close_qty).
    std::string opening_block;
    {
        const bool drawdown_pct_breach = config_.max_drawdown_pct > 0.0 &&
                                         pf.peak_equity_usd > 0.0 &&
                                         pf.drawdown_pct >= config_.max_drawdown_pct;
        const bool drawdown_usd_breach = config_.max_drawdown_usd > 0.0 &&
                                         pf.drawdown_usd >= config_.max_drawdown_usd;
        const bool have_margin = margin_ != nullptr && margin_->has(ctx.spec->root_symbol);
        if (drawdown_pct_breach || drawdown_usd_breach) {
            opening_block = "drawdown_kill_switch";
        } else if (config_.max_daily_loss_usd > 0.0 &&
                   (pf.day_start_equity_usd - pf.equity_usd) >= config_.max_daily_loss_usd) {
            opening_block = "daily_loss_limit";
        } else if (config_.stale_mark == StaleMarkPolicy::RejectRiskIncreasing &&
                   (!ctx.reference_price_present || ctx.reference_price_stale ||
                    pf.has_stale_mark)) {
            opening_block = "stale_mark";
        } else if (!have_margin && config_.missing_margin == MissingMarginPolicy::Reject) {
            opening_block = "missing_margin_metadata";
        }
    }

    // ---- largest feasible OPENING quantity (downward scan). max_order_contracts
    //      caps the opening leg (never the close -- that would trap reduction).
    int  ceiling_open = requested_open;
    bool order_capped = false;
    if (config_.max_order_contracts > 0 && ceiling_open > config_.max_order_contracts) {
        ceiling_open = config_.max_order_contracts;
        order_capped = true;
    }

    std::string binding = opening_block;
    int feasible_open = 0;
    if (opening_block.empty()) {
        for (int oq = ceiling_open; oq >= 1; --oq) {
            const int q_total = close_qty + oq;
            const std::string reason =
                check_post_trade_limits(order, ctx, config_, margin_, sign * q_total);
            if (reason.empty()) {
                feasible_open = oq;
                break;
            }
            binding = reason;  // reason at the smallest opening qty tried so far
        }
        if (order_capped && feasible_open == ceiling_open && feasible_open < requested_open) {
            binding = "max_order_contracts";
        }
    }
    if (binding.empty()) binding = "risk_limit";  // defensive; unreachable in practice

    const int approved = close_qty + feasible_open;

    if (approved == 0) {
        // Reachable only for a pure opening order (close_qty == 0).
        d.verdict           = RiskVerdict::Reject;
        d.approved_quantity = 0;
        d.reason_code       = binding;
        return d;
    }
    if (approved == requested) {
        d.verdict           = RiskVerdict::Approve;
        d.approved_quantity = requested;
        d.reason_code       = "ok";
        return d;
    }
    // 0 < approved < requested -- the close leg (and any feasible opening) clears;
    // the rest is trimmed. Never resize away the risk-reducing close portion.
    d.verdict           = RiskVerdict::Resize;
    d.approved_quantity = approved;
    d.reason_code       = binding;
    return d;
}

Order apply_risk_decision(const Order& order, const RiskDecision& decision) {
    if (decision.order_id != order.order_id) {
        throw RiskInvariantError("RiskDecision.order_id " + std::to_string(decision.order_id) +
                                 " does not match Order " + trace_tag('O', order.order_id));
    }
    if (decision.instrument_id != order.instrument_id) {
        throw RiskInvariantError("RiskDecision.instrument_id does not match Order " +
                                 trace_tag('O', order.order_id));
    }
    validate_risk_decision(decision);
    if (decision.verdict == RiskVerdict::Reject) {
        throw RiskInvariantError("Order " + trace_tag('O', order.order_id) +
                                 " rejected by risk: " + decision.reason_code);
    }
    Order approved = order;
    approved.quantity = decision.approved_quantity;
    return approved;
}

}  // namespace quant
