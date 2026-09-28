#include "quant_core/execution_simulator.hpp"

#include "quant_core/contract.hpp"
#include "quant_core/domain_model.hpp"

#include <string>

namespace quant {
namespace {

struct PricePick {
    double      reference{0.0};
    bool        gap_through{false};
    bool        reachable{true};
};

// Limit order: fills only when the bar's range makes the limit price reachable.
// No queue-priority assumption -- OHLCV cannot prove one, so the fill price is
// never better than the limit EXCEPT when the bar opens already through it (the
// open is the real first print of the bar), which is genuine price improvement.
PricePick pick_limit(Side side, double limit, const MarketBar& bar) {
    PricePick p;
    if (side == Side::Buy) {
        if (bar.low > limit) { p.reachable = false; return p; }
        p.gap_through = bar.open <= limit;
        p.reference   = p.gap_through ? bar.open : limit;
    } else {
        if (bar.high < limit) { p.reachable = false; return p; }
        p.gap_through = bar.open >= limit;
        p.reference   = p.gap_through ? bar.open : limit;
    }
    return p;
}

// Stop order: triggers when the bar trades to/through the stop price. Once
// triggered it is marketable -- it fills at the stop price plus adverse
// slippage, or at the bar's open when the bar gapped through the stop (you were
// already triggered before the first print, and cannot fill better than it).
PricePick pick_stop(Side side, double stop, const MarketBar& bar) {
    PricePick p;
    if (side == Side::Buy) {
        if (bar.high < stop) { p.reachable = false; return p; }
        p.gap_through = bar.open >= stop;
        p.reference   = p.gap_through ? bar.open : stop;
    } else {
        if (bar.low > stop) { p.reachable = false; return p; }
        p.gap_through = bar.open <= stop;
        p.reference   = p.gap_through ? bar.open : stop;
    }
    return p;
}

}  // namespace

ExecReport BarExecutionSimulator::execute(const Order& order, const ExecutionRequest& req,
                                          FillId fill_id) const {
    const ContractSpec* spec = registry_.find_by_instrument_id(order.instrument_id);
    if (spec == nullptr) {
        return ExecReport{ExecOutcome::Rejected, std::nullopt, 0.0, "unknown_instrument"};
    }
    const double tick = spec->tick_size;                       // ContractSpec is the sole authority
    const double dir  = (order.side == Side::Buy) ? 1.0 : -1.0;
    const bool   marketable_costs = !req.administrative;
    const double adverse_ticks_cfg =
        marketable_costs ? (config_.slippage_ticks + config_.spread_ticks) : 0.0;

    double      reference     = 0.0;
    double      applied_ticks = 0.0;
    std::string detail;

    switch (order.order_type) {
        case OrderType::Market: {
            reference     = req.ref_price_override.value_or(req.bar.open);
            applied_ticks = adverse_ticks_cfg;
            detail        = req.administrative ? "administrative" : "market";
            break;
        }
        case OrderType::Limit: {
            const PricePick p = pick_limit(order.side, order.limit_price.value(), req.bar);
            if (!p.reachable) {
                return ExecReport{ExecOutcome::NoFill, std::nullopt, 0.0, "limit_unreachable"};
            }
            reference     = p.reference;
            applied_ticks = 0.0;  // a limit order never pays slippage past its own price
            detail        = p.gap_through ? "limit_gap_through" : "limit_at_price";
            break;
        }
        case OrderType::Stop: {
            const PricePick p = pick_stop(order.side, order.stop_price.value(), req.bar);
            if (!p.reachable) {
                return ExecReport{ExecOutcome::NoFill, std::nullopt, 0.0, "stop_not_triggered"};
            }
            reference     = p.reference;
            applied_ticks = adverse_ticks_cfg;  // a triggered stop is marketable -> it slips
            detail        = p.gap_through ? "stop_gap_through" : "stop_triggered";
            break;
        }
    }

    // Execution consumes ONLY normalized raw-contract prices. A continuous /
    // back-adjusted price (rejected by make_fill via price_domain) or an
    // un-normalized Databento fixed-point value (~1e13) must never become a fill
    // -- guard the reference before it is priced (defence in depth ahead of
    // make_fill's finite + magnitude check). A zero or negative reference (e.g.
    // negative CL) is valid and passes; all Market/Limit/Stop comparisons above
    // are relational and hold across zero.
    if (!is_plausible_raw_price(reference)) {
        return ExecReport{ExecOutcome::Rejected, std::nullopt, 0.0, "implausible_raw_price"};
    }

    // Slippage in ticks -> a raw reference price. make_fill() then performs the
    // final contract validation and tick-grid rounding (no product-specific tick
    // constants anywhere here).
    const double raw_price = reference + dir * applied_ticks * tick;

    FillRequest fr;
    fr.fill_id             = fill_id;
    fr.order_id            = order.order_id;
    fr.ts_fill_ns          = req.ts_ns;
    fr.instrument_id       = order.instrument_id;
    fr.side                = order.side;
    fr.quantity            = order.quantity;
    fr.price               = raw_price;
    fr.price_domain        = PriceDomain::RawContract;
    fr.commission_usd      = config_.commission_per_contract_usd * order.quantity;
    if (!config_.commission_per_contract_usd_by_root.empty()) {
        const ContractSpec* spec = registry_.find_by_instrument_id(order.instrument_id);
        if (spec != nullptr) {
            const auto rate = config_.commission_per_contract_usd_by_root.find(spec->root_symbol);
            if (rate != config_.commission_per_contract_usd_by_root.end()) {
                fr.commission_usd = rate->second * order.quantity;
            }
        }
    }
    fr.slippage_ticks      = applied_ticks;
    fr.expected_raw_symbol = order.raw_symbol;

    Fill fill = make_fill(registry_, fr);  // execution-domain guards + tick rounding live here
    return ExecReport{ExecOutcome::Filled, fill, applied_ticks, detail};
}

BracketExit resolve_intrabar_bracket(const MarketBar& bar, Side exit_side,
                                     double stop_price, double target_price) noexcept {
    bool stop_reachable   = false;
    bool target_reachable = false;
    bool open_through_stop   = false;
    bool open_through_target = false;

    if (exit_side == Side::Sell) {          // exiting a long: stop below, target above
        stop_reachable      = bar.low  <= stop_price;
        target_reachable    = bar.high >= target_price;
        open_through_stop   = bar.open <= stop_price;
        open_through_target = bar.open >= target_price;
    } else {                               // exiting a short: stop above, target below
        stop_reachable      = bar.high >= stop_price;
        target_reachable    = bar.low  <= target_price;
        open_through_stop   = bar.open >= stop_price;
        open_through_target = bar.open <= target_price;
    }

    if (open_through_stop)              return BracketExit::Stop;    // gapped past the stop
    if (open_through_target)            return BracketExit::Target;  // gapped past the target
    if (stop_reachable && target_reachable) return BracketExit::Stop;  // conservative: worst case
    if (stop_reachable)                 return BracketExit::Stop;
    if (target_reachable)              return BracketExit::Target;
    return BracketExit::None;
}

}  // namespace quant
