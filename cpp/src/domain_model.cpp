#include "quant_core/domain_model.hpp"

#include "quant_core/contract.hpp"

#include <cmath>
#include <string>

namespace quant {

bool is_plausible_raw_price(double price) noexcept {
    // Finite + magnitude only. Sign is unconstrained: a normalized futures price
    // may be positive, zero, or negative (historical negative CL). The magnitude
    // bound is the un-normalized Databento fixed-point tripwire.
    return std::isfinite(price) && std::fabs(price) < kMaxPlausibleRawPrice;
}

bool is_integral_within_tol(double x) noexcept {
    return std::isfinite(x) && std::fabs(x - std::round(x)) <= kTargetUnitsIntegralTol;
}

// ---- MarketEvent --------------------------------------------------------

void validate_market_event(const MarketEvent& ev, const ContractRegistry& registry) {
    if (ev.ts_event_ns <= 0) {
        throw InvalidMarketEvent("MarketEvent.ts_event_ns must be a positive UTC nanosecond value");
    }
    if (ev.instrument_id == 0) {
        throw InvalidMarketEvent("MarketEvent.instrument_id is 0 (unresolved contract)");
    }
    if (registry.find_by_instrument_id(ev.instrument_id) == nullptr) {
        throw InvalidMarketEvent("MarketEvent.instrument_id " + std::to_string(ev.instrument_id) +
                                 " does not resolve through the ContractRegistry");
    }
    // Phase 05 implements only Bar. Any other MarketEventType would carry a
    // payload this validator cannot check, so it is rejected until Phase 06+
    // adds the typed payload and the matching validation branch together.
    if (ev.type != MarketEventType::Bar) {
        throw InvalidMarketEvent(
            "MarketEvent.type: only Bar is implemented in Phase 05 "
            "(SessionOpen / SessionClose / Roll are reserved for Phase 06+)");
    }
    // Signed OHLC is valid (historical negative CL). The guard is finite +
    // magnitude only: a NaN/inf or an un-normalized vendor fixed-point value is
    // rejected; a legitimately zero or negative price is not.
    const double prices[4] = {ev.open, ev.high, ev.low, ev.close};
    for (const double p : prices) {
        if (!is_plausible_raw_price(p)) {
            throw InvalidMarketEvent(
                "MarketEvent bar price is non-finite or looks like un-normalized vendor "
                "fixed-point (|price| >= " + std::to_string(kMaxPlausibleRawPrice) + ")");
        }
    }
    // OHLC ordering is arithmetic and holds for signed values:
    //   low <= open <= high  and  low <= close <= high.
    if (ev.high < ev.low || ev.high < ev.open || ev.high < ev.close ||
        ev.low > ev.open || ev.low > ev.close) {
        throw InvalidMarketEvent("MarketEvent bar OHLC inconsistent (open/close outside [low, high])");
    }
    if (ev.volume < 0) {
        throw InvalidMarketEvent("MarketEvent bar volume is negative");
    }
}

// ---- Signal -----------------------------------------------------------

void validate_signal(const Signal& sig) {
    if (sig.signal_id == 0) {
        throw InvalidSignal("Signal.signal_id is 0 (the engine must stamp a deterministic id)");
    }
    if (sig.ts_decision_ns <= 0) {
        throw InvalidSignal("Signal.ts_decision_ns must be a positive UTC nanosecond value");
    }
    if (sig.root_symbol.empty()) {
        throw InvalidSignal("Signal.root_symbol is empty");
    }
    // A bare root ("NQ") has no '.'; this rejects continuous ("NQ.v.0") and
    // parent ("NQ.FUT") symbols. A Signal carries root intent, never a contract.
    if (sig.root_symbol.find('.') != std::string::npos || is_continuous_symbol(sig.root_symbol)) {
        throw InvalidSignal("Signal.root_symbol '" + sig.root_symbol +
                            "' must be a bare root (e.g. \"NQ\"), not a continuous or parent symbol");
    }
    if (!std::isfinite(sig.target_units)) {
        throw InvalidSignal("Signal.target_units is not finite");
    }
    // MVP invariant B: target_units is an integer contract target carried in a
    // double. A fractional value (e.g. 0.51) is a strategy bug -- reject it
    // rather than silently rounding it to a position.
    if (!is_integral_within_tol(sig.target_units)) {
        throw InvalidSignal("Signal.target_units " + std::to_string(sig.target_units) +
                            " is not a whole contract count (fractional contracts are not "
                            "representable; the strategy must round to an integer)");
    }
}

// ---- Order ----------------------------------------------------------

void validate_order(const Order& ord, const ContractRegistry& registry) {
    if (ord.order_id == 0) {
        throw InvalidOrder("Order.order_id is 0");
    }
    if (ord.signal_id == 0) {
        throw InvalidOrder("Order.signal_id is 0 (orphan order -- breaks the traceability chain)");
    }
    if (ord.ts_created_ns <= 0) {
        throw InvalidOrder("Order.ts_created_ns must be a positive UTC nanosecond value");
    }
    if (ord.quantity <= 0) {
        throw InvalidOrder("Order.quantity must be > 0, got " + std::to_string(ord.quantity));
    }
    if (ord.instrument_id == 0) {
        throw InvalidOrder("Order.instrument_id is 0 (unresolved contract)");
    }
    const ContractSpec* spec = registry.find_by_instrument_id(ord.instrument_id);
    if (spec == nullptr) {
        throw InvalidOrder("Order.instrument_id " + std::to_string(ord.instrument_id) +
                           " does not resolve through the ContractRegistry");
    }
    if (!is_tradable_contract_symbol(ord.raw_symbol)) {
        throw InvalidOrder("Order.raw_symbol '" + ord.raw_symbol +
                           "' is not a tradable raw contract "
                           "(continuous / back-adjusted identities are forbidden)");
    }
    if (ord.raw_symbol != spec->raw_symbol) {
        throw InvalidOrder("Order.raw_symbol '" + ord.raw_symbol + "' != registry raw_symbol '" +
                           spec->raw_symbol + "' for instrument_id " +
                           std::to_string(ord.instrument_id));
    }
    switch (ord.order_type) {
        case OrderType::Market:
            break;  // no limit / stop required
        case OrderType::Limit:
            // A limit price may be zero or negative (negative CL); it must be
            // present and a plausible normalized value (finite, magnitude bound).
            if (!ord.limit_price.has_value() || !is_plausible_raw_price(*ord.limit_price)) {
                throw InvalidOrder("Limit Order requires a present, finite, normalized limit_price");
            }
            break;
        case OrderType::Stop:
            if (!ord.stop_price.has_value() || !is_plausible_raw_price(*ord.stop_price)) {
                throw InvalidOrder("Stop Order requires a present, finite, normalized stop_price");
            }
            break;
    }
}

// ---- RiskDecision ---------------------------------------------------

void validate_risk_decision(const RiskDecision& d) {
    if (d.requested_quantity < 0) {
        throw RiskInvariantError("RiskDecision.requested_quantity must be >= 0");
    }
    if (d.approved_quantity < 0) {
        throw RiskInvariantError("RiskDecision.approved_quantity must be >= 0");
    }
    switch (d.verdict) {
        case RiskVerdict::Approve:
            if (d.requested_quantity <= 0 || d.approved_quantity != d.requested_quantity) {
                throw RiskInvariantError(
                    "APPROVE must have approved_quantity == requested_quantity > 0");
            }
            break;
        case RiskVerdict::Resize:
            if (d.approved_quantity <= 0 || d.approved_quantity >= d.requested_quantity) {
                throw RiskInvariantError(
                    "RESIZE must have 0 < approved_quantity < requested_quantity");
            }
            break;
        case RiskVerdict::Reject:
            if (d.approved_quantity != 0) {
                throw RiskInvariantError("REJECT must have approved_quantity == 0");
            }
            break;
    }
}

// ---- Signal -> Order (target-position delta) -----------------------

std::optional<Order> make_order_from_signal(const Signal& sig,
                                            const ActiveContractResolver& resolver,
                                            const MarketState& market_state,
                                            int current_position_units,
                                            OrderId order_id,
                                            std::int64_t ts_created_ns) {
    validate_signal(sig);  // also guarantees target_units is integral

    const long long target = std::llround(sig.target_units);
    const long long delta  = target - static_cast<long long>(current_position_units);
    if (delta == 0) {
        return std::nullopt;
    }

    const ContractSpec& spec = resolver.resolve(sig.root_symbol, market_state);

    Order ord;
    ord.order_id      = order_id;
    ord.signal_id     = sig.signal_id;
    ord.ts_created_ns = ts_created_ns;
    ord.instrument_id = spec.instrument_id;
    ord.raw_symbol    = spec.raw_symbol;
    ord.side          = (delta > 0) ? Side::Buy : Side::Sell;
    ord.quantity      = static_cast<int>(delta > 0 ? delta : -delta);
    ord.order_type    = OrderType::Market;
    ord.tif           = TimeInForce::Day;
    return ord;
}

}  // namespace quant
