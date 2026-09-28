#include "quant_core/fill.hpp"

#include "quant_core/domain_model.hpp"

#include <cmath>
#include <string>

namespace quant {
namespace {

const char* domain_name(PriceDomain d) {
    switch (d) {
        case PriceDomain::RawContract:   return "RawContract";
        case PriceDomain::RawContinuous: return "RawContinuous";
        case PriceDomain::BackAdjusted:  return "BackAdjusted";
    }
    return "unknown";
}

}  // namespace

Fill make_fill(const ContractRegistry& registry, const FillRequest& req) {
    // 1. Price domain: only real raw-contract prices may become fills.
    if (req.price_domain != PriceDomain::RawContract) {
        throw FillResolutionError(
            std::string("make_fill: price_domain must be RawContract, got ") +
            domain_name(req.price_domain) +
            " -- back-adjusted and continuous prices never reach the execution path");
    }

    // 2/3. Resolve the real contract.
    if (req.instrument_id == 0) {
        throw FillResolutionError("make_fill: instrument_id is 0 (unresolved contract)");
    }
    const ContractSpec* spec = registry.find_by_instrument_id(req.instrument_id);
    if (spec == nullptr) {
        throw FillResolutionError("make_fill: unknown instrument_id " +
                                  std::to_string(req.instrument_id));
    }
    if (!spec->is_valid() || !is_tradable_contract_symbol(spec->raw_symbol)) {
        throw FillResolutionError("make_fill: registry entry for instrument_id " +
                                  std::to_string(req.instrument_id) +
                                  " is not a tradable contract ('" + spec->raw_symbol + "')");
    }

    // 4. Optional caller cross-check.
    if (!req.expected_raw_symbol.empty() && req.expected_raw_symbol != spec->raw_symbol) {
        throw FillResolutionError("make_fill: expected_raw_symbol '" + req.expected_raw_symbol +
                                  "' does not match registry '" + spec->raw_symbol + "'");
    }

    // 5. Contract must have been tradable at the fill timestamp.
    if (!spec->is_live_at(req.ts_fill_ns)) {
        throw FillResolutionError("make_fill: contract '" + spec->raw_symbol +
                                  "' was not tradable at ts " + std::to_string(req.ts_fill_ns));
    }

    // 6. Quantity / price sanity.
    if (req.quantity <= 0) {
        throw FillResolutionError("make_fill: quantity must be > 0, got " +
                                  std::to_string(req.quantity));
    }
    // A normalized raw-contract execution price may be positive, zero, or
    // negative (historical negative CL). Reject only non-finite values and
    // un-normalized vendor fixed-point (magnitude tripwire). Missing data never
    // arrives here as a price -- it is an upstream rejection / optional.
    if (!is_plausible_raw_price(req.price)) {
        throw FillResolutionError(
            "make_fill: price must be a finite normalized raw-contract value "
            "(|price| < " + std::to_string(kMaxPlausibleRawPrice) +
            "); got " + std::to_string(req.price));
    }

    // 7. Round to the contract tick grid (deterministic, documented). std::round
    //    is symmetric about zero (round-half-away-from-zero), so a negative price
    //    aligns by the SAME rule: -20.007 / 0.01 = -2000.7 -> -2001 -> -20.01.
    const double ticks = std::round(req.price / spec->tick_size);
    const double aligned_price = ticks * spec->tick_size;

    Fill fill;
    fill.fill_id        = req.fill_id;
    fill.order_id       = req.order_id;
    fill.ts_fill_ns     = req.ts_fill_ns;
    fill.instrument_id  = spec->instrument_id;
    fill.raw_symbol     = spec->raw_symbol;
    fill.side           = req.side;
    fill.quantity       = req.quantity;
    fill.fill_price     = aligned_price;
    fill.tick_size      = spec->tick_size;   // from the contract, not the caller
    fill.multiplier     = spec->multiplier;  // from the contract, not the caller
    fill.commission_usd = req.commission_usd;
    fill.slippage_ticks = req.slippage_ticks;
    fill.price_domain   = PriceDomain::RawContract;
    return fill;
}

}  // namespace quant
