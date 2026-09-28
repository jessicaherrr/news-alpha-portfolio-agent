#pragma once

#include "quant_core/contract_registry.hpp"
#include "quant_core/domain_errors.hpp"
#include "quant_core/events.hpp"

#include <string>

namespace quant {

// A proposed fill, before it has been checked against the contract registry.
struct FillRequest {
    FillId        fill_id{0};
    OrderId       order_id{0};
    std::int64_t  ts_fill_ns{0};
    std::uint32_t instrument_id{0};
    Side          side{Side::Buy};
    int           quantity{0};
    double        price{0.0};
    PriceDomain   price_domain{PriceDomain::RawContract};
    double        commission_usd{0.0};
    double        slippage_ticks{0.0};
    // Optional cross-check: the symbol the caller believed it was trading. When
    // non-empty it must match the registry's raw_symbol for `instrument_id`.
    std::string   expected_raw_symbol;
};

class FillResolutionError : public ExecutionDomainError {
public:
    using ExecutionDomainError::ExecutionDomainError;
};

// The ONLY sanctioned way to build a Fill for the backtester.
//
// Enforces the execution invariant (docs/BOUNDARY_CONTRACT.md section E):
//   * price_domain must be PriceDomain::RawContract
//   * instrument_id must be non-zero and present in `registry`
//   * the registry's raw_symbol must be a real tradable contract symbol
//   * the contract must have been live at ts_fill_ns
//   * quantity > 0 and price finite and positive
//   * tick_size and multiplier are taken FROM the ContractSpec, never the caller
//   * fill_price is rounded to the contract tick grid
//
// Throws FillResolutionError on any violation.
Fill make_fill(const ContractRegistry& registry, const FillRequest& request);

}  // namespace quant
