#pragma once

#include <cstdint>

namespace quant {

// One OHLCV bar of a single real futures contract.
//
// Prices are the actual traded prices of the contract identified by
// `instrument_id` (PriceDomain::RawContract). A bar never holds a back-adjusted
// or stitched-continuous price; those live in separate series and separate types
// (see docs/BOUNDARY_CONTRACT.md). `raw_symbol`, tick size and multiplier are
// intentionally NOT stored here -- they are resolved from the ContractRegistry
// via `instrument_id`, which keeps the bar a small trivially-copyable value.
struct MarketBar {
    std::int64_t  ts_event_ns{};    // UTC nanoseconds since epoch; start of the interval (event time)
    std::uint32_t instrument_id{};  // real contract this bar belongs to; 0 == unknown / unresolved
    double        open{};
    double        high{};
    double        low{};
    double        close{};
    std::int64_t  volume{};         // contracts traded during the interval
};

// The Phase 01 Position / Trade / BacktestConfig / BacktestResult value types
// were retired in Phase 06 together with the ad-hoc run_backtest() loop. The
// event-driven engine's result types live in engine.hpp; the position ledger's
// in position_ledger.hpp. Contract tick / multiplier now always come from the
// ContractSpec, never from a config default.

}  // namespace quant
