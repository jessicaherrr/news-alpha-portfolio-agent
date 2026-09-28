#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace quant {

// A real, tradable futures contract.
//
// All timestamps are UTC nanoseconds since the Unix epoch. `tick_size` and
// `multiplier` are mandatory and must be positive -- they are the only numbers
// the execution layer is allowed to use for price rounding and PnL conversion.
// (The Phase 01 BacktestConfig tick/multiplier defaults were removed in Phase 06;
// the ContractSpec is now the sole authority.)
//
// `multiplier` is the **point value / pnl_multiplier**: USD PnL per 1.0 move in
// the quoted price, per contract. For NQ (E-mini Nasdaq-100) it is 20.0, so
// `tick_size * multiplier == 0.25 * 20.0 == 5.0` == the USD tick value. It is
// DERIVED from the Databento definition (unit_of_measure_qty + the documented
// price-scale derivation), never taken from `contract_multiplier` -- see
// docs/CONTRACT_ECONOMICS.md.
struct ContractSpec {
    std::uint32_t instrument_id{0};   // Databento per-dataset instrument id; 0 == invalid / unset
    std::string   raw_symbol;         // e.g. "NQZ6" -- the actual exchange contract symbol
    std::string   root_symbol;        // e.g. "NQ"
    std::string   exchange;           // e.g. "XCME" (ISO 10383 MIC) or "GLBX"
    double        tick_size{0.0};     // minimum price increment, in quoted price points
    double        multiplier{0.0};    // point value: USD PnL per 1.0 quoted-price move per contract
    std::int64_t  activation_ns{0};   // first instant the contract is tradable
    std::int64_t  expiration_ns{0};   // nominal expiry instant
    std::optional<std::int64_t> first_notice_ns{};  // physically-delivered products (CL, GC, ZN...)
    std::optional<std::int64_t> last_trade_ns{};    // when distinct from expiration_ns

    // Structural validity only -- says nothing about rolls or market state.
    bool is_valid() const noexcept {
        return instrument_id != 0
            && !raw_symbol.empty()
            && !root_symbol.empty()
            && tick_size > 0.0
            && multiplier > 0.0
            && activation_ns > 0
            && expiration_ns > activation_ns;
    }

    // Latest instant this contract can still be traded.
    std::int64_t tradable_until_ns() const noexcept {
        return last_trade_ns.value_or(expiration_ns);
    }

    // Was the contract tradable at `ts_ns`?
    bool is_live_at(std::int64_t ts_ns) const noexcept {
        return ts_ns >= activation_ns && ts_ns <= tradable_until_ns();
    }
};

// Databento continuous / synthetic symbology: "<ROOT>.<rule>.<rank>" with rule
// in {c (calendar), v (volume), n (open interest)}, e.g. "NQ.v.0", "ES.c.1".
// Such a symbol denotes a rolling *series*, never a tradable contract.
bool is_continuous_symbol(std::string_view symbol) noexcept;

// A real contract symbol is non-empty and contains no '.' (which would make it a
// continuous symbol like "NQ.v.0" or a parent symbol like "NQ.FUT"). This is the
// rule the execution layer enforces before it will build a Fill.
bool is_tradable_contract_symbol(std::string_view symbol) noexcept;

}  // namespace quant
