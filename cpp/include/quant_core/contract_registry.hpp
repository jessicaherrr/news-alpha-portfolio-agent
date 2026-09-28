#pragma once

#include "quant_core/contract.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace quant {

// Lookup table of real contracts, keyed by Databento instrument id.
//
// The backtester uses this to turn the `instrument_id` on a bar (or on a
// proposed fill) into a full ContractSpec, so every simulated fill can be tied
// to the actual tradable contract that was live at that timestamp.
class ContractRegistry {
public:
    // Throws std::invalid_argument if `spec` is structurally invalid or if its
    // instrument_id / raw_symbol collides with an existing entry.
    void add(ContractSpec spec);

    const ContractSpec* find_by_instrument_id(std::uint32_t id) const noexcept;
    const ContractSpec& by_instrument_id(std::uint32_t id) const;  // throws std::out_of_range
    const ContractSpec* find_by_raw_symbol(std::string_view raw_symbol) const noexcept;

    // METADATA QUERY ONLY -- not an active-contract selector.
    //
    // Returns the contract of `root` live at `ts_ns` with the latest activation
    // (nullptr if none). This is an activation-date heuristic: around a roll
    // several contracts of one root are simultaneously live, so this cannot know
    // which one a continuous feed is actually trading. The ACTIVE contract comes
    // from current market state via ActiveContractResolver (contract_selector.hpp).
    // Use this only for diagnostics / coverage checks, never to route an Order.
    const ContractSpec* latest_live_contract(std::string_view root,
                                             std::int64_t ts_ns) const noexcept;

    std::size_t size() const noexcept { return by_id_.size(); }
    bool empty() const noexcept { return by_id_.empty(); }

private:
    std::unordered_map<std::uint32_t, ContractSpec> by_id_;
    std::unordered_map<std::string, std::uint32_t> id_by_symbol_;
};

}  // namespace quant
