#include "quant_core/contract_selector.hpp"

#include "quant_core/domain_errors.hpp"

#include <string>

namespace quant {

const ContractSpec& RegistryActiveContractResolver::resolve(std::string_view root_symbol,
                                                            const MarketState& state) const {
    if (state.active_instrument_id == 0) {
        throw ContractResolutionError(
            "ActiveContractResolver: current market state has no active contract "
            "(instrument_id 0); cannot resolve a Signal for root '" +
            std::string(root_symbol) + "'");
    }

    const ContractSpec* spec = registry_.find_by_instrument_id(state.active_instrument_id);
    if (spec == nullptr) {
        throw ContractResolutionError(
            "ActiveContractResolver: market-state instrument_id " +
            std::to_string(state.active_instrument_id) +
            " does not resolve through the ContractRegistry");
    }

    if (spec->root_symbol != root_symbol) {
        throw ContractResolutionError(
            "ActiveContractResolver: Signal root '" + std::string(root_symbol) +
            "' does not match the active feed contract '" + spec->raw_symbol +
            "' (root '" + spec->root_symbol + "')");
    }

    if (!spec->is_live_at(state.as_of_ts_ns)) {
        throw ContractResolutionError(
            "ActiveContractResolver: active contract '" + spec->raw_symbol +
            "' is not tradable at ts " + std::to_string(state.as_of_ts_ns));
    }

    return *spec;
}

}  // namespace quant
