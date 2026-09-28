#include "quant_core/contract_registry.hpp"

#include <stdexcept>
#include <string>

namespace quant {

void ContractRegistry::add(ContractSpec spec) {
    if (!spec.is_valid()) {
        throw std::invalid_argument("ContractRegistry::add: structurally invalid ContractSpec for '" +
                                    spec.raw_symbol + "'");
    }
    if (!is_tradable_contract_symbol(spec.raw_symbol)) {
        throw std::invalid_argument("ContractRegistry::add: raw_symbol is not a tradable contract: '" +
                                    spec.raw_symbol + "'");
    }
    if (by_id_.count(spec.instrument_id) != 0) {
        throw std::invalid_argument("ContractRegistry::add: duplicate instrument_id " +
                                    std::to_string(spec.instrument_id));
    }
    if (id_by_symbol_.count(spec.raw_symbol) != 0) {
        throw std::invalid_argument("ContractRegistry::add: duplicate raw_symbol '" + spec.raw_symbol +
                                    "'");
    }
    const std::uint32_t id = spec.instrument_id;
    const std::string sym = spec.raw_symbol;
    by_id_.emplace(id, std::move(spec));
    id_by_symbol_.emplace(sym, id);
}

const ContractSpec* ContractRegistry::find_by_instrument_id(std::uint32_t id) const noexcept {
    const auto it = by_id_.find(id);
    return it == by_id_.end() ? nullptr : &it->second;
}

const ContractSpec& ContractRegistry::by_instrument_id(std::uint32_t id) const {
    const auto it = by_id_.find(id);
    if (it == by_id_.end()) {
        throw std::out_of_range("ContractRegistry::by_instrument_id: unknown instrument_id " +
                                std::to_string(id));
    }
    return it->second;
}

const ContractSpec* ContractRegistry::find_by_raw_symbol(std::string_view raw_symbol) const noexcept {
    const auto it = id_by_symbol_.find(std::string(raw_symbol));
    if (it == id_by_symbol_.end()) return nullptr;
    return find_by_instrument_id(it->second);
}

const ContractSpec* ContractRegistry::latest_live_contract(std::string_view root,
                                                           std::int64_t ts_ns) const noexcept {
    const ContractSpec* best = nullptr;
    for (const auto& [id, spec] : by_id_) {
        (void)id;
        if (spec.root_symbol != root) continue;
        if (!spec.is_live_at(ts_ns)) continue;
        if (best == nullptr || spec.activation_ns > best->activation_ns) {
            best = &spec;
        }
    }
    return best;
}

}  // namespace quant
