#include "quant_core/contract.hpp"

namespace quant {

bool is_continuous_symbol(std::string_view symbol) noexcept {
    // Any of the explicit roll-rule tokens.
    if (symbol.find(".c.") != std::string_view::npos) return true;
    if (symbol.find(".v.") != std::string_view::npos) return true;
    if (symbol.find(".n.") != std::string_view::npos) return true;

    // Compact form "<ROOT>.<rule>.<rank>" -- exactly two dots, single-letter
    // rule in {c,v,n}, all-digit rank.
    const auto d1 = symbol.find('.');
    if (d1 == std::string_view::npos) return false;
    const auto d2 = symbol.find('.', d1 + 1);
    if (d2 == std::string_view::npos) return false;
    if (symbol.find('.', d2 + 1) != std::string_view::npos) return false;

    const std::string_view rule = symbol.substr(d1 + 1, d2 - d1 - 1);
    const std::string_view rank = symbol.substr(d2 + 1);
    if (rule.size() != 1) return false;
    if (rule[0] != 'c' && rule[0] != 'v' && rule[0] != 'n') return false;
    if (rank.empty()) return false;
    for (const char c : rank) {
        if (c < '0' || c > '9') return false;
    }
    return true;
}

bool is_tradable_contract_symbol(std::string_view symbol) noexcept {
    if (symbol.empty()) return false;
    // Real CME futures symbols (e.g. "NQZ6", "ESH25") contain no '.'. A dot means
    // a continuous ("NQ.v.0") or parent ("NQ.FUT") symbol, neither of which is a
    // single tradable instrument.
    if (symbol.find('.') != std::string_view::npos) return false;
    return !is_continuous_symbol(symbol);
}

}  // namespace quant
