#include "quant_core/contract_io.hpp"

#include "quant_core/detail/csv.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace quant {
namespace {

constexpr std::array<const char*, 10> kHeader{
    "instrument_id", "raw_symbol", "root_symbol", "exchange", "tick_size",
    "multiplier", "activation_ns", "expiration_ns", "first_notice_ns", "last_trade_ns",
};

std::optional<std::int64_t> parse_opt_i64(const std::string& s) {
    if (s.empty()) return std::nullopt;
    return static_cast<std::int64_t>(std::stoll(s));
}

}  // namespace

std::vector<ContractSpec> parse_contracts_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("parse_contracts_csv: cannot open '" + path + "'");

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("parse_contracts_csv: empty file '" + path + "'");
    }
    const auto header = detail::split_csv_line(line);
    if (header.size() != kHeader.size()) {
        throw std::runtime_error("parse_contracts_csv: expected " +
                                 std::to_string(kHeader.size()) + " columns, got " +
                                 std::to_string(header.size()));
    }
    for (std::size_t i = 0; i < kHeader.size(); ++i) {
        if (header[i] != kHeader[i]) {
            throw std::runtime_error("parse_contracts_csv: column " + std::to_string(i) +
                                     " should be '" + kHeader[i] + "', got '" + header[i] + "'");
        }
    }

    std::vector<ContractSpec> out;
    std::size_t row = 1;
    while (std::getline(in, line)) {
        ++row;
        if (line.empty()) continue;
        const auto f = detail::split_csv_line(line);
        if (f.size() != kHeader.size()) {
            throw std::runtime_error("parse_contracts_csv: row " + std::to_string(row) +
                                     " has " + std::to_string(f.size()) + " fields");
        }
        try {
            ContractSpec spec;
            spec.instrument_id  = static_cast<std::uint32_t>(std::stoul(f[0]));
            spec.raw_symbol     = f[1];
            spec.root_symbol    = f[2];
            spec.exchange       = f[3];
            spec.tick_size      = std::stod(f[4]);
            spec.multiplier     = std::stod(f[5]);
            spec.activation_ns  = static_cast<std::int64_t>(std::stoll(f[6]));
            spec.expiration_ns  = static_cast<std::int64_t>(std::stoll(f[7]));
            spec.first_notice_ns = parse_opt_i64(f[8]);
            spec.last_trade_ns   = parse_opt_i64(f[9]);
            out.push_back(std::move(spec));
        } catch (const std::exception& e) {
            throw std::runtime_error("parse_contracts_csv: row " + std::to_string(row) +
                                     " unparseable: " + e.what());
        }
    }
    return out;
}

ContractRegistry load_contract_registry(const std::string& path) {
    ContractRegistry registry;
    for (auto& spec : parse_contracts_csv(path)) {
        registry.add(std::move(spec));
    }
    return registry;
}

}  // namespace quant
