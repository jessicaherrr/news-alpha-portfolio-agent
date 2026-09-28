#include "quant_core/margin.hpp"

#include "quant_core/detail/csv.hpp"

#include <array>
#include <fstream>
#include <stdexcept>

namespace quant {

void MarginModel::set(MarginRequirement req) {
    if (!req.is_valid()) {
        throw std::invalid_argument(
            "MarginModel::set: invalid MarginRequirement for root '" + req.root_symbol +
            "' (need non-empty root, initial > 0, 0 < maintenance <= initial)");
    }
    by_root_[req.root_symbol] = std::move(req);
}

bool MarginModel::has(std::string_view root_symbol) const noexcept {
    return by_root_.find(root_symbol) != by_root_.end();
}

const MarginRequirement* MarginModel::find(std::string_view root_symbol) const noexcept {
    const auto it = by_root_.find(root_symbol);
    return it == by_root_.end() ? nullptr : &it->second;
}

std::optional<double> MarginModel::initial_per_contract(std::string_view root_symbol) const noexcept {
    const auto it = by_root_.find(root_symbol);
    if (it == by_root_.end()) return std::nullopt;
    return it->second.initial_margin_usd;
}

std::optional<double> MarginModel::maintenance_per_contract(
    std::string_view root_symbol) const noexcept {
    const auto it = by_root_.find(root_symbol);
    if (it == by_root_.end()) return std::nullopt;
    return it->second.maintenance_margin_usd;
}

std::vector<MarginRequirement> MarginModel::requirements() const {
    std::vector<MarginRequirement> out;
    out.reserve(by_root_.size());
    for (const auto& [root, req] : by_root_) {
        (void)root;
        out.push_back(req);
    }
    return out;  // by_root_ is ordered by root
}

MarginModel load_margin_model(const std::string& path) {
    static constexpr std::array<const char*, 5> kHeader{
        "root_symbol", "initial_margin_usd", "maintenance_margin_usd", "source", "as_of_ns",
    };

    std::ifstream in(path);
    if (!in) throw std::runtime_error("load_margin_model: cannot open '" + path + "'");

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("load_margin_model: empty file '" + path + "'");
    }
    const auto header = detail::split_csv_line(line);
    if (header.size() != kHeader.size()) {
        throw std::runtime_error("load_margin_model: expected " + std::to_string(kHeader.size()) +
                                 " columns, got " + std::to_string(header.size()));
    }
    for (std::size_t i = 0; i < kHeader.size(); ++i) {
        if (header[i] != kHeader[i]) {
            throw std::runtime_error("load_margin_model: column " + std::to_string(i) +
                                     " should be '" + kHeader[i] + "', got '" + header[i] + "'");
        }
    }

    MarginModel model;
    std::size_t row = 1;
    while (std::getline(in, line)) {
        ++row;
        if (line.empty()) continue;
        const auto f = detail::split_csv_line(line);
        if (f.size() != kHeader.size()) {
            throw std::runtime_error("load_margin_model: row " + std::to_string(row) + " has " +
                                     std::to_string(f.size()) + " fields");
        }
        try {
            MarginRequirement req;
            req.root_symbol             = f[0];
            req.initial_margin_usd      = std::stod(f[1]);
            req.maintenance_margin_usd  = std::stod(f[2]);
            req.source                  = f[3];
            req.as_of_ns                = static_cast<std::int64_t>(std::stoll(f[4]));
            model.set(std::move(req));  // re-validates
        } catch (const std::exception& e) {
            throw std::runtime_error("load_margin_model: row " + std::to_string(row) +
                                     " unusable: " + e.what());
        }
    }
    return model;
}

}  // namespace quant
