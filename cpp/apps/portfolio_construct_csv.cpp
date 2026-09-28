// News Alpha Phase G CLI: deterministic multi-asset portfolio construction.
//
// Usage:
//   quant_portfolio_construct_csv <input_dir>
//
// <input_dir> holds four machine-written CSVs (alpha_agent.portfolio.allocator
// writes them; never hand-edited, never proposed by an LLM):
//
//   instruments.csv  instrument_key,domain,root_symbol,asset_class,sector,price,
//                    multiplier,annual_vol,adv_usd,max_units,decision_ts_ns
//   signals.csv      signal_id,instrument_key,cluster_id,direction,priority
//   correlations.csv instrument_a,instrument_b,correlation   (a < b)
//   limits.csv       one row: capital_usd,target_annual_vol,max_gross_leverage,
//                    max_net_exposure,max_instrument_gross_share,
//                    max_sector_gross_share,max_asset_class_gross_share,
//                    max_cluster_risk_share,max_adv_participation,min_adv_usd,
//                    max_turnover,shorting_allowed,allowed_domains
//                    (allowed_domains pipe-separated, e.g. FUTURES|ETF)
//
// and optionally previous_units.csv (instrument_key,units) for turnover.
//
// Output: one JSON line (quant::ConstructionResult). Every number is printed
// with %.17g so the Python side reads back exactly what C++ computed -- it
// never recomputes a notional, a unit or a risk contribution. Exit 1 with an
// `error:` line on stderr for a malformed input.
//
// This is an ADDITIVE Phase G path: it builds no Order, no Fill and touches
// no engine, execution, accounting or risk-manager code.

#include "quant_core/detail/csv.hpp"
#include "quant_core/portfolio_construction.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using quant::detail::split_csv_line;

std::vector<std::vector<std::string>> read_csv(const std::filesystem::path& path,
                                               const std::vector<std::string>& header) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open '" + path.string() + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty csv '" + path.string() + "'");
    if (split_csv_line(line) != header) {
        std::string want;
        for (const auto& h : header) want += (want.empty() ? "" : ",") + h;
        throw std::runtime_error(path.filename().string() + ": header must be exactly '" + want + "'");
    }
    std::vector<std::vector<std::string>> rows;
    while (std::getline(in, line)) {
        if (line.empty() || line == "\r") continue;
        auto f = split_csv_line(line);
        if (f.size() != header.size()) {
            throw std::runtime_error(path.filename().string() + ": row has " + std::to_string(f.size()) +
                                     " fields, expected " + std::to_string(header.size()));
        }
        rows.push_back(std::move(f));
    }
    return rows;
}

double num(const std::string& s, const char* field) {
    try {
        std::size_t used = 0;
        const double v = std::stod(s, &used);
        if (used != s.size()) throw std::invalid_argument(s);
        return v;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("field '") + field + "' is not a number: '" + s + "'");
    }
}

long long integer(const std::string& s, const char* field) {
    try {
        std::size_t used = 0;
        const long long v = std::stoll(s, &used);
        if (used != s.size()) throw std::invalid_argument(s);
        return v;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("field '") + field + "' is not an integer: '" + s + "'");
    }
}

quant::ConstructionInput read_input(const std::filesystem::path& dir) {
    quant::ConstructionInput in;
    for (const auto& f : read_csv(dir / "instruments.csv",
                                  {"instrument_key", "domain", "root_symbol", "asset_class", "sector", "price",
                                   "multiplier", "annual_vol", "adv_usd", "max_units", "decision_ts_ns"})) {
        quant::ConstructionInstrument i;
        i.key = f[0];
        i.domain = f[1];
        i.root_symbol = f[2];
        i.asset_class = f[3];
        i.sector = f[4];
        i.price = num(f[5], "price");
        i.multiplier = num(f[6], "multiplier");
        i.annual_vol = num(f[7], "annual_vol");
        i.adv_usd = num(f[8], "adv_usd");
        i.max_units = static_cast<int>(integer(f[9], "max_units"));
        i.decision_ts_ns = integer(f[10], "decision_ts_ns");
        in.instruments.push_back(std::move(i));
    }
    for (const auto& f : read_csv(dir / "signals.csv",
                                  {"signal_id", "instrument_key", "cluster_id", "direction", "priority"})) {
        in.signals.push_back(quant::ConstructionSignal{f[0], f[1], f[2], static_cast<int>(integer(f[3], "direction")),
                                                       static_cast<int>(integer(f[4], "priority"))});
    }
    for (const auto& f : read_csv(dir / "correlations.csv", {"instrument_a", "instrument_b", "correlation"})) {
        if (!in.correlations.emplace(std::make_pair(f[0], f[1]), num(f[2], "correlation")).second) {
            throw std::runtime_error("correlations.csv: duplicate pair (" + f[0] + ", " + f[1] + ")");
        }
    }
    const auto limits = read_csv(dir / "limits.csv",
                                 {"capital_usd", "target_annual_vol", "max_gross_leverage", "max_net_exposure",
                                  "max_instrument_gross_share", "max_sector_gross_share",
                                  "max_asset_class_gross_share", "max_cluster_risk_share", "max_adv_participation",
                                  "min_adv_usd", "max_turnover", "shorting_allowed", "allowed_domains"});
    if (limits.size() != 1) throw std::runtime_error("limits.csv must hold exactly one data row");
    const auto& l = limits[0];
    auto& lim = in.limits;
    lim.capital_usd = num(l[0], "capital_usd");
    lim.target_annual_vol = num(l[1], "target_annual_vol");
    lim.max_gross_leverage = num(l[2], "max_gross_leverage");
    lim.max_net_exposure = num(l[3], "max_net_exposure");
    lim.max_instrument_gross_share = num(l[4], "max_instrument_gross_share");
    lim.max_sector_gross_share = num(l[5], "max_sector_gross_share");
    lim.max_asset_class_gross_share = num(l[6], "max_asset_class_gross_share");
    lim.max_cluster_risk_share = num(l[7], "max_cluster_risk_share");
    lim.max_adv_participation = num(l[8], "max_adv_participation");
    lim.min_adv_usd = num(l[9], "min_adv_usd");
    lim.max_turnover = num(l[10], "max_turnover");
    if (l[11] != "true" && l[11] != "false") throw std::runtime_error("shorting_allowed must be true or false");
    lim.shorting_allowed = l[11] == "true";
    std::stringstream domains(l[12]);
    for (std::string d; std::getline(domains, d, '|');) {
        if (!d.empty()) lim.allowed_domains.push_back(d);
    }
    const auto prev = dir / "previous_units.csv";
    if (std::filesystem::exists(prev)) {
        for (const auto& f : read_csv(prev, {"instrument_key", "units"})) {
            if (!in.previous_units.emplace(f[0], static_cast<int>(integer(f[1], "units"))).second) {
                throw std::runtime_error("previous_units.csv: duplicate instrument '" + f[0] + "'");
            }
        }
    }
    return in;
}

// ---- JSON ------------------------------------------------------------------

std::string g(double v) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

std::string q(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) < 0x20) continue;
        out += c;
    }
    return out + "\"";
}

std::string b(bool v) { return v ? "true" : "false"; }

std::string strings(const std::vector<std::string>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + q(v[i]);
    return out + "]";
}

std::string stats(const quant::PortfolioStats& s) {
    return "{\"gross_exposure\":" + g(s.gross_exposure) + ",\"net_exposure\":" + g(s.net_exposure) +
           ",\"long_exposure\":" + g(s.long_exposure) + ",\"short_exposure\":" + g(s.short_exposure) +
           ",\"annual_vol\":" + g(s.annual_vol) + ",\"positions\":" + std::to_string(s.positions) +
           ",\"effective_exposures\":" + g(s.effective_exposures) +
           ",\"max_cluster_risk_share\":" + g(s.max_cluster_risk_share) + "}";
}

void write_json(std::ostream& os, const quant::ConstructionResult& r) {
    os << "{\"method\":" << q(r.method) << ",\"status\":" << q(r.status)
       << ",\"capital_usd\":" << g(r.capital_usd) << ",\"target_annual_vol\":" << g(r.target_annual_vol);
    os << ",\"instruments\":[";
    for (std::size_t i = 0; i < r.instruments.size(); ++i) {
        const auto& a = r.instruments[i];
        const auto& in = a.input;
        os << (i ? "," : "") << "{\"instrument_key\":" << q(in.key) << ",\"domain\":" << q(in.domain)
           << ",\"root_symbol\":" << q(in.root_symbol) << ",\"asset_class\":" << q(in.asset_class)
           << ",\"sector\":" << q(in.sector) << ",\"price\":" << g(in.price) << ",\"multiplier\":" << g(in.multiplier)
           << ",\"unit_notional_usd\":" << g(in.unit_notional_usd()) << ",\"annual_vol\":" << g(in.annual_vol)
           << ",\"adv_usd\":" << g(in.adv_usd) << ",\"max_units\":" << in.max_units
           << ",\"decision_ts_ns\":" << in.decision_ts_ns << ",\"admitted\":" << b(a.admitted)
           << ",\"reason\":" << q(a.reason) << ",\"target_weight\":" << g(a.target_weight)
           << ",\"constrained_weight\":" << g(a.constrained_weight) << ",\"previous_units\":" << a.previous_units
           << ",\"units\":" << a.units << ",\"executable_weight\":" << g(a.executable_weight)
           << ",\"target_notional_usd\":" << g(a.target_notional_usd)
           << ",\"executable_notional_usd\":" << g(a.executable_notional_usd)
           << ",\"rounding_residual_usd\":" << g(a.rounding_residual_usd)
           << ",\"adv_participation\":" << g(a.adv_participation)
           << ",\"risk_contribution\":" << g(a.risk_contribution)
           << ",\"unit_risk_fraction\":" << g(a.unit_risk_fraction)
           << ",\"below_one_unit\":" << b(a.below_one_unit) << "}";
    }
    os << "],\"signals\":[";
    for (std::size_t i = 0; i < r.signals.size(); ++i) {
        const auto& s = r.signals[i];
        os << (i ? "," : "") << "{\"signal_id\":" << q(s.input.signal_id)
           << ",\"instrument_key\":" << q(s.input.instrument_key) << ",\"cluster_id\":" << q(s.input.cluster_id)
           << ",\"direction\":" << s.input.direction << ",\"priority\":" << s.input.priority
           << ",\"allocated\":" << b(s.allocated) << ",\"reason\":" << q(s.reason)
           << ",\"target_weight\":" << g(s.target_weight) << ",\"executable_weight\":" << g(s.executable_weight)
           << ",\"risk_contribution\":" << g(s.risk_contribution) << "}";
    }
    os << "],\"clusters\":[";
    for (std::size_t i = 0; i < r.clusters.size(); ++i) {
        const auto& c = r.clusters[i];
        os << (i ? "," : "") << "{\"cluster_id\":" << q(c.cluster_id) << ",\"signal_ids\":" << strings(c.signal_ids)
           << ",\"allocated\":" << b(c.allocated) << ",\"reason\":" << q(c.reason)
           << ",\"composite_vol\":" << g(c.composite_vol) << ",\"erc_weight\":" << g(c.erc_weight)
           << ",\"target_risk_contribution\":" << g(c.target_risk_contribution)
           << ",\"executable_risk_contribution\":" << g(c.executable_risk_contribution) << "}";
    }
    os << "],\"constraints\":[";
    for (std::size_t i = 0; i < r.constraints.size(); ++i) {
        const auto& c = r.constraints[i];
        os << (i ? "," : "") << "{\"name\":" << q(c.name) << ",\"scope\":" << q(c.scope)
           << ",\"stage\":" << q(c.stage) << ",\"status\":" << q(quant::to_string(c.status))
           << ",\"limit\":" << g(c.limit) << ",\"before\":" << g(c.before) << ",\"after\":" << g(c.after)
           << ",\"affected\":" << strings(c.affected) << "}";
    }
    os << "],\"target\":" << stats(r.target) << ",\"constrained\":" << stats(r.constrained)
       << ",\"executable\":" << stats(r.executable) << ",\"erc_sweeps\":" << r.erc_sweeps
       << ",\"erc_max_deviation\":" << g(r.erc_max_deviation) << ",\"turnover_target\":" << g(r.turnover_target)
       << ",\"turnover_executable\":" << g(r.turnover_executable)
       << ",\"repair_units_removed\":" << r.repair_units_removed << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: quant_portfolio_construct_csv <input_dir>\n"
                     "  <input_dir>: instruments.csv, signals.csv, correlations.csv, limits.csv "
                     "[, previous_units.csv]\n";
        return 2;
    }
    try {
        const auto result = quant::construct_portfolio(read_input(argv[1]));
        write_json(std::cout, result);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
