#pragma once

// ============================================================================
// Shared boundary-file readers for the ADDITIVE replay CLIs (News Alpha Phase H)
// ============================================================================
//
// The readers below parse the frozen Phase 02.5 / 13.1 / 13.5C / Phase 6 /
// Phase 21 boundary files with EXACTLY the semantics of the copies that live in
// apps/backtest_targets_csv.cpp (the frozen reference CLI) and
// apps/paper_trading_targets_csv.cpp. Those two files keep their own copies on
// purpose -- they are byte-stable reference paths and are not rewritten to use
// this header. New CLIs (apps/portfolio_plan_replay_csv.cpp) include this header
// instead of adding a third and fourth copy.
//
// Parsing only: nothing here adds or changes execution, accounting, fill, roll,
// corporate-action or risk semantics. Every file is machine-generated; every
// malformed input is a loud std::runtime_error.

#include "quant_core/corporate_actions.hpp"
#include "quant_core/detail/csv.hpp"
#include "quant_core/risk_config.hpp"
#include "quant_core/types.hpp"

#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quant::detail {

// bars.csv -- header exactly `ts_event_ns,instrument_id,open,high,low,close,volume`.
inline std::vector<MarketBar> read_bars_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open bars csv '" + path + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty bars csv");
    const std::vector<std::string> header{"ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume"};
    if (split_csv_line(line) != header) {
        throw std::runtime_error("bars csv header must be exactly "
                                 "'ts_event_ns,instrument_id,open,high,low,close,volume'");
    }
    std::vector<MarketBar> bars;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = split_csv_line(line);
        if (f.size() != header.size()) throw std::runtime_error("bars csv: malformed row");
        bars.push_back(MarketBar{
            .ts_event_ns = static_cast<std::int64_t>(std::stoll(f[0])),
            .instrument_id = static_cast<std::uint32_t>(std::stoul(f[1])),
            .open = std::stod(f[2]),
            .high = std::stod(f[3]),
            .low = std::stod(f[4]),
            .close = std::stod(f[5]),
            .volume = static_cast<std::int64_t>(std::stoll(f[6]))});
    }
    return bars;
}

// validation_days.csv -- header exactly `boundary_ts_ns`, strictly ascending.
inline std::vector<std::int64_t> read_validation_days_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open validation-days csv '" + path + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty validation-days csv");
    const auto header = split_csv_line(line);
    if (header.size() != 1 || header[0] != "boundary_ts_ns") {
        throw std::runtime_error("validation-days csv header must be exactly 'boundary_ts_ns'");
    }
    std::vector<std::int64_t> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const std::int64_t v = static_cast<std::int64_t>(std::stoll(line));
        if (!out.empty() && v <= out.back()) {
            throw std::runtime_error("validation-days csv must be strictly ascending");
        }
        out.push_back(v);
    }
    return out;
}

// roll_close_marks.csv -- header exactly `instrument_id,ts_event_ns,close`;
// a duplicate key is a loud error.
inline std::map<std::pair<std::uint32_t, std::int64_t>, double> read_roll_close_marks_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open roll-close-marks csv '" + path + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty roll-close-marks csv");
    const auto header = split_csv_line(line);
    if (header.size() != 3 || header[0] != "instrument_id" || header[1] != "ts_event_ns" || header[2] != "close") {
        throw std::runtime_error("roll-close-marks csv header must be exactly 'instrument_id,ts_event_ns,close'");
    }
    std::map<std::pair<std::uint32_t, std::int64_t>, double> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = split_csv_line(line);
        if (f.size() != 3) continue;
        const auto key = std::pair<std::uint32_t, std::int64_t>{static_cast<std::uint32_t>(std::stoul(f[0])),
                                                                 static_cast<std::int64_t>(std::stoll(f[1]))};
        if (!out.emplace(key, std::stod(f[2])).second) {
            throw std::runtime_error("roll-close-marks csv: duplicate (instrument_id, ts_event_ns) for instrument_id " +
                                     f[0] + " at ts " + f[1]);
        }
    }
    return out;
}

// splits.csv -- header exactly `instrument_id,effective_ts_ns,ratio`; a
// duplicate key is a loud error.
inline std::map<std::pair<std::uint32_t, std::int64_t>, SplitAction> read_splits_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open splits csv '" + path + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty splits csv");
    const auto header = split_csv_line(line);
    if (header.size() != 3 || header[0] != "instrument_id" || header[1] != "effective_ts_ns" || header[2] != "ratio") {
        throw std::runtime_error("splits csv header must be exactly 'instrument_id,effective_ts_ns,ratio'");
    }
    std::map<std::pair<std::uint32_t, std::int64_t>, SplitAction> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = split_csv_line(line);
        if (f.size() != 3) continue;
        SplitAction action;
        action.instrument_id = static_cast<std::uint32_t>(std::stoul(f[0]));
        action.effective_ts_ns = static_cast<std::int64_t>(std::stoll(f[1]));
        action.ratio = std::stod(f[2]);
        if (!out.emplace(std::pair{action.instrument_id, action.effective_ts_ns}, action).second) {
            throw std::runtime_error("splits csv: duplicate (instrument_id, effective_ts_ns) for instrument_id " + f[0] +
                                     " at ts " + f[1]);
        }
    }
    return out;
}

// risk_config.csv -- the Phase 21 16-column RiskConfig row
// (alpha_agent.paper.risk_policy.PaperRiskPolicy.csv_rows).
inline RiskConfig read_risk_config_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open risk-config csv '" + path + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty risk-config csv");
    const std::vector<std::string> header{
        "max_contracts_per_symbol", "max_contracts_per_root", "max_gross_contracts",
        "max_order_contracts", "max_gross_exposure_usd", "max_gross_leverage",
        "max_net_leverage", "max_margin_utilization_pct", "missing_margin_policy",
        "max_daily_loss_usd", "max_drawdown_pct", "max_drawdown_usd", "stale_mark_policy",
        "starting_capital_usd", "mark_staleness_tolerance_ns", "day_boundary_offset_ns"};
    if (split_csv_line(line) != header) throw std::runtime_error("risk-config csv: header mismatch");
    if (!std::getline(in, line) || line.empty()) throw std::runtime_error("risk-config csv: missing data row");
    const auto f = split_csv_line(line);
    if (f.size() != header.size()) throw std::runtime_error("risk-config csv: malformed data row");
    RiskConfig cfg;
    cfg.max_contracts_per_symbol = std::stoi(f[0]);
    cfg.max_contracts_per_root = std::stoi(f[1]);
    cfg.max_gross_contracts = std::stoi(f[2]);
    cfg.max_order_contracts = std::stoi(f[3]);
    cfg.max_gross_exposure_usd = std::stod(f[4]);
    cfg.max_gross_leverage = std::stod(f[5]);
    cfg.max_net_leverage = std::stod(f[6]);
    cfg.max_margin_utilization_pct = std::stod(f[7]);
    if (f[8] == "reject") {
        cfg.missing_margin = MissingMarginPolicy::Reject;
    } else if (f[8] == "treat_as_zero") {
        cfg.missing_margin = MissingMarginPolicy::TreatAsZero;
    } else {
        throw std::runtime_error("risk-config csv: bad missing_margin_policy '" + f[8] + "'");
    }
    cfg.max_daily_loss_usd = std::stod(f[9]);
    cfg.max_drawdown_pct = std::stod(f[10]);
    cfg.max_drawdown_usd = std::stod(f[11]);
    if (f[12] == "reject_risk_increasing") {
        cfg.stale_mark = StaleMarkPolicy::RejectRiskIncreasing;
    } else if (f[12] == "ignore") {
        cfg.stale_mark = StaleMarkPolicy::Ignore;
    } else {
        throw std::runtime_error("risk-config csv: bad stale_mark_policy '" + f[12] + "'");
    }
    cfg.portfolio.starting_capital_usd = std::stod(f[13]);
    cfg.portfolio.mark_staleness_tolerance_ns = static_cast<std::int64_t>(std::stoll(f[14]));
    cfg.portfolio.day_boundary_offset_ns = static_cast<std::int64_t>(std::stoll(f[15]));
    return cfg;
}

}  // namespace quant::detail
