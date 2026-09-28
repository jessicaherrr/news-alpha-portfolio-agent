// Phase 21 reference CLI: replay a precomputed Python target schedule through
// the deterministic C++ BacktestEngine UNDER THE HARD PORTFOLIO RISK MANAGER
// (kill switches: drawdown, daily loss, position/exposure caps, stale-mark
// protection) instead of the research reference CLI's pass-through risk.
//
// Usage:
//   quant_paper_trading_targets_csv <bars.csv> <contracts.csv> <targets.csv> \
//       <risk_config.csv> [policy] [commission] [slippage] [spread] \
//       [validation_days.csv] [roll_close_marks.csv]
//       [--margin-config=<path>] [--trades-out=<path>] [--fills-out=<path>]
//       [--end-of-test=leave_open|force_liquidate]
//
// bars.csv / contracts.csv / targets.csv: identical frozen boundary formats to
// quant_backtest_targets_csv (apps/backtest_targets_csv.cpp) -- see that file's
// header for the full column contract.
//
// risk_config.csv (Phase 21, REQUIRED, positional arg 4): exactly one data row
// after a fixed header naming every quant::RiskConfig field this CLI wires
// (see kRiskConfigHeader below). Written deterministically by
// alpha_agent.paper.risk_policy.PaperRiskPolicy -- never edited by hand and
// never proposed by an LLM (CLAUDE.md risk rule 1: hard risk limits are
// deterministic and cannot be overridden by an LLM).
//
// --margin-config=<path> (optional) is the frozen quant::load_margin_model CSV
// (root_symbol,initial_margin_usd,maintenance_margin_usd,source,as_of_ns).
// Omitted => no MarginModel; RiskConfig.missing_margin should then be
// TreatAsZero (a Reject default with no margin data blocks every opening
// order in every root).
//
// --end-of-test= (optional, default leave_open) selects
// EndOfTestPolicy::LeaveOpen (a paper position must normally survive into the
// next step's replay) or ForceLiquidateFinalClose (the one deliberate "flatten
// and stop this paper run" step).
//
// --trades-out= / --fills-out= (optional) are the identical Phase 15A
// read-only CSV audit exports quant_backtest_targets_csv already supports.
//
// This is a NEW, ADDITIVE reference path (Phase 21). It shares no code with
// quant_backtest_targets_csv, which is unchanged, still wired to
// PassThroughRiskManager, and remains the frozen historical research CLI.

#include "quant_core/contract_io.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/detail/csv.hpp"
#include "quant_core/margin.hpp"
#include "quant_core/paper_trading_run.hpp"
#include "quant_core/risk_config.hpp"
#include "quant_core/scheduled_target_strategy.hpp"
#include "quant_core/trade_export.hpp"
#include "quant_core/types.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::array<const char*, 7> kBarHeader{
    "ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume",
};

std::vector<quant::MarketBar> read_bars(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open bars csv '" + path + "'");

    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty bars csv");
    const auto header = quant::detail::split_csv_line(line);
    if (header.size() != kBarHeader.size()) {
        throw std::runtime_error("bars csv: expected 7 columns, got " +
                                 std::to_string(header.size()));
    }
    for (std::size_t i = 0; i < kBarHeader.size(); ++i) {
        if (header[i] != kBarHeader[i]) {
            throw std::runtime_error("bars csv: column " + std::to_string(i) + " should be '" +
                                     kBarHeader[i] + "', got '" + header[i] + "'");
        }
    }

    std::vector<quant::MarketBar> bars;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = quant::detail::split_csv_line(line);
        if (f.size() != kBarHeader.size()) continue;
        bars.push_back(quant::MarketBar{
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

std::vector<std::int64_t> read_validation_days(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open validation-days csv '" + path + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty validation-days csv");
    const auto header = quant::detail::split_csv_line(line);
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

std::map<std::pair<std::uint32_t, std::int64_t>, double> read_roll_close_marks(
    const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open roll-close-marks csv '" + path + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty roll-close-marks csv");
    const auto header = quant::detail::split_csv_line(line);
    if (header.size() != 3 || header[0] != "instrument_id" || header[1] != "ts_event_ns" ||
        header[2] != "close") {
        throw std::runtime_error(
            "roll-close-marks csv header must be exactly 'instrument_id,ts_event_ns,close'");
    }
    std::map<std::pair<std::uint32_t, std::int64_t>, double> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = quant::detail::split_csv_line(line);
        if (f.size() != 3) continue;
        const auto key = std::pair<std::uint32_t, std::int64_t>{
            static_cast<std::uint32_t>(std::stoul(f[0])),
            static_cast<std::int64_t>(std::stoll(f[1]))};
        const double px = std::stod(f[2]);
        if (!out.emplace(key, px).second) {
            throw std::runtime_error(
                "roll-close-marks csv: duplicate (instrument_id, ts_event_ns) for instrument_id " +
                f[0] + " at ts " + f[1]);
        }
    }
    return out;
}

// Phase 21 risk-config CSV: exactly one data row after this fixed header.
constexpr std::array<const char*, 16> kRiskConfigHeader{
    "max_contracts_per_symbol", "max_contracts_per_root", "max_gross_contracts",
    "max_order_contracts", "max_gross_exposure_usd", "max_gross_leverage",
    "max_net_leverage", "max_margin_utilization_pct", "missing_margin_policy",
    "max_daily_loss_usd", "max_drawdown_pct", "max_drawdown_usd", "stale_mark_policy",
    "starting_capital_usd", "mark_staleness_tolerance_ns", "day_boundary_offset_ns",
};

quant::MissingMarginPolicy parse_missing_margin_policy(const std::string& s) {
    if (s == "reject") return quant::MissingMarginPolicy::Reject;
    if (s == "treat_as_zero") return quant::MissingMarginPolicy::TreatAsZero;
    throw std::runtime_error("risk_config.csv: missing_margin_policy must be "
                             "'reject' or 'treat_as_zero', got '" + s + "'");
}

quant::StaleMarkPolicy parse_stale_mark_policy(const std::string& s) {
    if (s == "reject_risk_increasing") return quant::StaleMarkPolicy::RejectRiskIncreasing;
    if (s == "ignore") return quant::StaleMarkPolicy::Ignore;
    throw std::runtime_error("risk_config.csv: stale_mark_policy must be "
                             "'reject_risk_increasing' or 'ignore', got '" + s + "'");
}

quant::RiskConfig read_risk_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open risk-config csv '" + path + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty risk-config csv");
    const auto header = quant::detail::split_csv_line(line);
    if (header.size() != kRiskConfigHeader.size()) {
        throw std::runtime_error("risk-config csv: expected " +
                                 std::to_string(kRiskConfigHeader.size()) + " columns, got " +
                                 std::to_string(header.size()));
    }
    for (std::size_t i = 0; i < kRiskConfigHeader.size(); ++i) {
        if (header[i] != kRiskConfigHeader[i]) {
            throw std::runtime_error("risk-config csv: column " + std::to_string(i) +
                                     " should be '" + std::string(kRiskConfigHeader[i]) +
                                     "', got '" + header[i] + "'");
        }
    }
    if (!std::getline(in, line) || line.empty()) {
        throw std::runtime_error("risk-config csv: missing the one required data row");
    }
    const auto f = quant::detail::split_csv_line(line);
    if (f.size() != kRiskConfigHeader.size()) {
        throw std::runtime_error("risk-config csv: data row has " + std::to_string(f.size()) +
                                 " fields, expected " + std::to_string(kRiskConfigHeader.size()));
    }
    quant::RiskConfig cfg;
    cfg.max_contracts_per_symbol = std::stoi(f[0]);
    cfg.max_contracts_per_root = std::stoi(f[1]);
    cfg.max_gross_contracts = std::stoi(f[2]);
    cfg.max_order_contracts = std::stoi(f[3]);
    cfg.max_gross_exposure_usd = std::stod(f[4]);
    cfg.max_gross_leverage = std::stod(f[5]);
    cfg.max_net_leverage = std::stod(f[6]);
    cfg.max_margin_utilization_pct = std::stod(f[7]);
    cfg.missing_margin = parse_missing_margin_policy(f[8]);
    cfg.max_daily_loss_usd = std::stod(f[9]);
    cfg.max_drawdown_pct = std::stod(f[10]);
    cfg.max_drawdown_usd = std::stod(f[11]);
    cfg.stale_mark = parse_stale_mark_policy(f[12]);
    cfg.portfolio.starting_capital_usd = std::stod(f[13]);
    cfg.portfolio.mark_staleness_tolerance_ns = static_cast<std::int64_t>(std::stoll(f[14]));
    cfg.portfolio.day_boundary_offset_ns = static_cast<std::int64_t>(std::stoll(f[15]));
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    std::string trades_out_path;
    std::string fills_out_path;
    std::string margin_config_path;
    std::string end_of_test_str = "leave_open";
    std::vector<char*> pos;
    pos.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        static constexpr const char* kTradesFlag = "--trades-out=";
        static constexpr const char* kFillsFlag = "--fills-out=";
        static constexpr const char* kMarginFlag = "--margin-config=";
        static constexpr const char* kEotFlag = "--end-of-test=";
        if (i > 0 && std::strncmp(argv[i], kTradesFlag, std::strlen(kTradesFlag)) == 0) {
            trades_out_path = argv[i] + std::strlen(kTradesFlag);
            continue;
        }
        if (i > 0 && std::strncmp(argv[i], kFillsFlag, std::strlen(kFillsFlag)) == 0) {
            fills_out_path = argv[i] + std::strlen(kFillsFlag);
            continue;
        }
        if (i > 0 && std::strncmp(argv[i], kMarginFlag, std::strlen(kMarginFlag)) == 0) {
            margin_config_path = argv[i] + std::strlen(kMarginFlag);
            continue;
        }
        if (i > 0 && std::strncmp(argv[i], kEotFlag, std::strlen(kEotFlag)) == 0) {
            end_of_test_str = argv[i] + std::strlen(kEotFlag);
            continue;
        }
        pos.push_back(argv[i]);
    }
    argc = static_cast<int>(pos.size());
    argv = pos.data();

    if (argc < 5) {
        std::cerr
            << "usage: quant_paper_trading_targets_csv <bars.csv> <contracts.csv> "
               "<targets.csv> <risk_config.csv> "
               "[no_decision|flat|require_row|reemit_previous] "
               "[commission_per_contract_usd] [slippage_ticks] [spread_ticks] "
               "[validation_days.csv] [roll_close_marks.csv] "
               "[--margin-config=<path>] [--trades-out=<path>] [--fills-out=<path>] "
               "[--end-of-test=leave_open|force_liquidate]\n"
               "  Every Order is gated by the hard PortfolioRiskManager built from "
               "risk_config.csv (Phase 21) -- NOT the pass-through risk manager "
               "quant_backtest_targets_csv uses for historical research.\n";
        return 2;
    }
    try {
        const std::string bars_path = argv[1];
        const std::string contracts_path = argv[2];
        const std::string targets_path = argv[3];
        const std::string risk_config_path = argv[4];
        const std::string policy_str = (argc >= 6) ? argv[5] : "no_decision";
        const double commission_usd = (argc >= 7) ? std::stod(argv[6]) : 2.0;
        const double slippage_ticks = (argc >= 8) ? std::stod(argv[7]) : 0.0;
        const double spread_ticks   = (argc >= 9) ? std::stod(argv[8]) : 0.0;
        const std::string vday_path = (argc >= 10) ? argv[9] : "";
        const std::string roll_marks_path = (argc >= 11) ? argv[10] : "";
        if (!roll_marks_path.empty() && vday_path.empty()) {
            throw std::runtime_error(
                "roll_close_marks.csv requires validation_days.csv to also be given "
                "(positional CLI arg 9) so the argument order is unambiguous");
        }

        quant::EndOfTestPolicy eot;
        if (end_of_test_str == "leave_open") {
            eot = quant::EndOfTestPolicy::LeaveOpen;
        } else if (end_of_test_str == "force_liquidate") {
            eot = quant::EndOfTestPolicy::ForceLiquidateFinalClose;
        } else {
            throw std::runtime_error("--end-of-test= must be 'leave_open' or 'force_liquidate', "
                                     "got '" + end_of_test_str + "'");
        }

        const auto bars = read_bars(bars_path);
        const auto registry = quant::load_contract_registry(contracts_path);

        std::string fingerprint;
        auto rows = quant::parse_targets_csv(targets_path, fingerprint);

        quant::PaperTradingRunConfig cfg;
        cfg.schedule_policy = policy_str;
        cfg.commission_per_contract_usd = commission_usd;
        cfg.slippage_ticks = slippage_ticks;
        cfg.spread_ticks = spread_ticks;
        cfg.end_of_test = eot;
        if (!vday_path.empty()) cfg.validation_day_boundaries_ns = read_validation_days(vday_path);
        if (!roll_marks_path.empty()) cfg.roll_close_marks = read_roll_close_marks(roll_marks_path);
        cfg.risk_config = read_risk_config(risk_config_path);

        std::optional<quant::MarginModel> margin;
        const quant::MarginModel* margin_ptr = nullptr;
        if (!margin_config_path.empty()) {
            margin = quant::load_margin_model(margin_config_path);
            margin_ptr = &margin.value();
        }

        const auto out = quant::run_paper_trading_backtest(bars, registry, std::move(rows),
                                                            fingerprint, cfg, margin_ptr);
        const auto& r = out.result;

        std::cout << "{\"strategy_fingerprint\":\"" << out.strategy_fingerprint << "\""
                  << ",\"schedule_policy\":\"" << out.schedule_policy << "\""
                  << ",\"end_of_test\":\"" << end_of_test_str << "\""
                  << ",\"target_rows\":" << out.target_rows
                  << ",\"target_rows_applied\":" << out.target_rows_applied
                  << ",\"trades\":" << r.closed_trades
                  << ",\"gross_pnl_usd\":" << r.gross_realized_pnl_usd
                  << ",\"costs_usd\":" << r.costs_usd
                  << ",\"net_pnl_usd\":" << r.net_realized_pnl_usd
                  << ",\"max_drawdown_usd\":" << r.max_drawdown_usd
                  << ",\"unrealized_pnl_usd\":" << r.unrealized_pnl_usd_at_end
                  << ",\"events\":" << r.events_processed
                  << ",\"signals\":" << r.signals_generated
                  << ",\"orders\":" << r.orders_generated
                  << ",\"fills\":" << r.fills_generated
                  << ",\"rolls\":" << r.rolls
                  << ",\"rolls_priced_contemporaneous\":" << r.rolls_priced_contemporaneous
                  << ",\"rolls_priced_auxiliary_marks\":" << r.rolls_priced_auxiliary_marks
                  << ",\"rolls_priced_stale\":" << r.rolls_priced_stale
                  << ",\"rolls_deferred\":" << r.rolls_deferred
                  << ",\"non_fills\":" << r.non_fills
                  << ",\"eot_liquidations\":" << r.eot_liquidations.size()
                  << ",\"eot_positions_left_open_stale\":" << r.eot_positions_left_open_stale
                  << ",\"slippage_ticks\":" << out.slippage_ticks
                  << ",\"commission_per_contract_usd\":" << out.commission_per_contract_usd
                  << ",\"unique_contracts\":" << r.contracts_traded.size()
                  << ",\"open_positions\":" << r.final_positions.size()
                  << ",\"starting_capital_usd\":" << r.portfolio_at_end.starting_capital_usd
                  << ",\"cash_usd\":" << r.portfolio_at_end.cash_usd
                  << ",\"equity_usd\":" << r.portfolio_at_end.equity_usd
                  << ",\"gross_exposure_usd\":" << r.portfolio_at_end.gross_exposure_usd
                  << ",\"net_exposure_usd\":" << r.portfolio_at_end.net_exposure_usd
                  << ",\"gross_leverage\":" << r.portfolio_at_end.gross_leverage
                  << ",\"peak_equity_usd\":" << r.portfolio_at_end.peak_equity_usd
                  << ",\"portfolio_drawdown_usd\":" << r.portfolio_at_end.drawdown_usd
                  << ",\"portfolio_drawdown_pct\":" << r.portfolio_at_end.drawdown_pct
                  << ",\"day_start_equity_usd\":" << r.portfolio_at_end.day_start_equity_usd
                  << ",\"day_realized_pnl_usd\":" << r.portfolio_at_end.day_realized_pnl_usd
                  << ",\"margin_complete\":" << (r.portfolio_at_end.margin_complete ? "true" : "false")
                  << ",\"has_stale_mark\":" << (r.portfolio_at_end.has_stale_mark ? "true" : "false")
                  << ",\"risk_rejects\":" << r.risk_rejects
                  << ",\"risk_resizes\":" << r.risk_resizes
                  << ",\"bars\":" << r.bars_processed
                  << ",\"contracts_resolved\":" << out.contracts_resolved
                  << ",\"net_equity_usd_at_end\":" << r.net_equity_usd_at_end;

        std::cout << ",\"risk_decisions\":[";
        for (std::size_t i = 0; i < r.risk_decisions.size(); ++i) {
            const auto& d = r.risk_decisions[i];
            if (i != 0) std::cout << ',';
            const char* verdict = d.verdict == quant::RiskVerdict::Approve ? "approve"
                                 : d.verdict == quant::RiskVerdict::Resize ? "resize"
                                                                            : "reject";
            std::cout << "{\"ts_decision_ns\":" << d.ts_decision_ns
                      << ",\"verdict\":\"" << verdict << "\""
                      << ",\"approved_quantity\":" << d.approved_quantity
                      << ",\"reason_code\":\"" << d.reason_code << "\"}";
        }
        std::cout << "]";

        // ADDITIVE (Phase 21.1): the authoritative C++ position snapshot --
        // BacktestResult::portfolio_at_end.positions (Phase 08 PositionExposure,
        // already computed by the PortfolioAccountant), serialized verbatim so
        // Python never reconstructs a position from fills. One entry per held
        // (units != 0) instrument, ordered by instrument_id (portfolio.hpp).
        auto b = [](bool v) { return v ? "true" : "false"; };
        std::cout << ",\"positions\":[";
        for (std::size_t i = 0; i < r.portfolio_at_end.positions.size(); ++i) {
            const auto& p = r.portfolio_at_end.positions[i];
            if (i != 0) std::cout << ',';
            std::cout << "{\"instrument_id\":" << p.instrument_id
                      << ",\"raw_symbol\":\"" << p.raw_symbol << "\""
                      << ",\"root_symbol\":\"" << p.root_symbol << "\""
                      << ",\"units\":" << p.units
                      << ",\"avg_entry_price\":" << p.avg_entry_price
                      << ",\"multiplier\":" << p.multiplier
                      << ",\"mark_price\":" << p.mark_price
                      << ",\"mark_ts_ns\":" << p.mark_ts_ns
                      << ",\"mark_age_ns\":" << p.mark_age_ns
                      << ",\"mark_present\":" << b(p.mark_present)
                      << ",\"mark_is_stale\":" << b(p.mark_is_stale)
                      << ",\"valuation_is_estimated\":" << b(p.valuation_is_estimated)
                      << ",\"gross_notional_usd\":" << p.gross_notional_usd
                      << ",\"signed_notional_usd\":" << p.signed_notional_usd
                      << ",\"unrealized_pnl_usd\":" << p.unrealized_pnl_usd
                      << ",\"initial_margin_usd\":" << p.initial_margin_usd
                      << ",\"maintenance_margin_usd\":" << p.maintenance_margin_usd
                      << ",\"margin_known\":" << b(p.margin_known) << "}";
        }
        std::cout << "]";

        auto g = [](double v) {
            char buf[40];
            std::snprintf(buf, sizeof buf, "%.12g", v);
            return std::string(buf);
        };
        std::cout << ",\"daily_equity_basis\":\"" << out.daily_equity_basis << "\"";
        std::cout << ",\"daily_equity\":[";
        for (std::size_t i = 0; i < r.daily_equity.size(); ++i) {
            const auto& p = r.daily_equity[i];
            if (i != 0) std::cout << ',';
            std::cout << '[' << p.session_day_index << ',' << p.ts_ns << ','
                      << g(p.equity_usd) << ',' << g(p.net_realized_pnl_usd) << ','
                      << g(p.unrealized_pnl_usd) << ',' << g(p.costs_usd) << ','
                      << p.fills_cumulative << ',' << p.bars_cumulative << ']';
        }
        std::cout << "]}\n";

        if (!trades_out_path.empty()) {
            std::ofstream tout(trades_out_path);
            if (!tout) throw std::runtime_error("cannot open trades-out csv '" + trades_out_path + "'");
            quant::write_closed_trades_csv(tout, r.trades);
            if (!tout) throw std::runtime_error("failed writing trades-out csv '" + trades_out_path + "'");
        }
        if (!fills_out_path.empty()) {
            std::ofstream fout(fills_out_path);
            if (!fout) throw std::runtime_error("cannot open fills-out csv '" + fills_out_path + "'");
            quant::write_fills_csv(fout, r.fills);
            if (!fout) throw std::runtime_error("failed writing fills-out csv '" + fills_out_path + "'");
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
