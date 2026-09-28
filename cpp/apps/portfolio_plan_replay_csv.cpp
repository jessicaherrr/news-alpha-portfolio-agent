// News Alpha Phase G CLI: replay a PortfolioPlan's execution targets through
// the deterministic BacktestEngine, gated by the hard PortfolioRiskManager.
//
// Usage:
//   quant_portfolio_plan_replay_csv <bars.csv> <contracts.csv> <plan_targets.csv> \
//       <risk_config.csv> [commission_per_contract_usd] [slippage_ticks] [spread_ticks] \
//       [validation_days.csv] [roll_close_marks.csv] [splits.csv] \
//       [--end-of-test=leave_open|force_liquidate] [--commission-schedule=<csv>]
//
// `--commission-schedule` (Phase H acceptance patch): header exactly
// `root_symbol,commission_per_unit_usd` -- USD per contract for a futures root,
// per share for an ETF root (ExecutionConfig::commission_per_contract_usd_by_root).
// It must name EVERY root the bars trade, or the run is refused: a mixed
// contract/share book is never charged a scalar meant for the other unit.
// Omitted => the scalar commission for every root, byte-identical to before.
//
// ADDITIVE (News Alpha Phase H). The three optional files are the frozen
// Phase 13.1 / 13.5C / Phase 6 boundary files, read with the reference CLI's
// semantics (quant_core/detail/replay_csv.hpp); an omitted or empty-string
// argument means "not supplied". `--end-of-test` (anywhere in argv) defaults to
// leave_open -- the Phase G behaviour; a validation replay passes
// force_liquidate, exactly like the reference research path, so the final
// daily-equity point carries complete realized PnL. The JSON line gains the
// reference CLI's PnL / roll / cost fields and its `daily_equity` trace; every
// Phase G field keeps its name and meaning.
//
// bars.csv / contracts.csv: the frozen Phase 02.5 boundary formats.
// plan_targets.csv: header exactly
//   ts_event_ns,root_symbol,target_units,portfolio_plan_fingerprint
// -- target-position INTENT per ROOT at its decision bar, one plan per file
// (a single `portplan1:` fingerprint). It is a portfolio plan, not a DSL
// strategy, so it is deliberately NOT a Phase 11 targets.csv (whose frozen
// parser requires a `stratdsl1:` fingerprint and is not touched here).
// risk_config.csv: the Phase 21 16-column RiskConfig row
// (alpha_agent.paper.risk_policy.PaperRiskPolicy.csv_rows), built by
// alpha_agent.portfolio.handoff from the SAME constraints the plan was built
// under -- the engine's hard gate re-checks the plan independently.
//
// Wiring: quant::run_paper_trading_backtest (paper_trading_run.hpp) with
// EndOfTestPolicy::LeaveOpen -- the same engine, resolver, risk gate and Fill
// path as Phase 21 paper trading. This file adds NO execution, accounting,
// fill, roll or risk semantics; it only parses its own intent file and prints
// the engine's own result (every number %.17g). Official positions and
// exposure are the C++ PortfolioAccountant's `portfolio_at_end`.

#include "quant_core/contract_io.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/detail/csv.hpp"
#include "quant_core/detail/replay_csv.hpp"
#include "quant_core/paper_trading_run.hpp"
#include "quant_core/risk_config.hpp"
#include "quant_core/scheduled_target_strategy.hpp"
#include "quant_core/types.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using quant::detail::split_csv_line;

std::vector<quant::ScheduledTarget> read_plan_targets(const std::string& path, std::string& fingerprint) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open plan targets csv '" + path + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty plan targets csv");
    const std::vector<std::string> header{"ts_event_ns", "root_symbol", "target_units", "portfolio_plan_fingerprint"};
    if (split_csv_line(line) != header) {
        throw std::runtime_error("plan targets csv header must be exactly "
                                 "'ts_event_ns,root_symbol,target_units,portfolio_plan_fingerprint'");
    }
    fingerprint.clear();
    std::vector<quant::ScheduledTarget> rows;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = split_csv_line(line);
        if (f.size() != header.size()) throw std::runtime_error("plan targets csv: malformed row");
        if (f[3].rfind("portplan1:", 0) != 0 && f[3].rfind("portstrat1:", 0) != 0) {
            throw std::runtime_error("plan targets csv: fingerprint '" + f[3] +
                                     "' is not a portplan1 / portstrat1 fingerprint");
        }
        if (fingerprint.empty()) fingerprint = f[3];
        if (f[3] != fingerprint) throw std::runtime_error("plan targets csv: mixed plan fingerprints");
        if (f[1].empty() || f[1].find('.') != std::string::npos) {
            throw std::runtime_error("plan targets csv: root_symbol '" + f[1] + "' is not a bare root");
        }
        std::size_t used = 0;
        const long long units = std::stoll(f[2], &used);
        if (used != f[2].size()) throw std::runtime_error("plan targets csv: non-integer target_units '" + f[2] + "'");
        rows.push_back(quant::ScheduledTarget{f[1], static_cast<std::int64_t>(std::stoll(f[0])),
                                              static_cast<int>(units)});
    }
    if (rows.empty()) throw std::runtime_error("plan targets csv: no rows");
    return rows;
}

// Phase H acceptance patch: `root_symbol,commission_per_unit_usd`; a duplicate
// root or a negative / non-finite rate is a loud error.
std::map<std::string, double> read_commission_schedule(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open commission schedule csv '" + path + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty commission schedule csv");
    const std::vector<std::string> header{"root_symbol", "commission_per_unit_usd"};
    if (split_csv_line(line) != header) {
        throw std::runtime_error("commission schedule csv header must be exactly 'root_symbol,commission_per_unit_usd'");
    }
    std::map<std::string, double> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = split_csv_line(line);
        if (f.size() != 2 || f[0].empty()) throw std::runtime_error("commission schedule csv: malformed row");
        const double rate = std::stod(f[1]);
        if (!(rate >= 0.0) || rate > 1e9) throw std::runtime_error("commission schedule csv: bad rate for " + f[0]);
        if (!out.emplace(f[0], rate).second) throw std::runtime_error("commission schedule csv: duplicate root " + f[0]);
    }
    return out;
}

std::string g(double v) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    constexpr const char* kEotFlag = "--end-of-test=";
    constexpr const char* kScheduleFlag = "--commission-schedule=";
    std::string end_of_test = "leave_open";
    std::string schedule_path;
    std::vector<char*> pos;
    for (int i = 0; i < argc; ++i) {
        if (i > 0 && std::strncmp(argv[i], kEotFlag, std::strlen(kEotFlag)) == 0) {
            end_of_test = argv[i] + std::strlen(kEotFlag);
            continue;
        }
        if (i > 0 && std::strncmp(argv[i], kScheduleFlag, std::strlen(kScheduleFlag)) == 0) {
            schedule_path = argv[i] + std::strlen(kScheduleFlag);
            if (schedule_path.empty()) {
                std::cerr << "error: --commission-schedule= requires a path\n";
                return 2;
            }
            continue;
        }
        pos.push_back(argv[i]);
    }
    argc = static_cast<int>(pos.size());
    argv = pos.data();
    if (argc < 5 || argc > 11) {
        std::cerr << "usage: quant_portfolio_plan_replay_csv <bars.csv> <contracts.csv> <plan_targets.csv> "
                     "<risk_config.csv> [commission_per_contract_usd] [slippage_ticks] [spread_ticks] "
                     "[validation_days.csv] [roll_close_marks.csv] [splits.csv] "
                     "[--end-of-test=leave_open|force_liquidate]\n";
        return 2;
    }
    try {
        const auto bars = quant::detail::read_bars_csv(argv[1]);
        const auto registry = quant::load_contract_registry(argv[2]);
        std::string fingerprint;
        auto rows = read_plan_targets(argv[3], fingerprint);
        const auto optional_path = [&](int i) { return argc > i ? std::string(argv[i]) : std::string(); };

        quant::PaperTradingRunConfig cfg;
        cfg.schedule_policy = "no_decision";
        cfg.commission_per_contract_usd = argc >= 6 ? std::stod(argv[5]) : 0.0;
        cfg.slippage_ticks = argc >= 7 ? std::stod(argv[6]) : 0.0;
        cfg.spread_ticks = argc >= 8 ? std::stod(argv[7]) : 0.0;
        cfg.risk_config = quant::detail::read_risk_config_csv(argv[4]);
        if (end_of_test == "leave_open") {
            cfg.end_of_test = quant::EndOfTestPolicy::LeaveOpen;
        } else if (end_of_test == "force_liquidate") {
            cfg.end_of_test = quant::EndOfTestPolicy::ForceLiquidateFinalClose;
        } else {
            throw std::runtime_error("--end-of-test must be leave_open or force_liquidate, got '" + end_of_test + "'");
        }
        if (const auto p = optional_path(8); !p.empty()) {
            cfg.validation_day_boundaries_ns = quant::detail::read_validation_days_csv(p);
        }
        if (const auto p = optional_path(9); !p.empty()) cfg.roll_close_marks = quant::detail::read_roll_close_marks_csv(p);
        if (const auto p = optional_path(10); !p.empty()) cfg.corporate_actions.splits = quant::detail::read_splits_csv(p);
        if (!schedule_path.empty()) {
            cfg.commission_per_contract_usd_by_root = read_commission_schedule(schedule_path);
            for (const auto& bar : bars) {  // every traded root must be priced explicitly
                const auto* spec = registry.find_by_instrument_id(bar.instrument_id);
                if (spec != nullptr && !cfg.commission_per_contract_usd_by_root.count(spec->root_symbol)) {
                    throw std::runtime_error("commission schedule incomplete: root '" + spec->root_symbol +
                                             "' has no commission_per_unit_usd entry");
                }
            }
        }

        const auto out = quant::run_paper_trading_backtest(bars, registry, std::move(rows), fingerprint, cfg);
        const auto& r = out.result;
        const auto& p = r.portfolio_at_end;

        std::cout << "{\"portfolio_plan_fingerprint\":\"" << out.strategy_fingerprint << "\""
                  << ",\"target_rows\":" << out.target_rows << ",\"target_rows_applied\":" << out.target_rows_applied
                  << ",\"bars\":" << r.bars_processed << ",\"signals\":" << r.signals_generated
                  << ",\"orders\":" << r.orders_generated << ",\"fills\":" << r.fills_generated
                  << ",\"non_fills\":" << r.non_fills << ",\"risk_rejects\":" << r.risk_rejects
                  << ",\"risk_resizes\":" << r.risk_resizes << ",\"costs_usd\":" << g(r.costs_usd)
                  << ",\"starting_capital_usd\":" << g(p.starting_capital_usd) << ",\"equity_usd\":" << g(p.equity_usd)
                  << ",\"gross_exposure_usd\":" << g(p.gross_exposure_usd)
                  << ",\"net_exposure_usd\":" << g(p.net_exposure_usd) << ",\"gross_leverage\":" << g(p.gross_leverage)
                  << ",\"net_leverage\":" << g(p.net_leverage)
                  << ",\"margin_complete\":" << (p.margin_complete ? "true" : "false")
                  << ",\"valuation_complete\":" << (p.valuation_complete ? "true" : "false");
        std::cout << ",\"risk_decisions\":[";
        for (std::size_t i = 0; i < r.risk_decisions.size(); ++i) {
            const auto& d = r.risk_decisions[i];
            const char* verdict = d.verdict == quant::RiskVerdict::Approve  ? "approve"
                                  : d.verdict == quant::RiskVerdict::Resize ? "resize"
                                                                            : "reject";
            std::cout << (i ? "," : "") << "{\"ts_decision_ns\":" << d.ts_decision_ns << ",\"verdict\":\""
                      << verdict << "\",\"approved_quantity\":" << d.approved_quantity << ",\"reason_code\":\""
                      << d.reason_code << "\"}";
        }
        std::cout << "],\"positions\":[";
        for (std::size_t i = 0; i < p.positions.size(); ++i) {
            const auto& x = p.positions[i];
            std::cout << (i ? "," : "") << "{\"instrument_id\":" << x.instrument_id << ",\"raw_symbol\":\""
                      << x.raw_symbol << "\",\"root_symbol\":\"" << x.root_symbol << "\",\"units\":" << x.units
                      << ",\"avg_entry_price\":" << g(x.avg_entry_price) << ",\"multiplier\":" << g(x.multiplier)
                      << ",\"mark_price\":" << g(x.mark_price) << ",\"gross_notional_usd\":" << g(x.gross_notional_usd)
                      << ",\"signed_notional_usd\":" << g(x.signed_notional_usd) << "}";
        }
        // ADDITIVE (Phase H): the reference CLI's PnL / roll / cost fields and
        // its daily_equity trace, verbatim from the same BacktestResult.
        std::cout << "],\"commission_schedule\":[";
        std::size_t k = 0;
        for (const auto& [root, rate] : cfg.commission_per_contract_usd_by_root) {
            std::cout << (k++ ? "," : "") << "{\"root_symbol\":\"" << root << "\",\"commission_per_unit_usd\":"
                      << g(rate) << "}";
        }
        std::cout << "],\"end_of_test\":\"" << end_of_test << "\""
                  << ",\"commission_per_contract_usd\":" << g(out.commission_per_contract_usd)
                  << ",\"slippage_ticks\":" << g(out.slippage_ticks) << ",\"spread_ticks\":" << g(out.spread_ticks)
                  << ",\"trades\":" << r.closed_trades << ",\"gross_pnl_usd\":" << g(r.gross_realized_pnl_usd)
                  << ",\"net_pnl_usd\":" << g(r.net_realized_pnl_usd)
                  << ",\"unrealized_pnl_usd\":" << g(r.unrealized_pnl_usd_at_end)
                  << ",\"net_equity_usd_at_end\":" << g(r.net_equity_usd_at_end)
                  << ",\"max_drawdown_usd\":" << g(r.max_drawdown_usd) << ",\"rolls\":" << r.rolls
                  << ",\"rolls_priced_contemporaneous\":" << r.rolls_priced_contemporaneous
                  << ",\"rolls_priced_auxiliary_marks\":" << r.rolls_priced_auxiliary_marks
                  << ",\"rolls_priced_stale\":" << r.rolls_priced_stale << ",\"rolls_deferred\":" << r.rolls_deferred
                  << ",\"eot_liquidations\":" << r.eot_liquidations.size()
                  << ",\"daily_equity_basis\":\"" << out.daily_equity_basis << "\",\"daily_equity\":[";
        for (std::size_t i = 0; i < r.daily_equity.size(); ++i) {
            const auto& d = r.daily_equity[i];
            std::cout << (i ? "," : "") << '[' << d.session_day_index << ',' << d.ts_ns << ',' << g(d.equity_usd) << ','
                      << g(d.net_realized_pnl_usd) << ',' << g(d.unrealized_pnl_usd) << ',' << g(d.costs_usd) << ','
                      << d.fills_cumulative << ',' << d.bars_cumulative << ']';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
