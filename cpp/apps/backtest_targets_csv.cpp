// Phase 11 reference CLI: replay a precomputed Python target schedule through
// the deterministic C++ BacktestEngine.
//
// Usage:
//   quant_backtest_targets_csv <bars.csv> <contracts.csv> <targets.csv> [policy]
//
// bars.csv / contracts.csv: the FROZEN Phase 02.5 boundary (unchanged).
// targets.csv: the SEPARATE Phase 11 research boundary -- header exactly
//   ts_event_ns,root_symbol,target_units,strategy_fingerprint,matched_rule_id
// carrying TARGET INTENT only (no fill/execution price, raw_symbol, instrument_id,
// slippage, or risk decision). `policy` (default no_decision) is the absent-row
// schedule policy: no_decision | flat | require_row | reemit_previous. Absence of
// a row means NO NEW STRATEGY DECISION -- never an implicit flat or retry.
//
// ADDITIVE (Phase 15A), READ-ONLY: `--trades-out=<path>` and `--fills-out=<path>`
// may appear anywhere in argv. They write the engine's already-recorded
// Fill-derived closed-trade and fill audit trails to CSV and change no
// execution, accounting, fill, roll, risk or portfolio semantics and no field of
// the JSON line. Both are stripped from argv before positional parsing, so every
// existing positional invocation is byte-identical with or without this build.
//
// The engine loads the registry, hard-verifies every bar resolves to a real
// contract live at its timestamp, then replays the bars. Official fills / PnL /
// costs / drawdown / portfolio come only from the C++ BacktestResult. Output is
// one JSON line (a superset of quant_backtest_csv's keys; additive).

#include "quant_core/contract_io.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/corporate_actions.hpp"
#include "quant_core/detail/csv.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/risk_manager.hpp"
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
#include <set>
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

// Phase 13.1: read a canonical validation-day boundary file. Header exactly
// `boundary_ts_ns`; one strictly-ascending int64 UTC-ns value per canonical
// futures trading_day (the ts of that day's last eligible event). Python derives
// it from the SessionCalendar -- the C++ core never carries a tz database.
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

// Phase 13.5C: read an auxiliary roll-close-marks file. Header exactly
//   instrument_id,ts_event_ns,close
// One same-timestamp OUTGOING-contract close per (instrument_id, ts_event_ns),
// derived deterministically from the already-acquired immutable roll-overlap raw
// bars. Consulted ONLY by the engine's roll close-leg (never a MarketEvent). A
// duplicate (instrument_id, ts_event_ns) is a loud error -- rows are otherwise
// order-independent (a keyed map).
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

// this phase: read an optional splits CSV. Header exactly
//   instrument_id,effective_ts_ns,ratio
// One row per SplitAction, keyed by (instrument_id, effective_ts_ns) exactly
// as EngineConfig::CorporateActionsConfig::splits needs. A duplicate key is a
// loud error; rows are otherwise order-independent (a keyed map, mirroring
// read_roll_close_marks).
std::map<std::pair<std::uint32_t, std::int64_t>, quant::SplitAction> read_splits_csv(
    const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open splits csv '" + path + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty splits csv");
    const auto header = quant::detail::split_csv_line(line);
    if (header.size() != 3 || header[0] != "instrument_id" || header[1] != "effective_ts_ns" ||
        header[2] != "ratio") {
        throw std::runtime_error(
            "splits csv header must be exactly 'instrument_id,effective_ts_ns,ratio'");
    }
    std::map<std::pair<std::uint32_t, std::int64_t>, quant::SplitAction> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = quant::detail::split_csv_line(line);
        if (f.size() != 3) continue;
        quant::SplitAction action;
        action.instrument_id   = static_cast<std::uint32_t>(std::stoul(f[0]));
        action.effective_ts_ns = static_cast<std::int64_t>(std::stoll(f[1]));
        action.ratio            = std::stod(f[2]);
        const auto key = std::pair<std::uint32_t, std::int64_t>{action.instrument_id,
                                                                 action.effective_ts_ns};
        if (!out.emplace(key, action).second) {
            throw std::runtime_error(
                "splits csv: duplicate (instrument_id, effective_ts_ns) for instrument_id " +
                f[0] + " at ts " + f[1]);
        }
    }
    return out;
}

// this phase: read an optional cash-distributions CSV. Header exactly
//   instrument_id,ex_date_ts_ns,pay_date_ts_ns,amount_per_share_usd
// One row per CashDistribution, split into its ex-date and pay-date legs --
// EngineConfig::CorporateActionsConfig keys them separately (ex_date_ts_ns /
// pay_date_ts_ns) because the engine consults each leg at its own bar. A
// duplicate (instrument_id, ex_date_ts_ns) or (instrument_id, pay_date_ts_ns)
// is a loud error; rows are otherwise order-independent.
struct DistributionLegs {
    std::map<std::pair<std::uint32_t, std::int64_t>, quant::CashDistribution> ex_date;
    std::map<std::pair<std::uint32_t, std::int64_t>, quant::CashDistribution> pay_date;
};

DistributionLegs read_distributions_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open distributions csv '" + path + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty distributions csv");
    const auto header = quant::detail::split_csv_line(line);
    if (header.size() != 4 || header[0] != "instrument_id" || header[1] != "ex_date_ts_ns" ||
        header[2] != "pay_date_ts_ns" || header[3] != "amount_per_share_usd") {
        throw std::runtime_error(
            "distributions csv header must be exactly "
            "'instrument_id,ex_date_ts_ns,pay_date_ts_ns,amount_per_share_usd'");
    }
    DistributionLegs out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = quant::detail::split_csv_line(line);
        if (f.size() != 4) continue;
        quant::CashDistribution action;
        action.instrument_id       = static_cast<std::uint32_t>(std::stoul(f[0]));
        action.ex_date_ts_ns        = static_cast<std::int64_t>(std::stoll(f[1]));
        action.pay_date_ts_ns       = static_cast<std::int64_t>(std::stoll(f[2]));
        action.amount_per_share_usd = std::stod(f[3]);
        const auto ex_key = std::pair<std::uint32_t, std::int64_t>{action.instrument_id,
                                                                    action.ex_date_ts_ns};
        if (!out.ex_date.emplace(ex_key, action).second) {
            throw std::runtime_error(
                "distributions csv: duplicate (instrument_id, ex_date_ts_ns) for instrument_id " +
                f[0] + " at ts " + f[1]);
        }
        const auto pay_key = std::pair<std::uint32_t, std::int64_t>{action.instrument_id,
                                                                     action.pay_date_ts_ns};
        if (!out.pay_date.emplace(pay_key, action).second) {
            throw std::runtime_error(
                "distributions csv: duplicate (instrument_id, pay_date_ts_ns) for instrument_id " +
                f[0] + " at ts " + f[2]);
        }
    }
    return out;
}

std::size_t verify_contract_resolution(const std::vector<quant::MarketBar>& bars,
                                       const quant::ContractRegistry& registry) {
    std::set<std::uint32_t> seen;
    for (const auto& bar : bars) {
        const quant::ContractSpec* spec = registry.find_by_instrument_id(bar.instrument_id);
        if (spec == nullptr) {
            throw std::runtime_error("bar at ts " + std::to_string(bar.ts_event_ns) +
                                     " references unknown instrument_id " +
                                     std::to_string(bar.instrument_id));
        }
        if (!spec->is_live_at(bar.ts_event_ns)) {
            throw std::runtime_error("bar at ts " + std::to_string(bar.ts_event_ns) +
                                     " uses contract '" + spec->raw_symbol +
                                     "' outside its tradable window");
        }
        seen.insert(bar.instrument_id);
    }
    return seen.size();
}

}  // namespace

int main(int argc, char** argv) {
    // ADDITIVE (Phase 15A): pull `--trades-out=<path>` out of argv first so the
    // frozen positional contract below sees exactly the argument vector it saw
    // before this flag existed.
    std::string trades_out_path;
    std::string fills_out_path;
    std::vector<char*> pos;
    pos.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        static constexpr const char* kTradesFlag = "--trades-out=";
        static constexpr const char* kFillsFlag = "--fills-out=";
        if (i > 0 && std::strncmp(argv[i], kTradesFlag, std::strlen(kTradesFlag)) == 0) {
            trades_out_path = argv[i] + std::strlen(kTradesFlag);
            if (trades_out_path.empty()) {
                std::cerr << "error: --trades-out= requires a path\n";
                return 2;
            }
            continue;
        }
        if (i > 0 && std::strncmp(argv[i], kFillsFlag, std::strlen(kFillsFlag)) == 0) {
            fills_out_path = argv[i] + std::strlen(kFillsFlag);
            if (fills_out_path.empty()) {
                std::cerr << "error: --fills-out= requires a path\n";
                return 2;
            }
            continue;
        }
        pos.push_back(argv[i]);
    }
    argc = static_cast<int>(pos.size());
    argv = pos.data();

    if (argc < 4) {
        std::cerr << "usage: quant_backtest_targets_csv <bars.csv> <contracts.csv> "
                     "<targets.csv> [no_decision|flat|require_row|reemit_previous] "
                     "[commission_per_contract_usd] [slippage_ticks] [spread_ticks] "
                     "[validation_days.csv] [roll_close_marks.csv] [splits.csv] "
                     "[distributions.csv]\n"
                     "  The three optional cost values default to the reference "
                     "assumptions (2.0, 0.0, 0.0). They set ExecutionConfig only -- "
                     "no strategy / accounting semantics change. Phase 13 cost stress "
                     "reruns this exact engine path with scaled cost values.\n"
                     "  validation_days.csv (header 'boundary_ts_ns', strictly "
                     "ascending) makes daily_equity a canonical futures trading_day "
                     "series; omitted => legacy UTC-day buckets (diagnostics only).\n"
                     "  roll_close_marks.csv (header "
                     "'instrument_id,ts_event_ns,close') supplies same-timestamp "
                     "OUTGOING-contract closes for the roll close-leg only "
                     "(Phase 13.5C); omitted => the frozen RejectDefer path is "
                     "byte-identical. Requires validation_days.csv to also be "
                     "given so the positional CLI order stays unambiguous.\n"
                     "  splits.csv (header 'instrument_id,effective_ts_ns,ratio') "
                     "supplies ETF corporate-action splits consulted ONLY at the "
                     "exact (instrument_id, ts_event_ns) of a bar, applied before "
                     "that bar's own open mark / fills; omitted => the frozen path "
                     "is byte-identical. Requires roll_close_marks.csv to also be "
                     "given.\n"
                     "  distributions.csv (header "
                     "'instrument_id,ex_date_ts_ns,pay_date_ts_ns,"
                     "amount_per_share_usd') supplies ETF cash distributions, each "
                     "row's ex-date and pay-date legs consulted ONLY at their own "
                     "exact (instrument_id, ts_event_ns); omitted => the frozen "
                     "path is byte-identical. Requires splits.csv to also be "
                     "given.\n"
                     "  --trades-out=<path> / --fills-out=<path> (anywhere in argv, "
                     "Phase 15A) write the Fill-derived closed-trade and fill audit "
                     "trails as CSV. Read-only exports: no execution/accounting "
                     "semantics and no JSON field changes. The fills export carries "
                     "every commission, which ClosedTrade::costs_usd (the CLOSING "
                     "fill's commission only) cannot express.\n";
        return 2;
    }
    try {
        const std::string bars_path = argv[1];
        const std::string contracts_path = argv[2];
        const std::string targets_path = argv[3];
        const std::string policy_str = (argc >= 5) ? argv[4] : "no_decision";
        // Optional Phase 13 cost-stress overrides. Defaults reproduce the frozen
        // reference assumptions exactly.
        const double commission_usd = (argc >= 6) ? std::stod(argv[5]) : 2.0;
        const double slippage_ticks = (argc >= 7) ? std::stod(argv[6]) : 0.0;
        const double spread_ticks   = (argc >= 8) ? std::stod(argv[7]) : 0.0;
        const std::string vday_path = (argc >= 9) ? argv[8] : "";
        const std::string roll_marks_path = (argc >= 10) ? argv[9] : "";
        const std::string splits_path = (argc >= 11) ? argv[10] : "";
        const std::string distributions_path = (argc >= 12) ? argv[11] : "";
        if (commission_usd < 0.0 || slippage_ticks < 0.0 || spread_ticks < 0.0) {
            throw std::runtime_error("cost overrides must be non-negative");
        }
        if (!roll_marks_path.empty() && vday_path.empty()) {
            throw std::runtime_error(
                "roll_close_marks.csv requires validation_days.csv to also be given "
                "(positional CLI arg 8) so the argument order is unambiguous");
        }
        if (!splits_path.empty() && roll_marks_path.empty()) {
            throw std::runtime_error(
                "splits.csv requires roll_close_marks.csv to also be given "
                "(positional CLI arg 9) so the argument order is unambiguous");
        }
        if (!distributions_path.empty() && splits_path.empty()) {
            throw std::runtime_error(
                "distributions.csv requires splits.csv to also be given "
                "(positional CLI arg 10) so the argument order is unambiguous");
        }

        const auto bars = read_bars(bars_path);
        const auto registry = quant::load_contract_registry(contracts_path);
        const std::size_t n_contracts = verify_contract_resolution(bars, registry);

        std::string fingerprint;
        auto rows = quant::parse_targets_csv(targets_path, fingerprint);

        // A target root must not collide with a real raw contract symbol.
        for (const auto& row : rows) {
            if (registry.find_by_raw_symbol(row.root_symbol) != nullptr) {
                throw std::runtime_error("targets csv: root_symbol '" + row.root_symbol +
                                         "' is a real raw contract in the registry");
            }
        }

        const auto policy = quant::parse_absent_policy(policy_str);
        const quant::ScheduledTargetStrategy strategy(std::move(rows), fingerprint, policy);

        const quant::RegistryActiveContractResolver resolver(registry);
        const quant::PassThroughRiskManager risk;  // frozen reference path (Phase 19 wires RiskConfig)
        quant::EngineConfig config;
        config.execution.commission_per_contract_usd = commission_usd;
        config.execution.slippage_ticks = slippage_ticks;
        config.execution.spread_ticks = spread_ticks;
        config.end_of_test = quant::EndOfTestPolicy::ForceLiquidateFinalClose;
        if (!vday_path.empty()) {
            config.validation_day_boundaries_ns = read_validation_days(vday_path);
        }
        if (!roll_marks_path.empty()) {
            config.roll.close_marks = read_roll_close_marks(roll_marks_path);
        }
        if (!splits_path.empty()) {
            config.corporate_actions.splits = read_splits_csv(splits_path);
        }
        if (!distributions_path.empty()) {
            const DistributionLegs legs = read_distributions_csv(distributions_path);
            config.corporate_actions.ex_date_distributions = legs.ex_date;
            config.corporate_actions.payment_date_distributions = legs.pay_date;
        }
        const char* daily_equity_basis =
            config.validation_day_boundaries_ns.empty() ? "utc_day" : "trading_day";

        const quant::BacktestEngine engine(registry, resolver, risk, config);
        const auto r = engine.run(bars, strategy);

        std::cout << "{\"strategy_fingerprint\":\"" << strategy.fingerprint() << "\""
                  << ",\"schedule_policy\":\"" << policy_str << "\""
                  << ",\"target_rows\":" << strategy.row_count()
                  << ",\"target_rows_applied\":" << strategy.rows_applied()
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
                  << ",\"slippage_ticks\":" << config.execution.slippage_ticks
                  << ",\"commission_per_contract_usd\":" << config.execution.commission_per_contract_usd
                  << ",\"unique_contracts\":" << r.contracts_traded.size()
                  << ",\"open_positions\":" << r.final_positions.size()
                  << ",\"starting_capital_usd\":" << r.portfolio_at_end.starting_capital_usd
                  << ",\"cash_usd\":" << r.portfolio_at_end.cash_usd
                  << ",\"equity_usd\":" << r.portfolio_at_end.equity_usd
                  << ",\"distribution_receivable_usd\":" << r.portfolio_at_end.distribution_receivable_usd
                  << ",\"distribution_income_usd\":" << r.portfolio_at_end.distribution_income_usd
                  << ",\"gross_exposure_usd\":" << r.portfolio_at_end.gross_exposure_usd
                  << ",\"net_exposure_usd\":" << r.portfolio_at_end.net_exposure_usd
                  << ",\"gross_leverage\":" << r.portfolio_at_end.gross_leverage
                  << ",\"peak_equity_usd\":" << r.portfolio_at_end.peak_equity_usd
                  << ",\"portfolio_drawdown_usd\":" << r.portfolio_at_end.drawdown_usd
                  << ",\"portfolio_drawdown_pct\":" << r.portfolio_at_end.drawdown_pct
                  << ",\"risk_rejects\":" << r.risk_rejects
                  << ",\"risk_resizes\":" << r.risk_resizes
                  << ",\"bars\":" << bars.size()
                  << ",\"contracts_resolved\":" << n_contracts
                  << ",\"net_equity_usd_at_end\":" << r.net_equity_usd_at_end;

        // ADDITIVE (Phase 13): chronological end-of-session-day equity trace, at
        // high precision so Python can difference it into a daily return series
        // without rounding noise. Each entry:
        //   [session_day_index, ts_ns, equity_usd, net_realized_pnl_usd,
        //    unrealized_pnl_usd, costs_usd, fills_cumulative, bars_cumulative]
        auto g = [](double v) {
            char buf[40];
            std::snprintf(buf, sizeof buf, "%.12g", v);
            return std::string(buf);
        };
        std::cout << ",\"daily_equity_basis\":\"" << daily_equity_basis << "\"";
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

        // ADDITIVE (Phase 15A): read-only serialisation of r.trades. Written
        // after the JSON line so a failure here can never truncate it.
        if (!trades_out_path.empty()) {
            std::ofstream tout(trades_out_path);
            if (!tout) {
                throw std::runtime_error("cannot open trades-out csv '" + trades_out_path + "'");
            }
            quant::write_closed_trades_csv(tout, r.trades);
            if (!tout) {
                throw std::runtime_error("failed writing trades-out csv '" + trades_out_path + "'");
            }
        }
        if (!fills_out_path.empty()) {
            std::ofstream fout(fills_out_path);
            if (!fout) {
                throw std::runtime_error("cannot open fills-out csv '" + fills_out_path + "'");
            }
            quant::write_fills_csv(fout, r.fills);
            if (!fout) {
                throw std::runtime_error("failed writing fills-out csv '" + fills_out_path + "'");
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
