// Reference CLI boundary between the Python research layer and the C++ core.
//
// Usage:
//   quant_backtest_csv <bars.csv> <contracts.csv> <lookback> <threshold_return>
//
// bars.csv header (exact order, see docs/BOUNDARY_CONTRACT.md section G):
//   ts_event_ns,instrument_id,open,high,low,close,volume
// contracts.csv: see docs/BOUNDARY_CONTRACT.md section G / parse_contracts_csv.
//
// Phase 06: this runs the deterministic event-driven BacktestEngine. It loads
// the contract registry, hard-verifies every bar resolves to a real contract
// live at its timestamp, then replays the bars through the engine. The engine
// orders bars canonically by (ts_event_ns, instrument_id) and stamps `seq`
// itself -- CSV row order does not affect the result. Output is one JSON line.
// The legacy PnL fields are derived from the new engine (realized-from-Fills
// only), not the retired ad-hoc math.

#include "quant_core/contract_io.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/detail/csv.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/momentum_strategy.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/types.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
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

// Every bar must resolve to a real contract that was live at the bar timestamp.
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
    if (argc < 5) {
        std::cerr << "usage: quant_backtest_csv <bars.csv> <contracts.csv> "
                     "<lookback> <threshold_return>\n";
        return 2;
    }
    try {
        const std::string bars_path = argv[1];
        const std::string contracts_path = argv[2];
        const auto lookback = static_cast<std::size_t>(std::stoul(argv[3]));
        const double threshold = std::stod(argv[4]);

        const auto bars = read_bars(bars_path);
        const auto registry = quant::load_contract_registry(contracts_path);
        const std::size_t n_contracts = verify_contract_resolution(bars, registry);

        const quant::RegistryActiveContractResolver resolver(registry);
        const quant::PassThroughRiskManager risk;
        quant::EngineConfig config;
        // Phase 07: execution modelling is explicit and configurable. The frozen
        // CLI signature stays reproducible -- a flat commission, no slippage, a
        // Market order at the execution bar's open -- until the pybind boundary
        // (Phase 19) exposes ExecutionConfig directly.
        config.execution.commission_per_contract_usd = 2.0;
        config.execution.slippage_ticks = 0.0;
        config.execution.spread_ticks = 0.0;
        config.end_of_test = quant::EndOfTestPolicy::ForceLiquidateFinalClose;

        const quant::BacktestEngine engine(registry, resolver, risk, config);
        const quant::TimeSeriesMomentum strategy(lookback, threshold);
        const auto r = engine.run(bars, strategy);

        std::cout << "{\"trades\":" << r.closed_trades
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
                  << ",\"rolls_priced_stale\":" << r.rolls_priced_stale
                  << ",\"rolls_deferred\":" << r.rolls_deferred
                  << ",\"roll_fallback_audit_records\":" << r.roll_fallback_audit.size()
                  << ",\"non_fills\":" << r.non_fills
                  << ",\"eot_liquidations\":" << r.eot_liquidations.size()
                  << ",\"eot_positions_left_open_stale\":" << r.eot_positions_left_open_stale
                  << ",\"slippage_ticks\":" << config.execution.slippage_ticks
                  << ",\"commission_per_contract_usd\":" << config.execution.commission_per_contract_usd
                  << ",\"unique_contracts\":" << r.contracts_traded.size()
                  << ",\"open_positions\":" << r.final_positions.size()
                  // Phase 08 portfolio view (reporting-only on the frozen CLI path:
                  // it runs PassThroughRiskManager, so no hard limits are enforced
                  // here -- pybind (Phase 19) wires RiskConfig / MarginModel).
                  << ",\"starting_capital_usd\":" << r.portfolio_at_end.starting_capital_usd
                  << ",\"cash_usd\":" << r.portfolio_at_end.cash_usd
                  << ",\"equity_usd\":" << r.portfolio_at_end.equity_usd
                  << ",\"gross_exposure_usd\":" << r.portfolio_at_end.gross_exposure_usd
                  << ",\"net_exposure_usd\":" << r.portfolio_at_end.net_exposure_usd
                  << ",\"gross_leverage\":" << r.portfolio_at_end.gross_leverage
                  << ",\"initial_margin_usd\":" << r.portfolio_at_end.initial_margin_usd
                  << ",\"margin_utilization_pct\":" << r.portfolio_at_end.margin_utilization_pct
                  << ",\"margin_complete\":" << (r.portfolio_at_end.margin_complete ? "true" : "false")
                  << ",\"valuation_complete\":" << (r.portfolio_at_end.valuation_complete ? "true" : "false")
                  << ",\"peak_equity_usd\":" << r.portfolio_at_end.peak_equity_usd
                  << ",\"portfolio_drawdown_usd\":" << r.portfolio_at_end.drawdown_usd
                  << ",\"portfolio_drawdown_pct\":" << r.portfolio_at_end.drawdown_pct
                  << ",\"risk_rejects\":" << r.risk_rejects
                  << ",\"risk_resizes\":" << r.risk_resizes
                  << ",\"bars\":" << bars.size()
                  << ",\"contracts_resolved\":" << n_contracts
                  << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
