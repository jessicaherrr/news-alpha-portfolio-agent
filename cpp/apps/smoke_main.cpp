#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/momentum_strategy.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/types.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr std::uint32_t kInstr = 1;
    constexpr std::int64_t  kMin   = 60'000'000'000LL;

    std::vector<quant::MarketBar> bars;
    double price = 100.0;
    for (int i = 0; i < 300; ++i) {
        const double drift = (i < 170) ? 0.15 : -0.12;
        const double open = price;
        price += drift;
        bars.push_back(quant::MarketBar{
            .ts_event_ns = static_cast<std::int64_t>(i + 1) * kMin,
            .instrument_id = kInstr,
            .open = open,
            .high = price + 0.1,
            .low = open - 0.1,
            .close = price,
            .volume = 1000});
    }

    quant::ContractRegistry registry;
    registry.add(quant::ContractSpec{
        .instrument_id = kInstr, .raw_symbol = "NQH6", .root_symbol = "NQ", .exchange = "XCME",
        .tick_size = 0.25, .multiplier = 20.0,
        .activation_ns = 1, .expiration_ns = bars.back().ts_event_ns + kMin});

    const quant::RegistryActiveContractResolver resolver(registry);
    const quant::PassThroughRiskManager risk;
    quant::EngineConfig config;
    config.execution.commission_per_contract_usd = 2.0;

    const quant::BacktestEngine engine(registry, resolver, risk, config);
    const quant::TimeSeriesMomentum strategy(20, 0.01);
    const auto result = engine.run(bars, strategy);

    std::cout << "events=" << result.events_processed
              << " signals=" << result.signals_generated
              << " orders=" << result.orders_generated
              << " fills=" << result.fills_generated
              << " trades=" << result.closed_trades
              << " net_pnl=" << result.net_realized_pnl_usd
              << " max_dd=" << result.max_drawdown_usd << '\n';
    return result.closed_trades == 0 ? 1 : 0;
}
