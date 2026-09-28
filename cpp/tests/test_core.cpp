#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/momentum_strategy.hpp"
#include "quant_core/risk.hpp"
#include "quant_core/risk_manager.hpp"

#include "test_support.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

void test_momentum_backtest_runs() {
    constexpr std::uint32_t kInstr = 1;
    constexpr std::int64_t  kMin   = 60'000'000'000LL;

    std::vector<quant::MarketBar> bars;
    double p = 100.0;
    for (int i = 0; i < 250; ++i) {
        const double open = p;
        p += (i < 130 ? 0.20 : -0.18);
        bars.push_back(quant::MarketBar{
            .ts_event_ns = static_cast<std::int64_t>(i + 1) * kMin,
            .instrument_id = kInstr,
            .open = open,
            .high = p + 0.1,
            .low = open - 0.1,
            .close = p,
            .volume = 100});
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
    const quant::TimeSeriesMomentum strategy(10, 0.005);
    const auto result = engine.run(bars, strategy);

    CHECK(!result.trades.empty());
    CHECK(result.fills_generated == result.orders_generated);  // no order silently dropped here
    CHECK(std::isfinite(result.net_realized_pnl_usd));
    CHECK(result.costs_usd > 0.0);
    // net realized = gross realized - costs must hold in aggregate.
    CHECK_CLOSE(result.net_realized_pnl_usd, result.gross_realized_pnl_usd - result.costs_usd, 1e-9);
    CHECK(result.max_drawdown_usd >= 0.0);
    // ForceLiquidate default -> flat at the end.
    CHECK(result.final_positions.empty());
    CHECK_CLOSE(result.unrealized_pnl_usd_at_end, 0.0, 1e-9);

    // Deterministic replay: identical run twice.
    const auto again = engine.run(bars, strategy);
    CHECK(again.fills_generated == result.fills_generated);
    CHECK_CLOSE(again.net_realized_pnl_usd, result.net_realized_pnl_usd, 1e-9);
    CHECK_CLOSE(again.max_drawdown_usd, result.max_drawdown_usd, 1e-9);
}

void test_risk_order_size_limits() {
    const quant::RiskLimits limits{};  // max_contracts_per_symbol == 5
    CHECK(quant::validate_order_size(0, limits));
    CHECK(quant::validate_order_size(3, limits));
    CHECK(quant::validate_order_size(5, limits));
    CHECK(!quant::validate_order_size(6, limits));
    CHECK(!quant::validate_order_size(9, limits));
    CHECK(!quant::validate_order_size(-1, limits));
}

}  // namespace

int main() {
    test_momentum_backtest_runs();
    test_risk_order_size_limits();
    return quant::test::summary("quant_core_tests");
}
