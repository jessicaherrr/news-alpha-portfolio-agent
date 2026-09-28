#pragma once

// ============================================================================
// Phase 19 -- thin scheduled-target run helper (pybind fast boundary only)
// ============================================================================
//
// A thin helper that replays a precomputed Phase 11 target schedule through the
// deterministic C++ BacktestEngine. It exists so the Phase 19 pybind11 fast
// boundary (bindings/pybind_module.cpp) has an in-process entry point with the
// same engine wiring the reference CLI uses.
//
// THIS IS NOT THE REFERENCE IMPLEMENTATION. apps/backtest_targets_csv.cpp is the
// frozen reference: it is wired independently and is NOT refactored onto this
// helper (that would touch the frozen official C++ execution path). This helper
// MIRRORS that wiring; it does not share code with the CLI.
//
// Semantic equivalence between the CLI and the pybind path is therefore an
// enforced invariant, not a structural guarantee: deterministic CLI-vs-pybind
// parity tests (tests/python/test_phase_19_pybind.py) run both transports on the
// same fixtures and assert every result field agrees. cpp/tests/test_targets_run.cpp
// additionally pins this helper against a hand-built BacktestEngine run. If the
// two wiring sites ever diverge, those tests fail.
//
// This header is engine WIRING ONLY. It adds NO execution, accounting, fill,
// roll, risk, portfolio or PnL semantics: it constructs the same objects the
// frozen CLI constructs -- a PassThroughRiskManager (frozen reference risk path),
// a RegistryActiveContractResolver, an EngineConfig with
// EndOfTestPolicy::ForceLiquidateFinalClose and the caller's cost overrides --
// and calls BacktestEngine::run. Official PnL / fill / risk / accounting
// authority stays entirely in the C++ core. CSV / JSON parsing and serialisation
// stay on the caller side; this helper owns no transport.
//
// Determinism: identical `bars` + `registry` + `rows` + `cfg` yield an identical
// BacktestResult, matching BacktestEngine::run's own guarantee. No timestamps,
// RNG, hash-order or filesystem access here.

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/scheduled_target_strategy.hpp"
#include "quant_core/types.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quant {

// Everything the wiring needs beyond the bars / registry / schedule rows. The
// three cost values default to the frozen Phase 11 reference assumptions; an
// empty `validation_day_boundaries_ns` reproduces the frozen legacy UTC-day
// daily-equity bucketing (diagnostics only); an empty `roll_close_marks` is the
// frozen RejectDefer roll path, byte-for-byte.
struct TargetsRunConfig {
    std::string schedule_policy{"no_decision"};  // no_decision | flat | require_row | reemit_previous
    double commission_per_contract_usd{2.0};
    double slippage_ticks{0.0};
    double spread_ticks{0.0};
    // STRICTLY ASCENDING canonical CME trading-day boundary timestamps (Phase
    // 13.1). Non-empty => BacktestResult::daily_equity is a trading_day series.
    std::vector<std::int64_t> validation_day_boundaries_ns{};
    // Auxiliary same-timestamp OUTGOING-contract closes for the roll close-leg
    // only (Phase 13.5C). Keyed (instrument_id, ts_event_ns) -> close; the map
    // key makes a duplicate impossible by construction.
    std::map<std::pair<std::uint32_t, std::int64_t>, double> roll_close_marks{};
};

struct TargetsRunOutput {
    BacktestResult result;
    std::string    strategy_fingerprint;
    std::string    schedule_policy;
    std::size_t    target_rows{0};
    std::size_t    target_rows_applied{0};
    std::size_t    contracts_resolved{0};
    // Echoed back so the caller serialises the SAME values the engine ran with.
    double         commission_per_contract_usd{0.0};
    double         slippage_ticks{0.0};
    double         spread_ticks{0.0};
    std::size_t    n_bars{0};
    std::string    daily_equity_basis;  // "utc_day" | "trading_day"
};

// Verify every bar resolves to a real contract live at its timestamp -- the same
// hard check apps/backtest_targets_csv.cpp performs before replay. Returns the
// number of distinct instruments seen. Throws std::runtime_error on any
// unresolved / out-of-window bar.
inline std::size_t verify_targets_contract_resolution(const std::vector<MarketBar>& bars,
                                                      const ContractRegistry& registry) {
    std::set<std::uint32_t> seen;
    for (const auto& bar : bars) {
        const ContractSpec* spec = registry.find_by_instrument_id(bar.instrument_id);
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

// Replay `rows` (already parsed) against `bars` through the deterministic
// BacktestEngine, with wiring that MIRRORS the frozen reference CLI
// (apps/backtest_targets_csv.cpp) -- it does not share code with it; the
// CLI-vs-pybind parity tests are what hold the two in agreement. `registry`
// must outlive the call. `rows` is consumed.
inline TargetsRunOutput run_targets_backtest(const std::vector<MarketBar>& bars,
                                             const ContractRegistry& registry,
                                             std::vector<ScheduledTarget> rows,
                                             const std::string& strategy_fingerprint,
                                             const TargetsRunConfig& cfg) {
    if (cfg.commission_per_contract_usd < 0.0 || cfg.slippage_ticks < 0.0 ||
        cfg.spread_ticks < 0.0) {
        throw std::runtime_error("cost overrides must be non-negative");
    }
    for (std::size_t i = 1; i < cfg.validation_day_boundaries_ns.size(); ++i) {
        if (cfg.validation_day_boundaries_ns[i] <= cfg.validation_day_boundaries_ns[i - 1]) {
            throw std::runtime_error("validation_day_boundaries_ns must be strictly ascending");
        }
    }

    const std::size_t n_contracts = verify_targets_contract_resolution(bars, registry);

    // A target root must not collide with a real raw contract symbol (the
    // schedule carries bare roots, never raw contracts).
    for (const auto& row : rows) {
        if (registry.find_by_raw_symbol(row.root_symbol) != nullptr) {
            throw std::runtime_error("targets: root_symbol '" + row.root_symbol +
                                     "' is a real raw contract in the registry");
        }
    }

    const auto policy = parse_absent_policy(cfg.schedule_policy);
    const ScheduledTargetStrategy strategy(std::move(rows), strategy_fingerprint, policy);

    const RegistryActiveContractResolver resolver(registry);
    const PassThroughRiskManager risk;  // frozen reference risk path
    EngineConfig config;
    config.execution.commission_per_contract_usd = cfg.commission_per_contract_usd;
    config.execution.slippage_ticks = cfg.slippage_ticks;
    config.execution.spread_ticks = cfg.spread_ticks;
    config.end_of_test = EndOfTestPolicy::ForceLiquidateFinalClose;
    config.validation_day_boundaries_ns = cfg.validation_day_boundaries_ns;
    config.roll.close_marks = cfg.roll_close_marks;

    const bool trading_day = !config.validation_day_boundaries_ns.empty();
    const BacktestEngine engine(registry, resolver, risk, config);

    TargetsRunOutput out;
    out.result = engine.run(bars, strategy);
    out.strategy_fingerprint = strategy.fingerprint();
    out.schedule_policy = cfg.schedule_policy;
    out.target_rows = strategy.row_count();
    out.target_rows_applied = strategy.rows_applied();
    out.contracts_resolved = n_contracts;
    out.commission_per_contract_usd = config.execution.commission_per_contract_usd;
    out.slippage_ticks = config.execution.slippage_ticks;
    out.spread_ticks = config.execution.spread_ticks;
    out.n_bars = bars.size();
    out.daily_equity_basis = trading_day ? "trading_day" : "utc_day";
    return out;
}

}  // namespace quant
