#pragma once

// ============================================================================
// Phase 21 -- paper-trading scheduled-target run helper (additive)
// ============================================================================
//
// The Phase 11/19 reference wiring (targets_run.hpp, apps/backtest_targets_csv.cpp)
// deliberately runs the frozen historical research/validation path under
// `PassThroughRiskManager` -- that reference path is UNCHANGED by this file.
//
// Paper trading is a DIFFERENT, additive wiring: it replays the same
// precomputed target schedule through the same deterministic `BacktestEngine`,
// but gates every Order through the hard Phase 08 `PortfolioRiskManager`
// (position caps, gross/root/symbol exposure, drawdown / daily-loss kill
// switches, stale-mark protection) instead of pass-through. It adds NO new
// execution / accounting / fill semantics of its own -- every number still
// comes from `BacktestEngine::run` via a validated `Fill`; this header only
// changes WHICH `RiskManager` sits at the mandatory risk gate every Order
// passes through (risk_manager.hpp). Official PnL / fill / risk / accounting
// authority stays entirely in the C++ core (CLAUDE.md architecture boundary 1).
//
// PAPER-TRADING STEP MODEL. One call to `run_paper_trading_backtest` is one
// "step": replay the FULL predeclared window (window start .. current
// watermark) from scratch through the engine -- never an incremental /
// mutated in-process state -- so the result is always independently
// reproducible from (bars, schedule, config) alone, exactly the determinism
// guarantee `run_targets_backtest` gives the research path. The Python caller
// (alpha_agent.paper) diffs consecutive steps' `BacktestResult.fills` /
// `.trades` to find what is NEW since the previous step; it never recomputes
// a fill, a PnL number, or a risk decision itself.
//
// `end_of_test` defaults to `EndOfTestPolicy::LeaveOpen`: an open paper
// position MUST carry forward into the next step's replay -- forcing a
// liquidation every step would fabricate a round-trip trade that never really
// happened. `ForceLiquidateFinalClose` is for the one deliberate "flatten and
// stop this paper run" step.
//
// Margin: `RiskConfig.missing_margin` is left at its type default
// (`MissingMarginPolicy::Reject`) only when the caller supplies a real
// `MarginModel`. When no per-root margin schedule is available the caller
// (alpha_agent.paper.risk_policy) sets `MissingMarginPolicy::TreatAsZero`
// explicitly and disables `max_margin_utilization_pct` -- CLAUDE.md forbids
// guessing a margin figure from notional, so an unknown margin schedule is
// surfaced as `portfolio_at_end.margin_complete == false` (visible), never
// silently treated as satisfied, and margin utilization is not enforced as a
// limit against data that does not exist.

#include "quant_core/contract.hpp"
#include "quant_core/contract_registry.hpp"
#include "quant_core/contract_selector.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/margin.hpp"
#include "quant_core/risk_config.hpp"
#include "quant_core/risk_manager.hpp"
#include "quant_core/scheduled_target_strategy.hpp"
#include "quant_core/targets_run.hpp"  // reuse verify_targets_contract_resolution (identical check)
#include "quant_core/types.hpp"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quant {

struct PaperTradingRunConfig {
    std::string schedule_policy{"no_decision"};  // no_decision | flat | require_row | reemit_previous
    double commission_per_contract_usd{2.0};
    double slippage_ticks{0.0};
    double spread_ticks{0.0};
    // STRICTLY ASCENDING canonical CME trading-day boundary timestamps, exactly
    // as targets_run.hpp / EngineConfig. Non-empty => trading_day daily_equity.
    std::vector<std::int64_t> validation_day_boundaries_ns{};
    // Auxiliary same-timestamp OUTGOING-contract closes for the roll close-leg
    // only (Phase 13.5C semantics, unchanged). Empty => frozen RejectDefer.
    std::map<std::pair<std::uint32_t, std::int64_t>, double> roll_close_marks{};
    // ADDITIVE (News Alpha Phase H): ETF corporate actions (splits /
    // distributions), passed through to EngineConfig::corporate_actions
    // verbatim -- the engine already implements them (Phase 6). Empty (the
    // default, and every Phase 21 paper run) => byte-identical to before.
    CorporateActionsConfig corporate_actions{};
    // ADDITIVE (Phase H acceptance patch): ExecutionConfig's per-root
    // commission schedule, passed through verbatim. Empty => the scalar
    // commission_per_contract_usd for every root, byte-identical to before.
    std::map<std::string, double> commission_per_contract_usd_by_root{};
    // The mandatory hard-risk gate for THIS paper-trading run. Unlike the
    // research reference path (always PassThroughRiskManager), paper trading
    // always runs under PortfolioRiskManager -- there is no pass-through mode
    // here. A human sets this once when a paper run starts
    // (alpha_agent.paper.risk_policy.PaperRiskPolicy); an LLM never sets or
    // edits it (CLAUDE.md risk rule 1).
    RiskConfig risk_config{};
    // See file header: LeaveOpen carries a position across steps; use
    // ForceLiquidateFinalClose only for a deliberate final "flatten and stop".
    EndOfTestPolicy end_of_test{EndOfTestPolicy::LeaveOpen};
};

struct PaperTradingRunOutput {
    BacktestResult result;
    std::string    strategy_fingerprint;
    std::string    schedule_policy;
    std::size_t    target_rows{0};
    std::size_t    target_rows_applied{0};
    std::size_t    contracts_resolved{0};
    double         commission_per_contract_usd{0.0};
    double         slippage_ticks{0.0};
    double         spread_ticks{0.0};
    std::size_t    n_bars{0};
    std::string    daily_equity_basis;  // "utc_day" | "trading_day"
};

// Replay `rows` (already parsed) against `bars` through the deterministic
// BacktestEngine, gated by a `PortfolioRiskManager` built from
// `cfg.risk_config` (+ the optional `margin` schedule, nullable). `registry`
// and `margin` must outlive the call. `rows` is consumed. Throws
// std::runtime_error on the same structural violations as
// `run_targets_backtest` (unresolved / out-of-window bar, non-ascending
// validation days, negative cost overrides, a target root colliding with a
// real raw contract symbol).
inline PaperTradingRunOutput run_paper_trading_backtest(
    const std::vector<MarketBar>& bars, const ContractRegistry& registry,
    std::vector<ScheduledTarget> rows, const std::string& strategy_fingerprint,
    const PaperTradingRunConfig& cfg, const MarginModel* margin = nullptr) {
    if (cfg.commission_per_contract_usd < 0.0 || cfg.slippage_ticks < 0.0 ||
        cfg.spread_ticks < 0.0) {
        throw std::runtime_error("cost overrides must be non-negative");
    }
    for (const auto& [root, rate] : cfg.commission_per_contract_usd_by_root) {
        if (root.empty() || !(rate >= 0.0) || rate > 1e9) {
            throw std::runtime_error("commission schedule: root '" + root +
                                     "' needs a finite, non-negative rate");
        }
    }
    for (std::size_t i = 1; i < cfg.validation_day_boundaries_ns.size(); ++i) {
        if (cfg.validation_day_boundaries_ns[i] <= cfg.validation_day_boundaries_ns[i - 1]) {
            throw std::runtime_error("validation_day_boundaries_ns must be strictly ascending");
        }
    }

    const std::size_t n_contracts = verify_targets_contract_resolution(bars, registry);

    // A target root must not collide with a real raw contract symbol (the
    // schedule carries bare roots, never raw contracts) -- identical guard to
    // the research path.
    for (const auto& row : rows) {
        if (registry.find_by_raw_symbol(row.root_symbol) != nullptr) {
            throw std::runtime_error("targets: root_symbol '" + row.root_symbol +
                                     "' is a real raw contract in the registry");
        }
    }

    const auto policy = parse_absent_policy(cfg.schedule_policy);
    const ScheduledTargetStrategy strategy(std::move(rows), strategy_fingerprint, policy);

    const RegistryActiveContractResolver resolver(registry);
    // THE difference from the research reference path: a hard PortfolioRiskManager,
    // never PassThroughRiskManager, gates every Order this engine generates.
    const PortfolioRiskManager risk(cfg.risk_config, margin);
    EngineConfig config;
    config.execution.commission_per_contract_usd = cfg.commission_per_contract_usd;
    config.execution.slippage_ticks = cfg.slippage_ticks;
    config.execution.spread_ticks = cfg.spread_ticks;
    config.end_of_test = cfg.end_of_test;
    config.validation_day_boundaries_ns = cfg.validation_day_boundaries_ns;
    config.roll.close_marks = cfg.roll_close_marks;
    config.corporate_actions = cfg.corporate_actions;
    config.execution.commission_per_contract_usd_by_root = cfg.commission_per_contract_usd_by_root;

    const bool trading_day = !config.validation_day_boundaries_ns.empty();
    const BacktestEngine engine(registry, resolver, risk, config);

    PaperTradingRunOutput out;
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
