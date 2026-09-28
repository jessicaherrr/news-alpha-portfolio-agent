# Back-adjustment is intentionally NOT here. It is research-data logic and lives
# in the Python layer (alpha_agent.data.backadjust). The deterministic C++ core
# only ever sees real raw-contract prices (PriceDomain::RawContract).
add_library(quant_core
  src/contract.cpp
  src/contract_io.cpp
  src/contract_registry.cpp
  src/contract_selector.cpp
  src/domain_model.cpp
  src/engine.cpp
  src/execution_simulator.cpp
  src/fill.cpp
  src/margin.cpp
  src/momentum_strategy.cpp
  src/portfolio.cpp
  src/portfolio_construction.cpp
  src/position_ledger.cpp
  src/risk_manager.cpp
  src/scheduled_target_strategy.cpp
)

target_include_directories(quant_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(quant_core PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

add_executable(quant_core_smoke apps/smoke_main.cpp)
target_link_libraries(quant_core_smoke PRIVATE quant_core)

add_executable(quant_backtest_csv apps/backtest_csv.cpp)
target_link_libraries(quant_backtest_csv PRIVATE quant_core)

add_executable(quant_backtest_targets_csv apps/backtest_targets_csv.cpp)
target_link_libraries(quant_backtest_targets_csv PRIVATE quant_core)

# Phase 21 -- additive paper-trading CLI: identical bars/contracts/targets
# boundary, but gated by the hard PortfolioRiskManager (quant_core/
# paper_trading_run.hpp) instead of quant_backtest_targets_csv's pass-through
# reference risk. Shares no code with the frozen research CLI above.
add_executable(quant_paper_trading_targets_csv apps/paper_trading_targets_csv.cpp)
target_link_libraries(quant_paper_trading_targets_csv PRIVATE quant_core)

# News Alpha Phase G -- additive deterministic multi-asset portfolio construction
# (quant_core/portfolio_construction.hpp) and the replay of a plan's targets
# through the unchanged engine under the hard PortfolioRiskManager (wired like
# the Phase 21 paper-trading helper). Neither touches the frozen CLIs above.
add_executable(quant_portfolio_construct_csv apps/portfolio_construct_csv.cpp)
target_link_libraries(quant_portfolio_construct_csv PRIVATE quant_core)

add_executable(quant_portfolio_plan_replay_csv apps/portfolio_plan_replay_csv.cpp)
target_link_libraries(quant_portfolio_plan_replay_csv PRIVATE quant_core)

# Keep assertions live in test binaries even for Release builds. The custom CHECK
# macros in tests/test_support.hpp are NDEBUG-independent, but this also protects
# any future <cassert> usage in tests from being silently stripped.
add_executable(quant_core_tests tests/test_core.cpp)
target_link_libraries(quant_core_tests PRIVATE quant_core)
target_compile_options(quant_core_tests PRIVATE -UNDEBUG)
add_test(NAME quant_core_tests COMMAND quant_core_tests)

add_executable(quant_contract_boundary_tests tests/test_contract_boundary.cpp)
target_link_libraries(quant_contract_boundary_tests PRIVATE quant_core)
target_compile_options(quant_contract_boundary_tests PRIVATE -UNDEBUG)
add_test(NAME quant_contract_boundary_tests COMMAND quant_contract_boundary_tests)

add_executable(quant_domain_model_tests tests/test_domain_model.cpp)
target_link_libraries(quant_domain_model_tests PRIVATE quant_core)
target_compile_options(quant_domain_model_tests PRIVATE -UNDEBUG)
add_test(NAME quant_domain_model_tests COMMAND quant_domain_model_tests)

add_executable(quant_engine_tests tests/test_engine.cpp)
target_link_libraries(quant_engine_tests PRIVATE quant_core)
target_compile_options(quant_engine_tests PRIVATE -UNDEBUG)
add_test(NAME quant_engine_tests COMMAND quant_engine_tests)

add_executable(quant_execution_simulator_tests tests/test_execution_simulator.cpp)
target_link_libraries(quant_execution_simulator_tests PRIVATE quant_core)
target_compile_options(quant_execution_simulator_tests PRIVATE -UNDEBUG)
add_test(NAME quant_execution_simulator_tests COMMAND quant_execution_simulator_tests)

add_executable(quant_portfolio_tests tests/test_portfolio.cpp)
target_link_libraries(quant_portfolio_tests PRIVATE quant_core)
target_compile_options(quant_portfolio_tests PRIVATE -UNDEBUG)
add_test(NAME quant_portfolio_tests COMMAND quant_portfolio_tests)

# Phase 6 ETF pilot -- additive corporate actions (splits / cash distributions).
# Shares no Futures-path code; see quant_core/corporate_actions.hpp.
add_executable(quant_corporate_actions_tests tests/test_corporate_actions.cpp)
target_link_libraries(quant_corporate_actions_tests PRIVATE quant_core)
target_compile_options(quant_corporate_actions_tests PRIVATE -UNDEBUG)
add_test(NAME quant_corporate_actions_tests COMMAND quant_corporate_actions_tests)

# Corporate actions wired into the full deterministic BacktestEngine replay
# (EngineConfig::corporate_actions) -- mirrors RollExecutionPolicy::close_marks's
# wiring; needs engine.hpp/bar/registry fixtures, so it is a separate binary from
# the standalone PortfolioAccountant-level tests above.
add_executable(quant_corporate_actions_engine_tests tests/test_corporate_actions_engine.cpp)
target_link_libraries(quant_corporate_actions_engine_tests PRIVATE quant_core)
target_compile_options(quant_corporate_actions_engine_tests PRIVATE -UNDEBUG)
add_test(NAME quant_corporate_actions_engine_tests COMMAND quant_corporate_actions_engine_tests)

add_executable(quant_risk_manager_tests tests/test_risk_manager.cpp)
target_link_libraries(quant_risk_manager_tests PRIVATE quant_core)
target_compile_options(quant_risk_manager_tests PRIVATE -UNDEBUG)
add_test(NAME quant_risk_manager_tests COMMAND quant_risk_manager_tests)

add_executable(quant_scheduled_target_tests tests/test_scheduled_target.cpp)
target_link_libraries(quant_scheduled_target_tests PRIVATE quant_core)
target_compile_options(quant_scheduled_target_tests PRIVATE -UNDEBUG)
add_test(NAME quant_scheduled_target_tests COMMAND quant_scheduled_target_tests)

# Phase 19: the thin scheduled-target run helper the pybind fast boundary calls
# (quant_core/targets_run.hpp). Its wiring mirrors the frozen reference CLI
# apps/backtest_targets_csv.cpp; the CLI is unchanged and shares no code with it.
add_executable(quant_targets_run_tests tests/test_targets_run.cpp)
target_link_libraries(quant_targets_run_tests PRIVATE quant_core)
target_compile_options(quant_targets_run_tests PRIVATE -UNDEBUG)
add_test(NAME quant_targets_run_tests COMMAND quant_targets_run_tests)

# Phase 21 -- quant_core/paper_trading_run.hpp (hard-risk-gated paper trading
# wiring; mirrors targets_run.hpp's pattern, does not share code with it).
add_executable(quant_paper_trading_run_tests tests/test_paper_trading_run.cpp)
target_link_libraries(quant_paper_trading_run_tests PRIVATE quant_core)
target_compile_options(quant_paper_trading_run_tests PRIVATE -UNDEBUG)
add_test(NAME quant_paper_trading_run_tests COMMAND quant_paper_trading_run_tests)

add_executable(quant_trade_export_tests tests/test_trade_export.cpp)
target_link_libraries(quant_trade_export_tests PRIVATE quant_core)
target_compile_options(quant_trade_export_tests PRIVATE -UNDEBUG)
add_test(NAME quant_trade_export_tests COMMAND quant_trade_export_tests)

add_executable(quant_latency_queue_tests tests/test_latency_queue.cpp)
target_link_libraries(quant_latency_queue_tests PRIVATE quant_core)
target_compile_options(quant_latency_queue_tests PRIVATE -UNDEBUG)
add_test(NAME quant_latency_queue_tests COMMAND quant_latency_queue_tests)

add_executable(quant_validation_trace_tests tests/test_validation_trace.cpp)
target_link_libraries(quant_validation_trace_tests PRIVATE quant_core)
target_compile_options(quant_validation_trace_tests PRIVATE -UNDEBUG)
add_test(NAME quant_validation_trace_tests COMMAND quant_validation_trace_tests)

add_executable(quant_portfolio_construction_tests tests/test_portfolio_construction.cpp)
target_link_libraries(quant_portfolio_construction_tests PRIVATE quant_core)
target_compile_options(quant_portfolio_construction_tests PRIVATE -UNDEBUG)
add_test(NAME quant_portfolio_construction_tests COMMAND quant_portfolio_construction_tests)

# News Alpha Phase H acceptance patch -- per-root commission schedule
# (ExecutionConfig::commission_per_contract_usd_by_root).
add_executable(quant_commission_schedule_tests tests/test_commission_schedule.cpp)
target_link_libraries(quant_commission_schedule_tests PRIVATE quant_core)
target_compile_options(quant_commission_schedule_tests PRIVATE -UNDEBUG)
add_test(NAME quant_commission_schedule_tests COMMAND quant_commission_schedule_tests)
