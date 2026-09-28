// ============================================================================
// Phase 19 -- pybind11 fast boundary
// ============================================================================
//
// Direct in-process replacement for the CSV-bundle + subprocess + JSON round
// trip of alpha_agent.adapters.targets_bridge, for HIGH-VOLUME research (the
// Phase 13.5C matrix, the Phase 15B economics trials). It is deliberately THIN:
//
//   * it re-uses the frozen boundary parsers verbatim -- quant::load_contract_
//     registry for contracts.csv and quant::parse_targets_csv for targets.csv
//     (the same Phase 02.5 / Phase 11 boundary the CLI parses);
//   * it calls the thin helper quant::run_targets_backtest
//     (quant_core/targets_run.hpp), whose engine wiring MIRRORS the frozen
//     reference CLI apps/backtest_targets_csv.cpp. The two are wired
//     independently -- the CLI is NOT refactored onto this helper -- so
//     equivalence is an ENFORCED invariant, held by the deterministic
//     CLI-vs-pybind parity tests (tests/python/test_phase_19_pybind.py), not a
//     structural guarantee from shared code;
//   * only the BAR ARRAY -- the one high-volume payload -- crosses as typed
//     NumPy columns instead of a CSV, forcecast to the engine's own dtypes and
//     copied once into std::vector<quant::MarketBar>.
//
// apps/backtest_targets_csv.cpp remains the frozen reference implementation and
// is unchanged.
//
// It moves NOTHING into Python. Official PnL / fill / risk / contract resolution
// / roll pricing / portfolio accounting / the daily-equity trace authority stays
// entirely in the C++ core, exactly as on the CLI path; Python still only reads
// what the engine already decided. No mutable risk internals are exposed:
// RiskConfig, the RiskManager and PortfolioAccountant are not bound, and this
// path runs PassThroughRiskManager just like the frozen CLI.
//
// Build: cmake -DBUILD_PYBIND=ON (see scripts/build_pybind.sh). The module is
// optional -- core C++ and Python tests do not need it.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "quant_core/contract_io.hpp"
#include "quant_core/engine.hpp"
#include "quant_core/scheduled_target_strategy.hpp"
#include "quant_core/targets_run.hpp"
#include "quant_core/trade_export.hpp"
#include "quant_core/types.hpp"

#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using I64 = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

// Build the engine's bar vector from typed NumPy columns. Every column is
// forcecast to the engine dtype and validated; a single contiguous copy per
// column, then one std::vector<MarketBar>. No look-ahead, no reordering -- the
// engine still canonically orders (ts_event_ns, instrument_id) and stamps seq.
std::vector<quant::MarketBar> bars_from_columns(const I64& ts_event_ns,
                                                const I64& instrument_id,
                                                const F64& open,
                                                const F64& high,
                                                const F64& low,
                                                const F64& close,
                                                const I64& volume) {
    const auto n = static_cast<std::size_t>(ts_event_ns.size());
    const std::pair<const char*, std::size_t> cols[] = {
        {"instrument_id", static_cast<std::size_t>(instrument_id.size())},
        {"open", static_cast<std::size_t>(open.size())},
        {"high", static_cast<std::size_t>(high.size())},
        {"low", static_cast<std::size_t>(low.size())},
        {"close", static_cast<std::size_t>(close.size())},
        {"volume", static_cast<std::size_t>(volume.size())},
    };
    for (const auto& [name, sz] : cols) {
        if (sz != n) {
            throw std::runtime_error(std::string("bars column '") + name + "' has length " +
                                     std::to_string(sz) + " but ts_event_ns has " +
                                     std::to_string(n));
        }
    }
    const auto* ts = ts_event_ns.data();
    const auto* iid = instrument_id.data();
    const auto* o = open.data();
    const auto* h = high.data();
    const auto* l = low.data();
    const auto* c = close.data();
    const auto* v = volume.data();

    std::vector<quant::MarketBar> bars;
    bars.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (iid[i] <= 0 || iid[i] > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error("bars: instrument_id out of uint32 range at row " +
                                     std::to_string(i));
        }
        bars.push_back(quant::MarketBar{
            .ts_event_ns = ts[i],
            .instrument_id = static_cast<std::uint32_t>(iid[i]),
            .open = o[i],
            .high = h[i],
            .low = l[i],
            .close = c[i],
            .volume = v[i],
        });
    }
    return bars;
}

// Owns a completed run. Read-only: every accessor copies a value the engine
// already computed. The property names mirror the CLI JSON keys 1:1 so the
// Python adapter builds the identical result dict (docs/BOUNDARY_CONTRACT.md G).
class TargetsRunResult {
public:
    explicit TargetsRunResult(quant::TargetsRunOutput out) : out_(std::move(out)) {}

    const quant::BacktestResult& r() const { return out_.result; }
    const quant::PortfolioState& p() const { return out_.result.portfolio_at_end; }

    std::string strategy_fingerprint() const { return out_.strategy_fingerprint; }
    std::string schedule_policy() const { return out_.schedule_policy; }
    std::size_t target_rows() const { return out_.target_rows; }
    std::size_t target_rows_applied() const { return out_.target_rows_applied; }
    std::size_t contracts_resolved() const { return out_.contracts_resolved; }
    std::size_t n_bars() const { return out_.n_bars; }
    std::string daily_equity_basis() const { return out_.daily_equity_basis; }
    double commission_per_contract_usd() const { return out_.commission_per_contract_usd; }
    double slippage_ticks() const { return out_.slippage_ticks; }

    std::size_t trades() const { return r().closed_trades; }
    double gross_pnl_usd() const { return r().gross_realized_pnl_usd; }
    double costs_usd() const { return r().costs_usd; }
    double net_pnl_usd() const { return r().net_realized_pnl_usd; }
    double max_drawdown_usd() const { return r().max_drawdown_usd; }
    double unrealized_pnl_usd() const { return r().unrealized_pnl_usd_at_end; }
    double net_equity_usd_at_end() const { return r().net_equity_usd_at_end; }
    std::size_t events() const { return r().events_processed; }
    std::size_t signals() const { return r().signals_generated; }
    std::size_t orders() const { return r().orders_generated; }
    std::size_t fills() const { return r().fills_generated; }
    std::size_t rolls() const { return r().rolls; }
    std::size_t rolls_priced_contemporaneous() const { return r().rolls_priced_contemporaneous; }
    std::size_t rolls_priced_auxiliary_marks() const { return r().rolls_priced_auxiliary_marks; }
    std::size_t rolls_priced_stale() const { return r().rolls_priced_stale; }
    std::size_t rolls_deferred() const { return r().rolls_deferred; }
    std::size_t non_fills() const { return r().non_fills; }
    std::size_t eot_liquidations() const { return r().eot_liquidations.size(); }
    std::size_t unique_contracts() const { return r().contracts_traded.size(); }
    std::size_t open_positions() const { return r().final_positions.size(); }
    std::size_t risk_rejects() const { return r().risk_rejects; }
    std::size_t risk_resizes() const { return r().risk_resizes; }

    double starting_capital_usd() const { return p().starting_capital_usd; }
    double cash_usd() const { return p().cash_usd; }
    double equity_usd() const { return p().equity_usd; }
    double gross_exposure_usd() const { return p().gross_exposure_usd; }
    double net_exposure_usd() const { return p().net_exposure_usd; }
    double gross_leverage() const { return p().gross_leverage; }
    double peak_equity_usd() const { return p().peak_equity_usd; }
    double portfolio_drawdown_usd() const { return p().drawdown_usd; }
    double portfolio_drawdown_pct() const { return p().drawdown_pct; }

    // Same 8-column layout the CLI emits in its "daily_equity" JSON array:
    //   [session_day_index, ts_ns, equity_usd, net_realized_pnl_usd,
    //    unrealized_pnl_usd, costs_usd, fills_cumulative, bars_cumulative]
    py::list daily_equity() const {
        py::list rows;
        for (const auto& d : r().daily_equity) {
            py::list row;
            row.append(static_cast<std::int64_t>(d.session_day_index));
            row.append(static_cast<std::int64_t>(d.ts_ns));
            row.append(d.equity_usd);
            row.append(d.net_realized_pnl_usd);
            row.append(d.unrealized_pnl_usd);
            row.append(d.costs_usd);
            row.append(static_cast<std::int64_t>(d.fills_cumulative));
            row.append(static_cast<std::int64_t>(d.bars_cumulative));
            rows.append(std::move(row));
        }
        return rows;
    }

    // ADDITIVE, READ-ONLY audit exports -- byte-identical to the CLI's
    // --trades-out / --fills-out files (they call the same writers). Written
    // outside the event loop; they change nothing.
    void write_trades_csv(const std::string& path) const {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("cannot open trades-out csv '" + path + "'");
        quant::write_closed_trades_csv(f, r().trades);
        if (!f) throw std::runtime_error("failed writing trades-out csv '" + path + "'");
    }
    void write_fills_csv(const std::string& path) const {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("cannot open fills-out csv '" + path + "'");
        quant::write_fills_csv(f, r().fills);
        if (!f) throw std::runtime_error("failed writing fills-out csv '" + path + "'");
    }

private:
    quant::TargetsRunOutput out_;
};

TargetsRunResult run_scheduled_targets(
    I64 ts_event_ns, I64 instrument_id, F64 open, F64 high, F64 low, F64 close, I64 volume,
    const std::string& contracts_csv_path,
    const std::string& targets_csv_path,
    const std::string& schedule_policy,
    double commission_per_contract_usd,
    double slippage_ticks,
    double spread_ticks,
    std::vector<std::int64_t> validation_day_boundaries_ns,
    std::vector<std::tuple<std::int64_t, std::int64_t, double>> roll_close_marks) {

    auto bars = bars_from_columns(ts_event_ns, instrument_id, open, high, low, close, volume);
    const auto registry = quant::load_contract_registry(contracts_csv_path);

    std::string fingerprint;
    auto rows = quant::parse_targets_csv(targets_csv_path, fingerprint);

    quant::TargetsRunConfig cfg;
    cfg.schedule_policy = schedule_policy;
    cfg.commission_per_contract_usd = commission_per_contract_usd;
    cfg.slippage_ticks = slippage_ticks;
    cfg.spread_ticks = spread_ticks;
    cfg.validation_day_boundaries_ns = std::move(validation_day_boundaries_ns);
    for (const auto& [iid, ts, close_px] : roll_close_marks) {
        if (iid <= 0 || iid > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error("roll_close_marks: instrument_id out of uint32 range");
        }
        const auto key = std::pair<std::uint32_t, std::int64_t>{static_cast<std::uint32_t>(iid), ts};
        if (!cfg.roll_close_marks.emplace(key, close_px).second) {
            throw std::runtime_error("roll_close_marks: duplicate (instrument_id, ts_event_ns)");
        }
    }

    return TargetsRunResult(
        quant::run_targets_backtest(bars, registry, std::move(rows), fingerprint, cfg));
}

}  // namespace

PYBIND11_MODULE(quant_core_py, m) {
    m.doc() = "Phase 19 pybind11 fast boundary to the deterministic C++ quant core. "
              "Thin transport only: no PnL / fill / risk / validation / identity authority "
              "lives here (CLAUDE.md architecture boundary 3).";

    // Boundary identity. Bump BOUNDARY_ABI whenever the argument contract or the
    // result surface of run_scheduled_targets changes, so a stale build is
    // caught. This is TRANSPORT provenance, never research identity.
    m.attr("BOUNDARY_ABI") = "phase19-targets/1";
    m.attr("ENGINE_PATH") = "cpp_quant_core__run_targets_backtest";

    py::class_<TargetsRunResult>(m, "TargetsRunResult")
        .def_property_readonly("strategy_fingerprint", &TargetsRunResult::strategy_fingerprint)
        .def_property_readonly("schedule_policy", &TargetsRunResult::schedule_policy)
        .def_property_readonly("target_rows", &TargetsRunResult::target_rows)
        .def_property_readonly("target_rows_applied", &TargetsRunResult::target_rows_applied)
        .def_property_readonly("contracts_resolved", &TargetsRunResult::contracts_resolved)
        .def_property_readonly("bars", &TargetsRunResult::n_bars)
        .def_property_readonly("daily_equity_basis", &TargetsRunResult::daily_equity_basis)
        .def_property_readonly("commission_per_contract_usd",
                               &TargetsRunResult::commission_per_contract_usd)
        .def_property_readonly("slippage_ticks", &TargetsRunResult::slippage_ticks)
        .def_property_readonly("trades", &TargetsRunResult::trades)
        .def_property_readonly("gross_pnl_usd", &TargetsRunResult::gross_pnl_usd)
        .def_property_readonly("costs_usd", &TargetsRunResult::costs_usd)
        .def_property_readonly("net_pnl_usd", &TargetsRunResult::net_pnl_usd)
        .def_property_readonly("max_drawdown_usd", &TargetsRunResult::max_drawdown_usd)
        .def_property_readonly("unrealized_pnl_usd", &TargetsRunResult::unrealized_pnl_usd)
        .def_property_readonly("net_equity_usd_at_end", &TargetsRunResult::net_equity_usd_at_end)
        .def_property_readonly("events", &TargetsRunResult::events)
        .def_property_readonly("signals", &TargetsRunResult::signals)
        .def_property_readonly("orders", &TargetsRunResult::orders)
        .def_property_readonly("fills", &TargetsRunResult::fills)
        .def_property_readonly("rolls", &TargetsRunResult::rolls)
        .def_property_readonly("rolls_priced_contemporaneous",
                               &TargetsRunResult::rolls_priced_contemporaneous)
        .def_property_readonly("rolls_priced_auxiliary_marks",
                               &TargetsRunResult::rolls_priced_auxiliary_marks)
        .def_property_readonly("rolls_priced_stale", &TargetsRunResult::rolls_priced_stale)
        .def_property_readonly("rolls_deferred", &TargetsRunResult::rolls_deferred)
        .def_property_readonly("non_fills", &TargetsRunResult::non_fills)
        .def_property_readonly("eot_liquidations", &TargetsRunResult::eot_liquidations)
        .def_property_readonly("unique_contracts", &TargetsRunResult::unique_contracts)
        .def_property_readonly("open_positions", &TargetsRunResult::open_positions)
        .def_property_readonly("risk_rejects", &TargetsRunResult::risk_rejects)
        .def_property_readonly("risk_resizes", &TargetsRunResult::risk_resizes)
        .def_property_readonly("starting_capital_usd", &TargetsRunResult::starting_capital_usd)
        .def_property_readonly("cash_usd", &TargetsRunResult::cash_usd)
        .def_property_readonly("equity_usd", &TargetsRunResult::equity_usd)
        .def_property_readonly("gross_exposure_usd", &TargetsRunResult::gross_exposure_usd)
        .def_property_readonly("net_exposure_usd", &TargetsRunResult::net_exposure_usd)
        .def_property_readonly("gross_leverage", &TargetsRunResult::gross_leverage)
        .def_property_readonly("peak_equity_usd", &TargetsRunResult::peak_equity_usd)
        .def_property_readonly("portfolio_drawdown_usd", &TargetsRunResult::portfolio_drawdown_usd)
        .def_property_readonly("portfolio_drawdown_pct", &TargetsRunResult::portfolio_drawdown_pct)
        .def_property_readonly("daily_equity", &TargetsRunResult::daily_equity)
        .def("write_trades_csv", &TargetsRunResult::write_trades_csv, py::arg("path"))
        .def("write_fills_csv", &TargetsRunResult::write_fills_csv, py::arg("path"));

    m.def("run_scheduled_targets", &run_scheduled_targets,
          py::arg("ts_event_ns"), py::arg("instrument_id"), py::arg("open"), py::arg("high"),
          py::arg("low"), py::arg("close"), py::arg("volume"),
          py::arg("contracts_csv_path"), py::arg("targets_csv_path"),
          py::arg("schedule_policy") = "no_decision",
          py::arg("commission_per_contract_usd") = 2.0,
          py::arg("slippage_ticks") = 0.0,
          py::arg("spread_ticks") = 0.0,
          py::arg("validation_day_boundaries_ns") = std::vector<std::int64_t>{},
          py::arg("roll_close_marks") =
              std::vector<std::tuple<std::int64_t, std::int64_t, double>>{},
          "Replay a Phase 11 target schedule through the deterministic C++ "
          "BacktestEngine in-process. Engine wiring mirrors the frozen reference "
          "CLI quant_backtest_targets_csv (independently wired -- equivalence is "
          "enforced by the CLI-vs-pybind parity tests); only the bar array "
          "crosses as typed NumPy columns instead of bars.csv.");
}
