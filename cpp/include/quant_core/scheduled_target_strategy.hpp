#pragma once

#include "quant_core/strategy.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace quant {

// ============================================================================
// Phase 11 -- ScheduledTargetStrategy
// ============================================================================
//
// The smallest C++ Strategy adapter that replays a PRECOMPUTED target schedule
// produced by the Python Phase 10 reference evaluator (targets.csv, see
// docs/BASELINE_STRATEGIES.md / docs/BOUNDARY_CONTRACT.md).
//
// At MarketEvent T it looks up (root_symbol, T) in the schedule:
//   * row present  -> emit Signal(target_units) exactly as decided in Python
//   * row absent    -> NO DECISION (std::nullopt) under the default policy
//
// "No row at T" == "no new strategy decision" -- NOT an implicit flat and NOT an
// implicit re-emission of a previous target. In particular a target that was
// risk-rejected on an earlier bar is never automatically retried just because a
// later bar has no schedule row.
//
// It is a NORMAL Strategy and goes through the existing engine unchanged: the
// engine still stamps the signal id, resolves the real execution contract via
// MarketState -> ActiveContractResolver, runs the mandatory RiskManager gate,
// and simulates the Fill at the next eligible bar (decision time != execution
// time -- no pre-shift here, the Python schedule row is stamped at decision-time
// T and the C++ engine adds the execution latency).
//
// It MUST NOT and CAN NOT: know a fill price, select a contract, read the
// RiskManager / Portfolio, or compute PnL. In the normal path it stores NO
// previous-target state at all -- it only sees the StrategyContext (decision ts +
// root) and its own immutable schedule.
//
// KEEP_PREVIOUS_TARGET is Phase 10 Python semantics: the reference evaluator
// resolves it and writes an EXPLICIT schedule row at T (previous target = +1 ->
// row target_units = +1 -> a new Signal +1 here). KEEP_PREVIOUS_TARGET is NOT
// NO DECISION -- the two are never merged. This class invents no target-state
// semantics of its own.

struct ScheduledTarget {
    std::string  root_symbol;    // bare root, e.g. "NQ" -- never a raw contract
    std::int64_t ts_event_ns{0}; // decision-time timestamp (matches a MarketEvent ts)
    int          target_units{0};
};

class ScheduledTargetStrategy final : public Strategy {
public:
    // What to do at a bar with NO schedule row for this root.
    enum class AbsentPolicy : int {
        // DEFAULT and normal Phase 11 behaviour: return NO DECISION
        // (std::nullopt). No Signal, no signal_id, no retry of any earlier
        // intent. Stores no previous-target state.
        NoDecision = 0,
        // Explicit opt-in: treat a missing row as an explicit flat decision --
        // emit Signal(target_units = 0).
        Flat = 1,
        // Strict: throw if a decision bar has no row (for schedules that must be
        // dense over the traded window).
        RequireRow = 2,
        // EXPLICIT OPT-IN, testing / research only: re-emit the last target this
        // strategy emitted for the root (seeded flat) as a BRAND NEW Signal. This
        // DOES create a fresh intent every absent bar and can retry a previously
        // rejected target -- it is never the default and never used by the
        // Phase 11 reference path.
        ReemitPreviousTarget = 3,
    };

    ScheduledTargetStrategy(std::vector<ScheduledTarget> rows,
                            std::string strategy_fingerprint,
                            AbsentPolicy absent_policy = AbsentPolicy::NoDecision);

    std::optional<Signal> decide(const StrategyContext& ctx) const override;

    const std::string& fingerprint() const noexcept { return fingerprint_; }
    std::size_t row_count() const noexcept { return row_count_; }
    // Rows whose (root, ts) actually landed on a decision bar during the last run.
    std::size_t rows_applied() const noexcept { return rows_applied_; }
    AbsentPolicy absent_policy() const noexcept { return absent_policy_; }

private:
    std::unordered_map<std::string, std::unordered_map<std::int64_t, int>> by_root_ts_;
    std::string  fingerprint_;
    AbsentPolicy absent_policy_;
    std::size_t  row_count_{0};

    mutable std::size_t rows_applied_{0};
    // Populated ONLY under AbsentPolicy::ReemitPreviousTarget -- the normal path
    // stores no previous-target state.
    mutable std::unordered_map<std::string, double> last_emitted_;
};

// Parse a Phase 11 targets.csv. Header MUST be exactly:
//   ts_event_ns,root_symbol,target_units,strategy_fingerprint,matched_rule_id
// Throws std::runtime_error on: a missing file, a header mismatch (which also
// rejects any execution-plane column such as fill_price / raw_symbol), a
// non-integer target_units, a root that looks like a raw contract, or MIXED
// strategy fingerprints (a schedule carries exactly one). `fingerprint_out` is
// set to the single fingerprint found.
std::vector<ScheduledTarget> parse_targets_csv(const std::string& path,
                                               std::string& fingerprint_out);

ScheduledTargetStrategy::AbsentPolicy parse_absent_policy(const std::string& s);

}  // namespace quant
