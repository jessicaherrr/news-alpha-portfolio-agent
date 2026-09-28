#include "quant_core/scheduled_target_strategy.hpp"

#include "quant_core/detail/csv.hpp"

#include <array>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>

namespace quant {

namespace {

// A bare root is [A-Z0-9]{1,12} and NOT a month-coded contract (e.g. NQU6 /
// CLZ26) or a continuous symbol (NQ.v.0). This mirrors the Python bridge guard
// so neither side can name a raw contract in the schedule (section 14).
bool looks_like_raw_contract(const std::string& s) {
    if (s.find('.') != std::string::npos) return true;
    // <1-3 uppercase letters><month code><1-2 digits>, parsed from the end so a
    // month-code letter is never greedily consumed as part of the root (NQU6 ->
    // root "NQ", month "U", year "6").
    constexpr std::string_view kMonthCodes = "FGHJKMNQUVXZ";
    std::size_t end = s.size();
    std::size_t digits = 0;
    while (end > 0 && std::isdigit(static_cast<unsigned char>(s[end - 1]))) {
        --end;
        ++digits;
    }
    if (digits < 1 || digits > 2 || end == 0) return false;
    if (kMonthCodes.find(s[end - 1]) == std::string_view::npos) return false;
    const std::size_t letters = end - 1;
    if (letters < 1 || letters > 3) return false;
    for (std::size_t i = 0; i < letters; ++i) {
        if (!std::isupper(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

bool is_bare_root(const std::string& s) {
    if (s.empty() || s.size() > 12) return false;
    for (const char c : s) {
        const bool ok = (std::isupper(static_cast<unsigned char>(c)) != 0) ||
                        (std::isdigit(static_cast<unsigned char>(c)) != 0);
        if (!ok) return false;
    }
    return !looks_like_raw_contract(s);
}

}  // namespace

ScheduledTargetStrategy::ScheduledTargetStrategy(std::vector<ScheduledTarget> rows,
                                                 std::string strategy_fingerprint,
                                                 AbsentPolicy absent_policy)
    : fingerprint_(std::move(strategy_fingerprint)), absent_policy_(absent_policy) {
    for (const auto& row : rows) {
        auto& per_root = by_root_ts_[row.root_symbol];
        const auto [it, inserted] = per_root.emplace(row.ts_event_ns, row.target_units);
        if (!inserted && it->second != row.target_units) {
            throw std::runtime_error(
                "scheduled target schedule has conflicting target_units for (" +
                row.root_symbol + ", ts=" + std::to_string(row.ts_event_ns) + ")");
        }
        if (inserted) ++row_count_;
    }
}

std::optional<Signal> ScheduledTargetStrategy::decide(const StrategyContext& ctx) const {
    const std::int64_t ts = ctx.decision_ts_ns();
    const std::string  root(ctx.root_symbol());

    const auto root_it = by_root_ts_.find(root);
    if (root_it != by_root_ts_.end()) {
        const auto ts_it = root_it->second.find(ts);
        if (ts_it != root_it->second.end()) {
            Signal sig;
            sig.ts_decision_ns = ts;
            sig.root_symbol    = root;
            sig.target_units   = static_cast<double>(ts_it->second);
            sig.rationale_code = "sched_target";
            ++rows_applied_;
            if (absent_policy_ == AbsentPolicy::ReemitPreviousTarget) {
                last_emitted_[root] = sig.target_units;
            }
            return sig;
        }
    }

    // No row at T -> no new strategy decision, by default. Absence is NEITHER
    // an implicit flat NOR an implicit re-emission of a previous target.
    switch (absent_policy_) {
        case AbsentPolicy::NoDecision:
            return std::nullopt;
        case AbsentPolicy::Flat: {
            Signal sig;
            sig.ts_decision_ns = ts;
            sig.root_symbol    = root;
            sig.target_units   = 0.0;
            sig.rationale_code = "sched_absent_flat";
            return sig;
        }
        case AbsentPolicy::ReemitPreviousTarget: {
            const auto it = last_emitted_.find(root);
            Signal sig;
            sig.ts_decision_ns = ts;
            sig.root_symbol    = root;
            sig.target_units   = (it != last_emitted_.end()) ? it->second : 0.0;
            sig.rationale_code = "sched_reemit_previous";  // a BRAND NEW Signal
            return sig;
        }
        case AbsentPolicy::RequireRow:
            throw std::runtime_error("ScheduledTargetStrategy: no schedule row for (" + root +
                                     ", ts=" + std::to_string(ts) +
                                     ") under RequireRow policy");
    }
    return std::nullopt;  // unreachable
}

ScheduledTargetStrategy::AbsentPolicy parse_absent_policy(const std::string& s) {
    if (s.empty() || s == "no_decision") return ScheduledTargetStrategy::AbsentPolicy::NoDecision;
    if (s == "flat") return ScheduledTargetStrategy::AbsentPolicy::Flat;
    if (s == "require_row") return ScheduledTargetStrategy::AbsentPolicy::RequireRow;
    if (s == "reemit_previous") return ScheduledTargetStrategy::AbsentPolicy::ReemitPreviousTarget;
    throw std::runtime_error("unknown schedule policy '" + s +
                             "' (expected no_decision | flat | require_row | reemit_previous)");
}

std::vector<ScheduledTarget> parse_targets_csv(const std::string& path,
                                               std::string& fingerprint_out) {
    static constexpr std::array<const char*, 5> kHeader{
        "ts_event_ns", "root_symbol", "target_units", "strategy_fingerprint", "matched_rule_id",
    };

    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open targets csv '" + path + "'");

    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty targets csv");
    const auto header = detail::split_csv_line(line);
    if (header.size() != kHeader.size()) {
        throw std::runtime_error("targets csv: expected 5 columns, got " +
                                 std::to_string(header.size()) +
                                 " -- execution-plane columns are not permitted");
    }
    for (std::size_t i = 0; i < kHeader.size(); ++i) {
        if (header[i] != kHeader[i]) {
            throw std::runtime_error("targets csv: column " + std::to_string(i) +
                                     " should be '" + kHeader[i] + "', got '" + header[i] +
                                     "' -- the schedule carries TARGET INTENT only");
        }
    }

    std::vector<ScheduledTarget> rows;
    fingerprint_out.clear();
    std::int64_t prev_ts = -1;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = detail::split_csv_line(line);
        if (f.size() != kHeader.size()) {
            throw std::runtime_error("targets csv: row has " + std::to_string(f.size()) +
                                     " fields, expected 5");
        }
        const std::int64_t ts = static_cast<std::int64_t>(std::stoll(f[0]));
        const std::string& root = f[1];
        // target_units must be a plain integer -- reject "1.0" / "1.5" / "".
        if (f[2].empty() || f[2].find_first_not_of("-0123456789") != std::string::npos) {
            throw std::runtime_error("targets csv: target_units '" + f[2] +
                                     "' is not a plain integer");
        }
        const int units = std::stoi(f[2]);
        const std::string& fp = f[3];

        if (!is_bare_root(root)) {
            throw std::runtime_error("targets csv: root_symbol '" + root +
                                     "' is not a bare root -- the schedule may not name a raw "
                                     "contract (the engine resolves the execution contract via "
                                     "MarketState -> ActiveContractResolver)");
        }
        if (fp.rfind("stratdsl1:", 0) != 0) {
            throw std::runtime_error("targets csv: strategy_fingerprint '" + fp +
                                     "' is not a stratdsl1 fingerprint");
        }
        if (fingerprint_out.empty()) {
            fingerprint_out = fp;
        } else if (fp != fingerprint_out) {
            throw std::runtime_error(
                "targets csv: mixed strategy fingerprints (" + fp + " != " + fingerprint_out +
                ") -- a target schedule carries exactly one StrategySpec fingerprint");
        }
        if (ts <= prev_ts) {
            throw std::runtime_error("targets csv: rows must be strictly time-ordered");
        }
        prev_ts = ts;
        rows.push_back(ScheduledTarget{.root_symbol = root, .ts_event_ns = ts, .target_units = units});
    }
    return rows;
}

}  // namespace quant
