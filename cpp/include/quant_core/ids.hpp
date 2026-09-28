#pragma once

#include <cstdint>
#include <string>

namespace quant {

// Deterministic domain identifiers (Phase 05).
//
// Every id is a plain monotonically increasing integer produced by MonotonicId.
// The same input event stream always produces the same id chain -- no UUIDs, no
// wall-clock, no external sequence service, no database. This is what makes a
// backtest run byte-for-byte reproducible and auditable:
//
//   MarketEvent(seq) -> Signal S123 -> Order O456 -> RiskDecision(O456) -> Fill F789
using SeqNum   = std::uint64_t;  // per-stream event ordering tie-breaker
using SignalId = std::uint64_t;
using OrderId  = std::uint64_t;
using FillId   = std::uint64_t;

// A single monotonically increasing counter. Use one instance per id space
// (signals / orders / fills); each hands out 1, 2, 3, ... deterministically.
class MonotonicId {
public:
    explicit MonotonicId(std::uint64_t first = 1) noexcept : next_(first) {}

    std::uint64_t next() noexcept { return next_++; }
    std::uint64_t peek() const noexcept { return next_; }

private:
    std::uint64_t next_;
};

// Human / log-readable trace tag, e.g. trace_tag('S', 123) == "S123".
inline std::string trace_tag(char prefix, std::uint64_t id) {
    return std::string(1, prefix) + std::to_string(id);
}

}  // namespace quant
