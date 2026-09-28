#pragma once

#include "quant_core/types.hpp"

#include <cstddef>
#include <stdexcept>

namespace quant {

// A read-only window over bar history that ENDS at the decision bar.
//
// A strategy is handed one of these instead of the full bar vector, so it is
// structurally impossible to read a bar dated at or after the next execution
// point. Every accessor is bounds-checked and throws std::out_of_range rather
// than reading past the visible range.
class BarHistoryView {
public:
    BarHistoryView(const MarketBar* data, std::size_t visible_count) noexcept
        : data_(data), size_(visible_count) {}

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    // Oldest-first index; `i` must be < size().
    const MarketBar& at(std::size_t i) const {
        if (i >= size_) {
            throw std::out_of_range("BarHistoryView::at: index at or past the decision bar");
        }
        return data_[i];
    }

    // The decision bar itself (most recent visible bar).
    const MarketBar& latest() const {
        if (size_ == 0) throw std::out_of_range("BarHistoryView::latest: empty history");
        return data_[size_ - 1];
    }

    // `n` bars before latest(); ago(0) == latest(). `n` must be < size().
    const MarketBar& ago(std::size_t n) const {
        if (n >= size_) {
            throw std::out_of_range("BarHistoryView::ago: reaches before the start of history");
        }
        return data_[size_ - 1 - n];
    }

private:
    const MarketBar* data_{nullptr};
    std::size_t size_{0};
};

}  // namespace quant
