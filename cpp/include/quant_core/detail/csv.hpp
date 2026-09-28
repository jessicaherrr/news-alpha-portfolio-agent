#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace quant::detail {

// Split a single CSV line on ','. No quoting support -- the boundary files this
// core reads are machine-generated with plain numeric / symbol fields only.
inline std::vector<std::string> split_csv_line(std::string_view line) {
    std::vector<std::string> fields;
    std::string current;
    for (const char c : line) {
        if (c == ',') {
            fields.push_back(current);
            current.clear();
        } else if (c != '\r') {
            current.push_back(c);
        }
    }
    fields.push_back(current);
    return fields;
}

}  // namespace quant::detail
