#pragma once

// Minimal dependency-free test harness.
//
// Rationale: plain <cassert> asserts are compiled out when NDEBUG is defined
// (CMake does this for Release/RelWithDebInfo/MinSizeRel builds), which would let
// the test binary pass without checking anything. These macros are always active
// regardless of NDEBUG and make the process exit non-zero on the first failure so
// CTest reports it.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

namespace quant::test {

inline int& check_count() {
    static int n = 0;
    return n;
}

inline void fail(const char* expr, const char* file, int line, const std::string& detail = {}) {
    std::fprintf(stderr, "CHECK FAILED: %s\n  at %s:%d\n", expr, file, line);
    if (!detail.empty()) std::fprintf(stderr, "  %s\n", detail.c_str());
    std::exit(1);
}

inline void pass() { ++check_count(); }

inline int summary(const char* name) {
    std::printf("%s: %d checks passed\n", name, check_count());
    return 0;
}

}  // namespace quant::test

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (cond) {                                                       \
            ::quant::test::pass();                                        \
        } else {                                                          \
            ::quant::test::fail(#cond, __FILE__, __LINE__);               \
        }                                                                 \
    } while (0)

#define CHECK_CLOSE(a, b, tol)                                            \
    do {                                                                  \
        const double _a = (a);                                            \
        const double _b = (b);                                            \
        const double _t = (tol);                                          \
        if (std::fabs(_a - _b) <= _t) {                                   \
            ::quant::test::pass();                                        \
        } else {                                                          \
            ::quant::test::fail(#a " ~= " #b, __FILE__, __LINE__,         \
                                "|" + std::to_string(_a) + " - " +        \
                                    std::to_string(_b) + "| > " +         \
                                    std::to_string(_t));                  \
        }                                                                 \
    } while (0)

// Passes iff evaluating `expr` throws an exception derived from std::exception.
#define CHECK_THROWS(expr)                                                \
    do {                                                                  \
        bool _threw = false;                                              \
        try {                                                             \
            (void)(expr);                                                 \
        } catch (const std::exception&) {                                 \
            _threw = true;                                                \
        }                                                                 \
        if (_threw) {                                                     \
            ::quant::test::pass();                                        \
        } else {                                                          \
            ::quant::test::fail("expected throw: " #expr, __FILE__, __LINE__); \
        }                                                                 \
    } while (0)

// Passes iff evaluating `expr` throws an exception of (or derived from) `ex`.
#define CHECK_THROWS_AS(expr, ex)                                         \
    do {                                                                  \
        bool _threw_as = false;                                           \
        try {                                                             \
            (void)(expr);                                                 \
        } catch (const ex&) {                                             \
            _threw_as = true;                                             \
        } catch (...) {                                                   \
        }                                                                 \
        if (_threw_as) {                                                  \
            ::quant::test::pass();                                        \
        } else {                                                          \
            ::quant::test::fail("expected throw " #ex ": " #expr, __FILE__, __LINE__); \
        }                                                                 \
    } while (0)
