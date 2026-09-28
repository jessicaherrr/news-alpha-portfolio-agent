// Phase 15A: the ADDITIVE, READ-ONLY closed-trade CSV export.
//
// Proves the export is a faithful, deterministic serialisation of the
// already-recorded Fill-derived audit trail and that it computes nothing.

#include "quant_core/trade_export.hpp"

#include "test_support.hpp"

#include <sstream>
#include <string>
#include <vector>

namespace {

quant::ClosedTrade make_trade(std::int64_t open_ns, std::int64_t close_ns, int direction,
                              double net, const std::string& reason) {
    quant::ClosedTrade t;
    t.instrument_id = 42;
    t.raw_symbol = "NQZ4";
    t.root_symbol = "NQ";
    t.ts_open_ns = open_ns;
    t.ts_close_ns = close_ns;
    t.quantity = 1;
    t.direction = direction;
    t.entry_price = 20000.25;
    t.exit_price = 20010.75;
    t.gross_pnl_usd = net + 4.0;
    t.costs_usd = 4.0;
    t.net_pnl_usd = net;
    t.close_reason = reason;
    return t;
}

std::vector<std::string> lines_of(const std::string& text) {
    std::vector<std::string> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) out.push_back(line);
    return out;
}

void test_header_and_row_order() {
    const std::vector<quant::ClosedTrade> trades{
        make_trade(100, 200, +1, 105.5, "signal"),
        make_trade(200, 300, -1, -37.25, "roll"),
        make_trade(300, 400, +1, 0.0, "eot"),
    };
    std::ostringstream out;
    quant::write_closed_trades_csv(out, trades);
    const auto lines = lines_of(out.str());

    CHECK(lines.size() == 4);
    CHECK(lines[0] == std::string(quant::kClosedTradeCsvHeader));
    // trade_index is the engine's chronological audit order.
    CHECK(lines[1].rfind("0,42,NQZ4,NQ,100,200,1,1,", 0) == 0);
    CHECK(lines[2].rfind("1,42,NQZ4,NQ,200,300,1,-1,", 0) == 0);
    CHECK(lines[3].rfind("2,42,NQZ4,NQ,300,400,1,1,", 0) == 0);
    // close_reason is carried verbatim as the last field.
    CHECK(lines[1].substr(lines[1].rfind(',') + 1) == "signal");
    CHECK(lines[2].substr(lines[2].rfind(',') + 1) == "roll");
    CHECK(lines[3].substr(lines[3].rfind(',') + 1) == "eot");
}

void test_empty_trades_writes_header_only() {
    std::ostringstream out;
    quant::write_closed_trades_csv(out, {});
    CHECK(out.str() == std::string(quant::kClosedTradeCsvHeader) + "\n");
}

void test_values_are_verbatim_not_recomputed() {
    // gross - costs deliberately != net here. The export must NOT "fix" it:
    // it is a serialiser, never an accountant.
    quant::ClosedTrade t = make_trade(1'000'000'000, 2'000'000'000, -1, 11.0, "signal");
    t.gross_pnl_usd = 999.0;
    t.costs_usd = 1.0;
    t.net_pnl_usd = 11.0;
    std::ostringstream out;
    quant::write_closed_trades_csv(out, {t});
    const auto lines = lines_of(out.str());
    CHECK(lines.size() == 2);
    CHECK(lines[1].find(",999,1,11,signal") != std::string::npos);
}

void test_deterministic_repeated_serialisation() {
    const std::vector<quant::ClosedTrade> trades{
        make_trade(100, 200, +1, 1.0 / 3.0, "signal"),
        make_trade(200, 300, -1, -2.0 / 7.0, "roll"),
    };
    std::ostringstream a;
    std::ostringstream b;
    quant::write_closed_trades_csv(a, trades);
    quant::write_closed_trades_csv(b, trades);
    CHECK(a.str() == b.str());
    // %.12g keeps enough precision to round-trip the fraction.
    CHECK(a.str().find("0.333333333333") != std::string::npos);
}

void test_csv_hostile_symbol_is_a_loud_error() {
    quant::ClosedTrade t = make_trade(100, 200, +1, 1.0, "signal");
    t.raw_symbol = "NQ,Z4";
    std::ostringstream out;
    CHECK_THROWS(quant::write_closed_trades_csv(out, {t}));
}

quant::Fill make_fill(std::int64_t ts, quant::Side side, double commission) {
    quant::Fill f;
    f.fill_id = 7;
    f.order_id = 3;
    f.ts_fill_ns = ts;
    f.instrument_id = 42;
    f.raw_symbol = "NQZ4";
    f.side = side;
    f.quantity = 1;
    f.fill_price = 20000.25;
    f.tick_size = 0.25;
    f.multiplier = 20.0;
    f.commission_usd = commission;
    f.slippage_ticks = 0.0;
    return f;
}

void test_fills_export_carries_every_commission() {
    // The reason this export exists: ClosedTrade::costs_usd is the CLOSING
    // fill's commission only, so the entry commission is only visible here.
    const std::vector<quant::Fill> fills{
        make_fill(100, quant::Side::Buy, 2.0),
        make_fill(200, quant::Side::Sell, 2.0),
    };
    std::ostringstream out;
    quant::write_fills_csv(out, fills);
    const auto lines = lines_of(out.str());

    CHECK(lines.size() == 3);
    CHECK(lines[0] == std::string(quant::kFillCsvHeader));
    CHECK(lines[1].rfind("0,7,3,100,42,NQZ4,buy,1,", 0) == 0);
    CHECK(lines[2].rfind("1,7,3,200,42,NQZ4,sell,1,", 0) == 0);
    CHECK(lines[1].find(",2,0") != std::string::npos);
}

void test_empty_fills_writes_header_only() {
    std::ostringstream out;
    quant::write_fills_csv(out, {});
    CHECK(out.str() == std::string(quant::kFillCsvHeader) + "\n");
}

}  // namespace

int main() {
    test_header_and_row_order();
    test_empty_trades_writes_header_only();
    test_values_are_verbatim_not_recomputed();
    test_deterministic_repeated_serialisation();
    test_csv_hostile_symbol_is_a_loud_error();
    test_fills_export_carries_every_commission();
    test_empty_fills_writes_header_only();
    return quant::test::summary("test_trade_export");
}
