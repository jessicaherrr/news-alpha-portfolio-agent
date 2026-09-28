"""Phase 4 -- the quant-rigor pipeline walkthrough
(`alpha_agent.learn.pipeline_explainer`): Signal -> Next-Bar Execution -> Fill
-> Contract Economics -> Trade PnL -> Equity -> Validation."""
from __future__ import annotations

from alpha_agent.learn.pipeline_explainer import build_pipeline_walkthrough

_EXPECTED_ORDER = (
    "signal", "next_bar_execution", "fill", "contract_economics", "trade_pnl", "equity", "validation",
)


def test_pipeline_has_the_seven_fixed_stages_in_order():
    w = build_pipeline_walkthrough(trade_ledger=None)
    assert tuple(s.stage_id for s in w.stages) == _EXPECTED_ORDER


def test_no_ledger_means_no_fabricated_example_anywhere():
    w = build_pipeline_walkthrough(trade_ledger=None)
    assert w.has_real_example is False
    assert w.example_source is None
    assert all(s.example is None for s in w.stages)


def test_next_bar_execution_links_the_look_ahead_bias_concept():
    w = build_pipeline_walkthrough(trade_ledger=None)
    stage = next(s for s in w.stages if s.stage_id == "next_bar_execution")
    assert stage.concept_id == "look_ahead_bias"


def test_contract_economics_stage_links_the_contract_economics_concept():
    w = build_pipeline_walkthrough(trade_ledger=None)
    stage = next(s for s in w.stages if s.stage_id == "contract_economics")
    assert stage.concept_id == "contract_economics"


def test_a_real_bound_ledger_attaches_the_first_trades_own_real_values():
    ledger = {
        "source_artifact": "outputs/fake/trades.csv",
        "trades": [
            {"trade_index": "2", "fill_price": "101.5", "net_pnl_usd": "-40.0"},
            {"trade_index": "1", "fill_price": "100.25", "net_pnl_usd": "125.50"},
        ],
    }
    w = build_pipeline_walkthrough(trade_ledger=ledger)
    assert w.has_real_example is True
    assert w.example_source == "outputs/fake/trades.csv"
    fill_stage = next(s for s in w.stages if s.stage_id == "fill")
    pnl_stage = next(s for s in w.stages if s.stage_id == "trade_pnl")
    # first trade by trade_index order (1, not 2)
    assert "100.25" in fill_stage.example
    assert "125.50" in pnl_stage.example


def test_empty_trades_list_behaves_like_no_ledger():
    w = build_pipeline_walkthrough(trade_ledger={"trades": [], "source_artifact": "x"})
    assert w.has_real_example is False
    assert all(s.example is None for s in w.stages)


def test_every_stage_carries_at_least_one_real_pointer():
    w = build_pipeline_walkthrough(trade_ledger=None)
    for stage in w.stages:
        assert stage.pointers, f"{stage.stage_id} has no code pointer"
