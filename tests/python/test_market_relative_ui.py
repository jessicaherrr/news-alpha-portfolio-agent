"""Pure row-builder tests for `alpha_agent.ui.market_relative` (Checkpoint
D). Streamlit-level integration is covered in `test_market_product_detail.py`.
"""
from __future__ import annotations

from datetime import UTC, datetime

from alpha_agent.marketdata.relative_markets import RelationType, RelativeMarketSnapshot
from alpha_agent.ui import market_relative as mr


def _snapshot(*, insufficient: bool = False, **overrides) -> RelativeMarketSnapshot:
    base = {
        "root_symbol": "NQ", "peer_root_symbol": "ES", "relation_type": RelationType.EQUITY_INDEX_PEER,
        "mapping_reason": "test", "common_observations": 20, "insufficient_data": insufficient,
        "as_of": datetime.now(UTC),
    }
    if not insufficient:
        base.update(
            window_return_pct=1.5, peer_window_return_pct=0.5, realized_volatility_pct=12.0,
            peer_realized_volatility_pct=10.0, correlation=0.87, relative_performance_pct=1.0, price_ratio=2.5,
        )
    base.update(overrides)
    return RelativeMarketSnapshot(**base)


def test_snapshot_row_shows_insufficient_data_never_a_fake_number():
    row = mr.snapshot_row(_snapshot(insufficient=True, common_observations=3))
    assert row["Correlation"] == "INSUFFICIENT DATA"
    assert row["Window Return"] == "INSUFFICIENT DATA"
    assert row["Common Obs."] == 3


def test_snapshot_row_shows_real_metrics():
    row = mr.snapshot_row(_snapshot())
    assert row["Product"] == "ES"
    assert row["Correlation"] == "0.87"
    assert row["Relative Performance"] == "+1.00%"


def test_selected_root_row_uses_a_usable_snapshot_and_marks_self():
    snaps = (_snapshot(insufficient=True, common_observations=2), _snapshot())
    row = mr.selected_root_row("NQ", snaps)
    assert row["Product"] == "NQ"
    assert "self" in row["Correlation"]
    assert row["Window Return"] == "+1.50%"  # taken from the SELECTED side of the usable snapshot


def test_selected_root_row_never_fabricates_when_everything_is_insufficient():
    snaps = (_snapshot(insufficient=True, common_observations=1),)
    row = mr.selected_root_row("NQ", snaps)
    assert row["Window Return"] == "N/A"
    assert row["Common Obs."] == 0
