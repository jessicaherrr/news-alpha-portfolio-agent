"""Deterministic SYNTHETIC fixtures for the Phase 15A test suite.

Every timestamp here is inside 2018-01-01 .. 2024-12-31 and every value is
generated from a fixed seed. Nothing in this module reads market data, touches
the network, or produces a research result.
"""
from __future__ import annotations

import numpy as np
from alpha_agent.backtest.targets import TargetSchedule, TargetScheduleRow
from alpha_agent.ml.episodes import EpisodeEconomics
from alpha_agent.ml.splits import EventTimeline
from alpha_agent.ml.trade_export import ClosedTradeRecord, FillRecord

DAY_NS = 86_400_000_000_000
#: 2018-01-02T00:00:00Z -- inside the development corpus, never near 2025.
T0 = 1_514_851_200_000_000_000
FAKE_FP = "stratdsl1:" + "a" * 64


def ts(day: int) -> int:
    return T0 + day * DAY_NS


def make_schedule(targets: list[int], *, root: str = "NQ", start_day: int = 0) -> TargetSchedule:
    """One target row per synthetic trading day."""
    rows = tuple(
        TargetScheduleRow(
            ts_event_ns=ts(start_day + i),
            root_symbol=root,
            target_units=t,
            strategy_fingerprint=FAKE_FP,
            matched_rule_id=None,
        )
        for i, t in enumerate(targets)
    )
    return TargetSchedule(
        root_symbol=root,
        strategy_fingerprint=FAKE_FP,
        strategy_id="synthetic-primary",
        strategy_dsl_version="strategy-dsl/1",
        feature_engine_version="0.9.2",
        warmup_bars=0,
        rows=rows,
    )


def make_trade(
    *, open_day: int, close_day: int, net: float, direction: int = 1,
    reason: str = "signal", index: int = 0, root: str = "NQ", quantity: int = 1,
    instrument_id: int = 1, symbol: str | None = None, gross: float | None = None,
    costs: float = 2.0,
) -> ClosedTradeRecord:
    return ClosedTradeRecord(
        trade_index=index,
        instrument_id=instrument_id,
        raw_symbol=symbol or f"{root}Z4",
        root_symbol=root,
        ts_open_ns=ts(open_day),
        ts_close_ns=ts(close_day),
        quantity=quantity,
        direction=direction,
        entry_price=100.0,
        exit_price=100.0 + net,
        gross_pnl_usd=(net + costs) if gross is None else gross,
        costs_usd=costs,
        net_pnl_usd=((net + costs) if gross is None else gross) - costs,
        close_reason=reason,
    )


def make_fill(
    *, day: int, side: str, quantity: int, index: int, commission_per_contract: float = 2.0,
    instrument_id: int = 1, root: str = "NQ", symbol: str | None = None, price: float = 100.0,
) -> FillRecord:
    """One fill. Commission is a flat USD-per-contract charge, as in the engine."""
    return FillRecord(
        fill_index=index,
        fill_id=index,
        order_id=index,
        ts_fill_ns=ts(day),
        instrument_id=instrument_id,
        raw_symbol=symbol or f"{root}Z4",
        side=side,
        quantity=quantity,
        fill_price=price,
        commission_usd=commission_per_contract * quantity,
        slippage_ticks=0.0,
    )


def make_economics(index: int, *, net: float, horizon_day: int, n: int = 1) -> EpisodeEconomics:
    return EpisodeEconomics(
        episode_index=index,
        n_attributed_trades=n,
        gross_pnl_usd=net + 2.0 * n,
        costs_usd=2.0 * n,
        net_pnl_usd=net,
        information_horizon_ts_ns=ts(horizon_day),
    )


def synthetic_dataset(
    n_events: int = 600, *, n_features: int = 4, seed: int = 7, label_horizon_days: int = 20,
    spacing_days: int = 4,
) -> tuple[np.ndarray, np.ndarray, EventTimeline, tuple[str, ...], tuple[str, ...]]:
    """A separable-ish binary problem on a chronological event stream.

    Events are spaced ``spacing_days`` apart so the stream spans the corpus,
    and each label resolves ``label_horizon_days`` later, which makes overlapping
    label windows -- exactly the condition purge and embargo exist for.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_events, n_features))
    logit = 0.9 * X[:, 0] - 0.6 * X[:, 1]
    y = (logit + rng.normal(scale=0.5, size=n_events) > 0).astype(int)

    decision = [ts(30 + spacing_days * i) for i in range(n_events)]
    label_start = list(decision)
    label_end = [d + label_horizon_days * DAY_NS for d in decision]
    timeline = EventTimeline(
        decision_ts_ns=tuple(decision),
        label_start_ts_ns=tuple(label_start),
        label_end_ts_ns=tuple(label_end),
    )
    event_ids = tuple(f"EV{i:05d}" for i in range(n_events))
    roots = tuple(["ES", "NQ", "CL", "GC", "ZN"][i % 5] for i in range(n_events))
    return X, y, timeline, event_ids, roots
