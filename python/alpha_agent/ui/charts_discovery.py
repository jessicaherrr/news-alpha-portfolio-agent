"""Alpha Discovery campaign, Part J -- investor visualizations for a NEW
discovery run's real, experiment-bound daily-equity artifact (task spec
sections 51-53).

Companion to the existing `alpha_agent.ui.charts.equity_curve_chart` /
`.drawdown_chart` (per-TRADE granularity, driven by a verifiably-bound trade
ledger -- `services.find_experiment_bound_trade_ledger`, still `None` for
every experiment in the registry today) and `.regime_bar` (already real:
`regime_evidence.buckets` is populated for every canonical trial with
evaluated regime evidence -- no new chart needed there, see module docstring
note below).

These two functions instead plot the real, per-DAY official equity trace
(`report.oos_daily`, persisted by `alpha_agent.artifacts.store` -- see
`services.find_experiment_bound_artifact_bundle` /
`.daily_equity_rows_from_bundle`). Only produced for a NEW discovery run
executed with `artifact_dir` set (Part I); every pre-existing registry
experiment has no bound daily-equity artifact and must render the existing
`components.empty_state` "PATH DATA NOT AVAILABLE" treatment instead of
either of these functions -- never a fabricated curve from a summary
Sharpe/Net PnL.

Price + Position/Signal (Release UX Part G, task spec section 51 item 3;
originally out of scope here -- see the live-research campaign's Checkpoint
11, which added the raw executed `price_bars` artifact to the Part I bundle
specifically so this chart could be built without ever reconstructing a bar).
`price_signal_chart` renders a REAL, bounded window of `price_bars` around
one trade, with markers placed ONLY at that trade's own real fill
timestamps/prices (`services.price_signal_window_from_bundle`) -- entries and
exits are never inferred from `net_pnl_usd` or any other summary statistic,
and no signal timestamp is fabricated: the C++ engine's own `ts_open_ns` /
`ts_close_ns` on a closed trade already ARE its opening/closing fill
timestamps, so "entry" and "fill" coincide here by construction (mission
section 27 -- if a distinct pre-fill signal timestamp is ever added to a
future export, this function must show both, never invent one from the other).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alpha_agent.ui import palette
from alpha_agent.ui.charts import _hex_fill, dark_layout


def _floats(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in rows]


def daily_equity_curve_chart(daily_rows: Sequence[dict[str, Any]], *, height: int = 240) -> go.Figure:
    """The real, official daily equity trace -- one point per trading day,
    from `services.daily_equity_rows_from_bundle`. Never gross AND net (the
    persisted trace does not carry gross separately from costs per day; see
    module docstring) -- this is the NET equity path."""
    equity = _floats(daily_rows, "equity_usd")
    labels = [r.get("trading_day", str(i)) for i, r in enumerate(daily_rows)]
    color = palette.GREEN if (equity and equity[-1] >= equity[0]) else palette.RED
    fig = go.Figure(go.Scatter(
        x=labels, y=equity, mode="lines", line={"color": color, "width": 1.8},
        fill="tozeroy", fillcolor=_hex_fill(color, 0.10),
    ))
    dark_layout(fig, height=height)
    fig.update_yaxes(title={"text": "Net Equity (USD)", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    fig.update_xaxes(title={"text": "Trading Day", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    return fig


def daily_drawdown_chart(daily_rows: Sequence[dict[str, Any]], *, height: int = 200) -> go.Figure:
    """Peak-to-trough of the same real per-day equity trace, in USD."""
    equity = _floats(daily_rows, "equity_usd")
    labels = [r.get("trading_day", str(i)) for i, r in enumerate(daily_rows)]
    drawdown: list[float] = []
    peak = float("-inf")
    for e in equity:
        peak = max(peak, e)
        drawdown.append(e - peak)
    fig = go.Figure(go.Scatter(
        x=labels, y=drawdown, mode="lines", line={"color": palette.RED, "width": 1.6},
        fill="tozeroy", fillcolor=_hex_fill(palette.RED, 0.14),
    ))
    dark_layout(fig, height=height)
    fig.update_yaxes(title={"text": "Drawdown (USD)", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    fig.update_xaxes(title={"text": "Trading Day", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    return fig


def price_signal_chart(window: dict[str, Any], *, height: int = 420) -> go.Figure:
    """Price + real fill markers + actual position, for ONE trade's bounded
    real bar window (`services.price_signal_window_from_bundle`'s return).
    `window["bars"]` are the exact executed price_bars rows the C++ engine
    ran against; `window["fills"]` are the real fills inside that same
    window; `window["trade"]` is the one closed trade this window is
    centered on. Empty/missing input renders an honest empty figure, never a
    fabricated candle."""
    bars = window.get("bars") or []
    if not bars:
        fig = make_subplots(rows=2, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
        dark_layout(fig, height=height)
        fig.add_annotation(text="No real price_bars available for this trade", showarrow=False,
                            font={"color": palette.TEXT_MUTED, "size": 12})
        return fig

    x = [int(b["ts_event_ns"]) for b in bars]
    fig = make_subplots(rows=2, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_trace(
        go.Candlestick(
            x=x, open=_floats(bars, "open"), high=_floats(bars, "high"),
            low=_floats(bars, "low"), close=_floats(bars, "close"),
            increasing_line_color=palette.GREEN, decreasing_line_color=palette.RED,
            increasing_fillcolor=palette.GREEN, decreasing_fillcolor=palette.RED, name="Price",
        ),
        row=1, col=1,
    )

    trade = window.get("trade") or {}
    direction = int(trade.get("direction", 1)) if trade else 1
    quantity = float(trade.get("quantity", 0)) if trade else 0.0
    for fill in window.get("fills") or []:
        side = (fill.get("side") or "").lower()
        is_entry = int(fill.get("ts_fill_ns", -1)) == int(trade.get("ts_open_ns", -2))
        is_exit = int(fill.get("ts_fill_ns", -1)) == int(trade.get("ts_close_ns", -3))
        marker_color = palette.GREEN if side == "buy" else palette.RED
        marker_symbol = "triangle-up" if side == "buy" else "triangle-down"
        label = "Entry" if is_entry else ("Exit" if is_exit else side.title())
        fig.add_trace(
            go.Scatter(
                x=[int(fill["ts_fill_ns"])], y=[float(fill["fill_price"])], mode="markers+text",
                marker={"color": marker_color, "symbol": marker_symbol, "size": 13,
                        "line": {"color": palette.TEXT_PRIMARY, "width": 1}},
                text=[label], textposition="top center", textfont={"size": 10, "color": palette.TEXT_SECONDARY},
                name=label, showlegend=False,
            ),
            row=1, col=1,
        )

    # Position step-line, derived ONLY from this trade's own real
    # ts_open_ns/ts_close_ns/direction/quantity -- never a summary-stat guess.
    ts_open = int(trade.get("ts_open_ns", x[0]))
    ts_close = int(trade.get("ts_close_ns", x[-1]))
    signed_qty = direction * quantity
    pos_x, pos_y = [], []
    for ts in x:
        pos_x.append(ts)
        pos_y.append(signed_qty if ts_open <= ts < ts_close else 0.0)
    color = palette.GREEN if signed_qty >= 0 else palette.RED
    fig.add_trace(
        go.Scatter(x=pos_x, y=pos_y, mode="lines", line={"color": color, "width": 1.8, "shape": "hv"},
                   fill="tozeroy", fillcolor=_hex_fill(color, 0.14), name="Position", showlegend=False),
        row=2, col=1,
    )

    dark_layout(fig, height=height)
    fig.update_layout(xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False)
    fig.update_yaxes(title={"text": "Price", "font": {"size": 11, "color": palette.TEXT_MUTED}}, row=1, col=1)
    fig.update_yaxes(title={"text": "Position", "font": {"size": 11, "color": palette.TEXT_MUTED}}, row=2, col=1)
    return fig


def max_drawdown_fraction(daily_rows: Sequence[dict[str, Any]]) -> float | None:
    """Max drawdown as a fraction of the running peak equity -- the same
    real trace, summarized to one number for a metric card."""
    equity = _floats(daily_rows, "equity_usd")
    if not equity:
        return None
    peak = float("-inf")
    worst = 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            worst = min(worst, (e - peak) / peak)
    return abs(worst)
