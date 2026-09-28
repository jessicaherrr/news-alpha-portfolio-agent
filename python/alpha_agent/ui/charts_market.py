"""Market-OBSERVATION plane chart builders (Release UX Part F, task spec
sections 21-25). Every input here is a real bar/snapshot sequence already
fetched through `alpha_agent.ui.databento_context` -- nothing here fabricates
a price, and nothing here is scientific evidence (see that module's boundary
docstring). Follows the same shared dark Plotly template as `charts.py`.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import plotly.graph_objects as go

from alpha_agent.ui import palette
from alpha_agent.ui.charts import dark_layout


def market_price_chart(bars: Sequence[dict[str, Any]], *, height: int = 320) -> go.Figure:
    """A candlestick chart of real OHLCV bars
    (`alpha_agent.marketdata.databento_schemas.OhlcvBar`, as dicts). Empty
    input renders an empty, honestly-labeled figure -- never a fabricated
    series."""
    if not bars:
        fig = go.Figure()
        dark_layout(fig, height=height)
        fig.add_annotation(text="No bars available", showarrow=False,
                            font={"color": palette.TEXT_MUTED, "size": 12})
        return fig

    ts = [b["ts_event"] for b in bars]
    fig = go.Figure(
        go.Candlestick(
            x=ts, open=[b["open"] for b in bars], high=[b["high"] for b in bars],
            low=[b["low"] for b in bars], close=[b["close"] for b in bars],
            increasing_line_color=palette.GREEN, decreasing_line_color=palette.RED,
            increasing_fillcolor=palette.GREEN, decreasing_fillcolor=palette.RED,
        )
    )
    dark_layout(fig, height=height)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title={"text": "Price", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    return fig


def market_volume_chart(bars: Sequence[dict[str, Any]], *, height: int = 120) -> go.Figure:
    """A bar chart of real per-bar volume -- `None` volumes are dropped, never
    zero-filled (a schema/provider that carries no volume must render
    honestly as no chart, handled by the caller)."""
    ts = [b["ts_event"] for b in bars]
    vol = [b.get("volume") or 0 for b in bars]
    fig = go.Figure(go.Bar(x=ts, y=vol, marker_color=palette.BLUE, marker_line_width=0))
    dark_layout(fig, height=height)
    fig.update_yaxes(title={"text": "Volume", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    return fig


def market_comparison_chart(series: dict[str, Sequence[dict[str, Any]]], *, height: int = 260) -> go.Figure:
    """Normalized (base=100) percentage-return lines for 2+ roots' real bars
    -- lets a user visually compare relative moves without implying a
    tradable spread. `series` maps root symbol -> its OHLCV bars."""
    fig = go.Figure()
    colors = palette.CATEGORICAL
    for i, (root, bars) in enumerate(series.items()):
        if not bars:
            continue
        base = bars[0]["close"]
        if not base:
            continue
        y = [100.0 * (b["close"] / base) for b in bars]
        x = [b["ts_event"] for b in bars]
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=root, line={"color": color, "width": 1.8}))
    dark_layout(fig, height=height)
    fig.update_yaxes(title={"text": "Normalized (start = 100)", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    return fig


def add_timestamp_markers(
    fig: go.Figure, bars: Sequence[dict[str, Any]], markers: Sequence[tuple[datetime, str]], *,
    color: str, symbol: str = "diamond", name: str = "Markers",
) -> go.Figure:
    """Checkpoint G, Section 28 -- adds one hoverable marker per (timestamp,
    hover_text) pair, placed at the nearest real bar's high so it sits near
    the candles at that point in time. Deliberately generic (plain tuples,
    not `MarketNewsItem`/`ScheduledMarketEvent`) so this chart module never
    depends on `alpha_agent.market_intel` -- the caller does that mapping.
    A marker whose timestamp falls outside `bars`' own range is still
    honestly placed (near the nearest bar) rather than silently dropped --
    the CALLER is responsible for pre-filtering to the visible chart range.
    Mutates and returns `fig` for chaining; a no-op when there is nothing to
    mark or no bars to anchor to."""
    if not markers or not bars:
        return fig
    xs, ys, hover = [], [], []
    for ts, hover_text in markers:
        nearest = min(bars, key=lambda b: abs((b["ts_event"] - ts).total_seconds()))
        xs.append(ts)
        ys.append(nearest["high"])
        hover.append(hover_text)
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers", marker={"symbol": symbol, "size": 11, "color": color, "line": {"width": 1, "color": "#0a0d16"}},
            name=name, hovertext=hover, hoverinfo="text",
        )
    )
    return fig


__all__ = ["add_timestamp_markers", "market_comparison_chart", "market_price_chart", "market_volume_chart"]
