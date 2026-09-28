"""Phase 20.1 -- one shared dark Plotly template + small chart builders.

Every figure in the app is built here or through `dark_layout(...)` so the
terminal look stays consistent: transparent surfaces (the card behind it shows
through), subtle gridlines, thin axes, compact margins, one shared font. No
builder here fabricates a series that is not already present in a committed
artifact -- `backtests.py` / `validation.py` decide whether to call a builder
at all based on `services.py`'s explicit-only evidence.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import plotly.graph_objects as go

from alpha_agent.ui import palette

FONT = {"family": "'Inter', -apple-system, 'Segoe UI', sans-serif", "size": 12, "color": palette.TEXT_SECONDARY}


def dark_layout(fig: go.Figure, *, title: str | None = None, height: int = 260, **kwargs: Any) -> go.Figure:
    """Apply the shared dark template in place and return the figure.

    Margins are generous on purpose: with `theme=None` on `st.plotly_chart`
    (required -- see `components.plotly_chart`), Streamlit no longer
    auto-fits axis titles/tick labels, so this template must reserve enough
    space itself or labels clip/overlap."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=FONT,
        title={"text": title, "font": {"color": palette.TEXT_PRIMARY, "size": 13}} if title else None,
        margin={"l": 56, "r": 20, "t": 40 if title else 16, "b": 44},
        height=height,
        hoverlabel={"bgcolor": palette.CARD_BG_RAISED, "font": {"color": palette.TEXT_PRIMARY, "size": 11},
                         "bordercolor": palette.BORDER},
        legend={"font": {"color": palette.TEXT_SECONDARY, "size": 11}, "bgcolor": "rgba(0,0,0,0)"},
        **kwargs,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color=palette.TEXT_SECONDARY,
                      linecolor=palette.BORDER, tickfont={"size": 11, "color": palette.TEXT_SECONDARY})
    fig.update_yaxes(showgrid=True, gridcolor=palette.BORDER_SOFT, zeroline=False,
                      color=palette.TEXT_SECONDARY, tickfont={"size": 11, "color": palette.TEXT_SECONDARY},
                      title={"font": {"size": 11, "color": palette.TEXT_MUTED}})
    return fig


def _hex_fill(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def sparkline(y: Sequence[float], *, color: str = palette.BLUE, height: int = 40) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=list(y), mode="lines", line={"color": color, "width": 1.6},
        fill="tozeroy", fillcolor=_hex_fill(color),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=height, showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def bar_chart(labels: Sequence[str], values: Sequence[float], *, color: str = palette.CATEGORICAL[0],
              title: str | None = None, height: int = 220, yaxis_title: str | None = None) -> go.Figure:
    fig = go.Figure(go.Bar(x=list(labels), y=list(values), marker_color=color, marker_line_width=0))
    dark_layout(fig, title=title, height=height)
    if yaxis_title:
        fig.update_yaxes(title={"text": yaxis_title, "font": {"size": 11, "color": palette.TEXT_MUTED}})
    return fig


def ci_range_chart(*, point: float, ci_low: float, ci_high: float, label: str = "Annualized Sharpe",
                    height: int = 160) -> go.Figure:
    """A single bootstrap confidence interval, drawn exactly as the three
    committed numbers (`point`, `ci_low`, `ci_high`) -- no fabricated sample
    distribution, since the committed artifact only ever carries the summary."""
    includes_zero = ci_low <= 0 <= ci_high
    color = palette.AMBER if includes_zero else palette.GREEN
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[ci_low, ci_high], y=[label, label], mode="lines",
        line={"color": color, "width": 6}, showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[point], y=[label], mode="markers",
        marker={"color": palette.TEXT_PRIMARY, "size": 10, "line": {"color": color, "width": 2}},
        showlegend=False,
    ))
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color=palette.TEXT_MUTED)
    dark_layout(fig, height=height)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return fig


def regime_bar(buckets: Sequence[dict[str, Any]], *, height: int = 220) -> go.Figure:
    labels = [b["label"] for b in buckets]
    shares = [b["pnl_share"] for b in buckets]
    colors = [palette.GREEN if s >= 0 else palette.RED for s in shares]
    fig = go.Figure(go.Bar(x=labels, y=shares, marker_color=colors, marker_line_width=0))
    dark_layout(fig, height=height)
    fig.update_yaxes(title={"text": "Share of net PnL", "font": {"size": 11, "color": palette.TEXT_MUTED}}, tickformat=".0%")
    return fig


def cost_stress_line(scenarios: Sequence[dict[str, Any]], *, height: int = 220) -> go.Figure:
    labels = [s["label"] for s in scenarios]
    values = [s["net_pnl_usd"] for s in scenarios]
    fig = go.Figure(go.Scatter(x=labels, y=values, mode="lines+markers",
                                line={"color": palette.CATEGORICAL[0], "width": 2},
                                marker={"size": 7, "color": palette.CATEGORICAL[0]}))
    dark_layout(fig, height=height)
    fig.update_yaxes(title={"text": "Net PnL (USD)", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    return fig


def verdict_donut(counts: dict[str, int], *, height: int = 240) -> go.Figure:
    """Canonical headline-verdict distribution -- a direct plot of
    `RegistrySummary.canonical_verdict_counts`. Segment colors are the shared
    verdict palette (`palette.state_color`), never a generic categorical
    color, so this reads consistently with every badge elsewhere in the app."""
    order = [k for k in ("PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED") if counts.get(k, 0) > 0]
    values = [counts[k] for k in order]
    colors = [palette.state_color(k) for k in order]
    labels = [palette.state_label(k) for k in order]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62, marker={"colors": colors, "line": {"width": 0}},
        textinfo="value", textfont={"color": palette.TEXT_PRIMARY, "size": 13},
        sort=False, direction="clockwise",
    ))
    dark_layout(fig, height=height)
    fig.update_layout(showlegend=True, legend={"orientation": "h", "y": -0.08,
                       "font": {"color": palette.TEXT_SECONDARY, "size": 11}})
    return fig


#: Short in-cell verdict codes -- the full word is in the hover text and in
#: the shared legend/badge vocabulary everywhere else on the page; a heatmap
#: cell is too narrow for "NOT_ADJUDICATED (2)" without overlapping its
#: neighbours (theme=None means Plotly does not auto-fit this for us).
_VERDICT_CODE = {"PASS": "P", "REJECT": "R", "INCONCLUSIVE": "I", "NOT_ADJUDICATED": "NA", "MIXED": "MX"}


def _short_family(name: str) -> str:
    """`ml_meta_label.tsmom` -> `ML·tsmom`; anything else passes through
    unchanged -- both are the real, committed `strategy_family` values."""
    prefix = "ml_meta_label."
    return f"ML·{name[len(prefix):]}" if name.startswith(prefix) else name


def family_market_heatmap(cells: Sequence[dict[str, Any]], *, height: int = 320) -> go.Figure:
    """Root x strategy-family grid, one cell per real canonical-trial group,
    colored by that cell's verdict (or `MIXED` if its trials disagree) --
    never a synthesized "expected" cell for a combination that was not
    actually run."""
    roots = sorted({c["root_symbol"] for c in cells})
    families = sorted({c["strategy_family"] for c in cells})
    family_labels = [_short_family(f) for f in families]
    # "NO_DATA" (no canonical trial at all for this root/family) is its own
    # bucket -- it must never share a z-index with a real verdict. A prior
    # version encoded it as z=-1 with zmin=0, which Plotly's Heatmap silently
    # clamps into the FIRST colorscale color (PASS/green): a combination that
    # was never run rendered as a false PASS. Bug found via a real screenshot
    # of the silver_bullet column (ZN/GC/ES/CL all green, no label) while
    # relocating this chart onto the Strategies page.
    order = ["NO_DATA", "PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED", "MIXED"]
    by_cell = {(c["root_symbol"], c["strategy_family"]): c for c in cells}
    z = [[order.index(by_cell[(r, f)]["verdict"]) if (r, f) in by_cell else 0 for f in families] for r in roots]
    text = [[
        f"{_VERDICT_CODE[by_cell[(r, f)]['verdict']]} ({by_cell[(r, f)]['count']})" if (r, f) in by_cell else ""
        for f in families
    ] for r in roots]
    hover = [[
        f"{r} / {f}<br>{by_cell[(r, f)]['verdict']} -- {by_cell[(r, f)]['count']} canonical trial(s)"
        if (r, f) in by_cell else f"{r} / {f}<br>no canonical trial"
        for f in families
    ] for r in roots]
    n = len(order)
    state_colors = [palette.BORDER, *(palette.state_color(s) for s in order[1:])]
    colorscale = [
        stop
        for i, color in enumerate(state_colors)
        for stop in ([i / n, color], [(i + 1) / n, color])
    ]
    fig = go.Figure(go.Heatmap(
        z=z, x=family_labels, y=roots, text=text, texttemplate="%{text}",
        textfont={"size": 11, "color": palette.TEXT_PRIMARY},
        customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        colorscale=colorscale, zmin=-0.5, zmax=n - 0.5, showscale=False,
        xgap=4, ygap=4,
    ))
    dark_layout(fig, height=height)
    fig.update_layout(margin={"l": 48, "r": 20, "t": 74, "b": 16})
    fig.update_xaxes(side="top", tickangle=-30, tickfont={"size": 10.5, "color": palette.TEXT_SECONDARY})
    fig.update_yaxes(tickfont={"size": 11, "color": palette.TEXT_SECONDARY})
    return fig


def sharpe_fdr_scatter(points: Sequence[dict[str, Any]], *, height: int = 280) -> go.Figure:
    """One marker per canonical trial with both a committed `annualized_sharpe`
    and `bh_q` -- exactly the two already-persisted `ResultRecord` fields,
    colored by that trial's own verdict. No fitted line, no fabricated trend."""
    fig = go.Figure()
    for verdict in ("PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED"):
        subset = [p for p in points if p["verdict"] == verdict]
        if not subset:
            continue
        fig.add_trace(go.Scatter(
            x=[p["bh_q"] for p in subset], y=[p["annualized_sharpe"] for p in subset],
            mode="markers", name=palette.state_label(verdict),
            marker={"color": palette.state_color(verdict), "size": 9,
                    "line": {"color": palette.BG, "width": 1}},
            text=[f"{p['root_symbol']} {p['strategy_family']}" for p in subset],
            hovertemplate="%{text}<br>BH q=%{x:.3f}<br>Sharpe=%{y:.2f}<extra></extra>",
        ))
    dark_layout(fig, height=height)
    fig.update_layout(showlegend=True, legend={"orientation": "h", "y": -0.18,
                       "font": {"color": palette.TEXT_SECONDARY, "size": 10}})
    fig.update_xaxes(title={"text": "BH-FDR q-value", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    fig.update_yaxes(title={"text": "Annualized Sharpe", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    return fig


def _short_label(text: str, max_len: int = 24) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def hbar_chart(labels: Sequence[str], values: Sequence[float], *, color: str = palette.CATEGORICAL[0],
               height: int = 260) -> go.Figure:
    """Horizontal bar, labels top-to-bottom in descending value order (the
    caller is expected to have already sorted `labels`/`values`). Labels are
    shortened on-axis (never the underlying value) with the full text
    available on hover, since `theme=None` (required, see
    `components.plotly_chart`) means Plotly no longer auto-expands the left
    margin for long tick labels."""
    short = [_short_label(t) for t in labels]
    fig = go.Figure(go.Bar(
        x=list(values), y=short, orientation="h", marker_color=color, marker_line_width=0,
        customdata=list(labels), hovertemplate="%{customdata}: %{x}<extra></extra>",
    ))
    dark_layout(fig, height=height)
    fig.update_layout(margin={"l": 168, "r": 20, "t": 16, "b": 44})
    fig.update_yaxes(autorange="reversed", tickfont={"size": 10.5, "color": palette.TEXT_SECONDARY})
    return fig


def gate_stack_bar(rows: Sequence[dict[str, Any]], *, height: int = 300) -> go.Figure:
    """One stacked horizontal bar per validation gate, segmented by the same
    explicit-only states `services.gate_state` already resolves -- an
    aggregate of the per-experiment badges shown elsewhere, not a new
    judgment."""
    labels = [_short_label(r["label"], 22) for r in rows]
    order = ("FAIL", "REFUSED_BEFORE_GATE", "NOT_AVAILABLE", "NOT_EVALUATED", "NOT_ADJUDICATED", "PASS")
    fig = go.Figure()
    for state in order:
        values = [r["states"].get(state, 0) for r in rows]
        if not any(values):
            continue
        fig.add_trace(go.Bar(
            x=values, y=labels, orientation="h", name=palette.state_label(state),
            marker_color=palette.state_color(state), marker_line_width=0,
        ))
    fig.update_layout(barmode="stack")
    dark_layout(fig, height=height)
    fig.update_layout(margin={"l": 172, "r": 20, "t": 16, "b": 44})
    fig.update_yaxes(autorange="reversed", tickfont={"size": 10.5, "color": palette.TEXT_SECONDARY})
    fig.update_layout(showlegend=True, legend={"orientation": "h", "y": -0.12,
                       "font": {"color": palette.TEXT_SECONDARY, "size": 10}})
    return fig


def equity_curve_chart(cum_pnl: Sequence[float], *, height: int = 240) -> go.Figure:
    """A cumulative-net-PnL line over a REAL, verifiably-bound trade ledger's
    own rows (`services.trade_ledger_equity_series`) -- a running sum of real
    per-trade numbers, never inferred from a headline aggregate."""
    color = palette.GREEN if (cum_pnl and cum_pnl[-1] >= 0) else palette.RED
    fig = go.Figure(go.Scatter(
        y=list(cum_pnl), mode="lines", line={"color": color, "width": 1.8},
        fill="tozeroy", fillcolor=_hex_fill(color, 0.10),
    ))
    dark_layout(fig, height=height)
    fig.update_yaxes(title={"text": "Cumulative Net PnL (USD)", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    fig.update_xaxes(title={"text": "Trade #", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    return fig


def drawdown_chart(drawdown_usd: Sequence[float], *, height: int = 200) -> go.Figure:
    """The running peak-to-trough of the same real cumulative series."""
    fig = go.Figure(go.Scatter(
        y=list(drawdown_usd), mode="lines", line={"color": palette.RED, "width": 1.6},
        fill="tozeroy", fillcolor=_hex_fill(palette.RED, 0.14),
    ))
    dark_layout(fig, height=height)
    fig.update_yaxes(title={"text": "Drawdown (USD)", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    fig.update_xaxes(title={"text": "Trade #", "font": {"size": 11, "color": palette.TEXT_MUTED}})
    return fig


def validation_funnel_chart(rows: Sequence[dict[str, Any]], *, height: int = 320) -> go.Figure:
    """Phase 4 -- the Validation Funnel (task spec section 7): one horizontal
    bar per gate, in `GATE_DEFINITIONS` declared order (first = earliest
    evaluated), colored by its own already-adjudicated state. Every bar is
    drawn the SAME width -- this is a sequence/funnel of gates, not a
    magnitude comparison, so width never encodes a statistic. `rows` is
    exactly `services.gate_table(...)`'s output: ``[{"label": ..., "state": ...}, ...]``."""
    labels = [r["label"] for r in rows]
    states = [r["state"] for r in rows]
    colors = [palette.state_color(s) for s in states]
    # reversed so the FIRST gate (earliest evaluated) renders at the TOP of a
    # horizontal bar chart (Plotly draws y-categories bottom-to-top).
    fig = go.Figure(go.Bar(
        x=[1] * len(labels), y=labels[::-1], orientation="h",
        marker={"color": colors[::-1]},
        text=[palette.state_label(s) for s in states[::-1]],
        textposition="inside", insidetextanchor="middle",
        hovertext=[f"{lbl}: {palette.state_label(s)}" for lbl, s in zip(labels[::-1], states[::-1], strict=True)],
        hoverinfo="text",
    ))
    dark_layout(fig, height=height)
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(showgrid=False, title=None, automargin=True)
    # `dark_layout`'s default left margin (56px) is tuned for short numeric/date
    # y-labels; the longest real gate label here ("Walk-Forward Consistency")
    # needs much more room, so this chart overrides it after the fact (`margin`
    # can't be passed through `dark_layout`'s own kwargs -- it already sets that
    # key itself). `automargin=True` above is a second layer of defense: Plotly
    # expands the margin further still if a label is even longer than this.
    fig.update_layout(showlegend=False, bargap=0.25, margin={"l": 190, "r": 20, "t": 16, "b": 44})
    return fig


def percentile_gauge(fraction: float, *, height: int = 90) -> go.Figure:
    """A horizontal percentile position bar for one committed statistic in
    [0, 1] (e.g. `canonical_sharpe_percentile`) -- a direct plot of that number,
    not a derived judgment."""
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=1, fillcolor=palette.BORDER_SOFT, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=max(0.0, min(1.0, fraction)), y0=0, y1=1,
                  fillcolor=palette.BLUE, line_width=0)
    fig.update_xaxes(range=[0, 1], visible=False)
    fig.update_yaxes(range=[0, 1], visible=False)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=height, showlegend=False)
    return fig


def signed_bar_chart(labels: Sequence[str], values: Sequence[float | None], *, hover: Sequence[str],
                     muted: Sequence[bool] | None = None, yaxis_title: str | None = None,
                     height: int = 220) -> go.Figure:
    """One series of signed values (e.g. a Rank IC per horizon or per year)
    against a zero baseline. A ``None`` value draws no bar -- never a zero --
    and ``muted`` bars (not evaluable) render grey; ``hover`` carries the full
    per-bar detail, so no number is printed on the marks."""
    muted = list(muted) if muted is not None else [False] * len(labels)
    fig = go.Figure(go.Bar(
        x=list(labels), y=[v if v is not None else None for v in values],
        marker_color=[palette.GREY if m else palette.BLUE for m in muted], marker_line_width=0,
        marker_cornerradius=4, customdata=list(hover), hovertemplate="%{customdata}<extra></extra>",
    ))
    dark_layout(fig, height=height)
    fig.add_hline(y=0, line_width=1, line_color=palette.TEXT_MUTED)
    fig.update_layout(bargap=0.45)
    if yaxis_title:
        fig.update_yaxes(title={"text": yaxis_title, "font": {"size": 11, "color": palette.TEXT_MUTED}})
    return fig


def factor_series_chart(days: Sequence[str], values: Sequence[float | None], *, name: str,
                        height: int = 220) -> go.Figure:
    """A factor's own values over its window; gaps stay gaps (never filled)."""
    fig = go.Figure(go.Scatter(
        x=list(days), y=list(values), mode="lines", name=name, connectgaps=False,
        line={"color": palette.BLUE, "width": 2}, hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))
    dark_layout(fig, height=height, hovermode="x unified")
    fig.add_hline(y=0, line_width=1, line_color=palette.TEXT_MUTED)
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikecolor=palette.TEXT_MUTED)
    return fig
