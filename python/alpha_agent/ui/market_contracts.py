"""Market Intelligence Data Completion Pass, Checkpoint C -- Contract Ladder
+ Term Structure UI. Pure row/format helpers are separated from the
Streamlit rendering entry point (`render_contracts_tab`) so the actual
table/label logic is unit-testable without a Streamlit runtime, mirroring
`alpha_agent.ui.market_scanner`'s split.

Everything here reads real Databento data through
`alpha_agent.ui.market_home`'s cached wrappers -- never a second network
path, never a fabricated value where a real one is simply unavailable
(Section 6: Open Interest is "N/A -- source unavailable", never ``0``).
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from alpha_agent.data.fractional_price import decimal_to_display
from alpha_agent.marketdata.databento_schemas import (
    UNAVAILABLE_CAPABILITIES,
    ContractEconomicsView,
    ContractLadderResult,
    CurveShape,
    FuturesContract,
    TermStructureResult,
)
from alpha_agent.ui import charts, components, learn_links, market_home, palette

__all__ = [
    "contract_row",
    "format_display_price",
    "render_contracts_tab",
    "term_structure_row",
]


# ---------------------------------------------------------------------------
# pure formatting (Section 6/7/10)
# ---------------------------------------------------------------------------


def format_display_price(price: float | None, econ: ContractEconomicsView | None) -> tuple[str, str | None]:
    """Section 10 -- for a fractionally-quoted product (CBOT Treasuries),
    reuse the EXISTING `alpha_agent.data.fractional_price.decimal_to_display`
    utility for the primary exchange-style quote, and return the plain
    decimal as a second "Data Details" value (never invents new conversion
    logic -- CLAUDE.md/Section 10). Every other product's primary display
    stays plain decimal, with no second value. ``price=None`` renders as the
    honest "N/A", never a fabricated ``0.00``."""
    if price is None:
        return "N/A", None
    if econ is not None and econ.quote_convention == "fractional_32" and econ.quote_tick_size > 0:
        tick_fraction = round(1.0 / econ.quote_tick_size)
        try:
            exchange_style = decimal_to_display(price, tick_fraction=tick_fraction)
        except ValueError:
            return f"{price:,.6f}", None
        return exchange_style, f"{price:,.6f} (decimal)"
    return f"{price:,.2f}", None


def _fmt_or_na(value: float | None, *, digits: int = 0) -> str:
    return f"{value:,.{digits}f}" if value is not None else "N/A"


def contract_row(contract: FuturesContract, *, as_of, econ: ContractEconomicsView | None = None) -> dict[str, Any]:
    """One Contracts-tab table row -- pure, no Streamlit. ``Status`` marks
    FRONT/DISPLAY distinctly (Section 7); either, both, or neither may
    apply to a given row (the ZN case: front-month and display-continuous
    contract can genuinely differ)."""
    status_parts = []
    if contract.is_front_month:
        status_parts.append("FRONT")
    if contract.is_display_contract:
        status_parts.append("DISPLAY")
    last_display, _ = format_display_price(contract.last, econ)
    return {
        "Contract": contract.raw_symbol,
        "Expiry": contract.expiration.date().isoformat(),
        "Days": contract.days_to_expiry if contract.days_to_expiry is not None else "N/A",
        "Last": last_display,
        "Volume": _fmt_or_na(contract.volume),
        "OI": _fmt_or_na(contract.open_interest),
        "Status": " / ".join(status_parts) if status_parts else "",
    }


def term_structure_row(point) -> dict[str, Any]:
    return {
        "Contract": point.raw_symbol,
        "Expiry": point.expiration.date().isoformat(),
        "Price": f"{point.price:,.4f}" if point.price is not None else "N/A",
        "Source": point.price_source,
        "Volume": _fmt_or_na(point.volume),
        "OI": _fmt_or_na(point.open_interest),
    }


_SHAPE_LABEL = {
    CurveShape.CONTANGO: "Contango",
    CurveShape.BACKWARDATION: "Backwardation",
    CurveShape.FLAT: "Flat",
    CurveShape.MIXED: "Mixed",
    CurveShape.INSUFFICIENT_DATA: "Insufficient Data",
}


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------


def _render_ladder(root: str, ladder: ContractLadderResult, econ: ContractEconomicsView | None) -> None:
    with components.card("market-contracts-ladder"):
        st.markdown(f'<div class="aa-gate-title">{root} CONTRACTS</div>', unsafe_allow_html=True)
        st.caption(
            f"Real outright futures from Databento definitions (parent symbology {root}.FUT) · "
            f"{ladder.detail}"
        )
        if not ladder.contracts:
            components.empty_state(
                "Contract Ladder", "No active outright contracts resolved for this root.",
                key="market-contracts-empty",
            )
            return
        rows = [contract_row(c, as_of=ladder.as_of, econ=econ) for c in ladder.contracts]
        st.dataframe(rows, width="stretch", hide_index=True, height=min(480, 44 + 35 * len(rows)))
        st.caption(
            "FRONT = nearest unexpired outright contract by real expiry. DISPLAY = the contract Databento's "
            "own continuous front-month proxy currently resolves to for charting -- these can legitimately "
            "differ (e.g. Treasury futures often roll well before the front month's own expiry)."
        )
        if econ is not None and econ.quote_convention == "fractional_32":
            with st.expander("Data Details -- decimal prices", expanded=False):
                for c in ladder.contracts:
                    _, decimal_detail = format_display_price(c.last, econ)
                    if decimal_detail:
                        components.provenance_row(c.raw_symbol, decimal_detail)


def _render_term_structure(root: str, term_structure: TermStructureResult) -> None:
    with components.card("market-contracts-term-structure"):
        st.markdown('<div class="aa-gate-title">TERM STRUCTURE</div>', unsafe_allow_html=True)
        priced = [p for p in term_structure.points if p.price is not None]
        if len(priced) < 2:
            components.empty_state(
                "Term Structure", "Fewer than 2 priced contracts -- insufficient data for a curve.",
                key="market-term-structure-empty",
            )
            return
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Scatter(
                x=[p.expiration for p in priced], y=[p.price for p in priced], mode="lines+markers",
                line={"color": palette.BLUE, "width": 2},
                text=[p.raw_symbol for p in priced],
                hovertemplate="%{text}<br>Expiry: %{x|%Y-%m-%d}<br>Price: %{y:.4f}<extra></extra>",
            )
        )
        charts.dark_layout(fig, height=280)
        fig.update_yaxes(title={"text": "Price", "font": {"size": 11, "color": palette.TEXT_MUTED}})
        fig.update_xaxes(title={"text": "Expiry", "font": {"size": 11, "color": palette.TEXT_MUTED}})
        components.plotly_chart(fig, key="market-term-structure-fig")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="aa-metric-label">CURVE SHAPE</div>'
                f'<div class="aa-hero-contract-value">{_SHAPE_LABEL[term_structure.curve_shape]}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            spread = term_structure.front_second_spread
            st.markdown(
                f'<div class="aa-metric-label">FRONT/SECOND SPREAD</div>'
                f'<div class="aa-hero-contract-value">{spread:+.4f}</div>' if spread is not None
                else '<div class="aa-metric-label">FRONT/SECOND SPREAD</div><div class="aa-hero-contract-value">N/A</div>',
                unsafe_allow_html=True,
            )
        with c3:
            spread3 = term_structure.front_third_spread
            st.markdown(
                f'<div class="aa-metric-label">FRONT/THIRD SPREAD</div>'
                f'<div class="aa-hero-contract-value">{spread3:+.4f}</div>' if spread3 is not None
                else '<div class="aa-metric-label">FRONT/THIRD SPREAD</div><div class="aa-hero-contract-value">N/A</div>',
                unsafe_allow_html=True,
            )
        st.caption(
            f"Tolerance: {term_structure.tolerance_pct:.2f}% of the front price -- a descriptive observation "
            "of the current curve shape, never a trading signal."
        )
        learn_links.render_learn_why(
            "term_structure_backwardation", key=f"market-term-structure-{root}", label="Learn Why: Contango vs. Backwardation",
        )
        with st.expander("Priced contracts", expanded=False):
            st.dataframe([term_structure_row(p) for p in term_structure.points], width="stretch", hide_index=True)


def render_contracts_tab(root: str) -> None:
    """Product Detail -> Contracts (Checkpoint C)."""
    ladder, _ = market_home.get_contract_ladder_cached(root, max_contracts=12)
    if ladder is None or ladder.capability in UNAVAILABLE_CAPABILITIES or not ladder.fetched:
        detail = ladder.detail if ladder else "Databento is not currently connected."
        components.empty_state("Contracts", f"Real contract data is not available right now: {detail}",
                                key="market-contracts-na")
        return

    econ, _ = market_home.get_contract_metadata_cached(root)
    _render_ladder(root, ladder, econ)

    term_structure, _ = market_home.get_term_structure_cached(root, max_contracts=12)
    if term_structure is not None:
        _render_term_structure(root, term_structure)
