"""Market data for one affected market in a Research Thread's Market step:
"is the affected market already reacting?"

MARKET-OBSERVATION plane only (the same boundary as the Markets page:
`market_home` / `databento_context`). Nothing here feeds a signal, a
validation or the registry; the panel says so. Prices load ONLY behind an
explicit button (the provider cost-checks every fetch and refuses above its
ceiling) and are then shown from this session's cache on every later render.
Asset-specific fields that no connected source provides are labelled as such
-- never a blank, never a made-up number.
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from alpha_agent.marketdata.databento_schemas import UNAVAILABLE_CAPABILITIES
from alpha_agent.news_alpha import MandateDomain
from alpha_agent.ui import charts_market, components, databento_context, market_home
from alpha_agent.ui.research_thread import MarketGroup
from alpha_agent.ui.workspace.common import ago, esc, facts_html, info_html

__all__ = ["WINDOWS", "field_states", "has_price_feed", "render_market_data"]

#: Window -> (bar timeframe, bars). Bounded by the delayed feed's 30-day
#: lookback ceiling (`DatabentoMarketDataConfig.max_lookback_days`), so longer
#: windows are not offered rather than offered and refused.
WINDOWS: dict[str, tuple[str, int]] = {"1D": ("1h", 24), "5D": ("1h", 120), "1M": ("1d", 22)}
_BARS_PER_YEAR = {"1h": 24 * 252, "1d": 252}

_NOT_CONNECTED = "Not connected"
_NOT_ACQUIRED = "Data not acquired"

#: The fields each asset class is described by, in reading order. Values the
#: loaded bars provide are filled in; everything else keeps its honest state.
_FIELDS: dict[MandateDomain, tuple[tuple[str, str], ...]] = {
    MandateDomain.FUTURES: (
        ("Price", "bars"), ("Return", "bars"), ("Volume", "bars"), ("Volatility", "bars"),
        ("Open interest", "markets"), ("Term structure", "markets"), ("Carry / basis", "markets"),
        ("Inventory", _NOT_CONNECTED),
    ),
    MandateDomain.ETF: (
        ("Price", _NOT_CONNECTED), ("Return", _NOT_CONNECTED), ("Volume", _NOT_CONNECTED),
        ("Volatility", _NOT_CONNECTED), ("Relative performance", _NOT_CONNECTED),
        ("Sector / factor exposure", _NOT_CONNECTED),
    ),
    MandateDomain.EQUITY: (
        ("Price", _NOT_ACQUIRED), ("1D / 5D / 20D return", _NOT_ACQUIRED), ("Volume", _NOT_ACQUIRED),
        ("Volatility", _NOT_ACQUIRED), ("Earnings context", _NOT_CONNECTED), ("Fundamentals", "measure"),
    ),
    MandateDomain.OPTIONS: (
        ("Underlying price", _NOT_CONNECTED), ("ATM implied vol", _NOT_CONNECTED), ("Skew", _NOT_CONNECTED),
        ("Vol term structure", _NOT_CONNECTED), ("Open interest", _NOT_CONNECTED),
    ),
    MandateDomain.CRYPTO: (
        ("Price", _NOT_CONNECTED), ("Funding", _NOT_CONNECTED), ("Basis", _NOT_CONNECTED),
        ("Open interest", _NOT_CONNECTED), ("On-chain", _NOT_CONNECTED),
    ),
}


def _observable_root(group: MarketGroup) -> str | None:
    if group.domain is not MandateDomain.FUTURES:
        return None
    return next((s for s in group.symbols if s.upper() in databento_context.OBSERVABLE_ROOTS), None)


def has_price_feed(group: MarketGroup) -> bool:
    """Whether a recent price source exists for this market (a catalogued futures root)."""
    return _observable_root(group) is not None


def _bars(result) -> list[dict]:
    return [b.model_dump() for b in result.bars] if result is not None else []


def _fmt(v: float | None, fmt: str) -> str:
    return "--" if v is None else format(v, fmt)


def field_states(group: MarketGroup, bars: list[dict], timeframe: str | None) -> list[tuple[str, str]]:
    """``(field, value-or-state)`` for the asset class -- formatting of the
    loaded observation bars only."""
    out = []
    for name, source in _FIELDS.get(group.domain, ()):
        if source == "bars":
            if not bars:
                value = "Load prices"
            elif name == "Price":
                value = _fmt(bars[-1]["close"], ",.2f")
            elif name == "Return":
                change = market_home.window_change_pct(bars)
                value = "--" if change is None else f"{change:+.2f}%"
            elif name == "Volume":
                vol = bars[-1].get("volume")
                value = "--" if vol is None else f"{vol:,.0f} (last bar)"
            else:
                rv = market_home.realized_volatility_pct(bars, bars_per_year=_BARS_PER_YEAR.get(timeframe or "1h", 6048))
                value = "--" if rv is None else f"{rv:.1f}% annualized"
        elif source == "markets":
            value = "In full detail"
        elif source == "measure":
            value = "See Measure it"
        else:
            value = source
        out.append((name, value))
    return out


def render_market_data(group: MarketGroup, *, key: str) -> None:
    root = _observable_root(group)
    left, right = st.columns([1.55, 1], gap="medium")
    bars: list[dict] = []
    timeframe: str | None = None
    with left:
        st.markdown('<div class="aa-eyebrow">Price trend' + info_html(
            "Delayed CME prices for context: is the affected market already reacting? Observation only -- never "
            "used in a signal, a backtest or a validation.") + "</div>", unsafe_allow_html=True)
        if root is None:
            _render_unconnected(group)
        else:
            window = st.segmented_control("Window", list(WINDOWS), default="5D", key=f"{key}-window",
                                          label_visibility="collapsed") or "5D"
            timeframe, n = WINDOWS[window]
            peek = market_home.peek_ohlcv_cached(root, timeframe=timeframe, lookback_bars=n)
            if peek is None:
                with components.card(f"{key}-price-empty"):
                    st.markdown(
                        f'<div class="aa-subnote" style="margin:0.2rem 0">Recent {esc(window)} prices for '
                        f"<b>{esc(root)}</b> are not loaded yet.</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button(f"Load {window} prices", key=f"{key}-load", icon=":material/download:",
                                 help="Fetches delayed CME bars for this contract through the same cost-checked path "
                                      "Explore All Markets uses (typically $0; refused above the configured ceiling). "
                                      "Can take up to a minute."):
                        with st.spinner(f"Loading {root} prices…"):
                            market_home.get_ohlcv_cached(root, timeframe=timeframe, lookback_bars=n)
                        st.rerun()
            else:
                result, loaded_at = peek
                bars = _render_chart(result, loaded_at, root=root, key=key)
    with right:
        st.markdown('<div class="aa-eyebrow">Market data' + info_html(
            "The statistics that describe this asset class. 'Not connected' / 'Data not acquired' mean no source "
            "for it exists in this build yet.") + "</div>", unsafe_allow_html=True)
        st.markdown(facts_html(field_states(group, bars, timeframe)), unsafe_allow_html=True)
        if root is not None and st.button(
            "Full market detail", key=f"{key}-open-market", icon=":material/open_in_new:", type="tertiary",
            help="Contract ladder, term structure, relative moves, news and events for this market -- in Market → "
                 "Explore All Markets.",
        ):
            from alpha_agent.ui.views import market

            market.open_explorer(root)


def _render_chart(result, loaded_at: datetime, *, root: str, key: str) -> list[dict]:
    if result is None or result.capability in UNAVAILABLE_CAPABILITIES or not result.bars:
        detail = (result.detail if result is not None and result.detail else
                  "The delayed market-data feed is not connected in this environment.")
        components.empty_state("Prices unavailable", detail, key=f"{key}-price-unavailable")
        return []
    bars = _bars(result)
    components.plotly_chart(charts_market.market_price_chart(bars, height=250), key=f"{key}-price-chart")
    st.caption(f"{root} · delayed CME data (Databento) · {len(bars)} bars · loaded {ago(loaded_at)}. Observation "
               "only -- not research data.")
    return bars


def _render_unconnected(group: MarketGroup) -> None:
    text = {
        MandateDomain.ETF: "Recent ETF prices are not connected in this build. Signals use the acquired 2018-2022 "
                           "daily history of these funds.",
        MandateDomain.EQUITY: "Equity price data has not been acquired yet, so no equity chart or statistic can be "
                              "shown -- and no equity signal can be built.",
    }.get(group.domain, "No price source is connected for this asset class yet.")
    with components.card(f"mx-unconnected-{group.key}"):
        st.markdown(f'<div class="aa-subnote" style="margin:0.2rem 0">{esc(text)}</div>', unsafe_allow_html=True)
