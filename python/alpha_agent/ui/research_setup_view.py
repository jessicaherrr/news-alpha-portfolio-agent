"""Research Setup -- the persistent context every Research Thread is read
under (user-facing name for the backend `ResearchMandate`).

Not a destination: a compact summary that sits on News and in every thread,
plus one editor dialog. The two halves keep their existing single sources of
truth -- asset access / shorting / leverage / liquidity / instrument limits
through `news_alpha_context.save_mandate` (`MandateStore`), horizon / risk /
drawdown / capital / turnover through the saved `InvestorProfile` -- so the
Setup can never disagree with what the rest of the app reads.

Changing the Setup never changes a scientific result: it scopes which
domains research may continue in and the constraints a portfolio is sized
under. Threads notice the change themselves (`research_thread.reconcile_setup`).
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.core.instrument import InstrumentIdentity
from alpha_agent.news_alpha import (
    DOMAIN_LABELS,
    LiquidityRequirement,
    MandateDomain,
    ResearchMandate,
    registry_domain_for,
)
from alpha_agent.recommendation.profile import (
    HoldingPeriod,
    MaxDrawdown,
    RiskStyle,
    TurnoverSensitivity,
)
from alpha_agent.ui import news_alpha_context, panels, services

__all__ = ["open_editor", "render_setup_dialog_if_open", "render_setup_summary", "setup_items"]

_OPEN_KEY = "research_setup_editor_open"

_LIQUIDITY_TEXT = {
    LiquidityRequirement.ANY: "Any", LiquidityRequirement.STANDARD: "Standard", LiquidityRequirement.HIGH: "High",
}
_SUPPORT_TEXT = {
    "RESEARCH_READY": "research-ready",
    "DATA_NOT_ACQUIRED": "data not acquired yet",
    "FOUNDATION_ONLY": "foundation only",
    "SYNTHETIC_ONLY": "synthetic only",
    "NOT_SUPPORTED": "not supported yet",
}


def _money(v: float | None) -> str:
    if not v:
        return "Not stated"
    return f"${v / 1e6:,.1f}M" if v >= 1e6 else f"${v:,.0f}"


def setup_items(mandate: ResearchMandate) -> list[tuple[str, str]]:
    """``(label, value)`` pairs in reading order -- formatting only."""
    domains = " + ".join(DOMAIN_LABELS[d] for d in mandate.allowed_domains) or "None"
    risk = f"{mandate.risk_style.value} · max drawdown {mandate.risk_profile.max_drawdown.value}"
    turnover = mandate.turnover_sensitivity.value if mandate.turnover_sensitivity else "Not stated"
    return [
        ("Assets", domains),
        ("Horizon", mandate.investment_horizon.value),
        ("Risk", risk),
        ("Shorting", "Allowed" if mandate.shorting_allowed else "Not allowed"),
        ("Leverage", f"Max {mandate.max_gross_leverage:g}x" if mandate.max_gross_leverage else "Not stated"),
        ("Liquidity", _LIQUIDITY_TEXT[mandate.liquidity_requirement]),
        ("Capital", _money(mandate.capital_usd)),
        ("Turnover sensitivity", turnover),
    ]


def open_editor() -> None:
    st.session_state[_OPEN_KEY] = True


def _close_editor() -> None:
    st.session_state[_OPEN_KEY] = False


def render_setup_summary(mandate: ResearchMandate, *, key: str, layout: str = "strip") -> None:
    """The persistent Setup summary. ``layout="strip"``: one wrapped line of
    label/value chips (News, thread header). ``layout="stack"``: label over
    value rows (a side panel)."""
    items = setup_items(mandate)
    default = "" if news_alpha_context.has_saved_mandate() else (
        '<span class="aa-setup-default" title="These are the platform defaults -- edit to make them yours.">'
        "default</span>"
    )
    if layout == "stack":
        rows = "".join(
            f'<div class="aa-ctx-row"><div class="aa-ctx-label">{label}</div>'
            f'<div class="aa-ctx-value">{value}</div></div>'
            for label, value in items
        )
        st.markdown(f'<div class="aa-ctx">{rows}</div>', unsafe_allow_html=True)
    else:
        chips = "".join(
            f'<span class="aa-setup-chip"><span class="aa-setup-k">{label}</span>{value}</span>'
            for label, value in items[:5]
        )
        with st.container(horizontal=True, vertical_alignment="center", gap="small", key=f"{key}-setup-strip"):
            st.markdown(
                f'<div class="aa-setup-strip"><span class="aa-eyebrow">Research Setup</span>{default}{chips}</div>',
                unsafe_allow_html=True, width="content",
            )
            _edit_button(key)
        return
    _edit_button(key)


def _edit_button(key: str) -> None:
    st.button(
        "Edit setup", key=f"{key}-edit-setup", icon=":material/tune:", type="tertiary", on_click=open_editor,
        help="Change which asset classes research may continue in, your horizon, risk limits, shorting, leverage, "
             "liquidity and capital. Affects research scope and portfolio constraints -- never a scientific result.",
    )


def render_setup_dialog_if_open(mandate: ResearchMandate) -> None:
    """Call once per page render, after the summary. The dialog's open state
    lives in session state (not in the click itself), so it survives the
    reruns its own widgets trigger."""
    if st.session_state.get(_OPEN_KEY):
        _setup_dialog(mandate)


@st.dialog("Research Setup", width="large", on_dismiss=_close_editor)
def _setup_dialog(mandate: ResearchMandate) -> None:
    universe = news_alpha_context.allowed_universe(ResearchMandate(allowed_domains=tuple(MandateDomain)))
    profile = panels.current_investor_profile()
    st.caption(
        "The context every research thread is read under. It decides where research may continue and the limits a "
        "portfolio is built within -- it never changes what the evidence says."
    )
    with st.form(key="research-setup-form", border=False):
        domains = st.pills(
            "Asset classes", list(MandateDomain), selection_mode="multi", default=list(mandate.allowed_domains),
            format_func=lambda d: DOMAIN_LABELS[d], key="research-setup-domains",
            help="Where research may continue. Excluded classes still show up as 'not in your setup', so you can "
                 "see what you are leaving out. Platform support: " + "; ".join(
                     f"{DOMAIN_LABELS[s.domain]} {_SUPPORT_TEXT.get(s.support.value, s.support.value.lower())}"
                     for s in universe.scopes),
        )
        c1, c2 = st.columns(2)
        with c1:
            horizon = st.selectbox(
                "Investment horizon", list(HoldingPeriod), index=list(HoldingPeriod).index(profile.holding_period),
                format_func=lambda h: h.value, key="research-setup-horizon",
                help="How long you typically hold. Signals whose prediction horizon does not match are flagged, never "
                     "hidden.",
            )
            risk = st.segmented_control(
                "Risk style", list(RiskStyle), default=profile.risk_style, format_func=lambda r: r.value,
                key="research-setup-risk", help="Sets the portfolio's target volatility (conservative 6%, balanced "
                                                 "10%, aggressive 15%).",
            )
            drawdown = st.segmented_control(
                "Max drawdown", list(MaxDrawdown), default=profile.max_drawdown, format_func=lambda d: d.value,
                key="research-setup-drawdown",
            )
            turnover = st.segmented_control(
                "Turnover sensitivity", list(TurnoverSensitivity), default=profile.turnover_sensitivity,
                format_func=lambda t: t.value, key="research-setup-turnover",
                help="How much trading activity you are comfortable with. Optional.",
            )
        with c2:
            shorting = st.toggle("Shorting allowed", value=mandate.shorting_allowed, key="research-setup-shorting",
                                 help="Without shorting, a signal that points short is held flat, never dropped.")
            leverage = st.number_input(
                "Max gross leverage (x)", min_value=0.0, max_value=20.0, step=0.5,
                value=float(mandate.max_gross_leverage or 0.0), key="research-setup-leverage",
                help="0 means not stated -- portfolios then use 1.0x and say so.",
            )
            liquidity = st.segmented_control(
                "Liquidity requirement", list(LiquidityRequirement), default=mandate.liquidity_requirement,
                format_func=_LIQUIDITY_TEXT.__getitem__, key="research-setup-liquidity",
            )
            capital = st.number_input(
                "Capital (USD)", min_value=0.0, step=100_000.0, value=float(profile.approximate_capital_usd or 0.0),
                key="research-setup-capital", format="%.0f",
                help="0 means not stated -- portfolios then size against a labelled $1M reference.",
            )
        with st.expander("Limit instruments (optional)"):
            limits: dict[MandateDomain, list[str]] = {}
            for scope in universe.scopes:
                if not scope.instruments:
                    continue
                current = [i.symbol for i in mandate.instrument_allowlist if i.asset_domain.value == scope.domain.value]
                limits[scope.domain] = st.multiselect(
                    f"{DOMAIN_LABELS[scope.domain]} (empty = the whole universe)",
                    [i.symbol for i in scope.instruments], default=current, key=f"research-setup-limit-{scope.domain.value}",
                )
        saved = st.form_submit_button("Save setup", type="primary", icon=":material/check:")
    if saved:
        chosen = tuple(domains or ())
        new_profile = profile.model_copy(update={
            "holding_period": horizon, "risk_style": risk or profile.risk_style,
            "max_drawdown": drawdown or profile.max_drawdown, "turnover_sensitivity": turnover,
            "approximate_capital_usd": capital or None,
        })
        allowlist = tuple(
            InstrumentIdentity(asset_domain=registry_domain_for(d), symbol=sym)
            for d, syms in limits.items() if d in chosen and registry_domain_for(d) is not None
            for sym in syms
        )
        try:
            new = ResearchMandate(
                allowed_domains=chosen, instrument_allowlist=allowlist,
                instrument_denylist=mandate.instrument_denylist, shorting_allowed=shorting,
                max_gross_leverage=leverage or None, liquidity_requirement=liquidity or mandate.liquidity_requirement,
                risk_profile=new_profile,
            )
        except ValueError as exc:
            st.error(f"Setup not saved: {exc}")
            return
        if new_profile != profile:
            services.save_investor_profile(new_profile)
            st.session_state[panels._PROFILE_KEY] = new_profile
        news_alpha_context.save_mandate(new)
        _close_editor()
        st.rerun()
