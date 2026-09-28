"""Phase 20.1 -- reusable rendering components for the terminal-style UI.

Every page is built out of these instead of hand-rolling markup inline, so the
"same visual shell" requirement (persistent card geometry, badge vocabulary,
spacing) is enforced structurally rather than by convention. Nothing here
computes a value -- each component only formats/positions a value its caller
already has from `alpha_agent.ui.services`.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime

import streamlit as st

from alpha_agent.ui import palette, services

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text).strip("-").lower()


@contextmanager
def metric_row(key: str):
    """A row of metric/verdict cards (or native `st.metric` values) wrapped in
    a stable keyed container (`st-key-metric-row-<key>`) so the injected
    responsive CSS (`layout._CSS`) turns its `st.columns(...)` into a true CSS
    auto-fit grid -- a value never shrinks into "$16" / "3" the way a fixed
    flex-basis column does; the row wraps into more lines instead. Callers
    build columns/cards exactly as before; this only adds the scoping
    wrapper, so it changes presentation, never a value."""
    with st.container(key=f"metric-row-{_slug(key)}"):
        yield


@contextmanager
def gate_row(key: str):
    """Same mechanism as `metric_row`, sized for the wider Robustness /
    Validation gate cards (title + badge + stat + note need more than a bare
    metric's 150px minimum) -- see `render_gate_grid`."""
    with st.container(key=f"gate-row-{_slug(key)}"):
        yield


@contextmanager
def card(key: str):
    """A bordered card surface. ``key`` must be unique on the page; Streamlit
    stamps the resulting `st.container` itself with class ``st-key-card-<key>``,
    which `layout.py`'s injected CSS targets directly for the navy card look."""
    with st.container(border=True, key=f"card-{_slug(key)}"):
        yield


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="aa-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="aa-section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def badge(state: str, *, label: str | None = None) -> str:
    """An inline HTML badge string for the given state (see `palette.STATE_COLOR`).
    Callers embed it inside their own `st.markdown(..., unsafe_allow_html=True)`."""
    color = palette.state_color(state)
    text = label if label is not None else palette.state_label(state)
    icon = palette.state_icon(state)
    return (
        f'<span class="aa-badge" style="color:{color};border-color:{color}66;'
        f'background:{color}1f">{icon} {text}</span>'
    )


def render_badge(state: str, *, label: str | None = None) -> None:
    st.markdown(badge(state, label=label), unsafe_allow_html=True)


def metric_card(
    key: str,
    label: str,
    value: str,
    *,
    sub: str | None = None,
    sub_color: str | None = None,
    accent: str | None = None,
) -> None:
    """One compact top-row metric card: muted label, large value, small sub-line."""
    with card(key):
        st.markdown(f'<div class="aa-metric-label">{label}</div>', unsafe_allow_html=True)
        color = f"color:{accent}" if accent else ""
        st.markdown(f'<div class="aa-metric-value" style="{color}">{value}</div>', unsafe_allow_html=True)
        if sub:
            c = sub_color or palette.TEXT_MUTED
            st.markdown(f'<div class="aa-metric-sub" style="color:{c}">{sub}</div>', unsafe_allow_html=True)


def verdict_metric_card(key: str, label: str, verdict: str | None) -> None:
    v = verdict or "NOT_AVAILABLE"
    color = palette.state_color(v)
    text = palette.state_label(v)
    size = "1.2rem" if len(text) <= 7 else "0.92rem"
    with card(key):
        st.markdown(f'<div class="aa-metric-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            # No inline `white-space:nowrap` -- a long state label ("NOT
            # ADJUDICATED") may still wrap at its own space if the grid ever
            # gives it less room than that; forcing nowrap would instead let
            # it bleed sideways into the next card (see `.aa-metric-value`'s
            # comment in `layout._CSS`).
            f'<div class="aa-metric-value" style="color:{color};font-size:{size}">{text}</div>',
            unsafe_allow_html=True,
        )


def unavailable_metric_card(key: str, label: str, reason: str, *, short: str = "not committed") -> None:
    with card(key):
        st.markdown(f'<div class="aa-metric-label">{label}</div>', unsafe_allow_html=True)
        st.markdown('<div class="aa-metric-value aa-muted">N/A</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="aa-metric-sub" title="{reason}">{short}</div>', unsafe_allow_html=True)


def gate_card(key: str, label: str, state: str, *, stat_text: str = "", interpretation: str = "") -> None:
    """One robustness/validation gate card: title, badge, a factual statistic
    line, and a short interpretation -- never a threshold applied here."""
    with card(key):
        top = st.container()
        with top:
            c1, c2 = st.columns([3, 2])
            c1.markdown(f'<div class="aa-gate-title">{label}</div>', unsafe_allow_html=True)
            with c2:
                render_badge(state)
        if stat_text:
            st.markdown(f'<div class="aa-gate-stat">{stat_text}</div>', unsafe_allow_html=True)
        if interpretation:
            st.markdown(f'<div class="aa-gate-note">{interpretation}</div>', unsafe_allow_html=True)


def committed_series_notice(
    *,
    key: str | None = None,
    text: str = (
        "Detailed path-dependent series are not committed for this experiment. "
        "Summary and statistical evidence below remain authoritative."
    ),
) -> None:
    """One compact callout replacing several giant NOT-AVAILABLE chart panels
    when no experiment-bound equity/drawdown/trade/position series exists --
    the invariant (never fabricate one) is unchanged, only how much page
    space its absence is allowed to claim."""
    ctx = card(key) if key else _null_ctx()
    with ctx:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.65rem;padding:0.15rem 0.05rem;">'
            f'{badge("NOT_AVAILABLE", label="NO COMMITTED SERIES")}'
            f'<span style="font-size:0.8rem;color:{palette.TEXT_SECONDARY};line-height:1.35">{text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def empty_state(title: str, reason: str, *, key: str | None = None) -> None:
    """A clearly-labeled NOT AVAILABLE placeholder card -- used wherever no
    committed, experiment-bound artifact exists. Never a fabricated chart."""
    ctx = card(key) if key else _null_ctx()
    with ctx:
        st.markdown(
            f'<div class="aa-empty"><div class="aa-empty-title">{title}</div>'
            f'<div class="aa-empty-badge">{badge("NOT_AVAILABLE")}</div>'
            f'<div class="aa-empty-reason">{reason}</div></div>',
            unsafe_allow_html=True,
        )


@contextmanager
def _null_ctx():
    yield


def provenance_row(label: str, value: str | None) -> None:
    v = value if value else "--"
    st.markdown(
        f'<div class="aa-prov-row"><span class="aa-prov-label">{label}</span>'
        f'<span class="aa-prov-value">{v}</span></div>',
        unsafe_allow_html=True,
    )


def activity_row(icon: str, title: str, detail: str, when: str) -> None:
    """``icon`` is a semantic state key (see `palette.STATE_COLOR`, e.g.
    ``"OK"``/``"WARN"``), not a literal glyph -- rendered as a small status
    dot, never a decorative emoji character."""
    dot_color = palette.state_color(icon)
    st.markdown(
        f'<div class="aa-activity-row">'
        f'<div class="aa-activity-icon" style="background:{dot_color}"></div>'
        f'<div class="aa-activity-body"><div class="aa-activity-title">{title}</div>'
        f'<div class="aa-activity-detail">{detail}</div></div>'
        f'<div class="aa-activity-time">{when}</div></div>',
        unsafe_allow_html=True,
    )


def status_row(label: str, state: str, detail: str = "") -> None:
    color = palette.state_color(state)
    st.markdown(
        f'<div class="aa-status-row"><span class="aa-status-dot" style="background:{color}"></span>'
        f'<span class="aa-status-label">{label}</span>'
        f'<span class="aa-status-detail">{detail}</span></div>',
        unsafe_allow_html=True,
    )


def feature_tag(text: str) -> str:
    return f'<span class="aa-tag">{text}</span>'


def render_tags(tags: Iterable[str]) -> None:
    st.markdown('<div class="aa-tag-row">' + "".join(feature_tag(t) for t in tags) + "</div>",
                unsafe_allow_html=True)


def now_utc_str() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def render_gate_grid(
    *, result: dict | None, trial_role: str, verdict: str | None, per_row: int = 3, key_prefix: str = "gate"
) -> None:
    """The Robustness & Statistical Validation grid: `services.GATE_DEFINITIONS`
    laid out `per_row` at a time, each an explicit-only `gate_state` badge plus
    its factual statistic line from `services.gate_evidence_text`."""
    reason_codes = tuple((result or {}).get("reason_codes") or ())
    stats = services.gate_evidence_text(result)
    defs = services.GATE_DEFINITIONS
    for i in range(0, len(defs), per_row):
        with gate_row(f"{key_prefix}-{i}"):
            cols = st.columns(per_row)
            for col, (label, fail_code) in zip(cols, defs[i : i + per_row]):
                state = services.gate_state(
                    fail_code=fail_code, reason_codes=reason_codes, verdict=verdict, trial_role=trial_role
                )
                with col:
                    gate_card(f"{key_prefix}-{_slug(label)}", label, state, stat_text=stats.get(label, ""))


def three_lens(
    key: str, *, intuition: str, quant: str, implementation: str, pointers: Iterable[str] = ()
) -> None:
    """Phase 4 -- the shared Intuition / Quant / Implementation explainer
    (task spec section 3). Every caller supplies real text; this only lays
    it out identically everywhere it appears (Concepts, Strategies)."""
    with card(key):
        tabs = st.tabs(["Intuition", "Quant", "Implementation"])
        with tabs[0]:
            st.write(intuition)
        with tabs[1]:
            st.write(quant)
        with tabs[2]:
            st.write(implementation)
            pointers = tuple(pointers)
            if pointers:
                st.caption("Code: " + " &middot; ".join(f"`{p}`" for p in pointers), unsafe_allow_html=True)


def render_validation_funnel(rows: list[dict[str, str]], *, key: str) -> None:
    """Phase 4 -- the Validation Funnel (task spec section 7): the primary,
    at-a-glance view of `rows` (`services.gate_table(...)`'s output), with the
    existing gate-by-gate detail cards staying available immediately below on
    the calling page for anyone who wants the full statistic per gate."""
    from alpha_agent.ui.charts import validation_funnel_chart

    with card(f"funnel-{_slug(key)}"):
        plotly_chart(validation_funnel_chart(rows), key=f"funnel-chart-{_slug(key)}")
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["state"]] = counts.get(r["state"], 0) + 1
        summary = " &middot; ".join(f"{n} {palette.state_label(s)}" for s, n in counts.items())
        st.caption(summary, unsafe_allow_html=True)


def plotly_chart(fig, *, key: str | None = None) -> None:
    """The one place `st.plotly_chart` is called. `theme=None` is required --
    Streamlit's own theme post-processing wraps a titleless figure's title in
    `<b><b>undefined</b></b>` (it string-interpolates `fig.layout.title.text`
    without a null check); every chart here is already styled by
    `charts.dark_layout`, so Streamlit's theme layer is unwanted regardless.
    Pass an explicit `key` whenever the SAME chart-building call renders more
    than once on one page (e.g. one sparkline per ticker card) -- Streamlit
    auto-derives an element id from the figure's type and content, and two
    calls that happen to produce identical figures (same shape, same values)
    otherwise collide with `StreamlitDuplicateElementId`."""
    st.plotly_chart(fig, width="stretch", theme=None, key=key)


def header_pill(label: str, state: str) -> str:
    color = palette.state_color(state)
    palette.state_icon(state)
    return (
        f'<span class="aa-pill"><span class="aa-pill-dot" style="background:{color}"></span>'
        f'<span class="aa-pill-label">{label}</span></span>'
    )
