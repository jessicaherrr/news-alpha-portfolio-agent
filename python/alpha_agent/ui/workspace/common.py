"""Small, pure HTML/text formatters shared by the News page and every
workspace step. Formatting only: each takes a value the backend already
computed and lays it out. Every piece of text that came from data (a
headline, a state label) is HTML-escaped before it enters markup."""
from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

import streamlit as st

from alpha_agent.news_alpha import (
    DOMAIN_LABELS,
    AvailabilityStatus,
    ImpactLevel,
    InitialImpactScan,
    MandateDomain,
    PathType,
    TransmissionLag,
)
from alpha_agent.news_alpha.channels import ImpactHorizon
from alpha_agent.ui import palette

__all__ = [
    "PATH_TYPE_COLOR",
    "PATH_TYPE_TEXT",
    "ago",
    "chain_html",
    "esc",
    "facts_html",
    "impact_row_html",
    "info_html",
    "kind_html",
    "lag_text",
    "level_text",
    "meta_html",
    "subhead",
]

PATH_TYPE_TEXT: dict[PathType | None, str] = {
    PathType.DIRECT: "Direct", PathType.SUPPLY_CHAIN: "Supply chain", PathType.CROSS_SECTOR: "Cross sector",
    None: "Unclassified",
}
#: Categorical slots (never verdict colors) -- mirrors the card rules in `layout._CSS`.
PATH_TYPE_COLOR: dict[PathType | None, str] = {
    PathType.DIRECT: palette.CATEGORICAL[0], PathType.SUPPLY_CHAIN: palette.CATEGORICAL[2],
    PathType.CROSS_SECTOR: palette.CATEGORICAL[1], None: palette.TEXT_MUTED,
}
_LAG_TEXT = {
    TransmissionLag.IMMEDIATE: "Immediate", TransmissionLag.WEEKS: "Weeks", TransmissionLag.MONTHS: "Months",
    TransmissionLag.QUARTERS: "Quarters", TransmissionLag.YEARS: "Years", TransmissionLag.UNKNOWN: "Unknown",
}
_HORIZON_TEXT = {
    ImpactHorizon.INTRADAY_TO_DAYS: "Intraday to days", ImpactHorizon.DAYS_TO_WEEKS: "Days to weeks",
    ImpactHorizon.WEEKS_TO_MONTHS: "Weeks to months",
}
_OUT_OF_SCOPE = {
    AvailabilityStatus.EXCLUDED_BY_MANDATE: "not in your setup",
    AvailabilityStatus.DOMAIN_NOT_SUPPORTED: "not supported yet",
}


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def lag_text(lag: TransmissionLag) -> str:
    return _LAG_TEXT[lag]


def horizon_text(h: ImpactHorizon | None) -> str:
    return _HORIZON_TEXT.get(h, "Unknown") if h is not None else "Unknown"


def level_text(level: ImpactLevel | None) -> str:
    return "--" if level is None else level.value.replace("_", " ")


def kind_html(path_type: PathType | None) -> str:
    return (f'<span class="aa-kind" style="color:{PATH_TYPE_COLOR[path_type]}">'
            f'{esc(PATH_TYPE_TEXT[path_type])}</span>')


def chain_html(labels: Sequence[str], *, market: str | None = None) -> str:
    """A chain of economic states as nodes joined by arrows; the first node
    is the event's starting state, the last the consequence. ``market``
    appends the market expression as a final, dashed node."""
    parts = []
    for i, label in enumerate(labels):
        cls = "aa-node"
        if i == 0:
            cls += " aa-node-start"
        if i == len(labels) - 1 and market is None:
            cls += " aa-node-end"
        parts.append(f'<span class="{cls}">{esc(label)}</span>')
    if market is not None:
        parts.append(f'<span class="aa-node aa-node-market">{esc(market)}</span>')
    return '<div class="aa-chain">' + '<span class="aa-arrow">→</span>'.join(parts) + "</div>"


def facts_html(pairs: Iterable[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div><div class="aa-fact-k">{esc(k)}</div>'
        f'<div class="aa-fact-v{" aa-dim" if v in ("--", "Unknown", "Not available") else ""}">{esc(v)}</div></div>'
        for k, v in pairs
    )
    return f'<div class="aa-facts">{cells}</div>'


def _meter(level: ImpactLevel) -> str:
    on = level.rank
    return '<span class="aa-meter">' + "".join(
        f'<i class="{"on" if i <= on else ""}"></i>' for i in range(1, len(ImpactLevel))
    ) + "</span>"


def impact_row_html(scan: InitialImpactScan, *, domains: Sequence[MandateDomain] | None = None) -> str:
    """The initial scan per asset class: a small level meter for every
    assessed class in scope, then the out-of-scope ones, dashed. The level is
    research-triage relevance -- never a return, a probability or a size."""
    chips: list[str] = []
    for a in scan.assessments:
        if domains is not None and a.asset_domain not in domains:
            continue
        label = esc(DOMAIN_LABELS[a.asset_domain])
        if a.impact_level is not None:
            chips.append(
                f'<span class="aa-impact" title="Initial impact scan: {esc(a.availability_note)}">{label} '
                f'{_meter(a.impact_level)} <b>{esc(level_text(a.impact_level))}</b></span>'
            )
        else:
            why = _OUT_OF_SCOPE.get(a.availability, a.availability.value.replace("_", " ").lower())
            chips.append(f'<span class="aa-impact aa-impact-out" title="{esc(a.availability_note)}">'
                         f'{label} · {esc(why)}</span>')
    return '<div class="aa-impact-row">' + "".join(chips) + "</div>"


def ago(ts: datetime, *, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    seconds = (now - ts).total_seconds()
    if seconds < 0:
        days = -seconds / 86400
        return "in <1 day" if days < 1 else f"in {int(days)} day{'s' if int(days) != 1 else ''}"
    if seconds < 120:
        return "1 min ago"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    days = int(seconds // 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def info_html(tip: str) -> str:
    """A small inline "i" that explains its neighbour on hover or focus."""
    return f'<span class="aa-info" tabindex="0" data-tip="{esc(tip)}">i</span>'


def meta_html(pairs: Iterable[tuple[str, str]]) -> str:
    """Compact one-line label/value pairs for cards."""
    return '<div class="aa-meta">' + "".join(f"<span><b>{esc(k)}</b>{esc(v)}</span>" for k, v in pairs) + "</div>"


def subhead(title: str, note: str | None = None, *, help: str | None = None) -> None:
    st.markdown(f'<div class="aa-subhead">{esc(title)}{info_html(help) if help else ""}</div>',
                unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="aa-subnote">{esc(note)}</div>', unsafe_allow_html=True)
