"""Phase 20.1 -- the dark institutional-terminal design system's color tokens.

One place defines every color; `layout.py` turns these into the injected CSS,
`charts.py` turns them into the shared Plotly dark template, and every page
renders badges/text through the helpers here so a color is never picked ad hoc
in a page module. Status colors are reserved for verdict/gate state and never
reused as a generic series color; the chart categorical slots are used in
fixed order (never cycled per-row).
"""
from __future__ import annotations

# -- surfaces ----------------------------------------------------------------
BG = "#0a0d16"              # page background, near-black navy
BG_SIDEBAR = "#0d1120"      # sidebar / header, one step lighter
CARD_BG = "#121729"         # card surface
CARD_BG_RAISED = "#161c33"  # nested / hovered card surface
BORDER = "#232a42"          # thin card border
BORDER_SOFT = "#1a2035"     # hairline / divider

# -- ink -----------------------------------------------------------------
TEXT_PRIMARY = "#eef1f8"
TEXT_SECONDARY = "#8992ab"
TEXT_MUTED = "#5b6480"

# -- accent ----------------------------------------------------------------
BLUE = "#3b82f6"
BLUE_SOFT = "rgba(59, 130, 246, 0.14)"
BLUE_BORDER = "rgba(59, 130, 246, 0.35)"

# -- status (reserved; never a generic series color) --------------------------
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
GREY = "#6b7280"

# -- chart categorical (fixed order) ------------------------------------------
CATEGORICAL = [
    "#3b82f6",  # 1 blue
    "#22d3ee",  # 2 cyan
    "#a78bfa",  # 3 violet
    "#f59e0b",  # 4 amber
    "#f472b6",  # 5 pink
    "#34d399",  # 6 green
    "#fb923c",  # 7 orange
    "#f87171",  # 8 red
]

SEQUENTIAL_BLUE = ["#0f2f66", "#1c4d95", "#2a6ec4", "#3b82f6", "#7db1fb", "#c3ddfe"]

# -- gate / verdict state -> color + icon ------------------------------------
STATE_COLOR = {
    "PASS": GREEN,
    "FAIL": RED,
    "REJECT": RED,
    "INCONCLUSIVE": AMBER,
    "NOT_EVALUATED": GREY,
    "NOT_AVAILABLE": GREY,
    "NOT_ADJUDICATED": GREY,
    "REFUSED_BEFORE_GATE": GREY,
    "MIXED": AMBER,
    "OK": GREEN,
    "READY": GREEN,
    "WARN": AMBER,
    "OFFLINE": GREY,
    "LOCKED": GREEN,  # holdout locked is the GOOD state
    # Opportunity V1 research-triage states (never a verdict, never a trade
    # instruction) -- INVESTIGATE is the most actionable ("a genuine research
    # gap"), WATCH is neutral, WAIT is a timing caution (same tone as WARN).
    "INVESTIGATE": BLUE,
    "WATCH": GREY,
    "WAIT": AMBER,
    # Phase 7 Cross-Asset Alpha Graph evidence-coverage states (never a
    # verdict, never a graph score -- see alpha_agent.alpha_graph.schemas
    # .EvidenceCoverage).
    "RESEARCHED": GREEN,
    "WEAKLY_RESEARCHED": AMBER,
    "UNDEREXPLORED": BLUE,
    "NO_EVIDENCE": GREY,
    # Phase 8 Community Alpha Network -- EvidenceMaturity (never a verdict;
    # see alpha_agent.community.schemas.EvidenceMaturity), VisibilityLevel,
    # and ReplicationOutcome. "REJECT"/"INCONCLUSIVE" above are reused
    # verbatim where the string is identical (INCONCLUSIVE); REJECTED is
    # Community's own spelling and gets its own entry rather than aliasing.
    "PROPOSED": BLUE,
    "REPLICATING": AMBER,
    "EMERGING_EVIDENCE": GREEN,
    "REJECTED": RED,
    "PRIVATE": GREY,
    "SHARED": AMBER,
    "PUBLIC": BLUE,
    "CONFIRMS": GREEN,
    "CONFLICTS": RED,
    "PARTIAL": AMBER,
    "NOT_COMPARABLE": GREY,
    # News Alpha Phase A -- Initial Impact Scan research-triage levels (never
    # a verdict, return, or trade instruction; see alpha_agent.news_alpha
    # .schemas.ImpactLevel). A blue intensity ramp, deliberately not the
    # green/red verdict colors. Prefixed so a bare "HIGH"/"LOW" elsewhere
    # keeps its existing default.
    "IMPACT_VERY_HIGH": SEQUENTIAL_BLUE[4],
    "IMPACT_HIGH": BLUE,
    "IMPACT_MEDIUM": SEQUENTIAL_BLUE[2],
    "IMPACT_LOW": GREY,
    "IMPACT_NONE": GREY,
}

STATE_ICON = {
    "PASS": "✓",
    "FAIL": "✕",
    "REJECT": "✕",
    "INCONCLUSIVE": "◐",
    "NOT_EVALUATED": "—",
    "NOT_AVAILABLE": "—",
    "NOT_ADJUDICATED": "—",
    "REFUSED_BEFORE_GATE": "⚠",
    "MIXED": "◐",
    "OK": "●",
    "READY": "●",
    "WARN": "▲",
    "OFFLINE": "○",
    "LOCKED": "\U0001f512",
    "INVESTIGATE": "●",
    "WATCH": "○",
    "WAIT": "▲",
    "RESEARCHED": "●",
    "WEAKLY_RESEARCHED": "◐",
    "UNDEREXPLORED": "○",
    "NO_EVIDENCE": "—",
    "IMPACT_VERY_HIGH": "●",
    "IMPACT_HIGH": "●",
    "IMPACT_MEDIUM": "◐",
    "IMPACT_LOW": "○",
    "IMPACT_NONE": "—",
}

STATE_LABEL = {
    "NOT_EVALUATED": "NOT EVALUATED",
    "NOT_AVAILABLE": "NOT AVAILABLE",
    "NOT_ADJUDICATED": "NOT ADJUDICATED",
    "REFUSED_BEFORE_GATE": "REFUSED BEFORE GATE",
    "WEAKLY_RESEARCHED": "WEAKLY RESEARCHED",
    "UNDEREXPLORED": "UNDEREXPLORED",
    "NO_EVIDENCE": "NO EVIDENCE",
    "EMERGING_EVIDENCE": "EMERGING EVIDENCE",
    "NOT_COMPARABLE": "NOT COMPARABLE",
}


def state_color(state: str | None) -> str:
    return STATE_COLOR.get((state or "").upper(), GREY)


def state_icon(state: str | None) -> str:
    return STATE_ICON.get((state or "").upper(), "—")


def state_label(state: str | None) -> str:
    s = (state or "NOT_AVAILABLE").upper()
    return STATE_LABEL.get(s, s)
