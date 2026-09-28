"""Research Map -- the Phase 7 Cross-Asset Alpha Graph (prompt 7).

Lives INSIDE Research as a tab (never a new top-level page), purely a read
surface over `alpha_agent.ui.services`' Phase 7 wrappers, which are
themselves a thin, typed projection of `alpha_agent.alpha_graph` (a read
model over the registry / Alpha Memory / My Alpha, renamed from "Factor
Library" in Phase 8) -- opening or
browsing anything on this page never runs a backtest, calls Claude, calls a
market-data API, or writes the registry.

A structured, expandable relationship view -- Mechanism -> Instruments ->
Factors -> Evidence status -- deliberately NOT a force-directed graph
(prompt section 7). "Researched" here means real registry evidence exists;
it is never a claim that the evidence PASSED, and a Mechanism's evidence on
one instrument is never merged or transferred to another (prompt section 4).
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from alpha_agent.ui import components, layout, services

PAGE_TITLE = "Research Map"

_SELECTED_KEY = "alpha_graph_selected_mechanism"

_COVERAGE_ORDER = ("RESEARCHED", "WEAKLY_RESEARCHED", "UNDEREXPLORED", "NO_EVIDENCE")
_COVERAGE_LABELS = {
    "RESEARCHED": "Researched", "WEAKLY_RESEARCHED": "Weakly Researched",
    "UNDEREXPLORED": "Underexplored", "NO_EVIDENCE": "No Evidence",
}
_COVERAGE_DESCRIPTIONS = {
    "RESEARCHED": "Real registry evidence exists -- multiple experiments and/or a canonical headline verdict.",
    "WEAKLY_RESEARCHED": "Real evidence exists, but only one real execution attempt so far -- not yet adjudicated.",
    "UNDEREXPLORED": (
        "This Mechanism is mapped to a real, structurally researchable Strategy family here, but no "
        "experiment has been run on this instrument yet."
    ),
    "NO_EVIDENCE": (
        "No candidate Strategy family represents this Mechanism on this instrument's asset domain today -- "
        "not structurally representable yet, not merely untried."
    ),
}


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active="learn")
    layout.render_header(
        subtitle="Cross-asset research relationships -- where a Mechanism has been researched, across which "
        "assets, and what remains underexplored."
    )
    render_body(services.approved_universe()[0])
    layout.render_disclaimer()


def render_body(root: str) -> None:
    """No page chrome -- factored out so `views/research.py`'s consolidated
    landing can render this exact content inline (same convention as
    `views/alpha_library.py::render_body`)."""
    selected = st.session_state.get(_SELECTED_KEY)
    if selected:
        _render_mechanism_detail(selected)
        return
    _render_landing()


# ---------------------------------------------------------------------------
# Landing -- every Mechanism + a flattened Research Gaps table
# ---------------------------------------------------------------------------


def _render_landing() -> None:
    components.section_header(
        "Research Map",
        "A navigation layer over existing Futures + ETF research memory -- never a second scientific "
        "database, never an Alpha Score. Cross-asset evidence is related evidence, never transferred proof: "
        "a verdict on one instrument never changes another's.",
    )
    rows = services.alpha_graph_summary()
    if not rows:
        components.empty_state(
            "Research Map", "No mechanism universe is mapped yet.", key="alpha-graph-empty",
        )
        return

    gaps = services.alpha_graph_research_gaps()
    cross_asset = [r for r in rows if r["is_cross_asset"]]
    total_researched = sum(r["n_researched"] + r["n_weakly_researched"] for r in rows)

    with components.metric_row("alpha-graph-metrics"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            components.metric_card("alpha-graph-mechanisms", "Mechanisms", str(len(rows)))
        with c2:
            components.metric_card("alpha-graph-cross-asset", "Cross-Asset Mechanisms", str(len(cross_asset)))
        with c3:
            components.metric_card("alpha-graph-researched", "Researched Instrument-Mechanisms", str(total_researched))
        with c4:
            components.metric_card("alpha-graph-gaps", "Research Gaps", str(len(gaps)))

    st.caption(
        "A \"Cross-Asset Mechanism\" has real registry evidence in more than one asset domain (e.g. "
        "Futures and ETF) -- research relationship, never a combined or transferred verdict."
    )

    for row in sorted(rows, key=lambda r: (-r["is_cross_asset"], r["mechanism"])):
        _render_mechanism_card(row)

    with st.expander(f"Research Gaps ({len(gaps)}) -- underexplored and unmapped Mechanism x Instrument pairs"):
        st.caption(
            "First-class research memory, never a failure: absence of evidence is not negative evidence "
            "(prompt 7 section 6)."
        )
        if not gaps:
            st.caption("No research gaps in the considered universe.")
        else:
            st.dataframe(
                [
                    {
                        "Mechanism": g["mechanism"], "Instrument": g["instrument"]["root_symbol"],
                        "Asset Domain": g["instrument"]["asset_domain"], "Coverage": g["coverage"],
                    }
                    for g in gaps
                ],
                width="stretch", hide_index=True, height=min(420, 44 + 35 * len(gaps)),
            )


def _render_mechanism_card(row: dict[str, Any]) -> None:
    with components.card(f"alpha-graph-mech-{row['mechanism']}"):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"**{row['mechanism']}**")
            domains = ", ".join(row["domains_with_evidence"]) or "none yet"
            st.caption(f"Researched in: {domains}")
            st.caption(
                f"Researched {row['n_researched']} · Weakly researched {row['n_weakly_researched']} · "
                f"Underexplored {row['n_underexplored']} · No evidence {row['n_no_evidence']}"
            )
        with c2:
            if row["is_cross_asset"]:
                components.render_badge("OK", label="CROSS-ASSET")
            if st.button("Open", key=f"alpha-graph-open-{row['mechanism']}", width="stretch"):
                st.session_state[_SELECTED_KEY] = row["mechanism"]
                st.rerun()


# ---------------------------------------------------------------------------
# Detail -- one Mechanism's full graph view
# ---------------------------------------------------------------------------


def _render_mechanism_detail(mechanism: str) -> None:
    if st.button("< Back to Research Map", key="alpha-graph-back"):
        st.session_state.pop(_SELECTED_KEY, None)
        st.rerun()

    view = services.alpha_graph_for_mechanism(mechanism)
    st.markdown(f"## {view['mechanism']}")

    components.section_header(
        "Where does this come from?",
        "Event Category -> Mechanism: the real, frozen news-category mapping this platform already uses to "
        "propose this Mechanism from a KIND of market event (never a live fetch, and never a specific "
        "historical occurrence -- a category such as PETROLEUM or FOMC_POLICY is a class of releases, not "
        "a concrete historical event on this page).",
    )
    if view["event_categories"]:
        with components.card("alpha-graph-events"):
            for ev in view["event_categories"]:
                st.markdown(f"**{ev['category']}**")
                st.caption(ev["evidence_basis"])
    else:
        st.caption("No Event Category maps to this Mechanism in the current mechanism library.")

    components.section_header(
        "Which Factors represent this Mechanism?",
        "One Factor per Strategy family this Mechanism structurally maps to today, in each asset domain.",
    )
    with components.card("alpha-graph-factors"):
        for f in view["factors"]:
            families = ", ".join(f["factor"]["related_strategy_families"])
            instruments = ", ".join(i["root_symbol"] for i in f["instruments_with_evidence"]) or "none yet"
            st.markdown(f"**{families}** ({f['asset_domain']})")
            st.caption(f"Instruments with real evidence: {instruments}")

    components.section_header(
        "Where has it been tested?",
        "Every considered instrument, grouped by real evidence coverage -- never a successful/failed boolean.",
    )
    by_coverage: dict[str, list[dict[str, Any]]] = {c: [] for c in _COVERAGE_ORDER}
    for e in view["instrument_evidence"]:
        by_coverage.setdefault(e["coverage"], []).append(e)

    for coverage in _COVERAGE_ORDER:
        group = by_coverage.get(coverage, [])
        st.markdown(f"#### {_COVERAGE_LABELS[coverage]} ({len(group)})")
        st.caption(_COVERAGE_DESCRIPTIONS[coverage])
        if not group:
            st.caption("None.")
            continue
        for e in sorted(group, key=lambda e: (e["instrument"]["asset_domain"], e["instrument"]["root_symbol"])):
            _render_instrument_row(e)

    _render_cross_asset_synthesis(mechanism)


def _render_instrument_row(e: dict[str, Any]) -> None:
    with components.card(f"alpha-graph-inst-{e['instrument']['asset_domain']}-{e['instrument']['root_symbol']}"):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"**{e['instrument']['root_symbol']}** ({e['instrument']['asset_domain']})")
            if e["scientific_evidence"]:
                verdicts = "; ".join(f"{fam}={v}" for fam, v in e["scientific_evidence"].items())
                st.caption(f"Own scientific verdict (not shared with any other instrument): {verdicts}")
            if e["repeated_failure_reason_codes"]:
                st.caption(
                    "Repeated reason codes: "
                    + ", ".join(f"{k} x{v}" for k, v in e["repeated_failure_reason_codes"].items())
                )
        with c2:
            components.render_badge(e["coverage"])
            for alpha_id in e["alpha_ids"]:
                if st.button("Open evidence", key=f"alpha-graph-goto-{alpha_id}", width="stretch"):
                    st.session_state["alpha_library_selected_id"] = alpha_id
                    st.session_state["learn_landing_focus"] = "alpha_library"
                    st.session_state.pop(_SELECTED_KEY, None)
                    st.rerun()


def _render_cross_asset_synthesis(mechanism: str) -> None:
    components.section_header(
        "Cross-Asset Synthesis",
        "What is common, what differs, what repeatedly failed, and what remains underexplored -- a "
        "deterministic summary, never a live model call, never a trade instruction.",
    )
    synth = services.alpha_graph_synthesis(mechanism)
    with components.card("alpha-graph-synthesis"):
        if synth["common"]:
            st.markdown("**What is common**")
            for line in synth["common"]:
                st.caption(f"- {line}")
        if synth["differs"]:
            st.markdown("**What differs**")
            for line in synth["differs"]:
                st.caption(f"- {line}")
        if synth["repeated_failures"]:
            st.markdown("**Repeated failures**")
            for line in synth["repeated_failures"]:
                st.caption(f"- {line}")
        if synth["underexplored"]:
            st.markdown("**Underexplored**")
            for line in synth["underexplored"]:
                st.caption(f"- {line}")
        st.caption(synth["note"])
