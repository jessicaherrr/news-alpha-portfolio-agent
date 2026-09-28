"""My Alpha -- Personal Alpha Memory, Factor-first (Phase 2 origin,
evolved by the Phase 6 closure: "Discover Factor -> Test -> Remember -> Learn
-> Improve the next research idea"; renamed from "Factor Library" to "My
Alpha" in Phase 8's Research UI consolidation -- same tab/focus key
(`alpha_library`), same module, same behavior; a user-facing label change
only, distinguishing this PERSONAL research memory from the new Community
tab's SHARED research memory).

Lives INSIDE Research (a tab on `views/research.py`'s landing), never a new
top-level page. Purely a read surface over `alpha_agent.ui.services`'
wrappers, which are themselves a thin, typed, read-only projection of
`alpha_agent.alpha_memory` (Futures) + `alpha_agent.etf.alpha_memory_bridge`
(ETF) over the registry: opening or browsing any object here never runs a
backtest, calls Claude, calls a market-data API, or writes the registry
(section 18) -- there is no button on this page that does any of those
things.

"Factor" here means RESEARCH OBJECT, never proven profitable alpha (Phase 2
section 1): an object may carry an all-REJECT evidence profile and still be
first-class research memory. The landing groups Factors into three
EVIDENCE-ORIENTED sections (never a "successful/failed" boolean) --
VALIDATED / UNDER_RESEARCH / RESEARCH_ARCHIVE, see
`alpha_agent.alpha_memory.factor_status`. Research Maturity and Scientific
Verdict are rendered as two clearly separate things -- never combined into a
score.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from alpha_agent.ui import components, layout, learn_links, services

PAGE_TITLE = "My Alpha"

_SELECTED_KEY = "alpha_library_selected_id"
_MATURITY_ORDER = ("IDEA", "FORMALIZED", "TESTED", "REPLICATED", "ADJUDICATED")

_STATUS_ORDER = ("VALIDATED", "UNDER_RESEARCH", "RESEARCH_ARCHIVE")
_STATUS_LABELS = {
    "VALIDATED": "Validated", "UNDER_RESEARCH": "Under Research", "RESEARCH_ARCHIVE": "Research Archive",
}
_STATUS_DESCRIPTIONS = {
    "VALIDATED": (
        "At least one strategy implementation genuinely PASSed under the current, frozen "
        "validation policy -- never fabricated to make this section non-empty."
    ),
    "UNDER_RESEARCH": (
        "No PASS yet, but real open research remains: inconclusive, not yet adjudicated, "
        "insufficiently researched, or evidence disagrees across implementations."
    ),
    "RESEARCH_ARCHIVE": (
        "Every tested implementation so far came back REJECT. A real, informative negative "
        "result -- NOT a claim that the underlying economic idea is false."
    ),
}


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active="learn")
    layout.render_header(
        subtitle="Personal research memory -- factors, strategies, experiments, and repeated failure patterns."
    )
    render_body(services.approved_universe()[0])
    layout.render_disclaimer()


def render_body(root: str) -> None:
    """No page chrome -- factored out so `views/research.py`'s consolidated
    landing can render this exact content inline (same convention as
    `views/strategies.py::render_body`)."""
    selected = st.session_state.get(_SELECTED_KEY)
    if selected:
        _render_detail(selected)
        return
    _render_library(root)


# ---------------------------------------------------------------------------
# Library -- Factors / Strategies / Experiments / Rejections tabs
# ---------------------------------------------------------------------------


def _render_library(default_root: str) -> None:
    components.section_header(
        "My Alpha",
        "Discover Factor -> Test -> Remember -> Learn -> Improve the next research idea. Real, "
        "registry-grounded research memory, grouped by evidence -- never an Alpha Score, never a ranked pool.",
    )
    universe = services.approved_universe()
    objects = services.list_alpha_research_objects()
    if not objects:
        components.empty_state(
            "My Alpha",
            "No registry-grounded research memory exists yet -- run research on the Workflow tab or via Agent.",
            key="alpha-lib-empty",
        )
        return

    with components.metric_row("alpha-lib-metrics"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            components.metric_card("alpha-lib-total", "Factors", str(len(objects)))
        with c2:
            components.metric_card(
                "alpha-lib-roots", "Markets Covered", str(len({o["root_symbol"] for o in objects}))
            )
        with c3:
            components.metric_card(
                "alpha-lib-mechanisms", "Mechanisms", str(len({o["mechanism"] for o in objects}))
            )
        with c4:
            adjudicated = sum(1 for o in objects if o["research_maturity"] == "ADJUDICATED")
            components.metric_card("alpha-lib-adjudicated", "Adjudicated", str(adjudicated))

    domain_options = ["(any)", *sorted({o["asset_domain"] for o in objects})]
    root_options = ["(any)", *universe, *sorted({o["root_symbol"] for o in objects if o["root_symbol"] not in universe})]
    f1, f2, f3 = st.columns(3)
    domain_filter = f1.selectbox("Asset Domain", domain_options, key="alpha-lib-domain-filter")
    root_filter = f2.selectbox(
        "Market", root_options,
        index=root_options.index(default_root) if default_root in root_options else 0,
        key="alpha-lib-root-filter",
    )
    maturity_filter = f3.selectbox(
        "Research Maturity", ["(any)", *_MATURITY_ORDER], key="alpha-lib-maturity-filter",
    )

    rows = objects
    if domain_filter != "(any)":
        rows = [o for o in rows if o["asset_domain"] == domain_filter]
    if root_filter != "(any)":
        rows = [o for o in rows if o["root_symbol"] == root_filter]
    if maturity_filter != "(any)":
        rows = [o for o in rows if o["research_maturity"] == maturity_filter]

    _render_status_sections(rows)

    with st.expander("Browse raw evidence tables (Strategies / Experiments / Failure Patterns)"):
        tab_strategies, tab_experiments, tab_failures = st.tabs(
            ["Strategies", "Experiments", "Rejections / Failure Patterns"]
        )
        with tab_strategies:
            _render_strategies_tab(rows)
        with tab_experiments:
            _render_experiments_tab(rows)
        with tab_failures:
            _render_failures_tab(rows)


def _open(alpha_id: str) -> None:
    st.session_state[_SELECTED_KEY] = alpha_id
    st.rerun()


def _render_status_sections(objects: list[dict[str, Any]]) -> None:
    """My Alpha's primary landing structure (Phase 6 closure):
    three evidence-oriented sections, never a "successful/failed" boolean.
    `factor_status` is computed once per object by
    `alpha_agent.ui.services.list_alpha_research_objects`
    (`alpha_memory.factor_status.factor_status`), never recomputed here."""
    by_status: dict[str, list[dict[str, Any]]] = {s: [] for s in _STATUS_ORDER}
    for o in objects:
        by_status.setdefault(o["factor_status"], []).append(o)

    for status in _STATUS_ORDER:
        group = by_status.get(status, [])
        st.markdown(f"#### {_STATUS_LABELS[status]} ({len(group)})")
        st.caption(_STATUS_DESCRIPTIONS[status])
        if not group:
            if status == "VALIDATED":
                st.info(
                    "No Factor has a genuinely validated PASS yet under the current evidence "
                    "semantics -- shown honestly empty, never manufactured."
                )
            else:
                st.caption("None for the selected filters.")
            continue
        for o in group:
            _render_factor_card(o)


def _render_factor_card(o: dict[str, Any]) -> None:
    """One compact card. Only useful summary information -- name, asset/
    instrument, mechanism, status, related-experiment count, a compact
    evidence summary. NEVER raw JSON, a hash, or a fingerprint (those live
    behind "Advanced Details" on the detail page only)."""
    with components.card(f"alpha-lib-factor-{o['alpha_id'][:24]}"):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"**{o['mechanism']}** on **{o['root_symbol']}**")
            strat_names = ", ".join(v["strategy_display_name"] for v in o["strategy_variants"])
            st.caption(f"{o['evidence_profile']['related_experiment_label']} -- {strat_names}")
            st.caption(o["summary"])
        with c2:
            components.render_badge(o["asset_domain"])
            components.render_badge(o["research_maturity"])
            if st.button("Open", key=f"alpha-lib-open-{o['alpha_id']}", width="stretch"):
                _open(o["alpha_id"])


def _render_strategies_tab(objects: list[dict[str, Any]]) -> None:
    rows = [
        {
            "Market": o["root_symbol"],
            "Mechanism": o["mechanism"],
            "Strategy": v["strategy_display_name"],
            "Family": v["strategy_family"],
            "Experiments": v["n_experiments"],
            "Canonical Scientific Evidence": o["evidence_profile"]["scientific_evidence"].get(
                v["strategy_family"], "NO_CANONICAL_TRIAL"
            ),
            "Distinct StrategySpec Fingerprints": len(v["strategy_fingerprints"]),
        }
        for o in objects
        for v in o["strategy_variants"]
    ]
    if not rows:
        components.empty_state(
            "Strategies", "No strategy variants match the selected filters.", key="alpha-lib-strategies-empty",
        )
        return
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_experiments_tab(objects: list[dict[str, Any]]) -> None:
    rows = [
        {
            "Market": o["root_symbol"],
            "Mechanism": o["mechanism"],
            "Family": e["strategy_family"],
            "Experiment": e["experiment_id"],
            "Role": e["trial_role"],
            "Verdict": e["headline_verdict"] or "NOT_ADJUDICATED",
            "Valid Attempts": e["n_valid_attempts"],
            "Invalid Attempts": e["n_invalid_attempts"],
            "Window": e["market_window"],
        }
        for o in objects
        for e in o["experiments"]
    ]
    if not rows:
        components.empty_state(
            "Experiments", "No experiments match the selected filters.", key="alpha-lib-experiments-empty",
        )
        return
    st.dataframe(rows, width="stretch", hide_index=True, height=min(420, 44 + 35 * len(rows)))


def _render_failures_tab(objects: list[dict[str, Any]]) -> None:
    reason_totals: dict[str, int] = {}
    class_totals: dict[str, int] = {}
    lessons: list[str] = []
    for o in objects:
        for code, n in o["repeated_failure_reason_codes"].items():
            reason_totals[code] = reason_totals.get(code, 0) + n
        for cls, n in o["repeated_failure_classes"].items():
            class_totals[cls] = class_totals.get(cls, 0) + n
        for lesson in o["engineering_lessons"]:
            if lesson not in lessons:
                lessons.append(lesson)

    if not reason_totals and not class_totals and not lessons:
        components.empty_state(
            "Rejections / Failure Patterns",
            "No repeated-failure evidence for the selected filters.",
            key="alpha-lib-failures-empty",
        )
        return

    st.caption(
        "Structured reason codes and failure classes, preserved verbatim -- never summarized as "
        "\"strategy didn't work\"."
    )
    c1, c2 = st.columns(2)
    with c1, components.card("alpha-lib-reason-codes"):
        st.markdown("**Repeated Reason Codes**")
        if reason_totals:
            st.dataframe(
                [{"Reason Code": k, "Count": v} for k, v in sorted(reason_totals.items(), key=lambda kv: -kv[1])],
                width="stretch", hide_index=True,
            )
        else:
            st.caption("None recorded.")
    with c2, components.card("alpha-lib-failure-classes"):
        st.markdown("**Repeated Failure Classes**")
        if class_totals:
            st.dataframe(
                [{"Failure Class": k, "Count": v} for k, v in sorted(class_totals.items(), key=lambda kv: -kv[1])],
                width="stretch", hide_index=True,
            )
        else:
            st.caption("None recorded.")
    if lessons:
        with components.card("alpha-lib-engineering-lessons"):
            st.markdown("**Engineering Lessons** (root-scoped, apply regardless of strategy family)")
            for lesson in lessons:
                st.caption(f"- {lesson}")


# ---------------------------------------------------------------------------
# Detail -- one AlphaResearchObject
# ---------------------------------------------------------------------------


def _render_detail(alpha_id: str) -> None:
    if st.button("< Back to My Alpha", key="alpha-lib-back"):
        st.session_state.pop(_SELECTED_KEY, None)
        st.rerun()

    obj = services.get_alpha_research_object(alpha_id)
    if obj is None:
        components.empty_state(
            "My Alpha", "This research object is no longer available.", key="alpha-lib-detail-missing",
        )
        return

    st.markdown(f"## {obj['mechanism']} on {obj['root_symbol']}")
    components.render_badge(obj["asset_domain"])
    components.render_badge(obj["factor_status"])
    st.caption(_STATUS_DESCRIPTIONS[obj["factor_status"]])
    st.caption(obj["summary"])

    with components.metric_row("alpha-lib-detail-metrics"):
        c1, c2, c3 = st.columns(3)
        with c1:
            components.metric_card("alpha-lib-detail-maturity", "Research Maturity", obj["research_maturity"])
        with c2:
            components.metric_card(
                "alpha-lib-detail-related-experiments", "Related Experiments",
                obj["evidence_profile"]["related_experiment_label"],
            )
        with c3:
            components.metric_card("alpha-lib-detail-variants", "Strategy Variants", str(len(obj["strategy_variants"])))

    components.section_header(
        "What it is, and why it may work",
        "Mechanism and Factor -- an economic concept, not an implementation.",
    )
    with components.card("alpha-lib-detail-idea"):
        components.provenance_row("Mechanism", obj["mechanism"])
        st.caption(obj["mechanism_provenance"])
        st.caption(obj["factor"]["provenance_note"])

    components.section_header(
        "Factor definition",
        "What data this Factor needs, and how each strategy implementation currently measures it.",
    )
    with components.card("alpha-lib-detail-definition"):
        st.caption(
            "Required data shape (informational -- not part of Factor identity): "
            + (", ".join(obj["factor"]["structural_signature"]) or "--")
        )
        for v in obj["strategy_variants"]:
            if v["canonical_params"]:
                st.markdown(f"**{v['strategy_display_name']}** canonical parameters")
                st.json(v["canonical_params"])

    components.section_header(
        "Relevant context",
        "Live market-context matching for the current market state.",
    )
    st.caption(
        "Not duplicated here -- this page stays a pure, offline read over registry evidence "
        "(browsing a Factor never calls a market-data API). See the Agent page's "
        "Context-Aware Research Ranking for how this Factor ranks against the CURRENT market context."
    )

    components.section_header(
        "How was it implemented?", "Strategy variants -- same factor, different trading strategies.",
    )
    for v in obj["strategy_variants"]:
        with components.card(f"alpha-lib-detail-variant-{v['strategy_family']}"):
            st.markdown(f"**{v['strategy_display_name']}** (`{v['strategy_family']}`)")
            c1, c2 = st.columns(2)
            with c1:
                st.caption(
                    f"{v['n_experiments']} experiment(s) -- "
                    f"{len(v['strategy_fingerprints'])} distinct StrategySpec fingerprint(s)"
                )
            with c2:
                st.caption(
                    "Canonical scientific evidence: "
                    + obj["evidence_profile"]["scientific_evidence"].get(v["strategy_family"], "NO_CANONICAL_TRIAL")
                )
            if v["canonical_params"]:
                st.json(v["canonical_params"])
            digest = v["digest"]
            st.caption(
                f"Execution attempts: {digest['valid_execution_attempts']} valid / "
                f"{digest['invalid_execution_attempts']} invalid. "
                f"Scientific verdicts: {digest['scientific_verdict_counts'] or 'none yet'}"
            )

    components.section_header("What did I test?", "Every real registry experiment behind this research memory.")
    with components.card("alpha-lib-detail-experiments"):
        st.dataframe(
            [
                {
                    "Experiment": e["experiment_id"], "Family": e["strategy_family"], "Role": e["trial_role"],
                    "Verdict": e["headline_verdict"] or "NOT_ADJUDICATED", "Window": e["market_window"],
                    "Valid Attempts": e["n_valid_attempts"], "Invalid Attempts": e["n_invalid_attempts"],
                    "Holdout Eligible": e["holdout_eligible"],
                }
                for e in obj["experiments"]
            ],
            width="stretch", hide_index=True, height=min(360, 44 + 35 * len(obj["experiments"])),
        )

    components.section_header(
        "What did the evidence say?", "Evidence Profile -- categorical and factual, never a combined score.",
    )
    ep = obj["evidence_profile"]
    with components.metric_row("alpha-lib-detail-evidence"):
        c1, c2, c3 = st.columns(3)
        with c1:
            components.metric_card("alpha-lib-ep-cost", "Cost Robustness", ep["cost_robustness"])
        with c2:
            components.metric_card("alpha-lib-ep-param", "Parameter Stability", ep["parameter_stability"])
        with c3:
            components.metric_card("alpha-lib-ep-regime", "Regime Breadth", ep["regime_breadth"])
    with components.card("alpha-lib-ep-scientific"):
        st.markdown("**Scientific Evidence** (per strategy variant's canonical trial)")
        for family, verdict in ep["scientific_evidence"].items():
            components.status_row(family, verdict)

    _render_promising_and_failed(obj)

    components.section_header(
        "What have I learned?", "Repeated failure reasons -- never summarized as \"strategy didn't work\".",
    )
    with components.card("alpha-lib-detail-failures"):
        if obj["repeated_failure_reason_codes"]:
            st.markdown("**Repeated reason codes**")
            for code, n in sorted(obj["repeated_failure_reason_codes"].items(), key=lambda kv: -kv[1]):
                st.caption(f"`{code}` x {n}")
        if obj["repeated_failure_classes"]:
            st.markdown("**Repeated failure classes**")
            for cls, n in sorted(obj["repeated_failure_classes"].items(), key=lambda kv: -kv[1]):
                st.caption(f"`{cls}` x {n}")
        if not obj["repeated_failure_reason_codes"] and not obj["repeated_failure_classes"]:
            st.caption("No repeated failure evidence recorded for this research object.")
        for lesson in obj["engineering_lessons"]:
            st.info(lesson)

    if obj["related_alpha_ids"]:
        components.section_header(
            "Related Research Memory",
            "Other factors on this market whose mapped strategy families overlap this one -- same "
            "implementation, different economic framing. Never a similarity score, never a merge.",
        )
        with components.card("alpha-lib-detail-related"):
            for rid in obj["related_alpha_ids"]:
                related = services.get_alpha_research_object(rid)
                if related is None:
                    continue
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(
                        f"**{related['mechanism']}** on {related['root_symbol']} -- {related['research_maturity']}"
                    )
                with c2:
                    if st.button("Open", key=f"alpha-lib-related-{rid}", width="stretch"):
                        _open(rid)

    components.section_header("Learn Why", "Concept explanations behind this Factor's validation gates.")
    for concept_id in ("look_ahead_bias", "bh_fdr", "dsr"):
        learn_links.render_learn_why(concept_id, key=f"alpha-lib-detail-{alpha_id}-{concept_id}")

    with st.expander("Advanced Details -- provenance, identities, fingerprints, C++ execution evidence"):
        st.markdown("**Factor identity**")
        components.provenance_row("Factor identity hash", obj["factor"]["factor_identity"])
        components.provenance_row("Related strategy families", ", ".join(obj["factor"]["related_strategy_families"]))
        components.provenance_row("Schema version", obj["schema_version"])
        st.markdown("**Experiment identities**")
        st.dataframe(
            [
                {"Experiment ID": e["experiment_id"], "experiment_identity": e["experiment_identity"]}
                for e in obj["experiments"]
            ],
            width="stretch", hide_index=True,
        )
        st.markdown("**StrategySpec fingerprints per implementation**")
        for v in obj["strategy_variants"]:
            st.caption(f"{v['strategy_display_name']} StrategySpec fingerprint(s): " + ", ".join(v["strategy_fingerprints"]))
        st.caption(
            "Official PnL, fills, and C++ execution evidence for any experiment above live in Research -> "
            "Experiments / Validation (the registry's own detail view), never duplicated into this page."
        )


def _render_promising_and_failed(obj: dict[str, Any]) -> None:
    """Reuses Phase 4's real Failure Autopsy (`alpha_agent.learn.autopsy`) for
    every canonical experiment behind this Factor whose verdict is REJECT or
    INCONCLUSIVE -- never a second, parallel "why it failed" computation. A
    rejected StrategySpec does not invalidate the broader Factor: this
    section is per-EXPERIMENT, clearly labelled by which strategy variant it
    belongs to."""
    canonical_negative = [
        e for e in obj["experiments"]
        if e["trial_role"] == "CANONICAL" and e["headline_verdict"] in ("REJECT", "INCONCLUSIVE")
    ]
    if not canonical_negative:
        return
    components.section_header(
        "What looked promising, and what failed",
        "Reused directly from Learn's Failure Autopsy (Phase 4) -- descriptive evidence and gates explicitly "
        "satisfied are shown separately from what actually failed; never merged.",
    )
    for e in canonical_negative:
        autopsy = services.learn_failure_autopsy(e["experiment_id"])
        with components.card(f"alpha-lib-autopsy-{e['experiment_id']}"):
            st.markdown(f"**{e['strategy_family']}** -- `{e['experiment_id']}` ({e['headline_verdict']})")
            if autopsy["what_looked_promising"]:
                st.markdown("_What looked promising (gates explicitly satisfied):_")
                for item in autopsy["what_looked_promising"]:
                    st.caption(f"+ {item}")
            if autopsy["descriptive_evidence"]:
                st.markdown("_Descriptive evidence -- not validated alpha:_")
                for item in autopsy["descriptive_evidence"]:
                    st.caption(f"- {item}")
            if autopsy["what_failed"]:
                st.markdown("_What failed:_")
                st.caption(autopsy["what_failed"])
            st.caption(autopsy["scope_note"])
