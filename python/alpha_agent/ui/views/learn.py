"""Page -- Learn: what the research taught us, what has been researched
before, and how the pieces work.

Four tabs: My Alpha (personal research memory -- research threads, recorded
transmission-route memory, and the factor/strategy library), Research Map
(what has actually been researched, on which assets, with what evidence --
never the economic Mechanism Graph, which lives in a thread's Reasoning
step), Community (replication and reproducibility, not a leaderboard) and
Guides (the Phase 4 explanation layer: Concepts, Strategies, Failure
autopsy, Learning paths). Every fact is read from committed registry evidence
or fixed, hand-authored explanation content -- never a second source of
scientific truth.

Deep links: `st.session_state["learn_landing_focus"]` selects a tab
(`alpha_library` / `alpha_graph` / `community` / `guides`) or a Guides
section (`concepts` / `strategies` / `my_research` / `paths` -- the keys
every "Learn why" hand-off already uses).
"""
from __future__ import annotations

import os

import streamlit as st

from alpha_agent.ui import (
    components,
    layout,
    learn_links,
    palette,
    research_loop_view,
    research_thread,
    services,
)

PAGE_TITLE = "Learn"

_TABS: dict[str, str] = {
    "alpha_library": "My Alpha", "alpha_graph": "Research Map", "community": "Community", "guides": "Guides",
}
_GUIDES: dict[str, str] = {
    "concepts": "Concepts", "strategies": "Strategies", "my_research": "Failure autopsy", "paths": "Learning paths",
}
_TABS_KEY = "learn-tabs"
_GUIDE_KEY = "learn-guide-section"


def _apply_focus() -> None:
    focus = st.session_state.pop("learn_landing_focus", None)
    if focus in _TABS:
        st.session_state[_TABS_KEY] = _TABS[focus]
    elif focus in _GUIDES:
        st.session_state[_TABS_KEY] = _TABS["guides"]
        st.session_state[_GUIDE_KEY] = focus


def render() -> None:
    layout.inject_style()
    _apply_focus()
    current = st.session_state.get(_TABS_KEY, _TABS["alpha_library"])
    active_sub = next((k for k, v in _TABS.items() if v == current), None)
    layout.render_sidebar_nav(active="learn", active_sub=active_sub)
    layout.render_header(subtitle="Your research memory, what has been researched before, and how it all works.")

    st.markdown('<div class="aa-page-title">Learn</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aa-page-lede">What your research has taught you, what has been researched before and with '
        "what evidence, independent replication -- and guides to the ideas behind every step.</div>",
        unsafe_allow_html=True,
    )
    root = services.approved_universe()[0]
    tabs = st.tabs(list(_TABS.values()), key=_TABS_KEY, on_change="rerun")
    renderers = (_render_my_alpha, lambda: _render_body("alpha_graph", root), lambda: _render_body("community", root),
                 _render_guides)
    for tab, renderer in zip(tabs, renderers, strict=True):
        with tab:
            if tab.open is not False:
                renderer()
    layout.render_disclaimer()


def _render_body(name: str, root: str) -> None:
    from alpha_agent.ui.views import alpha_graph, community

    {"alpha_graph": alpha_graph, "community": community}[name].render_body(root)


# ---------------------------------------------------------------------------
# My Alpha -- threads, recorded route memory, factor & strategy library
# ---------------------------------------------------------------------------


def _open_thread(thread_id: str) -> None:
    from alpha_agent.ui.views import workspace

    workspace.open_thread(thread_id)


def _render_my_alpha() -> None:
    from alpha_agent.ui.views import alpha_library
    from alpha_agent.ui.workspace.common import ago, esc

    root = services.approved_universe()[0]
    if st.session_state.get("alpha_library_selected_id"):
        alpha_library.render_body(root)
        return
    components.section_header(
        "Research threads",
        "Each thread follows one event from the news to the evidence. Open one to continue where you left off.",
    )
    threads = research_thread.THREAD_STORE.list()
    if not threads:
        st.caption("No research thread yet -- start one from News.")
    cols = st.columns(3)
    for i, t in enumerate(threads):
        with cols[i % 3], components.card(f"learn-thread-{t.thread_id}"):
            st.markdown(
                f'<div class="aa-eyebrow">{esc(research_thread.STEP_LABELS[t.current_step])} · updated '
                f"{esc(ago(t.updated_at))}</div>"
                f'<div class="aa-news-headline" style="font-size:0.95rem">{esc(t.name)}</div>'
                f'<div class="aa-news-summary">{esc(t.event.headline)}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Open thread", key=f"learn-open-thread-{t.thread_id}", icon=":material/arrow_forward:",
                         icon_position="right", type="tertiary"):
                _open_thread(t.thread_id)

    components.section_header(
        "Transmission routes remembered",
        "Every route a News Alpha research run recorded, from any event: how far its hypotheses got and why they "
        "stopped. A failure belongs to the route, market, measurement and signal it happened at -- nothing is "
        "blacklisted.",
    )
    try:
        memories = services.signal_path_memory_all()
    except Exception as exc:  # noqa: BLE001 -- a missing/older registry is an honest "not available"
        memories = []
        st.caption(f"Route memory is not available: {type(exc).__name__}.")
    if memories:
        st.dataframe(research_loop_view.memory_rows(memories), hide_index=True, width="stretch",
                     column_config={"What would change it": st.column_config.TextColumn(width="large")})
    else:
        st.caption("No route is recorded yet. Recording happens when a validation run records its evidence.")

    alpha_library.render_body(root)


# ---------------------------------------------------------------------------
# Guides -- the Phase 4 explanation layer
# ---------------------------------------------------------------------------


def _render_guides() -> None:
    section = st.segmented_control(
        "Guide", list(_GUIDES), default="concepts", format_func=_GUIDES.__getitem__, key=_GUIDE_KEY,
        label_visibility="collapsed",
    ) or "concepts"
    st.caption(
        "An explanation layer over the real research system, not a replacement for it. Every idea here connects "
        "to a real validation gate, a real strategy family, or your own committed experiments."
    )
    {"concepts": _render_concepts_tab, "strategies": _render_strategies_tab,
     "my_research": _render_my_research_tab, "paths": _render_paths_tab}[section]()


# ---------------------------------------------------------------------------
# Concepts -- Three-Lens explanation of each validation gate / market idea
# ---------------------------------------------------------------------------


def _render_concepts_tab() -> None:
    concepts = services.learn_concepts()
    ids = [c["concept_id"] for c in concepts]
    by_id = {c["concept_id"]: c for c in concepts}
    if st.session_state.get("learn_focus_concept") not in ids:
        st.session_state["learn_focus_concept"] = ids[0]

    category_filter = st.selectbox(
        "Category", ["All"] + list(dict.fromkeys(c["category"] for c in concepts)), key="learn-concept-category",
    )
    visible_ids = [i for i in ids if category_filter == "All" or by_id[i]["category"] == category_filter]
    if st.session_state["learn_focus_concept"] not in visible_ids:
        st.session_state["learn_focus_concept"] = visible_ids[0] if visible_ids else ids[0]

    selected_id = st.selectbox(
        "Concept", visible_ids or ids, format_func=lambda cid: by_id[cid]["title"], key="learn_focus_concept",
    )
    concept = by_id[selected_id]
    st.markdown(f"**{concept['one_line']}**")
    components.three_lens(
        f"learn-concept-{selected_id}",
        intuition=concept["intuition"], quant=concept["quant"], implementation=concept["implementation"],
        pointers=concept["pointers"],
    )


# ---------------------------------------------------------------------------
# Strategies -- economic idea -> equations -> feature -> StrategySpec -> C++
# ---------------------------------------------------------------------------


def _render_strategies_tab() -> None:
    explainers = services.learn_strategy_explainers()
    keys = [e["family_key"] for e in explainers]
    by_key = {e["family_key"]: e for e in explainers}
    if st.session_state.get("learn_focus_family") not in keys:
        st.session_state["learn_focus_family"] = keys[0]

    selected = st.selectbox(
        "Strategy family", keys, format_func=lambda k: by_key[k]["name"], key="learn_focus_family",
    )
    exp = by_key[selected]

    with components.card(f"learn-strategy-{selected}"):
        tabs = st.tabs(["Intuition", "Quant"])
        with tabs[0]:
            st.write(exp["intuition"])
        with tabs[1]:
            st.write(exp["quant"])

    st.markdown("#### Idea &rarr; Execution Chain", unsafe_allow_html=True)
    st.caption("The same closed pipeline every strategy family compiles and runs through -- only the formula above changes.")
    for i, step in enumerate(exp["chain"]):
        with components.card(f"learn-chain-{selected}-{i}"):
            st.markdown(f"**{step['stage']}**")
            st.write(step["text"])
            if step["pointers"]:
                st.caption("Code: " + " &middot; ".join(f"`{p}`" for p in step["pointers"]), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# My Research -- Failure Autopsy + quant-rigor pipeline walkthrough
# ---------------------------------------------------------------------------


#: Explicit verdict semantics (semantic hardening patch): a Failure Autopsy
#: is fundamentally about REJECT/INCONCLUSIVE trials -- a real scientific
#: outcome that failed or fell short of adjudication. PASS is a real,
#: different state (Validation's own page, not this one); NOT_ADJUDICATED
#: (a predeclared parameter neighbour, or a real typed refusal the frozen
#: policy deliberately never headline-adjudicates -- e.g. a Phase 15B ML
#: trial refused for insufficient training events) is neither PASS nor a
#: failure and must never be offered here as one. `verdict is None` (never
#: executed) is excluded the same way. This is a CLOSED allowlist, never
#: "verdict != PASS" (that silently let NOT_ADJUDICATED trials -- 40 real
#: ones in this registry today -- through as if they were failures).
_FAILURE_AUTOPSY_VERDICTS = ("REJECT", "INCONCLUSIVE")


def _render_my_research_tab() -> None:
    rows = services.list_experiments(trial_role=None)
    candidates = [r for r in rows if r["trial_role"] == "CANONICAL" and r["verdict"] in _FAILURE_AUTOPSY_VERDICTS]
    if not candidates:
        components.empty_state(
            "My Research",
            "No REJECT or INCONCLUSIVE canonical experiment exists in the registry yet -- run research first.",
            key="learn-my-research-empty",
        )
        return

    labels = {r["experiment_id"]: f"{r['root_symbol']} / {services.strategy_name(r['strategy_family'])} -- {r['verdict']}" for r in candidates}
    ids = list(labels)
    if st.session_state.get("learn_focus_experiment") not in ids:
        st.session_state["learn_focus_experiment"] = ids[0]

    selected_id = st.selectbox(
        "Experiment", ids, format_func=lambda i: labels[i], key="learn_focus_experiment",
    )

    autopsy = services.learn_failure_autopsy(selected_id)
    _render_failure_autopsy(autopsy)

    st.divider()
    st.markdown("#### Quant Rigor: Signal to Validation")
    st.caption("The same pipeline every trade in this experiment moved through, in order.")
    walkthrough = services.learn_pipeline_walkthrough(selected_id)
    _render_pipeline_walkthrough(walkthrough)


def _render_failure_autopsy(autopsy: dict) -> None:
    with components.metric_row("learn-autopsy-top"):
        c1, c2, c3 = st.columns(3)
        with c1:
            components.verdict_metric_card("learn-autopsy-verdict", "Verdict", autopsy["verdict"])
        with c2:
            components.metric_card("learn-autopsy-root", "Market", autopsy["root_symbol"])
        with c3:
            components.metric_card("learn-autopsy-family", "Strategy", services.strategy_name(autopsy["strategy_family"]))

    st.caption(autopsy["scope_note"])

    if autopsy["descriptive_evidence"] or autopsy["what_looked_promising"]:
        components.section_header("What Looked Promising")
        with components.card("learn-autopsy-promising"):
            if autopsy["descriptive_evidence"]:
                st.markdown("**DESCRIPTIVE EVIDENCE** &middot; raw backtest facts, *not validated alpha*", unsafe_allow_html=True)
                for stmt in autopsy["descriptive_evidence"]:
                    st.markdown(f"- {stmt}")
                st.caption(
                    "A later validation failure does not erase these facts -- it means they were not "
                    "enough to survive the gates below."
                )
            if autopsy["what_looked_promising"]:
                if autopsy["descriptive_evidence"]:
                    st.markdown("---")
                st.markdown("**Gates Explicitly Satisfied** &middot; validated by the frozen policy", unsafe_allow_html=True)
                for stmt in autopsy["what_looked_promising"]:
                    st.markdown(f"- {stmt}")

    components.section_header("What Failed, and Why It Matters")
    with components.card("learn-autopsy-failed"):
        st.write(autopsy["what_failed"])
        if autopsy["first_failed_gate_concept_id"]:
            concept = services.learn_concept(autopsy["first_failed_gate_concept_id"])
            if concept:
                st.caption(concept["one_line"])
            learn_links.render_learn_why(
                autopsy["first_failed_gate_concept_id"], key=f"learn-autopsy-{autopsy['experiment_id']}",
                label=f"Learn Why: {autopsy['first_failed_gate']}", already_on_learn_page=True,
            )

    if autopsy["similar_failures"]:
        components.section_header(
            "Other Experiments with Shared Failure Gates",
            "Same committed reason code(s) only -- not proven Factor, Strategy, or Mechanism similarity.",
        )
        with components.card("learn-autopsy-similar"):
            st.dataframe(
                [
                    {
                        "Experiment": s["experiment_id"], "Market": s["root_symbol"],
                        "Strategy": services.strategy_name(s["strategy_family"]), "Verdict": s["verdict"] or "--",
                        "Shared reason codes": ", ".join(s["shared_reason_codes"]),
                    }
                    for s in autopsy["similar_failures"]
                ],
                width="stretch", hide_index=True,
            )

    components.section_header("What a Next Experiment Would Need")
    with components.card("learn-autopsy-change"):
        st.write(autopsy["what_would_need_to_change"])

    if autopsy["engineering_notes"]:
        components.section_header(
            "System / Engineering History",
            "Historical data/execution lessons returned by FailureMemory. These notes are not necessarily "
            "causal or specific to this experiment -- each one's own scope is stated below.",
        )
        with components.card("learn-autopsy-engineering"):
            for note in autopsy["engineering_notes"]:
                if note["applies_to_this_experiment"]:
                    scope_text = "specific to this experiment"
                elif note["scope"] == "EXPERIMENT":
                    tag = note["root_symbol"] or note["strategy_family"] or "a different experiment"
                    scope_text = f"specific to a DIFFERENT experiment ({tag})"
                else:
                    scope_text = "system-wide"
                st.markdown(f"- **[{scope_text}]** {note['failure_code']}: {note['summary']}")

    _render_claude_narration_opt_in(autopsy)

    with st.expander("Details -- full gate table, reason codes, identity"):
        st.caption(f"Experiment identity: `{autopsy['experiment_identity']}`")
        if autopsy["reason_codes"]:
            st.markdown("**Reason codes:** " + ", ".join(f"`{c}`" for c in autopsy["reason_codes"]))
        st.dataframe(
            [{"Gate": g["label"], "State": g["state"], "Evidence": g["evidence"] or "--"} for g in autopsy["gates"]],
            width="stretch", hide_index=True,
        )


def _render_claude_narration_opt_in(autopsy: dict) -> None:
    """Optional, explicit opt-in Claude commentary (CLAUDE.md: "Claude
    narrates but never assigns verdict") -- never called by a default page
    render; see `test_phase4_learn_ui.py` for the static/monkeypatch guard
    proving this."""
    claude_configured = bool(os.environ.get("ANTHROPIC_API_KEY"))
    with st.expander("Explain this in plain language with Claude (optional)", expanded=False):
        if not claude_configured:
            st.caption("Claude API is not configured in this environment (see Settings -> Claude API).")
            return
        if st.button("Ask Claude to explain this result", key=f"learn-narrate-{autopsy['experiment_id']}"):
            from alpha_agent.agents.llm import AnthropicClient

            narrative = services.learn_failure_autopsy_narrative(autopsy["experiment_id"], client=AnthropicClient())
            if narrative:
                st.info(narrative)
            else:
                st.caption("Claude did not return a usable explanation for this result.")


def _render_pipeline_walkthrough(walkthrough: dict) -> None:
    if not walkthrough["has_real_example"]:
        st.caption("No committed, hash-verified trade ledger is bound to this experiment yet -- showing the "
                   "general pipeline only, honestly, with no fabricated example values.")
    for stage in walkthrough["stages"]:
        with components.card(f"learn-pipeline-{stage['stage_id']}"):
            top = st.columns([3, 1]) if stage["concept_id"] else (st.container(),)
            with top[0]:
                st.markdown(f"**{stage['title']}**")
                st.write(stage["text"])
                if stage["example"]:
                    st.markdown(f"<span style='color:{palette.GREEN}'>{stage['example']}</span>", unsafe_allow_html=True)
                if stage["pointers"]:
                    st.caption("Code: " + " &middot; ".join(f"`{p}`" for p in stage["pointers"]), unsafe_allow_html=True)
            if stage["concept_id"]:
                with top[1]:
                    learn_links.render_learn_why(
                        stage["concept_id"], key=f"learn-pipeline-link-{stage['stage_id']}", already_on_learn_page=True,
                    )


# ---------------------------------------------------------------------------
# Learning Paths
# ---------------------------------------------------------------------------


def _render_paths_tab() -> None:
    paths = services.learn_learning_paths()
    for path in paths:
        with components.card(f"learn-path-{path['path_id']}"):
            st.markdown(f"**{path['title']}**")
            st.caption(path["summary"])
            for i, step in enumerate(path["steps"], start=1):
                cols = st.columns([5, 2])
                with cols[0]:
                    title = _step_title(step)
                    st.markdown(f"{i}. {title} -- {step['why']}")
                with cols[1]:
                    if step["kind"] == "concept":
                        learn_links.render_learn_why(
                            step["ref_id"], key=f"learn-path-{path['path_id']}-{i}", label="Open",
                            already_on_learn_page=True,
                        )
                    elif st.button("Open", key=f"learn-path-strategy-{path['path_id']}-{i}"):
                        st.session_state["learn_landing_focus"] = "strategies"
                        st.session_state["learn_focus_family"] = step["ref_id"]
                        st.rerun()


def _step_title(step: dict) -> str:
    if step["kind"] == "concept":
        concept = services.learn_concept(step["ref_id"])
        return concept["title"] if concept else step["ref_id"]
    return services.strategy_name(step["ref_id"])
