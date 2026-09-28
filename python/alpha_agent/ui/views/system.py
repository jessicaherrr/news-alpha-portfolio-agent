"""Page 8 -- Settings. Sidebar IA pass (Release UX product-consolidation
acceptance pass, task spec section 11): the home for engineering/
configuration detail -- Data Providers / Runtime / Claude API / About --
every value read from real git / catalog / registry state, never a
fabricated "connected" indicator. This is also where the global header's
former five backend-engineering pills, git commit, and build timestamp now
live (task spec section 10A/10B), plus Agent Runtime status, Recent
Activity, and the Labs entry point that were already here.
"""
from __future__ import annotations

import os

import streamlit as st

from alpha_agent.ui import components, layout, panels, services

PAGE_TITLE = "Settings"

#: Professional, functional component names for the normal Settings surface
#: (Agent Evidence + Research Provenance acceptance pass, task spec section
#: 7A) -- no development-phase numbers. `_DEVELOPER_ARCHITECTURE_STEPS` below
#: pairs each of these with the internal name it replaces, for the
#: "Developer Details" expander (section 7B) -- raw phase/class names never
#: appear on the normal surface, only there.
_ARCHITECTURE_STEPS = (
    "Market Data", "Feature Engine", "Research Planner", "Hypothesis Spec",
    "Strategy Compiler", "Strategy Spec", "Quant Core", "Validation Engine",
    "Experiment Registry",
)

#: Internal/implementation names for the SAME pipeline stages, in the same
#: order -- shown only under "Developer Details" (section 7B/7C), never on
#: the normal Architecture surface.
_DEVELOPER_ARCHITECTURE_STEPS = (
    "Data (Databento GLBX.MDP3)", "Features (alpha_agent.features)",
    "Research Agent (Phase 16)", "HypothesisSpec", "Strategy Compiler (Phase 17)",
    "StrategySpec", "C++ Quant Core (backtest / fill / risk)", "Validation (Phase 13)",
    "Experiment Registry (Phase 14)",
)


def render() -> None:
    layout.inject_style()
    focus = st.session_state.get("settings_landing_focus")
    layout.render_sidebar_nav(active="settings", active_sub=focus)
    layout.render_header(subtitle="Runtime, data, and research configuration.")

    st.markdown("## Settings")
    st.caption(
        "Infrastructure, data, research-safety, and provenance -- read from real state, never simulated. "
        "Developer surfaces and labs live here too, out of the primary navigation."
    )

    sections: dict[str, tuple[str, object]] = {
        "providers": ("Data Providers", _render_data_providers_tab),
        "runtime": ("Runtime", _render_runtime_tab),
        "claude": ("Claude API", _render_claude_tab),
        "about": ("About", _render_about_tab),
    }
    # Sidebar IA pass (task spec section 7E): Settings -> {Data Providers,
    # Runtime, Claude API, About} sub-rows deep-link here the same way
    # Paper's own sub-rows do (see `views/paper_trading.py`'s identical
    # pattern) -- `st.tabs` cannot be preselected programmatically, so a
    # focused view renders that ONE section directly; the unfocused default
    # renders every section as tabs (all bodies present in the DOM, which is
    # what the existing "2025 STATUS" / "Agent Runtime" regression tests key
    # off of regardless of which tab happens to be visually active).
    if focus in sections:
        label, renderer = sections[focus]
        if st.button("< Back to all Settings", key="sys-focus-back"):
            st.session_state.pop("settings_landing_focus", None)
            st.rerun()
        st.markdown(f'<div class="aa-gate-title">{label}</div>', unsafe_allow_html=True)
        renderer()
    else:
        tabs = st.tabs([label for label, _ in sections.values()])
        for tab, (_label, renderer) in zip(tabs, sections.values(), strict=True):
            with tab:
                renderer()

    panels.render_agent_runtime_status()

    # Product refactor (section 12): Agent Status / Recent Activity moved here
    # from Research -- these are infrastructure/session-runtime concerns, not
    # part of a research artifact's own scientific evidence.
    panels.render_activity_feed()

    components.section_header(
        "Labs",
        "Experimental / non-primary research surfaces -- not part of the certified Research Universe "
        "and never part of the real BH/FDR statistical family.",
    )
    with components.card("sys-labs"):
        st.markdown("**Crypto Lab**")
        st.caption(
            "Synthetic-scaffold-only BTC/ETH derivatives research (Phase 22) -- real C++ Quant Core and "
            "validation over SYNTHETIC market data, never a real ExperimentRegistry entry."
        )
        if st.button("Open Crypto Lab", key="sys-open-crypto-lab"):
            from alpha_agent.ui.views import crypto_lab

            st.switch_page(st.Page(crypto_lab.render, url_path="crypto-lab"))

    layout.render_disclaimer()


# ---------------------------------------------------------------------------
# Data Providers (task spec section 11A) -- existing real diagnostics only,
# never a new provider.
# ---------------------------------------------------------------------------


def _render_data_providers_tab() -> None:
    components.section_header("Data")
    catalog = services.catalog_summary()
    with components.card("sys-data"):
        with components.metric_row("sys-data"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Dataset", "Databento GLBX.MDP3")
            c2.metric("Roots", ", ".join(catalog["roots"]))
            c3.metric("Catalog entries", catalog["n_rows"])
        st.dataframe(
            [
                {"Root": root_sym, "Entries": info["n_entries"], "Components": ", ".join(info["components"]),
                 "Schemas": ", ".join(info["schemas"]), "Start": info["start"], "End (exclusive)": info["end"],
                 "Rows": info["row_count"]}
                for root_sym, info in catalog["by_root"].items()
            ],
            width="stretch", hide_index=True,
        )
        st.caption("An end date is an EXCLUSIVE boundary -- e.g. `2025-01-01` means data through 2024-12-31, not 2025 access.")

    components.section_header(
        "News & Events Sources",
        "Real official market-news and economic-event connectors backing the Market page's News & Events "
        "tab and Agent's catalyst evidence -- never a fabricated CONNECTED state.",
    )
    with components.card("sys-news-sources"):
        from alpha_agent.marketdata.capability import CapabilityState
        from alpha_agent.ui import market_intel_context

        news_health = market_intel_context.connector_health()
        event_health = market_intel_context.event_connector_health()
        for name, state in sorted(news_health.items()):
            components.status_row(f"{name} (news)", "OK" if state is CapabilityState.AVAILABLE else "OFFLINE", state.value)
        for name, state in sorted(event_health.items()):
            components.status_row(f"{name} (events)", "OK" if state is CapabilityState.AVAILABLE else "OFFLINE", state.value)


# ---------------------------------------------------------------------------
# Runtime (task spec section 11B) -- existing real states only, never a new
# runtime health score. Includes the deeper research-safety/holdout-
# lifecycle audit detail (distinct from Research -> Research Scope's simple,
# canonical Research/Validation/Holdout-LOCKED summary -- task spec section
# 10A: "do not duplicate Holdout status globally" refers to that simple
# summary, which now lives ONLY on Research; this is separate engineering-
# audit detail that never lived in the shared sidebar to begin with).
# ---------------------------------------------------------------------------


def _render_runtime_tab() -> None:
    panels.render_system_status()

    components.section_header("Research Safety")
    with components.card("sys-safety"):
        w = services.research_window()
        holdout = services.holdout_status()
        with components.metric_row("sys-safety"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Research Window", w["research"])
            c2.metric("Validation Window", w["validation"])
            c3.metric("Holdout", "2025 -- LOCKED")
        st.caption(holdout["declaration"])
        lifecycle = services.holdout_lifecycle_status("2025")
        st.caption(
            f"2025 STATUS: {lifecycle['status']} -- 2025 ACCESSED: "
            f"{'YES' if lifecycle['accessed'] else 'NO'} "
            "(Alpha Discovery Part L lifecycle state; SEALED/NO is the honest default until an "
            "explicit, separately authorized final evaluation is ever run)."
        )
        reg = services.registry_summary()
        st.write(f"Reliability policy / multiple-testing family: BH-FDR across {reg['authoritative_statistical_hypotheses']} "
                 f"authoritative hypotheses; identity schema `{reg['identity_schema']}`.")

    components.section_header("Registry")
    with components.card("sys-registry"):
        components.provenance_row("Registry schema version", str(reg["schema_version"]))
        components.provenance_row("Registry content digest", reg["content_digest"])


# ---------------------------------------------------------------------------
# Claude API (task spec section 11C) -- Configured / Connected only. Never
# the API key, model/temperature UI, raw secrets, or a token dashboard.
# ---------------------------------------------------------------------------


def _render_claude_tab() -> None:
    claude_configured = bool(os.environ.get("ANTHROPIC_API_KEY"))
    with components.card("sys-claude-api"):
        components.status_row(
            "Claude API", "OK" if claude_configured else "OFFLINE",
            "Configured -- ANTHROPIC_API_KEY present in the environment (value never read/displayed by this app)."
            if claude_configured else "Not configured -- ANTHROPIC_API_KEY is not set; Claude Research mode will "
                                       "fail honestly rather than silently falling back.",
        )
        st.caption(
            "This app never displays the key value, a model/temperature picker, or a token-usage dashboard here."
        )


# ---------------------------------------------------------------------------
# About (task spec section 11D) -- real commit/branch/working-tree state
# only; no fabricated build timestamp (this app does not record one).
# ---------------------------------------------------------------------------


def _render_about_tab() -> None:
    git = services.git_provenance()
    engine = services.engine_provenance()
    with components.card("sys-about"):
        components.provenance_row("Git commit", git.get("commit"))
        components.provenance_row("Branch", git.get("branch"))
        components.provenance_row("Working tree", "dirty (uncommitted changes)" if git.get("dirty") else "clean")
        components.provenance_row("Reference C++ CLI", engine.get("reference_cli_path") or "not built")

    components.section_header("Architecture")
    with components.card("sys-architecture"):
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:0.4rem;align-items:center;margin-bottom:0.6rem;">'
            + " <span style='color:#5b6480'>&rarr;</span> ".join(
                f'<span class="aa-tag">{s}</span>' for s in _ARCHITECTURE_STEPS
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        st.caption("The LLM only appears at Research Planner / Strategy Compiler; every other node is deterministic.")
        with st.expander("Developer Details", expanded=False):
            st.caption("Same pipeline, internal implementation names (development-phase numbers included):")
            st.markdown(
                '<div style="display:flex;flex-wrap:wrap;gap:0.4rem;align-items:center;">'
                + " <span style='color:#5b6480'>&rarr;</span> ".join(
                    f'<span class="aa-tag">{s}</span>' for s in _DEVELOPER_ARCHITECTURE_STEPS
                )
                + "</div>",
                unsafe_allow_html=True,
            )
