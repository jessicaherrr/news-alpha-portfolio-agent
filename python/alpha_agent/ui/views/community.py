"""Community -- the Phase 8 Community Alpha Network (prompt 8).

NOT a strategy marketplace: no copy trading, no raw-return leaderboard, no
"Like". Lives INSIDE Research (a tab on `views/research.py`'s landing),
never a new top-level page -- mirrors `views/alpha_graph.py`/
`views/alpha_library.py`'s own convention. A purely thin rendering shell
over `alpha_agent.ui.services`' Phase 8 wrappers, which are themselves thin,
typed projections of `alpha_agent.community` (its own module docstrings are
the full boundary: real registry-verified evidence only, no fake
provenance, no Alpha Score). Nothing on this page runs a backtest, calls
Claude, or writes the scientific registry -- every write lands in the
SEPARATE Community store.

There is no authentication anywhere in this local research tool: "who am I"
is a plain typed researcher name, remembered for convenience in
`st.session_state` for this browser session only -- never a real login, and
never presented as one (every contributor/replicator name renders with a
"(self-attested)" caption).

Two internal modes, the same landing/detail pattern `alpha_library.py` uses:

* LANDING -- three tabs: Community Feed (PUBLIC full detail + SHARED
  redacted-unless-you're-its-owner; PRIVATE never appears here for anyone),
  My Contributions (every visibility level, full detail, your own typed
  name), and Share a Contribution (cites an existing registry
  ``experiment_id`` -- never fabricates one).
* DETAIL -- one Contribution: its evidence, its full replication history
  (every dimension preserved distinctly, conflicting evidence shown
  plainly, never merged), its derived EvidenceMaturity, a Replicate form,
  and its contributor's Reputation profile.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from alpha_agent.ui import components, layout, services

PAGE_TITLE = "Community"

_SELECTED_KEY = "community_selected_contribution_id"
_NAME_KEY = "community_researcher_name"

_CONTRIBUTION_KINDS = (
    "HYPOTHESIS", "MECHANISM", "FACTOR_DEFINITION", "STRATEGY_SPEC", "EVIDENCE_BUNDLE", "RESEARCH_NOTES",
)
_VISIBILITY_LEVELS = ("PRIVATE", "SHARED", "PUBLIC")
_VISIBILITY_HELP = (
    "PRIVATE: only you, ever -- excluded from every feed and aggregate. SHARED: benefits community "
    "aggregates and is independently replicable, but your exact strategy parameters stay redacted from "
    "everyone but you. PUBLIC: fully visible, parameters included."
)

_COMPARISON_DIMENSIONS = (
    ("Root Symbol", "same_root_symbol"), ("Asset Domain", "same_asset_domain"),
    ("Strategy Family", "same_strategy_family"), ("Factor Identity", "same_factor_identity"),
    ("Market Window", "same_market_window"), ("Reliability Policy", "same_reliability_policy"),
    ("Dataset", "same_dataset_fingerprint"),
)


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active="learn")
    layout.render_header(subtitle="Reproducible research and independent replication -- not a marketplace.")
    render_body(services.approved_universe()[0])
    layout.render_disclaimer()


def render_body(root: str) -> None:
    """No page chrome -- factored out so `views/research.py`'s consolidated
    landing can render this exact content inline (same convention as
    `views/alpha_library.py::render_body`)."""
    selected = st.session_state.get(_SELECTED_KEY)
    if selected:
        _render_detail(selected)
        return
    _render_landing(root)


def _researcher_name() -> str:
    return st.session_state.get(_NAME_KEY, "")


def _open(contribution_id: str) -> None:
    st.session_state[_SELECTED_KEY] = contribution_id
    st.rerun()


def _render_identity_bar() -> None:
    with components.card("community-identity"):
        st.markdown('<div class="aa-gate-title">Your researcher identity</div>', unsafe_allow_html=True)
        st.caption(
            "Self-attested only -- there is no authentication in this local research tool. Two "
            "contributions typed under the same name resolve to the same contributor."
        )
        name = st.text_input("Researcher name", value=_researcher_name(), key="community-name-input")
        st.session_state[_NAME_KEY] = name


# ---------------------------------------------------------------------------
# Landing -- Community Feed / My Contributions / Share a Contribution
# ---------------------------------------------------------------------------


def _render_landing(default_root: str) -> None:
    components.section_header(
        "Community Alpha Network",
        "Reproducible research and independent replication, not a strategy marketplace. The core "
        "interaction is Replicate, not Like -- there is no raw-return leaderboard here.",
    )
    _render_identity_bar()
    name = _researcher_name()

    tab_feed, tab_mine, tab_share = st.tabs(["Community Feed", "My Contributions", "Share a Contribution"])
    with tab_feed:
        _render_feed(name)
    with tab_mine:
        _render_mine(name)
    with tab_share:
        _render_share_form(name, default_root)


def _render_feed(viewer_name: str) -> None:
    rows = services.community_feed(viewer_name or None)
    if not rows:
        components.empty_state(
            "Community Feed", "No SHARED or PUBLIC contributions exist yet.", key="community-feed-empty",
        )
        return
    with components.metric_row("community-feed-metrics"):
        c1, c2, c3 = st.columns(3)
        with c1:
            components.metric_card("community-feed-n", "Contributions", str(len(rows)))
        with c2:
            components.metric_card(
                "community-feed-contributors", "Contributors",
                str(len({r["contributor"]["contributor_id"] for r in rows})),
            )
        with c3:
            components.metric_card(
                "community-feed-public", "Public", str(sum(1 for r in rows if r["visibility"] == "PUBLIC")),
            )
    for r in rows:
        _render_contribution_card(r)


def _render_mine(name: str) -> None:
    if not name.strip():
        st.caption("Type a researcher name above to see your own contributions, including PRIVATE ones.")
        return
    rows = services.community_my_contributions(name)
    if not rows:
        components.empty_state(
            "My Contributions", "No contributions yet -- share one on the \"Share a Contribution\" tab.",
            key="community-mine-empty",
        )
        return
    for r in rows:
        _render_contribution_card(r)
    components.section_header("Your Reputation")
    _render_reputation(services.community_reputation(name))


def _render_contribution_card(c: dict[str, Any]) -> None:
    """One compact card -- name, market, mechanism, verdict, visibility.
    Never raw JSON/fingerprints (those live on the detail page's evidence
    section, and only the fields a caller is entitled to see are ever
    non-``None`` -- see `alpha_agent.community.visibility`)."""
    ev = c["evidence"]
    with components.card(f"community-card-{c['contribution_id'][:24]}"):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{c['title']}**")
            mech = f" -- {c['mechanism']}" if c.get("mechanism") else ""
            st.caption(f"{c['kind']} -- {c['root_symbol']}{mech}")
            st.caption(f"Contributor: {c['contributor']['display_name']} (self-attested)")
            if c.get("notes"):
                st.caption(c["notes"])
            window = f"{ev['market_window_label']} {ev['market_window_start']}..{ev['market_window_end']}"
            st.caption(f"Evidence: {ev['strategy_family']} on {ev['root_symbol']} ({window})")
            if ev["strategy_params"] is None and c["visibility"] == "SHARED":
                st.caption(
                    "Strategy parameters redacted (SHARED visibility) -- benefits community aggregates "
                    "without revealing exact logic."
                )
        with col2:
            components.render_badge(c["visibility"])
            components.render_badge(ev["headline_verdict"] or "NOT_ADJUDICATED")
            if c["recycling_decision"] == "DEPRIORITIZE_NO_NOVELTY":
                st.caption("possible duplicate")
            if st.button("Open", key=f"community-open-{c['contribution_id']}", width="stretch"):
                _open(c["contribution_id"])


def _render_share_form(name: str, default_root: str) -> None:
    st.caption(
        "A contribution must cite an existing, already-committed registry experiment (its CANONICAL "
        "trial) -- this page never runs a backtest, calls Claude, or fabricates evidence."
    )
    mechanism_options = ["(none)", *services.alpha_graph_mechanisms()]
    kind = st.selectbox("Kind", _CONTRIBUTION_KINDS, key="community-share-kind")
    vis = st.selectbox("Visibility", _VISIBILITY_LEVELS, key="community-share-visibility", help=_VISIBILITY_HELP)
    experiment_id = st.text_input(
        "Registry Experiment ID (its CANONICAL trial)", key="community-share-expid",
        help="e.g. an id shown on Research -> Experiments, or a My Alpha detail view.",
    )
    title = st.text_input("Title", key="community-share-title")
    mechanism = st.selectbox("Economic Mechanism (optional)", mechanism_options, key="community-share-mechanism")
    notes = st.text_area("Notes", key="community-share-notes", height=90)
    novelty = st.text_area(
        "Novelty notes (optional)", key="community-share-novelty", height=68,
        help="What is different this time vs. a prior community contribution, if anything.",
    )

    if not st.button("Share", type="primary", key="community-share-submit"):
        return
    if not name.strip():
        st.error("Type a researcher name above first.")
        return
    if not experiment_id.strip() or not title.strip():
        st.error("An experiment ID and a title are both required.")
        return
    try:
        c = services.community_create_contribution(
            kind=kind, visibility=vis, contributor_display_name=name, experiment_id=experiment_id.strip(),
            title=title, notes=notes, mechanism=None if mechanism == "(none)" else mechanism,
            novelty_notes=novelty,
        )
    except Exception as exc:  # noqa: BLE001 -- an honest refusal (unknown id, cherry-picked trial, holdout), never a crash
        st.error(f"Could not create contribution: {exc}")
        return
    st.success(f"Shared as {c['contribution_id']}. {c['recycling_decision']}: {c['recycling_explanation']}")
    _open(c["contribution_id"])


def _render_reputation(rep: dict[str, Any]) -> None:
    with components.metric_row("community-reputation"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            components.metric_card("community-rep-contrib", "Contributions", str(rep["n_contributions"]))
        with c2:
            components.metric_card(
                "community-rep-performed", "Replications Performed", str(rep["n_replications_performed"]),
            )
        with c3:
            components.metric_card(
                "community-rep-received", "Times Replicated", str(rep["n_times_own_work_replicated"]),
            )
        with c4:
            components.metric_card(
                "community-rep-confirm", "Confirming / Conflicting",
                f"{rep['n_confirming_replications_received']} / {rep['n_conflicting_replications_received']}",
            )
    st.caption(rep["reproducibility_note"])
    st.caption(
        "Never a single blended score -- reputation emphasizes reproducibility, provenance, and "
        "independent replication. A REJECT or INCONCLUSIVE contribution counts toward Contributions "
        "exactly the same as a PASS one."
    )


# ---------------------------------------------------------------------------
# Detail -- one Contribution
# ---------------------------------------------------------------------------


def _render_detail(contribution_id: str) -> None:
    if st.button("< Back to Community", key="community-back"):
        st.session_state.pop(_SELECTED_KEY, None)
        st.rerun()

    state = services.community_evidence_state(contribution_id)
    if state is None:
        components.empty_state(
            "Community", "This contribution is no longer available.", key="community-detail-missing",
        )
        return
    c, reps = state["contribution"], state["replications"]

    st.markdown(f"## {c['title']}")
    components.render_badge(c["visibility"])
    components.render_badge(state["evidence_maturity"])
    st.caption(f"Contributed by {c['contributor']['display_name']} (self-attested) -- {c['kind']}")
    if c.get("notes"):
        st.caption(c["notes"])
    st.caption(state["maturity_rationale"])
    if c["recycling_decision"] == "DEPRIORITIZE_NO_NOVELTY":
        st.warning(c["recycling_explanation"])
    else:
        st.caption(c["recycling_explanation"])

    _render_evidence_section(c["evidence"])

    components.section_header(
        "Replications",
        "Replicate, not Like -- every independent replication preserves its own contributor and evidence; "
        "nothing here is ever merged into the original.",
    )
    if not reps:
        components.empty_state(
            "Replications", "No independent replication yet -- be the first to replicate this.",
            key="community-detail-no-reps",
        )
    else:
        for r in reps:
            _render_replication_card(r)

    with st.expander("Replicate this contribution"):
        _render_replicate_form(contribution_id)

    components.section_header("Contributor Reputation")
    _render_reputation(services.community_reputation(c["contributor"]["display_name"]))


def _render_evidence_section(ev: dict[str, Any]) -> None:
    components.section_header("Evidence", "Read verbatim from the real registry -- never a free-text claim.")
    with components.card("community-detail-evidence"):
        components.provenance_row("Experiment ID", ev["experiment_id"])
        components.provenance_row("Root / Family", f"{ev['root_symbol']} / {ev['strategy_family']}")
        components.provenance_row("Asset Domain", ev["asset_domain"])
        components.provenance_row("Headline Verdict", ev["headline_verdict"] or "NOT_ADJUDICATED")
        components.provenance_row("Reason Codes", ", ".join(ev["reason_codes"]) or "--")
        components.provenance_row(
            "Market Window", f"{ev['market_window_label']} {ev['market_window_start']}..{ev['market_window_end']}",
        )
        if ev["strategy_params"] is not None:
            st.markdown("**Strategy Parameters**")
            st.json(ev["strategy_params"])
        else:
            st.caption("Strategy parameters redacted for this visibility level and viewer.")
        with st.expander("Advanced provenance"):
            components.provenance_row("Experiment Identity", ev["experiment_identity"])
            components.provenance_row("Reliability Policy Fingerprint", ev["reliability_policy_fingerprint"])
            components.provenance_row("Dataset Fingerprint", ev["dataset_fingerprint"])
            components.provenance_row("Code Commit", ev["code_commit"] or "--")


def _render_replication_card(r: dict[str, Any]) -> None:
    cmp = r["comparison"]
    with components.card(f"community-rep-{r['replication_id'][:24]}"):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{r['replicator']['display_name']}** (self-attested)")
            ev = r["evidence"]
            st.caption(f"Evidence: {ev['experiment_id']} -- verdict {ev['headline_verdict'] or 'NOT_ADJUDICATED'}")
            st.caption(cmp["notes"])
            dims = []
            for label, key in _COMPARISON_DIMENSIONS:
                v = cmp[key]
                dims.append(f"{label}: {'same' if v else ('n/a' if v is None else 'differs')}")
            st.caption(" | ".join(dims))
            if r.get("notes"):
                st.caption(r["notes"])
        with col2:
            components.render_badge(cmp["outcome"])


def _render_replicate_form(contribution_id: str) -> None:
    replicator = st.text_input(
        "Your researcher name", value=_researcher_name(), key=f"community-replicate-name-{contribution_id}",
    )
    exp_id = st.text_input(
        "Your OWN independent registry Experiment ID (its CANONICAL trial)",
        key=f"community-replicate-expid-{contribution_id}",
        help="Must be a different experiment_identity than the original -- re-citing the same run is refused.",
    )
    notes = st.text_area("Notes", key=f"community-replicate-notes-{contribution_id}", height=68)

    if not st.button("Replicate", type="primary", key=f"community-replicate-submit-{contribution_id}"):
        return
    if not replicator.strip() or not exp_id.strip():
        st.error("A researcher name and an experiment ID are both required.")
        return
    try:
        services.community_create_replication(
            contribution_id=contribution_id, replicator_display_name=replicator, experiment_id=exp_id.strip(),
            notes=notes,
        )
    except Exception as exc:  # noqa: BLE001 -- an honest refusal (same evidence, unknown id, holdout), never a crash
        st.error(f"Could not create replication: {exc}")
        return
    st.success("Replication recorded.")
    st.rerun()
