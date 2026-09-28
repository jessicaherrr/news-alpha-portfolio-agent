"""Alpha Discovery campaign, Part K -- the "Discover Strategies" page (task
spec sections 1/54-60), hardened by the Release UX bugfix pass (source-status
semantics, default product mode, Fast Screen honesty, candidate-card
information hierarchy).

    Your Profile -> Research Target -> Discover Strategies -> research
    progress -> frozen candidates (ready for a separate, explicit strict-
    validation run)

Every number this page shows comes straight from a real, typed
`alpha_agent.ui.discovery_campaign.DiscoveryCampaignOutcome` -- it never
fabricates a sample metric. The "Run real C++ Fast Screen" toggle is OFF by
default (candidate generation alone is fast; a real Fast Screen is one C++
subprocess per candidate, several seconds to about a minute each) and is
clearly labelled as such before a user opts in. This page never triggers
strict validation or a registry write -- see `discovery_campaign`'s own
module docstring for that boundary.

BUGFIX PASS -- what changed and why:

1. Source status honesty: a connector that was never REQUESTED for this
   campaign (Research Sources = "Internal + Classic only") now shows
   `DISABLED`, never `NOT_CONNECTED` -- the latter is reserved for a
   connector that WAS attempted (live research requested) and could not
   connect. See `alpha_agent.knowledge.external_sources.disabled_adapters`.
2. Default product mode: when `ANTHROPIC_API_KEY` is configured, "Connected
   Research" (Claude + all six knowledge sources) is the default,
   prominently-labelled mode; "Offline / Deterministic" and the
   offline/connected source toggle move into a collapsed "Advanced
   Settings" expander. When the key is NOT configured, an honest banner
   says so explicitly ("Connected sources available. Reasoning model
   unavailable.") rather than silently degrading to a demo-shaped
   experience -- external sources still default to available, since they
   need no Anthropic key at all.
3. Fast Screen semantics: compiled-but-unscreened candidates get a
   prominent "CANDIDATES NOT YET SCREENED" state and a "Run C++ Fast
   Screen" CTA (`discovery_campaign.run_fast_screen_on_pool` -- screens the
   EXACT already-compiled pool, never restarts discovery/generation).
   Candidates are labelled "Draft Candidate" and are NEVER numbered #1/#2
   until a real Fast Screen score exists to rank them by.
4. No `H-DEMO-*` hypothesis id, no experiment identity, no internal
   fingerprint on the primary card surface -- moved into a collapsed
   "Technical Details" expander. The display title is derived from the
   real `strategy_family` (the same friendly label every other page uses)
   or, for a custom blueprint with no Phase 11 family, from the real
   `hypothesis_title` with its own leading root token stripped -- the
   previous "CL CL dual moving-average trend" duplication is structural
   (the title already led with its own root) and cannot recur.
"""
from __future__ import annotations

import os
import re

import streamlit as st

from alpha_agent.knowledge import IngestionStatus, SourceType
from alpha_agent.recommendation.fit import PersonalizationState, score_user_fit
from alpha_agent.recommendation.promise import score_research_promise
from alpha_agent.screening.fast_screen import FastScreenStatus, rank_pool_members
from alpha_agent.ui import (
    components,
    discovery_campaign,
    layout,
    llm_demo,
    panels,
    recommendation_views,
    services,
)

PAGE_TITLE = "Discover"
_RESULT_KEY = "discover_outcome"
_VALIDATION_RESULT_KEY = "discover_strict_validation_outcome"
_CONFIRM_KEY = "discover-confirm-strict-validation"
_OBJECTIVE_KEY = "discover-objective"
_OBJECTIVE_ROOT_SYNC_KEY = "discover-objective-synced-root"

#: `strategy_family` values with an existing, already-used-elsewhere friendly
#: label (`services.strategy_name`) -- a candidate compiled into one of these
#: uses that label as its display title. A custom blueprint (no Phase 11
#: family; `strategy_family` is a synthetic DSL fingerprint like
#: "dsl:daily_trading_day:ma_spread+volatility") falls back to its own real
#: hypothesis title instead (see `_candidate_title`).
_KNOWN_STRATEGY_FAMILIES = frozenset({"tsmom", "ma_trend", "breakout", "mean_reversion", "silver_bullet"})

#: Presentation-only mechanism-category labels, independently declared (never
#: imported from `alpha_agent.recommendation.fit`'s own private mapping --
#: same "must not import that one" discipline `alpha_agent.screening.
#: fast_screen` already documents for its own independently-declared
#: constants).
_MECHANISM_LABELS = {
    "tsmom": "Trend / momentum continuation",
    "ma_trend": "Trend continuation",
    "breakout": "Breakout continuation",
    "mean_reversion": "Mean reversion",
}


def _anthropic_configured() -> bool:
    """Mirrors `alpha_agent.ui.views.agent._render_status_strip`'s own check
    exactly -- one definition of "is Claude configured" used consistently
    across pages. Never reads `.env` itself (matches `llm_demo`'s documented
    "requires ANTHROPIC_API_KEY in the environment Streamlit was launched
    from" contract) -- this app never silently pulls a key from a dotenv
    file the user did not export into the actual process environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active=None)
    layout.render_header(subtitle="Market fixed, strategy mechanism open -- an internal research workflow.")
    # An explicit hand-off target (e.g. Market's "Discover Strategies" button
    # seeds this before navigating here) always wins; otherwise a plain
    # default -- never the sidebar's former global market selector, which no
    # longer exists (Sidebar IA pass, task spec section 1).
    root = st.session_state.pop("discover_target_root", None) or services.approved_universe()[0]

    st.markdown("## Discover Strategies")
    st.caption(
        "Market fixed, strategy mechanism open: for the selected market, the Agent considers several "
        "economically distinct mechanisms (trend, momentum, mean reversion, breakout, regime-conditioned "
        "trend, ...), fast-screens the ones it can compile, and freezes the strongest for strict "
        "validation. Nothing here is a scientific verdict."
    )

    with components.card("discover-profile"):
        st.markdown('<div class="aa-gate-title">Your Profile</div>', unsafe_allow_html=True)
        panels.render_profile_panel(key_prefix="discover")

    mode, research_sources, run_backtests, clicked, objective = _render_research_target(root)

    if clicked:
        with st.spinner("Researching..."):
            profile = panels.current_investor_profile()
            outcome = discovery_campaign.run_discovery_campaign(
                mode=mode, objective=objective, root=root, family_stem=f"discover_{root.lower()}",
                profile=profile, run_fast_screen_backtests=run_backtests, research_sources=research_sources,
            )
        st.session_state[_RESULT_KEY] = outcome
        # a new discovery pass invalidates any pending confirmation / prior
        # strict-validation result for the OLD frozen set.
        st.session_state.pop(_VALIDATION_RESULT_KEY, None)
        st.session_state[_CONFIRM_KEY] = False

    outcome = st.session_state.get(_RESULT_KEY)
    if outcome is None:
        return
    if not outcome.accepted:
        st.error(outcome.error or "Discovery pass failed.")
        return
    if outcome.error:
        st.warning(outcome.error)

    _render_research_sources(outcome)
    _render_progress(outcome)
    _render_fast_screen_cta(outcome, root=root)
    _render_candidates(outcome, root=root)
    _render_strict_validation_gate(outcome, root=root, objective=objective)


# ---------------------------------------------------------------------------
# Research Target -- objective (issue 9), default product mode (issue 2)
# ---------------------------------------------------------------------------


def _mismatched_roots(objective: str, root: str) -> list[str]:
    """Every OTHER approved root mentioned (whole word) in `objective` while
    the Research Target is `root` -- issue 9's mismatch warning. Word-
    boundary matching only, since these are short all-caps tickers that must
    never match inside an unrelated word."""
    upper = objective.upper()
    return [
        other for other in services.approved_universe()
        if other != root and re.search(rf"(?<![A-Za-z0-9]){re.escape(other)}(?![A-Za-z0-9])", upper)
    ]


def _render_objective(root: str) -> str:
    """The objective text always follows the currently selected Research
    Target by default (issue 9): switching Market resets the box to the
    root-specific default UNLESS the user is mid-edit for that SAME root --
    the same "peek the widget's own session-state key, resync only when the
    thing it should track actually changed" pattern `agent.py`'s Root
    selector already uses for its preset sync. A user who then manually
    types a different approved root's ticker sees an explicit mismatch
    warning rather than a silent inconsistency."""
    default_text = f"Find robust alpha opportunities for {root}."
    if st.session_state.get(_OBJECTIVE_ROOT_SYNC_KEY) != root:
        st.session_state[_OBJECTIVE_KEY] = default_text
        st.session_state[_OBJECTIVE_ROOT_SYNC_KEY] = root
    objective = st.text_area("Objective", key=_OBJECTIVE_KEY)
    mismatched = _mismatched_roots(objective, root)
    if mismatched:
        st.warning(
            f"Objective mentions {', '.join(mismatched)}, but Research Target (market) is **{root}**. "
            f"The research pipeline targets {root} only -- edit the objective if this was unintentional."
        )
    return objective


def _render_research_target(root: str) -> tuple[str, str, bool, bool, str]:
    """Returns `(mode, research_sources, run_backtests, clicked, objective)`.

    Issue 2 -- default product mode: when Claude is configured, "Connected
    Research" (live sources + Claude reasoning) is the prominent default;
    the Offline/Deterministic engine and the source-subset toggle move into
    a collapsed "Advanced Settings" expander so they no longer dominate the
    primary experience. External research sources default to available
    REGARDLESS of Claude's availability (GitHub/Academic/Community/
    Practitioner need no Anthropic key) -- only the reasoning-engine default
    depends on it, and its unavailability is stated plainly, never silently
    swapped for a demo-shaped fallback.
    """
    with components.card("discover-target"):
        st.markdown('<div class="aa-gate-title">Research Target</div>', unsafe_allow_html=True)
        st.markdown(f"**Market**  \n{root} -- {services.market_name(root)}")
        objective = _render_objective(root)

        anthropic_ok = _anthropic_configured()
        if anthropic_ok:
            st.markdown(components.badge("OK", label="CONNECTED RESEARCH"), unsafe_allow_html=True)
            st.caption(
                "Primary mode: Claude reasoning plus Internal, Classic, GitHub, Academic, Community, "
                "and Practitioner sources. Change this in Advanced Settings."
            )
        else:
            st.warning(
                "Connected sources available. Reasoning model unavailable -- `ANTHROPIC_API_KEY` is not "
                "configured in this environment, so Offline / Deterministic reasoning will be used instead "
                "of Claude. Live GitHub / Academic / Community / Practitioner sources are unaffected by "
                "this and still default to available."
            )

        with st.expander("Advanced Settings", expanded=False):
            mode = st.radio(
                "Research Engine", [llm_demo.SCRIPTED_MODE, llm_demo.LIVE_MODE],
                index=1 if anthropic_ok else 0,
                format_func=lambda m: "Offline / Deterministic" if m == llm_demo.SCRIPTED_MODE else "Claude Research",
                horizontal=True, key="discover-mode",
                help="Offline / Deterministic replays a canned, schema-valid response -- no network call, "
                     "no cost. Claude Research makes a real, paid Anthropic API call per hypothesis.",
            )
            research_sources = st.radio(
                "Research Sources", ["offline", "connected"],
                index=1,
                format_func=lambda m: "Internal + Classic only" if m == "offline" else "Internal + Classic + Live (GitHub, Academic, Community, Practitioner)",
                horizontal=True, key="discover-research-sources",
                help="Connected mode makes real, read-only calls to public research APIs (GitHub, "
                     "OpenAlex/Crossref, Hacker News). A source that fails or is unreachable is shown "
                     "honestly -- it never blocks the others or fabricates a result. Offline mode never "
                     "attempts any of these -- they show DISABLED, not a fabricated failure.",
            )
        run_backtests = st.checkbox(
            "Run real C++ Fast Screen (slower -- one real backtest per candidate, ~1 min each)",
            value=False, key="discover-run-backtests",
        )
        clicked = st.button("Discover Strategies", type="primary", key="discover-run-button")
    return mode, research_sources, run_backtests, clicked, objective


_SOURCE_LABELS = {
    SourceType.INTERNAL: "Internal Research", SourceType.CLASSIC: "Classic Strategies",
    SourceType.GITHUB: "GitHub", SourceType.ACADEMIC: "Academic Papers",
    SourceType.COMMUNITY: "Community", SourceType.PRACTITIONER: "Practitioner",
}
_EXTERNAL_SOURCE_TYPES = (SourceType.GITHUB, SourceType.ACADEMIC, SourceType.COMMUNITY, SourceType.PRACTITIONER)


def _render_research_sources(outcome) -> None:
    """Task spec "RESEARCH SOURCES" / section 61 -- a compact, honest status
    row. Internal/Classic are always READY (no network); external sources
    show the REAL `IngestionResult.status` this exact pass produced --
    DISABLED when not requested (issue 1), never a fabricated status either
    way."""
    kb = outcome.knowledge_base
    if kb is None:
        return
    with components.card("discover-research-sources"):
        st.markdown('<div class="aa-gate-title">Research Sources</div>', unsafe_allow_html=True)
        cols = st.columns(6)
        cols[0].metric(_SOURCE_LABELS[SourceType.INTERNAL], "READY")
        cols[1].metric(_SOURCE_LABELS[SourceType.CLASSIC], "READY")
        by_type = {r.source_type: r for r in kb.external_ingestion_results}
        for i, st_ in enumerate(_EXTERNAL_SOURCE_TYPES, start=2):
            result = by_type.get(st_)
            status = result.status.value if result else IngestionStatus.DISABLED.value
            show_items = result is not None and result.status != IngestionStatus.DISABLED
            n_items = len(result.items) if result else 0
            cols[i].metric(
                _SOURCE_LABELS[st_], status,
                delta=f"{n_items} item(s)" if show_items else None, delta_color="off",
            )
        with st.expander("Technical connector detail", expanded=False):
            for r in kb.external_ingestion_results:
                st.caption(f"{_SOURCE_LABELS[r.source_type]}: {r.status.value} -- {r.detail}")


def _render_progress(outcome) -> None:
    """Task spec issue 6: splits the previous single "Sources Retrieved"
    metric (which conflated always-local Internal/Classic knowledge with
    actual live external retrieval) into "Local Knowledge Items" and "Live
    Sources Retrieved"."""
    pool = outcome.pool
    kb = outcome.knowledge_base
    local_items = [i for i in kb.items if i.source_type in (SourceType.INTERNAL, SourceType.CLASSIC)] if kb else []
    n_local = len(local_items)
    n_live = sum(len(r.items) for r in kb.external_ingestion_results) if kb else 0
    n_mechanisms = len({i.economic_mechanism for i in kb.items}) if kb else 0
    with components.card("discover-progress"):
        st.markdown('<div class="aa-gate-title">Research Process</div>', unsafe_allow_html=True)
        cols = st.columns(7)
        cols[0].metric("Local Knowledge Items", n_local)
        cols[1].metric("Live Sources Retrieved", n_live)
        cols[2].metric("Unique Mechanisms", n_mechanisms)
        cols[3].metric("Ideas Considered", pool.ideas_considered if pool else 0)
        cols[4].metric("Compiled", len(pool.members) if pool else 0)
        cols[5].metric("Fast Screened", len(outcome.trials))
        cols[6].metric("Frozen", outcome.frozen.frozen_count if outcome.frozen else 0)


# ---------------------------------------------------------------------------
# Fast Screen CTA (issue 7) -- screens the ALREADY-COMPILED pool, never
# restarts discovery/generation/source retrieval.
# ---------------------------------------------------------------------------


def _render_fast_screen_cta(outcome, *, root: str) -> None:
    pool = outcome.pool
    if not pool or not pool.members or outcome.trials:
        return
    with components.card("discover-fastscreen-cta"):
        st.markdown(components.badge("WARN", label="CANDIDATES NOT YET SCREENED"), unsafe_allow_html=True)
        st.markdown("**These ideas are compile-ready only. No performance evidence has been produced.**")
        st.caption(
            "These candidates are structurally valid but have not yet been tested economically. Running "
            "Fast Screen re-uses the exact candidates already compiled above -- it does not restart "
            "discovery, re-query research sources, or call the reasoning model again."
        )
        run_clicked = st.button("Run C++ Fast Screen", type="primary", key="discover-run-fastscreen-cta")
    if run_clicked:
        _execute_fast_screen(outcome, root=root)


def _execute_fast_screen(outcome, *, root: str) -> None:
    with st.spinner("Running real C++ Fast Screen on the already-compiled candidates..."):
        updated = discovery_campaign.run_fast_screen_on_pool(
            pool=outcome.pool, knowledge_base=outcome.knowledge_base, root=root,
            family_stem=f"discover_{root.lower()}",
        )
    st.session_state[_RESULT_KEY] = updated
    st.session_state.pop(_VALIDATION_RESULT_KEY, None)
    st.session_state[_CONFIRM_KEY] = False
    st.rerun()


# ---------------------------------------------------------------------------
# Candidate cards (issues 3/4/5) -- Draft Candidate until Fast Screen exists,
# ranked only once real scores exist; no H-DEMO id / experiment id / internal
# fingerprint on the primary surface.
# ---------------------------------------------------------------------------


def _candidate_title(member) -> str:
    """The primary display title -- never the raw `hypothesis_id`, never a
    duplicated root. A known Phase 11 family gets the SAME friendly label
    every other page already uses (`services.strategy_name`); a custom
    blueprint (whose `strategy_family` is a synthetic DSL fingerprint, e.g.
    "dsl:daily_trading_day:ma_spread+volatility" -- not human-friendly)
    falls back to its own real `hypothesis_title`, with a leading root token
    stripped if present (every curated hypothesis title conventionally
    leads with its own root, e.g. "CL dual moving-average trend" -- the
    exact cause of the previous "CL CL dual moving-average trend" bug)."""
    if member.strategy_family in _KNOWN_STRATEGY_FAMILIES:
        return services.strategy_name(member.strategy_family)
    title = (member.hypothesis_title or "").strip()
    root_prefix = f"{member.root_symbol} "
    if title[: len(root_prefix)].upper() == root_prefix.upper():
        title = title[len(root_prefix):]
    if not title:
        return member.strategy_family or "Custom Strategy"
    return title[:1].upper() + title[1:]


def _mechanism_label(member) -> str:
    return _MECHANISM_LABELS.get(member.strategy_family, "Regime-conditioned / custom logic")


def _why_it_stands_out(score) -> str:
    """Derived ONLY from the real `ResearchScreenScore` sub-components --
    never a new judgment, never fabricated prose."""
    weights = {"Sharpe": 40.0, "cost efficiency": 25.0, "drawdown control": 20.0, "trading activity": 15.0}
    fractions = {
        "Sharpe": score.sharpe_component / weights["Sharpe"],
        "cost efficiency": score.cost_component / weights["cost efficiency"],
        "drawdown control": score.drawdown_component / weights["drawdown control"],
        "trading activity": score.activity_component / weights["trading activity"],
    }
    best_label, best_frac = max(fractions.items(), key=lambda kv: kv[1])
    return f"Strongest on {best_label} in the 2018-2022 research window ({best_frac:.0%} of that component's available weight)."


def _render_candidates(outcome, *, root: str) -> None:
    pool = outcome.pool
    if not pool or not pool.members:
        components.empty_state(
            "Most Promising Research Candidates",
            "No mechanism compiled into an executable candidate for this market/objective this pass.",
            key="discover-empty",
        )
        return

    trial_by_id = {t.experiment_identity: t for t in outcome.trials}
    frozen_ids = {m.experiment_identity for m in outcome.frozen.manifest.members} if outcome.frozen else set()

    def _is_screened(member) -> bool:
        t = trial_by_id.get(member.experiment_identity)
        return t is not None and t.status is FastScreenStatus.SCREENED and t.score is not None

    any_screened = any(_is_screened(m) for m in pool.members)
    profile = panels.current_investor_profile()

    # Ranking (issue 3): only once real Fast Screen evidence exists for at
    # least one candidate -- a compile-only pool is never presented as if it
    # were already ranked.
    ordered = rank_pool_members(list(pool.members), list(outcome.trials)) if any_screened else pool.members

    for position, member in enumerate(ordered, start=1):
        trial = trial_by_id.get(member.experiment_identity)
        screened = _is_screened(member)
        with components.card(f"discover-candidate-{member.experiment_identity[-20:]}"):
            st.markdown(f"**{_candidate_title(member).upper()}**")
            st.caption(f"{member.root_symbol} {services.market_name(member.root_symbol)}")
            if member.experiment_identity in frozen_ids:
                st.markdown(components.badge("OK", label="FROZEN FOR STRICT VALIDATION"), unsafe_allow_html=True)

            if screened:
                if any_screened:
                    st.caption(f"Rank #{position} of {len(ordered)} by Fast Screen Score.")
                _render_screened_candidate_body(member, trial, profile=profile)
            else:
                _render_draft_candidate_body(outcome, member, trial)

            with st.expander("Technical Details", expanded=False):
                st.caption(f"Hypothesis ID: `{member.hypothesis_id}`")
                st.caption(f"Experiment identity: `{member.experiment_identity}`")
                st.caption(f"Strategy fingerprint: `{member.strategy_fingerprint}`")
                st.caption(f"Family: `{member.strategy_family}`")

            _render_source_inspirations(outcome, member, trial=trial)


def _render_draft_candidate_body(outcome, member, trial) -> None:
    st.markdown(components.badge("WARN", label="DRAFT CANDIDATE"), unsafe_allow_html=True)
    with components.metric_row(f"discover-draft-{member.experiment_identity[-20:]}"):
        c1, c2 = st.columns(2)
        c1.metric("Mechanism", _mechanism_label(member))
        n_related = len(_related_knowledge_items(outcome, member))
        c2.metric("Research Inspiration", f"{n_related} source(s)" if n_related else "None yet")
    st.caption(
        f"Why considered: targets {_mechanism_label(member).lower()} in {member.root_symbol}, and compiled "
        "successfully against the closed strategy DSL."
    )
    if trial is not None and trial.status is not FastScreenStatus.SCREENED:
        st.caption(
            f"Fast Screen: {trial.status.value}"
            + (f" ({trial.capability_reason.value})" if trial.capability_reason else "")
            + " -- a technical research constraint, not a scientific rejection."
        )
    else:
        st.caption("Status: compile-ready only. No performance evidence has been produced yet.")


def _render_screened_candidate_body(member, trial, *, profile) -> None:
    score = trial.score
    metrics = trial.metrics
    with components.metric_row(f"discover-screened-{member.experiment_identity[-20:]}"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fast Screen Score", f"{score.total:.1f}/100")
        c2.metric("Sharpe (2018-2022)", f"{metrics.get('annualized_sharpe', 0):.2f}")
        c3.metric("Net PnL (2018-2022)", f"${metrics.get('net_pnl_usd', 0):,.0f}")
        mdd = metrics.get("max_drawdown_fraction")
        c4.metric("Max Drawdown", f"{mdd:.1%}" if mdd is not None else "N/A")

    # Research Promise is a POST-VALIDATION score
    # (`alpha_agent.recommendation.promise.score_research_promise`) that
    # reads DSR/BH-q/gating-p-value fields a Fast Screen trial never
    # produces -- `score_research_promise(None)` honestly resolves to
    # `missing_inputs=("result",)` and total=0.0 (task spec issue 5, "use
    # only real values"): rendered here as an explicit "Not yet available"
    # so it is never mistaken for a real negative scientific signal.
    #
    # Fit for You CAN carry a real signal even pre-validation: its mechanism
    # dimension is profile-only (never reads `result`) -- calling the real
    # `score_user_fit` with `result=None` surfaces that honestly (real
    # value when the user has a stated mechanism preference; "not
    # personalized" when they do not) rather than blanket-hiding it.
    promise = score_research_promise(None)
    fit = score_user_fit(strategy_family=member.strategy_family, result=None, profile=profile)
    with components.metric_row(f"discover-screened-promise-{member.experiment_identity[-20:]}"):
        c1, c2 = st.columns(2)
        c1.metric("Research Promise", "Not yet available")
        if fit.personalization_state == PersonalizationState.NOT_PERSONALIZED:
            c2.metric("Fit for You", "Not personalized")
        else:
            c2.metric("Fit for You", f"{fit.total:.0f}%")
    st.caption(
        f"Research Promise requires a committed strict-validation result ({', '.join(promise.missing_inputs)}) "
        "-- Fast Screen alone cannot produce one."
    )
    st.caption(f"Why it stands out: {_why_it_stands_out(score)}")
    st.caption(
        "What needs validation: 2023-2024 out-of-sample performance, BH-FDR multiple-testing correction, "
        "and the Deflated Sharpe Ratio."
    )


def _render_strict_validation_gate(outcome, *, root: str, objective: str) -> None:
    """Checkpoint 4 -- Review Frozen Candidates -> explicit confirmation ->
    Run Strict Validation -> real Verdict / Promise / Fit. Never shown until
    a Fast Screen + Freeze pass has actually produced a frozen manifest."""
    frozen = outcome.frozen
    if frozen is None:
        return

    with components.card("discover-review-frozen"):
        st.markdown('<div class="aa-gate-title">Review Frozen Candidates</div>', unsafe_allow_html=True)
        st.caption(
            f"{frozen.frozen_count} candidate(s) survived the Fast Screen and are frozen into one "
            f"predeclared multiple-testing family (`{frozen.manifest.family_id[:24]}...`)."
        )
        for m in frozen.manifest.members:
            st.caption(f"🔒 {m.root_symbol} -- {m.hypothesis_title} (`{m.strategy_family}`, {m.signal_cadence})")

        st.warning(discovery_campaign.STRICT_VALIDATION_WARNING)
        confirmed = st.checkbox(
            "I understand -- these candidates are frozen and cannot change once validation begins.",
            key=_CONFIRM_KEY,
        )
        st.caption(
            "Running strict validation executes a real 2023-2024 C++ backtest and the real frozen "
            "validation pipeline for every frozen candidate, and writes the result to the experiment "
            "registry."
        )
        run_clicked = st.button(
            "Run Strict Validation", type="primary", key="discover-run-strict-validation",
            disabled=not confirmed,
        )

    if run_clicked:
        with st.spinner("Running real 2023-2024 strict validation..."):
            result = discovery_campaign.run_strict_validation(
                frozen=frozen, objective=objective, root=root,
            )
        st.session_state[_VALIDATION_RESULT_KEY] = result

    result = st.session_state.get(_VALIDATION_RESULT_KEY)
    if result is None:
        return
    _render_strict_validation_result(result)


def _render_strict_validation_result(result) -> None:
    with components.card("discover-strict-validation-result"):
        st.markdown('<div class="aa-gate-title">Strict Validation Result</div>', unsafe_allow_html=True)
        if not result.accepted:
            st.error(result.error or "Strict validation was refused.")
            return
        report = result.report
        cols = st.columns(3)
        cols[0].metric("Frozen Candidates", report.predeclared_family_size)
        cols[1].metric("Completed", len(report.member_results))
        cols[2].metric("BH-Rejected", report.n_bh_rejected)

    for member_result in result.report.member_results:
        summary = services.candidate_summary_for_experiment(member_result.experiment_id)
        if summary is None:
            continue
        recommendation_views.render_candidate_card(summary, key_prefix="discover-strict-result")


def _related_knowledge_items(outcome, member) -> list:
    """A member's research inspirations, matched two ways: (1) the classic-
    library `candidate_dsl_template` link (Part C), and (2) the LIVE lineage
    the ResearchAgent itself cited (`member.source_inspirations`, Checkpoint
    9) -- matched by title substring, since that is exactly what the agent
    was asked to quote."""
    kb = outcome.knowledge_base
    if kb is None:
        return []
    items = kb.query(mechanism=None)
    related = [i for i in items if i.candidate_dsl_template == member.strategy_family]
    cited_ids = {i.knowledge_id for i in related}
    for cited_text in member.source_inspirations:
        for item in items:
            if item.knowledge_id in cited_ids:
                continue
            if item.title and item.title in cited_text:
                related.append(item)
                cited_ids.add(item.knowledge_id)
    return related


def _render_source_inspirations(outcome, member, *, trial=None) -> None:
    related = _related_knowledge_items(outcome, member)
    if not related:
        return
    with st.expander(f"Research Inspiration ({len(related)})", expanded=False):
        for item in related[:6]:
            st.markdown(f"**[{item.source_type.value}]** {item.title}")
            meta_bits = []
            if item.authors:
                meta_bits.append(", ".join(item.authors[:3]))
            if item.publication_date:
                meta_bits.append(str(item.publication_date))
            if item.source_quality:
                meta_bits.append(f"quality: {item.source_quality.value}")
            if meta_bits:
                st.caption(" -- ".join(meta_bits))
            if item.source_url:
                st.caption(item.source_url)
            st.caption(f"Mechanism extracted: {item.economic_mechanism.value}. {item.entry_logic_summary}")
            st.caption(
                "Source reported performance: "
                + (item.reported_results if item.reported_results else "not claimed / not applicable")
                + " -- NEVER our evidence."
            )
            st.divider()
        if trial is not None:
            if trial.status is FastScreenStatus.SCREENED and trial.score is not None:
                st.caption(
                    f"Our actual evidence (2018-2022 Fast Screen): Sharpe "
                    f"{trial.metrics.get('annualized_sharpe', 0):.2f}, "
                    f"net PnL ${trial.metrics.get('net_pnl_usd', 0):,.0f}."
                )
            else:
                st.caption("Our actual evidence: not yet screened this pass.")
        if any(i.source_type == SourceType.INTERNAL for i in related):
            st.caption("Includes internal registry history -- past experiments, not new evidence.")


__all__ = ["PAGE_TITLE", "render"]
