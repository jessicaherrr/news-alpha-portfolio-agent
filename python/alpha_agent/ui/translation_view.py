"""Research Translation -- the Phase 1 hand-off from a news/event item to a
single-market, falsifiable STRATEGY hypothesis (certified Futures roots
only), with personal / cross-asset / community research memory and the
context-aware research ranking beside it.

Moved verbatim from the Agent page's retired News Impact Triage card; it now
sits in a Research Thread's Signals step as the alternative route "test this
as a single-market strategy". "Research This" seeds the SAME Agent research
composer as before (`views.agent.start_inline_research_from_translation`)
and opens Ask, where the hypothesis is compiled and run -- no duplicated
research execution path. Translation itself is deterministic by default;
Claude is only called on the per-item opt-in.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.news_alpha import DOMAIN_LABELS, TranslationHandoffPlan
from alpha_agent.translation.schemas import ObservationTranslation
from alpha_agent.ui import (
    components,
    context_retrieval_context,
    panels,
    services,
    translation_context,
)

__all__ = ["render_translation_handoffs", "view_alpha_memory_button"]

_TRANSLATIONS_KEY = "agent_observation_translations"

_RESEARCHABILITY_BADGE_TONE = {
    "AVAILABLE": "OK",
    "PARTIALLY_AVAILABLE": "WARN",
    "DATA_MISSING": "NOT_AVAILABLE",
    "NOT_EXECUTABLE": "REJECT",
}


def render_translation_handoffs(plan: TranslationHandoffPlan, *, card_key: str) -> None:
    if not plan.handoffs and not plan.gaps:
        return
    st.session_state.setdefault(_TRANSLATIONS_KEY, {})
    st.markdown('<div class="aa-gate-title" style="margin-top:0.4rem;">RESEARCH TRANSLATION</div>', unsafe_allow_html=True)
    if plan.handoffs:
        use_claude = st.checkbox(
            "Use Claude Research for mechanism reasoning",
            key=f"agent-translate-claude-{card_key}",
            help="Off (default): the deterministic mechanism library reasons about this observation -- no network "
                 "call. On: one bounded live Claude call proposes the mechanism/factor narrative; researchability "
                 "and feature availability are still decided deterministically either way, never by Claude.",
        )
        cols = st.columns(min(len(plan.handoffs), 5))
        for i, handoff in enumerate(plan.handoffs):
            with cols[i % len(cols)]:
                if st.button(f"Translate {handoff.root_symbol}", key=f"agent-translate-go-{handoff.key}", width="stretch"):
                    mode = translation_context.LIVE_MODE if use_claude else translation_context.SCRIPTED_MODE
                    translation, error = translation_context.translate_observation(handoff.observation, mode=mode)
                    st.session_state[_TRANSLATIONS_KEY][handoff.key] = {"translation": translation, "error": error}
                    st.rerun()
        for note in dict.fromkeys(h.fidelity_note for h in plan.handoffs if h.category_basis.value != "SOURCE_CATEGORY"):
            st.caption(note)
    if plan.gaps:
        # One line, not one per domain: every gap shares the same cause today
        # (Phase 1 translates certified Futures roots only); the full typed
        # reason stays on each `TranslationGap` and in "Why these domains?".
        names = ", ".join(
            f"{DOMAIN_LABELS[g.asset_domain]} ({g.impact_level.value.replace('_', ' ')})" for g in plan.gaps
        )
        st.caption(
            f"{names}: no translation path yet -- {plan.gaps[0].reason}"
            if len(plan.gaps) == 1
            else f"{names}: no translation path yet -- translation accepts certified Futures roots only; these "
            "continue at the Mechanism Graph / Asset Expression stages."
        )

    for handoff in plan.handoffs:
        cached = st.session_state.get(_TRANSLATIONS_KEY, {}).get(handoff.key)
        if cached is None:
            continue
        if cached.get("error"):
            st.error(cached["error"])
            continue
        _render_translation_card(cached["translation"], key_prefix=f"agent-translate-{handoff.key}")


def _render_translation_card(translation: ObservationTranslation, *, key_prefix: str) -> None:
    obs = translation.observation

    st.markdown('<div class="aa-gate-title">OBSERVATION</div>', unsafe_allow_html=True)
    st.caption(f"{obs.summary} · {obs.source} · {obs.root_symbol}")
    st.caption(f"Observed {obs.observed_at.strftime('%Y-%m-%d %H:%M:%S UTC')} (origin vintage {obs.origin_vintage})")

    st.markdown('<div class="aa-gate-title" style="margin-top:0.5rem;">POSSIBLE MECHANISMS</div>', unsafe_allow_html=True)
    for m in translation.mechanism_candidates:
        st.markdown(f"**{m.mechanism.value}** &middot; *{m.provenance.value}*", unsafe_allow_html=True)
        st.caption(m.explanation)
        if m.causal_chain:
            st.caption(" → ".join(m.causal_chain))
    if not translation.mechanism_candidates:
        st.caption("No deterministic mechanism template exists for this observation's category yet.")

    with st.expander("Measurable variables", expanded=False):
        if not translation.measurable_variables:
            st.caption("No measurable variables were proposed for this observation.")
        for v in translation.measurable_variables:
            st.markdown(f"**{v.name}** -- {v.economic_meaning}")
            st.caption(f"Requires: {v.required_source}")
            # Three-state, never collapsed to a bool: UNKNOWN (a Claude-
            # proposed variable's free text is never scanned for
            # availability markers) must never render as "NO" -- that would
            # claim unavailability that isn't actually known (Phase 1
            # acceptance patch, section 1).
            if v.point_in_time_available is None:
                pit = "Point-in-time availability: UNKNOWN. "
            elif v.point_in_time_available:
                pit = "Point-in-time availability: YES. "
            else:
                pit = "Point-in-time availability: NO. "
            st.caption(pit + v.point_in_time_note)

    st.markdown(
        '<div class="aa-gate-title" style="margin-top:0.5rem;">FACTOR CANDIDATES &amp; RESEARCHABILITY</div>',
        unsafe_allow_html=True,
    )
    for f in translation.factor_candidates:
        tone = _RESEARCHABILITY_BADGE_TONE.get(f.researchability.value, "NOT_AVAILABLE")
        st.markdown(
            f"**{f.concept}** " + components.badge(tone, label=f.researchability.value),
            unsafe_allow_html=True,
        )
        st.caption(f.researchability_reason)
        st.caption(f"Provenance: {f.provenance.value}")
        if f.missing_requirements:
            st.caption("Missing: " + "; ".join(f.missing_requirements))

    with st.expander("Research memory", expanded=False):
        st.caption(translation.research_memory.summary)
        for lesson in translation.research_memory.engineering_lessons:
            st.caption(f"• {lesson}")
        _render_personal_alpha_memory_links(translation, key_prefix=key_prefix)

    _render_context_aware_retrieval(translation, key_prefix=key_prefix)

    st.markdown('<div class="aa-gate-title" style="margin-top:0.5rem;">HYPOTHESIS</div>', unsafe_allow_html=True)
    if translation.hypothesis is not None:
        h = translation.hypothesis
        st.markdown(f"**{h.title}**")
        st.caption(h.signal_description)
        st.markdown('<div class="aa-gate-title" style="margin-top:0.3rem;">FALSIFICATION</div>', unsafe_allow_html=True)
        st.caption(h.falsification_test)
        if st.button("Research This", key=f"{key_prefix}-research-this", width="stretch"):
            _research_this(translation)
    else:
        st.info(translation.research_gap_note or "No falsifiable hypothesis could be constructed from this observation yet.")

    with st.expander("Evidence & provenance", expanded=False):
        st.caption(f"Generated by: {translation.generated_by}")
        st.caption(translation.reasoning_provenance_note)
        for ref in obs.evidence_refs:
            st.caption(f"• {ref}")


def _render_personal_alpha_memory_links(translation: ObservationTranslation, *, key_prefix: str) -> None:
    """Phase 1 <-> Phase 2 integration (prompt 2 section 17; hardened by the
    identity-hardening patch, then the final Phase 2 semantic fix, section
    2): "have I researched something like this before?" resolved against
    real Personal Alpha Memory.

    Iterates ``translation.factor_candidates`` (never just the coarser
    ``mechanism_candidates`` list) so each mechanism's real, typed
    ``concept``/``transform_or_proxy`` -- not just its enum value -- is shown
    next to whatever memory is retrieved, letting a reader judge the match
    themselves rather than the system silently asserting one (section 5:
    "do not identify a Phase 1 candidate only by its mechanism"). Phase 1
    does not yet compute a candidate's own ``strategy_family``, so this
    mechanism-only lookup can NEVER claim a confident "matching factor" --
    every real hit renders as "Related Mechanism Memory", however many real
    Factor objects exist under the mechanism (cardinality is not proof); a
    future integration point that does know the candidate's
    ``strategy_family`` would see ``services.mechanism_memory_lookup``
    return ``"MATCHING_FACTOR"`` instead, which this renderer already
    handles. Renders nothing for a mechanism with no registry-grounded
    evidence yet -- an honest "not researched before" is not itself an
    error."""
    seen: set[str] = set()
    for f in translation.factor_candidates:
        if f.mechanism.value in seen:
            continue
        seen.add(f.mechanism.value)
        lookup = services.mechanism_memory_lookup(
            mechanism=f.mechanism.value, root_symbol=translation.observation.root_symbol,
        )
        if lookup["match_kind"] == "NONE":
            continue
        st.caption(f"Factor candidate: **{f.concept}** ({f.transform_or_proxy})")
        if lookup["match_kind"] == "MATCHING_FACTOR":
            obj = lookup["objects"][0]
            st.caption(f"Personal Alpha Memory (matching factor) -- {obj['summary']}")
            view_alpha_memory_button(alpha_id=obj["alpha_id"], key=f"{key_prefix}-alpha-memory-{f.mechanism.value}")
        else:  # RELATED_MECHANISM -- mechanism-only lookup can never claim an exact Factor match
            st.caption(
                f"Related Mechanism Memory -- {len(lookup['objects'])} real Factor(s) implement "
                f"{f.mechanism.value} on {translation.observation.root_symbol}; mechanism alone cannot "
                "prove which one (if any) this candidate corresponds to:"
            )
            for i, obj in enumerate(lookup["objects"]):
                st.caption(f"- {obj['summary']}")
                view_alpha_memory_button(
                    alpha_id=obj["alpha_id"], key=f"{key_prefix}-alpha-memory-{f.mechanism.value}-{i}",
                )
        _render_cross_asset_related_evidence(
            mechanism=f.mechanism.value, root_symbol=translation.observation.root_symbol,
        )
        _render_community_memory_caption(
            mechanism=f.mechanism.value, root_symbol=translation.observation.root_symbol,
        )


def _render_community_memory_caption(*, mechanism: str, root_symbol: str) -> None:
    """Phase 8 (Community Alpha Network) integration -- the small,
    deterministic Community Memory read (prompt 8 section 8: Event/Context
    -> Personal + Community Memory -> research prioritization), same
    additive pattern as `_render_cross_asset_related_evidence` just above.
    Renders nothing when no SHARED/PUBLIC community evidence exists yet for
    this (mechanism, root) -- absence is not itself a signal. Purely
    informational: `alpha_agent.community.memory`'s own note is swept by a
    dedicated test for trade-instruction vocabulary and always ends with an
    explicit "never a guaranteed trade recommendation" disclaimer."""
    lookup = services.community_memory_lookup(mechanism=mechanism, root_symbol=root_symbol)
    community = lookup["community"]
    if community is None:
        return
    st.caption(
        f"Community Alpha Network -- {community['n_public_contributions']} public and "
        f"{community['n_shared_contributions']} shared contribution(s) from {community['n_contributors']} "
        f"contributor(s), {community['n_replications']} independent replication(s) for {mechanism} on "
        f"{root_symbol}. Research context only -- see the Community tab."
    )


def _render_cross_asset_related_evidence(*, mechanism: str, root_symbol: str) -> None:
    """Phase 7 (Cross-Asset Alpha Graph) integration: the small, deterministic
    Agent retrieval layer's "analogous research" answer -- OTHER instruments
    (any asset domain) with real evidence for this Mechanism. Purely
    informational: it may say "this Mechanism has related evidence in NQ and
    XLE", never "because it worked in NQ, buy XLE" -- related evidence is
    never transferred as a verdict onto `root_symbol` (prompt 7 section 4/5).
    Renders nothing when no analogous research exists yet."""
    related = services.alpha_graph_analogous_research(mechanism=mechanism, root_symbol=root_symbol)
    if not related:
        return
    instruments = ", ".join(f"{r['instrument']['root_symbol']} ({r['instrument']['asset_domain']})" for r in related)
    st.caption(
        f"Cross-Asset Research Map -- {mechanism} also has real, separately-adjudicated registry evidence "
        f"in: {instruments}. Related evidence for research prioritization only, never transferred proof."
    )


_CONTEXT_RETRIEVAL_KEY = "agent_context_retrieval"

_CONTEXT_MATCH_BADGE_TONE = {"STRONG": "OK", "MODERATE": "WARN", "WEAK": "WARN", "NONE": "NOT_AVAILABLE"}
_SCIENTIFIC_RELIABILITY_BADGE_TONE = {
    "PASSED": "PASS", "MIXED": "MIXED", "REJECTED": "REJECT", "INCONCLUSIVE": "INCONCLUSIVE",
    "NOT_ADJUDICATED": "NOT_ADJUDICATED", "NO_EVIDENCE": "NOT_AVAILABLE",
}
_RESEARCH_COVERAGE_BADGE_TONE = {
    "FULLY_TESTED": "OK", "PARTIALLY_TESTED": "WARN", "MAPPED_UNTESTED": "WARN", "NOT_MAPPED": "NOT_AVAILABLE",
}


def _render_context_aware_retrieval(translation: ObservationTranslation, *, key_prefix: str) -> None:
    """Phase 3 (Agentic Alpha Evolution) -- context-aware retrieval +
    transparent research ranking for this same real observation. A separate,
    explicit action (never auto-run on every card render): it fetches real
    current market state (trend/volatility/curve/related-market confirmation)
    through `context_retrieval_context`'s own cache-first accessors, exactly
    like Top Opportunities/Market Context already do elsewhere on this page,
    but only when the user asks for it here.

    Deterministic/versioned/explainable ranking ONLY (RETRIEVAL_RANK_POLICY)
    -- never expected return, probability of success, or trade confidence;
    see `alpha_agent.context_retrieval.schemas` module docstring."""
    with st.expander("Context-Aware Research Ranking (Phase 3)", expanded=False):
        st.caption(
            "When this same kind of market/news event recurs, which mechanisms/factors/strategies already "
            "researched are most RELEVANT to it right now -- a research-prioritization ranking, never "
            "expected return, probability of success, or trade confidence."
        )
        st.session_state.setdefault(_CONTEXT_RETRIEVAL_KEY, {})
        if st.button("Retrieve Context-Aware Ranking", key=f"{key_prefix}-context-retrieval-go"):
            observation = translation.observation
            profile = panels.current_investor_profile()
            result = context_retrieval_context.retrieve_for_observation(observation, investor_profile=profile)
            st.session_state[_CONTEXT_RETRIEVAL_KEY][key_prefix] = result
            st.rerun()

        result = st.session_state[_CONTEXT_RETRIEVAL_KEY].get(key_prefix)
        if result is None:
            return
        _render_context_retrieval_result(result, key_prefix=key_prefix)


def _render_context_retrieval_result(result, *, key_prefix: str) -> None:
    ctx = result.context
    st.markdown('<div class="aa-gate-title">CURRENT CONTEXT</div>', unsafe_allow_html=True)
    components.provenance_row("Root / Event Category", f"{ctx.root_symbol} / {ctx.event_category}")
    components.provenance_row("Event Importance", f"{ctx.event_importance} ({ctx.event_importance_rule})")
    components.provenance_row("Objective Event Magnitude", ctx.objective_event_magnitude)
    st.caption(ctx.objective_event_magnitude_detail)
    components.provenance_row("Trend / Volatility", f"{ctx.trend} / {ctx.volatility}")
    components.provenance_row("Curve State", ctx.curve_state or "N/A")
    components.provenance_row(
        "Related-Market Confirmation", f"{ctx.related_market_confirming}/{ctx.related_market_total}",
    )
    components.provenance_row("Freshness", f"{ctx.freshness_bucket.value} ({ctx.freshness_seconds / 3600.0:.1f}h)")
    st.caption(f"Fingerprint: {ctx.fingerprint_hash}")

    if result.related_prior_event_occurrences:
        st.markdown(
            '<div class="aa-gate-title" style="margin-top:0.5rem;">RELATED PRIOR EVENT OCCURRENCES</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Real prior events of this same (root, event category) -- proves this KIND of event recurs, "
            "never that the historical market context (trend/volatility/curve) at those past timestamps "
            "matched today's; never fabricated."
        )
        for occ in result.related_prior_event_occurrences:
            st.caption(f"- {occ.observed_at.strftime('%Y-%m-%d')} -- {occ.headline} ({occ.match_reason})")

    st.markdown(
        '<div class="aa-gate-title" style="margin-top:0.5rem;">RANKED MECHANISM CANDIDATES</div>',
        unsafe_allow_html=True,
    )
    if not result.ranked_candidates:
        st.caption("No mechanism candidates from the deterministic library apply to this observation's category.")
    for c in result.ranked_candidates:
        tone = _CONTEXT_MATCH_BADGE_TONE.get(c.scores.context_match.value, "NOT_AVAILABLE")
        st.markdown(
            f"**#{c.retrieval_rank} &middot; {c.mechanism.value}** " + components.badge(tone, label=f"CONTEXT MATCH: {c.scores.context_match.value}"),
            unsafe_allow_html=True,
        )
        st.caption(c.mechanism_candidate.explanation)
        st.caption("Context signals: " + "; ".join(c.scores.context_match_reasons))

        sci_tone = _SCIENTIFIC_RELIABILITY_BADGE_TONE.get(c.scores.scientific_reliability.value, "NOT_AVAILABLE")
        coverage_tone = _RESEARCH_COVERAGE_BADGE_TONE.get(c.scores.research_coverage.value, "NOT_AVAILABLE")
        st.markdown(
            components.badge(sci_tone, label=f"SCIENTIFIC RELIABILITY: {c.scores.scientific_reliability.value}")
            + " " + components.badge(coverage_tone, label=f"RESEARCH COVERAGE: {c.scores.research_coverage.value}"),
            unsafe_allow_html=True,
        )
        st.caption(c.scores.research_coverage_detail)

        with st.expander(f"All six dimensions, kept separate ({c.mechanism.value})", expanded=False):
            st.caption(
                "Scientific Reliability, Context Match, Execution Robustness, Independent Replication, User "
                "Fit, and Research Coverage never combine into one score -- each stays independently "
                "inspectable."
            )
            components.provenance_row("Scientific Reliability", c.scores.scientific_reliability.value)
            components.provenance_row(
                "  by family", ", ".join(f"{k}={v}" for k, v in sorted(c.scores.scientific_reliability_detail.items())) or "n/a",
            )
            components.provenance_row("Context Match", c.scores.context_match.value)
            components.provenance_row(
                "Execution Robustness",
                ", ".join(f"{k}={v}" for k, v in sorted(c.scores.execution_robustness_detail.items())) or "no evidence",
            )
            components.provenance_row(
                "Independent Replication",
                c.scores.independent_replication.value + " -- no independent-replication evidence source "
                "exists in this repository yet (never inferred from related-experiment counts)",
            )
            components.provenance_row(
                "  Related experiment coverage (Research Coverage, not replication)",
                ", ".join(f"{k}={v}" for k, v in sorted(c.scores.related_experiment_counts.items())) or "n/a",
            )
            components.provenance_row(
                "User Fit",
                ", ".join(f"{k}={v}" for k, v in sorted(c.scores.user_fit_detail.items())) or "not measured",
            )
            components.provenance_row("Research Coverage", c.scores.research_coverage.value)

        if c.factors:
            with st.expander(f"Relevant factors ({c.mechanism.value})", expanded=False):
                for f in c.factors:
                    st.caption(f"- {f.concept}: {f.researchability.value} ({f.researchability_reason})")

        if c.strategy_evidence_summary:
            with st.expander(f"Strategy evidence ({c.mechanism.value})", expanded=False):
                for summary, alpha_id in zip(c.strategy_evidence_summary, c.alpha_ids, strict=True):
                    st.caption(f"- {summary}")
                    view_alpha_memory_button(alpha_id=alpha_id, key=f"{key_prefix}-context-retrieval-alpha-{alpha_id}")
                if c.repeated_failure_reason_codes:
                    st.caption(
                        "Repeated failure reasons: "
                        + ", ".join(f"{code} x{n}" for code, n in sorted(c.repeated_failure_reason_codes.items()))
                    )

        if c.research_gap_notes:
            with st.expander(f"Research gaps ({c.mechanism.value})", expanded=False):
                for note in c.research_gap_notes:
                    st.caption(f"- {note}")

    if result.rank_explanation:
        st.markdown('<div class="aa-gate-title" style="margin-top:0.5rem;">#1 VS #2</div>', unsafe_allow_html=True)
        st.caption(result.rank_explanation)

    if result.suggested_next_experiment is not None:
        nx = result.suggested_next_experiment
        st.markdown(
            '<div class="aa-gate-title" style="margin-top:0.5rem;">SUGGESTED NEXT EXPERIMENT</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{nx.mechanism.value}" + (f" -- {nx.concept}" if nx.concept else ""))
        st.caption(nx.rationale)
        st.caption("Why this (not predicted profit): " + nx.information_value_reason)


def view_alpha_memory_button(*, alpha_id: str, key: str) -> None:
    """Opens My Alpha's detail view for ``alpha_id`` on Learn -- purely a
    navigation hand-off (no research is triggered)."""
    if st.button("View Alpha Memory", key=key):
        from alpha_agent.ui.views import learn

        st.session_state["learn_landing_focus"] = "alpha_library"
        st.session_state["alpha_library_selected_id"] = alpha_id
        st.switch_page(st.Page(learn.render, url_path="learn"))


def _research_this(translation: ObservationTranslation) -> None:
    """Seeds the Agent research composer with this hypothesis (the SAME
    function the retired triage card called) and opens Ask, where the user
    reviews it and explicitly generates / runs it."""
    from alpha_agent.ui.views import agent

    agent.start_inline_research_from_translation(translation)
    st.switch_page(st.Page(agent.render, url_path="agent"))
