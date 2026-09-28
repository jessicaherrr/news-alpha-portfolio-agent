"""Agent -- the default landing page (Product Consolidation + Opportunity V1
campaign, Checkpoint C: "WHAT DESERVES MY ATTENTION?"; Market = "what is
happening?"; Research = "what does the evidence say?"; Paper = "what am I
currently testing?").

`render()`'s primary surface, in order: a compact hero ("What deserves
attention?" + one-line Market/Profile/Horizon workspace context), "Ask
Agentic Alpha" (the existing conversational composer), the unified
transcript, and Top Opportunities -- a small deterministic
`alpha_agent.opportunity` layer, never a BUY/SELL model, a probability-of-
profit figure, or a Scientific Verdict (see
`docs/PRODUCT_CONSOLIDATION_AND_OPPORTUNITY_V1.md`). The full objective/root/
preset hypothesis composer below is unchanged from the prior design, just
demoted into a collapsed "Propose a hypothesis directly (advanced)"
expander -- nothing was deleted, only reordered/demoted.

Every stage in that advanced composer is a REAL call through the same
unmodified Phase 16/17/18
pipeline `research.py`/`strategies.py` already used
(`llm_demo.propose_hypothesis` / `compile_hypothesis`,
`services.find_registry_evidence_for_compiled`). The free-text objective in
the composer -- edited or not -- is exactly what `ResearchContext.objective`
receives; a preset only fills in a suggestion. The Root selector narrows
`ResearchContext.market_universe` to exactly the one selected market -- the
SAME sanctioned per-call customization point every caller of
`build_context_for_objective` already uses, never a change to
`ResearchAgent`/`ResearchContext` semantics -- which is what makes the
compiled `StrategySpec`'s own `root_symbol` always equal to what the user
picked: `ResearchAgent.propose` already rejects (honestly, not silently) any
hypothesis whose declared universe is not a subset of `market_universe`, so a
root/hypothesis mismatch can never reach compilation in the first place (the
previous Research page's separate "Root for this Deep Research run" selector
was the actual bug this eliminates -- see `_render_execution_target`).

The Research Engine selector chooses which LLM transport runs: Offline /
Deterministic (`ScriptedLLMClient`, the repository default -- no network, no
cost) or Claude Research (the real `AnthropicClient`, requiring
`ANTHROPIC_API_KEY` already in the environment). Either way the LLM proposes a
`HypothesisSpec` and (via the compiler) a `StrategySpec`; it is never asked
for, and never supplies, PnL, a fill, a risk decision, a verdict, an FDR/DSR
number, or paper-trading eligibility.

Generate Hypothesis remains PROPOSAL-ONLY: it assembles `ResearchContext`
(with pre-proposal `FailureMemory`), calls `ResearchAgent`, compiles via
`StrategyCompilerAgent`, and inspects existing registry evidence -- it never
runs C++, never runs a new backtest, and never writes the registry. "Run This
Hypothesis" (below the compiled proposal) is the one explicit, separately
confirmed action that can -- it reuses `alpha_agent.ui.deep_research.
run_deep_research` verbatim (no duplicated orchestration logic), bound to the
already-compiled proposal's own root/family/fingerprint.

Optional delayed market context (IBKR, when connected) may be appended to
the agent's `ResearchContext.knowledge_base` as one clearly tagged
`OBSERVATIONAL_CONTEXT_ONLY` string -- see `alpha_agent.marketdata` and
CLAUDE.md's boundary notes. It never becomes a feature, never enters
validation, and never enters `experiment_identity`.
"""
from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from alpha_agent.opportunity.schemas import OpportunitySnapshot, OpportunityState
from alpha_agent.registry.holdout_guard import HOLDOUT_START
from alpha_agent.translation.schemas import ObservationTranslation
from alpha_agent.ui import (
    claude_conversation,
    components,
    deep_research,
    layout,
    llm_demo,
    market_context,
    opportunity_context,
    palette,
    panels,
    services,
)
from alpha_agent.ui.conversation_engine import ConversationBoundContext
from alpha_agent.ui.views import market, research

PAGE_TITLE = "Agent"

_TRANSCRIPT_KEY = "agent_transcript"
_CLAUDE_LABEL = "Claude Research"
_OFFLINE_LABEL = "Offline / Deterministic"
_FAILURE_VERDICTS = ("REJECT", "INCONCLUSIVE")


def render() -> None:
    """Layout order (Agent Experience Consolidation campaign, section 14;
    True Inline Research Workspace pass, task spec section 3): hero -> Top
    Opportunities -> Conversation Timeline -> composer. The composer is
    deliberately placed LAST in source order (item 16) so it reads as the
    bar that follows the conversation, with `.st-key-card-agent-composer-bar`
    (`layout.py`) making it sticky to the bottom of the viewport; it is
    still the ONE input on this page.

    The Research Workspace (composer + Research Actions + Research Result --
    `_render_composer`/`_render_research_actions`/`_render_execution_result`,
    all UNCHANGED, reused verbatim -- task spec section 3E: never duplicate
    research execution logic) now renders as a TRUE CONVERSATION ARTIFACT: a
    `type="research_workspace"` transcript turn, positioned at the exact
    point a research thread started (`_ensure_research_workspace_turn`), so
    it appears IN `_render_transcript()`'s own interleaved sequence --
    "User message -> Claude response -> Research Workspace artifact -> User
    follow-up -> Claude response" (task spec section 3A) -- not as a
    separate panel permanently pinned below the conversation. Before any
    thread exists at all, the SAME composer/actions/result functions render
    inside a collapsed "Propose a hypothesis directly (advanced)" expander
    here at the page bottom instead -- the one-time entry point for starting
    a thread from scratch; once a thread is active (task spec section 3D:
    `agent_research_origin` alone is sufficient evidence, even before any
    proposal exists), that bottom expander disappears -- its content now
    lives inline in the transcript, so nothing renders it twice."""
    layout.inject_style()
    layout.render_sidebar_nav(active="agent")
    layout.render_header(subtitle="Global research assistant across every certified market.")

    _render_attention_hero()
    _render_top_opportunities()
    _render_transcript()
    _render_ask_the_agent()

    if not _has_active_research_thread():
        with st.expander("Propose a hypothesis directly (advanced)", key="agent-research-expander"):
            _render_composer()
            _render_research_actions()
            _render_execution_result()

    layout.render_disclaimer()


def _has_active_research_thread() -> bool:
    """True from the moment a research thread starts (task spec section 3D:
    an explicit `Research This` hand-off, BEFORE any proposal exists) through
    to its result -- the single condition that decides whether the Research
    Workspace lives inline in the transcript (this function True) or in the
    one-time bottom "advanced" entry point (this function False)."""
    return bool(
        st.session_state.get("agent_research_origin")
        or st.session_state.get("lab_compiled")
        or st.session_state.get("ra_proposal")
        or st.session_state.get("agent_run_outcome")
    )


def _ensure_research_workspace_turn() -> None:
    """Find-or-create: appends ONE `type="research_workspace"` transcript
    turn the first time a research thread starts, and never again (task spec
    section 3C: "one research workflow -> one Research Workspace artifact").
    The turn carries NO research state of its own -- only the minimal
    occurrence reference `_append` always stamps (`event_id`/`at`); its
    renderer (`_render_research_workspace_card`) reads the SAME live
    `ra_proposal`/`lab_compiled`/`agent_last_evidence`/`agent_run_outcome`/
    `agent_research_origin` session state every other part of this page
    already reads (task spec section 3B: never copy full state into the
    transcript). This session's research-thread state is a single slot (a
    pre-existing architectural property, not introduced here -- starting a
    new thread already overwrites the old one's `ra_proposal`/`lab_compiled`/
    etc.), so a second research initiation intentionally reuses this SAME
    turn/position rather than creating a second, competing artifact that
    would immediately show identical (the newest) state."""
    if any(t["type"] == "research_workspace" for t in _transcript()):
        return
    _append("research_workspace")


# ---------------------------------------------------------------------------
# Product Consolidation + Opportunity V1 campaign, Checkpoint C -- the
# simple landing hero. No giant market terminal, no giant registry
# statistics, no raw experiment IDs, no technical architecture dump (task
# spec section 5): just "what deserves attention?", a compact one-line
# workspace context (market / profile / horizon), and the collapsed profile
# editor that was already here. The real market terminal lives on Market;
# the full registry inventory lives on Research; infrastructure status
# lives in the header (`layout.render_header`) and on Settings.
# ---------------------------------------------------------------------------


def _render_attention_hero() -> None:
    """No global "Market: X" framing (Agent Experience Consolidation
    campaign, sections 1-2; Sidebar IA pass, task spec section 2): the Agent
    answers "what deserves attention across markets?", a GLOBAL question, so
    its own header must not read as though it is scoped to whatever one
    product Market's own local selection happens to have. "Scope: All
    Research Markets" is a lightweight, read-only indicator, not a control --
    market selection lives entirely on the Market page
    (`market_selected_root`), unaffected by this page."""
    st.markdown(
        '<div style="font-size:1.3rem;font-weight:800;color:var(--aa-text);margin:0.1rem 0 0.2rem 0;">'
        "What deserves attention?</div>",
        unsafe_allow_html=True,
    )
    profile = panels.current_investor_profile()
    context_bits = ["Scope: <b>All Research Markets</b>"]
    context_bits.append(f"Profile: <b>{profile.risk_style.value}</b>")
    context_bits.append(f"Horizon: <b>{profile.holding_period.value}</b>")
    st.markdown(
        f'<div style="font-size:0.84rem;color:var(--aa-text-secondary);margin-bottom:0.5rem;">'
        f'{" &middot; ".join(context_bits)}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Your Profile", expanded=False):
        panels.render_profile_panel(key_prefix="agent")


# ---------------------------------------------------------------------------
# Opportunity V1 (Checkpoint C) -- the centerpiece of this campaign. A small,
# deterministic "what deserves investigation now?" layer, never a BUY/SELL
# model, a probability-of-profit figure, or a Scientific Verdict -- see
# `alpha_agent.opportunity.schemas` for the typed contract this renders
# verbatim (no card here computes anything of its own).
#
# FIRST PAINT (Product Acceptance Fix Pass, item 1/1B): a fresh multi-root
# Databento fetch is NOT run on every render -- that blocked first paint for
# 80-200+ seconds in real-browser verification, which fails this product's
# own "simple, calm" acceptance bar, and re-ran on every ordinary rerun
# (including simply navigating back to this page), not just a deliberate
# refresh. The last computed result is cached in `st.session_state` --
# Streamlit's own session state is the smallest reliable mechanism that
# survives a normal rerun AND real page navigation (one Python session
# backs every page `st.navigation` renders; `market_home.py`'s own
# `_CACHE_KEY` cross-page cache already relies on exactly this). Only the
# explicit "Refresh Opportunities" button ever calls
# `opportunity_context.top_opportunities` -- a bare render/rerun/navigation
# only ever READS that stored snapshot, never recomputes it, and an absent
# snapshot renders an honest "not loaded yet" empty state rather than
# fabricating or silently fetching data.
# ---------------------------------------------------------------------------

#: Moved to `services.OPPORTUNITIES_SNAPSHOT_KEY` so other views (the
#: Research Workflow's Decision Brief step) can read the same real, already-
#: refreshed snapshot without a circular import -- see that constant's own
#: docstring. Kept as a local alias so every existing reference below is
#: unchanged.
_OPPORTUNITIES_SNAPSHOT_KEY = services.OPPORTUNITIES_SNAPSHOT_KEY

_OPPORTUNITY_STATE_HELP = {
    OpportunityState.WATCH: "Interesting current conditions -- not yet enough evidence to justify a research workflow.",
    OpportunityState.INVESTIGATE: "Enough evidence to justify a research workflow.",
    OpportunityState.WAIT: "Potentially interesting, but a timing/risk condition argues against investigating right now.",
}


def _freshness_caption(observed_at: datetime) -> str:
    age_seconds = (datetime.now(UTC) - observed_at).total_seconds()
    if age_seconds < 60:
        age = f"{max(int(age_seconds), 0)}s ago"
    elif age_seconds < 3600:
        age = f"{int(age_seconds // 60)}m ago"
    else:
        age = f"{age_seconds / 3600:.1f}h ago"
    return f"Observed {observed_at.strftime('%Y-%m-%d %H:%M:%S UTC')} ({age})"


def _render_top_opportunities() -> None:
    components.section_header(
        "Top Opportunities",
        "Current research attention, produced by a deterministic Opportunity Engine -- never Claude, never a "
        "BUY/SELL signal, a probability of profit, or a Scientific Verdict. It becomes scientifically stronger "
        "only after historical execution, validation, and Registry adjudication (see Evidence & provenance "
        "on each card).",
    )

    snapshot_state = st.session_state.get(_OPPORTUNITIES_SNAPSHOT_KEY)
    c1, c2 = st.columns([4, 1])
    with c1:
        if snapshot_state is not None:
            st.caption(_freshness_caption(snapshot_state["observed_at"]))
        else:
            st.caption("Opportunity data not loaded yet.")
    with c2:
        refresh_clicked = st.button("Refresh Opportunities", key="agent-opp-refresh", width="stretch")

    if refresh_clicked:
        try:
            with st.spinner("Gathering opportunity evidence -- this fetches real, current Databento data and "
                             "can take a while..."):
                opportunities = opportunity_context.top_opportunities(limit=3)
        except Exception:  # noqa: BLE001 -- an evidence-assembly hiccup must never crash the landing page
            opportunities = []
        st.session_state[_OPPORTUNITIES_SNAPSHOT_KEY] = {
            "opportunities": opportunities, "observed_at": datetime.now(UTC),
        }
        st.rerun()

    snapshot_state = st.session_state.get(_OPPORTUNITIES_SNAPSHOT_KEY)
    if snapshot_state is None:
        components.empty_state(
            "Top Opportunities",
            "Opportunity data not loaded yet. Click Refresh Opportunities for current market evidence -- "
            "this page never fetches it automatically.",
            key="agent-opp-empty",
        )
        return

    opportunities = snapshot_state["opportunities"]
    if not opportunities:
        components.empty_state(
            "Top Opportunities",
            "No opportunity evidence was available at the last refresh -- this needs a connected "
            "market-data observation plane for at least one certified research market. Click Refresh "
            "Opportunities to try again, or check Market / Settings.",
            key="agent-opp-empty-stale",
        )
        return

    cols = st.columns(len(opportunities))
    for col, snap in zip(cols, opportunities, strict=True):
        with col:
            _render_opportunity_card(snap, key_prefix="agent-opp")


def _evidence_value(snap: OpportunitySnapshot, key: str) -> str | None:
    prefix = f"{key}="
    for ref in snap.evidence_refs:
        if ref.startswith(prefix):
            return ref[len(prefix):]
    return None


def _scientific_status_label(snap: OpportunitySnapshot) -> str:
    verdict = _evidence_value(snap, "research_verdict")
    return "VALIDATED (PASS)" if verdict == "PASS" else "NOT VALIDATED"


#: Human-readable labels for `OpportunitySnapshot.evidence_refs`' `key=value`
#: pairs (task spec section 1A: "do not list an input unless the Opportunity
#: engine actually used it" -- this only relabels refs that are ALREADY on
#: the typed snapshot, it never adds one).
_EVIDENCE_REF_LABELS = {
    "trend_state": "Trend",
    "volatility_state": "Volatility",
    "volume_context_state": "Volume",
    "session_position_state": "Session position",
    "curve_shape": "Curve / term structure",
    "peer_confirmation": "Related-market confirmation",
    "news_count_24h": "News (24h)",
}


def _format_evidence_ref(ref: str) -> str | None:
    """`key=value` -> a human-readable "Label: value" line. `None` for
    `research_verdict` -- that is RESEARCH EVIDENCE, already shown separately
    as "Scientific status" (task spec section 1: never re-presented here as
    if it were current-observation evidence the Opportunity Engine itself
    computed the setup/state from)."""
    key, _, value = ref.partition("=")
    if key == "research_verdict":
        return None
    return f"{_EVIDENCE_REF_LABELS.get(key, key)}: {value}"


def _render_opportunity_card(snap: OpportunitySnapshot, *, key_prefix: str) -> None:
    key = f"{key_prefix}-{snap.product}"
    with components.card(key):
        st.markdown(
            f'<div class="aa-metric-label">{snap.product} &middot; {snap.display_name.upper()}</div>'
            f'<div style="font-size:1.02rem;font-weight:700;color:var(--aa-text);margin-bottom:0.35rem;">'
            f'{snap.setup}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            components.badge(snap.opportunity_state.value, label=snap.opportunity_state.value),
            unsafe_allow_html=True,
        )
        st.caption(_OPPORTUNITY_STATE_HELP[snap.opportunity_state])

        st.markdown('<div class="aa-gate-title" style="margin-top:0.5rem;">WHY NOW</div>', unsafe_allow_html=True)
        for line in snap.why_now:
            st.caption(f"• {line}")

        st.markdown('<div class="aa-gate-title" style="margin-top:0.5rem;">CATALYST</div>', unsafe_allow_html=True)
        st.caption(snap.catalyst)

        st.markdown('<div class="aa-gate-title" style="margin-top:0.5rem;">RISKS</div>', unsafe_allow_html=True)
        for line in snap.risks:
            st.caption(f"• {line}")

        with st.expander("What changes this view?", expanded=False):
            st.markdown("**Strengthens if:**")
            for line in snap.strengthen_if:
                st.caption(f"• {line}")
            st.markdown("**Invalidates if:**")
            for line in snap.invalidate_if:
                st.caption(f"• {line}")

        st.markdown('<div class="aa-gate-title" style="margin-top:0.5rem;">RESEARCH</div>', unsafe_allow_html=True)
        status = _scientific_status_label(snap)
        st.markdown(
            f'<span style="font-size:0.82rem;color:var(--aa-text-secondary);">Scientific status: '
            f'<b style="color:{palette.GREEN if "VALIDATED (PASS)" == status else palette.TEXT_SECONDARY}">'
            f'{status}</b></span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Next action: {snap.next_action}")

        with st.expander("Evidence & provenance", expanded=False):
            st.caption(
                "Generated by: **Opportunity Engine** (deterministic, rule-based) -- Claude is never the "
                "source of this card's setup, trend, curve, catalyst, or evidence."
            )
            st.caption(f"Observation: {snap.observed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            st.caption(f"Data freshness: {_freshness_caption(snap.observed_at)}")
            st.markdown("**Evidence used**")
            for ref in snap.evidence_refs:
                formatted = _format_evidence_ref(ref)
                if formatted:
                    st.caption(f"• {formatted}")
            st.caption(
                f"Scientific status: **{status}** (sourced from the Experiment Registry -- a separate, "
                "authoritative system, not the Opportunity Engine)."
            )
            st.caption(
                "Opportunity = current research attention -- not a trade signal, a predicted return, a "
                "validated strategy, or a scientific PASS."
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Research This", key=f"{key}-research-this", width="stretch"):
                _start_inline_research_from_opportunity(snap)
                st.rerun()
        with c2:
            if st.button("Open Market", key=f"{key}-open-market", width="stretch"):
                # Explicit Agent -> Market hand-off (task spec section 1E):
                # sets Market's OWN local selected-product state, never
                # Agent's global evidence scope -- see `market.py`'s
                # `market_selected_root` (Market-page-local, not a hidden
                # global selector).
                market.open_explorer(snap.product)


def start_inline_research_from_translation(translation: ObservationTranslation) -> None:
    """"Research This" on a Research Translation card (prompt 1 section 13):
    seeds the SAME existing advanced research composer, preserving the
    originating observation/mechanism/factor/researchability/falsification
    for display on `agent_research_origin` -- never mutates the source
    translation. The objective text deliberately omits `observed_at` (see
    `_origin_vintage_fields`'s docstring above for why)."""
    h = translation.hypothesis
    if h is None:
        return
    obs = translation.observation
    # The specific factor `_build_hypothesis` actually used -- never just
    # `factor_candidates[0]` (that candidate may be a NOT_EXECUTABLE/
    # DATA_MISSING one the hypothesis was never built from; `title` is always
    # `f"{factor.concept} in {root}"`, so matching on that prefix recovers
    # the real one rather than mislabeling this hand-off's researchability).
    used_factor = next((f for f in translation.factor_candidates if h.title.startswith(f.concept)), None)
    mechanism = (
        next((m for m in translation.mechanism_candidates if m.mechanism == used_factor.mechanism), None)
        if used_factor
        else None
    ) or (translation.mechanism_candidates[0] if translation.mechanism_candidates else None)
    st.session_state["agent-root-synced-scenario"] = _current_scenario_key()
    st.session_state["agent-root"] = obs.root_symbol
    st.session_state["agent-objective"] = (
        f"Investigate {h.title}. Mechanism: {h.economic_mechanism} Signal: {h.signal_description} "
        f"Falsification: {h.falsification_test}"
    )
    st.session_state["agent_research_origin"] = {
        "source": "observation_translation",
        "root": obs.root_symbol,
        "observation_summary": obs.summary,
        "observation_source": obs.source,
        "mechanism": mechanism.mechanism.value if mechanism else None,
        "factor_concept": h.title,
        "researchability": used_factor.researchability.value if used_factor else None,
        "falsification_test": h.falsification_test,
        "evidence_refs": list(obs.evidence_refs),
        "observed_at": obs.observed_at.isoformat(),
        "origin_vintage": obs.origin_vintage,
        "holdout_eligible": obs.holdout_eligible,
    }
    st.session_state["agent-research-expander"] = True
    _ensure_research_workspace_turn()


# ---------------------------------------------------------------------------
# inline Research Workspace hand-off (sections 19/20/25/26) -- REUSES the
# existing advanced research composer/pipeline below verbatim; this is only
# the UI seam that keeps the user on Agent instead of navigating away.
#
# ORIGIN OBSERVATION vs. SCIENTIFIC RESEARCH OBJECTIVE (Agent Evidence +
# Research Provenance acceptance pass, task spec section 2A): a live
# Opportunity's `observed_at` is provenance about WHERE this thread came
# from, never scientific input. It is preserved verbatim on
# `agent_research_origin` for display, but it must never be embedded in the
# `objective` text handed to `ResearchContext` -- that text is holdout-
# scanned verbatim (`assert_no_holdout_market_data`), and a live
# observation's own wall-clock is always "now". Once "now" reads
# >= 2025-01-01, embedding it there previously tripped a `HoldoutAccessError`
# on every single "Research This" click (task spec section 2) -- a false
# positive, not a real holdout violation: the setup/why_now text itself
# never names a date, so the objective never needed the timestamp at all.
# ---------------------------------------------------------------------------

#: The one frozen boundary this module compares against for DISPLAY-ONLY
#: provenance (task spec section 2B) -- imported, never redefined, from the
#: same `alpha_agent.registry.holdout_guard` constant the actual guard uses,
#: so this can never silently drift from the real boundary. This computation
#: never touches `ResearchContext`/the registry/`ValidationSpec` -- it only
#: decides what the Research Workspace HEADER says about where a thread's
#: seed evidence came from.
_HOLDOUT_START_DT = datetime.fromisoformat(HOLDOUT_START).replace(tzinfo=UTC)


def _origin_vintage_fields(observed_at: datetime) -> dict:
    """Typed, display-only post-holdout-origin marker. `origin_vintage` is
    the calendar date of the seed observation; `holdout_eligible` is True
    only when that observation predates the locked 2025 holdout boundary --
    honestly False for every live/current-market Opportunity while this
    sandbox's wall clock reads 2026. This never changes what historical
    (2018-2024) research is available (task spec section 2C) -- it only
    prevents this session from silently implying an untouched-2025-holdout
    claim for a hypothesis seeded from post-boundary information."""
    return {
        "origin_vintage": observed_at.date().isoformat(),
        "holdout_eligible": observed_at < _HOLDOUT_START_DT,
    }


def _start_inline_research_from_opportunity(snap: OpportunitySnapshot) -> None:
    """"Research This" on an Opportunity Card (item 25): seeds the SAME
    existing advanced research composer with this opportunity's root and an
    objective built from it, and forces that composer open, in place, on
    Agent -- no `st.switch_page`, no automatic Generate Hypothesis/Fast
    Screen/Freeze/Validate (those stay explicit user actions below). Also
    sets `agent-root-synced-scenario` to the CURRENT preset so that
    preset's own root-sync guard (see `_render_composer`) does not
    immediately overwrite the root this just set. Preserves the origin
    observation (product/setup/opportunity_state/observed_at/why_now) for
    display -- never mutates the source Opportunity snapshot itself. The
    objective text deliberately omits `observed_at` -- see this section's
    docstring above for why."""
    st.session_state["agent-root-synced-scenario"] = _current_scenario_key()
    st.session_state["agent-root"] = snap.product
    st.session_state["agent-objective"] = (
        f"Investigate the {snap.setup} setup noted for {snap.product} in Top Opportunities "
        f"(opportunity state: {snap.opportunity_state.value}). Why now: {'; '.join(snap.why_now)}"
    )
    st.session_state["agent_research_origin"] = {
        "source": "opportunity", "product": snap.product, "setup": snap.setup,
        "opportunity_state": snap.opportunity_state.value, "observed_at": snap.observed_at.isoformat(),
        "why_now": list(snap.why_now),
        **_origin_vintage_fields(snap.observed_at),
    }
    st.session_state["agent-research-expander"] = True
    _ensure_research_workspace_turn()


def _start_inline_research_from_text(hypothesis_text: str) -> None:
    """"Research This Hypothesis" on a Claude-proposed, explicitly
    NOT-VALIDATED hypothesis in conversation (item 26): seeds the SAME
    advanced composer's Objective with Claude's own text and forces it open.
    Nothing is compiled or executed here -- the user still has to read,
    optionally edit, and explicitly click Generate Hypothesis, which is the
    one thing that actually invokes the real typed `ResearchAgent`. The
    conversation this hypothesis came from is itself happening "now" --
    the same origin-vintage marker applies (task spec section 2B)."""
    st.session_state["agent-root-synced-scenario"] = _current_scenario_key()
    st.session_state["agent-objective"] = hypothesis_text
    now = datetime.now(UTC)
    st.session_state["agent_research_origin"] = {
        "source": "claude_hypothesis", "observed_at": now.isoformat(), **_origin_vintage_fields(now),
    }
    st.session_state["agent-research-expander"] = True
    _ensure_research_workspace_turn()


# ---------------------------------------------------------------------------
# transcript state
# ---------------------------------------------------------------------------


def _transcript() -> list[dict]:
    return st.session_state.setdefault(_TRANSCRIPT_KEY, [])


def _append(turn_type: str, **data) -> None:
    """Append one transcript turn, stamping it with a stable UI OCCURRENCE
    identifier -- `event_id` -- distinct from any scientific/research-object
    identity the turn carries (`hypothesis_id`, `strategy_fingerprint`,
    `experiment_id`). The same scientific object legitimately appears in
    multiple turns (e.g. the same objective sent twice in scripted/
    deterministic mode reproduces the exact same hypothesis_id); `event_id` is
    what makes each occurrence's Streamlit element keys unique, never the
    scientific identity itself.

    `event_id` is derived from the turn's position at append time -- computed
    ONCE, here, when the turn is appended -- and then stored verbatim on the
    turn dict. It is never regenerated during rendering/reruns (that would
    make Streamlit element identity unstable across reruns), and never a
    random id (`uuid.uuid4()` would do the same)."""
    turns = _transcript()
    event_id = f"agent-event-{len(turns):06d}"
    turns.append({"type": turn_type, "event_id": event_id, "at": components.now_utc_str(), **data})


def _event_id(turn: dict, index: int) -> str:
    """Backward-compatible fallback for a transcript entry appended before
    `event_id` existed (e.g. a session_state carried over from an older
    version of this page): derive a deterministic id from the turn's stable
    position in the transcript list instead. Never used for a freshly
    appended turn, which always already has one from `_append`."""
    return turn.get("event_id") or f"agent-legacy-{index:06d}"


# ---------------------------------------------------------------------------
# Ask the Agent -- conversational, read-only (Release UX Part A). Free-text
# questions route through the typed intent router
# (`alpha_agent.agents.conversation.classify_intent`) and are answered by
# `alpha_agent.ui.conversation_engine`, using the SAME real services this
# page's Generate Hypothesis pipeline already uses. This is a QUESTION
# surface, never an execution one: no call in this section can run a
# backtest, freeze a manifest, or write the registry -- an action-shaped
# question (e.g. "run fast screen") only ever produces a suggestion, never a
# side effect (see `conversation_engine`'s own module docstring / static test).
# ---------------------------------------------------------------------------


def _current_bound_context() -> ConversationBoundContext:
    """Whatever this conversation already has assembled -- the SAME session
    state `_build_research_details_target` hands to Research Details, so a
    conversational answer is grounded in exactly the object on screen.

    Deliberately sourced ONLY from LOCAL research-thread state
    (`agent_last_root`, set only by an explicit Generate Hypothesis /
    Research This action on this page) -- never from Market's own local
    selected product (`st.session_state["market_selected_root"]`), which is
    a Market-page-local browsing concept and must never leak into what the
    Agent conversation treats as "the market this thread is
    about" (Agent Experience Consolidation campaign, sections 1-3)."""
    proposal = st.session_state.get("ra_proposal") or {}
    return ConversationBoundContext(
        root=st.session_state.get("agent_last_root"),
        hypothesis=proposal.get("hypothesis") if proposal.get("accepted") else None,
        compiled=st.session_state.get("lab_compiled"),
        evidence=st.session_state.get("agent_last_evidence"),
    )


def _render_ask_the_agent() -> None:
    """The ONE composer for this page (item 15/16: visually primary, no
    duplicate input, no model/temperature/token controls) -- rendered AFTER
    the conversation timeline in source order so it reads as "the bar that
    follows the conversation" rather than a form sitting above it; `layout`'s
    `.st-key-card-agent-composer-bar` CSS makes this container sticky to the
    bottom of the viewport. "Claude · Connected" stays small/secondary next
    to the "Ask Agentic Alpha" label, never a model picker or settings row."""
    claude_connected = claude_conversation.is_connected()
    st.markdown(
        '<div class="aa-gate-title" style="display:inline-block;margin-right:0.6rem;">Ask Agentic Alpha</div>'
        + components.badge(
            "OK" if claude_connected else "OFFLINE",
            label="Claude · Connected" if claude_connected else "Claude · Not Connected",
        ),
        unsafe_allow_html=True,
    )
    with components.card("agent-composer-bar"):
        c1, c2 = st.columns([5, 1])
        with c1:
            question = st.text_input(
                "Ask Agentic Alpha", key="agent-ask-input", label_visibility="collapsed",
                placeholder="Ask about current opportunities, markets, or research...",
            )
        with c2:
            ask_clicked = st.button("Ask", key="agent-ask-send", type="primary", width="stretch")

    if ask_clicked and question.strip():
        turns_before = _transcript()
        has_history = any(t["type"] in ("conversation", "hypothesis") for t in turns_before)
        _append("user", text=question)
        response, engine, error = claude_conversation.handle_turn(
            question,
            context=_current_bound_context(),
            has_history=has_history,
            transcript=_transcript(),
            opportunity_snapshot=st.session_state.get(_OPPORTUNITIES_SNAPSHOT_KEY),
        )
        if error:
            _append("error", stage="Claude", text=error)
        elif not (response.text or "").strip():
            # An empty reply (e.g. the model's response was cut off before any
            # text content block, or returned only a non-text block) must
            # never render as an ordinary, silently-blank Agentic Alpha card
            # (task spec section 6) -- an honest retry prompt instead.
            _append(
                "error", stage="Claude",
                text="Claude returned an empty response -- this is a transport/response issue, not an "
                     "answer. Please try asking again.",
            )
        else:
            _append(
                "conversation", intent=response.intent.value, text=response.text,
                suggested_action=response.suggested_action, citations=list(response.citations),
                engine=engine, proposed_hypothesis=response.proposed_hypothesis,
            )
        st.rerun()


_CONVERSATION_ENGINE_LABEL = {
    "live": ("CLAUDE", "OK"),
    "scripted": ("DETERMINISTIC", "OK"),
}


def _render_conversation_engine_badge(engine: str | None) -> None:
    """Which engine actually answered this turn -- Claude (live model call,
    grounded context) or the always-available deterministic engine
    (`alpha_agent.ui.conversation_engine`). Mirrors `_render_engine_badge`'s
    own defaulting pattern for a pre-existing transcript turn with no
    recorded `engine` (renders as DETERMINISTIC, the only engine that
    existed before this field was added)."""
    label, tone = _CONVERSATION_ENGINE_LABEL.get(engine or "scripted", ("DETERMINISTIC", "OK"))
    st.markdown(components.badge(tone, label=label), unsafe_allow_html=True)


def _render_conversation_turn(turn: dict, event_id: str) -> None:
    """No raw internal intent label (e.g. "GENERAL_QUANT_QUESTION") in the
    primary header (item 18/37) -- this reads like developer telemetry to a
    professional user. The header simply identifies the responder; the
    internal intent classification remains available, verbatim, under a
    "Details" expander for anyone who wants it -- the intent router itself is
    unchanged, only how prominently its label is displayed."""
    with components.card(f"agent-conv-{event_id}"):
        st.markdown('<div class="aa-gate-title">Agentic Alpha</div>', unsafe_allow_html=True)
        _render_conversation_engine_badge(turn.get("engine"))
        st.write(turn["text"])
        if turn.get("citations"):
            st.caption("Sources: " + "; ".join(turn["citations"]))
        if turn.get("suggested_action"):
            st.caption(
                f"Suggested action: **{turn['suggested_action'].replace('_', ' ').title()}** -- "
                "use the explicit action for this on the relevant page; a chat message never runs it."
            )
        # A hypothesis Claude proposed in free text (item 26) -- exposed ONLY
        # when the response carries the typed `proposed_hypothesis` field
        # (task spec section 5), never inferred from a string search over the
        # displayed text (e.g. "NOT VALIDATED" appearing anywhere in it).
        # Clicking this only SEEDS the existing advanced composer's
        # Objective; it never compiles or executes anything itself, and the
        # field means only "Claude proposed a hypothesis" -- never validated,
        # accepted, or scientifically supported.
        proposed_hypothesis = turn.get("proposed_hypothesis")
        if proposed_hypothesis and st.button(
            "Research This Hypothesis", key=f"agent-research-hyp-{event_id}"
        ):
            _start_inline_research_from_text(proposed_hypothesis)
            st.rerun()
        with st.expander("Details", expanded=False):
            st.caption(f"Internal intent classification: `{turn['intent']}`")


# ---------------------------------------------------------------------------
# composer -> runs the real pipeline, once, synchronously
# ---------------------------------------------------------------------------


def _mode_picker() -> str:
    """Research Engine selector. Reuses the exact `llm_demo.build_llm_client` /
    `AnthropicClient` mechanism the rest of the app uses -- no second LLM
    integration. Safe repository default is Offline / Deterministic; Live
    Claude Research requires `ANTHROPIC_API_KEY` in the environment Streamlit
    was launched from and is never silently downgraded to scripted on
    failure -- a broken key surfaces as an honest error turn, not a
    fabricated proposal."""
    label = st.radio(
        "Research Engine", options=[_OFFLINE_LABEL, _CLAUDE_LABEL],
        index=0, key="agent-mode", horizontal=True,
        help="Offline / Deterministic replays a canned, schema-valid response through the real "
             "ResearchAgent/StrategyCompilerAgent -- no network call, no cost. Claude Research builds "
             "AnthropicClient() with no key argument; the SDK reads ANTHROPIC_API_KEY from your "
             "environment -- this app never reads, stores, or displays that key.",
    )
    if label == _CLAUDE_LABEL:
        st.caption("Claude Research requires `ANTHROPIC_API_KEY` in the environment Streamlit was launched from.")
        return llm_demo.LIVE_MODE
    return llm_demo.SCRIPTED_MODE


def _preset_root(scenario_key: str) -> str | None:
    sc = llm_demo.scenario(scenario_key)
    return sc.plan.get("template", {}).get("root_symbol") or sc.plan.get("blueprint", {}).get("root_symbol")


def _current_scenario_key() -> str:
    """Peeks the preset selectbox's OWN persisted value without rendering it
    -- the same "peek a keyed widget's state before it re-renders" pattern
    `_mode_picker`'s caller already relies on (Streamlit updates a changed
    widget's session_state BEFORE the script reruns, so this is never
    stale). Shared by `_render_composer` and the inline "Research This" /
    "Research This Hypothesis" hand-offs, which need to know which preset's
    root-sync guard they must satisfy when they seed `agent-root` directly."""
    scenario_options = {s.label: s.key for s in llm_demo.SCENARIOS}
    default_label = next(iter(scenario_options))
    current_label = st.session_state.get("agent-scenario-select", default_label)
    return scenario_options.get(current_label, scenario_options[default_label])


#: Fixed execution/validation authority statement (task spec sections 8/9):
#: never conditioned on anything -- Quant Core and the Validation Engine are
#: always the authority for every research thread on this page, regardless
#: of which Hypothesis Mode is selected.
_EXECUTION_AUTHORITY_LINE = "Execution: Quant Core (C++) · Validation: Validation Engine"


def _research_seed_scientific_status() -> str:
    """The CURRENT thread's own scientific status -- reads the SAME session
    state `_render_execution_result`/`_render_evidence_cards` already read,
    so this header can never disagree with the detailed cards below it. Real
    evidence only: "NOT VALIDATED" is the honest default before any run/
    evidence exists, never a fabricated status."""
    outcome = st.session_state.get("agent_run_outcome")
    if outcome is not None and outcome.accepted and outcome.report and outcome.report.member_results:
        mr = outcome.report.member_results[0]
        return mr.final_verdict.value if mr.final_verdict else "NOT_ADJUDICATED"
    evidence = st.session_state.get("agent_last_evidence")
    if evidence and evidence.get("result"):
        return evidence["result"].get("headline_verdict") or "NOT_ADJUDICATED"
    return "NOT VALIDATED"


def _render_research_seed(origin: dict | None) -> None:
    """Research Workspace provenance header (task spec sections 2D/8): why
    this thread exists, where it came from, and which engine is
    authoritative for each stage -- shown BEFORE the composer itself.
    Reuses only already-computed session state; introduces no new judgment.
    Compact by design (task spec: "do not overfill the card")."""
    if origin and origin.get("source") == "opportunity":
        origin_line = (
            f"Opportunity Engine · {origin['product']} · {origin['setup']} (observed {origin['observed_at']})"
        )
    elif origin and origin.get("source") == "claude_hypothesis":
        origin_line = "Conversation · a hypothesis Claude proposed (Claude never validates it)"
    elif origin and origin.get("source") == "observation_translation":
        origin_line = (
            f"Research Translation · {origin['root']} · {origin.get('mechanism') or 'mechanism unspecified'} "
            f"(observation: {origin.get('observation_summary', '')})"
        )
    else:
        origin_line = "Manual entry on this page"

    with components.card("agent-research-seed"):
        st.markdown('<div class="aa-gate-title">Research Seed</div>', unsafe_allow_html=True)
        st.caption(f"Origin: {origin_line}")
        st.caption(f"{_EXECUTION_AUTHORITY_LINE} · Scientific status: {_research_seed_scientific_status()}")
        if origin and origin.get("holdout_eligible") is False:
            w = services.research_window()
            st.caption(
                "Holdout eligibility: **NOT ELIGIBLE FOR 2025 HOLDOUT** -- this thread is seeded from a "
                f"contemporary observation ({origin.get('origin_vintage', 'current')}), so it cannot honestly "
                f"claim an untouched 2025 holdout test. Historical research remains available (research "
                f"{w['research']}, validation {w['validation']}) -- this is an information-set boundary, "
                "not a failed strategy."
            )


def _render_how_this_works() -> None:
    """Compact pipeline explanation (task spec sections 3/3A) -- what each
    stage means and who is authoritative for it, in user-facing names.
    Collapsed by default; never forced open."""
    with st.expander("How this works", expanded=False):
        st.markdown(
            "Research question &rarr; **Hypothesis Generator** &rarr; **Strategy Compiler** &rarr; "
            "**Quant Core** (backtest / execution) &rarr; **Validation Engine** &rarr; **Experiment Registry**",
            unsafe_allow_html=True,
        )
        st.caption(
            "**Offline / Deterministic** replays a canned, schema-valid response through the real pipeline "
            "below -- no network call, no cost. **Claude Research** lets Claude reason over the supplied "
            "evidence and propose a hypothesis in the same closed structure. Either way, this stage only "
            "PROPOSES: it never computes a fill, a PnL, a Sharpe ratio, or a scientific verdict."
        )
        st.caption(
            "**Strategy Compiler** turns an accepted hypothesis into a typed StrategySpec inside a closed "
            "DSL -- Claude can never write arbitrary executable strategy code. **Quant Core** (C++) is the "
            "sole authority for fills, PnL, and execution economics. **Validation Engine** is the sole "
            "authority for a PASS / REJECT / INCONCLUSIVE verdict, under the frozen reliability policy "
            "(multiple-testing correction, cost sensitivity, regime stability). **Experiment Registry** "
            "stores the authoritative, append-only evidence -- nothing here can be overridden by a chat answer."
        )


def _render_composer() -> None:
    components.section_header(
        "Research",
        "Propose a falsifiable hypothesis, compile it, and (only on explicit confirmation) run it.",
    )
    origin = st.session_state.get("agent_research_origin")
    _render_research_seed(origin)
    _render_how_this_works()
    with components.card("agent-composer"):
        mode = _mode_picker()
        st.caption(
            "Either engine only PROPOSES a hypothesis -- Quant Core computes fills/PnL and the Validation "
            "Engine decides PASS/REJECT/INCONCLUSIVE. Neither engine's choice is a scientific result."
        )
        scenario_options = {s.label: s.key for s in llm_demo.SCENARIOS}
        scenario_key = _current_scenario_key()
        sc = llm_demo.scenario(scenario_key)

        objective = st.text_area(
            "Objective", value=sc.objective, height=100, key="agent-objective",
            help="This exact text is sent to the Research Agent. Editing it changes the objective it receives.",
        )

        universe = list(services.approved_universe())
        preset_root = _preset_root(scenario_key)
        # Root FOLLOWS the active preset by default (so the very first Send
        # on a fresh session works, and switching presets never leaves a
        # stale, mismatched Root behind) but stays user-overridable in
        # between -- only resynced when the preset itself changes.
        if st.session_state.get("agent-root-synced-scenario") != scenario_key:
            st.session_state["agent-root"] = preset_root if preset_root in universe else universe[0]
            st.session_state["agent-root-synced-scenario"] = scenario_key
        selected_root = st.selectbox(
            "Research Root", universe, key="agent-root",
            help="The research target for this proposal -- distinct from the sidebar's market browser. "
                 "The agent's approved market universe for THIS call is narrowed to exactly this one root "
                 "-- ResearchAgent already rejects (honestly) any hypothesis outside its given universe, "
                 "so the compiled StrategySpec's root always matches what you pick here; you never "
                 "re-enter an execution root separately later.",
        )

        with st.expander("Preset objective (optional starting point)", expanded=False):
            st.caption(
                "Picking a preset only fills in a suggested objective/root above -- it is a starting "
                "point, not a fixed choice. The text you send (edited or not) is exactly what reaches "
                "the agent."
            )
            st.selectbox(
                "Preset research objective", list(scenario_options.keys()), key="agent-scenario-select",
            )

        _render_compact_context(selected_root)

        c1, c2 = st.columns([1, 3])
        with c1:
            send = st.button("Generate Hypothesis", type="primary", key="agent-send", width="stretch")
        with c2:
            if st.button("Clear conversation", key="agent-clear"):
                st.session_state[_TRANSCRIPT_KEY] = []
                st.session_state["agent-run-confirm-pending"] = False
                # A full reset of the research thread, not just the visible
                # transcript -- `_has_active_research_thread()` now gates
                # whether the Research Workspace renders at all (True Inline
                # Research Workspace pass, task spec section 3), so leaving
                # any of this set would strand the page with neither the
                # inline artifact (no transcript turn left to anchor it) nor
                # the bottom "advanced" entry point (still masked as active).
                for key in (
                    "ra_proposal", "lab_compiled", "agent_last_evidence", "agent_last_objective",
                    "agent_last_root", "agent_run_outcome", "agent_research_origin",
                ):
                    st.session_state.pop(key, None)
                st.rerun()

    if send:
        _run_pipeline(scenario_key=scenario_key, objective=objective, mode=mode, root=selected_root)
        st.rerun()


def _render_compact_context(selected_root: str) -> None:
    """Compact Research Context summary (product refactor, section 2D): a
    one-line digest always visible, full detail behind one expander -- never
    a large context block shown by default. Failure memory shown here is the
    SAME real, already-computed `relevant_failure_memory` sweep
    `build_context_for_objective` assembles into `ResearchContext` before any
    proposal -- this is a read of what the agent will see, not a second,
    independent judgment."""
    universe = services.approved_universe()
    reg = services.registry_summary()
    features = services.feature_catalog()
    pre_fm = llm_demo.relevant_failure_memory(market_universe=(selected_root,))
    n_prior_failures = sum(
        1 for fm in pre_fm for p in fm.prior_experiments if p.headline_verdict in _FAILURE_VERDICTS
    )
    holdout_badge = components.badge("LOCKED", label="LOCKED")

    st.markdown(
        f'<div style="margin-top:0.3rem;font-size:0.86rem;color:var(--aa-text-secondary);line-height:1.6;">'
        f'<b>Research Context</b> &middot; {len(universe)} markets &middot; '
        f'{reg["authoritative_statistical_hypotheses"]} authoritative hypotheses &middot; '
        f'{len(features)} feature kinds<br>'
        f'Relevant prior failures ({selected_root}): <b>{n_prior_failures}</b> &middot; Holdout: {holdout_badge}'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("View context", expanded=False):
        st.caption(f"Approved market universe: {', '.join(universe)}")
        st.caption(
            f"Registry digest: {reg['authoritative_statistical_hypotheses']} authoritative hypotheses, "
            f"identity schema `{reg['identity_schema']}`."
        )
        st.caption(f"Feature catalog: {len(features)} registered kinds.")
        st.markdown(f"**Failure memory for `{selected_root}`** (assembled BEFORE proposing)")
        if not pre_fm:
            st.caption("No prior experiments for this root yet -- a proposal here would be genuinely new.")
        for fm in pre_fm:
            v = fm.verdict_counts
            st.write(
                f"`{fm.query.get('strategy_family')}`: "
                + (", ".join(f"{k}={n}" for k, n in v.items()) or "no adjudicated verdict yet")
            )
        st.caption("Holdout: 2025 data is never accessed, queried, or reported by this pipeline (fail-loud guard).")


def _run_pipeline(*, scenario_key: str, objective: str, mode: str, root: str) -> None:
    """The real Phase 16 -> 17 -> registry-lookup pipeline, run once per
    Generate Hypothesis click. `mode` is `llm_demo.SCRIPTED_MODE` (offline,
    deterministic, zero cost -- the repository default) or
    `llm_demo.LIVE_MODE` (a real Claude call through the unmodified
    `AnthropicClient`). `objective` is the ACTUAL text the user submitted
    (edited or not). `root` narrows `market_universe` to exactly one market
    -- see this module's docstring for why that eliminates the root/family
    execution mismatch that used to require a separate "Deep Research root"
    selector. Failure memory is looked up BEFORE the hypothesis exists:
    `llm_demo.propose_hypothesis` builds it into the `ResearchContext` handed
    to the agent and returns it back unchanged, so what is rendered here is
    provably what the agent saw, not a second independent lookup. This path
    NEVER runs C++ and NEVER writes the registry -- see `_execute_run_this_hypothesis`
    for the one explicit action that does."""
    universe = (root,)
    _append("user", text=objective)
    _ensure_research_workspace_turn()
    st.session_state["agent_last_objective"] = objective
    st.session_state["agent_last_root"] = root
    # A fresh proposal must not inherit a stale confirmation prompt or
    # execution-result summary from a PREVIOUS compiled hypothesis.
    st.session_state["agent-run-confirm-pending"] = False
    st.session_state.pop("agent_run_outcome", None)

    market_note = market_context.observational_context_note(root)
    proposal, error, fm = llm_demo.propose_hypothesis(
        mode=mode, scenario_key=scenario_key, universe=universe, objective=objective,
        extra_knowledge=(market_note,) if market_note else (),
    )
    fm_data = [f.model_dump(mode="json") for f in fm]
    if fm:
        _append("failure_memory", data=fm_data, pre_proposal=True, root=root)

    if error or not proposal:
        _append("error", stage="Research Agent", text=error or "no proposal")
        return
    if not proposal.accepted:
        _append("error", stage="Research Agent",
                text=f"{proposal.rejection_code}: {proposal.rejection_detail}")
        return

    h = proposal.hypothesis
    st.session_state["ra_proposal"] = proposal.model_dump(mode="json")
    st.session_state["ra_scenario_key"] = scenario_key
    _append("hypothesis", data=h.model_dump(mode="json"), engine=mode, fm=fm_data)
    panels.log_activity("OK", "Hypothesis generated", h.title)

    compiled, cerror = llm_demo.compile_hypothesis(
        mode=mode, scenario_key=scenario_key, hypothesis=h, universe=universe,
    )
    if cerror or not compiled or not compiled.accepted:
        st.session_state["lab_compiled"] = None
        _append("error", stage="Strategy Compiler Agent",
                text=cerror or (compiled.rejection_detail if compiled else "no compiled spec"))
        return

    cd = compiled.model_dump(mode="json")
    st.session_state["lab_compiled"] = cd
    _append("compiled", data=cd, engine=mode)
    panels.log_activity("OK", "Strategy compiled", cd["strategy_fingerprint"][:24] + "...")

    evidence = services.find_registry_evidence_for_compiled(cd)
    st.session_state["agent_last_evidence"] = evidence
    _append(
        "evidence", data=evidence, family=cd.get("family_key"), root=cd.get("root_symbol"),
        strategy_fingerprint=cd.get("strategy_fingerprint"), build_mode=cd.get("build_mode"),
    )

    suggestion = _bounded_next_suggestion(fm, evidence)
    if suggestion:
        _append("suggestion", text=suggestion)


def _bounded_next_suggestion(
    fm: tuple, evidence: dict | None
) -> str | None:
    """A deterministic, template-based next-step suggestion drawn ONLY from
    already-computed real evidence (pre-proposal failure-memory lessons, the
    registry verdict just looked up) -- never a new LLM call, never a new
    judgment. The agent may explain the frozen system's own output; it may not
    add one."""
    if evidence and evidence.get("result"):
        verdict = evidence["result"]["headline_verdict"]
        if verdict == "PASS":
            return ("This exact strategy already holds an authoritative PASS. Consider checking "
                    "Paper Trading eligibility on the Experiment Log / Paper Trading pages rather "
                    "than re-proposing it.")
        if verdict in _FAILURE_VERDICTS:
            codes = ", ".join(evidence["result"].get("reason_codes") or []) or "no reason codes recorded"
            return (f"The registry already holds a {verdict} for this family/root ({codes}). A "
                    "genuinely new hypothesis would need a different economic mechanism or market, "
                    "not a re-run of the same one -- see Failure Memory above before proposing again.")
    for f in fm:
        if f.lessons:
            return f.lessons[0]
    return "No prior registry evidence for this family/root -- this would be a genuinely new hypothesis to evaluate."


# ---------------------------------------------------------------------------
# transcript rendering
# ---------------------------------------------------------------------------


def _render_transcript() -> None:
    """The unified Conversation Timeline (Agent Experience Consolidation
    campaign, sections 12-14): user/Claude turns and inline research-pipeline
    stages (hypothesis/failure-memory/compiled/evidence/suggestion) render in
    ONE interleaved sequence, in the order they actually happened -- there is
    no separate "AI Research Agent" surface competing with this one. Renders
    NOTHING when the transcript is empty (no giant "No conversation yet"
    card) -- a fresh Agent page should feel clean, not like an empty
    engineering module; the composer's own placeholder text already invites
    the first question."""
    turns = _transcript()
    if not turns:
        return

    renderers = {
        "user": _render_user_turn,
        "hypothesis": _render_hypothesis_card,
        "failure_memory": _render_failure_memory_card,
        "compiled": _render_compiled_card,
        "evidence": _render_evidence_cards,
        "suggestion": _render_suggestion_card,
        "error": _render_error_turn,
        "conversation": _render_conversation_turn,
        "research_workspace": _render_research_workspace_card,
    }
    for idx, turn in enumerate(turns):
        renderers[turn["type"]](turn, _event_id(turn, idx))


def _render_research_workspace_card(turn: dict, event_id: str) -> None:
    """The Research Workspace as a TRUE CONVERSATION ARTIFACT (task spec
    section 3): the composer, Research Actions, and Research Result render
    HERE, at the transcript position where the research thread actually
    started -- never a separate panel permanently pinned below the
    conversation. This turn carries no research state of its own; every
    value below is read live from the SAME session state
    (`ra_proposal`/`lab_compiled`/`agent_last_evidence`/`agent_run_outcome`/
    `agent_research_origin`) the rest of this page already reads (task spec
    section 3B). The pipeline-stage cards this composer's own Generate
    Hypothesis click produces (hypothesis/failure-memory/compiled/evidence/
    suggestion) are separate transcript turns that follow immediately after
    this one in chronological order -- this card is only the interactive
    compose/act/result shell around them, reusing
    `_render_composer`/`_render_research_actions`/`_render_execution_result`
    verbatim (task spec section 3E: never duplicate research execution
    logic)."""
    with st.expander("Research Workspace", key="agent-research-expander", expanded=True):
        _render_composer()
        _render_research_actions()
        _render_execution_result()


def _render_user_turn(turn: dict, event_id: str) -> None:
    """A right-aligned bubble, own vertical block, bounded width (item 17):
    `display:inline-block` + `max-width` keeps a long single-line message
    from stretching the full row width, and a `word-wrap` fallback keeps an
    unbroken long token from bleeding past the bubble -- never a negative
    margin or absolute/transform positioning to visually pull turns
    together."""
    st.markdown(
        f'<div style="text-align:right;margin:0.4rem 0 1.1rem 0;">'
        f'<span class="aa-tag" style="display:inline-block;max-width:72%;font-size:0.78rem;'
        f'padding:0.4rem 0.9rem;text-align:left;word-wrap:break-word;">{turn["text"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_error_turn(turn: dict, event_id: str) -> None:
    st.error(f"{turn['stage']}: {turn['text']}")


_ENGINE_LABEL = {
    "live": ("CLAUDE PROPOSAL", "OK"),
    "scripted": ("SCRIPTED PROPOSAL", "OK"),
}


def _render_engine_badge(engine: str | None) -> None:
    """Explicit execution-origin badge -- a professional user must be able to
    tell, at a glance, whether a proposal came from a real Claude call or the
    offline deterministic replay. Never collapsed into a generic label."""
    label, tone = _ENGINE_LABEL.get(engine or "scripted", ("SCRIPTED PROPOSAL", "OK"))
    st.markdown(components.badge(tone, label=label), unsafe_allow_html=True)


def _reasoning_summary(h: dict, fm_entries: list[dict]) -> list[str]:
    """A concise, user-facing rationale built ONLY from already-produced
    structured fields (the hypothesis's own `economic_mechanism` /
    `novelty_notes` / `falsification_test`, and the pre-proposal Failure
    Memory the agent was actually given) -- never a second LLM call, never
    hidden chain-of-thought. Product refactor section 3: "Reasoning Summary"."""
    lines: list[str] = []
    lessons = [lesson for fm in fm_entries for lesson in (fm.get("lessons") or [])]
    if lessons:
        lines.append(f"Prior evidence: {lessons[0]}")
    else:
        lines.append("Prior evidence: no prior registry evidence for this family/root -- a genuinely new area.")
    lines.append(f"Why it is worth testing: {h['economic_mechanism']}")
    lines.append(
        f"How it differs from prior failed work: {h['novelty_notes']}"
        if h.get("novelty_notes") else
        "How it differs from prior failed work: no novelty notes were provided by the proposal."
    )
    lines.append(f"What would falsify it: {h['falsification_test']}")
    return lines


def _render_hypothesis_card(turn: dict, event_id: str) -> None:
    h = turn["data"]
    fm_entries = turn.get("fm") or []
    # Keyed on (event_id, hypothesis_id): hypothesis_id is the RESEARCH-OBJECT
    # identity and may legitimately repeat across turns (e.g. the same
    # scripted objective sent twice) -- event_id is the UI OCCURRENCE identity
    # that keeps each turn's Streamlit element key unique. Never key on
    # hypothesis_id alone.
    with components.card(f"agent-hyp-{event_id}-{h['hypothesis_id']}"):
        st.markdown('<div class="aa-gate-title">HYPOTHESIS &middot; Research Agent</div>', unsafe_allow_html=True)
        _render_engine_badge(turn.get("engine"))
        st.markdown(f"**{h['title']}**")

        st.markdown("**Reasoning summary**")
        for line in _reasoning_summary(h, fm_entries):
            st.caption(f"- {line}")

        with st.expander("Full hypothesis detail", expanded=False):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Market:** {', '.join(h['universe'])}")
            c1.markdown(f"**Horizon:** {h['horizon']}")
            c2.markdown(f"**Expected regime:** {h['expected_regime']}")
            c2.markdown(f"**Failure regime:** {h['failure_regime']}")
            st.markdown("**Required features**")
            components.render_tags(h["required_features"])
            st.markdown("**Falsification condition**")
            st.write(h["falsification_test"])
            if h.get("novelty_notes"):
                st.markdown("**Novelty notes**")
                st.write(h["novelty_notes"])


def _render_failure_memory_card(turn: dict, event_id: str) -> None:
    """Renders PRE-PROPOSAL failure memory: one or more `FailureMemoryResponse`
    digests -- one per (strategy_family, root_symbol) combination already
    tested for the selected root -- looked up and placed into the
    `ResearchContext` BEFORE the hypothesis below was proposed, not a post-hoc
    lookup keyed off what the agent happened to pick.

    Presentation-only compaction: the primary workflow shows a short "PRIOR
    RESEARCH MEMORY" digest (neighbourhood/failure counts + top lessons); the
    FULL evidence -- every digest's verdict counts, prior-experiment table, and
    every lesson -- stays reachable in the "View full Failure Memory" expander
    below, and unabridged in Research Details -> Memory & Lineage. Nothing is
    removed from `turn["data"]` (the same `ResearchContext.failure_memory` the
    agent actually saw) -- only how much of it renders inline by default.

    Keyed on `event_id`, not `turn["at"]`: the wall-clock timestamp string has
    one-second resolution and two turns in the same pipeline run (or two fast
    scripted Sends) can share one, which is exactly the class of bug this
    checkpoint fixes."""
    entries = turn["data"]
    root = turn.get("root") or "the selected root"
    with components.card(f"agent-fm-{event_id}"):
        st.markdown('<div class="aa-gate-title">PRIOR RESEARCH MEMORY &middot; checked BEFORE proposing</div>',
                    unsafe_allow_html=True)
        if not entries:
            st.caption("No prior experiments for any approved root yet -- this proposal starts with a clean slate.")
            return

        n_neighbourhoods = len(entries)
        n_canonical_failures = sum(
            1
            for fm in entries
            for p in fm["prior_experiments"]
            if p["trial_role"] == "CANONICAL" and p["headline_verdict"] in _FAILURE_VERDICTS
        )
        lessons: list[str] = []
        for fm in entries:
            for lesson in fm["lessons"]:
                if lesson not in lessons:
                    lessons.append(lesson)

        st.markdown(f"**{root}**")
        st.caption(
            f"{n_neighbourhoods} related research neighbourhood(s) &middot; "
            f"{n_canonical_failures} prior canonical failure(s)"
        )
        if lessons:
            st.markdown("**Main lessons**")
            for lesson in lessons[:3]:
                st.caption(f"- {lesson}")

        with st.expander("View full Failure Memory", expanded=False):
            _render_full_failure_memory(entries)


def _render_full_failure_memory(entries: list[dict]) -> None:
    """The full evidence a compact PRIOR RESEARCH MEMORY digest summarises --
    every (family, root) neighbourhood's verdict counts, prior-experiment
    table, and every lesson. Unchanged content from the pre-refactor card;
    only its default visibility moved behind an expander."""
    for fm in entries:
        v = fm["verdict_counts"]
        st.markdown(f"**`{fm['query'].get('strategy_family')}` / `{fm['query'].get('root_symbol') or 'any root'}`**")
        st.write(
            f"Prior authoritative result(s): **{sum(v.values())}** " +
            (", ".join(f"{k}={n}" for k, n in v.items()) if v else "(none)")
        )
        if fm["prior_experiments"]:
            st.dataframe(
                [
                    {"Experiment": p["experiment_id"], "Role": p["trial_role"],
                     "Verdict": p["headline_verdict"], "Net PnL": p["net_pnl_usd"]}
                    for p in fm["prior_experiments"]
                ],
                width="stretch", hide_index=True, height=min(200, 44 + 35 * len(fm["prior_experiments"])),
            )
        for lesson in fm["lessons"]:
            st.caption(f"- {lesson}")


def _render_compiled_card(turn: dict, event_id: str) -> None:
    cd = turn["data"]
    supported = services.execution_supported_family(cd.get("family_key"))
    # strategy_fingerprint is research-object identity and may repeat across
    # turns (same objective -> same compiled StrategySpec); event_id keeps
    # the element key unique per occurrence.
    with components.card(f"agent-compiled-{event_id}-{cd['strategy_fingerprint'][:12]}"):
        st.markdown('<div class="aa-gate-title">COMPILATION &middot; Strategy Compiler Agent</div>',
                    unsafe_allow_html=True)
        _render_engine_badge(turn.get("engine"))
        st.markdown(components.badge("OK", label="VALID STRATEGYSPEC"), unsafe_allow_html=True)
        with components.metric_row(f"agent-compiled-{event_id}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Market", cd["root_symbol"] or "-")
            strategy_label = services.strategy_name(cd["family_key"]) if cd["family_key"] else "blueprint"
            c2.metric("Strategy / build mode", f"{strategy_label} / {cd['build_mode']}")
            c3.metric("Execution support", "SUPPORTED" if supported else "UNSUPPORTED")
        st.code(cd["strategy_fingerprint"], language="text")
        if not supported:
            st.caption(
                "UNSUPPORTED: this build mode/family is not one the current ExecutionValidationService "
                "can run -- a genuine capability gap, never a scientific outcome. Run This Hypothesis "
                "will be disabled below."
            )

        with st.expander("Full StrategySpec / feature graph / fingerprint history", expanded=False):
            spec = cd["strategy_spec"]
            features = spec.get("features") or []
            if features:
                st.markdown(f"**Feature graph** ({len(features)} feature(s))")
                components.render_tags(sorted({f.get("spec", {}).get("kind", "?") for f in features}))
            rules = spec.get("rules") or []
            if rules:
                st.caption(f"{len(rules)} rule(s) in the compiled DSL.")
            st.json(spec)
            de = cd.get("duplicate_evidence") or {}
            if de:
                _render_duplicate_evidence(de)


def _render_duplicate_evidence(de: dict) -> None:
    """STRATEGY FINGERPRINT HISTORY (this card) vs. SCIENTIFIC EXPERIMENT
    IDENTITY (`_render_execution_target`'s "Registry match", below) are kept
    visibly distinct: a frozen-candidate-manifest match is never presented as
    if it were a current scientific duplicate -- "seen before" for the live
    registry always comes from `matches` tagged
    `source="live_experiment_registry"`, and a frozen-manifest match is always
    labelled precisely as "present in the frozen candidate manifest"."""
    st.markdown("**Strategy fingerprint history** (evidence only -- never a re-execution decision)")
    if not de.get("strategy_fingerprint_seen"):
        st.caption("Fingerprint not seen in the frozen candidate manifest or the live registry.")
        return
    matches = de.get("matches") or []
    live = [m for m in matches if m.get("source") == "live_experiment_registry"]
    frozen = [m for m in matches if m.get("source") == "frozen_candidate_manifest"]
    other = [m for m in matches if m not in live and m not in frozen]
    if live:
        st.caption(
            f"Seen before in the **live Experiment Registry**: {len(live)} record(s), "
            f"{de.get('prior_valid_authoritative_results', 0)} with a VALID authoritative result."
        )
        for m in live:
            st.caption(f"- `{m.get('experiment_id') or m.get('experiment_identity', '')[:32]}`")
    if frozen:
        st.caption(
            f"Present in the **frozen Phase 13.5C candidate manifest** ({len(frozen)} entry/entries) -- "
            "this is prior-manifest evidence only, not a current scientific-experiment-identity duplicate."
        )
    for m in other:
        st.caption(f"Seen before ({m.get('source', 'unknown source')}).")
    st.caption(de.get("advisory", ""))


def _registry_match_label(evidence: dict | None) -> str:
    """NEW / EXISTING / NEAR-DUPLICATE, derived ONLY from the real scientific
    registry lookup (`services.find_registry_evidence_for_compiled`) -- never
    from strategy-fingerprint-history evidence (`duplicate_evidence`), which
    is explicitly evidence-only and never a re-execution/duplicate decision
    (see `_render_duplicate_evidence`)."""
    if not evidence or not evidence.get("result"):
        return "NEW"
    if evidence.get("match_type") == "exact_fingerprint":
        return "EXISTING"
    return "NEAR-DUPLICATE"


def _render_evidence_cards(turn: dict, event_id: str) -> None:
    """Product refactor section 4: the Execution Target -- root, family,
    strategy fingerprint, registry match -- is ALWAYS derived from the just-
    compiled StrategySpec (`turn["root"]`/`turn["family"]`/
    `turn["strategy_fingerprint"]`), never a separately re-entered value. This
    is exactly what `_execute_run_this_hypothesis` runs against, so the
    displayed target and the actual execution target can never disagree."""
    evidence = turn.get("data")
    family, root = turn.get("family"), turn.get("root")
    fingerprint = turn.get("strategy_fingerprint") or ""
    build_mode = turn.get("build_mode")

    with components.card(f"agent-exec-target-{event_id}"):
        st.markdown('<div class="aa-gate-title">EXECUTION TARGET</div>', unsafe_allow_html=True)
        with components.metric_row(f"agent-exec-target-{event_id}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Market", root or "-")
            c2.metric("Strategy", services.strategy_name(family) if family else (build_mode or "blueprint"))
            c3.metric("Strategy", (fingerprint[:14] + "...") if fingerprint else "-")
            c4.metric("Registry match", _registry_match_label(evidence))

    if not evidence or not evidence.get("result"):
        with components.card(f"agent-noeval-{event_id}"):
            st.markdown('<div class="aa-gate-title">BACKTEST RESULT &middot; VALIDATION</div>', unsafe_allow_html=True)
            st.markdown(
                components.badge("NOT_AVAILABLE", label="NOT YET EVALUATED") +
                f'<span style="margin-left:0.6rem;color:var(--aa-text-secondary);font-size:0.82rem;">'
                f"No committed registry evidence exists for `{family or '?'}` / `{root or '?'}` yet -- "
                "running this would be a genuinely new experiment, not something proposal-only Generate "
                "Hypothesis executes.</span>",
                unsafe_allow_html=True,
            )
        return

    result = evidence["result"]
    exp = evidence["experiment"]
    label = ("this exact StrategySpec" if evidence["match_type"] == "exact_fingerprint"
             else "the same family/root's canonical trial (not this exact variant)")
    # experiment_id is scientific-registry identity and may legitimately be
    # retrieved again in a later turn (same family/root re-queried); event_id
    # disambiguates the occurrence.
    with components.card(f"agent-backtest-{event_id}-{exp['experiment_id']}"):
        st.markdown('<div class="aa-gate-title">PRIOR REGISTRY EVIDENCE &middot; C++ Quant Core</div>',
                    unsafe_allow_html=True)
        st.markdown(components.badge("OK", label="REGISTRY_RETRIEVAL"), unsafe_allow_html=True)
        st.caption(
            f"No new backtest was executed. Existing authoritative result retrieved for {label}: "
            f"`{exp['experiment_id']}`."
        )
        with components.metric_row(f"agent-backtest-{event_id}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Net PnL", f"${result['net_pnl_usd']:,.0f}" if result.get("net_pnl_usd") is not None else "N/A")
            c2.metric("Sharpe", f"{result['annualized_sharpe']:.2f}" if result.get("annualized_sharpe") is not None else "N/A")
            with c3:
                components.render_badge(result["headline_verdict"] or "NOT_ADJUDICATED")
        st.caption("Full gate-by-gate validation evidence: **View Research Details** below.")

    _view_research_details_button(f"agent-view-details-{event_id}")


def _render_suggestion_card(turn: dict, event_id: str) -> None:
    with components.card(f"agent-suggest-{event_id}"):
        st.markdown('<div class="aa-gate-title">Next research suggestion</div>', unsafe_allow_html=True)
        st.caption(turn["text"])


# ---------------------------------------------------------------------------
# Research Actions -- Run This Hypothesis (product refactor, sections 5/6)
# ---------------------------------------------------------------------------


def _already_tested(evidence: dict | None) -> bool:
    """True only when the just-compiled StrategySpec's OWN fingerprint already
    carries a committed, VALID authoritative result (`_registry_match_label`'s
    "EXISTING") -- never a same-family/root canonical match for a DIFFERENT
    variant, and never an invalid-only attempt history (that is a genuinely
    new, legitimate re-execution, not "already tested")."""
    return bool(evidence and evidence.get("match_type") == "exact_fingerprint" and evidence.get("result"))


def _render_research_actions() -> None:
    compiled = st.session_state.get("lab_compiled")
    if not compiled or not compiled.get("accepted"):
        return
    cd = compiled
    supported = services.execution_supported_family(cd.get("family_key"))
    evidence = st.session_state.get("agent_last_evidence")
    already_tested = _already_tested(evidence)
    strategy_label = services.strategy_name(cd["family_key"]) if cd["family_key"] else "blueprint"

    components.section_header("Research Actions")

    if already_tested:
        # Product spec (Release UX, investor-readability pass): an EXISTING
        # exact strategy leads with its historical result, not a prominent
        # re-run button -- re-running would not change the scientific record.
        result = evidence["result"]
        with components.card("agent-already-tested"):
            st.markdown(components.badge("OK", label="ALREADY TESTED"), unsafe_allow_html=True)
            st.caption(
                "This exact StrategySpec already has a committed, authoritative result -- running it "
                "again would not change the scientific record."
            )
            with components.metric_row("agent-already-tested"):
                c1, c2 = st.columns(2)
                with c1:
                    components.verdict_metric_card(
                        "agent-already-tested-verdict", "Historical Result", result["headline_verdict"],
                    )
                with c2:
                    components.metric_card(
                        "agent-already-tested-pnl", "Net PnL",
                        f"${result['net_pnl_usd']:,.0f}" if result.get("net_pnl_usd") is not None else "N/A",
                    )
            st.caption("Full gate-by-gate evidence: **View Research Details** below.")

    with components.card("agent-research-actions"):
        st.caption(
            f"Executes the compiled StrategySpec above -- market **{cd['root_symbol'] or '-'}**, "
            f"strategy **{strategy_label}**, fingerprint `{cd['strategy_fingerprint'][:20]}...`. "
            "A real C++ backtest, frozen validation, and (only if genuinely novel) an ExperimentRegistry write."
        )
        if not supported:
            st.warning(
                "Execution support: UNSUPPORTED for this build mode/family -- a genuine capability gap "
                "(a full custom blueprint is not yet executable by this release), never a scientific "
                "outcome. Run This Hypothesis is disabled; propose a Phase 11 template hypothesis instead.",
                icon="🚧",
            )
        elif already_tested:
            st.caption(
                "Ready for Validation is not the primary next step here -- see ALREADY TESTED above. "
                "This action remains available only to reproduce the existing evidence."
            )
        else:
            st.caption("Ready for Validation -- no committed result exists yet for this exact strategy.")
        run_clicked = st.button(
            "Run This Hypothesis" if not already_tested else "Re-run anyway (reproduce evidence)",
            key="agent-run-hypothesis",
            type=("primary" if not already_tested else "secondary"),
            disabled=not supported,
        )
    if run_clicked:
        st.session_state["agent-run-confirm-pending"] = True
        st.rerun()

    if st.session_state.get("agent-run-confirm-pending"):
        _render_run_confirmation(cd)


def _render_run_confirmation(cd: dict) -> None:
    """The most reliable Streamlit pattern for a two-step explicit
    confirmation: a session-state flag gates a second, clearly-labeled block
    with Cancel/Confirm buttons (never a permanent pre-ticked checkbox). This
    is also the pattern an automated `AppTest` regression can actually drive
    (a real `st.dialog` overlay is not reliably inspectable there)."""
    with components.card("agent-run-confirm"):
        st.markdown("**Run scientific experiment?**")
        st.markdown(
            "This action may:\n"
            "- run the local C++ backtest\n"
            "- run frozen scientific validation\n"
            "- append a genuinely novel result to the ExperimentRegistry\n\n"
            "It will NOT:\n"
            "- access the 2025 holdout\n"
            "- submit broker orders\n\n"
            "Typical runtime: roughly 1-3 minutes."
        )
        c1, c2 = st.columns(2)
        with c1:
            cancel = st.button("Cancel", key="agent-run-cancel", width="stretch")
        with c2:
            confirm = st.button("Confirm Run", key="agent-run-confirm-button", type="primary", width="stretch")
    if cancel:
        st.session_state["agent-run-confirm-pending"] = False
        st.rerun()
    if confirm:
        st.session_state["agent-run-confirm-pending"] = False
        _execute_run_this_hypothesis(cd)
        st.rerun()


def _execute_run_this_hypothesis(cd: dict) -> None:
    """Reuses `ui.deep_research.run_deep_research` verbatim -- no duplicated
    orchestration logic. `root`/`family_stem` are DERIVED from the compiled
    StrategySpec (`cd`), never a separately re-entered value (product
    refactor section 4)."""
    mode_label = st.session_state.get("agent-mode", _OFFLINE_LABEL)
    mode = llm_demo.LIVE_MODE if mode_label == _CLAUDE_LABEL else llm_demo.SCRIPTED_MODE
    scenario_key = st.session_state.get("ra_scenario_key")
    objective = st.session_state.get("agent_last_objective", "")
    root = cd["root_symbol"]
    family_stem = f"agent-{cd.get('family_key') or 'blueprint'}"
    with st.spinner("Planning -> executing the real C++ backtest -> validating -> finalizing..."):
        outcome = deep_research.run_deep_research(
            mode=mode, objective=objective, root=root, family_stem=family_stem,
            scenario_key=scenario_key if mode == llm_demo.SCRIPTED_MODE else None,
        )
    st.session_state["agent_run_outcome"] = outcome
    if outcome.accepted:
        panels.log_activity("OK", "Run This Hypothesis finalized", outcome.report.status.value if outcome.report else "")
    else:
        panels.log_activity("WARN", "Run This Hypothesis did not complete", outcome.error or "")


# ---------------------------------------------------------------------------
# Research Result summary (product refactor, section 7)
# ---------------------------------------------------------------------------


def _render_execution_result() -> None:
    outcome = st.session_state.get("agent_run_outcome")
    if outcome is None:
        return

    components.section_header("Research Result")
    with components.card("agent-run-result"):
        if not outcome.accepted:
            st.markdown(components.badge("FAIL", label="EXECUTION FAILED"), unsafe_allow_html=True)
            st.error(outcome.error)
            _view_research_details_button("agent-view-details-run-error")
            return

        report = outcome.report
        if report.status.value == "INCOMPLETE_NOT_ADJUDICATED":
            # VALID execution is not the same as scientific PASS, and an
            # unresolved engineering defect is not a scientific verdict
            # either (df861c1's UI-honesty rules): this is the one case
            # that gets an error tone, not green, not a plain REJECT tone.
            st.markdown(components.badge("FAIL", label="INVALID EXECUTION"), unsafe_allow_html=True)
            st.error(
                "An unresolved engineering/execution defect left this family not adjudicated. "
                "This is NOT a scientific verdict, and nothing was written to the registry."
            )
            _view_research_details_button("agent-view-details-run-invalid")
            return

        mr = report.member_results[0] if report.member_results else None
        if mr is None:
            st.caption("No member result recorded for this run.")
            _view_research_details_button("agent-view-details-run-empty")
            return

        detail = services.get_experiment(mr.experiment_id) if mr.experiment_id else None
        result = detail["result"] if detail else None
        paper_ok = services.paper_eligible_experiment(mr.experiment_id) if mr.experiment_id else False
        verdict = mr.final_verdict.value if mr.final_verdict else "NOT_ADJUDICATED"
        reason = (
            (result.get("reason_codes") or ["none"])[0] if result
            else (mr.reason_codes[0] if mr.reason_codes else "none")
        )

        with components.metric_row("agent-run-result-top"):
            cols = st.columns(3)
            with cols[0]:
                components.metric_card("agent-run-exec", "Execution", mr.attempt_status.value)
            with cols[1]:
                components.verdict_metric_card("agent-run-verdict", "Scientific Verdict", verdict)
            with cols[2]:
                components.metric_card(
                    "agent-run-paper", "Paper Eligible", "YES" if paper_ok else "NO",
                    accent=palette.GREEN if paper_ok else palette.RED,
                )
        with components.metric_row("agent-run-result-bottom"):
            cols2 = st.columns(3)
            with cols2[0]:
                components.metric_card(
                    "agent-run-pnl", "Net PnL",
                    f"${result['net_pnl_usd']:,.0f}" if result and result.get("net_pnl_usd") is not None else "N/A",
                )
            with cols2[1]:
                components.metric_card(
                    "agent-run-sharpe", "Sharpe",
                    f"{result['annualized_sharpe']:.2f}" if result and result.get("annualized_sharpe") is not None else "N/A",
                )
            with cols2[2]:
                components.metric_card("agent-run-reason", "Key Scientific Reason", reason)
        st.caption(
            f"Experiment: `{mr.experiment_id}` &middot; attempt {mr.attempt_ordinal} of {mr.n_execution_attempts}.",
            unsafe_allow_html=True,
        )
        _view_research_details_button("agent-view-details-run-result")


# ---------------------------------------------------------------------------
# Research Details hand-off (product refactor, section 10)
# ---------------------------------------------------------------------------



def _build_research_details_target() -> dict:
    """Everything Research Details needs to render this conversation's
    current state, WITHOUT re-deriving anything -- plain session state, read
    by `research._resolve_selection` on the other side of the direct
    `st.switch_page` navigation below."""
    proposal = st.session_state.get("ra_proposal") or {}
    return {
        "source": "agent",
        "objective": st.session_state.get("agent_last_objective"),
        "root": st.session_state.get("agent_last_root"),
        "hypothesis": proposal.get("hypothesis") if proposal.get("accepted") else None,
        "compiled": st.session_state.get("lab_compiled"),
        "evidence": st.session_state.get("agent_last_evidence"),
        "run_outcome": st.session_state.get("agent_run_outcome"),
    }


def _view_research_details_button(key: str) -> None:
    """Stores the exact same hand-off `research._resolve_selection` has
    always read, then navigates there directly -- no "open Research from the
    sidebar" toast, no manual click required.

    `st.switch_page` accepts a `Page` object whose ``url_path`` matches a page
    already registered by `app.py`'s `st.navigation` -- it need not be the
    identical object `app.py` built (Streamlit hashes a page by its
    ``url_path`` alone, and its own `_validate_registered_page` explicitly
    lets a callable-sourced `Page` match another callable-sourced
    registration by that hash: see `streamlit.navigation.page`). So
    constructing `st.Page(research.render, url_path="lab")` here reaches
    the ACTUAL registered Research page -- not a lookalike copy of it --
    without `app.py` having to hand this module anything. This is public,
    documented `st.switch_page(Page)` behavior, not a URL/query-param hack."""
    if st.button("View Research Details", key=key):
        st.session_state["research_details_target"] = _build_research_details_target()
        st.switch_page(st.Page(research.render, url_path="lab"))
